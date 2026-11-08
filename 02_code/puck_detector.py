"""
Red puck detector for the forearm camera's pictures (640 x 480, BGR).

The puck is a red wooden cylinder (2.4 cm across, 2.9 cm tall) on white paper. The
detector thresholds the picture in HSV twice (hysteresis): strict "seed" pixels
mark colour that is almost certainly the puck, and a looser "grow" mask gives the
blob its full extent wherever it holds seeds, so a paler or shaded side of the puck
still belongs to it. Red hue wraps around zero in OpenCV's hue (0 to 179), so every
hue window here is written in signed hue, hue minus 180 from 90 up, with red at
zero; negative values lean towards magenta and positive values towards orange. The
puck's red leans to magenta (green is its weakest channel), while the wood, skin
and cabinets around the table lean to orange (blue is their weakest channel), so
the windows reach far to the negative side and only a little to the positive.

The camera's white balance wanders: in some episodes the white paper turns
blue-cyan, and the puck then looks pale, or violet in the arm's shadow. The
detector measures that cast on the paper and widens the windows towards violet,
and lowers the saturation limits, only as far as the cast calls for (in those
pictures the black block, the line and the blue brackets sit at signed hue -62 and
below, the shaded puck at -55 to -45).

Candidates are found on the picture shrunk to half its size, which is several times
faster and still leaves the puck more than ten pixels across. Each blob of the grow
mask that holds seed pixels is measured: colour, shape, size, and what surrounds it
(light paper, the occluders of the hand, or wood). A candidate must pass the colour
and surroundings tests, and then either the shape tests of a whole puck (solid,
round enough, of the expected size) or, when it touches the magnet block, the blue
brackets, a dark shadow, the picture's edge or the hand region at the bottom of the
picture, the looser tests of a partly hidden puck, which is sized by the smallest
circle holding its visible part. The survivors are scored and the best one is
measured again at full size: centre, area, box and lowest point. For a partly hidden
puck a circle is also fitted to the free part of its outline; when it fits well its
centre is reported as the puck's centre, otherwise the visible blob's centre is,
and the flags in the result say which. Only numpy and OpenCV are used.

Measured on the team laptop (AMD Ryzen AI 5 PRO 340, OpenCV 5.0.0, numpy 2.4.6; the
1697 pictures of episodes 037 to 045 held in memory, median of five calls each, three
runs): a kept ``PuckDetector`` takes 1.5 to 1.6 ms per picture on average and 4.0 to
4.3 ms on the slowest picture with OpenCV limited to one thread
(``cv2.setNumThreads(1)``, recommended), and 1.6 to 2.1 ms on average, 4.3 to 5.0 ms
on the slowest, with OpenCV's default twelve threads, whose start-up costs more than
it saves on pictures this small. ``detect_puck`` builds a detector per call, which
added 0 to 0.2 ms.
"""

import copy
import math
from typing import Dict, List, Optional

import cv2
import numpy

from config import PUCK_DETECTOR_PARAMETERS


class PuckDetector:

    def __init__(
            self,
            detector_parameters: Optional[dict] = None
    ):
        """
        Initialise the detector with the default parameters and any overrides.

        Parameters
        ----------
        detector_parameters : dict, optional
            Overrides for entries of ``PUCK_DETECTOR_PARAMETERS``; ``score_weights``
            may be given in part.
        """

        self.__initialize_variables()
        self.update_detector_parameters(detector_parameters)


    def __initialize_variables(self):

        self.detector_parameters = copy.deepcopy(PUCK_DETECTOR_PARAMETERS)
        self.__build_kernels()


    def update_detector_parameters(self, detector_parameters: Optional[dict]):
        """
        Merge overrides into the current parameters and rebuild the kernels.

        Parameters
        ----------
        detector_parameters : dict or None
            Entries to replace; ``score_weights`` is merged key by key.
        """

        if not detector_parameters:
            return
        for key, value in detector_parameters.items():
            if key == "score_weights":
                self.detector_parameters["score_weights"].update(value)
            else:
                self.detector_parameters[key] = copy.deepcopy(value)
        self.__build_kernels()


    def __build_kernels(self):

        parameters = self.detector_parameters
        scale = parameters["analysis_scale"]

        def ellipse_kernel(size: int) -> numpy.ndarray:
            size = max(int(size), 1)
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        self.opening_kernel = ellipse_kernel(parameters["opening_kernel_px"])
        self.closing_kernel = ellipse_kernel(parameters["closing_kernel_px"])
        self.ring_px = max(int(round(parameters["context_ring_px"] * scale)), 2)
        self.ring_kernel = ellipse_kernel(2 * self.ring_px + 1)
        self.refine_opening_kernel = ellipse_kernel(parameters["refine_opening_kernel_px"])
        self.refine_closing_kernel = ellipse_kernel(parameters["refine_closing_kernel_px"])
        self.touch_kernel = ellipse_kernel(parameters["touch_kernel_px"])


    @staticmethod
    def __signed_hue(hue: numpy.ndarray) -> numpy.ndarray:
        """Return OpenCV hue (0 to 179) as signed hue, red at zero (-90 to 89)."""

        hue = hue.astype(numpy.int16)
        return numpy.where(hue >= 90, hue - 180, hue)


    @staticmethod
    def __hue_window_mask(
            picture_hsv: numpy.ndarray,
            signed_hue_range: List[int],
            saturation_minimum: int,
            value_minimum: int
    ) -> numpy.ndarray:
        """
        Mask the pixels whose signed hue lies in a window and that are saturated and bright enough.

        Parameters
        ----------
        picture_hsv : numpy.ndarray
            The picture in OpenCV's HSV (hue 0 to 179).
        signed_hue_range : list of int
            ``[minimum, maximum]`` in signed hue, -90 to 89; the window may cross zero.
        saturation_minimum, value_minimum : int
            Lower limits on saturation and value.

        Returns
        -------
        numpy.ndarray
            ``uint8`` mask, 255 where the pixel passes.
        """

        hue_minimum, hue_maximum = signed_hue_range
        lower_limits = []
        upper_limits = []
        if hue_minimum < 0:
            # the negative part of the window is the top of OpenCV's hue scale
            lower_limits.append((180 + hue_minimum, saturation_minimum, value_minimum))
            upper_limits.append((179 if hue_maximum >= 0 else 180 + hue_maximum, 255, 255))
        if hue_maximum >= 0:
            lower_limits.append((max(hue_minimum, 0), saturation_minimum, value_minimum))
            upper_limits.append((hue_maximum, 255, 255))
        mask = cv2.inRange(picture_hsv, lower_limits[0], upper_limits[0])
        if len(lower_limits) == 2:
            mask = cv2.bitwise_or(mask, cv2.inRange(picture_hsv, lower_limits[1], upper_limits[1]))
        return mask


    def __colour_masks(
            self,
            picture_hsv: numpy.ndarray,
            cast_share: float,
            opening_kernel: numpy.ndarray,
            closing_kernel: numpy.ndarray
    ) -> tuple:
        """
        Return the seed mask and the cleaned grow mask of a picture or crop.

        Parameters
        ----------
        picture_hsv : numpy.ndarray
            Picture or crop in OpenCV's HSV.
        cast_share : float
            The picture's blue cast along the adaptation range, 0 to 1.
        opening_kernel, closing_kernel : numpy.ndarray
            Structuring elements for the grow mask.

        Returns
        -------
        tuple of numpy.ndarray
            ``(seed_mask, grow_mask)``, both ``uint8`` with 255 for passing pixels.
        """

        parameters = self.detector_parameters
        seed_extension = int(round(cast_share * parameters["seed_violet_extension_maximum"]))
        grow_extension = int(round(cast_share * parameters["grow_violet_extension_maximum"]))
        seed_hue_minimum, seed_hue_maximum = parameters["seed_hue_range"]
        grow_hue_minimum, grow_hue_maximum = parameters["grow_hue_range"]
        seed_mask = self.__hue_window_mask(
            picture_hsv = picture_hsv,
            signed_hue_range = [seed_hue_minimum - seed_extension, seed_hue_maximum],
            saturation_minimum = int(round(parameters["seed_saturation_minimum"]
                                           - cast_share * parameters["seed_saturation_cast_reduction"])),
            value_minimum = parameters["seed_value_minimum"]
        )
        grow_mask = self.__hue_window_mask(
            picture_hsv = picture_hsv,
            signed_hue_range = [grow_hue_minimum - grow_extension, grow_hue_maximum],
            saturation_minimum = int(round(parameters["grow_saturation_minimum"]
                                           - cast_share * parameters["grow_saturation_cast_reduction"])),
            value_minimum = parameters["grow_value_minimum"]
        )
        # every seed pixel also belongs to the grow mask, whatever the two windows are
        grow_mask = cv2.bitwise_or(grow_mask, seed_mask)
        grow_mask = cv2.morphologyEx(grow_mask, cv2.MORPH_OPEN, opening_kernel)
        grow_mask = cv2.morphologyEx(grow_mask, cv2.MORPH_CLOSE, closing_kernel)
        return seed_mask, grow_mask


    def measure_blue_cast(self, picture_hsv: numpy.ndarray, picture_bgr: numpy.ndarray) -> Optional[float]:
        """
        Measure the picture's colour cast as blue over red on the white paper.

        The paper is taken as the brightest pixels with little colour on a sparse
        grid, which is plenty for an average and costs little. Brightness is ranked
        rather than thresholded because wood can look grey in a blue-cast picture,
        though it is darker than the paper.

        Parameters
        ----------
        picture_hsv : numpy.ndarray
            The picture (or a scaled copy) in OpenCV's HSV.
        picture_bgr : numpy.ndarray
            The same picture in BGR.

        Returns
        -------
        float or None
            Mean blue over mean red of the paper pixels (about 1.03 in a neutral
            picture, about 1.25 in the blue-cast episodes), or ``None`` when too
            little paper shows.
        """

        parameters = self.detector_parameters
        step = max(int(round(parameters["paper_sample_step_px"] * parameters["analysis_scale"])), 1)
        sample_hsv = picture_hsv[::step, ::step].reshape(-1, 3)
        sample_bgr = picture_bgr[::step, ::step].reshape(-1, 3)
        value = sample_hsv[:, 2]
        rank = int((1.0 - parameters["paper_brightest_share"]) * (len(value) - 1))
        brightness_threshold = max(int(numpy.partition(value, rank)[rank]), parameters["paper_value_minimum"])
        is_paper = (value >= brightness_threshold) & (sample_hsv[:, 1] <= parameters["paper_saturation_maximum"])
        if numpy.count_nonzero(is_paper) < parameters["paper_share_minimum"] * len(value):
            return None
        paper_pixels = sample_bgr[is_paper].astype(numpy.float32)
        return float(paper_pixels[:, 0].mean()) / max(float(paper_pixels[:, 2].mean()), 1.0)


    def __cast_share(self, blue_cast: Optional[float]) -> float:
        """Return how far the picture's blue cast lies along the adaptation range, 0 (neutral) to 1."""

        if blue_cast is None:
            return 0.0
        cast_start, cast_full = self.detector_parameters["violet_extension_cast_range"]
        return min(max((blue_cast - cast_start) / (cast_full - cast_start), 0.0), 1.0)


    def __occluder_mask(
            self,
            crop_hsv: numpy.ndarray,
            cast_share: float,
            crop_top: int,
            pixel_scale: float
    ) -> numpy.ndarray:
        """
        Return a boolean mask of the dark block (and dark shadow) and the blue brackets in a crop.

        Outside the hand region the blue saturation floor rises with the cast, because
        shaded paper turns as blue as the brackets there; inside it the floor stays at
        its neutral value, so that the block, slate blue in a blue-cast picture, still
        counts as an occluder.

        Parameters
        ----------
        crop_hsv : numpy.ndarray
            The crop in OpenCV's HSV.
        cast_share : float
            The picture's blue cast along the adaptation range, 0 to 1.
        crop_top : int
            Row of the crop's top in the (possibly scaled) picture.
        pixel_scale : float
            Scale of the picture the crop comes from (the analysis scale, or 1).

        Returns
        -------
        numpy.ndarray
            Boolean mask, True where an occluder is.
        """

        parameters = self.detector_parameters
        hue = crop_hsv[:, :, 0]
        saturation = crop_hsv[:, :, 1]
        value = crop_hsv[:, :, 2]
        is_dark = value <= parameters["occluder_value_maximum"]
        blue_minimum, blue_maximum = parameters["occluder_blue_hue_range"]
        neutral_floor, cast_floor = parameters["occluder_blue_saturation_range"]
        saturation_floor = numpy.full((crop_hsv.shape[0], 1), neutral_floor + cast_share * (cast_floor - neutral_floor))
        hand_region_first_row = int(math.ceil(parameters["hand_region_top_v"] * pixel_scale)) - crop_top
        saturation_floor[max(hand_region_first_row, 0):] = neutral_floor
        is_blue = ((hue >= blue_minimum) & (hue <= blue_maximum) & (saturation >= saturation_floor)
                   & (value >= parameters["occluder_blue_value_minimum"]))
        return is_dark | is_blue


    def __describe_candidate(
            self,
            label_image: numpy.ndarray,
            label: int,
            statistics: numpy.ndarray,
            seed_mask: numpy.ndarray,
            picture_hsv: numpy.ndarray,
            cast_share: float
    ) -> Optional[Dict]:
        """
        Measure one grow-mask blob at the analysis scale: colour, shape, occlusion and surroundings.

        Parameters
        ----------
        label_image : numpy.ndarray
            Connected-component labels of the grow mask.
        label : int
            The blob's label.
        statistics : numpy.ndarray
            Row of ``cv2.connectedComponentsWithStats`` for the blob.
        seed_mask : numpy.ndarray
            Seed mask at the analysis scale.
        picture_hsv : numpy.ndarray
            The scaled picture in OpenCV's HSV.
        cast_share : float
            The picture's blue cast along the adaptation range, 0 to 1.

        Returns
        -------
        dict or None
            Measurements with lengths and areas converted to full-size pixels, or
            ``None`` if the blob holds no seed pixel or no contour could be traced.
        """

        parameters = self.detector_parameters
        scale = parameters["analysis_scale"]
        picture_height, picture_width = label_image.shape
        left = int(statistics[cv2.CC_STAT_LEFT])
        top = int(statistics[cv2.CC_STAT_TOP])
        width = int(statistics[cv2.CC_STAT_WIDTH])
        height = int(statistics[cv2.CC_STAT_HEIGHT])
        area = int(statistics[cv2.CC_STAT_AREA])

        # work in a crop around the blob, wide enough to hold the context ring
        margin = self.ring_px + 2
        crop_left = max(left - margin, 0)
        crop_top = max(top - margin, 0)
        crop_right = min(left + width + margin, picture_width)
        crop_bottom = min(top + height + margin, picture_height)
        is_blob = label_image[crop_top:crop_bottom, crop_left:crop_right] == label
        seed_count = int(numpy.count_nonzero(seed_mask[crop_top:crop_bottom, crop_left:crop_right][is_blob]))
        if seed_count == 0:
            return None
        blob = is_blob.astype(numpy.uint8)
        crop_hsv = picture_hsv[crop_top:crop_bottom, crop_left:crop_right]

        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        contour = max(contours, key = cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_mask = numpy.zeros_like(blob)
        cv2.fillConvexPoly(hull_mask, hull, 1)
        hull_area = max(int(numpy.count_nonzero(hull_mask)), 1)
        occluder = self.__occluder_mask(crop_hsv, cast_share, crop_top, scale)
        hidden_in_hull = int(numpy.count_nonzero(hull_mask.astype(bool) & ~is_blob & occluder))
        solidity = min(area / hull_area, 1.0)
        completed_solidity = min((area + hidden_in_hull) / hull_area, 1.0)
        completed_extent = min((area + hidden_in_hull) / float(width * height), 1.0)
        if len(contour) >= 5:
            (_, _), (axis_one, axis_two), _ = cv2.fitEllipse(contour)
            axis_ratio = max(axis_one, axis_two) / max(min(axis_one, axis_two), 1.0)
        else:
            axis_ratio = max(width, height) / max(min(width, height), 1.0)
        (_, _), enclosing_radius = cv2.minEnclosingCircle(contour)

        rows, columns = numpy.nonzero(blob)
        centre_u = float(columns.mean()) + crop_left
        centre_v = float(rows.mean()) + crop_top

        # surroundings: the ring around the blob, split into the part above and below its centre
        ring = cv2.dilate(blob, self.ring_kernel).astype(bool) & ~is_blob
        ring_rows = numpy.nonzero(ring)[0] + crop_top
        ring_hsv = crop_hsv[ring]
        ring_count = max(len(ring_hsv), 1)
        ring_hue = self.__signed_hue(ring_hsv[:, 0])
        ring_saturation = ring_hsv[:, 1]
        ring_value = ring_hsv[:, 2]
        ring_occluder = occluder[ring]
        is_light = ((ring_value >= parameters["light_value_minimum"])
                    & (ring_saturation <= parameters["light_saturation_maximum"]) & ~ring_occluder)
        is_wood = ((ring_hue >= parameters["wood_hue_range"][0]) & (ring_hue <= parameters["wood_hue_range"][1])
                   & (ring_saturation >= parameters["wood_saturation_minimum"])
                   & (ring_value >= parameters["wood_value_minimum"]))
        is_above = ring_rows < centre_v
        is_below = ~is_above
        light_above_fraction = float(numpy.count_nonzero(is_light & is_above)) / max(
            int(numpy.count_nonzero(is_above)), 1)
        occluder_below_fraction = float(numpy.count_nonzero(ring_occluder & is_below)) / max(
            int(numpy.count_nonzero(is_below)), 1)
        occluder_fraction = float(numpy.count_nonzero(ring_occluder)) / ring_count
        is_possibly_occluded = (occluder_below_fraction >= parameters["occluded_below_fraction_minimum"]
                                and light_above_fraction >= parameters["occluded_light_above_fraction_minimum"])
        is_touching_picture_edge = (left <= 0 or top <= 0 or left + width >= picture_width
                                    or top + height >= picture_height)

        blob_hsv = crop_hsv[is_blob]
        return {
            "label": label,
            "centre_u": (centre_u + 0.5) / scale - 0.5,
            "centre_v": (centre_v + 0.5) / scale - 0.5,
            "area_px": int(round(area / (scale * scale))),
            "equivalent_radius_px": math.sqrt(area / math.pi) / scale,
            "bounding_box": [int(round(left / scale)), int(round(top / scale)), int(round(width / scale)),
                             int(round(height / scale))],
            "seed_fraction": seed_count / float(area),
            "solidity": solidity,
            "completed_solidity": completed_solidity,
            "completed_extent": completed_extent,
            "axis_ratio": axis_ratio,
            "light_fraction": float(numpy.count_nonzero(is_light)) / ring_count,
            "occluder_fraction": occluder_fraction,
            "wood_fraction": float(numpy.count_nonzero(is_wood)) / ring_count,
            "light_above_fraction": light_above_fraction,
            "occluder_below_fraction": occluder_below_fraction,
            "median_signed_hue": float(numpy.median(self.__signed_hue(blob_hsv[:, 0]))),
            "median_saturation": float(numpy.median(blob_hsv[:, 1])),
            "median_value": float(numpy.median(blob_hsv[:, 2])),
            "size_ratio": math.sqrt(area / math.pi) / scale / parameters["expected_radius_px"],
            "enclosing_size_ratio": (enclosing_radius + 0.5) / scale / parameters["expected_radius_px"],
            "bottom_v_analysis": (top + height - 0.5) / scale,
            "cast_share": cast_share,
            "is_touching_picture_edge": bool(is_touching_picture_edge),
            "is_possibly_occluded": bool(is_possibly_occluded),
            "is_beside_occluder": bool(occluder_fraction >= parameters["adjacent_occluder_fraction_minimum"])
        }


    def __distance_from_expected(self, candidate: Dict) -> float:
        """Return the candidate's distance in pixels from where the puck usually sits."""

        expected_u, expected_v = self.detector_parameters["expected_centre_uv"]
        return math.hypot(candidate["centre_u"] - expected_u, candidate["centre_v"] - expected_v)


    def __colour_and_context_rejection(self, candidate: Dict) -> Optional[str]:
        """
        Return why a candidate fails the colour and surroundings tests, or ``None``.

        Parameters
        ----------
        candidate : dict
            Measurements from ``__describe_candidate``.

        Returns
        -------
        str or None
            A short reason such as ``"pale"``.
        """

        parameters = self.detector_parameters
        if candidate["area_px"] < parameters["minimum_area_px"]:
            return "area_small"
        if candidate["area_px"] > parameters["maximum_area_px"]:
            return "area_large"
        if candidate["seed_fraction"] < parameters["minimum_seed_fraction"]:
            return "seed_fraction"
        pale_limit = parameters["minimum_median_saturation"] - candidate["cast_share"] * parameters[
            "pale_cast_reduction"]
        if candidate["median_saturation"] < pale_limit:
            return "pale"
        # 8.11.26: under the ceiling lights the puck in the hand's shadow is as dark as the block and the gaps, but
        # stays nearer red in hue than they do, so a dark violet blob that near red is kept
        if (candidate["median_signed_hue"] < parameters["grow_hue_range"][0]
                and candidate["median_value"] < parameters["violet_minimum_median_value"]
                and candidate["median_signed_hue"] < parameters["dark_violet_puck_hue_minimum"]):
            return "dark_violet"
        if candidate["wood_fraction"] > parameters["maximum_wood_fraction"]:
            return "wood_around"
        if candidate["light_fraction"] < parameters["minimum_light_fraction"]:
            # a puck deep in the arm's shadow has little light paper around it, but it lies near the
            # usual position and is puck-sized (by its area, or by its enclosing circle when an
            # occluder hides part of it); the mouse wheel and the cable ends do not
            size_ratio = max(candidate["size_ratio"],
                             candidate["enclosing_size_ratio"] if candidate["is_beside_occluder"] else 0.0)
            is_puck_like = (self.__distance_from_expected(candidate)
                            <= parameters["dark_surroundings_maximum_distance_px"]
                            and size_ratio >= parameters["dark_surroundings_minimum_size_ratio"])
            if not is_puck_like:
                return "no_light_around"
        return None


    def __shape_rejection(
            self,
            candidate: Dict,
            is_partial: bool
    ) -> Optional[str]:
        """
        Return why a candidate fails the shape tests of a whole or a partly hidden puck, or ``None``.

        A partly hidden puck is allowed a longer shape and is sized by the smallest
        circle that holds its visible part instead of by its area.

        Parameters
        ----------
        candidate : dict
            Measurements from ``__describe_candidate``.
        is_partial : bool
            Whether to apply the tests of a partly hidden puck.

        Returns
        -------
        str or None
            A short reason such as ``"solidity"``.
        """

        parameters = self.detector_parameters
        size_minimum, size_maximum = parameters["size_ratio_range"]
        suffix = "_partial" if is_partial else ""
        size_ratio = candidate["enclosing_size_ratio" if is_partial else "size_ratio"]
        if (is_partial and not candidate["is_touching_picture_edge"]
                and candidate["bottom_v_analysis"] < parameters["hand_region_top_v"]):
            return "high"
        if candidate["completed_solidity"] < parameters["minimum_completed_solidity" + suffix]:
            return "solidity"
        if candidate["axis_ratio"] > parameters["maximum_axis_ratio" + suffix]:
            return "axis_ratio"
        if candidate["completed_extent"] < parameters["minimum_completed_extent" + suffix]:
            return "extent"
        if not size_minimum <= size_ratio <= size_maximum:
            return "size"
        return None


    def __score_candidate(self, candidate: Dict) -> float:
        """
        Score a candidate that passed its tests, from 0 (poor) to 1 (ideal).

        Parameters
        ----------
        candidate : dict
            Measurements from ``__describe_candidate``, with ``is_partial``.

        Returns
        -------
        float
            Weighted sum of the partial scores, lowered for a blob at the picture's edge
            and for a partly hidden puck.
        """

        parameters = self.detector_parameters
        weights = parameters["score_weights"]
        # colour: a puck in plain view is mostly seed colour (full marks from "full_colour_seed_fraction")
        # and strongly saturated (full marks from "full_colour_saturation_span" above the pale limit)
        seed_score = min(candidate["seed_fraction"] / parameters["full_colour_seed_fraction"], 1.0)
        pale_limit = parameters["minimum_median_saturation"] - candidate["cast_share"] * parameters[
            "pale_cast_reduction"]
        saturation_score = min(max((candidate["median_saturation"] - pale_limit) / parameters["full_colour_saturation_span"], 0.0), 1.0)
        colour_score = 0.5 * seed_score + 0.5 * saturation_score
        size_minimum, size_maximum = parameters["size_ratio_range"]
        # shape: completed solidity from its limit (0) to 1 (1); an axis ratio up to "full_shape_axis_ratio"
        # (a puck seen from the side) scores 1, falling to 0 at the limit
        suffix = "_partial" if candidate["is_partial"] else ""
        maximum_axis_ratio = parameters["maximum_axis_ratio" + suffix]
        minimum_solidity = parameters["minimum_completed_solidity" + suffix]
        solidity_score = (candidate["completed_solidity"] - minimum_solidity) / (1.0 - minimum_solidity)
        full_axis_ratio = parameters["full_shape_axis_ratio"]
        axis_score = 1.0 - max(candidate["axis_ratio"] - full_axis_ratio, 0.0) / (maximum_axis_ratio - full_axis_ratio)
        shape_score = 0.5 * max(solidity_score, 0.0) + 0.5 * max(axis_score, 0.0)
        size_ratio = candidate["enclosing_size_ratio" if candidate["is_partial"] else "size_ratio"]
        # surroundings: light background or an occluder around the blob, no wood
        surroundings_score = min(candidate["light_fraction"] + parameters["occluder_surroundings_weight"] * candidate["occluder_fraction"], 1.0) * (
            1.0 - candidate["wood_fraction"])
        # size: 1 at the expected radius, 0 at the limits of the accepted range
        size_logarithm = math.log(max(size_ratio, 1e-3))
        size_limit = math.log(size_maximum) if size_logarithm > 0 else -math.log(size_minimum)
        size_score = max(1.0 - abs(size_logarithm) / size_limit, 0.0)
        position_score = math.exp(-0.5 * (self.__distance_from_expected(candidate)
                                          / parameters["expected_centre_sigma_px"]) ** 2)

        score = (weights["colour"] * colour_score + weights["shape"] * shape_score
                 + weights["surroundings"] * surroundings_score + weights["size"] * size_score
                 + weights["position"] * position_score)
        if candidate["is_touching_picture_edge"]:
            score *= parameters["picture_edge_score_factor"]
        if candidate["is_partial"]:
            score *= parameters["partial_score_factor"]
        return score


    def __full_size_blob(
            self,
            picture_bgr: numpy.ndarray,
            bounding_box: List[int],
            margin: int,
            cast_share: float
    ) -> Optional[Dict]:
        """
        Rebuild the grow mask at full size in a crop around a box and return the blob
        that overlaps the box most, with the crop's HSV and occluder mask.

        Parameters
        ----------
        picture_bgr : numpy.ndarray
            The full-size picture in BGR.
        bounding_box : list of int
            ``[left, top, width, height]`` in full-size pixels.
        margin : int
            Pixels added around the box on every side.
        cast_share : float
            The picture's blue cast along the adaptation range, 0 to 1.

        Returns
        -------
        dict or None
            ``blob`` (``uint8`` mask of the crop), ``crop_left``, ``crop_top``,
            ``crop_hsv``, ``occluder`` and ``area_px``, or ``None`` if no blob
            overlaps the box.
        """

        picture_height, picture_width = picture_bgr.shape[:2]
        left, top, width, height = bounding_box
        crop_left = max(left - margin, 0)
        crop_top = max(top - margin, 0)
        crop_right = min(left + width + margin, picture_width)
        crop_bottom = min(top + height + margin, picture_height)
        crop_hsv = cv2.cvtColor(picture_bgr[crop_top:crop_bottom, crop_left:crop_right], cv2.COLOR_BGR2HSV)
        _, grow_mask = self.__colour_masks(
            picture_hsv = crop_hsv,
            cast_share = cast_share,
            opening_kernel = self.refine_opening_kernel,
            closing_kernel = self.refine_closing_kernel
        )
        label_count, label_image, statistics, _ = cv2.connectedComponentsWithStats(grow_mask, connectivity = 8)
        if label_count <= 1:
            return None

        # the blob with the most pixels inside the box is the same blob at full size
        inside_box = numpy.zeros(grow_mask.shape, dtype = bool)
        inside_box[max(top - crop_top, 0):top - crop_top + height, max(left - crop_left, 0):left - crop_left + width] = True
        overlaps = numpy.bincount(label_image[inside_box], minlength = label_count)
        overlaps[0] = 0
        best_label = int(numpy.argmax(overlaps))
        if overlaps[best_label] == 0:
            return None
        return {
            "blob": (label_image == best_label).astype(numpy.uint8),
            "crop_left": crop_left,
            "crop_top": crop_top,
            "crop_hsv": crop_hsv,
            "occluder": self.__occluder_mask(crop_hsv, cast_share, crop_top, 1.0),
            "area_px": int(statistics[best_label, cv2.CC_STAT_AREA]),
            "box": [int(statistics[best_label, cv2.CC_STAT_LEFT]) + crop_left,
                    int(statistics[best_label, cv2.CC_STAT_TOP]) + crop_top,
                    int(statistics[best_label, cv2.CC_STAT_WIDTH]),
                    int(statistics[best_label, cv2.CC_STAT_HEIGHT])]
        }


    @staticmethod
    def __fit_circle(points: numpy.ndarray) -> Optional[List[float]]:
        """
        Fit a circle to points by least squares on the distance to the circle.

        An algebraic fit (linear in the unknowns) gives the start, and a few
        Gauss-Newton steps then minimise the geometric distance, since an algebraic
        fit alone underestimates the radius on a short arc.

        Parameters
        ----------
        points : numpy.ndarray
            ``(N, 2)`` array of ``(u, v)`` points.

        Returns
        -------
        list of float or None
            ``[centre_u, centre_v, radius, rms_residual]``, or ``None`` if the fit
            is degenerate.
        """

        u = points[:, 0]
        v = points[:, 1]
        design = numpy.column_stack([u, v, numpy.ones(len(points))])
        target = -(u * u + v * v)
        (coefficient_a, coefficient_b, coefficient_c), _, _, _ = numpy.linalg.lstsq(design, target, rcond = None)
        centre_u = -coefficient_a / 2.0
        centre_v = -coefficient_b / 2.0
        radius_squared = centre_u * centre_u + centre_v * centre_v - coefficient_c
        if radius_squared <= 0:
            return None
        radius = math.sqrt(radius_squared)

        for _ in range(6):
            difference_u = u - centre_u
            difference_v = v - centre_v
            distance = numpy.maximum(numpy.hypot(difference_u, difference_v), 1e-9)
            residual = distance - radius
            jacobian = numpy.column_stack([-difference_u / distance, -difference_v / distance,
                                           -numpy.ones(len(points))])
            step, _, _, _ = numpy.linalg.lstsq(jacobian, -residual, rcond = None)
            centre_u += step[0]
            centre_v += step[1]
            radius += step[2]
            if abs(step).max() < 1e-3:
                break
        if radius <= 0:
            return None
        residual = numpy.hypot(u - centre_u, v - centre_v) - radius
        return [float(centre_u), float(centre_v), float(radius), float(math.sqrt(numpy.mean(residual * residual)))]


    def __fit_circle_to_free_outline(
            self,
            blob: numpy.ndarray,
            occluder: numpy.ndarray,
            crop_origin: tuple,
            picture_shape: tuple
    ) -> Dict:
        """
        Fit a circle to the part of a blob's outline that touches neither an occluder nor the picture's edge.

        The fit is repeated on the points that lie close to the previous circle, so that
        stretches of outline against a shadow or the hand's metal, which are not the puck's
        own edge, drop out.

        Parameters
        ----------
        blob : numpy.ndarray
            ``uint8`` mask of the blob in a full-size crop.
        occluder : numpy.ndarray
            Boolean occluder mask of the same crop.
        crop_origin : tuple
            ``(left, top)`` of the crop in the picture.
        picture_shape : tuple
            ``(height, width)`` of the picture.

        Returns
        -------
        dict
            ``is_circle_good`` and ``circle_reason`` (``None`` when good), and when a
            circle was fitted: ``circle_centre_u``, ``circle_centre_v``,
            ``circle_radius_px``, ``circle_residual_fraction``, ``circle_inlier_share``,
            ``circle_arc_deg``, ``circle_inside_fraction`` and ``visible_fraction``.
        """

        parameters = self.detector_parameters
        crop_left, crop_top = crop_origin
        picture_height, picture_width = picture_shape
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {"is_circle_good": False, "circle_reason": "no_outline"}
        points = max(contours, key = cv2.contourArea).reshape(-1, 2)
        near_occluder = cv2.dilate(occluder.astype(numpy.uint8), self.touch_kernel).astype(bool)
        columns = points[:, 0]
        rows = points[:, 1]
        is_free = ~near_occluder[rows, columns]
        is_free &= (columns + crop_left > 1) & (columns + crop_left < picture_width - 2)
        is_free &= (rows + crop_top > 1) & (rows + crop_top < picture_height - 2)
        free_points = points[is_free].astype(numpy.float64)
        if len(free_points) < parameters["circle_minimum_free_points"]:
            return {"is_circle_good": False, "circle_reason": "short_outline"}

        circle = self.__fit_circle(free_points)
        if circle is None:
            return {"is_circle_good": False, "circle_reason": "degenerate"}
        is_inlier = numpy.ones(len(free_points), dtype = bool)
        for _ in range(parameters["circle_refit_rounds"]):
            centre_u, centre_v, radius, _ = circle
            residual = numpy.abs(numpy.hypot(free_points[:, 0] - centre_u, free_points[:, 1] - centre_v) - radius)
            threshold = max(parameters["circle_inlier_px"], parameters["circle_inlier_fraction_of_radius"] * radius)
            is_inlier = residual <= threshold
            if numpy.count_nonzero(is_inlier) < parameters["circle_minimum_free_points"]:
                break
            refitted = self.__fit_circle(free_points[is_inlier])
            if refitted is None:
                break
            circle = refitted
        centre_u, centre_v, radius, residual_rms = circle
        inlier_points = free_points[is_inlier] if numpy.count_nonzero(is_inlier) else free_points

        # angular span of the inliers around the centre: the largest gap between neighbouring
        # angles is the part of the circle that the fitted outline leaves open
        angles = numpy.sort(numpy.arctan2(inlier_points[:, 1] - centre_v, inlier_points[:, 0] - centre_u))
        gaps = numpy.diff(numpy.concatenate([angles, angles[:1] + 2 * math.pi]))
        rows_grid, columns_grid = numpy.indices(blob.shape)
        distance = numpy.hypot(columns_grid - centre_u, rows_grid - centre_v)
        is_blob = blob.astype(bool)
        blob_count = max(int(numpy.count_nonzero(is_blob)), 1)
        result = {
            "circle_centre_u": centre_u + crop_left,
            "circle_centre_v": centre_v + crop_top,
            "circle_radius_px": radius,
            "circle_residual_fraction": residual_rms / max(radius, 1e-6),
            "circle_inlier_share": float(numpy.count_nonzero(is_inlier)) / len(free_points),
            "circle_arc_deg": math.degrees(2 * math.pi - float(gaps.max())),
            "circle_inside_fraction": float(numpy.count_nonzero(is_blob & (distance <= radius + parameters["circle_inside_tolerance_px"]))) / blob_count,
            "visible_fraction": min(float(numpy.count_nonzero(is_blob & (distance <= radius)))
                                    / max(math.pi * radius * radius, 1.0), 1.0)
        }

        minimum_radius, maximum_radius = parameters["circle_radius_range_px"]
        reason = None
        if not minimum_radius <= radius <= maximum_radius:
            reason = "radius"
        elif result["circle_residual_fraction"] > parameters["circle_maximum_residual_fraction"]:
            reason = "residual"
        elif result["circle_inlier_share"] < parameters["circle_minimum_inlier_share"]:
            reason = "inliers"
        elif result["circle_arc_deg"] < parameters["circle_minimum_arc_deg"]:
            reason = "arc"
        elif result["circle_inside_fraction"] < parameters["circle_minimum_inside_fraction"]:
            reason = "outside"
        result["is_circle_good"] = reason is None
        result["circle_reason"] = reason
        return result


    def __measure_at_full_size(
            self,
            picture_bgr: numpy.ndarray,
            candidate: Dict,
            cast_share: float
    ) -> Dict:
        """
        Measure the chosen blob again at full size: visible centre, area, box, bottom point
        and, for a blob accepted as partly hidden, the circle fitted to its free outline.

        Parameters
        ----------
        picture_bgr : numpy.ndarray
            The full-size picture in BGR.
        candidate : dict
            The chosen candidate.
        cast_share : float
            The picture's blue cast along the adaptation range, 0 to 1.

        Returns
        -------
        dict
            Full-size geometry; empty if no blob was found in the crop, in which
            case the scaled-up analysis values stand.
        """

        parameters = self.detector_parameters
        picture_shape = picture_bgr.shape[:2]
        full_size = self.__full_size_blob(
            picture_bgr = picture_bgr,
            bounding_box = candidate["bounding_box"],
            margin = parameters["circle_margin_px" if candidate["is_partial"] else "refine_margin_px"],
            cast_share = cast_share
        )
        if full_size is None:
            return {}
        blob = full_size["blob"]
        crop_left = full_size["crop_left"]
        crop_top = full_size["crop_top"]
        rows, columns = numpy.nonzero(blob)
        bottom_row = int(rows.max())
        bottom_columns = columns[rows == bottom_row]
        bottom_u = float(bottom_columns.mean())

        # the lowest point is the puck's own outline only when light paper lies just below it; anything
        # dark or blue there (the block, a bracket, but also the black line or a dark shadow) or the
        # picture's edge leaves open whether part of the puck is hidden below it
        search = parameters["bottom_occluder_search_px"]
        below = full_size["occluder"][bottom_row + 1:bottom_row + 1 + search,
                                      int(bottom_columns.min()):int(bottom_columns.max()) + 1]
        is_bottom_blocked = bool(below.size and below.any()) or bottom_row + crop_top >= picture_shape[0] - 2
        area = full_size["area_px"]
        measured = {
            "visible_centre_u": float(columns.mean()) + crop_left,
            "visible_centre_v": float(rows.mean()) + crop_top,
            "area_px": area,
            "equivalent_radius_px": math.sqrt(area / math.pi),
            "bounding_box": full_size["box"],
            "bottom_u": bottom_u + crop_left,
            "bottom_v": float(bottom_row + crop_top),
            "is_bottom_clear": not is_bottom_blocked
        }
        if candidate["is_partial"]:
            measured.update(self.__fit_circle_to_free_outline(
                blob = blob,
                occluder = full_size["occluder"],
                crop_origin = (crop_left, crop_top),
                picture_shape = picture_shape
            ))
        return measured


    def find_candidates(self, picture_bgr: numpy.ndarray) -> tuple:
        """
        Return every candidate blob with its measurements, score and rejection reason.

        Meant for tuning and inspection; ``detect`` uses it and keeps the best.

        Parameters
        ----------
        picture_bgr : numpy.ndarray
            The camera picture in OpenCV's BGR order.

        Returns
        -------
        tuple
            ``(candidates, blue_cast, cast_share)``: the candidates sorted by score,
            highest first (rejected ones carry a ``rejection_reason`` and a score of
            0), the measured cast and its share of the adaptation range.
        """

        parameters = self.detector_parameters
        scale = parameters["analysis_scale"]
        if scale != 1.0:
            analysis_bgr = cv2.resize(picture_bgr, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA)
        else:
            analysis_bgr = picture_bgr
        analysis_hsv = cv2.cvtColor(analysis_bgr, cv2.COLOR_BGR2HSV)
        blue_cast = self.measure_blue_cast(analysis_hsv, analysis_bgr)
        cast_share = self.__cast_share(blue_cast)
        seed_mask, grow_mask = self.__colour_masks(
            picture_hsv = analysis_hsv,
            cast_share = cast_share,
            opening_kernel = self.opening_kernel,
            closing_kernel = self.closing_kernel
        )

        label_count, label_image, statistics, _ = cv2.connectedComponentsWithStats(grow_mask, connectivity = 8)
        minimum_area = parameters["minimum_area_px"] * scale * scale
        maximum_area = parameters["candidate_area_limit_factor"] * parameters["maximum_area_px"] * scale * scale
        candidates = []
        for label in range(1, label_count):
            area = statistics[label, cv2.CC_STAT_AREA]
            # blobs far outside the area limits are not worth measuring
            if area < minimum_area or area > maximum_area:
                continue
            candidate = self.__describe_candidate(
                label_image = label_image,
                label = label,
                statistics = statistics[label],
                seed_mask = seed_mask,
                picture_hsv = analysis_hsv,
                cast_share = cast_share
            )
            if candidate is None:
                continue
            candidate["is_partial"] = False
            candidate["rejection_reason"] = self.__colour_and_context_rejection(candidate)
            if candidate["rejection_reason"] is None:
                shape_reason = self.__shape_rejection(candidate, is_partial = False)
                # a blob that fails the whole-puck tests may be a puck partly hidden by the block, a
                # bracket or a dark shadow, or cut by the picture's edge
                may_be_hidden = (candidate["is_beside_occluder"] or candidate["is_possibly_occluded"]
                                 or candidate["is_touching_picture_edge"]
                                 or candidate["bottom_v_analysis"] >= parameters["hand_region_top_v"])
                if shape_reason is not None and may_be_hidden:
                    partial_reason = self.__shape_rejection(candidate, is_partial = True)
                    if partial_reason is None:
                        candidate["is_partial"] = True
                        shape_reason = None
                    else:
                        shape_reason = f"{shape_reason}+partial_{partial_reason}"
                candidate["rejection_reason"] = shape_reason
            candidate["score"] = 0.0 if candidate["rejection_reason"] else self.__score_candidate(candidate)
            if candidate["rejection_reason"] is None and candidate["score"] < parameters["minimum_score"]:
                candidate["rejection_reason"] = "low_score"
            candidates.append(candidate)

        candidates.sort(key = lambda candidate: candidate["score"], reverse = True)
        return candidates, blue_cast, cast_share


    def detect(self, picture_bgr: numpy.ndarray) -> Optional[Dict]:
        """
        Find the red puck in one camera picture.

        Parameters
        ----------
        picture_bgr : numpy.ndarray
            The camera picture in OpenCV's BGR order (as returned by ``cv2.imread``
            or ``cv2.imdecode``), normally 640 x 480.

        Returns
        -------
        dict or None
            ``None`` when no blob passes. Otherwise, in full-size pixels:
            ``centre_u``, ``centre_v`` (the best estimate of the puck's centre: the
            fitted circle's centre when part of the puck is hidden and the circle
            fits well, else the mean of the visible blob's pixels),
            ``centre_source`` (``"circle_fit"`` or ``"blob"``),
            ``visible_centre_u``, ``visible_centre_v`` (mean of the visible blob's
            pixels), ``area_px``, ``equivalent_radius_px``, ``bounding_box``
            (``[left, top, width, height]`` of the visible blob), ``bottom_u``,
            ``bottom_v`` (the visible blob's lowest row and the mean column there),
            ``is_bottom_clear`` (True when light paper lies just below that point,
            so it is the puck's own outline; False when something dark or blue
            lies there, the block, a bracket, the black line or a shadow, or the
            picture's edge, so part of the puck may be hidden below it),
            ``confidence`` (the score, 0 to 1), ``is_partial`` (accepted only by
            the looser tests of a partly hidden puck), ``is_possibly_occluded``
            (occluder below the blob, light paper above it), ``is_beside_occluder``,
            ``is_touching_picture_edge``, for a partly hidden puck the circle fit
            (``circle_centre_u``, ``circle_centre_v``, ``circle_radius_px``,
            ``is_circle_good``, ``circle_reason``, ``visible_fraction`` and the
            fit's quality measures), ``blue_cast`` and the measurements behind the
            score.
        """

        candidates, blue_cast, cast_share = self.find_candidates(picture_bgr)
        accepted = [candidate for candidate in candidates if candidate["rejection_reason"] is None]
        if not accepted:
            return None

        best = dict(accepted[0])
        best["visible_centre_u"] = best["centre_u"]
        best["visible_centre_v"] = best["centre_v"]
        best["bottom_u"] = best["centre_u"]
        best["bottom_v"] = float(best["bounding_box"][1] + best["bounding_box"][3] - 1)
        best["is_bottom_clear"] = None
        best.update(self.__measure_at_full_size(picture_bgr, best, cast_share))
        # the circle's centre is used only for a puck accepted as partly hidden: the outline of a puck
        # in plain view is a cylinder seen at a slant, which a circle follows only roughly (inlier
        # share 0.36 to 0.94 on whole pucks), so there the blob's own centre is the better estimate
        if best.get("is_circle_good") and best["is_partial"]:
            best["centre_u"] = best["circle_centre_u"]
            best["centre_v"] = best["circle_centre_v"]
            best["centre_source"] = "circle_fit"
        else:
            best["centre_u"] = best["visible_centre_u"]
            best["centre_v"] = best["visible_centre_v"]
            best["centre_source"] = "blob"
        best["confidence"] = round(best.pop("score"), 4)
        best["candidates_accepted"] = len(accepted)
        best["runner_up_confidence"] = round(accepted[1]["score"], 4) if len(accepted) > 1 else None
        best["blue_cast"] = None if blue_cast is None else round(blue_cast, 3)
        for internal_key in ("label", "rejection_reason"):
            del best[internal_key]
        return best


def detect_puck(
        picture_bgr: numpy.ndarray,
        detector_parameters: Optional[dict] = None
) -> Optional[Dict]:
    """
    Find the red puck in one camera picture with a detector built for this call.

    A caller that detects in many pictures should keep one ``PuckDetector`` instead,
    which builds its kernels once.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The camera picture in OpenCV's BGR order, normally 640 x 480.
    detector_parameters : dict, optional
        Overrides for ``PUCK_DETECTOR_PARAMETERS``.

    Returns
    -------
    dict or None
        See ``PuckDetector.detect``.
    """

    return PuckDetector(detector_parameters = detector_parameters).detect(picture_bgr)
