"""
Where the red puck is, seen from the forearm camera, and which part of the table the picture shows: for the recording
tool's orientation window.

For display only (Emranul, 5.11.26: "track the red puck properly, like in real life, in the orientation window"):
nothing the robot does depends on it, so every recorded action stays the person's own. The puck is found in the picture
by puck_detector.py; its pixel becomes a table position through the forearm camera model (forearm_camera_model.py) with
the camera fitted on 6.11.26 (config.FOREARM_CAMERA_PARAMETERS), and only while the camera mount in use is the one that
fit belongs to. Table heights come from the taught table plane (config.table_height).
"""

import copy
from typing import Dict, Optional, Sequence

import numpy

from config import PUCK_TRACKING_PARAMETERS, WORKSPACE_PARAMETERS, table_height
from forearm_camera_model import merged_camera_parameters, picture_footprint_on_plane, pixel_to_table
from puck_detector import PuckDetector


class PuckTracker:

    def __init__(
            self,
            tracking_parameters: Optional[Dict] = None
    ):
        """
        Initialise the tracker with the default settings and any overrides.

        Parameters
        ----------
        tracking_parameters : dict, optional
            Overrides for ``PUCK_TRACKING_PARAMETERS``; also "camera_parameters" (overrides for
            ``FOREARM_CAMERA_PARAMETERS``) and "detector_parameters" (overrides for ``PUCK_DETECTOR_PARAMETERS``).
        """

        self.__initialize_variables()
        self.update_tracking_parameters(tracking_parameters)


    def __initialize_variables(self):

        self.tracking_parameters = copy.deepcopy(PUCK_TRACKING_PARAMETERS)
        self.tracking_parameters.update({"camera_parameters": {}, "detector_parameters": {}})
        self.camera_parameters = merged_camera_parameters()
        self.detector = PuckDetector()


    def update_tracking_parameters(self, tracking_parameters: Optional[Dict]):
        """
        Merge overrides into the current settings; the camera and the detector follow their own overrides.

        Parameters
        ----------
        tracking_parameters : dict or None
            Entries to replace; "camera_parameters" and "detector_parameters" are merged key by key into the overrides
            given before, so a later call keeps the earlier ones.
        """

        if not tracking_parameters:
            return
        overrides = copy.deepcopy(tracking_parameters)
        camera_overrides = overrides.pop("camera_parameters", None)
        detector_overrides = overrides.pop("detector_parameters", None)
        self.tracking_parameters.update(overrides)
        if camera_overrides is not None:
            self.tracking_parameters["camera_parameters"].update(camera_overrides)
            self.camera_parameters = merged_camera_parameters(self.tracking_parameters["camera_parameters"])
        if detector_overrides is not None:
            self.tracking_parameters["detector_parameters"].update(detector_overrides)
            self.detector.update_detector_parameters(detector_overrides)


    def is_calibrated(self) -> bool:
        """Whether the camera parameters were fitted for the camera mount in use, so that pixels may become table positions."""

        return self.tracking_parameters["camera_mount"] == self.camera_parameters.get("camera_mount")


    def see_puck(
            self,
            picture_bgr: numpy.ndarray,
            joints: Sequence[float]
    ) -> Optional[Dict]:
        """
        Find the red puck in a picture and, with a calibrated camera, place it on the table.

        Parameters
        ----------
        picture_bgr : numpy.ndarray
            The camera picture, as OpenCV decodes it.
        joints : sequence of float
            The arm's six joint angles when the picture was taken, in radians.

        Returns
        -------
        dict or None
            None when no puck is seen; otherwise "pixel" ([u, v], the puck's centre in the picture), "radius_px",
            "confidence", "is_partial" (part of the puck hidden by the hand or the picture's edge), "table_xy" ([x, y]
            in metres in the robot's base frame, or None while the camera is not calibrated for the mount in use) and
            "is_position_rough" (True when only a sliver shows and no circle fitted its outline: the centre is then the
            visible part's middle, pulled away from whatever hides the rest, by about 6 to 10 mm on the table in the
            recorded cases).
        """

        detection = self.detector.detect(picture_bgr)
        if detection is None or detection["confidence"] < self.tracking_parameters["minimum_confidence"]:
            return None

        pixel = [float(detection["centre_u"]), float(detection["centre_v"])]
        # A partly hidden puck whose outline gave a good circle is drawn with that circle; otherwise the visible blob's
        # size is the best there is.
        is_circle_fit = detection["centre_source"] == "circle_fit"
        radius_px = float(detection["circle_radius_px"] if is_circle_fit else detection["equivalent_radius_px"])
        table_xy = None
        if self.is_calibrated():
            table_xy = self.__pixel_on_table(pixel, joints, self.tracking_parameters["puck_point_height_above_table_m"])
        return {
            "pixel": pixel,
            "radius_px": radius_px,
            "confidence": float(detection["confidence"]),
            "is_partial": bool(detection["is_partial"]),
            "table_xy": table_xy,
            "is_position_rough": bool(detection["is_partial"]) and not is_circle_fit
        }


    def picture_outline_on_table(self, joints: Sequence[float]) -> Optional[numpy.ndarray]:
        """
        The table area the picture covers, for drawing it on the map.

        Parameters
        ----------
        joints : sequence of float
            The arm's six joint angles, in radians.

        Returns
        -------
        numpy.ndarray or None
            The outline on the table, shape (4 * outline_points_per_edge, 2), walked from the picture's top-left corner
            clockwise (with one point per edge: top left, top right, bottom right, bottom left); None while the camera
            is not calibrated for the mount in use.
        """

        if not self.is_calibrated():
            return None
        width, height = self.camera_parameters["image_size_px"]
        centre_xy = self.__pixel_on_table([(width - 1) / 2.0, (height - 1) / 2.0], joints, 0.0)
        # The table's height where the picture's centre lands (at the start of the line when the centre looks above the
        # horizon); the table tilts by about 1 mm in 10 cm, too little to bend the outline visibly.
        plane_xy = centre_xy if centre_xy is not None else WORKSPACE_PARAMETERS["line_start_xy"]
        footprint_settings = {"points_per_edge": self.tracking_parameters["outline_points_per_edge"],
                              "maximum_range_m": self.tracking_parameters["outline_maximum_range_m"]}
        return picture_footprint_on_plane(joints, table_height(plane_xy[0], plane_xy[1]), self.camera_parameters, footprint_settings)


    def __pixel_on_table(
            self,
            pixel: Sequence[float],
            joints: Sequence[float],
            height_above_table_m: float
    ) -> Optional[list]:
        """
        Where the ray through ``pixel`` meets the plane ``height_above_table_m`` above the taught table, or None when it
        does not meet it in front of the camera.
        """

        plane_xy = WORKSPACE_PARAMETERS["line_start_xy"]
        point_xy = None
        for _ in range(max(1, int(self.tracking_parameters["height_passes"]))):
            plane_height_m = table_height(plane_xy[0], plane_xy[1]) + height_above_table_m
            point_xy = pixel_to_table(pixel[0], pixel[1], joints, plane_height_m, self.camera_parameters)
            if point_xy is None:
                return None
            plane_xy = point_xy
        return [float(point_xy[0]), float(point_xy[1])]



def see_puck(
        picture_bgr: numpy.ndarray,
        joints: Sequence[float],
        tracking_parameters: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Find the red puck in one picture with a tracker built for this call (see ``PuckTracker.see_puck``).

    A caller that looks at many pictures should keep one ``PuckTracker`` instead, which builds the detector once.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The camera picture, as OpenCV decodes it.
    joints : sequence of float
        The arm's six joint angles when the picture was taken, in radians.
    tracking_parameters : dict, optional
        Overrides for ``PUCK_TRACKING_PARAMETERS``.

    Returns
    -------
    dict or None
        See ``PuckTracker.see_puck``.
    """

    return PuckTracker(tracking_parameters = tracking_parameters).see_puck(picture_bgr, joints)



def picture_outline_on_table(
        joints: Sequence[float],
        tracking_parameters: Optional[Dict] = None
) -> Optional[numpy.ndarray]:
    """
    The table area the picture covers, with a tracker built for this call (see ``PuckTracker.picture_outline_on_table``).

    Parameters
    ----------
    joints : sequence of float
        The arm's six joint angles, in radians.
    tracking_parameters : dict, optional
        Overrides for ``PUCK_TRACKING_PARAMETERS``.

    Returns
    -------
    numpy.ndarray or None
        See ``PuckTracker.picture_outline_on_table``.
    """

    return PuckTracker(tracking_parameters = tracking_parameters).picture_outline_on_table(joints)
