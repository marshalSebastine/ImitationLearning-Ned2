"""
Question 6 of the orientation-coverage analysis: could training pictures be rotated to imitate other orientations? An
assessment for this camera, not a training run.

The camera is taped to the forearm, so the robot's own parts (block, gripper jaws) stay fixed in the lower part of the
picture while the table turns round them when the forearm twists or the sheet lies differently. Four checks:
    A  calculated (arm model and fitted camera): where the magnet tip's table point lies in the picture, the point the
       table content turns about; how far a plain picture rotation about that point is from the true picture of a
       turned table; and how far a rotation about the picture's centre is off;
    B  calculated: how much of the picture has no content to show after such a rotation (from outside the frame and
       from behind the robot's parts);
    C  measured: pairs of recorded pictures between which the forearm twisted while the tip stood still; the picture
       motion between them (rotation angle and the point that stays fixed) against the calculation;
    D  measured: what a rotation does to the labels. The step turns with the picture; the forearm twist does not, and
       the demonstrators' twist depends on where the line lies in the picture.

Printed and saved: 06_rotation_augmentation.out.txt, 06_twist_pairs.csv, 06_rotation_example.png (one recorded picture,
its table content turned by 30 degrees about the tip with the robot's parts kept, and the same turned about the picture
centre).

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe 06_rotation_augmentation.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy
import pandas

import orientation_common
from orientation_common import EVIDENCE_FOLDER, demonstration_frames, wrapped_deg

from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, pushing_height, table_height  # noqa: E402
from forearm_camera_model import pixels_to_plane, project_points  # noqa: E402
from record_episodes import forearm_camera_heading_rad  # noqa: E402
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed  # noqa: E402

ROTATION_PARAMETERS = {
    "reaches_cm": [15.0, 25.0, 35.0, 45.0],
    "joint_4_values_deg": [-90.0, -45.0, 0.0, 45.0, 90.0],
    "wrist_held_deg": MODEL_MODE_PARAMETERS["wrist_held_deg"],
    "hand_held_deg": MODEL_MODE_PARAMETERS["hand_held_deg"],
    "rotation_angles_deg": [5.0, 10.0, 20.0, 30.0, 45.0, 90.0],
    "ring_distances_cm": [3.0, 5.0, 8.0, 12.0],
    "picture_size_px": [640, 480],
    "network_picture_width_px": 160,
    # the robot's own parts: as in 04_physics.py
    "mask_pictures": 1500,
    "dark_grey_below": 85,
    "blue_minus_red_above": 75,
    "blue_least": 120,
    "red_minus_green_most": 15,
    "seed": 11_11_26,
    # the bend ahead of a row: the path direction between "future_rows" rows later, minus the path direction now
    "future_rows": [5, 10],
    "future_turn_edges_deg": [-180.0, -60.0, -30.0, -10.0, 10.0, 30.0, 60.0, 180.0],
    # twist-in-place pairs: the tip moved less than this, the forearm twisted between these, the wrist and hand
    # changed less than this, and at most this many pairs are aligned
    "pair_tip_most_cm": 0.2,
    "pair_twist_deg": [5.0, 12.0],
    "pair_other_joint_most_deg": 0.5,
    "pairs_most": 160,
    # the picture motion is fitted (rotation and shift) on the part of the picture above the robot's parts
    "align_rows_above": 330,
    "align_scale": 0.5,
    "example_plan": "01-04",
    "example_row": 12,
    "example_angle_deg": 30.0
}



def driver_joints(
        reach_cm: float,
        joint_4_deg: float,
        parameters: Dict
) -> Optional[List[float]]:
    """Return the joints the driver would hold with the tip ``reach_cm`` in front of the base and joint 4 at ``joint_4_deg``."""

    x = reach_cm / 100.0
    target = numpy.array([x, 0.0, pushing_height(x, 0.0) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    joints = solve_with_wrist_fixed(target, math.radians(joint_4_deg), math.radians(parameters["wrist_held_deg"]), [0.0, -0.8, -0.6])
    if joints is None:
        return None
    joints[5] = math.radians(parameters["hand_held_deg"])
    return joints



def rotated_about(
        points: numpy.ndarray,
        centre: numpy.ndarray,
        angle_deg: float
) -> numpy.ndarray:
    """Return 2-D ``points`` turned by ``angle_deg`` about ``centre`` (counter-clockwise for x right, y up)."""

    cosine, sine = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    return (points - centre) @ numpy.array([[cosine, sine], [-sine, cosine]]) + centre



def rotation_geometry(parameters: Dict) -> pandas.DataFrame:
    """
    Calculate the tip's picture point and the error of plain picture rotations, per reach and forearm twist.

    For table points on rings round the tip, the true picture of the table turned by an angle about the tip is found by
    turning the points on the table and projecting them; a plain picture rotation turns their pixels about the tip's
    pixel by the same angle (clockwise on the screen for a counter-clockwise table turn is avoided by the sign check
    below: the picture rotation that fits best is used).

    Returns
    -------
    pandas.DataFrame
        One row per reach, twist and angle: the tip's pixel, the median and largest error of the plain rotation about
        the tip's pixel over the rings (pixels of 640 x 480), the same within 5 cm of the tip, and the shift of the
        tip's pixel itself when the picture is instead rotated about its centre.
    """

    width, height = parameters["picture_size_px"]
    centre = numpy.array([(width - 1) / 2.0, (height - 1) / 2.0])
    rows = []
    for reach_cm in parameters["reaches_cm"]:
        for joint_4_deg in parameters["joint_4_values_deg"]:
            joints = driver_joints(reach_cm, joint_4_deg, parameters)
            if joints is None:
                continue
            tip = magnet_tip_and_lean(joints)[0]
            table_z = table_height(tip[0], tip[1])
            heading = forearm_camera_heading_rad(joints)
            tip_pixel = project_points(numpy.array([[tip[0], tip[1], table_z]]), joints)[0]
            offsets, distances = [], []
            for distance in parameters["ring_distances_cm"]:
                for direction in numpy.radians(numpy.arange(-90.0, 91.0, 15.0)):
                    offsets.append([distance / 100.0 * math.cos(heading + direction), distance / 100.0 * math.sin(heading + direction)])
                    distances.append(distance)
            offsets, distances = numpy.array(offsets), numpy.array(distances)
            points = numpy.column_stack([tip[0] + offsets[:, 0], tip[1] + offsets[:, 1], numpy.full(len(offsets), table_z)])
            pixels = project_points(points, joints)
            for angle_deg in parameters["rotation_angles_deg"]:
                turned = rotated_about(offsets, numpy.zeros(2), angle_deg)
                turned_points = numpy.column_stack([tip[0] + turned[:, 0], tip[1] + turned[:, 1], numpy.full(len(turned), table_z)])
                true_pixels = project_points(turned_points, joints)
                # on the screen y points down, so a counter-clockwise table turn is a rotation by minus the angle in
                # pixel coordinates; both signs are tried and the better one kept, which also checks the sign
                errors = {}
                for sign in (1.0, -1.0):
                    plain = rotated_about(pixels, tip_pixel, sign * angle_deg)
                    errors[sign] = numpy.linalg.norm(plain - true_pixels, axis = 1)
                sign = min(errors, key = lambda value: numpy.nanmedian(errors[value]))
                error = errors[sign]
                is_seen = numpy.isfinite(error) & (true_pixels[:, 0] >= 0) & (true_pixels[:, 0] < width) & (true_pixels[:, 1] >= 0) & (true_pixels[:, 1] < height)
                near = is_seen & (distances <= 5.0)
                about_centre = rotated_about(tip_pixel[None, :], centre, sign * angle_deg)[0]
                rows.append({
                    "reach_cm": reach_cm, "joint_4_deg": joint_4_deg, "angle_deg": angle_deg,
                    "tip_u": round(float(tip_pixel[0]), 1), "tip_v": round(float(tip_pixel[1]), 1), "pixel_sign": sign,
                    "plain_rotation_error_median_px": round(float(numpy.median(error[is_seen])), 1) if is_seen.any() else numpy.nan,
                    "plain_rotation_error_most_px": round(float(error[is_seen].max()), 1) if is_seen.any() else numpy.nan,
                    "error_within_5_cm_median_px": round(float(numpy.median(error[near])), 1) if near.any() else numpy.nan,
                    "tip_shift_if_about_centre_px": round(float(numpy.linalg.norm(about_centre - tip_pixel)), 1)
                })
    return pandas.DataFrame(rows)



def robot_parts_mask(parameters: Dict) -> numpy.ndarray:
    """Return the mask of the robot's own parts at 480 x 640 (True on the parts), found as in 04_physics.py."""

    cache = numpy.load(Path(parameters["index_folder"]) / "images_wide_160.npy", mmap_mode = "r")
    randomness_generator = numpy.random.default_rng(parameters["seed"])
    chosen = numpy.sort(randomness_generator.choice(cache.shape[0], parameters["mask_pictures"], replace = False))
    median = numpy.median(numpy.array(cache[chosen]).astype(numpy.int16), axis = 0)
    red, green, blue = median[..., 0], median[..., 1], median[..., 2]
    grey = median.mean(axis = 2)
    is_dark = (grey < parameters["dark_grey_below"]) & (red - green <= parameters["red_minus_green_most"])
    is_blue = (blue - red > parameters["blue_minus_red_above"]) & (blue > parameters["blue_least"])
    small = cv2.morphologyEx((is_dark | is_blue).astype(numpy.uint8), cv2.MORPH_OPEN, numpy.ones((2, 2), numpy.uint8))
    small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, numpy.ones((3, 3), numpy.uint8))
    small = numpy.maximum.accumulate(small, axis = 0)
    return cv2.resize(small, (640, 480), interpolation = cv2.INTER_NEAREST) > 0



def table_turn_map(
        joints: List[float],
        angle_deg: float,
        parameters: Dict
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return, for every pixel of the turned picture, the pixel of the recorded picture that shows the same table point
    when the table content is turned by ``angle_deg`` about the tip's table point (maps for cv2.remap; NaN-free, pixels
    without a source point far outside the frame).
    """

    width, height = parameters["picture_size_px"]
    tip = magnet_tip_and_lean(joints)[0]
    table_z = table_height(tip[0], tip[1])
    grid_u, grid_v = numpy.meshgrid(numpy.arange(0, width, 4, dtype = float), numpy.arange(0, height, 4, dtype = float))
    pixels = numpy.column_stack([grid_u.ravel(), grid_v.ravel()])
    places = pixels_to_plane(pixels, joints, table_z)
    source_places = rotated_about(places, tip[:2], -angle_deg)
    source_points = numpy.column_stack([source_places, numpy.full(len(source_places), table_z)])
    source_pixels = project_points(source_points, joints)
    source_pixels[~numpy.isfinite(source_pixels).all(axis = 1)] = -10_000.0
    map_u = cv2.resize(source_pixels[:, 0].reshape(grid_u.shape).astype(numpy.float32), (width, height), interpolation = cv2.INTER_LINEAR)
    map_v = cv2.resize(source_pixels[:, 1].reshape(grid_u.shape).astype(numpy.float32), (width, height), interpolation = cv2.INTER_LINEAR)
    return map_u, map_v



def empty_share_table(
        mask: numpy.ndarray,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Calculate how much of the turned picture has nothing to show, per angle (reach 25 cm, forearm straight).

    A pixel of the turned picture outside the robot's parts is empty when its table point lay outside the recorded
    frame, or behind the robot's parts, before the turn.

    Returns
    -------
    pandas.DataFrame
        One row per angle: the share of the table part of the picture that is empty, split by cause.
    """

    width, height = parameters["picture_size_px"]
    joints = driver_joints(25.0, 0.0, parameters)
    rows = []
    for angle_deg in parameters["rotation_angles_deg"]:
        shares = []
        for sign in (1.0, -1.0):
            map_u, map_v = table_turn_map(joints, sign * angle_deg, parameters)
            is_outside = (map_u < 0) | (map_u > width - 1) | (map_v < 0) | (map_v > height - 1)
            source_u = numpy.clip(numpy.round(map_u).astype(int), 0, width - 1)
            source_v = numpy.clip(numpy.round(map_v).astype(int), 0, height - 1)
            is_behind_parts = ~is_outside & mask[source_v, source_u]
            table_part = ~mask
            shares.append((float((is_outside & table_part).sum() / table_part.sum()), float((is_behind_parts & table_part).sum() / table_part.sum())))
        rows.append({"angle_deg": angle_deg, "empty_from_outside_share": round(float(numpy.mean([share[0] for share in shares])), 3),
                     "empty_from_behind_parts_share": round(float(numpy.mean([share[1] for share in shares])), 3)})
    table = pandas.DataFrame(rows)
    table["empty_share"] = (table["empty_from_outside_share"] + table["empty_from_behind_parts_share"]).round(3)
    return table



def twist_pairs(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Align pairs of recorded pictures between which the forearm twisted in place, and measure the picture's motion.

    Each pair is a row and its next row with the tip moved less than "pair_tip_most_cm", joint 4 changed by
    "pair_twist_deg" and the wrist and hand unchanged. The motion of the picture above the robot's parts is fitted as a
    rotation and a shift (cv2.findTransformECC, coarse to fine); the point that the fitted motion leaves in place is the
    centre of the rotation.

    Returns
    -------
    pandas.DataFrame
        One row per aligned pair: the forearm twist, the calculated change of the picture's heading, the fitted
        rotation, the fixed point (pixels of 640 x 480), the tip's calculated pixel and the reach.
    """

    grouped = frames.groupby("episode_id", sort = False)
    next_joint_5 = numpy.degrees(grouped["joint_5"].shift(-1) - frames["joint_5"]).abs()
    next_joint_6 = numpy.degrees(grouped["joint_6"].shift(-1) - frames["joint_6"]).abs()
    next_heading = wrapped_deg(grouped["picture_heading_deg"].shift(-1) - frames["picture_heading_deg"])
    is_pair = (frames["is_next_consecutive"] & (frames["step_cm"] < parameters["pair_tip_most_cm"])
               & frames["forearm_change_deg"].abs().between(*parameters["pair_twist_deg"])
               & (next_joint_5 < parameters["pair_other_joint_most_deg"]) & (next_joint_6 < parameters["pair_other_joint_most_deg"]))
    candidates = numpy.flatnonzero(is_pair.to_numpy())
    randomness_generator = numpy.random.default_rng(parameters["seed"])
    chosen = numpy.sort(randomness_generator.permutation(candidates)[:parameters["pairs_most"]])
    scale = parameters["align_scale"]
    rows_above = int(parameters["align_rows_above"] * scale)
    results = []
    for position in chosen:
        first, second = frames.iloc[position], frames.iloc[position + 1]
        pictures = [cv2.imread(str(Path(parameters["raw_folder"]) / row["frame_path"]), cv2.IMREAD_GRAYSCALE) for row in (first, second)]
        if pictures[0] is None or pictures[1] is None:
            continue
        small = [cv2.GaussianBlur(cv2.resize(picture, None, fx = scale, fy = scale, interpolation = cv2.INTER_AREA), (5, 5), 0).astype(numpy.float32)
                 for picture in pictures]
        mask = numpy.zeros(small[0].shape, dtype = numpy.uint8)
        mask[:rows_above] = 255
        warp = numpy.eye(2, 3, dtype = numpy.float32)
        try:
            # coarse to fine: a quarter of the size first, then the full small size
            coarse = [cv2.resize(picture, None, fx = 0.25, fy = 0.25, interpolation = cv2.INTER_AREA) for picture in small]
            coarse_mask = cv2.resize(mask, (coarse[0].shape[1], coarse[0].shape[0]), interpolation = cv2.INTER_NEAREST)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            _, warp = cv2.findTransformECC(coarse[0], coarse[1], warp, cv2.MOTION_EUCLIDEAN, criteria, coarse_mask, 5)
            warp[:, 2] *= 4.0
            correlation, warp = cv2.findTransformECC(small[0], small[1], warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 5)
        except cv2.error:
            continue
        # the warp maps a pixel of the first picture to where the same content lies in the second
        rotation = warp[:, :2].astype(float)
        shift = warp[:, 2].astype(float) / scale
        angle_deg = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        if abs(angle_deg) < 1.0:
            continue
        fixed = numpy.linalg.solve(numpy.eye(2) - rotation, shift)
        joints = [float(first[f"joint_{joint}"]) for joint in range(1, 7)]
        tip_pixel = project_points(numpy.array([[first["tip_x_m"], first["tip_y_m"], table_height(first["tip_x_m"], first["tip_y_m"])]]), joints)[0]
        results.append({
            "plan": first["plan_number"], "row": int(first["row_in_episode"]), "reach_cm": round(float(first["reach_cm"]), 1),
            "joint_4_deg": round(float(first["joint_4_deg"]), 1), "forearm_change_deg": round(float(first["forearm_change_deg"]), 2),
            "heading_change_deg": round(float(numpy.asarray(next_heading)[position]), 2), "picture_rotation_deg": round(angle_deg, 2),
            "fixed_u": round(float(fixed[0]), 0), "fixed_v": round(float(fixed[1]), 0), "tip_u": round(float(tip_pixel[0]), 0), "tip_v": round(float(tip_pixel[1]), 0),
            "correlation": round(float(correlation), 3)
        })
    return pandas.DataFrame(results)



def example_picture(
        frames: pandas.DataFrame,
        mask: numpy.ndarray,
        parameters: Dict
) -> None:
    """Save one recorded picture beside two rotated versions: the table turned about the tip with the robot's parts kept, and a plain rotation about the centre."""

    width, height = parameters["picture_size_px"]
    row = frames[(frames["plan_number"] == parameters["example_plan"]) & (frames["row_in_episode"] == parameters["example_row"])].iloc[0]
    picture = cv2.imread(str(Path(parameters["raw_folder"]) / row["frame_path"]))
    joints = [float(row[f"joint_{joint}"]) for joint in range(1, 7)]
    angle_deg = parameters["example_angle_deg"]
    map_u, map_v = table_turn_map(joints, angle_deg, parameters)
    turned = cv2.remap(picture, map_u, map_v, cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 255, 255))
    source_u = numpy.clip(numpy.round(map_u).astype(int), 0, width - 1)
    source_v = numpy.clip(numpy.round(map_v).astype(int), 0, height - 1)
    is_inside = (map_u >= 0) & (map_u <= width - 1) & (map_v >= 0) & (map_v <= height - 1)
    turned[is_inside & mask[source_v, source_u]] = (0, 255, 255)
    turned[mask] = picture[mask]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle_deg, 1.0)
    plain = cv2.warpAffine(picture, matrix, (width, height), borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 255, 255))
    for image, text in [(picture, "recorded"), (turned, f"table turned {angle_deg:.0f} deg about the tip, robot parts kept"),
                        (plain, f"plain rotation {angle_deg:.0f} deg about the picture centre")]:
        cv2.putText(image, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(EVIDENCE_FOLDER / "06_rotation_example.png"), numpy.hstack([picture, turned, plain]))



def main(parameters: Optional[Dict] = None) -> None:
    """Print the four checks and save the pair table and the example picture."""

    settings = orientation_common.merged_parameters(ROTATION_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 500)
    pandas.set_option("display.max_columns", 40)
    frames = demonstration_frames(settings)
    width, height = settings["picture_size_px"]
    to_network = settings["network_picture_width_px"] / width

    print("=== A. The point the table content turns about, and how good a plain picture rotation is (calculated: arm model, fitted camera) ===")
    geometry = rotation_geometry(settings)
    tip = geometry.drop_duplicates(["reach_cm", "joint_4_deg"])
    print("The tip's table point in the picture (pixels of 640 x 480; the picture's centre is (319.5, 239.5)):")
    print(tip.pivot(index = "reach_cm", columns = "joint_4_deg", values = "tip_u").round(0).to_string())
    print(tip.pivot(index = "reach_cm", columns = "joint_4_deg", values = "tip_v").round(0).to_string())
    print(f"  over all reaches and twists: u {tip['tip_u'].min():.0f} to {tip['tip_u'].max():.0f} (median {tip['tip_u'].median():.0f}), "
          f"v {tip['tip_v'].min():.0f} to {tip['tip_v'].max():.0f} (median {tip['tip_v'].median():.0f}); at the network's 160 x 120: "
          f"({tip['tip_u'].median() * to_network:.0f}, {tip['tip_v'].median() * to_network:.0f})")
    print(f"  pixel sign of the best plain rotation: {sorted(geometry['pixel_sign'].unique())} (-1 = a counter-clockwise table turn is a counter-clockwise turn on the screen)")
    print("\nA plain picture rotation about the tip's pixel against the true picture of the turned table (pixels of 640 x 480; forearm straight):")
    straight = geometry[geometry["joint_4_deg"] == 0.0]
    for value, title in [("error_within_5_cm_median_px", "median error of table points within 5 cm of the tip"),
                         ("plain_rotation_error_median_px", "median error of table points 3 to 12 cm from the tip"),
                         ("plain_rotation_error_most_px", "largest error, 3 to 12 cm"),
                         ("tip_shift_if_about_centre_px", "shift of the tip's own pixel when the picture is rotated about its centre instead")]:
        print(f"{title}:")
        print(straight.pivot(index = "reach_cm", columns = "angle_deg", values = value).to_string())
    print(f"  one pixel of the network's picture is {1 / to_network:.0f} pixels here.")

    print("\n=== B. How much of the turned picture has nothing to show (calculated; reach 25 cm, forearm straight; shares of the table part of the picture) ===")
    mask = robot_parts_mask(settings)
    empty = empty_share_table(mask, settings)
    print(empty.to_string(index = False))
    print(f"  the robot's own parts cover {100.0 * mask.mean():.1f} % of the picture and must stay as recorded.")

    print("\n=== C. Recorded pairs with the forearm twisting in place: how the picture really moves (measured) ===")
    pairs = twist_pairs(frames, settings)
    pairs.to_csv(EVIDENCE_FOLDER / "06_twist_pairs.csv", index = False)
    good = pairs[pairs["correlation"] >= 0.9]
    print(f"Pairs aligned: {len(pairs)}; with an alignment correlation of 0.9 or more: {len(good)}")
    if len(good) >= 10:
        ratio = good["picture_rotation_deg"] / good["heading_change_deg"]
        slope = numpy.polyfit(good["heading_change_deg"], good["picture_rotation_deg"], 1)
        print(f"  fitted picture rotation against the calculated change of the picture's heading: slope {slope[0]:+.2f}, offset {slope[1]:+.2f} degrees; "
              f"ratio median {ratio.median():+.2f} (25th to 75th percentile {ratio.quantile(0.25):+.2f} to {ratio.quantile(0.75):+.2f})")
        print(f"  the point the picture turns about: u median {good['fixed_u'].median():.0f} (25th to 75th percentile {good['fixed_u'].quantile(0.25):.0f} to "
              f"{good['fixed_u'].quantile(0.75):.0f}), v median {good['fixed_v'].median():.0f} ({good['fixed_v'].quantile(0.25):.0f} to {good['fixed_v'].quantile(0.75):.0f}); "
              f"the tip's calculated pixel in the same pictures: u {good['tip_u'].median():.0f}, v {good['tip_v'].median():.0f}; the picture's centre: (320, 240)")
        print(f"  distance of the fitted fixed point from the tip's pixel: median {numpy.hypot(good['fixed_u'] - good['tip_u'], good['fixed_v'] - good['tip_v']).median():.0f} px; "
              f"from the picture's centre: median {numpy.hypot(good['fixed_u'] - 320, good['fixed_v'] - 240).median():.0f} px")

    print("\n=== D. What a rotation does to the labels (measured from the index) ===")
    rows = frames[frames["path_table_deg"].notna() & (frames["action"] != "stop")].copy()
    rows["line_in_picture_bin"] = (numpy.round(rows["path_picture_deg"] / 30.0) * 30.0).astype(int)
    rows.loc[rows["line_in_picture_bin"] == -180, "line_in_picture_bin"] = 180
    summary = rows.groupby("line_in_picture_bin").agg(
        pictures = ("frame_index", "size"), twist_mean_deg = ("forearm_change_deg", "mean"), twist_median_deg = ("forearm_change_deg", "median"),
        twist_towards_line_share = ("forearm_change_deg", lambda values: float("nan")),
        step_cm_median = ("step_cm", "median"))
    for bin_centre in summary.index:
        part = rows[rows["line_in_picture_bin"] == bin_centre]
        if bin_centre in (0, 180):
            continue
        # joint 4 falling turns the camera to the left: towards a line that lies to the left (positive direction)
        summary.loc[bin_centre, "twist_towards_line_share"] = float((numpy.sign(part["forearm_change_deg"]) == -numpy.sign(bin_centre)).mean())
    print("The demonstrated forearm twist of a row, by where the path of the next three steps lies in the picture (0 = to its top, + = to its left):")
    print(summary.round(2).to_string())
    middle = rows[rows["path_picture_deg"].abs() <= 90]
    slope = numpy.polyfit(middle["path_picture_deg"], middle["forearm_change_deg"], 1)[0]
    twisting = rows.loc[rows["forearm_change_deg"].abs() >= 2, "forearm_change_deg"].abs().median()
    print(f"  within 90 degrees of the picture's top the mean twist differs by {slope:+.3f} degrees per degree of the line's direction in the picture "
          f"({30 * slope:+.1f} degrees between pictures whose lines lie 30 degrees apart; the median |twist| of the rows that twist by 2 degrees or more is {twisting:.1f}). "
          "So the twist label is tied to where the line lies in the picture, and it cannot stay unchanged when the picture is turned.")
    grouped = frames.groupby("episode_id", sort = False)
    first_row, last_row = settings["future_rows"]
    ahead_x = grouped["tip_x_m"].shift(-last_row) - grouped["tip_x_m"].shift(-first_row)
    ahead_y = grouped["tip_y_m"].shift(-last_row) - grouped["tip_y_m"].shift(-first_row)
    is_usable = frames["path_table_deg"].notna() & ahead_x.notna() & (numpy.hypot(ahead_x, ahead_y) > 0.02) & (frames["action"] != "stop")
    bends = frames[is_usable].copy()
    bends["future_turn_deg"] = wrapped_deg(numpy.degrees(numpy.arctan2(ahead_y[is_usable], ahead_x[is_usable])) - bends["path_table_deg"])
    bends["future_turn_bin"] = pandas.cut(bends["future_turn_deg"], settings["future_turn_edges_deg"])
    print(f"\nThe same twist by the bend ahead (the path direction {first_row} to {last_row} rows later minus the path direction now; + = the path will turn left):")
    print(bends.groupby("future_turn_bin", observed = True).agg(
        pictures = ("frame_index", "size"), twist_mean_deg = ("forearm_change_deg", "mean"),
        picture_top_ahead_of_path_deg = ("path_picture_deg", lambda values: -float(values.mean()))).round(2).to_string())
    print(f"  correlation of the twist with the bend ahead {numpy.corrcoef(bends['forearm_change_deg'], bends['future_turn_deg'])[0, 1]:+.2f}, with the line's direction in "
          f"the picture {numpy.corrcoef(bends['forearm_change_deg'], bends['path_picture_deg'])[0, 1]:+.2f}: the demonstrators turn the camera into the bend ahead "
          "(joint 4 falling turns it left), so the picture's top runs ahead of the path; the twist answers the line's shape ahead, which a rotated picture changes too.")
    print("  The step label (tip forward, tip left) and the five chunk points turn with the picture by the rotation angle; the stop label stays;")
    print("  the forearm twist, lean and hand labels have no rule: they would have to be left out of the loss for rotated pictures.")

    example_picture(frames, mask, settings)
    print("\nSaved 06_twist_pairs.csv, 06_rotation_example.png")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding = "utf-8")
    main()
