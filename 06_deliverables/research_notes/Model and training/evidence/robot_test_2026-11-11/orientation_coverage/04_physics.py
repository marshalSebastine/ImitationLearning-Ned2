"""
Question 4 of the orientation-coverage analysis: what does the arm physically allow?

The camera is taped to the forearm and turns with joint 4, so the picture's top can be turned away from the arm's radial
direction only as far as joint 4 goes. Calculated here from the arm model with the robot driver's own rule (wrist held
at 13.5 degrees, joints 1 to 3 put the tip on its target), and checked against the demonstrations:
    A  how far the picture's top turns from radial, and how far the rod leans, for every reach and forearm twist;
    B  what the camera sees round the tip: for which travel directions in the picture the puck and the line ahead are in
       view, and where the robot's own parts hide them (fitted camera model and a mask of the robot's parts taken from
       the median of the recorded pictures);
    C  the same measured on recorded pictures: how often the puck detector finds the puck, per travel direction in the
       picture, and how large the puck is;
    D  the consequence per line direction: which directions from radial can be followed with the camera looking along
       the line, which only sideways in the picture, and how the demonstrators handled each; forearm flips;
    E  how fast the lines of the six course sheets and the six practice tracks bend, against how fast the driver can
       turn the camera while it moves.

Printed and saved: 04_physics.out.txt, 04_heading_and_lean.csv, 04_view_by_direction.csv, 04_line_bends.csv,
04_physics.png.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe 04_physics.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
from matplotlib import pyplot

import orientation_common
from orientation_common import EVIDENCE_FOLDER, arc_lengths, course_sheet_lines, demonstration_frames, practice_track_lines, wrapped_deg

from config import MODEL_MODE_PARAMETERS, PUCK_PARAMETERS, PUCK_TRACKING_PARAMETERS, PUSHER_PARAMETERS, pushing_height, table_height  # noqa: E402
from forearm_camera_model import pixels_to_plane, project_points  # noqa: E402
from puck_detector import PuckDetector  # noqa: E402
from record_episodes import forearm_camera_heading_rad  # noqa: E402
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed  # noqa: E402

PHYSICS_PARAMETERS = {
    "reaches_cm": [12.0, 15.0, 18.0, 21.0, 25.0, 30.0, 35.0, 40.0, 45.0, 47.0, 48.0, 49.5],
    "joint_4_values_deg": [-117.0, -100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 117.0],
    "joint_4_most_deg": MODEL_MODE_PARAMETERS["joint_4_most_deg"],
    "wrist_held_deg": MODEL_MODE_PARAMETERS["wrist_held_deg"],
    "hand_held_deg": MODEL_MODE_PARAMETERS["hand_held_deg"],
    "rod_lean_most_deg": MODEL_MODE_PARAMETERS["rod_lean_most_deg"],
    "twist_step_most_deg": MODEL_MODE_PARAMETERS["twist_step_most_deg"],
    # the view round the tip: reaches and twists looked at, the travel directions in the picture, and how far ahead of
    # the tip the line is followed (a straight line in the travel direction, a simplification)
    "view_reaches_cm": [15.0, 25.0, 35.0, 45.0],
    "view_direction_step_deg": 15.0,
    "line_ahead_cm": [3.0, 15.0],
    "line_ahead_step_cm": 0.5,
    # the puck's centre lies this far ahead of the tip while it is pushed: measured with the puck detector on forward
    # pushes (part C: median 2.56 cm; config.py's 1.85 cm is the contact distance of the bare rod tip, not of the block)
    "puck_ahead_of_tip_cm": orientation_common.ORIENTATION_PARAMETERS["tip_behind_puck_cm"],
    "puck_height_cm": 100.0 * PUCK_PARAMETERS["height_m"],
    # the robot's own parts in the picture: from the per-pixel median of this many cached pictures, the pixels darker
    # than "dark_grey_below" (the block, the black box) or bluer than "blue_minus_red_above" (the gripper's jaws), but
    # not redder than "red_minus_green_most" (the puck, which sits in front of the block in most pictures); everything
    # below such a pixel in its column counts too (the grey rod and wrist between the jaws)
    "mask_pictures": 1500,
    "dark_grey_below": 85,
    "blue_minus_red_above": 75,
    "blue_least": 120,
    "red_minus_green_most": 15,
    "seed": 11_11_26,
    # the detector check: this many recorded pictures per 30-degree bin of the travel direction in the picture
    "detector_pictures_per_bin": 300,
    # looking along the line: the travel direction within this angle of the picture's top
    "looking_along_deg": 30.0,
    # a forearm flip: joint 4 passes from beyond this angle on one side to beyond it on the other
    "flip_side_deg": 60.0,
    # a line's bend: the turn between two chords of this length, one chord length apart, per cm
    "bend_window_cm": 2.0,
    "step_cm": 100.0 * MODEL_MODE_PARAMETERS["step_m"],
    "figure_path": EVIDENCE_FOLDER / "04_physics.png"
}



def driver_joints(
        reach_cm: float,
        joint_4_deg: float,
        parameters: Dict
) -> Optional[List[float]]:
    """
    Return the joints the robot driver would hold with the tip ``reach_cm`` straight in front of the base and the
    forearm twisted by ``joint_4_deg`` (wrist and hand held, as model_mode.model_step), or None when out of reach.
    """

    x = reach_cm / 100.0
    target = numpy.array([x, 0.0, pushing_height(x, 0.0) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    joints = solve_with_wrist_fixed(target, math.radians(joint_4_deg), math.radians(parameters["wrist_held_deg"]), [0.0, -0.8, -0.6])
    if joints is None:
        return None
    joints[5] = math.radians(parameters["hand_held_deg"])
    return joints



def heading_and_lean_table(parameters: Dict) -> pandas.DataFrame:
    """
    Calculate, for every reach and forearm twist, how far the picture's top is turned from radial and how far the rod leans.

    Returns
    -------
    pandas.DataFrame
        One row per reach and twist: "reach_cm", "joint_4_deg", "is_reachable", "heading_from_radial_deg",
        "rod_lean_deg" and "is_lean_allowed" (within the driver's limit).
    """

    rows = []
    for reach_cm in parameters["reaches_cm"]:
        for joint_4_deg in parameters["joint_4_values_deg"]:
            joints = driver_joints(reach_cm, joint_4_deg, parameters)
            if joints is None:
                rows.append({"reach_cm": reach_cm, "joint_4_deg": joint_4_deg, "is_reachable": False,
                             "heading_from_radial_deg": numpy.nan, "rod_lean_deg": numpy.nan, "is_lean_allowed": False})
                continue
            lean_deg = magnet_tip_and_lean(joints)[1]
            heading_deg = float(wrapped_deg(math.degrees(forearm_camera_heading_rad(joints))))
            rows.append({"reach_cm": reach_cm, "joint_4_deg": joint_4_deg, "is_reachable": True,
                         "heading_from_radial_deg": round(heading_deg, 1), "rod_lean_deg": round(lean_deg, 1),
                         "is_lean_allowed": lean_deg <= parameters["rod_lean_most_deg"]})
    return pandas.DataFrame(rows)



def robot_parts_mask(parameters: Dict) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Find where the robot's own parts sit in the picture.

    The camera is fixed to the forearm, so the block, the gripper's jaws and the black box stay in place while the table
    moves; in the per-pixel median of many recorded pictures the table blurs to a light colour and the parts stay dark
    or blue. The grey rod and wrist between the jaws are neither, so every pixel below a part in its column is counted
    as hidden too.

    Returns
    -------
    tuple of numpy.ndarray
        The median picture (120 x 160 x 3, RGB) and the mask at 480 x 640 (True on the robot's parts).
    """

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
    mask = cv2.resize(small, (640, 480), interpolation = cv2.INTER_NEAREST) > 0
    return median.astype(numpy.uint8), mask



def view_by_direction(
        mask: numpy.ndarray,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Calculate what the camera sees for every travel direction in the picture.

    For each reach (forearm straight) and each direction, the puck touching the block on that side and a straight line
    running on from it are projected into the picture with the fitted camera.

    Returns
    -------
    pandas.DataFrame
        One row per reach and direction: where the puck's top centre lands ("puck_u", "puck_v"), "is_puck_in_view"
        (inside the picture and off the robot's parts) and "line_in_view_cm" (how much of the line between
        "line_ahead_cm" of the tip is inside the picture and off the robot's parts).
    """

    rows = []
    distances_cm = numpy.arange(parameters["line_ahead_cm"][0], parameters["line_ahead_cm"][1] + 1e-9, parameters["line_ahead_step_cm"])
    for reach_cm in parameters["view_reaches_cm"]:
        joints = driver_joints(reach_cm, 0.0, parameters)
        if joints is None:
            continue
        tip = magnet_tip_and_lean(joints)[0]
        heading = forearm_camera_heading_rad(joints)
        table_z = table_height(tip[0], tip[1])
        for direction_deg in numpy.arange(-180.0 + parameters["view_direction_step_deg"], 180.0 + 1e-9, parameters["view_direction_step_deg"]):
            angle = heading + math.radians(direction_deg)
            unit = numpy.array([math.cos(angle), math.sin(angle)])
            puck_xy = tip[:2] + parameters["puck_ahead_of_tip_cm"] / 100.0 * unit
            puck_pixel = project_points(numpy.array([[puck_xy[0], puck_xy[1], table_z + parameters["puck_height_cm"] / 100.0]]), joints)[0]
            line_points = numpy.array([[tip[0] + distance / 100.0 * unit[0], tip[1] + distance / 100.0 * unit[1], table_z] for distance in distances_cm])
            line_pixels = project_points(line_points, joints)

            def is_seen(pixel: numpy.ndarray) -> bool:
                if not numpy.all(numpy.isfinite(pixel)):
                    return False
                u, v = int(round(pixel[0])), int(round(pixel[1]))
                return 0 <= u < 640 and 0 <= v < 480 and not mask[v, u]

            rows.append({
                "reach_cm": reach_cm, "direction_in_picture_deg": float(direction_deg),
                "puck_u": round(float(puck_pixel[0]), 0), "puck_v": round(float(puck_pixel[1]), 0),
                "is_puck_in_view": is_seen(puck_pixel),
                "line_in_view_cm": round(parameters["line_ahead_step_cm"] * sum(is_seen(pixel) for pixel in line_pixels), 1)
            })
    return pandas.DataFrame(rows)



def detector_check(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Measure on recorded pictures how often the puck is found, per travel direction in the picture.

    A sample of rows with a three-step path direction is drawn per 30-degree bin; each picture goes through the puck
    detector of the recording tool.

    Returns
    -------
    pandas.DataFrame
        One row per bin: "pictures", "found_share", "whole_share" (found and neither partial nor possibly hidden),
        and the puck's median place relative to the tip on the table ("puck_ahead_cm" along the travel direction of
        the bin, "puck_from_tip_cm"), plus the puck's measured width in cm over the whole detections.
    """

    detector = PuckDetector()
    randomness_generator = numpy.random.default_rng(parameters["seed"])
    rows = frames[frames["path_table_deg"].notna()].copy()
    rows["bin"] = numpy.round(rows["path_picture_deg"] / 30.0) * 30.0
    rows.loc[rows["bin"] == -180.0, "bin"] = 180.0
    results = []
    for bin_centre, part in rows.groupby("bin"):
        chosen = part.iloc[randomness_generator.permutation(len(part))[:parameters["detector_pictures_per_bin"]]]
        found, whole, distances, aheads, widths = 0, 0, [], [], []
        for _, row in chosen.iterrows():
            picture = cv2.imread(str(Path(parameters["raw_folder"]) / row["frame_path"]))
            if picture is None:
                continue
            sighting = detector.detect(picture)
            if sighting is None:
                continue
            found += 1
            joints = [row[f"joint_{number}"] for number in range(1, 7)]
            height = table_height(row["tip_x_m"], row["tip_y_m"]) + PUCK_TRACKING_PARAMETERS["puck_point_height_above_table_m"]
            place = pixels_to_plane(numpy.array([[sighting["centre_u"], sighting["centre_v"]]]), joints, height)[0]
            if numpy.all(numpy.isfinite(place)):
                offset = 100.0 * (place - numpy.array([row["tip_x_m"], row["tip_y_m"]]))
                travel = math.radians(row["path_table_deg"])
                distances.append(float(numpy.hypot(*offset)))
                aheads.append(float(offset[0] * math.cos(travel) + offset[1] * math.sin(travel)))
            is_whole = not (sighting["is_partial"] or sighting["is_possibly_occluded"] or sighting["is_touching_picture_edge"])
            whole += int(is_whole)
            left, top, width, box_height = sighting["bounding_box"]
            if is_whole and abs(width - box_height) <= 6:
                top_height = table_height(row["tip_x_m"], row["tip_y_m"]) + PUCK_PARAMETERS["height_m"]
                edges = pixels_to_plane(numpy.array([[left, sighting["centre_v"]], [left + width, sighting["centre_v"]]], dtype = float), joints, top_height)
                if numpy.all(numpy.isfinite(edges)):
                    widths.append(100.0 * float(numpy.hypot(*(edges[1] - edges[0]))))
        results.append({
            "direction_in_picture_deg": bin_centre, "pictures": len(chosen), "found_share": round(found / len(chosen), 3),
            "whole_share": round(whole / len(chosen), 3),
            "puck_from_tip_cm": round(float(numpy.median(distances)), 2) if distances else numpy.nan,
            "puck_ahead_cm": round(float(numpy.median(aheads)), 2) if aheads else numpy.nan,
            "puck_width_cm": round(float(numpy.median(widths)), 2) if widths else numpy.nan, "width_pictures": len(widths)
        })
    return pandas.DataFrame(results)



def forearm_flips(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Find the forearm flips of the demonstrations: joint 4 passing from beyond "flip_side_deg" on one side to beyond it
    on the other.

    Returns
    -------
    pandas.DataFrame
        One row per flip: plan, sheet, the rows it took (from the last picture beyond the first side to the first
        beyond the other), the forearm degrees turned, the tip's travel during it, how many of its rows were twists in
        place (tip step under "smallest_step_cm"), and the tip's mean reach.
    """

    side = parameters["flip_side_deg"]
    flips = []
    for episode_id, group in frames.groupby("episode_id", sort = False):
        joint_4 = group["joint_4_deg"].to_numpy()
        state = numpy.where(joint_4 >= side, 1, numpy.where(joint_4 <= -side, -1, 0))
        last_side, last_index = 0, None
        for index, value in enumerate(state):
            if value == 0:
                continue
            if last_side != 0 and value != last_side:
                window = group.iloc[last_index:index]
                flips.append({
                    "plan": group["plan_number"].iloc[0], "sheet": group["sheet_number"].iloc[0], "from_side": int(last_side),
                    "rows": index - last_index, "degrees": round(float(abs(joint_4[index] - joint_4[last_index])), 0),
                    "tip_travel_cm": round(float(window["step_cm"].sum()), 1),
                    "rows_in_place": int((window["step_cm"] < parameters["smallest_step_cm"]).sum()),
                    "reach_cm": round(float(window["reach_cm"].mean()), 1),
                    "travel_from_radial_deg": round(orientation_common.circular_mean_deg(window["step_from_radial_deg"]), 0)
                })
            last_side, last_index = value, index
    return pandas.DataFrame(flips)



def line_bends(parameters: Dict) -> pandas.DataFrame:
    """
    Measure how fast each line bends, from the traced course sheets and the practice tracks' drawings.

    Returns
    -------
    pandas.DataFrame
        One row per shape: its length, the sharpest bend (degrees per cm and the radius that goes with it) and the
        centimetres of line that bend faster than the driver can twist while it moves ("twist_step_most_deg" per
        "step_cm").
    """

    shapes = {f"sheet_{number}": sheet["line"] for number, sheet in course_sheet_lines().items()}
    shapes.update({name: track["line"] for name, track in practice_track_lines(parameters).items()})
    most_rate = parameters["twist_step_most_deg"] / parameters["step_cm"]
    window = int(round(parameters["bend_window_cm"] / 0.1))
    rows = []
    for name, line in shapes.items():
        chords = line[window:] - line[:-window]
        direction = numpy.unwrap(numpy.arctan2(chords[:, 1], chords[:, 0]))
        rate = numpy.degrees(numpy.abs(direction[window:] - direction[:-window])) / parameters["bend_window_cm"]
        rows.append({"shape": name, "line_cm": round(float(arc_lengths(line)[-1]), 1), "sharpest_bend_deg_per_cm": round(float(rate.max()), 1),
                     "smallest_radius_cm": round(57.3 / max(float(rate.max()), 1e-6), 1),
                     "cm_bending_faster_than_the_driver_twists": round(0.1 * float((rate > most_rate).sum()), 1),
                     "cm_bending_over_20_deg_per_cm": round(0.1 * float((rate > 2 * most_rate).sum()), 1)})
    return pandas.DataFrame(rows)



def draw_figure(
        table: pandas.DataFrame,
        frames: pandas.DataFrame,
        median: numpy.ndarray,
        mask: numpy.ndarray,
        view: pandas.DataFrame,
        detector: pandas.DataFrame,
        parameters: Dict
) -> None:
    """Draw the four physics panels."""

    figure, axes = pyplot.subplots(2, 2, figsize = (16, 12))

    axis = axes[0, 0]
    sample = frames.iloc[::7]
    axis.scatter(sample["joint_4_deg"], sample["heading_from_radial_deg"], s = 2, c = sample["reach_cm"], cmap = "viridis", alpha = 0.5)
    for reach_cm, style in [(15.0, "-"), (25.0, "--"), (35.0, "-."), (45.0, ":")]:
        part = table[(table["reach_cm"] == reach_cm) & table["is_reachable"]]
        axis.plot(part["joint_4_deg"], part["heading_from_radial_deg"], style, color = "red", linewidth = 1.5, label = f"arm model, reach {reach_cm:.0f} cm")
    axis.axvline(parameters["joint_4_most_deg"], color = "0.5", linewidth = 0.8)
    axis.axvline(-parameters["joint_4_most_deg"], color = "0.5", linewidth = 0.8)
    axis.set_xlabel("forearm twist, joint 4 (degrees)")
    axis.set_ylabel("picture's top, from radial (degrees)")
    axis.set_title("How far the camera can be turned from the arm's direction\n(dots: demonstrations, coloured by reach; lines: arm model, wrist held)", fontsize = 10)
    axis.legend(fontsize = 8)
    axis.grid(alpha = 0.3)

    axis = axes[0, 1]
    picture = cv2.resize(median, (640, 480), interpolation = cv2.INTER_LINEAR).copy()
    picture[mask] = (0.5 * picture[mask] + 0.5 * numpy.array([255, 0, 0])).astype(numpy.uint8)
    axis.imshow(picture)
    part = view[view["reach_cm"] == 25.0]
    for _, row in part.iterrows():
        colour = "lime" if row["is_puck_in_view"] else "black"
        if 0 <= row["puck_u"] < 640 and 0 <= row["puck_v"] < 480:
            axis.plot(row["puck_u"], row["puck_v"], "o", color = colour, markersize = 7, markeredgecolor = "white")
            axis.text(row["puck_u"], row["puck_v"] - 9, f"{row['direction_in_picture_deg']:.0f}", fontsize = 6, ha = "center", color = "white")
    axis.set_title("Median of 1,500 recorded pictures; red: the robot's own parts;\ndots: where the puck sits for each travel direction in the picture (reach 25 cm)", fontsize = 10)
    axis.set_xlim(0, 640)
    axis.set_ylim(480, 0)

    axis = axes[1, 0]
    for reach_cm in parameters["view_reaches_cm"]:
        part = view[view["reach_cm"] == reach_cm]
        axis.plot(part["direction_in_picture_deg"], part["line_in_view_cm"], "-o", markersize = 3, label = f"line in view, reach {reach_cm:.0f} cm")
    axis.set_xlabel("travel direction in the picture (degrees; 0 = to its top, 180 = to its bottom)")
    axis.set_ylabel("centimetres of a straight line in view, 3 to 15 cm ahead of the tip")
    second = axis.twinx()
    second.plot(detector["direction_in_picture_deg"], 100.0 * detector["found_share"], "s--", color = "black", label = "puck found by the detector (%)")
    second.plot(detector["direction_in_picture_deg"], 100.0 * detector["whole_share"], "^:", color = "0.4", label = "puck found whole (%)")
    second.set_ylim(0, 105)
    second.set_ylabel("recorded pictures (%)")
    axis.set_title("What the camera sees ahead, per travel direction in the picture", fontsize = 10)
    axis.legend(fontsize = 7, loc = "lower left")
    second.legend(fontsize = 7, loc = "lower right")
    axis.grid(alpha = 0.3)

    axis = axes[1, 1]
    limit = table[(table["joint_4_deg"] == parameters["joint_4_most_deg"]) & table["is_reachable"]].copy()
    limit["most_turn"] = limit["heading_from_radial_deg"].abs()
    reaches = limit["reach_cm"].to_numpy()
    along = parameters["looking_along_deg"]
    axis.fill_between(reaches, 0, limit["most_turn"], color = "tab:green", alpha = 0.5, label = "camera can point straight along the line")
    axis.fill_between(reaches, limit["most_turn"], numpy.minimum(limit["most_turn"] + along, 180), color = "tab:olive", alpha = 0.5,
                      label = f"line within {along:.0f} degrees of the picture's top at full twist")
    axis.fill_between(reaches, numpy.minimum(limit["most_turn"] + along, 180), 180, color = "tab:orange", alpha = 0.5,
                      label = "sideways in the picture even at full twist")
    moving = frames[(frames["step_cm"] >= parameters["smallest_step_cm"]) & frames["path_table_deg"].notna()]
    sample = moving.iloc[::5]
    is_along = sample["path_picture_deg"].abs() <= along
    axis.scatter(sample.loc[is_along, "reach_cm"], sample.loc[is_along, "path_from_radial_deg"].abs(), s = 2, color = "black", alpha = 0.4, label = "demonstrated, looking along")
    axis.scatter(sample.loc[~is_along, "reach_cm"], sample.loc[~is_along, "path_from_radial_deg"].abs(), s = 2, color = "red", alpha = 0.4, label = "demonstrated, not looking along")
    axis.set_xlabel("tip reach from the base axis (cm)")
    axis.set_ylabel("line direction from radial, either side (degrees; 180 = back to the base)")
    axis.set_title("Which line directions the camera can look along (joint 4 within 117 degrees)", fontsize = 10)
    axis.legend(fontsize = 7, loc = "lower right")
    axis.set_ylim(0, 180)

    figure.tight_layout()
    figure.savefig(parameters["figure_path"], dpi = 80)
    pyplot.close(figure)



def main(parameters: Optional[Dict] = None) -> None:
    """Print the four parts and save the tables and the figure."""

    settings = orientation_common.merged_parameters(PHYSICS_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 500)
    pandas.set_option("display.max_columns", 40)
    frames = demonstration_frames(settings)

    print("=== A. Picture's top from radial and rod lean, by reach and forearm twist (calculated: arm model, wrist held at "
          f"{settings['wrist_held_deg']} degrees as the driver holds it) ===")
    table = heading_and_lean_table(settings)
    table.to_csv(EVIDENCE_FOLDER / "04_heading_and_lean.csv", index = False)
    print("Picture's top from radial (degrees); rows: reach in cm, columns: joint 4:")
    print(table.pivot(index = "reach_cm", columns = "joint_4_deg", values = "heading_from_radial_deg").round(0).to_string())
    print("\nRod lean from vertical (degrees); the driver refuses a step above "
          f"{settings['rod_lean_most_deg']:.0f}:")
    print(table.pivot(index = "reach_cm", columns = "joint_4_deg", values = "rod_lean_deg").round(0).to_string())
    most = table[table["joint_4_deg"] == settings["joint_4_most_deg"]].set_index("reach_cm")["heading_from_radial_deg"].abs()
    print(f"\nLargest turn of the picture's top from radial (joint 4 at {settings['joint_4_most_deg']:.0f} degrees), by reach:")
    print("  " + ", ".join(f"{reach:g} cm: {turn:.0f}" for reach, turn in most.items()))
    straight = table[(table["joint_4_deg"] == 0.0) & table["is_reachable"]]
    refused = straight[~straight["is_lean_allowed"]]["reach_cm"]
    print("  With the forearm straight the rod's lean passes the driver's limit from reach "
          f"{refused.min() if len(refused) else float('nan'):g} cm on (so the usable reach ends just before it).")
    slope = numpy.polyfit(frames["joint_4_deg"], frames["heading_from_radial_deg"], 1)[0]
    far_turned = frames[frames["joint_4_deg"].abs() >= 100]
    print(f"Measured in the demonstrations: slope {slope:+.3f} degrees of picture turn per degree of joint 4; with |joint 4| of 100 or more "
          f"({len(far_turned)} pictures) the picture's top is a median {far_turned['heading_from_radial_deg'].abs().median():.1f} degrees from radial "
          f"(largest {frames['heading_from_radial_deg'].abs().max():.1f}).")
    reach_bins = pandas.cut(far_turned["reach_cm"], [12, 20, 27.5, 35, 42.5, 49.5])
    print("  the same by reach (median |picture top from radial|, pictures):")
    print(far_turned.groupby(reach_bins, observed = True)["heading_from_radial_deg"].agg(lambda values: round(float(values.abs().median()), 1)).to_string())

    print("\n=== B. What the camera sees round the tip (calculated: fitted camera, forearm straight, mask of the robot's parts) ===")
    median, mask = robot_parts_mask(settings)
    print(f"Robot's own parts cover {100.0 * mask.mean():.1f} % of the picture; rows 320 to 480 of 480: "
          f"{100.0 * mask[320:].mean():.1f} %; nothing above row {int(numpy.nonzero(mask.any(axis = 1))[0].min())}.")
    view = view_by_direction(mask, settings)
    view.to_csv(EVIDENCE_FOLDER / "04_view_by_direction.csv", index = False)
    print("Is the puck in view (1) or behind the robot's parts or outside the picture (0); rows: travel direction in the picture:")
    print(view.pivot(index = "direction_in_picture_deg", columns = "reach_cm", values = "is_puck_in_view").astype(int).to_string())
    print("\nCentimetres of a straight line in view between 3 and 15 cm ahead of the tip (12 cm = all of it):")
    print(view.pivot(index = "direction_in_picture_deg", columns = "reach_cm", values = "line_in_view_cm").to_string())

    print("\n=== C. Measured on recorded pictures: the puck detector per travel direction in the picture ===")
    detector = detector_check(frames, settings)
    print(detector.to_string(index = False))
    print(f"Puck width over the whole detections: median {detector['puck_width_cm'].median():.2f} cm "
          f"(config.py PUCK_PARAMETERS: {100.0 * PUCK_PARAMETERS['diameter_m']:.1f} cm; the brief says 4 cm).")

    print("\n=== D. Which line directions can be followed how ===")
    along = settings["looking_along_deg"]
    print(f"A line that runs at an angle b from radial at reach r can be met by the picture's top only if |b| is at most the largest turn above. "
          f"Beyond it the line lies at least |b| minus that turn off the picture's top:")
    for reach in [15.0, 25.0, 35.0, 45.0]:
        turn = float(most.loc[reach])
        print(f"  reach {reach:.0f} cm: straight along the line up to {turn:.0f} degrees from radial; within {along:.0f} degrees of the picture's top up to "
              f"{min(turn + along, 180):.0f}; straight back to the base (180) lies {180 - turn:.0f} degrees off the picture's top at best")
    moving = frames[(frames["step_cm"] >= settings["smallest_step_cm"]) & frames["path_table_deg"].notna()].copy()
    moving["from_radial_bin"] = (numpy.round(moving["path_from_radial_deg"].abs() / 30.0) * 30.0).astype(int)
    print("\nWhat the demonstrators did, by line direction from radial (either side; three-step paths, measured):")
    summary = moving.groupby("from_radial_bin").agg(
        pictures = ("frame_index", "size"),
        joint_4_median_abs = ("joint_4_deg", lambda values: round(float(values.abs().median()), 0)),
        off_picture_top_median = ("path_picture_deg", lambda values: round(float(values.abs().median()), 0)),
        looking_along_share = ("path_picture_deg", lambda values: round(float((values.abs() <= along).mean()), 2)),
        over_90_off_top_share = ("path_picture_deg", lambda values: round(float((values.abs() > 90).mean()), 2)),
        over_120_off_top_share = ("path_picture_deg", lambda values: round(float((values.abs() > 120).mean()), 2)))
    print(summary.to_string())

    flips = forearm_flips(frames, settings)
    print(f"\nForearm flips in the demonstrations (joint 4 from beyond {settings['flip_side_deg']:.0f} degrees on one side to beyond it on the other): "
          f"{len(flips)} in {flips['plan'].nunique() if len(flips) else 0} episodes")
    if len(flips):
        print(flips.groupby("sheet").agg(flips = ("rows", "size"), episodes = ("plan", "nunique"), rows_median = ("rows", "median"),
                                         degrees_median = ("degrees", "median"), tip_travel_cm_median = ("tip_travel_cm", "median"),
                                         rows_in_place_median = ("rows_in_place", "median"), reach_cm_median = ("reach_cm", "median")).to_string())
        print(f"  all flips: median {flips['rows'].median():.0f} rows, {flips['degrees'].median():.0f} degrees, {flips['tip_travel_cm'].median():.1f} cm of tip travel, "
              f"{flips['rows_in_place'].median():.0f} rows in place; degrees per cm of travel: median {(flips['degrees'] / flips['tip_travel_cm'].clip(lower = 0.1)).median():.1f}")
    full_flip = 2 * settings["joint_4_most_deg"]
    print(f"  The driver turns at most {settings['twist_step_most_deg']:.0f} degrees per step and always moves 1 cm: a full flip of {full_flip:.0f} degrees "
          f"takes at least {math.ceil(full_flip / settings['twist_step_most_deg'])} steps, so at least {math.ceil(full_flip / settings['twist_step_most_deg'])} cm of travel.")

    print("\n=== E. How fast the lines bend (measured on the traced course sheets; calculated from the practice tracks' drawings) ===")
    bends = line_bends(settings)
    bends.to_csv(EVIDENCE_FOLDER / "04_line_bends.csv", index = False)
    print(bends.to_string(index = False))
    twisting = frames[(frames["forearm_change_deg"].abs() >= 6.0) & (frames["action"] != "stop")]
    print(f"The driver turns the forearm at most {settings['twist_step_most_deg']:.0f} degrees per {settings['step_cm']:.0f}-cm step. In the demonstrations the "
          f"{len(twisting)} rows that twist by 6 degrees or more move the tip a median {twisting['step_cm'].median():.2f} cm; "
          f"{100.0 * (twisting['step_cm'] < 0.5).mean():.0f} % of them move it under 0.5 cm (the twist is made nearly in place).")

    draw_figure(table, frames, median, mask, view, detector, settings)
    print(f"\nSaved {settings['figure_path'].name}, 04_heading_and_lean.csv, 04_view_by_direction.csv, 04_line_bends.csv")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding = "utf-8")
    main()
