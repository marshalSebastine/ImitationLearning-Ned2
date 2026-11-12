"""
Step 0 of the dashed-line analysis: how far can table positions taken from the pictures be trusted?

Every measurement of this analysis (where the line lies, where the puck is beside it) comes from the fitted forearm
camera model (config.FOREARM_CAMERA_PARAMETERS, fitted on 6.11.26 at the old desk). Two checks on the recordings used
here:
    1. the line drawn from above from the pictures with the forearm twisted less than 30 degrees and from those with
       it twisted more than 60 degrees should lie on the same place: the shift between the two drawings is measured;
    2. the reference line of sheet 06 (step 1), laid onto run 8 (step 2), is drawn into pictures of run 8 taken with
       the forearm twisted 61 to 102 degrees: the drawn dashes should cover the dashes the picture shows.
Check 2 needs reference_line_06.csv and registrations.csv (steps 1 and 2); check 1 runs alone.

Writes 00_camera_model_check.png and prints the shifts.

Run: D:/AI_Project/.venv/Scripts/python.exe 00_camera_model_check.py > 00_camera_model_check.out.txt
"""

import cv2
import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, JOINT_NAMES, read_run, top_view_of_line
from line_geometry import ink_share_of, moved_line

CHECK_PARAMETERS = {
    "low_twist_most_deg": 30.0,
    "high_twist_least_deg": 60.0,
    "map_cell_m": 0.002,
    "most_shift_cells": 8,
    "demonstration_plans": ["06-03", "06-06", "05-03", "05-06"],
    "overlay_steps": [14, 18, 22, 26, 28, 30, 32, 36]
}



def shift_between(
        first: dict,
        second: dict,
        most_cells: int
) -> tuple:
    """Return the (x, y) shift in cells that lays the second top view best onto the first, and the correlation there."""

    share_first, share_second = ink_share_of(first, 1), ink_share_of(second, 1)
    seen_both = (first["seen_votes"] >= 1) & (second["seen_votes"] >= 1)
    best = (-1.0, 0, 0, 0)
    for shift_x in range(-most_cells, most_cells + 1):
        for shift_y in range(-most_cells, most_cells + 1):
            moved = numpy.roll(numpy.roll(share_second, shift_x, axis = 0), shift_y, axis = 1)
            both = numpy.roll(numpy.roll(seen_both, shift_x, axis = 0), shift_y, axis = 1) & seen_both
            if both.sum() < 500:
                continue
            correlation = float(numpy.corrcoef(share_first[both], moved[both])[0, 1])
            if correlation > best[0]:
                best = (correlation, shift_x, shift_y, int(both.sum()))
    return best



def main() -> None:

    settings = CHECK_PARAMETERS
    frames = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "frames.csv", low_memory = False)
    recordings = []
    for name, key in (("robot run 7 (sheet 05)", "run_7_folder"), ("robot run 8 (sheet 06)", "run_8_folder")):
        steps = read_run(ANALYSIS_PARAMETERS[key])
        recordings.append((name, list(steps["frame_file"]), steps[JOINT_NAMES].to_numpy()))
    for plan in settings["demonstration_plans"]:
        part = frames[(frames["plan_number"] == plan) & (frames["score"] > 0)].sort_values("row_in_episode")
        recordings.append((f"demonstration {plan}", [str(ANALYSIS_PARAMETERS["raw_folder"] / value) for value in part["frame_path"]], part[JOINT_NAMES].to_numpy()))
    print("Check 1: the line drawn from pictures with little forearm twist against pictures with much twist")
    for name, pictures, joints in recordings:
        twist = numpy.abs(numpy.degrees(joints[:, 3]))
        low, high = numpy.flatnonzero(twist < settings["low_twist_most_deg"]), numpy.flatnonzero(twist > settings["high_twist_least_deg"])
        if len(low) < 4 or len(high) < 4:
            print(f"  {name}: too few pictures (little twist {len(low)}, much twist {len(high)})")
            continue
        view_low = top_view_of_line([pictures[position] for position in low], joints[low], {"map_cell_m": settings["map_cell_m"]})
        view_high = top_view_of_line([pictures[position] for position in high], joints[high], {"map_cell_m": settings["map_cell_m"]})
        correlation, shift_x, shift_y, cells = shift_between(view_low, view_high, settings["most_shift_cells"])
        millimetres = 1000.0 * settings["map_cell_m"] * numpy.hypot(shift_x, shift_y)
        print(f"  {name}: {len(low)} pictures under {settings['low_twist_most_deg']:.0f} deg, {len(high)} over {settings['high_twist_least_deg']:.0f} deg: "
              f"the two drawings fit best when shifted by ({shift_x}, {shift_y}) cells of 2 mm = {millimetres:.1f} mm (correlation {correlation:.2f} on {cells} cells)")

    # check 2: the placed reference line drawn into high-twist pictures of run 8
    from picture_edits import band_labels, runs_where
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv").set_index("recording")
    reference = pandas.read_csv(EVIDENCE_FOLDER / "reference_line_06.csv")
    points = reference[["x_cm", "y_cm"]].to_numpy() / 100.0
    placed = registrations.loc["run_8"]
    line = moved_line(points - points.mean(axis = 0), placed["angle_deg"], (placed["shift_x_cm"] / 100.0, placed["shift_y_cm"] / 100.0))
    is_ink_point = reference["is_ink"].to_numpy() > 0
    steps = read_run(ANALYSIS_PARAMETERS["run_8_folder"])
    tiles = []
    for step in settings["overlay_steps"]:
        picture = cv2.imread(steps["frame_file"][step])
        known = (band_labels(line, runs_where(is_ink_point), [float(steps.loc[step, name]) for name in JOINT_NAMES], 0.010) > 0).astype(numpy.uint8)
        outlines, _ = cv2.findContours(known, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(picture, outlines, -1, (0, 255, 255), 2)
        cv2.putText(picture, f"run 8, step {step}, forearm {steps['joint_4_deg'][step]:.0f} deg", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(picture)
    sheet = numpy.vstack([numpy.hstack(tiles[:4]), numpy.hstack(tiles[4:])])
    cv2.imwrite(str(EVIDENCE_FOLDER / "00_camera_model_check.png"), cv2.resize(sheet, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA))
    print("Check 2: 00_camera_model_check.png shows the placed dashes (yellow outlines) on pictures of run 8 with the forearm twisted 61 to 102 degrees")



if __name__ == "__main__":
    main()
