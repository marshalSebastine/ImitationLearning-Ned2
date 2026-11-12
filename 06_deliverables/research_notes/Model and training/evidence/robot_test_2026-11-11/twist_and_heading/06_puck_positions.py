"""
Where the red puck is in every picture, on the table: all demonstration pictures and all pictures of the ten robot runs.

The team's own puck tracker (02_code/puck_tracking.py, a colour detector and the fitted forearm camera model; no robot)
gives the puck centre in the picture and on the table (metres, base frame). With the magnet tip from the arm model
this gives, per picture, the PUSH direction (from the tip to the puck) on the table and in the picture. The table is
only a cache for 07_aim_of_the_camera.py.

Output: puck_positions_demonstrations.csv, puck_positions_runs.csv, 06_puck_positions.out.txt
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 06_puck_positions.py
(about 6 minutes on one thread: it reads all 18,032 demonstration pictures)
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy
import pandas

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, Report, demonstration_frames, robot_run_steps

from puck_tracking import PuckTracker
from side_view_test import magnet_tip_and_lean

PUCK_POSITION_PARAMETERS = {
    "opencv_threads": 1
}



def seen_row(
        tracker: PuckTracker,
        picture_path: Path,
        joints: List[float]
) -> Dict:
    """Return the puck's place in one picture (pixel and table, NaN when not seen) and the magnet tip on the table."""

    tip = magnet_tip_and_lean(list(joints))[0]
    row = {"tip_x_cm": 100.0 * float(tip[0]), "tip_y_cm": 100.0 * float(tip[1]), "puck_u": numpy.nan, "puck_v": numpy.nan,
           "puck_x_cm": numpy.nan, "puck_y_cm": numpy.nan, "confidence": numpy.nan, "is_partial": numpy.nan, "is_position_rough": numpy.nan}
    picture = cv2.imread(str(picture_path))
    if picture is None:
        return row
    seen = tracker.see_puck(picture, joints)
    if seen is None:
        return row
    row.update({"puck_u": seen["pixel"][0], "puck_v": seen["pixel"][1], "confidence": seen["confidence"], "is_partial": int(bool(seen["is_partial"])),
                "is_position_rough": int(bool(seen["is_position_rough"]))})
    if seen["table_xy"] is not None:
        row.update({"puck_x_cm": 100.0 * float(seen["table_xy"][0]), "puck_y_cm": 100.0 * float(seen["table_xy"][1])})
    return row



def main(parameters: Optional[Dict] = None) -> None:
    """Find the puck in every picture of the robot runs and of the demonstrations and save where it is."""

    settings = dict(PUCK_POSITION_PARAMETERS)
    settings.update(parameters or {})
    cv2.setNumThreads(settings["opencv_threads"])
    report = Report("06_puck_positions.out.txt")
    tracker = PuckTracker()
    report.say(f"puck tracker calibrated for the camera mount in use: {tracker.is_calibrated()}")
    joint_columns = [f"joint_{number}" for number in range(1, 7)]

    started = time.perf_counter()
    rows = []
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        steps = robot_run_steps(run)
        folder = Path(ANALYSIS_PARAMETERS["model_runs_folder"]) / run["folder"]
        for _, step in steps.iterrows():
            row = {"run": run["number"], "step": int(step["step"])}
            row.update(seen_row(tracker, folder / step["frame"], step[joint_columns].to_numpy(dtype = float).tolist()))
            rows.append(row)
    runs_table = pandas.DataFrame(rows)
    runs_table.to_csv(EVIDENCE_FOLDER / "puck_positions_runs.csv", index = False)
    report.say(f"robot runs: {len(runs_table)} pictures, puck seen in {runs_table['puck_u'].notna().mean():.3f}, placed on the table in {runs_table['puck_x_cm'].notna().mean():.3f} ({time.perf_counter() - started:.0f} s)")

    started = time.perf_counter()
    frames = demonstration_frames()
    rows = []
    for position in range(len(frames)):
        frame = frames.iloc[position]
        row = {"frame_index": int(frame["frame_index"])}
        row.update(seen_row(tracker, Path(ANALYSIS_PARAMETERS["raw_folder"]) / frame["frame_path"], frame[joint_columns].to_numpy(dtype = float).tolist()))
        rows.append(row)
    demonstrations_table = pandas.DataFrame(rows)
    demonstrations_table.to_csv(EVIDENCE_FOLDER / "puck_positions_demonstrations.csv", index = False)
    report.say(f"demonstrations: {len(demonstrations_table)} pictures, puck seen in {demonstrations_table['puck_u'].notna().mean():.3f}, placed on the table in "
               f"{demonstrations_table['puck_x_cm'].notna().mean():.3f}, partly hidden in {demonstrations_table['is_partial'].mean():.3f} of those seen ({time.perf_counter() - started:.0f} s)")
    report.save()



if __name__ == "__main__":
    main()
