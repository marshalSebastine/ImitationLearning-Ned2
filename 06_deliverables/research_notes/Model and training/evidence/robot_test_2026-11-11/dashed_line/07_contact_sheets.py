"""
Step 7 of the dashed-line analysis: the pictures behind the numbers, as contact sheets.

    07_run_8_all_steps.jpg            every picture of robot run 8 (sheet 06) with what model E asked;
    07_run_7_walk_round.jpg           steps 28 to 62 of robot run 7 (sheet 05): the walk round the puck the model did;
    07_demonstration_06-03.jpg        demonstration 06-03 (end A, score 3), every picture with the demonstrated step:
                                      the walk round the puck is pictures 35 to 56.
The green arrow is the step in the picture's frame (its length the asked or demonstrated length); the label gives
the step number, its direction (a, degrees, + = left), length (L, cm), forearm twist (t, degrees), the forearm's
angle (j4), and the stop chance (s) or the wrist angle (j5).

Run: D:/AI_Project/.venv/Scripts/python.exe 07_contact_sheets.py
"""

import math

import cv2
import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, read_run

SHEET_PARAMETERS = {
    "tile_size": [256, 192],
    "arrow_px_per_cm": 60.0,
    # the magnet tip's place in the picture with the wrist at 13.5 degrees (the camera model's projection)
    "tip_pixel": [313, 372],
    "jpeg_quality": 80
}



def tile(
        picture: numpy.ndarray,
        forward_cm: float,
        left_cm: float,
        label: str
) -> numpy.ndarray:
    """Return a small labelled picture with the step drawn as an arrow from the tip's place."""

    settings = SHEET_PARAMETERS
    start = tuple(settings["tip_pixel"])
    end = (int(start[0] - settings["arrow_px_per_cm"] * left_cm), int(start[1] - settings["arrow_px_per_cm"] * forward_cm))
    cv2.arrowedLine(picture, start, end, (0, 220, 0), 4)
    small = cv2.resize(picture, tuple(settings["tile_size"]), interpolation = cv2.INTER_AREA)
    cv2.rectangle(small, (0, 0), (settings["tile_size"][0], 15), (0, 0, 0), -1)
    cv2.putText(small, label, (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)
    return small



def written_sheet(
        tiles: list,
        columns: int,
        name: str
) -> None:
    """Write tiles as one picture with ``columns`` per row."""

    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[first:first + columns]) for first in range(0, len(tiles), columns)]
    cv2.imwrite(str(EVIDENCE_FOLDER / name), numpy.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, SHEET_PARAMETERS["jpeg_quality"]])



def main() -> None:

    for key, name, first, last, columns in (("run_8_folder", "07_run_8_all_steps.jpg", 0, 40, 7), ("run_7_folder", "07_run_7_walk_round.jpg", 28, 62, 7)):
        steps = read_run(ANALYSIS_PARAMETERS[key])
        tiles = []
        for _, step in steps.iloc[first:last + 1].iterrows():
            angle = math.degrees(math.atan2(step["model_tip_left_cm"], step["model_tip_forward_cm"]))
            label = (f"{int(step['step'])} a{angle:+.0f} L{step['asked_length_cm']:.2f} t{step['model_twist_deg']:+.1f} j4 {step['joint_4_deg']:.0f} "
                     f"s{step['model_stop_probability']:.2f}")
            tiles.append(tile(cv2.imread(step["frame_file"]), step["model_tip_forward_cm"], step["model_tip_left_cm"], label))
        written_sheet(tiles, columns, name)

    frames = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "frames.csv", low_memory = False)
    part = frames[(frames["plan_number"] == "06-03") & (frames["score"] > 0)].sort_values("row_in_episode")
    tiles = []
    for _, row in part.iterrows():
        forward = 100.0 * (row["tip_forward_m"] if pandas.notna(row["tip_forward_m"]) else 0.0)
        left = 100.0 * (row["tip_left_m"] if pandas.notna(row["tip_left_m"]) else 0.0)
        twist = row["forearm_change_deg"] if pandas.notna(row["forearm_change_deg"]) else 0.0
        label = (f"{int(row['row_in_episode'])} a{math.degrees(math.atan2(left, forward)):+.0f} L{math.hypot(forward, left):.2f} t{twist:+.1f} "
                 f"j4 {math.degrees(row['joint_4']):.0f} j5 {math.degrees(row['joint_5']):.0f}")
        tiles.append(tile(cv2.imread(str(ANALYSIS_PARAMETERS["raw_folder"] / row["frame_path"])), forward, left, label))
    written_sheet(tiles, 10, "07_demonstration_06-03.jpg")
    print("written: 07_run_8_all_steps.jpg, 07_run_7_walk_round.jpg, 07_demonstration_06-03.jpg")



if __name__ == "__main__":
    main()
