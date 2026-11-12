"""
Contact sheets: chosen pictures of a robot run (or of a demonstration) side by side, each with the direction of travel
in the picture drawn as a green arrow from the bottom centre and the twist and step length written on it.

Run from this folder, for example:
    D:/AI_Project/.venv/Scripts/python.exe contact_sheet.py run 5 0 3 6 9 11 13 15 17 19 22 26 30
    D:/AI_Project/.venv/Scripts/python.exe contact_sheet.py plan 03-01 0 60 3
(run: the run number and the picture numbers; plan: the plan number, then first row, last row and every how many)
Output: sheet_run_NN.png or sheet_plan_XX-YY.png in this folder.
"""

import sys
from pathlib import Path
from typing import List

import cv2
import numpy

from common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, demonstration_frames, robot_run_steps

CONTACT_SHEET_PARAMETERS = {
    "tile_size": [320, 240],
    "tiles_per_row": 4,
    "arrow_px": 60
}



def labelled_tile(
        picture: numpy.ndarray,
        angle_deg: float,
        text: str
) -> numpy.ndarray:
    """Return the picture shrunk to a tile with the travel direction drawn and ``text`` on a dark band."""

    width, height = CONTACT_SHEET_PARAMETERS["tile_size"]
    tile = cv2.resize(picture, (width, height))
    if numpy.isfinite(angle_deg):
        angle = numpy.radians(angle_deg)
        start = (width // 2, height - 10)
        end = (int(start[0] - CONTACT_SHEET_PARAMETERS["arrow_px"] * numpy.sin(angle)), int(start[1] - CONTACT_SHEET_PARAMETERS["arrow_px"] * numpy.cos(angle)))
        cv2.arrowedLine(tile, start, end, (0, 220, 0), 2)
    cv2.rectangle(tile, (0, 0), (width, 18), (0, 0, 0), -1)
    cv2.putText(tile, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return tile



def stacked(tiles: List[numpy.ndarray]) -> numpy.ndarray:
    """Return the tiles laid out in rows."""

    per_row = CONTACT_SHEET_PARAMETERS["tiles_per_row"]
    while len(tiles) % per_row:
        tiles.append(numpy.zeros_like(tiles[0]))
    return numpy.vstack([numpy.hstack(tiles[position:position + per_row]) for position in range(0, len(tiles), per_row)])



def main() -> None:

    kind = sys.argv[1]
    if kind == "run":
        number = int(sys.argv[2])
        run = [run for run in ANALYSIS_PARAMETERS["robot_runs"] if run["number"] == number][0]
        steps = robot_run_steps(run)
        folder = Path(ANALYSIS_PARAMETERS["model_runs_folder"]) / run["folder"] / "frames"
        tiles = []
        for picture_number in [int(value) for value in sys.argv[3:]]:
            if picture_number >= len(steps):
                continue
            row = steps.iloc[picture_number]
            picture = cv2.imread(str(folder / f"{picture_number:03d}.jpg"))
            tiles.append(labelled_tile(picture, row["model_angle_deg"],
                                       f"run {number} step {picture_number} dir {row['model_angle_deg']:+.0f} twist {row['model_twist_deg']:+.1f} len {row['asked_cm']:.2f} j4 {row['joint_4_deg']:+.0f}"))
        cv2.imwrite(str(EVIDENCE_FOLDER / f"sheet_run_{number:02d}.png"), stacked(tiles))
    else:
        plan = sys.argv[2]
        first, last, every = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        frames = demonstration_frames()
        rows = frames[frames["plan_number"] == plan]
        rows = rows[(rows["row_in_episode"] >= first) & (rows["row_in_episode"] <= last)].iloc[::every]
        tiles = []
        for _, row in rows.iterrows():
            picture = cv2.imread(str(Path(ANALYSIS_PARAMETERS["raw_folder"]) / row["frame_path"]))
            tiles.append(labelled_tile(picture, row["move_angle_deg"] if row["step_cm"] >= 0.05 else float("nan"),
                                       f"{plan} row {int(row['row_in_episode'])} dir {row['move_angle_deg']:+.0f} twist {row['forearm_change_deg']:+.1f} len {row['step_cm']:.2f} j4 {row['joint_4_deg']:+.0f}"))
        cv2.imwrite(str(EVIDENCE_FOLDER / f"sheet_plan_{plan}.png"), stacked(tiles))



if __name__ == "__main__":
    main()
