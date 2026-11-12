"""
Verifier's own look at the logged pictures of a robot run: contact sheets with the step number and the asked step drawn in.

Read-only: reads frames/NNN.jpg and steps.csv of 03_data/model_runs, writes JPEG sheets beside this script. Nothing
connects to the robot.

Use: python v01_run_pictures.py <run folder name> <first step> <last step> [<every>]
"""

import math
import sys
from pathlib import Path

import cv2
import numpy
import pandas

SHEET_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "columns": 5,
    "tile_width": 384,
    "tile_height": 288,
    "arrow_pixels_per_cm": 60.0
}


def contact_sheet(run_name: str, first_step: int, last_step: int, every: int = 1) -> Path:
    """
    Write one contact sheet of a run's pictures.

    Parameters
    ----------
    run_name : str
        Folder name inside the model runs folder.
    first_step, last_step : int
        The steps shown (inclusive).
    every : int, optional
        Show every n-th step.

    Returns
    -------
    pathlib.Path
        The written file.
    """

    settings = dict(SHEET_PARAMETERS)
    run_folder = settings["model_runs_folder"] / run_name
    steps = pandas.read_csv(run_folder / "steps.csv")
    chosen = steps[(steps["step"] >= first_step) & (steps["step"] <= last_step)].iloc[::every]
    tiles = []
    for _, row in chosen.iterrows():
        picture = cv2.imread(str(run_folder / row["frame"]), cv2.IMREAD_COLOR)
        tile = cv2.resize(picture, (settings["tile_width"], settings["tile_height"]), interpolation = cv2.INTER_AREA)
        centre = (settings["tile_width"] // 2, settings["tile_height"] // 2)
        forward_cm, left_cm = float(row["model_tip_forward_cm"]), float(row["model_tip_left_cm"])
        length_cm = math.hypot(forward_cm, left_cm)
        # the asked step, true to length (yellow) and as a unit direction (cyan, what the fixed 1 cm step drove)
        end_true = (int(centre[0] - settings["arrow_pixels_per_cm"] * left_cm), int(centre[1] - settings["arrow_pixels_per_cm"] * forward_cm))
        if length_cm > 1e-6:
            end_unit = (int(centre[0] - settings["arrow_pixels_per_cm"] * left_cm / length_cm),
                        int(centre[1] - settings["arrow_pixels_per_cm"] * forward_cm / length_cm))
            cv2.arrowedLine(tile, centre, end_unit, (255, 255, 0), 2, tipLength = 0.25)
        cv2.arrowedLine(tile, centre, end_true, (0, 255, 255), 3, tipLength = 0.3)
        text = f"{int(row['step'])}: {length_cm:.2f} cm {float(row['model_angle_deg']):+.0f} deg tw {float(row['model_twist_deg']):+.1f} j4 {math.degrees(float(row['joint_4'])):+.0f}"
        cv2.rectangle(tile, (0, 0), (settings["tile_width"], 22), (0, 0, 0), -1)
        cv2.putText(tile, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    columns = settings["columns"]
    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[start:start + columns]) for start in range(0, len(tiles), columns)]
    sheet = numpy.vstack(rows)
    output_path = Path(__file__).resolve().parent / f"v01_{run_name[11:]}_steps_{first_step:03d}_{last_step:03d}.jpg"
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return output_path


if __name__ == "__main__":
    name = sys.argv[1]
    first, last = int(sys.argv[2]), int(sys.argv[3])
    step_every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    print(contact_sheet(name, first, last, step_every))
