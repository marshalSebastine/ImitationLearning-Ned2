"""
One contact sheet per robot run of 11.11.26: every picture the model decided on, small, in step order, with the step
number, the model's direction (green arrow, as the driver's window drew it), the asked step length and twist.

The sheets are for looking at the runs: where the puck left the line, what was in view (a blue hoodie, the table's
edge), and how the view changed. Nothing connects to the robot; only the saved pictures and steps.csv are read.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe contact_sheets.py
Output: figures/contact_run_NN.png
"""

import math
from typing import Dict, Optional

import cv2
import numpy

from common import EVIDENCE_FOLDER, ROBOT_RUNS, OutputLog, run_picture_paths, run_steps, save_montage

CONTACT_SHEET_PARAMETERS = {
    "columns": 8,
    "tile_size": [240, 180],
    "arrow_length_px": 45
}



def contact_sheet(
        run_number: int,
        parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Build the contact sheet of one run.

    Parameters
    ----------
    run_number : int
        1 to 10, as in ``ROBOT_RUNS``.
    parameters : dict, optional
        Overrides of ``CONTACT_SHEET_PARAMETERS``.

    Returns
    -------
    numpy.ndarray
        The sheet as a BGR picture.
    """

    settings = dict(CONTACT_SHEET_PARAMETERS)
    settings.update(parameters or {})
    steps = run_steps(run_number)
    paths = run_picture_paths(run_number)
    width, height = settings["tile_size"]
    columns = settings["columns"]
    rows = math.ceil(len(paths) / columns)
    sheet = numpy.full((rows * height, columns * width, 3), 255, dtype = numpy.uint8)

    for position, path in enumerate(paths):
        picture = cv2.imread(str(path), cv2.IMREAD_COLOR)
        tile = cv2.resize(picture, (width, height), interpolation = cv2.INTER_AREA)
        row = steps.iloc[position]
        angle = math.radians(row["model_angle_deg"])
        start = (width // 2, height - 6)
        end = (int(start[0] - settings["arrow_length_px"] * math.sin(angle)), int(start[1] - settings["arrow_length_px"] * math.cos(angle)))
        cv2.arrowedLine(tile, start, end, (0, 220, 0), 2, tipLength = 0.3)
        asked_cm = math.hypot(row["model_tip_forward_cm"], row["model_tip_left_cm"])
        text = f"{int(row['step'])}  {asked_cm:.1f}cm  tw{row['model_twist_deg']:+.0f}  s{row['model_stop_probability']:.2f}"
        cv2.rectangle(tile, (0, 0), (width, 16), (0, 0, 0), -1)
        cv2.putText(tile, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        top, left = (position // columns) * height, (position % columns) * width
        sheet[top:top + height, left:left + width] = tile
    return sheet



def main() -> None:

    log = OutputLog(EVIDENCE_FOLDER / "contact_sheets.out.txt")
    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    for run_number, run in ROBOT_RUNS.items():
        sheet = contact_sheet(run_number)
        path = figure_folder / f"contact_run_{run_number:02d}.png"
        save_montage(path, sheet)
        log.say(f"run {run_number:2d} ({run['folder']}): {len(run_picture_paths(run_number))} pictures -> {path.name}, {sheet.shape[1]} x {sheet.shape[0]}")
    log.save()



if __name__ == "__main__":
    main()
