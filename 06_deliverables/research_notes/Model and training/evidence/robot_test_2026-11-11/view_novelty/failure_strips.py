"""
Enlarged strips of the pictures around the point where each failed run went wrong, to fix two steps per run by eye:
    first_wrong_decision    the first picture whose asked direction (green arrow) clearly leaves the line ahead;
    puck_off_line           the first picture in which the puck no longer touches the line.

The steps read off these strips are written down in FAILURE_STEPS in common.py, with the reason, and every other script
marks them in its plots. Nothing connects to the robot.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe failure_strips.py
Output: figures/failure_strip_run_NN.png
"""

import math
from typing import Dict, Optional

import cv2
import numpy

from common import EVIDENCE_FOLDER, OutputLog, run_picture_paths, run_steps, save_montage

FAILURE_STRIP_PARAMETERS = {
    # run number -> (first step, last step) shown
    "ranges": {3: (0, 11), 5: (0, 14), 6: (9, 26), 8: (0, 17), 9: (0, 14), 10: (0, 14), 1: (0, 11), 4: (18, 35)},
    "columns": 6,
    "tile_size": [320, 240],
    "arrow_length_px": 70
}



def failure_strip(
        run_number: int,
        parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Build the enlarged strip of one run.

    Parameters
    ----------
    run_number : int
        A run named in "ranges".
    parameters : dict, optional
        Overrides of ``FAILURE_STRIP_PARAMETERS``.

    Returns
    -------
    numpy.ndarray
        The strip as a BGR picture.
    """

    settings = dict(FAILURE_STRIP_PARAMETERS)
    settings.update(parameters or {})
    first_step, last_step = settings["ranges"][run_number]
    steps = run_steps(run_number)
    paths = run_picture_paths(run_number)
    width, height = settings["tile_size"]
    columns = settings["columns"]
    chosen = list(range(first_step, min(last_step, len(paths) - 1) + 1))
    rows = math.ceil(len(chosen) / columns)
    strip = numpy.full((rows * height, columns * width, 3), 255, dtype = numpy.uint8)

    for position, step in enumerate(chosen):
        tile = cv2.resize(cv2.imread(str(paths[step]), cv2.IMREAD_COLOR), (width, height), interpolation = cv2.INTER_AREA)
        row = steps.iloc[step]
        angle = math.radians(row["model_angle_deg"])
        start = (width // 2, height - 8)
        end = (int(start[0] - settings["arrow_length_px"] * math.sin(angle)), int(start[1] - settings["arrow_length_px"] * math.cos(angle)))
        cv2.arrowedLine(tile, start, end, (0, 220, 0), 2, tipLength = 0.25)
        asked_cm = math.hypot(row["model_tip_forward_cm"], row["model_tip_left_cm"])
        text = f"step {int(row['step'])}  asked {asked_cm:.2f} cm at {row['model_angle_deg']:+.0f} deg  twist {row['model_twist_deg']:+.1f}"
        cv2.rectangle(tile, (0, 0), (width, 18), (0, 0, 0), -1)
        cv2.putText(tile, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        top, left = (position // columns) * height, (position % columns) * width
        strip[top:top + height, left:left + width] = tile
    return strip



def main() -> None:

    log = OutputLog(EVIDENCE_FOLDER / "failure_strips.out.txt")
    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    for run_number, (first_step, last_step) in FAILURE_STRIP_PARAMETERS["ranges"].items():
        path = figure_folder / f"failure_strip_run_{run_number:02d}.png"
        save_montage(path, failure_strip(run_number))
        log.say(f"run {run_number:2d}: steps {first_step} to {last_step} -> {path.name}")
    log.save()



if __name__ == "__main__":
    main()
