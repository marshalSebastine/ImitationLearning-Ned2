"""
Montages of raw robot pictures, so the verifier can read the failure steps off the pictures without the analyst's
figures. Each cell shows one logged JPEG with the step, what model asked (length, direction, twist) and the stop chance.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe look_at_runs.py <run> <first step> <last step> [stride]
Output: figures/own_run_<run>_steps_<first>_<last>.png
"""

import sys

import cv2
import numpy

from verify_common import VERIFY_FOLDER, labelled_montage, read_picture, run_log, run_picture_path



def main() -> None:

    run_number, first, last = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    columns = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    log = run_log(run_number)
    pictures, labels = [], []
    for step in range(first, min(last, len(log) - 1) + 1, stride):
        row = log.iloc[step]
        length = float(numpy.hypot(row["model_tip_forward_cm"], row["model_tip_left_cm"]))
        pictures.append(read_picture(run_picture_path(run_number, step)))
        labels.append(f"run {run_number} step {step}: {length:.2f} cm at {row['model_angle_deg']:+.0f}, twist {row['model_twist_deg']:+.1f}, stop {row['model_stop_probability']:.2f}")
    montage = labelled_montage(pictures, labels, columns = columns, cell_width = 480)
    path = VERIFY_FOLDER / "figures" / f"own_run_{run_number:02d}_steps_{first:03d}_{last:03d}.png"
    cv2.imencode(".png", montage)[1].tofile(str(path))
    print(path, montage.shape)



if __name__ == "__main__":
    main()
