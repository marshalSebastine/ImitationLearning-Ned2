"""
Check 1: does the fitted forearm camera (config.FOREARM_CAMERA_PARAMETERS, fitted 6.11.26 on the recordings of 4 and
5.11.26) still put table points in the right place of the picture, in the recordings of 6 to 9.11.26 and in the robot
test of 11.11.26?

The data audit found that the robot's own hand sits about 15 px further right in the pictures from 8.11.26 19:33 on and
could not tell whether the camera or the hand had moved. Here the far field decides: every picture of a run votes where
the line lies on the table; if the camera model is right, pictures taken with the forearm turned differently agree, and
if the whole picture is shifted they disagree, the more the further the forearm turned. The picture shift (added to the
camera's principal point) that makes the pictures agree best is searched for each camera period, on runs with large
forearm turns, then a small roll of the camera about its viewing direction, and both are written to camera_shift.json
for the other scripts.

Run from this folder:    python camera_shift.py robot_test | before_move | after_move
Nothing connects to the robot; pictures and logged joints are only read.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy
import pandas

import run_measures

from config import MODEL_MODE_PARAMETERS, TRAINING_DATA_PARAMETERS


CAMERA_SHIFT_PARAMETERS = {
    # runs with large forearm turns, a few per camera period; every so many pictures are used
    "runs": {
        "robot_test": ["2026-11-11_173036_E_sheet02_run2", "2026-11-11_174448_E_sheet05_run1",
                       "2026-11-11_174108_E_sheet04_run1", "2026-11-11_173549_E_sheet03_run1"],
        "before_move": ["path_04_meander_episode_015_plan_04-08_try_1", "path_03_two_curve_episode_007_plan_03-07_try_1",
                        "path_03_two_curve_episode_004_plan_03-04_try_1", "path_02_arc_episode_022_plan_02-14_try_1"],
        "after_move": ["path_05_alpha_loop_episode_020_plan_05-15_try_1", "path_06_alpha_loop_dashed_episode_002_plan_06-02_try_1",
                       "path_04_meander_episode_039_plan_04-31_try_1", "path_05_alpha_loop_episode_018_plan_05-13_try_1"]
    },
    "every": {"robot_test": 3, "before_move": 5, "after_move": 5},
    # the search: a coarse grid of shifts (pixels right, pixels down), then a finer one around its best
    "coarse_shifts_u_px": [-16, -8, 0, 8, 16, 24],
    "coarse_shifts_v_px": [-24, -16, -8, 0, 8],
    "fine_step_px": 4,
    "rolls_deg": [-0.75, 0.0, 0.75, 1.5, 2.25],
    "map_mm_per_px": 1.0
}



def period_views(period: str) -> List[Dict]:
    """The kept pictures of the period's runs: a list of {"name", "views", "x_range", "y_range"}."""

    loaded = []
    if period == "robot_test":
        for name in CAMERA_SHIFT_PARAMETERS["runs"][period]:
            steps = run_measures.robot_run_steps(Path(MODEL_MODE_PARAMETERS["run_folder"]) / name)
            x_range, y_range = run_measures.map_ranges(steps)
            loaded.append({"name": name, "views": run_measures.loaded_views(steps, CAMERA_SHIFT_PARAMETERS["every"][period]),
                           "x_range": x_range, "y_range": y_range})
        return loaded
    frames = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False)
    for name in CAMERA_SHIFT_PARAMETERS["runs"][period]:
        episode_ids = [episode_id for episode_id in frames["episode_id"].unique() if episode_id.endswith(name)]
        steps = run_measures.demonstration_steps(frames, episode_ids[0])
        x_range, y_range = run_measures.map_ranges(steps)
        loaded.append({"name": name, "views": run_measures.loaded_views(steps, CAMERA_SHIFT_PARAMETERS["every"][period]),
                       "x_range": x_range, "y_range": y_range})
    return loaded



def sharpness_of_shift(
        loaded: List[Dict],
        shift_u_px: float,
        shift_v_px: float,
        roll_deg: float = 0.0
) -> List[float]:
    """The line map's sharpness of every run with the camera's principal point moved by the shift and the camera rolled."""

    camera_parameters = run_measures.corrected_camera_parameters([shift_u_px, shift_v_px], roll_deg)
    measure_parameters = {"map_mm_per_px": CAMERA_SHIFT_PARAMETERS["map_mm_per_px"]}
    return [run_measures.map_sharpness(run["views"], run["x_range"], run["y_range"], camera_parameters, measure_parameters)
            for run in loaded]



def main(period: str) -> None:

    loaded = period_views(period)
    print(f"period {period}: " + ", ".join(f"{run['name']} ({len(run['views'])} pictures)" for run in loaded))
    at_zero = numpy.array(sharpness_of_shift(loaded, 0.0, 0.0))
    print("sharpness without a shift, per run:", numpy.round(at_zero, 2).tolist())

    # each run's sharpness is taken relative to its own value without a shift, so that every run counts the same
    def gain(shift_u_px: float, shift_v_px: float) -> float:
        return float(numpy.mean(numpy.log(numpy.array(sharpness_of_shift(loaded, shift_u_px, shift_v_px)) / at_zero)))

    gains = {}
    print("\ngain in sharpness (mean natural logarithm of the ratio to no shift); rows: shift right, columns: shift down")
    print("        " + "".join(f"{shift_v:8d}" for shift_v in CAMERA_SHIFT_PARAMETERS["coarse_shifts_v_px"]))
    for shift_u in CAMERA_SHIFT_PARAMETERS["coarse_shifts_u_px"]:
        for shift_v in CAMERA_SHIFT_PARAMETERS["coarse_shifts_v_px"]:
            gains[(shift_u, shift_v)] = gain(shift_u, shift_v)
        print(f"{shift_u:8d}" + "".join(f"{gains[(shift_u, shift_v)]:8.4f}" for shift_v in CAMERA_SHIFT_PARAMETERS["coarse_shifts_v_px"]), flush = True)
    coarse_best = max(gains, key = gains.get)

    step = CAMERA_SHIFT_PARAMETERS["fine_step_px"]
    fine_u = [coarse_best[0] - step, coarse_best[0], coarse_best[0] + step]
    fine_v = [coarse_best[1] - step, coarse_best[1], coarse_best[1] + step]
    print(f"\nfiner search around {coarse_best}")
    print("        " + "".join(f"{shift_v:8d}" for shift_v in fine_v))
    for shift_u in fine_u:
        for shift_v in fine_v:
            if (shift_u, shift_v) not in gains:
                gains[(shift_u, shift_v)] = gain(shift_u, shift_v)
        print(f"{shift_u:8d}" + "".join(f"{gains[(shift_u, shift_v)]:8.4f}" for shift_v in fine_v), flush = True)
    best = max(gains, key = gains.get)

    # the top of a parabola through the best value and its two neighbours, along each axis
    def refined(axis: int) -> float:
        lower = tuple(best[index] - (step if index == axis else 0) for index in range(2))
        upper = tuple(best[index] + (step if index == axis else 0) for index in range(2))
        if lower not in gains:
            gains[lower] = gain(*lower)
        if upper not in gains:
            gains[upper] = gain(*upper)
        curvature = gains[lower] - 2.0 * gains[best] + gains[upper]
        if curvature >= 0.0:
            return float(best[axis])
        return float(best[axis] + 0.5 * step * (gains[lower] - gains[upper]) / curvature)

    shift = [round(refined(0), 1), round(refined(1), 1)]
    print(f"\nbest grid shift {best} with gain {gains[best]:.4f}; refined shift {shift} px (right, down)")

    # second stage: a roll of the camera about its viewing direction (or, which looks the same from here, an offset of
    # the forearm joint's zero), searched with the shift held, then the shift refined once more with the roll held
    def gain_with(shift_u_px: float, shift_v_px: float, roll_deg: float) -> float:
        return float(numpy.mean(numpy.log(numpy.array(sharpness_of_shift(loaded, shift_u_px, shift_v_px, roll_deg)) / at_zero)))

    rolls = CAMERA_SHIFT_PARAMETERS["rolls_deg"]
    roll_gains = [gain_with(shift[0], shift[1], roll) for roll in rolls]
    print("gain by roll (degrees) at the refined shift:", {roll: round(value, 4) for roll, value in zip(rolls, roll_gains)})
    top = int(numpy.argmax(roll_gains))
    roll = float(rolls[top])
    if 0 < top < len(rolls) - 1:
        curvature = roll_gains[top - 1] - 2.0 * roll_gains[top] + roll_gains[top + 1]
        if curvature < 0.0:
            roll = float(rolls[top] + 0.5 * (rolls[top + 1] - rolls[top]) * (roll_gains[top - 1] - roll_gains[top + 1]) / curvature)
    roll = round(roll, 2)
    centre_gain = gain_with(shift[0], shift[1], roll)
    for axis in range(2):
        lower = [shift[0] - (step if axis == 0 else 0), shift[1] - (step if axis == 1 else 0)]
        upper = [shift[0] + (step if axis == 0 else 0), shift[1] + (step if axis == 1 else 0)]
        lower_gain, upper_gain = gain_with(lower[0], lower[1], roll), gain_with(upper[0], upper[1], roll)
        curvature = lower_gain - 2.0 * centre_gain + upper_gain
        if curvature < 0.0:
            shift[axis] = round(float(shift[axis] + 0.5 * step * (lower_gain - upper_gain) / curvature), 1)
    at_best = sharpness_of_shift(loaded, shift[0], shift[1], roll)
    print(f"roll {roll} degrees; shift refined with the roll held: {shift} px (right, down)")
    print("sharpness with the shift and the roll, per run:", numpy.round(at_best, 2).tolist())
    print(f"mean gain against no shift: {100.0 * (numpy.exp(numpy.mean(numpy.log(numpy.array(at_best) / at_zero))) - 1.0):.1f} %")
    # how flat the top is: every shift whose gain is within 0.005 of the best
    near_best = sorted(key for key, value in gains.items() if value >= gains[best] - 0.005)
    print("shifts within 0.005 of the best gain:", near_best)

    result_path = run_measures.EVIDENCE_FOLDER / run_measures.RUN_MEASURE_PARAMETERS["camera_shift_file"]
    results = json.loads(result_path.read_text(encoding = "utf-8")) if result_path.exists() else {}
    results[period] = {"shift_px": shift, "roll_deg": roll, "best_grid_shift_px": list(best), "gain": gains[best],
                       "sharpness_without_shift": at_zero.round(3).tolist(), "sharpness_with_shift": numpy.round(at_best, 3).tolist(),
                       "runs": [run["name"] for run in loaded], "near_best_shifts_px": [list(key) for key in near_best]}
    result_path.write_text(json.dumps(results, indent = 1), encoding = "utf-8")
    print(f"written to {result_path.name}")



if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "robot_test")
