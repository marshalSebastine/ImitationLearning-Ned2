"""
Check 1: does the camera correction found for the robot test of 11.11.26 (camera_shift.json, period "robot_test": the
picture shifted 9.9 px right and 2.7 px up, rolled 1.26 degrees) still hold for the visit of 12.11.26?

The same test as camera_shift.py of 11.11: every picture of a run votes where the line lies on the table; if the camera
model is right, pictures taken with the forearm turned differently agree and the votes pile up on few cells ("sharpness",
larger is better). The sharpness is compared for no correction, the correction of 11.11 and a small grid of shifts
around it, on four runs of the visit with large forearm turns.

Run from this folder:  python s01_camera_check.py
"""

import numpy

import visit_common
import run_measures


CHECK_PARAMETERS = {
    "runs": ["R02", "R05", "R22", "R29"],
    "every": 2,
    "shifts_u_px": [0.0, 4.0, 8.0, 9.9, 12.0, 16.0],
    "shifts_v_px": [-8.0, -2.7, 4.0],
    "rolls_deg": [0.0, 1.26, 2.5],
    "map_mm_per_px": 1.0
}



def main() -> None:

    folders = visit_common.run_folders()
    loaded = []
    for label in CHECK_PARAMETERS["runs"]:
        steps = run_measures.robot_run_steps(folders[label])
        x_range, y_range = run_measures.map_ranges(steps)
        loaded.append({"label": label, "views": run_measures.loaded_views(steps, CHECK_PARAMETERS["every"]),
                       "x_range": x_range, "y_range": y_range})
    measure_parameters = {"map_mm_per_px": CHECK_PARAMETERS["map_mm_per_px"]}

    def sharpness(shift_u: float, shift_v: float, roll: float) -> numpy.ndarray:
        camera_parameters = run_measures.corrected_camera_parameters([shift_u, shift_v], roll)
        return numpy.array([run_measures.map_sharpness(run["views"], run["x_range"], run["y_range"], camera_parameters, measure_parameters)
                            for run in loaded])

    at_zero = sharpness(0.0, 0.0, 0.0)
    print("runs:", [run["label"] for run in loaded], "pictures:", [len(run["views"]) for run in loaded])
    print("sharpness without correction:", numpy.round(at_zero, 3).tolist())
    results = []
    for roll in CHECK_PARAMETERS["rolls_deg"]:
        for shift_u in CHECK_PARAMETERS["shifts_u_px"]:
            for shift_v in CHECK_PARAMETERS["shifts_v_px"]:
                values = sharpness(shift_u, shift_v, roll)
                gain = float(numpy.mean(numpy.log(values / at_zero)))
                results.append((gain, shift_u, shift_v, roll, values))
    results.sort(key = lambda row: -row[0])
    print("\nbest ten (gain = mean natural logarithm of sharpness over no correction):")
    for gain, shift_u, shift_v, roll, values in results[:10]:
        print(f"  shift ({shift_u:5.1f}, {shift_v:5.1f}) px, roll {roll:4.2f} deg: gain {gain:.4f}  per run {numpy.round(values, 2).tolist()}")
    of_11_11 = [row for row in results if row[1] == 9.9 and row[2] == -2.7 and row[3] == 1.26][0]
    print(f"\nthe correction of 11.11 (9.9, -2.7) px, roll 1.26 deg: gain {of_11_11[0]:.4f}, rank {results.index(of_11_11) + 1} of {len(results)}")



if __name__ == "__main__":
    main()
