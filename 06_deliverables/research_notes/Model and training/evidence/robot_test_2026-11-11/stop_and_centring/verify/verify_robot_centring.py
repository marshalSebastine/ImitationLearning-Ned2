"""
Verifier's check 5 (11.11.26): the sideways offset per step of the robot's runs 1, 2, 4 and 7 with my own
one-picture measure (centring_tools.py), beside the analyst's offsets (../robot_runs_steps.csv: "offset_mm" from the
votes of the last 12 pictures, "single_offset_mm" from one picture).
Writes verify_robot_centring.csv. Reads pictures and steps.csv only.
"""

from pathlib import Path

import cv2
import numpy
import pandas

import centring_tools
import my_measure

cv2.setNumThreads(2)
HERE = Path(__file__).resolve().parent
RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RUNS = {1: "2026-11-11_172021_check_no_puck", 2: "2026-11-11_172421_E_sheet01_run1",
        4: "2026-11-11_173036_E_sheet02_run2", 7: "2026-11-11_174448_E_sheet05_run1"}
CAMERA = my_measure.shifted_camera((9.9, -2.7), 1.26)

analyst = pandas.read_csv(HERE.parent / "robot_runs_steps.csv")
rows = []
for number, folder in RUNS.items():
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    tips = steps[["tip_x", "tip_y"]].to_numpy(dtype = float)
    for index, row in steps.iterrows():
        joints = [row[f"joint_{joint}"] for joint in range(1, 7)]
        travel = tips[min(index + 2, len(tips) - 1)] - tips[max(index - 2, 0)]
        travel = travel / max(float(numpy.linalg.norm(travel)), 1e-9)
        found = centring_tools.same_picture_offset(cv2.imread(str(RUNS_FOLDER / folder / row["frame"])), joints, CAMERA, travel)
        entry = {"run": number, "step": int(row["step"])}
        if found is not None:
            entry.update({key: found[key] for key in ("offset_mm", "extrapolated_mm", "nearest_ink_mm", "seen_mm", "bend_per_m", "spread_mm", "residual_px")})
            entry.update({"puck_x": found["puck_xy"][0], "puck_y": found["puck_xy"][1]})
        rows.append(entry)
mine = pandas.DataFrame(rows).merge(analyst[["run", "step", "puck_x", "puck_y", "offset_mm", "single_offset_mm", "is_clean", "is_single_clean"]],
                                    on = ["run", "step"], how = "left", suffixes = ("", "_analyst"))
mine.to_csv(HERE / "verify_robot_centring.csv", index = False, float_format = "%.3f")

pandas.set_option("display.width", 250)
for number in RUNS:
    run = mine[mine["run"] == number]
    puck_gap = 1000.0 * numpy.hypot(run["puck_x"] - run["puck_x_analyst"], run["puck_y"] - run["puck_y_analyst"])
    print(f"run {number}: pictures {len(run)}; my puck against the analyst's puck on the table: median {puck_gap.median():.1f} mm apart, largest {puck_gap.max():.1f} mm")
    # my one-picture offset is trusted where the line is seen close to the puck or is followed back only a little
    for label, is_good in (("line followed back at most 10 mm", run["extrapolated_mm"] <= 10.0),
                           ("line followed back 10 to 35 mm, bend under 8 per m, 30 mm or more of line seen", (run["extrapolated_mm"] > 10.0) & (run["extrapolated_mm"] <= 35.0) & (run["bend_per_m"] < 8.0) & (run["seen_mm"] >= 30.0))):
        part = run[is_good & run["offset_mm_analyst"].notna()]
        if len(part) == 0:
            print(f"  {label}: no pictures")
            continue
        difference = part["offset_mm"] - part["offset_mm_analyst"]
        print(f"  {label}: {len(part)} pictures; mine minus the analyst's 12-picture offset: median {difference.median():+.1f} mm, median size {difference.abs().median():.1f} mm,"
              f" 90th percentile of the size {difference.abs().quantile(0.9):.1f} mm, largest {difference.abs().max():.1f} mm")
    usable = run[(run["extrapolated_mm"] <= 10.0) | ((run["extrapolated_mm"] <= 35.0) & (run["bend_per_m"] < 8.0) & (run["seen_mm"] >= 30.0))]
    sizes = usable["offset_mm"].abs()
    theirs = run["offset_mm_analyst"].abs().dropna()
    print(f"  my offsets on the {len(usable)} usable pictures: median {sizes.median():.1f} mm, within 7 mm {100 * (sizes <= 7).mean():.0f} %, over 12 mm {100 * (sizes > 12).mean():.0f} %, largest {sizes.max():.1f} mm")
    print(f"  the analyst's on {len(theirs)} pictures:        median {theirs.median():.1f} mm, within 7 mm {100 * (theirs <= 7).mean():.0f} %, over 12 mm {100 * (theirs > 12).mean():.0f} %, largest {theirs.max():.1f} mm")
print()
print("the last steps of run 2 and the middle of run 4 (my offset; how far the line was followed back; the analyst's offset):")
for number, first, last in ((2, 14, 24), (4, 16, 32), (1, 12, 25)):
    part = mine[(mine["run"] == number) & mine["step"].between(first, last)]
    print(part[["run", "step", "offset_mm", "extrapolated_mm", "nearest_ink_mm", "bend_per_m", "offset_mm_analyst", "single_offset_mm"]].round(1).to_string(index = False))
