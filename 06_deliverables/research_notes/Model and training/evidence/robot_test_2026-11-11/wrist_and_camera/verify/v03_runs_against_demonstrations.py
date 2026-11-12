"""
Verifier, check 3: the robot runs' logged joints against the demonstrations at the same places. Measured from the runs'
steps.csv and from frames.csv; the "family" of demonstrations (same sheet placement and same course as the run's plan)
is found from the PATHS themselves, not from the start_end labels: a start-mark episode of the same sheet belongs to the
family when its tip path over the first 20 cm of travel stays within 1.5 cm (mean) of the plan episode's path.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe v03_runs_against_demonstrations.py
"""

import math
import re

import numpy
import pandas

INDEX_FOLDER = "D:/AI_Project/03_data/processed/v01_training_index"
RUNS_FOLDER = "D:/AI_Project/03_data/model_runs"
CORRECTED_ENDS = "D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv"
HELD_WRIST_DEG = 13.5
RUNS = [
    (1, "2026-11-11_172021_check_no_puck", "path_01", "01-01"),
    (2, "2026-11-11_172421_E_sheet01_run1", "path_01", "01-01"),
    (4, "2026-11-11_173036_E_sheet02_run2", "path_02", "02-01"),
    (5, "2026-11-11_173549_E_sheet03_run1", "path_03", "03-01"),
    (6, "2026-11-11_174108_E_sheet04_run1", "path_04", "04-01"),
    (7, "2026-11-11_174448_E_sheet05_run1", "path_05", "05-01"),
    (8, "2026-11-11_174936_E_sheet06_run1", "path_06", "06-01"),
    (9, "2026-11-11_175203_E_sheet03_run2_no_hoodie", "path_03", "03-01"),
    (10, "2026-11-11_175544_B_team5_sheet03", "path_03", "03-01"),
]
NEAR_M = 0.02
FAMILY_TRAVEL_M = 0.20
FAMILY_MEAN_GAP_M = 0.015

pandas.set_option("display.width", 260)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)


def path_by_travel(x: numpy.ndarray, y: numpy.ndarray, most_m: float, spacing_m: float = 0.005) -> numpy.ndarray:
    """The tip path resampled every ``spacing_m`` of travelled distance, up to ``most_m``; shorter paths are cut there."""

    travel = numpy.concatenate([[0.0], numpy.cumsum(numpy.hypot(numpy.diff(x), numpy.diff(y)))])
    wanted = numpy.arange(0.0, min(most_m, travel[-1]) + 1e-9, spacing_m)
    return numpy.column_stack([numpy.interp(wanted, travel, x), numpy.interp(wanted, travel, y)])


frames = pandas.read_csv(f"{INDEX_FOLDER}/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "tip_x_m", "tip_y_m", "joint_4", "joint_5", "joint_6",
                                    "rod_from_vertical_deg", "step_fraction", "picture_heading_deg", "frame_path"])
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
for name in ("joint_4", "joint_5", "joint_6"):
    frames[name + "_deg"] = numpy.degrees(frames[name])
frames["plan"] = frames["episode_id"].str.extract(r"plan_(\d\d-\d\d)")[0]
frames["is_start_mark"] = frames["sheet_orientation"].str.contains("START MARK", na = False)
corrected = pandas.read_csv(CORRECTED_ENDS).set_index("episode_id")

summary_rows = []
for run_number, folder, sheet, plan in RUNS:
    steps = pandas.read_csv(f"{RUNS_FOLDER}/{folder}/steps.csv")
    for name in ("joint_4", "joint_5", "joint_6"):
        steps[name + "_deg"] = numpy.degrees(steps[name])
    sheet_frames = frames[(frames["sheet"] == sheet) & frames["is_start_mark"]]
    plan_episodes = sheet_frames.loc[sheet_frames["plan"] == plan, "episode_id"].unique()
    if len(plan_episodes) == 0:
        print(f"run {run_number}: no episode of plan {plan}")
        continue
    reference = sheet_frames[sheet_frames["episode_id"] == plan_episodes[0]]
    reference_path = path_by_travel(reference["tip_x_m"].to_numpy(), reference["tip_y_m"].to_numpy(), FAMILY_TRAVEL_M)
    family = []
    for episode_id, group in sheet_frames.groupby("episode_id"):
        path = path_by_travel(group["tip_x_m"].to_numpy(), group["tip_y_m"].to_numpy(), FAMILY_TRAVEL_M)
        count = min(len(path), len(reference_path))
        gap = float(numpy.hypot(*(path[:count] - reference_path[:count]).T).mean())
        if gap <= FAMILY_MEAN_GAP_M and count >= 0.7 * len(reference_path):
            family.append(episode_id)
    family_frames = sheet_frames[sheet_frames["episode_id"].isin(family)]
    ends = corrected.loc[family, "corrected_end"].value_counts().to_dict()
    kinds = corrected.loc[family, "start_kind"].value_counts().to_dict()
    print(f"\n=== run {run_number} ({folder}), sheet {sheet}, plan {plan}: family of {len(family)} start-mark episodes by path "
          f"(corrected ends {ends}; kinds {kinds}); robot wrist {steps['joint_5_deg'].median():.2f}, hand {steps['joint_6_deg'].median():.2f} ===")

    demo_xy = family_frames[["tip_x_m", "tip_y_m"]].to_numpy()
    table = []
    for _, step in steps.iterrows():
        distance = numpy.hypot(demo_xy[:, 0] - step["tip_x"], demo_xy[:, 1] - step["tip_y"])
        near = family_frames[distance <= NEAR_M]
        # the run's first pass only: demonstration rows of the same part of the course (travel fraction within the
        # first 60 % of the episode for the first 20 pictures), so a later pass over the same spot does not mix in
        row = {"picture": int(step["step"]), "reach_cm": 100 * math.hypot(step["tip_x"], step["tip_y"]), "robot_twist": step["joint_4_deg"],
               "robot_wrist": step["joint_5_deg"], "robot_lean": step.get("lean_deg", numpy.nan), "neighbours": len(near),
               "nearest_cm": 100 * float(distance.min())}
        if len(near) >= 5:
            wrist = near["joint_5_deg"]
            twist = near["joint_4_deg"]
            row.update({"near_wrist_p10": wrist.quantile(0.1), "near_wrist_median": wrist.median(), "near_wrist_p90": wrist.quantile(0.9),
                        "near_wrist_within_5.7": float(((wrist - HELD_WRIST_DEG).abs() <= 5.7).mean()),
                        "near_twist_p10": twist.quantile(0.1), "near_twist_median": twist.median(), "near_twist_p90": twist.quantile(0.9),
                        "near_lean_median": near["rod_from_vertical_deg"].median(),
                        "twist_gap": step["joint_4_deg"] - twist.median(),
                        "outside_band": float(step["joint_4_deg"] < twist.quantile(0.1) or step["joint_4_deg"] > twist.quantile(0.9))})
        table.append(row)
    table = pandas.DataFrame(table)
    print(table.head(20).round(1).to_string(index = False))
    with_near = table.dropna(subset = ["near_wrist_median"]) if "near_wrist_median" in table else table.iloc[0:0]
    first = with_near[with_near["picture"] < 13]
    summary_rows.append({
        "run": run_number, "sheet": sheet, "pictures": len(table), "family": len(family),
        "with_5_neighbours": len(with_near),
        "neighbour_wrist_median_within_5.7_share": float(((with_near["near_wrist_median"] - HELD_WRIST_DEG).abs() <= 5.7).mean()) if len(with_near) else numpy.nan,
        "first13_with_neighbours": len(first),
        "first13_wrist_median_least": first["near_wrist_median"].min() if len(first) else numpy.nan,
        "first13_wrist_median_most": first["near_wrist_median"].max() if len(first) else numpy.nan,
        "first13_rows_within_5.7_least": first["near_wrist_within_5.7"].min() if len(first) else numpy.nan,
        "first13_largest_twist_gap": first["twist_gap"].abs().max() if len(first) else numpy.nan,
        "first13_outside_band": int(first["outside_band"].sum()) if len(first) else 0,
        "whole_run_wrist_band_least_p10": with_near["near_wrist_p10"].min() if len(with_near) else numpy.nan,
        "whole_run_wrist_band_most_p90": with_near["near_wrist_p90"].max() if len(with_near) else numpy.nan})

print("\n=== summary ===")
summary = pandas.DataFrame(summary_rows)
print(summary.round(2).to_string(index = False))
total_with = int(summary["with_5_neighbours"].sum())
print("\nall listed runs: pictures with 5 or more neighbours: %d of %d" % (total_with, int(summary["pictures"].sum())))
summary.to_csv("v03_runs_against_demonstrations_summary.csv", index = False)

# the start of every run against the demonstrated start pose
start_mark_first = frames[frames["is_start_mark"]].groupby("episode_id").first()
start_x, start_y = start_mark_first["tip_x_m"].median(), start_mark_first["tip_y_m"].median()
print("\ndemonstrated start (149 episodes): tip %.2f, %.2f cm; twist %.2f to %.2f; picture heading %.2f to %.2f" % (
    100 * start_x, 100 * start_y, start_mark_first["joint_4_deg"].min(), start_mark_first["joint_4_deg"].max(),
    start_mark_first["picture_heading_deg"].min(), start_mark_first["picture_heading_deg"].max()))
all_runs = RUNS + [(3, "2026-11-11_172743_E_sheet02_run1", "path_02", "02-01")]
for run_number, folder, sheet, plan in sorted(all_runs):
    first_step = pandas.read_csv(f"{RUNS_FOLDER}/{folder}/steps.csv").iloc[0]
    print("  run %2d: start tip %.1f, %.1f cm; gap to the demonstrated start %.1f cm; twist %+.1f; wrist %.2f; hand %.2f; picture heading %+.1f" % (
        run_number, 100 * first_step["tip_x"], 100 * first_step["tip_y"], 100 * math.hypot(first_step["tip_x"] - start_x, first_step["tip_y"] - start_y),
        math.degrees(first_step["joint_4"]), math.degrees(first_step["joint_5"]), math.degrees(first_step["joint_6"]), first_step["picture_heading_deg"]))
