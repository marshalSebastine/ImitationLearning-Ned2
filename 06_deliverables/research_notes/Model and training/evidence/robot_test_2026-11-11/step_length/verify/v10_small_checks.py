"""
Verifier: small recomputations of numbers the analyst's answer quotes.

A. frames.csv as recorded: demonstrated step statistics, the 1.2 cm clamp's percentiles, the longest demonstration.
B. steps.csv: what the model asked BEFORE each failed run's off-the-line decision (the decisions are the analyst's,
   checked by eye on contact sheets), and the turn of joint 4 in run 6 before its loss.
C. Demonstrated twist per cm per sheet against what was done and asked on the robot.
D. The existing stall rule of 02_code/model_mode.py (30 pictures within 3 cm and under 15 degrees) on the ten logs,
   for the tips as they really went on 11.11.26.

Run from this folder:  python v10_small_checks.py > v10_small_checks.out.txt
"""

import os

import numpy
import pandas

INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = [
    "2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1",
    "2026-11-11_173036_E_sheet02_run2", "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1",
    "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie",
    "2026-11-11_175544_B_team5_sheet03"
]
RUN_SHEETS = ["path_01", "path_01", "path_02", "path_02", "path_03", "path_04", "path_05", "path_06", "path_03", "path_03"]
OFF_LINE = {3: 3, 5: 9, 6: 10, 8: 29, 9: 16, 10: 29}
pandas.set_option("display.width", 250)

frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "sheet", "plan_number", "action", "tip_forward_m", "tip_left_m",
                                                             "forearm_change_deg", "tip_x_m", "tip_y_m", "joint_4"])
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
frames["step_cm"] = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
frames["twist"] = frames["forearm_change_deg"].fillna(0.0)
steps_rows = frames[frames["action"] != "stop"]
s = steps_rows["step_cm"]
print("A. frames.csv as recorded, rows that are not stop rows:", len(steps_rows))
print(f"   step: median {s.median():.3f}, mean {s.mean():.3f}; under 0.5 cm {100 * (s < 0.5).mean():.1f} %; under 0.2 cm {100 * (s < 0.2).mean():.1f} %; "
      f"with a twist of 6 degrees or more: median {s[steps_rows['twist'].abs() >= 6].median():.3f}; over 2 cm: {int((s > 2.0).sum())} rows")
kept = steps_rows[s <= 2.0]
real = kept[kept["step_cm"] >= 0.3]
print(f"   without the steps over 2 cm: 1.2 cm or shorter {100 * (kept['step_cm'] <= 1.2).mean():.1f} %; 1.0 cm or shorter {100 * (kept['step_cm'] <= 1.0).mean():.1f} %; "
      f"real moves (0.3 cm or more): median {real['step_cm'].median():.3f}, 75th percentile {real['step_cm'].quantile(0.75):.3f}, 90th {real['step_cm'].quantile(0.9):.3f}, 99th {real['step_cm'].quantile(0.99):.3f}")
moved = 100.0 * numpy.hypot(frames["tip_x_m"] - frames.groupby("episode_id")["tip_x_m"].shift(1), frames["tip_y_m"] - frames.groupby("episode_id")["tip_y_m"].shift(1)).fillna(0.0)
episodes = pandas.DataFrame({"episode_id": frames["episode_id"], "sheet": frames["sheet"], "plan": frames["plan_number"], "moved": moved}).groupby("episode_id").agg(
    sheet = ("sheet", "first"), plan = ("plan", "first"), rows = ("moved", "size"), path_cm = ("moved", "sum"))
longest = episodes.sort_values("path_cm").iloc[-1]
print(f"   longest demonstration: plan {longest['plan']} ({longest['sheet']}), {longest['path_cm']:.0f} cm of tip path in {int(longest['rows'])} rows; at 0.66 cm per decision {longest['path_cm'] / 0.66:.0f} decisions")
print("   rows and tip path per sheet (median over episodes):")
print(episodes.groupby("sheet").agg(episodes = ("rows", "size"), rows_median = ("rows", "median"), rows_most = ("rows", "max"), path_median_cm = ("path_cm", "median"), path_most_cm = ("path_cm", "max")).round(1).to_string())
per_sheet = steps_rows.groupby("sheet").apply(lambda group: pandas.Series({"twist_per_cm_pooled": group["twist"].abs().sum() / group["step_cm"].sum()}), include_groups = False)
per_episode = steps_rows.groupby(["sheet", "episode_id"]).apply(lambda group: group["twist"].abs().sum() / group["step_cm"].sum(), include_groups = False).groupby("sheet").median()
print("C. demonstrated forearm turn per cm per sheet: pooled " + ", ".join(f"{name[-2:]} {value:.2f}" for name, value in per_sheet["twist_per_cm_pooled"].items())
      + "; median over episodes " + ", ".join(f"{name[-2:]} {value:.2f}" for name, value in per_episode.items()))

print()
print("B. what the model asked before the off-the-line decision of each failed run")
all_logs = {}
for number, folder in enumerate(RUN_FOLDERS, start = 1):
    log = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
    log["asked_cm"] = numpy.hypot(log["model_tip_forward_cm"], log["model_tip_left_cm"])
    all_logs[number] = log
for number, off in OFF_LINE.items():
    log = all_logs[number]
    before = log.iloc[:off]
    last_six = log.iloc[max(0, off - 6):off]
    after = log.iloc[off:]
    after = after[after["action"] == "model_step"]
    print(f"   run {number:2d}, decisions 0 to {off - 1}: asked median {before['asked_cm'].median():.2f} cm, under 0.5 cm on {100 * (before['asked_cm'] < 0.5).mean():.0f} %; "
          f"twist asked median size {before['model_twist_deg'].abs().median():.1f} degrees; driver over asked {1.0 / before['asked_cm'].median():.1f} times; "
          f"last six before the mark {last_six['asked_cm'].min():.2f} to {last_six['asked_cm'].max():.2f} cm; after the mark median {after['asked_cm'].median():.2f} cm ({len(after)} moves)")
log = all_logs[6]
joint_4 = numpy.degrees(log["joint_4"].to_numpy())
print(f"   run 6: joint 4 at decision 0 {joint_4[0]:.1f}, at decision 10 {joint_4[10]:.1f} (turned {joint_4[10] - joint_4[0]:+.1f}), largest turn from the start {numpy.abs(joint_4 - joint_4[0]).max():.1f} degrees "
      f"at decision {int(numpy.abs(joint_4 - joint_4[0]).argmax())}; sum of positive twists done in decisions 0 to 9: {numpy.clip(numpy.diff(joint_4[:11]), 0, None).sum():.1f}")

print()
print("C. on the robot, per run: twist done per cm done, twist asked per cm asked, against the sheet's demonstrated turn per cm (pooled)")
for number, log in all_logs.items():
    moves = log[log["action"] == "model_step"]
    tips = 100.0 * log[["tip_x", "tip_y"]].to_numpy()
    done = numpy.linalg.norm(numpy.diff(tips, axis = 0), axis = 1).sum()
    turned = numpy.abs(numpy.diff(numpy.degrees(log["joint_4"].to_numpy()))).sum()
    print(f"   run {number:2d} ({RUN_SHEETS[number - 1][-2:]}): done {turned / done:.2f}, asked {moves['model_twist_deg'].abs().sum() / moves['asked_cm'].sum():.2f}, "
          f"demonstrated {per_sheet.loc[RUN_SHEETS[number - 1], 'twist_per_cm_pooled']:.2f} deg/cm; mean asked {moves['asked_cm'].mean():.2f} cm, so own-length decisions for the path done: "
          f"{done / numpy.minimum(moves['asked_cm'], 1.2).mean():.0f} against {len(moves)} moves ({done / numpy.minimum(moves['asked_cm'], 1.2).mean() / len(moves):.2f} times)")

print()
print("D. the stall rule that is already in 02_code/model_mode.py (stall_steps 30, stall_extent_m 0.03, stall_twist_deg 15) on the tips as they really went")
for number, log in all_logs.items():
    tips = log[["tip_x", "tip_y"]].to_numpy()
    joint_4 = numpy.degrees(log["joint_4"].to_numpy())
    first = None
    for index in range(29, len(log)):
        recent = tips[index - 29:index + 1]
        extent = numpy.hypot(recent[:, 0].max() - recent[:, 0].min(), recent[:, 1].max() - recent[:, 1].min())
        turned = joint_4[index - 29:index + 1].max() - joint_4[index - 29:index + 1].min()
        if extent < 0.03 and turned < 15.0:
            first = index
            break
    print(f"   run {number:2d}: {'stalled at decision ' + str(first) if first is not None else 'never stalled'} ({len(log)} pictures)")
