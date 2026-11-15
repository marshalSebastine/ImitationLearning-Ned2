"""
Re-check of the brief's quick ("scout") numbers that touch the bends, and two more numbers the findings lean on.

    1. the demonstrated step lengths (median, mean, shares under 0.5 and 0.2 cm, steps with a twist of 6 degrees or more);
    2. the forearm twist per cm of tip travel: the demonstrations per sheet, and each robot run three ways: per cm
       DRIVEN (1 cm per move), per cm ASKED (the model's own step lengths) and per move;
    3. the first steps of the demonstration plan 03-04 (a turn in place with jumping directions);
    4. how long the average of neighbouring demonstrated steps is, per kind of place, beside the predicted step length:
       a model that answers with the average of the steps of look-alike pictures gives this length;
    5. where the tip stood when the first bend of sheet 03 was taken (2 cm to the left of the start), demonstrations
       against the three robot runs;
    6. where each run's tip started, against the starts of the demonstrations of the same sheet and end.

Run from this folder:  python s08_scout_checks.py
Output: s08_scout_checks.out.txt.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

import common
from config import TRAINING_DATA_PARAMETERS

CHECK_PARAMETERS = {
    "run_prefix": "2026-11-11_17",
    "run_sheets": {"172021": "path_01", "172421": "path_01", "172743": "path_02", "173036": "path_02", "173549": "path_03", "174108": "path_04",
                   "174448": "path_05", "174936": "path_06", "175203": "path_03", "175544": "path_03"},
    "run_start_ends": {"path_01": "A", "path_02": "A", "path_03": "A", "path_04": "A", "path_05": "B", "path_06": "A"},
    "plan_03_04": "plan_03-04_try_1",
    "taken_left_cm": 2.0,
    "average_windows": [3, 5]
}



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(CHECK_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s08_scout_checks.out.txt")
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    frames = common.load_demonstration_frames(settings)
    moving = frames[~frames["is_stop"]]

    report("== 1. demonstrated step length (cleaned rows without the stop rows; measured, frames.csv) ==")
    report(f"rows {len(moving)} | median {moving['step_cm'].median():.2f} cm | mean {moving['step_cm'].mean():.2f} cm | under 0.5 cm: {float((moving['step_cm'] < 0.5).mean()):.3f} | "
           f"under 0.3 cm: {float((moving['step_cm'] < 0.3).mean()):.3f} | under 0.2 cm: {float((moving['step_cm'] < 0.2).mean()):.3f}")
    twisting = moving[moving["twist_deg"].abs() >= 6.0]
    report(f"steps with a twist of 6 degrees or more: {len(twisting)} ({len(twisting) / len(moving):.3f}) | their median length {twisting['step_cm'].median():.2f} cm")
    raw = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False)
    raw = raw[raw["action"] != "stop"]
    raw_length = 100.0 * numpy.hypot(raw["tip_forward_m"].fillna(0.0), raw["tip_left_m"].fillna(0.0))
    report(f"all {len(raw)} rows of frames.csv without the stop rows (no cleaning): median {raw_length.median():.2f} cm | mean {raw_length.mean():.2f} | "
           f"under 0.5 cm: {float((raw_length < 0.5).mean()):.3f} | under 0.2 cm: {float((raw_length < 0.2).mean()):.3f} | "
           f"twist of 6 degrees or more: median length {raw_length[raw['forearm_change_deg'].abs() >= 6.0].median():.2f} cm")

    report("")
    report("== 2. forearm twist per cm of tip travel ==")
    per_episode = moving.groupby(["sheet", "episode_id"]).agg(twist = ("twist_deg", lambda values: float(values.abs().sum())), travel = ("step_cm", "sum"),
                                                              steps = ("step_cm", "size")).reset_index()
    per_episode["twist_per_cm"] = per_episode["twist"] / per_episode["travel"]
    per_episode["twist_per_step"] = per_episode["twist"] / per_episode["steps"]
    demonstration = per_episode.groupby("sheet").agg(episodes = ("episode_id", "size"), twist_per_cm_median = ("twist_per_cm", "median"),
                                                     twist_per_step_median = ("twist_per_step", "median")).round(2)
    report("-- demonstrations, per sheet (median over the episodes of: sum of absolute twists / tip travel, and / steps) --")
    report(demonstration.to_string())
    report("-- robot runs --")
    for run_folder in sorted(Path(settings["model_runs_folder"]).glob(settings["run_prefix"] + "*")):
        steps = pandas.read_csv(run_folder / "steps.csv")
        moves = steps[steps["action"] == "model_step"]
        asked = numpy.hypot(moves["model_tip_forward_cm"], moves["model_tip_left_cm"])
        sheet = settings["run_sheets"][run_folder.name[11:17]]
        report(f"{run_folder.name[11:]:32s} | {sheet} | moves {len(moves):3d} | twist used per cm driven {moves['twist_used_deg'].abs().sum() / len(moves):.2f} | "
               f"twist asked per cm ASKED {moves['model_twist_deg'].abs().sum() / asked.sum():.2f} | twist asked per move {moves['model_twist_deg'].abs().mean():.2f} | "
               f"demonstrations of that sheet: {demonstration.loc[sheet, 'twist_per_cm_median']:.2f} per cm, {demonstration.loc[sheet, 'twist_per_step_median']:.2f} per step")

    report("")
    report(f"== 3. the first steps of the demonstration {settings['plan_03_04']} ==")
    episode = frames[frames["episode_id"].str.contains(settings["plan_03_04"], regex = False)].head(12)
    report(episode[["row_in_episode", "action", "step_cm", "step_picture_deg", "step_table_deg", "twist_deg", "forearm_turn_deg", "place"]].round(2).to_string(index = False))

    report("")
    report("== 4. how long the average of neighbouring demonstrated steps is (cm), beside the predicted step ==")
    direction = numpy.radians(frames["step_table_deg"].to_numpy())
    table = pandas.DataFrame({"episode": frames["episode_id"], "east": frames["step_cm"].to_numpy() * numpy.cos(direction),
                              "north": frames["step_cm"].to_numpy() * numpy.sin(direction), "length": frames["step_cm"].to_numpy(), "place": frames["place"]})
    table.loc[frames["is_stop"].to_numpy(), ["east", "north", "length"]] = numpy.nan
    for window in settings["average_windows"]:
        east = table.groupby("episode")["east"].transform(lambda values: values.rolling(window, center = True, min_periods = window).mean())
        north = table.groupby("episode")["north"].transform(lambda values: values.rolling(window, center = True, min_periods = window).mean())
        table[f"average_{window}"] = numpy.hypot(east, north)
    recipe_e = numpy.load(Path(settings["cache_folder"]) / "fold_outputs_recipe_e.npz")
    model_e = numpy.load(Path(settings["cache_folder"]) / "all_sheets_e_on_demonstrations.npz")
    table["held_out"] = numpy.linalg.norm(recipe_e["exact"][:, :2], axis = 1)
    table["trained_on"] = numpy.linalg.norm(model_e["exact"][:, :2], axis = 1)
    largest_window = max(settings["average_windows"])
    for place in common.PLACE_NAMES:
        part = table[(table["place"] == place) & table[f"average_{largest_window}"].notna()]
        averages = " | ".join(f"average of {window} steps: {part[f'average_{window}'].median():.2f}" for window in settings["average_windows"])
        report(f"{place:12s} | pictures {len(part):5d} | demonstrated step median {part['length'].median():.2f} | {averages} | "
               f"predicted, held-out sheet (recipe E): {part['held_out'].median():.2f} | predicted by model E on pictures it was trained on: {part['trained_on'].median():.2f}")

    report("")
    report(f"== 5. sheet 03 from end A: the tip's x when it is {settings['taken_left_cm']} cm to the left of its start (the first bend taken) ==")
    sheet = frames[(frames["sheet"] == "path_03") & (frames["start_end"] == "A") & frames["start_kind"].isin(["STD", "OL1", "OR1", "OL1.5", "OR1.5", "TL10", "TR10", "TL15"])]
    rows = []
    for episode_id, episode in sheet.groupby("episode_id"):
        left = 100.0 * (episode["tip_y_m"] - episode["tip_y_m"].iloc[0])
        is_taken = left >= settings["taken_left_cm"]
        if is_taken.any():
            first = int(numpy.argmax(is_taken.to_numpy()))
            rows.append({"start_kind": episode["start_kind"].iloc[0], "x_cm": 100.0 * episode["tip_x_m"].iloc[first], "x_advance_cm": 100.0 * (episode["tip_x_m"].iloc[first] - episode["tip_x_m"].iloc[0]),
                         "farthest_advance_cm": 100.0 * (episode["tip_x_m"].iloc[:first + 1].max() - episode["tip_x_m"].iloc[0]),
                         "steps": first, "tip_travel_cm": float(episode["step_cm"].iloc[:first].sum()), "twist_sum_deg": float(episode["twist_deg"].iloc[:first].sum())})
    taken = pandas.DataFrame(rows)
    report(f"demonstrations ({len(taken)} episodes with a start on the start mark): x {taken['x_cm'].median():.1f} cm (from {taken['x_cm'].min():.1f} to {taken['x_cm'].max():.1f}) | "
           f"advance in x since the start {taken['x_advance_cm'].median():.1f} cm (from {taken['x_advance_cm'].min():.1f} to {taken['x_advance_cm'].max():.1f}) | "
           f"farthest advance in x before that {taken['farthest_advance_cm'].median():.1f} cm (from {taken['farthest_advance_cm'].min():.1f} to {taken['farthest_advance_cm'].max():.1f}) | "
           f"steps {taken['steps'].median():.0f} | tip travel {taken['tip_travel_cm'].median():.1f} cm | forearm twist summed {taken['twist_sum_deg'].median():.0f} deg")
    for run_folder in sorted(Path(settings["model_runs_folder"]).glob(settings["run_prefix"] + "*sheet03*")):
        steps = pandas.read_csv(run_folder / "steps.csv")
        left = 100.0 * (steps["tip_y"] - steps["tip_y"].iloc[0])
        first = int(numpy.argmax((left >= settings["taken_left_cm"]).to_numpy()))
        asked = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
        report(f"{run_folder.name[11:]:32s} | x {100 * steps['tip_x'].iloc[first]:.1f} cm | advance in x since the start {100 * (steps['tip_x'].iloc[first] - steps['tip_x'].iloc[0]):.1f} cm | "
               f"farthest advance in x before that {100 * (steps['tip_x'].iloc[:first + 1].max() - steps['tip_x'].iloc[0]):.1f} cm | "
               f"moves {first} (1 cm each) | the model had asked for {asked.iloc[:first].sum():.1f} cm over these moves | forearm twist used {steps['twist_used_deg'].iloc[:first].sum():.0f} deg")

    report("")
    report("== 6. where each run started, against where the demonstrations with a standard start of the same sheet and end started (tip, cm, robot's base frame) ==")
    for run_folder in sorted(Path(settings["model_runs_folder"]).glob(settings["run_prefix"] + "*")):
        steps = pandas.read_csv(run_folder / "steps.csv")
        sheet_name = settings["run_sheets"][run_folder.name[11:17]]
        start_end = settings["run_start_ends"][sheet_name]
        reference = frames[(frames["sheet"] == sheet_name) & (frames["start_kind"] == "STD") & (frames["start_end"] == start_end)].groupby("episode_id").first()
        report(f"{run_folder.name[11:]:32s} | {sheet_name} from end {start_end} | run: x {100 * steps['tip_x'].iloc[0]:.1f}, y {100 * steps['tip_y'].iloc[0]:+.1f}, forearm {numpy.degrees(steps['joint_4'].iloc[0]):+.1f} deg | "
               f"{len(reference)} demonstrations: x {100 * reference['tip_x_m'].median():.1f} (from {100 * reference['tip_x_m'].min():.1f} to {100 * reference['tip_x_m'].max():.1f}), "
               f"y {100 * reference['tip_y_m'].median():+.1f} (from {100 * reference['tip_y_m'].min():+.1f} to {100 * reference['tip_y_m'].max():+.1f}), "
               f"forearm {reference['forearm_turn_deg'].median():+.1f} deg | "
               f"apart {numpy.hypot(100 * steps['tip_x'].iloc[0] - 100 * reference['tip_x_m'].median(), 100 * steps['tip_y'].iloc[0] - 100 * reference['tip_y_m'].median()):.1f} cm")



if __name__ == "__main__":
    main()
