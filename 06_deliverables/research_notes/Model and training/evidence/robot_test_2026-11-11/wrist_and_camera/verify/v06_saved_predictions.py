"""
Verifier, check 6: what the SAVED predictions of the fold models (each scored on the sheet it never saw) say about
stops and about the lean output, on the REAL recorded pictures (no imitated view, no inference here).

    a) the stop rule (0.5 on two pictures in a row, training_metrics.stop_metrics) per episode, as recorded, split by
       how far the wrist at the episode's end was from the driver's 13.5 degrees;
    b) the stop chance on the stop rows themselves, split the same way;
    c) the lean output against the lean label: sign, correlation;
    d) what following the lean output would do to the wrist, linearised: every degree of lean error becomes
       1 / 0.867 degrees of wrist (the slope of lean against wrist is re-measured here from the wrist's own steps), so
       the wrist's distance from the demonstrated one is the running sum of (predicted - demonstrated) lean change.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe v06_saved_predictions.py
"""

import sys

import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
from training_metrics import stop_metrics
from training_targets import load_training_frames

RUNS_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014"
RUNS = {"E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "E recipe seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2",
        "A recipe": "2026-11-09_resnet18_final_candidate"}
HELD_WRIST_DEG = 13.5
pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)

frames = load_training_frames()
frames["wrist_deg"] = numpy.degrees(frames["joint_5"])
frames["twist_deg"] = numpy.degrees(frames["joint_4"])
frames["is_start_mark"] = frames["sheet_orientation"].str.contains("START MARK", na = False)
print("cleaned rows:", len(frames), "episodes:", frames["episode_id"].nunique(), "stop rows:", int((frames["action"] == "stop").sum()))
last = frames.groupby("episode_id").last()
last["end_offset"] = last["wrist_deg"] - HELD_WRIST_DEG
last["end_group"] = numpy.where(last["end_offset"].abs() <= 5.7, "within 5.7 of 13.5", numpy.where(last["end_offset"] < 0, "below (more than 5.7)", "above (more than 5.7)"))


def stop_outcome(rows: pandas.DataFrame) -> str:
    """Where the rule (0.5 on two pictures in a row) first fires in one episode: my own few lines, not the project's."""

    passing = (rows["stop_probability"].to_numpy() >= 0.5).astype(int)
    for position in range(1, len(passing)):
        if passing[position] and passing[position - 1]:
            distance = float(rows["goal_distance_cm"].to_numpy()[position])
            return "right" if distance <= 2.0 else ("early" if distance > 5.0 else "between")
    return "never"


for label, folder in RUNS.items():
    parts = []
    for sheet_number in range(1, 7):
        saved = numpy.load(f"{RUNS_FOLDER}/{folder}/predictions_without_path_{sheet_number:02d}.npz")
        parts.append(pandas.DataFrame({"frame_index": saved["frame_index"], "stop_probability": saved["stop_probability"], "lean_predicted": saved["lean_deg"],
                                       "hand_predicted": saved["hand_deg"], "twist_predicted": saved["twist_deg"]}))
    scored = frames.merge(pandas.concat(parts), on = "frame_index", how = "inner").sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    print(f"\n================ {label} ({folder}): {len(scored)} scored rows, {scored['episode_id'].nunique()} episodes ================")

    # a) the stop rule per episode
    project = stop_metrics((scored["action"] == "stop").to_numpy(), scored["stop_probability"].to_numpy(), scored["goal_distance_cm"].to_numpy(),
                           scored["episode_id"].to_numpy(), threshold = 0.5, consecutive_pictures = 2)
    count = scored["episode_id"].nunique()
    print("a) project's stop_metrics, 0.5 twice in a row: stopped right %.0f, early %.0f, never %.0f of %d episodes" % (
        project["episodes_stopped_right"] * count, project["episodes_stopped_early"] * count, project["episodes_never_stopped"] * count, count))
    outcomes = scored.groupby("episode_id").apply(stop_outcome).rename("outcome").to_frame().join(last[["end_group", "end_offset", "sheet", "is_start_mark"]])
    print("   my own rule (right: within 2 cm of the goal; early: more than 5 cm before it):", outcomes["outcome"].value_counts().to_dict())
    table = outcomes.groupby("end_group")["outcome"].value_counts().unstack(fill_value = 0)
    table["episodes"] = table.sum(axis = 1)
    table["right_share"] = (table.get("right", 0) / table["episodes"]).round(3)
    print("   as recorded, by the wrist at the episode's end:")
    print(table.to_string())
    by_start = outcomes.groupby(["is_start_mark", "end_group"])["outcome"].agg(episodes = "size", right = lambda v: int((v == "right").sum()))
    by_start["right_share"] = (by_start["right"] / by_start["episodes"]).round(3)
    print("   split by start kind (True: start mark):")
    print(by_start.to_string())
    by_sheet = outcomes[outcomes["is_start_mark"]].groupby(["sheet", "end_group"])["outcome"].agg(episodes = "size", right = lambda v: int((v == "right").sum()))
    print("   start-mark episodes by sheet:")
    print(by_sheet.T.to_string())

    # b) the stop chance at the stop rows
    stops = scored[scored["action"] == "stop"].copy()
    stops["offset"] = stops["wrist_deg"] - HELD_WRIST_DEG
    stops["group"] = numpy.where(stops["offset"].abs() <= 5.7, "within 5.7 of 13.5", numpy.where(stops["offset"] < 0, "below", "above"))
    print("b) stop rows as recorded: share with a stop chance of 0.5 or more")
    print(stops.groupby("group")["stop_probability"].agg(rows = "size", would_stop = lambda v: round(float((v >= 0.5).mean()), 3), mean_chance = lambda v: round(float(v.mean()), 3)).to_string())
    print("   without sheet 06:")
    print(stops[stops["sheet"] != "path_06"].groupby("group")["stop_probability"].agg(rows = "size", would_stop = lambda v: round(float((v >= 0.5).mean()), 3)).to_string())

    # c) the lean output against the label
    steps = scored[scored["action"] != "stop"]
    label_lean = steps["lean_change_deg"].to_numpy(dtype = float)
    predicted = steps["lean_predicted"].to_numpy(dtype = float)
    is_leaning = numpy.abs(label_lean) >= 0.5
    print("c) lean output: %d steps; predicted mean %.3f sd %.3f; label mean %.3f sd %.3f; correlation %.3f; sign right on %.1f %% of the %d steps with a label of 0.5 degrees or more" % (
        len(steps), predicted.mean(), predicted.std(), label_lean.mean(), label_lean.std(), numpy.corrcoef(predicted, label_lean)[0, 1],
        100 * (numpy.sign(predicted[is_leaning]) == numpy.sign(label_lean[is_leaning])).mean(), int(is_leaning.sum())))
    print("   share of those labels that are positive (a constant 'always forward' answer scores this): %.1f %%" % (100 * (label_lean[is_leaning] > 0).mean()))
    wrist_step = scored.groupby("episode_id")["wrist_deg"].diff().shift(-1)
    same_episode = scored["episode_id"] == scored["episode_id"].shift(-1)
    jump = (wrist_step.abs() > 1.0) & same_episode & (scored["action"] != "stop")
    print("   steps in which the wrist moved by more than 1 degree: %d; predicted lean against the wrist's own change there: correlation %.3f" % (
        int(jump.sum()), numpy.corrcoef(scored.loc[jump, "lean_predicted"], wrist_step[jump])[0, 1]))
    hand = scored["hand_predicted"]
    print("   hand output: mean %.3f, 1st to 99th percentile %.3f to %.3f degrees per step" % (hand.mean(), hand.quantile(0.01), hand.quantile(0.99)))

    # d) following the lean output, linearised
    moved = jump & (scored.groupby("episode_id")["twist_deg"].diff().shift(-1).abs() < 0.5) & (numpy.hypot(scored["tip_forward_m"], scored["tip_left_m"]) < 0.002)
    slope = numpy.polyfit(wrist_step[moved], scored.loc[moved, "lean_change_deg"], 1)[0] if moved.sum() > 20 else numpy.nan
    print("d) wrist steps in place (tip under 2 mm, twist under 0.5 degrees): %d; lean label per degree of wrist there: %.3f" % (int(moved.sum()), slope))
    slope = 0.867 if not numpy.isfinite(slope) else slope
    error = (scored["lean_predicted"] - scored["lean_change_deg"]).where(scored["action"] != "stop", 0.0).fillna(0.0)
    scored["wrist_gap_followed"] = (error.groupby(scored["episode_id"]).cumsum() - error) / slope
    scored["wrist_gap_constant"] = HELD_WRIST_DEG - scored["wrist_deg"]
    for name, part in (("start-mark episodes", scored[scored["is_start_mark"]]), ("taught-start episodes", scored[~scored["is_start_mark"]])):
        followed = part["wrist_gap_followed"] + (HELD_WRIST_DEG - part.groupby("episode_id")["wrist_deg"].transform("first"))
        print("   %s (%d rows): mean distance to the demonstrated wrist: following the lean output from 13.5: %.1f degrees (median %.1f, 90th %.1f); constant 13.5: %.1f (median %.1f, 90th %.1f)" % (
            name, len(part), followed.abs().mean(), followed.abs().median(), followed.abs().quantile(0.9),
            part["wrist_gap_constant"].abs().mean(), part["wrist_gap_constant"].abs().median(), part["wrist_gap_constant"].abs().quantile(0.9)))
    final = scored[scored["is_start_mark"]].groupby("episode_id")["wrist_gap_followed"].last()
    print("   start-mark episodes, at the last row: followed wrist minus demonstrated: median %+.1f, 5th %+.1f, 95th %+.1f degrees" % (
        final.median(), final.quantile(0.05), final.quantile(0.95)))
