"""
Extra read-only checks for the evidence critique of the targets section.

Reads frames.csv, the audit's view rows, the targets section's saved model outputs and the final candidate's saved
predictions; writes only extra_checks.out.txt next to this script.

Checks
------
A. Twist calibration (least-squares gain through zero, sign by predicted size) per held-out sheet, for the earlier
   end-to-end exact run (sheets 03, 05, 06) and for the final candidate (sheets 01, 02).
B. Final candidate, sheet 03 (finished after the section was written): backward moves within 45 deg.
C. Final candidate, sheets 01 to 03: stop outcomes at fixed 0.5, at the validation-tuned thresholds of the fold files,
   and with the threshold chosen on the other finished sheets (the section's leave-one-sheet-out rule).
"""

import json
from pathlib import Path

import numpy
import pandas

PROCESSED_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RUNS_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs")
SCRATCHPAD_FOLDER = Path("C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad")
TARGETS_FOLDER = SCRATCHPAD_FOLDER / "design" / "targets"
AUDIT_FOLDER = SCRATCHPAD_FOLDER / "audit"
OUTPUT_PATH = Path(__file__).resolve().parent / "extra_checks.out.txt"
FINAL_FOLDER = RUNS_FOLDER / "2026-11-09_resnet18_final_candidate"
THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
NOMINAL_ANGLES = {"forward": 0.0, "forward_left": 60.0, "left": 90.0, "back_left": 135.0, "back": 180.0,
                  "back_right": -135.0, "right": -90.0, "forward_right": -60.0}
DIRECTION_NAMES = list(NOMINAL_ANGLES)
BACKWARD_NAMES = ["back", "back_left", "back_right"]

output_lines = []


def say(text: str) -> None:
    """Keep a line for the output file and print it."""

    output_lines.append(text)
    print(text, flush = True)


def nearest_direction(angles: numpy.ndarray) -> numpy.ndarray:
    """Return the nearest nominal direction name for each angle in degrees (the targets section's rule)."""

    names = numpy.array(DIRECTION_NAMES)
    values = numpy.array([NOMINAL_ANGLES[name] for name in DIRECTION_NAMES])
    differences = numpy.abs((numpy.asarray(angles)[:, None] - values[None, :] + 180.0) % 360.0 - 180.0)
    return names[numpy.argmin(differences, axis = 1)]


def twist_calibration(table: pandas.DataFrame, label: str) -> None:
    """Report the least-squares gain of the demonstrated on the predicted twist and the sign by predicted size."""

    table = table[table["action"] != "stop"]
    predicted = table["twist_deg"].to_numpy()
    demonstrated = table["forearm_change_deg"].fillna(0.0).clip(-15.0, 15.0).to_numpy()
    gain = float((predicted * demonstrated).sum() / (predicted * predicted).sum())
    twisted = numpy.abs(demonstrated) >= 2.0
    sign_right = float((numpy.sign(predicted[twisted]) == numpy.sign(demonstrated[twisted])).mean())
    parts = []
    for low, high in ((0, 1), (1, 3), (3, 20)):
        in_bin = (numpy.abs(predicted) >= low) & (numpy.abs(predicted) < high)
        in_bin_twisted = in_bin & twisted
        bin_sign = float((numpy.sign(predicted[in_bin_twisted]) == numpy.sign(demonstrated[in_bin_twisted])).mean()) if in_bin_twisted.any() else float("nan")
        parts.append(f"|pred| {low}-{high}: n {int(in_bin.sum())}, sign {bin_sign:.3f}")
    say(f"A {label}: non-stop pictures {len(table)}; gain {gain:.2f}; twisted rows {int(twisted.sum())}, sign right {sign_right:.3f}; "
        f"median |pred| on twisted {numpy.median(numpy.abs(predicted[twisted])):.2f} against |demo| {numpy.median(numpy.abs(demonstrated[twisted])):.2f}; "
        + "; ".join(parts))


def episode_outcomes(table: pandas.DataFrame, threshold: float, consecutive: int) -> pandas.DataFrame:
    """Per episode: right (within 2 cm), early (over 3 cm before), between, or never (the section's rule)."""

    results = []
    for episode_id, group in table.groupby("episode_id", sort = False):
        passing = group["stop_probability"].to_numpy() >= threshold
        run_length = numpy.zeros(len(passing), dtype = int)
        for position in range(len(passing)):
            run_length[position] = (run_length[position - 1] + 1 if position > 0 else 1) if passing[position] else 0
        hits = numpy.flatnonzero(run_length >= consecutive)
        if len(hits) == 0:
            results.append({"episode_id": episode_id, "outcome": "never", "distance": numpy.nan, "no_line_ahead": False})
            continue
        row = group.iloc[hits[0]]
        distance = float(row["goal_distance_cm"])
        outcome = "right" if distance <= 2.0 else ("early" if distance > 3.0 else "between")
        results.append({"episode_id": episode_id, "outcome": outcome, "distance": distance, "no_line_ahead": bool(row["no_line_ahead"])})
    return pandas.DataFrame(results)


def outcome_shares(outcomes: pandas.DataFrame) -> dict:
    """Shares of right, early and never stops, and the episode count."""

    counts = outcomes["outcome"].value_counts()
    total = len(outcomes)
    return {"right": counts.get("right", 0) / total, "early": counts.get("early", 0) / total,
            "never": counts.get("never", 0) / total, "episodes": total,
            "right_count": int(counts.get("right", 0)), "early_count": int(counts.get("early", 0))}


def main() -> None:
    frames = pandas.read_csv(PROCESSED_FOLDER / "frames.csv", low_memory = False,
                             usecols = ["frame_index", "episode_id", "row_in_episode", "sheet", "action", "forearm_change_deg",
                                        "move_angle_deg", "goal_distance_cm"])
    view_rows = pandas.read_csv(AUDIT_FOLDER / "view_and_tracks" / "s09_view_rows.csv", usecols = ["frame_index", "line_share_top_half"])
    frames = frames.merge(view_rows, on = "frame_index", how = "left")
    frames["no_line_ahead"] = frames["line_share_top_half"].fillna(1.0) <= 0.002
    say(f"frames {len(frames)}; goal_distance_cm missing on {int(frames['goal_distance_cm'].isna().sum())} rows "
        f"({int((frames['goal_distance_cm'].isna() & (frames['action'] != 'stop')).sum())} of them non-stop rows)")
    frames["goal_distance_cm"] = frames["goal_distance_cm"].fillna(0.0)

    # A. twist calibration per sheet
    for fold in ("without_path_03", "without_path_05", "without_path_06"):
        outputs = pandas.read_csv(TARGETS_FOLDER / f"exact_outputs_{fold}.csv").merge(frames, on = "frame_index", how = "left")
        twist_calibration(outputs, f"earlier end-to-end exact, {fold}")
    for fold in ("without_path_01", "without_path_02"):
        outputs = pandas.read_csv(TARGETS_FOLDER / f"final_outputs_{fold}.csv").merge(frames, on = "frame_index", how = "left")
        twist_calibration(outputs, f"final candidate, {fold}")

    # B and C: final candidate predictions
    lookup = frames.set_index("frame_index")
    tables = {}
    for fold in ("without_path_01", "without_path_02", "without_path_03"):
        path = FINAL_FOLDER / f"predictions_{fold}.npz"
        if not path.exists():
            say(f"{fold}: no predictions")
            continue
        saved = numpy.load(path)
        say(f"{fold}: saved arrays {sorted(saved.files)}")
        table = lookup.loc[saved["frame_index"]].reset_index()
        table["stop_probability"] = saved["stop_probability"]
        if "angle_deg" in saved.files:
            table["angle_deg"] = saved["angle_deg"]
        if "twist_deg" in saved.files:
            table["twist_deg"] = saved["twist_deg"]
        tables[fold] = table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)

    for fold, table in tables.items():
        if "angle_deg" not in table:
            continue
        moving = table[table["action"].isin(DIRECTION_NAMES)]
        error = numpy.abs((moving["angle_deg"] - moving["move_angle_deg"] + 180.0) % 360.0 - 180.0)
        is_backward = numpy.isin(nearest_direction(moving["move_angle_deg"].to_numpy()), BACKWARD_NAMES)
        say(f"B {fold}: moving rows {len(moving)}; backward {int(is_backward.sum())}; within 45 on backward {(error[is_backward] <= 45).mean():.3f}, "
            f"on the others {(error[~is_backward] <= 45).mean():.3f}; predicted straight back {(nearest_direction(moving['angle_deg'].to_numpy()) == 'back').mean():.3f}")
        if "twist_deg" in table:
            twist_calibration(table, f"final candidate (saved predictions), {fold}")

    grid = {}
    for fold, table in tables.items():
        for consecutive in (1, 2):
            for threshold in THRESHOLDS:
                grid[(fold, consecutive, threshold)] = episode_outcomes(table, threshold, consecutive)
    for fold, table in tables.items():
        measures = json.loads((FINAL_FOLDER / f"fold_{fold}.json").read_text(encoding = "utf-8"))["measures"]
        far = table[table["goal_distance_cm"] > 5.0]
        say(f"C {fold}: episodes {table['episode_id'].nunique()}; far no-line p90 {far.loc[far['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f} "
            f"(n {int(far['no_line_ahead'].sum())}), with line p90 {far.loc[~far['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f}; "
            f"stop pictures median {table.loc[table['action'] == 'stop', 'stop_probability'].median():.3f}")
        for consecutive, key in ((1, "tuned_stop_threshold"), (2, "tuned_two_stop_threshold")):
            for rule, threshold in (("fixed", 0.5), ("validation-tuned", measures[key])):
                outcomes = episode_outcomes(table, threshold, consecutive)
                shares = outcome_shares(outcomes)
                early = outcomes[outcomes["outcome"] == "early"]
                say(f"  {consecutive} in a row, {rule} {threshold}: right {shares['right_count']}/{shares['episodes']} ({shares['right']:.3f}), "
                    f"early {shares['early_count']} ({shares['early']:.3f}), never {shares['never']:.3f}; early median {early['distance'].median() if len(early) else float('nan'):.1f} cm, "
                    f"no line ahead {int(early['no_line_ahead'].sum())} of {len(early)}")
        other_folds = [other for other in tables if other != fold]
        if other_folds:
            for consecutive in (1, 2):
                def pooled(threshold):
                    return outcome_shares(pandas.concat([grid[(other, consecutive, threshold)] for other in other_folds]))
                chosen = max(THRESHOLDS, key = lambda threshold: (pooled(threshold)["right"], -pooled(threshold)["early"]))
                shares = outcome_shares(grid[(fold, consecutive, chosen)])
                say(f"  {consecutive} in a row, chosen on the other finished sheets {[other[-2:] for other in other_folds]}: threshold {chosen}; "
                    f"right {shares['right_count']}/{shares['episodes']} ({shares['right']:.3f}), early {shares['early_count']} ({shares['early']:.3f}), never {shares['never']:.3f}")

    OUTPUT_PATH.write_text("\n".join(output_lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
