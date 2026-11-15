"""
Read-only checks for judge 2 (robustness and simplicity), on the final candidate's finished held-out folds.

Reads frames.csv of the v01 index and the final candidate's saved predictions and fold files; writes only
judge2_checks.out.txt next to this script.

Checks
------
A. Stop rules: fixed thresholds with one or two pictures in a row, the inner-validation tuned rules of the fold
   files, and a rule chosen leave-one-sheet-out on the other finished final-candidate sheets; per sheet and pooled,
   as counts of right (within 2 cm), early (more than 3 cm before), between and never, with Wilson intervals.
B. Backward moves: within 45 deg on backward moves and on the others, per sheet.
C. Twist: least-squares gain through zero and sign accuracy on rows that twisted 2 deg or more, per sheet.
D. Early stops under the fixed 0.5 one-picture rule: median distance before the goal.
"""

import json
import math
from pathlib import Path

import numpy
import pandas

PROCESSED_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RUN_FOLDER = Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_resnet18_final_candidate")
OUTPUT_PATH = Path(__file__).resolve().parent / "judge2_checks.out.txt"
NOMINAL_ANGLES = {"forward": 0.0, "forward_left": 60.0, "left": 90.0, "back_left": 135.0, "back": 180.0,
                  "back_right": -135.0, "right": -90.0, "forward_right": -60.0}
DIRECTION_NAMES = list(NOMINAL_ANGLES)
BACKWARD_NAMES = ["back", "back_left", "back_right"]
THRESHOLDS = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RULES = [(threshold, consecutive) for threshold in THRESHOLDS for consecutive in (1, 2)]

output_lines = []


def say(text: str) -> None:
    """Keep a line for the output file and print it."""

    output_lines.append(text)
    print(text, flush = True)


def wilson_interval(successes: int, trials: int, z_value: float = 1.96) -> tuple:
    """Return the Wilson score interval for a share."""

    if trials == 0:
        return (float("nan"), float("nan"))
    share = successes / trials
    denominator = 1.0 + z_value ** 2 / trials
    centre = (share + z_value ** 2 / (2 * trials)) / denominator
    half_width = z_value * math.sqrt(share * (1 - share) / trials + z_value ** 2 / (4 * trials ** 2)) / denominator
    return (centre - half_width, centre + half_width)


def nearest_direction(angles: numpy.ndarray) -> numpy.ndarray:
    """Return the nearest nominal direction name for each angle in degrees."""

    names = numpy.array(DIRECTION_NAMES)
    values = numpy.array([NOMINAL_ANGLES[name] for name in DIRECTION_NAMES])
    differences = numpy.abs((numpy.asarray(angles)[:, None] - values[None, :] + 180.0) % 360.0 - 180.0)
    return names[numpy.argmin(differences, axis = 1)]


def episode_outcomes(table: pandas.DataFrame, threshold: float, consecutive: int) -> dict:
    """Return counts of right, early, between and never stops, and the early-stop distances."""

    counts = {"right": 0, "early": 0, "between": 0, "never": 0}
    early_distances = []
    for _, group in table.groupby("episode_id", sort = False):
        passing = group["stop_probability"].to_numpy() >= threshold
        run_length = numpy.zeros(len(passing), dtype = int)
        for position in range(len(passing)):
            run_length[position] = (run_length[position - 1] + 1 if position > 0 else 1) if passing[position] else 0
        hits = numpy.flatnonzero(run_length >= consecutive)
        if len(hits) == 0:
            counts["never"] += 1
            continue
        distance = float(group["goal_distance_cm"].iloc[hits[0]])
        if distance <= 2.0:
            counts["right"] += 1
        elif distance > 3.0:
            counts["early"] += 1
            early_distances.append(distance)
        else:
            counts["between"] += 1
    counts["early_distances"] = early_distances
    return counts


def format_counts(counts: dict) -> str:
    """Return right / early / between / never as text."""

    return f"{counts['right']} / {counts['early']} / {counts['between']} / {counts['never']}"


def main() -> None:
    frames = pandas.read_csv(PROCESSED_FOLDER / "frames.csv", low_memory = False,
                             usecols = ["frame_index", "episode_id", "row_in_episode", "sheet", "action", "forearm_change_deg",
                                        "move_angle_deg", "goal_distance_cm"])
    folds = sorted(path.stem.replace("fold_", "") for path in RUN_FOLDER.glob("fold_without_path_*.json"))
    tables = {}
    fold_measures = {}
    for fold in folds:
        prediction_path = RUN_FOLDER / f"predictions_{fold}.npz"
        if not prediction_path.exists():
            continue
        data = numpy.load(prediction_path)
        table = pandas.DataFrame({"frame_index": data["frame_index"], "angle_deg": data["angle_deg"],
                                  "stop_probability": data["stop_probability"].astype(float),
                                  "twist_deg": data["twist_deg"].astype(float)})
        table = table.merge(frames, on = "frame_index", how = "left").sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
        table["goal_distance_cm"] = table["goal_distance_cm"].fillna(0.0)
        sheet = fold[-2:]
        tables[sheet] = table
        fold_measures[sheet] = json.loads((RUN_FOLDER / f"fold_{fold}.json").read_text(encoding = "utf-8"))["measures"]
    sheets = sorted(tables)
    say(f"final candidate folds with predictions: {sheets}; episodes per sheet: "
        + ", ".join(f"{sheet} {tables[sheet]['episode_id'].nunique()}" for sheet in sheets))

    # A. stop rules
    say("\nA. stop outcomes, right / early / between / never (episodes)")
    grid = {sheet: {rule: episode_outcomes(tables[sheet], *rule) for rule in RULES} for sheet in sheets}
    check = {sheet: grid[sheet][(0.5, 1)] for sheet in sheets}
    for sheet in sheets:
        measures = fold_measures[sheet]
        episodes = measures["test_episodes"]
        stored = (round(measures["episodes_stopped_right"] * episodes), round(measures["episodes_stopped_early"] * episodes),
                  round(measures["episodes_never_stopped"] * episodes))
        say(f"   check sheet {sheet}: recomputed 0.5 one picture {format_counts(check[sheet])}; fold file right/early/never {stored}")
    total_episodes = sum(tables[sheet]["episode_id"].nunique() for sheet in sheets)
    say(f"   pooled over {total_episodes} episodes on sheets {sheets}")
    for rule in RULES:
        pooled = {key: sum(grid[sheet][rule][key] for sheet in sheets) for key in ("right", "early", "between", "never")}
        low, high = wilson_interval(pooled["right"], total_episodes)
        early_low, early_high = wilson_interval(pooled["early"], total_episodes)
        per_sheet = "; ".join(f"{sheet} {format_counts(grid[sheet][rule])}" for sheet in sheets)
        say(f"   threshold {rule[0]:.2f}, {rule[1]} in a row: pooled {format_counts(pooled)} | right {pooled['right'] / total_episodes:.3f}"
            f" ({low:.3f} to {high:.3f}), early {pooled['early'] / total_episodes:.3f} ({early_low:.3f} to {early_high:.3f}) | {per_sheet}")
    for prefix, consecutive in (("tuned_", 1), ("tuned_two_", 2)):
        pooled = {"right": 0, "early": 0, "between": 0, "never": 0}
        parts = []
        for sheet in sheets:
            threshold = fold_measures[sheet][f"{prefix}stop_threshold"]
            result = episode_outcomes(tables[sheet], threshold, consecutive)
            for key in pooled:
                pooled[key] += result[key]
            parts.append(f"{sheet} at {threshold} {format_counts(result)}")
        low, high = wilson_interval(pooled["right"], total_episodes)
        say(f"   inner-validation tuned, {consecutive} in a row: pooled {format_counts(pooled)} | right {pooled['right'] / total_episodes:.3f}"
            f" ({low:.3f} to {high:.3f}), early {pooled['early'] / total_episodes:.3f} | " + "; ".join(parts))
    for label, early_cost in (("right minus early", 1.0), ("right minus twice early", 2.0)):
        pooled = {"right": 0, "early": 0, "between": 0, "never": 0}
        parts = []
        for sheet in sheets:
            others = [other for other in sheets if other != sheet]
            scores = {rule: sum(grid[other][rule]["right"] - early_cost * grid[other][rule]["early"] for other in others) for rule in RULES}
            best_rule = max(RULES, key = lambda rule: (scores[rule], -rule[1], rule[0]))
            result = grid[sheet][best_rule]
            for key in pooled:
                pooled[key] += result[key]
            parts.append(f"{sheet} rule {best_rule} {format_counts(result)}")
        low, high = wilson_interval(pooled["right"], total_episodes)
        say(f"   leave-one-sheet-out, criterion {label}: pooled {format_counts(pooled)} | right {pooled['right'] / total_episodes:.3f}"
            f" ({low:.3f} to {high:.3f}), early {pooled['early'] / total_episodes:.3f} | " + "; ".join(parts))
    pooled_choice = {rule: sum(grid[sheet][rule]["right"] - grid[sheet][rule]["early"] for sheet in sheets) for rule in RULES}
    best_pooled = max(RULES, key = lambda rule: (pooled_choice[rule], -rule[1], rule[0]))
    say(f"   one rule for all sheets chosen on all of them (right minus early, optimistic): {best_pooled}")

    # B. backward moves
    say("\nB. direction within 45 deg on backward moves and the others (moving rows: action is a direction)")
    pooled_back = [0, 0]
    pooled_other = [0, 0]
    for sheet in sheets:
        table = tables[sheet]
        table = table[table["action"].isin(DIRECTION_NAMES)]
        error = numpy.abs((table["angle_deg"] - table["move_angle_deg"] + 180.0) % 360.0 - 180.0).to_numpy()
        is_back = numpy.isin(nearest_direction(table["move_angle_deg"].to_numpy()), BACKWARD_NAMES)
        predicted_back = numpy.isin(nearest_direction(table["angle_deg"].to_numpy()), BACKWARD_NAMES)
        pooled_back[0] += int((error[is_back] <= 45).sum())
        pooled_back[1] += int(is_back.sum())
        pooled_other[0] += int((error[~is_back] <= 45).sum())
        pooled_other[1] += int((~is_back).sum())
        say(f"   sheet {sheet}: moving {len(table)}, backward {int(is_back.sum())}; within 45 on backward {(error[is_back] <= 45).mean():.3f},"
            f" on others {(error[~is_back] <= 45).mean():.3f}; all {(error <= 45).mean():.3f}; predicted a backward class on"
            f" {predicted_back.mean():.3f} of pictures against {is_back.mean():.3f} demonstrated")
    say(f"   pooled: backward {pooled_back[0]} of {pooled_back[1]} ({pooled_back[0] / pooled_back[1]:.3f}),"
        f" others {pooled_other[0]} of {pooled_other[1]} ({pooled_other[0] / pooled_other[1]:.3f})")

    # C. twist
    say("\nC. twist: least-squares gain through zero (demonstrated on predicted), sign on rows twisting 2 deg or more")
    for sheet in sheets:
        table = tables[sheet]
        table = table[table["action"] != "stop"]
        predicted = table["twist_deg"].to_numpy()
        demonstrated = table["forearm_change_deg"].fillna(0.0).clip(-15.0, 15.0).to_numpy()
        gain = float((predicted * demonstrated).sum() / (predicted * predicted).sum())
        twisted = numpy.abs(demonstrated) >= 2.0
        sign_right = float((numpy.sign(predicted[twisted]) == numpy.sign(demonstrated[twisted])).mean())
        confident = twisted & (numpy.abs(predicted) >= 3.0)
        confident_sign = float((numpy.sign(predicted[confident]) == numpy.sign(demonstrated[confident])).mean()) if confident.any() else float("nan")
        say(f"   sheet {sheet}: non-stop pictures {len(table)}; gain {gain:.2f}; twisted rows {int(twisted.sum())}, sign right {sign_right:.3f};"
            f" median |predicted| on twisted {numpy.median(numpy.abs(predicted[twisted])):.2f} against {numpy.median(numpy.abs(demonstrated[twisted])):.2f};"
            f" predicted 3 deg or more: {int(confident.sum())} rows, sign right {confident_sign:.3f}")

    # D. early stops at fixed 0.5, one picture
    say("\nD. early stops at the fixed 0.5 threshold, one picture: distance before the goal (cm)")
    for sheet in sheets:
        distances = grid[sheet][(0.5, 1)]["early_distances"]
        if distances:
            say(f"   sheet {sheet}: {len(distances)} early stops, median {numpy.median(distances):.1f} cm, range {min(distances):.1f} to {max(distances):.1f}")
        else:
            say(f"   sheet {sheet}: no early stops")
    OUTPUT_PATH.write_text("\n".join(output_lines) + "\n", encoding = "utf-8")


if __name__ == "__main__":
    main()
