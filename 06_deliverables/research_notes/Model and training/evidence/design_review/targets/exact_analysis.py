"""
What the end-to-end exact-step model predicts on held-out sheets 03, 05 and 06 (read only; inputs from
exact_inference.py): the predicted step length near the goal and elsewhere, the twist it predicts, how often it would
stand still away from the goal, and whether a stop rule that also asks for a short predicted step stops better.
Writes exact_analysis.out.txt next to this script.
"""

import numpy
import pandas

from common_frames import AUDIT_FOLDER, DIRECTION_NAMES, OUTPUT_FOLDER, read_frames

FOLDS = ["without_path_03", "without_path_05", "without_path_06"]
PROBABILITY_THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
LENGTH_THRESHOLDS_CM = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush = True)


def outcomes(table: pandas.DataFrame, passing: numpy.ndarray, consecutive: int) -> dict:
    counts = {"right": 0, "early": 0, "never": 0, "between": 0}
    for _, group in table.groupby("episode_id", sort = False):
        rows = group.index.to_numpy()
        flags = passing[rows]
        run = 0
        hit = None
        for position, flag in enumerate(flags):
            run = run + 1 if flag else 0
            if run >= consecutive:
                hit = position
                break
        if hit is None:
            counts["never"] += 1
            continue
        distance = float(group["goal_distance_cm"].iloc[hit])
        counts["right" if distance <= 2.0 else ("early" if distance > 3.0 else "between")] += 1
    total = sum(counts.values())
    return {name: value / total for name, value in counts.items()}


def main() -> None:
    frames = read_frames()
    view = pandas.read_csv(AUDIT_FOLDER / "view_and_tracks" / "s09_view_rows.csv")
    frames = frames.merge(view[["frame_index", "line_share_top_half"]], on = "frame_index", how = "left")
    frames["no_line_ahead"] = frames["line_share_top_half"].fillna(1.0) <= 0.002
    frames["goal_distance_cm"] = frames["goal_distance_cm"].fillna(0.0)
    keep = ["frame_index", "episode_id", "sheet", "row_in_episode", "goal_distance_cm", "action", "tip_forward_m", "tip_left_m",
            "forearm_change_deg", "lean_change_deg", "no_line_ahead", "waypoint"]
    tables = {}
    for fold in FOLDS:
        outputs = pandas.read_csv(OUTPUT_FOLDER / f"exact_outputs_{fold}.csv")
        table = outputs.merge(frames[keep], on = "frame_index", how = "left")
        table = table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
        table["predicted_cm"] = numpy.hypot(table["forward_cm"], table["left_cm"])
        table["demonstrated_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"].fillna(0.0), table["tip_left_m"].fillna(0.0))
        tables[fold] = table
    pooled = pandas.concat(tables.values(), ignore_index = True)
    moving = pooled["action"].isin(DIRECTION_NAMES)
    twist_in_place = (pooled["action"] == "small_motion") & (pooled["forearm_change_deg"].abs() >= 3.0)
    null_rows = (pooled["action"] == "small_motion") & (pooled["forearm_change_deg"].abs() < 1.0)
    stop = pooled["action"] == "stop"
    say(f"pictures {len(pooled)} in {pooled['episode_id'].nunique()} episodes (held-out sheets 03, 05, 06)")
    say("--- predicted against demonstrated step length (cm): median predicted / median demonstrated / share predicted under 0.3 cm")
    for name, mask in (("moving rows", moving), ("twists in place", twist_in_place), ("null rows (row 0, after corrections, duplicates)", null_rows), ("stop rows", stop)):
        part = pooled[mask]
        say(f"  {name}: n {len(part)}, {part['predicted_cm'].median():.3f} / {part['demonstrated_cm'].median():.3f} / {(part['predicted_cm'] < 0.3).mean():.3f}")
    say("  moving rows by demonstrated length: " + ", ".join(
        f"{low}-{high} cm: pred {pooled.loc[moving & pooled['demonstrated_cm'].between(low, high), 'predicted_cm'].median():.2f}"
        for low, high in ((0.3, 0.7), (0.7, 1.0), (1.0, 1.3), (1.3, 2.0))))
    bins = [-0.01, 1.0, 2.0, 3.0, 5.0, 10.0, 1000.0]
    pooled["goal_bin"] = pandas.cut(pooled["goal_distance_cm"], bins)
    profile = pooled[~stop].groupby("goal_bin", observed = True).agg(n = ("predicted_cm", "size"), predicted = ("predicted_cm", "median"),
                                                                       demonstrated = ("demonstrated_cm", "median"),
                                                                       short = ("predicted_cm", lambda values: float((values < 0.3).mean())))
    say("--- non-stop pictures by distance to the goal: n, median predicted cm, median demonstrated cm, share predicted under 0.3 cm")
    for interval, row in profile.iterrows():
        say(f"  {interval}: {int(row['n'])}, {row['predicted']:.2f}, {row['demonstrated']:.2f}, {row['short']:.3f}")
    far = (pooled["goal_distance_cm"] > 5.0) & ~stop
    for flag, name in ((True, "no line ahead"), (False, "line in view")):
        part = pooled[far & (pooled["no_line_ahead"] == flag)]
        say(f"  over 5 cm from the goal, {name}: n {len(part)}, median predicted {part['predicted_cm'].median():.2f} cm, "
            f"share under 0.3 cm {(part['predicted_cm'] < 0.3).mean():.3f}, median stop probability {part['stop_probability'].median():.4f}, "
            f"p90 {part['stop_probability'].quantile(0.9):.3f}")

    say("--- twist (deg) on rows where the demonstration twisted at least 2 deg")
    twisted = pooled[(pooled["forearm_change_deg"].abs() >= 2.0) & ~stop]
    say(f"  n {len(twisted)}; median |demonstrated| {twisted['forearm_change_deg'].abs().median():.2f}, median |predicted| {twisted['twist_deg'].abs().median():.2f}; "
        f"sign right {(numpy.sign(twisted['twist_deg']) == numpy.sign(twisted['forearm_change_deg'])).mean():.3f}; "
        f"predicted |twist| under 1 deg {(twisted['twist_deg'].abs() < 1.0).mean():.3f}; correlation {numpy.corrcoef(twisted['twist_deg'], twisted['forearm_change_deg'])[0, 1]:.3f}")
    in_place = pooled[twist_in_place]
    say(f"  twists in place: n {len(in_place)}; median |predicted twist| {in_place['twist_deg'].abs().median():.2f} against {in_place['forearm_change_deg'].abs().median():.2f}; "
        f"sign right {(numpy.sign(in_place['twist_deg']) == numpy.sign(in_place['forearm_change_deg'])).mean():.3f}; predicted step median {in_place['predicted_cm'].median():.2f} cm")
    straight = pooled[(pooled["forearm_change_deg"].abs() < 1.0) & moving]
    say(f"  moving rows without a twist (under 1 deg): n {len(straight)}; median |predicted twist| {straight['twist_deg'].abs().median():.2f}; "
        f"predicted |twist| over 3 deg {(straight['twist_deg'].abs() > 3.0).mean():.3f}")

    say("--- standing still away from the goal (predicted step under 0.3 cm and |twist| under 1 deg, more than 3 cm from the goal)")
    away = (pooled["goal_distance_cm"] > 3.0) & ~stop
    still = away & (pooled["predicted_cm"] < 0.3) & (pooled["twist_deg"].abs() < 1.0)
    episodes_with = pooled[still]["episode_id"].nunique()
    say(f"  {int(still.sum())} of {int(away.sum())} pictures ({still.sum() / away.sum():.4f}); in {episodes_with} of {pooled['episode_id'].nunique()} episodes; "
        f"of these pictures {pooled.loc[still, 'no_line_ahead'].mean():.2f} show no line ahead; demonstrated action there: "
        f"{pooled.loc[still, 'action'].value_counts().head(5).to_dict()}")
    still_any = away & (pooled["predicted_cm"] < 0.3)
    say(f"  predicted step under 0.3 cm (any twist): {int(still_any.sum())} pictures ({still_any.sum() / away.sum():.4f})")

    say("--- stop rules per held-out sheet (right / early / never); thresholds chosen on the other two sheets")
    rules = {}
    for fold, table in tables.items():
        probability = table["stop_probability"].to_numpy()
        length = table["predicted_cm"].to_numpy()
        for consecutive in (1, 2):
            for threshold in PROBABILITY_THRESHOLDS:
                rules[("probability", consecutive, threshold, None, fold)] = outcomes(table, probability >= threshold, consecutive)
            for cut in LENGTH_THRESHOLDS_CM:
                rules[("length", consecutive, None, cut, fold)] = outcomes(table, length < cut, consecutive)
                for threshold in PROBABILITY_THRESHOLDS:
                    rules[("both", consecutive, threshold, cut, fold)] = outcomes(table, (probability >= threshold) & (length < cut), consecutive)
    for kind in ("probability", "length", "both"):
        for consecutive in (1, 2):
            keys = sorted({(key[2], key[3]) for key in rules if key[0] == kind and key[1] == consecutive}, key = lambda item: (item[0] or 0, item[1] or 0))
            chosen_results, best_results = [], []
            for fold in FOLDS:
                others = [other for other in FOLDS if other != fold]
                def pooled_score(parameters):
                    right = numpy.mean([rules[(kind, consecutive, parameters[0], parameters[1], other)]["right"] for other in others])
                    early = numpy.mean([rules[(kind, consecutive, parameters[0], parameters[1], other)]["early"] for other in others])
                    return (right, -early)
                chosen = max(keys, key = pooled_score)
                chosen_results.append((fold, chosen, rules[(kind, consecutive, chosen[0], chosen[1], fold)]))
                best = max(keys, key = lambda parameters: (rules[(kind, consecutive, parameters[0], parameters[1], fold)]["right"],
                                                           -rules[(kind, consecutive, parameters[0], parameters[1], fold)]["early"]))
                best_results.append((fold, best, rules[(kind, consecutive, best[0], best[1], fold)]))
            for label, results in (("chosen on the other sheets", chosen_results), ("best on own sheet", best_results)):
                right = [result[2]["right"] for result in results]
                early = [result[2]["early"] for result in results]
                never = [result[2]["never"] for result in results]
                say(f"  {kind}, {consecutive} in a row, {label}: right {numpy.mean(right):.3f} ({min(right):.2f} to {max(right):.2f}), "
                    f"early {numpy.mean(early):.3f} ({min(early):.2f} to {max(early):.2f}), never {numpy.mean(never):.3f}; "
                    + " ".join(f"{result[0][-2:]}:{result[2]['right']:.2f}/{result[2]['early']:.2f}/{result[2]['never']:.2f} at {result[1]}" for result in results))
    (OUTPUT_FOLDER / "exact_analysis.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
