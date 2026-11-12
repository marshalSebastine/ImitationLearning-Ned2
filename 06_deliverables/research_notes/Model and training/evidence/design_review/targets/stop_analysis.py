"""
Stop decision analysis on the saved held-out predictions (read only).

For every saved prediction set (frozen probes, six held-out sheets; end-to-end runs, three held-out sheets) this
script replays the per-episode stop rule of training_metrics.stop_metrics (stop at the first picture where the stop
probability has passed the threshold on k pictures in a row; right within 2 cm of where the demonstration stopped, early
more than 3 cm before it, never when no picture passes) for k = 1, 2 and 3 and a grid of thresholds, and reports:
    best      the best threshold on the held-out sheet itself (an optimistic ceiling: it shows how well the stop output
              ranks the pictures, whatever its calibration);
    sheets    the threshold chosen on the other held-out sheets of the same run (each fold's threshold never sees its own
              sheet), a stand-in for choosing the threshold on unseen tracks;
and, for the early stops, where they happened: the distance to the goal, the share of the episode, and whether the
picture shows no line in its top half (the audit's line_share_top_half from view_and_tracks/s09_view_rows.csv).
Writes stop_analysis.out.txt and stop_analysis_summary.csv next to this script.
"""

import json
from pathlib import Path

import numpy
import pandas

from common_frames import AUDIT_FOLDER, OUTPUT_FOLDER, RUNS_FOLDER, read_frames

PROBE_FOLDER = RUNS_FOLDER / "2026-11-09_frozen_feature_probes" / "predictions"
FOLDS = [f"without_path_0{number}" for number in range(1, 7)]
THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SOURCES = {
    "probe exact, stop on stop picture": ("probe", "exact_stop_head_on_stop_picture"),
    "probe exact seed 11": ("probe", "exact_seed_11"),
    "probe exact seed 12": ("probe", "exact_seed_12"),
    "probe exact seed 13": ("probe", "exact_seed_13"),
    "probe exact, stop near goal": ("probe", "exact_stop_head_on_near_goal"),
    "probe exact, goal distance": ("probe", "exact_stop_on_goal_distance"),
    "probe chunk of 5": ("probe", "chunk_of_5_steps"),
    "probe mixed": ("probe", "mixed_classes_and_exact"),
    "probe classes, stop as class": ("probe", "classes_stop_as_class"),
    "probe classes, own stop on stop picture": ("probe", "classes_stop_head_on_stop_picture"),
    "probe classes, own stop near goal": ("probe", "classes_stop_head_on_near_goal"),
    "probe classes, goal distance": ("probe", "classes_stop_head_on_goal_distance"),
    "end-to-end ResNet-18 exact": ("run", "2026-11-09_resnet18_finetune_exact"),
    "end-to-end ResNet-18 classes": ("run", "2026-11-09_resnet18_finetune_classes"),
    "end-to-end small CNN classes": ("run", "2026-11-09_small_cnn_scratch_classes"),
    "end-to-end small CNN line and puck": ("run", "2026-11-09_small_cnn_line_and_puck_classes"),
    "end-to-end final candidate": ("run", "2026-11-09_resnet18_final_candidate")
}

lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush = True)


def load_fold(kind: str, name: str, fold: str):
    path = (PROBE_FOLDER / f"{name}_{fold}.npz") if kind == "probe" else (RUNS_FOLDER / name / f"predictions_{fold}.npz")
    if not path.exists():
        return None
    data = numpy.load(path)
    return {"frame_index": data["frame_index"], "stop_probability": data["stop_probability"]}


def episode_outcomes(table: pandas.DataFrame, threshold: float, consecutive: int) -> pandas.DataFrame:
    """Return per episode: the outcome (right, early, between, never) and the first passing row."""

    results = []
    for episode_id, group in table.groupby("episode_id", sort = False):
        passing = group["stop_probability"].to_numpy() >= threshold
        run = numpy.zeros(len(passing), dtype = int)
        for position in range(len(passing)):
            run[position] = (run[position - 1] + 1 if position > 0 else 1) if passing[position] else 0
        hits = numpy.flatnonzero(run >= consecutive)
        if len(hits) == 0:
            results.append({"episode_id": episode_id, "outcome": "never", "position": -1})
            continue
        row = group.iloc[hits[0]]
        distance = float(row["goal_distance_cm"])
        outcome = "right" if distance <= 2.0 else ("early" if distance > 3.0 else "between")
        results.append({"episode_id": episode_id, "outcome": outcome, "position": int(hits[0]), "distance": distance,
                        "step_fraction": float(row["step_fraction"]), "no_line_ahead": bool(row["no_line_ahead"]),
                        "frame_index": int(row["frame_index"]), "sheet": row["sheet"]})
    return pandas.DataFrame(results)


def shares(outcomes: pandas.DataFrame) -> dict:
    counts = outcomes["outcome"].value_counts()
    total = len(outcomes)
    return {name: counts.get(name, 0) / total for name in ("right", "early", "never", "between")} | {"episodes": total}


def main() -> None:
    frames = read_frames()
    view = pandas.read_csv(AUDIT_FOLDER / "view_and_tracks" / "s09_view_rows.csv")
    frames = frames.merge(view[["frame_index", "line_share_top_half"]], on = "frame_index", how = "left")
    frames["no_line_ahead"] = frames["line_share_top_half"].fillna(1.0) <= 0.002
    frames["goal_distance_cm"] = frames["goal_distance_cm"].fillna(0.0)
    columns = ["frame_index", "episode_id", "sheet", "row_in_episode", "goal_distance_cm", "step_fraction", "no_line_ahead", "action"]
    lookup = frames[columns].set_index("frame_index")
    far = frames["goal_distance_cm"] > 3.0
    say(f"no line in the top half (line_share_top_half <= 0.002): {int(frames['no_line_ahead'].sum())} of {len(frames)} pictures; "
        f"among pictures more than 3 cm from the goal {frames.loc[far, 'no_line_ahead'].mean():.3f}; among stop pictures "
        f"{frames.loc[frames['action'] == 'stop', 'no_line_ahead'].mean():.3f}")
    say("no line ahead, share of pictures more than 3 cm from the goal, by sheet: " + ", ".join(
        f"{sheet[-2:]} {group.loc[group['goal_distance_cm'] > 3.0, 'no_line_ahead'].mean():.3f}" for sheet, group in frames.groupby("sheet")))

    summary_rows = []
    for label, (kind, name) in SOURCES.items():
        tables = {}
        for fold in FOLDS:
            loaded = load_fold(kind, name, fold)
            if loaded is None:
                continue
            table = lookup.loc[loaded["frame_index"]].reset_index()
            table["stop_probability"] = loaded["stop_probability"]
            table = table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
            tables[fold] = table
        if not tables:
            say(f"=== {label}: no predictions yet")
            continue
        say(f"=== {label} ({len(tables)} folds: {', '.join(fold[-2:] for fold in tables)})")
        grid = {}
        for fold, table in tables.items():
            for consecutive in (1, 2, 3):
                for threshold in THRESHOLDS:
                    grid[(fold, consecutive, threshold)] = episode_outcomes(table, threshold, consecutive)
        for consecutive in (1, 2, 3):
            best_rows, sheet_rows, fixed_rows = [], [], []
            for fold in tables:
                options = [(shares(grid[(fold, consecutive, threshold)]), threshold) for threshold in THRESHOLDS]
                best, best_threshold = max(options, key = lambda item: (item[0]["right"], -item[0]["early"]))
                best_rows.append((fold, best_threshold, best))
                other_folds = [other for other in tables if other != fold]
                if other_folds:
                    def pooled(threshold):
                        outcome = pandas.concat([grid[(other, consecutive, threshold)] for other in other_folds])
                        return shares(outcome)
                    chosen = max(THRESHOLDS, key = lambda threshold: (pooled(threshold)["right"], -pooled(threshold)["early"]))
                    sheet_rows.append((fold, chosen, shares(grid[(fold, consecutive, chosen)])))
                fixed_rows.append((fold, 0.5, shares(grid[(fold, consecutive, 0.5)])))
            for rule, rows in (("best on own sheet", best_rows), ("chosen on other sheets", sheet_rows), ("fixed 0.5", fixed_rows)):
                if not rows:
                    continue
                right = [row[2]["right"] for row in rows]
                early = [row[2]["early"] for row in rows]
                never = [row[2]["never"] for row in rows]
                say(f"  {consecutive} in a row, {rule}: right {numpy.mean(right):.3f} ({min(right):.2f} to {max(right):.2f}), "
                    f"early {numpy.mean(early):.3f} ({min(early):.2f} to {max(early):.2f}), never {numpy.mean(never):.3f}; "
                    f"thresholds {[row[1] for row in rows]}; per fold right/early "
                    + " ".join(f"{row[0][-2:]}:{row[2]['right']:.2f}/{row[2]['early']:.2f}" for row in rows))
                summary_rows.append({"source": label, "consecutive": consecutive, "rule": rule, "folds": len(rows),
                                     "right_mean": round(float(numpy.mean(right)), 4), "right_min": min(right), "right_max": max(right),
                                     "early_mean": round(float(numpy.mean(early)), 4), "early_min": min(early), "early_max": max(early),
                                     "never_mean": round(float(numpy.mean(never)), 4),
                                     "thresholds": json.dumps([row[1] for row in rows])})
        # where the early stops happen, with the threshold chosen on the other sheets, two in a row
        early_tables = []
        base_tables = []
        for fold, table in tables.items():
            other_folds = [other for other in tables if other != fold] or [fold]
            def pooled_two(threshold):
                return shares(pandas.concat([grid[(other, 2, threshold)] for other in other_folds]))
            chosen = max(THRESHOLDS, key = lambda threshold: (pooled_two(threshold)["right"], -pooled_two(threshold)["early"]))
            outcome = grid[(fold, 2, chosen)]
            early_tables.append(outcome[outcome["outcome"] == "early"])
            base_tables.append(table[table["goal_distance_cm"] > 3.0])
        early = pandas.concat(early_tables)
        base = pandas.concat(base_tables)
        if len(early):
            say(f"  early stops (2 in a row, threshold from the other sheets): {len(early)} episodes; distance to goal median "
                f"{early['distance'].median():.1f} cm (p25 {early['distance'].quantile(0.25):.1f}, p75 {early['distance'].quantile(0.75):.1f}); "
                f"share of the episode median {early['step_fraction'].median():.2f}; no line ahead at the stop picture "
                f"{early['no_line_ahead'].mean():.2f} against {base['no_line_ahead'].mean():.2f} of all pictures over 3 cm from the goal; "
                f"by sheet {early['sheet'].str[-2:].value_counts().sort_index().to_dict()}")
        # stop probability by distance to the goal, and by line in view
        pooled_table = pandas.concat(tables.values())
        bins = [-0.01, 1.0, 2.0, 3.0, 5.0, 10.0, 1000.0]
        pooled_table["goal_bin"] = pandas.cut(pooled_table["goal_distance_cm"], bins)
        profile = pooled_table.groupby("goal_bin", observed = True)["stop_probability"].median()
        say("  median stop probability by distance to the goal (cm): " + ", ".join(f"{interval}: {value:.3f}" for interval, value in profile.items()))
        far_table = pooled_table[pooled_table["goal_distance_cm"] > 5.0]
        say(f"  pictures over 5 cm from the goal: median stop probability with no line ahead {far_table.loc[far_table['no_line_ahead'], 'stop_probability'].median():.3f} "
            f"(n {int(far_table['no_line_ahead'].sum())}) against {far_table.loc[~far_table['no_line_ahead'], 'stop_probability'].median():.3f} "
            f"(n {int((~far_table['no_line_ahead']).sum())}); p90 {far_table.loc[far_table['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f} "
            f"against {far_table.loc[~far_table['no_line_ahead'], 'stop_probability'].quantile(0.9):.3f}")
        stop_rows = pooled_table[pooled_table["action"] == "stop"]
        say(f"  stop pictures: median stop probability {stop_rows['stop_probability'].median():.3f} (n {len(stop_rows)})")
    pandas.DataFrame(summary_rows).to_csv(OUTPUT_FOLDER / "stop_analysis_summary.csv", index = False)
    (OUTPUT_FOLDER / "stop_analysis.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
