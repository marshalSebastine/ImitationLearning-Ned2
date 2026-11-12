"""
Numbers for the validation section of the training design (Team M, 9.11.26).

Reads only (never writes): frames.csv of the v01 training index, and the result and prediction files in 04_runs of the
training-analysis worktree. Run from this folder with the project's virtual environment, isolated and without byte
code, so that nothing is written anywhere else:

    D:/AI_Project/.venv/Scripts/python.exe -I -B validation_numbers.py > validation_numbers_output.txt

Parts
-----
1. The held-out-episode and held-out-sheet predictions re-scored sheet by sheet (checked against the stored fold
   numbers first), with each sheet's always-forward rule on the same pictures.
2. How much a held-out-sheet number moves with the sampled episodes (episode bootstrap) and with the seed.
3. Paired model differences on the same pictures, with episode-bootstrap intervals.
4. The pictures after a hand correction (recovery pictures) scored apart from the rest.
5. Steps per centimetre of line in full runs, for a step budget.
6. End-to-end fold measures and training histories.
7. Robot-trial statistics: intervals, tests, choosing between two models, clustering, robot time.
"""

import json
import math
from pathlib import Path

import numpy
import pandas


ANALYSIS_PARAMETERS = {
    "frames_path": Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv"),
    "runs_folder": Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs"),
    "direction_names": ["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"],
    "threshold_deg": 45.0,
    "stop_threshold": 0.5,
    "right_stop_cm": 2.0,
    "early_stop_cm": 3.0,
    "bootstrap_count": 4000,
    "seed": 91126,
    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"],
    # cap-to-cap line lengths, audit table 10.1 (traced lines of 2026-11-06_sheet_measurements.json)
    "line_length_cm": {"path_01": 23.3, "path_02": 31.3, "path_03": 50.7, "path_04": 72.4, "path_05": 56.6, "path_06": 46.5},
    # minutes per trial, main_session_facts.md: median demonstration steps x 1.22 s, plus a guessed 1 minute set-up
    "trial_minutes": {"path_01": 1.6, "path_02": 2.4, "path_03": 3.2, "path_04": 4.4, "path_05": 3.1, "path_06": 2.7},
    "median_steps": {"path_01": 31.5, "path_02": 66.0, "path_03": 106.0, "path_04": 168.0, "path_05": 103.5, "path_06": 82.0},
    "seconds_per_step": 1.22
}



def angle_error_deg(first_deg: numpy.ndarray, second_deg: numpy.ndarray) -> numpy.ndarray:
    """Return the absolute difference of two angles in degrees, between 0 and 180 (as training_metrics.py)."""

    return numpy.abs((numpy.asarray(first_deg) - numpy.asarray(second_deg) + 180.0) % 360.0 - 180.0)



def load_frames(parameters: dict) -> pandas.DataFrame:
    """
    Read the columns of frames.csv this analysis needs.

    Returns
    -------
    pandas.DataFrame
        Indexed by frame_index, in episode and step order, with "is_moving" (the rows the direction measures score,
        as training_probes.training_arrays) and "corrections_before" (how many hand corrections came earlier in the
        episode, from the restart of the waypoint count, as the audit's waypoint marker).
    """

    columns = ["episode_id", "episode_folder", "sheet", "start_kind", "start_end", "score", "is_scored", "action",
               "waypoint", "row_in_episode", "goal_distance_cm", "move_angle_deg", "frame_index", "is_mistake_row",
               "lighting", "step_count", "seconds_to_next_row"]
    frames = pandas.read_csv(parameters["frames_path"], usecols = columns, low_memory = False)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    is_restart = ((frames["waypoint"].fillna(-1) == 0) & (frames["row_in_episode"] > 0)).astype(int)
    frames["corrections_before"] = is_restart.groupby(frames["episode_id"]).cumsum()
    frames["is_moving"] = frames["action"].isin(parameters["direction_names"])
    return frames.set_index("frame_index", drop = False)



def prediction_table(path: Path, frames: pandas.DataFrame) -> pandas.DataFrame:
    """Return one prediction file's rows joined with their frames, in episode and step order."""

    with numpy.load(path) as archive:
        table = pandas.DataFrame({"frame_index": archive["frame_index"]})
        for key in ("angle_deg", "stop_probability"):
            if key in archive.files:
                table[key] = archive[key]
    table = table.join(frames.drop(columns = ["frame_index"]), on = "frame_index")
    return table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)



def direction_share(table: pandas.DataFrame, parameters: dict, angle_column: str = "angle_deg") -> tuple:
    """Return the share of moving rows within the threshold, and the number of moving rows."""

    moving = table[table["is_moving"]]
    if len(moving) == 0:
        return (float("nan"), 0)
    errors = angle_error_deg(moving["move_angle_deg"].to_numpy(), moving[angle_column].to_numpy())
    return (float((errors <= parameters["threshold_deg"]).mean()), int(len(moving)))



def episode_hits(table: pandas.DataFrame, parameters: dict, angle_column: str = "angle_deg") -> pandas.DataFrame:
    """Return per episode the moving rows within the threshold ("hits") and the moving rows ("rows")."""

    moving = table[table["is_moving"]].copy()
    moving["hit"] = (angle_error_deg(moving["move_angle_deg"].to_numpy(), moving[angle_column].to_numpy())
                     <= parameters["threshold_deg"]).astype(int)
    grouped = moving.groupby("episode_id")
    return pandas.DataFrame({"hits": grouped["hit"].sum(), "rows": grouped["hit"].size(),
                             "sheet": grouped["sheet"].first()})



def stop_outcomes(table: pandas.DataFrame, parameters: dict) -> dict:
    """
    Score the stop per episode as training_metrics.stop_metrics does with one picture past the threshold.

    Returns
    -------
    dict
        Counts "right", "early", "never" and "episodes".
    """

    counts = {"right": 0, "early": 0, "never": 0, "episodes": 0}
    for _, episode in table.groupby("episode_id", sort = False):
        counts["episodes"] += 1
        passing = episode[episode["stop_probability"].to_numpy() >= parameters["stop_threshold"]]
        if len(passing) == 0:
            counts["never"] += 1
            continue
        distance = float(numpy.nan_to_num(passing["goal_distance_cm"].iloc[0], nan = 0.0))
        if distance <= parameters["right_stop_cm"]:
            counts["right"] += 1
        elif distance > parameters["early_stop_cm"]:
            counts["early"] += 1
    return counts



def bootstrap_shares(hits: numpy.ndarray, rows: numpy.ndarray, count: int, generator: numpy.random.Generator) -> numpy.ndarray:
    """Return ``count`` pooled shares from episodes drawn with replacement."""

    picks = generator.integers(0, len(hits), size = (count, len(hits)))
    return hits[picks].sum(axis = 1) / rows[picks].sum(axis = 1)



def interval_text(values: numpy.ndarray) -> str:
    """Return the 2.5 and 97.5 percentiles as text."""

    return f"{numpy.percentile(values, 2.5):.3f} to {numpy.percentile(values, 97.5):.3f}"



def probe_folds(file_name: str, result_name: str, parameters: dict) -> list:
    """Return the stored fold measures of one probe variant."""

    path = parameters["runs_folder"] / "2026-11-09_frozen_feature_probes" / file_name
    for result in json.loads(path.read_text(encoding = "utf-8"))["results"]:
        if result["name"] == result_name:
            return result["folds"]
    raise KeyError(result_name)



def part_one_splits(frames: pandas.DataFrame, parameters: dict) -> dict:
    """Re-score the split experiment sheet by sheet; return the tables for the later parts."""

    print("\n=== 1. Held-out episodes against held-out sheets, sheet by sheet (splits experiment, frozen ResNet-18 grid) ===")
    folder = parameters["runs_folder"] / "2026-11-09_frozen_feature_probes" / "predictions"
    tables = {}
    for design in ("classes", "exact"):
        print(f"\n--- design: {design}")
        stored_episodes = probe_folds("splits.json", f"{design}_held_out_episodes", parameters)
        stored_sheets = probe_folds("splits.json", f"{design}_held_out_sheets", parameters)
        episode_parts = []
        print("check against stored folds (recomputed / stored): within 45, direction rows, stopped right/early/never")
        for fold in stored_episodes:
            part = prediction_table(folder / f"{design}_held_out_episodes_{fold['fold']}.npz", frames)
            episode_parts.append(part)
            share, rows = direction_share(part, parameters)
            stops = stop_outcomes(part, parameters)
            print(f"  {fold['fold']}: {share:.4f} / {fold['within_45_deg']}, {rows} / {fold['direction_rows']}, "
                  f"{stops['right'] / stops['episodes']:.4f}/{stops['early'] / stops['episodes']:.4f}/{stops['never'] / stops['episodes']:.4f}"
                  f" / {fold['episodes_stopped_right']}/{fold['episodes_stopped_early']}/{fold['episodes_never_stopped']}, episodes {stops['episodes']}")
        sheet_parts = []
        for fold in stored_sheets:
            part = prediction_table(folder / f"{design}_held_out_sheets_{fold['fold']}.npz", frames)
            sheet_parts.append(part)
            share, rows = direction_share(part, parameters)
            stops = stop_outcomes(part, parameters)
            print(f"  {fold['fold']}: {share:.4f} / {fold['within_45_deg']}, {rows} / {fold['direction_rows']}, "
                  f"{stops['right'] / stops['episodes']:.4f}/{stops['early'] / stops['episodes']:.4f}/{stops['never'] / stops['episodes']:.4f}"
                  f" / {fold['episodes_stopped_right']}/{fold['episodes_stopped_early']}/{fold['episodes_never_stopped']}, episodes {stops['episodes']}")
        episodes_table = pandas.concat(episode_parts, ignore_index = True)
        sheets_table = pandas.concat(sheet_parts, ignore_index = True)
        tables[design] = {"episodes": episodes_table, "sheets": sheets_table}
        print("fold spread of the stored numbers (within 45): episodes "
              f"{min(f['within_45_deg'] for f in stored_episodes):.3f} to {max(f['within_45_deg'] for f in stored_episodes):.3f}; "
              f"sheets {min(f['within_45_deg'] for f in stored_sheets):.3f} to {max(f['within_45_deg'] for f in stored_sheets):.3f}")
        print("fold spread of the stored numbers (path within 45): episodes "
              f"{min(f['path_within_45_deg'] for f in stored_episodes):.3f} to {max(f['path_within_45_deg'] for f in stored_episodes):.3f}; "
              f"sheets {min(f['path_within_45_deg'] for f in stored_sheets):.3f} to {max(f['path_within_45_deg'] for f in stored_sheets):.3f}")
        print("fold spread of the stored numbers (stopped right, fixed 0.5): episodes "
              f"{min(f['episodes_stopped_right'] for f in stored_episodes):.3f} to {max(f['episodes_stopped_right'] for f in stored_episodes):.3f}; "
              f"sheets {min(f['episodes_stopped_right'] for f in stored_sheets):.3f} to {max(f['episodes_stopped_right'] for f in stored_sheets):.3f}")

        print("\nper sheet: moving rows (episodes split / sheets split), within 45 held-out episodes, within 45 held-out sheet, "
              "gap, always forward on the same rows, stopped right/early/never (episodes split; sheets split) as counts of episodes")
        gaps = []
        episode_values = []
        sheet_values = []
        forward_values = []
        for sheet in parameters["sheets"]:
            episode_rows = episodes_table[episodes_table["sheet"] == sheet]
            sheet_rows = sheets_table[sheets_table["sheet"] == sheet]
            episode_share, episode_count = direction_share(episode_rows, parameters)
            sheet_share, sheet_count = direction_share(sheet_rows, parameters)
            forward_rows = sheet_rows.assign(forward_deg = 0.0)
            forward_share, _ = direction_share(forward_rows, parameters, "forward_deg")
            episode_stops = stop_outcomes(episode_rows, parameters)
            sheet_stops = stop_outcomes(sheet_rows, parameters)
            gaps.append(episode_share - sheet_share)
            episode_values.append(episode_share)
            sheet_values.append(sheet_share)
            forward_values.append(forward_share)
            print(f"  {sheet}: rows {episode_count}/{sheet_count}, episodes {episode_stops['episodes']}/{sheet_stops['episodes']}: "
                  f"{episode_share:.3f} {sheet_share:.3f} gap {100 * (episode_share - sheet_share):+.1f} points; always forward {forward_share:.3f}; "
                  f"stops {episode_stops['right']}/{episode_stops['early']}/{episode_stops['never']} of {episode_stops['episodes']}; "
                  f"{sheet_stops['right']}/{sheet_stops['early']}/{sheet_stops['never']} of {sheet_stops['episodes']}")
        print(f"  six-sheet means: episodes split {numpy.mean(episode_values):.3f}, sheets split {numpy.mean(sheet_values):.3f}, "
              f"always forward {numpy.mean(forward_values):.3f}; mean gap {100 * numpy.mean(gaps):+.1f} points "
              f"(smallest {100 * min(gaps):+.1f}, largest {100 * max(gaps):+.1f})")
        pooled_episode, pooled_episode_rows = direction_share(episodes_table, parameters)
        pooled_sheet, pooled_sheet_rows = direction_share(sheets_table, parameters)
        print(f"  pooled over all pictures: episodes split {pooled_episode:.3f} ({pooled_episode_rows} rows), "
              f"sheets split {pooled_sheet:.3f} ({pooled_sheet_rows} rows)")
        total_episode_stops = stop_outcomes(episodes_table, parameters)
        total_sheet_stops = stop_outcomes(sheets_table, parameters)
        print(f"  stops pooled over all episodes: episodes split right {total_episode_stops['right']}, early {total_episode_stops['early']}, "
              f"never {total_episode_stops['never']} of {total_episode_stops['episodes']}; sheets split right {total_sheet_stops['right']}, "
              f"early {total_sheet_stops['early']}, never {total_sheet_stops['never']} of {total_sheet_stops['episodes']}")
    return tables



def part_two_noise(tables: dict, frames: pandas.DataFrame, parameters: dict) -> None:
    """Episode-bootstrap intervals of held-out-sheet numbers, and the spread between seeds."""

    print("\n=== 2. How much a held-out-sheet number moves: sampled episodes and seeds ===")
    generator = numpy.random.default_rng(parameters["seed"])
    for design in ("classes", "exact"):
        per_episode = episode_hits(tables[design]["sheets"], parameters)
        sheet_draws = []
        print(f"\n--- design: {design}, held-out sheets, within 45, 95 % episode-bootstrap interval ({parameters['bootstrap_count']} draws)")
        for sheet in parameters["sheets"]:
            part = per_episode[per_episode["sheet"] == sheet]
            draws = bootstrap_shares(part["hits"].to_numpy(), part["rows"].to_numpy(), parameters["bootstrap_count"], generator)
            sheet_draws.append(draws)
            value = part["hits"].sum() / part["rows"].sum()
            print(f"  {sheet}: {value:.3f}, interval {interval_text(draws)} ({len(part)} episodes), half-width about "
                  f"{100 * (numpy.percentile(draws, 97.5) - numpy.percentile(draws, 2.5)) / 2:.1f} points")
        mean_draws = numpy.mean(numpy.vstack(sheet_draws), axis = 0)
        values = numpy.array([per_episode[per_episode["sheet"] == sheet]["hits"].sum() / per_episode[per_episode["sheet"] == sheet]["rows"].sum()
                              for sheet in parameters["sheets"]])
        standard_deviation = float(numpy.std(values, ddof = 1))
        t_quantile = 2.5706
        print(f"  six-sheet mean {values.mean():.3f}: episode-bootstrap interval {interval_text(mean_draws)}; "
              f"between-sheet standard deviation {standard_deviation:.3f}, t interval over the six sheets "
              f"{values.mean() - t_quantile * standard_deviation / math.sqrt(6):.3f} to {values.mean() + t_quantile * standard_deviation / math.sqrt(6):.3f}")

    print("\n--- seeds: held-out sheets, within 45 per sheet (default seed, then seeds 11, 12, 13), range per sheet")
    for design in ("classes", "exact"):
        variants = [probe_folds("splits.json", f"{design}_held_out_sheets", parameters)] + \
                   [probe_folds("seeds.json", f"{design}_seed_{seed}", parameters) for seed in (11, 12, 13)]
        print(f"  design {design}:")
        means = [numpy.mean([fold["within_45_deg"] for fold in variant]) for variant in variants]
        path_means = [numpy.mean([fold["path_within_45_deg"] for fold in variant]) for variant in variants]
        stop_means = [numpy.mean([fold["episodes_stopped_right"] for fold in variant]) for variant in variants]
        for position, sheet in enumerate(parameters["sheets"]):
            values = [variant[position]["within_45_deg"] for variant in variants]
            path_values = [variant[position]["path_within_45_deg"] for variant in variants]
            stop_values = [variant[position]["episodes_stopped_right"] for variant in variants]
            print(f"    {sheet}: step {' '.join(f'{value:.3f}' for value in values)} (range {100 * (max(values) - min(values)):.1f} points); "
                  f"path {' '.join(f'{value:.3f}' for value in path_values)} (range {100 * (max(path_values) - min(path_values)):.1f}); "
                  f"stopped right {' '.join(f'{value:.3f}' for value in stop_values)} (range {100 * (max(stop_values) - min(stop_values)):.1f})")
        print(f"    six-sheet means: step {' '.join(f'{value:.3f}' for value in means)} (range {100 * (max(means) - min(means)):.1f} points); "
              f"path {' '.join(f'{value:.3f}' for value in path_means)} (range {100 * (max(path_means) - min(path_means)):.1f}); "
              f"stopped right {' '.join(f'{value:.3f}' for value in stop_means)} (range {100 * (max(stop_means) - min(stop_means)):.1f})")



def paired_difference(first: pandas.DataFrame, second: pandas.DataFrame, parameters: dict, generator: numpy.random.Generator) -> tuple:
    """
    Return the difference second minus first in the within-threshold share on the same moving rows, with a
    paired episode-bootstrap interval, and the number of episodes.
    """

    joined = first[["frame_index", "episode_id", "sheet", "is_moving", "move_angle_deg", "angle_deg"]].merge(
        second[["frame_index", "angle_deg"]], on = "frame_index", suffixes = ("_first", "_second"))
    joined = joined[joined["is_moving"]]
    true_angle = joined["move_angle_deg"].to_numpy()
    joined["hit_first"] = (angle_error_deg(true_angle, joined["angle_deg_first"].to_numpy()) <= parameters["threshold_deg"]).astype(int)
    joined["hit_second"] = (angle_error_deg(true_angle, joined["angle_deg_second"].to_numpy()) <= parameters["threshold_deg"]).astype(int)
    grouped = joined.groupby("episode_id")
    hits_first = grouped["hit_first"].sum().to_numpy()
    hits_second = grouped["hit_second"].sum().to_numpy()
    rows = grouped["hit_first"].size().to_numpy()
    difference = (hits_second.sum() - hits_first.sum()) / rows.sum()
    picks = generator.integers(0, len(rows), size = (parameters["bootstrap_count"], len(rows)))
    draws = (hits_second[picks].sum(axis = 1) - hits_first[picks].sum(axis = 1)) / rows[picks].sum(axis = 1)
    return difference, draws, len(rows), int(rows.sum())



def part_three_paired(tables: dict, frames: pandas.DataFrame, parameters: dict) -> None:
    """Paired differences between models on the same pictures."""

    print("\n=== 3. Paired differences on the same pictures (second minus first, within 45), episode-bootstrap 95 % intervals ===")
    generator = numpy.random.default_rng(parameters["seed"] + 1)
    probes = parameters["runs_folder"] / "2026-11-09_frozen_feature_probes" / "predictions"
    comparisons = []
    for sheet in parameters["sheets"]:
        comparisons.append((f"probe exact - probe classes, {sheet}",
                            tables["classes"]["sheets"][tables["classes"]["sheets"]["sheet"] == sheet],
                            tables["exact"]["sheets"][tables["exact"]["sheets"]["sheet"] == sheet]))
    for sheet in parameters["sheets"]:
        comparisons.append((f"probe classes seed 12 - seed 11, {sheet}",
                            prediction_table(probes / f"classes_seed_11_without_{sheet}.npz", frames),
                            prediction_table(probes / f"classes_seed_12_without_{sheet}.npz", frames)))
    end_to_end = {}
    for run in ("resnet18_finetune_classes", "resnet18_finetune_exact", "small_cnn_scratch_classes", "small_cnn_line_and_puck_classes"):
        end_to_end[run] = {sheet: prediction_table(parameters["runs_folder"] / f"2026-11-09_{run}" / f"predictions_without_{sheet}.npz", frames)
                           for sheet in ("path_03", "path_05", "path_06")}
    for sheet in ("path_03", "path_05", "path_06"):
        comparisons.append((f"end-to-end ResNet-18 exact - classes, {sheet}", end_to_end["resnet18_finetune_classes"][sheet], end_to_end["resnet18_finetune_exact"][sheet]))
    for sheet in ("path_03", "path_05", "path_06"):
        comparisons.append((f"end-to-end ResNet-18 classes - small CNN classes, {sheet}", end_to_end["small_cnn_scratch_classes"][sheet], end_to_end["resnet18_finetune_classes"][sheet]))
    for sheet in ("path_03", "path_05", "path_06"):
        comparisons.append((f"end-to-end ResNet-18 exact - probe exact, {sheet}",
                            tables["exact"]["sheets"][tables["exact"]["sheets"]["sheet"] == sheet], end_to_end["resnet18_finetune_exact"][sheet]))
    for label, first, second in comparisons:
        difference, draws, episode_count, row_count = paired_difference(first, second, parameters, generator)
        print(f"  {label}: {100 * difference:+.1f} points, interval {100 * numpy.percentile(draws, 2.5):+.1f} to "
              f"{100 * numpy.percentile(draws, 97.5):+.1f} ({episode_count} episodes, {row_count} moving rows)")
    for run, by_sheet in end_to_end.items():
        values = []
        for sheet, table in by_sheet.items():
            share, rows = direction_share(table, parameters)
            stops = stop_outcomes(table, parameters) if "stop_probability" in table.columns else None
            values.append(share)
            stop_text = f", stops right/early/never {stops['right']}/{stops['early']}/{stops['never']} of {stops['episodes']}" if stops else ""
            print(f"  end-to-end {run} {sheet}: within 45 {share:.3f} on {rows} rows{stop_text}")



def part_four_recovery(tables: dict, frames: pandas.DataFrame, parameters: dict) -> None:
    """Score the pictures after a hand correction apart from the others, on held-out sheets."""

    print("\n=== 4. Pictures after a hand correction (recovery pictures) against the others, held-out sheets, within 45 ===")
    kept = frames[frames["score"].isin([1, 2, 3]) & (frames["is_mistake_row"] == 0)]
    print(f"  all rows of frames.csv after a correction: {int((frames['corrections_before'] > 0).sum())} of {len(frames)}; "
          f"in the probes' rows: {int((kept['corrections_before'] > 0).sum())} of {len(kept)}; moving: "
          f"{int(((kept['corrections_before'] > 0) & kept['is_moving']).sum())} of {int(kept['is_moving'].sum())}")
    end_to_end = {run: pandas.concat([prediction_table(parameters["runs_folder"] / f"2026-11-09_{run}" / f"predictions_without_{sheet}.npz", frames)
                                      for sheet in ("path_03", "path_05", "path_06")], ignore_index = True)
                  for run in ("resnet18_finetune_classes", "resnet18_finetune_exact")}
    sources = [("probe classes", tables["classes"]["sheets"]), ("probe exact", tables["exact"]["sheets"]),
               ("end-to-end classes (03, 05, 06)", end_to_end["resnet18_finetune_classes"]),
               ("end-to-end exact (03, 05, 06)", end_to_end["resnet18_finetune_exact"])]
    for label, table in sources:
        before = table[table["corrections_before"] == 0]
        after = table[table["corrections_before"] > 0]
        before_share, before_rows = direction_share(before, parameters)
        after_share, after_rows = direction_share(after, parameters)
        forward_before, _ = direction_share(before.assign(forward_deg = 0.0), parameters, "forward_deg")
        forward_after, _ = direction_share(after.assign(forward_deg = 0.0), parameters, "forward_deg")
        print(f"  {label}: before any correction {before_share:.3f} ({before_rows} rows; always forward {forward_before:.3f}); "
              f"after a correction {after_share:.3f} ({after_rows} rows; always forward {forward_after:.3f})")
        per_sheet = []
        for sheet in parameters["sheets"]:
            sheet_after = after[after["sheet"] == sheet]
            sheet_before = before[before["sheet"] == sheet]
            if len(sheet_after) == 0:
                continue
            after_value, after_count = direction_share(sheet_after, parameters)
            before_value, before_count = direction_share(sheet_before, parameters)
            per_sheet.append(f"{sheet[-2:]} {before_value:.3f}/{after_value:.3f} ({before_count}/{after_count})")
        print(f"    per sheet, before/after (rows): {'; '.join(per_sheet)}")



def part_five_steps(frames: pandas.DataFrame, parameters: dict) -> None:
    """Steps per centimetre of line in the episodes, by start kind, for a step budget."""

    print("\n=== 5. Steps per cm of line (rows per episode / cap-to-cap line length), episodes with score 1 to 3 ===")
    kept = frames[frames["score"].isin([1, 2, 3])]
    episodes = kept.groupby("episode_id").agg(sheet = ("sheet", "first"), start_kind = ("start_kind", "first"),
                                              rows = ("row_in_episode", "size"))
    print("  start kinds (episodes): " + ", ".join(f"{kind} {count}" for kind, count in episodes["start_kind"].fillna("empty").value_counts().items()))
    episodes["per_cm"] = episodes["rows"] / episodes["sheet"].map(parameters["line_length_cm"])
    for label, chosen in (("STD starts", episodes[episodes["start_kind"] == "STD"]),
                          ("all episodes", episodes)):
        print(f"  {label}:")
        for sheet in parameters["sheets"]:
            part = chosen[chosen["sheet"] == sheet]["per_cm"]
            if len(part) == 0:
                continue
            print(f"    {sheet}: {len(part)} episodes, rows per cm median {part.median():.2f}, 90th percentile {part.quantile(0.9):.2f}, "
                  f"largest {part.max():.2f}; over 3 per cm {int((part > 3).sum())}, over 4 per cm {int((part > 4).sum())}")
        print(f"    all sheets: {len(chosen)} episodes, median {chosen['per_cm'].median():.2f}, 90th percentile {chosen['per_cm'].quantile(0.9):.2f}, "
              f"largest {chosen['per_cm'].max():.2f}; over 3 per cm {int((chosen['per_cm'] > 3).sum())}, over 4 per cm {int((chosen['per_cm'] > 4).sum())}")
    seconds = kept["seconds_to_next_row"].dropna()
    print(f"  seconds to the next row: median {seconds.median():.2f}, 90th percentile {seconds.quantile(0.9):.2f}")



def part_six_end_to_end(parameters: dict) -> None:
    """Print the end-to-end fold measures with their denominators, and when the validation loss was lowest."""

    print("\n=== 6. End-to-end folds (stored measures) and training histories ===")
    for run in ("resnet18_finetune_classes", "resnet18_finetune_exact", "small_cnn_scratch_classes", "small_cnn_line_and_puck_classes"):
        for sheet in ("path_03", "path_05", "path_06"):
            path = parameters["runs_folder"] / f"2026-11-09_{run}" / f"fold_without_{sheet}.json"
            content = json.loads(path.read_text(encoding = "utf-8"))
            measures = content["measures"]
            episodes = measures["test_episodes"]
            history = content.get("history", [])
            losses = [entry.get("validation_loss") for entry in history]
            lowest = int(numpy.argmin(losses)) + 1 if losses else None
            tuned_right = measures.get("tuned_two_episodes_stopped_right")
            tuned_text = (f"; tuned two in a row right/early/never {round(tuned_right * episodes)}/{round(measures['tuned_two_episodes_stopped_early'] * episodes)}/"
                          f"{round(measures['tuned_two_episodes_never_stopped'] * episodes)} (threshold {measures.get('tuned_two_stop_threshold')})") if tuned_right is not None else ""
            print(f"  {run} {sheet}: episodes {episodes}, within 45 {measures['within_45_deg']} of {measures['direction_rows']} rows, path within 45 "
                  f"{measures['path_within_45_deg']} of {measures['path_direction_rows']}; stop 0.5 right/early/never "
                  f"{round(measures['episodes_stopped_right'] * episodes)}/{round(measures['episodes_stopped_early'] * episodes)}/{round(measures['episodes_never_stopped'] * episodes)}"
                  f"{tuned_text}; epochs {len(history)}, lowest validation loss at epoch {lowest}")



def wilson_interval(successes: int, trials: int, z: float = 1.959963985) -> tuple:
    """Return the Wilson score interval of a binomial share."""

    share = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (share + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(share * (1.0 - share) / trials + z * z / (4.0 * trials * trials)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))



def upper_tail(successes: int, trials: int, probability: float) -> float:
    """Return P(X >= successes) for X binomial(trials, probability)."""

    return sum(math.comb(trials, count) * probability ** count * (1.0 - probability) ** (trials - count)
               for count in range(successes, trials + 1))



def clopper_pearson_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple:
    """Return the exact (Clopper-Pearson) interval by bisection on the binomial tails."""

    def solved(function, target: float) -> float:
        low, high = 0.0, 1.0
        for _ in range(80):
            middle = (low + high) / 2.0
            if function(middle) < target:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0

    lower = 0.0 if successes == 0 else solved(lambda probability: upper_tail(successes, trials, probability), alpha / 2.0)
    upper = 1.0 if successes == trials else solved(lambda probability: upper_tail(successes + 1, trials, probability), 1.0 - alpha / 2.0)
    return (lower, upper)



def probability_mass(trials: int, probability: float) -> list:
    """Return the binomial probabilities of 0 to ``trials`` successes."""

    return [math.comb(trials, count) * probability ** count * (1.0 - probability) ** (trials - count) for count in range(trials + 1)]



def better_shows_more(trials: int, better: float, worse: float) -> float:
    """Return the chance that the better model shows more successes, ties split evenly."""

    better_mass = probability_mass(trials, better)
    worse_mass = probability_mass(trials, worse)
    total = 0.0
    for better_count, better_probability in enumerate(better_mass):
        for worse_count, worse_probability in enumerate(worse_mass):
            if better_count > worse_count:
                total += better_probability * worse_probability
            elif better_count == worse_count:
                total += 0.5 * better_probability * worse_probability
    return total



def newcombe_interval(first_successes: int, first_trials: int, second_successes: int, second_trials: int) -> tuple:
    """Return Newcombe's hybrid score interval for the difference second minus first."""

    first_share = first_successes / first_trials
    second_share = second_successes / second_trials
    first_lower, first_upper = wilson_interval(first_successes, first_trials)
    second_lower, second_upper = wilson_interval(second_successes, second_trials)
    difference = second_share - first_share
    lower = difference - math.sqrt((second_share - second_lower) ** 2 + (first_upper - first_share) ** 2)
    upper = difference + math.sqrt((second_upper - second_share) ** 2 + (first_share - first_lower) ** 2)
    return (lower, upper)



def posterior_better(first_successes: int, first_trials: int, second_successes: int, second_trials: int, points: int = 20001) -> float:
    """Return P(second rate > first rate) with uniform priors, by numerical integration."""

    grid = numpy.linspace(0.0, 1.0, points)[1:-1]

    def density(successes: int, trials: int) -> numpy.ndarray:
        alpha, beta = successes + 1.0, trials - successes + 1.0
        logarithm = (alpha - 1.0) * numpy.log(grid) + (beta - 1.0) * numpy.log1p(-grid) - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
        return numpy.exp(logarithm)

    step = grid[1] - grid[0]
    first_density = density(first_successes, first_trials)
    first_cumulative = numpy.cumsum(first_density) * step
    return float(numpy.sum(density(second_successes, second_trials) * first_cumulative) * step)



def part_seven_robot_statistics(parameters: dict) -> None:
    """Intervals, tests, choosing between two models, clustered trials and robot time."""

    print("\n=== 7. Robot-trial statistics (calculated) ===")
    print("\n--- 95 % intervals: Wilson and Clopper-Pearson (exact)")
    for trials in (10, 20, 30):
        for share in (0.5, 0.7, 0.8, 0.9, 1.0):
            successes = int(round(share * trials))
            wilson_lower, wilson_upper = wilson_interval(successes, trials)
            exact_lower, exact_upper = clopper_pearson_interval(successes, trials)
            print(f"  {successes}/{trials}: Wilson {wilson_lower:.3f} to {wilson_upper:.3f} (half-width {100 * (wilson_upper - wilson_lower) / 2:.1f} points); "
                  f"Clopper-Pearson {exact_lower:.3f} to {exact_upper:.3f}")

    print("\n--- one model against a 70 % bar (exact binomial test, one-sided 5 %), true rate 90 %")
    first_power = None
    for trials in range(10, 41):
        critical = next(count for count in range(trials + 1) if upper_tail(count, trials, 0.7) <= 0.05)
        power = upper_tail(critical, trials, 0.9)
        if first_power is None and power >= 0.8:
            first_power = trials
        if trials in (10, 15, 20, 25, 28, 30, 33, 35, 40):
            print(f"  {trials} trials: pass with {critical}/{trials} or more; size {upper_tail(critical, trials, 0.7):.3f}; power {power:.3f}")
    print(f"  first trial count with power 0.8: {first_power}")

    print("\n--- choosing the better of two models: chance that the better shows more successes (ties split)")
    for better, worse in ((0.9, 0.7), (0.9, 0.8), (0.7, 0.5), (0.6, 0.4)):
        print(f"  {better:.0%} against {worse:.0%}: " + ", ".join(f"{trials} each {better_shows_more(trials, better, worse):.3f}" for trials in (10, 12, 15, 20, 24, 30, 50)))

    print("\n--- an observed 20-point gap: Newcombe 95 % interval of the difference, and P(second better) with uniform priors")
    for first, second, trials in ((7, 9, 10), (14, 18, 20), (21, 27, 30)):
        lower, upper = newcombe_interval(first, trials, second, trials)
        print(f"  {first}/{trials} against {second}/{trials}: difference {lower:+.3f} to {upper:+.3f}; P(second better) {posterior_better(first, trials, second, trials):.3f}")

    print("\n--- trials clustered on tracks: design effect 1 + (m - 1) x rho and effective number of trials out of 30 (rho guessed)")
    for tracks, per_track in ((3, 10), (6, 5), (10, 3), (15, 2), (30, 1)):
        effects = []
        for rho in (0.05, 0.15, 0.30):
            effect = 1.0 + (per_track - 1) * rho
            effects.append(f"rho {rho:.2f}: effect {effect:.2f}, effective {30 / effect:.1f}")
        print(f"  {tracks} tracks x {per_track}: " + "; ".join(effects))
    for tracks, per_track in ((3, 10), (10, 3)):
        effect = 1.0 + (per_track - 1) * 0.15
        effective = 30 / effect
        lower, upper = wilson_interval(round(0.9 * effective), round(effective))
        print(f"  27/30 seen as {round(0.9 * effective)}/{round(effective)} effective trials ({tracks} x {per_track}, rho 0.15): Wilson {lower:.3f} to {upper:.3f}")
    lower, upper = wilson_interval(27, 30)
    print(f"  27/30 as 30 independent trials: Wilson {lower:.3f} to {upper:.3f}")

    print("\n--- robot time (minutes): per-sheet minutes from main_session_facts.md (steps x 1.22 s + 1 min set-up, a guess),")
    print("    and a slower case: 1.5 x the demonstration steps and a 2.5-minute reset (both guesses)")
    slow = {sheet: (1.5 * parameters["median_steps"][sheet] * parameters["seconds_per_step"]) / 60.0 + 2.5 for sheet in parameters["sheets"]}
    for sheet in parameters["sheets"]:
        print(f"  {sheet}: {parameters['trial_minutes'][sheet]:.1f} min per trial; slow case {slow[sheet]:.1f}")
    harder = ("path_04", "path_05", "path_06")
    new_track_minutes = numpy.mean([parameters["trial_minutes"][sheet] for sheet in ("path_03", "path_04", "path_05", "path_06")])
    new_track_slow = numpy.mean([slow[sheet] for sheet in ("path_03", "path_04", "path_05", "path_06")])
    print(f"  a new track, assumed like sheets 03 to 06: {new_track_minutes:.2f} min; slow case {new_track_slow:.2f}")
    phases = {
        "phase 0, pipeline check: sheet 01, 6 runs": (6 * parameters["trial_minutes"]["path_01"], 6 * slow["path_01"]),
        "phase 1, choice: 24 paired blocks (48 trials): sheets 04, 05, 06 x 2 ends x 2, and 2 validation tracks x 2 ends x 3":
            (sum(8 * parameters["trial_minutes"][sheet] for sheet in harder) + 24 * new_track_minutes,
             sum(8 * slow[sheet] for sheet in harder) + 24 * new_track_slow),
        "phase 2, final: 6 sheets x 5 trials, 10 sealed tracks x 3 trials (60 trials)":
            (sum(5 * parameters["trial_minutes"][sheet] for sheet in parameters["sheets"]) + 30 * new_track_minutes,
             sum(5 * slow[sheet] for sheet in parameters["sheets"]) + 30 * new_track_slow),
        "light check: sheets 03, 05, 06 x 2 under a changed light (6 trials)":
            (2 * sum(parameters["trial_minutes"][sheet] for sheet in ("path_03", "path_05", "path_06")),
             2 * sum(slow[sheet] for sheet in ("path_03", "path_05", "path_06")))
    }
    total = [0.0, 0.0]
    for label, (minutes, slow_minutes) in phases.items():
        total[0] += minutes
        total[1] += slow_minutes
        print(f"  {label}: {minutes:.0f} min ({minutes / 60:.1f} h); slow case {slow_minutes:.0f} min ({slow_minutes / 60:.1f} h)")
    print(f"  all phases: {total[0] / 60:.1f} h; slow case {total[1] / 60:.1f} h (without breaks, warm-up runs and void re-runs)")



def main() -> None:

    parameters = dict(ANALYSIS_PARAMETERS)
    frames = load_frames(parameters)
    print(f"frames.csv: {len(frames)} rows, {frames['episode_id'].nunique()} episodes")
    tables = part_one_splits(frames, parameters)
    part_two_noise(tables, frames, parameters)
    part_three_paired(tables, frames, parameters)
    part_four_recovery(tables, frames, parameters)
    part_five_steps(frames, parameters)
    part_six_end_to_end(parameters)
    part_seven_robot_statistics(parameters)



if __name__ == "__main__":
    main()
