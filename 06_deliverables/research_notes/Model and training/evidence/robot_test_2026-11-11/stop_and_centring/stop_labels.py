"""
How the demonstrations define the stop, and where the fold models would stop measured on the table.

Part 1 reads the labels only: the goal of record_episodes.goal_labels is the magnet tip's place at the episode's own
stop row, so goal_distance_cm is 0 and is_at_goal is 1 at every stop row by construction; the labels say where the
demonstrator stopped, not whether the puck covers the line's end.

Part 2 joins the saved answers of the fold models (each scored on the sheet it never saw) with the puck's measured place
against the line's end in the last eight pictures of every demonstration (demonstration_ends.py): how far short of the
end's centre the puck stands in the first picture whose stop chance reaches the driver's threshold, and where the
driver's rule (two such pictures in a row, with a step between) would leave it.

Run from this folder, after demonstration_ends.py:    python stop_labels.py
Writes stop_probability_by_shortfall.png; the printed tables are kept in stop_labels.out.txt.
"""

import glob
from pathlib import Path
from typing import Dict

import numpy
import pandas
from matplotlib import pyplot

import picture_measures
import run_measures

from config import MODEL_MODE_PARAMETERS, RECORDING_PARAMETERS, TRAINING_DATA_PARAMETERS
from training_targets import load_training_frames


STOP_LABEL_PARAMETERS = {
    # the fold runs whose saved answers are read: the recipes of the models used on the robot (A and its second seed,
    # E and its second seed)
    "fold_runs": {
        "A recipe (final candidate)": "2026-11-09_resnet18_final_candidate",
        "A recipe, seed 2": "2026-11-09_resnet18_final_candidate_seed_2",
        "E recipe (chunk, stop weight 25)": "2026-11-10_resnet18_chunk_stop_weight_25",
        "E recipe, seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2"
    },
    "runs_folder": picture_measures.ROOT_FOLDER / "04_runs" / "gpu_ki014",
    # an end counts as measured reliably when bare paper was seen beyond it, the line's width in the map is plausible
    # and the puck's outline fitted well
    "line_width_range_mm": [8.0, 12.5],
    "fit_quality_minimum": 0.8,
    # the puck hides the round end completely while the two centres are at most this far apart (12 - 5 mm)
    "cover_limit_mm": 7.0,
    "short_bins_mm": [-30.0, -5.0, 0.0, 5.0, 10.0, 15.0, 25.0, 45.0, 90.0],
    "final_pushes_mm": [0.0, 5.0, 8.0, 10.0],
    "other_thresholds": [0.3, 0.7, 0.9],
    "figure_dpi": 110
}



def reliable_ends() -> pandas.DataFrame:
    """The episodes of demonstration_ends_*.csv whose end was measured reliably."""

    ends = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(run_measures.EVIDENCE_FOLDER / "demonstration_ends_0*.csv")))],
                         ignore_index = True)
    width_least, width_most = STOP_LABEL_PARAMETERS["line_width_range_mm"]
    is_reliable = (ends["is_end_found"] & (ends["is_end_seen"] == True) & ends["line_width_mm"].between(width_least, width_most)
                   & (ends["fit_quality"] >= STOP_LABEL_PARAMETERS["fit_quality_minimum"]))
    ends["is_reliable"] = is_reliable
    return ends



def stop_chances(run_folder: Path) -> Dict[int, float]:
    """The stop chance of every picture a fold run was scored on, by frame_index."""

    chances: Dict[int, float] = {}
    for path in sorted(run_folder.glob("predictions_without_path_*.npz")):
        saved = numpy.load(path)
        chances.update(zip(saved["frame_index"].tolist(), saved["stop_probability"].tolist()))
    return chances



def main() -> None:

    frames = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    stops = frames[frames["action"] == "stop"]
    print("PART 1: the labels")
    print(f"episodes {frames['episode_id'].nunique()}, stop rows {len(stops)} (one per episode: {bool((stops.groupby('episode_id').size() == 1).all())})")
    print(f"goal_distance_cm at the stop rows: smallest {stops['goal_distance_cm'].min():.2f}, largest {stops['goal_distance_cm'].max():.2f}; "
          f"is_at_goal at the stop rows: {int(stops['is_at_goal'].sum())} of {len(stops)}")
    print(f"  (record_episodes.goal_labels: the goal is the magnet tip's place at the episode's last stop row; is_at_goal is "
          f"'within {RECORDING_PARAMETERS['at_goal_within_cm']} cm of it'. Both say where the demonstrator stopped, nothing about the line's end.)")
    grouped = frames.groupby("episode_id")
    at_goal_rows = grouped["is_at_goal"].sum()
    print(f"rows with is_at_goal 1 per episode: median {at_goal_rows.median():.0f}, range {at_goal_rows.min():.0f} to {at_goal_rows.max():.0f}; "
          f"all rows with is_at_goal 1: {int(frames['is_at_goal'].sum())} of {len(frames)}")
    for back in (1, 2, 3, 4):
        distance = grouped["goal_distance_cm"].apply(lambda column: column.iloc[-1 - back] if len(column) > back else numpy.nan)
        print(f"  tip's distance to its stop place {back} picture(s) before the stop: median {distance.median():.2f} cm, "
              f"10th to 90th percentile {distance.quantile(0.1):.2f} to {distance.quantile(0.9):.2f} cm")
    last_move = grouped["move_length_m"].apply(lambda column: column.iloc[-2] if len(column) > 1 else numpy.nan) * 100.0
    print(f"the demonstrated last move before the stop row: median {last_move.median():.2f} cm, under 0.5 cm in {int((last_move < 0.5).sum())} of {int(last_move.notna().sum())} episodes, "
          f"under 0.2 cm in {int((last_move < 0.2).sum())}")
    print("the operators' own verdict (episode.json 'outcome'):", frames.groupby("episode_id")["outcome"].first().value_counts(dropna = False).to_dict())

    cleaned = load_training_frames({"is_audit_cleaning": True})
    is_stop = cleaned["action"] == "stop"
    print(f"\nthe training rows of models A and E (audit cleaning on): {len(cleaned)} rows of {cleaned['episode_id'].nunique()} episodes; "
          f"stop rows {int(is_stop.sum())} ({int(cleaned['is_relabelled_stop'].sum())} of them relabelled: the row before the stop row shows the same picture); "
          f"that is {100.0 * is_stop.mean():.2f} % of the rows")
    print(f"the driver's stop rule: stop chance at least {MODEL_MODE_PARAMETERS['stop_threshold']} on {MODEL_MODE_PARAMETERS['stop_pictures_in_a_row']} pictures in a row, "
          f"with a fixed step of {100.0 * MODEL_MODE_PARAMETERS['step_m']:.1f} cm between them")

    print("\nPART 2: where the fold models would stop, on the table")
    ends = reliable_ends()
    reliable_ids = set(ends.loc[ends["is_reliable"], "episode_id"])
    print(f"demonstrations with a reliably measured end: {len(reliable_ids)} of {len(ends)}")
    last_steps = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(run_measures.EVIDENCE_FOLDER / "demonstration_last_steps_0*.csv")))],
                               ignore_index = True)
    last_steps = last_steps[last_steps["episode_id"].isin(reliable_ids) & (last_steps["fit_quality"] >= STOP_LABEL_PARAMETERS["fit_quality_minimum"])]
    arrival = last_steps.pivot_table(index = "episode_id", columns = "steps_before_stop", values = "short_mm")
    print("the demonstrations' own arrival: the puck's centre short of the end's centre, by pictures before the stop (mm)")
    for back in sorted(arrival.columns):
        column = arrival[back].dropna()
        print(f"  {back} before: median {column.median():+6.1f}, 10th to 90th percentile {column.quantile(0.1):+6.1f} to {column.quantile(0.9):+6.1f}  ({len(column)} episodes)")

    figure, axes = pyplot.subplots(1, 2, figsize = (13, 4.6))
    bins = STOP_LABEL_PARAMETERS["short_bins_mm"]
    limit = STOP_LABEL_PARAMETERS["cover_limit_mm"]
    threshold = MODEL_MODE_PARAMETERS["stop_threshold"]
    for label, folder in STOP_LABEL_PARAMETERS["fold_runs"].items():
        run_folder = STOP_LABEL_PARAMETERS["runs_folder"] / folder
        if not run_folder.exists():
            print(f"\n{label}: {run_folder} is missing")
            continue
        chances = stop_chances(run_folder)
        scored = last_steps.copy()
        scored["stop_chance"] = scored["frame_index"].map(chances)
        scored = scored.dropna(subset = ["stop_chance"])
        print(f"\n{label} ({folder}): {scored['episode_id'].nunique()} episodes, {len(scored)} of their last pictures scored")
        scored["short_bin"] = pandas.cut(scored["short_mm"], bins)
        by_bin = scored.groupby("short_bin", observed = True).agg(pictures = ("stop_chance", "size"), share_at_threshold = ("stop_chance", lambda column: float((column >= threshold).mean())),
                                                                  median_chance = ("stop_chance", "median"))
        print("  stop chance by how far short of the end's centre the puck stands (mm):")
        print("  " + by_bin.round(3).to_string().replace("\n", "\n  "))
        axes[0].plot([interval.mid for interval in by_bin.index], by_bin["share_at_threshold"], marker = "o", label = label)

        first_short, first_aside, final_short, final_gap, never = [], [], [], [], 0
        for episode_id, episode in scored.groupby("episode_id"):
            episode = episode.sort_values("steps_before_stop", ascending = False)
            is_stop_like = (episode["stop_chance"] >= threshold).to_numpy()
            if not is_stop_like.any():
                never += 1
                continue
            first = int(numpy.argmax(is_stop_like))
            first_short.append(float(episode["short_mm"].iloc[first]))
            first_aside.append(float(episode["aside_mm"].iloc[first]))
            # the driver stops at the second stop-like picture in a row; in the recording the next picture is the
            # demonstrator's next step (about 1 cm further), which stands in for the driver's fixed step
            if first + 1 < len(episode) and is_stop_like[first + 1]:
                final_short.append(float(episode["short_mm"].iloc[first + 1]))
                final_gap.append(float(numpy.hypot(episode["short_mm"].iloc[first + 1], episode["aside_mm"].iloc[first + 1])))
        first_short = numpy.array(first_short)
        print(f"  first picture of the last eight with a stop chance of {threshold} or more: found in {len(first_short)} episodes, never in {never}")
        print(f"    the puck's centre is then short of the end's centre by: median {numpy.median(first_short):+.1f} mm, mean {first_short.mean():+.1f} mm, "
              f"10th to 90th percentile {numpy.percentile(first_short, 10):+.1f} to {numpy.percentile(first_short, 90):+.1f} mm; "
              f"more than {limit:.0f} mm short in {int((first_short > limit).sum())}, beyond the centre in {int((first_short < 0).sum())}")
        if final_short:
            final_short = numpy.array(final_short)
            print(f"    with the next picture also stop-like (the driver's rule), {len(final_short)} episodes: the puck then stands "
                  f"median {numpy.median(final_short):+.1f} mm short (negative: beyond), 10th to 90th percentile {numpy.percentile(final_short, 10):+.1f} to {numpy.percentile(final_short, 90):+.1f} mm; "
                  f"centres at most {limit:.0f} mm apart in {int((numpy.array(final_gap) <= limit).sum())} of {len(final_gap)}")
        # a last straight push of a fixed length after the first stop-like picture: how often the two centres then
        # lie within the cover limit, along the line alone and with the sideways offset left as it was
        first_aside = numpy.array(first_aside)
        for push in STOP_LABEL_PARAMETERS["final_pushes_mm"]:
            along_share = float((numpy.abs(first_short - push) <= limit).mean())
            both_share = float((numpy.hypot(first_short - push, first_aside) <= limit).mean())
            print(f"    a last straight push of {push:4.1f} mm after that picture: within {limit:.0f} mm along the line in {100.0 * along_share:.0f} %, "
                  f"with the sideways offset as it was in {100.0 * both_share:.0f} %")
        print(f"    sideways at that picture: median size {numpy.median(numpy.abs(first_aside)):.1f} mm, over {limit:.0f} mm in {int((numpy.abs(first_aside) > limit).sum())} of {len(first_aside)}")
        axes[1].hist(first_short, bins = numpy.arange(-20.0, 62.5, 2.5), histtype = "step", linewidth = 1.4, label = label)
        # other thresholds for the first stop-like picture: a higher one waits longer but misses more ends
        for other in STOP_LABEL_PARAMETERS["other_thresholds"]:
            shorts, missed = [], 0
            for _, episode in scored.groupby("episode_id"):
                episode = episode.sort_values("steps_before_stop", ascending = False)
                is_over = (episode["stop_chance"] >= other).to_numpy()
                if is_over.any():
                    shorts.append(float(episode["short_mm"].iloc[int(numpy.argmax(is_over))]))
                else:
                    missed += 1
            shorts = numpy.array(shorts)
            print(f"    threshold {other:.1f}: first such picture in {len(shorts)} episodes (never in {missed}); puck then short by median {numpy.median(shorts):+.1f} mm, "
                  f"10th to 90th percentile {numpy.percentile(shorts, 10):+.1f} to {numpy.percentile(shorts, 90):+.1f} mm; over {limit:.0f} mm short in {int((shorts > limit).sum())}")

    axes[0].axvline(0.0, color = "black", linewidth = 0.6)
    axes[0].set_xlabel("puck's centre short of the end's centre, mm (bin middles; negative: beyond the end)")
    axes[0].set_ylabel(f"share of pictures with a stop chance of {threshold} or more")
    axes[0].set_title("How sharply the stop output follows the puck's real place", fontsize = 10)
    axes[0].legend(fontsize = 8)
    axes[1].axvline(0.0, color = "black", linewidth = 0.6)
    axes[1].set_xlabel("puck's centre short of the end's centre at the first stop-like picture, mm")
    axes[1].set_ylabel("demonstrations (held-out sheets)")
    axes[1].set_title("Where the first stop-like picture finds the puck", fontsize = 10)
    axes[1].legend(fontsize = 8)
    figure.tight_layout()
    figure.savefig(run_measures.EVIDENCE_FOLDER / "stop_probability_by_shortfall.png", dpi = STOP_LABEL_PARAMETERS["figure_dpi"])
    pyplot.close(figure)
    print("\nwritten: stop_probability_by_shortfall.png")



if __name__ == "__main__":
    main()
