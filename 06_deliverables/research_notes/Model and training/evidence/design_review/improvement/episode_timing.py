"""
Measured robot time per demonstration episode and per hand correction (read only).

Reads every episode folder under D:/AI_Project/03_data/raw/main_data (episode.json, review.json, guided_path.csv,
correction_NN_guided_path.csv, steps.csv) and writes nothing outside this script's own folder. It measures:

- the initial hand guiding (first to last point of guided_path.csv);
- the replay (first to last logged row), split into plain stepping (row-to-row gaps under 10 s) and pauses;
- every hand correction: the pause from the row before the correction to the first replayed row of the correction
  (rows with waypoint 0 after row 0, the audit's marker), the guiding inside it (correction_NN_guided_path.csv), and
  the number of replayed rows of the correction;
- the overheads: start to the first row minus the guiding, last row to finished, finished to the next start;
- the sittings (start-time gaps over 90 minutes, as in the audit's table 5.4) and the time they cost per kept episode.

Usage: python episode_timing.py
"""

import datetime
import json
import math
import statistics
from pathlib import Path

import numpy
import pandas

PARAMETERS = {
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "output_folder": Path(__file__).resolve().parent,
    "pause_threshold_s": 10.0,
    "sitting_gap_minutes": 90.0,
    "bootstrap_draws": 2000,
    "seed": 91126,
}


def local_timestamp(text):
    """
    Turn an ISO time written by the tool (local time, no zone) into seconds since the epoch.
    """
    return datetime.datetime.fromisoformat(text).timestamp()


def guided_seconds(path):
    """
    Seconds from the first to the last point of a guided path file, or None.
    """
    if not path.exists():
        return None
    table = pandas.read_csv(path, usecols = ["unix_time"])
    if len(table) < 2:
        return 0.0
    return float(table["unix_time"].iloc[-1] - table["unix_time"].iloc[0])


def score_of(folder):
    """
    The score from review.json, else from the folder name, else None.
    """
    review_path = folder / "review.json"
    if review_path.exists():
        with open(review_path, "r", encoding = "utf-8") as handle:
            review = json.load(handle)
        if review.get("score") is not None:
            return int(review["score"]), review.get("reviewed_at")
    if "_score_" in folder.name:
        return int(folder.name.rsplit("_score_", 1)[1]), None
    return None, None


def episode_record(folder, parameters):
    """
    Collect the timing of one episode folder.

    Returns
    -------
    dict, list of dict
        The episode's row and one row per hand correction.
    """
    with open(folder / "episode.json", "r", encoding = "utf-8") as handle:
        episode = json.load(handle)
    score, reviewed_at = score_of(folder)
    record = {
        "folder": f"{folder.parent.name}/{folder.name}",
        "sheet": episode.get("path_id", "")[:7],
        "mode": episode.get("mode"),
        "status": episode.get("status"),
        "outcome": episode.get("outcome"),
        "operator": episode.get("operator"),
        "score": score,
        "started_s": local_timestamp(episode["started"]) if episode.get("started") else math.nan,
        "finished_s": local_timestamp(episode["finished"]) if episode.get("finished") else math.nan,
        "reviewed_s": local_timestamp(reviewed_at) if reviewed_at else math.nan,
        "hand_corrections_logged": episode.get("hand_corrections"),
        "steps_logged": episode.get("steps"),
        "guiding_s": guided_seconds(folder / "guided_path.csv"),
    }
    corrections = []
    steps_path = folder / "steps.csv"
    if steps_path.exists():
        steps = pandas.read_csv(steps_path, usecols = ["unix_time", "waypoint", "action"])
    else:
        steps = pandas.DataFrame(columns = ["unix_time", "waypoint", "action"])
    record["rows"] = len(steps)
    if len(steps) >= 2:
        times = steps["unix_time"].to_numpy(dtype = float)
        gaps = numpy.diff(times)
        is_correction_start = (steps["waypoint"].to_numpy() == 0) & (numpy.arange(len(steps)) > 0)
        correction_rows = numpy.flatnonzero(is_correction_start)
        is_pause = gaps >= parameters["pause_threshold_s"]
        record["first_row_s"] = times[0]
        record["last_row_s"] = times[-1]
        record["replay_span_s"] = times[-1] - times[0]
        record["stepping_s"] = float(gaps[~is_pause].sum())
        record["plain_steps"] = int((~is_pause).sum())
        record["pause_s"] = float(gaps[is_pause].sum())
        record["stop_pause_s"] = float(gaps[-1]) if steps["action"].iloc[-1] == "stop" else math.nan
        boundaries = list(correction_rows) + [len(steps) - 1]
        for number, row in enumerate(correction_rows, start = 1):
            next_boundary = boundaries[number]
            correction_guiding = guided_seconds(folder / f"correction_{number:02d}_guided_path.csv")
            corrections.append({
                "folder": record["folder"],
                "sheet": record["sheet"],
                "score": score,
                "number": number,
                "pause_s": float(times[row] - times[row - 1]),
                "guiding_s": correction_guiding,
                "replayed_rows": int(next_boundary - row),
                "row_before": int(row),
            })
        record["corrections_found"] = len(correction_rows)
    else:
        record["first_row_s"] = math.nan
        record["last_row_s"] = math.nan
        record["replay_span_s"] = math.nan
        record["stepping_s"] = math.nan
        record["plain_steps"] = 0
        record["pause_s"] = math.nan
        record["stop_pause_s"] = math.nan
        record["corrections_found"] = 0
    return record, corrections


def quantiles(values):
    """
    Median, quartiles and count of a list, ignoring missing values.
    """
    clean = [value for value in values if value is not None and not (isinstance(value, float) and math.isnan(value))]
    if not clean:
        return "n/a"
    array = numpy.array(clean, dtype = float)
    return (f"median {numpy.median(array):.2f}, quartiles {numpy.percentile(array, 25):.2f} to "
            f"{numpy.percentile(array, 75):.2f}, 10th to 90th {numpy.percentile(array, 10):.2f} to "
            f"{numpy.percentile(array, 90):.2f}, n {len(array)}")


def main(parameters = None):
    """
    Build the tables and print them.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    records = []
    corrections = []
    for session_folder in sorted(settings["raw_folder"].iterdir()):
        if not session_folder.is_dir():
            continue
        for folder in sorted(session_folder.iterdir()):
            if folder.is_dir() and (folder / "episode.json").exists():
                record, episode_corrections = episode_record(folder, settings)
                records.append(record)
                corrections.extend(episode_corrections)
    episodes = pandas.DataFrame(records).sort_values("started_s").reset_index(drop = True)
    corrections = pandas.DataFrame(corrections)
    episodes["is_kept"] = episodes["score"].isin([1, 2, 3])
    episodes["duration_min"] = (episodes["finished_s"] - episodes["started_s"]) / 60.0
    episodes["before_first_row_min"] = (episodes["first_row_s"] - episodes["started_s"]) / 60.0
    episodes["setup_without_guiding_min"] = episodes["before_first_row_min"] - episodes["guiding_s"].fillna(0) / 60.0
    episodes["after_last_row_min"] = (episodes["finished_s"] - episodes["last_row_s"]) / 60.0
    start_gaps = episodes["started_s"].diff() / 60.0
    episodes["sitting"] = (start_gaps.isna() | (start_gaps > settings["sitting_gap_minutes"])).cumsum()
    episodes["to_next_start_min"] = (episodes["started_s"].shift(-1) - episodes["finished_s"]) / 60.0
    episodes["cycle_min"] = (episodes["started_s"].shift(-1) - episodes["started_s"]) / 60.0
    is_last_of_sitting = episodes["sitting"] != episodes["sitting"].shift(-1)
    episodes.loc[is_last_of_sitting, ["to_next_start_min", "cycle_min"]] = math.nan

    print("episode folders with episode.json:", len(episodes), "; kept (score 1 to 3):", int(episodes["is_kept"].sum()),
          "; score 0:", int((episodes["score"] == 0).sum()), "; no score:", int(episodes["score"].isna().sum()))
    print("modes:", episodes["mode"].value_counts().to_dict())
    print("statuses:", episodes["status"].value_counts().to_dict())
    print()
    print("=== sittings (start gaps over 90 min): span from the first start to the last finish ===")
    total_minutes = 0.0
    total_kept = 0
    for sitting, group in episodes.groupby("sitting"):
        span = (group["finished_s"].max() - group["started_s"].min()) / 60.0
        total_minutes += span
        kept = int(group["is_kept"].sum())
        total_kept += kept
        first = datetime.datetime.fromtimestamp(group["started_s"].min()).strftime("%d.%m %H:%M")
        last = datetime.datetime.fromtimestamp(group["finished_s"].max()).strftime("%d.%m %H:%M")
        sheets = group.loc[group["is_kept"], "sheet"].value_counts().sort_index().to_dict()
        print(f"sitting {sitting}: {first} to {last}, {span:.0f} min, folders {len(group)}, kept {kept}, "
              f"score 0 {int((group['score'] == 0).sum())}, min per kept episode {span / max(kept, 1):.2f}, kept by sheet {sheets}")
    print(f"all sittings: {total_minutes:.0f} min = {total_minutes / 60:.1f} h for {total_kept} kept episodes, "
          f"{total_minutes / total_kept:.2f} min per kept episode")
    print()
    kept = episodes[episodes["is_kept"]]
    print("=== kept episodes, per sheet (minutes unless marked) ===")
    for sheet, group in kept.groupby("sheet"):
        print(f"--- {sheet}: {len(group)} kept episodes")
        print("  rows per episode:", quantiles(group["rows"].tolist()))
        print("  start to finish (min):", quantiles(group["duration_min"].tolist()))
        print("  initial guiding (s):", quantiles(group["guiding_s"].tolist()))
        print("  start to first row, without guiding (min):", quantiles(group["setup_without_guiding_min"].tolist()))
        print("  replay span (min):", quantiles((group["replay_span_s"] / 60.0).tolist()))
        print("  plain stepping (min):", quantiles((group["stepping_s"] / 60.0).tolist()))
        print("  pauses inside the replay (min):", quantiles((group["pause_s"] / 60.0).tolist()))
        print("  last row to finish (min):", quantiles(group["after_last_row_min"].tolist()))
        print("  finish to next start (min):", quantiles(group["to_next_start_min"].tolist()))
        print("  start to next start (min):", quantiles(group["cycle_min"].tolist()))
        print("  hand corrections:", quantiles(group["corrections_found"].tolist()),
              "sum", int(group["corrections_found"].sum()))
    print()
    print("=== all kept episodes ===")
    for column, scale, label in (("duration_min", 1.0, "start to finish (min)"), ("guiding_s", 1.0, "initial guiding (s)"),
                                 ("setup_without_guiding_min", 1.0, "start to first row without guiding (min)"),
                                 ("stepping_s", 1.0 / 60.0, "plain stepping (min)"),
                                 ("after_last_row_min", 1.0, "last row to finish (min)"),
                                 ("to_next_start_min", 1.0, "finish to next start (min)"),
                                 ("cycle_min", 1.0, "start to next start (min)"),
                                 ("stop_pause_s", 1.0, "pause before the stop row (s)")):
        values = (kept[column] * scale).tolist()
        print(f"  {label}: {quantiles(values)}")
    seconds_per_plain_step = kept["stepping_s"].sum() / kept["plain_steps"].sum()
    print(f"  plain stepping per step: {seconds_per_plain_step:.3f} s (sum over {int(kept['plain_steps'].sum())} steps)")
    print()
    print("=== hand corrections in kept episodes ===")
    kept_corrections = corrections[corrections["score"].isin([1, 2, 3])]
    print("count:", len(kept_corrections), "in", kept_corrections["folder"].nunique(), "episodes")
    print("pause from the row before to the first replayed row (s):", quantiles(kept_corrections["pause_s"].tolist()))
    print("guiding inside the correction (s):", quantiles(kept_corrections["guiding_s"].tolist()))
    overhead = kept_corrections["pause_s"] - kept_corrections["guiding_s"]
    print("pause minus guiding (s):", quantiles(overhead.tolist()))
    print("replayed rows per correction:", quantiles(kept_corrections["replayed_rows"].tolist()))
    for sheet, group in kept_corrections.groupby("sheet"):
        print(f"  {sheet}: n {len(group)}, pause (s) {quantiles(group['pause_s'].tolist())}")
    print()
    print("=== start to finish against rows and hand corrections (least squares, kept episodes) ===")
    model_rows = kept.dropna(subset = ["duration_min", "rows"])
    design = numpy.column_stack([numpy.ones(len(model_rows)), model_rows["rows"].to_numpy(dtype = float),
                                 model_rows["corrections_found"].to_numpy(dtype = float)])
    target = model_rows["duration_min"].to_numpy(dtype = float)
    coefficients, *_ = numpy.linalg.lstsq(design, target, rcond = None)
    print(f"minutes = {coefficients[0]:.2f} + {coefficients[1] * 60:.2f} s per row + {coefficients[2]:.2f} min per correction")
    randomness_generator = numpy.random.default_rng(settings["seed"])
    draws = []
    for _ in range(settings["bootstrap_draws"]):
        picked = randomness_generator.integers(0, len(model_rows), len(model_rows))
        draw, *_ = numpy.linalg.lstsq(design[picked], target[picked], rcond = None)
        draws.append(draw)
    draws = numpy.array(draws)
    for index, name in enumerate(["intercept (min)", "per row (min)", "per correction (min)"]):
        low, high = numpy.percentile(draws[:, index], [2.5, 97.5])
        print(f"  {name}: {coefficients[index]:.3f}, 95 % bootstrap {low:.3f} to {high:.3f}")
    episodes.to_csv(settings["output_folder"] / "episode_timing_episodes.csv", index = False)
    corrections.to_csv(settings["output_folder"] / "episode_timing_corrections.csv", index = False)


if __name__ == "__main__":
    main()
