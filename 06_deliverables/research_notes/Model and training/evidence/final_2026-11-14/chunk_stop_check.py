"""
Does the model's own look-ahead say where to stop? (14.11.26, exploratory.)

The chunk model predicts where the rod tip will be after each of the next five steps; near the end of a line the
demonstrated tip stays where it stopped, so the predicted reach (how far the tip is predicted to be after five steps)
shrinks to nothing. That is a stop signal learned from the last five pictures of every episode instead of one. Here it
is scored on the saved predictions of the 16 held-out new tracks, without any new training, against the stop output:

    stop output     stop chance at least 0.5 on two pictures in a row (the fixed rule);
    reach           predicted reach under a limit on two pictures in a row;
    either / both   the stop output (with its own limit) or / and the reach.

Every limit is chosen without the track it is used on (for each held-out track, on the other fifteen), and the rule
chosen on the first seed is then applied unchanged to the second seed's models.

Run from 02_code:
    ..\\.venv\\Scripts\\python.exe "../06_deliverables/research_notes/Model and training/evidence/final_2026-11-14/chunk_stop_check.py"
"""

import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "02_code"))

from config import FULL_INDEX_FOLDER, TRAINING_METRIC_PARAMETERS, TRAINING_RUN_PARAMETERS


CHUNK_STOP_PARAMETERS = {
    "runs_folder": Path(__file__).resolve().parents[5] / "04_runs" / "gpu_ki014",
    "runs": ["2026-11-14_resnet18_full_new_shapes", "2026-11-14_resnet18_full_new_shapes_seed_2"],
    "reach_limits_cm": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0],
    "stop_limits": [0.1, 0.2, 0.3, 0.5, 0.7],
    "pictures_in_a_row": 2,
    "mid_paper_end_tracks": TRAINING_RUN_PARAMETERS["mid_paper_end_tracks"],
    "output_path": Path(__file__).resolve().parent / "chunk_stop_scores.json"
}



def load_run(
        run_folder: Path,
        frames: pandas.DataFrame
) -> pandas.DataFrame:
    """
    Return one row per test picture of every finished fold of a run.

    Parameters
    ----------
    run_folder : pathlib.Path
        The run's folder with its predictions_without_T*.npz files.
    frames : pandas.DataFrame
        The index's frames.csv, indexed by frame_index.

    Returns
    -------
    pandas.DataFrame
        Track, episode, step, distance to the end, stop chance and predicted reach, sorted by episode and step.
    """

    parts = []
    for prediction_path in sorted(run_folder.glob("predictions_without_T*.npz")):
        saved = numpy.load(prediction_path)
        part = frames.loc[saved["frame_index"], ["episode_id", "row_in_episode", "goal_distance_cm"]].copy()
        part["track"] = prediction_path.stem.split("_")[-1]
        part["stop_chance"] = saved["stop_probability"]
        part["reach_cm"] = numpy.linalg.norm(saved["chunk_cm"][:, -1, :], axis = 1)
        parts.append(part)
    return pandas.concat(parts).sort_values(["track", "episode_id", "row_in_episode"]) if parts else pandas.DataFrame()



def outcomes(
        table: pandas.DataFrame,
        is_stop_said: numpy.ndarray,
        parameters: Dict
) -> Dict[str, str]:
    """
    Return how every episode ends under one stop signal.

    Parameters
    ----------
    table : pandas.DataFrame
        From ``load_run``.
    is_stop_said : numpy.ndarray
        Per picture, whether the signal says stop.
    parameters : dict
        Supplies "pictures_in_a_row".

    Returns
    -------
    dict
        Per episode "right" (within the scorer's 2 cm of the end), "short" (2 to 3 cm before it), "early" or "never".
    """

    result = {}
    table = table.assign(is_stop_said = is_stop_said)
    for episode_id, episode in table.groupby("episode_id", sort = False):
        distances = episode["goal_distance_cm"].fillna(0.0).to_numpy()
        run_length, stop_position = 0, -1
        for position, value in enumerate(episode["is_stop_said"].to_numpy()):
            run_length = run_length + 1 if value else 0
            if run_length >= parameters["pictures_in_a_row"]:
                stop_position = position
                break
        if stop_position < 0:
            result[episode_id] = "never"
        elif distances[stop_position] <= TRAINING_METRIC_PARAMETERS["right_stop_cm"]:
            result[episode_id] = "right"
        elif distances[stop_position] > TRAINING_METRIC_PARAMETERS["early_stop_cm"]:
            result[episode_id] = "early"
        else:
            result[episode_id] = "short"
    return result



def said(
        table: pandas.DataFrame,
        rule: str,
        stop_limit: float,
        reach_limit: float
) -> numpy.ndarray:
    """Return per picture whether the rule ("stop output", "reach", "either" or "both") says stop."""

    by_stop = table["stop_chance"].to_numpy() >= stop_limit
    by_reach = table["reach_cm"].to_numpy() < reach_limit
    return {"stop output": by_stop, "reach": by_reach, "either": by_stop | by_reach, "both": by_stop & by_reach}[rule]



def counted(results: Dict[str, str]) -> Dict[str, int]:
    """Return how many episodes ended each way."""

    return {name: sum(1 for value in results.values() if value == name) for name in ("right", "short", "early", "never")}



def nested_outcomes(
        table: pandas.DataFrame,
        rule: str,
        parameters: Dict
) -> Tuple[Dict[str, str], List[Tuple[float, float]]]:
    """
    Return the outcomes of a rule whose limits are chosen, for every held-out track, on the other tracks only.

    The limits with the most right stops on the other tracks win; ties go to fewer early stops.

    Returns
    -------
    tuple of dict and list
        The outcome per episode, and the limits chosen for each track.
    """

    stop_limits = parameters["stop_limits"] if rule != "reach" else [0.5]
    reach_limits = parameters["reach_limits_cm"] if rule != "stop output" else [0.0]
    grid = list(itertools.product(stop_limits, reach_limits))
    per_limit = {limits: outcomes(table, said(table, rule, *limits), parameters) for limits in grid}
    track_of_episode = table.groupby("episode_id")["track"].first().to_dict()
    results, chosen = {}, []
    for track in sorted(table["track"].unique()):

        def quality(limits: Tuple[float, float]) -> Tuple[int, int]:
            others = [value for episode_id, value in per_limit[limits].items() if track_of_episode[episode_id] != track]
            return (sum(value == "right" for value in others), -sum(value == "early" for value in others))

        best = max(grid, key = quality)
        chosen.append(best)
        results.update({episode_id: value for episode_id, value in per_limit[best].items() if track_of_episode[episode_id] == track})
    return results, chosen



def main(parameters: Dict = CHUNK_STOP_PARAMETERS) -> None:

    columns = ["frame_index", "episode_id", "row_in_episode", "goal_distance_cm"]
    frames = pandas.read_csv(Path(FULL_INDEX_FOLDER) / "frames.csv", usecols = columns).set_index("frame_index")
    scores = {}
    first_choice = {}
    for run_number, run_name in enumerate(parameters["runs"]):
        table = load_run(parameters["runs_folder"] / run_name, frames)
        if table.empty:
            continue
        track_of_episode = table.groupby("episode_id")["track"].first()
        is_mid_paper = track_of_episode.isin(parameters["mid_paper_end_tracks"])
        print(f"\n{run_name}: {table['track'].nunique()} tracks, {table['episode_id'].nunique()} episodes "
              f"({int(is_mid_paper.sum())} end in the middle of the paper)")
        scores[run_name] = {}
        fixed = outcomes(table, said(table, "stop output", TRAINING_METRIC_PARAMETERS["stop_threshold"], 0.0), parameters)
        rows = [("stop output, fixed 0.5", fixed, "")]
        for rule in ("stop output", "reach", "either", "both"):
            results, chosen = nested_outcomes(table, rule, parameters)
            most_common = max(set(chosen), key = chosen.count)
            rows.append((f"{rule}, limits chosen without the track", results, f"mostly stop {most_common[0]}, reach {most_common[1]} cm"))
            if run_number == 0:
                first_choice[rule] = most_common
            else:
                carried = outcomes(table, said(table, rule, *first_choice[rule]), parameters)
                rows.append((f"{rule}, the first seed's limits unchanged", carried, f"stop {first_choice[rule][0]}, reach {first_choice[rule][1]} cm"))
        for label, results, remark in rows:
            everything = counted(results)
            mid_paper = counted({episode_id: value for episode_id, value in results.items() if is_mid_paper[episode_id]})
            scores[run_name][label] = {"all": everything, "mid_paper": mid_paper, "remark": remark}
            print(f"  {label:48s} right {everything['right']:2d}  short {everything['short']:2d}  early {everything['early']:2d}  never {everything['never']:2d}"
                  f"  | mid-paper right {mid_paper['right']:2d} of {sum(mid_paper.values()):2d}  {remark}")
    parameters["output_path"].write_text(json.dumps(scores, indent = 1), encoding = "utf-8")



if __name__ == "__main__":
    main()
