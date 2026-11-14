"""
Find where the mid-way model (14.11.26) stopped on each held-out new-track episode, by the fixed rule of the scorer
(stop chance at least 0.5 on two pictures in a row), and lay the pictures side by side for the episodes it got wrong.

Read-only on the data and the runs. Run with the project's Python from any folder:
    D:/AI_Project/.venv/Scripts/python.exe find_wrong_stops.py
"""

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy
import pandas


WRONG_STOP_PARAMETERS = {
    "project_folder": Path("D:/AI_Project"),
    "run_folder": Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014/2026-11-14_resnet18_round_2_part_new_shapes"),
    "frames_path": Path("D:/AI_Project/03_data/processed/v02_training_index/frames.csv"),
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "output_folder": Path(__file__).resolve().parent / "output",
    # the scorer's fixed rule (TRAINING_METRIC_PARAMETERS in config.py)
    "stop_threshold": 0.5,
    "pictures_in_a_row": 2,
    "right_stop_cm": 2.0,
    "early_stop_cm": 3.0,
    "tile_width_px": 320,
    "label_height_px": 44
}



def stop_row_of_episode(stop_probability: numpy.ndarray, parameters: Dict) -> int:
    """
    Return the position of the first picture where the rule stops the episode, or -1 when it never stops.

    Parameters
    ----------
    stop_probability : numpy.ndarray
        The model's stop chance per picture, in step order.
    parameters : dict
        Supplies "stop_threshold" and "pictures_in_a_row".

    Returns
    -------
    int
        The position within the episode, or -1.
    """

    run_length = 0
    for position, chance in enumerate(stop_probability):
        run_length = run_length + 1 if chance >= parameters["stop_threshold"] else 0
        if run_length >= parameters["pictures_in_a_row"]:
            return position
    return -1



def episode_outcomes(parameters: Dict = WRONG_STOP_PARAMETERS) -> List[Dict]:
    """
    Return one record per held-out episode: where the rule stopped, how far from the end, and the stop chances.

    Parameters
    ----------
    parameters : dict
        See ``WRONG_STOP_PARAMETERS``.

    Returns
    -------
    list of dict
        The episodes of all nine folds, in fold order.
    """

    columns = ["episode_id", "plan_number", "path_id", "row_in_episode", "steps_to_end", "goal_distance_cm", "frame_path",
               "frame_index", "action", "score"]
    frames = pandas.read_csv(parameters["frames_path"], usecols = columns).set_index("frame_index", drop = False)
    records = []
    for prediction_path in sorted(parameters["run_folder"].glob("predictions_without_T*.npz")):
        saved = numpy.load(prediction_path)
        fold_rows = frames.loc[saved["frame_index"]].assign(stop_probability = saved["stop_probability"])
        for episode_id, episode in fold_rows.groupby("episode_id", sort = False):
            episode = episode.sort_values("row_in_episode")
            chances = episode["stop_probability"].to_numpy()
            distances = episode["goal_distance_cm"].fillna(0.0).to_numpy()
            stop_position = stop_row_of_episode(chances, parameters)
            if stop_position < 0:
                outcome = "never"
            elif distances[stop_position] <= parameters["right_stop_cm"]:
                outcome = "right"
            elif distances[stop_position] > parameters["early_stop_cm"]:
                outcome = "early"
            else:
                outcome = "2_to_3_cm_short"
            peak_position = int(numpy.argmax(chances))
            records.append({
                "track": prediction_path.stem.split("_")[-1],
                "plan_number": str(episode["plan_number"].iloc[0]),
                "path_id": str(episode["path_id"].iloc[0]),
                "score": float(episode["score"].iloc[0]),
                "pictures": int(len(episode)),
                "outcome": outcome,
                "stop_position": int(stop_position),
                "stop_distance_cm": None if stop_position < 0 else round(float(distances[stop_position]), 1),
                "stop_steps_to_end": None if stop_position < 0 else int(episode["steps_to_end"].iloc[stop_position]),
                "peak_chance": round(float(chances[peak_position]), 3),
                "peak_distance_cm": round(float(distances[peak_position]), 1),
                "end_chance": round(float(chances[-1]), 3),
                "last_three_chances": [round(float(value), 3) for value in chances[-3:]],
                "early_passes": int(((chances >= parameters["stop_threshold"]) & (distances > parameters["early_stop_cm"])).sum()),
                "stop_frame_path": None if stop_position < 0 else str(episode["frame_path"].iloc[stop_position]),
                "peak_frame_path": str(episode["frame_path"].iloc[peak_position]),
                "end_frame_path": str(episode["frame_path"].iloc[-1])
            })
    return records



def labelled_tile(frame_path: str, label: str, parameters: Dict) -> numpy.ndarray:
    """Return one camera picture scaled to the tile width, with a white label strip above it."""

    picture = cv2.imread(str(parameters["raw_folder"] / frame_path))
    if picture is None:
        picture = numpy.zeros((240, 320, 3), dtype = numpy.uint8)
    height = int(round(picture.shape[0] * parameters["tile_width_px"] / picture.shape[1]))
    picture = cv2.resize(picture, (parameters["tile_width_px"], height), interpolation = cv2.INTER_AREA)
    strip = numpy.full((parameters["label_height_px"], parameters["tile_width_px"], 3), 255, dtype = numpy.uint8)
    for line_number, text in enumerate(label.split("\n")[:2]):
        cv2.putText(strip, text, (4, 17 + 20 * line_number), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return numpy.vstack([strip, picture])



def write_picture_sheet(records: List[Dict], outcome: str, parameters: Dict = WRONG_STOP_PARAMETERS) -> Path:
    """
    Write one sheet for an outcome: per episode, the picture where the model stopped (or, when it never stopped, where
    its stop chance was highest) beside the picture where the demonstration really stopped.
    """

    rows = []
    for record in (record for record in records if record["outcome"] == outcome):
        name = f"{record['plan_number']} {record['track']} {record['path_id'].split('_')[-1]}"
        if outcome == "never":
            first = labelled_tile(record["peak_frame_path"], f"{name}: highest stop chance\n{record['peak_chance']:.2f} at {record['peak_distance_cm']} cm before the end", parameters)
        else:
            first = labelled_tile(record["stop_frame_path"], f"{name}: model stops here\n{record['stop_distance_cm']} cm before the end", parameters)
        second = labelled_tile(record["end_frame_path"], f"real end (demonstration stopped)\nstop chance {record['end_chance']:.2f}", parameters)
        rows.append(numpy.hstack([first, numpy.full((first.shape[0], 6, 3), 255, dtype = numpy.uint8), second]))
    sheet = numpy.vstack([numpy.vstack([row, numpy.full((8, row.shape[1], 3), 255, dtype = numpy.uint8)]) for row in rows])
    output_path = parameters["output_folder"] / f"stops_{outcome}.png"
    cv2.imwrite(str(output_path), sheet)
    return output_path



def main(parameters: Dict = WRONG_STOP_PARAMETERS) -> None:

    parameters["output_folder"].mkdir(parents = True, exist_ok = True)
    records = episode_outcomes(parameters)
    (parameters["output_folder"] / "stop_outcomes.json").write_text(json.dumps(records, indent = 1), encoding = "utf-8")
    counts = pandas.Series([record["outcome"] for record in records]).value_counts().to_dict()
    print("outcomes:", counts)
    for record in records:
        print(record["track"], record["plan_number"], record["path_id"].split("_")[-1], f"score {record['score']:.0f}",
              record["outcome"], "stop at", record["stop_distance_cm"], "cm |", "peak", record["peak_chance"], "at",
              record["peak_distance_cm"], "cm | end", record["last_three_chances"], "| early passes", record["early_passes"])
    for outcome in ("early", "never", "2_to_3_cm_short"):
        if outcome in counts:
            print("sheet:", write_picture_sheet(records, outcome, parameters))



if __name__ == "__main__":
    main()
