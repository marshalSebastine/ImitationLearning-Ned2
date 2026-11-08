"""
Offline check (8.11.26, Emranul's question): can Claude act as a third operator, naming the next action from one
forearm-camera picture the way Emranul and Marshal do? Nothing connects to the robot; the recorded data is only read.

Two steps, run from this folder:
    ..\\.venv\\Scripts\\python.exe claude_operator_study.py select [data_folder]
        picks the fixed test set, writes the picture list (no labels) to the study folder and the labels to the
        answer key in 03_data, which is not read until the predictions are written;
    ..\\.venv\\Scripts\\python.exe claude_operator_study.py score
        joins predictions.csv (written by hand from the pictures) with the answer key into test_set.csv and prints
        the scores, per sheet and per situation, beside the "always forward" baseline.
Settings: CLAUDE_OPERATOR_STUDY_PARAMETERS and RECORDING_PARAMETERS["actions"] in config.py.
"""

import csv
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from config import CLAUDE_OPERATOR_STUDY_PARAMETERS, RECORDING_PARAMETERS

TRUE_VALUES = ("1", "true", "True")
SITUATIONS = ["start", "straight", "bend", "near_goal", "stop"]



def read_steps(episode_folder: Path) -> list:
    """The usable rows of an episode's steps.csv: mistakes and lost-puck rows left out, a picture on disk for each."""

    with open(episode_folder / "steps.csv", newline = "") as steps_file:
        rows = list(csv.DictReader(steps_file))
    usable_rows = []
    for row in rows:
        if row.get("is_mistake") in TRUE_VALUES or row.get("is_lost") in TRUE_VALUES:
            continue
        if not (episode_folder / "frames" / f"{int(row['step']):03d}.jpg").exists():
            continue
        usable_rows.append(row)
    return usable_rows



def heading_change_deg(
        rows: list,
        index: int,
        window: int
        ) -> float:
    """The change in the rod's table direction over ``window`` rows before and after row ``index``, in degrees."""

    if index - window < 0 or index + window >= len(rows):
        return 0.0
    points = [(float(rows[position]["x"]), float(rows[position]["y"])) for position in (index - window, index, index + window)]
    before = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    after = (points[2][0] - points[1][0], points[2][1] - points[1][1])
    if math.hypot(*before) < 0.005 or math.hypot(*after) < 0.005:
        return 0.0
    change = math.degrees(math.atan2(after[1], after[0]) - math.atan2(before[1], before[0]))
    return abs((change + 180) % 360 - 180)



def describe_row(
        rows: list,
        index: int,
        is_start: bool,
        parameters: dict
        ) -> dict:
    """The answer-key entry for one row: its label, the next few labels, and the situation it is in."""

    row = rows[index]
    stop_row = rows[-1]
    goal_distance_cm = 100 * math.hypot(float(row["x"]) - float(stop_row["x"]), float(row["y"]) - float(stop_row["y"]))
    turn_deg = heading_change_deg(rows = rows, index = index, window = parameters["bend_window_steps"])
    if row["action"] == "stop":
        situation = "stop"
    elif is_start:
        situation = "start"
    elif goal_distance_cm <= parameters["near_goal_cm"]:
        situation = "near_goal"
    elif turn_deg > parameters["bend_threshold_deg"]:
        situation = "bend"
    else:
        situation = "straight"
    series = [rows[position]["action"] for position in range(index, min(index + parameters["series_length"], len(rows)))]
    return {
        "step": int(row["step"]),
        "situation": situation,
        "goal_distance_cm": round(goal_distance_cm, 1),
        "turn_deg": round(turn_deg, 1),
        "forearm_turn_deg": row.get("forearm_turn_deg", ""),
        "label": row["action"],
        "label_series": " ".join(series)
    }



def select_test_set(parameters: dict) -> None:
    """Pick the fixed test set and write the blind picture list and the hidden answer key."""

    generator = random.Random(parameters["seed"])
    episodes_by_sheet = defaultdict(list)
    for session_folder in sorted(Path(parameters["data_folder"]).iterdir()):
        if not session_folder.is_dir():
            continue
        for episode_folder in sorted(session_folder.iterdir()):
            match = re.search(parameters["episode_pattern"], episode_folder.name)
            if match and episode_folder.is_dir():
                rows = read_steps(episode_folder)
                if len(rows) >= parameters["minimum_frames"] and rows[-1]["action"] == "stop":
                    episodes_by_sheet[match.group(1)].append((episode_folder, rows))

    entries = []
    for sheet, episodes in sorted(episodes_by_sheet.items()):
        count = min(parameters["episodes_per_sheet"], len(episodes))
        chosen = [episodes[round(position * (len(episodes) - 1) / max(count - 1, 1))] for position in range(count)]
        for episode_folder, rows in chosen:
            moving_indices = list(range(len(rows) - 1))
            indices = {round(fraction * (len(moving_indices) - 1)) for fraction in parameters["positions_per_episode"]}
            stop_row = rows[-1]
            near_goal_indices = [
                index for index in moving_indices
                if 100 * math.hypot(float(rows[index]["x"]) - float(stop_row["x"]), float(rows[index]["y"]) - float(stop_row["y"]))
                <= parameters["near_goal_cm"] and index not in indices
            ]
            if near_goal_indices:
                indices.add(generator.choice(near_goal_indices))
            indices.add(len(rows) - 1)
            for index in sorted(indices):
                entry = describe_row(rows = rows, index = index, is_start = index == 0, parameters = parameters)
                entry.update({
                    "sheet": sheet,
                    "episode": f"{episode_folder.parent.name}/{episode_folder.name}",
                    "picture_path": str(episode_folder / "frames" / f"{entry['step']:03d}.jpg")
                })
                entries.append(entry)

    # Shuffled so the pictures are not seen in episode order; every n-th also asks for a series of actions.
    generator.shuffle(entries)
    for number, entry in enumerate(entries, start = 1):
        entry["test_id"] = number
        entry["asks_series"] = int(number % parameters["series_every_nth"] == 0)

    parameters["study_folder"].mkdir(parents = True, exist_ok = True)
    write_rows(path = parameters["study_folder"] / "test_pictures.csv", rows = entries,
               columns = ["test_id", "picture_path", "asks_series"])
    Path(parameters["answer_key_path"]).parent.mkdir(parents = True, exist_ok = True)
    write_rows(path = parameters["answer_key_path"], rows = entries, columns = list(entries[0].keys()))
    print(f"{len(entries)} test pictures from {sum(len(episodes) for episodes in episodes_by_sheet.values())} episodes")



def write_rows(
        path: Path,
        rows: list,
        columns: list
        ) -> None:
    """Write ``rows`` (dictionaries) to a CSV with the given ``columns``."""

    with open(path, "w", newline = "") as output_file:
        writer = csv.DictWriter(output_file, fieldnames = columns, extrasaction = "ignore")
        writer.writeheader()
        writer.writerows(rows)



def is_close(
        prediction: str,
        label: str,
        parameters: dict
        ) -> bool:
    """True for the same action, or two moving actions within "close_angle_deg" of each other."""

    if prediction == label:
        return True
    angles = RECORDING_PARAMETERS["actions"]
    if angles.get(prediction) is None or angles.get(label) is None:
        return False
    difference = abs((angles[prediction] - angles[label] + 180) % 360 - 180)
    return difference <= parameters["close_angle_deg"]



def summarise(rows: list, prediction_column: str, parameters: dict) -> str:
    """Exact, close and stop-right percentages of ``prediction_column`` over ``rows``."""

    count = len(rows)
    if count == 0:
        return "n = 0"
    exact = sum(row[prediction_column] == row["label"] for row in rows)
    close = sum(is_close(prediction = row[prediction_column], label = row["label"], parameters = parameters) for row in rows)
    stop_right = sum((row[prediction_column] == "stop") == (row["label"] == "stop") for row in rows)
    return f"n = {count:3d}   exact {100 * exact / count:5.1f} %   close {100 * close / count:5.1f} %   stop right {100 * stop_right / count:5.1f} %"



def score_predictions(parameters: dict) -> None:
    """Join the predictions with the answer key, write test_set.csv and print the scores."""

    with open(parameters["answer_key_path"], newline = "") as key_file:
        key = {row["test_id"]: row for row in csv.DictReader(key_file)}
    with open(parameters["study_folder"] / "predictions.csv", newline = "") as predictions_file:
        predictions = {row["test_id"]: row for row in csv.DictReader(predictions_file)}

    rows = []
    for test_id, entry in key.items():
        prediction = predictions[test_id]
        entry.update({column: prediction.get(column, "") for column in ["prediction", "reason", "prediction_series", "seconds"]})
        entry["baseline"] = "forward"
        entry["exact"] = int(entry["prediction"] == entry["label"])
        entry["close"] = int(is_close(prediction = entry["prediction"], label = entry["label"], parameters = parameters))
        entry["stop_right"] = int((entry["prediction"] == "stop") == (entry["label"] == "stop"))
        rows.append(entry)
    columns = [
        "test_id", "sheet", "episode", "step", "picture_path", "situation", "goal_distance_cm", "turn_deg",
        "forearm_turn_deg", "label", "prediction", "reason", "exact", "close", "stop_right", "label_series",
        "prediction_series", "seconds"
    ]
    write_rows(path = parameters["study_folder"] / "test_set.csv", rows = rows, columns = columns)

    # "small_motion" is written by the tool when the tip moved under 3 mm (the lowering at a start, small touch-ups); no
    # operator picks it from the picture, so the scores are also given without those rows.
    chosen_rows = [row for row in rows if row["label"] != "small_motion"]
    for row_set_name, row_set in [("all labels", rows), ("without small_motion", chosen_rows)]:
        for name, column in [("Claude", "prediction"), ("always forward", "baseline")]:
            print(f"\n{name}, {row_set_name}\n  all          {summarise(rows = row_set, prediction_column = column, parameters = parameters)}")
            for sheet in sorted({row["sheet"] for row in row_set}):
                subset = [row for row in row_set if row["sheet"] == sheet]
                print(f"  {sheet:<12} {summarise(rows = subset, prediction_column = column, parameters = parameters)}")
            for situation in SITUATIONS:
                subset = [row for row in row_set if row["situation"] == situation]
                print(f"  {situation:<12} {summarise(rows = subset, prediction_column = column, parameters = parameters)}")

    series_rows = [row for row in rows if row["prediction_series"]]
    position_hits = [0] * parameters["series_length"]
    position_counts = [0] * parameters["series_length"]
    all_hits = 0
    for row in series_rows:
        labels = row["label_series"].split()
        guesses = row["prediction_series"].split()
        for position, label in enumerate(labels):
            position_counts[position] += 1
            position_hits[position] += int(position < len(guesses) and guesses[position] == label)
        all_hits += int(guesses[:len(labels)] == labels)
    print(f"\nSeries of {parameters['series_length']} ({len(series_rows)} pictures): exact per position "
          + ", ".join(f"{100 * hits / max(count, 1):.0f} %" for hits, count in zip(position_hits, position_counts))
          + f"; whole series exact {100 * all_hits / max(len(series_rows), 1):.0f} %")

    label_counts = defaultdict(int)
    for row in rows:
        label_counts[(row["label"], row["prediction"])] += 1
    print("\nlabel -> prediction counts")
    for (label, prediction), count in sorted(label_counts.items()):
        print(f"  {label:<14} -> {prediction:<14} {count}")
    seconds = [float(row["seconds"]) for row in rows if row["seconds"]]
    if seconds:
        print(f"\nseconds per decision: mean {sum(seconds) / len(seconds):.1f}")



if __name__ == "__main__":
    study_parameters = dict(CLAUDE_OPERATOR_STUDY_PARAMETERS)
    if len(sys.argv) > 2:
        study_parameters.update({"data_folder": Path(sys.argv[2])})
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        select_test_set(parameters = study_parameters)
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        score_predictions(parameters = study_parameters)
    else:
        print(__doc__)
