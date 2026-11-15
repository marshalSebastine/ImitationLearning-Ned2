"""
How many recorded pictures lie around a correction by hand (the h key of the exact replay), 15.11.26.

Asked by Emranul: "in more than 80 % of the episodes we had to press h and bring the puck back on the line; why is it
280 pictures only?" The 280 counted only the first eight pictures of the 35 episodes that start beside the line. This
script counts the corrections themselves.

A correction is found in the step table where the waypoint counter starts again inside an episode: after an h press
the tool replays a newly guided path from its first waypoint. Pictures before that row show the replay going on while
the puck drifts; pictures after it show the robot replaying the correction.

Run from 02_code:  python "<this file>"  [output folder for the picture strips]
"""

import sys
from pathlib import Path

import cv2
import numpy
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "02_code"))

import config
from training_targets import load_training_frames


CHECK_PARAMETERS = {
    "run_name": "resnet18_full_new_shapes",
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "row_ranges": {"8 steps before an h press": [-8, -1], "3 steps before an h press": [-3, -1],
                   "first row after (robot stands)": [0, 0], "steps 2 to 8 after": [1, 7]},
    "strip_offsets": [-8, -5, -3, -1, 0, 1, 3, 5, 8],
    "strip_count": 8,
    "tile_size": [320, 240],
}


def marked_frames(parameters: dict) -> pandas.DataFrame:
    """
    All pictures of the run's training index, in episode order, with the corrections marked.

    Parameters
    ----------
    parameters : dict
        ``CHECK_PARAMETERS``; "run_name" names the training run whose index and cleaning are used.

    Returns
    -------
    pandas.DataFrame
        The index's frames with "is_restart" (first row after an h press) and "is_in_training" (kept by the run's
        cleaning).
    """

    run = config.TRAINING_RUNS[parameters["run_name"]]
    target_parameters = dict(config.TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in run.items() if key in target_parameters})
    data_parameters = dict(config.TRAINING_DATA_PARAMETERS)
    data_parameters.update({key: value for key, value in run.items() if key in data_parameters})
    frames = pandas.read_csv(Path(data_parameters["processed_folder"]) / "frames.csv", low_memory = False)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    previous_waypoint = frames.groupby("episode_id")["waypoint"].shift(1)
    frames["is_restart"] = (frames["waypoint"] < previous_waypoint) & previous_waypoint.notna()
    kept = load_training_frames(target_parameters, data_parameters)
    kept_keys = set(zip(kept["episode_id"], kept["row_in_episode"]))
    frames["is_in_training"] = [key in kept_keys for key in zip(frames["episode_id"], frames["row_in_episode"])]
    return frames


def rows_around(frames: pandas.DataFrame, low: int, high: int) -> list:
    """The rows ``low`` to ``high`` steps from any correction, inside the same episode, each row once."""

    episode_of = frames["episode_id"].to_numpy()
    rows = set()
    for restart_row in frames.index[frames["is_restart"]]:
        for offset in range(low, high + 1):
            row = restart_row + offset
            if 0 <= row < len(frames) and episode_of[row] == episode_of[restart_row]:
                rows.add(row)
    return sorted(rows)


def picture_strips(frames: pandas.DataFrame, parameters: dict, output_folder: Path) -> None:
    """Write two picture sheets: evenly chosen corrections, each with its pictures before and after the h press."""

    restart_rows = frames.index[frames["is_restart"]].to_numpy()
    chosen = restart_rows[numpy.linspace(0, len(restart_rows) - 1, parameters["strip_count"]).astype(int)]
    episode_of = frames["episode_id"].to_numpy()
    width, height = parameters["tile_size"]
    strips = []
    for restart_row in chosen:
        tiles = []
        for offset in parameters["strip_offsets"]:
            row = restart_row + offset
            tile = numpy.full((height, width, 3), 40, dtype = numpy.uint8)
            if 0 <= row < len(frames) and episode_of[row] == episode_of[restart_row]:
                record = frames.iloc[row]
                picture = cv2.imread(str(parameters["raw_folder"] / str(record["frame_path"])))
                if picture is not None:
                    tile = cv2.resize(picture, (width, height))
                colour = (0, 200, 255) if offset < 0 else (0, 255, 0)
                text = f"{offset:+d}  {record['action']}"
                cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(tile, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            tiles.append(tile)
        strip = numpy.hstack(tiles)
        first = frames.iloc[restart_row]
        title = f"{first['sheet']}  {first['plan_number']}  step {int(first['row_in_episode'])}   orange = before the h press, green = after it"
        bar = numpy.full((26, strip.shape[1], 3), 255, dtype = numpy.uint8)
        cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        strips.append(numpy.vstack([bar, strip]))
    half = len(strips) // 2
    for number, part in enumerate((strips[:half], strips[half:]), start = 1):
        cv2.imwrite(str(output_folder / f"hand_corrections_strips_{number}.jpg"), numpy.vstack(part), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main() -> None:
    """Print the counts and write the picture strips."""

    parameters = dict(CHECK_PARAMETERS)
    output_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    frames = marked_frames(parameters)
    episodes = frames.drop_duplicates("episode_id")
    print(f"episodes {len(episodes)}, pictures {len(frames)}, in the training set {int(frames['is_in_training'].sum())}")
    print(f"h presses by the episodes' own count: {int(episodes['hand_corrections'].fillna(0).sum())} in "
          f"{int((episodes['hand_corrections'].fillna(0) > 0).sum())} episodes; found in the step tables: {int(frames['is_restart'].sum())}")
    for label, (low, high) in parameters["row_ranges"].items():
        part = frames.iloc[rows_around(frames, low, high)]
        print(f"{label:32s} pictures {len(part):5d} | in the training set {int(part['is_in_training'].sum()):5d} | marked as a mistake (u) {int(part['is_mistake_row'].sum())}")
    is_beside_start = frames["start_kind"].astype(str).str.startswith(("OL", "OR")) & (frames["row_in_episode"] < 8)
    print(f"first 8 pictures of the starts beside the line (the old count): {int(is_beside_start.sum())}")
    picture_strips(frames, parameters, output_folder)


if __name__ == "__main__":
    main()
