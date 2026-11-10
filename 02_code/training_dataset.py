"""
The training data as one table (9.11.26, Model and training session).

Reads the main data in 03_data/raw/main_data without changing anything there, and writes into the processed folder
(config "processed_folder", 03_data/processed/v01_training_index):

    frames.csv          one row per picture of every indexed episode: the recording's own steps.csv columns, the
                        episode's facts (sheet, plan number, start kind, start end, operator, lighting, score), the
                        magnet tip on the table from the arm model, and measured picture statistics (brightness, red
                        share, dark share outside the robot's own area, sharpness, a difference hash);
    episodes.csv        one row per indexed episode;
    thumbnails.npy      a small grey copy of every picture (config "thumbnail_size"), for finding near-duplicates;
    images_<name>.npy   every picture resized for training (config "image_caches"), RGB, uint8, in frames.csv order;
    build_summary.json  what was read, and what looked wrong.

Score-0 episodes are left out; episodes without a score yet are indexed with an empty score.

Run from 02_code:
    ..\\.venv\\Scripts\\python.exe training_dataset.py
Settings: TRAINING_DATA_PARAMETERS in config.py.
"""

import csv
import datetime
import hashlib
import json
import math
import multiprocessing
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy

from config import RECORDING_PAGE_PARAMETERS, TRAINING_DATA_PARAMETERS
from side_view_test import magnet_tip_and_lean

SCORE_PATTERN = re.compile(r"_score_(\d\d)$")
SESSION_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})_session_(\d\d)_(\w+)$")
TRUE_VALUES = ("1", "true", "True")



def merged_parameters(parameters: Optional[Dict] = None) -> Dict:
    """
    Return ``TRAINING_DATA_PARAMETERS`` with any overrides merged over it.

    Parameters
    ----------
    parameters : dict, optional
        Keys to change, for example another ``raw_folder`` for a test.

    Returns
    -------
    dict
        A new dictionary; the configuration itself is left unchanged.
    """

    merged = dict(TRAINING_DATA_PARAMETERS)
    merged.update(parameters or {})
    return merged



def score_of(folder_name: str) -> Optional[int]:
    """
    Return the review score in an episode folder's name, or ``None`` when the name carries none.

    Parameters
    ----------
    folder_name : str
        For example ``path_04_meander_episode_008_plan_04-01_try_6_score_02``.

    Returns
    -------
    int or None
        The score 0 to 3, or ``None`` for an episode that was never scored.
    """

    match = SCORE_PATTERN.search(folder_name)
    if match is None:
        return None
    return int(match.group(1))



def indexed_episode_folders(parameters: Dict) -> List[Tuple[Path, Optional[int]]]:
    """
    List the episode folders to index: every scored episode except score 0, and every unscored one.

    Parameters
    ----------
    parameters : dict
        Supplies "raw_folder".

    Returns
    -------
    list of tuple of pathlib.Path and int or None
        Each episode folder with its score, sorted by session and folder name.
    """

    raw_folder = Path(parameters["raw_folder"])
    episode_folders = []
    for session_folder in sorted(raw_folder.glob("*_session_*")):
        if not session_folder.is_dir():
            continue
        for episode_folder in sorted(session_folder.glob("path_*_episode_*")):
            if not (episode_folder / "steps.csv").exists():
                continue
            score = score_of(episode_folder.name)
            if score == 0:
                continue
            episode_folders.append((episode_folder, score))
    return episode_folders



def plan_start_kinds() -> Dict[str, Tuple[str, str]]:
    """
    Return the start kind and start end of every plan number in the recording page's episode plan.

    The plan gives either one kind for a range of numbers or one kind per number (for example ``["OL1", "OR1"]``).

    Returns
    -------
    dict of str to tuple of str
        ``"04-07"`` to ``("OR1", "A")``.
    """

    start_kinds = {}
    for path_id, entries in RECORDING_PAGE_PARAMETERS["episode_plan"].items():
        sheet_number = path_id.split("_")[1]
        for first_number, last_number, kinds, start_end, _ in entries:
            for number in range(first_number, last_number + 1):
                kind = kinds[number - first_number] if isinstance(kinds, list) else kinds
                start_kinds[f"{sheet_number}-{number:02d}"] = (kind, start_end)
    return start_kinds



def normalised_lighting(
        lighting_note: Optional[str],
        parameters: Dict
) -> str:
    """
    Return "ceiling", "natural" or "unknown" for a lighting note typed by hand.

    A note that names the ceiling lights counts as ceiling even when it also names daylight, because one note says the
    episode was wrongly marked as natural light.

    Parameters
    ----------
    lighting_note : str or None
        The note in episode.json.
    parameters : dict
        Supplies "lighting_keywords".

    Returns
    -------
    str
        The lighting group.
    """

    text = (lighting_note or "").lower()
    for group in ("ceiling", "natural"):
        if any(keyword in text for keyword in parameters["lighting_keywords"][group]):
            return group
    return "unknown"



def read_json(path: Path) -> Dict:
    """Return the dictionary in a JSON file, or an empty one when the file is missing."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding = "utf-8"))



def picture_flags(episode_folder: Path) -> Dict[str, str]:
    """
    Return the reasons picture_check.py flagged each picture of an episode for, by frame name.

    Parameters
    ----------
    episode_folder : pathlib.Path
        The episode's folder.

    Returns
    -------
    dict of str to str
        ``"000"`` to the reasons joined by " | "; pictures that were not flagged are absent.
    """

    check = read_json(episode_folder / "picture_check.json")
    return {entry["frame"]: " | ".join(entry.get("reasons", [])) for entry in check.get("flagged", [])}



def float_or_none(value: Optional[str]) -> Optional[float]:
    """Return a steps.csv value as a number, or ``None`` when it is empty."""

    if value in (None, ""):
        return None
    return float(value)



def episode_record_and_rows(
        episode_folder: Path,
        score: Optional[int],
        start_kinds: Dict[str, Tuple[str, str]],
        parameters: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    Read one episode into its episode record and its picture rows (without the picture statistics).

    Parameters
    ----------
    episode_folder : pathlib.Path
        The episode's folder.
    score : int or None
        The score in the folder's name.
    start_kinds : dict
        From ``plan_start_kinds``.
    parameters : dict
        Supplies "raw_folder" and "lighting_keywords".

    Returns
    -------
    tuple of dict and list of dict
        The episode record and one row per steps.csv row, in step order.
    """

    raw_folder = Path(parameters["raw_folder"])
    episode = read_json(episode_folder / "episode.json")
    review = read_json(episode_folder / "review.json")
    flags = picture_flags(episode_folder)
    session_match = SESSION_PATTERN.match(episode_folder.parent.name)
    with open(episode_folder / "steps.csv", encoding = "utf-8", newline = "") as steps_file:
        steps = list(csv.DictReader(steps_file))

    plan_number = episode.get("plan_number", "")
    start_kind, plan_end = start_kinds.get(plan_number, ("", ""))
    orientation = episode.get("sheet_orientation", "") or ""
    end_match = re.search(r"end ([AB])", orientation)
    episode_name = SCORE_PATTERN.sub("", episode_folder.name)
    offset = episode.get("deliberate_offset") or {}
    record = {
        "episode_id": f"{episode_folder.parent.name}/{episode_name}",
        "episode_folder": str(episode_folder.relative_to(raw_folder)).replace("\\", "/"),
        "session": episode_folder.parent.name,
        "session_date": session_match.group(1) if session_match else "",
        "operator": episode.get("operator", session_match.group(3) if session_match else ""),
        "path_id": episode.get("path_id", ""),
        "sheet": episode.get("path_id", "")[:7],
        "plan_number": plan_number,
        "start_kind": start_kind,
        "start_end": end_match.group(1) if end_match else plan_end,
        "plan_end": plan_end,
        "sheet_orientation": orientation,
        "start_index": episode.get("start_index", ""),
        "offset_lateral_cm": offset.get("lateral_cm", ""),
        "offset_angle_deg": offset.get("angle_deg", ""),
        "start_push_direction_deg": episode.get("start_push_direction_deg", ""),
        "start_turn_deg": episode.get("start_turn_deg", ""),
        "lighting_note": episode.get("lighting_note", ""),
        "lighting": normalised_lighting(episode.get("lighting_note"), parameters),
        "score": "" if score is None else score,
        "is_scored": int(score is not None),
        "review_issues": ";".join(review.get("issues", [])),
        "review_note": review.get("note", ""),
        "reviewer": review.get("reviewer", ""),
        "mode": episode.get("mode", ""),
        "status": episode.get("status", ""),
        "outcome": episode.get("outcome", ""),
        "hand_corrections": episode.get("hand_corrections", ""),
        "camera_mount": episode.get("camera_mount", ""),
        "camera_shift_px": episode.get("camera_shift_px", ""),
        "code_commit": episode.get("code_commit", ""),
        "started": episode.get("started", ""),
        "finished": episode.get("finished", ""),
        "step_count": len(steps),
        "flagged_picture_count": len(flags)
    }

    rows = []
    last_step = len(steps) - 1
    previous_md5 = None
    for position, step in enumerate(steps):
        row = dict(record)
        row.update(step)
        joints = [float(step[f"joint_{number}"]) for number in range(1, 7)]
        tip, lean_from_vertical_deg = magnet_tip_and_lean(joints)
        forward_m = float_or_none(step.get("tip_forward_m")) or 0.0
        left_m = float_or_none(step.get("tip_left_m")) or 0.0
        frame_name = Path(step["frame"]).stem
        move_return_time = float_or_none(step.get("move_return_time"))
        capture_time = float_or_none(step.get("capture_time"))
        decision_time = float_or_none(step.get("decision_time"))
        row.update({
            "frame_path": f"{record['episode_folder']}/{step['frame']}",
            "row_in_episode": position,
            "step_fraction": round(position / last_step, 4) if last_step > 0 else 0.0,
            "steps_to_end": last_step - position,
            "tip_x_m": round(float(tip[0]), 5),
            "tip_y_m": round(float(tip[1]), 5),
            "tip_z_m": round(float(tip[2]), 5),
            "rod_from_vertical_deg": round(lean_from_vertical_deg, 2),
            "move_length_m": round(math.hypot(forward_m, left_m), 5),
            "move_angle_deg": round(math.degrees(math.atan2(left_m, forward_m)), 2),
            "capture_after_move_s": "" if None in (capture_time, move_return_time) else round(capture_time - move_return_time, 3),
            "decision_after_capture_s": "" if None in (decision_time, capture_time) else round(decision_time - capture_time, 3),
            "is_same_md5_as_previous": int(previous_md5 is not None and step.get("frame_md5") == previous_md5),
            "picture_flags": flags.get(frame_name, ""),
            "is_flagged": int(frame_name in flags),
            "is_mistake_row": int(step.get("is_mistake") in TRUE_VALUES),
            "is_lost_row": int(step.get("is_lost") in TRUE_VALUES)
        })
        previous_md5 = step.get("frame_md5")
        rows.append(row)

    for position, row in enumerate(rows):
        next_time = float_or_none(rows[position + 1]["unix_time"]) if position + 1 < len(rows) else None
        this_time = float_or_none(row["unix_time"])
        row["seconds_to_next_row"] = "" if None in (next_time, this_time) else round(next_time - this_time, 3)
    return record, rows



def difference_hash(grey_picture: numpy.ndarray) -> str:
    """
    Return the 64-bit difference hash of a grey picture as 16 hexadecimal digits.

    The picture is shrunk to 9 x 8 pixels; each bit says whether a pixel is brighter than its right-hand neighbour.
    Pictures whose hashes differ in only a few bits look almost the same.
    """

    small = cv2.resize(grey_picture, (9, 8), interpolation = cv2.INTER_AREA).astype(numpy.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"

# Reference
# Source: Neal Krawetz, "Kind of Like That" - the difference hash (shrink to 9 x 8, compare neighbouring pixels).
# Link: https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html



def picture_statistics(
        picture_bgr: numpy.ndarray,
        parameters: Dict
) -> Dict:
    """
    Measure simple statistics of one camera picture.

    Parameters
    ----------
    picture_bgr : numpy.ndarray
        The picture as OpenCV reads it.
    parameters : dict
        Supplies the red and dark colour limits and "robot_area_px".

    Returns
    -------
    dict
        "brightness_mean" and "brightness_std" (grey, 0 to 255), "red_share" (pixels as red as the puck),
        "dark_share_outside_robot" (dark pixels outside the robot's own area of the picture), "sharpness_measured"
        (variance of the Laplacian, as the recording tool logs it) and "difference_hash".
    """

    grey = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    low_edge, high_edge = parameters["red_hue_edges"]
    is_red = (((hue < low_edge) | (hue > high_edge)) & (saturation >= parameters["red_minimum_saturation"])
              & (value >= parameters["red_minimum_value"]))
    is_dark = value < parameters["dark_value_below"]
    left, top, right, bottom = parameters["robot_area_px"]
    outside_robot = numpy.ones(grey.shape, dtype = bool)
    outside_robot[top:bottom, left:right] = False
    return {
        "brightness_mean": round(float(grey.mean()), 2),
        "brightness_std": round(float(grey.std()), 2),
        "red_share": round(float(is_red.mean()), 5),
        "dark_share_outside_robot": round(float(is_dark[outside_robot].mean()), 5),
        "sharpness_measured": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 1),
        "difference_hash": difference_hash(grey)
    }



def process_picture_chunk(arguments: Tuple) -> List[Tuple[int, Dict]]:
    """
    Measure and cache one chunk of pictures, in a worker process.

    Every worker opens the shared cache files and writes only the rows of its own chunk, so no picture travels back to
    the main process.

    Parameters
    ----------
    arguments : tuple
        ``(chunk, parameters, cache_paths, frame_count)``: the chunk is a list of ``(index, picture_path)``.

    Returns
    -------
    list of tuple of int and dict
        Each index with its statistics, the file's own MD5 and whether the picture could be read.
    """

    chunk, parameters, cache_paths, frame_count = arguments
    thumbnail_width, thumbnail_height = parameters["thumbnail_size"]
    thumbnails = numpy.lib.format.open_memmap(cache_paths["thumbnails"], mode = "r+")
    caches = {name: numpy.lib.format.open_memmap(cache_paths[name], mode = "r+") for name in parameters["image_caches"]}
    results = []
    for index, picture_path in chunk:
        picture_bytes = Path(picture_path).read_bytes() if Path(picture_path).exists() else b""
        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, dtype = numpy.uint8), cv2.IMREAD_COLOR) if picture_bytes else None
        if picture is None:
            results.append((index, {"is_picture_readable": 0, "file_md5": ""}))
            continue
        statistics = picture_statistics(picture, parameters)
        statistics["is_picture_readable"] = 1
        statistics["file_md5"] = hashlib.md5(picture_bytes).hexdigest()
        statistics["picture_width"] = picture.shape[1]
        statistics["picture_height"] = picture.shape[0]
        grey = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        thumbnails[index] = cv2.resize(grey, (thumbnail_width, thumbnail_height), interpolation = cv2.INTER_AREA)
        rgb = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        for name, (width, height) in parameters["image_caches"].items():
            caches[name][index] = cv2.resize(rgb, (width, height), interpolation = cv2.INTER_AREA)
        results.append((index, statistics))
    thumbnails.flush()
    for cache in caches.values():
        cache.flush()
    return results



def write_csv(
        path: Path,
        rows: List[Dict]
) -> None:
    """Write dictionaries to a CSV file, with every key that occurs in any row as a column, in first-seen order."""

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", encoding = "utf-8", newline = "") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = columns, extrasaction = "ignore")
        writer.writeheader()
        writer.writerows(rows)



def build_index(parameters: Optional[Dict] = None) -> Dict:
    """
    Build the frames and episodes tables and the picture caches.

    Parameters
    ----------
    parameters : dict, optional
        Overrides for ``TRAINING_DATA_PARAMETERS``.

    Returns
    -------
    dict
        The build summary that is also written to build_summary.json.
    """

    parameters = merged_parameters(parameters)
    processed_folder = Path(parameters["processed_folder"])
    processed_folder.mkdir(parents = True, exist_ok = True)
    start_kinds = plan_start_kinds()

    episode_records = []
    frame_rows = []
    for episode_folder, score in indexed_episode_folders(parameters):
        record, rows = episode_record_and_rows(episode_folder, score, start_kinds, parameters)
        episode_records.append(record)
        frame_rows.extend(rows)
    for index, row in enumerate(frame_rows):
        row["frame_index"] = index

    frame_count = len(frame_rows)
    thumbnail_width, thumbnail_height = parameters["thumbnail_size"]
    cache_paths = {"thumbnails": str(processed_folder / "thumbnails.npy")}
    numpy.lib.format.open_memmap(cache_paths["thumbnails"], mode = "w+", dtype = numpy.uint8,
                                 shape = (frame_count, thumbnail_height, thumbnail_width)).flush()
    for name, (width, height) in parameters["image_caches"].items():
        cache_paths[name] = str(processed_folder / f"images_{name}.npy")
        numpy.lib.format.open_memmap(cache_paths[name], mode = "w+", dtype = numpy.uint8,
                                     shape = (frame_count, height, width, 3)).flush()

    raw_folder = Path(parameters["raw_folder"])
    jobs = [(row["frame_index"], str(raw_folder / row["frame_path"])) for row in frame_rows]
    chunk_size = max(1, math.ceil(len(jobs) / (parameters["worker_count"] * 8)))
    chunks = [jobs[start:start + chunk_size] for start in range(0, len(jobs), chunk_size)]
    worker_arguments = [(chunk, parameters, cache_paths, frame_count) for chunk in chunks]
    with multiprocessing.Pool(processes = parameters["worker_count"]) as pool:
        for results in pool.imap_unordered(process_picture_chunk, worker_arguments):
            for index, statistics in results:
                frame_rows[index].update(statistics)

    unreadable = [row["frame_path"] for row in frame_rows if not row.get("is_picture_readable")]
    md5_mismatches = [row["frame_path"] for row in frame_rows
                      if row.get("is_picture_readable") and row.get("frame_md5") and row["frame_md5"] != row["file_md5"]]
    per_episode_brightness = {}
    for row in frame_rows:
        per_episode_brightness.setdefault(row["episode_id"], []).append(row.get("brightness_mean", 0.0))
    for record in episode_records:
        values = per_episode_brightness.get(record["episode_id"], [0.0])
        record["brightness_median"] = round(float(numpy.median(values)), 2)

    write_csv(processed_folder / "frames.csv", frame_rows)
    write_csv(processed_folder / "episodes.csv", episode_records)
    summary = {
        "built": datetime.datetime.now().isoformat(timespec = "seconds"),
        "raw_folder": str(raw_folder),
        "episodes": len(episode_records),
        "scored_episodes": sum(record["is_scored"] for record in episode_records),
        "unscored_episodes": [record["episode_id"] for record in episode_records if not record["is_scored"]],
        "frames": frame_count,
        "frames_in_scored_episodes": sum(1 for row in frame_rows if row["is_scored"]),
        "unreadable_pictures": unreadable,
        "md5_mismatches": md5_mismatches,
        "image_caches": {name: list(size) for name, size in parameters["image_caches"].items()},
        "thumbnail_size": list(parameters["thumbnail_size"])
    }
    (processed_folder / "build_summary.json").write_text(json.dumps(summary, indent = 2), encoding = "utf-8")
    return summary



def main() -> None:

    summary = build_index()
    print(json.dumps({key: value for key, value in summary.items() if key not in ("unreadable_pictures", "md5_mismatches")}, indent = 2))
    print(f"unreadable pictures: {len(summary['unreadable_pictures'])}, MD5 mismatches: {len(summary['md5_mismatches'])}")



if __name__ == "__main__":
    main()
