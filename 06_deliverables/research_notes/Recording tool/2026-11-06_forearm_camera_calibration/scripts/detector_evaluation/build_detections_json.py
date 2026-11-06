"""
Detect the puck in every picture of episodes 037 to 045 and write detections_037_045.json.

For every frame (key: the frame's full path) the file holds the detection dictionary
or null, the episode, the frame index, and the arm's state when the frame was taken:
the joints (and the rod's pose) from the frame's row in steps.csv, matched by the
picture's md5. Episode 041 has no steps.csv (the recording tool crashed before
writing it); its joints are rebuilt from guided_path.csv with the exact replay's
waypoint rule (rebuild_exact_replay_joints.py, checked on episode 043) and marked so.
Pictures without a detection carry the category given to them by eye
(miss_review.py). The extra pictures of the same camera (camera checks, correction
references) are listed in a section of their own.
"""

import os
import sys
import csv
import json
import time
import hashlib
from collections import Counter, OrderedDict

import cv2
import numpy

from frame_catalogue import catalogue_pictures, EPISODE_FOLDERS, RAW_DATA_FOLDER
from puck_detector import PuckDetector, PUCK_DETECTOR_PARAMETERS
from miss_review import review_for_missed_frame, MISSED_EXTRA_PICTURE_REVIEW, START_PICTURE_REVIEW
from rebuild_exact_replay_joints import exact_waypoints, load_guided_path


OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PARAMETERS = {
    "file_name": "detections_037_045.json",
    "float_decimals": 3,
    # columns of the steps.csv row copied beside the joints
    "copied_step_columns": ["step", "action", "source", "forearm_turn_deg", "picture_heading_deg", "capture_time",
                            "x", "y", "z", "roll", "pitch", "yaw"],
    # measurements of the detection that are internal to the scoring and left out of the file
    "dropped_detection_keys": ["bottom_v_analysis", "cast_share", "label"]
}


def rounded(value, decimals: int):
    """Round floats (also inside lists and dictionaries) for a compact file."""

    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, (numpy.floating, numpy.integer)):
        return rounded(value.item(), decimals)
    if isinstance(value, list):
        return [rounded(item, decimals) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item, decimals) for key, item in value.items()}
    return value


def read_steps(episode_folder: str) -> dict:
    """Return the rows of steps.csv keyed by the frame's md5, or an empty dictionary."""

    steps_path = os.path.join(episode_folder, "steps.csv")
    if not os.path.exists(steps_path):
        return {}
    with open(steps_path, newline = "", encoding = "utf-8") as steps_file:
        return {row["frame_md5"]: row for row in csv.DictReader(steps_file)}


def rebuilt_joints_041() -> dict:
    """Return the rebuilt joints of every frame of episode 041, keyed by frame index."""

    episode_folder = f"{RAW_DATA_FOLDER}/{EPISODE_FOLDERS['041']}"
    waypoints = exact_waypoints(load_guided_path(episode_folder))
    with open(os.path.join(episode_folder, "steps_from_terminal_log.csv"), newline = "",
              encoding = "utf-8") as log_file:
        log_rows = {int(row["step"]): row for row in csv.DictReader(log_file)}
    joints_by_frame = {}
    for frame_index, row in log_rows.items():
        joints = waypoints[0] if frame_index == 0 else waypoints[min(frame_index - 1, len(waypoints) - 1)]
        joints_by_frame[frame_index] = {
            "joints": [round(float(value), 5) for value in joints],
            "log_row": row
        }
    return joints_by_frame


def main():

    cv2.setNumThreads(1)
    detector = PuckDetector()
    pictures = catalogue_pictures()
    decimals = OUTPUT_PARAMETERS["float_decimals"]
    steps_by_episode = {episode: read_steps(f"{RAW_DATA_FOLDER}/{folder}") for episode, folder in EPISODE_FOLDERS.items()}
    joints_041 = rebuilt_joints_041()

    frames = OrderedDict()
    extra_pictures = OrderedDict()
    durations_ms = []
    md5_mismatches = []
    for entry in pictures:
        with open(entry["path"], "rb") as picture_file:
            picture_bytes = picture_file.read()
        picture_md5 = hashlib.md5(picture_bytes).hexdigest()
        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        start = time.perf_counter()
        detection = detector.detect(picture)
        durations_ms.append(1000.0 * (time.perf_counter() - start))
        if detection is not None:
            for key in OUTPUT_PARAMETERS["dropped_detection_keys"]:
                detection.pop(key, None)
            detection = rounded(detection, decimals)

        record = OrderedDict()
        record["episode"] = entry["episode"]
        record["frame_index"] = entry["frame_index"]
        record["joints"] = None
        record["joints_source"] = None
        record["step_row"] = None
        steps = steps_by_episode[entry["episode"]]
        if picture_md5 in steps:
            row = steps[picture_md5]
            record["joints"] = [float(row[f"joint_{index}"]) for index in range(1, 7)]
            record["joints_source"] = "steps.csv"
            record["step_row"] = {column: (float(row[column]) if column not in ("action", "source") else row[column])
                                  for column in OUTPUT_PARAMETERS["copied_step_columns"] if row.get(column, "") != ""}
            record["step_row"]["step"] = int(row["step"])
            if entry["kind"] == "frame" and row["frame"] != f"frames/{entry['frame_index']:03d}.jpg":
                md5_mismatches.append(entry["path"])
        elif entry["episode"] == "041" and entry["kind"] == "frame":
            rebuilt = joints_041[entry["frame_index"]]
            record["joints"] = rebuilt["joints"]
            record["joints_source"] = "rebuilt_from_guided_path"
            log_row = rebuilt["log_row"]
            record["step_row"] = {
                "step": int(log_row["step"]),
                "action": log_row["action"],
                "source": log_row["source"],
                "forearm_turn_deg": float(log_row["forearm_turn_deg"]),
                "picture_heading_deg": float(log_row["picture_heading_deg"]),
                "x": round(float(log_row["x_cm"]) / 100.0, 4),
                "y": round(float(log_row["y_cm"]) / 100.0, 4),
                "note": "from steps_from_terminal_log.csv; x and y printed to 1 mm"
            }
        elif entry["kind"] == "frame":
            md5_mismatches.append(entry["path"])
        record["picture_md5"] = picture_md5
        record["detection"] = detection

        key = entry["path"]
        if entry["kind"] == "frame":
            if detection is None:
                category, note = review_for_missed_frame(entry["episode"], entry["frame_index"])
                record["no_detection_review"] = {"category": category, "note": note}
            start_review = START_PICTURE_REVIEW.get(f"{entry['episode']}/{entry['frame_index']:03d}")
            if start_review:
                record["start_picture_review"] = start_review
            frames[key] = record
        else:
            record["kind"] = entry["kind"]
            if record["joints"] is None:
                record["joints_note"] = ("not logged: the camera check is taken at the start pose before frame 000; "
                                         "the correction reference is the picture shown when the correction began")
            short_key = f"{entry['episode']}/{entry['kind']}"
            if detection is None and short_key in MISSED_EXTRA_PICTURE_REVIEW:
                category, note = MISSED_EXTRA_PICTURE_REVIEW[short_key]
                record["no_detection_review"] = {"category": category, "note": note}
            if short_key in START_PICTURE_REVIEW:
                record["start_picture_review"] = START_PICTURE_REVIEW[short_key]
            extra_pictures[key] = record

    output = OrderedDict()
    output["description"] = (
        "Red puck detections in every picture of episodes 037 to 045 (forearm camera, bare camera on the forearm "
        "cover), made by puck_detector.py with the parameters below. 'frames' is keyed by the frame's full path; "
        "'detection' is null where no puck was found. 'joints' (radians, joint_1 to joint_6) and 'step_row' come "
        "from the frame's row in steps.csv, matched by the picture's md5 (the arm's state when the frame was "
        "taken); for episode 041 they are rebuilt from guided_path.csv ('joints_source' says which). Pixel "
        "coordinates are in the full 640 x 480 picture, u to the right, v down, origin at the top-left pixel's "
        "centre. 'centre_u/v' is the best centre estimate (centre_source 'blob': mean of the visible blob's "
        "pixels; 'circle_fit': centre of a circle fitted to the free outline of a partly hidden puck). "
        "'bottom_u/v' is the blob's lowest row and its mean column there; 'is_bottom_clear' is true when light "
        "paper lies right below that point (it is then the puck's own outline) and false when something dark or "
        "blue lies there (the block, a bracket, but also the black line or a shadow) or the picture's edge, so "
        "false does not prove that part of the puck is hidden. 'is_partial' marks detections accepted only by the "
        "looser tests of a partly hidden puck. 'no_detection_review' gives the by-eye category of each picture "
        "without a detection."
    )
    output["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    output["detector"] = {
        "file": "puck_detector.py",
        "parameters": PUCK_DETECTOR_PARAMETERS,
        "opencv_version": cv2.__version__,
        "numpy_version": numpy.__version__,
        "opencv_threads": 1
    }
    output["joints_rebuilt_041"] = (
        "Episode 041 has no steps.csv. Frame k (k >= 1) was taken at exact-replay waypoint k - 1 and frame 0 at the "
        "start pose; the waypoints follow record_episodes.py's rule (first guided point, then every 1 cm of tip "
        "travel or 5 degrees of any joint, and the last point). Checked on episode 043, which has a steps.csv: the "
        "logged joints differ from the rebuilt waypoints by 0.44 degrees (median of the largest joint difference "
        "per frame), 1.58 degrees at most. For 041 the rebuilt rod positions match the terminal log within 0.27 cm "
        "(median 0.04 cm). The rebuilt joints are the commanded waypoints, not measured ones."
    )
    output["frames"] = frames
    output["extra_pictures"] = extra_pictures

    with open(os.path.join(OUTPUT_FOLDER, OUTPUT_PARAMETERS["file_name"]), "w") as output_file:
        json.dump(output, output_file, indent = 1)

    detected = sum(record["detection"] is not None for record in frames.values())
    print(f"frames {len(frames)}, detected {detected}, extra pictures {len(extra_pictures)}, "
          f"detected {sum(record['detection'] is not None for record in extra_pictures.values())}")
    print(f"frames without joints or with a md5 / frame mismatch: {len(md5_mismatches)} {md5_mismatches[:5]}")
    print("joints sources:", Counter(record["joints_source"] for record in frames.values()))
    print("categories of frames without a detection:",
          Counter(record["no_detection_review"]["category"] for record in frames.values() if record["detection"] is None))
    per_episode = OrderedDict()
    for record in frames.values():
        episode_counts = per_episode.setdefault(record["episode"], Counter())
        episode_counts["frames"] += 1
        if record["detection"] is not None:
            episode_counts["detected"] += 1
            episode_counts["partial"] += int(record["detection"]["is_partial"])
            episode_counts["circle_centre"] += int(record["detection"]["centre_source"] == "circle_fit")
            episode_counts["bottom_clear"] += int(bool(record["detection"]["is_bottom_clear"]))
        else:
            episode_counts["miss_" + record["no_detection_review"]["category"]] += 1
    for episode, counts in per_episode.items():
        print(episode, dict(counts))
    print(f"detect time in this run: mean {numpy.mean(durations_ms):.2f} ms, max {numpy.max(durations_ms):.2f} ms")
    for key, record in extra_pictures.items():
        print(key.split("raw/")[1], "detected" if record["detection"] else "none",
              record.get("step_row", {}) and record["step_row"].get("step"))


if __name__ == "__main__":
    main()
