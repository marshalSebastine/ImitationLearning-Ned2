"""
Check recorded episodes and tell, within seconds, whether they are good training data.

For a session folder of 03_data/raw (default: the newest one) or a single episode folder, the checker prints per
episode: the mode, camera mount, status and outcome; steps against pictures (missing, extra and repeated pictures by
md5); the time per step; step lengths in the picture's frame and on the table; label counts and whether each label
agrees with the measured motion; the forearm turn and rod lean ranges; the camera shift; the mistake and lost flags;
the discard reason; the start (row 0 against the start pose, the puck in the first picture); and, for hand-guided
episodes, how closely the replay followed the guided path. Then the warnings, each with its threshold
(CHECK_PARAMETERS), and a table per sheet of the episodes kept, discarded and without an outcome.

Read-only: nothing under the data folders is written. With --report the same text is also written as Markdown to
the path given (never into 03_data/raw unless that path is given).

Measured, calculated, assumed. Rows, pictures, md5 sums, times, flags and the robot's reported poses are read from the
files (measured at recording time). Tip positions, rod lean, picture headings and step lengths are calculated from the
logged joints with the arm model (ned2_fk, through side_view_test.magnet_tip_and_lean), as the recording tool itself
does; the model's rod end agrees with the robot's own reported rod end to about 0.02 mm in most rows. "Puck seen" comes
from the red-puck detector (puck_detector.py; on 037 to 045 it missed only slivers beside the block and pucks held in a
hand, and found nothing in the run without the puck), tuned on one desk and one room's lamps: it points at pictures to
look at; on a new table, check it by eye on the first episodes.

Run from 02_code with the project's Python (no robot is needed or contacted):
    python check_episodes.py
    python check_episodes.py D:/AI_Project/03_data/raw/2026-11-05_session_01_emranul --report check.md
    python check_episodes.py D:/AI_Project/03_data/raw/2026-11-05_session_01_emranul/path_04_meander_episode_045
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy


from config import CHECK_PARAMETERS, RECORDING_PARAMETERS
from puck_detector import PuckDetector
from side_view_test import magnet_tip_and_lean, ned2_fk

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

# record_episodes.py is changed often (twice on the night of 5 to 6.11.26 alone); should a work-in-progress version not
# import, the checker still runs and skips only what needs it: the camera-shift recomputation and the picture heading
# of episodes that did not log it.
try:
    from record_episodes import camera_shift_px, forearm_camera_heading_rad
    RECORDING_TOOL_IMPORT_PROBLEM = None
except Exception as import_error:
    camera_shift_px, forearm_camera_heading_rad = None, None
    RECORDING_TOOL_IMPORT_PROBLEM = repr(import_error)

EPISODE_NAME = re.compile(r"(.+)_episode_(\d+)")
SESSION_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})_session_(\d+)_(.+)")
POSE_COLUMNS = ["x", "y", "z", "roll", "pitch", "yaw"]
HAND_GUIDED_MODES = ("hand_guided", "hand_guided_exact")
KEPT_STATUSES = ("completed", "step_budget_reached")


def read_csv_rows(path: Path) -> Optional[List[Dict[str, str]]]:
    """
    Read a CSV file's rows.

    Parameters
    ----------
    path : Path
        The CSV file (with a header row).

    Returns
    -------
    list of dict or None
        One dictionary per row, keyed by the header; None when the file does not exist.
    """

    if not path.exists():
        return None
    with open(path, newline = "", encoding = "utf-8") as file:
        return list(csv.DictReader(file))


def number_or_none(value: Optional[str]) -> Optional[float]:
    """A CSV cell as a float, or None when it is empty or not a number."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def spread_of(values: List[Optional[float]]) -> Optional[Dict[str, float]]:
    """
    Summarise the values that are numbers.

    Parameters
    ----------
    values : list of float or None
        The values; None and NaN are left out.

    Returns
    -------
    dict or None
        "count", "min", "median" and "max"; None when no value is a number.
    """

    numbers = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not numbers:
        return None
    array = numpy.array(numbers)
    return {"count": len(numbers), "min": float(array.min()), "median": float(numpy.median(array)), "max": float(array.max())}


def wrap_degrees(angle_deg: float) -> float:
    """An angle brought into [-180, 180) degrees."""

    return (angle_deg + 180.0) % 360.0 - 180.0


def short_list(values: List, most: int = 8) -> str:
    """The first ``most`` values joined by commas, with the full count when there are more."""

    shown = ", ".join(str(value) for value in values[:most])
    return shown + (f", ... ({len(values)} in all)" if len(values) > most else "")


def format_spread(
        spread: Optional[Dict[str, float]],
        unit: str,
        digits: int = 1
) -> str:
    """A spread as "min a, median b, max c unit"; "n/a" when there is none."""

    if spread is None:
        return "n/a"
    return f"min {spread['min']:.{digits}f}, median {spread['median']:.{digits}f}, max {spread['max']:.{digits}f} {unit}"


def distances_to_polyline(
        points: numpy.ndarray,
        path: numpy.ndarray
) -> numpy.ndarray:
    """
    The distance from every point to the nearest point of a polyline, in the units of the inputs.

    Parameters
    ----------
    points : numpy.ndarray
        Shape (n, 2).
    path : numpy.ndarray
        The polyline's vertices, shape (m, 2).

    Returns
    -------
    numpy.ndarray
        Shape (n,).
    """

    if len(path) == 1:
        return numpy.linalg.norm(points - path[0], axis = 1)
    starts = path[:-1]
    segments = path[1:] - starts
    lengths_squared = numpy.maximum((segments ** 2).sum(axis = 1), 1e-18)
    relative = points[:, None, :] - starts[None, :, :]
    # The foot of the perpendicular on each segment, clipped to the segment's ends.
    fractions = numpy.clip((relative * segments[None, :, :]).sum(axis = 2) / lengths_squared[None, :], 0.0, 1.0)
    nearest = starts[None, :, :] + fractions[:, :, None] * segments[None, :, :]
    return numpy.linalg.norm(nearest - points[:, None, :], axis = 2).min(axis = 1)


def newest_session_folder(raw_folder: Path) -> Optional[Path]:
    """
    Find the newest session folder.

    Parameters
    ----------
    raw_folder : Path
        The folder that holds the session folders (<date>_session_<number>_<operator>).

    Returns
    -------
    Path or None
        The session with the latest date and number (then the latest change); None when there is none, also when
        ``raw_folder`` does not exist (03_data is not in git, so a fresh clone has no recordings).
    """

    if not raw_folder.is_dir():
        return None
    sessions = [folder for folder in raw_folder.iterdir() if folder.is_dir() and SESSION_NAME.fullmatch(folder.name)]
    if not sessions:
        return None

    def order(folder: Path) -> Tuple[str, int, float]:
        match = SESSION_NAME.fullmatch(folder.name)
        return match.group(1), int(match.group(2)), folder.stat().st_mtime

    return max(sessions, key = order)


def is_episode_folder(folder: Path) -> bool:
    """True for a folder that holds one episode (its name, or the files an episode leaves)."""

    return (bool(EPISODE_NAME.fullmatch(folder.name)) or (folder / "episode.json").exists()
            or (folder / "steps.csv").exists() or (folder / "frames").is_dir())



class EpisodeChecker:

    def __init__(self, check_parameters: Optional[dict] = None):
        """
        Prepare a checker with the default thresholds, overridden by ``check_parameters``.

        Parameters
        ----------
        check_parameters : dict, optional
            Overrides merged over CHECK_PARAMETERS; dictionaries are merged key by key.
        """

        self.__initialize_variables()
        self.update_check_parameters(check_parameters)


    def __initialize_variables(self):

        # A deep copy, so that one checker's overrides never change the module's defaults.
        self.check_parameters = json.loads(json.dumps(CHECK_PARAMETERS))
        self.puck_detector = PuckDetector(detector_parameters = self.check_parameters["detector_parameters"])


    def update_check_parameters(self, check_parameters: Optional[dict]):
        """
        Merge overrides into the current thresholds.

        Parameters
        ----------
        check_parameters : dict or None
            Entries to replace; dictionaries (such as "detector_parameters", passed on to the puck detector) are merged
            key by key.
        """

        if not check_parameters:
            return
        for name, value in check_parameters.items():
            if isinstance(value, dict) and isinstance(self.check_parameters.get(name), dict):
                self.check_parameters[name].update(value)
            else:
                self.check_parameters[name] = value
        if check_parameters.get("detector_parameters"):
            self.puck_detector.update_detector_parameters(check_parameters["detector_parameters"])


    def check_folders(self, folders: List[Path]) -> List[Dict]:
        """
        Check every episode in ``folders``: an episode folder is checked itself, a session folder episode by episode.

        Parameters
        ----------
        folders : list of Path
            Session or episode folders.

        Returns
        -------
        list of dict
            One result per episode, in folder order and, within a session, by episode name.
        """

        results = []
        for folder in folders:
            if is_episode_folder(folder):
                episode_folders = [folder]
            else:
                episode_folders = sorted(child for child in folder.iterdir() if child.is_dir() and EPISODE_NAME.fullmatch(child.name))
            for episode_folder in episode_folders:
                try:
                    results.append(self.check_episode(episode_folder))
                except Exception as error:
                    # One unreadable episode must not hide the others; its failure is reported in its place.
                    place = traceback.extract_tb(error.__traceback__)[-1]
                    results.append({"episode": episode_folder.name, "folder": str(episode_folder), "session": episode_folder.parent.name,
                                    "path_id": "unknown", "verdict": "check failed", "lines": [],
                                    "warnings": [("warning", f"the check itself failed: {error!r} (in {place.name}, line {place.lineno})")],
                                    "rows": 0, "frames": 0, "mode": None, "status": None, "outcome": None})
        return results


    def check_episode(self, episode_folder: Path) -> Dict:
        """
        Check one episode folder; nothing in it is written.

        Parameters
        ----------
        episode_folder : Path
            The episode's folder (steps.csv, episode.json, frames/).

        Returns
        -------
        dict
            The findings ("lines" for the report, "warnings" as (level, text) pairs, "verdict") and the numbers behind them.
        """

        started = time.perf_counter()
        result = {"episode": episode_folder.name, "folder": str(episode_folder), "session": episode_folder.parent.name,
                  "lines": [], "warnings": []}
        data = self.__read_episode(episode_folder)
        self.__scan_frames(data)

        self.__describe_identity(result, data)
        self.__check_frames(result, data)
        self.__check_timing(result, data)
        self.__check_motion(result, data)
        self.__check_replay(result, data)
        self.__check_camera(result, data)
        self.__check_start_and_puck(result, data)
        self.__check_flags(result, data)

        result["verdict"] = self.__verdict(result)
        result["check_seconds"] = round(time.perf_counter() - started, 2)
        return result


    def __read_episode(self, episode_folder: Path) -> Dict:
        """Read every file the recording tool leaves in an episode folder (missing ones as None)."""

        summary, summary_problem = None, None
        summary_path = episode_folder / "episode.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding = "utf-8"))
            except (OSError, ValueError) as error:
                summary_problem = f"episode.json cannot be read ({error})"

        rows = read_csv_rows(episode_folder / "steps.csv")
        in_progress_rows = read_csv_rows(episode_folder / "steps_in_progress.csv")
        guided_paths = {path.name: read_csv_rows(path) or [] for path in sorted(episode_folder.glob("*guided_path.csv"))}
        frames_folder = episode_folder / "frames"
        frame_paths = sorted(frames_folder.glob("*.jpg")) if frames_folder.is_dir() else []
        correction_references = sorted(path.name for path in episode_folder.glob("correction_*_reference.jpg"))
        known_names = ({"episode.json", "steps.csv", "steps_in_progress.csv", "camera_check.jpg", "frames"}
                       | set(guided_paths) | set(correction_references))
        other_files = sorted(child.name for child in episode_folder.iterdir() if child.name not in known_names)

        # The checks run on steps.csv; without it, on the log the tool writes at every decision (since 5.11.26).
        used_rows = rows if rows is not None else (in_progress_rows or [])
        return {"folder": episode_folder, "summary": summary, "summary_problem": summary_problem, "rows": rows,
                "in_progress_rows": in_progress_rows, "used_rows": used_rows, "guided_paths": guided_paths,
                "frame_paths": frame_paths, "correction_references": correction_references, "other_files": other_files}


    def __scan_frames(self, data: Dict) -> None:
        """One pass over the pictures: the md5 of each file and, when pictures are checked, the puck detector."""

        settings = self.check_parameters
        md5_by_name, detections, unreadable, picture_size = {}, {}, [], None
        for index, path in enumerate(data["frame_paths"]):
            picture_bytes = path.read_bytes()
            md5_by_name[path.name] = hashlib.md5(picture_bytes).hexdigest()
            if not settings["is_picture_check_on"]:
                continue
            picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
            if picture is None:
                unreadable.append(path.name)
                continue
            if index == 0:
                picture_size = [int(picture.shape[1]), int(picture.shape[0])]
            detection = self.puck_detector.detect(picture)
            if detection is not None:
                height, width = picture.shape[:2]
                detection = {"centre": [round(detection["centre_u"] / width, 3), round(detection["centre_v"] / height, 3)],
                             "confidence": detection["confidence"], "is_partial": detection["is_partial"]}
            detections[path.name] = detection
            if index == 0:
                top, bottom, left, right = settings["puck_start_region"]
                is_in_front = (detection is not None and top <= detection["centre"][1] <= bottom and left <= detection["centre"][0] <= right)
                data["puck_at_start"] = detection if is_in_front else None
        data.update({"frame_md5": md5_by_name, "puck_detections": detections, "unreadable_frames": unreadable,
                     "picture_size": picture_size})


    def __warn(
            self,
            result: Dict,
            level: str,
            text: str
    ) -> None:
        """Add a finding: "warning" (look before training on the episode) or "note" (for information)."""

        result["warnings"].append((level, text))


    def __describe_identity(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """Mode, mount, mapping, status, outcome and the files present; warnings for missing files and other mounts."""

        settings = self.check_parameters
        summary = data["summary"] or {}
        match = EPISODE_NAME.fullmatch(result["episode"])
        result.update({
            "path_id": summary.get("path_id") or (match.group(1) if match else "unknown"),
            "episode_number": int(match.group(2)) if match else None,
            "mode": summary.get("mode"),
            "status": summary.get("status"),
            "outcome": summary.get("outcome"),
            "discard_reason": summary.get("discard_reason"),
            "is_simulated": bool(summary.get("is_simulated", False)),
            "camera_mount_written": summary.get("camera_mount"),
            "mapping_version": summary.get("mapping_version"),
            "start_pose": summary.get("start_pose"),
            "rows": len(data["used_rows"]),
            "frames": len(data["frame_paths"])
        })
        result["camera_mount"] = self.__corrected_mount(result["camera_mount_written"], result["path_id"], result["episode_number"])

        mount_text = result["camera_mount"] or "unknown (no episode.json)"
        if result["camera_mount"] and result["camera_mount"] != result["camera_mount_written"]:
            mount_text += f" (written as {result['camera_mount_written']}, corrected in CHECK_PARAMETERS)"
        mount_text += " (current)" if result["camera_mount"] == settings["current_camera_mount"] else ""
        mapping_text = result["mapping_version"] or "unknown"
        mapping_text += " (current)" if result["mapping_version"] == settings["current_mapping_version"] else ""
        result["lines"].append(("mode, operator", f"{result['mode'] or 'unknown'}, {summary.get('operator', 'unknown')}; started "
                                                  f"{summary.get('started', '?')}, finished {summary.get('finished', '?')}, code {summary.get('code_commit', '?')}"))
        result["lines"].append(("camera mount", f"{mount_text}; mapping {mapping_text}"))
        ending = result["outcome"] or "no outcome"
        result["lines"].append(("status, outcome", f"{result['status'] or 'unknown'}, {ending}; discard reason: {result['discard_reason'] or 'none'}"
                                                   + (f"; hand corrections: {summary['hand_corrections']}" if summary.get("hand_corrections") else "")))
        if data["other_files"]:
            described = []
            for name in data["other_files"]:
                other_rows = read_csv_rows(data["folder"] / name) if name.endswith(".csv") else None
                described.append(f"{name} ({len(other_rows)} rows)" if other_rows is not None else name)
            result["lines"].append(("other files", ", ".join(described)))

        if data["summary"] is None:
            self.__warn(result, "warning", data["summary_problem"] or "no episode.json: the episode did not finish (the tool writes it last), "
                                                                     "so its status and outcome are unknown")
        if data["rows"] is None:
            if data["in_progress_rows"] is not None:
                self.__warn(result, "warning", f"no steps.csv; the {len(data['in_progress_rows'])} rows of steps_in_progress.csv were checked instead "
                                               "(the tool stopped before finishing the episode)")
            else:
                self.__warn(result, "warning", f"no steps.csv: {len(data['frame_paths'])} pictures have no labels")
        elif data["in_progress_rows"] is not None:
            self.__warn(result, "warning", "steps_in_progress.csv is still there next to steps.csv (the tool deletes it when the episode is written)")
        if result["is_simulated"]:
            self.__warn(result, "warning", "simulated episode: not robot data")
        if result["camera_mount"] and result["camera_mount"] != settings["current_camera_mount"]:
            self.__warn(result, "warning", f"camera mount {result['camera_mount']} is not the current {settings['current_camera_mount']}: "
                                           "never mix two mounts in one training set")
        if result["mapping_version"] and result["mapping_version"] != settings["current_mapping_version"]:
            self.__warn(result, "warning", f"mapping {result['mapping_version']} is not the current {settings['current_mapping_version']}: "
                                           "the labels may mean other directions")
        if summary and result["status"] in KEPT_STATUSES and result["outcome"] not in ("puck_in_goal", "puck_not_in_goal"):
            self.__warn(result, "note", f"outcome {result['outcome'] or 'missing'}: whether the puck reached the goal is not recorded")
        if result["status"] == "quit":
            self.__warn(result, "note", "ended with Esc (quit): no outcome was asked")


    def __corrected_mount(
            self,
            written_mount: Optional[str],
            path_id: str,
            episode_number: Optional[int]
    ) -> Optional[str]:
        """The camera mount an episode really had, after the corrections listed in "camera_mount_corrections"."""

        for correction in self.check_parameters["camera_mount_corrections"]:
            is_in_range = episode_number is not None and correction["first_episode"] <= episode_number <= correction["last_episode"]
            if written_mount == correction["written"] and path_id == correction["path_id"] and is_in_range:
                return correction["actual"]
        return written_mount


    def __check_frames(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """Rows against pictures: missing, extra, repeated (md5) and altered pictures, numbering and picture size."""

        settings = self.check_parameters
        rows = data["used_rows"]
        present = {f"frames/{path.name}": path.name for path in data["frame_paths"]}
        expected = {(row.get("frame") or "").replace("\\", "/"): row for row in rows}

        missing_steps = [row.get("step") for name, row in expected.items() if name not in present]
        extra_frames = sorted(name.split("/")[-1] for name in present if name not in expected)
        altered_steps = [row.get("step") for name, row in expected.items()
                         if name in present and row.get("frame_md5") and row["frame_md5"] != data["frame_md5"][present[name]]]
        names_by_md5 = {}
        for name, digest in data["frame_md5"].items():
            names_by_md5.setdefault(digest, []).append(name)
        repeated_groups = [sorted(names) for names in names_by_md5.values() if len(names) > 1]
        ordered_names = sorted(data["frame_md5"])
        repeated_in_a_row = [f"{first}={second}" for first, second in zip(ordered_names, ordered_names[1:])
                             if data["frame_md5"][first] == data["frame_md5"][second]]
        misnumbered_steps = [row.get("step") for index, row in enumerate(rows)
                             if row.get("step") != str(index) or (row.get("frame") or "").replace("\\", "/") != f"frames/{index:03d}.jpg"]

        steps_in_summary = (data["summary"] or {}).get("steps")
        rows_text = f"{len(rows)} rows" + (f" (episode.json says {steps_in_summary})" if steps_in_summary is not None and steps_in_summary != len(rows) else "")
        size_text = f"; picture {data['picture_size'][0]} x {data['picture_size'][1]}" if data.get("picture_size") else ""
        result["lines"].append(("steps vs pictures", f"{rows_text}, {len(present)} pictures; missing {len(missing_steps)}, extra {len(extra_frames)}, "
                                                     f"repeated (same md5) {sum(len(group) - 1 for group in repeated_groups)}, md5 not as logged {len(altered_steps)}{size_text}"))
        result.update({"missing_frames": missing_steps, "extra_frames": extra_frames, "repeated_frames": repeated_groups,
                       "altered_frames": altered_steps})

        if len(missing_steps) > settings["missing_frames_allowed"]:
            self.__warn(result, "warning", f"{len(missing_steps)} rows have no picture (steps {short_list(missing_steps)}); allowed {settings['missing_frames_allowed']}")
        if len(extra_frames) > settings["extra_frames_allowed"] and rows:
            self.__warn(result, "warning", f"{len(extra_frames)} pictures have no row ({short_list(extra_frames)}); allowed {settings['extra_frames_allowed']}")
        repeated_count = sum(len(group) - 1 for group in repeated_groups)
        if repeated_count > settings["duplicate_frames_allowed"]:
            self.__warn(result, "warning", f"{repeated_count} pictures repeat another picture byte for byte ({short_list(['='.join(group) for group in repeated_groups], 4)}; "
                                           f"{len(repeated_in_a_row)} of them in a row): the camera sent nothing new; allowed {settings['duplicate_frames_allowed']}")
        if altered_steps:
            self.__warn(result, "warning", f"{len(altered_steps)} pictures differ from the md5 logged with them (steps {short_list(altered_steps)}): changed after recording")
        if misnumbered_steps:
            self.__warn(result, "warning", f"{len(misnumbered_steps)} rows are not numbered 0, 1, 2, ... with frames/NNN.jpg (steps {short_list(misnumbered_steps)})")
        if data["unreadable_frames"]:
            self.__warn(result, "warning", f"{len(data['unreadable_frames'])} pictures cannot be decoded ({short_list(data['unreadable_frames'])})")
        if data.get("picture_size") and data["picture_size"] != list(settings["picture_size"]):
            self.__warn(result, "warning", f"pictures are {data['picture_size'][0]} x {data['picture_size'][1]}, not {settings['picture_size'][0]} x {settings['picture_size'][1]}")


    def __check_timing(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """Time per step (picture to picture), the wait for a fresh picture after each move, and long pauses."""

        settings = self.check_parameters
        rows = data["used_rows"]
        if not rows:
            result["lines"].append(("time per step", "n/a (no rows)"))
            return

        capture = [number_or_none(row.get("capture_time")) for row in rows]
        if all(value is None for value in capture):
            capture = [number_or_none(row.get("unix_time")) for row in rows]
        move_return = [number_or_none(row.get("move_return_time")) for row in rows]
        decision = [number_or_none(row.get("decision_time")) for row in rows]

        step_times = [later - earlier for earlier, later in zip(capture, capture[1:]) if earlier is not None and later is not None]
        picture_waits = [taken - returned for taken, returned in zip(capture, move_return) if taken is not None and returned is not None]
        ending = decision[-1] if decision[-1] is not None else capture[-1]
        duration_s = ending - capture[0] if ending is not None and capture[0] is not None else None
        step_spread = spread_of(step_times)
        wait_spread = spread_of(picture_waits)
        result.update({"step_time_s": step_spread, "picture_wait_s": wait_spread, "duration_s": duration_s})

        text = format_spread(step_spread, "s", 2) + " picture to picture"
        if duration_s is not None:
            text += f"; {duration_s / 60.0:.1f} min from the first picture to the last decision"
        if wait_spread is not None:
            # Only the least and the median: after a refused step the tool keeps the earlier move's return time, so
            # the longest "waits" include the person's thinking time.
            text += f"; picture taken at least {wait_spread['min']:.2f} s (median {wait_spread['median']:.2f} s) after the move returned"
        result["lines"].append(("time per step", text))

        short_waits = [rows[index].get("step") for index, (taken, returned) in enumerate(zip(capture, move_return))
                       if taken is not None and returned is not None and taken - returned < settings["picture_wait_note_s"]]
        if short_waits:
            self.__warn(result, "note", f"{len(short_waits)} pictures were taken less than {settings['picture_wait_note_s']} s after the move returned "
                                        f"(steps {short_list(short_waits)}): unusually quick; look whether they show motion")
        pauses = [f"{rows[index].get('step')}->{rows[index + 1].get('step')} ({later - earlier:.0f} s)"
                  for index, (earlier, later) in enumerate(zip(capture, capture[1:]))
                  if earlier is not None and later is not None and later - earlier > settings["long_pause_s"]]
        if decision[-1] is not None and capture[-1] is not None and decision[-1] - capture[-1] > settings["long_pause_s"]:
            pauses.append(f"{rows[-1].get('step')} decided {decision[-1] - capture[-1]:.0f} s after its picture")
        if pauses:
            self.__warn(result, "note", f"{len(pauses)} pauses over {settings['long_pause_s']:.0f} s ({short_list(pauses, 5)}); the picture of such a step "
                                        "may no longer show the scene when the decision was made")


    def __check_motion(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """
        Tip positions, step lengths, labels against the measured motion, forearm turn and rod lean, from the joints.

        The magnet tip of every row comes from the arm model on the logged joints (calc), as in the recording tool;
        the step of row k is the tip's move from row k to row k + 1, measured in the picture's frame of row k.
        """

        settings = self.check_parameters
        rows = data["used_rows"]
        summary = data["summary"] or {}
        data["tips"] = None
        if not rows:
            return

        joints = [[number_or_none(row.get(f"joint_{index}")) for index in range(1, 7)] for row in rows]
        poses = [[number_or_none(row.get(name)) for name in POSE_COLUMNS] for row in rows]
        has_joints = all(value is not None for values in joints for value in values)
        has_poses = all(value is not None for values in poses for value in values[:3])
        if has_joints and not result["is_simulated"]:
            tips_and_leans = [magnet_tip_and_lean(values) for values in joints]
            tips = numpy.array([tip[:2] for tip, _ in tips_and_leans])
            leans = [float(lean) for _, lean in tips_and_leans]
            tip_source = "magnet tip, arm model on the logged joints, calc"
        elif has_poses:
            # The simulated robot's joints are only approximate; its reported rod end is the tip it tracks.
            tips = numpy.array([values[:2] for values in poses])
            leans = []
            tip_source = "reported rod end (no usable joints)"
        else:
            result["lines"].append(("tip positions", "n/a (no joints or poses in the rows)"))
            return
        data["tips"] = tips

        if has_joints and has_poses:
            mismatch_mm = [1000.0 * float(numpy.linalg.norm(numpy.array(ned2_fk.fk(values)["tool"][0]) - numpy.array(pose[:3])))
                           for values, pose in zip(joints, poses)]
            mismatch_spread = spread_of(mismatch_mm)
            far_rows = [rows[index].get("step") for index, value in enumerate(mismatch_mm) if value > settings["model_mismatch_mm"]]
            result["lines"].append(("arm model vs robot", f"rod end, model on the joints against the robot's reported pose: {format_spread(mismatch_spread, 'mm', 2)}; "
                                                          f"{len(far_rows)} rows over {settings['model_mismatch_mm']} mm"))
            if len(far_rows) > settings["model_mismatch_note_fraction"] * len(rows):
                self.__warn(result, "note", f"{len(far_rows)} of {len(rows)} rows have joints and pose more than {settings['model_mismatch_mm']} mm apart "
                                            f"(steps {short_list(far_rows)}); a guess: read while the arm was still settling")

        # The picture's heading at every row: logged since 4.11.26 (v12); otherwise calculated for the forearm camera.
        headings_deg = [number_or_none(row.get("picture_heading_deg")) for row in rows]
        heading_source = "logged"
        if any(value is None for value in headings_deg):
            is_forearm_view = has_joints and bool(result["camera_mount"]) and "forearm" in result["camera_mount"]
            if is_forearm_view and forearm_camera_heading_rad is not None:
                headings_deg = [math.degrees(forearm_camera_heading_rad(values)) for values in joints]
                heading_source = "calculated with today's forearm-camera model"
            elif is_forearm_view:
                headings_deg = None
                heading_source = f"not calculated (record_episodes could not be imported: {RECORDING_TOOL_IMPORT_PROBLEM})"
            else:
                headings_deg = None
                heading_source = "unknown for this mount"

        # Table steps of the tip between consecutive pictures (the last row has no step after it).
        displacements = numpy.diff(tips, axis = 0)
        table_steps_mm = [1000.0 * float(numpy.linalg.norm(move)) for move in displacements]
        actions = [row.get("action", "") for row in rows]
        moving_steps_mm = [length for index, length in enumerate(table_steps_mm) if actions[index] != "stop"]
        table_spread = spread_of(moving_steps_mm)
        result["table_step_mm"] = table_spread
        result["lines"].append(("step on the table", f"{format_spread(table_spread, 'mm')} over {len(moving_steps_mm)} steps ({tip_source})"))

        step_over_actions = summary.get("step_over_actions", RECORDING_PARAMETERS["step_over_actions"])
        step_over_mm = 1000.0 * float(summary.get("step_over_m", RECORDING_PARAMETERS["step_over_m"]))
        jumps = [f"{rows[index].get('step')} {actions[index]} {length:.0f} mm" for index, length in enumerate(table_steps_mm)
                 if length > (step_over_mm + settings["step_over_jump_allowance_mm"] if actions[index] in step_over_actions
                              else settings["step_jump_warning_mm"])]
        if jumps:
            self.__warn(result, "warning", f"{len(jumps)} steps move the tip more than {settings['step_jump_warning_mm']:.0f} mm between two pictures ({short_list(jumps, 5)})")

        # The logged motion in the picture's frame (exact replay and gamepad rows store it beside the label).
        logged_forward = [number_or_none(row.get("tip_forward_m")) for row in rows]
        logged_left = [number_or_none(row.get("tip_left_m")) for row in rows]
        logged_mm = [1000.0 * math.hypot(forward, left) for forward, left in zip(logged_forward, logged_left) if forward is not None and left is not None]
        picture_text = f"{format_spread(spread_of(logged_mm), 'mm')} logged with the label" if logged_mm else "not logged in this mode"
        result["picture_step_mm"] = spread_of(logged_mm)

        left_sign = summary.get("picture_left_sign", RECORDING_PARAMETERS["picture_left_sign"])
        action_angles = summary.get("actions", RECORDING_PARAMETERS["actions"])
        classes = {name: float(angle) for name, angle in action_angles.items() if angle is not None}
        turn_actions = summary.get("turn_actions", RECORDING_PARAMETERS["turn_actions"])

        if headings_deg is not None:
            differences_mm = []
            agreement = {"moving_checked": 0, "moving_agreeing": 0, "small_checked": 0, "small_agreeing": 0, "angle_errors": []}
            disagreements = []
            for index, move in enumerate(displacements):
                heading = math.radians(headings_deg[index])
                forward_m = float(move @ numpy.array([math.cos(heading), math.sin(heading)]))
                left_m = float(move @ numpy.array([math.cos(heading + left_sign * math.pi / 2), math.sin(heading + left_sign * math.pi / 2)]))
                length_m = math.hypot(forward_m, left_m)
                if logged_forward[index] is not None and logged_left[index] is not None:
                    differences_mm.append(1000.0 * math.hypot(forward_m - logged_forward[index], left_m - logged_left[index]))
                label = actions[index]
                if label in classes:
                    if length_m < settings["small_motion_m"] - settings["small_motion_tolerance_m"]:
                        agreement["moving_checked"] += 1
                        disagreements.append(f"{rows[index].get('step')} {label} moved {1000 * length_m:.1f} mm")
                        continue
                    if length_m < settings["small_motion_m"]:
                        continue
                    angle_deg = math.degrees(math.atan2(left_m, forward_m))
                    error_to_label = abs(wrap_degrees(angle_deg - classes[label]))
                    nearest = min(classes, key = lambda name: abs(wrap_degrees(angle_deg - classes[name])))
                    error_to_nearest = abs(wrap_degrees(angle_deg - classes[nearest]))
                    agreement["moving_checked"] += 1
                    agreement["angle_errors"].append(error_to_label)
                    if error_to_label <= error_to_nearest + settings["label_boundary_tolerance_deg"]:
                        agreement["moving_agreeing"] += 1
                    else:
                        disagreements.append(f"{rows[index].get('step')} {label} measured {nearest} ({angle_deg:+.0f} deg)")
                elif label == "small_motion":
                    agreement["small_checked"] += 1
                    if length_m < settings["small_motion_m"] + settings["small_motion_tolerance_m"]:
                        agreement["small_agreeing"] += 1
                    else:
                        disagreements.append(f"{rows[index].get('step')} small_motion moved {1000 * length_m:.1f} mm")
            if differences_mm:
                picture_text += f"; logged against measured: {format_spread(spread_of(differences_mm), 'mm')}"
            result["lines"].append(("step in the picture", picture_text))

            checked = agreement["moving_checked"] + agreement["small_checked"]
            error_spread = spread_of(agreement["angle_errors"])
            label_text = (f"{agreement['moving_agreeing']} of {agreement['moving_checked']} moving labels and {agreement['small_agreeing']} of "
                          f"{agreement['small_checked']} small_motion labels agree with the measured motion (heading {heading_source})")
            if error_spread is not None:
                label_text += f"; direction off its label: median {error_spread['median']:.1f}, max {error_spread['max']:.1f} deg"
            result["lines"].append(("labels vs motion", label_text))
            result["label_disagreements"] = disagreements
            if checked and len(disagreements) > settings["label_disagreement_warning_fraction"] * checked:
                self.__warn(result, "warning", f"{len(disagreements)} of {checked} labels disagree with the measured tip motion (more than "
                                               f"{100 * settings['label_disagreement_warning_fraction']:.0f} %): {short_list(disagreements, 5)}")
            elif disagreements:
                self.__warn(result, "note", f"{len(disagreements)} of {checked} labels disagree with the measured tip motion: {short_list(disagreements, 5)}")
        else:
            result["lines"].append(("step in the picture", picture_text))
            result["lines"].append(("labels vs motion", f"not checked: picture heading {heading_source}"))

        # Turn labels against joint 4 (turn_left = joint 4 grows), the forearm's range and the rod's lean.
        if has_joints:
            forearm_deg = [math.degrees(values[3]) for values in joints]
            wrong_turns = []
            turn_count = 0
            for index in range(len(rows) - 1):
                if actions[index] not in turn_actions:
                    continue
                turn_count += 1
                change_deg = forearm_deg[index + 1] - forearm_deg[index]
                expected_sign = 1.0 if turn_actions[actions[index]] > 0 else -1.0
                if change_deg * expected_sign < settings["turn_minimum_deg"]:
                    wrong_turns.append(f"{rows[index].get('step')} {actions[index]} joint 4 {change_deg:+.1f} deg")
            turn_text = f"{turn_count - len(wrong_turns)} of {turn_count} turn labels match joint 4" if turn_count else "no turn labels"
            result["lines"].append(("forearm (joint 4)", f"{min(forearm_deg):+.1f} to {max(forearm_deg):+.1f} deg; {turn_text}"))
            result["forearm_range_deg"] = [min(forearm_deg), max(forearm_deg)]
            if wrong_turns:
                self.__warn(result, "warning", f"{len(wrong_turns)} turn labels do not match the forearm's turn: {short_list(wrong_turns, 5)}")
        if leans:
            lean_spread = spread_of(leans)
            over_note = [rows[index].get("step") for index, lean in enumerate(leans) if lean > settings["lean_note_deg"]]
            over_warning = [rows[index].get("step") for index, lean in enumerate(leans) if lean > settings["lean_warning_deg"]]
            result["lines"].append(("rod lean (calc)", f"{format_spread(lean_spread, 'deg')}; {len(over_note)} rows over {settings['lean_note_deg']} deg, "
                                                       f"{len(over_warning)} over {settings['lean_warning_deg']} deg"))
            result["lean_deg"] = lean_spread
            if over_warning:
                self.__warn(result, "warning", f"the rod leans more than {settings['lean_warning_deg']} deg in {len(over_warning)} rows (max {lean_spread['max']:.1f} deg; "
                                               f"steps {short_list(over_warning)}): more than ever reached with the puck")
            elif over_note:
                self.__warn(result, "note", f"the rod leans more than {settings['lean_note_deg']} deg (the keyboard tool's limit) in {len(over_note)} rows "
                                            f"(max {lean_spread['max']:.1f} deg; steps {short_list(over_note)})")

        label_counts = Counter(actions)
        source_counts = Counter(row.get("source", "") for row in rows)
        result["label_counts"] = dict(label_counts)
        result["lines"].append(("labels", ", ".join(f"{name} {count}" for name, count in label_counts.most_common())
                                + "; sources: " + ", ".join(f"{name} {count}" for name, count in source_counts.most_common())))
        small_count = label_counts.get("small_motion", 0)
        if small_count > settings["small_motion_note_fraction"] * len(rows):
            self.__warn(result, "note", f"{small_count} of {len(rows)} rows are small_motion ({100 * small_count / len(rows):.0f} %): "
                                        "not one of the moving classes; decide how training treats them")


    def __check_replay(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """
        Replay fidelity of the hand-guided modes: each replayed tip's distance to the path the hand drew.

        In the exact replay a correction by hand ("h") restarts the waypoint count at 0 and replays its own path
        (correction_NN_guided_path.csv); each such segment is compared with the correction path whose first point
        lies nearest its first tip.
        """

        settings = self.check_parameters
        rows = data["used_rows"]
        if result["mode"] not in HAND_GUIDED_MODES:
            return
        main_path_rows = data["guided_paths"].get("guided_path.csv")
        if not main_path_rows or data.get("tips") is None or not rows:
            result["lines"].append(("replay vs guided path", "n/a (no guided path or no replayed rows)"))
            return

        def path_points(path_rows: List[Dict[str, str]]) -> numpy.ndarray:
            return numpy.array([[float(row["x"]), float(row["y"])] for row in path_rows if row.get("x") not in (None, "")])

        tips = data["tips"]
        segment_starts = [0]
        if result["mode"] == "hand_guided_exact":
            waypoints = [number_or_none(row.get("waypoint")) for row in rows]
            for index in range(1, len(rows)):
                if waypoints[index] is not None and waypoints[index - 1] is not None and waypoints[index] < waypoints[index - 1]:
                    segment_starts.append(index)
        segment_ends = segment_starts[1:] + [len(rows)]

        corrections = sorted(name for name in data["guided_paths"] if name.startswith("correction_"))
        unused_corrections = list(corrections)
        texts, all_distances, warnings = [], [], []
        last_path = None
        for segment_index, (first, end) in enumerate(zip(segment_starts, segment_ends)):
            if segment_index == 0:
                name = "guided_path.csv"
            else:
                candidates = [(float(numpy.linalg.norm(path_points(data["guided_paths"][candidate])[0] - tips[first])), candidate)
                              for candidate in unused_corrections if len(data["guided_paths"][candidate]) >= 2]
                candidates.sort()
                if not candidates or 1000.0 * candidates[0][0] > settings["correction_match_mm"]:
                    warnings.append(f"rows {first} to {end - 1} restart the waypoints but no correction path starts within {settings['correction_match_mm']} mm of them")
                    continue
                name = candidates[0][1]
                unused_corrections = [candidate for candidate in unused_corrections if candidate != name]
            path = path_points(data["guided_paths"][name])
            if len(path) == 0:
                warnings.append(f"{name} has no points")
                continue
            last_path = path
            distances_mm = 1000.0 * distances_to_polyline(tips[first:end], path)
            all_distances.extend(distances_mm.tolist())
            path_length_cm = 100.0 * float(numpy.linalg.norm(numpy.diff(path, axis = 0), axis = 1).sum())
            texts.append(f"{name} ({len(path)} points, {path_length_cm:.1f} cm) for rows {first} to {end - 1}: "
                         f"median {numpy.median(distances_mm):.1f}, max {distances_mm.max():.1f} mm")
            if numpy.median(distances_mm) > settings["replay_median_warning_mm"] or distances_mm.max() > settings["replay_max_warning_mm"]:
                worst = int(numpy.argmax(distances_mm))
                warnings.append(f"replay of {name} strays from it: median {numpy.median(distances_mm):.1f} mm (limit {settings['replay_median_warning_mm']}), "
                                f"max {distances_mm.max():.1f} mm at step {rows[first + worst].get('step')} (limit {settings['replay_max_warning_mm']})")

        end_text = "n/a"
        if last_path is not None:
            end_distance_mm = 1000.0 * float(numpy.linalg.norm(tips[-1] - last_path[-1]))
            is_stop = rows[-1].get("action") == "stop"
            end_text = f"last row {end_distance_mm:.1f} mm from the path's end ({'a stop' if is_stop else 'not a stop: logged before its last move'})"
            result["replay_end_mm"] = end_distance_mm
            if end_distance_mm > settings["replay_end_warning_mm"]:
                warnings.append(f"the replay ended {end_distance_mm:.0f} mm from the guided path's end (limit {settings['replay_end_warning_mm']} mm)")
        if corrections:
            discarded = [name for name in unused_corrections]
            if discarded:
                texts.append(f"not replayed (discarded corrections?): {', '.join(discarded)}")
        overall = spread_of(all_distances)
        result["replay_distance_mm"] = overall
        result["lines"].append(("replay vs guided path", "; ".join(texts + [end_text])))
        for text in warnings:
            self.__warn(result, "warning", text)
        if overall is None:
            self.__warn(result, "warning", "no replayed row could be compared with a guided path")


    def __check_camera(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """
        The camera shift: the value in episode.json, and the same measurement redone on camera_check.jpg and on
        frame 000 against the reference picture for the mount and start (both are taken at the start pose).
        """

        settings = self.check_parameters
        summary = data["summary"] or {}
        limit = settings["camera_shift_warning_px"]
        logged_shift = summary.get("camera_shift_px")
        texts = [f"{logged_shift} px in episode.json (reference {summary.get('camera_reference', 'the episode itself')})"
                 if logged_shift is not None else "not in episode.json (checked by the tool since 5.11.26)"]

        reference_path = self.__camera_reference(result, data)
        recomputed = {}
        if camera_shift_px is None:
            texts.append(f"not recomputed (record_episodes could not be imported: {RECORDING_TOOL_IMPORT_PROBLEM})")
        elif reference_path is not None and settings["is_picture_check_on"]:
            reference_bytes = reference_path.read_bytes()
            for label, path in (("camera_check.jpg", data["folder"] / "camera_check.jpg"),
                                ("frame 000", data["frame_paths"][0] if data["frame_paths"] else None)):
                if path is None or not path.exists():
                    continue
                shift_x, shift_y = camera_shift_px(reference_bytes, path.read_bytes(), settings["camera_check_region"])
                recomputed[label] = math.hypot(shift_x, shift_y)
            if recomputed:
                texts.append("recomputed against " + reference_path.name + ": "
                             + ", ".join(f"{label} {value:.2f} px" for label, value in recomputed.items()))
        elif reference_path is None:
            texts.append("no reference picture for this mount and start")
        result["camera_shift_px"] = {"logged": logged_shift, **recomputed}
        result["lines"].append(("camera shift", "; ".join(texts)))

        if logged_shift is not None and logged_shift > limit:
            self.__warn(result, "warning", f"camera shift {logged_shift} px at the start (episode.json), above {limit} px: check the mount")
        if recomputed.get("frame 000", 0.0) > limit:
            self.__warn(result, "warning", f"frame 000 is {recomputed['frame 000']:.1f} px off the reference picture (limit {limit} px): "
                                           "the camera may have moved (in the hand-guided modes, during the guiding)")


    def __camera_reference(
            self,
            result: Dict,
            data: Dict
    ) -> Optional[Path]:
        """
        The reference picture for the episode's camera mount and start, as record_episodes names it, or None.

        The name holds the puck's start point, which episode.json does not store; it is taken as the start pose plus
        "puck_centre_ahead_of_tip_m" along "start_push_direction_deg" of today's config (an assumption: 0 degrees on
        every sheet so far).
        """

        summary = data["summary"] or {}
        reference_folder = PROJECT_FOLDER / self.check_parameters["raw_folder"] / "camera_check" / "camera_reference"
        if summary.get("camera_reference"):
            named = reference_folder / summary["camera_reference"]
            return named if named.exists() else None
        start_pose = summary.get("start_pose")
        if not start_pose:
            return None
        push = math.radians(RECORDING_PARAMETERS["start_push_direction_deg"])
        ahead = RECORDING_PARAMETERS["puck_centre_ahead_of_tip_m"]
        start_x, start_y = start_pose[0] + ahead * math.cos(push), start_pose[1] + ahead * math.sin(push)
        for mount in (result.get("camera_mount_written"), result.get("camera_mount")):
            if not mount:
                continue
            candidate = reference_folder / f"{mount}_x{1000 * start_x:.0f}_y{1000 * start_y:.0f}mm.jpg"
            if candidate.exists():
                return candidate
        return None


    def __check_start_and_puck(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """
        The start (row 0's rod end against episode.json's start pose) and the puck by the detector: in frame 000 in
        front of the block, the first picture it is seen in, how often it is seen, and in the last picture.
        """

        settings = self.check_parameters
        rows = data["used_rows"]
        start_pose = result.get("start_pose")
        if rows and start_pose:
            first_x, first_y = number_or_none(rows[0].get("x")), number_or_none(rows[0].get("y"))
            if first_x is not None and first_y is not None:
                offset_mm = 1000.0 * math.hypot(first_x - start_pose[0], first_y - start_pose[1])
                forearm_deg = math.degrees(number_or_none(rows[0].get("joint_4")) or 0.0)
                result["start_offset_mm"] = offset_mm
                result["lines"].append(("start", f"start pose (rod end) x {start_pose[0]:.4f}, y {start_pose[1]:.4f} m; row 0 is {offset_mm:.1f} mm from it, "
                                                 f"forearm {forearm_deg:+.1f} deg"))
                if offset_mm > settings["start_offset_warning_mm"]:
                    self.__warn(result, "warning", f"row 0 is {offset_mm:.1f} mm from the start pose in episode.json (limit {settings['start_offset_warning_mm']} mm)")
        elif rows:
            result["lines"].append(("start", "no start pose (no episode.json)"))

        if not settings["is_picture_check_on"] or not data["frame_paths"]:
            return
        names = [path.name for path in data["frame_paths"]]
        detections = [data["puck_detections"].get(name) for name in names]
        seen = [detection is not None for detection in detections]
        first_seen = next((index for index, is_seen in enumerate(seen) if is_seen), None)
        at_start = data.get("puck_at_start")
        # The last picture counts as showing the puck only with a confident sighting (review of 6.11.26: a red object in
        # the far background of episode 035's last picture scored 0.50 and hid the warning below).
        last_detection = detections[-1]
        is_seen_at_end = last_detection is not None and last_detection["confidence"] >= settings["puck_last_picture_minimum_confidence"]
        result["puck"] = {"at_start": at_start is not None, "first_seen": first_seen, "seen": sum(seen), "pictures": len(seen),
                          "in_last_picture": is_seen_at_end}
        start_text = (f"in front of the block in frame 000 (detector score {at_start['confidence']:.2f})" if at_start is not None
                      else "NOT found in front of the block in frame 000")
        if is_seen_at_end:
            last_text = "seen"
        elif last_detection is not None:
            last_text = f"only an unsure sighting (score {last_detection['confidence']:.2f})"
        else:
            last_text = "not seen"
        result["lines"].append(("puck (detector)", f"{start_text}; first seen in frame {first_seen if first_seen is not None else '-'}; "
                                                           f"seen in {sum(seen)} of {len(seen)} pictures ({100 * sum(seen) / len(seen):.0f} %); "
                                                           f"last picture: {last_text}"))

        # The start region fits the forearm camera's view (the block at the bottom middle), and the detector was tuned on
        # it; for other mounts and for episodes without episode.json its findings are shown but not turned into
        # warnings.
        is_view_known = bool(result["camera_mount"]) and "forearm" in result["camera_mount"]
        if not is_view_known or not rows:
            return
        if at_start is None:
            # The detector also misses a puck held in a hand (review of 6.11.26: episode 038's frame 000), so the
            # warning names what to look for instead of stating that the puck was missing.
            if first_seen is not None and first_seen > 1:
                missing_text = (f"it is first found in frame {first_seen:03d}: if frames 000 to {first_seen - 1:03d} really show no puck, "
                                f"those steps teach moves without one")
            elif first_seen == 1:
                missing_text = "it is first found in frame 001: if frame 000 really shows no puck, step 000 teaches a move without one"
            else:
                missing_text = "it is not found in any picture"
            self.__warn(result, "warning", f"the puck detector finds no puck in front of the block in frame 000; {missing_text}. Look at "
                                           f"frame 000: a start picture needs the puck in front of the block and no hands (a puck held "
                                           f"in a hand is not found either)")
        if result["outcome"] == "puck_in_goal" and not is_seen_at_end:
            self.__warn(result, "warning", f"the outcome is puck_in_goal but the last picture shows no puck the detector is sure of (score "
                                           f"{settings['puck_last_picture_minimum_confidence']} or more): look at it")
        if sum(seen) < settings["puck_seen_note_fraction"] * len(seen):
            self.__warn(result, "note", f"the puck is seen in only {sum(seen)} of {len(seen)} pictures (puck detector; under "
                                        f"{100 * settings['puck_seen_note_fraction']:.0f} %): the camera often did not see it, or the detector missed it")


    def __check_flags(
            self,
            result: Dict,
            data: Dict
    ) -> None:
        """Mistake and lost flags, their counts in episode.json, sharpness and the step budget."""

        settings = self.check_parameters
        rows = data["used_rows"]
        summary = data["summary"] or {}
        mistakes = [row.get("step") for row in rows if row.get("is_mistake") == "1"]
        lost = [row.get("step") for row in rows if row.get("is_lost") == "1"]
        result.update({"mistake_steps": mistakes, "lost_steps": lost})
        flag_text = f"mistake {len(mistakes)}" + (f" (steps {short_list(mistakes)})" if mistakes else "")
        flag_text += f", lost {len(lost)}" + (f" (steps {short_list(lost)})" if lost else "")
        result["lines"].append(("flags", flag_text))
        if summary and rows:
            if summary.get("mistake_steps") is not None and summary["mistake_steps"] != len(mistakes):
                self.__warn(result, "warning", f"episode.json counts {summary['mistake_steps']} mistakes, steps.csv {len(mistakes)}")
            if summary.get("lost_steps") is not None and summary["lost_steps"] != len(lost):
                self.__warn(result, "warning", f"episode.json counts {summary['lost_steps']} lost steps, steps.csv {len(lost)}")
            if summary.get("steps") is not None and summary["steps"] != len(rows):
                self.__warn(result, "warning", f"episode.json says {summary['steps']} steps, steps.csv has {len(rows)} rows")
        if mistakes or lost:
            self.__warn(result, "note", f"{len(mistakes)} mistake and {len(lost)} lost flags: training drops those steps (and the steps undoing a mistake)")

        sharpness = [number_or_none(row.get("sharpness")) for row in rows]
        sharpness_spread = spread_of(sharpness)
        if sharpness_spread is not None:
            out_of_focus = [row.get("step") for row, value in zip(rows, sharpness) if value is not None and value < settings["sharpness_warning_below"]]
            drops = []
            each_side = int(settings["sharpness_neighbours_each_side"])
            for index, value in enumerate(sharpness):
                neighbours = [other for other in sharpness[max(0, index - each_side):index] + sharpness[index + 1:index + 1 + each_side]
                              if other is not None]
                if value is not None and neighbours and value < settings["sharpness_drop_fraction"] * float(numpy.median(neighbours)):
                    drops.append(f"{rows[index].get('step')} ({value:.0f} against {numpy.median(neighbours):.0f})")
            result["lines"].append(("sharpness", f"{format_spread(sharpness_spread, '', 0).strip()}; {len(out_of_focus)} under "
                                                 f"{settings['sharpness_warning_below']:.0f}, {len(drops)} under {settings['sharpness_drop_fraction']:.0%} of their neighbours"))
            if out_of_focus:
                self.__warn(result, "warning", f"{len(out_of_focus)} pictures look out of focus (sharpness under {settings['sharpness_warning_below']:.0f}; "
                                               f"steps {short_list(out_of_focus)})")
            if drops:
                self.__warn(result, "note", f"{len(drops)} pictures are much less sharp than their neighbours, probably blurred by motion: {short_list(drops, 5)}")

        budget = summary.get("step_budget")
        if budget and len(rows) > budget:
            self.__warn(result, "note", f"{len(rows)} steps, more than the step budget of {budget} (1.5 to 2 times the line length)")
        if summary.get("hand_corrections"):
            self.__warn(result, "note", f"{summary['hand_corrections']} corrections by hand (h): the puck was put back by hand each time")


    @staticmethod
    def __verdict(result: Dict) -> str:
        """
        "kept" (completed with the puck in the goal), "discarded", or "no outcome" (quit, unfinished, or the outcome
        was not recorded), from the tool's own record; the warnings say whether a kept episode needs a look.
        """

        if result.get("is_simulated"):
            return "simulated"
        status, outcome = result.get("status"), result.get("outcome")
        if status in KEPT_STATUSES:
            if outcome == "puck_in_goal":
                return "kept"
            if outcome == "puck_not_in_goal":
                return "discarded"
            return "no outcome"
        if status is None or status == "quit":
            return "no outcome"
        return "discarded"



def warning_counts(result: Dict) -> Tuple[int, int]:
    """The numbers of warnings and of notes of one episode."""

    levels = [level for level, _ in result["warnings"]]
    return levels.count("warning"), levels.count("note")


def render_report(
        results: List[Dict],
        folders: List[Path],
        check_parameters: Dict,
        seconds: float
) -> str:
    """
    Write the report as Markdown (it reads as plain text in a terminal too): an overview, a table per sheet, the
    session's warnings, then every episode's findings and warnings.

    Parameters
    ----------
    results : list of dict
        The episodes' results (EpisodeChecker.check_folders).
    folders : list of Path
        The folders that were checked.
    check_parameters : dict
        The thresholds used.
    seconds : float
        How long the check took.

    Returns
    -------
    str
        The report.
    """

    lines = [f"# Episode check: {', '.join(folder.name for folder in folders)}", "",
             f"{len(results)} episodes checked in {seconds:.1f} s on {datetime.datetime.now().isoformat(timespec = 'minutes')} "
             f"(check_episodes.py; read-only; thresholds in CHECK_PARAMETERS). Folders: {', '.join(str(folder) for folder in folders)}.",
             "Calculated, not measured: tips, step lengths, lean and headings (arm model on the logged joints). "
             "\"Puck seen\" comes from the puck detector: a pointer for a look at the pictures.",
             "Verdict, from the tool's own record: kept = completed with the puck in the goal; discarded; no outcome = quit, "
             "unfinished or outcome not recorded. A WARNING means: look before training on the episode; a note is for information.", ""]

    lines += ["## Overview", "", "| episode | mode | status, outcome | rows | pictures | verdict | warnings, notes |", "|---|---|---|---|---|---|---|"]
    for result in results:
        warnings, notes = warning_counts(result)
        ending = result.get("outcome") or result.get("discard_reason") or "-"
        lines.append(f"| {result['episode']} | {result.get('mode') or '?'} | {result.get('status') or '?'}, {ending} | {result.get('rows', 0)} | "
                     f"{result.get('frames', 0)} | {result['verdict']} | {warnings}, {notes} |")

    lines += ["", "## Per sheet", "",
              "| sheet | episodes | kept | discarded | no outcome | rows in kept | rows in all | flagged rows in kept | kept with warnings | camera mounts (kept) | start poses (rod end x, y) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for path_id in sorted({result.get("path_id", "unknown") for result in results}):
        group = [result for result in results if result.get("path_id", "unknown") == path_id]
        kept = [result for result in group if result["verdict"] == "kept"]
        verdicts = Counter(result["verdict"] for result in group)
        flagged = sum(len(result.get("mistake_steps", [])) + len(result.get("lost_steps", [])) for result in kept)
        mounts = sorted({result.get("camera_mount") or "unknown" for result in kept})
        starts = sorted({f"{result['start_pose'][0]:.4f}, {result['start_pose'][1]:.4f}" for result in group if result.get("start_pose")})
        lines.append(f"| {path_id} | {len(group)} | {len(kept)} | {verdicts.get('discarded', 0)} | {verdicts.get('no outcome', 0)} | "
                     f"{sum(result.get('rows', 0) for result in kept)} | {sum(result.get('rows', 0) for result in group)} | {flagged} | "
                     f"{sum(1 for result in kept if warning_counts(result)[0])} | {'; '.join(mounts) or '-'} | {'; '.join(starts) or '-'} |")

    session_warnings = []
    kept_results = [result for result in results if result["verdict"] == "kept"]
    kept_mounts = {result.get("camera_mount") for result in kept_results}
    kept_mappings = {result.get("mapping_version") for result in kept_results}
    if len(kept_mounts) > 1:
        session_warnings.append(f"kept episodes come from {len(kept_mounts)} camera mounts ({', '.join(sorted(str(mount) for mount in kept_mounts))}): never mix them in one training set")
    if len(kept_mappings) > 1:
        session_warnings.append(f"kept episodes use {len(kept_mappings)} mappings ({', '.join(sorted(str(mapping) for mapping in kept_mappings))})")
    current = [result for result in kept_results if result.get("camera_mount") == check_parameters["current_camera_mount"]
               and result.get("mapping_version") == check_parameters["current_mapping_version"]]
    session_warnings.append(f"{len(current)} of {len(kept_results)} kept episodes have today's camera mount and mapping "
                            f"({sum(result.get('rows', 0) for result in current)} rows); "
                            f"{sum(1 for result in current if not warning_counts(result)[0])} of them without warnings")
    lines += ["", "## Session", ""] + [f"- {text}" for text in session_warnings]

    lines += ["", "## Episodes"]
    for result in results:
        warnings, notes = warning_counts(result)
        ending = result.get("outcome") or result.get("discard_reason") or "no outcome"
        lines += ["", f"### {result['episode']}", "",
                  f"**{result.get('mode') or '?'}; {result.get('status') or '?'}, {ending}: {result['verdict'].upper()}"
                  f"{f', {warnings} warnings' if warnings else ', no warnings'}{f', {notes} notes' if notes else ''}**", ""]
        lines += [f"- {label}: {text}" for label, text in result.get("lines", [])]
        if result["warnings"]:
            lines.append("- findings:")
            ordered = [item for item in result["warnings"] if item[0] == "warning"] + [item for item in result["warnings"] if item[0] == "note"]
            lines += [f"  - {'WARNING' if level == 'warning' else 'note'}: {text}" for level, text in ordered]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Check the folders named on the command line (default: the newest session) and print the report."""

    parser = argparse.ArgumentParser(description = "Check recorded episodes (read-only) and tell whether they are good training data.")
    parser.add_argument("folders", nargs = "*", help = "session or episode folders (default: the newest session in 03_data/raw)")
    parser.add_argument("--report", help = "also write the report as Markdown to this path")
    parser.add_argument("--settings", help = "a JSON file of CHECK_PARAMETERS overrides, merged over the defaults")
    parser.add_argument("--no-pictures", action = "store_true", help = "skip decoding the pictures (puck and camera-shift checks); md5 sums are still checked")
    arguments = parser.parse_args()

    check_parameters = json.loads(Path(arguments.settings).read_text(encoding = "utf-8")) if arguments.settings else {}
    if arguments.no_pictures:
        check_parameters["is_picture_check_on"] = False
    checker = EpisodeChecker(check_parameters = check_parameters)

    if arguments.folders:
        folders = [Path(folder).resolve() for folder in arguments.folders]
    else:
        raw_folder = PROJECT_FOLDER / checker.check_parameters["raw_folder"]
        if not raw_folder.is_dir():
            sys.exit(f"no recordings on this laptop: {raw_folder} does not exist (03_data is not in git); name a session or episode folder")
        newest = newest_session_folder(raw_folder)
        if newest is None:
            sys.exit(f"no session folder in {raw_folder}")
        folders = [newest]
    for folder in folders:
        if not folder.is_dir():
            sys.exit(f"not a folder: {folder}")

    started = time.perf_counter()
    results = checker.check_folders(folders)
    if not results:
        sys.exit(f"no episode folders (<sheet>_episode_NNN) in {', '.join(str(folder) for folder in folders)}")
    report = render_report(results, folders, checker.check_parameters, time.perf_counter() - started)

    # The report file first, so that a terminal pipe closed early (head, Select-Object -First) cannot lose it.
    if arguments.report:
        report_path = Path(arguments.report).resolve()
        report_path.parent.mkdir(parents = True, exist_ok = True)
        report_path.write_text(report, encoding = "utf-8")
    sys.stdout.reconfigure(encoding = "utf-8", errors = "replace")
    try:
        print(report)
        if arguments.report:
            print(f"report written to {report_path}")
    except BrokenPipeError:
        sys.stderr.close()


if __name__ == "__main__":
    main()
