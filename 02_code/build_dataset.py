"""
Build one table of training pictures from the main data (Marshal, 13.11.26).

The recording tool writes one folder per episode in ``03_data/raw/main_data``. This script reads them all and writes,
into ``03_data/processed/<dataset name>/``:

- samples.csv: one row per kept picture, with its path (relative to the raw folder), the episode's sheet, plan number,
  operator, lighting, score and split, and the step's move name and exact numbers copied from steps.csv;
- episodes.csv: one row per episode folder, kept or not, and why;
- summary.json: the counts per sheet, split and move, the settings used and the code version.

Nothing in the raw folder is changed, copied or renamed. What is left out, and why, is set in ``DATASET_PARAMETERS``
in config.py: episodes that did not finish, moves no picture can show, rows marked as a mistake or lost, and the
pictures and episodes the operators' notes name (a hand in view, a puck tipped over).

Run from 02_code:
    python build_dataset.py
    python build_dataset.py --name v02_main_data --overwrite
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DATASET_PARAMETERS


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
EPISODE_NAME_PATTERN = re.compile(r"^(?P<sheet>path_\d\d)_.*?_episode_(?P<number>\d+)")
SPLIT_NAMES = ["train", "validation", "test"]


def merged_dataset_parameters(dataset_parameters: Optional[Dict] = None) -> Dict:
    """``DATASET_PARAMETERS`` with any overrides; relative folders are resolved against the project folder."""

    parameters = dict(DATASET_PARAMETERS)
    parameters.update(dataset_parameters or {})
    for key in ("raw_folder", "processed_folder"):
        folder = Path(parameters[key])
        parameters[key] = folder if folder.is_absolute() else PROJECT_FOLDER / folder
    return parameters


def episode_key(episode_folder: Path) -> Optional[str]:
    """
    The key the exclusion lists use: "<session>/<sheet name>_episode_NNN".

    The key leaves out the plan number, try and score, so it stays the same when a folder gains its score. Folders that
    do not look like an episode give None.
    """

    match = re.match(r"^(.*_episode_\d+)", episode_folder.name)
    if match is None:
        return None
    return f"{episode_folder.parent.name}/{match.group(1)}"


def lighting_group(
        lighting_note: Optional[str],
        lighting_groups: List
) -> str:
    """The lighting group of a hand-typed note: the first group one of whose words appears in it, else "unknown"."""

    note = (lighting_note or "").lower()
    for group, words in lighting_groups:
        if any(word in note for word in words):
            return group
    return "unknown"


def split_of_episodes(
        episode_keys: List[str],
        parameters: Dict
) -> Dict[str, str]:
    """
    Give each episode of one sheet a split.

    The episodes are ordered by a hash of their key and the seed, so the choice is random but the same on every run
    and every computer. The first ones go to test, the next to validation, the rest to train; each of test and
    validation gets at least one episode when the sheet has three or more.

    Parameters
    ----------
    episode_keys : list of str
        The keys of one sheet's kept episodes.
    parameters : dict
        Merged ``DATASET_PARAMETERS``; uses "test_share", "validation_share" and "split_seed".

    Returns
    -------
    dict
        Episode key to "train", "validation" or "test".
    """

    def hashed(key: str) -> str:
        return hashlib.sha256(f"{parameters['split_seed']}/{key}".encode("utf-8")).hexdigest()

    ordered = sorted(episode_keys, key = hashed)
    count = len(ordered)
    minimum = 1 if count >= 3 else 0
    test_count = max(minimum, round(count * parameters["test_share"]))
    validation_count = max(minimum, round(count * parameters["validation_share"]))
    splits = {}
    for index, key in enumerate(ordered):
        if index < test_count:
            splits[key] = "test"
        elif index < test_count + validation_count:
            splits[key] = "validation"
        else:
            splits[key] = "train"
    return splits


def read_json(path: Path) -> Optional[Dict]:
    """A JSON file's content, or None when it is missing or broken."""

    try:
        return json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None


def file_md5(path: Path) -> str:
    """The MD5 checksum of a file, as the recording tool writes it into steps.csv."""

    return hashlib.md5(path.read_bytes()).hexdigest()


def code_version() -> str:
    """The short git commit of the code, with "+changes" when files are changed but not committed; "unknown" without git."""

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd = PROJECT_FOLDER, capture_output = True,
                                text = True, timeout = 30).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--", "02_code"], cwd = PROJECT_FOLDER,
                                 capture_output = True, text = True, timeout = 30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{commit}+changes" if commit and changed else (commit or "unknown")


def read_episode(
        episode_folder: Path,
        parameters: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    Read one episode folder and decide which of its pictures are kept.

    Parameters
    ----------
    episode_folder : Path
        One episode's folder, with episode.json, review.json, steps.csv and frames/.
    parameters : dict
        Merged ``DATASET_PARAMETERS``.

    Returns
    -------
    tuple of (dict, list of dict)
        The episode's row for episodes.csv ("is_kept" and "reason" say whether and why not), and its kept pictures'
        rows for samples.csv, without the split, which is given once all episodes are known.
    """

    key = episode_key(episode_folder)
    name_match = EPISODE_NAME_PATTERN.match(episode_folder.name)
    summary = read_json(episode_folder / "episode.json") or {}
    review = read_json(episode_folder / "review.json") or {}
    episode = {
        "episode_key": key, "folder": episode_folder.relative_to(parameters["raw_folder"]).as_posix(),
        "sheet": name_match.group("sheet") if name_match else "", "plan_number": summary.get("plan_number", ""),
        "operator": summary.get("operator", ""), "mode": summary.get("mode", ""), "status": summary.get("status", ""),
        "outcome": summary.get("outcome", ""), "score": review.get("score", ""),
        "lighting": lighting_group(summary.get("lighting_note"), parameters["lighting_groups"]),
        "lighting_note": summary.get("lighting_note", ""), "is_kept": False, "reason": "", "pictures_kept": 0,
        "pictures_left_out": 0
    }

    if not summary:
        episode["reason"] = "no episode.json"
        return episode, []
    if summary.get("status") not in parameters["kept_statuses"]:
        episode["reason"] = f"status {summary.get('status')}"
        return episode, []
    if key in parameters["excluded_episodes"]:
        episode["reason"] = f"excluded: {parameters['excluded_episodes'][key]}"
        return episode, []
    steps_path = episode_folder / "steps.csv"
    if not steps_path.exists():
        episode["reason"] = "no steps.csv"
        return episode, []

    excluded_frames = set(parameters["excluded_pictures"].get(key, [[], ""])[0])
    left_out = Counter()
    samples = []
    with open(steps_path, encoding = "utf-8", newline = "") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        frame_path = episode_folder / row.get("frame", "")
        frame_name = Path(row.get("frame", "")).stem
        if not row.get("frame") or not frame_path.is_file():
            left_out["picture missing"] += 1
        elif row.get("action") in parameters["dropped_actions"]:
            left_out[f"action {row.get('action')}"] += 1
        elif parameters["is_mistake_dropped"] and row.get("is_mistake") == "1":
            left_out["marked a mistake"] += 1
        elif parameters["is_lost_dropped"] and row.get("is_lost") == "1":
            left_out["marked lost"] += 1
        elif frame_name in excluded_frames:
            left_out["excluded picture"] += 1
        elif parameters["is_checksum_checked"] and row.get("frame_md5") and file_md5(frame_path) != row["frame_md5"]:
            left_out["checksum differs"] += 1
        else:
            sample = {"picture": frame_path.relative_to(parameters["raw_folder"]).as_posix(), "episode_key": key}
            for column in ("sheet", "plan_number", "operator", "mode", "score", "lighting"):
                sample[column] = episode[column]
            for column in parameters["copied_columns"]:
                sample[column] = row.get(column, "")
            samples.append(sample)

    episode["is_kept"] = bool(samples)
    episode["reason"] = "; ".join(f"{reason} {count}" for reason, count in sorted(left_out.items())) or "all kept"
    episode["pictures_kept"] = len(samples)
    episode["pictures_left_out"] = sum(left_out.values())
    episode["left_out"] = dict(left_out)
    return episode, samples


def write_csv(
        path: Path,
        rows: List[Dict],
        columns: List[str]
) -> None:
    """Write rows as a CSV file with these columns, in this order; other keys are left out."""

    with open(path, "w", encoding = "utf-8", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = columns, extrasaction = "ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
        dataset_parameters: Optional[Dict] = None,
        is_overwrite_allowed: bool = False
) -> Dict:
    """
    Build the dataset table from every episode in the raw folder and write it to the processed folder.

    Parameters
    ----------
    dataset_parameters : dict, optional
        Overrides for ``DATASET_PARAMETERS`` (for example "dataset_name", "raw_folder" or "processed_folder").
    is_overwrite_allowed : bool, default False
        Replace the files of a dataset of the same name; without it an existing dataset stops the build.

    Returns
    -------
    dict
        The summary written to summary.json.
    """

    parameters = merged_dataset_parameters(dataset_parameters)
    output_folder = parameters["processed_folder"] / parameters["dataset_name"]
    if (output_folder / "samples.csv").exists() and not is_overwrite_allowed:
        raise FileExistsError(f"{output_folder} already holds a dataset; choose a new --name or pass --overwrite")
    if not parameters["raw_folder"].is_dir():
        raise FileNotFoundError(f"no raw folder at {parameters['raw_folder']}")

    episode_folders = sorted(folder for folder in parameters["raw_folder"].glob("*_session_*/*_episode_*") if folder.is_dir())
    episodes, samples = [], []
    for episode_folder in episode_folders:
        episode, episode_samples = read_episode(episode_folder, parameters)
        episodes.append(episode)
        samples.extend(episode_samples)

    # A typing error in an exclusion key would quietly keep the pictures it meant to leave out.
    found_keys = {episode["episode_key"] for episode in episodes}
    unmatched = sorted(key for key in list(parameters["excluded_episodes"]) + list(parameters["excluded_pictures"])
                       if key not in found_keys)

    kept_by_sheet = defaultdict(list)
    for episode in episodes:
        if episode["is_kept"]:
            kept_by_sheet[episode["sheet"]].append(episode["episode_key"])
    splits = {}
    for sheet_keys in kept_by_sheet.values():
        splits.update(split_of_episodes(sheet_keys, parameters))
    for episode in episodes:
        episode["split"] = splits.get(episode["episode_key"], "")
    for sample in samples:
        sample["split"] = splits[sample["episode_key"]]

    counts = {
        "pictures": len(samples),
        "episodes_kept": len(splits),
        "episodes_left_out": len(episodes) - len(splits),
        "pictures_per_sheet_and_split": {sheet: dict(Counter(sample["split"] for sample in samples if sample["sheet"] == sheet))
                                         for sheet in sorted(kept_by_sheet)},
        "episodes_per_sheet_and_split": {sheet: dict(Counter(splits[key] for key in keys)) for sheet, keys in sorted(kept_by_sheet.items())},
        "actions": dict(Counter(sample["action"] for sample in samples).most_common()),
        "actions_per_split": {split: dict(Counter(sample["action"] for sample in samples if sample["split"] == split).most_common())
                              for split in SPLIT_NAMES},
        "forearm_labels": dict(Counter(sample["forearm_label"] for sample in samples).most_common()),
        "lighting": dict(Counter(sample["lighting"] for sample in samples).most_common()),
        "operators": dict(Counter(sample["operator"] for sample in samples).most_common()),
        "scores": dict(Counter(str(sample["score"]) for sample in samples).most_common()),
        "left_out_pictures": dict(sum((Counter(episode.get("left_out", {})) for episode in episodes), Counter()).most_common())
    }
    settings = {key: (str(value) if isinstance(value, Path) else value) for key, value in parameters.items()}
    summary = {"dataset_name": parameters["dataset_name"], "built": datetime.now().isoformat(timespec = "seconds"),
               "code_version": code_version(), "counts": counts, "unmatched_exclusion_keys": unmatched, "settings": settings}

    output_folder.mkdir(parents = True, exist_ok = True)
    sample_columns = ["picture", "split", "episode_key", "sheet", "plan_number", "operator", "mode", "score", "lighting"]
    sample_columns += parameters["copied_columns"]
    write_csv(output_folder / "samples.csv", samples, sample_columns)
    episode_columns = ["episode_key", "folder", "sheet", "plan_number", "operator", "mode", "status", "outcome", "score",
                       "lighting", "lighting_note", "is_kept", "split", "pictures_kept", "pictures_left_out", "reason"]
    write_csv(output_folder / "episodes.csv", episodes, episode_columns)
    (output_folder / "summary.json").write_text(json.dumps(summary, indent = 2), encoding = "utf-8")
    return summary


def main() -> None:
    """Build the dataset named on the command line (default: the name in config) and print a short report."""

    parser = argparse.ArgumentParser(description = "Build the training table from the main data.")
    parser.add_argument("--name", default = DATASET_PARAMETERS["dataset_name"], help = "dataset name (folder under 03_data/processed)")
    parser.add_argument("--overwrite", action = "store_true", help = "replace an existing dataset of the same name")
    arguments = parser.parse_args()
    try:
        summary = build_dataset({"dataset_name": arguments.name}, is_overwrite_allowed = arguments.overwrite)
    except (FileExistsError, FileNotFoundError) as error:
        sys.exit(str(error))

    counts = summary["counts"]
    print(f"dataset {summary['dataset_name']}: {counts['pictures']} pictures from {counts['episodes_kept']} episodes "
          f"({counts['episodes_left_out']} episode folders left out); code {summary['code_version']}")
    for sheet, split_counts in counts["episodes_per_sheet_and_split"].items():
        pictures = counts["pictures_per_sheet_and_split"][sheet]
        print(f"  {sheet}: episodes {split_counts}, pictures {pictures}")
    print(f"  left out pictures: {counts['left_out_pictures']}")
    if summary["unmatched_exclusion_keys"]:
        print(f"WARNING: these exclusion keys match no episode folder: {summary['unmatched_exclusion_keys']}")


if __name__ == "__main__":
    main()
