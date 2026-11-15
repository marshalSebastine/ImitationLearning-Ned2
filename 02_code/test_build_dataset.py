"""
Tests of build_dataset.py on small made-up episodes; no robot and no recorded episode is used.

Run from 02_code:
    python test_build_dataset.py
"""

import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from build_dataset import build_dataset, lighting_group, split_of_episodes
from config import DATASET_PARAMETERS


def write_episode(
        raw_folder: Path,
        session: str,
        folder_name: str,
        actions: List[str],
        status: str = "completed",
        score: Optional[int] = 3,
        lighting_note: str = "Cieling light",
        marks: Optional[Dict[int, str]] = None,
        is_checksum_wrong_at: Optional[int] = None
) -> Path:
    """
    Write one made-up episode folder: a small file per picture, steps.csv, episode.json and review.json.

    ``marks`` sets "is_mistake" or "is_lost" on the step with that index; ``is_checksum_wrong_at`` writes a wrong
    checksum for that step.
    """

    episode_folder = raw_folder / session / folder_name
    (episode_folder / "frames").mkdir(parents = True)
    columns = ["step", "frame", "frame_md5", "action", "is_mistake", "is_lost", "forearm_label", "tip_forward_m"]
    rows = []
    for index, action in enumerate(actions):
        content = f"{folder_name} picture {index}".encode("utf-8")
        (episode_folder / "frames" / f"{index:03d}.jpg").write_bytes(content)
        checksum = hashlib.md5(content).hexdigest() if index != is_checksum_wrong_at else "0" * 32
        mark = (marks or {}).get(index)
        rows.append({"step": index, "frame": f"frames/{index:03d}.jpg", "frame_md5": checksum, "action": action,
                     "is_mistake": "1" if mark == "mistake" else "0", "is_lost": "1" if mark == "lost" else "0",
                     "forearm_label": "none", "tip_forward_m": 0.01})
    with open(episode_folder / "steps.csv", "w", encoding = "utf-8", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = columns)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"status": status, "operator": "marshal", "mode": "hand_guided_exact", "plan_number": "01-01",
               "lighting_note": lighting_note, "outcome": "puck_in_goal" if status == "completed" else None}
    (episode_folder / "episode.json").write_text(json.dumps(summary), encoding = "utf-8")
    if score is not None:
        (episode_folder / "review.json").write_text(json.dumps({"score": score}), encoding = "utf-8")
    return episode_folder


def read_samples(output_folder: Path) -> List[Dict]:
    """The rows of a built dataset's samples.csv."""

    with open(output_folder / "samples.csv", encoding = "utf-8", newline = "") as file:
        return list(csv.DictReader(file))


def test_rows_and_episodes_are_left_out_as_set() -> None:
    """Unfinished and excluded episodes, small motions, marked rows, excluded pictures and bad checksums are left out."""

    with tempfile.TemporaryDirectory() as folder:
        raw_folder = Path(folder) / "raw"
        session = "2026-11-08_session_01_marshal"
        write_episode(raw_folder, session, "path_01_straight_line_episode_001_plan_01-01_try_1_score_03",
                      ["small_motion", "forward", "forward", "forward_left", "stop"],
                      marks = {2: "mistake"}, is_checksum_wrong_at = 3)
        write_episode(raw_folder, session, "path_01_straight_line_episode_002_plan_01-02_try_1_score_02",
                      ["forward", "forward", "right", "stop"], score = 2)
        write_episode(raw_folder, session, "path_01_straight_line_episode_003_plan_01-03_try_1",
                      ["forward", "forward"], status = "discarded", score = None)
        write_episode(raw_folder, session, "path_01_straight_line_episode_004_plan_01-04_try_1_score_01",
                      ["forward", "stop"], score = 1)
        overrides = {
            "raw_folder": raw_folder, "processed_folder": Path(folder) / "processed", "dataset_name": "test",
            "excluded_episodes": {f"{session}/path_01_straight_line_episode_004": "made-up reason"},
            "excluded_pictures": {f"{session}/path_01_straight_line_episode_002": [["002"], "hand"],
                                  f"{session}/path_09_typo_episode_001": [["000"], "no such episode"]}
        }
        summary = build_dataset(overrides)
        samples = read_samples(Path(folder) / "processed" / "test")
        kept = sorted((sample["episode_key"].split("/")[1], sample["picture"].split("/")[-1]) for sample in samples)
        assert kept == [("path_01_straight_line_episode_001", "001.jpg"), ("path_01_straight_line_episode_001", "004.jpg"),
                        ("path_01_straight_line_episode_002", "000.jpg"), ("path_01_straight_line_episode_002", "001.jpg"),
                        ("path_01_straight_line_episode_002", "003.jpg")], kept
        left_out = summary["counts"]["left_out_pictures"]
        assert left_out == {"action small_motion": 1, "marked a mistake": 1, "checksum differs": 1, "excluded picture": 1}, left_out
        assert summary["counts"]["episodes_kept"] == 2 and summary["counts"]["episodes_left_out"] == 2, summary["counts"]
        assert summary["unmatched_exclusion_keys"] == [f"{session}/path_09_typo_episode_001"], summary["unmatched_exclusion_keys"]
        assert {sample["score"] for sample in samples} == {"3", "2"}
        assert all(sample["lighting"] == "ceiling" for sample in samples)
        with open(Path(folder) / "processed" / "test" / "episodes.csv", encoding = "utf-8", newline = "") as file:
            reasons = {row["episode_key"].split("/")[1]: row["reason"] for row in csv.DictReader(file)}
        assert reasons["path_01_straight_line_episode_003"] == "status discarded", reasons
        assert reasons["path_01_straight_line_episode_004"] == "excluded: made-up reason", reasons


def test_splits_keep_each_episode_together() -> None:
    """Every episode's pictures sit in one split, each sheet has test and validation episodes, and the split repeats."""

    with tempfile.TemporaryDirectory() as folder:
        raw_folder = Path(folder) / "raw"
        for sheet in ("path_01_straight_line", "path_04_meander"):
            for number in range(1, 21):
                write_episode(raw_folder, "2026-11-08_session_01_emranul", f"{sheet}_episode_{number:03d}_plan_x_try_1_score_03",
                              ["forward", "forward_left", "forward", "stop"])
        overrides = {"raw_folder": raw_folder, "processed_folder": Path(folder) / "processed", "dataset_name": "test",
                     "excluded_episodes": {}, "excluded_pictures": {}}
        summary = build_dataset(overrides)
        samples = read_samples(Path(folder) / "processed" / "test")
        splits_per_episode = {}
        for sample in samples:
            splits_per_episode.setdefault(sample["episode_key"], set()).add(sample["split"])
        assert all(len(splits) == 1 for splits in splits_per_episode.values()), splits_per_episode
        for sheet, counts in summary["counts"]["episodes_per_sheet_and_split"].items():
            assert counts == {"train": 14, "validation": 3, "test": 3}, (sheet, counts)
        first = {sample["picture"]: sample["split"] for sample in samples}
        build_dataset(overrides, is_overwrite_allowed = True)
        second = {sample["picture"]: sample["split"] for sample in read_samples(Path(folder) / "processed" / "test")}
        assert first == second


def test_existing_dataset_is_not_overwritten() -> None:
    """A second build under the same name stops unless overwriting is allowed."""

    with tempfile.TemporaryDirectory() as folder:
        raw_folder = Path(folder) / "raw"
        write_episode(raw_folder, "2026-11-08_session_01_emranul", "path_01_straight_line_episode_001_plan_x_try_1_score_03", ["forward", "stop"])
        overrides = {"raw_folder": raw_folder, "processed_folder": Path(folder) / "processed", "dataset_name": "test",
                     "excluded_episodes": {}, "excluded_pictures": {}}
        build_dataset(overrides)
        try:
            build_dataset(overrides)
        except FileExistsError:
            pass
        else:
            raise AssertionError("a second build overwrote the dataset")


def test_lighting_notes_and_small_sheets() -> None:
    """The hand-typed lighting notes of the main data fall into the right group; a two-episode sheet is all train."""

    groups = DATASET_PARAMETERS["lighting_groups"]
    notes = {"Cieling light": "ceiling", "ceiling lamps on": "ceiling", "natural light": "natural",
             "cloudy day, episodes done on natural light": "natural",
             "ceiling lights (corrected 8.11.26 by Emranul: wrongly marked as natural light)": "ceiling", None: "unknown"}
    for note, expected in notes.items():
        assert lighting_group(note, groups) == expected, (note, lighting_group(note, groups))
    splits = split_of_episodes(["a/path_01_episode_001", "a/path_01_episode_002"], dict(DATASET_PARAMETERS))
    assert set(splits.values()) == {"train"}, splits


if __name__ == "__main__":
    test_rows_and_episodes_are_left_out_as_set()
    print("test_rows_and_episodes_are_left_out_as_set: passed")
    test_splits_keep_each_episode_together()
    print("test_splits_keep_each_episode_together: passed")
    test_existing_dataset_is_not_overwritten()
    print("test_existing_dataset_is_not_overwritten: passed")
    test_lighting_notes_and_small_sheets()
    print("test_lighting_notes_and_small_sheets: passed")
    print("all tests passed")
