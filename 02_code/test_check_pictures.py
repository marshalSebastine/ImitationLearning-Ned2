"""
Tests of check_pictures.py on drawn pictures; no robot and no recorded episode is used.

Run from 02_code:
    python test_check_pictures.py
"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy

from check_pictures import check_episode_pictures


def drawn_picture(
        puck_x: int,
        has_puck: bool = True,
        has_laptop: bool = False,
        has_hand: bool = False
) -> numpy.ndarray:
    """A 640 x 480 picture like the forearm camera's: wooden table, white sheet, grey line, red puck, the robot below."""

    picture = numpy.full((480, 640, 3), (120, 150, 175), dtype = numpy.uint8)
    cv2.rectangle(picture, (120, 80), (520, 470), (235, 235, 235), -1)
    cv2.rectangle(picture, (315, 90), (325, 300), (150, 150, 150), -1)
    if has_puck:
        cv2.circle(picture, (puck_x, 290), 31, (40, 40, 200), -1)
    cv2.rectangle(picture, (260, 330), (380, 480), (30, 30, 30), -1)
    if has_laptop:
        cv2.rectangle(picture, (0, 0), (110, 70), (40, 40, 40), -1)
    if has_hand:
        cv2.ellipse(picture, (300, 120), (200, 110), 0, 0, 360, (140, 170, 215), -1)
    return picture


def write_episode(
        episode_folder: Path,
        pictures: list
) -> None:
    """Write the pictures as frames/000.jpg on, with a steps.csv giving each a sharpness of 150 and no forearm turn."""

    (episode_folder / "frames").mkdir(parents = True)
    lines = ["step,frame,sharpness,forearm_turn_deg"]
    for index, picture in enumerate(pictures):
        cv2.imwrite(str(episode_folder / "frames" / f"{index:03d}.jpg"), picture)
        lines.append(f"{index},frames/{index:03d}.jpg,150.0,0.0")
    (episode_folder / "steps.csv").write_text("\n".join(lines) + "\n", encoding = "utf-8")


def test_clean_episode_has_no_flags() -> None:
    """Ten clean pictures with the puck moving a little each step: nothing flagged, no picture_check.jpg."""

    with tempfile.TemporaryDirectory() as folder:
        episode_folder = Path(folder) / "path_01_straight_line_episode_001"
        write_episode(episode_folder, [drawn_picture(300 + 2 * index) for index in range(10)])
        result = check_episode_pictures(episode_folder)
        assert result["total"] == 10 and result["flagged_count"] == 0, result
        assert not (episode_folder / "picture_check.jpg").exists()
        assert json.loads((episode_folder / "picture_check.json").read_text(encoding = "utf-8")) == result


def test_hand_laptop_and_missing_puck_are_flagged() -> None:
    """A hand in the first picture, a laptop in picture 5 and no puck in picture 7 are flagged; the others not."""

    with tempfile.TemporaryDirectory() as folder:
        episode_folder = Path(folder) / "path_01_straight_line_episode_002"
        pictures = [drawn_picture(300 + 2 * index) for index in range(10)]
        pictures[0] = drawn_picture(300, has_hand = True)
        pictures[5] = drawn_picture(310, has_laptop = True)
        pictures[7] = drawn_picture(314, has_puck = False)
        write_episode(episode_folder, pictures)
        result = check_episode_pictures(episode_folder)
        flagged = {entry["frame"]: " ".join(entry["reasons"]) for entry in result["flagged"]}
        assert set(flagged) == {"000", "005", "007"}, flagged
        assert "sudden change" in flagged["000"] and "dark object" in flagged["005"] and "no puck" in flagged["007"], flagged
        assert (episode_folder / "picture_check.jpg").exists()


if __name__ == "__main__":
    test_clean_episode_has_no_flags()
    print("test_clean_episode_has_no_flags: passed")
    test_hand_laptop_and_missing_puck_are_flagged()
    print("test_hand_laptop_and_missing_puck_are_flagged: passed")
    print("all tests passed")
