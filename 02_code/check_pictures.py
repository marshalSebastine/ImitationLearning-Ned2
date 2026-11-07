"""
Flag camera pictures of an episode that may not belong in the training data (Emranul, 7.11.26: "after each episode the
program runs and we get a popup in the form while we are scoring: 5 flagged of 30").

Four checks, each a simple measurement, not a judgement of what the picture shows; a flag means "look at this picture",
and the person decides. Tried on sheet 01 of 6.11.26 (23 episodes, 736 pictures):

- blurred: the sharpness logged in steps.csv is under "blurred_below" (lower than the page's median warning).
- no puck: puck_detector finds no red puck.
- dark object: a dark patch outside the robot's own part of the picture (the laptop, a person, a shadow, a hand);
  skipped with the forearm turned far, when the robot's own base comes into view.
- sudden change: the picture differs from the next one far more than the episode's steps usually do (a hand put in
  or taken out, mostly in the first picture).

The result goes into picture_check.json beside the episode's files, and the flagged pictures, marked, into
picture_check.jpg; nothing the recording tool wrote is changed.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy

from config import PICTURE_CHECK_PARAMETERS
from puck_detector import PuckDetector


def merged_check_parameters(check_parameters: Optional[Dict] = None) -> Dict:
    """``PICTURE_CHECK_PARAMETERS`` with any overrides."""

    parameters = dict(PICTURE_CHECK_PARAMETERS)
    parameters.update(check_parameters or {})
    return parameters


def logged_steps(episode_folder: Path) -> Dict[str, Dict]:
    """The rows steps.csv logged, by frame name ("000"); empty without steps.csv."""

    steps_path = episode_folder / "steps.csv"
    if not steps_path.exists():
        return {}
    with open(steps_path, encoding = "utf-8", newline = "") as file:
        return {Path(row["frame"]).stem: row for row in csv.DictReader(file) if row.get("frame")}


def logged_value(
        row: Optional[Dict],
        column: str
) -> Optional[float]:
    """A number from a steps.csv row, or None when the row or the value is missing."""

    if not row or row.get(column) in (None, ""):
        return None
    return float(row[column])


def largest_dark_patch_px(
        picture: numpy.ndarray,
        parameters: Dict
) -> int:
    """The area in pixels of the largest dark patch outside the robot's own part of the picture."""

    dark = (cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)[:, :, 2] < parameters["dark_value_below"]).astype(numpy.uint8)
    left, top, right, bottom = parameters["robot_area_px"]
    dark[top:bottom, left:right] = 0
    count, _, statistics, _ = cv2.connectedComponentsWithStats(dark)
    return int(statistics[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0


def check_episode_pictures(
        episode_folder: Path,
        check_parameters: Optional[Dict] = None
) -> Dict:
    """
    Check every picture of one episode and write picture_check.json and picture_check.jpg beside it.

    Parameters
    ----------
    episode_folder : Path
        The episode's folder, with its pictures in frames/.
    check_parameters : dict, optional
        Overrides for ``PICTURE_CHECK_PARAMETERS``.

    Returns
    -------
    dict
        "episode", "total" (pictures checked), "flagged_count" and "flagged": one entry per flagged picture with
        "frame" and "reasons" (plain words).
    """

    parameters = merged_check_parameters(check_parameters)
    episode_folder = Path(episode_folder)
    frames = sorted((episode_folder / "frames").glob("*.jpg"))
    steps = logged_steps(episode_folder)
    detector = PuckDetector()
    reasons: Dict[str, List[str]] = {}
    small_pictures = []
    for frame in frames:
        picture = cv2.imread(str(frame))
        found = []
        sharpness = logged_value(steps.get(frame.stem), "sharpness")
        if sharpness is not None and sharpness < parameters["blurred_below"]:
            found.append(f"blurred (sharpness {sharpness:.0f})")
        if detector.detect(picture) is None:
            found.append("no puck seen")
        turn_deg = logged_value(steps.get(frame.stem), "forearm_turn_deg") or 0.0
        dark_px = largest_dark_patch_px(picture, parameters) if abs(turn_deg) <= parameters["dark_check_turn_most_deg"] else 0
        if dark_px > parameters["dark_patch_above_px"]:
            found.append(f"dark object in view ({dark_px} px): laptop, person, shadow or hand?")
        reasons[frame.stem] = found
        small_pictures.append(cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), tuple(parameters["change_picture_size"])).astype(float))

    # A step changes the picture a little; a hand put in or taken out changes it a lot, from one picture to the next.
    changes = [float(numpy.abs(small_pictures[index] - small_pictures[index + 1]).mean()) for index in range(len(small_pictures) - 1)]
    if changes:
        limit = max(parameters["change_minimum"], parameters["change_ratio"] * float(numpy.median(changes)))
        for index, frame in enumerate(frames):
            before = changes[index - 1] if index > 0 else None
            after = changes[index] if index < len(changes) else None
            # The first picture has only a next one; any other must differ from both neighbours to stand out on its own.
            if index == 0:
                is_sudden = after is not None and after > limit
            else:
                is_sudden = before is not None and after is not None and before > limit and after > limit
            if is_sudden:
                reasons[frame.stem].append("sudden change from the next picture: hand or object put in or taken out?")

    flagged = [{"frame": name, "reasons": found} for name, found in reasons.items() if found]
    result = {"episode": episode_folder.name, "total": len(frames), "flagged_count": len(flagged), "flagged": flagged}
    (episode_folder / "picture_check.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    write_flagged_sheet(episode_folder, flagged, parameters)
    return result


def write_flagged_sheet(
        episode_folder: Path,
        flagged: List[Dict],
        parameters: Dict
) -> None:
    """Put the flagged pictures, small and with their reasons written on them, side by side in picture_check.jpg."""

    sheet_path = episode_folder / "picture_check.jpg"
    if not flagged:
        sheet_path.unlink(missing_ok = True)
        return
    width, height = parameters["sheet_tile_size"]
    tiles = []
    for entry in flagged[:parameters["sheet_tiles_most"]]:
        tile = cv2.resize(cv2.imread(str(episode_folder / "frames" / f"{entry['frame']}.jpg")), (width, height))
        cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (0, 0, 255), 4)
        cv2.putText(tile, entry["frame"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for line, reason in enumerate(entry["reasons"]):
            short_reason = reason.split(":")[0].split(" (")[0]
            cv2.putText(tile, short_reason, (8, height - 12 - 20 * line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(tile)
    columns = min(parameters["sheet_columns"], len(tiles))
    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[index:index + columns]) for index in range(0, len(tiles), columns)]
    cv2.imwrite(str(sheet_path), numpy.vstack(rows))


if __name__ == "__main__":
    # Check episodes by hand: python check_pictures.py <episode folder> [<episode folder> ...]
    for folder in sys.argv[1:]:
        print(json.dumps(check_episode_pictures(Path(folder)), indent = 2))
