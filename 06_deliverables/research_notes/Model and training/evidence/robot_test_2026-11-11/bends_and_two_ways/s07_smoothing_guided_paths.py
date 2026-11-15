"""
What smoothing the hand-guided path, or cutting its waypoints differently, would do to the labels (on paper).

The recording tool keeps the path the hand drew (guided_path.csv in every episode folder: a point every 2 mm of tip
travel or 5 degrees on any joint) and thins it into the waypoints the exact replay visits
(record_episodes.py, ``__exact_waypoints``, line 1105: a waypoint after 1 cm of tip travel OR 5 degrees on any joint).
Every waypoint becomes one picture and one label. This script reads the 209 recorded guided paths (read only), applies
the same thinning rule and variants of it, and counts what the labels would look like:
    as recorded          1 cm of tip travel or 5 degrees on any joint (must give back about the recorded step counts);
    smoothed             the six joint angles averaged over neighbouring guided points before the thinning;
    forearm rule         a waypoint after 1 cm of tip travel or 10 degrees of forearm twist only (the other joints do
                         not cut a waypoint), so a twist in place makes half as many pictures.
For every variant: steps, short steps, twists in place, direction jumps between consecutive steps, and how far the
smoothed tip strays from where the hand had it (sideways and in height, from the arm model).

It cannot say what the pictures would show or whether the puck would still follow: a smoothed replay pushes the puck
along a slightly different path, which only a recording session can test.

Run from this folder:  python s07_smoothing_guided_paths.py
Output: s07_smoothing_guided_paths.out.txt, s07_smoothing_variants.csv.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas

import common
from config import TRAINING_DATA_PARAMETERS
from record_episodes import forearm_camera_heading_rad
from side_view_test import magnet_tip_and_lean

SMOOTHING_PARAMETERS = {
    "raw_folder": Path(TRAINING_DATA_PARAMETERS["raw_folder"]),
    "variants": {
        "as recorded: 1 cm or 5 deg on any joint": {"half_window_points": 0, "tip_m": 0.01, "any_joint_deg": 5.0, "forearm_deg": None},
        "smoothed over 5 guided points, same cutting": {"half_window_points": 2, "tip_m": 0.01, "any_joint_deg": 5.0, "forearm_deg": None},
        "smoothed over 9 guided points, same cutting": {"half_window_points": 4, "tip_m": 0.01, "any_joint_deg": 5.0, "forearm_deg": None},
        "not smoothed, 1 cm or 10 deg of forearm twist": {"half_window_points": 0, "tip_m": 0.01, "any_joint_deg": None, "forearm_deg": 10.0},
        "smoothed over 5 points, 1 cm or 10 deg of forearm twist": {"half_window_points": 2, "tip_m": 0.01, "any_joint_deg": None, "forearm_deg": 10.0}
    },
    "short_step_cm": 0.3,
    "twist_in_place_deg": 3.0
}



def smoothed_joints(
        joints: numpy.ndarray,
        half_window_points: int
) -> numpy.ndarray:
    """
    Return the guided joint angles averaged over neighbouring guided points.

    The window shrinks towards both ends, so the first and the last guided point stay where the hand had them (the
    start on the puck and the end at the goal).

    Parameters
    ----------
    joints : numpy.ndarray
        (points, 6), radians.
    half_window_points : int
        Points before and after that are averaged in; 0 returns the path unchanged.

    Returns
    -------
    numpy.ndarray
        (points, 6), radians.
    """

    if half_window_points == 0:
        return joints.copy()
    result = numpy.zeros_like(joints)
    count = len(joints)
    for position in range(count):
        reach = min(half_window_points, position, count - 1 - position)
        result[position] = joints[position - reach:position + reach + 1].mean(axis = 0)
    return result



def waypoint_positions(
        tip_xy: numpy.ndarray,
        joints: numpy.ndarray,
        rule: Dict
) -> List[int]:
    """
    Return the guided points the exact replay would visit, as ``record_episodes.__exact_waypoints`` picks them.

    Parameters
    ----------
    tip_xy : numpy.ndarray
        (points, 2), the tip in metres.
    joints : numpy.ndarray
        (points, 6), radians.
    rule : dict
        "tip_m" (tip travel that cuts a waypoint), "any_joint_deg" (turn of any joint that cuts one, or None) and
        "forearm_deg" (turn of joint 4 that cuts one, or None).

    Returns
    -------
    list of int
        Positions into the guided path: the first point, every cut and the last point.
    """

    positions = [0]
    last = 0
    for position in range(1, len(tip_xy)):
        tip_moved = float(numpy.hypot(*(tip_xy[position] - tip_xy[last])))
        is_cut = tip_moved >= rule["tip_m"]
        if rule["any_joint_deg"] is not None:
            is_cut = is_cut or float(numpy.max(numpy.abs(joints[position] - joints[last]))) >= math.radians(rule["any_joint_deg"])
        if rule["forearm_deg"] is not None:
            is_cut = is_cut or abs(float(joints[position, 3] - joints[last, 3])) >= math.radians(rule["forearm_deg"])
        if is_cut:
            positions.append(position)
            last = position
    if positions[-1] != len(tip_xy) - 1:
        positions.append(len(tip_xy) - 1)
    return positions



def labels_of_waypoints(
        tips: numpy.ndarray,
        headings_deg: numpy.ndarray,
        joints: numpy.ndarray,
        positions: List[int]
) -> pandas.DataFrame:
    """Return the labels the replay would write for a list of waypoints: step length, direction on the table and twist."""

    chosen = numpy.array(positions)
    moved = tips[chosen[1:], :2] - tips[chosen[:-1], :2]
    return pandas.DataFrame({
        "step_cm": 100.0 * numpy.linalg.norm(moved, axis = 1),
        "table_deg": numpy.degrees(numpy.arctan2(moved[:, 1], moved[:, 0])),
        "picture_deg": numpy.degrees(numpy.arctan2(moved[:, 1], moved[:, 0])) - headings_deg[chosen[:-1]],
        "twist_deg": numpy.degrees(joints[chosen[1:], 3] - joints[chosen[:-1], 3])
    })



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(SMOOTHING_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s07_smoothing_guided_paths.out.txt")
    pandas.set_option("display.width", 320)
    pandas.set_option("display.max_columns", 40)
    frames = common.load_demonstration_frames(settings)
    episodes = frames.groupby("episode_id").agg(episode_folder = ("episode_folder", "first"), sheet = ("sheet", "first"),
                                                hand_corrections = ("hand_corrections", "first"), recorded_steps = ("step_count", "first")).reset_index()

    collected = {name: [] for name in settings["variants"]}
    deviation = {name: [] for name in settings["variants"]}
    height = {name: [] for name in settings["variants"]}
    check_rows = []
    guided_points = 0
    for _, episode in episodes.iterrows():
        guided_file = settings["raw_folder"] / episode["episode_folder"] / "guided_path.csv"
        if not guided_file.exists():
            continue
        guided = pandas.read_csv(guided_file)
        joints = guided[[f"joint_{number}" for number in range(1, 7)]].to_numpy()
        if len(joints) < 3:
            continue
        guided_points += len(joints)
        original_tips = None
        cache = {}
        for name, rule in settings["variants"].items():
            window = rule["half_window_points"]
            if window not in cache:
                smooth = smoothed_joints(joints, window)
                tips = numpy.array([magnet_tip_and_lean(list(row))[0] for row in smooth])
                headings = numpy.degrees(numpy.array([forearm_camera_heading_rad(list(row)) for row in smooth]))
                cache[window] = (smooth, tips, headings)
            smooth, tips, headings = cache[window]
            if window == 0:
                original_tips = tips
            positions = waypoint_positions(tips[:, :2], smooth, rule)
            labels = labels_of_waypoints(tips, headings, smooth, positions)
            labels["episode_id"] = episode["episode_id"]
            labels["sheet"] = episode["sheet"]
            collected[name].append(labels)
            deviation[name].append(1000.0 * numpy.linalg.norm(tips[:, :2] - cache[0][1][:, :2], axis = 1))
            height[name].append(1000.0 * (tips[:, 2] - cache[0][1][:, 2]))
            if name.startswith("as recorded"):
                check_rows.append({"episode_id": episode["episode_id"], "hand_corrections": episode["hand_corrections"],
                                   "recorded_steps": episode["recorded_steps"], "waypoints_here": len(positions)})

    check = pandas.DataFrame(check_rows)
    plain = check[check["hand_corrections"] == 0]
    difference = plain["waypoints_here"] - plain["recorded_steps"]
    report(f"guided paths read: {len(check)} episodes, {guided_points} guided points")
    report(f"check of the thinning rule on the {len(plain)} episodes without a hand correction: waypoints computed here minus the steps recorded "
           f"(steps.csv has one row per waypoint and a stop row): median {difference.median():.0f}, from {difference.min():.0f} to {difference.max():.0f}; "
           f"exactly one less in {float((difference == -1).mean()):.2f} of them")

    rows = []
    for name, parts in collected.items():
        labels = pandas.concat(parts, ignore_index = True)
        is_same_episode = (labels["episode_id"] == labels["episode_id"].shift(-1)).to_numpy()
        jump = common.angle_difference_deg(labels["table_deg"].to_numpy(), labels["table_deg"].shift(-1).fillna(0.0).to_numpy())
        is_real = (labels["step_cm"] >= settings["short_step_cm"]).to_numpy()
        is_pair_real = is_same_episode & is_real & numpy.r_[is_real[1:], False]
        all_deviation = numpy.concatenate(deviation[name])
        all_height = numpy.concatenate(height[name])
        hairpin_sheet = labels[labels["sheet"] == "path_04"]
        rows.append({
            "variant": name,
            "steps": int(len(labels)),
            "steps against as recorded": round(len(labels) / max(1, len(pandas.concat(collected[list(collected)[0]]))), 2),
            "step median cm": round(float(labels["step_cm"].median()), 2),
            "share under 0.3 cm": round(float((labels["step_cm"] < settings["short_step_cm"]).mean()), 3),
            "share twists in place": round(float(((labels["step_cm"] < settings["short_step_cm"]) & (labels["twist_deg"].abs() >= settings["twist_in_place_deg"])).mean()), 3),
            "twist median abs deg": round(float(labels["twist_deg"].abs().median()), 2),
            "twist 99th percentile abs deg": round(float(labels["twist_deg"].abs().quantile(0.99)), 1),
            "jumps over 90 deg, all steps": round(float((jump[is_same_episode] > 90).mean()), 3),
            "jumps over 90 deg, real moves": round(float((jump[is_pair_real] > 90).mean()), 3),
            "jump median, real moves deg": round(float(numpy.median(jump[is_pair_real])), 1),
            "sheet 04: share under 0.3 cm": round(float((hairpin_sheet["step_cm"] < settings["short_step_cm"]).mean()), 3),
            "tip off the hand's path: median mm": round(float(numpy.median(all_deviation)), 2),
            "tip off the hand's path: 95th percentile mm": round(float(numpy.percentile(all_deviation, 95)), 2),
            "tip off the hand's path: largest mm": round(float(all_deviation.max()), 1),
            "tip height change: lowest mm": round(float(all_height.min()), 2),
            "tip height change: highest mm": round(float(all_height.max()), 2)
        })
    table = pandas.DataFrame(rows)
    table.to_csv(common.ANALYSIS_FOLDER / "s07_smoothing_variants.csv", index = False)
    report("")
    report("== the labels each variant would give (calculated from the recorded guided paths and the arm model; no robot, no pictures) ==")
    report(table.set_index("variant").T.to_string())



if __name__ == "__main__":
    main()
