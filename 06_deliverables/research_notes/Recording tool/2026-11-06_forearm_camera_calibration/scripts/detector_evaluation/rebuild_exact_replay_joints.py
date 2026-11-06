"""
Rebuild the joints of each frame of an exact hand-guided replay from its guided path.

Episode 041 has no steps.csv (the tool crashed before writing it); its labels were
recovered from the terminal log, without joints. The exact replay visits waypoints
picked from the guided path by a fixed rule (record_episodes.py, ``__exact_waypoints``:
the first guided point, then every point where the magnet tip has moved 1 cm or any
joint has turned 5 degrees since the last waypoint, and the last point), and takes
each frame before moving on: frame k shows the arm at waypoint k - 1, frame 0 at the
start pose. The rule is checked on episode 043, which has a steps.csv, before it is
applied to 041, where the rebuilt rod positions are compared with the positions the
terminal log printed.
"""

import os
import sys
import csv
import math
import json

import numpy

sys.path.insert(0, "D:/AI_Project/02_code")
from side_view_test import magnet_tip_and_lean, ned2_fk


RAW_DATA_FOLDER = "D:/AI_Project/03_data/raw"
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))
WAYPOINT_RULE = {"exact_waypoint_m": 0.01, "exact_waypoint_joint_deg": 5.0}


def read_csv_rows(path: str) -> list:

    with open(path, newline = "", encoding = "utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def exact_waypoints(guided_path: numpy.ndarray, rule: dict = WAYPOINT_RULE) -> list:
    """
    Pick the waypoints of an exact replay from a guided path, as the recording tool does.

    Parameters
    ----------
    guided_path : numpy.ndarray
        Rows of tip x, tip y, joints 1 to 6.
    rule : dict, optional
        ``exact_waypoint_m`` and ``exact_waypoint_joint_deg``.

    Returns
    -------
    list of numpy.ndarray
        The joints of each waypoint.
    """

    # Reference
    # Source: the team's own recording tool, RobotRecorder.__exact_waypoints (same rule, copied).
    # Link: D:/AI_Project/02_code/record_episodes.py
    waypoints = [guided_path[0, 2:8]]
    last = guided_path[0]
    for point in guided_path[1:]:
        tip_moved = float(numpy.hypot(point[0] - last[0], point[1] - last[1]))
        joint_turned = float(numpy.max(numpy.abs(point[2:8] - last[2:8])))
        if tip_moved >= rule["exact_waypoint_m"] or joint_turned >= math.radians(rule["exact_waypoint_joint_deg"]):
            waypoints.append(point[2:8])
            last = point
    if last is not guided_path[-1]:
        waypoints.append(guided_path[-1, 2:8])
    return waypoints


def load_guided_path(episode_folder: str) -> numpy.ndarray:

    rows = read_csv_rows(os.path.join(episode_folder, "guided_path.csv"))
    return numpy.array([[float(row["x"]), float(row["y"])] + [float(row[f"joint_{index}"]) for index in range(1, 7)]
                        for row in rows])


def check_on_episode_043() -> dict:
    """Compare the rebuilt joints with the logged joints of episode 043 and return the differences."""

    episode_folder = f"{RAW_DATA_FOLDER}/2026-11-05_session_01_emranul/path_04_meander_episode_043"
    waypoints = exact_waypoints(load_guided_path(episode_folder))
    steps = read_csv_rows(os.path.join(episode_folder, "steps.csv"))
    differences = []
    for row in steps[1:]:
        frame_index = int(row["step"])
        logged = numpy.array([float(row[f"joint_{index}"]) for index in range(1, 7)])
        if frame_index - 1 < len(waypoints):
            differences.append(float(numpy.max(numpy.abs(logged - waypoints[frame_index - 1]))))
    differences = numpy.degrees(numpy.array(differences))
    start_difference = numpy.degrees(numpy.max(numpy.abs(
        numpy.array([float(steps[0][f"joint_{index}"]) for index in range(1, 7)]) - waypoints[0])))
    summary = {
        "waypoints": len(waypoints),
        "steps": len(steps),
        "frames_compared": len(differences),
        "largest_joint_difference_deg_median": round(float(numpy.median(differences)), 3),
        "largest_joint_difference_deg_95_percent": round(float(numpy.percentile(differences, 95)), 3),
        "largest_joint_difference_deg_max": round(float(differences.max()), 3),
        "frame_0_against_first_waypoint_deg": round(float(start_difference), 3)
    }
    print("episode 043:", summary)
    return summary


def rebuild_episode_041() -> dict:
    """Rebuild the joints of every frame of episode 041 and compare rod positions with the terminal log."""

    episode_folder = f"{RAW_DATA_FOLDER}/2026-11-04_session_01_emranul/path_04_meander_episode_041"
    guided_path = load_guided_path(episode_folder)
    waypoints = exact_waypoints(guided_path)
    log_rows = read_csv_rows(os.path.join(episode_folder, "steps_from_terminal_log.csv"))
    frames = {}
    position_errors_cm = []
    for row in log_rows:
        frame_index = int(row["step"])
        # frame 0 is taken at the start pose; the first guided point is the closest record of it
        joints = waypoints[0] if frame_index == 0 else waypoints[min(frame_index - 1, len(waypoints) - 1)]
        rod_end = numpy.array(ned2_fk.fk(list(joints))["tool"][0])
        tip, lean_deg = magnet_tip_and_lean(list(joints))
        logged_xy = numpy.array([float(row["x_cm"]), float(row["y_cm"])])
        error_cm = float(numpy.hypot(*(100 * rod_end[:2] - logged_xy)))
        position_errors_cm.append(error_cm)
        frames[frame_index] = {
            "joints": [round(float(value), 5) for value in joints],
            "waypoint_index": 0 if frame_index == 0 else frame_index - 1,
            "rod_end_xy_cm_rebuilt": [round(100 * float(rod_end[0]), 2), round(100 * float(rod_end[1]), 2)],
            "rod_end_xy_cm_terminal_log": [float(row["x_cm"]), float(row["y_cm"])],
            "forearm_turn_deg_terminal_log": float(row["forearm_turn_deg"]),
            "forearm_turn_deg_rebuilt": round(math.degrees(float(joints[3])), 1)
        }
    errors = numpy.array(position_errors_cm)
    turn_errors = numpy.array([abs(frame["forearm_turn_deg_terminal_log"] - frame["forearm_turn_deg_rebuilt"])
                               for frame in frames.values()])
    summary = {
        "guided_points": len(guided_path),
        "waypoints": len(waypoints),
        "frames": len(log_rows),
        "rod_position_error_cm_median": round(float(numpy.median(errors)), 3),
        "rod_position_error_cm_max": round(float(errors.max()), 3),
        "forearm_turn_error_deg_max": round(float(turn_errors.max()), 2)
    }
    print("episode 041:", summary)
    return {"summary": summary, "frames": frames}


def main():

    check = check_on_episode_043()
    rebuilt = rebuild_episode_041()
    with open(os.path.join(OUTPUT_FOLDER, "rebuilt_joints_041.json"), "w") as output_file:
        json.dump({"check_on_043": check, "episode_041": rebuilt}, output_file, indent = 1)


if __name__ == "__main__":
    main()
