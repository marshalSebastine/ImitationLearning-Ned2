"""
Record demonstration episodes: one fresh picture and one 1 cm step at a time.

Every step runs the same loop the trained model will later run: wait until the move has finished, take the first
fresh picture, ask a decider for an action, log the picture with the action and the robot's state, then move the
pusher's tip 1 cm in that direction, measured from the pusher's narrow face (config.action_step_to_table). An episode ends
with an explicit "stop" step.

Two modes:
    keyboard     the decider is a person pressing keys while looking at the picture on screen only;
    hand_guided  a person first guides the arm by hand (motors off) and pushes the puck to the goal while the tool
                 logs the tip's path; the person then puts the puck back, and the robot replays that path in 1 cm
                 steps, snapped back onto the path at every step. Each replay step runs only when Enter is pressed;
                 a direction key overrides it, and the episode can be discarded if the puck does not follow. Only
                 the replay becomes training data.

Keys (RECORDING_PARAMETERS in config): q w e = forward-left, forward, forward-right; a s d = left, back, right; z c = back-left, back-right; r f = step over the puck forward, back;
Space = stop; x = discard the episode; u = the previous step was a mistake (kept in the file, flagged); l = the line
is lost in this picture (the next step is flagged); Enter = accept a proposed step, or go on; Esc = quit. After each
episode the tool asks whether the puck reached the goal, or why the episode was discarded. Keys pressed
while the robot moves are thrown away, so a key always refers to the picture on screen.

Output, one folder per episode, never edited afterwards:
    03_data/raw/<yyyy-mm-dd>_session_NN_<operator>/session_notes.md
    03_data/raw/<yyyy-mm-dd>_session_NN_<operator>/<path_id>_episode_NNN/frames/NNN.jpg, steps.csv, episode.json
    (hand_guided also guided_path.csv)

Run from 02_code, robot connected, nothing else connected to it, stop button in reach:
    python record_episodes.py --operator marshal --mode keyboard
Without the robot (simulated sheet, puck and camera):
    python record_episodes.py --operator marshal --mode keyboard --simulated
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy

from config import (PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS, ROBOT_PARAMETERS, SHEET_PARAMETERS, WORKSPACE_PARAMETERS,
                    action_step_to_table, camera_heading_rad, pushing_height, step_budget_for)
from side_view_test import magnet_tip_and_lean, ned2_fk, solve_least_lean

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
WINDOW_NAME = "record_episodes"
ORIENTATION_WINDOW_NAME = "orientation"
# Three times per step: when the move returned, when the logged picture was read, when the action was chosen.
# is_mistake is set afterwards by the "u" key on the step it refers to; is_lost by the "l" key on the step whose
# picture shows no line. Training drops flagged steps (and the steps that undo a mistake); the episode survives.
STEP_COLUMNS = ["step", "unix_time", "move_return_time", "capture_time", "decision_time", "frame", "frame_md5", "sharpness",
                "action", "source", "is_mistake", "is_lost", "forearm_turn_deg", "picture_heading_deg", "x", "y", "z", "roll", "pitch", "yaw",
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


def read_window_key(wait_ms: int) -> Optional[str]:
    """
    Read one key from the recording window.

    Parameters
    ----------
    wait_ms : int
        How long to wait in milliseconds; 0 waits until a key is pressed.

    Returns
    -------
    str or None
        The key as a one-character string, "enter" or "escape", or None when no key came.
    """

    # Wait in half-second slices so the window can be checked between them: three times on 4.11.26 the picture
    # window vanished (hidden or closed) while the tool waited for a key, and the keys then went nowhere.
    deadline = None if wait_ms == 0 else time.monotonic() + wait_ms / 1000.0
    while True:
        slice_ms = 500 if deadline is None else max(1, min(500, int(1000 * (deadline - time.monotonic()))))
        code = cv2.waitKey(slice_ms)
        if code >= 0:
            break
        if deadline is not None and time.monotonic() >= deadline:
            return None
        ensure_window_is_shown()

    code = code & 0xFF
    if code in (10, 13):
        return "enter"
    if code == 27:
        return "escape"
    return chr(code).lower()


def forearm_picture_up_in_forearm_frame() -> numpy.ndarray:
    """
    The forearm camera's picture-up axis in the forearm's own frame (joint 4's frame of the arm model): at a reference
    pose with the tip 25 cm out and the forearm untwisted (it then leans slightly outwards), the part of straight up
    at right angles to the forearm, which points up and away from the base.
    """

    reference_joints = [0.0, -0.809, -0.588, 0.0, -0.173, 0.0]
    rotation = numpy.array(ned2_fk.fk(reference_joints)["joint_4"][1])
    forearm = rotation[:, 2]
    straight_up = numpy.array([0.0, 0.0, 1.0])
    picture_up = straight_up - (straight_up @ forearm) * forearm
    return rotation.T @ (picture_up / numpy.linalg.norm(picture_up))


FOREARM_PICTURE_UP_IN_FOREARM_FRAME = forearm_picture_up_in_forearm_frame()


def forearm_camera_heading_rad(
        joints: List[float],
        turns_with_joint_4: bool = RECORDING_PARAMETERS["forearm_camera_turns_with_joint_4"]
) -> float:
    """
    The table direction the top of the picture points to, for a camera on the forearm cover looking down the forearm.

    The camera is fixed to the forearm, so its picture-up axis is a fixed direction in the forearm's own frame: the
    one that points up and away from the base at a reference pose (tip 25 cm out, forearm untwisted, leaning slightly
    outwards). The first version (4.11.26 evening) worked it out as "the part of straight up at right angles to the
    forearm" at every pose, which flips by 180 degrees when the forearm passes upright: closer than about 21 cm to the
    base the forearm leans back towards the robot, and w went backwards at the start of sheet 04 (Emranul). Without
    the twist (``turns_with_joint_4`` False) the forearm frame is taken untwisted. A table point moves straight up the
    picture when it moves at right angles to the picture's right axis, so the heading is the table direction at right
    angles to that axis, on the picture-up side; with the forearm untwisted it is the direction from the base
    towards the rod.
    """

    if turns_with_joint_4:
        rotation = numpy.array(ned2_fk.fk(list(joints))["joint_4"][1])
    else:
        untwisted = list(joints)
        untwisted[3] = 0.0
        rotation = numpy.array(ned2_fk.fk(untwisted)["joint_4"][1])
    looking = rotation[:, 2]
    picture_up = rotation @ FOREARM_PICTURE_UP_IN_FOREARM_FRAME
    straight_up = numpy.array([0.0, 0.0, 1.0])
    picture_right = numpy.cross(looking, picture_up)
    top = numpy.cross(straight_up, numpy.array([picture_right[0], picture_right[1], 0.0]))
    if numpy.linalg.norm(top) < 1e-9:
        top = picture_up
    if top @ picture_up < -1e-9 or (abs(top @ picture_up) <= 1e-9 and top @ looking < 0.0):
        top = -top
    return math.atan2(top[1], top[0]) + RECORDING_PARAMETERS["picture_top_offset_rad"]


def picture_heading_rad(
        state: Dict[str, List[float]],
        recorder_parameters: Dict
) -> float:
    """The table direction the top of the picture points to now, for the configured camera position."""

    if recorder_parameters["camera_heading_model"] == "forearm":
        return forearm_camera_heading_rad(state["joints"], recorder_parameters["forearm_camera_turns_with_joint_4"])
    return camera_heading_rad(state["pose"], "hand")


LAST_SHOWN = {"picture": None}


def show_in_window(picture: numpy.ndarray) -> None:
    """Show ``picture`` in the recording window, bring the window to the front, and remember it for re-showing."""

    LAST_SHOWN["picture"] = picture
    cv2.imshow(WINDOW_NAME, picture)
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass


def ensure_window_is_shown() -> None:
    """Re-open the recording window with the last picture if it was closed or is no longer visible."""

    if LAST_SHOWN["picture"] is None:
        return
    try:
        is_visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        is_visible = False
    if not is_visible:
        print("the picture window had gone; opened it again")
        show_in_window(LAST_SHOWN["picture"])


def sharpness_of(picture: numpy.ndarray) -> float:
    """Variance of the Laplacian, the same number view_camera.py shows: higher is sharper (about 100 to 150 when in focus)."""

    return float(cv2.Laplacian(cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def rotation_from_roll_pitch_yaw(
        roll: float,
        pitch: float,
        yaw: float
) -> numpy.ndarray:
    """The rotation matrix of pyniryo's pose angles: extrinsic rotations about x (roll), then y (pitch), then z (yaw)."""

    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rotation_x = numpy.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
    rotation_y = numpy.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    rotation_z = numpy.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    return rotation_z @ rotation_y @ rotation_x


def roll_pitch_yaw_from_rotation(rotation: numpy.ndarray) -> Tuple[float, float, float]:
    """The pose angles (roll, pitch, yaw) of a rotation matrix, the inverse of rotation_from_roll_pitch_yaw."""

    pitch = math.asin(max(-1.0, min(1.0, -rotation[2, 0])))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def rotation_about_axis(
        axis: numpy.ndarray,
        angle: float
) -> numpy.ndarray:
    """Rodrigues' rotation matrix for ``angle`` radians about the unit vector ``axis``."""

    axis = axis / numpy.linalg.norm(axis)
    cross = numpy.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return numpy.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def hand_orientation(
        turn_deg: float,
        picture_left_sign: int = RECORDING_PARAMETERS["picture_left_sign"]
) -> List[float]:
    """
    The orientation [roll, pitch, yaw] of the vertical rod with the hand turned by ``turn_deg`` about the vertical.

    A positive turn carries the picture's top towards picture-left of the untouched hand (``picture_left_sign`` sets
    which way round that is on the table). The untouched hand is [0, tool_pitch_rad, 0], camera looking along +x.
    """

    untouched = rotation_from_roll_pitch_yaw(0.0, WORKSPACE_PARAMETERS["tool_pitch_rad"], 0.0)
    turned = rotation_about_axis(numpy.array([0.0, 0.0, 1.0]), picture_left_sign * math.radians(turn_deg)) @ untouched
    return list(roll_pitch_yaw_from_rotation(turned))


def drain_window_keys(most_keys: int = 50) -> int:
    """
    Throw away keys pressed while the robot was moving, so that the next key read refers to the picture on screen.

    Returns
    -------
    int
        How many keys were thrown away; a double press during a move shows up here.
    """

    drained = 0
    while drained < most_keys and cv2.waitKey(1) >= 0:
        drained += 1
    return drained



class EpisodeRecorder:

    def __init__(
            self,
            robot_link,
            operator: str,
            path_id: str = SHEET_PARAMETERS["path_id"],
            recorder_parameters: Optional[dict] = None,
            key_reader: Callable[[int], Optional[str]] = read_window_key,
            key_drainer: Optional[Callable[[], int]] = None
    ):
        """
        Prepare a recording session on an open robot link.

        Parameters
        ----------
        robot_link : RobotLink or SimulatedRobotLink
            The robot to drive.
        operator : str
            Who records, written into every episode (the course grades each student's share).
        path_id : str
            The sheet's path_id, e.g. "path_01_straight_line"; it starts every episode's name.
        recorder_parameters : dict, optional
            Overrides merged over RECORDING_PARAMETERS, plus "start_point_xy" (the puck's start), "is_new_session",
            "is_window_shown", "raw_folder", "is_simulated", "step_budget" (default: step_budget_for(path_id)) and the
            episode metadata "camera_mount", "mapping_version", "sheet_orientation", "start_index",
            "deliberate_offset" and "lighting_note".
        key_reader : callable, optional
            Returns the next key; replaced in tests by a scripted sequence.
        key_drainer : callable, optional
            Throws away keys pressed during a move. Defaults to the window's drain when ``key_reader`` reads the
            window, and to nothing otherwise (a scripted sequence must not be eaten).
        """

        self.robot_link = robot_link
        self.operator = operator
        self.path_id = path_id
        self.key_reader = key_reader
        if key_drainer is None:
            key_drainer = drain_window_keys if key_reader is read_window_key else (lambda: 0)
        self.key_drainer = key_drainer

        self.__initialize_variables()
        self.update_recorder_parameters(recorder_parameters)
        self.session_folder = self.__open_session_folder()


    def __initialize_variables(self):

        self.recorder_parameters = dict(RECORDING_PARAMETERS)
        self.recorder_parameters.update({
            "start_point_xy": WORKSPACE_PARAMETERS["line_start_xy"],
            "step_budget": step_budget_for(self.path_id),
            "is_new_session": False,
            "is_window_shown": True,
            "is_simulated": False,
            "sheet_orientation": "",
            "start_index": 0,
            "deliberate_offset": {"lateral_cm": 0.0, "angle_deg": 0.0},
            "lighting_note": "",
            "laptop": platform.node()
        })
        self.last_picture_md5 = None
        self.is_free_motion = False
        self.forearm_turn_deg = 0.0
        self.tip_xy = None
        self.is_arm_model_checked = False
        self.is_lost_pending = False
        self.current_rows = []


    def update_recorder_parameters(self, recorder_parameters: Optional[dict]):

        if recorder_parameters:
            self.recorder_parameters.update(recorder_parameters)


    def record_session(self, mode: str) -> List[Dict]:
        """
        Record episodes in ``mode`` ("keyboard" or "hand_guided") until the operator quits; returns their summaries.
        """

        summaries = []
        print(f"camera mount: {self.recorder_parameters['camera_mount']}, stepping rule: {self.recorder_parameters['stepping_rule_version']}")
        self.__append_session_notes(f"\n## {time.strftime('%H:%M')}: {self.operator}, {mode}, {self.path_id}"
                                    f"{', simulated robot' if self.recorder_parameters['is_simulated'] else ''}\n\n")
        try:
            self.robot_link.move_home(self.recorder_parameters["home_joints"])
            while True:
                self.__show_message(f"Put the puck at the start of the line. Enter = next {mode} episode, Esc = finish.")
                if self.__wait_for(["enter", "escape"]) == "escape":
                    break
                self.__simulate_puck_reset()

                summary = self.__record_episode(mode)
                summaries.append(summary)
                if summary["status"] == "quit":
                    break
        finally:
            # Nothing may move while a person holds a torque-free arm: if the motors are off (an exception during
            # guiding, Ctrl-C included), they stay off until the person confirms that the arm is released.
            if self.is_free_motion:
                self.__release_arm_then_motors_on()
            self.robot_link.move_home(self.recorder_parameters["home_joints"])
            self.robot_link.close()
            if self.recorder_parameters["is_window_shown"]:
                cv2.destroyAllWindows()

        return summaries


    def __record_episode(self, mode: str) -> Dict:
        """Record one episode from the start pose and write its files; returns its summary."""

        episode_folder = self.__next_episode_folder()
        (episode_folder / "frames").mkdir(parents = True)
        summary = {
            "path_id": self.path_id,
            "episode": episode_folder.name,
            "mode": mode,
            "operator": self.operator,
            "started": datetime.datetime.now().isoformat(timespec = "seconds"),
            "is_simulated": self.recorder_parameters["is_simulated"],
            "code_commit": self.__code_commit(),
            "laptop": self.recorder_parameters["laptop"],
            "camera_mount": self.recorder_parameters["camera_mount"],
            "mapping_version": self.recorder_parameters["mapping_version"],
            "stepping_rule_version": self.recorder_parameters["stepping_rule_version"],
            "sheet_orientation": self.recorder_parameters["sheet_orientation"],
            "start_index": self.recorder_parameters["start_index"],
            "deliberate_offset": self.recorder_parameters["deliberate_offset"],
            "lighting_note": self.recorder_parameters["lighting_note"],
            "start_pose": self.__start_pose(),
            "step_m": self.recorder_parameters["step_m"],
            "step_budget": self.recorder_parameters["step_budget"],
            "max_turn_lean_deg": self.recorder_parameters["max_turn_lean_deg"],
            "actions": self.recorder_parameters["actions"],
            "turn_actions": self.recorder_parameters["turn_actions"],
            "step_over_actions": self.recorder_parameters["step_over_actions"],
            "step_over_m": self.recorder_parameters["step_over_m"],
            "picture_left_sign": self.recorder_parameters["picture_left_sign"]
        }

        try:
            self.__lower_to_start()
        except Exception as error:
            # Seen on 31.10.26: a start 12 cm from the base axis cannot be reached with the rod vertical. Say so, lift
            # what was reached and go back to the prompt instead of crashing with the robot low over the sheet.
            print(f"cannot reach the start pose ({error}). Move the sheet so the line start is at least about 16 cm "
                  f"from the base, teach it again with teach_sheet_start.py, and restart with the new --start-x and --start-y.")
            summary["status"] = "start_unreachable"
            self.__lift()
            return self.__finish_episode(episode_folder, summary, [])

        if mode == "hand_guided":
            guided_path = self.__record_guided_path(episode_folder)
            if guided_path is None:
                summary["status"] = "discarded_while_guiding"
                self.__lift()
                self.__ask_how_episode_ended(summary)
                return self.__finish_episode(episode_folder, summary, [])
            self.__lift()
            self.__show_message("Motors on. Put the puck back at the start of the line, then press Enter.")
            self.__wait_for(["enter"])
            self.__simulate_puck_reset()
            self.__lower_to_start()
            decider = self.__make_replay_decider(guided_path)
        else:
            decider = self.__keyboard_decider

        status, rows = self.__run_steps(episode_folder, decider)
        summary["status"] = status
        self.__lift()
        self.__ask_how_episode_ended(summary)
        return self.__finish_episode(episode_folder, summary, rows)


    def __ask_how_episode_ended(self, summary: Dict) -> None:
        """Store whether the puck reached the goal, or why the episode was discarded; a quit episode asks nothing."""

        summary.update({"outcome": None, "discard_reason": None})
        if summary["status"] in ("completed", "step_budget_reached"):
            summary["outcome"] = self.__ask_choice("Puck in the goal? y = yes, n = no", self.recorder_parameters["outcome_keys"])
        elif summary["status"] in ("discarded", "discarded_while_guiding"):
            summary["discard_reason"] = self.__ask_choice("Why discard? t tipped, o off sheet, r robot, h hand, m other",
                                                          self.recorder_parameters["discard_reason_keys"])


    def __ask_choice(
            self,
            message: str,
            choices: Dict[str, str]
    ) -> str:
        """Show ``message`` and return the meaning of the key pressed; Esc returns "not_recorded"."""

        self.key_drainer()
        self.__show_message(message)
        return choices.get(self.__wait_for(list(choices) + ["escape"]), "not_recorded")


    def __run_steps(
            self,
            episode_folder: Path,
            decider: Callable[[bytes, Dict, int], Tuple[str, str]]
    ) -> Tuple[str, List[Dict]]:
        """
        The step loop shared by every mode: fresh picture, decision, log, 1 cm move.

        Returns
        -------
        tuple of str and list of dict
            The episode's status ("completed", "discarded", "quit" or "step_budget_reached") and the logged steps.
        """

        rows = []
        self.current_rows = rows
        self.is_lost_pending = False
        self.forearm_turn_deg = 0.0
        self.tip_xy = None
        self.is_arm_model_checked = False
        # Where the puck is believed to be, for the landing after a forearm turn: at the start point, then moved by
        # every step that reaches it (see __track_puck). Since 4.11.26 after test 3; a running count of how far the
        # block stood back drifted to 10 cm when sideways and back steps were taken in a turned picture.
        self.puck_xy_estimate = numpy.array(self.recorder_parameters["start_point_xy"], dtype = float)
        move_return_time = time.time()
        while not self.recorder_parameters["is_step_budget_enforced"] or len(rows) < self.recorder_parameters["step_budget"]:
            picture_bytes = self.__fresh_picture()
            capture_time = time.time()
            state = self.robot_link.read_state()
            action, source = decider(picture_bytes, state, len(rows))
            decision_time = time.time()

            if action in ("discard", "quit"):
                return ("discarded" if action == "discard" else "quit"), rows

            if action in self.recorder_parameters["turn_actions"]:
                least, most = self.recorder_parameters["min_turn_deg"], self.recorder_parameters["max_turn_deg"]
                turn_deg = max(least, min(most, self.forearm_turn_deg + self.recorder_parameters["turn_actions"][action]))
                if turn_deg == self.forearm_turn_deg:
                    print(f"refused: the forearm cannot turn further that way (allowed {least} to {most} degrees)")
                    continue
                # The tip now: tracked while the forearm is turned (the rod leans), the rod's end when it is straight
                # (rod upright above the tip). Test 1 of 4.11.26 (episode 033, step 72) landed 14.9 cm away because a
                # tip remembered from an earlier turn back to straight was used after 33 ordinary steps.
                tip_xy = list(self.tip_xy) if self.forearm_turn_deg and self.tip_xy is not None else list(state["pose"][:2])
                # Land back from the puck, straight away from it, only by what the block does not already stand back
                # (4.11.26: two turns in a row put the tip 5 cm away; a turn after backing off moved it further still).
                away_from_puck = numpy.array(tip_xy) - self.puck_xy_estimate
                distance_to_puck = float(numpy.linalg.norm(away_from_puck))
                clearance_m = max(0.0, distance_to_puck - self.recorder_parameters["puck_centre_ahead_of_tip_m"])
                retreat_m = max(0.0, self.recorder_parameters["turn_landing_retreat_m"] - clearance_m)
                retreat_direction = away_from_puck / distance_to_puck if distance_to_puck > 1e-9 else numpy.array([-1.0, 0.0])
                landing_xy = list(numpy.array(tip_xy) + retreat_m * retreat_direction)
                if not self.__is_inside_workspace(landing_xy[0], landing_xy[1]):
                    print("refused: the landing point behind the puck would be outside the workspace")
                    continue
                try:
                    plan = self.__turn_plan(self.forearm_turn_deg, turn_deg, tip_xy, landing_xy, state["joints"])
                    self.__check_arm_model(plan[-1])
                except Exception as error:
                    print(f"refused: {error}; nothing moved")
                    continue
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
                try:
                    for joints in plan:
                        self.robot_link.move_joints(joints)
                except Exception as error:
                    print(f"the robot refused that turn ({error}); choose another step")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    continue
                self.forearm_turn_deg = turn_deg
                self.tip_xy = landing_xy if turn_deg else None
                print(f"forearm turned {turn_deg:+.0f} degrees, rod leaning {magnet_tip_and_lean(plan[-1])[1]:.1f} degrees, tip moved "
                      f"{100 * retreat_m:.1f} cm back from the puck ({100 * (clearance_m + retreat_m):.1f} cm clear); the picture's top now points "
                      f"{math.degrees(picture_heading_rad(self.robot_link.read_state(), self.recorder_parameters)):+.0f} degrees from the table's x axis")
                move_return_time = time.time()
                continue

            if action in self.recorder_parameters["flick_actions"]:
                # q / e (4.11.26 evening): a fast turn of the hand about the rod and straight back; the rod stays put.
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
                counter_clockwise_rad = self.recorder_parameters["picture_left_sign"] * math.radians(self.recorder_parameters["flick_actions"][action])
                try:
                    # Joint 6 turns the hand clockwise seen from above as it grows (with the rod pointing down).
                    self.robot_link.turn_hand_and_back(-counter_clockwise_rad, self.recorder_parameters["flick_velocity_percent"])
                except Exception as error:
                    print(f"the robot refused that flick ({error}); choose another step")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    continue
                move_return_time = time.time()
                continue

            if action in self.recorder_parameters["lean_actions"]:
                target_pose = self.__leaned_pose(state["pose"], self.recorder_parameters["lean_actions"][action])
                if target_pose is None:
                    print(f"refused: already leaning {self.recorder_parameters['max_lean_deg']} degrees that way")
                    continue
                times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
                rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
                try:
                    self.robot_link.move_to(target_pose, is_linear = False)
                except Exception as error:
                    print(f"the robot refused that lean ({error}); choose another step")
                    rows.pop()
                    (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                    self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                    continue
                move_return_time = time.time()
                continue

            if action != "stop":
                # While the forearm is turned the rod leans, so the commanded magnet tip is tracked instead of the
                # rod's end. The keys are measured in the PICTURE's frame, read from the camera's current direction
                # (Emranul, 4.11.26: he steers from the picture, so "forward" must be the top of the picture also
                # when the forearm is turned; the model sees nothing but the picture either).
                x, y = self.tip_xy if self.forearm_turn_deg else state["pose"][:2]
                step_x, step_y = action_step_to_table(action, state["pose"], self.recorder_parameters["hand_turn_deg"],
                                                      heading_rad = picture_heading_rad(state, self.recorder_parameters))
                target_x, target_y = x + step_x, y + step_y
                # A step-over leaves the puck where it is, so only the rod's landing point needs checking.
                is_allowed = (self.__is_inside_workspace(target_x, target_y) if self.__is_step_over(action)
                              else self.__is_step_allowed(target_x, target_y, step_x, step_y))
                if not is_allowed:
                    print(f"refused: {action} would carry the rod or the puck out of the workspace; choose another step")
                    continue

            times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
            rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
            if action == "stop":
                return "completed", rows

            # The rod is asked to be vertical at every step (the start pose's orientation); if the bracket's wrist
            # limit forbids that near the base, RobotLink leans it just enough and says so (2.11.26). A lean from the
            # n / m test keys therefore lasts one step only. With the forearm turned, the step is solved as joints
            # (least lean, magnet tip on the target) and the turn lasts.
            target_pose = [target_x, target_y, pushing_height(target_x, target_y)] + self.__turned_orientation(self.recorder_parameters["hand_turn_deg"])
            try:
                if self.forearm_turn_deg:
                    self.robot_link.move_joints(self.__turned_joints([target_x, target_y], self.forearm_turn_deg, 0.0))
                    self.tip_xy = [target_x, target_y]
                elif self.__is_step_over(action):
                    self.__step_over(target_pose)
                else:
                    self.robot_link.move_to(target_pose)
            except Exception as error:
                # The robot refused the step (seen on 31.10.26 near the base: "cannot be reached with a linear
                # trajectory"). The step did not happen, so its row and picture are taken back and the person
                # chooses another step from the same picture; the episode goes on.
                print(f"the robot refused that step ({error}); choose another step")
                if self.forearm_turn_deg and "lean" in str(error):
                    print(self.__turn_hint([target_x, target_y]))
                rows.pop()
                (episode_folder / "frames" / f"{len(rows):03d}.jpg").unlink(missing_ok = True)
                self.last_picture_md5 = rows[-1]["frame_md5"] if rows else None
                continue
            self.__track_puck(action, [target_x, target_y], step_x, step_y)
            move_return_time = time.time()

        return "step_budget_reached", rows


    def __track_puck(
            self,
            action: str,
            tip_xy: List[float],
            step_x: float,
            step_y: float
    ) -> None:
        """
        Move the believed puck position after a step that happened, for the landing after a forearm turn.

        When the tip ends closer to the believed puck centre than the contact distance (puck radius plus tip radius),
        the step has reached the puck and pushed it: the puck is put at contact distance ahead of the tip along the
        step. Otherwise the puck stays. A step-over passes above the puck and leaves it where it is.
        """

        step = numpy.array([step_x, step_y])
        length = float(numpy.linalg.norm(step))
        if length < 1e-12 or self.__is_step_over(action):
            return
        contact = self.recorder_parameters["puck_centre_ahead_of_tip_m"]
        tip = numpy.array(tip_xy, dtype = float)
        if float(numpy.linalg.norm(self.puck_xy_estimate - tip)) < contact - 1e-9:
            self.puck_xy_estimate = tip + contact * step / length


    def __turn_hint(self, target_xy: List[float]) -> str:
        """
        After a turned step was refused for the rod's lean: the largest forearm turn, in key presses back towards
        straight, at which the same step would go through (straight always does).
        """

        turn_step_deg = abs(self.recorder_parameters["turn_actions"]["turn_left"])
        back_action = "turn_right" if self.forearm_turn_deg > 0 else "turn_left"
        back_key = next((key for key, action in self.recorder_parameters["keys"].items() if action == back_action), "?")
        candidate_deg = self.forearm_turn_deg
        presses = 0
        while abs(candidate_deg) > 1e-9:
            candidate_deg -= math.copysign(min(turn_step_deg, abs(candidate_deg)), candidate_deg)
            presses += 1
            try:
                self.__turned_joints(target_xy, candidate_deg, 0.0)
                break
            except Exception:
                continue
        return (f"hint: that spot can be reached with the forearm at {candidate_deg:+.0f} degrees: press {back_key} {presses} "
                f"time{'s' if presses > 1 else ''} (the picture turns, so the keys point differently afterwards)")


    def __leaned_pose(
            self,
            pose: List[float],
            lean_deg: float
    ) -> Optional[List[float]]:
        """
        The pose after leaning rod and camera together by ``lean_deg`` about the camera's forward axis, the rod's end
        staying in place; None when the lean would exceed ``max_lean_deg``.

        The orientation is rotated as a matrix (pyniryo's roll, pitch, yaw are extrinsic x, y, z), because at the
        vertical rod's pitch of 90 degrees roll and yaw both spin the rod about its own axis and cannot express a lean.
        """

        x, y, z = pose[:3]
        heading = camera_heading_rad(pose)
        forward_axis = numpy.array([math.cos(heading), math.sin(heading), 0.0])
        current = rotation_from_roll_pitch_yaw(*pose[3:])
        leaned = rotation_about_axis(forward_axis, math.radians(lean_deg)) @ current

        # Niryo's tool frame points along its x axis (pitch 90 degrees = rod straight down), so the rod's own axis is
        # the first column; its angle from straight down is the total lean.
        rod_axis = leaned[:, 0]
        total_lean_deg = math.degrees(math.acos(max(-1.0, min(1.0, -rod_axis[2]))))
        if total_lean_deg > self.recorder_parameters["max_lean_deg"] + 0.5:
            return None

        return [x, y, z] + list(roll_pitch_yaw_from_rotation(leaned))


    def __turned_orientation(self, turn_deg: float) -> List[float]:
        """The vertical rod's orientation with the hand turned by ``turn_deg``; see hand_orientation."""

        return hand_orientation(turn_deg, self.recorder_parameters["picture_left_sign"])


    def __turned_joints(
            self,
            tip_xy: List[float],
            turn_deg: float,
            lift: float,
            is_lean_limited: bool = True,
            near_wrist_rad: Optional[float] = None
    ) -> List[float]:
        """
        The joints that put the magnet tip at ``tip_xy``, ``lift`` metres above pushing height, with the forearm
        (joint 4) turned by ``turn_deg`` and the rod leaning least (Marshal's solver, 4.11.26).

        ``near_wrist_rad`` narrows the wrist search to 8 degrees either side of a neighbouring solution (a quarter of
        the time; the full search is the fallback). With the forearm untwisted, joint 6 is set equal to joint 1, the
        hand's usual orientation for a vertical rod, so the first step after turning back does not spin the block
        against the puck. Raises when no joint angles within the limits reach the point, or when the rod would lean
        more than the limit for that height ("max_turn_lean_deg" down, "max_turn_lifted_lean_deg" lifted). The wrist
        (joint 5) is kept below the bracket's limit.
        """

        height = pushing_height(tip_xy[0], tip_xy[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"] + lift
        wrist_max_deg = min(60.0, math.degrees(self.recorder_parameters["wrist_pitch_max_rad"]))
        target = numpy.array([tip_xy[0], tip_xy[1], height])
        joints = None
        if near_wrist_rad is not None:
            near_deg = math.degrees(near_wrist_rad)
            joints = solve_least_lean(target, math.radians(turn_deg), wrist_range_deg = (max(-60.0, near_deg - 8.0), min(wrist_max_deg, near_deg + 8.0)))
        if joints is None:
            joints = solve_least_lean(target, math.radians(turn_deg), wrist_range_deg = (-60.0, wrist_max_deg))
        if joints is None:
            raise RuntimeError(f"no arm position reaches ({tip_xy[0]:.3f}, {tip_xy[1]:.3f}) with the forearm turned {turn_deg:.0f} degrees")
        if abs(turn_deg) < 1e-9:
            joints[5] = joints[0]
        lean_deg = magnet_tip_and_lean(joints)[1]
        limit = self.recorder_parameters["max_turn_lifted_lean_deg" if lift > 0.0 else "max_turn_lean_deg"]
        if is_lean_limited and lean_deg > limit:
            raise RuntimeError(f"the rod would lean {lean_deg:.1f} degrees at ({tip_xy[0]:.3f}, {tip_xy[1]:.3f}) with the forearm "
                               f"turned {turn_deg:.0f} degrees, above the limit of {limit}")
        return joints


    def __turn_plan(
            self,
            from_turn_deg: float,
            to_turn_deg: float,
            tip_xy: List[float],
            landing_xy: List[float],
            current_joints: List[float]
    ) -> List[List[float]]:
        """
        The joint positions of a forearm turn: up over ``tip_xy`` at the current turn (through the near-paper heights,
        so the tip rises nearly straight), lifted at every "turn_increment_deg" on the way, lifted over ``landing_xy``,
        then down there at the new turn through the same heights. All are solved before anything moves, so a refusal
        moves nothing.
        """

        lift = self.recorder_parameters["turn_lift_m"]
        increment = self.recorder_parameters["turn_increment_deg"]
        near_paper = sorted(height for height in self.recorder_parameters["turn_near_paper_heights_m"] if height < lift)
        stops = [(from_turn_deg, height, tip_xy) for height in near_paper] + [(from_turn_deg, lift, tip_xy)]
        turn_deg = from_turn_deg
        while abs(to_turn_deg - turn_deg) > 1e-9:
            turn_deg += math.copysign(min(increment, abs(to_turn_deg - turn_deg)), to_turn_deg - turn_deg)
            stops.append((turn_deg, lift, tip_xy))
        stops.append((to_turn_deg, lift, landing_xy))
        stops += [(to_turn_deg, height, landing_xy) for height in reversed(near_paper)]
        stops.append((to_turn_deg, 0.0, landing_xy))

        plan = []
        near_wrist_rad = current_joints[4]
        for turn_deg, height, point_xy in stops:
            joints = self.__turned_joints(point_xy, turn_deg, height, near_wrist_rad = near_wrist_rad)
            plan.append(joints)
            near_wrist_rad = joints[4]
        return plan


    def __check_arm_model(self, joints: List[float]) -> None:
        """
        Once per episode, before the first forearm turn: the arm model and the robot's own kinematics must agree on
        where ``joints`` put the rod's end, within "turn_model_tolerance_mm"; nothing moves for the check.
        """

        if self.is_arm_model_checked:
            return
        model = numpy.array(ned2_fk.fk(list(joints))["tool"][0])
        robot = numpy.array(self.robot_link.predicted_position(list(joints)))
        error_mm = 1000.0 * float(numpy.linalg.norm(model - robot))
        if error_mm > self.recorder_parameters["turn_model_tolerance_mm"]:
            raise RuntimeError(f"the arm model and the robot disagree by {error_mm:.1f} mm on the turned position")
        self.is_arm_model_checked = True


    def __turn_back_lifted(self) -> None:
        """
        Lift at the current forearm turn and turn back to straight in increments; the arm stays lifted.

        The lean limit does not apply here: lifted, the rod touches nothing, and on 4.11.26 the limit refused the
        retreat at 38 cm (lifted lean 34.6 degrees) and the arm went to the parking pose from low down instead.
        """

        lift = self.recorder_parameters["turn_lift_m"]
        increment = self.recorder_parameters["turn_increment_deg"]
        turn_deg = self.forearm_turn_deg
        stops = [(turn_deg, height) for height in sorted(self.recorder_parameters["turn_near_paper_heights_m"]) if height < lift] + [(turn_deg, lift)]
        while abs(turn_deg) > 1e-9:
            turn_deg -= math.copysign(min(increment, abs(turn_deg)), turn_deg)
            stops.append((turn_deg, lift))
        plan = []
        near_wrist_rad = self.robot_link.read_state()["joints"][4]
        for turn_deg, height in stops:
            joints = self.__turned_joints(self.tip_xy, turn_deg, height, is_lean_limited = False, near_wrist_rad = near_wrist_rad)
            plan.append(joints)
            near_wrist_rad = joints[4]
        for joints in plan:
            self.robot_link.move_joints(joints)
        self.forearm_turn_deg = 0.0
        self.tip_xy = None


    def __is_step_over(self, action: str) -> bool:

        return action in self.recorder_parameters["step_over_actions"]


    def __step_over(self, target_pose: List[float]) -> None:
        """Lift, cross above the puck and come down on its other side, so the puck is not touched (Marshal, 4.11.26)."""

        height = self.recorder_parameters["approach_height_m"]
        self.__lift()
        self.robot_link.move_to([target_pose[0], target_pose[1], target_pose[2] + height] + target_pose[3:])
        self.robot_link.move_to(target_pose)


    def __handle_flag_key(
            self,
            action: Optional[str],
            picture_bytes: bytes,
            step: int
    ) -> bool:
        """
        Apply the "mistake" and "lost" keys, which flag a step instead of choosing one; True when ``action`` was one.
        """

        if action == "mistake":
            if self.current_rows:
                self.current_rows[-1]["is_mistake"] = 1
                self.__show_picture(picture_bytes, f"step {step}: step {step - 1} flagged as a mistake. Now choose the step.")
            return True
        if action == "lost":
            self.is_lost_pending = True
            self.__show_picture(picture_bytes, f"step {step}: flagged as line lost. Now choose the step.")
            return True
        return False


    def __show_orientation(self, state: Dict) -> None:
        """
        A second window, "orientation": the table from above with the robot's base at the bottom, the sheet, the
        believed puck position, the tip, and an arrow for every step key showing where that key moves the tip now
        (Emranul, 4.11.26: after a forearm turn the new orientation was not obvious from the picture alone).
        """

        if not self.recorder_parameters["is_window_shown"]:
            return

        scale = 900.0
        width, height = 400, 440
        centre_x, bottom = width // 2, height - 20

        def pixel(point_xy) -> tuple:
            return (int(round(centre_x - point_xy[1] * scale)), int(round(bottom - point_xy[0] * scale)))

        canvas = numpy.full((height, width, 3), 245, dtype = numpy.uint8)
        cv2.circle(canvas, pixel([0.0, 0.0]), int(0.07 * scale), (200, 160, 90), -1)
        cv2.putText(canvas, "robot base", (centre_x - 40, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        corners = numpy.array([pixel(corner) for corner in WORKSPACE_PARAMETERS["sheet_corners_xy"]], dtype = numpy.int32)
        cv2.polylines(canvas, [corners], isClosed = True, color = (150, 150, 150), thickness = 1)

        tip_xy = list(self.tip_xy) if self.forearm_turn_deg and self.tip_xy is not None else list(state["pose"][:2])
        puck_xy = getattr(self, "puck_xy_estimate", None)
        if puck_xy is not None:
            cv2.circle(canvas, pixel(puck_xy), int(PUCK_PARAMETERS["diameter_m"] / 2 * scale), (60, 60, 210), -1)
        cv2.circle(canvas, pixel(tip_xy), 5, (40, 40, 40), -1)

        heading_rad = picture_heading_rad(state, self.recorder_parameters)
        key_of = {action: key for key, action in self.recorder_parameters["keys"].items()}
        for action, angle in self.recorder_parameters["actions"].items():
            if angle is None or action not in key_of:
                continue
            step_x, step_y = action_step_to_table(action, state["pose"], self.recorder_parameters["hand_turn_deg"], heading_rad = heading_rad)
            length = math.hypot(step_x, step_y)
            end = [tip_xy[0] + 0.045 * step_x / length, tip_xy[1] + 0.045 * step_y / length]
            colour = (40, 160, 40) if action == "forward" else (90, 90, 90)
            cv2.arrowedLine(canvas, pixel(tip_xy), pixel(end), colour, 2 if action == "forward" else 1, tipLength = 0.2)
            label_at = [tip_xy[0] + 0.055 * step_x / length, tip_xy[1] + 0.055 * step_y / length]
            label_pixel = pixel(label_at)
            cv2.putText(canvas, key_of[action], (label_pixel[0] - 5, label_pixel[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.putText(canvas, f"forearm turned {self.forearm_turn_deg:+.0f} deg", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f"picture top points {math.degrees(heading_rad):+.0f} deg (0 = straight out)", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        cv2.imshow(ORIENTATION_WINDOW_NAME, canvas)


    def __keyboard_decider(
            self,
            picture_bytes: bytes,
            state: Dict,
            step: int
    ) -> Tuple[str, str]:
        """The person chooses the step from the picture on screen."""

        turned = f" (forearm turned {self.forearm_turn_deg:+.0f})" if self.forearm_turn_deg else ""
        self.__show_orientation(state)
        self.__show_picture(picture_bytes, f"step {step}{turned}: q w e / a s d / z c = steps, v b = forearm turn, Space = stop, u l x Esc")
        self.__drain_keys(step)
        while True:
            key = self.key_reader(0)
            if key == "escape":
                return "quit", "human"
            action = self.recorder_parameters["keys"].get(key)
            if self.__handle_flag_key(action, picture_bytes, step):
                continue
            if self.__is_executable(action):
                return action, "human"


    def __make_replay_decider(self, guided_path: numpy.ndarray) -> Callable[[bytes, Dict, int], Tuple[str, str]]:
        """
        A decider that follows the hand-guided path, snapped back onto it at every step, and lets the person override.

        The proposed step runs only when Enter is pressed (decision of 31.10.26: a 400 ms window was shorter than a
        considered seven-way choice, so overrides landed a step late); a direction key overrides, there is no timeout.

        Parameters
        ----------
        guided_path : numpy.ndarray
            The logged tip positions, shape (points, 2), in metres.
        """

        segment_lengths = numpy.linalg.norm(numpy.diff(guided_path, axis = 0), axis = 1)
        arc_lengths = numpy.concatenate([[0.0], numpy.cumsum(segment_lengths)])
        progress = {"segment": 0}
        step_m = self.recorder_parameters["step_m"]
        moving_actions = [action for action, angle in self.recorder_parameters["actions"].items() if angle is not None]

        def replay_decider(
                picture_bytes: bytes,
                state: Dict,
                step: int
        ) -> Tuple[str, str]:

            position = numpy.array(state["pose"][:2])
            arc_now = self.__project_onto_path(position, guided_path, arc_lengths, progress)

            if arc_lengths[-1] - arc_now < step_m / 2:
                proposal = "stop"
            else:
                target = self.__point_at_arc_length(guided_path, arc_lengths, arc_now + step_m)
                proposal = min(moving_actions, key = lambda action: numpy.linalg.norm(
                    position + numpy.array(action_step_to_table(action, state["pose"], self.recorder_parameters["hand_turn_deg"],
                                                                heading_rad = picture_heading_rad(state, self.recorder_parameters))) - target))

            self.__show_orientation(state)
            self.__show_picture(picture_bytes, f"replay step {step}: {proposal}. Enter = do it, direction key = override, u = mistake, l = lost, x = discard")
            self.__drain_keys(step)
            while True:
                key = self.key_reader(0)
                if key == "escape":
                    return "quit", "human"
                action = self.recorder_parameters["keys"].get(key)
                if self.__handle_flag_key(action, picture_bytes, step):
                    continue
                if action == "accept":
                    return proposal, "replay"
                if self.__is_executable(action):
                    return action, "human"

        return replay_decider


    def __is_executable(self, action: Optional[str]) -> bool:
        """True for a key that ends the wait for a decision: a class, a step-over, a lean or turn key, or discard."""

        return (action in self.recorder_parameters["actions"]
                or action in self.recorder_parameters["flick_actions"]
                or action in self.recorder_parameters["step_over_actions"]
                or action in self.recorder_parameters["lean_actions"]
                or action in self.recorder_parameters["turn_actions"]
                or action == "discard")


    @staticmethod
    def __project_onto_path(
            position: numpy.ndarray,
            path: numpy.ndarray,
            arc_lengths: numpy.ndarray,
            progress: Dict
    ) -> float:
        """
        The arc length of the path point closest to ``position``, searching forward from the last match only, so
        that a path crossing itself (a loop) is not jumped along.
        """

        best_distance, best_arc = numpy.inf, arc_lengths[progress["segment"]]
        last_segment = min(len(path) - 1, progress["segment"] + 50)
        for segment in range(progress["segment"], last_segment):
            start, end = path[segment], path[segment + 1]
            length_squared = float((end - start) @ (end - start))
            fraction = 0.0 if length_squared == 0 else float(numpy.clip((position - start) @ (end - start) / length_squared, 0, 1))
            distance = numpy.linalg.norm(start + fraction * (end - start) - position)
            if distance < best_distance:
                best_distance = distance
                best_arc = arc_lengths[segment] + fraction * (arc_lengths[segment + 1] - arc_lengths[segment])
                progress["segment"] = segment

        return best_arc


    @staticmethod
    def __point_at_arc_length(
            path: numpy.ndarray,
            arc_lengths: numpy.ndarray,
            arc_length: float
    ) -> numpy.ndarray:
        """The point ``arc_length`` metres along the path, clipped to its end."""

        arc_length = min(arc_length, arc_lengths[-1])
        return numpy.array([numpy.interp(arc_length, arc_lengths, path[:, 0]), numpy.interp(arc_length, arc_lengths, path[:, 1])])


    def __record_guided_path(self, episode_folder: Path) -> Optional[numpy.ndarray]:
        """
        Let the person push the puck to the goal by hand while the tip's path is logged; None when discarded.
        """

        self.__show_message("Hold the arm at the elbow. Enter = motors OFF (the arm will sag), x = discard.")
        if self.__wait_for(["enter", "x"]) == "x":
            return None
        self.__set_free_motion(True)

        rows = []
        is_discarded = False
        try:
            while True:
                state = self.robot_link.read_state()
                position = state["pose"][:2]
                if not rows or numpy.hypot(position[0] - rows[-1]["x"], position[1] - rows[-1]["y"]) >= self.recorder_parameters["guided_path_sample_m"]:
                    rows.append({"unix_time": round(time.time(), 3), "x": position[0], "y": position[1], "z": state["pose"][2],
                                 "roll": state["pose"][3], "pitch": state["pose"][4], "yaw": state["pose"][5]})

                self.__show_picture(self.robot_link.read_picture_bytes(), f"MOTORS OFF, guiding: {len(rows)} points. Space = puck at goal, x = discard")
                key = self.key_reader(int(self.recorder_parameters["guided_path_poll_s"] * 1000))
                if key in (" ", "x", "escape"):
                    is_discarded = key != " "
                    break
        finally:
            # Whatever ended the guiding (Space, x, Ctrl-C, an error), the motors come back on only after the person
            # has let go and said so; the arm must never be powered against a holding hand.
            self.__release_arm_then_motors_on()

        with open(episode_folder / "guided_path.csv", "w", newline = "", encoding = "utf-8") as file:
            writer = csv.DictWriter(file, fieldnames = ["unix_time", "x", "y", "z", "roll", "pitch", "yaw"])
            writer.writeheader()
            writer.writerows(rows)

        if is_discarded or len(rows) < 2:
            return None
        return numpy.array([[row["x"], row["y"]] for row in rows])


    def __fresh_picture(self) -> bytes:
        """
        Wait for ``steps_until_fresh_picture`` new pictures after the move returned and take the last one, so no
        picture taken during the move is logged.
        """

        previous = self.robot_link.read_picture_bytes()
        new_pictures = 0
        deadline = time.monotonic() + self.recorder_parameters["fresh_picture_timeout_s"]
        while new_pictures < self.recorder_parameters["steps_until_fresh_picture"]:
            if time.monotonic() > deadline:
                raise RuntimeError("no new camera picture arrived; is the camera plugged in and streaming?")
            time.sleep(0.01)
            picture_bytes = self.robot_link.read_picture_bytes()
            if picture_bytes != previous:
                new_pictures += 1
                previous = picture_bytes

        return previous


    def __log_step(
            self,
            episode_folder: Path,
            step: int,
            picture_bytes: bytes,
            state: Dict,
            action: str,
            source: str,
            times: Dict[str, float]
    ) -> Dict:
        """Write the picture exactly as the robot sent it (no re-compression) and return the step's row."""

        frame_name = f"{step:03d}.jpg"
        (episode_folder / "frames" / frame_name).write_bytes(picture_bytes)
        picture_md5 = hashlib.md5(picture_bytes).hexdigest()
        if picture_md5 == self.last_picture_md5:
            print(f"warning: step {step} has the same picture as the step before")
        self.last_picture_md5 = picture_md5

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        row = {"step": step, "unix_time": round(time.time(), 3), "frame": f"frames/{frame_name}", "frame_md5": picture_md5,
               "sharpness": round(sharpness_of(picture), 1),
               "action": action, "source": source, "is_mistake": 0, "is_lost": int(self.is_lost_pending),
               "forearm_turn_deg": float(self.forearm_turn_deg),
               # The table direction the picture's top pointed to (degrees from the table's x axis, counter-clockwise):
               # the key's table direction is this plus the action's angle. Not a model input (the camera is the only
               # input); for checking and relabelling the data, and the controller reads it live from the robot.
               "picture_heading_deg": round(math.degrees(picture_heading_rad(state, self.recorder_parameters)), 2)}
        row.update({name: round(value, 3) for name, value in times.items()})
        row.update({name: round(value, 5) for name, value in zip(["x", "y", "z", "roll", "pitch", "yaw"], state["pose"])})
        row.update({f"joint_{index + 1}": round(value, 5) for index, value in enumerate(state["joints"])})
        self.is_lost_pending = False
        return row


    def __drain_keys(self, step: int) -> None:
        """Throw away keys pressed during the move and say so, so a double press never labels an unseen picture."""

        drained = self.key_drainer()
        if drained:
            print(f"step {step}: {drained} key press(es) made during the move were ignored")


    def __set_free_motion(self, is_enabled: bool) -> None:
        """Switch the motors off or on and remember the state, so the shutdown path knows whether a hand may be on the arm."""

        self.robot_link.set_free_motion(is_enabled)
        self.is_free_motion = is_enabled


    def __release_arm_then_motors_on(self) -> None:
        """Keep the motors off until the person confirms the arm is released, then switch them back on."""

        if not self.is_free_motion:
            return
        self.__show_message("MOTORS OFF. Let go of the arm completely, then press Enter = motors back ON.")
        self.__wait_for(["enter"])
        self.__set_free_motion(False)
        print("motors ON")


    def __finish_episode(
            self,
            episode_folder: Path,
            summary: Dict,
            rows: List[Dict]
    ) -> Dict:
        """Write steps.csv and episode.json and a line in the session notes."""

        with open(episode_folder / "steps.csv", "w", newline = "", encoding = "utf-8") as file:
            writer = csv.DictWriter(file, fieldnames = STEP_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        human_steps = sum(1 for row in rows if row["source"] == "human")
        summary.update({
            "finished": datetime.datetime.now().isoformat(timespec = "seconds"),
            "steps": len(rows),
            "human_steps": human_steps,
            "mistake_steps": sum(row["is_mistake"] for row in rows),
            "lost_steps": sum(row["is_lost"] for row in rows)
        })
        (episode_folder / "episode.json").write_text(json.dumps(summary, indent = 2), encoding = "utf-8")

        overrides = f", {human_steps} overridden by hand" if summary["mode"] == "hand_guided" else ""
        ending = summary["outcome"] or summary["discard_reason"]
        ending = f", {ending}" if ending else ""
        self.__append_session_notes(f"- {episode_folder.name}: {summary['status']}{ending}, {len(rows)} steps{overrides}\n")
        print(f"{episode_folder.name}: {summary['status']}{ending}, {len(rows)} steps")
        return summary


    def __simulate_puck_reset(self) -> None:
        """On the simulated robot, do what the person does by hand on the real one: put the puck at the start."""

        if hasattr(self.robot_link, "place_puck_at_start"):
            self.robot_link.place_puck_at_start()


    def __start_pose(self) -> List[float]:
        """
        The rod's pose behind the puck at the start point: opposite to the first push ("start_push_direction_deg"),
        the hand turned by "hand_turn_deg" so the camera sits beside the rod (4.11.26; until then the rod stood on
        the camera's side of the puck, which was the same thing while the camera looked along the push).
        """

        puck_x, puck_y = self.recorder_parameters["start_point_xy"]
        push = math.radians(self.recorder_parameters["start_push_direction_deg"])
        offset = self.recorder_parameters["puck_centre_ahead_of_tip_m"]
        x = puck_x - offset * math.cos(push)
        y = puck_y - offset * math.sin(push)
        return [x, y, pushing_height(x, y)] + self.__turned_orientation(self.recorder_parameters["hand_turn_deg"])


    def __lower_to_start(self) -> None:
        """
        Come down onto the start pose from above, so the pusher does not sweep across the sheet.

        Close to the base the robot can refuse a straight-line descent ("cannot be reached with a linear trajectory",
        seen on 31.10.26 at x = 0.12 m); the joint-wise descent is then used, a slight arc over 5 cm before any contact.
        """

        start = self.__start_pose()
        above = list(start)
        above[2] += self.recorder_parameters["approach_height_m"]
        self.robot_link.move_to(above, is_linear = False)
        try:
            self.robot_link.move_to(start)
        except Exception as error:
            print(f"straight descent refused ({error}); descending joint-wise instead")
            self.robot_link.move_to(start, is_linear = False)


    def __lift(self) -> None:

        if self.forearm_turn_deg:
            # Lift first so the tip cannot sweep the puck, then turn the forearm straight again (Marshal's retreat).
            try:
                self.__turn_back_lifted()
                return
            except Exception as error:
                print(f"turning the forearm back failed ({str(error).splitlines()[0]}); going to the parking pose instead")
                self.forearm_turn_deg = 0.0
                self.tip_xy = None
                self.robot_link.move_home(self.recorder_parameters["home_joints"])
                return

        pose = list(self.robot_link.read_state()["pose"])
        pose[2] += self.recorder_parameters["approach_height_m"]
        try:
            self.robot_link.move_to(pose)
        except Exception as error:
            # 3.11.26: after a refused start the arm was still at the parking pose, 5 cm above which is out of reach;
            # the tool crashed here. Going to the parking pose is the safe alternative to lifting.
            print(f"lift refused ({str(error).splitlines()[0]}); going to the parking pose instead")
            self.robot_link.move_home(self.recorder_parameters["home_joints"])


    @staticmethod
    def __is_inside_workspace(
            x: float,
            y: float
    ) -> bool:

        box_x, box_y = WORKSPACE_PARAMETERS["box_x"], WORKSPACE_PARAMETERS["box_y"]
        return box_x[0] <= x <= box_x[1] and box_y[0] <= y <= box_y[1]


    def __is_step_allowed(
            self,
            target_x: float,
            target_y: float,
            step_x: float,
            step_y: float
    ) -> bool:
        """
        True when the rod's target and the puck's predicted centre both stay inside the workspace box and the puck
        stays within the sheet plus its margin.

        The puck's centre is taken as "puck_centre_ahead_of_tip_m" beyond the rod along the step; the box's far edge
        is the table edge, so checking the rod alone would let a forward key push the puck off the table.
        """

        if not self.__is_inside_workspace(target_x, target_y):
            return False

        step_length = numpy.hypot(step_x, step_y)
        if step_length < 1e-9:
            return True
        ahead = self.recorder_parameters["puck_centre_ahead_of_tip_m"]
        puck_x = target_x + ahead * step_x / step_length
        puck_y = target_y + ahead * step_y / step_length
        if not self.__is_inside_workspace(puck_x, puck_y):
            return False

        corners = numpy.array(WORKSPACE_PARAMETERS["sheet_corners_xy"])
        margin = self.recorder_parameters["sheet_margin_m"]
        return (corners[:, 0].min() - margin <= puck_x <= corners[:, 0].max() + margin
                and corners[:, 1].min() - margin <= puck_y <= corners[:, 1].max() + margin)


    def __open_session_folder(self) -> Path:
        """
        Today's latest session folder of this operator, or a new one when there is none yet or a new one was asked for.

        The operator is part of the name, so two laptops recording on the same day never write into folders of the same
        name when the recordings are merged.
        """

        raw_folder = PROJECT_FOLDER / self.recorder_parameters["raw_folder"]
        today = datetime.date.today().isoformat()
        operator = re.sub(r"[^a-z0-9]+", "_", self.operator.lower()).strip("_")
        numbers = [int(match.group(1)) for folder in raw_folder.glob(f"{today}_session_*_{operator}")
                   if (match := re.fullmatch(rf"{today}_session_(\d+)_{re.escape(operator)}", folder.name))]

        number = max(numbers, default = 0)
        if number == 0 or self.recorder_parameters["is_new_session"]:
            number += 1
        folder = raw_folder / f"{today}_session_{number:02d}_{operator}"
        folder.mkdir(parents = True, exist_ok = True)
        return folder


    def __next_episode_folder(self) -> Path:
        """Episode numbers count up per path_id across every session on this laptop (README rule 4)."""

        raw_folder = PROJECT_FOLDER / self.recorder_parameters["raw_folder"]
        numbers = [int(match.group(1)) for folder in raw_folder.glob(f"*/{self.path_id}_episode_*")
                   if (match := re.fullmatch(rf"{re.escape(self.path_id)}_episode_(\d+)", folder.name))]
        return self.session_folder / f"{self.path_id}_episode_{max(numbers, default = 0) + 1:03d}"


    def __append_session_notes(self, text: str) -> None:

        notes = self.session_folder / "session_notes.md"
        if not notes.exists():
            notes.write_text(f"# Session {self.session_folder.name}\n", encoding = "utf-8")
        with open(notes, "a", encoding = "utf-8") as file:
            file.write(text)


    @staticmethod
    def __code_commit() -> str:

        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd = PROJECT_FOLDER, capture_output = True,
                                  text = True, check = True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"


    def __wait_for(self, accepted_keys: List[str]) -> str:

        while True:
            key = self.key_reader(0)
            if key in accepted_keys:
                return key


    def __show_message(self, message: str) -> None:

        print(message)
        self.__show_picture(self.robot_link.read_picture_bytes(), message)


    def __show_picture(
            self,
            picture_bytes: bytes,
            status: str
    ) -> None:
        """Show the picture with a status bar BELOW it, so the bar never covers what the camera sees."""

        if not self.recorder_parameters["is_window_shown"]:
            return

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        status_bar = numpy.zeros((32, picture.shape[1], 3), dtype = numpy.uint8)
        cv2.putText(status_bar, f"sharpness {sharpness_of(picture):.0f} | {status}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        show_in_window(numpy.vstack([picture, status_bar]))



def main() -> None:

    parser = argparse.ArgumentParser(description = "Record demonstration episodes, one fresh picture and one 1 cm step at a time.")
    parser.add_argument("--operator", required = True, help = "who records, e.g. marshal or emranul")
    parser.add_argument("--mode", choices = ["keyboard", "hand_guided"], default = "keyboard")
    parser.add_argument("--path-id", default = SHEET_PARAMETERS["path_id"], help = "the sheet, e.g. path_01_straight_line")
    parser.add_argument("--start-x", type = float, help = "the puck's start x in metres (default: the taught line start)")
    parser.add_argument("--start-y", type = float, help = "the puck's start y in metres")
    parser.add_argument("--new-session", action = "store_true", help = "open a new session folder even if one exists today")
    parser.add_argument("--simulated", action = "store_true", help = "use the simulated robot instead of the real one")
    parser.add_argument("--camera-mount", default = RECORDING_PARAMETERS["camera_mount"],
                        help = "which mount the camera is on, e.g. stock_wrist_holder or bracket_F; never mix two in one training set")
    parser.add_argument("--orientation", default = "", help = "how the sheet lies, e.g. 'line end away from base' or 'rotated 60 deg'")
    parser.add_argument("--start-index", type = int, default = 0, help = "which marked start point along the line, 0 = the line start")
    parser.add_argument("--offset-cm", type = float, default = 0.0, help = "deliberate sideways offset of the puck from the line, in cm")
    parser.add_argument("--offset-deg", type = float, default = 0.0, help = "deliberate angle of the rod off the line's tangent, in degrees")
    parser.add_argument("--lighting", default = "", help = "a short note on the room light, e.g. 'ceiling lamps, blinds closed'")
    parser.add_argument("--max-turn-lean-deg", type = float, default = RECORDING_PARAMETERS["max_turn_lean_deg"],
                        help = "for a measurement run only: the rod lean allowed with the forearm turned, tip down (config: the measured limit)")
    arguments = parser.parse_args()

    recorder_parameters = {
        "is_new_session": arguments.new_session,
        "is_simulated": arguments.simulated,
        "camera_mount": arguments.camera_mount,
        "sheet_orientation": arguments.orientation,
        "start_index": arguments.start_index,
        "deliberate_offset": {"lateral_cm": arguments.offset_cm, "angle_deg": arguments.offset_deg},
        "lighting_note": arguments.lighting,
        "max_turn_lean_deg": arguments.max_turn_lean_deg
    }
    if arguments.max_turn_lean_deg != RECORDING_PARAMETERS["max_turn_lean_deg"]:
        print(f"MEASUREMENT RUN: turned steps allowed up to a {arguments.max_turn_lean_deg} degree lean (config: {RECORDING_PARAMETERS['max_turn_lean_deg']})")
    if arguments.start_x is not None and arguments.start_y is not None:
        recorder_parameters["start_point_xy"] = [arguments.start_x, arguments.start_y]

    if arguments.simulated:
        from robot_link import SimulatedRobotLink, simulated_hand_path
        robot_link = SimulatedRobotLink({"move_seconds": 0.5, "hand_path_xy": simulated_hand_path()})
        recorder_parameters["raw_folder"] = "03_data/raw_simulated"
    else:
        from robot_link import RobotLink
        robot_link = RobotLink(ROBOT_PARAMETERS["ip_address"])

    recorder = EpisodeRecorder(robot_link, operator = arguments.operator, path_id = arguments.path_id,
                               recorder_parameters = recorder_parameters)
    recorder.record_session(arguments.mode)


if __name__ == "__main__":
    main()
