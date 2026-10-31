"""
Record demonstration episodes: one fresh picture and one 1 cm step at a time.

Every step runs the same loop the trained model will later run: wait until the move has finished, take the first
fresh picture, ask a decider for an action, log the picture with the action and the robot's state, then move the
pusher's tip 1 cm in that direction, measured in the picture's frame (config.picture_step_to_table). An episode ends
with an explicit "stop" step.

Two modes:
    keyboard     the decider is a person pressing keys while looking at the picture on screen only;
    hand_guided  a person first guides the arm by hand (motors off) and pushes the puck to the goal while the tool
                 logs the tip's path; the person then puts the puck back, and the robot replays that path in 1 cm
                 steps, snapped back onto the path at every step. Each replay step runs only when Enter is pressed;
                 a direction key overrides it, and the episode can be discarded if the puck does not follow. Only
                 the replay becomes training data.

Keys (RECORDING_PARAMETERS in config): q w e = forward-left, forward, forward-right; a s d = left, back, right;
Space = stop; x = discard the episode; u = the previous step was a mistake (kept in the file, flagged); l = the line
is lost in this picture (the next step is flagged); Enter = accept a proposed step, or go on; Esc = quit. Keys pressed
while the robot moves are thrown away, so a key always refers to the picture on screen.

Output, one folder per episode, never edited afterwards:
    03_data/raw/<yyyy-mm-dd>_session_NN_<operator>/session_notes.md
    03_data/raw/<yyyy-mm-dd>_session_NN_<operator>/<path_id>_episode_NNN/frames/NNN.jpg, steps.csv, episode.json
    (hand_guided also guided_path.csv)

Run from 02_code, robot connected, nothing else connected to it, stop button in reach:
    python record_episodes.py --operator marshal --mode keyboard --camera-mount stock_wrist_holder
Without the robot (simulated sheet, puck and camera):
    python record_episodes.py --operator marshal --mode keyboard --simulated
"""

import argparse
import csv
import datetime
import hashlib
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy

from config import (RECORDING_PARAMETERS, ROBOT_PARAMETERS, SHEET_PARAMETERS, WORKSPACE_PARAMETERS,
                    picture_step_to_table, pushing_height, rod_pose_behind, step_budget_for)

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
WINDOW_NAME = "record_episodes"
# Three times per step: when the move returned, when the logged picture was read, when the action was chosen.
# is_mistake is set afterwards by the "u" key on the step it refers to; is_lost by the "l" key on the step whose
# picture shows no line. Training drops flagged steps (and the steps that undo a mistake); the episode survives.
STEP_COLUMNS = ["step", "unix_time", "move_return_time", "capture_time", "decision_time", "frame", "frame_md5",
                "action", "source", "is_mistake", "is_lost", "x", "y", "z", "roll", "pitch", "yaw",
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

    code = cv2.waitKey(wait_ms)
    if code < 0:
        return None

    code = code & 0xFF
    if code in (10, 13):
        return "enter"
    if code == 27:
        return "escape"
    return chr(code).lower()


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
            "sheet_orientation": self.recorder_parameters["sheet_orientation"],
            "start_index": self.recorder_parameters["start_index"],
            "deliberate_offset": self.recorder_parameters["deliberate_offset"],
            "lighting_note": self.recorder_parameters["lighting_note"],
            "start_pose": self.__start_pose(),
            "step_m": self.recorder_parameters["step_m"],
            "step_budget": self.recorder_parameters["step_budget"],
            "actions": self.recorder_parameters["actions"],
            "picture_left_sign": self.recorder_parameters["picture_left_sign"]
        }

        self.__lower_to_start()
        if mode == "hand_guided":
            guided_path = self.__record_guided_path(episode_folder)
            if guided_path is None:
                summary["status"] = "discarded_while_guiding"
                self.__lift()
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
        return self.__finish_episode(episode_folder, summary, rows)


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
        move_return_time = time.time()
        while len(rows) < self.recorder_parameters["step_budget"]:
            picture_bytes = self.__fresh_picture()
            capture_time = time.time()
            state = self.robot_link.read_state()
            action, source = decider(picture_bytes, state, len(rows))
            decision_time = time.time()

            if action in ("discard", "quit"):
                return ("discarded" if action == "discard" else "quit"), rows

            if action != "stop":
                x, y = state["pose"][0], state["pose"][1]
                step_x, step_y = picture_step_to_table(action, x, y)
                target_x, target_y = x + step_x, y + step_y
                if not self.__is_step_allowed(target_x, target_y, step_x, step_y):
                    print(f"refused: {action} would carry the rod or the puck out of the workspace; choose another step")
                    continue

            times = {"move_return_time": move_return_time, "capture_time": capture_time, "decision_time": decision_time}
            rows.append(self.__log_step(episode_folder, len(rows), picture_bytes, state, action, source, times))
            if action == "stop":
                return "completed", rows

            pose = state["pose"]
            self.robot_link.move_to([target_x, target_y, pushing_height(target_x, target_y), pose[3], pose[4], pose[5]])
            move_return_time = time.time()

        return "step_budget_reached", rows


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


    def __keyboard_decider(
            self,
            picture_bytes: bytes,
            state: Dict,
            step: int
    ) -> Tuple[str, str]:
        """The person chooses the step from the picture on screen."""

        self.__show_picture(picture_bytes, f"step {step}: q w e / a s d = move, Space = stop, u = mistake, l = lost, x = discard, Esc = quit")
        self.__drain_keys(step)
        while True:
            key = self.key_reader(0)
            if key == "escape":
                return "quit", "human"
            action = self.recorder_parameters["keys"].get(key)
            if self.__handle_flag_key(action, picture_bytes, step):
                continue
            if action in self.recorder_parameters["actions"] or action == "discard":
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
                    position + numpy.array(picture_step_to_table(action, position[0], position[1])) - target))

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
                if action in self.recorder_parameters["actions"] or action == "discard":
                    return action, "human"

        return replay_decider


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

        row = {"step": step, "unix_time": round(time.time(), 3), "frame": f"frames/{frame_name}", "frame_md5": picture_md5,
               "action": action, "source": source, "is_mistake": 0, "is_lost": int(self.is_lost_pending)}
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
        self.__append_session_notes(f"- {episode_folder.name}: {summary['status']}, {len(rows)} steps{overrides}\n")
        print(f"{episode_folder.name}: {summary['status']}, {len(rows)} steps")
        return summary


    def __simulate_puck_reset(self) -> None:
        """On the simulated robot, do what the person does by hand on the real one: put the puck at the start."""

        if hasattr(self.robot_link, "place_puck_at_start"):
            self.robot_link.place_puck_at_start()


    def __start_pose(self) -> List[float]:

        return rod_pose_behind(self.recorder_parameters["start_point_xy"])


    def __lower_to_start(self) -> None:
        """Come down onto the start pose from above, so the pusher does not sweep across the sheet."""

        start = self.__start_pose()
        above = list(start)
        above[2] += self.recorder_parameters["approach_height_m"]
        self.robot_link.move_to(above, is_linear = False)
        self.robot_link.move_to(start)


    def __lift(self) -> None:

        pose = list(self.robot_link.read_state()["pose"])
        pose[2] += self.recorder_parameters["approach_height_m"]
        self.robot_link.move_to(pose)


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
        cv2.putText(status_bar, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, numpy.vstack([picture, status_bar]))



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
    arguments = parser.parse_args()

    recorder_parameters = {
        "is_new_session": arguments.new_session,
        "is_simulated": arguments.simulated,
        "camera_mount": arguments.camera_mount,
        "sheet_orientation": arguments.orientation,
        "start_index": arguments.start_index,
        "deliberate_offset": {"lateral_cm": arguments.offset_cm, "angle_deg": arguments.offset_deg},
        "lighting_note": arguments.lighting
    }
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
