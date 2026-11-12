"""
The model mode: the robot pushes the puck by itself, one fresh picture and one 1 cm step at a time, with a trained model
as its only decider. The camera picture is the model's only input; the robot's joint angles are used only to carry out
the model's answer, as every move command does (report section 4.6, question 1 for the professor).

The rules of report section 4.6, set in MODEL_MODE_PARAMETERS:
    - the model's exact step gives a direction in the picture; the magnet tip moves a fixed "step_m" that way (or, with
      "step_length_rule" "predicted", the length the model asked for, so that a twist in place stays in place), turned
      onto the table with the forearm camera's heading (the same heading the recording tool labelled the steps with);
    - the forearm twist follows the model, at most "twist_step_most_deg" per step and within "joint_4_most_deg";
    - the wrist (joint 5) and the hand (joint 6) are held, and joints 1 to 3 put the tip on its target; not the
      least-lean solver of the keyboard and gamepad modes, whose views the model never saw;
    - the run ends when the stop chance reaches "stop_threshold" on "stop_pictures_in_a_row" pictures in a row, at the
      step budget, at the first refused step, or on Esc.

Four ways to run it, from 02_code:
    python model_mode.py executor    no robot: recorded steps go through the executor, which must give back their
                                     direction and twist
    python model_mode.py replay      no robot: a fold model on the raw recorded pictures must give back the predictions
                                     saved when it was scored
    python model_mode.py simulated   no robot: the whole loop against the simulated robot (it checks the software, not
                                     the model's skill: the simulated pictures look nothing like the real ones)
    python model_mode.py robot       THE REAL ROBOT: only with Emranul's go, the checklist done, nothing else connected
                                     and the stop button in reach
"""

import argparse
import csv
import msvcrt
import datetime
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy
import pandas
import torch

from config import (MODEL_MODE_PARAMETERS, PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS, TRAINING_DATA_PARAMETERS,
                    TRAINING_TARGET_PARAMETERS, pushing_height, start_pose)
from record_episodes import forearm_camera_heading_rad, read_window_key, show_in_window
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed
from train_policy import PictureToMoveModel, augmented_pictures, merged_run_settings
from training_probes import design_outputs, predictions_from_outputs


class ModelPolicy:

    def __init__(
            self,
            run_name: str = MODEL_MODE_PARAMETERS["run_name"],
            checkpoint_path = MODEL_MODE_PARAMETERS["checkpoint_path"],
            metrics_path: Path = MODEL_MODE_PARAMETERS["metrics_path"]
    ):
        """
        Load a trained model, or a team of models of one recipe whose answers are averaged (an ensemble).

        Parameters
        ----------
        run_name : str, optional
            The run in TRAINING_RUNS whose recipe built the model (model 1 by default). The "exact" design and the
            "chunk" design (which also learns the exact step) can be carried out; only the exact step is used.
        checkpoint_path : pathlib.Path or list of pathlib.Path, optional
            The saved weights; several paths make a team, each with the same recipe and a different seed.
        metrics_path : pathlib.Path, optional
            A metrics.json of the recipe, which holds the picture normalisation and the class names used in training.
        """

        paths = [Path(value) for value in (checkpoint_path if isinstance(checkpoint_path, (list, tuple)) else [checkpoint_path])]
        for needed in paths + [Path(metrics_path)]:
            if not needed.exists():
                raise FileNotFoundError(f"{needed} is missing: copy the trained model there first (it is not in git)")
        metrics = json.loads(Path(metrics_path).read_text(encoding = "utf-8"))
        self.run_name = run_name
        self.checkpoint_paths = [str(value) for value in paths]
        self.settings = merged_run_settings(run_name)
        if self.settings["design"] not in ("exact", "chunk"):
            raise ValueError(f"the model mode carries out the exact step; run {run_name} has the design {self.settings['design']}")
        self.settings["chunk_steps"] = TRAINING_TARGET_PARAMETERS["chunk_steps"]
        self.settings["normalisation"] = metrics["settings"]["normalisation"]
        # the checkpoint holds every weight, so the ImageNet start weights need not be fetched (no internet on the
        # robot's hotspot)
        self.settings["is_pretrained"] = False
        self.class_names = metrics["class_names"]
        output_sizes = design_outputs(self.settings["design"], len(self.class_names), self.settings["chunk_steps"],
                                      self.settings.get("is_stop_separate", False))
        self.models = []
        for checkpoint in paths:
            model = PictureToMoveModel(self.settings, output_sizes)
            model.load_state_dict(torch.load(checkpoint, map_location = "cpu"))
            model.eval()
            self.models.append(model)
        torch.set_num_threads(self.settings["threads"])


    def picture_input(self, picture_bytes: bytes) -> torch.Tensor:
        """
        Turn a camera JPEG into the network's input exactly as training_dataset.py made the training cache: decoded,
        BGR to RGB, shrunk to the run's picture size with area averaging, then normalised as in training (no
        augmentation).
        """

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, dtype = numpy.uint8), cv2.IMREAD_COLOR)
        if picture is None:
            raise ValueError("the camera picture could not be decoded")
        width, height = self.settings["picture_size"]
        small = cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (width, height), interpolation = cv2.INTER_AREA)
        inputs, _ = augmented_pictures(torch.from_numpy(small[None]), self.settings, numpy.random.default_rng(0), is_training = False)
        return inputs


    def decide(self, picture_bytes: bytes) -> Dict[str, float]:
        """
        The model's answer to one picture.

        Parameters
        ----------
        picture_bytes : bytes
            The camera's JPEG.

        Returns
        -------
        dict
            "tip_forward_cm" and "tip_left_cm" (the step in the picture's frame), "twist_deg", "lean_deg", "hand_deg",
            "angle_deg" (the step's direction in the picture, 0 = towards its top, positive = to its left) and
            "stop_probability"; for a team, the mean of its members' exact steps and stop chances.
        """

        inputs = self.picture_input(picture_bytes)
        exact_steps, stop_probabilities = [], []
        with torch.inference_mode():
            for model in self.models:
                outputs = model(inputs)
                predictions = predictions_from_outputs(outputs, self.class_names, self.settings["stop_target"])
                exact_steps.append(outputs["exact"][0].numpy())
                stop_probabilities.append(float(predictions["stop_probability"][0]))
        exact = numpy.mean(exact_steps, axis = 0)
        if not numpy.all(numpy.isfinite(exact)) or not numpy.isfinite(numpy.mean(stop_probabilities)):
            raise ValueError("the model gave a number that is not finite")
        return {
            "tip_forward_cm": float(exact[0]),
            "tip_left_cm": float(exact[1]),
            "twist_deg": float(exact[2]),
            "lean_deg": float(exact[3]),
            "hand_deg": float(exact[4]),
            "angle_deg": float(numpy.degrees(numpy.arctan2(exact[1], exact[0]))),
            "stop_probability": float(numpy.mean(stop_probabilities))
        }



class StopRule:

    def __init__(
            self,
            threshold: float = MODEL_MODE_PARAMETERS["stop_threshold"],
            pictures_in_a_row: int = MODEL_MODE_PARAMETERS["stop_pictures_in_a_row"]
    ):
        """Stop when the stop chance reaches ``threshold`` on ``pictures_in_a_row`` pictures in a row."""

        self.threshold = threshold
        self.pictures_in_a_row = pictures_in_a_row
        self.count = 0


    def update(self, stop_probability: float) -> bool:
        """Count this picture's stop chance; True when the run should stop now."""

        self.count = self.count + 1 if stop_probability >= self.threshold else 0
        return self.count >= self.pictures_in_a_row



def escape_pressed() -> Optional[str]:
    """
    "escape" when Esc was pressed in the picture window or in the terminal since the last call, else None. Every queued
    key is read, so an Esc behind other keys is not left waiting for later steps.
    """

    is_escape = False
    while True:
        key = read_window_key(1)
        if key is None:
            break
        is_escape = is_escape or key == "escape"
    while msvcrt.kbhit():
        is_escape = is_escape or msvcrt.getwch() == "\x1b"
    return "escape" if is_escape else None



def picture_axes(joints: List[float]) -> tuple:
    """The picture's up and left directions on the table, [x, y] unit vectors, for the forearm camera at ``joints``."""

    heading = forearm_camera_heading_rad(list(joints))
    left = heading + RECORDING_PARAMETERS["picture_left_sign"] * math.pi / 2
    return numpy.array([math.cos(heading), math.sin(heading)]), numpy.array([math.cos(left), math.sin(left)])



def model_step(
        joints_now: List[float],
        decision: Dict[str, float],
        last_direction_xy: Optional[numpy.ndarray] = None,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Carry out one decision on paper: the joints of the next step, or the reason it is refused. Nothing moves.

    Parameters
    ----------
    joints_now : list of float
        The robot's six joint angles now, in radians.
    decision : dict
        From ModelPolicy.decide ("tip_forward_cm", "tip_left_cm", "twist_deg").
    last_direction_xy : numpy.ndarray, optional
        The table direction of the last step, kept when the model's direction is too short to trust.
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS.

    Returns
    -------
    dict
        "joints_next" (None when refused), "refusal" (None or the reason), "tip_now", "tip_next", "direction_xy" (unit,
        on the table), "step_cm" (the length of the step), "twist_used_deg", "is_turn_clamped", "is_direction_kept",
        "lean_deg" and "picture_heading_deg".
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    up_xy, left_xy = picture_axes(joints_now)
    tip_now = magnet_tip_and_lean(list(joints_now))[0]
    result = {"joints_next": None, "refusal": None, "tip_now": tip_now, "tip_next": None, "direction_xy": None,
              "twist_used_deg": 0.0, "is_turn_clamped": False, "is_direction_kept": False, "lean_deg": None,
              "step_cm": 0.0, "picture_heading_deg": math.degrees(math.atan2(up_xy[1], up_xy[0]))}

    vector_xy = decision["tip_forward_cm"] * up_xy + decision["tip_left_cm"] * left_xy
    predicted_cm = math.hypot(decision["tip_forward_cm"], decision["tip_left_cm"])
    if predicted_cm >= settings["smallest_direction_cm"]:
        direction_xy = vector_xy / numpy.linalg.norm(vector_xy)
    elif last_direction_xy is not None:
        direction_xy = numpy.array(last_direction_xy, dtype = float)
        result["is_direction_kept"] = True
    else:
        direction_xy = up_xy
        result["is_direction_kept"] = True
    result["direction_xy"] = direction_xy

    twist_deg = max(-settings["twist_step_most_deg"], min(settings["twist_step_most_deg"], settings["twist_gain"] * decision["twist_deg"]))
    joint_4_now_deg = math.degrees(joints_now[3])
    most = settings["joint_4_most_deg"]
    joint_4_next_deg = max(-most, min(most, joint_4_now_deg + twist_deg))
    result["is_turn_clamped"] = abs(joint_4_next_deg - (joint_4_now_deg + twist_deg)) > 1e-9
    result["twist_used_deg"] = joint_4_next_deg - joint_4_now_deg

    # the step's length: fixed, or what the model asked for (a twist in place then moves the tip hardly at all)
    if settings["step_length_rule"] == "predicted":
        step_m = min(settings["step_most_m"], settings["step_length_gain"] * predicted_cm / 100.0)
    else:
        step_m = settings["step_m"]
    result["step_cm"] = 100.0 * step_m
    target_xy = tip_now[:2] + step_m * direction_xy
    target = numpy.array([target_xy[0], target_xy[1],
                          pushing_height(target_xy[0], target_xy[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"]])
    result["tip_next"] = target

    reach = float(numpy.hypot(*target_xy))
    least_reach, most_reach = settings["tip_reach_m"]
    if not least_reach <= reach <= most_reach:
        result["refusal"] = f"the tip would be {100 * reach:.1f} cm from the base, outside {100 * least_reach:.1f} to {100 * most_reach:.1f} cm"
        return result
    area_x, area_y = settings["workspace_x_m"], settings["workspace_y_m"]
    if not (area_x[0] <= target_xy[0] <= area_x[1] and area_y[0] <= target_xy[1] <= area_y[1]):
        result["refusal"] = f"the tip would go to ({100 * target_xy[0]:.1f}, {100 * target_xy[1]:.1f}) cm, outside the workspace"
        return result
    puck_xy = target_xy + (PUCK_PARAMETERS["diameter_m"] / 2 + PUSHER_PARAMETERS["tip_diameter_m"] / 2) * direction_xy
    margin = settings["puck_margin_m"]
    if not (area_x[0] - margin <= puck_xy[0] <= area_x[1] + margin and area_y[0] - margin <= puck_xy[1] <= area_y[1] + margin):
        result["refusal"] = f"the puck would be pushed to ({100 * puck_xy[0]:.1f}, {100 * puck_xy[1]:.1f}) cm, outside the workspace"
        return result

    joints = solve_with_wrist_fixed(target, math.radians(joint_4_next_deg), math.radians(settings["wrist_held_deg"]),
                                    [joints_now[0], joints_now[1], joints_now[2]])
    if joints is None:
        result["refusal"] = "no arm position reaches the next tip point with the wrist held (out of reach or a joint limit)"
        return result
    joints[5] = math.radians(settings["hand_held_deg"])
    lean_deg = magnet_tip_and_lean(joints)[1]
    result["lean_deg"] = lean_deg
    if lean_deg > settings["rod_lean_most_deg"]:
        result["refusal"] = f"the rod would lean {lean_deg:.1f} degrees, above the limit of {settings['rod_lean_most_deg']}"
        return result
    if joints[4] > RECORDING_PARAMETERS["wrist_pitch_max_rad"]:
        result["refusal"] = "the wrist would pass the bracket's safe limit"
        return result
    result["joints_next"] = joints
    return result



def held_joints_at(
        joints_now: List[float],
        lift_m: float = 0.0,
        parameters: Optional[Dict] = None
) -> Optional[List[float]]:
    """
    The joints that keep the tip where it is now (``lift_m`` higher, at pushing height when 0), the forearm twist within
    its limit, and the wrist and hand at their held values: the settling move before the first step and the lift after
    the last. None when out of reach.
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    tip_now = magnet_tip_and_lean(list(joints_now))[0]
    most = settings["joint_4_most_deg"]
    joint_4_deg = max(-most, min(most, math.degrees(joints_now[3])))
    height = pushing_height(tip_now[0], tip_now[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"] + lift_m
    joints = solve_with_wrist_fixed(numpy.array([tip_now[0], tip_now[1], height]), math.radians(joint_4_deg),
                                    math.radians(settings["wrist_held_deg"]), [joints_now[0], joints_now[1], joints_now[2]])
    if joints is not None:
        joints[5] = math.radians(settings["hand_held_deg"])
    return joints



class ModelRun:

    def __init__(
            self,
            robot_link,
            policy: ModelPolicy,
            parameters: Optional[Dict] = None,
            key_reader: Optional[Callable[[], Optional[str]]] = None,
            is_shown: bool = True
    ):
        """
        One run of the model mode on a robot link (the real RobotLink or the SimulatedRobotLink).

        Parameters
        ----------
        robot_link
            Offers read_state, read_picture_bytes and move_joints, as robot_link.py does.
        policy : ModelPolicy
            The model.
        parameters : dict, optional
            Overrides of MODEL_MODE_PARAMETERS.
        key_reader : callable, optional
            Returns the key pressed since the last call or None; Esc ("escape") ends the run. By default the picture
            window is read.
        is_shown : bool, optional
            Show every picture with the model's answer in the window.
        """

        self.robot_link = robot_link
        self.policy = policy
        self.parameters = dict(MODEL_MODE_PARAMETERS)
        self.parameters.update(parameters or {})
        self.key_reader = key_reader or escape_pressed
        self.is_shown = is_shown


    def run(self, label: str) -> Dict:
        """
        Push until the stop rule, the step budget, a refused step or Esc ends the run, then lift the tip. Every
        picture, the model's raw answer and what was done with it are logged in a new folder under "run_folder".

        Returns
        -------
        dict
            The run's summary, also written to summary.json.
        """

        started = datetime.datetime.now()
        folder = Path(self.parameters["run_folder"]) / f"{started:%Y-%m-%d_%H%M%S}_{label}"
        (folder / "frames").mkdir(parents = True, exist_ok = True)
        stop_rule = StopRule(self.parameters["stop_threshold"], self.parameters["stop_pictures_in_a_row"])
        rows = []
        tips_and_twists = []
        moves = 0
        last_direction_xy = None
        ending = "step_budget_reached"
        error_raised = None
        try:
            for step in range(self.parameters["step_budget"]):
                picture_bytes = self.fresh_picture()
                state = self.robot_link.read_state()
                decision_started = time.perf_counter()
                decision = self.policy.decide(picture_bytes)
                is_stopping = stop_rule.update(decision["stop_probability"])
                planned = None if is_stopping else model_step(state["joints"], decision, last_direction_xy, self.parameters)
                row = self.logged_row(folder, step, picture_bytes, state, decision, planned, time.perf_counter() - decision_started)
                rows.append(row)
                # kept up to date at every step, so a crash keeps what the model did (the recording tool lost an
                # episode's labels on 5.11.26 when it wrote them only at the end)
                self.write_steps(folder, rows, "steps_in_progress.csv")
                if self.is_shown:
                    self.show(picture_bytes, row)
                if self.key_reader() == "escape":
                    ending = "quit"
                    break
                if is_stopping:
                    ending = "stopped"
                    break
                tips_and_twists.append((row["tip_x"], row["tip_y"], math.degrees(state["joints"][3])))
                if self.is_stalled(tips_and_twists):
                    ending = "stalled"
                    print(f"[step {step}] stalled: the tip stayed within {100 * self.parameters['stall_extent_m']:.0f} cm and the forearm turned "
                          f"less than {self.parameters['stall_twist_deg']:.0f} degrees over {self.parameters['stall_steps']} pictures; the run ends here")
                    break
                if planned["refusal"]:
                    ending = "refused"
                    print(f"[step {step}] refused: {planned['refusal']}; the run ends here")
                    break
                try:
                    self.robot_link.move_joints(planned["joints_next"])
                except Exception as error:
                    row.update({"action": "refused", "refusal": f"the robot refused the move: {str(error).splitlines()[0]}"})
                    ending = "refused"
                    print(f"[step {step}] {row['refusal']}; the run ends here")
                    break
                moves += 1
                last_direction_xy = planned["direction_xy"]
        except KeyboardInterrupt:
            ending = "quit"
        except Exception as error:
            ending = f"error: {str(error).splitlines()[0]}"
            error_raised = error
        finally:
            self.write_steps(folder, rows, "steps.csv")
            is_lifted = self.lift()
            summary = {"label": label, "ending": ending, "pictures": len(rows), "moves": moves, "is_lifted": is_lifted,
                       "started": started.isoformat(timespec = "seconds"),
                       "seconds": round((datetime.datetime.now() - started).total_seconds(), 1),
                       "run_name": self.policy.run_name, "checkpoints": self.policy.checkpoint_paths,
                       "parameters": {key: (str(value) if isinstance(value, Path) else value) for key, value in self.parameters.items()}}
            (folder / "summary.json").write_text(json.dumps(summary, indent = 1, default = str), encoding = "utf-8")
            print(f"run {label} ended: {ending} after {len(rows)} pictures and {moves} moves; logged in {folder}")
        if error_raised is not None:
            raise error_raised
        summary["folder"] = str(folder)
        return summary


    def is_stalled(self, tips_and_twists: List[tuple]) -> bool:
        """
        Whether the model is hesitating on the spot or shuffling back and forth.

        Parameters
        ----------
        tips_and_twists : list of tuple
            Per picture so far: the tip's x and y in metres and joint 4 in degrees.

        Returns
        -------
        bool
            True when over the last "stall_steps" pictures the tips all lie within "stall_extent_m" (the diagonal of
            the box round them) and joint 4 moved over less than "stall_twist_deg".
        """

        count = self.parameters["stall_steps"]
        if len(tips_and_twists) < count:
            return False
        recent = numpy.array(tips_and_twists[-count:])
        extent = float(numpy.hypot(recent[:, 0].max() - recent[:, 0].min(), recent[:, 1].max() - recent[:, 1].min()))
        turned_deg = float(recent[:, 2].max() - recent[:, 2].min())
        return extent < self.parameters["stall_extent_m"] and turned_deg < self.parameters["stall_twist_deg"]


    def lift(self) -> bool:
        """
        Lift the tip "lift_at_end_m" straight up from where it is, with the wrist and hand held.

        Returns
        -------
        bool
            True when the arm was lifted; when not, the person is told to lift it by hand with the motors off.
        """

        try:
            lifted = held_joints_at(self.robot_link.read_state()["joints"], self.parameters["lift_at_end_m"], self.parameters)
            if lifted is None:
                raise RuntimeError("no arm position reaches the point above the tip")
            self.robot_link.move_joints(lifted)
            return True
        except Exception as error:
            print(f"THE ARM IS STILL LOW: the lift after the run failed ({str(error).splitlines()[0]}); "
                  "switch the motors off, hold the arm and lift it by hand")
            return False


    def fresh_picture(self) -> bytes:
        """Wait for "pictures_after_move" new pictures and take the last one, so no picture taken during the move is used."""

        previous = self.robot_link.read_picture_bytes()
        new_pictures = 0
        deadline = time.monotonic() + self.parameters["fresh_picture_timeout_s"]
        while new_pictures < self.parameters["pictures_after_move"]:
            if time.monotonic() > deadline:
                raise RuntimeError("no new camera picture arrived; is the camera plugged in and streaming?")
            time.sleep(0.01)
            picture_bytes = self.robot_link.read_picture_bytes()
            if picture_bytes != previous:
                new_pictures += 1
                previous = picture_bytes
        return previous


    @staticmethod
    def logged_row(
            folder: Path,
            step: int,
            picture_bytes: bytes,
            state: Dict,
            decision: Dict[str, float],
            planned: Optional[Dict],
            decision_seconds: float
    ) -> Dict:
        """Save the picture exactly as the robot sent it and return the step's row: the model's raw answer and what was done with it."""

        (folder / "frames" / f"{step:03d}.jpg").write_bytes(picture_bytes)
        row = {"step": step, "unix_time": round(time.time(), 3), "frame": f"frames/{step:03d}.jpg", "decision_ms": round(1000 * decision_seconds, 1)}
        row.update({f"model_{name}": round(value, 5) for name, value in decision.items()})
        row.update({name: round(value, 5) for name, value in zip(["x", "y", "z", "roll", "pitch", "yaw"], state["pose"])})
        row.update({f"joint_{index + 1}": round(value, 5) for index, value in enumerate(state["joints"])})
        tip = magnet_tip_and_lean(list(state["joints"]))[0]
        row.update({"tip_x": round(float(tip[0]), 5), "tip_y": round(float(tip[1]), 5)})
        if planned is None:
            row.update({"action": "stop", "refusal": None})
            return row
        row.update({"action": "refused" if planned["refusal"] else "model_step", "refusal": planned["refusal"],
                    "picture_heading_deg": round(planned["picture_heading_deg"], 2),
                    "step_cm": round(planned["step_cm"], 3),
                    "twist_used_deg": round(planned["twist_used_deg"], 3), "is_turn_clamped": int(planned["is_turn_clamped"]),
                    "is_direction_kept": int(planned["is_direction_kept"]),
                    "lean_deg": None if planned["lean_deg"] is None else round(planned["lean_deg"], 2),
                    "tip_next_x": round(float(planned["tip_next"][0]), 5), "tip_next_y": round(float(planned["tip_next"][1]), 5)})
        return row


    @staticmethod
    def write_steps(
            folder: Path,
            rows: List[Dict],
            file_name: str
    ) -> None:
        """Write ``rows`` to ``file_name`` in ``folder``, with every column any row has (missing ones left empty)."""

        names = list(dict.fromkeys(name for row in rows for name in row)) or ["step"]
        with open(folder / file_name, "w", newline = "", encoding = "utf-8") as file:
            writer = csv.DictWriter(file, fieldnames = names)
            writer.writeheader()
            writer.writerows(rows)


    def show(
            self,
            picture_bytes: bytes,
            row: Dict
    ) -> None:
        """Show the picture with the model's direction drawn from the bottom centre and its answer written on it."""

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        height, width = picture.shape[:2]
        angle = math.radians(row["model_angle_deg"])
        start = (width // 2, height - 10)
        end = (int(start[0] - 80 * math.sin(angle)), int(start[1] - 80 * math.cos(angle)))
        cv2.arrowedLine(picture, start, end, (0, 200, 0), 3)
        # 11.11.26 (Emranul, at the robot): one long line drawn twice was unreadable, so two short lines on a dark band
        lines = [f"step {row['step']}   {row['action']}   stop chance {row['model_stop_probability']:.2f} (stops at 0.50 twice)",
                 f"twist {row['model_twist_deg']:+.1f} deg   step {row.get('step_cm', 0.0):.1f} cm   Esc = end the run"]
        cv2.rectangle(picture, (0, 0), (width, 58), (0, 0, 0), -1)
        for number, line in enumerate(lines):
            cv2.putText(picture, line, (10, 22 + 26 * number), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        show_in_window(picture)



def executor_check(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    Recorded steps through the executor, no robot: from each recorded state, the recorded step (its tip move in the
    picture's frame and its twist) is carried out on paper. The executor's direction must match the recorded tip's
    actual move on the table, and its twist the recorded joint-4 change; refusals show how often the held wrist cannot
    reach a demonstrated point.

    Returns
    -------
    pandas.DataFrame
        One row per checked step: "direction_error_deg", "twist_error_deg", "refusal".
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    frames = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv",
                             usecols = ["episode_id", "row_in_episode", "action", "tip_forward_m", "tip_left_m", "forearm_change_deg"]
                             + [f"joint_{index}" for index in range(1, 7)])
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    has_next = (frames["episode_id"].shift(-1) == frames["episode_id"]).to_numpy()
    is_step = (numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0)) >= settings["executor_check_smallest_step_m"]).to_numpy()
    candidates = numpy.flatnonzero(has_next & is_step)
    chosen = numpy.random.default_rng(settings["executor_check_seed"]).choice(candidates, size = min(len(candidates), settings["executor_check_rows"]), replace = False)

    results = []
    joint_columns = [f"joint_{index}" for index in range(1, 7)]
    for position in sorted(chosen):
        joints_now = frames.loc[position, joint_columns].to_numpy(dtype = float).tolist()
        joints_after = frames.loc[position + 1, joint_columns].to_numpy(dtype = float).tolist()
        decision = {"tip_forward_cm": 100.0 * frames.loc[position, "tip_forward_m"], "tip_left_cm": 100.0 * frames.loc[position, "tip_left_m"],
                    "twist_deg": float(frames.loc[position, "forearm_change_deg"])}
        planned = model_step(joints_now, decision, None, settings)
        recorded_move = magnet_tip_and_lean(joints_after)[0][:2] - magnet_tip_and_lean(joints_now)[0][:2]
        recorded_angle = math.degrees(math.atan2(recorded_move[1], recorded_move[0]))
        planned_angle = math.degrees(math.atan2(planned["direction_xy"][1], planned["direction_xy"][0]))
        recorded_twist = math.degrees(joints_after[3] - joints_now[3])
        is_clamped = planned["is_turn_clamped"] or abs(decision["twist_deg"]) > settings["twist_step_most_deg"]
        planned_twist = None if planned["joints_next"] is None else math.degrees(planned["joints_next"][3] - joints_now[3])
        results.append({"row": int(position), "direction_error_deg": abs((planned_angle - recorded_angle + 180.0) % 360.0 - 180.0),
                        "twist_error_deg": None if is_clamped or planned_twist is None else abs(planned_twist - recorded_twist),
                        "is_turn_clamped": is_clamped, "refusal": planned["refusal"]})
    return pandas.DataFrame(results)



def replay_check(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    A fold model on the raw recorded pictures of its held-out sheet, no robot: through ModelPolicy (its own decoding,
    resizing and normalisation) it must give back the angles and stop chances saved when the fold was scored from the
    training cache. A difference means the robot would see the pictures differently from training.

    Returns
    -------
    pandas.DataFrame
        One row per picture: "angle_difference_deg", "stop_difference".
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    run_folder = Path(settings["replay_run_folder"])
    fold = settings["replay_fold"]
    policy = ModelPolicy(settings["replay_run_name"], run_folder / "checkpoints" / f"{fold}.pt", run_folder / "metrics.json")
    saved = numpy.load(run_folder / f"predictions_{fold}.npz")
    frames = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", usecols = ["frame_index", "frame_path"])
    frame_path = frames.set_index("frame_index")["frame_path"]
    positions = numpy.linspace(0, len(saved["frame_index"]) - 1, min(len(saved["frame_index"]), settings["replay_rows"])).astype(int)

    results = []
    for position in positions:
        frame_index = int(saved["frame_index"][position])
        picture_bytes = (Path(TRAINING_DATA_PARAMETERS["raw_folder"]) / frame_path[frame_index]).read_bytes()
        decision = policy.decide(picture_bytes)
        results.append({"frame_index": frame_index,
                        "angle_difference_deg": abs((decision["angle_deg"] - float(saved["angle_deg"][position]) + 180.0) % 360.0 - 180.0),
                        "stop_difference": abs(decision["stop_probability"] - float(saved["stop_probability"][position]))})
    return pandas.DataFrame(results)



def simulated_run(
        parameters: Optional[Dict] = None,
        policy: Optional[ModelPolicy] = None
) -> Dict:
    """
    The whole loop against the simulated robot: it starts behind the simulated puck at the line's start, settles with
    the wrist and hand held, and runs until the stop rule, the budget or a refusal. It checks the software path (fresh
    pictures, the executor, the guards, the logging), not the model's skill.
    """

    from robot_link import SimulatedRobotLink

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update({"step_budget": MODEL_MODE_PARAMETERS["simulated_step_budget"]})
    settings.update(parameters or {})
    robot_link = SimulatedRobotLink()
    start = start_pose()
    start_joints = solve_with_wrist_fixed(numpy.array([start[0], start[1], pushing_height(start[0], start[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"]]),
                                          0.0, math.radians(settings["wrist_held_deg"]), [math.atan2(start[1], start[0])] + settings["solver_start_joints_2_and_3_rad"])
    start_joints[5] = math.radians(settings["hand_held_deg"])
    robot_link.move_joints(start_joints)
    run = ModelRun(robot_link, policy or ModelPolicy(), settings, key_reader = lambda: None, is_shown = False)
    return run.run("simulated")



def robot_run(
        label: str,
        parameters: Optional[Dict] = None
) -> Dict:
    """
    THE REAL ROBOT. The person puts the puck at the line's start, places the tip behind it with the motors off, lets
    go, and confirms; the tool turns the motors on, settles the wrist and hand to their held values from above, and the
    model pushes. Esc in the picture window ends the run at once.
    """

    from robot_link import RobotLink

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    policy = ModelPolicy(settings["run_name"], settings["checkpoint_path"], settings["metrics_path"])
    robot_link = RobotLink()
    are_motors_off = False
    try:
        input("Put the puck at the line's start. Press Enter to switch the motors OFF (hold the arm: it will sag) ... ")
        robot_link.set_free_motion(True)
        are_motors_off = True
        input("Guide the magnet tip just behind the puck, facing along the line. LET GO of the arm, then press Enter ... ")
        joints_placed = robot_link.read_state()["joints"]
        robot_link.set_free_motion(False)
        are_motors_off = False
        problem = placed_point_problem(joints_placed, settings)
        if problem:
            print(f"not started: {problem}")
            return {"label": label, "ending": "not_started", "reason": problem}
        lifted = held_joints_at(joints_placed, settings["settle_lift_m"], settings)
        settled = held_joints_at(joints_placed, 0.0, settings)
        if lifted is None or settled is None:
            print("not started: the placed point cannot be reached with the wrist and hand held; place the tip elsewhere")
            return {"label": label, "ending": "not_started"}
        robot_link.move_joints(lifted)
        robot_link.move_joints(settled)
        if not go_ahead_in_window(robot_link.read_picture_bytes()):
            ModelRun(robot_link, policy, settings).lift()
            print("nothing moved by the model; ending")
            return {"label": label, "ending": "not_started"}
        return ModelRun(robot_link, policy, settings).run(label)
    finally:
        if are_motors_off:
            # the motors come back on only after the person has let go, whatever ended the guiding (as the recording
            # tool's __release_arm_then_motors_on does)
            input("THE MOTORS ARE STILL OFF. Hold the arm, set it down safely, LET GO, then press Enter to switch them on ... ")
            robot_link.set_free_motion(False)
        robot_link.close()
        cv2.destroyAllWindows()



def placed_point_problem(
        joints_placed: List[float],
        parameters: Dict
) -> Optional[str]:
    """
    Check the hand-placed start against the same guards the steps use, before anything moves.

    Parameters
    ----------
    joints_placed : list of float
        The joint angles read after the person let go.
    parameters : dict
        MODEL_MODE_PARAMETERS with any overrides.

    Returns
    -------
    str or None
        Why the run should not start, or None when the start is fine.
    """

    tip = magnet_tip_and_lean(list(joints_placed))[0]
    pushing_tip_height = pushing_height(tip[0], tip[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"]
    if abs(tip[2] - pushing_tip_height) > parameters["start_height_tolerance_m"]:
        return f"the tip is {100 * (tip[2] - pushing_tip_height):+.1f} cm from pushing height; place it again just above the sheet"
    reach = float(numpy.hypot(tip[0], tip[1]))
    if not parameters["tip_reach_m"][0] <= reach <= parameters["tip_reach_m"][1]:
        return f"the tip is {100 * reach:.1f} cm from the base, outside the allowed reach"
    area_x, area_y = parameters["workspace_x_m"], parameters["workspace_y_m"]
    if not (area_x[0] <= tip[0] <= area_x[1] and area_y[0] <= tip[1] <= area_y[1]):
        return "the tip is outside the workspace"
    if abs(math.degrees(joints_placed[3])) > parameters["joint_4_most_deg"]:
        return f"the forearm is turned {math.degrees(joints_placed[3]):+.0f} degrees, beyond the limit of {parameters['joint_4_most_deg']}; turn it back a little"
    settled = held_joints_at(joints_placed, 0.0, parameters)
    if settled is not None and magnet_tip_and_lean(settled)[1] > parameters["rod_lean_most_deg"]:
        return "with the wrist held the rod would lean too much there; place the tip closer to the base"
    return None



def go_ahead_in_window(picture_bytes: bytes) -> bool:
    """
    Show the first picture and wait for Enter (start) or Esc (do not start) in the picture window, so that the window
    has the keyboard when the run begins and Esc then reaches it.
    """

    picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
    cv2.rectangle(picture, (0, 0), (picture.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(picture, "Enter = let the model push, Esc = do not start", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    show_in_window(picture)
    print("click the picture window, then Enter there lets the model push; Esc there or in this terminal ends a run")
    while True:
        key = read_window_key(0)
        if key in ("enter", "escape"):
            return key == "enter"



def main() -> None:

    parser = argparse.ArgumentParser(description = "The model mode: the robot pushes by itself with a trained model.")
    parser.add_argument("part", choices = ["executor", "replay", "simulated", "robot"])
    parser.add_argument("--label", default = "model_1", help = "a short name for the run's log folder (robot runs)")
    parser.add_argument("--run-name", default = None, help = "the recipe of the model, a key of TRAINING_RUNS (default: model 1)")
    parser.add_argument("--checkpoint", nargs = "*", default = None, help = "the model's .pt file; several make a team")
    parser.add_argument("--metrics", default = None, help = "a metrics.json of that recipe")
    parser.add_argument("--twist-gain", type = float, default = None, help = "multiply the predicted twist (1.0 = as predicted)")
    parser.add_argument("--step-length", choices = ["fixed", "predicted"], default = None,
                        help = "fixed: always step_m; predicted: the length the model asks for, at most step_most_m")
    parser.add_argument("--length-gain", type = float, default = None, help = "multiply the predicted length (with --step-length predicted)")
    parser.add_argument("--step-budget", type = int, default = None, help = "the most steps of a run")
    arguments = parser.parse_args()
    overrides = {key: value for key, value in (("run_name", arguments.run_name), ("metrics_path", arguments.metrics),
                                               ("checkpoint_path", [Path(value) for value in arguments.checkpoint] if arguments.checkpoint else None),
                                               ("twist_gain", arguments.twist_gain), ("step_length_rule", arguments.step_length),
                                               ("step_length_gain", arguments.length_gain), ("step_budget", arguments.step_budget))
                 if value is not None}

    if arguments.part == "executor":
        results = executor_check()
        refused = results["refusal"].notna()
        print(f"{len(results)} recorded steps; refused {int(refused.sum())} ({100 * refused.mean():.1f} %)")
        print(f"direction error, degrees: median {results['direction_error_deg'].median():.3f}, largest {results['direction_error_deg'].max():.3f}")
        print(f"twist against the recorded joint-4 change, degrees: median {results['twist_error_deg'].median():.3f}, largest "
              f"{results['twist_error_deg'].max():.3f} (unclamped steps); clamped turns {int(results['is_turn_clamped'].sum())}")
        print(results.loc[refused, "refusal"].str.split(",").str[0].value_counts().to_string())
    elif arguments.part == "replay":
        results = replay_check()
        print(f"{len(results)} pictures; angle difference, degrees: median {results['angle_difference_deg'].median():.4f}, "
              f"largest {results['angle_difference_deg'].max():.4f}; stop chance difference: largest {results['stop_difference'].max():.6f}")
    elif arguments.part == "simulated":
        settings = dict(MODEL_MODE_PARAMETERS)
        settings.update(overrides)
        print(json.dumps(simulated_run(overrides, ModelPolicy(settings["run_name"], settings["checkpoint_path"], settings["metrics_path"])), indent = 1, default = str))
    else:
        print(json.dumps(robot_run(arguments.label, overrides), indent = 1, default = str))



if __name__ == "__main__":
    main()
