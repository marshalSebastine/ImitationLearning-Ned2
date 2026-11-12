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

Five ways to run it, from 02_code:
    python model_mode.py executor    no robot: recorded steps go through the executor, which must give back their
                                     direction and twist
    python model_mode.py replay      no robot: a fold model on the raw recorded pictures must give back the predictions
                                     saved when it was scored
    python model_mode.py simulated   no robot: the whole loop against the simulated robot (it checks the software, not
                                     the model's skill: the simulated pictures look nothing like the real ones)
    python model_mode.py rehearsal   no robot: the robot part's own questions, placing aid and go-ahead window against
                                     the simulated robot, to try the keys and the picture window before a visit
    python model_mode.py robot       THE REAL ROBOT: only with Emranul's go, the checklist done, nothing else connected
                                     and the stop button in reach

Additions of 12.11.26 for the test visits. Each one is off unless its flag is given, so that a run without them does what
the driver as committed on 12.11.26 (b0a12df) does, and none of them adds a move of the arm: --step-cm only changes how
long the fixed step is. The ten robot runs of 11.11.26 were driven before the stall rule and the step length rule
existed, so "stalled" is an ending the robot has not produced yet.
    --step-cm 0.65       the fixed step's length in centimetres for this run (not with --step-length predicted)
    --stall log          the stall rule only logs: one line, the picture in the summary, and the run goes on
    --place-help         robot part: while the arm is guided by hand, a line in the terminal shows how far the tip is
                         from the recorded start and from pushing height
    --ghost PATH         robot part: the go-ahead window shows a live picture with the picture at PATH laid faintly
                         over it, so that the sheet and the puck can be set onto their faint copies
    --fresh-keys         robot part: keys pressed before the first prompt is on the screen do not answer it, Esc ends
                         a run also when further keys wait in the terminal, and Ctrl+C is held back until the robot
                         has answered (visit_runs.py gives this flag)
    --run-folder PATH    log the run in another folder than "run_folder" (tests and rehearsals)
A robot run also prints how far the hand-placed start is from the recorded start and writes it into the summary.
visit_runs.py starts one run of a visit from a run list with these flags.

End a robot run with Esc, never with Ctrl+C. Ctrl+C that arrives while the driver waits for the robot's answer (a move
or a picture, which is most of a run) can leave the link to the robot out of step; the lift after the run then waits
for an answer for ever (12.11.26, measured on a stand-in link with the robot library's receiving code, not at the
robot). The placing aid and the live go-ahead window hold Ctrl+C back until their own question to the robot is
answered, and with --fresh-keys every question of the run is held to its end in the same way; without that flag a run
handles Ctrl+C as the committed driver does.
"""

import argparse
import csv
import msvcrt
import datetime
import json
import math
import shutil
import signal
import sys
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



def keys_typed_in_terminal() -> List[str]:
    """
    The keys typed in the terminal since the last call, in the order typed: "enter", "escape", or "other" for any
    other key. Every queued key is read, as in escape_pressed. The picture window is not read.
    """

    names = []
    while msvcrt.kbhit():
        code = msvcrt.getch()
        if code == b"\x03":
            # Ctrl+C that arrives as a key ends the wait as it ends input()
            raise KeyboardInterrupt
        if code in (b"\x00", b"\xe0") and msvcrt.kbhit():
            # an arrow or function key sends two codes; the second one waits at once and is not a typed character. Read
            # with getch, not getwch: after getwch the waiting second code does not show in kbhit (seen 12.11.26)
            msvcrt.getch()
            continue
        names.append("enter" if code in (b"\r", b"\n") else "escape" if code == b"\x1b" else "other")
    return names



def enter_pressed() -> Optional[str]:
    """
    "enter" when Enter was pressed in the terminal since the last call, else None. This serves the placing aid, before
    the picture window is on the screen.
    """

    return "enter" if "enter" in keys_typed_in_terminal() else None



def escape_pressed_in_terminal() -> Optional[str]:
    """
    "escape" when Esc was pressed in the terminal since the last call, else None. This serves the live go-ahead window,
    whose own keys (Enter, Esc) are read from the picture window.
    """

    return "escape" if "escape" in keys_typed_in_terminal() else None



def escape_pressed_with_queue_emptied() -> Optional[str]:
    """
    The key reader of a run started with "is_typed_ahead_discarded" (--fresh-keys): "escape" when Esc was pressed in
    the picture window or in the terminal since the last call, else None, as escape_pressed; but every key that waits
    in the terminal is read, whatever came before it.

    escape_pressed stops reading the terminal once it has seen Esc. When another key waits there (Esc pressed twice or
    held down, Esc and then Enter, or Esc in the picture window while a key waits in the terminal) it never comes
    back, and the run stands still with the arm held where it is (12.11.26, found with a stand-in keyboard, not at the
    robot). It is left as it is, so that a run without the new flags does what the committed driver does.
    """

    is_escape = False
    while True:
        key = read_window_key(1)
        if key is None:
            break
        is_escape = is_escape or key == "escape"
    is_escape = "escape" in keys_typed_in_terminal() or is_escape
    return "escape" if is_escape else None



def with_interrupt_kept(call: Callable) -> object:
    """
    Carry out one question to the robot to its end, even when Ctrl+C arrives meanwhile.

    The robot library sends a command and then reads one answer, which it takes for that command's. Ctrl+C that
    arrives while the answer is awaited is raised once the answer's first bytes have been read; the rest of the answer
    then stays in the link, every later answer is taken for the wrong command, and the next question waits for ever
    (12.11.26, measured on a stand-in link with the library's receiving code, not at the robot). Here Ctrl+C only
    leaves a mark while ``call`` runs and is raised when the whole answer has been read, so that the link stays in
    step.

    Parameters
    ----------
    call : callable
        The question to the robot, without arguments (for example the robot link's read_state).

    Returns
    -------
    object
        What ``call`` returned.

    Raises
    ------
    KeyboardInterrupt
        After ``call`` has returned, when Ctrl+C arrived while it ran.
    """

    marks = []
    try:
        previous = signal.signal(signal.SIGINT, lambda number, frame: marks.append(number))
    except ValueError:
        # outside the program's main thread Ctrl+C cannot be held back, and it is not raised there either
        return call()
    try:
        result = call()
    finally:
        signal.signal(signal.SIGINT, previous)
    if marks:
        raise KeyboardInterrupt
    return result



class LinkWithInterruptKept:

    def __init__(self, robot_link: object):
        """
        A robot link whose every question to the robot is carried out to its end before Ctrl+C is raised
        (with_interrupt_kept), for a run started with "is_typed_ahead_discarded" (--fresh-keys). Ctrl+C during a move
        then ends the run after that move, with the lift, instead of leaving the link to the robot out of step.

        Parameters
        ----------
        robot_link : object
            The link itself (the real RobotLink or the SimulatedRobotLink); every call is handed on to it.
        """

        self.robot_link = robot_link


    def __getattr__(self, name: str) -> object:
        """Hand a call on to the link itself, held to its end as with_interrupt_kept holds it."""

        attribute = getattr(self.robot_link, name)
        if not callable(attribute):
            return attribute
        return lambda *arguments, **named_arguments: with_interrupt_kept(lambda: attribute(*arguments, **named_arguments))



def discard_typed_keys() -> int:
    """
    Throw away the keys typed in the terminal so far, so that a prompt which follows is answered only by a key pressed
    once it is on the screen: an Enter pressed while the model was loading must not switch the motors off.

    Returns
    -------
    int
        How many key codes were thrown away.
    """

    count = 0
    while msvcrt.kbhit():
        msvcrt.getch()
        count += 1
    return count



def is_terminal_a_console() -> bool:
    """Whether keys typed in the terminal can be read one by one (a console window, not a pipe or an editor's output pane)."""

    return sys.stdin is not None and sys.stdin.isatty()



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
            is_shown: bool = True,
            start_offset: Optional[Dict[str, float]] = None
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
        start_offset : dict, optional
            The hand-placed start's offset from the recorded start (from offset_from_recorded_start), written into
            the summary; None for a run whose start was not placed by hand.
        """

        self.robot_link = robot_link
        self.policy = policy
        self.parameters = dict(MODEL_MODE_PARAMETERS)
        self.parameters.update(parameters or {})
        self.key_reader = key_reader or escape_pressed
        self.is_shown = is_shown
        self.start_offset = start_offset


    def run(self, label: str) -> Dict:
        """
        Push until the stop rule, the step budget, a refused step or Esc ends the run, then lift the tip. Every
        picture, the model's raw answer and what was done with it are logged in a new folder under "run_folder". The
        stall rule ends the run too, unless "stall_action" is "log": then it prints one line the first time it fires,
        the summary names that picture ("stall_logged_at_picture") and the run goes on.

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
        stall_logged_at_picture = None
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
                is_stall_rule_fired = self.is_stalled(tips_and_twists)
                if is_stall_rule_fired and self.parameters["stall_action"] == "log":
                    # the stall rule only logs (--stall log): one line the first time it fires, that picture in the
                    # summary, and the run goes on until something else ends it
                    if stall_logged_at_picture is None:
                        # the step number goes into the summary: it counts from 0, as the column "step" and the frame
                        # files do, so step 29 is the run's picture number 30
                        stall_logged_at_picture = step
                        print(f"[step {step}] the stall rule fired (the run's picture number {step + 1}; steps are counted from 0): the tip stayed within "
                              f"{100 * self.parameters['stall_extent_m']:.0f} cm and the forearm turned less than {self.parameters['stall_twist_deg']:.0f} "
                              f"degrees over {self.parameters['stall_steps']} pictures; it is only logged and the run goes on (Esc ends it)")
                elif is_stall_rule_fired:
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
                       # 12.11.26: the time from the first to the last picture ("seconds" also counts the lift), the
                       # fixed step's length (None when the model's own length is used), the first picture at which
                       # the stall rule fired when it only logs, and the hand-placed start's offset
                       "seconds_first_to_last_picture": round(rows[-1]["unix_time"] - rows[0]["unix_time"], 1) if rows else None,
                       "step_length_rule": self.parameters["step_length_rule"],
                       "fixed_step_cm": None if self.parameters["step_length_rule"] == "predicted" else round(100.0 * self.parameters["step_m"], 4),
                       "stall_action": self.parameters["stall_action"],
                       "stall_logged_at_picture": stall_logged_at_picture,
                       "start_offset": None if self.start_offset is None else {name: round(float(value), 2) + 0.0 for name, value in self.start_offset.items()},
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
        lines = window_lines(row, self.parameters)
        cv2.rectangle(picture, (0, 0), (width, 58), (0, 0, 0), -1)
        for number, line in enumerate(lines):
            cv2.putText(picture, line, (10, 22 + 26 * number), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        show_in_window(picture)



def window_lines(
        row: Dict,
        parameters: Dict
) -> List[str]:
    """
    The two lines written on a run's picture in the window.

    Parameters
    ----------
    row : dict
        The step's row, from ModelRun.logged_row.
    parameters : dict
        MODEL_MODE_PARAMETERS with any overrides ("step_length_rule").

    Returns
    -------
    list of str
        The step's number, what was done and the stop chance; then the model's twist, the step's length and how the
        run is ended.
    """

    # a fixed step that is not a whole millimetre (--step-cm 0.65) is written in full, since "0.7 cm" would misstate
    # it; every other length keeps its one decimal
    step_cm = row.get("step_cm", 0.0)
    is_written_in_full = parameters["step_length_rule"] != "predicted" and abs(step_cm - round(step_cm, 1)) > 1e-6
    step_text = f"{step_cm:.2f}" if is_written_in_full else f"{step_cm:.1f}"
    return [f"step {row['step']}   {row['action']}   stop chance {row['model_stop_probability']:.2f} (stops at 0.50 twice)",
            f"twist {row['model_twist_deg']:+.1f} deg   step {step_text} cm   Esc = end the run"]



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
        policy: Optional[ModelPolicy] = None,
        label: str = "simulated"
) -> Dict:
    """
    The whole loop against the simulated robot: it starts behind the simulated puck at the line's start, settles with
    the wrist and hand held, and runs until the stop rule, the budget or a refusal. It checks the software path (fresh
    pictures, the executor, the guards, the logging), not the model's skill. ``label`` names the run's log folder.
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
    return run.run(label)



def rehearsal_hand_path(parameters: Dict) -> List[List[float]]:
    """
    The poses of a simulated hand that brings the magnet tip to the recorded start, one pose per reading of the
    robot's state while the motors are off.

    Parameters
    ----------
    parameters : dict
        MODEL_MODE_PARAMETERS with any overrides ("recorded_start_joints_deg", "rehearsal_hand_start_m",
        "rehearsal_hand_start_forearm_deg", "rehearsal_hand_readings").

    Returns
    -------
    list of list of float
        "rehearsal_hand_readings" joint sets in radians: the first lies "rehearsal_hand_start_m" from the recorded
        start with the forearm turned by "rehearsal_hand_start_forearm_deg", the last is the recorded start itself,
        and the ones between come nearer in equal steps (a pose the arm model cannot reach is left out).
    """

    recorded_joints = [math.radians(value) for value in parameters["recorded_start_joints_deg"]]
    recorded_tip = magnet_tip_and_lean(recorded_joints)[0]
    count = parameters["rehearsal_hand_readings"]
    poses = []
    for index in range(count - 1):
        share_left = 1.0 - index / (count - 1)
        target = recorded_tip + share_left * numpy.array(parameters["rehearsal_hand_start_m"], dtype = float)
        forearm_rad = recorded_joints[3] + share_left * math.radians(parameters["rehearsal_hand_start_forearm_deg"])
        joints = solve_with_wrist_fixed(target, forearm_rad, recorded_joints[4], recorded_joints[:3])
        if joints is not None:
            joints[5] = recorded_joints[5]
            poses.append(joints)
    return poses + [recorded_joints]



def rehearsal_run(
        label: str = "rehearsal",
        parameters: Optional[Dict] = None,
        stand_in_policy: Optional[ModelPolicy] = None
) -> Dict:
    """
    The robot part without a robot: robot_run's own questions, placing aid, go-ahead window and run against the
    simulated robot, so that the keys and the picture window can be tried at the laptop before a visit. No robot is
    connected and nothing real moves. The run is logged under "rehearsal_run_folder", not in 03_data.

    With the placing aid a simulated hand brings the tip to the recorded start while the motors are "off"
    (rehearsal_hand_path), so that the placing line changes as it will at the robot; without the aid the simulated arm
    stands at the recorded start. Like the simulated part it checks the software and its handling, not the model's
    skill.

    Parameters
    ----------
    label : str, optional
        A short name for the run's log folder.
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS, as the robot part takes them.
    stand_in_policy : ModelPolicy, optional
        A stand-in for the model (tests); left out, the model of the settings is loaded.

    Returns
    -------
    dict
        What robot_run returns.
    """

    from robot_link import SimulatedRobotLink
    from side_view_test import ned2_fk

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update({"step_budget": MODEL_MODE_PARAMETERS["simulated_step_budget"], "run_folder": MODEL_MODE_PARAMETERS["rehearsal_run_folder"]})
    settings.update(parameters or {})
    hand_path = rehearsal_hand_path(settings) if settings["is_place_help_shown"] else []
    # the simulated hand moves the arm one pose further at every reading of the state while the motors are off
    robot_link = SimulatedRobotLink({"hand_path_xy": [[float(value) for value in ned2_fk.tool_pose(list(joints))[0][:2]] for joints in hand_path],
                                     "hand_path_joints": hand_path})
    robot_link.move_joints(hand_path[0] if hand_path else [math.radians(value) for value in settings["recorded_start_joints_deg"]])
    print("REHEARSAL: no robot is connected and nothing real moves. The questions, the keys and the picture window are the robot part's own;\n"
          "the arm and the camera are simulated (the simulated pictures look nothing like the real ones).")
    if hand_path:
        print("While the motors are \"off\", a simulated hand brings the tip to the recorded start within about "
              f"{len(hand_path) * settings['place_help_refresh_s']:.0f} s, so that the placing line changes as it will at the robot.")
    print(f"The run is logged in {settings['run_folder']}.")
    return robot_run(label, settings, stand_in_link = robot_link, stand_in_policy = stand_in_policy)



def start_settings_problem(
        parameters: Dict,
        ghost: Optional[numpy.ndarray]
) -> Optional[str]:
    """
    Check the settings of the additions of 12.11.26 before the robot is connected, so that a missing file or a mistake
    in config.py shows before the motors are switched off, not after.

    Parameters
    ----------
    parameters : dict
        MODEL_MODE_PARAMETERS with any overrides.
    ghost : numpy.ndarray or None
        The reference picture read from "ghost_path"; None when none is asked for or it could not be read.

    Returns
    -------
    str or None
        Why the run should not start, or None when the settings are fine.
    """

    if parameters["ghost_path"] and ghost is None:
        return f"the reference picture {parameters['ghost_path']} is missing or cannot be read as a picture (--ghost)"
    if parameters["stall_action"] not in ("end", "log"):
        return f"the setting \"stall_action\" is {parameters['stall_action']!r}; it must be \"end\" or \"log\" (config.py, or --stall)"
    if ghost is not None:
        try:
            ghost_blend(ghost, ghost, parameters["ghost_strength"])
            float(parameters["ghost_refresh_s"])
        except (TypeError, ValueError, cv2.error):
            return (f"the settings \"ghost_strength\" ({parameters['ghost_strength']!r}) and \"ghost_refresh_s\" ({parameters['ghost_refresh_s']!r}) "
                    "must both be numbers (config.py)")
    return None



def robot_run(
        label: str,
        parameters: Optional[Dict] = None,
        stand_in_link: Optional[object] = None,
        stand_in_policy: Optional[ModelPolicy] = None
) -> Dict:
    """
    THE REAL ROBOT. The person puts the puck at the line's start, places the tip behind it with the motors off, lets
    go, and confirms; the tool turns the motors on, settles the wrist and hand to their held values from above, and the
    model pushes. Esc in the picture window ends the run at once.

    How far the placed start is from the recorded start is printed and written into the summary; no run is refused
    for it. With "is_place_help_shown" (--place-help) that offset is also shown while the arm is guided, and with
    "ghost_path" (--ghost) the go-ahead window shows a live picture under a faint reference picture; neither commands
    a move. With "is_typed_ahead_discarded" (--fresh-keys) keys pressed before the first prompt do not answer it, the
    run reads its keys with escape_pressed_with_queue_emptied, so that Esc ends it also when further keys wait in the
    terminal, and every question to the robot is held to its end before Ctrl+C is raised (LinkWithInterruptKept).
    ``stand_in_link`` and ``stand_in_policy`` serve the tests and the rehearsal, which have no robot: left out, the real
    robot is connected and the model of the settings is loaded.
    """

    from robot_link import RobotLink

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    # the reference picture and the new settings are tried before the robot is connected, so that a missing file or
    # a mistake in config.py shows before the motors are switched off, not after
    ghost = reference_picture(settings["ghost_path"]) if settings["ghost_path"] else None
    problem = start_settings_problem(settings, ghost)
    if problem:
        print(f"not started: {problem}")
        return {"label": label, "ending": "not_started", "reason": problem}
    start_offset_lines(offset_from_recorded_start([math.radians(value) for value in settings["recorded_start_joints_deg"]], settings), settings)
    policy = ModelPolicy(settings["run_name"], settings["checkpoint_path"], settings["metrics_path"]) if stand_in_policy is None else stand_in_policy
    robot_link = RobotLink() if stand_in_link is None else stand_in_link
    if settings["is_typed_ahead_discarded"]:
        # with --fresh-keys Ctrl+C never cuts an answer of the robot in two: after a move it ends the run with the lift
        robot_link = LinkWithInterruptKept(robot_link)
    are_motors_off = False
    try:
        if settings["is_typed_ahead_discarded"]:
            if not is_terminal_a_console():
                # keys can be thrown away only where they are read one by one; elsewhere the person is told so
                print("keys pressed early cannot be thrown away in this terminal (it is not a console window such as PowerShell, cmd or the terminal "
                      "of VS Code): press no key until a question is on the screen")
            elif discard_typed_keys():
                print("keys pressed before this prompt were ignored")
        input("Put the puck at the line's start. Press Enter to switch the motors OFF (hold the arm: it will sag) ... ")
        robot_link.set_free_motion(True)
        are_motors_off = True
        if not placed_with_help(robot_link, settings):
            input("Guide the magnet tip just behind the puck, facing along the line. LET GO of the arm, then press Enter ... ")
        joints_placed = robot_link.read_state()["joints"]
        robot_link.set_free_motion(False)
        are_motors_off = False
        start_offset = offset_from_recorded_start(joints_placed, settings)
        problem = placed_point_problem(joints_placed, settings)
        # the second line's advice (Esc in the picture window) is left out when the start is refused: no window comes then
        print("\n".join(start_offset_lines(start_offset, settings)[:1 if problem else None]))
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
        if ghost is None:
            is_go_ahead_given = go_ahead_in_window(robot_link.read_picture_bytes())
        else:
            try:
                is_go_ahead_given = go_ahead_with_ghost(robot_link.read_picture_bytes, read_window_key, ghost, settings,
                                                        read_terminal_key = escape_pressed_in_terminal)
            except Exception as error:
                # the live window itself failed (not the camera, which it waits for): nothing is moved without a key
                # from the person, so the arm is not lifted here; the person is told where it stands
                problem = f"the live window stopped ({type(error).__name__}: {(str(error).splitlines() or [''])[0]})"
                print(f"not started: {problem}. THE ARM IS STILL LOW, held by its motors: nothing has lifted it. Start the run again")
                return {"label": label, "ending": "not_started", "reason": problem}
        if not is_go_ahead_given:
            ModelRun(robot_link, policy, settings).lift()
            print("nothing moved by the model; ending")
            return {"label": label, "ending": "not_started"}
        # with --fresh-keys the run empties the terminal's key queue at every step; without it the committed reader is used
        key_reader = escape_pressed_with_queue_emptied if settings["is_typed_ahead_discarded"] else None
        return ModelRun(robot_link, policy, settings, key_reader = key_reader, start_offset = start_offset).run(label)
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



def offset_from_recorded_start(
        joints_placed: List[float],
        parameters: Optional[Dict] = None
) -> Dict[str, float]:
    """
    How far a hand-placed start is from the recorded start, the arm pose from which most demonstrations begin
    ("recorded_start_joints_deg"). Worked out with the arm model: nothing is read from the robot and nothing moves.

    Parameters
    ----------
    joints_placed : list of float
        The six joint angles of the placed arm, in radians.
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS.

    Returns
    -------
    dict
        "tip_distance_cm": how far the placed magnet tip is from the recorded start's tip, measured on the table;
        "tip_along_x_cm" and "tip_along_y_cm": the two parts of that distance in the robot's frame (x positive =
        further out from the base, y positive = to the left as seen from the base looking out along x);
        "forearm_deg": joint 4 minus the recorded joint 4, in degrees.
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    recorded_joints = [math.radians(value) for value in settings["recorded_start_joints_deg"]]
    recorded_tip = magnet_tip_and_lean(recorded_joints)[0]
    placed_tip = magnet_tip_and_lean(list(joints_placed))[0]
    along_x_cm = 100.0 * float(placed_tip[0] - recorded_tip[0])
    along_y_cm = 100.0 * float(placed_tip[1] - recorded_tip[1])
    return {
        "tip_distance_cm": math.hypot(along_x_cm, along_y_cm),
        "tip_along_x_cm": along_x_cm,
        "tip_along_y_cm": along_y_cm,
        "forearm_deg": math.degrees(joints_placed[3] - recorded_joints[3])
    }



def is_start_close(
        offset: Dict[str, float],
        parameters: Optional[Dict] = None
) -> bool:
    """
    Whether a start offset lies within both warning limits, "start_offset_warn_m" and "start_forearm_warn_deg". The
    numbers are judged as they are printed, with one decimal, so that a line never shows "1.0 cm" beside the warning
    "further off than 1 cm".
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    return (round(offset["tip_distance_cm"], 1) <= 100.0 * settings["start_offset_warn_m"]
            and round(abs(offset["forearm_deg"]), 1) <= settings["start_forearm_warn_deg"])



def start_offset_lines(
        offset: Dict[str, float],
        parameters: Optional[Dict] = None
) -> List[str]:
    """
    The start offset in words for the terminal.

    Parameters
    ----------
    offset : dict
        From offset_from_recorded_start.
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS.

    Returns
    -------
    list of str
        One line with the four numbers; and, when the start is further off than a warning limit, a second line that
        says how to place it again. The lines inform only: no run is refused for its offset.
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    # a part that rounds to 0.0 is worded as "further out" and "to the left", not by the sign of its last digits, and
    # a forearm difference that rounds to zero is written +0.0 (adding 0.0 turns a rounded minus zero into zero)
    along_x = f"{abs(offset['tip_along_x_cm']):.1f} cm {'further out' if round(offset['tip_along_x_cm'], 1) >= 0 else 'closer to the base'}"
    along_y = f"{abs(offset['tip_along_y_cm']):.1f} cm to the {'left' if round(offset['tip_along_y_cm'], 1) >= 0 else 'right'}"
    lines = [f"start: tip {offset['tip_distance_cm']:.1f} cm from the recorded start ({along_x}, {along_y}; left and right as seen from the "
             f"robot's base looking out along x), forearm {round(offset['forearm_deg'], 1) + 0.0:+.1f} degrees"]
    if not is_start_close(offset, settings):
        lines.append(f"the start is further off than {100.0 * settings['start_offset_warn_m']:g} cm or {settings['start_forearm_warn_deg']:g} degrees: "
                     "to place it again, press Esc in the picture window and start the same run again")
    return lines



def height_above_pushing_m(joints_placed: List[float]) -> float:
    """
    How far the magnet tip is above (positive) or below (negative) pushing height at ``joints_placed`` (radians), in
    metres, worked out with the arm model as placed_point_problem does; nothing is read from the robot.
    """

    tip = magnet_tip_and_lean(list(joints_placed))[0]
    pushing_tip_height = pushing_height(tip[0], tip[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"]
    return float(tip[2] - pushing_tip_height)



def placing_line(
        offset: Dict[str, float],
        parameters: Optional[Dict] = None,
        height_m: Optional[float] = None
) -> str:
    """
    The start offset as one short line for the placing aid.

    Parameters
    ----------
    offset : dict
        From offset_from_recorded_start.
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS.
    height_m : float, optional
        The tip's height against pushing height, from height_above_pushing_m; left out, no height is shown.

    Returns
    -------
    str
        Where the tip is against the recorded start ("too far out", "too far left": it is to be moved the other way),
        the forearm's difference and the height, and a last word: "close" when both warning limits are kept, "not
        close" when not, and "too high" or "too low" when the height is beyond "start_height_tolerance_m", where a run
        does not start.
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    out_or_close = "too far out" if round(offset["tip_along_x_cm"], 1) >= 0 else "too close"
    left_or_right = "too far left" if round(offset["tip_along_y_cm"], 1) >= 0 else "too far right"
    line = (f"tip {offset['tip_distance_cm']:4.1f} cm off: {abs(offset['tip_along_x_cm']):.1f} {out_or_close}, "
            f"{abs(offset['tip_along_y_cm']):.1f} {left_or_right}, forearm {round(offset['forearm_deg'], 1) + 0.0:+6.1f} deg")
    last_word = "close" if is_start_close(offset, settings) else "not close"
    if height_m is not None:
        line += f", height {round(100.0 * height_m, 1) + 0.0:+5.1f} cm"
        # the same comparison as placed_point_problem makes, so that "close" is never shown where the run is refused
        if abs(height_m) > settings["start_height_tolerance_m"]:
            last_word = "too high" if height_m > 0 else "too low"
    return f"{line}: {last_word}"



def placing_help(
        read_joints: Callable[[], List[float]],
        read_key: Callable[[], Optional[str]],
        parameters: Optional[Dict] = None
) -> bool:
    """
    The placing aid: while the arm is guided by hand with the motors off, show how far the tip is from the recorded
    start and from pushing height, a new reading every "place_help_refresh_s" on one line that overwrites itself,
    until Enter. The robot's state is only read; nothing moves. Ctrl+C that arrives during a reading is held back
    until the reading is complete (with_interrupt_kept), so that the link to the robot stays in step for the command
    that switches the motors back on.

    Parameters
    ----------
    read_joints : callable
        Returns the arm's six joint angles now, in radians.
    read_key : callable
        Returns "enter" when Enter was pressed since its last call, else None (enter_pressed for the terminal).
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS.

    Returns
    -------
    bool
        True when Enter ended the wait; Enter counts once "place_help_readings_before_enter" readings have been shown
        (until then the line ends in "(wait)"). False when the aid could not go on (a reading failed): it says so, and
        the caller asks for Enter in the plain way instead.
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    print("Guide the magnet tip just behind the puck, facing along the line.\n"
          f"The line below says where the tip IS against the recorded start, a new reading every {settings['place_help_refresh_s']:g} s.\n"
          "Move the tip the other way until the numbers shrink:\n"
          "  too far out / too close = further from / nearer to the base than the recorded start;\n"
          "  too far left / right    = as seen from the robot's base looking out along x;\n"
          f"  forearm = joint 4; close = within {100.0 * settings['start_offset_warn_m']:g} cm and {settings['start_forearm_warn_deg']:g} degrees "
          "of the recorded start;\n"
          "  height  = the tip above (+) or below (-) pushing height; too high / too low = more than "
          f"{100.0 * settings['start_height_tolerance_m']:g} cm from it,\n"
          "            where a run does not start.\n"
          "LET GO of the arm, then press Enter ...")
    readings_shown = 0
    is_narrow_terminal_told = False
    try:
        # keys typed before the aid was on the screen do not count as its Enter
        read_key()
        while True:
            # one column is left free, so that the line never wraps and the next reading lands on top of it
            columns = max(20, shutil.get_terminal_size((120, 30)).columns - 1)
            joints = with_interrupt_kept(read_joints)
            readings_shown += 1
            # Enter counts from the second reading on: a long or double press of the Enter that switched the motors
            # off must not end the placing while the arm is still in the hand; the line says "(wait)" until then
            is_enter_counted = readings_shown >= settings["place_help_readings_before_enter"]
            line = placing_line(offset_from_recorded_start(joints, settings), settings, height_above_pushing_m(joints))
            line += "   Enter = placed" if is_enter_counted else "   (wait)"
            if len(line) > columns and not is_narrow_terminal_told:
                # said once, on a line of its own: the last word of the live line (close, too high) would be cut off unseen
                print(("\n" if readings_shown > 1 else "") + f"(this terminal is {columns + 1} columns wide and cuts the end of the line below off; "
                      "widen the window to read all of it)")
                is_narrow_terminal_told = True
            print("\r" + line[:columns].ljust(columns), end = "", flush = True)
            next_reading_time = time.monotonic() + settings["place_help_refresh_s"]
            while True:
                if read_key() == "enter" and is_enter_counted:
                    print()
                    return True
                if time.monotonic() >= next_reading_time:
                    break
                time.sleep(settings["place_help_key_poll_s"])
    except KeyboardInterrupt:
        # Ctrl+C: the line is ended first, so that the question about the motors that follows starts on a line of its own
        print()
        raise
    except Exception as error:
        print(f"\nthe placing aid stopped ({(str(error).splitlines() or [type(error).__name__])[0]}); placing goes on without it")
        return False



def placed_with_help(
        robot_link: object,
        parameters: Dict
) -> bool:
    """
    The placing aid on a robot link, when "is_place_help_shown" asks for it and the terminal can be read key by key.

    Parameters
    ----------
    robot_link
        Offers read_state, as robot_link.py does; nothing else is asked of it.
    parameters : dict
        MODEL_MODE_PARAMETERS with any overrides.

    Returns
    -------
    bool
        True when the person ended the placing with Enter in the aid; False when the plain prompt is to be used (the
        aid is off, the terminal is not a console window, or a reading failed).
    """

    if not parameters["is_place_help_shown"]:
        return False
    if not is_terminal_a_console():
        print("the placing aid reads Enter from a console window (PowerShell, cmd or the terminal of VS Code) and this is not one; placing goes on without it")
        return False
    return placing_help(lambda: robot_link.read_state()["joints"], enter_pressed, parameters)



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



def reference_picture(path: Path) -> Optional[numpy.ndarray]:
    """
    The picture file at ``path``, decoded as the camera's pictures are.

    Parameters
    ----------
    path : pathlib.Path
        A JPEG of the same camera, for example the first picture of a demonstration (a path given as text is taken
        too).

    Returns
    -------
    numpy.ndarray or None
        The picture (height, width, 3; blue, green, red), or None when the file is missing or is not a picture.
    """

    try:
        picture_bytes = Path(path).read_bytes()
    except OSError:
        return None
    if not picture_bytes:
        return None
    try:
        return cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None



def ghost_blend(
        live_picture: numpy.ndarray,
        ghost_picture: numpy.ndarray,
        strength: float
) -> numpy.ndarray:
    """
    Lay a reference picture faintly over a live one.

    Parameters
    ----------
    live_picture : numpy.ndarray
        The camera's picture now (height, width, 3), 8 bit.
    ghost_picture : numpy.ndarray
        The reference picture, also (height, width, 3) and 8 bit, as reference_picture gives it; it is resized when
        its size is not the live picture's.
    strength : float
        How strongly the reference shows, from 0 (only the live picture) to 1 (only the reference).

    Returns
    -------
    numpy.ndarray
        A new picture of the live picture's size: (1 - strength) of the live picture plus strength of the reference.
    """

    height, width = live_picture.shape[:2]
    if ghost_picture.shape[:2] != (height, width):
        is_shrunk = ghost_picture.shape[0] * ghost_picture.shape[1] > height * width
        ghost_picture = cv2.resize(ghost_picture, (width, height), interpolation = cv2.INTER_AREA if is_shrunk else cv2.INTER_LINEAR)
    strength = max(0.0, min(1.0, float(strength)))
    return cv2.addWeighted(live_picture, 1.0 - strength, ghost_picture, strength, 0.0)



def go_ahead_with_ghost(
        read_picture_bytes: Callable[[], bytes],
        read_key: Callable[[int], Optional[str]],
        ghost: numpy.ndarray,
        parameters: Optional[Dict] = None,
        show: Optional[Callable[[numpy.ndarray], None]] = None,
        read_terminal_key: Optional[Callable[[], Optional[str]]] = None
) -> bool:
    """
    The go-ahead window with a faint reference picture over a LIVE picture (a fresh one every "ghost_refresh_s"), so
    that the sheet can be slid and the puck set until the line and the puck lie on their faint copies. It waits for
    Enter (start) or Esc (do not start) in the picture window, as go_ahead_in_window does. Only pictures are read: the
    arm stands still with its motors on.

    A camera picture that cannot be read (the request fails, or what comes is no picture) does not end the wait: the
    window then shows the reference alone and says so, the terminal says it once, and Enter does not count until a
    picture has come again. Nothing here ends the wait by itself, so that the arm is lifted only after a key from the
    person. Ctrl+C is taken as Esc; when it arrives during a camera request it is held back until the answer is
    complete (with_interrupt_kept), so that the link to the robot stays in step for the lift.

    Parameters
    ----------
    read_picture_bytes : callable
        Returns the camera's latest JPEG (the robot link's read_picture_bytes).
    read_key : callable
        Takes a wait in milliseconds and returns the key pressed in the picture window within it, or None
        (read_window_key).
    ghost : numpy.ndarray
        The reference picture, from reference_picture.
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS ("ghost_strength", "ghost_refresh_s").
    show : callable, optional
        Shows a picture; by default the picture window (show_in_window).
    read_terminal_key : callable, optional
        Returns "escape" when Esc was pressed in the terminal since its last call, else None
        (escape_pressed_in_terminal); left out, the terminal is not read.

    Returns
    -------
    bool
        True for Enter in the picture window under a live picture; False for Esc there or in the terminal, and for
        Ctrl+C.
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    show = show or show_in_window
    wait_ms = max(1, int(round(1000 * settings["ghost_refresh_s"])))
    print("the picture window is live, with the reference picture laid faintly over it: slide the sheet and set the puck until the line\n"
          "and the puck lie on their faint copies. Do not touch the arm: its motors are on.")
    print("click the picture window, then Enter there lets the model push; Esc there or in this terminal ends a run")
    is_missing_picture_told = False
    is_other_size_told = False
    try:
        while True:
            picture, problem = None, "what came is not a picture"
            try:
                picture_bytes = with_interrupt_kept(read_picture_bytes)
                picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR) if picture_bytes else None
            except Exception as error:
                problem = f"{type(error).__name__}: {(str(error).splitlines() or [''])[0]}"
            if picture is not None:
                if picture.shape[:2] != ghost.shape[:2] and not is_other_size_told:
                    # a reference of another size is stretched to fit; said once, since it may be the wrong file
                    print(f"note: the reference picture is {ghost.shape[1]} x {ghost.shape[0]}, the camera's picture {picture.shape[1]} x {picture.shape[0]}: "
                          "the reference is stretched to fit. Is it a picture of this camera?")
                    is_other_size_told = True
                shown = ghost_blend(picture, ghost, settings["ghost_strength"])
                band_text = "Enter = let the model push, Esc = do not start"
            else:
                if not is_missing_picture_told:
                    print(f"a camera picture could not be read ({problem}); the window shows the reference alone until one comes. "
                          "Esc in the picture window ends the wait")
                    is_missing_picture_told = True
                # the reference alone, so that a window is there for Esc and a still picture is not taken for a live one
                shown = ghost.copy()
                band_text = "NO CAMERA PICTURE: Esc = do not start"
            cv2.rectangle(shown, (0, 0), (shown.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(shown, band_text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            show(shown)
            key = read_key(wait_ms)
            if key == "escape" or (read_terminal_key is not None and read_terminal_key() == "escape"):
                return False
            if key == "enter" and picture is not None:
                return True
    except KeyboardInterrupt:
        print("Ctrl+C: taken as Esc, the run is not started")
        return False



def command_line(argument_list: Optional[List[str]] = None) -> tuple:
    """
    Read the command line: the part to run and the settings its flags override.

    Parameters
    ----------
    argument_list : list of str, optional
        The arguments; by default the ones the script was started with.

    Returns
    -------
    tuple of argparse.Namespace and dict
        The arguments as read, and the overrides of MODEL_MODE_PARAMETERS: only the settings a flag was given for, so
        that no flag means no override. --step-cm is refused together with the step length rule "predicted" and
        outside "step_cm_allowed".
    """

    parser = argparse.ArgumentParser(description = "The model mode: the robot pushes by itself with a trained model.")
    parser.add_argument("part", choices = ["executor", "replay", "simulated", "rehearsal", "robot"])
    parser.add_argument("--label", default = None,
                        help = "a short name for the run's log folder (default: model_1 for a robot run, simulated for a simulated one, "
                               "rehearsal for a rehearsal)")
    parser.add_argument("--run-name", default = None, help = "the recipe of the model, a key of TRAINING_RUNS (default: model 1)")
    parser.add_argument("--checkpoint", nargs = "*", default = None, help = "the model's .pt file; several make a team")
    parser.add_argument("--metrics", default = None, help = "a metrics.json of that recipe")
    parser.add_argument("--twist-gain", type = float, default = None, help = "multiply the predicted twist (1.0 = as predicted)")
    parser.add_argument("--step-length", choices = ["fixed", "predicted"], default = None,
                        help = "fixed: always step_m; predicted: the length the model asks for, at most step_most_m")
    parser.add_argument("--length-gain", type = float, default = None, help = "multiply the predicted length (with --step-length predicted)")
    parser.add_argument("--step-budget", type = int, default = None, help = "the most steps of a run")
    parser.add_argument("--step-cm", type = float, default = None,
                        help = "the fixed step's length in centimetres, for example 0.65 (default: step_m; not with --step-length predicted)")
    parser.add_argument("--stall", choices = ["end", "log"], default = None,
                        help = "what the stall rule does: end the run (the default), or only log it and go on")
    parser.add_argument("--place-help", action = "store_true",
                        help = "robot part and rehearsal: show the tip's offset from the recorded start while the arm is guided by hand")
    parser.add_argument("--ghost", default = None, metavar = "PATH",
                        help = "robot part and rehearsal: lay this reference picture faintly over a live picture in the go-ahead window")
    parser.add_argument("--fresh-keys", action = "store_true",
                        help = "robot part and rehearsal: keys pressed before the first prompt is on the screen do not answer it, Esc ends a run "
                               "also when further keys wait in the terminal, and Ctrl+C is held back until the robot has answered")
    parser.add_argument("--run-folder", default = None, metavar = "PATH",
                        help = "log the run in this folder instead of run_folder (tests and rehearsals)")
    arguments = parser.parse_args(argument_list)
    if arguments.step_cm is not None:
        if (arguments.step_length or MODEL_MODE_PARAMETERS["step_length_rule"]) == "predicted":
            parser.error("--step-cm sets the length of the fixed step and does not go together with the step length rule \"predicted\" "
                         "(--step-length predicted), where the model's own length is used; leave one of the two out")
        least_cm, most_cm = MODEL_MODE_PARAMETERS["step_cm_allowed"]
        if not least_cm <= arguments.step_cm <= most_cm:
            parser.error(f"--step-cm {arguments.step_cm!r} is not a step length between {least_cm:g} and {most_cm:g} cm (\"step_cm_allowed\" in "
                         "config.py); it is given in centimetres, for example 0.65")
    overrides = {key: value for key, value in (("run_name", arguments.run_name), ("metrics_path", arguments.metrics),
                                               ("checkpoint_path", [Path(value) for value in arguments.checkpoint] if arguments.checkpoint else None),
                                               ("twist_gain", arguments.twist_gain), ("step_length_rule", arguments.step_length),
                                               ("step_length_gain", arguments.length_gain), ("step_budget", arguments.step_budget),
                                               ("step_m", None if arguments.step_cm is None else round(arguments.step_cm / 100.0, 6)),
                                               ("stall_action", arguments.stall),
                                               ("is_place_help_shown", True if arguments.place_help else None),
                                               ("ghost_path", Path(arguments.ghost) if arguments.ghost else None),
                                               ("is_typed_ahead_discarded", True if arguments.fresh_keys else None),
                                               ("run_folder", Path(arguments.run_folder) if arguments.run_folder else None))
                 if value is not None}
    return arguments, overrides



def main() -> None:

    arguments, overrides = command_line()
    if arguments.part not in ("robot", "rehearsal") and (arguments.place_help or arguments.ghost or arguments.fresh_keys):
        print("note: --place-help, --ghost and --fresh-keys act in the robot part and its rehearsal only; they do nothing here")

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
        policy = ModelPolicy(settings["run_name"], settings["checkpoint_path"], settings["metrics_path"])
        print(json.dumps(simulated_run(overrides, policy, arguments.label or "simulated"), indent = 1, default = str))
    elif arguments.part == "rehearsal":
        print(json.dumps(rehearsal_run(arguments.label or "rehearsal", overrides), indent = 1, default = str))
    else:
        print(json.dumps(robot_run(arguments.label or "model_1", overrides), indent = 1, default = str))



if __name__ == "__main__":
    main()
