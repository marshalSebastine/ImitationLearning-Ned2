"""
A trained policy as the decider of the recording tool (Marshal, 14.11.26).

``PolicyDriver`` loads a checkpoint that train_policy.py wrote, prepares each live picture exactly as training did
(shrunk with cv2.INTER_AREA to the cache size, RGB, centre crop, ImageNet normalisation), asks the policy, and turns
its answer into the gamepad command the recording tool already knows how to carry out safely: a stick push of one
1 cm step in the picture's frame and a forearm turn. record_episodes.py --mode policy uses it; every step is shown
and waits for Enter unless the operator lets it run on.

Run from 02_code, robot connected, nothing else connected to it, stop button in reach (see the checklist
06_deliverables/research_notes/Policy runs/first_policy_run_checklist.md):
    python record_episodes.py --operator emranul --mode policy --checkpoint ../04_runs/<run>/checkpoints/best.pt --path-id path_03_two_curve
Without the robot:
    python record_episodes.py --operator marshal --mode policy --checkpoint <best.pt> --simulated
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy
import torch

from config import POLICY_RUN_PARAMETERS, RECORDING_PARAMETERS
from policy_data import picture_batch
from policy_models import build_policy_model
from train_policy import choose_device, class_counts


class PolicyDriver:
    """
    One trained policy, ready to choose the next step from a live picture.

    Parameters
    ----------
    checkpoint_path : str or Path
        A ``best.pt`` written by train_policy.py.
    run_parameters : dict, optional
        Overrides for ``POLICY_RUN_PARAMETERS``.
    """

    def __init__(
            self,
            checkpoint_path,
            run_parameters: Optional[Dict] = None
    ):

        self.parameters = dict(POLICY_RUN_PARAMETERS)
        self.parameters.update(run_parameters or {})
        self.checkpoint_path = Path(checkpoint_path)
        self.device = choose_device(self.parameters["device"])
        checkpoint = torch.load(self.checkpoint_path, map_location = self.device, weights_only = False)
        self.model_name = checkpoint["model_name"]
        self.training_parameters = checkpoint["training_parameters"]
        # The checkpoint already holds the trained weights; nothing is downloaded.
        self.model = build_policy_model(self.model_name, class_counts(self.training_parameters), checkpoint["model_parameters"],
                                        is_pretrained = False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.exact_mean = torch.tensor(checkpoint["exact_mean"], device = self.device)
        self.exact_std = torch.tensor(checkpoint["exact_std"], device = self.device)
        self.validation_score = checkpoint.get("validation", {}).get("hard_sheet_score")
        self.history_length = getattr(self.model, "history_length", 1)
        self.generator = torch.Generator(device = self.device).manual_seed(0)
        self.start_episode()


    def description(self) -> Dict:
        """What drives the robot, written into the episode's summary."""

        return {"checkpoint": str(self.checkpoint_path), "model_name": self.model_name, "device": str(self.device),
                "validation_hard_sheet_score": self.validation_score, "turn_deg": self.parameters["turn_deg"],
                "step_m": self.parameters["step_m"]}


    def start_episode(self) -> None:
        """Forget the pictures and moves of the last episode (the history policy looks back within one episode only)."""

        self.past_pictures: List[numpy.ndarray] = []
        self.previous_move = len(self.training_parameters["move_classes"])
        self.pending = None


    def prepare_picture(self, picture_bytes: bytes) -> numpy.ndarray:
        """A camera JPEG shrunk as in training: cv2.INTER_AREA to the cache size, BGR to RGB; uint8 (height, width, 3)."""

        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        width, height = self.training_parameters["cache_size"]
        return cv2.cvtColor(cv2.resize(picture, (width, height), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


    @torch.no_grad()
    def decide(
            self,
            picture_bytes: bytes,
            step: int
    ) -> Dict:
        """
        The policy's next step for this picture.

        A decision is only remembered as the "previous" one (picture and move) once the episode has gone on to the next
        step, so a step the robot refused and asks about again does not count twice.

        Returns
        -------
        dict
            "action" ("stop" or "pad"), "pad_command" (the gamepad command for "pad"), "move" and "forearm" (the
            names, as in the training labels) and "text" (one line for the screen).
        """

        if self.pending is not None and step > self.pending["step"]:
            self.past_pictures.insert(0, self.pending["picture"])
            del self.past_pictures[max(0, self.history_length - 1):]
            self.previous_move = self.pending["move_index"]
        prepared = self.prepare_picture(picture_bytes)
        parameters = self.training_parameters
        move_classes, forearm_classes = parameters["move_classes"], parameters["forearm_classes"]

        if self.model_name == "history_classifier":
            # The first picture of an episode stands in for the earlier ones it does not have, as in training.
            stack = [prepared] + self.past_pictures
            stack += [stack[-1]] * (self.history_length - len(stack))
            pictures = torch.from_numpy(numpy.stack(stack)).to(self.device)
            batch = picture_batch(pictures, torch.arange(self.history_length, device = self.device)[None], False, parameters, self.device)
            outputs = self.model(batch, torch.tensor([self.previous_move], device = self.device))
        else:
            pictures = torch.from_numpy(prepared[None]).to(self.device)
            batch = picture_batch(pictures, torch.zeros(1, dtype = torch.long, device = self.device), False, parameters, self.device)
            outputs = None if self.model.kind == "diffusion" else self.model(batch)

        if self.model.kind == "classifier":
            move_probabilities = torch.softmax(outputs["move"][0].float(), dim = 0)
            move_index = int(move_probabilities.argmax())
            forearm = forearm_classes[int(outputs["forearm"][0].argmax())]
            move = move_classes[move_index]
            angle = parameters["action_angles_deg"].get(move)
            forward_m = self.parameters["step_m"] * math.cos(math.radians(angle)) if angle is not None else 0.0
            left_m = self.parameters["step_m"] * math.sin(math.radians(angle)) if angle is not None else 0.0
            turn_deg = {"turn_left": self.parameters["turn_deg"], "turn_right": -self.parameters["turn_deg"]}.get(forearm, 0.0)
            confidence = f"{float(move_probabilities[move_index]):.2f}"
        else:
            if self.model.kind == "diffusion":
                first = self.model.sample(batch, generator = self.generator)[0, 0]
                exact_scaled, is_stop = first[:-1], bool(first[-1] > 0)
            else:
                exact_scaled = outputs["exact"][0] if self.model.kind == "regressor" else outputs["exact"][0, 0]
                stop_logit = outputs["stop"][0] if self.model.kind == "regressor" else outputs["stop"][0, 0]
                is_stop = bool(stop_logit > 0)
            exact = (exact_scaled.float() * self.exact_std + self.exact_mean).tolist()
            forward_m, left_m, turn_deg = exact[0], exact[1], exact[2]
            if is_stop:
                move = "stop"
            else:
                picture_angle = math.degrees(math.atan2(left_m, forward_m))
                angles = parameters["action_angles_deg"]
                move = min(angles, key = lambda name: abs((picture_angle - angles[name] + 180.0) % 360.0 - 180.0))
            threshold = parameters["forearm_label_deg"]
            forearm = "turn_left" if turn_deg >= threshold else ("turn_right" if turn_deg <= -threshold else "none")
            move_index = move_classes.index(move)
            confidence = "-"

        self.pending = {"step": step, "picture": prepared, "move_index": move_index}
        result = {"move": move, "forearm": forearm, "confidence": confidence}
        if move == "stop":
            result.update({"action": "stop", "pad_command": None, "text": f"policy: STOP (confidence {confidence})"})
            return result
        result.update({"action": "pad", "pad_command": self.pad_command(forward_m, left_m, turn_deg),
                       "text": f"policy: {move}, forearm {forearm} {turn_deg:+.0f} deg (confidence {confidence})"})
        return result


    def pad_command(
            self,
            forward_m: float,
            left_m: float,
            turn_deg: float
    ) -> Dict[str, float]:
        """
        The gamepad command for a tip step (metres, picture frame) and a forearm turn (degrees, + is turn_left).

        The recording tool steps "pad_step_m" times how far the left stick is pushed, forward = stick up and left =
        stick left, and turns the forearm by "pad_twist_deg" per full right-stick push, a push to the left (negative)
        turning it towards +.
        """

        length = math.hypot(forward_m, left_m)
        push = min(1.0, max(self.parameters["minimum_step_fraction"], length / RECORDING_PARAMETERS["pad_step_m"])) if length > 1e-9 else 0.0
        move_x = -push * left_m / length if length > 1e-9 else 0.0
        move_y = push * forward_m / length if length > 1e-9 else 0.0
        twist = max(-1.0, min(1.0, -turn_deg / RECORDING_PARAMETERS["pad_twist_deg"]))
        return {"pad_move_x": move_x, "pad_move_y": move_y, "pad_twist": twist, "pad_lean": 0.0, "pad_hand": 0.0,
                "pad_up": 0.0, "pad_down": 0.0}
