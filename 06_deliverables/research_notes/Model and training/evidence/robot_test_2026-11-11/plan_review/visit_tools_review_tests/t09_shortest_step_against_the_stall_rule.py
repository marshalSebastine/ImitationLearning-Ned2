"""Review attack 8: the shortest allowed fixed step against the stall rule, on the simulated robot with a model that drives straight on."""

import contextlib
import io
import tempfile

import safe_import
from safe_import import model_mode
from config import MODEL_MODE_PARAMETERS


class StraightOn:

    run_name = "straight_on_stand_in"
    checkpoint_paths = []

    def decide(self, picture_bytes):

        return {"tip_forward_cm": 0.8, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 0.0}


print("stall rule of config.py:", {name: MODEL_MODE_PARAMETERS[name] for name in ("stall_steps", "stall_extent_m", "stall_twist_deg", "stall_action", "step_cm_allowed")})
with tempfile.TemporaryDirectory() as folder:
    for step_cm in ("0.1", "0.103", "0.104", "0.11", "0.2", "0.65"):
        overrides = model_mode.command_line(["simulated", "--step-cm", step_cm])[1]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            summary = model_mode.simulated_run(dict(overrides, run_folder = folder, step_budget = 45), StraightOn(), f"step_{step_cm}")
        print(f"--step-cm {step_cm}: a model that always drives straight on ends as '{summary['ending']}' after {summary['pictures']} pictures and {summary['moves']} moves")
