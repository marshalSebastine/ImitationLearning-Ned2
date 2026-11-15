"""Review attack 2: the stall rule that only logs, and the fixed step's length, on the simulated robot with stand-in models."""

import contextlib
import io
import json
import math
import tempfile
from pathlib import Path

import numpy
import pandas

import safe_import
from safe_import import model_mode
from config import MODEL_MODE_PARAMETERS


class Hesitating:

    run_name = "hesitating_stand_in"
    checkpoint_paths = []

    def decide(self, picture_bytes):

        return {"tip_forward_cm": 0.02, "tip_left_cm": 0.0, "twist_deg": 0.0, "lean_deg": 0.0, "hand_deg": 0.0, "angle_deg": 0.0, "stop_probability": 0.0}


class Scripted:

    run_name = "scripted_stand_in"
    checkpoint_paths = []

    def __init__(self, decisions):

        self.decisions, self.count = decisions, 0

    def decide(self, picture_bytes):

        answer = self.decisions(self.count)
        self.count += 1
        return answer


def decision(forward, left, twist, stop):

    return {"tip_forward_cm": forward, "tip_left_cm": left, "twist_deg": twist, "lean_deg": 0.0, "hand_deg": 0.0,
            "angle_deg": math.degrees(math.atan2(left, forward)), "stop_probability": stop}


def run(settings, policy, label = "x"):

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        summary = model_mode.simulated_run(settings, policy, label)
    return summary, printed.getvalue()


with tempfile.TemporaryDirectory() as folder:
    base = {"run_folder": folder, "step_length_rule": "predicted", "stall_steps": 5, "step_budget": 20}

    print("--- 1. log: goes on to the budget; fired at picture ...")
    summary, printed = run(dict(base, stall_action = "log"), Hesitating())
    print(summary["ending"], summary["pictures"], summary["moves"], "stall_logged_at_picture =", summary["stall_logged_at_picture"], "| printed lines with 'stall':", [line for line in printed.splitlines() if "stall" in line])
    print("fixed_step_cm under predicted:", summary["fixed_step_cm"], "step_length_rule:", summary["step_length_rule"])
    written = json.loads((Path(summary["folder"]) / "summary.json").read_text(encoding = "utf-8"))
    print("summary.json keys:", sorted(written))

    print("--- 2. end (default): stalled at ...")
    summary, printed = run(dict(base), Hesitating())
    print(summary["ending"], summary["pictures"], summary["moves"], summary["stall_logged_at_picture"], summary["stall_action"])

    print("--- 3. a value that is neither end nor log, given as a setting (a typing mistake in config.py)")
    for odd in ("Log", "LOG", "logs", "", None, "off", True):
        summary, printed = run(dict(base, stall_action = odd), Hesitating())
        print(repr(odd), "->", summary["ending"], summary["pictures"], "summary stall_action:", repr(summary["stall_action"]))

    print("--- 4. log, then the model stalls, moves on, stalls again: named once, the first picture")
    # hesitate 8 pictures, move 12 pictures 1 cm each, hesitate again
    policy = Scripted(lambda step: decision(0.02, 0.0, 0.0, 0.0) if step < 8 or step >= 20 else decision(1.0, 0.0, 0.0, 0.0))
    summary, printed = run(dict(base, stall_action = "log", step_budget = 40), policy)
    print(summary["ending"], summary["pictures"], "stall_logged_at_picture =", summary["stall_logged_at_picture"], "| count of printed stall lines:", printed.count("the stall rule fired"))
    steps = pandas.read_csv(Path(summary["folder"]) / "steps.csv")
    print("rows:", len(steps), "frames:", len(list((Path(summary["folder"]) / "frames").glob("*.jpg"))))

    print("--- 5. log, and the stop rule still ends the run after the stall fired")
    policy = Scripted(lambda step: decision(0.02, 0.0, 0.0, 0.9 if step >= 9 else 0.0))
    summary, printed = run(dict(base, stall_action = "log", step_budget = 40), policy)
    print(summary["ending"], summary["pictures"], summary["stall_logged_at_picture"])

    print("--- 6. log with the FIXED rule and a twist-in-place model: the tip marches on (1 cm steps); with log nothing ends it but the budget or a refusal")
    policy = Scripted(lambda step: decision(0.02, 0.0, 0.0, 0.0))
    summary, printed = run({"run_folder": folder, "stall_steps": 5, "step_budget": 60, "stall_action": "log"}, policy)
    print(summary["ending"], summary["pictures"], summary["moves"], summary["stall_logged_at_picture"], [line for line in printed.splitlines() if "refused" in line or "stall" in line])

    print("--- 7. step lengths at the allowed ends and the default against --step-cm 1.0")
    class Steady:
        run_name = "steady"
        checkpoint_paths = []
        def decide(self, picture_bytes):
            return decision(0.8, 0.1, 2.0, 0.0)
    rows = {}
    for flags in ([], ["--step-cm", "1.0"], ["--step-cm", "0.1"], ["--step-cm", "0.65"], ["--step-cm", "0.33"]):
        overrides = model_mode.command_line(["simulated"] + flags)[1]
        summary, printed = run(dict(overrides, run_folder = folder, step_budget = 6), Steady(), "s" + "_".join(flags).replace("-", ""))
        steps = pandas.read_csv(Path(summary["folder"]) / "steps.csv")
        moves = numpy.hypot(numpy.diff(steps["tip_x"]), numpy.diff(steps["tip_y"]))
        rows[tuple(flags)] = steps.drop(columns = ["unix_time", "decision_ms"])
        print(flags, summary["ending"], "fixed_step_cm", summary["fixed_step_cm"], "tip moves mm", numpy.round(1000 * moves, 3), "step_cm column", sorted(set(steps["step_cm"])))
    print("default and --step-cm 1.0 give the same rows:", rows[()].equals(rows[("--step-cm", "1.0")]))

    print("--- 8. the window text for odd step lengths")
    shown = []
    model_mode.show_in_window = shown.append
    for rule, step_cm in (("fixed", 1.0), ("fixed", 0.65), ("fixed", 0.5), ("fixed", 0.1), ("fixed", 0.333), ("predicted", 0.649), ("predicted", 0.0)):
        runner = model_mode.ModelRun(None, None, {"step_length_rule": rule}, is_shown = False)
        row = {"step": 3, "action": "model_step", "model_stop_probability": 0.1, "model_twist_deg": 1.0, "model_angle_deg": 0.0, "step_cm": step_cm}
        text = f"{step_cm:.2f}" if rule != "predicted" and abs(step_cm - round(step_cm, 1)) > 1e-6 else f"{step_cm:.1f}"
        print(rule, step_cm, "->", text)
