"""Verifier's own loaders (independent of the analyst's common.py). Reads only; nothing connects to the robot."""
import sys
from pathlib import Path
import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
from training_targets import load_training_frames  # the project's own cleaning: the rows the models train on

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
TRAINED = ROOT / "04_runs" / "gpu_ki014"
RUNS = {1: ("2026-11-11_172021_check_no_puck", "A", "path_01"), 2: ("2026-11-11_172421_E_sheet01_run1", "E", "path_01"),
        3: ("2026-11-11_172743_E_sheet02_run1", "E", "path_02"), 4: ("2026-11-11_173036_E_sheet02_run2", "E", "path_02"),
        5: ("2026-11-11_173549_E_sheet03_run1", "E", "path_03"), 6: ("2026-11-11_174108_E_sheet04_run1", "E", "path_04"),
        7: ("2026-11-11_174448_E_sheet05_run1", "E", "path_05"), 8: ("2026-11-11_174936_E_sheet06_run1", "E", "path_06"),
        9: ("2026-11-11_175203_E_sheet03_run2_no_hoodie", "E", "path_03"), 10: ("2026-11-11_175544_B_team5_sheet03", "B", "path_03")}


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


def demos():
    """Cleaned demonstration rows, sorted, with the joint-4 angle in degrees and the labelled step length in cm."""
    frames = load_training_frames({"is_audit_cleaning": True})
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    frames["j4"] = numpy.degrees(frames["joint_4"])
    frames["len_cm"] = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    frames["twist"] = frames["forearm_change_deg"].fillna(0.0)
    return frames


def run_log(number):
    """One robot run: per picture the measured tip move and joint-4 change up to the NEXT picture."""
    folder, model, sheet = RUNS[number]
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    steps["j4"] = numpy.degrees(steps["joint_4"])
    dx = steps["tip_x"].shift(-1) - steps["tip_x"]
    dy = steps["tip_y"].shift(-1) - steps["tip_y"]
    steps["moved_cm"] = (100.0 * numpy.hypot(dx, dy)).fillna(0.0)
    steps["twist_done"] = (steps["j4"].shift(-1) - steps["j4"]).fillna(0.0)
    steps["asked_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
    return steps
