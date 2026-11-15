"""
Shared loaders for the twist and heading analysis of the robot test of 11.11.26 (analyst key: twist_and_heading).

Nothing here connects to the robot: it reads the logs of the ten runs (03_data/model_runs), the training index
(03_data/processed/v01_training_index) through training_targets.load_training_frames with the audit's cleaning, and the
saved predictions of the fold models (04_runs/gpu_ki014). Every script of this folder runs from this folder:
    D:/AI_Project/.venv/Scripts/python.exe 01_twist_per_cm.py
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas

EVIDENCE_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = EVIDENCE_FOLDER.parents[5]
CODE_FOLDER = ROOT_FOLDER / "02_code"
if str(CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(CODE_FOLDER))

from training_targets import load_training_frames  # noqa: E402

ANALYSIS_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "trained_runs_folder": ROOT_FOLDER / "04_runs" / "gpu_ki014",
    # the ten robot runs of 11.11.26 in order: number, folder, model, sheet, what happened (brief.md), and the plan
    # whose sheet placement Emranul named (None: not as in a plan)
    "robot_runs": [
        {"number": 1, "folder": "2026-11-11_172021_check_no_puck", "model": "A", "sheet": "path_01", "outcome": "reached the end", "plan": None},
        {"number": 2, "folder": "2026-11-11_172421_E_sheet01_run1", "model": "E", "sheet": "path_01", "outcome": "reached the end", "plan": None},
        {"number": 3, "folder": "2026-11-11_172743_E_sheet02_run1", "model": "E", "sheet": "path_02", "outcome": "failed (lost the line, Esc)", "plan": None},
        {"number": 4, "folder": "2026-11-11_173036_E_sheet02_run2", "model": "E", "sheet": "path_02", "outcome": "reached the end", "plan": "02-01"},
        {"number": 5, "folder": "2026-11-11_173549_E_sheet03_run1", "model": "E", "sheet": "path_03", "outcome": "failed (overshot the first bend)", "plan": "03-01"},
        {"number": 6, "folder": "2026-11-11_174108_E_sheet04_run1", "model": "E", "sheet": "path_04", "outcome": "failed (stuck at the hairpin)", "plan": "04-01"},
        {"number": 7, "folder": "2026-11-11_174448_E_sheet05_run1", "model": "E", "sheet": "path_05", "outcome": "reached the end", "plan": "05-01"},
        {"number": 8, "folder": "2026-11-11_174936_E_sheet06_run1", "model": "E", "sheet": "path_06", "outcome": "failed (left the dashed loop)", "plan": "06-01"},
        {"number": 9, "folder": "2026-11-11_175203_E_sheet03_run2_no_hoodie", "model": "E", "sheet": "path_03", "outcome": "failed (overshot the first bend)", "plan": "03-01"},
        {"number": 10, "folder": "2026-11-11_175544_B_team5_sheet03", "model": "B", "sheet": "path_03", "outcome": "failed (overshot the first bend)", "plan": "03-01"}
    ],
    # the fold models scored on their held-out sheets: label and run folder
    "fold_runs": {
        "A recipe (exact)": "2026-11-09_resnet18_final_candidate",
        "A recipe seed 2": "2026-11-09_resnet18_final_candidate_seed_2",
        "chunk": "2026-11-10_resnet18_final_candidate_chunk",
        "E recipe (chunk, stop 25)": "2026-11-10_resnet18_chunk_stop_weight_25",
        "E recipe seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2",
        "chunk twist focus": "2026-11-10_resnet18_chunk_twist_focus",
        "chunk twist focus seed 2": "2026-11-10_resnet18_chunk_twist_focus_seed_2"
    },
    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"],
    "window_cm": 5.0
}



def wrapped_deg(angle_deg):
    """Return an angle (or array of angles) in degrees wrapped into -180 to 180."""

    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0



def demonstration_frames() -> pandas.DataFrame:
    """
    The demonstrations as the models were trained on them (the cleaning of the audit: 18,032 rows, 209 episodes),
    sorted by episode and step, with added columns: "next_is_consecutive" (the next row is the next recorded step of
    the same episode), "step_cm" (the labelled step length), "arc_cm" (the path length of the tip from the first kept
    row of the episode, from the tip positions of the arm model) and "joint_4_deg".
    """

    frames = load_training_frames({"is_audit_cleaning": True})
    is_same_episode = (frames["episode_id"].shift(-1) == frames["episode_id"])
    frames["next_is_consecutive"] = (is_same_episode & (frames["row_in_episode"].shift(-1) == frames["row_in_episode"] + 1)).to_numpy()
    frames["step_cm"] = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    moved_cm = 100.0 * numpy.hypot(frames.groupby("episode_id")["tip_x_m"].diff().fillna(0.0), frames.groupby("episode_id")["tip_y_m"].diff().fillna(0.0))
    frames["arc_cm"] = moved_cm.groupby(frames["episode_id"]).cumsum()
    frames["joint_4_deg"] = numpy.degrees(frames["joint_4"])
    return frames



def robot_run_steps(
        run: Dict,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    The log of one robot run (steps.csv) with added columns: "moved_cm" (the measured move of the tip from this picture
    to the next, from the logged joints), "arc_cm" (the path length up to this picture), "joint_4_deg",
    "twist_done_deg" (the measured joint-4 change to the next picture) and "asked_cm" (the length of the step the model
    asked for).
    """

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    steps = pandas.read_csv(Path(settings["model_runs_folder"]) / run["folder"] / "steps.csv")
    steps["joint_4_deg"] = numpy.degrees(steps["joint_4"])
    moved_cm = 100.0 * numpy.hypot(steps["tip_x"].diff().shift(-1), steps["tip_y"].diff().shift(-1))
    steps["moved_cm"] = moved_cm.fillna(0.0)
    steps["arc_cm"] = steps["moved_cm"].cumsum().shift(1).fillna(0.0)
    steps["twist_done_deg"] = steps["joint_4_deg"].diff().shift(-1).fillna(0.0)
    steps["asked_cm"] = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"])
    return steps



def fold_predictions(
        run_folder: str,
        frames: pandas.DataFrame,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    The saved predictions of a fold run on all six held-out sheets, joined to the demonstration rows by frame_index.
    Added columns: "predicted_twist_deg", "predicted_angle_deg", "predicted_stop" and, for chunk runs,
    "predicted_chunk_angle_deg".
    """

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    parts = []
    for sheet in settings["sheets"]:
        saved = numpy.load(Path(settings["trained_runs_folder"]) / run_folder / f"predictions_without_{sheet}.npz")
        part = pandas.DataFrame({"frame_index": saved["frame_index"], "predicted_twist_deg": saved["twist_deg"],
                                 "predicted_angle_deg": saved["angle_deg"], "predicted_stop": saved["stop_probability"]})
        if "angle_vector_deg" in saved.files:
            part["predicted_chunk_angle_deg"] = saved["angle_vector_deg"]
        parts.append(part)
    predictions = pandas.concat(parts, ignore_index = True)
    joined = frames.merge(predictions, on = "frame_index", how = "inner", validate = "one_to_one")
    return joined.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)



class Report:

    def __init__(self, file_name: str):
        """Print lines to the screen and keep them for ``file_name`` in the evidence folder."""

        self.path = EVIDENCE_FOLDER / file_name
        self.lines: List[str] = []


    def say(self, text: str = "") -> None:
        """Print ``text`` and remember it."""

        print(text)
        self.lines.append(str(text))


    def save(self) -> None:
        """Write everything said to the file of the report."""

        self.path.write_text("\n".join(self.lines) + "\n", encoding = "utf-8")
