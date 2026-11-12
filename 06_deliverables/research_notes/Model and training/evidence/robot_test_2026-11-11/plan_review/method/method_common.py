"""
Shared readers for the method critic's checks of the plan of 11.11.26 (plan_review/method).

Nothing here connects to the robot, trains a model or changes a file outside this folder: the scripts read the training
index, the cleaned rows (training_targets.load_training_frames), the fold outputs the step-length analyst saved (checked
by the verifier against an independent forward pass) and the twist analyst's puck positions.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy
import pandas

REVIEW_FOLDER = Path(__file__).resolve().parent
EVIDENCE_FOLDER = REVIEW_FOLDER.parents[1]
ROOT_FOLDER = REVIEW_FOLDER.parents[6]
CODE_FOLDER = ROOT_FOLDER / "02_code"
if str(CODE_FOLDER) not in sys.path:
    sys.path.insert(0, str(CODE_FOLDER))

METHOD_REVIEW_PARAMETERS = {
    "run_name": "resnet18_chunk_stop_weight_25",
    "fold_outputs": {"E": EVIDENCE_FOLDER / "step_length" / "fold_outputs_E_recipe.npz",
                     "A": EVIDENCE_FOLDER / "step_length" / "fold_outputs_A_recipe.npz"},
    "puck_positions": EVIDENCE_FOLDER / "twist_and_heading" / "puck_positions_demonstrations.csv",
    # one recorded step: a waypoint is cut after this much tip travel or this much turn on any joint (config.py,
    # RECORDING_PARAMETERS "exact_waypoint_m" and "exact_waypoint_joint_deg")
    "unit_length_cm": 1.0,
    "unit_turn_deg": 5.0,
    # the picture's heading turns by about this many degrees per degree of forearm twist, with the opposite sign
    # (the check: -0.90 to -0.93)
    "heading_per_twist": 0.92,
    "joint_4_most_deg": 117.0
}



def merged_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``METHOD_REVIEW_PARAMETERS`` with any overrides merged over it."""

    merged = dict(METHOD_REVIEW_PARAMETERS)
    merged.update(parameters or {})
    return merged



def cleaned_frames(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    Return the rows the models of 10.11.26 were trained and scored on (the audit cleaning), sorted by episode and step.

    Parameters
    ----------
    parameters : dict, optional
        Overrides of ``METHOD_REVIEW_PARAMETERS`` ("run_name" names the recipe whose target settings are used).

    Returns
    -------
    pandas.DataFrame
        One row per kept picture, as ``training_targets.load_training_frames`` gives it.
    """

    from config import TRAINING_TARGET_PARAMETERS
    from train_policy import merged_run_settings
    from training_targets import load_training_frames

    settings_here = merged_parameters(parameters)
    settings = merged_run_settings(settings_here["run_name"])
    target_parameters = dict(TRAINING_TARGET_PARAMETERS)
    target_parameters.update({key: value for key, value in settings.items() if key in TRAINING_TARGET_PARAMETERS})
    return load_training_frames(target_parameters)



def wrapped_degrees(angle_deg: numpy.ndarray) -> numpy.ndarray:
    """Return angles wrapped into -180 to 180 degrees."""

    return (numpy.asarray(angle_deg) + 180.0) % 360.0 - 180.0
