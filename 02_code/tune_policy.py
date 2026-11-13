"""
Optuna search over the settings of one policy family, the factorised classifier (Marshal, 13.11.26).

Each trial trains for ``OPTUNA_PARAMETERS["epochs"]`` epochs with settings Optuna picks, and is scored by the
validation hard-sheet score (sheets 03, 04 and 06), the same score train_policy.py uses. The median pruner stops a trial
that falls behind the others. The study lives in ``04_runs/<batch>_optuna.db`` (SQLite), so a night that stops can be
continued with the same command. When the trials or the hours are used up, the best settings are trained in full as
``<batch>_factorised_classifier_tuned_seed1`` and the search is written to ``<batch>_optuna.md``.

The search tunes one family only: which family is best is decided on the robot, not by a validation number.

Run from 02_code:
    python tune_policy.py --batch 2026-11-13_overnight
"""

import argparse
import json
import time
from datetime import date
from typing import Dict, Optional

import optuna

from config import OPTUNA_PARAMETERS
from policy_data import merged_training_parameters
from train_policy import train_one_run


def suggested_settings(
        trial: optuna.Trial,
        search: Dict
) -> Dict:
    """The training and model settings of one trial, as Optuna suggests them within the ranges in ``search``."""

    score_1_weight = trial.suggest_float("score_1_weight", *search["score_1_weight"])
    training = {
        "learning_rate": trial.suggest_float("learning_rate", *search["learning_rate"], log = True),
        "weight_decay": trial.suggest_float("weight_decay", *search["weight_decay"], log = True),
        "class_weight_power": trial.suggest_float("class_weight_power", *search["class_weight_power"]),
        "colour_jitter": trial.suggest_float("colour_jitter", *search["colour_jitter"]),
        "batch_size": trial.suggest_categorical("batch_size", search["batch_size"]),
        "score_weights": {"1": score_1_weight, "2": 1.0, "3": 1.0}
    }
    model = {"dropout": trial.suggest_float("dropout", *search["dropout"])}
    return {"training": training, "model": model}


def tune(
        batch_name: str,
        optuna_overrides: Optional[Dict] = None,
        training_overrides: Optional[Dict] = None,
        device_name: Optional[str] = None
) -> Dict:
    """
    Run (or continue) the Optuna study of a batch, then train the best settings in full.

    Parameters
    ----------
    batch_name : str
        Shared with the night's other runs; names the study file and the trial folders.
    optuna_overrides : dict, optional
        Overrides for ``OPTUNA_PARAMETERS`` (for example "trials" or "hours").
    training_overrides : dict, optional
        Training settings fixed for every trial (tests use these for a tiny dataset).
    device_name : str, optional
        "cuda", "mps" or "cpu"; chosen automatically when not given.

    Returns
    -------
    dict
        "best_settings", "best_value", "trials_finished" and the tuned run's metrics.
    """

    search = dict(OPTUNA_PARAMETERS)
    search.update(optuna_overrides or {})
    runs_folder = merged_training_parameters(training_overrides)["runs_folder"]
    runs_folder.mkdir(parents = True, exist_ok = True)
    study = optuna.create_study(
        study_name = batch_name, storage = f"sqlite:///{runs_folder / f'{batch_name}_optuna.db'}", load_if_exists = True,
        direction = "maximize", sampler = optuna.samplers.TPESampler(seed = 13_11_26),
        pruner = optuna.pruners.MedianPruner(n_startup_trials = search["startup_trials"], n_warmup_steps = 3))
    started = time.time()

    def objective(trial: optuna.Trial) -> float:
        settings = suggested_settings(trial, search)
        training = dict(training_overrides or {})
        training.update(settings["training"])
        training["epochs"] = search["epochs"]
        training["patience_epochs"] = search["epochs"]
        training["is_checkpoint_kept"] = False
        result = train_one_run(search["model_name"], 1, batch_name, training_overrides = training, model_overrides = settings["model"],
                               run_label = f"optuna_trial_{trial.number:03d}", device_name = device_name, trial = trial)
        if result["stopped_because"] == "pruned by Optuna":
            raise optuna.TrialPruned()
        return result["validation"]["hard_sheet_score"]

    finished = [trial for trial in study.trials if trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    remaining = max(0, search["trials"] - len(finished))
    if remaining:
        study.optimize(objective, n_trials = remaining, timeout = search["hours"] * 3600.0, catch = (RuntimeError,))
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("no Optuna trial finished; see the trial folders' history.csv")

    best = study.best_trial
    settings = suggested_settings(optuna.trial.FixedTrial(best.params), search)
    training = dict(training_overrides or {})
    training.update(settings["training"])
    tuned = train_one_run(search["model_name"], 1, batch_name, training_overrides = training, model_overrides = settings["model"],
                          run_label = f"{search['model_name']}_tuned", device_name = device_name, is_done_skipped = True)

    lines = [f"# Optuna search {batch_name}", "", f"Model: {search['model_name']}; {len(completed)} trials finished, "
             f"{len(study.trials) - len(completed)} pruned or failed; {(time.time() - started) / 3600:.2f} hours this time.", "",
             f"Best validation hard-sheet score in the search ({search['epochs']} epochs): {best.value:.3f} (trial {best.number})", "",
             "Best settings:", "", "```", json.dumps(best.params, indent = 2), "```", "",
             f"Trained in full with these settings: validation hard-sheet score {tuned['validation']['hard_sheet_score']:.3f}.", "",
             "| Trial | State | Score | Settings |", "|---|---|---|---|"]
    for trial in study.trials:
        value = f"{trial.value:.3f}" if trial.value is not None else "-"
        lines.append(f"| {trial.number} | {trial.state.name.lower()} | {value} | {json.dumps(trial.params)} |")
    (runs_folder / f"{batch_name}_optuna.md").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return {"best_settings": best.params, "best_value": best.value, "trials_finished": len(completed), "tuned": tuned}


def main() -> None:
    """Command line: run or continue the study of one batch."""

    parser = argparse.ArgumentParser(description = "Optuna search over the factorised classifier's settings.")
    parser.add_argument("--batch", default = f"{date.today().isoformat()}_trial")
    parser.add_argument("--trials", type = int, help = "total trials of the study (default from config)")
    parser.add_argument("--hours", type = float, help = "time limit for this call (default from config)")
    parser.add_argument("--device", help = "cuda, mps or cpu (default: the best available)")
    arguments = parser.parse_args()
    overrides = {key: value for key, value in (("trials", arguments.trials), ("hours", arguments.hours)) if value is not None}
    result = tune(arguments.batch, overrides, device_name = arguments.device)
    print(f"best settings {result['best_settings']}, search score {result['best_value']:.3f}, "
          f"tuned run {result['tuned']['validation']['hard_sheet_score']:.3f}")


if __name__ == "__main__":
    main()
