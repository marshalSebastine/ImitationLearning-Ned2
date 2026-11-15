"""
Optuna search over the settings of one policy (Marshal, 13.11.26; any policy and three final seeds since 14.11.26).

Each trial trains for ``OPTUNA_PARAMETERS["epochs"]`` epochs with settings Optuna picks, and is scored by the
validation hard-sheet score (sheets 03, 04 and 06), the same score train_policy.py uses; the unseen tracks are never
used to choose. The median pruner stops a trial that falls behind the others. The study lives in
``04_runs/<batch>_<model>_optuna.db`` (SQLite), so a stopped search carries on with the same command. When the trials or
the hours are used up, the best settings are trained in full with every seed of "final_seeds" as
``<batch>_<model>_optuna_best_seed<N>``, and the search is written to ``<batch>_<model>_optuna.md``. Training the best
settings again with several seeds shows whether the search found a real gain or a lucky trial.

The search tunes one family at a time: which family is best is decided on the robot, not by a validation number.

Run from 02_code:
    python tune_policy.py --batch 2026-11-14_tuning
    python tune_policy.py --batch 2026-11-14_tuning --model factorised_classifier --trials 40 --hours 3
"""

import argparse
import json
import time
from datetime import date
from typing import Dict, Optional

import numpy
import optuna

from config import OPTUNA_PARAMETERS, POLICY_MODEL_PARAMETERS
from policy_data import merged_training_parameters
from train_policy import train_one_run


def suggested_settings(
        trial: optuna.Trial,
        search: Dict,
        model_name: str
) -> Dict:
    """The training and model settings of one trial, as Optuna suggests them within the ranges in ``search``."""

    score_1_weight = trial.suggest_float("score_1_weight", *search["score_1_weight"])
    training = {
        "learning_rate": trial.suggest_float("learning_rate", *search["learning_rate"], log = True),
        "weight_decay": trial.suggest_float("weight_decay", *search["weight_decay"], log = True),
        "class_weight_power": trial.suggest_float("class_weight_power", *search["class_weight_power"]),
        "colour_jitter": trial.suggest_float("colour_jitter", *search["colour_jitter"]),
        "batch_size": trial.suggest_categorical("batch_size", search["batch_size"]),
        "sheet_balance_power": trial.suggest_float("sheet_balance_power", *search["sheet_balance_power"]),
        "score_weights": {"1": score_1_weight, "2": 1.0, "3": 1.0}
    }
    model = {"dropout": trial.suggest_float("dropout", *search["dropout"])}
    if "previous_move_dropout" in POLICY_MODEL_PARAMETERS[model_name]:
        model["previous_move_dropout"] = trial.suggest_float("previous_move_dropout", *search["previous_move_dropout"])
    return {"training": training, "model": model}


def tune(
        batch_name: str,
        optuna_overrides: Optional[Dict] = None,
        training_overrides: Optional[Dict] = None,
        device_name: Optional[str] = None
) -> Dict:
    """
    Run (or continue) the Optuna study of one policy in a batch, then train the best settings with every final seed.

    Parameters
    ----------
    batch_name : str
        Shared with the night's other runs; names the study file and the run folders.
    optuna_overrides : dict, optional
        Overrides for ``OPTUNA_PARAMETERS`` (for example "model_name", "trials" or "hours").
    training_overrides : dict, optional
        Training settings fixed for every trial (the dataset name; tests use these for a tiny dataset).
    device_name : str, optional
        "cuda", "mps" or "cpu"; chosen automatically when not given.

    Returns
    -------
    dict
        "best_settings", "best_value", "trials_finished", and "final": the retrained runs' metrics, one per seed.
    """

    search = dict(OPTUNA_PARAMETERS)
    search.update(optuna_overrides or {})
    model_name = search["model_name"]
    runs_folder = merged_training_parameters(training_overrides)["runs_folder"]
    runs_folder.mkdir(parents = True, exist_ok = True)
    study_label = f"{batch_name}_{model_name}"
    study = optuna.create_study(
        study_name = study_label, storage = f"sqlite:///{runs_folder / f'{study_label}_optuna.db'}", load_if_exists = True,
        direction = "maximize", sampler = optuna.samplers.TPESampler(seed = 13_11_26),
        pruner = optuna.pruners.MedianPruner(n_startup_trials = search["startup_trials"], n_warmup_steps = 3))
    started = time.time()

    def objective(trial: optuna.Trial) -> float:
        settings = suggested_settings(trial, search, model_name)
        training = dict(training_overrides or {})
        training.update(settings["training"])
        training["epochs"] = search["epochs"]
        training["patience_epochs"] = search["epochs"]
        training["is_checkpoint_kept"] = False
        result = train_one_run(model_name, 1, batch_name, training_overrides = training, model_overrides = settings["model"],
                               run_label = f"{model_name}_optuna_trial_{trial.number:03d}", device_name = device_name, trial = trial)
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
    settings = suggested_settings(optuna.trial.FixedTrial(best.params), search, model_name)
    training = dict(training_overrides or {})
    training.update(settings["training"])
    final = [train_one_run(model_name, seed, batch_name, training_overrides = training, model_overrides = settings["model"],
                           run_label = f"{model_name}_optuna_best", device_name = device_name, is_done_skipped = True)
             for seed in search["final_seeds"]]

    def mean_of(split: str) -> str:
        values = [run[split]["hard_sheet_score"] for run in final if split in run]
        return f"{numpy.mean(values):.3f}" + (f" ± {numpy.std(values):.3f}" if len(values) > 1 else "") if values else "-"

    lines = [f"# Optuna search {batch_name}, {model_name}", "", f"{len(completed)} trials finished, "
             f"{len(study.trials) - len(completed)} pruned or failed; {(time.time() - started) / 3600:.2f} hours this time.", "",
             f"Best validation hard-sheet score in the search ({search['epochs']} epochs, seed 1): {best.value:.3f} (trial {best.number})", "",
             "Best settings:", "", "```", json.dumps(best.params, indent = 2), "```", "",
             f"Trained in full with these settings and seeds {search['final_seeds']}: validation hard-sheet score {mean_of('validation')}, "
             f"unseen-track score {mean_of('unseen')}. Compare these, not the search's best trial, with the untuned runs.", "",
             "| Trial | State | Score | Settings |", "|---|---|---|---|"]
    for trial in study.trials:
        value = f"{trial.value:.3f}" if trial.value is not None else "-"
        lines.append(f"| {trial.number} | {trial.state.name.lower()} | {value} | {json.dumps(trial.params)} |")
    (runs_folder / f"{study_label}_optuna.md").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return {"best_settings": best.params, "best_value": best.value, "trials_finished": len(completed), "final": final}


def main() -> None:
    """Command line: run or continue the study of one policy in one batch."""

    parser = argparse.ArgumentParser(description = "Optuna search over one policy's settings.")
    parser.add_argument("--batch", default = f"{date.today().isoformat()}_trial")
    parser.add_argument("--model", choices = sorted(POLICY_MODEL_PARAMETERS), help = "the policy to tune (default from config)")
    parser.add_argument("--dataset", help = "dataset name in 03_data/processed (default: the one in config)")
    parser.add_argument("--trials", type = int, help = "total trials of the study (default from config)")
    parser.add_argument("--hours", type = float, help = "time limit for this call (default from config)")
    parser.add_argument("--device", help = "cuda, mps or cpu (default: the best available)")
    arguments = parser.parse_args()
    overrides = {key: value for key, value in (("model_name", arguments.model), ("trials", arguments.trials), ("hours", arguments.hours))
                 if value is not None}
    training = {"dataset_name": arguments.dataset} if arguments.dataset else None
    result = tune(arguments.batch, overrides, training, device_name = arguments.device)
    scores = [run["validation"]["hard_sheet_score"] for run in result["final"]]
    print(f"best settings {result['best_settings']}, search score {result['best_value']:.3f}, "
          f"retrained with {len(scores)} seeds: {numpy.mean(scores):.3f}")


if __name__ == "__main__":
    main()
