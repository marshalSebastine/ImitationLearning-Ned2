"""
One table of every model compared so far (9.11.26, Model and training session).

Collects the frozen-feature probes (04_runs/<date>_frozen_feature_probes/<experiment>.json, from training_probes.py)
and the end-to-end runs (04_runs/<date>_<run name>/metrics.json, from train_policy.py) into 04_runs/comparison.md
and 04_runs/comparison.csv, with the same measures side by side. Every number is the mean over the folds, and the
worst fold is shown beside the most important measures, because a model must work on every unseen track, not on
average.

Run from 02_code:
    ..\\.venv\\Scripts\\python.exe compare_runs.py
Settings: TRAINING_COMPARISON_PARAMETERS in config.py.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas

import numpy

from config import TRAINING_COMPARISON_PARAMETERS, TRAINING_TARGET_PARAMETERS
from training_metrics import stop_metrics, summarised_folds
from training_targets import load_training_frames



def probe_rows(runs_folder: Path) -> List[Dict]:
    """Return one row per probe variant found in the frozen-feature probe folders."""

    rows = []
    for experiment_path in sorted(runs_folder.glob("*_frozen_feature_probes/*.json")):
        experiment = json.loads(experiment_path.read_text(encoding = "utf-8"))
        # baselines.json holds the rules without a model, in its own layout; the report shows it separately
        if "results" not in experiment:
            continue
        for result in experiment["results"]:
            settings = result["settings"]
            row = {"kind": "probe", "experiment": experiment_path.stem, "name": result["name"], "split": settings.get("split"),
                   "backbone": settings.get("backbone"), "features": settings.get("feature_kind"), "design": settings.get("design"),
                   "folds": len(result["folds"]), "fold_names": " ".join(fold["fold"] for fold in result["folds"])}
            row.update(result["summary"])
            row.update(matched_summary(result["folds"]))
            rows.append(row)
    return rows



def end_to_end_rows(runs_folder: Path) -> List[Dict]:
    """Return one row per end-to-end run that has a metrics.json."""

    rows = []
    for metrics_path in sorted(runs_folder.glob("*/metrics.json")):
        result = json.loads(metrics_path.read_text(encoding = "utf-8"))
        settings = result["settings"]
        # a run trained in parts (some folds, then the others) keeps one fold_<name>.json per fold, while metrics.json
        # holds only the last part, so the summary is rebuilt from every fold file in the folder
        fold_results = [json.loads(path.read_text(encoding = "utf-8"))["measures"] for path in sorted(metrics_path.parent.glob("fold_*.json"))]
        for fold in fold_results:
            if "fixed_two_episodes_stopped_right" not in fold:
                fold.update(rescored_stop(metrics_path.parent, fold["fold"], settings))
        row = {"kind": "end_to_end", "experiment": metrics_path.parent.name, "name": result["name"], "split": settings.get("split"),
               "backbone": settings.get("backbone"), "features": "fine-tuned" if settings.get("is_pretrained") else "from scratch",
               "design": settings.get("design"), "folds": len(fold_results), "fold_names": " ".join(fold["fold"] for fold in fold_results),
               "milliseconds_per_picture": result.get("milliseconds_per_picture")}
        row.update(summarised_folds(fold_results) if fold_results else result["summary"])
        row.update(matched_summary(fold_results))
        rows.append(row)
    return rows



FRAMES_BY_RULES = {}



def rescored_stop(
        run_folder: Path,
        fold_name: str,
        settings: Dict,
        threshold: float = 0.5,
        consecutive_pictures: int = 2
) -> Dict:
    """
    Score a fold's saved stop probabilities with the fixed stop rule, for runs trained before the rule was recorded.

    The rows come from the same cleaning the run used (its "is_audit_cleaning" and score settings), so each saved
    prediction meets the right episode, step order and goal distance.

    Returns
    -------
    dict
        "fixed_two_episodes_stopped_right", "_early" and "_never_stopped" (shares of the fold's episodes), or nothing
        when the run saved no predictions or no stop probability.
    """

    predictions_path = run_folder / f"predictions_{fold_name}.npz"
    if not predictions_path.exists():
        return {}
    predictions = numpy.load(predictions_path)
    if "stop_probability" not in predictions:
        return {}
    rules = {key: settings[key] for key in ("kept_scores", "is_unscored_kept", "is_mistake_dropped", "is_flagged_dropped",
                                            "is_static_dropped", "is_audit_cleaning") if key in settings}
    rule_key = json.dumps(rules, sort_keys = True)
    if rule_key not in FRAMES_BY_RULES:
        target_parameters = dict(TRAINING_TARGET_PARAMETERS)
        target_parameters.update(rules)
        FRAMES_BY_RULES[rule_key] = load_training_frames(target_parameters)
    frames = FRAMES_BY_RULES[rule_key]
    position_of_frame = pandas.Series(numpy.arange(len(frames)), index = frames["frame_index"].to_numpy())
    positions = position_of_frame.reindex(predictions["frame_index"]).to_numpy()
    is_known = ~numpy.isnan(positions)
    positions = positions[is_known].astype(int)
    order = numpy.argsort(positions)
    positions = positions[order]
    probability = predictions["stop_probability"][is_known][order]
    rows = frames.iloc[positions]
    result = stop_metrics((rows["action"] == "stop").to_numpy().astype(float), probability,
                          rows["goal_distance_cm"].fillna(0.0).to_numpy(), rows["episode_id"].to_numpy(),
                          threshold = threshold, consecutive_pictures = consecutive_pictures)
    return {f"fixed_two_{name}": value for name, value in result.items() if name.startswith("episodes_")}



def matched_summary(
        fold_results: List[Dict],
        parameters: Dict = TRAINING_COMPARISON_PARAMETERS
) -> Dict:
    """
    Return the fold means over only the "matched_folds", with "_matched" added to each name.

    The end-to-end runs were trained on fewer held-out sheets than the probes, so models are compared fairly only on
    the folds they share. Empty when a result lacks one of those folds.
    """

    matched = [fold for fold in fold_results if fold.get("fold") in parameters["matched_folds"]]
    if len(matched) != len(parameters["matched_folds"]):
        return {}
    return {f"{name}_matched": value for name, value in summarised_folds(matched).items() if name.endswith("_mean")}



def comparison_table(parameters: Dict = TRAINING_COMPARISON_PARAMETERS) -> pandas.DataFrame:
    """
    Build the comparison table and write it as markdown and CSV.

    Parameters
    ----------
    parameters : dict
        "runs_folder", "columns" (measure, heading) and "sort_by".

    Returns
    -------
    pandas.DataFrame
        One row per probe variant or run.
    """

    runs_folder = Path(parameters["runs_folder"])
    table = pandas.DataFrame(probe_rows(runs_folder) + end_to_end_rows(runs_folder))
    if table.empty:
        return table
    table.to_csv(runs_folder / "comparison.csv", index = False)
    shown = ["kind", "experiment", "name", "split", "backbone", "features", "design", "folds"]
    headings = {column: column for column in shown}
    for column, heading in parameters["columns"]:
        if column in table.columns:
            shown.append(column)
            headings[column] = heading
    view = table[shown].rename(columns = headings).sort_values(parameters["sort_by"], ascending = False)
    lines = ["# Model comparison", "",
             "Every number is the mean over the folds; \"worst\" is the worst fold. Direction measures count only pictures after",
             "which the tip really moved. Built by 02_code/compare_runs.py from the files in 04_runs.", "",
             "## All results, each over its own folds", ""]
    lines += markdown_rows(view)
    matched_columns = [column for column, _ in parameters["columns"] if f"{column}_matched" in table.columns]
    if matched_columns:
        matched = table[table[f"{matched_columns[0]}_matched"].notna()]
        matched_view = matched[["kind", "experiment", "name", "backbone", "features", "design"] + [f"{column}_matched" for column in matched_columns]]
        matched_view = matched_view.rename(columns = {f"{column}_matched": heading for column, heading in parameters["columns"]})
        matched_view = matched_view.sort_values(parameters["sort_by"], ascending = False)
        lines += ["", f"## Compared on the same held-out sheets only ({', '.join(parameters['matched_folds'])})", ""]
        lines += markdown_rows(matched_view)
    (runs_folder / "comparison.md").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    return table



def markdown_rows(view: pandas.DataFrame) -> List[str]:
    """Return a table as markdown lines, numbers with three decimals."""

    lines = ["| " + " | ".join(view.columns) + " |", "|" + "---|" * len(view.columns)]
    for _, row in view.iterrows():
        cells = ["" if pandas.isna(value) else (f"{value:.3f}" if isinstance(value, float) else str(value)) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines



def main() -> None:

    table = comparison_table()
    print(f"{len(table)} rows written to {TRAINING_COMPARISON_PARAMETERS['runs_folder']}")



if __name__ == "__main__":
    main()
