"""
How steady is "the share of the demonstrated net turn the model gives back" from seed to seed and recipe to recipe?

A measure can only carry a rule fixed in advance if two trainings of the same recipe give nearly the same value. This
script reads the saved predictions of every fold run of 9 to 11.11.26 (04_runs/gpu_ki014/<run>/predictions_<fold>.npz,
written by train_policy.py on ki-014; twist per picture, each sheet answered by the model that never saw it) and
computes, per run, the net turn given back over windows of 10 rows with a demonstrated net turn of 30 degrees or more
(the verifier's measure, as in m04 and m05), next to the run's own path within 45 degrees and stop counts.

Nothing is trained, nothing connects to the robot.

Run from this folder:
    python m07_seed_spread_of_the_turn_measure.py > m07_seed_spread_of_the_turn_measure.out.txt
"""

import json
from typing import Dict, Optional

import numpy
import pandas

import method_common

SEED_SPREAD_PARAMETERS = {
    "runs_folder": method_common.ROOT_FOLDER / "04_runs" / "gpu_ki014",
    "window_rows": 10,
    "net_turn_least_deg": 30.0,
    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"]
}



def seed_spread_report(parameters: Optional[Dict] = None) -> None:
    """Print one line per fold run: the net turn given back, path within 45 and the stop counts."""

    settings = method_common.merged_parameters(SEED_SPREAD_PARAMETERS)
    settings.update(parameters or {})
    frames = method_common.cleaned_frames(settings)
    steps = frames[frames["action"] != "stop"]
    windows = []
    for episode_id, episode in steps.groupby("episode_id", sort = False):
        shown = episode["forearm_change_deg"].fillna(0.0).to_numpy()
        index = episode["frame_index"].to_numpy()
        for start in range(0, len(episode) - settings["window_rows"] + 1, settings["window_rows"]):
            part = slice(start, start + settings["window_rows"])
            if abs(shown[part].sum()) >= settings["net_turn_least_deg"]:
                windows.append((episode["sheet"].iloc[0], index[part], float(shown[part].sum())))
    print(f"windows of {settings['window_rows']} rows with a demonstrated net turn of {settings['net_turn_least_deg']:.0f} degrees or more: {len(windows)}")
    print("run                                                   net turn given back: all sheets   per sheet 01 to 06                     path within 45   stopped right / early / never (fixed 0.5, two in a row; share of episodes, mean of sheets)")
    rows = []
    for run_folder in sorted(settings["runs_folder"].iterdir()):
        files = [run_folder / f"predictions_without_{sheet}.npz" for sheet in settings["sheets"]]
        if not run_folder.is_dir() or not all(path.exists() for path in files) or not (run_folder / "metrics.json").exists():
            continue
        twist = {}
        is_complete = True
        for path in files:
            data = numpy.load(path)
            if "twist_deg" not in data.files:
                is_complete = False
                break
            twist.update(zip(data["frame_index"].tolist(), data["twist_deg"].tolist()))
        if not is_complete:
            continue
        given = {sheet: 0.0 for sheet in settings["sheets"]}
        shown = {sheet: 0.0 for sheet in settings["sheets"]}
        is_covered = True
        for sheet, index, net in windows:
            if any(frame not in twist for frame in index):
                is_covered = False
                break
            given[sheet] += numpy.sign(net) * sum(twist[frame] for frame in index)
            shown[sheet] += abs(net)
        if not is_covered:
            continue
        summary = json.loads((run_folder / "metrics.json").read_text(encoding = "utf-8"))["summary"]
        share = sum(given.values()) / sum(shown.values())
        per_sheet = "  ".join(f"{given[sheet] / shown[sheet]:.2f}" for sheet in settings["sheets"])
        print(f"{run_folder.name[11:]:52s}  {share:30.3f}   {per_sheet}   {summary.get('path_within_45_deg_mean', float('nan')):14.3f}"
              f"   {summary.get('fixed_two_episodes_stopped_right_mean', float('nan')):.2f} / {summary.get('fixed_two_episodes_stopped_early_mean', float('nan')):.2f}"
              f" / {summary.get('fixed_two_episodes_never_stopped_mean', float('nan')):.2f}")
        rows.append({"run": run_folder.name[11:], "share": share, "path": summary.get("path_within_45_deg_mean", float("nan"))})
    table = pandas.DataFrame(rows)
    pairs = []
    for run in table["run"]:
        if run.endswith("_seed_2") and run[:-7] in set(table["run"]):
            first = table[table["run"] == run[:-7]].iloc[0]
            second = table[table["run"] == run].iloc[0]
            pairs.append((run[:-7], abs(first["share"] - second["share"]), 100 * abs(first["path"] - second["path"])))
    print()
    print("seed pairs (the same recipe trained twice): difference in the net turn given back / in path within 45 (points)")
    for name, share_difference, path_difference in pairs:
        print(f"   {name:48s} {share_difference:.3f} / {path_difference:.1f}")
    if pairs:
        print(f"   largest difference between two seeds: net turn given back {max(pair[1] for pair in pairs):.3f}, path within 45 {max(pair[2] for pair in pairs):.1f} points")
    print(f"   over all {len(table)} fold runs the net turn given back runs from {table['share'].min():.3f} to {table['share'].max():.3f} and path within 45 from"
          f" {table['path'].min():.3f} to {table['path'].max():.3f}; correlation between the two over the runs {table['share'].corr(table['path']):+.2f}")



if __name__ == "__main__":
    seed_spread_report()
