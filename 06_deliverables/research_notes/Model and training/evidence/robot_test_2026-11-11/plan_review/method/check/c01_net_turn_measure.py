"""
Check of the critic's central measure, "the share of the demonstrated net turn the model gives back" (m04, m05, m07).

Question: is a value near one half a sign that the model turns too little, or is it what any honest (calibrated) but
imperfect predictor gives when the windows are picked by what the DEMONSTRATION did? Measured on the fold models'
saved answers on sheets they never saw; nothing is trained, nothing connects to the robot.

    1. Windows of 10 rows: correlation between the asked and the shown net twist, both regression slopes, and the
       share given back when windows are picked (a) by the shown net turn of 30 degrees or more (the critic's way),
       (b) by the ASKED net turn of 30 degrees or more (the mirror image), (c) not picked at all.
    2. What a plain scaling of the twist does to the measure and to the error (a rule can be passed by scaling alone).
    3. Run to run over the 23 fold trainings: the measure against the window correlation and against the mean size.

Run from the folder above:
    python check/c01_net_turn_measure.py > check/c01_net_turn_measure.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import method_common

CHECK_PARAMETERS = {"window_rows": 10, "net_turn_least_deg": 30.0,
                    "runs_folder": method_common.ROOT_FOLDER / "04_runs" / "gpu_ki014",
                    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"]}



def window_table(steps: pandas.DataFrame, asked_column: str, window_rows: int) -> pandas.DataFrame:
    """Return one row per window of ``window_rows`` rows: sheet, shown net twist, asked net twist."""

    rows = []
    for episode_id, episode in steps.groupby("episode_id", sort = False):
        shown = episode["forearm_change_deg"].fillna(0.0).to_numpy()
        asked = episode[asked_column].to_numpy()
        for start in range(0, len(episode) - window_rows + 1, window_rows):
            part = slice(start, start + window_rows)
            rows.append({"sheet": episode["sheet"].iloc[0], "shown": shown[part].sum(), "asked": asked[part].sum()})
    return pandas.DataFrame(rows)



def share_given_back(windows: pandas.DataFrame) -> float:
    """Return the critic's measure over the given windows."""

    return float((numpy.sign(windows["shown"]) * windows["asked"]).sum() / windows["shown"].abs().sum())



def report() -> None:
    settings = method_common.merged_parameters(CHECK_PARAMETERS)
    frames = method_common.cleaned_frames(settings)
    least = settings["net_turn_least_deg"]
    print("1. Windows of 10 rows on sheets the model never saw")
    for recipe, path in settings["fold_outputs"].items():
        data = numpy.load(path)
        answers = pandas.DataFrame({"frame_index": data["frame_index"], "asked_twist": data["exact"][:, 2]})
        steps = frames.merge(answers, on = "frame_index", how = "left")
        steps = steps[steps["action"] != "stop"]
        windows = window_table(steps, "asked_twist", settings["window_rows"])
        correlation = numpy.corrcoef(windows["shown"], windows["asked"])[0, 1]
        slope_asked_on_shown = numpy.polyfit(windows["shown"], windows["asked"], 1)[0]
        slope_shown_on_asked = numpy.polyfit(windows["asked"], windows["shown"], 1)[0]
        by_shown = windows[windows["shown"].abs() >= least]
        by_asked = windows[windows["asked"].abs() >= least]
        print(f"   recipe {recipe}: {len(windows)} windows; correlation asked with shown {correlation:.3f} (squared {correlation ** 2:.3f});"
              f" slope of asked on shown {slope_asked_on_shown:.3f}; slope of shown on asked {slope_shown_on_asked:.3f}")
        print(f"      picked by the SHOWN net turn of {least:.0f} or more ({len(by_shown)} windows): the model gives back {share_given_back(by_shown):.3f} of the shown turn")
        mirror = float((numpy.sign(by_asked["asked"]) * by_asked["shown"]).sum() / by_asked["asked"].abs().sum())
        print(f"      picked by the ASKED net turn of {least:.0f} or more ({len(by_asked)} windows): the demonstration shows {mirror:.3f} of the asked turn"
              f" (the model asks {1 / mirror:.2f} times what was shown)")
        print(f"      not picked (all {len(windows)} windows): sum of |asked| over sum of |shown| {windows['asked'].abs().sum() / windows['shown'].abs().sum():.3f};"
              f" mean asked {windows['asked'].mean():+.2f}, mean shown {windows['shown'].mean():+.2f} degrees")
        print("      per sheet: correlation / share given back (picked by shown) / windows picked")
        for sheet, group in windows.groupby("sheet"):
            picked = group[group["shown"].abs() >= least]
            print(f"         {sheet[-2:]}: {numpy.corrcoef(group['shown'], group['asked'])[0, 1]:+.2f} / {share_given_back(picked) if len(picked) else float('nan'):.2f} / {len(picked)} of {len(group)}")

        print(f"   2. recipe {recipe}: the asked twist multiplied by a fixed number")
        for factor in (1.0, 1.25, 1.5, 2.0):
            scaled = windows.assign(asked = windows["asked"] * factor)
            picked = scaled[scaled["shown"].abs() >= least]
            error = numpy.sqrt(((scaled["asked"] - scaled["shown"]) ** 2).mean())
            explained = 1.0 - ((scaled["asked"] - scaled["shown"]) ** 2).sum() / ((scaled["shown"] - scaled["shown"].mean()) ** 2).sum()
            print(f"      x{factor:4.2f}: share given back {share_given_back(picked):.3f}; root mean square error of the window's net turn {error:5.1f} degrees;"
                  f" explained share of the shown net turn's variance {explained:+.3f}")

    print()
    print("3. The 23 fold trainings: share given back against the window correlation and the mean size of the asked twist")
    steps = frames[frames["action"] != "stop"]
    rows = []
    for run_folder in sorted(settings["runs_folder"].iterdir()):
        files = [run_folder / f"predictions_without_{sheet}.npz" for sheet in settings["sheets"]]
        if not run_folder.is_dir() or not all(path.exists() for path in files):
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
        table = steps.assign(asked_twist = steps["frame_index"].map(twist))
        if table["asked_twist"].isna().any():
            continue
        windows = window_table(table, "asked_twist", settings["window_rows"])
        picked = windows[windows["shown"].abs() >= least]
        rows.append({"run": run_folder.name[11:], "share": share_given_back(picked), "correlation": numpy.corrcoef(windows["shown"], windows["asked"])[0, 1],
                     "size": table["asked_twist"].abs().mean() / table["forearm_change_deg"].fillna(0.0).abs().mean(),
                     "slope_shown_on_asked": numpy.polyfit(windows["asked"], windows["shown"], 1)[0]})
    table = pandas.DataFrame(rows)
    print(table.round(3).to_string(index = False))
    print(f"   over the {len(table)} runs: correlation of the share given back with the window correlation {table['share'].corr(table['correlation']):+.2f},"
          f" with the mean size of the asked twist {table['share'].corr(table['size']):+.2f}")
    print(f"   window correlation runs from {table['correlation'].min():.3f} to {table['correlation'].max():.3f}; squared {table['correlation'].min() ** 2:.3f} to {table['correlation'].max() ** 2:.3f};"
          f" slope of shown on asked from {table['slope_shown_on_asked'].min():.2f} to {table['slope_shown_on_asked'].max():.2f}")



if __name__ == "__main__":
    report()
