"""
Judge 1: extra checks on the saved held-out-sheet predictions (read only; writes judge_extra_checks.out.txt here).

1. Per held-out sheet and run: episodes with a streak of 5 or more consecutive path pictures more than 45 degrees off
   (open loop on the demonstrated pictures), and the share of path pictures more than 90 degrees off.
2. Twist calibration of every exact-design fold (final candidate, finetune_exact, finetune_chunk): least-squares slope
   of the demonstrated forearm change on the predicted one (through the origin), sign right where the demonstration
   twisted 2 degrees or more, sign right where the prediction is 3 degrees or more, mean |predicted| / mean |demonstrated|.
3. Final candidate: stop outcome by finishing end under fixed 0.5 with one picture and with two in a row.
"""

import json
import sys
from pathlib import Path

import numpy
import pandas

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_rescoring

SETTINGS = dict(judge_rescoring.PARAMETERS)



def streaks_per_sheet(table, angle_column, settings):
    """
    Return (episodes, episodes with a 5+ streak over 45 deg, share of path pictures over 90 deg) for one fold.
    """

    rows = table["rows"]
    path_error = judge_rescoring.angle_difference_deg(rows["path_angle_deg"], rows[angle_column])
    is_path_row = rows["is_path_row"].to_numpy()
    episodes = with_streak = 0
    for episode_id in rows["episode_id"].unique():
        mask = (rows["episode_id"] == episode_id).to_numpy()
        longest = judge_rescoring.longest_bad_streak(path_error[mask], is_path_row[mask], None)
        episodes += 1
        with_streak += int(longest >= settings["streak_length"])
    over_90 = float((path_error[is_path_row] > 90.0).mean())
    return episodes, with_streak, round(over_90, 4)



def twist_calibration(table):
    """
    Return the twist calibration numbers of one exact-design fold.
    """

    rows = table["rows"]
    is_non_stop = (rows["action"] != "stop").to_numpy()
    demonstrated = rows["forearm_change_deg"].fillna(0.0).clip(-15.0, 15.0).to_numpy()[is_non_stop]
    predicted = rows["twist_deg"].to_numpy()[is_non_stop]
    slope = float((predicted * demonstrated).sum() / max(1e-9, (predicted * predicted).sum()))
    is_twisting = numpy.abs(demonstrated) >= 2.0
    is_confident = numpy.abs(predicted) >= 3.0
    sign_twisting = float((numpy.sign(predicted[is_twisting]) == numpy.sign(demonstrated[is_twisting])).mean())
    sign_confident = float((numpy.sign(predicted[is_confident & is_twisting]) == numpy.sign(demonstrated[is_confident & is_twisting])).mean()) if (is_confident & is_twisting).any() else float("nan")
    size_ratio = float(numpy.abs(predicted).mean() / max(1e-9, numpy.abs(demonstrated).mean()))
    episode_ids = rows["episode_id"].to_numpy()[is_non_stop]
    signed_sum = pandas.Series(predicted - demonstrated).groupby(episode_ids).sum()
    return {"non_stop_rows": int(is_non_stop.sum()), "slope_demonstrated_on_predicted": round(slope, 2),
            "twisting_rows": int(is_twisting.sum()), "sign_right_twisting": round(sign_twisting, 3),
            "confident_twisting_rows": int((is_confident & is_twisting).sum()), "sign_right_confident": round(sign_confident, 3),
            "mean_abs_predicted_over_demonstrated": round(size_ratio, 2),
            "episode_signed_error_sum_median_deg": round(float(signed_sum.median()), 1),
            "episode_signed_error_sum_abs_90th_deg": round(float(signed_sum.abs().quantile(0.9)), 1)}



def main():
    """
    Print the three checks.
    """

    settings = SETTINGS
    frames = judge_rescoring.loaded_frames(settings)
    lines = []

    def say(text = ""):
        print(text, flush = True)
        lines.append(text)

    say("== 1. Streaks of path errors over 45 degrees (5 or more in a row), per held-out sheet ==")
    for run_name, run_settings in settings["runs"].items():
        parts = []
        for sheet in settings["sheets"]:
            table = judge_rescoring.fold_table(run_name, run_settings, sheet, frames, settings)
            if table is None:
                continue
            angle_column = "angle_expected_deg" if run_settings["kind"] == "classes" else "angle_deg"
            episodes, with_streak, over_90 = streaks_per_sheet(table, angle_column, settings)
            forward_rows = table["rows"].assign(forward_deg = 0.0)
            forward_episodes, forward_streak, forward_over_90 = streaks_per_sheet({"rows": forward_rows}, "forward_deg", settings)
            parts.append(f"{sheet}: {with_streak} of {episodes} (always forward {forward_streak}); over 90 deg {over_90:.3f} (always forward {forward_over_90:.3f})")
        say(f"  {run_name} [{'mean direction' if run_settings['kind'] == 'classes' else 'predicted direction'}]: " + "; ".join(parts))

    say()
    say("== 2. Twist calibration of the exact-design folds ==")
    for run_name in ("final_candidate", "finetune_exact", "finetune_chunk"):
        run_settings = settings["runs"][run_name]
        for sheet in settings["sheets"]:
            table = judge_rescoring.fold_table(run_name, run_settings, sheet, frames, settings)
            if table is None or "twist_deg" not in table["rows"]:
                continue
            say(f"  {run_name} sheet {sheet}: {json.dumps(twist_calibration(table))}")

    say()
    say("== 3. Final candidate: stop by finishing end under fixed rules ==")
    run_settings = settings["runs"]["final_candidate"]
    for threshold, consecutive in ((0.5, 1), (0.5, 2)):
        end_counts = {}
        for sheet in settings["sheets"]:
            table = judge_rescoring.fold_table("final_candidate", run_settings, sheet, frames, settings)
            if table is None:
                continue
            outcomes = judge_rescoring.episode_outcomes(table, threshold, consecutive, settings)
            rows = table["rows"]
            for episode_id, (outcome, _) in outcomes.items():
                finish_end = rows.loc[rows["episode_id"] == episode_id, "finish_end"].iloc[0]
                end_counts.setdefault((f"path_{sheet}", finish_end), {"right": 0, "early": 0, "near": 0, "never": 0})[outcome] += 1
        far = {"right": 0, "early": 0, "near": 0, "never": 0}
        close = {"right": 0, "early": 0, "near": 0, "never": 0}
        for (sheet, end), counts in end_counts.items():
            target = far if settings["edge_distance_cm"].get((sheet, end), 0.0) >= 5.0 else close
            for key in target:
                target[key] += counts[key]
        say(f"  ({threshold}, {consecutive}): " + "; ".join(
            f"{sheet[-2:]}{end} ({settings['edge_distance_cm'].get((sheet, end))} cm): {c['right']}/{c['early']}/{c['near']}/{c['never']}"
            for (sheet, end), c in sorted(end_counts.items())))
        say(f"     ends 5 cm or more from the edge (right/early/near/never): {far}; under 5 cm: {close}")

    (Path(__file__).resolve().parent / "judge_extra_checks.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



if __name__ == "__main__":
    main()
