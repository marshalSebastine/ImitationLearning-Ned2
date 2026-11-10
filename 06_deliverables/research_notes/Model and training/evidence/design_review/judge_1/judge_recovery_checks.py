"""
Judge 1: recovery-like strata on the saved held-out-sheet predictions (read only; writes judge_recovery_checks.out.txt).

1. Backward moves (back, back_left, back_right) within 45 degrees, split by the forearm turn |joint 4| under 30 degrees
   (a straight camera: the backward move is a real back-up) and 30 degrees or more (often pushing seen through a
   turned camera).
2. Moving pictures in the first 5 rows after a hand-correction restart (waypoint 0 at row_in_episode over 0), step and
   path within 45 degrees.
Runs and sheets: the final candidate on all six sheets and on 03, 05, 06; finetune_exact, finetune_chunk,
finetune_classes (mean direction) and finetune_mirror (mean direction) on 03, 05, 06.
"""

import sys
from pathlib import Path

import numpy

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_rescoring

SETTINGS = dict(judge_rescoring.PARAMETERS)
SETTINGS.update({"recovery_rows": 5, "straight_forearm_deg": 30.0})



def rows_since_restart(frames):
    """
    Return, per row of frames.csv, the number of rows since the latest hand-correction restart (NaN before any).
    """

    since = numpy.full(len(frames), numpy.nan)
    episode_ids = frames["episode_id"].to_numpy()
    waypoints = frames["waypoint"].fillna(-1).to_numpy()
    row_numbers = frames["row_in_episode"].to_numpy()
    last_restart = None
    for position in range(len(frames)):
        if position == 0 or episode_ids[position] != episode_ids[position - 1]:
            last_restart = None
        if waypoints[position] == 0 and row_numbers[position] > 0:
            last_restart = row_numbers[position]
        if last_restart is not None:
            since[position] = row_numbers[position] - last_restart
    return since



def main():
    """
    Print both strata per run.
    """

    settings = SETTINGS
    frames = judge_rescoring.loaded_frames(settings)
    frames["rows_since_restart"] = rows_since_restart(frames)
    lines = []

    def say(text = ""):
        print(text, flush = True)
        lines.append(text)

    plan = [("final_candidate", ["01", "02", "03", "04", "05", "06"]), ("final_candidate", ["03", "05", "06"]),
            ("finetune_exact", ["03", "05", "06"]), ("finetune_chunk", ["03", "05", "06"]),
            ("finetune_classes", ["03", "05", "06"]), ("finetune_mirror", ["03", "05", "06"])]
    for run_name, sheets in plan:
        run_settings = settings["runs"][run_name]
        angle_column = "angle_expected_deg" if run_settings["kind"] == "classes" else "angle_deg"
        totals = {"straight_back_rows": 0, "straight_back_hits": 0, "turned_back_rows": 0, "turned_back_hits": 0,
                  "recovery_rows": 0, "recovery_step_hits": 0, "recovery_path_rows": 0, "recovery_path_hits": 0,
                  "recovery_forward_step_hits": 0}
        for sheet in sheets:
            table = judge_rescoring.fold_table(run_name, run_settings, sheet, frames, settings)
            if table is None:
                continue
            rows = table["rows"]
            rows["rows_since_restart"] = frames.loc[rows["frame_index"].to_numpy(), "rows_since_restart"].to_numpy()
            step_error = judge_rescoring.angle_difference_deg(rows["move_angle_deg"], rows[angle_column])
            path_error = judge_rescoring.angle_difference_deg(rows["path_angle_deg"], rows[angle_column])
            forward_error = judge_rescoring.angle_difference_deg(rows["move_angle_deg"], 0.0)
            is_moving = rows["is_moving"].to_numpy()
            is_backward = rows["action"].isin(settings["backward_names"]).to_numpy() & is_moving
            is_straight = (rows["joint_4"].abs() < numpy.radians(settings["straight_forearm_deg"])).to_numpy() if rows["joint_4"].abs().max() < 7 else (rows["joint_4"].abs() < settings["straight_forearm_deg"]).to_numpy()
            is_recovery = (rows["rows_since_restart"].fillna(999).to_numpy() <= settings["recovery_rows"])
            totals["straight_back_rows"] += int((is_backward & is_straight).sum())
            totals["straight_back_hits"] += int((step_error[is_backward & is_straight] <= 45).sum())
            totals["turned_back_rows"] += int((is_backward & ~is_straight).sum())
            totals["turned_back_hits"] += int((step_error[is_backward & ~is_straight] <= 45).sum())
            totals["recovery_rows"] += int((is_moving & is_recovery).sum())
            totals["recovery_step_hits"] += int((step_error[is_moving & is_recovery] <= 45).sum())
            totals["recovery_forward_step_hits"] += int((forward_error[is_moving & is_recovery] <= 45).sum())
            is_path_row = rows["is_path_row"].to_numpy()
            totals["recovery_path_rows"] += int((is_path_row & is_recovery).sum())
            totals["recovery_path_hits"] += int((path_error[is_path_row & is_recovery] <= 45).sum())
        say(f"{run_name} [{angle_column}] sheets {', '.join(sheets)}: backward within 45 with |joint 4| under 30 deg "
            f"{totals['straight_back_hits']} of {totals['straight_back_rows']}, at 30 deg or more {totals['turned_back_hits']} of "
            f"{totals['turned_back_rows']}; first {settings['recovery_rows']} rows after a correction restart: step 45 "
            f"{totals['recovery_step_hits']} of {totals['recovery_rows']} (always forward {totals['recovery_forward_step_hits']}), path 45 "
            f"{totals['recovery_path_hits']} of {totals['recovery_path_rows']}")

    (Path(__file__).resolve().parent / "judge_recovery_checks.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



if __name__ == "__main__":
    main()
