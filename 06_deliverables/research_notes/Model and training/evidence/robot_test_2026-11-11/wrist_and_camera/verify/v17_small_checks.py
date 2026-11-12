"""
Verifier, check 17: three small re-counts.
    a) sheet 06: the demonstrated wrist by tenth of the episode (start-mark episodes);
    b) the driver's tip height (5 mm above the fitted table) against the demonstrated height: view shift of a table
       point 5 cm ahead of the tip, wrist as demonstrated, 400 random rows (project functions);
    c) the size of the predicted twist against the demonstrated twist, fold models on sheets never seen and model E's
       logged answers are not needed here: saved predictions only.
    D:/AI_Project/.venv/Scripts/python.exe v17_small_checks.py
"""
import math
import sys
import numpy
import pandas
sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
import forearm_camera_model
from config import PUCK_PARAMETERS, table_height
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed
from training_targets import load_training_frames

JOINTS = [f"joint_{index}" for index in range(1, 7)]
frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "step_fraction", "frame_index"] + JOINTS)
frames["wrist"] = numpy.degrees(frames["joint_5"])
mark = frames[frames["sheet_orientation"].str.contains("START MARK", na = False)].copy()
mark["tenth"] = numpy.minimum((10 * mark["step_fraction"]).astype(int), 9)
print("a) median demonstrated wrist by tenth of the episode, start-mark episodes:")
print(mark.pivot_table(index = "sheet", columns = "tenth", values = "wrist", aggfunc = "median").round(1).to_string())

print("\nb) the tip at the driver's height instead of the demonstrated height (wrist as demonstrated), 400 random rows:")
chosen = frames.sample(400, random_state = 17)
shifts, moves = [], []
for _, row in chosen.iterrows():
    joints = row[JOINTS].to_numpy(dtype = float).tolist()
    tip, _ = magnet_tip_and_lean(joints)
    target = numpy.array([tip[0], tip[1], table_height(tip[0], tip[1]) + PUCK_PARAMETERS["pusher_clearance_above_sheet_m"]])
    other = solve_with_wrist_fixed(target, joints[3], joints[4], joints[:3])
    if other is None:
        continue
    other[5] = joints[5]
    centre, axes = forearm_camera_model.camera_pose_in_base(joints)
    ahead = numpy.array([tip[0], tip[1], 0.0]) + 0.05 * numpy.array([axes[0, 2], axes[1, 2], 0.0]) / math.hypot(axes[0, 2], axes[1, 2])
    ahead[2] = table_height(ahead[0], ahead[1])
    shift = forearm_camera_model.project_points(ahead, other)[0] - forearm_camera_model.project_points(ahead, joints)[0]
    shifts.append(float(numpy.hypot(*shift)))
    moves.append(1000 * float(numpy.linalg.norm(forearm_camera_model.camera_pose_in_base(other)[0] - centre)))
print("   table point 5 cm ahead: shift median %.1f px at 640, 90th %.1f, most %.1f; camera move median %.1f mm (%d rows solved)" % (
    numpy.median(shifts), numpy.quantile(shifts, 0.9), max(shifts), numpy.median(moves), len(shifts)))

print("\nc) predicted twist against demonstrated twist (saved predictions, sheets never seen):")
cleaned = load_training_frames()
for label, folder in (("E recipe", "2026-11-10_resnet18_chunk_stop_weight_25"), ("A recipe", "2026-11-09_resnet18_final_candidate")):
    parts = [pandas.DataFrame({"frame_index": saved["frame_index"], "twist_predicted": saved["twist_deg"]})
             for saved in (numpy.load(f"D:/AI_Project/.claude/worktrees/training-analysis/04_runs/gpu_ki014/{folder}/predictions_without_path_{n:02d}.npz") for n in range(1, 7))]
    scored = cleaned.merge(pandas.concat(parts), on = "frame_index")
    scored = scored[scored["action"] != "stop"]
    turning = scored["forearm_change_deg"].abs() >= 2.0
    print("   %s: mean |predicted| %.2f against mean |demonstrated| %.2f degrees per step (ratio %.2f); on the %d steps that twist by 2 degrees or more: ratio %.2f, sign right %.1f %%" % (
        label, scored["twist_predicted"].abs().mean(), scored["forearm_change_deg"].abs().mean(), scored["twist_predicted"].abs().mean() / scored["forearm_change_deg"].abs().mean(),
        int(turning.sum()), scored.loc[turning, "twist_predicted"].abs().mean() / scored.loc[turning, "forearm_change_deg"].abs().mean(),
        100 * (numpy.sign(scored.loc[turning, "twist_predicted"]) == numpy.sign(scored.loc[turning, "forearm_change_deg"])).mean()))
