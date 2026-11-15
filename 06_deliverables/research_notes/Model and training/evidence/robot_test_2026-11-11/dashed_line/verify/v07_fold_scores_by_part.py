"""
Verifier check 7: how the fold models (trained without one sheet) score on the held-out sheets 05 and 06, by part of
the line, recomputed from the saved predictions with my own targets and my own placing of the pictures on the line.

Mine: the path direction (tip now to the tip three kept rows later, in the current picture's frame, from tip_x_m,
tip_y_m and picture_heading_deg), the parts of the line (before the crossing, in the loop, after the loop, from
v05_demonstration_tracks_*.csv), the twist and backward-step scores. From the project: the list of cleaned rows
(training_targets.load_training_frames) and the saved predictions (04_runs/gpu_ki014/<run>/predictions_<fold>.npz).
Also prints the data counts of the dashed sheet.

Run:  D:/AI_Project/.venv/Scripts/python.exe v07_fold_scores_by_part.py > v07_fold_scores_by_part.out.txt
A few seconds. Nothing connects to the robot, nothing is trained.
"""

import numpy
import pandas

from verify_common import ROOT_FOLDER, VERIFY_FOLDER

from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 300)
RUNS_FOLDER = ROOT_FOLDER / "04_runs" / "gpu_ki014"
RUNS = {
    "E recipe (chunk, stop weight 25)": "2026-11-10_resnet18_chunk_stop_weight_25",
    "E recipe, seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2",
    "chunk recipe": "2026-11-10_resnet18_final_candidate_chunk",
    "chunk recipe, seed 2": "2026-11-10_resnet18_final_candidate_chunk_seed_2",
    "A recipe (exact)": "2026-11-09_resnet18_final_candidate",
    "A recipe, seed 2": "2026-11-09_resnet18_final_candidate_seed_2"
}
PATH_STEPS = 3
PATH_MINIMUM_CM = 1.5
TWIST_THRESHOLD_DEG = 2.0


def angle_gap(first: numpy.ndarray, second: numpy.ndarray) -> numpy.ndarray:
    """Return the absolute difference of two angles in degrees, folded into 0 to 180."""

    return numpy.abs((first - second + 180.0) % 360.0 - 180.0)


def my_path_targets(frames: pandas.DataFrame) -> pandas.DataFrame:
    """Add my own path direction ("my_path_angle"), its length and the exact step's direction and length."""

    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    path_angle = numpy.full(len(frames), numpy.nan)
    path_length = numpy.zeros(len(frames))
    for _, part in frames.groupby("episode_id", sort = False):
        positions = part.index.to_numpy()
        tip = part[["tip_x_m", "tip_y_m"]].to_numpy()
        heading = numpy.radians(part["picture_heading_deg"].to_numpy())
        future = numpy.minimum(numpy.arange(len(part)) + PATH_STEPS, len(part) - 1)
        moved = 100.0 * (tip[future] - tip)
        forward = moved[:, 0] * numpy.cos(heading) + moved[:, 1] * numpy.sin(heading)
        left = -moved[:, 0] * numpy.sin(heading) + moved[:, 1] * numpy.cos(heading)
        path_angle[positions] = numpy.degrees(numpy.arctan2(left, forward))
        path_length[positions] = numpy.hypot(forward, left)
    frames["my_path_angle"] = path_angle
    frames["my_path_length_cm"] = path_length
    frames["is_path_row"] = (frames["action"] != "stop") & (path_length >= PATH_MINIMUM_CM)
    frames["step_angle"] = numpy.degrees(numpy.arctan2(frames["tip_left_m"].fillna(0.0), frames["tip_forward_m"].fillna(0.0)))
    frames["step_length_cm"] = 100.0 * numpy.hypot(frames["tip_left_m"].fillna(0.0), frames["tip_forward_m"].fillna(0.0))
    return frames


def main() -> None:

    frames = load_training_frames({"is_audit_cleaning": True})
    raw = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False)
    print("Data counts")
    print(f"  frames.csv: {len(raw)} pictures in {raw['episode_id'].nunique()} episodes; after the audit cleaning: {len(frames)} pictures in {frames['episode_id'].nunique()} episodes")
    counts = pandas.DataFrame({"episodes": raw.groupby("sheet")["episode_id"].nunique(), "pictures": raw.groupby("sheet").size(),
                               "cleaned_episodes": frames.groupby("sheet")["episode_id"].nunique(), "cleaned_pictures": frames.groupby("sheet").size()})
    counts["share_of_cleaned_%"] = 100.0 * counts["cleaned_pictures"] / len(frames)
    print(counts.round(1).to_string())
    dashed = raw[raw["sheet"] == "path_06"]
    episodes = dashed.drop_duplicates("episode_id")
    print(f"  sheet 06: scores {episodes['score'].value_counts().to_dict()}; operators {episodes['operator'].value_counts().to_dict()};"
          f" lighting {episodes['lighting'].value_counts().to_dict()}; sessions {episodes['session'].value_counts().to_dict()}")
    print(f"  sheet 06 recorded from {episodes['started'].min()} to {episodes['finished'].max()}")
    cleaned_dashed = frames[frames["sheet"] == "path_06"]
    by_end = cleaned_dashed.groupby("start_end").agg(episodes = ("episode_id", "nunique"), pictures = ("episode_id", "size"))
    print("  sheet 06 after cleaning, by the start end the training used:")
    print(by_end.to_string())
    cleaned_solid = frames[frames["sheet"] == "path_05"]
    print("  sheet 05 after cleaning, by the start end the training used:")
    print(cleaned_solid.groupby("start_end").agg(episodes = ("episode_id", "nunique"), pictures = ("episode_id", "size")).to_string())

    frames = my_path_targets(frames)
    placed = pandas.concat([pandas.read_csv(VERIFY_FOLDER / f"v05_demonstration_tracks_{sheet}.csv") for sheet in ("05", "06")])
    progress = placed.groupby("plan")["from_start_cm"].agg(lambda values: values.max() - values.min())
    bad_fit = {"05-28", "05-30", "06-22", "06-23", "06-24", "06-25"}
    placed = placed[~placed["plan"].isin(bad_fit)]
    placed["part"] = numpy.where(placed["along_cm"] < 0.0, "before the crossing",
                                 numpy.where(placed["along_cm"] <= placed["loop_length_cm"], "in the loop", "after the loop"))
    placed.loc[placed["along_cm"].isna(), "part"] = "not placed"
    frames = frames.merge(placed[["frame_index", "part", "along_cm", "beside_cm", "travel_from", "loop_length_cm"]], on = "frame_index", how = "left")
    frames["part"] = frames["part"].fillna("not placed")
    print(f"\n  my placing: episodes left out for a poor line fit: {sorted(bad_fit)} (short starts near an end)")

    print("\nPath direction within 45 degrees on the held-out sheet, by part of the line (rows scored: not the stop row, the tip moves 1.5 cm or more over three rows)")
    summary_rows = []
    for label, folder in RUNS.items():
        for sheet in ("05", "06"):
            path = RUNS_FOLDER / folder / f"predictions_without_path_{sheet}.npz"
            if not path.exists():
                print(f"  missing: {path}")
                continue
            saved = numpy.load(path)
            predictions = pandas.DataFrame({"frame_index": saved["frame_index"], "angle_deg": saved["angle_deg"], "twist_deg": saved["twist_deg"],
                                            "stop_probability": saved["stop_probability"]})
            if "angle_vector_deg" in saved.files:
                predictions["angle_vector_deg"] = saved["angle_vector_deg"]
            if "chunk_cm" in saved.files:
                predictions["first_chunk_length_cm"] = numpy.hypot(saved["chunk_cm"][:, 0, 0], saved["chunk_cm"][:, 0, 1])
            table = frames[frames["sheet"] == f"path_{sheet}"].merge(predictions, on = "frame_index", how = "inner")
            scored = table[table["is_path_row"]]
            row = {"model": label, "sheet": sheet, "rows": len(table), "path_rows": len(scored),
                   "whole_sheet": (angle_gap(scored["angle_deg"], scored["my_path_angle"]) <= 45.0).mean()}
            for part in ("before the crossing", "in the loop", "after the loop"):
                inside = scored[scored["part"] == part]
                row[part] = (angle_gap(inside["angle_deg"], inside["my_path_angle"]) <= 45.0).mean()
                row[f"n {part.split()[0]}"] = len(inside)
            loop = table[table["part"] == "in the loop"]
            has_twist = loop["forearm_change_deg"].abs() >= TWIST_THRESHOLD_DEG
            row["loop twist sign right"] = (numpy.sign(loop.loc[has_twist, "twist_deg"]) == numpy.sign(loop.loc[has_twist, "forearm_change_deg"])).mean()
            row["loop twist size ratio"] = loop.loc[has_twist, "twist_deg"].abs().mean() / loop.loc[has_twist, "forearm_change_deg"].abs().mean()
            row["loop twist projected ratio"] = (loop.loc[has_twist, "twist_deg"] * numpy.sign(loop.loc[has_twist, "forearm_change_deg"])).sum() / loop.loc[has_twist, "forearm_change_deg"].abs().sum()
            is_backward = (loop["step_angle"].abs() > 90.0) & (loop["step_length_cm"] > 0.3) & (loop["action"] != "stop")
            row["loop backward rows"] = int(is_backward.sum())
            row["loop backward also asked"] = (loop.loc[is_backward, "angle_deg"].abs() > 90.0).mean()
            if "first_chunk_length_cm" in table:
                moving = loop[loop["action"] != "stop"]
                row["loop first chunk step cm"] = moving["first_chunk_length_cm"].median()
                row["loop demonstrated step cm"] = moving["step_length_cm"].median()
            summary_rows.append(row)
    summary = pandas.DataFrame(summary_rows)
    print(summary[["model", "sheet", "rows", "path_rows", "whole_sheet", "before the crossing", "n before", "in the loop", "n in", "after the loop", "n after"]].round(3).to_string(index = False))
    print("\nIn the loop: twist (rows with a demonstrated twist of 2 degrees or more) and backward steps (demonstrated step beyond 90 degrees and longer than 0.3 cm)")
    print(summary[["model", "sheet", "loop twist sign right", "loop twist size ratio", "loop twist projected ratio", "loop backward rows", "loop backward also asked",
                   "loop first chunk step cm", "loop demonstrated step cm"]].round(3).to_string(index = False))
    summary.round(4).to_csv(VERIFY_FOLDER / "v07_fold_scores_by_part.csv", index = False)
    frames[frames["sheet"].isin(["path_05", "path_06"])][["frame_index", "sheet", "plan_number", "row_in_episode", "start_end", "action", "part", "along_cm", "beside_cm",
                                                           "my_path_angle", "my_path_length_cm", "is_path_row", "step_angle", "step_length_cm",
                                                           "forearm_change_deg", "frame_path", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]].to_csv(
        VERIFY_FOLDER / "v07_rows_05_06.csv", index = False)



if __name__ == "__main__":
    main()
