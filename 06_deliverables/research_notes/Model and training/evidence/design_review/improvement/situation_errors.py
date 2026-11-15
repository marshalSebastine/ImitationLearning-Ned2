"""
Where the end-to-end models and two probes go wrong, per situation, including the crossings (read only).

Uses the project's own error analysis (02_code/training_error_analysis.py: situation_columns, prediction_files, the
same direction hits) and adds situations from the audit's two line registrations:
- the recompute checker's registration (audit/view_and_tracks/check_recompute/c07_rows.csv, good fits only, cost under
  0.8 cm): the puck's place along the line, to mark "puck within 2 cm of a crossing" and "2 to 7 cm from a crossing"
  (crossings at 12.7 and 45.0 cm from end A on sheet 05, 6.2 and 31.6 cm on sheet 06, audit note), and the puck's
  distance from the line;
- the auditor's registration (audit/view_and_tracks/s15_rows.csv): the line radius at the puck and the line left to go.
Nothing is written outside this script's folder.

Usage: python situation_errors.py <run or probe name> [...]
"""

import sys
from pathlib import Path

CODE_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/02_code"
sys.path.insert(0, CODE_FOLDER)

import numpy
import pandas

from config import TRAINING_ERROR_ANALYSIS_PARAMETERS, TRAINING_METRIC_PARAMETERS
from training_error_analysis import prediction_files, situation_columns
from training_metrics import angle_difference_deg
from training_targets import chunk_targets, load_training_frames

AUDIT_FOLDER = Path("C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/audit/view_and_tracks")

PARAMETERS = {
    "registration_rows": AUDIT_FOLDER / "check_recompute" / "c07_rows.csv",
    "registration_episodes": AUDIT_FOLDER / "check_recompute" / "c07_episodes.csv",
    "auditor_rows": AUDIT_FOLDER / "s15_rows.csv",
    "good_fit_cost_cm": 0.8,
    "crossings_cm": {"path_05": [12.7, 45.0], "path_06": [6.2, 31.6]},
    "near_crossing_cm": 2.0,
    "approach_crossing_cm": 7.0,
    "output_folder": Path(__file__).resolve().parent,
}


def registration_situations(frames, parameters = None):
    """
    Add registration-based situations to every row of ``frames`` (missing where no good fit exists).
    """
    settings = {**PARAMETERS, **(parameters or {})}
    rows = pandas.read_csv(settings["registration_rows"])
    episodes = pandas.read_csv(settings["registration_episodes"])
    good_keys = set(episodes.loc[episodes["cost"] < settings["good_fit_cost_cm"], "key"])
    rows = rows[rows["key"].isin(good_keys)]
    rows = rows.rename(columns = {"key": "episode_id", "row": "row_in_episode"})
    merged = frames[["episode_id", "row_in_episode", "sheet", "frame_index"]].merge(
        rows[["episode_id", "row_in_episode", "arc_from_a_cm", "line_distance_cm"]],
        on = ["episode_id", "row_in_episode"], how = "left")
    crossing_place = numpy.full(len(merged), "no fit", dtype = object)
    has_fit = merged["arc_from_a_cm"].notna().to_numpy()
    crossing_place[has_fit] = "sheet without crossing"
    for sheet, crossings in settings["crossings_cm"].items():
        on_sheet = (merged["sheet"] == sheet).to_numpy() & has_fit
        distance = numpy.min(numpy.abs(merged["arc_from_a_cm"].to_numpy()[:, None] - numpy.array(crossings)[None, :]),
                             axis = 1)
        crossing_place[on_sheet] = numpy.where(distance[on_sheet] <= settings["near_crossing_cm"], "within 2 cm",
                                               numpy.where(distance[on_sheet] <= settings["approach_crossing_cm"],
                                                           "2 to 7 cm away", "loop sheet, elsewhere"))
    off_line = numpy.full(len(merged), "no fit", dtype = object)
    distance_to_line = merged["line_distance_cm"].to_numpy()
    off_line[has_fit] = numpy.where(distance_to_line[has_fit] <= 1.0, "puck within 1 cm of the line",
                                    numpy.where(distance_to_line[has_fit] <= 2.0, "1 to 2 cm off", "over 2 cm off"))
    auditor = pandas.read_csv(settings["auditor_rows"], usecols = ["frame_index", "line_radius_cm", "line_to_go_cm"])
    merged = merged.merge(auditor, on = "frame_index", how = "left")
    radius = merged["line_radius_cm"].to_numpy()
    radius_bin = numpy.full(len(merged), "no fit", dtype = object)
    has_radius = ~numpy.isnan(radius)
    radius_bin[has_radius] = numpy.select(
        [radius[has_radius] < 1.5, radius[has_radius] < 2.5, radius[has_radius] < 5.0, radius[has_radius] < 10.0],
        ["under 1.5 cm", "1.5 to 2.5 cm", "2.5 to 5 cm", "5 to 10 cm"], default = "10 cm or more")
    to_go = merged["line_to_go_cm"].to_numpy()
    to_go_bin = numpy.full(len(merged), "no fit", dtype = object)
    has_to_go = ~numpy.isnan(to_go)
    to_go_bin[has_to_go] = numpy.select([to_go[has_to_go] <= 3.0, to_go[has_to_go] <= 8.0],
                                        ["last 3 cm of line", "3 to 8 cm to go"], default = "more than 8 cm to go")
    return pandas.DataFrame({"crossing": crossing_place, "off_line": off_line, "line_radius": radius_bin,
                             "line_to_go": to_go_bin}, index = frames.index)


def error_rows(name, frames, situations):
    """
    Score one run's held-out predictions per situation.
    """
    index_of_frame = pandas.Series(numpy.arange(len(frames)), index = frames["frame_index"].to_numpy())
    chunk = chunk_targets(frames, TRAINING_METRIC_PARAMETERS["path_steps"])[:, -1, :]
    path_angle = numpy.degrees(numpy.arctan2(chunk[:, 1], chunk[:, 0]))
    has_path = (numpy.linalg.norm(chunk, axis = 1) >= TRAINING_METRIC_PARAMETERS["path_minimum_cm"]) & (frames["action"] != "stop").to_numpy()
    is_moving = frames["action"].isin(["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"]).to_numpy()
    parts = []
    files = prediction_files(name)
    for file in files:
        predictions = numpy.load(file)
        positions = index_of_frame.reindex(predictions["frame_index"]).to_numpy()
        is_known = ~numpy.isnan(positions)
        positions = positions[is_known].astype(int)
        predicted = predictions["angle_deg"][is_known]
        part = situations.iloc[positions].copy()
        step_error = angle_difference_deg(predicted, frames["move_angle_deg"].to_numpy()[positions])
        part["step_hit"] = numpy.where(is_moving[positions], step_error <= 45, numpy.nan)
        part["path_hit"] = numpy.where(has_path[positions], angle_difference_deg(predicted, path_angle[positions]) <= 45, numpy.nan)
        part["forward_path_hit"] = numpy.where(has_path[positions], angle_difference_deg(0.0, path_angle[positions]) <= 45, numpy.nan)
        part["forward_step_hit"] = numpy.where(is_moving[positions], angle_difference_deg(0.0, frames["move_angle_deg"].to_numpy()[positions]) <= 45, numpy.nan)
        part["episode_id"] = frames["episode_id"].to_numpy()[positions]
        parts.append(part)
    scored = pandas.concat(parts)
    rows = []
    for column in ["sheet", "place", "shape", "forearm", "crossing", "off_line", "line_radius", "line_to_go"]:
        for value, group in scored.groupby(column):
            moving = group["step_hit"].notna()
            rows.append({"run": name, "situation": column, "value": value,
                         "moving_pictures": int(moving.sum()),
                         "episodes": int(group.loc[moving, "episode_id"].nunique()),
                         "step_within_45": round(float(group["step_hit"].mean()), 3),
                         "always_forward_step_within_45": round(float(group["forward_step_hit"].mean()), 3),
                         "path_within_45": round(float(group["path_hit"].mean()), 3),
                         "always_forward_path_within_45": round(float(group["forward_path_hit"].mean()), 3)})
    return pandas.DataFrame(rows), [file.name for file in files]


def main(names, parameters = None):
    """
    Print and save one table per run.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    frames = load_training_frames()
    situations = situation_columns(frames, TRAINING_ERROR_ANALYSIS_PARAMETERS)
    situations = pandas.concat([situations, registration_situations(frames, settings)], axis = 1)
    print("rows:", len(frames))
    print("crossing situation counts (all rows):", situations["crossing"].value_counts().to_dict())
    tables = []
    for name in names:
        table, files = error_rows(name, frames, situations)
        print()
        print(name, "files:", files)
        print(table.to_string(index = False))
        tables.append(table)
    pandas.concat(tables).to_csv(settings["output_folder"] / "situation_errors.csv", index = False)


if __name__ == "__main__":
    main(sys.argv[1:])
