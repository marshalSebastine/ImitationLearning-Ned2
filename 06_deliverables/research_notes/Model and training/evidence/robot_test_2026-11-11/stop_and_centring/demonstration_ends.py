"""
How the recorded demonstrations end: at the stop picture of every episode, where the puck stands against the line's
round end, measured from the last pictures of the episode (the same measure as for the robot's runs in robot_runs.py).

The end's place is hidden by the puck and the hand during the last steps, and where the arm turns on the spot it is
hidden for many pictures, so the line's end is mapped from the latest pictures that showed its place in the open,
however long before the stop they were taken. The puck is fitted in the last eight pictures, which also shows how the
demonstrator arrived.

Run from this folder:    python demonstration_ends.py [first sheet number] [last sheet number]
(by default all six sheets; the sheets can be run in parts to keep each run short). Writes
demonstration_ends_<first>_<last>.csv (one row per episode) and demonstration_last_steps_<first>_<last>.csv (one row
per episode and picture of the last eight); summary.py reads all parts. Nothing connects to the robot and nothing is
written into 03_data.
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy
import pandas

import picture_measures
import run_measures

from config import RECORDING_PARAMETERS, TRAINING_DATA_PARAMETERS
from puck_detector import PuckDetector


DEMONSTRATION_END_PARAMETERS = {
    # the pictures that map the line's end: the latest so many in which the place where the puck will stop lay inside
    # the picture (this far from its edges) and this far from the magnet tip in the picture (clear of the hand and the
    # puck), together with the last pictures before the stop, in which the puck is fitted
    "pictures_showing_end": 24,
    "inside_margin_px": 30.0,
    "clear_of_tip_px": 110.0,
    "last_pictures": 8,
    "travel_baseline_m": 0.03,
    # ink that the stop picture itself still shows ahead of the puck is counted this far ahead of the puck's centre
    "ahead_most_m": 0.035
}



def pictures_showing_end(
        steps: pandas.DataFrame,
        camera_parameters: Dict
) -> List[int]:
    """
    The rows whose pictures map the line's end (see ``DEMONSTRATION_END_PARAMETERS``), in rising order.

    Parameters
    ----------
    steps : pandas.DataFrame
        From ``run_measures.demonstration_steps``.
    camera_parameters : dict
        The camera of the episode's period.

    Returns
    -------
    list of int
        Row numbers; the last one is the stop row.
    """

    settings = DEMONSTRATION_END_PARAMETERS
    tips_xy = steps[["tip_x", "tip_y"]].to_numpy(dtype = float)
    travel = run_measures.travel_direction(tips_xy, len(tips_xy) - 1, settings["travel_baseline_m"])
    # roughly where the puck stands at the stop: its centre lies ahead of the magnet tip that pushed it
    place = tips_xy[-1] + RECORDING_PARAMETERS["puck_centre_ahead_of_tip_m"] * travel
    showing = []
    for index in range(len(steps)):
        row = steps.iloc[index]
        camera = picture_measures.TableCamera([row[name] for name in run_measures.JOINT_COLUMNS], camera_parameters)
        pixel = camera.project_table(place[None, :])[0]
        if not numpy.all(numpy.isfinite(pixel)):
            continue
        is_inside = (settings["inside_margin_px"] <= pixel[0] <= camera.picture_width - 1 - settings["inside_margin_px"]
                     and settings["inside_margin_px"] <= pixel[1] <= camera.picture_height - 1 - settings["inside_margin_px"])
        is_clear = float(numpy.linalg.norm(pixel - camera.tip_pixel())) >= settings["clear_of_tip_px"]
        if is_inside and is_clear:
            showing.append(index)
    last = range(max(0, len(steps) - settings["last_pictures"]), len(steps))
    return sorted(set(showing[-settings["pictures_showing_end"]:]) | set(last))



def ink_seen_ahead(
        measured: Dict,
        end: Dict
) -> float:
    """
    The line the stop picture itself shows ahead of the puck, in square millimetres on the table: line cells of the
    last picture alone that lie ahead of the puck's centre, within the line's band and "ahead_most_m". The puck's own
    body hides the table just beyond it from the camera, so this can be zero while ink sticks out seen from above.
    """

    line_map = measured["line_map"]
    rows, columns = numpy.nonzero(measured["last"]["is_line"])
    if len(rows) == 0:
        return 0.0
    cells_xy = line_map.point_of(numpy.column_stack([rows, columns]))
    puck = measured["last"]["fit"]["base_xy"]
    direction = end["direction_xy"]
    left = numpy.array([-direction[1], direction[0]])
    along = (cells_xy - puck[None, :]) @ direction
    across = (cells_xy - end["end_centre_xy"][None, :]) @ left
    is_ahead = (along > 0.0) & (along <= DEMONSTRATION_END_PARAMETERS["ahead_most_m"]) & (numpy.abs(across) <= picture_measures.MEASURE_PARAMETERS["end_band_m"])
    return float(numpy.count_nonzero(is_ahead) * (1000.0 * line_map.cell_m) ** 2)



def main(first_sheet: int, last_sheet: int) -> None:

    frames = pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False)
    sheets = [f"path_{number:02d}" for number in range(first_sheet, last_sheet + 1)]
    detector = PuckDetector()
    results: List[Dict] = []
    last_steps: List[Dict] = []
    episode_ids = [episode_id for episode_id in frames["episode_id"].unique()
                   if frames.loc[frames["episode_id"] == episode_id, "sheet"].iloc[0] in sheets]
    for count, episode_id in enumerate(episode_ids, start = 1):
        steps = run_measures.demonstration_steps(frames, episode_id)
        first_row = steps.iloc[0]
        period = run_measures.camera_period(first_row["started"])
        entry = {"episode_id": episode_id, "sheet": first_row["sheet"], "plan_number": first_row["plan_number"],
                 "start_kind": first_row["start_kind"], "score": first_row["score"], "outcome": first_row["outcome"],
                 "operator": first_row["operator"], "camera_period": period, "pictures": len(steps),
                 "last_action": steps.iloc[-1]["action"],
                 "last_three_moves_cm": 100.0 * float(steps["move_length_m"].iloc[-4:-1].sum())}
        camera_parameters = run_measures.camera_parameters_for(period)
        used_steps = pictures_showing_end(steps, camera_parameters)
        entry["pictures_used"] = len(used_steps)
        measured = run_measures.measure_steps(steps, camera_parameters, parameters = {"end_window_pictures": len(used_steps)},
                                              is_offset_measured = False, detector = detector, used_steps = used_steps,
                                              fitted_last_steps = DEMONSTRATION_END_PARAMETERS["last_pictures"])
        last = measured["last"]
        entry["is_puck_found"] = bool(last is not None and last["fit"] is not None)
        if entry["is_puck_found"]:
            entry.update({"fit_quality": last["fit"]["quality"], "puck_x": float(last["fit"]["base_xy"][0]), "puck_y": float(last["fit"]["base_xy"][1])})
        end = measured["end"]
        entry["is_end_found"] = end is not None
        if end is not None:
            entry.update({"short_mm": 1000.0 * end["short_m"], "aside_mm": 1000.0 * end["aside_m"],
                          "centre_gap_mm": 1000.0 * end["centre_gap_m"], "uncovered_mm": 1000.0 * end["uncovered_m"],
                          "ink_beyond_mm2": end["ink_beyond_mm2"], "is_end_seen": end["is_end_seen"],
                          "line_width_mm": 1000.0 * end["line_width_m"], "ink_seen_ahead_mm2": ink_seen_ahead(measured, end),
                          "end_x": float(end["end_centre_xy"][0]), "end_y": float(end["end_centre_xy"][1]),
                          "is_direction_from_line": end["is_direction_from_line"]})
            # the puck's place against the end in each of the last pictures
            direction = end["direction_xy"]
            left = numpy.array([-direction[1], direction[0]])
            fitted = measured["rows"].dropna(subset = ["puck_x"])
            for _, fitted_row in fitted.iterrows():
                step = int(fitted_row["step"])
                gap = end["end_centre_xy"] - numpy.array([fitted_row["puck_x"], fitted_row["puck_y"]])
                move_length = steps.iloc[step]["move_length_m"]
                last_steps.append({"episode_id": episode_id, "sheet": first_row["sheet"], "step": step, "steps_before_stop": len(steps) - 1 - step,
                                   "frame_index": int(steps.iloc[step]["frame_index"]), "action": steps.iloc[step]["action"],
                                   "move_length_cm": 100.0 * float(move_length) if pandas.notna(move_length) else numpy.nan,
                                   "fit_quality": fitted_row["fit_quality"], "short_mm": 1000.0 * float(gap @ direction),
                                   "aside_mm": -1000.0 * float(gap @ left)})
        results.append(entry)
        if count % 10 == 0 or count == len(episode_ids):
            print(f"{count} of {len(episode_ids)} episodes measured", flush = True)

    table = pandas.DataFrame(results)
    out_path = run_measures.EVIDENCE_FOLDER / f"demonstration_ends_{first_sheet:02d}_{last_sheet:02d}.csv"
    table.to_csv(out_path, index = False, float_format = "%.4f")
    pandas.DataFrame(last_steps).to_csv(run_measures.EVIDENCE_FOLDER / f"demonstration_last_steps_{first_sheet:02d}_{last_sheet:02d}.csv",
                                        index = False, float_format = "%.3f")
    print(f"written: {out_path.name} ({len(table)} episodes; end found in {int(table['is_end_found'].sum())})")



if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
