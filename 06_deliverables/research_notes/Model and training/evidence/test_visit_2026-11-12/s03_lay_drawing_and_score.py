"""
Step 3: lay the printed drawing onto each run's votes, read how far along the line the puck has come and how far it
stands to the left or right of the drawn centre line ("method B"), and apply the rule of the visit with both methods.

For every run the traced drawing (course_sheets/C0N_course.svg, what was printed for 12.11.26) is laid onto the line
that all of the run's pictures voted for, by a turn and a shift (visit_common.register_drawing). As checks the same is
done (1) with a free scale, which says how far the camera model's scale on the table is off, and (2) with the original
drawing (Niryo N.svg), whose fit should be worse if the traced redrawing was indeed the print.

Rule (checklist, "Part 2"): a run keeps the line when the puck's centre is never more than 12 mm from the centre line
over 10 mm or more of tip travel, from the start until the puck has passed 119 mm (sheet 03) or 160 mm (sheet 04) of
line from end A's cap centre. Both methods are scored; how much the verdict depends on the mark is tried by moving the
mark 10 mm either way.

Run from this folder after s02:  python s03_lay_drawing_and_score.py
Writes s03_steps.csv (per picture, both methods), s03_lay_out.csv (per run: the fit) and s03_verdicts.csv.
"""

import math
from pathlib import Path

import numpy
import pandas

import visit_common
import picture_measures


SCORE_PARAMETERS = {
    "mark_shifts_mm": [-10.0, 10.0],
    "offset_limits_tried_mm": [10.0, 14.0],
    # pictures used for the puck: a fit of at least this quality (the 11.11 threshold)
    "fit_quality_minimum": 0.80
}



def votes_of(label: str) -> dict:
    """The votes of a run (from s02) and a line map with the same grid."""

    parameters = visit_common.merged_visit_parameters()
    stored = numpy.load(Path(parameters["scratch_folder"]) / f"{label}_votes.npz")
    line_map = picture_measures.LineMap(stored["x_range"], stored["y_range"])
    return {"line_votes": stored["line_votes"], "paper_votes": stored["paper_votes"], "line_map": line_map}



def middle_points(votes: dict) -> numpy.ndarray:
    """The table points of the line's middle cells voted by all pictures of a run."""

    settings = picture_measures.merged_measure_parameters()
    line_votes, paper_votes = votes["line_votes"], votes["paper_votes"]
    is_line = (line_votes >= settings["minimum_line_votes"]) & (line_votes >= settings["line_vote_share"] * (line_votes.astype(numpy.float32) + paper_votes))
    middle = picture_measures.middle_of_line(is_line, votes["line_map"].cell_m, settings)
    return votes["line_map"].point_of(middle["cells"])



def main() -> None:

    parameters = visit_common.merged_visit_parameters()
    steps_a = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s02_steps_method_a.csv")
    folders = visit_common.run_folders()
    drawings = {}
    lay_out_rows, step_tables, verdict_rows = [], [], []
    for label, folder in folders.items():
        if label == "R01":
            # no puck: nothing to lay the drawing against at the start; R01 is described from its tip path
            continue
        sheet = visit_common.sheet_of(label)
        if sheet not in drawings:
            drawings[sheet] = {"traced": visit_common.drawing_of_sheet(sheet, "traced"),
                               "original": visit_common.drawing_of_sheet(sheet, "original")}
        rows = steps_a[steps_a["label"] == label].sort_values("step").reset_index(drop = True)
        is_good = rows["fit_quality"] >= SCORE_PARAMETERS["fit_quality_minimum"]
        puck_xy = numpy.where(is_good.to_numpy()[:, None], rows[["puck_x", "puck_y"]].to_numpy(), numpy.nan)
        start_xy = puck_xy[numpy.flatnonzero(numpy.isfinite(puck_xy[:, 0]))[0]]
        votes = votes_of(label)
        middle_xy = middle_points(votes)
        paper_lookup = visit_common.paper_lookup_of(votes["line_votes"], votes["paper_votes"], votes["line_map"])

        # the drawing is tried from both of its ends: every run of the run list starts at end A, the loop demo (no
        # line in the run list, started "as on 11.11") turned out to start at the traced file's other end
        traced_forward = drawings[sheet]["traced"]["points_mm"]
        # (sheet 01's straight line fits equally well both ways, so the runs of the run list are held to end A)
        tries = {"from end A": visit_common.register_drawing(traced_forward, middle_xy, paper_lookup, start_xy),
                 "from end B": visit_common.register_drawing(traced_forward[::-1].copy(), middle_xy, paper_lookup, start_xy)}
        direction = min(tries, key = lambda name: tries[name]["cost"]) if label == "demo" else "from end A"
        fitted = tries[direction]
        traced_points = traced_forward if direction == "from end A" else traced_forward[::-1].copy()
        original_points = drawings[sheet]["original"]["points_mm"]
        original_points = original_points if direction == "from end A" else original_points[::-1].copy()
        scaled = visit_common.register_drawing(traced_points, middle_xy, paper_lookup, start_xy,
                                               is_scale_free = True, start_turn_rad = math.radians(fitted["turn_deg"]))
        original = visit_common.register_drawing(original_points, middle_xy, paper_lookup, start_xy,
                                                  start_turn_rad = math.radians(fitted["turn_deg"]))
        lay_out_rows.append({"label": label, "sheet": sheet, "direction": direction,
                             "other_direction_cost": tries["from end B" if direction == "from end A" else "from end A"]["cost"],
                             "middle_cells": len(middle_xy), "turn_deg": fitted["turn_deg"],
                             "shift_x_m": fitted["shift_m"][0], "shift_y_m": fitted["shift_m"][1],
                             "end_a_x_m": fitted["drawing"].points_m[0, 0], "end_a_y_m": fitted["drawing"].points_m[0, 1],
                             "cost": fitted["cost"], "middle_median_mm": fitted["middle_distance_mm"][0],
                             "middle_p90_mm": fitted["middle_distance_mm"][1], "end_a_from_first_puck_mm": fitted["start_moved_mm"],
                             "free_scale": scaled["scale"], "free_scale_cost": scaled["cost"],
                             "original_cost": original["cost"], "original_middle_median_mm": original["middle_distance_mm"][0],
                             "original_middle_p90_mm": original["middle_distance_mm"][1]})

        laid = fitted["drawing"]
        placed = visit_common.progress_and_offset(puck_xy, laid)
        placed_original = visit_common.progress_and_offset(puck_xy, original["drawing"])
        table = rows[["label", "step", "puck_x", "puck_y", "fit_quality", "tip_x", "tip_y", "offset_mm", "whole_run_offset_mm", "single_offset_mm"]].copy()
        table = table.rename(columns = {"offset_mm": "offset_a_mm", "whole_run_offset_mm": "offset_a_whole_run_mm", "single_offset_mm": "offset_a_single_mm"})
        table.loc[~is_good, ["offset_a_mm", "offset_a_whole_run_mm", "offset_a_single_mm"]] = numpy.nan
        table["arc_mm"] = [entry["arc_mm"] for entry in placed]
        table["offset_b_mm"] = [entry["offset_mm"] for entry in placed]
        table["arc_original_mm"] = [entry["arc_mm"] for entry in placed_original]
        table["offset_original_mm"] = [entry["offset_mm"] for entry in placed_original]
        travel = visit_common.tip_travel_mm(rows[["tip_x", "tip_y"]].to_numpy())
        table["tip_travel_mm"] = travel
        step_tables.append(table)

        summary = visit_common.run_summary(folder)
        verdict = {"label": label, "sheet": sheet, "ending": summary["ending"], "pictures": len(rows),
                   "step_cm": summary.get("fixed_step_cm") if summary.get("step_length_rule") == "fixed" else "own",
                   "tip_travel_mm": float(travel[-1]), "furthest_arc_mm": float(numpy.nanmax(table["arc_mm"])),
                   "last_arc_mm": float(table["arc_mm"].dropna().iloc[-1]), "direction": direction}
        # where the puck first left the line over the whole run, by each method: the first stretch over 12 mm for
        # 10 mm of travel, and how far along the line the puck then stood
        for method in ["a", "b"]:
            index = visit_common.stretch_verdict(table[f"offset_{method}_mm"].to_numpy(), travel, len(rows) - 1)["first_failing_from"]
            verdict[f"left_line_picture_{method}"] = index
            verdict[f"left_line_arc_{method}_mm"] = None if index is None else round(float(table["arc_mm"].iloc[index]), 0)
        if sheet in parameters["mark_mm"]:
            mark = parameters["mark_mm"][sheet]
            for name, shift in [("", 0.0)] + [(f"_mark_{int(shift):+d}", shift) for shift in SCORE_PARAMETERS["mark_shifts_mm"]]:
                passed = numpy.flatnonzero(table["arc_mm"].to_numpy() >= mark + shift)
                last_index = int(passed[0]) if len(passed) else len(rows) - 1
                verdict["mark_reached" + name] = bool(len(passed))
                verdict["mark_picture" + name] = last_index if len(passed) else None
                for method in ["a", "b"]:
                    found = visit_common.stretch_verdict(table[f"offset_{method}_mm"].to_numpy(), travel, last_index)
                    verdict[f"kept_{method}" + name] = found["is_kept"]
                    if name == "":
                        verdict[f"kept_{method}_shortest"] = found["is_kept_shortest"]
                        verdict[f"kept_{method}_longest"] = found["is_kept_longest"]
                        verdict[f"longest_stretch_{method}_mm"] = round(found["longest_stretch_mm"], 1)
                        verdict[f"stretch_{method}_pictures"] = None if found["stretch_from"] is None else f"{found['stretch_from']}-{found['stretch_to']}"
                        verdict[f"first_over_{method}"] = found["first_over"]
                        window = table[f"offset_{method}_mm"].to_numpy()[:last_index + 1]
                        verdict[f"largest_offset_{method}_mm"] = round(float(numpy.nanmax(numpy.abs(window))), 1)
                        # the measuring error is about 2 to 3 mm: the verdict with the limit 2 mm lower and higher
                        for limit in SCORE_PARAMETERS["offset_limits_tried_mm"]:
                            tried = visit_common.stretch_verdict(table[f"offset_{method}_mm"].to_numpy(), travel, last_index,
                                                                 {"offset_limit_mm": limit})
                            verdict[f"kept_{method}_limit_{int(limit)}"] = tried["is_kept"]
            # with the original drawing (its own 119 or 160 mm): does the verdict hold?
            passed = numpy.flatnonzero(table["arc_original_mm"].to_numpy() >= mark)
            last_index = int(passed[0]) if len(passed) else len(rows) - 1
            verdict["kept_b_original_drawing"] = visit_common.stretch_verdict(table["offset_original_mm"].to_numpy(), travel, last_index)["is_kept"]
        verdict_rows.append(verdict)
        print(f"{label}: turn {fitted['turn_deg']:.1f} deg, fit median {fitted['middle_distance_mm'][0]:.2f} mm, "
              f"free scale {scaled['scale']:.4f}, original fit median {original['middle_distance_mm'][0]:.2f} mm; furthest {verdict['furthest_arc_mm']:.0f} mm", flush = True)

    pandas.concat(step_tables, ignore_index = True).to_csv(visit_common.EVIDENCE_FOLDER / "s03_steps.csv", index = False, float_format = "%.5g")
    pandas.DataFrame(lay_out_rows).to_csv(visit_common.EVIDENCE_FOLDER / "s03_lay_out.csv", index = False, float_format = "%.5g")
    verdicts = pandas.DataFrame(verdict_rows)
    verdicts.to_csv(visit_common.EVIDENCE_FOLDER / "s03_verdicts.csv", index = False, float_format = "%.4g")
    with pandas.option_context("display.width", 250, "display.max_columns", 60):
        print(verdicts.to_string())



if __name__ == "__main__":
    main()
