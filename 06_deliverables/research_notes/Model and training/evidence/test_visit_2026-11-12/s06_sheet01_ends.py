"""
Step 6: sheet 01, where the puck's centre stood against the centre of the line's end (end B's cap centre) when the
robot stopped, for R00, R02 to R04 (start by the placing line and the faint picture) and X01 to X03 (start as on 11.11).

Two methods:
    method A    end_cover of 11.11 (picture_measures.py): the line's end found in the line map voted by the last 16
                pictures before the stop, its round end's centre half a line width before the tip;
    method B    the traced drawing laid onto all of the run's votes (step 3): end B's cap centre is the drawing's last
                point.
Both use the puck's foot fitted in the stop picture (the last picture; no move follows a stop). Along the line:
positive = the puck's centre short of the end's centre. Across: positive = left of the line looking from end A to end B
(the robot's left when it pushes away from its base).

The rule: finished when the two centres are at most 17 mm apart; the 12 mm and 7 mm readings are written beside it.

Run from this folder after s03:  python s06_sheet01_ends.py
Writes s06_sheet01_ends.csv.
"""

import json
import math

import numpy
import pandas

import visit_common


ENDS_PARAMETERS = {
    "labels": ["R00", "R02", "R03", "R04", "X01", "X02", "X03"],
    "readings_mm": [17.0, 12.0, 7.0]
}



def main() -> None:

    steps = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_steps.csv")
    lay_out = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_lay_out.csv", dtype = {"sheet": str})
    ends_a = json.loads((visit_common.EVIDENCE_FOLDER / "s02_ends_method_a.json").read_text(encoding = "utf-8"))
    drawing = visit_common.drawing_of_sheet("01")["points_mm"]
    rows = []
    for label in ENDS_PARAMETERS["labels"]:
        run = steps[steps["label"] == label].sort_values("step")
        last = run.iloc[-1]
        puck = numpy.array([last["puck_x"], last["puck_y"]])
        layout = lay_out[lay_out["label"] == label].iloc[0]
        laid = visit_common.DrawingOnTable(drawing, math.radians(layout["turn_deg"]), [layout["shift_x_m"], layout["shift_y_m"]])
        end_centre = laid.points_m[-1]
        direction = laid.tangents[-1]
        left = numpy.array([-direction[1], direction[0]])
        gap = end_centre - puck
        row = {"label": label, "start": "placing line and faint picture" if label.startswith("R") else "as on 11.11",
               "stop_picture": int(last["step"]), "fit_quality": round(float(last["fit_quality"]), 3),
               "short_b_mm": round(1000.0 * float(gap @ direction), 1), "left_b_mm": round(-1000.0 * float(gap @ left), 1),
               "gap_b_mm": round(1000.0 * float(numpy.linalg.norm(gap)), 1)}
        end_a = ends_a.get(label)
        if end_a is not None:
            row.update({"short_a_mm": round(1000.0 * end_a["short_m"], 1), "left_a_mm": round(1000.0 * end_a["aside_m"], 1),
                        "gap_a_mm": round(1000.0 * end_a["centre_gap_m"], 1), "is_end_seen_a": end_a["is_end_seen"],
                        "line_width_a_mm": round(1000.0 * end_a["line_width_m"], 1)})
        for reading in ENDS_PARAMETERS["readings_mm"]:
            row[f"within_{int(reading)}_b"] = bool(row["gap_b_mm"] <= reading)
            if end_a is not None:
                row[f"within_{int(reading)}_a"] = bool(row["gap_a_mm"] <= reading)
        rows.append(row)
    table = pandas.DataFrame(rows)
    table.to_csv(visit_common.EVIDENCE_FOLDER / "s06_sheet01_ends.csv", index = False)
    with pandas.option_context("display.width", 250, "display.max_columns", 40):
        print(table.to_string())
        for start, group in table.groupby("start"):
            print(f"\n{start}: left of the end's centre, method B {group['left_b_mm'].median():.1f} mm (median; "
                  f"{group['left_b_mm'].min():.1f} to {group['left_b_mm'].max():.1f}), method A {group['left_a_mm'].median():.1f} mm; "
                  f"short, B {group['short_b_mm'].median():.1f} mm, A {group['short_a_mm'].median():.1f} mm")



if __name__ == "__main__":
    main()
