"""
Step 7: a third measure of the puck's offset, "method C": the drawing laid onto the ink of each picture by itself.

Why: the look at the pictures (s04) showed that the drawing laid onto a whole run's votes (method B) can sit 3 to 5 mm
beside the ink of one picture near the first bend of sheet 03, because the camera model puts the same piece of line a
few millimetres apart from different arm poses. The puck's foot is found in one picture, through the same camera model;
an error of the camera model that is common to the puck and the ink near it cancels when both come from the same
picture. So here, for every picture, the drawing is laid (turn and shift, started
from method B's lay) onto the line that this one picture shows (its middle cells within 90 mm of the puck, its bare
paper), and the puck's foot of the same picture is measured against it. The whole drawing is laid (every stretch of
line the picture shows near the puck pulls on it), and the puck is measured against the stretch within
"nearest_window_mm" of method B's place along the line, so that it is not handed to a neighbouring stretch. The drawing's shape carries the line under the
puck, where the picture cannot see it.

Run from this folder after s03:  python s07_single_picture_lay.py
Writes s07_steps_method_c.csv (per picture: offset_c_mm, arc_c_mm, the fit's cells and median distance).
"""

import math
import multiprocessing

import cv2
import numpy
import pandas

import visit_common
import picture_measures
import run_measures
from picture_measures import TableCamera
from puck_detector import PuckDetector


SINGLE_PARAMETERS = {
    "local_half_span_mm": 90.0,
    "map_half_size_m": 0.11,
    "least_middle_cells": 150,
    "nearest_window_mm": 40.0,
    "processes": 6
}
JOINT_COLUMNS = [f"joint_{number}" for number in range(1, 7)]



def measure_run(label: str) -> pandas.DataFrame:
    """Method C for every picture of one run."""

    import scipy.optimize

    parameters = visit_common.merged_visit_parameters()
    cv2.setNumThreads(1)
    folder = visit_common.run_folders()[label]
    run_steps = pandas.read_csv(folder / "steps.csv")
    steps = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_steps.csv")
    steps = steps[steps["label"] == label].sort_values("step").reset_index(drop = True)
    lay_out = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_lay_out.csv", dtype = {"sheet": str})
    layout = lay_out[lay_out["label"] == label].iloc[0]
    drawing = visit_common.drawing_of_sheet(visit_common.sheet_of(label))["points_mm"]
    if layout["direction"] == "from end B":
        drawing = drawing[::-1].copy()
    laid_b = visit_common.DrawingOnTable(drawing, math.radians(layout["turn_deg"]), [layout["shift_x_m"], layout["shift_y_m"]])
    camera_parameters = run_measures.camera_parameters_for(parameters["camera_period"])
    detector = PuckDetector()
    rows = []
    for _, measured in steps.iterrows():
        result = {"label": label, "step": int(measured["step"]), "offset_c_mm": numpy.nan, "arc_c_mm": numpy.nan,
                  "c_middle_cells": 0, "c_median_mm": numpy.nan, "c_moved_mm": numpy.nan}
        if not (numpy.isfinite(measured["puck_x"]) and numpy.isfinite(measured["arc_mm"]) and measured["fit_quality"] >= 0.8):
            rows.append(result)
            continue
        step = int(measured["step"])
        picture = cv2.imread(str(folder / "frames" / f"{step:03d}.jpg"))
        camera = TableCamera([float(run_steps.loc[step, name]) for name in JOINT_COLUMNS], camera_parameters)
        blob = picture_measures.puck_blob(picture, detector)
        puck = numpy.array([measured["puck_x"], measured["puck_y"]])
        half = SINGLE_PARAMETERS["map_half_size_m"]
        line_map = picture_measures.LineMap([puck[0] - half, puck[0] + half], [puck[1] - half, puck[1] + half])
        hidden = picture_measures.hidden_table_mask(picture.shape[:2], camera, blob)
        seen = line_map.view_of(picture, camera, hidden)
        middle_xy = line_map.point_of(picture_measures.middle_of_line(seen["is_line"], line_map.cell_m)["cells"])
        span = SINGLE_PARAMETERS["local_half_span_mm"] / 1000.0
        middle_xy = middle_xy[numpy.hypot(middle_xy[:, 0] - puck[0], middle_xy[:, 1] - puck[1]) <= span]
        result["c_middle_cells"] = len(middle_xy)
        if len(middle_xy) < SINGLE_PARAMETERS["least_middle_cells"]:
            rows.append(result)
            continue
        local_drawing = drawing
        is_paper = seen["is_paper"] & ~seen["is_line"]

        def paper_lookup(points_xy: numpy.ndarray) -> numpy.ndarray:
            cells = numpy.round(line_map.cell_of(points_xy)).astype(int)
            is_inside = (cells[:, 0] >= 0) & (cells[:, 0] < is_paper.shape[0]) & (cells[:, 1] >= 0) & (cells[:, 1] < is_paper.shape[1])
            found = numpy.zeros(len(points_xy), dtype = bool)
            found[is_inside] = is_paper[cells[is_inside, 0], cells[is_inside, 1]]
            return found

        def cost_of(values: numpy.ndarray) -> float:
            laid = visit_common.DrawingOnTable(local_drawing, values[0], values[1:3])
            return visit_common.registration_cost(laid, middle_xy, paper_lookup, parameters)

        start = numpy.array([math.radians(layout["turn_deg"]), layout["shift_x_m"], layout["shift_y_m"]])
        simplex = numpy.array([start, start + [math.radians(1.0), 0, 0], start + [0, 0.002, 0], start + [0, 0, 0.002]])
        found = scipy.optimize.minimize(cost_of, start, method = "Nelder-Mead",
                                        options = {"initial_simplex": simplex, "xatol": 1e-6, "fatol": 1e-4, "maxiter": 1500})
        laid_c = visit_common.DrawingOnTable(local_drawing, found.x[0], found.x[1:3])
        distances, _ = laid_c.tree.query(middle_xy)
        # the nearest point of the stretch around method B's place along the line
        is_window = numpy.abs(laid_c.arc_mm - measured["arc_mm"]) <= SINGLE_PARAMETERS["nearest_window_mm"]
        window_indices = numpy.flatnonzero(is_window)
        nearest = int(window_indices[numpy.argmin(numpy.hypot(*(laid_c.points_m[window_indices] - puck[None, :]).T))])
        tangent = laid_c.tangents[nearest]
        relative = puck - laid_c.points_m[nearest]
        result.update({"offset_c_mm": 1000.0 * float(tangent[0] * relative[1] - tangent[1] * relative[0]),
                       "arc_c_mm": float(laid_c.arc_mm[nearest] + 1000.0 * float(relative @ tangent)),
                       "c_median_mm": 1000.0 * float(numpy.median(distances)),
                       "c_moved_mm": 1000.0 * float(numpy.linalg.norm(laid_c.point_at(float(laid_c.arc_mm[nearest])) - laid_b.point_at(float(laid_c.arc_mm[nearest])))),
                       "c_turn_rad": float(found.x[0]), "c_shift_x_m": float(found.x[1]), "c_shift_y_m": float(found.x[2])})
        rows.append(result)
    return pandas.DataFrame(rows)



def main() -> None:

    lay_out = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_lay_out.csv", dtype = {"sheet": str})
    labels = [label for label in lay_out["label"]]
    with multiprocessing.Pool(SINGLE_PARAMETERS["processes"]) as pool:
        tables = pool.map(measure_run, labels)
    table = pandas.concat(tables, ignore_index = True)
    table.to_csv(visit_common.EVIDENCE_FOLDER / "s07_steps_method_c.csv", index = False, float_format = "%.5g")
    print(table.groupby("label")[["offset_c_mm", "c_median_mm", "c_moved_mm"]].agg(["count", "median"]).round(2).to_string())



if __name__ == "__main__":
    main()
