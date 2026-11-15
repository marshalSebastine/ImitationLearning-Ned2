"""
Step 5: was the sheet laid as in the recording? The first picture of R00, R05, R29 and R30 against the reference picture
("ghost" column of the run list: the first picture of one recorded demonstration, shown faintly over the live picture).

For each of the two pictures the line it shows alone is voted onto the table (picture_measures.LineMap.view_of, with
the picture's own joints and the camera correction of its period: "before_move" for the recordings of 6 and 8.11,
"robot_test" for the visit), and the traced drawing is laid onto it, end A on the puck. Two comparisons:

    on the table    the turn between the two laid drawings, and how far apart the line's centre lies at 0, 20, 40 and
                    60 mm from end A (what the sheet's place on the table was, independent of where the arm stood);
    in the picture  the same arc points projected through each picture's own camera: how far apart, in pixels and in
                    millimetres at the puck, the live line and its faint copy lay, and the turn between them in the
                    picture (what Emranul saw when he matched the faint picture).

Run from this folder:  python s05_lay_out_against_ghost.py
Writes s05_lay_out_against_ghost.csv and s05_lay_out_<label>.jpg (the live picture with the faint one over it, the
live line's laid centre in green and the faint one's in magenta).
"""

import math
from pathlib import Path

import cv2
import numpy
import pandas

import visit_common
import picture_measures
import run_measures
from picture_measures import TableCamera
from puck_detector import PuckDetector


LAY_OUT_PARAMETERS = {
    "labels": ["R00", "R05", "R29", "R30", "R31"],
    "arc_points_mm": [0.0, 20.0, 40.0, 60.0],
    "ghost_camera_period": "before_move",
    "ghost_strength": 0.35
}
JOINT_COLUMNS = [f"joint_{number}" for number in range(1, 7)]



def laid_from_picture(
        picture: numpy.ndarray,
        joints: list,
        camera_parameters: dict,
        drawing_mm: numpy.ndarray,
        detector: PuckDetector
) -> dict:
    """Lay the drawing onto the line one picture shows, with end A started on the puck."""

    camera = TableCamera(joints, camera_parameters)
    blob = picture_measures.puck_blob(picture, detector)
    fit = picture_measures.fit_puck_on_table(blob, camera)
    tip = numpy.asarray(camera.table_from_pixels(numpy.array([[320.0, 240.0]]))[0])
    puck = fit["base_xy"]
    margin = 0.12
    line_map = picture_measures.LineMap([min(puck[0], tip[0]) - margin, max(puck[0], tip[0]) + 0.2],
                                        [min(puck[1], tip[1]) - 0.2, max(puck[1], tip[1]) + 0.2])
    hidden = picture_measures.hidden_table_mask(picture.shape[:2], camera, blob)
    seen = line_map.add_picture(picture, camera, hidden)
    middle = picture_measures.middle_of_line(seen["is_line"], line_map.cell_m)
    middle_xy = line_map.point_of(middle["cells"])
    paper_lookup = visit_common.paper_lookup_of(line_map.line_votes, line_map.paper_votes, line_map)
    fitted = visit_common.register_drawing(drawing_mm, middle_xy, paper_lookup, puck)
    return {"camera": camera, "fit": fit, "laid": fitted["drawing"], "turn_deg": fitted["turn_deg"],
            "middle_median_mm": fitted["middle_distance_mm"][0], "middle_cells": len(middle_xy)}



def main() -> None:

    parameters = visit_common.merged_visit_parameters()
    folders = visit_common.run_folders()
    detector = PuckDetector()
    live_camera_parameters = run_measures.camera_parameters_for(parameters["camera_period"])
    ghost_camera_parameters = run_measures.camera_parameters_for(LAY_OUT_PARAMETERS["ghost_camera_period"])
    rows = []
    for label in LAY_OUT_PARAMETERS["labels"]:
        folder = folders[label]
        summary = visit_common.run_summary(folder)
        ghost_path = Path(summary["parameters"]["ghost_path"])
        ghost_steps = pandas.read_csv(ghost_path.parents[1] / "steps.csv")
        live_steps = pandas.read_csv(folder / "steps.csv")
        drawing = visit_common.drawing_of_sheet(visit_common.sheet_of(label))["points_mm"]
        live_picture = cv2.imread(str(folder / "frames" / "000.jpg"))
        ghost_picture = cv2.imread(str(ghost_path))
        live = laid_from_picture(live_picture, [float(live_steps.loc[0, name]) for name in JOINT_COLUMNS], live_camera_parameters, drawing, detector)
        ghost = laid_from_picture(ghost_picture, [float(ghost_steps.loc[0, name]) for name in JOINT_COLUMNS], ghost_camera_parameters, drawing, detector)

        # the gap between the two lines is taken square to the line: from each point of the live line to the nearest
        # point of the faint one (the same arc length would mix in how far the two ends A lie apart along the line,
        # which on the straight sheet 01 the single pictures cannot tell, since the puck hides end A)
        arcs = LAY_OUT_PARAMETERS["arc_points_mm"]
        live_points = numpy.array([live["laid"].point_at(arc) for arc in arcs])
        is_ghost_near = ghost["laid"].arc_mm <= 200.0
        ghost_curve = ghost["laid"].points_m[is_ghost_near]
        table_gaps = numpy.array([1000.0 * numpy.min(numpy.hypot(*(ghost_curve - point[None, :]).T)) for point in live_points])
        turn = (live["turn_deg"] - ghost["turn_deg"] + 180.0) % 360.0 - 180.0
        live_pixels = live["camera"].project_table(live_points)
        ghost_curve_pixels = ghost["camera"].project_table(ghost_curve)
        ghost_curve_pixels = ghost_curve_pixels[numpy.all(numpy.isfinite(ghost_curve_pixels), axis = 1)]
        nearest_ghost = [int(numpy.argmin(numpy.hypot(*(ghost_curve_pixels - pixel[None, :]).T))) for pixel in live_pixels]
        ghost_pixels = ghost_curve_pixels[nearest_ghost]
        pixel_gaps = numpy.hypot(*(live_pixels - ghost_pixels).T)
        # millimetres per pixel at the puck in the live picture: one millimetre on the table at the puck, in pixels
        puck = live["fit"]["base_xy"]
        one_mm = live["camera"].project_table(numpy.array([puck, puck + [0.001, 0.0], puck + [0.0, 0.001]]))
        pixels_per_mm = 0.5 * (numpy.hypot(*(one_mm[1] - one_mm[0])) + numpy.hypot(*(one_mm[2] - one_mm[0])))
        live_direction = live_pixels[2] - live_pixels[0]
        ghost_direction = ghost_pixels[2] - ghost_pixels[0]
        picture_turn = math.degrees(math.atan2(live_direction[1], live_direction[0]) - math.atan2(ghost_direction[1], ghost_direction[0]))
        picture_turn = (picture_turn + 180.0) % 360.0 - 180.0
        rows.append({"label": label, "ghost": ghost_path.parents[1].name, "start_offset_cm": summary["start_offset"]["tip_distance_cm"],
                     "table_turn_deg": round(turn, 1),
                     "table_end_a_apart_mm": round(1000.0 * float(numpy.hypot(*(live["laid"].points_m[0] - ghost["laid"].points_m[0]))), 1), **{f"table_gap_{int(arc)}mm": round(float(gap), 1) for arc, gap in zip(arcs, table_gaps)},
                     "picture_turn_deg": round(picture_turn, 1), **{f"picture_gap_{int(arc)}mm_px": round(float(gap), 1) for arc, gap in zip(arcs, pixel_gaps)},
                     "pixels_per_mm_at_puck": round(float(pixels_per_mm), 2),
                     "picture_gap_20mm_in_mm": round(float(pixel_gaps[1] / pixels_per_mm), 1),
                     "live_fit_median_mm": round(live["middle_median_mm"], 2), "ghost_fit_median_mm": round(ghost["middle_median_mm"], 2),
                     "live_puck_to_end_a_mm": round(1000.0 * float(numpy.hypot(*(live["fit"]["base_xy"] - live["laid"].points_m[0]))), 1),
                     "ghost_puck_to_end_a_mm": round(1000.0 * float(numpy.hypot(*(ghost["fit"]["base_xy"] - ghost["laid"].points_m[0]))), 1)})

        blended = cv2.addWeighted(live_picture, 1.0 - LAY_OUT_PARAMETERS["ghost_strength"], ghost_picture, LAY_OUT_PARAMETERS["ghost_strength"], 0.0)
        for laid, camera, colour in [(live["laid"], live["camera"], (0, 200, 0)), (ghost["laid"], ghost["camera"], (255, 0, 255))]:
            is_near = laid.arc_mm <= 150.0
            pixels = camera.project_table(laid.points_m[is_near])
            pixels = pixels[numpy.all(numpy.isfinite(pixels), axis = 1)]
            cv2.polylines(blended, [numpy.round(pixels).astype(numpy.int32)], False, colour, 2)
        cv2.putText(blended, f"{label}: live line green, faint (reference) line magenta", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(str(visit_common.EVIDENCE_FOLDER / f"s05_lay_out_{label}.jpg"), blended)

    table = pandas.DataFrame(rows)
    table.to_csv(visit_common.EVIDENCE_FOLDER / "s05_lay_out_against_ghost.csv", index = False)
    with pandas.option_context("display.width", 250, "display.max_columns", 40):
        print(table.to_string())



if __name__ == "__main__":
    main()
