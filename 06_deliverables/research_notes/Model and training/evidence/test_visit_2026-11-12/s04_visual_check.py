"""
Step 4: a check of the measuring by eye. On a sample of pictures the measured points are drawn into the picture itself,
so that a reader can see whether they sit where the puck and the printed line really are, and read the puck's offset
from the printed ink without the measuring's own line.

Drawn on each picture (cropped around the puck): the fitted puck's foot as a thin cyan circle and its centre as a cyan
dot; the laid drawing's centre line ("method B") in green and the drawing laid on this picture alone ("method C", from s07) in
magenta, over 60 mm either side of the puck; and a ruler square to the
drawn line through the point of the line nearest the puck, with ticks every 5 mm (long ticks at 0, 10 and 20 mm), in
white. The puck's centre lies on the ruler. The reader writes down where the middle of the dark printed band crosses the
ruler and where the puck's centre lies on it; the difference is the offset by eye. The tiles carry only a number: the
measured values are written to s04_visual_check_values.csv, to be opened only after the reading.

Sample: 24 pictures drawn at random (seed 20261113) from the sheet 03 and 04 runs, and the 10 pictures on which the
verdict of a borderline run rests.

Run from this folder after s03 and s07:  python s04_visual_check.py
Writes s04_visual_check_values.csv and s04_visual_check_tiles_<n>.jpg (six tiles each).
"""

import math
import random

import cv2
import numpy
import pandas

import visit_common
import picture_measures
import run_measures
from picture_measures import TableCamera
from puck_detector import PuckDetector


VISUAL_PARAMETERS = {
    "seed": 20261113,
    "random_pictures": 24,
    "deciding_pictures": [("R05", 6), ("R06", 9), ("R06", 10), ("R12", 9), ("R12", 10), ("R18", 7), ("R15", 7), ("R15", 8),
                          ("R10", 14), ("R19", 22), ("R02", 28), ("X01", 25)],
    "crop_half_size_px": [150, 110],
    "tiles_per_sheet": 6,
    "ruler_half_length_mm": 25.0,
    "line_half_span_mm": 60.0
}
JOINT_COLUMNS = [f"joint_{number}" for number in range(1, 7)]



def laid_drawing_of(label: str, lay_out: pandas.DataFrame) -> visit_common.DrawingOnTable:
    """The drawing laid for a run in step 3, rebuilt from s03_lay_out.csv."""

    row = lay_out[lay_out["label"] == label].iloc[0]
    points = visit_common.drawing_of_sheet(row["sheet"] if isinstance(row["sheet"], str) else f"{int(row['sheet']):02d}")["points_mm"]
    if row["direction"] == "from end B":
        points = points[::-1].copy()
    return visit_common.DrawingOnTable(points, math.radians(row["turn_deg"]), [row["shift_x_m"], row["shift_y_m"]])



def drawn_tile(
        picture: numpy.ndarray,
        camera: TableCamera,
        fit: dict,
        laid: visit_common.DrawingOnTable,
        arc_mm: float,
        number: int,
        laid_c: visit_common.DrawingOnTable = None
) -> numpy.ndarray:
    """The picture with the measured points drawn in, cropped around the puck, with its number."""

    drawn = picture.copy()
    angles = numpy.linspace(0.0, 2.0 * math.pi, 72)
    foot = fit["base_xy"]
    ring = foot[None, :] + 0.012 * numpy.column_stack([numpy.cos(angles), numpy.sin(angles)])
    cv2.polylines(drawn, [numpy.round(camera.project_table(ring)).astype(numpy.int32)], True, (255, 255, 0), 1, cv2.LINE_AA)
    centre = camera.project_table(foot[None, :])[0]
    cv2.circle(drawn, tuple(int(round(value)) for value in centre), 3, (255, 255, 0), -1, cv2.LINE_AA)
    is_near = numpy.abs(laid.arc_mm - arc_mm) <= VISUAL_PARAMETERS["line_half_span_mm"]
    line_pixels = camera.project_table(laid.points_m[is_near])
    cv2.polylines(drawn, [numpy.round(line_pixels).astype(numpy.int32)], False, (0, 220, 0), 1, cv2.LINE_AA)
    if laid_c is not None:
        is_near_c = numpy.abs(laid_c.arc_mm - arc_mm) <= VISUAL_PARAMETERS["line_half_span_mm"]
        pixels_c = camera.project_table(laid_c.points_m[is_near_c])
        cv2.polylines(drawn, [numpy.round(pixels_c).astype(numpy.int32)], False, (255, 0, 255), 1, cv2.LINE_AA)
    nearest = int(numpy.argmin(numpy.abs(laid.arc_mm - arc_mm)))
    base = laid.points_m[nearest]
    tangent = laid.tangents[nearest]
    left = numpy.array([-tangent[1], tangent[0]])
    half = VISUAL_PARAMETERS["ruler_half_length_mm"]
    ends = camera.project_table(numpy.array([base - half / 1000.0 * left, base + half / 1000.0 * left]))
    cv2.line(drawn, tuple(numpy.round(ends[0]).astype(int)), tuple(numpy.round(ends[1]).astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
    for tick_mm in numpy.arange(-20.0, 20.5, 5.0):
        length = 0.003 if tick_mm % 10 == 0 else 0.0015
        point = base + tick_mm / 1000.0 * left
        tick = camera.project_table(numpy.array([point - length * tangent, point + length * tangent]))
        cv2.line(drawn, tuple(numpy.round(tick[0]).astype(int)), tuple(numpy.round(tick[1]).astype(int)), (255, 255, 255), 1, cv2.LINE_AA)
    # the ruler's left end (positive offsets) carries an L
    label_pixel = camera.project_table((base + (half + 3.0) / 1000.0 * left)[None, :])[0]
    cv2.putText(drawn, "L", tuple(numpy.round(label_pixel).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    half_width, half_height = VISUAL_PARAMETERS["crop_half_size_px"]
    centre_u = int(numpy.clip(round(centre[0]), half_width, drawn.shape[1] - half_width))
    centre_v = int(numpy.clip(round(centre[1]), half_height, drawn.shape[0] - half_height))
    tile = drawn[centre_v - half_height:centre_v + half_height, centre_u - half_width:centre_u + half_width].copy()
    tile = cv2.resize(tile, None, fx = 1.4, fy = 1.4, interpolation = cv2.INTER_CUBIC)
    cv2.putText(tile, f"#{number}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    return tile



def main() -> None:

    parameters = visit_common.merged_visit_parameters()
    steps = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_steps.csv")
    method_c = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s07_steps_method_c.csv")
    lay_out = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_lay_out.csv", dtype = {"sheet": str})
    folders = visit_common.run_folders()
    scored_labels = [label for label in lay_out["label"] if label != "demo" and visit_common.sheet_of(label) in ("03", "04")]
    candidates = steps[steps["label"].isin(scored_labels) & steps["offset_a_mm"].notna()]
    chooser = random.Random(VISUAL_PARAMETERS["seed"])
    chosen = [tuple(row) for row in candidates[["label", "step"]].itertuples(index = False)]
    sample = chooser.sample(chosen, VISUAL_PARAMETERS["random_pictures"]) + list(VISUAL_PARAMETERS["deciding_pictures"])
    camera_parameters = run_measures.camera_parameters_for(parameters["camera_period"])
    detector = PuckDetector()
    values, tiles = [], []
    for number, (label, step) in enumerate(sample, start = 1):
        run_steps = pandas.read_csv(folders[label] / "steps.csv")
        picture = cv2.imread(str(folders[label] / "frames" / f"{int(step):03d}.jpg"))
        camera = TableCamera([float(run_steps.loc[int(step), name]) for name in JOINT_COLUMNS], camera_parameters)
        blob = picture_measures.puck_blob(picture, detector)
        fit = picture_measures.fit_puck_on_table(blob, camera)
        measured = steps[(steps["label"] == label) & (steps["step"] == step)].iloc[0]
        laid = laid_drawing_of(label, lay_out)
        laid_c = None
        found_c = method_c[(method_c["label"] == label) & (method_c["step"] == step)]
        if len(found_c) and numpy.isfinite(found_c.iloc[0]["c_turn_rad"]):
            laid_c = visit_common.DrawingOnTable(laid.drawing_mm, found_c.iloc[0]["c_turn_rad"], [found_c.iloc[0]["c_shift_x_m"], found_c.iloc[0]["c_shift_y_m"]])
        tiles.append(drawn_tile(picture, camera, fit, laid, float(measured["arc_mm"]), number, laid_c))
        values.append({"number": number, "label": label, "step": int(step), "is_deciding": number > VISUAL_PARAMETERS["random_pictures"],
                       "offset_a_mm": round(float(measured["offset_a_mm"]), 1), "offset_b_mm": round(float(measured["offset_b_mm"]), 1),
                       "offset_c_mm": round(float(found_c.iloc[0]["offset_c_mm"]), 1) if len(found_c) else None,
                       "arc_mm": round(float(measured["arc_mm"]), 0), "fit_quality": round(float(fit["quality"]), 3)})
    per_sheet = VISUAL_PARAMETERS["tiles_per_sheet"]
    for start in range(0, len(tiles), per_sheet):
        group = tiles[start:start + per_sheet]
        while len(group) < per_sheet:
            group.append(numpy.zeros_like(tiles[0]))
        rows = [numpy.hstack(group[index:index + 3]) for index in range(0, per_sheet, 3)]
        cv2.imwrite(str(visit_common.EVIDENCE_FOLDER / f"s04_visual_check_tiles_{start // per_sheet + 1}.jpg"), numpy.vstack(rows))
    pandas.DataFrame(values).to_csv(visit_common.EVIDENCE_FOLDER / "s04_visual_check_values.csv", index = False)
    print(f"{len(values)} pictures drawn into {math.ceil(len(tiles) / per_sheet)} tile sheets")



if __name__ == "__main__":
    main()
