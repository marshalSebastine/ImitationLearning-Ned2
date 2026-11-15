"""
Verifier check 13: model E on run 7's own pictures (sheet 05, solid line) with dashes cut into the line by my own edit.

Gaps of 1.0 cm after every 2.0 cm of line (as sheet 06 measures) are painted in the paper's local colour along the
course-photo centre line that my own fit placed on the table, into every picture of run 7; the band is first shifted
by up to 14 px so that it covers the picture's ink. Model E answers the recorded and the cut pictures. The control is
v08_control_run_7_repainted.py (the same line repainted solid: what the edit itself does).

Nothing moves: the same model on changed copies of pictures the solid run produced.

Run:  D:/AI_Project/.venv/Scripts/python.exe v13_run_7_dashes_cut.py > v13_run_7_dashes_cut.out.txt
      D:/AI_Project/.venv/Scripts/python.exe v13_run_7_dashes_cut.py 20 10 15 > v13_run_7_dashes_cut_20_10_15.out.txt   (dash, gap, shift of the pattern in mm)
      D:/AI_Project/.venv/Scripts/python.exe v13_run_7_dashes_cut.py 35 15 0 > v13_run_7_dashes_cut_35_15_0.out.txt     (the practice tracks' pattern)
Writes v13_run_7_dashes_cut<tag>.csv and v13_cut_examples<tag>.jpg. About half a minute each.
"""

import cv2
import numpy
import pandas
import torch

import v08_run_8_solid_line as solid
from verify_common import (VERIFY_FOLDER, VERIFY_PARAMETERS, along_and_beside, colour_masks, pixels_on_table, puck_pixel,
                           register_line, sheet_line, top_view)

from model_mode import ModelPolicy  # noqa: E402

torch.set_num_threads(2)
cv2.setNumThreads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
import sys
DASH_MM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
GAP_MM = int(sys.argv[2]) if len(sys.argv) > 2 else 10
PHASE_MM = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # where along the line the first gap starts, after the first dash
OUT_TAG = "" if len(sys.argv) <= 1 else f"_{DASH_MM}_{GAP_MM}_{PHASE_MM}"


def cut_copy(picture: numpy.ndarray, line: numpy.ndarray, joints: numpy.ndarray, is_robot: numpy.ndarray) -> tuple:
    """Return the picture with gaps painted into the line in the local paper colour, and how many pixels were painted."""

    masks = colour_masks(picture)
    whole_band = solid.line_band_mask(line, joints, half_width_m = 0.0055)
    _, shift = solid.aligned_band(whole_band, masks["ink"] & ~is_robot)
    wide_puck = cv2.dilate(masks["puck"].astype(numpy.uint8), numpy.ones((7, 7), numpy.uint8)) > 0
    edited = picture.astype(numpy.float32)
    painted = 0
    period = DASH_MM + GAP_MM
    for start in range(DASH_MM + PHASE_MM, len(line) - 2, period):
        piece = line[start:start + GAP_MM + 1]
        if len(piece) < 3:
            continue
        gap = solid.line_band_mask(piece, joints, half_width_m = 0.008)
        gap = numpy.roll(numpy.roll(gap, shift[1], axis = 0), shift[0], axis = 1)
        gap &= ~wide_puck & ~is_robot
        if gap.sum() < 20:
            continue
        ring = (cv2.dilate(gap.astype(numpy.uint8), numpy.ones((31, 31), numpy.uint8)) > 0) & ~gap & masks["paper"]
        if ring.sum() < 30:
            continue
        paper_colour = numpy.median(picture[ring], axis = 0)
        weight = cv2.GaussianBlur(gap.astype(numpy.float32), (0, 0), 1.2)
        edited = edited * (1.0 - weight[:, :, None]) + paper_colour[None, None, :] * weight[:, :, None]
        painted += int(gap.sum())
    return numpy.clip(edited, 0, 255).astype(numpy.uint8), painted


def main() -> None:

    folder = VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS["run_7_folder"]
    steps = pandas.read_csv(folder / "steps.csv")
    pictures = [str(folder / name) for name in steps["frame"]]
    joints = steps[JOINT_NAMES].to_numpy()
    view = top_view(pictures, joints)
    placed = register_line(sheet_line("05"), view)
    line = placed["line_m"]
    puck_xy = numpy.full((len(steps), 2), numpy.nan)
    for position, picture_path in enumerate(pictures):
        found = puck_pixel(colour_masks(cv2.imread(picture_path))["puck"])
        if found is not None:
            puck_xy[position] = pixels_on_table(numpy.array([[found[0], found[1]]]), joints[position], VERIFY_PARAMETERS["puck_point_height_m"])[0]
    index, beside = along_and_beside(line, puck_xy)
    if index[-1] < index[0]:
        line = line[::-1].copy()
        index, beside = along_and_beside(line, puck_xy)

    policy = ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", solid.MODEL_FOLDER / "checkpoints" / "all_sheets.pt", solid.MODEL_FOLDER / "metrics.json")
    torch.set_num_threads(2)
    is_robot = solid.robot_mask_of_run([cv2.imread(picture_path) for picture_path in pictures])
    rows, examples = [], []
    for position, picture_path in enumerate(pictures):
        picture = cv2.imread(picture_path)
        edited, painted = cut_copy(picture, line, joints[position], is_robot)
        recorded = policy.decide(solid.jpeg_bytes(picture))
        cut = policy.decide(solid.jpeg_bytes(edited))
        rows.append({"step": position, "joint_4": numpy.degrees(joints[position, 3]), "painted_px": painted,
                     "recorded_angle": recorded["angle_deg"], "cut_angle": cut["angle_deg"],
                     "change_deg": float(abs(solid.wrapped(cut["angle_deg"] - recorded["angle_deg"]))),
                     "recorded_length": numpy.hypot(recorded["tip_forward_cm"], recorded["tip_left_cm"]),
                     "cut_length": numpy.hypot(cut["tip_forward_cm"], cut["tip_left_cm"]),
                     "recorded_twist": recorded["twist_deg"], "cut_twist": cut["twist_deg"],
                     "recorded_stop": recorded["stop_probability"], "cut_stop": cut["stop_probability"]})
        if position in (3, 10, 16, 22, 28, 34, 45, 58, 62, 70, 78, 85):
            examples.append(numpy.hstack([cv2.resize(picture, (320, 240)), cv2.resize(edited, (320, 240))]))
    table = pandas.DataFrame(rows)
    table.round(3).to_csv(VERIFY_FOLDER / f"v13_run_7_dashes_cut{OUT_TAG}.csv", index = False)
    cv2.imwrite(str(VERIFY_FOLDER / f"v13_cut_examples{OUT_TAG}.jpg"), numpy.vstack([numpy.hstack(examples[start:start + 3]) for start in range(0, len(examples), 3)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    control = pandas.read_csv(VERIFY_FOLDER / "v08_control_run_7_repainted.csv")
    print(f"run 7 (sheet 05), {len(table)} pictures; dashes of {DASH_MM} mm with gaps of {GAP_MM} mm cut into the line; pictures with a gap painted: {int((table['painted_px'] > 0).sum())}")
    print(table.round(2).to_string(index = False))
    for name, first, last in (("steps 0 to 15 (straight start)", 0, 15), ("steps 16 to 30 (into the loop)", 16, 30), ("steps 31 to 56 (walk round the puck)", 31, 56),
                              ("steps 57 to 70", 57, 70), ("steps 71 to 88", 71, 88), ("all steps", 0, 88)):
        part = table[(table["step"] >= first) & (table["step"] <= last)]
        control_part = control[(control["step"] >= first) & (control["step"] <= last)]
        print(f"  {name}: direction changed by {part['change_deg'].mean():.1f} deg on average ({part['change_deg'].max():.1f} at most; control, line repainted solid:"
              f" {control_part['change_deg'].mean():.1f}); twist summed {part['recorded_twist'].sum():+.1f} recorded, {part['cut_twist'].sum():+.1f} cut"
              f" (control {control_part['solid_twist'].sum():+.1f}); length {part['recorded_length'].mean():.2f} against {part['cut_length'].mean():.2f} cm")
    print(f"  answers beyond 90 deg: {int((table['recorded_angle'].abs() > 90).sum())} recorded, {int((table['cut_angle'].abs() > 90).sum())} cut;"
          f" steps with a change over 20 deg: {int((table['change_deg'] > 20).sum())} of {len(table)} (control {int((control['change_deg'] > 20).sum())});"
          f" stop chance 0.5 or more before step 87: {int((table['recorded_stop'].iloc[:87] >= 0.5).sum())} recorded, {int((table['cut_stop'].iloc[:87] >= 0.5).sum())} cut;"
          f" at steps 87 and 88: {table['recorded_stop'].iloc[87:].round(2).tolist()} recorded, {table['cut_stop'].iloc[87:].round(2).tolist()} cut")
    walk = table[(table["step"] >= 31) & (table["step"] <= 56)]
    print(f"  the walk (steps 31 to 56): twist asked {walk['recorded_twist'].sum():.0f} deg recorded against {walk['cut_twist'].sum():.0f} deg cut"
          f" ({100 * (1 - walk['cut_twist'].sum() / walk['recorded_twist'].sum()):.0f} % less); answers beyond 90 deg {int((walk['recorded_angle'].abs() > 90).sum())} against {int((walk['cut_angle'].abs() > 90).sum())}")



if __name__ == "__main__":
    main()
