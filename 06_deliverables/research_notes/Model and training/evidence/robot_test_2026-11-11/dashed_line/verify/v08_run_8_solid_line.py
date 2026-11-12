"""
Verifier check 8: model E on run 8's own pictures, as recorded and with the dashed line painted solid by my own edit.

1. The saved model E (all sheets) is run on the 41 recorded pictures: do its answers match the log?
2. My own edit: the course-photo centre line, placed on the table by my own fit (v04), is drawn 1.1 cm wide into
   every picture in the picture's own ink colour, after the band was shifted by up to 14 px so that it covers the
   most ink the picture shows (the camera model is a few pixels off per picture); it is painted only on bluish
   pixels (paper, ink, shadowed paper), never on the puck or on the robot's own parts (a mask of the pixels that stay
   robot-coloured through the run). The model answers the edited pictures. Nothing moves: this is the same model on changed copies of
   pictures that the dashed run produced, not a second run.
3. Each answer's direction on the table against the line's direction at the puck's place (+ = to the left of the
   direction of travel, which is the outside of sheet 06's loop when driven from end A).

Run:  D:/AI_Project/.venv/Scripts/python.exe v08_run_8_solid_line.py > v08_run_8_solid_line.out.txt
Writes v08_run_8_solid_line.csv and v08_edit_examples.jpg. About half a minute. Nothing connects to the robot.
"""

import cv2
import numpy
import pandas
import torch

from verify_common import (ROOT_FOLDER, VERIFY_FOLDER, VERIFY_PARAMETERS, along_and_beside, colour_masks, pixels_on_table,
                           puck_pixel, register_line, sheet_line, top_view)

from config import WORKSPACE_PARAMETERS  # noqa: E402
from forearm_camera_model import project_points  # noqa: E402
from model_mode import ModelPolicy  # noqa: E402

torch.set_num_threads(2)
cv2.setNumThreads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
MODEL_FOLDER = ROOT_FOLDER / "04_runs" / "gpu_ki014" / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets"
ALIGN_MOST_PX = 14


def robot_mask_of_run(pictures: list) -> numpy.ndarray:
    """
    Return the pixels that show the robot's own parts in this run: the camera sits on the forearm and the wrist and
    hand were held, so the block, the wrist and the covers stay at the same pixels while the table moves. A pixel
    counts as robot when it is bluish or black and not paper-bright (hue 85 to 130 of 180, value under 190: the black
    block reads as very dark blue with this camera) or strongly blue like the covers (saturation 140 or more) in at
    least 85 % of the run's pictures, in the lower half of the
    picture; cleaned and widened by 6 px. (A first version used the spread of the grey value; shadows made it miss
    most of the block, and the gap edit then painted paper onto the block.)
    """

    count = numpy.zeros((480, 640), numpy.float32)
    for picture in pictures:
        hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
        is_dark_bluish = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 2] < 190)
        is_cover_blue = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] >= 140)
        count += (is_dark_bluish | is_cover_blue).astype(numpy.float32)
    is_static = (count >= 0.85 * len(pictures)).astype(numpy.uint8)
    is_static[:240] = 0
    is_static = cv2.morphologyEx(is_static, cv2.MORPH_OPEN, numpy.ones((5, 5), numpy.uint8))
    is_static = cv2.morphologyEx(is_static, cv2.MORPH_CLOSE, numpy.ones((25, 25), numpy.uint8))
    return cv2.dilate(is_static, numpy.ones((13, 13), numpy.uint8)) > 0


def aligned_band(band: numpy.ndarray, is_ink: numpy.ndarray) -> tuple:
    """Shift the band by up to 14 px so that it covers the most ink pixels of the picture; returns the band and the shift."""

    if band.sum() < 300 or is_ink.sum() < 300:
        return band, (0, 0)
    best = (int((band & is_ink).sum()), 0, 0)
    for shift_rows in range(-ALIGN_MOST_PX, ALIGN_MOST_PX + 1, 2):
        for shift_columns in range(-ALIGN_MOST_PX, ALIGN_MOST_PX + 1, 2):
            moved = numpy.roll(numpy.roll(band, shift_rows, axis = 0), shift_columns, axis = 1)
            covered = int((moved & is_ink).sum())
            if covered > best[0]:
                best = (covered, shift_rows, shift_columns)
    moved = numpy.roll(numpy.roll(band, best[1], axis = 0), best[2], axis = 1)
    return moved, (best[2], best[1])


def line_band_mask(line_m: numpy.ndarray, joints: numpy.ndarray, half_width_m: float = 0.005) -> numpy.ndarray:
    """Return the picture mask of the 1 cm wide band along a line placed on the table."""

    plane = WORKSPACE_PARAMETERS["table_plane"]
    tangent = numpy.gradient(line_m, axis = 0)
    tangent /= numpy.linalg.norm(tangent, axis = 1)[:, None]
    normal = numpy.column_stack([-tangent[:, 1], tangent[:, 0]])
    mask = numpy.zeros((480, 640), numpy.uint8)
    edges = []
    for side in (1.0, -1.0):
        points = line_m + side * half_width_m * normal
        height = plane["a"] * points[:, 0] + plane["b"] * points[:, 1] + plane["c"]
        edges.append(project_points(numpy.column_stack([points, height]), list(joints)))
    left, right = edges
    for position in range(len(line_m) - 1):
        quad = numpy.array([left[position], left[position + 1], right[position + 1], right[position]])
        if not numpy.all(numpy.isfinite(quad)):
            continue
        if (quad[:, 0].max() < -200) or (quad[:, 0].min() > 840) or (quad[:, 1].max() < -200) or (quad[:, 1].min() > 680):
            continue
        cv2.fillConvexPoly(mask, numpy.round(quad).astype(numpy.int32), 1)
    return mask > 0


def solid_copy(picture: numpy.ndarray, band: numpy.ndarray, is_robot: numpy.ndarray) -> numpy.ndarray:
    """Return the picture with the band painted in the picture's own ink colour where paper or ink shows."""

    masks = colour_masks(picture)
    wide_puck = cv2.dilate(masks["puck"].astype(numpy.uint8), numpy.ones((7, 7), numpy.uint8)) > 0
    # paper in the robot's shadow is darker than my paper rule allows: every bluish pixel that is not robot counts
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    is_bluish = (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 2] >= 70)
    paintable = (masks["paper"] | masks["ink"] | is_bluish) & ~wide_puck & ~is_robot
    ink_colour = numpy.median(picture[masks["ink"]], axis = 0) if masks["ink"].sum() > 200 else numpy.array([150.0, 95.0, 55.0])
    paint = (band & paintable).astype(numpy.float32)
    paint = cv2.GaussianBlur(paint, (0, 0), 1.0)
    edited = picture.astype(numpy.float32) * (1.0 - paint[:, :, None]) + ink_colour[None, None, :] * paint[:, :, None]
    return numpy.clip(edited, 0, 255).astype(numpy.uint8)


def jpeg_bytes(picture: numpy.ndarray) -> bytes:

    is_fine, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes()


def wrapped(angle: numpy.ndarray) -> numpy.ndarray:

    return (numpy.asarray(angle) + 180.0) % 360.0 - 180.0


def main() -> None:

    folder = VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS["run_8_folder"]
    steps = pandas.read_csv(folder / "steps.csv")
    pictures = [str(folder / name) for name in steps["frame"]]
    joints = steps[JOINT_NAMES].to_numpy()
    view = top_view(pictures, joints)
    placed = register_line(sheet_line("06"), view)
    puck_xy = numpy.full((len(steps), 2), numpy.nan)
    for position, picture_path in enumerate(pictures):
        found = puck_pixel(colour_masks(cv2.imread(picture_path))["puck"])
        if found is not None:
            puck_xy[position] = pixels_on_table(numpy.array([[found[0], found[1]]]), joints[position], VERIFY_PARAMETERS["puck_point_height_m"])[0]
    line = placed["line_m"]
    index, beside = along_and_beside(line, puck_xy)
    if index[-1] < index[0]:
        line = line[::-1].copy()
        index, beside = along_and_beside(line, puck_xy)
    tangent = numpy.gradient(line, axis = 0)
    line_direction = numpy.degrees(numpy.arctan2(tangent[index, 1], tangent[index, 0]))

    policy = ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", MODEL_FOLDER / "checkpoints" / "all_sheets.pt", MODEL_FOLDER / "metrics.json")
    torch.set_num_threads(2)
    rows, examples = [], []
    is_robot = robot_mask_of_run([cv2.imread(picture_path) for picture_path in pictures])
    cv2.imwrite(str(VERIFY_FOLDER / "v08_robot_mask.png"), 255 * is_robot.astype(numpy.uint8))
    shifts = []
    for position, picture_path in enumerate(pictures):
        picture = cv2.imread(picture_path)
        band = line_band_mask(line, joints[position], half_width_m = 0.0055)
        band, shift = aligned_band(band, colour_masks(picture)["ink"] & ~is_robot)
        shifts.append(shift)
        edited = solid_copy(picture, band, is_robot)
        answers = {
            "recorded": policy.decide(open(picture_path, "rb").read()),
            "recoded": policy.decide(jpeg_bytes(picture)),
            "solid": policy.decide(jpeg_bytes(edited))
        }
        ink_before = colour_masks(picture)["ink"]
        ink_after = colour_masks(edited)["ink"]
        row = {"step": int(steps["step"].iloc[position]), "joint_4": numpy.degrees(joints[position, 3]), "beside_cm": 100.0 * beside[position],
               "logged_angle": steps["model_angle_deg"].iloc[position], "logged_twist": steps["model_twist_deg"].iloc[position],
               "logged_stop": steps["model_stop_probability"].iloc[position], "heading": steps["picture_heading_deg"].iloc[position],
               "line_direction": line_direction[position], "ink_px_before": int(ink_before.sum()), "ink_px_after": int(ink_after.sum())}
        for name, answer in answers.items():
            row[f"{name}_angle"] = answer["angle_deg"]
            row[f"{name}_length"] = numpy.hypot(answer["tip_forward_cm"], answer["tip_left_cm"])
            row[f"{name}_twist"] = answer["twist_deg"]
            row[f"{name}_stop"] = answer["stop_probability"]
            row[f"{name}_vs_line"] = float(wrapped(row["heading"] + answer["angle_deg"] - line_direction[position]))
        rows.append(row)
        if position in (5, 12, 17, 18, 20, 22, 23, 24, 25, 26, 27, 29):
            examples.append(numpy.hstack([cv2.resize(picture, (320, 240)), cv2.resize(edited, (320, 240))]))
    table = pandas.DataFrame(rows)
    table["change_deg"] = numpy.abs(wrapped(table["solid_angle"] - table["recoded_angle"]))
    table.round(3).to_csv(VERIFY_FOLDER / "v08_run_8_solid_line.csv", index = False)
    sheet = numpy.vstack([numpy.hstack(examples[start:start + 3]) for start in range(0, len(examples), 3)])
    cv2.imwrite(str(VERIFY_FOLDER / "v08_edit_examples.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    print(f"band shifted per picture to cover the most ink: {numpy.abs(numpy.array(shifts)).mean(axis = 0).round(1).tolist()} px on average (columns, rows),"
          f" {numpy.abs(numpy.array(shifts)).max(axis = 0).tolist()} at most; robot mask: {int(is_robot.sum())} px")
    print(f"line placed: ink cells {placed['ink_to_line_mm']:.1f} mm (median) from the line, {100 * placed['line_on_ink_share']:.0f} % of seen line points on ink")
    gap = numpy.abs(wrapped(table["recorded_angle"] - table["logged_angle"]))
    print(f"1. the model run again on the recorded pictures against the log: direction differs by {gap.max():.3f} deg at most,"
          f" twist by {(table['recorded_twist'] - table['logged_twist']).abs().max():.4f} deg, stop chance by {(table['recorded_stop'] - table['logged_stop']).abs().max():.5f}")
    gap = numpy.abs(wrapped(table["recoded_angle"] - table["recorded_angle"]))
    print(f"   saving the same picture again as JPEG changes the direction by {gap.mean():.2f} deg on average, {gap.max():.2f} at most")
    print("\n2. and 3. per step (angles in the picture: 0 = up, + = left; 'vs_line': the asked direction on the table against the line's direction at the puck, + = outside)")
    print(table[["step", "joint_4", "beside_cm", "recoded_angle", "solid_angle", "change_deg", "recoded_vs_line", "solid_vs_line", "recoded_length", "solid_length",
                 "recoded_twist", "solid_twist", "recoded_stop", "solid_stop", "ink_px_before", "ink_px_after"]].round(2).to_string(index = False))
    for name, first, last in (("steps 0 to 15", 0, 15), ("steps 16 to 30", 16, 30), ("steps 31 to 40", 31, 40), ("steps 19 to 23", 19, 23), ("steps 24 to 27", 24, 27)):
        part = table[(table["step"] >= first) & (table["step"] <= last)]
        print(f"   {name}: direction changed by {part['change_deg'].mean():.1f} deg on average ({part['change_deg'].max():.1f} at most);"
              f" against the line: recorded {part['recoded_vs_line'].mean():+.1f}, solid {part['solid_vs_line'].mean():+.1f} deg on average;"
              f" twist summed {part['recoded_twist'].sum():+.1f} against {part['solid_twist'].sum():+.1f} deg;"
              f" length {part['recoded_length'].mean():.2f} against {part['solid_length'].mean():.2f} cm")
    print(f"   whole run: twist summed {table['recoded_twist'].sum():.1f} deg recorded, {table['solid_twist'].sum():.1f} deg solid;"
          f" answers beyond 90 deg: {int((table['recoded_angle'].abs() > 90).sum())} recorded, {int((table['solid_angle'].abs() > 90).sum())} solid")
    print(f"   stop chance at steps 38 to 40: recorded {table['recoded_stop'].iloc[38:41].round(2).tolist()}, solid {table['solid_stop'].iloc[38:41].round(2).tolist()}")



if __name__ == "__main__":
    main()
