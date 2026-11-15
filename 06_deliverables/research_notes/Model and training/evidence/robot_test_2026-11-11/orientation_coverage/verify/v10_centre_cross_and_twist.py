"""
The verifier's own check of two calculated claims.

1. "With the paper's centre on a cross 31 cm in front of the base axis all 576 placements are possible (tip 15.1 to
   46.9 cm)": the six course lines and the six practice lines, 24 turns, both ends; the tip follows the line 2.5 cm
   behind the puck (and starts 2.5 cm behind the start cap on the line's backward extension).
2. "The demonstrators turn the camera into the bend ahead (correlation of the twist with the bend 5 to 10 rows ahead
   -0.45)".

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v10_centre_cross_and_twist.py > v10_centre_cross_and_twist.out.txt
"""

import re
import sys
from pathlib import Path

import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from setup_visuals import load_sheet_lines  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

PRACTICE_FOLDER = ROOT_FOLDER / "06_deliverables" / "research_notes" / "Model and training" / "test_tracks" / "practice"


def wrap(angle_deg):
    return (numpy.asarray(angle_deg, dtype = float) + 180.0) % 360.0 - 180.0


def resampled(line, step = 0.1):
    arc = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(line, axis = 0), axis = 1))])
    samples = numpy.arange(0.0, arc[-1], step)
    return numpy.column_stack([numpy.interp(samples, arc, line[:, 0]), numpy.interp(samples, arc, line[:, 1])])


shapes = {f"sheet_{number}": resampled(numpy.asarray(sheet["line"], dtype = float)) for number, sheet in load_sheet_lines().items()}
for path in sorted(PRACTICE_FOLDER.glob("practice_*.svg")):
    text = path.read_text(encoding = "utf-8")
    height_mm = float(re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', text).group(2))
    numbers = re.findall(r"[ML] ?([0-9.]+)[ ,]([0-9.]+)", re.search(r'<path d="([^"]+)"', text).group(1))
    points = numpy.array([[float(x) / 10.0, (height_mm - float(y)) / 10.0] for x, y in numbers])
    shapes[path.stem] = resampled(points)

print("=== 1. Paper centre on (31, 0) cm: reach of the tip over 24 turns and both ends (calculated) ===")
paper_centre = numpy.array([10.5, 14.85])
total, possible, least_all, most_all = 0, 0, 1e9, 0.0
for name, line in shapes.items():
    least_shape, most_shape, count_ok = 1e9, 0.0, 0
    for end in ("A", "B"):
        ordered = line if end == "A" else line[::-1]
        for turn in numpy.arange(0.0, 360.0, 15.0):
            angle = numpy.radians(turn)
            rotation = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
            table_line = (ordered - paper_centre[None, :]) @ rotation.T + numpy.array([31.0, 0.0])
            # the tip: 2.5 cm behind each line point along the local direction of travel (25 samples = 2.5 cm), and the
            # start 2.5 cm behind the start cap on the line's backward extension
            tangent_start = table_line[5] - table_line[0]
            tangent_start = tangent_start / numpy.linalg.norm(tangent_start)
            tips = numpy.vstack([table_line[0] - 2.5 * tangent_start, table_line[:-25]])
            reach = numpy.hypot(tips[:, 0], tips[:, 1])
            total += 1
            is_ok = (reach.min() >= 12.0) and (reach.max() <= 47.5) and (tips[:, 0].min() >= 11.5) and (tips[:, 0].max() <= 49.0) and (tips[:, 1].min() >= -30.0) and (tips[:, 1].max() <= 25.0)
            possible += int(is_ok)
            count_ok += int(is_ok)
            least_shape, most_shape = min(least_shape, reach.min()), max(most_shape, reach.max())
    least_all, most_all = min(least_all, least_shape), max(most_all, most_shape)
    print(f"  {name}: possible {count_ok} of 48; tip reach {least_shape:.1f} to {most_shape:.1f} cm")
print(f"  all shapes: possible {possible} of {total}; tip reach {least_all:.1f} to {most_all:.1f} cm (limits used: reach 12 to 47.5, x 11.5 to 49, y -30 to 25)")

print("\n=== 2. The forearm twist against the bend ahead (measured from the audit-cleaned index) ===")
frames = load_training_frames({"is_audit_cleaning": True}).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
grouped = frames.groupby("episode_id", sort = False)
x = frames["tip_x_m"].to_numpy()
y = frames["tip_y_m"].to_numpy()


def path_direction(shift_from, shift_to):
    x_from = grouped["tip_x_m"].shift(-shift_from).to_numpy() if shift_from else x
    y_from = grouped["tip_y_m"].shift(-shift_from).to_numpy() if shift_from else y
    x_to = grouped["tip_x_m"].shift(-shift_to).to_numpy()
    y_to = grouped["tip_y_m"].shift(-shift_to).to_numpy()
    length = numpy.hypot(x_to - x_from, y_to - y_from)
    direction = numpy.degrees(numpy.arctan2(y_to - y_from, x_to - x_from))
    direction[~(length >= 0.009)] = numpy.nan
    return direction


now = path_direction(0, 3)
later = path_direction(5, 10)
bend_ahead = wrap(later - now)
twist = frames["forearm_change_deg"].to_numpy()
valid = numpy.isfinite(bend_ahead) & numpy.isfinite(twist) & (frames["action"] != "stop").to_numpy()
print(f"  rows with both: {valid.sum()}; correlation of the twist with the bend ahead: {numpy.corrcoef(twist[valid], bend_ahead[valid])[0, 1]:+.2f}")
in_picture = wrap(now - frames["picture_heading_deg"].to_numpy())
valid_picture = numpy.isfinite(in_picture) & numpy.isfinite(twist) & (numpy.abs(in_picture) <= 90)
print(f"  correlation of the twist with where the path lies in the picture (within 90 degrees of its top): {numpy.corrcoef(twist[valid_picture], in_picture[valid_picture])[0, 1]:+.2f}")
left_bend = valid & (bend_ahead > 30)
right_bend = valid & (bend_ahead < -30)
print(f"  mean twist where the path will turn left by more than 30 degrees: {twist[left_bend].mean():+.2f}; right: {twist[right_bend].mean():+.2f} (joint 4 falling turns the camera left)")
