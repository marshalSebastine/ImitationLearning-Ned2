"""
The verifier's own check that the ten robot runs of 11.11.26 started in lay-outs the demonstrations hold: the puck's
place and the direction in which the line leaves the puck (2.5 to 6 cm from it) in the first picture of every run,
against the same measure in the first kept picture of every start-mark demonstration of the same sheet.

Own finders (red blob, bluish line pixels joined to the puck) on the fitted forearm camera. Nothing connects to the robot.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe v8_runs_layout.py > v8_runs_layout.out.txt
"""

import sys
from pathlib import Path

import cv2
import numpy
import pandas

VERIFY_FOLDER = Path(__file__).resolve().parent
ROOT_FOLDER = VERIFY_FOLDER.parents[6]
sys.path.insert(0, str(ROOT_FOLDER / "02_code"))

from config import table_height  # noqa: E402
from forearm_camera_model import pixels_to_plane  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 400)

RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
ROBOT_RUNS = [(1, "2026-11-11_172021_check_no_puck", "01"), (2, "2026-11-11_172421_E_sheet01_run1", "01"), (3, "2026-11-11_172743_E_sheet02_run1", "02"),
              (4, "2026-11-11_173036_E_sheet02_run2", "02"), (5, "2026-11-11_173549_E_sheet03_run1", "03"), (6, "2026-11-11_174108_E_sheet04_run1", "04"),
              (7, "2026-11-11_174448_E_sheet05_run1", "05"), (8, "2026-11-11_174936_E_sheet06_run1", "06"), (9, "2026-11-11_175203_E_sheet03_run2_no_hoodie", "03"),
              (10, "2026-11-11_175544_B_team5_sheet03", "03")]


def red_blob(picture_bgr):
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    for saturation, value in ((110, 70), (80, 50)):
        mask = (((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 165)) & (hsv[:, :, 1] >= saturation) & (hsv[:, :, 2] >= value)).astype(numpy.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, numpy.ones((3, 3), numpy.uint8))
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity = 8)
        if count > 1:
            biggest = 1 + int(numpy.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[biggest, cv2.CC_STAT_AREA] >= 300:
                return float(centroids[biggest, 0]), float(centroids[biggest, 1]), int(stats[biggest, cv2.CC_STAT_AREA])
    return None


def start_view(picture_path, joints):
    """Return the puck's table place (cm) and the direction in which the line leaves it (degrees, count, spread), or None."""
    picture = cv2.imread(str(picture_path))
    if picture is None:
        return None
    blob = red_blob(picture)
    if blob is None:
        return None
    guess = pixels_to_plane(numpy.array([[blob[0], blob[1]]]), joints, 0.0145)[0]
    if not numpy.all(numpy.isfinite(guess)):
        return None
    puck = 100.0 * pixels_to_plane(numpy.array([[blob[0], blob[1]]]), joints, table_height(float(guess[0]), float(guess[1])) + 0.0145)[0]
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    grey = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    is_line = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 125) & (hsv[:, :, 1] >= 110) & (grey < 170)
    is_line[330:, :] = False
    near = numpy.zeros(grey.shape, numpy.uint8)
    cv2.circle(near, (int(round(blob[0])), int(round(blob[1]))), int(numpy.sqrt(blob[2] / numpy.pi)) + 12, 1, -1)
    count, labels = cv2.connectedComponents(is_line.astype(numpy.uint8), connectivity = 8)
    touching = numpy.unique(labels[(near > 0) & (labels > 0)])
    is_line = numpy.isin(labels, touching) & (labels > 0)
    rows, columns = numpy.nonzero(is_line)
    result = {"puck_x": float(puck[0]), "puck_y": float(puck[1]), "line_deg": numpy.nan, "line_pixels": 0, "spread_deg": numpy.nan}
    if len(rows) < 30:
        return result
    points = 100.0 * pixels_to_plane(numpy.column_stack([columns, rows]).astype(float), joints, table_height(puck[0] / 100.0, puck[1] / 100.0))
    offset = points - puck[None, :]
    distance = numpy.hypot(offset[:, 0], offset[:, 1])
    ring = (distance >= 2.5) & (distance <= 6.0)
    if ring.sum() < 30:
        return result
    angles = numpy.arctan2(offset[ring, 1], offset[ring, 0])
    strength = numpy.hypot(numpy.sin(angles).mean(), numpy.cos(angles).mean())
    result.update({"line_deg": float(numpy.degrees(numpy.arctan2(numpy.sin(angles).mean(), numpy.cos(angles).mean()))), "line_pixels": int(ring.sum()),
                   "spread_deg": float(numpy.degrees(numpy.sqrt(-2.0 * numpy.log(max(strength, 1e-9)))))})
    return result


frames = load_training_frames({"is_audit_cleaning": True}).sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
frames["start_kind"] = frames["start_kind"].fillna("STD")
first_rows = frames.groupby("episode_id", sort = False).head(1)
first_rows = first_rows[~first_rows["start_kind"].isin(["NE", "MID", "X1", "X2"])]
rows = []
for _, row in first_rows.iterrows():
    view = start_view(RAW_FOLDER / row["frame_path"], row[JOINT_NAMES].to_numpy(dtype = float))
    if view is None:
        continue
    rows.append({"sheet": row["sheet"][-2:], "plan": row["plan_number"], "start_kind": row["start_kind"], "end": row["start_end"], **view})
demonstrations = pandas.DataFrame(rows)
demonstrations.to_csv(VERIFY_FOLDER / "v8_demonstration_start_views.csv", index = False)
clean = demonstrations[(demonstrations["spread_deg"] <= 12.0) & (demonstrations["line_pixels"] >= 300)]
print(f"start-mark demonstrations with a puck found: {len(demonstrations)} of {len(first_rows)}; with a clean line reading (spread 12 degrees or less): {len(clean)}")
print("\nDirection in which the line leaves the puck in the first picture (0 = straight away from the robot, + = to its left), clean readings:")
print(clean.groupby(["sheet", "start_kind"])["line_deg"].agg(["count", "min", "median", "max"]).round(0).to_string())
absolute = clean["line_deg"].abs()
print(f"\nall clean readings: within 20 degrees of straight away {int((absolute <= 20).sum())}, 20 to 40: {int(((absolute > 20) & (absolute <= 40)).sum())}, over 40: {int((absolute > 40).sum())}"
      f" (largest {absolute.max():.0f}); over 40 by kind: {clean[absolute > 40]['start_kind'].value_counts().to_dict()}")

print("\nThe ten robot runs, first picture:")
for number, folder, sheet in ROBOT_RUNS:
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    joints = steps.loc[0, JOINT_NAMES].to_numpy(dtype = float)
    view = start_view(RUNS_FOLDER / folder / "frames" / "000.jpg", joints)
    same_sheet = clean[clean["sheet"] == sheet]
    if view is None:
        print(f"  run {number} (sheet {sheet}): no puck found by the red-blob finder")
        continue
    if numpy.isfinite(view["line_deg"]):
        nearest = same_sheet.iloc[(same_sheet["line_deg"] - view["line_deg"]).abs().argsort()[:3]]
        nearest_text = ", ".join(f"{plan} {kind} ({line:+.0f})" for plan, kind, line in zip(nearest["plan"], nearest["start_kind"], nearest["line_deg"]))
        inside = same_sheet["line_deg"].min() - 5 <= view["line_deg"] <= same_sheet["line_deg"].max() + 5
    else:
        nearest_text, inside = "no line reading", None
    print(f"  run {number} (sheet {sheet}): puck ({view['puck_x']:.1f}, {view['puck_y']:.1f}), line leaves at {view['line_deg']:+.0f} degrees (spread {view['spread_deg']:.0f}, {view['line_pixels']} pixels);"
          f" demonstrations of this sheet: {same_sheet['line_deg'].min():+.0f} to {same_sheet['line_deg'].max():+.0f} ({len(same_sheet)} clean); inside that range: {inside}; nearest: {nearest_text}")
