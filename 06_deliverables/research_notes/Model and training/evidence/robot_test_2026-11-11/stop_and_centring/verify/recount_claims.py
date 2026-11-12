"""
Verifier's check 9 (11.11.26): smaller claims of the analyst recounted from the analyst's tables with my own code, and
three things recomputed from the camera model and the pictures:

    A   the visible share of the line's last centimetre at the demonstrated stops and the robot's stops (my own
        geometry on a grid), and how each robot stop ranks among the demonstrations;
    B   the last three pictures before the demonstrated stop: how far the puck moves sideways, and how many episodes
        that are more than 7 mm aside three pictures before the stop are within 7 mm at the stop;
    C   the demonstrated recoveries (starts with the puck put aside) by picture number, and whether an offset of 7 mm
        or more has shrunk five steps later (demonstrations and robot);
    D   how steeply the camera looks down on the puck at the stops of runs 1, 2, 4 and 7, and how much table the puck
        hides beyond itself;
    E   the puck's size: my top-face fit with a radius of 1.2 cm against 2.0 cm;
    F   whether the stop picture shows line ink close to the puck (anywhere round it, not only ahead) for covered and
        uncovered demonstration ends;
    G   how much the measure's own scatter lowers the share of covered ends (a simulation).
"""

import glob
import json
import math
from pathlib import Path

import cv2
import numpy
import pandas

import end_tools
import my_measure

cv2.setNumThreads(2)
HERE = Path(__file__).resolve().parent
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")
RAW_FOLDER = Path("D:/AI_Project/03_data/raw/main_data")
RUNS_FOLDER = Path("D:/AI_Project/03_data/model_runs")
RUNS = {1: "2026-11-11_172021_check_no_puck", 2: "2026-11-11_172421_E_sheet01_run1",
        4: "2026-11-11_173036_E_sheet02_run2", 7: "2026-11-11_174448_E_sheet05_run1"}
ROBOT_CAMERA = my_measure.shifted_camera((9.9, -2.7), 1.26)
pandas.set_option("display.width", 250)


def visible_share(short_mm: float, aside_mm: float) -> float:
    """The share of the line's last centimetre (round end plus 5 mm of line before it) not under a 12 mm puck."""

    step = 0.2
    along, across = numpy.meshgrid(numpy.arange(-5.0, 5.0 + step / 2, step), numpy.arange(-5.0, 5.0 + step / 2, step))
    is_line = (along <= 0.0) | (numpy.hypot(along, across) <= 5.0)
    is_hidden = numpy.hypot(along + short_mm, across - aside_mm) <= 12.0
    return float((is_line & ~is_hidden).sum()) / float(is_line.sum())


ends = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_ends_01_03.csv", "demonstration_ends_04_06.csv")], ignore_index = True)
ends["is_reliable"] = (ends["is_end_found"] & (ends["is_end_seen"] == True) & ends["line_width_mm"].between(8.0, 12.5) & (ends["fit_quality"] >= 0.8))
reliable = ends[ends["is_reliable"]].copy()

print("A  the visible share of the line's last centimetre (my own grid, from the analyst's two distances)")
reliable["visible"] = [visible_share(row["short_mm"], row["aside_mm"]) for _, row in reliable.iterrows()]
print(f"   demonstrations ({len(reliable)}): nothing visible in {int((reliable['visible'] == 0).sum())}, some in {int((reliable['visible'] > 0).sum())},"
      f" up to a tenth in {int(((reliable['visible'] > 0) & (reliable['visible'] <= 0.1)).sum())}, a tenth to a third in {int(((reliable['visible'] > 0.1) & (reliable['visible'] <= 1 / 3)).sum())},"
      f" more than a third in {int((reliable['visible'] > 1 / 3).sum())}")
robot_ends = pandas.read_csv(HERE.parent / "robot_runs_ends.csv")
for _, row in robot_ends[robot_ends["run"].isin(RUNS)].iterrows():
    further = float((reliable["centre_gap_mm"] > row["end_centre_gap_mm"]).mean())
    print(f"   robot run {int(row['run'])}: visible share {100 * visible_share(row['end_short_mm'], row['end_aside_mm']):.0f} %; {100 * further:.0f} % of the demonstrations end further from the end's centre,"
          f" so the run ends further away than {100 * (1 - further):.0f} %")
print(f"   by the 12 mm rule of config.py (centre within one puck radius): {int((reliable['centre_gap_mm'] <= 12).sum())} of {len(reliable)} demonstrations pass;"
      f" robot runs passing: {[int(row['run']) for _, row in robot_ends[robot_ends['run'].isin(RUNS)].iterrows() if row['end_centre_gap_mm'] <= 12]}")

print()
print("B  the last pictures before the demonstrated stop (the analyst's ../demonstration_last_steps_*.csv, reliable ends only)")
last_steps = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_last_steps_01_03.csv", "demonstration_last_steps_04_06.csv")], ignore_index = True)
last_steps = last_steps[last_steps["episode_id"].isin(reliable["episode_id"])]
wide = last_steps.pivot_table(index = "episode_id", columns = "steps_before_stop", values = ["aside_mm", "short_mm"])
has_both = wide[("aside_mm", 3)].notna() & wide[("aside_mm", 0)].notna()
sideways_move = (wide[("aside_mm", 0)] - wide[("aside_mm", 3)])[has_both].abs()
was_aside = has_both & (wide[("aside_mm", 3)].abs() > 7.0)
print(f"   episodes with a puck place 3 pictures before the stop and at the stop: {int(has_both.sum())}; sideways move over those pictures: median {sideways_move.median():.1f} mm")
print(f"   more than 7 mm aside 3 pictures before the stop: {int(was_aside.sum())}; of these within 7 mm at the stop: {int((was_aside & (wide[('aside_mm', 0)].abs() <= 7.0)).sum())}")
was_close = has_both & (wide[("aside_mm", 3)].abs() <= 7.0)
print(f"   within 7 mm aside 3 pictures before the stop: {int(was_close.sum())}; of these more than 7 mm aside at the stop: {int((was_close & (wide[('aside_mm', 0)].abs() > 7.0)).sum())}")

print()
print("C  recoveries and whether an offset shrinks (the analyst's centring tables)")
centring = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(HERE.parent / "demonstration_centring_0*.csv")))], ignore_index = True)
aside_starts = centring[centring["start_kind"].isin(["OL1", "OL1.5", "OR1", "OR1.5"]) & centring["offset_mm"].notna()]
for first, last in ((1, 3), (4, 6), (7, 10), (11, 15), (16, 25)):
    part = aside_starts[aside_starts["step"].between(first, last)]["offset_mm"].abs()
    print(f"   puck put aside, pictures {first:2d} to {last:2d}: median {part.median():.1f} mm ({len(part)} pictures of {aside_starts['episode_id'].nunique()} episodes)")


def shrink(table: pandas.DataFrame, group_column: str) -> str:
    changes = []
    for _, run in table.groupby(group_column):
        sizes = run.sort_values("step")["offset_mm"].abs().to_numpy()
        for position in range(len(sizes) - 5):
            if numpy.isfinite(sizes[position]) and sizes[position] >= 7.0 and numpy.isfinite(sizes[position + 5]):
                changes.append(sizes[position + 5] - sizes[position])
    changes = numpy.array(changes)
    return f"{len(changes)} steps, change median {numpy.median(changes):+.1f} mm, smaller by 2 mm or more in {100 * (changes <= -2).mean():.0f} %, larger by 2 mm or more in {100 * (changes >= 2).mean():.0f} %"


robot_steps = pandas.read_csv(HERE.parent / "robot_runs_steps.csv")
print("   offset of 7 mm or more, 5 steps later, demonstrations:", shrink(centring, "episode_id"))
print("   offset of 7 mm or more, 5 steps later, robot runs 1, 2, 4, 7:", shrink(robot_steps[robot_steps["run"].isin(RUNS)], "run"))
frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False)
kinds = frames.groupby("episode_id")["start_kind"].first().fillna("none")
print("   start kinds of the 210 episodes:", kinds.value_counts().to_dict())
print(f"   puck put aside (OL1, OL1.5, OR1, OR1.5): {int(kinds.isin(['OL1', 'OL1.5', 'OR1', 'OR1.5']).sum())}; sampled by the analyst: {aside_starts['episode_id'].nunique()} of {centring['episode_id'].nunique()}")

print()
print("D  how steeply the camera looks down on the puck at the robot's stops (the camera model with the analyst's correction)")
for number, folder in RUNS.items():
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    row = steps.iloc[-1]
    joints = [row[f"joint_{index}"] for index in range(1, 7)]
    puck = my_measure.puck_foot(cv2.imread(str(RUNS_FOLDER / folder / row["frame"])), joints, ROBOT_CAMERA)
    position, rotation = my_measure.forearm_camera_model.camera_pose_in_base(joints, {**my_measure.forearm_camera_model.merged_camera_parameters(), **ROBOT_CAMERA})
    top = numpy.array([puck["xy"][0], puck["xy"][1], my_measure.table_height(*puck["xy"]) + my_measure.PUCK_HEIGHT_M])
    ray = top - position
    elevation = math.degrees(math.atan2(-ray[2], math.hypot(ray[0], ray[1])))
    print(f"   run {number}: the ray to the puck's top comes down at {elevation:.0f} degrees; the puck hides the table up to {1000 * my_measure.PUCK_HEIGHT_M / math.tan(math.radians(elevation)):.0f} mm beyond its far edge,"
          f" that is {12 + 1000 * my_measure.PUCK_HEIGHT_M / math.tan(math.radians(elevation)):.0f} mm beyond its centre along the camera's view")

print()
print("E  the puck's size: my top-face fit with two radii (pictures 4, 10, 16 and the stop of runs 2 and 4; root mean square misfit in px)")
for number in (2, 4):
    steps = pandas.read_csv(RUNS_FOLDER / RUNS[number] / "steps.csv")
    for index in (4, 10, 16, len(steps) - 1):
        row = steps.iloc[index]
        joints = [row[f"joint_{joint}"] for joint in range(1, 7)]
        picture = cv2.imread(str(RUNS_FOLDER / RUNS[number] / row["frame"]))
        residual = {}
        for radius in (0.012, 0.016, 0.020):
            my_measure.PUCK_RADIUS_M = radius
            found = my_measure.puck_foot(picture, joints, ROBOT_CAMERA)
            residual[radius] = float("nan") if found is None else found["residual_px"]
        my_measure.PUCK_RADIUS_M = 0.012
        print(f"   run {number} picture {index:2d}: radius 1.2 cm {residual[0.012]:.1f} px, 1.6 cm {residual[0.016]:.1f} px, 2.0 cm {residual[0.020]:.1f} px")

print()
print("F  does the stop picture show line ink close to the puck (my line mask, table points within 30 mm of my puck foot, anywhere round it)?")
shifts = json.loads((HERE.parent / "camera_shift.json").read_text(encoding = "utf-8"))
counts = []
for _, entry in reliable.iterrows():
    row = frames[frames["episode_id"] == entry["episode_id"]].sort_values("row_in_episode").iloc[-1]
    camera = my_measure.shifted_camera(shifts[entry["camera_period"]]["shift_px"], shifts[entry["camera_period"]]["roll_deg"])
    joints = [row[f"joint_{joint}"] for joint in range(1, 7)]
    seen = end_tools.line_points_on_table(cv2.imread(str(RAW_FOLDER / row["frame_path"])), joints, camera)
    near = 0
    if seen["puck"] is not None and len(seen["points"]):
        near = int((numpy.hypot(seen["points"][:, 0] - seen["puck"]["xy"][0], seen["points"][:, 1] - seen["puck"]["xy"][1]) <= 0.030).sum())
    counts.append(near)
reliable["ink_pixels_near"] = counts
for label, part in (("covered (centres at most 7 mm apart)", reliable[reliable["centre_gap_mm"] <= 7]),
                    ("7 to 12 mm apart", reliable[(reliable["centre_gap_mm"] > 7) & (reliable["centre_gap_mm"] <= 12)]),
                    ("over 12 mm apart", reliable[reliable["centre_gap_mm"] > 12])):
    print(f"   {label}: {len(part)} demonstrations; the stop picture shows 40 or more line pixels within 30 mm of the puck in {int((part['ink_pixels_near'] >= 40).sum())}")
for number, folder in RUNS.items():
    steps = pandas.read_csv(RUNS_FOLDER / folder / "steps.csv")
    row = steps.iloc[-1]
    joints = [row[f"joint_{index}"] for index in range(1, 7)]
    seen = end_tools.line_points_on_table(cv2.imread(str(RUNS_FOLDER / folder / row["frame"])), joints, ROBOT_CAMERA)
    near = int((numpy.hypot(seen["points"][:, 0] - seen["puck"]["xy"][0], seen["points"][:, 1] - seen["puck"]["xy"][1]) <= 0.030).sum()) if len(seen["points"]) else 0
    print(f"   robot run {number}, stop picture: {near} line pixels within 30 mm of the puck")

print()
print("G  how much a scatter of the measure lowers the share of covered ends (simulation, 200,000 draws)")
generator = numpy.random.default_rng(20261111)
for noise in (0.0, 2.0, 3.0):
    true_short_sd = math.sqrt(max(4.2 ** 2 - noise ** 2, 0.0))
    true_aside_sd = math.sqrt(max(6.0 ** 2 - noise ** 2, 0.0))
    true_short = generator.normal(2.0, true_short_sd, 200_000)
    true_aside = generator.normal(0.2, true_aside_sd, 200_000)
    print(f"   if the measure scatters by {noise:.0f} mm in each direction, the true spreads are {true_short_sd:.1f} mm along and {true_aside_sd:.1f} mm across,"
          f" and the true share of covered ends is {100 * (numpy.hypot(true_short, true_aside) <= 7).mean():.0f} % (normal spreads assumed)")
