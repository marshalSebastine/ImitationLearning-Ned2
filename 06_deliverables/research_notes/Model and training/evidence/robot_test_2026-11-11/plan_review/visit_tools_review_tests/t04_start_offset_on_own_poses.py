"""Review attack 3: the start offset on poses of my own, against the arm model used directly and against the robot's own recorded pose."""

import math

import numpy
import pandas

import safe_import
from safe_import import model_mode
from config import MODEL_MODE_PARAMETERS, PUSHER_PARAMETERS, TRAINING_DATA_PARAMETERS, pushing_height
from side_view_test import ned2_fk, solve_with_wrist_fixed

recorded_deg = MODEL_MODE_PARAMETERS["recorded_start_joints_deg"]
recorded = [math.radians(value) for value in recorded_deg]


def tip_from_fk(joints):
    """The magnet tip worked out here from ned2_fk, not through magnet_tip_and_lean."""

    frames = ned2_fk.fk(list(joints))
    rod = numpy.array(frames["joint_6"][1])[:, 2]
    return numpy.array(frames["tool"][0]) + PUSHER_PARAMETERS["extension_below_rod_end_m"] * rod


def joints_at(tip_xy, turn_deg = 0.0, wrist_deg = MODEL_MODE_PARAMETERS["wrist_held_deg"], lift = 0.0):

    target = numpy.array([tip_xy[0], tip_xy[1], pushing_height(*tip_xy) - PUSHER_PARAMETERS["extension_below_rod_end_m"] + lift])
    joints = solve_with_wrist_fixed(target, math.radians(turn_deg), math.radians(wrist_deg), [math.atan2(tip_xy[1], tip_xy[0]), -0.8, -0.6])
    return joints


recorded_tip = tip_from_fk(recorded)
print("recorded tip from ned2_fk, cm:", numpy.round(100 * recorded_tip, 3))
print("offset at the recorded start:", model_mode.offset_from_recorded_start(recorded))

print("\n--- hand-made tips: asked offset against returned offset (cm), with odd forearm turns, wrists and heights")
worst = 0.0
for dx_cm, dy_cm, turn_deg, wrist_deg, lift in [(1.0, 0.0, 0.0, 13.5, 0.0), (-1.0, 0.0, 0.0, 13.5, 0.0), (0.0, 1.0, 0.0, 13.5, 0.0), (0.0, -1.0, 0.0, 13.5, 0.0),
                                                 (0.6, 0.5, 2.1, 13.5, 0.0), (3.0, -4.0, -35.0, 13.5, 0.0), (10.0, 10.0, 80.0, 20.0, 0.0), (0.3, -0.2, 0.0, 13.5, 0.04),
                                                 (20.0, -15.0, -110.0, 5.0, 0.01), (0.04, -0.04, 0.04, 13.5, 0.0), (-0.9, 0.3, -3.0, 13.5, 0.0)]:
    tip_xy = [recorded_tip[0] + dx_cm / 100, recorded_tip[1] + dy_cm / 100]
    joints = joints_at(tip_xy, turn_deg, wrist_deg, lift)
    if joints is None:
        print((dx_cm, dy_cm, turn_deg, wrist_deg, lift), "no joints (out of reach)")
        continue
    joints[5] = math.radians(77.0)   # the hand turned any way: the tip must not care
    offset = model_mode.offset_from_recorded_start(joints)
    independent = 100 * (tip_from_fk(joints)[:2] - recorded_tip[:2])
    error = max(abs(offset["tip_along_x_cm"] - dx_cm), abs(offset["tip_along_y_cm"] - dy_cm), abs(offset["tip_distance_cm"] - math.hypot(dx_cm, dy_cm)),
                abs(offset["forearm_deg"] - (turn_deg - recorded_deg[3])), abs(offset["tip_along_x_cm"] - independent[0]), abs(offset["tip_along_y_cm"] - independent[1]))
    worst = max(worst, error)
    print((dx_cm, dy_cm, turn_deg, wrist_deg, lift), "->", {name: round(value, 3) for name, value in offset.items()}, "| lines:", model_mode.start_offset_lines(offset)[0][:95], "|", model_mode.placing_line(offset))
print("largest error against what was asked, cm or degrees:", round(worst, 4))

print("\n--- the words at the edges of the warning limits and at values that round to zero")
for offset in [{"tip_distance_cm": 1.0, "tip_along_x_cm": 1.0, "tip_along_y_cm": 0.0, "forearm_deg": 3.0},
               {"tip_distance_cm": 1.04, "tip_along_x_cm": 1.04, "tip_along_y_cm": 0.0, "forearm_deg": 0.0},
               {"tip_distance_cm": 0.96, "tip_along_x_cm": 0.96, "tip_along_y_cm": 0.0, "forearm_deg": 3.04},
               {"tip_distance_cm": 0.06, "tip_along_x_cm": -0.04, "tip_along_y_cm": -0.04, "forearm_deg": -0.04},
               {"tip_distance_cm": 0.07, "tip_along_x_cm": -0.05, "tip_along_y_cm": -0.05, "forearm_deg": -0.05},
               {"tip_distance_cm": 12.3, "tip_along_x_cm": -10.0, "tip_along_y_cm": 7.2, "forearm_deg": -117.0},
               {"tip_distance_cm": float("nan"), "tip_along_x_cm": float("nan"), "tip_along_y_cm": float("nan"), "forearm_deg": float("nan")}]:
    lines = model_mode.start_offset_lines(offset)
    print(len(lines), "line(s):", lines[0].replace("; left and right as seen from the robot's base looking out along x", ""), "||", model_mode.placing_line(offset))

print("\n--- the recorded data: the robot's own reported x, y against the offset, on the first rows of the 210 demonstrations")
frames = pandas.read_csv(TRAINING_DATA_PARAMETERS["processed_folder"] / "frames.csv", usecols = ["episode_id", "row_in_episode", "x", "y"] + [f"joint_{index}" for index in range(1, 7)])
first = frames.sort_values(["episode_id", "row_in_episode"]).groupby("episode_id").head(1)
near = first[(numpy.abs(numpy.degrees(first[[f"joint_{index}" for index in range(1, 7)]].to_numpy()) - numpy.array(recorded_deg)).max(axis = 1) <= 0.5)]
reference_xy = near[["x", "y"]].mean().to_numpy()
offsets = numpy.array([[model_mode.offset_from_recorded_start(list(row))[name] for name in ("tip_along_x_cm", "tip_along_y_cm")]
                       for row in first[[f"joint_{index}" for index in range(1, 7)]].to_numpy(dtype = float)])
reported = 100 * (first[["x", "y"]].to_numpy() - reference_xy)
far = numpy.hypot(*offsets.T) > 1.0
print("first rows further than 1 cm from the recorded start:", int(far.sum()))
print("along x: model offset against the robot's reported rod-end shift, correlation", round(float(numpy.corrcoef(offsets[far, 0], reported[far, 0])[0, 1]), 4),
      " along y:", round(float(numpy.corrcoef(offsets[far, 1], reported[far, 1])[0, 1]), 4))
print("same sign along y in", int((numpy.sign(offsets[far, 1]) == numpy.sign(reported[far, 1])).sum()), "of", int(far.sum()),
      "; along x in", int((numpy.sign(offsets[far, 0]) == numpy.sign(reported[far, 0])).sum()))
print("largest difference between the two, cm:", round(float(numpy.abs(offsets[far] - reported[far]).max()), 2))
