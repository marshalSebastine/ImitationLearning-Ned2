"""
Verifier: were the model's DIRECTIONS at the first bend of sheet 03 right, so that only the length was wrong?

Test: when the forearm has turned as far as it had after the robot's first 15 decisions, where is the tip relative to
its start? Compared in the frame of each episode's own first picture heading (forward = the picture's up at the start,
left = its left), for
    the eight demonstrations (frames.csv),
    the run as it was done on 11.11.26 (steps.csv),
    the same logged answers carried out with the asked length through model_mode.model_step (open loop, on paper).
If the answers carried out as asked put the tip about where the demonstrations put it, the directions were right and
the length was the driver's error. Nothing connects to the robot.

Run from this folder:  python v14_net_move_at_same_turn.py > v14_net_move_at_same_turn.out.txt
"""

import contextlib
import io
import math
import os

import numpy
import pandas

with contextlib.redirect_stdout(io.StringIO()):
    import v09_rule_replay as replay_module

INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
PLANS = ["03-01", "03-02", "03-03", "03-04", "03-05", "03-25", "03-26", "03-27"]
pandas.set_option("display.width", 250)


def in_start_frame(vector_cm, heading_deg):
    angle = math.radians(heading_deg)
    up = numpy.array([math.cos(angle), math.sin(angle)])
    left = numpy.array([-math.sin(angle), math.cos(angle)])
    return float(vector_cm @ up), float(vector_cm @ left)


frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "plan_number", "tip_x_m", "tip_y_m", "joint_4", "picture_heading_deg"])
frames = frames.sort_values(["episode_id", "row_in_episode"])

for number in (5, 9, 10):
    steps = pandas.read_csv(os.path.join(replay_module.RUNS, replay_module.RUN_FOLDERS[number - 1], "steps.csv"))
    tips = 100.0 * steps[["tip_x", "tip_y"]].to_numpy()
    joint_4 = numpy.degrees(steps["joint_4"].to_numpy())
    turn = abs(joint_4[15] - joint_4[0])
    heading = float(steps["picture_heading_deg"].iloc[0])
    done_forward, done_left = in_start_frame(tips[15] - tips[0], heading)
    table, ending, final_tip, final_joint_4 = replay_module.replay(steps, is_guarded = False, decisions = 15)
    start_tip = table[["tip_x_cm", "tip_y_cm"]].iloc[0].to_numpy()
    asked_forward, asked_left = in_start_frame(final_tip - start_tip, heading)
    rows = []
    for plan in PLANS:
        group = frames[frames["plan_number"] == plan]
        demo_tips = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        demo_joint_4 = numpy.degrees(group["joint_4"].to_numpy())
        turned = numpy.abs(demo_joint_4 - demo_joint_4[0])
        reached = int(numpy.flatnonzero(turned >= turn)[0])
        share = (turn - turned[reached - 1]) / (turned[reached] - turned[reached - 1])
        tip_at = demo_tips[reached - 1] + share * (demo_tips[reached] - demo_tips[reached - 1])
        forward, left = in_start_frame(tip_at - demo_tips[0], float(group["picture_heading_deg"].iloc[0]))
        travel = numpy.linalg.norm(numpy.diff(demo_tips[:reached + 1], axis = 0), axis = 1).sum()
        rows.append({"plan": plan, "rows": reached, "travel_cm": travel, "forward_cm": forward, "left_cm": left, "net_cm": math.hypot(forward, left)})
    demonstrations = pandas.DataFrame(rows)
    print(f"--- run {number}: after 15 decisions the forearm had turned {turn:.0f} degrees; the tip against its start, in the frame of the first picture")
    print(demonstrations.round(2).to_string(index = False))
    print(f"   demonstrations at the same forearm turn: forward median {demonstrations['forward_cm'].median():.1f} cm ({demonstrations['forward_cm'].min():.1f} to {demonstrations['forward_cm'].max():.1f}), "
          f"left median {demonstrations['left_cm'].median():+.1f} cm ({demonstrations['left_cm'].min():+.1f} to {demonstrations['left_cm'].max():+.1f}), "
          f"net median {demonstrations['net_cm'].median():.1f} cm ({demonstrations['net_cm'].min():.1f} to {demonstrations['net_cm'].max():.1f})")
    print(f"   done on 11.11.26 (1 cm per decision):   forward {done_forward:.1f} cm, left {done_left:+.1f} cm, net {math.hypot(done_forward, done_left):.1f} cm")
    print(f"   the same answers with the asked length: forward {asked_forward:.1f} cm, left {asked_left:+.1f} cm, net {math.hypot(asked_forward, asked_left):.1f} cm")
    print()
