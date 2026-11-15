"""
Verifier's own replay of the proposed length rule on the logged answers of the ten robot runs, ON PAPER ONLY.

Different from the analyst's replay in one respect that matters for the geometry: every step goes through the
driver's own function model_mode.model_step (which "carries out one decision on paper: nothing moves") with
"step_length_rule" = "predicted", so the joints, and with them the picture's heading on the table, are those the arm
would really have after the shorter steps. The analyst turned the logged answers onto the table with the headings
logged in the real run (1-cm steps). Still open loop: the pictures, and so the answers, would differ.

Nothing here connects to the robot: no RobotLink, no NiryoRobot, no run of model_mode.py.

Run from this folder:  python v09_rule_replay.py > v09_rule_replay.out.txt
"""

import math
import os
import sys

import numpy
import pandas

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")
from model_mode import model_step, picture_axes
from side_view_test import magnet_tip_and_lean

RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = [
    "2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1",
    "2026-11-11_173036_E_sheet02_run2", "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1",
    "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie",
    "2026-11-11_175544_B_team5_sheet03"
]
JOINTS = [f"joint_{number}" for number in range(1, 7)]
RULE = {"step_length_rule": "predicted", "step_length_gain": 1.0, "step_most_m": 0.012, "twist_gain": 1.0}
GUARD = {"window": 6, "net_move_cm": 1.0, "net_turn_deg": 10.0, "stop_chance": 0.25, "nudge_cm": 0.5, "nudges_most": 3, "smallest_direction_cm": 0.05}
pandas.set_option("display.width", 250)


def replay(steps, is_guarded, decisions = None):
    """Carry the logged answers out with the rule; returns one row per decision and how the replay ended."""
    joints = [float(value) for value in steps.loc[0, JOINTS]]
    rows, history = [], []
    stop_count, stalled_windows = 0, 0
    last_direction = None
    ending = "log ends"
    count = len(steps) if decisions is None else min(decisions, len(steps))
    for index in range(count):
        answer = steps.iloc[index]
        tip = magnet_tip_and_lean(list(joints))[0]
        up_xy, left_xy = picture_axes(joints)
        decision = {"tip_forward_cm": float(answer["model_tip_forward_cm"]), "tip_left_cm": float(answer["model_tip_left_cm"]), "twist_deg": float(answer["model_twist_deg"])}
        asked = math.hypot(decision["tip_forward_cm"], decision["tip_left_cm"])
        vector_table = decision["tip_forward_cm"] * up_xy + decision["tip_left_cm"] * left_xy
        stop_count = stop_count + 1 if answer["model_stop_probability"] >= 0.5 else 0
        row = {"step": index, "asked_cm": asked, "twist_deg": decision["twist_deg"], "tip_x_cm": 100 * tip[0], "tip_y_cm": 100 * tip[1], "joint_4_deg": math.degrees(joints[3]),
               "heading_deg": math.degrees(math.atan2(up_xy[1], up_xy[0])), "logged_heading_deg": answer["picture_heading_deg"], "stop_chance": answer["model_stop_probability"], "kind": "step"}
        if stop_count >= 2:
            row["kind"] = "stop"
            rows.append(row)
            ending = "stop"
            break
        history.append({"tip": numpy.array(tip[:2]) * 100.0, "joint_4": math.degrees(joints[3]), "vector": vector_table, "stop_chance": answer["model_stop_probability"]})
        if is_guarded and len(history) > GUARD["window"]:
            window = history[-GUARD["window"] - 1:]
            net_move = float(numpy.linalg.norm(window[-1]["tip"] - window[0]["tip"]))
            net_turn = abs(window[-1]["joint_4"] - window[0]["joint_4"])
            if net_move < GUARD["net_move_cm"] and net_turn < GUARD["net_turn_deg"]:
                stalled_windows += 1
                # the stop chances of the six decisions of the window (the pictures the six moves were decided on)
                if max(entry["stop_chance"] for entry in window[1:]) >= GUARD["stop_chance"] or window[0]["stop_chance"] >= GUARD["stop_chance"]:
                    row["kind"] = "stalled near the end"
                    rows.append(row)
                    ending = "stalled near the end"
                    break
                if stalled_windows > GUARD["nudges_most"]:
                    row["kind"] = "stalled"
                    rows.append(row)
                    ending = "stalled"
                    break
                summed = numpy.sum([entry["vector"] for entry in window[1:]], axis = 0)
                direction = summed / numpy.linalg.norm(summed) if numpy.linalg.norm(summed) >= GUARD["smallest_direction_cm"] else up_xy
                decision = {"tip_forward_cm": GUARD["nudge_cm"] * float(direction @ up_xy), "tip_left_cm": GUARD["nudge_cm"] * float(direction @ left_xy), "twist_deg": decision["twist_deg"]}
                row["kind"] = "nudge"
                history = history[-1:]
            else:
                stalled_windows = 0
        planned = model_step(joints, decision, last_direction, RULE)
        row["rule_cm"] = planned["step_cm"]
        if planned["refusal"]:
            row["kind"] = "refused: " + planned["refusal"]
            rows.append(row)
            ending = "refused"
            break
        rows.append(row)
        joints = [float(value) for value in planned["joints_next"]]
        last_direction = planned["direction_xy"]
    final_tip = magnet_tip_and_lean(list(joints))[0]
    return pandas.DataFrame(rows), ending, numpy.array(final_tip[:2]) * 100.0, math.degrees(joints[3])


print("THE FIRST 15 DECISIONS OF RUNS 5, 9 AND 10, replayed through model_mode.model_step with the predicted length (gain 1.0, at most 1.2 cm), no stall guard")
for number in (5, 9, 10):
    steps = pandas.read_csv(os.path.join(RUNS, RUN_FOLDERS[number - 1], "steps.csv"))
    table, ending, final_tip, final_joint_4 = replay(steps, is_guarded = False, decisions = 15)
    start_tip = table[["tip_x_cm", "tip_y_cm"]].iloc[0].to_numpy()
    travel = table["rule_cm"].sum()
    turn = final_joint_4 - table["joint_4_deg"].iloc[0]
    net = float(numpy.linalg.norm(final_tip - start_tip))
    done_tip = 100.0 * steps[["tip_x", "tip_y"]].to_numpy()
    done_travel = float(numpy.linalg.norm(numpy.diff(done_tip[:16], axis = 0), axis = 1).sum())
    done_net = float(numpy.linalg.norm(done_tip[15] - done_tip[0]))
    done_turn = math.degrees(steps["joint_4"].iloc[15] - steps["joint_4"].iloc[0])
    heading_gap = (table["heading_deg"] - table["logged_heading_deg"]).abs()
    print(f"--- run {number}: ended '{ending}'")
    print(table[["step", "asked_cm", "rule_cm", "twist_deg", "tip_x_cm", "tip_y_cm", "joint_4_deg", "heading_deg", "logged_heading_deg", "kind"]].round(2).to_string(index = False))
    print(f"   rule: steps {table['rule_cm'].min():.2f} to {table['rule_cm'].max():.2f} cm; travel {travel:.1f} cm; net {net:.1f} cm; forearm turn {turn:.0f} degrees; {abs(turn) / travel:.1f} degrees per cm")
    print(f"   done on 11.11.26: travel {done_travel:.1f} cm; net {done_net:.1f} cm; forearm turn {done_turn:.0f} degrees; {abs(done_turn) / done_travel:.1f} degrees per cm")
    print(f"   the replayed heading differs from the logged heading by up to {heading_gap.max():.1f} degrees (median {heading_gap.median():.1f}): the size of the error of a replay with the logged headings")
    print()

print("THE RULE WITH THE STALL GUARD ON THE TEN LOGGED RUNS (open loop)")
summary = []
for number, folder in enumerate(RUN_FOLDERS, start = 1):
    steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
    table, ending, final_tip, final_joint_4 = replay(steps, is_guarded = True)
    nudges = table[table["kind"] == "nudge"]["step"].tolist()
    unguarded, unguarded_ending, _, _ = replay(steps, is_guarded = False)
    tiny = int((unguarded["asked_cm"] < 0.05).sum())
    summary.append({"run": number, "pictures_logged": len(steps), "decisions_replayed": len(table), "ends_as": ending, "nudges_at": nudges, "asked_under_0.05_cm": tiny,
                    "rule_travel_cm": table["rule_cm"].sum(), "without_guard_ends_as": unguarded_ending, "without_guard_decisions": len(unguarded)})
    if number == 6:
        print("run 6, decisions 15 to the end of the guarded replay:")
        print(table.iloc[15:][["step", "asked_cm", "rule_cm", "twist_deg", "tip_x_cm", "tip_y_cm", "joint_4_deg", "stop_chance", "kind"]].round(2).to_string(index = False))
        part = steps.iloc[19:51]
        asked = numpy.hypot(part["model_tip_forward_cm"], part["model_tip_left_cm"])
        tips = 100.0 * steps[["tip_x", "tip_y"]].to_numpy()
        angle = part["model_angle_deg"].to_numpy() + part["picture_heading_deg"].to_numpy()
        flips = (numpy.abs((numpy.diff(angle) + 180.0) % 360.0 - 180.0) > 90.0).sum()
        print(f"run 6, logged decisions 19 to 50: asked median {asked.median():.2f} cm, largest {asked.max():.2f}; twist asked median size {part['model_twist_deg'].abs().median():.2f} degrees; "
              f"the driver moved {numpy.linalg.norm(numpy.diff(tips[19:52], axis = 0), axis = 1).sum():.1f} cm for a net {numpy.linalg.norm(tips[51] - tips[19]):.1f} cm; "
              f"the asked table direction flips by more than 90 degrees on {int(flips)} of {len(angle) - 1} changes")
    if number == 10:
        print("run 10, decisions 50 to the end of the guarded replay:")
        print(table.iloc[50:][["step", "asked_cm", "rule_cm", "twist_deg", "tip_x_cm", "tip_y_cm", "joint_4_deg", "stop_chance", "kind"]].round(2).to_string(index = False))
print(pandas.DataFrame(summary).round(1).to_string(index = False))
