"""
Verifier's own first-bend numbers for sheets 03 and 04: demonstrations (frames.csv as recorded) against the robot runs
(steps.csv). No registration, no camera model: only joint 4 and the tip positions of each episode itself.

Measures, per episode:
  travel_60   tip travel (cm) from the first row until joint 4 has turned 60 degrees from its first value
  steps_60    rows needed for that
  twists_in_place_60   steps in that stretch with a tip move under 0.3 cm and a joint-4 change of 3 degrees or more
  deg_per_cm_15        |joint-4 change| over tip travel across the first 15 steps
Run from this folder:  python v04_first_bend.py > v04_first_bend.out.txt
"""

import os

import numpy
import pandas

INDEX = "D:/AI_Project/03_data/processed/v01_training_index/frames.csv"
RUNS = "D:/AI_Project/03_data/model_runs"
RUN_FOLDERS = {5: "2026-11-11_173549_E_sheet03_run1", 9: "2026-11-11_175203_E_sheet03_run2_no_hoodie", 10: "2026-11-11_175544_B_team5_sheet03",
               6: "2026-11-11_174108_E_sheet04_run1"}
PLANS = {"path_03": ["03-01", "03-02", "03-03", "03-04", "03-05", "03-25", "03-26", "03-27"],
         "path_04": ["04-01", "04-02", "04-03", "04-04", "04-05", "04-13", "04-14", "04-15", "04-16", "04-17", "04-18"]}
pandas.set_option("display.width", 250)


def bend_measures(tip_cm, joint_4_deg, label_step_cm = None, label_twist_deg = None, turn_deg = 60.0):
    moved = numpy.linalg.norm(numpy.diff(tip_cm, axis = 0), axis = 1)
    twist = numpy.diff(joint_4_deg)
    travelled = numpy.r_[0.0, numpy.cumsum(moved)]
    turned = numpy.abs(joint_4_deg - joint_4_deg[0])
    reached = numpy.flatnonzero(turned >= turn_deg)
    result = {"rows": len(tip_cm), "joint_4_first": joint_4_deg[0]}
    if len(reached):
        last = int(reached[0])
        # interpolate the travel at exactly 60 degrees between the row before and the row that passed it
        before, after = turned[last - 1], turned[last]
        share = (turn_deg - before) / (after - before) if after > before else 1.0
        result.update({"travel_60": travelled[last], "travel_60_interpolated": travelled[last - 1] + share * (travelled[last] - travelled[last - 1]),
                       "steps_60": last,
                       "twists_in_place_60": int(((moved[:last] < 0.3) & (numpy.abs(twist[:last]) >= 3.0)).sum()),
                       "short_steps_60_under_0.5": int((moved[:last] < 0.5).sum()),
                       "mean_step_60": travelled[last] / last,
                       "deg_per_cm_60": turned[last] / travelled[last]})
        if label_step_cm is not None:
            result["twists_in_place_60_from_labels"] = int(((label_step_cm[:last] < 0.3) & (numpy.abs(label_twist_deg[:last]) >= 3.0)).sum())
    else:
        result.update({"travel_60": numpy.nan, "steps_60": numpy.nan, "largest_turn": turned.max()})
    count = min(15, len(moved))
    result["deg_per_cm_15"] = abs(joint_4_deg[count] - joint_4_deg[0]) / travelled[count]
    result["turn_15"] = joint_4_deg[count] - joint_4_deg[0]
    result["travel_15"] = travelled[count]
    return result


frames = pandas.read_csv(INDEX, low_memory = False, usecols = ["episode_id", "row_in_episode", "sheet", "plan_number", "start_kind", "start_end", "action",
                                                             "tip_forward_m", "tip_left_m", "forearm_change_deg", "joint_4", "tip_x_m", "tip_y_m"])
frames = frames.sort_values(["episode_id", "row_in_episode"])

for sheet, plans in PLANS.items():
    print(f"=== {sheet}: demonstrations (frames.csv as recorded)")
    rows = []
    for plan in plans:
        group = frames[frames["plan_number"] == plan]
        if group["episode_id"].nunique() != 1:
            print(f"   plan {plan}: {group['episode_id'].nunique()} episodes!")
        tip = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        joint_4 = numpy.degrees(group["joint_4"].to_numpy())
        label_step = 100.0 * numpy.hypot(group["tip_forward_m"].fillna(0.0), group["tip_left_m"].fillna(0.0)).to_numpy()
        label_twist = group["forearm_change_deg"].fillna(0.0).to_numpy()
        measures = bend_measures(tip, joint_4, label_step, label_twist)
        measures["plan"] = plan
        measures["start_kind"] = group["start_kind"].iloc[0]
        measures["start_end"] = group["start_end"].iloc[0]
        # how well the labels agree with the tip positions (label of row i against the move from row i to row i + 1)
        moved = numpy.linalg.norm(numpy.diff(tip, axis = 0), axis = 1)
        measures["label_minus_position_step_median_abs_cm"] = float(numpy.median(numpy.abs(label_step[:-1] - moved)))
        rows.append(measures)
    table = pandas.DataFrame(rows).set_index("plan")
    print(table.round(2).to_string())
    numeric = table.select_dtypes("number")
    print("median:", numeric.median().round(2).to_dict())
    print("min:   ", numeric.min().round(2).to_dict())
    print("max:   ", numeric.max().round(2).to_dict())
    print()
    print(f"=== {sheet}: robot runs (steps.csv), done and 'as asked' (same logged answers, the asked length instead of 1 cm, twist as asked)")
    rows = []
    for number, folder in RUN_FOLDERS.items():
        steps = pandas.read_csv(os.path.join(RUNS, folder, "steps.csv"))
        if (sheet == "path_04") != (number == 6):
            continue
        tip = 100.0 * steps[["tip_x", "tip_y"]].to_numpy()
        joint_4 = numpy.degrees(steps["joint_4"].to_numpy())
        measures = bend_measures(tip, joint_4)
        measures["run"] = f"{number} done"
        rows.append(measures)
        # as asked, open loop: step i has the asked length and the asked twist; the path is laid out as a chain of segment
        # lengths only (travel does not need the directions)
        asked = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"]).to_numpy()
        asked = numpy.minimum(asked, 1.2)
        twist = numpy.clip(steps["model_twist_deg"].to_numpy(), -10.0, 10.0)
        fake_tip = numpy.column_stack([numpy.r_[0.0, numpy.cumsum(asked[:-1])], numpy.zeros(len(asked))])
        fake_joint_4 = joint_4[0] + numpy.r_[0.0, numpy.cumsum(twist[:-1])]
        measures = bend_measures(fake_tip, fake_joint_4)
        measures["run"] = f"{number} as asked"
        rows.append(measures)
    print(pandas.DataFrame(rows).set_index("run").round(2).to_string())
    print()
