"""
Verifier's own numbers for three things the decision rests on (fold models on held-out sheets, no training, no robot):

A. NET turn per centimetre over windows of 10 rows (signed sums, so jitter on straight stretches cancels), for each
   driver rule, as a share of the demonstrated. The analyst's "0.89 to 0.95" sums ABSOLUTE twists over all rows.
B. The first bend of sheet 03 as an unseen sheet (fold model without sheet 03 on the demonstrations' pictures until
   joint 4 has turned 60 degrees): turn per cm with the own length and with a fixed 1 cm.
C. The stall guard's base rates: windows of 6 steps with a net tip move under 1.0 cm AND a net forearm turn under
   10 degrees, in the demonstrations and in the fold models' own steps on the same pictures.

Run from this folder:  python v08_bend_and_stall_numbers.py > v08_bend_and_stall_numbers.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
from config import RECORDING_PARAMETERS, TRAINING_TARGET_PARAMETERS
from train_policy import merged_run_settings
from training_targets import load_training_frames

STEP_LENGTH = ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length"
pandas.set_option("display.width", 250)
LEFT_SIGN = RECORDING_PARAMETERS["picture_left_sign"]

target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
cleaned = load_training_frames(target_parameters)
cleaned = cleaned.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
print(f"picture_left_sign {LEFT_SIGN}; cleaned rows {len(cleaned)}")

# check of the frame convention on the labels themselves: the labelled step turned onto the table with the picture's
# heading must give the tip's real move to the next row
heading = numpy.radians(cleaned["picture_heading_deg"].to_numpy())
up = numpy.column_stack([numpy.cos(heading), numpy.sin(heading)])
left = numpy.column_stack([numpy.cos(heading + LEFT_SIGN * numpy.pi / 2), numpy.sin(heading + LEFT_SIGN * numpy.pi / 2)])
label_table = (cleaned["tip_forward_m"].fillna(0.0).to_numpy()[:, None] * up + cleaned["tip_left_m"].fillna(0.0).to_numpy()[:, None] * left) * 100.0
same_episode_next = (cleaned["episode_id"].shift(-1) == cleaned["episode_id"]).to_numpy() & (cleaned["row_in_episode"].shift(-1) == cleaned["row_in_episode"] + 1).to_numpy()
real_move = 100.0 * (cleaned[["tip_x_m", "tip_y_m"]].shift(-1).to_numpy() - cleaned[["tip_x_m", "tip_y_m"]].to_numpy())
difference = numpy.linalg.norm(label_table - real_move, axis = 1)[same_episode_next & (cleaned["action"] != "stop").to_numpy()]
print(f"label turned onto the table against the tip's real move to the next row: median difference {numpy.nanmedian(difference):.3f} cm, 90th percentile {numpy.nanpercentile(difference, 90):.3f} cm "
      f"({len(difference)} rows with a next row)")

by_frame = cleaned.set_index("frame_index")
for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    predicted = pandas.DataFrame({"frame_index": data["frame_index"], "p_forward": data["exact"][:, 0], "p_left": data["exact"][:, 1], "p_twist": data["exact"][:, 2]})
    table = cleaned.merge(predicted, on = "frame_index", how = "left")
    assert table["p_forward"].notna().all()
    table = table[table["action"] != "stop"].copy()
    table["d_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"].fillna(0.0), table["tip_left_m"].fillna(0.0))
    table["d_twist"] = table["forearm_change_deg"].fillna(0.0)
    table["p_cm"] = numpy.hypot(table["p_forward"], table["p_left"])
    heading = numpy.radians(table["picture_heading_deg"].to_numpy())
    up = numpy.column_stack([numpy.cos(heading), numpy.sin(heading)])
    left = numpy.column_stack([numpy.cos(heading + LEFT_SIGN * numpy.pi / 2), numpy.sin(heading + LEFT_SIGN * numpy.pi / 2)])
    predicted_table = table["p_forward"].to_numpy()[:, None] * up + table["p_left"].to_numpy()[:, None] * left
    table["p_table_x"], table["p_table_y"] = predicted_table[:, 0], predicted_table[:, 1]
    demonstrated_table = 100.0 * (table["tip_forward_m"].fillna(0.0).to_numpy()[:, None] * up + table["tip_left_m"].fillna(0.0).to_numpy()[:, None] * left)
    table["d_table_x"], table["d_table_y"] = demonstrated_table[:, 0], demonstrated_table[:, 1]
    print()
    print(f"================ {recipe}: {len(table)} step rows")

    # A. net turn per cm over windows of 10 rows
    rules = {"fixed 1.0 cm": (lambda p: numpy.ones_like(p), 1.0), "own length, at most 1.2 cm": (lambda p: numpy.minimum(p, 1.2), 1.0),
             "gain 1.6, clamp 0.5 to 1.5 (report 4.4)": (lambda p: numpy.clip(1.6 * p, 0.5, 1.5), 1.0), "fixed 1.0 cm, twist gain 1.6": (lambda p: numpy.ones_like(p), 1.6),
             "length and twist both x 1.4, at most 1.2 cm": (lambda p: numpy.minimum(1.4 * p, 1.2), 1.4)}
    window = 10
    sums = {name: [0.0, 0.0] for name in rules}
    sums_sharp = {name: [0.0, 0.0] for name in rules}
    demonstrated = [0.0, 0.0]
    demonstrated_sharp = [0.0, 0.0]
    per_sheet = {}
    for episode_id, group in table.groupby("episode_id", sort = False):
        d_twist, d_cm = group["d_twist"].to_numpy(), group["d_cm"].to_numpy()
        p_twist, p_cm = group["p_twist"].to_numpy(), group["p_cm"].to_numpy()
        sheet = group["sheet"].iloc[0]
        for start in range(0, len(group) - window + 1, window):
            part = slice(start, start + window)
            net_demonstrated = abs(d_twist[part].sum())
            is_sharp = net_demonstrated >= 30.0
            demonstrated[0] += net_demonstrated
            demonstrated[1] += d_cm[part].sum()
            if is_sharp:
                demonstrated_sharp[0] += net_demonstrated
                demonstrated_sharp[1] += d_cm[part].sum()
            for name, (length_rule, twist_gain) in rules.items():
                twist_used = numpy.clip(twist_gain * p_twist[part], -10.0, 10.0)
                # the predicted net turn counted along the demonstrated net turn's sign (a turn the wrong way counts against)
                signed = twist_used.sum() * numpy.sign(d_twist[part].sum())
                length = length_rule(p_cm[part]).sum()
                sums[name][0] += signed
                sums[name][1] += length
                if is_sharp:
                    sums_sharp[name][0] += signed
                    sums_sharp[name][1] += length
                    key = (name, sheet)
                    entry = per_sheet.setdefault(key, [0.0, 0.0, 0.0, 0.0])
                    entry[0] += signed
                    entry[1] += length
                    entry[2] += net_demonstrated
                    entry[3] += d_cm[part].sum()
    print(f"A. windows of {window} rows: demonstrated net turn per cm {demonstrated[0] / demonstrated[1]:.3f} deg/cm; in sharp windows (net turn of 30 degrees or more) "
          f"{demonstrated_sharp[0] / demonstrated_sharp[1]:.3f} deg/cm")
    for name in rules:
        share = (sums[name][0] / sums[name][1]) / (demonstrated[0] / demonstrated[1])
        share_sharp = (sums_sharp[name][0] / sums_sharp[name][1]) / (demonstrated_sharp[0] / demonstrated_sharp[1])
        sheets = ", ".join(f"{sheet[-2:]} {(per_sheet[(name, sheet)][0] / per_sheet[(name, sheet)][1]) / (per_sheet[(name, sheet)][2] / per_sheet[(name, sheet)][3]):.2f}"
                           for sheet in sorted({key[1] for key in per_sheet}))
        print(f"   {name:44s}: net turn per cm as a share of the demonstrated: all windows {share:.3f}; sharp windows {share_sharp:.3f} (per held-out sheet: {sheets})")

    # B. the first bend of sheet 03 as an unseen sheet
    rows = []
    for plan in ["03-01", "03-02", "03-03", "03-04", "03-05", "03-25", "03-26", "03-27"]:
        group = table[table["plan_number"] == plan]
        joint_4 = numpy.degrees(group["joint_4"].to_numpy())
        turned = numpy.abs(joint_4 - joint_4[0])
        reached = numpy.flatnonzero(turned >= 60.0)
        last = int(reached[0]) if len(reached) else len(group)
        part = group.iloc[:last]
        rows.append({"plan": plan, "rows": last, "first_row_in_episode": int(group["row_in_episode"].iloc[0]),
                     "demonstrated_travel": part["d_cm"].sum(), "demonstrated_turn": part["d_twist"].sum(),
                     "predicted_travel": numpy.minimum(part["p_cm"], 1.2).sum(), "predicted_turn": part["p_twist"].sum(),
                     "predicted_length_median": part["p_cm"].median(), "predicted_twist_median": part["p_twist"].median()})
    bend = pandas.DataFrame(rows)
    bend["demonstrated_deg_per_cm"] = bend["demonstrated_turn"].abs() / bend["demonstrated_travel"]
    bend["own_length_deg_per_cm"] = bend["predicted_turn"].abs() / bend["predicted_travel"]
    bend["fixed_deg_per_cm"] = bend["predicted_turn"].abs() / bend["rows"]
    print("B. first bend of sheet 03 with the fold model that never saw sheet 03:")
    print(bend.round(2).to_string(index = False))
    print(f"   medians: demonstrated {bend['demonstrated_deg_per_cm'].median():.2f} deg/cm; own length {bend['own_length_deg_per_cm'].median():.2f} "
          f"({bend['own_length_deg_per_cm'].median() / bend['demonstrated_deg_per_cm'].median():.2f} of the demonstrated; pooled over the 8: "
          f"{(bend['predicted_turn'].abs().sum() / bend['predicted_travel'].sum()) / (bend['demonstrated_turn'].abs().sum() / bend['demonstrated_travel'].sum()):.2f}); fixed 1 cm "
          f"{bend['fixed_deg_per_cm'].median():.2f} ({bend['fixed_deg_per_cm'].median() / bend['demonstrated_deg_per_cm'].median():.2f}; pooled "
          f"{(bend['predicted_turn'].abs().sum() / bend['rows'].sum()) / (bend['demonstrated_turn'].abs().sum() / bend['demonstrated_travel'].sum()):.2f}); "
          f"predicted length median {bend['predicted_length_median'].median():.2f}; predicted turn as a share of the demonstrated turn (pooled) "
          f"{bend['predicted_turn'].abs().sum() / bend['demonstrated_turn'].abs().sum():.2f}")

    # C. the stall guard's base rates
    for steps in (6,):
        demonstrated_stalled, model_stalled, windows = 0, 0, 0
        model_net = []
        for episode_id, group in table.groupby("episode_id", sort = False):
            if len(group) < steps:
                continue
            d_xy = group[["d_table_x", "d_table_y"]].to_numpy()
            p_xy = numpy.column_stack([group["p_table_x"].to_numpy(), group["p_table_y"].to_numpy()])
            p_length = numpy.linalg.norm(p_xy, axis = 1)
            p_xy = p_xy * (numpy.minimum(p_length, 1.2) / numpy.maximum(p_length, 1e-9))[:, None]
            d_twist, p_twist = group["d_twist"].to_numpy(), numpy.clip(group["p_twist"].to_numpy(), -10.0, 10.0)
            cumulative = lambda values: numpy.concatenate([numpy.zeros((1,) + values.shape[1:]), numpy.cumsum(values, axis = 0)])
            d_cumulative, p_cumulative = cumulative(d_xy), cumulative(p_xy)
            d_turn, p_turn = cumulative(d_twist), cumulative(p_twist)
            d_net = numpy.linalg.norm(d_cumulative[steps:] - d_cumulative[:-steps], axis = 1)
            p_net = numpy.linalg.norm(p_cumulative[steps:] - p_cumulative[:-steps], axis = 1)
            d_net_turn = numpy.abs(d_turn[steps:] - d_turn[:-steps])
            p_net_turn = numpy.abs(p_turn[steps:] - p_turn[:-steps])
            demonstrated_stalled += int(((d_net < 1.0) & (d_net_turn < 10.0)).sum())
            model_stalled += int(((p_net < 1.0) & (p_net_turn < 10.0)).sum())
            windows += len(d_net)
            model_net.append(p_net)
        model_net = numpy.concatenate(model_net)
        print(f"C. windows of {steps} steps ({windows}): net tip move under 1.0 cm AND net forearm turn under 10 degrees: demonstrations {100 * demonstrated_stalled / windows:.2f} %; "
              f"the model's own steps on the same pictures {100 * model_stalled / windows:.2f} % (model net move median {numpy.median(model_net):.2f} cm)")
