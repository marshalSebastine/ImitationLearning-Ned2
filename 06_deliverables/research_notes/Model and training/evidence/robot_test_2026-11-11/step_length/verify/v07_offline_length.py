"""
Verifier's own offline numbers on the predicted step length (fold models on their held-out sheets).

Uses the analyst's fold_outputs_<recipe>.npz only after v06 showed that they equal my own inference; every statistic
below is computed here from scratch against the cleaned labels (training_targets.load_training_frames). No training,
nothing connects to the robot.

Run from this folder:  python v07_offline_length.py > v07_offline_length.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

ROOT = Path("D:/AI_Project/.claude/worktrees/training-analysis")
sys.path.insert(0, str(ROOT / "02_code"))
from config import TRAINING_TARGET_PARAMETERS
from train_policy import merged_run_settings
from training_targets import load_training_frames

STEP_LENGTH = ROOT / "06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length"
pandas.set_option("display.width", 250)


def area_under_curve(scores_positive, scores_negative):
    """Chance that a random positive has a LOWER score than a random negative (ties count half)."""
    scores = numpy.concatenate([scores_positive, scores_negative])
    ranks = pandas.Series(scores).rank(method = "average").to_numpy()
    rank_sum_positive = ranks[:len(scores_positive)].sum()
    u = rank_sum_positive - len(scores_positive) * (len(scores_positive) + 1) / 2.0
    return 1.0 - u / (len(scores_positive) * len(scores_negative))


def angle_between(first, second):
    """Absolute angle in degrees between the rows of two (rows, 2) arrays."""
    first_angle = numpy.degrees(numpy.arctan2(first[:, 1], first[:, 0]))
    second_angle = numpy.degrees(numpy.arctan2(second[:, 1], second[:, 0]))
    return numpy.abs((first_angle - second_angle + 180.0) % 360.0 - 180.0)


# the rows and labels the fold models were scored on: the run's own target settings merged over the defaults
target_parameters = dict(TRAINING_TARGET_PARAMETERS)
run_settings = merged_run_settings("resnet18_chunk_stop_weight_25")
target_parameters.update({key: value for key, value in run_settings.items() if key in TRAINING_TARGET_PARAMETERS})
other_settings = merged_run_settings("resnet18_final_candidate")
differing = {key: (run_settings[key], other_settings[key]) for key in TRAINING_TARGET_PARAMETERS if key in run_settings and run_settings.get(key) != other_settings.get(key)}
print(f"target settings that differ between the E and the A recipe: {differing}")
cleaned = load_training_frames(target_parameters)
print(f"cleaned frames: {len(cleaned)} rows, {cleaned['episode_id'].nunique()} episodes; actions: {cleaned['action'].value_counts().to_dict()}")
cleaned = cleaned.set_index("frame_index")

for recipe in ["E_recipe", "A_recipe"]:
    data = numpy.load(STEP_LENGTH / f"fold_outputs_{recipe}.npz")
    rows = cleaned.loc[data["frame_index"]]
    is_step = (rows["action"] != "stop").to_numpy()
    demonstrated_vector = 100.0 * rows[["tip_forward_m", "tip_left_m"]].fillna(0.0).to_numpy()
    demonstrated_twist = rows["forearm_change_deg"].fillna(0.0).to_numpy()
    predicted_vector = data["exact"][:, :2].astype(float)
    predicted_twist = data["exact"][:, 2].astype(float)
    sheet = rows["sheet"].to_numpy()
    print()
    print(f"================ {recipe}: {len(rows)} pictures, {int(is_step.sum())} without the stop rows")
    dv, pv = demonstrated_vector[is_step], predicted_vector[is_step]
    d, p = numpy.linalg.norm(dv, axis = 1), numpy.linalg.norm(pv, axis = 1)
    dt, pt = demonstrated_twist[is_step], predicted_twist[is_step]
    s = sheet[is_step]
    names = sorted(set(s))
    print(f"demonstrated step: median {numpy.median(d):.3f}, mean {d.mean():.3f}; under 0.5 cm {100 * (d < 0.5).mean():.1f} %, under 0.2 cm {100 * (d < 0.2).mean():.1f} %; "
          f"with a twist of 6 degrees or more: median {numpy.median(d[numpy.abs(dt) >= 6]):.3f}")
    print(f"predicted length: median {numpy.median(p):.3f}, mean {p.mean():.3f}; under 0.3 cm {100 * (p < 0.3).mean():.1f} %; over 1.2 cm {100 * (p > 1.2).mean():.2f} %; largest {p.max():.2f}")

    is_twist_in_place = (d < 0.3) & (numpy.abs(dt) >= 3.0)
    is_move = d >= 0.3
    print(f"1. twists in place {int(is_twist_in_place.sum())} ({100 * is_twist_in_place.mean():.1f} % of rows), real moves {int(is_move.sum())}, neither {int((~is_twist_in_place & ~is_move).sum())}")
    print("   per sheet share of twists in place: " + ", ".join(f"{name[-2:]} {100 * is_twist_in_place[s == name].mean():.1f} %" for name in names))
    print(f"   predicted length median: twists in place {numpy.median(p[is_twist_in_place]):.3f}, real moves {numpy.median(p[is_move]):.3f}")
    print(f"   area under the curve (predicted length lower for twists in place): {area_under_curve(p[is_twist_in_place], p[is_move]):.3f}; per sheet: "
          + ", ".join(f"{name[-2:]} {area_under_curve(p[is_twist_in_place & (s == name)], p[is_move & (s == name)]):.3f}" for name in names))
    ratio_score = numpy.abs(pt) / numpy.maximum(p, 1e-6)
    print(f"   other scores: |predicted twist| / length (higher for twists in place): {area_under_curve(-ratio_score[is_twist_in_place], -ratio_score[is_move]):.3f}; "
          f"|predicted twist| alone: {area_under_curve(-numpy.abs(pt)[is_twist_in_place], -numpy.abs(pt)[is_move]):.3f}")
    for threshold in (0.2, 0.3, 0.5):
        missed = (p[is_twist_in_place] >= threshold).mean()
        frozen = (p[is_move] < threshold).mean()
        under = p < threshold
        print(f"   threshold {threshold:.1f} cm: twists in place missed {100 * missed:.1f} %; real moves frozen {100 * frozen:.1f} %; of the {int(under.sum())} rows under it "
              f"{100 * is_twist_in_place[under].mean():.1f} % are twists in place")
    thresholds = numpy.linspace(0.05, 1.5, 2901)
    missed = numpy.array([(p[is_twist_in_place] >= value).mean() for value in thresholds])
    frozen = numpy.array([(p[is_move] < value).mean() for value in thresholds])
    best = int(numpy.argmin(numpy.abs(missed - frozen)))
    print(f"   equal-error threshold {thresholds[best]:.2f} cm: missed {100 * missed[best]:.1f} %, frozen {100 * frozen[best]:.1f} %")

    print(f"2. gain, moving rows ({int(is_move.sum())}): median of ratios {numpy.median(d[is_move] / p[is_move]):.3f}; ratio of medians {numpy.median(d[is_move]) / numpy.median(p[is_move]):.3f}; "
          f"ratio of means {d[is_move].mean() / p[is_move].mean():.3f}; predicted median {numpy.median(p[is_move]):.3f}, demonstrated median {numpy.median(d[is_move]):.3f}")
    print("   per sheet (median of ratios): " + ", ".join(f"{name[-2:]} {numpy.median(d[is_move & (s == name)] / p[is_move & (s == name)]):.2f}" for name in names))
    print(f"   correlation predicted with demonstrated length: all step rows {numpy.corrcoef(p, d)[0, 1]:.3f}; moving rows {numpy.corrcoef(p[is_move], d[is_move])[0, 1]:.3f}; "
          f"rank correlation all {pandas.Series(p).corr(pandas.Series(d), method = 'spearman'):.3f}")

    short = p < 0.3
    clear = d >= 0.5
    angle_error = angle_between(pv, dv)
    print(f"3. where predicted < 0.3 cm ({int(short.sum())} rows): demonstrated median {numpy.median(d[short]):.3f} cm, mean {d[short].mean():.3f}; direction within 45 degrees "
          f"(demonstrated >= 0.5 cm) {100 * (angle_error[short & clear] <= 45).mean():.1f} %, median error {numpy.median(angle_error[short & clear]):.1f}; "
          f"where predicted >= 0.9: within 45 {100 * (angle_error[(p >= 0.9) & clear] <= 45).mean():.1f} %, median error {numpy.median(angle_error[(p >= 0.9) & clear]):.1f}")
    print(f"   all clear rows: direction within 45 degrees {100 * (angle_error[clear] <= 45).mean():.1f} %")

    demonstrated_rate = numpy.abs(dt).sum() / d.sum()
    print(f"4. forearm turn per cm: demonstrated {demonstrated_rate:.3f} deg/cm; shrink of length {p.mean() / d.mean():.3f}, of twist size {numpy.abs(pt).mean() / numpy.abs(dt).mean():.3f}")
    rules = {"fixed 1.0 cm": (numpy.ones_like(p), 1.0), "own length, at most 1.2": (numpy.minimum(p, 1.2), 1.0), "gain 1.5, at most 1.0": (numpy.minimum(1.5 * p, 1.0), 1.0),
             "gain 1.6, clamp 0.5 to 1.5 (report 4.4)": (numpy.clip(1.6 * p, 0.5, 1.5), 1.0), "gain 1.4 on length and twist, at most 1.2": (numpy.minimum(1.4 * p, 1.2), 1.4),
             "fixed 1.0 cm, twist gain 1.6": (numpy.ones_like(p), 1.6)}
    bend = numpy.abs(dt) >= 5.0
    straight = numpy.abs(dt) < 2.0
    direction = pv / numpy.maximum(p, 1e-9)[:, None]
    for name, (length, twist_gain) in rules.items():
        twist_used = numpy.clip(twist_gain * pt, -10.0, 10.0)
        share = (numpy.abs(twist_used).sum() / length.sum()) / demonstrated_rate
        per_sheet = ", ".join(f"{sheet_name[-2:]} {(numpy.abs(twist_used[s == sheet_name]).sum() / length[s == sheet_name].sum()) / (numpy.abs(dt[s == sheet_name]).sum() / d[s == sheet_name].sum()):.2f}"
                              for sheet_name in names)
        landing = numpy.linalg.norm(direction * length[:, None] - dv, axis = 1)
        print(f"   {name:42s}: turn per cm as a share of demonstrated {share:.3f} ({per_sheet}); landing error one step ahead, mean: all {landing.mean():.3f}, "
              f"bend rows {landing[bend].mean():.3f}, straight rows {landing[straight].mean():.3f}, twists in place {landing[is_twist_in_place].mean():.3f}; mean step {length.mean():.3f}; "
              f"twist clamped at 10 degrees on {100 * (numpy.abs(twist_gain * pt) > 10).mean():.2f} %")
    sign_right = (numpy.sign(pt) == numpy.sign(dt))[numpy.abs(dt) >= 2.0].mean()
    print(f"   twist sign right where the demonstrated twist is 2 degrees or more: {100 * sign_right:.1f} %; bend rows {int(bend.sum())}, straight rows {int(straight.sum())}")
    # the same turn per cm on bend rows only, where it matters
    for name, (length, twist_gain) in rules.items():
        twist_used = numpy.clip(twist_gain * pt, -10.0, 10.0)
        share = (numpy.abs(twist_used[bend]).sum() / length[bend].sum()) / (numpy.abs(dt[bend]).sum() / d[bend].sum())
        signed = ((twist_used[bend] * numpy.sign(dt[bend])).sum() / length[bend].sum()) / (numpy.abs(dt[bend]).sum() / d[bend].sum())
        print(f"   bend rows only, {name:42s}: turn per cm share {share:.3f}; counting a wrong-sign twist as negative {signed:.3f}")

    if "chunk" in data.files:
        chunk = data["chunk"][is_step].astype(float)
        first = numpy.linalg.norm(chunk[:, 0, :], axis = 1)
        with_start = numpy.concatenate([numpy.zeros((len(chunk), 1, 2)), chunk], axis = 1)
        hops = numpy.linalg.norm(numpy.diff(with_start, axis = 1), axis = 2)
        print(f"5. chunk output: length of its first position median {numpy.median(first):.3f} (exact step {numpy.median(p):.3f}); mean hop over the 5 positions median {numpy.median(hops.mean(axis = 1)):.3f}; "
              f"net distance of the fifth position / 5 median {numpy.median(numpy.linalg.norm(chunk[:, 4, :], axis = 1) / 5):.3f}; correlation of first-position length with exact length "
              f"{numpy.corrcoef(first, p)[0, 1]:.3f}")
        for position in range(5):
            chunk_error = angle_between(chunk[:, position, :], dv)
            print(f"   direction of chunk position {position + 1} within 45 degrees of the demonstrated step (demonstrated >= 0.5 cm): {100 * (chunk_error[clear] <= 45).mean():.1f} % "
                  f"(exact step {100 * (angle_error[clear] <= 45).mean():.1f} %)")
