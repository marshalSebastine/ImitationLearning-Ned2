"""
Exam-risk checks on the targets section (read only).

Reads frames.csv (through the targets folder's common_frames helpers, which only read), the audit's view table and the
saved per-picture outputs of the end-to-end exact run (held-out sheets 03, 05, 06) and of the final candidate (held-out
sheets 01, 02). Writes exam_risk_checks.out.txt next to this script and nothing else.

Questions:
    A. Does the push direction get worse when the forearm (and camera) is turned far, near the joint 4 limit?
    B. How close do the demonstrations come to the planned joint 4 clamp of 115 deg, per sheet?
    C. Near the limit, does the model still twist outwards (so the clamp would bind)?
    D. Open-loop drift: how far does the summed predicted twist drift from the demonstrated forearm path?
    E. Oscillation proxies: twist sign flips and direction reversals between consecutive pictures.
    F. Stop threshold curves per held-out sheet, and an early-stop-averse choice against the section's rule.
    G. Start and near-goal pictures against always forward.
    H. A twist-in-place rule for the executor: how often it would fire and how often it would be right.
    I. Step budget at 2 steps per cm against the demonstrated step counts.
    J. Light: direction and false stop firing by lighting and brightness.
"""

import sys
from pathlib import Path

import numpy
import pandas

TARGETS_FOLDER = Path("C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/targets")
sys.path.insert(0, str(TARGETS_FOLDER))

from common_frames import AUDIT_FOLDER, DIRECTION_NAMES, audit_cleaned, read_frames  # noqa: E402

OUTPUT_FOLDER = Path(__file__).resolve().parent
BACK_NAMES = ["back", "back_left", "back_right"]
EXACT_FOLDS = ["without_path_03", "without_path_05", "without_path_06"]
FINAL_FOLDS = ["without_path_01", "without_path_02"]
THRESHOLDS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
LINE_LENGTH_CM = {"path_01": 24, "path_02": 27, "path_03": 55, "path_04": 77, "path_05": 57, "path_06": 57}
lines = []


def say(text: str = "") -> None:
    """Collect one line of the printout and echo it."""

    lines.append(text)
    print(text, flush = True)


def wrapped_difference_deg(first: numpy.ndarray, second: numpy.ndarray) -> numpy.ndarray:
    """Return the absolute angle between two directions in degrees (0 to 180)."""

    return numpy.abs((first - second + 180.0) % 360.0 - 180.0)


def load_table(csv_path: Path, frames: pandas.DataFrame) -> pandas.DataFrame:
    """Join one fold's saved outputs to the frame rows and add the angles used below."""

    outputs = pandas.read_csv(csv_path)
    table = outputs.merge(frames, on = "frame_index", how = "left")
    table = table.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    table["predicted_cm"] = numpy.hypot(table["forward_cm"], table["left_cm"])
    table["predicted_angle_deg"] = numpy.degrees(numpy.arctan2(table["left_cm"], table["forward_cm"]))
    table["demonstrated_angle_deg"] = numpy.degrees(numpy.arctan2(table["tip_left_m"], table["tip_forward_m"]))
    table["angle_error_deg"] = wrapped_difference_deg(table["predicted_angle_deg"].to_numpy(), table["demonstrated_angle_deg"].to_numpy())
    table["forward_error_deg"] = numpy.abs(table["demonstrated_angle_deg"])
    table["joint_4_deg"] = numpy.degrees(table["joint_4"])
    table["goal_distance_cm"] = table["goal_distance_cm"].fillna(0.0)
    table["is_moving"] = table["action"].isin(DIRECTION_NAMES)
    table["is_back"] = table["action"].isin(BACK_NAMES)
    return table


def stop_outcomes(table: pandas.DataFrame, passing: numpy.ndarray, consecutive: int) -> dict:
    """Per episode: stop at the first run of passing pictures; right within 2 cm, early beyond 3 cm, else between."""

    counts = {"right": 0, "early": 0, "between": 0, "never": 0}
    for _, group in table.groupby("episode_id", sort = False):
        flags = passing[group.index.to_numpy()]
        run = 0
        hit = None
        for position, flag in enumerate(flags):
            run = run + 1 if flag else 0
            if run >= consecutive:
                hit = position
                break
        if hit is None:
            counts["never"] += 1
            continue
        distance = float(group["goal_distance_cm"].iloc[hit])
        counts["right" if distance <= 2.0 else ("early" if distance > 3.0 else "between")] += 1
    total = sum(counts.values())
    shares = {name: value / total for name, value in counts.items()}
    shares["episodes"] = total
    return shares


def section_a_direction_by_turn(tables: dict, label: str) -> None:
    """Direction within 45 deg by the size of the forearm turn, against always forward."""

    pooled = pandas.concat(tables.values(), ignore_index = True)
    moving = pooled[pooled["is_moving"]]
    bins = [0, 30, 60, 90, 105, 125]
    moving = moving.assign(turn_bin = pandas.cut(moving["joint_4_deg"].abs(), bins, right = False))
    say(f"A. {label}: moving rows by |joint 4| (deg): n, model within 45, always forward within 45, back share, model within 45 on back rows (n)")
    for interval, part in moving.groupby("turn_bin", observed = True):
        back = part[part["is_back"]]
        say(f"   {interval}: {len(part)}, {(part['angle_error_deg'] <= 45).mean():.3f}, {(part['forward_error_deg'] <= 45).mean():.3f}, "
            f"{part['is_back'].mean():.3f}, {(back['angle_error_deg'] <= 45).mean() if len(back) else float('nan'):.3f} ({len(back)})")
    for fold, table in tables.items():
        part = table[table["is_moving"]]
        far = part[part["joint_4_deg"].abs() >= 90]
        near = part[part["joint_4_deg"].abs() < 60]
        say(f"   {fold}: |joint 4| under 60: {(near['angle_error_deg'] <= 45).mean():.3f} of {len(near)}; 90 or more: "
            f"{(far['angle_error_deg'] <= 45).mean() if len(far) else float('nan'):.3f} of {len(far)}")


def section_b_joint_limit(frames: pandas.DataFrame) -> None:
    """How close the demonstrations come to the joint 4 clamp, per sheet."""

    frames = frames.assign(joint_4_deg = numpy.degrees(frames["joint_4"]))
    say("B. joint 4 in the demonstrations (raw rows), per sheet: episodes, episodes with max |joint 4| >= 110 / >= 115, "
        "rows with |joint 4| >= 115, median start joint 4, median episode range (max - min)")
    for sheet, part in frames.groupby("sheet"):
        per_episode = part.groupby("episode_id")["joint_4_deg"]
        largest = per_episode.apply(lambda values: values.abs().max())
        start = part.sort_values("row_in_episode").groupby("episode_id")["joint_4_deg"].first()
        spread = per_episode.max() - per_episode.min()
        say(f"   {sheet}: {len(largest)}, {(largest >= 110).sum()} / {(largest >= 115).sum()}, {(part['joint_4_deg'].abs() >= 115).sum()} of {len(part)}, "
            f"{start.median():.1f}, {spread.median():.1f}")
    ordered = frames.sort_values(["episode_id", "row_in_episode"])
    next_joint = ordered.groupby("episode_id")["joint_4_deg"].shift(-1)
    difference = (next_joint - ordered["joint_4_deg"]) - ordered["forearm_change_deg"]
    say(f"   sign check: median |next joint 4 - joint 4 - forearm_change_deg| {difference.abs().median():.3f} deg "
        f"(small means a positive forearm_change_deg raises joint 4)")


def section_c_outward_twist(tables: dict, label: str) -> None:
    """Near the limit: share of pictures whose twist turns joint 4 further out, model against demonstration."""

    pooled = pandas.concat(tables.values(), ignore_index = True)
    for low in (90, 105):
        part = pooled[(pooled["joint_4_deg"].abs() >= low) & (pooled["action"] != "stop")]
        if len(part) == 0:
            say(f"C. {label}: no pictures with |joint 4| >= {low}")
            continue
        outward_model = (numpy.sign(part["twist_deg"]) == numpy.sign(part["joint_4_deg"])) & (part["twist_deg"].abs() >= 1.0)
        outward_demo = (numpy.sign(part["forearm_change_deg"]) == numpy.sign(part["joint_4_deg"])) & (part["forearm_change_deg"].abs() >= 1.0)
        say(f"C. {label}: pictures with |joint 4| >= {low}: {len(part)} in {part['episode_id'].nunique()} episodes; twist of 1 deg or more outwards: "
            f"model {outward_model.mean():.3f}, demonstration {outward_demo.mean():.3f}; median outward twist when outward: model "
            f"{part.loc[outward_model, 'twist_deg'].abs().median():.2f}, demonstration {part.loc[outward_demo, 'forearm_change_deg'].abs().median():.2f}")


def section_d_open_loop_drift(tables: dict, label: str) -> None:
    """Sum the predicted twists along each demonstrated episode and compare with the demonstrated forearm path."""

    gaps_end = []
    gaps_most = []
    crossings_model = 0
    crossings_demo = 0
    episodes = 0
    for table in tables.values():
        for _, group in table.groupby("episode_id", sort = False):
            group = group[group["action"] != "stop"]
            if len(group) < 5:
                continue
            episodes += 1
            start = float(group["joint_4_deg"].iloc[0])
            model_path = start + numpy.cumsum(group["twist_deg"].to_numpy())
            demo_path = start + numpy.cumsum(group["forearm_change_deg"].fillna(0.0).to_numpy())
            gap = model_path - demo_path
            gaps_end.append(abs(gap[-1]))
            gaps_most.append(numpy.abs(gap).max())
            crossings_model += int(numpy.abs(model_path).max() >= 115)
            crossings_demo += int(numpy.abs(demo_path).max() >= 115)
    gaps_end = numpy.array(gaps_end)
    gaps_most = numpy.array(gaps_most)
    say(f"D. {label}: {episodes} episodes; summed predicted twist against the demonstrated forearm path (open loop, no feedback): "
        f"gap at the end median {numpy.median(gaps_end):.1f} deg (p90 {numpy.quantile(gaps_end, 0.9):.1f}); largest gap median "
        f"{numpy.median(gaps_most):.1f} (p90 {numpy.quantile(gaps_most, 0.9):.1f}); episodes whose path passes |115| deg: model "
        f"{crossings_model}, demonstration {crossings_demo}")


def section_e_oscillation(tables: dict, label: str) -> None:
    """Twist sign flips and direction reversals between consecutive pictures."""

    for fold, table in tables.items():
        flips_model = []
        flips_demo = []
        reversals_model = []
        reversals_demo = []
        for _, group in table.groupby("episode_id", sort = False):
            group = group[group["action"] != "stop"]
            twist = group["twist_deg"].to_numpy()
            demo = group["forearm_change_deg"].fillna(0.0).to_numpy()
            both = (numpy.abs(twist[1:]) >= 1.0) & (numpy.abs(twist[:-1]) >= 1.0)
            flips_model.extend(list((numpy.sign(twist[1:]) != numpy.sign(twist[:-1]))[both]))
            both_demo = (numpy.abs(demo[1:]) >= 1.0) & (numpy.abs(demo[:-1]) >= 1.0)
            flips_demo.extend(list((numpy.sign(demo[1:]) != numpy.sign(demo[:-1]))[both_demo]))
            moving = group[group["is_moving"]]
            predicted = moving["predicted_angle_deg"].to_numpy()
            demonstrated = moving["demonstrated_angle_deg"].to_numpy()
            reversals_model.extend(list(wrapped_difference_deg(predicted[1:], predicted[:-1]) > 90.0))
            reversals_demo.extend(list(wrapped_difference_deg(demonstrated[1:], demonstrated[:-1]) > 90.0))
        say(f"E. {label} {fold}: twist sign flips between consecutive pictures (both 1 deg or more): model {numpy.mean(flips_model):.3f} of {len(flips_model)}, "
            f"demonstration {numpy.mean(flips_demo):.3f} of {len(flips_demo)}; direction reversals over 90 deg between consecutive moving pictures: "
            f"model {numpy.mean(reversals_model):.3f}, demonstration {numpy.mean(reversals_demo):.3f} of {len(reversals_model)}")


def section_f_stop(tables: dict, label: str) -> dict:
    """Threshold curves per held-out sheet and the stop probability on the stop pictures."""

    curves = {}
    for fold, table in tables.items():
        probability = table["stop_probability"].to_numpy()
        stop_rows = table[table["action"] == "stop"]
        say(f"F. {label} {fold}: episodes {table['episode_id'].nunique()}; stop probability on stop pictures median {stop_rows['stop_probability'].median():.3f} "
            f"(p25 {stop_rows['stop_probability'].quantile(0.25):.3f}); threshold: right/early/never, 1 picture | 2 in a row")
        for threshold in THRESHOLDS:
            one = stop_outcomes(table, probability >= threshold, 1)
            two = stop_outcomes(table, probability >= threshold, 2)
            curves[(fold, threshold, 1)] = one
            curves[(fold, threshold, 2)] = two
            say(f"   {threshold:.2f}: {one['right']:.2f}/{one['early']:.2f}/{one['never']:.2f} | {two['right']:.2f}/{two['early']:.2f}/{two['never']:.2f}")
        for consecutive in (1, 2):
            best = max(curves[(fold, threshold, consecutive)]["right"] for threshold in THRESHOLDS)
            plateau = [threshold for threshold in THRESHOLDS if curves[(fold, threshold, consecutive)]["right"] >= best - 0.05]
            say(f"   {consecutive} in a row: best right {best:.2f}; thresholds within 0.05 of it: {plateau}")
    return curves


def choose_on_others(curves: dict, folds: list, consecutive: int, rule: str) -> list:
    """Choose a threshold on the other folds (pooled means) by a rule; return each fold's result at its choice."""

    results = []
    for fold in folds:
        others = [other for other in folds if other != fold]

        def score(threshold: float) -> tuple:
            right = numpy.mean([curves[(other, threshold, consecutive)]["right"] for other in others])
            early = numpy.mean([curves[(other, threshold, consecutive)]["early"] for other in others])
            if rule == "section":
                return (right, threshold)
            return (right - 2.0 * early, threshold)

        chosen = max(THRESHOLDS, key = score)
        results.append((fold, chosen, curves[(fold, chosen, consecutive)]))
    return results


def section_g_start_and_goal(tables: dict, label: str) -> None:
    """The first three moving pictures of each episode, and moving pictures within 3 cm of the goal."""

    pooled = pandas.concat(tables.values(), ignore_index = True)
    moving = pooled[pooled["is_moving"]].copy()
    moving["moving_position"] = moving.groupby("episode_id").cumcount()
    start = moving[moving["moving_position"] < 3]
    near_goal = moving[moving["goal_distance_cm"] <= 3.0]
    rest = moving[(moving["moving_position"] >= 3) & (moving["goal_distance_cm"] > 3.0)]
    for name, part in (("first 3 moving pictures", start), ("within 3 cm of the goal", near_goal), ("the rest", rest)):
        say(f"G. {label}, {name}: n {len(part)}; model within 45 {(part['angle_error_deg'] <= 45).mean():.3f}, always forward "
            f"{(part['forward_error_deg'] <= 45).mean():.3f}; model median error {part['angle_error_deg'].median():.1f} deg")


def section_h_twist_in_place(tables: dict, label: str) -> None:
    """Rule: twist only (no tip move) when the predicted twist is 3 deg or more and the predicted step is short."""

    pooled = pandas.concat(tables.values(), ignore_index = True)
    pooled = pooled[pooled["action"] != "stop"]
    demonstrated_in_place = (pooled["action"] == "small_motion") & (pooled["forearm_change_deg"].abs() >= 3.0)
    for length_cut in (0.3, 0.4, 0.5):
        fires = (pooled["twist_deg"].abs() >= 3.0) & (pooled["predicted_cm"] < length_cut)
        say(f"H. {label}, twist in place when |predicted twist| >= 3 deg and predicted step < {length_cut} cm: fires on {fires.mean():.3f} of "
            f"{len(pooled)} pictures; of these {demonstrated_in_place[fires].mean():.3f} were demonstrated twists in place; catches "
            f"{fires[demonstrated_in_place].mean():.3f} of {int(demonstrated_in_place.sum())} demonstrated twists in place; twist sign right where it fires "
            f"{(numpy.sign(pooled.loc[fires, 'twist_deg']) == numpy.sign(pooled.loc[fires, 'forearm_change_deg'])).mean():.3f}")


def section_i_step_budget(frames: pandas.DataFrame) -> None:
    """Demonstrated steps per cm of line against a budget of 2 steps per cm."""

    say("I. steps per episode (raw rows) against 2 steps per cm of line: sheet, line cm, budget, median steps, p90, max, "
        "episodes over budget; the same for episodes without hand corrections")
    for sheet, part in frames.groupby("sheet"):
        length = LINE_LENGTH_CM[sheet]
        budget = 2 * length
        steps = part.groupby("episode_id").size()
        corrections = part.groupby("episode_id")["hand_corrections"].first().fillna(0)
        clean = steps[corrections == 0]
        say(f"   {sheet}: {length}, {budget}, {steps.median():.0f}, {steps.quantile(0.9):.0f}, {steps.max()}, {(steps > budget).sum()} of {len(steps)}; "
            f"without corrections: median {clean.median() if len(clean) else float('nan'):.0f}, max {clean.max() if len(clean) else float('nan')}, "
            f"over budget {(clean > budget).sum()} of {len(clean)}")


def section_j_light(tables: dict, label: str) -> None:
    """Direction and false stop firing far from the goal, by the lighting note and by brightness third."""

    pooled = pandas.concat(tables.values(), ignore_index = True)
    pooled = pooled[pooled["action"] != "stop"]
    say(f"J. {label}: lighting notes in these folds {pooled['lighting'].value_counts().to_dict()}")
    thirds = pandas.qcut(pooled["brightness_mean"], 3, labels = ["darkest third", "middle third", "brightest third"])
    for name, part in pooled.groupby(thirds, observed = True):
        moving = part[part["is_moving"]]
        far = part[part["goal_distance_cm"] > 5.0]
        say(f"   {name} (brightness {part['brightness_mean'].min():.0f} to {part['brightness_mean'].max():.0f}): moving within 45 "
            f"{(moving['angle_error_deg'] <= 45).mean():.3f} of {len(moving)}; far pictures with stop probability >= 0.3: "
            f"{(far['stop_probability'] >= 0.3).mean():.4f} of {len(far)}")


def main() -> None:
    raw = read_frames()
    view = pandas.read_csv(AUDIT_FOLDER / "view_and_tracks" / "s09_view_rows.csv")
    raw = raw.merge(view[["frame_index", "line_share_top_half"]], on = "frame_index", how = "left")
    cleaned = audit_cleaned(read_frames())
    cleaned = cleaned.merge(view[["frame_index", "line_share_top_half"]], on = "frame_index", how = "left")
    keep = ["frame_index", "episode_id", "sheet", "row_in_episode", "goal_distance_cm", "action", "tip_forward_m", "tip_left_m",
            "forearm_change_deg", "joint_4", "hand_corrections", "lighting", "brightness_mean", "line_share_top_half"]
    exact_tables = {fold: load_table(TARGETS_FOLDER / f"exact_outputs_{fold}.csv", raw[keep]) for fold in EXACT_FOLDS}
    final_tables = {fold: load_table(TARGETS_FOLDER / f"final_outputs_{fold}.csv", cleaned[keep]) for fold in FINAL_FOLDS}
    say(f"rows: end-to-end exact {sum(len(table) for table in exact_tables.values())}, final candidate {sum(len(table) for table in final_tables.values())}; "
        f"unmatched frame rows: {sum(int(table['episode_id'].isna().sum()) for table in list(exact_tables.values()) + list(final_tables.values()))}")

    section_a_direction_by_turn(exact_tables, "end-to-end exact, sheets 03, 05, 06")
    section_a_direction_by_turn(final_tables, "final candidate, sheets 01, 02")
    section_b_joint_limit(raw)
    section_c_outward_twist(exact_tables, "end-to-end exact, sheets 03, 05, 06")
    section_c_outward_twist(final_tables, "final candidate, sheets 01, 02")
    section_d_open_loop_drift(exact_tables, "end-to-end exact, sheets 03, 05, 06")
    section_d_open_loop_drift(final_tables, "final candidate, sheets 01, 02")
    section_e_oscillation(exact_tables, "end-to-end exact")
    section_e_oscillation(final_tables, "final candidate")
    exact_curves = section_f_stop(exact_tables, "end-to-end exact")
    section_f_stop(final_tables, "final candidate")
    for consecutive in (1, 2):
        for rule in ("section", "early_averse"):
            results = choose_on_others(exact_curves, EXACT_FOLDS, consecutive, rule)
            right = numpy.mean([result[2]["right"] for result in results])
            early = numpy.mean([result[2]["early"] for result in results])
            never = numpy.mean([result[2]["never"] for result in results])
            say(f"F. end-to-end exact, threshold from the other two sheets, {consecutive} in a row, rule {rule}: right {right:.3f}, early {early:.3f}, "
                f"never {never:.3f}; " + " ".join(f"{result[0][-2:]}@{result[1]}:{result[2]['right']:.2f}/{result[2]['early']:.2f}/{result[2]['never']:.2f}"
                                                   for result in results))
    section_g_start_and_goal(exact_tables, "end-to-end exact, sheets 03, 05, 06")
    section_g_start_and_goal(final_tables, "final candidate, sheets 01, 02")
    section_h_twist_in_place(exact_tables, "end-to-end exact, sheets 03, 05, 06")
    section_h_twist_in_place(final_tables, "final candidate, sheets 01, 02")
    section_i_step_budget(raw)
    section_j_light(exact_tables, "end-to-end exact, sheets 03, 05, 06")
    section_j_light(final_tables, "final candidate, sheets 01, 02")
    (OUTPUT_FOLDER / "exam_risk_checks.out.txt").write_text("\n".join(lines), encoding = "utf-8")


if __name__ == "__main__":
    main()
