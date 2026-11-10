"""
Judge 1: re-score the saved held-out-sheet predictions of every end-to-end run (read only).

Nothing under D:/AI_Project is written; the printout goes next to this file (judge_rescoring.out.txt and
judge_rescoring.out.json). The rules copy training_metrics.stop_metrics and training_probes.path_directions:

- a stop rule (threshold, pictures in a row) stops an episode at the first picture where that many pictures in a row
  reach the threshold; "right" within 2 cm of the demonstrated stop, "early" more than 3 cm before it, "never" when no
  picture passes, "near" between 2 and 3 cm;
- the path direction runs from the tip now to where it is 3 kept rows later (the last row when fewer remain), in the
  current picture's frame; it is scored on non-stop rows whose path is at least 1.5 cm long;
- moving rows are those whose action is one of the 8 direction names.

Parts:
1. direction per fold, reproduced against the fold files, plus three strata (backward moves, pictures after a hand
   correction, the last 3 cm) and always-forward on the same rows;
2. stop outcomes per fold and pooled for: fixed 0.5 (one picture and two in a row), the fold's own inner-validation
   rules (read from the fold file), and rules chosen on the OTHER held-out sheets of the same run (leave one sheet
   out) by two criteria;
3. stop outcomes by the line end the episode finishes at (distance from the paper edge, audit table 10.1);
4. a finishing proxy per episode (judgement): stopped right under the leave-one-sheet-out rule and no streak of 5 or
   more consecutive path pictures more than 45 degrees off before the stop.
"""

import json
import math
from pathlib import Path

import numpy
import pandas

PARAMETERS = {
    "runs_folder": Path("D:/AI_Project/.claude/worktrees/training-analysis/04_runs"),
    "frames_path": Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv"),
    "corrected_ends_path": Path("D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/corrected_ends.csv"),
    "output_folder": Path(__file__).resolve().parent,
    "runs": {
        "final_candidate": {"folder": "2026-11-09_resnet18_final_candidate", "kind": "exact"},
        "finetune_exact": {"folder": "2026-11-09_resnet18_finetune_exact", "kind": "exact"},
        "finetune_chunk": {"folder": "2026-11-09_resnet18_finetune_chunk", "kind": "chunk"},
        "finetune_classes": {"folder": "2026-11-09_resnet18_finetune_classes", "kind": "classes"},
        "finetune_mirror": {"folder": "2026-11-09_resnet18_finetune_mirror", "kind": "classes"},
        "small_cnn_classes": {"folder": "2026-11-09_small_cnn_scratch_classes", "kind": "classes"},
        "small_cnn_line_and_puck": {"folder": "2026-11-09_small_cnn_line_and_puck_classes", "kind": "classes"},
    },
    "sheets": ["01", "02", "03", "04", "05", "06"],
    "direction_names": ["forward", "forward_left", "forward_right", "left", "right", "back", "back_left", "back_right"],
    "backward_names": ["back", "back_left", "back_right"],
    "picture_left_sign": 1,
    "path_steps": 3,
    "path_minimum_cm": 1.5,
    "right_stop_cm": 2.0,
    "early_stop_cm": 3.0,
    "thresholds": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "streak_length": 5,
    "edge_distance_cm": {("path_01", "A"): 3.7, ("path_01", "B"): 3.0, ("path_02", "A"): 6.2, ("path_02", "B"): 6.4,
                         ("path_03", "A"): 3.6, ("path_03", "B"): 2.5, ("path_04", "A"): 2.4, ("path_04", "B"): 2.8,
                         ("path_05", "A"): 5.9, ("path_05", "B"): 6.0, ("path_06", "A"): 4.4, ("path_06", "B"): 8.0},
}



def angle_difference_deg(first_deg, second_deg):
    """
    Return the absolute difference of two angles in degrees, folded into 0 to 180.
    """

    return numpy.abs((numpy.asarray(first_deg, dtype = float) - numpy.asarray(second_deg, dtype = float) + 180.0) % 360.0 - 180.0)



def wilson_interval(successes, trials, z_value = 1.96):
    """
    Return the 95 % Wilson interval of a share as a pair of floats.
    """

    if trials == 0:
        return (float("nan"), float("nan"))
    share = successes / trials
    denominator = 1.0 + z_value ** 2 / trials
    centre = (share + z_value ** 2 / (2 * trials)) / denominator
    half_width = z_value * math.sqrt(share * (1 - share) / trials + z_value ** 2 / (4 * trials ** 2)) / denominator
    return (round(centre - half_width, 3), round(centre + half_width, 3))



def loaded_frames(parameters):
    """
    Read the columns of frames.csv the scoring needs, with the hand-correction count and the finishing end.
    """

    columns = ["frame_index", "episode_id", "sheet", "row_in_episode", "action", "goal_distance_cm", "move_angle_deg",
               "waypoint", "tip_x_m", "tip_y_m", "picture_heading_deg", "forearm_change_deg", "joint_4", "start_end"]
    frames = pandas.read_csv(parameters["frames_path"], usecols = columns, low_memory = False)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    is_restart = ((frames["waypoint"].fillna(-1) == 0) & (frames["row_in_episode"] > 0)).astype(int)
    frames["corrections_before"] = is_restart.groupby(frames["episode_id"]).cumsum()
    corrected_ends = pandas.read_csv(parameters["corrected_ends_path"]).set_index("episode_id")["corrected_end"]
    frames["start_end_corrected"] = frames["episode_id"].map(corrected_ends).fillna(frames["start_end"])
    frames["finish_end"] = frames["start_end_corrected"].map({"A": "B", "B": "A"})
    return frames.set_index("frame_index", drop = False)



def path_angles(test_frames, parameters):
    """
    Return the path direction and its length for the test rows, computed within each episode of the test rows.
    """

    tip_x = test_frames["tip_x_m"].to_numpy()
    tip_y = test_frames["tip_y_m"].to_numpy()
    heading = numpy.radians(test_frames["picture_heading_deg"].to_numpy())
    episode_ids = test_frames["episode_id"].to_numpy()
    angle = numpy.full(len(test_frames), numpy.nan)
    length_cm = numpy.zeros(len(test_frames))
    starts = numpy.flatnonzero(numpy.r_[True, episode_ids[1:] != episode_ids[:-1]])
    ends = numpy.r_[starts[1:], len(test_frames)]
    sign = parameters["picture_left_sign"]
    for start, end in zip(starts, ends):
        for position in range(start, end):
            future = min(position + parameters["path_steps"], end - 1)
            moved_x = tip_x[future] - tip_x[position]
            moved_y = tip_y[future] - tip_y[position]
            forward = moved_x * math.cos(heading[position]) + moved_y * math.sin(heading[position])
            left = moved_x * math.cos(heading[position] + sign * math.pi / 2) + moved_y * math.sin(heading[position] + sign * math.pi / 2)
            angle[position] = math.degrees(math.atan2(left, forward))
            length_cm[position] = 100.0 * math.hypot(forward, left)
    return angle, length_cm



def stop_outcome(stop_probability, goal_distance_cm, threshold, consecutive_pictures, parameters):
    """
    Return the outcome of one episode ("right", "early", "near" or "never") and the stopping position.
    """

    is_passing = stop_probability >= threshold
    run_length = numpy.zeros(len(is_passing), dtype = int)
    for position in range(len(is_passing)):
        run_length[position] = (run_length[position - 1] + 1 if position > 0 else 1) if is_passing[position] else 0
    hits = numpy.flatnonzero(run_length >= consecutive_pictures)
    if len(hits) == 0:
        return "never", None
    distance = float(goal_distance_cm[hits[0]])
    if distance <= parameters["right_stop_cm"]:
        return "right", int(hits[0])
    if distance > parameters["early_stop_cm"]:
        return "early", int(hits[0])
    return "near", int(hits[0])



def fold_table(run_name, run_settings, sheet, frames, parameters):
    """
    Return the joined test rows of one fold with the predictions, or None when the fold is missing.
    """

    folder = parameters["runs_folder"] / run_settings["folder"]
    prediction_path = folder / f"predictions_without_path_{sheet}.npz"
    fold_path = folder / f"fold_without_path_{sheet}.json"
    if not prediction_path.exists() or not fold_path.exists():
        return None
    with numpy.load(prediction_path) as content:
        predictions = {key: content[key] for key in content.files}
    test_frames = frames.loc[predictions["frame_index"]].copy()
    test_frames["order"] = numpy.arange(len(test_frames))
    for key in ("angle_deg", "angle_expected_deg", "angle_vector_deg", "stop_probability", "twist_deg"):
        if key in predictions:
            test_frames[key] = predictions[key].astype(float)
    test_frames = test_frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    path_angle, path_length = path_angles(test_frames, parameters)
    test_frames["path_angle_deg"] = path_angle
    test_frames["path_length_cm"] = path_length
    test_frames["is_moving"] = test_frames["action"].isin(parameters["direction_names"])
    # rows the final candidate relabelled as stop are small_motion in frames.csv; their path is under 1.5 cm anyway
    test_frames["is_path_row"] = (test_frames["action"] != "stop") & (test_frames["path_length_cm"] >= parameters["path_minimum_cm"])
    fold_measures = json.loads(fold_path.read_text(encoding = "utf-8"))["measures"]
    return {"rows": test_frames, "measures": fold_measures}



def direction_part(table, angle_column, parameters):
    """
    Return the direction measures of one fold for one predicted angle column.
    """

    rows = table["rows"]
    step_error = angle_difference_deg(rows["move_angle_deg"], rows[angle_column])
    path_error = angle_difference_deg(rows["path_angle_deg"], rows[angle_column])
    forward_step_error = angle_difference_deg(rows["move_angle_deg"], 0.0)
    forward_path_error = angle_difference_deg(rows["path_angle_deg"], 0.0)
    is_moving = rows["is_moving"].to_numpy()
    is_path_row = rows["is_path_row"].to_numpy()
    is_backward = rows["action"].isin(parameters["backward_names"]).to_numpy()
    is_after_correction = (rows["corrections_before"] > 0).to_numpy()
    is_last_3_cm = (rows["goal_distance_cm"].fillna(0.0) <= 3.0).to_numpy()
    result = {
        "moving_rows": int(is_moving.sum()),
        "step_within_45": round(float((step_error[is_moving] <= 45).mean()), 4),
        "path_rows": int(is_path_row.sum()),
        "path_within_45": round(float((path_error[is_path_row] <= 45).mean()), 4),
        "always_forward_step_within_45": round(float((forward_step_error[is_moving] <= 45).mean()), 4),
        "always_forward_path_within_45": round(float((forward_path_error[is_path_row] <= 45).mean()), 4),
    }
    for label, mask in (("backward", is_moving & is_backward), ("after_correction", is_moving & is_after_correction),
                        ("before_any_correction", is_moving & ~is_after_correction), ("last_3_cm", is_moving & is_last_3_cm)):
        result[f"{label}_rows"] = int(mask.sum())
        result[f"{label}_within_45_hits"] = int((step_error[mask] <= 45).sum())
        result[f"{label}_always_forward_hits"] = int((forward_step_error[mask] <= 45).sum())
    for label, mask in (("after_correction_path", is_path_row & is_after_correction), ("before_any_correction_path", is_path_row & ~is_after_correction)):
        result[f"{label}_rows"] = int(mask.sum())
        result[f"{label}_within_45_hits"] = int((path_error[mask] <= 45).sum())
    return result, path_error



def episode_outcomes(table, threshold, consecutive_pictures, parameters):
    """
    Return the outcome per episode of one fold under one stop rule, as a dict episode -> (outcome, position).
    """

    rows = table["rows"]
    outcomes = {}
    for episode_id, group in rows.groupby("episode_id", sort = False):
        outcomes[episode_id] = stop_outcome(group["stop_probability"].to_numpy(), group["goal_distance_cm"].fillna(0.0).to_numpy(),
                                            threshold, consecutive_pictures, parameters)
    return outcomes



def counted(outcomes):
    """
    Return the counts of right, early, near and never over a dict of episode outcomes.
    """

    values = [outcome for outcome, _ in outcomes.values()]
    return {"episodes": len(values), "right": values.count("right"), "early": values.count("early"),
            "near": values.count("near"), "never": values.count("never")}



def chosen_rule(grid_by_sheet, other_sheets, criterion, allowed_consecutive):
    """
    Return the (threshold, pictures in a row) that scores best on the other sheets by the criterion.
    """

    best_rule, best_key = None, None
    for rule in grid_by_sheet[other_sheets[0]]:
        if rule[1] not in allowed_consecutive:
            continue
        right = sum(grid_by_sheet[sheet][rule]["right"] for sheet in other_sheets)
        early = sum(grid_by_sheet[sheet][rule]["early"] for sheet in other_sheets)
        if criterion == "most_right_then_fewer_early":
            key = (right, -early, rule[0], -rule[1])
        else:
            key = (right - early, right, rule[0], -rule[1])
        if best_key is None or key > best_key:
            best_rule, best_key = rule, key
    return best_rule



def longest_bad_streak(path_error, is_path_row, end_position):
    """
    Return the longest run of consecutive path rows more than 45 degrees off, up to the stopping position.
    """

    errors = path_error[: end_position + 1] if end_position is not None else path_error
    mask = is_path_row[: len(errors)]
    longest = current = 0
    for error in errors[mask]:
        if error > 45.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest



def main(parameters = None):
    """
    Print and save every part of the re-scoring.
    """

    settings = {**PARAMETERS, **(parameters or {})}
    frames = loaded_frames(settings)
    lines = []
    output = {"runs": {}}

    def say(text = ""):
        print(text, flush = True)
        lines.append(text)

    rules = [(threshold, consecutive) for threshold in settings["thresholds"] for consecutive in (1, 2)]
    for run_name, run_settings in settings["runs"].items():
        tables = {}
        for sheet in settings["sheets"]:
            table = fold_table(run_name, run_settings, sheet, frames, settings)
            if table is not None:
                tables[sheet] = table
        if not tables:
            continue
        run_output = {"sheets": list(tables), "direction": {}, "stop": {}}
        say(f"=== {run_name} ({run_settings['folder']}): held-out sheets {', '.join(tables)} ===")

        # 1. direction, reproduced and stratified
        path_errors = {}
        angle_columns = ["angle_deg"] + (["angle_expected_deg"] if run_settings["kind"] == "classes" else []) + (["angle_vector_deg"] if run_settings["kind"] == "chunk" else [])
        for angle_column in angle_columns:
            pooled = {}
            for sheet, table in tables.items():
                result, path_error = direction_part(table, angle_column, settings)
                if angle_column == "angle_deg":
                    path_errors[sheet] = path_error
                prefix = {"angle_deg": "", "angle_expected_deg": "expected_", "angle_vector_deg": "vector_"}[angle_column]
                stored_step = table["measures"].get(f"{prefix}within_45_deg")
                stored_path = table["measures"].get(f"path_{prefix}within_45_deg")
                say(f"  [{angle_column}] sheet {sheet}: step 45 {result['step_within_45']:.4f} of {result['moving_rows']} (fold file {stored_step}), "
                    f"path 45 {result['path_within_45']:.4f} of {result['path_rows']} (fold file {stored_path}); always forward "
                    f"{result['always_forward_step_within_45']:.3f} / {result['always_forward_path_within_45']:.3f}; backward "
                    f"{result['backward_within_45_hits']} of {result['backward_rows']}; after a correction "
                    f"{result['after_correction_within_45_hits']} of {result['after_correction_rows']} (path "
                    f"{result['after_correction_path_within_45_hits']} of {result['after_correction_path_rows']}); before any "
                    f"{result['before_any_correction_within_45_hits']} of {result['before_any_correction_rows']}; last 3 cm "
                    f"{result['last_3_cm_within_45_hits']} of {result['last_3_cm_rows']} (always forward {result['last_3_cm_always_forward_hits']})")
                run_output["direction"].setdefault(angle_column, {})[sheet] = result
                for key, value in result.items():
                    if key.endswith("_rows") or key.endswith("_hits"):
                        pooled[key] = pooled.get(key, 0) + value
            say(f"  [{angle_column}] pooled: backward {pooled['backward_within_45_hits']} of {pooled['backward_rows']} "
                f"(always forward {pooled['backward_always_forward_hits']}); after a correction {pooled['after_correction_within_45_hits']} of "
                f"{pooled['after_correction_rows']} (always forward {pooled['after_correction_always_forward_hits']}); before any correction "
                f"{pooled['before_any_correction_within_45_hits']} of {pooled['before_any_correction_rows']}; last 3 cm "
                f"{pooled['last_3_cm_within_45_hits']} of {pooled['last_3_cm_rows']} (always forward {pooled['last_3_cm_always_forward_hits']})")
            run_output["direction"].setdefault(angle_column, {})["pooled"] = pooled

        # 2. stop rules
        if not all("stop_probability" in table["rows"] for table in tables.values()):
            output["runs"][run_name] = run_output
            say()
            continue
        grid = {sheet: {rule: counted(episode_outcomes(table, rule[0], rule[1], settings)) for rule in rules}
                for sheet, table in tables.items()}
        named_rules = {}
        for sheet, table in tables.items():
            measures = table["measures"]
            named_rules.setdefault("fixed 0.5, one picture", {})[sheet] = (0.5, 1)
            named_rules.setdefault("fixed 0.5, two in a row", {})[sheet] = (0.5, 2)
            named_rules.setdefault("fixed 0.3, two in a row", {})[sheet] = (0.3, 2)
            if "tuned_stop_threshold" in measures:
                named_rules.setdefault("inner validation, one picture", {})[sheet] = (measures["tuned_stop_threshold"], 1)
                named_rules.setdefault("inner validation, two in a row", {})[sheet] = (measures["tuned_two_stop_threshold"], 2)
            if len(tables) >= 2:
                others = [other for other in tables if other != sheet]
                named_rules.setdefault("other sheets, one picture, most right", {})[sheet] = chosen_rule(grid, others, "most_right_then_fewer_early", (1,))
                named_rules.setdefault("other sheets, two in a row, most right", {})[sheet] = chosen_rule(grid, others, "most_right_then_fewer_early", (2,))
                named_rules.setdefault("other sheets, either, right minus early", {})[sheet] = chosen_rule(grid, others, "right_minus_early", (1, 2))
        for rule_name, rule_per_sheet in named_rules.items():
            total = {"episodes": 0, "right": 0, "early": 0, "near": 0, "never": 0}
            parts = []
            for sheet, rule in rule_per_sheet.items():
                rule = (float(rule[0]), int(rule[1]))
                counts = grid[sheet].get(rule) or counted(episode_outcomes(tables[sheet], rule[0], rule[1], settings))
                for key in total:
                    total[key] += counts[key]
                parts.append(f"{sheet}: {counts['right']}/{counts['early']}/{counts['never']} of {counts['episodes']} at {rule}")
            right_interval = wilson_interval(total["right"], total["episodes"])
            early_interval = wilson_interval(total["early"], total["episodes"])
            say(f"  stop [{rule_name}] pooled right {total['right']} early {total['early']} near {total['near']} never {total['never']} of "
                f"{total['episodes']} (right {right_interval}, early {early_interval}) | " + "; ".join(parts))
            run_output["stop"][rule_name] = {"pooled": total, "per_sheet": {sheet: [float(rule[0]), int(rule[1])] for sheet, rule in rule_per_sheet.items()}}
        best_pooled = max(rules, key = lambda rule: (sum(grid[sheet][rule]["right"] for sheet in tables), -sum(grid[sheet][rule]["early"] for sheet in tables)))
        say(f"  stop [one rule for all sheets, chosen with hindsight] {best_pooled}: right {sum(grid[s][best_pooled]['right'] for s in tables)} "
            f"early {sum(grid[s][best_pooled]['early'] for s in tables)} never {sum(grid[s][best_pooled]['never'] for s in tables)}")

        # 3. by finishing end, and 4. the finishing proxy, under the leave-one-sheet-out one-picture rule (or 0.5 if one fold)
        proxy_rule_name = "other sheets, one picture, most right" if "other sheets, one picture, most right" in named_rules else "fixed 0.5, one picture"
        end_counts = {}
        proxy = {"episodes": 0, "right": 0, "right_and_no_streak": 0, "no_streak_whole_episode": 0, "streak_lengths": []}
        early_places = []
        for sheet, table in tables.items():
            rule = named_rules[proxy_rule_name][sheet]
            outcomes = episode_outcomes(table, float(rule[0]), int(rule[1]), settings)
            rows = table["rows"]
            path_error = path_errors[sheet]
            for episode_id, (outcome, position) in outcomes.items():
                episode_mask = (rows["episode_id"] == episode_id).to_numpy()
                episode_rows = rows[episode_mask]
                finish_end = episode_rows["finish_end"].iloc[0]
                key = (episode_rows["sheet"].iloc[0], finish_end)
                end_counts.setdefault(key, {"right": 0, "early": 0, "near": 0, "never": 0})[outcome] += 1
                episode_errors = path_error[episode_mask]
                episode_is_path = episode_rows["is_path_row"].to_numpy()
                whole = longest_bad_streak(episode_errors, episode_is_path, None)
                before_stop = longest_bad_streak(episode_errors, episode_is_path, position)
                proxy["episodes"] += 1
                proxy["streak_lengths"].append(whole)
                proxy["no_streak_whole_episode"] += int(whole < settings["streak_length"])
                if outcome == "right":
                    proxy["right"] += 1
                    proxy["right_and_no_streak"] += int(before_stop < settings["streak_length"])
                if outcome == "early":
                    goal_distance = float(episode_rows["goal_distance_cm"].fillna(0.0).iloc[position])
                    early_places.append((sheet, round(goal_distance, 1), round((position + 1) / len(episode_rows), 2)))
        say(f"  by finishing end [{proxy_rule_name}]: " + "; ".join(
            f"{sheet[-2:]} end {end} ({settings['edge_distance_cm'].get((sheet, end))} cm from the edge): right {c['right']} early {c['early']} never {c['never']} near {c['near']}"
            for (sheet, end), c in sorted(end_counts.items())))
        far = {"right": 0, "early": 0, "never": 0, "near": 0}
        near_edge = {"right": 0, "early": 0, "never": 0, "near": 0}
        for (sheet, end), c in end_counts.items():
            target = far if settings["edge_distance_cm"].get((sheet, end), 0.0) >= 5.0 else near_edge
            for key in target:
                target[key] += c[key]
        say(f"  line ends 5 cm or more from the edge: {far}; under 5 cm: {near_edge}")
        if early_places:
            distances = [place[1] for place in early_places]
            shares = [place[2] for place in early_places]
            say(f"  early stops: {len(early_places)}; distance to the goal median {numpy.median(distances):.1f} cm (range {min(distances)} to {max(distances)}); "
                f"more than 8 cm: {sum(distance > 8.0 for distance in distances)}; share of the episode driven median {numpy.median(shares):.2f}")
        say(f"  finishing proxy [{proxy_rule_name}]: right {proxy['right']} of {proxy['episodes']}; right with no streak of "
            f"{settings['streak_length']}+ path errors over 45 deg before the stop {proxy['right_and_no_streak']}; episodes without such a streak "
            f"anywhere {proxy['no_streak_whole_episode']}; longest streak median {numpy.median(proxy['streak_lengths']):.1f}")
        run_output["by_end"] = {f"{sheet}_{end}": counts for (sheet, end), counts in end_counts.items()}
        run_output["proxy"] = {key: value for key, value in proxy.items() if key != "streak_lengths"}
        run_output["early_places"] = early_places
        output["runs"][run_name] = run_output
        say()

    (settings["output_folder"] / "judge_rescoring.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    (settings["output_folder"] / "judge_rescoring.out.json").write_text(json.dumps(output, indent = 1, default = str), encoding = "utf-8")



if __name__ == "__main__":
    main()
