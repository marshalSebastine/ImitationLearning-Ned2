"""
Verifier check 9: smaller claims, each recomputed my own way.

(a) dash and gap lengths of sheet 06 from my top views of run 8 and of demonstration 06-15 (started at the other end);
(b) the tightest bends of the six course sheets from the Recording session's course-photo centre lines (circle through
    three points 1 cm apart along the line, and the same after smoothing over 1.5 cm), against the practice tracks'
    settings in config.TEST_TRACK_PARAMETERS;
(c) per sheet: share of demonstrated steps more than 60 degrees away from "forward in the picture", and the wrist;
(d) how well the fold models predict the lean change (the output the driver ignores);
(e) the fold model that never saw sheet 06 on run 8's pictures: stop chance and direction;
(f) the demonstrations' step direction against the line's direction near the place of run 8's steps 19 to 30.

Run:  D:/AI_Project/.venv/Scripts/python.exe v09_sheet_and_data_checks.py > v09_sheet_and_data_checks.out.txt
About one minute. Nothing connects to the robot, nothing is trained.
"""

import json

import cv2
import numpy
import pandas
import torch

from verify_common import (ROOT_FOLDER, VERIFY_FOLDER, VERIFY_PARAMETERS, along_and_beside, colour_masks, crossing_indices, ink_cells,
                           pixels_on_table, puck_pixel, register_line, sheet_line, top_view)

from config import TEST_TRACK_PARAMETERS  # noqa: E402
from model_mode import ModelPolicy  # noqa: E402
from training_targets import load_training_frames  # noqa: E402

torch.set_num_threads(2)
cv2.setNumThreads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
RUNS_FOLDER = ROOT_FOLDER / "04_runs" / "gpu_ki014"


def wrapped(angle):

    return (numpy.asarray(angle) + 180.0) % 360.0 - 180.0


def runs_of(flags: numpy.ndarray) -> list:
    """Return (value, first, length) of every run in a boolean sequence."""

    edges = numpy.flatnonzero(numpy.diff(flags.astype(int)) != 0) + 1
    starts = numpy.r_[0, edges]
    ends = numpy.r_[edges, len(flags)]
    return [(bool(flags[start]), int(start), int(end - start)) for start, end in zip(starts, ends)]


def dash_lengths(view: dict, line: numpy.ndarray, name: str) -> None:
    """Print the runs of ink and of no ink along a placed line (1 mm steps), from a top view."""

    is_ink = ink_cells(view)
    is_ink = cv2.dilate(is_ink.astype(numpy.uint8), numpy.ones((3, 3), numpy.uint8)) > 0  # a line point 2 mm beside the ink still counts
    cell = VERIFY_PARAMETERS["map_cell_m"]
    x_index = numpy.clip(numpy.floor((line[:, 0] - view["x_m"][0]) / cell).astype(int), 0, is_ink.shape[0] - 1)
    y_index = numpy.clip(numpy.floor((line[:, 1] - view["y_m"][0]) / cell).astype(int), 0, is_ink.shape[1] - 1)
    on_ink = is_ink[x_index, y_index]
    is_seen = view["seen"][x_index, y_index] >= 2
    crossing = crossing_indices(line)
    runs = runs_of(on_ink)
    dashes, gaps = [], []
    for value, start, length in runs[1:-1]:
        inside = slice(start, start + length)
        if not is_seen[inside].all():
            continue
        # leave out runs that touch the crossing, where the other pass's ink confuses the count
        if any(abs(start + offset - place) < 12 for place in crossing for offset in (0, length)):
            continue
        (dashes if value else gaps).append(0.1 * length)
    print(f"  {name}: {len(dashes)} dashes: median {numpy.median(dashes):.1f} cm (range {min(dashes):.1f} to {max(dashes):.1f});"
          f" {len(gaps)} gaps: median {numpy.median(gaps):.1f} cm (range {min(gaps):.1f} to {max(gaps):.1f});"
          f" ink on {100 * on_ink[is_seen].mean():.0f} % of the seen line; dash plus gap {numpy.median(dashes) + numpy.median(gaps):.1f} cm")
    print(f"    dashes: {[round(value, 1) for value in dashes]}")
    print(f"    gaps:   {[round(value, 1) for value in gaps]}")


def smallest_radius(line_cm: numpy.ndarray, chord_cm: float, smooth_cm: float) -> tuple:
    """Return the smallest circle radius through points ``chord_cm`` apart along a line sampled every millimetre, and where."""

    if smooth_cm > 0:
        window = int(round(smooth_cm / 0.1)) | 1
        kernel = numpy.ones(window) / window
        padded = numpy.pad(line_cm, ((window // 2, window // 2), (0, 0)), mode = "edge")
        line_cm = numpy.column_stack([numpy.convolve(padded[:, 0], kernel, mode = "valid"), numpy.convolve(padded[:, 1], kernel, mode = "valid")])
    step = int(round(chord_cm / 0.1))
    first, middle, last = line_cm[:-2 * step], line_cm[step:-step], line_cm[2 * step:]
    side_a = numpy.hypot(*(middle - first).T)
    side_b = numpy.hypot(*(last - middle).T)
    side_c = numpy.hypot(*(last - first).T)
    area = 0.5 * numpy.abs((middle[:, 0] - first[:, 0]) * (last[:, 1] - first[:, 1]) - (last[:, 0] - first[:, 0]) * (middle[:, 1] - first[:, 1]))
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        radius = side_a * side_b * side_c / (4.0 * area)
    radius[area < 1e-9] = numpy.inf
    place = int(numpy.argmin(radius))
    return float(radius[place]), 0.1 * (place + step), radius


def main() -> None:

    # ---------------------------------------------------------------- (a) dashes
    print("(a) Dashes and gaps of sheet 06, measured along the course-photo centre line placed on my top views")
    folder = VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS["run_8_folder"]
    steps = pandas.read_csv(folder / "steps.csv")
    run_pictures = [str(folder / name) for name in steps["frame"]]
    run_joints = steps[JOINT_NAMES].to_numpy()
    low_twist = numpy.flatnonzero(numpy.abs(numpy.degrees(run_joints[:, 3])) < 45.0)
    view = top_view([run_pictures[position] for position in low_twist], run_joints[low_twist])
    placed = register_line(sheet_line("06"), view)
    dash_lengths(view, placed["line_m"], "run 8, pictures with the forearm under 45 degrees")
    frames_all = pandas.read_csv(VERIFY_PARAMETERS["index_folder"] / "frames.csv", low_memory = False)
    for plan in ("06-15", "06-03"):
        part = frames_all[frames_all["plan_number"] == plan].sort_values("row_in_episode")
        joints = part[JOINT_NAMES].to_numpy()
        chosen = numpy.flatnonzero(numpy.abs(numpy.degrees(joints[:, 3])) < 45.0)
        demo_view = top_view([str(VERIFY_PARAMETERS["raw_folder"] / part["frame_path"].iloc[position]) for position in chosen], joints[chosen])
        demo_placed = register_line(sheet_line("06"), demo_view)
        dash_lengths(demo_view, demo_placed["line_m"], f"demonstration {plan}, forearm under 45 degrees")
    print(f"  practice tracks (config.TEST_TRACK_PARAMETERS): dash {TEST_TRACK_PARAMETERS['dash_mm']} mm, gap {TEST_TRACK_PARAMETERS['gap_mm']} mm,"
          f" smallest radius {TEST_TRACK_PARAMETERS['smallest_radius_mm']} mm, line width {TEST_TRACK_PARAMETERS['line_width_mm']} mm")

    # ---------------------------------------------------------------- (b) tight bends
    print("\n(b) Tightest bend of each course sheet's centre line (course photos, Recording session's file), radius in cm")
    measurements = json.loads(VERIFY_PARAMETERS["sheet_measurements_path"].read_text())
    for sheet in ("01", "02", "03", "04", "05", "06"):
        line_cm = 100.0 * sheet_line(sheet)
        raw_radius, raw_place, _ = smallest_radius(line_cm, 1.0, 0.0)
        smooth_radius, smooth_place, radius = smallest_radius(line_cm, 1.0, 1.5)
        tight = 0.1 * numpy.sum(radius < 3.5)
        print(f"  sheet {sheet}: length {measurements[sheet]['line_length_cap_to_cap_cm']} cm; smallest radius {raw_radius:.1f} cm at {raw_place:.1f} cm"
              f" (unsmoothed, points 1 cm apart), {smooth_radius:.1f} cm at {smooth_place:.1f} cm (smoothed over 1.5 cm);"
              f" {tight:.1f} cm of the line bend tighter than 3.5 cm radius (smoothed)")

    # ---------------------------------------------------------------- (c) per sheet
    print("\n(c) Per sheet, cleaned training pictures: steps away from forward, and the wrist (the driver holds 13.5 deg)")
    frames = load_training_frames({"is_audit_cleaning": True})
    frames["step_length_cm"] = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    frames["step_angle"] = numpy.degrees(numpy.arctan2(frames["tip_left_m"].fillna(0.0), frames["tip_forward_m"].fillna(0.0)))
    frames["wrist_deg"] = numpy.degrees(frames["joint_5"])
    moving = frames[(frames["action"] != "stop") & (frames["step_length_cm"] >= 0.3)]
    table = pandas.DataFrame({
        "pictures": frames.groupby("sheet").size(),
        "moving_steps": moving.groupby("sheet").size(),
        "over_60_from_forward": moving.groupby("sheet")["step_angle"].apply(lambda values: (values.abs() > 60).mean()),
        "over_90_backward": moving.groupby("sheet")["step_angle"].apply(lambda values: (values.abs() > 90).mean()),
        "wrist_over_5_from_13.5": frames.groupby("sheet")["wrist_deg"].apply(lambda values: ((values - 13.5).abs() > 5).mean()),
        "wrist_median": frames.groupby("sheet")["wrist_deg"].median(),
        "wrist_90th": frames.groupby("sheet")["wrist_deg"].quantile(0.9),
        "step_median_cm": frames[frames["action"] != "stop"].groupby("sheet")["step_length_cm"].median(),
        "steps_under_0.5_cm": frames[frames["action"] != "stop"].groupby("sheet")["step_length_cm"].apply(lambda values: (values < 0.5).mean())
    })
    print(table.round(3).to_string())

    # ---------------------------------------------------------------- (d) lean output
    print("\n(d) The lean output of the fold models on their held-out sheet (rows with a demonstrated lean change of 1 degree or more)")
    for label, run_folder in (("E recipe", "2026-11-10_resnet18_chunk_stop_weight_25"), ("A recipe", "2026-11-09_resnet18_final_candidate")):
        for sheet in ("03", "04", "05", "06"):
            saved = numpy.load(RUNS_FOLDER / run_folder / f"predictions_without_path_{sheet}.npz")
            predictions = pandas.DataFrame({"frame_index": saved["frame_index"], "lean_deg": saved["lean_deg"]})
            merged = frames[frames["sheet"] == f"path_{sheet}"].merge(predictions, on = "frame_index")
            true = merged["lean_change_deg"].fillna(0.0)
            has_lean = true.abs() >= 1.0
            correlation = numpy.corrcoef(true, merged["lean_deg"])[0, 1]
            print(f"  {label}, sheet {sheet}: {int(has_lean.sum())} of {len(merged)} rows change the lean by 1 deg or more; right sign {(numpy.sign(merged.loc[has_lean, 'lean_deg']) == numpy.sign(true[has_lean])).mean():.2f};"
                  f" asked size {merged.loc[has_lean, 'lean_deg'].abs().mean() / true[has_lean].abs().mean():.2f} of the demonstrated; correlation over all rows {correlation:.2f};"
                  f" summed over an episode: demonstrated {true.groupby(merged['episode_id']).sum().abs().median():.1f} deg, asked {merged['lean_deg'].groupby(merged['episode_id']).sum().abs().median():.1f} deg (medians of absolute sums)")

    # ---------------------------------------------------------------- (e) fold model on run 8
    print("\n(e) Run 8's recorded pictures answered by models that never saw sheet 06 (E recipe folds) and by model E")
    line = placed["line_m"]
    puck_xy = numpy.full((len(steps), 2), numpy.nan)
    for position, picture_path in enumerate(run_pictures):
        found = puck_pixel(colour_masks(cv2.imread(picture_path))["puck"])
        if found is not None:
            puck_xy[position] = pixels_on_table(numpy.array([[found[0], found[1]]]), run_joints[position], VERIFY_PARAMETERS["puck_point_height_m"])[0]
    index, beside = along_and_beside(line, puck_xy)
    if index[-1] < index[0]:
        line = line[::-1].copy()
        index, beside = along_and_beside(line, puck_xy)
    tangent = numpy.gradient(line, axis = 0)
    line_direction = numpy.degrees(numpy.arctan2(tangent[index, 1], tangent[index, 0]))
    answers = {"step": steps["step"].to_numpy(), "beside_cm": 100.0 * beside, "E_angle": steps["model_angle_deg"].to_numpy(),
               "E_twist": steps["model_twist_deg"].to_numpy(), "E_stop": steps["model_stop_probability"].to_numpy(),
               "E_vs_line": wrapped(steps["picture_heading_deg"].to_numpy() + steps["model_angle_deg"].to_numpy() - line_direction)}
    for label, run_folder in (("fold", "2026-11-10_resnet18_chunk_stop_weight_25"), ("fold2", "2026-11-10_resnet18_chunk_stop_weight_25_seed_2")):
        policy = ModelPolicy("resnet18_chunk_stop_weight_25", RUNS_FOLDER / run_folder / "checkpoints" / "without_path_06.pt", RUNS_FOLDER / run_folder / "metrics.json")
        torch.set_num_threads(2)
        decided = [policy.decide(open(picture_path, "rb").read()) for picture_path in run_pictures]
        answers[f"{label}_angle"] = [value["angle_deg"] for value in decided]
        answers[f"{label}_twist"] = [value["twist_deg"] for value in decided]
        answers[f"{label}_length"] = [numpy.hypot(value["tip_forward_cm"], value["tip_left_cm"]) for value in decided]
        answers[f"{label}_stop"] = [value["stop_probability"] for value in decided]
        answers[f"{label}_vs_line"] = wrapped(steps["picture_heading_deg"].to_numpy() + numpy.array(answers[f"{label}_angle"]) - line_direction)
    answers = pandas.DataFrame(answers)
    print(answers.round(2).to_string(index = False))
    answers.round(3).to_csv(VERIFY_FOLDER / "v09_fold_models_on_run_8.csv", index = False)

    # ---------------------------------------------------------------- (f) demonstrations near steps 19 to 30
    print("\n(f) Step direction against the line's direction (+ = outside the loop): run 8's asked steps and the demonstrations' steps within 0.75 cm along the line")
    demos = pandas.read_csv(VERIFY_FOLDER / "v05_demonstration_tracks_06.csv")
    demos = demos[(demos["travel_from"] == "first_end") & (demos["plan"] <= "06-13")].merge(
        frames_all[["frame_index", "picture_heading_deg"]], on = "frame_index").reset_index(drop = True)
    run_tracks = pandas.read_csv(VERIFY_FOLDER / "v04_run_tracks.csv")
    run_tracks = run_tracks[run_tracks["run"] == "run_8"].set_index("step")
    # every demonstration uses the same centre line, only turned and shifted on the table: the line's direction at a
    # demonstration picture's place is the run's line direction at the same place along the line plus the difference
    # of the two placement angles (read from v05's printed table)
    crossing = crossing_indices(line)
    demo_index = numpy.clip(numpy.round(demos["along_cm"].fillna(0.0) * 10.0).astype(int) + crossing[0], 0, len(line) - 1).to_numpy()
    run_line_direction = numpy.degrees(numpy.arctan2(tangent[:, 1], tangent[:, 0]))
    placements = {}
    for text_line in (VERIFY_FOLDER / "v05_demonstration_tracks_06.out.txt").read_text().splitlines():
        parts = text_line.split()
        if len(parts) > 6 and parts[0].startswith("06-"):
            placements[parts[0]] = float(parts[4])
    run_angle = numpy.degrees(placed["placement"][0]) % 360.0
    demos["line_direction"] = run_line_direction[demo_index] + demos["plan"].map(placements).to_numpy() - run_angle
    demos["labelled_direction"] = demos["picture_heading_deg"] + demos["move_angle_deg"]
    demos["step_vs_line"] = wrapped(demos["labelled_direction"] - demos["line_direction"])
    print(f"  (run 8's line placed at {run_angle:.1f} deg; the demonstrations' at {min(placements.values()):.1f} to {max(placements.values()):.1f} deg)")
    table = []
    for step in range(16, 34):
        along = run_tracks.loc[step, "along_cm"]
        near = demos[((demos["along_cm"] - along).abs() <= 0.75) & (demos["move_length_cm"] >= 0.3)]
        table.append({"step": step, "along_cm": along, "run_asked_vs_line": answers.loc[step, "E_vs_line"], "demo_steps": len(near),
                      "demo_median_vs_line": near["step_vs_line"].median(), "demo_25": near["step_vs_line"].quantile(0.25), "demo_75": near["step_vs_line"].quantile(0.75)})
    print(pandas.DataFrame(table).round(1).to_string(index = False))



if __name__ == "__main__":
    main()
