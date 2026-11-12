"""
Would another wrist rule bring the robot's wrist closer to the demonstrated one? (key: wrist_and_camera, part 3)

Four rules are compared on every demonstration row, each against the demonstrated wrist (joint 5):
    constant            the wrist held at 13.5 degrees (the driver of 11.11.26)
    fitted function     the wrist from the reach and the forearm twist, fitted on the other five sheets
                        (a table of medians, and a least-squares curve)
    follow the lean     the wrist that gives the rod the lean change the model predicts, step after step, along the
                        demonstrated tip path and twist; the predictions are those of the fold model that never saw
                        the sheet (saved by the training runs), the run starts at 13.5 degrees as the driver does
    follow the labels   the same with the demonstrated lean changes: a check of the simulation, and a measure of how
                        far the wrist drifts when even perfect lean changes are followed step after step

It also measures what the model's lean output knows: the demonstrated lean change is split into the part that a held
wrist gives anyway when the tip moves and the forearm twists (kinematics) and the rest, which needs the wrist to move.

Calculated from logged joints and saved predictions; nothing is measured on the robot here. Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe wrist_options.py
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import wrist_geometry


OPTION_PARAMETERS = {
    "prediction_runs": {
        "E recipe (chunk, stop weight 25)": wrist_geometry.ROOT_FOLDER / "04_runs" / "gpu_ki014" / "2026-11-10_resnet18_chunk_stop_weight_25",
        "A recipe (exact step)": wrist_geometry.ROOT_FOLDER / "04_runs" / "gpu_ki014" / "2026-11-09_resnet18_final_candidate"
    },
    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"],
    # a wrist within this many degrees keeps the view within the training shift augmentation (16 px at 640, 2.80 px per degree)
    "wrist_range_deg": 16.0 / 2.803,
    "pixels_per_wrist_degree": 2.803,
    "wrist_search_deg": [-60.0, 80.0],
    "secant_iterations": 8,
    "reach_bin_m": 0.04,
    "twist_bin_deg": 30.0,
    "fewest_rows_per_bin": 30,
    "rod_lean_most_deg": 58.0,
    "figure_dpi": 120
}



def load_predicted_leans(
        run_folder: Path,
        parameters: Dict
) -> pandas.Series:
    """The lean change (degrees per step) each fold model predicted for the rows of the sheet it never saw, by frame index."""

    parts = []
    for sheet in parameters["sheets"]:
        saved = numpy.load(Path(run_folder) / f"predictions_without_{sheet}.npz")
        parts.append(pandas.Series(saved["lean_deg"].astype(float), index = saved["frame_index"].astype(int)))
    return pandas.concat(parts)



def joints_for_wrist(
        tips: numpy.ndarray,
        twists_rad: numpy.ndarray,
        wrists_rad: numpy.ndarray,
        starts: numpy.ndarray
) -> numpy.ndarray:
    """The joints that put the tip on its target for the given twist and wrist (the hand at zero: it changes nothing here)."""

    return wrist_geometry.solve_many_with_wrist_fixed(tips, twists_rad, wrists_rad, starts)[0]



def wrist_for_forward_lean(
        tips: numpy.ndarray,
        twists_rad: numpy.ndarray,
        headings: numpy.ndarray,
        target_leans_deg: numpy.ndarray,
        first_wrists_rad: numpy.ndarray,
        starts: numpy.ndarray,
        parameters: Dict
) -> numpy.ndarray:
    """
    The wrist angle that gives the rod the wanted lean towards the picture's top, with the tip on its target and the
    forearm twisted as given: a secant search on the wrist, row by row, kept within "wrist_search_deg".

    Returns
    -------
    numpy.ndarray
        The wrist of every row in radians.
    """

    lowest, highest = (math.radians(value) for value in parameters["wrist_search_deg"])

    def lean_error(wrists_rad: numpy.ndarray) -> numpy.ndarray:
        joints = joints_for_wrist(tips, twists_rad, wrists_rad, starts)
        return wrist_geometry.forward_leans_deg(joints, headings) - target_leans_deg

    wrists_before = numpy.clip(first_wrists_rad, lowest, highest)
    errors_before = lean_error(wrists_before)
    # one degree of wrist changes the forward lean by about one degree, so the first step uses that slope
    wrists_now = numpy.clip(wrists_before - numpy.radians(errors_before), lowest, highest)
    for _ in range(parameters["secant_iterations"]):
        errors_now = lean_error(wrists_now)
        if numpy.max(numpy.abs(errors_now)) < 1e-4:
            break
        differences = errors_now - errors_before
        slopes = numpy.where(numpy.abs(differences) > 1e-9, (wrists_now - wrists_before) / numpy.where(numpy.abs(differences) > 1e-9, differences, 1.0), math.radians(1.0))
        wrists_next = numpy.clip(wrists_now - errors_now * slopes, lowest, highest)
        wrists_before, errors_before = wrists_now, errors_now
        wrists_now = wrists_next
    return wrists_now



def follow_lean_changes(
        frames: pandas.DataFrame,
        lean_changes: numpy.ndarray,
        first_wrists_rad: numpy.ndarray,
        parameters: Dict
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Carry out lean changes step after step along every demonstration's own tip path and twist.

    All episodes are stepped together: at step t every episode that still has a next row moves its tip to the
    demonstrated next tip, twists the forearm to the demonstrated next twist and sets the wrist so that the rod's lean
    towards the picture's top (as seen at the current joints) changes by the given amount.

    Parameters
    ----------
    frames : pandas.DataFrame
        The demonstration rows in episode order (wrist_geometry.load_demonstration_frames).
    lean_changes : numpy.ndarray
        The lean change to carry out at every row, in degrees; NaN counts as zero (the wrist then moves only as much
        as a constant lean asks for).
    first_wrists_rad : numpy.ndarray
        The wrist at every episode's first row, one value per row of ``frames`` (only the first rows are read).
    parameters : dict
        OPTION_PARAMETERS.

    Returns
    -------
    tuple of numpy.ndarray
        The simulated wrist in degrees and the simulated rod lean from straight down in degrees, one per row.
    """

    joints_demonstrated = frames[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)
    tips = frames[["tip_x_m", "tip_y_m", "tip_z_m"]].to_numpy(dtype = float)
    episode_codes = pandas.factorize(frames["episode_id"])[0]
    row_numbers = frames.groupby("episode_id").cumcount().to_numpy()
    episode_starts = numpy.flatnonzero(row_numbers == 0)
    episode_lengths = numpy.bincount(episode_codes)
    changes = numpy.nan_to_num(numpy.asarray(lean_changes, dtype = float), nan = 0.0)

    wrists = numpy.full(len(frames), numpy.nan)
    leans = numpy.full(len(frames), numpy.nan)
    wrists[episode_starts] = first_wrists_rad[episode_starts]
    current_joints = joints_for_wrist(tips[episode_starts], joints_demonstrated[episode_starts, 3], wrists[episode_starts], joints_demonstrated[episode_starts, :3])
    leans[episode_starts] = wrist_geometry.tips_and_leans(current_joints)[1]
    active = numpy.arange(len(episode_starts))
    for step in range(int(episode_lengths.max()) - 1):
        is_continuing = episode_lengths[active] > step + 1
        active = active[is_continuing]
        current_joints = current_joints[is_continuing]
        if len(active) == 0:
            break
        rows_now = episode_starts[active] + step
        rows_next = rows_now + 1
        headings = wrist_geometry.headings_rad(current_joints)
        targets = wrist_geometry.forward_leans_deg(current_joints, headings) + changes[rows_now]
        next_wrists = wrist_for_forward_lean(tips[rows_next], joints_demonstrated[rows_next, 3], headings, targets, current_joints[:, 4],
                                             current_joints[:, :3], parameters)
        current_joints = joints_for_wrist(tips[rows_next], joints_demonstrated[rows_next, 3], next_wrists, current_joints[:, :3])
        wrists[rows_next] = next_wrists
        leans[rows_next] = wrist_geometry.tips_and_leans(current_joints)[1]
    return numpy.degrees(wrists), leans



def fitted_wrists(
        frames: pandas.DataFrame,
        parameters: Dict
) -> Dict[str, numpy.ndarray]:
    """
    The wrist from the reach and the forearm twist, fitted on five sheets and applied to the sixth, for every sheet.

    Returns
    -------
    dict
        "table of medians" (the median wrist per cell of reach and |twist|; cells with too few rows take the overall
        median) and "least squares" (a curve of second degree in reach and |twist|), one wrist in degrees per row.
    """

    reach = frames["reach_m"].to_numpy(dtype = float)
    twist = frames["joint_4_deg"].abs().to_numpy(dtype = float)
    wrist = frames["joint_5_deg"].to_numpy(dtype = float)
    reach_cells = numpy.floor(reach / parameters["reach_bin_m"]).astype(int)
    twist_cells = numpy.floor(twist / parameters["twist_bin_deg"]).astype(int)
    features = numpy.column_stack([numpy.ones(len(frames)), reach, reach ** 2, twist / 100.0, (twist / 100.0) ** 2, reach * twist / 100.0])
    results = {"table of medians": numpy.full(len(frames), numpy.nan), "least squares": numpy.full(len(frames), numpy.nan)}
    for sheet in parameters["sheets"]:
        is_test = (frames["sheet"] == sheet).to_numpy()
        is_train = ~is_test
        table = pandas.DataFrame({"reach_cell": reach_cells[is_train], "twist_cell": twist_cells[is_train], "wrist": wrist[is_train]})
        cells = table.groupby(["reach_cell", "twist_cell"])["wrist"].agg(["median", "size"])
        cells = cells[cells["size"] >= parameters["fewest_rows_per_bin"]]["median"]
        overall = float(numpy.median(wrist[is_train]))
        keys = pandas.MultiIndex.from_arrays([reach_cells[is_test], twist_cells[is_test]])
        results["table of medians"][is_test] = cells.reindex(keys).fillna(overall).to_numpy()
        weights = numpy.linalg.lstsq(features[is_train], wrist[is_train], rcond = None)[0]
        results["least squares"][is_test] = features[is_test] @ weights
    return results



def rule_scores(
        frames: pandas.DataFrame,
        wrists_by_rule: Dict[str, numpy.ndarray],
        masks: Dict[str, numpy.ndarray],
        parameters: Dict
) -> pandas.DataFrame:
    """For every rule and every group of rows: the mean and median distance to the demonstrated wrist and the share within range."""

    demonstrated = frames["joint_5_deg"].to_numpy(dtype = float)
    rows = []
    for rule, wrists in wrists_by_rule.items():
        for group, mask in masks.items():
            is_used = mask & numpy.isfinite(wrists)
            distances = numpy.abs(wrists[is_used] - demonstrated[is_used])
            rows.append({"rule": rule, "rows": group, "count": int(is_used.sum()), "mean_distance_deg": distances.mean(),
                         "median_distance_deg": numpy.median(distances), "p90_distance_deg": numpy.quantile(distances, 0.9),
                         "within_range_share": float((distances <= parameters["wrist_range_deg"]).mean()),
                         "mean_view_shift_640_px": parameters["pixels_per_wrist_degree"] * distances.mean()})
    return pandas.DataFrame(rows)



def main(parameters: Optional[Dict] = None) -> None:
    """Print what the lean output knows and how each wrist rule compares with the demonstrated wrist; save a figure."""

    settings = dict(OPTION_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 200)
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    held_deg = wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]

    frames = wrist_geometry.load_demonstration_frames()
    episodes = pandas.read_csv(wrist_geometry.ANALYSIS_PARAMETERS["episodes_file"])
    on_mark = episodes[episodes["sheet_orientation"].str.contains("START MARK")]["episode_id"]
    is_on_mark = frames["episode_id"].isin(on_mark).to_numpy()
    joints = frames[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)
    tips = frames[["tip_x_m", "tip_y_m", "tip_z_m"]].to_numpy(dtype = float)
    has_next = (frames["episode_id"].shift(-1) == frames["episode_id"]).to_numpy()
    rows_now = numpy.flatnonzero(has_next)
    rows_next = rows_now + 1

    # --- what a held wrist does to the lean anyway, and what is left for the wrist to do ---
    headings = wrist_geometry.headings_rad(joints[rows_now])
    lean_now = wrist_geometry.forward_leans_deg(joints[rows_now], headings)
    lean_next = wrist_geometry.forward_leans_deg(joints[rows_next], headings)
    held_next = joints_for_wrist(tips[rows_next], joints[rows_next, 3], joints[rows_now, 4], joints[rows_now, :3])
    lean_next_held = wrist_geometry.forward_leans_deg(held_next, headings)
    steps = pandas.DataFrame({
        "frame_index": frames["frame_index"].to_numpy()[rows_now],
        "sheet": frames["sheet"].to_numpy()[rows_now],
        "label_deg": frames["lean_change_deg"].to_numpy(dtype = float)[rows_now],
        "demonstrated_deg": lean_next - lean_now,
        "kinematic_deg": lean_next_held - lean_now,
        "wrist_change_deg": numpy.degrees(joints[rows_next, 4] - joints[rows_now, 4])
    })
    steps["wrist_part_deg"] = steps["demonstrated_deg"] - steps["kinematic_deg"]
    label_gap = (steps["label_deg"] - steps["demonstrated_deg"]).abs()
    print(f"steps (a row and its next row in the same episode): {len(steps)}")
    print(f"lean change recomputed against the label column: median gap {label_gap.median():.3f}, 99th percentile {label_gap.quantile(0.99):.3f} degrees")
    print(f"demonstrated lean change: mean {steps['demonstrated_deg'].mean():.3f}, standard deviation {steps['demonstrated_deg'].std():.3f} degrees per step")
    print(f"the part a held wrist gives anyway (kinematics): standard deviation {steps['kinematic_deg'].std():.3f}; "
          f"it explains {100.0 * numpy.corrcoef(steps['kinematic_deg'], steps['demonstrated_deg'])[0, 1] ** 2:.1f} % of the variance of the demonstrated lean change")
    print(f"the part that needs the wrist to move: standard deviation {steps['wrist_part_deg'].std():.3f}; its correlation with the wrist's own change: "
          f"{numpy.corrcoef(steps['wrist_part_deg'], steps['wrist_change_deg'])[0, 1]:.3f}; "
          f"slope {numpy.polyfit(steps['wrist_change_deg'], steps['wrist_part_deg'], 1)[0]:.3f} degrees of lean per degree of wrist")
    is_wrist_step = steps["wrist_change_deg"].abs() > 1.0
    print(f"steps in which the wrist moved more than 1 degree: {int(is_wrist_step.sum())} ({100.0 * is_wrist_step.mean():.1f} %)")

    wrists_by_rule = {"constant 13.5": numpy.full(len(frames), held_deg),
                      "the episode's first-row wrist": numpy.degrees(frames.groupby("episode_id")["joint_5"].transform("first").to_numpy())}
    wrists_by_rule.update({f"fitted on reach and twist ({name})": values for name, values in fitted_wrists(frames, settings).items()})

    first_rows_demonstrated = frames["joint_5"].to_numpy(dtype = float)
    first_rows_held = numpy.full(len(frames), math.radians(held_deg))

    # check of the wrist search on single steps: with the lean of the reached next row as the target it must give the
    # demonstrated next wrist back; with the label (the lean change of the commanded waypoint) a small gap is left
    exact_wrists = wrist_for_forward_lean(tips[rows_next], joints[rows_next, 3], headings, lean_next, joints[rows_now, 4], joints[rows_now, :3], settings)
    exact_gap = numpy.abs(numpy.degrees(exact_wrists - joints[rows_next, 4]))
    label_wrists = wrist_for_forward_lean(tips[rows_next], joints[rows_next, 3], headings, lean_now + frames["lean_change_deg"].fillna(0.0).to_numpy(dtype = float)[rows_now],
                                          joints[rows_now, 4], joints[rows_now, :3], settings)
    label_gap = numpy.degrees(label_wrists - joints[rows_next, 4])
    print(f"\ncheck of the wrist search on single steps: with the reached lean as the target the demonstrated next wrist comes back within {exact_gap.max():.4f} degrees (largest); "
          f"with the label as the target the gap is {numpy.abs(label_gap).mean():.3f} degrees per step on average ({label_gap.mean():+.3f} with its sign)")

    checked_wrists, _ = follow_lean_changes(frames, frames["lean_change_deg"].to_numpy(dtype = float), first_rows_demonstrated, settings)
    check_gap = numpy.abs(checked_wrists - frames["joint_5_deg"].to_numpy())
    print(f"followed step after step along whole episodes (demonstrated lean labels, started at the demonstrated wrist): the wrist ends up within "
          f"{numpy.nanmedian(check_gap):.3f} degrees of the demonstrated one (median over rows), {numpy.nanquantile(check_gap, 0.99):.3f} (99th percentile), {numpy.nanmax(check_gap):.3f} (largest): "
          "the labels describe the commanded waypoints, the next rows the reached joints, and the small gap of every step adds up, so even perfect lean changes drift")
    wrists_by_rule["follow the demonstrated lean labels from 13.5"] = follow_lean_changes(frames, frames["lean_change_deg"].to_numpy(dtype = float), first_rows_held, settings)[0]

    simulated_leans = {}
    for run_name, run_folder in settings["prediction_runs"].items():
        predicted = load_predicted_leans(run_folder, settings)
        frames[f"predicted_lean_{run_name}"] = frames["frame_index"].map(predicted)
        steps["predicted_deg"] = steps["frame_index"].map(predicted)
        scored = steps.dropna(subset = ["predicted_deg"])
        is_clear = scored["demonstrated_deg"].abs() >= 0.5
        wrist_steps = scored[scored["wrist_change_deg"].abs() > 1.0]
        print(f"\n=== the lean output of the fold models, {run_name} ({len(scored)} scored steps on sheets the model never saw) ===")
        print(f"  predicted lean change: mean {scored['predicted_deg'].mean():.3f}, standard deviation {scored['predicted_deg'].std():.3f} degrees per step")
        print(f"  correlation with the demonstrated lean change: {numpy.corrcoef(scored['predicted_deg'], scored['demonstrated_deg'])[0, 1]:.3f}; "
              f"sign right on {100.0 * (numpy.sign(scored.loc[is_clear, 'predicted_deg']) == numpy.sign(scored.loc[is_clear, 'demonstrated_deg'])).mean():.1f} % "
              f"of the {int(is_clear.sum())} steps that lean by 0.5 degrees or more")
        print(f"  correlation with the kinematic part: {numpy.corrcoef(scored['predicted_deg'], scored['kinematic_deg'])[0, 1]:.3f}; "
              f"the kinematic part alone has the right sign on {100.0 * (numpy.sign(scored.loc[is_clear, 'kinematic_deg']) == numpy.sign(scored.loc[is_clear, 'demonstrated_deg'])).mean():.1f} %")
        print(f"  correlation of (predicted minus kinematic) with the part that needs the wrist: "
              f"{numpy.corrcoef(scored['predicted_deg'] - scored['kinematic_deg'], scored['wrist_part_deg'])[0, 1]:.3f}")
        print(f"  on the {len(wrist_steps)} steps in which the wrist moved more than 1 degree: demonstrated wrist part mean |x| "
              f"{wrist_steps['wrist_part_deg'].abs().mean():.2f} degrees; predicted minus kinematic mean |x| {(wrist_steps['predicted_deg'] - wrist_steps['kinematic_deg']).abs().mean():.2f}; "
              f"sign right on {100.0 * (numpy.sign(wrist_steps['predicted_deg'] - wrist_steps['kinematic_deg']) == numpy.sign(wrist_steps['wrist_part_deg'])).mean():.1f} %; "
              f"correlation {numpy.corrcoef(wrist_steps['predicted_deg'] - wrist_steps['kinematic_deg'], wrist_steps['wrist_part_deg'])[0, 1]:.3f}")
        wrists, leans = follow_lean_changes(frames, frames[f"predicted_lean_{run_name}"].to_numpy(dtype = float), first_rows_held, settings)
        wrists_by_rule[f"follow the model's lean output from 13.5, {run_name}"] = wrists
        simulated_leans[run_name] = leans
        last_rows = frames.groupby("episode_id").tail(1).index.to_numpy()
        drift = wrists[last_rows] - held_deg
        print(f"  followed along the demonstrated paths: wrist at the last row minus 13.5: median {numpy.median(drift):.1f}, 5th to 95th percentile "
              f"{numpy.quantile(drift, 0.05):.1f} to {numpy.quantile(drift, 0.95):.1f} degrees; demonstrated: median "
              f"{numpy.median(frames['joint_5_deg'].to_numpy()[last_rows] - held_deg):.1f}, 5th to 95th {numpy.quantile(frames['joint_5_deg'].to_numpy()[last_rows] - held_deg, 0.05):.1f} to "
              f"{numpy.quantile(frames['joint_5_deg'].to_numpy()[last_rows] - held_deg, 0.95):.1f}")
        print(f"  rod lean when followed: 99th percentile {numpy.nanquantile(leans, 0.99):.1f}, largest {numpy.nanmax(leans):.1f} degrees; over the driver's "
              f"{settings['rod_lean_most_deg']:.0f} in {int((leans > settings['rod_lean_most_deg']).sum())} rows of {int((leans > settings['rod_lean_most_deg']).sum() > 0) and frames.loc[leans > settings['rod_lean_most_deg'], 'episode_id'].nunique()} episodes "
              f"(demonstrated: {int((frames['rod_from_vertical_deg'] > settings['rod_lean_most_deg']).sum())} rows)")

    masks = {"all rows": numpy.ones(len(frames), dtype = bool), "start-mark episodes": is_on_mark, "taught-start episodes": ~is_on_mark}
    for sheet in settings["sheets"]:
        masks[f"start-mark, {sheet}"] = is_on_mark & (frames["sheet"] == sheet).to_numpy()
    scores = rule_scores(frames, wrists_by_rule, masks, settings)
    print(f"\n=== each wrist rule against the demonstrated wrist (within range: within {settings['wrist_range_deg']:.1f} degrees, the training shift augmentation) ===")
    for group in ["all rows", "start-mark episodes", "taught-start episodes"]:
        print(f"\n{group}")
        print(scores[scores["rows"] == group].drop(columns = "rows").round(3).to_string(index = False))
    print("\nstart-mark episodes by sheet: mean distance to the demonstrated wrist (degrees)")
    by_sheet = scores[scores["rows"].str.startswith("start-mark, ")].pivot(index = "rule", columns = "rows", values = "mean_distance_deg")
    print(by_sheet.round(2).to_string())
    scores.to_csv(output_folder / "wrist_options_scores.csv", index = False)

    figure, axes = pyplot.subplots(1, 3, figsize = (17, 4.8))
    first_run = list(settings["prediction_runs"])[0]
    scored = steps.dropna(subset = ["predicted_deg"])
    sample = scored.sample(n = min(6000, len(scored)), random_state = wrist_geometry.ANALYSIS_PARAMETERS["check_seed"])
    axes[0].scatter(sample["kinematic_deg"], sample["demonstrated_deg"], s = 3, color = "tab:grey", label = "what a held wrist gives anyway")
    axes[0].scatter(sample["predicted_deg"], sample["demonstrated_deg"], s = 3, color = "tab:blue", alpha = 0.5, label = "the model's lean output (last run listed)")
    axes[0].plot([-6, 6], [-6, 6], color = "black", linewidth = 1)
    axes[0].set_xlim(-6, 6)
    axes[0].set_ylim(-12, 12)
    axes[0].set_xlabel("lean change per step (degrees): kinematic part, or predicted")
    axes[0].set_ylabel("demonstrated lean change (degrees)")
    axes[0].set_title("The lean label is mostly kinematics")
    axes[0].legend(fontsize = 8)
    axes[0].grid(alpha = 0.3)

    example = frames[is_on_mark & (frames["sheet"] == "path_06")]
    example_episode = example.groupby("episode_id")["joint_5_deg"].agg(lambda values: values.max() - values.min()).sort_values().index[len(example["episode_id"].unique()) // 2]
    is_example = (frames["episode_id"] == example_episode).to_numpy()
    axes[1].plot(frames.loc[is_example, "row_in_episode"], frames.loc[is_example, "joint_5_deg"], color = "black", label = "demonstrated wrist")
    axes[1].axhline(held_deg, color = "tab:red", label = "constant 13.5")
    for rule, colour in [(f"follow the model's lean output from 13.5, {first_run}", "tab:blue"), ("follow the demonstrated lean labels from 13.5", "tab:green"),
                         ("fitted on reach and twist (table of medians)", "tab:orange")]:
        axes[1].plot(frames.loc[is_example, "row_in_episode"], wrists_by_rule[rule][is_example], color = colour, label = rule.replace(", " + first_run, ""))
    axes[1].set_xlabel("row of the episode")
    axes[1].set_ylabel("wrist, joint 5 (degrees)")
    axes[1].set_title(f"One sheet-06 demonstration (median wrist range): {example_episode.split('/')[-1][-28:]}", fontsize = 9)
    axes[1].legend(fontsize = 7)
    axes[1].grid(alpha = 0.3)

    shown = scores[scores["rows"] == "start-mark episodes"].sort_values("mean_distance_deg")
    axes[2].barh(range(len(shown)), shown["mean_distance_deg"], color = "tab:blue")
    axes[2].set_yticks(range(len(shown)))
    axes[2].set_yticklabels([rule.replace("follow the model's lean output from 13.5, ", "follow lean output, ").replace("fitted on reach and twist", "fitted") for rule in shown["rule"]], fontsize = 7)
    axes[2].set_xlabel("mean distance to the demonstrated wrist (degrees)")
    axes[2].set_title("Wrist rules, start-mark episodes")
    axes[2].grid(alpha = 0.3, axis = "x")
    figure.suptitle("Wrist rules compared on the demonstrations (calculated; fold models' predictions on sheets they never saw)", fontsize = 10)
    figure.tight_layout()
    figure.savefig(output_folder / "wrist_options.png", dpi = settings["figure_dpi"])
    pyplot.close(figure)
    print(f"\nsaved {output_folder / 'wrist_options.png'} and wrist_options_scores.csv")



if __name__ == "__main__":
    main()
