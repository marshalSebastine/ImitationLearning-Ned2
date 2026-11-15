"""
The ten robot runs of 11.11.26 against the demonstrations at the same places (key: wrist_and_camera, part 2).

For every picture of a robot run, the demonstration rows of the same sheet, laid the same way (the same line end on
the taped start mark as in plan 0X-01, by the audit's corrected ends; standard start kinds only), whose magnet tip was
within "radius_m" of the robot's tip are looked up. Their
wrist (joint 5), forearm twist (joint 4) and rod lean are compared with the robot's logged values. The wrist decides
where the hand, the block and the table sit in the picture (2.8 px at 640 per degree, held_wrist_views.py), so the
difference between the held 13.5 degrees and the neighbours' wrist is the view mismatch the held wrist caused at that
place; the difference in joint 4 is the mismatch that the twist caused (the picture turns by about that angle).

Measured: the robot's joints (steps.csv) and the demonstrations' joints (frames.csv). Calculated: tips, leans, shifts.
Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe runs_against_demonstrations.py
"""

from pathlib import Path
from typing import Dict, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import wrist_geometry


COMPARISON_PARAMETERS = {
    "radius_m": 0.02,
    "twist_window_deg": 20.0,
    "fewest_neighbours": 5,
    # 2.80 px at 640 per degree of wrist (held_wrist_views.py); the training shift augmentation covers 16 px at 640
    "pixels_per_wrist_degree": 2.803,
    "augmentation_shift_640_px": 16.0,
    "standard_kinds": ["STD", "OL1", "OL1.5", "OR1", "OR1.5", "TL10", "TL15", "TR10", "TR15"],
    "corrected_ends_file": wrist_geometry.ROOT_FOLDER / "06_deliverables" / "research_notes" / "Model and training" / "evidence" / "corrected_ends.csv",
    # run 3: sheet 02 "lying straight (not as in plan 02-01)" (the brief), so no single demonstrated placement is its match
    "runs_not_laid_as_planned": [3],
    # the run and the pictures shown beside their nearest demonstration pictures (the approach to sheet 03's first bend)
    "picture_pair_run": 5,
    "picture_pair_numbers": [2, 6, 10],
    "figure_dpi": 120
}



def placement_family(
        frames: pandas.DataFrame,
        episodes: pandas.DataFrame,
        sheet: str,
        is_laid_as_planned: bool,
        parameters: Dict
) -> pandas.DataFrame:
    """
    The demonstration rows of ``sheet`` recorded with the sheet laid as in its plan 01: the same line end on the taped
    start mark and a standard start kind (not the square-to-the-robot kinds SQP and SQL of sheet 02, not a taught
    start). The line end comes from the audit's table of corrected ends ("corrected_ends_file"): the index's own
    "start_end" is wrong for 30 episodes, most of them on sheets 02, 05 and 06, where it would mix the two directions
    of travel. For a run whose sheet did not lie as planned (run 3), every start-mark episode of the sheet is taken.
    """

    corrected = pandas.read_csv(parameters["corrected_ends_file"]).set_index("episode_id")["corrected_end"]
    of_sheet = episodes[episodes["sheet"] == sheet].copy()
    of_sheet["corrected_end"] = of_sheet["episode_id"].map(corrected).fillna(of_sheet["start_end"])
    of_sheet["kind"] = of_sheet["sheet_orientation"].str.split(", ", expand = True)[2]
    is_on_mark = of_sheet["sheet_orientation"].str.contains("START MARK")
    if is_laid_as_planned:
        first_plan = of_sheet[of_sheet["sheet_orientation"].str.contains(r"plan \d\d-01,", regex = True)]
        start_end = first_plan["corrected_end"].iloc[0]
        chosen = of_sheet[is_on_mark & of_sheet["kind"].isin(parameters["standard_kinds"]) & (of_sheet["corrected_end"] == start_end)]["episode_id"]
    else:
        start_end = "any"
        chosen = of_sheet[is_on_mark]["episode_id"]
    family = frames[frames["episode_id"].isin(chosen)].copy()
    family.attrs["start_end"] = start_end
    family.attrs["episodes"] = len(chosen)
    return family



def neighbours_of_run(
        steps: pandas.DataFrame,
        family: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    For every picture of a run: how many demonstration rows of the family lie within the radius, and their wrist,
    twist and lean, once for any twist and once for a twist within the window of the robot's.
    """

    family_xy = family[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    family_wrist = family["joint_5_deg"].to_numpy(dtype = float)
    family_twist = family["joint_4_deg"].to_numpy(dtype = float)
    family_lean = family["rod_from_vertical_deg"].to_numpy(dtype = float)
    robot_leans = wrist_geometry.tips_and_leans(steps[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float))[1]
    held = wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]
    rows = []
    for position, (_, step) in enumerate(steps.iterrows()):
        distances = numpy.hypot(family_xy[:, 0] - step["tip_x"], family_xy[:, 1] - step["tip_y"])
        is_near = distances <= parameters["radius_m"]
        is_near_and_turned_alike = is_near & (numpy.abs(family_twist - step["joint_4_deg"]) <= parameters["twist_window_deg"])
        row = {"run_number": int(step["run_number"]), "step": int(step["step"]), "tip_x": step["tip_x"], "tip_y": step["tip_y"],
               "reach_cm": 100.0 * step["reach_m"], "joint_4_deg": step["joint_4_deg"], "joint_5_deg": step["joint_5_deg"],
               "lean_deg": robot_leans[position], "nearest_demonstration_cm": 100.0 * float(distances.min()),
               "neighbours": int(is_near.sum()), "neighbours_turned_alike": int(is_near_and_turned_alike.sum())}
        for label, mask in [("near", is_near), ("alike", is_near_and_turned_alike)]:
            if mask.sum() >= parameters["fewest_neighbours"]:
                offsets = family_wrist[mask] - held
                row[f"{label}_wrist_median_deg"] = float(numpy.median(family_wrist[mask]))
                row[f"{label}_wrist_p10_deg"] = float(numpy.quantile(family_wrist[mask], 0.1))
                row[f"{label}_wrist_p90_deg"] = float(numpy.quantile(family_wrist[mask], 0.9))
                row[f"{label}_share_wrist_within_range"] = float((numpy.abs(offsets) <= parameters["augmentation_shift_640_px"] / parameters["pixels_per_wrist_degree"]).mean())
                row[f"{label}_twist_median_deg"] = float(numpy.median(family_twist[mask]))
                row[f"{label}_twist_p10_deg"] = float(numpy.quantile(family_twist[mask], 0.1))
                row[f"{label}_twist_p90_deg"] = float(numpy.quantile(family_twist[mask], 0.9))
                row[f"{label}_lean_median_deg"] = float(numpy.median(family_lean[mask]))
        rows.append(row)
    return pandas.DataFrame(rows)



def save_picture_pairs(
        run: Dict,
        steps: pandas.DataFrame,
        family: pandas.DataFrame,
        parameters: Dict
) -> None:
    """
    Save a robot run's pictures beside the demonstration pictures taken nearest to them: nearest in tip position (cm)
    plus a tenth of the difference in forearm twist (degrees), within the run's placement family. Left the robot's
    picture, right the demonstration's; the wrist of both is written on them.
    """

    raw_folder = Path(wrist_geometry.ANALYSIS_PARAMETERS["raw_folder"])
    run_folder = Path(wrist_geometry.ANALYSIS_PARAMETERS["model_runs_folder"]) / run["folder"]
    tiles = []
    for picture_number in parameters["picture_pair_numbers"]:
        step = steps.iloc[picture_number]
        distances = (100.0 * numpy.hypot(family["tip_x_m"] - step["tip_x"], family["tip_y_m"] - step["tip_y"]) + (family["joint_4_deg"] - step["joint_4_deg"]).abs() / 10.0)
        nearest = family.loc[distances.idxmin()]
        robot_picture = cv2.imread(str(run_folder / step["frame"]))
        demonstration_picture = cv2.imread(str(raw_folder / nearest["frame_path"]))
        gap_cm = 100.0 * float(numpy.hypot(nearest["tip_x_m"] - step["tip_x"], nearest["tip_y_m"] - step["tip_y"]))
        cv2.putText(robot_picture, f"robot run {run['number']}, picture {picture_number}: twist {step['joint_4_deg']:.0f}, wrist {step['joint_5_deg']:.1f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(demonstration_picture, f"demonstration {gap_cm:.1f} cm away: twist {nearest['joint_4_deg']:.0f}, wrist {nearest['joint_5_deg']:.1f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(numpy.hstack([robot_picture, demonstration_picture]))
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    cv2.imwrite(str(output_folder / f"run_{run['number']:02d}_pictures_against_nearest_demonstration.jpg"), numpy.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])



def main(parameters: Optional[Dict] = None) -> None:
    """Print the per-run comparison, save the per-picture table, one figure per run and the picture pairs of one run."""

    settings = dict(COMPARISON_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 400)
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    frames = wrist_geometry.load_demonstration_frames()
    episodes = pandas.read_csv(wrist_geometry.ANALYSIS_PARAMETERS["episodes_file"])
    held = wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]
    range_deg = settings["augmentation_shift_640_px"] / settings["pixels_per_wrist_degree"]
    print(f"neighbours: demonstration rows of the same sheet and placement with the tip within {100 * settings['radius_m']:.0f} cm; "
          f"'turned alike': joint 4 within {settings['twist_window_deg']:.0f} degrees of the robot's; "
          f"a wrist within {range_deg:.1f} degrees of the held {held} keeps the view within the training shift augmentation (16 px at 640)")

    # --- where and how each run started, against the one start pose of the start-mark demonstrations ---
    on_mark = episodes[episodes["sheet_orientation"].str.contains("START MARK")]["episode_id"]
    first_rows = frames[(frames["row_in_episode"] == 0) & frames["episode_id"].isin(on_mark)]
    first_headings = numpy.degrees(wrist_geometry.headings_rad(first_rows[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)))
    print(f"\n=== the start: {len(first_rows)} start-mark demonstrations began at one pose ===")
    print(f"tip x {100 * first_rows['tip_x_m'].min():.2f} to {100 * first_rows['tip_x_m'].max():.2f} cm, y {100 * first_rows['tip_y_m'].min():.2f} to {100 * first_rows['tip_y_m'].max():.2f} cm; "
          f"twist {first_rows['joint_4_deg'].min():+.2f} to {first_rows['joint_4_deg'].max():+.2f}; wrist {first_rows['joint_5_deg'].min():.2f} to {first_rows['joint_5_deg'].max():.2f}; "
          f"hand {first_rows['joint_6_deg'].min():.2f} to {first_rows['joint_6_deg'].max():.2f} degrees; picture heading {first_headings.min():+.2f} to {first_headings.max():+.2f} degrees")
    starts = []
    for run in wrist_geometry.ANALYSIS_PARAMETERS["robot_runs"]:
        steps = wrist_geometry.load_robot_run(run)
        heading = float(numpy.degrees(wrist_geometry.headings_rad(steps[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)[:1]))[0])
        starts.append({"run": run["number"], "sheet": run["sheet"][-2:], "tip_x_cm": 100 * steps["tip_x"].iloc[0], "tip_y_cm": 100 * steps["tip_y"].iloc[0],
                       "gap_to_demonstrated_start_cm": 100 * float(numpy.hypot(steps["tip_x"].iloc[0] - first_rows["tip_x_m"].median(), steps["tip_y"].iloc[0] - first_rows["tip_y_m"].median())),
                       "twist_deg": steps["joint_4_deg"].iloc[0], "picture_heading_deg": heading, "heading_gap_deg": heading - float(numpy.median(first_headings))})
    starts = pandas.DataFrame(starts)
    print("the ten robot runs (hand-placed starts; the driver keeps the twist the hand left):")
    print(starts.round(1).to_string(index = False))
    print(f"gap to the demonstrated start tip: {starts['gap_to_demonstrated_start_cm'].min():.1f} to {starts['gap_to_demonstrated_start_cm'].max():.1f} cm (median "
          f"{starts['gap_to_demonstrated_start_cm'].median():.1f}); start twist {starts['twist_deg'].min():+.1f} to {starts['twist_deg'].max():+.1f} degrees; "
          f"picture heading {starts['heading_gap_deg'].abs().min():.1f} to {starts['heading_gap_deg'].abs().max():.1f} degrees from the demonstrated one (median {starts['heading_gap_deg'].abs().median():.1f})")
    starts.round(2).to_csv(output_folder / "run_starts_against_demonstrated_start.csv", index = False)

    all_tables = []
    summaries = []
    for run in wrist_geometry.ANALYSIS_PARAMETERS["robot_runs"]:
        steps = wrist_geometry.load_robot_run(run)
        family = placement_family(frames, episodes, run["sheet"], run["number"] not in settings["runs_not_laid_as_planned"], settings)
        table = neighbours_of_run(steps, family, settings)
        all_tables.append(table)
        if run["number"] == settings["picture_pair_run"]:
            save_picture_pairs(run, steps, family, settings)
        has_near = table["near_wrist_median_deg"].notna() if "near_wrist_median_deg" in table else pandas.Series(False, index = table.index)
        has_alike = table["alike_wrist_median_deg"].notna() if "alike_wrist_median_deg" in table else pandas.Series(False, index = table.index)
        summary = {
            "run": run["number"], "sheet": run["sheet"][-2:], "model": run["model"], "pictures": len(table),
            "family_episodes": family.attrs["episodes"], "family_end": family.attrs["start_end"],
            "robot_wrist_deg": round(float(table["joint_5_deg"].median()), 2),
            "with_neighbours_share": float(has_near.mean()),
            "turned_alike_share": float(has_alike.mean())
        }
        if has_near.any():
            near = table[has_near]
            summary["near_wrist_median_deg"] = float(near["near_wrist_median_deg"].median())
            summary["near_wrist_off_range_share"] = float(((near["near_wrist_median_deg"] - held).abs() > range_deg).mean())
            summary["near_rows_in_range_share"] = float(near["near_share_wrist_within_range"].mean())
            summary["twist_gap_median_deg"] = float((near["joint_4_deg"] - near["near_twist_median_deg"]).abs().median())
            summary["twist_gap_p90_deg"] = float((near["joint_4_deg"] - near["near_twist_median_deg"]).abs().quantile(0.9))
            summary["twist_outside_p10_p90_share"] = float(((near["joint_4_deg"] < near["near_twist_p10_deg"] - 5.0) | (near["joint_4_deg"] > near["near_twist_p90_deg"] + 5.0)).mean())
            summary["lean_gap_median_deg"] = float((near["lean_deg"] - near["near_lean_median_deg"]).median())
        if has_alike.any():
            alike = table[has_alike]
            summary["alike_wrist_median_deg"] = float(alike["alike_wrist_median_deg"].median())
            summary["alike_rows_in_range_share"] = float(alike["alike_share_wrist_within_range"].mean())
            summary["alike_view_shift_median_px"] = float(((alike["alike_wrist_median_deg"] - held).abs() * settings["pixels_per_wrist_degree"]).median())
            summary["alike_view_shift_p90_px"] = float(((alike["alike_wrist_median_deg"] - held).abs() * settings["pixels_per_wrist_degree"]).quantile(0.9))
        summaries.append(summary)

        figure, axes = pyplot.subplots(1, 3, figsize = (17, 5.2))
        points = axes[0].scatter(family["tip_y_m"] * 100.0, family["tip_x_m"] * 100.0, c = family["joint_5_deg"], s = 5, cmap = "coolwarm", vmin = held - 20.0, vmax = held + 20.0)
        figure.colorbar(points, ax = axes[0], label = "demonstrated wrist, joint 5 (degrees); white = the held 13.5")
        axes[0].plot(steps["tip_y"] * 100.0, steps["tip_x"] * 100.0, color = "black", linewidth = 1.5, marker = ".", markersize = 4, label = "robot run (wrist 13.5)")
        axes[0].plot(steps["tip_y"].iloc[0] * 100.0, steps["tip_x"].iloc[0] * 100.0, marker = "o", color = "lime", markersize = 8, markeredgecolor = "black", label = "run start")
        axes[0].invert_xaxis()
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("y (cm), robot's left to the left")
        axes[0].set_ylabel("x (cm), away from the base")
        axes[0].set_title(f"Tips seen from above: {family.attrs['episodes']} demonstrations (end {family.attrs['start_end']} on the mark)")
        axes[0].legend(fontsize = 8)

        axes[1].plot(table["step"], table["joint_4_deg"], color = "black", label = "robot's joint 4")
        if has_near.any():
            axes[1].fill_between(table["step"], table["near_twist_p10_deg"], table["near_twist_p90_deg"], color = "tab:orange", alpha = 0.3, label = "demonstrations within 2 cm: 10th to 90th percentile")
            axes[1].plot(table["step"], table["near_twist_median_deg"], color = "tab:orange", label = "their median")
        axes[1].set_xlabel("picture number of the run")
        axes[1].set_ylabel("forearm twist, joint 4 (degrees)")
        axes[1].set_title("Forearm twist at the same places")
        axes[1].legend(fontsize = 8)
        axes[1].grid(alpha = 0.3)

        axes[2].axhline(held, color = "black", label = "robot's wrist (held)")
        axes[2].axhspan(held - range_deg, held + range_deg, color = "grey", alpha = 0.2, label = "view within the training shift augmentation")
        if has_near.any():
            axes[2].fill_between(table["step"], table["near_wrist_p10_deg"], table["near_wrist_p90_deg"], color = "tab:blue", alpha = 0.3, label = "demonstrations within 2 cm: 10th to 90th percentile")
            axes[2].plot(table["step"], table["near_wrist_median_deg"], color = "tab:blue", label = "their median")
        if has_alike.any():
            axes[2].plot(table["step"], table["alike_wrist_median_deg"], color = "tab:green", linestyle = "--", label = "median of those also turned alike (20 degrees)")
        axes[2].set_ylim(-32.0, 47.0)
        axes[2].set_xlabel("picture number of the run")
        axes[2].set_ylabel("wrist, joint 5 (degrees)")
        axes[2].set_title("Wrist at the same places")
        axes[2].legend(fontsize = 8)
        axes[2].grid(alpha = 0.3)
        figure.suptitle(f"Run {run['number']} ({run['folder']}), model {run['model']}: robot against demonstrations at the same places", fontsize = 11)
        figure.tight_layout()
        figure.savefig(output_folder / f"run_{run['number']:02d}_against_demonstrations.png", dpi = settings["figure_dpi"])
        pyplot.close(figure)

    summary_table = pandas.DataFrame(summaries)
    print("\n=== per run ===")
    print(summary_table.round(2).to_string(index = False))
    pandas.concat(all_tables, ignore_index = True).to_csv(output_folder / "runs_against_demonstrations_steps.csv", index = False)
    summary_table.to_csv(output_folder / "runs_against_demonstrations_summary.csv", index = False)

    print("\n=== where the failing runs went wrong: pictures 0 to 30 of runs 5, 6, 8, 9 and 10 ===")
    combined = pandas.concat(all_tables, ignore_index = True)
    columns = ["run_number", "step", "reach_cm", "joint_4_deg", "near_twist_median_deg", "near_twist_p10_deg", "near_twist_p90_deg", "lean_deg",
               "near_lean_median_deg", "neighbours", "neighbours_turned_alike", "near_wrist_p10_deg", "near_wrist_median_deg", "near_wrist_p90_deg",
               "near_share_wrist_within_range", "nearest_demonstration_cm"]
    for number in (5, 6, 8, 9, 10):
        print(f"\nrun {number}")
        part = combined[(combined["run_number"] == number) & (combined["step"] <= 30) & (combined["step"] % 2 == 0)]
        print(part[[column for column in columns if column in part]].round(1).to_string(index = False))



if __name__ == "__main__":
    main()
