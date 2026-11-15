"""
The two quality requirements side by side for the demonstrations and the robot's runs of 11.11.26, from the tables the
other scripts wrote (robot_runs.py, demonstration_ends.py, demonstration_centring.py).

    ends        where the puck's centre stands against the centre of the line's round end at the stop;
    centring    the sideways offset of the puck's centre from the line's middle during the push, and whether an offset
                shrinks again over the next steps.

Run from this folder, after the other scripts:    python summary.py
Writes ends_scatter.png and centring_distribution.png; the printed tables are kept in summary.out.txt.
"""

import glob
from typing import Dict

import numpy
import pandas
from matplotlib import pyplot

import picture_measures
import run_measures


SUMMARY_PARAMETERS = {
    # the puck hides the line's full width, and the round end completely, while its centre is within its radius minus
    # half the line's width of the line's middle or of the end's centre (12 - 5 = 7 mm); beyond radius plus half the
    # line's width (17 mm) the puck's foot does not touch the line at all
    "cover_limit_mm": 7.0,
    "well_off_mm": 12.0,
    "off_line_mm": 17.0,
    "line_width_range_mm": [8.0, 12.5],
    "fit_quality_minimum": 0.8,
    "robot_runs_reached": [1, 2, 4, 7],
    # whether an offset shrinks: steps with an offset of at least "restoring_from_mm" are followed this many steps
    "restoring_from_mm": 7.0,
    "restoring_steps": 5,
    "start_kind_groups": {"standard": ["STD"], "puck put to the side": ["OL1", "OL1.5", "OR1", "OR1.5"],
                          "start turned": ["TL10", "TL15", "TR10", "TR15"], "near end or middle": ["NE", "MID", "X1", "X2"],
                          "square sheet": ["SQP", "SQL"]},
    "figure_dpi": 110
}



def offset_summary(offsets: pandas.Series) -> Dict:
    """Numbers on the size of the sideways offsets of a group of steps."""

    sizes = offsets.dropna().abs()
    if sizes.empty:
        return {"steps": 0}
    return {"steps": int(len(sizes)), "median_mm": round(float(sizes.median()), 1), "p75_mm": round(float(sizes.quantile(0.75)), 1),
            "p90_mm": round(float(sizes.quantile(0.9)), 1), "largest_mm": round(float(sizes.max()), 1),
            "within_2_5": round(float((sizes <= 2.5).mean()), 2), "within_5": round(float((sizes <= 5.0).mean()), 2),
            "within_7": round(float((sizes <= SUMMARY_PARAMETERS["cover_limit_mm"]).mean()), 2),
            "over_12": round(float((sizes > SUMMARY_PARAMETERS["well_off_mm"]).mean()), 2),
            "off_the_line": round(float((sizes > SUMMARY_PARAMETERS["off_line_mm"]).mean()), 2)}



def end_summary(ends: pandas.DataFrame) -> Dict:
    """Numbers on where the puck stands against the line's end over a group of episodes."""

    limit = SUMMARY_PARAMETERS["cover_limit_mm"]
    return {"episodes": int(len(ends)), "short_mean_mm": round(float(ends["short_mm"].mean()), 1), "short_sd_mm": round(float(ends["short_mm"].std()), 1),
            "aside_mean_mm": round(float(ends["aside_mm"].mean()), 1), "aside_sd_mm": round(float(ends["aside_mm"].std()), 1),
            "aside_size_median_mm": round(float(ends["aside_mm"].abs().median()), 1),
            "gap_median_mm": round(float(ends["centre_gap_mm"].median()), 1), "gap_p90_mm": round(float(ends["centre_gap_mm"].quantile(0.9)), 1),
            "covered": int((ends["centre_gap_mm"] <= limit).sum()), "covered_share": round(float((ends["centre_gap_mm"] <= limit).mean()), 2),
            "short_over_limit": int((ends["short_mm"] > limit).sum()), "beyond_over_limit": int((ends["short_mm"] < -limit).sum()),
            "aside_over_limit": int((ends["aside_mm"].abs() > limit).sum())}



def visible_share_of_end(
        short_mm: float,
        aside_mm: float
) -> float:
    """
    How much of the line's last centimetre (the round end and half a centimetre of line before it) is not under the
    puck's foot, seen from above, for a puck whose centre stands ``short_mm`` before the end's centre and ``aside_mm``
    to its side. Calculated from the two measured distances and the sizes of the puck and the line.

    Returns
    -------
    float
        The visible share of that area, 0 (all hidden) to 1.
    """

    half_width = 1000.0 * picture_measures.MEASURE_PARAMETERS["line_width_m"] / 2.0
    radius = 1000.0 * picture_measures.MEASURE_PARAMETERS["puck_radius_m"]
    along, across = numpy.meshgrid(numpy.arange(-half_width, half_width + 0.125, 0.25), numpy.arange(-half_width, half_width + 0.125, 0.25))
    is_line = (numpy.abs(across) <= half_width) & ((along <= 0.0) | (along ** 2 + across ** 2 <= half_width ** 2))
    # the puck's centre lies at -short along the line from the end's centre
    is_hidden = numpy.hypot(along + short_mm, across - aside_mm) <= radius
    return float(numpy.count_nonzero(is_line & ~is_hidden)) / float(numpy.count_nonzero(is_line))



def restoring(
        steps: pandas.DataFrame,
        group_column: str
) -> Dict:
    """
    Whether an offset shrinks again: for every step with an offset of "restoring_from_mm" or more, the change of the
    offset's size "restoring_steps" steps later in the same run.
    """

    changes = []
    ahead = SUMMARY_PARAMETERS["restoring_steps"]
    for _, run in steps.groupby(group_column):
        run = run.sort_values("step")
        sizes = run["offset_mm"].abs().to_numpy()
        for position in range(len(sizes) - ahead):
            if numpy.isfinite(sizes[position]) and sizes[position] >= SUMMARY_PARAMETERS["restoring_from_mm"] and numpy.isfinite(sizes[position + ahead]):
                changes.append((sizes[position], sizes[position + ahead] - sizes[position]))
    if not changes:
        return {"steps": 0}
    changes = numpy.array(changes)
    return {"steps": int(len(changes)), "start_median_mm": round(float(numpy.median(changes[:, 0])), 1),
            "change_median_mm": round(float(numpy.median(changes[:, 1])), 1),
            "share_smaller_by_2_mm": round(float((changes[:, 1] <= -2.0).mean()), 2),
            "share_larger_by_2_mm": round(float((changes[:, 1] >= 2.0).mean()), 2)}



def main() -> None:

    folder = run_measures.EVIDENCE_FOLDER
    limit = SUMMARY_PARAMETERS["cover_limit_mm"]
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 30)

    # ---------------------------------------------------------------- the ends
    ends = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(folder / "demonstration_ends_0*.csv")))], ignore_index = True)
    width_least, width_most = SUMMARY_PARAMETERS["line_width_range_mm"]
    ends["is_reliable"] = (ends["is_end_found"] & (ends["is_end_seen"] == True) & ends["line_width_mm"].between(width_least, width_most)
                           & (ends["fit_quality"] >= SUMMARY_PARAMETERS["fit_quality_minimum"]))
    reliable = ends[ends["is_reliable"]]
    print("THE END OF THE LINE")
    print(f"demonstrations: {len(ends)}; end found in {int(ends['is_end_found'].sum())}; measured reliably (bare paper seen beyond the end, "
          f"line {width_least:.0f} to {width_most:.1f} mm wide in the map, puck fitted well) in {len(reliable)}")
    print("reliable ends by sheet:", reliable.groupby("sheet").size().to_dict(), "of", ends.groupby("sheet").size().to_dict())
    print("\nwhere the demonstrations stop (puck's centre against the centre of the round end; short: before it along the line, aside: left of the line):")
    rows = {"all sheets": end_summary(reliable)}
    for sheet, group in reliable.groupby("sheet"):
        rows[sheet] = end_summary(group)
    for score, group in reliable.groupby("score"):
        rows[f"score {int(score)}"] = end_summary(group)
    for operator, group in reliable.groupby("operator"):
        rows[f"operator {operator}"] = end_summary(group)
    for period, group in reliable.groupby("camera_period"):
        rows[f"camera {period}"] = end_summary(group)
    for name, kinds in SUMMARY_PARAMETERS["start_kind_groups"].items():
        group = reliable[reliable["start_kind"].isin(kinds)]
        if len(group):
            rows[f"start: {name}"] = end_summary(group)
    print(pandas.DataFrame(rows).T.to_string())
    print(f"\n'covered' means the two centres are at most {limit:.0f} mm apart, so that the puck (12 mm radius) hides the round end (5 mm radius) completely.")
    print(f"the stop picture itself shows ink ahead of the puck in {int((reliable['ink_seen_ahead_mm2'] > 2.0).sum())} of {len(reliable)} reliable demonstrations; "
          f"seen from above, ink sticks out ahead of the puck (over 5 square mm) in {int((reliable['ink_beyond_mm2'] > 5.0).sum())}")
    print(f"the operators' own verdict was 'puck_in_goal' in {int((reliable['outcome'] == 'puck_in_goal').sum())} of {len(reliable)}; measured covered: {int((reliable['centre_gap_mm'] <= limit).sum())}")

    visible = numpy.array([visible_share_of_end(row["short_mm"], row["aside_mm"]) for _, row in reliable.iterrows()])
    print(f"how much of the line's last centimetre is not under the puck at the demonstrated stop, seen from above (calculated from the two distances): "
          f"nothing in {int((visible == 0).sum())} of {len(visible)}, up to a tenth in {int(((visible > 0) & (visible <= 0.1)).sum())}, "
          f"a tenth to a third in {int(((visible > 0.1) & (visible <= 1.0 / 3.0)).sum())}, more than a third in {int((visible > 1.0 / 3.0).sum())}; "
          f"median {100.0 * numpy.median(visible):.0f} %, largest {100.0 * visible.max():.0f} %")

    robot_ends = pandas.read_csv(folder / "robot_runs_ends.csv")
    robot_ends = robot_ends[robot_ends["run"].isin(SUMMARY_PARAMETERS["robot_runs_reached"])].copy()
    robot_ends["visible_share_of_last_cm"] = [visible_share_of_end(row["end_short_mm"], row["end_aside_mm"]) for _, row in robot_ends.iterrows()]
    print("\nwhere the robot stopped, the four runs that reached the end:")
    print(robot_ends[["run", "model", "sheet", "end_short_mm", "end_aside_mm", "end_centre_gap_mm", "end_uncovered_mm", "visible_share_of_last_cm",
                      "end_ink_seen_ahead_mm2", "line_width_in_map_mm"]].round(2).to_string(index = False))
    print("('end_uncovered_mm': how far the round end's rim sticks out from under the puck at its worst point; 'visible_share_of_last_cm': the share of the line's")
    print(" last centimetre not under the puck, seen from above; 'end_ink_seen_ahead_mm2': the line the stop picture itself shows ahead of the puck)")
    worse = {int(row["run"]): f"{100.0 * float((reliable['centre_gap_mm'] > row['end_centre_gap_mm']).mean()):.0f} %" for _, row in robot_ends.iterrows()}
    print("share of the demonstrations that end further from the end's centre than each robot run:", worse)

    figure, axis = pyplot.subplots(figsize = (7.5, 7.5))
    axis.scatter(reliable["aside_mm"], -reliable["short_mm"], s = 14, color = "0.45", label = f"demonstrations ({len(reliable)})")
    angles = numpy.linspace(0.0, 2.0 * numpy.pi, 200)
    axis.plot(limit * numpy.cos(angles), limit * numpy.sin(angles), color = "tab:green", linewidth = 1.5, label = f"end covered completely (centres within {limit:.0f} mm)")
    for _, row in robot_ends.iterrows():
        axis.scatter([row["end_aside_mm"]], [-row["end_short_mm"]], s = 90, marker = "*", color = "tab:red", zorder = 4)
        axis.annotate(f"run {int(row['run'])} ({row['model']}, sheet {int(row['sheet']):02d})", (row["end_aside_mm"], -row["end_short_mm"]),
                      textcoords = "offset points", xytext = (6, 5), fontsize = 8, color = "tab:red")
    axis.scatter([], [], s = 90, marker = "*", color = "tab:red", label = "robot, 11.11.26")
    axis.axhline(0.0, color = "black", linewidth = 0.5)
    axis.axvline(0.0, color = "black", linewidth = 0.5)
    axis.set_xlim(27, -27)
    axis.set_ylim(-27, 27)
    axis.set_aspect("equal")
    axis.set_xlabel("puck's centre left of the line's middle, mm (left is on the left)")
    axis.set_ylabel("puck's centre beyond the end's centre, mm (below zero: short of it)")
    axis.set_title("Where the puck stands at the stop, seen from above with the line coming from below", fontsize = 10)
    axis.legend(fontsize = 8, loc = "lower left")
    figure.tight_layout()
    figure.savefig(folder / "ends_scatter.png", dpi = SUMMARY_PARAMETERS["figure_dpi"])
    pyplot.close(figure)

    # ---------------------------------------------------------------- the middle of the line
    centring = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(folder / "demonstration_centring_0*.csv")))], ignore_index = True)
    robot_steps = pandas.read_csv(folder / "robot_runs_steps.csv")
    reached = robot_steps[robot_steps["run"].isin(SUMMARY_PARAMETERS["robot_runs_reached"])]
    print("\n\nTHE MIDDLE OF THE LINE")
    print(f"demonstrations sampled: {centring['episode_id'].nunique()} episodes, {len(centring)} pictures; puck fitted in {int(centring['puck_x'].notna().sum())}, "
          f"offset measured in {int(centring['offset_mm'].notna().sum())}")
    rows = {"demonstrations, all sampled": offset_summary(centring["offset_mm"])}
    for sheet, group in centring.groupby("sheet"):
        rows[f"demonstrations, {sheet} ({group['episode_id'].nunique()} episodes)"] = offset_summary(group["offset_mm"])
    for name, kinds in SUMMARY_PARAMETERS["start_kind_groups"].items():
        group = centring[centring["start_kind"].isin(kinds)]
        if len(group):
            rows[f"demonstrations, start: {name} ({group['episode_id'].nunique()})"] = offset_summary(group["offset_mm"])
    for score, group in centring.groupby("score"):
        rows[f"demonstrations, score {int(score)} ({group['episode_id'].nunique()})"] = offset_summary(group["offset_mm"])
    for run, group in reached.groupby("run"):
        rows[f"robot run {run} (model {'A' if run == 1 else 'E'})"] = offset_summary(group["offset_mm"])
    rows["robot, the six runs that lost the line"] = offset_summary(robot_steps.loc[~robot_steps["run"].isin(SUMMARY_PARAMETERS["robot_runs_reached"]), "offset_mm"])
    is_one_stretch = centring["is_clean"] & centring["is_aligned"]
    rows["demonstrations, only pictures with one clean stretch of line beside the puck"] = offset_summary(centring.loc[is_one_stretch, "offset_mm"])
    rows["demonstrations, standard starts after the first 15 pictures"] = offset_summary(centring.loc[(centring["start_kind"] == "STD") & (centring["step"] > 15), "offset_mm"])
    print("size of the sideways offset between the puck's centre and the line's middle, per picture (mm; shares of pictures):")
    print(pandas.DataFrame(rows).T.to_string())
    print(f"('within_7': the puck hides the line's full width; 'over_12': the middle of the line is no longer under the puck; "
          f"'off_the_line': over {SUMMARY_PARAMETERS['off_line_mm']:.0f} mm, the puck's foot does not touch the line.")
    print(" How good the measure is: two pictures 12 steps apart put the same piece of line a median 1.8 to 3.3 mm apart, the worst tenth 2.7 to 6.8 mm")
    print(" (measure_check.out.txt and the same check on three demonstrations), so the shares at 2.5, 5 and 7 mm are rough where the forearm turns,")
    print(" while offsets over 12 mm are safely real.)")

    per_episode = centring.groupby("episode_id")["offset_mm"].agg(lambda column: column.abs().median())
    print(f"\nper demonstration, the median offset: smallest {per_episode.min():.1f}, median {per_episode.median():.1f}, largest {per_episode.max():.1f} mm; "
          f"demonstrations whose largest offset stays within {limit:.0f} mm: {int((centring.groupby('episode_id')['offset_mm'].agg(lambda column: column.abs().max()) <= limit).sum())} of {centring['episode_id'].nunique()}")
    robot_by_sheet = {1: "path_01", 2: "path_01", 4: "path_02", 7: "path_05"}
    for run, sheet in robot_by_sheet.items():
        robot_median = float(reached.loc[reached["run"] == run, "offset_mm"].abs().median())
        same_sheet = centring[(centring["sheet"] == sheet)].groupby("episode_id")["offset_mm"].agg(lambda column: column.abs().median())
        print(f"  robot run {run}: median {robot_median:.1f} mm; the sampled demonstrations of {sheet}: medians {same_sheet.min():.1f} to {same_sheet.max():.1f} mm "
              f"({int((same_sheet > robot_median).sum())} of {len(same_sheet)} worse than the robot)")

    print(f"\ndoes an offset shrink again? steps with an offset of {SUMMARY_PARAMETERS['restoring_from_mm']:.0f} mm or more, {SUMMARY_PARAMETERS['restoring_steps']} steps later:")
    print("  demonstrations:", restoring(centring, "episode_id"))
    print("  robot, the four runs that reached the end:", restoring(reached, "run"))
    side_starts = centring[centring["start_kind"].isin(SUMMARY_PARAMETERS["start_kind_groups"]["puck put to the side"])]
    print("\nthe demonstrated recoveries (starts with the puck put 1 or 1.5 cm to the side): offset by picture")
    for first, last in ((1, 3), (4, 6), (7, 10), (11, 15), (16, 25)):
        part = side_starts[side_starts["step"].between(first, last)]["offset_mm"].dropna().abs()
        if len(part):
            print(f"  pictures {first:2d} to {last:2d}: median {part.median():.1f} mm, 90th percentile {part.quantile(0.9):.1f} mm ({len(part)} pictures of {side_starts['episode_id'].nunique()} episodes)")

    both = centring[centring["is_clean"] & centring["is_aligned"] & centring["is_single_clean"] & centring["is_single_aligned"]].dropna(subset = ["offset_mm", "single_offset_mm"])
    difference = (both["single_offset_mm"] - both["offset_mm"]).abs()
    measured = int((centring["is_clean"] & centring["is_aligned"] & centring["offset_mm"].notna()).sum())
    print(f"\none picture alone against the votes of the last 12 pictures, on the demonstrations: of {measured} clean steps the one picture gives a clean offset in {len(both)}; "
          f"difference median size {difference.median():.1f} mm, 90th percentile {difference.quantile(0.9):.1f} mm, over 5 mm in {int((difference > 5.0).sum())}")

    figure, axis = pyplot.subplots(figsize = (9, 5))
    grid = numpy.linspace(0.0, 40.0, 161)
    for sheet, group in centring.groupby("sheet"):
        sizes = group["offset_mm"].dropna().abs().to_numpy()
        axis.plot(grid, [(sizes <= value).mean() for value in grid], linewidth = 1.0, linestyle = "--", label = f"demonstrations, {sheet} ({group['episode_id'].nunique()} episodes)")
    for run, colour in zip(SUMMARY_PARAMETERS["robot_runs_reached"], ("black", "tab:red", "tab:purple", "tab:brown")):
        sizes = reached.loc[reached["run"] == run, "offset_mm"].dropna().abs().to_numpy()
        axis.plot(grid, [(sizes <= value).mean() for value in grid], linewidth = 2.0, color = colour, label = f"robot run {run}")
    axis.axvline(limit, color = "tab:green", linewidth = 1.0)
    axis.text(limit + 0.3, 0.03, f"{limit:.0f} mm: the puck hides\nthe line's full width", fontsize = 8, color = "tab:green")
    axis.axvline(SUMMARY_PARAMETERS["off_line_mm"], color = "0.5", linewidth = 1.0)
    axis.text(SUMMARY_PARAMETERS["off_line_mm"] + 0.3, 0.03, "17 mm: the puck no\nlonger touches the line", fontsize = 8, color = "0.4")
    axis.set_xlabel("sideways offset between the puck's centre and the line's middle, mm")
    axis.set_ylabel("share of pictures with at most this offset")
    axis.set_title("How well the puck is kept on the line: demonstrations (dashed) and the robot's four runs that reached the end", fontsize = 9)
    axis.legend(fontsize = 7, loc = "lower right")
    axis.set_xlim(0, 40)
    figure.tight_layout()
    figure.savefig(folder / "centring_distribution.png", dpi = SUMMARY_PARAMETERS["figure_dpi"])
    pyplot.close(figure)
    print("\nwritten: ends_scatter.png, centring_distribution.png")



if __name__ == "__main__":
    main()
