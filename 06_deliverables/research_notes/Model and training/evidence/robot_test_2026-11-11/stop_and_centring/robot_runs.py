"""
The two quality requirements measured on the ten model runs of 11.11.26 (03_data/model_runs): at every step where the
puck stands and how far to the side of the line's middle; at the stop how the puck stands on the line's round end.

Run from this folder:    python robot_runs.py
Writes robot_runs_steps.csv, robot_runs_ends.csv and the figures robot_runs_maps.png, robot_runs_offsets.png and
robot_runs_ends.png; the printed tables are kept in robot_runs.out.txt. Nothing connects to the robot.
"""

from pathlib import Path
from typing import Dict, List

import numpy
import pandas
from matplotlib import pyplot

import demonstration_ends
import picture_measures
import run_measures

from config import MODEL_MODE_PARAMETERS


ROBOT_RUN_PARAMETERS = {
    # the ten runs in the order of the brief's table: folder, model, sheet, and whether the run reached the line's end
    # and stopped there by itself (the brief's table)
    "runs": [
        ("2026-11-11_172021_check_no_puck", "A", "01", True),
        ("2026-11-11_172421_E_sheet01_run1", "E", "01", True),
        ("2026-11-11_172743_E_sheet02_run1", "E", "02", False),
        ("2026-11-11_173036_E_sheet02_run2", "E", "02", True),
        ("2026-11-11_173549_E_sheet03_run1", "E", "03", False),
        ("2026-11-11_174108_E_sheet04_run1", "E", "04", False),
        ("2026-11-11_174448_E_sheet05_run1", "E", "05", True),
        ("2026-11-11_174936_E_sheet06_run1", "E", "06", False),
        ("2026-11-11_175203_E_sheet03_run2_no_hoodie", "E", "03", False),
        ("2026-11-11_175544_B_team5_sheet03", "B", "03", False)
    ],
    # the puck hides the line's full width while its centre is within its radius minus half the line's width of the
    # line's middle (12 - 5 = 7 mm); the same distance decides whether the round end is hidden completely
    "cover_limit_mm": 7.0,
    "last_steps_shown": 6,
    "figure_dpi": 110
}



def run_summary(rows: pandas.DataFrame) -> Dict:
    """Numbers on the sideways offset of one run: over the steps where a line was found beside the puck."""

    offsets = rows["offset_mm"].dropna()
    limit = ROBOT_RUN_PARAMETERS["cover_limit_mm"]
    if offsets.empty:
        return {"steps_measured": 0}
    return {
        "steps_measured": int(len(offsets)),
        "steps_on_one_clean_stretch": int((rows["is_clean"] & rows["is_aligned"]).sum()),
        "median_abs_mm": float(offsets.abs().median()),
        "mean_signed_mm": float(offsets.mean()),
        "p90_abs_mm": float(offsets.abs().quantile(0.9)),
        "largest_abs_mm": float(offsets.abs().max()),
        "share_within_2_5_mm": float((offsets.abs() <= 2.5).mean()),
        "share_within_5_mm": float((offsets.abs() <= 5.0).mean()),
        "share_within_cover_limit": float((offsets.abs() <= limit).mean()),
        "share_puck_off_the_line": float((offsets.abs() > 17.0).mean())
    }



def draw_map(
        axis,
        measured: Dict,
        rows: pandas.DataFrame,
        tips_xy: numpy.ndarray,
        title: str
) -> None:
    """Draw one run from above: the line every picture voted for, the tip's path and the puck's places, coloured by offset."""

    line_map = measured["line_map"]
    line_votes = measured["all_line_votes"].astype(float)
    paper_votes = measured["all_paper_votes"].astype(float)
    is_line = (line_votes >= 2) & (line_votes >= 0.5 * (line_votes + paper_votes))
    shown = numpy.full(is_line.shape + (3,), 0.55)
    shown[paper_votes > 0] = 0.93
    shown[is_line] = 0.1
    extent = [100.0 * line_map.y_range[1], 100.0 * line_map.y_range[0], 100.0 * line_map.x_range[0], 100.0 * line_map.x_range[1]]
    axis.imshow(shown, extent = extent, interpolation = "nearest")
    axis.plot(100.0 * tips_xy[:, 1], 100.0 * tips_xy[:, 0], color = "tab:blue", linewidth = 0.8, label = "magnet tip")
    has_puck = rows["puck_x"].notna()
    has_offset = has_puck & rows["offset_mm"].notna()
    axis.scatter(100.0 * rows.loc[has_puck & ~has_offset, "puck_y"], 100.0 * rows.loc[has_puck & ~has_offset, "puck_x"], c = "white",
                 s = 14, edgecolors = "black", linewidths = 0.3, zorder = 3)
    scatter = axis.scatter(100.0 * rows.loc[has_offset, "puck_y"], 100.0 * rows.loc[has_offset, "puck_x"], c = rows.loc[has_offset, "offset_mm"].abs(),
                           cmap = "autumn_r", vmin = 0.0, vmax = 20.0, s = 14, edgecolors = "black", linewidths = 0.3, zorder = 3)
    axis.set_title(title, fontsize = 9)
    axis.set_xlabel("y in cm (the base's left is on the left)", fontsize = 7)
    axis.set_ylabel("x in cm (away from the base)", fontsize = 7)
    axis.tick_params(labelsize = 7)
    return scatter



def draw_end(
        axis,
        measured: Dict,
        end: Dict,
        rows: pandas.DataFrame,
        title: str
) -> None:
    """Draw the end of one run from above: the line's end as the last pictures voted it, the puck's foot and its last places."""

    line_map = measured["line_map"]
    is_line = line_map.line_cells()
    shown = numpy.full(is_line.shape + (3,), 0.93)
    shown[is_line] = 0.1
    extent = [100.0 * line_map.y_range[1], 100.0 * line_map.y_range[0], 100.0 * line_map.x_range[0], 100.0 * line_map.x_range[1]]
    axis.imshow(shown, extent = extent, interpolation = "nearest")
    last = rows.dropna(subset = ["puck_x"]).iloc[-1]
    angles = numpy.linspace(0.0, 2.0 * numpy.pi, 90)
    radius_cm = 100.0 * picture_measures.MEASURE_PARAMETERS["puck_radius_m"]
    axis.plot(100.0 * last["puck_y"] + radius_cm * numpy.sin(angles), 100.0 * last["puck_x"] + radius_cm * numpy.cos(angles),
              color = "red", linewidth = 1.5, label = "puck's foot at the stop")
    recent = rows.dropna(subset = ["puck_x"]).iloc[-ROBOT_RUN_PARAMETERS["last_steps_shown"]:]
    axis.plot(100.0 * recent["puck_y"], 100.0 * recent["puck_x"], color = "red", marker = "o", markersize = 3, linewidth = 0.8,
              label = "puck's centre, last steps")
    axis.plot(100.0 * end["end_centre_xy"][1], 100.0 * end["end_centre_xy"][0], marker = "+", color = "cyan", markersize = 10,
              label = "centre of the round end")
    axis.set_xlim(100.0 * last["puck_y"] + 4.0, 100.0 * last["puck_y"] - 4.0)
    axis.set_ylim(100.0 * last["puck_x"] - 4.0, 100.0 * last["puck_x"] + 4.0)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize = 8)
    axis.tick_params(labelsize = 7)



def main() -> None:

    run_folder = Path(MODEL_MODE_PARAMETERS["run_folder"])
    camera_parameters = run_measures.camera_parameters_for("robot_test")
    print("camera: principal point", numpy.round(camera_parameters["principal_point_px"], 1).tolist(),
          "(the fitted one moved by the shift of camera_shift.json, period robot_test)")
    all_rows: List[pandas.DataFrame] = []
    ends = []
    kept = {}
    for number, (name, model, sheet, is_end_reached) in enumerate(ROBOT_RUN_PARAMETERS["runs"], start = 1):
        steps = run_measures.robot_run_steps(run_folder / name)
        measured = run_measures.measure_steps(steps, camera_parameters)
        rows = measured["rows"]
        rows.insert(0, "run", number)
        for column in ("model_tip_forward_cm", "model_tip_left_cm", "model_twist_deg", "model_stop_probability", "tip_x", "tip_y", "action"):
            rows[column] = steps[column].to_numpy()[rows["step"].to_numpy()]
        all_rows.append(rows)
        summary = run_summary(rows)
        end = measured["end"]
        print(f"\nrun {number} ({name[11:]}; model {model}, sheet {sheet}; {len(steps)} pictures; reached the end: {is_end_reached})")
        print("  puck found in", int(rows["puck_x"].notna().sum()), "pictures; median fit quality", round(float(rows["fit_quality"].median()), 3))
        print("  sideways offset, steps with the line beside the puck:", {key: (round(value, 2) if isinstance(value, float) else value) for key, value in summary.items()})
        entry = {"run": number, "folder": name, "model": model, "sheet": sheet, "is_end_reached": is_end_reached}
        entry.update(summary)
        if end is not None and is_end_reached:
            entry.update({"end_short_mm": 1000.0 * end["short_m"], "end_aside_mm": 1000.0 * end["aside_m"],
                          "end_centre_gap_mm": 1000.0 * end["centre_gap_m"], "end_uncovered_mm": 1000.0 * end["uncovered_m"],
                          "end_ink_beyond_mm2": end["ink_beyond_mm2"], "end_ink_seen_ahead_mm2": demonstration_ends.ink_seen_ahead(measured, end),
                          "is_end_seen": end["is_end_seen"], "line_width_in_map_mm": 1000.0 * end["line_width_m"]})
            print(f"  at the stop: puck centre {1000.0 * end['short_m']:+.1f} mm short of the round end's centre (negative: beyond it), "
                  f"{1000.0 * end['aside_m']:+.1f} mm to the left of the line (negative: right); centres {1000.0 * end['centre_gap_m']:.1f} mm apart; "
                  f"the end's rim sticks out {1000.0 * end['uncovered_m']:+.1f} mm at its worst point (0 or less: covered completely); "
                  f"line width in the map {1000.0 * end['line_width_m']:.1f} mm; end seen: {end['is_end_seen']}")
            print(f"  ink ahead of the puck's centre and outside its foot, seen from above: {entry['end_ink_beyond_mm2']:.0f} square mm; "
                  f"ink the stop picture itself shows ahead of the puck: {entry['end_ink_seen_ahead_mm2']:.0f} square mm")
            # where the puck stood at each of the last steps, against the end
            direction = end["direction_xy"]
            left = numpy.array([-direction[1], direction[0]])
            print("  last steps (short of the end's centre along the line, left of the line, in mm; the model's answer in cm):")
            for _, row in rows.dropna(subset = ["puck_x"]).iloc[-ROBOT_RUN_PARAMETERS["last_steps_shown"]:].iterrows():
                gap = end["end_centre_xy"] - numpy.array([row["puck_x"], row["puck_y"]])
                print(f"    step {int(row['step']):3d}: short {1000.0 * float(gap @ direction):+6.1f}, left {-1000.0 * float(gap @ left):+6.1f}; "
                      f"model forward {row['model_tip_forward_cm']:+.2f}, left {row['model_tip_left_cm']:+.2f} "
                      f"(length {numpy.hypot(row['model_tip_forward_cm'], row['model_tip_left_cm']):.2f}), stop chance {row['model_stop_probability']:.3f}, {row['action']}")
        ends.append(entry)
        kept[number] = (measured, rows, steps[["tip_x", "tip_y"]].to_numpy(dtype = float), end, name, model, sheet, is_end_reached)

    steps_table = pandas.concat(all_rows, ignore_index = True)
    steps_table.to_csv(run_measures.EVIDENCE_FOLDER / "robot_runs_steps.csv", index = False, float_format = "%.5f")
    ends_table = pandas.DataFrame(ends)
    ends_table.to_csv(run_measures.EVIDENCE_FOLDER / "robot_runs_ends.csv", index = False, float_format = "%.3f")

    # two checks of the measure itself, over the steps where the puck followed one clean stretch of line
    reached_runs = [entry["run"] for entry in ends if entry["is_end_reached"]]
    is_good = steps_table["is_clean"] & steps_table["is_aligned"] & steps_table["offset_mm"].notna()
    print()
    print("check A, the camera model: the offset against the votes of the last 12 pictures and against the votes of every picture of the run")
    for label, part in (("the four runs that reached the end", steps_table[is_good & steps_table["run"].isin(reached_runs)]),
                        ("all ten runs", steps_table[is_good])):
        part = part.dropna(subset = ["whole_run_offset_mm"])
        difference = part["whole_run_offset_mm"] - part["offset_mm"]
        print(f"  {label}: {len(part)} steps, difference median {difference.median():+.2f} mm, median size {difference.abs().median():.2f} mm, "
              f"90th percentile {difference.abs().quantile(0.9):.2f} mm, largest {difference.abs().max():.1f} mm")
    print()
    print("check B, one picture alone against the votes of the last 12 pictures")
    for label, part in (("the four runs that reached the end", steps_table[is_good & steps_table["run"].isin(reached_runs)]),
                        ("all ten runs", steps_table[is_good])):
        with_puck = len(part)
        part = part.dropna(subset = ["single_offset_mm"])
        usable = part[part["is_single_clean"] & part["is_single_aligned"]]
        difference = usable["single_offset_mm"] - usable["offset_mm"]
        print(f"  {label}: {with_puck} steps; one picture gives an offset in {len(part)} and a clean one in {len(usable)}; "
              f"difference median {difference.median():+.2f} mm, median size {difference.abs().median():.2f} mm, "
              f"90th percentile {difference.abs().quantile(0.9):.2f} mm, over 5 mm in {int((difference.abs() > 5.0).sum())} steps, largest {difference.abs().max():.1f} mm")
        for least, most in ((0.0, 10.0), (10.0, 30.0), (30.0, 80.0)):
            ranged = usable[(usable["single_extrapolated_mm"] >= least) & (usable["single_extrapolated_mm"] < most)]
            if len(ranged):
                ranged_difference = (ranged["single_offset_mm"] - ranged["offset_mm"]).abs()
                print(f"    line followed back {least:.0f} to {most:.0f} mm: {len(ranged):3d} steps, median size {ranged_difference.median():.2f} mm, 90th percentile {ranged_difference.quantile(0.9):.2f} mm")

    figure, axes = pyplot.subplots(2, 5, figsize = (20, 10))
    for number, axis in zip(kept, axes.ravel()):
        measured, rows, tips_xy, end, name, model, sheet, is_end_reached = kept[number]
        scatter = draw_map(axis, measured, rows, tips_xy, f"run {number}: model {model}, sheet {sheet}" + ("" if is_end_reached else " (line lost)"))
    figure.colorbar(scatter, ax = axes, shrink = 0.6, label = "puck's centre beside the line's middle, mm (red: 20 or more; white: no line found near the puck)")
    figure.suptitle("The ten model runs of 11.11.26 from above: the line as the pictures saw it (black), the magnet tip (blue) and the puck's centre at every picture")
    figure.savefig(run_measures.EVIDENCE_FOLDER / "robot_runs_maps.png", dpi = ROBOT_RUN_PARAMETERS["figure_dpi"], bbox_inches = "tight")
    pyplot.close(figure)

    reached = [number for number in kept if kept[number][7]]
    figure, axes = pyplot.subplots(len(reached), 1, figsize = (10, 2.6 * len(reached)), sharex = False)
    limit = ROBOT_RUN_PARAMETERS["cover_limit_mm"]
    for number, axis in zip(reached, axes):
        rows = kept[number][1]
        axis.axhspan(-limit, limit, color = "tab:green", alpha = 0.15, label = f"puck hides the line's full width (within {limit:.0f} mm)")
        axis.axhline(0.0, color = "black", linewidth = 0.6)
        axis.plot(rows["step"], rows["offset_mm"], marker = "o", markersize = 3, color = "tab:red", label = "against the map of the last 12 pictures")
        axis.plot(rows["step"], rows["single_offset_mm"], marker = "x", markersize = 3, linewidth = 0.6, color = "tab:blue", label = "from the one picture alone")
        axis.set_ylabel("left of the line, mm", fontsize = 8)
        axis.set_title(f"run {number}: model {kept[number][5]}, sheet {kept[number][6]}", fontsize = 9)
        axis.set_ylim(-25, 25)
        axis.tick_params(labelsize = 7)
    axes[0].legend(fontsize = 7, ncol = 3, loc = "upper left")
    axes[-1].set_xlabel("picture (step)")
    figure.suptitle("Sideways offset of the puck's centre from the line's middle, the four runs that reached the end", fontsize = 10)
    figure.tight_layout()
    figure.savefig(run_measures.EVIDENCE_FOLDER / "robot_runs_offsets.png", dpi = ROBOT_RUN_PARAMETERS["figure_dpi"])
    pyplot.close(figure)

    figure, axes = pyplot.subplots(1, len(reached), figsize = (4.2 * len(reached), 4.6))
    for number, axis in zip(reached, axes):
        measured, rows, _, end, name, model, sheet, _ = kept[number]
        if end is None:
            continue
        draw_end(axis, measured, end, rows, f"run {number} (model {model}, sheet {sheet}): centres {1000.0 * end['centre_gap_m']:.0f} mm apart,\n"
                                            f"{1000.0 * end['short_m']:+.0f} mm short, {1000.0 * end['aside_m']:+.0f} mm left; rim out {1000.0 * end['uncovered_m']:+.0f} mm")
    axes[0].legend(fontsize = 6, loc = "lower left")
    figure.suptitle("The stop from above (x up, y to the left, cm): the line's end and the puck's foot", fontsize = 10)
    figure.tight_layout()
    figure.savefig(run_measures.EVIDENCE_FOLDER / "robot_runs_ends.png", dpi = ROBOT_RUN_PARAMETERS["figure_dpi"])
    pyplot.close(figure)
    print("\nwritten: robot_runs_steps.csv, robot_runs_ends.csv, robot_runs_maps.png, robot_runs_offsets.png, robot_runs_ends.png")



if __name__ == "__main__":
    main()
