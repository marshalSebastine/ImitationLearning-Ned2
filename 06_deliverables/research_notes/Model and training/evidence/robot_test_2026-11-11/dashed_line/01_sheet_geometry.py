"""
Step 1 of the dashed-line analysis: are sheets 05 and 06 the same shape?

The line of each sheet is drawn from above from the pictures of the robot run on that sheet (run 7 on sheet 05, run 8
on sheet 06) with the fitted forearm camera model, a centre line is laid through the ink, and both sheets are measured
with the same ruler: length, where the crossing is, how long and how wide the loop is, how tightly it bends, and (sheet
06) how long the dashes and the gaps are. The first dash of sheet 06 lay under the puck in run 8, so its length is
read from a demonstration that started at the other end (plan 06-15).

Writes reference_line_05.csv, reference_line_06.csv, 01_sheet_geometry.png and prints the numbers (saved by hand as
01_sheet_geometry.out.txt). No robot, no training.

Run: D:/AI_Project/.venv/Scripts/python.exe 01_sheet_geometry.py > 01_sheet_geometry.out.txt
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, JOINT_NAMES, read_run, top_view_of_line
from line_geometry import (crossing_of, ink_share_of, line_description, map_values, refined_centre_line, registered_line,
                           spline_through, tangents_and_normals)

SHEET_GEOMETRY_PARAMETERS = {
    # the analyst's reading of the zoomed top views, in cm in the robot's base frame of each run: points along the
    # middle of the ink in the order the puck travels; they are only the first guess that the snapping then corrects
    "control_points_cm": {
        "05": [(18.5, 1.35), (20.0, 1.7), (22.0, 2.2), (24.0, 2.8), (26.0, 3.5), (28.0, 4.3), (29.6, 5.0), (30.6, 5.7),
               (31.4, 6.9), (31.8, 8.5), (31.6, 10.0), (30.8, 11.3), (29.5, 12.3), (28.0, 13.0), (26.0, 13.5), (24.5, 13.5),
               (22.5, 13.1), (21.1, 12.3), (20.3, 11.1), (20.0, 9.9), (20.4, 8.8), (21.5, 8.0), (23.0, 7.3), (25.0, 6.6),
               (27.0, 5.9), (29.6, 5.0), (31.5, 4.3), (33.5, 3.4), (35.3, 2.2), (36.8, 0.8), (37.9, -0.8), (38.6, -2.4)],
        "06": [(17.6, 0.2), (20.0, 0.2), (22.6, 0.2), (25.0, 0.2), (26.5, 0.15), (28.3, -0.35), (29.7, -0.85), (31.0, -1.4),
               (32.2, -2.2), (33.1, -3.2), (33.75, -4.3), (33.6, -5.2), (32.8, -5.8), (31.5, -6.2), (30.0, -6.35),
               (28.5, -6.15), (27.1, -5.55), (26.1, -4.6), (25.45, -3.4), (25.1, -2.0), (25.0, -0.5), (25.0, 0.2),
               (25.0, 2.35), (25.05, 5.25), (24.75, 6.9), (24.15, 8.1), (23.3, 9.2), (22.2, 10.05), (20.9, 10.6),
               (19.4, 10.9), (18.5, 10.95)]
    },
    "runs": {"05": "run_7_folder", "06": "run_8_folder"},
    # the demonstration that started at end B of sheet 06, whose pictures show the dash that lay under the puck in run 8
    "other_end_plan": "06-15",
    # a point of the centre line counts as ink when the top view's ink share there is above this
    "ink_share_least": 0.5,
    # the round cap of a solid line's end: the centre line ends half a line width before the ink does
    "line_width_m": 0.010,
    "course_measurements_path": EVIDENCE_FOLDER.parents[3] / "Recording tool" / "2026-11-06_sheet_measurements.json"
}



def runs_of_ink(is_ink: numpy.ndarray) -> list:
    """Return (is ink, first position, length) for every run of equal values along the line."""

    edges = numpy.flatnonzero(numpy.diff(is_ink.astype(int)) != 0) + 1
    starts = numpy.r_[0, edges]
    ends = numpy.r_[edges, len(is_ink)]
    return [(bool(is_ink[start]), int(start), int(end - start)) for start, end in zip(starts, ends)]



def main() -> None:

    settings = SHEET_GEOMETRY_PARAMETERS
    frames = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "frames.csv", low_memory = False)
    course = json.loads(settings["course_measurements_path"].read_text(encoding = "utf-8"))
    figure, axes = pyplot.subplots(1, 3, figsize = (27, 9.5))
    summary = {}
    for axis, sheet in zip(axes[:2], ("05", "06")):
        steps = read_run(ANALYSIS_PARAMETERS[settings["runs"][sheet]])
        view = top_view_of_line(list(steps["frame_file"]), steps[JOINT_NAMES].to_numpy())
        share = ink_share_of(view)
        first_guess = spline_through(numpy.array(settings["control_points_cm"][sheet]) / 100.0, 0.001)
        line = refined_centre_line(first_guess, view)
        ink_along = map_values(share, view, line) > settings["ink_share_least"]

        if sheet == "06":
            # the first dash lay under the puck in run 8: read it from a demonstration that started at the other end
            other = frames[(frames["plan_number"] == settings["other_end_plan"]) & (frames["score"] > 0)].sort_values("row_in_episode")
            other_view = top_view_of_line([str(ANALYSIS_PARAMETERS["raw_folder"] / value) for value in other["frame_path"]],
                                          other[JOINT_NAMES].to_numpy())
            placed = registered_line(line, ink_along, other_view)
            other_ink = map_values(ink_share_of(other_view), other_view, placed["line_m"]) > settings["ink_share_least"]
            first_seen = int(numpy.flatnonzero(ink_along)[0])
            agreement = float((other_ink[first_seen:] == ink_along[first_seen:]).mean())
            print(f"sheet 06: reference laid onto plan {settings['other_end_plan']} (started at end B): ink share under the dashes "
                  f"{placed['ink_share_on_ink_points']:.2f}, under the gaps {placed['ink_share_on_gap_points']:.2f}; "
                  f"ink or gap agrees with run 8 on {100 * agreement:.1f} % of the line run 8 saw")
            # the straight piece before run 8's first seen dash is only a guess of the direction, so the hidden dash is
            # looked for up to 4 mm beside it
            _, other_normal = tangents_and_normals(placed["line_m"])
            beside = [map_values(ink_share_of(other_view), other_view, placed["line_m"] + offset * other_normal) for offset in numpy.arange(-0.004, 0.0041, 0.001)]
            ink_along[:first_seen] = (numpy.max(beside, axis = 0) > settings["ink_share_least"])[:first_seen]
            check_figure, check_axis = pyplot.subplots(figsize = (9, 9))
            other_share = ink_share_of(other_view)
            check_axis.imshow(numpy.where(other_view["seen_votes"] >= 2, other_share, numpy.nan).T, origin = "lower", cmap = "gray_r", vmin = 0, vmax = 1.6,
                              extent = [100 * other_view["x_m"][0], 100 * other_view["x_m"][-1], 100 * other_view["y_m"][0], 100 * other_view["y_m"][-1]])
            check_axis.scatter(100 * placed["line_m"][:, 0], 100 * placed["line_m"][:, 1], c = numpy.where(ink_along, "red", "deepskyblue"), s = 2)
            check_axis.set_aspect("equal")
            check_axis.set_xlim(10, 40)
            check_axis.set_ylim(-25, 5)
            check_axis.grid(True, linewidth = 0.3)
            check_axis.set_title(f"Check: sheet 06's reference line laid onto plan {settings['other_end_plan']} (red = dash, blue = gap)")
            check_figure.savefig(EVIDENCE_FOLDER / "01_check_first_dash.png", dpi = 70)
            pyplot.close(check_figure)
            first_ink = int(numpy.flatnonzero(ink_along)[0])
            line, ink_along = line[first_ink:], ink_along[first_ink:]
        else:
            ink_positions = numpy.flatnonzero(ink_along)
            cap = int(round(0.5 * settings["line_width_m"] / 0.001))
            line, ink_along = line[ink_positions[0] + cap:ink_positions[-1] + 1 - cap], ink_along[ink_positions[0] + cap:ink_positions[-1] + 1 - cap]

        described = line_description(line)
        crossing = crossing_of(line)
        length_cm = 100.0 * described["distance_m"][-1]
        loop = slice(crossing[0], crossing[1] + 1)
        loop_points = line[loop]
        loop_length_cm = 0.1 * (crossing[1] - crossing[0])
        turned_deg = described["direction_deg"][crossing[1]] - described["direction_deg"][crossing[0]]
        # the loop's size along and across the direction in which the line enters it
        entry = numpy.radians(described["direction_deg"][crossing[0]])
        along = (loop_points - loop_points[0]) @ numpy.array([numpy.cos(entry), numpy.sin(entry)])
        across = (loop_points - loop_points[0]) @ numpy.array([-numpy.sin(entry), numpy.cos(entry)])
        radius_cm = 100.0 * described["radius_m"]
        tight = radius_cm[loop] < 3.0
        summary[sheet] = {
            "length_cm": length_cm, "start_to_crossing_cm": 0.1 * crossing[0], "loop_length_cm": loop_length_cm,
            "crossing_to_far_end_cm": length_cm - 0.1 * crossing[1], "loop_turn_deg": float(turned_deg),
            "loop_along_cm": float(100 * (along.max() - along.min())), "loop_across_cm": float(100 * (across.max() - across.min())),
            "loop_smallest_radius_cm": float(radius_cm[loop].min()), "loop_median_radius_cm": float(numpy.median(radius_cm[loop])),
            "loop_cm_tighter_than_3_cm_radius": float(0.1 * tight.sum()),
            "tightest_at_cm_after_crossing": float(0.1 * radius_cm[loop].argmin()),
            "whole_line_smallest_radius_cm": float(radius_cm.min())
        }
        print(f"\nsheet {sheet} (measured on the top view of {settings['runs'][sheet]}; course photo: "
              f"{course[sheet]['line_length_cap_to_cap_cm']} cm cap to cap)")
        for name, value in summary[sheet].items():
            print(f"    {name}: {value:.1f}")
        if sheet == "06":
            pieces = runs_of_ink(ink_along)
            dashes = [0.1 * length for is_ink, _, length in pieces if is_ink]
            gaps = [0.1 * length for is_ink, _, length in pieces if not is_ink]
            print("    dashes along the line, cm: " + ", ".join(f"{value:.1f}" for value in dashes))
            print("    gaps along the line, cm:   " + ", ".join(f"{value:.1f}" for value in gaps))
            inner_dashes = dashes[1:-1]
            print(f"    {len(dashes)} dashes; median dash {numpy.median(inner_dashes):.1f} cm (the piece at the crossing counts once per pass), "
                  f"median gap {numpy.median(gaps):.1f} cm, ink on {100 * ink_along.mean():.0f} % of the line's length")
            summary[sheet].update({"dash_median_cm": float(numpy.median(inner_dashes)), "gap_median_cm": float(numpy.median(gaps)),
                                   "ink_share_of_length": float(ink_along.mean()), "dashes": len(dashes)})

        table = pandas.DataFrame({"distance_cm": 100.0 * described["distance_m"], "x_cm": 100.0 * line[:, 0], "y_cm": 100.0 * line[:, 1],
                                  "direction_deg": described["direction_deg"], "radius_cm": radius_cm, "is_ink": ink_along.astype(int),
                                  "is_in_loop": [int(crossing[0] <= position <= crossing[1]) for position in range(len(line))]})
        table.round(3).to_csv(EVIDENCE_FOLDER / f"reference_line_{sheet}.csv", index = False)

        extent = [100 * view["x_m"][0], 100 * view["x_m"][-1], 100 * view["y_m"][0], 100 * view["y_m"][-1]]
        axis.imshow(numpy.where(view["seen_votes"] >= 2, share, numpy.nan).T, origin = "lower", extent = extent, cmap = "gray_r", vmin = 0, vmax = 1.6)
        drawn = axis.scatter(100 * line[:, 0], 100 * line[:, 1], c = numpy.minimum(radius_cm, 10.0), cmap = "turbo_r", s = 4, vmin = 1.0, vmax = 10.0)
        axis.plot(*(100 * line[crossing[0]]), "o", color = "white", markeredgecolor = "black", markersize = 9)
        axis.plot(*(100 * line[0]), "s", color = "lime", markeredgecolor = "black", markersize = 8)
        axis.plot(*(100 * line[-1]), "X", color = "magenta", markeredgecolor = "black", markersize = 9)
        for position in range(0, len(line), 50):
            axis.annotate(f"{position // 10}", 100 * line[position], fontsize = 7, color = "tab:blue", xytext = (3, 3), textcoords = "offset points")
        axis.set_aspect("equal")
        axis.set_xlim(15, 41)
        axis.set_ylim(-9, 17) if sheet == "06" else axis.set_ylim(-6, 20)
        axis.grid(True, linewidth = 0.3)
        axis.set_xlabel("x from the base axis (cm)")
        axis.set_ylabel("y (cm), left is up")
        axis.set_title(f"Sheet {sheet} as it lay in run {'7' if sheet == '05' else '8'}: line {length_cm:.1f} cm, loop {loop_length_cm:.1f} cm long,\n"
                       f"tightest bend {summary[sheet]['loop_smallest_radius_cm']:.1f} cm radius (numbers = cm along the line)")
        figure.colorbar(drawn, ax = axis, shrink = 0.7, label = "bend radius (cm), 10 = 10 or more")
        summary[sheet]["radius_profile"] = (0.1 * (numpy.arange(len(line)) - crossing[0]), radius_cm, crossing)

    axis = axes[2]
    for sheet, colour in (("05", "tab:blue"), ("06", "tab:red")):
        distance, radius, crossing = summary[sheet].pop("radius_profile")
        axis.plot(distance, numpy.minimum(radius, 12.0), color = colour, label = f"sheet {sheet}")
        axis.axvline(0.1 * (crossing[1] - crossing[0]), color = colour, linestyle = ":", linewidth = 1)
    axis.axvline(0.0, color = "black", linestyle = ":", linewidth = 1)
    axis.set_xlabel("cm along the line, 0 = first pass of the crossing (dotted: the passes)")
    axis.set_ylabel("bend radius of the centre line (cm), 12 = 12 or more")
    axis.set_title("How tightly each line bends (2 cm window)")
    axis.grid(True, linewidth = 0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "01_sheet_geometry.png", dpi = 80)
    (EVIDENCE_FOLDER / "01_sheet_geometry_summary.json").write_text(json.dumps(summary, indent = 1), encoding = "utf-8")



if __name__ == "__main__":
    main()
