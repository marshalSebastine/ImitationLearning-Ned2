"""
Verifier check 4: where the puck was beside the line in robot runs 8 (sheet 06) and 7 (sheet 05), measured again.

My own pipeline: my colour rules for ink and puck, my own top view of each run's pictures, the Recording session's
course-photo centre line laid onto it (not the analyst's reference line), and the puck's distance beside that line.
Three top views per run are compared: all pictures, only pictures with the forearm twisted under 45 degrees, only
pictures twisted over 60 degrees; the late steps of run 8 are also measured against the line placed from the
far-twisted pictures alone, so that an error of the camera model that grows with the twist would show.

Run:  D:/AI_Project/.venv/Scripts/python.exe v04_run_tracks.py > v04_run_tracks.out.txt
Writes v04_run_tracks.csv and v04_run_tracks.png. About one minute. Nothing connects to the robot.
"""

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy
import pandas

from verify_common import (VERIFY_FOLDER, VERIFY_PARAMETERS, along_and_beside, colour_masks, crossing_indices, ink_cells,
                           pixels_on_table, puck_pixel, register_line, sheet_line, top_view)

cv2.setNumThreads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]


def main() -> None:

    analyst_tracks = pandas.read_csv(VERIFY_FOLDER.parent / "tracks.csv")
    figure, axes = pyplot.subplots(1, 2, figsize = (18, 9))
    all_rows = []
    for axis, (run_label, folder_key, sheet) in zip(axes, (("run_8", "run_8_folder", "06"), ("run_7", "run_7_folder", "05"))):
        folder = VERIFY_PARAMETERS["model_runs_folder"] / VERIFY_PARAMETERS[folder_key]
        steps = pandas.read_csv(folder / "steps.csv")
        pictures = [str(folder / name) for name in steps["frame"]]
        joints = steps[JOINT_NAMES].to_numpy()
        twist_deg = numpy.degrees(joints[:, 3])
        line_sheet = sheet_line(sheet)

        print(f"\n===== {run_label} (sheet {sheet}): {len(steps)} pictures, forearm {twist_deg.min():.0f} to {twist_deg.max():.0f} deg =====")
        views, placements = {}, {}
        for name, chosen in (("all", numpy.arange(len(steps))), ("under_45", numpy.flatnonzero(numpy.abs(twist_deg) < 45.0)),
                             ("over_60", numpy.flatnonzero(numpy.abs(twist_deg) > 60.0))):
            if len(chosen) < 5:
                continue
            views[name] = top_view([pictures[position] for position in chosen], joints[chosen])
            first_guess = None
            if "all" in placements:
                first = placements["all"]
                first_guess = (first["placement"][0], first["placement"][1], first["placement"][2], first["is_mirrored"])
            placements[name] = register_line(line_sheet, views[name], first_guess = first_guess)
            placed = placements[name]
            print(f"  top view '{name}': {len(chosen)} pictures, {placed['ink_cells']} ink cells; line placed at"
                  f" {numpy.degrees(placed['placement'][0]) % 360:.1f} deg, ({100 * placed['placement'][1]:.2f}, {100 * placed['placement'][2]:.2f}) cm,"
                  f" mirrored {placed['is_mirrored']}; ink cells lie {placed['ink_to_line_mm']:.1f} mm (median) and"
                  f" {placed['ink_to_line_90_mm']:.1f} mm (90 %) from the line; {100 * placed['line_seen_share']:.0f} % of the line seen,"
                  f" ink within 3 mm of {100 * placed['line_on_ink_share']:.0f} % of the seen line points")
        if "under_45" in placements and "over_60" in placements:
            low, high = placements["under_45"], placements["over_60"]
            shift = numpy.hypot(*(high["line_m"] - low["line_m"]).T)
            print(f"  the line placed from far-twisted pictures lies {1000 * numpy.median(shift):.1f} mm (median), {1000 * shift.max():.1f} mm (most)"
                  f" from the line placed from little-twisted pictures; turned by {numpy.degrees(high['placement'][0] - low['placement'][0]):.2f} deg")

        # the puck
        puck_xy = numpy.full((len(steps), 2), numpy.nan)
        puck_px = numpy.full((len(steps), 3), numpy.nan)
        for position, picture_path in enumerate(pictures):
            found = puck_pixel(colour_masks(cv2.imread(picture_path))["puck"])
            if found is None:
                continue
            puck_px[position] = found
            puck_xy[position] = pixels_on_table(numpy.array([[found[0], found[1]]]), joints[position], VERIFY_PARAMETERS["puck_point_height_m"])[0]
        puck_xy_top = numpy.full((len(steps), 2), numpy.nan)
        for position in range(len(steps)):
            if numpy.isfinite(puck_px[position, 0]):
                puck_xy_top[position] = pixels_on_table(puck_px[position:position + 1, :2], joints[position], 0.029)[0]
        print(f"  puck found in {int(numpy.isfinite(puck_xy[:, 0]).sum())} of {len(steps)} pictures; blob size {numpy.nanmin(puck_px[:, 2]):.0f} to {numpy.nanmax(puck_px[:, 2]):.0f} px")
        print(f"  how much the assumed height of the blob's centre matters: at 2.9 cm instead of 1.45 cm the puck moves"
              f" {100 * numpy.nanmedian(numpy.hypot(*(puck_xy_top - puck_xy).T)):.2f} cm (median), {100 * numpy.nanmax(numpy.hypot(*(puck_xy_top - puck_xy).T)):.2f} cm (most)")

        results = {}
        for name, placed in placements.items():
            line = placed["line_m"]
            index, beside = along_and_beside(line, puck_xy)
            valid = index >= 0
            if numpy.corrcoef(numpy.flatnonzero(valid), index[valid])[0, 1] < 0 or index[valid][-1] < index[valid][0]:
                line = line[::-1].copy()
                index, beside = along_and_beside(line, puck_xy)
            crossing = crossing_indices(line)
            results[name] = {"line": line, "index": index, "beside": beside, "crossing": crossing}
            placements[name]["line_travel"] = line
        main_result = results["all"]
        crossing = main_result["crossing"]
        line = main_result["line"]
        print(f"  line length {0.1 * (len(line) - 1):.1f} cm; crossing passed at {0.1 * crossing[0]:.1f} and {0.1 * crossing[1]:.1f} cm from the start end:"
              f" loop {0.1 * (crossing[1] - crossing[0]):.1f} cm, after the loop {0.1 * (len(line) - 1 - crossing[1]):.1f} cm")

        analyst = analyst_tracks[analyst_tracks["recording"] == run_label].set_index("row")
        table = pandas.DataFrame({
            "step": steps["step"], "joint_4": twist_deg.round(1),
            "along_cm": 0.1 * (main_result["index"] - crossing[0]),
            "beside_all_cm": 100 * main_result["beside"],
            "beside_under45_cm": 100 * results["under_45"]["beside"] if "under_45" in results else numpy.nan,
            "beside_over60_cm": 100 * results["over_60"]["beside"] if "over_60" in results else numpy.nan,
            "analyst_along_cm": analyst["puck_along_cm"].reindex(steps["step"]).to_numpy(),
            "analyst_beside_cm": analyst["puck_beside_cm"].reindex(steps["step"]).to_numpy(),
            "puck_x_cm": 100 * puck_xy[:, 0], "puck_y_cm": 100 * puck_xy[:, 1],
            "analyst_puck_x_cm": analyst["puck_x_cm"].reindex(steps["step"]).to_numpy(),
            "analyst_puck_y_cm": analyst["puck_y_cm"].reindex(steps["step"]).to_numpy()
        })
        table["puck_gap_cm"] = numpy.hypot(table["puck_x_cm"] - table["analyst_puck_x_cm"], table["puck_y_cm"] - table["analyst_puck_y_cm"])
        table["beside_gap_cm"] = table["beside_all_cm"] - table["analyst_beside_cm"]
        table.insert(0, "run", run_label)
        all_rows.append(table)
        print(table.drop(columns = ["run"]).round(2).to_string(index = False))
        beside = table["beside_all_cm"].abs()
        print(f"  mine:    puck centre within 0.5 cm of the line in {int((beside <= 0.5).sum())} of {int(beside.notna().sum())} pictures,"
              f" further than 1.0 cm in {int((beside > 1.0).sum())}; largest {beside.max():.1f} cm; median {beside.median():.2f} cm")
        analyst_beside = table["analyst_beside_cm"].abs()
        print(f"  analyst: within 0.5 cm in {int((analyst_beside <= 0.5).sum())} of {int(analyst_beside.notna().sum())},"
              f" further than 1.0 cm in {int((analyst_beside > 1.0).sum())}; largest {analyst_beside.max():.1f} cm; median {analyst_beside.median():.2f} cm")
        print(f"  my puck positions against the analyst's: median gap {table['puck_gap_cm'].median():.2f} cm, most {table['puck_gap_cm'].max():.2f} cm;"
              f" beside the line: median absolute difference {table['beside_gap_cm'].abs().median():.2f} cm, most {table['beside_gap_cm'].abs().max():.2f} cm")

        is_ink = ink_cells(views["all"])
        axis.imshow(is_ink.T, origin = "lower", cmap = "gray_r", vmin = 0, vmax = 2.5,
                    extent = [100 * views["all"]["x_m"][0], 100 * views["all"]["x_m"][-1], 100 * views["all"]["y_m"][0], 100 * views["all"]["y_m"][-1]])
        axis.plot(100 * line[:, 0], 100 * line[:, 1], "-", color = "deepskyblue", linewidth = 1.0, label = "course-photo centre line, placed")
        axis.plot(100 * puck_xy[:, 0], 100 * puck_xy[:, 1], ".-", color = "red", markersize = 4, linewidth = 0.8, label = "puck (mine)")
        axis.plot(table["analyst_puck_x_cm"], table["analyst_puck_y_cm"], "-", color = "limegreen", linewidth = 0.8, label = "puck (analyst)")
        axis.plot(100 * steps["tip_x"], 100 * steps["tip_y"], "-", color = "orange", linewidth = 0.8, label = "magnet tip (log)")
        for step in range(0, len(steps), 5):
            if numpy.isfinite(puck_xy[step, 0]):
                axis.annotate(str(step), (100 * puck_xy[step, 0], 100 * puck_xy[step, 1]), fontsize = 7, color = "darkred")
        axis.plot(*(100 * line[0]), "s", color = "yellow", markeredgecolor = "black")
        axis.set_xlim(100 * line[:, 0].min() - 8, 100 * line[:, 0].max() + 8)
        axis.set_ylim(100 * line[:, 1].min() - 8, 100 * line[:, 1].max() + 8)
        axis.set_aspect("equal")
        axis.set_xlabel("x in cm (robot base frame)")
        axis.set_ylabel("y in cm")
        axis.set_title(f"{run_label}, sheet {sheet}: my top view, the placed line and the puck's path")
        axis.legend(fontsize = 8, loc = "upper left")
    figure.tight_layout()
    figure.savefig(VERIFY_FOLDER / "v04_run_tracks.png", dpi = 70)
    pandas.concat(all_rows).round(3).to_csv(VERIFY_FOLDER / "v04_run_tracks.csv", index = False)



if __name__ == "__main__":
    main()
