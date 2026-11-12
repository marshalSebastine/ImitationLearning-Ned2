"""
Step 3 of the dashed-line analysis: robot runs 7 (sheet 05) and 8 (sheet 06) against the demonstrations of the same
sheet from the same end, in the sheet's own frame.

From tracks.csv (step 2): where the puck was beside the line at every centimetre along it, how far the forearm was
twisted there, how many pictures the demonstrations spent there, and the step the model asked for against the steps
the demonstrations took at the same place. The step where a run leaves the demonstrations is the first step from
which the puck stays further beside the line than the demonstrations' outer tenth (10th to 90th percentile) at that
place.

Writes 03_run_against_demonstrations.png and prints the tables (03_run_against_demonstrations.out.txt).

Run: D:/AI_Project/.venv/Scripts/python.exe 03_run_against_demonstrations.py > 03_run_against_demonstrations.out.txt
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy
import pandas

from dashed_common import EVIDENCE_FOLDER
from line_geometry import moved_line

COMPARISON_PARAMETERS = {
    "pairs": [{"sheet": 6, "run": "run_8", "start_end": "A"}, {"sheet": 5, "run": "run_7", "start_end": "B"}],
    # demonstrations that start at the line's start and drive the whole line
    "whole_line_kinds": ["STD", "OL1", "OR1", "OL1.5", "OR1.5", "TL10", "TR10", "TL15", "TR15", "X1"],
    # a recording's line counts as placed when this share of the reference's ink points, where seen, lies on ink
    "placed_ink_share": 0.75,
    "bin_cm": 1.0,
    # the puck's centre is "on the line" within half the line's width, and "off the line" beyond the line's width
    "on_line_cm": 0.5,
    "off_line_cm": 1.0,
    # the demonstrations' steps compared with a run's step: those whose puck was within this many cm along the line
    "nearby_cm": 0.75,
    # the walk round the puck: bins next to the loop's busiest centimetre with at least this many pictures per demonstration
    "busy_pictures": 2.5
}



def prepared_tracks() -> pandas.DataFrame:
    """Return tracks.csv of the placed recordings with the step's direction, length and the tip's real move added."""

    tracks = pandas.read_csv(EVIDENCE_FOLDER / "tracks.csv")
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv")
    placed = registrations.loc[registrations["ink_share_on_seen_ink_points"] >= COMPARISON_PARAMETERS["placed_ink_share"], "recording"]
    tracks = tracks[tracks["recording"].isin(placed)].sort_values(["recording", "row"]).reset_index(drop = True)
    tracks["angle_deg"] = numpy.degrees(numpy.arctan2(tracks["left_cm"], tracks["forward_cm"]))
    tracks["length_cm"] = numpy.hypot(tracks["left_cm"], tracks["forward_cm"])
    is_same = tracks["recording"].shift(-1) == tracks["recording"]
    move_x = (tracks["tip_x_cm"].shift(-1) - tracks["tip_x_cm"]).where(is_same)
    move_y = (tracks["tip_y_cm"].shift(-1) - tracks["tip_y_cm"]).where(is_same)
    tracks["tip_move_cm"] = numpy.hypot(move_x, move_y)
    # the direction the step asks for on the table: the picture's heading plus the step's angle in the picture
    tracks["step_direction_deg"] = tracks["heading_deg"] + tracks["angle_deg"]
    tracks["step_beside_line_deg"] = (tracks["step_direction_deg"] - tracks["line_direction_deg"] + 180.0) % 360.0 - 180.0
    tracks["camera_beside_line_deg"] = (tracks["heading_deg"] - tracks["line_direction_deg"] + 180.0) % 360.0 - 180.0
    puck_x = (tracks["puck_x_cm"].shift(-1) - tracks["puck_x_cm"]).where(is_same)
    puck_y = (tracks["puck_y_cm"].shift(-1) - tracks["puck_y_cm"]).where(is_same)
    tracks["puck_move_cm"] = numpy.hypot(puck_x, puck_y)
    return tracks



def in_sheet_frame(
        points_cm: numpy.ndarray,
        registration: pandas.Series,
        reference_mean_cm: numpy.ndarray
) -> numpy.ndarray:
    """Return table points of a recording in the frame of the sheet's reference line (the robot run's frame)."""

    centred = points_cm - numpy.array([registration["shift_x_cm"], registration["shift_y_cm"]])
    return moved_line(centred, -registration["angle_deg"], (0.0, 0.0)) + reference_mean_cm



def main() -> None:

    settings = COMPARISON_PARAMETERS
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 300)
    tracks = prepared_tracks()
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv").set_index("recording")
    figure, axes = pyplot.subplots(2, 4, figsize = (30, 14))
    for row_of_axes, pair in zip(axes, settings["pairs"]):
        sheet_name = f"{pair['sheet']:02d}"
        reference = pandas.read_csv(EVIDENCE_FOLDER / f"reference_line_{sheet_name}.csv")
        reference_points = reference[["x_cm", "y_cm"]].to_numpy()
        the_run = tracks[tracks["recording"] == pair["run"]].copy()
        demonstrations = tracks[(tracks["sheet"] == pair["sheet"]) & (tracks["start_end"] == pair["start_end"])
                                & tracks["start_kind"].isin(settings["whole_line_kinds"]) & (tracks["score"] > 0)].copy()
        demonstrations["bin"] = numpy.floor(demonstrations["puck_along_cm"] / settings["bin_cm"]) * settings["bin_cm"]
        the_run["bin"] = numpy.floor(the_run["puck_along_cm"] / settings["bin_cm"]) * settings["bin_cm"]
        episodes = demonstrations["plan"].nunique()
        print(f"\n===== sheet {sheet_name}: {pair['run']} against {episodes} demonstrations from end {pair['start_end']} "
              f"({len(demonstrations)} pictures; kinds {sorted(demonstrations['start_kind'].unique())}) =====")

        # the puck beside the line: per demonstration the median in each centimetre, then the spread over demonstrations
        per_episode = demonstrations.groupby(["bin", "plan"])["puck_beside_cm"].median().unstack()
        corridor = pandas.DataFrame({
            "demonstrations": per_episode.notna().sum(axis = 1), "least": per_episode.min(axis = 1),
            "tenth": per_episode.quantile(0.1, axis = 1), "median": per_episode.median(axis = 1),
            "ninetieth": per_episode.quantile(0.9, axis = 1), "most": per_episode.max(axis = 1)})
        dwell = demonstrations.groupby("bin").agg(
            pictures_per_demonstration = ("row", lambda values: len(values) / episodes),
            twist_per_demonstration_deg = ("twist_deg", lambda values: values.sum() / episodes),
            joint_4_median_deg = ("joint_4_deg", "median"), joint_4_least_deg = ("joint_4_deg", "min"), joint_4_most_deg = ("joint_4_deg", "max"),
            step_length_median_cm = ("length_cm", "median"), tip_beside_median_cm = ("tip_beside_cm", "median"),
            wrist_median_deg = ("joint_5_deg", "median"), wrist_most_deg = ("joint_5_deg", "max"))
        run_bins = the_run.groupby("bin").agg(run_pictures = ("row", "size"), run_twist_deg = ("twist_deg", "sum"), run_joint_4_deg = ("joint_4_deg", "median"))
        print("\nPer centimetre along the line (0 = first pass of the crossing): the demonstrations' puck beside the line (cm, + = left of the")
        print("direction of travel), pictures and forearm twist spent there per demonstration, and the run's pictures and twist there")
        print(corridor.join(dwell[["pictures_per_demonstration", "twist_per_demonstration_deg", "joint_4_median_deg", "wrist_most_deg"]]).join(run_bins).round(2).to_string())

        merged = the_run.merge(corridor, left_on = "bin", right_index = True, how = "left")
        is_outside = (merged["puck_beside_cm"] > merged["ninetieth"]) | (merged["puck_beside_cm"] < merged["tenth"])
        is_beyond_all = (merged["puck_beside_cm"] > merged["most"]) | (merged["puck_beside_cm"] < merged["least"])
        stays_outside = is_outside.to_numpy()[::-1].cumprod()[::-1].astype(bool)
        stays_beyond = is_beyond_all.to_numpy()[::-1].cumprod()[::-1].astype(bool)
        leaving_step = int(merged["row"].to_numpy()[stays_outside][0]) if stays_outside.any() else None
        beyond_step = int(merged["row"].to_numpy()[stays_beyond][0]) if stays_beyond.any() else None
        off = the_run["puck_beside_cm"].abs()
        print(f"\nThe run's puck: on the line (centre within {settings['on_line_cm']} cm) in {int((off <= settings['on_line_cm']).sum())} of {len(the_run)} "
              f"pictures, further than {settings['off_line_cm']} cm beside it in {int((off > settings['off_line_cm']).sum())}; largest {off.max():.1f} cm; "
              f"median {off.median():.2f} cm")
        demo_off = demonstrations["puck_beside_cm"].abs()
        print(f"The demonstrations' puck: on the line in {100 * (demo_off <= settings['on_line_cm']).mean():.0f} % of pictures, further than "
              f"{settings['off_line_cm']} cm in {100 * (demo_off > settings['off_line_cm']).mean():.0f} %; median {demo_off.median():.2f} cm; "
              f"per demonstration the largest is {demonstrations.groupby('plan')['puck_beside_cm'].apply(lambda values: values.abs().quantile(0.98)).median():.1f} cm (median of the 98th percentiles)")
        print(f"First step from which the run's puck stays outside the demonstrations' 10th to 90th percentile: {leaving_step}; "
              f"outside every demonstration: {beyond_step}")

        table = merged[["row", "puck_along_cm", "puck_beside_cm", "tenth", "median", "ninetieth", "most", "tip_beside_cm", "joint_4_deg",
                        "angle_deg", "length_cm", "twist_deg", "step_beside_line_deg", "camera_beside_line_deg", "stop_probability"]].copy()
        nearby_rows = []
        for _, step in the_run.iterrows():
            near = demonstrations[(demonstrations["puck_along_cm"] - step["puck_along_cm"]).abs() <= settings["nearby_cm"]]
            near = near[near["action"] != "stop"]
            nearby_rows.append({
                "demo_pictures": len(near), "demo_step_beside_line_deg": near.loc[near["length_cm"] > 0.3, "step_beside_line_deg"].median(),
                "demo_camera_beside_line_deg": near["camera_beside_line_deg"].median(), "demo_length_cm": near["length_cm"].median(),
                "demo_twist_deg": near["twist_deg"].median(), "demo_twist_tenth": near["twist_deg"].quantile(0.1), "demo_twist_ninetieth": near["twist_deg"].quantile(0.9),
                "demo_joint_4_deg": near["joint_4_deg"].median(), "demo_backward_share": float((near["forward_cm"] < -0.3).mean()) if len(near) else numpy.nan})
        table = pandas.concat([table.reset_index(drop = True), pandas.DataFrame(nearby_rows)], axis = 1)
        print("\nStep by step: the run (asked direction in the picture, asked length, asked twist; the step's direction and the camera's direction")
        print("against the line) and the demonstrations' medians within 0.75 cm along the line (share of their steps that go backwards in the picture)")
        shown = table.rename(columns = {"puck_along_cm": "along", "puck_beside_cm": "beside", "tip_beside_cm": "tip_beside", "joint_4_deg": "joint_4",
                                        "angle_deg": "asked_angle", "length_cm": "asked_len", "twist_deg": "asked_twist", "step_beside_line_deg": "step_vs_line",
                                        "camera_beside_line_deg": "camera_vs_line", "stop_probability": "stop", "demo_step_beside_line_deg": "d_step_vs_line",
                                        "demo_camera_beside_line_deg": "d_camera_vs_line", "demo_length_cm": "d_len", "demo_twist_deg": "d_twist",
                                        "demo_twist_tenth": "d_twist_10", "demo_twist_ninetieth": "d_twist_90", "demo_joint_4_deg": "d_joint_4",
                                        "demo_backward_share": "d_backward", "demo_pictures": "d_pictures"})
        print(shown.drop(columns = ["tenth", "ninetieth", "most", "median"]).round(2).to_string(index = False))

        # the walk round the puck: where the demonstrations spend many pictures without the puck moving on
        # the bins around the busiest centimetre of the loop in which the demonstrations spend "busy_pictures" or more
        loop_dwell = dwell[(dwell.index >= 1.0) & (dwell.index < float(demonstrations["loop_length_cm"].iloc[0]))]["pictures_per_demonstration"]
        busiest = float(loop_dwell.idxmax())
        first_busy = last_busy = busiest
        while first_busy - settings["bin_cm"] in loop_dwell.index and loop_dwell[first_busy - settings["bin_cm"]] >= settings["busy_pictures"]:
            first_busy -= settings["bin_cm"]
        while last_busy + settings["bin_cm"] in loop_dwell.index and loop_dwell[last_busy + settings["bin_cm"]] >= settings["busy_pictures"]:
            last_busy += settings["bin_cm"]
        if True:
            zone = (first_busy, last_busy + settings["bin_cm"])
            in_zone = demonstrations[(demonstrations["puck_along_cm"] >= zone[0]) & (demonstrations["puck_along_cm"] < zone[1])]
            per_demo = in_zone.groupby("plan").agg(pictures = ("row", "size"), twist_deg = ("twist_deg", "sum"),
                                                   backward = ("forward_cm", lambda values: int((values < -0.3).sum())),
                                                   joint_4_first = ("joint_4_deg", "first"), joint_4_last = ("joint_4_deg", "last"),
                                                   wrist_most = ("joint_5_deg", "max"), puck_moved_cm = ("puck_move_cm", "sum"), tip_moved_cm = ("tip_move_cm", "sum"))
            run_zone = the_run[(the_run["puck_along_cm"] >= zone[0]) & (the_run["puck_along_cm"] < zone[1])]
            print(f"\nThe walk round the puck: around the busiest centimetre of the loop the demonstrations spend {settings['busy_pictures']} or more pictures per "
                  f"centimetre from {zone[0]:.0f} to {zone[1]:.0f} cm after the crossing. Per demonstration there:")
            print(per_demo.round(1).to_string())
            print("medians:", per_demo.median().round(1).to_dict())
            print(f"the run there: {len(run_zone)} pictures, twist asked {run_zone['twist_deg'].sum():.1f} deg in total, "
                  f"{int((run_zone['forward_cm'] < -0.3).sum())} backward steps, puck moved {run_zone['puck_move_cm'].sum():.1f} cm, "
                  f"asked length median {run_zone['length_cm'].median():.2f} cm, forearm from {run_zone['joint_4_deg'].iloc[0]:.0f} to {run_zone['joint_4_deg'].iloc[-1]:.0f} deg"
                  if len(run_zone) else "the run has no pictures there")

        # the figure
        reference_mean = reference_points.mean(axis = 0)
        axis = row_of_axes[0]
        is_ink = reference["is_ink"].to_numpy() > 0
        axis.scatter(reference_points[is_ink, 0], reference_points[is_ink, 1], s = 22, color = "0.25", marker = "s", linewidths = 0)
        for plan, part in demonstrations.groupby("plan"):
            points = in_sheet_frame(part[["puck_x_cm", "puck_y_cm"]].to_numpy(), registrations.loc[part["recording"].iloc[0]], reference_mean)
            axis.plot(points[:, 0], points[:, 1], "-", color = "tab:blue", linewidth = 0.7, alpha = 0.55)
        axis.plot([], [], "-", color = "tab:blue", linewidth = 0.9, label = f"puck, {episodes} demonstrations")
        run_registration = registrations.loc[pair["run"]]
        puck = in_sheet_frame(the_run[["puck_x_cm", "puck_y_cm"]].to_numpy(), run_registration, reference_mean)
        tip = in_sheet_frame(the_run[["tip_x_cm", "tip_y_cm"]].to_numpy(), run_registration, reference_mean)
        axis.plot(tip[:, 0], tip[:, 1], ":", color = "tab:orange", linewidth = 1.3, label = "magnet tip, robot run")
        axis.plot(puck[:, 0], puck[:, 1], "-o", color = "red", linewidth = 1.6, markersize = 3, label = "puck, robot run")
        for position in range(0, len(puck), 5):
            axis.annotate(str(int(the_run["row"].iloc[position])), puck[position], fontsize = 8, color = "darkred", xytext = (4, 4), textcoords = "offset points")
        axis.set_aspect("equal")
        axis.grid(True, linewidth = 0.3)
        axis.legend(loc = "upper right", fontsize = 8)
        axis.set_title(f"Sheet {sheet_name} from above (cm, the robot run's frame); red numbers = steps")
        axis.set_xlabel("x from the base axis (cm)")
        axis.set_ylabel("y (cm), left is up")

        axis = row_of_axes[1]
        for plan, part in demonstrations.groupby("plan"):
            axis.plot(part["puck_along_cm"], part["puck_beside_cm"], "-", color = "tab:blue", linewidth = 0.6, alpha = 0.5)
        axis.fill_between(corridor.index + 0.5 * settings["bin_cm"], corridor["tenth"], corridor["ninetieth"], color = "tab:blue", alpha = 0.18, label = "demonstrations, 10th to 90th percentile")
        axis.plot(the_run["puck_along_cm"], the_run["puck_beside_cm"], "-o", color = "red", markersize = 3, label = "robot run")
        if leaving_step is not None:
            where = the_run[the_run["row"] == leaving_step].iloc[0]
            axis.axvline(where["puck_along_cm"], color = "red", linestyle = "--", linewidth = 1)
            axis.annotate(f"leaves at step {leaving_step}", (where["puck_along_cm"], where["puck_beside_cm"]), xytext = (8, 12), textcoords = "offset points", color = "red")
        axis.axhspan(-settings["on_line_cm"], settings["on_line_cm"], color = "0.5", alpha = 0.15)
        loop_length = float(demonstrations["loop_length_cm"].iloc[0])
        for crossing in (0.0, loop_length):
            axis.axvline(crossing, color = "black", linestyle = ":", linewidth = 1)
        axis.set_ylim(-4, 8)
        axis.set_xlabel("cm along the line (0 = first pass of the crossing; dotted: both passes)")
        axis.set_ylabel("puck beside the line (cm, + = left of travel); grey band = on the ink")
        axis.set_title("Puck beside the line")
        axis.grid(True, linewidth = 0.3)
        axis.legend(loc = "upper left", fontsize = 8)

        axis = row_of_axes[2]
        for plan, part in demonstrations.groupby("plan"):
            axis.plot(part["puck_along_cm"], part["joint_4_deg"], "-", color = "tab:blue", linewidth = 0.6, alpha = 0.5)
        axis.plot(the_run["puck_along_cm"], the_run["joint_4_deg"], "-o", color = "red", markersize = 3, label = "robot run")
        for limit in (-117.0, 117.0):
            axis.axhline(limit, color = "0.4", linestyle = "--", linewidth = 0.8)
        for crossing in (0.0, loop_length):
            axis.axvline(crossing, color = "black", linestyle = ":", linewidth = 1)
        axis.set_xlabel("cm along the line")
        axis.set_ylabel("forearm twist, joint 4 (degrees); dashed = the driver's limit")
        axis.set_title("Forearm twist where the puck is (blue = demonstrations)")
        axis.grid(True, linewidth = 0.3)
        axis.legend(loc = "upper left", fontsize = 8)

        axis = row_of_axes[3]
        axis.bar(dwell.index + 0.5 * settings["bin_cm"], dwell["pictures_per_demonstration"], width = 0.9 * settings["bin_cm"], color = "tab:blue", alpha = 0.45,
                 label = "demonstrations: pictures per cm of line")
        axis.bar(run_bins.index + 0.5 * settings["bin_cm"], run_bins["run_pictures"], width = 0.45 * settings["bin_cm"], color = "red", alpha = 0.8, label = "robot run: pictures per cm of line")
        axis.set_xlabel("cm along the line")
        axis.set_ylabel("pictures per centimetre of line")
        axis.grid(True, linewidth = 0.3)
        second = axis.twinx()
        second.plot(the_run["puck_along_cm"], the_run["length_cm"], "-", color = "darkred", linewidth = 1.2, label = "robot run: asked step length (cm)")
        second.plot(the_run["puck_along_cm"], the_run["stop_probability"], "-", color = "black", linewidth = 1.2, label = "robot run: stop chance")
        second.set_ylim(0, 1.6)
        second.set_ylabel("asked step length (cm) and stop chance")
        handles = axis.get_legend_handles_labels()
        more = second.get_legend_handles_labels()
        axis.legend(handles[0] + more[0], handles[1] + more[1], loc = "upper left", fontsize = 8)
        axis.set_title("Where the pictures are spent (the walk round the puck) and what the model asked")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "03_run_against_demonstrations.png", dpi = 60)



if __name__ == "__main__":
    main()
