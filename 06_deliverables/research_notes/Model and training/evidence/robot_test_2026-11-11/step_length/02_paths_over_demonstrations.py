"""
The robot's tip path over the demonstrations' tip paths on sheets 03 and 04, and what happened at the first sharp bend.

For each robot run on sheet 03 (runs 5, 9, 10) and sheet 04 (run 6) the standard-start demonstrations of the same sheet
and end are laid onto the run: the puck track of each demonstration (which lies on the line of its own sheet
placement) is turned and shifted onto the line map read from the run's pictures (line_and_puck.register_track_to_line),
and the same move carries the demonstration's tip path. So the paths are aligned where the puck lay at the start
and along the printed line, not by guessing where the sheet lay.

Measured for every run and every demonstration:
    the first bend    how far the tip went past the puck's start place along the first leg of the line before it
                      turned;
    turn and travel   how far the tip travelled while the forearm (joint 4) turned its first 60 degrees, and while the
                      picture's heading turned through a fixed 40-degree window that every path passes;
    twists in place   steps with a tip move under 0.3 cm and a twist of 3 degrees or more, in that stretch.

Run from this folder after line_and_puck.py:
    python 02_paths_over_demonstrations.py
Writes 02_paths_over_demonstrations.out.txt, 02_registration.csv, 02_first_bend.csv, 02_top_view_run_NN.png and
02_turn_against_travel.png.
"""

import math
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import line_and_puck
import step_length_common as common

PATH_PARAMETERS = {
    "runs": [5, 9, 10, 6],
    # the forearm turn whose tip travel is measured, degrees from the start value of joint 4
    "first_turn_deg": 60.0,
    # a window of the picture's heading, measured from the direction of the line's first leg, that every path on the
    # sheet passes on its way round the first bend (sheet 03: the heading goes from about 0 to about 90 degrees; sheet
    # 04: from about 0 to about -60 at the first bend, so its window is negative)
    "heading_window_deg": {"path_03": [20.0, 60.0], "path_04": [-10.0, -40.0]},
    # the first leg's direction is the puck's mean direction over this much of its first travel (cm)
    "first_leg_cm": 2.0,
    # rows of a path that belong to the first bend: until the tip has travelled this far (cm)
    "first_bend_travel_cm": 16.0,
    "decisions_shown": 15,
    # the tip has left the demonstrated paths when it is further from every demonstration's tip path than the
    # demonstrations are from each other on all but this share of their own rows, on this many pictures in a row
    "corridor_percentile": 99.0,
    "corridor_pictures_in_a_row": 2
}



def travel_at_level(
        travelled_cm: numpy.ndarray,
        values: numpy.ndarray,
        level: float,
        from_travel_cm: float = 0.0
) -> float:
    """
    Return the tip travel at which ``values`` first reaches ``level`` (from below or from above), linear between rows.

    Parameters
    ----------
    travelled_cm : numpy.ndarray
        The path length of the tip up to each row.
    values : numpy.ndarray
        The angle at each row (joint 4 from its start value, or the picture's heading from the first leg).
    level : float
        The angle looked for.
    from_travel_cm : float, optional
        Rows before this travel are skipped.

    Returns
    -------
    float
        The travel in cm; NaN when the level is never reached.
    """

    first = int(numpy.searchsorted(travelled_cm, from_travel_cm, side = "left"))
    if first >= len(values):
        return float("nan")
    sign = 1.0 if level >= values[first] else -1.0
    if sign * (values[first] - level) >= 0.0:
        return float(travelled_cm[first])
    for index in range(first + 1, len(values)):
        if sign * (values[index] - level) >= 0.0:
            span = values[index] - values[index - 1]
            share = 0.0 if abs(span) < 1e-9 else (level - values[index - 1]) / span
            return float(travelled_cm[index - 1] + share * (travelled_cm[index] - travelled_cm[index - 1]))
    return float("nan")



def unwrapped_deg(angles_deg: numpy.ndarray) -> numpy.ndarray:
    """Return angles in degrees without jumps of 360."""

    return numpy.degrees(numpy.unwrap(numpy.radians(numpy.asarray(angles_deg, dtype = float))))



def path_measures(
        tip_xy_m: numpy.ndarray,
        joint_4_deg: numpy.ndarray,
        heading_deg: numpy.ndarray,
        step_cm: numpy.ndarray,
        twist_deg: numpy.ndarray,
        puck_start_xy_m: numpy.ndarray,
        first_leg_deg: float,
        sheet: str,
        parameters: Dict
) -> Dict[str, float]:
    """
    Measure one path (a robot run or a demonstration, already in the robot run's frame) at the first bend.

    Parameters
    ----------
    tip_xy_m : numpy.ndarray
        The tip positions, (rows, 2).
    joint_4_deg, heading_deg : numpy.ndarray
        The forearm twist and the picture's heading on the table at each row (the heading in the run's frame).
    step_cm, twist_deg : numpy.ndarray
        The tip move and the forearm change from each row to the next (what was done).
    puck_start_xy_m : numpy.ndarray
        Where the puck lay at the start (on the line, 2 to 3 cm after the place where the ink begins).
    first_leg_deg : float
        The direction of the line's first leg on the table, degrees.
    sheet : str
        "path_03" or "path_04" (chooses the heading window).
    parameters : dict
        ``PATH_PARAMETERS``.

    Returns
    -------
    dict
        The measures named in the module's docstring.
    """

    moved_cm = 100.0 * numpy.hypot(numpy.diff(tip_xy_m[:, 0]), numpy.diff(tip_xy_m[:, 1]))
    travelled = numpy.r_[0.0, numpy.cumsum(moved_cm)]
    is_first_bend = travelled <= parameters["first_bend_travel_cm"]
    leg = numpy.array([math.cos(math.radians(first_leg_deg)), math.sin(math.radians(first_leg_deg))])
    along_cm = 100.0 * (tip_xy_m - puck_start_xy_m) @ leg
    turn = joint_4_deg - joint_4_deg[0]
    turn_sign = -1.0 if numpy.nanmean(turn[is_first_bend]) < 0 else 1.0
    travel_first_turn = travel_at_level(travelled, turn_sign * turn, parameters["first_turn_deg"])
    rows_first_turn = int(numpy.searchsorted(travelled, travel_first_turn, side = "left")) if not math.isnan(travel_first_turn) else 0
    relative_heading = unwrapped_deg(heading_deg - first_leg_deg)
    relative_heading = relative_heading - 360.0 * round(relative_heading[0] / 360.0)
    window = parameters["heading_window_deg"][sheet]
    window_start = travel_at_level(travelled, relative_heading, window[0])
    window_end = travel_at_level(travelled, relative_heading, window[1], window_start) if not math.isnan(window_start) else float("nan")
    window_travel = window_end - window_start
    stretch = slice(0, max(rows_first_turn, 1))
    is_twist_in_place = (step_cm[stretch] < common.STEP_LENGTH_PARAMETERS["twist_in_place_tip_cm"]) & (numpy.abs(twist_deg[stretch]) >= common.STEP_LENGTH_PARAMETERS["twist_in_place_twist_deg"])
    in_window = (travelled[:-1] >= window_start) & (travelled[:-1] < window_end) if not math.isnan(window_travel) else numpy.zeros(len(travelled) - 1, dtype = bool)
    is_twist_in_window = (step_cm[:len(in_window)][in_window] < common.STEP_LENGTH_PARAMETERS["twist_in_place_tip_cm"]) & (numpy.abs(twist_deg[:len(in_window)][in_window]) >= common.STEP_LENGTH_PARAMETERS["twist_in_place_twist_deg"])
    return {
        "start_heading_from_first_leg_deg": float(relative_heading[0]),
        "start_tip_along_first_leg_cm": float(along_cm[0]),
        "tip_furthest_along_first_leg_cm": float(numpy.max(along_cm[is_first_bend])),
        "travel_for_first_turn_cm": travel_first_turn,
        "steps_for_first_turn": rows_first_turn,
        "twists_in_place_in_first_turn": int(is_twist_in_place.sum()),
        "mean_step_in_first_turn_cm": travel_first_turn / rows_first_turn if rows_first_turn else float("nan"),
        "travel_in_heading_window_cm": window_travel,
        "steps_in_heading_window": int(in_window.sum()),
        "twists_in_place_in_heading_window": int(is_twist_in_window.sum())
    }



def corridor_departure(
        run_tip_xy_m: numpy.ndarray,
        demonstration_tips: List[numpy.ndarray],
        parameters: Dict
) -> Dict[str, float]:
    """
    Find the first step at which the robot's tip is outside the corridor of the demonstrations' tip paths.

    The corridor's half width is taken from the demonstrations themselves: the distance of every tip point of one
    demonstration from the nearest other demonstration's path, at the percentile "corridor_percentile". The robot's tip
    starts behind the puck, the demonstrations' tips at the taught start mark, so the run is judged from the first step
    at which its tip is inside the corridor.

    Returns
    -------
    dict
        "corridor_half_width_cm", "first_step_inside", "first_step_outside" (NaN when the tip never leaves) and
        "distance_at_step_15_cm".
    """

    own_distances = []
    for index, tip in enumerate(demonstration_tips):
        others = [common.polyline_distance_cm(tip, other)[:, 0] for other_index, other in enumerate(demonstration_tips) if other_index != index]
        own_distances.append(numpy.min(others, axis = 0))
    half_width = float(numpy.percentile(numpy.concatenate(own_distances), parameters["corridor_percentile"]))
    distance = numpy.min([common.polyline_distance_cm(run_tip_xy_m, tip)[:, 0] for tip in demonstration_tips], axis = 0)
    inside = numpy.flatnonzero(distance <= half_width)
    first_inside = int(inside[0]) if len(inside) else 0
    first_outside, count = float("nan"), 0
    for index in range(first_inside, len(distance)):
        count = count + 1 if distance[index] > half_width else 0
        if count >= parameters["corridor_pictures_in_a_row"]:
            first_outside = float(index - count + 1)
            break
    return {"corridor_half_width_cm": half_width, "first_step_inside": first_inside, "first_step_outside": first_outside,
            "largest_distance_in_first_15_cm": float(distance[:16].max()), "distance_cm": distance}



def first_leg_direction_deg(puck_xy_m: numpy.ndarray, leg_cm: float) -> float:
    """Return the direction (degrees on the table) from the first puck position to where the puck has moved ``leg_cm``."""

    seen = puck_xy_m[numpy.isfinite(puck_xy_m[:, 0])]
    distance_cm = 100.0 * numpy.hypot(seen[:, 0] - seen[0, 0], seen[:, 1] - seen[0, 1])
    reached = numpy.flatnonzero(distance_cm >= leg_cm)
    far = seen[reached[0]] if len(reached) else seen[-1]
    return math.degrees(math.atan2(far[1] - seen[0, 1], far[0] - seen[0, 0]))



def main() -> None:

    settings = dict(PATH_PARAMETERS)
    lines: List[str] = []
    pandas.set_option("display.width", 250)
    frames = common.corrected_start_ends(common.load_demonstration_frames())
    runs = {run["number"]: run for run in common.STEP_LENGTH_PARAMETERS["robot_runs"]}
    registration_rows, measure_rows, curves = [], [], {}

    for number in settings["runs"]:
        run = runs[number]
        steps = common.load_robot_run(run)
        track = dict(numpy.load(line_and_puck.run_track_path(number)))
        distance_map = line_and_puck.distance_to_line_m(track, is_start_added = True)
        run_puck = track["puck_xy_m"]
        run_tip = steps[["tip_x", "tip_y"]].to_numpy()
        run_puck_start = run_puck[numpy.isfinite(run_puck[:, 0])][0]
        demonstrations = line_and_puck.reference_demonstrations(frames, run)

        moved_demonstrations = []
        first_leg_angles = []
        for episode_id, group in demonstrations.groupby("episode_id", sort = False):
            plan_number = group["plan_number"].iloc[0]
            saved = numpy.load(line_and_puck.demonstration_track_path(plan_number))
            puck = saved["puck_xy_m"]
            puck = puck[numpy.isfinite(puck[:, 0])]
            fit = line_and_puck.register_track_to_line(puck, track, run_puck_start - puck[0])
            pivot, shift = (fit["pivot_x_m"], fit["pivot_y_m"]), (fit["shift_x_m"], fit["shift_y_m"])
            tip_moved = line_and_puck.moved_points(group[["tip_x_m", "tip_y_m"]].to_numpy(), fit["turn_deg"], shift, pivot)
            puck_moved = line_and_puck.moved_points(puck, fit["turn_deg"], shift, pivot)
            registration_rows.append({"run": number, "plan": plan_number, "turn_deg": fit["turn_deg"], "shift_x_cm": 100 * fit["shift_x_m"], "shift_y_cm": 100 * fit["shift_y_m"],
                                      "puck_start_before_cm": f"({100 * puck[0, 0]:.1f}, {100 * puck[0, 1]:.1f})",
                                      "mean_distance_to_line_cm": 100 * fit["mean_distance_m"], "share_of_puck_track_within_5_mm_of_ink": fit["share_within_5_mm"]})
            first_leg_angles.append(first_leg_direction_deg(puck_moved, settings["first_leg_cm"]))
            moved_demonstrations.append({"plan": plan_number, "tip": tip_moved, "puck": puck_moved, "group": group, "turn_deg": fit["turn_deg"]})

        # the direction of the line's first leg in the run's frame: the middle of the demonstrations' puck directions
        first_leg_deg = float(numpy.median(first_leg_angles))
        for demonstration in moved_demonstrations:
            group = demonstration["group"]
            measures = path_measures(demonstration["tip"], group["joint_4_deg"].to_numpy(), group["picture_heading_deg"].to_numpy() + demonstration["turn_deg"],
                                     group["step_cm"].to_numpy()[:-1], group["twist_deg"].to_numpy()[:-1], demonstration["puck"][0], first_leg_deg, run["sheet"], settings)
            measure_rows.append({"run": number, "path": f"demonstration {demonstration['plan']}", **measures})
        done_cm = steps["done_cm"].to_numpy()[:-1]
        twist_done = steps["twist_done_deg"].to_numpy()[:-1]
        heading = steps["picture_heading_deg"].ffill().to_numpy()
        measures = path_measures(run_tip, steps["joint_4_deg"].to_numpy(), heading, done_cm, twist_done, run_puck_start, first_leg_deg, run["sheet"], settings)
        measure_rows.append({"run": number, "path": f"robot run {number} (done)", **measures})

        # the same decisions carried out with the asked length instead of 1 cm (open loop: later pictures would differ)
        asked = steps["asked_cm"].to_numpy()[:-1]
        direction = numpy.column_stack([steps["tip_next_x"] - steps["tip_x"], steps["tip_next_y"] - steps["tip_y"]])[:-1]
        direction = direction / numpy.maximum(numpy.linalg.norm(direction, axis = 1, keepdims = True), 1e-9)
        asked_tip = numpy.vstack([run_tip[0], run_tip[0] + numpy.cumsum(0.01 * asked[:, None] * direction, axis = 0)])
        measures = path_measures(asked_tip, steps["joint_4_deg"].to_numpy(), heading, asked, twist_done, run_puck_start, first_leg_deg, run["sheet"], settings)
        measure_rows.append({"run": number, "path": f"robot run {number} (as asked)", **measures})
        corridor = corridor_departure(run_tip, [demonstration["tip"] for demonstration in moved_demonstrations], settings)
        curves[number] = {"run": run, "steps": steps, "moved_demonstrations": moved_demonstrations, "first_leg_deg": first_leg_deg, "asked_tip": asked_tip,
                          "track": track, "run_puck_start": run_puck_start, "corridor": corridor}

        # top view
        figure, axis = pyplot.subplots(figsize = (11, 10))
        is_line = line_and_puck.line_mask(track)
        extent = [100 * track["x_m"][0], 100 * (track["x_m"][-1] + 0.002), 100 * track["y_m"][0], 100 * (track["y_m"][-1] + 0.002)]
        axis.imshow(numpy.where(is_line, 0.45, 0.0).T, origin = "lower", extent = extent, cmap = "gray_r", vmin = 0, vmax = 1, zorder = 0)
        for index, demonstration in enumerate(moved_demonstrations):
            axis.plot(100 * demonstration["tip"][:, 0], 100 * demonstration["tip"][:, 1], color = "tab:green", linewidth = 0.9, alpha = 0.75,
                      label = "tip paths of the demonstrations (laid onto this run's line)" if index == 0 else None, zorder = 2)
        axis.plot(100 * run_puck[:, 0], 100 * run_puck[:, 1], color = "tab:blue", linewidth = 1.3, marker = ".", markersize = 4, label = "puck in the robot run (from the pictures)", zorder = 3)
        axis.plot(100 * run_tip[:, 0], 100 * run_tip[:, 1], color = "tab:red", linewidth = 1.6, marker = ".", markersize = 5, label = "tip in the robot run", zorder = 4)
        shown = min(settings["decisions_shown"], len(asked_tip) - 1)
        axis.plot(100 * asked_tip[:shown + 1, 0], 100 * asked_tip[:shown + 1, 1], color = "tab:orange", linewidth = 1.6, marker = ".", markersize = 5, linestyle = "--",
                  label = f"the first {shown} decisions carried out with the asked length (open loop)", zorder = 5)
        for index in range(0, len(steps), 5):
            axis.annotate(str(index), (100 * run_tip[index, 0], 100 * run_tip[index, 1]), color = "tab:red", fontsize = 8, xytext = (3, 3), textcoords = "offset points")
            if numpy.isfinite(run_puck[index, 0]):
                axis.annotate(str(index), (100 * run_puck[index, 0], 100 * run_puck[index, 1]), color = "tab:blue", fontsize = 8, xytext = (3, -9), textcoords = "offset points")
        axis.plot(100 * run_puck_start[0], 100 * run_puck_start[1], marker = "o", markersize = 11, markerfacecolor = "none", markeredgecolor = "black", linestyle = "none", label = "puck at the start")
        axis.set_aspect("equal")
        centre = 100 * numpy.nanmean(numpy.vstack([run_tip, run_puck]), axis = 0)
        axis.set_xlim(centre[0] - 17, centre[0] + 17)
        axis.set_ylim(centre[1] - 15, centre[1] + 15)
        axis.grid(True, alpha = 0.3)
        axis.set_xlabel("x from the base axis, cm")
        axis.set_ylabel("y, cm (left is up)")
        axis.legend(fontsize = 8, loc = "lower right")
        axis.set_title(f"Run {number} (model {run['model']}, sheet {run['sheet'][-2:]}): grey = the line read from the run's pictures; numbers = decision numbers", fontsize = 10)
        figure.tight_layout()
        figure.savefig(common.EVIDENCE_FOLDER / f"02_top_view_run_{number:02d}.png", dpi = 90)
        pyplot.close(figure)

    registration = pandas.DataFrame(registration_rows)
    registration.round(3).to_csv(common.EVIDENCE_FOLDER / "02_registration.csv", index = False)
    measures = pandas.DataFrame(measure_rows)
    measures.round(3).to_csv(common.EVIDENCE_FOLDER / "02_first_bend.csv", index = False)

    common.print_and_keep(lines, "HOW THE DEMONSTRATIONS WERE LAID ONTO EACH ROBOT RUN [calculated: each demonstration's puck track fitted onto the run's line map]")
    common.print_and_keep(lines, registration.round(2).to_string(index = False))
    summary = registration.groupby("run").agg(demonstrations = ("plan", "size"), turn_median_deg = ("turn_deg", "median"), turn_least_deg = ("turn_deg", "min"), turn_most_deg = ("turn_deg", "max"),
                                              shift_x_median_cm = ("shift_x_cm", "median"), shift_y_median_cm = ("shift_y_cm", "median"),
                                              mean_distance_median_cm = ("mean_distance_to_line_cm", "median"), share_within_5_mm_median = ("share_of_puck_track_within_5_mm_of_ink", "median"))
    common.print_and_keep(lines)
    common.print_and_keep(lines, "the sheet of each robot run against the demonstrations' sheet (turn and shift of the demonstrations onto the run):")
    common.print_and_keep(lines, summary.round(2).to_string())
    common.print_and_keep(lines)

    common.print_and_keep(lines, "THE FIRST BEND [measured: frames.csv (demonstrations) and steps.csv (runs); 'as asked' is calculated, open loop]")
    common.print_and_keep(lines, f"travel_for_first_turn_cm: tip travel until joint 4 has turned {settings['first_turn_deg']:.0f} degrees from its start value; "
                                 f"travel_in_heading_window_cm: tip travel while the picture's heading (from the line's first leg) went through {settings['heading_window_deg']}")
    for number in settings["runs"]:
        part = measures[measures["run"] == number]
        common.print_and_keep(lines, f"--- run {number} (sheet {runs[number]['sheet'][-2:]}), first leg at {curves[number]['first_leg_deg']:.1f} degrees on the table")
        common.print_and_keep(lines, part.drop(columns = "run").round(2).to_string(index = False))
        is_demonstration = part["path"].str.startswith("demonstration")
        numeric = part.drop(columns = ["run", "path"])
        common.print_and_keep(lines, "demonstrations, median (least to most):  " + "; ".join(
            f"{name} {numeric.loc[is_demonstration, name].median():.2f} ({numeric.loc[is_demonstration, name].min():.2f} to {numeric.loc[is_demonstration, name].max():.2f})"
            for name in ["tip_furthest_along_first_leg_cm", "travel_for_first_turn_cm", "steps_for_first_turn", "twists_in_place_in_first_turn", "mean_step_in_first_turn_cm",
                         "travel_in_heading_window_cm", "steps_in_heading_window", "twists_in_place_in_heading_window"]))
        done = part[part["path"].str.endswith("(done)")].iloc[0]
        asked = part[part["path"].str.endswith("(as asked)")].iloc[0]
        middle = numeric.loc[is_demonstration].median()
        common.print_and_keep(lines, f"run {number}: the tip went {done['tip_furthest_along_first_leg_cm']:.1f} cm past the puck's start place along the first leg, the demonstrations "
                                     f"{middle['tip_furthest_along_first_leg_cm']:.1f} cm (median): overshoot {done['tip_furthest_along_first_leg_cm'] - middle['tip_furthest_along_first_leg_cm']:+.1f} cm; "
                                     f"as asked (open loop) {asked['tip_furthest_along_first_leg_cm']:.1f} cm")
        corridor = curves[number]["corridor"]
        common.print_and_keep(lines, f"run {number}: the tip is inside the corridor of the demonstrations' tip paths (half width {corridor['corridor_half_width_cm']:.1f} cm, the "
                                     f"{settings['corridor_percentile']:.0f}th percentile of the demonstrations' own spread) from step {corridor['first_step_inside']} and first outside it at step "
                                     f"{corridor['first_step_outside']:.0f}; distance from the nearest demonstrated tip path at steps 0, 5, 10, 15, 20: "
                                     + ", ".join(f"{corridor['distance_cm'][index]:.1f}" for index in (0, 5, 10, 15, 20) if index < len(corridor['distance_cm'])) + " cm")
        common.print_and_keep(lines, f"run {number}: heading window travel done {done['travel_in_heading_window_cm']:.1f} cm, as asked {asked['travel_in_heading_window_cm']:.1f} cm, "
                                     f"demonstrations {middle['travel_in_heading_window_cm']:.1f} cm (median)")
        common.print_and_keep(lines)

    # the forearm's turn and the picture's heading against the tip's travel
    figure, axes = pyplot.subplots(2, len(settings["runs"]), figsize = (6.2 * len(settings["runs"]), 11))
    for column, number in enumerate(settings["runs"]):
        curve = curves[number]
        steps = curve["steps"]
        for row, name in enumerate(["joint 4 from its start value, degrees", "picture heading from the line's first leg, degrees"]):
            axis = axes[row, column]
            for index, demonstration in enumerate(curve["moved_demonstrations"]):
                group = demonstration["group"]
                travelled = group["travelled_cm"].to_numpy()
                values = (group["joint_4_deg"].to_numpy() - group["joint_4_deg"].iloc[0] if row == 0
                          else unwrapped_deg(group["picture_heading_deg"].to_numpy() + demonstration["turn_deg"] - curve["first_leg_deg"]))
                axis.plot(travelled, values, color = "tab:green", linewidth = 0.9, alpha = 0.8, label = "demonstrations" if index == 0 else None)
            heading = steps["picture_heading_deg"].ffill().to_numpy()
            values = steps["joint_4_deg"].to_numpy() - steps["joint_4_deg"].iloc[0] if row == 0 else unwrapped_deg(heading - curve["first_leg_deg"])
            axis.plot(steps["travelled_cm"], values, color = "tab:red", linewidth = 1.8, marker = ".", markersize = 4, label = "robot run, 1 cm per decision (done)")
            asked_travel = numpy.r_[0.0, numpy.cumsum(steps["asked_cm"].to_numpy()[:-1])]
            axis.plot(asked_travel, values, color = "tab:orange", linewidth = 1.6, linestyle = "--", marker = ".", markersize = 4,
                      label = "the same decisions with the asked length (open loop)")
            axis.set_xlim(0, 30)
            axis.set_xlabel("tip travel, cm")
            axis.set_ylabel(name)
            axis.grid(True, alpha = 0.3)
            axis.set_title(f"run {number} (sheet {curve['run']['sheet'][-2:]}, model {curve['run']['model']})", fontsize = 10)
            limits = (-130, 60) if curve["run"]["sheet"] == "path_03" else (-60, 130)
            axis.set_ylim(*(limits if row == 0 else (-limits[1], -limits[0])))
        axes[0, column].legend(fontsize = 8)
    figure.suptitle("How fast the forearm (and with it the camera) turned per centimetre of tip travel at the start of sheets 03 and 04", fontsize = 12)
    figure.tight_layout(rect = (0, 0, 1, 0.97))
    figure.savefig(common.EVIDENCE_FOLDER / "02_turn_against_travel.png", dpi = 90)
    (common.EVIDENCE_FOLDER / "02_paths_over_demonstrations.out.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")



if __name__ == "__main__":
    main()
