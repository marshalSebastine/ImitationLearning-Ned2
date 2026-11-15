"""
Question 1 of the orientation-coverage analysis: which placements of the six course sheets are in the demonstrations?

For every episode the photo-traced line of its sheet is laid onto the magnet tip's recorded path by a rigid fit (one
turn and one shift, no stretch), which measures how the sheet really lay on the table; the set-up pictures show only
how it was planned to lie, and the plan is recomputed here with the set-up pictures' own code (setup_visuals) for the
comparison. The start kinds and ends come from the index after the audit's corrections. Printed and saved:
    01_placements.out.txt          what the start codes mean, the plan per sheet, the measured lay-outs and their groups
    01_placements_per_episode.csv  one row per episode: the fitted turn and shift, the fit's quality, the start
    01_placements_groups.csv       one row per distinct lay-out (sheet, end, turn, place) with its episodes
    01_placements_map.png          top view: every distinct lay-out's paper and line per sheet

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe 01_placements.py
"""

import math
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
from matplotlib import pyplot
from scipy.optimize import least_squares

import orientation_common
from orientation_common import EVIDENCE_FOLDER, arc_lengths, course_sheet_lines, demonstration_frames, rotated, wrapped_deg

from config import MODEL_MODE_PARAMETERS, RECORDING_PAGE_PARAMETERS, SETUP_VISUALS_PARAMETERS  # noqa: E402

PLACEMENT_PARAMETERS = {
    # the fit: the traced line is lengthened backwards at its start by this much, because the tip starts behind the puck
    "start_extension_cm": 3.0,
    # how strongly the last tip point is tied to the place "tip_behind_puck_cm" before the line's end (per cm, against
    # 1 per path point); it settles the shift along a straight line and the turn of a short near-end path
    "end_anchor_weight": 1.0,
    "first_guess_turns_deg": list(range(-180, 180, 10)),
    "robust_scale_cm": 0.5,
    # starts a few centimetres before the end give a path too short to fix the sheet's turn: their fit is reported
    # but they form no group of their own; each joins the nearest lay-out of its sheet and end
    "kinds_without_own_group": ["NE", "X2"],
    # two episodes share a lay-out when their sheets lay within this turn and this distance (at the paper's centre)
    "same_layout_turn_deg": 10.0,
    "same_layout_shift_cm": 3.0,
    "paper_size_cm": SETUP_VISUALS_PARAMETERS["paper_size_cm"],
    "base_half_width_cm": SETUP_VISUALS_PARAMETERS["base_half_width_cm"],
    "reach_cm": [100.0 * value for value in MODEL_MODE_PARAMETERS["tip_reach_m"]],
    "figure_path": EVIDENCE_FOLDER / "01_placements_map.png"
}



def fitted_layout(
        tip_cm: numpy.ndarray,
        line_cm: numpy.ndarray,
        parameters: Dict
) -> Dict:
    """
    Fit the turn and shift that lay a sheet's traced line onto the recorded tip path.

    Parameters
    ----------
    tip_cm : numpy.ndarray
        The magnet tip's positions of one episode in the robot's base frame, shape (N, 2), in cm.
    line_cm : numpy.ndarray
        The sheet's traced line in the sheet's own frame, ordered from the episode's start end, shape (M, 2), in cm.
    parameters : dict
        ``PLACEMENT_PARAMETERS`` merged over ``ORIENTATION_PARAMETERS``.

    Returns
    -------
    dict
        "turn_deg" (the sheet frame's turn on the table, counter-clockwise), "shift_cm" (where the sheet's bottom-left
        corner lies), "median_cm" and "p90_cm" (the tip points' distances from the laid line) and "end_gap_cm" (the
        last tip point's distance from its expected place before the line's end).
    """

    start_direction = line_cm[0] - line_cm[10]
    start_direction = start_direction / numpy.linalg.norm(start_direction)
    extension = line_cm[0] + numpy.outer(numpy.arange(parameters["start_extension_cm"], 0.0, -0.5), start_direction)
    target = numpy.vstack([extension, line_cm[::5]])
    lengths = arc_lengths(line_cm)
    anchor_arc = lengths[-1] - parameters["tip_behind_puck_cm"]
    anchor = numpy.array([numpy.interp(anchor_arc, lengths, line_cm[:, 0]), numpy.interp(anchor_arc, lengths, line_cm[:, 1])])

    def residuals(values: numpy.ndarray) -> numpy.ndarray:
        placed = rotated(target, math.degrees(values[0])) + values[1:3]
        distances = numpy.sqrt(((tip_cm[:, None, :] - placed[None, :, :]) ** 2).sum(axis = 2)).min(axis = 1)
        end_gap = rotated(anchor[None, :], math.degrees(values[0]))[0] + values[1:3] - tip_cm[-1]
        return numpy.concatenate([distances, parameters["end_anchor_weight"] * end_gap])

    best_cost, best_start = math.inf, None
    for turn_deg in parameters["first_guess_turns_deg"]:
        shift = tip_cm[-1] - rotated(anchor[None, :], turn_deg)[0]
        start = numpy.array([math.radians(turn_deg), shift[0], shift[1]])
        cost = float(numpy.sum(numpy.minimum(residuals(start), 3.0) ** 2))
        if cost < best_cost:
            best_cost, best_start = cost, start
    solution = least_squares(residuals, best_start, loss = "soft_l1", f_scale = parameters["robust_scale_cm"])
    final = residuals(solution.x)
    return {
        "turn_deg": float(wrapped_deg(math.degrees(solution.x[0]))),
        "shift_cm": solution.x[1:3].copy(),
        "median_cm": float(numpy.median(final[:-2])),
        "p90_cm": float(numpy.percentile(final[:-2], 90)),
        "end_gap_cm": float(numpy.hypot(*final[-2:]) / parameters["end_anchor_weight"])
    }



def long_edge_name(turn_deg: float) -> str:
    """
    Name how the paper lay from the sheet frame's turn: the long edges run along the sheet's own y axis, so they point
    the turn plus 90 degrees on the table. "portrait" = long edges within 20 degrees of the robot's x axis (pointing
    away from the robot), "landscape" = within 20 degrees of its y axis, else "diagonal".
    """

    long_edge = abs(float(wrapped_deg(2.0 * (turn_deg + 90.0))) / 2.0)
    if long_edge <= 20.0:
        return "portrait"
    if long_edge >= 70.0:
        return "landscape"
    return "diagonal"



def per_episode_layouts(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Fit every episode's lay-out and describe its start.

    Returns
    -------
    pandas.DataFrame
        One row per episode: plan, sheet, corrected start end, start kind, pictures, tip path length, the fit, the
        paper's lay ("portrait", "landscape" or "diagonal"), the paper centre on the table, the line's direction at
        the start end (table frame), and the tip's first place and its first 3 cm of travel (measured).
    """

    sheets = course_sheet_lines()
    width, height = parameters["paper_size_cm"]
    rows = []
    for episode_id, group in frames.groupby("episode_id", sort = False):
        sheet_number = group["sheet_number"].iloc[0]
        start_end = group["start_end"].iloc[0]
        line = sheets[sheet_number]["line"] if start_end == "A" else sheets[sheet_number]["line"][::-1]
        tip = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        fit = fitted_layout(tip, line, parameters)
        placed_start = rotated(line[:21], fit["turn_deg"]) + fit["shift_cm"]
        line_start_direction = math.degrees(math.atan2(*(placed_start[20] - placed_start[0])[::-1]))
        centre = rotated(numpy.array([[width / 2, height / 2]]), fit["turn_deg"])[0] + fit["shift_cm"]
        travelled = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(tip, axis = 0), axis = 1))])
        is_moving = group["is_next_consecutive"].to_numpy() & (group["step_cm"].to_numpy() >= parameters["smallest_step_cm"])
        consecutive_cm = float(group["step_cm"].to_numpy()[is_moving].sum())
        first_far = numpy.flatnonzero(numpy.linalg.norm(tip - tip[0], axis = 1) >= 3.0)
        first_travel = math.degrees(math.atan2(*(tip[first_far[0]] - tip[0])[::-1])) if len(first_far) else float("nan")
        rows.append({
            "episode_id": episode_id, "plan": group["plan_number"].iloc[0], "sheet": sheet_number, "end": start_end,
            "start_kind": group["start_kind"].iloc[0], "session_date": group["session_date"].iloc[0],
            "lighting": group["lighting"].iloc[0], "pictures": len(group), "pushed_cm": round(consecutive_cm, 1),
            "turn_deg": round(fit["turn_deg"], 1), "shift_x_cm": round(float(fit["shift_cm"][0]), 2), "shift_y_cm": round(float(fit["shift_cm"][1]), 2),
            "fit_median_cm": round(fit["median_cm"], 2), "fit_p90_cm": round(fit["p90_cm"], 2), "end_gap_cm": round(fit["end_gap_cm"], 2),
            "paper": long_edge_name(fit["turn_deg"]), "paper_centre_x_cm": round(float(centre[0]), 1), "paper_centre_y_cm": round(float(centre[1]), 1),
            "line_start_direction_deg": round(line_start_direction, 1),
            "tip_start_x_cm": round(float(tip[0, 0]), 2), "tip_start_y_cm": round(float(tip[0, 1]), 2),
            "first_3_cm_direction_deg": round(first_travel, 1)
        })
    return pandas.DataFrame(rows)



def planned_layouts(layouts: pandas.DataFrame) -> pandas.DataFrame:
    """
    Add how each episode's sheet was planned to lie, as its set-up picture draws it.

    The plan is recomputed with setup_visuals.compute_setup for the episode's start kind and its corrected start end
    (the set-up picture of an episode whose end the audit corrected shows the other end).

    Returns
    -------
    pandas.DataFrame
        ``layouts`` with "start_cap_x_cm", "start_cap_y_cm" (fitted), "planned_turn_deg", "planned_cap_x_cm",
        "planned_cap_y_cm", "turn_minus_plan_deg" and "cap_from_plan_cm"; NaN for the two episodes without a plan number.
    """

    from setup_visuals import compute_setup, load_sheet_lines

    sheet_lines = load_sheet_lines()
    path_ids = {details["number"]: path_id for path_id, details in RECORDING_PAGE_PARAMETERS["sheets"].items()}
    sheets = course_sheet_lines()
    layouts = layouts.copy()
    for index, row in layouts.iterrows():
        line = sheets[row["sheet"]]["line"] if row["end"] == "A" else sheets[row["sheet"]]["line"][::-1]
        cap = rotated(line[:1], row["turn_deg"])[0] + numpy.array([row["shift_x_cm"], row["shift_y_cm"]])
        layouts.loc[index, ["start_cap_x_cm", "start_cap_y_cm"]] = [round(float(cap[0]), 1), round(float(cap[1]), 1)]
        if not isinstance(row["plan"], str):
            continue
        entry = {"start_code": row["start_kind"], "line_end": row["end"], "note": ""}
        setup = compute_setup(path_ids[row["sheet"]], row["plan"], entry, sheet_lines)
        bottom_edge = setup["paper"][1] - setup["paper"][0]
        planned_turn = math.degrees(math.atan2(bottom_edge[1], bottom_edge[0]))
        layouts.loc[index, "planned_turn_deg"] = round(planned_turn, 1)
        layouts.loc[index, ["planned_cap_x_cm", "planned_cap_y_cm"]] = [round(float(setup["start_cap"][0]), 1), round(float(setup["start_cap"][1]), 1)]
        layouts.loc[index, "turn_minus_plan_deg"] = round(float(wrapped_deg(row["turn_deg"] - planned_turn)), 1)
        layouts.loc[index, "cap_from_plan_cm"] = round(float(numpy.hypot(*(cap - setup["start_cap"]))), 1)
    return layouts



def grouped_layouts(
        layouts: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Group the episodes of each sheet and end whose sheets lay alike (single linkage on turn and paper centre).

    Episodes with a short path (the start kinds in "kinds_without_own_group") do not form groups; each joins the group
    of the nearest long episode of its sheet and end when it lies within twice the limits, else it stays apart ("short").

    Returns
    -------
    pandas.DataFrame
        ``layouts`` with the extra column "layout" (for example "02B-1").
    """

    layouts = layouts.copy()
    layouts["layout"] = ""
    for (sheet, end), part in layouts.groupby(["sheet", "end"]):
        long_ones = part[~part["start_kind"].isin(parameters["kinds_without_own_group"])].sort_values("turn_deg")
        groups: List[List[int]] = []
        for index, row in long_ones.iterrows():
            home = None
            for group in groups:
                for member in group:
                    other = layouts.loc[member]
                    turn_gap = abs(float(wrapped_deg(row["turn_deg"] - other["turn_deg"])))
                    shift_gap = math.hypot(row["paper_centre_x_cm"] - other["paper_centre_x_cm"], row["paper_centre_y_cm"] - other["paper_centre_y_cm"])
                    if turn_gap <= parameters["same_layout_turn_deg"] and shift_gap <= parameters["same_layout_shift_cm"]:
                        home = group
                        break
                if home is not None:
                    break
            if home is None:
                groups.append([index])
            else:
                home.append(index)
        groups.sort(key = lambda group: -len(group))
        for number, group in enumerate(groups, start = 1):
            layouts.loc[group, "layout"] = f"{sheet}{end}-{number}"
        for index, row in part[part["start_kind"].isin(parameters["kinds_without_own_group"])].iterrows():
            best_name, best_gap = f"{sheet}{end}-short", math.inf
            for member in long_ones.index:
                other = layouts.loc[member]
                turn_gap = abs(float(wrapped_deg(row["turn_deg"] - other["turn_deg"]))) / (2 * parameters["same_layout_turn_deg"])
                shift_gap = math.hypot(row["paper_centre_x_cm"] - other["paper_centre_x_cm"], row["paper_centre_y_cm"] - other["paper_centre_y_cm"]) / (2 * parameters["same_layout_shift_cm"])
                if max(turn_gap, shift_gap) <= 1.0 and max(turn_gap, shift_gap) < best_gap:
                    best_name, best_gap = other["layout"], max(turn_gap, shift_gap)
            layouts.loc[index, "layout"] = best_name
    return layouts



def draw_map(
        layouts: pandas.DataFrame,
        parameters: Dict
) -> None:
    """Draw, per sheet, the paper and the line of every distinct lay-out (its median fit) on the table."""

    sheets = course_sheet_lines()
    width, height = parameters["paper_size_cm"]
    paper = numpy.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]])
    half = parameters["base_half_width_cm"]
    figure, axes = pyplot.subplots(2, 3, figsize = (17, 11))
    for axis, sheet_number in zip(axes.ravel(), sorted(sheets)):
        part = layouts[(layouts["sheet"] == sheet_number) & ~layouts["layout"].str.endswith("short")]
        names = part.groupby("layout").size().sort_values(ascending = False).index
        colours = pyplot.get_cmap("tab20")(numpy.linspace(0, 1, 20))
        for position, name in enumerate(names):
            members = part[part["layout"] == name]
            turn = orientation_common.circular_mean_deg(members["turn_deg"])
            shift = numpy.array([members["shift_x_cm"].median(), members["shift_y_cm"].median()])
            end = members["end"].iloc[0]
            line = sheets[sheet_number]["line"] if end == "A" else sheets[sheet_number]["line"][::-1]
            placed_paper = rotated(paper, turn) + shift
            placed_line = rotated(line, turn) + shift
            colour = colours[position % 20]
            axis.plot(placed_paper[:, 0], placed_paper[:, 1], "-", color = colour, linewidth = 0.8, alpha = 0.7)
            kinds = ",".join(sorted(set(members["start_kind"])))
            axis.plot(placed_line[:, 0], placed_line[:, 1], "-", color = colour, linewidth = 2.0,
                      label = f"{name}: {len(members)} ep, turn {turn:+.0f} ({kinds})")
            axis.plot(placed_line[0, 0], placed_line[0, 1], "o", color = colour, markersize = 6)
        axis.add_patch(pyplot.Rectangle((-half, -half), 2 * half, 2 * half, color = "0.7"))
        for reach in parameters["reach_cm"]:
            axis.add_patch(pyplot.Circle((0.0, 0.0), reach, fill = False, linestyle = "--", color = "0.4", linewidth = 0.8))
        axis.set_aspect("equal")
        axis.set_xlim(-12, 56)
        axis.set_ylim(-36, 36)
        axis.grid(alpha = 0.3)
        axis.set_title(f"Sheet {sheet_number}: {len(names)} distinct lay-outs in {len(part)} long episodes (dot = start end)")
        axis.set_xlabel("x away from the robot (cm)")
        axis.set_ylabel("y to the robot's left (cm)")
        axis.legend(fontsize = 6, loc = "upper left", ncol = 2)
    figure.suptitle("Measured lay-outs of the six course sheets in the demonstrations (rigid fit of the traced line to the tip path; "
                    "dashed rings: tip reach 12 and 49.5 cm)", fontsize = 11)
    figure.tight_layout()
    figure.savefig(parameters["figure_path"], dpi = 80)
    pyplot.close(figure)



def main(parameters: Optional[Dict] = None) -> None:
    """Print the plan and the measured lay-outs, and save the tables and the map."""

    settings = orientation_common.merged_parameters(PLACEMENT_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 500)
    pandas.set_option("display.max_columns", 40)
    frames = demonstration_frames(settings)

    print("=== What the start codes mean (config.py, RECORDING_PAGE_PARAMETERS['start_setups']) ===")
    for code, text in RECORDING_PAGE_PARAMETERS["start_setups"].items():
        print(f"  {code:6s} {text}")
    print("  Ends: A is the traced line's 'first' end, B its 'last' (SETUP_VISUALS_PARAMETERS['end_keys']); the start end")
    print("  used here is the audit's corrected end (evidence/corrected_ends.csv), not the plan's.")

    episodes = frames.groupby("episode_id", sort = False).agg(
        sheet = ("sheet_number", "first"), end = ("start_end", "first"), kind = ("start_kind", "first"), pictures = ("frame_index", "size"))
    print("\n=== Episodes and pictures per sheet, corrected start end and start kind (measured, audit-cleaned index) ===")
    counts = episodes.groupby(["sheet", "end", "kind"]).agg(episodes = ("pictures", "size"), pictures = ("pictures", "sum"))
    print(counts.to_string())
    print("\nPer sheet and end:")
    print(episodes.groupby(["sheet", "end"]).agg(episodes = ("pictures", "size"), pictures = ("pictures", "sum")).to_string())
    print("\nPer start kind over all sheets:")
    print(episodes.groupby("kind").agg(episodes = ("pictures", "size"), pictures = ("pictures", "sum")).sort_values("episodes", ascending = False).to_string())

    layouts = planned_layouts(grouped_layouts(per_episode_layouts(frames, settings), settings))
    layouts.to_csv(EVIDENCE_FOLDER / "01_placements_per_episode.csv", index = False)
    line_directions = layouts[~layouts["start_kind"].isin(["NE", "MID", "X1", "X2"])]["line_start_direction_deg"]
    print(f"\nThe line's direction at the start cap over the {len(line_directions)} start-mark episodes (fitted; 0 = straight away from the robot): within 20 degrees "
          f"in {int((line_directions.abs() <= 20).sum())}, 20 to 40 in {int(((line_directions.abs() > 20) & (line_directions.abs() <= 40)).sum())}, over 40 in "
          f"{int((line_directions.abs() > 40).sum())} (range {line_directions.min():+.0f} to {line_directions.max():+.0f})")
    print("\n=== Measured lay-out per episode (calculated: rigid fit of the traced line to the measured tip path) ===")
    print("turn_deg = turn of the sheet's own frame on the table; paper = how the long edges lay; line_start_direction = the")
    print("line's direction at the start end on the table (0 = straight away from the robot); first_3_cm_direction = where the")
    print("tip really went in its first 3 cm (measured).")
    shown = ["plan", "end", "start_kind", "pictures", "pushed_cm", "turn_deg", "paper", "paper_centre_x_cm", "paper_centre_y_cm",
             "line_start_direction_deg", "first_3_cm_direction_deg", "tip_start_x_cm", "tip_start_y_cm", "fit_median_cm", "fit_p90_cm", "layout"]
    for sheet_number, part in layouts.groupby("sheet"):
        print(f"\n--- sheet {sheet_number} ---")
        print(part.sort_values(["end", "layout", "plan"])[shown].to_string(index = False))

    long_ones = layouts[~layouts["layout"].str.endswith("short")]
    groups = long_ones.groupby(["sheet", "end", "layout"]).agg(
        episodes = ("plan", "size"), pictures = ("pictures", "sum"),
        turn_deg = ("turn_deg", orientation_common.circular_mean_deg),
        paper = ("paper", lambda values: values.mode().iloc[0]),
        paper_centre_x_cm = ("paper_centre_x_cm", "median"), paper_centre_y_cm = ("paper_centre_y_cm", "median"),
        line_start_direction_deg = ("line_start_direction_deg", orientation_common.circular_mean_deg),
        start_kinds = ("start_kind", lambda values: ",".join(sorted(set(values)))),
        plans = ("plan", lambda values: " ".join(sorted(str(value) for value in values)))).reset_index()
    groups[["turn_deg", "line_start_direction_deg"]] = groups[["turn_deg", "line_start_direction_deg"]].round(1)
    groups.to_csv(EVIDENCE_FOLDER / "01_placements_groups.csv", index = False)
    print("\n=== Distinct lay-outs (episodes whose sheets lay within "
          f"{settings['same_layout_turn_deg']:.0f} degrees and {settings['same_layout_shift_cm']:.0f} cm of a neighbour) ===")
    print(groups.to_string(index = False))
    print("\nDistinct lay-outs per sheet (long episodes only):")
    summary = groups.groupby("sheet").agg(layouts = ("layout", "size"), episodes = ("episodes", "sum"), pictures = ("pictures", "sum"),
                                         largest_layout_episodes = ("episodes", "max"))
    summary["paper_lays"] = groups.groupby("sheet")["paper"].agg(lambda values: ", ".join(f"{name} {count}" for name, count in values.value_counts().items()))
    print(summary.to_string())
    print(f"\nAll sheets: {len(groups)} distinct lay-outs in {int(groups['episodes'].sum())} long episodes; "
          f"{int((layouts['layout'].str.endswith('short')).sum())} short episodes matched no long lay-out.")
    print("\nFit quality over all episodes (cm): median of the per-episode median distance "
          f"{layouts['fit_median_cm'].median():.2f}, 90th percentile of it {layouts['fit_median_cm'].quantile(0.9):.2f}")

    print("\n=== Measured lay-out against the set-up picture's plan (start-mark episodes; the plan recomputed with setup_visuals) ===")
    compared = layouts[~layouts["start_kind"].isin(["NE", "MID", "X1", "X2"]) & layouts["planned_turn_deg"].notna()]
    print(f"{len(compared)} episodes. Turn of the sheet minus the planned turn (degrees): median {compared['turn_minus_plan_deg'].median():+.1f}, "
          f"median of the absolute difference {compared['turn_minus_plan_deg'].abs().median():.1f}, over 15 degrees in {int((compared['turn_minus_plan_deg'].abs() > 15).sum())} episodes, "
          f"over 30 in {int((compared['turn_minus_plan_deg'].abs() > 30).sum())}.")
    print(f"Fitted start cap: x median {compared['start_cap_x_cm'].median():.1f} cm (planned {compared['planned_cap_x_cm'].median():.1f}), y median "
          f"{compared['start_cap_y_cm'].median():.1f} (planned {compared['planned_cap_y_cm'].median():.1f}); distance from the planned place: median "
          f"{compared['cap_from_plan_cm'].median():.1f} cm, 90th percentile {compared['cap_from_plan_cm'].quantile(0.9):.1f}.")
    print(compared.groupby(["sheet", "end"]).agg(
        episodes = ("plan", "size"), turn_minus_plan_median = ("turn_minus_plan_deg", "median"),
        turn_minus_plan_least = ("turn_minus_plan_deg", "min"), turn_minus_plan_most = ("turn_minus_plan_deg", "max"),
        start_cap_x_median = ("start_cap_x_cm", "median"), cap_from_plan_cm_median = ("cap_from_plan_cm", "median")).round(1).to_string())
    print("Episodes whose sheet lay more than 20 degrees from the plan:")
    print(compared[compared["turn_minus_plan_deg"].abs() > 20][["plan", "end", "start_kind", "turn_deg", "planned_turn_deg", "turn_minus_plan_deg", "layout"]].to_string(index = False))

    print("\n=== How the line starts on the table, STD starts only (measured first 3 cm of tip travel, 0 = +x) ===")
    standard = layouts[layouts["start_kind"] == "STD"]
    print(standard.groupby(["sheet", "end"])["first_3_cm_direction_deg"].describe()[["count", "min", "50%", "max"]].round(1).to_string())
    print("\nTip start place of the start-mark episodes (not NE, MID, X1, X2), cm:")
    marked = layouts[~layouts["start_kind"].isin(["NE", "MID", "X1", "X2"])]
    print(marked[["tip_start_x_cm", "tip_start_y_cm"]].describe().loc[["count", "min", "50%", "max"]].round(2).to_string())

    draw_map(layouts, settings)
    print(f"\nSaved {settings['figure_path'].name}, 01_placements_per_episode.csv, 01_placements_groups.csv")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding = "utf-8")
    main()
