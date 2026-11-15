"""
How the demonstrations get round sharp bends and hairpins: steps, twists in place, retreats and direction jumps.

Every demonstration picture gets a kind of place from the traced line at the puck's place (common.py): straight, gentle
bend, sharp bend or hairpin. Per sheet and kind of place the script counts
    steps per cm of line     pictures taken per cm the puck advanced along the line;
    step length              the labelled tip move after the picture, and the share under "short_step_cm";
    twists in place          tip move under "short_step_cm" with a forearm twist of "twist_in_place_deg" or more;
    retreats and side steps  real moves (at least "short_step_cm") that go against the line's direction of travel (more
                             than 120 degrees off it) or across it (60 to 120 degrees);
    jitter                   the share of consecutive steps whose directions on the table differ by more than
                             "jitter_angle_deg", over all steps and over real moves only.
The same numbers are given per pass through each named bend (a window of 3 cm of line before and after its middle), with
the move names of the pass, and a check whether the passes of one travel direction split into two ways (the sign of the
forearm twist summed over the pass, and the side of the line on which the tip goes round).

Run from this folder:  python s01_demonstrations_at_bends.py
Output: s01_demonstrations_at_bends.out.txt, s01_place_table.csv, s01_feature_passes.csv, s01_hairpin_paths.png.
"""

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
from matplotlib import pyplot

import common

BEND_PARAMETERS = {
    # named bends: sheet, the middle of the bend along the line from end A (cm), and what the line does there
    "features": [
        {"name": "03 first bend (125 deg in 6 cm)", "sheet": "path_03", "arc_cm": 3.6},
        {"name": "03 pointed corner (148 deg)", "sheet": "path_03", "arc_cm": 24.6},
        {"name": "03 third bend (118 deg)", "sheet": "path_03", "arc_cm": 34.8},
        {"name": "04 first bend and pointed corner (S, 191 deg)", "sheet": "path_04", "arc_cm": 9.1},
        {"name": "04 hairpin 1 (172 deg)", "sheet": "path_04", "arc_cm": 24.9},
        {"name": "04 hairpin 2 (161 deg)", "sheet": "path_04", "arc_cm": 43.9},
        {"name": "05 tight bend of the loop (114 deg)", "sheet": "path_05", "arc_cm": 24.5},
        {"name": "06 tight bend of the loop (130 deg)", "sheet": "path_06", "arc_cm": 17.5},
        {"name": "01 straight middle (control)", "sheet": "path_01", "arc_cm": 11.6},
        {"name": "03 straight tail (control)", "sheet": "path_03", "arc_cm": 45.0},
        {"name": "02 arc middle (control, 34 deg)", "sheet": "path_02", "arc_cm": 15.6}
    ],
    "feature_half_window_cm": 3.0,
    "retreat_from_deg": 120.0,
    "sideways_from_deg": 60.0,
    "progress_clip_cm": [-2.0, 3.0],
    "example_passes": 3,
    "figure_features": ["04 hairpin 1 (172 deg)", "04 hairpin 2 (161 deg)", "03 first bend (125 deg in 6 cm)", "03 pointed corner (148 deg)"],
    "figure_passes": 8
}

SHORT_NAMES = {"forward": "F", "forward_left": "FL", "forward_right": "FR", "left": "L", "right": "R", "back": "B", "back_left": "BL",
               "back_right": "BR", "small_motion": "t", "turn_left": "T", "turn_right": "T", "stop": "STOP"}



def with_step_relations(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Add what each step is relative to the next step and to the line: jumps of direction, retreats and progress.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``common.load_demonstration_frames``.
    parameters : dict
        The merged settings.

    Returns
    -------
    pandas.DataFrame
        ``frames`` with "is_real_move", "is_twist_in_place", "has_next" (the next kept row is the next recorded step),
        "jump_deg" (change of table direction to the next step), "is_pair_real" (both steps are real moves),
        "to_line_deg" (angle between the step and the line's direction of travel at the puck), "is_retreat",
        "is_sideways" and "progress_cm" (how far the puck advanced along the line to the next row).
    """

    frames = frames.copy()
    is_same_episode = (frames["episode_id"] == frames["episode_id"].shift(-1)).to_numpy()
    is_next_row = (frames["row_in_episode"].shift(-1) - frames["row_in_episode"]).to_numpy() == 1
    frames["has_next"] = is_same_episode & is_next_row & ~frames["is_stop"].to_numpy() & ~frames["is_stop"].shift(-1, fill_value = True).to_numpy()
    frames["is_real_move"] = (~frames["is_stop"]) & (frames["step_cm"] >= parameters["short_step_cm"])
    frames["is_twist_in_place"] = (~frames["is_stop"]) & (frames["step_cm"] < parameters["short_step_cm"]) & (frames["twist_deg"].abs() >= parameters["twist_in_place_deg"])
    next_direction = frames["step_table_deg"].shift(-1).to_numpy()
    frames["jump_deg"] = numpy.where(frames["has_next"], common.angle_difference_deg(frames["step_table_deg"].to_numpy(), next_direction), numpy.nan)
    frames["is_pair_real"] = frames["has_next"] & frames["is_real_move"] & frames["is_real_move"].shift(-1, fill_value = False)

    # the line's direction of travel at the puck, in the sheet's frame, against the tip's step in the same frame
    profiles = numpy.load(parameters["line_profiles_path"])
    frames["to_line_deg"] = numpy.nan
    for sheet, group in frames.groupby("sheet"):
        number = sheet[-2:]
        points = profiles[f"points_{number}"]
        arc = profiles[f"arc_{number}"]
        heading = numpy.degrees(numpy.arctan2(numpy.gradient(points[:, 1]), numpy.gradient(points[:, 0])))
        nearest = numpy.clip(numpy.searchsorted(arc, group["puck_arc_cm"].fillna(0.0).to_numpy()), 0, len(arc) - 1)
        travel_heading = numpy.where(group["travels_towards_A"].fillna(False).astype(bool), heading[nearest] + 180.0, heading[nearest])
        next_x = frames["tip_x_sheet_cm"].shift(-1).to_numpy()[group.index]
        next_y = frames["tip_y_sheet_cm"].shift(-1).to_numpy()[group.index]
        step_heading = numpy.degrees(numpy.arctan2(next_y - group["tip_y_sheet_cm"].to_numpy(), next_x - group["tip_x_sheet_cm"].to_numpy()))
        is_known = group["has_next"].to_numpy() & group["puck_arc_cm"].notna().to_numpy() & group["is_real_move"].to_numpy()
        frames.loc[group.index, "to_line_deg"] = numpy.where(is_known, common.angle_difference_deg(step_heading, travel_heading), numpy.nan)
    frames["is_retreat"] = frames["to_line_deg"] > parameters["retreat_from_deg"]
    frames["is_sideways"] = (frames["to_line_deg"] > parameters["sideways_from_deg"]) & ~frames["is_retreat"]
    progress = frames["travel_arc_cm"].shift(-1) - frames["travel_arc_cm"]
    frames["progress_cm"] = numpy.where(frames["has_next"], progress.clip(*parameters["progress_clip_cm"]), 0.0)
    frames["progress_cm"] = frames["progress_cm"].fillna(0.0)
    return frames



def summary_of(rows: pandas.DataFrame) -> Dict:
    """Return the numbers of one group of pictures (one sheet and kind of place, or one pass through a bend)."""

    moving = rows[~rows["is_stop"]]
    real = moving[moving["is_real_move"]]
    pairs = moving[moving["has_next"]]
    real_pairs = moving[moving["is_pair_real"]]
    known = real[real["to_line_deg"].notna()]
    progress = float(moving["progress_cm"].sum())
    return {
        "pictures": int(len(moving)),
        "episodes": int(moving["episode_id"].nunique()),
        "line_progress_cm": round(progress, 1),
        "steps_per_cm_of_line": round(len(moving) / progress, 2) if progress > 1.0 else numpy.nan,
        "tip_travel_per_cm_of_line": round(float(moving["step_cm"].sum()) / progress, 2) if progress > 1.0 else numpy.nan,
        "step_median_cm": round(float(moving["step_cm"].median()), 2) if len(moving) else numpy.nan,
        "share_short_steps": round(float((moving["step_cm"] < common.ANALYSIS_PARAMETERS["short_step_cm"]).mean()), 3) if len(moving) else numpy.nan,
        "share_twist_in_place": round(float(moving["is_twist_in_place"].mean()), 3) if len(moving) else numpy.nan,
        "twist_median_abs_deg": round(float(moving["twist_deg"].abs().median()), 2) if len(moving) else numpy.nan,
        "twist_per_cm_of_line_deg": round(float(moving["twist_deg"].abs().sum()) / progress, 2) if progress > 1.0 else numpy.nan,
        "share_retreat_of_real_moves": round(float(known["is_retreat"].mean()), 3) if len(known) else numpy.nan,
        "share_sideways_of_real_moves": round(float(known["is_sideways"].mean()), 3) if len(known) else numpy.nan,
        "jitter_all_steps": round(float((pairs["jump_deg"] > common.ANALYSIS_PARAMETERS["jitter_angle_deg"]).mean()), 3) if len(pairs) else numpy.nan,
        "jitter_real_moves": round(float((real_pairs["jump_deg"] > common.ANALYSIS_PARAMETERS["jitter_angle_deg"]).mean()), 3) if len(real_pairs) else numpy.nan,
        "jump_median_real_moves_deg": round(float(real_pairs["jump_deg"].median()), 1) if len(real_pairs) else numpy.nan
    }



def feature_passes(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Return one row per complete pass of a demonstration through each named bend.

    A pass is complete when the episode has pictures before and after the window, so near-end starts that begin inside
    a bend are left out.

    Returns
    -------
    pandas.DataFrame
        Per pass: the feature, the episode, the travel direction, the numbers of ``summary_of``, "net_twist_deg" (the
        forearm twist summed with its sign), "tip_side_cm" (how far the tip was from the line at the bend's middle,
        positive on the left of the direction of travel) and "moves" (the pass as a string of short move names).
    """

    profiles = numpy.load(parameters["line_profiles_path"])
    half = parameters["feature_half_window_cm"]
    rows = []
    for feature in parameters["features"]:
        number = feature["sheet"][-2:]
        points = profiles[f"points_{number}"]
        arc = profiles[f"arc_{number}"]
        middle = int(numpy.argmin(numpy.abs(arc - feature["arc_cm"])))
        tangent = points[min(middle + 3, len(points) - 1)] - points[max(middle - 3, 0)]
        tangent = tangent / numpy.linalg.norm(tangent)
        sheet_rows = frames[(frames["sheet"] == feature["sheet"]) & frames["puck_arc_cm"].notna()]
        for episode_id, episode in sheet_rows.groupby("episode_id", sort = False):
            inside = episode[(episode["puck_arc_cm"] - feature["arc_cm"]).abs() <= half]
            if len(inside) < 2:
                continue
            is_before = (episode["row_in_episode"] < inside["row_in_episode"].min()).any()
            is_after = (episode["row_in_episode"] > inside["row_in_episode"].max()).any()
            if not (is_before and is_after):
                continue
            is_towards_a = bool(inside["travels_towards_A"].iloc[0])
            travel = -tangent if is_towards_a else tangent
            left = numpy.array([-travel[1], travel[0]])
            nearest = inside.iloc[int(numpy.argmin(numpy.abs(inside["puck_arc_cm"].to_numpy() - feature["arc_cm"])))]
            tip_offset = numpy.array([nearest["tip_x_sheet_cm"], nearest["tip_y_sheet_cm"]]) - points[middle]
            summary = summary_of(inside)
            summary.update({
                "feature": feature["name"], "episode_id": episode_id, "start_kind": inside["start_kind"].iloc[0],
                "from_end": "B" if is_towards_a else "A",
                "net_twist_deg": round(float(inside["twist_deg"].sum()), 1),
                "sum_abs_twist_deg": round(float(inside["twist_deg"].abs().sum()), 1),
                "twists_in_place": int(inside["is_twist_in_place"].sum()),
                "retreat_steps": int(inside["is_retreat"].sum()),
                "sideways_steps": int(inside["is_sideways"].sum()),
                "tip_travel_cm": round(float(inside["step_cm"].sum()), 1),
                "tip_side_cm": round(float(tip_offset @ left), 2),
                "moves": " ".join(SHORT_NAMES.get(action, "?") for action in inside["action"])
            })
            rows.append(summary)
    return pandas.DataFrame(rows)



def draw_hairpin_paths(
        frames: pandas.DataFrame,
        passes: pandas.DataFrame,
        parameters: Dict
) -> None:
    """Draw, for four bends, the traced line and the tip's path of several passes from end A, in the sheet's frame."""

    profiles = numpy.load(parameters["line_profiles_path"])
    figure, axes = pyplot.subplots(1, len(parameters["figure_features"]), figsize = (5.2 * len(parameters["figure_features"]), 5.6))
    for axis, feature_name in zip(axes, parameters["figure_features"]):
        feature = next(entry for entry in parameters["features"] if entry["name"] == feature_name)
        number = feature["sheet"][-2:]
        points = profiles[f"points_{number}"]
        arc = profiles[f"arc_{number}"]
        near = numpy.abs(arc - feature["arc_cm"]) <= 7.0
        axis.plot(points[:, 0], points[:, 1], color = "0.8", linewidth = 9, solid_capstyle = "round", zorder = 1)
        axis.plot(points[near, 0], points[near, 1], color = "0.45", linewidth = 9, solid_capstyle = "round", zorder = 2)
        chosen = passes[(passes["feature"] == feature_name) & (passes["from_end"] == "A")].head(parameters["figure_passes"])
        for _, one_pass in chosen.iterrows():
            episode = frames[(frames["episode_id"] == one_pass["episode_id"]) & ((frames["puck_arc_cm"] - feature["arc_cm"]).abs() <= 5.0)]
            line, = axis.plot(episode["tip_x_sheet_cm"], episode["tip_y_sheet_cm"], "-", linewidth = 1.1, zorder = 3)
            twists = episode[episode["is_twist_in_place"]]
            axis.plot(twists["tip_x_sheet_cm"], twists["tip_y_sheet_cm"], "o", markersize = 3.5, color = line.get_color(), zorder = 4)
        middle = points[int(numpy.argmin(numpy.abs(arc - feature["arc_cm"])))]
        axis.set_xlim(middle[0] - 8.0, middle[0] + 8.0)
        axis.set_ylim(middle[1] - 8.0, middle[1] + 8.0)
        axis.set_aspect("equal")
        axis.grid(alpha = 0.3)
        axis.set_title(f"{feature_name}\n{len(chosen)} passes from end A: tip paths; dots = twists in place", fontsize = 9)
        axis.set_xlabel("sheet x (cm)")
        axis.set_ylabel("sheet y (cm)")
    figure.suptitle("Where the magnet tip goes while the puck rounds a bend (grey: the 1 cm line; the puck is about 2.5 cm ahead of the tip)", fontsize = 11)
    figure.tight_layout()
    figure.savefig(common.ANALYSIS_FOLDER / "s01_hairpin_paths.png", dpi = 110)
    pyplot.close(figure)



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(BEND_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s01_demonstrations_at_bends.out.txt")
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 400)
    frames = with_step_relations(common.load_demonstration_frames(settings), settings)
    report(f"demonstration pictures (audit cleaning): {len(frames)}; with a line position: {int(frames['puck_arc_cm'].notna().sum())}; "
           f"episodes: {frames['episode_id'].nunique()}")
    report("kind of place = the traced line's turning within 3 cm before and after the puck's place (absolute turning: straight < 15, gentle 15 to 60, "
           "sharp from 60; hairpin = the line's direction changes by 150 degrees or more)")
    report("")
    report("== pictures per sheet and kind of place ==")
    report(frames.groupby(["sheet", "place"]).size().unstack(fill_value = 0).to_string())

    rows = []
    for (sheet, place), group in frames[frames["place"] != "no fit"].groupby(["sheet", "place"]):
        rows.append({"sheet": sheet, "place": place, **summary_of(group)})
    for place, group in frames[frames["place"] != "no fit"].groupby("place"):
        rows.append({"sheet": "all sheets", "place": place, **summary_of(group)})
    rows.append({"sheet": "all sheets", "place": "all places (also no fit)", **summary_of(frames)})
    table = pandas.DataFrame(rows)
    table["place"] = pandas.Categorical(table["place"], categories = common.PLACE_NAMES + ["all places (also no fit)"], ordered = True)
    table = table.sort_values(["sheet", "place"]).reset_index(drop = True)
    table.to_csv(common.ANALYSIS_FOLDER / "s01_place_table.csv", index = False)
    report("")
    report("== the demonstrations per sheet and kind of place (measured; frames.csv through load_training_frames, line positions from the audit) ==")
    report(table.to_string(index = False))

    report("")
    report("== direction jumps between consecutive steps, by what the two steps are (all sheets) ==")
    pairs = frames[frames["has_next"]].copy()
    pairs["pair_kind"] = numpy.where(pairs["is_pair_real"], "both real moves (0.3 cm or more)", "at least one short step (under 0.3 cm)")
    for (place, pair_kind), group in pairs[pairs["place"] != "no fit"].groupby(["place", "pair_kind"]):
        report(f"{place:12s} | {pair_kind:40s} | pairs {len(group):5d} | over 90 deg: {float((group['jump_deg'] > 90).mean()):.3f} | "
               f"over 120 deg: {float((group['jump_deg'] > 120).mean()):.3f} | median jump {float(group['jump_deg'].median()):.1f} deg")

    passes = feature_passes(frames, settings)
    passes.to_csv(common.ANALYSIS_FOLDER / "s01_feature_passes.csv", index = False)
    report("")
    report("== complete passes through the named bends: medians per pass (window: 3 cm of line before and after the bend's middle) ==")
    grouped = passes.groupby(["feature", "from_end"], sort = False)
    feature_table = grouped.agg(
        passes = ("episode_id", "size"),
        steps_median = ("pictures", "median"),
        steps_least = ("pictures", "min"),
        steps_most = ("pictures", "max"),
        tip_travel_cm = ("tip_travel_cm", "median"),
        twists_in_place = ("twists_in_place", "median"),
        retreat_steps = ("retreat_steps", "median"),
        sideways_steps = ("sideways_steps", "median"),
        sum_abs_twist_deg = ("sum_abs_twist_deg", "median"),
        net_twist_deg = ("net_twist_deg", "median"),
        jitter_all = ("jitter_all_steps", "median"),
        jitter_real = ("jitter_real_moves", "median"),
        share_net_twist_positive = ("net_twist_deg", lambda values: round(float((values > 0).mean()), 2)),
        share_tip_left_of_line = ("tip_side_cm", lambda values: round(float((values > 0).mean()), 2)),
        tip_side_median_cm = ("tip_side_cm", "median")
    ).reset_index()
    report(feature_table.to_string(index = False))

    report("")
    report("== which move names the passes are made of (share of the pictures inside the window; t = twist in place or other move under 0.3 cm) ==")
    for (feature_name, from_end), group in grouped:
        names = pandas.Series(" ".join(group["moves"]).split()).value_counts(normalize = True).round(3)
        report(f"{feature_name} | from end {from_end} | {len(group)} passes | " + ", ".join(f"{name} {share:.2f}" for name, share in names.items()))

    report("")
    report("== example passes (move names in order; F forward, L left, R right, B back, t = small motion, mostly a twist in place) ==")
    for (feature_name, from_end), group in grouped:
        for _, one_pass in group.head(settings["example_passes"]).iterrows():
            report(f"{feature_name} | from {from_end} | {one_pass['episode_id'].split('/')[-1][-22:]} | {one_pass['pictures']} steps, "
                   f"net twist {one_pass['net_twist_deg']} deg | {one_pass['moves']}")

    draw_hairpin_paths(frames, passes, settings)
    report("")
    report("figure: s01_hairpin_paths.png")



if __name__ == "__main__":
    main()
