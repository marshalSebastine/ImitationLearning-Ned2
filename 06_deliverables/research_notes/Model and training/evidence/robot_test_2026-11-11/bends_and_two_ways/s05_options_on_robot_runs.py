"""
The ten robot runs of 11.11.26 again, on paper: what the chunk said on the same pictures, and what two driver rules
would have done with the logged answers.

Nothing is driven: the logged pictures go through model E once more (cache of s02_model_outputs.py, job robot_runs),
which gives the chunk the robot did not use. Per run the script counts
    asked length      the exact step's length (the robot moved 1.0 cm whatever it was);
    reversals         consecutive answers whose directions on the table differ by more than 90 degrees, for the exact
                      step and for the chunk's place 3 and 5 steps ahead;
    keep-last rule    "keep the last direction when the new one differs from it by more than 120 degrees", applied to
                      the logged answers in order (open loop: the pictures after the first changed step would have been
                      different, so only the first firing is certain);
    chunk steering    how far the chunk's direction is from the exact step's on the same picture.
For run 6 the steps round the back-and-forth are listed, for run 7 the walk round the puck that the keep-last rule
would have interrupted, and for the runs on sheets 03 and 04 the first 16 steps, beside what the demonstrations did over
their first 16 steps.

Run from this folder:  python s05_options_on_robot_runs.py
Output: s05_options_on_robot_runs.out.txt, s05_run_table.csv, s05_sheet03_first_bend.png, s05_run7_walk_round.jpg,
s05_run9_first_bend.jpg.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
from matplotlib import pyplot

import common

RUN_PARAMETERS = {
    "run_prefix": "2026-11-11_17",
    "keep_last_from_deg": 120.0,
    "reversal_from_deg": 90.0,
    "short_step_cm": 0.3,
    "short_chunk_cm": 0.5,
    # the driver's rule as set in config.py on 11.11.26 (30 pictures, a box of 3 cm, 15 degrees), and two others
    "stall_rules": {
        "driver's rule (30 pictures, box 3 cm)": {"steps": 30, "measure": "box", "extent_m": 0.03, "twist_deg": 15.0},
        "box 4 cm over 15 pictures": {"steps": 15, "measure": "box", "extent_m": 0.04, "twist_deg": 15.0},
        "net travel under 1.5 cm over 12 pictures": {"steps": 12, "measure": "net", "extent_m": 0.015, "twist_deg": 15.0}
    },
    "run_6": "2026-11-11_174108_E_sheet04_run1",
    "run_6_steps": [12, 54],
    "run_7": "2026-11-11_174448_E_sheet05_run1",
    "run_7_steps": [42, 60],
    "run_7_pictures": [44, 46, 47, 48, 49, 51, 53, 55, 56, 57, 58, 60],
    "sheet_03_runs": ["2026-11-11_173549_E_sheet03_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie", "2026-11-11_175544_B_team5_sheet03"],
    "sheet_03_steps": 16,
    "sheet_03_pictures": [0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 16, 18],
    "sheet_03_start_kinds": ["STD"],
    "sheet_03_first_cm": 12.0
}



def direction_changes(direction_deg: numpy.ndarray) -> numpy.ndarray:
    """Return the change of direction between consecutive entries, in degrees (one fewer than the entries)."""

    return common.angle_difference_deg(direction_deg[:-1], direction_deg[1:])



def keep_last_rule(
        direction_deg: numpy.ndarray,
        limit_deg: float
) -> Dict:
    """
    Apply "keep the last direction when the new one differs by more than ``limit_deg``" to a series of directions.

    Parameters
    ----------
    direction_deg : numpy.ndarray
        The answers' directions on the table, in the order they were given.
    limit_deg : float
        The largest change that is still followed.

    Returns
    -------
    dict
        "fired" (how many answers were replaced by the kept direction), "first" (the first such step, or None),
        "longest" (the longest row of replaced answers) and "used_deg" (the directions the rule would drive).
    """

    used = [float(direction_deg[0])]
    is_fired = [False]
    for direction in direction_deg[1:]:
        if common.angle_difference_deg(direction, used[-1]) > limit_deg:
            used.append(used[-1])
            is_fired.append(True)
        else:
            used.append(float(direction))
            is_fired.append(False)
    is_fired = numpy.array(is_fired)
    longest = 0
    current = 0
    for value in is_fired:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return {"fired": int(is_fired.sum()), "first": int(numpy.flatnonzero(is_fired)[0]) if is_fired.any() else None,
            "longest": int(longest), "used_deg": numpy.array(used), "is_fired": is_fired}



def save_run_pictures(
        run_name: str,
        detail: Dict,
        picture_steps: List[int],
        file_name: str,
        parameters: Dict
) -> None:
    """
    Save some logged pictures of one run side by side, each with the direction and length the model asked for.

    Parameters
    ----------
    run_name : str
        The run's folder under the model runs folder.
    detail : dict
        The run's moves and answers, as collected in ``main``.
    picture_steps : list of int
        The steps to show.
    file_name : str
        The picture file to write into this folder.
    parameters : dict
        Supplies "model_runs_folder".
    """

    tiles = []
    for step in picture_steps:
        picture = cv2.imread(str(Path(parameters["model_runs_folder"]) / run_name / "frames" / f"{step:03d}.jpg"))
        picture = cv2.resize(picture, (400, 300), interpolation = cv2.INTER_AREA)
        angle = math.atan2(detail["exact"][step, 1], detail["exact"][step, 0])
        cv2.arrowedLine(picture, (200, 150), (int(200 - 80 * math.sin(angle)), int(150 - 80 * math.cos(angle))), (0, 255, 0), 3, tipLength = 0.3)
        cv2.putText(picture, f"step {step}: asked {detail['exact_length'][step]:.2f} cm, twist {detail['moves']['model_twist_deg'].iloc[step]:+.1f}",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(picture)
    columns = 4
    while len(tiles) % columns:
        tiles.append(numpy.zeros_like(tiles[0]))
    cv2.imwrite(str(common.ANALYSIS_FOLDER / file_name), numpy.vstack([numpy.hstack(tiles[start:start + columns]) for start in range(0, len(tiles), columns)]),
                [cv2.IMWRITE_JPEG_QUALITY, 80])



def first_steps_section(
        frames: pandas.DataFrame,
        details: Dict,
        sheet_name: str,
        run_names: List[str],
        step_count: int,
        report,
        title: str
) -> None:
    """
    Print the first steps of robot runs on one sheet beside what the demonstrations did over their first steps.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``common.load_demonstration_frames``.
    details : dict
        Per run name, the moves and the directions computed in ``main``.
    sheet_name : str
        For example "path_03"; the demonstrations with a standard start at end A of this sheet are the reference.
    run_names : list of str
        The run folders to list.
    step_count : int
        How many first steps to list.
    report : callable
        Prints a line to the screen and the output file.
    title : str
        The section's heading.
    """

    report("")
    report(f"== {title} ==")
    for name in run_names:
        detail = details[name]
        moves = detail["moves"].iloc[:step_count]
        listing = pandas.DataFrame({
            "step": numpy.arange(len(moves)),
            "tip x cm": (100 * moves["tip_x"]).round(1), "tip y cm": (100 * moves["tip_y"]).round(1),
            "asked cm": detail["exact_length"][:len(moves)].round(2),
            "asked deg (picture)": moves["model_angle_deg"].round(0),
            "asked twist": moves["model_twist_deg"].round(1),
            "forearm deg": numpy.degrees(moves["joint_4"]).round(1),
            "camera heading": moves["picture_heading_deg"].round(0),
            "driven deg (table)": detail["exact_table"][:len(moves)].round(0),
            "chunk-3 deg (picture)": numpy.degrees(numpy.arctan2(detail["chunk"][:len(moves), 2, 1], detail["chunk"][:len(moves), 2, 0])).round(0),
            "chunk-5 deg (picture)": numpy.degrees(numpy.arctan2(detail["chunk"][:len(moves), 4, 1], detail["chunk"][:len(moves), 4, 0])).round(0)
        })
        report(f"-- {name} --")
        report(listing.to_string(index = False))
        report(f"over these {len(moves)} moves: asked {detail['exact_length'][:len(moves)].sum():.1f} cm, driven {len(moves)}.0 cm; twist asked {moves['model_twist_deg'].sum():+.0f} deg "
               f"(sum of sizes {moves['model_twist_deg'].abs().sum():.0f})")
    sheet = frames[(frames["sheet"] == sheet_name) & (frames["start_kind"] == "STD") & (frames["start_end"] == "A")].copy()
    report("")
    report(f"-- the demonstrations with a standard start at end A of this sheet ({sheet['episode_id'].nunique()} episodes): per step number, medians --")
    sheet["step_number"] = sheet.groupby("episode_id").cumcount()
    early = sheet[sheet["step_number"] < step_count]
    medians = early.groupby("step_number").agg(
        tip_x_cm = ("tip_x_m", lambda values: round(100 * float(values.median()), 1)),
        tip_y_cm = ("tip_y_m", lambda values: round(100 * float(values.median()), 1)),
        step_cm = ("step_cm", lambda values: round(float(values.median()), 2)),
        share_short = ("step_cm", lambda values: round(float((values < 0.3).mean()), 2)),
        twist_deg = ("twist_deg", lambda values: round(float(values.median()), 1)),
        forearm_deg = ("forearm_turn_deg", lambda values: round(float(values.median()), 1)),
        camera_heading = ("picture_heading_deg", lambda values: round(float(values.median()), 0)),
        step_table_deg = ("step_table_deg", lambda values: round(float(numpy.degrees(numpy.arctan2(numpy.sin(numpy.radians(values)).mean(), numpy.cos(numpy.radians(values)).mean()))), 0)),
        puck_along_line_cm = ("travel_arc_cm", lambda values: round(float(values.median()), 1))
    )
    report(medians.to_string())
    per_episode = early.groupby("episode_id").agg(travel = ("step_cm", "sum"), twist = ("twist_deg", lambda values: float(values.abs().sum())))
    report(f"over their first {step_count} steps the demonstrations moved the tip {per_episode['travel'].median():.1f} cm (median; from {per_episode['travel'].min():.1f} to "
           f"{per_episode['travel'].max():.1f}) and twisted the forearm by {per_episode['twist'].median():.0f} degrees in all (sum of sizes, median)")



def main(parameters: Optional[Dict] = None) -> None:

    settings = common.merged_parameters(dict(RUN_PARAMETERS, **(parameters or {})))
    report = common.TeeOutput(common.ANALYSIS_FOLDER / "s05_options_on_robot_runs.out.txt")
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 400)
    robot = numpy.load(Path(settings["cache_folder"]) / "all_sheets_e_on_robot_runs.npz")
    run_folders = sorted(Path(settings["model_runs_folder"]).glob(settings["run_prefix"] + "*"))

    rows = []
    details = {}
    for run_folder in run_folders:
        name = run_folder.name
        steps = pandas.read_csv(run_folder / "steps.csv")
        moves = steps[steps["action"] == "model_step"].reset_index(drop = True)
        count = len(moves)
        heading = moves["picture_heading_deg"].to_numpy()
        exact = moves[["model_tip_forward_cm", "model_tip_left_cm"]].to_numpy()
        chunk = robot[f"{name}__chunk"].reshape(-1, 5, 2)[:count]
        exact_e = robot[f"{name}__exact"][:count]
        is_model_e = bool(numpy.abs(exact_e[:, :2] - exact).max() < 1e-3)
        exact_length = numpy.linalg.norm(exact, axis = 1)
        exact_table = numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0])) + heading
        chunk_table = {ahead: numpy.degrees(numpy.arctan2(chunk[:, ahead - 1, 1], chunk[:, ahead - 1, 0])) + heading for ahead in (1, 3, 5)}
        chunk_length = {ahead: numpy.linalg.norm(chunk[:, ahead - 1], axis = 1) for ahead in (1, 3, 5)}
        rule_exact = keep_last_rule(exact_table, settings["keep_last_from_deg"])
        rule_chunk = keep_last_rule(chunk_table[3], settings["keep_last_from_deg"])
        apart = common.angle_difference_deg(exact_table, chunk_table[3])
        tip = moves[["tip_x", "tip_y"]].to_numpy() * 100.0
        rows.append({
            "run": name[11:], "moves": count, "pictures are model E's": is_model_e,
            "asked length median cm": round(float(numpy.median(exact_length)), 2),
            "asked under 0.3 cm": round(float((exact_length < settings["short_step_cm"]).mean()), 2),
            "asked path cm": round(float(exact_length.sum()), 1),
            "net tip travel cm": round(float(numpy.linalg.norm(tip[-1] - tip[0])), 1),
            "exact reversals": int((direction_changes(exact_table) > settings["reversal_from_deg"]).sum()),
            "chunk-3 reversals": int((direction_changes(chunk_table[3]) > settings["reversal_from_deg"]).sum()),
            "chunk-5 reversals": int((direction_changes(chunk_table[5]) > settings["reversal_from_deg"]).sum()),
            "chunk-3 length median cm": round(float(numpy.median(chunk_length[3])), 2),
            "chunk-3 under 0.5 cm": round(float((chunk_length[3] < settings["short_chunk_cm"]).mean()), 2),
            "exact to chunk-3, median deg": round(float(numpy.median(apart)), 1),
            "exact to chunk-3 over 45 deg": round(float((apart > 45).mean()), 2),
            "keep-last fired (exact)": rule_exact["fired"], "first at step": rule_exact["first"], "longest row": rule_exact["longest"],
            "keep-last fired (chunk-3)": rule_chunk["fired"],
            "kept directions": int(moves["is_direction_kept"].sum()),
            "twist asked, sum abs deg": round(float(moves["model_twist_deg"].abs().sum()), 0),
            "forearm range deg": round(float(numpy.degrees(moves["joint_4"]).max() - numpy.degrees(moves["joint_4"]).min()), 0)
        })
        details[name] = {"moves": moves, "exact_table": exact_table, "exact_length": exact_length, "chunk_table": chunk_table, "chunk_length": chunk_length,
                         "rule_exact": rule_exact, "chunk": chunk, "exact": exact}
    table = pandas.DataFrame(rows)
    table.to_csv(common.ANALYSIS_FOLDER / "s05_run_table.csv", index = False)

    # ---- the driver's stall rule (added to model_mode.py on 11.11.26 after the test) on the logged runs ----
    stall_lines = []
    for name, detail in details.items():
        moves = detail["moves"]
        tips = moves[["tip_x", "tip_y"]].to_numpy()
        forearm = numpy.degrees(moves["joint_4"].to_numpy())
        firsts = {}
        for label, rule in settings["stall_rules"].items():
            firsts[label] = None
            for last_position in range(rule["steps"] - 1, len(moves)):
                window = slice(last_position - rule["steps"] + 1, last_position + 1)
                if rule["measure"] == "box":
                    extent = float(numpy.hypot(tips[window, 0].max() - tips[window, 0].min(), tips[window, 1].max() - tips[window, 1].min()))
                else:
                    extent = float(numpy.linalg.norm(tips[last_position] - tips[last_position - rule["steps"] + 1]))
                turned = float(forearm[window].max() - forearm[window].min())
                if extent < rule["extent_m"] and turned < rule["twist_deg"]:
                    firsts[label] = last_position
                    break
        stall_lines.append(f"{name[11:]:32s} | " + " | ".join(f"{label}: {'never' if first is None else 'picture ' + str(first)}" for label, first in firsts.items()))
    report("== every run: the logged exact step, the chunk of model E on the same pictures, and the keep-last rule on the logged answers ==")
    report("(measured from steps.csv and from model E run again on frames/NNN.jpg; for runs 1 and 10 the robot's model was A and the team of five, so the chunk "
           "is what model E would have said to their pictures)")
    report(table.T.to_string())
    report("")
    report("== a stall rule on the logged tips and forearm angles (the runs were driven with the fixed 1 cm step; first picture at which each rule would end the run) ==")
    for label, rule in settings["stall_rules"].items():
        report(f"   {label}: over the last {rule['steps']} pictures, "
               f"{'the box round the tips is under' if rule['measure'] == 'box' else 'the tip ends under'} {100 * rule['extent_m']:.1f} cm "
               f"{'across' if rule['measure'] == 'box' else 'from where it began'} and the forearm turned over less than {rule['twist_deg']:.0f} degrees")
    for line in stall_lines:
        report(line)

    # ---- run 6 in detail ----
    name = settings["run_6"]
    detail = details[name]
    first, last = settings["run_6_steps"]
    listing = pandas.DataFrame({
        "step": numpy.arange(len(detail["moves"])),
        "exact cm": detail["exact_length"].round(2),
        "exact deg (picture)": numpy.degrees(numpy.arctan2(detail["exact"][:, 1], detail["exact"][:, 0])).round(0),
        "twist": detail["moves"]["model_twist_deg"].round(1),
        "chunk-3 cm": detail["chunk_length"][3].round(2),
        "chunk-3 deg (picture)": numpy.degrees(numpy.arctan2(detail["chunk"][:, 2, 1], detail["chunk"][:, 2, 0])).round(0),
        "chunk-5 cm": detail["chunk_length"][5].round(2),
        "chunk-5 deg (picture)": numpy.degrees(numpy.arctan2(detail["chunk"][:, 4, 1], detail["chunk"][:, 4, 0])).round(0),
        "driven deg (table)": detail["exact_table"].round(0),
        "keep-last would drive (table)": detail["rule_exact"]["used_deg"].round(0),
        "rule fired": detail["rule_exact"]["is_fired"]
    })
    report("")
    report(f"== run 6 ({name}), steps {first} to {last}: exact step, chunk and the keep-last rule ==")
    report(listing.iloc[first:last + 1].to_string(index = False))
    flip = listing.iloc[19:51]
    report("")
    report(f"run 6, steps 19 to 50: exact step median {flip['exact cm'].median():.2f} cm; chunk-3 median {flip['chunk-3 cm'].median():.2f} cm; chunk-5 median {flip['chunk-5 cm'].median():.2f} cm "
           f"(steps 0 to 4: chunk-3 {listing.iloc[0:5]['chunk-3 cm'].median():.2f}, chunk-5 {listing.iloc[0:5]['chunk-5 cm'].median():.2f})")
    for label, column in (("exact step", "exact deg (picture)"), ("chunk-3", "chunk-3 deg (picture)"), ("chunk-5", "chunk-5 deg (picture)")):
        on_table = flip[column].to_numpy() + detail["moves"]["picture_heading_deg"].to_numpy()[19:51]
        report(f"run 6, steps 19 to 50: {label}: reversals over 90 deg between consecutive answers: {int((direction_changes(on_table) > 90).sum())} of {len(on_table) - 1}; "
               f"answers to the picture's left: {int((flip[column] > 0).sum())}, to its right: {int((flip[column] < 0).sum())}")

    # ---- run 7, the one run that went round a loop: where the keep-last rule would have stepped in ----
    name = settings["run_7"]
    detail = details[name]
    first, last = settings["run_7_steps"]
    listing = pandas.DataFrame({
        "step": numpy.arange(len(detail["moves"])),
        "exact cm": detail["exact_length"].round(2),
        "exact deg (picture)": numpy.degrees(numpy.arctan2(detail["exact"][:, 1], detail["exact"][:, 0])).round(0),
        "twist": detail["moves"]["model_twist_deg"].round(1),
        "forearm deg": numpy.degrees(detail["moves"]["joint_4"]).round(1),
        "driven deg (table)": ((detail["exact_table"] + 180.0) % 360.0 - 180.0).round(0),
        "keep-last would drive (table)": ((detail["rule_exact"]["used_deg"] + 180.0) % 360.0 - 180.0).round(0),
        "rule fired": detail["rule_exact"]["is_fired"]
    })
    report("")
    report(f"== run 7 ({name}), steps {first} to {last}: the walk round the puck at the loop's tight bend, which the run completed, and the keep-last rule ==")
    report(listing.iloc[first:last + 1].to_string(index = False))
    save_run_pictures(name, detail, settings["run_7_pictures"], "s05_run7_walk_round.jpg", settings)
    report("figure: s05_run7_walk_round.jpg (the pictures of these steps with the asked direction)")
    save_run_pictures(settings["sheet_03_runs"][1], details[settings["sheet_03_runs"][1]], settings["sheet_03_pictures"], "s05_run9_first_bend.jpg", settings)
    report("figure: s05_run9_first_bend.jpg (run 9 at the first bend of sheet 03: the swing to the outside of the bend, driven 1 cm per picture)")

    # ---- the first bends of sheets 03 and 04 ----
    frames = common.load_demonstration_frames(settings)
    first_steps_section(frames, details, "path_03", settings["sheet_03_runs"], settings["sheet_03_steps"], report,
                        "sheet 03, the first bend (125 degrees of turning within 6 cm, to the left on the table): the first steps of the three runs")
    first_steps_section(frames, details, "path_04", [settings["run_6"]], settings["sheet_03_steps"], report,
                        "sheet 04, the first bend (to the right on the table, then the pointed corner to the left): the first steps of run 6")
    sheet = frames[(frames["sheet"] == "path_03") & (frames["start_kind"].isin(settings["sheet_03_start_kinds"])) & (frames["start_end"] == "A")]

    figure, axis = pyplot.subplots(figsize = (8.5, 7.5))
    for episode_id, episode in sheet.groupby("episode_id"):
        axis.plot(100 * episode["tip_x_m"], 100 * episode["tip_y_m"], color = "0.6", linewidth = 0.9, zorder = 1)
    for name, colour in zip(settings["sheet_03_runs"], ["tab:red", "tab:blue", "tab:green"]):
        moves = details[name]["moves"]
        axis.plot(100 * moves["tip_x"], 100 * moves["tip_y"], "-", marker = ".", color = colour, linewidth = 1.4, label = name[11:], zorder = 3)
        axis.plot(100 * moves["tip_x"].iloc[0], 100 * moves["tip_y"].iloc[0], "o", color = colour, zorder = 4)
    axis.set_aspect("equal")
    axis.grid(alpha = 0.3)
    axis.set_xlabel("x from the robot's base (cm)")
    axis.set_ylabel("y (cm), the robot's left is up")
    axis.set_title("Sheet 03 from end A: where the magnet tip went\n(grey: the demonstrations with a standard start; colours: the three robot runs)", fontsize = 10)
    axis.legend(fontsize = 8)
    figure.tight_layout()
    figure.savefig(common.ANALYSIS_FOLDER / "s05_sheet03_first_bend.png", dpi = 110)
    pyplot.close(figure)
    report("")
    report("figure: s05_sheet03_first_bend.png")



if __name__ == "__main__":
    main()
