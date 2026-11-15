"""
Verifier's own recomputation of the numbers taken from the ten robot runs of 11.11.26 (steps.csv only, no model).

Checked here: the asked step length per run, the first 16 moves at the first bends, the back-and-forth of run 6, the
stall rules, the keep-the-last-direction rule and the start poses against the demonstrations' standard start.

Read-only: reads 03_data/model_runs/*/steps.csv and frames.csv; prints to the screen and to v03_robot_runs.out.txt.
Nothing connects to the robot.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas

RUN_PARAMETERS = {
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "frames_path": Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv"),
    "runs": ["2026-11-11_172021_check_no_puck", "2026-11-11_172421_E_sheet01_run1", "2026-11-11_172743_E_sheet02_run1",
             "2026-11-11_173036_E_sheet02_run2", "2026-11-11_173549_E_sheet03_run1", "2026-11-11_174108_E_sheet04_run1",
             "2026-11-11_174448_E_sheet05_run1", "2026-11-11_174936_E_sheet06_run1", "2026-11-11_175203_E_sheet03_run2_no_hoodie",
             "2026-11-11_175544_B_team5_sheet03"],
    "short_cm": 0.3,
    "first_moves": 16,
    "flip_limit_deg": 120.0,
    "stall_rules": {"driver (30 pictures, diagonal 3 cm, 15 deg)": [30, 3.0, 15.0, "diagonal"],
                    "15 pictures, diagonal 4 cm, 15 deg": [15, 4.0, 15.0, "diagonal"],
                    "15 pictures, longest side 4 cm, 15 deg": [15, 4.0, 15.0, "side"],
                    "12 pictures, net travel 1.5 cm, 15 deg": [12, 1.5, 15.0, "net"]}
}

OUTPUT_LINES: List[str] = []


def say(*values) -> None:
    """Print and remember a line for the output file."""

    text = " ".join(str(value) for value in values)
    print(text, flush = True)
    OUTPUT_LINES.append(text)


def angle_between(first_deg: float, second_deg: float) -> float:
    """Return the size of the angle between two directions, 0 to 180 degrees."""

    return abs((first_deg - second_deg + 180.0) % 360.0 - 180.0)


def load_run(name: str, parameters: Dict) -> pandas.DataFrame:
    """Return the move rows of one run with the asked length and the asked direction on the table added."""

    steps = pandas.read_csv(parameters["model_runs_folder"] / name / "steps.csv")
    moves = steps[steps["action"] == "model_step"].copy().reset_index(drop = True)
    moves["asked_cm"] = numpy.hypot(moves["model_tip_forward_cm"], moves["model_tip_left_cm"])
    moves["asked_picture_deg"] = numpy.degrees(numpy.arctan2(moves["model_tip_left_cm"], moves["model_tip_forward_cm"]))
    moves["asked_table_deg"] = (moves["picture_heading_deg"] + moves["asked_picture_deg"] + 180.0) % 360.0 - 180.0
    moves["driven_cm"] = 100.0 * numpy.hypot(moves["tip_next_x"] - moves["tip_x"], moves["tip_next_y"] - moves["tip_y"])
    moves["driven_table_deg"] = numpy.degrees(numpy.arctan2(moves["tip_next_y"] - moves["tip_y"], moves["tip_next_x"] - moves["tip_x"]))
    moves["forearm_deg"] = numpy.degrees(moves["joint_4"])
    return moves


def stall_first_picture(moves: pandas.DataFrame, rule: List) -> Optional[int]:
    """Return the first picture (step number) at which a stall rule would end the run, or None."""

    count, extent_cm, twist_deg, kind = rule
    tips = 100.0 * moves[["tip_x", "tip_y"]].to_numpy()
    forearm = moves["forearm_deg"].to_numpy()
    for last in range(count - 1, len(moves)):
        window = tips[last - count + 1:last + 1]
        spans = window.max(axis = 0) - window.min(axis = 0)
        if kind == "diagonal":
            extent = float(numpy.hypot(*spans))
        elif kind == "side":
            extent = float(spans.max())
        else:
            extent = float(numpy.linalg.norm(window[-1] - window[0]))
        turned = float(forearm[last - count + 1:last + 1].max() - forearm[last - count + 1:last + 1].min())
        if extent < extent_cm and turned < twist_deg:
            return int(moves["step"].iloc[last])
    return None


def keep_last_rule(moves: pandas.DataFrame, flip_limit_deg: float) -> Dict:
    """
    Apply "keep the last direction when the new one flips by more than the limit" to the logged answers, in order.

    The comparison is made against the direction the rule itself would have driven (so a kept direction stays the
    reference), on the table. Open loop: the pictures are the logged ones.
    """

    fired = []
    driven_last = None
    for _, row in moves.iterrows():
        asked = float(row["asked_table_deg"])
        if driven_last is not None and angle_between(asked, driven_last) > flip_limit_deg:
            fired.append(int(row["step"]))
        else:
            driven_last = asked
    longest, current, previous = 0, 0, None
    for step in fired:
        current = current + 1 if previous is not None and step == previous + 1 else 1
        longest = max(longest, current)
        previous = step
    return {"fired": fired, "count": len(fired), "longest_row": longest}


if __name__ == "__main__":
    settings = dict(RUN_PARAMETERS)
    runs = {name: load_run(name, settings) for name in settings["runs"]}

    say("== 1. per run: what the model asked against what was driven (measured, steps.csv) ==")
    for number, (name, moves) in enumerate(runs.items(), start = 1):
        twist_asked = moves["model_twist_deg"].abs().sum()
        say(f"run {number:2d} {name[11:]:32s} moves {len(moves):3d} | asked median {moves['asked_cm'].median():.2f} cm, under 0.3 cm "
            f"{100 * (moves['asked_cm'] < settings['short_cm']).mean():4.1f} % | asked path {moves['asked_cm'].sum():5.1f} cm, driven "
            f"{moves['driven_cm'].sum():5.1f} cm | twist: sum of sizes asked {twist_asked:5.1f} deg, per cm driven "
            f"{twist_asked / moves['driven_cm'].sum():.2f}, per cm asked {twist_asked / moves['asked_cm'].sum():.2f} | "
            f"direction kept {int(moves['is_direction_kept'].sum())}, turn clamped {int(moves['is_turn_clamped'].sum())}")

    say("")
    say(f"== 2. the first {settings['first_moves']} moves (first bend of sheets 03 and 04) ==")
    for number in [5, 9, 10, 6]:
        name = settings["runs"][number - 1]
        first = runs[name].iloc[:settings["first_moves"]]
        say(f"run {number:2d}: asked {first['asked_cm'].sum():.1f} cm, driven {first['driven_cm'].sum():.1f} cm, ratio driven/asked "
            f"{first['driven_cm'].sum() / first['asked_cm'].sum():.2f}; twist asked net {first['model_twist_deg'].sum():+.0f} deg, sum of sizes "
            f"{first['model_twist_deg'].abs().sum():.0f}; twist used net {first['twist_used_deg'].sum():+.0f}; asked length steps 3 to 7: "
            f"{first['asked_cm'].iloc[3:8].min():.2f} to {first['asked_cm'].iloc[3:8].max():.2f} cm; forearm at start "
            f"{first['forearm_deg'].iloc[0]:+.1f} deg; tip start ({100 * first['tip_x'].iloc[0]:.1f}, {100 * first['tip_y'].iloc[0]:.1f}) cm")

    say("")
    say("== 3. run 6, steps 19 to 50 (the back-and-forth) ==")
    six = runs[settings["runs"][5]]
    part = six[(six["step"] >= 19) & (six["step"] <= 50)]
    picture_flips = sum(angle_between(a, b) > 90.0 for a, b in zip(part["asked_picture_deg"].iloc[:-1], part["asked_picture_deg"].iloc[1:]))
    table_flips = sum(angle_between(a, b) > 90.0 for a, b in zip(part["asked_table_deg"].iloc[:-1], part["asked_table_deg"].iloc[1:]))
    tips = 100.0 * part[["tip_x", "tip_y"]].to_numpy()
    say(f"answers {len(part)}; asked length median {part['asked_cm'].median():.2f} cm, mean {part['asked_cm'].mean():.2f}, sum "
        f"{part['asked_cm'].sum():.1f} cm, under 0.3 cm {int((part['asked_cm'] < 0.3).sum())} of {len(part)}")
    say(f"reversals over 90 deg between consecutive answers: picture frame {picture_flips} of {len(part) - 1}, table frame {table_flips} of {len(part) - 1}")
    say(f"tip box: x {tips[:, 0].min():.1f} to {tips[:, 0].max():.1f} cm, y {tips[:, 1].min():.1f} to {tips[:, 1].max():.1f} cm "
        f"(sides {numpy.ptp(tips[:, 0]):.1f} x {numpy.ptp(tips[:, 1]):.1f} cm, diagonal {float(numpy.hypot(numpy.ptp(tips[:, 0]), numpy.ptp(tips[:, 1]))):.1f} cm)")
    say(f"net travel first to last tip of these pictures: {float(numpy.linalg.norm(tips[-1] - tips[0])):.1f} cm over {len(part) - 1} moves; "
        f"driven {part['driven_cm'].iloc[:-1].sum():.1f} cm; forearm {part['forearm_deg'].min():+.1f} to {part['forearm_deg'].max():+.1f} deg")
    say("sign of twist against the camera's turning (run 6, all moves): correlation of the twist used with the change of picture heading "
        f"{float(numpy.corrcoef(six['twist_used_deg'].iloc[:-1], numpy.diff(six['picture_heading_deg']))[0, 1]):+.2f}")
    left = part[part["model_tip_left_cm"] > 0]
    right = part[part["model_tip_left_cm"] < 0]
    say(f"answers to the picture's left: {len(left)}, twist positive in {100 * (left['model_twist_deg'] > 0).mean():.0f} %; answers to its right: "
        f"{len(right)}, twist negative in {100 * (right['model_twist_deg'] < 0).mean():.0f} %")
    whole_flips = sum(angle_between(a, b) > 90.0 for a, b in zip(six["asked_picture_deg"].iloc[:-1], six["asked_picture_deg"].iloc[1:]))
    say(f"whole run 6: reversals over 90 deg (picture frame) {whole_flips} of {len(six) - 1}")

    say("")
    say("== 4. stall rules on the logged tips and forearm angles (first picture at which the rule would end the run) ==")
    for label, rule in settings["stall_rules"].items():
        results = [stall_first_picture(moves, rule) for moves in runs.values()]
        say(f"{label:45s}: " + ", ".join(f"run {number} {'never' if result is None else 'picture ' + str(result)}" for number, result in enumerate(results, start = 1)))
    tips_all = 100.0 * six[["tip_x", "tip_y"]].to_numpy()
    for count in [30, 20, 15, 12]:
        diagonals = [float(numpy.hypot(*(tips_all[last - count + 1:last + 1].max(axis = 0) - tips_all[last - count + 1:last + 1].min(axis = 0))))
                     for last in range(count - 1, len(six))]
        say(f"run 6: smallest diagonal of the tip box over any {count} pictures in a row: {min(diagonals):.2f} cm")
    for number, (name, moves) in enumerate(runs.items(), start = 1):
        tips_run = 100.0 * moves[["tip_x", "tip_y"]].to_numpy()
        forearm = moves["forearm_deg"].to_numpy()
        best = None
        for last in range(14, len(moves)):
            window = tips_run[last - 14:last + 1]
            diagonal = float(numpy.hypot(*(window.max(axis = 0) - window.min(axis = 0))))
            turned = float(numpy.ptp(forearm[last - 14:last + 1]))
            if turned < 15.0 and (best is None or diagonal < best):
                best = diagonal
        say(f"run {number}: smallest 15-picture diagonal among windows with under 15 deg of forearm turn: {'none' if best is None else f'{best:.2f} cm'}")

    say("")
    say(f"== 5. keep the last direction when the new one flips by more than {settings['flip_limit_deg']:.0f} deg (open loop on the logged answers) ==")
    for number, (name, moves) in enumerate(runs.items(), start = 1):
        result = keep_last_rule(moves, settings["flip_limit_deg"])
        say(f"run {number:2d}: fires {result['count']:2d} times, longest row {result['longest_row']}, at steps {result['fired']}")
    seven = runs[settings["runs"][6]]
    say("run 7, steps 45 to 56 (asked cm, asked direction on the table, twist, forearm):")
    for _, row in seven[(seven["step"] >= 45) & (seven["step"] <= 56)].iterrows():
        say(f"   step {int(row['step'])}: {row['asked_cm']:.2f} cm, table {row['asked_table_deg']:+6.0f} deg, picture {row['asked_picture_deg']:+6.0f} deg, "
            f"twist {row['model_twist_deg']:+.1f}, forearm {row['forearm_deg']:+.1f}")
    span = seven[(seven["step"] >= 42) & (seven["step"] <= 60)]
    say(f"run 7, steps 42 to 60: forearm from {span['forearm_deg'].iloc[0]:+.1f} to {span['forearm_deg'].iloc[-1]:+.1f} deg")

    say("")
    say("== 6. start poses: the runs against the demonstrations' standard start (frames.csv, first row of every episode) ==")
    frames = pandas.read_csv(settings["frames_path"], low_memory = False)
    first_rows = frames.sort_values(["episode_id", "row_in_episode"]).groupby("episode_id").head(1)
    standard = first_rows[first_rows["start_kind"] == "STD"]
    say(f"standard-start episodes: {len(standard)}; tip x {100 * standard['tip_x_m'].min():.2f} to {100 * standard['tip_x_m'].max():.2f} cm "
        f"(median {100 * standard['tip_x_m'].median():.2f}), tip y {100 * standard['tip_y_m'].min():.2f} to {100 * standard['tip_y_m'].max():.2f} "
        f"(median {100 * standard['tip_y_m'].median():.2f}); joint 4 {numpy.degrees(standard['joint_4']).min():+.2f} to {numpy.degrees(standard['joint_4']).max():+.2f} deg")
    say("all start kinds, first row: tip x, tip y (cm), joint 4 (deg), median and range")
    for kind, group in first_rows.groupby("start_kind"):
        say(f"   {kind:6s} n {len(group):3d}: x {100 * group['tip_x_m'].median():5.1f} ({100 * group['tip_x_m'].min():5.1f} to {100 * group['tip_x_m'].max():5.1f}), "
            f"y {100 * group['tip_y_m'].median():+5.1f} ({100 * group['tip_y_m'].min():+5.1f} to {100 * group['tip_y_m'].max():+5.1f}), "
            f"joint 4 {numpy.degrees(group['joint_4']).median():+6.1f} ({numpy.degrees(group['joint_4']).min():+6.1f} to {numpy.degrees(group['joint_4']).max():+6.1f})")
    centre = numpy.array([100 * standard["tip_x_m"].median(), 100 * standard["tip_y_m"].median()])
    for number, (name, moves) in enumerate(runs.items(), start = 1):
        start = 100.0 * moves[["tip_x", "tip_y"]].iloc[0].to_numpy()
        say(f"run {number:2d}: tip start ({start[0]:.1f}, {start[1]:+.1f}) cm, {float(numpy.linalg.norm(start - centre)):.1f} cm from the standard start; "
            f"forearm {moves['forearm_deg'].iloc[0]:+.1f} deg; picture heading {moves['picture_heading_deg'].iloc[0]:+.1f} deg; joints 5 and 6 "
            f"{math.degrees(moves['joint_5'].iloc[0]):+.1f}, {math.degrees(moves['joint_6'].iloc[0]):+.1f} deg")
    say(f"demonstrations, standard start, first row: picture heading median {standard['picture_heading_deg'].median():+.1f} deg "
        f"({standard['picture_heading_deg'].min():+.1f} to {standard['picture_heading_deg'].max():+.1f}); joint 5 median "
        f"{numpy.degrees(standard['joint_5']).median():+.1f}, joint 6 median {numpy.degrees(standard['joint_6']).median():+.1f}")

    (Path(__file__).resolve().parent / "v03_robot_runs.out.txt").write_text("\n".join(OUTPUT_LINES) + "\n", encoding = "utf-8")
