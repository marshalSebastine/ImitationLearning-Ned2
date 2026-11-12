"""
What the held wrist does to the rod's lean, and how the rod leaned against the push direction (key: wrist_and_camera,
part 4).

Part 1 (calculated with the arm model): with the wrist held, the rod's lean is fixed by where the tip is and how far
the forearm is twisted. The lean is tabulated over reach and twist for the driver's 13.5 degrees and for two other
wrist values, with the tip 5 mm above the fitted table as the driver puts it, and the reach at which the driver's
58-degree lean limit or the arm's joint limits would refuse a step is read off.

Part 2 (measured joints, calculated leans): the rod's lean along the push direction (positive: the tip leads, the
block's body follows) and across it, for the demonstrations and for the ten robot runs. A rod that leans back against
the push would put the block's body ahead of the tip.

Part 3: the hand (joint 6) in the runs and in the demonstrations, and what the model's hand output asked for.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe lean_and_reach.py
"""

import math
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import wrist_geometry
from config import MODEL_MODE_PARAMETERS


LEAN_PARAMETERS = {
    "wrists_deg": [13.5, 0.0, -10.0],
    "reaches_m": [round(0.12 + 0.01 * index, 2) for index in range(38)],
    "twists_deg": [-117.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 117.0],
    "twist_march_deg": 3.0,
    # joints 1 to 3 of a recorded pushing pose near the base (run 5, picture 0), the solver's first guess
    "first_guess_rad": [0.0, -0.743, -1.134],
    "rod_lean_most_deg": MODEL_MODE_PARAMETERS["rod_lean_most_deg"],
    "smallest_push_m": 0.005,
    "figure_dpi": 120
}



def lean_table(
        wrist_deg: float,
        parameters: Dict
) -> pandas.DataFrame:
    """
    The rod's lean from straight down over reach and forearm twist for one held wrist value, the tip on the x axis at
    the driver's height. Each reach starts from the solution of the reach before it, each twist from the twist before
    it, so the arm stays in the elbow-up shape of the recordings. NaN where no joint angles within the limits reach
    the point.
    """

    reaches = numpy.array(parameters["reaches_m"])
    tips = numpy.column_stack([reaches, numpy.zeros(len(reaches)), numpy.zeros(len(reaches))])
    tips[:, 2] = wrist_geometry.table_heights(tips[:, :2]) + wrist_geometry.ANALYSIS_PARAMETERS["driver_tip_above_table_m"]
    wrist_rad = math.radians(wrist_deg)

    untwisted = numpy.zeros((len(reaches), 3))
    guess = numpy.array(parameters["first_guess_rad"], dtype = float)
    for index in range(len(reaches)):
        joints, is_solved = wrist_geometry.solve_many_with_wrist_fixed(tips[index:index + 1], numpy.zeros(1), numpy.full(1, wrist_rad), guess[None, :])
        untwisted[index] = joints[0, :3] if is_solved[0] else numpy.nan
        if is_solved[0]:
            guess = joints[0, :3]

    table = pandas.DataFrame(index = [round(100 * reach) for reach in reaches], columns = parameters["twists_deg"], dtype = float)
    for direction in (1.0, -1.0):
        current = untwisted.copy()
        is_alive = numpy.all(numpy.isfinite(current), axis = 1)
        twist = 0.0
        while abs(twist) <= max(abs(value) for value in parameters["twists_deg"]) + 1e-9:
            starts = numpy.where(is_alive[:, None], current, 0.0)
            joints, is_solved = wrist_geometry.solve_many_with_wrist_fixed(tips, numpy.full(len(reaches), math.radians(twist)), numpy.full(len(reaches), wrist_rad), starts)
            is_alive = is_alive & is_solved
            current = numpy.where(is_alive[:, None], joints[:, :3], current)
            for wanted in parameters["twists_deg"]:
                if abs(wanted - twist) < 1e-6:
                    leans = wrist_geometry.tips_and_leans(joints)[1]
                    table[wanted] = numpy.where(is_alive, leans, numpy.nan)
            twist += direction * parameters["twist_march_deg"]
    return table



def leans_against_push(
        joint_sets: numpy.ndarray,
        push_xy: numpy.ndarray
) -> pandas.DataFrame:
    """
    The rod's lean along and across the push direction for many rows.

    Parameters
    ----------
    joint_sets : numpy.ndarray
        Joint angles in radians, shape (N, 6).
    push_xy : numpy.ndarray
        The push direction on the table for every row (need not be unit length), shape (N, 2).

    Returns
    -------
    pandas.DataFrame
        "lean_along_deg" (positive: the tip is ahead of the rod's upper end in the push direction), "lean_across_deg"
        (positive: the tip is to the left of the push) and "lean_deg" (the angle from straight down).
    """

    rod = wrist_geometry.frames_of_many(joint_sets)["joint_6"][1][:, :, 2]
    push = push_xy / numpy.linalg.norm(push_xy, axis = 1, keepdims = True)
    left = numpy.column_stack([-push[:, 1], push[:, 0]])
    return pandas.DataFrame({
        "lean_along_deg": numpy.degrees(numpy.arctan2(rod[:, 0] * push[:, 0] + rod[:, 1] * push[:, 1], -rod[:, 2])),
        "lean_across_deg": numpy.degrees(numpy.arctan2(rod[:, 0] * left[:, 0] + rod[:, 1] * left[:, 1], -rod[:, 2])),
        "lean_deg": numpy.degrees(numpy.arccos(numpy.clip(-rod[:, 2], -1.0, 1.0)))
    })



def lean_summary(leans: pandas.DataFrame) -> Dict[str, float]:
    """The numbers printed for one group of pushes."""

    return {
        "pushes": len(leans),
        "lean_median_deg": leans["lean_deg"].median(),
        "lean_p95_deg": leans["lean_deg"].quantile(0.95),
        "along_p05_deg": leans["lean_along_deg"].quantile(0.05),
        "along_median_deg": leans["lean_along_deg"].median(),
        "along_p95_deg": leans["lean_along_deg"].quantile(0.95),
        "back_over_10_share": float((leans["lean_along_deg"] < -10.0).mean()),
        "back_over_20_share": float((leans["lean_along_deg"] < -20.0).mean()),
        "across_abs_median_deg": leans["lean_across_deg"].abs().median(),
        "across_abs_p95_deg": leans["lean_across_deg"].abs().quantile(0.95)
    }



def main(parameters: Optional[Dict] = None) -> None:
    """Print the three parts and save the figure."""

    settings = dict(LEAN_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 200)
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]

    print("=== part 1: the rod's lean (degrees from straight down) with the wrist held, by reach (cm, rows) and forearm twist (degrees, columns) ===")
    tables = {}
    for wrist_deg in settings["wrists_deg"]:
        tables[wrist_deg] = lean_table(wrist_deg, settings)
        print(f"\nwrist held at {wrist_deg} degrees (blank: no joint angles within the arm model's limits reach the point)")
        print(tables[wrist_deg].iloc[::2].round(1).to_string(na_rep = ""))
        within = tables[wrist_deg] <= settings["rod_lean_most_deg"]
        furthest = {twist: (max(within.index[within[twist]]) if within[twist].any() else None) for twist in tables[wrist_deg].columns}
        nearest = {twist: (min(within.index[within[twist]]) if within[twist].any() else None) for twist in tables[wrist_deg].columns}
        print(f"  reach (cm) within the {settings['rod_lean_most_deg']:.0f}-degree lean limit and the joint limits, by twist: "
              + ", ".join(f"{twist:+.0f}: {nearest[twist]} to {furthest[twist]}" for twist in tables[wrist_deg].columns))

    frames = wrist_geometry.load_demonstration_frames()
    print(f"\ndemonstrated reach: 1st to 99th percentile {100 * frames['reach_m'].quantile(0.01):.1f} to {100 * frames['reach_m'].quantile(0.99):.1f} cm, "
          f"largest {100 * frames['reach_m'].max():.1f} cm; rows beyond 44 cm: {int((frames['reach_m'] > 0.44).sum())}, beyond 46 cm: {int((frames['reach_m'] > 0.46).sum())}")

    print("\n=== part 2: the rod's lean against the push direction ===")
    episodes = pandas.read_csv(wrist_geometry.ANALYSIS_PARAMETERS["episodes_file"])
    on_mark = episodes[episodes["sheet_orientation"].str.contains("START MARK")]["episode_id"]
    has_next = (frames["episode_id"].shift(-1) == frames["episode_id"]).to_numpy()
    rows_now = numpy.flatnonzero(has_next)
    tips = frames[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    moves = tips[rows_now + 1] - tips[rows_now]
    is_push = numpy.linalg.norm(moves, axis = 1) >= settings["smallest_push_m"]
    rows_now, moves = rows_now[is_push], moves[is_push]
    demonstrated = leans_against_push(frames[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)[rows_now], moves)
    demonstrated["sheet"] = frames["sheet"].to_numpy()[rows_now]
    demonstrated["is_on_mark"] = frames["episode_id"].isin(on_mark).to_numpy()[rows_now]
    demonstrated["wrist_deg"] = frames["joint_5_deg"].to_numpy()[rows_now]

    rows = []
    for sheet, group in demonstrated[demonstrated["is_on_mark"]].groupby("sheet"):
        rows.append(dict({"who": "demonstrations (start mark)", "sheet": sheet[-2:]}, **lean_summary(group)))
    run_leans = {}
    for run in wrist_geometry.ANALYSIS_PARAMETERS["robot_runs"]:
        steps = wrist_geometry.load_robot_run(run)
        steps = steps[(steps["action"] == "model_step") & steps["tip_next_x"].notna()]
        push = numpy.column_stack([steps["tip_next_x"] - steps["tip_x"], steps["tip_next_y"] - steps["tip_y"]])
        leans = leans_against_push(steps[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float), push)
        run_leans[run["number"]] = leans
        rows.append(dict({"who": f"robot run {run['number']} (model {run['model']})", "sheet": run["sheet"][-2:]}, **lean_summary(leans)))
    summary = pandas.DataFrame(rows).sort_values(["sheet", "who"])
    print(summary.round(2).to_string(index = False))
    summary.to_csv(output_folder / "lean_against_push.csv", index = False)
    near_held = demonstrated[demonstrated["is_on_mark"] & ((demonstrated["wrist_deg"] - 13.5).abs() <= 2.0)]
    far_from_held = demonstrated[demonstrated["is_on_mark"] & ((demonstrated["wrist_deg"] - 13.5).abs() > 5.7)]
    print("\ndemonstrated pushes (start mark) with the wrist within 2 degrees of 13.5, and more than 5.7 degrees away:")
    print(pandas.DataFrame([dict({"wrist": "within 2 of 13.5"}, **lean_summary(near_held)), dict({"wrist": "over 5.7 away"}, **lean_summary(far_from_held))]).round(2).to_string(index = False))

    print("\n=== part 3: the hand (joint 6) ===")
    hand = frames["joint_6_deg"]
    held = wrist_geometry.ANALYSIS_PARAMETERS["hand_held_deg"]
    print(f"demonstrations: hand within 0.3 degrees of the held {held} in {100.0 * ((hand - held).abs() <= 0.3).mean():.1f} % of rows, within 0.3 of 0 in "
          f"{100.0 * (hand.abs() <= 0.3).mean():.1f} %, more than 10 from {held} in {100.0 * ((hand - held).abs() > 10.0).mean():.1f} % ({int(((hand - held).abs() > 10.0).sum())} rows)")
    hand_change = frames.groupby("episode_id")["joint_6_deg"].diff()
    print(f"demonstrated steps in which the hand turned more than 1 degree: {int((hand_change.abs() > 1.0).sum())} of {int(hand_change.notna().sum())}")
    sample = frames[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)[::37]
    turned = sample.copy()
    turned[:, 5] += math.radians(30.0)
    tip_move = 1000.0 * numpy.linalg.norm(wrist_geometry.tips_and_leans(turned)[0] - wrist_geometry.tips_and_leans(sample)[0], axis = 1).max()
    camera_move = 1000.0 * numpy.linalg.norm(wrist_geometry.camera_poses(turned)[0] - wrist_geometry.camera_poses(sample)[0], axis = 1).max()
    print(f"arm model: turning the hand by 30 degrees moves the tip by at most {tip_move:.6f} mm and the camera by at most {camera_move:.6f} mm ({len(sample)} poses)")
    for run in wrist_geometry.ANALYSIS_PARAMETERS["robot_runs"]:
        steps = wrist_geometry.load_robot_run(run)
        print(f"run {run['number']:2d}: logged hand {steps['joint_6_deg'].min():.2f} to {steps['joint_6_deg'].max():.2f} degrees; the model's hand output "
              f"{steps['model_hand_deg'].min():+.3f} to {steps['model_hand_deg'].max():+.3f} degrees per step, summed over the run {steps['model_hand_deg'].sum():+.2f}; "
              f"its lean output {steps['model_lean_deg'].min():+.2f} to {steps['model_lean_deg'].max():+.2f}, summed {steps['model_lean_deg'].sum():+.1f}; "
              f"the lean the held wrist gave: {run_leans[run['number']]['lean_deg'].iloc[0]:.1f} at the first step, {run_leans[run['number']]['lean_deg'].max():.1f} at most")

    figure, axes = pyplot.subplots(1, 2, figsize = (14, 4.8))
    table = tables[settings["wrists_deg"][0]]
    for twist, colour in zip([0.0, 30.0, 60.0, 90.0, 117.0], ["black", "tab:blue", "tab:green", "tab:orange", "tab:red"]):
        axes[0].plot(table.index, table[twist], color = colour, label = f"twist +{twist:.0f}")
        axes[0].plot(table.index, table[-twist], color = colour, linestyle = "--")
    axes[0].scatter(100.0 * frames["reach_m"].iloc[::9], frames["rod_from_vertical_deg"].iloc[::9], s = 2, color = "lightgrey", label = "demonstration rows (every 9th)", zorder = 0)
    axes[0].axhline(settings["rod_lean_most_deg"], color = "grey", linestyle = ":")
    axes[0].text(29.0, settings["rod_lean_most_deg"] + 1.0, "the driver refuses above 58 degrees", fontsize = 8)
    axes[0].set_xlabel("reach: tip distance from the base (cm)")
    axes[0].set_ylabel("rod lean from straight down (degrees)")
    axes[0].set_title("Wrist held at 13.5: the lean follows reach and twist (dashed: negative twist)")
    axes[0].legend(fontsize = 8)
    axes[0].grid(alpha = 0.3)

    bins = numpy.arange(-60.0, 62.0, 4.0)
    axes[1].hist(demonstrated.loc[demonstrated["is_on_mark"], "lean_along_deg"], bins = bins, density = True, color = "lightgrey", label = "demonstrations (start mark)")
    all_runs = pandas.concat(run_leans.values(), ignore_index = True)
    axes[1].hist(all_runs["lean_along_deg"], bins = bins, density = True, histtype = "step", color = "black", linewidth = 1.5, label = "the ten robot runs")
    axes[1].axvline(0.0, color = "grey", linewidth = 1)
    axes[1].set_xlabel("rod lean along the push direction (degrees); positive: the tip leads")
    axes[1].set_ylabel("share of pushes (density)")
    axes[1].set_title("How the rod leaned against the push")
    axes[1].legend(fontsize = 8)
    axes[1].grid(alpha = 0.3)
    figure.tight_layout()
    figure.savefig(output_folder / "lean_and_reach.png", dpi = settings["figure_dpi"])
    pyplot.close(figure)
    print(f"\nsaved {output_folder / 'lean_and_reach.png'} and lean_against_push.csv")



if __name__ == "__main__":
    main()
