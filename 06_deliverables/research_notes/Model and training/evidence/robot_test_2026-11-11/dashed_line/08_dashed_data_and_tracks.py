"""
Step 8 of the dashed-line analysis: how much dashed data there is, how the six sheets differ in what the
demonstrations do, and whether the printable practice tracks match the course sheets.

    1. Sheet 06 in the training data: episodes, pictures, start ends, the night it was recorded, and the sheet
       placements found in step 2 (the angle and the place of the line on the table).
    2. Per sheet, from the cleaned training rows: how many demonstrated steps go sideways or backwards in the
       picture, how far the forearm is twisted, how often the wrist is away from the 13.5 degrees the driver holds.
    3. The practice tracks' settings (config.TEST_TRACK_PARAMETERS) against the dashes measured in step 1 and the
       tightest bends of the six course sheets (course-photo centre lines, same ruler as step 1).
    4. Where the walk round the puck starts in the demonstrations, against the two robot runs.

Run: D:/AI_Project/.venv/Scripts/python.exe 08_dashed_data_and_tracks.py > 08_dashed_data_and_tracks.out.txt
"""

import json

import numpy
import pandas
from scipy.signal import savgol_filter

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER  # first: it puts 02_code on the path
from config import MODEL_MODE_PARAMETERS, TEST_TRACK_PARAMETERS
from line_geometry import line_description, resampled
from training_targets import load_training_frames

DATA_PARAMETERS = {
    "course_measurements_path": EVIDENCE_FOLDER.parents[3] / "Recording tool" / "2026-11-06_sheet_measurements.json",
    "whole_line_kinds": ["STD", "OL1", "OR1", "OL1.5", "OR1.5", "TL10", "TR10", "TL15", "TR15", "X1"],
    "placed_ink_share": 0.75,
    "walk_zones_cm": {"06": ("A", 14.0, 20.0, "run_8"), "05": ("B", 15.0, 19.0, "run_7")}
}



def main() -> None:

    settings = DATA_PARAMETERS
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 200)
    episodes = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "episodes.csv")
    frames = load_training_frames({"is_audit_cleaning": True})
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv")

    print("1. Sheet 06 (dashed) in the training data")
    six = episodes[episodes["sheet"] == "path_06"]
    cleaned = frames[frames["sheet"] == "path_06"]
    print(f"   episodes {len(six)} (scores {six['score'].value_counts().sort_index().to_dict()}), recorded {six['started'].min()} to {six['finished'].max()} by "
          f"{', '.join(six['operator'].unique())}, lighting {', '.join(six['lighting'].unique())}")
    print(f"   pictures: {int((pandas.read_csv(ANALYSIS_PARAMETERS['processed_folder'] / 'frames.csv', usecols = ['sheet'])['sheet'] == 'path_06').sum())} recorded, "
          f"{len(cleaned)} after cleaning = {100 * len(cleaned) / len(frames):.1f} % of the {len(frames)} cleaned pictures of all sheets")
    print(cleaned.groupby("start_end").agg(episodes = ("episode_id", "nunique"), pictures = ("frame_index", "size")).to_string())
    print("   cleaned pictures per sheet:")
    print(frames.groupby("sheet").agg(episodes = ("episode_id", "nunique"), pictures = ("frame_index", "size")).to_string())
    for sheet in (6, 5):
        placed = registrations[(registrations["sheet"] == sheet) & (registrations["start_kind"] != "RUN")].copy()
        placed["angle"] = ((placed["angle_deg"] + 180.0) % 360.0 - 180.0).round(0)
        placed["placement"] = [f"{angle:+.0f} deg, line's middle at ({x:.0f}, {y:.0f}) cm" if share >= settings["placed_ink_share"] else "not found"
                               for angle, x, y, share in zip(placed["angle"], placed["shift_x_cm"], placed["shift_y_cm"], placed["ink_share_on_seen_ink_points"])]
        print(f"   sheet {sheet:02d}: where the sheet lay (angle against the sheet's place in the robot run of 11.11.26, rounded to 1 degree and 1 cm)")
        print(placed.groupby(["start_end", "placement"]).agg(episodes = ("plan", "size"), plans = ("plan", lambda values: ", ".join(values))).to_string())
        run = registrations[(registrations["sheet"] == sheet) & (registrations["start_kind"] == "RUN")].iloc[0]
        print(f"   the robot run on sheet {sheet:02d}: {((run['angle_deg'] + 180) % 360 - 180):+.0f} deg, line's middle at ({run['shift_x_cm']:.0f}, {run['shift_y_cm']:.0f}) cm")

    print("\n2. What the demonstrations do, per sheet (cleaned rows without the stop pictures)")
    rows = frames[frames["action"] != "stop"].copy()
    rows["angle"] = numpy.degrees(numpy.arctan2(rows["tip_left_m"].fillna(0.0), rows["tip_forward_m"].fillna(0.0))).abs()
    rows["length_cm"] = 100.0 * numpy.hypot(rows["tip_left_m"].fillna(0.0), rows["tip_forward_m"].fillna(0.0))
    rows["forearm"] = numpy.degrees(rows["joint_4"]).abs()
    rows["wrist"] = numpy.degrees(rows["joint_5"])
    moving = rows[rows["length_cm"] >= 0.3]
    table = pandas.DataFrame({
        "pictures": rows.groupby("sheet").size(),
        "steps_forward_within_30_deg": moving.groupby("sheet")["angle"].apply(lambda values: float((values <= 30).mean())),
        "steps_sideways_60_to_120_deg": moving.groupby("sheet")["angle"].apply(lambda values: float(((values > 60) & (values <= 120)).mean())),
        "steps_backward_beyond_120_deg": moving.groupby("sheet")["angle"].apply(lambda values: float((values > 120).mean())),
        "forearm_twisted_beyond_60_deg": rows.groupby("sheet")["forearm"].apply(lambda values: float((values > 60).mean())),
        "wrist_over_5_deg_from_held": rows.groupby("sheet")["wrist"].apply(lambda values: float(((values - MODEL_MODE_PARAMETERS["wrist_held_deg"]).abs() > 5).mean())),
        "wrist_median_deg": rows.groupby("sheet")["wrist"].median(), "wrist_90th_percentile_deg": rows.groupby("sheet")["wrist"].quantile(0.9)})
    print(table.round(3).to_string())

    print("\n3. The practice tracks against the course sheets")
    summary = json.loads((EVIDENCE_FOLDER / "01_sheet_geometry_summary.json").read_text(encoding = "utf-8"))
    print(f"   dashes: practice tracks {TEST_TRACK_PARAMETERS['dash_mm']:.0f} mm with gaps of {TEST_TRACK_PARAMETERS['gap_mm']:.0f} mm; sheet 06 measured "
          f"{10 * summary['06']['dash_median_cm']:.0f} mm with gaps of {10 * summary['06']['gap_median_cm']:.0f} mm (step 1, median of {summary['06']['dashes']} dashes)")
    print(f"   tightest bend allowed in a practice track: {TEST_TRACK_PARAMETERS['smallest_radius_mm'] / 10:.1f} cm radius; the course sheets "
          "(course-photo centre lines, smoothed over 1.5 cm, bend measured over 2 cm, so nothing under about 1 cm can show):")
    course = json.loads(settings["course_measurements_path"].read_text(encoding = "utf-8"))
    for sheet in ("01", "02", "03", "04", "05", "06"):
        line = resampled(numpy.array(course[sheet]["line_xy_cm"]) / 100.0, 0.001)
        line = numpy.column_stack([savgol_filter(line[:, 0], 15, 3), savgol_filter(line[:, 1], 15, 3)])
        radius = 100.0 * line_description(line)["radius_m"][10:-10]
        limit = TEST_TRACK_PARAMETERS["smallest_radius_mm"] / 10.0
        print(f"      sheet {sheet}: tightest bend {radius.min():.1f} cm radius; {0.1 * (radius < limit).sum():.1f} cm of its {0.1 * len(line):.1f} cm are tighter than {limit:.1f} cm, "
              f"{0.1 * (radius < 2.0).sum():.1f} cm tighter than 2.0 cm")
    for sheet in ("05", "06"):
        print(f"      sheet {sheet} from the robot run's top view (step 1): tightest bend {summary[sheet]['whole_line_smallest_radius_cm']:.1f} cm radius")

    print("\n4. The state in which the walk round the puck starts (the first backward step of 0.3 cm or more near the busy stretch of the loop)")
    tracks = pandas.read_csv(EVIDENCE_FOLDER / "tracks.csv")
    placed_recordings = set(registrations.loc[registrations["ink_share_on_seen_ink_points"] >= settings["placed_ink_share"], "recording"])
    for sheet, (start_end, first_cm, last_cm, run_name) in settings["walk_zones_cm"].items():
        demonstrations = tracks[(tracks["sheet"] == int(sheet)) & (tracks["start_end"] == start_end) & tracks["start_kind"].isin(settings["whole_line_kinds"])
                                & (tracks["score"] > 0) & tracks["recording"].isin(placed_recordings)]
        starts = []
        for plan, part in demonstrations.groupby("plan"):
            backward = part[(part["puck_along_cm"] >= first_cm - 3.0) & (part["puck_along_cm"] < last_cm) & (part["forward_cm"] < -0.3)].sort_values("row")
            if len(backward):
                starts.append(backward.iloc[0][["plan", "row", "puck_along_cm", "puck_beside_cm", "joint_4_deg", "joint_5_deg"]])
        starts = pandas.DataFrame(starts)
        print(f"   sheet {sheet} from end {start_end}, {len(starts)} demonstrations: puck {starts['puck_along_cm'].min():.1f} to {starts['puck_along_cm'].max():.1f} cm after the crossing, "
              f"beside the line {starts['puck_beside_cm'].min():.1f} to {starts['puck_beside_cm'].max():.1f} cm (median {starts['puck_beside_cm'].median():.1f}), "
              f"forearm {starts['joint_4_deg'].min():.0f} to {starts['joint_4_deg'].max():.0f} deg (median {starts['joint_4_deg'].median():.0f}), "
              f"wrist {starts['joint_5_deg'].min():.0f} to {starts['joint_5_deg'].max():.0f} deg (median {starts['joint_5_deg'].median():.0f}; "
              f"{int(((starts['joint_5_deg'] - MODEL_MODE_PARAMETERS['wrist_held_deg']).abs() > 5).sum())} of {len(starts)} more than 5 deg from the held 13.5)")
        the_run = tracks[tracks["recording"] == run_name]
        in_zone = the_run[(the_run["puck_along_cm"] >= first_cm) & (the_run["puck_along_cm"] < last_cm)]
        backward = the_run[the_run["forward_cm"] < -0.3]
        print(f"      {run_name} between {first_cm:.0f} and {last_cm:.0f} cm: steps {int(in_zone['row'].min())} to {int(in_zone['row'].max())}, puck beside the line "
              f"{in_zone['puck_beside_cm'].min():.1f} to {in_zone['puck_beside_cm'].max():.1f} cm, forearm {in_zone['joint_4_deg'].min():.0f} to {in_zone['joint_4_deg'].max():.0f} deg, "
              f"wrist {in_zone['joint_5_deg'].median():.1f} deg; " + (f"first backward step asked at step {int(backward['row'].iloc[0])} ({backward['puck_along_cm'].iloc[0]:.1f} cm, puck "
              f"{backward['puck_beside_cm'].iloc[0]:.1f} cm beside the line, forearm {backward['joint_4_deg'].iloc[0]:.0f} deg)" if len(backward) else "no backward step asked in the whole run"))



if __name__ == "__main__":
    main()
