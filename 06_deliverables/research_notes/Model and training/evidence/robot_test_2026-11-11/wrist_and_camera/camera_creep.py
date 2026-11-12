"""
Has the taped camera moved on the forearm between the demonstrations and the robot test? (key: wrist_and_camera, part 5)

The blue fork at the bottom of every picture is part of the forearm, as the camera is, so it sits at the same pixels
in every picture unless the camera moves on its tape. The rod and the block inside the fork move only with the wrist
and the hand, which the driver held at the start-mark demonstrations' start values (13.5 and -6.1 degrees). So the
bottom strip of a robot-run picture can be compared with the bottom strip of the recording tool's reference picture
exactly as the tool's own camera check does at the start of every episode (record_episodes.camera_shift_px, phase
correlation on the strip "camera_check_region"), and with the shifts the tool logged for the 211 demonstrations.

Measured on the saved pictures. Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe camera_creep.py
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import wrist_geometry
from config import RECORDING_PARAMETERS
from record_episodes import camera_shift_px


CREEP_PARAMETERS = {
    "reference_picture": Path("D:/AI_Project/03_data/raw/camera_check/camera_reference/forearm_cover_tape_bare_camera_x159_y-15mm.jpg"),
    "region": RECORDING_PARAMETERS["camera_check_region"],
    # only pictures whose wrist and hand are this close to the reference pose show the same rod and block in the strip
    "wrist_window_deg": 0.5,
    "hand_window_deg": 0.5,
    "demonstration_pictures_per_episode": 6,
    # an episode whose fork sits further right than this (px at 640) counts as recorded after the camera moved
    "moved_x_px": 8.0,
    "figure_dpi": 120
}



def shifts_of_pictures(
        paths: List[Path],
        reference_bytes: bytes,
        parameters: Dict
) -> pandas.DataFrame:
    """The strip's shift (x to the right, y downwards, in px at 640) of every picture against the reference picture."""

    rows = []
    for path in paths:
        shift_x, shift_y = camera_shift_px(reference_bytes, Path(path).read_bytes(), parameters["region"])
        rows.append({"path": str(path), "shift_x_px": shift_x, "shift_y_px": shift_y, "shift_px": float(numpy.hypot(shift_x, shift_y))})
    return pandas.DataFrame(rows)



def main(parameters: Optional[Dict] = None) -> None:
    """Print the camera's shift per recording day and sheet and per robot run, and save a figure."""

    settings = dict(CREEP_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 200)
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    reference_bytes = Path(settings["reference_picture"]).read_bytes()
    held_wrist = wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]
    held_hand = wrist_geometry.ANALYSIS_PARAMETERS["hand_held_deg"]

    episodes = pandas.read_csv(wrist_geometry.ANALYSIS_PARAMETERS["episodes_file"])
    print("=== the recording tool's own camera check at the start of each demonstration (episodes.csv, camera_shift_px) ===")
    logged = episodes.groupby(["session_date", "sheet"])["camera_shift_px"].agg(["size", "median", "min", "max"]).round(1)
    print(logged.to_string())

    frames = wrist_geometry.load_demonstration_frames()
    frames = frames.merge(episodes[["episode_id", "session_date"]], on = "episode_id", how = "left")
    is_reference_pose = ((frames["joint_5_deg"] - held_wrist).abs() <= settings["wrist_window_deg"]) & ((frames["joint_6_deg"] - held_hand).abs() <= settings["hand_window_deg"])
    usable = frames[is_reference_pose]
    chosen_rows = []
    for _, group in usable.groupby("episode_id"):
        positions = numpy.unique(numpy.linspace(0, len(group) - 1, min(len(group), settings["demonstration_pictures_per_episode"])).astype(int))
        chosen_rows.extend(group.index[positions])
    chosen = usable.loc[chosen_rows]
    raw_folder = Path(wrist_geometry.ANALYSIS_PARAMETERS["raw_folder"])
    measured = shifts_of_pictures([raw_folder / path for path in chosen["frame_path"]], reference_bytes, settings)
    measured["episode_id"] = chosen["episode_id"].to_numpy()
    measured["session_date"] = chosen["session_date"].to_numpy()
    measured["sheet"] = chosen["sheet"].to_numpy()
    per_episode = measured.groupby(["session_date", "sheet", "episode_id"])[["shift_x_px", "shift_y_px", "shift_px"]].median().reset_index()
    spread = measured.groupby("episode_id")[["shift_x_px", "shift_y_px"]].agg(lambda values: values.max() - values.min())
    print(f"\n=== measured here the same way on {len(measured)} demonstration pictures with the wrist and hand at the reference pose "
          f"({per_episode['episode_id'].nunique()} episodes, up to {settings['demonstration_pictures_per_episode']} pictures each) ===")
    print(f"scatter of the measure inside one episode (largest minus smallest of its pictures): x median {spread['shift_x_px'].median():.1f} px, "
          f"y median {spread['shift_y_px'].median():.1f} px; 90th percentile x {spread['shift_x_px'].quantile(0.9):.1f}, y {spread['shift_y_px'].quantile(0.9):.1f}")
    by_day = per_episode.groupby(["session_date", "sheet"]).agg(episodes = ("episode_id", "size"), shift_x_median = ("shift_x_px", "median"), shift_y_median = ("shift_y_px", "median"),
                                                               shift_median = ("shift_px", "median"), shift_most = ("shift_px", "max")).round(1)
    print(by_day.to_string())

    timed = per_episode.merge(episodes[["episode_id", "started", "plan_number"]], on = "episode_id").sort_values("started")
    is_moved = timed["shift_x_px"] > settings["moved_x_px"]
    first_moved = timed[is_moved].iloc[0]
    last_before = timed[timed["started"] < first_moved["started"]].iloc[-1]
    back_again = int((~is_moved & (timed["started"] > first_moved["started"])).sum())
    episodes_after = episodes[episodes["started"] >= first_moved["started"]]["episode_id"]
    rows_after = int(frames["episode_id"].isin(episodes_after).sum())
    print(f"\nthe fork's sideways shift steps once: last episode at the old place {last_before['started']} (plan {last_before['plan_number']}, x {last_before['shift_x_px']:+.1f} px), "
          f"first at the new place {first_moved['started']} (plan {first_moved['plan_number']}, x {first_moved['shift_x_px']:+.1f} px); episodes back at the old place afterwards: {back_again}")
    print(f"before the step: x median {timed.loc[~is_moved, 'shift_x_px'].median():+.1f} px, y median {timed.loc[~is_moved, 'shift_y_px'].median():+.1f}; "
          f"after it: x {timed.loc[is_moved, 'shift_x_px'].median():+.1f}, y {timed.loc[is_moved, 'shift_y_px'].median():+.1f}")
    print(f"demonstrations recorded after the step (camera where it is now): {len(episodes_after)} of {len(episodes)} episodes, {rows_after} of {len(frames)} rows "
          f"({100.0 * rows_after / len(frames):.1f} %); by sheet: "
          + ", ".join(f"{sheet[-2:]}: {int(frames.loc[frames['sheet'] == sheet, 'episode_id'].isin(episodes_after).sum())} of {int((frames['sheet'] == sheet).sum())}" for sheet in sorted(frames["sheet"].unique())))

    print("\n=== the robot runs of 11.11.26 (wrist and hand held at the reference pose in every picture) ===")
    run_rows = []
    run_tables = {}
    for run in wrist_geometry.ANALYSIS_PARAMETERS["robot_runs"]:
        folder = Path(wrist_geometry.ANALYSIS_PARAMETERS["model_runs_folder"]) / run["folder"] / "frames"
        table = shifts_of_pictures(sorted(folder.glob("*.jpg")), reference_bytes, settings)
        run_tables[run["number"]] = table
        run_rows.append({"run": run["number"], "sheet": run["sheet"][-2:], "pictures": len(table), "shift_x_median": table["shift_x_px"].median(), "shift_y_median": table["shift_y_px"].median(),
                         "shift_median": table["shift_px"].median(), "shift_x_p10": table["shift_x_px"].quantile(0.1), "shift_x_p90": table["shift_x_px"].quantile(0.9),
                         "shift_y_p10": table["shift_y_px"].quantile(0.1), "shift_y_p90": table["shift_y_px"].quantile(0.9)})
    run_summary = pandas.DataFrame(run_rows)
    print(run_summary.round(1).to_string(index = False))
    all_runs = pandas.concat(run_tables.values(), ignore_index = True)
    last_day = per_episode[per_episode["session_date"] == per_episode["session_date"].max()]
    print(f"\nall {len(all_runs)} robot-run pictures: median shift x {all_runs['shift_x_px'].median():+.1f}, y {all_runs['shift_y_px'].median():+.1f} px against the reference; "
          f"the last recording day ({per_episode['session_date'].max()}): x {last_day['shift_x_px'].median():+.1f}, y {last_day['shift_y_px'].median():+.1f}; "
          f"all demonstrations: x {per_episode['shift_x_px'].median():+.1f}, y {per_episode['shift_y_px'].median():+.1f}")
    print(f"robot test minus the last recording day: x {all_runs['shift_x_px'].median() - last_day['shift_x_px'].median():+.1f}, "
          f"y {all_runs['shift_y_px'].median() - last_day['shift_y_px'].median():+.1f} px at 640 "
          f"({(all_runs['shift_x_px'].median() - last_day['shift_x_px'].median()) / 4.0:+.1f}, {(all_runs['shift_y_px'].median() - last_day['shift_y_px'].median()) / 4.0:+.1f} px at the model's 160)")
    print(f"robot test minus all demonstrations (median): x {all_runs['shift_x_px'].median() - per_episode['shift_x_px'].median():+.1f}, "
          f"y {all_runs['shift_y_px'].median() - per_episode['shift_y_px'].median():+.1f} px at 640")
    per_episode.to_csv(output_folder / "camera_creep_demonstrations.csv", index = False)
    run_summary.to_csv(output_folder / "camera_creep_runs.csv", index = False)

    figure, axis = pyplot.subplots(1, 1, figsize = (8.5, 6.5))
    colours = {"2026-11-06": "tab:blue", "2026-11-07": "tab:cyan", "2026-11-08": "tab:green", "2026-11-09": "tab:orange"}
    for day, group in per_episode.groupby("session_date"):
        axis.scatter(group["shift_x_px"], group["shift_y_px"], s = 18, color = colours.get(day, "grey"), label = f"demonstrations of {day} (one dot per episode)")
    axis.scatter(run_summary["shift_x_median"], run_summary["shift_y_median"], s = 90, marker = "*", color = "black", label = "robot runs of 2026-11-11 (median of each run)")
    axis.add_patch(pyplot.Rectangle((per_episode["shift_x_px"].median() - 16.0, per_episode["shift_y_px"].median() - 16.0), 32.0, 32.0, fill = False, linestyle = ":", color = "grey"))
    axis.text(per_episode["shift_x_px"].median() - 15.5, per_episode["shift_y_px"].median() - 14.0, "training shift augmentation\naround the demonstrations' median\n(4 px at 160)", fontsize = 8, color = "grey")
    axis.set_xlabel("shift of the fork to the right (px at 640)")
    axis.set_ylabel("shift of the fork downwards (px at 640)")
    axis.invert_yaxis()
    axis.set_aspect("equal")
    axis.set_title("Where the fork sits in the picture against the reference picture\n(a move of the fork is a move of the camera on its tape)")
    axis.legend(fontsize = 8)
    axis.grid(alpha = 0.3)
    figure.tight_layout()
    figure.savefig(output_folder / "camera_creep.png", dpi = settings["figure_dpi"])
    pyplot.close(figure)
    print(f"\nsaved {output_folder / 'camera_creep.png'}, camera_creep_demonstrations.csv and camera_creep_runs.csv")



if __name__ == "__main__":
    main()
