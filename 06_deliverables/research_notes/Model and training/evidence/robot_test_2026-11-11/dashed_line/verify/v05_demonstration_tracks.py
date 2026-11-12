"""
Verifier check 5: the demonstrations of one sheet placed on the line with my own pipeline.

For every episode of the sheet: my top view from every second picture, the course-photo centre line laid onto it, the
puck in every picture, its place along and beside the line. The end an episode started from is read from the puck's
travel along the placed line (the line's first point is the course photo's "first" end), without the index's
start_end column and without the audit's corrected ends.

Run:  D:/AI_Project/.venv/Scripts/python.exe v05_demonstration_tracks.py 06 > v05_demonstration_tracks_06.out.txt
      D:/AI_Project/.venv/Scripts/python.exe v05_demonstration_tracks.py 05 > v05_demonstration_tracks_05.out.txt
Writes v05_demonstration_tracks_<sheet>.csv. Two to four minutes per sheet. Nothing connects to the robot.
"""

import sys
import time

import cv2
import numpy
import pandas

from verify_common import (VERIFY_FOLDER, VERIFY_PARAMETERS, along_and_beside, colour_masks, crossing_indices,
                           pixels_on_table, puck_pixel, register_line, sheet_line, top_view)

cv2.setNumThreads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 300)
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]


def main() -> None:

    sheet = sys.argv[1]
    started = time.time()
    frames = pandas.read_csv(VERIFY_PARAMETERS["index_folder"] / "frames.csv", low_memory = False)
    frames = frames[frames["sheet"] == f"path_{sheet}"].sort_values(["episode_id", "row_in_episode"])
    line_sheet = sheet_line(sheet)
    rows, summary = [], []
    for episode_id, part in frames.groupby("episode_id", sort = False):
        pictures = [str(VERIFY_PARAMETERS["raw_folder"] / value) for value in part["frame_path"]]
        joints = part[JOINT_NAMES].to_numpy()
        chosen = numpy.arange(0, len(part), 2)
        view = top_view([pictures[position] for position in chosen], joints[chosen])
        placed = register_line(line_sheet, view)
        puck_xy = numpy.full((len(part), 2), numpy.nan)
        for position, picture_path in enumerate(pictures):
            found = puck_pixel(colour_masks(cv2.imread(picture_path))["puck"])
            if found is not None:
                puck_xy[position] = pixels_on_table(numpy.array([[found[0], found[1]]]), joints[position], VERIFY_PARAMETERS["puck_point_height_m"])[0]
        # the travel direction: place the puck's path on the line both ways and keep the way with the smaller mean distance
        options = {}
        for name, line in (("first_end", placed["line_m"]), ("last_end", placed["line_m"][::-1].copy())):
            # the first puck position may lie anywhere on the line (starts near the end, at the crossing): search it all
            valid = numpy.flatnonzero(numpy.isfinite(puck_xy[:, 0]))
            if len(valid) == 0:
                continue
            first_point = puck_xy[valid[0]]
            distances = numpy.hypot(*(line - first_point).T)
            candidates = [int(numpy.argmin(distances))]
            # the crossing gives a second candidate place for the first point
            order = numpy.argsort(distances)
            for other in order[:200]:
                if abs(int(other) - candidates[0]) > 50 and distances[other] < distances[candidates[0]] + 0.01:
                    candidates.append(int(other))
                    break
            best_option = None
            for start_index in candidates:
                index, shifted_beside = along_and_beside(line, puck_xy, back_most = 30, ahead_most = 60, first_index = start_index)
                mean_distance = float(numpy.nanmean(numpy.abs(shifted_beside)))
                progress = float(index[valid[-1]] - index[valid[0]])
                if best_option is None or mean_distance < best_option["mean_distance"]:
                    best_option = {"index": index, "beside": shifted_beside, "mean_distance": mean_distance, "progress": progress, "line": line}
            options[name] = best_option
        if not options:
            continue
        travel = min(options, key = lambda name: options[name]["mean_distance"] - 0.00001 * options[name]["progress"])
        result = options[travel]
        other = options["last_end" if travel == "first_end" else "first_end"]
        crossing = crossing_indices(result["line"])
        table = pandas.DataFrame({
            "episode_id": episode_id, "plan": part["plan_number"].to_numpy(), "start_kind": part["start_kind"].to_numpy(),
            "index_start_end": part["start_end"].to_numpy(), "travel_from": travel,
            "row": part["row_in_episode"].to_numpy(), "frame_index": part["frame_index"].to_numpy(), "action": part["action"].to_numpy(),
            "along_cm": 0.1 * (result["index"] - crossing[0]), "from_start_cm": 0.1 * result["index"],
            "beside_cm": 100.0 * result["beside"],
            "puck_x_cm": 100.0 * puck_xy[:, 0], "puck_y_cm": 100.0 * puck_xy[:, 1],
            "joint_4_deg": numpy.degrees(joints[:, 3]), "joint_5_deg": numpy.degrees(joints[:, 4]), "joint_6_deg": numpy.degrees(joints[:, 5]),
            "move_angle_deg": part["move_angle_deg"].to_numpy(), "move_length_cm": 100.0 * part["move_length_m"].to_numpy(),
            "forearm_change_deg": part["forearm_change_deg"].to_numpy(), "lean_change_deg": part["lean_change_deg"].to_numpy(),
            "tip_x_cm": 100.0 * part["tip_x_m"].to_numpy(), "tip_y_cm": 100.0 * part["tip_y_m"].to_numpy(),
            "score": part["score"].to_numpy(),
            "loop_length_cm": 0.1 * (crossing[1] - crossing[0]), "line_length_cm": 0.1 * (len(result["line"]) - 1)
        })
        table.loc[result["index"] < 0, ["along_cm", "from_start_cm"]] = numpy.nan
        rows.append(table)
        summary.append({
            "plan": part["plan_number"].iloc[0], "kind": part["start_kind"].iloc[0], "index_end": part["start_end"].iloc[0],
            "pictures": len(part), "angle_deg": numpy.degrees(placed["placement"][0]) % 360.0,
            "x_cm": 100.0 * placed["placement"][1], "y_cm": 100.0 * placed["placement"][2], "mirrored": placed["is_mirrored"],
            "ink_to_line_mm": placed["ink_to_line_mm"], "line_seen": placed["line_seen_share"], "line_on_ink": placed["line_on_ink_share"],
            "travel_from": travel, "mean_beside_cm": 100.0 * result["mean_distance"], "other_way_cm": 100.0 * other["mean_distance"],
            "progress_cm": 0.1 * result["progress"], "start_from_cm": float(table["from_start_cm"].dropna().iloc[0]) if table["from_start_cm"].notna().any() else numpy.nan,
            "puck_seen": int(numpy.isfinite(puck_xy[:, 0]).sum())
        })
    tracks = pandas.concat(rows, ignore_index = True)
    tracks.round(3).to_csv(VERIFY_FOLDER / f"v05_demonstration_tracks_{sheet}.csv", index = False)
    summary = pandas.DataFrame(summary)
    print(f"sheet {sheet}: {len(summary)} episodes, {len(tracks)} pictures, {time.time() - started:.0f} s")
    print("line placed per episode (angle and place of the line's middle on the table) and the end the puck travelled from")
    print("('first_end' and 'last_end' are the two ends of the course-photo centre line):")
    print(summary.round(2).to_string(index = False))
    print("\nepisodes by the index's start_end and by the travel I measured:")
    print(summary.groupby(["index_end", "travel_from"]).size().to_string())



if __name__ == "__main__":
    main()
