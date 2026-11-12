"""
Step 2 of the dashed-line analysis: where the puck and the magnet tip were on the line, picture by picture.

For robot runs 7 and 8 and for every demonstration of sheets 05 and 06 (55 episodes): the line is drawn from above
from the recording's own pictures, the sheet's reference centre line (step 1) is laid onto it, the red puck is found
in every picture and put on the table (puck_tracking.PuckTracker), and the puck and the tip get their place along the
line (cm from the first pass of the crossing, in the direction of travel) and their distance beside it. For sheet 06
the ink just ahead of the puck is also measured in the picture itself.

Writes tracks.csv (one row per picture), registrations.csv (one row per recording, with how well the line fitted) and
02_registration_check.png. No robot, no training; about four minutes.

Run: D:/AI_Project/.venv/Scripts/python.exe 02_tracks.py > 02_tracks.out.txt
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy
import pandas

from dashed_common import (ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, JOINT_NAMES, ink_and_paper_masks, pixels_from_table_points, read_run,
                           top_view_of_line)
from line_geometry import ink_share_of, line_description, place_along_line, registered_line, tangents_and_normals
from puck_tracking import PuckTracker
from side_view_test import magnet_tip_and_lean

cv2.setNumThreads(2)

TRACK_PARAMETERS = {
    # the top view used to place the line: cell size, every how many pictures, and only pictures with the forearm
    # twisted less than this (the far-twisted pictures smear the map by about 4 mm); all pictures when fewer than
    # "least_pictures" remain
    "map_cell_m": 0.002,
    "map_picture_stride": 2,
    "map_twist_most_deg": 60.0,
    "least_pictures": 8,
    # an episode whose pictures showed less than this share of the line is placed as a neighbour episode was, when
    # that fits (moved by at most 1.5 cm and 5 degrees in the refinement)
    "short_view_share": 0.75,
    "short_view_most_move_cm": 1.5,
    "short_view_most_turn_deg": 5.0,
    "short_view_search_handicap": 0.1,
    # the puck's centre is 1.2 cm from its front edge; the ink is measured on the centre line from the front edge to
    # "ink_ahead_m" further, a line point counting as ink when any pixel within "ink_tolerance_px" of it is ink
    "puck_radius_m": 0.012,
    "ink_ahead_m": 0.02,
    "ink_tolerance_px": 7,
    "corrected_ends_path": EVIDENCE_FOLDER.parents[1] / "corrected_ends.csv",
    # which end each reference line starts from (the end the robot run started at)
    "reference_start_end": {"05": "B", "06": "A"}
}



def recordings_table() -> list:
    """Return every recording to track: the two robot runs and the demonstrations of sheets 05 and 06."""

    frames = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "frames.csv", low_memory = False)
    frames = frames[frames["sheet"].isin(["path_05", "path_06"])].sort_values(["episode_id", "row_in_episode"])
    ends = pandas.read_csv(TRACK_PARAMETERS["corrected_ends_path"]).set_index("episode_id")["corrected_end"]
    recordings = []
    for name, key, sheet in (("run_7", "run_7_folder", "05"), ("run_8", "run_8_folder", "06")):
        steps = read_run(ANALYSIS_PARAMETERS[key])
        recordings.append({
            "recording": name, "sheet": sheet, "plan": "", "start_kind": "RUN", "start_end": TRACK_PARAMETERS["reference_start_end"][sheet],
            "score": numpy.nan, "pictures": list(steps["frame_file"]), "joints": steps[JOINT_NAMES].to_numpy(),
            "row": steps["step"].to_numpy(), "frame_index": numpy.full(len(steps), -1),
            "heading_deg": steps["picture_heading_deg"].to_numpy(),
            "forward_cm": steps["model_tip_forward_cm"].to_numpy(), "left_cm": steps["model_tip_left_cm"].to_numpy(),
            "twist_deg": steps["model_twist_deg"].to_numpy(), "action": steps["action"].to_numpy(),
            "stop_probability": steps["model_stop_probability"].to_numpy()
        })
    for episode_id, part in frames.groupby("episode_id", sort = False):
        recordings.append({
            "recording": episode_id, "sheet": part["sheet"].iloc[0][-2:], "plan": part["plan_number"].iloc[0],
            "start_kind": part["start_kind"].iloc[0], "start_end": ends[episode_id], "score": part["score"].iloc[0],
            "pictures": [str(ANALYSIS_PARAMETERS["raw_folder"] / value) for value in part["frame_path"]],
            "joints": part[JOINT_NAMES].to_numpy(), "row": part["row_in_episode"].to_numpy(), "frame_index": part["frame_index"].to_numpy(),
            "heading_deg": part["picture_heading_deg"].to_numpy(),
            "forward_cm": 100.0 * part["tip_forward_m"].fillna(0.0).to_numpy(), "left_cm": 100.0 * part["tip_left_m"].fillna(0.0).to_numpy(),
            "twist_deg": part["forearm_change_deg"].fillna(0.0).to_numpy(), "action": part["action"].to_numpy(),
            "stop_probability": numpy.full(len(part), numpy.nan)
        })
    return recordings



def main() -> None:

    settings = TRACK_PARAMETERS
    references = {sheet: pandas.read_csv(EVIDENCE_FOLDER / f"reference_line_{sheet}.csv") for sheet in ("05", "06")}
    tracker = PuckTracker()
    recordings = recordings_table()
    track_rows, registration_rows = [], []
    figure, axes = pyplot.subplots(6, 10, figsize = (40, 24))
    axes = axes.ravel()
    started = time.time()
    for number, recording in enumerate(recordings):
        reference = references[recording["sheet"]]
        reference_points = reference[["x_cm", "y_cm"]].to_numpy() / 100.0
        is_ink_point = reference["is_ink"].to_numpy() > 0
        joints = recording["joints"]
        twist_deg = numpy.degrees(joints[:, 3])
        chosen = numpy.flatnonzero(numpy.abs(twist_deg) <= settings["map_twist_most_deg"])[::settings["map_picture_stride"]]
        if len(chosen) < settings["least_pictures"]:
            chosen = numpy.arange(len(joints))
        view = top_view_of_line([recording["pictures"][position] for position in chosen], joints[chosen], {"map_cell_m": settings["map_cell_m"]})
        placed = registered_line(reference_points, is_ink_point, view)
        placed["method"] = "searched"
        if placed["seen_share_of_line"] < settings["short_view_share"]:
            # a short episode shows only a few dashes, which fit many places: the sheet is then assumed to lie as in one
            # of the well-fitted recordings of the same sheet, and the one whose ink fits best where it was seen is taken
            candidates = [placed]
            for earlier in registration_rows:
                if earlier["sheet"] == recording["sheet"] and earlier["method"] == "searched" and earlier["seen_share_of_line"] >= settings["short_view_share"]:
                    candidate = registered_line(reference_points, is_ink_point, view, parameters = {"coarse_blur_m": 0.003},
                                                first_guess = (earlier["angle_deg"], earlier["shift_x_cm"] / 100.0, earlier["shift_y_cm"] / 100.0))
                    moved_cm = 100.0 * numpy.hypot(candidate["shift_m"][0] - earlier["shift_x_cm"] / 100.0, candidate["shift_m"][1] - earlier["shift_y_cm"] / 100.0)
                    turned_deg = abs((candidate["angle_deg"] - earlier["angle_deg"] + 180.0) % 360.0 - 180.0)
                    if moved_cm <= settings["short_view_most_move_cm"] and turned_deg <= settings["short_view_most_turn_deg"]:
                        candidate["method"] = f"as {earlier['plan'] or earlier['recording']}"
                        candidates.append(candidate)
            placed = max(candidates, key = lambda candidate: candidate["ink_share_on_seen_ink_points"] - 0.5 * numpy.nan_to_num(candidate["ink_share_on_seen_gap_points"])
                         - (0.0 if candidate["method"] != "searched" else settings["short_view_search_handicap"]))
        line = placed["line_m"]
        is_travel_reversed = recording["start_end"] != settings["reference_start_end"][recording["sheet"]]
        in_loop = numpy.flatnonzero(reference["is_in_loop"].to_numpy() > 0)
        first_pass, second_pass = int(in_loop[0]), int(in_loop[-1])
        if is_travel_reversed:
            line = line[::-1].copy()
            is_ink_point = is_ink_point[::-1].copy()
            first_pass, second_pass = len(line) - 1 - second_pass, len(line) - 1 - first_pass
        described = line_description(line)
        tangent, _ = tangents_and_normals(line)

        puck_xy = numpy.full((len(joints), 2), numpy.nan)
        puck_pixel = numpy.full((len(joints), 2), numpy.nan)
        tip_xy = numpy.zeros((len(joints), 2))
        pictures = []
        for position, (picture_path, joint_row) in enumerate(zip(recording["pictures"], joints)):
            picture = cv2.imread(picture_path)
            pictures.append(picture)
            tip_xy[position] = magnet_tip_and_lean(list(joint_row))[0][:2]
            seen = tracker.see_puck(picture, list(joint_row))
            if seen is not None and seen["table_xy"] is not None:
                puck_xy[position] = seen["table_xy"]
                puck_pixel[position] = seen["pixel"]
        puck_position, puck_beside = place_along_line(line, puck_xy)
        tip_position = numpy.full(len(joints), numpy.nan)
        tip_beside = numpy.full(len(joints), numpy.nan)
        for position in range(len(joints)):
            if numpy.isfinite(puck_position[position]):
                found, beside = place_along_line(line, tip_xy[position:position + 1], most_jump_m = 0.05, start_position = int(puck_position[position]))
                tip_position[position], tip_beside[position] = found[0], beside[0]

        # the ink just ahead of the puck, measured in the picture
        ink_ahead = numpy.full(len(joints), numpy.nan)
        ink_ahead_design = numpy.full(len(joints), numpy.nan)
        is_over_ink_design = numpy.full(len(joints), numpy.nan)
        kernel = numpy.ones((2 * settings["ink_tolerance_px"] + 1, 2 * settings["ink_tolerance_px"] + 1), numpy.uint8)
        for position in range(len(joints)):
            if not numpy.isfinite(puck_position[position]):
                continue
            here = int(puck_position[position])
            first = here + int(round(settings["puck_radius_m"] / 0.001))
            last = first + int(round(settings["ink_ahead_m"] / 0.001))
            is_over_ink_design[position] = float(is_ink_point[here])
            if first >= len(line):
                continue
            ahead = numpy.arange(first, min(last, len(line)))
            ink_ahead_design[position] = float(is_ink_point[ahead].mean())
            is_ink, is_paper, is_puck = ink_and_paper_masks(pictures[position])
            wide_ink = cv2.dilate(is_ink.astype(numpy.uint8), kernel) > 0
            pixels = pixels_from_table_points(line[ahead], list(joints[position]))
            with numpy.errstate(invalid = "ignore"):
                is_inside = numpy.isfinite(pixels).all(axis = 1) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= 639) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= 479)
            columns = numpy.clip(numpy.round(numpy.nan_to_num(pixels[:, 0])).astype(int), 0, 639)
            rows = numpy.clip(numpy.round(numpy.nan_to_num(pixels[:, 1])).astype(int), 0, 479)
            is_free = is_inside & ~is_puck[rows, columns]
            if is_free.sum() >= 5:
                ink_ahead[position] = float(wide_ink[rows[is_free], columns[is_free]].mean())

        for position in range(len(joints)):
            here = int(puck_position[position]) if numpy.isfinite(puck_position[position]) else -1
            track_rows.append({
                "recording": recording["recording"], "sheet": recording["sheet"], "plan": recording["plan"], "start_kind": recording["start_kind"],
                "start_end": recording["start_end"], "score": recording["score"], "row": int(recording["row"][position]),
                "frame_index": int(recording["frame_index"][position]), "action": recording["action"][position],
                "puck_x_cm": 100.0 * puck_xy[position, 0], "puck_y_cm": 100.0 * puck_xy[position, 1],
                "tip_x_cm": 100.0 * tip_xy[position, 0], "tip_y_cm": 100.0 * tip_xy[position, 1],
                "puck_along_cm": 0.1 * (here - first_pass) if here >= 0 else numpy.nan,
                "puck_beside_cm": 100.0 * puck_beside[position],
                "tip_along_cm": 0.1 * (tip_position[position] - first_pass), "tip_beside_cm": 100.0 * tip_beside[position],
                "line_direction_deg": described["direction_deg"][here] % 360.0 if here >= 0 else numpy.nan,
                "line_radius_cm": 100.0 * described["radius_m"][here] if here >= 0 else numpy.nan,
                "line_curvature_per_m": described["curvature_per_m"][here] if here >= 0 else numpy.nan,
                "is_in_loop": int(first_pass <= here <= second_pass) if here >= 0 else -1,
                "loop_length_cm": 0.1 * (second_pass - first_pass), "line_end_along_cm": 0.1 * (len(line) - 1 - first_pass),
                "is_puck_over_ink": is_over_ink_design[position], "ink_ahead_design": ink_ahead_design[position], "ink_ahead_picture": ink_ahead[position],
                "heading_deg": recording["heading_deg"][position], "joint_4_deg": twist_deg[position],
                "joint_5_deg": float(numpy.degrees(joints[position, 4])), "joint_6_deg": float(numpy.degrees(joints[position, 5])),
                "forward_cm": recording["forward_cm"][position], "left_cm": recording["left_cm"][position],
                "twist_deg": recording["twist_deg"][position], "stop_probability": recording["stop_probability"][position]
            })
        registration_rows.append({
            "recording": recording["recording"], "sheet": recording["sheet"], "plan": recording["plan"], "start_kind": recording["start_kind"],
            "start_end": recording["start_end"], "pictures": len(joints), "pictures_in_map": len(chosen),
            "angle_deg": placed["angle_deg"] % 360.0, "shift_x_cm": 100.0 * placed["shift_m"][0], "shift_y_cm": 100.0 * placed["shift_m"][1],
            "ink_share_on_ink_points": placed["ink_share_on_ink_points"], "ink_share_on_gap_points": placed["ink_share_on_gap_points"],
            "seen_share_of_line": placed["seen_share_of_line"], "ink_share_on_seen_ink_points": placed["ink_share_on_seen_ink_points"],
            "ink_share_on_seen_gap_points": placed["ink_share_on_seen_gap_points"], "method": placed["method"],
            "puck_seen": int(numpy.isfinite(puck_xy[:, 0]).sum()),
            "puck_beside_median_abs_cm": float(numpy.nanmedian(numpy.abs(100.0 * puck_beside)))
        })

        if number < len(axes):
            axis = axes[number]
            share = ink_share_of(view)
            axis.imshow(numpy.where(view["seen_votes"] >= 2, share, numpy.nan).T, origin = "lower", cmap = "gray_r", vmin = 0, vmax = 1.8,
                        extent = [100 * view["x_m"][0], 100 * view["x_m"][-1], 100 * view["y_m"][0], 100 * view["y_m"][-1]])
            axis.scatter(100 * line[::4, 0], 100 * line[::4, 1], c = numpy.where(is_ink_point[::4], "red", "deepskyblue"), s = 1)
            axis.plot(100 * puck_xy[:, 0], 100 * puck_xy[:, 1], "-", color = "limegreen", linewidth = 0.8)
            axis.plot(*(100 * line[0]), "s", color = "yellow", markeredgecolor = "black", markersize = 5)
            centre = 100 * line.mean(axis = 0)
            axis.set_xlim(centre[0] - 16, centre[0] + 16)
            axis.set_ylim(centre[1] - 16, centre[1] + 16)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            title = recording["recording"] if recording["plan"] == "" else f"{recording['plan']} {recording['start_kind']} from {recording['start_end']}"
            axis.set_title(f"{title}: ink {placed['ink_share_on_seen_ink_points']:.2f} ({placed['method']})", fontsize = 9)
    for axis in axes[len(recordings):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "02_registration_check.png", dpi = 45)
    tracks = pandas.DataFrame(track_rows)
    tracks.round(4).to_csv(EVIDENCE_FOLDER / "tracks.csv", index = False)
    registrations = pandas.DataFrame(registration_rows)
    registrations.round(3).to_csv(EVIDENCE_FOLDER / "registrations.csv", index = False)
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 200)
    print(f"{len(recordings)} recordings, {len(tracks)} pictures, {time.time() - started:.0f} s")
    print(registrations.drop(columns = ["recording"]).round(2).to_string(index = False))
    print("\nline fit, by sheet: ink share under the reference's ink points where the line was seen (mean and smallest)")
    print(registrations.groupby("sheet")["ink_share_on_seen_ink_points"].agg(["mean", "min"]).round(2).to_string())
    print("\npuck found in", int(tracks["puck_x_cm"].notna().sum()), "of", len(tracks), "pictures")



if __name__ == "__main__":
    main()
