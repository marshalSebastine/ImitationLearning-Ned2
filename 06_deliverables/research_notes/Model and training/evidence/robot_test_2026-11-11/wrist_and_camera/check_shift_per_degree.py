"""
Check the arm model and the fitted camera on real pictures where the wrist moved (key: wrist_and_camera, check).

held_wrist_views.py calculates how far the hand and the table move in the picture when the wrist differs. That rests
on the arm model and the camera fitted on 6.11.26. Here the calculation is tested on pairs of neighbouring
demonstration pictures between which the wrist jumped by 4 degrees or more while the forearm twist and the hand
stayed the same. Two things are measured by patch matching (normalised cross-correlation, cv2.matchTemplate) and
compared with what the models calculate for the same two joint sets:

    the block     a patch around the black magnet block: it must move as a point on the rod does, between the rod's end
                  (4.8 cm from the wrist's axis) and the magnet tip (7.4 cm from it)
    the table     patches of line and paper ahead of the hand: each must move as the table point it shows

Measured on the saved pictures; the predictions are calculated. Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe check_shift_per_degree.py
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import wrist_geometry


CHECK_PARAMETERS = {
    "smallest_wrist_jump_deg": 4.0,
    "largest_twist_change_deg": 1.0,
    "largest_hand_change_deg": 1.0,
    # the patch around the block: half its width and height in px at 640, centred this far from the tip's pixel
    "block_patch_half_px": [55, 45],
    "block_patch_offset_px": [0, -20],
    # table patches: offsets of their centres from the tip's pixel (u to the right, v downwards), all ahead of the hand
    "table_patch_half_px": [50, 40],
    "table_patch_offsets_px": [[0, -170], [-120, -170], [120, -170], [0, -230], [-150, -110], [150, -110]],
    "smallest_table_patch_contrast": 25.0,
    "search_px": 60,
    "smallest_match_score": 0.8,
    "figure_dpi": 120
}



def patch_move(
        first: numpy.ndarray,
        second: numpy.ndarray,
        centre_px: Tuple[int, int],
        half_px: Tuple[int, int],
        search_px: int
) -> Optional[Tuple[float, float, float, float]]:
    """
    Where a patch of the first picture is found in the second one.

    Returns
    -------
    tuple of float, or None
        The patch's move (u to the right, v downwards) in px, the match score (1 = identical) and the patch's own
        contrast (standard deviation of its grey levels); None when the patch or its search window leaves the picture.
    """

    centre_u, centre_v = centre_px
    half_u, half_v = half_px
    if centre_u - half_u - search_px < 0 or centre_u + half_u + search_px >= first.shape[1]:
        return None
    if centre_v - half_v - search_px < 0 or centre_v + half_v + search_px >= first.shape[0]:
        return None
    patch = first[centre_v - half_v:centre_v + half_v, centre_u - half_u:centre_u + half_u]
    window = second[centre_v - half_v - search_px:centre_v + half_v + search_px, centre_u - half_u - search_px:centre_u + half_u + search_px]
    scores = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
    _, best_score, _, best_place = cv2.minMaxLoc(scores)
    return float(best_place[0] - search_px), float(best_place[1] - search_px), float(best_score), float(patch.std())



def main(parameters: Optional[Dict] = None) -> None:
    """Measure the block's and the table's moves across wrist jumps and compare them with the calculated moves."""

    settings = dict(CHECK_PARAMETERS)
    settings.update(parameters or {})
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    raw_folder = Path(wrist_geometry.ANALYSIS_PARAMETERS["raw_folder"])
    frames = wrist_geometry.load_demonstration_frames()
    joints = frames[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)
    has_next = (frames["episode_id"].shift(-1) == frames["episode_id"]).to_numpy()
    rows_now = numpy.flatnonzero(has_next)
    changes = numpy.degrees(joints[rows_now + 1] - joints[rows_now])
    is_chosen = ((numpy.abs(changes[:, 4]) >= settings["smallest_wrist_jump_deg"]) & (numpy.abs(changes[:, 3]) <= settings["largest_twist_change_deg"])
                 & (numpy.abs(changes[:, 5]) <= settings["largest_hand_change_deg"]))
    rows_now, changes = rows_now[is_chosen], changes[is_chosen]
    print(f"steps with a wrist jump of {settings['smallest_wrist_jump_deg']} degrees or more, the twist within {settings['largest_twist_change_deg']} and the hand within "
          f"{settings['largest_hand_change_deg']} degree: {len(rows_now)}")

    frames_now = wrist_geometry.frames_of_many(joints[rows_now])
    frames_next = wrist_geometry.frames_of_many(joints[rows_now + 1])
    tips_now = wrist_geometry.tips_and_leans(joints[rows_now])[0]
    tips_next = wrist_geometry.tips_and_leans(joints[rows_now + 1])[0]
    tip_pixels_now = wrist_geometry.project_many(tips_now, joints[rows_now])
    tip_moves = wrist_geometry.project_many(tips_next, joints[rows_now + 1]) - tip_pixels_now
    rod_end_moves = wrist_geometry.project_many(frames_next["tool"][0], joints[rows_now + 1]) - wrist_geometry.project_many(frames_now["tool"][0], joints[rows_now])
    table_points = tips_now.copy()
    table_points[:, 2] = wrist_geometry.table_heights(tips_now[:, :2])
    normal = wrist_geometry.table_plane_normal()

    block_rows, table_rows = [], []
    for position, row in enumerate(rows_now):
        first = cv2.imread(str(raw_folder / frames["frame_path"].iloc[row]), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(raw_folder / frames["frame_path"].iloc[row + 1]), cv2.IMREAD_GRAYSCALE)
        if first is None or second is None or not numpy.all(numpy.isfinite(tip_pixels_now[position])):
            continue
        tip_u, tip_v = int(round(tip_pixels_now[position, 0])), int(round(tip_pixels_now[position, 1]))
        found = patch_move(first, second, (tip_u + settings["block_patch_offset_px"][0], tip_v + settings["block_patch_offset_px"][1]),
                           tuple(settings["block_patch_half_px"]), settings["search_px"])
        if found is not None:
            block_rows.append({"row": int(row), "wrist_change_deg": changes[position, 4], "measured_dv_px": found[1], "match_score": found[2],
                               "tip_dv_px": tip_moves[position, 1], "rod_end_dv_px": rod_end_moves[position, 1]})
        for offset_u, offset_v in settings["table_patch_offsets_px"]:
            centre = (tip_u + offset_u, tip_v + offset_v)
            found = patch_move(first, second, centre, tuple(settings["table_patch_half_px"]), settings["search_px"])
            if found is None or found[3] < settings["smallest_table_patch_contrast"]:
                continue
            point = wrist_geometry.pixels_to_table(numpy.array([centre], dtype = float), joints[row:row + 1], table_points[position:position + 1], normal)
            predicted = wrist_geometry.project_many(point, joints[row + 1:row + 2])[0] - numpy.array(centre, dtype = float)
            table_rows.append({"row": int(row), "wrist_change_deg": changes[position, 4], "measured_du_px": found[0], "measured_dv_px": found[1], "match_score": found[2],
                               "predicted_du_px": predicted[0], "predicted_dv_px": predicted[1]})

    block = pandas.DataFrame(block_rows)
    block = block[block["match_score"] >= settings["smallest_match_score"]]
    measured_slope = numpy.polyfit(block["wrist_change_deg"], block["measured_dv_px"], 1)[0]
    tip_slope = numpy.polyfit(block["wrist_change_deg"], block["tip_dv_px"], 1)[0]
    rod_end_slope = numpy.polyfit(block["wrist_change_deg"], block["rod_end_dv_px"], 1)[0]
    print(f"\nthe block ({len(block)} pairs with a match score of {settings['smallest_match_score']} or more): it moves by {measured_slope:.2f} px at 640 per degree of wrist "
          f"(measured; negative = up the picture when the wrist grows); calculated for the same pairs: the magnet tip {tip_slope:.2f}, the rod's end {rod_end_slope:.2f}")
    print(f"  correlation of the measured move with the calculated tip move {numpy.corrcoef(block['measured_dv_px'], block['tip_dv_px'])[0, 1]:.3f}, "
          f"with the calculated rod-end move {numpy.corrcoef(block['measured_dv_px'], block['rod_end_dv_px'])[0, 1]:.3f}; "
          f"measured minus rod-end move: median |x| {(block['measured_dv_px'] - block['rod_end_dv_px']).abs().median():.2f} px")

    table = pandas.DataFrame(table_rows)
    table = table[table["match_score"] >= settings["smallest_match_score"]]
    gap_v = table["measured_dv_px"] - table["predicted_dv_px"]
    gap_u = table["measured_du_px"] - table["predicted_du_px"]
    print(f"\nthe table ({len(table)} patches from {table['row'].nunique()} pairs): measured against calculated move down the picture: slope "
          f"{numpy.polyfit(table['predicted_dv_px'], table['measured_dv_px'], 1)[0]:.3f}, correlation {numpy.corrcoef(table['predicted_dv_px'], table['measured_dv_px'])[0, 1]:.3f}; "
          f"measured minus calculated: median {gap_v.median():+.2f}, median |x| {gap_v.abs().median():.2f}, 90th percentile |x| {gap_v.abs().quantile(0.9):.2f} px")
    print(f"  sideways: slope {numpy.polyfit(table['predicted_du_px'], table['measured_du_px'], 1)[0]:.3f}, correlation {numpy.corrcoef(table['predicted_du_px'], table['measured_du_px'])[0, 1]:.3f}, "
          f"median |gap| {gap_u.abs().median():.2f} px; the moves themselves: median |x| {table['measured_dv_px'].abs().median():.1f} px down, {table['measured_du_px'].abs().median():.1f} px sideways")
    print("  (between these neighbouring pictures the hand-guided wrist jump moved the block over the table; the forearm, and with it the camera, moved little, so the table "
          "moved little. With the wrist HELD, joints 1 to 3 must make up for it and the camera moves: that case is calculated in held_wrist_views.py with the same models.)")
    block.to_csv(output_folder / "check_shift_per_degree_block.csv", index = False)
    table.to_csv(output_folder / "check_shift_per_degree_table.csv", index = False)

    figure, axes = pyplot.subplots(1, 2, figsize = (12.5, 4.8))
    axes[0].scatter(block["wrist_change_deg"], block["measured_dv_px"], s = 14, color = "tab:blue", label = "measured (patch on the block)")
    axes[0].scatter(block["wrist_change_deg"], block["tip_dv_px"], s = 8, color = "tab:orange", label = "calculated: magnet tip")
    axes[0].scatter(block["wrist_change_deg"], block["rod_end_dv_px"], s = 8, color = "tab:green", label = "calculated: rod's end")
    axes[0].set_xlabel("wrist change between two neighbouring pictures (degrees)")
    axes[0].set_ylabel("move down the picture (px at 640)")
    axes[0].set_title(f"The block: measured {measured_slope:.2f}, tip {tip_slope:.2f}, rod's end {rod_end_slope:.2f} px per degree")
    axes[0].legend(fontsize = 8)
    axes[0].grid(alpha = 0.3)
    axes[1].scatter(table["predicted_dv_px"], table["measured_dv_px"], s = 10, color = "tab:blue")
    limit = float(max(table["predicted_dv_px"].abs().max(), table["measured_dv_px"].abs().max()))
    axes[1].plot([-limit, limit], [-limit, limit], color = "black", linewidth = 1)
    axes[1].set_xlabel("calculated move of the table patch down the picture (px at 640)")
    axes[1].set_ylabel("measured move (px at 640)")
    axes[1].set_title(f"The table: {len(table)} patches, median |gap| {gap_v.abs().median():.1f} px")
    axes[1].grid(alpha = 0.3)
    figure.suptitle("Arm model and fitted camera against real picture pairs across wrist jumps", fontsize = 10)
    figure.tight_layout()
    figure.savefig(output_folder / "check_shift_per_degree.png", dpi = settings["figure_dpi"])
    pyplot.close(figure)
    print(f"\nsaved {output_folder / 'check_shift_per_degree.png'} and two csv tables")



if __name__ == "__main__":
    main()
