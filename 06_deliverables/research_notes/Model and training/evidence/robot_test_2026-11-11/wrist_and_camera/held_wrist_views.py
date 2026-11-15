"""
How far does the held wrist move the camera's view away from the demonstrated view? (key: wrist_and_camera, part 1)

For every demonstration row the driver's joints are worked out for the SAME tip point and the SAME forearm twist
(joint 4), with the wrist (joint 5) held at 13.5 degrees and the hand (joint 6) at -6.1 degrees, as model_mode.py does.
The forearm camera's pose (fixed in the arm model's joint_4 frame) is compared with the demonstrated one, and the
change is turned into pixels with the fitted camera model: where the magnet tip, a point 2 cm ahead of it on the table
and a grid of table points seen in the demonstrated picture land in the held-wrist picture.

Four variants:
    held_same_tip      wrist 13.5, hand -6.1, the demonstrated tip point (the wrist's effect alone)
    held_driver_height wrist 13.5, hand -6.1, the tip 5 mm above the fitted table as the driver puts it
    height_only        the demonstrated wrist, the tip 5 mm above the fitted table (the height's effect alone)
    held_first_row     the wrist of the episode's first row (the earlier audit's variant, the source of "23.3 %")

Everything is calculated from the logged joints with the arm model and the camera model; nothing is measured on the
robot here. Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe held_wrist_views.py
"""

import math
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas

from matplotlib import pyplot

import wrist_geometry
from config import RECORDING_PARAMETERS
from record_episodes import forearm_camera_heading_rad


VIEW_PARAMETERS = {
    # a grid of pixels of the demonstrated 640 x 480 picture whose table points are followed into the other view
    "grid_u_px": [64.0, 192.0, 320.0, 448.0, 576.0],
    "grid_v_px": [60.0, 180.0, 300.0, 420.0],
    "picture_centre_px": [320.0, 240.0],
    # a point this far ahead of the tip on the table, towards the picture's top: about where the puck's foot is
    "ahead_of_tip_m": 0.02,
    "model_picture_width_px": 160.0,
    "camera_picture_width_px": 640.0,
    "twist_bins_deg": [0.0, 30.0, 60.0, 90.0, 120.0],
    "reach_bins_m": [0.12, 0.20, 0.26, 0.32, 0.38, 0.50],
    "wrist_offset_bins_deg": [-45.0, -20.0, -10.0, -5.0, -2.0, 2.0, 5.0, 10.0, 20.0, 35.0],
    "thresholds_640_px": [8.0, 16.0, 20.0, 40.0],
    "figure_dpi": 130
}



def headings_deg(joint_sets: numpy.ndarray) -> numpy.ndarray:
    """The table direction of the picture's top for many joint sets, in degrees (the recording tool's own function)."""

    return numpy.degrees(numpy.array([forearm_camera_heading_rad(list(joints)) for joints in joint_sets]))



def view_change(
        joints_demonstrated: numpy.ndarray,
        joints_other: numpy.ndarray,
        is_solved: numpy.ndarray,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    The camera's move and the picture's shift between the demonstrated joints and another set of joints, row by row.

    Parameters
    ----------
    joints_demonstrated : numpy.ndarray
        The logged joints of every demonstration row, shape (N, 6).
    joints_other : numpy.ndarray
        The joints of the variant, shape (N, 6).
    is_solved : numpy.ndarray
        Whether the variant has a solution for the row; the other rows get NaN.
    parameters : dict, optional
        Overrides of VIEW_PARAMETERS.

    Returns
    -------
    pandas.DataFrame
        Per row: "camera_move_mm", "camera_turn_deg" (the angle between the two viewing directions), "heading_change_deg"
        (the picture's top on the table), "lean_other_deg", the shift in 640-px pixels of the tip ("tip_du_px",
        "tip_dv_px", "tip_shift_px"), of the point ahead of the tip ("ahead_shift_px"), of the table point at the
        picture's centre ("centre_shift_px"), and the mean and the largest shift over the grid ("grid_mean_shift_px",
        "grid_most_shift_px", "grid_mean_du_px", "grid_mean_dv_px").
    """

    settings = dict(VIEW_PARAMETERS)
    settings.update(parameters or {})
    count = len(joints_demonstrated)
    normal = wrist_geometry.table_plane_normal()

    centres_demonstrated, axes_demonstrated = wrist_geometry.camera_poses(joints_demonstrated)
    centres_other, axes_other = wrist_geometry.camera_poses(joints_other)
    tips_demonstrated, _ = wrist_geometry.tips_and_leans(joints_demonstrated)
    tips_other, leans_other = wrist_geometry.tips_and_leans(joints_other)
    cosines = numpy.clip(numpy.einsum("ni,ni->n", axes_demonstrated[:, :, 2], axes_other[:, :, 2]), -1.0, 1.0)

    result = pandas.DataFrame({
        "camera_move_mm": 1000.0 * numpy.linalg.norm(centres_other - centres_demonstrated, axis = 1),
        "camera_rise_mm": 1000.0 * (centres_other[:, 2] - centres_demonstrated[:, 2]),
        "camera_turn_deg": numpy.degrees(numpy.arccos(cosines)),
        "lean_other_deg": leans_other
    })
    heading_demonstrated = headings_deg(joints_demonstrated)
    heading_other = headings_deg(joints_other)
    result["heading_change_deg"] = (heading_other - heading_demonstrated + 180.0) % 360.0 - 180.0

    # the tip is on the hand, so each view shows its own tip; the table points are the same points in both views
    tip_pixels_demonstrated = wrist_geometry.project_many(tips_demonstrated, joints_demonstrated)
    tip_pixels_other = wrist_geometry.project_many(tips_other, joints_other)
    result["tip_u_demonstrated_px"] = tip_pixels_demonstrated[:, 0]
    result["tip_v_demonstrated_px"] = tip_pixels_demonstrated[:, 1]
    result["tip_du_px"] = tip_pixels_other[:, 0] - tip_pixels_demonstrated[:, 0]
    result["tip_dv_px"] = tip_pixels_other[:, 1] - tip_pixels_demonstrated[:, 1]
    result["tip_shift_px"] = numpy.hypot(result["tip_du_px"], result["tip_dv_px"])

    # table points under the demonstrated tip: the plane through the point below the tip, parallel to the fitted table
    table_points = tips_demonstrated.copy()
    table_points[:, 2] = wrist_geometry.table_heights(tips_demonstrated[:, :2])
    heading_rad = numpy.radians(heading_demonstrated)
    ahead = table_points.copy()
    ahead[:, 0] += settings["ahead_of_tip_m"] * numpy.cos(heading_rad)
    ahead[:, 1] += settings["ahead_of_tip_m"] * numpy.sin(heading_rad)
    ahead[:, 2] = wrist_geometry.table_heights(ahead[:, :2])
    ahead_shift = wrist_geometry.project_many(ahead, joints_other) - wrist_geometry.project_many(ahead, joints_demonstrated)
    result["ahead_du_px"] = ahead_shift[:, 0]
    result["ahead_dv_px"] = ahead_shift[:, 1]
    result["ahead_shift_px"] = numpy.hypot(ahead_shift[:, 0], ahead_shift[:, 1])

    centre_pixels = numpy.tile(numpy.array(settings["picture_centre_px"]), (count, 1))
    centre_points = wrist_geometry.pixels_to_table(centre_pixels, joints_demonstrated, table_points, normal)
    centre_shift = wrist_geometry.project_many(centre_points, joints_other) - centre_pixels
    result["centre_shift_px"] = numpy.hypot(centre_shift[:, 0], centre_shift[:, 1])

    grid_shifts = []
    for grid_u in settings["grid_u_px"]:
        for grid_v in settings["grid_v_px"]:
            pixels = numpy.tile(numpy.array([grid_u, grid_v]), (count, 1))
            points = wrist_geometry.pixels_to_table(pixels, joints_demonstrated, table_points, normal)
            grid_shifts.append(wrist_geometry.project_many(points, joints_other) - pixels)
    grid_shifts = numpy.array(grid_shifts)
    grid_lengths = numpy.hypot(grid_shifts[:, :, 0], grid_shifts[:, :, 1])
    with numpy.errstate(invalid = "ignore", all = "ignore"):
        result["grid_mean_shift_px"] = numpy.nanmean(grid_lengths, axis = 0)
        result["grid_most_shift_px"] = numpy.nanmax(grid_lengths, axis = 0)
        result["grid_mean_du_px"] = numpy.nanmean(grid_shifts[:, :, 0], axis = 0)
        result["grid_mean_dv_px"] = numpy.nanmean(grid_shifts[:, :, 1], axis = 0)

    result.loc[~is_solved, :] = numpy.nan
    return result



def shift_summary(
        values: pandas.Series,
        parameters: Optional[Dict] = None
) -> Dict[str, float]:
    """The median, 90th and 95th percentile, the largest value and the shares above the thresholds, for one shift column."""

    settings = dict(VIEW_PARAMETERS)
    settings.update(parameters or {})
    values = values.dropna()
    summary = {"rows": len(values), "median": values.median(), "p90": values.quantile(0.9), "p95": values.quantile(0.95), "most": values.max()}
    for threshold in settings["thresholds_640_px"]:
        summary[f"over_{threshold:.0f}px"] = float((values > threshold).mean())
    return summary



def main(parameters: Optional[Dict] = None) -> None:
    """Compute the four variants for every demonstration row, print the tables and save the figures and the row table."""

    settings = dict(VIEW_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 200)
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    scale = settings["model_picture_width_px"] / settings["camera_picture_width_px"]

    frames = wrist_geometry.load_demonstration_frames()
    joints_demonstrated = frames[wrist_geometry.JOINT_COLUMNS].to_numpy(dtype = float)
    count = len(frames)
    tips, leans = wrist_geometry.tips_and_leans(joints_demonstrated)
    table = wrist_geometry.table_heights(tips[:, :2])
    frames["tip_above_table_mm"] = 1000.0 * (tips[:, 2] - table)
    frames["lean_fk_deg"] = leans
    frames["wrist_offset_deg"] = frames["joint_5_deg"] - wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]

    wrist_held = numpy.full(count, math.radians(wrist_geometry.ANALYSIS_PARAMETERS["wrist_held_deg"]))
    hand_held = numpy.full(count, math.radians(wrist_geometry.ANALYSIS_PARAMETERS["hand_held_deg"]))
    driver_tips = tips.copy()
    driver_tips[:, 2] = table + wrist_geometry.ANALYSIS_PARAMETERS["driver_tip_above_table_m"]
    first_row_wrist = frames.groupby("episode_id")["joint_5"].transform("first").to_numpy()
    starts = joints_demonstrated[:, :3]

    variants = {
        "held_same_tip": wrist_geometry.solve_many_with_wrist_fixed(tips, joints_demonstrated[:, 3], wrist_held, starts, hand_held),
        "held_driver_height": wrist_geometry.solve_many_with_wrist_fixed(driver_tips, joints_demonstrated[:, 3], wrist_held, starts, hand_held),
        "height_only": wrist_geometry.solve_many_with_wrist_fixed(driver_tips, joints_demonstrated[:, 3], joints_demonstrated[:, 4], starts, joints_demonstrated[:, 5]),
        "held_first_row": wrist_geometry.solve_many_with_wrist_fixed(tips, joints_demonstrated[:, 3], first_row_wrist, starts, joints_demonstrated[:, 5])
    }

    print(f"demonstration rows: {count}; tip above the fitted table (mm): median {frames['tip_above_table_mm'].median():.1f}, "
          f"5th to 95th percentile {frames['tip_above_table_mm'].quantile(0.05):.1f} to {frames['tip_above_table_mm'].quantile(0.95):.1f} "
          f"(the driver: {1000.0 * wrist_geometry.ANALYSIS_PARAMETERS['driver_tip_above_table_m']:.1f})")
    print(f"demonstrated wrist minus the held 13.5 degrees: within 2 degrees in {100.0 * (frames['wrist_offset_deg'].abs() <= 2.0).mean():.1f} % of rows, "
          f"over 5 in {100.0 * (frames['wrist_offset_deg'].abs() > 5.0).mean():.1f} %, over 10 in {100.0 * (frames['wrist_offset_deg'].abs() > 10.0).mean():.1f} %")

    changes = {}
    for name, (joints_other, is_solved) in variants.items():
        changes[name] = view_change(joints_demonstrated, joints_other, is_solved, settings)
        print(f"\n=== variant {name}: solved for {int(is_solved.sum())} of {count} rows ({100.0 * is_solved.mean():.2f} %) ===")
        rows = []
        for column in ["camera_move_mm", "camera_turn_deg", "tip_shift_px", "ahead_shift_px", "centre_shift_px", "grid_mean_shift_px", "grid_most_shift_px"]:
            rows.append(dict({"quantity": column}, **shift_summary(changes[name][column], settings)))
        print(pandas.DataFrame(rows).round(3).to_string(index = False))
        print(f"  heading change (degrees): median |x| {changes[name]['heading_change_deg'].abs().median():.2f}, 95th percentile {changes[name]['heading_change_deg'].abs().quantile(0.95):.2f}")
        print(f"  at the model's 160-px width the same shifts are a quarter: tip median {scale * changes[name]['tip_shift_px'].median():.2f} px, "
              f"90th percentile {scale * changes[name]['tip_shift_px'].quantile(0.9):.2f} px, largest {scale * changes[name]['tip_shift_px'].max():.2f} px; "
              f"over the augmentation's 4 px (16 px at 640) in {100.0 * (changes[name]['tip_shift_px'] > 16.0).mean():.1f} % of rows")

    held = changes["held_same_tip"]
    frames = pandas.concat([frames, held.add_prefix("held_")], axis = 1)
    frames["driver_tip_shift_px"] = changes["held_driver_height"]["tip_shift_px"]
    frames["driver_ahead_shift_px"] = changes["held_driver_height"]["ahead_shift_px"]
    frames["driver_grid_mean_shift_px"] = changes["held_driver_height"]["grid_mean_shift_px"]
    frames["is_held_solved"] = variants["held_same_tip"][1]
    frames["abs_joint_4_deg"] = frames["joint_4_deg"].abs()
    frames["twist_bin"] = pandas.cut(frames["abs_joint_4_deg"], settings["twist_bins_deg"], include_lowest = True)
    frames["reach_bin"] = pandas.cut(frames["reach_m"], settings["reach_bins_m"], include_lowest = True)
    frames["wrist_offset_bin"] = pandas.cut(frames["wrist_offset_deg"], settings["wrist_offset_bins_deg"], include_lowest = True)

    def group_table(keys) -> pandas.DataFrame:
        grouped = frames.groupby(keys, observed = True)
        return pandas.DataFrame({
            "rows": grouped.size(),
            "wrist_offset_median_deg": grouped["wrist_offset_deg"].median(),
            "abs_wrist_offset_p90_deg": grouped["wrist_offset_deg"].apply(lambda values: values.abs().quantile(0.9)),
            "tip_shift_median_px": grouped["held_tip_shift_px"].median(),
            "tip_shift_p90_px": grouped["held_tip_shift_px"].quantile(0.9),
            "ahead_shift_p90_px": grouped["held_ahead_shift_px"].quantile(0.9),
            "grid_mean_shift_p90_px": grouped["held_grid_mean_shift_px"].quantile(0.9),
            "tip_over_16px_share": grouped["held_tip_shift_px"].apply(lambda values: (values > 16.0).mean()),
            "tip_over_20px_share": grouped["held_tip_shift_px"].apply(lambda values: (values > 20.0).mean()),
            "tip_over_40px_share": grouped["held_tip_shift_px"].apply(lambda values: (values > 40.0).mean()),
            "not_solved_share": grouped["is_held_solved"].apply(lambda values: 1.0 - values.mean())
        })

    print("\n=== held_same_tip, by sheet (pixels at 640; divide by 4 for the model's 160) ===")
    print(group_table("sheet").round(3).to_string())
    print("\n=== held_same_tip, by |joint 4| ===")
    print(group_table("twist_bin").round(3).to_string())
    print("\n=== held_same_tip, by reach (tip distance from the base) ===")
    print(group_table("reach_bin").round(3).to_string())
    print("\n=== held_same_tip, by |joint 4| and reach: share of rows with the tip shifted over 20 px ===")
    print(frames.pivot_table(index = "twist_bin", columns = "reach_bin", values = "held_tip_shift_px", aggfunc = lambda values: (values > 20.0).mean(), observed = True).round(3).to_string())
    print("\n=== held_same_tip, by |joint 4| and reach: 90th percentile of the tip shift (px at 640) ===")
    print(frames.pivot_table(index = "twist_bin", columns = "reach_bin", values = "held_tip_shift_px", aggfunc = lambda values: values.quantile(0.9), observed = True).round(1).to_string())
    print("\n=== held_same_tip, by |joint 4| and reach: rows ===")
    print(frames.pivot_table(index = "twist_bin", columns = "reach_bin", values = "held_tip_shift_px", aggfunc = "size", observed = True).to_string())
    print("\n=== held_same_tip, by the demonstrated wrist minus 13.5 degrees ===")
    by_offset = frames.groupby("wrist_offset_bin", observed = True)
    print(pandas.DataFrame({
        "rows": by_offset.size(),
        "tip_du_median_px": by_offset["held_tip_du_px"].median(),
        "tip_dv_median_px": by_offset["held_tip_dv_px"].median(),
        "tip_shift_median_px": by_offset["held_tip_shift_px"].median(),
        "ahead_shift_median_px": by_offset["held_ahead_shift_px"].median(),
        "centre_shift_median_px": by_offset["held_centre_shift_px"].median(),
        "grid_mean_shift_median_px": by_offset["held_grid_mean_shift_px"].median(),
        "camera_move_median_mm": by_offset["held_camera_move_mm"].median(),
        "camera_turn_median_deg": by_offset["held_camera_turn_deg"].median(),
        "heading_change_median_deg": by_offset["held_heading_change_deg"].median()
    }).round(2).to_string())

    is_fit = frames["is_held_solved"] & (frames["wrist_offset_deg"].abs() > 1.0)
    slope = numpy.polyfit(frames.loc[is_fit, "wrist_offset_deg"], frames.loc[is_fit, "held_tip_dv_px"], 1)
    print(f"\ntip's vertical shift against the wrist offset (rows over 1 degree off): {slope[0]:.3f} px at 640 per degree "
          f"({scale * slope[0]:.3f} px at 160 per degree), intercept {slope[1]:.2f} px")
    slope_ahead = numpy.polyfit(frames.loc[is_fit, "wrist_offset_deg"], frames.loc[is_fit, "held_ahead_shift_px"] * numpy.sign(frames.loc[is_fit, "wrist_offset_deg"]), 1)
    print(f"shift of the table point 2 cm ahead of the tip against the wrist offset: {slope_ahead[0]:.3f} px at 640 per degree")

    lean_demonstrated = frames["lean_fk_deg"]
    lean_held = frames["held_lean_other_deg"]
    print(f"\nrod lean (degrees from straight down): demonstrated median {lean_demonstrated.median():.1f}, 95th {lean_demonstrated.quantile(0.95):.1f}, "
          f"99th {lean_demonstrated.quantile(0.99):.1f}, largest {lean_demonstrated.max():.1f}; with the wrist held: median {lean_held.median():.1f}, "
          f"95th {lean_held.quantile(0.95):.1f}, 99th {lean_held.quantile(0.99):.1f}, largest {lean_held.max():.1f}; "
          f"over the driver's 58-degree limit in {int((lean_held > 58.0).sum())} rows, not solvable in {int((~frames['is_held_solved']).sum())} rows")

    # a compact row table (the picture is found again through frame_index in frames.csv)
    kept_columns = ["frame_index", "sheet", "row_in_episode", "joint_4_deg", "joint_5_deg", "reach_m", "tip_above_table_mm", "held_camera_move_mm",
                    "held_camera_turn_deg", "held_heading_change_deg", "held_lean_other_deg", "held_tip_dv_px", "held_tip_shift_px", "held_ahead_shift_px",
                    "held_centre_shift_px", "held_grid_mean_shift_px", "held_grid_most_shift_px", "driver_grid_mean_shift_px"]
    frames[kept_columns].round(3).to_csv(output_folder / "held_wrist_views_rows.csv", index = False)

    figure, axes = pyplot.subplots(1, 3, figsize = (16, 4.6))
    for name, colour in [("held_same_tip", "tab:blue"), ("held_driver_height", "tab:red"), ("height_only", "tab:green"), ("held_first_row", "tab:grey")]:
        values = numpy.sort(changes[name]["grid_mean_shift_px"].dropna().to_numpy())
        axes[0].plot(values, numpy.linspace(0.0, 1.0, len(values)), color = colour, label = name)
    axes[0].axvline(16.0, color = "black", linestyle = ":", linewidth = 1)
    axes[0].text(16.5, 0.05, "training shift\naugmentation\n(4 px at 160)", fontsize = 8)
    axes[0].set_xlim(0.0, 100.0)
    axes[0].set_xlabel("mean shift of the table points in the picture (px at 640 x 480)")
    axes[0].set_ylabel("share of demonstration rows below")
    axes[0].set_title("How far the table sits from its demonstrated place")
    axes[0].legend(fontsize = 8, loc = "lower right")
    axes[0].grid(alpha = 0.3)

    sample = frames[frames["is_held_solved"]].sample(n = 6000, random_state = wrist_geometry.ANALYSIS_PARAMETERS["check_seed"])
    points = axes[1].scatter(sample["wrist_offset_deg"], sample["held_tip_dv_px"], c = sample["abs_joint_4_deg"], s = 4, cmap = "viridis")
    figure.colorbar(points, ax = axes[1], label = "|joint 4| (degrees)")
    axes[1].set_xlabel("demonstrated wrist minus the held 13.5 (degrees)")
    axes[1].set_ylabel("tip moves down the picture by (px at 640)")
    axes[1].set_title(f"The tip's place follows the wrist alone: {slope[0]:.2f} px per degree")
    axes[1].grid(alpha = 0.3)

    share = frames.pivot_table(index = "twist_bin", columns = "reach_bin", values = "held_tip_shift_px", aggfunc = lambda values: (values > 20.0).mean(), observed = True)
    image = axes[2].imshow(100.0 * share.to_numpy(dtype = float), cmap = "Reds", vmin = 0.0, vmax = 60.0, aspect = "auto")
    axes[2].set_xticks(range(len(share.columns)))
    axes[2].set_xticklabels([f"{100 * interval.left:.0f} to {100 * interval.right:.0f}" for interval in share.columns], fontsize = 8)
    axes[2].set_yticks(range(len(share.index)))
    axes[2].set_yticklabels([f"{interval.left:.0f} to {interval.right:.0f}" for interval in share.index])
    for row in range(share.shape[0]):
        for column in range(share.shape[1]):
            value = share.to_numpy(dtype = float)[row, column]
            if numpy.isfinite(value):
                axes[2].text(column, row, f"{100.0 * value:.0f}", ha = "center", va = "center", fontsize = 9)
    axes[2].set_xlabel("reach: tip distance from the base (cm)")
    axes[2].set_ylabel("|joint 4| (degrees)")
    axes[2].set_title("Rows with the tip over 20 px from its demonstrated place (%)")
    figure.colorbar(image, ax = axes[2])
    figure.suptitle("Wrist held at 13.5 degrees against the demonstrated wrist, all 18,522 demonstration rows (calculated: arm model and fitted camera)", fontsize = 10)
    figure.tight_layout()
    figure.savefig(output_folder / "held_wrist_views.png", dpi = settings["figure_dpi"])
    pyplot.close(figure)
    print(f"\nsaved {output_folder / 'held_wrist_views.png'} and held_wrist_views_rows.csv")



if __name__ == "__main__":
    main()
