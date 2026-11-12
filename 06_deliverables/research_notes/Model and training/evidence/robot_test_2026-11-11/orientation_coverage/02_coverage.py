"""
Question 2 of the orientation-coverage analysis: what do the demonstrations cover, and where are the gaps?

Measured from the audit-cleaned index (18,032 pictures of 209 episodes):
    - the direction of travel on the table relative to the arm's radial direction ("from radial": 0 = straight away
      from the base, +90 = sideways to the left, 180 = straight back to the base), per labelled step and per path of
      the next three steps;
    - the forearm twist (joint 4) and how far the picture's top is turned from the radial direction;
    - the direction of travel in the picture's frame (0 = towards the picture's top);
    - the two-way tables that show the gaps: direction from radial against reach, and against joint 4;
    - the first five moving steps of every episode;
    - where the ten robot runs of 11.11.26 fall in the same tables.

Printed and saved: 02_coverage.out.txt, 02_coverage_cells.csv (pictures, episodes and lay-outs per cell),
02_first_five_steps.csv (one row per episode), 02_coverage.png.

Run from this folder (after 01_placements.py, whose lay-out names it reads):
    D:/AI_Project/.venv/Scripts/python.exe 02_coverage.py
"""

import sys
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import numpy
import pandas
from matplotlib import pyplot

import orientation_common
from orientation_common import EVIDENCE_FOLDER, circular_mean_deg, demonstration_frames, robot_run_steps, wrapped_deg

COVERAGE_PARAMETERS = {
    # direction bins of 30 degrees centred on 0, 30, ... (the bin "180" holds 165 to 180 and -180 to -165)
    "direction_bin_deg": 30.0,
    "reach_edges_cm": [12.0, 20.0, 27.5, 35.0, 42.5, 49.5],
    "joint_4_edges_deg": [-120.0, -90.0, -60.0, -30.0, -10.0, 10.0, 30.0, 60.0, 90.0, 120.0],
    # a cell counts as thinly covered below this many pictures or lay-outs (a choice, stated with every result)
    "thin_cell_pictures": 100,
    "thin_cell_layouts": 3,
    "first_steps": 5,
    # a robot step's neighbours: demonstrated moving steps within these distances in forearm twist, travel direction
    # from radial and reach; a step with fewer than "few_neighbours" of them lies outside what was demonstrated
    "neighbour_joint_4_deg": 10.0,
    "neighbour_direction_deg": 15.0,
    "neighbour_reach_cm": 5.0,
    "few_neighbours": 20,
    "figure_path": EVIDENCE_FOLDER / "02_coverage.png"
}



def direction_bin(angle_deg, bin_deg: float) -> numpy.ndarray:
    """Return the centre of the direction bin of each angle: a multiple of ``bin_deg`` in -180 + bin to 180."""

    centres = numpy.round(numpy.asarray(angle_deg, dtype = float) / bin_deg) * bin_deg
    centres = wrapped_deg(centres)
    centres[centres == -180.0] = 180.0
    return centres



def share_table(
        frames: pandas.DataFrame,
        column: str,
        group: str = "sheet_number"
) -> pandas.DataFrame:
    """Return the share (%) of rows per value of ``column``, per ``group`` and for all rows."""

    counts = pandas.crosstab(frames[column], frames[group])
    counts["all"] = counts.sum(axis = 1)
    shares = 100.0 * counts / counts.sum(axis = 0)
    shares["pictures"] = counts["all"]
    return shares.round(1)



def first_steps_table(
        frames: pandas.DataFrame,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Describe the first moving steps of every episode.

    Returns
    -------
    pandas.DataFrame
        One row per episode: the net direction of the first "first_steps" moving steps on the table (from +x and
        from radial) and in the first picture's frame, the net length, how many rows passed before the tip had
        travelled 1 cm, the forearm twist over those steps and over the first 3 cm of travel, and the reach at the start.
    """

    rows = []
    for episode_id, group in frames.groupby("episode_id", sort = False):
        moving = group[group["step_cm"] >= parameters["smallest_step_cm"]].head(parameters["first_steps"])
        if len(moving) == 0:
            continue
        heading = numpy.radians(moving["step_table_deg"].to_numpy())
        net_x = float((moving["step_cm"] * numpy.cos(heading)).sum())
        net_y = float((moving["step_cm"] * numpy.sin(heading)).sum())
        net_table = float(numpy.degrees(numpy.arctan2(net_y, net_x)))
        first = group.iloc[0]
        tip = 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy()
        travelled = numpy.linalg.norm(tip - tip[0], axis = 1)
        rows_to_one_cm = int(numpy.argmax(travelled >= 1.0)) if (travelled >= 1.0).any() else len(group)
        rows_to_three_cm = int(numpy.argmax(travelled >= 3.0)) if (travelled >= 3.0).any() else len(group) - 1
        rows.append({
            "episode_id": episode_id, "plan": first["plan_number"], "sheet": first["sheet_number"], "end": first["start_end"],
            "start_kind": first["start_kind"], "start_reach_cm": round(float(first["reach_cm"]), 1),
            "net_direction_table_deg": round(net_table, 1),
            "net_direction_from_radial_deg": round(float(wrapped_deg(net_table - first["radial_deg"])), 1),
            "net_direction_in_first_picture_deg": round(float(wrapped_deg(net_table - first["picture_heading_deg"])), 1),
            "net_cm": round(float(numpy.hypot(net_x, net_y)), 2),
            "summed_cm": round(float(moving["step_cm"].sum()), 2),
            "rows_before_1_cm": rows_to_one_cm,
            "joint_4_start_deg": round(float(first["joint_4_deg"]), 1),
            "joint_4_change_first_steps_deg": round(float(moving["joint_4_deg"].iloc[-1] + moving["forearm_change_deg"].iloc[-1] - first["joint_4_deg"]), 1),
            "joint_4_change_first_3_cm_deg": round(float(group["joint_4_deg"].iloc[rows_to_three_cm] - first["joint_4_deg"]), 1),
            "largest_step_from_forward_deg": round(float(moving["step_picture_deg"].abs().max()), 1)
        })
    return pandas.DataFrame(rows)



def draw_figure(
        frames: pandas.DataFrame,
        moving: pandas.DataFrame,
        cells: pandas.DataFrame,
        first_steps: pandas.DataFrame,
        runs: Dict[int, pandas.DataFrame],
        parameters: Dict
) -> None:
    """Draw the six coverage panels."""

    sheets = sorted(frames["sheet_number"].unique())
    colours = pyplot.get_cmap("tab10")(numpy.arange(len(sheets)))
    figure = pyplot.figure(figsize = (19, 12))
    bin_deg = parameters["direction_bin_deg"]
    centres = numpy.arange(-180.0 + bin_deg, 180.0 + 1e-9, bin_deg)

    for position, (column, title) in enumerate([("step_from_radial_deg", "Travel direction on the table, from radial\n(0 = away from the base, 180 = back to it)"),
                                                 ("step_picture_deg", "Travel direction in the picture\n(0 = towards the picture's top)")]):
        axis = figure.add_subplot(2, 3, 1 + 2 * position, projection = "polar")
        axis.set_theta_zero_location("N")
        bottom = numpy.zeros(len(centres))
        binned = direction_bin(moving[column], bin_deg)
        for sheet, colour in zip(sheets, colours):
            counts = numpy.array([((binned == centre) & (moving["sheet_number"] == sheet).to_numpy()).sum() for centre in centres])
            axis.bar(numpy.radians(centres), counts, width = numpy.radians(bin_deg) * 0.95, bottom = bottom, color = colour, label = f"sheet {sheet}")
            bottom += counts
        axis.set_title(title, fontsize = 10)
        if position == 0:
            axis.legend(fontsize = 7, loc = "lower left", bbox_to_anchor = (-0.25, -0.1))

    axis = figure.add_subplot(2, 3, 2)
    edges = numpy.arange(-120.0, 120.1, 10.0)
    axis.hist([frames.loc[frames["sheet_number"] == sheet, "joint_4_deg"] for sheet in sheets], bins = edges, stacked = True, color = colours, label = sheets)
    for run_number, steps in runs.items():
        axis.plot(steps["joint_4_deg"], numpy.full(len(steps), 1400 + 45 * run_number), ".", markersize = 3, color = "black")
        axis.text(-128, 1400 + 45 * run_number, f"run {run_number}", fontsize = 6, va = "center", ha = "right")
    axis.set_xlim(-150, 125)
    axis.set_xlabel("forearm twist, joint 4 (degrees); black dots: the robot runs' pictures")
    axis.set_ylabel("pictures")
    axis.set_title("Forearm twist in the demonstrations (stacked by sheet)", fontsize = 10)

    axis = figure.add_subplot(2, 3, 4)
    reach_labels = [f"{low:g} to {high:g}" for low, high in zip(parameters["reach_edges_cm"][:-1], parameters["reach_edges_cm"][1:])]
    grid = cells.pivot(index = "reach_cm", columns = "from_radial_deg", values = "pictures").reindex(index = reach_labels, columns = centres).fillna(0)
    image = axis.imshow(grid.to_numpy(), aspect = "auto", cmap = "viridis", origin = "lower")
    layouts = cells.pivot(index = "reach_cm", columns = "from_radial_deg", values = "layouts").reindex(index = reach_labels, columns = centres).fillna(0)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            axis.text(column, row, f"{int(grid.iloc[row, column])}\n({int(layouts.iloc[row, column])})", ha = "center", va = "center",
                      fontsize = 6.5, color = "white" if grid.iloc[row, column] < grid.to_numpy().max() * 0.6 else "black")
    axis.set_xticks(range(len(centres)))
    axis.set_xticklabels([f"{centre:.0f}" for centre in centres], fontsize = 7)
    axis.set_yticks(range(len(reach_labels)))
    axis.set_yticklabels(reach_labels, fontsize = 7)
    axis.set_xlabel("travel direction from radial (degrees, bins of 30)")
    axis.set_ylabel("tip reach from the base axis (cm)")
    axis.set_title("Pictures per cell (lay-outs in brackets), moving steps", fontsize = 10)
    figure.colorbar(image, ax = axis, fraction = 0.03)

    axis = figure.add_subplot(2, 3, 5)
    axis.hist2d(moving["joint_4_deg"], moving["step_from_radial_deg"], bins = [numpy.arange(-120, 121, 10), numpy.arange(-180, 181, 15)], cmap = "viridis", cmin = 1)
    for run_number, steps in runs.items():
        axis.plot(steps["joint_4_deg"], steps["step_from_radial_deg"], ".", markersize = 2.5, color = "red" if run_number in (3, 5, 6, 8, 9, 10) else "white")
    axis.set_xlabel("forearm twist, joint 4 (degrees)")
    axis.set_ylabel("travel direction from radial (degrees)")
    axis.set_title("Demonstrated steps (colour) and the robot runs' steps\n(white: runs that reached the end; red: runs that failed)", fontsize = 10)

    axis = figure.add_subplot(2, 3, 6)
    marked = first_steps[~first_steps["start_kind"].isin(["NE", "MID", "X1", "X2"])]
    taught = first_steps[first_steps["start_kind"].isin(["NE", "MID", "X1", "X2"])]
    axis.hist([marked["net_direction_from_radial_deg"], taught["net_direction_from_radial_deg"]], bins = numpy.arange(-180, 181, 15), stacked = True,
              label = [f"start mark ({len(marked)} episodes)", f"taught start: NE, MID, X ({len(taught)})"], color = ["tab:blue", "tab:orange"])
    axis.set_xlabel("net direction of the first five moving steps, from radial (degrees)")
    axis.set_ylabel("episodes")
    axis.set_title("How the demonstrations start", fontsize = 10)
    axis.legend(fontsize = 8)

    figure.suptitle("Orientation coverage of the 209 demonstrations (audit-cleaned, 18,032 pictures)", fontsize = 12)
    figure.tight_layout()
    figure.savefig(parameters["figure_path"], dpi = 80)
    pyplot.close(figure)



def main(parameters: Optional[Dict] = None) -> None:
    """Print the coverage tables and save the cell table, the first-steps table and the figure."""

    settings = orientation_common.merged_parameters(COVERAGE_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 500)
    pandas.set_option("display.max_columns", 40)
    frames = demonstration_frames(settings)
    layouts = pandas.read_csv(EVIDENCE_FOLDER / "01_placements_per_episode.csv")
    frames["layout"] = frames["episode_id"].map(layouts.set_index("episode_id")["layout"])
    bin_deg = settings["direction_bin_deg"]
    is_moving = (frames["action"] != "stop") & (frames["step_cm"] >= settings["smallest_step_cm"])
    moving = frames[is_moving].copy()
    print(f"Pictures {len(frames)}, episodes {frames['episode_id'].nunique()}; moving steps of {settings['smallest_step_cm']} cm or more: "
          f"{len(moving)} ({100.0 * len(moving) / len(frames):.1f} %); rows with a three-step path direction: {int(frames['path_table_deg'].notna().sum())}")

    print("\n=== 1. Direction of travel on the table, from radial: share of moving steps (%) per sheet ===")
    moving["from_radial_bin"] = direction_bin(moving["step_from_radial_deg"], bin_deg)
    print(share_table(moving, "from_radial_bin").to_string())
    path_rows = frames[frames["path_table_deg"].notna()].copy()
    path_rows["path_bin"] = direction_bin(path_rows["path_from_radial_deg"], bin_deg)
    print("\nThe same from the path over the next three steps (less hand jitter), share of rows (%):")
    print(share_table(path_rows, "path_bin").to_string())
    absolute = moving["step_from_radial_deg"].abs()
    absolute_path = path_rows["path_from_radial_deg"].abs()
    for name, values in [("labelled steps", absolute), ("three-step paths", absolute_path)]:
        print(f"  {name}: within 45 degrees of radial-outwards {100.0 * (values <= 45).mean():.1f} %, 45 to 135 (sideways) "
              f"{100.0 * ((values > 45) & (values <= 135)).mean():.1f} %, beyond 135 (back towards the base) {100.0 * (values > 135).mean():.1f} %")
    print("  left against right of radial (three-step paths, sideways part only): "
          f"left {int(((path_rows['path_from_radial_deg'] > 45) & (path_rows['path_from_radial_deg'] <= 135)).sum())}, "
          f"right {int(((path_rows['path_from_radial_deg'] < -45) & (path_rows['path_from_radial_deg'] >= -135)).sum())}")

    print("\n=== 2. Forearm twist (joint 4, degrees): share of pictures (%) per sheet ===")
    frames["joint_4_bin"] = pandas.cut(frames["joint_4_deg"], settings["joint_4_edges_deg"], include_lowest = True).astype(str)
    order = [str(value) for value in pandas.cut(pandas.Series(settings["joint_4_edges_deg"][:-1]) + 1.0, settings["joint_4_edges_deg"], include_lowest = True)]
    print(share_table(frames, "joint_4_bin").reindex(order).to_string())
    print(f"  joint 4: min {frames['joint_4_deg'].min():.1f}, 5th percentile {frames['joint_4_deg'].quantile(0.05):.1f}, median "
          f"{frames['joint_4_deg'].median():.1f}, 95th {frames['joint_4_deg'].quantile(0.95):.1f}, max {frames['joint_4_deg'].max():.1f}; "
          f"|joint 4| median {frames['joint_4_deg'].abs().median():.1f}; beyond 60: {100.0 * (frames['joint_4_deg'].abs() > 60).mean():.1f} %, "
          f"beyond 90: {100.0 * (frames['joint_4_deg'].abs() > 90).mean():.1f} %, beyond 110: {100.0 * (frames['joint_4_deg'].abs() > 110).mean():.1f} %")
    print("  How far the picture's top is turned from radial (heading_from_radial_deg) against joint 4 (measured):")
    print(f"    least-squares slope {numpy.polyfit(frames['joint_4_deg'], frames['heading_from_radial_deg'], 1)[0]:+.3f} degrees per degree; "
          f"range {frames['heading_from_radial_deg'].min():.1f} to {frames['heading_from_radial_deg'].max():.1f}")
    for low, high in [(100, 121), (80, 100), (50, 80)]:
        for sign in (1, -1):
            part = frames[(sign * frames["joint_4_deg"] >= low) & (sign * frames["joint_4_deg"] < high)]
            if len(part):
                print(f"    joint 4 {sign * low:+d} to {sign * high:+d}: picture top {part['heading_from_radial_deg'].median():+.1f} from radial "
                      f"(5th to 95th percentile {part['heading_from_radial_deg'].quantile(0.05):+.1f} to {part['heading_from_radial_deg'].quantile(0.95):+.1f}), {len(part)} pictures")

    print("\n=== 3. Direction of travel in the picture (0 = to its top, + = to its left): share of moving steps (%) per sheet ===")
    moving["picture_bin"] = direction_bin(moving["step_picture_deg"], bin_deg)
    print(share_table(moving, "picture_bin").to_string())
    path_rows["path_picture_bin"] = direction_bin(path_rows["path_picture_deg"], bin_deg)
    print("\nThe same from the three-step path, share of rows (%):")
    print(share_table(path_rows, "path_picture_bin").to_string())
    picture_absolute = path_rows["path_picture_deg"].abs()
    print(f"  three-step path within 30 degrees of the picture's top {100.0 * (picture_absolute <= 30).mean():.1f} %, 30 to 60 "
          f"{100.0 * ((picture_absolute > 30) & (picture_absolute <= 60)).mean():.1f} %, 60 to 120 (sideways in the picture) "
          f"{100.0 * ((picture_absolute > 60) & (picture_absolute <= 120)).mean():.1f} %, beyond 120 (towards the picture's bottom, under the arm) "
          f"{100.0 * (picture_absolute > 120).mean():.1f} %")

    print("\n=== 4. Gaps: travel direction from radial against reach (moving steps) ===")
    moving["reach_bin"] = pandas.cut(moving["reach_cm"], settings["reach_edges_cm"], include_lowest = True,
                                     labels = [f"{low:g} to {high:g}" for low, high in zip(settings["reach_edges_cm"][:-1], settings["reach_edges_cm"][1:])]).astype(str)
    cells = moving.groupby(["reach_bin", "from_radial_bin"]).agg(
        pictures = ("frame_index", "size"), episodes = ("episode_id", "nunique"), layouts = ("layout", "nunique"), sheets = ("sheet_number", "nunique"),
        looking_along_share = ("step_picture_deg", lambda values: float((values.abs() <= 30).mean()))).reset_index()
    cells = cells.rename(columns = {"reach_bin": "reach_cm", "from_radial_bin": "from_radial_deg"})
    centres = numpy.arange(-180.0 + bin_deg, 180.0 + 1e-9, bin_deg)
    reach_labels = [f"{low:g} to {high:g}" for low, high in zip(settings["reach_edges_cm"][:-1], settings["reach_edges_cm"][1:])]
    full = pandas.MultiIndex.from_product([reach_labels, centres], names = ["reach_cm", "from_radial_deg"])
    cells = cells.set_index(["reach_cm", "from_radial_deg"]).reindex(full).fillna(0).reset_index()
    cells[["pictures", "episodes", "layouts", "sheets"]] = cells[["pictures", "episodes", "layouts", "sheets"]].astype(int)
    cells["looking_along_share"] = cells["looking_along_share"].round(2)
    cells["is_thin"] = (cells["pictures"] < settings["thin_cell_pictures"]) | (cells["layouts"] < settings["thin_cell_layouts"])
    cells.to_csv(EVIDENCE_FOLDER / "02_coverage_cells.csv", index = False)
    for value, title in [("pictures", "pictures"), ("layouts", "distinct lay-outs (from 01_placements)"), ("sheets", "sheets"),
                         ("looking_along_share", "share of the steps taken within 30 degrees of the picture's top")]:
        print(f"\n{title} per cell (rows: reach in cm; columns: travel direction from radial):")
        print(cells.pivot(index = "reach_cm", columns = "from_radial_deg", values = value).reindex(reach_labels).to_string())
    thin = cells[cells["is_thin"]]
    print(f"\nThin cells (under {settings['thin_cell_pictures']} pictures or under {settings['thin_cell_layouts']} lay-outs): {len(thin)} of {len(cells)}")
    print(thin[["reach_cm", "from_radial_deg", "pictures", "episodes", "layouts", "sheets"]].to_string(index = False))
    print(f"Share of all moving steps that lie in thin cells: {100.0 * thin['pictures'].sum() / cells['pictures'].sum():.1f} %")

    print("\n=== 5. Travel direction from radial against joint 4 (moving steps, pictures) ===")
    moving["joint_4_bin"] = pandas.cut(moving["joint_4_deg"], settings["joint_4_edges_deg"], include_lowest = True).astype(str)
    print(pandas.crosstab(moving["joint_4_bin"], moving["from_radial_bin"]).reindex(order).to_string())
    print("\nTravel direction in the picture against joint 4 (three-step paths, rows):")
    path_rows["joint_4_bin"] = pandas.cut(path_rows["joint_4_deg"], settings["joint_4_edges_deg"], include_lowest = True).astype(str)
    print(pandas.crosstab(path_rows["joint_4_bin"], path_rows["path_picture_bin"]).reindex(order).to_string())

    print("\n=== 6. The first five moving steps of every episode ===")
    first_steps = first_steps_table(frames, settings)
    first_steps.to_csv(EVIDENCE_FOLDER / "02_first_five_steps.csv", index = False)
    first_steps["start"] = numpy.where(first_steps["start_kind"].isin(["NE", "MID", "X1", "X2"]), "taught (NE, MID, X)", "start mark")
    for name, part in first_steps.groupby("start"):
        from_radial = part["net_direction_from_radial_deg"]
        in_picture = part["net_direction_in_first_picture_deg"]
        print(f"\n{name}: {len(part)} episodes; start reach {part['start_reach_cm'].min():.1f} to {part['start_reach_cm'].max():.1f} cm "
              f"(median {part['start_reach_cm'].median():.1f}); joint 4 at the start {part['joint_4_start_deg'].min():.1f} to {part['joint_4_start_deg'].max():.1f} "
              f"(|joint 4| over 10 degrees in {int((part['joint_4_start_deg'].abs() > 10).sum())} episodes)")
        print(f"  net direction of the first five moving steps, from radial: mean {circular_mean_deg(from_radial):+.1f}; within 20 degrees of radial "
              f"{int((from_radial.abs() <= 20).sum())}, 20 to 45: {int(((from_radial.abs() > 20) & (from_radial.abs() <= 45)).sum())}, "
              f"45 to 90: {int(((from_radial.abs() > 45) & (from_radial.abs() <= 90)).sum())}, over 90: {int((from_radial.abs() > 90).sum())}")
        print(f"  the same in the first picture's frame: within 20 degrees of its top {int((in_picture.abs() <= 20).sum())}, 20 to 45: "
              f"{int(((in_picture.abs() > 20) & (in_picture.abs() <= 45)).sum())}, 45 to 90: {int(((in_picture.abs() > 45) & (in_picture.abs() <= 90)).sum())}, "
              f"over 90: {int((in_picture.abs() > 90).sum())}")
        print(f"  net travel of the five steps: median {part['net_cm'].median():.2f} cm of {part['summed_cm'].median():.2f} cm stepped; "
              f"forearm twist over the first 3 cm of travel: median |change| {part['joint_4_change_first_3_cm_deg'].abs().median():.1f} degrees, "
              f"over 20 degrees in {int((part['joint_4_change_first_3_cm_deg'].abs() > 20).sum())} episodes")
    print("\nStart-mark episodes by start kind (net direction of the first five moving steps from radial, degrees):")
    marked = first_steps[first_steps["start"] == "start mark"]
    print(marked.groupby("start_kind")["net_direction_from_radial_deg"].agg(["count", "min", "median", "max"]).round(1).to_string())
    print("\nStart-mark episodes whose first five steps go more than 45 degrees off radial, or that twist more than 20 degrees in the first 3 cm:")
    odd = marked[(marked["net_direction_from_radial_deg"].abs() > 45) | (marked["joint_4_change_first_3_cm_deg"].abs() > 20)]
    print(odd[["plan", "end", "start_kind", "net_direction_from_radial_deg", "net_cm", "joint_4_change_first_3_cm_deg"]].to_string(index = False))
    print("\nTaught starts: direction from radial and forearm twist at the start, by sheet and kind:")
    taught = first_steps[first_steps["start"] != "start mark"]
    print(taught.groupby(["sheet", "end", "start_kind"]).agg(
        episodes = ("plan", "size"), reach_cm = ("start_reach_cm", "median"), from_radial_deg = ("net_direction_from_radial_deg", circular_mean_deg),
        joint_4_start_deg = ("joint_4_start_deg", "median")).round(1).to_string())

    print("\n=== 7. The ten robot runs of 11.11.26 in the same terms ===")
    cell_pictures = cells.set_index(["reach_cm", "from_radial_deg"])["pictures"]
    cell_thin = cells.set_index(["reach_cm", "from_radial_deg"])["is_thin"]
    runs = {}
    summary = []
    for run in settings["robot_runs"]:
        steps = robot_run_steps(run, settings)
        steps = steps[steps["action"] == "model_step"].copy() if "action" in steps else steps
        runs[run["number"]] = steps
        reach_bin = pandas.cut(steps["reach_cm"], settings["reach_edges_cm"], include_lowest = True, labels = reach_labels).astype(str)
        radial_bin = direction_bin(steps["step_from_radial_deg"], bin_deg)
        thin_share = numpy.mean([bool(cell_thin.get((reach, direction), True)) for reach, direction in zip(reach_bin, radial_bin)])
        fewest = min(int(cell_pictures.get((reach, direction), 0)) for reach, direction in zip(reach_bin, radial_bin))
        neighbours = numpy.array([
            int(((numpy.abs(moving["joint_4_deg"].to_numpy() - joint_4) <= settings["neighbour_joint_4_deg"])
                 & (numpy.abs(wrapped_deg(moving["step_from_radial_deg"].to_numpy() - direction)) <= settings["neighbour_direction_deg"])
                 & (numpy.abs(moving["reach_cm"].to_numpy() - reach) <= settings["neighbour_reach_cm"])).sum())
            for joint_4, direction, reach in zip(steps["joint_4_deg"], steps["step_from_radial_deg"], steps["reach_cm"])])
        is_few = neighbours < settings["few_neighbours"]
        steps["neighbours"] = neighbours
        summary.append({
            "run": run["number"], "sheet": run["sheet"], "model": run["model"], "outcome": run["outcome"], "steps": len(steps),
            "neighbours_median": int(numpy.median(neighbours)), "steps_with_few_neighbours": int(is_few.sum()),
            "first_step_with_few_neighbours": int(numpy.argmax(is_few)) if is_few.any() else -1,
            "reach_cm": f"{steps['reach_cm'].min():.0f} to {steps['reach_cm'].max():.0f}",
            "joint_4_deg": f"{steps['joint_4_deg'].min():.0f} to {steps['joint_4_deg'].max():.0f}",
            "from_radial_deg": f"{steps['step_from_radial_deg'].min():.0f} to {steps['step_from_radial_deg'].max():.0f}",
            "steps_beyond_90_from_radial": int((steps["step_from_radial_deg"].abs() > 90).sum()),
            "steps_over_60_off_picture_top": int((steps["step_picture_deg"].abs() > 60).sum()),
            "share_in_thin_cells": round(float(thin_share), 2), "fewest_pictures_in_a_visited_cell": fewest
        })
    print(pandas.DataFrame(summary).to_string(index = False))
    print(f"neighbours: demonstrated moving steps within {settings['neighbour_joint_4_deg']:.0f} degrees of forearm twist, {settings['neighbour_direction_deg']:.0f} degrees of "
          f"travel direction from radial and {settings['neighbour_reach_cm']:.0f} cm of reach; few = under {settings['few_neighbours']}. For scale, the demonstrations' own "
          "moving steps (each counted against all others):")
    sample = moving.iloc[::40]
    own = numpy.array([
        int(((numpy.abs(moving["joint_4_deg"].to_numpy() - joint_4) <= settings["neighbour_joint_4_deg"])
             & (numpy.abs(wrapped_deg(moving["step_from_radial_deg"].to_numpy() - direction)) <= settings["neighbour_direction_deg"])
             & (numpy.abs(moving["reach_cm"].to_numpy() - reach) <= settings["neighbour_reach_cm"])).sum()) - 1
        for joint_4, direction, reach in zip(sample["joint_4_deg"], sample["step_from_radial_deg"], sample["reach_cm"])])
    print(f"  median {int(numpy.median(own))} neighbours; under {settings['few_neighbours']}: {100.0 * (own < settings['few_neighbours']).mean():.1f} % of {len(own)} sampled steps")

    draw_figure(frames, moving, cells, first_steps, runs, settings)
    print(f"\nSaved {settings['figure_path'].name}, 02_coverage_cells.csv, 02_first_five_steps.csv")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding = "utf-8")
    main()
