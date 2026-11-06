"""
Draw the figures that show how well the fitted forearm camera maps the recorded pictures onto the table.

    frames_with_projections.jpg  six pictures with the fitted sheet outline, a 5 cm table grid, the taught start
                                 point and the magnet tip projected into them (initial guess in grey for comparison);
    top_view_<episode>.png       the table seen from above: the fitted sheet, the tip's path, the puck mapped from
                                 every picture, and the picture's outline on the table at a few steps;
    line_consistency_<day>.png   the black line's pixels of a whole day mapped onto the table with the initial guess
                                 and with the fitted camera (a sharp 1 cm band means the pictures agree);
    push_check.png               tip-to-puck distance right after pushes, against the block's contact distance;
    residuals.png                edge and still-puck errors, fitted and held out.

Run after calibrate_from_recordings.py:
    D:/AI_Project/.venv/Scripts/python.exe make_figures.py
"""

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy
from matplotlib import pyplot
from matplotlib import patches

import calibration_data
import sheet_edges
import static_runs
from forearm_camera_model import FOREARM_CAMERA_PARAMETERS, picture_footprint_on_plane, pixels_to_plane, project_points

OUTPUT_FOLDER = Path(__file__).resolve().parent

FIGURE_SETTINGS = {
    "results_path": str(OUTPUT_FOLDER / "pipeline_results.pkl"),
    "figure_folder": str(OUTPUT_FOLDER / "figures"),
    "overlay_frames": [["037", 0], ["042", 0], ["045", 90], ["044", 155], ["043", 140], ["041", 100]],
    "top_view_episodes": ["037", "045"],
    "footprint_frames_per_episode": 6,
    "line_days": {"2026-11-04": ["037", "038", "039", "041"], "2026-11-05": ["042", "043", "044", "045"]},
    "line_frame_step": 2,
    "line_relative_darkness": 0.55,
    "map_bin_m": 0.001,
    "grid_step_m": 0.05,
    # BGR, as OpenCV draws
    "colours": {"fitted": (60, 200, 60), "initial": (150, 150, 150), "start": (0, 140, 255), "tip": (0, 230, 255), "puck": (0, 0, 255)}
}



def sheet_corners(sheet: List[float], size_m: List[float]) -> numpy.ndarray:
    """
    Return the four corners of a sheet rectangle on the table (far left, far right, near right, near left).

    Parameters
    ----------
    sheet : list of float
        [centre x, centre y, angle].
    size_m : list of float
        Length (along the sheet's x) and width.

    Returns
    -------
    numpy.ndarray
        Corners (4, 2).
    """

    centre_x, centre_y, angle = sheet
    half_length, half_width = size_m[0] / 2.0, size_m[1] / 2.0
    local = numpy.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
    rotation = numpy.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return local @ rotation.T + numpy.array([centre_x, centre_y])



def table_points_3d(points_xy: numpy.ndarray, extra_height_m: float = 0.0) -> numpy.ndarray:
    """
    Return table points with their height on config's table plane (plus an extra height).

    Parameters
    ----------
    points_xy : numpy.ndarray
        Points (N, 2).
    extra_height_m : float, optional
        Added height.

    Returns
    -------
    numpy.ndarray
        Points (N, 3).
    """

    plane = calibration_data.CALIBRATION_TABLE_PLANE
    heights = plane["a"] * points_xy[:, 0] + plane["b"] * points_xy[:, 1] + plane["c"] + extra_height_m
    return numpy.column_stack([points_xy, heights])



def draw_polyline(
        picture: numpy.ndarray,
        pixels: numpy.ndarray,
        colour: tuple,
        thickness: int = 1
) -> None:
    """
    Draw the visible, finite stretches of a projected polyline.

    Parameters
    ----------
    picture : numpy.ndarray
        The picture, drawn on in place.
    pixels : numpy.ndarray
        Projected points (N, 2), NaN allowed.
    colour : tuple
        BGR colour.
    thickness : int, optional
        Line thickness.
    """

    for first, second in zip(pixels[:-1], pixels[1:]):
        if numpy.all(numpy.isfinite(first)) and numpy.all(numpy.isfinite(second)):
            if max(abs(first[0]), abs(second[0])) < 4_000 and max(abs(first[1]), abs(second[1])) < 4_000:
                cv2.line(picture, (int(round(first[0])), int(round(first[1]))), (int(round(second[0])), int(round(second[1]))), colour, thickness, cv2.LINE_AA)



def densified(polygon: numpy.ndarray, points_per_side: int = 60, is_closed: bool = True) -> numpy.ndarray:
    """
    Return a polygon with extra points along its sides, so that its projection bends with the lens.

    Parameters
    ----------
    polygon : numpy.ndarray
        Corners (N, 2).
    points_per_side : int, optional
        Points per side.
    is_closed : bool, optional
        Whether the last corner joins the first.

    Returns
    -------
    numpy.ndarray
        The densified points.
    """

    corners = numpy.vstack([polygon, polygon[:1]]) if is_closed else polygon
    pieces = [corners[index] + numpy.linspace(0.0, 1.0, points_per_side)[:, None] * (corners[index + 1] - corners[index])
              for index in range(len(corners) - 1)]
    return numpy.vstack(pieces)



def draw_overlay_frames(
        records_by_key: Dict,
        results: Dict,
        settings: Dict
) -> str:
    """
    Draw the fitted sheet outline, a 5 cm table grid, the taught start point and the magnet tip into six pictures.

    Parameters
    ----------
    records_by_key : dict
        (episode, frame index) -> frame record.
    results : dict
        The pipeline's results.
    settings : dict
        The figure settings.

    Returns
    -------
    str
        The figure's path.
    """

    delivered = results["delivered"]
    camera = delivered["camera"]
    height = delivered["puck_height_m"]
    colours = settings["colours"]
    start = numpy.array(results["settings"]["line_start_xy_m"])
    tiles = []
    for episode, frame_index in settings["overlay_frames"]:
        record = records_by_key.get((episode, frame_index))
        if record is None:
            continue
        picture = cv2.imread(record["picture_path"])
        corners = sheet_corners(delivered["sheets"][episode], delivered["sheet_size_m"])

        # a 5 cm grid on the table inside the sheet, then the sheet outline (fitted, and the initial guess in grey)
        low = corners.min(axis = 0)
        high = corners.max(axis = 0)
        step = settings["grid_step_m"]
        for x in numpy.arange(math.ceil(low[0] / step) * step, high[0], step):
            line = numpy.column_stack([numpy.full(80, x), numpy.linspace(low[1], high[1], 80)])
            draw_polyline(picture, project_points(table_points_3d(line), record["joints"], camera), (200, 200, 120))
        for y in numpy.arange(math.ceil(low[1] / step) * step, high[1], step):
            line = numpy.column_stack([numpy.linspace(low[0], high[0], 80), numpy.full(80, y)])
            draw_polyline(picture, project_points(table_points_3d(line), record["joints"], camera), (200, 200, 120))
        outline = densified(corners)
        draw_polyline(picture, project_points(table_points_3d(outline), record["joints"], FOREARM_CAMERA_PARAMETERS), colours["initial"], 1)
        draw_polyline(picture, project_points(table_points_3d(outline), record["joints"], camera), colours["fitted"], 2)

        # the taught start point on the table (circle) and at the detector's puck height (cross); initial guess in grey
        for parameters, colour in ((FOREARM_CAMERA_PARAMETERS, colours["initial"]), (camera, colours["start"])):
            on_table = project_points(table_points_3d(start[None, :]), record["joints"], parameters)[0]
            at_puck = project_points(table_points_3d(start[None, :], height), record["joints"], parameters)[0]
            if numpy.all(numpy.isfinite(on_table)):
                cv2.circle(picture, (int(round(on_table[0])), int(round(on_table[1]))), 7, colour, 2, cv2.LINE_AA)
            if numpy.all(numpy.isfinite(at_puck)):
                cv2.drawMarker(picture, (int(round(at_puck[0])), int(round(at_puck[1]))), colour, cv2.MARKER_CROSS, 14, 2)

        tip = project_points(record["tip_xyz"][None, :], record["joints"], camera)[0]
        if numpy.all(numpy.isfinite(tip)):
            cv2.drawMarker(picture, (int(round(tip[0])), int(round(tip[1]))), colours["tip"], cv2.MARKER_DIAMOND, 12, 2)
        text = f"{episode}/{frame_index:03d} forearm {record['forearm_deg']:+.0f} deg"
        detection = record["detection"]
        if detection is not None:
            cv2.drawMarker(picture, (int(round(detection["centre_u"])), int(round(detection["centre_v"]))), colours["puck"], cv2.MARKER_TILTED_CROSS, 12, 2)
            mapped = static_runs.estimate_puck_xy(record, camera, height)
            if mapped is not None:
                distance = 1000.0 * float(numpy.linalg.norm(mapped - record["tip_xyz"][:2]))
                text += f", puck ({100 * mapped[0]:.1f}, {100 * mapped[1]:.1f}) cm, {distance:.0f} mm from tip"
        cv2.rectangle(picture, (0, 0), (640, 24), (0, 0, 0), -1)
        cv2.putText(picture, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(picture)

    while len(tiles) % 3:
        tiles.append(numpy.zeros_like(tiles[0]))
    rows = [numpy.hstack(tiles[index:index + 3]) for index in range(0, len(tiles), 3)]
    legend = numpy.full((40, rows[0].shape[1], 3), 255, dtype = numpy.uint8)
    cv2.putText(legend, "green: fitted sheet outline (grid 5 cm)   grey: initial guess   orange circle / cross: taught start (0.1707, 0.0374) "
                        "on the table / at the puck point's height   yellow diamond: magnet tip   red x: detected puck", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    figure = numpy.vstack(rows + [legend])
    path = Path(settings["figure_folder"]) / "frames_with_projections.jpg"
    cv2.imwrite(str(path), figure, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return str(path)



def draw_top_view(
        episode: str,
        records: List[Dict],
        results: Dict,
        settings: Dict
) -> str:
    """
    Draw one episode from above: the fitted sheet, the tip's path, the mapped puck, and picture outlines.

    Parameters
    ----------
    episode : str
        The episode.
    records : list of dict
        Frame records.
    results : dict
        The pipeline's results.
    settings : dict
        The figure settings.

    Returns
    -------
    str
        The figure's path.
    """

    delivered = results["delivered"]
    camera = delivered["camera"]
    height = delivered["puck_height_m"]
    episode_records = calibration_data.records_by_episode(records)[episode]
    bins = settings["map_bin_m"]
    x_edges = numpy.arange(-0.02, 0.56 + bins, bins)
    y_edges = numpy.arange(-0.25, 0.25 + bins, bins)
    figure, axes = pyplot.subplots(1, 2, figsize = (17, 7.6), constrained_layout = True)
    for axis, (title, parameters, point_height) in zip(axes, (("initial guess", FOREARM_CAMERA_PARAMETERS, 0.0145), ("fitted camera", camera, height))):
        # the black line of this episode's pictures, mapped with the same camera, as a grey background
        counts = numpy.zeros((len(x_edges) - 1, len(y_edges) - 1))
        for record in episode_records[::3]:
            pixels = find_line_pixels(cv2.imread(record["picture_path"]), settings["line_relative_darkness"])
            if len(pixels) == 0:
                continue
            mapped = pixels_to_plane(pixels, record["joints"], calibration_data.table_height(0.3, 0.0), parameters)
            mapped = mapped[numpy.all(numpy.isfinite(mapped), axis = 1)]
            histogram, _, _ = numpy.histogram2d(mapped[:, 0], mapped[:, 1], bins = (x_edges, y_edges))
            counts += numpy.minimum(histogram, 1.0)
        if (counts > 0).any():
            axis.imshow(counts.T, origin = "lower", extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), cmap = "Greys",
                        vmax = max(2.0, numpy.percentile(counts[counts > 0], 90)), alpha = 0.9)
        corners = sheet_corners(delivered["sheets"][episode], delivered["sheet_size_m"])
        axis.add_patch(patches.Polygon(corners, closed = True, fill = False, edgecolor = "0.3", linewidth = 1.5, label = "sheet (fitted edges)"))
        tips = numpy.array([record["tip_xyz"][:2] for record in episode_records])
        axis.plot(tips[:, 0], tips[:, 1], "-", color = "0.55", linewidth = 1.0, label = "magnet tip path")
        mapped, order = [], []
        for record in episode_records:
            if static_runs.is_usable_detection(record["detection"], static_runs.STATIC_RUN_PARAMETERS):
                point = static_runs.estimate_puck_xy(record, parameters, point_height)
                if point is not None:
                    mapped.append(point)
                    order.append(record["frame_index"])
        mapped = numpy.array(mapped)
        scatter = axis.scatter(mapped[:, 0], mapped[:, 1], c = order, cmap = "viridis", s = 9, label = "puck, mapped from each picture")
        step = max(1, len(episode_records) // settings["footprint_frames_per_episode"])
        for index, record in enumerate(episode_records[::step]):
            outline = picture_footprint_on_plane(record["joints"], calibration_data.table_height(0.3, 0.0), parameters, {"points_per_edge": 16})
            axis.add_patch(patches.Polygon(outline, closed = True, fill = False, edgecolor = "tab:orange", linewidth = 0.8, alpha = 0.7,
                                           label = "picture outline on the table" if index == 0 else None))
        axis.plot(*results["settings"]["line_start_xy_m"], marker = "*", color = "tab:red", markersize = 13, linestyle = "none", label = "taught start (0.1707, 0.0374)")
        axis.plot(0.0, 0.0, marker = "s", color = "k", markersize = 8, linestyle = "none", label = "robot base")
        axis.set_xlim(-0.02, 0.56)
        axis.set_ylim(-0.25, 0.25)
        axis.set_aspect("equal")
        axis.grid(True, alpha = 0.3)
        axis.set_xlabel("x (m, robot base frame)")
        axis.set_ylabel("y (m)")
        axis.set_title(f"episode {episode}, {title}")
        axis.legend(loc = "upper left", fontsize = 8)
    figure.suptitle("grey: the black line of every third picture mapped onto the table; dots: the puck mapped from every picture; "
                    "orange: the picture's outline on the table at a few steps")
    figure.colorbar(scatter, ax = axes, shrink = 0.8, label = "frame")
    path = Path(settings["figure_folder"]) / f"top_view_{episode}.png"
    figure.savefig(path, dpi = 85)
    pyplot.close(figure)
    return str(path)



def find_line_pixels(picture: numpy.ndarray, relative_darkness: float) -> numpy.ndarray:
    """
    Return the black line's pixels: dark pixels inside the sheet, away from the robot and the puck.

    Parameters
    ----------
    picture : numpy.ndarray
        The picture.
    relative_darkness : float
        A pixel is line when its smallest colour channel is below this share of the paper's.

    Returns
    -------
    numpy.ndarray
        Pixels (N, 2), every second row and column.
    """

    result = sheet_edges.find_sheet_edge_points(picture, {"is_refined": False})
    if result["sheet_mask"] is None:
        return numpy.zeros((0, 2))
    mask = cv2.resize(result["sheet_mask"], (picture.shape[1], picture.shape[0]), interpolation = cv2.INTER_NEAREST) > 0
    mask = cv2.erode(mask.astype(numpy.uint8), numpy.ones((9, 9), numpy.uint8)) > 0
    smallest = picture.min(axis = 2).astype(float)
    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    paper_level = float(numpy.percentile(smallest[mask], 75)) if mask.any() else 255.0
    is_line = mask & (smallest < relative_darkness * paper_level) & (hsv[:, :, 1] < 120)
    rows, columns = numpy.nonzero(is_line[::2, ::2])
    return numpy.column_stack([2 * columns, 2 * rows]).astype(float)



def draw_line_consistency(
        records: List[Dict],
        results: Dict,
        settings: Dict
) -> List[str]:
    """
    Map the black line's pixels of every picture of a day onto the table, with the initial guess and with the fitted
    camera, and draw how often each square millimetre was hit.

    Parameters
    ----------
    records : list of dict
        Frame records.
    results : dict
        The pipeline's results.
    settings : dict
        The figure settings.

    Returns
    -------
    list of str
        The figures' paths.
    """

    delivered = results["delivered"]
    paths = []
    by_episode = calibration_data.records_by_episode(records)
    bins = settings["map_bin_m"]
    x_edges = numpy.arange(0.10, 0.50 + bins, bins)
    y_edges = numpy.arange(-0.16, 0.14 + bins, bins)
    for day, episodes in settings["line_days"].items():
        maps = {}
        for name, parameters in (("initial guess", FOREARM_CAMERA_PARAMETERS), ("fitted camera", delivered["camera"])):
            counts = numpy.zeros((len(x_edges) - 1, len(y_edges) - 1))
            for episode in episodes:
                for record in by_episode.get(episode, [])[::settings["line_frame_step"]]:
                    picture = cv2.imread(record["picture_path"])
                    pixels = find_line_pixels(picture, settings["line_relative_darkness"])
                    if len(pixels) == 0:
                        continue
                    mapped = pixels_to_plane(pixels, record["joints"], calibration_data.table_height(0.3, 0.0), parameters)
                    mapped = mapped[numpy.all(numpy.isfinite(mapped), axis = 1)]
                    histogram, _, _ = numpy.histogram2d(mapped[:, 0], mapped[:, 1], bins = (x_edges, y_edges))
                    counts += numpy.minimum(histogram, 1.0)
            maps[name] = counts
        figure, axes = pyplot.subplots(1, 2, figsize = (16, 7.5))
        for axis, (name, counts) in zip(axes, maps.items()):
            axis.imshow(counts.T, origin = "lower", extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]), cmap = "magma",
                        vmax = numpy.percentile(counts[counts > 0], 99) if (counts > 0).any() else 1)
            for episode in episodes:
                if episode in delivered["sheets"]:
                    corners = sheet_corners(delivered["sheets"][episode], delivered["sheet_size_m"])
                    axis.add_patch(patches.Polygon(corners, closed = True, fill = False, edgecolor = "cyan", linewidth = 0.6))
            axis.set_title(f"{day}: black line from every 2nd picture, {name}\n(colour: pictures that saw each 1 mm square)")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
        path = Path(settings["figure_folder"]) / f"line_consistency_{day}.png"
        figure.tight_layout()
        figure.savefig(path, dpi = 80)
        pyplot.close(figure)
        paths.append(str(path))
    return paths



def draw_push_check(results: Dict, settings: Dict) -> str:
    """
    Draw the distance from the magnet tip to the mapped puck right after pushes.

    Parameters
    ----------
    results : dict
        The pipeline's results.
    settings : dict
        The figure settings.

    Returns
    -------
    str
        The figure's path.
    """

    figure, axes = pyplot.subplots(1, 2, figsize = (15, 5.5))
    bins = numpy.arange(0, 70, 1.5)
    low, high = results["settings"]["expected_contact_distance_m"]
    axis = axes[0]
    for key, label, colour in (("push_check_initial_guess", "initial guess", "0.6"), ("push_check", "fitted camera", "tab:green")):
        values = 1000.0 * results[key]["distance_m"]
        axis.hist(values, bins = bins, alpha = 0.6, color = colour, label = f"{label}: median {numpy.median(values):.1f} mm (n = {len(values)})")
    axis.axvspan(1000 * low, 1000 * high, color = "tab:blue", alpha = 0.15, label = "contact: cover radius 10 to 13 mm + puck radius 12 mm")
    axis.axvline(18.5, color = "tab:red", linestyle = "--", label = "config puck_centre_ahead_of_tip_m (18.5 mm)")
    axis.set_xlabel("distance from magnet tip to mapped puck centre right after a push (mm)")
    axis.set_ylabel("pictures")
    axis.legend(fontsize = 8)
    axis = axes[1]
    ahead = 1000.0 * results["push_check"]["ahead_m"]
    sideways = 1000.0 * results["push_check"]["sideways_m"]
    axis.scatter(sideways, ahead, s = 8, alpha = 0.6, color = "tab:green")
    axis.add_patch(patches.Circle((0, 0), 12.5, fill = False, color = "k", linewidth = 1.0))
    axis.text(0, -16, "cover (26 mm)", ha = "center", fontsize = 8)
    axis.set_xlabel("sideways, left of the push (mm)")
    axis.set_ylabel("ahead of the tip along the push (mm)")
    axis.set_aspect("equal")
    axis.set_xlim(-45, 45)
    axis.set_ylim(-20, 70)
    axis.grid(True, alpha = 0.3)
    axis.set_title("mapped puck centre relative to the magnet tip, fitted camera")
    path = Path(settings["figure_folder"]) / "push_check.png"
    figure.tight_layout()
    figure.savefig(path, dpi = 80)
    pyplot.close(figure)
    return str(path)



def draw_residuals(results: Dict, settings: Dict) -> str:
    """
    Draw the edge and still-puck errors, fitted and held out.

    Parameters
    ----------
    results : dict
        The pipeline's results.
    settings : dict
        The figure settings.

    Returns
    -------
    str
        The figure's path.
    """

    figure, axes = pyplot.subplots(1, 3, figsize = (18, 5))
    held_out = results["held_out"]
    axis = axes[0]
    bins = numpy.arange(0, 20, 0.5)
    axis.hist(results["held_out"]["edges_fitted_frames_same_gate"]["pixel_errors_px"], bins = bins, alpha = 0.5, density = True, label = "fitted pictures")
    axis.hist(held_out["edges"]["pixel_errors_px"], bins = bins, alpha = 0.5, density = True, label = "held-out pictures")
    axis.set_xlabel("sheet edge pixel: distance from the fitted side (px)")
    axis.legend()
    axis = axes[1]
    bins = numpy.arange(0, 20, 0.5)
    axis.hist(1000 * held_out["edges_fitted_frames_same_gate"]["table_errors_m"], bins = bins, alpha = 0.5, density = True, label = "fitted pictures")
    axis.hist(1000 * held_out["edges"]["table_errors_m"], bins = bins, alpha = 0.5, density = True, label = "held-out pictures")
    axis.set_xlabel("sheet edge pixel mapped to the table: distance from the side (mm)")
    axis.legend()
    axis = axes[2]
    bins = numpy.arange(0, 30, 1.0)
    axis.hist(held_out["puck_initial_guess"]["leave_one_out_px"], bins = bins, alpha = 0.4, color = "0.5", label = "held-out runs, initial guess")
    axis.hist(held_out["puck_fitted_runs"]["leave_one_out_px"], bins = bins, alpha = 0.5, label = "fitted runs")
    axis.hist(held_out["puck"]["leave_one_out_px"], bins = bins, alpha = 0.5, label = "held-out runs")
    axis.set_xlabel("still puck, one picture predicted from the run's others (px)")
    axis.legend()
    path = Path(settings["figure_folder"]) / "residuals.png"
    figure.tight_layout()
    figure.savefig(path, dpi = 80)
    pyplot.close(figure)
    return str(path)



def main(settings: Optional[Dict] = None) -> None:

    figure_settings = dict(FIGURE_SETTINGS)
    if settings:
        figure_settings.update(settings)
    Path(figure_settings["figure_folder"]).mkdir(parents = True, exist_ok = True)
    with open(figure_settings["results_path"], "rb") as handle:
        results = pickle.load(handle)
    records = calibration_data.load_frame_records()
    records_by_key = {(record["episode"], record["frame_index"]): record for record in records}
    paths = [draw_overlay_frames(records_by_key, results, figure_settings)]
    paths += [draw_top_view(episode, records, results, figure_settings) for episode in figure_settings["top_view_episodes"]]
    paths += [draw_push_check(results, figure_settings), draw_residuals(results, figure_settings)]
    paths += draw_line_consistency(records, results, figure_settings)
    print(json.dumps(paths, indent = 1))


if __name__ == "__main__":
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    main(overrides)
