"""
Fit the forearm camera to the recorded episodes 037 to 045 without the robot, and judge the fit.

Stages (each one's camera starts the next):
    1. still-puck runs found with the initial guess; a fit of the camera's pose to them, the lens kept at the guess;
    2. the sheet's outer edges in every picture; a joint fit of the camera (all ten quantities), each episode's
       sheet rectangle and one sheet size, the edges alone;
    3. still-puck runs found again with that camera, now leaving out twists while the block touches the puck;
    4. the final joint fit: edges and still-puck runs together, plus the height of the puck point the detector
       marks; fitted on 80 percent of the pictures (blocks of ten consecutive pictures held out at random) and judged
       on the rest; then fitted again on all pictures for the delivered parameters;
    5. checks that the fit never saw: the puck in front of the block while pushing, the puck at the start, the
       sheet's size against A4, the sheet's pose on the two days, and leave-one-episode-out refits for the spread.

Run from this folder (the robot is never contacted):
    D:/AI_Project/.venv/Scripts/python.exe calibrate_from_recordings.py
Results go to pipeline_results.pkl and pipeline_summary.json in this folder.
"""

import copy
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy

import calibration_data
import fit_camera
import joint_fit
import sheet_edges
import static_runs
from forearm_camera_model import FIT_PARAMETER_NAMES, FOREARM_CAMERA_PARAMETERS, describe_camera, pixels_to_plane, project_points

OUTPUT_FOLDER = Path(__file__).resolve().parent

PIPELINE_SETTINGS = {
    "edge_points_path": str(OUTPUT_FOLDER / "edge_points.pkl"),
    "results_path": str(OUTPUT_FOLDER / "pipeline_results.pkl"),
    "summary_path": str(OUTPUT_FOLDER / "pipeline_summary.json"),
    "held_out_share": 0.2,
    "held_out_block_frames": 10,
    "split_seed": 42_069,
    "stage_1_run_parameters": {"far_from_puck_m": 0.055},
    "stage_3_run_parameters": {"in_place_minimum_distance_m": 0.033, "far_from_puck_m": 0.045},
    "edges_only_joint_parameters": {"is_puck_height_free": False},
    # the detector's centre is the mean of the visible red pixels: the top face and the near side wall. For an upright
    # cylinder seen whole, that silhouette's centroid is the projection of the axis at mid-height (calc, parallel
    # projection), 14.5 mm; the recorded runs cannot pin it down better (their cost is flat from 0 to 20 mm), so it is
    # fixed there; a block hiding the near side raises it
    "final_joint_parameters": {"is_puck_height_free": False, "puck_point_height_m": 0.0145},
    # a held-out edge pixel is matched to a side within this distance on the table (wider than the fit's last gate,
    # so the held-out errors are not cut off at the gate)
    "evaluation_gate_m": 0.02,
    "is_leave_one_episode_out_run": True,
    "is_variants_run": True,
    # Niryo's calibration reduced to one radial term (the initial guess): fixed for the variant that tests it
    "niryo_lens_variant": {"focal_length_px": 496.0, "radial_distortion_k1": -0.2726},
    # the push check
    "push_minimum_step_m": 0.005,
    "push_maximum_angle_deg": 45.0,
    "push_minimum_puck_move_m": 0.003,
    # a puck further than this from the tip before the step cannot have been reached by a 1 cm step
    "push_maximum_distance_before_m": 0.06,
    # contact: the electromagnet's cover (Niryo ELECTROMAGNET_ned2.step, diameters 20.5 / 24.5 / 26 mm) plus the
    # puck's radius, 12 mm: 22 to 25 mm (calc); config's "puck_centre_ahead_of_tip_m" says 18.5 mm (13 mm tip)
    "expected_contact_distance_m": [0.022, 0.025],
    "line_start_xy_m": [0.1707, 0.0374],
    # the line's start cap seen without a puck in picture 000 (camera_model step, measured by its detector): 4.11 and 5.11
    "start_cap_pixels": {"039": [322.6, 334.6], "044": [316.8, 306.6], "045": [317.2, 306.4]},
    "a4_size_m": [0.297, 0.210]
}



def frame_blocks_held_out(
        records: List[Dict],
        settings: Dict
) -> set:
    """
    Choose the held-out pictures: blocks of consecutive pictures in each episode, a seeded random share of them.

    Parameters
    ----------
    records : list of dict
        Frame records.
    settings : dict
        The pipeline settings.

    Returns
    -------
    set of tuple
        (episode, frame index) of every held-out picture.
    """

    randomness_generator = random.Random(settings["split_seed"])
    held_out = set()
    for episode, episode_records in calibration_data.records_by_episode(records).items():
        blocks = sorted(set(record["frame_index"] // settings["held_out_block_frames"] for record in episode_records))
        count = max(1, int(round(settings["held_out_share"] * len(blocks))))
        chosen = set(randomness_generator.sample(blocks, count))
        held_out.update((episode, record["frame_index"]) for record in episode_records
                        if record["frame_index"] // settings["held_out_block_frames"] in chosen)
    return held_out



def extract_edge_points(records: List[Dict], settings: Dict) -> Dict:
    """
    Return the sheet's edge pixels of every picture (cached in a file), the puck's box left out.

    Parameters
    ----------
    records : list of dict
        Frame records.
    settings : dict
        The pipeline settings.

    Returns
    -------
    dict
        (episode, frame index) -> {"points_px", "outward_px"}.
    """

    path = Path(settings["edge_points_path"])
    if path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)
    cv2.setNumThreads(1)
    edges = {}
    for record in records:
        picture = cv2.imread(record["picture_path"])
        result = sheet_edges.find_sheet_edge_points(picture)
        points, outward = result["points_px"], result["outward_px"]
        detection = record["detection"]
        if detection is not None and len(points):
            left, top, width, height = detection["bounding_box"]
            is_off_puck = ~((points[:, 0] >= left - 10) & (points[:, 0] <= left + width + 10)
                            & (points[:, 1] >= top - 10) & (points[:, 1] <= top + height + 10))
            points, outward = points[is_off_puck], outward[is_off_puck]
        edges[(record["episode"], record["frame_index"])] = {"points_px": points, "outward_px": outward}
    with open(path, "wb") as handle:
        pickle.dump(edges, handle)
    return edges



def puck_groups_from_runs(
        runs: List[Dict],
        records_by_key: Dict,
        camera_parameters: Dict,
        point_height_m: float
) -> List[Dict]:
    """
    Turn still-puck runs into groups for the joint fit (joints, pixels and a starting position).

    Parameters
    ----------
    runs : list of dict
        Static runs.
    records_by_key : dict
        (episode, frame index) -> frame record.
    camera_parameters : dict
        The camera for the starting positions.
    point_height_m : float
        Height of the observed puck point above the table.

    Returns
    -------
    list of dict
        Groups with "run_id", "joint_sets", "pixels", "initial_xy_m".
    """

    groups = []
    for run in runs:
        joint_sets, pixels, run_records = fit_camera.run_observations(run, records_by_key)
        _, estimate = fit_camera.run_plane_height(run_records, camera_parameters, point_height_m)
        if estimate is None:
            continue
        groups.append({"run_id": run["run_id"], "joint_sets": joint_sets, "pixels": pixels, "initial_xy_m": estimate.tolist()})
    return groups



def run_joint_fit(
        records: List[Dict],
        edges: Dict,
        frame_keys: List[Tuple[str, int]],
        groups: List[Dict],
        start_camera: Dict,
        start_sheets: Dict[str, List[float]],
        joint_parameters: Dict
) -> Dict:
    """
    Run one joint fit and return its camera, sheets, size, puck height and residuals.

    Parameters
    ----------
    records : list of dict
        Frame records.
    edges : dict
        Edge pixels per picture.
    frame_keys : list of tuple
        The pictures whose edges are fitted.
    groups : list of dict
        Still-puck groups.
    start_camera : dict
        Starting camera.
    start_sheets : dict
        Episode -> starting [x, y, angle] of its sheet.
    joint_parameters : dict
        Overrides for ``joint_fit.JOINT_FIT_PARAMETERS``.

    Returns
    -------
    dict
        "camera", "sheets", "sheet_size_m", "puck_height_m", "group_points", "rounds", "fit" (the fit object) and
        "vector".
    """

    observations = joint_fit.EdgeObservations(records, edges, frame_keys)
    sheets = {episode: pose for episode, pose in start_sheets.items() if episode in set(observations.episode_of_point)}
    fit = joint_fit.JointCameraFit(observations, groups, start_camera, sheets, joint_parameters)
    started = time.time()
    vector, information = fit.solve(fit.initial_vector())
    unpacked = fit.unpack(vector)
    return {
        "camera": fit.camera_parameters(vector),
        "values": unpacked["values"],
        "sheets": {episode: unpacked["sheets"][index].tolist() for index, episode in enumerate(fit.episodes)},
        "sheet_size_m": unpacked["sheet_size_m"].tolist(),
        "puck_height_m": float(unpacked["puck_height_m"]),
        "group_points": {group["run_id"]: unpacked["groups"][index].tolist() for index, group in enumerate(groups)},
        "rounds": information["rounds"],
        "seconds": time.time() - started,
        "fit": fit,
        "vector": vector,
        "jacobian": information["jacobian"]
    }



def evaluate_edges(
        records: List[Dict],
        edges: Dict,
        frame_keys: List[Tuple[str, int]],
        camera: Dict,
        sheets: Dict[str, List[float]],
        sheet_size_m: List[float],
        gate_m: float
) -> Dict:
    """
    Measure how far the edge pixels of the given pictures lie from the fitted sheet's sides: in pixels across the
    side's picture, and in millimetres on the table.

    Parameters
    ----------
    records : list of dict
        Frame records.
    edges : dict
        Edge pixels per picture.
    frame_keys : list of tuple
        The pictures to judge.
    camera : dict
        The camera.
    sheets : dict
        Episode -> [x, y, angle] of its sheet.
    sheet_size_m : list of float
        The sheet's length and width.
    gate_m : float
        Matching gate on the table.

    Returns
    -------
    dict
        "pixel_errors_px" and "table_errors_m" (absolute values), "points" (count) and "frames" (count).
    """

    observations = joint_fit.EdgeObservations(records, edges, frame_keys)
    known = {episode: pose for episode, pose in sheets.items() if episode in set(observations.episode_of_point)}
    fit = joint_fit.JointCameraFit(observations, [], camera, known, {"is_puck_height_free": False, "sheet_size_m": list(sheet_size_m),
                                                                     "is_sheet_size_free": False, "free_camera_parameters": []})
    vector = fit.initial_vector()
    fit.assign_sides(vector, gate_m)
    pixel_errors = numpy.abs(fit.edge_residuals_px(vector))
    geometry = fit.edge_geometry(vector)
    table_errors = geometry["distance"][fit.is_edge_used]
    return {"pixel_errors_px": pixel_errors, "table_errors_m": table_errors, "points": int(fit.is_edge_used.sum()),
            "frames": len(observations.frame_keys)}



def fit_sheet_with_camera_fixed(
        records: List[Dict],
        edges: Dict,
        frame_keys: List[Tuple[str, int]],
        camera: Dict,
        start_sheets: Dict[str, List[float]],
        sheet_size_m: List[float]
) -> Dict[str, List[float]]:
    """
    Fit only the sheet poses of the given pictures' episodes, the camera and the size fixed (for left-out episodes).

    Parameters
    ----------
    records, edges, frame_keys
        As for ``evaluate_edges``.
    camera : dict
        The fixed camera.
    start_sheets : dict
        Starting sheet poses.
    sheet_size_m : list of float
        The fixed size.

    Returns
    -------
    dict
        Episode -> fitted [x, y, angle].
    """

    result = run_joint_fit(records, edges, frame_keys, [], camera, start_sheets,
                           {"is_puck_height_free": False, "is_sheet_size_free": False, "sheet_size_m": list(sheet_size_m),
                            "free_camera_parameters": []})
    return result["sheets"]



def push_check(
        records: List[Dict],
        camera: Dict,
        point_height_m: float,
        settings: Dict
) -> Dict:
    """
    Measure where the puck is mapped relative to the magnet tip right after the robot pushed it.

    A picture counts as "just pushed" when the step before it moved the tip at least "push_minimum_step_m", within
    "push_maximum_angle_deg" of the direction from the tip to the puck, the puck was at most
    "push_maximum_distance_before_m" from the tip before the step, and the puck itself moved at least
    "push_minimum_puck_move_m" (both pictures with a usable detection). The puck is mapped with the camera to the
    plane at ``point_height_m`` above the table.

    Parameters
    ----------
    records : list of dict
        Frame records.
    camera : dict
        The camera.
    point_height_m : float
        Height of the detector's puck point above the table.
    settings : dict
        The pipeline settings.

    Returns
    -------
    dict
        Arrays "distance_m", "ahead_m", "sideways_m" and per-picture rows.
    """

    rows = []
    for episode, episode_records in calibration_data.records_by_episode(records).items():
        estimates = [static_runs.estimate_puck_xy(record, camera, point_height_m)
                     if static_runs.is_usable_detection(record["detection"], static_runs.STATIC_RUN_PARAMETERS) else None
                     for record in episode_records]
        for index in range(len(episode_records) - 1):
            current, following = episode_records[index], episode_records[index + 1]
            if following["frame_index"] != current["frame_index"] + 1:
                continue
            puck_before, puck_after = estimates[index], estimates[index + 1]
            if puck_before is None or puck_after is None:
                continue
            step = following["tip_xyz"][:2] - current["tip_xyz"][:2]
            step_length = float(numpy.linalg.norm(step))
            towards = puck_before - current["tip_xyz"][:2]
            towards_length = float(numpy.linalg.norm(towards))
            if step_length < settings["push_minimum_step_m"] or towards_length < 1e-9 or towards_length > settings["push_maximum_distance_before_m"]:
                continue
            angle = math.degrees(math.acos(max(-1.0, min(1.0, float(step @ towards) / (step_length * towards_length)))))
            puck_move = float(numpy.linalg.norm(puck_after - puck_before))
            if angle > settings["push_maximum_angle_deg"] or puck_move < settings["push_minimum_puck_move_m"]:
                continue
            direction = step / step_length
            offset = puck_after - following["tip_xyz"][:2]
            rows.append({"episode": episode, "frame_index": following["frame_index"], "mode": following["mode"],
                         "action": current["row"]["action"] if current["row"] else None,
                         "distance_m": float(numpy.linalg.norm(offset)), "ahead_m": float(offset @ direction),
                         "sideways_m": float(direction[0] * offset[1] - direction[1] * offset[0]),
                         "step_m": step_length, "puck_move_m": puck_move, "lean_deg": following["lean_deg"]})
    return {"rows": rows, "distance_m": numpy.array([row["distance_m"] for row in rows]),
            "ahead_m": numpy.array([row["ahead_m"] for row in rows]), "sideways_m": numpy.array([row["sideways_m"] for row in rows])}



def start_check(
        records_by_key: Dict,
        camera: Dict,
        point_height_m: float,
        sheets: Dict[str, List[float]],
        sheet_size_m: List[float],
        settings: Dict
) -> Dict:
    """
    Map the puck and the bare line start of picture 000 to the table and compare them with the taught start point,
    the block's contact distance and the sheet's near edge.

    Parameters
    ----------
    records_by_key : dict
        (episode, frame index) -> frame record.
    camera : dict
        The camera.
    point_height_m : float
        Height of the detector's puck point above the table.
    sheets : dict
        Episode -> [x, y, angle] of its sheet.
    sheet_size_m : list of float
        The sheet's size.
    settings : dict
        The pipeline settings.

    Returns
    -------
    dict
        Per picture: the mapped point, its offset from the taught start, its distance from the tip, and its distance
        from the sheet's near edge and left edge (along and across the sheet).
    """

    def sheet_coordinates(episode: str, point: numpy.ndarray) -> Dict:
        centre_x, centre_y, angle = sheets[episode]
        offset = point - numpy.array([centre_x, centre_y])
        local_x = math.cos(angle) * offset[0] + math.sin(angle) * offset[1]
        local_y = -math.sin(angle) * offset[0] + math.cos(angle) * offset[1]
        return {"from_near_edge_m": float(local_x + sheet_size_m[0] / 2.0), "from_left_edge_m": float(sheet_size_m[1] / 2.0 - local_y)}

    results = {}
    taught = numpy.array(settings["line_start_xy_m"])
    for episode in ["037", "042", "043"]:
        record = records_by_key.get((episode, 0))
        if record is None or record["detection"] is None:
            continue
        point = static_runs.estimate_puck_xy(record, camera, point_height_m)
        if point is None:
            continue
        entry = {"kind": "puck", "pixel": static_runs.detection_pixel(record["detection"]), "mapped_xy_m": point.tolist(),
                 "offset_from_taught_start_m": (point - taught).tolist(),
                 "distance_from_tip_m": float(numpy.linalg.norm(point - record["tip_xyz"][:2])),
                 "tip_xy_m": record["tip_xyz"][:2].tolist()}
        if episode in sheets:
            entry.update(sheet_coordinates(episode, point))
        results[f"{episode}/000 puck"] = entry
    for episode, pixel in settings["start_cap_pixels"].items():
        record = records_by_key.get((episode, 0))
        if record is None:
            continue
        height = calibration_data.table_height(*settings["line_start_xy_m"])
        point = pixels_to_plane(numpy.array([pixel], dtype = float), record["joints"], height, camera)[0]
        entry = {"kind": "line start cap (no puck)", "pixel": pixel, "mapped_xy_m": point.tolist(),
                 "offset_from_taught_start_m": (point - taught).tolist(),
                 "distance_from_tip_m": float(numpy.linalg.norm(point - record["tip_xyz"][:2]))}
        if episode in sheets:
            entry.update(sheet_coordinates(episode, point))
        results[f"{episode}/000 cap"] = entry
    return results



def per_frame_shifts(
        fit: joint_fit.JointCameraFit,
        vector: numpy.ndarray,
        minimum_points: int = 15
) -> Dict:
    """
    Estimate, for every picture with edge pixels on at least two differently directed sides, the common image shift
    (du, dv) that best explains its edge residuals: how far that picture's whole view sits from the fitted camera's.

    Parameters
    ----------
    fit : joint_fit.JointCameraFit
        A solved fit (sides assigned).
    vector : numpy.ndarray
        Its solution.
    minimum_points : int, optional
        Pictures with fewer used edge pixels are skipped.

    Returns
    -------
    dict
        "frame_keys" and "shifts_px" (M, 2).
    """

    residuals = fit.edge_residuals_px(vector)
    used = numpy.nonzero(fit.is_edge_used)[0]
    unpacked = fit.unpack(vector)
    camera = joint_fit.camera_from_values(unpacked["values"], fit.initial_rotation)
    points = joint_fit.map_pixels_to_table(fit.edges.pixels[used], fit.edges.point_positions[used], fit.edges.point_rotations[used], camera)
    sheets = unpacked["sheets"][fit.point_episode_index[used]]
    feet, directions, _, _ = joint_fit.rectangle_side_geometry(numpy.nan_to_num(points[:, :2]), fit.sides[used], sheets[:, :2], sheets[:, 2],
                                                               unpacked["sheet_size_m"] / 2.0)
    rotation, translation, focal, principal_point, k1 = camera
    from forearm_camera_model import project_with_forearm_frames
    first, _, _ = project_with_forearm_frames(joint_fit_table_points(feet), fit.edges.point_positions[used], fit.edges.point_rotations[used],
                                              rotation, translation, focal, principal_point, k1)
    second, _, _ = project_with_forearm_frames(joint_fit_table_points(feet + 0.01 * directions), fit.edges.point_positions[used],
                                               fit.edges.point_rotations[used], rotation, translation, focal, principal_point, k1)
    along = second - first
    along = along / numpy.maximum(numpy.linalg.norm(along, axis = 1), 1e-12)[:, None]
    normals = numpy.column_stack([-along[:, 1], along[:, 0]])
    frames = fit.edges.frame_of_point[used]
    keys, shifts = [], []
    for frame in numpy.unique(frames):
        selected = frames == frame
        if selected.sum() < minimum_points:
            continue
        matrix = normals[selected]
        if numpy.linalg.svd(matrix, compute_uv = False)[-1] < 0.3 * math.sqrt(selected.sum()):
            continue
        shift = numpy.linalg.lstsq(matrix, residuals[selected], rcond = None)[0]
        keys.append(fit.edges.frame_keys[frame])
        shifts.append(shift)
    return {"frame_keys": keys, "shifts_px": numpy.array(shifts)}



def joint_fit_table_points(points_xy: numpy.ndarray) -> numpy.ndarray:
    """
    Return table points (N, 3) on config's table plane.

    Parameters
    ----------
    points_xy : numpy.ndarray
        Points (N, 2).

    Returns
    -------
    numpy.ndarray
        Points with their table height.
    """

    plane = calibration_data.CALIBRATION_TABLE_PLANE
    return numpy.column_stack([points_xy, plane["a"] * points_xy[:, 0] + plane["b"] * points_xy[:, 1] + plane["c"]])



def conditioning_from_jacobian(result: Dict) -> Dict:
    """
    Formal standard deviations and correlations of the camera's quantities from the final Jacobian (scipy's
    Jacobian at the solution already carries the robust loss), the nuisance quantities eliminated by taking the
    camera's block of the inverse. The pictures' own pose errors are correlated from one picture to the next, so these
    deviations are lower bounds; the leave-one-episode-out spread is the realistic one.

    Parameters
    ----------
    result : dict
        A ``run_joint_fit`` result.

    Returns
    -------
    dict
        "standard_deviations" (camera names, puck height, sheet size) and "strong_correlations".
    """

    fit = result["fit"]
    jacobian = result["jacobian"]
    normal = jacobian.T @ jacobian
    normal = normal.toarray() if hasattr(normal, "toarray") else numpy.asarray(normal)
    # a robust estimate of the residuals' spread in sigma units (1.4826 times the median absolute value), squared
    residuals = fit.residuals(result["vector"])
    variance_factor = float((1.4826 * numpy.median(numpy.abs(residuals))) ** 2)
    covariance = numpy.linalg.pinv(normal) * variance_factor
    names = list(fit.free_names)
    names += ["puck_height_m"] if fit.joint_parameters["is_puck_height_free"] else []
    names += ["sheet_length_m", "sheet_width_m"] if fit.joint_parameters["is_sheet_size_free"] else []
    count = len(names)
    deviations = numpy.sqrt(numpy.maximum(numpy.diag(covariance)[:count], 0.0))
    correlations = covariance[:count, :count] / numpy.outer(numpy.maximum(deviations, 1e-300), numpy.maximum(deviations, 1e-300))
    strong = [(names[first], names[second], float(correlations[first, second])) for first in range(count) for second in range(first + 1, count)
              if abs(correlations[first, second]) >= 0.9]
    return {"names": names, "standard_deviations": dict(zip(names, deviations.tolist())), "variance_factor": variance_factor,
            "strong_correlations": strong, "correlations": correlations.tolist()}



def summary_statistics(values: numpy.ndarray, scale: float = 1.0) -> Dict:
    """
    Return count, median, 90th and 95th percentile and maximum of absolute values, times ``scale``.

    Parameters
    ----------
    values : numpy.ndarray
        The values.
    scale : float, optional
        Unit factor (1000 for metres to millimetres).

    Returns
    -------
    dict
        The statistics, rounded to 0.01.
    """

    values = numpy.abs(numpy.asarray(values, dtype = float))
    values = values[numpy.isfinite(values)]
    if len(values) == 0:
        return {"count": 0}
    return {"count": int(len(values)), "median": round(float(numpy.median(values)) * scale, 2),
            "p90": round(float(numpy.percentile(values, 90)) * scale, 2), "p95": round(float(numpy.percentile(values, 95)) * scale, 2),
            "maximum": round(float(values.max()) * scale, 2)}



def main(settings: Optional[Dict] = None) -> None:

    pipeline_settings = copy.deepcopy(PIPELINE_SETTINGS)
    if settings:
        pipeline_settings.update(settings)
    started = time.time()
    results = {"settings": pipeline_settings}

    records = calibration_data.load_frame_records()
    records_by_key = {(record["episode"], record["frame_index"]): record for record in records}
    edges = extract_edge_points(records, pipeline_settings)
    print(f"[{time.time() - started:6.0f} s] records {len(records)}, edge points {sum(len(entry['points_px']) for entry in edges.values())}")

    # stage 1: still-puck runs with the initial guess; the camera's pose fitted, the lens kept at the guess
    stage_1_runs = static_runs.find_static_runs(records, FOREARM_CAMERA_PARAMETERS, pipeline_settings["stage_1_run_parameters"])["runs"]
    experiment = fit_camera.merged_experiment_parameters()
    stage_1 = fit_camera.fit_variant(stage_1_runs, records_by_key, FOREARM_CAMERA_PARAMETERS,
                                     {"fixed_parameters": experiment["intrinsic_names"]}, experiment)
    stage_1_camera = stage_1["parameters"]
    results["stage_1"] = {"runs": len(stage_1_runs), "camera": stage_1_camera, "fitted_errors_px": stage_1["fitted_errors_px"]}
    print(f"[{time.time() - started:6.0f} s] stage 1: {len(stage_1_runs)} runs, fitted median {numpy.median(stage_1['fitted_errors_px']):.2f} px")

    # held-out pictures (blocks of consecutive pictures) and the fitted ones
    held_out_keys = frame_blocks_held_out(records, pipeline_settings)
    all_keys = [(record["episode"], record["frame_index"]) for record in records]
    fitted_keys = [key for key in all_keys if key not in held_out_keys]
    held_out_list = [key for key in all_keys if key in held_out_keys]
    results["split"] = {"held_out_frames": len(held_out_list), "fitted_frames": len(fitted_keys)}

    # stage 2: sheet edges alone, from the stage-1 camera
    observations = joint_fit.EdgeObservations(records, edges, fitted_keys)
    start_sheets = {episode: pose["centre_angle"] for episode, pose in joint_fit.initial_sheet_poses(observations, stage_1_camera).items()}
    start_size = numpy.median([pose["size_m"] for pose in joint_fit.initial_sheet_poses(observations, stage_1_camera).values()], axis = 0).tolist()
    edges_only_parameters = dict(pipeline_settings["edges_only_joint_parameters"])
    edges_only_parameters["sheet_size_m"] = start_size
    stage_2 = run_joint_fit(records, edges, fitted_keys, [], stage_1_camera, start_sheets, edges_only_parameters)
    results["stage_2"] = {key: stage_2[key] for key in ("camera", "values", "sheets", "sheet_size_m", "rounds", "seconds")}
    print(f"[{time.time() - started:6.0f} s] stage 2: size {numpy.round(stage_2['sheet_size_m'], 4)}, rounds {stage_2['rounds'][-1]}")

    # stage 3: still-puck runs again with the stage-2 camera, twists touching the puck left out
    stage_3_runs = static_runs.find_static_runs(records, stage_2["camera"], pipeline_settings["stage_3_run_parameters"])["runs"]
    fitted_runs = [run for run in stage_3_runs if not any((run["episode"], frame) in held_out_keys for frame in run["frame_indices"])]
    held_out_runs = [run for run in stage_3_runs if all((run["episode"], frame) in held_out_keys for frame in run["frame_indices"])]
    results["stage_3"] = {"runs": stage_3_runs, "fitted_runs": [run["run_id"] for run in fitted_runs], "held_out_runs": [run["run_id"] for run in held_out_runs]}
    print(f"[{time.time() - started:6.0f} s] stage 3: {len(stage_3_runs)} runs, {len(fitted_runs)} fitted, {len(held_out_runs)} held out")

    # stage 4a: the final joint fit on the fitted pictures
    final_parameters = dict(pipeline_settings["final_joint_parameters"])
    final_parameters["sheet_size_m"] = stage_2["sheet_size_m"]
    groups = puck_groups_from_runs(fitted_runs, records_by_key, stage_2["camera"], final_parameters["puck_point_height_m"])
    stage_4 = run_joint_fit(records, edges, fitted_keys, groups, stage_2["camera"], stage_2["sheets"], final_parameters)
    results["stage_4"] = {key: stage_4[key] for key in ("camera", "values", "sheets", "sheet_size_m", "puck_height_m", "group_points", "rounds", "seconds")}
    results["stage_4"]["conditioning"] = conditioning_from_jacobian(stage_4)
    fit = stage_4["fit"]
    edge_fitted = numpy.abs(fit.edge_residuals_px(stage_4["vector"]))
    edge_fitted_table = fit.edge_geometry(stage_4["vector"])["distance"][fit.is_edge_used]
    puck_fitted = numpy.hypot(*fit.puck_residuals_px(stage_4["vector"]).T)
    results["stage_4"]["fitted_edge_errors_px"] = edge_fitted
    results["stage_4"]["fitted_edge_errors_m"] = edge_fitted_table
    results["stage_4"]["fitted_puck_errors_px"] = puck_fitted
    print(f"[{time.time() - started:6.0f} s] stage 4: h {stage_4['puck_height_m'] * 1000:.1f} mm, size {numpy.round(stage_4['sheet_size_m'], 4)}, "
          f"edges median {numpy.median(edge_fitted):.2f} px, puck median {numpy.median(puck_fitted):.2f} px")

    # held-out judgement
    held_out_edges = evaluate_edges(records, edges, held_out_list, stage_4["camera"], stage_4["sheets"], stage_4["sheet_size_m"],
                                    pipeline_settings["evaluation_gate_m"])
    fitted_edges_wide = evaluate_edges(records, edges, fitted_keys, stage_4["camera"], stage_4["sheets"], stage_4["sheet_size_m"],
                                       pipeline_settings["evaluation_gate_m"])
    held_out_puck = fit_camera.evaluate_runs(held_out_runs, records_by_key, stage_4["camera"], stage_4["puck_height_m"])
    fitted_puck_eval = fit_camera.evaluate_runs(fitted_runs, records_by_key, stage_4["camera"], stage_4["puck_height_m"])
    initial_puck_eval = fit_camera.evaluate_runs(held_out_runs, records_by_key, FOREARM_CAMERA_PARAMETERS, 0.0145)
    initial_edges = evaluate_edges(records, edges, held_out_list, FOREARM_CAMERA_PARAMETERS, stage_4["sheets"], stage_4["sheet_size_m"], 0.05)
    results["held_out"] = {"edges": held_out_edges, "edges_fitted_frames_same_gate": fitted_edges_wide, "puck": held_out_puck,
                           "puck_fitted_runs": fitted_puck_eval, "puck_initial_guess": initial_puck_eval, "edges_initial_guess": initial_edges}
    print(f"[{time.time() - started:6.0f} s] held out: edges {summary_statistics(held_out_edges['pixel_errors_px'])} px, "
          f"{summary_statistics(held_out_edges['table_errors_m'], 1000)} mm; puck LOO {summary_statistics(held_out_puck['leave_one_out_px'])} px")

    # stage 4b: the delivered fit on all pictures and all runs
    all_groups = puck_groups_from_runs(stage_3_runs, records_by_key, stage_4["camera"], stage_4["puck_height_m"])
    all_parameters = dict(final_parameters)
    all_parameters.update({"puck_point_height_m": stage_4["puck_height_m"], "sheet_size_m": stage_4["sheet_size_m"],
                           "edge_gates_m": [0.012, 0.008, 0.008]})
    delivered = run_joint_fit(records, edges, all_keys, all_groups, stage_4["camera"], stage_4["sheets"], all_parameters)
    results["delivered"] = {key: delivered[key] for key in ("camera", "values", "sheets", "sheet_size_m", "puck_height_m", "group_points", "rounds", "seconds")}
    results["delivered"]["conditioning"] = conditioning_from_jacobian(delivered)
    delivered_fit = delivered["fit"]
    results["delivered"]["fitted_edge_errors_px"] = numpy.abs(delivered_fit.edge_residuals_px(delivered["vector"]))
    results["delivered"]["fitted_edge_errors_m"] = delivered_fit.edge_geometry(delivered["vector"])["distance"][delivered_fit.is_edge_used]
    results["delivered"]["fitted_puck_errors_px"] = numpy.hypot(*delivered_fit.puck_residuals_px(delivered["vector"]).T)
    results["delivered"]["describe"] = describe_camera(delivered["camera"])
    results["delivered"]["per_frame_shifts"] = per_frame_shifts(delivered_fit, delivered["vector"])
    print(f"[{time.time() - started:6.0f} s] delivered: h {delivered['puck_height_m'] * 1000:.1f} mm, size {numpy.round(delivered['sheet_size_m'], 4)}")

    # how strongly the still-puck runs prefer one height of the detector's puck point (camera fixed)
    height_profile = []
    for height in (0.0, 0.005, 0.01, 0.0145, 0.02, 0.025, 0.029):
        evaluation = fit_camera.evaluate_runs(stage_3_runs, records_by_key, delivered["camera"], height)
        refit = evaluation["refit_px"]
        height_profile.append({"height_m": height, "refit_median_px": float(numpy.median(refit)),
                               "huber_cost": float(numpy.where(refit <= 4.0, refit ** 2, 8.0 * refit - 16.0).sum()),
                               "leave_one_out_median_px": float(numpy.median(evaluation["leave_one_out_px"]))})
    results["height_profile"] = height_profile

    # checks the fit never saw
    results["push_check"] = push_check(records, delivered["camera"], delivered["puck_height_m"], pipeline_settings)
    results["push_check_initial_guess"] = push_check(records, FOREARM_CAMERA_PARAMETERS, 0.0145, pipeline_settings)
    results["start_check"] = start_check(records_by_key, delivered["camera"], delivered["puck_height_m"], delivered["sheets"],
                                         delivered["sheet_size_m"], pipeline_settings)
    print(f"[{time.time() - started:6.0f} s] push check: {summary_statistics(results['push_check']['distance_m'], 1000)} mm")

    # leave one episode out: refit without it, fit only its sheet, judge its edges and runs
    if pipeline_settings["is_leave_one_episode_out_run"]:
        leave_one_out = {}
        for episode in sorted(set(key[0] for key in all_keys)):
            kept_keys = [key for key in all_keys if key[0] != episode]
            left_keys = [key for key in all_keys if key[0] == episode]
            kept_groups = [group for group in all_groups if not group["run_id"].startswith(episode)]
            kept_sheets = {name: pose for name, pose in delivered["sheets"].items() if name != episode}
            refit = run_joint_fit(records, edges, kept_keys, kept_groups, delivered["camera"], kept_sheets, all_parameters)
            left_sheet = fit_sheet_with_camera_fixed(records, edges, left_keys, refit["camera"], {episode: delivered["sheets"][episode]},
                                                     refit["sheet_size_m"])
            left_edges = evaluate_edges(records, edges, left_keys, refit["camera"], left_sheet, refit["sheet_size_m"], pipeline_settings["evaluation_gate_m"])
            left_runs = [run for run in stage_3_runs if run["episode"] == episode]
            left_puck = fit_camera.evaluate_runs(left_runs, records_by_key, refit["camera"], refit["puck_height_m"])
            leave_one_out[episode] = {"camera": refit["camera"], "values": refit["values"], "sheet_size_m": refit["sheet_size_m"],
                                      "puck_height_m": refit["puck_height_m"], "left_sheet": left_sheet[episode],
                                      "delivered_sheet": delivered["sheets"][episode],
                                      "left_edges_px": summary_statistics(left_edges["pixel_errors_px"]),
                                      "left_edges_mm": summary_statistics(left_edges["table_errors_m"], 1000),
                                      "left_puck_leave_one_out_px": summary_statistics(left_puck["leave_one_out_px"]),
                                      "left_puck_table_mm": summary_statistics(left_puck["table_leave_one_out_m"], 1000),
                                      "left_edge_errors_px": left_edges["pixel_errors_px"], "left_edge_errors_m": left_edges["table_errors_m"],
                                      "push_check": push_check([record for record in records if record["episode"] == episode], refit["camera"],
                                                               refit["puck_height_m"], pipeline_settings)}
            print(f"[{time.time() - started:6.0f} s] without {episode}: edges {leave_one_out[episode]['left_edges_px']} px, "
                  f"{leave_one_out[episode]['left_edges_mm']} mm")
        results["leave_one_episode_out"] = leave_one_out

    # variants: Niryo's lens kept fixed; the edges alone; the still-puck runs alone were stage 1
    if pipeline_settings["is_variants_run"]:
        variants = {}
        niryo_camera = copy.deepcopy(delivered["camera"])
        niryo_camera.update(pipeline_settings["niryo_lens_variant"])
        niryo_parameters = dict(all_parameters)
        niryo_parameters["free_camera_parameters"] = [name for name in FIT_PARAMETER_NAMES if name not in ("focal_length_px", "radial_distortion_k1")]
        niryo = run_joint_fit(records, edges, fitted_keys, groups, niryo_camera, stage_4["sheets"], niryo_parameters)
        niryo_held_out = evaluate_edges(records, edges, held_out_list, niryo["camera"], niryo["sheets"], niryo["sheet_size_m"], pipeline_settings["evaluation_gate_m"])
        variants["niryo_lens_fixed"] = {"camera": niryo["camera"], "sheet_size_m": niryo["sheet_size_m"], "puck_height_m": niryo["puck_height_m"],
                                        "fitted_edges_px": summary_statistics(numpy.abs(niryo["fit"].edge_residuals_px(niryo["vector"]))),
                                        "held_out_edges_px": summary_statistics(niryo_held_out["pixel_errors_px"]),
                                        "held_out_edges_mm": summary_statistics(niryo_held_out["table_errors_m"], 1000),
                                        "cost": niryo["rounds"][-1]["cost"]}
        variants["final_cost"] = stage_4["rounds"][-1]["cost"]
        edges_only_eval = evaluate_edges(records, edges, held_out_list, stage_2["camera"], stage_2["sheets"], stage_2["sheet_size_m"], pipeline_settings["evaluation_gate_m"])
        variants["edges_only"] = {"camera": stage_2["camera"], "sheet_size_m": stage_2["sheet_size_m"],
                                  "held_out_edges_px": summary_statistics(edges_only_eval["pixel_errors_px"]),
                                  "held_out_puck_leave_one_out_px": summary_statistics(fit_camera.evaluate_runs(held_out_runs, records_by_key, stage_2["camera"], 0.0145)["leave_one_out_px"]),
                                  "push_check_mm": summary_statistics(push_check(records, stage_2["camera"], 0.0145, pipeline_settings)["distance_m"], 1000)}
        variants["puck_runs_only"] = {"camera": stage_1_camera,
                                      "held_out_puck_leave_one_out_px": summary_statistics(fit_camera.evaluate_runs(held_out_runs, records_by_key, stage_1_camera, 0.0145)["leave_one_out_px"]),
                                      "push_check_mm": summary_statistics(push_check(records, stage_1_camera, 0.0145, pipeline_settings)["distance_m"], 1000)}
        results["variants"] = variants
        print(f"[{time.time() - started:6.0f} s] variants done: Niryo lens held-out edges {variants['niryo_lens_fixed']['held_out_edges_px']}")

    for name in ("stage_4", "delivered"):
        results[name].pop("fit", None)
    with open(pipeline_settings["results_path"], "wb") as handle:
        pickle.dump(results, handle)
    print(f"[{time.time() - started:6.0f} s] saved {pipeline_settings['results_path']}")


if __name__ == "__main__":
    overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    main(overrides)
