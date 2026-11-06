"""
Fit the forearm camera to the recorded pictures of a puck that does not move, and judge the fit on held-out runs.

The samples are built from the static runs of ``static_runs``: each run is one unknown point on a horizontal plane,
the puck's axis at "puck_point_height_m" above the table (config's table plane), seen in every picture of the run.
The detector's centre is the mean of the visible red pixels, the top face and the near side wall together, so the
point it marks lies on the puck's axis somewhere between the table and the top face (2.9 cm); its height is not
known in advance and is scanned. Optionally the puck of picture 000 of episode 037 is added as a known point at the
taught line start.

Held-out runs are judged without refitting the camera: each run's point is estimated from all but one of its
pictures, and the left-out picture's pixel is predicted (leave one out), so the error is a prediction, not a fit.
"""

import copy
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy

import calibration_data
import static_runs
from forearm_camera_model import (FIT_PARAMETER_NAMES, FOREARM_CAMERA_PARAMETERS, fit_camera_parameters, known_point_sample,
                                  plane_point_group_sample, pixels_to_plane, project_points)

FIT_EXPERIMENT_PARAMETERS = {
    "puck_point_height_m": 0.0145,
    "held_out_share": 0.2,
    "split_seed": 42_069,
    # the taught line start of sheet 04 (teach_sheet_start.py, 31.10.26; used as --start-x / --start-y since 4.11)
    "line_start_xy": [0.1707, 0.0374],
    "known_point_frames": [],
    "known_point_sigma_px": 2.0,
    "intrinsic_names": ["focal_length_px", "principal_point_u_px", "principal_point_v_px", "radial_distortion_k1"]
}



def merged_experiment_parameters(experiment_parameters: Optional[Dict] = None) -> Dict:
    """
    Return ``FIT_EXPERIMENT_PARAMETERS`` with ``experiment_parameters`` merged over them.

    Parameters
    ----------
    experiment_parameters : dict, optional
        Overrides.

    Returns
    -------
    dict
        A new dictionary.
    """

    parameters = copy.deepcopy(FIT_EXPERIMENT_PARAMETERS)
    if experiment_parameters:
        parameters.update(copy.deepcopy(experiment_parameters))
    return parameters



def split_runs(
        runs: List[Dict],
        held_out_share: float,
        seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split the runs into fitted and held-out ones, episode by episode, with a seeded shuffle.

    Parameters
    ----------
    runs : list of dict
        Runs from ``static_runs.find_static_runs``.
    held_out_share : float
        The share of each episode's runs to hold out (rounded, at least one where an episode has three or more).
    seed : int
        Seed of the shuffle.

    Returns
    -------
    tuple of list of dict
        The fitted runs and the held-out runs.
    """

    randomness_generator = random.Random(seed)
    fitted, held_out = [], []
    episodes = sorted(set(run["episode"] for run in runs))
    for episode in episodes:
        episode_runs = sorted((run for run in runs if run["episode"] == episode), key = lambda run: run["run_id"])
        randomness_generator.shuffle(episode_runs)
        count = int(round(held_out_share * len(episode_runs)))
        if count == 0 and len(episode_runs) >= 3:
            count = 1
        held_out.extend(episode_runs[:count])
        fitted.extend(episode_runs[count:])
    return fitted, held_out



def run_observations(
        run: Dict,
        records_by_key: Dict[Tuple[str, int], Dict]
) -> Tuple[List[List[float]], List[List[float]], List[Dict]]:
    """
    Return the joints, the detected pixels and the records of a run's pictures.

    Parameters
    ----------
    run : dict
        A static run.
    records_by_key : dict
        (episode, frame index) -> frame record.

    Returns
    -------
    tuple
        Joint sets, pixels and records, in frame order.
    """

    run_records = [records_by_key[(run["episode"], frame_index)] for frame_index in run["frame_indices"]]
    joint_sets = [record["joints"] for record in run_records]
    pixels = [static_runs.detection_pixel(record["detection"]) for record in run_records]
    return joint_sets, pixels, run_records



def run_plane_height(
        run_records: List[Dict],
        camera_parameters: Dict,
        point_height_m: float
) -> Tuple[float, Optional[numpy.ndarray]]:
    """
    Return the plane height of a run's point (table height at the estimated puck plus the point's height) and the
    estimate.

    Parameters
    ----------
    run_records : list of dict
        The run's frame records.
    camera_parameters : dict
        The camera used for the estimate.
    point_height_m : float
        Height of the observed point above the table.

    Returns
    -------
    tuple
        The plane height in metres and the median position estimate (or None).
    """

    estimates = [static_runs.estimate_puck_xy(record, camera_parameters, point_height_m) for record in run_records]
    estimates = [estimate for estimate in estimates if estimate is not None]
    if not estimates:
        tip = run_records[0]["tip_xyz"]
        return calibration_data.table_height(float(tip[0]), float(tip[1])) + point_height_m, None
    median = numpy.median(numpy.array(estimates), axis = 0)
    return calibration_data.table_height(float(median[0]), float(median[1])) + point_height_m, median



def build_samples(
        runs: List[Dict],
        records_by_key: Dict[Tuple[str, int], Dict],
        camera_parameters: Dict,
        experiment_parameters: Dict
) -> List[Dict]:
    """
    Turn static runs (and the chosen known points) into samples for ``fit_camera_parameters``.

    Parameters
    ----------
    runs : list of dict
        The runs to fit.
    records_by_key : dict
        (episode, frame index) -> frame record.
    camera_parameters : dict
        The camera used for the starting estimates and the plane heights.
    experiment_parameters : dict
        The experiment settings.

    Returns
    -------
    list of dict
        Samples; each group carries its "run_id".
    """

    samples = []
    for run in runs:
        joint_sets, pixels, run_records = run_observations(run, records_by_key)
        plane_height, estimate = run_plane_height(run_records, camera_parameters, experiment_parameters["puck_point_height_m"])
        sample = plane_point_group_sample(plane_height, joint_sets, pixels, initial_xy_m = estimate)
        sample["run_id"] = run["run_id"]
        samples.append(sample)

    for episode, frame_index in experiment_parameters["known_point_frames"]:
        record = records_by_key.get((episode, frame_index))
        if record is None or record["detection"] is None:
            continue
        x, y = experiment_parameters["line_start_xy"]
        point = [x, y, calibration_data.table_height(x, y) + experiment_parameters["puck_point_height_m"]]
        sample = known_point_sample(record["joints"], static_runs.detection_pixel(record["detection"]), point,
                                    pixel_sigma_px = experiment_parameters["known_point_sigma_px"])
        sample["run_id"] = f"known_{episode}_{frame_index:03d}"
        samples.append(sample)
    return samples



def fit_point_on_plane(
        joint_sets: Sequence[Sequence[float]],
        pixels: Sequence[Sequence[float]],
        plane_height_m: float,
        camera_parameters: Dict,
        iterations: int = 20
) -> Optional[numpy.ndarray]:
    """
    Return the point on a horizontal plane whose projections best match the pixels, for a fixed camera
    (Gauss-Newton on the pixel errors, started from the mean of the back-projections).

    Parameters
    ----------
    joint_sets : sequence of sequence of float
        The joints of each picture.
    pixels : sequence of sequence of float
        The observed pixels.
    plane_height_m : float
        The plane's height.
    camera_parameters : dict
        The camera.
    iterations : int, optional
        Gauss-Newton iterations.

    Returns
    -------
    numpy.ndarray or None
        (x, y) in metres, or None when no pixel meets the plane.
    """

    starts = [pixels_to_plane(numpy.array([pixel], dtype = float), joints, plane_height_m, camera_parameters)[0]
              for joints, pixel in zip(joint_sets, pixels)]
    starts = [start for start in starts if numpy.all(numpy.isfinite(start))]
    if not starts:
        return None
    point = numpy.mean(starts, axis = 0)
    observed = numpy.asarray(pixels, dtype = float)

    def predicted(xy: numpy.ndarray) -> numpy.ndarray:
        return numpy.vstack([project_points(numpy.array([xy[0], xy[1], plane_height_m]), joints, camera_parameters)
                             for joints in joint_sets])

    for _ in range(iterations):
        errors = (predicted(point) - observed).ravel()
        if not numpy.all(numpy.isfinite(errors)):
            return point
        jacobian = numpy.zeros((len(errors), 2))
        for axis in range(2):
            step = numpy.zeros(2)
            step[axis] = 1e-6
            jacobian[:, axis] = ((predicted(point + step) - predicted(point - step)).ravel()) / 2e-6
        update = numpy.linalg.lstsq(jacobian, -errors, rcond = None)[0]
        point = point + update
        if numpy.linalg.norm(update) < 1e-9:
            break
    return point



def evaluate_runs(
        runs: List[Dict],
        records_by_key: Dict[Tuple[str, int], Dict],
        camera_parameters: Dict,
        point_height_m: float
) -> Dict:
    """
    Judge a camera on runs it was not fitted to: pixel and table errors, all-in and leave-one-out.

    For each run the point is fitted on the plane from its pictures (2 unknowns per run), and the pixel errors are
    measured ("refit"); then each picture is predicted from the point fitted to the run's other pictures
    ("leave_one_out"). On the table, each picture's detection is mapped to the plane and compared with the mean of the
    run's other pictures' mapped positions (the wander of a still puck on the map as the arm moves).

    Parameters
    ----------
    runs : list of dict
        Static runs.
    records_by_key : dict
        (episode, frame index) -> frame record.
    camera_parameters : dict
        The camera to judge.
    point_height_m : float
        Height of the observed point above the table.

    Returns
    -------
    dict
        Error arrays "refit_px", "leave_one_out_px", "table_leave_one_out_m", and per-run lists.
    """

    refit_errors, leave_one_out_errors, table_errors = [], [], []
    per_run = []
    for run in runs:
        joint_sets, pixels, run_records = run_observations(run, records_by_key)
        plane_height, _ = run_plane_height(run_records, camera_parameters, point_height_m)
        point = fit_point_on_plane(joint_sets, pixels, plane_height, camera_parameters)
        if point is None:
            continue
        run_refit = []
        for joints, pixel in zip(joint_sets, pixels):
            prediction = project_points(numpy.array([point[0], point[1], plane_height]), joints, camera_parameters)[0]
            run_refit.append(float(numpy.hypot(*(prediction - numpy.asarray(pixel)))))
        refit_errors.extend(run_refit)

        mapped = [pixels_to_plane(numpy.array([pixel], dtype = float), joints, plane_height, camera_parameters)[0]
                  for joints, pixel in zip(joint_sets, pixels)]
        run_leave_one_out, run_table = [], []
        for left_out in range(len(pixels)):
            others_joints = [joints for index, joints in enumerate(joint_sets) if index != left_out]
            others_pixels = [pixel for index, pixel in enumerate(pixels) if index != left_out]
            others_point = fit_point_on_plane(others_joints, others_pixels, plane_height, camera_parameters)
            if others_point is None:
                continue
            prediction = project_points(numpy.array([others_point[0], others_point[1], plane_height]), joint_sets[left_out], camera_parameters)[0]
            run_leave_one_out.append(float(numpy.hypot(*(prediction - numpy.asarray(pixels[left_out])))))
            others_mapped = [mapped[index] for index in range(len(mapped)) if index != left_out and numpy.all(numpy.isfinite(mapped[index]))]
            if others_mapped and numpy.all(numpy.isfinite(mapped[left_out])):
                run_table.append(float(numpy.linalg.norm(mapped[left_out] - numpy.mean(others_mapped, axis = 0))))
        leave_one_out_errors.extend(run_leave_one_out)
        table_errors.extend(run_table)
        per_run.append({"run_id": run["run_id"], "views": len(pixels), "point_xy_m": point.tolist(),
                        "refit_px": run_refit, "leave_one_out_px": run_leave_one_out, "table_leave_one_out_m": run_table,
                        "motion": run["motion"]})

    return {"refit_px": numpy.array(refit_errors), "leave_one_out_px": numpy.array(leave_one_out_errors),
            "table_leave_one_out_m": numpy.array(table_errors), "per_run": per_run}



def error_summary(errors: numpy.ndarray) -> Dict:
    """
    Return the count, median, 90th percentile, 95th percentile and largest value of a set of errors.

    Parameters
    ----------
    errors : numpy.ndarray
        Non-negative errors.

    Returns
    -------
    dict
        The statistics (None for an empty set).
    """

    errors = numpy.asarray(errors, dtype = float)
    errors = errors[numpy.isfinite(errors)]
    if len(errors) == 0:
        return {"count": 0, "median": None, "p90": None, "p95": None, "maximum": None}
    return {"count": int(len(errors)), "median": float(numpy.median(errors)), "p90": float(numpy.percentile(errors, 90)),
            "p95": float(numpy.percentile(errors, 95)), "maximum": float(errors.max())}



def fit_variant(
        fitted_runs: List[Dict],
        records_by_key: Dict[Tuple[str, int], Dict],
        start_parameters: Dict,
        fit_settings: Dict,
        experiment_parameters: Dict
) -> Dict:
    """
    Fit one variant (free parameters, priors, point height) and return the fit with its fitted-sample errors.

    Parameters
    ----------
    fitted_runs : list of dict
        Runs to fit.
    records_by_key : dict
        (episode, frame index) -> frame record.
    start_parameters : dict
        The camera to start from (and the values of fixed parameters).
    fit_settings : dict
        Overrides for the fit's settings.
    experiment_parameters : dict
        The experiment settings.

    Returns
    -------
    dict
        The fit's summary (see ``ForearmCameraFit.run``) plus "samples" and "fitted_errors_px".
    """

    samples = build_samples(fitted_runs, records_by_key, start_parameters, experiment_parameters)
    summary = fit_camera_parameters(samples, start_parameters, fit_settings)
    group_errors = [numpy.hypot(residuals[:, 0], residuals[:, 1]) for residuals in summary["group_residuals_px"]]
    known_errors = numpy.hypot(summary["known_point_residuals_px"][:, 0], summary["known_point_residuals_px"][:, 1])
    summary["fitted_errors_px"] = numpy.concatenate(group_errors + [known_errors]) if group_errors else known_errors
    summary["group_errors_px"] = group_errors
    summary["samples"] = samples
    return summary
