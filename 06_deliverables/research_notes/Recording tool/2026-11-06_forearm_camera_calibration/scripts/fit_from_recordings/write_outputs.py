"""
Write the fitted forearm camera as a parameter file ready for config.py, and the calibration's numbers as data.

    calibrated_camera_parameters.json  the camera-model dictionary (FOREARM_CAMERA_PARAMETERS) plus the puck-mapping
                                       height, how they were obtained, when they stop being valid, their uncertainty
                                       and the checks they passed;
    calibration_results.json           every number the checks produced (fitted and held-out errors, push check,
                                       start check, sheet checks, leave-one-episode-out spread, variants).

Run after calibrate_from_recordings.py:
    D:/AI_Project/.venv/Scripts/python.exe write_outputs.py
"""

import json
import math
import pickle
from pathlib import Path
from typing import Dict, List

import numpy

import calibration_data
import static_runs
from calibrate_from_recordings import PIPELINE_SETTINGS, push_check, summary_statistics
from forearm_camera_model import (FIT_PARAMETER_NAMES, FOREARM_CAMERA_PARAMETERS, describe_camera, rotation_matrix_from_vector,
                                  rotation_vector_from_matrix)

OUTPUT_FOLDER = Path(__file__).resolve().parent

OUTPUT_SETTINGS = {
    "results_path": str(OUTPUT_FOLDER / "pipeline_results.pkl"),
    "parameters_path": str(OUTPUT_FOLDER / "calibrated_camera_parameters.json"),
    "numbers_path": str(OUTPUT_FOLDER / "calibration_results.json"),
    "version": "fit_recordings_037_045_2026-11-06"
}



def rounded(value, digits: int = 6):
    """
    Return a JSON-friendly copy of a value with floats rounded (numpy types converted).

    Parameters
    ----------
    value : object
        A number, array, list or dictionary.
    digits : int, optional
        Decimal places for floats.

    Returns
    -------
    object
        The converted value.
    """

    if isinstance(value, dict):
        return {str(key): rounded(item, digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item, digits) for item in value]
    if isinstance(value, numpy.ndarray):
        return rounded(value.tolist(), digits)
    if isinstance(value, (numpy.floating, float)):
        return None if not math.isfinite(float(value)) else round(float(value), digits)
    if isinstance(value, (numpy.integer,)):
        return int(value)
    return value



def jackknife_spread(estimates: List[float], all_data_estimate: float) -> Dict:
    """
    Return the leave-one-out jackknife standard error and the range of a quantity over leave-one-episode-out refits.

    Parameters
    ----------
    estimates : list of float
        One estimate per left-out episode.
    all_data_estimate : float
        The estimate from all episodes.

    Returns
    -------
    dict
        "jackknife_standard_error", "minimum", "maximum", "largest_change_from_all_data".
    """

    values = numpy.asarray(estimates, dtype = float)
    count = len(values)
    error = math.sqrt((count - 1) / count * float(((values - values.mean()) ** 2).sum())) if count > 1 else float("nan")
    return {"jackknife_standard_error": error, "minimum": float(values.min()), "maximum": float(values.max()),
            "largest_change_from_all_data": float(numpy.abs(values - all_data_estimate).max())}



def mapping_spread(
        records: List[Dict],
        delivered_camera: Dict,
        refit_cameras: List[Dict],
        point_height_m: float
) -> Dict:
    """
    Map every detected puck with the delivered camera and with each leave-one-episode-out camera, and return how far
    the refits move the puck on the table (jackknife standard error per picture, then its median and 95th
    percentile over the pictures).

    Parameters
    ----------
    records : list of dict
        Frame records.
    delivered_camera : dict
        The delivered camera.
    refit_cameras : list of dict
        The leave-one-episode-out cameras.
    point_height_m : float
        Height of the detector's puck point.

    Returns
    -------
    dict
        Statistics in millimetres.
    """

    errors, largest = [], []
    count = len(refit_cameras)
    for record in records[::3]:
        if not static_runs.is_usable_detection(record["detection"], static_runs.STATIC_RUN_PARAMETERS):
            continue
        base = static_runs.estimate_puck_xy(record, delivered_camera, point_height_m)
        mapped = [static_runs.estimate_puck_xy(record, camera, point_height_m) for camera in refit_cameras]
        if base is None or any(point is None for point in mapped):
            continue
        mapped = numpy.array(mapped)
        mean = mapped.mean(axis = 0)
        errors.append(math.sqrt((count - 1) / count * float(((mapped - mean) ** 2).sum())))
        largest.append(float(numpy.linalg.norm(mapped - base, axis = 1).max()))
    errors, largest = numpy.array(errors), numpy.array(largest)
    return {"pictures": int(len(errors)), "jackknife_standard_error_mm_median": 1000 * float(numpy.median(errors)),
            "jackknife_standard_error_mm_p95": 1000 * float(numpy.percentile(errors, 95)),
            "largest_refit_change_mm_median": 1000 * float(numpy.median(largest)),
            "largest_refit_change_mm_p95": 1000 * float(numpy.percentile(largest, 95))}



def height_sensitivity(
        records: List[Dict],
        camera: Dict,
        point_height_m: float,
        height_change_m: float = 0.005
) -> Dict:
    """
    Return how far the mapped puck moves on the table when the height of the plane it is mapped to is wrong by
    ``height_change_m`` (a wrong table plane, or a wrong height of the detector's puck point).

    Parameters
    ----------
    records : list of dict
        Frame records.
    camera : dict
        The camera.
    point_height_m : float
        Height of the detector's puck point.
    height_change_m : float, optional
        The height error.

    Returns
    -------
    dict
        Median, 90th percentile and largest move in millimetres over every third picture with a usable detection.
    """

    moves = []
    for record in records[::3]:
        if not static_runs.is_usable_detection(record["detection"], static_runs.STATIC_RUN_PARAMETERS):
            continue
        first = static_runs.estimate_puck_xy(record, camera, point_height_m)
        second = static_runs.estimate_puck_xy(record, camera, point_height_m + height_change_m)
        if first is not None and second is not None:
            moves.append(float(numpy.linalg.norm(second - first)))
    moves = 1000.0 * numpy.array(moves)
    return {"height_change_mm": 1000.0 * height_change_m, "pictures": int(len(moves)), "median_mm": float(numpy.median(moves)),
            "p90_mm": float(numpy.percentile(moves, 90)), "maximum_mm": float(moves.max())}



def main() -> None:

    settings = dict(OUTPUT_SETTINGS)
    with open(settings["results_path"], "rb") as handle:
        results = pickle.load(handle)
    records = calibration_data.load_frame_records()
    delivered = results["delivered"]
    camera = delivered["camera"]
    height = delivered["puck_height_m"]

    # uncertainty from refits that each leave one episode out
    leave_one_out = results.get("leave_one_episode_out", {})
    spread = {}
    if leave_one_out:
        # the refits started from the delivered camera, so their rotation corrections are measured from it; the
        # delivered camera's own corrections are measured from the 80 percent fit and are not comparable, so the
        # rotations are compared as rotations: each refit's camera rotation relative to the delivered one, in degrees
        delivered_rotation = rotation_matrix_from_vector(camera["rotation_vector_in_forearm_rad"])
        relative = numpy.array([numpy.degrees(rotation_vector_from_matrix(delivered_rotation.T @ rotation_matrix_from_vector(entry["camera"]["rotation_vector_in_forearm_rad"])))
                                for entry in leave_one_out.values()])
        for axis, name in enumerate(("camera_tilt_deg", "camera_pan_deg", "camera_roll_deg")):
            spread[name] = jackknife_spread(relative[:, axis].tolist(), 0.0)
        spread["camera_rotation_largest_change_deg"] = float(numpy.linalg.norm(relative, axis = 1).max())
        for name in FIT_PARAMETER_NAMES[3:]:
            spread[name] = jackknife_spread([entry["values"][name] for entry in leave_one_out.values()], delivered["values"][name])
        spread["sheet_length_m"] = jackknife_spread([entry["sheet_size_m"][0] for entry in leave_one_out.values()], delivered["sheet_size_m"][0])
        spread["sheet_width_m"] = jackknife_spread([entry["sheet_size_m"][1] for entry in leave_one_out.values()], delivered["sheet_size_m"][1])
        readable = [describe_camera(entry["camera"]) for entry in leave_one_out.values()]
        delivered_readable = describe_camera(camera)
        for name in ("distance_from_forearm_axis_mm", "position_along_forearm_mm", "optical_axis_from_forearm_axis_deg",
                     "picture_roll_from_forearm_x_deg", "horizontal_field_of_view_deg"):
            spread[name] = jackknife_spread([entry[name] for entry in readable], delivered_readable[name])
        spread["puck_on_table"] = mapping_spread(records, camera, [entry["camera"] for entry in leave_one_out.values()], height)

    # the push check again with the pipeline's current settings (the reach limit before the step was added after the
    # first run), for the delivered camera and the initial guess
    check_settings = dict(results["settings"])
    check_settings.update({key: value for key, value in PIPELINE_SETTINGS.items() if key.startswith("push_")})
    results["push_check"] = push_check(records, camera, height, check_settings)
    results["push_check_initial_guess"] = push_check(records, FOREARM_CAMERA_PARAMETERS, 0.0145, check_settings)
    results["settings"] = check_settings
    for episode, entry in results.get("leave_one_episode_out", {}).items():
        entry["push_check"] = push_check([record for record in records if record["episode"] == episode], entry["camera"],
                                         entry["puck_height_m"], check_settings)
    with open(settings["results_path"], "wb") as handle:
        pickle.dump(results, handle)

    held_out = results["held_out"]
    push = results["push_check"]
    push_by_mode = {}
    for mode in sorted(set(row["mode"] for row in push["rows"])):
        rows = [row for row in push["rows"] if row["mode"] == mode]
        push_by_mode[mode] = {"pictures": len(rows),
                              "distance_mm": summary_statistics(numpy.array([row["distance_m"] for row in rows]), 1000),
                              "distance_mm_p10": 1000 * float(numpy.percentile([row["distance_m"] for row in rows], 10)),
                              "ahead_mm_median": 1000 * float(numpy.median([row["ahead_m"] for row in rows])),
                              "sideways_mm_p10_p90": [1000 * float(numpy.percentile([row["sideways_m"] for row in rows], 10)),
                                                      1000 * float(numpy.percentile([row["sideways_m"] for row in rows], 90))]}

    shifts = delivered["per_frame_shifts"]["shifts_px"]
    numbers = {
        "samples": {
            "pictures": len(records),
            "pictures_with_sheet_edges_used_in_delivered_fit": None,
            "held_out_pictures": results["split"]["held_out_frames"],
            "still_puck_runs": len(results["stage_3"]["runs"]),
            "still_puck_pictures": int(sum(len(run["frame_indices"]) for run in results["stage_3"]["runs"])),
            "still_puck_runs_fitted": len(results["stage_3"]["fitted_runs"]),
            "still_puck_runs_held_out": len(results["stage_3"]["held_out_runs"]),
            "run_reasons": {reason: int(sum(run["reasons"].count(reason) for run in results["stage_3"]["runs"]))
                            for reason in ("in_place", "lifted_turn", "away", "far")}
        },
        "fit_80_percent": {
            "camera": results["stage_4"]["camera"],
            "edge_pixel_errors_px": summary_statistics(results["stage_4"]["fitted_edge_errors_px"]),
            "edge_table_errors_mm": summary_statistics(results["stage_4"]["fitted_edge_errors_m"], 1000),
            "puck_pixel_errors_px": summary_statistics(results["stage_4"]["fitted_puck_errors_px"]),
            "edge_points_used": int(len(results["stage_4"]["fitted_edge_errors_px"])),
            "rounds": results["stage_4"]["rounds"]
        },
        "held_out": {
            "edges_px": summary_statistics(held_out["edges"]["pixel_errors_px"]),
            "edges_mm": summary_statistics(held_out["edges"]["table_errors_m"], 1000),
            "edges_fitted_pictures_same_gate_px": summary_statistics(held_out["edges_fitted_frames_same_gate"]["pixel_errors_px"]),
            "edges_fitted_pictures_same_gate_mm": summary_statistics(held_out["edges_fitted_frames_same_gate"]["table_errors_m"], 1000),
            "puck_leave_one_out_px": summary_statistics(held_out["puck"]["leave_one_out_px"]),
            "puck_refit_px": summary_statistics(held_out["puck"]["refit_px"]),
            "puck_table_wander_mm": summary_statistics(held_out["puck"]["table_leave_one_out_m"], 1000),
            "fitted_runs_puck_leave_one_out_px": summary_statistics(held_out["puck_fitted_runs"]["leave_one_out_px"]),
            "fitted_runs_puck_table_wander_mm": summary_statistics(held_out["puck_fitted_runs"]["table_leave_one_out_m"], 1000),
            "initial_guess_puck_leave_one_out_px": summary_statistics(held_out["puck_initial_guess"]["leave_one_out_px"]),
            "initial_guess_puck_table_wander_mm": summary_statistics(held_out["puck_initial_guess"]["table_leave_one_out_m"], 1000),
            "initial_guess_edges_mm_within_5_cm": summary_statistics(held_out["edges_initial_guess"]["table_errors_m"], 1000)
        },
        "delivered_fit": {
            "camera": camera,
            "readable": describe_camera(camera),
            "values": delivered["values"],
            "puck_point_height_m": height,
            "sheet_size_m": delivered["sheet_size_m"],
            "sheets": delivered["sheets"],
            "edge_pixel_errors_px": summary_statistics(delivered["fitted_edge_errors_px"]),
            "edge_table_errors_mm": summary_statistics(delivered["fitted_edge_errors_m"], 1000),
            "puck_pixel_errors_px": summary_statistics(delivered["fitted_puck_errors_px"]),
            "formal_conditioning": {key: delivered["conditioning"][key] for key in ("standard_deviations", "strong_correlations", "variance_factor")},
            "rounds": delivered["rounds"]
        },
        "per_picture_pose_scatter": {
            "pictures": int(len(shifts)),
            "shift_u_px": {"median": float(numpy.median(shifts[:, 0])), "p10": float(numpy.percentile(shifts[:, 0], 10)), "p90": float(numpy.percentile(shifts[:, 0], 90)),
                           "standard_deviation": float(shifts[:, 0].std())},
            "shift_v_px": {"median": float(numpy.median(shifts[:, 1])), "p10": float(numpy.percentile(shifts[:, 1], 10)), "p90": float(numpy.percentile(shifts[:, 1], 90)),
                           "standard_deviation": float(shifts[:, 1].std())},
            "magnitude_px": summary_statistics(numpy.hypot(shifts[:, 0], shifts[:, 1]))
        },
        "height_profile": results.get("height_profile"),
        "mapping_sensitivity_to_5_mm_height_error": height_sensitivity(records, camera, height),
        "push_check": {
            "pictures": len(push["rows"]),
            "distance_mm": summary_statistics(push["distance_m"], 1000),
            "distance_mm_p10": 1000 * float(numpy.percentile(push["distance_m"], 10)),
            "ahead_mm": {"p10": 1000 * float(numpy.percentile(push["ahead_m"], 10)), "median": 1000 * float(numpy.median(push["ahead_m"])),
                         "p90": 1000 * float(numpy.percentile(push["ahead_m"], 90))},
            "sideways_mm": {"p10": 1000 * float(numpy.percentile(push["sideways_m"], 10)), "median": 1000 * float(numpy.median(push["sideways_m"])),
                            "p90": 1000 * float(numpy.percentile(push["sideways_m"], 90))},
            "by_mode": push_by_mode,
            "initial_guess_distance_mm": summary_statistics(results["push_check_initial_guess"]["distance_m"], 1000),
            "expected_contact_distance_mm": [1000 * value for value in results["settings"]["expected_contact_distance_m"]]
        },
        "start_check": results["start_check"],
        "leave_one_episode_out": {episode: {key: entry[key] for key in ("values", "sheet_size_m", "left_sheet", "delivered_sheet", "left_edges_px",
                                                                         "left_edges_mm", "left_puck_leave_one_out_px", "left_puck_table_mm")}
                                  | {"push_distance_mm": summary_statistics(entry["push_check"]["distance_m"], 1000)}
                                  for episode, entry in leave_one_out.items()},
        "uncertainty_from_leave_one_episode_out": spread,
        "variants": {key: value for key, value in results.get("variants", {}).items()},
        "stage_1_puck_runs_only": {"runs": results["stage_1"]["runs"], "camera": results["stage_1"]["camera"],
                                   "fitted_errors_px": summary_statistics(results["stage_1"]["fitted_errors_px"])},
        "stage_2_edges_only": {"camera": results["stage_2"]["camera"], "sheet_size_m": results["stage_2"]["sheet_size_m"]}
    }
    with open(settings["numbers_path"], "w", encoding = "utf-8") as handle:
        json.dump(rounded(numbers), handle, indent = 1)

    # the parameter file: the camera dictionary for config.py and the height at which the detector's centre is mapped
    parameters = {key: camera[key] for key in ("parent_frame", "rotation_vector_in_forearm_rad", "translation_in_forearm_m", "focal_length_px",
                                                "principal_point_px", "radial_distortion_k1", "image_size_px")}
    parameters["version"] = settings["version"]
    readable = describe_camera(camera)
    initial_readable = describe_camera(FOREARM_CAMERA_PARAMETERS)
    document = {
        "FOREARM_CAMERA_PARAMETERS": rounded(parameters, 6),
        "PUCK_MAPPING_PARAMETERS": {
            "puck_point_height_above_table_m": round(float(height), 4),
            "note": "Map the detector's centre (puck_detector.py centre_u, centre_v) with pixel_to_table at the table's height "
                    "(config.table_height) plus this height. It is the puck's axis at mid-height: the centre of the visible red "
                    "silhouette of an upright cylinder (calc). The recordings cannot pin it down (cost flat from 0 to 20 mm); "
                    "an error of 7 mm here moves the puck by up to about 2 to 4 mm on the table in the views of the recordings (calc)."
        },
        "readable": rounded(readable, 3),
        "how_obtained": [
            "Fitted on 6.11.26 to the recorded episodes 037 to 045 (2026-11-04 and 2026-11-05 sessions, sheet 04, bare camera taped "
            "on the forearm cover), without the robot. Code: calibrate_from_recordings.py in this folder.",
            "Evidence: (1) the white sheet's straight outer edges, found in 1603 of 1692 pictures (sub-pixel, colour based: "
            "sheet_edges.py), each episode's sheet a rectangle of unknown pose and all of one unknown size; (2) the red puck in "
            f"{len(results['stage_3']['runs'])} runs of pictures in which the robot cannot have moved it (static_runs.py); every "
            "picture's joints from steps.csv (episode 041: rebuilt from guided_path.csv).",
            "Model: ned2_fk joint_4 frame (turns with joint 4), pinhole with one radial term (forearm_camera_model.py); all ten "
            "quantities free; robust least squares (scipy, soft L1); the table plane from config (taught 25.10.26 on the old desk).",
            "Judged on held-out pictures (20 percent, in blocks of ten) and by checks the fit never saw: see calibration_results.json."
        ],
        "valid_while": [
            "the camera stays where it was taped on 4.11.26 evening (camera check under about 4 px; the episodes used were within 5 px)",
            "the focus ring is not touched",
            "the robot's automatic calibration holds (joint zeros)",
            "heights come from a table plane taught on the table in use: the table plane in config is the old desk's; on the "
            "first-floor table teach it again before trusting heights (an error of 5 mm in the table moves the puck by 1 to 3 mm)"
        ],
        "uncertainty": rounded({
            "leave_one_episode_out": spread,
            "per_picture_pose_scatter_px": numbers["per_picture_pose_scatter"]["magnitude_px"],
            "note": "The jackknife over the eight episodes is the realistic spread of the parameters; the formal standard "
                    "deviations (calibration_results.json) are several times smaller because every picture carries its own "
                    "pose error of a few pixels that is shared by all its edge pixels."
        }, 4),
        "checks": rounded({
            "held_out_edges_mm": numbers["held_out"]["edges_mm"],
            "held_out_edges_px": numbers["held_out"]["edges_px"],
            "held_out_still_puck_px": numbers["held_out"]["puck_leave_one_out_px"],
            "held_out_still_puck_table_wander_mm": numbers["held_out"]["puck_table_wander_mm"],
            "sheet_size_mm": [1000 * value for value in delivered["sheet_size_m"]],
            "a4_size_mm": [297, 210],
            "push_distance_tip_to_puck_mm": numbers["push_check"]["distance_mm"],
            "start_check": results["start_check"]
        }, 4),
        "initial_guess_for_comparison": rounded({key: FOREARM_CAMERA_PARAMETERS[key] for key in ("rotation_vector_in_forearm_rad", "translation_in_forearm_m",
                                                                                                  "focal_length_px", "principal_point_px", "radial_distortion_k1")}
                                                | {"readable": initial_readable}, 4)
    }
    with open(settings["parameters_path"], "w", encoding = "utf-8") as handle:
        json.dump(document, handle, indent = 1)
    print(json.dumps({"parameters": settings["parameters_path"], "numbers": settings["numbers_path"]}, indent = 1))


if __name__ == "__main__":
    main()
