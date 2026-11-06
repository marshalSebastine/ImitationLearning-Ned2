"""
Find the runs of consecutive pictures in which the puck cannot have moved, for fitting the forearm camera.

A transition from picture k to picture k + 1 is the robot's move between them. It leaves the puck where it was when
the magnet block cannot have touched it:

    in place     the magnet tip ends within "tip_still_m" of where it started, and on the way (joints interpolated
                 linearly, as a joint move does roughly) it never strays further than "tip_path_still_m" sideways
                 while low over the paper: gamepad twists, the exact replay's small motions;
    lifted turn  a keyboard forearm turn, which lifts the tip 2.5 cm, turns in the air and lands where the block
                 already stood or 1.5 cm further back from the puck (record_episodes.py, turn actions);
    away         the tip moves away from the puck: the step points more than "away_angle_deg" from the direction
                 towards the puck's estimated centre;
    far          the tip's path stays at least "far_from_puck_m" from the puck's estimated centre.

"away" and "far" need the puck's position, estimated from its detection with a camera model; the margins allow for
that estimate being a few centimetres off. A run is a chain of static transitions whose pictures all hold a usable
detection; each run with at least "minimum_views" pictures and some camera motion is one unknown point for the fit.
"""

import math
from typing import Dict, List, Optional

import numpy

import calibration_data
from forearm_camera_model import pixels_to_plane, forearm_frame_in_base
from side_view_test import magnet_tip_and_lean

STATIC_RUN_PARAMETERS = {
    # the tip's end-to-end move that still counts as "in place"
    "tip_still_m": 0.002,
    # the largest sideways stray of the tip on the way while it is low (below "low_tip_height_m" above the table)
    "tip_path_still_m": 0.004,
    # for "in place": the tip must stand at least this far from the puck's estimated centre before and after (contact
    # is about 2.5 cm: the magnet's cover, 24.5 to 26 mm across, plus the puck's radius); 0 switches the test off
    "in_place_minimum_distance_m": 0.0,
    "low_tip_height_m": 0.03,
    "path_samples": 11,
    # keyboard turns lift the tip before turning (record_episodes.py, "turn_lift_m" 2.5 cm)
    "lifted_turn_actions": ["turn_left", "turn_right"],
    "lifted_turn_modes": ["keyboard"],
    "away_angle_deg": 120.0,
    # the tip's path must keep this far from the estimated puck centre for "far" (contact is about 1.9 cm, and the
    # estimate may be 2 cm off with the initial camera guess)
    "far_from_puck_m": 0.045,
    # the tip must start at least this far from the estimated puck centre for "away" (a step away from contact)
    "away_minimum_distance_m": 0.0,
    "is_away_used": True,
    "is_far_used": True,
    # usable detections: whole pucks (not partly hidden), off the picture's edge, confident
    "is_partial_allowed": False,
    "is_edge_allowed": False,
    "minimum_confidence": 0.6,
    "minimum_views": 2,
    # a run whose forearm frame neither turns by this much nor moves this far teaches the camera nothing
    "minimum_run_rotation_deg": 2.0,
    "minimum_run_translation_m": 0.005,
    # height of the observed puck point above the table, for the position estimates used by "away" and "far"
    "puck_point_height_m": 0.0145
}



def merged_run_parameters(run_parameters: Optional[Dict] = None) -> Dict:
    """
    Return ``STATIC_RUN_PARAMETERS`` with ``run_parameters`` merged over them.

    Parameters
    ----------
    run_parameters : dict, optional
        Overrides.

    Returns
    -------
    dict
        A new dictionary.
    """

    parameters = dict(STATIC_RUN_PARAMETERS)
    if run_parameters:
        parameters.update(run_parameters)
    return parameters



def is_usable_detection(
        detection: Optional[Dict],
        run_parameters: Dict
) -> bool:
    """
    Return whether a detection may serve as an observation of the puck's centre.

    Parameters
    ----------
    detection : dict or None
        The detector's dictionary.
    run_parameters : dict
        The run settings.

    Returns
    -------
    bool
        True for a confident whole-puck detection away from the picture's edge.
    """

    if detection is None:
        return False
    if detection.get("is_partial") and not run_parameters["is_partial_allowed"]:
        return False
    if detection.get("is_touching_picture_edge") and not run_parameters["is_edge_allowed"]:
        return False
    return float(detection.get("confidence", 0.0)) >= run_parameters["minimum_confidence"]



def detection_pixel(detection: Dict) -> List[float]:
    """
    Return the detector's best centre estimate (u, v) in pixels.

    Parameters
    ----------
    detection : dict
        The detector's dictionary.

    Returns
    -------
    list of float
        The pixel.
    """

    return [float(detection["centre_u"]), float(detection["centre_v"])]



def estimate_puck_xy(
        record: Dict,
        camera_parameters: Dict,
        point_height_m: float
) -> Optional[numpy.ndarray]:
    """
    Return the puck's table position estimated from one picture's detection.

    The observed point is taken on the puck's axis ``point_height_m`` above the table; the table's height comes from
    config's plane, evaluated at the estimate and refined twice.

    Parameters
    ----------
    record : dict
        A frame record with a detection.
    camera_parameters : dict
        The camera model's parameters.
    point_height_m : float
        Height of the observed point above the table.

    Returns
    -------
    numpy.ndarray or None
        (x, y) in metres, or None without a detection or when the ray misses the plane.
    """

    if record["detection"] is None:
        return None
    pixel = numpy.array([detection_pixel(record["detection"])])
    plane_height = calibration_data.table_height(float(record["tip_xyz"][0]), float(record["tip_xyz"][1])) + point_height_m
    point = None
    for _ in range(3):
        point = pixels_to_plane(pixel, record["joints"], plane_height, camera_parameters)[0]
        if not numpy.all(numpy.isfinite(point)):
            return None
        plane_height = calibration_data.table_height(float(point[0]), float(point[1])) + point_height_m
    return point



def tip_path(
        joints_from: List[float],
        joints_to: List[float],
        sample_count: int
) -> numpy.ndarray:
    """
    Return the magnet tip along a straight line in joint space between two arm positions.

    Parameters
    ----------
    joints_from, joints_to : list of float
        The two sets of six joint angles.
    sample_count : int
        Samples, both ends included.

    Returns
    -------
    numpy.ndarray
        Tip positions, shape (sample_count, 3).
    """

    start = numpy.asarray(joints_from, dtype = float)
    end = numpy.asarray(joints_to, dtype = float)
    fractions = numpy.linspace(0.0, 1.0, sample_count)
    return numpy.array([magnet_tip_and_lean(list(start + fraction * (end - start)))[0] for fraction in fractions])



def distance_to_segment(
        point: numpy.ndarray,
        segment_start: numpy.ndarray,
        segment_end: numpy.ndarray
) -> float:
    """
    Return the distance from a point to a line segment in the plane.

    Parameters
    ----------
    point, segment_start, segment_end : numpy.ndarray
        Points (x, y).

    Returns
    -------
    float
        The shortest distance.
    """

    direction = segment_end - segment_start
    length_squared = float(direction @ direction)
    if length_squared < 1e-18:
        return float(numpy.linalg.norm(point - segment_start))
    fraction = min(1.0, max(0.0, float((point - segment_start) @ direction) / length_squared))
    return float(numpy.linalg.norm(point - (segment_start + fraction * direction)))



def classify_transition(
        current: Dict,
        following: Dict,
        puck_xy: Optional[numpy.ndarray],
        run_parameters: Dict
) -> Dict:
    """
    Decide whether the move from one picture to the next can have left the puck in place, and why.

    Parameters
    ----------
    current, following : dict
        Consecutive frame records of one episode.
    puck_xy : numpy.ndarray or None
        The puck's estimated table position during the move (None when unknown).
    run_parameters : dict
        The run settings.

    Returns
    -------
    dict
        "is_static", "reason" ("in_place", "lifted_turn", "away", "far" or the reason it is not static), the tip's
        end-to-end move "tip_move_m", its largest low stray "tip_stray_m", the forearm's turn "forearm_change_deg",
        and the tip's distances to the puck "distance_before_m" and "distance_after_m" (None without an estimate).
    """

    action = current["row"]["action"] if current["row"] else None
    tip_from = current["tip_xyz"]
    tip_to = following["tip_xyz"]
    tip_move = float(numpy.linalg.norm(tip_to[:2] - tip_from[:2]))
    path = tip_path(current["joints"], following["joints"], run_parameters["path_samples"])
    heights_above_table = numpy.array([point[2] - calibration_data.table_height(float(point[0]), float(point[1])) for point in path])
    is_low = heights_above_table < run_parameters["low_tip_height_m"]
    strays = numpy.linalg.norm(path[:, :2] - tip_from[None, :2], axis = 1)
    tip_stray = float(strays[is_low].max()) if is_low.any() else 0.0

    summary = {
        "action": action,
        "tip_move_m": tip_move,
        "tip_stray_m": tip_stray,
        "forearm_change_deg": following["forearm_deg"] - current["forearm_deg"],
        "distance_before_m": None,
        "distance_after_m": None,
        "is_static": False,
        "reason": "moved"
    }
    if puck_xy is not None:
        summary["distance_before_m"] = float(numpy.linalg.norm(puck_xy - tip_from[:2]))
        summary["distance_after_m"] = float(numpy.linalg.norm(puck_xy - tip_to[:2]))

    if action in run_parameters["lifted_turn_actions"] and current["mode"] in run_parameters["lifted_turn_modes"]:
        summary.update({"is_static": True, "reason": "lifted_turn"})
        return summary

    if tip_move <= run_parameters["tip_still_m"] and tip_stray <= run_parameters["tip_path_still_m"]:
        # a twist in place still turns the block, whose 37 x 31 mm body can shove a puck it touches (seen in 044 and
        # 045), so the block must stand clear of the puck when the puck's position is known
        minimum = run_parameters["in_place_minimum_distance_m"]
        is_clear = (minimum <= 0.0 or (summary["distance_before_m"] is not None and summary["distance_before_m"] >= minimum
                                       and summary["distance_after_m"] >= minimum))
        if is_clear:
            summary.update({"is_static": True, "reason": "in_place"})
            return summary
        summary["reason"] = "in_place_touching"
        return summary

    if puck_xy is None:
        summary["reason"] = "moved_puck_unknown"
        return summary

    if run_parameters["is_far_used"]:
        nearest = min(distance_to_segment(puck_xy, path[index, :2], path[index + 1, :2]) for index in range(len(path) - 1))
        if nearest >= run_parameters["far_from_puck_m"]:
            summary.update({"is_static": True, "reason": "far"})
            return summary

    if run_parameters["is_away_used"]:
        towards_puck = puck_xy - tip_from[:2]
        towards_length = float(numpy.linalg.norm(towards_puck))
        step = tip_to[:2] - tip_from[:2]
        if towards_length > run_parameters["away_minimum_distance_m"] and towards_length > 1e-9 and tip_move > 1e-9:
            cosine = float(step @ towards_puck) / (tip_move * towards_length)
            # every point of the path must also be no closer to the puck than the start, so a curved joint move
            # cannot sweep into it on the way
            path_distances = numpy.linalg.norm(path[:, :2] - puck_xy[None, :], axis = 1)
            if cosine <= math.cos(math.radians(run_parameters["away_angle_deg"])) and path_distances.min() >= towards_length - 1e-4:
                summary.update({"is_static": True, "reason": "away"})
                return summary

    return summary



def forearm_motion_range(records: List[Dict]) -> Dict:
    """
    Return how far the forearm frame turns and moves over a set of pictures (largest pairwise values).

    Parameters
    ----------
    records : list of dict
        Frame records.

    Returns
    -------
    dict
        "rotation_deg" and "translation_m".
    """

    frames = [forearm_frame_in_base(record["joints"]) for record in records]
    largest_rotation, largest_translation = 0.0, 0.0
    for first in range(len(frames)):
        for second in range(first + 1, len(frames)):
            relative = frames[first][1].T @ frames[second][1]
            cosine = max(-1.0, min(1.0, (float(numpy.trace(relative)) - 1.0) / 2.0))
            largest_rotation = max(largest_rotation, math.degrees(math.acos(cosine)))
            largest_translation = max(largest_translation, float(numpy.linalg.norm(frames[first][0] - frames[second][0])))
    return {"rotation_deg": largest_rotation, "translation_m": largest_translation}



def find_static_runs(
        records: List[Dict],
        camera_parameters: Dict,
        run_parameters: Optional[Dict] = None
) -> Dict:
    """
    Classify every transition and collect the runs of pictures that show an unmoved puck.

    Parameters
    ----------
    records : list of dict
        Frame records from ``calibration_data.load_frame_records``.
    camera_parameters : dict
        The camera model used for the puck position estimates ("away" and "far").
    run_parameters : dict, optional
        Overrides for ``STATIC_RUN_PARAMETERS``.

    Returns
    -------
    dict
        "runs": list of dict ("run_id", "episode", "frame_indices", "reasons", "motion"), and "transitions": per
        episode, the classification of every transition.
    """

    parameters = merged_run_parameters(run_parameters)
    runs, transitions = [], {}
    for episode, episode_records in calibration_data.records_by_episode(records).items():
        estimates = [estimate_puck_xy(record, camera_parameters, parameters["puck_point_height_m"])
                     if is_usable_detection(record["detection"], parameters) else None for record in episode_records]
        episode_transitions = []
        for index in range(len(episode_records) - 1):
            current, following = episode_records[index], episode_records[index + 1]
            is_consecutive = following["frame_index"] == current["frame_index"] + 1
            known = [estimate for estimate in (estimates[index], estimates[index + 1]) if estimate is not None]
            puck_xy = numpy.mean(known, axis = 0) if known else None
            summary = classify_transition(current, following, puck_xy, parameters)
            if not is_consecutive:
                summary.update({"is_static": False, "reason": "gap_in_frames"})
            summary["frame_from"] = current["frame_index"]
            summary["frame_to"] = following["frame_index"]
            episode_transitions.append(summary)
        transitions[episode] = episode_transitions

        run_indices = [0]
        for index in range(1, len(episode_records) + 1):
            is_continued = (index < len(episode_records)
                            and episode_transitions[index - 1]["is_static"]
                            and estimates[index - 1] is not None and estimates[index] is not None)
            if is_continued:
                run_indices.append(index)
                continue
            usable = [position for position in run_indices if estimates[position] is not None]
            if len(usable) >= parameters["minimum_views"]:
                run_records = [episode_records[position] for position in usable]
                motion = forearm_motion_range(run_records)
                if (motion["rotation_deg"] >= parameters["minimum_run_rotation_deg"]
                        or motion["translation_m"] >= parameters["minimum_run_translation_m"]):
                    reasons = [episode_transitions[position]["reason"] for position in usable[:-1]]
                    runs.append({
                        "run_id": f"{episode}_{episode_records[usable[0]]['frame_index']:03d}",
                        "episode": episode,
                        "frame_indices": [episode_records[position]["frame_index"] for position in usable],
                        "reasons": reasons,
                        "motion": motion
                    })
            run_indices = [index]

    return {"runs": runs, "transitions": transitions}
