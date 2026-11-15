"""
Shared geometry for the held-wrist analysis of the robot test of 11.11.26 (key: wrist_and_camera).

The project's own arm model (``ned2_fk.fk``), its tip function (``side_view_test.magnet_tip_and_lean``) and its solver
for joints 1 to 3 (``side_view_test.solve_with_wrist_fixed``) work on one joint set at a time in plain Python. This
module holds numpy copies that work on many joint sets at once, so that all 18,522 demonstration rows can be
checked in seconds, and ``check_against_project_functions`` proves on a random sample that the copies give the same
numbers as the project's functions. Nothing here connects to the robot; it only reads files.

Run from this folder to print the check:
    D:/AI_Project/.venv/Scripts/python.exe wrist_geometry.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy
import pandas

ROOT_FOLDER = Path(__file__).resolve().parents[6]
CODE_FOLDER = ROOT_FOLDER / "02_code"
sys.path.insert(0, str(CODE_FOLDER))

import forearm_camera_model
import ned2_fk
from config import (FOREARM_CAMERA_PARAMETERS, MODEL_MODE_PARAMETERS, PUCK_PARAMETERS, PUSHER_PARAMETERS, RECORDING_PARAMETERS,
                    table_height)
from record_episodes import FOREARM_PICTURE_UP_IN_FOREARM_FRAME, forearm_camera_heading_rad, rod_lean_forward_deg
from side_view_test import magnet_tip_and_lean, solve_with_wrist_fixed


ANALYSIS_PARAMETERS = {
    "frames_file": Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv"),
    "episodes_file": Path("D:/AI_Project/03_data/processed/v01_training_index/episodes.csv"),
    "raw_folder": Path("D:/AI_Project/03_data/raw/main_data"),
    "model_runs_folder": Path("D:/AI_Project/03_data/model_runs"),
    "output_folder": Path(__file__).resolve().parent,
    "wrist_held_deg": MODEL_MODE_PARAMETERS["wrist_held_deg"],
    "hand_held_deg": MODEL_MODE_PARAMETERS["hand_held_deg"],
    # the driver puts the magnet tip this far above the fitted table plane (config.pushing_height minus the extension)
    "driver_tip_above_table_m": PUCK_PARAMETERS["pusher_clearance_above_sheet_m"],
    "solver_iterations": 40,
    "solver_tolerance_m": 1e-7,
    "check_rows": 300,
    "check_seed": 11_11_26,
    # the ten robot runs of 11.11.26: the folder, the sheet and the model
    "robot_runs": [
        {"number": 1, "folder": "2026-11-11_172021_check_no_puck", "sheet": "path_01", "model": "A"},
        {"number": 2, "folder": "2026-11-11_172421_E_sheet01_run1", "sheet": "path_01", "model": "E"},
        {"number": 3, "folder": "2026-11-11_172743_E_sheet02_run1", "sheet": "path_02", "model": "E"},
        {"number": 4, "folder": "2026-11-11_173036_E_sheet02_run2", "sheet": "path_02", "model": "E"},
        {"number": 5, "folder": "2026-11-11_173549_E_sheet03_run1", "sheet": "path_03", "model": "E"},
        {"number": 6, "folder": "2026-11-11_174108_E_sheet04_run1", "sheet": "path_04", "model": "E"},
        {"number": 7, "folder": "2026-11-11_174448_E_sheet05_run1", "sheet": "path_05", "model": "E"},
        {"number": 8, "folder": "2026-11-11_174936_E_sheet06_run1", "sheet": "path_06", "model": "E"},
        {"number": 9, "folder": "2026-11-11_175203_E_sheet03_run2_no_hoodie", "sheet": "path_03", "model": "E"},
        {"number": 10, "folder": "2026-11-11_175544_B_team5_sheet03", "sheet": "path_03", "model": "B"}
    ]
}

JOINT_COLUMNS = [f"joint_{index}" for index in range(1, 7)]
JOINT_LIMITS_RAD = numpy.radians(numpy.array(list(ned2_fk.LIM_DEG.values()), dtype = float))



def rotation_about_x(angle: float) -> numpy.ndarray:
    """The rotation matrix about the x axis by ``angle`` radians (one matrix, shape (3, 3))."""

    cosine, sine = math.cos(angle), math.sin(angle)
    return numpy.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])



def rotation_about_y(angle: float) -> numpy.ndarray:
    """The rotation matrix about the y axis by ``angle`` radians (one matrix, shape (3, 3))."""

    cosine, sine = math.cos(angle), math.sin(angle)
    return numpy.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])



def rotations_about_z(angles: numpy.ndarray) -> numpy.ndarray:
    """The rotation matrices about the z axis for many angles in radians, shape (N, 3, 3)."""

    angles = numpy.asarray(angles, dtype = float)
    matrices = numpy.zeros((len(angles), 3, 3))
    matrices[:, 0, 0] = numpy.cos(angles)
    matrices[:, 0, 1] = -numpy.sin(angles)
    matrices[:, 1, 0] = numpy.sin(angles)
    matrices[:, 1, 1] = numpy.cos(angles)
    matrices[:, 2, 2] = 1.0
    return matrices



def fixed_rotation(
        roll: float,
        pitch: float,
        yaw: float
) -> numpy.ndarray:
    """The URDF rotation Rz(yaw) Ry(pitch) Rx(roll), as ``ned2_fk.rpy`` builds it."""

    return rotations_about_z(numpy.array([yaw]))[0] @ rotation_about_y(pitch) @ rotation_about_x(roll)



def frames_of_many(joint_sets: numpy.ndarray) -> Dict[str, Tuple[numpy.ndarray, numpy.ndarray]]:
    """
    The arm model's frames for many joint sets at once: a numpy copy of ``ned2_fk.fk`` (same chain, same order).

    Parameters
    ----------
    joint_sets : numpy.ndarray
        Joint angles in radians, shape (N, 6).

    Returns
    -------
    dict
        Frame name to (origins of shape (N, 3), rotation matrices of shape (N, 3, 3)), each frame after its joint's
        own rotation, as in ``ned2_fk.fk``.
    """

    joint_sets = numpy.atleast_2d(numpy.asarray(joint_sets, dtype = float))
    count = len(joint_sets)
    rotations = numpy.broadcast_to(numpy.eye(3), (count, 3, 3)).copy()
    positions = numpy.zeros((count, 3))
    frames = {}
    joint_index = 0
    for name, offset, roll_pitch_yaw, is_movable in ned2_fk.CHAIN:
        positions = positions + numpy.einsum("nij,j->ni", rotations, numpy.array(offset, dtype = float))
        rotations = rotations @ fixed_rotation(*roll_pitch_yaw)
        if is_movable:
            rotations = rotations @ rotations_about_z(joint_sets[:, joint_index])
            joint_index += 1
        frames[name] = (positions.copy(), rotations.copy())
    return frames



def tips_and_leans(joint_sets: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    The magnet tip and the rod's lean for many joint sets: a numpy copy of ``side_view_test.magnet_tip_and_lean``.

    Returns
    -------
    tuple of numpy.ndarray
        Tips [x, y, z] in metres, shape (N, 3), and the rod's angle from straight down in degrees, shape (N,).
    """

    frames = frames_of_many(joint_sets)
    rod_directions = frames["joint_6"][1][:, :, 2]
    tips = frames["tool"][0] + PUSHER_PARAMETERS["extension_below_rod_end_m"] * rod_directions
    leans = numpy.degrees(numpy.arccos(numpy.clip(-rod_directions[:, 2], -1.0, 1.0)))
    return tips, leans



def solve_many_with_wrist_fixed(
        target_tips: numpy.ndarray,
        turns_rad: numpy.ndarray,
        wrists_rad: numpy.ndarray,
        starts: numpy.ndarray,
        hands_rad: Optional[numpy.ndarray] = None,
        parameters: Optional[Dict] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Joints 1 to 3 that put the tip on each target for fixed joints 4 and 5: ``side_view_test.solve_with_wrist_fixed``
    for many rows at once (Newton steps with a numeric Jacobian, the same three tests at the end).

    Parameters
    ----------
    target_tips : numpy.ndarray
        Tip targets in metres, shape (N, 3).
    turns_rad, wrists_rad : numpy.ndarray
        Joint 4 and joint 5 of every row in radians, shape (N,).
    starts : numpy.ndarray
        First guesses of joints 1 to 3, shape (N, 3).
    hands_rad : numpy.ndarray, optional
        Joint 6 of every row (it does not move the tip); zero when left out, as in the project's solver.
    parameters : dict, optional
        Overrides of ANALYSIS_PARAMETERS ("solver_iterations", "solver_tolerance_m").

    Returns
    -------
    tuple of numpy.ndarray
        The joints, shape (N, 6), and whether each row is solved: on target within 0.1 mm, within the arm model's joint
        limits and with the elbow above the base, the project solver's three tests.
    """

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    count = len(target_tips)
    joints = numpy.zeros((count, 6))
    joints[:, :3] = starts
    joints[:, 3] = turns_rad
    joints[:, 4] = wrists_rad
    joints[:, 5] = 0.0 if hands_rad is None else hands_rad
    step = 1e-6
    for _ in range(settings["solver_iterations"]):
        tips = tips_and_leans(joints)[0]
        errors = tips - target_tips
        if numpy.max(numpy.linalg.norm(errors, axis = 1)) < settings["solver_tolerance_m"]:
            break
        jacobians = numpy.zeros((count, 3, 3))
        for index in range(3):
            shifted = joints.copy()
            shifted[:, index] += step
            jacobians[:, :, index] = (tips_and_leans(shifted)[0] - tips) / step
        # a capped step keeps rows near a stretched-out arm from jumping away; rows that still do not land are
        # reported as not solved by the tests below
        changes = numpy.linalg.solve(jacobians + 1e-9 * numpy.eye(3)[None], errors[:, :, None])[:, :, 0]
        joints[:, :3] -= numpy.clip(changes, -0.5, 0.5)

    tips = tips_and_leans(joints)[0]
    is_on_target = numpy.linalg.norm(tips - target_tips, axis = 1) < 1e-4
    is_within_limits = numpy.all((joints >= JOINT_LIMITS_RAD[:, 0]) & (joints <= JOINT_LIMITS_RAD[:, 1]), axis = 1)
    is_elbow_above_base = frames_of_many(joints)["joint_3"][0][:, 2] > 0.0
    return joints, is_on_target & is_within_limits & is_elbow_above_base



def headings_rad(joint_sets: numpy.ndarray) -> numpy.ndarray:
    """
    The table direction of the picture's top for many joint sets, in radians: a numpy copy of
    ``record_episodes.forearm_camera_heading_rad`` for the camera that turns with joint 4.
    """

    rotations = frames_of_many(joint_sets)["joint_4"][1]
    looking = rotations[:, :, 2]
    picture_up = numpy.einsum("nij,j->ni", rotations, FOREARM_PICTURE_UP_IN_FOREARM_FRAME)
    picture_right = numpy.cross(looking, picture_up)
    flat_right = picture_right.copy()
    flat_right[:, 2] = 0.0
    top = numpy.cross(numpy.array([0.0, 0.0, 1.0])[None, :], flat_right)
    is_degenerate = numpy.linalg.norm(top, axis = 1) < 1e-9
    top[is_degenerate] = picture_up[is_degenerate]
    along_up = numpy.einsum("ni,ni->n", top, picture_up)
    along_looking = numpy.einsum("ni,ni->n", top, looking)
    is_flipped = (along_up < -1e-9) | ((numpy.abs(along_up) <= 1e-9) & (along_looking < 0.0))
    top[is_flipped] *= -1.0
    return numpy.arctan2(top[:, 1], top[:, 0]) + RECORDING_PARAMETERS["picture_top_offset_rad"]



def forward_leans_deg(
        joint_sets: numpy.ndarray,
        headings: numpy.ndarray
) -> numpy.ndarray:
    """
    The rod's lean towards the picture's top for many joint sets, in degrees from straight down (positive: the tip is
    further towards the picture's top than the rod's upper end): a numpy copy of ``record_episodes.rod_lean_forward_deg``.

    Parameters
    ----------
    joint_sets : numpy.ndarray
        Joint angles in radians, shape (N, 6).
    headings : numpy.ndarray
        The table direction of the picture's top for every row in radians, shape (N,).
    """

    rod_directions = frames_of_many(joint_sets)["joint_6"][1][:, :, 2]
    along = rod_directions[:, 0] * numpy.cos(headings) + rod_directions[:, 1] * numpy.sin(headings)
    return numpy.degrees(numpy.arctan2(along, -rod_directions[:, 2]))



def camera_poses(joint_sets: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    The fitted forearm camera's optical centres, shape (N, 3), and axes, shape (N, 3, 3), for many joint sets
    (``forearm_camera_model.camera_pose_in_base`` for many rows; the camera is fixed in the arm model's joint_4 frame).
    """

    forearm_positions, forearm_rotations = frames_of_many(joint_sets)["joint_4"]
    camera_rotation = forearm_camera_model.rotation_matrix_from_vector(FOREARM_CAMERA_PARAMETERS["rotation_vector_in_forearm_rad"])
    camera_translation = numpy.asarray(FOREARM_CAMERA_PARAMETERS["translation_in_forearm_m"], dtype = float)
    return forearm_positions + numpy.einsum("nij,j->ni", forearm_rotations, camera_translation), forearm_rotations @ camera_rotation



def project_many(
        points_base: numpy.ndarray,
        joint_sets: numpy.ndarray
) -> numpy.ndarray:
    """
    Where points in the base frame appear in the 640 x 480 picture, each point through its own joint set
    (``forearm_camera_model.project_points`` row by row). NaN behind the camera or outside the lens model's field.

    Parameters
    ----------
    points_base : numpy.ndarray
        Points in metres, shape (N, 3).
    joint_sets : numpy.ndarray
        Joint angles in radians, shape (N, 6).

    Returns
    -------
    numpy.ndarray
        Pixels (u to the right, v downwards), shape (N, 2).
    """

    forearm_positions, forearm_rotations = frames_of_many(joint_sets)["joint_4"]
    pixels, _, is_valid = forearm_camera_model.project_with_forearm_frames(
        numpy.asarray(points_base, dtype = float),
        forearm_positions,
        forearm_rotations,
        forearm_camera_model.rotation_matrix_from_vector(FOREARM_CAMERA_PARAMETERS["rotation_vector_in_forearm_rad"]),
        numpy.asarray(FOREARM_CAMERA_PARAMETERS["translation_in_forearm_m"], dtype = float),
        FOREARM_CAMERA_PARAMETERS["focal_length_px"],
        FOREARM_CAMERA_PARAMETERS["principal_point_px"],
        FOREARM_CAMERA_PARAMETERS["radial_distortion_k1"]
    )
    pixels = pixels.copy()
    pixels[~is_valid] = numpy.nan
    return pixels



def pixels_to_table(
        pixels: numpy.ndarray,
        joint_sets: numpy.ndarray,
        plane_points: numpy.ndarray,
        plane_normal: numpy.ndarray
) -> numpy.ndarray:
    """
    Where the viewing rays through ``pixels`` meet a plane, each pixel through its own joint set (the lens distortion
    is undone with the camera model's own function).

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixels (u, v), shape (N, 2).
    joint_sets : numpy.ndarray
        Joint angles in radians, shape (N, 6).
    plane_points : numpy.ndarray
        One point on the plane per row, shape (N, 3).
    plane_normal : numpy.ndarray
        The plane's normal, shape (3,).

    Returns
    -------
    numpy.ndarray
        The points on the plane in metres, shape (N, 3); NaN where the ray runs away from the plane.
    """

    centres, rotations = camera_poses(joint_sets)
    principal_point = numpy.asarray(FOREARM_CAMERA_PARAMETERS["principal_point_px"], dtype = float)
    distorted = (numpy.asarray(pixels, dtype = float) - principal_point[None, :]) / FOREARM_CAMERA_PARAMETERS["focal_length_px"]
    undistorted, _ = forearm_camera_model.undistort_normalised(distorted, FOREARM_CAMERA_PARAMETERS["radial_distortion_k1"])
    rays_camera = numpy.concatenate([undistorted, numpy.ones((len(undistorted), 1))], axis = 1)
    rays_base = numpy.einsum("nij,nj->ni", rotations, rays_camera)
    denominators = rays_base @ plane_normal
    safe_denominators = numpy.where(numpy.abs(denominators) > 1e-12, denominators, numpy.nan)
    distances = numpy.einsum("ni,i->n", plane_points - centres, plane_normal) / safe_denominators
    points = centres + distances[:, None] * rays_base
    points[~(distances > 0)] = numpy.nan
    return points



def table_plane_normal() -> numpy.ndarray:
    """The unit normal of the table plane fitted on 6.11.26 (config.WORKSPACE_PARAMETERS["table_plane"])."""

    slope_x = table_height(1.0, 0.0) - table_height(0.0, 0.0)
    slope_y = table_height(0.0, 1.0) - table_height(0.0, 0.0)
    normal = numpy.array([-slope_x, -slope_y, 1.0])
    return normal / numpy.linalg.norm(normal)



def table_heights(points_xy: numpy.ndarray) -> numpy.ndarray:
    """The fitted table's height in metres under each [x, y], shape (N,)."""

    points_xy = numpy.asarray(points_xy, dtype = float)
    return numpy.array([table_height(float(x), float(y)) for x, y in points_xy])



def load_demonstration_frames(parameters: Optional[Dict] = None) -> pandas.DataFrame:
    """
    The demonstration index (one row per picture) with the columns this analysis needs, in episode order, plus the
    joints in degrees ("joint_4_deg", "joint_5_deg", "joint_6_deg") and the tip's distance from the base ("reach_m").
    """

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    columns = ["episode_id", "row_in_episode", "sheet", "plan_number", "start_kind", "start_end", "operator", "score", "action",
               "tip_forward_m", "tip_left_m", "forearm_change_deg", "lean_change_deg", "hand_change_deg", "tip_x_m", "tip_y_m",
               "tip_z_m", "rod_from_vertical_deg", "picture_heading_deg", "move_length_m", "steps_to_end", "frame_path",
               "frame_index", "goal_distance_cm"] + JOINT_COLUMNS
    frames = pandas.read_csv(settings["frames_file"], usecols = columns, low_memory = False)
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    for index in (4, 5, 6):
        frames[f"joint_{index}_deg"] = numpy.degrees(frames[f"joint_{index}"])
    frames["reach_m"] = numpy.hypot(frames["tip_x_m"], frames["tip_y_m"])
    return frames



def load_robot_run(
        run: Dict,
        parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """One robot run's steps.csv with the run's number, sheet and model added and the joints in degrees."""

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    steps = pandas.read_csv(Path(settings["model_runs_folder"]) / run["folder"] / "steps.csv")
    steps["run_number"] = run["number"]
    steps["sheet"] = run["sheet"]
    steps["model"] = run["model"]
    steps["run_folder"] = run["folder"]
    for index in (4, 5, 6):
        steps[f"joint_{index}_deg"] = numpy.degrees(steps[f"joint_{index}"])
    steps["reach_m"] = numpy.hypot(steps["tip_x"], steps["tip_y"])
    return steps



def check_against_project_functions(parameters: Optional[Dict] = None) -> Dict[str, float]:
    """
    Compare the numpy copies with the project's own functions on a random sample of demonstration rows.

    Returns
    -------
    dict
        The largest differences found: tip (mm), lean (degrees), forearm frame origin (mm) and axes, projected pixel,
        the solver's joints (degrees) for the wrist held at the driver's value, the share of rows on which both solvers
        agree whether a solution exists, and the tip against the index file's tip columns (mm).
    """

    settings = dict(ANALYSIS_PARAMETERS)
    settings.update(parameters or {})
    frames = load_demonstration_frames(settings)
    chosen = numpy.random.default_rng(settings["check_seed"]).choice(len(frames), size = settings["check_rows"], replace = False)
    joint_sets = frames.loc[chosen, JOINT_COLUMNS].to_numpy(dtype = float)

    tips, leans = tips_and_leans(joint_sets)
    forearm_positions, forearm_rotations = frames_of_many(joint_sets)["joint_4"]
    wrist_rad = math.radians(settings["wrist_held_deg"])
    solved, is_solved = solve_many_with_wrist_fixed(tips, joint_sets[:, 3], numpy.full(len(tips), wrist_rad), joint_sets[:, :3])
    points_ahead = tips + numpy.array([0.03, 0.01, -0.005])
    pixels = project_many(points_ahead, joint_sets)

    headings = headings_rad(joint_sets)
    forward_leans = forward_leans_deg(joint_sets, headings)

    largest = {"tip_mm": 0.0, "lean_deg": 0.0, "forearm_origin_mm": 0.0, "forearm_axes": 0.0, "pixel_px": 0.0,
               "solver_joints_deg": 0.0, "solver_agreement_share": 0.0, "tip_against_index_mm": 0.0,
               "heading_deg": 0.0, "forward_lean_deg": 0.0}
    agreements = 0
    for row, joints in enumerate(joint_sets):
        reference_tip, reference_lean = magnet_tip_and_lean(list(joints))
        reference_heading = forearm_camera_heading_rad(list(joints))
        heading_difference = (headings[row] - reference_heading + math.pi) % (2.0 * math.pi) - math.pi
        largest["heading_deg"] = max(largest["heading_deg"], abs(math.degrees(heading_difference)))
        largest["forward_lean_deg"] = max(largest["forward_lean_deg"], abs(float(forward_leans[row]) - rod_lean_forward_deg(list(joints), reference_heading)))
        reference_position, reference_rotation = forearm_camera_model.forearm_frame_in_base(joints)
        reference_pixel = forearm_camera_model.project_points(points_ahead[row], joints)[0]
        reference_solved = solve_with_wrist_fixed(reference_tip, float(joints[3]), wrist_rad, list(joints[:3]))
        largest["tip_mm"] = max(largest["tip_mm"], 1000.0 * float(numpy.linalg.norm(tips[row] - reference_tip)))
        largest["lean_deg"] = max(largest["lean_deg"], abs(float(leans[row]) - reference_lean))
        largest["forearm_origin_mm"] = max(largest["forearm_origin_mm"], 1000.0 * float(numpy.linalg.norm(forearm_positions[row] - reference_position)))
        largest["forearm_axes"] = max(largest["forearm_axes"], float(numpy.abs(forearm_rotations[row] - reference_rotation).max()))
        if numpy.all(numpy.isfinite(reference_pixel)) and numpy.all(numpy.isfinite(pixels[row])):
            largest["pixel_px"] = max(largest["pixel_px"], float(numpy.linalg.norm(pixels[row] - reference_pixel)))
        if (reference_solved is not None) == bool(is_solved[row]):
            agreements += 1
            if reference_solved is not None:
                difference = numpy.abs(solved[row, :5] - numpy.array(reference_solved[:5]))
                largest["solver_joints_deg"] = max(largest["solver_joints_deg"], float(numpy.degrees(difference.max())))
    index_tips = frames.loc[chosen, ["tip_x_m", "tip_y_m", "tip_z_m"]].to_numpy(dtype = float)
    largest["tip_against_index_mm"] = 1000.0 * float(numpy.linalg.norm(tips - index_tips, axis = 1).max())
    largest["solver_agreement_share"] = agreements / len(joint_sets)
    return largest



if __name__ == "__main__":
    differences = check_against_project_functions()
    print(f"numpy copies against the project's functions on {ANALYSIS_PARAMETERS['check_rows']} random demonstration rows (largest differences):")
    for name, value in differences.items():
        print(f"  {name}: {value:.6g}")
