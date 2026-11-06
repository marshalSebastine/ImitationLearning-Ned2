"""
Check the forearm camera model (forearm_camera_model.py) without the robot or any recording: its geometry, and that the
fit recovers a known camera from synthetic pictures.

The fit test makes a ground-truth camera that differs from the fitted one in config.py by a few degrees, millimetres
and pixels, takes noisy synthetic pictures of known points and of unknown points on a plane from arm positions around
the start pose, fits from config's camera and checks that the truth comes back. It runs with the numpy solver, which
every laptop has, and also with scipy's where scipy is installed (02_code/requirements.txt does not list it). The
larger synthetic study of 6.11.26 (data shaped like the recordings, outliers, conditioning) is summarised in
06_deliverables/research_notes/Recording tool/2026-11-06_forearm_camera_calibration/README.md.

Run from 02_code (some seconds; nothing touches the robot, nothing is written):
    python test_forearm_camera_model.py
"""

import math
from typing import Dict, List, Tuple

import numpy

from forearm_camera_model import (camera_pose_in_base, fit_camera_parameters, merged_camera_parameters, picture_footprint_on_plane,
                                  pixel_to_table, pixels_to_plane, project_points, rotation_matrix_from_vector,
                                  rotation_vector_from_matrix)

TEST_PARAMETERS = {
    "seed": 69_420,
    # the arm at the start pose of sheet 04 (episode 037, row 0 of steps.csv)
    "start_joints": [0.24126, -0.73527, -1.01429, 0.00163, 0.17632, 0.23786],
    # arm positions for the synthetic pictures: each joint drawn uniformly around the start pose within these spans
    # (radians); joint 4 turns the camera, joints 5 and 6 do not move it
    "joint_spans_rad": [0.5, 0.15, 0.15, 1.2, 0.0, 0.0],
    "pose_count": 40,
    "table_height_m": 0.006,
    # the middle of the puck's visible silhouette above the table (PUCK_TRACKING_PARAMETERS)
    "puck_point_height_above_table_m": 0.0145,
    "known_points_per_pose": 6,
    "group_count": 6,
    "group_views": 6,
    "pixel_noise_px": 0.5,
    # pixels used for known points: away from the picture's edges and from the hand at the bottom middle
    "pixel_box_px": [40.0, 600.0, 40.0, 330.0],
    # the truth minus config's camera: about the size of the camera creeping on its tape or being re-taped carefully
    "truth_offsets": {
        "rotation_about_camera_axes_deg": [1.5, -1.0, 2.0],
        "translation_m": [0.004, -0.003, 0.006],
        "focal_length_px": -20.0,
        "principal_point_px": [-8.0, 6.0],
        "radial_distortion_k1": -0.03
    },
    # what counts as recovered (the noise is 0.5 px; these are several times the expected errors)
    "maximum_rotation_error_deg": 0.2,
    "maximum_translation_error_mm": 1.5,
    "maximum_focal_error_px": 3.0,
    "maximum_principal_point_error_px": 3.0,
    "maximum_k1_error": 0.01,
    "maximum_group_error_mm": 1.0
}


def truth_camera(test_parameters: Dict) -> Dict:
    """
    Build the ground-truth camera: config's camera moved by ``test_parameters["truth_offsets"]``.

    Parameters
    ----------
    test_parameters : dict
        TEST_PARAMETERS.

    Returns
    -------
    dict
        Camera parameters for forearm_camera_model.
    """

    offsets = test_parameters["truth_offsets"]
    camera = merged_camera_parameters()
    rotation = rotation_matrix_from_vector(camera["rotation_vector_in_forearm_rad"]) @ rotation_matrix_from_vector(
        numpy.radians(offsets["rotation_about_camera_axes_deg"]))
    return merged_camera_parameters({
        "rotation_vector_in_forearm_rad": rotation_vector_from_matrix(rotation).tolist(),
        "translation_in_forearm_m": (numpy.array(camera["translation_in_forearm_m"]) + offsets["translation_m"]).tolist(),
        "focal_length_px": camera["focal_length_px"] + offsets["focal_length_px"],
        "principal_point_px": (numpy.array(camera["principal_point_px"]) + offsets["principal_point_px"]).tolist(),
        "radial_distortion_k1": camera["radial_distortion_k1"] + offsets["radial_distortion_k1"],
        "version": "synthetic_truth"
    })


def synthetic_samples(
        camera: Dict,
        generator: numpy.random.Generator,
        test_parameters: Dict
) -> Tuple[List[Dict], numpy.ndarray]:
    """
    Make noisy synthetic pictures as fit samples: known points on the table, and unknown points on the puck's plane each
    seen from several arm positions.

    Parameters
    ----------
    camera : dict
        The camera that takes the pictures.
    generator : numpy.random.Generator
        The random numbers.
    test_parameters : dict
        TEST_PARAMETERS.

    Returns
    -------
    tuple
        The samples, and the true positions (x, y) of the unknown points, shape (group_count, 2).
    """

    start = numpy.array(test_parameters["start_joints"])
    spans = numpy.array(test_parameters["joint_spans_rad"])
    joint_sets = start + generator.uniform(-1.0, 1.0, (test_parameters["pose_count"], 6)) * spans
    left, right, top, bottom = test_parameters["pixel_box_px"]
    table_height_m = test_parameters["table_height_m"]
    samples = []
    for joints in joint_sets:
        pixels = generator.uniform([left, top], [right, bottom], (test_parameters["known_points_per_pose"], 2))
        points_xy = pixels_to_plane(pixels, list(joints), table_height_m, camera)
        for pixel, point_xy in zip(pixels, points_xy):
            if not numpy.all(numpy.isfinite(point_xy)):
                continue
            samples.append({"kind": "known_point", "joints": list(joints),
                            "pixel": (pixel + generator.normal(0.0, test_parameters["pixel_noise_px"], 2)).tolist(),
                            "point_base_m": [float(point_xy[0]), float(point_xy[1]), table_height_m]})

    # Unknown points: each placed where one picture's centre lands, then seen from every picture that shows it.
    plane_height_m = table_height_m + test_parameters["puck_point_height_above_table_m"]
    group_points = []
    for group_index in range(test_parameters["group_count"]):
        centre_joints = joint_sets[group_index]
        point_xy = pixels_to_plane(numpy.array([[320.0, 240.0]]), list(centre_joints), plane_height_m, camera)[0]
        point = numpy.array([[point_xy[0], point_xy[1], plane_height_m]])
        observations = []
        for joints in joint_sets:
            pixel = project_points(point, list(joints), camera)[0]
            if numpy.all(numpy.isfinite(pixel)) and left <= pixel[0] <= right and top <= pixel[1] <= bottom:
                observations.append({"joints": list(joints), "pixel": (pixel + generator.normal(0.0, test_parameters["pixel_noise_px"], 2)).tolist()})
            if len(observations) == test_parameters["group_views"]:
                break
        assert len(observations) >= 3, (group_index, len(observations))
        samples.append({"kind": "plane_point_group", "plane_height_m": plane_height_m, "observations": observations})
        group_points.append(point_xy)
    return samples, numpy.array(group_points)


def test_rotation_vector_round_trip() -> None:
    """Rotation vector to matrix and back gives the same rotation, also near half a turn (every branch of the inverse)."""

    generator = numpy.random.default_rng(TEST_PARAMETERS["seed"])
    vectors = [axis * angle for axis in numpy.eye(3) for angle in (1e-9, 0.3, math.pi - 1e-4, math.pi)]
    for _ in range(200):
        axis = generator.normal(size = 3)
        vectors.append(axis / numpy.linalg.norm(axis) * generator.uniform(0.0, math.pi))
    for vector in vectors:
        rotation = rotation_matrix_from_vector(vector)
        again = rotation_matrix_from_vector(rotation_vector_from_matrix(rotation))
        assert numpy.abs(again - rotation).max() < 1e-7, (vector, numpy.abs(again - rotation).max())


def test_camera_turns_with_joint_4_about_the_forearm() -> None:
    """A joint-4 turn moves the camera round the forearm's axis at the same distance; joints 5 and 6 do not move it."""

    joints = list(TEST_PARAMETERS["start_joints"])
    turned = list(joints)
    turned[3] += math.radians(30.0)
    wrist_moved = list(joints)
    wrist_moved[4] += 0.5
    wrist_moved[5] -= 0.7
    position, rotation = camera_pose_in_base(joints)
    turned_position, turned_rotation = camera_pose_in_base(turned)
    moved_position, moved_rotation = camera_pose_in_base(wrist_moved)
    assert numpy.abs(moved_position - position).max() < 1e-12 and numpy.abs(moved_rotation - rotation).max() < 1e-12
    relative = turned_rotation @ rotation.T
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, (numpy.trace(relative) - 1.0) / 2.0))))
    assert abs(angle_deg - 30.0) < 1e-6, angle_deg


def test_rays_and_planes() -> None:
    """A plane above the camera gives no point; a pixel beyond a strongly bent lens's range gives none either."""

    joints = list(TEST_PARAMETERS["start_joints"])
    camera_height_m = camera_pose_in_base(joints)[0][2]
    assert pixel_to_table(320.0, 240.0, joints, camera_height_m + 0.1) is None
    assert pixel_to_table(320.0, 240.0, joints, TEST_PARAMETERS["table_height_m"]) is not None
    # With k1 = -0.22 the lens model folds over inside the picture (fold radius 1 / sqrt(3 * 0.22), about 1.23 in
    # normalised units): the picture's corner lies beyond it.
    strongly_bent = {"radial_distortion_k1": -0.22}
    assert pixel_to_table(0.0, 0.0, joints, TEST_PARAMETERS["table_height_m"], strongly_bent) is None


def test_outline_edges_project_back() -> None:
    """The picture's outline on a plane, traced with two points per edge, projects back onto the picture's border."""

    joints = list(TEST_PARAMETERS["start_joints"])
    height_m = TEST_PARAMETERS["table_height_m"]
    outline = picture_footprint_on_plane(joints, height_m, footprint_settings = {"points_per_edge": 2, "maximum_range_m": 5.0})
    assert outline.shape == (8, 2), outline.shape
    back = project_points(numpy.column_stack([outline, numpy.full(8, height_m)]), joints)
    expected = numpy.array([[0.0, 0.0], [319.5, 0.0], [639.0, 0.0], [639.0, 239.5], [639.0, 479.0], [319.5, 479.0], [0.0, 479.0], [0.0, 239.5]])
    usable = numpy.all(numpy.isfinite(back), axis = 1)
    assert usable.sum() >= 6, usable
    assert numpy.abs(back[usable] - expected[usable]).max() < 1e-6, numpy.abs(back[usable] - expected[usable]).max()


def check_fit_recovers_the_camera(solver: str) -> Dict:
    """
    Fit from config's camera to synthetic pictures taken by a moved camera, with ``solver``, and check the truth comes
    back.

    Parameters
    ----------
    solver : str
        "numpy" or "scipy".

    Returns
    -------
    dict
        The errors found.
    """

    generator = numpy.random.default_rng(TEST_PARAMETERS["seed"])
    truth = truth_camera(TEST_PARAMETERS)
    samples, group_points = synthetic_samples(truth, generator, TEST_PARAMETERS)
    result = fit_camera_parameters(samples, merged_camera_parameters(), {"solver": solver, "pixel_sigma_px": TEST_PARAMETERS["pixel_noise_px"]})
    fitted = result["parameters"]
    relative = rotation_matrix_from_vector(truth["rotation_vector_in_forearm_rad"]).T @ rotation_matrix_from_vector(fitted["rotation_vector_in_forearm_rad"])
    errors = {
        "rotation_deg": math.degrees(numpy.linalg.norm(rotation_vector_from_matrix(relative))),
        "translation_mm": 1000.0 * float(numpy.abs(numpy.array(fitted["translation_in_forearm_m"]) - truth["translation_in_forearm_m"]).max()),
        "focal_px": abs(fitted["focal_length_px"] - truth["focal_length_px"]),
        "principal_point_px": float(numpy.abs(numpy.array(fitted["principal_point_px"]) - truth["principal_point_px"]).max()),
        "k1": abs(fitted["radial_distortion_k1"] - truth["radial_distortion_k1"]),
        "group_mm": 1000.0 * float(numpy.hypot(*(numpy.asarray(result["group_points_xy_m"]) - group_points).T).max()),
        "solver": result["solver"]
    }
    assert errors["solver"] == solver, errors
    assert errors["rotation_deg"] < TEST_PARAMETERS["maximum_rotation_error_deg"], errors
    assert errors["translation_mm"] < TEST_PARAMETERS["maximum_translation_error_mm"], errors
    assert errors["focal_px"] < TEST_PARAMETERS["maximum_focal_error_px"], errors
    assert errors["principal_point_px"] < TEST_PARAMETERS["maximum_principal_point_error_px"], errors
    assert errors["k1"] < TEST_PARAMETERS["maximum_k1_error"], errors
    assert errors["group_mm"] < TEST_PARAMETERS["maximum_group_error_mm"], errors
    return errors


def test_fit_recovers_the_camera_with_numpy() -> None:
    """The numpy solver (on every laptop) recovers a camera moved by a few degrees, millimetres and pixels."""

    errors = check_fit_recovers_the_camera("numpy")
    print(f"    numpy fit: rotation {errors['rotation_deg']:.3f} deg, translation {errors['translation_mm']:.2f} mm, focal "
          f"{errors['focal_px']:.2f} px, unknown points {errors['group_mm']:.2f} mm")


def test_fit_recovers_the_camera_with_scipy() -> None:
    """The same with scipy's solver, which "auto" picks where scipy is installed; skipped without scipy."""

    try:
        import scipy  # noqa: F401 (only whether it is installed matters)
    except ImportError:
        print("    scipy is not installed here: skipped")
        return
    errors = check_fit_recovers_the_camera("scipy")
    print(f"    scipy fit: rotation {errors['rotation_deg']:.3f} deg, translation {errors['translation_mm']:.2f} mm, focal "
          f"{errors['focal_px']:.2f} px, unknown points {errors['group_mm']:.2f} mm")


if __name__ == "__main__":
    for test in (test_rotation_vector_round_trip, test_camera_turns_with_joint_4_about_the_forearm, test_rays_and_planes,
                 test_outline_edges_project_back, test_fit_recovers_the_camera_with_numpy, test_fit_recovers_the_camera_with_scipy):
        test()
        print(f"{test.__name__}: passed")
    print("all tests passed")
