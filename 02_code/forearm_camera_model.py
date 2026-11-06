"""
Camera model of the Vision Set camera taped to the Ned2's forearm cover.

The module answers four questions: where a point given in the robot's base frame appears in the 640 x 480 picture
(``project_points``), where the ray through a pixel meets a horizontal plane (``pixel_to_table``), which part of such
a plane the picture covers (``picture_footprint_on_plane``), and which camera parameters explain a set of pixels seen
at known joint angles (``fit_camera_parameters``). The first three use numpy only; the fit uses scipy when it is
installed and an equivalent numpy solver otherwise.

Frame. The camera is taped to the black forearm cover just below the elbow seam. That cover is part of Niryo's
``forearm_link``, the child link of joint 4 in the URDF (its mesh, forearm_link.stl, holds the ring from 18 mm before
the joint-4 origin to the tube). In ``ned2_fk.fk`` this link's frame is the one stored under the name ``joint_4``:
the function applies each joint's rotation before it stores the frame, so ``fk(q)["joint_4"]`` already contains
``rz(q[3])``. Its origin lies on the forearm's axis, its z axis is that axis (from the elbow towards the wrist; joint 5
sits 0.17 m along it), and joints 5 and 6 do not move it. The frame stored under ``joint_3`` is the elbow, whose upper
cover carries the two screws; it does not turn with joint 4. The recorded pictures agree: in twelve pictures with joint
4 between -90 and +90 degrees (episodes 037, 039, 042, 043) the hand stays at the bottom middle of the picture while
the table turns around it. The camera's pose is therefore a constant rotation and translation in the ``joint_4`` frame.

Conventions. Camera frame as in OpenCV: x towards the picture's right, y towards its bottom, z along the viewing
direction. Pixels as in OpenCV: u to the right, v downwards, whole numbers at pixel centres, (0, 0) the top-left pixel.
Lens: a pinhole with square pixels and one radial term, (x, y)_distorted = (x, y) (1 + k1 r^2) with r^2 = x^2 + y^2 in
normalised coordinates, which is OpenCV's polynomial model cut after k1. With k1 < 0 that polynomial stops growing at
r = 1 / sqrt(-3 k1); points beyond it are reported as outside the model's field (NaN) so that they are never folded
back into the picture.
"""

import copy
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy

from config import CAMERA_FIT_SETTINGS, FOOTPRINT_SETTINGS, FOREARM_CAMERA_PARAMETERS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06_deliverables/research_notes/Camera mount options/scripts"))
import ned2_fk


# The ten quantities the fit can adjust, in this order. The three rotations are small corrections applied in the
# initial camera's own axes: tilt about the picture's right axis (the view moves up or down the picture), pan about the
# picture's down axis (it moves sideways), roll about the viewing direction (the picture turns). The translation is in
# the joint_4 frame: x towards the camera's side of the forearm, y towards the picture's right, z along the forearm.
FIT_PARAMETER_NAMES = [
    "camera_tilt_correction_rad",
    "camera_pan_correction_rad",
    "camera_roll_correction_rad",
    "translation_x_m",
    "translation_y_m",
    "translation_z_m",
    "focal_length_px",
    "principal_point_u_px",
    "principal_point_v_px",
    "radial_distortion_k1"
]



def merged_camera_parameters(camera_parameters: Optional[Dict] = None) -> Dict:
    """
    Return the default camera parameters with ``camera_parameters`` merged over them.

    Parameters
    ----------
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.

    Returns
    -------
    dict
        A new dictionary; the defaults are not changed.
    """

    parameters = copy.deepcopy(FOREARM_CAMERA_PARAMETERS)
    if camera_parameters:
        parameters.update(copy.deepcopy(camera_parameters))
    return parameters



def skew_matrix(vector: Sequence[float]) -> numpy.ndarray:
    """
    Return the cross-product matrix of a three-vector, so that ``skew_matrix(a) @ b`` equals ``a x b``.

    Parameters
    ----------
    vector : sequence of float
        The three-vector ``a``.

    Returns
    -------
    numpy.ndarray
        The 3 x 3 skew-symmetric matrix.
    """

    x, y, z = [float(value) for value in vector]
    return numpy.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])



def rotation_matrix_from_vector(rotation_vector: Sequence[float]) -> numpy.ndarray:
    """
    Return the rotation matrix of an axis-angle vector (the axis times the angle in radians).

    Parameters
    ----------
    rotation_vector : sequence of float
        The rotation's axis scaled by its angle.

    Returns
    -------
    numpy.ndarray
        The 3 x 3 rotation matrix.
    """

    vector = numpy.asarray(rotation_vector, dtype = float)
    angle = float(numpy.linalg.norm(vector))
    if angle < 1e-12:
        return numpy.eye(3) + skew_matrix(vector)

    cross = skew_matrix(vector / angle)
    rotation = numpy.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)

    # Reference
    # Source: Rodrigues' rotation formula, matrix form R = I + sin(theta) K + (1 - cos(theta)) K^2.
    # Link: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    return rotation



def rotation_vector_from_matrix(rotation: numpy.ndarray) -> numpy.ndarray:
    """
    Return the axis-angle vector of a rotation matrix, with the angle between 0 and pi.

    The matrix is first turned into a unit quaternion with Shepperd's branching on the largest diagonal term, which
    stays accurate near half a turn, where the usual formula from the antisymmetric part loses the axis.

    Parameters
    ----------
    rotation : numpy.ndarray
        A 3 x 3 rotation matrix.

    Returns
    -------
    numpy.ndarray
        The rotation's axis scaled by its angle in radians.
    """

    matrix = numpy.asarray(rotation, dtype = float)
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    # Reference
    # Source: Martin Baker, EuclideanSpace - rotation matrix to quaternion conversion, branching on the trace and the
    #         largest diagonal element (Shepperd's method).
    # Link: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    axis_length = math.sqrt(x * x + y * y + z * z)
    if axis_length < 1e-15:
        return numpy.zeros(3)
    angle = 2.0 * math.atan2(axis_length, w)
    return numpy.array([x, y, z]) * (angle / axis_length)



def largest_valid_radius(radial_distortion_k1: float) -> Tuple[float, float]:
    """
    Return how far from the picture's centre the one-term lens model stays one-to-one.

    With k1 < 0 the distorted radius r (1 + k1 r^2) grows only up to r = 1 / sqrt(-3 k1), where it reaches two thirds
    of that value; with k1 >= 0 it grows for ever.

    Parameters
    ----------
    radial_distortion_k1 : float
        The radial distortion term.

    Returns
    -------
    tuple of float
        The largest undistorted and the largest distorted radius, in normalised coordinates (infinite for k1 >= 0).
    """

    if radial_distortion_k1 >= 0.0:
        return math.inf, math.inf
    undistorted = 1.0 / math.sqrt(-3.0 * radial_distortion_k1)
    return undistorted, 2.0 * undistorted / 3.0



def distort_normalised(
        normalised: numpy.ndarray,
        radial_distortion_k1: float
) -> numpy.ndarray:
    """
    Apply the lens's radial distortion to normalised image coordinates.

    Parameters
    ----------
    normalised : numpy.ndarray
        Ideal pinhole coordinates (x / z, y / z), shape (N, 2).
    radial_distortion_k1 : float
        The radial distortion term.

    Returns
    -------
    numpy.ndarray
        The distorted coordinates, shape (N, 2).
    """

    radius_squared = (normalised ** 2).sum(axis = 1)
    return normalised * (1.0 + radial_distortion_k1 * radius_squared)[:, None]



def undistort_normalised(
        distorted: numpy.ndarray,
        radial_distortion_k1: float,
        iterations: int = 60
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Undo the lens's radial distortion: solve r + k1 r^3 = r_distorted for r with Newton's method, per point.

    For k1 < 0 Newton's method started at the distorted radius climbs monotonically to the root (the function is
    concave and increasing there); for k1 > 0 it descends monotonically. Points beyond the model's largest distorted
    radius have no solution: they are clamped to that radius's undistorted value and flagged.

    Parameters
    ----------
    distorted : numpy.ndarray
        Distorted normalised coordinates ((u - cx) / f, (v - cy) / f), shape (N, 2).
    radial_distortion_k1 : float
        The radial distortion term.
    iterations : int, optional
        Newton iterations; 60 reach machine precision even next to the fold.

    Returns
    -------
    tuple of numpy.ndarray
        The undistorted coordinates, shape (N, 2), and whether each point lies within the model's range, shape (N,).
    """

    distorted = numpy.atleast_2d(numpy.asarray(distorted, dtype = float))
    distorted_radius = numpy.hypot(distorted[:, 0], distorted[:, 1])
    largest_undistorted, largest_distorted = largest_valid_radius(radial_distortion_k1)
    is_valid = distorted_radius <= largest_distorted

    target = numpy.minimum(distorted_radius, largest_distorted)
    radius = target.copy()
    # Points clamped at the fold meet a zero derivative there and their steps overflow; their values are replaced below,
    # so the warnings say nothing about the points that are kept.
    with numpy.errstate(over = "ignore", invalid = "ignore"):
        for _ in range(iterations):
            function = radius + radial_distortion_k1 * radius ** 3 - target
            derivative = numpy.maximum(1.0 + 3.0 * radial_distortion_k1 * radius ** 2, 1e-12)
            radius = radius - function / derivative
    if math.isfinite(largest_undistorted):
        radius = numpy.where(is_valid, radius, largest_undistorted)

    scale = numpy.ones_like(distorted_radius)
    is_off_centre = distorted_radius > 1e-15
    scale[is_off_centre] = radius[is_off_centre] / distorted_radius[is_off_centre]
    return distorted * scale[:, None], is_valid



def forearm_frame_in_base(joints: Sequence[float]) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return the forearm's frame (``joint_4`` of the arm model, which turns with joint 4) in the robot's base frame.

    Parameters
    ----------
    joints : sequence of float
        The six joint angles in radians.

    Returns
    -------
    tuple of numpy.ndarray
        The frame's origin, shape (3,), and its rotation matrix, shape (3, 3), whose columns are its axes.
    """

    position, rotation = ned2_fk.fk([float(value) for value in joints])["joint_4"]
    return numpy.array(position, dtype = float), numpy.array(rotation, dtype = float)



def forearm_frames_in_base(joint_sets: Sequence[Sequence[float]]) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return the forearm frames of many joint sets, stacked.

    Parameters
    ----------
    joint_sets : sequence of sequence of float
        N sets of six joint angles in radians.

    Returns
    -------
    tuple of numpy.ndarray
        Origins, shape (N, 3), and rotation matrices, shape (N, 3, 3).
    """

    frames = [forearm_frame_in_base(joints) for joints in joint_sets]
    if not frames:
        return numpy.zeros((0, 3)), numpy.zeros((0, 3, 3))
    return numpy.array([frame[0] for frame in frames]), numpy.array([frame[1] for frame in frames])



def camera_pose_in_base(
        joints: Sequence[float],
        camera_parameters: Optional[Dict] = None
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return the camera's optical centre and orientation in the robot's base frame for a set of joint angles.

    Parameters
    ----------
    joints : sequence of float
        The six joint angles in radians.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.

    Returns
    -------
    tuple of numpy.ndarray
        The optical centre in metres, shape (3,), and the rotation matrix whose columns are the camera's x (picture
        right), y (picture down) and z (viewing direction) axes in the base frame, shape (3, 3).
    """

    parameters = merged_camera_parameters(camera_parameters)
    forearm_position, forearm_rotation = forearm_frame_in_base(joints)
    camera_rotation_in_forearm = rotation_matrix_from_vector(parameters["rotation_vector_in_forearm_rad"])
    camera_translation_in_forearm = numpy.asarray(parameters["translation_in_forearm_m"], dtype = float)
    return forearm_position + forearm_rotation @ camera_translation_in_forearm, forearm_rotation @ camera_rotation_in_forearm



def project_with_forearm_frames(
        points_base: numpy.ndarray,
        forearm_positions: numpy.ndarray,
        forearm_rotations: numpy.ndarray,
        camera_rotation_in_forearm: numpy.ndarray,
        camera_translation_in_forearm: numpy.ndarray,
        focal_length_px: float,
        principal_point_px: Sequence[float],
        radial_distortion_k1: float
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Project points into the picture, each through its own forearm frame (the workhorse of the projection and the fit).

    Parameters
    ----------
    points_base : numpy.ndarray
        Points in the base frame in metres, shape (N, 3).
    forearm_positions : numpy.ndarray
        The forearm frame's origin for each point, shape (N, 3) or (3,).
    forearm_rotations : numpy.ndarray
        The forearm frame's rotation for each point, shape (N, 3, 3) or (3, 3).
    camera_rotation_in_forearm : numpy.ndarray
        The camera's axes in the forearm frame, shape (3, 3).
    camera_translation_in_forearm : numpy.ndarray
        The optical centre in the forearm frame in metres, shape (3,).
    focal_length_px : float
        The focal length in pixels.
    principal_point_px : sequence of float
        The principal point (u, v) in pixels.
    radial_distortion_k1 : float
        The radial distortion term.

    Returns
    -------
    tuple of numpy.ndarray
        Pixels (u, v), shape (N, 2); depths along the viewing direction in metres, shape (N,); and whether each point is
        in front of the camera and inside the lens model's field, shape (N,).
    """

    points_base = numpy.atleast_2d(numpy.asarray(points_base, dtype = float))
    forearm_positions = numpy.asarray(forearm_positions, dtype = float)
    forearm_rotations = numpy.asarray(forearm_rotations, dtype = float)
    if forearm_rotations.ndim == 2:
        forearm_rotations = numpy.broadcast_to(forearm_rotations, (len(points_base), 3, 3))
        forearm_positions = numpy.broadcast_to(forearm_positions, (len(points_base), 3))

    camera_positions = forearm_positions + numpy.einsum("nij,j->ni", forearm_rotations, camera_translation_in_forearm)
    camera_rotations = numpy.einsum("nij,jk->nik", forearm_rotations, camera_rotation_in_forearm)
    offsets = points_base - camera_positions
    points_camera = numpy.einsum("nji,nj->ni", camera_rotations, offsets)

    depths = points_camera[:, 2]
    safe_depths = numpy.where(depths > 1e-9, depths, 1e-9)
    normalised = points_camera[:, :2] / safe_depths[:, None]
    largest_undistorted, _ = largest_valid_radius(radial_distortion_k1)
    is_valid = (depths > 1e-9) & ((normalised ** 2).sum(axis = 1) < largest_undistorted ** 2)

    distorted = distort_normalised(normalised, radial_distortion_k1)
    pixels = focal_length_px * distorted + numpy.asarray(principal_point_px, dtype = float)[None, :]
    return pixels, depths, is_valid



def project_points(
        points_base: numpy.ndarray,
        joints: Sequence[float],
        camera_parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Return where points given in the robot's base frame appear in the picture taken at ``joints``.

    Parameters
    ----------
    points_base : numpy.ndarray
        Points in metres, shape (N, 3) or (3,).
    joints : sequence of float
        The six joint angles in radians when the picture was taken.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.

    Returns
    -------
    numpy.ndarray
        Pixels (u, v), shape (N, 2); NaN for a point behind the camera or outside the lens model's field. A pixel
        outside 0..639 x 0..479 means the point is in front of the camera but out of the picture.
    """

    parameters = merged_camera_parameters(camera_parameters)
    forearm_position, forearm_rotation = forearm_frame_in_base(joints)
    pixels, _, is_valid = project_with_forearm_frames(
        points_base,
        forearm_position,
        forearm_rotation,
        rotation_matrix_from_vector(parameters["rotation_vector_in_forearm_rad"]),
        numpy.asarray(parameters["translation_in_forearm_m"], dtype = float),
        parameters["focal_length_px"],
        parameters["principal_point_px"],
        parameters["radial_distortion_k1"]
    )
    pixels[~is_valid] = numpy.nan
    return pixels



def is_inside_picture(
        pixels: numpy.ndarray,
        camera_parameters: Optional[Dict] = None,
        margin_px: float = 0.0
) -> numpy.ndarray:
    """
    Return whether pixels lie inside the picture, at least ``margin_px`` from its edges.

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixels (u, v), shape (N, 2); NaN counts as outside.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS`` (only the image size is used).
    margin_px : float, optional
        The distance to keep from the edges.

    Returns
    -------
    numpy.ndarray
        Booleans, shape (N,).
    """

    width, height = merged_camera_parameters(camera_parameters)["image_size_px"]
    pixels = numpy.atleast_2d(numpy.asarray(pixels, dtype = float))
    with numpy.errstate(invalid = "ignore"):
        return ((pixels[:, 0] >= margin_px) & (pixels[:, 0] <= width - 1 - margin_px)
                & (pixels[:, 1] >= margin_px) & (pixels[:, 1] <= height - 1 - margin_px))



def viewing_rays_in_base(
        pixels: numpy.ndarray,
        joints: Sequence[float],
        camera_parameters: Optional[Dict] = None,
        is_clamped_to_field: bool = False
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Return the rays of light that end in the given pixels, in the robot's base frame.

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixels (u, v), shape (N, 2).
    joints : sequence of float
        The six joint angles in radians when the picture was taken.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.
    is_clamped_to_field : bool, optional
        If True, a pixel beyond the lens model's range gets the ray at the edge of the model's field in the same
        direction from the principal point (used for drawing the picture's outline); otherwise it is flagged.

    Returns
    -------
    tuple of numpy.ndarray
        The optical centre, shape (3,); unit ray directions, shape (N, 3); whether each pixel lies within the lens
        model's range, shape (N,).
    """

    parameters = merged_camera_parameters(camera_parameters)
    pixels = numpy.atleast_2d(numpy.asarray(pixels, dtype = float))
    camera_position, camera_rotation = camera_pose_in_base(joints, parameters)
    principal_point = numpy.asarray(parameters["principal_point_px"], dtype = float)
    distorted = (pixels - principal_point[None, :]) / parameters["focal_length_px"]
    normalised, is_valid = undistort_normalised(distorted, parameters["radial_distortion_k1"])
    if is_clamped_to_field:
        is_valid = numpy.ones(len(pixels), dtype = bool)
    rays_camera = numpy.column_stack([normalised, numpy.ones(len(normalised))])
    rays_base = rays_camera @ camera_rotation.T
    rays_base = rays_base / numpy.linalg.norm(rays_base, axis = 1)[:, None]
    return camera_position, rays_base, is_valid



def pixels_to_plane(
        pixels: numpy.ndarray,
        joints: Sequence[float],
        plane_height_m: float,
        camera_parameters: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Return where the rays through many pixels meet the horizontal plane z = ``plane_height_m``.

    Parameters
    ----------
    pixels : numpy.ndarray
        Pixels (u, v), shape (N, 2).
    joints : sequence of float
        The six joint angles in radians when the picture was taken.
    plane_height_m : float
        The plane's height in the base frame. For the puck detector's centre (the middle of the puck's visible
        silhouette): the table's height there plus PUCK_TRACKING_PARAMETERS["puck_point_height_above_table_m"]
        (0.0145 m, as in the fit of 6.11.26); for a known point of the puck's top face: plus 0.029 m.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.

    Returns
    -------
    numpy.ndarray
        Points (x, y) in metres, shape (N, 2); NaN where the pixel lies outside the lens model's range or its ray does
        not meet the plane in front of the camera.
    """

    camera_position, rays, is_valid = viewing_rays_in_base(pixels, joints, camera_parameters)
    points = numpy.full((len(rays), 2), numpy.nan)
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        distances = (plane_height_m - camera_position[2]) / rays[:, 2]
    is_hit = is_valid & numpy.isfinite(distances) & (distances > 0.0)
    points[is_hit] = camera_position[None, :2] + distances[is_hit, None] * rays[is_hit, :2]
    return points



def pixel_to_table(
        pixel_u: float,
        pixel_v: float,
        joints: Sequence[float],
        plane_height_m: float,
        camera_parameters: Optional[Dict] = None
) -> Optional[Tuple[float, float]]:
    """
    Return where the ray through pixel (pixel_u, pixel_v) meets the horizontal plane z = ``plane_height_m``.

    Parameters
    ----------
    pixel_u, pixel_v : float
        The pixel (u to the right, v downwards).
    joints : sequence of float
        The six joint angles in radians when the picture was taken.
    plane_height_m : float
        The plane's height in the base frame. For the puck detector's centre (the middle of the puck's visible
        silhouette): the table's height there plus PUCK_TRACKING_PARAMETERS["puck_point_height_above_table_m"]
        (0.0145 m, as in the fit of 6.11.26); for a known point of the puck's top face: plus 0.029 m.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.

    Returns
    -------
    tuple of float or None
        The point (x, y) in metres in the base frame; None when the ray points away from the plane (or runs parallel
        to it) or the pixel lies outside the lens model's range.
    """

    point = pixels_to_plane(numpy.array([[pixel_u, pixel_v]], dtype = float), joints, plane_height_m, camera_parameters)[0]
    if not numpy.all(numpy.isfinite(point)):
        return None
    return float(point[0]), float(point[1])



def picture_footprint_on_plane(
        joints: Sequence[float],
        plane_height_m: float,
        camera_parameters: Optional[Dict] = None,
        footprint_settings: Optional[Dict] = None
) -> numpy.ndarray:
    """
    Return the outline of the part of a horizontal plane that the picture covers, for drawing it on a map.

    The picture's border is walked clockwise from the top-left pixel (top edge, right edge, bottom edge, left edge)
    and every border pixel's ray is followed to the plane, so the outline shows the lens's bending. Border pixels
    beyond the lens model's range (the picture's corners for strong barrel distortion) take the ray at the edge of
    the model's field. Rays that miss the plane in front of the camera, or meet it beyond the maximum range, are cut
    off at that range in their horizontal direction.

    Parameters
    ----------
    joints : sequence of float
        The six joint angles in radians.
    plane_height_m : float
        The plane's height in the base frame.
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.
    footprint_settings : dict, optional
        Overrides for ``FOOTPRINT_SETTINGS``.

    Returns
    -------
    numpy.ndarray
        The outline's points (x, y) in metres, shape (4 * points_per_edge, 2); with ``points_per_edge`` 1 the four
        corners: top left, top right, bottom right, bottom left.
    """

    settings = dict(FOOTPRINT_SETTINGS)
    if footprint_settings:
        settings.update(footprint_settings)
    parameters = merged_camera_parameters(camera_parameters)
    width, height = parameters["image_size_px"]
    corners = numpy.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])

    fractions = numpy.arange(int(settings["points_per_edge"])) / float(settings["points_per_edge"])
    border = numpy.vstack([corners[index] + fractions[:, None] * (corners[(index + 1) % 4] - corners[index]) for index in range(4)])

    camera_position, rays, _ = viewing_rays_in_base(border, joints, parameters, is_clamped_to_field = True)
    with numpy.errstate(divide = "ignore", invalid = "ignore"):
        distances = (plane_height_m - camera_position[2]) / rays[:, 2]
    hits = camera_position[None, :2] + distances[:, None] * rays[:, :2]
    horizontal_lengths = numpy.maximum(numpy.hypot(rays[:, 0], rays[:, 1]), 1e-12)
    far_points = camera_position[None, :2] + settings["maximum_range_m"] * rays[:, :2] / horizontal_lengths[:, None]
    hit_ranges = numpy.hypot(hits[:, 0] - camera_position[0], hits[:, 1] - camera_position[1])
    is_usable_hit = numpy.isfinite(distances) & (distances > 0.0) & (hit_ranges <= settings["maximum_range_m"])
    return numpy.where(is_usable_hit[:, None], hits, far_points)



def describe_camera(camera_parameters: Optional[Dict] = None) -> Dict:
    """
    Return the camera's pose and lens in quantities that can be checked against the robot by eye.

    Parameters
    ----------
    camera_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``.

    Returns
    -------
    dict
        "distance_from_forearm_axis_mm", "position_along_forearm_mm" (from the joint-4 origin towards the wrist),
        "angle_round_forearm_deg" (from the forearm frame's +x towards +y), "optical_axis_in_forearm",
        "optical_axis_from_forearm_axis_deg", "picture_up_in_forearm", "picture_roll_from_forearm_x_deg" (the angle of the
        picture's up direction, projected across the forearm, from +x), "horizontal_field_of_view_deg" and
        "vertical_field_of_view_deg" (through the principal point, to the picture's edges), and
        "largest_valid_radius_px" (how far from the principal point the lens model reaches).
    """

    parameters = merged_camera_parameters(camera_parameters)
    rotation = rotation_matrix_from_vector(parameters["rotation_vector_in_forearm_rad"])
    translation = numpy.asarray(parameters["translation_in_forearm_m"], dtype = float)
    optical_axis = rotation[:, 2]
    picture_up = -rotation[:, 1]
    focal = parameters["focal_length_px"]
    principal_u, principal_v = parameters["principal_point_px"]
    width, height = parameters["image_size_px"]

    edges = numpy.array([[0.0, principal_v], [width - 1.0, principal_v], [principal_u, 0.0], [principal_u, height - 1.0]])
    normalised, _ = undistort_normalised((edges - numpy.array([principal_u, principal_v])) / focal, parameters["radial_distortion_k1"])
    angles = numpy.degrees(numpy.arctan(numpy.abs(normalised)))
    _, largest_distorted = largest_valid_radius(parameters["radial_distortion_k1"])

    return {
        "distance_from_forearm_axis_mm": 1000.0 * float(math.hypot(translation[0], translation[1])),
        "position_along_forearm_mm": 1000.0 * float(translation[2]),
        "angle_round_forearm_deg": math.degrees(math.atan2(translation[1], translation[0])),
        "optical_axis_in_forearm": [float(value) for value in optical_axis],
        "optical_axis_from_forearm_axis_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(optical_axis[2]))))),
        "picture_up_in_forearm": [float(value) for value in picture_up],
        "picture_roll_from_forearm_x_deg": math.degrees(math.atan2(picture_up[1], picture_up[0])),
        "horizontal_field_of_view_deg": float(angles[0, 0] + angles[1, 0]),
        "vertical_field_of_view_deg": float(angles[2, 1] + angles[3, 1]),
        "largest_valid_radius_px": focal * largest_distorted if math.isfinite(largest_distorted) else math.inf
    }



def known_point_sample(
        joints: Sequence[float],
        pixel: Sequence[float],
        point_base_m: Sequence[float],
        pixel_sigma_px: Optional[float] = None
) -> Dict:
    """
    Build a fit sample for a point whose position in the base frame is known.

    Parameters
    ----------
    joints : sequence of float
        The six joint angles in radians when the picture was taken.
    pixel : sequence of float
        Where the point was seen (u, v).
    point_base_m : sequence of float
        The point's (x, y, z) in metres.
    pixel_sigma_px : float, optional
        This observation's pixel uncertainty, if it differs from the fit's default.

    Returns
    -------
    dict
        The sample, for ``fit_camera_parameters``.
    """

    sample = {"kind": "known_point", "joints": [float(value) for value in joints], "pixel": [float(value) for value in pixel],
              "point_base_m": [float(value) for value in point_base_m]}
    if pixel_sigma_px is not None:
        sample["pixel_sigma_px"] = float(pixel_sigma_px)
    return sample



def plane_point_group_sample(
        plane_height_m: float,
        joint_sets: Sequence[Sequence[float]],
        pixels: Sequence[Sequence[float]],
        initial_xy_m: Optional[Sequence[float]] = None,
        pixel_sigma_px: Optional[float] = None
) -> Dict:
    """
    Build a fit sample for one unknown point on a known horizontal plane, seen in several pictures.

    Parameters
    ----------
    plane_height_m : float
        The plane's height in the base frame: for the puck detector's centre the table's height there plus
        PUCK_TRACKING_PARAMETERS["puck_point_height_above_table_m"] (0.0145 m, as in the fit of 6.11.26); for a known
        point of the puck's top face plus 0.029 m; for a sheet corner or a point of the line the table's height.
    joint_sets : sequence of sequence of float
        The six joint angles of each picture, in radians.
    pixels : sequence of sequence of float
        Where the point was seen in each picture (u, v), in the same order.
    initial_xy_m : sequence of float, optional
        A starting position for the point; without it the fit starts from the initial camera's estimate.
    pixel_sigma_px : float, optional
        The pixel uncertainty of these observations, if it differs from the fit's default.

    Returns
    -------
    dict
        The sample, for ``fit_camera_parameters``.
    """

    if len(joint_sets) != len(pixels):
        raise ValueError(f"{len(joint_sets)} joint sets but {len(pixels)} pixels")
    sample = {"kind": "plane_point_group", "plane_height_m": float(plane_height_m),
              "observations": [{"joints": [float(value) for value in joints], "pixel": [float(value) for value in pixel]}
                               for joints, pixel in zip(joint_sets, pixels)]}
    if initial_xy_m is not None:
        sample["initial_xy_m"] = [float(value) for value in initial_xy_m]
    if pixel_sigma_px is not None:
        sample["pixel_sigma_px"] = float(pixel_sigma_px)
    return sample



class ForearmCameraFit:

    def __init__(
            self,
            samples: Sequence[Dict],
            initial_parameters: Optional[Dict] = None,
            fit_settings: Optional[Dict] = None
    ):
        """
        Prepare a robust least-squares fit of the forearm camera's parameters.

        Two kinds of sample are accepted, in any mix:

        ``{"kind": "known_point", "joints": [6 angles], "pixel": [u, v], "point_base_m": [x, y, z]}``
            A point whose position in the base frame is known, seen at one pixel.

        ``{"kind": "plane_point_group", "plane_height_m": z, "observations": [{"joints": [...], "pixel": [u, v]},
        ...]}``
            One point whose position is unknown but which lies on the horizontal plane z (for the puck detector's
            centre, as in the fit of 6.11.26: the table's height plus 0.0145 m, PUCK_TRACKING_PARAMETERS
            ["puck_point_height_above_table_m"]), seen in several pictures. Its x and y are fitted together with the
            camera.
            An optional "initial_xy_m" gives a starting position; otherwise it is found from the first pictures with
            the initial camera.

        Either kind may carry "pixel_sigma_px" to weight it differently from ``fit_settings["pixel_sigma_px"]``.

        Parameters
        ----------
        samples : sequence of dict
            The observations.
        initial_parameters : dict, optional
            Overrides for ``FOREARM_CAMERA_PARAMETERS``: the starting point of the fit, and the values the fixed
            parameters keep.
        fit_settings : dict, optional
            Overrides for ``CAMERA_FIT_SETTINGS``.
        """

        self.samples = list(samples)
        self.initial_parameters = merged_camera_parameters(initial_parameters)

        self.__initialize_variables()
        self.update_fit_settings(fit_settings)
        self.__prepare_parameters()
        self.__prepare_observations()


    def __initialize_variables(self):

        self.fit_settings = copy.deepcopy(CAMERA_FIT_SETTINGS)
        self.free_parameter_names = []
        self.initial_values = {}
        self.initial_rotation = numpy.eye(3)
        self.prior_items = []


    def update_fit_settings(self, fit_settings: Optional[Dict]):
        """
        Merge overrides into the current fit settings.

        Parameters
        ----------
        fit_settings : dict or None
            Entries to replace (see CAMERA_FIT_SETTINGS); dictionaries are merged key by key.
        """

        if fit_settings:
            for name, value in fit_settings.items():
                if isinstance(value, dict) and isinstance(self.fit_settings.get(name), dict):
                    self.fit_settings[name].update(value)
                else:
                    self.fit_settings[name] = value


    def __prepare_parameters(self):

        unknown_fixed = [name for name in self.fit_settings["fixed_parameters"] if name not in FIT_PARAMETER_NAMES]
        unknown_priors = [name for name in self.fit_settings["prior_sigmas"] if name not in FIT_PARAMETER_NAMES]
        if unknown_fixed or unknown_priors:
            raise ValueError(f"unknown parameter names {unknown_fixed + unknown_priors}; use names from FIT_PARAMETER_NAMES")
        if self.fit_settings["loss"] not in ("huber", "cauchy", "soft_l1", "linear"):
            raise ValueError(f"loss {self.fit_settings['loss']!r} is not one of huber, cauchy, soft_l1, linear")

        self.free_parameter_names = [name for name in FIT_PARAMETER_NAMES if name not in self.fit_settings["fixed_parameters"]]
        translation = self.initial_parameters["translation_in_forearm_m"]
        principal_point = self.initial_parameters["principal_point_px"]
        self.initial_values = {
            "camera_tilt_correction_rad": 0.0,
            "camera_pan_correction_rad": 0.0,
            "camera_roll_correction_rad": 0.0,
            "translation_x_m": float(translation[0]),
            "translation_y_m": float(translation[1]),
            "translation_z_m": float(translation[2]),
            "focal_length_px": float(self.initial_parameters["focal_length_px"]),
            "principal_point_u_px": float(principal_point[0]),
            "principal_point_v_px": float(principal_point[1]),
            "radial_distortion_k1": float(self.initial_parameters["radial_distortion_k1"])
        }
        self.initial_rotation = rotation_matrix_from_vector(self.initial_parameters["rotation_vector_in_forearm_rad"])
        self.prior_items = [(name, float(sigma)) for name, sigma in self.fit_settings["prior_sigmas"].items()
                            if name in self.free_parameter_names]


    def __prepare_observations(self):

        known_samples, group_samples = [], []
        for sample in self.samples:
            if sample.get("kind") == "known_point":
                known_samples.append(sample)
            elif sample.get("kind") == "plane_point_group":
                group_samples.append(sample)
            else:
                raise ValueError(f"sample kind {sample.get('kind')!r} is neither 'known_point' nor 'plane_point_group'")

        default_sigma = float(self.fit_settings["pixel_sigma_px"])
        self.known_count = len(known_samples)
        self.known_points = numpy.array([sample["point_base_m"] for sample in known_samples], dtype = float).reshape(-1, 3)
        self.known_pixels = numpy.array([sample["pixel"] for sample in known_samples], dtype = float).reshape(-1, 2)
        self.known_sigmas = numpy.array([sample.get("pixel_sigma_px", default_sigma) for sample in known_samples], dtype = float)
        self.known_forearm_positions, self.known_forearm_rotations = forearm_frames_in_base([sample["joints"] for sample in known_samples])

        self.group_count = len(group_samples)
        self.group_heights = numpy.array([sample["plane_height_m"] for sample in group_samples], dtype = float)
        observation_joints, observation_pixels, observation_groups, observation_sigmas = [], [], [], []
        for group_index, sample in enumerate(group_samples):
            if len(sample["observations"]) == 0:
                raise ValueError(f"plane_point_group {group_index} has no observations")
            for observation in sample["observations"]:
                observation_joints.append(observation["joints"])
                observation_pixels.append(observation["pixel"])
                observation_groups.append(group_index)
                observation_sigmas.append(observation.get("pixel_sigma_px", sample.get("pixel_sigma_px", default_sigma)))
        self.group_observation_count = len(observation_pixels)
        self.group_pixels = numpy.array(observation_pixels, dtype = float).reshape(-1, 2)
        self.group_of_observation = numpy.array(observation_groups, dtype = int)
        self.group_sigmas = numpy.array(observation_sigmas, dtype = float)
        self.group_forearm_positions, self.group_forearm_rotations = forearm_frames_in_base(observation_joints)

        self.group_initial_xy = numpy.zeros((self.group_count, 2))
        for group_index, sample in enumerate(group_samples):
            if sample.get("initial_xy_m") is not None:
                self.group_initial_xy[group_index] = sample["initial_xy_m"]
                continue
            guesses = [pixels_to_plane(numpy.array([observation["pixel"]], dtype = float), observation["joints"],
                                       sample["plane_height_m"], self.initial_parameters)[0]
                       for observation in sample["observations"][:5]]
            guesses = numpy.array([guess for guess in guesses if numpy.all(numpy.isfinite(guess))])
            if len(guesses) == 0:
                raise ValueError(f"plane_point_group {group_index}: no starting position; give 'initial_xy_m'")
            self.group_initial_xy[group_index] = numpy.median(guesses, axis = 0)

        if self.known_count + self.group_observation_count == 0:
            raise ValueError("no observations to fit")


    def __initial_vector(self) -> numpy.ndarray:

        camera_values = [self.initial_values[name] for name in self.free_parameter_names]
        return numpy.concatenate([numpy.array(camera_values, dtype = float), self.group_initial_xy.ravel()])


    def __values_from_vector(self, parameter_vector: numpy.ndarray) -> Dict:

        values = dict(self.initial_values)
        for index, name in enumerate(self.free_parameter_names):
            values[name] = float(parameter_vector[index])
        return values


    def __camera_from_values(self, values: Dict) -> Tuple[numpy.ndarray, numpy.ndarray, float, numpy.ndarray, float]:

        correction = numpy.array([values["camera_tilt_correction_rad"], values["camera_pan_correction_rad"], values["camera_roll_correction_rad"]])
        rotation = self.initial_rotation @ rotation_matrix_from_vector(correction)
        translation = numpy.array([values["translation_x_m"], values["translation_y_m"], values["translation_z_m"]])
        principal_point = numpy.array([values["principal_point_u_px"], values["principal_point_v_px"]])
        return rotation, translation, values["focal_length_px"], principal_point, values["radial_distortion_k1"]


    def __pixel_errors(self, parameter_vector: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Predicted minus observed pixels for the known points and the group observations, in pixels."""

        values = self.__values_from_vector(parameter_vector)
        rotation, translation, focal, principal_point, k1 = self.__camera_from_values(values)

        known_errors = numpy.zeros((0, 2))
        if self.known_count:
            pixels, _, is_valid = project_with_forearm_frames(self.known_points, self.known_forearm_positions,
                                                              self.known_forearm_rotations, rotation, translation,
                                                              focal, principal_point, k1)
            known_errors = pixels - self.known_pixels
            known_errors[~is_valid] = self.fit_settings["invalid_projection_residual_px"]

        group_errors = numpy.zeros((0, 2))
        if self.group_observation_count:
            group_xy = parameter_vector[len(self.free_parameter_names):].reshape(-1, 2)
            points = numpy.column_stack([group_xy[self.group_of_observation], self.group_heights[self.group_of_observation]])
            pixels, _, is_valid = project_with_forearm_frames(points, self.group_forearm_positions, self.group_forearm_rotations,
                                                              rotation, translation, focal, principal_point, k1)
            group_errors = pixels - self.group_pixels
            group_errors[~is_valid] = self.fit_settings["invalid_projection_residual_px"]

        return known_errors, group_errors


    def __residuals(self, parameter_vector: numpy.ndarray) -> numpy.ndarray:
        """All residuals in units of their sigma: known points, group observations (u and v each), then priors."""

        known_errors, group_errors = self.__pixel_errors(parameter_vector)
        values = self.__values_from_vector(parameter_vector)
        prior_residuals = [(values[name] - self.initial_values[name]) / sigma for name, sigma in self.prior_items]
        return numpy.concatenate([
            (known_errors / self.known_sigmas[:, None]).ravel(),
            (group_errors / self.group_sigmas[:, None]).ravel(),
            numpy.array(prior_residuals, dtype = float)
        ])


    def __jacobian(self, parameter_vector: numpy.ndarray) -> numpy.ndarray:
        """
        Central-difference Jacobian of the residuals. Each group's position moves only that group's rows, so all
        groups' x (then all groups' y) are stepped at once and the result is sorted into the right columns.
        """

        steps = self.fit_settings["finite_difference_steps"]
        camera_column_count = len(self.free_parameter_names)
        residual_count = 2 * (self.known_count + self.group_observation_count) + len(self.prior_items)
        jacobian = numpy.zeros((residual_count, len(parameter_vector)))

        for column, name in enumerate(self.free_parameter_names):
            step = steps[name]
            ahead = parameter_vector.copy()
            behind = parameter_vector.copy()
            ahead[column] += step
            behind[column] -= step
            jacobian[:, column] = (self.__residuals(ahead) - self.__residuals(behind)) / (2.0 * step)

        if self.group_count:
            step = steps["group_xy_m"]
            first_group_row = 2 * self.known_count
            observation_rows = first_group_row + 2 * numpy.arange(self.group_observation_count)
            for axis in (0, 1):
                ahead = parameter_vector.copy()
                behind = parameter_vector.copy()
                ahead[camera_column_count + axis::2] += step
                behind[camera_column_count + axis::2] -= step
                difference = (self.__residuals(ahead) - self.__residuals(behind)) / (2.0 * step)
                columns = camera_column_count + 2 * self.group_of_observation + axis
                jacobian[observation_rows, columns] = difference[observation_rows]
                jacobian[observation_rows + 1, columns] = difference[observation_rows + 1]

        return jacobian


    def __observation_weights(self, residual_vector: numpy.ndarray) -> numpy.ndarray:
        """Robust (IRLS) weight of every residual: one per observation from its 2D error, 1 for the priors."""

        threshold = float(self.fit_settings["loss_threshold_sigma"])
        observation_count = self.known_count + self.group_observation_count
        norms = numpy.hypot(*residual_vector[:2 * observation_count].reshape(-1, 2).T) if observation_count else numpy.zeros(0)
        loss = self.fit_settings["loss"]
        if loss == "huber":
            weights = numpy.where(norms <= threshold, 1.0, threshold / numpy.maximum(norms, 1e-300))
        elif loss == "cauchy":
            weights = 1.0 / (1.0 + (norms / threshold) ** 2)
        elif loss == "soft_l1":
            weights = 1.0 / numpy.sqrt(1.0 + (norms / threshold) ** 2)
        else:
            weights = numpy.ones_like(norms)
        return numpy.concatenate([numpy.repeat(weights, 2), numpy.ones(len(self.prior_items))])


    def __robust_cost(self, residual_vector: numpy.ndarray) -> float:
        """Half the sum of the robust loss over the observations' 2D errors, plus half the squared priors."""

        threshold = float(self.fit_settings["loss_threshold_sigma"])
        observation_count = self.known_count + self.group_observation_count
        squared = (residual_vector[:2 * observation_count].reshape(-1, 2) ** 2).sum(axis = 1)
        loss = self.fit_settings["loss"]
        if loss == "huber":
            norms = numpy.sqrt(squared)
            costs = numpy.where(norms <= threshold, squared, 2.0 * threshold * norms - threshold ** 2)
        elif loss == "cauchy":
            costs = threshold ** 2 * numpy.log1p(squared / threshold ** 2)
        elif loss == "soft_l1":
            costs = 2.0 * threshold ** 2 * (numpy.sqrt(1.0 + squared / threshold ** 2) - 1.0)
        else:
            costs = squared
        priors = residual_vector[2 * observation_count:]
        return 0.5 * float(costs.sum() + (priors ** 2).sum())


    def __solve_with_scipy(self, start: numpy.ndarray) -> Tuple[numpy.ndarray, Dict]:

        from scipy import optimize

        result = optimize.least_squares(
            self.__residuals,
            start,
            jac = self.__jacobian,
            loss = self.fit_settings["loss"],
            f_scale = float(self.fit_settings["loss_threshold_sigma"]),
            x_scale = "jac",
            method = "trf",
            max_nfev = 5 * int(self.fit_settings["maximum_iterations"]),
            ftol = 1e-12,
            xtol = 1e-12,
            gtol = 1e-12
        )
        return result.x, {"solver": "scipy", "is_converged": bool(result.status > 0), "message": str(result.message),
                          "iterations": int(result.nfev)}


    def __solve_with_numpy(self, start: numpy.ndarray) -> Tuple[numpy.ndarray, Dict]:
        """Levenberg-Marquardt on iteratively reweighted least squares, with the robust cost deciding each step."""

        vector = start.copy()
        residuals = self.__residuals(vector)
        cost = self.__robust_cost(residuals)
        damping = 1e-3
        message = "stopped at the iteration limit"
        is_converged = False
        iteration = 0

        for iteration in range(1, int(self.fit_settings["maximum_iterations"]) + 1):
            weights = self.__observation_weights(residuals)
            jacobian = self.__jacobian(vector)
            normal_matrix = jacobian.T @ (weights[:, None] * jacobian)
            gradient = jacobian.T @ (weights * residuals)
            diagonal = numpy.maximum(numpy.diag(normal_matrix), 1e-12)

            is_step_accepted = False
            for _ in range(20):
                damped = normal_matrix + damping * numpy.diag(diagonal)
                try:
                    step = -numpy.linalg.solve(damped, gradient)
                except numpy.linalg.LinAlgError:
                    step = -numpy.linalg.lstsq(damped, gradient, rcond = None)[0]
                candidate = vector + step
                candidate_residuals = self.__residuals(candidate)
                candidate_cost = self.__robust_cost(candidate_residuals)
                if candidate_cost < cost:
                    is_step_accepted = True
                    break
                damping *= 4.0

            if not is_step_accepted:
                message = "no step lowers the cost any more"
                is_converged = True
                break

            cost_decrease = cost - candidate_cost
            scaled_step = float(numpy.max(numpy.abs(step) * numpy.sqrt(diagonal)))
            vector, residuals, cost = candidate, candidate_residuals, candidate_cost
            damping = max(damping / 3.0, 1e-12)
            if cost_decrease <= 1e-12 * max(cost, 1e-12) or scaled_step < 1e-9:
                message = "the cost and the step stopped changing"
                is_converged = True
                break

        return vector, {"solver": "numpy", "is_converged": is_converged, "message": message, "iterations": iteration}


    def __conditioning_report(self, parameter_vector: numpy.ndarray) -> Dict:
        """
        How well the data fix each camera parameter: standard deviations, correlations and the singular values of the
        camera's part of the problem after the unknown group positions are eliminated (Schur complement).
        """

        residuals = self.__residuals(parameter_vector)
        weights = self.__observation_weights(residuals)
        jacobian = self.__jacobian(parameter_vector)
        normal_matrix = jacobian.T @ (weights[:, None] * jacobian)
        camera_count = len(self.free_parameter_names)
        degrees_of_freedom = max(len(residuals) - len(parameter_vector), 1)
        variance_factor = float((weights * residuals ** 2).sum() / degrees_of_freedom)

        camera_block = normal_matrix[:camera_count, :camera_count]
        coupling = normal_matrix[:camera_count, camera_count:]
        group_inverses = []
        schur = camera_block.copy()
        for group_index in range(self.group_count):
            columns = slice(2 * group_index, 2 * group_index + 2)
            block = normal_matrix[camera_count:, camera_count:][columns, columns]
            block_inverse = numpy.linalg.pinv(block)
            group_inverses.append(block_inverse)
            schur -= coupling[:, columns] @ block_inverse @ coupling[:, columns].T

        report = {"parameter_names": list(self.free_parameter_names), "variance_factor": variance_factor}
        if camera_count == 0:
            return report

        scale = 1.0 / numpy.sqrt(numpy.maximum(numpy.diag(schur), 1e-300))
        scaled = scale[:, None] * schur * scale[None, :]
        eigenvalues, eigenvectors = numpy.linalg.eigh(scaled)
        eigenvalues = numpy.maximum(eigenvalues, 0.0)
        singular_values = numpy.sqrt(eigenvalues[::-1])
        is_rank_deficient = bool(eigenvalues[0] <= 1e-12 * eigenvalues[-1])

        covariance = numpy.linalg.pinv(schur) if is_rank_deficient else numpy.linalg.inv(schur)
        standard_deviations = numpy.sqrt(numpy.maximum(numpy.diag(covariance), 0.0))
        if is_rank_deficient:
            standard_deviations[:] = numpy.inf
        correlations = covariance / numpy.outer(numpy.maximum(standard_deviations, 1e-300), numpy.maximum(standard_deviations, 1e-300))
        alone = 1.0 / numpy.sqrt(numpy.maximum(numpy.diag(schur), 1e-300))

        weakest = eigenvectors[:, 0]
        weakest = weakest / numpy.max(numpy.abs(weakest))
        strong_pairs = []
        for first in range(camera_count):
            for second in range(first + 1, camera_count):
                if abs(correlations[first, second]) >= self.fit_settings["strong_correlation"]:
                    strong_pairs.append((self.free_parameter_names[first], self.free_parameter_names[second], float(correlations[first, second])))

        group_deviations = numpy.zeros((self.group_count, 2))
        if not is_rank_deficient:
            for group_index, block_inverse in enumerate(group_inverses):
                columns = slice(2 * group_index, 2 * group_index + 2)
                link = block_inverse @ coupling[:, columns].T
                group_covariance = block_inverse + link @ covariance @ link.T
                group_deviations[group_index] = numpy.sqrt(numpy.maximum(numpy.diag(group_covariance), 0.0))
        else:
            group_deviations[:] = numpy.inf

        report.update({
            "standard_deviations": dict(zip(self.free_parameter_names, standard_deviations.tolist())),
            "standard_deviations_a_posteriori": dict(zip(self.free_parameter_names, (standard_deviations * math.sqrt(variance_factor)).tolist())),
            "standard_deviations_if_alone": dict(zip(self.free_parameter_names, alone.tolist())),
            "correlations": correlations,
            "strong_correlations": strong_pairs,
            "scaled_singular_values": singular_values.tolist(),
            "condition_number": float(singular_values[0] / max(singular_values[-1], 1e-300)),
            "is_rank_deficient": is_rank_deficient,
            "weakest_direction": dict(zip(self.free_parameter_names, weakest.tolist())),
            "group_point_standard_deviations_m": group_deviations
        })
        return report


    def run(self) -> Dict:
        """
        Fit the camera and return the parameters, the residuals and the conditioning report.

        Returns
        -------
        dict
            "parameters": the fitted camera, ready for ``project_points`` and ``pixel_to_table``;
            "fitted_values": every adjustable quantity by name; "free_parameter_names";
            "known_point_residuals_px" (K, 2) and "group_residuals_px" (list of (n, 2)), predicted minus observed;
            "group_points_xy_m" (G, 2); "observation_weights" (robust weights, known points then group observations);
            "rms_px", "median_error_px", "outlier_count" (errors above ``outlier_threshold_sigma`` sigmas);
            "solver", "is_converged", "message", "iterations"; and "conditioning" (see the conditioning report).
        """

        start = self.__initial_vector()
        solver = self.fit_settings["solver"]
        if solver == "auto":
            try:
                import scipy.optimize
                solver = "scipy"
            except ImportError:
                solver = "numpy"
        if solver == "scipy":
            vector, solver_information = self.__solve_with_scipy(start)
        else:
            vector, solver_information = self.__solve_with_numpy(start)

        values = self.__values_from_vector(vector)
        rotation, translation, focal, principal_point, k1 = self.__camera_from_values(values)
        parameters = copy.deepcopy(self.initial_parameters)
        parameters.update({
            "rotation_vector_in_forearm_rad": rotation_vector_from_matrix(rotation).tolist(),
            "translation_in_forearm_m": translation.tolist(),
            "focal_length_px": float(focal),
            "principal_point_px": principal_point.tolist(),
            "radial_distortion_k1": float(k1)
        })

        known_errors, group_errors = self.__pixel_errors(vector)
        all_errors = numpy.vstack([known_errors, group_errors])
        error_norms = numpy.hypot(all_errors[:, 0], all_errors[:, 1])
        sigmas = numpy.concatenate([self.known_sigmas, self.group_sigmas])
        weights = self.__observation_weights(self.__residuals(vector))[:2 * len(error_norms):2]
        group_residuals = [group_errors[self.group_of_observation == group_index] for group_index in range(self.group_count)]

        summary = {
            "parameters": parameters,
            "fitted_values": values,
            "free_parameter_names": list(self.free_parameter_names),
            "known_point_residuals_px": known_errors,
            "group_residuals_px": group_residuals,
            "group_points_xy_m": vector[len(self.free_parameter_names):].reshape(-1, 2).copy(),
            "observation_weights": weights,
            "rms_px": float(numpy.sqrt((error_norms ** 2).mean())) if len(error_norms) else 0.0,
            "median_error_px": float(numpy.median(error_norms)) if len(error_norms) else 0.0,
            "outlier_count": int((error_norms > self.fit_settings["outlier_threshold_sigma"] * sigmas).sum()),
            "conditioning": self.__conditioning_report(vector)
        }
        summary.update(solver_information)
        return summary



def fit_camera_parameters(
        samples: Sequence[Dict],
        initial_parameters: Optional[Dict] = None,
        fit_settings: Optional[Dict] = None
) -> Dict:
    """
    Fit the forearm camera's parameters to pixels seen at known joint angles, robustly.

    Parameters
    ----------
    samples : sequence of dict
        Known points and groups of views of one unknown point on a known horizontal plane; see ``ForearmCameraFit``.
    initial_parameters : dict, optional
        Overrides for ``FOREARM_CAMERA_PARAMETERS``: the starting point, and the values of the fixed parameters.
    fit_settings : dict, optional
        Overrides for ``CAMERA_FIT_SETTINGS`` ("fixed_parameters", "loss", "pixel_sigma_px", "prior_sigmas", ...).

    Returns
    -------
    dict
        See ``ForearmCameraFit.run``.
    """

    return ForearmCameraFit(samples, initial_parameters, fit_settings).run()
