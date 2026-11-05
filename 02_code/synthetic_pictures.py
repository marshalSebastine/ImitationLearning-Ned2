"""
Synthetic camera pictures for imitation learning, rendered with the Webots simulator.

Emranul chose on 5.11.26 to make the pictures in Webots. This module holds what the simulator needs from the project's
own measurements: the traced centre lines of the course sheets drawn as page textures, where a page lies on the table,
a wood texture for the table, and the command that runs a Webots world without its window in the way. Settings come
from ``config.SYNTHETIC_PARAMETERS``.

Write the textures, render the first-look pictures without a window, or open the scene in the Webots window:

    python 02_code/synthetic_pictures.py textures
    python 02_code/synthetic_pictures.py first_look
    python 02_code/synthetic_pictures.py open
"""

import csv
import json
import math
import os
import pathlib
import subprocess
import sys

import cv2
import numpy

import config

REPOSITORY_FOLDER = pathlib.Path(__file__).resolve().parents[1]
# Set by ``run_webots_world`` so the controller quits Webots after its pictures; opened in the window, the scene stays.
QUIT_WHEN_DONE_VARIABLE = "SYNTHETIC_PICTURES_QUIT_WHEN_DONE"
# Names the job file that the twin_replay controller renders.
TWIN_JOB_VARIABLE = "SYNTHETIC_PICTURES_TWIN_JOB"
# The twin's Webots camera renders a wide picture without distortion (100 degrees across 1024 pixels, so it covers
# every ray of the real 640 x 480 picture); Niryo's distortion is then applied in Python, exactly as calibrated.
WIDE_PICTURE = {"width": 1024, "height": 768, "field_of_view_rad": math.radians(100.0)}
# The Ned2's joints from Niryo's URDF (06_deliverables/research_notes/Camera mount options/niryo_files/
# niryo_ned2.urdf.xacro): for each joint, its origin in the parent frame (metres), the fixed roll, pitch and yaw before
# it turns, and whether it turns about its own z axis; the last entry is the fixed tool frame on the hand. Checked on
# 5.11.26 against every step of the episodes of 3 to 5.11.26: the tool position agrees with the robot's own reading to
# 0.01 mm (median), 1.9 mm at the 95th percentile (joints and pose are read at slightly different moments).
NED2_KINEMATIC_CHAIN = [
    {"origin_m": (0.0, 0.0, 0.1013), "rpy_rad": (0.0, 0.0, 0.0), "is_turning": True},
    {"origin_m": (0.0, 0.0, 0.065), "rpy_rad": (math.pi / 2, 0.0, 0.0), "is_turning": True},
    {"origin_m": (0.012, 0.221, 0.0), "rpy_rad": (0.0, 0.0, math.pi / 2), "is_turning": True},
    {"origin_m": (0.0325, -0.065, 0.0), "rpy_rad": (math.pi / 2, 0.0, 0.0), "is_turning": True},
    {"origin_m": (0.0, 0.0, 0.17), "rpy_rad": (-math.pi / 2, 0.0, 0.0), "is_turning": True},
    {"origin_m": (0.00925, -0.0197, 0.0), "rpy_rad": (math.pi / 2, 0.0, 0.0), "is_turning": True},
    {"origin_m": (0.0, 0.0, 0.0285), "rpy_rad": (math.pi, -math.pi / 2, 0.0), "is_turning": False}
]


def rotation_from_rpy(
        roll: float,
        pitch: float,
        yaw: float
) -> numpy.ndarray:
    """
    Return the rotation matrix of URDF roll, pitch and yaw angles (fixed axes x, then y, then z).

    Parameters
    ----------
    roll, pitch, yaw : float
        Angles in radians.

    Returns
    -------
    numpy.ndarray
        A 3 by 3 rotation matrix, ``Rz(yaw) Ry(pitch) Rx(roll)``.
    """

    cos_roll, sin_roll = math.cos(roll), math.sin(roll)
    cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    about_x = numpy.array([[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]])
    about_y = numpy.array([[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]])
    about_z = numpy.array([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]])
    return about_z @ about_y @ about_x


def forward_kinematics(joints: list) -> list:
    """
    Return the pose of every Ned2 frame for six joint angles, from Niryo's own description of the arm.

    Parameters
    ----------
    joints : list of float
        The six joint angles in radians, as the robot logs them.

    Returns
    -------
    list of numpy.ndarray
        Seven 4 by 4 transforms in the base frame: the frames after joints 1 to 6 (shoulder, arm, elbow, forearm,
        wrist, hand) and the tool frame, whose origin is the position the robot reports.
    """

    transform = numpy.eye(4)
    frames = []
    joint_index = 0
    for link in NED2_KINEMATIC_CHAIN:
        step = numpy.eye(4)
        step[:3, :3] = rotation_from_rpy(*link["rpy_rad"])
        step[:3, 3] = link["origin_m"]
        transform = transform @ step
        if link["is_turning"]:
            turn = numpy.eye(4)
            turn[:3, :3] = rotation_from_rpy(0.0, 0.0, joints[joint_index])
            transform = transform @ turn
            joint_index += 1
        frames.append(transform.copy())
    return frames


def forearm_camera_pose(
        joints: list,
        mount: dict
) -> numpy.ndarray:
    """
    Return the pose of the camera taped to the forearm cover, for six joint angles.

    The camera turns with joint 4, so its pose is the forearm frame followed by the fixed mount. The camera's axes
    follow OpenCV: x to the picture's right, y down the picture, z along the view.

    Parameters
    ----------
    joints : list of float
        The six joint angles in radians.
    mount : dict
        ``rotation_vector`` (radians) and ``translation_m`` of the camera in the forearm frame.

    Returns
    -------
    numpy.ndarray
        The camera's 4 by 4 pose in the robot's base frame.
    """

    mount_transform = numpy.eye(4)
    mount_transform[:3, :3] = cv2.Rodrigues(numpy.array(mount["rotation_vector"], dtype = float))[0]
    mount_transform[:3, 3] = mount["translation_m"]
    return forward_kinematics(joints)[3] @ mount_transform


def project_to_picture(
        points_world: numpy.ndarray,
        camera_in_world: numpy.ndarray,
        lens: dict
) -> tuple:
    """
    Project points given in the robot's base frame into the camera's picture, with the lens' distortion.

    Parameters
    ----------
    points_world : numpy.ndarray
        Points of shape ``(n, 3)`` in metres.
    camera_in_world : numpy.ndarray
        The camera's 4 by 4 pose, OpenCV axes.
    lens : dict
        ``fx``, ``fy``, ``cx``, ``cy`` in pixels and the radial coefficients ``k1``, ``k2``, ``k3``.

    Returns
    -------
    tuple of numpy.ndarray
        Pixel positions of shape ``(n, 2)`` and whether each point is in front of the camera, inside the zone where
        the distortion model holds, and inside the picture.
    """

    camera_points = (points_world - camera_in_world[:3, 3]) @ camera_in_world[:3, :3]
    depth = camera_points[:, 2]
    safe_depth = numpy.where(depth > 0.02, depth, 1.0)
    normalised_x, normalised_y = camera_points[:, 0] / safe_depth, camera_points[:, 1] / safe_depth
    radius_squared = normalised_x ** 2 + normalised_y ** 2
    radial = 1 + lens["k1"] * radius_squared + lens["k2"] * radius_squared ** 2 + lens["k3"] * radius_squared ** 3
    pixels = numpy.column_stack([lens["fx"] * normalised_x * radial + lens["cx"],
                                 lens["fy"] * normalised_y * radial + lens["cy"]])
    # Niryo's distortion polynomial folds back beyond a normalised radius of about 1.05; stay well inside it.
    is_visible = ((depth > 0.02) & (radius_squared < lens["valid_normalised_radius"] ** 2)
                  & (pixels[:, 0] >= 0) & (pixels[:, 0] < 640) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 480))
    return pixels, is_visible


def webots_pose(transform: numpy.ndarray) -> dict:
    """
    Return a 4 by 4 transform as Webots' translation and axis-angle rotation fields.

    Parameters
    ----------
    transform : numpy.ndarray
        A pose in the robot's base frame, which is Webots' world frame in the twin.

    Returns
    -------
    dict
        ``translation`` (three floats) and ``rotation`` (axis x, y, z and angle).
    """

    rotation_vector = cv2.Rodrigues(transform[:3, :3])[0].ravel()
    angle = float(numpy.linalg.norm(rotation_vector))
    axis = rotation_vector / angle if angle > 1e-9 else numpy.array([0.0, 0.0, 1.0])
    return {"translation": [float(value) for value in transform[:3, 3]],
            "rotation": [float(axis[0]), float(axis[1]), float(axis[2]), angle]}


def twin_poses(
        joints: list,
        mount: dict,
        puck_xy: list = None
) -> dict:
    """
    Return the Webots poses of the twin's camera, arm parts, magnet block and puck for one logged step.

    Webots' camera looks along its own x axis with z up the picture and y to its left, so the OpenCV camera's axes
    (x right, y down, z ahead) are reordered. The arm meshes are placed as Niryo's URDF places them: in each link's
    frame turned by -90 degrees about z. The magnet block hangs below the tool point along the rod.

    Parameters
    ----------
    joints : list of float
        The six logged joint angles in radians.
    mount : dict
        The camera's ``rotation_vector`` and ``translation_m`` in the forearm frame.
    puck_xy : list of float, optional
        Where the puck stands on the table; ``None`` hides it far from the table.

    Returns
    -------
    dict
        Webots poses keyed by the nodes' DEF names in ``forearm_twin.wbt``.
    """

    frames = forward_kinematics(joints)
    camera = forearm_camera_pose(joints, mount)
    rig = numpy.eye(4)
    rig[:3, :3] = numpy.column_stack([camera[:3, 2], -camera[:3, 0], -camera[:3, 1]])
    rig[:3, 3] = camera[:3, 3]

    mesh_turn = numpy.eye(4)
    mesh_turn[:3, :3] = rotation_from_rpy(0.0, 0.0, -math.pi / 2)
    block = frames[6].copy()
    block[:3, 3] = frames[6][:3, 3] + 0.013 * frames[6][:3, 0]

    puck = numpy.eye(4)
    if puck_xy is None:
        puck[:3, 3] = [2.0, 2.0, -1.0]
    else:
        puck[:3, 3] = [puck_xy[0], puck_xy[1],
                       config.table_height(puck_xy[0], puck_xy[1]) + config.PUCK_PARAMETERS["height_m"] / 2]
    return {"CAMERA_RIG": webots_pose(rig), "FOREARM": webots_pose(frames[3] @ mesh_turn),
            "WRIST": webots_pose(frames[4] @ mesh_turn), "HAND": webots_pose(frames[5] @ mesh_turn),
            "MAGNET_BLOCK": webots_pose(block), "PUCK": webots_pose(puck)}


def find_puck(
        picture: numpy.ndarray,
        camera_in_world: numpy.ndarray,
        lens: dict,
        tool_xy: list,
        largest_distance_from_tool_m: float = 0.08
) -> list:
    """
    Return where the puck stands on the table, from its red colour in a real picture, or ``None`` if none is seen.

    Each red blob's centre is taken as a puck's middle (half its height above the paper) and traced back through the
    lens onto that height above the table plane; the blob that lands nearest the tool is the puck, because the
    reddish wood of the table and the door also passes the colour test.

    Parameters
    ----------
    picture : numpy.ndarray
        A real picture of the forearm camera (BGR).
    camera_in_world : numpy.ndarray
        The camera's 4 by 4 pose for that picture, OpenCV axes.
    lens : dict
        The lens calibration (``fx``, ``fy``, ``cx``, ``cy``, ``k1``, ``k2``, ``k3``).
    tool_xy : list of float
        The tool's x and y in metres for that picture.
    largest_distance_from_tool_m : float, optional
        A red blob landing further from the tool is not taken for the puck.

    Returns
    -------
    list of float or None
        The puck centre's x and y in metres.
    """

    hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (((hue < 10) | (hue > 170)) & (saturation > 110) & (value > 50)).astype(numpy.uint8)
    count, _, statistics, centroids = cv2.connectedComponentsWithStats(red)

    camera_matrix = numpy.array([[lens["fx"], 0.0, lens["cx"]], [0.0, lens["fy"], lens["cy"]], [0.0, 0.0, 1.0]])
    distortion = numpy.array([lens["k1"], lens["k2"], 0.0, 0.0, lens["k3"]])
    origin = camera_in_world[:3, 3]
    plane = config.WORKSPACE_PARAMETERS["table_plane"]
    middle_height = plane["c"] + config.PUCK_PARAMETERS["height_m"] / 2
    best_point, best_distance = None, largest_distance_from_tool_m
    for blob in range(1, count):
        if not 300 <= statistics[blob, cv2.CC_STAT_AREA] <= 12_000:
            continue
        ray = cv2.undistortPoints(centroids[blob].reshape(1, 1, 2), camera_matrix, distortion).ravel()
        direction = camera_in_world[:3, :3] @ numpy.array([ray[0], ray[1], 1.0])
        distance = ((middle_height + plane["a"] * origin[0] + plane["b"] * origin[1] - origin[2])
                    / (direction[2] - plane["a"] * direction[0] - plane["b"] * direction[1]))
        point = origin + distance * direction
        distance_from_tool = math.hypot(point[0] - tool_xy[0], point[1] - tool_xy[1])
        if distance > 0 and distance_from_tool < best_distance:
            best_point, best_distance = [float(point[0]), float(point[1])], distance_from_tool
    return best_point


def match_colours(
        picture: numpy.ndarray,
        reference: numpy.ndarray
) -> numpy.ndarray:
    """
    Give ``picture`` the colour mean and spread of ``reference`` in Lab colour space (Reinhard's colour transfer).

    Used to show a twin picture with the real camera's automatic white balance and exposure of the same moment.

    Parameters
    ----------
    picture, reference : numpy.ndarray
        BGR pictures.

    Returns
    -------
    numpy.ndarray
        ``picture`` with the colour statistics of ``reference``.
    """

    picture_lab = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB).astype(numpy.float32)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(numpy.float32)
    picture_mean, picture_spread = picture_lab.mean(axis = (0, 1)), picture_lab.std(axis = (0, 1)) + 1e-3
    reference_mean, reference_spread = reference_lab.mean(axis = (0, 1)), reference_lab.std(axis = (0, 1))
    matched = (picture_lab - picture_mean) / picture_spread * reference_spread + reference_mean
    return cv2.cvtColor(numpy.clip(matched, 0, 255).astype(numpy.uint8), cv2.COLOR_LAB2BGR)

# Reference
# Source: Reinhard, Ashikhmin, Gooch and Shirley - Color transfer between images (mean and standard deviation per
#         channel in a decorrelated colour space), IEEE Computer Graphics and Applications 21(5), 2001.
# Link: https://doi.org/10.1109/38.946629


def distort_wide_picture(
        wide_picture: numpy.ndarray,
        lens: dict
) -> numpy.ndarray:
    """
    Turn the twin's wide undistorted picture into a 640 x 480 picture with the real lens' distortion.

    Parameters
    ----------
    wide_picture : numpy.ndarray
        The Webots camera's picture of ``WIDE_PICTURE`` size, without distortion.
    lens : dict
        The lens calibration (``fx``, ``fy``, ``cx``, ``cy``, ``k1``, ``k2``, ``k3``).

    Returns
    -------
    numpy.ndarray
        The picture the real camera would take, 640 by 480 pixels.
    """

    camera_matrix = numpy.array([[lens["fx"], 0.0, lens["cx"]], [0.0, lens["fy"], lens["cy"]], [0.0, 0.0, 1.0]])
    distortion = numpy.array([lens["k1"], lens["k2"], 0.0, 0.0, lens["k3"]])
    rows, columns = numpy.mgrid[0:480, 0:640]
    pixels = numpy.column_stack([columns.ravel(), rows.ravel()]).astype(numpy.float64)
    rays = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera_matrix, distortion).reshape(-1, 2)
    wide_focal_px = (WIDE_PICTURE["width"] / 2) / math.tan(WIDE_PICTURE["field_of_view_rad"] / 2)
    map_x = (WIDE_PICTURE["width"] / 2 + wide_focal_px * rays[:, 0]).astype(numpy.float32).reshape(480, 640)
    map_y = (WIDE_PICTURE["height"] / 2 + wide_focal_px * rays[:, 1]).astype(numpy.float32).reshape(480, 640)
    return cv2.remap(wide_picture, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)


def write_twin_job(
        episode_folder: pathlib.Path,
        steps: list,
        mount_name: str,
        table_photo: dict,
        output_folder: pathlib.Path
) -> pathlib.Path:
    """
    Write the job file that the twin_replay controller renders: one picture per logged step.

    Parameters
    ----------
    episode_folder : pathlib.Path
        A recorded episode (its ``steps.csv`` and frames are read, never changed).
    steps : list of int
        The step numbers to re-create.
    mount_name : str
        Key of ``config.SYNTHETIC_PARAMETERS["forearm_camera_mounts"]`` for the tape position of that episode.
    table_photo : dict
        ``texture`` (path relative to the world's folder) and ``corners`` (four table corners in metres).
    output_folder : pathlib.Path
        Where the wide twin pictures and the job file go.

    Returns
    -------
    pathlib.Path
        The job file.
    """

    parameters = config.SYNTHETIC_PARAMETERS
    mount = parameters["forearm_camera_mounts"][mount_name]
    lens = parameters["forearm_camera_lens"]
    output_folder.mkdir(parents = True, exist_ok = True)
    with open(episode_folder / "steps.csv", newline = "") as steps_file:
        rows = {int(row["step"]): row for row in csv.DictReader(steps_file)}

    pictures = []
    for step in steps:
        row = rows[step]
        joints = [float(row[f"joint_{index}"]) for index in range(1, 7)]
        real_picture = cv2.imread(str(episode_folder / row["frame"]))
        tool_xy = forward_kinematics(joints)[6][:2, 3]
        puck_xy = find_puck(real_picture, forearm_camera_pose(joints, mount), lens, tool_xy)
        name = f"{episode_folder.name}_step_{step:03d}"
        pictures.append({"name": name, "real_picture": str(episode_folder / row["frame"]),
                         "wide_picture": str(output_folder / f"{name}_wide.png"),
                         "poses": twin_poses(joints, mount, puck_xy)})

    job_path = output_folder / "twin_job.json"
    job_path.write_text(json.dumps({"table_photo": table_photo, "pictures": pictures}, indent = 1))
    return job_path


def run_twin_replay(
        job_path: pathlib.Path,
        timeout_s: float = 600.0
) -> list:
    """
    Render a twin job in Webots without its window in the way, then add the lens' distortion to every picture.

    Parameters
    ----------
    job_path : pathlib.Path
        A job file from ``write_twin_job``.
    timeout_s : float, optional
        How long to wait for Webots.

    Returns
    -------
    list of tuple
        For every picture: the real picture and the twin picture (both BGR, 640 by 480) and the picture's name.
    """

    os.environ[TWIN_JOB_VARIABLE] = str(job_path)
    run_webots_world("forearm_twin", timeout_s = timeout_s)
    lens = config.SYNTHETIC_PARAMETERS["forearm_camera_lens"]
    results = []
    for picture in json.loads(job_path.read_text())["pictures"]:
        wide = cv2.imread(picture["wide_picture"])
        if wide is None:
            continue
        twin_picture = distort_wide_picture(wide, lens)
        cv2.imwrite(picture["wide_picture"].replace("_wide.png", "_twin.png"), twin_picture)
        results.append((cv2.imread(picture["real_picture"]), twin_picture, picture["name"]))
    return results


def load_traced_line(path_id: str) -> numpy.ndarray:
    """
    Return the traced centre line of a course sheet in the frame of the sheet's photo.

    Parameters
    ----------
    path_id : str
        The sheet, for example ``"path_04_meander"``.

    Returns
    -------
    numpy.ndarray
        Points of shape ``(n, 2)`` in centimetres, x towards the photo's right and y towards its top (negative down
        the photo), from the first traced end to the other.
    """

    traced_sheets = json.loads((REPOSITORY_FOLDER / config.SYNTHETIC_PARAMETERS["sheet_paths_file"]).read_text())
    return numpy.array(traced_sheets[path_id.removeprefix("path_")]["points_cm"], dtype = float)


def line_on_page(path_id: str) -> numpy.ndarray:
    """
    Return the traced centre line in page coordinates.

    Parameters
    ----------
    path_id : str
        The sheet, for example ``"path_04_meander"``.

    Returns
    -------
    numpy.ndarray
        Points of shape ``(n, 2)``: centimetres down from the page's top edge, then centimetres right of its left edge.
    """

    corner_x_cm, corner_y_cm = config.SYNTHETIC_PARAMETERS["page_corner_in_photo_cm"]
    photo_points = load_traced_line(path_id)
    return numpy.stack([corner_y_cm - photo_points[:, 1], photo_points[:, 0] - corner_x_cm], axis = 1)


def page_pose_on_table(path_id: str) -> list:
    """
    Return where the page of a sheet lies on the table.

    The line's first traced point is put on the placement's ``line_start_xy``, and the page's downward direction is
    turned by ``page_yaw_rad`` from the table's +x axis, so the page's right-hand direction is a further quarter turn
    counter-clockwise seen from above.

    Parameters
    ----------
    path_id : str
        The sheet, for example ``"path_04_meander"``.

    Returns
    -------
    list of float
        The page centre's x and y in metres and the page's yaw in radians.
    """

    parameters = config.SYNTHETIC_PARAMETERS
    placement = parameters["sheet_placements"][path_id]
    page_width_cm, page_length_cm = parameters["page_size_cm"]

    start_down_cm, start_right_cm = line_on_page(path_id)[0]
    centre_down_cm = page_length_cm / 2 - start_down_cm
    centre_right_cm = page_width_cm / 2 - start_right_cm

    yaw_rad = placement["page_yaw_rad"]
    down_direction = numpy.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    right_direction = numpy.array([-math.sin(yaw_rad), math.cos(yaw_rad)])
    centre_xy = (numpy.array(placement["line_start_xy"])
                 + (centre_down_cm * down_direction + centre_right_cm * right_direction) / 100)
    return [float(centre_xy[0]), float(centre_xy[1]), yaw_rad]


def draw_page_texture(path_id: str) -> numpy.ndarray:
    """
    Draw the A4 page of a sheet as an image: white paper with the traced line at the sheet's line width.

    Parameters
    ----------
    path_id : str
        The sheet, for example ``"path_04_meander"``.

    Returns
    -------
    numpy.ndarray
        A BGR image, rows running down the page and columns to its right, ``texture_pixels_per_cm`` pixels per
        centimetre.
    """

    parameters = config.SYNTHETIC_PARAMETERS
    pixels_per_cm = parameters["texture_pixels_per_cm"]
    page_width_cm, page_length_cm = parameters["page_size_cm"]

    page = numpy.full((round(page_length_cm * pixels_per_cm), round(page_width_cm * pixels_per_cm), 3), 255, numpy.uint8)
    points_in_pixels = numpy.round(line_on_page(path_id)[:, ::-1] * pixels_per_cm).astype(numpy.int32)
    # OpenCV draws thick lines with round ends, which matches the printed sheets' rounded line ends.
    cv2.polylines(page, [points_in_pixels], isClosed = False, color = (38, 36, 36),
                  thickness = round(parameters["line_width_cm"] * pixels_per_cm), lineType = cv2.LINE_AA)
    return page


def draw_table_texture(
        size_pixels: int = 2048,
        seed: int = 69_420
) -> numpy.ndarray:
    """
    Draw a wood-grain texture for the table top.

    The grain is random noise smoothed far more along one axis than the other, coloured between a light and a dark
    tone of the beech-coloured desk seen in the bracket F3 pictures.

    Parameters
    ----------
    size_pixels : int, optional
        Width and height of the square texture.
    seed : int, optional
        Seed for the grain's noise.

    Returns
    -------
    numpy.ndarray
        A BGR image of shape ``(size_pixels, size_pixels, 3)``.
    """

    randomness_generator = numpy.random.default_rng(seed)
    noise = randomness_generator.normal(size = (size_pixels, size_pixels)).astype(numpy.float32)
    grain = cv2.GaussianBlur(noise, (0, 0), sigmaX = 40, sigmaY = 1.2)
    grain = (grain - grain.min()) / (grain.max() - grain.min())

    light_wood = numpy.array([125, 172, 214], dtype = numpy.float32)
    dark_wood = numpy.array([82, 128, 178], dtype = numpy.float32)
    table = dark_wood + (light_wood - dark_wood) * grain[:, :, None]
    return numpy.clip(table, 0, 255).astype(numpy.uint8)


def write_textures() -> None:
    """Write the page textures of every traced sheet and the table texture into the Webots world's texture folder."""

    texture_folder = REPOSITORY_FOLDER / config.SYNTHETIC_PARAMETERS["project_folder"] / "worlds" / "textures"
    texture_folder.mkdir(parents = True, exist_ok = True)

    traced_sheets = json.loads((REPOSITORY_FOLDER / config.SYNTHETIC_PARAMETERS["sheet_paths_file"]).read_text())
    for sheet_key in traced_sheets:
        path_id = f"path_{sheet_key}"
        cv2.imwrite(str(texture_folder / f"sheet_{path_id}.png"), draw_page_texture(path_id))
        print(f"Texture written: sheet_{path_id}.png")

    # JPEG keeps the noisy wood grain small enough for git (a PNG of it is close to a megabyte).
    cv2.imwrite(str(texture_folder / "table_wood.jpg"), draw_table_texture(), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("Texture written: table_wood.jpg")


def run_webots_world(
        world_name: str,
        timeout_s: float = 300.0
) -> int:
    """
    Run a world of the Webots project without its window in the way, and wait until its controller quits.

    Parameters
    ----------
    world_name : str
        The world file's name without ``.wbt``, for example ``"f3_first_look"``.
    timeout_s : float, optional
        How long to wait before giving up.

    Returns
    -------
    int
        Webots' exit code (0 when the controller quit normally).
    """

    parameters = config.SYNTHETIC_PARAMETERS
    world_path = REPOSITORY_FOLDER / parameters["project_folder"] / "worlds" / f"{world_name}.wbt"
    environment = environment_for_controllers()
    environment[QUIT_WHEN_DONE_VARIABLE] = "1"

    command = [parameters["webots_executable"], "--batch", "--mode=fast", "--minimize", "--stdout", "--stderr",
               str(world_path)]
    completed = subprocess.run(command, env = environment, timeout = timeout_s)
    return completed.returncode


def open_webots_world(world_name: str) -> None:
    """
    Open a world of the Webots project in the Webots window and return at once, leaving the window open.

    The simulation runs in real time; the controller takes its pictures and then keeps the scene running, so the
    camera's view stays visible in the corner of the 3D view.

    Parameters
    ----------
    world_name : str
        The world file's name without ``.wbt``, for example ``"f3_first_look"``.
    """

    parameters = config.SYNTHETIC_PARAMETERS
    world_path = REPOSITORY_FOLDER / parameters["project_folder"] / "worlds" / f"{world_name}.wbt"
    # webotsw.exe is the windowed start of Webots, without the console window that webots.exe opens.
    window_executable = pathlib.Path(parameters["webots_executable"]).with_name("webotsw.exe")
    subprocess.Popen([str(window_executable), str(world_path)], env = environment_for_controllers())


def environment_for_controllers() -> dict:
    """
    Return the environment for a Webots run whose Python controllers use the project's own interpreter.

    Webots starts a controller with the ``python`` it finds first on the search path, so the folder of the running
    interpreter (the project's environment) is put in front.

    Returns
    -------
    dict
        A copy of the current environment with the search path changed.
    """

    environment = dict(os.environ)
    environment.pop(QUIT_WHEN_DONE_VARIABLE, None)
    environment["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    return environment


if __name__ == "__main__":

    task = sys.argv[1] if len(sys.argv) > 1 else "textures"
    if task == "textures":
        write_textures()
    elif task == "first_look":
        print(f"Webots finished with code {run_webots_world('f3_first_look')}")
    elif task == "open":
        open_webots_world("f3_first_look")
    else:
        print(f"Unknown task {task!r}: use 'textures', 'first_look' or 'open'.")
