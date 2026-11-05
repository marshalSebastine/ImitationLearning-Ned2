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
