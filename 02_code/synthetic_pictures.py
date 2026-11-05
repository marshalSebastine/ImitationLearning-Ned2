"""
Synthetic camera pictures for imitation learning, rendered with the Webots simulator.

Emranul chose on 5.11.26 to make the pictures in Webots. This module holds what the simulator needs from the project's
own measurements: the traced centre lines of the course sheets drawn as page textures, where a page lies on the table,
a wood texture for the table, and the command that runs a Webots world without its window in the way. Settings come
from ``config.SYNTHETIC_PARAMETERS``.

Write the textures, then render the first-look pictures:

    python 02_code/synthetic_pictures.py textures
    python 02_code/synthetic_pictures.py first_look
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

    Webots starts the controller with the ``python`` it finds first on the search path, so the folder of the running
    interpreter is put in front: the controller then uses the project's own environment.

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
    environment = dict(os.environ)
    environment["PATH"] = str(pathlib.Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")

    command = [parameters["webots_executable"], "--batch", "--mode=fast", "--minimize", "--stdout", "--stderr",
               str(world_path)]
    completed = subprocess.run(command, env = environment, timeout = timeout_s)
    return completed.returncode


if __name__ == "__main__":

    task = sys.argv[1] if len(sys.argv) > 1 else "textures"
    if task == "textures":
        write_textures()
    elif task == "first_look":
        print(f"Webots finished with code {run_webots_world('f3_first_look')}")
    else:
        print(f"Unknown task {task!r}: use 'textures' or 'first_look'.")
