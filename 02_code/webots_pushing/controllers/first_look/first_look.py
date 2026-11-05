"""
Render the first synthetic bracket F3 pictures, for a side-by-side check against real pictures.

Runs inside Webots as the supervisor of the pusher (rod, magnet block and camera). For each situation in
``config.SYNTHETIC_PARAMETERS["first_look_situations"]`` it lays the situation's sheet on the table, puts the pusher at
the logged tip position with the puck touching it ahead along the table's +x axis, lets the puck settle and saves one
picture. The camera pose and the page placement come from the same settings; the world file's own positions are only
the starting view. Started by ``synthetic_pictures.run_webots_world`` it quits Webots when done; opened in the Webots
window it keeps the last situation running, so the scene and the camera's view stay on screen.
"""

import math
import os
import pathlib
import sys

from controller import Supervisor

REPOSITORY_FOLDER = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_FOLDER / "02_code"))

import config
import synthetic_pictures


def apply_camera_pose(supervisor: Supervisor) -> None:
    """
    Set the camera's position, tilt and field of view on the pusher from the settings.

    Webots' camera looks along its own x axis with its z axis as the picture's top; a quarter turn about the y axis
    makes it look straight down with the picture's top along the table's +x, and the extra tilt turns it back
    towards the rod.

    Parameters
    ----------
    supervisor : Supervisor
        The running supervisor, which owns the camera.
    """

    parameters = config.SYNTHETIC_PARAMETERS
    camera_node = supervisor.getFromDef("CAMERA")
    camera_node.getField("translation").setSFVec3f(parameters["camera_position_from_tip_m"])
    tilt_rad = math.radians(90.0 + parameters["camera_tilt_back_deg"])
    camera_node.getField("rotation").setSFRotation([0.0, 1.0, 0.0, tilt_rad])
    camera_node.getField("fieldOfView").setSFFloat(parameters["camera_field_of_view_rad"])


def lay_sheet(
        supervisor: Supervisor,
        path_id: str
) -> None:
    """
    Lay the page of a sheet on the table where it lay during the recordings, with its texture.

    Parameters
    ----------
    supervisor : Supervisor
        The running supervisor.
    path_id : str
        The sheet, for example ``"path_04_meander"``.
    """

    centre_x, centre_y, yaw_rad = synthetic_pictures.page_pose_on_table(path_id)
    sheet_node = supervisor.getFromDef("SHEET")
    sheet_node.getField("translation").setSFVec3f([centre_x, centre_y, 0.0002])
    sheet_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, yaw_rad])
    supervisor.getFromDef("SHEET_TEXTURE").getField("url").setMFString(0, f"textures/sheet_{path_id}.png")


def place_pusher_and_puck(
        supervisor: Supervisor,
        tip_xy: list
) -> None:
    """
    Put the pusher's tip over ``tip_xy`` and the puck touching it ahead, along the table's +x axis.

    Parameters
    ----------
    supervisor : Supervisor
        The running supervisor, which is the pusher itself.
    tip_xy : list of float
        The rod tip's logged x and y in metres.
    """

    puck_ahead_m = config.RECORDING_PARAMETERS["puck_centre_ahead_of_tip_m"]
    supervisor.getSelf().getField("translation").setSFVec3f([tip_xy[0], tip_xy[1], 0.0])

    puck_node = supervisor.getFromDef("PUCK")
    puck_node.getField("translation").setSFVec3f([tip_xy[0] + puck_ahead_m, tip_xy[1],
                                                  config.PUCK_PARAMETERS["height_m"] / 2])
    puck_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, 0.0])
    puck_node.resetPhysics()


def main() -> None:
    """Render and save one picture per first-look situation, then quit Webots or keep the scene running."""

    supervisor = Supervisor()
    time_step_ms = int(supervisor.getBasicTimeStep())
    camera = supervisor.getDevice("camera")
    camera.enable(time_step_ms)
    apply_camera_pose(supervisor)

    output_folder = REPOSITORY_FOLDER / config.SYNTHETIC_PARAMETERS["output_folder"] / "first_look"
    output_folder.mkdir(parents = True, exist_ok = True)
    settle_steps = math.ceil(1000 * config.SYNTHETIC_PARAMETERS["settle_time_s"] / time_step_ms)

    for situation_name, situation in config.SYNTHETIC_PARAMETERS["first_look_situations"].items():
        lay_sheet(supervisor, situation["path_id"])
        place_pusher_and_puck(supervisor, situation["tip_xy"])
        for _ in range(settle_steps):
            supervisor.step(time_step_ms)
        picture_path = output_folder / f"{situation_name}.png"
        is_saved = camera.saveImage(str(picture_path), 100) == 0
        print(f"{situation_name}: {'saved' if is_saved else 'NOT saved'} {picture_path}")

    if os.environ.get(synthetic_pictures.QUIT_WHEN_DONE_VARIABLE) == "1":
        supervisor.simulationQuit(0)
        return
    while supervisor.step(time_step_ms) != -1:
        pass


main()
