"""
Render twin pictures of logged robot steps: the controller of the forearm camera twin.

Runs inside Webots as the supervisor of the camera rig. It reads the job file named by the environment variable in
``synthetic_pictures.TWIN_JOB_VARIABLE``, which lists for every picture the pose of the camera, of the forearm, wrist,
hand and magnet block, and of the puck, all computed by ``synthetic_pictures.write_twin_job`` from the logged joints.
Started by ``synthetic_pictures.run_twin_replay`` it applies each step, saves the camera's wide undistorted picture
and quits Webots; opened in the Webots window it plays the steps over and over, holding each for ``hold_s`` seconds,
so a person can watch the episode from any side.
"""

import json
import os
import pathlib
import sys

from controller import Supervisor

REPOSITORY_FOLDER = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_FOLDER / "02_code"))

import synthetic_pictures


def set_pose(
        supervisor: Supervisor,
        node_name: str,
        pose: dict
) -> None:
    """
    Move a node of the scene to a pose of the job file.

    Parameters
    ----------
    supervisor : Supervisor
        The running supervisor.
    node_name : str
        The node's DEF name in the world file.
    pose : dict
        ``translation`` in metres and ``rotation`` as Webots' axis and angle.
    """

    node = supervisor.getSelf() if node_name == "CAMERA_RIG" else supervisor.getFromDef(node_name)
    node.getField("translation").setSFVec3f(pose["translation"])
    node.getField("rotation").setSFRotation(pose["rotation"])


def main() -> None:
    """Render every picture of the job, then quit Webots."""

    supervisor = Supervisor()
    time_step_ms = int(supervisor.getBasicTimeStep())
    camera = supervisor.getDevice("camera")
    camera.enable(time_step_ms)

    job = json.loads(pathlib.Path(os.environ[synthetic_pictures.TWIN_JOB_VARIABLE]).read_text())
    texture_field = supervisor.getFromDef("TABLE_PHOTO_TEXTURE").getField("url")
    texture_field.setMFString(0, job["table_photo"]["texture"])
    corners_field = supervisor.getFromDef("TABLE_PHOTO_CORNERS").getField("point")
    for index, corner in enumerate(job["table_photo"]["corners"]):
        corners_field.setMFVec3f(index, corner)
    supervisor.getFromDef("TABLE_PHOTO").getField("translation").setSFVec3f([0.0, 0.0, 0.0])

    if os.environ.get(synthetic_pictures.QUIT_WHEN_DONE_VARIABLE) == "1":
        for picture in job["pictures"]:
            for node_name, pose in picture["poses"].items():
                set_pose(supervisor, node_name, pose)
            for _ in range(3):
                supervisor.step(time_step_ms)
            is_saved = camera.saveImage(picture["wide_picture"], 100) == 0
            print(f"{picture['name']}: {'saved' if is_saved else 'NOT saved'}")
        supervisor.simulationQuit(0)
        return

    # In the window: play the episode's steps in a loop until Webots is closed (step returns -1).
    hold_steps = max(1, int(1000 * job.get("hold_s", 1.5) / time_step_ms))
    while True:
        for picture in job["pictures"]:
            for node_name, pose in picture["poses"].items():
                set_pose(supervisor, node_name, pose)
            print(f"showing {picture['name']}")
            for _ in range(hold_steps):
                if supervisor.step(time_step_ms) == -1:
                    return


main()
