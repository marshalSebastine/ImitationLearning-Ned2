"""
The robot as the recording tool sees it: the real Ned2 through pyniryo, or a simulated one.

Both classes offer the same six methods (move_to, move_home, read_state, read_picture_bytes, set_free_motion,
close), so record_episodes.py runs unchanged against either. The simulated robot exists to test the recording tool
on a laptop without the robot: it draws the sheet, the line and the puck as the wrist camera would see them, pushes
the puck when the pusher touches it, and hands out a new picture only about nine times per second, like the real
camera, so the stale-picture guard is exercised too. Its physics is a rough geometric model, not a measurement.
"""

import math
import time
from typing import Dict, List, Optional

import cv2
import numpy
from pyniryo import JointsPosition, NiryoRobot, PoseObject

from config import (CAMERA_PARAMETERS, PUCK_PARAMETERS, PUSHER_PARAMETERS, ROBOT_PARAMETERS, WORKSPACE_PARAMETERS,
                    camera_heading_rad, pushing_height, start_pose)


class RobotLink:

    def __init__(
            self,
            ip_address: str = ROBOT_PARAMETERS["ip_address"],
            velocity_percent: int = ROBOT_PARAMETERS["arm_max_velocity_percent"]
    ):
        """
        Connect to the robot. Only one program may be connected at a time; close any viewer first.

        Parameters
        ----------
        ip_address : str, optional
            The robot's address, by default the direct-Ethernet one from config.
        velocity_percent : int, optional
            The arm's speed cap in percent.
        """

        self.robot = NiryoRobot(ip_address)
        self.robot.set_arm_max_velocity(velocity_percent)


    def move_to(
            self,
            pose: List[float],
            is_linear: bool = True
    ) -> None:
        """Move the rod's end to ``pose`` = [x, y, z, roll, pitch, yaw]; returns when the move is finished."""

        self.robot.move(PoseObject(*pose), linear = is_linear)


    def move_home(self, home_joints: List[float]) -> None:
        """Move to the joint angles ``home_joints``."""

        self.robot.move(JointsPosition(*home_joints))


    def read_state(self) -> Dict[str, List[float]]:
        """The rod end's pose [x, y, z, roll, pitch, yaw] and the six joint angles, read now."""

        pose = self.robot.get_pose()
        return {
            "pose": [pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw],
            "joints": [float(value) for value in self.robot.get_joints()]
        }


    def read_picture_bytes(self) -> bytes:
        """The latest JPEG from the camera, exactly as the robot sends it (it may be the same one as last time)."""

        return bytes(self.robot.get_img_compressed())


    def set_free_motion(self, is_enabled: bool) -> None:
        """Switch the motors off (True, the arm can be guided by hand and sags) or back on (False)."""

        self.robot.set_learning_mode(is_enabled)


    def close(self) -> None:

        self.robot.close_connection()



class SimulatedRobotLink:

    def __init__(self, simulation_parameters: Optional[dict] = None):
        """
        A stand-in for the robot that needs no hardware.

        Parameters
        ----------
        simulation_parameters : dict, optional
            Overrides for the defaults below. ``line_xy`` is the line as a list of [x, y] points in metres;
            ``hand_path_xy`` is the path the simulated hand moves the tip along while the motors are off, one point
            per call of read_state.
        """

        self.simulation_parameters = {
            "line_xy": [WORKSPACE_PARAMETERS["line_start_xy"], WORKSPACE_PARAMETERS["line_end_xy"]],
            "puck_xy": list(WORKSPACE_PARAMETERS["line_start_xy"]),
            "hand_path_xy": [],
            "pixels_per_m": 7_500,
            "picture_bottom_ahead_of_rod_m": 0.01,
            "move_seconds": 0.0,
            "centring_weight": 0.3
        }
        if simulation_parameters:
            self.simulation_parameters.update(simulation_parameters)

        self.pose = list(start_pose())
        self.pose[2] += 0.10
        self.joints = [0.0] * 6
        self.puck_xy = numpy.array(self.simulation_parameters["puck_xy"], dtype = float)
        self.is_free_motion = False
        self.hand_path_index = 0
        self.picture_tick = None
        self.picture_count = 0
        self.picture_bytes = b""


    def move_to(
            self,
            pose: List[float],
            is_linear: bool = True
    ) -> None:

        start = numpy.array(self.pose[:2])
        end = numpy.array(pose[:2])
        is_low = pose[2] < pushing_height(pose[0], pose[1]) + 0.01
        substeps = max(1, int(numpy.linalg.norm(end - start) / 0.001))

        # Move in 1 mm substeps so the puck is pushed along the way rather than jumped over.
        for index in range(1, substeps + 1):
            self.pose[:2] = list(start + (end - start) * index / substeps)
            if is_low:
                self.__push_puck(end - start)

        self.pose = list(pose)
        time.sleep(self.simulation_parameters["move_seconds"])


    def move_home(self, home_joints: List[float]) -> None:

        self.pose[2] = start_pose()[2] + 0.10


    def read_state(self) -> Dict[str, List[float]]:

        hand_path = self.simulation_parameters["hand_path_xy"]
        if self.is_free_motion and self.hand_path_index < len(hand_path):
            motion = numpy.array(hand_path[self.hand_path_index]) - numpy.array(self.pose[:2])
            self.pose[:2] = list(hand_path[self.hand_path_index])
            self.hand_path_index += 1
            self.__push_puck(motion)

        return {"pose": list(self.pose), "joints": list(self.joints)}


    def read_picture_bytes(self) -> bytes:

        # A new picture only when the camera's clock has ticked; between ticks the same JPEG comes back.
        tick = int(time.monotonic() * CAMERA_PARAMETERS["frames_per_second"])
        if tick != self.picture_tick:
            self.picture_tick = tick
            self.picture_count += 1
            self.picture_bytes = self.__render_picture()

        return self.picture_bytes


    def set_free_motion(self, is_enabled: bool) -> None:

        self.is_free_motion = is_enabled


    def close(self) -> None:

        pass


    def place_puck_at_start(self) -> None:
        """What the person does by hand on the real robot: put the puck back and rewind the simulated hand."""

        self.puck_xy = numpy.array(self.simulation_parameters["puck_xy"], dtype = float)
        self.hand_path_index = 0


    def __push_puck(self, motion_xy: numpy.ndarray) -> None:
        """
        Push the puck when the tip overlaps it, a rough quasi-static model: the puck is put back at contact distance,
        in a direction between "away from the tip" and the tip's motion, so it drifts towards the front of the tip
        instead of sliding off sideways ("centring_weight" sets how strongly). Not measured; the real puck may slide off.
        """

        contact_distance = PUCK_PARAMETERS["diameter_m"] / 2 + PUSHER_PARAMETERS["tip_diameter_m"] / 2
        tip_xy = numpy.array(self.pose[:2])
        offset = self.puck_xy - tip_xy
        distance = numpy.linalg.norm(offset)
        motion_length = numpy.linalg.norm(motion_xy)
        if not 1e-9 < distance < contact_distance:
            return

        direction = offset / distance
        if motion_length > 1e-9 and direction @ motion_xy > 0:
            direction = direction + self.simulation_parameters["centring_weight"] * motion_xy / motion_length
            direction = direction / numpy.linalg.norm(direction)
        self.puck_xy = tip_xy + direction * contact_distance


    def __render_picture(self) -> bytes:
        """Draw what the wrist camera would see: the sheet, the line and the puck, top of picture = camera heading."""

        width, height = CAMERA_PARAMETERS["width"], CAMERA_PARAMETERS["height"]
        pixels_per_m = self.simulation_parameters["pixels_per_m"]
        heading = camera_heading_rad(self.pose[0], self.pose[1])
        forward = numpy.array([math.cos(heading), math.sin(heading)])
        left = numpy.array([-math.sin(heading), math.cos(heading)])
        bottom_centre = numpy.array(self.pose[:2]) + forward * self.simulation_parameters["picture_bottom_ahead_of_rod_m"]

        def to_pixel(point_xy) -> tuple:
            offset = numpy.array(point_xy) - bottom_centre
            return (int(round(width / 2 - offset @ left * pixels_per_m)), int(round(height - offset @ forward * pixels_per_m)))

        picture = numpy.full((height, width, 3), 235, dtype = numpy.uint8)
        line_pixels = numpy.array([to_pixel(point) for point in self.simulation_parameters["line_xy"]], dtype = numpy.int32)
        line_thickness = int(0.010 * pixels_per_m)
        cv2.polylines(picture, [line_pixels], isClosed = False, color = (20, 20, 20), thickness = line_thickness)
        cv2.circle(picture, to_pixel(self.puck_xy), int(PUCK_PARAMETERS["diameter_m"] / 2 * pixels_per_m), (40, 40, 200), -1)
        cv2.putText(picture, f"simulated {self.picture_count}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        return cv2.imencode(".jpg", picture)[1].tobytes()


def simulated_hand_path(
        wobble_m: float = 0.004,
        spacing_m: float = 0.002
) -> List[List[float]]:
    """
    A hand's path for the simulated robot: from the start pose along the taught line to the goal, with a gentle
    sideways wobble, as a hand would draw it.
    """

    start = numpy.array(start_pose()[:2])
    end = numpy.array(WORKSPACE_PARAMETERS["line_end_xy"])
    direction = (end - start) / numpy.linalg.norm(end - start)
    side = numpy.array([-direction[1], direction[0]])
    length = numpy.linalg.norm(end - start) - PUCK_PARAMETERS["diameter_m"] / 2 - PUSHER_PARAMETERS["tip_diameter_m"] / 2
    distances = numpy.arange(0.0, length, spacing_m)

    return [list(start + distance * direction + wobble_m * math.sin(distance / 0.03) * side) for distance in distances]
