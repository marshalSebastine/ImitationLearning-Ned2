"""
Test whether turning the forearm gives a usable side view for pulling the puck towards the base.

Turning joint 4 swings the camera to look along a pull, but the rod then leans (see SIDE_VIEW_TEST_PARAMETERS in
config). This script finds the least lean with the arm model, keeps the magnet tip on target, and checks every planned
position against the robot's own kinematics before anything moves. It records pictures and numbers only; nothing it
saves is training data.

    --part view: no puck. At each distance, turn the forearm with the tip lifted, come down, save a picture, lift.
    --part pull: at one distance, come down, the person sets the puck on the base side of the tip, then one key per
                 1 cm pull. Keys go to the picture window: Enter = next, x = stop this part, Esc = end the test.

Run from 02_code, robot connected, nothing else connected to it, stop button in reach (checklist:
06_deliverables/research_notes/Recording tool/side_view_test_checklist.md):
    python side_view_test.py --operator emranul --part view
    python side_view_test.py --operator emranul --part pull --distance 0.25
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy

from config import PUSHER_PARAMETERS, SIDE_VIEW_TEST_PARAMETERS, pushing_height

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "06_deliverables/research_notes/Camera mount options/scripts"))
import ned2_fk


JOINT_LIMITS_RAD = [(math.radians(low), math.radians(high)) for low, high in ned2_fk.LIM_DEG.values()]
WINDOW_NAME = "side_view_test"


def magnet_tip_and_lean(joints: List[float]) -> tuple:
    """
    The magnet tip's position and the rod's lean for ``joints``, from the arm model.

    Returns
    -------
    tuple of numpy.ndarray and float
        The tip [x, y, z] in metres and the angle of the rod from straight down in degrees.
    """

    frames = ned2_fk.fk(list(joints))
    rod_direction = numpy.array(frames["joint_6"][1])[:, 2]
    tip = numpy.array(frames["tool"][0]) + PUSHER_PARAMETERS["extension_below_rod_end_m"] * rod_direction
    lean_deg = math.degrees(math.acos(max(-1.0, min(1.0, -float(rod_direction[2])))))
    return tip, lean_deg


def solve_with_wrist_fixed(
        target_tip: numpy.ndarray,
        turn_rad: float,
        wrist_rad: float,
        start: List[float]
) -> Optional[List[float]]:
    """Joints 1 to 3 that put the tip on ``target_tip`` for fixed joints 4 and 5; None when out of reach or limits."""

    values = numpy.array(start, dtype = float)
    for _ in range(30):
        joints = [values[0], values[1], values[2], turn_rad, wrist_rad, 0.0]
        error = magnet_tip_and_lean(joints)[0] - target_tip
        if numpy.linalg.norm(error) < 1e-6:
            break
        jacobian = numpy.zeros((3, 3))
        for index in range(3):
            shifted = list(joints)
            shifted[index] += 1e-5
            jacobian[:, index] = (magnet_tip_and_lean(shifted)[0] - magnet_tip_and_lean(joints)[0]) / 1e-5
        values = values - numpy.linalg.lstsq(jacobian, error, rcond = None)[0]

    joints = [float(values[0]), float(values[1]), float(values[2]), turn_rad, wrist_rad, 0.0]
    is_on_target = numpy.linalg.norm(magnet_tip_and_lean(joints)[0] - target_tip) < 1e-4
    is_within_limits = all(low <= value <= high for value, (low, high) in zip(joints, JOINT_LIMITS_RAD))
    is_elbow_above_base = ned2_fk.fk(joints)["joint_3"][0][2] > 0.0
    return joints if is_on_target and is_within_limits and is_elbow_above_base else None


def solve_least_lean(
        target_tip: numpy.ndarray,
        turn_rad: float
) -> Optional[List[float]]:
    """
    The joints that put the magnet tip on ``target_tip`` with joint 4 at ``turn_rad`` and the rod leaning least.

    Joint 5 is searched in 1 degree steps and then 0.1 degree steps around the best one; joint 6 only spins the rod
    about its own axis, so it stays at zero. None when no joint angles within the limits reach the target.
    """

    best_lean, best_joints = math.inf, None
    start = [math.atan2(target_tip[1], target_tip[0]), -0.8, -0.6]
    for wrist_candidates in (numpy.arange(-60.0, 60.5, 1.0), None):
        if wrist_candidates is None:
            if best_joints is None:
                return None
            centre = math.degrees(best_joints[4])
            wrist_candidates = numpy.arange(centre - 1.0, centre + 1.05, 0.1)
        for wrist_deg in wrist_candidates:
            joints = solve_with_wrist_fixed(target_tip, turn_rad, math.radians(float(wrist_deg)), start)
            if joints is None:
                continue
            lean_deg = magnet_tip_and_lean(joints)[1]
            if lean_deg < best_lean:
                best_lean, best_joints = lean_deg, joints
    return best_joints



class SideViewTest:

    def __init__(
            self,
            robot_link,
            operator: str,
            test_parameters: Optional[dict] = None,
            key_reader: Optional[Callable[[int], Optional[str]]] = None
    ):
        """
        Prepare the test on an open robot link.

        Parameters
        ----------
        robot_link : RobotLink
            The robot; it needs move_joints, predicted_position, read_state, read_picture_bytes and close.
        operator : str
            Who runs the test, written into the results.
        test_parameters : dict, optional
            Overrides merged over SIDE_VIEW_TEST_PARAMETERS, plus "is_window_shown".
        key_reader : callable, optional
            Returns the next key; replaced in tests by a scripted sequence.
        """

        self.robot_link = robot_link
        self.operator = operator
        self.key_reader = key_reader
        self.test_parameters = dict(SIDE_VIEW_TEST_PARAMETERS)
        self.test_parameters["is_window_shown"] = True
        if test_parameters:
            self.test_parameters.update(test_parameters)

        self.last_picture = None
        self.current_joints = None
        self.current_tip_xy = None
        self.is_low = False
        self.has_moved = False
        self.rows = []
        self.folder = None


    def run(
            self,
            part: str,
            distance_m: Optional[float] = None
    ) -> List[Dict]:
        """
        Run ``part`` ("view" or "pull"), then lift, turn back and go home whatever happened.

        Returns
        -------
        list of dict
            One row per saved picture, also written to results.csv.
        """

        distances = self.test_parameters["distances_m"] if part == "view" else [distance_m]
        self.__check_plan(distances)
        self.folder = Path(__file__).resolve().parent.parent / self.test_parameters["output_folder"] / time.strftime(f"%Y-%m-%d_%H%M_{part}_{self.operator}")
        self.folder.mkdir(parents = True, exist_ok = True)

        try:
            if part == "view":
                self.__run_view()
            else:
                self.__run_pull(distance_m)
        except EndOfTest:
            print("test ended by the operator")
        finally:
            self.__write_results()
            self.__retreat()
        return self.rows


    def __check_plan(self, distances: List[float]) -> None:
        """Before anything moves: every planned position must be solvable, lean little and match the robot's kinematics."""

        turn_rad = math.radians(self.test_parameters["turn_deg"])
        for distance in distances:
            if distance < self.test_parameters["minimum_distance_m"]:
                raise ValueError(f"{distance} m is closer to the base than minimum_distance_m")
            for lift in (self.test_parameters["lift_height_m"], 0.0):
                joints = solve_least_lean(self.__target_tip([distance, self.test_parameters["line_y_m"]], lift), turn_rad)
                if joints is None:
                    raise ValueError(f"no solution at {distance} m with a {self.test_parameters['turn_deg']} degree turn")
                lean_deg = magnet_tip_and_lean(joints)[1]
                if lean_deg > self.__lean_limit(lift):
                    raise ValueError(f"{distance} m needs a {lean_deg:.1f} degree lean, above the limit of {self.__lean_limit(lift)}")

                model_position = numpy.array(ned2_fk.fk(joints)["tool"][0])
                robot_position = numpy.array(self.robot_link.predicted_position(joints))
                error_mm = 1000.0 * float(numpy.linalg.norm(model_position - robot_position))
                if error_mm > self.test_parameters["model_tolerance_mm"]:
                    raise ValueError(f"the arm model and the robot disagree by {error_mm:.1f} mm at {distance} m; nothing was moved")
            print(f"{distance} m: plan checked, lean {lean_deg:.1f} degrees, model within {error_mm:.2f} mm of the robot")


    def __run_view(self) -> None:

        for distance in self.test_parameters["distances_m"]:
            tip_xy = [distance, self.test_parameters["line_y_m"]]
            self.__go_lifted_and_turned(tip_xy)
            self.__wait_for_enter(f"{distance} m, lifted. Enter = come down")
            self.__move_tip(tip_xy, lift = 0.0, turn_deg = self.test_parameters["turn_deg"])
            self.is_low = True
            self.__save_picture(f"view_{int(distance * 100):02d}cm", "view", distance)
            self.__wait_for_enter(f"{distance} m, tip down. Check the view and the gap to the paper. Enter = lift")
            self.__move_tip(tip_xy, lift = self.test_parameters["lift_height_m"], turn_deg = self.test_parameters["turn_deg"])
            self.is_low = False


    def __run_pull(self, distance_m: float) -> None:

        tip_xy = [distance_m, self.test_parameters["line_y_m"]]
        self.__go_lifted_and_turned(tip_xy)
        self.__wait_for_enter(f"{distance_m} m, lifted, no puck on the sheet yet. Enter = come down")
        self.__move_tip(tip_xy, lift = 0.0, turn_deg = self.test_parameters["turn_deg"])
        self.is_low = True
        self.__wait_for_enter("Set the puck on the BASE side of the tip, touching it. Do not touch the arm. Enter = start")
        self.__save_picture("pull_000", "pull", distance_m)

        for step in range(1, self.test_parameters["pull_steps"] + 1):
            next_x = tip_xy[0] - self.test_parameters["step_m"]
            if next_x < self.test_parameters["minimum_distance_m"]:
                print(f"stopped: the next step would pass minimum_distance_m ({self.test_parameters['minimum_distance_m']} m)")
                return
            if self.__wait_for_enter_or_stop(f"pull step {step}: Enter = pull 1 cm, x = stop") == "stop":
                return
            tip_xy = [next_x, tip_xy[1]]
            self.__move_tip(tip_xy, lift = 0.0, turn_deg = self.test_parameters["turn_deg"])
            self.__save_picture(f"pull_{step:03d}", "pull", next_x)


    def __go_lifted_and_turned(self, tip_xy: List[float]) -> None:
        """Reach the point above ``tip_xy`` unturned, then turn the forearm in increments with the tip held in place."""

        lift = self.test_parameters["lift_height_m"]
        turn_deg = self.current_turn_deg() if self.current_joints is not None else 0.0
        self.__move_tip(tip_xy, lift = lift, turn_deg = turn_deg)
        increment = self.test_parameters["turn_increment_deg"]
        while turn_deg < self.test_parameters["turn_deg"]:
            turn_deg = min(turn_deg + increment, self.test_parameters["turn_deg"])
            self.__move_tip(tip_xy, lift = lift, turn_deg = turn_deg)


    def __lean_limit(self, lift: float) -> float:

        return self.test_parameters["maximum_lifted_lean_deg" if lift > 0.0 else "maximum_lean_deg"]


    def current_turn_deg(self) -> float:

        return math.degrees(self.current_joints[3])


    def __target_tip(
            self,
            tip_xy: List[float],
            lift: float
    ) -> numpy.ndarray:

        height = pushing_height(tip_xy[0], tip_xy[1]) - PUSHER_PARAMETERS["extension_below_rod_end_m"] + lift
        return numpy.array([tip_xy[0], tip_xy[1], height])


    def __move_tip(
            self,
            tip_xy: List[float],
            lift: float,
            turn_deg: float
    ) -> None:

        joints = solve_least_lean(self.__target_tip(tip_xy, lift), math.radians(turn_deg))
        if joints is None:
            raise ValueError(f"no solution at {tip_xy} with a {turn_deg} degree turn")
        lean_deg = magnet_tip_and_lean(joints)[1]
        if lean_deg > self.__lean_limit(lift):
            raise ValueError(f"{tip_xy} needs a {lean_deg:.1f} degree lean, above the limit of {self.__lean_limit(lift)}; not moving")
        self.robot_link.move_joints(joints)
        self.has_moved = True
        self.current_joints = joints
        self.current_tip_xy = list(tip_xy)


    def __wait_for_enter(self, message: str) -> None:

        self.__wait_for_enter_or_stop(message, stop_keys = ())


    def __wait_for_enter_or_stop(
            self,
            message: str,
            stop_keys: tuple = ("x",)
    ) -> str:
        """Show the picture and ``message``; "go" on Enter, "stop" on a stop key, and Esc ends the test."""

        print(message)
        self.__show_picture(self.robot_link.read_picture_bytes(), message)
        while True:
            key = self.key_reader(0)
            if key == "enter":
                return "go"
            if key in stop_keys:
                return "stop"
            if key == "escape":
                raise EndOfTest()


    def __save_picture(
            self,
            name: str,
            part: str,
            distance: float
    ) -> None:

        picture = self.__fresh_picture()
        (self.folder / f"{name}.jpg").write_bytes(picture)
        state = self.robot_link.read_state()
        lean_deg = magnet_tip_and_lean(state["joints"])[1]
        tip = magnet_tip_and_lean(state["joints"])[0]
        row = {"operator": self.operator, "part": part, "picture": f"{name}.jpg", "distance_m": round(distance, 4),
               "turn_deg": round(math.degrees(state["joints"][3]), 2), "lean_deg": round(lean_deg, 2),
               "tip_x": round(float(tip[0]), 5), "tip_y": round(float(tip[1]), 5), "tip_z": round(float(tip[2]), 5)}
        row.update({f"joint_{index + 1}": round(value, 5) for index, value in enumerate(state["joints"])})
        self.rows.append(row)
        print(f"{name}: lean {lean_deg:.1f} degrees, tip height {1000 * tip[2]:.1f} mm")


    def __fresh_picture(self) -> bytes:
        """Wait for the arm to settle and for a picture that differs from the last one saved."""

        time.sleep(self.test_parameters["settle_s"])
        deadline = time.monotonic() + self.test_parameters["fresh_picture_timeout_s"]
        picture = self.robot_link.read_picture_bytes()
        while picture == self.last_picture and time.monotonic() < deadline:
            time.sleep(0.02)
            picture = self.robot_link.read_picture_bytes()
        self.last_picture = picture
        return picture


    def __show_picture(
            self,
            picture_bytes: bytes,
            status: str
    ) -> None:

        if not self.test_parameters["is_window_shown"]:
            return
        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
        status_bar = numpy.zeros((32, picture.shape[1], 3), dtype = numpy.uint8)
        cv2.putText(status_bar, status[:90], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.imshow(WINDOW_NAME, numpy.vstack([picture, status_bar]))


    def __write_results(self) -> None:

        if not self.rows:
            return
        with open(self.folder / "results.csv", "w", newline = "", encoding = "utf-8") as file:
            writer = csv.DictWriter(file, fieldnames = list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"results saved in {self.folder}")


    def __retreat(self) -> None:
        """Lift first so the tip cannot sweep the puck, turn back in increments, go home, disconnect."""

        try:
            if self.has_moved:
                lift = self.test_parameters["lift_height_m"]
                turn_deg = self.current_turn_deg()
                self.__move_tip(self.current_tip_xy, lift = lift, turn_deg = turn_deg)
                while turn_deg > 0.0:
                    turn_deg = max(turn_deg - self.test_parameters["turn_increment_deg"], 0.0)
                    self.__move_tip(self.current_tip_xy, lift = lift, turn_deg = turn_deg)
                self.robot_link.move_joints(self.test_parameters["home_joints"])
        finally:
            self.robot_link.close()
            if self.test_parameters["is_window_shown"]:
                cv2.destroyAllWindows()



class EndOfTest(Exception):
    """The operator pressed Esc."""



def main() -> None:

    from record_episodes import read_window_key
    from robot_link import RobotLink

    parser = argparse.ArgumentParser(description = "Test the forearm-turned side view for pulling the puck towards the base.")
    parser.add_argument("--operator", required = True, help = "who runs the test, e.g. emranul")
    parser.add_argument("--part", choices = ["view", "pull"], required = True, help = "view: no puck, three distances; pull: one distance, with the puck")
    parser.add_argument("--distance", type = float, default = 0.25, help = "pull only: the tip's starting distance from the base in metres")
    parser.add_argument("--turn-deg", type = float, default = SIDE_VIEW_TEST_PARAMETERS["turn_deg"], help = "how far to turn joint 4, in degrees")
    arguments = parser.parse_args()

    robot_link = RobotLink(velocity_percent = SIDE_VIEW_TEST_PARAMETERS["velocity_percent"])
    test = SideViewTest(robot_link, arguments.operator, {"turn_deg": arguments.turn_deg}, read_window_key)
    test.run(arguments.part, arguments.distance)


if __name__ == "__main__":
    main()
