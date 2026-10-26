"""
First commanded movement from our own code: turn the base a little and come back.

The arm stays folded; only the base rotates. Clear the table, keep hands out of
reach and have Stop in view before running:
    ..\\.venv\\Scripts\\python.exe move_check.py
"""

from pyniryo import JointsPosition, NiryoRobot

from config import ROBOT_PARAMETERS

HOME_JOINTS = [0.0, 0.5, -1.25, 0.0, 0.0, 0.0]
BASE_TURN_RAD = 0.3


def turn_base_and_return(
        ip_address: str = ROBOT_PARAMETERS["ip_address"],
        velocity_percent: int = ROBOT_PARAMETERS["arm_max_velocity_percent"]
) -> None:
    """
    Move the base joint to ``BASE_TURN_RAD`` and back to the home pose at a capped speed.

    Parameters
    ----------
    ip_address : str
        The robot's address on the cable.
    velocity_percent : int
        Speed cap for every move, as a percentage of the robot's maximum.
    """

    robot = NiryoRobot(ip_address)

    try:
        robot.calibrate_auto()
        robot.set_arm_max_velocity(velocity_percent)
        print(f"Speed capped at {velocity_percent} %")

        turned = HOME_JOINTS.copy()
        turned[0] = BASE_TURN_RAD
        robot.move(JointsPosition(*turned))
        print(f"Base at {BASE_TURN_RAD} rad: {robot.get_joints()}")

        robot.move(JointsPosition(*HOME_JOINTS))
        print(f"Back home: {robot.get_joints()}")
    finally:
        robot.close_connection()
        print("Disconnected")


if __name__ == "__main__":
    turn_base_and_return()
