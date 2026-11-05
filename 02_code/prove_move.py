"""
Prove that a move happens: read the pose, move 4 cm above the taught start pose, read it again, wait, go home.

Written 2.11.26 after a stepped descent was reported as "the robot did not move" although every stage was logged.
Prints the robot's reported position before and after the move, so the eye and the robot can be compared. Run from
02_code, nothing else connected, stop button in reach:
    ..\\.venv\\Scripts\\python.exe move_check.py
"""

from pyniryo import JointsPosition, NiryoRobot, PoseObject

from config import RECORDING_PARAMETERS, ROBOT_PARAMETERS, start_pose


def report(robot: NiryoRobot, label: str) -> None:
    """Print the rod end's position and whether the motors hold the arm."""

    pose = robot.get_pose()
    print(f"{label}: x = {pose.x:.3f} m, y = {pose.y:.3f} m, z = {pose.z:.3f} m, learning mode = {robot.get_learning_mode()}")


def move_check() -> None:
    """Home, then 4 cm above the start pose, with the reported positions printed before and after."""

    x, y, z, roll, pitch, yaw = start_pose()
    robot = NiryoRobot(ROBOT_PARAMETERS["ip_address"])
    try:
        robot.set_arm_max_velocity(ROBOT_PARAMETERS["arm_max_velocity_percent"])
        report(robot, "before")
        robot.move(JointsPosition(*RECORDING_PARAMETERS["home_joints"]))
        report(robot, "at home")
        robot.move(PoseObject(x, y, z + 0.04, roll, pitch, yaw))
        report(robot, "4 cm above the start pose")
        input("Look at the arm. Press Enter to send it home. ")
        robot.move(JointsPosition(*RECORDING_PARAMETERS["home_joints"]))
        report(robot, "home again")
    finally:
        robot.close_connection()


if __name__ == "__main__":
    move_check()
