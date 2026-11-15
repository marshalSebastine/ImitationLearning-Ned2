"""Review helper: make sure nothing here can reach the robot, then import the driver under review."""

import sys
import types


class Refused:

    def __init__(self, *arguments, **named_arguments):

        raise AssertionError("a connection to the robot was attempted")


stand_in_pyniryo = types.ModuleType("pyniryo")
stand_in_pyniryo.NiryoRobot = Refused
stand_in_pyniryo.JointsPosition = Refused
stand_in_pyniryo.PoseObject = Refused
sys.modules["pyniryo"] = stand_in_pyniryo

CODE_FOLDER = "D:/AI_Project/.claude/worktrees/training-analysis/02_code"
sys.path.insert(0, CODE_FOLDER)

import robot_link

assert robot_link.NiryoRobot is Refused
robot_link.RobotLink = Refused

import model_mode
