"""
Machine-wide defaults for the robot code.

Every script reads its settings from these dictionaries, so a value is changed
in one place only. Later phases add the keys named in the work plan.
"""

import math
from typing import Optional

ROBOT_PARAMETERS = {
    "ip_address": "169.254.200.200",  # Ethernet cable (Emranul's laptop); 10.10.10.10 over the robot's hotspot
    "software_version": "6.1.2-b374",
    "arm_max_velocity_percent": 20,
    # The robot's inverse-kinematics service sometimes refuses a pose it solves a moment later (3.11.26: the
    # recording tool's start pose, refused once with error -232, solved at the next call from the same arm position).
    # RobotLink asks this many times, with the pause between, before it gives up on a pose.
    "inverse_kinematics_attempts": 3,
    "inverse_kinematics_retry_pause_s": 0.3
}

# The Vision Set camera plugs into the robot's USB port and streams through the
# robot: about 9 new pictures per second as JPEG (measured 24.10.26; the 16 of the first day was nominal).
# Plugged into the laptop instead it is a plain USB
# camera at 32 frames per second (OpenCV index 1; index 0 is the laptop's own webcam).
# Camera: in its original bracket, screwed on (25.10.26). The bracket sits on the wrist link: joint 6
# does not move the camera (four pictures with only joint 6 turned are identical), and the view stayed
# the same over the whole 24 cm push while the forearm tilted about 40 degrees, so the camera follows
# the rod's lean (joint 5), not the forearm. Confirmed on 26.10.26 from Niryo's own set-up guide: the holder
# screws into the wrist's aluminium brace beside the joint-6 motor. Niryo's calibration file gives a 64
# degree lens (fx 508 px). With the magnet block on the rod the wrist sits 2.6 cm higher, and
# from there the puck stays at the bottom of every picture over the whole line (scripted push).
# Reference picture 03_data/raw/camera_check/view_reference_rod_camera_start_pose.jpg. Do not touch
# the camera or its focus ring. Calibration: only the automatic one after a restart, never manual.
# 2.11.26: camera on bracket F (lens about 12 cm above the paper at pushing height); the one planned refocus was done
# at that distance: sharpness 416 at pushing height, 287 seven centimetres higher (focus_sweep.py). Do not touch the
# ring again unless the mount changes its distance to the paper.
CAMERA_PARAMETERS = {
    "source": "robot",
    "usb_index": 1,
    "width": 640,
    "height": 480,
    "frames_per_second": 9
}

# What the camera sees with the rod straight down, measured on 25.10.26 from the scripted-push pictures
# (03_data/raw/workspace/push_straight/): the 1 cm line is 75 px wide beside the puck and 63 px wide at
# the top of the 640 px picture, and the paper's far edge enters the top of the picture when the puck is
# about 7 cm short of it. The window always points straight away from the base, because a vertical rod
# forces the forearm roll (joint 4) to zero, so the line ahead stays in view only while the path heads
# within about 45 degrees of that direction; the puck itself stays in view up to about 60 degrees. When
# pushing straight the line sits about a third of the way in from one side (column 150 to 200).
CAMERA_VIEW_PARAMETERS = {
    "width_at_puck_m": 0.085,
    "width_at_top_m": 0.100,
    "look_ahead_from_puck_centre_m": 0.07,
    "usable_heading_from_radial_deg": 45,
    "puck_visible_heading_deg": 60
}

# Measured on 24.10.26 with measure_robot.py over the cable, arm speed capped at 20 %.
# Reads are quick, so a recorder can sample them far above the camera's 16 pictures per second.
# A move() call blocks until the move is finished and carries about half a second of fixed
# overhead even for a tiny step, so closed-loop control gets roughly one command per second.
MEASURED_TIMINGS_SECONDS = {
    "get_joints": 0.0025,
    "get_pose": 0.0032,
    "get_image": 0.0030,
    "move_base_0.01_rad": 0.58,
    "move_base_0.02_rad": 0.64,
    "move_base_0.05_rad": 0.87,
    "move_base_0.10_rad": 1.08,
    "shift_tip_pose_command_1_mm": 0.90,
    "shift_tip_pose_command_2_mm": 1.14,
    "shift_tip_pose_command_5_mm": 2.19,
    "shift_tip_pose_command_10_mm": 1.91,
    "shift_tip_pose_command_20_mm": 4.20,
    "shift_tip_pose_command_10_mm_at_40_percent": 2.66,
    "step_1_cm_pose_command_linear": 1.18,
    "step_1_cm_linear_during_push_at_20_percent": 0.77,
    "step_1_cm_linear_during_push_at_40_percent": 0.76,
    "step_2_cm_linear_during_push": 1.07,
    "step_1_cm_pose_command_joint_interpolated": 1.64,
    "step_1_cm_inverse_kinematics_then_joint_move": 1.76,
    "inverse_kinematics_call": 0.11
}

# Survey of 24.10.26 (survey_robot.py and measure_step_methods.py; full results in
# 03_data/raw/robot_survey/2026-10-24_survey.json). The robot's state can be polled about 400 times
# per second; picture reads return the latest JPEG about 190 times per second, but only about 9 of
# them per second are new pictures, so the camera's real rate is 9 per second in this light, not the
# 16 seen before. Tip shifts land within 0.5 mm and the arm returns home within 0.7 mm. A
# one-centimetre tip step costs 1.2 s as a straight-line pose command, the fastest of the three ways
# tried, so the control loop gets one decision every one to two seconds at the 20 % cap. Only one
# program can hold the robot: a second connection hangs at connect and never answers, while the
# first keeps working.
MEASURED_RATES_PER_SECOND = {
    "state_samples": 408,
    "camera_reads": 190,
    "camera_new_pictures": 8.9
}

# Measured on 24.10.26 with a ruler and a kitchen scale. The puck is taller than it is wide, so it
# tips more easily than a flat disc, and the bare metal rod at the wrist (the gripper is off) is
# shorter than the puck is tall. With the rod's end 10 mm above the sheet the rod touches the puck
# from 10 mm up its side, the push still acts below the tipping height (about 24 to 30 mm for wood
# on paper), and the wrist housing clears the puck's top by about 5 mm. Every step starts and stops
# gently. "In the goal" means the puck's centre within one puck radius of the end of the line.
PUCK_PARAMETERS = {
    "diameter_m": 0.024,
    "height_m": 0.029,
    "mass_kg": 0.009,
    "goal_tolerance_m": 0.012,
    "pusher_clearance_above_sheet_m": 0.005
}

# The robot reports the position of the bare rod's end. Whatever is clipped onto the rod reaches
# further down by "extension_below_rod_end_m", and "tip_diameter_m" is the width of the part that
# touches the puck. With nothing clipped on, both describe the bare rod.
PUSHER_PARAMETERS = {
    "kind": "magnet block snapped onto the rod (25.10.26); camera in its original wrist bracket",
    "diameter_m": 0.018,
    "length_below_wrist_m": 0.0245,
    "tip_diameter_m": 0.013,
    "extension_below_rod_end_m": 0.026
}

# The first sheet, measured on 24.10.26. The line is narrower than the puck, so the puck hides the
# line under it and the pictures show the line ahead. The step budget is 1.5 times the line's
# length in centimetre steps.
SHEET_PARAMETERS = {
    "path_id": "path_01_straight_line",
    "line_length_m": 0.24,
    "line_width_m": 0.010,
    "step_budget": 36
}

# Taught by touching six points with the rod's tip on 25.10.26 (teach_positions.py; raw readings in
# 03_data/raw/workspace/2026-10-25_positions.json). The line runs straight away from the robot, from
# 0.156 m to 0.398 m in front of the base, because the wrist camera looks straight ahead, away from the
# base, and sees only what lies beyond the rod. The four corner touches give the table plane (tilt 0.6 degrees);
# the pushing height is that plane plus the clearance in PUCK_PARAMETERS. The box's far edge is
# capped at the table's edge rather than the sheet plus margin.
WORKSPACE_PARAMETERS = {
    "line_start_xy": [0.156, -0.024],
    "line_end_xy": [0.398, -0.018],
    "push_direction_xy": [1.0, 0.025],
    "sheet_corners_xy": [[0.142, 0.070], [0.144, -0.111], [0.422, -0.110], [0.417, 0.080]],
    "box_x": [0.092, 0.450],
    "box_y": [-0.161, 0.130],
    "table_plane": {"a": 0.0090, "b": -0.0055, "c": 0.0047},
    "tool_pitch_rad": 1.57,
    "setup_by_tape_measure": {
        "gap_base_front_to_paper_near_edge_cm": 5,
        "paper_far_edge_to_table_far_edge_cm": 2,
        "base_to_table_side_edge_cm": 7,
        "photo": "06_deliverables/team_m_project6_documentation/figures/setup_2026-10-25_overhead.jpg"
    }
}


# The side-view test of 4.11.26 (side_view_test.py): forearm roll (joint 4) turned so the camera looks along a pull
# towards the base. The rod cannot be kept vertical when turned, except where the forearm is vertical; the lean is
# calculated from the arm model (about 1.5, 9.5 and 14 degrees at 20, 25 and 28 cm with a 90 degree turn), not measured.
SIDE_VIEW_TEST_PARAMETERS = {
    "distances_m": [0.20, 0.25, 0.28],
    "line_y_m": -0.02,
    "turn_deg": 90.0,
    "turn_increment_deg": 30.0,
    "lift_height_m": 0.05,
    "step_m": 0.01,
    "pull_steps": 10,
    "minimum_distance_m": 0.17,
    # Lean is limited tightly with the tip down, where the rod meets the puck, and loosely with it lifted, where the
    # same turn leans more (calc: about 17 degrees at 25 cm) but touches nothing.
    "maximum_lean_deg": 16.0,
    "maximum_lifted_lean_deg": 30.0,
    "model_tolerance_mm": 3.0,
    "velocity_percent": 20,
    "settle_s": 0.4,
    "fresh_picture_timeout_s": 3.0,
    "home_joints": [0.0, 0.5, -1.25, 0.0, 0.0, 0.0],
    "output_folder": "03_data/raw/camera_check/side_view_test"
}


def table_height(
        x: float,
        y: float
) -> float:
    """The table surface height in metres at (x, y), from the plane fitted to the four corners."""

    plane = WORKSPACE_PARAMETERS["table_plane"]
    return plane["a"] * x + plane["b"] * y + plane["c"]


def pushing_height(
        x: float,
        y: float
) -> float:
    """
    Where the robot's reported tool point should be while pushing at (x, y).

    The pusher's tip sits the clearance above the table; the reported point is the
    bare rod's end, which is the extension length higher than the tip.
    """

    return (table_height(x, y) + PUCK_PARAMETERS["pusher_clearance_above_sheet_m"]
            + PUSHER_PARAMETERS["extension_below_rod_end_m"])


def rod_pose_behind(point_xy: list) -> list:
    """
    The rod's pose when it stands behind a puck whose centre is at ``point_xy``.

    Behind means on the robot's side along the push direction, with the rod's side
    touching the puck's side, at pushing height, rod pointing straight down.
    """

    direction_x, direction_y = WORKSPACE_PARAMETERS["push_direction_xy"]
    offset = PUCK_PARAMETERS["diameter_m"] / 2 + PUSHER_PARAMETERS["tip_diameter_m"] / 2
    x = point_xy[0] - offset * direction_x
    y = point_xy[1] - offset * direction_y
    return [x, y, pushing_height(x, y), 0.0, WORKSPACE_PARAMETERS["tool_pitch_rad"], 0.0]


def start_pose() -> list:
    """The rod behind the puck at the start of the line."""

    return rod_pose_behind(WORKSPACE_PARAMETERS["line_start_xy"])


def goal_pose() -> list:
    """The rod behind the puck when the puck has reached the end of the line."""

    return rod_pose_behind(WORKSPACE_PARAMETERS["line_end_xy"])


# The recording tool (record_episodes.py), designed 28.10.26. Actions are 1 cm steps of the pusher's tip, named in
# the PICTURE's frame, as Emranul and Marshal decided on 28.10.26: "forward" is towards the top of the picture,
# "left" towards its left side. The angle is measured from "forward", counter-clockwise seen from above. Stop has no
# angle; it is recorded as its own step so the model learns when to stop. Keys are the ones pressed in the
# recording window; Esc always quits. "steps_until_fresh_picture" new pictures are waited for after a move has
# returned before the picture is taken, because a picture finished during the move can still arrive just after it
# (a stale frame was found in run 1419, step 23). Hand-guided episodes are replayed by the robot from the path the
# hand drew, snapped back onto that path at every step; "guided_path_sample_m" is how far the tip must move before
# the next path point is logged.
# Changed 4.11.26 (Marshal): step_over_forward and step_over_back added for reversing direction, keys r and f. A group
# of directional steps around the puck was rejected: a model without memory cannot tell how far round it has got, and
# those pictures look like the off-centre ones where the stepping rule says to go back behind the puck.
# Changed 4.11.26 (Marshal): back_left and back_right added, because a pull towards the base could otherwise only be
# steered by stopping to shift sideways; 1 cm steps at 135 degrees to each side of forward, keys z and c.
# Changed 31.10.26 after the data-collection decisions (06_deliverables/research_notes/Recording tool/
# 2026-10-30_data_collection_decisions.md): a replay step runs only when Enter is pressed (a direction key overrides,
# no timeout); "u" marks the previous step as a mistake and "l" marks the current picture as "line lost", both without
# a new action class; the step budget is 1.5 times the line length per sheet (the meander needs 116, not 80); the
# workspace guard checks the predicted puck centre, not only the rod; every episode records the camera mount and the
# version of the picture-to-table mapping, so pictures from different mounts are never mixed in one training set.
RECORDING_PARAMETERS = {
    "step_m": 0.01,
    "actions": {
        "forward": 0,
        # 45 until test 2; 60 since (Emranul, 4.11.26: sharper diagonals for the tight turns of the meander). Steeper
        # than about 60 the block tends to slide along the puck instead of pushing it (pushing physics, not measured).
        "forward_left": 60,
        "forward_right": -60,
        "left": 90,
        "right": -90,
        "back": 180,
        "back_left": 135,
        "back_right": -135,
        "stop": None
    },
    # One decision, one lift-across-lower move over the puck, for reversing direction. The length is contact to contact
    # across the puck (twice "puck_centre_ahead_of_tip_m") plus a gap so the tip does not land on it (calc, untested).
    "step_over_actions": {
        "step_over_forward": 0,
        "step_over_back": 180
    },
    "step_over_m": (PUCK_PARAMETERS["diameter_m"] + PUSHER_PARAMETERS["tip_diameter_m"]) + 0.005,
    "keys": {
        "w": "forward",
        "q": "forward_left",
        "e": "forward_right",
        "a": "left",
        "d": "right",
        "s": "back",
        "y": "back_left",  # y, not z: Emranul's German keyboard has y where z sits on an English one (4.11.26)
        "c": "back_right",
        " ": "stop",
        "x": "discard",
        "u": "mistake",
        "l": "lost",
        "enter": "accept",
        "n": "lean_left",
        "m": "lean_right",
        "v": "turn_left",
        "b": "turn_right"
    },
    # 4.11.26 evening (Emranul): the camera moved to the forearm cover and the forearm turn keys v / b stay bound; the
    # picture then turns with the forearm (see "forearm_camera_turns_with_joint_4").
    # Hand flick (4.11.26 evening, after test 1, Emranul): the hand turns fast about the rod by "flick_actions" degrees
    # (joint 6 only, the rod stays where it is) and straight back. Bound to q / e for test 2 only; after test 2 q / e
    # are the diagonal steps again and the flick has no key (code kept, bound again by adding keys here). Left is
    # counter-clockwise seen from above when picture_left_sign is 1.
    "flick_actions": {
        "flick_left": 45,
        "flick_right": -45
    },
    "flick_velocity_percent": 60,
    # 4.11.26, Emranul, after merging Marshal's keys of the same day: the back diagonals (z c) and the step-over keys
    # (r f) are not bound; the seven classes plus the camera turn are what the team records with. The actions and the
    # step-over code stay in the tool for Marshal's stepping rule v1, bound again by adding the keys here. The lean test
    # keys moved from y / c to n / m and the camera turn keys from r / t to v (side view) / b (back).
    # Hand turn about the vertical rod at the start of every episode, in degrees; positive = towards picture-left of
    # the untouched hand. 4.11.26: Emranul tried 90 (camera beside the rod, looking along the line both ways) as the
    # default for two episodes and reverted to 0 the same day: the camera ahead of the rod, as on 31.10 to 3.11. The
    # side view stays available during an episode with the turn keys below.
    "hand_turn_deg": 0,
    # Forearm turn keys (v = turn towards +, b = towards -), Emranul's request of 4.11.26 after the hand-guided trial: the
    # FOREARM (joint 4) turns by "turn_actions" degrees per press, between "min_turn_deg" and "max_turn_deg", so the
    # camera swings to the side and sees the line ahead of and behind the rod. The magnet tip stays on its spot and
    # the rod leans only as much as the arm's geometry forces, found with Marshal's solver from side_view_test.py
    # (least lean, calc: about 1.5 degrees at 20 cm, 9.5 at 25 cm, 14 at 28 cm from the base with a 90 degree turn;
    # refused above "max_turn_lean_deg"). The turn is done lifted by "approach_height_m" in "turn_increment_deg"
    # steps, then the tip comes down again; while turned, every step is solved the same way. Before the first turn of
    # an episode the arm model is checked against the robot's own kinematics ("turn_model_tolerance_mm"). The keys
    # follow the picture: w is towards the top of the turned picture (Emranul steers from the picture). Logged as
    # turn_left / turn_right with the forearm angle in every row. A hand turn about the vertical rod (joint 6, no
    # lean) was tried earlier the same day and dropped: it stopped 16 degrees short on both sides (cable or motor).
    # The table direction of the first push at the start point, in degrees from +x: the rod starts behind the puck
    # on the opposite side. 0 for sheets laid with the start end nearest the robot and the line running away.
    "start_push_direction_deg": 0.0,
    "turn_actions": {
        # 45 per press until test 3; 30 since (Emranul, 4.11.26): at the far end of sheet 04 (37 to 42 cm from the base)
        # a 90 or even 45 degree turn needs more than the 28 degree rod lean, a 30 degree turn does not (calc).
        "turn_left": 30,
        "turn_right": -30
    },
    "min_turn_deg": -90,  # both sides since 4.11.26 (Emranul): v turns towards +, b towards -
    "max_turn_deg": 90,
    # After a turn the tip stands at least this far back from the puck, opposite to the last push, instead of on its old
    # spot: the block turns with the forearm and after 90 degrees its wide side (37 mm) overlaps the puck's place by
    # about 12 mm (calc), and on 4.11.26 the tip came down onto the puck. The tool keeps count of how far the block
    # already stands back (back steps add to it, pushing steps clear it) and only moves the rest, so two turns in a row
    # or a turn after backing off do not carry the tip far away (Emranul, 4.11.26: "landing in a far away place").
    # 2.5 cm until test 2; 1.5 cm since (Emranul, 4.11.26: the step back was too much). The wide side overlaps the
    # puck's place by about 12 mm after 90 degrees (calc), so 15 mm leaves about 3 mm.
    "turn_landing_retreat_m": 0.015,
    # Intermediate heights for the rise and the descent of a turn, one solved position each. [0.01, 0.02] until test
    # 2; empty since (Emranul, 4.11.26: up in one move, down in one move).
    "turn_near_paper_heights_m": [],
    # How high the tip is lifted for a forearm turn (Emranul, 4.11.26: 5 cm, 3 cm after test 1, 2.5 cm after test 2).
    "turn_lift_m": 0.025,
    # Smaller steps of the lifted turn keep the lifted tip closer to its spot (was 30, from side_view_test.py).
    "turn_increment_deg": 15.0,
    # Camera on the forearm cover: True if the cover turns with joint 4, so the picture turns with the forearm and the
    # keys follow it. NOT YET CHECKED on the robot (4.11.26): the first "v" press shows it; if the picture does not
    # turn, set False.
    "forearm_camera_turns_with_joint_4": True,
    # MEASURED 4.11.26 by Emranul (session 2026-11-04, path_04_meander_episode_028, a measurement run with the limit
    # raised to 30): forearm turned 90 degrees, the puck pushed 31 steps from 32 to 38.5 cm from the base while the
    # rod's lean grew from 9.7 to 30 degrees; the puck never tipped and stayed on the block. Limit: 28, two degrees
    # under the largest lean pushed. One run, one direction (outwards); a sideways push with the block's wide face
    # is not measured. Marshal's side_view_test.py keeps its own, more cautious 16.
    "max_turn_lean_deg": 28.0,
    # Lifted, the rod touches nothing, so only the arm's reach matters; 45 leaves room for the retreat at 38 cm (34.6).
    "max_turn_lifted_lean_deg": 45.0,
    "turn_model_tolerance_mm": SIDE_VIEW_TEST_PARAMETERS["model_tolerance_mm"],
    # TEST keys of 31.10.26 (Emranul and Marshal asked for them after the camera lost the line at the meander's
    # hairpins): rod and camera lean together about the camera's forward axis by "lean_step_deg" per press, at most
    # "max_lean_deg" either way, the rod's end staying in place. Not one of the seven classes; logged as
    # lean_left / lean_right so their effect on the pictures and on the puck can be judged afterwards. The magnet tip
    # (2.6 cm below the rod's end) moves about 4.5 mm sideways per 10 degrees (calc).
    "lean_actions": {
        "lean_left": 10,
        "lean_right": -10
    },
    "max_lean_deg": 20,
    "steps_until_fresh_picture": 2,
    "fresh_picture_timeout_s": 3.0,
    # The step budget is 1.5 times the line length in centimetres, per sheet; "step_budget" is the fallback for a
    # sheet whose length is unknown. Lengths: sheet 01 measured (24 cm); 02 to 06 traced from the sheet photos on
    # 26.10.26 (06_deliverables/research_notes/Camera mount options/scripts/sheet_paths.json, calc, about 5 percent
    # uncertain; the dashed sheet 06 is taken as long as the loop it copies). Replace with the SVG arc lengths.
    "step_budget_per_cm": 2.0,  # was 1.5; the meander took 116 steps on 2.11 with its hairpins and ran out
    "step_budget": 80,
    # 4.11.26, Emranul: the tool no longer ends an episode when the budget is reached (episode 018 was cut at 154
    # steps on the meander); the budget is still written into episode.json so long episodes can be found afterwards.
    "is_step_budget_enforced": False,
    "line_length_cm": {
        "path_01_straight_line": 24,
        "path_02_arc": 27,
        "path_03_two_curve": 55,
        "path_04_meander": 77,
        "path_05_alpha_loop": 57,
        "path_06_alpha_loop_dashed": 57
    },
    "approach_height_m": 0.05,
    # Parking pose. Changed 2.11.26 with bracket F: at the old pose (wrist joint 5 at 0.0) the bracket stops the wrist
    # about 2 degrees before 0, the motor pushed against that stop and raised hardware error 32 (overload) twice,
    # 10 to 15 s after arriving. With the wrist tilted down to -0.3 rad the pose held for 60 s with zero errors and
    # 0.1 degree of drift (hold_test.py). Rule: never command the wrist (joint 5) above about -0.05 rad with the bracket.
    "home_joints": [0.0, 0.5, -1.25, 0.0, -0.3, 0.0],
    # The wrist's safe range with bracket F. Measured by hand on 2.11.26 (wrist_limit_by_hand.py, motors off): the
    # bracket's camera body meets the forearm at joint 5 = -0.035 rad (-2 degrees); with 3 degrees of margin the limit
    # is -0.09 rad. At the pushing pose 14 cm from the base the robot needs the wrist at about 0 rad, so that pose is
    # out of reach with this bracket. RobotLink refuses any pose whose inverse kinematics put joint 5 above this value;
    # find_wrist_limit.py prints where the usable range begins.
    # Bracket F3 fitted on 3.11.26: its post leans 25 degrees away from the forearm. Measured by hand the same day
    # (Emranul, wrist_limit_by_hand.py, motors off): the bracket meets the forearm cover at joint 5 = +0.321 rad
    # (+18.4 degrees), as the design predicted; with 3 degrees of margin the limit is +0.27 rad. The pushing poses
    # from about 14 cm from the base are reachable again with a vertical rod (bracket F: from about 25 cm).
    # 4.11.26 evening: the bracket left the hand (camera taped to the forearm cover), so the hand is bare and the
    # wrist may use the robot's own range again; 1.5 rad (86 degrees) is in effect no limit for the pushing poses.
    "wrist_pitch_max_rad": 1.5,
    # Automatic lean (2.11.26, Emranul's test on sheet 01 with bracket F): where a vertical rod would need the wrist
    # above the limit, RobotLink leans the rod about the sideways axis, tip ahead of the wrist, in steps of
    # "auto_lean_step_deg" up to "auto_lean_max_deg", until the wrist is within the limit; the camera tilts with it
    # (looks further ahead) and the magnet tip shifts about 4.5 mm per 10 degrees. Logged by the terminal line.
    "auto_lean_step_deg": 5.0,
    "auto_lean_max_deg": 25.0,
    "guided_path_sample_m": 0.002,
    "guided_path_poll_s": 0.05,
    # The puck's centre sits this far beyond the rod's end along the push (puck radius plus tip radius); the
    # workspace guard refuses a step that would carry it outside the box or beyond the sheet plus "sheet_margin_m".
    "puck_centre_ahead_of_tip_m": PUCK_PARAMETERS["diameter_m"] / 2 + PUSHER_PARAMETERS["tip_diameter_m"] / 2,
    "sheet_margin_m": 0.02,
    # Written into every episode.json. Change "camera_mount" when the bracket is fitted and "mapping_version" whenever
    # camera_heading_rad, picture_left_sign or the actions change; a training set must never mix two values of either.
    # Bracket F fitted on 2.11.26 (Emranul and Marshal): the camera sits about 5 cm ahead of the rod axis and 6 cm above
    # the wrist axis, looking nearly straight down; lens about 12 cm above the paper at pushing height; sharpness 160 to
    # 172 at every height tried, no refocus. The picture's forward direction and left sign are re-tested in the
    # direction test of the same day; see the mapping version below.
    # Bracket F3 fitted on 3.11.26 (Emranul): same camera orientation on the plate as F, but the post leans 25 degrees
    # so the lens sits about 2.5 cm further ahead of the rod and the plate is tilted 30 degrees back (F: 32). The
    # heading formula and the left sign are expected to be unchanged (guess, not yet tested on F3): re-run the
    # direction test (three "left" steps) before the first real episode.
    # v3 (4.11.26): the heading is read from the tool's orientation instead of atan2(y, x), see camera_heading_rad;
    # the turn actions were added. Episodes of v2 on sheet 04 carry step labels up to 24 degrees off the picture frame
    # (the logged x, y of consecutive steps give the true direction; relabel before training).
    # v4 (4.11.26, same day): the hand held turned 90 degrees (side camera); withdrawn the same day.
    # v5 (4.11.26): hand untouched again; the actions measured from the pusher's direction; dropped the same day.
    # v6 (4.11.26): actions in the PICTURE's frame again, read from the camera's current direction, also while the
    # forearm is turned (v / b keys); the forearm angle is in every row.
    # v7 (4.11.26 evening, Emranul and Marshal): the camera, in its F3 bracket, is taped with double-sided tape to
    # the black forearm cover under the elbow, looking down the forearm towards the wrist and beyond. The picture
    # no longer follows the hand: its top points the way the forearm points, away from the base, so the heading is
    # the direction from the base to the rod ("camera_heading_model" forearm). NOT YET TESTED: the first episode
    # starts with three "left" steps; the content must move RIGHT, else picture_left_sign becomes -1.
    # v8 (4.11.26 evening, test 2): q / e were hand flicks instead of diagonal steps; forearm turns lifted 3 cm.
    # v9 (4.11.26 evening, after test 2): q / e diagonal steps again (as v7); turns lift 2.5 cm in one move each way and
    # land 1.5 cm back from the puck.
    # v10 (4.11.26 evening, before test 3): q / e diagonal steps at 60 degrees instead of 45.
    # v11 (4.11.26 evening, after test 3): forearm turns in 30 degree steps (30, 60, 90) instead of 45.
    # v12 (4.11.26 evening, during test 4): back-left / back-right (z / c, Marshal's 135 degree steps) bound again
    # (Emranul); every step row also stores the picture's direction on the table ("picture_heading_deg").
    "camera_mount": "forearm_cover_tape_F3_bracket",
    "mapping_version": "picture_frame_v12_forearm_camera_back_diagonals_2026-11-04",
    # "hand": the picture's top follows the hand's orientation (camera on the hand: stock holder, bracket F, F3);
    # "forearm": it follows the arm's direction from the base (camera on the forearm cover, since 4.11.26 evening).
    "camera_heading_model": "forearm",
    # The direction the top of the picture points to, measured from the model's reference direction: 0 for the hand
    # mounts (tested in the direction tests below) and, untested, 0 for the forearm camera.
    "picture_top_offset_rad": 0.0,
    # TESTED 31.10.26 on the robot with the stock wrist holder (Emranul, session 2026-10-31_session_01_emranul,
    # episode 001, steps 2 to 4): three "left" steps moved the tip from y = -0.028 m to +0.001 m and the line in the
    # picture from column about 150 to about 430, i.e. the picture content moved RIGHT, so picture-left is
    # counter-clockwise from the camera heading seen from above and this sign is 1.
    # TESTED again for bracket F on 2.11.26 (session 2026-11-02_session_01_emranul, path_04_meander_episode_013, steps 1
    # to 3): three left steps moved the tip by +0.9 cm in y each and the line's end in the picture from column about
    # 300 to about 450, i.e. the content moved RIGHT: the sign stays 1 with bracket F.
    "picture_left_sign": 1,
    # Raise this whenever 06_deliverables/research_notes/Recording tool/stepping_rule_v1.md changes its version
    # (Marshal, 4.11.26).
    "stepping_rule_version": "v1",
    # Asked after each episode (Marshal, 4.11.26); Esc at either question stores "not_recorded".
    "outcome_keys": {
        "y": "puck_in_goal",
        "n": "puck_not_in_goal"
    },
    "discard_reason_keys": {
        "t": "puck_tipped",
        "o": "puck_off_sheet",
        "r": "robot_stopped",
        "h": "touched_by_hand",
        "m": "other"
    },
    "raw_folder": "03_data/raw"
}


def step_budget_for(path_id: str) -> int:
    """
    The most steps one episode on sheet ``path_id`` may take: 1.5 times the line length in centimetres.

    Falls back to ``RECORDING_PARAMETERS["step_budget"]`` for a sheet whose length is not listed.
    """

    length_cm = RECORDING_PARAMETERS["line_length_cm"].get(path_id)
    if length_cm is None:
        return RECORDING_PARAMETERS["step_budget"]
    return int(math.ceil(RECORDING_PARAMETERS["step_budget_per_cm"] * length_cm))


def camera_heading_rad(
        pose: list,
        model: str = RECORDING_PARAMETERS["camera_heading_model"]
) -> float:
    """
    The table direction that the top of the picture points to, for the rod's ``pose`` = [x, y, z, roll, pitch, yaw].

    ``model`` "hand": the camera sits on the hand, so the picture's top follows the hand's orientation, not the arm's
    direction. With the orientation the tool asks for ([roll 0, pitch 1.57, yaw 0]) the hand is held the same way
    everywhere: joint 6 cancels joint 1, and the camera looks along the table's +x axis at every point (checked
    4.11.26 on all 154 steps of path_04_meander_episode_018: heading 0.0 +- 0.4 degrees from the robot's own
    orientation reading, while atan2(y, x) gave up to -24 degrees). The heading is the direction of the tool's z axis
    in the table plane (pyniryo's roll, pitch, yaw are extrinsic rotations about x, y, z).

    ``model`` "forearm": the camera sits on the forearm cover (since 4.11.26 evening) and looks down the forearm, which
    lies in the vertical plane through the base and the rod, so the picture's top points from the base towards the
    rod: atan2(y, x). Both plus the mount's "picture_top_offset_rad".
    """

    if model == "forearm":
        return math.atan2(pose[1], pose[0]) + RECORDING_PARAMETERS["picture_top_offset_rad"]

    roll, pitch, yaw = pose[3:6]
    forward_x = math.sin(pitch) * math.cos(roll) * math.cos(yaw) + math.sin(roll) * math.sin(yaw)
    forward_y = math.sin(pitch) * math.cos(roll) * math.sin(yaw) - math.sin(roll) * math.cos(yaw)
    return math.atan2(forward_y, forward_x) + RECORDING_PARAMETERS["picture_top_offset_rad"]


def push_heading_rad(
        pose: list,
        hand_turn_deg: float = RECORDING_PARAMETERS["hand_turn_deg"],
        heading_rad: Optional[float] = None
) -> float:
    """
    The table direction the pusher pushes in, for the rod's ``pose``: the camera heading with the hand's current turn
    ``hand_turn_deg`` taken off. The actions are measured from this direction (4.11.26, Emranul: with a turned camera
    the keys were confusing in the picture frame; "forward" is the way the pusher pushes whatever the camera does).
    The camera is fixed to the hand, so for the model this is the picture frame turned by the logged turn angle.
    ``heading_rad`` replaces camera_heading_rad(pose) when the caller knows the picture's direction better (the
    recording tool computes it from the joints for the camera on the forearm).
    """

    picture_heading = camera_heading_rad(pose) if heading_rad is None else heading_rad
    return picture_heading - RECORDING_PARAMETERS["picture_left_sign"] * math.radians(hand_turn_deg)


def action_step_to_table(
        action: str,
        pose: list,
        hand_turn_deg: float = RECORDING_PARAMETERS["hand_turn_deg"],
        heading_rad: Optional[float] = None
) -> list:
    """The table displacement [dx, dy] in metres of one ``action`` (a step or a step-over) taken with the rod at ``pose``."""

    is_step_over = action in RECORDING_PARAMETERS["step_over_actions"]
    action_angle = RECORDING_PARAMETERS["step_over_actions"][action] if is_step_over else RECORDING_PARAMETERS["actions"][action]
    length = RECORDING_PARAMETERS["step_over_m"] if is_step_over else RECORDING_PARAMETERS["step_m"]
    angle = push_heading_rad(pose, hand_turn_deg, heading_rad) + RECORDING_PARAMETERS["picture_left_sign"] * math.radians(action_angle)
    return [length * math.cos(angle), length * math.sin(angle)]
