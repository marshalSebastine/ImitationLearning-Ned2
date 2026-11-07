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

# Taught on the first-floor table on 6.11.26 by touching six points with the magnet tip (teach_positions.py; raw
# readings in 03_data/raw/workspace/2026-11-06_positions.json). The values are the magnet tip, recomputed from the logged
# joints with the arm model (the rod leaned up to 44 degrees at the far corners, which puts the reported rod end up to
# 2 cm beside the touched spot). The line end touch landed on the paper's far edge (line_end picture), about 2.5 cm past
# the ink, so the line end is the start plus sheet 01's measured 24.3 cm of ink along the touched direction. Table plane
# fitted to all six touches: tilt 0.95 degrees, every touch within 1.8 mm of it. The values of 25.10.26 (old desk):
# start [0.156, -0.024], end [0.398, -0.018], corners [[0.142, 0.070], [0.144, -0.111], [0.422, -0.110],
# [0.417, 0.080]], plane a 0.0090, b -0.0055, c 0.0047.
WORKSPACE_PARAMETERS = {
    "line_start_xy": [0.159, -0.015],
    "line_end_xy": [0.402, -0.017],
    "push_direction_xy": [1.000, -0.009],
    "sheet_corners_xy": [[0.151, 0.074], [0.150, -0.107], [0.442, -0.132], [0.445, 0.080]],
    "box_x": [0.100, 0.450],
    "box_y": [-0.182, 0.130],
    "table_plane": {"a": 0.0158, "b": 0.0048, "c": 0.0056},
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


# The local recording page (recording_page.py, 6.11.26): a form that starts record_episodes.py with the right options.
# The start codes and sheet notes come from 06_deliverables/research_notes/Recording tool/2026-11-06_episode_plan_per_sheet.md;
# a start that "needs_taught_start" has no default position, so its --start-x and --start-y must be filled in.
RECORDING_PAGE_PARAMETERS = {
    "host": "127.0.0.1",
    "port": 8765,
    "log_folder": "03_data/recording_page_logs",
    "log_lines_shown": 200,
    # The page asks for its status every second; git takes up to 11 s per call on the lab laptop (6.11.26), so the code
    # version is read at most once in this many seconds.
    "code_state_cache_s": 60,
    # Teaching a start point from the page (teach_sheet_start.py, 6.11.26): the page's "Enter" button answers its prompts.
    "teach_script": "teach_sheet_start.py",
    # Sharpness warning on the page (Emranul, 6.11.26): a popup when the median sharpness (variance of the Laplacian) of
    # the running episode's last "sharpness_rows" pictures, or of its camera check picture before the first step, is
    # under "sharpness_warning_below". Measured: 4 to 5.11.26 episode medians 133 to 178; 6.11.26 with a hazy lens 37;
    # after cleaning the lens 184 (episode 003). 100 lies between (a choice, not a measured limit).
    "sharpness_warning_below": 100.0,
    "sharpness_rows": 5,
    "operators": ["emranul", "marshal"],
    "modes": {
        "gamepad": "Gamepad",
        "hand_guided": "Hand guided, replay in 1 cm steps",
        "hand_guided_exact": "Hand guided, replay of every joint"
    },
    "sheets": {
        "path_01_straight_line": {"number": "01", "label": "01 straight line", "plan": "20 episodes, either end"},
        "path_02_arc": {"number": "02", "label": "02 arc", "plan": "35 episodes, both ends"},
        "path_03_two_curve": {"number": "03", "label": "03 two curves", "plan": "45 episodes, end A first, end B if it reaches"},
        "path_04_meander": {"number": "04", "label": "04 meander", "plan": "45 episodes, end A, end B if it reaches"},
        "path_05_alpha_loop": {"number": "05", "label": "05 loop with a crossing", "plan": "30 episodes, the lower end (B) only"},
        "path_06_alpha_loop_dashed": {"number": "06", "label": "06 dashed loop", "plan": "25 episodes, the dashed tail (A) only"}
    },
    "line_ends": ["A", "B"],
    "start_codes": {
        "STD": {"label": "STD: on the start mark", "offset_cm": 0.0, "offset_deg": 0.0, "start_index": 0, "needs_taught_start": False},
        "OL1": {"label": "OL1: puck 1 cm left", "offset_cm": 1.0, "offset_deg": 0.0, "start_index": 0, "needs_taught_start": False},
        "OR1": {"label": "OR1: puck 1 cm right", "offset_cm": -1.0, "offset_deg": 0.0, "start_index": 0, "needs_taught_start": False},
        "OL1.5": {"label": "OL1.5: puck 1.5 cm left", "offset_cm": 1.5, "offset_deg": 0.0, "start_index": 0, "needs_taught_start": False},
        "OR1.5": {"label": "OR1.5: puck 1.5 cm right", "offset_cm": -1.5, "offset_deg": 0.0, "start_index": 0, "needs_taught_start": False},
        "TL10": {"label": "TL10: sheet turned 10 degrees left", "offset_cm": 0.0, "offset_deg": 10.0, "start_index": 0, "needs_taught_start": False},
        "TR10": {"label": "TR10: sheet turned 10 degrees right", "offset_cm": 0.0, "offset_deg": -10.0, "start_index": 0, "needs_taught_start": False},
        "TL15": {"label": "TL15: sheet turned 15 degrees left", "offset_cm": 0.0, "offset_deg": 15.0, "start_index": 0, "needs_taught_start": False},
        "TR15": {"label": "TR15: sheet turned 15 degrees right", "offset_cm": 0.0, "offset_deg": -15.0, "start_index": 0, "needs_taught_start": False},
        "NE": {"label": "NE: 4 to 8 cm before the end", "offset_cm": 0.0, "offset_deg": 0.0, "start_index": 1, "needs_taught_start": True},
        "MID": {"label": "MID: before a hard spot", "offset_cm": 0.0, "offset_deg": 0.0, "start_index": 2, "needs_taught_start": True},
        "X1": {"label": "X: before the crossing, first pass", "offset_cm": 0.0, "offset_deg": 0.0, "start_index": 2, "needs_taught_start": True},
        "X2": {"label": "X: before the crossing, second pass", "offset_cm": 0.0, "offset_deg": 0.0, "start_index": 3, "needs_taught_start": True}
    },
    "simulated_raw_folder": "03_data/raw_simulated",
    # The score given after each episode (6.11.26, Marshal), so that bad episodes can be left out later. The issues are the
    # rules of "How to demonstrate" in the episode plan.
    "score_levels": {
        "3": "Good: a clean example of how to demonstrate",
        "2": "Usable: small flaws, but the model may learn from it",
        "1": "Bad: leave out of training and record it again"
    },
    # The lowest score that counts a plan episode as done. 1 since 6.11.26 evening (Emranul): every finished episode is
    # training data, score 1 included, so a score of 1 is not recorded again.
    "done_minimum_score": 1,
    "score_issues": {
        "not_steady": "Not smooth or steady",
        "lost_contact": "Block lost contact with the puck",
        "off_line": "Puck's centre left the line",
        "rod_lean": "Rod leaned where it did not need to",
        "forearm_turn": "Forearm turned without need, or not turned back",
        "crossing": "Crossing or dashed gap not taken straight",
        "offset_recovery": "Offset start not back on the line within 5 to 10 cm",
        "finish": "Did not finish within about 1 cm of the end cap",
        "start": "Hand in the first picture, or puck not placed well",
        "other": "Something else (say what in the note)"
    },
    # What each kind of start and each sheet asks for, from the episode plan; shown as "Set up this episode".
    "start_setups": {
        "STD": "Start cap on the START MARK, the line's first centimetres pointing straight away from the robot. Puck centred on the start cap.",
        "OL1": "As STD, but the puck 1 cm LEFT of the start cap, seen from behind the robot. Bring it back onto the line in the first 5 to 10 cm.",
        "OR1": "As STD, but the puck 1 cm RIGHT of the start cap, seen from behind the robot. Bring it back onto the line in the first 5 to 10 cm.",
        "OL1.5": "As STD, but the puck 1.5 cm LEFT of the start cap, seen from behind the robot. Bring it back onto the line in the first 5 to 10 cm.",
        "OR1.5": "As STD, but the puck 1.5 cm RIGHT of the start cap, seen from behind the robot. Bring it back onto the line in the first 5 to 10 cm.",
        "TL10": "The sheet turned 10 degrees LEFT about the start cap; the cap stays on the mark, so the first push meets the line at an angle.",
        "TR10": "The sheet turned 10 degrees RIGHT about the start cap; the cap stays on the mark, so the first push meets the line at an angle.",
        "TL15": "The sheet turned 15 degrees LEFT about the start cap; the cap stays on the mark, so the first push meets the line at an angle.",
        "TR15": "The sheet turned 15 degrees RIGHT about the start cap; the cap stays on the mark, so the first push meets the line at an angle.",
        "NE": "The puck on the line 4 to 8 cm before the end cap. Teach the point once per sheet layout with teach_sheet_start.py, then place the puck's centre half a centimetre further from the robot than the point you touched.",
        "MID": "A start before a hard spot: 3 cm before the sharp turn (sheet 03) or at the middle hairpin (sheet 04). Teach the point once per sheet layout with teach_sheet_start.py, then place the puck's centre half a centimetre further from the robot than the point you touched.",
        "X1": "5 cm before the crossing, first pass. Teach the point once per sheet layout with teach_sheet_start.py, then place the puck's centre half a centimetre further from the robot than the point you touched.",
        "X2": "5 cm before the crossing, second pass. Teach the point once per sheet layout with teach_sheet_start.py, then place the puck's centre half a centimetre further from the robot than the point you touched."
    },
    "sheet_setups": {
        "path_01_straight_line": "No sheet marks a start: either end may be A. For end B, turn the sheet round so the other cap is on the mark.",
        "path_02_arc": "From end A the arc bends to the robot's left; from end B it bends to the right. Both are needed.",
        "path_03_two_curve": "End B is tight for reach (calc: 0.2 cm to spare). Use it only if its trial reached the goal without strain.",
        "path_04_meander": "Keep the rod upright; let it lean only where the arm cannot reach otherwise, at the far end of this sheet.",
        "path_05_alpha_loop": "Start from the lower end (B) only: from the other end the paper would lie under the robot's base. At the crossing, go straight across, never round the corner.",
        "path_06_alpha_loop_dashed": "Start from end A only: the end halfway up the sheet, about 4 cm from its right edge in the course photo. Keep the dashes' direction across the gaps, and go straight across the crossing."
    },
    "demonstration_rules": [
        "Push smoothly at a steady, unhurried speed. Block behind the puck and touching it; puck's centre on the line.",
        "Keep the rod upright; let it lean only where the arm cannot reach otherwise.",
        "Turn the forearm only when the line ahead would leave the picture, and turn it back on straight stretches.",
        "At a crossing, go straight across. On the dashed sheet, keep the direction across the gaps.",
        "Finish with the puck's centre on the end cap, within about 1 cm, then stop.",
        "Anything else that slips more than about 1 cm off the line: discard (x) and do it again.",
        "Before Enter at the start: puck placed, hands out of the picture. Never touch the puck once the robot is down."
    ],
    "sheet_placement_figure": "06_deliverables/research_notes/Recording tool/figures/2026-11-06_rule_a_placements.png",
    # One set-up picture per plan number, plan_<NN-NN>.png (setup_visuals.py, 6.11.26), shown in "Set up this episode".
    "setup_picture_folder": "06_deliverables/research_notes/Recording tool/figures/setup_per_episode",
    # Round 1 of the plan: enough for the first model. The page offers these before anything else.
    "first_round": {"sheets": ["path_01_straight_line", "path_02_arc", "path_04_meander"], "episodes_per_sheet": 12},
    # The episode plan per sheet: (first number, last number, kind of start or one per number, start end, what to know).
    "episode_plan": {
        "path_01_straight_line": [
            (1, 4, "STD", "A", ""), (5, 8, ["OL1", "OR1", "TL15", "TR15"], "A", ""),
            (9, 12, "NE", "A", "Puck 6 cm before the end cap."),
            (13, 16, "STD", "B", "The other end as the start: sheet turned round, the other cap on the mark."),
            (17, 18, ["OL1.5", "OR1.5"], "A", ""), (19, 20, "NE", "A", "Puck 4 cm before the end cap.")
        ],
        "path_02_arc": [
            (1, 3, "STD", "A", ""), (4, 6, "STD", "B", ""), (7, 8, ["OL1", "TR10"], "A", ""), (9, 10, ["OR1", "TL10"], "B", ""),
            (11, 12, "NE", "A", ""), (13, 15, "STD", "A", ""), (16, 18, "STD", "B", ""),
            (19, 21, ["OL1.5", "OR1.5", "TL15"], "A", ""), (22, 24, ["OL1.5", "OR1.5", "TR15"], "B", ""),
            (25, 28, "NE", "B", ""), (29, 30, "STD", "A", ""), (31, 32, "STD", "B", ""),
            (33, 35, "NE", "A", "Puck 4 cm before the end cap.")
        ],
        "path_03_two_curve": [
            (1, 5, "STD", "A", ""), (6, 9, ["OL1", "OR1", "TL10", "TR10"], "A", ""), (10, 12, "NE", "A", ""),
            (13, 17, "STD", "B", "Only if end B's trial reached the goal without strain; otherwise use end A."),
            (18, 21, ["OL1", "OR1", "TL10", "TR10"], "B", "Only if end B's trial reached the goal; otherwise use end A."),
            (22, 24, "NE", "B", "Only if end B's trial reached the goal; otherwise use end A."),
            (25, 27, "STD", "A", ""), (28, 30, "STD", "B", ""),
            (31, 33, ["OL1.5", "OR1.5", "TL15"], "A", ""), (34, 36, ["OL1.5", "OR1.5", "TR15"], "B", ""),
            (37, 41, "MID", "A", "3 cm before the sharp turn, from the end you use most."),
            (42, 43, "NE", "A", ""), (44, 45, "NE", "B", "")
        ],
        "path_04_meander": [
            (1, 5, "STD", "A", ""), (6, 9, ["OL1", "OR1", "TL10", "TR10"], "A", ""), (10, 12, "NE", "A", ""), (13, 18, "STD", "A", ""),
            (19, 24, ["OL1.5", "OR1.5", "TL15", "TR15", "OL1", "OR1"], "A", ""),
            (25, 29, "MID", "A", "At the middle hairpin."),
            (30, 34, "STD", "B", "Only if end B's trial reached the goal; otherwise use end A."),
            (35, 38, ["OL1", "OR1", "TL10", "TR10"], "B", "End B, or end A if its trial did not reach the goal."),
            (39, 42, "NE", "A", "4 to 6 cm before the end cap."), (43, 45, "NE", "B", "")
        ],
        "path_05_alpha_loop": [
            (1, 6, "STD", "B", ""), (7, 10, ["OL1", "OR1", "TL10", "TR10"], "B", ""),
            (11, 13, "X1", "B", ""), (14, 19, "STD", "B", ""),
            (20, 24, ["OL1.5", "OR1.5", "TL15", "TR15", "OL1"], "B", ""),
            (25, 27, "X2", "B", ""), (28, 30, "NE", "B", "")
        ],
        "path_06_alpha_loop_dashed": [
            (1, 6, "STD", "A", ""), (7, 10, ["OL1", "OR1", "TL10", "TR10"], "A", "For OL1, the puck beside the first dash."),
            (11, 13, "X1", "A", ""), (14, 18, "STD", "A", ""),
            (19, 21, ["OL1.5", "OR1.5", "TL15"], "A", ""), (22, 25, "NE", "A", "")
        ]
    },
    "safety_checks": [
        "NiryoStudio and every camera viewer are closed",
        "The Stop button is in reach and the table is clear around the robot",
        "\"Check robot\" says Calibration needed: False",
        "This table's positions are taught (config.py), and the sheet is taped down"
    ],
    # Which safety checks the page handles itself: the calibration box follows the result of "Check robot" for this many
    # minutes; the boxes in "safety_checks_every_start" must be ticked again for every episode, the others until the sheet changes.
    "safety_check_calibration": 2,
    "robot_check_valid_minutes": 60,
    "safety_checks_every_start": [1]
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


# Gamepad for the recording tool's gamepad mode (5.11.26, Emranul): a third-party PS4-style pad (HOME and TURBO
# buttons) on a USB cable, which Windows and pygame see as "Xbox 360 Controller" (USB VID 045E PID 028E, 6 axes, 11
# buttons, 1 hat); Bluetooth pairing did not work. Numbers are pygame's (SDL 2) axis and button numbers through
# SDL's XInput driver (gamepad.py turns SDL's raw-input driver off: through it only the triggers arrived, the sticks
# and buttons stayed at zero, seen 5.11.26). Buttons in Xbox order: A 0 = cross, B 1 = circle, X 2 = square,
# Y 3 = triangle, LB 4 = L1, RB 5 = R1, Back 6 = SHARE, Start 7 = OPTIONS. Axes in SDL's XInput order: left stick
# 0 and 1, L2 2, right stick 3 and 4, R2 5 (the triggers rest at -1 on 2 and 5, seen 5.11.26; which stick axis is
# which is SDL's order, still to be confirmed with gamepad_check.py). Stick values below "dead_zone" count as zero;
# axes in "inverted_axes" are flipped so that up and right are positive; "trigger_axes" are read as 0 (released) to
# 1 (fully pressed). Another computer or pad may number them differently: run gamepad_check.py there first.
GAMEPAD_PARAMETERS = {
    "axes": {"move_x": 0, "move_y": 1, "down": 2, "twist": 3, "lean": 4, "up": 5},
    "trigger_axes": ["down", "up"],
    "inverted_axes": ["move_y", "lean"],
    "buttons": {"stop": 0, "discard": 1, "mistake": 2, "lost": 3, "quit": 7, "hand_left": 4, "hand_right": 5},
    "dead_zone": 0.25
}


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
    # When the believed puck position is further than this from the block, it is taken as lost (test 5 of 4.11.26: it
    # drifted to 22 cm) and a turn steps back opposite to the last step instead.
    "puck_estimate_lost_m": 0.04,
    # The orientation map draws that believed position (the hollow "puck?") beside the puck the camera sees once the two
    # are further apart than this, in the views where a forearm turn lands back from it (keyboard and the 1 cm replay),
    # so the operator sees where a turn would land from (review of 6.11.26: the seen puck had hidden it).
    "puck_guess_shown_beyond_m": 0.015,
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
    # Exact hand-guided replay (mode hand_guided_exact, Emranul 4.11.26: "move any way we like with the motors off and
    # replicate that same movement"): a waypoint every "exact_waypoint_m" of tip travel or "exact_waypoint_joint_deg" of
    # any joint, each visited joint by joint with a picture before it. Each step is labelled with the nearest fixed move
    # (a turn when the forearm changes by "exact_turn_label_deg" or more, "small_motion" when the tip moves less than
    # "exact_small_motion_m"); the exact movement is stored beside the label.
    # Camera shift check at the start of every episode (5.11.26): the picture's bottom strip, [top, bottom, left,
    # right] as fractions of the picture, holds the robot's hand and the block, which look the same at the same pose;
    # a shift of more than "camera_shift_warning_px" from the reference picture means the camera has moved on its mount.
    "camera_check_region": [0.8, 1.0, 0.25, 0.75],
    "camera_shift_warning_px": 4.0,
    "exact_waypoint_m": 0.01,
    "exact_waypoint_joint_deg": 5.0,
    "exact_turn_label_deg": 15.0,
    # A lift in joint space, for when the straight 5 cm lift is refused (6.11.26: refused with error -3 after every hand
    # correction of sheet 01, so the arm went to the parking pose and back). Calculated with the arm model at the three
    # correction poses of 6.11.26: these changes raise the magnet tip 38 mm and move it at most 2 mm sideways.
    "joint_lift_change_rad": {"joint_2": 0.12, "joint_3": -0.05},
    # Putting the puck back after a hand correction: the live picture shows a circle where the puck was; within this
    # many pixels of its centre the status says "good" (the puck is about 60 px across near the block; chosen).
    "puck_back_good_px": 5,
    # Two labels beside the move name (Emranul and Marshal, 6.11.26): the rod's lean towards the picture's top or
    # bottom (episode 01-10: leaned so the black box on the hand no longer hid the puck) and the hand's turn about the
    # rod (joint 6; pushing with the block's corner at hairpins). A change of at least this many degrees in one step is
    # labelled; smaller changes are "none". The review of 20 episodes (6.11.26 evening) found the lean growing by a
    # median 1.6 degrees per step to 40 to 50 degrees, so 3 degrees labelled only 24 of 671 steps: the lean threshold
    # is 1 degree since, and every row also carries the rod's lean itself (rod_lean_deg). The forearm's turns grew by
    # under 12 degrees per step, below "exact_turn_label_deg", so no step was a turn: forearm_label marks a forearm
    # change of at least "forearm_label_deg" beside the move name.
    "lean_label_deg": 1.0,
    "hand_label_deg": 3.0,
    "forearm_label_deg": 3.0,
    # 7.11.26 (Emranul): a label beside the move name saying the puck is at the goal, so the model learns when to stop
    # from more than the one stop picture per episode. The goal is where the magnet tip stood at the stop step; a step
    # is "at goal" when its tip is within this distance of it (chosen: about the last two pushes of 1 cm).
    "at_goal_within_cm": 2.0,
    "exact_small_motion_m": 0.003,
    # Gamepad mode (5.11.26, Emranul: the keyboard teleop kept hitting limits that hand guiding does not have). One step
    # per decision, read from the pad when a stick leaves its dead zone or a hand button is pressed:
    #   left stick  = move the tip in any direction of the picture, "pad_step_m" times how far the stick is pushed;
    #   right stick = left / right twists the forearm by up to "pad_twist_deg", up / down bends the wrist by up to
    #                 "pad_lean_deg" (the rod leans; nothing else holds it upright);
    #   L1 / R1     = turn the hand by "pad_hand_deg".
    # The arm's other joints follow to keep the tip on its target at pushing height. Only the robot's own joint limits,
    # the workspace and "pad_max_lean_deg" refuse a step: hand guiding with the puck went to 38.5 degrees of lean and
    # 119 degrees of forearm twist on 4 to 5.11.26 (calc from guided_path.csv), so 45 leaves room without tipping risk
    # being measured beyond 38.5. Each row stores the exact movement and the nearest fixed move as its label.
    "pad_step_m": 0.01,
    "pad_min_step_m": 0.003,
    "pad_twist_deg": 15.0,
    "pad_lean_deg": 5.0,
    "pad_hand_deg": 15.0,
    # 5.11.26 (Emranul: "all the freedom of hand guiding"): no lean limit of our own any more (89 = the rod may not
    # point sideways or up), and no "puck stays on the sheet" check; only the robot's joint limits and the workspace
    # box (the table edge) refuse a step. L2 / R2 lower / raise the tip by "pad_height_step_m" per full press, between
    # pushing height (never lower: the tip must not press into the table) and "pad_max_height_m" above it.
    "pad_max_lean_deg": 89.0,
    "pad_check_puck_on_sheet": False,
    "pad_height_step_m": 0.01,
    "pad_max_height_m": 0.08,
    "pad_poll_s": 0.03,
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
    # Mount name corrected 5.11.26 (Emranul): the camera on the forearm cover is the BARE camera, no F3 bracket.
    # Episodes up to 043 carry "forearm_cover_tape_F3_bracket"; from test 5 (episode 037) on that was already this
    # bare-camera view (camera check), see 06_deliverables/research_notes/Camera mount options/forearm_mount/.
    "camera_mount": "forearm_cover_tape_bare_camera",
    "mapping_version": "picture_frame_v12_forearm_camera_back_diagonals_2026-11-04",
    # "hand": the picture's top follows the hand's orientation (camera on the hand: stock holder, bracket F, F3);
    # "forearm": it follows the arm's direction from the base (camera on the forearm cover, since 4.11.26 evening).
    "camera_heading_model": "forearm",
    # The direction the top of the picture points to, measured from the model's reference direction: 0 for the hand
    # mounts (tested in the direction tests below) and 0 for the forearm camera (checked 6.11.26 against the camera
    # fitted to the recordings, FOREARM_CAMERA_PARAMETERS: the picture's up within 0.5 degrees).
    "picture_top_offset_rad": 0.0,
    # TESTED 31.10.26 on the robot with the stock wrist holder (Emranul, session 2026-10-31_session_01_emranul,
    # episode 001, steps 2 to 4): three "left" steps moved the tip from y = -0.028 m to +0.001 m and the line in the
    # picture from column about 150 to about 430, i.e. the picture content moved RIGHT, so picture-left is
    # counter-clockwise from the camera heading seen from above and this sign is 1.
    # TESTED again for bracket F on 2.11.26 (session 2026-11-02_session_01_emranul, path_04_meander_episode_013, steps 1
    # to 3): three left steps moved the tip by +0.9 cm in y each and the line's end in the picture from column about
    # 300 to about 450, i.e. the content moved RIGHT: the sign stays 1 with bracket F.
    # CHECKED 6.11.26 for the forearm camera against the camera fitted to the recordings (FOREARM_CAMERA_PARAMETERS;
    # test_puck_tracking.test_tool_directions_match_the_fitted_camera): a table point moved "left" moves left in the
    # picture, within 1 degree, with the forearm straight and turned up to 57 degrees either way. The sign stays 1.
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
    # The main data, recorded from 6.11.26 on (Emranul): episode numbers start again at 001 per sheet, apart from the
    # tool tests up to episode 048, which stay in "tool_test_folder" and are never training data.
    "raw_folder": "03_data/raw/main_data",
    "tool_test_folder": "03_data/raw"
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


# The forearm camera (the bare camera taped on the forearm cover since 4.11.26 evening), fitted on 6.11.26 without the
# robot to the recorded episodes 037 to 045 (sheet 04 on the old desk): the white sheet's straight edges in 1603
# pictures and the puck in 85 runs of pictures in which the robot cannot have moved it, each picture with its logged
# joints. Evidence and scripts: 06_deliverables/research_notes/Recording tool/2026-11-06_forearm_camera_calibration/.
# The camera is fixed in the arm model's "joint_4" frame (it turns with joint 4; joints 5 and 6 do not move it): 58 mm
# from the forearm's axis and 18 mm along it from the joint-4 origin, looking 6.2 degrees off the axis; focal length 502
# px (Niryo's file: 508), field of view 65 x 51 degrees, almost no lens distortion. Checked on pictures the fit never
# saw: the sheet's edges land within 2.0 mm (median) and 5.1 mm (90th percentile) on the table, a puck that does not
# move wanders 1.9 mm (median) on the map, and the end of the line in episode 039's first picture maps 1.1 mm from the
# start taught on 31.10. Each picture's view also scatters by about 6 px (about 2 mm on the table), which no calibration
# removes. Valid while the camera stays as taped (camera check under about 4 px), the focus ring is untouched and the
# robot's automatic calibration holds; table heights come from WORKSPACE_PARAMETERS["table_plane"], so teach it again on
# a new table first. The values before the fit (Niryo's lens, the camera's place guessed from photos) were 1 to 2 cm off
# on the table.
FOREARM_CAMERA_PARAMETERS = {
    "parent_frame": "joint_4",
    # axis-angle vector; the columns of its matrix are the camera's x (picture right), y (picture down) and z (viewing
    # direction) axes in the forearm frame
    "rotation_vector_in_forearm_rad": [0.056883, 0.105629, 1.578688],
    "translation_in_forearm_m": [0.057967, 0.002716, 0.018138],
    "focal_length_px": 501.968951,
    "principal_point_px": [333.786188, 246.388646],
    # one radial term: (x, y) distorted = (x, y) (1 + k1 r^2) in normalised coordinates
    "radial_distortion_k1": -0.019613,
    "image_size_px": [640, 480],
    # the mount these values belong to (RECORDING_PARAMETERS["camera_mount"]); with any other mount the recording tool
    # draws no table positions from the picture
    "camera_mount": "forearm_cover_tape_bare_camera",
    "version": "fit_recordings_037_045_2026-11-06"
}

# How the picture's outline on a table plane is traced (forearm_camera_model.picture_footprint_on_plane): points along
# each picture edge, the start corner included and the end corner left out (1 gives the four corners), and the range at
# which rays that miss the plane, or meet it further away, are cut off (the top of the picture can look above the
# horizon).
FOOTPRINT_SETTINGS = {
    # points along each picture edge, the start corner included and the end corner left out (1 gives the four corners)
    "points_per_edge": 12,
    # rays that miss the plane in front of the camera, or meet it further away than this, are cut off at this range
    "maximum_range_m": 1.0
}

# Settings for refitting the forearm camera (forearm_camera_model.fit_camera_parameters).
CAMERA_FIT_SETTINGS = {
    # names from FIT_PARAMETER_NAMES that keep their initial value
    "fixed_parameters": [],
    # "huber" or "cauchy" (robust), "soft_l1", or "linear" (plain least squares)
    "loss": "huber",
    # the expected pixel noise; every pixel residual is divided by it, so the costs are in units of this noise
    "pixel_sigma_px": 2.0,
    # residuals beyond this many sigmas count less (Huber: linearly, Cauchy: logarithmically)
    "loss_threshold_sigma": 2.0,
    # optional soft pulls towards the initial values, name -> sigma in the parameter's unit (rotations: towards 0)
    "prior_sigmas": {},
    # "auto" (scipy if installed), "scipy" or "numpy"
    "solver": "auto",
    "maximum_iterations": 200,
    # central-difference steps for the Jacobian
    "finite_difference_steps": {
        "camera_tilt_correction_rad": 1e-6,
        "camera_pan_correction_rad": 1e-6,
        "camera_roll_correction_rad": 1e-6,
        "translation_x_m": 1e-6,
        "translation_y_m": 1e-6,
        "translation_z_m": 1e-6,
        "focal_length_px": 1e-3,
        "principal_point_u_px": 1e-3,
        "principal_point_v_px": 1e-3,
        "radial_distortion_k1": 1e-6,
        "group_xy_m": 1e-6
    },
    # observations further than this many sigmas from the fitted model are listed as outliers
    "outlier_threshold_sigma": 3.0,
    # correlations above this size are listed in the conditioning report
    "strong_correlation": 0.9,
    # the pixel residual given to a point behind the camera or outside the lens model's field
    "invalid_projection_residual_px": 1000.0
}

# The red-puck detector (puck_detector.py), tuned on 6.11.26 on the 1692 pictures of episodes 037 to 045 (old desk, this
# room's lamps, sheet 04): the puck is found in 1390 of the 1391 pictures that show more than a sliver of it, with no
# false detection in a checked sample of 175 pictures nor in episode 039 (run without the puck, with a red mouse and
# red cables in view); 1.6 ms per picture. It misses slivers beside the block and pucks held in a hand, and it needs
# some white paper in view. Signed hue is OpenCV's hue minus 180 from 90 up, red at 0; saturation and value run from
# 0 to 255. Check it again on the first pictures of a new table or room.
PUCK_DETECTOR_PARAMETERS = {
    # candidates are searched on the picture scaled by this factor; lengths and areas below
    # are in full-size pixels and are scaled internally
    "analysis_scale": 0.5,
    # seed colour: signed hue window, minimum saturation and value (OpenCV scale, 0 to 255)
    "seed_hue_range": [-25, 2],
    "seed_saturation_minimum": 95,
    "seed_value_minimum": 45,
    # grow colour: the looser window a blob may extend into from its seeds
    "grow_hue_range": [-35, 7],
    "grow_saturation_minimum": 65,
    "grow_value_minimum": 35,
    # blue cast of the picture, blue over red on the paper (the brightest pixels with little colour,
    # sampled on a sparse grid): from the first value of the range on, the windows reach further
    # towards violet, by up to the given number of hue steps at the second value (4.11.26 evening:
    # cast about 1.25, the shaded puck at signed hue -55 to -45)
    "paper_sample_step_px": 8,
    "paper_brightest_share": 0.15,
    "paper_saturation_maximum": 115,
    "paper_value_minimum": 120,
    "paper_share_minimum": 0.02,
    "violet_extension_cast_range": [1.08, 1.18],
    "seed_violet_extension_maximum": 31,
    "grow_violet_extension_maximum": 25,
    # the cast also pales the puck in full light (median saturation 81 to 85 in episode 037), so the
    # saturation limits drop by up to these amounts at the full cast
    "seed_saturation_cast_reduction": 15,
    "grow_saturation_cast_reduction": 10,
    "pale_cast_reduction": 13,
    # morphology on the grow mask (kernel sizes at the analysis scale): an opening removes specks
    # and thin lines (cabinet edges, cables), a closing fills small notches such as a glint
    "opening_kernel_px": 3,
    "closing_kernel_px": 3,
    # colour tests: area limits, and the least seed share and median saturation; a blob whose median
    # hue lies beyond the base grow window (violet: the shaded puck in a blue-cast picture) must not be
    # as dark as the shadows and gaps that turn violet there (shaded puck: median value 61 and up,
    # dark gaps between cabinets: 50 to 52)
    "minimum_area_px": 120,
    "maximum_area_px": 9_000,
    "minimum_seed_fraction": 0.05,
    "minimum_median_saturation": 85,
    "violet_minimum_median_value": 56,
    # shape: area (with hidden pixels counted) over convex hull area, long over short axis of the
    # fitted ellipse (a half disc gives 1.9), area (with hidden pixels counted) over bounding box area,
    # for a whole puck and for one partly hidden by the hand or cut by the picture's edge (looser,
    # because the block's screw heads and the brackets' edges notch the visible part, and shaded
    # stretches of the hand are not always recognised as occluders, so their hidden pixels are not
    # counted)
    "minimum_completed_solidity": 0.80,
    "minimum_completed_solidity_partial": 0.65,
    "maximum_axis_ratio": 2.4,
    "maximum_axis_ratio_partial": 3.0,
    "minimum_completed_extent": 0.45,
    "minimum_completed_extent_partial": 0.30,
    # the hand that hides part of the puck (block and brackets) is at the bottom of the picture: a blob
    # reaching down to this row may be partly hidden even where the block's colour is not recognised
    # (slate blue in a blue-cast picture, like shaded paper), and a partly hidden puck that does not
    # touch the picture's edge must reach down to it (measured: row 441 and lower in episodes 037 to 045)
    "hand_region_top_v": 340,
    # occluders: the black block (dark) and the blue brackets of the hand. The brackets' saturation
    # was measured at about 100 to 180 in a neutral picture and 200 to 250 in a blue-cast one, where
    # the shaded paper reaches about 140, so above the hand region the saturation floor follows the
    # cast like the hue windows (first value for a neutral picture, second for the full cast); inside
    # the hand region it stays at the first value
    "occluder_value_maximum": 60,
    "occluder_blue_hue_range": [95, 130],
    "occluder_blue_saturation_range": [100, 170],
    "occluder_blue_value_minimum": 50,
    "occluded_below_fraction_minimum": 0.35,
    "occluded_light_above_fraction_minimum": 0.20,
    "adjacent_occluder_fraction_minimum": 0.10,
    # surroundings: a ring this wide around the blob is sorted into light background (paper),
    # occluder and wood (or skin)
    "context_ring_px": 8,
    "light_value_minimum": 100,
    "light_saturation_maximum": 150,
    "wood_hue_range": [6, 30],
    "wood_saturation_minimum": 50,
    "wood_value_minimum": 60,
    "maximum_wood_fraction": 0.50,
    # a blob with less light background around it than this (a puck deep in the arm's shadow) is kept
    # only when it lies within the given distance of the usual position and is at least the given
    # share of the expected size (by its area, or by its enclosing circle when beside an occluder)
    "minimum_light_fraction": 0.08,
    "dark_surroundings_maximum_distance_px": 160.0,
    "dark_surroundings_minimum_size_ratio": 0.70,
    # size: apparent radius of a whole puck (measured 20 to 40 px, median 31 px, in 1312 confident
    # detections of episodes 037 to 045) and the accepted ratio to it; a partly hidden puck is
    # sized by the smallest circle that holds its visible part
    "expected_radius_px": 31.0,
    "size_ratio_range": [0.55, 1.6],
    # centre of a partly hidden puck: a circle fitted at full size to the outline points that touch
    # neither an occluder nor the picture's edge, refitted on its inliers to shed edges against
    # shadow or the hand's metal; its centre replaces the visible blob's centre when it fits well
    "circle_margin_px": 40,
    "circle_minimum_free_points": 30,
    "circle_inlier_px": 2.0,
    "circle_inlier_fraction_of_radius": 0.05,
    "circle_refit_rounds": 3,
    "circle_minimum_inlier_share": 0.5,
    "circle_minimum_arc_deg": 100.0,
    "circle_radius_range_px": [18.0, 48.0],
    "circle_maximum_residual_fraction": 0.06,
    "circle_minimum_inside_fraction": 0.90,
    # where the puck usually sits (in front of the block: median 333, 363 in 1312 confident
    # detections) and how quickly the prior fades
    "expected_centre_uv": [325.0, 360.0],
    "expected_centre_sigma_px": 200.0,
    # score weights; the score of the chosen blob is returned as its confidence
    "score_weights": {
        "colour": 0.30,
        "shape": 0.20,
        "surroundings": 0.20,
        "size": 0.15,
        "position": 0.15
    },
    "picture_edge_score_factor": 0.85,
    "partial_score_factor": 0.90,
    # how the partial scores are formed: full colour marks from this seed share and from this saturation above the pale
    # limit, full shape marks up to this axis ratio (a puck seen from the side), and an occluder in the ring around the
    # blob counts this much towards its surroundings (light background counts fully)
    "full_colour_seed_fraction": 0.6,
    "full_colour_saturation_span": 70.0,
    "full_shape_axis_ratio": 1.4,
    "occluder_surroundings_weight": 0.5,
    # an outline point within this kernel of an occluder (at full size) is not used for the circle fit, a blob pixel
    # up to this far beyond the fitted radius still counts as inside the circle, and blobs above this multiple of
    # "maximum_area_px" are dropped before they are measured
    "touch_kernel_px": 5,
    "circle_inside_tolerance_px": 3.0,
    "candidate_area_limit_factor": 2.0,
    "minimum_score": 0.50,
    # full-size measurement of the chosen blob: margin around its box, kernels at full size, and how
    # many rows below the blob's lowest point are searched for anything dark or blue (is_bottom_clear)
    "refine_margin_px": 12,
    "refine_opening_kernel_px": 3,
    "refine_closing_kernel_px": 5,
    "bottom_occluder_search_px": 4
}

# The puck and the picture's outline on the recording tool's orientation map (puck_tracking.py). Display only: nothing
# the robot does depends on them (Emranul, 5.11.26).
PUCK_TRACKING_PARAMETERS = {
    # the detector's centre is the middle of the puck's visible silhouette (its top face and near side), about half-way
    # up the 2.9 cm puck (calc); the fit could not pin it down, and 5 mm here moves the puck about 1 mm on the map
    "puck_point_height_above_table_m": 0.0145,
    # sightings with a lower detector score (0 to 1; measured 0.55 to 0.97, median 0.88) are not shown
    "minimum_confidence": 0.5,
    # the table's height under a pixel is not known before the pixel is placed: the ray is followed to the table this
    # many times (at least once), each time at the table's height where the previous pass landed (the plane tilts by
    # about 1 mm in 10 cm)
    "height_passes": 2,
    # the outline drawn on the map: points along each picture edge (1 = the four corners; more would show the lens's
    # slight bending, too small to see on the map), with rays beyond this range cut off (the map shows about 42 cm)
    "outline_points_per_edge": 1,
    "outline_maximum_range_m": 0.6,
    # the camera mount in use; the recording tool passes its own --camera-mount
    "camera_mount": RECORDING_PARAMETERS["camera_mount"]
}


# The episode checker (check_episodes.py, 6.11.26): every threshold, with the reason for its value. Pass overrides as a
# dictionary (or a JSON file with --settings); they are merged over these defaults (dictionaries key by key). Where the
# recording tool already has the number, it is read from RECORDING_PARAMETERS so the two cannot drift apart.
CHECK_PARAMETERS = {
    # Where the sessions are, from the project folder; the default session is the newest folder named
    # <yyyy-mm-dd>_session_NN_<operator>.
    "raw_folder": RECORDING_PARAMETERS["raw_folder"],
    # Rows and pictures. The tool writes exactly one picture per logged step, as the robot sent it, and takes the
    # picture back when a step is refused; so any missing, extra or repeated picture (same md5: the camera sent
    # nothing new between two steps) is a fault. Allowed counts, per episode.
    "missing_frames_allowed": 0,
    "extra_frames_allowed": 0,
    "duplicate_frames_allowed": 0,
    "picture_size": [640, 480],
    # The camera on its mount: the recording tool's own limit, measured on the same bottom strip of the picture (the
    # robot's hand and the block, which look the same at the start pose).
    "camera_shift_warning_px": RECORDING_PARAMETERS["camera_shift_warning_px"],
    "camera_check_region": RECORDING_PARAMETERS["camera_check_region"],
    # The camera mount and picture mapping that training uses now; a training set must never mix two of either
    # (config.py). Episodes with another value get a warning.
    "current_camera_mount": RECORDING_PARAMETERS["camera_mount"],
    "current_mapping_version": RECORDING_PARAMETERS["mapping_version"],
    # Mount names written wrongly at recording time and corrected afterwards: episodes 037 to 043 of sheet 04 say
    # "F3 bracket" but show the bare camera (session notes of 5.11.26, 21:03, and the comment in config.py).
    "camera_mount_corrections": [
        {"path_id": "path_04_meander", "first_episode": 37, "last_episode": 43,
         "written": "forearm_cover_tape_F3_bracket", "actual": "forearm_cover_tape_bare_camera"}
    ],
    # The start. Row 0 is logged at the start pose; the robot lands within about 0.5 mm of a commanded pose (survey of
    # 24.10.26), and the hand-guided modes come down onto the start again after guiding. More than this between row
    # 0's reported rod end and episode.json's start_pose means the episode did not begin where it says.
    "start_offset_warning_mm": 5.0,
    # Rod lean (calc, arm model). A note above the keyboard tool's limit: the largest lean pushed with the puck in a
    # measurement run was 30 degrees (4.11.26, episode 028), and the tool refuses turned keyboard steps above 28
    # (max_turn_lean_deg). A warning above the largest lean ever reached with the puck, 38.5 degrees in hand guiding
    # on 4 to 5.11.26 (calc from guided_path.csv, config.py): beyond it tipping is untested and the view is new.
    "lean_note_deg": RECORDING_PARAMETERS["max_turn_lean_deg"],
    "lean_warning_deg": 38.5,
    # Table steps of the magnet tip between two pictures (calc). A 1 cm step, a turn's landing (at most 1.5 cm back)
    # and a gamepad step stay well under this; a longer jump means the arm went somewhere else between two pictures.
    # A step-over (4.2 cm) is allowed its own length plus "step_over_jump_allowance_mm".
    "step_jump_warning_mm": 30.0,
    "step_over_jump_allowance_mm": 10.0,
    # Labels against the measured tip motion in the picture's frame (heading at the row's picture). Below
    # "small_motion_m" a motion is small_motion (the tool's own rule); "small_motion_tolerance_m" either side of that
    # boundary is not counted against a label. A moving label agrees when it is the class nearest the measured
    # direction, or within "label_boundary_tolerance_deg" of being it. A turn label agrees when joint 4 turned at
    # least "turn_minimum_deg" the labelled way (turn_left = joint 4 grows).
    "small_motion_m": RECORDING_PARAMETERS["exact_small_motion_m"],
    "small_motion_tolerance_m": 0.001,
    "label_boundary_tolerance_deg": 5.0,
    "turn_minimum_deg": 5.0,
    # Above this share of disagreeing labels the episode is flagged; single disagreements are listed as a note.
    "label_disagreement_warning_fraction": 0.05,
    # More small_motion rows than this share gets a note: small_motion is not one of the moving classes, and training
    # has to decide what to do with them.
    "small_motion_note_fraction": 0.10,
    # The arm model against the robot's reported rod end, row by row (both read in the same state). They agree to
    # about 0.02 mm in most rows; a row far apart suggests that joints and pose were read at different moments (a
    # guess: the arm may not have settled). Noted above this distance when more than the given share of rows is.
    "model_mismatch_mm": 2.0,
    "model_mismatch_note_fraction": 0.10,
    # Replay fidelity of the hand-guided modes: distance from each replayed magnet tip (calc) to the path the hand
    # drew (guided_path.csv, or the correction path after an "h"). The exact replay visits the guided joints, so it
    # should sit on the path within about a millimetre; the 1 cm replay is snapped back onto the path at every step,
    # so a few millimetres. The end: the last logged tip against the path's last point; the puck's goal tolerance is
    # 12 mm (PUCK_PARAMETERS), and the last row of a quit or discarded episode is logged one step before its end.
    "replay_median_warning_mm": 3.0,
    "replay_max_warning_mm": 10.0,
    "replay_end_warning_mm": 15.0,
    # A correction path belongs to a replay segment when its first point lies this close to the segment's first tip.
    "correction_match_mm": 10.0,
    # Timing. The tool waits for two new pictures after each move before it takes one. In every episode of 4 and
    # 5.11.26 that wait was 0.04 to 0.08 s at the least and about 0.12 s in the median (so the camera sends new
    # pictures faster than the 9 per second measured on 24.10.26); a wait under "picture_wait_note_s" is unusual
    # and noted (it does not prove a picture taken during the move). Pauses longer than "long_pause_s" between two
    # pictures are noted too (nothing wrong by themselves).
    "picture_wait_note_s": 0.04,
    "long_pause_s": 60.0,
    # Sharpness (variance of the Laplacian, logged per row; it depends on the scene as much as on focus). The forearm
    # camera's pictures of 4 to 5.11.26 lie at 49 to 290 (medians 133 to 178), the out-of-focus bracket pictures of
    # 4.11.26 afternoon at 9 to 31: under "sharpness_warning_below" a picture is out of focus. A picture under
    # "sharpness_drop_fraction" of the median of its neighbours ("sharpness_neighbours_each_side" before it and as many
    # after it; the scene changes little from one step to the next) is probably blurred by motion: noted.
    "sharpness_warning_below": 40.0,
    "sharpness_drop_fraction": 0.5,
    "sharpness_neighbours_each_side": 3,
    # The red puck in the pictures, found by puck_detector.py (PUCK_DETECTOR_PARAMETERS). At the start it must sit right
    # in front of the block, its centre inside this region ([top, bottom, left, right] as fractions of the picture; the
    # first pictures of episodes 037, 042 and 043 put it at u 321 to 327 and v 304 to 337 of 640 x 480).
    "puck_start_region": [0.5, 0.85, 0.35, 0.65],
    # In the last picture of an episode whose outcome is puck_in_goal, a sighting counts only from this detector score
    # (0 to 1) up: the far background can hold red things (episode 035's last picture: a red object beside a pen cup
    # scored 0.50), while the real puck at the goal scored 0.90 in episode 045. A weaker sighting gives the "look at
    # it" warning, which costs one look.
    "puck_last_picture_minimum_confidence": 0.6,
    # Below this share of pictures with the puck seen, a note: the camera lost the puck often (or the detector missed
    # it).
    "puck_seen_note_fraction": 0.5,
    # Overrides for the checker's puck detector (PUCK_DETECTOR_PARAMETERS), for example from --settings after the
    # detector was checked again in a new room; empty = the detector as in config.
    "detector_parameters": {},
    # Decode the pictures (puck and camera-shift checks); md5 sums are always checked.
    "is_picture_check_on": True
}


# Set-up pictures, one per plan number (setup_visuals.py, 6.11.26, Emranul's request): a top view in robot coordinates of
# where the sheet lies, where the start mark and start cap are and where the puck goes. Read-only beside the recording
# page. The lines come from the photo-traced sheets (2026-11-06_sheet_measurements.json, about 0.3 cm; the long-edge
# position of sheets 04 and 05 is uncertain by 0.7 and 0.3 cm); end A is the traced line's "first" end, end B its "last"
# (as in figures/2026-11-06_sheets.png). The start mark is WORKSPACE_PARAMETERS["line_start_xy"]: the taught point is
# where the ink begins, and the line's first centimetres point along the push direction. NE, MID and X positions are not
# taught yet: they are drawn from the plan's words ("near_end_cm" when the plan names no distance; the sharp turn and
# the hairpin as curvature peaks of the traced line; the crossing as the traced line's self-crossing), so they are a
# guide, not where the robot will go. The base square and the reach ring are as in the Rule A figure (calc, not measured).
SETUP_VISUALS_PARAMETERS = {
    "sheet_measurements": "06_deliverables/research_notes/Recording tool/2026-11-06_sheet_measurements.json",
    "output_folder": "06_deliverables/research_notes/Recording tool/figures/setup_per_episode",
    "end_keys": {"A": "first", "B": "last"},
    "paper_size_cm": [21.0, 29.7],
    "near_end_cm": 6.0,
    "crossing_lead_cm": 5.0,
    "sharp_turn_lead_cm": 3.0,
    # Which curvature peak is the plan's hard spot, counted from the start end: sheet 03's sharp turn is its tightest
    # bend, sheet 04's "middle hairpin" the middle one of its hairpins (tightest bends, at least this far apart).
    "mid_spots": {"path_03_two_curve": "tightest", "path_04_meander": "middle_hairpin"},
    "hairpin_count": 3,
    "hairpin_separation_cm": 6.0,
    "curvature_smoothing_cm": 1.0,
    "base_half_width_cm": 9.4,
    "reach_cm": [13.0, 44.0],
    "puck_diameter_cm": PUCK_PARAMETERS["diameter_m"] * 100,
    "view_x_cm": [-12.0, 50.0],
    "view_y_cm": [-26.0, 30.0],
    "figure_size_inches": [6.4, 6.0],
    "dots_per_inch": 110
}
