"""
Machine-wide defaults for the robot code.

Every script reads its settings from these dictionaries, so a value is changed
in one place only. Later phases add the keys named in the work plan.
"""

import math
import tempfile
from pathlib import Path
from typing import Dict, Optional

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
# Picture check after each episode (check_pictures.py; Emranul, 7.11.26): pictures that may not belong in the training
# data are flagged on the page while the episode is scored. Values from sheet 01 of 6.11.26 (736 pictures).
PICTURE_CHECK_PARAMETERS = {
    # Per picture, as the page's warning: sharp pictures showing mostly white paper measured 55 to 95 on 6.11.26 and
    # 40 to 60 in the dim room of 7.11.26 (lights off on purpose); the hazy lens of 6.11.26 gave 37 (chosen 30).
    "blurred_below": 30.0,
    # Dark: HSV value under this; the black box, the camera housing and the blue arm sit in "robot_area_px" (left, top,
    # right, bottom), where the camera on the forearm always sees them. The laptop and a person at the edge measured
    # 1,800 to 60,000 px; small dark bits of clean table under 1,200 px (chosen limit between).
    "dark_value_below": 70,
    "robot_area_px": [200, 270, 600, 480],
    "dark_patch_above_px": 1500,
    # With the forearm turned further than this, the robot's own blue base comes into the picture's side (episode 008,
    # turns of 73 to 105 degrees): the dark check is skipped there.
    "dark_check_turn_most_deg": 45.0,
    # Change between neighbouring pictures, mean grey difference at this size: a step of sheet 01 changed it by a median
    # 4 to 9; a hand taken out after the first picture by 12 to 54. Flagged above this many times the episode's median,
    # and above "change_minimum" (chosen).
    "change_picture_size": [80, 60],
    "change_ratio": 3.0,
    "change_minimum": 10.0,
    # picture_check.jpg: the flagged pictures side by side.
    "sheet_tile_size": [320, 240],
    "sheet_columns": 4,
    "sheet_tiles_most": 24,
}

RECORDING_PAGE_PARAMETERS = {
    "host": "127.0.0.1",
    "port": 8765,
    # 7.11.26 (Emranul): True lets a second laptop on the same network (Marshal's) open the page at
    # http://<this laptop's address>:8765; the address is printed at the start. Anyone on that network can then start
    # the robot, so only on a hotspot the team controls, never on open university Wi-Fi. Windows asks once to let the
    # port through its firewall (allow for private networks only). False = this computer only.
    "is_shared_on_network": False,
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
    # 7.11.26 (Emranul): lowered from 100 to 30 for recording in a dim room on purpose (lights off, cloudy: sharp
    # pictures measured 40 to 60, checked by eye); a dirty lens in bright light gave 37, so a smeared picture is still
    # caught by the picture check only when it falls under 30.
    "sharpness_warning_below": 30.0,
    "sharpness_rows": 5,
    # Closing the page (Ctrl-C) waits at most this long for running picture checks (check_pictures.py), so that a score
    # saved during a check is still added to the folder's name. Until 12.11.26 saving a score waited for the check.
    "picture_check_wait_s": 60,
    "operators": ["emranul", "marshal"],
    "modes": {
        "gamepad": "Gamepad",
        "hand_guided": "Hand guided, replay in 1 cm steps",
        "hand_guided_exact": "Hand guided, replay of every joint"
    },
    "sheets": {
        "path_01_straight_line": {"number": "01", "label": "01 straight line", "plan": "20 episodes, either end"},
        "path_02_arc": {"number": "02", "label": "02 arc", "plan": "45 episodes, both ends, 16 with the sheet square"},
        "path_03_two_curve": {"number": "03", "label": "03 two curves", "plan": "45 episodes, end A first, end B if it reaches"},
        "path_04_meander": {"number": "04", "label": "04 meander", "plan": "45 episodes, end A, end B if it reaches"},
        "path_05_alpha_loop": {"number": "05", "label": "05 loop with a crossing", "plan": "30 episodes, end B, 9 from end A (14 to 19, 28 to 30)"},
        "path_06_alpha_loop_dashed": {"number": "06", "label": "06 dashed loop", "plan": "25 episodes, end A, 10 from end B (14 to 21, 24 to 25)"}
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
        # 7.11.26 (Emranul and Marshal): at the exam the sheet lies square to the robot, portrait or landscape, for a
        # tidy look; a line that starts at an angle (the arc) then meets the first push at that angle. The turn from
        # the STD lay-out is worked out per sheet and end (setup_visuals.square_turn_deg), so "offset_deg" stays 0 here.
        "SQP": {"label": "SQP: sheet square to the robot, portrait", "offset_cm": 0.0, "offset_deg": 0.0, "start_index": 0, "needs_taught_start": False,
                "lay_out": "portrait"},
        "SQL": {"label": "SQL: sheet square to the robot, landscape", "offset_cm": 0.0, "offset_deg": 0.0, "start_index": 0, "needs_taught_start": False,
                "lay_out": "landscape"},
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
        # 8.11.26 (Emranul): score 1 stays in the training data too (see "done_minimum_score"); the score only lets
        # the worst episodes be left out later if the model shows they hurt.
        "1": "Poor: big flaws, kept for training for now"
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
        "SQP": "Start cap on the START MARK, the sheet upright (PORTRAIT) with its edges square to the robot, as drawn; the first push meets the line at its own angle. Puck centred on the start cap.",
        "SQL": "Start cap on the START MARK, the sheet sideways (LANDSCAPE) with its edges square to the robot, as drawn; the first push meets the line at its own angle. Puck centred on the start cap.",
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
        "path_05_alpha_loop": "Mostly from the lower end (B); episodes 14 to 19 and 28 to 30 from end A (Emranul, 8.11.26), with the loop far from the robot. At the crossing, go straight across, never round the corner.",
        "path_06_alpha_loop_dashed": "Mostly from end A, the dashed tail halfway up the sheet, about 4 cm from its right edge in the course photo; episodes 14 to 21 and 24 to 25 from end B (Emranul, 9.11.26). Keep the dashes' direction across the gaps, and go straight across the crossing."
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
    # 13.11.26 (Emranul): how to tape the new-track mark and lay a new track, shown in the page's corner (drawn by
    # setup_visuals.py --new-tracks beside table_marks.pdf)
    "table_marks_figure": "06_deliverables/research_notes/Recording tool/figures/setup_per_episode/table_marks.png",
    # One set-up picture per plan number, plan_<NN-NN>.png (setup_visuals.py, 6.11.26), shown in "Set up this episode".
    "setup_picture_folder": "06_deliverables/research_notes/Recording tool/figures/setup_per_episode",
    # Round 1 of the plan: the page offers these before anything else. Until 12.11.26 the first 12 episodes of sheets 01,
    # 02 and 04 (all 210 course-sheet episodes were done by then); since 12.11.26 (Emranul) the pilot of the new tracks,
    # plans 1 and 2 of the solid prints of T01 to T04 (NEW_TRACK_RECORDING_PARAMETERS below). After the round the page
    # stays on a sheet while it has episodes left, then goes to the next sheet in this dictionary's order.
    "first_round": {"sheets": ["path_11_T01_solid", "path_12_T02_solid", "path_13_T03_solid", "path_14_T04_solid"], "episodes_per_sheet": 2},
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
            # 7.11.26 (Emranul and Marshal): square sheets as at the exam, 6 portrait (13 to 18, not yet recorded, were
            # STD) and 10 landscape (36 to 45, new); numbers already recorded keep their meaning.
            (11, 12, "NE", "A", ""), (13, 15, "SQP", "A", ""), (16, 18, "SQP", "B", ""),
            (19, 21, ["OL1.5", "OR1.5", "TL15"], "A", ""), (22, 24, ["OL1.5", "OR1.5", "TR15"], "B", ""),
            (25, 28, "NE", "B", ""), (29, 30, "STD", "A", ""), (31, 32, "STD", "B", ""),
            (33, 35, "NE", "A", "Puck 4 cm before the end cap."), (36, 40, "SQL", "A", ""), (41, 45, "SQL", "B", "")
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
            # 8.11.26 (Emranul, after episode 07): 14 to 19 and 28 to 30 from end A instead of B, so the model also sees
            # the loop far from the robot (robustness). Untested there: the flip at the loop's bottom is about 40 cm out.
            (11, 13, "X1", "B", ""), (14, 19, "STD", "A", "End A: the loop lies far from the robot. Check the first trial's rod lean at the flip."),
            (20, 24, ["OL1.5", "OR1.5", "TL15", "TR15", "OL1"], "B", ""),
            (25, 27, "X2", "B", ""), (28, 30, "NE", "A", "End A: near end B, at the loop's far side.")
        ],
        "path_06_alpha_loop_dashed": [
            (1, 6, "STD", "A", ""), (7, 10, ["OL1", "OR1", "TL10", "TR10"], "A", "For OL1, the puck beside the first dash."),
            # 9.11.26 (Emranul, before episode 02): 14 to 21 and 24 to 25 from end B instead of A, so the model also
            # sees the loop from the other side (variety). Untested there: from end B the paper may reach the base.
            (11, 13, "X1", "A", ""), (14, 18, "STD", "B", "End B: the other end of the line. Check the paper clears the base."),
            (19, 21, ["OL1.5", "OR1.5", "TL15"], "B", ""), (22, 23, "NE", "A", ""), (24, 25, "NE", "B", "End B: near end A.")
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
    # 8.11.26 (Emranul): moves that push nothing run at this speed cap instead of the usual 20 %: to and from the
    # parking pose, and to the place above a start (5 cm up, clear of the 2.9 cm puck). Every move at or near the
    # puck (the steps, the last move down, the lift off it, stepping over it) keeps the usual speed. 35 is a choice,
    # not yet tried on the robot. Only when the rod's end starts at least "travel_minimum_height_m" above the sheet,
    # so a move that begins low over the puck (after a refused lift, or a crash) keeps the usual speed.
    # 8.11.26: raised from 35 to 70, then to 90 by Emranul (the taped camera may shift under quick starts and stops:
    # compare a start picture with an earlier one after the first episodes).
    "travel_velocity_percent": 90,
    "travel_minimum_height_m": 0.04,
    # 12.11.26 (Emranul: "make sure to increase that thing from 20% to 90%, I will take care of the puck"): the moves
    # beside the puck that push nothing run at this speed cap: the lift off the puck or the line (record_episodes
    # __lift, its fall-back with joints 2 and 3, the low first stops of __turn_back_lifted, the lift at the start of
    # __return_to_joints_from_above when the rod starts low) and the last move down (the end of __lower_to_start and of
    # __return_to_joints_from_above). Every pushing move (the steps of all modes, the exact replay's waypoints, the
    # replay of a correction), the step-over and the hand flick keep their speed. 20 (ROBOT_PARAMETERS
    # "arm_max_velocity_percent", the arm's usual cap) gives back the behaviour before 12.11.26. Not tried on the robot.
    "near_puck_velocity_percent": 90,
    # 12.11.26 (Emranul): straight after hand guiding or a hand correction the straight lift was refused in every
    # episode (the log shows "inverse kinematics refused the pose (attempt 1 of 3)", "(attempt 2 of 3)", then "straight
    # lift refused ...; lifted with joints 2 and 3 instead"); True lifts with joints 2 and 3 at once there. False gives
    # back the old order (straight lift first). The time saved is a guess until timed at the robot.
    "is_lift_after_guiding_by_joints": True,
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
    # The forearm's turn (joint 4) at the start in degrees, 0 = straight; taught with teach_sheet_start.py since
    # 7.11.26 (Marshal), so that the camera sees the puck at starts on a bend.
    "start_turn_deg": 0.0,
    "turn_actions": {
        # 45 per press until test 3; 30 since (Emranul, 4.11.26): at the far end of sheet 04 (37 to 42 cm from the base)
        # a 90 or even 45 degree turn needs more than the 28 degree rod lean, a 30 degree turn does not (calc).
        "turn_left": 30,
        "turn_right": -30
    },
    # both sides since 4.11.26 (Emranul): v turns towards +, b towards -; 11.11.26 (Emranul): no tighter than about 2 to 3
    # degrees inside the hardware limit (the arm model's 119.75), so recordings can use the forearm's full swing
    "min_turn_deg": -117,
    "max_turn_deg": 117,
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
    # 7.11.26 (Marshal): the gap in mm on the table (forearm camera model) when the camera places the puck; "good"
    # within this many mm (chosen: a quarter of the 1 cm step). An enlarged view of the circle's surroundings,
    # this many pixels each way, this many times bigger, sits in the picture's top corner.
    "puck_back_good_mm": 2.5,
    "puck_back_zoom_half_px": 50,
    "puck_back_zoom_scale": 2,
    # 7.11.26 (Emranul): with the camera model, the puck's real shape where it stood (a cylinder 2.4 cm wide, 2.9 cm
    # tall, projected into the picture) instead of a circle; drawn from this many points per rim. The shaded side is
    # not red enough to count, so the puck fills 47 to 80% of its own shape (8 recorded pictures, 6 to 7.11.26): the
    # live fill is said as a share of the reference picture's fill, "good" from this share up (chosen).
    "puck_back_outline_points": 48,
    "puck_back_good_fill": 0.9,
    # 8.11.26 (Emranul: putting the puck back got less accurate): the green shape comes from the camera model and sits a
    # little small and low on a puck beside the block, so on sheet 04 the puck went back 8 to 21 mm off in 9 of 52
    # corrections (2 mm typical on 7.11). The arm stands where the reference was taken (within 0.4 degrees, all 52), so
    # the reference picture is laid over the live one at this share: the old puck shows as a faint ghost in its true
    # size and place, and the live puck is moved until the two merge (chosen; 0 turns it off).
    "puck_back_ghost_share": 0.35,
    # A puck the detector misses in the reference (partly hidden, 7.11.26): the red part's outline is drawn instead.
    # Red in HSV (OpenCV hue 0 to 180): hue under the first edge or over the second, with these saturation and value
    # (chosen from the puck's colour in the 6.11 pictures, value low enough for the dimmed room of 7.11). At least
    # "puck_red_min_area_px" red pixels count as the puck; "good" when the puck covers this share of the outline.
    "puck_red_hue_edges": [10, 165],
    "puck_red_min_saturation": 90,
    "puck_red_min_value": 50,
    "puck_red_min_area_px": 150,
    "puck_back_good_cover": 0.9,
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


# The 16 training tracks of 12.11.26 (make_test_tracks.py; tracks_2026-11-12/tracks.json), recorded on from 14.11.26.
# Each track is printed twice, solid and dashed, and each print is a sheet of its own on the recording page: sheet
# number 10 + n for the solid print of Tn and 30 + n for the dashed one (11 to 26 and 31 to 46, clear of the course
# sheets' 01 to 06), sheet name path_<number>_<track>_<print>. Every print is laid by the rule of tracks.json
# ("lay_out_rule", the same "mark_m" as TEST_TRACK_PARAMETERS["lay_out"]): the centre of end A's round cap on the
# new-track mark, 19.5 cm in front of the base axis and 1.5 cm to its right, the first straight piece straight away from
# the robot, not turned; all 16 pass the lay-out check from end A unturned (lay_outs.csv, calc with the arm model). The
# recorder's own start (WORKSPACE_PARAMETERS["line_start_xy"]) is not changed: the arm comes down with its tip at
# 14.0, -1.5 cm as for every course-sheet start, 5.5 cm behind the cap's centre, as in the 209 recordings (caps a median
# 19.2 cm out). Lengths: the drawn centre line between the two cap centres (tracks.json "length_mm", calc from the
# drawing). The plan (Emranul, 12.11.26): 58 episodes, 48 standard starts and 10 near-end starts; the pilot is plans 1
# and 2 of the solid prints of T01 to T04; every print is taped once and all its episodes follow each other, its
# near-end start last. 14.11.26 (Emranul): he records every episode himself, Marshal none ("we do only me"), so the
# pilot no longer splits its plans between the two and no plan says "either operator".
NEW_TRACK_RECORDING_PARAMETERS = {
    "tracks_file": Path(__file__).resolve().parent.parent / "06_deliverables" / "research_notes" / "Model and training" / "tracks_2026-11-12" / "tracks.json",
    "mark_m": [0.195, -0.015],
    "solid_number_offset": 10,
    "dashed_number_offset": 30,
    "near_end_note": "Puck 6 cm before the end cap.",
    "pilot_notes": ["Pilot: recorded by Emranul.", "Pilot: recorded by Emranul."],
    "either_operator_note": "Recorded by Emranul.",
    # name, label, length in cm, standard starts on the solid print, standard starts on the dashed print, and the print
    # that also gets a near-end start (solid for the even numbers T06 to T14, dashed for the odd numbers T05 to T13)
    "tracks": [
        ("T01", "gentle S-curves", 32.7, 2, 1, None),
        ("T02", "two sharp bends", 30.2, 2, 1, None),
        ("T03", "loop", 39.7, 2, 1, None),
        ("T04", "hairpin and medium bend", 40.8, 2, 1, None),
        ("T05", "loop and corner", 45.6, 1, 2, "dashed"),
        ("T06", "hairpin and sharp bend", 42.4, 2, 1, "solid"),
        ("T07", "two corners", 34.4, 1, 2, "dashed"),
        ("T08", "loop and hairpin", 53.0, 2, 1, "solid"),
        ("T09", "short gentle", 26.5, 1, 2, "dashed"),
        ("T10", "loop and medium bend", 46.4, 2, 1, "solid"),
        ("T11", "three corners", 45.5, 1, 2, "dashed"),
        ("T12", "long meander", 67.3, 2, 1, "solid"),
        ("T13", "gentle", 41.4, 1, 2, "dashed"),
        ("T14", "loop and corner", 52.2, 2, 1, "solid"),
        ("T15", "mixed bends", 57.2, 1, 2, None),
        ("T16", "long gentle", 46.8, 2, 1, None)
    ]
}


def new_track_sheet_id(
        track_name: str,
        print_kind: str,
        parameters: Dict = NEW_TRACK_RECORDING_PARAMETERS
) -> str:
    """
    The sheet name of one print of a new track, for example "path_13_T03_solid" or "path_33_T03_dashed".

    Parameters
    ----------
    track_name : str
        The track's name, "T01" to "T16".
    print_kind : str
        "solid" or "dashed".
    parameters : dict, optional
        NEW_TRACK_RECORDING_PARAMETERS or a copy with changes.

    Returns
    -------
    str
        The sheet name, which starts every episode folder's name.
    """

    offset = parameters["solid_number_offset"] if print_kind == "solid" else parameters["dashed_number_offset"]
    return f"path_{offset + int(track_name[1:]):02d}_{track_name}_{print_kind}"


def add_new_track_sheets(parameters: Dict = NEW_TRACK_RECORDING_PARAMETERS) -> Dict[str, str]:
    """
    Add both prints of every new track to the recording page's sheets, set-up notes and episode plan, and their line
    lengths to the step budgets, in track order (solid before dashed), so that the page offers them sheet by sheet.

    Parameters
    ----------
    parameters : dict, optional
        NEW_TRACK_RECORDING_PARAMETERS or a copy with changes.

    Returns
    -------
    dict
        Sheet name -> the name of its shape ("T03" for both prints of track T03), for the training code that comes later.
    """

    shapes = {}
    mark_x_cm, mark_y_cm = 100 * parameters["mark_m"][0], 100 * parameters["mark_m"][1]
    for name, label, length_cm, solid_starts, dashed_starts, near_end_print in parameters["tracks"]:
        is_pilot = name in ("T01", "T02", "T03", "T04")
        for print_kind, standard_starts in (("solid", solid_starts), ("dashed", dashed_starts)):
            sheet_id = new_track_sheet_id(name, print_kind, parameters)
            number = sheet_id.split("_")[1]
            has_near_end = near_end_print == print_kind
            if is_pilot and print_kind == "solid":
                blocks = [(index + 1, index + 1, "STD", "A", parameters["pilot_notes"][index]) for index in range(standard_starts)]
            else:
                blocks = [(1, standard_starts, "STD", "A", parameters["either_operator_note"])]
            if has_near_end:
                blocks.append((standard_starts + 1, standard_starts + 1, "NE", "A", parameters["near_end_note"]))
            count = standard_starts + int(has_near_end)
            plan_text = f"{count} episode{'s' if count > 1 else ''}, end A: {standard_starts} standard start{'s' if standard_starts > 1 else ''}"
            plan_text += ", then 1 near-end start" if has_near_end else ""
            plan_text += " (pilot)" if is_pilot and print_kind == "solid" else ""
            setup_text = (f"New track {name} ({label}), {print_kind} print, end A written on its back. Lay it by the NEW-TRACK MARK, "
                          f"not by a course sheet's place: the centre of end A's round cap on the mark ({mark_x_cm:.1f} cm in front of "
                          f"the base axis, {abs(mark_y_cm):.1f} cm to its right, table_marks.pdf), the first straight piece along the "
                          f"direction line, the sheet not turned. Tape it down once and record all its episodes before the next sheet.")
            if print_kind == "dashed":
                setup_text += " Keep the dashes' direction across the gaps."
            RECORDING_PAGE_PARAMETERS["sheets"][sheet_id] = {"number": number, "label": f"{name} {label}, {print_kind}", "plan": plan_text}
            RECORDING_PAGE_PARAMETERS["sheet_setups"][sheet_id] = setup_text
            RECORDING_PAGE_PARAMETERS["episode_plan"][sheet_id] = blocks
            RECORDING_PARAMETERS["line_length_cm"][sheet_id] = length_cm
            shapes[sheet_id] = name
    return shapes


# Which sheets carry the same line (the shape): both prints of a new track share one; each course sheet is its own
# (sheet 06 is the same kind of loop as 05, not the same line).
SHAPE_OF_SHEET = {path_id: sheet["number"] for path_id, sheet in RECORDING_PAGE_PARAMETERS["sheets"].items()}
SHAPE_OF_SHEET.update(add_new_track_sheets())


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
    # 8.11.26 (Emranul: puck references often not found): under the ceiling lights the puck in the hand's shadow has a
    # median value of only 43 to 51, as dark as the block and the gaps, but its median signed hue is -39 to -50 while
    # theirs is -53 to -60 (correction pictures of 8.11); a dark violet blob with its median hue at or above this
    # is not rejected for its darkness
    "dark_violet_puck_hue_minimum": -51,
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

# Offline check of 8.11.26 (Emranul's question): can Claude, looking at one forearm-camera picture, name the next
# action the way Emranul and Marshal do? claude_operator_study.py picks a fixed test set, hides its labels in
# "answer_key_path" until Claude has predicted, then scores. Read-only on the data. A picture is a "bend" when the
# tip's direction over the "bend_window_steps" steps before and after it changes by more than "bend_threshold_deg";
# "near goal" is within "near_goal_cm" of the stop point. "Close" means a moving action within "close_angle_deg".
CLAUDE_OPERATOR_STUDY_PARAMETERS = {
    # Read-only; a worktree passes the main checkout's folder on the command line instead.
    "data_folder": Path(__file__).resolve().parent.parent / "03_data" / "raw" / "main_data",
    "episode_pattern": r"(path_0[123])_.*_score_0[23]$",
    "minimum_frames": 20,
    "episodes_per_sheet": 6,
    "positions_per_episode": [0.0, 0.35, 0.65],
    "near_goal_cm": 3.0,
    "bend_window_steps": 3,
    "bend_threshold_deg": 15.0,
    "series_length": 3,
    "series_every_nth": 3,
    "close_angle_deg": 60,
    "seed": 8_11_26,
    "study_folder": Path(__file__).resolve().parent.parent / "06_deliverables" / "research_notes" / "Claude operator study",
    "answer_key_path": Path(__file__).resolve().parent.parent / "03_data" / "claude_operator_study_answer_key.csv"
}


# ------------------------------------------------------------------------------------------------------------------
# Model and training (9.11.26, Model and training session).

def main_checkout_folder() -> Path:
    """
    Return the project folder that holds ``03_data``: this checkout, or the main checkout for a git worktree.

    ``03_data`` is never in git, so a git worktree of the project (for example one under ``.claude/worktrees``) has no
    data of its own. Its ``.git`` file names the worktree's folder inside the main checkout's ``.git`` folder, whose
    parent is the main checkout.

    Returns
    -------
    pathlib.Path
        The folder that contains ``03_data``, or this checkout when no other one can be found.
    """

    project_folder = Path(__file__).resolve().parent.parent
    if (project_folder / "03_data").exists():
        return project_folder
    git_file = project_folder / ".git"
    if git_file.is_file():
        git_line = git_file.read_text(encoding = "utf-8").strip()
        if git_line.startswith("gitdir:"):
            worktree_git_folder = Path(git_line.split(":", 1)[1].strip())
            return worktree_git_folder.parent.parent.parent
    return project_folder


# The training data index: training_dataset.py reads the main data without changing it and writes one row per picture,
# one row per episode, a small grey copy of every picture and the resized picture caches into "processed_folder".
# Kept: the scores in "kept_scores" (Emranul, 6.11.26: every finished episode trains, score 1 included; score 0 is left
# out). Episodes without a score yet (3 on 9.11.26) are indexed but marked; "is_unscored_kept" decides whether the
# training uses them. Lighting notes were typed by hand (seven spellings by 9.11.26); a note naming the ceiling lights
# counts as ceiling even when it also names daylight (one note says it was wrongly marked as natural light). The red and
# dark measurements reuse the recording tool's own colour limits, so the numbers mean the same in both places.
TRAINING_DATA_PARAMETERS = {
    "raw_folder": main_checkout_folder() / "03_data" / "raw" / "main_data",
    "processed_folder": main_checkout_folder() / "03_data" / "processed" / "v01_training_index",
    "kept_scores": [1, 2, 3],
    "is_unscored_kept": False,
    "lighting_keywords": {"ceiling": ["ceiling", "cieling", "lamp"], "natural": ["natural", "cloudy", "daylight", "sun"]},
    "thumbnail_size": [32, 24],
    "image_caches": {"square_224": [224, 224], "wide_160": [160, 120]},
    "red_hue_edges": RECORDING_PARAMETERS["puck_red_hue_edges"],
    "red_minimum_saturation": RECORDING_PARAMETERS["puck_red_min_saturation"],
    "red_minimum_value": RECORDING_PARAMETERS["puck_red_min_value"],
    "dark_value_below": PICTURE_CHECK_PARAMETERS["dark_value_below"],
    "robot_area_px": PICTURE_CHECK_PARAMETERS["robot_area_px"],
    "worker_count": 6
}

# Frozen picture features (training_features.py): pretrained backbones from the timm library, used without changing
# their weights, so that small heads can be compared in seconds (training_probes.py). "is_fixed_input" marks a vision
# transformer whose input size is fixed by its position table. The untrained ResNet-18 is a control: it shows how much
# a head learns from features that know nothing about pictures.
TRAINING_FEATURE_PARAMETERS = {
    "backbones": {
        "resnet18_imagenet": {"timm_name": "resnet18.tv_in1k", "is_pretrained": True},
        "resnet18_random": {"timm_name": "resnet18", "is_pretrained": False},
        "efficientnet_b0_imagenet": {"timm_name": "efficientnet_b0.ra_in1k", "is_pretrained": True},
        "mobilenetv3_large_imagenet": {"timm_name": "mobilenetv3_large_100.ra_in1k", "is_pretrained": True},
        "dinov2_small": {"timm_name": "vit_small_patch14_dinov2.lvd142m", "is_pretrained": True,
                         "creation_arguments": {"img_size": 224}, "is_fixed_input": True},
        "clip_vit_b32": {"timm_name": "vit_base_patch32_clip_224.openai", "is_pretrained": True, "is_fixed_input": True}
    },
    "picture_cache": "square_224",
    "grid_cells": 4,
    "batch_size": 64,
    "extraction_threads": 4,
    "speed_threads": 4,
    "speed_input_sizes": {"square_224": [224, 224], "wide_160": [160, 120]},
    "speed_repeats": 20,
    "seed": 9_11_26,
    "is_overwritten": False
}

# Training targets and validation (training_targets.py). Rows used: the scores in "kept_scores", without the rows the
# operators marked as a mistake; flagged pictures stay in unless "is_flagged_dropped". The plan's start end replaces
# episode.json's where they disagree (19 re-planned episodes of sheets 05 and 06, checked on the tip's path, 9.11.26);
# with "is_audit_cleaning" the audit's table of corrected ends is used instead (it also fixes 11 episodes of sheet 02).
# A tip move under 3 mm (small_motion, mostly a forearm twist) becomes the next real move ("next_move"), is dropped
# ("drop") or is its own class ("own_class"). A forearm change of "twist_threshold_deg" or more counts as a twist.
TRAINING_TARGET_PARAMETERS = {
    "kept_scores": [1, 2, 3],
    "is_unscored_kept": False,
    "is_mistake_dropped": True,
    "is_flagged_dropped": False,
    "corrected_start_end": "plan",
    "small_motion_rule": "next_move",
    "twist_threshold_deg": 2.0,
    "chunk_steps": 5,
    "episode_fold_count": 5,
    "inner_validation_share": 0.15,
    "seed": 9_11_26
}

# Offline measures (training_metrics.py), the same for every model. A stop counts as right within "right_stop_cm" of
# where the demonstration stopped and as too early more than "early_stop_cm" before it (the is_at_goal label uses 2 cm).
TRAINING_METRIC_PARAMETERS = {
    "angle_thresholds_deg": [30, 45, 60],
    "stop_threshold": 0.5,
    "right_stop_cm": 2.0,
    "early_stop_cm": 3.0,
    "twist_threshold_deg": 2.0
}

# Quick comparisons on frozen features (training_probes.py). A head with one hidden layer of "hidden_size" (0: linear)
# is trained on standardised features with AdamW and early stopping on whole episodes held back from the training
# sheets; "class_weight_power" 0.5 weighs each class by the square root of its rarity. Each experiment answers one
# question; a variant lists only what differs from these defaults and from the experiment's "shared" settings.
TRAINING_PROBE_PARAMETERS = {
    "backbone": "resnet18_imagenet",
    "feature_kind": "grid",
    "design": "classes",
    "split": "sheets",
    "history_lags": [],
    "is_difference_history": False,
    "hidden_size": 256,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "batch_size": 256,
    "max_epochs": 40,
    "patience": 5,
    "label_smoothing": 0.0,
    "class_weight_power": 0.5,
    "score_weights": {1: 1.0, 2: 1.0, 3: 1.0},
    "twist_loss_weight": 0.5,
    "angle_scale_deg": 5.0,
    "huber_delta": 1.0,
    "stop_positive_weight": 10.0,
    "seed": 9_11_26,
    "threads": 4,
    "output_folder": Path(__file__).resolve().parent.parent / "04_runs" / "2026-11-09_frozen_feature_probes"
}

PROBE_BACKBONES = ["resnet18_imagenet", "resnet18_random", "efficientnet_b0_imagenet", "mobilenetv3_large_imagenet",
                   "dinov2_small", "clip_vit_b32"]

TRAINING_PROBE_EXPERIMENTS = {
    "backbones": {
        "question": "Which frozen backbone, with a summary or a 4 x 4 grid of features, transfers best to a held-out sheet?",
        "shared": {"design": "classes", "split": "sheets"},
        "variants": [{"name": f"{backbone}_{kind}", "backbone": backbone, "feature_kind": kind}
                     for backbone in PROBE_BACKBONES for kind in ["global", "grid"]]
    },
    "designs": {
        "question": "What should the model predict: move classes, the exact step, both, or a chunk of several steps?",
        "shared": {"split": "sheets"},
        "variants": [
            {"name": "classes_small_motion_dropped", "design": "classes", "small_motion_rule": "drop"},
            {"name": "classes_small_motion_next_move", "design": "classes", "small_motion_rule": "next_move"},
            {"name": "classes_small_motion_own_class", "design": "classes", "small_motion_rule": "own_class"},
            {"name": "exact_step", "design": "exact"},
            {"name": "mixed_classes_and_exact", "design": "mixed", "small_motion_rule": "next_move"},
            {"name": "chunk_of_5_steps", "design": "chunk", "chunk_steps": 5}
        ]
    },
    "splits": {
        "question": "How much worse is a held-out sheet than held-out episodes of known sheets (the generalisation gap)?",
        "shared": {},
        "variants": [
            {"name": "classes_held_out_episodes", "design": "classes", "split": "episodes"},
            {"name": "exact_held_out_episodes", "design": "exact", "split": "episodes"},
            {"name": "classes_held_out_sheets", "design": "classes", "split": "sheets"},
            {"name": "exact_held_out_sheets", "design": "exact", "split": "sheets"}
        ]
    },
    "history": {
        "question": "Does a short history of earlier pictures help a one-picture model?",
        "shared": {"design": "classes", "split": "sheets"},
        "variants": [
            {"name": "one_picture", "history_lags": []},
            {"name": "plus_previous_picture", "history_lags": [1]},
            {"name": "plus_change_since_previous", "history_lags": [1], "is_difference_history": True},
            {"name": "plus_pictures_2_and_4_back", "history_lags": [2, 4]}
        ]
    },
    "scores": {
        "question": "Should score-1 and score-2 episodes train, be weighted down, or be left out?",
        "shared": {"design": "classes", "split": "sheets"},
        "variants": [
            {"name": "scores_1_2_3", "kept_scores": [1, 2, 3]},
            {"name": "scores_2_3", "kept_scores": [2, 3]},
            {"name": "score_3_only", "kept_scores": [3]},
            {"name": "scores_weighted", "kept_scores": [1, 2, 3], "score_weights": {1: 0.4, 2: 0.7, 3: 1.0}}
        ]
    },
    "balance": {
        "question": "How strongly should rare classes be weighted, and does label smoothing help?",
        "shared": {"design": "classes", "split": "sheets"},
        "variants": [
            {"name": "no_class_weights", "class_weight_power": 0.0},
            {"name": "square_root_weights", "class_weight_power": 0.5},
            {"name": "full_inverse_weights", "class_weight_power": 1.0},
            {"name": "square_root_weights_label_smoothing", "class_weight_power": 0.5, "label_smoothing": 0.1}
        ]
    },
    "head_size": {
        "question": "Is a linear head enough on frozen features, or does a hidden layer help?",
        "shared": {"design": "classes", "split": "sheets"},
        "variants": [
            {"name": "linear", "hidden_size": 0},
            {"name": "hidden_256", "hidden_size": 256},
            {"name": "hidden_1024", "hidden_size": 1024, "dropout": 0.3}
        ]
    },
    "cleaning": {
        "question": "Do flagged pictures and the rows marked as mistakes hurt?",
        "shared": {"design": "classes", "split": "sheets"},
        "variants": [
            {"name": "mistakes_dropped_flags_kept", "is_mistake_dropped": True, "is_flagged_dropped": False},
            {"name": "mistakes_and_flags_dropped", "is_mistake_dropped": True, "is_flagged_dropped": True},
            {"name": "nothing_dropped", "is_mistake_dropped": False, "is_flagged_dropped": False}
        ]
    }
}

# End-to-end training (train_policy.py). Defaults for every named run in TRAINING_RUNS; a run lists only what differs.
# Pictures come from the 160 x 120 cache (the camera's 4:3 shape kept). A pretrained backbone learns at
# "backbone_learning_rate_factor" times the head's rate, so its ImageNet knowledge is changed gently. "pooling" grid
# keeps a 3 x 4 grid of cells, so the head knows where the line runs relative to the puck. Augmentation strengths:
# brightness, contrast and saturation as plus or minus shares, gamma as plus or minus its natural logarithm, blur
# sigma in pixels, noise as a share of full scale, shift in pixels (the taped camera crept up to about 8 px between
# episodes at 640 px, about 2 px at 160 px), and the share of mirrored pictures (0: off). "device" "auto" trains on the
# graphics card when PyTorch sees one (the OTH GPU pool, 10.11.26) and on the processor otherwise (the laptop); the
# augmentation always runs on the processor, so "threads" still matters on the pool.
TRAINING_RUN_PARAMETERS = {
    "backbone": "resnet18.tv_in1k",
    "is_pretrained": True,
    "small_cnn_channels": [32, 64, 96, 128, 192],
    "pooling": "grid",
    "pooling_grid": [3, 4],
    "hidden_size": 256,
    "dropout": 0.2,
    "design": "classes",
    "split": "sheets",
    "picture_cache": "wide_160",
    "picture_size": [160, 120],
    "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "augmentation": {"brightness": 0.35, "contrast": 0.3, "saturation": 0.25, "gamma_log": 0.25, "blur_probability": 0.3,
                     "blur_sigma_most": 1.5, "noise_most": 0.03, "shift_px": 3, "mirror_probability": 0.0},
    "learning_rate": 1e-3,
    "backbone_learning_rate_factor": 0.3,
    "weight_decay": 0.05,
    "batch_size": 64,
    "epochs": 12,
    "warm_up_epochs": 1,
    "patience": 4,
    "evaluation_batch_size": 256,
    "class_weight_power": 0.5,
    "score_weights": {1: 1.0, 2: 1.0, 3: 1.0},
    "label_smoothing": 0.0,
    "twist_loss_weight": 0.5,
    "angle_scale_deg": 5.0,
    "huber_delta": 1.0,
    "stop_positive_weight": 10.0,
    "seed": 9_11_26,
    "threads": 6,
    "device": "auto",
    "date": None,
    "output_root": Path(__file__).resolve().parent.parent / "04_runs"
}

NO_AUGMENTATION = {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "gamma_log": 0.0, "blur_probability": 0.0,
                   "blur_sigma_most": 1.5, "noise_most": 0.0, "shift_px": 0, "mirror_probability": 0.0}

TRAINING_RUNS = {
    "small_cnn_scratch_classes": {"backbone": "small_cnn", "is_pretrained": False, "learning_rate": 2e-3, "epochs": 25, "patience": 6},
    "resnet18_finetune_classes": {},
    "resnet18_finetune_exact": {"design": "exact"},
    "resnet18_finetune_mixed": {"design": "mixed"},
    "resnet18_finetune_mirror": {"augmentation": dict(TRAINING_RUN_PARAMETERS["augmentation"], mirror_probability = 0.5)},
    "resnet18_finetune_no_augmentation": {"augmentation": NO_AUGMENTATION},
    "mobilenetv3_finetune_classes": {"backbone": "mobilenetv3_large_100.ra_in1k"},
    "efficientnet_b0_finetune_classes": {"backbone": "efficientnet_b0.ra_in1k"}
}

# 9.11.26: two more settings for the probes and the end-to-end runs. "is_stop_separate" gives a class model its own stop
# output instead of a stop class; "stop_target" trains a stop output on the episode's stop picture only ("stop_row")
# or on every picture within "at_goal_within_cm" of the goal ("at_goal", more examples, may stop a little early).
# The lean and hand changes are scored like the twist, from these thresholds.
TRAINING_PROBE_PARAMETERS.update({"is_stop_separate": False, "stop_target": "stop_row", "is_prediction_saved": False})
TRAINING_RUN_PARAMETERS.update({"is_stop_separate": False, "stop_target": "stop_row"})
TRAINING_METRIC_PARAMETERS.update({"lean_threshold_deg": 1.0, "hand_threshold_deg": 3.0})
TRAINING_PROBE_EXPERIMENTS["stop"] = {
    "question": "How should the model decide to stop: as one more class, or with its own output trained on the stop picture or on every picture near the goal?",
    "shared": {"split": "sheets"},
    "variants": [
        {"name": "classes_stop_as_class", "design": "classes"},
        {"name": "classes_stop_head_on_stop_picture", "design": "classes", "is_stop_separate": True, "stop_target": "stop_row"},
        {"name": "classes_stop_head_on_near_goal", "design": "classes", "is_stop_separate": True, "stop_target": "at_goal"},
        {"name": "exact_stop_head_on_stop_picture", "design": "exact", "stop_target": "stop_row"},
        {"name": "exact_stop_head_on_near_goal", "design": "exact", "stop_target": "at_goal"}
    ]
}

# One comparison table of every model (compare_runs.py): the measures shown, with their headings; sorted best first by
# the share of moving pictures whose predicted direction is within 45 degrees of the demonstrated one.
TRAINING_COMPARISON_PARAMETERS = {
    "runs_folder": Path(__file__).resolve().parent.parent / "04_runs",
    "columns": [
        ("within_45_deg_mean", "within 45 deg"),
        ("within_45_deg_minimum", "within 45 deg, worst fold"),
        ("within_30_deg_mean", "within 30 deg"),
        ("angle_error_median_deg_mean", "median angle error"),
        ("class_balanced_accuracy_mean", "balanced class accuracy"),
        ("episodes_stopped_right_mean", "stopped right"),
        ("episodes_stopped_early_mean", "stopped early"),
        ("episodes_never_stopped_mean", "never stopped"),
        ("twist_sign_accuracy_mean", "twist sign"),
        ("lean_sign_accuracy_mean", "lean sign"),
        ("milliseconds_per_picture", "ms per picture")
    ],
    "sort_by": "within 45 deg"
}

# 9.11.26: how the held-out-sheet result grows with more data, to decide whether more episodes or more different
# tracks are worth recording ("train_episode_share" of each training sheet's episodes; "train_sheet_count" of the five
# training sheets, chosen at random per fold). Each variant is repeated with three seeds, because one draw of sheets
# can be lucky.
TRAINING_PROBE_PARAMETERS.update({"train_episode_share": 1.0, "train_sheet_count": None})
TRAINING_PROBE_EXPERIMENTS["data_amount"] = {
    "question": "Does the held-out-sheet result still grow with more episodes per sheet, or with more different sheets?",
    "shared": {"design": "classes", "split": "sheets"},
    "variants": ([{"name": f"episode_share_{share}_seed_{seed}", "train_episode_share": share, "seed": seed}
                  for share in [0.25, 0.5, 1.0] for seed in [1, 2, 3]]
                 + [{"name": f"sheet_count_{count}_seed_{seed}", "train_sheet_count": count, "seed": seed}
                    for count in [1, 2, 3, 4] for seed in [1, 2, 3]])
}

# 9.11.26: the head standardises its own input from this many evenly spread training rows (enough for a stable mean and
# spread, and it keeps the features in half precision in memory).
TRAINING_PROBE_PARAMETERS.update({"standardisation_rows": 4000})

# 9.11.26: one end-to-end epoch over all 15,000 training pictures takes about 7 minutes on the laptop's processor
# (measured with other jobs running). Neighbouring pictures are about 1 cm apart and look alike, so each epoch uses a
# random "train_share_per_epoch" of them; 0.5 halves the time per epoch.
TRAINING_RUN_PARAMETERS.update({"train_share_per_epoch": 1.0})
TRAINING_RUNS["small_cnn_scratch_classes"].update({"train_share_per_epoch": 0.5, "epochs": 16, "patience": 5})
for run_name in ("resnet18_finetune_classes", "resnet18_finetune_exact", "resnet18_finetune_mixed", "resnet18_finetune_mirror",
                 "resnet18_finetune_no_augmentation", "mobilenetv3_finetune_classes", "efficientnet_b0_finetune_classes"):
    TRAINING_RUNS[run_name].update({"train_share_per_epoch": 0.5, "epochs": 8, "patience": 3})

# 9.11.26: the path direction, a less noisy target than one hand-guided step (the steps wobble by about 21 degrees
# from one to the next): from the tip now to where it is "path_steps" steps later, scored where that is at least
# "path_minimum_cm" long. The probes save their predictions (npz files under 04_runs, kept out of git) for error
# analysis per sheet and situation.
TRAINING_METRIC_PARAMETERS.update({"path_steps": 3, "path_minimum_cm": 1.5})
TRAINING_PROBE_PARAMETERS.update({"is_prediction_saved": True})
TRAINING_COMPARISON_PARAMETERS["columns"][3:3] = [("path_within_45_deg_mean", "path within 45 deg"),
                                                  ("path_within_45_deg_minimum", "path within 45 deg, worst fold"),
                                                  ("path_angle_error_median_deg_mean", "path median angle error")]

# 9.11.26: "input_channels" "rgb" feeds the colour picture; "line_and_puck" feeds three maps instead (how much darker than
# the picture's median a pixel is, how red it is, and the grey picture less its median), so light and table texture
# matter less (train_policy.line_and_puck_maps). Tested with the small network from scratch.
TRAINING_RUN_PARAMETERS.update({"input_channels": "rgb"})
TRAINING_RUNS["small_cnn_line_and_puck_classes"] = dict(TRAINING_RUNS["small_cnn_scratch_classes"], input_channels = "line_and_puck")
TRAINING_RUNS["resnet18_finetune_chunk"] = {"design": "chunk", "train_share_per_epoch": 0.5, "epochs": 8, "patience": 3}

# 9.11.26: rows where the robot stood still (tip moved under "static_move_m", forearm under "static_turn_deg", not the
# stop row: the first row of every episode and the first row after a hand correction, about 430) show nearly the same
# picture as the next row, which carries the real move. "is_static_dropped" leaves them out; tested in "cleaning".
TRAINING_TARGET_PARAMETERS.update({"is_static_dropped": False, "static_move_m": 0.001, "static_turn_deg": 1.0})
TRAINING_PROBE_EXPERIMENTS["cleaning"]["variants"].append(
    {"name": "mistakes_and_static_rows_dropped", "is_mistake_dropped": True, "is_static_dropped": True})

# 9.11.26, from the picture audit: the taped camera moved once by about 14 px right and 5 px down at 640 px (about 4 px
# at 160 px), so the shift is 4 px; sharpness hardly varies (at most about 0.5 px of blur at full size), so blur is
# rare and light; a per-channel gain of plus or minus "channel_gain" imitates the bluish daylight cast.
for augmentation in [TRAINING_RUN_PARAMETERS["augmentation"], NO_AUGMENTATION] + [run["augmentation"] for run in TRAINING_RUNS.values() if "augmentation" in run]:
    augmentation.setdefault("channel_gain", 0.0)
TRAINING_RUN_PARAMETERS["augmentation"].update({"shift_px": 4, "blur_probability": 0.15, "blur_sigma_most": 0.8, "channel_gain": 0.08})
TRAINING_RUNS["resnet18_finetune_mirror"]["augmentation"] = dict(TRAINING_RUN_PARAMETERS["augmentation"], mirror_probability = 0.5)

# Where a model goes wrong (training_error_analysis.py). "start" is the first "start_rows" rows; "near_goal" within
# "near_goal_cm" of where the demonstration stopped; a "bend" turns the tip's direction by more than
# "bend_threshold_deg" over "bend_window_steps" steps before and after; the forearm is "turned" from
# "forearm_turned_deg" and "far_turned" from "forearm_far_turned_deg". The taped camera moved once (picture audit,
# 9.11.26): view A before "camera_moved_from", view B from "camera_settled_at", "settling" between.
TRAINING_ERROR_ANALYSIS_PARAMETERS = {
    "start_rows": 3,
    "near_goal_cm": 3.0,
    "bend_window_steps": 3,
    "bend_threshold_deg": 15.0,
    "forearm_turned_deg": 30.0,
    "forearm_far_turned_deg": 75.0,
    "camera_moved_from": "2026-11-08T17:59:00",
    "camera_settled_at": "2026-11-08T21:00:00"
}

# 9.11.26: the error analysis also scores the predicted direction averaged over the last 1, 2, 3 or 5 pictures.
TRAINING_ERROR_ANALYSIS_PARAMETERS.update({"smoothing_windows": [1, 2, 3, 5]})

# 9.11.26: the stop threshold is also tuned per fold on the held-back validation episodes (never on the test sheet), from
# these candidates, for one picture and for two pictures in a row; the result is scored as "tuned_" and "tuned_two_".
TRAINING_METRIC_PARAMETERS.update({"stop_threshold_candidates": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
TRAINING_COMPARISON_PARAMETERS["columns"][-1:-1] = [("tuned_two_episodes_stopped_right_mean", "stopped right, tuned, 2 in a row"),
                                                    ("tuned_two_episodes_stopped_early_mean", "stopped early, tuned, 2 in a row")]

# 9.11.26: how much the held-out-sheet result moves between seeds alone, so that a difference between two variants
# can be judged against it (the same variant three times with different seeds).
TRAINING_PROBE_EXPERIMENTS["seeds"] = {
    "question": "How much does the result change between random seeds alone?",
    "shared": {"split": "sheets"},
    "variants": ([{"name": f"classes_seed_{seed}", "design": "classes", "seed": seed} for seed in [11, 12, 13]]
                 + [{"name": f"exact_seed_{seed}", "design": "exact", "seed": seed} for seed in [11, 12, 13]])
}

# 9.11.26: the runs of this comparison keep the analysis date in their folder name, so that folds trained after
# midnight land in the same folder as the others (04_runs/2026-11-09_<run>/).
TRAINING_RUN_PARAMETERS.update({"date": "2026-11-09"})

# 9.11.26: the end-to-end runs were trained on three held-out sheets (the two curves and both loops), so the comparison
# also shows every model's means over exactly these folds.
TRAINING_COMPARISON_PARAMETERS.update({"matched_folds": ["without_path_03", "without_path_05", "without_path_06"]})

# 9.11.26: a third way to learn the stop. "stop_target" "goal_distance" trains the stop output to predict the tip's
# distance to where the demonstration stopped (capped at "goal_distance_cap_cm"), from every picture; the robot stops
# when the predicted distance is small for two pictures in a row (the threshold tuned on validation episodes).
TRAINING_PROBE_PARAMETERS.update({"goal_distance_cap_cm": 10.0})
TRAINING_RUN_PARAMETERS.update({"goal_distance_cap_cm": 10.0})
TRAINING_PROBE_EXPERIMENTS["stop_distance"] = {
    "question": "Does a stop output that predicts the distance to the goal from every picture stop more reliably?",
    "shared": {"split": "sheets"},
    "variants": [
        {"name": "classes_stop_head_on_goal_distance", "design": "classes", "is_stop_separate": True, "stop_target": "goal_distance"},
        {"name": "exact_stop_on_goal_distance", "design": "exact", "stop_target": "goal_distance"}
    ]
}
TRAINING_RUNS["resnet18_finetune_exact_distance_stop"] = {"design": "exact", "stop_target": "goal_distance",
                                                          "train_share_per_epoch": 0.5, "epochs": 8, "patience": 3}

# 9.11.26: the cleaning the data audit recommends (06_deliverables/research_notes/Model and training/evidence/, section 11
# of audit_synthesis.md), applied when "is_audit_cleaning" is on: scores of unscored folders from review.json (2
# episodes, score 2); start ends from the audit's table of corrected ends (30 episodes changed: 19 on sheets 05 and 06,
# 11 on sheet 02); 469 rows dropped (row 0 of every episode and the first row after every hand correction, both null
# moves at waypoint 0; one more sudden-change picture; the half run 05-16; rows 169 to 175 of 04-01 try 6, where the
# puck was pulled back by pressing its top); 50 rows relabelled as stop (their joints differ from the next row, the
# stop row, by under 0.2 degrees, so the picture is the stop picture); the 25 steps over 2 cm and the few twists over
# 15 degrees left by recording stalls clipped.
TRAINING_TARGET_PARAMETERS.update({
    "is_audit_cleaning": False,
    "audit_cleaning": {
        "corrected_ends_path": Path(__file__).resolve().parent.parent / "06_deliverables" / "research_notes" / "Model and training" / "evidence" / "corrected_ends.csv",
        "static_joint_limit_deg": 0.2,
        "dropped_episode_patterns": ["_plan_05-16_"],
        "dropped_row_ranges": [("_plan_04-01_try_6", 169, 175)],
        "step_clip_m": 0.02,
        "twist_clip_deg": 15.0
    }
})
# The camera period from the audit: the camera view moved during the break on 8.11.26 between 18:07 and 19:33; from
# 04-30 (19:33) on, 71 episodes are in the later view.
TRAINING_ERROR_ANALYSIS_PARAMETERS.update({"camera_moved_from": "2026-11-08T18:07:00", "camera_settled_at": "2026-11-08T19:33:00"})

# 9.11.26: the final candidate from the comparison: the exact step (tip forward and left, forearm twist, lean, hand)
# with its own stop output trained on the stop pictures (259 after the audit's relabelling), the audit's cleaning, a
# fixed 8 epochs on half the pictures each, and the LAST weights kept (no choice by validation loss, which predicts
# robot success poorly); scored on all six held-out sheets.
TRAINING_RUN_PARAMETERS.update({"is_last_checkpoint_kept": False})
TRAINING_RUNS["resnet18_final_candidate"] = {"design": "exact", "stop_target": "stop_row", "is_audit_cleaning": True,
                                             "train_share_per_epoch": 0.5, "epochs": 8, "patience": 99,
                                             "is_last_checkpoint_kept": True}

# 9.11.26, from the design review: the backup set-up named by two of the three judges (the 8 move names with square-root
# class weights, whose probability-weighted mean direction is executed, together with the exact step, whose twist is
# executed, and its own stop output), on the final candidate's recipe; and a second seed of the final candidate, to
# measure how much end-to-end results move between seeds alone.
TRAINING_RUNS["resnet18_backup_names_plus_exact"] = {"design": "mixed", "is_stop_separate": True, "stop_target": "stop_row",
                                                     "is_audit_cleaning": True, "class_weight_power": 0.5,
                                                     "train_share_per_epoch": 0.5, "epochs": 8, "patience": 99,
                                                     "is_last_checkpoint_kept": True}
TRAINING_RUNS["resnet18_final_candidate_seed_2"] = dict(TRAINING_RUNS["resnet18_final_candidate"], seed = 10_11_26)

# 10.11.26, on the GPU pool: the final candidate with only the augmentation switched off, with both seeds, so that the
# effect of augmentation is measured alone ("resnet18_finetune_no_augmentation" differs from the final candidate in its
# design, cleaning and epochs as well, so it cannot answer this).
TRAINING_RUNS["resnet18_final_candidate_no_augmentation"] = dict(TRAINING_RUNS["resnet18_final_candidate"], augmentation = NO_AUGMENTATION)
TRAINING_RUNS["resnet18_final_candidate_no_augmentation_seed_2"] = dict(TRAINING_RUNS["resnet18_final_candidate_seed_2"], augmentation = NO_AUGMENTATION)

# 10.11.26 (Emranul): model 1 for the first robot tests, the final candidate's recipe (augmentation kept, as the fair
# test above decided) trained on every sheet and episode with no check pictures ("split" "all"), with both seeds so
# that a spare exists. It has no offline score of its own; the robot is its test.
TRAINING_RUNS["resnet18_model_1_all_sheets"] = dict(TRAINING_RUNS["resnet18_final_candidate"], split = "all", date = "2026-11-10")
TRAINING_RUNS["resnet18_model_1_all_sheets_seed_2"] = dict(TRAINING_RUNS["resnet18_final_candidate_seed_2"], split = "all", date = "2026-11-10")

# 10.11.26 night queue on ki-014 (Emranul: "if you want to train or test more, do it"): three more seeds of the final
# candidate, so that five seeds can be scored as one averaging team (an ensemble) on the six held-out sheets, and the
# same three on all sheets; the chunk target (the report's backup candidate if the names-plus-exact backup is dropped)
# on the final candidate's recipe, two seeds, and on all sheets; and the final candidate with 16 rounds instead of 8.
# Each six-sheet run is scored before its all-sheets copy is used on the robot.
for seed_number, seed in ((3, 11_11_26), (4, 12_11_26), (5, 13_11_26)):
    TRAINING_RUNS[f"resnet18_final_candidate_seed_{seed_number}"] = dict(TRAINING_RUNS["resnet18_final_candidate"], seed = seed, date = "2026-11-10")
    TRAINING_RUNS[f"resnet18_model_1_all_sheets_seed_{seed_number}"] = dict(TRAINING_RUNS["resnet18_final_candidate"], seed = seed, split = "all", date = "2026-11-10")
TRAINING_RUNS["resnet18_final_candidate_chunk"] = dict(TRAINING_RUNS["resnet18_final_candidate"], design = "chunk", date = "2026-11-10")
TRAINING_RUNS["resnet18_final_candidate_chunk_seed_2"] = dict(TRAINING_RUNS["resnet18_final_candidate_chunk"], seed = 10_11_26)
TRAINING_RUNS["resnet18_model_2_chunk_all_sheets"] = dict(TRAINING_RUNS["resnet18_final_candidate_chunk"], split = "all")
TRAINING_RUNS["resnet18_final_candidate_16_rounds"] = dict(TRAINING_RUNS["resnet18_final_candidate"], epochs = 16, date = "2026-11-10")
TRAINING_RUNS["resnet18_model_3_16_rounds_all_sheets"] = dict(TRAINING_RUNS["resnet18_final_candidate_16_rounds"], split = "all")

# 11.11.26, round 1 of the search before the first robot test (Emranul: reach the ceiling of what the data allows). The
# night showed the chunk target steering best (path within 45 0.817 and 0.802 against 0.772 to 0.793 for five seeds of
# the final candidate) but stopping worse, and every model turning the forearm about half as much as the demonstrations.
# Each idea changes one thing of the chunk recipe, with two seeds: the stop's weight, the twist's weight (a smaller
# angle scale makes a twist error count more), a longer chunk, and two larger backbones. Decision rule, fixed before the
# results: a change is kept only when both seeds beat the chunk's mean path within 45 (0.810) by 1.5 points or more and
# stop right no less often, or stop right at least 8 episodes more often (of 209) without losing a point of path.
CHUNK_IDEAS = {
    "chunk_stop_weight_25": {"stop_positive_weight": 25.0},
    "chunk_twist_focus": {"angle_scale_deg": 2.5},
    "chunk_8_steps": {"chunk_steps": 8},
    "chunk_resnet34": {"backbone": "resnet34.tv_in1k"},
    "chunk_efficientnet_b0": {"backbone": "efficientnet_b0.ra_in1k"}
}
for idea_name, idea in CHUNK_IDEAS.items():
    TRAINING_RUNS[f"resnet18_{idea_name}"] = dict(TRAINING_RUNS["resnet18_final_candidate_chunk"], **idea)
    TRAINING_RUNS[f"resnet18_{idea_name}_seed_2"] = dict(TRAINING_RUNS["resnet18_final_candidate_chunk_seed_2"], **idea)

# 11.11.26, round 1 by the rule above: two ideas passed on the stop branch, both seeds stopping right more often without
# losing a point of path within 45 (chunk 87 and 87 right stops, path 0.817 and 0.802): the stop weighted 25 (103 and 105,
# never stopped 71 and 51; path 0.808 and 0.800) and ResNet-34 (95 and 101; path 0.813 and 0.807). The twist focus, the
# 8-step chunk and EfficientNet-B0 did not pass. Both passing ideas are trained on all sheets for the robot (models 4 and
# 5); their combination is scored by the same rule before its all-sheets copy is used.
TRAINING_RUNS["resnet18_model_4_chunk_stop_25_all_sheets"] = dict(TRAINING_RUNS["resnet18_chunk_stop_weight_25"], split = "all")
TRAINING_RUNS["resnet18_model_5_chunk_resnet34_all_sheets"] = dict(TRAINING_RUNS["resnet18_chunk_resnet34"], split = "all")
TRAINING_RUNS["resnet18_chunk_resnet34_stop_25"] = dict(TRAINING_RUNS["resnet18_chunk_resnet34"], stop_positive_weight = 25.0)
TRAINING_RUNS["resnet18_chunk_resnet34_stop_25_seed_2"] = dict(TRAINING_RUNS["resnet18_chunk_resnet34_seed_2"], stop_positive_weight = 25.0)
TRAINING_RUNS["resnet18_model_6_chunk_resnet34_stop_25_all_sheets"] = dict(TRAINING_RUNS["resnet18_chunk_resnet34_stop_25"], split = "all")

# 14.11.26 (Emranul: one more model from the round 1 data before Marshal's model is tested): model E's recipe (chunk,
# stop weighted 25) shown the camera's previous picture as well ("picture_count" 2, six colour channels), so that it can
# see which way the puck came from in a loop or a bend, which one picture cannot show. Decision rule, fixed before the
# results: kept only when BOTH seeds beat model E's recipe (path within 45 0.808 and 0.800, mean 0.804) by 1.5 points
# or more, stop right no more than 8 episodes less often (103 and 105 of 209), AND lose no more than 1 point of path
# within 45 on the turning pictures (where the demonstrated direction changes by 20 degrees or more from the step
# before). The last part guards against a known trap of models with a history (de Haan and others 2019, "causal
# confusion"): they score well offline by repeating the last move and then fail on the robot exactly where the move
# must change.
TRAINING_RUN_PARAMETERS.update({"picture_count": 1})
TRAINING_RUNS["resnet18_two_pictures"] = dict(TRAINING_RUNS["resnet18_chunk_stop_weight_25"], picture_count = 2, date = "2026-11-14")
TRAINING_RUNS["resnet18_two_pictures_seed_2"] = dict(TRAINING_RUNS["resnet18_chunk_stop_weight_25_seed_2"], picture_count = 2, date = "2026-11-14")
TRAINING_RUNS["resnet18_model_7_two_pictures_all_sheets"] = dict(TRAINING_RUNS["resnet18_two_pictures"], split = "all")

# 14.11.26 (Emranul: "can we pause halfway and train a model to see the difference"): model E's recipe, unchanged, on
# round 1 plus the round 2 episodes recorded so far, from its own index (training_dataset.py into "v02_training_index",
# so round 1's index stays as it is). Only the data differs from model E, so the robot shows what the new data adds:
# both models on development tracks D01 to D10, which nobody records on, and on sheet 04.
TRAINING_RUN_PARAMETERS.update({"processed_folder": None})
ROUND_2_INDEX_FOLDER = main_checkout_folder() / "03_data" / "processed" / "v02_training_index"
TRAINING_RUNS["resnet18_model_8_round_2_part_all_sheets"] = dict(TRAINING_RUNS["resnet18_model_4_chunk_stop_25_all_sheets"],
                                                                 processed_folder = ROUND_2_INDEX_FOLDER, date = None)


# The model mode (model_mode.py, 10.11.26): the robot pushes by itself with a trained model, following report section
# 4.6. Each picture gives the exact step; the magnet tip moves a fixed "step_m" in its direction (in the picture's frame,
# turned onto the table with the forearm camera's heading), the forearm twist follows the model by at most
# "twist_step_most_deg" per step and stays within "joint_4_most_deg" (the demonstrations reached 119.35 against the arm
# model's 119.75). The wrist (joint 5) and the hand (joint 6) are held at the median of the demonstrations' first rows
# (13.5 and -6.1 degrees, frames.csv on 10.11.26) and joints 1 to 3 put the tip on its target. A direction shorter than
# "smallest_direction_cm" is too short to trust, so the last step's direction is kept. The run stops when the stop
# chance reaches "stop_threshold" on "stop_pictures_in_a_row" pictures in a row (the rule fixed in 7.2), after
# "step_budget" steps (the longest demonstration had 226), or at the first refused step. Guards [judgement, to be
# checked in phase 0a of report 7.4]: the rod may lean at most "rod_lean_most_deg" (the demonstrations: 99th percentile
# 49.1, 99.9th 55.8, largest 63.0; with the wrist held, 5 of 400 recorded steps needed 50.6 to 57.6), the tip stays
# "tip_reach_m" from the base (demonstrated 13.0 to 48.5 cm), and the tip and the puck's centre ahead of it stay inside
# "workspace_x_m" and "workspace_y_m", the demonstrated tips of 6 to 9.11.26 (x 12.5 to 48.1, y -28.7 to 23.7 cm) with
# about 1 cm to spare, widened by "puck_margin_m" for the puck. WORKSPACE_PARAMETERS' box is the old table's and refused
# 4 % of recorded steps. Runs are logged under "run_folder", never in 03_data/raw.
MODEL_MODE_PARAMETERS = {
    "run_name": "resnet18_model_1_all_sheets",
    "checkpoint_path": Path(__file__).resolve().parent.parent / "04_runs" / "gpu_ki014" / "2026-11-10_resnet18_model_1_all_sheets" / "checkpoints" / "all_sheets.pt",
    "metrics_path": Path(__file__).resolve().parent.parent / "04_runs" / "gpu_ki014" / "2026-11-10_resnet18_model_1_all_sheets" / "metrics.json",
    "step_m": 0.01,
    # 11.11.26, first robot test: with the fixed step the robot pushed on where the model asked for a short step (the
    # demonstrations twist the forearm in place at sharp bends): the median asked length was 0.31 cm on sheet 04 and 0.49
    # to 0.55 cm on sheet 03, where the runs failed, against 0.8 to 1.1 cm on sheets 01, 02 and 05, where they worked.
    # With "step_length_rule" "predicted" the tip moves the predicted length times "step_length_gain", at most
    # "step_most_m" (the backup of report 4.4, without its 0.5 cm floor, so that a twist in place stays in place). A run
    # whose tips all stay within "stall_extent_m" (the diagonal of the box round them) while the forearm turns by less
    # than "stall_twist_deg" over "stall_steps" pictures is ended as stalled [judgement; report 7.4 counts a stall or a
    # back-and-forth as a failed run].
    "step_length_rule": "fixed",
    "step_length_gain": 1.0,
    "step_most_m": 0.01,
    "stall_steps": 30,
    "stall_extent_m": 0.03,
    "stall_twist_deg": 15.0,
    "twist_step_most_deg": 10.0,
    # the models turn the forearm about half as much as the demonstrations (11.11.26: on unseen sheets the mean predicted
    # twist was 1.7 to 3.6 degrees against 4.1 to 5.8 demonstrated, right sign in 68 to 84 %), so the robot test can
    # try the predicted twist times "twist_gain" (1.0 = as predicted) before the clamps
    "twist_gain": 1.0,
    # 11.11.26 (Emranul): limits only about 2 to 3 degrees inside the hardware's (joint 4: 119.75 in the arm model);
    # the demonstrations reached 119.35, so 115 cut off part of the loops' forearm swing
    "joint_4_most_deg": 117.0,
    "wrist_held_deg": 13.5,
    "hand_held_deg": -6.1,
    "smallest_direction_cm": 0.05,
    "stop_threshold": 0.5,
    "stop_pictures_in_a_row": 2,
    "step_budget": 250,
    "rod_lean_most_deg": 58.0,
    "tip_reach_m": [0.12, 0.495],
    "workspace_x_m": [0.115, 0.49],
    "workspace_y_m": [-0.30, 0.25],
    "puck_margin_m": 0.03,
    "lift_at_end_m": 0.05,
    "settle_lift_m": 0.03,
    "pictures_after_move": RECORDING_PARAMETERS["steps_until_fresh_picture"],
    "fresh_picture_timeout_s": RECORDING_PARAMETERS["fresh_picture_timeout_s"],
    "run_folder": main_checkout_folder() / "03_data" / "model_runs",
    # the checks without the robot: the fold model and saved predictions of the laptop's first set-up for the replay,
    # and how many recorded rows the executor check and the replay use
    "replay_run_folder": Path(__file__).resolve().parent.parent / "04_runs" / "2026-11-09_resnet18_final_candidate",
    "replay_run_name": "resnet18_final_candidate",
    "replay_fold": "without_path_01",
    "replay_rows": 200,
    "executor_check_rows": 400,
    "executor_check_smallest_step_m": 0.005,
    "executor_check_seed": 10_11_26,
    "simulated_step_budget": 30,
    # the solver's first guess for joints 2 and 3 (radians), and how far the hand-placed tip may be from pushing height
    # before a robot run refuses to start (the demonstrated tip heights spread from 0.9 to 2.3 cm, 5th to 95th percentile)
    "solver_start_joints_2_and_3_rad": [-0.8, -0.6],
    "start_height_tolerance_m": 0.02,
    # 12.11.26, additions for the first test visit (plan after the first robot test, step 1). Each one is off, or has
    # the old value, by default: a robot run started without a new flag does what the driver as committed on 12.11.26
    # (b0a12df) does, step for step. The ten robot runs of 11.11.26 were driven before the stall rule and the step
    # length rule existed (both came that evening and have run on the simulated robot only), so "stalled" is an ending
    # the robot has not produced yet. None of the additions adds a move of the arm; --step-cm only changes how long
    # the fixed step is.
    # --step-cm sets the fixed step ("step_m") for one run, within "step_cm_allowed" [judgement: no longer than the
    # 1 cm driven on 11.11.26, so that a typing mistake such as 6.5 for 0.65 cannot ask for a long step; and at least
    # about three times "stall_extent_m" / ("stall_steps" - 1) = 0.103 cm, the length below which the stall rule ends
    # a run that drives straight on (29 moves of 0.1 cm cover 2.9 cm, less than the rule's 3 cm)].
    "step_cm_allowed": [0.3, 1.0],
    # what the stall rule does when it fires (--stall): "end" ends the run as stalled, as the driver has done since
    # the evening of 11.11.26 (simulated runs only so far); "log" prints one line the first time, writes that picture's
    # step number into the summary ("stall_logged_at_picture": counted from 0, as the column "step" of steps.csv and
    # the frame files are) and lets the run go on.
    "stall_action": "end",
    # the recorded start: the arm pose from which 149 of the 210 demonstrations begin (degrees; the first rows of
    # frames.csv, within 0.5 degrees). The arm model puts its magnet tip at x 14.0 cm, y -1.5 cm in the robot's frame.
    # A robot run prints how far the hand-placed start is from it (the tip's distance on the table, its parts along x
    # and y, and the forearm's difference) and writes the four numbers into the summary. Beyond "start_offset_warn_m"
    # or "start_forearm_warn_deg" (judged on the numbers as printed, with one decimal) a second line says how to place
    # the start again [judgement]; no run is refused for it.
    "recorded_start_joints_deg": [-6.06, -41.87, -61.93, 0.0, 13.53, -6.15],
    "start_offset_warn_m": 0.01,
    "start_forearm_warn_deg": 3.0,
    # the placing aid (--place-help): while the arm is guided by hand with the motors off, that offset and the tip's
    # height against pushing height are shown on one line of the terminal, a new reading every "place_help_refresh_s",
    # until Enter; the keyboard is looked at every "place_help_key_poll_s". Enter counts from reading number
    # "place_help_readings_before_enter" on, so that a long or double press of the Enter that switched the motors off
    # cannot end the placing while the arm is still in the hand. It only reads the robot's state.
    "is_place_help_shown": False,
    "place_help_refresh_s": 0.5,
    "place_help_key_poll_s": 0.05,
    "place_help_readings_before_enter": 2,
    # the faint reference picture (--ghost PATH): the picture at "ghost_path" (a JPEG of the same camera, for example
    # the first picture of a demonstration) is laid over a live picture in the go-ahead window with "ghost_strength"
    # (0 = only the live picture, 1 = only the reference), a fresh picture every "ghost_refresh_s", so that the sheet
    # and the puck can be set onto their faint copies. It only reads pictures; the arm stands still. While no camera
    # picture can be read the window shows the reference alone and says so, and Enter does not count.
    "ghost_path": None,
    "ghost_strength": 0.35,
    "ghost_refresh_s": 0.33,
    # visit_runs.py starts one run of a visit from a run list and checks every checkpoint file against the first
    # "run_list_sha256_characters" characters of its sha256, which the list gives. It refuses a step budget above
    # "run_list_step_budget_most" [judgement: twice the default; the longest demonstration had 226 steps, and a line
    # of 77 cm in steps of 0.3 cm needs 257]. It starts a robot run with --fresh-keys ("is_typed_ahead_discarded"):
    # keys pressed before the robot part's first prompt is on the screen are thrown away, so that an Enter pressed
    # while the model was loading cannot switch the motors off by itself; the run then reads every key that waits in
    # the terminal at each step, so that Esc ends it also when Esc was pressed twice or another key waits behind it
    # (the committed key reader never comes back then; it is left as it is for a run without the flag); and Ctrl+C is
    # held back until the robot has answered the question under way, so that it ends a run after the move, with the
    # lift, instead of leaving the link to the robot out of step.
    "run_list_sha256_characters": 12,
    "run_list_step_budget_most": 500,
    "is_typed_ahead_discarded": False,
    # the rehearsal ("python model_mode.py rehearsal", 12.11.26): the robot part's own questions, placing aid and
    # go-ahead window against the simulated robot, so that the keys and the picture window can be tried at the laptop
    # before a visit; no robot is connected. Its runs are logged under "rehearsal_run_folder", outside 03_data. With
    # the placing aid a simulated hand brings the tip to the recorded start in "rehearsal_hand_readings" readings,
    # from "rehearsal_hand_start_m" (metres further out, to the left and higher than the recorded start) and with the
    # forearm turned "rehearsal_hand_start_forearm_deg", so that the placing line changes as it will at the robot.
    "rehearsal_run_folder": Path(tempfile.gettempdir()) / "model_mode_rehearsal",
    "rehearsal_hand_start_m": [-0.01, 0.025, 0.03],
    "rehearsal_hand_start_forearm_deg": 8.0,
    "rehearsal_hand_readings": 12
}

# 10.11.26 (after the design review): every run is also scored with the stop rule the judges chose for all models, a
# fixed threshold of 0.5 on two pictures in a row ("fixed_two_"); compare_runs.py rescores older runs from their saved
# predictions.
TRAINING_COMPARISON_PARAMETERS["columns"][-1:-1] = [("fixed_two_episodes_stopped_right_mean", "stopped right, fixed 0.5, 2 in a row"),
                                                    ("fixed_two_episodes_stopped_early_mean", "stopped early, fixed 0.5, 2 in a row"),
                                                    ("fixed_two_episodes_never_stopped_mean", "never stopped, fixed 0.5, 2 in a row")]


# Printable test tracks (make_test_tracks.py, 11.11.26; report 7.4 "a written random rule"): A4 portrait like the six
# course sheets, the line "line_width_mm" thick with round ends (dashed: square-ended dashes of "dash_mm" with "gap_mm"
# between, as sheet 06), its centre at least "edge_margin_mm" from the paper's edge. A track is a smooth random curve, kept
# only when its length lies within the six sheets' 24 to 77 cm (lengths in RECORDING
# PARAMETERS), it bends no tighter than "smallest_radius_mm" (the loops of sheets 05 and 06 measure about 40 to 50 mm on
# the photos, a guess), its parts that are not neighbours keep "smallest_gap_mm" between centres except at a planned
# crossing (crossed at "smallest_crossing_deg" or more), and its far end lies near (< "near_end_mm") or far (>
# "far_end_mm") from the edge as planned. It is driven like a car through straight pieces ("straight_share" of them) and
# arcs, each "piece_mm" long, an arc bending to the other side than the last one "side_change_share" of the time (S-curves
# and meanders against loops). The start is within "start_from_bottom_mm" of the bottom edge, the robot's
# side when the sheet lies as in the recordings. Three sets: practice (may be trained on later), validation (only for
# choosing between models) and test (sealed: printed and kept out of sight until used, as report 7.4 asks).
TEST_TRACK_PARAMETERS = {
    "page_mm": [210.0, 297.0],
    "line_width_mm": 10.0,
    "dash_mm": 35.0,
    "gap_mm": 15.0,
    "edge_margin_mm": 22.0,
    "length_mm": [240.0, 770.0],
    "smallest_radius_mm": 35.0,
    "smallest_gap_mm": 25.0,
    "smallest_crossing_deg": 45.0,
    "near_end_mm": 45.0,
    "far_end_mm": 90.0,
    "start_from_bottom_mm": 70.0,
    "piece_mm": [30.0, 140.0],
    "straight_share": 0.2,
    "side_change_share": 0.6,
    "sets": {"practice": 6, "validation": 4, "test": 10},
    # 12.11.26: this seed is no longer sealed. Its "test" set was drawn and looked at during the check of the first robot
    # test; a sealed set is drawn later by Marshal with a seed of his own (plan after the first robot test, step 4).
    "seed": 11_11_26,
    "attempts_per_track": 60_000,
    "output_folder": Path(__file__).resolve().parent.parent / "06_deliverables" / "research_notes" / "Model and training" / "test_tracks",
    # 12.11.26: the new tracks for the recording of 14.11.26 ("python make_test_tracks.py"; the set of 11.11.26 above is
    # drawn with "python make_test_tracks.py old"). The rule is matched to the six course sheets as measured from their
    # traced centre lines (course_sheets_measured.csv in "new_output_folder"): tightest bends 5.3 to 5.6 mm on sheets 03
    # and 04, loops crossing once, dashes about 20 mm with 10 mm gaps on sheet 06. Lengths in millimetres on the paper,
    # x from the paper's left edge and y from its bottom edge, the bottom edge on the robot's side.
    "new_output_folder": Path(__file__).resolve().parent.parent / "06_deliverables" / "research_notes" / "Model and training" / "tracks_2026-11-12",
    "drawing_rule": {
        "page_mm": [210.0, 297.0],
        "line_width_mm": 10.0,
        # the line's colour on every new page: the professor's original SVG files of the six course sheets (Niryo1.svg to
        # Niryo6.svg from Moodle, found 12.11.26, in tracks_2026-11-12/course_sheets_original/) draw it in #575757, a
        # dark grey, not black
        "line_colour_hex": "#575757",
        # the centre line keeps this far from every paper edge
        "edge_margin_mm": 22.0,
        # two parts of the line more than "gap_apart_along_mm" apart along it keep "smallest_gap_mm" between their
        # centres, except near a crossing: there the parts within smallest_gap_mm / sin(crossing angle) +
        # "crossing_clear_mm" of the crossing point are left out (the course sheets 03 to 05 keep 26 to 28 mm by this
        # measure)
        "smallest_gap_mm": 25.0,
        "gap_apart_along_mm": 50.0,
        "crossing_clear_mm": 10.0,
        "smallest_crossing_deg": 60.0,
        # end A: within "start_from_bottom_mm" of the bottom edge, at least "start_from_side_mm" from the side edges,
        # leaving within "start_tilt_deg" of straight up the page, along a straight first piece
        "start_straight_mm": [30.0, 60.0],
        "start_from_bottom_mm": [24.0, 45.0],
        "start_from_side_mm": 40.0,
        "start_tilt_deg": 25.0,
        "step_mm": 0.5,
        # the pieces a track is built from: radius and turn of each arc. A hairpin turns "turn_deg" in all, as the
        # hairpins of sheet 04 (164 and 170 degrees, tightest 8.8 and 9.0 mm): a shoulder of "shoulder_radius_mm"
        # turning "shoulder_turn_deg" on each side and the tight arc of "radius_mm" between, so that its legs open. A
        # loop is a straight lead in, an arc of "turn_deg" and a straight lead out that crosses the lead in (crossing
        # angle 360 - turn: 65 to 115 degrees, read as 65 to 90)
        "pieces": {
            "straight": {"length_mm": [15.0, 60.0]},
            "gentle": {"radius_mm": [35.0, 90.0], "turn_deg": [25.0, 80.0]},
            "medium": {"radius_mm": [15.0, 35.0], "turn_deg": [40.0, 120.0]},
            "sharp": {"radius_mm": [8.0, 15.0], "turn_deg": [60.0, 120.0]},
            "corner": {"radius_mm": [6.0, 10.0], "turn_deg": [70.0, 140.0]},
            "hairpin": {"radius_mm": [6.0, 10.0], "turn_deg": [150.0, 180.0], "shoulder_radius_mm": [15.0, 30.0], "shoulder_turn_deg": [15.0, 35.0]},
            "loop": {"radius_mm": [25.0, 50.0], "turn_deg": [245.0, 295.0], "lead_extra_mm": [20.0, 40.0]}
        },
        # when the line, run straight on for "steer_look_ahead_mm", would leave the paper or the reach, the next filler
        # bends towards "steer_towards_mm"; otherwise it bends to the other side than the last one "side_change_share"
        # of the time
        "steer_look_ahead_mm": 45.0,
        "steer_towards_mm": [105.0, 150.0],
        "side_change_share": 0.6,
        # dashed prints: a dash and a gap drawn per track; square dash ends as on sheet 06 (photo); the dashes are laid
        # along the line lengthened by half a line width at each end, so that the ink reaches as far as the solid print's
        # round caps; the first and last dash are lengthened or shortened by the same amount so that both ends carry ink
        "dash_mm": [15.0, 35.0],
        "gap_mm": [8.0, 15.0],
        # end B "near": within "near_end_mm" of an edge; "far": at least "far_end_mm" from every edge
        "near_end_mm": 30.0,
        "far_end_mm": 80.0,
        # at most this share of a track's length may run back towards the robot (lay-out check)
        "running_back_share_most": 0.34,
        "attempts_per_track": 60_000,
        # how bends are named when measured from a drawn centre line: a bend is a stretch bending one way with a radius
        # below "bend_radius_below_mm" (circle through three points 5 mm apart); it is a corner below 8 mm, sharp
        # below 15, medium below 35, else gentle; a hairpin when the radius goes to "hairpin_radius_most_mm" or less and
        # the whole bend turns "hairpin_turn_deg" or more (the turn within the "hairpin_window_mm" of line around the
        # tightest point is listed too: 117 to 135 degrees on the course sheets' tight bends)
        "bend_radius_below_mm": 100.0,
        "radius_classes_mm": {"corner": 8.0, "sharp": 15.0, "medium": 35.0},
        "hairpin_radius_most_mm": 10.5,
        "hairpin_turn_deg": 150.0,
        "hairpin_window_mm": 30.0
    },
    # the lay-out at the recording: the centre of end A's cap on a mark 19.5 cm in front of the base axis and 1.5 cm to
    # its right, the first straight piece along +x (straight away from the robot). The tip is "tip_behind_puck_m"
    # behind the puck's centre along the line; wrist and hand held as MODEL_MODE_PARAMETERS holds them; the forearm
    # turned so that the camera's picture points along the line (within joint_4_most_deg). The base is the square of
    # SETUP_VISUALS_PARAMETERS["base_half_width_cm"] (calc, not measured). A line point counts as running back when the
    # line's direction there is more than "running_back_deg" from straight away from the base.
    "lay_out": {
        "mark_m": [0.195, -0.015],
        "start_tip_m": [0.140, -0.015],
        "tip_behind_puck_m": 0.0185,
        "reach_aim_m": 0.45,
        "running_back_deg": 110.0,
        "check_every_mm": 5.0,
        "turns_tried_deg": [5.0, -5.0, 10.0, -10.0, 15.0, -15.0]
    },
    # small marks on a track page's face (light grey, outside the line): name and print note along the top edge, a
    # 50 mm check bar, "A" at the start end, "B" at the other end, an arrow beside the first straight piece. OFF since
    # 12.11.26 ("are_marks_printed" False): the forearm camera sees the paper round the puck, and the course and exam
    # sheets carry nothing but the line, so a letter near the line's end could be learned as the stop. The face shows
    # the line only (and on the two circle pages the grey circles, without numbers); names, ends and the print order
    # are on the list pages of the merged print files, with thumbnails ("table_rows_per_page", "thumbnails_per_page").
    "page_marks": {
        "are_marks_printed": False,
        "table_rows_per_page": 40,
        "thumbnails_per_page": 16,
        "grey": 0.55,
        "check_bar_mm": 50.0,
        "letter_size_pt": 13.0,
        "text_size_pt": 7.5,
        "mark_line_mm": 0.35,
        "clearance_mm": 4.0,
        "printable_inset_mm": 7.0
    },
    # the sets: per track a fixed seed, the planned pieces (in order), the fillers between them, the length range and
    # where end B lies ("near", "far" or "any"). Training tracks are recorded on from 14.11.26; development tracks are
    # never recorded on and are used to test models. No sealed set here (see "seed" above).
    "training": [
        {"name": "T01", "label": "gentle S-curves", "seed": 14_11_26_01, "plan": [{"kind": "gentle", "side": "left"}, {"kind": "gentle", "side": "right"}, {"kind": "gentle", "side": "left"}, {"kind": "gentle", "side": "right"}], "fillers": ["straight"], "length_mm": [300.0, 420.0], "end_b": "any"},
        {"name": "T02", "label": "two sharp bends", "seed": 14_11_26_02, "plan": [{"kind": "sharp", "radius_mm": [8.0, 10.0]}, {"kind": "sharp", "radius_mm": [8.0, 10.0]}], "fillers": ["straight"], "length_mm": [250.0, 380.0], "end_b": "any"},
        {"name": "T03", "label": "loop", "seed": 14_11_26_03, "plan": [{"kind": "loop", "side": "left"}], "fillers": ["straight", "gentle", "medium"], "length_mm": [380.0, 520.0], "end_b": "any"},
        {"name": "T04", "label": "hairpin and medium bend", "seed": 14_11_26_04, "plan": ["hairpin", "medium"], "fillers": ["straight", "gentle"], "length_mm": [350.0, 500.0], "end_b": "any"},
        {"name": "T05", "label": "loop and corner", "seed": 14_11_26_05, "plan": [{"kind": "loop", "side": "right"}, "corner"], "fillers": ["straight", "gentle", "medium"], "length_mm": [450.0, 600.0], "end_b": "far"},
        {"name": "T06", "label": "hairpin and sharp bend", "seed": 14_11_26_06, "plan": ["hairpin", "sharp"], "fillers": ["straight", "gentle", "medium"], "length_mm": [400.0, 550.0], "end_b": "near"},
        {"name": "T07", "label": "two corners", "seed": 14_11_26_07, "plan": ["corner", "corner", "medium"], "fillers": ["straight", "gentle"], "length_mm": [300.0, 450.0], "end_b": "far"},
        {"name": "T08", "label": "loop and hairpin", "seed": 14_11_26_08, "plan": [{"kind": "loop", "side": "left"}, "hairpin"], "fillers": ["straight", "gentle", "medium"], "length_mm": [500.0, 650.0], "end_b": "any"},
        {"name": "T09", "label": "short gentle", "seed": 14_11_26_09, "plan": ["gentle", "gentle"], "fillers": ["straight", "gentle"], "length_mm": [250.0, 320.0], "end_b": "near"},
        {"name": "T10", "label": "loop and medium bend", "seed": 14_11_26_10, "plan": [{"kind": "loop", "side": "right"}, "medium"], "fillers": ["straight", "gentle"], "length_mm": [450.0, 600.0], "end_b": "far"},
        {"name": "T11", "label": "three corners", "seed": 14_11_26_11, "plan": ["corner", "corner", "corner", "medium"], "fillers": ["straight", "gentle"], "length_mm": [450.0, 600.0], "end_b": "near"},
        {"name": "T12", "label": "long meander", "seed": 14_11_26_12, "plan": ["hairpin", "medium", "medium", "corner"], "fillers": ["straight", "gentle", "medium"], "length_mm": [620.0, 750.0], "end_b": "any"},
        {"name": "T13", "label": "gentle", "seed": 14_11_26_13, "plan": ["gentle", "gentle", "gentle"], "fillers": ["straight", "gentle"], "length_mm": [330.0, 450.0], "end_b": "far"},
        {"name": "T14", "label": "loop and corner", "seed": 14_11_26_14, "plan": [{"kind": "loop", "side": "left"}, "corner"], "fillers": ["straight", "gentle", "medium"], "length_mm": [500.0, 650.0], "end_b": "near"},
        {"name": "T15", "label": "mixed bends", "seed": 14_11_26_15, "plan": ["medium", "sharp", "corner", "medium"], "fillers": ["straight", "gentle"], "length_mm": [550.0, 700.0], "end_b": "far"},
        {"name": "T16", "label": "long gentle", "seed": 14_11_26_16, "plan": ["gentle", "gentle", "gentle", "gentle"], "fillers": ["straight", "gentle"], "length_mm": [450.0, 600.0], "end_b": "near"}
    ],
    "development": [
        {"name": "D01", "label": "gentle bends","seed": 26_11_12_01, "plan": ["gentle", "gentle"], "fillers": ["straight", "gentle"], "length_mm": [280.0, 400.0], "end_b": "far"},
        {"name": "D02", "label": "two sharp bends", "seed": 26_11_12_02, "plan": ["sharp", "sharp"], "fillers": ["straight"], "length_mm": [250.0, 380.0], "end_b": "near"},
        {"name": "D03", "label": "loop and medium bend", "seed": 26_11_12_03, "plan": [{"kind": "loop", "side": "right"}, "medium"], "fillers": ["straight", "gentle"], "length_mm": [420.0, 560.0], "end_b": "any"},
        {"name": "D04", "label": "hairpin and corner", "seed": 26_11_12_04, "plan": ["hairpin", "corner"], "fillers": ["straight", "gentle"], "length_mm": [380.0, 520.0], "end_b": "far"},
        {"name": "D05", "label": "loop and sharp bend", "seed": 26_11_12_05, "plan": [{"kind": "loop", "side": "left"}, "sharp"], "fillers": ["straight", "gentle", "medium"], "length_mm": [450.0, 600.0], "end_b": "near"},
        {"name": "D06", "label": "three corners", "seed": 26_11_12_06, "plan": ["corner", "corner", "corner"], "fillers": ["straight", "gentle"], "length_mm": [350.0, 500.0], "end_b": "far"},
        {"name": "D07", "label": "long gentle", "seed": 26_11_12_07, "plan": ["gentle", "gentle", "gentle"], "fillers": ["straight", "gentle"], "length_mm": [450.0, 600.0], "end_b": "any"},
        {"name": "D08", "label": "long meander", "seed": 26_11_12_08, "plan": ["hairpin", "medium", "corner", "medium"], "fillers": ["straight", "gentle", "medium"], "length_mm": [560.0, 720.0], "end_b": "near"},
        {"name": "D09", "label": "loop and hairpin", "seed": 26_11_12_09, "plan": [{"kind": "loop", "side": "right"}, "hairpin"], "fillers": ["straight", "gentle"], "length_mm": [500.0, 650.0], "end_b": "any"},
        {"name": "D10", "label": "loop, corner and gentle", "seed": 26_11_12_10, "plan": [{"kind": "loop", "side": "left"}, "corner", "gentle"], "fillers": ["straight", "gentle"], "length_mm": [500.0, 650.0], "end_b": "far"}
    ],
    # the diagnostic pages: course sheets' traced centre lines (turned half a turn so that end A lies at the bottom),
    # solid or dashed, and two pages of sheet 03 with thin grey circles of the puck's size at the places of the plan's
    # sitting A2 (proofreading facts, items 14 and 15)
    "diagnostic": {
        "sheet_measurements": Path(__file__).resolve().parent.parent / "06_deliverables" / "research_notes" / "Recording tool" / "2026-11-06_sheet_measurements.json",
        "pages": [
            {"name": "C05_solid", "sheet": "05", "is_dashed": False},
            {"name": "C05_dashed", "sheet": "05", "is_dashed": True, "dash_mm": 20.0, "gap_mm": 10.0},
            {"name": "C06_solid", "sheet": "06", "is_dashed": False},
            {"name": "C06_dashed", "sheet": "06", "is_dashed": True, "dash_mm": 20.0, "gap_mm": 10.0},
            {"name": "C01_solid", "sheet": "01", "is_dashed": False}
        ],
        "circle_sheet": "03",
        "circle_diameter_mm": 24.0,
        "circle_grey": 0.75,
        "circle_line_mm": 0.3,
        "beside_mm": [7.0, 12.0, 17.0],
        "end_short_mm": 5.0,
        "end_aside_mm": 7.0,
        "far_reach_m": 0.40,
        "forearm_turned_deg": 60.0,
        # 12.11.26: plain copies of the six course sheets ("python make_test_tracks.py course", print_set_course_sheets.pdf),
        # drawn from the traced centre lines where the trace puts them on the sheet, the line only. Sheet 06 is dashed
        # along its whole length (photo); the dash places along the line could not be read from the photo, so its dashes
        # are laid by the dash rule with 20 mm and 10 mm, which gives the photo's count of 16 dashes.
        "course_sheet_pages": [
            {"name": "C01_course", "sheet": "01", "is_dashed": False},
            {"name": "C02_course", "sheet": "02", "is_dashed": False},
            {"name": "C03_course", "sheet": "03", "is_dashed": False},
            {"name": "C04_course", "sheet": "04", "is_dashed": False},
            {"name": "C05_course", "sheet": "05", "is_dashed": False},
            {"name": "C06_course", "sheet": "06", "is_dashed": True, "dash_mm": 20.0, "gap_mm": 10.0}
        ]
    }
}
