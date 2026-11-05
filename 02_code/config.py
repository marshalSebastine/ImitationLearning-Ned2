"""
Machine-wide defaults for the robot code.

Every script reads its settings from these dictionaries, so a value is changed
in one place only. Later phases add the keys named in the work plan.
"""

import math

ROBOT_PARAMETERS = {
    "ip_address": "169.254.200.200",  # Ethernet cable (Emranul's laptop); 10.10.10.10 over the robot's hotspot
    "software_version": "6.1.2-b374",
    "arm_max_velocity_percent": 20
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
        "forward_left": 45,
        "forward_right": -45,
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
        "z": "back_left",
        "c": "back_right",
        "r": "step_over_forward",
        "f": "step_over_back",
        " ": "stop",
        "x": "discard",
        "u": "mistake",
        "l": "lost",
        "enter": "accept"
    },
    "steps_until_fresh_picture": 2,
    "fresh_picture_timeout_s": 3.0,
    # The step budget is 1.5 times the line length in centimetres, per sheet; "step_budget" is the fallback for a
    # sheet whose length is unknown. Lengths: sheet 01 measured (24 cm); 02 to 06 traced from the sheet photos on
    # 26.10.26 (06_deliverables/research_notes/Camera mount options/scripts/sheet_paths.json, calc, about 5 percent
    # uncertain; the dashed sheet 06 is taken as long as the loop it copies). Replace with the SVG arc lengths.
    "step_budget_per_cm": 1.5,
    "step_budget": 80,
    "line_length_cm": {
        "path_01_straight_line": 24,
        "path_02_arc": 27,
        "path_03_two_curve": 55,
        "path_04_meander": 77,
        "path_05_alpha_loop": 57,
        "path_06_alpha_loop_dashed": 57
    },
    "approach_height_m": 0.05,
    "home_joints": [0.0, 0.5, -1.25, 0.0, 0.0, 0.0],
    "guided_path_sample_m": 0.002,
    "guided_path_poll_s": 0.05,
    # The puck's centre sits this far beyond the rod's end along the push (puck radius plus tip radius); the
    # workspace guard refuses a step that would carry it outside the box or beyond the sheet plus "sheet_margin_m".
    "puck_centre_ahead_of_tip_m": PUCK_PARAMETERS["diameter_m"] / 2 + PUSHER_PARAMETERS["tip_diameter_m"] / 2,
    "sheet_margin_m": 0.02,
    # Written into every episode.json. Change "camera_mount" when the bracket is fitted and "mapping_version" whenever
    # camera_heading_rad, picture_left_sign or the actions change; a training set must never mix two values of either.
    # Bracket F is the default since 4.11.26, because real data is recorded with it; a rehearsal on the old holder passes
    # --camera-mount stock_wrist_holder.
    "camera_mount": "bracket_F",
    # Raise this whenever 06_deliverables/research_notes/Recording tool/stepping_rule_v1.md changes its version.
    "stepping_rule_version": "v1",
    # Asked after each episode; Esc at either question stores "not_recorded".
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
    "mapping_version": "picture_frame_v3_step_over_2026-11-04",
    # UNTESTED: whether "left" in the picture is counter-clockwise from the camera's heading seen from above (a
    # camera looking straight down at the table normally shows it unmirrored). First robot test: one "left" step must
    # move the scene to the right in the picture. If it moves left, set this to -1.
    "picture_left_sign": 1,
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
        x: float,
        y: float
) -> float:
    """
    The table direction that the top of the picture points to, when the rod's end is at (x, y).

    With the camera in Niryo's wrist holder and the rod vertical, the picture always looks straight away from the
    base (CAMERA_VIEW_PARAMETERS), so its top points along the line from the base through the rod. A new camera
    mount needs this function checked again.
    """

    return math.atan2(y, x)


def picture_step_to_table(
        action: str,
        x: float,
        y: float
) -> list:
    """The table displacement [dx, dy] in metres of one ``action`` (a step or a step-over) with the rod's end at (x, y)."""

    is_step_over = action in RECORDING_PARAMETERS["step_over_actions"]
    picture_angle = RECORDING_PARAMETERS["step_over_actions"][action] if is_step_over else RECORDING_PARAMETERS["actions"][action]
    length = RECORDING_PARAMETERS["step_over_m"] if is_step_over else RECORDING_PARAMETERS["step_m"]
    angle = camera_heading_rad(x, y) + RECORDING_PARAMETERS["picture_left_sign"] * math.radians(picture_angle)
    return [length * math.cos(angle), length * math.sin(angle)]


# Synthetic camera pictures rendered with the Webots simulator (R2025a, installed on Emranul's laptop), chosen by Emranul
# on 5.11.26 instead of pasting cut-outs of real pictures: a simulated sheet, puck, rod, magnet block and camera; later a
# written rule chooses the key for each picture from the simulator's exact positions. Nothing here moves the robot.
# First mount: bracket F3 on the hand (3.11 to 4.11.26), where camera, rod and block are rigid to each other and the
# picture's top looks along the table's +x axis (checked on all steps of episode 018). The camera pose is a first estimate
# (calc) from episode 018, step 60: the 1 cm line is about 40 pixels wide at the top of the picture and 31 at the bottom,
# which with Niryo's lens calibration (fx 508 pixels) gives a tilt of about 15 degrees back towards the rod and a lens
# about 13.8 cm above the paper; it is refined by comparing renders with real pictures.
SYNTHETIC_PARAMETERS = {
    "webots_executable": "C:/Program Files/Webots/msys64/mingw64/bin/webots.exe",
    "project_folder": "02_code/webots_pushing",
    "output_folder": "03_data/synthetic/webots",
    "camera_mount": "bracket_F3",
    # From the point on the paper under the rod's tip: ahead (+x), left (+y) and up (+z), in metres.
    "camera_position_from_tip_m": [0.048, 0.0, 0.138],
    "camera_tilt_back_deg": 15.0,
    "camera_field_of_view_rad": 2 * math.atan(320 / 508),
    "camera_width": CAMERA_PARAMETERS["width"],
    "camera_height": CAMERA_PARAMETERS["height"],
    # The puck is put on the paper this long, then the picture is taken (the puck must stop wobbling).
    "settle_time_s": 0.2,
    # Sheets are A4 pages drawn from the traced centre lines of the course photos (sheet_paths.json, about 5 percent
    # uncertain). "page_corner_in_photo_cm" is where the page's top-left corner lies in the photo's frame, read from the
    # sheet 04 photo (calc, about 0.5 cm uncertain).
    "sheet_paths_file": "06_deliverables/research_notes/Camera mount options/scripts/sheet_paths.json",
    "page_size_cm": [21.0, 29.7],
    "page_corner_in_photo_cm": [-1.0, -3.0],
    "texture_pixels_per_cm": 60,
    "line_width_cm": 100 * SHEET_PARAMETERS["line_width_m"],
    # Where a page lay on the table: its line's first traced point (the end nearest the robot) under "line_start_xy", and
    # the page's downward direction turned by "page_yaw_rad" from the table's +x axis. Sheet 04 from 2.11 to 4.11.26: the
    # start is the puck's start position of those episodes (rod's start pose plus the contact offset).
    "sheet_placements": {
        "path_04_meander": {"line_start_xy": [0.1707, 0.0374], "page_yaw_rad": 0.0}
    },
    # Real pictures re-created for the first side-by-side check: the rod's logged tip position, the puck touching it ahead.
    "first_look_situations": {
        "episode_029_step_000": {"path_id": "path_04_meander", "tip_xy": [0.15137, 0.03722]},
        "episode_018_step_060": {"path_id": "path_04_meander", "tip_xy": [0.28692, 0.03800]}
    }
}
