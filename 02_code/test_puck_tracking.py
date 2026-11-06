"""
Check the puck tracking (puck_detector, forearm_camera_model, puck_tracking) and the recording tool's drawing of it,
without the robot.

Most tests need nothing but the code: made-up pictures (light paper, the black block, a red disc) stand in for the
camera. The tests on recorded pictures read episodes from 03_data/raw (not in git); on a laptop without them they say
SKIPPED and the last line counts them, so a run without the recordings never reads as fully checked. Run from 02_code
(some seconds; nothing touches the robot, nothing is written outside a temporary folder):
    python test_puck_tracking.py
"""

import csv
import math
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy

import record_episodes
from config import FOREARM_CAMERA_PARAMETERS, RECORDING_PARAMETERS, table_height
from forearm_camera_model import camera_pose_in_base, pixels_to_plane, project_points
from puck_detector import PuckDetector
from puck_tracking import PuckTracker
from record_episodes import ORIENTATION_LEGENDS, EpisodeRecorder, picture_axes_on_table
from robot_link import SimulatedRobotLink
from side_view_test import magnet_tip_and_lean

RAW_FOLDER = Path(__file__).resolve().parent.parent / RECORDING_PARAMETERS["raw_folder"]

# The arm at the start pose of sheet 04 (episode 037, row 0 of steps.csv): the rod 1.85 cm behind x 0.1707, y 0.0374.
START_JOINTS = [0.24126, -0.73527, -1.01429, 0.00163, 0.17632, 0.23786]

# Colours the orientation map draws (BGR): the seen puck, the seen puck whose place is rough, the tool's hollow guess.
SEEN_PUCK_COLOUR = (40, 40, 225)
ROUGH_PUCK_COLOUR = (130, 130, 235)
GUESS_COLOUR = (60, 60, 210)
# The ring the picture window draws round the seen puck.
RING_COLOUR = (0, 230, 255)


class RecordingsMissing(Exception):
    """Raised by a test on recorded pictures when the episode it reads is not on this laptop."""



def recorded_frame(
        session: str,
        episode: str,
        frame_index: int
) -> Optional[Dict]:
    """
    One recorded picture with the joints logged for it, or None when the episode is not on this laptop.

    Parameters
    ----------
    session : str
        The session folder in 03_data/raw.
    episode : str
        The episode folder in that session.
    frame_index : int
        The picture's number.

    Returns
    -------
    dict or None
        "picture" (BGR) and "joints" (six angles in radians).
    """

    episode_folder = RAW_FOLDER / session / episode
    picture_path = episode_folder / "frames" / f"{frame_index:03d}.jpg"
    if not picture_path.exists() or not (episode_folder / "steps.csv").exists():
        return None
    with open(episode_folder / "steps.csv", newline = "", encoding = "utf-8") as file:
        rows = [row for row in csv.DictReader(file) if row["frame"] == f"frames/{frame_index:03d}.jpg"]
    if not rows:
        return None
    return {"picture": cv2.imread(str(picture_path)), "joints": [float(rows[0][f"joint_{index}"]) for index in range(1, 7)]}


def required_frame(
        session: str,
        episode: str,
        frame_index: int
) -> Dict:
    """
    As ``recorded_frame``, but a missing episode skips the test that asked for it.

    Parameters
    ----------
    session : str
        The session folder in 03_data/raw.
    episode : str
        The episode folder in that session.
    frame_index : int
        The picture's number.

    Returns
    -------
    dict
        "picture" (BGR) and "joints" (six angles in radians).
    """

    frame = recorded_frame(session, episode, frame_index)
    if frame is None:
        raise RecordingsMissing(f"{episode} is not on this laptop")
    return frame


def synthetic_picture(
        puck_centre_px: Optional[Sequence[int]] = None,
        puck_radius_px: int = 31
) -> numpy.ndarray:
    """
    A made-up camera picture: light paper with a little noise, the black block at the bottom middle, and the red puck
    when a centre is given (31 px is the median radius of a whole puck in the recordings).

    Parameters
    ----------
    puck_centre_px : sequence of int, optional
        The puck's centre (u, v); None for a picture without the puck.
    puck_radius_px : int
        The puck's radius in the picture.

    Returns
    -------
    numpy.ndarray
        The picture, 640 x 480, BGR.
    """

    generator = numpy.random.default_rng(3)
    picture = numpy.clip(generator.normal((226, 229, 232), 2.0, (480, 640, 3)), 0, 255).astype(numpy.uint8)
    cv2.rectangle(picture, (250, 395), (400, 479), (35, 35, 35), -1)
    if puck_centre_px is not None:
        cv2.circle(picture, tuple(int(value) for value in puck_centre_px), puck_radius_px, (48, 42, 196), -1, lineType = cv2.LINE_AA)
    return picture


def start_state() -> Dict:
    """The robot's state at the start pose, as RobotLink.read_state gives it (the pose's angles are not used here)."""

    tip_xyz = magnet_tip_and_lean(START_JOINTS)[0]
    return {"joints": list(START_JOINTS), "pose": [float(tip_xyz[0]), float(tip_xyz[1]), float(tip_xyz[2]), 0.0, 0.0, 0.0]}


def map_recorder(
        folder: str,
        recorder_parameters: Optional[Dict] = None
) -> EpisodeRecorder:
    """
    A recorder on the simulated robot that draws as the real one does (windows on, joints taken as real).

    Parameters
    ----------
    folder : str
        A temporary folder for anything the recorder might write.
    recorder_parameters : dict, optional
        Further overrides.

    Returns
    -------
    EpisodeRecorder
        The recorder; nothing is shown until a test calls its drawing.
    """

    parameters = {"raw_folder": folder, "is_window_shown": True, "is_simulated": False}
    parameters.update(recorder_parameters or {})
    return EpisodeRecorder(SimulatedRobotLink(), operator = "test", recorder_parameters = parameters, key_reader = lambda wait_ms: None)


def colour_count(
        image: numpy.ndarray,
        colour: Sequence[int]
) -> int:
    """
    Count the pixels of exactly one colour.

    Parameters
    ----------
    image : numpy.ndarray
        A BGR image.
    colour : sequence of int
        The colour (B, G, R).

    Returns
    -------
    int
        How many pixels have it.
    """

    return int(numpy.all(image == numpy.array(colour, dtype = numpy.uint8), axis = 2).sum())


def test_pixels_return_to_themselves() -> None:
    """A pixel followed to the table and projected back lands on itself, across the whole picture."""

    width, height = FOREARM_CAMERA_PARAMETERS["image_size_px"]
    pixels = numpy.array([[u, v] for u in numpy.linspace(20, width - 20, 9) for v in numpy.linspace(20, height - 20, 7)])
    plane_height_m = table_height(0.25, 0.0)
    points_xy = pixels_to_plane(pixels, START_JOINTS, plane_height_m)
    is_on_table = numpy.all(numpy.isfinite(points_xy), axis = 1)
    assert is_on_table.sum() >= 40, is_on_table.sum()
    points = numpy.column_stack([points_xy[is_on_table], numpy.full(is_on_table.sum(), plane_height_m)])
    error_px = numpy.abs(project_points(points, START_JOINTS) - pixels[is_on_table]).max()
    assert error_px < 1e-6, error_px


def test_camera_turns_with_the_forearm_only() -> None:
    """Joint 4 turns the camera about the forearm's axis; joints 5 and 6 (the wrist) do not move it."""

    position, rotation = camera_pose_in_base(START_JOINTS)
    wrist_moved = list(START_JOINTS)
    wrist_moved[4] += 0.4
    wrist_moved[5] -= 0.7
    moved_position, moved_rotation = camera_pose_in_base(wrist_moved)
    assert numpy.abs(moved_position - position).max() < 1e-12 and numpy.abs(moved_rotation - rotation).max() < 1e-12

    turned = list(START_JOINTS)
    turned[3] += math.radians(30.0)
    turned_position, turned_rotation = camera_pose_in_base(turned)
    relative = turned_rotation @ rotation.T
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, (numpy.trace(relative) - 1.0) / 2.0))))
    assert abs(angle_deg - 30.0) < 1e-6, angle_deg
    assert abs(numpy.linalg.norm(turned_position - position) - 2 * 0.058 * math.sin(math.radians(15.0))) < 0.002


def test_outline_is_ordered_and_points_away_at_the_start() -> None:
    """At the start pose the outline runs top left, top right, bottom right, bottom left, and its top lies further out."""

    outline = PuckTracker().picture_outline_on_table(START_JOINTS)
    assert outline.shape == (4, 2), outline.shape
    top_middle, bottom_middle = (outline[0] + outline[1]) / 2, (outline[2] + outline[3]) / 2
    assert numpy.hypot(*top_middle) > numpy.hypot(*bottom_middle) + 0.05, (top_middle, bottom_middle)
    # Seen from above with x away from the robot and y to its left, the picture's right is the robot's right (-y)
    # while the forearm is straight.
    assert outline[1][1] < outline[0][1] and outline[2][1] < outline[3][1], outline


def test_tool_directions_match_the_fitted_camera() -> None:
    """
    The recording tool's picture directions (picture_axes_on_table: its heading model and picture_left_sign) agree with
    the camera fitted on 6.11.26: a table point moved the tool's "left" appears to the left in the picture, and one moved
    "up" appears higher, with the forearm straight and turned up to 57 degrees either way. picture_left_sign was tested
    on the robot only for the wrist mounts (config.py), so this is its check for the forearm camera.
    """

    for forearm_turn_rad in (0.0, 0.6, -0.6, 1.0, -1.0):
        joints = list(START_JOINTS)
        joints[3] = forearm_turn_rad
        up_xy, left_xy = picture_axes_on_table({"joints": joints, "pose": [0.0] * 6}, RECORDING_PARAMETERS)
        tip_xy = numpy.array(magnet_tip_and_lean(joints)[0][:2])
        ahead_xy = tip_xy + 0.05 * up_xy
        height_m = table_height(ahead_xy[0], ahead_xy[1])
        points = numpy.array([[*ahead_xy, height_m], [*(ahead_xy + 0.03 * left_xy), height_m], [*(ahead_xy + 0.03 * up_xy), height_m]])
        centre, moved_left, moved_up = project_points(points, joints)
        left_angle_deg = math.degrees(math.atan2(moved_left[1] - centre[1], centre[0] - moved_left[0]))
        up_angle_deg = math.degrees(math.atan2(moved_up[0] - centre[0], centre[1] - moved_up[1]))
        # Measured 6.11.26: at most 0.99 degrees at forearm turns from -57 to +57 degrees.
        assert abs(left_angle_deg) < 1.5 and abs(up_angle_deg) < 1.5, (forearm_turn_rad, left_angle_deg, up_angle_deg)


def test_recorder_uses_the_fitted_camera_mount() -> None:
    """The recording tool's camera mount is the one the camera was fitted for, so the map places the puck."""

    assert RECORDING_PARAMETERS["camera_mount"] == FOREARM_CAMERA_PARAMETERS["camera_mount"], (
        RECORDING_PARAMETERS["camera_mount"], FOREARM_CAMERA_PARAMETERS["camera_mount"])
    assert PuckTracker().is_calibrated()


def test_detector_finds_a_drawn_puck() -> None:
    """A red disc drawn on light paper in front of the block is found at its centre; a picture without it gives nothing."""

    detector = PuckDetector()
    for centre, radius in (((325, 345), 31), ((320, 250), 28), ((420, 330), 33), ((200, 300), 30)):
        detection = detector.detect(synthetic_picture(centre, radius))
        assert detection is not None, centre
        assert math.hypot(detection["centre_u"] - centre[0], detection["centre_v"] - centre[1]) < 1.0, (centre, detection["centre_u"], detection["centre_v"])
        assert detection["confidence"] > 0.9 and not detection["is_partial"], detection
    assert detector.detect(synthetic_picture()) is None

    sighting = PuckTracker().see_puck(synthetic_picture((325, 345)), START_JOINTS)
    outline = PuckTracker().picture_outline_on_table(START_JOINTS)
    assert sighting is not None and sighting["table_xy"] is not None and not sighting["is_position_rough"], sighting
    assert cv2.pointPolygonTest(outline.astype(numpy.float32), tuple(float(value) for value in sighting["table_xy"]), False) > 0


def test_other_mount_places_nothing() -> None:
    """With a camera mount the fit does not belong to, the tracker still finds the puck but places nothing on the table."""

    tracker = PuckTracker(tracking_parameters = {"camera_mount": "wrist_bracket_F"})
    assert not tracker.is_calibrated()
    assert tracker.picture_outline_on_table(START_JOINTS) is None
    sighting = tracker.see_puck(synthetic_picture((325, 345)), START_JOINTS)
    assert sighting is not None and sighting["table_xy"] is None, sighting


def test_settings_are_merged_not_replaced() -> None:
    """Camera and detector overrides given one after another add up instead of replacing each other."""

    tracker = PuckTracker()
    tracker.update_tracking_parameters({"camera_parameters": {"focal_length_px": 400.0}, "detector_parameters": {"minimum_score": 0.6}})
    tracker.update_tracking_parameters({"camera_parameters": {"principal_point_px": [320.0, 240.0]}})
    tracker.update_tracking_parameters({"detector_parameters": {"maximum_area_px": 8000}})
    assert tracker.camera_parameters["focal_length_px"] == 400.0 and tracker.camera_parameters["principal_point_px"] == [320.0, 240.0]
    assert tracker.detector.detector_parameters["minimum_score"] == 0.6 and tracker.detector.detector_parameters["maximum_area_px"] == 8000
    assert tracker.tracking_parameters["detector_parameters"] == {"minimum_score": 0.6, "maximum_area_px": 8000}
    # At least one pass is always made, whatever the setting says.
    assert PuckTracker({"height_passes": 0}).see_puck(synthetic_picture((325, 345)), START_JOINTS)["table_xy"] is not None


def test_map_draws_the_tracking_in_every_view() -> None:
    """
    The orientation map in every view: the seen puck solid (paler when its place is rough), and the tool's hollow guess
    only where forearm turns use it (keyboard and the 1 cm replay beside a seen puck 2.5 cm away; also gamepad when the
    camera cannot place the puck; never exact replay or guiding).
    """

    picture_bytes = cv2.imencode(".jpg", synthetic_picture((325, 345)), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    blank_bytes = cv2.imencode(".jpg", synthetic_picture(), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    state = start_state()
    with tempfile.TemporaryDirectory() as folder:
        recorder = map_recorder(folder)
        sighting = recorder._EpisodeRecorder__see_puck(picture_bytes, state)
        assert sighting is not None and sighting["table_xy"] is not None, sighting
        guess_beside = {"keyboard": True, "replay": True, "gamepad": False, "exact_replay": False, "guiding": False}
        guess_alone = {"keyboard": True, "replay": True, "gamepad": True, "exact_replay": False, "guiding": False}
        for view in ORIENTATION_LEGENDS:
            recorder.puck_sighting = sighting
            recorder.puck_xy_estimate = numpy.array(sighting["table_xy"]) + [0.0, 0.025]
            canvas = recorder._EpisodeRecorder__orientation_canvas(state, view)
            assert canvas.shape[1] == 480, canvas.shape
            assert colour_count(canvas, SEEN_PUCK_COLOUR) > 150, (view, colour_count(canvas, SEEN_PUCK_COLOUR))
            assert (colour_count(canvas, GUESS_COLOUR) > 20) == guess_beside[view], (view, colour_count(canvas, GUESS_COLOUR))

            recorder.puck_sighting = dict(sighting, is_position_rough = True)
            canvas = recorder._EpisodeRecorder__orientation_canvas(state, view)
            assert colour_count(canvas, ROUGH_PUCK_COLOUR) > 150 and colour_count(canvas, SEEN_PUCK_COLOUR) == 0, view

            recorder.puck_sighting = recorder._EpisodeRecorder__see_puck(blank_bytes, state)
            assert recorder.puck_sighting is None
            canvas = recorder._EpisodeRecorder__orientation_canvas(state, view)
            assert colour_count(canvas, SEEN_PUCK_COLOUR) == 0, view
            assert (colour_count(canvas, GUESS_COLOUR) > 20) == guess_alone[view], (view, colour_count(canvas, GUESS_COLOUR))

        # Twelve points per picture edge draw as well as the four corners.
        recorder.puck_tracker.update_tracking_parameters({"outline_points_per_edge": 12})
        recorder.puck_sighting = sighting
        assert recorder._EpisodeRecorder__orientation_canvas(state, "keyboard").shape[1] == 480


def test_map_survives_tracking_faults() -> None:
    """A fault in the tracker or in drawing it leaves the episode running: the map is drawn without it, said once."""

    picture_bytes = cv2.imencode(".jpg", synthetic_picture((325, 345)))[1].tobytes()
    state = start_state()
    shown = []
    saved = {name: getattr(record_episodes.cv2, name) for name in ("imshow", "setWindowProperty")}
    record_episodes.cv2.imshow = lambda name, image: shown.append(image)
    record_episodes.cv2.setWindowProperty = lambda *arguments: None
    try:
        with tempfile.TemporaryDirectory() as folder:
            recorder = map_recorder(folder)
            recorder.is_orientation_window_placed = True

            def failing_see_puck(picture, joints):
                raise RuntimeError("injected fault")

            recorder.puck_tracker.see_puck = failing_see_puck
            assert recorder._EpisodeRecorder__see_puck(picture_bytes, state) is None
            assert recorder.is_tracking_failing and recorder.is_tracking_error_reported

            recorder.puck_tracker.picture_outline_on_table = lambda joints: numpy.zeros((3,))
            recorder._EpisodeRecorder__show_orientation(state, view = "keyboard")
            assert len(shown) == 1 and shown[0].shape[1] == 480, len(shown)
    finally:
        for name, function in saved.items():
            setattr(record_episodes.cv2, name, function)


def test_picture_window_marks_the_puck_on_a_copy() -> None:
    """The picture window rings the seen puck, while the picture's bytes (the saved frame) stay as the camera took them."""

    picture_bytes = cv2.imencode(".jpg", synthetic_picture((325, 345)), [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    original = bytes(picture_bytes)
    shown = []
    saved_show = record_episodes.show_in_window
    record_episodes.show_in_window = lambda picture: shown.append(picture)
    try:
        with tempfile.TemporaryDirectory() as folder:
            recorder = map_recorder(folder)
            recorder.puck_sighting = recorder._EpisodeRecorder__see_puck(picture_bytes, start_state())
            recorder._EpisodeRecorder__show_picture(picture_bytes, "step 0")
    finally:
        record_episodes.show_in_window = saved_show
    assert len(shown) == 1 and colour_count(shown[0][:480], RING_COLOUR) > 50, colour_count(shown[0][:480], RING_COLOUR)
    assert picture_bytes == original
    decoded = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR)
    assert colour_count(decoded, RING_COLOUR) == 0


def test_recorded_start_pictures_place_the_puck() -> None:
    """
    The puck in the first pictures of episodes 037 and 042 lands where the fit of 6.11.26 put it (its start check):
    in 037 it touches the block, 2.2 cm from the magnet tip; in 042 it lay about 1.3 cm further out.
    """

    tracker = PuckTracker()
    expected = {("2026-11-04_session_01_emranul", "path_04_meander_episode_037"): ([322.36, 337.15], [0.1743, 0.0385]),
                ("2026-11-05_session_01_emranul", "path_04_meander_episode_042"): ([327.39, 306.99], [0.1870, 0.0394])}
    for (session, episode), (pixel, table_xy) in expected.items():
        frame = required_frame(session, episode, 0)
        sighting = tracker.see_puck(frame["picture"], frame["joints"])
        assert sighting is not None, episode
        assert numpy.hypot(*(numpy.array(sighting["pixel"]) - pixel)) < 1.0, (episode, sighting["pixel"])
        assert numpy.hypot(*(numpy.array(sighting["table_xy"]) - table_xy)) < 0.002, (episode, sighting["table_xy"])
        outline = tracker.picture_outline_on_table(frame["joints"])
        assert cv2.pointPolygonTest(outline.astype(numpy.float32), tuple(float(value) for value in sighting["table_xy"]), False) > 0


def test_no_puck_is_found_without_one() -> None:
    """Episode 039 was run without the puck, with a red mouse and red cables in view: none of its pictures shows a puck."""

    tracker = PuckTracker()
    frames = [recorded_frame("2026-11-04_session_01_emranul", "path_04_meander_episode_039", index) for index in range(0, 138)]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise RecordingsMissing("path_04_meander_episode_039 is not on this laptop")
    found = [index for index, frame in enumerate(frames) if tracker.see_puck(frame["picture"], frame["joints"]) is not None]
    assert not found, found


def test_puck_is_followed_through_a_push() -> None:
    """
    In the gamepad episode 045, which reached the goal, the puck is found in nearly every tenth picture, stays on the
    reachable sheet, and never jumps further than the rod's tip travelled in between plus 1.5 cm (measured 6.11.26: at
    most 3.4 mm more, the map's own scatter); a jump to some other red thing would break that.
    """

    tracker = PuckTracker()
    tips, positions = {}, {}
    for index in range(10, 391):
        frame = recorded_frame("2026-11-05_session_01_emranul", "path_04_meander_episode_045", index)
        if frame is None:
            continue
        tips[index] = numpy.array(magnet_tip_and_lean(frame["joints"])[0][:2])
        if index % 10 == 0:
            sighting = tracker.see_puck(frame["picture"], frame["joints"])
            positions[index] = None if sighting is None else numpy.array(sighting["table_xy"])
    if not positions:
        raise RecordingsMissing("path_04_meander_episode_045 is not on this laptop")
    seen = {index: position for index, position in positions.items() if position is not None}
    assert len(seen) >= 0.9 * len(positions), (len(seen), len(positions))
    assert all(0.12 < position[0] < 0.46 and -0.16 < position[1] < 0.14 for position in seen.values()), seen
    for index in sorted(seen):
        if index + 10 not in seen:
            continue
        travel_m = sum(float(numpy.linalg.norm(tips[step + 1] - tips[step])) for step in range(index, index + 10) if step in tips and step + 1 in tips)
        jump_m = float(numpy.linalg.norm(seen[index + 10] - seen[index]))
        assert jump_m <= travel_m + 0.015, (index, jump_m, travel_m)


if __name__ == "__main__":
    tests = (test_pixels_return_to_themselves, test_camera_turns_with_the_forearm_only, test_outline_is_ordered_and_points_away_at_the_start,
             test_tool_directions_match_the_fitted_camera, test_recorder_uses_the_fitted_camera_mount, test_detector_finds_a_drawn_puck,
             test_other_mount_places_nothing, test_settings_are_merged_not_replaced, test_map_draws_the_tracking_in_every_view,
             test_map_survives_tracking_faults, test_picture_window_marks_the_puck_on_a_copy,
             test_recorded_start_pictures_place_the_puck, test_no_puck_is_found_without_one, test_puck_is_followed_through_a_push)
    skipped = []
    for test in tests:
        try:
            test()
        except RecordingsMissing as reason:
            skipped.append(test.__name__)
            print(f"{test.__name__}: SKIPPED ({reason})")
            continue
        print(f"{test.__name__}: passed")
    if skipped:
        print(f"{len(tests) - len(skipped)} passed, {len(skipped)} SKIPPED: the recorded episodes are not on this laptop, so those checks did not run")
    else:
        print("all tests passed")
