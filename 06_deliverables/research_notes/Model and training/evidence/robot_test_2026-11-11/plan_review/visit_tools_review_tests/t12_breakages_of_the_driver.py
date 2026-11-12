"""
Review check: break a COPY of the driver in ways the builder's own list did not try, and run every new test against
each breakage. A breakage that no test notices marks a behaviour the tests do not really check.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

CODE = Path("D:/AI_Project/.claude/worktrees/training-analysis/02_code")
HERE = Path(tempfile.mkdtemp(prefix = "review_breakages_"))
source = (CODE / "model_mode.py").read_text(encoding = "utf-8")
(HERE / "test_model_mode.py").write_text((CODE / "test_model_mode.py").read_text(encoding = "utf-8"), encoding = "utf-8")

NEW_TESTS = ["test_step_cm_flag_is_read_and_refused_where_it_does_not_fit", "test_simulated_run_with_step_cm_logs_shorter_tip_moves",
             "test_stall_rule_can_log_and_let_the_run_go_on", "test_start_offset_is_zero_at_the_recorded_start_and_its_signs_are_right",
             "test_start_offset_lines_warn_only_beyond_the_limits", "test_placing_help_shows_each_reading_until_enter",
             "test_only_enter_ends_the_wait_in_the_terminal", "test_ghost_blend_lays_the_reference_over_the_live_picture",
             "test_reference_picture_is_none_when_missing_or_not_a_picture", "test_go_ahead_with_ghost_shows_live_pictures_until_a_key",
             "test_robot_run_without_new_flags_asks_what_it_asked_before", "test_robot_run_with_the_placing_aid_and_the_ghost",
             "test_keys_pressed_too_early_do_not_answer_the_first_prompt_with_fresh_keys", "test_missing_ghost_is_reported_before_anything_is_asked_of_the_robot",
             "test_a_hesitating_model_ends_the_run_as_stalled", "test_simulated_run_takes_one_centimetre_steps"]

MUTATIONS = [
    ("forearm difference with the wrong sign", 'math.degrees(joints_placed[3] - recorded_joints[3])', 'math.degrees(recorded_joints[3] - joints_placed[3])'),
    ("further out and closer swapped", 'along_x_cm = 100.0 * float(placed_tip[0] - recorded_tip[0])', 'along_x_cm = -100.0 * float(placed_tip[0] - recorded_tip[0])'),
    ("distance is not the straight-line distance", '"tip_distance_cm": math.hypot(along_x_cm, along_y_cm),', '"tip_distance_cm": abs(along_x_cm) + abs(along_y_cm),'),
    ("a forearm turned the negative way never warns", 'and abs(offset["forearm_deg"]) <= settings["start_forearm_warn_deg"])', 'and offset["forearm_deg"] <= settings["start_forearm_warn_deg"])'),
    ("fixed_step_cm is a number under the predicted rule", '"fixed_step_cm": None if self.parameters["step_length_rule"] == "predicted" else round(100.0 * self.parameters["step_m"], 4),',
     '"fixed_step_cm": round(100.0 * self.parameters["step_m"], 4),'),
    ("first-to-last-picture time is the whole run's time", '"seconds_first_to_last_picture": round(rows[-1]["unix_time"] - rows[0]["unix_time"], 1) if rows else None,',
     '"seconds_first_to_last_picture": round((datetime.datetime.now() - started).total_seconds(), 1) if rows else None,'),
    ("stall picture counted from 1", 'stall_logged_at_picture = step\n', 'stall_logged_at_picture = step + 1\n'),
    ("keys typed before the aid count as its Enter", '        # keys typed before the aid was on the screen do not count as its Enter\n        read_key()\n', ''),
    ("the aid runs although the terminal is no console", '    if not is_terminal_a_console():\n', '    if False:\n'),
    ("the live window refreshes a thousand times a second", 'wait_ms = max(1, int(round(1000 * settings["ghost_refresh_s"])))', 'wait_ms = 1'),
    ("start offset taken from the recorded joints (always zero)", 'start_offset = offset_from_recorded_start(joints_placed, settings)', 'start_offset = offset_from_recorded_start([math.radians(value) for value in settings["recorded_start_joints_deg"]], settings)'),
    ("the window still says 0.7 cm for a 0.65 cm step", 'step_text = f"{step_cm:.2f}" if is_written_in_full else f"{step_cm:.1f}"', 'step_text = f"{step_cm:.1f}"'),
    ("the start line is printed before the motors are back on", '''        joints_placed = robot_link.read_state()["joints"]
        robot_link.set_free_motion(False)
        are_motors_off = False
        start_offset = offset_from_recorded_start(joints_placed, settings)
        print("\\n".join(start_offset_lines(start_offset, settings)))''', '''        joints_placed = robot_link.read_state()["joints"]
        start_offset = offset_from_recorded_start(joints_placed, settings)
        print("\\n".join(start_offset_lines(start_offset, settings)))
        robot_link.set_free_motion(False)
        are_motors_off = False'''),
    ("the aid reads the state with the motors switched on first", '        if not placed_with_help(robot_link, settings):', '        if not placed_with_help(robot_link, dict(settings, is_place_help_shown = False)):'),
    ("the ghost go-ahead never asks the camera again (one still picture, blended)", '''    while True:
        picture_bytes = read_picture_bytes()
        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR) if picture_bytes else None
        if picture is not None:
            shown = ghost_blend(picture, ghost, settings["ghost_strength"])''', '''    picture_bytes = read_picture_bytes()
    while True:
        picture = cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR) if picture_bytes else None
        if picture is not None:
            shown = ghost_blend(picture, ghost, settings["ghost_strength"])'''),
    ("--fresh-keys throws the keys away after the first prompt, not before", '''        if settings["is_typed_ahead_discarded"] and discard_typed_keys():
            print("keys pressed before this prompt were ignored")
        input("Put the puck at the line's start. Press Enter to switch the motors OFF (hold the arm: it will sag) ... ")''',
     '''        input("Put the puck at the line's start. Press Enter to switch the motors OFF (hold the arm: it will sag) ... ")
        if settings["is_typed_ahead_discarded"] and discard_typed_keys():
            print("keys pressed before this prompt were ignored")'''),
    ("the ghost is read only after the robot is connected", '''    ghost = reference_picture(settings["ghost_path"]) if settings["ghost_path"] else None
    if settings["ghost_path"] and ghost is None:
        problem = f"the reference picture {settings['ghost_path']} is missing or cannot be read as a picture (--ghost)"
        print(f"not started: {problem}")
        return {"label": label, "ending": "not_started", "reason": problem}
    start_offset_lines(offset_from_recorded_start([math.radians(value) for value in settings["recorded_start_joints_deg"]], settings), settings)
    policy = ModelPolicy(settings["run_name"], settings["checkpoint_path"], settings["metrics_path"]) if stand_in_policy is None else stand_in_policy
    robot_link = RobotLink() if stand_in_link is None else stand_in_link
''', '''    policy = ModelPolicy(settings["run_name"], settings["checkpoint_path"], settings["metrics_path"]) if stand_in_policy is None else stand_in_policy
    robot_link = RobotLink() if stand_in_link is None else stand_in_link
    ghost = reference_picture(settings["ghost_path"]) if settings["ghost_path"] else None
    if settings["ghost_path"] and ghost is None:
        problem = f"the reference picture {settings['ghost_path']} is missing or cannot be read as a picture (--ghost)"
        print(f"not started: {problem}")
        return {"label": label, "ending": "not_started", "reason": problem}
    start_offset_lines(offset_from_recorded_start([math.radians(value) for value in settings["recorded_start_joints_deg"]], settings), settings)
'''),
    ("with --stall log the stalled run is also not moved on (the step is skipped)", '''                        print(f"[step {step}] the stall rule fired: the tip stayed within {100 * self.parameters['stall_extent_m']:.0f} cm and the forearm "''',
     '''                        print(f"[step {step}] the stall rule fired: the tip stayed within {100 * self.parameters['stall_extent_m']:.0f} cm and the forearm ", end = "")
                        print(f"[step {step}] "'''),
    ("the placing line says close when the tip is far but the forearm is right", "f\"{'close' if is_start_close(offset, parameters) else 'not close'}\")", "f\"{'close' if abs(offset['forearm_deg']) <= 3.0 else 'not close'}\")"),
    ("Ctrl+C in the placing aid is swallowed and placing ends as if Enter was pressed", '''    except KeyboardInterrupt:
        # Ctrl+C: the line is ended first, so that the question about the motors that follows starts on a line of its own
        print()
        raise''', '''    except KeyboardInterrupt:
        print()
        return True'''),
]

# two breakages of the first try are left out: one only moved a print (no requirement says where the start line is
# printed), the other changed nothing that a test could see (it was badly made, not a gap in the tests)
MUTATIONS = [mutation for mutation in MUTATIONS if not mutation[0].startswith(("with --stall log", "the start line is printed"))]


def run_tests() -> list:
    """Run every new test against the copy in this folder; the names of the tests that failed."""

    script = ("import sys, contextlib, io\nimport test_model_mode\nfailed = []\n"
              f"for name in {NEW_TESTS!r}:\n"
              "    try:\n"
              "        with contextlib.redirect_stdout(io.StringIO()):\n"
              "            getattr(test_model_mode, name)()\n"
              "    except BaseException as error:\n"
              "        failed.append(name)\n"
              "print('FAILED:' + ','.join(failed))\n")
    try:
        finished = subprocess.run([sys.executable, "-c", script], cwd = str(HERE), capture_output = True, text = True, env = dict(os.environ, PYTHONPATH = str(CODE)), timeout = 240)
    except subprocess.TimeoutExpired:
        return ["(the tests did not end within 240 s)"]
    lines = [line for line in finished.stdout.splitlines() if line.startswith("FAILED:")]
    if not lines:
        return ["(the test run broke: " + (finished.stderr.strip().splitlines() or ["?"])[-1][:120] + ")"]
    return [name for name in lines[-1][len("FAILED:"):].split(",") if name]


only = sys.argv[1:]
(HERE / "model_mode.py").write_text(source, encoding = "utf-8")
print("unbroken copy, failing tests:", run_tests(), flush = True)
for title, old, new in MUTATIONS:
    if only and not any(word in title for word in only):
        continue
    if source.count(old) != 1:
        print(f"SKIPPED (text found {source.count(old)} times): {title}", flush = True)
        continue
    (HERE / "model_mode.py").write_text(source.replace(old, new), encoding = "utf-8")
    failed = run_tests()
    print(f"{'caught' if failed else 'NOT CAUGHT'}: {title}" + (f" -> {[name[5:45] for name in failed]}" if failed else ""), flush = True)
(HERE / "model_mode.py").write_text(source, encoding = "utf-8")
