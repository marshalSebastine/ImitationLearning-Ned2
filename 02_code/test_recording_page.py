"""
Tests of recording_page.py; no robot is used and the recording tool is not started.

Run from 02_code:
    python test_recording_page.py
"""

import datetime
import http.client
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from config import RECORDING_PAGE_PARAMETERS, RECORDING_PARAMETERS, SHAPE_OF_SHEET, main_checkout_folder
import recording_page
from recording_page import (FormError, ToolRunner, build_recording_command, default_operator, done_plan_numbers, expand_episode_plan,
                            make_server, read_review_form)


TEST_PORT = 8766
COURSE_SHEETS = ["path_01_straight_line", "path_02_arc", "path_03_two_curve", "path_04_meander", "path_05_alpha_loop", "path_06_alpha_loop_dashed"]


def good_form(**changes) -> dict:

    form = {"operator": "emranul", "mode": "gamepad", "path_id": "path_02_arc", "line_end": "A", "start_code": "OL1",
            "plan_number": "02-07", "lighting": "ceiling lamps", "start_x": "", "start_y": "", "extra_note": "",
            "is_trial": False, "is_new_session": False, "is_simulated": False,
            "safety_checks": list(range(len(RECORDING_PAGE_PARAMETERS["safety_checks"])))}
    form.update(changes)
    return form


def option_value(command: list, name: str) -> str:

    return command[command.index(name) + 1]


def assert_refused(form: dict, text: str) -> None:

    try:
        build_recording_command(form)
    except FormError as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError(f"accepted: {text}")


def test_command_carries_the_form() -> None:

    command = build_recording_command(good_form())
    assert option_value(command, "--operator") == "emranul" and option_value(command, "--mode") == "gamepad"
    assert option_value(command, "--path-id") == "path_02_arc" and option_value(command, "--lighting") == "ceiling lamps"
    assert option_value(command, "--orientation") == "plan 02-07, end A, OL1, START MARK", command
    assert option_value(command, "--offset-cm") == "1.0" and option_value(command, "--start-index") == "0"
    assert "--start-x" not in command and "--simulated" not in command and "--new-session" not in command
    assert option_value(command, "--plan-number") == "02-07", command
    assert "--single-episode" in command


def test_taught_start_angles_reach_the_tool() -> None:
    """The taught push direction and forearm turn (7.11.26) go to the tool with the taught point; one alone is refused."""

    command = build_recording_command(good_form(start_code = "NE", start_x = "0.25", start_y = "-0.05",
                                                start_push_deg = "-35.5", start_turn_deg = "40.2"))
    assert option_value(command, "--start-push-deg") == "-35.5" and option_value(command, "--start-turn-deg") == "40.2", command
    assert "--start-turn-deg" not in build_recording_command(good_form(start_code = "NE", start_x = "0.25", start_y = "-0.05"))
    assert_refused(good_form(start_code = "NE", start_x = "0.25", start_y = "-0.05", start_turn_deg = "40"), "both start angles")
    match = recording_page.TAUGHT_START_IN_LOG_PATTERN.search("with:  --start-x 0.2500 --start-y -0.0500 --start-push-deg -35.5 --start-turn-deg 40.2")
    assert match.groups() == ("0.2500", "-0.0500", "-35.5", "40.2"), match.groups()


def test_notes_and_flags() -> None:

    command = build_recording_command(good_form(start_code = "NE", start_x = "0.31", start_y = "0.02", extra_note = "new lamp",
                                                 is_trial = True, is_new_session = True))
    assert option_value(command, "--orientation") == "plan 02-07, end A, NE, taught start, new lamp, trial", command
    assert option_value(command, "--start-x") == "0.31" and option_value(command, "--start-index") == "1"
    assert "--new-session" in command and "--trial" in command
    assert "--plan-number" not in command, "a trial keeps the plain folder name"

    practice = build_recording_command(good_form(is_simulated = True, plan_number = "", lighting = "", safety_checks = []))
    assert "--simulated" in practice and "--trial" not in practice


def test_refusals() -> None:

    assert_refused(good_form(safety_checks = [0]), "safety check")
    assert_refused(good_form(lighting = " "), "Light in the room")
    assert_refused(good_form(plan_number = ""), "Plan number")
    assert_refused(good_form(plan_number = "abc"), "Plan number")
    assert_refused(good_form(start_code = "NE"), "taught Start")
    assert_refused(good_form(start_x = "0.2"), "both")
    assert_refused(good_form(start_x = "2", start_y = "0.1"), "look wrong")
    assert_refused(good_form(extra_note = "two\nlines"), "one line")
    assert_refused(good_form(mode = "keyboard"), "mode")
    assert_refused(good_form(operator = "someone"), "recording")


def test_form_text_stays_one_argument() -> None:
    """A hostile or unlucky note must stay inside its own option, because no shell reads the command."""

    note = "x; rm -rf / && --simulated"
    command = build_recording_command(good_form(extra_note = note))
    assert note in option_value(command, "--orientation") and "--simulated" not in command


def test_only_one_tool_at_a_time() -> None:

    with tempfile.TemporaryDirectory() as folder:
        runner = ToolRunner(Path(folder))
        runner.start("Slow tool", [sys.executable, "-c", "import time; print('hello', flush = True); time.sleep(1.5)"])
        try:
            runner.start("Second tool", [sys.executable, "-c", "pass"])
        except FormError as error:
            assert "Slow tool" in str(error), error
        else:
            raise AssertionError("two tools ran at once")
        assert runner.status()["is_running"]
        runner.process.wait(timeout = 10)
        status = runner.status()
        assert not status["is_running"] and status["exit_code"] == 0 and "hello" in status["log"], status
        runner.start("Second tool", [sys.executable, "-c", "pass"])
        runner.process.wait(timeout = 10)


def test_plan_matches_the_episode_plan_file() -> None:
    """Counts per sheet, and some entries read off 2026-11-06_episode_plan_per_sheet.md, with every code known to the form."""

    plan = expand_episode_plan()
    counts = {path_id: len(entries) for path_id, entries in plan.items()}
    course_counts = {path_id: count for path_id, count in counts.items() if path_id in COURSE_SHEETS}
    assert course_counts == {"path_01_straight_line": 20, "path_02_arc": 45, "path_03_two_curve": 45, "path_04_meander": 45,
                             "path_05_alpha_loop": 30, "path_06_alpha_loop_dashed": 25}, counts
    # 210 since 7.11.26: sheet 02 gained 10 landscape episodes (square sheets as at the exam); 58 more on the new tracks
    # since 12.11.26 (test_new_tracks_are_offered_sheet_by_sheet).
    assert sum(course_counts.values()) == 210 and sum(counts.values()) == 268, counts
    for entries in plan.values():
        for entry in entries.values():
            assert entry["start_code"] in RECORDING_PAGE_PARAMETERS["start_codes"] and entry["line_end"] in RECORDING_PAGE_PARAMETERS["line_ends"], entry

    def entry(path_id: str, number: str) -> tuple:
        return plan[path_id][number]["start_code"], plan[path_id][number]["line_end"]

    assert entry("path_01_straight_line", "01-07") == ("TL15", "A") and entry("path_01_straight_line", "01-14") == ("STD", "B")
    assert entry("path_02_arc", "02-08") == ("TR10", "A") and entry("path_02_arc", "02-10") == ("TL10", "B")
    assert entry("path_02_arc", "02-24") == ("TR15", "B") and entry("path_02_arc", "02-31") == ("STD", "B")
    assert entry("path_03_two_curve", "03-36") == ("TR15", "B") and entry("path_04_meander", "04-24") == ("OR1", "A")
    assert entry("path_05_alpha_loop", "05-26") == ("X2", "B") and entry("path_06_alpha_loop_dashed", "06-21") == ("TL15", "B")


def page_order(
        plan: dict,
        done: set,
        page_parameters: dict = RECORDING_PAGE_PARAMETERS
) -> list:
    """
    The plan numbers in the order the page offers them when every episode is recorded as offered: the rules of
    suggestNextPlanNumber and nextOpenNumber in recording_page.html, written again in Python (the page's JavaScript is
    not run here). The sheet on the table stays while it has open episodes; then the round's sheets with an open
    episode among their first "episodes_per_sheet"; then the first sheet in the plan's order with one open.
    """

    done = set(done)

    def next_open(path_id: str, last_index: int = None) -> str:
        return next((number for index, number in enumerate(plan[path_id]) if number not in done and (last_index is None or index < last_index)), None)

    order = []
    sheet = next(iter(plan))
    while True:
        if not next_open(sheet):
            round_settings = page_parameters["first_round"]
            round_sheet = next((path_id for path_id in round_settings["sheets"] if next_open(path_id, round_settings["episodes_per_sheet"])), None)
            sheet = round_sheet or next((path_id for path_id in plan if next_open(path_id)), None)
            if sheet is None:
                return order
        number = next_open(sheet)
        order.append((sheet, number))
        done.add(number)


def test_new_tracks_are_offered_sheet_by_sheet() -> None:
    """
    The 58 episodes of the new tracks (Emranul, 12.11.26): every print taped once, its episodes in a row, the near-end
    start last; the pilot (T01 to T04 solid, plans 1 Emranul and 2 Marshal) first, then the dashed pilot prints, then T05
    solid to T16 dashed. Sheet numbers stay two digits and unique, so plan numbers and set-up pictures cannot collide.
    """

    plan = expand_episode_plan()
    course_done = {number for path_id in COURSE_SHEETS for number in plan[path_id]}
    order = page_order(plan, course_done)
    assert len(order) == 58 and len({number for _, number in order}) == 58, len(order)
    kinds = [plan[sheet][number]["start_code"] for sheet, number in order]
    assert kinds.count("STD") == 48 and kinds.count("NE") == 10, kinds

    taped = [sheet for index, (sheet, _) in enumerate(order) if index == 0 or order[index - 1][0] != sheet]
    expected = [f"path_{10 + track:02d}_T{track:02d}_solid" for track in range(1, 5)] + [f"path_{30 + track:02d}_T{track:02d}_dashed" for track in range(1, 5)]
    for track in range(5, 17):
        expected += [f"path_{10 + track:02d}_T{track:02d}_solid", f"path_{30 + track:02d}_T{track:02d}_dashed"]
    assert taped == expected, taped
    for sheet in taped:
        codes = [plan[sheet][number]["start_code"] for number in plan[sheet]]
        assert "NE" not in codes[:-1], (sheet, codes)
        assert all(plan[sheet][number]["line_end"] == "A" for number in plan[sheet]), sheet
    near_end_sheets = {sheet for sheet in taped if "NE" in [plan[sheet][number]["start_code"] for number in plan[sheet]]}
    assert near_end_sheets == {f"path_{10 + track:02d}_T{track:02d}_solid" for track in (6, 8, 10, 12, 14)} | \
        {f"path_{30 + track:02d}_T{track:02d}_dashed" for track in (5, 7, 9, 11, 13)}, near_end_sheets
    for track in range(1, 5):
        sheet = f"path_{10 + track:02d}_T{track:02d}_solid"
        notes = [plan[sheet][number]["note"] for number in plan[sheet]]
        assert len(notes) == 2 and "Emranul" in notes[0] and "Marshal" in notes[1], notes

    numbers = [sheet["number"] for sheet in RECORDING_PAGE_PARAMETERS["sheets"].values()]
    assert len(set(numbers)) == len(numbers) and all(len(number) == 2 for number in numbers), numbers
    for sheet, number in order:
        assert recording_page.PLAN_NUMBER_PATTERN.match(number) and recording_page.SETUP_PICTURE_PATH_PATTERN.fullmatch(f"/setup-picture/{number}.png")
        assert SHAPE_OF_SHEET[sheet] == sheet.split("_")[2] and RECORDING_PARAMETERS["line_length_cm"][sheet] > 20, sheet
    command = build_recording_command(good_form(path_id = "path_15_T05_solid", start_code = "STD", plan_number = "15-01"))
    assert option_value(command, "--path-id") == "path_15_T05_solid" and option_value(command, "--plan-number") == "15-01", command


def test_recordings_go_to_the_main_checkout() -> None:
    """Run from a git worktree (no 03_data), the page's logs and raw folders are the main checkout's (12.11.26)."""

    assert recording_page.DATA_FOLDER == main_checkout_folder() and (recording_page.DATA_FOLDER / "03_data").is_dir()
    runner = ToolRunner()
    assert runner.data_folder == recording_page.DATA_FOLDER and runner.log_folder.parent == recording_page.DATA_FOLDER / "03_data"


def test_every_episode_has_its_setup_steps() -> None:

    settings = RECORDING_PAGE_PARAMETERS
    assert set(settings["start_setups"]) == set(settings["start_codes"]), "a kind of start has no setup steps"
    assert set(settings["sheet_setups"]) == set(settings["sheets"]) == set(settings["episode_plan"]), "a sheet has no setup steps"
    assert (recording_page.PROJECT_FOLDER / settings["sheet_placement_figure"]).exists()
    assert set(settings["first_round"]["sheets"]) <= set(settings["sheets"])


def test_review_form() -> None:

    assert read_review_form({"score": "3"}) == {"score": 3, "issues": [], "note": ""}
    assert read_review_form({"score": "1", "issues": ["off_line"]})["score"] == 1
    for form, text in (({}, "Choose a score"), ({"score": "4"}, "Choose a score"), ({"score": "2"}, "at least one"),
                       ({"score": "2", "issues": ["nonsense"]}, "unknown issue"), ({"score": "1", "issues": ["other"]}, "needs a note")):
        try:
            read_review_form(form)
        except FormError as error:
            assert text in str(error), (text, str(error))
        else:
            raise AssertionError(f"accepted: {form}")


def fake_tool_writing(folder: Path, status: str = "completed", plan_note: str = "plan 01-01, end A, STD, START MARK") -> str:
    """A pretend recording tool: it writes one episode and ends, as record_episodes.py --single-episode does."""

    return f"""
import json, pathlib
folder = pathlib.Path({str(folder)!r})
folder.mkdir(parents = True, exist_ok = True)
(folder.parent / "session_notes.md").write_text("start\\n", encoding = "utf-8")
(folder / "episode.json").write_text(json.dumps({{"status": {status!r}, "operator": "emranul", "mode": "gamepad", "outcome": "puck_in_goal",
    "steps": 30, "mistake_steps": 0, "lost_steps": 0, "sheet_orientation": {plan_note!r}}}), encoding = "utf-8")
"""


def record_with_fake_tool(runner: ToolRunner, episode_folder: Path, **episode) -> None:

    runner.start_recording("Recording", [sys.executable, "-c", fake_tool_writing(episode_folder, **episode)], "path_01_straight_line", False)
    runner.process.wait(timeout = 10)
    time.sleep(0.01)


def test_episode_is_scored_before_the_next_start() -> None:
    """A finished episode blocks the next Start until scored; the score is a new file, the folder name gains the score."""

    with tempfile.TemporaryDirectory() as folder:
        data_folder = Path(folder)
        raw_folder = data_folder / RECORDING_PARAMETERS["raw_folder"]
        episode_folder = raw_folder / "2026-11-06_session_01_emranul/path_01_straight_line_episode_001"
        runner = ToolRunner(data_folder / "logs", data_folder)

        record_with_fake_tool(runner, episode_folder)
        waiting = runner.status()["episode_to_score"]
        assert waiting["episode"] == episode_folder.name and waiting["outcome"] == "puck_in_goal" and "folder" not in waiting, waiting
        # The picture check runs in the background while the episode is scored (Emranul, 7.11.26).
        for _ in range(100):
            check = runner.status()["picture_check"]
            if check and check["is_done"]:
                break
            time.sleep(0.1)
        assert check["episode"] == episode_folder.name and check["flagged_count"] == 0 and (episode_folder / "picture_check.json").exists(), check
        assert done_plan_numbers(raw_folder) == []
        original = (episode_folder / "episode.json").read_bytes()

        try:
            runner.start("Second tool", [sys.executable, "-c", "pass"])
        except FormError as error:
            assert "Score" in str(error), error
        else:
            raise AssertionError("started the next tool before the score")

        for wrong_form in ({"episode": "other", "score": "3"}, {"episode": episode_folder.name, "score": "2"}):
            try:
                runner.save_review(wrong_form)
            except FormError:
                pass
            else:
                raise AssertionError(f"accepted: {wrong_form}")
        assert not (episode_folder / "review.json").exists()

        runner.save_review({"episode": episode_folder.name, "score": "2", "issues": ["off_line"], "note": "drifted near the end"})
        assert not episode_folder.exists()
        episode_folder = episode_folder.with_name("path_01_straight_line_episode_001_score_02")
        review = json.loads((episode_folder / "review.json").read_text(encoding = "utf-8"))
        assert review["episode"] == episode_folder.name, review
        assert review["score"] == 2 and review["issues"] == ["off_line"] and review["note"] == "drifted near the end", review
        assert review["reviewer"] == "emranul" and (episode_folder / "episode.json").read_bytes() == original
        notes = (episode_folder.parent / "session_notes.md").read_text(encoding = "utf-8")
        assert "start" in notes and "score 2 (off_line, drifted near the end)" in notes, notes
        assert runner.status()["episode_to_score"] is None and done_plan_numbers(raw_folder) == ["01-01"]

        runner.start("Next tool", [sys.executable, "-c", "pass"])
        runner.process.wait(timeout = 10)


def test_bad_trial_and_discarded_episodes_stay_undone() -> None:
    """Score 1 counts as done (6.11.26); a trial and a discard leave their plan numbers open; a discard asks for no score."""

    with tempfile.TemporaryDirectory() as folder:
        data_folder = Path(folder)
        raw_folder = data_folder / RECORDING_PARAMETERS["raw_folder"]
        session = raw_folder / "2026-11-06_session_01_emranul"
        runner = ToolRunner(data_folder / "logs", data_folder)

        record_with_fake_tool(runner, session / "path_01_straight_line_episode_001")
        runner.status()
        runner.save_review({"episode": "path_01_straight_line_episode_001", "score": "1", "issues": ["off_line"]})
        assert done_plan_numbers(raw_folder) == ["01-01"]

        record_with_fake_tool(runner, session / "path_01_straight_line_episode_002", plan_note = "plan 01-02, end A, STD, START MARK, trial")
        runner.status()
        runner.save_review({"episode": "path_01_straight_line_episode_002", "score": "3"})
        assert done_plan_numbers(raw_folder) == ["01-01"]

        record_with_fake_tool(runner, session / "path_01_straight_line_episode_003", status = "discarded")
        assert runner.status()["episode_to_score"] is None
        assert (session / "path_01_straight_line_episode_003_score_00").is_dir(), "a failed run is named score 00"


def test_trial_is_found_with_the_tool_tests() -> None:
    """A trial is saved in the tool-test folder, not the main data, and is still offered for a score."""

    with tempfile.TemporaryDirectory() as folder:
        data_folder = Path(folder)
        episode_folder = data_folder / RECORDING_PARAMETERS["tool_test_folder"] / "2026-11-06_session_01_emranul/path_01_straight_line_episode_001"
        runner = ToolRunner(data_folder / "logs", data_folder)

        runner.start_recording("Recording", [sys.executable, "-c", fake_tool_writing(episode_folder)], "path_01_straight_line", False,
                               is_trial = True)
        runner.process.wait(timeout = 10)
        time.sleep(0.01)
        waiting = runner.status()["episode_to_score"]
        assert waiting is not None and waiting["episode"] == episode_folder.name, waiting


def test_operator_is_guessed_from_the_user_name() -> None:

    original = recording_page.getpass.getuser
    try:
        for user_name, expected in (("marshalsebastine", "marshal"), ("Emranul Hoque", "emranul"), ("emran", "emranul"), ("guest", "emranul")):
            recording_page.getpass.getuser = lambda name = user_name: name
            assert default_operator() == expected, (user_name, expected)
    finally:
        recording_page.getpass.getuser = original


def test_robot_check_result_is_read_from_its_output() -> None:
    """Calibrated or not comes from the check's own words; a check that says neither ticks nothing."""

    with tempfile.TemporaryDirectory() as folder:
        runner = ToolRunner(Path(folder))
        assert runner.status()["robot_check"] is None
        for printed, expected in (("Calibration needed: False", True), ("Calibration needed: True", False), ("could not connect", None)):
            runner.start("Check robot", [sys.executable, "-c", f"print({printed!r})"])
            runner.process.wait(timeout = 10)
            assert runner.status()["robot_check"]["is_calibrated"] is expected, printed

        runner.start("Other tool", [sys.executable, "-c", "print('Calibration needed: False')"])
        runner.process.wait(timeout = 10)
        assert runner.status()["robot_check"]["is_calibrated"] is None, "another tool's output counted as the robot check"

        runner.robot_check["checked_at"] -= datetime.timedelta(minutes = RECORDING_PAGE_PARAMETERS["robot_check_valid_minutes"] + 1)
        assert runner.status()["robot_check"] is None, "an old check still counted"


def request(
        method: str,
        path: str,
        headers: dict,
        body: str = None
) -> tuple:

    connection = http.client.HTTPConnection("127.0.0.1", TEST_PORT, timeout = 5)
    connection.request(method, path, body = body, headers = headers)
    response = connection.getresponse()
    return response.status, response.read()


def test_other_websites_cannot_start_the_robot() -> None:
    """A page on another site, or a request naming another host, is refused before anything is read."""

    with tempfile.TemporaryDirectory() as folder:
        runner = ToolRunner(Path(folder))
        server = make_server(port = TEST_PORT, runner = runner)
        threading.Thread(target = server.serve_forever, daemon = True).start()
        try:
            own_host = {"Host": f"127.0.0.1:{TEST_PORT}"}
            as_json = {"Content-Type": "application/json"}
            status, page = request("GET", "/", own_host)
            assert status == 200 and b"Start Recording" in page

            body = json.dumps(good_form(is_simulated = True, plan_number = "", lighting = ""))
            assert request("POST", "/start", {**own_host, **as_json, "Origin": "https://example.com"}, body)[0] == 403
            assert request("POST", "/start", {"Host": "example.com", **as_json}, body)[0] == 403
            assert request("POST", "/start", {**own_host, "Content-Type": "text/plain"}, body)[0] == 415
            assert runner.process is None

            status, reply = request("POST", "/start", {**own_host, **as_json}, json.dumps(good_form(safety_checks = [])))
            assert status == 400 and b"safety check" in reply and runner.process is None
        finally:
            server.shutdown()
            server.server_close()
            time.sleep(0.05)


def test_sharing_on_the_network_is_off_by_default() -> None:
    """Off: this computer only. On: also this laptop's network address, and still no other host's name."""

    with tempfile.TemporaryDirectory() as folder:
        assert RECORDING_PAGE_PARAMETERS["is_shared_on_network"] is False
        server = make_server(port = TEST_PORT, runner = ToolRunner(Path(folder)))
        try:
            assert server.server_address[0] == "127.0.0.1"
            assert server.RequestHandlerClass.allowed_hosts == (f"127.0.0.1:{TEST_PORT}", f"localhost:{TEST_PORT}")
        finally:
            server.server_close()
        server = make_server(port = TEST_PORT, runner = ToolRunner(Path(folder)), is_shared_on_network = True)
        try:
            allowed_hosts = server.RequestHandlerClass.allowed_hosts
            assert server.server_address[0] == "0.0.0.0"
            assert all(f"{address}:{TEST_PORT}" in allowed_hosts for address in recording_page.network_addresses())
            assert "example.com" not in " ".join(allowed_hosts)
        finally:
            server.server_close()
            time.sleep(0.05)


def test_teach_start_is_answered_from_the_page() -> None:
    """The page's Enter button answers the teach prompts, and the printed point is offered for Start x and Start y."""

    with tempfile.TemporaryDirectory() as folder:
        runner = ToolRunner(Path(folder) / "logs", Path(folder))
        fake_teach = ("input('motors off? '); input('touch the point, Enter '); input('motors on? '); "
                      "print('Start the recording tool with:  --start-x 0.3421 --start-y -0.0163')")
        runner.start("Teach start", [sys.executable, "-u", "-c", fake_teach], is_answered_from_page = True)
        for _ in range(3):
            runner.press_enter()
        runner.process.wait(timeout = 10)
        taught = runner.taught_start()
        assert taught is not None and taught["x"] == "0.3421" and taught["y"] == "-0.0163", taught

        runner.start("Check robot", [sys.executable, "-c", "pass"])
        try:
            runner.press_enter()
        except FormError:
            pass
        else:
            raise AssertionError("pressed Enter in a tool that does not read it")
        runner.process.wait(timeout = 10)


def test_blurred_pictures_raise_a_warning() -> None:
    """A median sharpness under the limit in the last pictures gives a warning; sharp pictures give none."""

    with tempfile.TemporaryDirectory() as folder:
        data_folder = Path(folder)
        raw_folder = data_folder / RECORDING_PARAMETERS["raw_folder"]
        runner = ToolRunner(data_folder / "logs", data_folder)
        limit = RECORDING_PAGE_PARAMETERS["sharpness_warning_below"]
        for index, sharpness in ((1, limit / 3), (2, limit * 2)):
            episode_folder = raw_folder / f"2026-11-06_session_01_emranul/path_01_straight_line_episode_00{index}"
            rows = "\n".join(f"{step},{sharpness}" for step in range(8))
            writer = (f"import pathlib; folder = pathlib.Path({str(episode_folder)!r}); folder.mkdir(parents = True); "
                      f"(folder / 'steps_in_progress.csv').write_text('step,sharpness\\n' + {rows!r})")
            runner.start_recording("Recording", [sys.executable, "-c", writer], "path_01_straight_line", False)
            runner.process.wait(timeout = 10)
            warning = runner.sharpness_warning()
            if index == 1:
                assert warning is not None and warning["episode"] == episode_folder.name and warning["sharpness"] == round(limit / 3), warning
            else:
                assert warning is None, warning


def slow_picture_check(seconds: float, is_failing: bool = False):
    """A stand-in for check_pictures.check_episode_pictures that takes ``seconds`` and writes into the folder, or fails."""

    def check(episode_folder: Path) -> dict:
        episode_folder = Path(episode_folder)
        with open(episode_folder / "picture_check.json", "w", encoding = "utf-8") as file:
            # The file stays open while the check runs: a rename of the folder now would fail on Windows.
            time.sleep(seconds)
            if is_failing:
                raise RuntimeError("stand-in check failed")
            file.write(json.dumps({"episode": episode_folder.name, "total": 3, "flagged_count": 0, "flagged": []}))
        return {"episode": episode_folder.name, "total": 3, "flagged_count": 0, "flagged": []}

    return check


def wait_until(condition, timeout_s: float = 20.0) -> bool:

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_saving_a_score_does_not_wait_for_the_picture_check() -> None:
    """
    12.11.26: Save score returns at once while a slow picture check still writes into the folder; the folder gains its
    score when the check has ended, with review.json, the session notes and the check's file as before. A failing
    check renames too; and the next episode can start before the last check ended, each folder keeping its own score.
    """

    original_check = recording_page.check_episode_pictures
    try:
        with tempfile.TemporaryDirectory() as folder:
            data_folder = Path(folder)
            session = data_folder / RECORDING_PARAMETERS["raw_folder"] / "2026-11-06_session_01_emranul"
            runner = ToolRunner(data_folder / "logs", data_folder)

            # A slow check: the save returns at once, the rename follows the check.
            recording_page.check_episode_pictures = slow_picture_check(1.5)
            first = session / "path_01_straight_line_episode_001"
            record_with_fake_tool(runner, first)
            assert runner.status()["picture_check"]["is_done"] is False
            started = time.monotonic()
            runner.save_review({"episode": first.name, "score": "3"})
            assert time.monotonic() - started < 0.5, time.monotonic() - started
            assert first.exists() and (first / "review.json").exists(), "renamed while the check still wrote into it"
            assert runner.status()["episode_to_score"] is None and runner.status()["done_plan_numbers"] == ["01-01"]

            # The next episode starts before that check has ended; its check fails.
            recording_page.check_episode_pictures = slow_picture_check(0.5, is_failing = True)
            second = session / "path_01_straight_line_episode_002"
            record_with_fake_tool(runner, second, plan_note = "plan 01-02, end A, STD, START MARK")
            assert runner.status()["episode_to_score"]["episode"] == second.name
            assert wait_until(lambda: runner.status()["picture_check"]["is_done"]), "the failing check never ended"
            assert "stand-in check failed" in runner.status()["picture_check"]["error"]
            runner.save_review({"episode": second.name, "score": "2", "issues": ["off_line"]})

            assert wait_until(lambda: not first.exists() and not second.exists()), [path.name for path in session.iterdir()]
            renamed_first = first.with_name(first.name + "_score_03")
            renamed_second = second.with_name(second.name + "_score_02")
            assert (renamed_first / "picture_check.json").exists() and (renamed_second / "review.json").exists()
            assert json.loads((renamed_first / "review.json").read_text(encoding = "utf-8"))["episode"] == renamed_first.name
            assert json.loads((renamed_second / "review.json").read_text(encoding = "utf-8"))["score"] == 2
            notes = (session / "session_notes.md").read_text(encoding = "utf-8")
            # (The pretend tool starts the notes afresh, so the first score line is gone; the late rename is written.)
            assert f"renamed to {renamed_first.name}" in notes, notes
            assert f"renamed to {renamed_second.name}" in notes, notes
            assert runner.status()["done_plan_numbers"] == ["01-01", "01-02"]
            runner.finish_picture_checks(timeout_s = 5)
    finally:
        recording_page.check_episode_pictures = original_check


def test_episode_is_offered_once_written() -> None:
    """
    12.11.26: the score form comes as soon as the tool has written episode.json and its line in the session notes, while
    the tool still parks the arm; an episode.json without its notes line waits for the tool to end.
    """

    with tempfile.TemporaryDirectory() as folder:
        data_folder = Path(folder)
        session = data_folder / RECORDING_PARAMETERS["raw_folder"] / "2026-11-06_session_01_emranul"
        episode_folder = session / "path_01_straight_line_episode_001"
        runner = ToolRunner(data_folder / "logs", data_folder)
        writer = fake_tool_writing(episode_folder) + "\nimport time\ntime.sleep(2.5)\n"
        notes_line = (f"\nwith open(folder.parent / 'session_notes.md', 'a', encoding = 'utf-8') as notes:\n"
                      f"    notes.write('- {episode_folder.name}: completed, puck_in_goal, 30 steps\\n')\n")

        runner.start_recording("Recording", [sys.executable, "-c", writer], "path_01_straight_line", False)
        assert wait_until(lambda: (episode_folder / "episode.json").exists(), 10)
        time.sleep(0.2)
        assert runner.is_running() and runner.status()["episode_to_score"] is None, "offered without its notes line"
        runner.process.wait(timeout = 10)
        assert runner.status()["episode_to_score"]["episode"] == episode_folder.name, "not offered after the tool ended"
        runner.save_review({"episode": episode_folder.name, "score": "3"})
        runner.finish_picture_checks(timeout_s = 10)

        second = session / "path_01_straight_line_episode_002"
        writer = fake_tool_writing(second, plan_note = "plan 01-02, end A, STD, START MARK").replace(str(episode_folder), str(second))
        writer = writer + notes_line.replace(episode_folder.name, second.name) + "\nimport time\ntime.sleep(2.5)\n"
        runner.start_recording("Recording", [sys.executable, "-c", writer], "path_01_straight_line", False)
        assert wait_until(lambda: runner.status()["episode_to_score"] is not None, 10), "not offered while the tool still ran"
        assert runner.is_running() and runner.status()["episode_to_score"]["episode"] == second.name
        runner.save_review({"episode": second.name, "score": "2", "note": "early"})
        runner.process.wait(timeout = 10)
        runner.finish_picture_checks(timeout_s = 10)
        assert (session / (second.name + "_score_02") / "review.json").exists() and runner.status()["episode_to_score"] is None


def test_status_never_waits_for_git_and_reads_little() -> None:
    """12.11.26: git runs in a background thread; the log is read from its end; the done plan numbers follow new scores."""

    original = recording_page.read_code_state
    saved_cache = dict(recording_page.CODE_STATE_CACHE)
    try:
        recording_page.read_code_state = lambda: (time.sleep(1.5), {"branch": "b", "commit": "c", "has_uncommitted_code": True})[1]
        recording_page.CODE_STATE_CACHE.update({"read_at": None, "state": None, "thread": None})
        started = time.monotonic()
        first = recording_page.code_state()
        assert time.monotonic() - started < 0.2 and first["is_reading"] and first["commit"] is None, first
        assert wait_until(lambda: recording_page.code_state().get("commit") == "c", 5)
    finally:
        recording_page.read_code_state = original
        recording_page.CODE_STATE_CACHE.update(saved_cache)

    with tempfile.TemporaryDirectory() as folder:
        log_path = Path(folder) / "tool.log"
        lines = [f"line {index} " + "x" * 80 for index in range(5_000)]
        log_path.write_bytes("\r\n".join(lines).encode("utf-8") + b"\r\n")
        with open(log_path, encoding = "utf-8") as log_file:
            expected = "".join(log_file.readlines()[-200:])
        assert recording_page.read_log_tail(log_path, 200) == expected
        log_path.write_text("short\nlog\n", encoding = "utf-8")
        assert recording_page.read_log_tail(log_path, 200) == "short\nlog\n"


if __name__ == "__main__":
    test_command_carries_the_form()
    test_notes_and_flags()
    test_taught_start_angles_reach_the_tool()
    test_refusals()
    test_form_text_stays_one_argument()
    test_only_one_tool_at_a_time()
    test_plan_matches_the_episode_plan_file()
    test_new_tracks_are_offered_sheet_by_sheet()
    test_recordings_go_to_the_main_checkout()
    test_every_episode_has_its_setup_steps()
    test_review_form()
    test_operator_is_guessed_from_the_user_name()
    test_robot_check_result_is_read_from_its_output()
    test_episode_is_scored_before_the_next_start()
    test_bad_trial_and_discarded_episodes_stay_undone()
    test_trial_is_found_with_the_tool_tests()
    test_teach_start_is_answered_from_the_page()
    test_blurred_pictures_raise_a_warning()
    test_other_websites_cannot_start_the_robot()
    test_sharing_on_the_network_is_off_by_default()
    test_saving_a_score_does_not_wait_for_the_picture_check()
    test_episode_is_offered_once_written()
    test_status_never_waits_for_git_and_reads_little()
    print("all tests passed")
