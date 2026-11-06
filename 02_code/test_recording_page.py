"""
Tests of recording_page.py; no robot is used and the recording tool is not started.

Run from 02_code:
    python test_recording_page.py
"""

import http.client
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from config import RECORDING_PAGE_PARAMETERS
from recording_page import (FormError, ToolRunner, build_recording_command, done_plan_numbers, expand_episode_plan, make_server,
                            read_review_form)


TEST_PORT = 8766


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
    assert "--single-episode" in command


def test_notes_and_flags() -> None:

    command = build_recording_command(good_form(start_code = "NE", start_x = "0.31", start_y = "0.02", extra_note = "new lamp",
                                                 is_trial = True, is_new_session = True))
    assert option_value(command, "--orientation") == "plan 02-07, end A, NE, taught start, new lamp, trial", command
    assert option_value(command, "--start-x") == "0.31" and option_value(command, "--start-index") == "1"
    assert "--new-session" in command

    practice = build_recording_command(good_form(is_simulated = True, plan_number = "", lighting = "", safety_checks = []))
    assert "--simulated" in practice


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
    assert counts == {"path_01_straight_line": 20, "path_02_arc": 35, "path_03_two_curve": 45, "path_04_meander": 45,
                      "path_05_alpha_loop": 30, "path_06_alpha_loop_dashed": 25}, counts
    assert sum(counts.values()) == 200
    for entries in plan.values():
        for entry in entries.values():
            assert entry["start_code"] in RECORDING_PAGE_PARAMETERS["start_codes"] and entry["line_end"] in RECORDING_PAGE_PARAMETERS["line_ends"], entry

    def entry(path_id: str, number: str) -> tuple:
        return plan[path_id][number]["start_code"], plan[path_id][number]["line_end"]

    assert entry("path_01_straight_line", "01-07") == ("TL15", "A") and entry("path_01_straight_line", "01-14") == ("STD", "B")
    assert entry("path_02_arc", "02-08") == ("TR10", "A") and entry("path_02_arc", "02-10") == ("TL10", "B")
    assert entry("path_02_arc", "02-24") == ("TR15", "B") and entry("path_02_arc", "02-31") == ("STD", "B")
    assert entry("path_03_two_curve", "03-36") == ("TR15", "B") and entry("path_04_meander", "04-24") == ("OR1", "A")
    assert entry("path_05_alpha_loop", "05-26") == ("X2", "B") and entry("path_06_alpha_loop_dashed", "06-21") == ("TL15", "A")


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
    """A finished episode blocks the next Start until scored; the score is a new file, and episode.json stays as it was."""

    with tempfile.TemporaryDirectory() as folder:
        data_folder = Path(folder)
        raw_folder = data_folder / "03_data/raw"
        episode_folder = raw_folder / "2026-11-06_session_01_emranul/path_01_straight_line_episode_001"
        runner = ToolRunner(data_folder / "logs", data_folder)

        record_with_fake_tool(runner, episode_folder)
        waiting = runner.status()["episode_to_score"]
        assert waiting["episode"] == episode_folder.name and waiting["outcome"] == "puck_in_goal" and "folder" not in waiting, waiting
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
        review = json.loads((episode_folder / "review.json").read_text(encoding = "utf-8"))
        assert review["score"] == 2 and review["issues"] == ["off_line"] and review["note"] == "drifted near the end", review
        assert review["reviewer"] == "emranul" and (episode_folder / "episode.json").read_bytes() == original
        notes = (episode_folder.parent / "session_notes.md").read_text(encoding = "utf-8")
        assert "start" in notes and "score 2 (off_line, drifted near the end)" in notes, notes
        assert runner.status()["episode_to_score"] is None and done_plan_numbers(raw_folder) == ["01-01"]

        runner.start("Next tool", [sys.executable, "-c", "pass"])
        runner.process.wait(timeout = 10)


def test_bad_trial_and_discarded_episodes_stay_undone() -> None:
    """Score 1, a trial and a discard leave their plan numbers open; a discard asks for no score, as it already gave a reason."""

    with tempfile.TemporaryDirectory() as folder:
        data_folder = Path(folder)
        raw_folder = data_folder / "03_data/raw"
        session = raw_folder / "2026-11-06_session_01_emranul"
        runner = ToolRunner(data_folder / "logs", data_folder)

        record_with_fake_tool(runner, session / "path_01_straight_line_episode_001")
        runner.status()
        runner.save_review({"episode": "path_01_straight_line_episode_001", "score": "1", "issues": ["off_line"]})
        assert done_plan_numbers(raw_folder) == []

        record_with_fake_tool(runner, session / "path_01_straight_line_episode_002", plan_note = "plan 01-02, end A, STD, START MARK, trial")
        runner.status()
        runner.save_review({"episode": "path_01_straight_line_episode_002", "score": "3"})
        assert done_plan_numbers(raw_folder) == []

        record_with_fake_tool(runner, session / "path_01_straight_line_episode_003", status = "discarded")
        assert runner.status()["episode_to_score"] is None


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


if __name__ == "__main__":
    test_command_carries_the_form()
    test_notes_and_flags()
    test_refusals()
    test_form_text_stays_one_argument()
    test_only_one_tool_at_a_time()
    test_plan_matches_the_episode_plan_file()
    test_review_form()
    test_episode_is_scored_before_the_next_start()
    test_bad_trial_and_discarded_episodes_stay_undone()
    test_other_websites_cannot_start_the_robot()
    print("all tests passed")
