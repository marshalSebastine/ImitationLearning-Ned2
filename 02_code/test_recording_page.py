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
from recording_page import FormError, ToolRunner, build_recording_command, make_server


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
    test_other_websites_cannot_start_the_robot()
    print("all tests passed")
