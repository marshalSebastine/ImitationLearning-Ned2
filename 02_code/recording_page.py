"""
A local web page that starts the recording tool: fill in the form, tick the safety checks, press Start.

The page only builds the command line and starts record_episodes.py, which then moves the robot and opens its picture and
orientation windows as it always does. It has no stop button on purpose: a running tool is stopped from its own windows or
with the robot's Stop button, so an episode is never killed by accident. Only one tool can be started at a time.

Run it from anywhere, then use the page it opens (http://127.0.0.1:8765):
    python recording_page.py
Keep this program running during a session: the page's status and log come from it.
"""

import argparse
import datetime
import json
import os
import re
import shlex
import socketserver
import subprocess
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from config import RECORDING_PAGE_PARAMETERS, RECORDING_PARAMETERS


CODE_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = CODE_FOLDER.parent
PLAN_NUMBER_PATTERN = re.compile(r"^\d{2}-\d{1,3}$")
LONGEST_NOTE = 200


class FormError(ValueError):
    """A form value that cannot be used; the message is shown to the person."""



def clean_note(
        form: Dict,
        name: str,
        label: str,
        is_required: bool = False
) -> str:
    """A free-text field with its spaces trimmed; refused when required and empty, too long, or holding a line break."""

    value = str(form.get(name) or "").strip()
    if is_required and not value:
        raise FormError(f"{label} is required.")
    if len(value) > LONGEST_NOTE or "\n" in value or "\r" in value:
        raise FormError(f"{label} must be one line of at most {LONGEST_NOTE} characters.")
    return value


def read_start_position(form: Dict) -> Optional[List[float]]:
    """The taught start [x, y] in metres, None when both boxes are empty; refused when only one is filled or out of range."""

    texts = [str(form.get(name) or "").strip() for name in ("start_x", "start_y")]
    if not any(texts):
        return None
    if not all(texts):
        raise FormError("Fill in both Start x and Start y, or neither.")
    try:
        position = [float(text) for text in texts]
    except ValueError as error:
        raise FormError("Start x and Start y must be numbers in metres, for example 0.1707.") from error
    if not (0.05 <= position[0] <= 0.60 and -0.40 <= position[1] <= 0.40):
        raise FormError("Start x and Start y look wrong: they are metres from the robot's base, for example 0.1707 and 0.0374.")
    return position


def build_recording_command(
        form: Dict,
        page_parameters: Dict = RECORDING_PAGE_PARAMETERS
) -> List[str]:
    """
    Check the form and build the record_episodes.py command, as a list so that no shell ever reads the form's text.

    Parameters
    ----------
    form : dict
        The fields sent by the page.
    page_parameters : dict, optional
        The choices the form offers.

    Returns
    -------
    list of str
        The command, starting with this Python.

    Raises
    ------
    FormError
        When a field cannot be used; nothing has been started then.
    """

    operator = form.get("operator")
    if operator not in page_parameters["operators"]:
        raise FormError("Choose who is recording.")
    mode = form.get("mode")
    if mode not in page_parameters["modes"]:
        raise FormError("Choose a recording mode.")
    path_id = form.get("path_id")
    if path_id not in page_parameters["sheets"]:
        raise FormError("Choose a sheet.")
    line_end = form.get("line_end")
    if line_end not in page_parameters["line_ends"]:
        raise FormError("Choose which end of the line is the start.")
    start_code = form.get("start_code")
    if start_code not in page_parameters["start_codes"]:
        raise FormError("Choose a kind of start.")

    is_simulated = form.get("is_simulated") is True
    is_trial = form.get("is_trial") is True
    if not is_simulated:
        ticked = form.get("safety_checks")
        if not isinstance(ticked, list) or sorted(ticked) != list(range(len(page_parameters["safety_checks"]))):
            raise FormError("Tick every safety check first.")

    plan_number = clean_note(form, "plan_number", "Plan number")
    if plan_number and not PLAN_NUMBER_PATTERN.match(plan_number):
        raise FormError("Plan number looks like 02-07: the sheet, a dash, the number in the episode plan.")
    if not plan_number and not (is_trial or is_simulated):
        raise FormError("Plan number is required, unless this is a trial or a practice run.")
    lighting = clean_note(form, "lighting", "Light in the room", is_required = not is_simulated)
    extra_note = clean_note(form, "extra_note", "Extra note")

    start = page_parameters["start_codes"][start_code]
    position = read_start_position(form)
    if start["needs_taught_start"] and position is None:
        raise FormError(f"{start_code} needs its taught Start x and Start y (teach_sheet_start.py).")

    note_parts = [f"plan {plan_number}"] if plan_number else []
    note_parts += [f"end {line_end}", start_code, "taught start" if position else "START MARK"]
    note_parts += [extra_note] if extra_note else []
    note_parts += ["trial"] if is_trial else []

    command = [sys.executable, "-u", str(CODE_FOLDER / "record_episodes.py"),
               "--operator", operator, "--mode", mode, "--path-id", path_id,
               "--orientation", ", ".join(note_parts), "--lighting", lighting,
               "--start-index", str(start["start_index"]),
               "--offset-cm", str(start["offset_cm"]), "--offset-deg", str(start["offset_deg"])]
    if position:
        command += ["--start-x", str(position[0]), "--start-y", str(position[1])]
    if form.get("is_new_session") is True:
        command.append("--new-session")
    if is_simulated:
        command.append("--simulated")
    return command



class ToolRunner:

    def __init__(self, log_folder: Optional[Path] = None):
        """
        Start one tool at a time and keep its output in a log file.

        Parameters
        ----------
        log_folder : Path, optional
            Where the logs go, by default "log_folder" in RECORDING_PAGE_PARAMETERS.
        """

        self.log_folder = log_folder or PROJECT_FOLDER / RECORDING_PAGE_PARAMETERS["log_folder"]
        self.process = None
        self.log_path = None
        self.title = None
        self.command_text = None
        self.started_at = None


    def is_running(self) -> bool:

        return self.process is not None and self.process.poll() is None


    def start(
            self,
            title: str,
            command: List[str]
    ) -> None:
        """
        Start ``command``; refused while another tool from this page runs, because only one program may use the robot.

        The tool gets its own process group, so Ctrl-C in this program's terminal does not reach it, and an episode
        survives this program being closed.
        """

        if self.is_running():
            raise FormError(f"{self.title} is still running. Finish it from its own window first.")

        self.log_folder.mkdir(parents = True, exist_ok = True)
        self.started_at = datetime.datetime.now()
        self.log_path = self.log_folder / f"{self.started_at:%Y-%m-%d_%H%M%S}_{re.sub(r'[^a-z0-9]+', '_', title.lower())}.log"
        self.title = title
        self.command_text = shlex.join(command)

        isolation = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
                     else {"start_new_session": True})
        with open(self.log_path, "w", encoding = "utf-8") as log_file:
            log_file.write(f"{self.command_text}\n\n")
            log_file.flush()
            self.process = subprocess.Popen(command, cwd = CODE_FOLDER, stdin = subprocess.DEVNULL, stdout = log_file,
                                            stderr = subprocess.STDOUT, env = self.__tool_environment(), **isolation)


    @staticmethod
    def __tool_environment() -> Dict:

        return {**os.environ, "PYTHONIOENCODING": "utf-8"}


    def status(self, log_lines: int = RECORDING_PAGE_PARAMETERS["log_lines_shown"]) -> Dict:
        """What the page shows: whether a tool runs, how it ended, and the end of its log."""

        log_text = ""
        if self.log_path and self.log_path.exists():
            with open(self.log_path, encoding = "utf-8", errors = "replace") as log_file:
                log_text = "".join(log_file.readlines()[-log_lines:])
        return {
            "is_running": self.is_running(),
            "title": self.title,
            "command": self.command_text,
            "exit_code": None if self.process is None else self.process.poll(),
            "started_at": self.started_at.strftime("%H:%M:%S") if self.started_at else None,
            "log_path": str(self.log_path) if self.log_path else None,
            "log": log_text
        }



def run_git(*arguments: str) -> str:

    try:
        result = subprocess.run(["git", *arguments], cwd = PROJECT_FOLDER, capture_output = True, text = True, timeout = 5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def code_state() -> Dict:
    """The code version every episode will record, and whether it has uncommitted changes."""

    changed_files = run_git("status", "--porcelain", "--", "02_code")
    return {"branch": run_git("rev-parse", "--abbrev-ref", "HEAD"), "commit": run_git("rev-parse", "--short", "HEAD"),
            "has_uncommitted_code": bool(changed_files)}


def latest_session_notes(line_count: int = 12) -> Dict:
    """The end of the newest real recording session's notes, so the page shows what was just recorded."""

    sessions = sorted((PROJECT_FOLDER / RECORDING_PARAMETERS["raw_folder"]).glob("*_session_*/session_notes.md"),
                      key = lambda path: path.stat().st_mtime)
    if not sessions:
        return {"session": None, "lines": ""}
    with open(sessions[-1], encoding = "utf-8", errors = "replace") as notes_file:
        return {"session": sessions[-1].parent.name, "lines": "".join(notes_file.readlines()[-line_count:])}



class PageRequestHandler(BaseHTTPRequestHandler):

    runner = None
    allowed_hosts = ()

    def do_GET(self) -> None:

        if not self.__is_from_this_computer():
            return
        path = urlparse(self.path).path
        if path == "/":
            self.__send(HTTPStatus.OK, (CODE_FOLDER / "recording_page.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/options":
            self.__send_json(HTTPStatus.OK, {
                "operators": RECORDING_PAGE_PARAMETERS["operators"], "modes": RECORDING_PAGE_PARAMETERS["modes"],
                "sheets": RECORDING_PAGE_PARAMETERS["sheets"], "line_ends": RECORDING_PAGE_PARAMETERS["line_ends"],
                "start_codes": RECORDING_PAGE_PARAMETERS["start_codes"], "safety_checks": RECORDING_PAGE_PARAMETERS["safety_checks"],
                "camera_mount": RECORDING_PARAMETERS["camera_mount"]})
        elif path == "/status":
            self.__send_json(HTTPStatus.OK, {**self.runner.status(), "code": code_state(), "notes": latest_session_notes()})
        else:
            self.__send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})


    def do_POST(self) -> None:

        if not self.__is_from_this_computer():
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.__send_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send JSON."})
            return

        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            form = json.loads(self.rfile.read(length) or b"{}")
            if path == "/start":
                command = build_recording_command(form)
                title = "Practice recording" if form.get("is_simulated") is True else "Recording"
                self.runner.start(title, command)
            elif path == "/check-robot":
                self.runner.start("Check robot", [sys.executable, "-u", str(CODE_FOLDER / "check_hardware.py")])
            elif path == "/check-episodes":
                self.runner.start("Check episodes", [sys.executable, "-u", str(CODE_FOLDER / "check_episodes.py")])
            else:
                self.__send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return
        except FormError as error:
            self.__send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        except (ValueError, TypeError):
            self.__send_json(HTTPStatus.BAD_REQUEST, {"error": "The page sent something unreadable. Reload it and try again."})
            return
        self.__send_json(HTTPStatus.OK, {"is_started": True})


    def __is_from_this_computer(self) -> bool:
        """
        Accept only this page: a website open in the browser must not be able to start the robot by calling this address.
        """

        origin = self.headers.get("Origin")
        is_allowed = self.headers.get("Host") in self.allowed_hosts and (origin is None or urlparse(origin).netloc in self.allowed_hosts)
        if not is_allowed:
            self.__send_json(HTTPStatus.FORBIDDEN, {"error": "Only this page may use this address."})
        return is_allowed


    def __send_json(
            self,
            status: HTTPStatus,
            content: Dict
    ) -> None:

        self.__send(status, json.dumps(content).encode("utf-8"), "application/json")


    def __send(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str
    ) -> None:

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


    def log_message(self, format: str, *arguments) -> None:
        """Keep the terminal quiet: the page asks for its status every second."""



class LocalPageServer(ThreadingHTTPServer):

    def server_bind(self) -> None:
        """Bind without looking up this computer's name, which can hang for a long time on some networks."""

        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]



def make_server(
        port: int = RECORDING_PAGE_PARAMETERS["port"],
        runner: Optional[ToolRunner] = None
) -> LocalPageServer:
    """The page's server, reachable only from this computer."""

    host = RECORDING_PAGE_PARAMETERS["host"]
    handler = type("BoundPageRequestHandler", (PageRequestHandler,), {
        "runner": runner or ToolRunner(),
        "allowed_hosts": (f"{host}:{port}", f"localhost:{port}")})
    return LocalPageServer((host, port), handler)


def main() -> None:

    parser = argparse.ArgumentParser(description = "The local page that starts the recording tool.")
    parser.add_argument("--no-browser", action = "store_true", help = "do not open the page in the browser")
    arguments = parser.parse_args()

    server = make_server()
    address = f"http://{RECORDING_PAGE_PARAMETERS['host']}:{RECORDING_PAGE_PARAMETERS['port']}"
    print(f"Recording page: {address}\nKeep this program running. Ctrl-C closes the page; a running recording goes on.")
    if not arguments.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPage closed.")


if __name__ == "__main__":
    main()
