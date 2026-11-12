"""
Tests of visit_runs.py; no robot is needed and none can be reached: nothing here starts a "robot" line (the tests that
look at one are dry, or have the starter replaced by one that only notes the command), and the one test that runs
model_mode.py runs its simulated part.

Run from 02_code:
    python test_visit_runs.py
"""

import builtins
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy

import visit_runs
from config import MODEL_MODE_PARAMETERS

KNOWN_SHA256 = {}


def sha256_start(path: Path) -> str:
    """The first 12 characters of a checkpoint's sha256, worked out once per file."""

    if path not in KNOWN_SHA256:
        KNOWN_SHA256[path] = visit_runs.sha256_of(path)
    return KNOWN_SHA256[path][:12]


def from_code_folder(path: Path) -> str:
    """``path`` as a run list gives it: relative to 02_code."""

    return os.path.relpath(path, visit_runs.CODE_FOLDER)


def list_line(**changes) -> dict:
    """A good "simulated" line of a run list, with ``changes`` laid over it."""

    checkpoint = MODEL_MODE_PARAMETERS["checkpoint_path"]
    line = {"label": "check_step_065", "part": "simulated", "sheet": "sheet 03", "lay_out": "STD", "model_key": "E",
            "run_name": MODEL_MODE_PARAMETERS["run_name"], "checkpoints": from_code_folder(checkpoint), "checkpoint_sha256": sha256_start(checkpoint),
            "metrics": from_code_folder(MODEL_MODE_PARAMETERS["metrics_path"]), "step_rule": "fixed", "step_cm": "0.65", "step_budget": "6",
            "stall": "log", "ghost": "", "place_help": "no", "note": "a test line"}
    line.update(changes)
    return line


def written_run_list(
        folder: str,
        lines: list,
        delimiter: str = ","
) -> str:
    """Write ``lines`` as a run list into ``folder`` and return the file's path."""

    path = Path(folder) / "run_list.csv"
    with open(path, "w", newline = "", encoding = "utf-8") as file:
        writer = csv.DictWriter(file, fieldnames = visit_runs.RUN_LIST_COLUMNS, delimiter = delimiter)
        writer.writeheader()
        writer.writerows(lines)
    return str(path)


def started_nothing(command: list) -> int:
    """Stands in the starter's place where a test must not start anything."""

    raise AssertionError(f"a test that must start nothing tried to start {command[:3]}")


def asked_for_nothing(prompt: str = "") -> str:
    """Stands in input's place where a test must not get as far as asking for Enter."""

    raise AssertionError("a test that must start nothing was asked for Enter")


def small_jpeg() -> bytes:
    """A small grey picture as a JPEG file's bytes, for the reference picture of a line."""

    return cv2.imencode(".jpg", numpy.full((48, 64, 3), 90, dtype = numpy.uint8))[1].tobytes()


def started_with(
        arguments: list,
        answer = "",
        starter_exit_code: int = 0,
        new_folder: str = "",
        at_question = None
) -> tuple:
    """
    Run visit_runs.main with ``arguments`` while the starter only notes the command and gives back
    ``starter_exit_code`` (it makes the run folder ``new_folder`` first when one is named, as model_mode.py would), and
    while the question for Enter is answered at once with ``answer``, or raises it when it is an exception
    (``at_question``, when given, is called at the moment the question is asked).

    Returns
    -------
    tuple of int, str, list and list
        The exit code, what was printed, the commands noted and the questions asked.
    """

    printed, commands, prompts = io.StringIO(), [], []

    def noting_starter(command: list) -> int:

        commands.append(command)
        if new_folder:
            Path(new_folder).mkdir()
        return starter_exit_code

    def answered(prompt: str = "") -> str:

        prompts.append(prompt)
        if at_question is not None:
            at_question()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    real_parts = (visit_runs.run_in_this_terminal, builtins.input)
    visit_runs.run_in_this_terminal, builtins.input = noting_starter, answered
    try:
        with contextlib.redirect_stdout(printed):
            exit_code = visit_runs.main(arguments)
    finally:
        visit_runs.run_in_this_terminal, builtins.input = real_parts
    return exit_code, printed.getvalue(), commands, prompts


def outcome_of(arguments: list) -> tuple:
    """
    Run visit_runs.main with ``arguments`` while nothing can be started and no Enter can be given.

    Returns
    -------
    tuple of int and str
        The exit code and what was printed.
    """

    printed = io.StringIO()
    real_parts = (visit_runs.run_in_this_terminal, builtins.input)
    visit_runs.run_in_this_terminal, builtins.input = started_nothing, asked_for_nothing
    try:
        with contextlib.redirect_stdout(printed):
            exit_code = visit_runs.main(arguments)
    finally:
        visit_runs.run_in_this_terminal, builtins.input = real_parts
    return exit_code, printed.getvalue()


def test_dry_prints_the_line_in_plain_words_and_the_expected_command() -> None:

    checkpoint = MODEL_MODE_PARAMETERS["checkpoint_path"]
    second_checkpoint = checkpoint.parent.parent.parent / "2026-11-10_resnet18_model_1_all_sheets_seed_2" / "checkpoints" / "all_sheets.pt"
    metrics = MODEL_MODE_PARAMETERS["metrics_path"]
    python, driver = sys.executable, str(visit_runs.CODE_FOLDER / "model_mode.py")
    with tempfile.TemporaryDirectory() as folder:
        ghost = Path(folder) / "first_picture.jpg"
        ghost.write_bytes(small_jpeg())
        team_line = list_line(label = "check_team", part = "robot", model_key = "B", step_rule = "predicted", step_cm = "", step_budget = "", stall = "",
                              checkpoints = from_code_folder(checkpoint) + ";" + from_code_folder(second_checkpoint),
                              checkpoint_sha256 = sha256_start(checkpoint) + ";" + sha256_start(second_checkpoint), ghost = str(ghost), place_help = "yes")
        run_list = written_run_list(folder, [list_line(), team_line])

        exit_code, printed = outcome_of([run_list, "check_step_065", "--dry", "--run-folder", folder])
        expected = [python, driver, "simulated", "--label", "check_step_065", "--run-name", "resnet18_model_1_all_sheets", "--checkpoint", str(checkpoint.resolve()),
                    "--metrics", str(metrics.resolve()), "--step-length", "fixed", "--step-cm", "0.65", "--step-budget", "6", "--stall", "log",
                    "--run-folder", str(Path(folder).resolve())]
        assert exit_code == 0 and printed.rstrip().endswith(subprocess.list2cmdline(expected)), printed
        for words in ("Run check_step_065: the simulated robot", "sheet 03, lay-out STD", "E (recipe resnet18_model_1_all_sheets), 1 checkpoint file, sha256 checked",
                      "fixed, 0.65 cm", "6 steps", "only logs, the run goes on", "nothing is started"):
            assert words in printed, (words, printed)
        # each checkpoint by its run folder's name first, which tells the models apart, then by its full path
        assert f"2026-11-10_resnet18_model_1_all_sheets / all_sheets.pt  ({sha256_start(checkpoint)})\n                   {checkpoint.resolve()}" in printed, printed

        # a team with the model's own step length, the placing aid and a reference picture; cells left empty give no flag
        exit_code, printed = outcome_of([run_list, "check_team", "--dry", "--run-folder", folder])
        expected = [python, driver, "robot", "--label", "check_team", "--run-name", "resnet18_model_1_all_sheets", "--checkpoint", str(checkpoint.resolve()),
                    str(second_checkpoint.resolve()), "--metrics", str(metrics.resolve()), "--step-length", "predicted", "--ghost", str(ghost),
                    "--run-folder", str(Path(folder).resolve()), "--place-help", "--fresh-keys"]
        assert exit_code == 0 and printed.rstrip().endswith(subprocess.list2cmdline(expected)), printed
        for words in ("Run check_team: THE REAL ROBOT", "a team of 2 checkpoint files", "the length the model asks for (predicted)", "the default (250 steps)",
                      "the default (ends the run)", "placing aid  yes", "first_picture.jpg", "2026-11-10_resnet18_model_1_all_sheets / all_sheets.pt",
                      "2026-11-10_resnet18_model_1_all_sheets_seed_2 / all_sheets.pt"):
            assert words in printed, (words, printed)

        # a list that a spreadsheet saved with ";" between the cells is read too
        exit_code, printed = outcome_of([written_run_list(folder, [list_line(), team_line], delimiter = ";"), "check_team", "--dry", "--run-folder", folder])
        assert exit_code == 0 and "a team of 2 checkpoint files" in printed, printed


def test_wrong_sha256_is_refused_and_the_file_is_named() -> None:

    checkpoint = MODEL_MODE_PARAMETERS["checkpoint_path"]
    with tempfile.TemporaryDirectory() as folder:
        run_list = written_run_list(folder, [list_line(checkpoint_sha256 = "0123456789ab")])
        for flags in ([], ["--dry"]):
            exit_code, printed = outcome_of([run_list, "check_step_065", "--run-folder", folder] + flags)
            assert exit_code == 1 and printed.startswith("not started:"), printed
            assert str(checkpoint.resolve()) in printed and "0123456789ab" in printed and sha256_start(checkpoint) in printed, printed
        # every one of the twelve characters counts: a checksum that differs in its last character only is refused
        right = sha256_start(checkpoint)
        last_changed = right[:11] + ("0" if right[11] != "0" else "1")
        exit_code, printed = outcome_of([written_run_list(folder, [list_line(checkpoint_sha256 = last_changed)]), "check_step_065", "--dry", "--run-folder", folder])
        assert exit_code == 1 and "is not the one the list means" in printed and last_changed in printed, printed
        # every file of a team is checked: a wrong checksum for the second file is refused, and the second file is named
        second_checkpoint = checkpoint.parent.parent.parent / "2026-11-10_resnet18_model_1_all_sheets_seed_2" / "checkpoints" / "all_sheets.pt"
        team_line = list_line(checkpoints = from_code_folder(checkpoint) + ";" + from_code_folder(second_checkpoint), checkpoint_sha256 = right + ";0123456789ab")
        exit_code, printed = outcome_of([written_run_list(folder, [team_line]), "check_step_065", "--dry", "--run-folder", folder])
        assert exit_code == 1 and f"the checkpoint file {second_checkpoint.resolve()} is not the one the list means" in printed, printed


def test_lines_that_do_not_fit_are_refused() -> None:

    cases = [(list_line(part = "real"), "part is"), (list_line(step_rule = "predicted"), "does not go together"), (list_line(step_cm = "6.5"), "step_cm is"),
             (list_line(step_cm = "0,65"), "written with a point"), (list_line(step_budget = "many"), "step_budget is"), (list_line(stall = "stop"), "stall is"),
             (list_line(place_help = "maybe"), "place_help is"), (list_line(run_name = "no_such_recipe"), "not a recipe of TRAINING_RUNS"),
             (list_line(checkpoints = "../04_runs/no_such_folder/all_sheets.pt"), "is missing"), (list_line(checkpoint_sha256 = "81cd"), "first 12 characters"),
             (list_line(checkpoint_sha256 = "1.23457E+11"), "a spreadsheet may have changed the cell"), (list_line(checkpoint_sha256 = ""), "every file needs its own"),
             (list_line(metrics = "../04_runs/no_such_folder/metrics.json"), "metrics file"), (list_line(ghost = "no_such_picture.jpg"), "reference picture"),
             (list_line(label = "two words"), "names a folder")]
    checkpoint = MODEL_MODE_PARAMETERS["checkpoint_path"]
    runs_folder = checkpoint.parent.parent.parent
    other_checkpoint = runs_folder / "2026-11-10_resnet18_model_3_16_rounds_all_sheets" / "checkpoints" / "all_sheets.pt"
    other_metrics = runs_folder / "2026-11-10_resnet18_model_2_chunk_all_sheets" / "metrics.json"
    # a robot line must say which sheet lies on the table and which model drives; a step budget far beyond any run is a
    # typing mistake; a label that ends in a dot would never find its run folder again (Windows drops the dot)
    cases += [(list_line(part = "robot", sheet = ""), "sheet is empty"), (list_line(part = "robot", model_key = ""), "model_key is empty"),
              (list_line(step_budget = "100000"), "step_budget is"), (list_line(step_budget = "501"), "from 1 to 500"), (list_line(step_budget = "0"), "step_budget is"),
              (list_line(label = "run."), "begin and end with a letter, a digit or"), (list_line(label = ".."), "begin and end with a letter, a digit or"),
              # the files must fit the recipe: another model's file under this recipe's name (with its own right checksum),
              # the same file twice in a team, another recipe's metrics file, a metrics file that is none
              (list_line(checkpoints = from_code_folder(other_checkpoint), checkpoint_sha256 = sha256_start(other_checkpoint)), "do not belong together"),
              (list_line(checkpoints = from_code_folder(checkpoint) + ";" + from_code_folder(checkpoint),
                         checkpoint_sha256 = sha256_start(checkpoint) + ";" + sha256_start(checkpoint)), "a team needs different files"),
              (list_line(metrics = from_code_folder(other_metrics)), "belongs to the recipe \"resnet18_model_2_chunk_all_sheets\""),
              (list_line(metrics = from_code_folder(checkpoint)), "cannot be read as the metrics.json")]
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "no_picture.jpg").write_text("no picture in here", encoding = "utf-8")
        cases.append((list_line(ghost = str(Path(folder) / "no_picture.jpg")), "cannot be read as a picture"))
        for line, words in cases:
            exit_code, printed = outcome_of([written_run_list(folder, [line]), line["label"], "--run-folder", folder])
            assert exit_code == 1 and printed.startswith("not started:") and words in printed, (words, printed)
        # a label that begins with "-" would be taken for a flag; the line itself is refused too
        for label in ("-start", "--dry", ".", ".hidden", "run-"):
            try:
                visit_runs.checked_run(list_line(label = label))
                raise AssertionError(f"the label {label} was taken")
            except visit_runs.RunListProblem as problem:
                assert "begin and end with a letter, a digit or" in str(problem), problem
        # what stays allowed: a simulated line without sheet and model key, and the largest step budget
        exit_code, printed = outcome_of([written_run_list(folder, [list_line(sheet = "", model_key = "", step_budget = "500")]), "check_step_065", "--dry", "--run-folder", folder])
        assert exit_code == 0 and "sheet        not given" in printed and "model        not given" in printed and "500 steps" in printed, printed
        # a list that a spreadsheet saved as plain "CSV" (the Windows character set) with an umlaut in a note is read; one
        # saved as "Unicode Text" is refused with the way out
        list_text = Path(written_run_list(folder, [list_line(note = "gr\u00fcn, 45\u00b0")])).read_text(encoding = "utf-8")
        (Path(folder) / "windows_list.csv").write_bytes(list_text.encode("cp1252"))
        exit_code, printed = outcome_of([str(Path(folder) / "windows_list.csv"), "check_step_065", "--dry", "--run-folder", folder])
        assert exit_code == 0 and "gr\u00fcn, 45\u00b0" in printed, printed
        (Path(folder) / "unicode_list.csv").write_bytes(list_text.encode("utf-16"))
        exit_code, printed = outcome_of([str(Path(folder) / "unicode_list.csv"), "check_step_065", "--dry", "--run-folder", folder])
        assert exit_code == 1 and "CSV UTF-8 (comma delimited)" in printed, printed
        # the list itself: no such label, a label on two lines, a column that is missing, no file
        exit_code, printed = outcome_of([written_run_list(folder, [list_line()]), "another_label", "--run-folder", folder])
        assert exit_code == 1 and "no line with the label \"another_label\"" in printed and "check_step_065" in printed, printed
        exit_code, printed = outcome_of([written_run_list(folder, [list_line(), list_line(note = "the same label again")]), "check_step_065", "--run-folder", folder])
        assert exit_code == 1 and "stands on 2 lines" in printed, printed
        (Path(folder) / "short_list.csv").write_text("label,part,sheet\ncheck_step_065,simulated,sheet 03\n", encoding = "utf-8")
        exit_code, printed = outcome_of([str(Path(folder) / "short_list.csv"), "check_step_065", "--run-folder", folder])
        assert exit_code == 1 and "lacks the column(s)" in printed and "checkpoint_sha256" in printed, printed
        exit_code, printed = outcome_of([str(Path(folder) / "no_such_list.csv"), "check_step_065", "--run-folder", folder])
        assert exit_code == 1 and "cannot be read" in printed, printed


def test_label_with_a_run_folder_is_refused_unless_again() -> None:

    with tempfile.TemporaryDirectory() as folder:
        run_list = written_run_list(folder, [list_line()])
        # a run of this label, and a run of another label that only begins in the same way
        (Path(folder) / "2026-11-12_101500_check_step_065").mkdir()
        (Path(folder) / "2026-11-12_101700_check_step_065_second").mkdir()
        assert [path.name for path in visit_runs.runs_with_label("check_step_065", Path(folder))] == ["2026-11-12_101500_check_step_065"]
        for flags in ([], ["--dry"]):
            exit_code, printed = outcome_of([run_list, "check_step_065", "--run-folder", folder] + flags)
            assert exit_code == 1 and "already has a run folder (2026-11-12_101500_check_step_065)" in printed and "--again" in printed, printed
        exit_code, printed = outcome_of([run_list, "check_step_065", "--dry", "--again", "--run-folder", folder])
        assert exit_code == 0 and "this label has 1 run folder(s) already" in printed, printed
        # capitals are not told apart, as Windows does not tell them apart in a folder's name
        capitals_list = written_run_list(folder, [list_line(label = "Check_STEP_065")])
        exit_code, printed = outcome_of([capitals_list, "Check_STEP_065", "--dry", "--run-folder", folder])
        assert exit_code == 1 and "already has a run folder (2026-11-12_101500_check_step_065)" in printed, printed


def test_a_simulated_line_runs_through() -> None:

    import pandas

    captured = []

    def quiet_starter(command: list) -> int:

        # the same command, from the same folder, with what it prints kept instead of shown
        finished = subprocess.run(command, cwd = str(visit_runs.CODE_FOLDER), capture_output = True, text = True)
        captured.append(finished.stdout + finished.stderr)
        return finished.returncode

    prompts = []
    real_parts = (visit_runs.run_in_this_terminal, builtins.input)
    with tempfile.TemporaryDirectory() as folder:
        run_list = written_run_list(folder, [list_line()])
        visit_runs.run_in_this_terminal, builtins.input = quiet_starter, lambda prompt = "": prompts.append(prompt) or ""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = visit_runs.main([run_list, "check_step_065", "--run-folder", folder])
        finally:
            visit_runs.run_in_this_terminal, builtins.input = real_parts
        # one Enter was asked for, model_mode.py ran its simulated part and logged the run under the label
        assert exit_code == 0 and len(prompts) == 1 and prompts[0].startswith("Press Enter to start this run"), (exit_code, prompts, captured)
        run_folders = visit_runs.runs_with_label("check_step_065", Path(folder))
        assert len(run_folders) == 1, (run_folders, captured)
        # the list's line lies beside the run's own log, so that the run stays tied to its sheet, lay-out, model key and note
        kept = json.loads((run_folders[0] / "run_list_line.json").read_text(encoding = "utf-8"))
        assert kept["line"] == list_line() and kept["run_list"] == str(Path(run_list).resolve()) and kept["command"][2] == "simulated", kept
        summary = json.loads((run_folders[0] / "summary.json").read_text(encoding = "utf-8"))
        steps = pandas.read_csv(run_folders[0] / "steps.csv")
        # the line's flags reached the run: the 0.65 cm step, the budget of 6, the stall rule that only logs
        assert summary["label"] == "check_step_065" and summary["ending"] in ("step_budget_reached", "stopped") and summary["is_lifted"], summary
        assert summary["fixed_step_cm"] == 0.65 and summary["stall_action"] == "log" and summary["parameters"]["step_budget"] == 6, summary
        assert 1 <= summary["pictures"] <= 6 and (steps["step_cm"].dropna() == 0.65).all(), steps["step_cm"].tolist()
        # the label now has a run folder, so a second start is refused
        exit_code, printed = outcome_of([run_list, "check_step_065", "--run-folder", folder])
        assert exit_code == 1 and "already has a run folder" in printed, printed


def test_starter_waits_for_the_command_and_gives_its_exit_code() -> None:

    # a harmless command in place of model_mode.py: it must run from 02_code and its exit code must come back
    script = "import os, sys; sys.exit(7 if os.path.samefile(os.getcwd(), sys.argv[1]) else 1)"
    assert visit_runs.run_in_this_terminal([sys.executable, "-c", script, str(visit_runs.CODE_FOLDER)]) == 7


class InterruptedProcess:
    """
    Stands in subprocess.Popen's place: the first two waits for the command are interrupted by Ctrl+C, the third gives
    the exit code 5. The command must never be ended by force (it would leave the arm low or the motors off).
    """

    waits = 0

    def __init__(self, command: list, cwd: str = None):

        InterruptedProcess.waits = 0

    def wait(self) -> int:

        InterruptedProcess.waits += 1
        if InterruptedProcess.waits <= 2:
            raise KeyboardInterrupt
        return 5

    def kill(self) -> None:

        raise AssertionError("the starter ended model_mode.py by force")

    terminate = send_signal = kill


def test_starter_goes_on_waiting_after_ctrl_c_and_says_once_what_to_do() -> None:

    printed = io.StringIO()
    real_popen = subprocess.Popen
    subprocess.Popen = InterruptedProcess
    try:
        with contextlib.redirect_stdout(printed):
            exit_code = visit_runs.run_in_this_terminal(["a command that is never started"])
    finally:
        subprocess.Popen = real_popen
    assert exit_code == 5 and InterruptedProcess.waits == 3, (exit_code, InterruptedProcess.waits)
    assert printed.getvalue().count("Ctrl+C was passed on to model_mode.py") == 1 and "close this window" in printed.getvalue(), printed.getvalue()


class WaitingKeys:
    """Stands in msvcrt's place: ``count`` keys that were pressed too early wait in the terminal."""

    def __init__(self, count: int):

        self.count = count

    def kbhit(self) -> bool:

        return self.count > 0

    def getch(self) -> bytes:

        self.count -= 1
        return b"\r"


def test_a_robot_line_is_started_only_by_an_enter_from_a_person() -> None:

    with tempfile.TemporaryDirectory() as folder:
        run_list = written_run_list(folder, [list_line(part = "robot", place_help = "yes")])
        arguments = [run_list, "check_step_065", "--run-folder", folder]
        # Ctrl+C at the question, or no keyboard to read from: nothing is started
        for no_enter in (KeyboardInterrupt(), EOFError()):
            exit_code, printed, commands, prompts = started_with(arguments, answer = no_enter)
            assert exit_code == 1 and commands == [] and len(prompts) == 1 and printed.rstrip().endswith("not started"), (exit_code, commands, printed)
        # keys pressed before the question is on the screen are thrown away and do not answer it
        real_msvcrt = visit_runs.msvcrt
        visit_runs.msvcrt = WaitingKeys(2)
        waiting = []
        try:
            exit_code, printed, commands, prompts = started_with(arguments + ["--dry"])
            assert visit_runs.msvcrt.count == 2 and commands == [] and prompts == [], "a dry run asks for nothing and touches no key"
            exit_code, printed, commands, prompts = started_with(arguments, at_question = lambda: waiting.append(visit_runs.msvcrt.count))
        finally:
            visit_runs.msvcrt = real_msvcrt
        assert waiting == [0] and "keys pressed before this question were ignored" in printed, (waiting, printed)
        # with Enter: the real robot is named before the question, and the command started is the one the dry run printed,
        # with the robot part, the placing aid and --fresh-keys
        assert exit_code == 0 and len(commands) == 1 and len(prompts) == 1, (exit_code, commands, prompts)
        assert printed.index("THE REAL ROBOT moves in this run") < printed.index("starting model_mode.py"), printed
        assert commands[0][2] == "robot" and commands[0][-2:] == ["--place-help", "--fresh-keys"], commands[0]
        dry_exit_code, dry_printed = outcome_of(arguments + ["--dry"])
        assert dry_exit_code == 0 and dry_printed.rstrip().endswith(subprocess.list2cmdline(commands[0])), dry_printed
        # the stand-in starter made no run folder, as when model_mode.py ends "not started": that is said
        assert "no run folder: the run was not started" in printed, printed


def test_after_a_run_its_folder_and_an_error_ending_are_named() -> None:

    with tempfile.TemporaryDirectory() as folder:
        run_list = written_run_list(folder, [list_line(part = "robot", note = "the third run of part A4")])
        new_folder = Path(folder) / "2026-11-12_150000_check_step_065"
        exit_code, printed, commands, prompts = started_with([run_list, "check_step_065", "--run-folder", folder], starter_exit_code = 3, new_folder = str(new_folder))
        # the exit code of model_mode.py comes back and is named when it is not 0
        assert exit_code == 3 and "model_mode.py ended with exit code 3" in printed, (exit_code, printed)
        # the run's folder is named, and the list's line lies in it
        assert f"the run's folder: {new_folder}" in printed, printed
        kept = json.loads((new_folder / "run_list_line.json").read_text(encoding = "utf-8"))
        assert kept["line"]["sheet"] == "sheet 03" and kept["line"]["model_key"] == "E" and kept["line"]["note"] == "the third run of part A4", kept
        assert kept["command"] == commands[0] and kept["line"]["part"] == "robot", kept
        # a clean ending says nothing about an exit code
        exit_code, printed, commands, prompts = started_with([run_list, "check_step_065", "--again", "--run-folder", folder],
                                                             new_folder = str(Path(folder) / "2026-11-12_150500_check_step_065"))
        assert exit_code == 0 and "exit code" not in printed and "2026-11-12_150500_check_step_065" in printed, printed


if __name__ == "__main__":
    for test in [test_dry_prints_the_line_in_plain_words_and_the_expected_command, test_wrong_sha256_is_refused_and_the_file_is_named,
                 test_lines_that_do_not_fit_are_refused, test_label_with_a_run_folder_is_refused_unless_again,
                 test_a_simulated_line_runs_through, test_starter_waits_for_the_command_and_gives_its_exit_code,
                 test_starter_goes_on_waiting_after_ctrl_c_and_says_once_what_to_do, test_a_robot_line_is_started_only_by_an_enter_from_a_person,
                 test_after_a_run_its_folder_and_an_error_ending_are_named]:
        test()
        print(f"{test.__name__}: passed")
    print("all tests passed")
