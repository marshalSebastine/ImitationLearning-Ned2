"""
Start one run of a visit from the visit's run list, so that nobody types a model's files or a run's flags at the robot.

Usage, from 02_code:
    python visit_runs.py <run_list.csv> <label>            check the line, say what it is, wait for Enter, run it
    python visit_runs.py <run_list.csv> <label> --dry      check the line, say what it is, print the command, run nothing
    python visit_runs.py <run_list.csv> <label> --again    run a label once more although it already has a run folder

The run list is a CSV file with one line per run and these columns:
    label               the run's name; its log folder is named after it (letters, digits, "_", "-" and "." only,
                        beginning and ending with a letter, a digit or "_")
    part                robot or simulated
    sheet, lay_out      which sheet lies on the table and how (only shown; a robot line must name the sheet)
    model_key           the model's short name in the visit's plan (only shown; a robot line must give it)
    run_name            the model's recipe, a key of TRAINING_RUNS in config.py
    checkpoints         the model's .pt file, or several separated by ";" (a team); relative to 02_code
    checkpoint_sha256   the first 12 characters of each file's sha256, separated by ";", in the same order
    metrics             the metrics.json of that recipe; relative to 02_code
    step_rule           fixed or predicted
    step_cm             the fixed step's length in centimetres; empty for the default of config.py
    step_budget         the most steps of the run; empty for the default
    stall               end or log; empty for the default
    ghost               a reference picture for the go-ahead window (relative to 02_code), or empty
    place_help          yes or no
    note                anything (only shown)

Nothing is started unless every checkpoint file is there and its sha256 begins with the characters the list gives,
unless the files fit the recipe (each checkpoint's run folder carries the recipe's name, and the metrics file names the
recipe as its own), and unless the label has no run folder yet (--again lifts that). The script proves which files a
line means; that the line is the one for the sheet on the table and the model that is meant, a person reads before
Enter: read every line once with --dry before a visit.

The run itself is model_mode.py, started in this terminal with the line's flags and the label; the keys go to it. End
a run with Esc, not with Ctrl+C: Ctrl+C that arrives while model_mode.py waits for the robot's answer can leave the
link to the robot out of step, and the run then never ends by itself (see model_mode.py). A "robot" line moves THE REAL
ROBOT: only with Emranul's go, the checklist done, nothing else connected and the stop button in reach. A robot line is
started with --fresh-keys: keys pressed too early do not count (neither for this script's own Enter nor for the first
question of model_mode.py, which comes some seconds later, when the model is loaded), Esc ends the run also when it
was pressed twice or another key waits in the terminal, and Ctrl+C is held back until the robot has answered. After
the run the list's line is written beside the run's own log as run_list_line.json, so that the run stays tied to its
sheet, lay-out, model key and note.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from config import MODEL_MODE_PARAMETERS, TRAINING_RUNS

try:
    import msvcrt
except ImportError:
    # not on Windows: keys typed ahead cannot be thrown away there; the list can still be checked and shown (--dry)
    msvcrt = None

CODE_FOLDER = Path(__file__).resolve().parent
RUN_LIST_COLUMNS = ["label", "part", "sheet", "lay_out", "model_key", "run_name", "checkpoints", "checkpoint_sha256", "metrics",
                    "step_rule", "step_cm", "step_budget", "stall", "ghost", "place_help", "note"]


class RunListProblem(Exception):
    """A reason not to start a run: the list, one of its files or the label is not as it has to be."""



def run_list_lines(run_list_path: Path) -> List[Dict[str, str]]:
    """
    Read the run list.

    Parameters
    ----------
    run_list_path : pathlib.Path
        The CSV file; separated by "," or, when a spreadsheet with German settings saved it, by ";".

    Returns
    -------
    list of dict
        One dict per line: the text of every column of RUN_LIST_COLUMNS, without spaces round it.

    Raises
    ------
    RunListProblem
        When the file cannot be read or a column is missing.
    """

    try:
        raw = Path(run_list_path).read_bytes()
    except OSError as error:
        raise RunListProblem(f"the run list {run_list_path} cannot be read ({error})")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw:
        raise RunListProblem(f"the run list {run_list_path} is not saved as text this script can read (it looks like \"Unicode Text\"); "
                             "save it as \"CSV UTF-8 (comma delimited)\"")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # a spreadsheet's plain "CSV" is written in the Windows character set, where an umlaut or a degree sign is one byte
        text = raw.decode("cp1252", errors = "replace")
    header = text.splitlines()[0] if text.strip() else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter = delimiter)
    names = [(name or "").strip() for name in (reader.fieldnames or [])]
    missing = [name for name in RUN_LIST_COLUMNS if name not in names]
    if missing:
        raise RunListProblem(f"the run list {run_list_path} lacks the column(s) {', '.join(missing)}; its first line must name "
                             f"these columns: {', '.join(RUN_LIST_COLUMNS)}")
    lines = []
    for row in reader:
        line = {(name or "").strip(): (value or "").strip() for name, value in row.items() if isinstance(value, (str, type(None)))}
        if any(line.get(name) for name in RUN_LIST_COLUMNS):
            lines.append({name: line.get(name, "") for name in RUN_LIST_COLUMNS})
    return lines



def line_with_label(
        lines: List[Dict[str, str]],
        label: str
) -> Dict[str, str]:
    """
    The one line of the run list that carries ``label``.

    Raises
    ------
    RunListProblem
        When no line or more than one line carries it.
    """

    matching = [line for line in lines if line["label"] == label]
    if not matching:
        raise RunListProblem(f"the run list has no line with the label \"{label}\"; its labels are: {', '.join(line['label'] for line in lines) or 'none'}")
    if len(matching) > 1:
        raise RunListProblem(f"the label \"{label}\" stands on {len(matching)} lines of the run list; every run needs a label of its own")
    return matching[0]



def path_from_code_folder(text: str) -> Path:
    """A path of the run list in full: as written when it is absolute, else taken from 02_code."""

    path = Path(text)
    return path if path.is_absolute() else (CODE_FOLDER / path).resolve()



def sha256_of(path: Path) -> str:
    """The sha256 of the file at ``path`` as 64 hexadecimal characters."""

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()



def checkpoint_run_folder(path: Path) -> str:
    """The name of the training run's folder a checkpoint file lies in ("<run folder>/checkpoints/<file>", or the file's own folder)."""

    return path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name



def is_picture_file(path: Path) -> bool:
    """Whether the file at ``path`` can be decoded as a picture, in the way model_mode.py reads a reference picture."""

    # imported here, so that the script stays light for a line without a reference picture
    import cv2
    import numpy

    try:
        picture_bytes = Path(path).read_bytes()
        return bool(picture_bytes) and cv2.imdecode(numpy.frombuffer(picture_bytes, numpy.uint8), cv2.IMREAD_COLOR) is not None
    except (OSError, cv2.error):
        return False



def checked_run(
        line: Dict[str, str],
        parameters: Optional[Dict] = None
) -> Dict:
    """
    Check one line of the run list and turn it into what the run needs. Nothing is started.

    Parameters
    ----------
    line : dict
        A line from run_list_lines.
    parameters : dict, optional
        Overrides of MODEL_MODE_PARAMETERS ("run_list_sha256_characters", "run_list_step_budget_most", "step_cm_allowed").

    Returns
    -------
    dict
        The line's cells, with "part", "step_rule" and "stall" in small letters, "checkpoints" as a list of full paths,
        "checkpoint_sha256" as the list of characters given for them, "metrics" and "ghost" as full paths ("ghost" empty
        when none is given) and "is_place_help_wanted" for the cell "place_help".

    Raises
    ------
    RunListProblem
        With the first thing that is wrong: a cell that does not fit, a file that is missing, a checkpoint file whose
        sha256 does not begin with the characters given for it (the file is named), or files that do not fit the
        recipe.
    """

    settings = dict(MODEL_MODE_PARAMETERS)
    settings.update(parameters or {})
    run = dict(line)
    label = run["label"]
    # a dot at the end is dropped from a folder's name by Windows (the label would never find its run folder again), and
    # a "-" at the start is taken for a flag
    if not re.fullmatch(r"[A-Za-z0-9_]([A-Za-z0-9_.-]*[A-Za-z0-9_])?", label):
        raise RunListProblem(f"the label \"{label}\" names a folder, so it may hold letters, digits, \"_\", \"-\" and \".\" only, and it must begin and "
                             "end with a letter, a digit or \"_\"")
    for name, allowed in (("part", ["robot", "simulated"]), ("step_rule", ["fixed", "predicted"]), ("stall", ["", "end", "log"]), ("place_help", ["", "yes", "no"])):
        run[name] = run[name].lower()
        if run[name] not in allowed:
            raise RunListProblem(f"line {label}: {name} is \"{line[name]}\"; it must be {' or '.join(value or 'empty' for value in allowed)}")
    run["is_place_help_wanted"] = run["place_help"] == "yes"
    if run["part"] == "robot":
        # what the person reads before Enter must be there: which sheet lies on the table and which model drives
        for name in ("sheet", "model_key"):
            if not run[name]:
                raise RunListProblem(f"line {label}: {name} is empty; a robot line must name the sheet and the model, so that both can be read before Enter")
    if run["run_name"] not in TRAINING_RUNS:
        raise RunListProblem(f"line {label}: run_name \"{run['run_name']}\" is not a recipe of TRAINING_RUNS in config.py")
    if run["step_cm"]:
        if run["step_rule"] == "predicted":
            raise RunListProblem(f"line {label}: step_cm is the length of the fixed step and does not go together with the step rule \"predicted\"; leave step_cm empty")
        least_cm, most_cm = settings["step_cm_allowed"]
        try:
            is_step_allowed = least_cm <= float(run["step_cm"]) <= most_cm
        except ValueError:
            is_step_allowed = False
        if not is_step_allowed:
            raise RunListProblem(f"line {label}: step_cm is \"{run['step_cm']}\"; it must be a number of centimetres between {least_cm:g} and {most_cm:g}, "
                                 "written with a point, for example 0.65")
    if run["step_budget"] and not (re.fullmatch(r"[0-9]+", run["step_budget"]) and 0 < int(run["step_budget"]) <= settings["run_list_step_budget_most"]):
        raise RunListProblem(f"line {label}: step_budget is \"{run['step_budget']}\"; it must be a whole number of steps from 1 to "
                             f"{settings['run_list_step_budget_most']} (\"run_list_step_budget_most\" in config.py), or empty")

    checkpoint_texts = [text.strip() for text in run["checkpoints"].split(";") if text.strip()]
    listed_characters = [text.strip().lower() for text in run["checkpoint_sha256"].split(";") if text.strip()]
    if not checkpoint_texts:
        raise RunListProblem(f"line {label}: no checkpoint file is given")
    if len(listed_characters) != len(checkpoint_texts):
        raise RunListProblem(f"line {label}: {len(checkpoint_texts)} checkpoint file(s) but {len(listed_characters)} sha256 given; every file needs its own")
    count = settings["run_list_sha256_characters"]
    run["checkpoints"] = [path_from_code_folder(text) for text in checkpoint_texts]
    run["checkpoint_sha256"] = listed_characters
    found_sha256 = {}
    for path, characters in zip(run["checkpoints"], listed_characters):
        if not (count <= len(characters) <= 64 and all(character in "0123456789abcdef" for character in characters)):
            raise RunListProblem(f"line {label}: \"{characters}\" is not the first {count} characters of a sha256 (digits and the letters a to f) "
                                 f"for {path.name}; a spreadsheet may have changed the cell")
        if not path.is_file():
            raise RunListProblem(f"line {label}: the checkpoint file {path} is missing (trained models are not in git: copy it there first)")
        found = sha256_of(path)
        if not found.startswith(characters):
            raise RunListProblem(f"line {label}: the checkpoint file {path} is not the one the list means: its sha256 begins with "
                                 f"{found[:len(characters)]}, the list says {characters}")
        # the file and the recipe must belong together: a training run's folder carries its recipe's name (a team's
        # seed folders carry it too), so a line whose file is another model's is not started under a wrong name
        if run["run_name"] not in checkpoint_run_folder(path):
            raise RunListProblem(f"line {label}: the checkpoint file {path} lies in the run folder \"{checkpoint_run_folder(path)}\", whose name does not "
                                 f"hold the recipe \"{run['run_name']}\" (run_name); the file and the recipe do not belong together")
        if found in found_sha256:
            raise RunListProblem(f"line {label}: the checkpoint file {path} is the same file as {found_sha256[found]}; a team needs different files")
        found_sha256[found] = path
    run["metrics"] = path_from_code_folder(run["metrics"]) if run["metrics"] else None
    if run["metrics"] is None or not run["metrics"].is_file():
        raise RunListProblem(f"line {label}: the metrics file {run['metrics'] or '(none given)'} is missing")
    try:
        metrics_name = json.loads(run["metrics"].read_text(encoding = "utf-8"))["name"]
    except (OSError, ValueError, KeyError, TypeError):
        raise RunListProblem(f"line {label}: the metrics file {run['metrics']} cannot be read as the metrics.json of a training run "
                             "(which names its recipe under \"name\")")
    if metrics_name != run["run_name"]:
        raise RunListProblem(f"line {label}: the metrics file {run['metrics']} belongs to the recipe \"{metrics_name}\", but the line's run_name is "
                             f"\"{run['run_name']}\"; give the metrics.json of that recipe")
    run["ghost"] = path_from_code_folder(run["ghost"]) if run["ghost"] else ""
    if run["ghost"] and not run["ghost"].is_file():
        raise RunListProblem(f"line {label}: the reference picture {run['ghost']} (ghost) is missing")
    if run["ghost"] and not is_picture_file(run["ghost"]):
        raise RunListProblem(f"line {label}: the reference picture {run['ghost']} (ghost) cannot be read as a picture")
    return run



def plain_words(run: Dict) -> str:
    """
    One checked line of the run list in plain words, for the person to read before Enter.

    Parameters
    ----------
    run : dict
        From checked_run.

    Returns
    -------
    str
        Several lines: the label and the part, the sheet and its lay-out, the model with its checked files, the step,
        the step budget, the stall rule, the reference picture, the placing aid and the note.
    """

    if run["step_rule"] == "predicted":
        step = f"the length the model asks for (predicted), at most {100.0 * MODEL_MODE_PARAMETERS['step_most_m']:g} cm"
    elif run["step_cm"]:
        step = f"fixed, {float(run['step_cm']):g} cm"
    else:
        step = f"fixed, the default of config.py ({100.0 * MODEL_MODE_PARAMETERS['step_m']:g} cm)"
    default_budget = MODEL_MODE_PARAMETERS["simulated_step_budget" if run["part"] == "simulated" else "step_budget"]
    budget = f"{int(run['step_budget'])} steps" if run["step_budget"] else f"the default ({default_budget} steps)"
    stall_words = {"end": "ends the run", "log": "only logs, the run goes on"}
    stall = stall_words[run["stall"]] if run["stall"] else f"the default ({stall_words.get(MODEL_MODE_PARAMETERS['stall_action'], MODEL_MODE_PARAMETERS['stall_action'])})"
    files = "1 checkpoint file" if len(run["checkpoints"]) == 1 else f"a team of {len(run['checkpoints'])} checkpoint files"
    lines = [f"Run {run['label']}: " + ("THE REAL ROBOT" if run["part"] == "robot" else "the simulated robot, nothing real moves"),
             f"  sheet        {run['sheet'] or 'not given'}" + (f", lay-out {run['lay_out']}" if run["lay_out"] else ""),
             f"  model        {run['model_key'] or 'not given'} (recipe {run['run_name']}), {files}, sha256 checked:"]
    for path, characters in zip(run["checkpoints"], run["checkpoint_sha256"]):
        # the run folder's name first: it tells the models apart, where the full paths differ only deep inside
        lines += [f"                 {checkpoint_run_folder(path)} / {path.name}  ({characters})", f"                   {path}"]
    lines += [f"  step         {step}",
              f"  step budget  {budget}",
              f"  stall rule   {stall}",
              f"  reference    {run['ghost'] or 'none'}" + (" (laid faintly over the go-ahead window)" if run["ghost"] else ""),
              f"  placing aid  {'yes' if run['is_place_help_wanted'] else 'no'}",
              f"  note         {run['note'] or 'none'}"]
    return "\n".join(lines)



def model_mode_command(
        run: Dict,
        run_folder: Optional[Path] = None
) -> List[str]:
    """
    The command that starts model_mode.py for one checked line, with full paths so that it means the same from any
    folder.

    Parameters
    ----------
    run : dict
        From checked_run.
    run_folder : pathlib.Path, optional
        Log the run there instead of the model mode's run folder (tests and rehearsals).

    Returns
    -------
    list of str
        This Python, model_mode.py, the part, the label and the line's flags; a cell left empty gives no flag, so that
        the default of config.py holds. A robot line also gets --fresh-keys.
    """

    command = [sys.executable, str(CODE_FOLDER / "model_mode.py"), run["part"], "--label", run["label"], "--run-name", run["run_name"],
               "--checkpoint"] + [str(path) for path in run["checkpoints"]] + ["--metrics", str(run["metrics"]), "--step-length", run["step_rule"]]
    for flag, value in (("--step-cm", run["step_cm"]), ("--step-budget", run["step_budget"]), ("--stall", run["stall"]),
                        ("--ghost", str(run["ghost"]) if run["ghost"] else ""), ("--run-folder", str(run_folder) if run_folder else "")):
        if value:
            command += [flag, value]
    if run["is_place_help_wanted"]:
        command.append("--place-help")
    if run["part"] == "robot":
        # an Enter pressed while the model loads must not answer the first question, which switches the motors off;
        # the flag also lets Esc end the run when further keys wait, and holds Ctrl+C back until the robot has answered
        command.append("--fresh-keys")
    return command



def runs_with_label(
        label: str,
        run_folder: Path
) -> List[Path]:
    """
    The run folders ``label`` already has under ``run_folder``; model_mode.py names them "<date>_<time>_<label>".
    Capitals are not told apart, as Windows does not tell them apart in a folder's name.
    """

    if not Path(run_folder).is_dir():
        return []
    name_of_a_run = re.compile(r"\d{4}-\d{2}-\d{2}_\d{6}_" + re.escape(label), re.IGNORECASE)
    return sorted(path for path in Path(run_folder).iterdir() if path.is_dir() and name_of_a_run.fullmatch(path.name))



def discard_typed_keys() -> int:
    """
    Throw away the keys typed in the terminal so far (on Windows), so that the Enter asked for next is one pressed
    after the question is on the screen. model_mode.py has the same helper; it is written here once more because
    importing model_mode.py loads torch and the model code, which this script does without.

    Returns
    -------
    int
        How many key codes were thrown away.
    """

    count = 0
    while msvcrt is not None and msvcrt.kbhit():
        msvcrt.getch()
        count += 1
    return count



def run_in_this_terminal(command: List[str]) -> int:
    """
    Run ``command`` from 02_code in this terminal and wait until it has ended; the keyboard stays with the command.

    Ctrl+C reaches the command itself. This script only goes on waiting and never ends the command by force, so that
    model_mode.py can end its run in its own way (the lift, the motors). That is not certain, though: Ctrl+C that
    arrives while model_mode.py waits for the robot's answer can leave the link to the robot out of step, and the lift
    then waits for ever (12.11.26, measured on a stand-in link, not at the robot). A run is therefore ended with Esc;
    when Ctrl+C is caught here, the person is told once what to do if nothing follows.

    Returns
    -------
    int
        The command's exit code.
    """

    # what this script has printed so far goes out first, so that the command's own lines come after it
    sys.stdout.flush()
    process = subprocess.Popen(command, cwd = str(CODE_FOLDER))
    is_interrupt_told = False
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            if not is_interrupt_told:
                print("\nCtrl+C was passed on to model_mode.py, which ends its run itself. If nothing follows within 10 seconds, the link to the robot is "
                      "stuck:\nclose this window; the arm stays as it is (held by its motors, unless they were switched off for placing).", flush = True)
                is_interrupt_told = True
            continue



def main(argument_list: Optional[List[str]] = None) -> int:
    """
    Check one line of a run list, say what it is, and start it (or, with --dry, print its command).

    Parameters
    ----------
    argument_list : list of str, optional
        The arguments; by default the ones the script was started with.

    Returns
    -------
    int
        0 after a dry run, 1 when the run was not started, else the exit code of model_mode.py. After a run the
        list's line is written into the run's new folder as run_list_line.json.
    """

    parser = argparse.ArgumentParser(description = "Start one run of a visit from the visit's run list.")
    parser.add_argument("run_list", help = "the visit's run list, a CSV file")
    parser.add_argument("label", help = "the label of the line to run")
    parser.add_argument("--dry", action = "store_true", help = "check the line and print the model_mode.py command; run nothing")
    parser.add_argument("--again", action = "store_true", help = "run the label although it already has a run folder")
    parser.add_argument("--run-folder", default = None, metavar = "PATH",
                        help = "look for and log runs in this folder instead of the model mode's run folder (tests and rehearsals)")
    arguments = parser.parse_args(argument_list)
    run_folder = Path(arguments.run_folder).resolve() if arguments.run_folder else Path(MODEL_MODE_PARAMETERS["run_folder"])
    try:
        line = line_with_label(run_list_lines(Path(arguments.run_list)), arguments.label)
        run = checked_run(line)
        earlier_runs = runs_with_label(run["label"], run_folder)
        if earlier_runs and not arguments.again:
            raise RunListProblem(f"the label {run['label']} already has a run folder ({', '.join(path.name for path in earlier_runs)}). To run it once "
                                 "more, add --again (its log gets a new folder beside the old one); else take the next line of the list")
    except RunListProblem as problem:
        print(f"not started: {problem}")
        return 1

    print(plain_words(run))
    if earlier_runs:
        print(f"  again        this label has {len(earlier_runs)} run folder(s) already; --again was given")
    command = model_mode_command(run, run_folder if arguments.run_folder else None)
    if arguments.dry:
        print("dry: nothing is started. The command of this run:")
        print(subprocess.list2cmdline(command))
        return 0
    if run["part"] == "robot":
        print("THE REAL ROBOT moves in this run: the checklist done, nothing else connected to the robot, the stop button in reach.")
    if discard_typed_keys():
        print("keys pressed before this question were ignored")
    try:
        input("Press Enter to start this run (Ctrl+C leaves without starting) ... ")
    except (KeyboardInterrupt, EOFError):
        # no Enter from a person (Ctrl+C, or no keyboard to read from): nothing is started
        print("\nnot started")
        return 1
    print("starting model_mode.py; it needs some seconds to load the model. Press no key until it asks its first question.")
    print(subprocess.list2cmdline(command))
    exit_code = run_in_this_terminal(command)
    if exit_code != 0:
        print(f"model_mode.py ended with exit code {exit_code}: it ended with an error; its last lines say which")
    new_runs = [path for path in runs_with_label(run["label"], run_folder) if path not in earlier_runs]
    if not new_runs:
        print("no run folder: the run was not started")
        return exit_code
    # the list's line beside the run's own log, so that the run stays tied to its sheet, lay-out, model key and note
    try:
        (new_runs[-1] / "run_list_line.json").write_text(json.dumps({"run_list": str(Path(arguments.run_list).resolve()), "line": line, "command": command},
                                                                    indent = 1), encoding = "utf-8")
        print(f"the run's folder: {new_runs[-1]} (the list's line is kept there as run_list_line.json)")
    except OSError as error:
        print(f"the run's folder: {new_runs[-1]} (the list's line could not be written there: {error})")
    return exit_code



if __name__ == "__main__":
    sys.exit(main())
