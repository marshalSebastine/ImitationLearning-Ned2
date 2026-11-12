"""Review attack 7: visit_runs.py with lists of every sort. Nothing is started here except "simulated" lines, logged in a temporary folder."""

import builtins
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "D:/AI_Project/.claude/worktrees/training-analysis/02_code")

import visit_runs
from config import MODEL_MODE_PARAMETERS

CHECKPOINT = MODEL_MODE_PARAMETERS["checkpoint_path"]
SECOND = CHECKPOINT.parent.parent.parent / "2026-11-10_resnet18_model_1_all_sheets_seed_2" / "checkpoints" / "all_sheets.pt"
METRICS = MODEL_MODE_PARAMETERS["metrics_path"]
SHA = visit_runs.sha256_of(CHECKPOINT)
SHA_SECOND = visit_runs.sha256_of(SECOND)


def relative(path):

    return os.path.relpath(path, visit_runs.CODE_FOLDER)


def line(**changes):

    cells = {"label": "check_1", "part": "simulated", "sheet": "sheet 03", "lay_out": "STD", "model_key": "E", "run_name": MODEL_MODE_PARAMETERS["run_name"],
             "checkpoints": relative(CHECKPOINT), "checkpoint_sha256": SHA[:12], "metrics": relative(METRICS), "step_rule": "fixed", "step_cm": "0.65",
             "step_budget": "4", "stall": "log", "ghost": "", "place_help": "no", "note": "a test line"}
    cells.update(changes)
    return cells


def refusing_starter(command):

    raise AssertionError("something was started")


def outcome(arguments, starter = refusing_starter, enter = None):

    printed = io.StringIO()
    real = (visit_runs.run_in_this_terminal, builtins.input)
    visit_runs.run_in_this_terminal = starter
    builtins.input = enter or (lambda prompt = "": (_ for _ in ()).throw(AssertionError("Enter was asked for")))
    try:
        with contextlib.redirect_stdout(printed):
            code = visit_runs.main(arguments)
    except BaseException as error:
        return f"RAISED {type(error).__name__}: {str(error)[:200]}", printed.getvalue()
    finally:
        visit_runs.run_in_this_terminal, builtins.input = real
    return code, printed.getvalue()


def write_list(path, lines, delimiter = ",", columns = None, encoding = "utf-8", newline = ""):

    with open(path, "w", newline = newline, encoding = encoding) as file:
        writer = csv.DictWriter(file, fieldnames = columns or visit_runs.RUN_LIST_COLUMNS, delimiter = delimiter, extrasaction = "ignore")
        writer.writeheader()
        writer.writerows(lines)
    return str(path)


def show(name, result):

    code, printed = result
    text = printed.strip().splitlines()
    print(f"--- {name}\n   exit {code}: {text[0][:330] if text else ''}")
    return printed


with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    runs = folder / "runs"
    runs.mkdir()
    flags = ["--dry", "--run-folder", str(runs)]

    print("=== the list file itself")
    path = write_list(folder / "list.csv", [line()])
    show("a good line, dry", outcome([path, "check_1"] + flags))
    for missing in ("label", "checkpoint_sha256", "note", "ghost", "step_cm"):
        columns = [name for name in visit_runs.RUN_LIST_COLUMNS if name != missing]
        show(f"column {missing} missing", outcome([write_list(folder / "short.csv", [line()], columns = columns), "check_1"] + flags))
    show("unknown label", outcome([path, "check_9"] + flags))
    show("label in other letters (Check_1)", outcome([path, "Check_1"] + flags))
    show("label with a space after it in the list", outcome([write_list(folder / "l.csv", [line(label = "check_1 ")]), "check_1"] + flags))
    show("empty file", outcome([str((folder / "empty.csv").write_text("") or folder / "empty.csv"), "check_1"] + flags))
    (folder / "only_header.csv").write_text(",".join(visit_runs.RUN_LIST_COLUMNS) + "\n", encoding = "utf-8")
    show("only the header", outcome([str(folder / "only_header.csv"), "check_1"] + flags))
    show("a folder given as the list", outcome([str(folder), "check_1"] + flags))
    show("header in capitals", outcome([write_list(folder / "caps.csv", [{name.upper(): value for name, value in line().items()}], columns = [name.upper() for name in visit_runs.RUN_LIST_COLUMNS]), "check_1"] + flags))
    show("with a byte order mark and CRLF (as Excel saves 'CSV UTF-8')", outcome([write_list(folder / "bom.csv", [line()], encoding = "utf-8-sig", newline = ""), "check_1"] + flags))
    show("saved as ANSI by Excel with an umlaut in the note", outcome([write_list(folder / "ansi.csv", [line(note = "Blatt 03, drei Laeufe; gruen \u00fc")], encoding = "cp1252"), "check_1"] + flags))
    show("saved as UTF-16 (Excel 'Unicode Text')", outcome([write_list(folder / "utf16.csv", [line()], encoding = "utf-16"), "check_1"] + flags))
    show("tab separated", outcome([write_list(folder / "tab.csv", [line()], delimiter = "\t"), "check_1"] + flags))
    show("semicolon separated with a team (quoted cell)", outcome([write_list(folder / "semi.csv", [line(checkpoints = relative(CHECKPOINT) + ";" + relative(SECOND), checkpoint_sha256 = SHA[:12] + ";" + SHA_SECOND[:12])], delimiter = ";"), "check_1"] + flags))
    # a semicolon list typed by hand: the team's cell not quoted, so its ";" is taken for a cell border
    by_hand = ";".join(visit_runs.RUN_LIST_COLUMNS) + "\n" + ";".join(str(line(checkpoints = relative(CHECKPOINT) + ";" + relative(SECOND), checkpoint_sha256 = SHA[:12] + ";" + SHA_SECOND[:12])[name]) for name in visit_runs.RUN_LIST_COLUMNS) + "\n"
    (folder / "by_hand.csv").write_text(by_hand, encoding = "utf-8")
    show("semicolon list typed by hand, team cell not quoted", outcome([str(folder / "by_hand.csv"), "check_1"] + flags))
    # a comma list typed by hand with a comma in the note
    by_hand = ",".join(visit_runs.RUN_LIST_COLUMNS) + "\n" + ",".join(str(line(note = "sheet 03, second try")[name]) for name in visit_runs.RUN_LIST_COLUMNS) + "\n"
    (folder / "comma_note.csv").write_text(by_hand, encoding = "utf-8")
    printed = show("comma list typed by hand, comma in the note (not quoted)", outcome([str(folder / "comma_note.csv"), "check_1"] + flags))
    print("   note shown:", [text for text in printed.splitlines() if "note" in text])
    by_hand = ",".join(visit_runs.RUN_LIST_COLUMNS) + "\n" + ",".join(str(line(sheet = "sheet 03, turned")[name]) for name in visit_runs.RUN_LIST_COLUMNS) + "\n"
    (folder / "comma_sheet.csv").write_text(by_hand, encoding = "utf-8")
    show("comma list typed by hand, comma in the sheet cell (not quoted)", outcome([str(folder / "comma_sheet.csv"), "check_1"] + flags))
    show("a blank first line", outcome([str((folder / "blank_first.csv").write_text("\n" + Path(path).read_text(encoding = "utf-8"), encoding = "utf-8") and folder / "blank_first.csv"), "check_1"] + flags))
    show("a comment line above the header", outcome([str((folder / "comment.csv").write_text("# visit of 12.11.26\n" + Path(path).read_text(encoding = "utf-8"), encoding = "utf-8") and folder / "comment.csv"), "check_1"] + flags))
    show("spaces after the commas", outcome([str((folder / "spaced.csv").write_text(Path(path).read_text(encoding = "utf-8").replace(",", ", "), encoding = "utf-8") and folder / "spaced.csv"), "check_1"] + flags))
    show("two lines with the same label", outcome([write_list(folder / "twice.csv", [line(), line()]), "check_1"] + flags))

    print()
    print("=== labels")
    for label in ("-start", "--dry", ".", "..", "run.", "a.b", "CON", "nul", "x" * 200, "a b", "a/b", "a\\b", "", "check_1\n"):
        result = outcome([write_list(folder / "label.csv", [line(label = label)]), label] + flags)
        show(f"label {label!r}", result)

    print()
    print("=== checksums")
    for name, characters in (("the whole sha256", SHA), ("in capitals", SHA[:12].upper()), ("13 characters", SHA[:13]), ("11 characters", SHA[:11]), ("65 characters", SHA + "0"),
                             ("wrong last character", SHA[:11] + ("0" if SHA[11] != "0" else "1")), ("with 0x in front", "0x" + SHA[:12]), ("with a space inside", SHA[:6] + " " + SHA[6:12]),
                             ("with an apostrophe in front (spreadsheet text mark)", "'" + SHA[:12]), ("two given for one file", SHA[:12] + ";" + SHA[:12]), ("ending in ';'", SHA[:12] + ";")):
        show(name, outcome([write_list(folder / "sha.csv", [line(checkpoint_sha256 = characters)]), "check_1"] + flags))
    show("team: second checksum wrong", outcome([write_list(folder / "team.csv", [line(checkpoints = relative(CHECKPOINT) + ";" + relative(SECOND), checkpoint_sha256 = SHA[:12] + ";" + SHA[:12])]), "check_1"] + flags))
    show("team: checksums in the wrong order", outcome([write_list(folder / "team.csv", [line(checkpoints = relative(CHECKPOINT) + ";" + relative(SECOND), checkpoint_sha256 = SHA_SECOND[:12] + ";" + SHA[:12])]), "check_1"] + flags))
    show("team: separated by ' ; ' with spaces", outcome([write_list(folder / "team.csv", [line(checkpoints = relative(CHECKPOINT) + " ; " + relative(SECOND), checkpoint_sha256 = SHA[:12] + " ; " + SHA_SECOND[:12])]), "check_1"] + flags))
    show("team: separated by ',' instead of ';'", outcome([write_list(folder / "team.csv", [line(checkpoints = relative(CHECKPOINT) + "," + relative(SECOND), checkpoint_sha256 = SHA[:12] + "," + SHA_SECOND[:12])]), "check_1"] + flags))
    show("the same file twice as a team", outcome([write_list(folder / "team.csv", [line(checkpoints = relative(CHECKPOINT) + ";" + relative(CHECKPOINT), checkpoint_sha256 = SHA[:12] + ";" + SHA[:12])]), "check_1"] + flags))
    show("checkpoint is a folder", outcome([write_list(folder / "dir.csv", [line(checkpoints = relative(CHECKPOINT.parent))]), "check_1"] + flags))
    show("metrics is the checkpoint file (not a json)", outcome([write_list(folder / "m.csv", [line(metrics = relative(CHECKPOINT))]), "check_1"] + flags))
    show("metrics of another run (not checked against the checkpoint)", outcome([write_list(folder / "m.csv", [line(metrics = relative(SECOND.parent.parent / "metrics.json"))]), "check_1"] + flags))

    print()
    print("=== cells")
    for name, changes in (("step_cm 1.0", {"step_cm": "1.0"}), ("step_cm 1,0", {"step_cm": "1,0"}), ("step_cm 0.05", {"step_cm": "0.05"}), ("step_cm nan", {"step_cm": "nan"}),
                          ("step_cm '0.65 cm'", {"step_cm": "0.65 cm"}), ("step_cm 6.5e-1", {"step_cm": "6.5e-1"}), ("step_budget 0", {"step_budget": "0"}),
                          ("step_budget 250.0", {"step_budget": "250.0"}), ("step_budget -5", {"step_budget": "-5"}), ("step_budget 1e2", {"step_budget": "1e2"}),
                          ("step_budget 9999999", {"step_budget": "9999999"}), ("step_rule empty", {"step_rule": ""}), ("step_rule Fixed", {"step_rule": "Fixed"}),
                          ("part Robot (dry)", {"part": "Robot"}), ("part empty", {"part": ""}), ("stall LOG", {"stall": "LOG"}), ("place_help Yes", {"place_help": "Yes"}),
                          ("place_help ja", {"place_help": "ja"}), ("place_help TRUE (spreadsheet)", {"place_help": "TRUE"}), ("place_help 1", {"place_help": "1"}),
                          ("run_name empty", {"run_name": ""}), ("metrics empty", {"metrics": ""}), ("ghost is a folder", {"ghost": "."}),
                          ("predicted with an empty step_cm", {"step_rule": "predicted", "step_cm": ""}), ("predicted with step_cm", {"step_rule": "predicted"})):
        printed = show(name, outcome([write_list(folder / "cell.csv", [line(**changes)]), "check_1"] + flags))
        if "dry: nothing is started" in printed:
            print("   command:", printed.strip().splitlines()[-1][-150:])

    print()
    print("=== paths with spaces, run for real on the simulated robot")
    spaced = folder / "a folder with spaces" / "model one"
    spaced.mkdir(parents = True)
    shutil.copy(CHECKPOINT, spaced / "all sheets.pt")
    shutil.copy(METRICS, spaced / "metrics of it.json")
    ghost = spaced / "first picture.jpg"
    ghost.write_bytes(b"x")
    spaced_runs = folder / "runs with spaces"
    spaced_list = write_list(spaced / "run list.csv", [line(label = "spaces_1", checkpoints = str(spaced / "all sheets.pt"), metrics = str(spaced / "metrics of it.json"), ghost = str(ghost),
                                                             place_help = "yes")])
    printed = show("dry", outcome([spaced_list, "spaces_1", "--dry", "--run-folder", str(spaced_runs)]))
    print("   command:", printed.strip().splitlines()[-1])
    captured = []

    def quiet_starter(command):
        finished = subprocess.run(command, cwd = str(visit_runs.CODE_FOLDER), capture_output = True, text = True, stdin = subprocess.DEVNULL)
        captured.append(finished.stdout[-1500:] + finished.stderr[-1500:])
        return finished.returncode

    result = outcome([spaced_list, "spaces_1", "--run-folder", str(spaced_runs)], starter = quiet_starter, enter = lambda prompt = "": "")
    show("for real (simulated)", result)
    print("   run folders:", [path.name for path in spaced_runs.iterdir()] if spaced_runs.is_dir() else None)
    print("   the run said:", [text for text in captured[0].splitlines() if "note:" in text or "ended" in text or "Error" in text][:5])
    summary_files = list(spaced_runs.glob("*/summary.json")) if spaced_runs.is_dir() else []
    if summary_files:
        summary = json.loads(summary_files[0].read_text(encoding = "utf-8"))
        print("   summary:", {name: summary[name] for name in ("label", "ending", "pictures", "fixed_step_cm", "stall_action", "checkpoints")})
    show("the same label again", outcome([spaced_list, "spaces_1", "--run-folder", str(spaced_runs)]))
    show("the same label again with --again, dry", outcome([spaced_list, "spaces_1", "--again", "--dry", "--run-folder", str(spaced_runs)]))

    print()
    print("=== the label check against run folders")
    (runs / "2026-11-12_101500_check_1").mkdir()
    show("label check_1 with a run folder", outcome([path, "check_1"] + flags))
    for other, expected in (("2026-11-12_101500_check_10", False), ("2026-11-12_101500_xcheck_1", False), ("2026-11-12_1015_check_1", False), ("2026-11-12_101500_check_1 ", True)):
        probe = folder / "probe"
        shutil.rmtree(probe, ignore_errors = True)
        probe.mkdir()
        try:
            (probe / other).mkdir()
        except OSError as error:
            print("   could not make", repr(other), error)
            continue
        print(f"   folder {other!r} (on disk {[item.name for item in probe.iterdir()]}) counts as a run of check_1:", bool(visit_runs.runs_with_label("check_1", probe)))
    probe = folder / "probe_dot"
    probe.mkdir()
    (probe / "2026-11-12_101500_run.").mkdir()
    print("   a run of the label 'run.' is stored as", [item.name for item in probe.iterdir()], "and found again:", bool(visit_runs.runs_with_label("run.", probe)))
    (probe / "2026-11-12_101500_check_1").write_text("a file, not a folder")
    print("   a FILE named like a run folder counts:", bool(visit_runs.runs_with_label("check_1", probe)))
    print("   run folder that is missing:", visit_runs.runs_with_label("check_1", folder / "no such folder"))
