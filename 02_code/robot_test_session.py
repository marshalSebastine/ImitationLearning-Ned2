"""
A robot test session with Marshal's trained policies, run from one command and without internet (Marshal, 15.11.26).

On the robot's own Wi-Fi hotspot the laptop cannot reach Claude, so this script does every step itself and asks the
person at the robot in plain words:

1. checks that the three policies and the packages are on this computer;
2. waits until the robot answers on the hotspot (join the robot's Wi-Fi first);
3. checks the calibration and, only after a yes, runs the robot's AUTOMATIC calibration (never a manual one);
4. goes through the safety list;
5. runs one policy episode after another: choose the policy and the sheet, place the puck, and record_episodes.py
   --mode policy drives the robot, each step only after Enter in the picture window (g lets it run on);
6. after each episode asks how far the puck got and what went wrong first, and adds the run to the results file.

Nothing moves the robot without the person: the calibration asks first, and every policy step waits for a key in the
picture window. Settings: ROBOT_TEST_SESSION_PARAMETERS in config.py.

Run from the project folder (Mac):
    .venv/bin/python 02_code/robot_test_session.py
"""

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config import ROBOT_TEST_SESSION_PARAMETERS, WORKSPACE_PARAMETERS


CODE_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = CODE_FOLDER.parent
RESULT_COLUMNS = ["Run", "Time", "Policy (rank)", "Sheet", "Start", "Status", "Puck in goal?", "Steps",
                  "How far (cm)", "What went wrong first", "Episode folder"]


def ask(question: str) -> str:
    """Ask in the terminal; Ctrl-D counts as an empty answer."""

    try:
        return input(question).strip()
    except EOFError:
        return ""


def is_robot_reachable(
        ip_address: str,
        port: int,
        timeout_s: float = 3.0
) -> bool:
    """True when the robot's control port answers (nothing is sent to it)."""

    try:
        with socket.create_connection((ip_address, port), timeout = timeout_s):
            return True
    except OSError:
        return False


def missing_files(parameters: Dict) -> List[str]:
    """The policy files that are not on this computer."""

    return [policy["checkpoint"] for policy in parameters["policies"] if not (PROJECT_FOLDER / policy["checkpoint"]).is_file()]


def episode_command(
        policy: Dict,
        path_id: str,
        start_xy: Optional[List[float]],
        lighting: str,
        ip_address: str,
        parameters: Dict,
        is_simulated: bool = False
) -> List[str]:
    """The record_episodes.py command for one policy episode (a list, run without a shell)."""

    command = [sys.executable, str(CODE_FOLDER / "record_episodes.py"), "--operator", parameters["operator"], "--mode", "policy",
               "--checkpoint", str(PROJECT_FOLDER / policy["checkpoint"]), "--path-id", path_id, "--single-episode",
               "--orientation", f"Marshal robot test, rank {policy['rank']} ({policy['name']}), STD",
               "--lighting", lighting, "--robot-ip", ip_address]
    if start_xy is not None:
        command += ["--start-x", str(start_xy[0]), "--start-y", str(start_xy[1])]
    if is_simulated:
        command.append("--simulated")
    return command


def newest_episode_since(
        folders: List[Path],
        since_time: float
) -> Optional[Path]:
    """The episode folder whose episode.json was written last after ``since_time``, in any of ``folders``."""

    found = [path for folder in folders if folder.is_dir() for path in folder.glob("*_session_*/*_episode_*/episode.json")
             if path.stat().st_mtime >= since_time]
    return max(found, key = lambda path: path.stat().st_mtime).parent if found else None


def append_result(
        results_path: Path,
        row: Dict[str, str]
) -> None:
    """Add one run to the results table, writing the file's heading first when it is new."""

    results_path.parent.mkdir(parents = True, exist_ok = True)
    if not results_path.exists():
        heading = ["# Robot test of Marshal's round 2 policies", "",
                   "Written by robot_test_session.py, one row per episode; the notes are the operator's own words.", "",
                   "| " + " | ".join(RESULT_COLUMNS) + " |", "|" + "---|" * len(RESULT_COLUMNS)]
        results_path.write_text("\n".join(heading) + "\n", encoding = "utf-8")
    cells = [str(row.get(column, "")).replace("|", "/").replace("\n", " ") for column in RESULT_COLUMNS]
    with open(results_path, "a", encoding = "utf-8") as file:
        file.write("| " + " | ".join(cells) + " |\n")


def shown_path(path: Path) -> str:
    """A path relative to the project folder when it lies inside it, else the full path."""

    try:
        return str(path.relative_to(PROJECT_FOLDER))
    except ValueError:
        return str(path)


def runs_so_far(results_path: Path) -> int:
    """How many runs the results file already holds."""

    if not results_path.exists():
        return 0
    return sum(1 for line in results_path.read_text(encoding = "utf-8").splitlines() if line.startswith("| ") and not line.startswith("| Run"))


def check_calibration(ip_address: str) -> bool:
    """
    Ask the robot whether it needs calibration; with the person's yes, run its AUTOMATIC calibration.

    Returns True when the robot is calibrated at the end.
    """

    from pyniryo import NiryoRobot

    robot = NiryoRobot(ip_address)
    try:
        is_needed = robot.need_calibration()
    finally:
        robot.close_connection()
    if not is_needed:
        print("  Calibration: OK, not needed.")
        return True
    print("  The robot needs calibration. The AUTOMATIC calibration moves the arm by itself for about a minute.")
    print("  Keep everyone and everything away from the arm, stop button in reach.")
    if ask("  Run the automatic calibration now? (y = yes, anything else = stop here): ").lower() != "y":
        return False
    from calibrate_robot import calibrate_robot
    return bool(calibrate_robot(ip_address))


def confirm_safety() -> bool:
    """Go through the safety list; True only when the person types yes."""

    print("""
SAFETY (the project's rules):
  - The robot's Stop button is in reach. Press it for any contact, cable pull, strange noise or unexpected move.
  - Nobody touches the arm while the motors are on; never push on a held arm.
  - Do NOT touch the camera or its focus ring.
  - Only this laptop is connected to the robot: Emranul's laptop, NiryoStudio and any viewer are disconnected.
  - The robot stays away from the table edge.
""")
    return ask("Type yes when all of this is true: ").lower() == "yes"


def choose(
        title: str,
        options: List[str]
) -> Optional[int]:
    """A numbered menu; returns the chosen index, or None for q."""

    print(f"\n{title}")
    for number, option in enumerate(options, 1):
        print(f"  {number}  {option}")
    while True:
        answer = ask("Number (q = finish the session): ").lower()
        if answer == "q":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print("  Please type one of the numbers, or q.")


def run_session(
        parameters: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        is_simulated: bool = False
) -> None:
    """The whole session, from the checks to the last episode. See the module's description."""

    settings = dict(ROBOT_TEST_SESSION_PARAMETERS)
    settings.update(parameters or {})
    ip_address = ip_address or settings["hotspot_ip_address"]
    results_path = PROJECT_FOLDER / settings["results_file"]
    print("=" * 78 + "\nRobot test of Marshal's round 2 policies" + (" (SIMULATED robot)" if is_simulated else "") + "\n" + "=" * 78)

    print("\nStep 1 of 4: files on this computer")
    missing = missing_files(settings)
    if missing:
        print("  Missing policy files:\n    " + "\n    ".join(missing) + "\n  Stop here and tell Claude when you are back online.")
        return
    print("  All three policies are here.")

    if not is_simulated:
        print(f"\nStep 2 of 4: the robot on its Wi-Fi ({ip_address})")
        while not is_robot_reachable(ip_address, settings["robot_port"]):
            print("  The robot does not answer. Join the robot's own Wi-Fi network on this Mac (its name starts with")
            print("  'NiryoRobot'), make sure Emranul's laptop is NOT connected to the robot, and wait until it is joined.")
            if ask("  Enter = try again, q = stop: ").lower() == "q":
                return
        print("  The robot answers.")
        print("\nStep 3 of 4: calibration")
        if not check_calibration(ip_address):
            print("  Not calibrated: stopping. Restart the robot and run this script again.")
            return
    else:
        print("\nSteps 2 and 3 skipped: simulated robot.")

    print("\nStep 4 of 4: safety")
    if not is_simulated and not confirm_safety():
        print("Stopped: start again when everything on the list is true.")
        return

    lighting = ask("\nThe light today, a few words (Enter = 'ceiling lights'): ") or "ceiling lights"
    print(f"\nResults go to: {results_path}")
    run_number = runs_so_far(results_path)
    policy_options = [f"rank {policy['rank']}: {policy['name']} (validation {policy['validation']:.3f}, unseen tracks {policy['unseen']:.3f})"
                      for policy in settings["policies"]]
    sheet_options = [label for _, label in settings["sheets"]]
    folders = [PROJECT_FOLDER / settings["policy_runs_folder"], PROJECT_FOLDER / "03_data" / "raw_simulated"]

    while True:
        policy_index = choose("Which policy?", policy_options)
        if policy_index is None:
            break
        sheet_index = choose("Which sheet is taped down?", sheet_options)
        if sheet_index is None:
            break
        policy = settings["policies"][policy_index]
        path_id, sheet_label = settings["sheets"][sheet_index]

        default_start = WORKSPACE_PARAMETERS["line_start_xy"]
        typed = ask(f"\nStart point: Enter = the taught start mark {default_start}, or type 'x y' in metres: ")
        start_xy = None
        if typed:
            try:
                start_xy = [float(value) for value in typed.replace(",", " ").split()[:2]]
                assert len(start_xy) == 2
            except (ValueError, AssertionError):
                print("  Not two numbers: using the start mark.")
                start_xy = None

        print(f"""
READY: policy rank {policy['rank']} on sheet {sheet_label}.
  - The sheet's start cap is on the start mark, the puck centred on the start, hands out of the picture.
  - In the PICTURE WINDOW (click it first): Enter at "Put the puck at the start".
  - Then each step shows the policy's proposal under the picture: Enter = do it. Do the first 10 steps one by one.
    g = let it run on, any key = pause, Space = stop by hand (puck >2 cm off the line, or the same wrong push
    again and again), x = discard (only for safety), Esc = quit.
  - At the end the window asks "Puck in the goal?": y or n.""")
        if ask("Enter = start this episode, s = skip: ").lower() == "s":
            continue

        started = time.time()
        command = episode_command(policy, path_id, start_xy, lighting, ip_address, settings, is_simulated)
        exit_code = subprocess.call(command, cwd = str(CODE_FOLDER))
        episode_folder = newest_episode_since(folders, started)
        summary = {}
        if episode_folder is not None:
            try:
                summary = json.loads((episode_folder / "episode.json").read_text(encoding = "utf-8"))
            except (OSError, ValueError):
                summary = {}
        status = summary.get("status") or f"tool ended (code {exit_code})"
        print(f"\nThe episode ended: {status}, puck in goal: {summary.get('outcome')}, {summary.get('steps', 0)} steps.")
        how_far = ask("How far along the line did the puck get, in cm (Enter = skip)? ")
        what_went_wrong = ask("What went wrong first, in a few words (Enter = nothing)? ")

        run_number += 1
        append_result(results_path, {
            "Run": run_number, "Time": datetime.now().strftime("%d.%m %H:%M"), "Policy (rank)": f"{policy['rank']}: {policy['name']}",
            "Sheet": path_id, "Start": "start mark" if start_xy is None else f"{start_xy[0]}, {start_xy[1]}", "Status": status,
            "Puck in goal?": summary.get("outcome") or "", "Steps": summary.get("steps", ""), "How far (cm)": how_far,
            "What went wrong first": what_went_wrong,
            "Episode folder": shown_path(episode_folder) if episode_folder else ""})
        print(f"Saved as run {run_number} in the results file.")

    print(f"""
Session finished. {runs_so_far(results_path)} runs in {results_path}.
Next: disconnect from the robot's Wi-Fi, join the normal internet, and tell Claude "read the robot test".
The pictures of every run are in {PROJECT_FOLDER / settings['policy_runs_folder']}.""")


def main() -> None:
    """Command line: the whole session; --simulated runs it without the robot (for checking the script only)."""

    parser = argparse.ArgumentParser(description = "Robot test of Marshal's round 2 policies, offline.")
    parser.add_argument("--robot-ip", help = "the robot's address (default: its own hotspot, 10.10.10.10)")
    parser.add_argument("--simulated", action = "store_true", help = "use the simulated robot (checking the script only)")
    arguments = parser.parse_args()
    try:
        run_session(ip_address = arguments.robot_ip, is_simulated = arguments.simulated)
    except KeyboardInterrupt:
        print("\nStopped with Ctrl-C. Runs so far are saved in the results file.")


if __name__ == "__main__":
    main()
