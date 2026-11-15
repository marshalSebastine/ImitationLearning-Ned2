"""
Tests of robot_test_session.py without the robot: the command it builds, the results file, the search for the newest
episode, and one whole session with scripted answers and a stand-in for the recording tool. A few seconds.

Run from 02_code:
    python test_robot_test_session.py
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import robot_test_session
from config import ROBOT_TEST_SESSION_PARAMETERS
from robot_test_session import append_result, episode_command, is_robot_reachable, newest_episode_since, run_session, runs_so_far


def test_command_and_reachability() -> None:
    """The command runs the policy mode on the hotspot address with the chosen start; a closed port is unreachable."""

    policy = ROBOT_TEST_SESSION_PARAMETERS["policies"][0]
    command = episode_command(policy, "path_03_two_curve", [0.16, -0.01], "ceiling lights", "10.10.10.10", ROBOT_TEST_SESSION_PARAMETERS)
    text = " ".join(command)
    for part in ("--mode policy", "--robot-ip 10.10.10.10", "--path-id path_03_two_curve", "--start-x 0.16", "--start-y -0.01",
                 "--single-episode", "--operator marshal"):
        assert part in text, (part, text)
    assert "--simulated" not in text and "--start-x" not in " ".join(episode_command(policy, "path_01_straight_line", None, "x", "1.2.3.4",
                                                                                      ROBOT_TEST_SESSION_PARAMETERS))
    assert not is_robot_reachable("127.0.0.1", 9, timeout_s = 0.5)
    help_text = subprocess.run([sys.executable, "record_episodes.py", "--help"], capture_output = True, text = True).stdout
    assert "--robot-ip" in help_text and "policy" in help_text, help_text


def test_results_file_and_newest_episode() -> None:
    """The table gets a heading once and a row per run; the newest episode after a time is found."""

    with tempfile.TemporaryDirectory() as folder:
        results = Path(folder) / "results.md"
        append_result(results, {"Run": 1, "Sheet": "path_03", "What went wrong first": "left | right"})
        append_result(results, {"Run": 2, "Sheet": "path_04"})
        assert runs_so_far(results) == 2, results.read_text()
        assert "left / right" in results.read_text()
        before = time.time() - 1
        for name in ("a_episode_001", "a_episode_002"):
            episode = Path(folder) / "runs" / "2026-11-15_session_01_marshal" / name
            episode.mkdir(parents = True)
            (episode / "episode.json").write_text("{}", encoding = "utf-8")
            time.sleep(0.05)
        assert newest_episode_since([Path(folder) / "runs"], before).name == "a_episode_002"
        assert newest_episode_since([Path(folder) / "runs"], time.time() + 10) is None


def test_whole_session_with_scripted_answers() -> None:
    """Policy 2 on sheet 03, then finish: the stand-in tool's episode is found and written to the results file."""

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        answers = iter(["", "2", "2", "", "", "35", "missed the sharp turn", "q"])
        commands = []

        def stand_in_tool(command, cwd = None):
            commands.append(command)
            episode = folder / "policy_runs" / "2026-11-15_session_01_marshal" / "path_03_two_curve_episode_001"
            episode.mkdir(parents = True)
            (episode / "episode.json").write_text(json.dumps({"status": "completed", "outcome": "puck_not_in_goal", "steps": 41}), encoding = "utf-8")
            return 0

        original_ask, original_call, original_project = robot_test_session.ask, robot_test_session.subprocess.call, robot_test_session.PROJECT_FOLDER
        robot_test_session.ask = lambda question: next(answers)
        robot_test_session.subprocess.call = stand_in_tool
        try:
            settings = {"results_file": str(folder / "results.md"), "policy_runs_folder": str(folder / "policy_runs")}
            run_session(settings, is_simulated = True)
        finally:
            robot_test_session.ask, robot_test_session.subprocess.call = original_ask, original_call
            robot_test_session.PROJECT_FOLDER = original_project
        assert len(commands) == 1 and "factorised_classifier_tuned_seed1" in " ".join(commands[0]), commands
        assert "--simulated" in commands[0]
        text = (folder / "results.md").read_text(encoding = "utf-8")
        assert "| 1 |" in text and "path_03_two_curve" in text and "missed the sharp turn" in text and "| 35 |" in text, text
        assert "puck_not_in_goal" in text and "| 41 |" in text, text


if __name__ == "__main__":
    for test in (test_command_and_reachability, test_results_file_and_newest_episode, test_whole_session_with_scripted_answers):
        test()
        print(f"{test.__name__}: passed", flush = True)
    print("all tests passed")
