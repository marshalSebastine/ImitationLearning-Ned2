"""
Tests of policy_data.py, policy_models.py, train_policy.py and tune_policy.py on a tiny made-up dataset of drawn
pictures, on the CPU, without ImageNet weights; no robot and no recorded episode is used. About a minute.

Run from 02_code:
    python test_policy_training.py
"""

import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Dict

import cv2
import numpy
import torch

from build_dataset import build_dataset
from config import TRAINING_PARAMETERS
from policy_data import PolicyData, picture_batch
from policy_models import POLICY_CLASSES
from train_policy import forearm_from_exact, moves_from_exact, train_one_run, write_summary
from tune_policy import tune


MOVES = ["forward", "forward_left", "forward", "forward_right", "left", "forward", "right", "back", "back_left", "forward", "back_right", "stop"]


def write_made_up_raw_folder(raw_folder: Path) -> None:
    """Two sheets, five episodes each, twelve steps each: drawn pictures with the puck moving, and a steps.csv."""

    angles = TRAINING_PARAMETERS["action_angles_deg"]
    columns = ["step", "frame", "frame_md5", "sharpness", "action", "source", "is_mistake", "is_lost", "forearm_label",
               "lean_label", "hand_label", "is_at_goal", "goal_distance_cm", "tip_forward_m", "tip_left_m", "forearm_change_deg",
               "lean_change_deg", "hand_change_deg", "rod_lean_deg", "forearm_turn_deg", "picture_heading_deg"]
    for sheet in ("path_01_straight_line", "path_04_meander"):
        for number in range(1, 6):
            episode_folder = raw_folder / "2026-11-08_session_01_emranul" / f"{sheet}_episode_{number:03d}_plan_x_try_1_score_03"
            (episode_folder / "frames").mkdir(parents = True)
            rows = []
            for step, move in enumerate(MOVES):
                picture = numpy.full((480, 640, 3), (120 + 10 * number, 150, 175), dtype = numpy.uint8)
                cv2.line(picture, (320, 480), (320 + 30 * number, 0), (60, 60, 60), 12)
                cv2.circle(picture, (300 + 5 * step, 300 - 3 * step), 30, (40, 40, 200), -1)
                path = episode_folder / "frames" / f"{step:03d}.jpg"
                cv2.imwrite(str(path), picture)
                angle = math.radians(angles.get(move, 0))
                is_stop = move == "stop"
                turn = 4.0 if move == "left" else (-4.0 if move == "right" else 0.0)
                rows.append({"step": step, "frame": f"frames/{step:03d}.jpg", "frame_md5": hashlib.md5(path.read_bytes()).hexdigest(),
                             "sharpness": 150, "action": move, "source": "replay_exact", "is_mistake": 0, "is_lost": 0,
                             "forearm_label": "" if is_stop else ("turn_left" if turn > 0 else ("turn_right" if turn < 0 else "none")),
                             "lean_label": "" if is_stop else "none", "hand_label": "none", "is_at_goal": int(step >= 10),
                             "goal_distance_cm": max(0, 11 - step), "tip_forward_m": 0.0 if is_stop else round(0.01 * math.cos(angle), 5),
                             "tip_left_m": 0.0 if is_stop else round(0.01 * math.sin(angle), 5), "forearm_change_deg": turn,
                             "lean_change_deg": 0.0, "hand_change_deg": 0.0, "rod_lean_deg": 15.0, "forearm_turn_deg": 0.0,
                             "picture_heading_deg": 0.0})
            with open(episode_folder / "steps.csv", "w", encoding = "utf-8", newline = "") as file:
                writer = csv.DictWriter(file, fieldnames = columns)
                writer.writeheader()
                writer.writerows(rows)
            (episode_folder / "episode.json").write_text(json.dumps({"status": "completed", "operator": "emranul", "plan_number": "x",
                                                                     "mode": "hand_guided_exact", "lighting_note": "ceiling lights"}),
                                                         encoding = "utf-8")
            (episode_folder / "review.json").write_text(json.dumps({"score": 3}), encoding = "utf-8")


def tiny_settings(folder: Path) -> Dict:
    """Training settings for the made-up dataset: small pictures, CPU, one short epoch, no ImageNet download."""

    return {"processed_folder": folder / "processed", "raw_folder": folder / "raw", "runs_folder": folder / "runs",
            "dataset_name": "tiny", "cache_size": [64, 48], "crop_size": [56, 40], "is_pretrained": False,
            "batch_size": 8, "evaluation_batch_size": 16, "epochs": 1, "max_batches_per_epoch": 2, "is_pictures_on_device": False}


def prepared_folder(folder: Path) -> Dict:
    """Write the made-up raw folder, build its dataset and return the tiny training settings."""

    write_made_up_raw_folder(folder / "raw")
    build_dataset({"raw_folder": folder / "raw", "processed_folder": folder / "processed", "dataset_name": "tiny",
                   "excluded_episodes": {}, "excluded_pictures": {}})
    return tiny_settings(folder)


def test_data_links_history_and_chunks() -> None:
    """History indices point back within the episode, chunks forward and past the end as -1; the batch has the crop size."""

    with tempfile.TemporaryDirectory() as folder:
        settings = prepared_folder(Path(folder))
        data = PolicyData(settings, chunk_length = 4, history_length = 3)
        assert data.count == 2 * 5 * len(MOVES), data.count
        first = 0
        assert list(data.history[first]) == [first, first, first], data.history[first]
        assert list(data.history[first + 2]) == [first + 2, first + 1, first], data.history[first + 2]
        last = len(MOVES) - 1
        assert list(data.chunk[last - 1]) == [last - 1, last, -1, -1], data.chunk[last - 1]
        assert data.previous_move[first] == len(TRAINING_PARAMETERS["move_classes"])
        assert (Path(folder) / "processed" / "tiny" / "pictures_64x48.npy").exists()
        batch = picture_batch(torch.from_numpy(data.pictures), torch.tensor([0, 1, 2]), True, data.parameters, torch.device("cpu"))
        assert tuple(batch.shape) == (3, 3, 40, 56), batch.shape
        weights = data.move_class_weights()
        assert weights[TRAINING_PARAMETERS["move_classes"].index("stop")] > weights[TRAINING_PARAMETERS["move_classes"].index("forward")]


def test_exact_steps_are_named_like_the_recording_tool() -> None:
    """A step 60 degrees left is forward_left, straight back is back, stop wins; a forearm change of 4 degrees is turn_left."""

    parameters = dict(TRAINING_PARAMETERS)
    classes = parameters["move_classes"]
    steps = torch.tensor([[0.01, 0.0], [0.005, 0.0087], [-0.01, 0.0], [0.0, -0.01], [0.01, 0.0]])
    is_stop = torch.tensor([False, False, False, False, True])
    names = [classes[index] for index in moves_from_exact(steps, is_stop, parameters).tolist()]
    assert names == ["forward", "forward_left", "back", "right", "stop"], names
    labels = [parameters["forearm_classes"][index] for index in forearm_from_exact(torch.tensor([4.0, 1.0, -5.0]), parameters).tolist()]
    assert labels == ["turn_left", "none", "turn_right"], labels


def test_every_policy_trains_and_writes_its_run() -> None:
    """Each of the five policies trains one short epoch, keeps a checkpoint and writes metrics; the summary lists all five."""

    with tempfile.TemporaryDirectory() as folder:
        settings = prepared_folder(Path(folder))
        for model_name in POLICY_CLASSES:
            overrides = {"chunk_length": 4} if model_name in ("chunk_transformer", "diffusion_policy") else {}
            if model_name == "chunk_transformer":
                overrides.update({"model_width": 32, "decoder_layers": 1, "heads": 2})
            if model_name == "diffusion_policy":
                overrides.update({"hidden_width": 32, "residual_blocks": 1, "diffusion_steps": 10, "sampling_steps": 3})
            training = dict(settings)
            training.update({"epochs": 1, "patience_epochs": 5})
            result = train_one_run(model_name, 1, "test_batch", training_overrides = training, model_overrides = overrides, device_name = "cpu")
            assert result["status"] == "done" and 0.0 <= result["validation"]["hard_sheet_score"] <= 1.0, result
            assert "test" in result and set(result["validation"]["per_sheet"]) <= {"path_01", "path_04"}, result["validation"]
            run_folder = Path(folder) / "runs" / f"test_batch_{model_name}_seed1"
            assert (run_folder / "checkpoints" / "best.pt").exists() and (run_folder / "history.csv").exists()
        again = train_one_run("factorised_classifier", 1, "test_batch", training_overrides = settings, device_name = "cpu", is_done_skipped = True)
        assert again["status"] == "done"
        summary = write_summary("test_batch", settings).read_text(encoding = "utf-8")
        assert all(model_name in summary for model_name in POLICY_CLASSES), summary


def test_held_out_sheet_is_scored_alone() -> None:
    """With a held-out sheet, training and validation hold none of its pictures, and it gets its own score."""

    with tempfile.TemporaryDirectory() as folder:
        settings = prepared_folder(Path(folder))
        settings["held_out_sheet"] = "path_04"
        result = train_one_run("factorised_classifier", 1, "held_out", training_overrides = settings, device_name = "cpu")
        assert set(result["validation"]["per_sheet"]) == {"path_01"}, result["validation"]["per_sheet"]
        assert set(result["held_out"]["per_sheet"]) == {"path_04"}, result["held_out"]["per_sheet"]


def test_optuna_search_runs_and_trains_the_best() -> None:
    """Two short trials, then the best settings trained in full; trial weights are deleted, the tuned run's are kept."""

    with tempfile.TemporaryDirectory() as folder:
        settings = prepared_folder(Path(folder))
        result = tune("tune_test", {"trials": 2, "hours": 0.1, "epochs": 1, "startup_trials": 5}, settings, device_name = "cpu")
        assert result["trials_finished"] >= 1 and result["tuned"]["status"] == "done", result
        runs = Path(folder) / "runs"
        assert not (runs / "tune_test_optuna_trial_000_seed1" / "checkpoints" / "best.pt").exists()
        assert (runs / "tune_test_factorised_classifier_tuned_seed1" / "checkpoints" / "best.pt").exists()
        assert (runs / "tune_test_optuna.md").exists() and (runs / "tune_test_optuna.db").exists()


if __name__ == "__main__":
    for test in (test_data_links_history_and_chunks, test_exact_steps_are_named_like_the_recording_tool,
                 test_every_policy_trains_and_writes_its_run, test_held_out_sheet_is_scored_alone,
                 test_optuna_search_runs_and_trains_the_best):
        test()
        print(f"{test.__name__}: passed", flush = True)
    print("all tests passed")
