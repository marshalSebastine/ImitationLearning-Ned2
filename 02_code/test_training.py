"""
Tests of the training code (training_targets.py, training_metrics.py, train_policy.py's augmentation) on small drawn
tables and pictures, plus one consistency check against the real frames table when it has been built. No robot is used
and nothing in 03_data is changed.

Run from 02_code:
    ..\\.venv\\Scripts\\python.exe test_training.py
"""

from pathlib import Path

import numpy
import pandas
import torch

from config import NO_AUGMENTATION, TRAINING_DATA_PARAMETERS, TRAINING_RUN_PARAMETERS
from train_policy import PictureToMoveModel, augmented_pictures, batch_targets
from training_metrics import angle_difference_deg, direction_metrics, nearest_direction, stop_metrics
from training_targets import (DIRECTION_NAMES, MIRRORED_DIRECTION, chunk_targets, class_weights, direction_targets,
                              episode_folds, exact_step_targets, leave_one_sheet_out_folds, mirrored_targets, previous_frame_indices,
                              previous_rows)


def toy_frames() -> pandas.DataFrame:
    """Two short episodes on two sheets, with small motions at the start and in the middle."""

    rows = []
    for episode_number, (sheet, actions) in enumerate([
            ("path_01", ["small_motion", "forward", "small_motion", "small_motion", "left", "stop"]),
            ("path_02", ["forward_left", "small_motion", "forward", "stop"])]):
        for position, action in enumerate(actions):
            rows.append({"episode_id": f"episode_{episode_number}", "sheet": sheet, "row_in_episode": position, "action": action,
                         "tip_forward_m": 0.01 if action == "forward" else 0.0, "tip_left_m": 0.01 if action == "left" else 0.0,
                         "forearm_change_deg": 5.0 if action == "small_motion" else 0.0, "lean_change_deg": 0.0,
                         "hand_change_deg": 1.0, "tip_x_m": 0.1 + 0.01 * position, "tip_y_m": 0.0, "picture_heading_deg": 0.0})
    return pandas.DataFrame(rows)


def test_small_motion_rules() -> None:

    frames = toy_frames()
    labels, names, is_used = direction_targets(frames, "next_move")
    named = [names[label] if used else None for label, used in zip(labels, is_used)]
    assert named == ["forward", "forward", "left", "left", "left", "stop", "forward_left", "forward", "forward", "stop"], named
    labels, names, is_used = direction_targets(frames, "drop")
    assert int(is_used.sum()) == 6, is_used
    labels, names, is_used = direction_targets(frames, "own_class")
    assert names[-1] == "small_motion" and int(is_used.sum()) == 10


def test_next_move_never_crosses_episodes() -> None:

    frames = toy_frames()
    frames.loc[5, "action"] = "small_motion"
    labels, names, is_used = direction_targets(frames, "next_move")
    assert not is_used[5], "a small motion at an episode's end has no next move and must not borrow the next episode's"


def test_mirror_swaps_left_and_right() -> None:

    names = DIRECTION_NAMES + ["stop"]
    labels = numpy.array([names.index("forward_left"), names.index("right"), names.index("stop"), -1])
    steps = numpy.array([[1.0, 0.5, 4.0, 1.0, 2.0]] * 4, dtype = numpy.float32)
    mirrored_labels, mirrored_steps = mirrored_targets(labels, names, steps)
    assert [names[label] if label >= 0 else None for label in mirrored_labels] == ["forward_right", "left", "stop", None]
    assert mirrored_steps[0].tolist() == [1.0, -0.5, -4.0, 1.0, -2.0]
    assert all(MIRRORED_DIRECTION[MIRRORED_DIRECTION[name]] == name for name in names)


def test_chunk_targets_follow_the_tip_in_the_picture_frame() -> None:

    frames = toy_frames()
    chunks = chunk_targets(frames, 3)
    assert numpy.allclose(chunks[0, :, 0], [1.0, 2.0, 3.0]) and numpy.allclose(chunks[0, :, 1], 0.0), chunks[0]
    assert numpy.allclose(chunks[5], 0.0), "after the stop row the tip stays"
    turned = frames.copy()
    turned["picture_heading_deg"] = 90.0
    assert numpy.allclose(chunk_targets(turned, 1)[0, 0], [0.0, -1.0], atol = 1e-6), "picture turned left: a move along x is to the right"


def test_folds_never_split_an_episode() -> None:

    frames = pandas.concat([toy_frames().assign(episode_id = lambda table, number = number: table["episode_id"] + f"_{number}")
                            for number in range(5)], ignore_index = True)
    for fold in episode_folds(frames, 5, 1):
        train_episodes = set(frames["episode_id"][fold["train"]])
        test_episodes = set(frames["episode_id"][fold["test"]])
        assert not train_episodes & test_episodes
        assert set(frames["sheet"][fold["test"]]) == {"path_01", "path_02"}
    sheet_folds = leave_one_sheet_out_folds(frames)
    assert len(sheet_folds) == 2 and all((fold["train"] ^ fold["test"]).all() for fold in sheet_folds)


def test_previous_rows_stay_in_the_episode() -> None:

    frames = toy_frames()
    assert previous_rows(frames, 2).tolist() == [0, 0, 0, 1, 2, 3, 6, 6, 6, 7]


def test_previous_frame_indices_count_every_picture_and_stay_in_the_episode() -> None:

    frames = pandas.DataFrame({"frame_index": [10, 12, 13, 20, 21], "row_in_episode": [0, 2, 3, 0, 1]})
    assert previous_frame_indices(frames, 1).tolist() == [10, 11, 12, 20, 20]
    assert previous_frame_indices(frames, 2).tolist() == [10, 10, 11, 20, 20]


def test_two_pictures_of_a_row_get_the_same_augmentation() -> None:

    settings = dict(TRAINING_RUN_PARAMETERS)
    generator = numpy.random.default_rng(1)
    picture = torch.from_numpy(generator.integers(0, 256, size = (4, 12, 16, 3), dtype = numpy.uint8))
    two, _ = augmented_pictures(torch.cat([picture, picture], dim = -1), settings, numpy.random.default_rng(5), is_training = True)
    one, _ = augmented_pictures(picture, settings, numpy.random.default_rng(5), is_training = True)
    assert two.shape == (4, 6, 12, 16)
    assert torch.allclose(two[:, :3], two[:, 3:]), "the two pictures were augmented differently"
    assert torch.allclose(two[:, :3], one), "the first picture's augmentation changed when a second picture was added"


def test_six_channel_model_takes_two_pictures() -> None:

    settings = dict(TRAINING_RUN_PARAMETERS, backbone = "small_cnn", is_pretrained = False, picture_count = 2)
    from training_probes import design_outputs
    model = PictureToMoveModel(settings, design_outputs("chunk", len(DIRECTION_NAMES), 5))
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.zeros(2, 6, 120, 160))
    assert outputs["exact"].shape[0] == 2


def test_class_weights_favour_rare_classes() -> None:

    weights = class_weights(numpy.array([0, 0, 0, 1, -1]), 3, 1.0)
    assert weights[1] > weights[0] and weights[2] == 0.0


def test_angle_measures() -> None:

    assert angle_difference_deg(numpy.array([170.0]), numpy.array([-170.0]))[0] == 20.0
    assert [DIRECTION_NAMES[index] for index in nearest_direction(numpy.array([0.0, 29.0, 31.0, 100.0, 179.0, -100.0]))] == \
        ["forward", "forward", "forward_left", "left", "back", "right"]
    result = direction_metrics(numpy.array([0.0, 90.0, 10.0]), numpy.array([20.0, 90.0, 80.0]), numpy.array([True, True, False]))
    assert result["direction_rows"] == 2 and result["within_30_deg"] == 1.0 and result["class_accuracy"] == 1.0


def test_stop_measures_count_early_stops() -> None:

    episode_ids = numpy.array(["a", "a", "a", "b", "b", "b"])
    is_stop = numpy.array([0, 0, 1, 0, 0, 1])
    probability = numpy.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.2])
    distance = numpy.array([5.0, 2.5, 0.0, 6.0, 3.0, 0.0])
    result = stop_metrics(is_stop, probability, distance, episode_ids)
    assert result["episodes_stopped_right"] == 0.5 and result["episodes_stopped_early"] == 0.5


def test_augmentation_off_only_normalises_and_mirror_flips_labels() -> None:

    settings = dict(TRAINING_RUN_PARAMETERS, augmentation = NO_AUGMENTATION)
    pictures = torch.randint(0, 255, (4, 12, 16, 3), dtype = torch.uint8)
    inputs, is_mirrored = augmented_pictures(pictures, settings, numpy.random.default_rng(0), is_training = True)
    mean = torch.tensor(settings["normalisation"]["mean"]).view(1, 3, 1, 1)
    standard_deviation = torch.tensor(settings["normalisation"]["std"]).view(1, 3, 1, 1)
    assert torch.allclose(inputs * standard_deviation + mean, pictures.permute(0, 3, 1, 2).float() / 255.0, atol = 1e-5)
    assert not is_mirrored.any()

    settings["augmentation"] = dict(NO_AUGMENTATION, mirror_probability = 1.0)
    inputs_mirrored, is_mirrored = augmented_pictures(pictures, settings, numpy.random.default_rng(0), is_training = True)
    assert is_mirrored.all() and torch.allclose(inputs_mirrored, inputs.flip(-1), atol = 1e-5)
    names = DIRECTION_NAMES + ["stop"]
    arrays = {"direction": numpy.array([names.index("left")] * 4), "twist": numpy.array([2, 1, 0, 2]),
              "exact": numpy.tile(numpy.array([[1.0, 1.0, 3.0, 0.0, 1.0]], dtype = numpy.float32), (4, 1)),
              "chunk": numpy.tile(numpy.array([[1.0, 0.5, 2.0, 1.0]], dtype = numpy.float32), (4, 1)),
              "stop": numpy.zeros(4, dtype = numpy.float32), "sample_weight": numpy.ones(4, dtype = numpy.float32)}
    mirror_index = numpy.array([names.index(MIRRORED_DIRECTION[name]) for name in names])
    targets = batch_targets(arrays, numpy.arange(4), is_mirrored, mirror_index, torch.ones(len(names)))
    assert all(names[int(label)] == "right" for label in targets["direction"])
    assert targets["twist"].tolist() == [0, 1, 2, 0]
    assert targets["exact"][0].tolist() == [1.0, -1.0, -3.0, 0.0, -1.0] and targets["chunk"][0].tolist() == [1.0, -0.5, 2.0, -1.0]


def test_labels_match_the_recording_tool_on_real_data() -> None:
    """The nearest-direction rule here must give the recording tool's own move names on every moving row."""

    frames_path = Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv"
    if not frames_path.exists():
        print("  skipped: frames.csv not built")
        return
    frames = pandas.read_csv(frames_path, usecols = ["action", "move_angle_deg"])
    moving = frames[frames["action"].isin(DIRECTION_NAMES)]
    predicted = [DIRECTION_NAMES[index] for index in nearest_direction(moving["move_angle_deg"].to_numpy())]
    agreement = float((numpy.array(predicted) == moving["action"].to_numpy()).mean())
    assert agreement > 0.999, agreement


if __name__ == "__main__":
    for test in [test_small_motion_rules, test_next_move_never_crosses_episodes, test_mirror_swaps_left_and_right,
                 test_chunk_targets_follow_the_tip_in_the_picture_frame, test_folds_never_split_an_episode,
                 test_previous_rows_stay_in_the_episode, test_previous_frame_indices_count_every_picture_and_stay_in_the_episode,
                 test_two_pictures_of_a_row_get_the_same_augmentation, test_six_channel_model_takes_two_pictures,
                 test_class_weights_favour_rare_classes, test_angle_measures,
                 test_stop_measures_count_early_stops, test_augmentation_off_only_normalises_and_mirror_flips_labels,
                 test_labels_match_the_recording_tool_on_real_data]:
        test()
        print(f"{test.__name__}: passed")
    print("all tests passed")
