"""
Tests of the training code (training_targets.py, training_metrics.py, train_policy_emranul.py's augmentation) on small drawn
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
from train_policy_emranul import (PictureToMoveModel, augmented_pictures, batch_targets, ended_targets, made_up_end, made_up_ends,
                          mirrored_batch, table_as_paper, turned_pictures, turned_targets)
from training_metrics import angle_difference_deg, direction_metrics, nearest_direction, stop_metrics
from training_targets import (DIRECTION_NAMES, MIRRORED_DIRECTION, chunk_targets, class_weights, direction_targets,
                              episode_folds, exact_step_targets, leave_one_sheet_out_folds, mirrored_targets, new_shape_folds,
                              previous_frame_indices, previous_rows)


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


def test_table_as_paper_repaints_only_the_table_of_the_chosen_pictures() -> None:

    parameters = TRAINING_RUN_PARAMETERS["table_as_paper"]
    table, paper, puck, dark = [0.51, 0.52, 0.46], [0.62, 0.76, 0.84], [0.80, 0.15, 0.15], [0.10, 0.10, 0.12]
    picture = torch.zeros(2, 3, 120, 160)
    for colour, (top, bottom, left, right) in [(table, (0, 120, 0, 160)), (paper, (40, 120, 0, 60)), (puck, (50, 60, 120, 130)),
                                               (dark, (100, 120, 130, 160))]:
        picture[:, :, top:bottom, left:right] = torch.tensor(colour).view(1, 3, 1, 1)
    repainted = table_as_paper(picture.clone(), numpy.array([True, False]), parameters)
    assert torch.allclose(repainted[1], picture[1]), "a picture that was not chosen changed"
    assert torch.allclose(repainted[0, :, 10, 100], torch.tensor(paper), atol = 0.02), "the table did not take the paper's colour"
    for name, (row, column) in {"paper": (80, 30), "puck": (55, 125), "dark": (110, 150)}.items():
        assert torch.allclose(repainted[0, :, row, column], picture[0, :, row, column]), f"the {name} changed"
    left, top, right, bottom = parameters["robot_area_px"]
    assert torch.allclose(repainted[0, :, top:bottom, left:right], picture[0, :, top:bottom, left:right]), "the robot's area changed"

    settings = dict(TRAINING_RUN_PARAMETERS)
    pictures = torch.from_numpy(numpy.random.default_rng(1).integers(0, 256, size = (4, 12, 16, 3), dtype = numpy.uint8))
    before, _ = augmented_pictures(pictures, settings, numpy.random.default_rng(5), is_training = True)
    off = dict(settings, augmentation = dict(settings["augmentation"], table_as_paper_probability = 0.0))
    after, _ = augmented_pictures(pictures, off, numpy.random.default_rng(5), is_training = True)
    assert torch.allclose(before, after), "switched off, the repaint must not change any random number"


def test_made_up_end_erases_the_line_ahead_of_the_puck_and_makes_a_stop() -> None:

    parameters = TRAINING_RUN_PARAMETERS["made_up_end"]
    paper, ink, puck = [0.62, 0.76, 0.84], [0.25, 0.35, 0.50], [0.80, 0.15, 0.15]
    picture = torch.tensor(paper).view(3, 1, 1).repeat(1, 120, 160)
    picture[:, 5:85, 78:83] = torch.tensor(ink).view(3, 1, 1)            # a line straight up, through the puck
    picture[:, 20:25, 10:40] = torch.tensor(ink).view(3, 1, 1)           # another part of the track, far to the left
    picture[:, 53:67, 73:88] = torch.tensor(puck).view(3, 1, 1)          # the puck on the line
    ended = made_up_end(picture, numpy.array([5.0, 0.0]), parameters)    # the demonstration goes on forward (up)
    assert ended is not None
    assert torch.allclose(ended[:, 10:50, 78:83], torch.tensor(paper).view(3, 1, 1), atol = 0.02), "the line ahead of the puck is still there"
    assert torch.allclose(ended[:, 72:85, 78:83], picture[:, 72:85, 78:83]), "the line behind the puck was erased"
    assert torch.allclose(ended[:, 20:25, 10:40], picture[:, 20:25, 10:40]), "another part of the track was erased"
    assert torch.allclose(ended[:, 53:67, 73:88], picture[:, 53:67, 73:88]), "the puck changed"
    assert made_up_end(picture, numpy.array([0.3, 0.0]), parameters) is None, "a picture where the demonstration hardly moves on was changed"
    blank = torch.tensor(paper).view(3, 1, 1).repeat(1, 120, 160)
    assert made_up_end(blank, numpy.array([5.0, 0.0]), parameters) is None, "a picture without a puck was changed"

    pictures = (picture.permute(1, 2, 0) * 255).round().to(torch.uint8).unsqueeze(0).repeat(3, 1, 1, 1)
    arrays = {"chunk": numpy.tile(numpy.array([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0], dtype = numpy.float32), (3, 1)),
              "stop": numpy.array([0.0, 1.0, 0.0], dtype = numpy.float32)}
    settings = dict(TRAINING_RUN_PARAMETERS, made_up_end_probability = 1.0)
    changed, is_ended = made_up_ends(pictures, arrays, numpy.arange(3), settings, numpy.random.default_rng(3))
    assert is_ended.tolist() == [True, False, True], "a real stop picture must stay as it is"
    assert (changed[1] == pictures[1]).all() and not (changed[0] == pictures[0]).all()
    targets = {"exact": torch.ones(3, 5), "chunk": torch.ones(3, 10), "stop": torch.zeros(3), "twist": torch.tensor([0, 2, 2]),
               "direction": torch.tensor([3, 4, 5])}
    stops = ended_targets(targets, is_ended)
    assert stops["stop"].tolist() == [1.0, 0.0, 1.0] and stops["twist"].tolist() == [1, 2, 1] and stops["direction"].tolist() == [-1, 4, -1]
    assert float(stops["chunk"][0].abs().sum()) == 0.0 and float(stops["chunk"][1].sum()) == 10.0 and float(stops["exact"][2].abs().sum()) == 0.0
    assert float(targets["stop"].sum()) == 0.0, "the batch's own targets must not change"


def test_turned_pictures_and_their_steps_turn_the_same_way() -> None:

    settings = dict(TRAINING_RUN_PARAMETERS, augmentation = dict(TRAINING_RUN_PARAMETERS["augmentation"], turn_most_deg = 12.0))
    centre_column, centre_row = settings["augmentation"]["turn_centre_px"]
    pictures = torch.zeros(2, 120, 160, 3, dtype = torch.uint8)
    pictures[:, centre_row - 41:centre_row - 38, centre_column - 1:centre_column + 2] = 255     # a dot 40 px straight ahead of the centre

    class FixedAngles:
        """Gives the angles the test wants in place of random ones."""

        def uniform(self, low, high, count):
            assert (low, high, count) == (-12.0, 12.0, 2)
            return numpy.array([10.0, -10.0])

    turned, angles = turned_pictures(pictures, settings, FixedAngles())
    assert turned.shape == pictures.shape and turned.dtype == torch.uint8 and angles.tolist() == [10.0, -10.0]
    for picture, angle in zip(turned, angles):
        rows, columns = torch.nonzero(picture[:, :, 0] > 127, as_tuple = True)
        forward, left = centre_row - rows.float().mean(), centre_column - columns.float().mean()
        # ahead of the centre by 39.5 px (the dot's middle), turned by the angle: positive goes to the left
        assert abs(float(forward) - 39.5 * numpy.cos(numpy.radians(angle))) < 1.0, "the dot is not where a turn about the centre puts it"
        assert abs(float(left) - 39.5 * numpy.sin(numpy.radians(angle))) < 1.0, "a positive angle must turn the picture to the left"

    targets = {"exact": torch.tensor([[1.0, 0.0, 3.0, 4.0, 5.0], [1.0, 0.0, 3.0, 4.0, 5.0]]),
               "chunk": torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]]), "twist": torch.tensor([0, 2]),
               "direction": torch.tensor([3, 4]), "stop": torch.zeros(2)}
    steps = turned_targets(targets, angles)
    sine, cosine = float(numpy.sin(numpy.radians(10.0))), float(numpy.cos(numpy.radians(10.0)))
    assert torch.allclose(steps["exact"][0], torch.tensor([cosine, sine, 3.0, 4.0, 5.0]), atol = 1e-5), "forward must turn towards the left"
    assert torch.allclose(steps["exact"][1], torch.tensor([cosine, -sine, 3.0, 4.0, 5.0]), atol = 1e-5)
    assert torch.allclose(steps["chunk"][0], torch.tensor([cosine, sine, 2 * cosine, 2 * sine]), atol = 1e-5)
    assert torch.allclose(steps["chunk"][1], torch.tensor([sine, cosine, 2 * sine, 2 * cosine]), atol = 1e-5), "left turned by -10 degrees leans forward"
    assert steps["direction"].tolist() == [-1, -1] and steps["twist"].tolist() == [0, 2]
    assert float(targets["exact"][0, 1]) == 0.0, "the batch's own targets must not change"

    mirrored = mirrored_batch(steps, numpy.array([True, False]))
    assert torch.allclose(mirrored["exact"][0], torch.tensor([cosine, -sine, -3.0, 4.0, -5.0]), atol = 1e-5)
    assert torch.allclose(mirrored["chunk"][0], torch.tensor([cosine, -sine, 2 * cosine, -2 * sine]), atol = 1e-5)
    assert mirrored["twist"].tolist() == [2, 2] and torch.allclose(mirrored["exact"][1], steps["exact"][1])

    usual = dict(TRAINING_RUN_PARAMETERS)
    noise = torch.from_numpy(numpy.random.default_rng(1).integers(0, 256, size = (4, 12, 16, 3), dtype = numpy.uint8))
    before, _ = augmented_pictures(noise, usual, numpy.random.default_rng(5), is_training = True)
    assert usual["augmentation"]["turn_most_deg"] == 0.0 and before.shape == (4, 3, 12, 16)


def test_new_shape_folds_with_fewer_tracks() -> None:

    sheets = ["path_01_straight_line"] * 2 + [f"path_{number}_T{number - 10:02d}_solid" for number in range(11, 17) for _ in range(2)]
    frames = pandas.DataFrame({"path_id": sheets})
    folds = new_shape_folds(frames)
    assert [fold["name"] for fold in folds] == [f"without_T{number:02d}" for number in range(1, 7)]
    assert all(fold["train"].sum() == len(frames) - 2 and fold["test"].sum() == 2 for fold in folds)
    fewer = new_shape_folds(frames, train_track_count = 3, seed = 7)
    for fold, full in zip(fewer, folds):
        assert (fold["test"] == full["test"]).all() and not (fold["train"] & fold["test"]).any()
        assert fold["train"].sum() == 2 + 3 * 2, "round 1 and three other tracks train"
    again = new_shape_folds(frames, train_track_count = 3, seed = 7)
    assert all((first["train"] == second["train"]).all() for first, second in zip(fewer, again)), "the draw must repeat"


def test_mid_paper_stop_pictures_count_more() -> None:

    from training_probes import training_arrays
    frames = toy_frames()
    frames["path_id"] = numpy.where(frames["sheet"] == "path_01", "path_13_T03_solid", "path_11_T01_solid")
    frames["score"] = 2
    frames["goal_distance_cm"] = 0.0
    frames["is_at_goal"] = 0
    target_parameters = {"small_motion_rule": "next_move", "twist_threshold_deg": 2.0, "chunk_steps": 5}
    settings = dict(TRAINING_RUN_PARAMETERS, design = "chunk")
    usual, _, _ = training_arrays(frames, settings, target_parameters)
    heavier, _, _ = training_arrays(frames, dict(settings, mid_paper_stop_weight = 4.0), target_parameters)
    is_stop = (frames["action"] == "stop").to_numpy()
    is_mid_paper = (frames["path_id"] == "path_13_T03_solid").to_numpy()
    assert numpy.allclose(heavier["sample_weight"][is_stop & is_mid_paper], 4.0 * usual["sample_weight"][is_stop & is_mid_paper])
    assert numpy.allclose(heavier["sample_weight"][~(is_stop & is_mid_paper)], usual["sample_weight"][~(is_stop & is_mid_paper)])
    frames["tip_x_m"] = numpy.where(frames["row_in_episode"] >= 4, 0.45, 0.30)
    farther, _, _ = training_arrays(frames, dict(settings, far_reach_weight = 3.0), target_parameters)
    is_far = (frames["tip_x_m"] > 0.40).to_numpy()
    assert is_far.any() and numpy.allclose(farther["sample_weight"][is_far], 3.0 * usual["sample_weight"][is_far]), "far pictures must count more"
    assert numpy.allclose(farther["sample_weight"][~is_far], usual["sample_weight"][~is_far])
    frames["start_kind"] = numpy.where(frames["episode_id"] == "episode_0", "OL1.5", "STD")
    recovering, _, _ = training_arrays(frames, dict(settings, recovery_weight = 5.0, recovery_rows = 2), target_parameters)
    is_recovery = ((frames["episode_id"] == "episode_0") & (frames["row_in_episode"] < 2)).to_numpy()
    assert numpy.allclose(recovering["sample_weight"][is_recovery], 5.0 * usual["sample_weight"][is_recovery]), "recovery pictures must count more"
    assert numpy.allclose(recovering["sample_weight"][~is_recovery], usual["sample_weight"][~is_recovery])
    without, _, _ = training_arrays(frames, dict(settings, weightless_episode_patterns = ["episode_1"]), target_parameters)
    is_left_out = (frames["episode_id"] == "episode_1").to_numpy()
    assert float(numpy.abs(without["sample_weight"][is_left_out]).sum()) == 0.0, "a left-out episode still counts in training"
    assert numpy.allclose(without["sample_weight"][~is_left_out], usual["sample_weight"][~is_left_out])


def test_steps_round_a_correction_by_hand() -> None:

    from training_probes import training_arrays
    from training_targets import correction_steps
    # episode_0: the replay's waypoints 0 to 3, an h press, then the corrected path's waypoints 0 to 1 and the stop
    frames = toy_frames()
    frames = pandas.concat([frames, frames[frames["episode_id"] == "episode_0"].tail(1).assign(row_in_episode = 6)], ignore_index = True)
    frames["waypoint"] = numpy.nan
    frames.loc[frames["episode_id"] == "episode_0", "waypoint"] = [0, 1, 2, 3, 0, 1, numpy.nan]
    frames.loc[frames["episode_id"] == "episode_1", "waypoint"] = [0, 1, 2, numpy.nan]
    marked = correction_steps(frames.sample(frac = 1.0, random_state = 3))
    first = marked[marked["episode_id"] == "episode_0"]
    assert first["row_in_episode"].tolist() == list(range(7)), "the rows must come back in step order"
    assert first["steps_to_next_correction"].fillna(-1).tolist() == [4, 3, 2, 1, 0, -1, -1]
    assert first["steps_since_correction"].fillna(-1).tolist() == [-1, -1, -1, -1, 0, 1, 2]
    second = marked[marked["episode_id"] == "episode_1"]
    assert second["steps_to_next_correction"].isna().all() and second["steps_since_correction"].isna().all(), "no h press, no marks"

    marked["score"] = 2
    marked["goal_distance_cm"] = 0.0
    marked["is_at_goal"] = 0
    target_parameters = {"small_motion_rule": "next_move", "twist_threshold_deg": 2.0, "chunk_steps": 5}
    settings = dict(TRAINING_RUN_PARAMETERS, design = "chunk")
    assert settings["before_correction_weight"] == 1.0 and settings["after_correction_weight"] == 1.0, "both are off unless a run asks"
    usual, _, _ = training_arrays(marked, settings, target_parameters)
    changed, _, _ = training_arrays(marked, dict(settings, before_correction_weight = 0.0, before_correction_rows = 2,
                                                 after_correction_weight = 3.0, after_correction_rows = 1), target_parameters)
    is_first = (marked["episode_id"] == "episode_0").to_numpy()
    row = marked["row_in_episode"].to_numpy()
    is_before = is_first & numpy.isin(row, [2, 3])
    is_after = is_first & (row == 5)
    assert float(numpy.abs(changed["sample_weight"][is_before]).sum()) == 0.0, "the steps before an h press must not count"
    assert numpy.allclose(changed["sample_weight"][is_after], 3.0 * usual["sample_weight"][is_after]), "the steps after must count more"
    assert numpy.allclose(changed["sample_weight"][~(is_before | is_after)], usual["sample_weight"][~(is_before | is_after)])
    marked["steps_to_end"] = marked.groupby("episode_id")["row_in_episode"].transform("max") - marked["row_in_episode"]
    marked["score"] = numpy.where(marked["episode_id"] == "episode_0", 3, 2)
    approach, _, _ = training_arrays(marked, dict(settings, approach_weight = 2.0, approach_rows = 2, approach_scores = [3]), target_parameters)
    is_approach = ((marked["episode_id"] == "episode_0") & marked["steps_to_end"].between(1, 2)).to_numpy()
    assert numpy.allclose(approach["sample_weight"][is_approach], 2.0 * usual["sample_weight"][is_approach]), "the approach must count more"
    assert numpy.allclose(approach["sample_weight"][~is_approach], usual["sample_weight"][~is_approach]), "only score-3 approaches, not the stop row"


def test_direction_memory_points_back_along_the_path() -> None:

    from training_targets import memory_from_path
    from train_policy_emranul import memory_batch
    straight = numpy.array([[0.10 + 0.01 * index, 0.0] for index in range(13)])
    forward, left = numpy.array([1.0, 0.0]), numpy.array([0.0, 1.0])
    assert numpy.allclose(memory_from_path(straight, forward, left, [2, 5, 10]), [-2, 0, -5, 0, -10, 0, 1, 1, 1], atol = 1e-4)
    assert numpy.allclose(memory_from_path(straight[:4], forward, left, [2, 5, 10]), [-2, 0, 0, 0, 0, 0, 1, 0, 0], atol = 1e-4), "a path too short must say unknown"
    # a quarter circle from heading +x to heading +y (a turn to the left): seen from its end, facing +y with -x on the
    # left, the place 10 cm back lies behind (forward 100 * (0.0159 - 0.1) = -8.4 cm) and on the left (4.6 cm)
    turn = numpy.array([[0.1 * numpy.sin(angle), 0.1 - 0.1 * numpy.cos(angle)] for angle in numpy.linspace(0.0, numpy.pi / 2, 300)])
    memory = memory_from_path(turn, numpy.array([0.0, 1.0]), numpy.array([-1.0, 0.0]), [10])
    assert numpy.allclose(memory, [-8.41, 4.60, 1.0], atol = 0.05), memory
    arrays = {"memory": numpy.array([[-2, 1, -5, 2, -10, 3, 1, 1, 1], [-2, 1, 0, 0, 0, 0, 1, 0, 0]], dtype = numpy.float32)}
    settings = {"memory_noise_cm": 0.0, "memory_hide_probability": 0.0}
    mirrored = memory_batch(arrays, numpy.array([0, 1]), numpy.array([True, False]), settings).numpy()
    assert numpy.allclose(mirrored[0], [-2, -1, -5, -2, -10, -3, 1, 1, 1]) and numpy.allclose(mirrored[1], arrays["memory"][1])
    hidden = memory_batch(arrays, numpy.array([0, 1] * 500), numpy.zeros(1000, dtype = bool), dict(settings, memory_hide_probability = 0.3),
                          numpy.random.default_rng(1)).numpy()
    assert 0.25 < (numpy.abs(hidden).sum(axis = 1) == 0).mean() < 0.35, "about 30 % must be hidden"


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
                 test_table_as_paper_repaints_only_the_table_of_the_chosen_pictures,
                 test_made_up_end_erases_the_line_ahead_of_the_puck_and_makes_a_stop,
                 test_turned_pictures_and_their_steps_turn_the_same_way, test_new_shape_folds_with_fewer_tracks,
                 test_mid_paper_stop_pictures_count_more, test_steps_round_a_correction_by_hand, test_direction_memory_points_back_along_the_path,
                 test_class_weights_favour_rare_classes, test_angle_measures,
                 test_stop_measures_count_early_stops, test_augmentation_off_only_normalises_and_mirror_flips_labels,
                 test_labels_match_the_recording_tool_on_real_data]:
        test()
        print(f"{test.__name__}: passed")
    print("all tests passed")
