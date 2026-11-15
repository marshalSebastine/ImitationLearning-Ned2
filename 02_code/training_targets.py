"""
Training targets, sample weights and validation splits from the frames table (9.11.26, Model and training session).

Every training script reads the data through ``load_training_frames`` so that all models are compared on exactly the
same pictures. The targets a model can learn, all taken from steps.csv and the arm model (training_dataset.py):
    direction   the move name: one of the 8 picture-frame directions or stop; "small_motion" (tip moved under 3 mm,
                mostly a forearm twist) and the 13 forearm turns are dropped, kept as their own class, or replaced by
                the next real move (``direction_targets``);
    exact       the exact step: tip forward and left in cm in the current picture's frame, and the forearm, lean and
                hand changes in degrees (``exact_step_targets``);
    chunk       the tip's place after each of the next few steps, in the current picture's frame (``chunk_targets``);
    twist       the forearm twist as three classes, right, none or left (``twist_targets``);
    stop        whether the row is the stop row, and whether the tip is within 2 cm of the goal (``stop_targets``).
Splits: leave one sheet out (the stand-in for an unseen track) and whole episodes held out within the sheets.

Settings: TRAINING_TARGET_PARAMETERS in config.py.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy
import pandas

from config import RECORDING_PARAMETERS, SHAPE_OF_SHEET, TRAINING_DATA_PARAMETERS, TRAINING_TARGET_PARAMETERS

DIRECTION_ANGLES_DEG = {name: angle for name, angle in RECORDING_PARAMETERS["actions"].items() if angle is not None}
DIRECTION_NAMES = list(DIRECTION_ANGLES_DEG)
MIRRORED_DIRECTION = {"forward": "forward", "back": "back", "left": "right", "right": "left",
                      "forward_left": "forward_right", "forward_right": "forward_left",
                      "back_left": "back_right", "back_right": "back_left", "stop": "stop", "small_motion": "small_motion"}
FORWARD_ONLY_ACTIONS = ("small_motion", "turn_left", "turn_right")



def merged_target_parameters(parameters: Optional[Dict] = None) -> Dict:
    """Return ``TRAINING_TARGET_PARAMETERS`` with any overrides merged over it."""

    merged = dict(TRAINING_TARGET_PARAMETERS)
    merged.update(parameters or {})
    return merged



def load_training_frames(
        parameters: Optional[Dict] = None,
        data_parameters: Optional[Dict] = None
) -> pandas.DataFrame:
    """
    Read frames.csv and keep the rows a model trains and is evaluated on.

    Parameters
    ----------
    parameters : dict, optional
        Overrides for ``TRAINING_TARGET_PARAMETERS``: "kept_scores", "is_unscored_kept", "is_mistake_dropped",
        "is_flagged_dropped", "is_static_dropped" (with "static_move_m" and "static_turn_deg") and
        "corrected_start_end".
    data_parameters : dict, optional
        Overrides for ``TRAINING_DATA_PARAMETERS`` (the processed folder).

    Returns
    -------
    pandas.DataFrame
        One row per kept picture, sorted by episode and step, with "frame_index" pointing into the picture caches and
        "start_end" corrected where the plan and the tip's path disagree with episode.json.
    """

    parameters = merged_target_parameters(parameters)
    data_parameters = data_parameters or TRAINING_DATA_PARAMETERS
    data_folder = Path(data_parameters["processed_folder"])
    frames = correction_steps(pandas.read_csv(data_folder / "frames.csv", low_memory = False))
    if parameters["is_audit_cleaning"]:
        frames = audit_cleaned_frames(frames, parameters, data_parameters)
    is_kept = frames["score"].isin(parameters["kept_scores"])
    if parameters["is_unscored_kept"]:
        is_kept = is_kept | (frames["is_scored"] == 0)
    if parameters["is_mistake_dropped"]:
        is_kept = is_kept & (frames["is_mistake_row"] == 0)
    if parameters["is_flagged_dropped"]:
        is_kept = is_kept & (frames["is_flagged"] == 0)
    if parameters["is_static_dropped"]:
        # the first row of every episode and the first row after a hand correction: the robot did not move, so the
        # picture is almost the same as the next one, which carries the real move (picture audit, 9.11.26)
        moved_m = numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
        is_static = ((frames["action"] != "stop") & (moved_m < parameters["static_move_m"])
                     & (frames["forearm_change_deg"].fillna(0.0).abs() < parameters["static_turn_deg"]))
        is_kept = is_kept & ~is_static
    frames = frames[is_kept].copy()
    # 9.11.26: episode.json kept the old end for the 19 re-planned episodes of sheets 05 and 06; the plan's end matches
    # where the tip's path went, so the plan's end is used (the audit's cleaning goes further, see audit_cleaned_frames).
    # with the audit's cleaning the start end already comes from its table of corrected ends, which also corrects the
    # plan's end where that is wrong (11 episodes of sheet 02), so the older rule must not overwrite it
    if parameters["corrected_start_end"] == "plan" and not parameters["is_audit_cleaning"]:
        frames["start_end"] = frames["plan_end"].where(frames["plan_end"].isin(["A", "B"]), frames["start_end"])
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    return frames



def correction_steps(
        frames: pandas.DataFrame
) -> pandas.DataFrame:
    """
    Add to every row how many steps it lies before the next correction by hand and after the last one.

    In the exact replay the operator presses h when the puck has drifted: the arm is guided again by hand, the puck is
    put back where it had drifted to, and the robot replays the new path from its first waypoint. So a correction is
    the row inside an episode whose waypoint counter is 0 again. The rows before it show the replay going on while
    the puck drifts; the rows after it show the robot bringing the puck back (15.11.26, Emranul: most episodes have
    such corrections; 305 in the index of 14.11.26).

    Parameters
    ----------
    frames : pandas.DataFrame
        frames.csv as built by training_dataset.py.

    Returns
    -------
    pandas.DataFrame
        The rows sorted by episode and step, with "steps_to_next_correction" (1 for the last row before an h press)
        and "steps_since_correction" (0 for the standing row after it, 1 for the first replayed step); empty where
        the episode has no correction after or before the row.
    """

    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    waypoint = frames["waypoint"] if "waypoint" in frames.columns else pandas.Series(numpy.nan, index = frames.index)
    is_correction = (waypoint.fillna(-1) == 0) & (frames["row_in_episode"] > 0)
    correction_row = frames["row_in_episode"].where(is_correction).groupby(frames["episode_id"], sort = False)
    frames["steps_to_next_correction"] = correction_row.bfill() - frames["row_in_episode"]
    frames["steps_since_correction"] = frames["row_in_episode"] - correction_row.ffill()
    return frames



def audit_cleaned_frames(
        frames: pandas.DataFrame,
        parameters: Dict,
        data_parameters: Dict
) -> pandas.DataFrame:
    """
    Apply the cleaning the data audit of 9.11.26 recommends, before the score and mistake rules.

    Every rule comes from the audit's synthesis (section 11 of the audit file named in "audit_cleaning"):
    the score of an unscored folder is read from its review.json; the start end comes from the audit's table of
    corrected ends; the null moves at waypoint 0 (row 0 of every episode, and the first row after every hand
    correction), the pictures flagged as a sudden change, the listed episodes and the listed rows are dropped; a row
    whose joints hardly differ from the stop row that follows it shows the stop picture and becomes a stop; and the
    few step lengths and forearm changes left by recording stalls are clipped.

    Parameters
    ----------
    frames : pandas.DataFrame
        frames.csv as built by training_dataset.py.
    parameters : dict
        Supplies "audit_cleaning" (the rules' settings).
    data_parameters : dict
        Supplies "raw_folder".

    Returns
    -------
    pandas.DataFrame
        The cleaned rows, with the extra column "is_relabelled_stop".
    """

    rules = parameters["audit_cleaning"]
    frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    raw_folder = Path(data_parameters["raw_folder"])
    for episode_id in frames.loc[frames["is_scored"] == 0, "episode_id"].unique():
        episode_folder = frames.loc[frames["episode_id"] == episode_id, "episode_folder"].iloc[0]
        review_path = raw_folder / episode_folder / "review.json"
        if review_path.exists():
            score = json.loads(review_path.read_text(encoding = "utf-8")).get("score")
            if score is not None:
                frames.loc[frames["episode_id"] == episode_id, ["score", "is_scored"]] = [score, 1]
    corrected_ends = pandas.read_csv(rules["corrected_ends_path"]).set_index("episode_id")["corrected_end"]
    frames["start_end"] = frames["episode_id"].map(corrected_ends).fillna(frames["start_end"])

    joint_columns = [f"joint_{number}" for number in range(1, 7)]
    grouped = frames.groupby("episode_id", sort = False)
    largest_change_to_next_deg = numpy.degrees((grouped[joint_columns].shift(-1) - frames[joint_columns]).abs().max(axis = 1, skipna = False))
    is_before_stop = grouped["action"].shift(-1) == "stop"
    is_relabelled_stop = (largest_change_to_next_deg < rules["static_joint_limit_deg"]) & is_before_stop
    flags = frames["picture_flags"].fillna("")
    is_dropped = (frames["row_in_episode"] == 0) | ((frames["waypoint"].fillna(-1) == 0) & (frames["row_in_episode"] > 0))
    is_dropped |= flags.str.contains("sudden change")
    for pattern in rules["dropped_episode_patterns"]:
        is_dropped |= frames["episode_folder"].str.contains(pattern, regex = False)
    for pattern, first_row, last_row in rules["dropped_row_ranges"]:
        is_dropped |= frames["episode_folder"].str.contains(pattern, regex = False) & frames["row_in_episode"].between(first_row, last_row)
    frames["is_relabelled_stop"] = (is_relabelled_stop & ~is_dropped).astype(int)
    frames.loc[frames["is_relabelled_stop"] == 1, "action"] = "stop"
    frames = frames[~is_dropped].copy()

    # recording stalls left a few very long steps and twists; the picture still shows a normal scene, so the row is
    # kept and only its target is shortened
    length_m = numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    scale = numpy.minimum(1.0, rules["step_clip_m"] / numpy.maximum(length_m, 1e-9))
    frames["tip_forward_m"] = frames["tip_forward_m"] * scale
    frames["tip_left_m"] = frames["tip_left_m"] * scale
    frames["move_length_m"] = numpy.minimum(frames["move_length_m"], rules["step_clip_m"])
    frames["forearm_change_deg"] = frames["forearm_change_deg"].clip(-rules["twist_clip_deg"], rules["twist_clip_deg"])
    return frames



def direction_targets(
        frames: pandas.DataFrame,
        small_motion_rule: str
) -> Tuple[numpy.ndarray, List[str], numpy.ndarray]:
    """
    Return the move-name class of every row.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``.
    small_motion_rule : str
        What a row whose tip hardly moved becomes: "drop" (not used), "own_class" (a class "small_motion") or
        "next_move" (the next row's real move or stop in the same episode, the move the twist was preparing).

    Returns
    -------
    tuple of numpy.ndarray, list of str and numpy.ndarray
        The class index per row, the class names, and whether the row is used.
    """

    actions = frames["action"].to_numpy(dtype = object).copy()
    is_small = numpy.isin(actions, FORWARD_ONLY_ACTIONS)
    class_names = DIRECTION_NAMES + ["stop"]
    is_used = numpy.ones(len(frames), dtype = bool)
    if small_motion_rule == "drop":
        is_used = ~is_small
    elif small_motion_rule == "own_class":
        class_names = class_names + ["small_motion"]
        actions[is_small] = "small_motion"
    elif small_motion_rule == "next_move":
        episode_ids = frames["episode_id"].to_numpy()
        next_real = None
        for position in range(len(frames) - 1, -1, -1):
            if position == len(frames) - 1 or episode_ids[position] != episode_ids[position + 1]:
                next_real = None
            if is_small[position]:
                if next_real is None:
                    is_used[position] = False
                else:
                    actions[position] = next_real
            else:
                next_real = actions[position]
    else:
        raise ValueError(f"unknown small_motion_rule {small_motion_rule}")
    index_of = {name: index for index, name in enumerate(class_names)}
    labels = numpy.array([index_of.get(action, -1) for action in actions], dtype = numpy.int64)
    is_used = is_used & (labels >= 0)
    return labels, class_names, is_used



def exact_step_targets(frames: pandas.DataFrame) -> numpy.ndarray:
    """
    Return the exact step of every row: tip forward and left in cm, forearm, lean and hand changes in degrees.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``.

    Returns
    -------
    numpy.ndarray
        (rows, 5), float32.
    """

    return numpy.stack([
        100.0 * frames["tip_forward_m"].fillna(0.0).to_numpy(),
        100.0 * frames["tip_left_m"].fillna(0.0).to_numpy(),
        frames["forearm_change_deg"].fillna(0.0).to_numpy(),
        frames["lean_change_deg"].fillna(0.0).to_numpy(),
        frames["hand_change_deg"].fillna(0.0).to_numpy()
    ], axis = 1).astype(numpy.float32)



def memory_from_path(
        positions_m: numpy.ndarray,
        forward_axis: numpy.ndarray,
        left_axis: numpy.ndarray,
        distances_cm: List[float]
) -> numpy.ndarray:
    """
    The direction memory of the last place of a path (15.11.26): for each distance, where the tip was that far back
    along the path it travelled, as an arrow from the tip now, in the current picture's frame, and whether the path is
    that long yet.

    The distance is measured along the path, not in steps, so that the hand-guided recordings (about 1 cm a step) and
    the robot (0.65 cm a step) mean the same; between two logged places the point is interpolated.

    Parameters
    ----------
    positions_m : numpy.ndarray
        (places, 2) tip positions on the table in metres, oldest first; the last one is the tip now.
    forward_axis, left_axis : numpy.ndarray
        The picture's forward and left directions on the table, unit vectors.
    distances_cm : list of float
        How far back along the path, for example [2, 5, 10].

    Returns
    -------
    numpy.ndarray
        2 * len(distances_cm) arrow components in cm (forward, left for each distance), then one 1/0 flag per distance;
        an arrow whose distance is longer than the path so far is 0 with flag 0. float32.
    """

    count = len(distances_cm)
    memory = numpy.zeros(3 * count, dtype = numpy.float32)
    positions_m = numpy.asarray(positions_m, dtype = float)
    if len(positions_m) < 2:
        return memory
    backwards = positions_m[::-1]
    lengths = numpy.linalg.norm(numpy.diff(backwards, axis = 0), axis = 1)
    walked = numpy.r_[0.0, numpy.cumsum(lengths)]
    now = backwards[0]
    for index, distance_cm in enumerate(distances_cm):
        distance_m = distance_cm / 100.0
        if walked[-1] < distance_m:
            continue
        segment = int(numpy.searchsorted(walked, distance_m, side = "right")) - 1
        segment = min(segment, len(lengths) - 1)
        share = (distance_m - walked[segment]) / max(lengths[segment], 1e-12)
        place = backwards[segment] + share * (backwards[segment + 1] - backwards[segment])
        arrow = place - now
        memory[2 * index] = 100.0 * float(arrow @ forward_axis)
        memory[2 * index + 1] = 100.0 * float(arrow @ left_axis)
        memory[2 * count + index] = 1.0
    return memory



def memory_arrows(
        frames: pandas.DataFrame,
        distances_cm: List[float]
) -> numpy.ndarray:
    """
    The direction memory of every row (``memory_from_path`` on the episode's tip positions up to that row, in the row's
    picture frame as ``chunk_targets`` turns it).

    Returns
    -------
    numpy.ndarray
        (rows, 3 * len(distances_cm)) float32.
    """

    tip = frames[["tip_x_m", "tip_y_m"]].to_numpy(dtype = float)
    heading = numpy.radians(frames["picture_heading_deg"].to_numpy(dtype = float))
    episode_ids = frames["episode_id"].to_numpy()
    sign = RECORDING_PARAMETERS["picture_left_sign"]
    memory = numpy.zeros((len(frames), 3 * len(distances_cm)), dtype = numpy.float32)
    episode_starts = numpy.flatnonzero(numpy.r_[True, episode_ids[1:] != episode_ids[:-1]])
    episode_ends = numpy.r_[episode_starts[1:], len(frames)]
    for start, end in zip(episode_starts, episode_ends):
        for position in range(start, end):
            forward_axis = numpy.array([math.cos(heading[position]), math.sin(heading[position])])
            left_axis = numpy.array([math.cos(heading[position] + sign * math.pi / 2), math.sin(heading[position] + sign * math.pi / 2)])
            memory[position] = memory_from_path(tip[start:position + 1], forward_axis, left_axis, distances_cm)
    return memory



def chunk_targets(
        frames: pandas.DataFrame,
        steps_ahead: int
) -> numpy.ndarray:
    """
    Return where the magnet tip will be after each of the next ``steps_ahead`` steps, in the current picture's frame.

    The places come from the arm model's tip positions (tip_x_m, tip_y_m) turned into the frame of the current
    picture (picture_heading_deg; forward towards the picture's top, left towards its left). After the episode's last
    row the tip stays where it stopped.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``, sorted by episode and step.
    steps_ahead : int
        How many future steps each row predicts.

    Returns
    -------
    numpy.ndarray
        (rows, steps_ahead, 2) in cm, float32: forward and left of the current tip.
    """

    tip_x = frames["tip_x_m"].to_numpy()
    tip_y = frames["tip_y_m"].to_numpy()
    heading = numpy.radians(frames["picture_heading_deg"].to_numpy())
    episode_ids = frames["episode_id"].to_numpy()
    sign = RECORDING_PARAMETERS["picture_left_sign"]
    targets = numpy.zeros((len(frames), steps_ahead, 2), dtype = numpy.float32)
    episode_starts = numpy.flatnonzero(numpy.r_[True, episode_ids[1:] != episode_ids[:-1]])
    episode_ends = numpy.r_[episode_starts[1:], len(frames)]
    for start, end in zip(episode_starts, episode_ends):
        for position in range(start, end):
            forward_axis = numpy.array([math.cos(heading[position]), math.sin(heading[position])])
            left_axis = numpy.array([math.cos(heading[position] + sign * math.pi / 2), math.sin(heading[position] + sign * math.pi / 2)])
            for ahead in range(1, steps_ahead + 1):
                future = min(position + ahead, end - 1)
                moved = numpy.array([tip_x[future] - tip_x[position], tip_y[future] - tip_y[position]])
                targets[position, ahead - 1] = (100.0 * float(moved @ forward_axis), 100.0 * float(moved @ left_axis))
    return targets



def twist_targets(
        frames: pandas.DataFrame,
        threshold_deg: float
) -> numpy.ndarray:
    """
    Return the forearm twist of every row as a class: 0 twist right, 1 none, 2 twist left.

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``.
    threshold_deg : float
        A forearm change smaller than this is "none".

    Returns
    -------
    numpy.ndarray
        int64 class per row.
    """

    change = frames["forearm_change_deg"].fillna(0.0).to_numpy()
    classes = numpy.ones(len(frames), dtype = numpy.int64)
    classes[change >= threshold_deg] = 2
    classes[change <= -threshold_deg] = 0
    return classes



def stop_targets(frames: pandas.DataFrame) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Return whether each row is the stop row, and whether its tip is within the at-goal distance of the goal."""

    is_stop = (frames["action"] == "stop").to_numpy()
    is_at_goal = (frames["is_at_goal"].fillna(0).to_numpy() > 0)
    return is_stop.astype(numpy.float32), is_at_goal.astype(numpy.float32)



def mirrored_targets(
        labels: numpy.ndarray,
        class_names: List[str],
        exact_steps: numpy.ndarray
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Return the targets that belong to the left-right mirrored picture.

    Mirroring the picture swaps left and right, so the left component of the step and the twist and hand changes
    change sign, and the left and right classes swap; forward, back and stop stay.

    Parameters
    ----------
    labels : numpy.ndarray
        Class indices into ``class_names``.
    class_names : list of str
        The class names.
    exact_steps : numpy.ndarray
        (rows, 5) as from ``exact_step_targets``.

    Returns
    -------
    tuple of numpy.ndarray
        The mirrored class indices and exact steps.
    """

    index_of = {name: index for index, name in enumerate(class_names)}
    mirror_index = numpy.array([index_of[MIRRORED_DIRECTION[name]] for name in class_names])
    mirrored_labels = numpy.where(labels >= 0, mirror_index[numpy.clip(labels, 0, None)], labels)
    mirrored_steps = exact_steps.copy()
    mirrored_steps[:, 1] = -mirrored_steps[:, 1]
    mirrored_steps[:, 2] = -mirrored_steps[:, 2]
    mirrored_steps[:, 4] = -mirrored_steps[:, 4]
    return mirrored_labels, mirrored_steps



def class_weights(
        labels: numpy.ndarray,
        class_count: int,
        power: float
) -> numpy.ndarray:
    """
    Return one loss weight per class, inverse to how often the class occurs, raised to ``power``.

    ``power`` 0 gives equal weights, 1 full inverse frequency, 0.5 the square root between them. The weights are
    scaled so that their mean over the rows is 1.
    """

    counts = numpy.bincount(labels[labels >= 0], minlength = class_count).astype(float)
    weights = numpy.where(counts > 0, (counts.sum() / numpy.maximum(counts, 1.0)) ** power, 0.0)
    row_mean = float((weights[labels[labels >= 0]]).mean()) if (labels >= 0).any() else 1.0
    return (weights / row_mean).astype(numpy.float32)



def score_weights(
        frames: pandas.DataFrame,
        weight_of_score: Dict[int, float]
) -> numpy.ndarray:
    """Return a loss weight per row from its episode's score (an unscored row gets the weight of score 2)."""

    scores = frames["score"].fillna(2).astype(int).to_numpy()
    return numpy.array([weight_of_score.get(int(score), 1.0) for score in scores], dtype = numpy.float32)



def previous_rows(
        frames: pandas.DataFrame,
        lag: int
) -> numpy.ndarray:
    """
    Return, for every row, the position of the row ``lag`` steps earlier in the same episode (or the episode's first row).

    Parameters
    ----------
    frames : pandas.DataFrame
        Sorted by episode and step.
    lag : int
        How many steps back.

    Returns
    -------
    numpy.ndarray
        int64 row positions into ``frames``.
    """

    positions = numpy.arange(len(frames))
    row_in_episode = frames.groupby("episode_id").cumcount().to_numpy()
    return positions - numpy.minimum(row_in_episode, lag)



def previous_frame_indices(
        frames: pandas.DataFrame,
        lag: int
) -> numpy.ndarray:
    """
    Return, for every row, the picture cache's index of the camera picture ``lag`` pictures earlier in the same episode.

    Unlike ``previous_rows``, this counts every picture the camera took, also those the cleaning dropped, because on
    the robot the picture before is always the one taken just before. The cache holds the pictures episode by episode
    in their order (frames.csv, checked 14.11.26), so the earlier picture is ``lag`` places back; an episode's first
    pictures repeat its first one.

    Parameters
    ----------
    frames : pandas.DataFrame
        With "frame_index" and "row_in_episode" (the picture's place in its episode, counting every picture).
    lag : int
        How many pictures back.

    Returns
    -------
    numpy.ndarray
        int64 indices into the picture caches.
    """

    frame_index = frames["frame_index"].to_numpy().astype(numpy.int64)
    row_in_episode = frames["row_in_episode"].to_numpy().astype(numpy.int64)
    return frame_index - numpy.minimum(row_in_episode, lag)



def new_track_shapes(frames: pandas.DataFrame) -> numpy.ndarray:
    """The new track each row's sheet carries ("T03" for both prints of T03), or "" for a course sheet (14.11.26)."""

    shapes = frames["path_id"].map(lambda path_id: str(SHAPE_OF_SHEET.get(path_id, ""))).to_numpy().astype(str)
    return numpy.where(numpy.char.startswith(shapes, "T"), shapes, "")



def new_shape_folds(
        frames: pandas.DataFrame,
        train_track_count: Optional[int] = None,
        seed: int = 0
) -> List[Dict]:
    """
    One fold per new track recorded so far: trained on everything else (round 1 and the other new tracks), tested on
    both prints of that track, a line the model has not seen (14.11.26, the mid-way check of round 2).

    Parameters
    ----------
    frames : pandas.DataFrame
        From ``load_training_frames``.
    train_track_count : int, optional
        Train on round 1 and only this many of the other new tracks, drawn at random per fold (14.11.26, to see whether
        the result still rises with the number of tracks); all the other new tracks when not given.
    seed : int, optional
        Seed of that draw.

    Returns
    -------
    list of dict
        "name", "train" and "test" masks over ``frames``.
    """

    shapes = new_track_shapes(frames)
    names = sorted(set(shapes) - {""})
    folds = []
    for fold_number, shape in enumerate(names):
        is_train = shapes != shape
        if train_track_count is not None:
            others = [name for name in names if name != shape]
            randomness_generator = numpy.random.default_rng([seed, fold_number])
            kept = randomness_generator.choice(others, size = min(train_track_count, len(others)), replace = False)
            is_train = (shapes == "") | numpy.isin(shapes, kept)
        folds.append({"name": f"without_{shape}", "train": is_train, "test": shapes == shape})
    return folds



def round_1_to_new_folds(frames: pandas.DataFrame) -> List[Dict]:
    """
    One fold: trained on round 1 only (the course sheets), tested on every new track, the comparison for
    ``new_shape_folds`` (14.11.26): the same recipe without any round 2 data.
    """

    shapes = new_track_shapes(frames)
    return [{"name": "round_1_only", "train": shapes == "", "test": shapes != ""}]



def leave_one_sheet_out_folds(frames: pandas.DataFrame) -> List[Dict]:
    """
    Return one fold per sheet: train on the other five sheets, test on this one.

    Returns
    -------
    list of dict
        "name" (the held-out sheet), "train" and "test" (boolean masks over ``frames``).
    """

    folds = []
    for sheet in sorted(frames["sheet"].unique()):
        is_test = (frames["sheet"] == sheet).to_numpy()
        folds.append({"name": f"without_{sheet}", "train": ~is_test, "test": is_test})
    return folds



def episode_folds(
        frames: pandas.DataFrame,
        fold_count: int,
        seed: int
) -> List[Dict]:
    """
    Return folds that hold out whole episodes, spread evenly over the sheets.

    Within each sheet the episodes are shuffled with ``seed`` and dealt to the folds in turn, so every fold tests
    every sheet. Pictures of one episode never sit on both sides.

    Returns
    -------
    list of dict
        "name", "train" and "test" masks over ``frames``.
    """

    randomness_generator = numpy.random.default_rng(seed)
    fold_of_episode = {}
    for sheet, group in frames.groupby("sheet"):
        episodes = sorted(group["episode_id"].unique())
        randomness_generator.shuffle(episodes)
        for position, episode_id in enumerate(episodes):
            fold_of_episode[episode_id] = position % fold_count
    episode_fold = frames["episode_id"].map(fold_of_episode).to_numpy()
    return [{"name": f"episodes_fold_{fold}", "train": episode_fold != fold, "test": episode_fold == fold}
            for fold in range(fold_count)]



def inner_validation_mask(
        frames: pandas.DataFrame,
        train_mask: numpy.ndarray,
        share: float,
        seed: int
) -> numpy.ndarray:
    """
    Pick whole episodes from the training part of a fold for early stopping, about ``share`` of them per sheet.

    Returns
    -------
    numpy.ndarray
        Boolean mask over ``frames``: True for the rows held back from training for validation.
    """

    randomness_generator = numpy.random.default_rng(seed)
    is_validation = numpy.zeros(len(frames), dtype = bool)
    training = frames[train_mask]
    for sheet, group in training.groupby("sheet"):
        episodes = sorted(group["episode_id"].unique())
        randomness_generator.shuffle(episodes)
        chosen = episodes[:max(1, int(round(share * len(episodes))))]
        is_validation |= frames["episode_id"].isin(chosen).to_numpy() & train_mask
    return is_validation
