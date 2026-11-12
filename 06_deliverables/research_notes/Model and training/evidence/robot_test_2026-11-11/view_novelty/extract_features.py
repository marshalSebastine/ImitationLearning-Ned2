"""
Read the picture descriptors out of the saved models, for every training picture and every robot picture of 11.11.26.

Two kinds of model are read:
    model_e     the all-sheets model E that pushed in runs 2 to 9. Its descriptors of the 18,522 training pictures and
                of the 554 robot pictures are the main yardstick. Its answers to the robot pictures are compared with
                the answers logged on the robot, which proves that the pictures are prepared here as on the robot.
    fold        a model of the same recipe trained WITHOUT one sheet. Neither that sheet's demonstrations nor the
                robot's pictures of that sheet were ever seen by it, so in its descriptor space the two can be
                compared on equal terms (the fair yardstick).

Nothing connects to the robot. Run from this folder (each call takes a few minutes on two threads):
    D:/AI_Project/.venv/Scripts/python.exe extract_features.py model_e
    D:/AI_Project/.venv/Scripts/python.exe extract_features.py fold without_path_03
    D:/AI_Project/.venv/Scripts/python.exe extract_features.py fold without_path_03 --all-pictures
Output: feature tables in the cache folder (see common.py), and extract_features_<part>.out.txt here.
"""

import argparse
import time
from typing import Dict

import numpy

from common import (EVIDENCE_FOLDER, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, cache_folder, load_policy, picture_descriptors,
                    run_small_pictures, run_steps, training_frames, training_pictures)



def save_descriptors(
        name: str,
        descriptors: Dict[str, numpy.ndarray],
        extra: Dict[str, numpy.ndarray]
) -> None:
    """
    Save a set of descriptors to the cache folder; the large "grid" table in half precision to keep the file small.

    Parameters
    ----------
    name : str
        The file's name without ending.
    descriptors : dict of str to numpy.ndarray
        From ``picture_descriptors``.
    extra : dict of str to numpy.ndarray
        Row labels saved beside them (for example "frame_index", or "run" and "step").
    """

    arrays = dict(descriptors)
    arrays["grid"] = arrays["grid"].astype(numpy.float16)
    arrays.update(extra)
    numpy.savez(cache_folder() / f"{name}.npz", **arrays)



def robot_pictures_and_labels() -> tuple:
    """Return every robot picture of the ten runs at 160 x 120 with its run number and step."""

    pictures, runs, steps = [], [], []
    for run_number in ROBOT_RUNS:
        run_pictures = run_small_pictures(run_number)
        pictures.append(run_pictures)
        runs += [run_number] * len(run_pictures)
        steps += list(range(len(run_pictures)))
    return numpy.concatenate(pictures), numpy.array(runs), numpy.array(steps)



def extract_model_e(log: OutputLog) -> None:
    """Descriptors of model E for all training pictures and all robot pictures, with the check against the robot's log."""

    policy = load_policy(VIEW_NOVELTY_PARAMETERS["model_e_run_name"], VIEW_NOVELTY_PARAMETERS["model_e_folder"])
    log.say(f"model E: {policy.checkpoint_paths[0]}")
    log.say(f"recipe: design {policy.settings['design']}, backbone {policy.settings['backbone']}, pooling {policy.settings['pooling']} "
            f"{policy.settings['pooling_grid']}, picture size {policy.settings['picture_size']}")

    robot_pictures, runs, steps = robot_pictures_and_labels()
    robot = picture_descriptors(policy, robot_pictures)
    save_descriptors("model_e_robot", robot, {"run": runs, "step": steps})
    log.say(f"robot pictures: {len(robot_pictures)}; descriptor sizes: grid {robot['grid'].shape[1]}, mean {robot['mean'].shape[1]}, hidden {robot['hidden'].shape[1]}")

    log.say("")
    log.say("Check: model E's answers computed here against the answers logged on the robot (steps.csv), runs 2 to 9")
    log.say("run  pictures  largest difference: forward cm, left cm, twist deg, stop chance")
    for run_number, run in ROBOT_RUNS.items():
        if run["model"] != "E":
            continue
        logged = run_steps(run_number)
        here = robot["exact"][runs == run_number]
        stop_here = robot["stop_probability"][runs == run_number]
        log.say(f"{run_number:3d}  {len(logged):8d}  {numpy.abs(here[:, 0] - logged['model_tip_forward_cm']).max():.5f}  "
                f"{numpy.abs(here[:, 1] - logged['model_tip_left_cm']).max():.5f}  {numpy.abs(here[:, 2] - logged['model_twist_deg']).max():.5f}  "
                f"{numpy.abs(stop_here - logged['model_stop_probability']).max():.5f}")

    log.say("")
    started = time.perf_counter()
    pictures = training_pictures()
    log.say(f"training pictures: {pictures.shape}")
    training = picture_descriptors(policy, pictures, log)
    save_descriptors("model_e_training", training, {"frame_index": numpy.arange(len(pictures))})
    log.say(f"training descriptors saved after {time.perf_counter() - started:.0f} s")



def extract_fold(
        fold_name: str,
        is_all_pictures: bool,
        log: OutputLog
) -> None:
    """
    Descriptors of the fold model trained without one sheet: for that sheet's demonstration pictures (or all training
    pictures with ``is_all_pictures``) and for all robot pictures.
    """

    sheet = fold_name.replace("without_", "")
    policy = load_policy(VIEW_NOVELTY_PARAMETERS["fold_run_name"], VIEW_NOVELTY_PARAMETERS["fold_folder"], fold_name)
    log.say(f"fold model {fold_name}: {policy.checkpoint_paths[0]}")
    frames = training_frames()
    frame_index = frames["frame_index"].to_numpy() if is_all_pictures else frames.loc[frames["sheet"] == sheet, "frame_index"].to_numpy()
    pictures = training_pictures()
    started = time.perf_counter()
    # fancy indexing of the memory map in ascending order reads only the wanted pictures
    chosen = numpy.asarray(pictures[numpy.sort(frame_index)])
    descriptors = picture_descriptors(policy, chosen, log)
    name = f"fold_{fold_name}_{'all' if is_all_pictures else 'sheet'}"
    save_descriptors(name, descriptors, {"frame_index": numpy.sort(frame_index)})
    log.say(f"{len(chosen)} training pictures ({'all sheets' if is_all_pictures else sheet}) saved as {name}.npz after {time.perf_counter() - started:.0f} s")

    robot_pictures, runs, steps = robot_pictures_and_labels()
    robot = picture_descriptors(policy, robot_pictures)
    save_descriptors(f"fold_{fold_name}_robot", robot, {"run": runs, "step": steps})
    log.say(f"{len(robot_pictures)} robot pictures saved as fold_{fold_name}_robot.npz")



def main() -> None:

    parser = argparse.ArgumentParser(description = "Picture descriptors of the saved models, for the view-novelty check.")
    parser.add_argument("part", choices = ["model_e", "fold"])
    parser.add_argument("fold_name", nargs = "?", default = None, help = "for part fold: without_path_01 to without_path_06")
    parser.add_argument("--all-pictures", action = "store_true", help = "for part fold: all training pictures, not only the held-out sheet's")
    arguments = parser.parse_args()

    if arguments.part == "model_e":
        log = OutputLog(EVIDENCE_FOLDER / "extract_features_model_e.out.txt")
        extract_model_e(log)
    else:
        suffix = "_all" if arguments.all_pictures else ""
        log = OutputLog(EVIDENCE_FOLDER / f"extract_features_fold_{arguments.fold_name}{suffix}.out.txt")
        extract_fold(arguments.fold_name, arguments.all_pictures, log)
    log.save()



if __name__ == "__main__":
    main()
