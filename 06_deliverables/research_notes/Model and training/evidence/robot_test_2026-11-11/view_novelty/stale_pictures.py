"""
Were consecutive robot pictures ever identical, that is, did the model ever decide on a stale camera picture?

The driver waits for two new camera pictures after every move, where "new" means that the JPEG's bytes differ from the
picture before. Camera noise alone changes the bytes, so a picture that still shows the scene from before the move
could pass that test. Checked here, for every pair of consecutive pictures of the ten runs:
    - are the two files byte for byte the same;
    - how much the picture changed (the mean absolute difference of the 160 x 120 grey pictures), against two
      yardsticks from the demonstrations: pairs in which the arm did not move (row 0 and row 1 of every episode: the
      "null move" the audit describes) and pairs in which the tip moved 0.8 to 1.2 cm;
    - how far the tip and the forearm moved between the two pictures (from steps.csv), and the time between them;
    - the sharpness of every picture, since a picture taken during a move would be blurred.

Nothing connects to the robot. Run from this folder (a few seconds):
    D:/AI_Project/.venv/Scripts/python.exe stale_pictures.py
Output: stale_pictures.out.txt, stale_pictures_per_pair.csv
"""

import hashlib
from typing import Dict, Optional

import cv2
import numpy
import pandas

from common import EVIDENCE_FOLDER, ROBOT_RUNS, OutputLog, run_picture_paths, run_small_pictures, run_steps, training_frames, training_pictures

STALE_PARAMETERS = {
    "moving_pair_cm": [0.8, 1.2],
    "demonstration_pairs": 3000,
    "still_quantile": 0.75,
    "seed": 11_11_26
}



def grey_difference(
        first: numpy.ndarray,
        second: numpy.ndarray
) -> float:
    """Return the mean absolute difference of two 160 x 120 RGB pictures after turning them grey (0 to 255)."""

    grey_first = cv2.cvtColor(numpy.ascontiguousarray(first), cv2.COLOR_RGB2GRAY).astype(numpy.int16)
    grey_second = cv2.cvtColor(numpy.ascontiguousarray(second), cv2.COLOR_RGB2GRAY).astype(numpy.int16)
    return float(numpy.abs(grey_first - grey_second).mean())



def demonstration_yardsticks(parameters: Dict) -> Dict[str, numpy.ndarray]:
    """
    Measure how much consecutive demonstration pictures differ when the arm stood still and when the tip moved about 1 cm.

    Returns
    -------
    dict of str to numpy.ndarray
        "still" (row 0 against row 1 of every episode whose row-1 move was under 1 mm and 1 degree... the audit's
        null move) and "moving" (a sample of pairs whose first picture's step was 0.8 to 1.2 cm long).
    """

    frames = training_frames().sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    cache = training_pictures()
    next_same = frames["episode_id"].shift(-1) == frames["episode_id"]
    length_cm = 100 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    # row 0 of every episode is the null move at waypoint 0: the arm stands still until the picture of row 1
    is_still = (frames["row_in_episode"] == 0) & next_same & (length_cm < 0.1) & (frames["forearm_change_deg"].fillna(0.0).abs() < 1.0)
    low, high = parameters["moving_pair_cm"]
    is_moving = next_same & (length_cm >= low) & (length_cm <= high) & (frames["action"] != "stop")
    randomness_generator = numpy.random.default_rng(parameters["seed"])
    moving_positions = numpy.flatnonzero(is_moving.to_numpy())
    moving_positions = numpy.sort(randomness_generator.choice(moving_positions, size = min(parameters["demonstration_pairs"], len(moving_positions)), replace = False))
    result = {}
    for name, positions in (("still", numpy.flatnonzero(is_still.to_numpy())), ("moving", moving_positions)):
        first_index = frames["frame_index"].to_numpy()[positions]
        second_index = frames["frame_index"].to_numpy()[positions + 1]
        result[name] = numpy.array([grey_difference(cache[first], cache[second]) for first, second in zip(first_index, second_index)])
    result["same_md5_rows"] = int(frames["is_same_md5_as_previous"].fillna(0).sum())
    return result



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(STALE_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "stale_pictures.out.txt")
    yardsticks = demonstration_yardsticks(settings)
    still, moving = yardsticks["still"], yardsticks["moving"]
    log.say("CONSECUTIVE PICTURES: how much the 160 x 120 grey picture changes from one picture to the next (mean absolute difference, grey levels 0 to 255)")
    log.say("")
    log.say("Yardsticks from the demonstrations (measured from the training cache)")
    log.say(f"  arm standing still (row 0 to row 1, {len(still)} pairs): 25 % {numpy.quantile(still, 0.25):.2f}, median {numpy.median(still):.2f}, 75 % {numpy.quantile(still, 0.75):.2f}, "
            f"90 % {numpy.quantile(still, 0.9):.2f}, largest {still.max():.2f} (the upper tail is pairs in which something else changed, such as a hand leaving the picture)")
    log.say(f"  tip moved {settings['moving_pair_cm'][0]} to {settings['moving_pair_cm'][1]} cm ({len(moving)} pairs): smallest {moving.min():.2f}, 1 % {numpy.quantile(moving, 0.01):.2f}, "
            f"5 % {numpy.quantile(moving, 0.05):.2f}, median {numpy.median(moving):.2f}")
    log.say(f"  demonstration rows whose JPEG is byte for byte the same as the row before (frames.csv, is_same_md5_as_previous): {yardsticks['same_md5_rows']}")
    # three quarters of the still pairs lie below this; the still pairs above it changed for another reason
    stale_limit = float(numpy.quantile(still, settings["still_quantile"]))
    log.say(f"  share of the moving pairs at or below the still pairs' 75 % point ({stale_limit:.2f}): {100 * (moving <= stale_limit).mean():.2f} %")

    pairs = []
    for run_number in ROBOT_RUNS:
        steps = run_steps(run_number)
        paths = run_picture_paths(run_number)
        small = run_small_pictures(run_number)
        digests = [hashlib.md5(path.read_bytes()).hexdigest() for path in paths]
        for step in range(1, len(paths)):
            pairs.append({"run": run_number, "step": step, "is_same_bytes": digests[step] == digests[step - 1],
                          "grey_difference": grey_difference(small[step - 1], small[step]),
                          "tip_moved_cm": 100 * float(numpy.hypot(steps.iloc[step]["tip_x"] - steps.iloc[step - 1]["tip_x"], steps.iloc[step]["tip_y"] - steps.iloc[step - 1]["tip_y"])),
                          "forearm_turned_deg": float(numpy.degrees(steps.iloc[step]["joint_4"] - steps.iloc[step - 1]["joint_4"])),
                          "seconds_between": float(steps.iloc[step]["unix_time"] - steps.iloc[step - 1]["unix_time"]),
                          "decision_ms": float(steps.iloc[step]["decision_ms"])})
    pairs = pandas.DataFrame(pairs)
    pairs["is_below_still_limit"] = pairs["grey_difference"] <= stale_limit

    log.say("")
    log.say(f"Robot runs of 11.11.26 (a pair counts as 'looks unchanged' when its difference is at or below the still pairs' 75 % point, {stale_limit:.2f})")
    log.say("run  pairs  same bytes  looks unchanged | difference: smallest  median | tip moved cm: smallest  median | seconds between pictures: median  largest | decision ms: median  largest")
    for run_number in ROBOT_RUNS:
        rows = pairs[pairs["run"] == run_number]
        log.say(f"{run_number:3d}  {len(rows):5d}  {int(rows['is_same_bytes'].sum()):10d}  {int(rows['is_below_still_limit'].sum()):15d} | {rows['grey_difference'].min():20.2f}  {rows['grey_difference'].median():6.2f} | "
                f"{rows['tip_moved_cm'].min():22.2f}  {rows['tip_moved_cm'].median():6.2f} | {rows['seconds_between'].median():32.2f}  {rows['seconds_between'].max():7.2f} | "
                f"{rows['decision_ms'].median():19.1f}  {rows['decision_ms'].max():7.1f}")
    log.say("")
    log.say(f"all {len(pairs)} pairs: byte-identical {int(pairs['is_same_bytes'].sum())}; looking unchanged {int(pairs['is_below_still_limit'].sum())}; "
            f"smallest difference {pairs['grey_difference'].min():.2f} (run {int(pairs.loc[pairs['grey_difference'].idxmin(), 'run'])}, step {int(pairs.loc[pairs['grey_difference'].idxmin(), 'step'])}, "
            f"tip moved {pairs.loc[pairs['grey_difference'].idxmin(), 'tip_moved_cm']:.2f} cm)")
    log.say(f"tip travel between consecutive pictures: smallest {pairs['tip_moved_cm'].min():.2f} cm, median {pairs['tip_moved_cm'].median():.2f} cm, largest {pairs['tip_moved_cm'].max():.2f} cm")
    lowest = pairs.nsmallest(8, "grey_difference")
    log.say("")
    log.say("The eight pairs that changed least")
    log.say(lowest[["run", "step", "grey_difference", "tip_moved_cm", "forearm_turned_deg", "seconds_between"]].round(2).to_string(index = False))

    pairs.round(4).to_csv(EVIDENCE_FOLDER / "stale_pictures_per_pair.csv", index = False)
    log.say("")
    log.say("written: stale_pictures_per_pair.csv (sharpness per picture is in setup_light_and_camera_per_picture.csv)")
    log.save()



if __name__ == "__main__":
    main()
