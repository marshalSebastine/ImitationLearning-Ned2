"""
Where the puck sat relative to the magnet tip's line of travel: did the block push through the puck's middle, in the
robot runs as in the demonstrations?

The red puck is found in every picture (puck_detector.py) and placed on the table through the fitted forearm camera
model with the joint angles logged for that picture (puck_tracking.py). Its place is then expressed along the tip's
next step: "ahead" (in front of the tip) and "left" (to the left of the line the tip travels on). A puck that sits to
one side of that line is pushed off-centre and slides away to that side.

The yardstick is the same measurement on pushing rows of the demonstrations of the last recording day, because the
camera had crept on its mount by then about as far as on the robot day (setup_light_and_camera.py: 13.6 px against
16.3 px), so the camera model's sideways error is nearly the same in both (the remaining 2.7 px are about 0.15 cm).

What this measure cannot show: a picture that lags behind the move. The puck travels with the arm, so it looks the
same distance ahead of the tip whether the picture is in step or late; picture_lag_check.py tests the lag with things
that stay on the table.

Nothing connects to the robot. Run from this folder (about a quarter of a minute):
    D:/AI_Project/.venv/Scripts/python.exe puck_beside_tip.py
Output: puck_beside_tip.out.txt, puck_beside_tip_per_picture.csv
"""

from typing import Dict, Optional

import cv2
import numpy
import pandas

from common import EVIDENCE_FOLDER, FAILURE_STEPS, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, run_picture_paths, run_steps, training_frames
from puck_tracking import PuckTracker

PUCK_BESIDE_TIP_PARAMETERS = {
    "demonstration_day": "2026-11-09",
    "demonstration_rows": 700,
    "pushing_step_cm": [0.8, 1.3],
    "pushing_actions": ["forward", "forward_left", "forward_right"],
    "seed": 11_11_26
}



def puck_relative_to_travel(
        puck_xy: numpy.ndarray,
        tip_xy: numpy.ndarray,
        next_tip_xy: numpy.ndarray
) -> Optional[Dict[str, float]]:
    """
    Express the puck's place relative to the tip in the direction the tip travels next.

    Parameters
    ----------
    puck_xy, tip_xy, next_tip_xy : numpy.ndarray
        Table positions in metres: the puck seen in the picture, the tip at the picture, the tip one step later.

    Returns
    -------
    dict or None
        "ahead_cm" (the puck's centre in front of the tip along the travel) and "left_cm" (to the left of the line
        of travel); None when the tip does not move.
    """

    travel = numpy.asarray(next_tip_xy, dtype = float) - numpy.asarray(tip_xy, dtype = float)
    length = float(numpy.linalg.norm(travel))
    if length < 1e-4:
        return None
    direction = travel / length
    offset = numpy.asarray(puck_xy, dtype = float) - numpy.asarray(tip_xy, dtype = float)
    return {"ahead_cm": 100 * float(offset @ direction), "left_cm": 100 * float(direction[0] * offset[1] - direction[1] * offset[0])}



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(PUCK_BESIDE_TIP_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "puck_beside_tip.out.txt")
    tracker = PuckTracker()
    log.say(f"camera model calibrated for the mount in use: {tracker.is_calibrated()}")

    frames = training_frames().sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    next_same = (frames["episode_id"].shift(-1) == frames["episode_id"]).to_numpy()
    length_cm = 100 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0)).to_numpy()
    low, high = settings["pushing_step_cm"]
    is_pushing = (next_same & (length_cm >= low) & (length_cm <= high) & frames["action"].isin(settings["pushing_actions"]).to_numpy()
                  & (frames["session_date"] == settings["demonstration_day"]).to_numpy() & (frames["row_in_episode"] >= 3).to_numpy())
    positions = numpy.flatnonzero(is_pushing)
    positions = numpy.sort(numpy.random.default_rng(settings["seed"]).choice(positions, size = min(settings["demonstration_rows"], len(positions)), replace = False))
    joint_columns = [f"joint_{number}" for number in range(1, 7)]
    demonstrations = []
    for position in positions:
        row, following = frames.iloc[position], frames.iloc[position + 1]
        sighting = tracker.see_puck(cv2.imread(str(VIEW_NOVELTY_PARAMETERS["raw_folder"] / row["frame_path"])), row[joint_columns].to_numpy(dtype = float))
        if sighting is None or sighting["table_xy"] is None or sighting["is_partial"]:
            continue
        relative = puck_relative_to_travel(sighting["table_xy"], [row["tip_x_m"], row["tip_y_m"]], [following["tip_x_m"], following["tip_y_m"]])
        if relative is not None:
            demonstrations.append({"sheet": row["sheet"], **relative})
    demonstrations = pandas.DataFrame(demonstrations)

    robot = []
    for run_number in ROBOT_RUNS:
        steps = run_steps(run_number)
        paths = run_picture_paths(run_number)
        for step in range(len(paths) - 1):
            row, following = steps.iloc[step], steps.iloc[step + 1]
            if row["action"] != "model_step":
                continue
            sighting = tracker.see_puck(cv2.imread(str(paths[step])), row[joint_columns].to_numpy(dtype = float))
            if sighting is None or sighting["table_xy"] is None or sighting["is_partial"]:
                continue
            relative = puck_relative_to_travel(sighting["table_xy"], [row["tip_x"], row["tip_y"]], [following["tip_x"], following["tip_y"]])
            if relative is not None:
                robot.append({"run": run_number, "step": step, **relative, "puck_x_cm": 100 * sighting["table_xy"][0], "puck_y_cm": 100 * sighting["table_xy"][1]})
    robot = pandas.DataFrame(robot)

    log.say("")
    log.say("THE PUCK'S PLACE RELATIVE TO THE TIP, along the tip's next step (cm): 'ahead' = in front of the tip, 'left' = to the left of its line of travel")
    log.say(f"yardstick: {len(demonstrations)} pushing rows of the demonstrations of {settings['demonstration_day']} (steps of {low} to {high} cm, the puck fully in view)")
    log.say(f"  ahead: 5 % {demonstrations['ahead_cm'].quantile(0.05):.2f}, 25 % {demonstrations['ahead_cm'].quantile(0.25):.2f}, median {demonstrations['ahead_cm'].median():.2f}, "
            f"75 % {demonstrations['ahead_cm'].quantile(0.75):.2f}, 95 % {demonstrations['ahead_cm'].quantile(0.95):.2f}")
    log.say(f"  left:  5 % {demonstrations['left_cm'].quantile(0.05):+.2f}, 25 % {demonstrations['left_cm'].quantile(0.25):+.2f}, median {demonstrations['left_cm'].median():+.2f}, "
            f"75 % {demonstrations['left_cm'].quantile(0.75):+.2f}, 95 % {demonstrations['left_cm'].quantile(0.95):+.2f}")
    log.say("")
    log.say("robot runs (the pictures while the puck was still pushed: all pictures of a run that reached the end, the pictures before the puck left the line otherwise)")
    log.say("run  pictures | ahead: 25 %  median  75 % | left: 25 %  median  75 %")
    kept = []
    for run_number in ROBOT_RUNS:
        rows = robot[robot["run"] == run_number]
        if run_number in FAILURE_STEPS:
            rows = rows[rows["step"] < FAILURE_STEPS[run_number]["puck_off_line"]]
        if rows.empty:
            continue
        kept.append(rows)
        log.say(f"{run_number:3d}  {len(rows):8d} | {rows['ahead_cm'].quantile(0.25):11.2f}  {rows['ahead_cm'].median():6.2f}  {rows['ahead_cm'].quantile(0.75):4.2f} | "
                f"{rows['left_cm'].quantile(0.25):+10.2f}  {rows['left_cm'].median():+6.2f}  {rows['left_cm'].quantile(0.75):+5.2f}")
    kept = pandas.concat(kept)
    log.say(f"all these robot pictures ({len(kept)}): ahead median {kept['ahead_cm'].median():.2f} cm (demonstrations {demonstrations['ahead_cm'].median():.2f}); "
            f"left median {kept['left_cm'].median():+.2f} cm (demonstrations {demonstrations['left_cm'].median():+.2f})")

    robot.round(3).to_csv(EVIDENCE_FOLDER / "puck_beside_tip_per_picture.csv", index = False)
    log.say("")
    log.say("written: puck_beside_tip_per_picture.csv")
    log.save()



if __name__ == "__main__":
    main()
