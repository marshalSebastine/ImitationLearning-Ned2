"""
Was the camera, in the robot runs, in places and at angles the demonstrations had it in? A check of the view's
geometry that needs no picture.

The driver holds the wrist at 13.5 degrees and the hand at -6.1 degrees and lets joints 1 to 3 carry the tip, while the
demonstrators moved every joint. From the joint angles alone (the fitted forearm camera model and the arm model) this
script works out, for every training picture and every robot picture:
    - the camera's height above the table and its tilt (the angle between its viewing direction and straight down);
    - the roll of the picture (how far the picture's right-hand axis leans out of the horizontal);
    - where the magnet tip falls in the picture (the pixel the model always sees the block at);
    - the wrist and hand angles themselves, and the forearm twist.
The robot pictures are placed in the demonstrations' spread of each quantity.

Nothing connects to the robot. Run from this folder (about half a minute):
    D:/AI_Project/.venv/Scripts/python.exe camera_pose_check.py
Output: camera_pose_check.out.txt, camera_pose_check_robot.csv
"""

import math
from typing import Dict, Optional

import numpy
import pandas

from common import EVIDENCE_FOLDER, FAILURE_STEPS, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, run_steps, trained_frames
from config import table_height
from forearm_camera_model import camera_pose_in_base, project_points
from side_view_test import magnet_tip_and_lean

POSE_PARAMETERS = {
    "quantiles": [0.01, 0.05, 0.5, 0.95, 0.99],
    "outside_quantiles": [0.01, 0.99]
}



def view_geometry(joints: numpy.ndarray) -> Dict[str, float]:
    """
    Work out the camera's place and the tip's place in the picture from one set of joint angles.

    Parameters
    ----------
    joints : numpy.ndarray
        The six joint angles in radians.

    Returns
    -------
    dict
        "camera_height_cm" (above the table under it), "tilt_deg" (viewing direction from straight down), "roll_deg"
        (the picture's right-hand axis out of the horizontal), "tip_u_px" and "tip_v_px" (the magnet tip in the
        640 x 480 picture), "wrist_deg", "hand_deg", "forearm_twist_deg" and "reach_cm" (the tip's distance from the
        base axis).
    """

    position, rotation = camera_pose_in_base(joints)
    tip = numpy.asarray(magnet_tip_and_lean(list(joints))[0], dtype = float)
    pixel = project_points(tip, joints)
    pixel = numpy.asarray(pixel, dtype = float).reshape(-1, 2)[0]
    return {"camera_height_cm": 100 * float(position[2] - table_height(position[0], position[1])),
            "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, -float(rotation[2, 2]))))),
            "roll_deg": math.degrees(math.asin(max(-1.0, min(1.0, float(rotation[2, 0]))))),
            "tip_u_px": float(pixel[0]), "tip_v_px": float(pixel[1]),
            "wrist_deg": math.degrees(joints[4]), "hand_deg": math.degrees(joints[5]), "forearm_twist_deg": math.degrees(joints[3]),
            "reach_cm": 100 * float(numpy.hypot(tip[0], tip[1]))}



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(POSE_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "camera_pose_check.out.txt")
    joint_columns = [f"joint_{number}" for number in range(1, 7)]
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    demonstrations = pandas.DataFrame([view_geometry(joints) for joints in trained[joint_columns].to_numpy(dtype = float)])
    demonstrations["sheet"] = trained["sheet"].to_numpy()

    robot_rows = []
    for run_number in ROBOT_RUNS:
        steps = run_steps(run_number)
        for step, joints in enumerate(steps[joint_columns].to_numpy(dtype = float)):
            robot_rows.append({"run": run_number, "step": step, **view_geometry(joints)})
    robot = pandas.DataFrame(robot_rows)

    names = ["camera_height_cm", "tilt_deg", "roll_deg", "tip_u_px", "tip_v_px", "wrist_deg", "hand_deg", "forearm_twist_deg", "reach_cm"]
    log.say("THE VIEW'S GEOMETRY FROM THE JOINT ANGLES (calculated with the fitted camera model and the arm model; joints measured: frames.csv and steps.csv)")
    log.say("")
    log.say(f"The {len(demonstrations)} training pictures: " + ", ".join(f"{100 * share:g} %" for share in settings["quantiles"]) + " points")
    for name in names:
        log.say(f"  {name:18s} " + "  ".join(f"{demonstrations[name].quantile(share):8.1f}" for share in settings["quantiles"]))
    log.say("")
    low_share, high_share = settings["outside_quantiles"]
    log.say(f"The robot runs: smallest, median, largest per run; and how many pictures lie outside the training pictures' {100 * low_share:g} % to {100 * high_share:g} % range")
    for name in names:
        low, high = demonstrations[name].quantile(low_share), demonstrations[name].quantile(high_share)
        log.say(f"  {name} (training {low:.1f} to {high:.1f})")
        for run_number in ROBOT_RUNS:
            rows = robot[robot["run"] == run_number]
            outside = rows[(rows[name] < low) | (rows[name] > high)]
            before = ""
            if run_number in FAILURE_STEPS and len(outside):
                before = f", {int((outside['step'] <= FAILURE_STEPS[run_number]['first_wrong_decision']).sum())} of them up to the first wrong decision"
            log.say(f"      run {run_number:2d}: {rows[name].min():8.1f}  {rows[name].median():8.1f}  {rows[name].max():8.1f}   outside: {len(outside):3d} of {len(rows)}{before}")

    log.say("")
    held_wrist, held_hand = robot["wrist_deg"].median(), robot["hand_deg"].median()
    near = ((demonstrations["wrist_deg"] - held_wrist).abs() <= 3.0) & ((demonstrations["hand_deg"] - held_hand).abs() <= 3.0)
    log.say(f"Wrist and hand as held by the driver ({held_wrist:.1f} and {held_hand:.1f} degrees): {100 * near.mean():.1f} % of the training pictures have both within 3 degrees of these; "
            f"the wrist alone within 3 degrees: {100 * ((demonstrations['wrist_deg'] - held_wrist).abs() <= 3.0).mean():.1f} %; the hand alone: {100 * ((demonstrations['hand_deg'] - held_hand).abs() <= 3.0).mean():.1f} %")
    log.say(f"Where the tip falls in the picture: training pictures median ({demonstrations['tip_u_px'].median():.0f}, {demonstrations['tip_v_px'].median():.0f}) px, "
            f"middle 90 % u {demonstrations['tip_u_px'].quantile(0.05):.0f} to {demonstrations['tip_u_px'].quantile(0.95):.0f}, v {demonstrations['tip_v_px'].quantile(0.05):.0f} to {demonstrations['tip_v_px'].quantile(0.95):.0f}; "
            f"robot pictures median ({robot['tip_u_px'].median():.0f}, {robot['tip_v_px'].median():.0f}) px, all within u {robot['tip_u_px'].min():.0f} to {robot['tip_u_px'].max():.0f}, v {robot['tip_v_px'].min():.0f} to {robot['tip_v_px'].max():.0f}")

    robot.round(3).to_csv(EVIDENCE_FOLDER / "camera_pose_check_robot.csv", index = False)
    log.say("")
    log.say("written: camera_pose_check_robot.csv")
    log.save()



if __name__ == "__main__":
    main()
