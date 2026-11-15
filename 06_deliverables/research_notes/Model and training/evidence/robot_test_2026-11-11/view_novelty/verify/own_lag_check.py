"""
Were the robot pictures in step with the robot, or one step late? The verifier's own test, without the camera model.

The camera sits on the forearm, so when the forearm twists by some degrees between two pictures the whole scene turns
in the picture by about that much. For every pair of consecutive pictures the turn of the scene is measured from the
pictures alone (matched ORB key points outside the robot's own body, a rotation-plus-shift fit with RANSAC) and
compared with the forearm twist the log says was carried out
    between those two pictures (in step),
    one step earlier (the picture would be one step late),
    one step later (one step early; a control).
If the pictures are in step, the first correlation is clearly the largest. (It turned out not to tell: the twist
changes too smoothly from step to step.) A second test therefore uses the scene's SHIFT under the tip in the pairs where
the tip's direction changed by 60 degrees or more from one step to the next (the back-and-forth of the failed runs).

Run from this folder (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe own_lag_check.py
Output: own_lag_check.out.txt
"""

import cv2
import numpy

from verify_common import RUN_FOLDERS, VERIFY_FOLDER, Report, read_picture, run_log, run_picture_path

LAG_PARAMETERS = {
    "own_body_top_row": 335,
    "key_points": 800,
    "least_matches": 25,
    # where the magnet tip sits in every robot picture (the analyst's number; only the scene's shift near it is needed)
    "tip_pixel": [313, 372],
    "direction_change_least_deg": 60.0
}



def scene_motion(first_bgr: numpy.ndarray, second_bgr: numpy.ndarray) -> tuple:
    """
    Return the motion of the scene between two pictures: its turn in degrees (positive = anticlockwise in the picture)
    and the shift in pixels (x to the right, y downwards) of the scene point under the magnet tip; NaN when not found.
    """

    settings = LAG_PARAMETERS
    mask = numpy.zeros(first_bgr.shape[:2], numpy.uint8)
    mask[:settings["own_body_top_row"]] = 255
    detector = cv2.ORB_create(settings["key_points"])
    first_points, first_codes = detector.detectAndCompute(cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY), mask)
    second_points, second_codes = detector.detectAndCompute(cv2.cvtColor(second_bgr, cv2.COLOR_BGR2GRAY), mask)
    nothing = (numpy.nan, numpy.nan, numpy.nan)
    if first_codes is None or second_codes is None:
        return nothing
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True).match(first_codes, second_codes)
    if len(matches) < settings["least_matches"]:
        return nothing
    source = numpy.float32([first_points[match.queryIdx].pt for match in matches])
    target = numpy.float32([second_points[match.trainIdx].pt for match in matches])
    transform, inliers = cv2.estimateAffinePartial2D(source, target, method = cv2.RANSAC, ransacReprojThreshold = 4.0)
    if transform is None or inliers.sum() < settings["least_matches"] // 2:
        return nothing
    tip = numpy.array(settings["tip_pixel"], dtype = float)
    moved = transform[:, :2] @ tip + transform[:, 2] - tip
    # image rows grow downwards, so the sign is turned to make anticlockwise in the picture positive
    return -float(numpy.degrees(numpy.arctan2(transform[1, 0], transform[0, 0]))), float(moved[0]), float(moved[1])



def correlation(first: numpy.ndarray, second: numpy.ndarray) -> float:
    """Pearson correlation of two series over the rows where both are finite."""

    is_valid = numpy.isfinite(first) & numpy.isfinite(second)
    if is_valid.sum() < 5:
        return numpy.nan
    return float(numpy.corrcoef(first[is_valid], second[is_valid])[0, 1])



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_lag_check.out.txt")
    report.say("THE SCENE'S TURN BETWEEN CONSECUTIVE PICTURES against the forearm twist carried out (degrees)")
    report.say("run  pairs measured  correlation with the twist: in step  one step late  one step early   slope in step (scene degrees per degree of twist)")
    everything = {"turn": [], "in_step": [], "late": [], "early": []}
    shifts = []
    for run_number in RUN_FOLDERS:
        log = run_log(run_number)
        pictures = [read_picture(run_picture_path(run_number, step)) for step in range(len(log))]
        motions = numpy.array([scene_motion(first, second) for first, second in zip(pictures[:-1], pictures[1:])])
        turns = motions[:, 0]
        # the tip's move of row k in the frame of picture k: its direction on the table less the picture's heading
        # (positive = to the picture's left), from where the log says the tip went
        move_deg = numpy.degrees(numpy.arctan2(log["tip_next_y"] - log["tip_y"], log["tip_next_x"] - log["tip_x"])).to_numpy()
        in_picture_deg = move_deg - log["picture_heading_deg"].to_numpy()
        for pair in range(1, len(motions)):
            shifts.append({"run": run_number, "pair": pair, "shift_x": motions[pair, 1], "shift_y": motions[pair, 2],
                           "in_step_deg": in_picture_deg[pair], "late_deg": in_picture_deg[pair - 1]})
        twist = log["twist_used_deg"].to_numpy(dtype = float)
        # the twist of row k is carried out between picture k and picture k + 1
        in_step = twist[:-1]
        late = numpy.concatenate([[numpy.nan], twist[:-2]])
        early = numpy.concatenate([twist[1:-1], [numpy.nan]]) if len(twist) > 2 else numpy.full(len(turns), numpy.nan)
        is_valid = numpy.isfinite(turns) & numpy.isfinite(in_step)
        slope = float(numpy.polyfit(in_step[is_valid], turns[is_valid], 1)[0]) if is_valid.sum() > 5 and numpy.ptp(in_step[is_valid]) > 1 else numpy.nan
        report.say(f"{run_number:3d}  {int(numpy.isfinite(turns).sum()):3d} of {len(turns):3d}  {correlation(turns, in_step):33.2f}  {correlation(turns, late):13.2f}  {correlation(turns, early):14.2f}   {slope:+.2f}")
        for name, values in (("turn", turns), ("in_step", in_step), ("late", late), ("early", early)):
            everything[name].append(values)
    turns = numpy.concatenate(everything["turn"])
    report.say(f"all runs together ({int(numpy.isfinite(turns).sum())} pairs): in step {correlation(turns, numpy.concatenate(everything['in_step'])):.2f}, "
               f"one step late {correlation(turns, numpy.concatenate(everything['late'])):.2f}, one step early {correlation(turns, numpy.concatenate(everything['early'])):.2f}")
    # the sharper test: pairs where the twist changed by 3 degrees or more from one step to the next, so the two readings differ
    in_step, late = numpy.concatenate(everything["in_step"]), numpy.concatenate(everything["late"])
    is_telling = numpy.isfinite(turns) & numpy.isfinite(late) & (numpy.abs(in_step - late) >= 3.0)
    if is_telling.sum() >= 5:
        slope = numpy.polyfit(numpy.concatenate(everything["in_step"])[numpy.isfinite(turns)], turns[numpy.isfinite(turns)], 1)
        fit_in_step = numpy.abs(turns[is_telling] - numpy.polyval(slope, in_step[is_telling]))
        fit_late = numpy.abs(turns[is_telling] - numpy.polyval(slope, late[is_telling]))
        report.say(f"pairs where the twist changed by 3 degrees or more from the step before ({int(is_telling.sum())}): the measured turn is nearer to the in-step twist in "
                   f"{int((fit_in_step < fit_late).sum())} of them (median misfit {numpy.median(fit_in_step):.2f} degrees in step, {numpy.median(fit_late):.2f} one step late)")
    # the second test: the scene's SHIFT under the tip. When the tip moves forward (up in the picture) the scene under it
    # moves down, when it moves to the picture's left the scene moves right: predicted shift direction (x, y) =
    # (sin, cos) of the move's direction in the picture. Told apart only where the direction changed a lot between steps.
    import pandas
    table = pandas.DataFrame(shifts).dropna()
    change = numpy.abs((table["in_step_deg"] - table["late_deg"] + 180) % 360 - 180)
    telling = table[change >= settings_direction_change()]
    measured = numpy.degrees(numpy.arctan2(telling["shift_x"], telling["shift_y"]))
    error_in_step = numpy.abs((measured - telling["in_step_deg"] + 180) % 360 - 180)
    error_late = numpy.abs((measured - telling["late_deg"] + 180) % 360 - 180)
    length = numpy.hypot(telling["shift_x"], telling["shift_y"])
    report.say("")
    report.say("THE SCENE'S SHIFT UNDER THE TIP against the direction of the tip's move, in the pairs where the move's direction changed by "
               f"{settings_direction_change():.0f} degrees or more from the step before")
    report.say(f"pairs: {len(telling)} (runs {sorted(telling['run'].unique().tolist())}); measured shift length median {length.median():.1f} px")
    report.say(f"direction error if the pictures are in step: median {error_in_step.median():.0f} degrees; if one step late: median {error_late.median():.0f} degrees")
    report.say(f"the in-step reading fits better in {int((error_in_step < error_late).sum())} of {len(telling)} pairs")
    report.save()



def settings_direction_change() -> float:
    """The smallest change of direction between two steps for a pair to count in the shift test."""

    return LAG_PARAMETERS["direction_change_least_deg"]



if __name__ == "__main__":
    main()
