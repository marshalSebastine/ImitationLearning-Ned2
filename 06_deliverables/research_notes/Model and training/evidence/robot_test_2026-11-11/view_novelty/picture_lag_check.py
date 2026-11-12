"""
Did each robot picture show the table as seen from the pose logged with it, or from an earlier pose (a picture that
lags behind the move)? stale_pictures.py shows that no two pictures were the same; this check is finer.

Things printed on the table (the sheet, the line) do not move, so the picture taken at pose A can be redrawn as it
would look from pose B with the fitted forearm camera model (forearm_camera_model.py): pixels of picture A are put on
the table plane and projected into picture B, which gives a mapping (a homography) between the two pictures. If the
logged poses are the poses the pictures were taken from, the redrawn picture A lies on top of picture B. If every
picture was taken "lag" steps earlier than its logged pose, the mapping computed from the logged poses is wrong
wherever the motion changes from one step to the next, and the two pictures no longer fit.

For every pair of consecutive pictures the misfit (mean absolute grey difference on the table, the robot's own area
and the red puck left out) is computed for five assumed lags: 0 (in step), 0.25, 0.5, 0.75 and 1 step behind, the
earlier poses being interpolated between the logged joint angles. The lag with the smallest misfit is the estimate.
The demonstrations, where each picture was taken at rest at its logged pose, are the yardstick.

Nothing connects to the robot. Run from this folder (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe picture_lag_check.py
Output: picture_lag_check.out.txt, picture_lag_check_per_pair.csv
"""

from typing import Dict, List, Optional

import cv2
import numpy
import pandas

from common import EVIDENCE_FOLDER, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, run_picture_paths, run_steps, training_frames
from config import PICTURE_CHECK_PARAMETERS, table_height
from forearm_camera_model import pixels_to_plane, project_points

LAG_PARAMETERS = {
    "lags": [0.0, 0.25, 0.5, 0.75, 1.0],
    # the grid of pixels of the earlier picture that are put on the table: the upper part, clear of the robot's own area
    "grid_u_px": [40, 600, 9],
    "grid_v_px": [30, 300, 7],
    "plane_point_xy": [0.28, 0.0],
    "puck_growth_px": 21,
    # a pair tells the lags apart only when the mappings of lag 0 and lag 1 differ by more than this on the grid
    "informative_px": 4.0,
    "demonstration_episodes": 24,
    "demonstration_days": ["2026-11-08", "2026-11-09"],
    "seed": 11_11_26
}



def table_mapping(
        joints_earlier: numpy.ndarray,
        joints_later: numpy.ndarray,
        parameters: Dict
) -> Optional[numpy.ndarray]:
    """
    Return the homography that carries table points from the picture taken at one pose into the picture taken at another.

    Parameters
    ----------
    joints_earlier, joints_later : numpy.ndarray
        The six joint angles (radians) of the two poses.
    parameters : dict
        ``LAG_PARAMETERS`` (the pixel grid and the table plane's reference point).

    Returns
    -------
    numpy.ndarray or None
        The 3 x 3 mapping, or None when fewer than six grid points are seen from both poses.
    """

    grid_u = numpy.linspace(*parameters["grid_u_px"][:2], int(parameters["grid_u_px"][2]))
    grid_v = numpy.linspace(*parameters["grid_v_px"][:2], int(parameters["grid_v_px"][2]))
    pixels = numpy.array([[u, v] for v in grid_v for u in grid_u], dtype = float)
    height = table_height(*parameters["plane_point_xy"])
    on_table = pixels_to_plane(pixels, joints_earlier, height)
    points = numpy.column_stack([on_table, numpy.full(len(on_table), height)])
    is_seen = numpy.all(numpy.isfinite(points), axis = 1)
    projected = numpy.full_like(pixels, numpy.nan)
    if is_seen.any():
        projected[is_seen] = project_points(points[is_seen], joints_later)
    is_usable = is_seen & numpy.all(numpy.isfinite(projected), axis = 1)
    if is_usable.sum() < 6:
        return None
    mapping, _ = cv2.findHomography(pixels[is_usable].astype(numpy.float32), projected[is_usable].astype(numpy.float32))
    return mapping



def pair_misfits(
        picture_earlier: numpy.ndarray,
        picture_later: numpy.ndarray,
        joints: List[numpy.ndarray],
        parameters: Dict
) -> Optional[Dict[str, float]]:
    """
    Compute how badly the earlier picture, redrawn for the later pose, fits the later picture, for every assumed lag.

    Parameters
    ----------
    picture_earlier, picture_later : numpy.ndarray
        Two consecutive camera pictures (BGR).
    joints : list of numpy.ndarray
        The logged joint angles of the step before the earlier picture, of the earlier picture and of the later
        picture, in that order.
    parameters : dict
        ``LAG_PARAMETERS``.

    Returns
    -------
    dict or None
        "misfit_<lag>" per assumed lag and "mapping_difference_px" (how far the mappings of lag 0 and lag 1 are
        apart on the grid); None when a mapping could not be made.
    """

    before, earlier, later = [numpy.asarray(value, dtype = float) for value in joints]
    height, width = picture_later.shape[:2]
    grey_earlier = cv2.cvtColor(picture_earlier, cv2.COLOR_BGR2GRAY)
    grey_later = cv2.cvtColor(picture_later, cv2.COLOR_BGR2GRAY).astype(numpy.int16)

    def red_mask(picture: numpy.ndarray) -> numpy.ndarray:
        hsv = cv2.cvtColor(picture, cv2.COLOR_BGR2HSV)
        is_red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 165)) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 50)
        size = parameters["puck_growth_px"]
        return cv2.dilate(is_red.astype(numpy.uint8), numpy.ones((size, size), dtype = numpy.uint8))

    left, top, right, bottom = PICTURE_CHECK_PARAMETERS["robot_area_px"]
    is_table_later = numpy.ones((height, width), dtype = bool)
    is_table_later[top:bottom, left:right] = False
    is_table_later &= red_mask(picture_later) == 0
    usable_earlier = numpy.ones((height, width), dtype = numpy.uint8)
    usable_earlier[top:bottom, left:right] = 0
    usable_earlier[red_mask(picture_earlier) > 0] = 0

    result, mappings = {}, {}
    for lag in parameters["lags"]:
        mapping = table_mapping((1 - lag) * earlier + lag * before, (1 - lag) * later + lag * earlier, parameters)
        if mapping is None:
            return None
        mappings[lag] = mapping
        redrawn = cv2.warpPerspective(grey_earlier, mapping, (width, height)).astype(numpy.int16)
        is_covered = cv2.warpPerspective(usable_earlier, mapping, (width, height), flags = cv2.INTER_NEAREST) > 0
        is_compared = is_covered & is_table_later
        if is_compared.sum() < 5000:
            return None
        result[f"misfit_{lag:g}"] = float(numpy.abs(redrawn - grey_later)[is_compared].mean())
    corners = numpy.array([[[u, v]] for v in (30, 300) for u in (40, 600)], dtype = numpy.float32)
    apart = cv2.perspectiveTransform(corners, mappings[parameters["lags"][0]]) - cv2.perspectiveTransform(corners, mappings[parameters["lags"][-1]])
    result["mapping_difference_px"] = float(numpy.linalg.norm(apart, axis = 2).mean())
    return result



def sequence_misfits(
        pictures: List,
        joints: numpy.ndarray,
        parameters: Dict
) -> pandas.DataFrame:
    """Compute the misfits of every pair of consecutive pictures of one run or episode; ``pictures`` are file paths."""

    rows = []
    earlier_picture = cv2.imread(str(pictures[1])) if len(pictures) > 2 else None
    for position in range(2, len(pictures)):
        later_picture = cv2.imread(str(pictures[position]))
        misfits = pair_misfits(earlier_picture, later_picture, [joints[position - 2], joints[position - 1], joints[position]], parameters)
        if misfits is not None:
            rows.append({"position": position, **misfits})
        earlier_picture = later_picture
    return pandas.DataFrame(rows)



def say_summary(
        label: str,
        table: pandas.DataFrame,
        parameters: Dict,
        log: OutputLog
) -> None:
    """Say, for one set of pairs, the misfit per assumed lag and how often each lag fits best, on the pairs that tell the lags apart."""

    columns = [f"misfit_{lag:g}" for lag in parameters["lags"]]
    informative = table[table["mapping_difference_px"] >= parameters["informative_px"]]
    if informative.empty:
        log.say(f"{label:34s} {len(table):5d} pairs, none of them tells the lags apart")
        return
    best = informative[columns].to_numpy().argmin(axis = 1)
    shares = [100 * float((best == index).mean()) for index in range(len(columns))]
    medians = [informative[column].median() for column in columns]
    log.say(f"{label:34s} {len(table):5d}  {len(informative):11d} | " + "  ".join(f"{value:6.2f}" for value in medians) + " | " + "  ".join(f"{share:5.0f} %" for share in shares))



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(LAG_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "picture_lag_check.out.txt")
    joint_columns = [f"joint_{number}" for number in range(1, 7)]

    frames = training_frames()
    candidates = frames[frames["session_date"].isin(settings["demonstration_days"])].groupby("episode_id").size()
    candidates = candidates[candidates >= 30].index.to_numpy()
    chosen = numpy.random.default_rng(settings["seed"]).choice(candidates, size = min(settings["demonstration_episodes"], len(candidates)), replace = False)
    demonstration_tables = []
    for episode_id in chosen:
        episode = frames[frames["episode_id"] == episode_id].sort_values("row_in_episode")
        paths = [VIEW_NOVELTY_PARAMETERS["raw_folder"] / path for path in episode["frame_path"]]
        table = sequence_misfits(paths, episode[joint_columns].to_numpy(dtype = float), settings)
        table["episode_id"] = episode_id
        demonstration_tables.append(table)
    demonstrations = pandas.concat(demonstration_tables).reset_index(drop = True)

    robot_tables = []
    for run_number in ROBOT_RUNS:
        steps = run_steps(run_number)
        table = sequence_misfits(run_picture_paths(run_number), steps[joint_columns].to_numpy(dtype = float), settings)
        table["run"] = run_number
        robot_tables.append(table)
    robot = pandas.concat(robot_tables).reset_index(drop = True)

    lag_names = "  ".join(f"{lag:6g}" for lag in settings["lags"])
    log.say("DOES THE PICTURE BELONG TO THE POSE LOGGED WITH IT? Misfit of consecutive pictures on the table (mean absolute grey difference), for five assumed lags")
    log.say(f"(a pair 'tells the lags apart' when the mappings of lag 0 and lag 1 differ by {settings['informative_px']} px or more; medians and shares are over those pairs)")
    log.say("")
    log.say(f"set                                pairs  telling apart | median misfit at lag {lag_names} | share of pairs fitting best at lag {lag_names}")
    say_summary(f"demonstrations ({len(chosen)} episodes)", demonstrations, settings, log)
    for run_number in ROBOT_RUNS:
        say_summary(f"robot run {run_number} ({ROBOT_RUNS[run_number]['sheet']})", robot[robot["run"] == run_number], settings, log)
    say_summary("all ten robot runs", robot, settings, log)

    robot.round(4).to_csv(EVIDENCE_FOLDER / "picture_lag_check_per_pair.csv", index = False)
    log.say("")
    log.say("written: picture_lag_check_per_pair.csv")
    log.save()



if __name__ == "__main__":
    main()
