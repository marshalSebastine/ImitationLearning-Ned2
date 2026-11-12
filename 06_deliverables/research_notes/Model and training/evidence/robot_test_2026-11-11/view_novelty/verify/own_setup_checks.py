"""
The verifier's own checks of the set-up claims of the view-novelty answer (robot test of 11.11.26):

    A  stale pictures       were consecutive robot pictures ever identical, or nearly so?
    B  where runs started   the tip and the forearm twist at step 0 against the demonstrations' standard start
    C  light                brightness of the robot pictures against the training pictures (same formula for both)
    D  the blue hoodie      when royal blue came into view in runs 4, 5, 6 and 9, and what model E answers when it is
                            painted over with the table's colour (run 6, the steps round its failure; run 4)
    E  the camera's creep   model E's answers with every robot picture slid 17 px to the left (the creep undone, as
                            measured in own_camera_creep.py), at the steps where the failed runs went wrong
    F  sheet 02 square      the corrected ends of plans 02-13 to 02-18 and 02-01, and which demonstration starts the
                            first pictures of runs 3 and 4 are nearest to (fold model that never saw sheet 02)

Nothing connects to the robot. Run from this folder (about two minutes):
    D:/AI_Project/.venv/Scripts/python.exe own_setup_checks.py
Output: own_setup_checks.out.txt, figures/own_hoodie_first_views.png
"""

import cv2
import numpy
import pandas
import torch

from own_novelty import load_table, nearest, trained_rows
from verify_common import (RUN_FOLDERS, RUN_MODELS, RUN_SHEETS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, labelled_montage, read_picture, run_log,
                           run_picture_path, small_rgb, training_frames)

SETUP_PARAMETERS = {
    "creep_px": 17,
    # royal blue of the hoodie (and of the robot's own base): strongly saturated blue, not dark
    "blue_hue": [104, 126],
    "blue_saturation_least": 200,
    "blue_value_least": 60,
    # the robot's own arm at the bottom of the picture is blue too and is left out (left, top, right, bottom)
    "own_arm_box": [150, 380, 640, 480],
    "in_view_share": 0.005,
    "large_object_px": 1500,
    "key_steps": {3: [0, 1, 2, 3], 5: [2, 3, 4, 5], 6: [6, 7, 8, 9, 10], 8: [7, 9, 12, 15], 9: [2, 3, 4, 5], 4: [0, 1, 2, 3]}
}



def model_e_policy():
    """Load model E through the robot driver's own loader."""

    from model_mode import ModelPolicy

    folder = VERIFY_PARAMETERS["trained_models_folder"] / "2026-11-10_resnet18_model_4_chunk_stop_25_all_sheets"
    policy = ModelPolicy("resnet18_model_4_chunk_stop_25_all_sheets", folder / "checkpoints" / "all_sheets.pt", folder / "metrics.json")
    torch.set_num_threads(VERIFY_PARAMETERS["threads"])
    return policy



def decide_on_picture(policy, picture_bgr: numpy.ndarray) -> dict:
    """Model E's answer to a picture given as an array (re-encoded as JPEG of high quality, then the driver's own path)."""

    return policy.decide(cv2.imencode(".jpg", picture_bgr, [cv2.IMWRITE_JPEG_QUALITY, 97])[1].tobytes())



def blue_mask(picture_bgr: numpy.ndarray) -> numpy.ndarray:
    """Return the royal-blue pixels outside the robot's own arm."""

    settings = SETUP_PARAMETERS
    hsv = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 0] >= settings["blue_hue"][0]) & (hsv[..., 0] <= settings["blue_hue"][1]) & (hsv[..., 1] >= settings["blue_saturation_least"])
            & (hsv[..., 2] >= settings["blue_value_least"]))
    left, top, right, bottom = settings["own_arm_box"]
    mask[top:bottom, left:right] = False
    return mask



def large_blue_object(picture_bgr: numpy.ndarray) -> numpy.ndarray:
    """
    Return the royal-blue pixels that belong to a large object at the picture's border (a hoodie, or the robot's own
    base when the forearm is twisted far): connected regions of ``blue_mask`` of 1,500 pixels or more that touch the
    border. The block's top face and glossy bits of the printed line, which are blue too, are small or away from the
    border and are left out.
    """

    mask = blue_mask(picture_bgr)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask.astype(numpy.uint8), connectivity = 8)
    height, width = mask.shape
    kept = numpy.zeros(mask.shape, bool)
    for label in range(1, count):
        left, top, box_width, box_height, area = statistics[label]
        touches_border = left == 0 or top == 0 or left + box_width >= width or top + box_height >= height
        if area >= SETUP_PARAMETERS["large_object_px"] and touches_border:
            kept |= labels == label
    return kept



def check_stale(report: Report) -> None:
    """A: identical or nearly identical consecutive pictures."""

    report.say("A. STALE PICTURES: consecutive pictures of every run")
    report.say("run  pairs  byte-identical  identical after decoding  smallest mean change (grey levels) at step  median change  tip moved cm: least  most  seconds between pictures: least  most")
    total_pairs, total_identical = 0, 0
    for run_number in RUN_FOLDERS:
        log = run_log(run_number)
        data = [run_picture_path(run_number, step).read_bytes() for step in range(len(log))]
        greys = [cv2.cvtColor(cv2.imdecode(numpy.frombuffer(item, numpy.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(numpy.float32) for item in data]
        identical_bytes = sum(first == second for first, second in zip(data[:-1], data[1:]))
        changes = numpy.array([numpy.abs(second - first).mean() for first, second in zip(greys[:-1], greys[1:])])
        identical_pixels = int((changes == 0).sum())
        moved = 100 * numpy.hypot(numpy.diff(log["tip_x"]), numpy.diff(log["tip_y"]))
        seconds = numpy.diff(log["unix_time"])
        total_pairs += len(changes)
        total_identical += identical_bytes
        report.say(f"{run_number:3d}  {len(changes):5d}  {identical_bytes:14d}  {identical_pixels:24d}  {changes.min():34.2f} at {int(changes.argmin()) + 1:3d}  {numpy.median(changes):13.2f}  "
                   f"{moved.min():19.2f}  {moved.max():.2f}  {seconds.min():31.2f}  {seconds.max():.2f}")
    report.say(f"all runs: {total_pairs} pairs, {total_identical} byte-identical")
    # the yardstick: how little a picture changes when the arm really stood still (demonstration rows flagged as the same picture)
    frames = training_frames()
    if "is_same_md5_as_previous" in frames.columns:
        report.say(f"for scale: {int(frames['is_same_md5_as_previous'].fillna(0).sum())} of {len(frames)} demonstration pictures are byte-identical to the one before (frames.csv)")



def check_starts(report: Report) -> None:
    """B: where each run started against the demonstrations' standard start."""

    frames = training_frames()
    first = frames[(frames["row_in_episode"] == 0) & (frames["start_kind"] == "STD")]
    report.say("")
    report.say("B. WHERE RUNS STARTED")
    report.say(f"demonstrations, row 0 of the {len(first)} standard-start episodes: tip x {100 * first['tip_x_m'].min():.2f} to {100 * first['tip_x_m'].max():.2f} cm "
               f"(median {100 * first['tip_x_m'].median():.2f}), tip y {100 * first['tip_y_m'].min():.2f} to {100 * first['tip_y_m'].max():.2f} cm (median {100 * first['tip_y_m'].median():.2f}), "
               f"forearm twist {numpy.degrees(first['joint_4']).min():.2f} to {numpy.degrees(first['joint_4']).max():.2f} degrees, "
               f"wrist median {numpy.degrees(first['joint_5']).median():.2f}, hand median {numpy.degrees(first['joint_6']).median():.2f}")
    # the row at which the tip first touches the puck is later than row 0: where is the tip when the pushing begins?
    moving = frames[(frames["start_kind"] == "STD") & (frames["row_in_episode"].between(1, 6))]
    report.say(f"demonstrations, rows 1 to 6 of the same episodes: tip x median {100 * moving['tip_x_m'].median():.2f} cm (5 % {100 * moving['tip_x_m'].quantile(0.05):.2f}, "
               f"95 % {100 * moving['tip_x_m'].quantile(0.95):.2f}), tip y median {100 * moving['tip_y_m'].median():.2f} cm")
    start_x, start_y = 100 * first["tip_x_m"].median(), 100 * first["tip_y_m"].median()
    report.say("run  sheet     tip x cm  tip y cm  from the demonstrations' row-0 start: forward  left  distance cm  forearm twist deg  wrist deg  hand deg")
    for run_number in RUN_FOLDERS:
        row = run_log(run_number).iloc[0]
        x, y = 100 * row["tip_x"], 100 * row["tip_y"]
        report.say(f"{run_number:3d}  {RUN_SHEETS[run_number]}  {x:8.1f}  {y:8.1f}  {x - start_x:+46.1f}  {y - start_y:+5.1f}  {numpy.hypot(x - start_x, y - start_y):11.1f}  "
                   f"{numpy.degrees(row['joint_4']):+17.1f}  {numpy.degrees(row['joint_5']):9.1f}  {numpy.degrees(row['joint_6']):8.1f}")



def check_light(report: Report) -> None:
    """C: brightness, the same formula for the training cache and the robot pictures (mean of the 160 x 120 RGB picture)."""

    cache = numpy.load(VERIFY_PARAMETERS["processed_folder"] / "images_wide_160.npy", mmap_mode = "r")
    sample = numpy.arange(0, len(cache), 6)
    training = numpy.array([float(cache[index].mean()) for index in sample])
    report.say("")
    report.say("C. LIGHT: mean of the 160 x 120 picture (0 to 255), every sixth training picture against every robot picture")
    report.say(f"training pictures ({len(training)}): median {numpy.median(training):.1f}, 1 % {numpy.quantile(training, 0.01):.1f}, 5 % {numpy.quantile(training, 0.05):.1f}, "
               f"95 % {numpy.quantile(training, 0.95):.1f}, 99 % {numpy.quantile(training, 0.99):.1f}")
    report.say("run  median  least  most  place of the median among the training pictures (percentile)  pictures outside the training pictures' 1 % to 99 %")
    for run_number in RUN_FOLDERS:
        values = numpy.array([float(small_rgb(read_picture(run_picture_path(run_number, step))).mean()) for step in range(len(run_log(run_number)))])
        outside = int(((values < numpy.quantile(training, 0.01)) | (values > numpy.quantile(training, 0.99))).sum())
        report.say(f"{run_number:3d}  {numpy.median(values):6.1f}  {values.min():5.1f}  {values.max():5.1f}  {100 * (training < numpy.median(values)).mean():60.0f}  {outside:10d} of {len(values)}")



def check_hoodie(report: Report, policy) -> None:
    """D: when royal blue came into view, and model E's answers with it painted over."""

    settings = SETUP_PARAMETERS
    report.say("")
    report.say("D. THE BLUE HOODIE: share of the picture covered by a LARGE royal-blue object at the picture's border, per picture")
    report.say("run  pictures with more than 0.5 % royal blue  first such step  largest share  steps (share in %)")
    shares = {}
    for run_number in RUN_FOLDERS:
        values = numpy.array([large_blue_object(read_picture(run_picture_path(run_number, step))).mean() for step in range(len(run_log(run_number)))])
        shares[run_number] = values
        in_view = numpy.flatnonzero(values > settings["in_view_share"])
        listed = " ".join(f"{step}({100 * values[step]:.1f})" for step in in_view[:14]) + (" ..." if len(in_view) > 14 else "")
        report.say(f"{run_number:3d}  {len(in_view):41d}  {(int(in_view[0]) if len(in_view) else -1):15d}  {100 * values.max():12.1f} %  {listed}")
    report.say("(royal blue is also the colour of the robot's own base, which comes into view at the bottom left when the forearm is twisted far to the")
    report.say(" negative side; the figure shows which is which: the hoodie is at the top of the picture in runs 4 and 6)")

    report.say("")
    report.say("Model E's answer with the large royal-blue object painted over by the table's colour (median of the picture's non-blue, non-paper pixels)")
    report.say("run  step  blue %  | as seen: cm  deg  twist  stop | painted over: cm  deg  twist  stop | change: deg  cm")
    for run_number, steps in ((6, range(11, 24)), (4, [20, 25, 30, 35, 38, 39, 40, 41])):
        log = run_log(run_number)
        for step in steps:
            picture = read_picture(run_picture_path(run_number, step))
            mask = large_blue_object(picture)
            if mask.mean() <= 0.001:
                report.say(f"{run_number:3d}  {step:4d}  {100 * mask.mean():6.1f}  | no royal blue in view")
                continue
            wide = cv2.dilate(mask.astype(numpy.uint8), numpy.ones((9, 9), numpy.uint8)) > 0
            lab = cv2.cvtColor(picture, cv2.COLOR_BGR2LAB)
            is_table = ~wide & (lab[..., 0] < 175) & (lab[..., 2] > 122)
            colour = numpy.median(picture[is_table], axis = 0) if is_table.sum() > 500 else numpy.array([140, 150, 140])
            painted = picture.copy()
            painted[wide] = colour
            seen, changed = decide_on_picture(policy, picture), decide_on_picture(policy, painted)
            seen_cm, changed_cm = numpy.hypot(seen["tip_forward_cm"], seen["tip_left_cm"]), numpy.hypot(changed["tip_forward_cm"], changed["tip_left_cm"])
            turn = (changed["angle_deg"] - seen["angle_deg"] + 180) % 360 - 180
            report.say(f"{run_number:3d}  {step:4d}  {100 * mask.mean():6.1f}  | {seen_cm:11.2f} {seen['angle_deg']:+5.0f} {seen['twist_deg']:+6.1f} {seen['stop_probability']:5.2f} | "
                       f"{changed_cm:16.2f} {changed['angle_deg']:+5.0f} {changed['twist_deg']:+6.1f} {changed['stop_probability']:5.2f} | {turn:+11.1f} {changed_cm - seen_cm:+5.2f}")

    pictures, names = [], []
    for run_number, step in ((6, 12), (6, 13), (6, 14), (6, 18), (5, 23), (5, 24), (9, 20), (4, 20), (4, 30)):
        picture = read_picture(run_picture_path(run_number, step))
        mask = large_blue_object(picture)
        shown = picture.copy()
        shown[mask] = (0, 255, 255)
        pictures.append(shown)
        names.append(f"run {run_number} step {step}: {100 * mask.mean():.1f} % large royal-blue object (yellow)")
    cv2.imencode(".png", labelled_montage(pictures, names, columns = 3, cell_width = 480))[1].tofile(str(VERIFY_FOLDER / "figures" / "own_hoodie_first_views.png"))



def check_creep_counterfactual(report: Report, policy) -> None:
    """E: model E's answers with the robot picture slid to the left by the measured creep."""

    settings = SETUP_PARAMETERS
    shift = settings["creep_px"]
    report.say("")
    report.say(f"E. THE CAMERA'S CREEP UNDONE: every robot picture slid {shift} px to the left (the right edge repeated), model E's answer as seen and slid")
    report.say("run  step | as seen: cm  deg  twist | slid: cm  deg  twist | change: deg  cm  twist")
    for run_number, steps in settings["key_steps"].items():
        for step in steps:
            picture = read_picture(run_picture_path(run_number, step))
            slid = numpy.concatenate([picture[:, shift:], numpy.repeat(picture[:, -1:], shift, axis = 1)], axis = 1)
            seen, changed = policy.decide(run_picture_path(run_number, step).read_bytes()), decide_on_picture(policy, slid)
            seen_cm, changed_cm = numpy.hypot(seen["tip_forward_cm"], seen["tip_left_cm"]), numpy.hypot(changed["tip_forward_cm"], changed["tip_left_cm"])
            turn = (changed["angle_deg"] - seen["angle_deg"] + 180) % 360 - 180
            report.say(f"{run_number:3d}  {step:4d} | {seen_cm:11.2f} {seen['angle_deg']:+5.0f} {seen['twist_deg']:+6.1f} | {changed_cm:8.2f} {changed['angle_deg']:+5.0f} {changed['twist_deg']:+6.1f} | "
                       f"{turn:+11.1f} {changed_cm - seen_cm:+5.2f} {changed['twist_deg'] - seen['twist_deg']:+6.1f}")
    report.say("")
    report.say("the same over whole runs (model E's runs): median and largest turn of the asked direction, pictures whose asked length is 0.3 cm or more")
    report.say("run  pictures  median turn deg  middle half        largest turn  median change of length cm  median change of twist deg")
    for run_number in RUN_FOLDERS:
        if RUN_MODELS[run_number] != "E":
            continue
        turns, lengths, twists = [], [], []
        for step in range(0, len(run_log(run_number)), 2):
            picture = read_picture(run_picture_path(run_number, step))
            slid = numpy.concatenate([picture[:, shift:], numpy.repeat(picture[:, -1:], shift, axis = 1)], axis = 1)
            seen, changed = decide_on_picture(policy, picture), decide_on_picture(policy, slid)
            seen_cm, changed_cm = numpy.hypot(seen["tip_forward_cm"], seen["tip_left_cm"]), numpy.hypot(changed["tip_forward_cm"], changed["tip_left_cm"])
            if seen_cm < 0.3:
                continue
            turns.append((changed["angle_deg"] - seen["angle_deg"] + 180) % 360 - 180)
            lengths.append(changed_cm - seen_cm)
            twists.append(changed["twist_deg"] - seen["twist_deg"])
        turns = numpy.array(turns)
        report.say(f"{run_number:3d}  {len(turns):8d}  {numpy.median(turns):+15.1f}  {numpy.quantile(turns, 0.25):+6.1f} to {numpy.quantile(turns, 0.75):+5.1f}  {numpy.abs(turns).max():12.1f}  "
                   f"{numpy.median(lengths):+26.2f}  {numpy.median(twists):+26.2f}")



def check_square_sheet(report: Report) -> None:
    """F: the square lay-out of sheet 02 in the data, and the nearest demonstration starts of runs 3 and 4."""

    report.say("")
    report.say("F. SHEET 02 LYING SQUARE")
    frames = training_frames()
    trained = trained_rows()
    raw_end = frames.groupby("plan_number")["start_end"].first()
    used_end = trained.groupby("plan_number")["start_end"].first()
    report.say("plan    kind  end in frames.csv  end as model E's training table has it (audit's corrected ends)  rows trained on  first-5-cm forearm twist deg")
    sheet = trained[trained["sheet"] == "path_02"]
    for plan in ["02-01"] + [f"02-{number}" for number in range(13, 19)] + ["02-31", "02-36", "02-41"]:
        episode = sheet[sheet["plan_number"] == plan].sort_values("row_in_episode")
        if episode.empty:
            report.say(f"{plan}   not in the training table")
            continue
        length = numpy.hypot(episode["tip_forward_m"].fillna(0.0), episode["tip_left_m"].fillna(0.0)).cumsum()
        twist = episode.loc[(length <= 0.05).to_numpy(), "forearm_change_deg"].fillna(0.0).sum()
        report.say(f"{plan}   {episode['start_kind'].iloc[0]:4s}  {raw_end[plan]:17s}  {used_end[plan]:62s}  {len(episode):15d}  {twist:+27.1f}")

    fold_sheet, fold_robot = load_table("fold_without_path_02_sheet"), load_table("fold_without_path_02_robot")
    sheet_frames = frames[frames["sheet"] == "path_02"].set_index("frame_index").loc[fold_sheet["frame_index"]].reset_index()
    end_of_plan = used_end.to_dict()
    is_start = (sheet_frames["row_in_episode"].between(1, 8)).to_numpy()
    start_frames = sheet_frames[is_start].reset_index(drop = True)
    report.say("")
    report.say("Nearest demonstration START pictures (rows 1 to 8 of every sheet-02 episode) of the first pictures of runs 3 and 4, fold model without sheet 02")
    report.say("run  step  5 nearest: plan kind end(row) distance ...                                                    | nearest distance by kind of start")
    for run_number in (3, 4):
        is_run = fold_robot["run"] == run_number
        distance, index = nearest(fold_robot["grid"][is_run][:3], fold_sheet["grid"][is_start], count = 5)
        every_distance, every_index = nearest(fold_robot["grid"][is_run][:3], fold_sheet["grid"][is_start], count = int(is_start.sum()))
        for step in range(3):
            neighbours = start_frames.iloc[index[step]]
            listed = "  ".join(f"{plan} {kind} {end_of_plan.get(plan, '?')}({int(row)}) {value:.3f}"
                               for plan, kind, row, value in zip(neighbours["plan_number"], neighbours["start_kind"], neighbours["row_in_episode"], distance[step]))
            kinds = start_frames["start_kind"].to_numpy()[every_index[step]] + " " + numpy.array([end_of_plan.get(plan, "?") for plan in start_frames["plan_number"].to_numpy()[every_index[step]]])
            by_kind = {}
            for kind, value in zip(kinds, every_distance[step]):
                by_kind.setdefault(kind, value)
            ordered = sorted(by_kind.items(), key = lambda item: item[1])[:6]
            report.say(f"{run_number:3d}  {step:4d}  {listed} | " + "  ".join(f"{kind} {value:.3f}" for kind, value in ordered))



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_setup_checks.out.txt")
    check_stale(report)
    check_starts(report)
    check_light(report)
    policy = model_e_policy()
    check_hoodie(report, policy)
    check_creep_counterfactual(report, policy)
    check_square_sheet(report)
    report.say("")
    report.say("written: figures/own_hoodie_first_views.png")
    report.save()



if __name__ == "__main__":
    main()
