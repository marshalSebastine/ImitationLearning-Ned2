"""
Has the camera moved on its tape mount between the recording days and the robot day? The verifier's own measurement,
independent of the recording tool's strip check the analyst used.

Idea: the wrist (joint 5) and the hand (joint 6) were held at 13.4 and -6.1 degrees on the robot day, so the robot's
own arm and block sit at one place in all 554 robot pictures. Their median picture is the template. A demonstration
picture taken with the wrist and the hand within half a degree of those values shows the same parts in the same pose;
if the camera has not moved, they sit at the same pixels. The shift that fits the template best (masked template
matching over plus or minus 32 pixels, on the pixels that hardly changed on the robot day) is the camera's shift
between that recording and the robot day.

Self-check: the robot pictures themselves must give a shift near 0.

Run from this folder (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe own_camera_creep.py
Output: own_camera_creep.out.txt, own_camera_creep_per_picture.csv, figures/own_camera_creep_examples.png
"""

import cv2
import numpy
import pandas

from verify_common import RUN_FOLDERS, VERIFY_FOLDER, VERIFY_PARAMETERS, Report, labelled_montage, read_picture, run_log, run_picture_path, training_frames

CREEP_PARAMETERS = {
    "template_box": [200, 392, 590, 478],
    "search_margin_px": 32,
    "static_spread_most": 14.0,
    "wrist_deg": 13.35,
    "hand_deg": -6.15,
    "joint_tolerance_deg": 0.5,
    "pictures_per_group": 40,
    "robot_pictures_per_run": 12
}



def robot_day_template() -> tuple:
    """
    Return the robot day's median picture of the arm region (grey) and the mask of the pixels that hardly changed.

    Returns
    -------
    tuple of numpy.ndarray
        (template, mask), both cut to "template_box".
    """

    stack = []
    for run_number in RUN_FOLDERS:
        for step in range(0, len(run_log(run_number)), 2):
            stack.append(cv2.cvtColor(read_picture(run_picture_path(run_number, step)), cv2.COLOR_BGR2GRAY))
    stack = numpy.stack(stack).astype(numpy.float32)
    left, top, right, bottom = CREEP_PARAMETERS["template_box"]
    template = numpy.median(stack, axis = 0)[top:bottom, left:right]
    mask = (stack.std(axis = 0)[top:bottom, left:right] < CREEP_PARAMETERS["static_spread_most"]).astype(numpy.float32)
    return template.astype(numpy.float32), mask



def best_shift(picture_bgr: numpy.ndarray, template: numpy.ndarray, mask: numpy.ndarray) -> tuple:
    """
    Find where the robot day's arm template fits best in a picture.

    Returns
    -------
    tuple
        (shift x, shift y, score): how far the arm sits to the right and down in this picture compared with the robot
        day, in pixels, and the normalised correlation of the fit.
    """

    margin = CREEP_PARAMETERS["search_margin_px"]
    left, top, right, bottom = CREEP_PARAMETERS["template_box"]
    grey = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
    padded = cv2.copyMakeBorder(grey, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
    search = padded[top:bottom + 2 * margin, left:right + 2 * margin]
    scores = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED, mask = mask)
    _, best, _, location = cv2.minMaxLoc(scores)
    return location[0] - margin, location[1] - margin, float(best)



def main() -> None:

    report = Report(VERIFY_FOLDER / "own_camera_creep.out.txt")
    settings = CREEP_PARAMETERS
    template, mask = robot_day_template()
    report.say(f"template: {template.shape[1]} x {template.shape[0]} pixels of the arm region, {int(mask.sum())} of them steady on the robot day")

    rows = []
    for run_number in RUN_FOLDERS:
        steps = numpy.linspace(0, len(run_log(run_number)) - 1, settings["robot_pictures_per_run"]).round().astype(int)
        for step in steps:
            shift_x, shift_y, score = best_shift(read_picture(run_picture_path(run_number, step)), template, mask)
            rows.append({"group": f"robot run {run_number:2d}", "day": "2026-11-11", "sheet": "", "shift_x": shift_x, "shift_y": shift_y, "score": score})

    frames = training_frames()
    wrist, hand = numpy.degrees(frames["joint_5"]), numpy.degrees(frames["joint_6"])
    is_same_pose = ((wrist - settings["wrist_deg"]).abs() <= settings["joint_tolerance_deg"]) & ((hand - settings["hand_deg"]).abs() <= settings["joint_tolerance_deg"])
    report.say(f"demonstration pictures with wrist and hand within {settings['joint_tolerance_deg']} degrees of the held values: {int(is_same_pose.sum())} of {len(frames)}")
    examples = []
    randomness_generator = numpy.random.default_rng(1111)
    for (day, sheet), group in frames[is_same_pose].groupby(["session_date", "sheet"]):
        chosen = group.iloc[numpy.sort(randomness_generator.choice(len(group), size = min(settings["pictures_per_group"], len(group)), replace = False))]
        for _, row in chosen.iterrows():
            picture = read_picture(VERIFY_PARAMETERS["raw_folder"] / row["frame_path"])
            shift_x, shift_y, score = best_shift(picture, template, mask)
            rows.append({"group": f"{day} {sheet}", "day": day, "sheet": sheet, "shift_x": shift_x, "shift_y": shift_y, "score": score,
                         "episode_id": row["episode_id"], "row_in_episode": row["row_in_episode"]})
        examples.append((day, sheet, chosen.iloc[0]["frame_path"]))
    table = pandas.DataFrame(rows)
    table.round(3).to_csv(VERIFY_FOLDER / "own_camera_creep_per_picture.csv", index = False)

    report.say("")
    report.say("WHERE THE ARM SITS IN THE PICTURE, against the robot day (pixels of 640 x 480; negative x = further LEFT than on the robot day)")
    report.say("group                   pictures   shift x: median  [5 %, 95 %]    shift y: median  [5 %, 95 %]   fit score median")
    for group, part in table.groupby("group", sort = False):
        report.say(f"{group:22s}  {len(part):8d}   {part['shift_x'].median():15.1f}  [{part['shift_x'].quantile(0.05):5.1f}, {part['shift_x'].quantile(0.95):5.1f}]   "
                   f"{part['shift_y'].median():15.1f}  [{part['shift_y'].quantile(0.05):5.1f}, {part['shift_y'].quantile(0.95):5.1f}]   {part['score'].median():.3f}")
    report.say("")
    report.say("BY RECORDING DAY (demonstrations only; good fits only, score 0.97 or more)")
    report.say("day          pictures   shift x: median  [5 %, 95 %]   shift y: median   so the arm sits this much further RIGHT on the robot day")
    good = table[(table["day"] != "2026-11-11") & (table["score"] >= 0.97)]
    for day, part in good.groupby("day"):
        report.say(f"{day}   {len(part):8d}   {part['shift_x'].median():15.1f}  [{part['shift_x'].quantile(0.05):5.1f}, {part['shift_x'].quantile(0.95):5.1f}]   {part['shift_y'].median():15.1f}   "
                   f"{-part['shift_x'].median():+.1f} px")
    report.say("")
    report.say("BY SHEET (demonstrations only; good fits only)")
    report.say("sheet     pictures   shift x: median  [5 %, 95 %]   share of pictures more than 8 px left of the robot day")
    for sheet, part in good.groupby("sheet"):
        report.say(f"{sheet}   {len(part):8d}   {part['shift_x'].median():15.1f}  [{part['shift_x'].quantile(0.05):5.1f}, {part['shift_x'].quantile(0.95):5.1f}]   {100 * (part['shift_x'] < -8).mean():.0f} %")

    # for the eye: one demonstration picture per day beside a robot picture, the arm region only, with a fixed column marked
    pictures, names = [], []
    left, top, right, bottom = settings["template_box"]
    shown = [("robot day, run 2 step 5", read_picture(run_picture_path(2, 5)))]
    seen_days = set()
    for day, sheet, frame_path in examples:
        if day in seen_days:
            continue
        seen_days.add(day)
        shown.append((f"{day} {sheet}", read_picture(VERIFY_PARAMETERS["raw_folder"] / frame_path)))
    for name, picture in shown:
        cut = picture[top - 60:bottom, left:right].copy()
        for column in (120, 195, 270):
            cv2.line(cut, (column, 0), (column, cut.shape[0] - 1), (0, 255, 0), 1)
        pictures.append(cut)
        names.append(name)
    montage = labelled_montage(pictures, names, columns = 1, cell_width = 780)
    cv2.imencode(".png", montage)[1].tofile(str(VERIFY_FOLDER / "figures" / "own_camera_creep_examples.png"))
    report.say("")
    report.say("written: own_camera_creep_per_picture.csv, figures/own_camera_creep_examples.png")
    report.save()



if __name__ == "__main__":
    main()
