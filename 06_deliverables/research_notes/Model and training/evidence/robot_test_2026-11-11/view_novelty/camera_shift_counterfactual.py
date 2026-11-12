"""
Does the camera's creep on its mount matter to model E? The robot day's pictures show the robot's own hand about
16 px (of 640) further right than the pictures of sheets 01 to 03 were recorded with (setup_light_and_camera.py).

Test: every robot picture is slid sideways by a few pixels (the border repeated), shrunk as usual, and given to model
E again. Sliding by -16 px puts the robot's hand back where the early demonstrations had it. Two things are read off:
    - the distance to the nearest training pictures as a function of the slide: if the creep is what makes the
      pictures less familiar, the distance is smallest near -16 px for the sheets recorded in the earlier view
      (01 to 03) and near 0 for those recorded in the later view (05, 06);
    - how far model E's answers move when the creep is undone: the size of the creep's effect on the decisions.

A sideways slide is not exactly what a camera turned on its mount sees, but for 16 px (1.8 degrees of view) the two
are close, and the training's own augmentation uses the same kind of slide (up to 4 px at 160 px wide = 16 px here).

Nothing connects to the robot. Run from this folder, after extract_features.py model_e (about one minute):
    D:/AI_Project/.venv/Scripts/python.exe camera_shift_counterfactual.py
Output: camera_shift_counterfactual.out.txt, camera_shift_counterfactual.csv
"""

from typing import Dict, Optional

import cv2
import numpy
import pandas

from common import (EVIDENCE_FOLDER, FAILURE_STEPS, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, load_descriptors, load_policy, picture_descriptors,
                    run_picture_paths, trained_frames, training_frames, unit_rows)
from neighbours import nearest_rows

SHIFT_PARAMETERS = {
    # slides in pixels of the 640 px wide picture; negative = the picture's content moves to the left
    "slides_px": [-32, -24, -16, -8, 0, 8, 16],
    "undo_px": -16,
    "neighbour_count": 5,
    "model_a_run_name": "resnet18_model_1_all_sheets",
    "model_a_folder": "2026-11-10_resnet18_model_1_all_sheets"
}



def slid_small_pictures(
        run_number: int,
        slide_px: int
) -> numpy.ndarray:
    """Return a run's pictures slid sideways by ``slide_px`` at full size (border repeated), then shrunk to 160 x 120 RGB as the model sees them."""

    pictures = []
    for path in run_picture_paths(run_number):
        picture = cv2.imread(str(path), cv2.IMREAD_COLOR)
        height, width = picture.shape[:2]
        slid = cv2.warpAffine(picture, numpy.float32([[1, 0, slide_px], [0, 1, 0]]), (width, height), borderMode = cv2.BORDER_REPLICATE)
        pictures.append(cv2.resize(cv2.cvtColor(slid, cv2.COLOR_BGR2RGB), (160, 120), interpolation = cv2.INTER_AREA))
    return numpy.stack(pictures)



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(SHIFT_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "camera_shift_counterfactual.out.txt")
    policy = load_policy(VIEW_NOVELTY_PARAMETERS["model_e_run_name"], VIEW_NOVELTY_PARAMETERS["model_e_folder"])
    frames = training_frames()
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    reference_index = frames.loc[frames["frame_index"].isin(trained["frame_index"]), "frame_index"].to_numpy()
    reference_unit = unit_rows(load_descriptors("model_e_training")["grid"][reference_index])

    tables = []
    for run_number in ROBOT_RUNS:
        for slide_px in settings["slides_px"]:
            descriptors = picture_descriptors(policy, slid_small_pictures(run_number, slide_px))
            distance = nearest_rows(unit_rows(descriptors["grid"]), reference_unit, {"all": None}, {"count": settings["neighbour_count"]})["all"]["distance"].mean(axis = 1)
            exact = descriptors["exact"]
            tables.append(pandas.DataFrame({"run": run_number, "step": numpy.arange(len(distance)), "slide_px": slide_px, "distance": distance,
                                            "forward_cm": exact[:, 0], "left_cm": exact[:, 1], "twist_deg": exact[:, 2], "stop_probability": descriptors["stop_probability"]}))
    table = pandas.concat(tables).reset_index(drop = True)
    table["length_cm"] = numpy.hypot(table["forward_cm"], table["left_cm"])
    table["angle_deg"] = numpy.degrees(numpy.arctan2(table["left_cm"], table["forward_cm"]))

    def kept(run_number: int) -> pandas.Series:
        # the pictures in which the puck was still on the line: there the scene is one the demonstrations show
        last = FAILURE_STEPS[run_number]["puck_off_line"] if run_number in FAILURE_STEPS else 10_000
        return (table["run"] == run_number) & (table["step"] < last)

    log.say("THE CAMERA'S CREEP: the robot pictures slid sideways before model E sees them (negative = content to the left; -16 px undoes the creep since sheets 01 to 03 were recorded)")
    log.say("")
    log.say("Median distance to the 5 nearest training pictures, per slide (pictures with the puck still on the line; the smallest per run is marked *)")
    log.say("run  sheet     view of that sheet's demonstrations   pictures | " + "  ".join(f"{slide:+5d} px" for slide in settings["slides_px"]))
    views = {"path_01": "earlier", "path_02": "earlier", "path_03": "earlier", "path_04": "72 % earlier", "path_05": "later", "path_06": "later"}
    for run_number, run in ROBOT_RUNS.items():
        part = table[kept(run_number)]
        medians = part.groupby("slide_px")["distance"].median()
        best = medians.idxmin()
        cells = [f"{medians[slide]:7.4f}{'*' if slide == best else ' '}" for slide in settings["slides_px"]]
        log.say(f"{run_number:3d}  {run['sheet']:8s}  {views[run['sheet']]:35s}  {int((part['slide_px'] == 0).sum()):8d} | " + "  ".join(cells))

    log.say("")
    log.say(f"How far model E's answers move when the creep is undone (slide {settings['undo_px']} px against no slide; the same pictures; medians, 90 % point in brackets)")
    log.say("run  pictures | direction change deg | step vector change cm | asked length cm: as seen, creep undone | twist change deg (abs) | stop chance: largest change")
    for run_number in ROBOT_RUNS:
        seen = table[kept(run_number) & (table["slide_px"] == 0)].set_index("step")
        undone = table[kept(run_number) & (table["slide_px"] == settings["undo_px"])].set_index("step")
        angle = numpy.abs((undone["angle_deg"] - seen["angle_deg"] + 180.0) % 360.0 - 180.0)
        vector = numpy.hypot(undone["forward_cm"] - seen["forward_cm"], undone["left_cm"] - seen["left_cm"])
        twist = (undone["twist_deg"] - seen["twist_deg"]).abs()
        log.say(f"{run_number:3d}  {len(seen):8d} | {angle.median():11.1f} ({angle.quantile(0.9):5.1f}) | {vector.median():12.2f} ({vector.quantile(0.9):4.2f}) | "
                f"{seen['length_cm'].median():25.2f}, {undone['length_cm'].median():5.2f} | {twist.median():13.2f} ({twist.quantile(0.9):4.2f}) | {(undone['stop_probability'] - seen['stop_probability']).abs().max():.3f}")

    log.say("")
    log.say("The first wrong decisions of the failed runs, as seen and with the creep undone (model E's answer: cm at degrees, twist)")
    for run_number, failure in FAILURE_STEPS.items():
        for step in range(failure["first_wrong_decision"], min(failure["first_wrong_decision"] + 3, failure["puck_off_line"] + 1)):
            seen = table[(table["run"] == run_number) & (table["step"] == step) & (table["slide_px"] == 0)].iloc[0]
            undone = table[(table["run"] == run_number) & (table["step"] == step) & (table["slide_px"] == settings["undo_px"])].iloc[0]
            log.say(f"  run {run_number:2d} step {step:2d}: as seen {seen['length_cm']:.2f} at {seen['angle_deg']:+4.0f}, twist {seen['twist_deg']:+5.1f}  |  "
                    f"creep undone {undone['length_cm']:.2f} at {undone['angle_deg']:+4.0f}, twist {undone['twist_deg']:+5.1f}")

    log.say("")
    log.say("Which way the creep pushes the asked direction: the signed change of direction when the picture is slid (median over the pictures with the puck on the line; positive = more to the left)")
    log.say("run  pictures | " + "  ".join(f"slide {slide:+3d} px" for slide in settings["slides_px"] if slide != 0))
    for run_number in ROBOT_RUNS:
        seen = table[kept(run_number) & (table["slide_px"] == 0)].set_index("step")
        cells = []
        for slide_px in settings["slides_px"]:
            if slide_px == 0:
                continue
            slid = table[kept(run_number) & (table["slide_px"] == slide_px)].set_index("step")
            cells.append(f"{(((slid['angle_deg'] - seen['angle_deg'] + 180.0) % 360.0) - 180.0).median():+12.1f}")
        log.say(f"{run_number:3d}  {len(seen):8d} | " + "  ".join(cells))

    # run 1 was driven by model A, so its own sensitivity is measured too
    model_a = load_policy(settings["model_a_run_name"], settings["model_a_folder"])
    answers = {}
    for slide_px in (settings["undo_px"], 0):
        exact = picture_descriptors(model_a, slid_small_pictures(1, slide_px))["exact"]
        answers[slide_px] = numpy.degrees(numpy.arctan2(exact[:, 1], exact[:, 0]))
    change = ((answers[settings["undo_px"]] - answers[0] + 180.0) % 360.0) - 180.0
    log.say("")
    log.say(f"Run 1 with the model that drove it (model A): the asked direction with the creep undone against as seen, over its {len(change)} pictures: "
            f"median {numpy.median(change):+.1f} degrees (positive = more to the left), middle half {numpy.quantile(change, 0.25):+.1f} to {numpy.quantile(change, 0.75):+.1f}; "
            f"steps 0 to 14: median {numpy.median(change[:15]):+.1f}")

    table.round(5).to_csv(EVIDENCE_FOLDER / "camera_shift_counterfactual.csv", index = False)
    log.say("")
    log.say("written: camera_shift_counterfactual.csv")
    log.save()



if __name__ == "__main__":
    main()
