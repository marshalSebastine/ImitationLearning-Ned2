"""
One block of measured numbers per robot run of 11.11.26, gathered from the tables the other scripts of this folder
wrote: were the pictures unfamiliar BEFORE the run went wrong, or only AFTER?

Nothing is computed anew except simple counts; every number names the table it comes from. A picture counts as
"unfamiliar" when its distance to the nearest demonstration pictures lies above the 95 % point of the demonstrations'
own distances on that sheet, in model E's descriptors or in the descriptors of the fold model that never saw the sheet.

Also printed: how the training pictures of each sheet split over the two camera periods the data audit found (the
camera moved on its mount during the break of 8.11.26, 18:07 to 19:33).

Run from this folder after every other script (a second):
    D:/AI_Project/.venv/Scripts/python.exe per_run_summary.py
Output: per_run_summary.out.txt, per_run_summary.csv
"""

import math
from typing import Dict, Optional

import numpy
import pandas

from common import EVIDENCE_FOLDER, FAILURE_STEPS, ROBOT_RUNS, VIEW_NOVELTY_PARAMETERS, OutputLog, run_steps, trained_frames
from config import TRAINING_ERROR_ANALYSIS_PARAMETERS

SUMMARY_PARAMETERS = {
    "unfamiliar_above_percentile": 95.0,
    "hoodie_in_view_share": 0.005,
    "demonstration_start": {"tip_x_m": 0.140, "tip_y_m": -0.015}
}



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(SUMMARY_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "per_run_summary.out.txt")
    model_e = pandas.read_csv(EVIDENCE_FOLDER / "novelty_model_e_per_picture.csv")
    fold = pandas.read_csv(EVIDENCE_FOLDER / "novelty_fold_models_per_picture.csv")
    hoodie = pandas.read_csv(EVIDENCE_FOLDER / "hoodie_check_per_picture.csv")
    labels = pandas.read_csv(EVIDENCE_FOLDER / "key_frames_neighbour_labels.csv")
    light = pandas.read_csv(EVIDENCE_FOLDER / "setup_light_and_camera_per_picture.csv")
    pairs = pandas.read_csv(EVIDENCE_FOLDER / "stale_pictures_per_pair.csv")
    limit = settings["unfamiliar_above_percentile"]

    log.say("PER RUN: were the pictures unfamiliar before the run went wrong? (every number is measured; the table it comes from is named in brackets)")
    log.say(f"'unfamiliar' = above the {limit:.0f} % point of the demonstrations' own distances on that sheet; E = model E's descriptors [novelty_model_e_per_picture.csv], "
            "fair = the fold model that never saw the sheet [novelty_fold_models_per_picture.csv]")
    rows = []
    for run_number, run in ROBOT_RUNS.items():
        steps = run_steps(run_number)
        e_rows, fold_rows = model_e[model_e["run"] == run_number], fold[fold["run"] == run_number]
        hoodie_rows, label_rows = hoodie[hoodie["run"] == run_number], labels[labels["run"] == run_number]
        start = steps.iloc[0]
        forward = 100 * (start["tip_x"] - settings["demonstration_start"]["tip_x_m"])
        left = 100 * (start["tip_y"] - settings["demonstration_start"]["tip_y_m"])
        hoodie_steps = hoodie_rows.loc[hoodie_rows["royal"] > settings["hoodie_in_view_share"], "step"]
        record = {"run": run_number, "model": run["model"], "sheet": run["sheet"], "pictures": len(steps), "ending": "went wrong" if run_number in FAILURE_STEPS else "reached the end",
                  "start_from_demonstration_start_cm": round(math.hypot(forward, left), 1), "start_forearm_twist_deg": round(math.degrees(start["joint_4"]), 1),
                  "hoodie_first_step": int(hoodie_steps.min()) if len(hoodie_steps) else None, "hoodie_pictures": int(len(hoodie_steps)),
                  "hoodie_largest_share_percent": round(100 * hoodie_rows["royal"].max(), 1),
                  "brightness_median": round(light.loc[light["run"] == run_number, "brightness_mean"].median(), 1),
                  "camera_strip_shift_x_px": round(light.loc[light["run"] == run_number, "strip_shift_x_px"].median(), 1),
                  "byte_identical_pairs": int(pairs.loc[pairs["run"] == run_number, "is_same_bytes"].sum())}

        log.say("")
        log.say(f"RUN {run_number} (sheet {run['sheet'][-2:]}, model {run['model']}, {len(steps)} pictures): {record['ending']}; {run['note']}")
        log.say(f"  start: tip {forward:+.1f} cm forward and {left:+.1f} cm left of the demonstrations' start ({record['start_from_demonstration_start_cm']} cm away), forearm twist {record['start_forearm_twist_deg']:+.1f} degrees "
                "(demonstrations: 0) [steps.csv, frames.csv]")
        if run_number in FAILURE_STEPS:
            wrong, off_line = FAILURE_STEPS[run_number]["first_wrong_decision"], FAILURE_STEPS[run_number]["puck_off_line"]
            log.say(f"  went wrong: first wrong decision at step {wrong}, puck off the line at step {off_line} (by eye, figures/failure_strip_run_{run_number:02d}.png): {FAILURE_STEPS[run_number]['reason']}")
            phases = {"before": (e_rows["step"] <= wrong, fold_rows["step"] <= wrong, label_rows["step"] <= wrong),
                      "between": ((e_rows["step"] > wrong) & (e_rows["step"] < off_line), (fold_rows["step"] > wrong) & (fold_rows["step"] < off_line), (label_rows["step"] > wrong) & (label_rows["step"] < off_line)),
                      "after": (e_rows["step"] >= off_line, fold_rows["step"] >= off_line, label_rows["step"] >= off_line)}
            names = {"before": f"seen up to the first wrong decision (steps 0 to {wrong})", "between": f"between (steps {wrong + 1} to {off_line - 1})", "after": f"after the puck left the line (steps {off_line} on)"}
        else:
            phases = {"whole": (e_rows["step"] >= 0, fold_rows["step"] >= 0, label_rows["step"] >= 0)}
            names = {"whole": "the whole run"}
        for phase, (is_e, is_fold, is_label) in phases.items():
            e_part, fold_part, label_part = e_rows[is_e], fold_rows[is_fold], label_rows[is_label]
            if e_part.empty:
                continue
            record.update({f"{phase}_pictures": len(e_part),
                           f"{phase}_e_percentile_median": round(e_part["grid_percentile_in_sheet"].median()), f"{phase}_e_percentile_largest": round(e_part["grid_percentile_in_sheet"].max()),
                           f"{phase}_e_unfamiliar_pictures": int((e_part["grid_percentile_in_sheet"] > limit).sum()),
                           f"{phase}_fair_percentile_median": round(fold_part["fold_percentile"].median()), f"{phase}_fair_percentile_largest": round(fold_part["fold_percentile"].max()),
                           f"{phase}_fair_unfamiliar_pictures": int((fold_part["fold_percentile"] > limit).sum()),
                           f"{phase}_same_sheet_neighbours_percent": round(100 * e_part["grid_same_sheet_neighbours"].mean()),
                           f"{phase}_model_step_cm": round(label_part["model_length_cm"].median(), 2), f"{phase}_neighbours_step_cm": round(label_part["neighbour_mean_step_length_cm"].median(), 2),
                           f"{phase}_model_twist_abs_deg": round(label_part["model_twist_deg"].abs().median(), 1), f"{phase}_neighbours_twist_abs_deg": round(label_part["neighbour_twist_deg"].abs().median(), 1)})
            log.say(f"  {names[phase]}, {len(e_part)} pictures:")
            log.say(f"      place among the demonstrations' own distances: E median {record[f'{phase}_e_percentile_median']}, largest {record[f'{phase}_e_percentile_largest']}, unfamiliar {record[f'{phase}_e_unfamiliar_pictures']} of {len(e_part)}; "
                    f"fair median {record[f'{phase}_fair_percentile_median']}, largest {record[f'{phase}_fair_percentile_largest']}, unfamiliar {record[f'{phase}_fair_unfamiliar_pictures']} of {len(fold_part)}")
            log.say(f"      the 5 nearest demonstration pictures: {record[f'{phase}_same_sheet_neighbours_percent']} % from the same sheet; their demonstrated step {record[f'{phase}_neighbours_step_cm']} cm and twist "
                    f"{record[f'{phase}_neighbours_twist_abs_deg']} degrees (medians), model E asked {record[f'{phase}_model_step_cm']} cm and {record[f'{phase}_model_twist_abs_deg']} degrees; the robot moved 1.00 cm "
                    "[key_frames_neighbour_labels.csv]")
        if len(hoodie_steps):
            before = int((hoodie_steps <= FAILURE_STEPS[run_number]["first_wrong_decision"]).sum()) if run_number in FAILURE_STEPS else None
            log.say(f"  hoodie (royal blue, more than 0.5 % of the picture): {len(hoodie_steps)} pictures, first at step {int(hoodie_steps.min())}, largest share {record['hoodie_largest_share_percent']} %"
                    + (f"; in view in {before} of the pictures seen up to the first wrong decision" if before is not None else "") + " [hoodie_check_per_picture.csv]")
        else:
            log.say("  hoodie (royal blue, more than 0.5 % of the picture): never in view [hoodie_check_per_picture.csv]")
        log.say(f"  light and camera: brightness median {record['brightness_median']} (training pictures: median 143.8, 5 % to 95 %: 119.2 to 157.6); bottom strip {record['camera_strip_shift_x_px']:+.1f} px right of the 6.11 reference "
                f"(last recording day +13.6) [setup_light_and_camera_per_picture.csv]; byte-identical consecutive pictures: {record['byte_identical_pairs']} [stale_pictures_per_pair.csv]")
        rows.append(record)

    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    settled = pandas.Timestamp(TRAINING_ERROR_ANALYSIS_PARAMETERS["camera_settled_at"])
    moved = pandas.Timestamp(TRAINING_ERROR_ANALYSIS_PARAMETERS["camera_moved_from"])
    started = pandas.to_datetime(trained["started"])
    period = numpy.where(started >= settled, "later view", numpy.where(started < moved, "earlier view", "during the move"))
    log.say("")
    log.say("THE TWO CAMERA PERIODS IN THE TRAINING DATA (the audit: the camera moved on its mount on 8.11.26 between 18:07 and 19:33; the robot day's view is the later one, about 3 px further)")
    log.say("sheet     training pictures: earlier view   later view   share in the later view")
    for sheet, group in trained.assign(period = period).groupby("sheet"):
        earlier, later = int((group["period"] == "earlier view").sum()), int((group["period"] == "later view").sum())
        log.say(f"{sheet:8s}  {earlier:31d}   {later:10d}   {100 * later / max(len(group), 1):22.0f} %")

    pandas.DataFrame(rows).to_csv(EVIDENCE_FOLDER / "per_run_summary.csv", index = False)
    log.say("")
    log.say("written: per_run_summary.csv")
    log.save()



if __name__ == "__main__":
    main()
