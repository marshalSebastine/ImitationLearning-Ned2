"""
Step 6 of the dashed-line analysis: do the dashes matter more where the line bends tightly?

Step 5 showed that on robot run 8's pictures model E asks for a clearly different push around the hairpin once the
dashes are painted solid. This script checks the same on the demonstration pictures, split by how tightly the line
bends where the puck is (radius under 3 cm, 3 to 6 cm, over 6 cm), with the edited pictures kept by step 5 (run step
5 first): sheet 06 as recorded against solid, sheet 05 as recorded against cut into dashes.

Prints the table (06_dashes_at_tight_bends.out.txt). No training; about two minutes.

Run: D:/AI_Project/.venv/Scripts/python.exe 06_dashes_at_tight_bends.py > 06_dashes_at_tight_bends.out.txt
"""

import importlib

import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, JOINT_NAMES
from model_answers import angle_error_deg, answers, loaded_policy

edit_step = importlib.import_module("05_dash_edits_and_models")

BEND_PARAMETERS = {
    "radius_edges_cm": [0.0, 3.0, 6.0, 1000.0],
    "radius_names": ["bend tighter than 3 cm", "bend of 3 to 6 cm", "bend wider than 6 cm or straight"],
    "models": ["E recipe without 06", "E recipe without 06, seed 2", "E recipe without 05", "E all sheets"],
    "twist_least_deg": 2.0
}



def main() -> None:

    settings = BEND_PARAMETERS
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 200)
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv").set_index("recording")
    references = {sheet: pandas.read_csv(EVIDENCE_FOLDER / f"reference_line_{sheet}.csv") for sheet in ("05", "06")}
    labelled = pandas.read_csv(EVIDENCE_FOLDER / "04_fold_rows.csv", low_memory = False)
    labelled["sheet"] = labelled["sheet"].map(lambda value: f"{int(value):02d}")
    labelled = labelled[labelled["part"] != "not placed"]
    frames = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "frames.csv", usecols = ["frame_index", "frame_path", "plan_number", "row_in_episode"] + JOINT_NAMES)
    labelled = labelled.merge(frames, on = "frame_index", how = "left")
    cache = numpy.load(ANALYSIS_PARAMETERS["processed_folder"] / "images_wide_160.npy", mmap_mode = "r")
    policies = {name: loaded_policy(name) for name in settings["models"]}
    table = []
    for sheet in ("06", "05"):
        rows = labelled[labelled["sheet"] == sheet].sort_values(["episode_id", "row_in_episode"])
        if sheet == "05":
            rows = rows.iloc[::edit_step.EDIT_TEST_PARAMETERS["sheet_05_picture_stride"]]
        rows = rows.reset_index(drop = True)
        originals = numpy.stack([cache[index] for index in rows["frame_index"]])
        edited, _, _ = edit_step.edited_pictures(rows, sheet, registrations, references)
        edit_name = "solid" if sheet == "06" else "cut into dashes"
        radius_class = numpy.digitize(rows["line_radius_cm"].to_numpy(), settings["radius_edges_cm"][1:-1])
        is_path = rows["is_path_row"].to_numpy()
        is_twisting = (rows["demo_twist_deg"].abs() >= settings["twist_least_deg"]).to_numpy() & ~rows["is_stop"].to_numpy()
        for model_name in settings["models"]:
            before = answers(policies[model_name], originals)
            after = answers(policies[model_name], edited)
            for number, radius_name in enumerate(settings["radius_names"]):
                mask = (radius_class == number) & (rows["part"] == "loop").to_numpy()
                path_mask = mask & is_path
                twist_mask = mask & is_twisting
                error_before = angle_error_deg(before["angle_deg"][path_mask], rows["path_angle_deg"].to_numpy()[path_mask])
                error_after = angle_error_deg(after["angle_deg"][path_mask], rows["path_angle_deg"].to_numpy()[path_mask])
                table.append({
                    "sheet": sheet, "model": model_name, "in the loop, where the line has a": radius_name, "path_pictures": int(path_mask.sum()),
                    "within_45_as_recorded": float((error_before <= 45).mean()), f"within_45_edited": float((error_after <= 45).mean()),
                    "error_median_as_recorded": float(numpy.median(error_before)), "error_median_edited": float(numpy.median(error_after)),
                    "answer_change_median_deg": float(numpy.median(angle_error_deg(before["angle_deg"][mask], after["angle_deg"][mask]))),
                    "answer_change_over_20_deg": float((angle_error_deg(before["angle_deg"][mask], after["angle_deg"][mask]) > 20).mean()),
                    "twist_asked_as_recorded": float(numpy.abs(before["twist_deg"][twist_mask]).mean()), "twist_asked_edited": float(numpy.abs(after["twist_deg"][twist_mask]).mean()),
                    "twist_demonstrated": float(rows["demo_twist_deg"].abs().to_numpy()[twist_mask].mean()), "edit": edit_name})
    table = pandas.DataFrame(table)
    print("In the loop, by how tightly the line bends where the puck is: the pictures as recorded against edited (sheet 06: line painted solid; sheet 05: cut into dashes)")
    print(table.round(3).to_string(index = False))



if __name__ == "__main__":
    main()
