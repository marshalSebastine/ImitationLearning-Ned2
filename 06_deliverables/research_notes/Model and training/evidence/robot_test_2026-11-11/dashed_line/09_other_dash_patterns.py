"""
Step 9 of the dashed-line analysis: how much does the look of the dashes matter? (The exam's dashes are unknown.)

Sheet 05's real pictures are cut into dashes of several patterns (picture_edits.cut_dashes) and given to the models:
the pattern measured on sheet 06 (20 mm dashes, 10 mm gaps), the practice tracks' pattern (35 and 15 mm), short
dashes (10 and 10 mm), long gaps (20 and 20 mm) and a sparse line (10 mm dashes, 20 mm gaps). Every sixth picture of
the placed episodes is used. Scored as in steps 4 and 5; also how often a picture that is not a stop picture gets a
stop chance of 0.5 or more, because a long gap ahead of the puck could look like the end of the line.

Prints the table (09_other_dash_patterns.out.txt). No training; about four minutes.

Run: D:/AI_Project/.venv/Scripts/python.exe 09_other_dash_patterns.py > 09_other_dash_patterns.out.txt
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import cv2
import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, JOINT_NAMES
from line_geometry import moved_line
from model_answers import angle_error_deg, answers, loaded_policy, small_picture
from picture_edits import cut_dashes, dash_pattern

cv2.setNumThreads(2)

PATTERN_PARAMETERS = {
    "patterns_mm": {"as sheet 06 (20 / 10)": (20, 10), "as the practice tracks (35 / 15)": (35, 15), "short dashes (10 / 10)": (10, 10),
                    "long gaps (20 / 20)": (20, 20), "sparse (10 / 20)": (10, 20)},
    "picture_stride": 6,
    "models": ["E all sheets", "E recipe without 05", "E recipe without 06", "A all sheets"],
    "twist_least_deg": 2.0,
    "backward_cm": -0.3
}



def measures(
        rows: pandas.DataFrame,
        answer: dict
) -> dict:
    """Return the scores of one model's answers on the rows."""

    settings = PATTERN_PARAMETERS
    is_path = rows["is_path_row"].to_numpy()
    is_step = ~rows["is_stop"].to_numpy()
    is_twisting = (rows["demo_twist_deg"].abs() >= settings["twist_least_deg"]).to_numpy() & is_step
    is_backward = (rows["demo_forward_cm"] <= settings["backward_cm"]).to_numpy() & is_step
    error = angle_error_deg(answer["angle_deg"][is_path], rows["path_angle_deg"].to_numpy()[is_path])
    return {"path_pictures": int(is_path.sum()), "path_within_45": float((error <= 45.0).mean()), "path_error_median_deg": float(numpy.median(error)),
            "twist_asked_mean_deg": float(numpy.abs(answer["twist_deg"][is_twisting]).mean()),
            "twist_sign_right": float((numpy.sign(answer["twist_deg"][is_twisting]) == numpy.sign(rows["demo_twist_deg"].to_numpy()[is_twisting])).mean()),
            "length_asked_median_cm": float(numpy.median(answer["length_cm"][is_step])),
            "backward_also_asked": float((answer["forward_cm"][is_backward] < 0.0).mean()),
            "false_stop_pictures": float((answer["stop_probability"][is_step] >= 0.5).mean()),
            "stop_chance_on_stop_pictures": float(answer["stop_probability"][~is_step].mean()) if (~is_step).any() else numpy.nan}



def main() -> None:

    settings = PATTERN_PARAMETERS
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 200)
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv").set_index("recording")
    reference = pandas.read_csv(EVIDENCE_FOLDER / "reference_line_05.csv")
    points = reference[["x_cm", "y_cm"]].to_numpy() / 100.0
    rows = pandas.read_csv(EVIDENCE_FOLDER / "04_fold_rows.csv", low_memory = False)
    rows = rows[(rows["sheet"] == 5) & (rows["part"] != "not placed")]
    frames = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "frames.csv", usecols = ["frame_index", "frame_path", "row_in_episode"] + JOINT_NAMES)
    rows = rows.merge(frames, on = "frame_index", how = "left").sort_values(["episode_id", "row_in_episode"]).iloc[::settings["picture_stride"]].reset_index(drop = True)
    cache = numpy.load(ANALYSIS_PARAMETERS["processed_folder"] / "images_wide_160.npy", mmap_mode = "r")
    picture_sets = {"as recorded (solid)": numpy.stack([cache[index] for index in rows["frame_index"]])}
    lines = {}
    for recording in rows["recording"].unique():
        placed = registrations.loc[recording]
        lines[recording] = moved_line(points - points.mean(axis = 0), placed["angle_deg"], (placed["shift_x_cm"] / 100.0, placed["shift_y_cm"] / 100.0))
    full_pictures = [cv2.imread(str(ANALYSIS_PARAMETERS["raw_folder"] / value)) for value in rows["frame_path"]]
    for pattern_name, (dash_mm, gap_mm) in settings["patterns_mm"].items():
        edited = numpy.zeros_like(picture_sets["as recorded (solid)"])
        for position, (_, row) in enumerate(rows.iterrows()):
            line = lines[row["recording"]]
            phase = 0.001 * (sum(ord(letter) for letter in row["recording"]) % (dash_mm + gap_mm))
            is_gap_point = dash_pattern(len(line), dash_mm / 1000.0, gap_mm / 1000.0, phase)
            edited[position] = small_picture(cut_dashes(full_pictures[position], line, is_gap_point, [float(row[name]) for name in JOINT_NAMES])[0])
        picture_sets[pattern_name] = edited
    table = []
    for model_name in settings["models"]:
        policy = loaded_policy(model_name)
        for set_name, pictures in picture_sets.items():
            table.append({"model": model_name, "sheet 05 pictures": set_name, **measures(rows, answers(policy, pictures))})
    table = pandas.DataFrame(table)
    print(f"Sheet 05, {len(rows)} pictures (every {settings['picture_stride']}th of the placed episodes), cut into dashes of several patterns (dash / gap in mm)")
    print(table.round(3).to_string(index = False))



if __name__ == "__main__":
    main()
