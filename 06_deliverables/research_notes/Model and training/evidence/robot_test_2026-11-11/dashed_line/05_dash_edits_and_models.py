"""
Step 5 of the dashed-line analysis: do the dashes themselves change what the models ask for?

Two edits of the real pictures (picture_edits.py) answer it without the robot and without training:
    sheet 06 with the gaps painted over (a solid line): if the dashes confuse a model, its answers should get better;
    sheet 05 with gaps cut into the line (dashes of 2.0 cm, gaps of 1.0 cm as measured on sheet 06): if dashes confuse
    a model, its answers should get worse.
Models: the fold model that never saw sheet 06, so never a dashed line (two seeds); the fold model that never saw
sheet 05; and model E (all sheets), which drove the robot. Scored as in step 4 (the exact step, which the driver
carries out). The same is done for the 41 pictures of robot run 8: would model E have decided differently on a
solid line?

Writes 05_dash_edits_and_models.png, 05_edit_examples.png, 05_run_8_filled.csv and prints the tables. No training;
about six minutes (the edits are kept in the system's temporary folder for a second run).

Run: D:/AI_Project/.venv/Scripts/python.exe 05_dash_edits_and_models.py > 05_dash_edits_and_models.out.txt
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")

import tempfile
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
import numpy
import pandas

from dashed_common import ANALYSIS_PARAMETERS, EVIDENCE_FOLDER, JOINT_NAMES, read_run
from line_geometry import moved_line
from model_answers import angle_error_deg, answers, loaded_policy, small_picture
from picture_edits import cut_dashes, dash_pattern, filled_gaps

cv2.setNumThreads(2)

EDIT_TEST_PARAMETERS = {
    "models": {"06": ["E recipe without 06", "E recipe without 06, seed 2", "E recipe without 05", "E all sheets"],
               "05": ["E recipe without 06", "E recipe without 06, seed 2", "E recipe without 05", "E all sheets"]},
    # the dashes cut into sheet 05: as measured on sheet 06 in step 1; the phase changes from episode to episode
    "dash_m": 0.020,
    "gap_m": 0.010,
    "sheet_05_picture_stride": 2,
    "twist_least_deg": 2.0,
    "backward_cm": -0.3,
    "cache_folder": Path(tempfile.gettempdir()) / "dashed_line_edits_2026-11-11",
    # changed whenever picture_edits.py changes, so that kept edits are not reused by mistake
    "edit_version": {"06": "whole_line_v2", "05": "cut_v1"},
    "example_frames": {"06": [("06-03", 8), ("06-03", 20), ("06-06", 60)], "05": [("05-03", 12), ("05-03", 30), ("05-06", 70)]}
}



def placed_line(
        recording: str,
        sheet: str,
        registrations: pandas.DataFrame,
        references: dict
) -> tuple:
    """Return a recording's centre line on the table (in the reference's order) and which of its points are ink."""

    reference = references[sheet][["x_cm", "y_cm"]].to_numpy() / 100.0
    row = registrations.loc[recording]
    line = moved_line(reference - reference.mean(axis = 0), row["angle_deg"], (row["shift_x_cm"] / 100.0, row["shift_y_cm"] / 100.0))
    return line, references[sheet]["is_ink"].to_numpy() > 0



def edited_pictures(
        rows: pandas.DataFrame,
        sheet: str,
        registrations: pandas.DataFrame,
        references: dict
) -> tuple:
    """
    Return the edited small pictures of a sheet's rows (gaps filled on 06, dashes cut on 05) and how much was painted.

    The result is kept in the temporary folder, so a second run of the script does not edit again.
    """

    settings = EDIT_TEST_PARAMETERS
    settings["cache_folder"].mkdir(parents = True, exist_ok = True)
    cache_path = settings["cache_folder"] / f"sheet_{sheet}_{len(rows)}_pictures_{settings['edit_version'][sheet]}.npz"
    if cache_path.exists():
        cached = numpy.load(cache_path)
        if numpy.array_equal(cached["frame_index"], rows["frame_index"].to_numpy()):
            return cached["pictures"], cached["painted_px"], cached["shift_px"]
    pictures = numpy.zeros((len(rows), 120, 160, 3), numpy.uint8)
    painted = numpy.zeros(len(rows), int)
    shifts = numpy.zeros((len(rows), 2), int)
    lines = {}
    for position, (_, row) in enumerate(rows.iterrows()):
        if row["recording"] not in lines:
            line, is_ink_point = placed_line(row["recording"], sheet, registrations, references)
            # the cut's phase differs from episode to episode, so the gaps do not always fall on the same places
            phase = 0.001 * (sum(ord(letter) for letter in row["recording"]) % 30)
            lines[row["recording"]] = (line, is_ink_point, dash_pattern(len(line), settings["dash_m"], settings["gap_m"], phase))
        line, is_ink_point, is_gap_point = lines[row["recording"]]
        picture = cv2.imread(str(ANALYSIS_PARAMETERS["raw_folder"] / row["frame_path"]))
        joints = [float(row[name]) for name in JOINT_NAMES]
        if sheet == "06":
            edited, information = filled_gaps(picture, line, is_ink_point, joints)
        else:
            edited, information = cut_dashes(picture, line, is_gap_point, joints)
        pictures[position] = small_picture(edited)
        painted[position] = information["painted_px"]
        shifts[position] = information["shift_px"]
    numpy.savez_compressed(cache_path, frame_index = rows["frame_index"].to_numpy(), pictures = pictures, painted_px = painted, shift_px = shifts)
    return pictures, painted, shifts



def scored(
        rows: pandas.DataFrame,
        answer: dict
) -> dict:
    """Return the measures of one model's answers on rows (as in step 4)."""

    settings = EDIT_TEST_PARAMETERS
    result = {"pictures": len(rows)}
    is_path = rows["is_path_row"].to_numpy()
    error = angle_error_deg(answer["angle_deg"][is_path], rows["path_angle_deg"].to_numpy()[is_path])
    result.update({"path_pictures": int(is_path.sum()), "path_within_45": float((error <= 45.0).mean()), "path_error_median_deg": float(numpy.median(error))})
    is_twisting = (rows["demo_twist_deg"].abs() >= settings["twist_least_deg"]).to_numpy() & ~rows["is_stop"].to_numpy()
    result.update({"twist_sign_right": float((numpy.sign(answer["twist_deg"][is_twisting]) == numpy.sign(rows["demo_twist_deg"].to_numpy()[is_twisting])).mean()),
                   "twist_asked_mean_deg": float(numpy.abs(answer["twist_deg"][is_twisting]).mean())})
    is_step = ~rows["is_stop"].to_numpy()
    result.update({"length_asked_median_cm": float(numpy.median(answer["length_cm"][is_step])), "stop_chance_mean": float(answer["stop_probability"][is_step].mean())})
    is_backward = (rows["demo_forward_cm"] <= settings["backward_cm"]).to_numpy() & is_step
    result["backward_also_asked"] = float((answer["forward_cm"][is_backward] < 0.0).mean()) if is_backward.any() else numpy.nan
    is_stop = rows["is_stop"].to_numpy()
    result["stop_chance_on_stop_pictures"] = float(answer["stop_probability"][is_stop].mean()) if is_stop.any() else numpy.nan
    return result



def main() -> None:

    settings = EDIT_TEST_PARAMETERS
    pandas.set_option("display.width", 260)
    pandas.set_option("display.max_columns", 40)
    pandas.set_option("display.max_rows", 200)
    started = time.time()
    registrations = pandas.read_csv(EVIDENCE_FOLDER / "registrations.csv").set_index("recording")
    references = {sheet: pandas.read_csv(EVIDENCE_FOLDER / f"reference_line_{sheet}.csv") for sheet in ("05", "06")}
    labelled = pandas.read_csv(EVIDENCE_FOLDER / "04_fold_rows.csv", low_memory = False)
    labelled["sheet"] = labelled["sheet"].map(lambda value: f"{int(value):02d}")
    labelled = labelled[labelled["part"] != "not placed"]
    frames = pandas.read_csv(ANALYSIS_PARAMETERS["processed_folder"] / "frames.csv", usecols = ["frame_index", "frame_path", "plan_number", "row_in_episode"] + JOINT_NAMES)
    labelled = labelled.merge(frames, on = "frame_index", how = "left")
    cache = numpy.load(ANALYSIS_PARAMETERS["processed_folder"] / "images_wide_160.npy", mmap_mode = "r")
    policies = {}
    result_rows, change_rows = [], []
    examples = {}
    for sheet in ("06", "05"):
        rows = labelled[labelled["sheet"] == sheet].sort_values(["episode_id", "row_in_episode"])
        if sheet == "05":
            rows = rows.iloc[::settings["sheet_05_picture_stride"]]
        rows = rows.reset_index(drop = True)
        originals = numpy.stack([cache[index] for index in rows["frame_index"]])
        edited, painted, shifts = edited_pictures(rows, sheet, registrations, references)
        edit_name = "gaps filled" if sheet == "06" else "dashes cut"
        changed = numpy.abs(edited.astype(int) - originals.astype(int)).reshape(len(rows), -1).mean(axis = 1)
        print(f"\nsheet {sheet}: {len(rows)} pictures, edit '{edit_name}': painted in {int((painted > 0).sum())} pictures (median {numpy.median(painted[painted > 0]):.0f} px of "
              f"307,200 at full size), mean grey change per small picture {changed.mean():.2f} levels; alignment shift median {numpy.median(numpy.hypot(*shifts.T)):.1f} px, "
              f"largest {numpy.hypot(*shifts.T).max():.0f} px ({time.time() - started:.0f} s)")
        for plan, row_number in settings["example_frames"][sheet]:
            found = rows[(rows["plan_number"] == plan) & (rows["row_in_episode"] >= row_number)]
            if len(found):
                examples[(sheet, plan, int(found["row_in_episode"].iloc[0]))] = (originals[found.index[0]], edited[found.index[0]])
        for model_name in settings["models"][sheet]:
            if model_name not in policies:
                policies[model_name] = loaded_policy(model_name)
            before = answers(policies[model_name], originals)
            after = answers(policies[model_name], edited)
            parts = [("whole sheet", numpy.ones(len(rows), bool))] + [(part, (rows["part"] == part).to_numpy()) for part in ("before the crossing", "loop", "after the loop")]
            for part, mask in parts:
                subset = rows[mask]
                for condition, answer in (("as recorded", before), (edit_name, after)):
                    result_rows.append({"sheet": sheet, "model": model_name, "part": part, "pictures as": condition,
                                        **scored(subset, {name: value[mask] for name, value in answer.items() if name != "chunk_cm"})})
                is_step = mask & ~rows["is_stop"].to_numpy()
                change_rows.append({"sheet": sheet, "model": model_name, "part": part, "pictures": int(is_step.sum()),
                                    "direction_change_median_deg": float(numpy.median(angle_error_deg(before["angle_deg"][is_step], after["angle_deg"][is_step]))),
                                    "direction_change_over_45_deg": float((angle_error_deg(before["angle_deg"][is_step], after["angle_deg"][is_step]) > 45.0).mean()),
                                    "twist_change_mean_abs_deg": float(numpy.abs(before["twist_deg"][is_step] - after["twist_deg"][is_step]).mean()),
                                    "length_change_mean_abs_cm": float(numpy.abs(before["length_cm"][is_step] - after["length_cm"][is_step]).mean()),
                                    "stop_chance_change_mean_abs": float(numpy.abs(before["stop_probability"][is_step] - after["stop_probability"][is_step]).mean())})
    results = pandas.DataFrame(result_rows)
    changes = pandas.DataFrame(change_rows)
    print("\nScores on the pictures as recorded and edited (path within 45: asked direction within 45 degrees of the demonstrated path)")
    print(results.round(3).to_string(index = False))
    print("\nHow much the same model's answer changes between the recorded and the edited picture")
    print(changes.round(3).to_string(index = False))
    results.round(4).to_csv(EVIDENCE_FOLDER / "05_edit_scores.csv", index = False)

    # robot run 8 with the gaps filled
    steps = read_run(ANALYSIS_PARAMETERS["run_8_folder"])
    line, is_ink_point = placed_line("run_8", "06", registrations, references)
    originals, filled, painted = [], [], []
    for _, step in steps.iterrows():
        picture = cv2.imread(step["frame_file"])
        edited_picture, information = filled_gaps(picture, line, is_ink_point, [float(step[name]) for name in JOINT_NAMES])
        originals.append(small_picture(picture))
        filled.append(small_picture(edited_picture))
        painted.append(information["painted_px"])
    originals, filled = numpy.stack(originals), numpy.stack(filled)
    run_rows = pandas.DataFrame({"step": steps["step"], "painted_px": painted})
    for model_name in ("E all sheets", "E recipe without 06"):
        if model_name not in policies:
            policies[model_name] = loaded_policy(model_name)
        before = answers(policies[model_name], originals)
        after = answers(policies[model_name], filled)
        short = "E" if model_name == "E all sheets" else "never_saw_06"
        if model_name == "E all sheets":
            print(f"\ncheck: model E on run 8's pictures as recorded against the run's log: largest difference in the asked direction "
                  f"{numpy.abs(before['forward_cm'] - steps['model_tip_forward_cm']).max():.5f} cm forward, stop chance {numpy.abs(before['stop_probability'] - steps['model_stop_probability']).max():.5f}")
        run_rows[f"{short}_angle"] = before["angle_deg"]
        run_rows[f"{short}_angle_filled"] = after["angle_deg"]
        run_rows[f"{short}_angle_change"] = angle_error_deg(before["angle_deg"], after["angle_deg"])
        run_rows[f"{short}_length"] = before["length_cm"]
        run_rows[f"{short}_length_filled"] = after["length_cm"]
        run_rows[f"{short}_twist"] = before["twist_deg"]
        run_rows[f"{short}_twist_filled"] = after["twist_deg"]
        run_rows[f"{short}_stop"] = before["stop_probability"]
        run_rows[f"{short}_stop_filled"] = after["stop_probability"]
    print("\nRobot run 8, picture by picture: model E (which drove) and the fold model that never saw sheet 06, on the pictures as recorded and with the gaps filled")
    print(run_rows.round(2).to_string(index = False))
    moved = run_rows[run_rows["painted_px"] > 0]
    print(f"\nmodel E on run 8: direction change with the gaps filled: median {run_rows['E_angle_change'].median():.1f} deg, largest {run_rows['E_angle_change'].max():.1f} deg "
          f"(step {int(run_rows.loc[run_rows['E_angle_change'].idxmax(), 'step'])}); steps with a change over 20 deg: {int((run_rows['E_angle_change'] > 20).sum())} of {len(run_rows)}; "
          f"twist asked in total {run_rows['E_twist'].sum():.1f} deg as recorded, {run_rows['E_twist_filled'].sum():.1f} deg filled; "
          f"backward steps asked: {int((numpy.abs(run_rows['E_angle']) > 90).sum())} as recorded, {int((numpy.abs(run_rows['E_angle_filled']) > 90).sum())} filled; "
          f"stop chance at steps 39 and 40: {run_rows['E_stop'].iloc[39]:.2f}, {run_rows['E_stop'].iloc[40]:.2f} as recorded, "
          f"{run_rows['E_stop_filled'].iloc[39]:.2f}, {run_rows['E_stop_filled'].iloc[40]:.2f} filled ({len(moved)} of {len(run_rows)} pictures had gaps in view to fill)")
    run_rows.round(4).to_csv(EVIDENCE_FOLDER / "05_run_8_filled.csv", index = False)

    # figures
    figure, axes = pyplot.subplots(2, len(examples), figsize = (3.4 * len(examples), 5.6))
    for column, ((sheet, plan, row_number), (original, edited_picture)) in enumerate(examples.items()):
        axes[0, column].imshow(original)
        axes[0, column].set_title(f"plan {plan}, picture {row_number}: as recorded", fontsize = 8)
        axes[1, column].imshow(edited_picture)
        axes[1, column].set_title("gaps filled" if sheet == "06" else "dashes cut (2.0 cm, gaps 1.0 cm)", fontsize = 8)
        for axis in axes[:, column]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("The two edits at the size the models see (160 x 120)", fontsize = 10)
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "05_edit_examples.png", dpi = 110)

    figure, axes = pyplot.subplots(1, 2, figsize = (17, 6))
    for axis, sheet in zip(axes, ("06", "05")):
        edit_name = "gaps filled" if sheet == "06" else "dashes cut"
        models = settings["models"][sheet]
        parts = ["before the crossing", "loop", "after the loop"]
        width = 0.8 / (2 * len(models))
        for number, model_name in enumerate(models):
            for offset, (condition, hatch) in enumerate((("as recorded", ""), (edit_name, "//"))):
                values = [results[(results["sheet"] == sheet) & (results["model"] == model_name) & (results["part"] == part) & (results["pictures as"] == condition)]["path_within_45"].iloc[0]
                          for part in parts]
                axis.bar(numpy.arange(len(parts)) - 0.4 + (2 * number + offset + 0.5) * width, values, width = width, color = f"C{number}", alpha = 1.0 if offset == 0 else 0.55,
                         hatch = hatch, edgecolor = "white", label = f"{model_name}: {condition}")
        axis.set_xticks(numpy.arange(len(parts)))
        axis.set_xticklabels(parts)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("share within 45 degrees of the demonstrated path")
        axis.set_title(f"Sheet {sheet} pictures as recorded and with {edit_name} (hatched)")
        axis.grid(True, axis = "y", linewidth = 0.3)
        axis.legend(fontsize = 7, loc = "lower left", ncol = 2)
    figure.tight_layout()
    figure.savefig(EVIDENCE_FOLDER / "05_dash_edits_and_models.png", dpi = 70)
    print(f"\ntotal {time.time() - started:.0f} s")



if __name__ == "__main__":
    main()
