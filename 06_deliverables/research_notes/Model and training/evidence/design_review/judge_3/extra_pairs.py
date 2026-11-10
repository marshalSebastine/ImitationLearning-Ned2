"""
Two more paired comparisons for judge 3 (9.11.26, read only), reusing the functions of fold_evidence.py:

1. ResNet-18 classes against the small CNN classes, both read as the probability-weighted mean direction;
2. the final candidate against the end-to-end chunk run (first predicted step), on the rows both scored.

Writes extra_pairs.out.txt next to this file.
"""

from pathlib import Path

import numpy
import pandas

import fold_evidence

EXTRA_PARAMETERS = {
    "output_text_path": Path(__file__).with_name("extra_pairs.out.txt"),
    "pairs": [
        ("finetune_classes", "angle_expected_deg", "small_cnn_classes", "angle_expected_deg", ["03", "05", "06"]),
        ("final_candidate", "angle_deg", "finetune_chunk", "angle_deg", ["03", "05"]),
        ("final_candidate", "angle_deg", "finetune_chunk", "angle_vector_deg", ["03", "05"]),
    ],
}



def main(parameters: dict = None) -> None:
    parameters = dict(fold_evidence.ANALYSIS_PARAMETERS, **EXTRA_PARAMETERS, **(parameters or {}))
    columns = ["frame_index", "episode_id", "row_in_episode", "action", "goal_distance_cm", "move_angle_deg",
               "forearm_change_deg", "tip_x_m", "tip_y_m", "picture_heading_deg"]
    frames = pandas.read_csv(parameters["frames_path"], usecols = columns, low_memory = False)
    joined_cache = {}
    lines = []
    for first_run, first_key, second_run, second_key, sheets in parameters["pairs"]:
        for sheet in sheets:
            parts = []
            for run_name in (first_run, second_run):
                if (run_name, sheet) not in joined_cache:
                    folder = fold_evidence.RUN_FOLDS[run_name][0]
                    predictions = dict(numpy.load(parameters["runs_folder"] / folder / f"predictions_without_path_{sheet}.npz"))
                    joined_cache[(run_name, sheet)] = fold_evidence.score_fold(predictions, frames, parameters)[1]
                parts.append(joined_cache[(run_name, sheet)])
            first_joined, second_joined = parts
            shared = numpy.intersect1d(first_joined["frame_index"].to_numpy(), second_joined["frame_index"].to_numpy())
            first_part = first_joined.set_index("frame_index").loc[shared]
            second_part = second_joined.set_index("frame_index").loc[shared]
            moving = first_part["is_moving"].to_numpy() & second_part["is_moving"].to_numpy()
            path_rows = first_part["is_path_row"].to_numpy() & second_part["is_path_row"].to_numpy()
            backward = moving & numpy.isin(first_part["action"].to_numpy(), parameters["backward_names"])
            episodes = first_part["episode_id"].to_numpy()
            shared_path_angle = first_part["path_angle_deg"].to_numpy()
            results = {}
            for measure, first_error, second_error, mask in (
                    ("step", first_part[f"step_error_{first_key}"].to_numpy(), second_part[f"step_error_{second_key}"].to_numpy(), moving),
                    ("path", fold_evidence.angle_difference_deg(shared_path_angle, first_part[f"predicted_{first_key}"].to_numpy()),
                     fold_evidence.angle_difference_deg(shared_path_angle, second_part[f"predicted_{second_key}"].to_numpy()), path_rows),
                    ("backward", first_part[f"step_error_{first_key}"].to_numpy(), second_part[f"step_error_{second_key}"].to_numpy(), backward)):
                first_hits = fold_evidence.episode_hits(first_error, mask, episodes, parameters)
                second_hits = fold_evidence.episode_hits(second_error, mask, episodes, parameters)
                results[measure] = fold_evidence.bootstrap_difference(first_hits, second_hits, parameters)
                results[measure]["first_share"] = round(float((first_error[mask] <= 45.0).mean()), 4)
                results[measure]["second_share"] = round(float((second_error[mask] <= 45.0).mean()), 4)
            line = f"{first_run}.{first_key} minus {second_run}.{second_key}, sheet {sheet} ({len(shared)} shared rows): "
            line += "; ".join(f"{measure} {entry['first_share']} against {entry['second_share']} on {entry['rows']} rows, "
                              f"{entry['difference_points']} {entry['interval_points']}" for measure, entry in results.items())
            lines.append(line)
            print(line, flush = True)
    parameters["output_text_path"].write_text("\n".join(lines), encoding = "utf-8")



if __name__ == "__main__":
    main()
