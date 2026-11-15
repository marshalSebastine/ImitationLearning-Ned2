"""
Do the stops that the held-wrist view loses belong to the episodes whose end view really moves? (key: wrist_and_camera,
part 9; reads the row table that held_wrist_offline_scores.py saved, so it runs in seconds)

held_wrist_offline_scores.py found fewer right stops when every held-out picture is changed to the driver's view. If
the wrist is the reason, the loss must sit in the episodes whose last pictures are moved far (their wrist was far from
13.5 degrees at the end), not in the episodes whose last pictures hardly change. The episodes are split by how far the
stop row's content is moved, and the project's stop rule (0.5 on two pictures in a row, training_metrics.stop_metrics)
is applied to each group as recorded and in the held-wrist view.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe held_wrist_offline_by_shift.py
"""

from typing import Dict, Optional

import numpy
import pandas

import wrist_geometry
from training_metrics import stop_metrics


SPLIT_PARAMETERS = {
    "stop_threshold": 0.5,
    "stop_pictures_in_a_row": 2,
    # groups of episodes by how far the content of the last picture is moved, in px at 160 (4 px = the training augmentation)
    "groups_px": [[0, 4], [5, 9], [10, 40]]
}



def outcomes(
        rows: pandas.DataFrame,
        column: str,
        parameters: Dict
) -> Dict[str, int]:
    """Episodes that stop right, too early or never under the project's stop rule, for one stop-chance column."""

    result = stop_metrics((rows["action"] == "stop").to_numpy().astype(float), rows[column].to_numpy(), rows["goal_distance_cm"].fillna(0.0).to_numpy(),
                          rows["episode_id"].to_numpy(), threshold = parameters["stop_threshold"], consecutive_pictures = parameters["stop_pictures_in_a_row"])
    episodes = rows["episode_id"].nunique()
    return {"right": int(round(result["episodes_stopped_right"] * episodes)), "early": int(round(result["episodes_stopped_early"] * episodes)),
            "never": int(round(result["episodes_never_stopped"] * episodes))}



def main(parameters: Optional[Dict] = None) -> None:
    """Print the stop outcomes by how far the episode's last picture is moved in the held-wrist view."""

    settings = dict(SPLIT_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    output_folder = wrist_geometry.ANALYSIS_PARAMETERS["output_folder"]
    rows = pandas.read_csv(output_folder / "held_wrist_offline_rows.csv")
    index = pandas.read_csv(wrist_geometry.ANALYSIS_PARAMETERS["frames_file"], usecols = ["frame_index", "episode_id"])
    rows = rows.merge(index, on = "frame_index", how = "left").sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
    last_shift = rows.groupby("episode_id")["shift_160_px"].transform("last").abs()
    last_signed = rows.groupby("episode_id")["shift_160_px"].transform("last")

    table = []
    for low, high in settings["groups_px"]:
        part = rows[(last_shift >= low) & (last_shift <= high)]
        recorded = outcomes(part, "stop_as_recorded", settings)
        changed = outcomes(part, "stop_held_view", settings)
        table.append({"last picture moved by (px at 160)": f"{low} to {high}", "episodes": part["episode_id"].nunique(),
                      "right as recorded": recorded["right"], "right held view": changed["right"], "never as recorded": recorded["never"], "never held view": changed["never"],
                      "early as recorded": recorded["early"], "early held view": changed["early"]})
    for label, is_part in [("moved up by 5 or more (wrist below 13.5 at the end)", last_signed <= -5), ("moved down by 5 or more (wrist above 13.5 at the end)", last_signed >= 5)]:
        part = rows[is_part]
        recorded = outcomes(part, "stop_as_recorded", settings)
        changed = outcomes(part, "stop_held_view", settings)
        table.append({"last picture moved by (px at 160)": label, "episodes": part["episode_id"].nunique(),
                      "right as recorded": recorded["right"], "right held view": changed["right"], "never as recorded": recorded["never"], "never held view": changed["never"],
                      "early as recorded": recorded["early"], "early held view": changed["early"]})
    table = pandas.DataFrame(table)
    print("episodes by how far the content of their last picture moves in the held-wrist view; stop rule 0.5 on two pictures in a row")
    print(table.to_string(index = False))
    table.to_csv(output_folder / "held_wrist_offline_by_shift.csv", index = False)
    print(f"\nsaved {output_folder / 'held_wrist_offline_by_shift.csv'}")



if __name__ == "__main__":
    main()
