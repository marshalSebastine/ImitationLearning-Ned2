"""
Verifier's check 11 (11.11.26): the analyst's driver variant (b), "once the model's own step is under 0.5 cm, keep the
direction of the previous step instead of the direction of the short answer", tried offline on the held-out sheets.

For every held-out row whose asked step is under 0.5 cm (fold models of the E recipe; their answers were saved by the
step-length analyst in ../../step_length/fold_outputs_E_recipe.npz: "exact" = forward cm, left cm, ...), the error of
the model's current direction and of the previous row's direction are measured against the demonstrated move of the
row, both on the table (picture heading plus the angle in the picture). Rows near the end of the line and rows
elsewhere (bends, twists in place) are counted apart. This follows the demonstrators' paths, not the model's own.
"""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent
INDEX_FOLDER = Path("D:/AI_Project/03_data/processed/v01_training_index")


def angle_between(first_deg: numpy.ndarray, second_deg: numpy.ndarray) -> numpy.ndarray:
    return numpy.abs((first_deg - second_deg + 180.0) % 360.0 - 180.0)


frames = pandas.read_csv(INDEX_FOLDER / "frames.csv", low_memory = False).set_index("frame_index")
for recipe in ("E_recipe", "A_recipe"):
    saved = numpy.load(HERE.parents[1] / "step_length" / f"fold_outputs_{recipe}.npz")
    table = frames.loc[saved["frame_index"], ["episode_id", "row_in_episode", "tip_forward_m", "tip_left_m", "picture_heading_deg", "steps_to_end", "action"]].copy()
    table["asked_forward"] = saved["exact"][:, 0]
    table["asked_left"] = saved["exact"][:, 1]
    table["stop_probability"] = saved["stop_probability"]
    table = table.sort_values(["episode_id", "row_in_episode"])
    table["asked_cm"] = numpy.hypot(table["asked_forward"], table["asked_left"])
    table["asked_table_deg"] = table["picture_heading_deg"] + numpy.degrees(numpy.arctan2(table["asked_left"], table["asked_forward"]))
    table["shown_table_deg"] = table["picture_heading_deg"] + numpy.degrees(numpy.arctan2(table["tip_left_m"], table["tip_forward_m"]))
    table["shown_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"], table["tip_left_m"])
    grouped = table.groupby("episode_id", sort = False)
    table["previous_table_deg"] = grouped["asked_table_deg"].shift(1)
    table["previous_row"] = grouped["row_in_episode"].shift(1)
    usable = table[(table["asked_cm"] < 0.5) & (table["shown_cm"] >= 0.3) & (table["previous_row"] == table["row_in_episode"] - 1) & (table["action"] != "stop")].copy()
    usable["error_current"] = angle_between(usable["asked_table_deg"].to_numpy(), usable["shown_table_deg"].to_numpy())
    usable["error_previous"] = angle_between(usable["previous_table_deg"].to_numpy(), usable["shown_table_deg"].to_numpy())
    print(f"{recipe}: held-out rows with an asked step under 0.5 cm and a demonstrated move of 0.3 cm or more: {len(usable)} of {len(table)}")
    for label, part in (("within 3 pictures of the stop", usable[usable["steps_to_end"] <= 3]),
                        ("stop chance 0.2 or more", usable[usable["stop_probability"] >= 0.2]),
                        ("elsewhere (more than 3 pictures before the stop)", usable[usable["steps_to_end"] > 3])):
        if len(part) == 0:
            continue
        print(f"  {label}: {len(part)} rows; the model's current direction is off by a median {part['error_current'].median():.0f} degrees (within 45 degrees in {100 * (part['error_current'] <= 45).mean():.0f} %);"
              f" the previous row's direction by {part['error_previous'].median():.0f} degrees (within 45 degrees in {100 * (part['error_previous'] <= 45).mean():.0f} %);"
              f" the previous one is the better of the two in {100 * (part['error_previous'] < part['error_current']).mean():.0f} %")
