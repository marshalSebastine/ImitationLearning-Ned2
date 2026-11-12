"""
Plan review (key: exam_and_schedule): does the plan's stall rule (step 1, item 2: the last 15 pictures within 4 cm and
under 15 degrees of forearm turn) stay quiet when the robot moves the length the model ASKS for?

The plan says the rule "fires in none of the 210 recordings". The recordings move about 1 cm per picture. The plan's
candidate step rules move 0.61 to 0.67 cm, so the same check is repeated here at the asked pace.

A proxy on paper, not a robot run: for every held-out demonstration (each sheet answered by the model of recipe E that
never saw it; file step_length/fold_outputs_E_recipe.npz, written by the step-length analyst) the asked steps are laid
one after the other on the table (each turned by that picture's own heading), and the asked twists are added up. The
pictures are the demonstrator's, so this is "teacher-forced": a real run would see other pictures.

Reads only. Run:  D:/AI_Project/.venv/Scripts/python.exe stall_rule_at_asked_pace.py > stall_rule_at_asked_pace.out.txt
"""

from pathlib import Path

import numpy
import pandas

INDEX_FILE = Path("D:/AI_Project/03_data/processed/v01_training_index/frames.csv")
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "step_length" / "fold_outputs_E_recipe.npz"


def windows_that_fire(points_cm: numpy.ndarray, forearm_deg: numpy.ndarray, pictures: int, extent_cm: float, twist_deg: float) -> numpy.ndarray:
    """Index of every window end where the last ``pictures`` tips lie in a box under ``extent_cm`` and the forearm range is under ``twist_deg``."""

    fired = []
    for end in range(pictures, len(points_cm) + 1):
        recent = points_cm[end - pictures:end]
        extent = numpy.hypot(recent[:, 0].max() - recent[:, 0].min(), recent[:, 1].max() - recent[:, 1].min())
        turned = forearm_deg[end - pictures:end].max() - forearm_deg[end - pictures:end].min()
        if extent < extent_cm and turned < twist_deg:
            fired.append(end - 1)
    return numpy.array(fired, dtype = int)


def main() -> None:
    outputs = numpy.load(OUTPUT_FILE)
    predicted = pandas.DataFrame({"frame_index": outputs["frame_index"], "asked_forward_cm": outputs["exact"][:, 0],
                                  "asked_left_cm": outputs["exact"][:, 1], "asked_twist_deg": outputs["exact"][:, 2],
                                  "stop_probability": outputs["stop_probability"]})
    frames = pandas.read_csv(INDEX_FILE, usecols = ["frame_index", "episode_id", "row_in_episode", "sheet", "action", "picture_heading_deg",
                                                    "tip_forward_m", "tip_left_m", "forearm_change_deg", "goal_distance_cm"])
    rows = frames.merge(predicted, on = "frame_index", how = "inner").sort_values(["episode_id", "row_in_episode"])
    rows = rows[rows["action"] != "stop"]
    print(f"held-out rows of recipe E without the stop rows: {len(rows)}; episodes: {rows['episode_id'].nunique()}")
    asked_length = numpy.hypot(rows["asked_forward_cm"], rows["asked_left_cm"])
    print(f"asked length, all rows: median {asked_length.median():.2f} cm; in the last 5 cm before the goal: median {asked_length[rows['goal_distance_cm'] < 5].median():.2f} cm, "
          f"share under 0.3 cm {100 * (asked_length[rows['goal_distance_cm'] < 5] < 0.3).mean():.0f} %")

    paces = {"demonstrated steps (the plan's own check)": None, "asked length, gain 1.0, at most 1.0 cm": "asked", "fixed 0.65 cm in the asked direction": 0.65,
             "fixed 1.0 cm in the asked direction": 1.0}
    rules = {"plan: 15 pictures, 4 cm, 15 degrees": (15, 4.0, 15.0), "driver today: 30 pictures, 3 cm, 15 degrees": (30, 3.0, 15.0)}
    for rule_name, (pictures, extent_cm, twist_deg) in rules.items():
        print(f"\n=== stall rule '{rule_name}' ===")
        for pace_name, pace in paces.items():
            fired_by_sheet, episodes_by_sheet, near_end, far_from_end = {}, {}, 0, 0
            for episode_id, episode in rows.groupby("episode_id", sort = False):
                heading = numpy.radians(episode["picture_heading_deg"].to_numpy())
                if pace is None:
                    forward, left = 100 * episode["tip_forward_m"].to_numpy(), 100 * episode["tip_left_m"].to_numpy()
                    twist = episode["forearm_change_deg"].to_numpy()
                else:
                    forward, left = episode["asked_forward_cm"].to_numpy(), episode["asked_left_cm"].to_numpy()
                    twist = numpy.clip(episode["asked_twist_deg"].to_numpy(), -10.0, 10.0)
                    length = numpy.hypot(forward, left)
                    if pace == "asked":
                        scale = numpy.minimum(length, 1.0) / numpy.maximum(length, 1e-9)
                    else:
                        scale = pace / numpy.maximum(length, 1e-9)
                    forward, left = forward * scale, left * scale
                step_x = forward * numpy.cos(heading) - left * numpy.sin(heading)
                step_y = forward * numpy.sin(heading) + left * numpy.cos(heading)
                # the tip at each picture is the sum of the steps made before it; the forearm likewise
                points = numpy.column_stack([numpy.concatenate([[0.0], numpy.cumsum(step_x)[:-1]]), numpy.concatenate([[0.0], numpy.cumsum(step_y)[:-1]])])
                forearm = numpy.concatenate([[0.0], numpy.cumsum(twist)[:-1]])
                fired = windows_that_fire(points, forearm, pictures, extent_cm, twist_deg)
                sheet = episode["sheet"].iloc[0]
                episodes_by_sheet[sheet] = episodes_by_sheet.get(sheet, 0) + 1
                if len(fired):
                    fired_by_sheet[sheet] = fired_by_sheet.get(sheet, 0) + 1
                    goal_distance = episode["goal_distance_cm"].to_numpy()[fired[0]]
                    if goal_distance < 5.0:
                        near_end += 1
                    else:
                        far_from_end += 1
            total_fired, total = sum(fired_by_sheet.values()), sum(episodes_by_sheet.values())
            per_sheet = ", ".join(f"{sheet[-2:] if sheet[-2:].isdigit() else sheet}: {fired_by_sheet.get(sheet, 0)}/{count}" for sheet, count in sorted(episodes_by_sheet.items()))
            print(f"  {pace_name}: fires in {total_fired} of {total} episodes ({100 * total_fired / total:.0f} %); first firing within 5 cm of the goal {near_end}, further away {far_from_end}; per sheet {per_sheet}")


if __name__ == "__main__":
    main()
