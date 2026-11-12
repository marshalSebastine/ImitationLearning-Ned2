"""
The first bend of sheet 03 in one number: how many pictures (decisions) and how much tip travel it took to turn the
forearm by a given angle, in the demonstrations and in the three robot runs that failed there.

Also the first ten decisions of every run: the step length and the twist the model asked for, which separate the runs
that began with a re-aiming manoeuvre (asked length about 0.4 to 0.5 cm) from those that began by pushing on (about 1 cm).

Output: 13_same_turn_how_much_travel.out.txt
Run from this folder: D:/AI_Project/.venv/Scripts/python.exe 13_same_turn_how_much_travel.py
"""

from typing import Dict, Optional

import pandas

from common import ANALYSIS_PARAMETERS, Report, demonstration_frames, robot_run_steps

SAME_TURN_PARAMETERS = {
    "sheet": "path_03",
    "start_kind": "STD",
    "start_end": "A",
    "runs": [5, 9, 10],
    "joint_4_reached_deg": [-50.0, -70.0],
    "first_decisions": 10
}



def main(parameters: Optional[Dict] = None) -> None:
    """Print the pictures and the tip travel needed to reach a forearm angle at the first bend of sheet 03, and the first decisions of every run."""

    settings = dict(SAME_TURN_PARAMETERS)
    settings.update(parameters or {})
    report = Report("13_same_turn_how_much_travel.out.txt")
    frames = demonstration_frames()
    reference = frames[(frames["sheet"] == settings["sheet"]) & (frames["start_kind"] == settings["start_kind"]) & (frames["start_end"] == settings["start_end"])]

    report.say("THE SAME FOREARM TURN, HOW MUCH TIP TRAVEL? The first bend of sheet 03 (measured: frames.csv with the audit cleaning, steps.csv of runs 5, 9, 10)")
    report.say("")
    for angle in settings["joint_4_reached_deg"]:
        rows = []
        for episode_id, episode in reference.groupby("episode_id"):
            episode = episode.sort_values("row_in_episode")
            reached = episode[episode["joint_4_deg"] <= angle]
            if len(reached):
                before = episode[episode["row_in_episode"] < reached.iloc[0]["row_in_episode"]]
                rows.append({"pictures": len(before), "travel_cm": before["step_cm"].sum()})
        rows = pandas.DataFrame(rows)
        report.say(f"joint 4 reaches {angle:.0f} degrees")
        report.say(f"   demonstrations (standard starts from end A; {len(rows)} of {reference['episode_id'].nunique()} reach it): after a median of {rows['pictures'].median():.0f} pictures "
                   f"(range {rows['pictures'].min()} to {rows['pictures'].max()}) and {rows['travel_cm'].median():.1f} cm of tip travel (range {rows['travel_cm'].min():.1f} to {rows['travel_cm'].max():.1f})")
        for run in ANALYSIS_PARAMETERS["robot_runs"]:
            if run["number"] not in settings["runs"]:
                continue
            steps = robot_run_steps(run)
            reached = steps[steps["joint_4_deg"] <= angle]
            if len(reached) == 0:
                report.say(f"   run {run['number']} (model {run['model']}): never (lowest joint 4 {steps['joint_4_deg'].min():.0f} degrees)")
                continue
            picture = int(reached.iloc[0]["step"])
            report.say(f"   run {run['number']} (model {run['model']}): at picture {picture}, after {steps['moved_cm'].iloc[:picture].sum():.1f} cm of tip travel; "
                       f"the lengths the model asked for up to there add up to {steps['asked_cm'].iloc[:picture].sum():.1f} cm")
        report.say("")

    report.say(f"The first {settings['first_decisions']} decisions of every run: what the model asked for")
    for run in ANALYSIS_PARAMETERS["robot_runs"]:
        first = robot_run_steps(run).iloc[:settings["first_decisions"]]
        report.say(f"   run {run['number']:2d} ({run['model']}, {run['sheet']}): mean asked length {first['asked_cm'].mean():.2f} cm, mean |asked twist| {first['model_twist_deg'].abs().mean():.1f} degrees, "
                   f"net asked twist {first['model_twist_deg'].sum():+.1f} degrees; {run['outcome']}")
    report.save()



if __name__ == "__main__":
    main()
