"""
Run 3: sheet 02 lay straight (square to the robot), not as in plan 02-01. Was that set-up in the demonstrations, and
what did the demonstrations do there?

Sheet 02 was recorded in several lay-outs: STD (the line's first centimetres point straight away from the robot),
TL/TR (turned 10 or 15 degrees), offsets, and SQP / SQL (the sheet square to the robot, portrait or landscape, so the
first push meets the line at its own angle). This script:
    - lists the kinds of start of sheet 02 with their episodes (ends as corrected by the audit, as model E saw them);
    - places the first pictures of run 3 (and of run 4, the same sheet laid as in plan 02-01, as a control) against the
      first pictures of every sheet-02 episode, in model E's descriptors and in the descriptors of the fold model that
      never saw sheet 02, and says which kind of start is nearest;
    - draws run 3's first picture beside the first pictures of one episode of each kind;
    - sums up what the demonstrators did at the start of the square episodes (how far the forearm twisted while the
      tip travelled its first centimetres) against what the robot did in run 3.

Nothing connects to the robot. Run from this folder, after extract_features.py model_e and fold without_path_02:
    D:/AI_Project/.venv/Scripts/python.exe sheet02_square_check.py
Output: sheet02_square_check.out.txt, figures/sheet02_square_starts.png
"""

import math
from typing import Dict, Optional

import cv2
import numpy
import pandas

from common import (EVIDENCE_FOLDER, VIEW_NOVELTY_PARAMETERS, OutputLog, load_descriptors, run_picture_paths, run_steps, save_montage, trained_frames, unit_rows)

SQUARE_CHECK_PARAMETERS = {
    "sheet": "path_02",
    "runs": [3, 4],
    "robot_steps": [0, 1, 2],
    "demonstration_rows": [1, 5],
    "travel_cm": [3.0, 5.0],
    "shown_plans": ["02-16", "02-17", "02-18", "02-36", "02-41", "02-02", "02-01"],
    "tile_size": [320, 240]
}



def first_travel_summary(
        rows: pandas.DataFrame,
        travel_cm: float
) -> Dict[str, float]:
    """
    Sum up the start of one demonstration or run: the forearm twist made while the tip travelled its first centimetres.

    Parameters
    ----------
    rows : pandas.DataFrame
        The episode's rows in order, with "length_cm" (the tip's step) and "twist_deg" (the forearm's change).
    travel_cm : float
        The path length that counts as the start.

    Returns
    -------
    dict
        "steps" (rows needed to travel that far), "twist_deg" (the summed twist over them) and "twist_per_cm".
    """

    travelled = rows["length_cm"].cumsum().to_numpy()
    last = int(numpy.searchsorted(travelled, travel_cm)) + 1
    part = rows.iloc[:last]
    return {"steps": len(part), "twist_deg": float(part["twist_deg"].sum()), "twist_per_cm": float(part["twist_deg"].sum() / max(part["length_cm"].sum(), 1e-9))}



def main(parameters: Optional[Dict] = None) -> None:

    settings = dict(SQUARE_CHECK_PARAMETERS)
    settings.update(parameters or {})
    log = OutputLog(EVIDENCE_FOLDER / "sheet02_square_check.out.txt")
    trained = trained_frames(VIEW_NOVELTY_PARAMETERS["model_e_run_name"])
    sheet = trained[trained["sheet"] == settings["sheet"]].copy()
    sheet["length_cm"] = 100 * numpy.hypot(sheet["tip_forward_m"].fillna(0.0), sheet["tip_left_m"].fillna(0.0))
    sheet["twist_deg"] = sheet["forearm_change_deg"].fillna(0.0)

    episodes = sheet.groupby("episode_id").agg(plan = ("plan_number", "first"), kind = ("start_kind", "first"), end = ("start_end", "first"), rows = ("row_in_episode", "size"))
    log.say("SHEET 02 IN THE TRAINING DATA: kinds of start (the start end as corrected by the audit, as model E was trained)")
    log.say("kind    end   episodes   plans")
    for (kind, end), group in episodes.groupby(["kind", "end"]):
        log.say(f"{kind:6s}  {end:3s}  {len(group):9d}   {' '.join(sorted(group['plan'].astype(str)))}")

    first_row, last_row = settings["demonstration_rows"]
    starts = sheet[sheet["row_in_episode"].between(first_row, last_row)]
    spaces = {"model E (trained on sheet 02)": ("model_e_training", "model_e_robot"),
              "fold model without sheet 02 (never saw it)": ("fold_without_path_02_all", "fold_without_path_02_robot")}
    for space_label, (training_name, robot_name) in spaces.items():
        training = load_descriptors(training_name)["grid"]
        robot = load_descriptors(robot_name)
        start_unit = unit_rows(training[starts["frame_index"].to_numpy()])
        log.say("")
        log.say(f"THE FIRST PICTURES AGAINST THE FIRST PICTURES OF EVERY SHEET-02 EPISODE (rows {first_row} to {last_row}), descriptors of {space_label}")
        log.say("(per kind of start and end: the cosine distance from the robot's pictures of steps 0 to 2 to the episode's first pictures, the nearest episode and the median over the episodes)")
        for run_number in settings["runs"]:
            is_robot = (robot["run"] == run_number) & numpy.isin(robot["step"], settings["robot_steps"])
            similarity = unit_rows(robot["grid"][is_robot]) @ start_unit.T
            # per episode: the nearest of its first pictures, averaged over the robot's first pictures
            per_episode = pandas.DataFrame({"episode_id": starts["episode_id"].to_numpy(), "distance": 0.0})
            nearest = {}
            for episode_id, positions in starts.reset_index(drop = True).groupby("episode_id").groups.items():
                nearest[episode_id] = float((1.0 - similarity[:, list(positions)].max(axis = 1)).mean())
            table = episodes.assign(distance = pandas.Series(nearest))
            log.say(f"  run {run_number}:")
            summary = table.groupby(["kind", "end"])["distance"].agg(["min", "median", "size"]).sort_values("min")
            for (kind, end), row in summary.iterrows():
                best_plan = table[(table["kind"] == kind) & (table["end"] == end)].sort_values("distance")["plan"].iloc[0]
                log.say(f"    {kind:6s} end {end}: nearest episode {row['min']:.3f} (plan {best_plan}), median {row['median']:.3f} over {int(row['size'])} episodes")

    log.say("")
    log.say("WHAT WAS DONE AT THE START: the forearm twist made while the tip travelled its first centimetres")
    low_cm, high_cm = settings["travel_cm"]
    log.say(f"kind    end  plan    rows to travel {low_cm:.0f} cm   twist over them, deg   twist per cm | rows to travel {high_cm:.0f} cm   twist deg   twist per cm")
    for (kind, end), group in episodes.sort_values("plan").groupby(["kind", "end"]):
        if kind not in ("SQP", "SQL", "STD"):
            continue
        for episode_id, episode in group.iterrows():
            rows = sheet[sheet["episode_id"] == episode_id].sort_values("row_in_episode")
            near, far = first_travel_summary(rows, low_cm), first_travel_summary(rows, high_cm)
            log.say(f"{kind:6s}  {end:3s}  {episode['plan']:6s}  {near['steps']:19d}   {near['twist_deg']:+20.1f}   {near['twist_per_cm']:+12.1f} | {far['steps']:19d}   {far['twist_deg']:+9.1f}   {far['twist_per_cm']:+12.1f}")
    for run_number in settings["runs"]:
        logged = run_steps(run_number)
        logged = logged[logged["action"] == "model_step"]
        # the robot moved a fixed 1 cm per step and twisted by "twist_used_deg"
        rows = pandas.DataFrame({"length_cm": 1.0, "twist_deg": logged["twist_used_deg"].to_numpy()})
        near, far = first_travel_summary(rows, low_cm), first_travel_summary(rows, high_cm)
        log.say(f"ROBOT   run {run_number}        {near['steps']:19d}   {near['twist_deg']:+20.1f}   {near['twist_per_cm']:+12.1f} | {far['steps']:19d}   {far['twist_deg']:+9.1f}   {far['twist_per_cm']:+12.1f}")
        asked = numpy.hypot(logged["model_tip_forward_cm"], logged["model_tip_left_cm"]).to_numpy()
        log.say(f"        run {run_number}: the step length model E ASKED for in its first 5 steps: {'  '.join(f'{value:.2f}' for value in asked[:5])} cm (done: 1.00 cm each)")

    width, height = settings["tile_size"]
    tiles = []
    for run_number in settings["runs"]:
        picture = cv2.resize(cv2.imread(str(run_picture_paths(run_number)[0])), (width, height), interpolation = cv2.INTER_AREA)
        cv2.rectangle(picture, (0, 0), (width, 18), (0, 0, 0), -1)
        cv2.putText(picture, f"ROBOT run {run_number}, step 0", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(picture)
    for plan in settings["shown_plans"]:
        rows = sheet[sheet["plan_number"] == plan].sort_values("row_in_episode")
        if rows.empty:
            continue
        row = rows.iloc[0]
        picture = cv2.resize(cv2.imread(str(VIEW_NOVELTY_PARAMETERS["raw_folder"] / row["frame_path"])), (width, height), interpolation = cv2.INTER_AREA)
        cv2.rectangle(picture, (0, 0), (width, 18), (0, 0, 0), -1)
        cv2.putText(picture, f"demo plan {plan} {row['start_kind']} end {row['start_end']}, row {int(row['row_in_episode'])}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(picture)
    columns = 3
    while len(tiles) % columns:
        tiles.append(numpy.full((height, width, 3), 255, dtype = numpy.uint8))
    figure_folder = EVIDENCE_FOLDER / "figures"
    figure_folder.mkdir(exist_ok = True)
    save_montage(figure_folder / "sheet02_square_starts.png", numpy.vstack([numpy.hstack(tiles[start:start + columns]) for start in range(0, len(tiles), columns)]))
    log.say("")
    log.say("written: figures/sheet02_square_starts.png")
    log.save()



if __name__ == "__main__":
    main()
