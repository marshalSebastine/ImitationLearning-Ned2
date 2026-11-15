"""
Rule 3 of the plan's step 3 against the forearm's limit: where the recorded pushes do NOT run up the picture, could the
demonstrator have turned the camera along the line, or did the limit of joint 4 forbid it?

Measured from the cleaned training rows and the twist analyst's puck positions (as in m01); nothing is trained, nothing
connects to the robot. Three questions:
    1. Pushing rows whose puck travels more than 45 or 60 degrees off the picture's top: in how many would the camera
       have had to turn beyond the limit to look along the travel ("forced"), and in how many was there room ("chosen")?
    2. The same picture situation with two different recorded answers: where the puck's travel runs more than 30
       degrees off the picture's top and joint 4 still has room, how often did the demonstrator push on, and how often
       turn nearly in place (what rule 3 asks for)?
    3. How much of each sheet's line runs in directions the camera can never look along, whatever the demonstrator does
       (more than "reach_of_the_camera_deg" from "straight away from the base")?

Run from this folder:
    python m02_rule_3_forced_or_chosen.py > m02_rule_3_forced_or_chosen.out.txt
"""

from typing import Dict, Optional

import numpy
import pandas

import m01_style_against_the_rules
import method_common

FORCED_PARAMETERS = {
    # joint 4 has "room" when the angle that would put the puck's travel straight up the picture stays this far inside
    # the driver's limit of 117 degrees
    "room_inside_limit_deg": 10.0,
    # the picture's top can be turned at most about this far from "straight away from the base" (the check: 106 to
    # 123 degrees between 12 and 45 cm reach with joint 4 at 117)
    "reach_of_the_camera_deg": 108.0,
    "push_least_cm": 0.7,
    "in_place_most_cm": 0.3
}



def forced_or_chosen_report(parameters: Optional[Dict] = None) -> None:
    """Print the three measurements named in the module's text."""

    settings = method_common.merged_parameters(m01_style_against_the_rules.STYLE_PARAMETERS)
    settings.update(FORCED_PARAMETERS)
    settings.update(parameters or {})
    frames = m01_style_against_the_rules.puck_travel_in_picture(method_common.cleaned_frames(settings), settings)
    is_step = frames["action"] != "stop"
    frames["length_cm"] = 100.0 * numpy.hypot(frames["tip_forward_m"].fillna(0.0), frames["tip_left_m"].fillna(0.0))
    frames["joint_4_deg"] = numpy.degrees(frames["joint_4"])
    frames["required_joint_4_deg"] = frames["joint_4_deg"] - frames["travel_picture_deg"] / settings["heading_per_twist"]
    limit = settings["joint_4_most_deg"]
    room = limit - settings["room_inside_limit_deg"]
    is_known = frames["travel_picture_deg"].notna() & is_step
    is_push = is_known & (frames["length_cm"] >= settings["push_least_cm"])
    off = frames["travel_picture_deg"].abs()

    print("1. Pushing rows whose puck does not travel up the picture: forced by the limit of joint 4, or chosen?")
    print(f"   (forced: looking along the travel would need |joint 4| over {limit:.0f}; room: it would need at most {room:.0f};"
          f" between the two: at the edge)")
    print("   sheet     pushing rows   off by over 45: rows   forced   at the edge   room   |   off by over 60: rows   forced   at the edge   room")
    for sheet, group in list(frames[is_push].groupby("sheet")) + [("all", frames[is_push])]:
        cells = []
        for least in (45.0, 60.0):
            part = group[group["travel_picture_deg"].abs() > least]
            needed = part["required_joint_4_deg"].abs()
            cells.append((len(part), 100 * (needed > limit).mean() if len(part) else float("nan"),
                          100 * ((needed > room) & (needed <= limit)).mean() if len(part) else float("nan"),
                          100 * (needed <= room).mean() if len(part) else float("nan")))
        print(f"   {sheet:8s}  {len(group):11d}   {cells[0][0]:19d}   {cells[0][1]:5.1f}%   {cells[0][2]:10.1f}%   {cells[0][3]:4.1f}%   |"
              f"   {cells[1][0]:19d}   {cells[1][1]:5.1f}%   {cells[1][2]:10.1f}%   {cells[1][3]:4.1f}%")
    push_rows = frames[is_push]
    forced = push_rows[(push_rows["travel_picture_deg"].abs() > 45.0) & (push_rows["required_joint_4_deg"].abs() > limit)]
    chosen = push_rows[(push_rows["travel_picture_deg"].abs() > 45.0) & (push_rows["required_joint_4_deg"].abs() <= room)]
    print(f"   of all {len(push_rows)} pushing rows: {100 * len(forced) / len(push_rows):.1f} % are pushes more than 45 degrees off the picture's top that the limit"
          f" forces, {100 * len(chosen) / len(push_rows):.1f} % are such pushes with room to turn")
    print(f"   recorded |joint 4| in the forced rows: median {forced['joint_4_deg'].abs().median():.0f} degrees, over 100 in"
          f" {100 * (forced['joint_4_deg'].abs() > 100).mean():.0f} %, under 60 in {100 * (forced['joint_4_deg'].abs() < 60).mean():.0f} %")
    is_same_side = numpy.sign(forced["required_joint_4_deg"]) == numpy.sign(forced["joint_4_deg"])
    print(f"   in the forced rows the forearm stood on the side the travel asks for in {100 * is_same_side.mean():.0f} % and on the other side in"
          f" {100 * (1 - is_same_side.mean()):.0f} % (other side: the camera would have to swing through its whole range to look along the travel)")
    print(f"   recorded |joint 4| in the rows with room: median {chosen['joint_4_deg'].abs().median():.0f} degrees")

    print()
    print("2. One picture situation, two recorded answers: travel more than 30 degrees off the picture's top WITH room to turn")
    with_room = frames[is_known & (off > 30.0) & (frames["required_joint_4_deg"].abs() <= room)]
    pushed = (with_room["length_cm"] >= settings["push_least_cm"]).mean()
    in_place = (with_room["length_cm"] < settings["in_place_most_cm"]).mean()
    print(f"   rows: {len(with_room)} of {int(is_known.sum())} with a known travel direction ({100 * len(with_room) / is_known.sum():.1f} %);"
          f" the demonstrator pushed on ({settings['push_least_cm']} cm or more) in {100 * pushed:.1f} %, moved {settings['in_place_most_cm']} to"
          f" {settings['push_least_cm']} cm in {100 * (1 - pushed - in_place):.1f} %, turned nearly in place (under {settings['in_place_most_cm']} cm) in {100 * in_place:.1f} %")
    towards = (-numpy.sign(with_room["travel_picture_deg"]) * with_room["forearm_change_deg"] >= 2.0).mean()
    away = (numpy.sign(with_room["travel_picture_deg"]) * with_room["forearm_change_deg"] >= 2.0).mean()
    print(f"   in these rows the forearm turned the camera towards the travel direction (2 degrees or more) in {100 * towards:.1f} %, away from it in"
          f" {100 * away:.1f} %, neither in {100 * (1 - towards - away):.1f} %")
    for sheet, group in with_room.groupby("sheet"):
        print(f"      {sheet}: {len(group):5d} rows; pushed on in {100 * (group['length_cm'] >= settings['push_least_cm']).mean():5.1f} %,"
              f" nearly in place in {100 * (group['length_cm'] < settings['in_place_most_cm']).mean():5.1f} %")

    print()
    print("3. How much of the recorded travel runs where the camera can never look along it")
    radial_deg = numpy.degrees(numpy.arctan2(frames["tip_y_m"], frames["tip_x_m"]))
    from_radial = pandas.Series(numpy.abs(method_common.wrapped_degrees(frames["travel_table_deg"] - radial_deg)), index = frames.index)
    reach = settings["reach_of_the_camera_deg"]
    print(f"   (travel direction more than {reach:.0f} degrees from 'straight away from the base'; rows with a known direction)")
    print("   sheet     rows   beyond the camera's reach   episodes with such a stretch (10 rows or more)")
    for sheet, group in list(frames[is_known].groupby("sheet")) + [("all", frames[is_known])]:
        beyond = from_radial[group.index] > reach
        per_episode = beyond.groupby(group["episode_id"]).sum()
        print(f"   {sheet:8s} {len(group):5d}   {100 * beyond.mean():23.1f}%   {int((per_episode >= 10).sum()):4d} of {group['episode_id'].nunique()}")
    known = frames[is_known].assign(is_beyond = from_radial[is_known] > reach)
    by_start = known.groupby(["sheet", "start_end"])["is_beyond"].agg(["size", "mean"])
    print("   by sheet and start end (rows, share beyond the reach):")
    print("   " + "; ".join(f"{sheet[-2:]}{end} {int(row['size'])} rows {100 * row['mean']:.0f} %" for (sheet, end), row in by_start.iterrows()))



if __name__ == "__main__":
    forced_or_chosen_report()
