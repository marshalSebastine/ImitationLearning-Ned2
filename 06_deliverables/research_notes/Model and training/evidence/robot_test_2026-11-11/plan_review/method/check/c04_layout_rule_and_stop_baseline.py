"""
Two checks for the review of the method critic's answer (method_check). Reads tables only; nothing is trained and
nothing connects to the robot.

    A. The critic's rule 0 ("among the placements that start within about 60 degrees of straight away, take the one
       with the fewest centimetres beyond 110 degrees"): what it gives on the orientation analyst's table of
       calculated placements (orientation_coverage/05_candidates.csv; "back_slanted_cm" is the line between 110 and
       140 degrees from straight away, "back_sector_cm" the line beyond 140), against the demonstrations' lay-out rule
       and against the analyst's "gentlest" placement, which the critic quotes.
    B. The stop over whole episodes, per sheet, when a stop-like LAST picture counts as a stop (the recording ends
       there, so "two in a row" can never be met by it), and the direction on the rows just BEFORE a hand correction.

Run from the folder above:
    python check/c04_layout_rule_and_stop_baseline.py > check/c04_layout_rule_and_stop_baseline.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import method_common



def report() -> None:
    candidates = pandas.read_csv(method_common.EVIDENCE_FOLDER / "orientation_coverage" / "05_candidates.csv")
    candidates["beyond_110_cm"] = candidates["back_slanted_cm"] + candidates["back_sector_cm"]
    possible = candidates[candidates["is_possible"]]
    print("A. Line beyond the camera's reach by lay-out rule (calculated placements of the orientation analyst; cm of line)")
    for name, group in (("the six course sheets", possible[possible["shape"].str.startswith("sheet")]),
                        ("the six practice tracks of 10.11", possible[possible["shape"].str.startswith("practice")])):
        print(f"   {name}")
        rows = []
        for (shape, end), part in group.groupby(["shape", "end"]):
            straight = part[(part["family"] == "mark") & (part["start_direction_deg"].abs() < 1.0)]
            within_60 = part[part["start_direction_deg"].abs() <= 60.0]
            gentlest = part.sort_values(["back_sector_cm", "flips"]).iloc[0]
            rule_0 = within_60.sort_values(["beyond_110_cm", "back_sector_cm"]).iloc[0] if len(within_60) else None
            any_start = part.sort_values(["beyond_110_cm", "back_sector_cm"]).iloc[0]
            rows.append({"shape": shape, "end": end, "line_cm": float(part["line_cm"].iloc[0]),
                         "as_recorded_110": float(straight["beyond_110_cm"].iloc[0]) if len(straight) else numpy.nan,
                         "as_recorded_140": float(straight["back_sector_cm"].iloc[0]) if len(straight) else numpy.nan,
                         "gentlest_110": float(gentlest["beyond_110_cm"]), "gentlest_140": float(gentlest["back_sector_cm"]),
                         "rule_0_110": float(rule_0["beyond_110_cm"]) if rule_0 is not None else numpy.nan,
                         "rule_0_140": float(rule_0["back_sector_cm"]) if rule_0 is not None else numpy.nan,
                         "rule_0_start_deg": float(rule_0["start_direction_deg"]) if rule_0 is not None else numpy.nan,
                         "rule_0_family": rule_0["family"] if rule_0 is not None else "",
                         "any_start_110": float(any_start["beyond_110_cm"]), "any_start_deg": float(any_start["start_direction_deg"])})
        table = pandas.DataFrame(rows)
        print(table.to_string(index = False))
        for column, label in (("as_recorded", "line leaves straight away on the start mark (as recorded)"), ("gentlest", "the analyst's gentlest placement (fewest cm beyond 140)"),
                              ("rule_0", "the critic's rule 0 (start within 60 degrees, fewest cm beyond 110)")):
            part = table[table[f"{column}_110"].notna()]
            print(f"   {label}: {len(part)} of {len(table)} shape ends possible; beyond 110 degrees {part[f'{column}_110'].sum():.0f} of {part['line_cm'].sum():.0f} cm"
                  f" ({100 * part[f'{column}_110'].sum() / part['line_cm'].sum():.1f} %); beyond 140 degrees {part[f'{column}_140'].sum():.0f} cm ({100 * part[f'{column}_140'].sum() / part['line_cm'].sum():.1f} %)")
        print(f"   any start direction, fewest cm beyond 110: {table['any_start_110'].sum():.0f} of {table['line_cm'].sum():.0f} cm ({100 * table['any_start_110'].sum() / table['line_cm'].sum():.1f} %)")
        print()

    print("B. The stop per sheet, recipe E on sheets never seen: the project's rule (0.5, two in a row) and the same with a stop-like last picture counted")
    from training_metrics import stop_metrics

    frames = method_common.cleaned_frames()
    data = numpy.load(method_common.METHOD_REVIEW_PARAMETERS["fold_outputs"]["E"])
    answers = pandas.DataFrame({"frame_index": data["frame_index"], "stop_chance": data["stop_probability"], "asked_forward": data["exact"][:, 0], "asked_left": data["exact"][:, 1]})
    table = frames.merge(answers, on = "frame_index", how = "left")
    totals = numpy.zeros(8, dtype = int)
    for sheet, group in table.groupby("sheet"):
        episodes = group["episode_id"].nunique()
        result = stop_metrics(numpy.zeros(len(group)), (group["stop_chance"].to_numpy() >= 0.5).astype(float), group["goal_distance_cm"].to_numpy(), group["episode_id"].to_numpy(),
                              threshold = 0.5, consecutive_pictures = 2)
        right, early, never = (round(result[key] * episodes) for key in ("episodes_stopped_right", "episodes_stopped_early", "episodes_never_stopped"))
        last_only = 0
        for episode_id, episode in group.groupby("episode_id", sort = False):
            passing = episode["stop_chance"].to_numpy() >= 0.5
            if not (passing[1:] & passing[:-1]).any() and passing[-1]:
                last_only += 1
        counts = numpy.array([right, early, episodes - right - early - never, never, right + last_only, early, episodes - right - early - never, never - last_only])
        totals += counts
        print(f"   {sheet[-2:]} ({episodes:2d} episodes): right / early / 2 to 3 cm before / never  {counts[0]:3d} / {counts[1]:2d} / {counts[2]:2d} / {counts[3]:2d}   |   last picture counted  {counts[4]:3d} / {counts[5]:2d} / {counts[6]:2d} / {counts[7]:2d}")
    print(f"   all (209 episodes):                                     {totals[0]:3d} / {totals[1]:2d} / {totals[2]:2d} / {totals[3]:2d}   |   last picture counted  {totals[4]:3d} / {totals[5]:2d} / {totals[6]:2d} / {totals[7]:2d}")

    print()
    print("   direction on the rows just BEFORE a hand correction (the replay had gone wrong there; their labels still say what the first guiding did)")
    grouped = table.groupby("episode_id", sort = False)
    previous_waypoint = grouped["waypoint"].shift(1)
    is_restart = (table["waypoint"] <= previous_waypoint) & previous_waypoint.notna()
    is_before = numpy.zeros(len(table), dtype = bool)
    positions = numpy.flatnonzero(is_restart.to_numpy())
    episode_ids = table["episode_id"].to_numpy()
    for position in positions:
        for back in range(1, 6):
            if position - back >= 0 and episode_ids[position - back] == episode_ids[position]:
                is_before[position - back] = True
    table["shown_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"].fillna(0.0), table["tip_left_m"].fillna(0.0))
    shown_angle = numpy.degrees(numpy.arctan2(table["tip_left_m"].fillna(0.0), table["tip_forward_m"].fillna(0.0)))
    asked_angle = numpy.degrees(numpy.arctan2(table["asked_left"], table["asked_forward"]))
    table["angle_error"] = numpy.abs(method_common.wrapped_degrees(asked_angle - shown_angle))
    moving = (table["action"] != "stop") & (table["shown_cm"] >= 0.3)
    part = table[moving & is_before]
    print(f"   the 5 rows before a hand correction: {len(part)} moving rows ({100 * len(part) / moving.sum():.1f} % of all moving rows); direction within 45 degrees of the label {100 * (part['angle_error'] <= 45).mean():.1f} %;"
          f" all moving rows {100 * (table[moving]['angle_error'] <= 45).mean():.1f} %")



if __name__ == "__main__":
    report()
