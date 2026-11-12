"""
Verifier's check 7 (11.11.26): the analyst's own table of demonstration offsets (../demonstration_centring_0N.csv),
recounted and split by what the puck is doing: being pushed (the fitted puck moves 4 mm or more to the next picture)
or standing still while the magnet tip moves round it (a correction of the puck with the rod's one contact point).
Also: the same shares with the analyst's one-picture offset instead of the 12-picture one, and the robot runs split
the same way. Reads the analyst's tables only.
"""

import glob
from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent


def shares(sizes: pandas.Series) -> str:
    sizes = sizes.dropna().abs()
    if len(sizes) == 0:
        return "no pictures"
    return (f"{len(sizes):5d} pictures: median {sizes.median():4.1f} mm, within 7 mm {100 * (sizes <= 7).mean():3.0f} %, over 12 mm {100 * (sizes > 12).mean():3.0f} %,"
            f" over 17 mm {100 * (sizes > 17).mean():3.0f} %")


def with_puck_moves(table: pandas.DataFrame, group_column: str) -> pandas.DataFrame:
    table = table.sort_values([group_column, "step"]).copy()
    grouped = table.groupby(group_column, sort = False)
    table["puck_move_mm"] = 1000.0 * numpy.hypot(grouped["puck_x"].shift(-1) - table["puck_x"], grouped["puck_y"].shift(-1) - table["puck_y"])
    table["tip_move_mm"] = 1000.0 * numpy.hypot(grouped["tip_x"].shift(-1) - table["tip_x"], grouped["tip_y"].shift(-1) - table["tip_y"])
    return table


demonstrations = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(HERE.parent / "demonstration_centring_0*.csv")))], ignore_index = True)
demonstrations = with_puck_moves(demonstrations, "episode_id")
measured = demonstrations[demonstrations["offset_mm"].notna()]
print("THE ANALYST'S TABLE OF DEMONSTRATION OFFSETS, recounted")
print("all sampled pictures with an offset:                 ", shares(measured["offset_mm"]))
print("episodes:", measured["episode_id"].nunique(), "; by start kind:", measured.groupby("start_kind")["episode_id"].nunique().to_dict())
print("standard starts after the first 15 pictures:         ", shares(measured[(measured["start_kind"] == "STD") & (measured["step"] > 15)]["offset_mm"]))
print()
is_pushed = measured["puck_move_mm"] >= 4.0
is_standing = (measured["puck_move_mm"] < 2.0) & (measured["tip_move_mm"] >= 4.0)
print("split by what the puck does until the next picture (the analyst's fitted puck places):")
print("  the puck is pushed 4 mm or more:                   ", shares(measured[is_pushed]["offset_mm"]))
print("  the puck stands (under 2 mm) while the tip moves:  ", shares(measured[is_standing]["offset_mm"]))
print("  the rest (small moves, twists in place, last rows):", shares(measured[~is_pushed & ~is_standing]["offset_mm"]))
far = measured[measured["offset_mm"].abs() > 12.0]
print(f"of the {len(far)} pictures over 12 mm: pushed {int((far['puck_move_mm'] >= 4).sum())}, standing while the tip moves {int(((far['puck_move_mm'] < 2) & (far['tip_move_mm'] >= 4)).sum())},"
      f" rest {int(len(far) - (far['puck_move_mm'] >= 4).sum() - ((far['puck_move_mm'] < 2) & (far['tip_move_mm'] >= 4)).sum())}")
print()
print("without sheet 06 (dashed: the measure has no line in the gaps):")
solid = measured[measured["sheet"] != "path_06"]
print("  all pictures:                                      ", shares(solid["offset_mm"]))
print("  pushed 4 mm or more:                               ", shares(solid[solid["puck_move_mm"] >= 4.0]["offset_mm"]))
print("  pushed, standard starts only:                      ", shares(solid[(solid["puck_move_mm"] >= 4.0) & (solid["start_kind"] == "STD")]["offset_mm"]))
print("  pushed, puck put aside at the start, after picture 15:", shares(solid[(solid["puck_move_mm"] >= 4.0) & solid["start_kind"].isin(["OL1", "OL1.5", "OR1", "OR1.5"]) & (solid["step"] > 15)]["offset_mm"]))
print("per sheet, pushed pictures:")
for sheet, part in measured[is_pushed].groupby("sheet"):
    print(f"  {sheet}:", shares(part["offset_mm"]))
print()
print("the same pictures by the analyst's ONE-PICTURE offset (clean and aligned ones only):")
single = demonstrations[demonstrations["single_offset_mm"].notna() & (demonstrations["is_single_clean"] == True) & (demonstrations["is_single_aligned"] == True)]
print("  all with a clean one-picture offset:               ", shares(single["single_offset_mm"]))
both = single[single["offset_mm"].notna()]
difference = (both["single_offset_mm"].abs() - both["offset_mm"].abs())
print(f"  both measures on {len(both)} pictures: 12-picture {shares(both['offset_mm'])}")
print(f"                                   one-picture {shares(both['single_offset_mm'])}")
print(f"  size of one-picture minus 12-picture: median {difference.median():+.1f} mm, median size {difference.abs().median():.1f} mm, over 5 mm apart in {int((difference.abs() > 5).sum())}")
print()
print("the episodes' own shares (12-picture offset, pushed pictures): how many of the sampled episodes keep the puck within 7 mm in 80 % or more of their pushed pictures")
per_episode = measured[is_pushed].groupby("episode_id")["offset_mm"].apply(lambda values: float((values.abs() <= 7).mean()))
print(f"  {int((per_episode >= 0.8).sum())} of {len(per_episode)} episodes; median share {100 * per_episode.median():.0f} %, range {100 * per_episode.min():.0f} to {100 * per_episode.max():.0f} %")

print()
print("THE ROBOT'S RUNS (the analyst's ../robot_runs_steps.csv), the same split")
robot = with_puck_moves(pandas.read_csv(HERE.parent / "robot_runs_steps.csv"), "run")
for run in (1, 2, 4, 7):
    part = robot[(robot["run"] == run) & robot["offset_mm"].notna()]
    print(f"  run {run}, all pictures:", shares(part["offset_mm"]), "; pushed 4 mm or more:", shares(part[part["puck_move_mm"] >= 4.0]["offset_mm"]))
