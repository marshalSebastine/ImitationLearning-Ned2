"""
Verifier's check 14 (11.11.26): what the analyst's retraining rule (2) would remove. The rule: "drop or down-weight
pictures where the middle of the line is not under the puck (over 12 mm) and the offset is not shrinking over the next
steps". Counted on the analyst's own table of sampled demonstrations (../demonstration_centring_0N.csv): how many
pictures the rule takes, what the puck is doing in them, and how many of them are the first part of a correction that
does bring the puck back (offset under 7 mm within the next 15 pictures).
"""

import glob
from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent

table = pandas.concat([pandas.read_csv(path) for path in sorted(glob.glob(str(HERE.parent / "demonstration_centring_0*.csv")))], ignore_index = True)
table = table.sort_values(["episode_id", "step"]).reset_index(drop = True)
rows = []
for episode_id, run in table.groupby("episode_id", sort = False):
    sizes = run["offset_mm"].abs().to_numpy()
    puck = run[["puck_x", "puck_y"]].to_numpy(dtype = float)
    tips = run[["tip_x", "tip_y"]].to_numpy(dtype = float)
    for position in range(len(run)):
        if not (numpy.isfinite(sizes[position]) and sizes[position] > 12.0):
            continue
        later = sizes[position + 5] if position + 5 < len(run) else numpy.nan
        is_shrinking = bool(numpy.isfinite(later) and later - sizes[position] <= -2.0)
        ahead = sizes[position + 1:position + 16]
        is_brought_back = bool(numpy.any(ahead[numpy.isfinite(ahead)] <= 7.0))
        puck_move = 1000.0 * float(numpy.linalg.norm(puck[position + 1] - puck[position])) if position + 1 < len(run) else numpy.nan
        tip_move = 1000.0 * float(numpy.linalg.norm(tips[position + 1] - tips[position])) if position + 1 < len(run) else numpy.nan
        rows.append({"episode_id": episode_id, "sheet": run["sheet"].iloc[0], "step": int(run["step"].iloc[position]), "offset_mm": sizes[position],
                     "is_shrinking": is_shrinking, "is_brought_back": is_brought_back, "puck_move_mm": puck_move, "tip_move_mm": tip_move})
far = pandas.DataFrame(rows)
measured = int(table["offset_mm"].notna().sum())
taken = far[~far["is_shrinking"]]
print(f"sampled pictures with an offset: {measured}; over 12 mm: {len(far)} ({100 * len(far) / measured:.0f} %)")
print(f"over 12 mm and NOT smaller by 2 mm or more five pictures later (what rule 2 takes): {len(taken)} ({100 * len(taken) / measured:.0f} % of all measured pictures)")
print(f"  of these, the puck is brought back within 7 mm during the next 15 pictures in {int(taken['is_brought_back'].sum())} ({100 * taken['is_brought_back'].mean():.0f} %): the first part of a correction that works")
is_standing = (taken["puck_move_mm"] < 2.0) & (taken["tip_move_mm"] >= 4.0)
print(f"  of these, the puck stands (under 2 mm) while the tip moves 4 mm or more in {int(is_standing.sum())} ({100 * is_standing.mean():.0f} %): the tip going round the puck")
print(f"  of these, the puck is pushed on (4 mm or more) in {int((taken['puck_move_mm'] >= 4.0).sum())} ({100 * (taken['puck_move_mm'] >= 4.0).mean():.0f} %)")
never = taken[~taken["is_brought_back"]]
print(f"  never brought back within 15 pictures: {len(never)} pictures ({100 * len(never) / measured:.0f} % of all measured); by sheet: {never.groupby('sheet').size().to_dict()}")
print("by sheet, pictures the rule takes:", taken.groupby("sheet").size().to_dict(), "of measured", table[table["offset_mm"].notna()].groupby("sheet").size().to_dict())
