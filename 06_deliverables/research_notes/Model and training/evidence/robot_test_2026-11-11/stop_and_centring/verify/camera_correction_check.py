"""
Verifier's check 8 (11.11.26): does the analyst's picture shift and roll make single views agree better on where the
line's end lies? And do the two camera periods really need different corrections?

Per demonstration: the range, over my reference pictures 2 to 17 steps before the stop, of the end's place against the
puck at the stop (verify_demonstration_ends_<camera>_01_06.csv; run verify_demonstration_ends.py with "fitted",
"corrected" and "swapped" first). A camera model that carries table points correctly from view to view gives a small
range. Reads my own tables only.
"""

from pathlib import Path

import pandas

HERE = Path(__file__).resolve().parent
NAMES = ("fitted", "corrected", "swapped")

tables = {name: pandas.read_csv(HERE / f"verify_demonstration_ends_{name}_01_06.csv") for name in NAMES}
both = tables["fitted"][["episode_id", "sheet", "period"]].copy()
for name in NAMES:
    table = tables[name].set_index("episode_id")
    both[f"references_{name}"] = both["episode_id"].map(table["references"])
    both[f"aside_range_{name}"] = both["episode_id"].map(table["aside_high_mm"] - table["aside_low_mm"])
    both[f"short_range_{name}"] = both["episode_id"].map(table["short_high_mm"] - table["short_low_mm"])
is_kept = (both[[f"references_{name}" for name in NAMES]] >= 4).all(axis = 1)
# leave out the episodes in which my end finder jumped to another piece of line (a range over 25 mm)
is_kept &= (both[[f"{kind}_range_{name}" for kind in ("aside", "short") for name in NAMES]] <= 25.0).all(axis = 1)
both = both[is_kept]
print("the range of the end's place over single reference views, per demonstration (smaller is better)")
for period, part in list(both.groupby("period")) + [("both periods", both)]:
    print(f"{period}: {len(part)} demonstrations with 4 or more reference pictures in every variant")
    for kind in ("aside", "short"):
        fitted, corrected, swapped = (part[f"{kind}_range_{name}"] for name in NAMES)
        print(f"  '{kind}': camera as fitted median {fitted.median():.1f} mm; the period's own correction {corrected.median():.1f} mm; the other period's correction {swapped.median():.1f} mm;"
              f" own correction smaller than as fitted in {int((corrected < fitted - 0.5).sum())}, larger in {int((corrected > fitted + 0.5).sum())};"
              f" own smaller than the other period's in {int((corrected < swapped - 0.5).sum())}, larger in {int((corrected > swapped + 0.5).sum())}")
