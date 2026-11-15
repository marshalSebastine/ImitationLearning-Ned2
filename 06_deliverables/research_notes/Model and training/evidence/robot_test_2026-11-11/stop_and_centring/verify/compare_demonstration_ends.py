"""
Verifier's check 4b (11.11.26): my demonstration ends (verify_demonstration_ends_*.csv) against the analyst's
(../demonstration_ends_*.csv): first the analyst's own counts recomputed from the analyst's table, then my numbers,
then the two side by side per episode.
"""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent
LIMIT_MM = 7.0

pandas.set_option("display.width", 250)
analyst = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_ends_01_03.csv", "demonstration_ends_04_06.csv")], ignore_index = True)
analyst["is_reliable"] = (analyst["is_end_found"] & (analyst["is_end_seen"] == True) & analyst["line_width_mm"].between(8.0, 12.5) & (analyst["fit_quality"] >= 0.8))
reliable = analyst[analyst["is_reliable"]]
print("THE ANALYST'S TABLE, recounted")
print(f"episodes {len(analyst)}, reliable {len(reliable)}; covered (centres at most 7 mm apart) {int((reliable['centre_gap_mm'] <= LIMIT_MM).sum())};"
      f" within 12 mm {int((reliable['centre_gap_mm'] <= 12.0).sum())}")
print(f"short: mean {reliable['short_mm'].mean():+.1f}, sd {reliable['short_mm'].std():.1f}; over 7 short {int((reliable['short_mm'] > 7).sum())}, over 7 beyond {int((reliable['short_mm'] < -7).sum())}")
print(f"aside: mean {reliable['aside_mm'].mean():+.1f}, sd {reliable['aside_mm'].std():.1f}, median size {reliable['aside_mm'].abs().median():.1f}; over 7 {int((reliable['aside_mm'].abs() > 7).sum())}")
sheet_01 = reliable[reliable["sheet"] == "path_01"]
print(f"sheet 01: reliable {len(sheet_01)}, covered {int((sheet_01['centre_gap_mm'] <= LIMIT_MM).sum())}, median size aside {sheet_01['aside_mm'].abs().median():.1f}, mean aside {sheet_01['aside_mm'].mean():+.1f}")
print(f"stop picture shows ink ahead (over 0 square mm): {int((reliable['ink_seen_ahead_mm2'] > 0).sum())}; ink beyond the puck seen from above over 5 square mm: {int((reliable['ink_beyond_mm2'] > 5).sum())}")
print(f"all 210, whatever the reliability: end found {int(analyst['is_end_found'].sum())}, covered {int((analyst['centre_gap_mm'] <= LIMIT_MM).sum())}")

for camera_name in ("corrected", "fitted"):
    mine = pandas.read_csv(HERE / f"verify_demonstration_ends_{camera_name}_01_06.csv")
    measured = mine[mine["references"] >= 1]
    steady = mine[(mine["references"] >= 3) & ((mine["aside_high_mm"] - mine["aside_low_mm"]) <= 8.0) & ((mine["short_high_mm"] - mine["short_low_mm"]) <= 10.0)]
    print()
    print(f"MY MEASURE, camera {camera_name}")
    for label, group in (("every episode with at least one reference picture", measured),
                         ("episodes with 3 or more reference pictures that agree (sideways range 8 mm, along 10 mm)", steady)):
        gap = numpy.hypot(group["short_mm"], group["aside_mm"])
        print(f"  {label}: {len(group)} episodes")
        print(f"    short: mean {group['short_mm'].mean():+.1f}, sd {group['short_mm'].std():.1f}, median {group['short_mm'].median():+.1f}; over 7 short {int((group['short_mm'] > 7).sum())}, over 7 beyond {int((group['short_mm'] < -7).sum())}")
        print(f"    aside: mean {group['aside_mm'].mean():+.1f}, sd {group['aside_mm'].std():.1f}, median size {group['aside_mm'].abs().median():.1f}; over 7 {int((group['aside_mm'].abs() > 7).sum())}")
        print(f"    centres at most 7 mm apart: {int((gap <= LIMIT_MM).sum())} of {len(group)} ({100 * (gap <= LIMIT_MM).mean():.0f} %); at most 12 mm: {int((gap <= 12).sum())}; median gap {gap.median():.1f} mm")
        by_sheet = group.assign(gap = gap).groupby("sheet").apply(lambda part: pandas.Series({
            "episodes": len(part), "covered": int((part["gap"] <= LIMIT_MM).sum()), "short_mean": round(part["short_mm"].mean(), 1),
            "aside_mean": round(part["aside_mm"].mean(), 1), "aside_sd": round(part["aside_mm"].std(), 1), "aside_size_median": round(part["aside_mm"].abs().median(), 1)}))
        print(by_sheet.to_string().replace("\n", "\n    "))
    both = steady.merge(analyst, on = "episode_id", suffixes = ("_mine", "_analyst"))
    both = both[both["is_reliable"]]
    difference_short = both["short_mm_mine"] - both["short_mm_analyst"]
    difference_aside = both["aside_mm_mine"] - both["aside_mm_analyst"]
    gap_mine = numpy.hypot(both["short_mm_mine"], both["aside_mm_mine"])
    print(f"  side by side on the {len(both)} episodes that both measured reliably:")
    print(f"    mine minus the analyst's, short: median {difference_short.median():+.1f} mm, median size {difference_short.abs().median():.1f}, 90th percentile of the size {difference_short.abs().quantile(0.9):.1f}")
    print(f"    mine minus the analyst's, aside: median {difference_aside.median():+.1f} mm, median size {difference_aside.abs().median():.1f}, 90th percentile of the size {difference_aside.abs().quantile(0.9):.1f}")
    print(f"    correlation short {numpy.corrcoef(both['short_mm_mine'], both['short_mm_analyst'])[0, 1]:.2f}, aside {numpy.corrcoef(both['aside_mm_mine'], both['aside_mm_analyst'])[0, 1]:.2f}")
    is_mine = gap_mine <= LIMIT_MM
    is_theirs = both["centre_gap_mm"] <= LIMIT_MM
    print(f"    covered by mine {int(is_mine.sum())}, by the analyst's {int(is_theirs.sum())}; both {int((is_mine & is_theirs).sum())}, only mine {int((is_mine & ~is_theirs).sum())},"
          f" only the analyst's {int((~is_mine & is_theirs).sum())}, neither {int((~is_mine & ~is_theirs).sum())}")
    both.assign(gap_mine = gap_mine)[["episode_id", "sheet_mine", "short_mm_mine", "short_mm_analyst", "aside_mm_mine", "aside_mm_analyst", "gap_mine", "centre_gap_mm", "references"]].to_csv(
        HERE / f"compare_demonstration_ends_{camera_name}.csv", index = False, float_format = "%.2f")
