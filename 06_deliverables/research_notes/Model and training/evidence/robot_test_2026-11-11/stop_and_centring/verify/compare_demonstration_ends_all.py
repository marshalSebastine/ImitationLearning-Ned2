"""
Verifier's check 4c (11.11.26): every demonstration with at least one of my reference pictures against the analyst's
reliable ends, and check 16: how many existing demonstrations would pass a gate of 5 mm and how often two measures of
the same stop disagree on the verdict. Reads my tables (verify_demonstration_ends_*.csv) and the analyst's
(../demonstration_ends_*.csv).
"""

from pathlib import Path

import numpy
import pandas

HERE = Path(__file__).resolve().parent

analyst = pandas.concat([pandas.read_csv(HERE.parent / name) for name in ("demonstration_ends_01_03.csv", "demonstration_ends_04_06.csv")], ignore_index = True)
analyst["is_reliable"] = (analyst["is_end_found"] & (analyst["is_end_seen"] == True) & analyst["line_width_mm"].between(8.0, 12.5) & (analyst["fit_quality"] >= 0.8))
reliable = analyst[analyst["is_reliable"]]
print("CHECK 4c: every episode with at least one of my reference pictures, against the analyst's reliable ends")
for camera in ("corrected", "fitted"):
    mine = pandas.read_csv(HERE / f"verify_demonstration_ends_{camera}_01_06.csv")
    mine = mine[mine["references"] >= 1]
    both = mine.merge(reliable, on = "episode_id", suffixes = ("_mine", "_analyst"))
    apart = numpy.hypot(both["short_mm_mine"] - both["short_mm_analyst"], both["aside_mm_mine"] - both["aside_mm_analyst"])
    gap_mine = numpy.hypot(both["short_mm_mine"], both["aside_mm_mine"])
    print(f"camera {camera}: {len(both)} episodes measured by both")
    print(f"  my place of the puck against the end differs from the analyst's by: median {apart.median():.1f} mm, 75th percentile {apart.quantile(0.75):.1f}, 90th {apart.quantile(0.9):.1f};"
          f" over 5 mm in {int((apart > 5).sum())}, over 10 mm in {int((apart > 10).sum())}")
    close = both[apart <= 5]
    print(f"  on the {len(close)} episodes where the two agree within 5 mm: covered (7 mm) by mine {int((numpy.hypot(close['short_mm_mine'], close['aside_mm_mine']) <= 7).sum())},"
          f" by the analyst's {int((close['centre_gap_mm'] <= 7).sum())}")
    print(f"     short: mine mean {close['short_mm_mine'].mean():+.1f} sd {close['short_mm_mine'].std():.1f}; analyst mean {close['short_mm_analyst'].mean():+.1f} sd {close['short_mm_analyst'].std():.1f}")
    print(f"     aside: mine mean {close['aside_mm_mine'].mean():+.1f} sd {close['aside_mm_mine'].std():.1f}; analyst mean {close['aside_mm_analyst'].mean():+.1f} sd {close['aside_mm_analyst'].std():.1f}")
    is_mine, is_theirs = gap_mine <= 7, both["centre_gap_mm"] <= 7
    print(f"  all {len(both)}: covered by mine {int(is_mine.sum())} ({100 * is_mine.mean():.0f} %), by the analyst's {int(is_theirs.sum())} ({100 * is_theirs.mean():.0f} %); same verdict in {int((is_mine == is_theirs).sum())}")
    for name in ("short_mm_mine", "aside_mm_mine", "short_mm_analyst", "aside_mm_analyst"):
        values = both[name]
        print(f"     {name}: median {values.median():+.1f}, robust spread (1.4826 x median absolute deviation) {1.4826 * (values - values.median()).abs().median():.1f} mm")
    worst = both.assign(apart = apart).sort_values("apart", ascending = False).head(8)
    print("  the eight largest disagreements (my simple end finder jumps to another piece of line on the meander and the dashed loop):")
    print(worst.assign(episode = worst["episode_id"].str[-42:])[["episode", "short_mm_mine", "short_mm_analyst", "aside_mm_mine", "aside_mm_analyst", "references"]].round(1).to_string(index = False))

print()
print("CHECK 16: the analyst's proposed gate for new episodes (centres at most 5 mm apart at the stop) on the existing demonstrations")
for limit in (5.0, 7.0, 12.0):
    print(f"  the analyst's table: centres at most {limit:.0f} mm apart in {int((reliable['centre_gap_mm'] <= limit).sum())} of {len(reliable)} ({100 * (reliable['centre_gap_mm'] <= limit).mean():.0f} %)")
mine = pandas.read_csv(HERE / "verify_demonstration_ends_corrected_01_06.csv")
both = mine[mine["references"] >= 1].merge(reliable[["episode_id", "short_mm", "aside_mm", "centre_gap_mm"]], on = "episode_id", suffixes = ("_mine", "_analyst"))
apart = numpy.hypot(both["short_mm_mine"] - both["short_mm_analyst"], both["aside_mm_mine"] - both["aside_mm_analyst"])
both = both[apart <= 10]
gap_mine = numpy.hypot(both["short_mm_mine"], both["aside_mm_mine"])
for limit in (5.0, 7.0):
    is_mine, is_theirs = gap_mine <= limit, both["centre_gap_mm"] <= limit
    print(f"  two measures on the same {len(both)} stops (without my finder's jumps), gate {limit:.0f} mm: pass by mine {int(is_mine.sum())}, by the analyst's {int(is_theirs.sum())},"
          f" same verdict in {int((is_mine == is_theirs).sum())} ({100 * (is_mine == is_theirs).mean():.0f} %), pass by one and fail by the other in {int((is_mine != is_theirs).sum())}")
