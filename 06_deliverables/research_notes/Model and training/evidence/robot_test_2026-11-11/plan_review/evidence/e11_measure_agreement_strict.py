"""
Plan review, key "evidence": the same comparison as e05 part 1, with the analyst's own stricter filter (summary.py
line 252: both measures clean AND aligned) and the signed difference the analyst used, per sheet. It reproduces the
analyst's pooled figure (1,298 pictures, median 3.1 mm, over 5 mm in 449) and splits it by sheet.
"""

import glob

import pandas

folder = "D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/stop_and_centring/"
parts = []
for path in sorted(glob.glob(folder + "demonstration_centring_0*.csv")):
    table = pandas.read_csv(path)
    table["sheet_number"] = path[-6:-4]
    parts.append(table)
centring = pandas.concat(parts)
both = centring[centring["is_clean"] & centring["is_aligned"] & centring["is_single_clean"] & centring["is_single_aligned"]].dropna(subset = ["offset_mm", "single_offset_mm"])
difference = (both["single_offset_mm"] - both["offset_mm"]).abs()
print(f"pooled: {len(both)} pictures, median {difference.median():.1f} mm, 90th percentile {difference.quantile(0.9):.1f} mm, over 5 mm in {int((difference > 5).sum())}")
for sheet, rows in both.groupby("sheet_number"):
    gap = (rows["single_offset_mm"] - rows["offset_mm"]).abs()
    size_gap = (rows["single_offset_mm"].abs() - rows["offset_mm"].abs()).abs()
    side_12 = ((rows["offset_mm"].abs() > 12.0) != (rows["single_offset_mm"].abs() > 12.0)).mean()
    side_7 = ((rows["offset_mm"].abs() > 7.0) != (rows["single_offset_mm"].abs() > 7.0)).mean()
    print(f"sheet {sheet}: {len(rows):4d} pictures; signed difference median {gap.median():5.1f} mm, over 5 mm in {100 * (gap > 5).mean():3.0f} %, over 10 mm in {100 * (gap > 10).mean():3.0f} %; "
          f"sizes differ by a median {size_gap.median():4.1f} mm, over 5 mm in {100 * (size_gap > 5).mean():3.0f} %; the two disagree on 'over 7 mm' in {100 * side_7:3.0f} % and on 'over 12 mm' in {100 * side_12:3.0f} %")
