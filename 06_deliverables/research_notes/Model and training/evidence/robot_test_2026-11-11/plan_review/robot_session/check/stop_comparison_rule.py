"""
Check of the critic's fix in P11: "compare where the puck stops with the 3 control runs; if the median is more than
3 mm shorter, the end push is looked at". How often does that rule fire by luck with 3 against 3 runs, and how often
does it catch a real shift? The scatter of a stop along the line is taken as 2.8 mm (the four robot stops of 11.11.26:
3.9, 1.9, 7.5 and 1.3 mm beyond the end's centre) and as 4.2 mm (the demonstrations' ends, check verdict 4 of stop and
centring). Simulated with a fixed seed; a guess of the shape (normal), not a measurement.

Run:  D:/AI_Project/.venv/Scripts/python.exe stop_comparison_rule.py > stop_comparison_rule.out.txt
"""

import numpy

generator = numpy.random.default_rng(11_11_2026)
SESSIONS = 200_000
print("scatter of the four robot stops of 11.11 along the line:", round(float(numpy.std([3.9, 1.9, 7.5, 1.3], ddof = 1)), 1), "mm")
print("scatter mm | true shift mm | critic's rule (median of 3 more than 3 mm shorter) | rule 'all three new stops shorter than all three old ones'")
for scatter in (2.8, 4.2):
    for shift in (0.0, 3.5, 5.0, 8.0):
        old = generator.normal(0.0, scatter, (SESSIONS, 3))
        new = generator.normal(-shift, scatter, (SESSIONS, 3))
        by_median = numpy.mean(numpy.median(new, axis = 1) < numpy.median(old, axis = 1) - 3.0)
        all_shorter = numpy.mean(new.max(axis = 1) < old.min(axis = 1))
        print(f"   {scatter:.1f} | {shift:.1f} | {by_median:.2f} | {all_shorter:.2f}")
print("(3.5 mm is what a 0.65 cm step takes off the last move, 8 mm what the model's own length takes off [calculated in the check])")
