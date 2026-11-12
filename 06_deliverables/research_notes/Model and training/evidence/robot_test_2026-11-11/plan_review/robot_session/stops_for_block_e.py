"""
Robot-session critic: block E wants "measure and photo within 3 mm at 10 or more stops". Stops at a line's end can only
come from the full runs of blocks B and C (12 + 8 = 20; block A's runs are ended by hand after the bend, block D's rule
looks at the first 10 cm). Chance that 20 full runs give 10 or more stops at the end, for several true finish rates.
On 11.11.26 four of ten runs stopped at the end, all on sheets their model had trained on. Calculated.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe stops_for_block_e.py > stops_for_block_e.out.txt
"""

from math import comb

for runs in (20, 24):
    for chance in (0.2, 0.3, 0.4, 0.5, 0.6):
        enough = sum(comb(runs, count) * chance ** count * (1 - chance) ** (runs - count) for count in range(10, runs + 1))
        print(f"{runs} full runs, each ending at the line's end with chance {chance:.1f}: 10 or more stops with chance {enough:.2f}; expected stops {runs * chance:.0f}")
