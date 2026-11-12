"""
Verifier check 15: how well the demonstrations and the two runs keep the puck's centre on the line (my own placing).

Run:  D:/AI_Project/.venv/Scripts/python.exe v15_puck_on_the_line.py > v15_puck_on_the_line.out.txt
Needs v04 and v05 outputs.
"""

import numpy
import pandas

from verify_common import VERIFY_FOLDER

for sheet, travel, end, run_label in (("06", "first_end", "A", "run_8"), ("05", "last_end", "B", "run_7")):
    tracks = pandas.read_csv(VERIFY_FOLDER / f"v05_demonstration_tracks_{sheet}.csv")
    progress = tracks.groupby("plan")["from_start_cm"].agg(lambda values: values.max() - values.min())
    whole = tracks[(tracks["travel_from"] == travel) & (tracks["plan"].map(progress) >= 35.0)]
    beside = whole["beside_cm"].abs().dropna()
    loop = whole[(whole["along_cm"] >= 0) & (whole["along_cm"] <= whole["loop_length_cm"])]["beside_cm"].abs().dropna()
    print(f"sheet {sheet} from end {end}: {whole['plan'].nunique()} whole-line demonstrations, {len(beside)} pictures with the puck placed:")
    print(f"   puck's centre within 0.5 cm of the line's middle in {100 * (beside <= 0.5).mean():.0f} % of pictures, further than 1.0 cm in {100 * (beside > 1.0).mean():.0f} %,"
          f" further than 1.7 cm (puck no longer touching the line) in {100 * (beside > 1.7).mean():.0f} %; median {beside.median():.2f} cm")
    print(f"   in the loop only: within 0.5 cm in {100 * (loop <= 0.5).mean():.0f} %, further than 1.0 cm in {100 * (loop > 1.0).mean():.0f} %, median {loop.median():.2f} cm")
    per_plan = whole.groupby("plan")["beside_cm"].agg(lambda values: values.abs().quantile(0.98))
    print(f"   per demonstration, the 98th percentile of the distance: median {per_plan.median():.1f} cm, range {per_plan.min():.1f} to {per_plan.max():.1f} cm")
    run = pandas.read_csv(VERIFY_FOLDER / "v04_run_tracks.csv")
    run = run[run["run"] == run_label]
    run_beside = run["beside_all_cm"].abs()
    loop_length = whole["loop_length_cm"].iloc[0]
    run_loop = run[(run["along_cm"] >= 0) & (run["along_cm"] <= loop_length)]["beside_all_cm"].abs()
    print(f"   {run_label}: within 0.5 cm in {int((run_beside <= 0.5).sum())} of {len(run_beside)} pictures, further than 1.0 cm in {int((run_beside > 1.0).sum())},"
          f" further than 1.7 cm in {int((run_beside > 1.7).sum())}; largest {run_beside.max():.1f} cm; median {run_beside.median():.2f} cm; in the loop: median {run_loop.median():.2f} cm")
