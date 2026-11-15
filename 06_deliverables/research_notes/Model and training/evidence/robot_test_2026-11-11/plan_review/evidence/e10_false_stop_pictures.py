"""
Plan review, key "evidence": the last two pictures (the two stop pictures) of the five failed runs that ended by the
model's own stop rule (runs 5, 6, 8, 9, 10), with the logged stop chance written on each. Reads pictures and logs only.
"""

from pathlib import Path

import pandas
from PIL import Image, ImageDraw

MODEL_RUNS = Path("D:/AI_Project/03_data/model_runs")
RUNS = [(5, "2026-11-11_173549_E_sheet03_run1"), (6, "2026-11-11_174108_E_sheet04_run1"), (8, "2026-11-11_174936_E_sheet06_run1"),
        (9, "2026-11-11_175203_E_sheet03_run2_no_hoodie"), (10, "2026-11-11_175544_B_team5_sheet03")]
sheet = Image.new("RGB", (480 * 3, 360 * len(RUNS)), (255, 255, 255))
for row, (number, name) in enumerate(RUNS):
    steps = pandas.read_csv(MODEL_RUNS / name / "steps.csv")
    for column, index in enumerate((len(steps) - 3, len(steps) - 2, len(steps) - 1)):
        picture = Image.open(MODEL_RUNS / name / steps["frame"].iloc[index]).convert("RGB").resize((480, 360))
        ImageDraw.Draw(picture).text((8, 6), f"run {number}, picture {int(steps['step'].iloc[index])}, stop chance {steps['model_stop_probability'].iloc[index]:.2f}, {steps['action'].iloc[index]}", fill = (255, 255, 0))
        sheet.paste(picture, (480 * column, 360 * row))
sheet.save(Path(__file__).resolve().parent / "e10_false_stop_pictures.jpg", quality = 85)
print("written e10_false_stop_pictures.jpg", sheet.size)
