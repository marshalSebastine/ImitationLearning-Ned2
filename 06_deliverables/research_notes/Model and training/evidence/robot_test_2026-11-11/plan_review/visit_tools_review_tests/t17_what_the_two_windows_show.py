"""Review check: what the two windows really show (the run window with a 0.65 cm step; the go-ahead window with the faint reference over real camera pictures)."""

from pathlib import Path

import cv2
import numpy
import pandas

import safe_import
from safe_import import model_mode
from config import TRAINING_DATA_PARAMETERS

HERE = Path(__file__).resolve().parent
frames = pandas.read_csv(TRAINING_DATA_PARAMETERS["processed_folder"] / "frames.csv", usecols = ["episode_id", "row_in_episode", "frame_path", "sheet"])
frames = frames.sort_values(["episode_id", "row_in_episode"])
first = frames.groupby("episode_id").head(1)
print(first.groupby("sheet").size().to_dict())
raw = Path(TRAINING_DATA_PARAMETERS["raw_folder"])
same_sheet = first[first["sheet"] == first["sheet"].iloc[0]]
ghost_path = raw / same_sheet["frame_path"].iloc[0]
live_path = raw / same_sheet["frame_path"].iloc[min(5, len(same_sheet) - 1)]
other_sheet = first[first["sheet"] != first["sheet"].iloc[0]]
other_path = raw / other_sheet["frame_path"].iloc[0]
print("ghost:", ghost_path.name, ghost_path.parent.parent.name, "| live:", live_path.parent.parent.name, "| other sheet:", other_path.parent.parent.name)

shown = []
ghost = model_mode.reference_picture(ghost_path)
print("ghost shape:", ghost.shape)
keys = iter([None, "enter"])
model_mode.print = lambda *values, **options: None
for name, path in (("same_sheet", live_path), ("other_sheet", other_path)):
    keys = iter(["enter"])
    model_mode.go_ahead_with_ghost(lambda: path.read_bytes(), lambda wait: next(keys), ghost, show = shown.append)


# the run window with a 0.65 cm step
model_mode.show_in_window = shown.append
runner = model_mode.ModelRun(None, None, {"step_m": 0.0065})
row = {"step": 17, "action": "model_step", "model_stop_probability": 0.07, "model_twist_deg": -2.4, "model_angle_deg": 12.0, "step_cm": 0.65}
runner.show(live_path.read_bytes(), row)

strip = numpy.vstack([shown[0], shown[1], shown[2]])
cv2.imwrite(str(HERE / "t17_what_the_two_windows_show.jpg"), strip)
print("written: t17_what_the_two_windows_show.jpg (the go-ahead window on the same sheet, on another sheet, and the run window with a 0.65 cm step)")
