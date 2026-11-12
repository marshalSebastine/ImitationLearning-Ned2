"""
Verifier, check 5c: the vertical part of the camera step, same method as v05_camera_step.py but with room to move up
and down (template rows 400 to 462 searched in rows 380 to 480, columns 160 to 480 in 120 to 520).
    D:/AI_Project/.venv/Scripts/python.exe v05_camera_step_vertical.py
"""
import glob
import cv2
import numpy
import pandas

RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"

def blueness(path):
    picture = cv2.imread(path).astype(numpy.float32)
    return numpy.clip(picture[:, :, 0] - picture[:, :, 2], 0, 255)

def refined(values, index):
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2 * centre + right
    return float(index) if abs(denominator) < 1e-12 else index + 0.5 * (left - right) / denominator

def shift_of(picture, template):
    scores = cv2.matchTemplate(numpy.ascontiguousarray(picture[380:480, 120:520]), template, cv2.TM_CCOEFF_NORMED)
    _, best, _, location = cv2.minMaxLoc(scores)
    return refined(scores[location[1], :], location[0]) - 40, refined(scores[:, location[0]], location[1]) - 20, best

frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "frame_path", "started"])
first = frames[(frames["row_in_episode"] == 0) & frames["sheet_orientation"].str.contains("START MARK", na = False)].sort_values("started").reset_index(drop = True)
template = numpy.ascontiguousarray(blueness(f"{RAW}/{first.loc[0, 'frame_path']}")[400:462, 160:480])
rows = [shift_of(blueness(f"{RAW}/{path}"), template) for path in first["frame_path"]]
demos = pandas.DataFrame(rows, columns = ["x", "y", "score"])
demos["started"] = first["started"]
is_after = demos["started"] >= "2026-11-08T19:33:39"
for label, part in (("before the step", demos[~is_after]), ("after the step", demos[is_after])):
    print("%s (%d first pictures): x median %+.1f, y median %+.1f (5th %+.1f, 95th %+.1f), least score %.2f" % (
        label, len(part), part["x"].median(), part["y"].median(), part["y"].quantile(0.05), part["y"].quantile(0.95), part["score"].min()))
run_values = []
for folder in sorted(glob.glob(f"{RUNS}/2026-11-11_17*")):
    shifts = numpy.array([shift_of(blueness(path), template) for path in sorted(glob.glob(f"{folder}/frames/*.jpg"))[::3]])
    good = shifts[shifts[:, 2] > 0.6]
    run_values.append((numpy.median(good[:, 0]), numpy.median(good[:, 1])))
    print("  %-34s x %+.1f, y %+.1f (%d of %d pictures with a score over 0.6)" % (folder.replace("\\", "/").split("/")[-1][11:], run_values[-1][0], run_values[-1][1], len(good), len(shifts)))
run_values = numpy.array(run_values)
print("robot runs, median of the run medians: x %+.1f, y %+.1f" % (numpy.median(run_values[:, 0]), numpy.median(run_values[:, 1])))
