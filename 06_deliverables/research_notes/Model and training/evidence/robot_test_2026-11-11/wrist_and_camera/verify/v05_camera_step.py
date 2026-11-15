"""
Verifier, check 5b: did the camera step on its tape on 8.11.26, and where does it sit in the robot test? Measured on the
saved pictures with my OWN method, not the recording tool's phase correlation: the blue fork at the bottom of the
picture is part of the forearm, as the camera is. I turn the bottom strip into a "blueness" picture (blue minus red,
which is near zero on wood, paper, the grey rod and the black block), cut a template from one early demonstration and
find it in every other picture by normalised cross-correlation (cv2.matchTemplate), with a parabola for sub-pixel.

Pictures: the first row of every start-mark demonstration, and every third picture of the ten robot runs.
    D:/AI_Project/.venv/Scripts/python.exe v05_camera_step.py
"""

import glob

import cv2
import numpy
import pandas

RAW = "D:/AI_Project/03_data/raw/main_data"
RUNS = "D:/AI_Project/03_data/model_runs"
STRIP = (slice(395, 480), slice(120, 520))     # rows, columns of the search strip
MARGIN = 40                                    # the template is the strip without this margin left and right, 12 px on top


def blueness(path: str) -> numpy.ndarray:
    picture = cv2.imread(path).astype(numpy.float32)
    return numpy.clip(picture[:, :, 0] - picture[:, :, 2], 0, 255)[STRIP]


def refined(values: numpy.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    left, centre, right = values[index - 1], values[index], values[index + 1]
    denominator = left - 2 * centre + right
    return float(index) if abs(denominator) < 1e-12 else index + 0.5 * (left - right) / denominator


def shift_of(strip: numpy.ndarray, template: numpy.ndarray) -> tuple:
    scores = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
    _, best, _, location = cv2.minMaxLoc(scores)
    x = refined(scores[location[1], :], location[0]) - MARGIN
    y = refined(scores[:, location[0]], location[1]) - 12
    return x, y, best


frames = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False,
                         usecols = ["episode_id", "row_in_episode", "sheet", "sheet_orientation", "frame_path", "started", "camera_shift_px", "joint_5", "joint_6"])
first = frames[(frames["row_in_episode"] == 0) & frames["sheet_orientation"].str.contains("START MARK", na = False)].sort_values("started").reset_index(drop = True)
first["plan"] = first["episode_id"].str.extract(r"plan_(\d\d-\d\d)")[0]
reference = blueness(f"{RAW}/{first.loc[0, 'frame_path']}")
template = numpy.ascontiguousarray(reference[12:, MARGIN:-MARGIN])
print("reference picture:", first.loc[0, "frame_path"])

rows = []
for _, row in first.iterrows():
    x, y, score = shift_of(blueness(f"{RAW}/{row['frame_path']}"), template)
    rows.append({"started": row["started"], "sheet": row["sheet"], "plan": row["plan"], "shift_x": x, "shift_y": y, "score": score, "tool_shift_px": row["camera_shift_px"]})
demos = pandas.DataFrame(rows)
demos.to_csv("v05_camera_step_demonstrations.csv", index = False)
pandas.set_option("display.width", 200)
pandas.set_option("display.max_rows", 400)
print("\nfirst pictures of the 149 start-mark demonstrations, by day and sheet (px at 640, against the first demonstration of 6.11):")
demos["day"] = demos["started"].str[:10]
print(demos.groupby(["day", "sheet"]).agg(episodes = ("shift_x", "size"), x_median = ("shift_x", "median"), x_least = ("shift_x", "min"), x_most = ("shift_x", "max"),
                                          y_median = ("shift_y", "median"), score_least = ("score", "min")).round(1).to_string())
moved = demos["shift_x"] > 8.0
first_moved = int(numpy.flatnonzero(moved.to_numpy())[0])
print("\nrows around the first picture with the fork over 8 px to the right:")
print(demos.iloc[max(0, first_moved - 6):first_moved + 6][["started", "plan", "shift_x", "shift_y", "score", "tool_shift_px"]].round(1).to_string())
print("\nafter that picture, pictures back under 8 px: %d of %d; before it, pictures over 8 px: %d of %d" % (
    int((~moved.iloc[first_moved:]).sum()), len(demos) - first_moved, int(moved.iloc[:first_moved].sum()), first_moved))
print("before: x median %+.1f (5th %+.1f, 95th %+.1f), y median %+.1f; after: x median %+.1f (5th %+.1f, 95th %+.1f), y median %+.1f" % (
    demos["shift_x"].iloc[:first_moved].median(), demos["shift_x"].iloc[:first_moved].quantile(0.05), demos["shift_x"].iloc[:first_moved].quantile(0.95),
    demos["shift_y"].iloc[:first_moved].median(),
    demos["shift_x"].iloc[first_moved:].median(), demos["shift_x"].iloc[first_moved:].quantile(0.05), demos["shift_x"].iloc[first_moved:].quantile(0.95),
    demos["shift_y"].iloc[first_moved:].median()))
step_time = demos.loc[first_moved, "started"]

# how many training rows were taken before the step (by the episode's start time)
all_rows = pandas.read_csv("D:/AI_Project/03_data/processed/v01_training_index/frames.csv", low_memory = False, usecols = ["episode_id", "sheet", "started"])
before = all_rows["started"] < step_time
print("\nrows of episodes started before %s: %d of %d (%.1f %%)" % (step_time, int(before.sum()), len(all_rows), 100 * before.mean()))
print(all_rows.assign(before = before).groupby("sheet")["before"].agg(["sum", "size"]).to_string())

print("\nthe robot runs of 11.11.26 (every third picture):")
run_rows = []
for folder in sorted(glob.glob(f"{RUNS}/2026-11-11_17*")):
    shifts = numpy.array([shift_of(blueness(path), template) for path in sorted(glob.glob(f"{folder}/frames/*.jpg"))[::3]])
    good = shifts[shifts[:, 2] > 0.6]
    run_rows.append({"run": folder.replace("\\", "/").split("/")[-1][11:], "pictures": len(shifts), "good": len(good), "x_median": numpy.median(good[:, 0]), "x_p10": numpy.quantile(good[:, 0], 0.1),
                     "x_p90": numpy.quantile(good[:, 0], 0.9), "y_median": numpy.median(good[:, 1]), "score_median": numpy.median(shifts[:, 2])})
runs = pandas.DataFrame(run_rows)
print(runs.round(1).to_string(index = False))
print("\nrobot runs: x median of the run medians %+.1f px, y %+.1f px; demonstrations after the step: x %+.1f, y %+.1f; before the step: x %+.1f, y %+.1f" % (
    runs["x_median"].median(), runs["y_median"].median(), demos["shift_x"].iloc[first_moved:].median(), demos["shift_y"].iloc[first_moved:].median(),
    demos["shift_x"].iloc[:first_moved].median(), demos["shift_y"].iloc[:first_moved].median()))
runs.to_csv("v05_camera_step_runs.csv", index = False)
