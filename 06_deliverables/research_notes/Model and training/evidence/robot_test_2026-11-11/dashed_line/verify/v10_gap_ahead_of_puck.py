"""
Verifier check 10: is the model wrong more often when a gap lies just ahead of the puck?

For every cleaned picture of sheet 06 that my own placing put on the line: the first centimetre of the centre line
beyond the puck's front edge (1.2 to 2.2 cm ahead of the puck's centre along the line) is projected into the picture
and the share of ink pixels on it is measured in the picture itself (my ink rule, widened by 5 px for the camera
model's error; points on the puck or on the robot's parts do not count). "Gap ahead": under 30 % ink; "dash ahead":
over 70 %. The same for 2.2 to 7.2 cm ahead ("line in view ahead"). Then the fold models that never saw sheet 06 are
scored on my path direction per class.

Run:  D:/AI_Project/.venv/Scripts/python.exe v10_gap_ahead_of_puck.py > v10_gap_ahead_of_puck.out.txt
Needs v05 and v07 outputs. About one minute. Nothing connects to the robot.
"""

import cv2
import numpy
import pandas

from verify_common import (ROOT_FOLDER, VERIFY_FOLDER, VERIFY_PARAMETERS, colour_masks, crossing_indices, placed_line, sheet_line)

from config import WORKSPACE_PARAMETERS  # noqa: E402
from forearm_camera_model import project_points  # noqa: E402

cv2.setNumThreads(2)
pandas.set_option("display.width", 250)
pandas.set_option("display.max_rows", 200)
JOINT_NAMES = [f"joint_{number}" for number in range(1, 7)]
RUNS_FOLDER = ROOT_FOLDER / "04_runs" / "gpu_ki014"
RUNS = {"E recipe": "2026-11-10_resnet18_chunk_stop_weight_25", "E recipe, seed 2": "2026-11-10_resnet18_chunk_stop_weight_25_seed_2",
        "A recipe": "2026-11-09_resnet18_final_candidate"}


def angle_gap(first, second):

    return numpy.abs((numpy.asarray(first) - numpy.asarray(second) + 180.0) % 360.0 - 180.0)


def ink_share_on(line_points: numpy.ndarray, joints: numpy.ndarray, wide_ink: numpy.ndarray, is_blocked: numpy.ndarray) -> tuple:
    """Return the share of line points that lie on ink in the picture and how many points could be judged."""

    if len(line_points) == 0:
        return numpy.nan, 0
    plane = WORKSPACE_PARAMETERS["table_plane"]
    height = plane["a"] * line_points[:, 0] + plane["b"] * line_points[:, 1] + plane["c"]
    pixels = project_points(numpy.column_stack([line_points, height]), list(joints))
    with numpy.errstate(invalid = "ignore"):
        is_inside = numpy.isfinite(pixels).all(axis = 1) & (pixels[:, 0] >= 0) & (pixels[:, 0] <= 639) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= 479)
    columns = numpy.clip(numpy.round(numpy.nan_to_num(pixels[:, 0])).astype(int), 0, 639)
    rows = numpy.clip(numpy.round(numpy.nan_to_num(pixels[:, 1])).astype(int), 0, 479)
    is_judged = is_inside & ~is_blocked[rows, columns]
    if is_judged.sum() == 0:
        return numpy.nan, 0
    return float(wide_ink[rows[is_judged], columns[is_judged]].mean()), int(is_judged.sum())


def main() -> None:

    placements = {}
    for text_line in (VERIFY_FOLDER / "v05_demonstration_tracks_06.out.txt").read_text().splitlines():
        parts = text_line.split()
        if len(parts) > 12 and parts[0].startswith("06-"):
            placements[parts[0]] = {"angle": numpy.radians(float(parts[4])), "x": float(parts[5]) / 100.0, "y": float(parts[6]) / 100.0,
                                    "mirrored": parts[7] == "True", "travel": parts[11]}
    rows = pandas.read_csv(VERIFY_FOLDER / "v07_rows_05_06.csv")
    rows = rows[(rows["sheet"] == "path_06") & rows["along_cm"].notna() & (rows["part"] != "not placed")].reset_index(drop = True)
    line_sheet = sheet_line("06")
    kernel = numpy.ones((11, 11), numpy.uint8)
    lines = {}
    near_share, near_points, far_share, far_points = [], [], [], []
    for row in rows.itertuples():
        if row.plan_number not in lines:
            placement = placements[row.plan_number]
            line = placed_line(line_sheet, [placement["angle"], placement["x"], placement["y"]], placement["mirrored"])
            if placement["travel"] == "last_end":
                line = line[::-1].copy()
            lines[row.plan_number] = (line, crossing_indices(line))
        line, crossing = lines[row.plan_number]
        here = int(round(row.along_cm * 10.0)) + crossing[0]
        picture = cv2.imread(str(VERIFY_PARAMETERS["raw_folder"] / row.frame_path))
        masks = colour_masks(picture)
        wide_ink = cv2.dilate(masks["ink"].astype(numpy.uint8), kernel) > 0
        is_blocked = cv2.dilate(masks["puck"].astype(numpy.uint8), numpy.ones((5, 5), numpy.uint8)) > 0
        is_blocked[330:, 190:] = True
        joints = [getattr(row, name) for name in JOINT_NAMES]
        share, points = ink_share_on(line[min(here + 12, len(line)):min(here + 22, len(line))], joints, wide_ink, is_blocked)
        near_share.append(share)
        near_points.append(points)
        share, points = ink_share_on(line[min(here + 22, len(line)):min(here + 72, len(line))], joints, wide_ink, is_blocked)
        far_share.append(share)
        far_points.append(points)
    rows["ink_first_cm_ahead"] = near_share
    rows["points_first_cm_ahead"] = near_points
    rows["ink_2_to_7_cm_ahead"] = far_share
    rows["points_2_to_7_cm_ahead"] = far_points
    rows["ahead"] = numpy.where(rows["points_first_cm_ahead"] < 5, "not visible",
                                numpy.where(rows["ink_first_cm_ahead"] < 0.3, "gap ahead", numpy.where(rows["ink_first_cm_ahead"] > 0.7, "dash ahead", "edge of a dash")))
    rows["view"] = numpy.where(rows["points_2_to_7_cm_ahead"] < 10, "line ahead not in view",
                               numpy.where(rows["ink_2_to_7_cm_ahead"] < 0.2, "hardly any ink ahead", "ink ahead in view"))
    rows.to_csv(VERIFY_FOLDER / "v10_gap_ahead_rows.csv", index = False)

    print(f"sheet 06: {len(rows)} cleaned pictures placed on the line ({(rows['part'] == 'in the loop').sum()} in the loop)")
    print("pictures by what lies in the first centimetre beyond the puck (measured in the picture):")
    print(rows.groupby(["part", "ahead"]).size().unstack(fill_value = 0).to_string())
    print("\npath direction within 45 degrees by class, models that never saw sheet 06 (scored rows only):")
    table = []
    for label, folder in RUNS.items():
        saved = numpy.load(RUNS_FOLDER / folder / "predictions_without_path_06.npz")
        predictions = pandas.DataFrame({"frame_index": saved["frame_index"], "angle_deg": saved["angle_deg"], "twist_deg": saved["twist_deg"]})
        merged = rows.merge(predictions, on = "frame_index")
        merged = merged[merged["is_path_row"]]
        merged["is_right"] = angle_gap(merged["angle_deg"], merged["my_path_angle"]) <= 45.0
        for part_name, part in (("in the loop", merged[merged["part"] == "in the loop"]), ("whole line", merged)):
            for column in ("ahead", "view"):
                for name, group in part.groupby(column):
                    table.append({"model": label, "part": part_name, "class": name, "pictures": len(group), "within_45": group["is_right"].mean(),
                                  "standard_error": numpy.sqrt(group["is_right"].mean() * (1 - group["is_right"].mean()) / max(len(group), 1))})
    table = pandas.DataFrame(table)
    print(table.pivot_table(index = ["part", "class"], columns = "model", values = "within_45").round(3).to_string())
    print("\npictures and one standard error per class (E recipe):")
    print(table[table["model"] == "E recipe"][["part", "class", "pictures", "standard_error"]].round(3).to_string(index = False))



if __name__ == "__main__":
    main()
