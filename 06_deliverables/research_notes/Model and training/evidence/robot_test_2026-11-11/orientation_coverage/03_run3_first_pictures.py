"""
Question 3 of the orientation-coverage analysis: was robot run 3 (sheet 02 "lying straight") really unseen, or close to
the square lay-outs SQP and SQL of the demonstrations?

Three checks, all on recorded files:
    A  picture match: every picture of runs 3 and 4 is compared with all 18,032 demonstration pictures (grey, shrunk
       to 40 x 30, brightness and contrast taken out, correlation); the nearest pictures name the lay-out they came from;
    B  the line's direction on the table where it leaves the puck, measured in the first picture of each run with the
       fitted camera, against the fitted line start of every sheet-02 lay-out (01_placements_per_episode.csv);
    C  what the model asked for at the start of run 3 against what the demonstrators did in the nearest demonstration
       pictures and at the start of the six SQP episodes (step length, forearm twist, tip travel while twisting);
    D  how far the puck lay ahead of the tip in the first picture of every start-mark demonstration and of the ten
       robot runs (puck detector and fitted camera).

Printed and saved: 03_run3_first_pictures.out.txt, 03_run3_nearest.csv, 03_run3_nearest.png (run 3's first pictures
beside their nearest demonstration pictures), 03_sqp_start_sequences.png (the start of two SQP demonstrations and of
run 3, picture by picture) and 03_start_gaps.csv (the puck's distance from the tip at every start).

Run from this folder (after 01_placements.py):
    D:/AI_Project/.venv/Scripts/python.exe 03_run3_first_pictures.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy
import pandas

import orientation_common
from orientation_common import EVIDENCE_FOLDER, demonstration_frames, robot_run_steps, wrapped_deg

from config import table_height  # noqa: E402
from forearm_camera_model import pixels_to_plane  # noqa: E402
from puck_detector import PuckDetector  # noqa: E402

RUN_3_PARAMETERS = {
    "runs_compared": [3, 4],
    "pictures_per_run": 13,
    "thumbnail_size": [40, 30],
    "neighbours": 10,
    # the line where it leaves the puck: dark pixels (grey under this share of the picture's 90th-percentile grey)
    # whose table point lies between these distances from the puck's centre, above the robot's parts
    "line_dark_share": 0.55,
    "line_ring_cm": [2.5, 6.0],
    "robot_parts_from_row": 340,
    # the start of an episode or run: up to the first row where the forearm has twisted this far
    "twist_reference_deg": 30.0,
    "square_kinds": ["SQP", "SQL"],
    "figure_columns": 6
}



def thumbnails(pictures_rgb: numpy.ndarray, size: List[int]) -> numpy.ndarray:
    """
    Turn pictures into small grey vectors with brightness and contrast taken out.

    Parameters
    ----------
    pictures_rgb : numpy.ndarray
        (N, height, width, 3) uint8.
    size : list of int
        The thumbnail's [width, height].

    Returns
    -------
    numpy.ndarray
        (N, width x height) float32, each row with zero mean and unit length, so that a dot product of two rows is
        their correlation.
    """

    vectors = numpy.zeros((len(pictures_rgb), size[0] * size[1]), dtype = numpy.float32)
    for index, picture in enumerate(pictures_rgb):
        grey = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(grey, (size[0], size[1]), interpolation = cv2.INTER_AREA).astype(numpy.float32).ravel()
        small -= small.mean()
        vectors[index] = small / max(float(numpy.linalg.norm(small)), 1e-6)
    return vectors



def run_pictures(folder: Path, count: int) -> List[numpy.ndarray]:
    """Return the first ``count`` pictures of a robot run as BGR arrays (640 x 480)."""

    pictures = []
    for index in range(count):
        picture = cv2.imread(str(folder / "frames" / f"{index:03d}.jpg"))
        if picture is None:
            break
        pictures.append(picture)
    return pictures



def line_direction_on_table(
        picture_bgr: numpy.ndarray,
        joints: List[float],
        tip_xy: numpy.ndarray,
        detector: PuckDetector,
        parameters: Dict
) -> Optional[Dict]:
    """
    Measure where the line leaves the puck, on the table.

    The dark pixels of the picture above the robot's parts are placed on the table with the fitted camera; those in a
    ring round the puck's centre belong to the line where it leaves the puck (the puck hides the line under it). Their
    mean direction from the puck's centre is the line's direction.

    Returns
    -------
    dict or None
        "direction_deg" (table frame), "from_picture_top_deg", "puck_xy_cm" and "pixels" (how many dark pixels were
        used); None when the puck or the line was not found.
    """

    sighting = detector.detect(picture_bgr)
    if sighting is None:
        return None
    height = table_height(tip_xy[0], tip_xy[1])
    puck = pixels_to_plane(numpy.array([[sighting["centre_u"], sighting["centre_v"]]]), joints, height + 0.0145)[0]
    grey = cv2.cvtColor(picture_bgr, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
    is_dark = grey < parameters["line_dark_share"] * numpy.percentile(grey, 90)
    is_dark[parameters["robot_parts_from_row"]:] = False
    rows, columns = numpy.nonzero(is_dark)
    if len(rows) == 0:
        return None
    points = pixels_to_plane(numpy.column_stack([columns, rows]).astype(float), joints, height)
    offsets = 100.0 * (points - puck[None, :])
    distance = numpy.hypot(offsets[:, 0], offsets[:, 1])
    in_ring = numpy.isfinite(distance) & (distance >= parameters["line_ring_cm"][0]) & (distance <= parameters["line_ring_cm"][1])
    if in_ring.sum() < 30:
        return None
    angles = numpy.arctan2(offsets[in_ring, 1], offsets[in_ring, 0])
    direction = math.degrees(math.atan2(numpy.sin(angles).mean(), numpy.cos(angles).mean()))
    spread = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(1e-9, float(numpy.hypot(numpy.sin(angles).mean(), numpy.cos(angles).mean())))))))
    return {"direction_deg": direction, "spread_deg": spread, "puck_xy_cm": 100.0 * puck, "pixels": int(in_ring.sum())}



def start_summary(
        joint_4_deg: numpy.ndarray,
        tip_xy_cm: numpy.ndarray,
        step_cm: numpy.ndarray,
        twist_reference_deg: float
) -> Dict:
    """
    Describe how a start turned the forearm: the rows and the tip travel until joint 4 had changed by
    ``twist_reference_deg`` from its first value, and how much of that travel came before the twist began (5 degrees).
    """

    change = numpy.abs(joint_4_deg - joint_4_deg[0])
    travelled = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(tip_xy_cm, axis = 0), axis = 1))])
    reached = numpy.flatnonzero(change >= twist_reference_deg)
    begun = numpy.flatnonzero(change >= 5.0)
    if len(reached) == 0:
        return {"rows_to_reference": None, "travel_to_reference_cm": None, "travel_before_twist_cm": float(travelled[begun[0]]) if len(begun) else None,
                "travel_while_twisting_cm": None, "largest_twist_deg": float(change.max()), "median_step_while_twisting_cm": None}
    first, last = int(begun[0]), int(reached[0])
    return {"rows_to_reference": last, "travel_to_reference_cm": float(travelled[last]), "travel_before_twist_cm": float(travelled[first]),
            "travel_while_twisting_cm": float(travelled[last] - travelled[first]), "largest_twist_deg": float(change.max()),
            "median_step_while_twisting_cm": float(numpy.median(step_cm[first:last])) if last > first else None}



def puck_on_table_cm(
        picture_bgr: numpy.ndarray,
        joints: List[float],
        tip_xy_m: numpy.ndarray,
        detector: PuckDetector
) -> Optional[numpy.ndarray]:
    """Return where the puck's centre lies on the table, [x, y] in cm, or None when it is not found."""

    if picture_bgr is None:
        return None
    sighting = detector.detect(picture_bgr)
    if sighting is None:
        return None
    height = table_height(tip_xy_m[0], tip_xy_m[1]) + 0.0145
    puck = pixels_to_plane(numpy.array([[sighting["centre_u"], sighting["centre_v"]]]), joints, height)[0]
    if not numpy.all(numpy.isfinite(puck)):
        return None
    return 100.0 * puck



def start_gaps(
        frames: pandas.DataFrame,
        detector: PuckDetector,
        parameters: Dict
) -> pandas.DataFrame:
    """
    Measure the puck's distance from the tip in the first kept picture of every demonstration and every robot run.

    Returns
    -------
    pandas.DataFrame
        One row per episode or run: "what" ("demonstration" or "robot run"), "episode_id", "name", "sheet",
        "start_kind", "puck_x_cm", "puck_y_cm" and "puck_from_tip_cm" (NaN when the puck was not found).
    """

    def row_for(what: str, episode_id: str, name: str, sheet: str, kind: str, picture, joints: List[float], tip_xy_m: numpy.ndarray) -> Dict:
        puck = puck_on_table_cm(picture, joints, tip_xy_m, detector)
        if puck is None:
            return {"what": what, "episode_id": episode_id, "name": name, "sheet": sheet, "start_kind": kind,
                    "puck_x_cm": numpy.nan, "puck_y_cm": numpy.nan, "puck_from_tip_cm": numpy.nan}
        return {"what": what, "episode_id": episode_id, "name": name, "sheet": sheet, "start_kind": kind, "puck_x_cm": round(float(puck[0]), 2),
                "puck_y_cm": round(float(puck[1]), 2), "puck_from_tip_cm": round(float(numpy.hypot(*(puck - 100.0 * tip_xy_m))), 2)}

    rows = []
    for episode_id, group in frames.groupby("episode_id", sort = False):
        first = group.iloc[0]
        picture = cv2.imread(str(Path(parameters["raw_folder"]) / first["frame_path"]))
        joints = [float(first[f"joint_{joint}"]) for joint in range(1, 7)]
        rows.append(row_for("demonstration", episode_id, first["plan_number"], first["sheet_number"], first["start_kind"], picture, joints,
                            numpy.array([first["tip_x_m"], first["tip_y_m"]])))
    for run in parameters["robot_runs"]:
        steps = robot_run_steps(run, parameters)
        first = steps.iloc[0]
        picture = cv2.imread(str(Path(parameters["model_runs_folder"]) / run["folder"] / "frames" / "000.jpg"))
        joints = [float(first[f"joint_{joint}"]) for joint in range(1, 7)]
        rows.append(row_for("robot run", run["folder"], f"run {run['number']}", run["sheet"], run["outcome"], picture, joints,
                            numpy.array([first["tip_x"], first["tip_y"]])))
    return pandas.DataFrame(rows)



def main(parameters: Optional[Dict] = None) -> None:
    """Print the four checks and save the tables and the two contact sheets."""

    settings = orientation_common.merged_parameters(RUN_3_PARAMETERS)
    settings.update(parameters or {})
    pandas.set_option("display.width", 250)
    pandas.set_option("display.max_rows", 500)
    pandas.set_option("display.max_columns", 40)
    frames = demonstration_frames(settings)
    layouts = pandas.read_csv(EVIDENCE_FOLDER / "01_placements_per_episode.csv", dtype = {"sheet": str}).set_index("episode_id")
    frames["layout"] = frames["episode_id"].map(layouts["layout"])
    cache = numpy.load(Path(settings["index_folder"]) / "images_wide_160.npy", mmap_mode = "r")
    demonstration_vectors = thumbnails(numpy.array(cache[frames["frame_index"].to_numpy()]), settings["thumbnail_size"])
    detector = PuckDetector()
    runs = {run["number"]: run for run in settings["robot_runs"]}

    print("=== A. Nearest demonstration pictures to the first pictures of runs 3 and 4 (measured: grey 40 x 30 correlation) ===")
    nearest_rows, tiles = [], []
    for number in settings["runs_compared"]:
        run = runs[number]
        folder = Path(settings["model_runs_folder"]) / run["folder"]
        pictures = run_pictures(folder, settings["pictures_per_run"])
        steps = robot_run_steps(run, settings)
        small = numpy.array([cv2.resize(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB), (160, 120), interpolation = cv2.INTER_AREA) for picture in pictures])
        vectors = thumbnails(small, settings["thumbnail_size"])
        similarity = vectors @ demonstration_vectors.T
        for index in range(len(pictures)):
            order = numpy.argsort(-similarity[index])[:settings["neighbours"]]
            near = frames.iloc[order]
            best = near.iloc[0]
            kinds = near["start_kind"].value_counts()
            nearest_rows.append({
                "run": number, "picture": index, "joint_4_deg": round(float(steps["joint_4_deg"].iloc[index]), 1),
                "asked_cm": round(float(numpy.hypot(steps["model_tip_forward_cm"].iloc[index], steps["model_tip_left_cm"].iloc[index])), 2),
                "asked_twist_deg": round(float(steps["model_twist_deg"].iloc[index]), 1),
                "best_plan": best["plan_number"], "best_row": int(best["row_in_episode"]), "best_kind": best["start_kind"], "best_layout": best["layout"],
                "best_correlation": round(float(similarity[index, order[0]]), 3),
                "tenth_correlation": round(float(similarity[index, order[-1]]), 3),
                "neighbours_on_sheet_02": int((near["sheet_number"] == "02").sum()),
                "neighbour_kinds": ", ".join(f"{kind} {count}" for kind, count in kinds.items()),
                "neighbour_layouts": ", ".join(f"{layout} {count}" for layout, count in near["layout"].value_counts().items()),
                "label_step_cm_median": round(float(near["step_cm"].median()), 2),
                "label_twist_deg_median": round(float(near["forearm_change_deg"].median()), 1),
                "neighbour_joint_4_deg_median": round(float(near["joint_4_deg"].median()), 1)
            })
            if number == 3 and index in (0, 2, 4, 6, 8, 10):
                left = cv2.resize(pictures[index], (320, 240))
                cv2.putText(left, f"run 3, picture {index}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                row_tiles = [left]
                for rank in range(3):
                    match = frames.iloc[order[rank]]
                    tile = cv2.resize(cv2.imread(str(Path(settings["raw_folder"]) / match["frame_path"])), (320, 240))
                    cv2.putText(tile, f"{match['plan_number']} {match['start_kind']} row {int(match['row_in_episode'])} r {similarity[index, order[rank]]:.2f}",
                                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    row_tiles.append(tile)
                tiles.append(numpy.hstack(row_tiles))
    nearest = pandas.DataFrame(nearest_rows)
    nearest.to_csv(EVIDENCE_FOLDER / "03_run3_nearest.csv", index = False)
    print(nearest[["run", "picture", "joint_4_deg", "best_plan", "best_row", "best_kind", "best_layout", "best_correlation", "tenth_correlation",
                   "neighbours_on_sheet_02", "neighbour_kinds"]].to_string(index = False))
    cv2.imwrite(str(EVIDENCE_FOLDER / "03_run3_nearest.png"), numpy.vstack(tiles))
    for number in settings["runs_compared"]:
        part = nearest[nearest["run"] == number]
        square = part["best_kind"].isin(settings["square_kinds"]).sum()
        print(f"Run {number}: best match from a square lay-out (SQP or SQL) in {int(square)} of {len(part)} pictures; best matches by kind: "
              f"{part['best_kind'].value_counts().to_dict()}; by lay-out: {part['best_layout'].value_counts().to_dict()}; "
              f"median best correlation {part['best_correlation'].median():.3f}")
    # how alike are two demonstrations of one lay-out? the best match of each SQP first-contact picture among the other episodes
    square_frames = frames[(frames["start_kind"] == "SQP") & (frames["row_in_episode"] <= 8)]
    others = []
    for position in numpy.flatnonzero(((frames["start_kind"] == "SQP") & (frames["row_in_episode"] <= 8)).to_numpy()):
        similarity = demonstration_vectors[position] @ demonstration_vectors.T
        similarity[(frames["episode_id"] == frames["episode_id"].iloc[position]).to_numpy()] = -1.0
        others.append(float(similarity.max()))
    print(f"For scale: the first 8 rows of the six SQP demonstrations ({len(square_frames)} pictures) match their best picture in another episode at a "
          f"median correlation of {numpy.median(others):.3f} (10th percentile {numpy.percentile(others, 10):.3f}).")

    print("\n=== B. The line's direction where it leaves the puck, on the table (measured in the first picture of each run, fitted camera) ===")
    for number in settings["runs_compared"]:
        run = runs[number]
        steps = robot_run_steps(run, settings)
        picture = cv2.imread(str(Path(settings["model_runs_folder"]) / run["folder"] / "frames" / "000.jpg"))
        first = steps.iloc[0]
        joints = [float(first[f"joint_{joint}"]) for joint in range(1, 7)]
        measured = line_direction_on_table(picture, joints, numpy.array([first["tip_x"], first["tip_y"]]), detector, settings)
        if measured is None:
            print(f"Run {number}: the line could not be measured")
            continue
        print(f"Run {number}: tip starts at ({100 * first['tip_x']:.1f}, {100 * first['tip_y']:.1f}) cm, puck at ({measured['puck_xy_cm'][0]:.1f}, {measured['puck_xy_cm'][1]:.1f}) cm; "
              f"the line leaves the puck at {measured['direction_deg']:+.0f} degrees on the table (0 = straight away, + = to the left; spread {measured['spread_deg']:.0f}, "
              f"{measured['pixels']} pixels), {float(wrapped_deg(measured['direction_deg'] - first['picture_heading_deg'])):+.0f} degrees from the picture's top; joint 4 {first['joint_4_deg']:+.1f}")
    sheet_02 = layouts[(layouts["sheet"] == "02") & ~layouts["start_kind"].isin(["NE"])]
    print("\nFitted line start direction of the sheet-02 demonstrations (calculated in 01_placements.py; degrees on the table):")
    print(sheet_02.groupby(["end", "start_kind"]).agg(episodes = ("plan", "size"), line_start_direction_deg = ("line_start_direction_deg", "median"),
                                                     least = ("line_start_direction_deg", "min"), most = ("line_start_direction_deg", "max"),
                                                     turn_deg = ("turn_deg", "median")).round(1).to_string())

    print("\n=== C. What the model asked for at the start of run 3 against the demonstrations ===")
    print("Nearest demonstration pictures of run 3's first pictures: their labels against what the model asked (measured):")
    print(nearest[nearest["run"] == 3][["picture", "joint_4_deg", "asked_cm", "asked_twist_deg", "label_step_cm_median", "label_twist_deg_median",
                                        "neighbour_joint_4_deg_median", "neighbour_layouts"]].to_string(index = False))
    reference = settings["twist_reference_deg"]
    rows = []
    for episode_id, group in frames[frames["start_kind"] == "SQP"].groupby("episode_id", sort = False):
        summary = start_summary(group["joint_4_deg"].to_numpy(), 100.0 * group[["tip_x_m", "tip_y_m"]].to_numpy(), group["step_cm"].to_numpy(), reference)
        summary.update({"what": f"demonstration {group['plan_number'].iloc[0]}"})
        rows.append(summary)
    for number in settings["runs_compared"]:
        steps = robot_run_steps(runs[number], settings)
        asked = numpy.hypot(steps["model_tip_forward_cm"], steps["model_tip_left_cm"]).to_numpy()
        summary = start_summary(steps["joint_4_deg"].to_numpy(), 100.0 * steps[["tip_x", "tip_y"]].to_numpy(), numpy.full(len(steps), 1.0), reference)
        summary.update({"what": f"robot run {number}", "median_asked_step_first_10_cm": float(numpy.median(asked[:10])),
                        "median_asked_twist_first_10_deg": float(numpy.median(steps["model_twist_deg"].to_numpy()[:10]))})
        rows.append(summary)
    table = pandas.DataFrame(rows).set_index("what").round(2)
    print(f"\nHow the forearm was turned at the start (rows and tip travel until joint 4 had turned {reference:.0f} degrees; measured):")
    print(table.to_string())
    square_start = frames[(frames["start_kind"] == "SQP") & (frames["row_in_episode"] <= 16)]
    twisting = square_start[square_start["forearm_change_deg"].abs() >= 3.0]
    print(f"\nIn the first 16 rows of the six SQP demonstrations, the rows that twist the forearm by 3 degrees or more ({len(twisting)} rows) move the tip a median "
          f"{twisting['step_cm'].median():.2f} cm (75th percentile {twisting['step_cm'].quantile(0.75):.2f}); {100.0 * (twisting['step_cm'] < 0.5).mean():.0f} % of them move it under 0.5 cm. "
          "The driver moved 1.0 cm on every step.")

    sequence_tiles = []
    for plan in ("02-16", "02-13"):
        group = frames[frames["plan_number"] == plan]
        row_tiles = []
        for row_number in (1, 3, 5, 7, 9, 11, 13, 16):
            match = group[group["row_in_episode"] == row_number]
            if len(match) == 0:
                continue
            tile = cv2.resize(cv2.imread(str(Path(settings["raw_folder"]) / match["frame_path"].iloc[0])), (240, 180))
            cv2.putText(tile, f"{plan} row {row_number} j4 {match['joint_4_deg'].iloc[0]:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            row_tiles.append(tile)
        sequence_tiles.append(numpy.hstack(row_tiles))
    run = runs[3]
    steps = robot_run_steps(run, settings)
    row_tiles = []
    for index in (0, 2, 4, 6, 8, 10, 12, 15):
        tile = cv2.resize(cv2.imread(str(Path(settings["model_runs_folder"]) / run["folder"] / "frames" / f"{index:03d}.jpg")), (240, 180))
        cv2.putText(tile, f"run 3 picture {index} j4 {steps['joint_4_deg'].iloc[index]:.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        row_tiles.append(tile)
    sequence_tiles.append(numpy.hstack(row_tiles))
    cv2.imwrite(str(EVIDENCE_FOLDER / "03_sqp_start_sequences.png"), numpy.vstack(sequence_tiles))

    print("\n=== D. How far the puck lay ahead of the tip at the start (measured: puck detector and fitted camera, first kept picture) ===")
    gaps = start_gaps(frames, detector, settings)
    gaps.to_csv(EVIDENCE_FOLDER / "03_start_gaps.csv", index = False)
    touching = settings["tip_behind_puck_cm"]
    demonstrations = gaps[(gaps["what"] == "demonstration") & ~gaps["start_kind"].isin(["NE", "MID", "X1", "X2"])]
    found = demonstrations.dropna(subset = ["puck_from_tip_cm"])
    print(f"A puck that the block touches lies about {touching} cm from the tip (04_physics.py, part C). Start-mark demonstrations: puck found in "
          f"{len(found)} of {len(demonstrations)} first pictures; distance median {found['puck_from_tip_cm'].median():.1f} cm, 10th to 90th percentile "
          f"{found['puck_from_tip_cm'].quantile(0.1):.1f} to {found['puck_from_tip_cm'].quantile(0.9):.1f}; within 1 cm of touching (under {touching + 1.0} cm): "
          f"{int((found['puck_from_tip_cm'] < touching + 1.0).sum())} episodes; more than 3 cm of free travel (over {touching + 3.0} cm): "
          f"{int((found['puck_from_tip_cm'] > touching + 3.0).sum())}")
    print(found.groupby("sheet")["puck_from_tip_cm"].describe()[["count", "min", "50%", "max"]].round(1).to_string())
    print("Sheet 02 by start kind:")
    print(found[found["sheet"] == "02"].groupby("start_kind")["puck_from_tip_cm"].describe()[["count", "min", "50%", "max"]].round(1).to_string())
    taught = gaps[(gaps["what"] == "demonstration") & gaps["start_kind"].isin(["NE", "MID", "X1", "X2"])].dropna(subset = ["puck_from_tip_cm"])
    print(f"Taught starts (NE, MID, X): median {taught['puck_from_tip_cm'].median():.1f} cm, 10th to 90th percentile "
          f"{taught['puck_from_tip_cm'].quantile(0.1):.1f} to {taught['puck_from_tip_cm'].quantile(0.9):.1f} ({len(taught)} episodes)")
    plain = found[~found["start_kind"].str.startswith(("OL", "OR"))].copy()
    plain["cap_x"] = plain["episode_id"].map(layouts["start_cap_x_cm"])
    plain["cap_y"] = plain["episode_id"].map(layouts["start_cap_y_cm"])
    apart = numpy.hypot(plain["puck_x_cm"] - plain["cap_x"], plain["puck_y_cm"] - plain["cap_y"])
    print(f"A check of the lay-out fits of 01_placements.py: in the {len(plain)} start-mark episodes whose puck was laid on the start cap (no offset start), the "
          f"puck seen in the first picture lies a median {apart.median():.1f} cm from the fitted start cap (90th percentile {apart.quantile(0.9):.1f} cm).")
    print("The ten robot runs:")
    print(gaps[gaps["what"] == "robot run"][["name", "sheet", "start_kind", "puck_from_tip_cm"]].round(1).to_string(index = False))
    print("\nSaved 03_run3_nearest.csv, 03_run3_nearest.png, 03_sqp_start_sequences.png, 03_start_gaps.csv")



if __name__ == "__main__":
    sys.stdout.reconfigure(encoding = "utf-8")
    main()
