"""
How far the picture measures can be trusted: three checks and two sheets of pictures to look at.

    1. the puck's size: the outline fit is repeated with other radii; the radius that fits the red blob best should be
       the measured 1.2 cm (the brief of 11.11.26 says "a red puck (4 cm)", config.py says 2.4 cm across);
    2. the camera model: the same piece of line seen in two pictures 6 and 12 steps apart should land on the same place
       of the table, with the camera as fitted on 6.11.26 and with the shift and roll of camera_shift.json;
    3. overlays: the fitted puck (its outline green, its foot on the table cyan), the line as the run's pictures voted
       it (yellow) and the centre of the line's end (magenta cross) drawn into pictures of the robot's runs and of
       demonstrations, to be looked at.

Run from this folder, after camera_shift.py:    python measure_check.py
Writes overlays_robot_runs.png and overlays_demonstrations.png; the printed numbers are kept in measure_check.out.txt.
Nothing connects to the robot.
"""

from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy
import pandas

import demonstration_ends
import picture_measures
import run_measures

from config import MODEL_MODE_PARAMETERS, TRAINING_DATA_PARAMETERS
from puck_detector import PuckDetector


MEASURE_CHECK_PARAMETERS = {
    "radius_runs": ["2026-11-11_172421_E_sheet01_run1", "2026-11-11_173036_E_sheet02_run2"],
    "radius_every": 3,
    "radii_m": [0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.020],
    "registration_runs": ["2026-11-11_173036_E_sheet02_run2", "2026-11-11_174448_E_sheet05_run1"],
    "registration_steps_apart": [6, 12],
    # the pictures drawn: for each robot run some steps (-1 is the stop picture), for each demonstration steps counted
    # back from the stop
    "overlay_robot": [("2026-11-11_172021_check_no_puck", [12, 20, -1]), ("2026-11-11_172421_E_sheet01_run1", [12, 20, -1]),
                      ("2026-11-11_173036_E_sheet02_run2", [20, 26, -1]), ("2026-11-11_174448_E_sheet05_run1", [30, 50, -1])],
    "overlay_demonstrations": [("path_01_straight_line_episode_006_plan_01-04_try_1", [6, 3, 0]),
                               ("path_02_arc_episode_001_plan_02-01_try_1", [6, 3, 0]),
                               ("path_03_two_curve_episode_004_plan_03-04_try_1", [6, 3, 0]),
                               ("path_05_alpha_loop_episode_003_plan_05-01_try_3", [6, 3, 0])],
    "crop_px": 300,
    "zoom": 1.4
}



def radius_check(detector: PuckDetector) -> None:
    """Fit the puck's outline with several radii and print which radius covers the red blob best."""

    camera_parameters = run_measures.camera_parameters_for("robot_test")
    costs: List[List[float]] = []
    for name in MEASURE_CHECK_PARAMETERS["radius_runs"]:
        steps = run_measures.robot_run_steps(Path(MODEL_MODE_PARAMETERS["run_folder"]) / name)
        for index in range(2, len(steps), MEASURE_CHECK_PARAMETERS["radius_every"]):
            row = steps.iloc[index]
            picture = cv2.imread(row["picture_path"])
            blob = picture_measures.puck_blob(picture, detector)
            if blob is None:
                continue
            camera = picture_measures.TableCamera([row[column] for column in run_measures.JOINT_COLUMNS], camera_parameters)
            costs.append([picture_measures.fit_puck_on_table(blob, camera, radius_m = radius)["cost"] for radius in MEASURE_CHECK_PARAMETERS["radii_m"]])
    costs_table = numpy.array(costs)
    best = numpy.argmin(costs_table, axis = 1)
    print(f"check 1, the puck's size ({len(costs_table)} pictures of the robot's runs 2 and 4):")
    print("  radius cm      " + "".join(f"{100.0 * radius:7.1f}" for radius in MEASURE_CHECK_PARAMETERS["radii_m"]))
    print("  median misfit  " + "".join(f"{value:7.0f}" for value in numpy.median(costs_table, axis = 0)) + "   (pixels of blob and outline that disagree, weighted)")
    print("  best in        " + "".join(f"{int((best == position).sum()):7d}" for position in range(len(MEASURE_CHECK_PARAMETERS["radii_m"]))) + "   pictures")



def registration_check(detector: PuckDetector) -> None:
    """Print how far apart pictures 6 and 12 steps apart put the same piece of line, with the fitted and the corrected camera."""

    print("\ncheck 2, the camera model: the same piece of line seen in two pictures of a run")
    for name in MEASURE_CHECK_PARAMETERS["registration_runs"]:
        steps = run_measures.robot_run_steps(Path(MODEL_MODE_PARAMETERS["run_folder"]) / name)
        views = run_measures.loaded_views(steps, 1, detector)
        x_range, y_range = run_measures.map_ranges(steps)
        for label, camera_parameters in (("as fitted", None), ("corrected", run_measures.camera_parameters_for("robot_test"))):
            for apart in MEASURE_CHECK_PARAMETERS["registration_steps_apart"]:
                pairs = run_measures.view_registration(views, x_range, y_range, apart, camera_parameters)
                if pairs.empty:
                    continue
                turned = pairs[pairs["heading_change_deg"].abs() >= 10.0]
                print(f"  {name[11:]}, {label:9s}, {apart:2d} steps apart: {len(pairs):3d} pairs, disagreement median {pairs['shift_mm'].median():.2f} mm, "
                      f"90th percentile {pairs['shift_mm'].quantile(0.9):.2f} mm; forearm turned 10 degrees or more between the two: "
                      f"{len(turned)} pairs, median {turned['shift_mm'].median():.2f} mm" if len(turned) else
                      f"  {name[11:]}, {label:9s}, {apart:2d} steps apart: {len(pairs):3d} pairs, disagreement median {pairs['shift_mm'].median():.2f} mm, "
                      f"90th percentile {pairs['shift_mm'].quantile(0.9):.2f} mm")



def drawn_picture(
        picture: numpy.ndarray,
        camera: picture_measures.TableCamera,
        line_map: picture_measures.LineMap,
        is_line: numpy.ndarray,
        end: Dict,
        detector: PuckDetector,
        label: str
) -> numpy.ndarray:
    """One picture with the line map, the fitted puck and the end's centre drawn in, cropped around the puck."""

    shown = picture.copy()
    contours, _ = cv2.findContours(is_line.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if len(contour) < 20:
            continue
        cells = contour.reshape(-1, 2)[:, ::-1].astype(float)
        pixels = camera.project_table(line_map.point_of(cells))
        if numpy.all(numpy.isfinite(pixels)):
            cv2.polylines(shown, [numpy.round(pixels).astype(numpy.int32)], True, (0, 255, 255), 1)
    centre = numpy.array([camera.picture_width / 2.0, camera.picture_height / 2.0 + 60.0])
    blob = picture_measures.puck_blob(picture, detector)
    if blob is not None:
        fit = picture_measures.fit_puck_on_table(blob, camera)
        cv2.polylines(shown, [numpy.round(fit["outline_px"]).astype(numpy.int32)], True, (0, 255, 0), 1)
        angles = numpy.linspace(0.0, 2.0 * numpy.pi, 60)
        foot = fit["base_xy"][None, :] + picture_measures.MEASURE_PARAMETERS["puck_radius_m"] * numpy.column_stack([numpy.cos(angles), numpy.sin(angles)])
        cv2.polylines(shown, [numpy.round(camera.project_table(foot)).astype(numpy.int32)], True, (255, 255, 0), 1)
        centre = fit["base_px"]
    if end is not None:
        cross = camera.project_table(end["end_centre_xy"][None, :])[0]
        if numpy.all(numpy.isfinite(cross)):
            cv2.drawMarker(shown, tuple(numpy.round(cross).astype(int)), (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
    size = MEASURE_CHECK_PARAMETERS["crop_px"]
    left = int(min(max(centre[0] - size / 2, 0), shown.shape[1] - size))
    top = int(min(max(centre[1] - size * 0.6, 0), shown.shape[0] - size))
    crop = cv2.resize(shown[top:top + size, left:left + size], None, fx = MEASURE_CHECK_PARAMETERS["zoom"], fy = MEASURE_CHECK_PARAMETERS["zoom"],
                      interpolation = cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, 22), (40, 40, 40), -1)
    cv2.putText(crop, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return crop



def overlay_sheet(
        entries: Sequence,
        is_robot: bool,
        detector: PuckDetector,
        file_name: str
) -> None:
    """Draw the chosen pictures of the chosen runs, one run per row, into one picture file."""

    frames = None if is_robot else pandas.read_csv(Path(TRAINING_DATA_PARAMETERS["processed_folder"]) / "frames.csv", low_memory = False)
    sheet_rows = []
    for name, chosen in entries:
        if is_robot:
            steps = run_measures.robot_run_steps(Path(MODEL_MODE_PARAMETERS["run_folder"]) / name)
            camera_parameters = run_measures.camera_parameters_for("robot_test")
            indices = [len(steps) - 1 if step == -1 else step for step in chosen]
        else:
            episode_id = [episode for episode in frames["episode_id"].unique() if episode.endswith(name)][0]
            steps = run_measures.demonstration_steps(frames, episode_id)
            camera_parameters = run_measures.camera_parameters_for(run_measures.camera_period(steps.iloc[0]["started"]))
            indices = [len(steps) - 1 - back for back in chosen]
        measured = run_measures.measure_steps(steps, camera_parameters, is_offset_measured = False, detector = detector)
        line_votes = measured["all_line_votes"].astype(numpy.float32)
        is_line = (line_votes >= 2) & (line_votes >= 0.5 * (line_votes + measured["all_paper_votes"]))
        end = measured["end"]
        if not is_robot:
            # the end as demonstration_ends.py measures it: from the pictures that showed the end's place
            used_steps = demonstration_ends.pictures_showing_end(steps, camera_parameters)
            end = run_measures.measure_steps(steps, camera_parameters, parameters = {"end_window_pictures": len(used_steps)},
                                             is_offset_measured = False, detector = detector, used_steps = used_steps)["end"]
        tiles = []
        for index in indices:
            row = steps.iloc[index]
            camera = picture_measures.TableCamera([row[column] for column in run_measures.JOINT_COLUMNS], camera_parameters)
            label = f"{name[11:] if is_robot else name.split('_plan_')[-1][:5]}  picture {index}" + ("  (stop)" if index == len(steps) - 1 else "")
            tiles.append(drawn_picture(cv2.imread(row["picture_path"]), camera, measured["line_map"], is_line, end, detector, label))
        sheet_rows.append(numpy.hstack(tiles))
    cv2.imwrite(str(run_measures.EVIDENCE_FOLDER / file_name), numpy.vstack(sheet_rows))
    print(f"written: {file_name}")



def main() -> None:

    cv2.setNumThreads(run_measures.RUN_MEASURE_PARAMETERS["opencv_threads"])
    detector = PuckDetector()
    radius_check(detector)
    registration_check(detector)
    print("\ncheck 3, overlays (yellow: the line as all pictures of the run voted it; green: the fitted puck; cyan: its foot on the table; magenta cross: the centre of the line's end)")
    overlay_sheet(MEASURE_CHECK_PARAMETERS["overlay_robot"], True, detector, "overlays_robot_runs.png")
    overlay_sheet(MEASURE_CHECK_PARAMETERS["overlay_demonstrations"], False, detector, "overlays_demonstrations.png")



if __name__ == "__main__":
    main()
