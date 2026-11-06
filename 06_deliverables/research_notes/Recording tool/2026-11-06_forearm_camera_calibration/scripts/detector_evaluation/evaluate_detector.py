"""
Run the puck detector on every picture of episodes 037 to 045 and build contact
sheets with the detections drawn, for checking by eye.

Usage:
    python evaluate_detector.py run TAG
    python evaluate_detector.py montage_all TAG
    python evaluate_detector.py montage_list TAG NAME KEY [KEY ...]    (KEY: 037/012 or 043/camera_check)
    python evaluate_detector.py montage_candidates TAG
"""

import os
import sys
import json
import time
import statistics

import cv2
import numpy

from frame_catalogue import catalogue_pictures
from puck_detector import PuckDetector


OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))

REJECTION_ABBREVIATIONS = {
    "area_small": "as",
    "area_large": "al",
    "seed_fraction": "sf",
    "solidity": "so",
    "axis_ratio": "ar",
    "extent": "ex",
    "wood_around": "wd",
    "low_score": "ls",
    "pale": "pa",
    "no_light_around": "nl",
    "size": "sz"
}


def picture_key(entry: dict) -> str:
    """Short key of a picture: ``037/012`` for a frame, ``043/camera_check`` otherwise."""

    if entry["frame_index"] is None:
        return f"{entry['episode']}/{entry['kind']}"
    return f"{entry['episode']}/{entry['frame_index']:03d}"


def run_folder(tag: str) -> str:

    folder = os.path.join(OUTPUT_FOLDER, "runs", tag)
    os.makedirs(folder, exist_ok = True)
    return folder


def run_detector(tag: str):
    """
    Detect in every picture, time each call and save detections and candidates.

    Parameters
    ----------
    tag : str
        Name of the run; results go to ``runs/TAG/results.json``.
    """

    # one OpenCV thread: on 320 x 240 masks the thread start-up costs more than it saves
    # (measured 2.1 ms per picture with one thread against 2.7 ms with twelve)
    cv2.setNumThreads(1)
    detector = PuckDetector()
    pictures = catalogue_pictures()
    results = []
    durations_ms = []

    for entry in pictures:
        picture = cv2.imread(entry["path"])
        candidates, blue_cast, cast_share = detector.find_candidates(picture)
        # the timed figure is one ``detect`` call; ``find_candidates`` above is for the record only
        start = time.perf_counter()
        detection = detector.detect(picture)
        end = time.perf_counter()
        durations_ms.append(1000.0 * (end - start))
        results.append({
            **entry,
            "key": picture_key(entry),
            "blue_cast": blue_cast,
            "cast_share": cast_share,
            "detection": detection,
            "candidates": [
                {name: value for name, value in candidate.items() if name != "label"}
                for candidate in candidates
            ],
            "detect_ms": round(1000.0 * (end - start), 3)
        })

    with open(os.path.join(run_folder(tag), "results.json"), "w") as results_file:
        json.dump(results, results_file, indent = 1)

    print(f"pictures {len(results)}, detected {sum(result['detection'] is not None for result in results)}")
    for episode in sorted({result["episode"] for result in results}):
        episode_frames = [result for result in results if result["episode"] == episode and result["kind"] == "frame"]
        detected = sum(result["detection"] is not None for result in episode_frames)
        print(f"   episode {episode}: frames {len(episode_frames)}, detected {detected}"
              + (f" ({100 * detected / len(episode_frames):.1f} %)" if episode_frames else ""))
    for result in results:
        if result["kind"] != "frame":
            print(f"   {result['key']}: {'detected' if result['detection'] else 'none'}")
    print(f"detect time per picture: mean {statistics.mean(durations_ms):.2f} ms, median "
          f"{statistics.median(durations_ms):.2f} ms, 95th percentile "
          f"{numpy.percentile(durations_ms, 95):.2f} ms, max {max(durations_ms):.2f} ms")


def load_results(tag: str) -> list:

    with open(os.path.join(run_folder(tag), "results.json")) as results_file:
        return json.load(results_file)


def draw_result(
        picture: numpy.ndarray,
        result: dict,
        line_scale: int = 1,
        is_showing_rejected: bool = True
) -> numpy.ndarray:
    """
    Draw the detection (green circle, centre cross, yellow bottom point) and the
    rejected candidates (thin red boxes with a reason code) onto a copy of the picture.

    Parameters
    ----------
    picture : numpy.ndarray
        The picture in BGR.
    result : dict
        One entry of ``results.json``.
    line_scale : int, optional
        Line thickness factor, larger for pictures that will be shrunk a lot.
    is_showing_rejected : bool, optional
        Whether rejected candidates of 150 pixels or more are outlined.

    Returns
    -------
    numpy.ndarray
        The annotated copy.
    """

    painted = picture.copy()
    if is_showing_rejected:
        for candidate in result["candidates"]:
            if candidate["rejection_reason"] is None or candidate["area_px"] < 150:
                continue
            left, top, width, height = candidate["bounding_box"]
            cv2.rectangle(painted, (left, top), (left + width, top + height), (0, 0, 255), line_scale)
            cv2.putText(painted, abbreviate(candidate["rejection_reason"]), (left, max(top - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * line_scale, (0, 0, 255), line_scale, cv2.LINE_AA)
    detection = result["detection"]
    if detection is not None:
        centre = (int(round(detection["centre_u"])), int(round(detection["centre_v"])))
        # whole puck: green circle of the blob's area; partly hidden puck with a good circle fit:
        # the fitted circle in cyan and the visible blob's centre as a small green dot; partly
        # hidden without a good fit: the blob's circle in magenta
        if detection.get("centre_source") == "circle_fit":
            radius = int(round(detection["circle_radius_px"]))
            colour = (255, 255, 0)
            visible = (int(round(detection["visible_centre_u"])), int(round(detection["visible_centre_v"])))
            cv2.circle(painted, visible, 2 * line_scale, (0, 255, 0), -1)
        else:
            radius = int(round(detection["equivalent_radius_px"]))
            colour = (255, 0, 255) if detection.get("is_partial") else (0, 255, 0)
        cv2.circle(painted, centre, radius, colour, 2 * line_scale)
        cv2.drawMarker(painted, centre, colour, cv2.MARKER_CROSS, 10 * line_scale, 2 * line_scale)
        bottom = (int(round(detection["bottom_u"])), int(round(detection["bottom_v"])))
        bottom_colour = (0, 255, 255) if detection.get("is_bottom_clear") else (0, 128, 255)
        cv2.circle(painted, bottom, 3 * line_scale, bottom_colour, -1)
    return painted


def abbreviate(reason: str) -> str:
    """Short code of a rejection reason for drawing: known codes, else the first letters of each part."""

    parts = []
    for part in reason.split("+"):
        if part in REJECTION_ABBREVIATIONS:
            parts.append(REJECTION_ABBREVIATIONS[part])
        else:
            parts.append("".join(word[0] for word in part.split("_")))
    return "+".join(parts)


def build_sheet(
        thumbnails: list,
        labels: list,
        label_colours: list,
        columns: int
) -> numpy.ndarray:
    """
    Place equally sized thumbnails in a grid with a caption in each top-left corner.

    Parameters
    ----------
    thumbnails : list of numpy.ndarray
        Pictures of one size.
    labels : list of str
        Captions.
    label_colours : list of tuple
        BGR colour of each caption.
    columns : int
        Thumbnails per row.

    Returns
    -------
    numpy.ndarray
        The sheet.
    """

    thumbnail_height, thumbnail_width = thumbnails[0].shape[:2]
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = numpy.full((rows * thumbnail_height, columns * thumbnail_width, 3), 255, dtype = numpy.uint8)
    for index, (thumbnail, label, colour) in enumerate(zip(thumbnails, labels, label_colours)):
        thumbnail = thumbnail.copy()
        cv2.rectangle(thumbnail, (0, 0), (thumbnail_width - 1, thumbnail_height - 1), (0, 0, 0), 1)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(thumbnail, (0, 0), (text_width + 6, text_height + 8), (0, 0, 0), -1)
        cv2.putText(thumbnail, label, (3, text_height + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        row = index // columns
        column = index % columns
        sheet[row * thumbnail_height:(row + 1) * thumbnail_height,
              column * thumbnail_width:(column + 1) * thumbnail_width] = thumbnail
    return sheet


def label_for(result: dict) -> tuple:

    detection = result["detection"]
    if detection is None:
        return f"{result['key']} NONE", (80, 80, 255)
    return f"{result['key']} {detection['confidence']:.2f}", (0, 255, 255)


def montage_results(
        results: list,
        output_folder: str,
        name: str,
        columns: int,
        thumbnail_width: int,
        per_sheet: int,
        line_scale: int
):
    """
    Save annotated contact sheets of the given results.

    Parameters
    ----------
    results : list of dict
        Entries of ``results.json`` to show, in order.
    output_folder : str
        Folder for the sheets.
    name : str
        File name stem; sheets are numbered.
    columns, thumbnail_width, per_sheet, line_scale : int
        Grid layout, thumbnail size, thumbnails per sheet and line thickness factor.
    """

    os.makedirs(output_folder, exist_ok = True)
    thumbnail_height = thumbnail_width * 3 // 4
    for sheet_index in range(0, len(results), per_sheet):
        chunk = results[sheet_index:sheet_index + per_sheet]
        thumbnails = []
        labels = []
        colours = []
        for result in chunk:
            painted = draw_result(cv2.imread(result["path"]), result, line_scale = line_scale)
            thumbnails.append(cv2.resize(painted, (thumbnail_width, thumbnail_height), interpolation = cv2.INTER_AREA))
            label, colour = label_for(result)
            labels.append(label)
            colours.append(colour)
        sheet = build_sheet(thumbnails, labels, colours, columns)
        cv2.imwrite(os.path.join(output_folder, f"{name}_{sheet_index // per_sheet:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])


def montage_candidate_crops(results: list, output_folder: str):
    """
    Save sheets of crops around every rejected candidate of 150 pixels or more, with
    the reason and the main measurements, to look for pucks that were thrown away.

    Parameters
    ----------
    results : list of dict
        Entries of ``results.json``.
    output_folder : str
        Folder for the sheets.
    """

    os.makedirs(output_folder, exist_ok = True)
    crop_size = 160
    crops = []
    labels = []
    for result in results:
        rejected = [candidate for candidate in result["candidates"]
                    if candidate["rejection_reason"] is not None and candidate["area_px"] >= 150]
        if not rejected:
            continue
        picture = None
        for candidate in rejected:
            if picture is None:
                picture = cv2.imread(result["path"])
            left, top, width, height = candidate["bounding_box"]
            centre_u = left + width // 2
            centre_v = top + height // 2
            half = max(width, height, 60) // 2 + 15
            padded = cv2.copyMakeBorder(picture, half, half, half, half, cv2.BORDER_CONSTANT, value = (128, 128, 128))
            crop = padded[centre_v:centre_v + 2 * half, centre_u:centre_u + 2 * half].copy()
            cv2.rectangle(crop, (half - width // 2, half - height // 2), (half - width // 2 + width,
                          half - height // 2 + height), (0, 0, 255), 1)
            crops.append(cv2.resize(crop, (crop_size, crop_size), interpolation = cv2.INTER_AREA))
            labels.append(f"{result['key']} {abbreviate(candidate['rejection_reason'])} a{candidate['area_px']}")
    per_sheet = 48
    for sheet_index in range(0, len(crops), per_sheet):
        sheet = build_sheet(crops[sheet_index:sheet_index + per_sheet], labels[sheet_index:sheet_index + per_sheet],
                            [(0, 255, 255)] * per_sheet, columns = 8)
        cv2.imwrite(os.path.join(output_folder, f"rejected_{sheet_index // per_sheet:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(crops)} rejected candidates of 150 px or more on {(len(crops) + per_sheet - 1) // per_sheet} sheets")


def main():

    command = sys.argv[1]
    tag = sys.argv[2]

    if command == "run":
        run_detector(tag)
        return

    results = load_results(tag)
    if command == "montage_all":
        frames = [result for result in results]
        montage_results(frames, os.path.join(run_folder(tag), "all"), "all", columns = 6, thumbnail_width = 213,
                        per_sheet = 36, line_scale = 2)
    elif command == "montage_list":
        name = sys.argv[3]
        keys = sys.argv[4:]
        by_key = {result["key"]: result for result in results}
        chosen = [by_key[key] for key in keys if key in by_key]
        montage_results(chosen, os.path.join(run_folder(tag), name), name, columns = 4, thumbnail_width = 320,
                        per_sheet = 16, line_scale = 1)
    elif command == "montage_candidates":
        montage_candidate_crops(results, os.path.join(run_folder(tag), "rejected"))
    elif command == "montage_misses":
        misses = [result for result in results if result["detection"] is None and result["episode"] != "039"]
        print(f"{len(misses)} pictures without a detection outside episode 039")
        montage_results(misses, os.path.join(run_folder(tag), "misses"), "misses", columns = 5, thumbnail_width = 256,
                        per_sheet = 30, line_scale = 2)


if __name__ == "__main__":
    main()
