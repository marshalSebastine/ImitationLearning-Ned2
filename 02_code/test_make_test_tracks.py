"""
Tests of make_test_tracks.py and of the pages it wrote: true size, a face with the line only, the two print files in
their order, line width, margins, self-distance, the list against the drawings, a given centre line, the dashes and the
lay-out check. No robot is used.

Run from 02_code, after python make_test_tracks.py (or "pilot" for T01 to T04 only):
    python test_make_test_tracks.py
"""

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List

import fitz
import numpy

from config import TEST_TRACK_PARAMETERS
from make_test_tracks import (arc_lengths_mm, course_sheet_lines, dash_intervals, feature_names, ink_paths, layout_check,
                              lengthened_curve, line_colour_rgb, line_summary, placed_marks, resampled_curve, write_page_pdf)


FOLDER = Path(TEST_TRACK_PARAMETERS["new_output_folder"])
RULE = TEST_TRACK_PARAMETERS["drawing_rule"]
MILLIMETRES_PER_POINT = 25.4 / 72.0
PAGE_TOLERANCE_MM = 0.01



def listed_tracks() -> List[Dict]:
    """The tracks of tracks.json, their centre lines as arrays."""

    tracks = json.loads((FOLDER / "tracks.json").read_text(encoding = "utf-8"))["tracks"]
    for track in tracks:
        track["curve"] = numpy.array(track["centre_line_mm"])
    return tracks



def page_file(track: Dict, style: str) -> Path:
    """The PDF page of ``track`` in ``style`` ("solid" or "dashed")."""

    if track["role"] == "diagnostic":
        return FOLDER / "diagnostic" / (f"{track['name']}_page_1.pdf" if track["name"] == "C03_circles" else f"{track['name']}.pdf")
    return FOLDER / track["role"] / f"{track['name']}_{style}.pdf"



def drawn_paths(path: Path) -> List[Dict]:
    """
    Every stroked path of a one-page PDF, its points in millimetres from the paper's left and bottom edges.

    Returns
    -------
    list of dict
        Per path: "points" (array), "width_mm", "colour" and "rect_mm" (left, bottom, right, top).
    """

    document = fitz.open(str(path))
    assert document.page_count == 1, f"{path.name}: {document.page_count} pages"
    page = document[0]
    height = page.rect.height
    found = []
    for drawing in page.get_drawings():
        points = []
        for item in drawing["items"]:
            for point in item[1:]:
                if isinstance(point, fitz.Point):
                    points.append([point.x * MILLIMETRES_PER_POINT, (height - point.y) * MILLIMETRES_PER_POINT])
        rectangle = drawing["rect"]
        found.append({"points": numpy.array(points), "width_mm": (drawing.get("width") or 0.0) * MILLIMETRES_PER_POINT,
                      "colour": tuple(drawing.get("color") or ()), "kinds": [item[0] for item in drawing["items"]],
                      "rect_mm": (rectangle.x0 * MILLIMETRES_PER_POINT, (height - rectangle.y1) * MILLIMETRES_PER_POINT,
                                  rectangle.x1 * MILLIMETRES_PER_POINT, (height - rectangle.y0) * MILLIMETRES_PER_POINT)})
    return found



def ink_of(paths: List[Dict]) -> List[Dict]:
    """The paths in the line's colour ("line_colour_hex") and wider than 5 mm: the line itself."""

    colour = line_colour_rgb()
    return [path for path in paths if len(path["colour"]) == 3 and max(abs(a - b) for a, b in zip(path["colour"], colour)) < 0.002
            and path["width_mm"] > 5.0]



def chained(paths: List[Dict]) -> numpy.ndarray:
    """The points of a solid line's one path, consecutive duplicates removed."""

    points = paths[0]["points"]
    keep = numpy.r_[True, numpy.linalg.norm(numpy.diff(points, axis = 0), axis = 1) > 1e-9]
    return points[keep]



def page_size_mm(path: Path) -> tuple:
    """Width and height of a PDF's first page in millimetres."""

    page = fitz.open(str(path))[0]
    return page.rect.width * MILLIMETRES_PER_POINT, page.rect.height * MILLIMETRES_PER_POINT



def bar_lengths_mm(paths: List[Dict], is_across: bool) -> List[float]:
    """The lengths of the single thin straight lines across (or down) the page."""

    lengths = []
    for path in paths:
        if path["kinds"] != ["l"] or path["width_mm"] > 1.0:
            continue
        start, end = path["points"]
        if is_across and abs(start[1] - end[1]) < 1e-6:
            lengths.append(abs(end[0] - start[0]))
        if not is_across and abs(start[0] - end[0]) < 1e-6:
            lengths.append(abs(end[1] - start[1]))
    return lengths



def distance_to_polyline(points: numpy.ndarray, polyline: numpy.ndarray) -> numpy.ndarray:
    """The distance from every point to the nearest segment of ``polyline``."""

    starts, ends = polyline[:-1], polyline[1:]
    steps = ends - starts
    squared = numpy.maximum(numpy.sum(steps ** 2, axis = 1), 1e-12)
    offsets = points[:, None, :] - starts[None, :, :]
    fraction = numpy.clip(numpy.sum(offsets * steps[None, :, :], axis = 2) / squared[None, :], 0.0, 1.0)
    nearest = starts[None, :, :] + fraction[:, :, None] * steps[None, :, :]
    return numpy.linalg.norm(points[:, None, :] - nearest, axis = 2).min(axis = 1)



def test_printer_page_is_true_size() -> None:
    """The print-me-first page is A4, its two bars are 100 mm, its thick line is 10 mm wide."""

    path = FOLDER / "print_me_first_100mm_bar.pdf"
    width, height = page_size_mm(path)
    assert abs(width - 210.0) < PAGE_TOLERANCE_MM and abs(height - 297.0) < PAGE_TOLERANCE_MM, (width, height)
    paths = drawn_paths(path)
    assert any(abs(length - 100.0) < 0.01 for length in bar_lengths_mm(paths, True)), "no 100 mm bar across"
    assert any(abs(length - 100.0) < 0.01 for length in bar_lengths_mm(paths, False)), "no 100 mm bar down"
    assert any(abs(path["width_mm"] - 10.0) < 0.01 for path in paths), "no 10 mm wide line"



def face_pages() -> List[tuple]:
    """Every track page's file with its track and style; the circle pages of sheet 03 both."""

    pages = []
    for track in listed_tracks():
        styles = ["solid", "dashed"] if track["role"] != "diagnostic" else ["dashed" if track["dash_mm"] is not None else "solid"]
        pages += [(track, style, page_file(track, style)) for style in styles]
        if track["name"] == "C03_circles":
            pages.append((track, "solid", FOLDER / "diagnostic" / "C03_circles_page_2.pdf"))
    return pages



def test_every_face_holds_the_line_only() -> None:
    """
    Every track page: one A4 page whose face holds nothing but strokes 10 mm wide in the line colour (on the two circle pages also
    the listed grey circles of 24 mm): no text, no picture, no other drawing, nothing past the paper.
    """

    places = json.loads((FOLDER / "tracks.json").read_text(encoding = "utf-8")).get("diagnostic_places") or []
    checked = 0
    for track, style, path in face_pages():
        width, height = page_size_mm(path)
        assert abs(width - 210.0) < PAGE_TOLERANCE_MM and abs(height - 297.0) < PAGE_TOLERANCE_MM, (path.name, width, height)
        page = fitz.open(str(path))[0]
        assert page.get_text().strip() == "", f"{path.name}: text on the face: {page.get_text()!r}"
        assert not page.get_images(), f"{path.name}: a picture on the face"
        paths = drawn_paths(path)
        ink = ink_of(paths)
        others = [drawn for drawn in paths if not any(drawn is line for line in ink)]
        assert ink and all(abs(drawn["width_mm"] - 10.0) < 0.01 for drawn in ink), f"{path.name}: line not 10 mm wide"
        if path.name.startswith("C03_circles_page_"):
            number = int(path.stem[-1])
            listed = [place for place in places if place["page"] == number]
            assert len(others) == len(listed) == 10, (path.name, len(others), len(listed))
            for drawn in others:
                left, bottom, right, top = drawn["rect_mm"]
                centre = numpy.array([(left + right) / 2.0, (bottom + top) / 2.0])
                assert abs(right - left - 24.0) < 0.05 and abs(top - bottom - 24.0) < 0.05, path.name
                assert min(numpy.linalg.norm(centre - numpy.array(place["centre_mm"])) for place in listed) < 0.1, path.name
                assert abs(drawn["width_mm"] - 0.3) < 0.01 and min(drawn["colour"]) > 0.5, path.name
        else:
            assert not others, f"{path.name}: {len(others)} drawings besides the line"
        for drawn in paths:
            left, bottom, right, top = drawn["rect_mm"]
            assert left >= 0 and bottom >= 0 and right <= 210.0 and top <= 297.0, f"{path.name}: a drawing reaches past the paper"
        for drawn in ink:
            reach = drawn["width_mm"] / 2.0
            points = drawn["points"]
            assert points[:, 0].min() - reach >= 5.0 and points[:, 0].max() + reach <= 205.0, f"{path.name}: ink within 5 mm of a side"
            assert points[:, 1].min() - reach >= 5.0 and points[:, 1].max() + reach <= 292.0, f"{path.name}: ink within 5 mm of an end"
        checked += 1
    assert checked >= 8, checked



def drawing_items(page) -> List[list]:
    """Every drawing item of a page as its kind and its points, rounded to 0.01 point."""

    return [[item[0]] + [(round(point.x, 2), round(point.y, 2)) for point in item[1:] if isinstance(point, fitz.Point)]
            for drawing in page.get_drawings() for item in drawing["items"]]



def test_print_files_hold_the_listed_pages_in_order() -> None:
    """Each print file: the pages of its order list in that order, each 210 x 297 mm, each the same drawing as its source page."""

    files = [name for name in ("print_set_saturday.pdf", "print_set_tests.pdf") if (FOLDER / name).exists()]
    assert files, "no print file"
    for name in files:
        order = json.loads((FOLDER / f"{Path(name).stem}_order.json").read_text(encoding = "utf-8"))
        merged = fitz.open(str(FOLDER / name))
        assert merged.page_count == len(order), (name, merged.page_count, len(order))
        assert [row["page"] for row in order] == list(range(1, len(order) + 1)), name
        assert order[0]["source"] == "print_me_first_100mm_bar.pdf" and order[1]["name"] == "list", name
        list_path = FOLDER / f"{Path(name).stem}_list.pdf"
        list_pages = sum(row["name"] == "list" for row in order)
        assert fitz.open(str(list_path)).page_count == list_pages, name
        for row, page in zip(order, merged):
            assert abs(page.rect.width * MILLIMETRES_PER_POINT - 210.0) < PAGE_TOLERANCE_MM, (name, row)
            assert abs(page.rect.height * MILLIMETRES_PER_POINT - 297.0) < PAGE_TOLERANCE_MM, (name, row)
            if row["name"] == "list":
                source_page = fitz.open(str(list_path))[int(row["source"].split()[-1]) - 1]
                assert "Print single-sided at 100 %" in source_page.get_text() or list_pages > 1, name
            else:
                source_page = fitz.open(str(FOLDER / row["source"]))[0]
                if row["name"] != "printer check":
                    assert Path(row["source"]).stem in (row["name"], f"{row['name']}_{row['style']}"), row
            assert drawing_items(page) == drawing_items(source_page), (name, row)
            assert page.get_text() == source_page.get_text(), (name, row)
        names = [f"{row['name']}_{row['style']}" for row in order[1 + list_pages:]]
        if name == "print_set_saturday.pdf":
            assert names == [f"T{number:02d}_{style}" for number in range(1, 17) for style in ("solid", "dashed")], names
        else:
            expected = [f"D{number:02d}_{style}" for number in range(1, 11) for style in ("solid", "dashed")]
            expected += ["C05_solid_solid", "C05_dashed_dashed", "C06_solid_solid", "C06_dashed_dashed", "C01_solid_solid",
                         "C03_circles_page_1_solid", "C03_circles_page_2_solid"]
            assert names == expected, names



def test_course_sheet_copies() -> None:
    """
    The six course-sheet copies: plain A4 faces (strokes 10 mm wide in the line colour, no text), each drawn centre line within 1 mm of
    the traced one, sheet 06 dashed; the print file holds the printer page, one list page and the faces 01 to 06.
    """

    traced = course_sheet_lines()
    order = json.loads((FOLDER / "print_set_course_sheets_order.json").read_text(encoding = "utf-8"))
    merged = fitz.open(str(FOLDER / "print_set_course_sheets.pdf"))
    assert merged.page_count == 8 == len(order), merged.page_count
    assert [row["name"] for row in order] == ["printer check", "list"] + [f"C0{number}_course" for number in range(1, 7)]
    for row, merged_page in zip(order, merged):
        assert abs(merged_page.rect.width * MILLIMETRES_PER_POINT - 210.0) < PAGE_TOLERANCE_MM, row
        assert abs(merged_page.rect.height * MILLIMETRES_PER_POINT - 297.0) < PAGE_TOLERANCE_MM, row
        if row["name"] == "list":
            source_page = fitz.open(str(FOLDER / "print_set_course_sheets_list.pdf"))[0]
            assert "Write each page's name on its BACK." in source_page.get_text().replace("\n", " ")
        else:
            source_page = fitz.open(str(FOLDER / row["source"]))[0]
        assert drawing_items(merged_page) == drawing_items(source_page), row
    for row in order[2:]:
        path = FOLDER / row["source"]
        sheet = row["name"][1:3]
        width, height = page_size_mm(path)
        assert abs(width - 210.0) < PAGE_TOLERANCE_MM and abs(height - 297.0) < PAGE_TOLERANCE_MM, path.name
        page = fitz.open(str(path))[0]
        assert page.get_text().strip() == "" and not page.get_images(), path.name
        paths = drawn_paths(path)
        ink = ink_of(paths)
        assert len(ink) == len(paths) and all(abs(drawn["width_mm"] - 10.0) < 0.01 for drawn in ink), path.name
        line = traced[sheet]
        if sheet == "06":
            assert row["style"] == "dashed" and len(ink) > 10, path.name
            lengthened = lengthened_curve(line, RULE["line_width_mm"] / 2.0)
            for drawn in ink:
                assert distance_to_polyline(drawn["points"][::3], resampled_curve(lengthened, 0.5)).max() < 1.0, path.name
        else:
            assert row["style"] == "solid" and len(ink) == 1, path.name
            drawn = chained(ink)
            assert distance_to_polyline(resampled_curve(line, 1.0), drawn).max() < 1.0, path.name
            assert distance_to_polyline(drawn, resampled_curve(line, 0.5)).max() < 1.0, path.name



def test_margins_self_distance_and_start() -> None:
    """Every new track keeps its margins, its 25 mm self-distance and begins with a straight piece of 30 to 60 mm."""

    for track in listed_tracks():
        if track["role"] == "diagnostic":
            continue
        curve = track["curve"]
        margin = RULE["edge_margin_mm"]
        assert curve[:, 0].min() >= margin - 1e-6 and curve[:, 0].max() <= RULE["page_mm"][0] - margin + 1e-6, track["name"]
        assert curve[:, 1].min() >= margin - 1e-6 and curve[:, 1].max() <= RULE["page_mm"][1] - margin + 1e-6, track["name"]
        summary = line_summary(curve)
        assert summary["smallest_self_distance_mm"] >= RULE["smallest_gap_mm"], (track["name"], summary["smallest_self_distance_mm"])
        first = track["pieces"][0]
        assert first["kind"] == "start straight" and RULE["start_straight_mm"][0] <= first["length_mm"] <= RULE["start_straight_mm"][1], track["name"]
        line = resampled_curve(curve, 1.0)
        straight = line[:int(first["length_mm"])]
        direction = (straight[-1] - straight[0]) / numpy.linalg.norm(straight[-1] - straight[0])
        offsets = straight - straight[0]
        sideways = numpy.abs(offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0])
        assert sideways.max() < 0.05, (track["name"], sideways.max())
        for crossing in summary["crossings"]:
            assert crossing["angle_deg"] >= RULE["smallest_crossing_deg"], (track["name"], crossing["angle_deg"])



def test_list_equals_what_is_drawn() -> None:
    """The centre line read back from each solid page is the listed one, and its tightest radius and features are the listed ones."""

    for track in listed_tracks():
        if track["role"] == "diagnostic" and track["dash_mm"] is not None:
            continue
        drawn = chained(ink_of(drawn_paths(page_file(track, "solid"))))
        listed = track["curve"]
        assert len(drawn) == len(listed) and numpy.abs(drawn - listed).max() < 0.01, track["name"]
        summary = line_summary(drawn)
        assert abs(summary["tightest_radius_mm"] - track["tightest_radius_mm"]) <= max(0.1, 0.01 * track["tightest_radius_mm"]), (track["name"], summary["tightest_radius_mm"], track["tightest_radius_mm"])
        assert abs(summary["length_mm"] - track["length_mm"]) <= 0.1, track["name"]
        assert feature_names(summary) == track["features"], (track["name"], feature_names(summary), track["features"])
        assert len(summary["crossings"]) == len(track["summary"]["crossings"]), track["name"]



def test_given_centre_line_comes_out_within_1_mm() -> None:
    """Sheet 05's traced centre line, drawn as a page and read back, lies within 1 mm of the traced line, both ways."""

    traced = course_sheet_lines()["05"]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "sheet_05.pdf"
        write_page_pdf(path, ink_paths(traced, False), False, ["sheet 05 test", "test"], placed_marks(traced))
        drawn = chained(ink_of(drawn_paths(path)))
    assert distance_to_polyline(resampled_curve(traced, 1.0), drawn).max() < 1.0
    assert distance_to_polyline(drawn, resampled_curve(traced, 0.5)).max() < 1.0



def test_dashes_have_the_listed_lengths() -> None:
    """On every dashed page the inner dashes and gaps are as listed, the end dashes carry ink, and every dash lies on the line."""

    for track in listed_tracks():
        if track["dash_mm"] is None:
            continue
        path = page_file(track, "dashed")
        dashes = ink_of(drawn_paths(path))
        lengthened = lengthened_curve(track["curve"], RULE["line_width_mm"] / 2.0)
        along = arc_lengths_mm(lengthened)
        expected = dash_intervals(float(along[-1]), track["dash_mm"], track["gap_mm"])
        assert len(dashes) == len(expected), (track["name"], len(dashes), len(expected))
        lengths = [float(arc_lengths_mm(dash["points"])[-1]) for dash in dashes]
        for index, (length, (first, last)) in enumerate(zip(lengths, expected)):
            assert abs(length - (last - first)) < 0.15, (track["name"], index, length, last - first)
            if 0 < index < len(expected) - 1:
                assert abs(length - track["dash_mm"]) < 0.15, (track["name"], index, length)
            else:
                assert 0.5 * track["dash_mm"] <= length <= 1.5 * track["dash_mm"] + 0.5 * track["gap_mm"], (track["name"], index, length)
        for (_, last), (first, _) in zip(expected[:-1], expected[1:]):
            assert abs(first - last - track["gap_mm"]) < 1e-6, track["name"]
        for dash in dashes:
            assert distance_to_polyline(dash["points"][::4], lengthened).max() < 0.05, track["name"]
        assert numpy.linalg.norm(dashes[0]["points"][0] - lengthened[0]) < 0.01 and numpy.linalg.norm(dashes[-1]["points"][-1] - lengthened[-1]) < 0.01, track["name"]



def test_dash_rule_on_its_own() -> None:
    """The dash rule: exact inner dashes and gaps, the ends changed alike, the whole length covered, for many lengths."""

    for total in numpy.arange(150.0, 800.0, 7.3):
        for dash, gap in ((15.0, 8.0), (20.0, 10.0), (35.0, 15.0)):
            intervals = dash_intervals(float(total), dash, gap)
            assert intervals[0][0] == 0.0 and abs(intervals[-1][1] - total) < 1e-9
            inner = [last - first for first, last in intervals[1:-1]]
            assert all(abs(length - dash) < 1e-9 for length in inner)
            ends = [intervals[0][1] - intervals[0][0], intervals[-1][1] - intervals[-1][0]]
            assert abs(ends[0] - ends[1]) < 1e-9 and min(ends) >= dash - (dash + gap) / 4.0 - 1e-9, (total, dash, gap, ends)



def test_training_tracks_pass_the_layout_check() -> None:
    """Every training (and development) track passes the lay-out check as listed, recomputed from tracks.json alone."""

    for track in listed_tracks():
        if track["role"] == "diagnostic":
            continue
        listed = track["lay_out"]
        check = layout_check(track["curve"], listed["start_end"], listed["turn_deg"])
        assert check["passes"], (track["name"], check)
        assert abs(check["largest_line_reach_cm"] - listed["largest_line_reach_cm"]) < 0.05, track["name"]
        assert abs(check["running_back_cm"] - listed["running_back_cm"]) < 0.05, track["name"]
        assert check["running_back_cm"] * 10.0 <= RULE["running_back_share_most"] * track["length_mm"], track["name"]



def test_measures_on_known_shapes() -> None:
    """The measures on shapes with known answers: a circle arc's radius, a hairpin, a loop's crossing angle."""

    angles = numpy.radians(numpy.arange(0.0, 120.0, 0.5))
    arc = numpy.column_stack([100 + 20 * numpy.cos(angles), 100 + 20 * numpy.sin(angles)])
    assert abs(line_summary(arc)["tightest_radius_mm"] - 20.0) < 0.05
    straight_in = numpy.column_stack([numpy.full(60, 50.0), numpy.linspace(20.0, 80.0, 60)])
    turn = numpy.radians(numpy.linspace(180.0, 0.0, 60))
    hairpin = numpy.vstack([straight_in, numpy.column_stack([58.0 + 8.0 * numpy.cos(turn), 80.0 + 8.0 * numpy.sin(turn)]),
                            numpy.column_stack([numpy.full(60, 66.0), numpy.linspace(80.0, 20.0, 60)])])
    summary = line_summary(hairpin)
    assert summary["hairpins"] == 1 and abs(summary["tightest_radius_mm"] - 8.0) < 0.3, summary["bends"]
    loop_angles = numpy.radians(numpy.arange(0.0, 300.0, 1.0))
    loop = numpy.column_stack([100 + 30 * numpy.cos(loop_angles), 150 + 30 * numpy.sin(loop_angles)])
    lead_in = loop[0] - numpy.outer(numpy.linspace(60.0, 0.5, 60), [0.0, 1.0])
    end_direction = numpy.array([-math.sin(loop_angles[-1]), math.cos(loop_angles[-1])])
    lead_out = loop[-1] + numpy.outer(numpy.linspace(0.5, 60.0, 60), end_direction)
    crossing = line_summary(numpy.vstack([lead_in, loop, lead_out]))["crossings"]
    assert len(crossing) == 1 and abs(crossing[0]["angle_deg"] - 60.0) < 1.5, crossing



if __name__ == "__main__":
    test_measures_on_known_shapes()
    test_dash_rule_on_its_own()
    test_given_centre_line_comes_out_within_1_mm()
    test_printer_page_is_true_size()
    test_every_face_holds_the_line_only()
    test_print_files_hold_the_listed_pages_in_order()
    test_course_sheet_copies()
    test_margins_self_distance_and_start()
    test_list_equals_what_is_drawn()
    test_dashes_have_the_listed_lengths()
    test_training_tracks_pass_the_layout_check()
    print(f"All track tests passed ({len(listed_tracks())} tracks in tracks.json).")
