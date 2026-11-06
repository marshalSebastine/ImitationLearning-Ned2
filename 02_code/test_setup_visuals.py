"""
Tests of setup_visuals.py: every plan number has a picture, and what is drawn matches the plan table. No robot is used.

Run from 02_code, after python setup_visuals.py:
    python test_setup_visuals.py
"""

import math
import re

import numpy

from config import RECORDING_PAGE_PARAMETERS, SETUP_VISUALS_PARAMETERS, WORKSPACE_PARAMETERS
from recording_page import expand_episode_plan
from setup_visuals import PROJECT_FOLDER, all_setups, arc_lengths, self_crossings


TOLERANCE_CM = 0.05



def line_angle_deg(line: numpy.ndarray) -> float:
    """The direction of the line's first centimetre, in degrees counter-clockwise from +x."""

    lengths = arc_lengths(line)
    ahead = line[numpy.searchsorted(lengths, 1.0)] - line[0]
    return math.degrees(math.atan2(ahead[1], ahead[0]))



def test_every_plan_number_has_a_picture_and_a_card() -> None:
    """One picture per plan number, the same numbers in the same order as the plan table, and every one on the page."""

    folder = PROJECT_FOLDER / SETUP_VISUALS_PARAMETERS["output_folder"]
    plan = expand_episode_plan()
    planned = [number for entries in plan.values() for number in entries]
    drawn = [setup["plan_number"] for setup in all_setups()]
    assert drawn == planned, "the drawn plan numbers differ from the plan table"
    assert len(planned) == 200, len(planned)
    page = (folder / "index.html").read_text(encoding = "utf-8")
    for number in planned:
        assert (folder / f"plan_{number}.png").is_file(), f"no picture for {number}"
        assert f'id="{number}"' in page, f"{number} is not on the page"
    pictures = {path.name for path in folder.glob("plan_*.png")}
    assert pictures == {f"plan_{number}.png" for number in planned}, "a picture is left over from an old plan"



def test_setups_follow_the_plan_table() -> None:
    """Start code, end, offset side and size, turn side and size, and the start position, for every plan number."""

    plan = expand_episode_plan()
    push = numpy.array(WORKSPACE_PARAMETERS["push_direction_xy"])
    push_angle = math.degrees(math.atan2(push[1], push[0]))
    mark = numpy.array(WORKSPACE_PARAMETERS["line_start_xy"]) * 100
    for setup in all_setups():
        entry = plan[setup["path_id"]][setup["plan_number"]]
        start = RECORDING_PAGE_PARAMETERS["start_codes"][entry["start_code"]]
        assert setup["start_code"] == entry["start_code"] and setup["line_end"] == entry["line_end"], setup["plan_number"]

        # The start cap lies on the mark, half a centimetre (cap minus ink tip) along the push direction.
        assert abs(numpy.linalg.norm(setup["start_cap"] - mark) - 0.5) < 0.1, setup["plan_number"]

        # The sheet's turn: the paper's edge is turned by offset_deg (left +) from where it lies for a standard start, and
        # the line's first centimetre points roughly along the push direction turned the same way (sheet 03's end A
        # bends 5 degrees within that centimetre, so this second check is loose).
        edge, standard_edge = setup["paper"][1] - setup["paper"][0], setup["standard_paper"][1] - setup["standard_paper"][0]
        paper_turn = math.degrees(math.atan2(edge[1], edge[0]) - math.atan2(standard_edge[1], standard_edge[0]))
        assert abs((paper_turn - start["offset_deg"] + 180) % 360 - 180) < 0.01, (setup["plan_number"], paper_turn)
        line_turn = (line_angle_deg(setup["line"]) - push_angle - start["offset_deg"] + 180) % 360 - 180
        assert abs(line_turn) < 6.0, (setup["plan_number"], line_turn)

        # The puck: offset_cm to the left (+y side of the push direction) of the cap, or on the line for taught starts.
        if entry["start_code"] in ("NE", "MID", "X1", "X2"):
            distance_to_line = numpy.min(numpy.linalg.norm(setup["line"] - setup["puck"], axis = 1))
            assert distance_to_line < TOLERANCE_CM, setup["plan_number"]
            assert setup["teach_point"] is not None
        else:
            shift = setup["puck"] - setup["start_cap"]
            direction = push / numpy.linalg.norm(push)
            sideways = float(direction[0] * shift[1] - direction[1] * shift[0])
            assert abs(sideways - start["offset_cm"]) < TOLERANCE_CM, (setup["plan_number"], sideways)
            assert setup["teach_point"] is None



def test_taught_starts_lie_where_the_plan_says() -> None:
    """NE before the end cap by the note's distance; X1 and X2 five centimetres before the first and second crossing."""

    plan = expand_episode_plan()
    for setup in all_setups():
        note = plan[setup["path_id"]][setup["plan_number"]]["note"]
        if setup["start_code"] == "NE":
            match = re.search(r"(\d+)(?: to (\d+))? cm before the end cap", note)
            wanted = SETUP_VISUALS_PARAMETERS["near_end_cm"] if match is None else (float(match.group(1)) + float(match.group(2) or match.group(1))) / 2
            assert abs(setup["line_length_cm"] - setup["puck_arc_cm"] - wanted) < TOLERANCE_CM, setup["plan_number"]
        if setup["start_code"] in ("X1", "X2"):
            passes = self_crossings(setup["line"])
            assert len(passes) == 1, f"{setup['plan_number']}: expected one crossing, found {len(passes)}"
            crossing = passes[0][0] if setup["start_code"] == "X1" else passes[0][1]
            assert abs(crossing - setup["puck_arc_cm"] - SETUP_VISUALS_PARAMETERS["crossing_lead_cm"]) < TOLERANCE_CM
        if setup["start_code"] == "MID":
            assert 0 < setup["puck_arc_cm"] < setup["line_length_cm"], setup["plan_number"]



def test_ends_are_the_traced_first_and_last() -> None:
    """End A starts at the traced line's first point and end B at its last, as on the sheets figure."""

    for setup in all_setups():
        other = setup["line"][-1]
        assert numpy.linalg.norm(setup["line"][0] - setup["start_cap"]) < 0.6, setup["plan_number"]
        assert numpy.linalg.norm(other - setup["end_cap"]) < 0.6, setup["plan_number"]


if __name__ == "__main__":
    test_every_plan_number_has_a_picture_and_a_card()
    test_setups_follow_the_plan_table()
    test_taught_starts_lie_where_the_plan_says()
    test_ends_are_the_traced_first_and_last()
    print("All set-up picture tests passed.")
