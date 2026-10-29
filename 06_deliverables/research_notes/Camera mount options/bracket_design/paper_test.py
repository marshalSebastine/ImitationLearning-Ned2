"""
Print this PDF at 100 percent (no "fit to page") and lay the outlines on the
real parts: the foot pattern on the aluminium brace plate, the top-plate
pattern on the back of Niryo's camera holder.  If every hole lines up, the
printed bracket will too.  A 50 mm check bar is drawn to verify the scale.
"""

import pathlib

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from bracket_f_design import DEFAULT_BRACKET_PARAMETERS as parameters, derived_geometry


def draw_pattern(
        page: canvas.Canvas,
        centre_x: float,
        centre_y: float,
        title: str,
        upper_hole_diameter: float,
        lower_hole_diameter: float,
) -> None:

    width = parameters["plate_width_z"] * mm
    height = parameters["plate_height_y"] * mm
    hole_z = parameters["hole_spacing_z"] / 2.0 * mm
    hole_y = parameters["hole_spacing_y"] / 2.0 * mm
    page.setLineWidth(0.3)
    page.roundRect(centre_x - width / 2, centre_y - height / 2, width, height, parameters["corner_radius"] * mm)
    for sign in (-1, 1):
        page.circle(centre_x + sign * hole_z, centre_y + hole_y, upper_hole_diameter / 2.0 * mm)
        page.circle(centre_x + sign * hole_z, centre_y - hole_y, lower_hole_diameter / 2.0 * mm)
        # Cross hairs so the hole centres can be read through the paper.
        for (dx, dy) in ((1, 0), (0, 1)):
            for row in (hole_y, -hole_y):
                page.line(centre_x + sign * hole_z - dx * 4 * mm, centre_y + row - dy * 4 * mm,
                          centre_x + sign * hole_z + dx * 4 * mm, centre_y + row + dy * 4 * mm)
    page.setFont("Helvetica", 9)
    page.drawCentredString(centre_x, centre_y + height / 2 + 4 * mm, title)
    page.drawCentredString(centre_x, centre_y - height / 2 - 6 * mm,
                           f"holes {parameters['hole_spacing_z']:.0f} apart sideways, "
                           f"{parameters['hole_spacing_y']:.0f} apart up/down; plate "
                           f"{parameters['plate_width_z']:.0f} x {parameters['plate_height_y']:.0f} mm")


output = pathlib.Path(__file__).resolve().parent / "ned2_camera_bracket_F_paper_test.pdf"
page = canvas.Canvas(str(output), pagesize = A4)
page_width, page_height = A4
page.setFont("Helvetica-Bold", 12)
page.drawString(20 * mm, page_height - 20 * mm, "Camera bracket F: 1:1 paper test (print at 100 %, no scaling)")
page.setFont("Helvetica", 9)
page.drawString(20 * mm, page_height - 27 * mm, "1. Measure the bar below with a ruler: it must be exactly 50 mm. If not, the print was scaled.")
page.setLineWidth(0.8)
page.line(20 * mm, page_height - 35 * mm, 70 * mm, page_height - 35 * mm)
page.line(20 * mm, page_height - 33 * mm, 20 * mm, page_height - 37 * mm)
page.line(70 * mm, page_height - 33 * mm, 70 * mm, page_height - 37 * mm)
page.drawString(72 * mm, page_height - 36 * mm, "50 mm")
page.drawString(20 * mm, page_height - 45 * mm, "2. Cut out pattern A, hold it on the aluminium plate in the wrist: the two upper holes must sit over the two screw holes.")
page.drawString(20 * mm, page_height - 51 * mm, "3. Cut out pattern B, hold it on the back of the camera holder: upper holes over the screw holes, lower holes over the pins.")
page.drawString(20 * mm, page_height - 57 * mm, "4. If anything is off by more than 1 mm, tell Claude the direction and the amount before the file is sent.")

draw_pattern(page, page_width / 2, page_height - 95 * mm, "A: foot, goes on the aluminium brace plate (upper holes 3.0 mm)",
             upper_hole_diameter = parameters["screw_thread_diameter"] + parameters["clearance_hole_extra"],
             lower_hole_diameter = parameters["pin_diameter"] + parameters["pin_hole_extra"])
draw_pattern(page, page_width / 2, page_height - 150 * mm, "B: top plate, goes on the back of the camera holder (upper 2.2, lower 2.6 mm)",
             upper_hole_diameter = parameters["self_tap_hole_diameter"],
             lower_hole_diameter = parameters["pin_diameter"] + parameters["pin_hole_extra"])

# Side profile of the whole bracket, 1:1, to hold beside the wrist.
origin_x, origin_y = 60 * mm, 40 * mm
page.setLineWidth(0.4)
foot_half = parameters["plate_height_y"] / 2.0
post_back = parameters["arm_length_x"]
post_front = post_back + parameters["post_thickness_x"]
raise_mm = derived_geometry(parameters)["raise_mm"]
import math
geometry = derived_geometry(parameters)
raise_mm = geometry["raise_mm"]
lean = math.radians(geometry["plate_tilt_degrees"])
plate_thickness = parameters["top_plate_thickness_x"]
plate_height = parameters["plate_height_y"]
pivot = (post_front, raise_mm - foot_half)


def leaning(dx: float, dy: float) -> tuple:
    """A point of the upright top plate, given from its bottom-back corner, after the lean."""

    return (pivot[0] + dx * math.cos(lean) + dy * math.sin(lean),
            pivot[1] - dx * math.sin(lean) + dy * math.cos(lean))


profile = [(0, -foot_half), (post_front, -foot_half), pivot,
           leaning(plate_thickness, 0), leaning(plate_thickness, plate_height), leaning(0, plate_height),
           (post_front, raise_mm + foot_half), (post_back, raise_mm + foot_half), (post_back, foot_half), (0, foot_half)]
path = page.beginPath()
path.moveTo(origin_x + profile[0][0] * mm, origin_y + profile[0][1] * mm)
for x, y in profile[1:]:
    path.lineTo(origin_x + x * mm, origin_y + y * mm)
path.close()
page.drawPath(path, stroke = 1, fill = 0)
page.setFont("Helvetica", 9)
page.drawString(origin_x + 35 * mm, origin_y + 30 * mm, "C: side outline of the bracket, 1:1")
page.drawString(origin_x + 35 * mm, origin_y + 24 * mm, "(left edge = metal plate face; camera holder")
page.drawString(origin_x + 35 * mm, origin_y + 18 * mm, "screws onto the leaning plate at the top)")
page.save()
print(output)
