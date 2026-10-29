"""
Camera bracket "F" for the Niryo Ned2 wrist: a printed L-shaped riser that
moves Niryo's own camera holder up and forward.

Frame used throughout (millimetres):

* origin: the centre of the four-hole pattern on the aluminium brace plate,
  on the plate's outer face;
* x: forward, away from the brace face (towards where the camera sits);
* y: up, away from the tool, along the wrist;
* z: sideways, across the wrist.

The bracket consists of a foot (screwed to the brace with the two stock
screw holes), a short arm leaving the cover window, a post rising beside
the wrist, and a top plate that copies the brace's four-hole pattern, so the
stock holder screws onto it exactly as onto the brace, tilted back by a few
degrees.  Every number is tagged in DEFAULT_BRACKET_PARAMETERS as measured,
assumed or chosen.

Run with the project interpreter; outputs land next to this file.
"""

import json
import math
import pathlib

import cadquery
from matplotlib import pyplot


OUTPUT_DIRECTORY = pathlib.Path(__file__).resolve().parent

DEFAULT_BRACKET_PARAMETERS = {
    # Measured on the robot by Emranul, 28.10.26 (measurement_hour_sheet.md).
    "plate_width_z": 34.0,              # brace plate 34.0 to 34.2 wide
    "plate_height_y": 22.0,             # brace plate 22.0 to 22.5 tall
    "hole_spacing_z": 26.0,             # holder ear holes, centre to centre
    "screw_thread_diameter": 2.5,       # M2.5 x 6, 2 mm hex key
    "screw_head_diameter": 4.5,         # typical M2.5 socket head (assumed)
    # Assumed until measured (ask for them before printing).
    "hole_spacing_y": 16.0,             # screw row to pin row; Ned v1 holder pattern
    "pin_diameter": 2.0,                # the holder's two locating pins
    "ear_thickness": 3.0,               # the holder's ear under the screw head
    # Where the camera is today, fitted from the team's own pictures on 26.10.26
    # (scripts/cam_calibration.json, pixel_fit): lens 25 mm ahead of the rod
    # axis, 6 mm above the joint-5 axis, looking 22 degrees forward of straight
    # down.  Relative to the brace pattern the lens is taken to sit 16 mm in
    # front of the brace face at the pattern's centre height (assumed, +-5 mm).
    "lens_today_ahead_of_brace_face": 16.0,
    "lens_today_above_pattern_centre": 0.0,
    "lens_today_tilt_forward_degrees": 22.4,
    # Target (option F of design_review_one_print.md, M17 and the mock-up list):
    # lens 51 mm ahead of the rod axis, 60 mm above the joint-5 axis, looking
    # 10 degrees back towards the robot.  Expressed as the change from today.
    "lens_up_mm": 54.0,                 # 60 - 6
    "lens_tilt_back_degrees": 10.0,     # target: 10 degrees towards the robot
    # Print and strength choices (Maker Space: PLA or PETG, 0.4 mm nozzle).
    "foot_thickness_x": 3.0,            # same as the ear, so the stock screws fit
    "arm_length_x": 12.0,               # foot to post, clears the cover rim
    "post_thickness_x": 12.0,           # camera moves forward by arm + post
    "post_width_z": 16.0,               # narrow so a hex key reaches the foot screws
    "top_plate_thickness_x": 6.0,
    "gusset_size": 10.0,
    "clearance_hole_extra": 0.5,        # printed holes shrink; M2.5 needs about 3.0
    "self_tap_hole_diameter": 2.2,      # M2.5 cuts its own thread in PLA
    "pin_hole_extra": 0.6,
    "corner_radius": 3.0,
}


def derived_geometry(p: dict) -> dict:
    """
    Turn the lens target into the plate's rotation and height.

    The top plate copies the brace pattern, so the holder puts the lens at the
    same offset from the pattern as today.  Rotating the plate about its
    bottom edge by the required tilt change moves that offset; the raise is
    corrected so the lens, not the plate, ends up the wanted distance higher.

    Returns
    -------
    dict
        plate_tilt_degrees, raise_mm, and the predicted lens position.
    """

    half_height = p["plate_height_y"] / 2.0
    post_front_x = p["arm_length_x"] + p["post_thickness_x"]
    plate_tilt = p["lens_today_tilt_forward_degrees"] + p["lens_tilt_back_degrees"]
    angle = math.radians(plate_tilt)
    # Lens offset from the plate's bottom edge (the pivot) today.
    offset_x = p["lens_today_ahead_of_brace_face"]
    offset_y = p["lens_today_above_pattern_centre"] + half_height
    # Plate top leans forward: rotate the offset by -angle about z.
    rotated_x = offset_x * math.cos(angle) + offset_y * math.sin(angle)
    rotated_y = -offset_x * math.sin(angle) + offset_y * math.cos(angle)
    raise_mm = p["lens_up_mm"] + (offset_y - rotated_y)
    lens_forward_change = post_front_x + p["top_plate_thickness_x"] + rotated_x - offset_x
    return {
        "plate_tilt_degrees": round(plate_tilt, 1),
        "raise_mm": round(raise_mm, 1),
        "lens_moves_forward_mm": round(lens_forward_change, 1),
        "lens_moves_up_mm": p["lens_up_mm"],
        "lens_ahead_of_rod_axis_mm": round(25.0 + lens_forward_change, 1),
        "lens_above_joint5_axis_mm": round(6.0 + p["lens_up_mm"], 1),
        "lens_tilt_degrees_forward": round(-p["lens_tilt_back_degrees"], 1),
    }


def build_bracket(parameters: dict = None) -> cadquery.Workplane:
    """
    Build the bracket solid.

    Parameters
    ----------
    parameters : dict, optional
        Values that override DEFAULT_BRACKET_PARAMETERS.

    Returns
    -------
    cadquery.Workplane
        The bracket as a single solid, millimetres, in the frame described
        in the module docstring.
    """

    p = dict(DEFAULT_BRACKET_PARAMETERS)
    p.update(parameters or {})

    half_height = p["plate_height_y"] / 2.0
    hole_z = p["hole_spacing_z"] / 2.0
    hole_y = p["hole_spacing_y"] / 2.0
    post_back_x = p["arm_length_x"]
    post_front_x = p["arm_length_x"] + p["post_thickness_x"]
    geometry = derived_geometry(p)
    top_plate_centre_y = geometry["raise_mm"]
    post_top_y = top_plate_centre_y + half_height
    tilt = math.radians(geometry["plate_tilt_degrees"])

    # Foot: a plate on the brace face with two clearance holes for the stock
    # screws.  The brace's lower pin holes stay free.
    clearance_diameter = p["screw_thread_diameter"] + p["clearance_hole_extra"]
    foot = (
        cadquery.Workplane("YZ")
        .rect(p["plate_height_y"], p["plate_width_z"])
        .extrude(p["foot_thickness_x"])
        .edges("|X").fillet(p["corner_radius"])
        .faces(">X").workplane()
        .pushPoints([(hole_y, -hole_z), (hole_y, hole_z)])
        .hole(clearance_diameter)
    )

    # Arm: a bar leaving the cover window, the full foot height but only the
    # post's width, so the screw heads beside it stay reachable with the key.
    arm = (
        cadquery.Workplane("YZ")
        .rect(p["plate_height_y"], p["post_width_z"])
        .extrude(post_front_x)
    )

    # Post: rises from the arm to the top plate.
    post = (
        cadquery.Workplane("XY")
        .center((post_back_x + post_front_x) / 2.0, (half_height + post_top_y) / 2.0)
        .rect(p["post_thickness_x"], post_top_y - half_height)
        .extrude(p["post_width_z"] / 2.0, both = True)
    )

    # Gusset behind the post, inside the cover window, against the bending load.
    gusset = (
        cadquery.Workplane("XY")
        .polyline([
            (post_back_x, half_height),
            (post_back_x, half_height + p["gusset_size"]),
            (post_back_x - p["gusset_size"] + 2.0, half_height),
        ]).close()
        .extrude(p["post_width_z"] / 2.0, both = True)
    )

    # Top plate: copies the brace pattern, tilted back about its bottom edge.
    # Holes: the upper pair takes the stock M2.5 screws (self-tapping), the
    # lower pair takes the holder's two locating pins (blind).
    top_plate = (
        cadquery.Workplane("YZ")
        .workplane(offset = post_front_x)
        .center(top_plate_centre_y, 0.0)
        .rect(p["plate_height_y"], p["plate_width_z"])
        .extrude(p["top_plate_thickness_x"])
        .edges("|X").fillet(p["corner_radius"])
    )
    top_plate = (
        top_plate.faces(">X").workplane(origin = (0, 0, 0))
        .pushPoints([(top_plate_centre_y + hole_y, -hole_z), (top_plate_centre_y + hole_y, hole_z)])
        .hole(p["self_tap_hole_diameter"])
        .faces(">X").workplane(origin = (0, 0, 0))
        .pushPoints([(top_plate_centre_y - hole_y, -hole_z), (top_plate_centre_y - hole_y, hole_z)])
        .hole(p["pin_diameter"] + p["pin_hole_extra"], depth = p["top_plate_thickness_x"] - 1.5)
    )
    pivot = cadquery.Vector(post_front_x, top_plate_centre_y - half_height, 0.0)
    # Negative angle: the plate's top leans forward, so the holder's lens, which
    # looks down (-y), turns back towards the rod (-x).
    top_plate = top_plate.rotate(pivot, pivot + cadquery.Vector(0, 0, 1), -geometry["plate_tilt_degrees"])

    # A wedge filling the gap between the straight post and the leaning plate.
    wedge = (
        cadquery.Workplane("XY")
        .polyline([
            (post_front_x, top_plate_centre_y - half_height),
            (post_front_x, post_top_y),
            (post_front_x + p["plate_height_y"] * math.sin(tilt),
             top_plate_centre_y - half_height + p["plate_height_y"] * math.cos(tilt)),
        ]).close()
        .extrude(p["post_width_z"] / 2.0, both = True)
    )

    bracket = foot.union(arm).union(post).union(gusset).union(wedge).union(top_plate)
    return bracket


def export_bracket(
        bracket: cadquery.Workplane,
        parameters: dict,
        stem: str = "ned2_camera_bracket_F",
) -> dict:
    """
    Write STEP, STL and 3MF files (millimetres) and a JSON of the parameters.

    Returns
    -------
    dict
        File paths, bounding box in millimetres and volume.
    """

    paths = {}
    for extension in ("step", "stl", "3mf"):
        path = OUTPUT_DIRECTORY / f"{stem}.{extension}"
        cadquery.exporters.export(bracket, str(path))
        paths[extension] = str(path)
    bounding_box = bracket.val().BoundingBox()
    summary = {
        "bounding_box_mm": {
            "x_forward": [round(bounding_box.xmin, 2), round(bounding_box.xmax, 2)],
            "y_up": [round(bounding_box.ymin, 2), round(bounding_box.ymax, 2)],
            "z_sideways": [round(bounding_box.zmin, 2), round(bounding_box.zmax, 2)],
        },
        "volume_cm3": round(bracket.val().Volume() / 1000.0, 1),
        "derived_geometry": derived_geometry(parameters),
        "parameters": parameters,
        "files": paths,
    }
    with open(OUTPUT_DIRECTORY / f"{stem}_parameters.json", "w") as handle:
        json.dump(summary, handle, indent = 2)
    return summary


def draw_side_view(parameters: dict, stem: str = "ned2_camera_bracket_F") -> str:
    """Draw the x-y profile with the camera on it, for a plain-language check."""

    p = parameters
    geometry = derived_geometry(p)
    half_height = p["plate_height_y"] / 2.0
    post_front_x = p["arm_length_x"] + p["post_thickness_x"]
    centre_y = geometry["raise_mm"]
    pivot = (post_front_x, centre_y - half_height)
    angle = -geometry["plate_tilt_degrees"]

    figure, axis = pyplot.subplots(figsize = (6, 8))
    axis.add_patch(pyplot.Rectangle((0, -half_height), post_front_x, p["plate_height_y"], color = "tab:blue"))
    axis.add_patch(pyplot.Rectangle((p["arm_length_x"], half_height), p["post_thickness_x"], centre_y, color = "tab:blue"))
    axis.add_patch(pyplot.Rectangle(pivot, p["top_plate_thickness_x"], p["plate_height_y"],
                                    angle = angle, rotation_point = pivot, color = "tab:blue"))
    # Camera as today: its lens face lies along x (53 long), the case 31 tall
    # above it; drawn in the plate's frame and rotated with the plate.
    case_x = post_front_x + p["top_plate_thickness_x"] + p["ear_thickness"]
    axis.add_patch(pyplot.Rectangle((case_x, centre_y - half_height), 53.1, 31.0,
                                    angle = angle, rotation_point = pivot, color = "black", alpha = 0.6))
    lens_x = post_front_x + p["top_plate_thickness_x"] + p["lens_today_ahead_of_brace_face"]
    lens_y = centre_y + p["lens_today_above_pattern_centre"]
    a = math.radians(angle)
    lx = pivot[0] + (lens_x - pivot[0]) * math.cos(a) - (lens_y - pivot[1]) * math.sin(a)
    ly = pivot[1] + (lens_x - pivot[0]) * math.sin(a) + (lens_y - pivot[1]) * math.cos(a)
    axis.plot(lx, ly, "o", color = "red")
    look = math.radians(geometry["lens_tilt_degrees_forward"])
    axis.annotate("", xy = (lx + 40 * math.sin(look), ly - 40 * math.cos(look)), xytext = (lx, ly),
                  arrowprops = {"arrowstyle": "->", "color": "red"})
    axis.text(lx + 3, ly + 3, "lens: looks down,\n10 deg back", color = "red", fontsize = 8)
    axis.axvline(0, color = "grey", linestyle = "--")
    axis.text(-1.5, 125, "metal plate face (wrist)", rotation = 90, ha = "right", va = "top", color = "grey")
    axis.axvline(-9.0, color = "grey", linestyle = ":")
    axis.text(-10.5, 125, "rod axis (Niryo STEP)", rotation = 90, ha = "right", va = "top", color = "grey", fontsize = 8)
    axis.set_xlim(-30, 90)
    axis.set_ylim(-35, 130)
    axis.set_aspect("equal")
    axis.set_xlabel("x: forward from the metal plate face (mm)")
    axis.set_ylabel("y: up from the screw-pattern centre (mm)")
    axis.set_title(f"Bracket F: plate {centre_y:.0f} mm up, {post_front_x:.0f} mm forward, leaning {-angle:.0f} deg", fontsize = 10)
    path = OUTPUT_DIRECTORY / f"{stem}_side_view.png"
    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    return str(path)


if __name__ == "__main__":
    parameters = dict(DEFAULT_BRACKET_PARAMETERS)
    bracket = build_bracket(parameters)
    summary = export_bracket(bracket, parameters)
    summary["side_view"] = draw_side_view(parameters)
    print(json.dumps({key: value for key, value in summary.items() if key != "parameters"}, indent = 2))
