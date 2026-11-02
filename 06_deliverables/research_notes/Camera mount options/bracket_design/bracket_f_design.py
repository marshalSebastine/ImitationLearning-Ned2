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
    # F2 (2.11.26): 27.0 as in Niryo's CAD of the holder (cad check of 30.10), not the 26 the calipers gave on 28.10;
    # fitting F on 2.11 Emranul found the screws hard to get in and wanted 0.5 to 1.5 mm more between the holes.
    "hole_spacing_z": 27.0,             # holder ear holes and brace holes, centre to centre
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
    # F2 (2.11.26): 20 degrees instead of 10, so the lens, which the leaning post
    # carries about 24 mm further forward, still aims at the puck: with the lens
    # about 120 mm above the paper at pushing height, 82 mm ahead of the rod and
    # looking 20 degrees back, it aims about 38 mm ahead of the rod axis, where
    # F aimed with 58 mm and 10 degrees (calc; the mock-up picture decides).
    # 30 on 2.11.26 (afternoon), after the meander test showed that a leg running back towards the robot must be
    # pushed from the far side of the puck, where a camera aimed ahead of the rod sees neither puck nor line. At 30
    # degrees the picture covers from 7.8 cm behind the rod axis to 8.3 cm ahead (calc, lens 93 mm ahead and
    # 120 mm up, 25 degree vertical half-angle): about 5 cm of line beyond the puck in both pushing directions.
    "lens_tilt_back_degrees": 30.0,
    # Print and strength choices (Maker Space: PLA or PETG, 0.4 mm nozzle).
    "foot_thickness_x": 3.0,            # same as the ear, so the stock screws fit
    "arm_length_x": 12.0,               # foot to post, clears the cover rim
    "post_thickness_x": 12.0,           # camera moves forward by arm + post
    # F2 (2.11.26): the post leans forward by this angle. Measured on the robot
    # with F fitted: the back of the straight post meets the forearm cover when
    # the wrist (joint 5) is tilted up to -2 degrees; the pushing pose nearest
    # the base (rod 14 cm out) needs +14 degrees, so +18 with margin, i.e. 20
    # degrees more travel. A point 65 mm above the joint-5 axis sweeps about
    # 22 mm over 20 degrees; a 70 mm post leaning 20 degrees moves its top back
    # edge 25 mm forward. The foot and the cross-bar stay as printed in F.
    # Set to 25 after the mesh check (scripts/review_f2_clearance.py, Niryo's forearm STL): the model predicts F
    # touching at wrist +4 degrees where the robot showed -2, so it is about 5 mm optimistic; 20 degrees left 2 to
    # 6 mm of modelled clearance at the near poses, 25 degrees leaves 6 to 10 mm.
    "post_lean_degrees": 25.0,
    "post_width_z": 16.0,               # narrow so a hex key reaches the foot screws
    "top_plate_thickness_x": 6.0,
    # The brace plate has a raised rounded boss around the shaft in its middle
    # (seen 29.10, not measured; the stock holder has a gap around it).  The
    # foot therefore only touches the plate on two side strips, like the
    # holder's ears, and nothing of the bracket comes closer to the plate than
    # pocket_depth_x within half_pocket_width_z of the centre line.
    "half_pocket_width_z": 9.5,
    "pocket_depth_x": 10.0,
    # Below this height the legs and cross-bar may fill the space in front of
    # the plate; above it the space in front of the screw heads stays open so
    # the hex key reaches them.
    "open_above_y": 5.0,
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
    # The leaning post carries its top, and with it the top plate, forward.
    post_height = raise_mm + half_height - p["open_above_y"]
    post_top_shift = post_height * math.tan(math.radians(p.get("post_lean_degrees", 0.0)))
    lens_forward_change = post_front_x + post_top_shift + p["top_plate_thickness_x"] + rotated_x - offset_x
    return {
        "plate_tilt_degrees": round(plate_tilt, 1),
        "raise_mm": round(raise_mm, 1),
        "post_lean_degrees": p.get("post_lean_degrees", 0.0),
        "post_top_shift_mm": round(post_top_shift, 1),
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
    # F2: the post leans forward, so everything above it (top plate, wedge, camera)
    # sits post_top_shift further forward than the post's foot.
    shift = geometry["post_top_shift_mm"]
    top_front_x = post_front_x + shift

    # Foot: a 3 mm plate on the brace face with two clearance holes for the
    # stock screws; its middle is cut away below (pocket) for the plate's boss.
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

    # Legs: from the foot forward to the cross-bar, only below the screw heads.
    legs_height = p["open_above_y"] + half_height
    legs = (
        cadquery.Workplane("XY")
        .center(p["pocket_depth_x"] / 2.0, -half_height + legs_height / 2.0)
        .rect(p["pocket_depth_x"], legs_height)
        .extrude(p["plate_width_z"] / 2.0, both = True)
    )

    # Cross-bar: full width, in front of the boss, joins the legs to the post.
    cross_bar = (
        cadquery.Workplane("XY")
        .center((p["pocket_depth_x"] + post_front_x) / 2.0, -half_height + legs_height / 2.0)
        .rect(post_front_x - p["pocket_depth_x"], legs_height)
        .extrude(p["plate_width_z"] / 2.0, both = True)
    )

    # Pocket for the boss: nothing within half_pocket_width_z of the centre
    # line closer to the plate than pocket_depth_x.
    pocket = (
        cadquery.Workplane("XY")
        .center(p["pocket_depth_x"] / 2.0 - 1.0, 0.0)
        .rect(p["pocket_depth_x"] + 2.0, p["plate_height_y"] + 4.0)
        .extrude(p["half_pocket_width_z"], both = True)
    )

    # Post: rises from the cross-bar to the top plate; narrow, so the hex key
    # reaches the foot screws beside it.
    post_bottom_y = p["open_above_y"]
    # A parallelogram: straight for F (shift 0), leaning forward for F2.
    post = (
        cadquery.Workplane("XY")
        .polyline([
            (post_back_x, post_bottom_y),
            (post_front_x, post_bottom_y),
            (post_front_x + shift, post_top_y),
            (post_back_x + shift, post_top_y),
        ]).close()
        .extrude(p["post_width_z"] / 2.0, both = True)
    )

    # Top plate: copies the brace pattern, tilted back about its bottom edge.
    # Holes: the upper pair takes the stock M2.5 screws (self-tapping), the
    # lower pair takes the holder's two locating pins (blind).
    top_plate = (
        cadquery.Workplane("YZ")
        .workplane(offset = top_front_x)
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
    pivot = cadquery.Vector(top_front_x, top_plate_centre_y - half_height, 0.0)
    # Negative angle: the plate's top leans forward, so the holder's lens, which
    # looks down (-y), turns back towards the rod (-x).
    top_plate = top_plate.rotate(pivot, pivot + cadquery.Vector(0, 0, 1), -geometry["plate_tilt_degrees"])

    # A wedge filling the gap between the post's front and the leaning plate.
    # Its lower corner sits where the post's front face is at the plate's
    # bottom edge, which for a leaning post lies a little behind top_front_x.
    plate_bottom_y = top_plate_centre_y - half_height
    lean = math.radians(p.get("post_lean_degrees", 0.0))
    front_at_plate_bottom = post_front_x + (plate_bottom_y - post_bottom_y) * math.tan(lean)
    wedge = (
        cadquery.Workplane("XY")
        .polyline([
            (front_at_plate_bottom, plate_bottom_y),
            (top_front_x, post_top_y),
            (top_front_x + p["plate_height_y"] * math.sin(tilt),
             plate_bottom_y + p["plate_height_y"] * math.cos(tilt)),
        ]).close()
        .extrude(p["post_width_z"] / 2.0, both = True)
    )

    bracket = foot.union(legs).union(cross_bar).cut(pocket).union(post).union(wedge).union(top_plate)
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
    shift = geometry["post_top_shift_mm"]
    centre_y = geometry["raise_mm"]
    pivot = (post_front_x + shift, centre_y - half_height)
    angle = -geometry["plate_tilt_degrees"]

    figure, axis = pyplot.subplots(figsize = (6, 8))
    legs_height = p["open_above_y"] + half_height
    axis.add_patch(pyplot.Rectangle((0, -half_height), p["foot_thickness_x"], p["plate_height_y"], color = "tab:blue"))
    axis.add_patch(pyplot.Rectangle((0, -half_height), post_front_x, legs_height, color = "tab:blue"))
    axis.add_patch(pyplot.Rectangle((0, -half_height), p["pocket_depth_x"], p["plate_height_y"], color = "white", alpha = 0.5,
                                    hatch = "//", edgecolor = "grey"))
    axis.text(1, -half_height - 3, "open in the middle\nfor the plate's boss", fontsize = 7, color = "grey", va = "top")
    post_top_y = centre_y + half_height
    axis.add_patch(pyplot.Polygon([
        (p["arm_length_x"], p["open_above_y"]), (post_front_x, p["open_above_y"]),
        (post_front_x + shift, post_top_y), (p["arm_length_x"] + shift, post_top_y)], color = "tab:blue"))
    axis.add_patch(pyplot.Rectangle(pivot, p["top_plate_thickness_x"], p["plate_height_y"],
                                    angle = angle, rotation_point = pivot, color = "tab:blue"))
    # Camera as today: its lens face lies along x (53 long), the case 31 tall
    # above it; drawn in the plate's frame and rotated with the plate.
    case_x = post_front_x + shift + p["top_plate_thickness_x"] + p["ear_thickness"]
    axis.add_patch(pyplot.Rectangle((case_x, centre_y - half_height), 53.1, 31.0,
                                    angle = angle, rotation_point = pivot, color = "black", alpha = 0.6))
    lens_x = post_front_x + shift + p["top_plate_thickness_x"] + p["lens_today_ahead_of_brace_face"]
    lens_y = centre_y + p["lens_today_above_pattern_centre"]
    a = math.radians(angle)
    lx = pivot[0] + (lens_x - pivot[0]) * math.cos(a) - (lens_y - pivot[1]) * math.sin(a)
    ly = pivot[1] + (lens_x - pivot[0]) * math.sin(a) + (lens_y - pivot[1]) * math.cos(a)
    axis.plot(lx, ly, "o", color = "red")
    look = math.radians(geometry["lens_tilt_degrees_forward"])
    axis.annotate("", xy = (lx + 40 * math.sin(look), ly - 40 * math.cos(look)), xytext = (lx, ly),
                  arrowprops = {"arrowstyle": "->", "color": "red"})
    axis.text(lx + 3, ly + 3, f"lens: looks down,\n{p['lens_tilt_back_degrees']:.0f} deg back", color = "red", fontsize = 8)
    axis.axvline(0, color = "grey", linestyle = "--")
    axis.text(-1.5, 125, "metal plate face (wrist)", rotation = 90, ha = "right", va = "top", color = "grey")
    axis.axvline(-9.0, color = "grey", linestyle = ":")
    axis.text(-10.5, 125, "rod axis (Niryo STEP)", rotation = 90, ha = "right", va = "top", color = "grey", fontsize = 8)
    axis.set_xlim(-30, 120)
    axis.set_ylim(-35, 130)
    axis.set_aspect("equal")
    axis.set_xlabel("x: forward from the metal plate face (mm)")
    axis.set_ylabel("y: up from the screw-pattern centre (mm)")
    axis.set_title(f"Bracket {stem[-2:]}: plate {centre_y:.0f} mm up, post leaning {p.get('post_lean_degrees', 0):.0f} deg, "
                   f"top {post_front_x + shift:.0f} mm forward, plate {-angle:.0f} deg", fontsize = 9)
    path = OUTPUT_DIRECTORY / f"{stem}_side_view.png"
    figure.tight_layout()
    figure.savefig(path, dpi = 150)
    return str(path)


if __name__ == "__main__":
    # F2 = the corrective version of 2.11.26 (leaning post); the printed F keeps its files.
    STEM = "ned2_camera_bracket_F2"
    parameters = dict(DEFAULT_BRACKET_PARAMETERS)
    bracket = build_bracket(parameters)
    summary = export_bracket(bracket, parameters, stem = STEM)
    summary["side_view"] = draw_side_view(parameters, stem = STEM)
    print(json.dumps({key: value for key, value in summary.items() if key != "parameters"}, indent = 2))
