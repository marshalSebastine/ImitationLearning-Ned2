"""
Clearance of brackets F (printed) and F2 (leaning post, 2.11.26) against Niryo's forearm mesh, per pushing pose.

Uses the design review's machinery (review_e1_body.py: Niryo's meshes in the hand frame, a signed-distance grid) and
the bracket geometry of bracket_design/bracket_f_design.py. For each rod distance x (rod vertical at pushing height) the
wrist angle the robot needs is solved, and the smallest distance between the forearm and (a) the post's back face,
(b) the whole post and top plate, (c) the camera body is printed. A negative number means the parts overlap.
Frame mapping: the bracket's design frame has its origin at the brace pattern centre on the plate face, 9 mm ahead of
the rod axis (lens today 25 ahead of the rod axis, 16 ahead of the plate face) and 6 mm above the joint-5 axis, which
sits 19.7 mm above the hand face in the review's hand frame (z down). Run from this folder with the project venv:
    ..\\..\\..\\..\\.venv\\Scripts\\python.exe review_f2_clearance.py
"""

import math
import pathlib
import sys

import numpy

import camgeo
import cam_options as co
import review_e1_body as rb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "bracket_design"))
import bracket_f_design as bf

PLATE_FACE_AHEAD_OF_ROD_AXIS_MM = 9.0
PATTERN_CENTRE_ABOVE_JOINT5_MM = 6.0
JOINT5_ABOVE_HAND_FACE_MM = 19.7
CAMERA_BODY_DEPTH_MM = 25.0
ROD_DISTANCES_M = [0.13, 0.14, 0.16, 0.20, 0.42]

rb.XR, rb.YR, rb.ZR = (-95, 150), (-95, 95), (-150, 75)


def design_to_hand(points_design: numpy.ndarray) -> numpy.ndarray:
    """Design frame (x forward from the plate face, y up, z sideways) to the review's hand frame (X ahead, Y side, z down)."""

    x, y, z = points_design[:, 0], points_design[:, 1], points_design[:, 2]
    return numpy.stack([PLATE_FACE_AHEAD_OF_ROD_AXIS_MM + x, z,
                        -(JOINT5_ABOVE_HAND_FACE_MM + PATTERN_CENTRE_ABOVE_JOINT5_MM) - y], axis = 1)


def bracket_points(parameters: dict, spacing: float = 1.5) -> dict:
    """Sample points of the post's back face, the post and top plate, and the camera body, in the hand frame."""

    p = dict(bf.DEFAULT_BRACKET_PARAMETERS)
    p.update(parameters)
    geometry = bf.derived_geometry(p)
    half_height = p["plate_height_y"] / 2.0
    post_back = p["arm_length_x"]
    post_front = post_back + p["post_thickness_x"]
    bottom = p["open_above_y"]
    top = geometry["raise_mm"] + half_height
    shift = geometry["post_top_shift_mm"]
    half_width = p["post_width_z"] / 2.0

    ys = numpy.arange(bottom, top + 1e-6, spacing)
    zs = numpy.arange(-half_width, half_width + 1e-6, spacing)
    fraction = (ys - bottom) / (top - bottom)
    back_x = post_back + fraction * shift
    front_x = post_front + fraction * shift
    back_face = numpy.array([(bx, y, z) for bx, y in zip(back_x, ys) for z in zs])
    front_face = numpy.array([(fx, y, z) for fx, y in zip(front_x, ys) for z in zs])

    # Top plate: 22 mm tall, 6 mm thick, rotated forward about its bottom edge at (post_front + shift, raise - half).
    tilt = math.radians(geometry["plate_tilt_degrees"])
    pivot = numpy.array([post_front + shift, geometry["raise_mm"] - half_height])
    plate = []
    for along in numpy.arange(0.0, p["plate_height_y"] + 1e-6, spacing):
        for thick in (0.0, p["top_plate_thickness_x"]):
            local = numpy.array([thick, along])
            rotated = numpy.array([local[0] * math.cos(tilt) + local[1] * math.sin(tilt),
                                   -local[0] * math.sin(tilt) + local[1] * math.cos(tilt)])
            for z in numpy.arange(-p["plate_width_z"] / 2.0, p["plate_width_z"] / 2.0 + 1e-6, spacing):
                plate.append((pivot[0] + rotated[0], pivot[1] + rotated[1], z))
    plate = numpy.array(plate)

    camera = rb.camera_points(geometry["lens_ahead_of_rod_axis_mm"],
                              -(JOINT5_ABOVE_HAND_FACE_MM + geometry["lens_above_joint5_axis_mm"]),
                              geometry["lens_tilt_degrees_forward"], CAMERA_BODY_DEPTH_MM, cable = True)
    return {"post back face": design_to_hand(back_face),
            "post and plate": numpy.vstack([design_to_hand(back_face), design_to_hand(front_face), design_to_hand(plate)]),
            "camera body": camera}


def forearm_grid(joints, meshes):
    """Signed-distance grid of the forearm mesh alone, in the hand frame of this pose."""

    frames = camgeo.frames(joints)
    origin, rotation = frames["joint_4"]
    triangles = meshes["forearm"] @ rotation.T + origin * rb.MM
    keep = (numpy.linalg.norm(triangles - frames["joint_6"][0] * rb.MM, axis = 2) < 200).any(axis = 1)
    points = rb.to_hand_frame(rb.surface_points(triangles[keep]).reshape(-1, 3), frames["joint_6"])
    return rb.signed_distance_grid(points)


def main() -> None:
    meshes = rb.load_meshes()
    variants = {"F (printed, straight post)": {"post_lean_degrees": 0.0, "lens_tilt_back_degrees": 10.0},
                "F2 (post leaning 20 deg)": {"post_lean_degrees": 20.0, "lens_tilt_back_degrees": 20.0},
                "F2b (post leaning 25 deg)": {"post_lean_degrees": 25.0, "lens_tilt_back_degrees": 22.0},
                "F2c (lean 25, camera 30 back)": {"post_lean_degrees": 25.0, "lens_tilt_back_degrees": 30.0}}
    parts = {name: bracket_points(parameters) for name, parameters in variants.items()}

    print("clearance to the forearm mesh in mm (negative = overlap); wrist = joint 5 the pose needs")
    for x in ROD_DISTANCES_M:
        pose = co.pose_vertical(x)
        joints = pose["joints"]
        if not joints:
            print(f"x = {x:.2f} m: no pose")
            continue
        grid = forearm_grid(joints, meshes)
        line = f"x = {x:.2f} m, wrist {math.degrees(joints[4]):+6.1f} deg:"
        for name, pieces in parts.items():
            clearances = {part: float(rb.sample_grid(grid, pts).min()) for part, pts in pieces.items()}
            line += f"   {name.split(' ')[0]}: back {clearances['post back face']:6.1f}, post+plate {clearances['post and plate']:6.1f}, camera {clearances['camera body']:6.1f}"
        print(line)


if __name__ == "__main__":
    main()
