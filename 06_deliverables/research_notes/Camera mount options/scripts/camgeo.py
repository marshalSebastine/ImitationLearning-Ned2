"""
Arm and camera geometry for the camera-mount comparison of the puck-pushing task.

The forward kinematics come from ``ned2_fk.py`` (validated against recorded poses on 25.10.26,
0.6 mm worst case). This module adds an inverse-kinematics solver with joint limits, a pinhole
camera that can be bolted to the wrist link or to the hand link, the table window seen by that
camera, and visibility tests for the puck and the line ahead of it that account for the puck,
the pusher and the arm standing in the way.

Conventions (robot base frame): x forward from the base, z up. With the rod vertical the wrist
frame has x pointing radially away from the base, y up and z to the right; the hand frame has
x radially outward at joint 6 = 0, y to the right and z down along the rod.
"""

import math

import numpy
from scipy.optimize import least_squares

import ned2_fk


GEOMETRY = {
    "table_plane": {"a": 0.0090, "b": -0.0055, "c": 0.0047},   # measured 25.10.26 (config.py)
    "pusher_clearance_above_sheet_m": 0.005,                   # measured (config.py)
    "extension_below_rod_end_m": 0.026,                        # measured: magnet tip below the rod end
    "magnet_radius_m": 0.0065,                                 # measured: tip 1.3 cm wide
    "magnet_length_m": 0.010,                                  # assumed: silver magnet cap length
    "block_radius_m": 0.009,                                   # measured: block 1.8 cm wide
    "rod_radius_m": 0.009,                                     # measured: bare rod 1.8 cm wide
    "rod_length_m": 0.025,                                     # measured: bare rod 2.5 cm long
    "housing_radius_m": 0.023,                                 # assumed: wrist housing about 4.6 cm wide
    "housing_height_above_hand_m": 0.075,                      # assumed: housing from the hand origin up past the wrist axis
    "forearm_radius_m": 0.030,                                 # assumed: forearm about 6 cm wide
    "puck_radius_m": 0.012,                                    # measured
    "puck_height_m": 0.029,                                    # measured
    "puck_offset_from_rod_axis_m": 0.0185,                     # measured: puck radius + magnet radius
    "image_width_px": 640,
    "image_height_px": 480,
    "hand_tool_m": 0.0285                                      # URDF distance_hand_tool
}

JOINT_LIMITS_RAD = [
    (math.radians(low), math.radians(high))
    for low, high in [(-171.88, 171.88), (-105.0, 34.96), (-76.78, 85.5),
                      (-119.75, 119.75), (-110.01, 110.17), (-144.96, 144.96)]
]



def table_height(
        x: float,
        y: float
) -> float:
    """Table surface height in metres at (x, y) from the plane fitted to the four taught corners."""

    plane = GEOMETRY["table_plane"]
    return plane["a"] * x + plane["b"] * y + plane["c"]



def rod_end_height(
        x: float,
        y: float
) -> float:
    """Height of the reported tool point (rod end) while pushing at (x, y)."""

    return (table_height(x, y) + GEOMETRY["pusher_clearance_above_sheet_m"]
            + GEOMETRY["extension_below_rod_end_m"])



def frames(joints) -> dict:
    """Forward kinematics as numpy arrays: frame name -> (origin, rotation matrix)."""

    raw = ned2_fk.fk(list(joints))
    return {name: (numpy.array(p), numpy.array(R)) for name, (p, R) in raw.items()}



def rod_axis_down(joints) -> numpy.ndarray:
    """Unit vector along the rod from the wrist towards the magnet tip (hand z)."""

    return frames(joints)["joint_6"][1][:, 2]



def rod_tilt_deg(joints) -> float:
    """Angle of the rod from straight down, in degrees."""

    z = rod_axis_down(joints)[2]
    return math.degrees(math.acos(max(-1.0, min(1.0, -z))))



def solve_joints(
        target_tip,
        rod_direction = (0.0, 0.0, -1.0),
        fixed: dict = None,
        starts = None,
        position_weight: float = 1_000.0,
        direction_weight: float = 100.0
) -> tuple:
    """
    Inverse kinematics with joint limits.

    Finds joint angles that place the rod end at ``target_tip`` (metres) with the rod pointing
    along ``rod_direction``; ``fixed`` maps joint index (0 to 5) to a value held constant. Several
    starting guesses are tried and the best least-squares solution is returned together with the
    tip position error in millimetres and the angle between the achieved and requested rod
    directions in degrees. The elbow is kept above the base (joint_3 origin z > 0) by rejecting
    the folded branch.

    Returns
    -------
    tuple
        (joints as a list of six floats, tip error in mm, direction error in deg).
    """

    fixed = dict(fixed or {})
    free = [index for index in range(6) if index not in fixed]
    target = numpy.asarray(target_tip, dtype = float)
    wanted = numpy.asarray(rod_direction, dtype = float)
    wanted = wanted / numpy.linalg.norm(wanted)
    base_turn = math.atan2(target[1], target[0])

    def assemble(values) -> list:
        joints = [0.0] * 6
        for index, value in zip(free, values):
            joints[index] = float(value)
        for index, value in fixed.items():
            joints[index] = float(value)
        return joints

    def residual(values) -> numpy.ndarray:
        joints = assemble(values)
        frame = ned2_fk.fk(joints)
        tip = numpy.array(frame["tool"][0])
        axis = numpy.array(frame["joint_6"][1])[:, 2]
        return numpy.concatenate([position_weight * (tip - target), direction_weight * (axis - wanted)])

    if starts is None:
        starts = []
        for s2 in (-0.8, -1.2, -0.5):
            for s3 in (-1.0, -0.4, 0.3, 0.9):
                for s5 in (-0.8, 0.0, 0.4, -1.4):
                    starts.append([base_turn, s2, s3, 0.0, s5, 0.0])
    lower = [JOINT_LIMITS_RAD[index][0] for index in free]
    upper = [JOINT_LIMITS_RAD[index][1] for index in free]
    best = None
    for start in starts:
        guess = numpy.clip([start[index] for index in free], lower, upper)
        solution = least_squares(residual, guess, bounds = (lower, upper), xtol = 1e-10, ftol = 1e-10)
        joints = assemble(solution.x)
        if frames(joints)["joint_3"][0][2] <= 0.0:
            continue
        if best is None or solution.cost < best[0].cost:
            best = (solution, joints)
    if best is None:
        return None, float("inf"), float("inf")
    solution, joints = best
    frame = frames(joints)
    tip_error_mm = 1_000.0 * numpy.linalg.norm(frame["tool"][0] - target)
    axis = frame["joint_6"][1][:, 2]
    direction_error_deg = math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ wanted)))))
    return joints, float(tip_error_mm), float(direction_error_deg)



def rotation_about(axis, angle_rad: float) -> numpy.ndarray:
    """Rodrigues rotation matrix about a unit axis."""

    axis = numpy.asarray(axis, dtype = float)
    axis = axis / numpy.linalg.norm(axis)
    cross = numpy.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (math.cos(angle_rad) * numpy.eye(3) + math.sin(angle_rad) * cross
            + (1 - math.cos(angle_rad)) * numpy.outer(axis, axis))



class Camera:
    """
    Pinhole camera bolted to the wrist link or to the hand link.

    Parameters
    ----------
    parameters : dict
        ``hfov_deg`` horizontal field of view; ``mount`` "wrist" or "hand"; ``offset_m``
        (forward, up, right) of the lens from the mount frame origin, in the mount frame with the
        rod vertical; ``tilt_deg`` of the optical axis from straight down towards forward;
        ``yaw_deg`` of the optical axis about the vertical, positive towards the right;
        ``roll_deg`` about the optical axis.
    """

    DEFAULTS = {"hfov_deg": 52.0, "mount": "wrist", "offset_m": (0.0325, 0.0, 0.0),
                "tilt_deg": 10.0, "yaw_deg": 0.0, "roll_deg": 0.0}


    def __init__(
            self,
            parameters: dict = None
    ):

        self.parameters = dict(self.DEFAULTS)
        self.parameters.update(parameters or {})
        self.width = GEOMETRY["image_width_px"]
        self.height = GEOMETRY["image_height_px"]
        self.focal_px = (self.width / 2.0) / math.tan(math.radians(self.parameters["hfov_deg"]) / 2.0)


    def vfov_deg(self) -> float:
        """Vertical field of view for a 4:3 picture with square pixels."""

        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.focal_px))


    def pose(
            self,
            joints
    ) -> tuple:
        """
        Lens position and camera axes in the base frame.

        Returns
        -------
        tuple
            (position, right, up, forward) as numpy vectors; "up" is the direction of the top of
            the picture, "forward" the optical axis.
        """

        frame = frames(joints)
        if self.parameters["mount"] == "wrist":
            origin, rotation = frame["joint_5"]
            forward_axis, up_axis, right_axis = rotation[:, 0], rotation[:, 1], rotation[:, 2]
        else:
            origin, rotation = frame["joint_6"]
            forward_axis, up_axis, right_axis = rotation[:, 0], -rotation[:, 2], rotation[:, 1]
        offset = self.parameters["offset_m"]
        position = origin + offset[0] * forward_axis + offset[1] * up_axis + offset[2] * right_axis
        tilt = math.radians(self.parameters["tilt_deg"])
        forward = -math.cos(tilt) * up_axis + math.sin(tilt) * forward_axis
        up = math.cos(tilt) * forward_axis + math.sin(tilt) * up_axis
        yaw = rotation_about(up_axis, -math.radians(self.parameters["yaw_deg"]))
        forward, up = yaw @ forward, yaw @ up
        right = numpy.cross(forward, up)
        roll = rotation_about(forward, math.radians(self.parameters["roll_deg"]))
        up, right = roll @ up, roll @ right
        return position, right, up, forward


    def project(
            self,
            pose: tuple,
            points: numpy.ndarray
    ) -> tuple:
        """Pixel columns, rows and depths of world points (N x 3)."""

        position, right, up, forward = pose
        vectors = numpy.atleast_2d(points) - position
        depth = vectors @ forward
        safe = numpy.where(numpy.abs(depth) < 1e-9, 1e-9, depth)
        columns = self.width / 2.0 + self.focal_px * (vectors @ right) / safe
        rows = self.height / 2.0 - self.focal_px * (vectors @ up) / safe
        return columns, rows, depth


    def in_picture(
            self,
            pose: tuple,
            points: numpy.ndarray
    ) -> numpy.ndarray:
        """Boolean mask of world points that fall inside the picture and in front of the lens."""

        columns, rows, depth = self.project(pose, points)
        return (depth > 0) & (columns >= 0) & (columns < self.width) & (rows >= 0) & (rows < self.height)


    def table_point(
            self,
            pose: tuple,
            column: float,
            row: float,
            table_z: float
    ) -> numpy.ndarray:
        """Where the ray through a pixel meets the horizontal plane z = table_z (NaN if it does not)."""

        position, right, up, forward = pose
        ray = (forward + (column - self.width / 2.0) / self.focal_px * right
               - (row - self.height / 2.0) / self.focal_px * up)
        if ray[2] >= -1e-9:
            return numpy.array([numpy.nan, numpy.nan, numpy.nan])
        scale = (table_z - position[2]) / ray[2]
        return position + scale * ray


    def table_window(
            self,
            pose: tuple,
            table_z: float,
            ground_point,
            azimuth_rad: float
    ) -> dict:
        """
        The patch of table inside the picture, measured from ``ground_point`` along ``azimuth_rad``.

        Returns near and far edge distances (bottom-centre and top-centre pixels), widths at the
        bottom and top edges, and the sideways offset of the bottom-centre pixel, all in metres.
        A far edge above the horizon is reported as infinity.
        """

        ahead = numpy.array([math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0])
        sideways = numpy.array([-math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0])
        origin = numpy.array([ground_point[0], ground_point[1], table_z])

        def along(column, row):
            point = self.table_point(pose, column, row, table_z)
            if numpy.isnan(point[0]):
                return math.inf, math.nan
            return float((point - origin) @ ahead), float((point - origin) @ sideways)

        near, near_side = along(self.width / 2.0, self.height - 1)
        far, far_side = along(self.width / 2.0, 0)
        bottom_left = self.table_point(pose, 0, self.height - 1, table_z)
        bottom_right = self.table_point(pose, self.width - 1, self.height - 1, table_z)
        top_left = self.table_point(pose, 0, 0, table_z)
        top_right = self.table_point(pose, self.width - 1, 0, table_z)
        near_width = float(numpy.linalg.norm(bottom_right - bottom_left))
        far_width = float(numpy.linalg.norm(top_right - top_left)) if not numpy.isnan(top_left[0]) else math.inf
        return {"near_m": near, "far_m": far, "near_width_m": near_width, "far_width_m": far_width,
                "near_sideways_m": near_side, "far_sideways_m": far_side}



def cylinder(
        start,
        end,
        radius: float
) -> dict:
    """A finite cylinder between two axis points."""

    return {"start": numpy.asarray(start, dtype = float), "end": numpy.asarray(end, dtype = float),
            "radius": float(radius)}



def pusher_cylinders(joints) -> list:
    """Magnet, block, rod and wrist housing as cylinders around the rod axis."""

    frame = frames(joints)
    hand_origin = frame["joint_6"][0]
    down = frame["joint_6"][1][:, 2]
    rod_end = hand_origin + GEOMETRY["hand_tool_m"] * down
    magnet_tip = rod_end + GEOMETRY["extension_below_rod_end_m"] * down
    magnet_top = magnet_tip - GEOMETRY["magnet_length_m"] * down
    return [
        cylinder(magnet_top, magnet_tip, GEOMETRY["magnet_radius_m"]),
        cylinder(rod_end, magnet_top, GEOMETRY["block_radius_m"]),
        cylinder(hand_origin, rod_end, GEOMETRY["rod_radius_m"]),
        cylinder(hand_origin - GEOMETRY["housing_height_above_hand_m"] * down, hand_origin,
                 GEOMETRY["housing_radius_m"])
    ]



def arm_cylinders(joints) -> list:
    """Forearm between the elbow roll joint and the wrist axis."""

    frame = frames(joints)
    return [cylinder(frame["joint_4"][0], frame["joint_5"][0], GEOMETRY["forearm_radius_m"])]



def puck_cylinder(
        centre_xy,
        table_z: float
) -> dict:
    """The puck standing on the table."""

    base = numpy.array([centre_xy[0], centre_xy[1], table_z])
    top = base + numpy.array([0.0, 0.0, GEOMETRY["puck_height_m"]])
    return cylinder(base, top, GEOMETRY["puck_radius_m"])



def blocked(
        eye: numpy.ndarray,
        points: numpy.ndarray,
        cylinders: list,
        samples: int = 48,
        shrink: float = 0.002
) -> numpy.ndarray:
    """
    Boolean mask of world points whose line of sight from ``eye`` passes through a cylinder.

    The sight line is sampled; a point counts as blocked when a sample lies inside a cylinder.
    The last ``shrink`` metres before the point are ignored so a point on a cylinder's own surface
    does not block itself.
    """

    points = numpy.atleast_2d(points)
    fractions = numpy.linspace(0.0, 1.0, samples)[None, :, None]
    lengths = numpy.linalg.norm(points - eye, axis = 1)[:, None, None]
    fractions = fractions * numpy.clip(1.0 - shrink / numpy.maximum(lengths, 1e-9), 0.0, 1.0)
    samples_xyz = eye[None, None, :] + fractions * (points - eye)[:, None, :]
    mask = numpy.zeros(points.shape[0], dtype = bool)
    for body in cylinders:
        axis = body["end"] - body["start"]
        length = numpy.linalg.norm(axis)
        if length < 1e-9:
            continue
        axis = axis / length
        relative = samples_xyz - body["start"]
        along = relative @ axis
        radial = numpy.linalg.norm(relative - along[..., None] * axis, axis = 2)
        inside = (along >= 0.0) & (along <= length) & (radial < body["radius"])
        mask |= inside.any(axis = 1)
    return mask



def line_ahead_points(
        puck_centre_xy,
        heading_rad: float,
        table_z: float,
        length_m: float = 0.20,
        step_m: float = 0.0025
) -> numpy.ndarray:
    """Points along a straight line on the table from the puck centre in the push direction."""

    distances = numpy.arange(0.0, length_m + 1e-9, step_m)
    return numpy.stack([puck_centre_xy[0] + distances * math.cos(heading_rad),
                        puck_centre_xy[1] + distances * math.sin(heading_rad),
                        numpy.full_like(distances, table_z)], axis = 1)



def puck_rim_points(
        centre_xy,
        table_z: float,
        count: int = 24
) -> numpy.ndarray:
    """Points on the puck's top rim and bottom rim."""

    angles = numpy.linspace(0.0, 2.0 * math.pi, count, endpoint = False)
    ring = numpy.stack([centre_xy[0] + GEOMETRY["puck_radius_m"] * numpy.cos(angles),
                        centre_xy[1] + GEOMETRY["puck_radius_m"] * numpy.sin(angles)], axis = 1)
    top = numpy.column_stack([ring, numpy.full(count, table_z + GEOMETRY["puck_height_m"])])
    bottom = numpy.column_stack([ring, numpy.full(count, table_z)])
    return numpy.vstack([top, bottom])



def view_at_heading(
        camera: Camera,
        joints,
        heading_rad: float,
        include_arm: bool = True,
        line_length_m: float = 0.20,
        line_step_m: float = 0.0025,
        edge_margin_px: float = 12.0
) -> dict:
    """
    What the camera sees of the puck and of the line ahead for one push heading.

    The rod stands at its current tip position; the puck touches the magnet in the push direction
    and the line runs straight on from the puck centre. Headings are absolute (radians in the base
    frame). Returns the visible length of the line ahead, the furthest visible line point, the
    nearest visible line point, and the fraction of puck rim points that are in the picture and
    not hidden by the pusher or the arm.
    """

    frame = frames(joints)
    tip = frame["tool"][0]
    ground = numpy.array([tip[0], tip[1]])
    table_z = table_height(tip[0], tip[1])
    puck_centre = ground + GEOMETRY["puck_offset_from_rod_axis_m"] * numpy.array([math.cos(heading_rad), math.sin(heading_rad)])
    pose = camera.pose(joints)
    eye = pose[0]
    occluders = pusher_cylinders(joints) + (arm_cylinders(joints) if include_arm else [])

    line = line_ahead_points(puck_centre, heading_rad, table_z, line_length_m, line_step_m)
    line_visible = camera.in_picture(pose, line) & ~blocked(eye, line, occluders + [puck_cylinder(puck_centre, table_z)])
    distances = numpy.arange(0.0, line_length_m + 1e-9, line_step_m)
    visible_length = float(line_visible.sum() * line_step_m)
    furthest = float(distances[line_visible].max()) if line_visible.any() else 0.0
    nearest = float(distances[line_visible].min()) if line_visible.any() else math.nan

    rim = puck_rim_points(puck_centre, table_z)
    rim_visible = camera.in_picture(pose, rim) & ~blocked(eye, rim, occluders)
    # the puck's leading edge: the top-rim point in the push direction, needed with a margin inside the picture
    edge = numpy.array([[puck_centre[0] + GEOMETRY["puck_radius_m"] * math.cos(heading_rad),
                         puck_centre[1] + GEOMETRY["puck_radius_m"] * math.sin(heading_rad),
                         table_z + GEOMETRY["puck_height_m"]]])
    columns, rows, depth = camera.project(pose, edge)
    margin = edge_margin_px
    edge_visible = bool(depth[0] > 0 and margin <= columns[0] < camera.width - margin and margin <= rows[0] < camera.height - margin
                        and not blocked(eye, edge, occluders)[0])
    return {"line_visible_m": visible_length, "line_furthest_m": furthest, "line_nearest_m": nearest,
            "puck_fraction": float(rim_visible.mean()), "puck_top_fraction": float(rim_visible[:24].mean()),
            "puck_edge_visible": edge_visible, "puck_edge_row": float(rows[0]), "puck_edge_column": float(columns[0])}
