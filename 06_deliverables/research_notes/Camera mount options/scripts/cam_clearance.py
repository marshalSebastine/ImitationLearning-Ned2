"""
Clearance of the proposed lens positions (E1, E2, F) from the forearm, the wrist housing and the
puck over the pushing radii and, for the hand mounts, over the joint-6 pan range.

Usage: py -3.13 cam_clearance.py
"""

import json
import math
import os

import numpy

import camgeo
import cam_options

HERE = os.path.dirname(os.path.abspath(__file__))


def point_to_segment_distance(point, start, end) -> float:
    axis = end - start
    length = numpy.linalg.norm(axis)
    t = float(numpy.clip((point - start) @ axis / (length ** 2), 0.0, 1.0))
    return float(numpy.linalg.norm(point - (start + t * axis)))


def clearances(camera: camgeo.Camera, joints, puck_heading_rad: float = 0.0) -> dict:
    frame = camgeo.frames(joints)
    lens = camera.pose(joints)[0]
    forearm = camgeo.arm_cylinders(joints)[0]
    housing = camgeo.pusher_cylinders(joints)[3]
    tip = frame["tool"][0]
    table_z = camgeo.table_height(tip[0], tip[1])
    puck_centre = tip[:2] + cam_options.CONFIG["puck_offset_m"] * numpy.array([math.cos(puck_heading_rad), math.sin(puck_heading_rad)])
    puck_top = numpy.array([puck_centre[0], puck_centre[1], table_z + camgeo.GEOMETRY["puck_height_m"]])
    horizontal = float(numpy.hypot(lens[0] - puck_centre[0], lens[1] - puck_centre[1]))
    return {"forearm_surface_cm": round(100 * (point_to_segment_distance(lens, forearm["start"], forearm["end"]) - forearm["radius"]), 1),
            "housing_surface_cm": round(100 * (point_to_segment_distance(lens, housing["start"], housing["end"]) - housing["radius"]), 1),
            "lens_above_puck_top_cm": round(100 * (lens[2] - puck_top[2]), 1),
            "lens_horizontal_from_puck_centre_cm": round(horizontal, 1),
            "lens_height_cm": round(100 * (lens[2] - table_z), 1)}


def main() -> None:
    results = json.load(open(os.path.join(HERE, "cam_options_results.json")))
    options = results["options"]["52"]
    designs = {
        "E1_collar": {"hfov_deg": 52.0, "mount": "hand", "offset_m": (0.030, 0.0, 0.0), "tilt_deg": options["E1_collar"]["design"]["tilt_deg"]},
        "E2_arm": {"hfov_deg": 52.0, "mount": "hand", "offset_m": (0.035, 0.040, 0.0), "tilt_deg": options["E2_arm"]["design"]["tilt_deg"]},
        "F_best": options["F"]["best_design"]["parameters"]
    }
    out = {}
    for name, parameters in designs.items():
        camera = camgeo.Camera(parameters)
        rows = {}
        for radius in (0.14, 0.22, 0.30, 0.38, 0.42):
            pose = cam_options.pose_vertical(radius)
            pans = [0, 90, -90, 145, -145] if parameters["mount"] == "hand" else [0]
            worst = None
            for pan in pans:
                joints = list(pose["joints"]); joints[5] = math.radians(pan)
                c = clearances(camera, joints, puck_heading_rad = math.radians(-pan))
                c["pan_deg"] = pan
                if worst is None or c["forearm_surface_cm"] < worst["forearm_surface_cm"]:
                    worst = c
            rows[radius] = worst
            print(f"{name} r={radius}: worst over pans -> {worst}")
        out[name] = rows
    with open(os.path.join(HERE, "cam_clearance.json"), "w") as handle:
        json.dump(out, handle, indent = 1)


if __name__ == "__main__":
    main()
