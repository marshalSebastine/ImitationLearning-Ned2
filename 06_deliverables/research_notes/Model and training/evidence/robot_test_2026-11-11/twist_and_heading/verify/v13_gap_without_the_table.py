"""
Verifier 13: how far the forearm was behind the demonstrations at the first bend of sheet 03, measured WITHOUT comparing
places on the table (the sheet and the tip did not start where the demonstrations started: see v06).
Forearm turn since the start, read at the first picture where the tip is a given straight-line distance from its start.
Demonstrations: the eight standard starts from end A. Robot: runs 5, 9, 10. Also the rule the question asked about,
applied to the logged answers of the first ten pictures: which way would "turn towards the predicted travel" have turned?
"""
import numpy
from vcommon import demos, run_log

frames = demos()
reference = frames[(frames["sheet"] == "path_03") & (frames["start_kind"] == "STD") & (frames["start_end"] == "A")]


def turn_at(x, y, j4, distances):
    net = numpy.hypot(x - x[0], y - y[0])
    out = []
    for distance in distances:
        hit = numpy.flatnonzero(net >= distance)
        out.append(float(j4[hit[0]] - j4[0]) if len(hit) else numpy.nan)
    return out


distances = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
demo_turns = numpy.array([turn_at(100 * e["tip_x_m"].to_numpy(), 100 * e["tip_y_m"].to_numpy(), e["j4"].to_numpy(), distances) for _, e in reference.groupby("episode_id")])
print("forearm turn since the start when the tip is first d cm (straight line) from its start")
print("   d cm:            " + "  ".join(f"{d:6.0f}" for d in distances))
print("   demonstrations:  " + "  ".join(f"{v:+6.0f}" for v in numpy.nanmedian(demo_turns, axis = 0)) + "   (median of 8)")
print("      smallest turn:" + "  ".join(f"{v:+6.0f}" for v in numpy.nanmax(demo_turns, axis = 0)))
print("      largest turn: " + "  ".join(f"{v:+6.0f}" for v in numpy.nanmin(demo_turns, axis = 0)))
for number in (5, 9, 10):
    s = run_log(number)
    robot = turn_at(100 * s["tip_x"].to_numpy(), 100 * s["tip_y"].to_numpy(), s["j4"].to_numpy(), distances)
    print(f"   run {number:2d}:          " + "  ".join(f"{v:+6.0f}" for v in robot) + "   behind the median by " + "  ".join(f"{v - m:+.0f}" for v, m in zip(robot, numpy.nanmedian(demo_turns, axis = 0))))

print("\nthe rule 'turn the forearm towards the predicted direction of travel' on the logged answers of the first ten pictures")
print("(a twist of +1 degree turns the picture heading by -0.9 degrees, so turning towards a travel direction on the picture-right, angle < 0, needs a POSITIVE twist)")
for number in (3, 5, 9, 10):
    s = run_log(number).iloc[:10]
    wrong = numpy.mean(numpy.sign(-s["model_angle_deg"]) != numpy.sign(s["model_twist_deg"]))
    print(f"   run {number:2d}: mean asked travel angle {s['model_angle_deg'].mean():+.0f}, mean asked twist {s['model_twist_deg'].mean():+.1f}; the rule's twist would have the opposite sign to the model's twist in {wrong:.1f} of the ten pictures")
