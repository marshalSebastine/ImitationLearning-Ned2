"""
Verifier 18: does the step the models ASKED for at the first bend of sheet 03 add up to the move the demonstrations made?
In the frame of the first picture (forward = the first picture's top, left = its left): the tip's net move from the start to
the picture where joint 4 reaches -70 degrees.
    demonstrations   measured from the tip positions (eight standard starts from end A);
    robot, done      measured from the logged tip positions (1-cm steps);
    robot, asked     calculated: the logged answers carried out with their own length (factor 1.0 and 1.3) along the
                     logged headings (first order: the pictures would have been different after the first step).
"""
import numpy
from vcommon import demos, run_log

frames = demos()
reference = frames[(frames["sheet"] == "path_03") & (frames["start_kind"] == "STD") & (frames["start_end"] == "A")]


def in_first_frame(dx, dy, heading_deg):
    h = numpy.radians(heading_deg)
    return dx * numpy.cos(h) + dy * numpy.sin(h), -dx * numpy.sin(h) + dy * numpy.cos(h)


print("net tip move from the start to the picture where joint 4 reaches -70 degrees, in the first picture's frame (cm): forward, left (negative = right), length")
rows = []
for _, e in reference.groupby("episode_id"):
    e = e.reset_index(drop = True)
    k = int(numpy.flatnonzero(e["j4"].to_numpy() <= -70.0)[0])
    f, l = in_first_frame(100 * (e["tip_x_m"].iloc[k] - e["tip_x_m"].iloc[0]), 100 * (e["tip_y_m"].iloc[k] - e["tip_y_m"].iloc[0]), e["picture_heading_deg"].iloc[0])
    rows.append((f, l))
rows = numpy.array(rows)
for f, l in rows:
    print("   demonstration: %+5.1f %+5.1f  %.1f" % (f, l, numpy.hypot(f, l)))
print("   demonstrations, median: %+5.1f %+5.1f  %.1f" % (numpy.median(rows[:, 0]), numpy.median(rows[:, 1]), numpy.median(numpy.hypot(rows[:, 0], rows[:, 1]))))
for number in (5, 9, 10):
    s = run_log(number)
    k = int(numpy.flatnonzero(s["j4"].to_numpy() <= -70.0)[0])
    f, l = in_first_frame(100 * (s["tip_x"].iloc[k] - s["tip_x"].iloc[0]), 100 * (s["tip_y"].iloc[k] - s["tip_y"].iloc[0]), s["picture_heading_deg"].iloc[0])
    print("   run %2d, done (picture %d):            %+5.1f %+5.1f  %.1f" % (number, k, f, l, numpy.hypot(f, l)))
    for factor in (1.0, 1.3):
        heading = numpy.radians(s["picture_heading_deg"].iloc[:k].to_numpy())
        forward, left = factor * s["model_tip_forward_cm"].iloc[:k].to_numpy(), factor * s["model_tip_left_cm"].iloc[:k].to_numpy()
        dx = (forward * numpy.cos(heading) - left * numpy.sin(heading)).sum()
        dy = (forward * numpy.sin(heading) + left * numpy.cos(heading)).sum()
        f, l = in_first_frame(dx, dy, s["picture_heading_deg"].iloc[0])
        print("   run %2d, asked length x %.1f (calculated): %+5.1f %+5.1f  %.1f" % (number, factor, f, l, numpy.hypot(f, l)))
