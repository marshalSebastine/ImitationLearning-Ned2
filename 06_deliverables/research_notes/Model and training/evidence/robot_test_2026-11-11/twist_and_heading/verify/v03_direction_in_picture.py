"""
Verifier 03: sign conventions, the direction of travel in the picture frame, and whether a rule "turn the forearm
towards the direction of travel" has the right or the wrong sign against the demonstrations.
"""
import numpy
import pandas
from vcommon import demos, run_log, RUNS, wrap

frames = demos()
moving = frames[frames["action"] != "stop"].copy()

# (a) what a positive joint-4 change does to the picture heading (consecutive kept rows of one episode)
same = (frames["episode_id"].shift(-1) == frames["episode_id"]) & (frames["row_in_episode"].shift(-1) == frames["row_in_episode"] + 1)
heading_change = wrap(frames["picture_heading_deg"].shift(-1) - frames["picture_heading_deg"])
j4_change = frames["j4"].shift(-1) - frames["j4"]
ok = same & (frames["action"] != "stop")
x, y = j4_change[ok].to_numpy(), heading_change[ok]
print("(a) consecutive rows:", int(ok.sum()))
print("    label forearm_change_deg against measured joint-4 change to the next row: correlation %.4f, median |difference| %.3f" % (numpy.corrcoef(frames.loc[ok, "twist"], x)[0, 1], numpy.median(numpy.abs(frames.loc[ok, "twist"] - x))))
print("    heading change per degree of joint-4 change: slope %.3f, correlation %.3f" % (numpy.polyfit(x, y, 1)[0], numpy.corrcoef(x, y)[0, 1]))
big = numpy.abs(x) > 4
print("    steps with |joint-4 change| > 4: heading change has the SAME sign in %.3f, opposite in %.3f" % (numpy.mean(numpy.sign(x[big]) == numpy.sign(y[big])), numpy.mean(numpy.sign(x[big]) == -numpy.sign(y[big]))))

# (b) the convention of move_angle_deg
angle_from_labels = numpy.degrees(numpy.arctan2(moving["tip_left_m"], moving["tip_forward_m"]))
difference = wrap(angle_from_labels - moving["move_angle_deg"])
longer = moving["len_cm"] > 0.05
print("(b) move_angle_deg against atan2(tip_left, tip_forward): median |difference| %.3f, 99th percentile %.3f (steps longer than 0.05 cm: %d)" % (numpy.nanmedian(numpy.abs(difference[longer])), numpy.nanquantile(numpy.abs(difference[longer]), 0.99), int(longer.sum())))
moving["angle"] = angle_from_labels

# (c) how the travel direction is distributed
steps = moving[moving["len_cm"] > 0.05]
a = steps["angle"].abs()
print("(c) demonstrated steps (longer than 0.05 cm): %d; median |angle| %.1f; within 15: %.3f; within 45: %.3f; beyond 90: %.3f" % (len(steps), a.median(), (a <= 15).mean(), (a <= 45).mean(), (a > 90).mean()))
push = steps[steps["len_cm"] >= 0.7]
print("    pushing steps (0.7 cm or more): %d; median |angle| %.1f; within 45: %.3f" % (len(push), push["angle"].abs().median(), (push["angle"].abs() <= 45).mean()))
print("    length-weighted share of travel within 45 degrees of picture-up: %.3f; mean forward share of travel (sum forward / sum length): %.3f" % (steps.loc[a <= 45, "len_cm"].sum() / steps["len_cm"].sum(), 100 * steps["tip_forward_m"].sum() / steps["len_cm"].sum()))
for sheet, rows in steps.groupby("sheet"):
    print("    %s: median |angle| %.1f, within 45: %.3f" % (sheet, rows["angle"].abs().median(), (rows["angle"].abs() <= 45).mean()))

# (d) twist against travel direction
inside = steps[steps["angle"].abs() <= 90]
print("(d) within 90 degrees: correlation twist against travel angle %.3f, slope %.4f degrees of twist per degree" % (numpy.corrcoef(inside["angle"], inside["twist"])[0, 1], numpy.polyfit(inside["angle"], inside["twist"], 1)[0]))
for sheet, rows in inside.groupby("sheet"):
    print("    %s: correlation %.3f" % (sheet, numpy.corrcoef(rows["angle"], rows["twist"])[0, 1]))
for low, high in ((45, 90), (-90, -45), (15, 45), (-45, -15)):
    rows = steps[(steps["angle"] >= low) & (steps["angle"] <= high)]
    print("    travel angle %4d to %4d (positive = picture-left): n %5d; twist > +1: %.2f; twist < -1: %.2f; mean twist %+.2f" % (low, high, len(rows), (rows["twist"] > 1).mean(), (rows["twist"] < -1).mean(), rows["twist"].mean()))

# (e) what the travel direction and the twist do one after the other: does the travel direction in the NEXT pictures move back towards picture-up after a twist?
# In a fixed table direction of travel, a heading change of +h changes the travel angle in the picture by -h. So a rule "turn towards the travel direction" is: heading change has the sign of the angle.
slope_heading = numpy.polyfit(x, y, 1)[0]
print("(e) a twist of +1 degree turns the picture heading by %+.2f degrees; so 'turn the camera towards a travel direction on the picture-left (angle > 0)' asks for a twist of sign %+d" % (slope_heading, int(numpy.sign(slope_heading))))
print("    demonstrated: on steps travelling 45 to 90 degrees to the picture-left the twist is positive (heading %s) far more often than negative" % ("increases: camera turns LEFT, towards the travel" if slope_heading > 0 else "decreases: camera turns RIGHT, away from the travel"))

# (f) the robot runs: model_angle_deg over the run
print("(f) robot runs: |model_angle_deg| median over the run, and in blocks of 10 pictures")
for number in RUNS:
    s = run_log(number)
    s = s[s["action"] == "model_step"]
    blocks = [float(numpy.median(numpy.abs(s["model_angle_deg"].iloc[i:i + 10]))) for i in range(0, len(s), 10)]
    print("    run %2d: median %.1f; blocks %s" % (number, numpy.median(numpy.abs(s["model_angle_deg"])), [round(b) for b in blocks]))
