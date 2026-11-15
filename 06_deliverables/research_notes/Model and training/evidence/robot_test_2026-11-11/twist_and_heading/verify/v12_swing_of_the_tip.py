"""
Verifier 12: is the tip move of a "re-aim" step simply the sideways swing that a forearm twist gives the tip?

The tip is not on the forearm's axis, so turning joint 4 alone moves it. For every demonstrated step the arm model gives
the tip move that the joint-4 change ALONE would make (joints 1, 2, 3, 5, 6 kept as they were). Compared with the tip
move that was recorded. If the short steps are that swing, then:
    - "re-aim" steps are not a manoeuvre round the puck that a model must choose, they come with the twist;
    - the correlation between twist and travel direction in the picture is partly geometry, not intent.
Also: the direction of the NET tip move over the next five steps in the current picture frame, to see how much of the
"travel is not up in the picture" is the step-to-step jitter of the guiding hand.
Arm model only (ned2_fk through side_view_test.magnet_tip_and_lean); nothing connects to the robot.
"""
import numpy
import pandas
from vcommon import demos, wrap
from side_view_test import magnet_tip_and_lean

frames = demos()
same = ((frames["episode_id"].shift(-1) == frames["episode_id"]) & (frames["row_in_episode"].shift(-1) == frames["row_in_episode"] + 1) & (frames["action"] != "stop")).to_numpy()
joint_columns = [f"joint_{k}" for k in range(1, 7)]
joints = frames[joint_columns].to_numpy(dtype = float)
index = numpy.flatnonzero(same)
rng = numpy.random.default_rng(11)
# all short steps would take long with the arm model; a random 4,000 of the consecutive steps is enough
chosen = numpy.sort(rng.choice(index, size = 4000, replace = False))
rows = []
for i in chosen:
    now, after = joints[i], joints[i + 1]
    tip_now = magnet_tip_and_lean(now.tolist())[0][:2]
    tip_after = magnet_tip_and_lean(after.tolist())[0][:2]
    only_twist = now.copy()
    only_twist[3] = after[3]
    tip_swing = magnet_tip_and_lean(only_twist.tolist())[0][:2]
    heading = numpy.radians(frames["picture_heading_deg"].iloc[i])
    up, left = numpy.array([numpy.cos(heading), numpy.sin(heading)]), numpy.array([-numpy.sin(heading), numpy.cos(heading)])
    real = 100.0 * (tip_after - tip_now)
    swing = 100.0 * (tip_swing - tip_now)
    rest = real - swing
    rows.append({"twist": numpy.degrees(after[3] - now[3]), "real_cm": numpy.hypot(*real), "swing_cm": numpy.hypot(*swing), "rest_cm": numpy.hypot(*rest),
                 "real_left": float(real @ left), "real_forward": float(real @ up), "swing_left": float(swing @ left), "swing_forward": float(swing @ up),
                 "rest_left": float(rest @ left), "rest_forward": float(rest @ up), "label_cm": frames["len_cm"].iloc[i],
                 "joint_5": numpy.degrees(now[4]), "sheet": frames["sheet"].iloc[i]})
t = pandas.DataFrame(rows)
print("sample of consecutive demonstrated steps:", len(t), "; tip move from the arm model against the labelled length: median |difference| %.3f cm" % (t["real_cm"] - t["label_cm"]).abs().median())
big = t[t["twist"].abs() >= 3]
print("swing of the tip per degree of twist (steps twisting 3 degrees or more): median %.3f cm per degree = lever arm %.1f cm; sideways (picture-left) per degree of POSITIVE twist: median %+.3f cm; forward %+.3f cm" % (
    (big["swing_cm"] / big["twist"].abs()).median(), numpy.degrees((big["swing_cm"] / big["twist"].abs()).median()), (big["swing_left"] / big["twist"]).median(), (big["swing_forward"] / big["twist"]).median()))
short = t[(t["label_cm"] < 0.7) & (t["twist"].abs() >= 3)]
longer = t[t["label_cm"] >= 0.7]
print("re-aim steps (under 0.7 cm, twist 3 degrees or more): %d; recorded tip move median %.2f cm; the swing from the twist alone median %.2f cm; what is left after taking the swing away median %.2f cm" % (
    len(short), short["real_cm"].median(), short["swing_cm"].median(), short["rest_cm"].median()))
cosine = (short["real_left"] * short["swing_left"] + short["real_forward"] * short["swing_forward"]) / (short["real_cm"] * short["swing_cm"])
print("   direction of the recorded move against the direction of the swing: median angle between them %.0f degrees; within 30 degrees in %.2f; share of re-aim steps where taking the swing away leaves under 0.25 cm: %.2f" % (
    numpy.degrees(numpy.arccos(numpy.clip(cosine, -1, 1))).median(), (numpy.degrees(numpy.arccos(numpy.clip(cosine, -1, 1))) <= 30).mean(), (short["rest_cm"] < 0.25).mean()))
print("pushing steps (0.7 cm or more): %d; recorded median %.2f cm; swing median %.2f cm; left after taking the swing away median %.2f cm" % (len(longer), longer["real_cm"].median(), longer["swing_cm"].median(), longer["rest_cm"].median()))

inside = t[(t["real_cm"] > 0.05)]
angle_real = numpy.degrees(numpy.arctan2(inside["real_left"], inside["real_forward"]))
ok = angle_real.abs() <= 90
print("\ntwist against the travel direction in the picture (within 90 degrees): correlation %+.2f with the recorded move (n %d)" % (numpy.corrcoef(angle_real[ok], inside.loc[ok, "twist"])[0, 1], int(ok.sum())))
rest_ok = inside[inside["rest_cm"] > 0.05]
angle_rest = numpy.degrees(numpy.arctan2(rest_ok["rest_left"], rest_ok["rest_forward"]))
ok = angle_rest.abs() <= 90
print("   after taking the swing of the twist away from every step: correlation %+.2f (n %d); median |direction| %.1f against %.1f with the swing in; within 45 degrees of picture-up: %.2f against %.2f" % (
    numpy.corrcoef(angle_rest[ok], rest_ok.loc[ok, "twist"])[0, 1], int(ok.sum()), angle_rest.abs().median(), angle_real.abs().median(), (angle_rest.abs() <= 45).mean(), (angle_real.abs() <= 45).mean()))
push = inside[inside["label_cm"] >= 0.7]
angle_push = numpy.degrees(numpy.arctan2(push["real_left"], push["real_forward"]))
ok = angle_push.abs() <= 90
print("   pushing steps only (0.7 cm or more), recorded move: correlation %+.2f (n %d)" % (numpy.corrcoef(angle_push[ok], push.loc[ok, "twist"])[0, 1], int(ok.sum())))
push_rest = push[push["rest_cm"] > 0.05]
angle_push_rest = numpy.degrees(numpy.arctan2(push_rest["rest_left"], push_rest["rest_forward"]))
ok = angle_push_rest.abs() <= 90
print("   pushing steps only, swing taken away: correlation %+.2f (n %d)" % (numpy.corrcoef(angle_push_rest[ok], push_rest.loc[ok, "twist"])[0, 1], int(ok.sum())))

# the net move over the next five steps in the current picture frame (all rows, from the tip positions)
episode = frames["episode_id"].to_numpy()
row = frames["row_in_episode"].to_numpy()
x, y = 100.0 * frames["tip_x_m"].to_numpy(), 100.0 * frames["tip_y_m"].to_numpy()
heading = numpy.radians(frames["picture_heading_deg"].to_numpy())
n = len(frames)
angles, path_over_net, twists = [], [], []
length = frames["len_cm"].to_numpy()
twist = frames["twist"].to_numpy()
for i in range(n - 5):
    if episode[i + 5] == episode[i] and row[i + 5] - row[i] == 5:
        dx, dy = x[i + 5] - x[i], y[i + 5] - y[i]
        net = numpy.hypot(dx, dy)
        if net < 0.5:
            continue
        angles.append(float(wrap(numpy.degrees(numpy.arctan2(dy, dx) - heading[i]))))
        path_over_net.append(length[i:i + 5].sum() / net)
        twists.append(twist[i])
angles, path_over_net, twists = numpy.array(angles), numpy.array(path_over_net), numpy.array(twists)
print("\nnet tip move over the next five steps, in the current picture frame (%d rows with a net move of 0.5 cm or more):" % len(angles))
print("   median |direction| %.1f degrees; within 15: %.2f; within 45: %.2f; beyond 90: %.2f; path over net move, median %.2f" % (numpy.median(numpy.abs(angles)), (numpy.abs(angles) <= 15).mean(), (numpy.abs(angles) <= 45).mean(), (numpy.abs(angles) > 90).mean(), numpy.median(path_over_net)))
ok = numpy.abs(angles) <= 90
print("   correlation of this step's twist with that direction (within 90 degrees): %+.2f" % numpy.corrcoef(angles[ok], twists[ok])[0, 1])
