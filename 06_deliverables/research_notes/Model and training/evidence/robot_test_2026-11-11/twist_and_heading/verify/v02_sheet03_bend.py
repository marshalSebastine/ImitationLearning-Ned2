"""
Verifier 02: the first bend of sheet 03. How many pictures, how much tip PATH and how much NET tip displacement until
joint 4 reaches -50 and -70 degrees: standard-start demonstrations from end A against robot runs 5, 9, 10.
Also twist per decision and per cm over the first 13 decisions, and the first ten decisions of all runs.
"""
import numpy
import pandas
from vcommon import demos, run_log, RUNS

frames = demos()
sheet3 = frames[(frames["sheet"] == "path_03")]
print("sheet 03 start kinds x end:")
print(sheet3.groupby(["start_kind", "start_end"])["episode_id"].nunique().unstack(fill_value = 0))
reference = sheet3[(sheet3["start_kind"] == "STD") & (sheet3["start_end"] == "A")]
print("\nreference episodes (STD, end A):", reference["episode_id"].nunique())
for episode, e in reference.groupby("episode_id"):
    print("  ", episode.split("/")[-1], "rows", len(e), "first row_in_episode", int(e["row_in_episode"].iloc[0]), "joint 4 first %.1f" % e["j4"].iloc[0],
          "first tip (%.1f, %.1f) cm" % (100 * e["tip_x_m"].iloc[0], 100 * e["tip_y_m"].iloc[0]), "plan", e["plan_number"].iloc[0])

for angle in (-50.0, -70.0):
    rows = []
    for episode, e in reference.groupby("episode_id"):
        e = e.reset_index(drop = True)
        hit = numpy.flatnonzero(e["j4"].to_numpy() <= angle)
        if len(hit) == 0:
            continue
        k = hit[0]
        path_label = e["len_cm"].iloc[:k].sum()
        path_tip = 100.0 * numpy.hypot(numpy.diff(e["tip_x_m"].iloc[:k + 1]), numpy.diff(e["tip_y_m"].iloc[:k + 1])).sum()
        net = 100.0 * numpy.hypot(e["tip_x_m"].iloc[k] - e["tip_x_m"].iloc[0], e["tip_y_m"].iloc[k] - e["tip_y_m"].iloc[0])
        rows.append({"pictures": k, "path_label_cm": path_label, "path_tip_cm": path_tip, "net_cm": net, "turn": e["j4"].iloc[k] - e["j4"].iloc[0],
                     "turn_per_picture": (e["j4"].iloc[k] - e["j4"].iloc[0]) / k})
    table = pandas.DataFrame(rows)
    print(f"\njoint 4 reaches {angle:.0f}: demonstrations ({len(table)} episodes)")
    print(table.round(2).to_string())
    print("  medians:", table.median().round(2).to_dict())
    for number in (5, 9, 10):
        s = run_log(number)
        hit = numpy.flatnonzero(s["j4"].to_numpy() <= angle)
        if len(hit) == 0:
            print(f"  run {number}: never"); continue
        k = hit[0]
        net = 100.0 * numpy.hypot(s["tip_x"].iloc[k] - s["tip_x"].iloc[0], s["tip_y"].iloc[k] - s["tip_y"].iloc[0])
        print(f"  run {number}: picture {k}, path {s['moved_cm'].iloc[:k].sum():.1f} cm, net displacement {net:.1f} cm, asked lengths sum {s['asked_cm'].iloc[:k].sum():.1f} cm,"
              f" start joint 4 {s['j4'].iloc[0]:.1f}, turn {s['j4'].iloc[k] - s['j4'].iloc[0]:.1f}, turn per decision {(s['j4'].iloc[k] - s['j4'].iloc[0]) / k:.2f}")

print("\nfirst 10 cm of demonstrated path (labelled lengths): pictures, net joint-4 change, net displacement")
rows = []
for episode, e in reference.groupby("episode_id"):
    e = e.reset_index(drop = True)
    arc = numpy.r_[0.0, numpy.cumsum(e["len_cm"].to_numpy())]
    k = int(numpy.searchsorted(arc, 10.0))  # number of steps needed to pass 10 cm
    net = 100.0 * numpy.hypot(e["tip_x_m"].iloc[k] - e["tip_x_m"].iloc[0], e["tip_y_m"].iloc[k] - e["tip_y_m"].iloc[0])
    rows.append({"pictures": k, "turn": e["j4"].iloc[k] - e["j4"].iloc[0], "path": arc[k], "net_cm": net})
table = pandas.DataFrame(rows)
table["deg_per_picture"] = table["turn"] / table["pictures"]
table["deg_per_cm"] = table["turn"] / table["path"]
table["deg_per_net_cm"] = table["turn"] / table["net_cm"]
table["pictures_per_cm"] = table["pictures"] / table["path"]
print(table.round(2).to_string())
print("  medians:", table.median().round(2).to_dict())

print("\nrobot, first 13 decisions")
for number in (5, 9, 10):
    s = run_log(number)
    first = s.iloc[:13]
    turn = s["j4"].iloc[13] - s["j4"].iloc[0]
    path = first["moved_cm"].sum()
    net = 100.0 * numpy.hypot(s["tip_x"].iloc[13] - s["tip_x"].iloc[0], s["tip_y"].iloc[13] - s["tip_y"].iloc[0])
    print(f"  run {number}: turn {turn:.1f} over {path:.1f} cm path ({net:.1f} cm net): {turn / 13:.2f} per decision, {turn / path:.2f} per cm, {turn / net:.2f} per net cm;"
          f" asked: mean length {first['asked_cm'].mean():.2f} cm, sum {first['asked_cm'].sum():.1f} cm, mean twist {first['model_twist_deg'].mean():.2f}, twist per asked cm {first['model_twist_deg'].sum() / first['asked_cm'].sum():.1f}")
    forward = first["model_tip_forward_cm"].to_numpy(); left = first["model_tip_left_cm"].to_numpy()
    print("        asked net vector length in the (turning) picture frame is not meaningful; asked angles:", numpy.round(first["model_angle_deg"].to_numpy()).astype(int).tolist())
    print("        asked lengths:", numpy.round(first["asked_cm"].to_numpy(), 2).tolist())
    print("        asked twists :", numpy.round(first["model_twist_deg"].to_numpy(), 1).tolist())

print("\nfirst 10 decisions of every run")
for number in RUNS:
    s = run_log(number).iloc[:10]
    print(f"  run {number:2d} {RUNS[number][1]} {RUNS[number][2]}: mean asked length {s['asked_cm'].mean():.2f} cm, mean |asked twist| {s['model_twist_deg'].abs().mean():.1f}, net asked twist {s['model_twist_deg'].sum():+.1f}")
