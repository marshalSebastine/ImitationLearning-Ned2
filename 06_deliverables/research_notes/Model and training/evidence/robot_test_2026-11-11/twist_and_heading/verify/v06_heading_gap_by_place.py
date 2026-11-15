"""
Verifier 06: the camera heading of the robot against the demonstrations AT THE SAME PLACE on the table.

For every picture of a robot run: the nearest tip position of each standard-start demonstration of the same sheet and
end (only its rows up to that demonstration's row limit, so that a later pass of the path is not matched), the distance
to it and the demonstration's picture heading there. Gap = robot heading minus the median of the demonstrations'
headings; "on the path" = the nearest demonstrated tip position is within 1.0 cm.
All measured from steps.csv and frames.csv (tip positions and headings from the arm model in both).
"""
import numpy
import pandas
from vcommon import demos, run_log, RUNS, wrap

frames = demos()
PLACEMENT = {4: ("path_02", "B"), 5: ("path_03", "A"), 6: ("path_04", "A"), 7: ("path_05", "B"), 8: ("path_06", "A"), 9: ("path_03", "A"), 10: ("path_03", "A")}

for number, (sheet, end) in PLACEMENT.items():
    reference = frames[(frames["sheet"] == sheet) & (frames["start_kind"] == "STD") & (frames["start_end"] == end)]
    episodes = [e.reset_index(drop = True) for _, e in reference.groupby("episode_id")]
    s = run_log(number)
    starts = numpy.array([[e["tip_x_m"].iloc[0], e["tip_y_m"].iloc[0]] for e in episodes])
    start_gap = 100.0 * numpy.hypot(starts[:, 0] - s["tip_x"].iloc[0], starts[:, 1] - s["tip_y"].iloc[0])
    every = numpy.vstack([e[["tip_x_m", "tip_y_m"]].to_numpy() for e in episodes])
    nearest_any = 100.0 * numpy.hypot(every[:, 0] - s["tip_x"].iloc[0], every[:, 1] - s["tip_y"].iloc[0]).min()
    print(f"run {number} ({sheet}, end {end}, {len(episodes)} standard-start demonstrations): robot start tip ({100 * s['tip_x'].iloc[0]:.1f}, {100 * s['tip_y'].iloc[0]:.1f}) cm,"
          f" joint 4 {s['j4'].iloc[0]:+.1f}, heading {s['picture_heading_deg'].iloc[0]:+.1f}; demonstrated start tips {numpy.round(100 * starts.mean(axis = 0), 1).tolist()} (distance {start_gap.min():.1f} to {start_gap.max():.1f} cm);"
          f" nearest demonstrated tip position anywhere {nearest_any:.1f} cm; demonstrated first headings {numpy.round([e['picture_heading_deg'].iloc[0] for e in episodes], 1).tolist()}")
    lines = []
    left_at = None
    progress = [0] * len(episodes)
    for k in range(len(s)):
        distances, headings = [], []
        for index, e in enumerate(episodes):
            # search only forward of the last match and not more than 25 rows ahead, so that crossings and returns do not match
            low, high = max(0, progress[index] - 3), min(len(e), progress[index] + 25)
            d = 100.0 * numpy.hypot(e["tip_x_m"].iloc[low:high] - s["tip_x"].iloc[k], e["tip_y_m"].iloc[low:high] - s["tip_y"].iloc[k]).to_numpy()
            j = int(numpy.argmin(d))
            if d[j] <= 1.0:
                progress[index] = low + j
            distances.append(d[j]); headings.append(e["picture_heading_deg"].iloc[low + j])
        distances, headings = numpy.array(distances), numpy.array(headings)
        on_path = distances <= 1.0
        if on_path.sum() >= 3:
            reference_heading = numpy.degrees(numpy.arctan2(numpy.sin(numpy.radians(headings[on_path])).mean(), numpy.cos(numpy.radians(headings[on_path])).mean()))
            gap = float(wrap(s["picture_heading_deg"].iloc[k] - reference_heading))
            spread = float(numpy.std(wrap(headings[on_path] - reference_heading)))
            lines.append((k, distances.min(), int(on_path.sum()), gap, spread))
        elif left_at is None and k > 3:
            left_at = k
    table = pandas.DataFrame(lines, columns = ["picture", "nearest_cm", "demos_near", "gap_deg", "demo_spread_deg"])
    before = table[table["picture"] < (left_at if left_at is not None else 10 ** 6)]
    print("   pictures on the path before it is first left: %d; first picture with fewer than 3 demonstrations within 1 cm: %s" % (len(before), left_at))
    if len(before):
        worst = before.loc[before["gap_deg"].abs().idxmax()]
        print("   gap over those pictures: median |gap| %.1f, largest %.1f at picture %d (nearest demonstrated tip %.2f cm, %d demonstrations near, their spread %.1f)" % (
            before["gap_deg"].abs().median(), worst["gap_deg"], worst["picture"], worst["nearest_cm"], worst["demos_near"], worst["demo_spread_deg"]))
        print("   gap by picture: " + ", ".join(f"{int(r.picture)}:{r.gap_deg:+.0f}" for r in before.itertuples()))
    print()
