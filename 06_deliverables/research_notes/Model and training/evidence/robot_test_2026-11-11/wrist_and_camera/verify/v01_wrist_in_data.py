"""
Verifier, check 1: what the demonstrations' wrist (joint 5) and hand (joint 6) really did. Measured from frames.csv and
episodes.csv only; no arm model. Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe v01_wrist_in_data.py
"""

import numpy
import pandas

INDEX_FOLDER = "D:/AI_Project/03_data/processed/v01_training_index"
HELD_WRIST_DEG = 13.5
HELD_HAND_DEG = -6.1

pandas.set_option("display.width", 250)
pandas.set_option("display.max_columns", 40)
pandas.set_option("display.max_rows", 300)

frames = pandas.read_csv(f"{INDEX_FOLDER}/frames.csv", low_memory = False)
episodes = pandas.read_csv(f"{INDEX_FOLDER}/episodes.csv")
frames = frames.sort_values(["episode_id", "row_in_episode"]).reset_index(drop = True)
print("rows", len(frames), "episodes", frames["episode_id"].nunique(), "episodes.csv rows", len(episodes))
for name in ("joint_4", "joint_5", "joint_6"):
    frames[name + "_deg"] = numpy.degrees(frames[name])
frames["reach_cm"] = 100.0 * numpy.hypot(frames["tip_x_m"], frames["tip_y_m"])

wrist = frames["joint_5_deg"]
print("\nwrist (joint 5), all rows: 1st %.1f, 5th %.1f, 50th %.2f, 95th %.1f, 99th %.1f, least %.1f, most %.1f" % (
    wrist.quantile(0.01), wrist.quantile(0.05), wrist.median(), wrist.quantile(0.95), wrist.quantile(0.99), wrist.min(), wrist.max()))
offset = wrist - HELD_WRIST_DEG
for limit in (2.0, 5.0, 5.7, 10.0):
    print("  |wrist - 13.5| within %.1f: %.1f %%, over: %.1f %%" % (limit, 100 * (offset.abs() <= limit).mean(), 100 * (offset.abs() > limit).mean()))

print("\nstart kinds (episodes):")
first = frames.groupby("episode_id").first()
kind = first["start_kind"].fillna("(none)")
print(kind.value_counts().to_string())
print("\nsheet_orientation examples:")
print(first["sheet_orientation"].str.replace(r"plan \d\d-\d\d, ", "", regex = True).value_counts().head(40).to_string())

first = first.assign(kind = kind)
first["is_taught"] = first["sheet_orientation"].str.contains("TAUGHT", case = False, na = False)
first["is_start_mark"] = first["sheet_orientation"].str.contains("START MARK", case = False, na = False)
print("\nstart mark episodes:", int(first["is_start_mark"].sum()), " taught-start episodes:", int(first["is_taught"].sum()),
      " neither:", int((~first["is_start_mark"] & ~first["is_taught"]).sum()))
for label, group in (("start mark", first[first["is_start_mark"]]), ("not start mark", first[~first["is_start_mark"]])):
    print(f"\nfirst rows, {label}: {len(group)} episodes")
    for column in ("tip_x_m", "tip_y_m", "joint_4_deg", "joint_5_deg", "joint_6_deg", "picture_heading_deg"):
        print("   %-22s least %9.3f  median %9.3f  most %9.3f" % (column, group[column].min(), group[column].median(), group[column].max()))
print("\nfirst rows of the not-start-mark episodes by start kind:")
print(first[~first["is_start_mark"]].groupby("kind")[["joint_5_deg", "joint_6_deg", "joint_4_deg"]].agg(["count", "min", "median", "max"]).round(2).to_string())

# the wrist's travel inside episodes
frames["wrist_step_deg"] = frames.groupby("episode_id")["joint_5_deg"].diff()
frames["hand_step_deg"] = frames.groupby("episode_id")["joint_6_deg"].diff()
steps = frames["wrist_step_deg"].dropna()
print("\nsteps inside episodes:", len(steps))
big = steps.abs() > 1.0
print("  wrist steps over 1 degree: %d (%.2f %%), they carry %.1f %% of the wrist's travel" % (int(big.sum()), 100 * big.mean(), 100 * steps.abs()[big].sum() / steps.abs().sum()))
hand_steps = frames["hand_step_deg"].dropna()
print("  hand steps over 1 degree: %d" % int((hand_steps.abs() > 1.0).sum()))

frames = frames.merge(first[["is_start_mark"]], left_on = "episode_id", right_index = True)
frames["is_far"] = (frames["joint_5_deg"] - HELD_WRIST_DEG).abs() > 5.7
print("\nrows with the wrist more than 5.7 degrees from 13.5: start mark %.1f %% of %d rows, others %.1f %% of %d rows" % (
    100 * frames.loc[frames["is_start_mark"], "is_far"].mean(), int(frames["is_start_mark"].sum()),
    100 * frames.loc[~frames["is_start_mark"], "is_far"].mean(), int((~frames["is_start_mark"]).sum())))
frames["tenth"] = numpy.minimum((10 * frames["step_fraction"]).astype(int), 9)
table = frames[frames["is_start_mark"]].pivot_table(index = "sheet", columns = "tenth", values = "is_far", aggfunc = "mean")
print("\nstart-mark rows, share with the wrist over 5.7 degrees from 13.5, by sheet and tenth of the episode:")
print((100 * table).round(0).to_string())
print("\nby sheet (all rows): wrist percentiles and share far")
print(frames.groupby("sheet").agg(rows = ("joint_5_deg", "size"), p05 = ("joint_5_deg", lambda v: v.quantile(0.05)), median = ("joint_5_deg", "median"),
                                  p95 = ("joint_5_deg", lambda v: v.quantile(0.95)), far_share = ("is_far", "mean")).round(2).to_string())

# stop rows
stops = frames[frames["action"] == "stop"]
print("\nstop rows:", len(stops), "; wrist over 5.7 from 13.5 in %d (%.1f %%); median wrist %.1f, 5th to 95th %.1f to %.1f" % (
    int(stops["is_far"].sum()), 100 * stops["is_far"].mean(), stops["joint_5_deg"].median(), stops["joint_5_deg"].quantile(0.05), stops["joint_5_deg"].quantile(0.95)))
print(stops.groupby("sheet").agg(stops = ("is_far", "size"), far = ("is_far", "mean"), wrist_median = ("joint_5_deg", "median")).round(2).to_string())
print("stops by start mark:", stops.groupby("is_start_mark")["is_far"].agg(["size", "mean"]).round(3).to_string())
last = frames.groupby("episode_id").last()
print("last rows of all %d episodes: wrist over 5.7 from 13.5 in %.1f %%" % (len(last), 100 * ((last["joint_5_deg"] - HELD_WRIST_DEG).abs() > 5.7).mean()))

# the hand
hand = frames["joint_6_deg"]
print("\nhand (joint 6): within 0.3 of -6.1 in %.1f %% of rows; within 0.3 of 0.0 in %.1f %%; over 10 from -6.1 in %.1f %% (%d rows, %d episodes)" % (
    100 * ((hand - HELD_HAND_DEG).abs() <= 0.3).mean(), 100 * (hand.abs() <= 0.3).mean(), 100 * ((hand - HELD_HAND_DEG).abs() > 10).mean(),
    int(((hand - HELD_HAND_DEG).abs() > 10).sum()), frames.loc[(hand - HELD_HAND_DEG).abs() > 10, "episode_id"].nunique()))
print("episodes with the hand over 10 degrees from -6.1 in some row:")
odd = frames[(hand - HELD_HAND_DEG).abs() > 10].groupby("episode_id").agg(rows = ("joint_6_deg", "size"), hand_median = ("joint_6_deg", "median"), sheet = ("sheet", "first"))
print(odd.to_string())

# operators: range of the wrist per episode
ranges = frames.groupby("episode_id").agg(operator = ("operator", "first"), wrist_range = ("joint_5_deg", lambda v: v.max() - v.min()))
print("\nwrist range per episode by operator (median):")
print(ranges.groupby("operator")["wrist_range"].agg(["count", "median", "mean"]).round(2).to_string())
