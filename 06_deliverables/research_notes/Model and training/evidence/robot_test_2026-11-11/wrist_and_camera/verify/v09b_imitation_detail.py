"""
Verifier, check 9b: where the imitated views of v09 win and lose right stops, episode by episode (reads
v09_own_imitation_rows.csv; no inference).
    D:/AI_Project/.venv/Scripts/python.exe v09b_imitation_detail.py
"""
import numpy
import pandas

rows = pandas.read_csv("v09_own_imitation_rows.csv").sort_values(["episode_number", "row_in_episode"]).rename(columns = {"episode_number": "episode_id"})


def outcome(part, column):
    passing = part[column].to_numpy() >= 0.5
    distance = part["goal_distance_cm"].to_numpy()
    for position in range(1, len(passing)):
        if passing[position] and passing[position - 1]:
            return "right" if distance[position] <= 2.0 else "before"
    return "never"


records = []
for episode_id, part in rows.groupby("episode_id"):
    last = part.iloc[-1]
    record = {"episode_id": episode_id, "sheet": part["sheet"].iloc[0], "is_start_mark": bool(part["is_start_mark"].iloc[0]), "end_shift_px": int(last["shift_px"]),
              "moved_rows_share": float((part["shift_px"] != 0).mean())}
    for way in ("recorded", "replicate", "reflect", "null"):
        record[way] = outcome(part, f"stop_{way}")
    records.append(record)
table = pandas.DataFrame(records)
print("my rule here: 'right' = first firing within 2 cm of the goal, 'before' = any earlier firing")
for way in ("recorded", "replicate", "reflect", "null"):
    print("  %-10s %s" % (way, table[way].value_counts().to_dict()))
print("\nright stops lost and won against the recorded pictures, per way of imitating:")
for way in ("replicate", "reflect", "null"):
    lost = table[(table["recorded"] == "right") & (table[way] != "right")]
    won = table[(table["recorded"] != "right") & (table[way] == "right")]
    print("  %-10s lost %d (to never %d, to before %d), won %d; net %+d" % (way, len(lost), int((lost[way] == "never").sum()), int((lost[way] == "before").sum()), len(won), len(won) - len(lost)))
table["end_bin"] = pandas.cut(table["end_shift_px"].abs(), [-1, 4, 9, 40], labels = ["0 to 4", "5 to 9", "10 or more"])
print("\nright stops by how far the last picture is moved (px at 160):")
print(table.groupby("end_bin", observed = True).agg(episodes = ("recorded", "size"), **{way: (way, lambda v: int((v == "right").sum())) for way in ("recorded", "replicate", "reflect", "null")}).to_string())
print("\nright stops by sheet:")
print(table.groupby("sheet").agg(episodes = ("recorded", "size"), **{way: (way, lambda v: int((v == "right").sum())) for way in ("recorded", "replicate", "reflect", "null")}).to_string())

# the two rows the rule needs: the stop row and the row before it
rows["from_end"] = rows.groupby("episode_id").cumcount(ascending = False)
last_two = rows[rows["from_end"] <= 1]
print("\nlast two rows of every episode: share with a stop chance of 0.5 or more")
last_two = last_two.assign(end_bin = pandas.cut(last_two["shift_px"].abs(), [-1, 0, 4, 9, 40], labels = ["0", "1 to 4", "5 to 9", "10 or more"]))
print(last_two.groupby("end_bin", observed = True).agg(rows = ("shift_px", "size"), **{way: (f"stop_{way}", lambda v: round(float((v >= 0.5).mean()), 3)) for way in ("recorded", "replicate", "reflect", "null")}).to_string())
