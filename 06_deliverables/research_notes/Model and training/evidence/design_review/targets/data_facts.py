"""
Data facts for the targets section (read only): what the cleaned rows hold, how long a step takes for a given tip
move and twist, how much time goes to twists in place, and what the demonstrations do near the stop.
Writes data_facts.out.txt next to this script.
"""

import numpy
import pandas

from common_frames import OUTPUT_FOLDER, audit_cleaned, read_frames

lines = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text)


def main() -> None:
    raw = read_frames()
    say(f"raw rows {len(raw)}, episodes {raw['episode_id'].nunique()}")
    cleaned = audit_cleaned(raw)
    say(f"cleaned rows before score and mistake rules {int(cleaned['rows_before_scores'].iloc[0])}; after: {len(cleaned)} in "
        f"{cleaned['episode_id'].nunique()} episodes; stop rows {int((cleaned['action'] == 'stop').sum())} "
        f"(relabelled {int(cleaned['is_relabelled_stop'].sum())})")
    tip_mm = 1000.0 * numpy.hypot(cleaned["tip_forward_m"].fillna(0.0), cleaned["tip_left_m"].fillna(0.0))
    twist = cleaned["forearm_change_deg"].fillna(0.0)
    is_stop = cleaned["action"] == "stop"
    say("--- labels after cleaning")
    say(cleaned["action"].value_counts().to_string())
    is_small = cleaned["action"] == "small_motion"
    is_turn = cleaned["action"].isin(["turn_left", "turn_right"])
    say(f"small_motion rows {int(is_small.sum())}: |twist| >= 3 deg {int((is_small & (twist.abs() >= 3)).sum())}, "
        f">= 1 deg {int((is_small & (twist.abs() >= 1)).sum())}, both tip < 1 mm and |twist| < 1 deg {int((is_small & (tip_mm < 1) & (twist.abs() < 1)).sum())}; "
        f"twist median |deg| {twist[is_small].abs().median():.2f}; turns {int(is_turn.sum())}, their |twist| {twist[is_turn].abs().round(1).tolist()}")
    non_stop = ~is_stop
    say(f"non-stop rows {int(non_stop.sum())}; |twist| >= 2 deg {int((non_stop & (twist.abs() >= 2)).sum())} "
        f"({(non_stop & (twist.abs() >= 2)).sum() / non_stop.sum():.3f}); >= 3 deg {(non_stop & (twist.abs() >= 3)).sum() / non_stop.sum():.3f}")
    hand = cleaned["hand_change_deg"].fillna(0.0)
    lean = cleaned["lean_change_deg"].fillna(0.0)
    say(f"hand |change| >= 3 deg: {int((non_stop & (hand.abs() >= 3)).sum())} rows; lean |change| >= 1 deg: {int((non_stop & (lean.abs() >= 1)).sum())} rows")
    joint_4_deg = numpy.degrees(cleaned["joint_4"])
    say(f"|joint 4| < 5 deg in {(joint_4_deg.abs() < 5).mean():.3f} of cleaned pictures, < 15 deg in {(joint_4_deg.abs() < 15).mean():.3f}")
    say("per sheet: twist-in-place rows (small_motion) share of rows, and |twist| >= 2 deg share of non-stop rows")
    for sheet, group in cleaned.groupby("sheet"):
        group_twist = group["forearm_change_deg"].fillna(0.0)
        group_non_stop = group["action"] != "stop"
        say(f"  {sheet}: rows {len(group)}, small_motion {(group['action'] == 'small_motion').mean():.3f}, "
            f"twist >= 2 {(group_non_stop & (group_twist.abs() >= 2)).sum() / group_non_stop.sum():.3f}")

    say("--- step time against the tip move and the twist (raw rows; next row is the next waypoint of the same episode)")
    raw = raw.copy()
    grouped = raw.groupby("episode_id", sort = False)
    next_return = grouped["move_return_time"].shift(-1)
    next_waypoint = grouped["waypoint"].shift(-1)
    is_plain = (next_waypoint == raw["waypoint"] + 1) & (raw["row_in_episode"] > 0) & (raw["action"] != "stop")
    raw["move_s"] = next_return - raw["unix_time"]
    raw["tip_mm"] = 1000.0 * numpy.hypot(raw["tip_forward_m"].fillna(0.0), raw["tip_left_m"].fillna(0.0))
    raw["twist_abs"] = raw["forearm_change_deg"].fillna(0.0).abs()
    plain = raw[is_plain & raw["move_s"].between(0.05, 10.0)]
    say(f"plain steps {len(plain)}; median move {plain['move_s'].median():.3f} s; median row to row {plain['seconds_to_next_row'].median():.3f} s; "
        f"median overhead (row to row minus move) {(plain['seconds_to_next_row'] - plain['move_s']).median():.3f} s")
    straight = plain[plain["twist_abs"] < 1.0]
    say("tip move bins with |twist| < 1 deg: count, median move s, median row-to-row s, median s per cm (row to row / tip cm)")
    for low, high in ((3, 7), (7, 9), (9, 11), (11, 13), (13, 16), (16, 20), (20, 80)):
        part = straight[(straight["tip_mm"] >= low) & (straight["tip_mm"] < high)]
        if len(part) == 0:
            continue
        say(f"  {low:>2} to {high:>2} mm: {len(part):>5}, move {part['move_s'].median():.3f}, row-to-row {part['seconds_to_next_row'].median():.3f}, "
            f"per cm {(part['seconds_to_next_row'] / (part['tip_mm'] / 10.0)).median():.3f}")
    in_place = plain[plain["tip_mm"] < 3.0]
    say("twist in place (tip < 3 mm): |twist| bins: count, median move s, median row-to-row s")
    for low, high in ((1, 3), (3, 5), (5, 7), (7, 10), (10, 16)):
        part = in_place[(in_place["twist_abs"] >= low) & (in_place["twist_abs"] < high)]
        if len(part):
            say(f"  {low:>2} to {high:>2} deg: {len(part):>5}, move {part['move_s'].median():.3f}, row-to-row {part['seconds_to_next_row'].median():.3f}")
    with_twist = plain[(plain["tip_mm"] >= 9) & (plain["tip_mm"] < 12)]
    say("tip 9 to 12 mm by |twist| bin: count, median row-to-row s")
    for low, high in ((0, 1), (1, 3), (3, 5), (5, 8), (8, 16)):
        part = with_twist[(with_twist["twist_abs"] >= low) & (with_twist["twist_abs"] < high)]
        if len(part):
            say(f"  {low:>2} to {high:>2} deg: {len(part):>5}, row-to-row {part['seconds_to_next_row'].median():.3f}")

    say("--- time spent in twist-in-place steps (cleaned rows, small_motion, using seconds_to_next_row capped at 5 s)")
    seconds = cleaned["seconds_to_next_row"].clip(upper = 5.0)
    for sheet, group in cleaned.groupby("sheet"):
        group_seconds = seconds.loc[group.index]
        share = group_seconds[group["action"] == "small_motion"].sum() / group_seconds[group["action"] != "stop"].sum()
        per_episode = group.groupby("episode_id")["action"].apply(lambda actions: int((actions == "small_motion").sum()))
        say(f"  {sheet}: twist-in-place share of step time {share:.3f}; per episode median {per_episode.median():.1f} rows")
    total_share = seconds[cleaned["action"] == "small_motion"].sum() / seconds[cleaned["action"] != "stop"].sum()
    say(f"  all: {total_share:.3f}")

    say("--- near the stop (cleaned rows)")
    goal = cleaned["goal_distance_cm"].fillna(99.0)
    per_episode = []
    for episode_id, group in cleaned.groupby("episode_id", sort = False):
        group_goal = group["goal_distance_cm"].fillna(99.0).to_numpy()
        actions = group["action"].to_numpy()
        stop_positions = numpy.flatnonzero(actions == "stop")
        first_stop = stop_positions[0] if len(stop_positions) else len(actions) - 1
        before = group_goal[:first_stop]
        last_moves = 1000.0 * numpy.hypot(group["tip_forward_m"].fillna(0.0).to_numpy(), group["tip_left_m"].fillna(0.0).to_numpy())
        per_episode.append({
            "episode_id": episode_id,
            "sheet": group["sheet"].iloc[0],
            "stop_rows": len(stop_positions),
            "within_2_before_stop": int((before <= 2.0).sum()),
            "consecutive_within_2_at_end": int(numpy.argmax(before[::-1] > 2.0)) if (before > 2.0).any() else len(before),
            "row_before_stop_goal_cm": float(before[-1]) if len(before) else numpy.nan,
            "last_move_mm": float(last_moves[first_stop - 1]) if first_stop > 0 else numpy.nan,
            "rows": len(group)
        })
    episodes = pandas.DataFrame(per_episode)
    say(f"episodes {len(episodes)}; stop rows per episode {episodes['stop_rows'].value_counts().to_dict()}")
    say(f"rows within 2 cm of the goal before the first stop row: median {episodes['within_2_before_stop'].median()}, "
        f"0 in {int((episodes['within_2_before_stop'] == 0).sum())} episodes; consecutive at the end: median {episodes['consecutive_within_2_at_end'].median()}, "
        f"0 in {int((episodes['consecutive_within_2_at_end'] == 0).sum())}, 1 in {int((episodes['consecutive_within_2_at_end'] == 1).sum())}, "
        f">= 2 in {int((episodes['consecutive_within_2_at_end'] >= 2).sum())}")
    say(f"goal distance of the row before the first stop row: median {episodes['row_before_stop_goal_cm'].median():.2f} cm, "
        f"p90 {episodes['row_before_stop_goal_cm'].quantile(0.9):.2f}, over 2 cm in {int((episodes['row_before_stop_goal_cm'] > 2).sum())} episodes")
    say(f"last move before the stop: median {episodes['last_move_mm'].median():.2f} mm, p90 {episodes['last_move_mm'].quantile(0.9):.2f}")
    at_goal_not_stop = (goal <= 2.0) & ~is_stop
    say(f"non-stop rows within 2 cm of the goal: {int(at_goal_not_stop.sum())} of {len(cleaned)}; stop rows {int(is_stop.sum())}")
    say(f"non-stop rows within 3 cm: {int(((goal <= 3.0) & ~is_stop).sum())}")
    (OUTPUT_FOLDER / "data_facts.out.txt").write_text("\n".join(lines), encoding = "utf-8")
    episodes.to_csv(OUTPUT_FOLDER / "data_facts_episodes.csv", index = False)


if __name__ == "__main__":
    main()
