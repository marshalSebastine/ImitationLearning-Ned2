"""
Further checks of the method critic's numbers and fixes (method_check). Reads the cleaned training rows, the saved
fold answers and the twist analyst's puck positions; nothing is trained, nothing connects to the robot.

    A. "A quarter of the way lies beyond the camera's reach" (m02, flat limit of 108 degrees): the same count with the
       limit the check calculated per reach (106 / 112 / 116 / 123 degrees at 12 / 25 / 35 / 45 cm).
    B. m06's split (camera looks along the travel or not): is it a split by the forearm's limit, by reach, or by the
       camera? Window correlation and share for "along" and "off" windows, with the off windows split into forced
       (beyond the limit) and free, and the along windows split by reach.
    C. The 71 episodes that "never stop" under the two-in-a-row rule: how many have no stop-like picture at all?
    D. Rows just after a hand correction (the nearest thing to recovery pictures): direction within 45 degrees.
    E. P3's row tags: how many old rows can be tagged at all, and how many "follow the rules".
    F. P7's pilot rule (8 episodes on 4 shapes, "net turn given back 0.59 or more"): how much the number moves when
       8 episodes are drawn from the recordings we have.

Run from the folder above:
    python check/c03_other_checks.py > check/c03_other_checks.out.txt
"""

import sys
from pathlib import Path

import numpy
import pandas

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import m01_style_against_the_rules
import method_common

CHECK_PARAMETERS = {"window_rows": 10, "net_turn_least_deg": 30.0, "reach_cm": [12.0, 25.0, 35.0, 45.0],
                    "limit_deg": [106.0, 112.0, 116.0, 123.0], "wrist_held_deg": 13.5, "hand_held_deg": -6.1,
                    "draws": 2000, "seed": 11_11_26}



def report() -> None:
    settings = method_common.merged_parameters(m01_style_against_the_rules.STYLE_PARAMETERS)
    settings.update(CHECK_PARAMETERS)
    frames = m01_style_against_the_rules.puck_travel_in_picture(method_common.cleaned_frames(settings), settings)
    data = numpy.load(settings["fold_outputs"]["E"])
    answers = pandas.DataFrame({"frame_index": data["frame_index"], "asked_forward": data["exact"][:, 0], "asked_left": data["exact"][:, 1],
                                "asked_twist": data["exact"][:, 2], "stop_chance": data["stop_probability"]})
    table = frames.merge(answers, on = "frame_index", how = "left")
    table["shown_twist"] = table["forearm_change_deg"].fillna(0.0)
    table["joint_4_deg"] = numpy.degrees(table["joint_4"])
    table["reach_cm"] = 100.0 * numpy.hypot(table["tip_x_m"], table["tip_y_m"])
    radial_deg = numpy.degrees(numpy.arctan2(table["tip_y_m"], table["tip_x_m"]))
    table["from_radial_deg"] = numpy.abs(method_common.wrapped_degrees(table["travel_table_deg"] - radial_deg))
    table["limit_here_deg"] = numpy.interp(table["reach_cm"], settings["reach_cm"], settings["limit_deg"])
    is_step = table["action"] != "stop"
    is_known = table["travel_picture_deg"].notna() & is_step

    print("A. Travel beyond the camera's reach, rows with a known direction of puck travel")
    known = table[is_known]
    print(f"   rows {len(known)}; reach of the tip from the base: median {known['reach_cm'].median():.1f} cm (10th to 90th percentile {known['reach_cm'].quantile(0.1):.1f} to {known['reach_cm'].quantile(0.9):.1f})")
    print(f"   flat limit of 108 degrees (the critic's): {100 * (known['from_radial_deg'] > 108.0).mean():.1f} %; limit by reach (106 to 123, the check's table): {100 * (known['from_radial_deg'] > known['limit_here_deg']).mean():.1f} %;"
          f" beyond 123 degrees everywhere: {100 * (known['from_radial_deg'] > 123.0).mean():.1f} %; beyond 140 (the orientation analyst's 'back sector'): {100 * (known['from_radial_deg'] > 140.0).mean():.1f} %")
    per_episode = (known["from_radial_deg"] > known["limit_here_deg"]).groupby(known["episode_id"]).sum()
    print(f"   episodes with 10 rows or more beyond the limit by reach: {int((per_episode >= 10).sum())} of {known['episode_id'].nunique()}")
    for sheet, group in known.groupby("sheet"):
        print(f"      {sheet[-2:]}: flat 108: {100 * (group['from_radial_deg'] > 108.0).mean():5.1f} %; by reach: {100 * (group['from_radial_deg'] > group['limit_here_deg']).mean():5.1f} %")

    print()
    print("B. The windows of m06 (10 rows, shown net turn of 30 degrees or more), recipe E on sheets it never saw")
    steps = table[is_step].copy()
    steps["off_deg"] = steps["travel_picture_deg"].abs()
    steps["needed_joint_4"] = (steps["joint_4_deg"] - steps["travel_picture_deg"] / settings["heading_per_twist"]).abs()
    rows = []
    for episode_id, group in steps.groupby("episode_id", sort = False):
        shown, asked = group["shown_twist"].to_numpy(), group["asked_twist"].to_numpy()
        off, needed, reach, joint_4 = group["off_deg"].to_numpy(), group["needed_joint_4"].to_numpy(), group["reach_cm"].to_numpy(), group["joint_4_deg"].abs().to_numpy()
        for start in range(0, len(group) - settings["window_rows"] + 1, settings["window_rows"]):
            part = slice(start, start + settings["window_rows"])
            if numpy.isfinite(off[part]).sum() >= settings["window_rows"] // 2:
                rows.append({"sheet": group["sheet"].iloc[0], "shown": shown[part].sum(), "asked": asked[part].sum(), "off": numpy.nanmedian(off[part]),
                             "needed": numpy.nanmedian(needed[part]), "reach": numpy.median(reach[part]), "joint_4": numpy.median(joint_4[part])})
    windows = pandas.DataFrame(rows)

    def line(name: str, part: pandas.DataFrame) -> None:
        sharp = part[part["shown"].abs() >= settings["net_turn_least_deg"]]
        share = (numpy.sign(sharp["shown"]) * sharp["asked"]).sum() / sharp["shown"].abs().sum() if len(sharp) else float("nan")
        correlation = numpy.corrcoef(part["shown"], part["asked"])[0, 1] if len(part) > 5 else float("nan")
        print(f"   {name:62s} windows {len(part):4d} (sharp {len(sharp):3d}); correlation asked with shown {correlation:+.2f}; share given back in sharp windows {share:.2f};"
              f" spread of the shown net turn {part['shown'].std():5.1f} degrees")

    along = windows[windows["off"] < 20.0]
    off = windows[windows["off"] >= 45.0]
    line("travel within 20 degrees of the picture's top (along)", along)
    line("   along, tip within 30 cm of the base", along[along["reach"] < 30.0])
    line("   along, tip 30 to 37 cm from the base", along[(along["reach"] >= 30.0) & (along["reach"] < 37.0)])
    line("   along, tip 37 cm or more from the base", along[along["reach"] >= 37.0])
    line("   along, |joint 4| under 60", along[along["joint_4"] < 60.0])
    line("   along, |joint 4| 60 or more", along[along["joint_4"] >= 60.0])
    line("travel 45 degrees or more off the picture's top (off)", off)
    line("   off, forced (looking along would need |joint 4| over 117)", off[off["needed"] > 117.0])
    line("   off, free (it would need 107 or less)", off[off["needed"] <= 107.0])
    line("   off, tip within 30 cm of the base", off[off["reach"] < 30.0])
    line("   off, tip 37 cm or more from the base", off[off["reach"] >= 37.0])
    print("   inside each held-out sheet (along / off): windows, correlation, share given back in sharp windows")
    for sheet in sorted(windows["sheet"].unique()):
        cells = []
        for part in (along[along["sheet"] == sheet], off[off["sheet"] == sheet]):
            sharp = part[part["shown"].abs() >= settings["net_turn_least_deg"]]
            share = (numpy.sign(sharp["shown"]) * sharp["asked"]).sum() / sharp["shown"].abs().sum() if len(sharp) else float("nan")
            correlation = numpy.corrcoef(part["shown"], part["asked"])[0, 1] if len(part) > 5 else float("nan")
            cells.append(f"{len(part):3d} ({len(sharp):3d} sharp), {correlation:+.2f}, {share:.2f}")
        print(f"      {sheet[-2:]}: " + "   |   ".join(cells))

    print()
    print("C. The stop over whole episodes, recipe E, sheets never seen (stop chance of 0.5 or more)")
    never_two, never_one, only_last, last_two = 0, 0, 0, 0
    late_rows = []
    for episode_id, episode in table.groupby("episode_id", sort = False):
        passing = (episode["stop_chance"].to_numpy() >= 0.5)
        in_a_row = passing[1:] & passing[:-1]
        if not in_a_row.any():
            never_two += 1
            if not passing.any():
                never_one += 1
            elif passing[-1]:
                only_last += 1
                late_rows.append(float(episode["goal_distance_cm"].to_numpy()[numpy.flatnonzero(passing)[0]]))
        if passing[-1] and len(passing) > 1 and passing[-2]:
            last_two += 1
    print(f"   never two in a row: {never_two} of {table['episode_id'].nunique()}; of these, no picture at all reaches 0.5: {never_one}; the LAST picture (the stop row) reaches 0.5: {only_last}"
          f" (the recording ends there, so there is no second picture to confirm; first stop-like picture a median {numpy.median(late_rows) if late_rows else float('nan'):.1f} cm from the stop place)")
    print(f"   the remaining {never_two - never_one - only_last} have single stop-like pictures earlier in the episode and none at the end")
    print(f"   stop rows in the cleaned data: {int((table['action'] == 'stop').sum())} in {table['episode_id'].nunique()} episodes; relabelled by the cleaning: {int(table['is_relabelled_stop'].sum())}")

    print()
    print("D. Rows just after a hand correction (the replay starts again from waypoint 1), recipe E, sheets never seen")
    grouped = table.groupby("episode_id", sort = False)
    previous_waypoint = grouped["waypoint"].shift(1)
    is_restart = (table["waypoint"] <= previous_waypoint) & previous_waypoint.notna()
    segment = is_restart.groupby(table["episode_id"]).cumsum()
    rows_since = table.groupby([table["episode_id"], segment]).cumcount()
    table["is_after_correction"] = (segment > 0) & (rows_since < 5)
    table["shown_cm"] = 100.0 * numpy.hypot(table["tip_forward_m"].fillna(0.0), table["tip_left_m"].fillna(0.0))
    shown_angle = numpy.degrees(numpy.arctan2(table["tip_left_m"].fillna(0.0), table["tip_forward_m"].fillna(0.0)))
    asked_angle = numpy.degrees(numpy.arctan2(table["asked_left"], table["asked_forward"]))
    table["angle_error"] = numpy.abs(method_common.wrapped_degrees(asked_angle - shown_angle))
    moving = table[(table["action"] != "stop") & (table["shown_cm"] >= 0.3)]
    print(f"   hand corrections found: {int(is_restart.sum())} in {int((is_restart.groupby(table['episode_id']).sum() > 0).sum())} of {table['episode_id'].nunique()} episodes"
          f" (per-episode column 'hand_corrections' sums to {int(table.groupby('episode_id')['hand_corrections'].first().sum())})")
    for name, part in (("the first 5 rows after a hand correction", moving[moving["is_after_correction"]]), ("all other moving rows", moving[~moving["is_after_correction"]])):
        print(f"   {name}: {len(part)} rows; direction within 45 degrees of the shown step {100 * (part['angle_error'] <= 45).mean():.1f} %; more than 90 degrees off {100 * (part['angle_error'] > 90).mean():.1f} %;"
              f" always forward would be within 45 in {100 * (numpy.abs(numpy.degrees(numpy.arctan2(part['tip_left_m'].fillna(0.0), part['tip_forward_m'].fillna(0.0)))) <= 45).mean():.1f} %")
    per_sheet = is_restart.groupby(table["sheet"]).sum() / table.groupby("sheet")["episode_id"].nunique()
    print("   hand corrections per episode by sheet: " + ", ".join(f"{sheet[-2:]} {value:.2f}" for sheet, value in per_sheet.items()))

    print()
    print("E. The row tags of P3 on the old rows (step rows)")
    old = table[is_step]
    is_travel_known = old["travel_picture_deg"].notna()
    needed = (old["joint_4_deg"] - old["travel_picture_deg"] / settings["heading_per_twist"]).abs()
    is_along = old["travel_picture_deg"].abs() <= 30.0
    is_forced = needed > 117.0
    is_wrist = (numpy.degrees(old["joint_5"]) - settings["wrist_held_deg"]).abs() <= 3.0
    is_hand = (numpy.degrees(old["joint_6"]) - settings["hand_held_deg"]).abs() <= 3.0
    print(f"   step rows {len(old)}; travel direction known in {100 * is_travel_known.mean():.1f} %; of the known: within 30 degrees of the picture's top {100 * is_along[is_travel_known].mean():.1f} %,"
          f" further off but forced by the limit {100 * (~is_along & is_forced)[is_travel_known].mean():.1f} %, further off with the limit not in the way {100 * (~is_along & ~is_forced)[is_travel_known].mean():.1f} %")
    print(f"   wrist within 3 degrees of 13.5: {100 * is_wrist.mean():.1f} % of step rows; wrist and hand both within 3 degrees of the driver's values: {100 * (is_wrist & is_hand).mean():.1f} %")
    follows = is_travel_known & (is_along | is_forced) & is_wrist & is_hand
    print(f"   rows that can be shown to follow rules 3 and 6 (travel known, along or forced, wrist and hand as the driver holds them): {100 * follows.mean():.1f} % of step rows"
          f" ({int(follows.sum())}); without the wrist rule: {100 * (is_travel_known & (is_along | is_forced)).mean():.1f} %")
    runs = (follows != follows.groupby(old["episode_id"]).shift(1)).groupby(old["episode_id"]).sum()
    print(f"   kept rows would lie in a median of {int(((runs + 1) // 2).median())} separate pieces per episode (90th percentile {int(((runs + 1) // 2).quantile(0.9))})")
    for sheet, group in old.assign(follows = follows).groupby("sheet"):
        print(f"      {sheet[-2:]}: {100 * group['follows'].mean():5.1f} % of step rows would be kept")

    print()
    print("F. P7's pilot rule: the share given back and the window correlation on 8 episodes (2 from each of 4 sheets), drawn from the recordings")
    randomness_generator = numpy.random.default_rng(settings["seed"])
    whole = steps[steps["start_kind"] == "STD"]
    episode_windows = {}
    for episode_id, group in whole.groupby("episode_id", sort = False):
        shown, asked = group["shown_twist"].to_numpy(), group["asked_twist"].to_numpy()
        pairs = [(shown[start:start + 10].sum(), asked[start:start + 10].sum()) for start in range(0, len(group) - 9, 10)]
        episode_windows[episode_id] = (group["sheet"].iloc[0], numpy.array(pairs).reshape(-1, 2))
    by_sheet = {}
    for episode_id, (sheet, pairs) in episode_windows.items():
        by_sheet.setdefault(sheet, []).append(pairs)
    sheets = [sheet for sheet in sorted(by_sheet) if sheet != "path_01"]
    shares, sharp_counts = [], []
    for _ in range(settings["draws"]):
        chosen = randomness_generator.choice(sheets, size = 4, replace = False)
        pairs = numpy.concatenate([by_sheet[sheet][index] for sheet in chosen for index in randomness_generator.choice(len(by_sheet[sheet]), size = 2, replace = False)])
        sharp = pairs[numpy.abs(pairs[:, 0]) >= settings["net_turn_least_deg"]]
        if len(sharp):
            shares.append((numpy.sign(sharp[:, 0]) * sharp[:, 1]).sum() / numpy.abs(sharp[:, 0]).sum())
            sharp_counts.append(len(sharp))
    shares = numpy.array(shares)
    print(f"   whole-line episodes (standard starts) of sheets 02 to 06; {settings['draws']} draws; sharp windows per draw: median {int(numpy.median(sharp_counts))}")
    print(f"   share given back: median {numpy.median(shares):.2f}, 5th to 95th percentile {numpy.percentile(shares, 5):.2f} to {numpy.percentile(shares, 95):.2f}; draws at 0.59 or more: {100 * (shares >= 0.59).mean():.0f} %")



if __name__ == "__main__":
    report()
