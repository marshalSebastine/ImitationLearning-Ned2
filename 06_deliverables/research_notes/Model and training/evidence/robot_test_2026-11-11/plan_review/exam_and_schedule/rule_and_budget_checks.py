"""
Plan review (key: exam_and_schedule): how the plan's fixed rules behave, and the robot-hour sum.

Everything here is calculated on paper from numbers measured elsewhere (sources in the print-out).
Run:  D:/AI_Project/.venv/Scripts/python.exe rule_and_budget_checks.py > rule_and_budget_checks.out.txt
"""

import math


def binomial(count: int, successes: int, chance: float) -> float:
    return math.comb(count, successes) * chance ** successes * (1 - chance) ** (count - successes)


def at_least(count: int, least: int, chance: float) -> float:
    return sum(binomial(count, successes, chance) for successes in range(least, count + 1))


def at_most(count: int, most: int, chance: float) -> float:
    return sum(binomial(count, successes, chance) for successes in range(0, most + 1))


def block_a_rule() -> None:
    """Plan, step 2, block A: a new step rule is taken on only with >= 3 of 4 passes where fixed 1.0 cm has <= 1 of 4."""

    print("=== block A: at least 3 of 4 passes where fixed 1.0 cm has at most 1 of 4 ===")
    print("  true pass chance (new rule, fixed 1.0 cm) -> chance that the rule takes the new setting on")
    for new, old in [(0.9, 0.33), (0.75, 0.33), (0.75, 0.5), (0.6, 0.33), (0.5, 0.5), (0.33, 0.33), (0.9, 0.1), (1.0, 0.0)]:
        chance = at_least(4, 3, new) * at_most(4, 1, old)
        print(f"    new {new:.2f}, old {old:.2f}: {chance:.2f}   (new passes >= 3 of 4: {at_least(4, 3, new):.2f}; old <= 1 of 4: {at_most(4, 1, old):.2f})")
    print("  the old driver passed this bend 1 time in 3 on 11.11.26 (run 10 passed; runs 5 and 9 did not), so 0.33 is the measured guess for the old setting")
    either = (1 - (1 - at_least(4, 3, 0.33)) ** 2) * at_most(4, 1, 0.33)
    print(f"  two new settings that are NO better (0.33 each, same control runs): at least one is taken on by luck with chance about {either:.3f}")


def block_c_rule() -> None:
    """Plan, block C: dashes are a main cause if the solid print is finished >= 2 of 3 and the dashed print fails >= 2 of 3."""

    print("")
    print("=== block C: solid print finished >= 2 of 3 AND dashed print fails >= 2 of 3 ===")
    for solid, dashed in [(0.5, 0.5), (0.6, 0.6), (0.7, 0.3), (0.9, 0.1)]:
        chance = at_least(3, 2, solid) * at_most(3, 1, dashed)
        print(f"    true finish chance solid {solid:.1f}, dashed {dashed:.1f}: the rule says dashes with chance {chance:.2f}")


def wilson(successes: int, count: int, z: float = 1.96) -> tuple:
    share = successes / count
    centre = (share + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(share * (1 - share) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return centre - half, centre + half


def what_three_runs_say() -> None:
    print("")
    print("=== what k of n runs say about the true finish chance (95 % Wilson interval) ===")
    for successes, count in [(0, 3), (1, 3), (2, 3), (3, 3), (0, 4), (2, 4), (4, 4), (12, 18), (15, 18)]:
        low, high = wilson(successes, count)
        print(f"    {successes} of {count}: {low:.2f} to {high:.2f}")


def attempts() -> None:
    print("")
    print("=== why the number of attempts per track matters: chance to finish a track at least once ===")
    for chance in [0.3, 0.5, 0.7]:
        print(f"    one-run finish chance {chance:.1f}: 1 attempt {chance:.2f}, 2 attempts {1 - (1 - chance) ** 2:.2f}, 3 attempts {1 - (1 - chance) ** 3:.2f}")


def stop_gain() -> None:
    """Earlier analysis 8.2: +8.7 points of stopped right per doubling of pictures (frozen-probe fit; each episode holds one stop picture)."""

    print("")
    print("=== expected gain of stopped right on unseen sheets from the recording round (frozen-probe fit of report 8.2, an extrapolation) ===")
    stop_rows_now = 259   # verified results, stop and centring, claim 3: 259 stop pictures (209 plus 50 relabelled)
    for new_episodes in [48 + 10, 48 + 30, 48 + 30 + 30]:
        factor = (stop_rows_now + new_episodes) / stop_rows_now
        points = 8.7 * math.log2(factor)
        print(f"    {new_episodes} new episodes with an end: stop pictures x{factor:.2f} -> about {points:+.1f} points; 106 of 209 becomes about {106 + 209 * points / 100:.0f} of 209")
    print("    per sheet, model E on its unseen sheet today (never / earlier / right; verifier, check_early_and_never_stops.out.txt):")
    table = {"01": (1, 3, 16), "02": (29, 4, 12), "03": (8, 3, 34), "04": (6, 9, 30), "05": (12, 11, 6), "06": (15, 2, 8)}
    for sheet, (never, early, right) in table.items():
        total = never + early + right
        print(f"      sheet {sheet}: right {right} of {total} = {100 * right / total:.0f} %, never {100 * never / total:.0f} %, too early {100 * early / total:.0f} %")


def robot_hours() -> None:
    """Sum of the plan's robot items with unit times measured in timing_checks.out.txt and report 7.4 / 8.4."""

    run_min = 3.93            # measured: start-to-start of the ten runs of 11.11.26
    episode_min = 8.13        # measured: all sittings, per kept demonstration
    layout_min = 10.0         # GUESS of the orientation analyst, used by its verifier too
    layouts_inside_mean = 48 / 5.7   # the measured mean already holds about one new lay-out per 5.7 episodes (37 lay-outs in 210)

    print("")
    print("=== robot hours, plan item by item (low to high) ===")
    items = []
    session_2_low = 36 * run_min / 60 + 0.3 + 24 * 0.75 / 60 + 8 * 4 / 60
    session_2_high = 36 * run_min / 60 + 0.5 + 24 * 1.5 / 60 + 8 * 7 / 60 + 0.3
    items.append(("session 2: 36 runs at the measured 3.93 min, plus block 0, a photo at 24 stops, 8 sheet changes (plan: about 3)", session_2_low, session_2_high))
    items.append(("hand trial of the demonstration rules (plan: 0.5)", 0.5, 0.5))
    new_low = (48 * episode_min + max(0.0, 12 - layouts_inside_mean) * layout_min) / 60
    new_mid = (48 * episode_min + (24 - layouts_inside_mean) * layout_min) / 60
    new_high = (48 * episode_min + (48 - layouts_inside_mean) * layout_min) / 60
    print(f"  new shapes, 48 episodes: one lay-out per track {new_low:.1f} h; two lay-outs per track {new_mid:.1f} h; a new lay-out or print at every episode {new_high:.1f} h (plan: about 6)")
    items.append(("recording: new shapes, 48 episodes (plan: about 6)", new_mid, new_high))
    items.append(("recording: 30 recoveries (plan: about 3; report 8.4: 2.8 to 3.6)", 2.8, 3.6))
    items.append(("recording: 10 to 30 line ends (plan: 1 to 3)", 1.0, 3.0))
    session_3_low = (6 + 6) * 3 * 4.0 / 60
    session_3_high = (12 + 6) * 3 * 4.5 / 60
    items.append(("session 3: development tracks and six sheets, 3 runs each (plan: no number)", session_3_low, session_3_high))
    items.append(("correction pilot, 5 episodes (report 8.4; not in the plan)", 0.4, 0.6))
    items.append(("correction round 1, 50 episodes (plan: 4 to 6; report: 3.7 to 6.3)", 3.7, 6.3))
    items.append(("paired test new against old (plan: no number; report 7.4: 12 to 24 blocks)", 1.7, 4.8))
    items.append(("final test, 60 runs (plan: no number; report 7.4: 4.0 to 5.7)", 4.0, 5.7))
    items.append(("light block and timed rehearsal (report: 0.4 to 0.5; rehearsal a guess of 1)", 1.4, 1.5))
    low = sum(item[1] for item in items)
    high = sum(item[2] for item in items)
    for name, item_low, item_high in items:
        print(f"  {name}: {item_low:.1f} to {item_high:.1f} h")
    print(f"  SUM without correction round 2 and without the open recording item: {low:.0f} to {high:.0f} h")
    print(f"  with correction round 2 (report 8.4: 2.2 to 6.3 h): {low + 2.2:.0f} to {high + 6.3:.0f} h")
    weeks = 39 / 7   # 12.11 to 20.12.26 = 39 days
    print(f"  12.11 to 20.12.26 is {weeks:.1f} weeks: {low / weeks:.1f} to {high / weeks:.1f} h a week, every week; at 2.5 h a sitting that is {low / 2.5:.0f} to {high / 2.5:.0f} sittings")
    print("  what the plan itself writes down as hours: 3 + 0.5 + (6 + 3 + 1 to 3) + (4 to 6) = 17.5 to 21.5 h; the other items have no number")
    two_weeks_low = new_mid + 2.8 + 1.0 + session_3_low
    two_weeks_high = new_high + 3.6 + 3.0 + session_3_high
    print("")
    print("  the two weeks 23.11 to 6.12 in the plan's calendar (finish recording round 2, session 3, freeze the driver):")
    print(f"    if the recording of new shapes starts only after the answers of 23.11: {two_weeks_low:.0f} to {two_weeks_high:.0f} h in two weeks = {two_weeks_low / 2:.0f} to {two_weeks_high / 2:.0f} h a week")
    print(f"  the week 7.12 to 13.12 (correction pilot, round 1, paired test): {0.4 + 3.7 + 1.7:.1f} to {0.6 + 6.3 + 4.8:.1f} h, plus round 2 if round 1 helped")


def time_per_run() -> None:
    print("")
    print("=== time per run (second criterion) ===")
    print("  measured 11.11.26: 1.04 s per move at fixed 1 cm (timing_checks.out.txt); meeting-1 note of 25.10.26: a 1 cm step takes 0.77 s at a 20 % cap and 0.76 s at 40 %, so the speed cap is no lever")
    for name, steps in [("sheet 01", 31.5), ("sheet 03", 106), ("sheet 04", 168), ("sheet 05", 103.5)]:
        print(f"    {name}: demonstrated median {steps} steps -> about {steps * 1.04:.0f} s at 1 cm; with the asked length (verifier: about 1.3 times longer) to fixed 0.65 cm (1.54 times): {steps * 1.04 * 1.3:.0f} to {steps * 1.04 * 1.54:.0f} s")
    print("    step budget 400 x 1.04 s = about 7 minutes for a run that never stops")


if __name__ == "__main__":
    block_a_rule()
    block_c_rule()
    what_three_runs_say()
    attempts()
    stop_gain()
    robot_hours()
    time_per_run()
