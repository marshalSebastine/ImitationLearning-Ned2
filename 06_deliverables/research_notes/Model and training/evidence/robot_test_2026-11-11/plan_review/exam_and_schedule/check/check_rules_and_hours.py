"""
Check (key: exam_and_schedule_check): recompute the critic's chances for block A, block C and the pair rule, the
plan's own hour sum, and the hours of the items the plan gives no number for. Arithmetic only; reads no data.
Run:  D:/AI_Project/.venv/Scripts/python.exe check_rules_and_hours.py > check_rules_and_hours.out.txt
"""

from math import comb


def at_least(successes: int, trials: int, chance: float) -> float:
    return sum(comb(trials, count) * chance ** count * (1 - chance) ** (trials - count) for count in range(successes, trials + 1))


def main() -> None:
    print("=== block A: new setting passes 3 or more of 4 AND fixed 1.0 cm passes at most 1 of 4 ===")
    for new, old in [(0.9, 0.33), (0.75, 0.33), (0.9, 0.2), (0.9, 0.1), (0.9, 0.0)]:
        taken = at_least(3, 4, new) * (1 - at_least(2, 4, old))
        print(f"  true pass chance new {new:.2f}, old {old:.2f}: taken on with chance {taken:.2f}; old passes 2 or more of 4 with chance {at_least(2, 4, old):.2f}")
    print("  model E with fixed 1.0 cm passed the bend 0 times in 2 on 11.11 (runs 5 and 9); 1 in 3 counts run 10, which was model B")

    print("\n=== block C: solid finished 2 or more of 3 AND dashed fails 2 or more of 3, both prints with the same true chance ===")
    for chance in [0.5, 0.6]:
        print(f"  finish chance {chance}: the rule says 'dashes' with chance {at_least(2, 3, chance) * (1 - at_least(2, 3, chance)):.2f}")

    print("\n=== the critic's pair rule: new setting better in 9 or more of 12 pairs ===")
    for chance in [0.5, 0.65, 0.8, 0.9]:
        print(f"  chance to be the better run of a pair {chance}: {at_least(9, 12, chance):.3f}")
    print("  same rule with 9 pairs (three sheets x 3 pairs), 7 or more of 9:")
    for chance in [0.5, 0.8]:
        print(f"    chance {chance}: {at_least(7, 9, chance):.3f}")

    print("\n=== 3 runs against 3 runs: how often does '3 against 0' appear by luck ===")
    for chance in [0.3, 0.5, 0.7]:
        print(f"  both settings truly finish with chance {chance}: one of them shows 3 of 3 and the other 0 of 3 with chance {2 * chance ** 3 * (1 - chance) ** 3:.3f}")

    print("\n=== the plan's own hours and the items it gives no number for ===")
    plan_low, plan_high = 3 + 0.5 + 6 + 3 + 1 + 4, 3 + 0.5 + 6 + 3 + 3 + 6
    print(f"  written in the plan: {plan_low} to {plan_high} h")
    unnumbered = {"session 3 (critic: 2.4 to 4.0)": (2.4, 4.0), "paired test (report 7.4: 1.7 to 4.8)": (1.7, 4.8),
                  "final test, 60 runs (report 7.4: 4.0 to 5.7)": (4.0, 5.7), "light block (report 7.4: 0.4 to 0.5)": (0.4, 0.5),
                  "timed rehearsal (a guess: 1)": (1.0, 1.0), "correction pilot (report 8.4: 0.4 to 0.6; not in the plan)": (0.4, 0.6)}
    extra_low, extra_high = sum(value[0] for value in unnumbered.values()), sum(value[1] for value in unnumbered.values())
    print(f"  items without a number: {extra_low:.1f} to {extra_high:.1f} h")
    print(f"  the plan's own numbers plus those items: {plan_low + extra_low:.1f} to {plan_high + extra_high:.1f} h; the earlier analysis (section 8.4): 31 to 46 h; the critic: 30 to 48 h")

    print("\n=== new shapes: 48 episodes (12 tracks x 2 lay-outs x 2 prints) ===")
    for name, minutes in [("best sitting of 6 to 9.11, all in (sitting 5: 5.64 min per kept episode)", 5.64), ("median cycle (the plan's 5.8)", 5.78),
                          ("all sittings, all in (8.13 min per kept episode)", 8.13)]:
        for layouts, layout_min in [(24, 5), (24, 10)]:
            print(f"  {name}; {layouts} lay-outs at a guessed {layout_min} min each: {48 * minutes / 60 + layouts * layout_min / 60:.1f} h")

    print("\n=== the critic's own calendar, robot hours added up (low to high) ===")
    items = {"session 2a (critic)": (2.6, 2.7), "session 2b: development tracks solid and dashed, block D, ruler photos (my guess)": (1.5, 2.5),
             "hand trial": (0.5, 0.5), "recording round 2 (critic: 13 to 20)": (13.0, 20.0), "session 3 as paired test": (2.4, 4.8),
             "correction pilot": (0.4, 0.6), "correction round 1": (3.7, 6.3), "final test, 38 runs (report 7.4 short form, about 2.5 to 3.6)": (2.5, 3.6),
             "light block and rehearsal": (1.4, 1.5), "orientation ladder, 8 placements (verifier: about 1)": (1.0, 1.0),
             "readiness check in January": (1.0, 1.0)}
    low, high = sum(value[0] for value in items.values()), sum(value[1] for value in items.values())
    for name, (item_low, item_high) in items.items():
        print(f"  {name}: {item_low} to {item_high}")
    print(f"  sum: {low:.0f} to {high:.0f} h (the speed block of 1.6 to 2.4 h comes on top if it is driven)")


if __name__ == "__main__":
    main()
