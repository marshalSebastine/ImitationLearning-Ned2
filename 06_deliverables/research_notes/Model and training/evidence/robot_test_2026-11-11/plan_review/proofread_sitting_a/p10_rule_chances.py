"""
Proofreader sitting_a, check 10: the chances behind the count rules of part A4, recalculated, and the chances of the
rules that are missing (the team of five has no rule; the own length is compared with 0.65 cm only).

"By luck" = both settings keep the line with the same true chance p; the largest value over p is printed.
Pure arithmetic. Run:  D:/AI_Project/.venv/Scripts/python.exe -B p10_rule_chances.py > p10_rule_chances.out.txt
"""

from math import comb

import numpy


def binomial(count, total, chance):
    return comb(total, count) * chance ** count * (1 - chance) ** (total - count)


def chance_of(rule, total_new, total_old, chance_new, chance_old):
    return sum(binomial(new, total_new, chance_new) * binomial(old, total_old, chance_old)
               for new in range(total_new + 1) for old in range(total_old + 1) if rule(new, old))


grid = numpy.linspace(0.0, 1.0, 1001)
rules = {
    "plan: 0.65 cm goes on when it keeps the line in at least 3 more runs than 1.0 cm (5 against 5)": (lambda new, old: new - old >= 3, 5, 5),
    "plan: the own length replaces 0.65 cm when it keeps all 4 while 0.65 cm loses 2 or more of 5": (lambda new, old: new == 4 and old <= 3, 4, 5),
    "plan: with the model that never saw the sheet, 0.65 cm keeps the line in fewer runs than 1.0 cm (3 against 3)": (lambda new, old: new < old, 3, 3),
    "missing rule, proposal: the team of five counts as better at the bend when it keeps 3 or 4 of 4 while model E keeps at most 1 of 5": (lambda new, old: new >= 3 and old <= 1, 4, 5),
    "missing rule, proposal: the own length goes on against 1.0 cm when it keeps all 4 while 1.0 cm keeps at most 1 of 5": (lambda new, old: new == 4 and old <= 1, 4, 5),
    "plan, A5: through the first S-bend in at least 2 of 3 runs (one setting; chance shown for a true chance of 0.3, 0.5, 0.7)": None,
    "plan, A3: fewer than 2 of 3 finished (chance shown for a true chance of 0.5, 0.7, 0.9)": None,
}
for name, rule in rules.items():
    if rule is None:
        continue
    test, total_new, total_old = rule
    by_luck = [chance_of(test, total_new, total_old, chance, chance) for chance in grid]
    worst = int(numpy.argmax(by_luck))
    real = {f"{old:.1f} to {new:.1f}": chance_of(test, total_new, total_old, new, old) for old, new in ((0.2, 0.8), (0.33, 0.67), (0.1, 0.9))}
    print(f"{name}\n   by luck at most {by_luck[worst]:.3f} (at p = {grid[worst]:.2f}), which is 1 sitting in {1 / by_luck[worst]:.0f}; "
          + "; ".join(f"a real step from {key}: {value:.2f}" for key, value in real.items()))
print("plan, A5: through the first S-bend in at least 2 of 3 runs: " + ", ".join(f"true chance {p:.1f} -> {sum(binomial(k, 3, p) for k in (2, 3)):.2f}" for p in (0.3, 0.5, 0.7)))
print("plan, A3: fewer than 2 of 3 finished: " + ", ".join(f"true chance {p:.1f} -> {sum(binomial(k, 3, p) for k in (0, 1)):.2f}" for p in (0.5, 0.7, 0.9)))
