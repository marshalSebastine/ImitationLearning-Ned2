"""
A count rule for block A that always gives an answer: "the new step is taken on when it keeps the line in at least D
more runs than the old step", n runs each. Exact binomial arithmetic: the chance that a setting that is no better is
taken on (worst case over the common pass chance p), and the chance that a really better one is taken on.
Calculated. Run:  D:/AI_Project/.venv/Scripts/python.exe count_rule_options.py > count_rule_options.out.txt
"""

from math import comb


def binomial(successes, runs, chance):
    return comb(runs, successes) * chance ** successes * (1.0 - chance) ** (runs - successes)


def taken_on(runs, difference, old_chance, new_chance):
    return sum(binomial(new, runs, new_chance) * binomial(old, runs, old_chance) for new in range(runs + 1) for old in range(runs + 1) if new - old >= difference)


print("runs each | at least D more | no better: at p = 1/3, at p = 1/2, worst over p | better: 1/3 -> 2/3 | 0.2 -> 0.8 | 0.1 -> 0.9 | 0 -> 0.6")
for runs, difference in ((4, 2), (4, 3), (5, 3), (5, 4), (6, 3), (6, 4)):
    worst = max(taken_on(runs, difference, value / 200, value / 200) for value in range(1, 200))
    print(f"   {runs} | {difference} | {taken_on(runs, difference, 1 / 3, 1 / 3):.3f}, {taken_on(runs, difference, 0.5, 0.5):.3f}, {worst:.3f} | "
          f"{taken_on(runs, difference, 1 / 3, 2 / 3):.3f} | {taken_on(runs, difference, 0.2, 0.8):.3f} | {taken_on(runs, difference, 0.1, 0.9):.3f} | {taken_on(runs, difference, 0.0, 0.6):.3f}")
print("\nthe plan's rule for comparison (4 runs each, new at least 3 and old at most 1), one named new setting:")
plan = lambda old_chance, new_chance: sum(binomial(new, 4, new_chance) for new in (3, 4)) * sum(binomial(old, 4, old_chance) for old in (0, 1))
worst = max(plan(value / 200, value / 200) for value in range(1, 200))
print(f"   no better: {plan(1 / 3, 1 / 3):.3f}, {plan(0.5, 0.5):.3f}, worst {worst:.3f} | {plan(1 / 3, 2 / 3):.3f} | {plan(0.2, 0.8):.3f} | {plan(0.1, 0.9):.3f} | {plan(0.0, 0.6):.3f}")
