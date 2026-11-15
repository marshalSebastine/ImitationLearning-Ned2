"""
Robot-session critic: what the pass rules of the plan's blocks A, C and D can show with the planned numbers of runs.
Plain binomial arithmetic (each run passes with the same chance, runs independent). Calculated, nothing measured.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe rule_probabilities.py > rule_probabilities.out.txt
"""

from math import comb


def binomial(successes: int, runs: int, chance: float) -> float:
    return comb(runs, successes) * chance ** successes * (1.0 - chance) ** (runs - successes)


def at_least(least: int, runs: int, chance: float) -> float:
    return sum(binomial(count, runs, chance) for count in range(least, runs + 1))


def at_most(most: int, runs: int, chance: float) -> float:
    return sum(binomial(count, runs, chance) for count in range(0, most + 1))


print("BLOCK A. Rule: a new setting is taken on with at least 3 of 4 passes where fixed 1.0 cm has at most 1 of 4.")
print("  (a) No setting is better: every setting passes a run with the same chance p. Two new settings are tried.")
for chance in (0.2, 1 / 3, 0.4, 0.5, 0.6):
    new = at_least(3, 4, chance)
    old = at_most(1, 4, chance)
    one = new * old
    either = old * (1.0 - (1.0 - new) ** 2)
    print(f"      p = {chance:.2f}: new setting gets 3 of 4 or more {new:.3f}; old gets 1 of 4 or fewer {old:.3f}; rule met by ONE named new setting {one:.3f}; "
          f"by at least one of the two new settings {either:.3f}")
worst = max(((at_most(1, 4, value / 1000) * (1.0 - (1.0 - at_least(3, 4, value / 1000)) ** 2)), value / 1000) for value in range(1, 1000))
print(f"      worst case over p: {worst[0]:.3f} at p = {worst[1]:.2f}")
print("  (b) A real difference: old passes with p_old, new with p_new. Chance that the rule takes the new setting on:")
for old_chance, new_chance in ((1 / 3, 2 / 3), (0.25, 0.75), (0.2, 0.8), (0.1, 0.9), (1 / 3, 0.9), (0.5, 0.9)):
    print(f"      p_old {old_chance:.2f}, p_new {new_chance:.2f}: {at_most(1, 4, old_chance) * at_least(3, 4, new_chance):.3f}")
print("  (c) The same rule with more runs: at least 6 of 8 new, at most 2 of 8 old; and 8 of 10 against 3 of 10:")
for runs, new_least, old_most in ((8, 6, 2), (10, 8, 3)):
    for chance in (1 / 3, 0.5):
        new = at_least(new_least, runs, chance)
        old = at_most(old_most, runs, chance)
        print(f"      {runs} runs, no difference, p = {chance:.2f}: one named new setting {new * old:.4f}; at least one of two {old * (1.0 - (1.0 - new) ** 2):.4f}")
    for old_chance, new_chance in ((1 / 3, 2 / 3), (0.25, 0.75), (0.2, 0.8)):
        print(f"      {runs} runs, p_old {old_chance:.2f}, p_new {new_chance:.2f}: taken on with chance {at_most(old_most, runs, old_chance) * at_least(new_least, runs, new_chance):.3f}")
print("  (d) 11.11.26: the first bend of sheet 03 was passed in 1 of 3 runs (model E 0 of 2, team B 1 of 1). 'Both of two runs pass' by luck at p = 1/3:", round((1 / 3) ** 2, 3))

print("\nBLOCK C. Rule: 'dashes are a main cause' if the solid print of shape 06 is finished at least 2 of 3 AND the dashed print of shape 05 fails at least 2 of 3.")
for chance in (0.3, 0.5, 0.7):
    print(f"      dashes do nothing, every print is finished with chance {chance:.1f}: rule says 'dashes' with chance {at_least(2, 3, chance) * at_most(1, 3, chance):.3f}")
print("      truth 'shape': shape 06 finished 0.2, shape 05 finished 0.8 whatever the line style: rule says 'dashes' with chance", round(at_least(2, 3, 0.2) * at_most(1, 3, 0.8), 3))
print("      truth 'dashes': solid finished 0.8, dashed finished 0.2: rule says 'dashes' with chance", round(at_least(2, 3, 0.8) * at_most(1, 3, 0.2), 3))
print("      truth 'dashes', weaker: solid 0.7, dashed 0.3:", round(at_least(2, 3, 0.7) * at_most(1, 3, 0.3), 3))
print("      one repeat run of sheet 05 fails although its true chance is 0.7:", 0.3, "; one repeat of sheet 06 is finished although its true chance is 0.3:", 0.3)

print("\nBLOCK D. Two runs per start. 'Keeps the puck on the line for the first 10 cm':")
for chance in (0.3, 0.5, 0.7):
    print(f"      two starts that are equally good (chance {chance:.1f} per run): one start 2 of 2 and the other 0 of 2: {2 * chance ** 2 * (1 - chance) ** 2:.3f}; "
          f"the two starts show different counts at all: {1.0 - sum(binomial(count, 2, chance) ** 2 for count in range(3)):.3f}")
print("      a start that works 8 times in 10 against one that works 2 in 10: the better one has strictly more passes with chance",
      round(sum(binomial(better, 2, 0.8) * binomial(worse, 2, 0.2) for better in range(3) for worse in range(3) if better > worse), 3))

print("\nBLOCK B. Three full runs per condition, no rule. Width of what 3 runs can say (Clopper-Pearson 90 % interval, calculated by search):")
for successes in range(4):
    lower = next(value / 1000 for value in range(0, 1001) if at_least(successes, 3, value / 1000) >= 0.05) if successes else 0.0
    upper = next(value / 1000 for value in range(1000, -1, -1) if at_most(successes, 3, value / 1000) >= 0.05) if successes < 3 else 1.0
    print(f"      {successes} of 3 finished: the true chance lies between {lower:.2f} and {upper:.2f}")
