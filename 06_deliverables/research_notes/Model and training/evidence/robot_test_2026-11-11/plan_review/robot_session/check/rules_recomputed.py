"""
Check of the critic's P2, P3, P12 and P13: the probabilities recomputed with my own code, and two things the critic did
not compute: (a) the rank rule's power when runs fall into two groups ("kept the line" about 6 mm, "lost it" about
35 mm), which is what the three sheet-03 runs of 11.11.26 look like (5.7, 31.0, 40.7 mm by the traced-line measure),
instead of the critic's log-normal guess; (b) the critic's rule for the model's own length ("replaces 0.65 cm only if
its median is lower") when the two are equally good. Calculated and simulated (fixed seed); nothing measured here.

Run:  D:/AI_Project/.venv/Scripts/python.exe rules_recomputed.py > rules_recomputed.out.txt
"""

import itertools
from math import comb

import numpy


def binomial(successes, runs, chance):
    return comb(runs, successes) * chance ** successes * (1.0 - chance) ** (runs - successes)


def at_least(least, runs, chance):
    return sum(binomial(count, runs, chance) for count in range(least, runs + 1))


def at_most(most, runs, chance):
    return sum(binomial(count, runs, chance) for count in range(0, most + 1))


print("1. The plan's block A rule (new at least 3 of 4, old at most 1 of 4), every setting with the same pass chance p:")
for chance in (1 / 3, 0.47, 0.5):
    one = at_least(3, 4, chance) * at_most(1, 4, chance)
    either = at_most(1, 4, chance) * (1.0 - (1.0 - at_least(3, 4, chance)) ** 2)
    print(f"   p = {chance:.2f}: one named new setting {one:.3f}; at least one of the two new settings {either:.3f}")
print(f"   a real step from 1/3 to 2/3: taken on with chance {at_most(1, 4, 1 / 3) * at_least(3, 4, 2 / 3):.3f}; from 0.1 to 0.9: {at_most(1, 4, 0.1) * at_least(3, 4, 0.9):.3f}")
print("   model E alone lost the first bend in 2 of 2 runs on 11.11 (run 10 was the team of five): Clopper-Pearson 90 % for its pass chance: 0 to", round(1 - 0.05 ** 0.5, 2))

print("\n2. The critic's rank rule, 5 against 5, rank sum of the new setting at most 19: orders that meet it")
met = sum(sum(chosen) <= 19 for chosen in itertools.combinations(range(1, 11), 5))
print(f"   {met} of {comb(10, 5)} = {met / comb(10, 5):.3f}")

generator = numpy.random.default_rng(2026_11_11)
SESSIONS = 40_000


def two_group_runs(kept_chance, size):
    """Largest distance per run, mm: a run keeps the line (about 6 mm, spread 2) or loses it (about 35 mm, spread 6)."""

    is_kept = generator.random(size) < kept_chance
    return numpy.where(is_kept, generator.normal(6.0, 2.0, size), generator.normal(35.0, 6.0, size))


print("\n3. Power when runs fall into two groups (kept about 6 mm, lost about 35 mm), 40,000 simulated sessions:")
print("   chance old keeps, chance new keeps | rank rule 5v5 (sum <= 19), with the critic's guard (old keeps at most 3 of 5) | plan's count rule 4v4 (>=3, <=1) | count rule 5v5 (new - old >= 3)")
for old_chance, new_chance in ((1 / 3, 1 / 3), (0.5, 0.5), (1 / 3, 2 / 3), (0.2, 0.8), (0.1, 0.9), (0.0, 0.5), (0.0, 0.8)):
    old = two_group_runs(old_chance, (SESSIONS, 5))
    new = two_group_runs(new_chance, (SESSIONS, 5))
    ranks = numpy.argsort(numpy.argsort(numpy.concatenate([new, old], axis = 1), axis = 1), axis = 1) + 1
    is_rank_met = ranks[:, :5].sum(axis = 1) <= 19
    old_kept, new_kept = (old <= 17.0).sum(axis = 1), (new <= 17.0).sum(axis = 1)
    with_guard = is_rank_met & (old_kept <= 3)
    count_4 = at_most(1, 4, old_chance) * at_least(3, 4, new_chance)
    count_5 = float(numpy.mean(new_kept - old_kept >= 3))
    print(f"   {old_chance:.2f}, {new_chance:.2f} | {is_rank_met.mean():.3f}, {with_guard.mean():.3f} | {count_4:.3f} | {count_5:.3f}")

print("\n4. The critic's rule for the model's own length (4 runs): 'it replaces fixed 0.65 cm (5 runs) only if its median is lower'. Both equally good:")
for kept_chance in (0.3, 0.5, 0.8):
    own = two_group_runs(kept_chance, (SESSIONS, 4))
    fixed = two_group_runs(kept_chance, (SESSIONS, 5))
    print(f"   both keep the line with chance {kept_chance:.1f}: the own length has the lower median in {100 * numpy.mean(numpy.median(own, axis = 1) < numpy.median(fixed, axis = 1)):.0f} % of sessions")
print("   a stricter rule, 'own length keeps the line in all 4 runs and fixed 0.65 cm loses it in at least 2 of 5':")
for kept_chance in (0.3, 0.5, 0.8):
    print(f"   both keep the line with chance {kept_chance:.1f}: met by luck with chance {at_least(4, 4, kept_chance) * at_most(3, 5, kept_chance):.3f}")
print(f"   own keeps with 0.9, fixed with 0.4: met with chance {at_least(4, 4, 0.9) * at_most(3, 5, 0.4):.3f}")

print("\n5. Block C, the plan's rule ('dashes' if solid 06 finished at least 2 of 3 and dashed 05 fails at least 2 of 3), all prints finished with chance p:")
for chance in (0.3, 0.5, 0.7):
    print(f"   p = {chance}: {at_least(2, 3, chance) * at_most(1, 3, chance):.3f}")
print("   the critic's 2 x 2 rule (solid at least 3 of 4, dashed at most 1 of 4): no effect p = 0.5:", round(at_least(3, 4, 0.5) * at_most(1, 4, 0.5), 3),
      "; solid 0.8, dashed 0.2:", round(at_least(3, 4, 0.8) * at_most(1, 4, 0.2), 3))
print("   third truth (learnt print 0.8, any changed print 0.2): the plan's rule says 'shape' (solid 06 fails at least 2 of 3) with chance", round(at_most(1, 3, 0.2), 3))

print("\n6. Block D, 2 runs per start, both starts equally good with chance p of keeping the line: the counts differ with chance")
for chance in (0.3, 0.5, 0.7):
    print(f"   p = {chance}: {1.0 - sum(binomial(count, 2, chance) ** 2 for count in range(3)):.3f}")

print("\n7. Block B, 3 runs: Clopper-Pearson 90 % limits: 0 of 3 -> upper", round(1 - 0.05 ** (1 / 3), 2), "; 3 of 3 -> lower", round(0.05 ** (1 / 3), 2))
print("   6 short runs of the unseen-sheet model, 3 per step (the critic's fix): chance that 0.65 cm keeps the line in 3 of 3 and 1.0 cm in 0 of 3 when both keep it with chance p:")
for chance in (1 / 3, 0.5):
    print(f"   p = {chance:.2f}: {binomial(3, 3, chance) * binomial(0, 3, chance):.3f}; '0.65 cm has at least 2 more passes than 1.0 cm': "
          f"{sum(binomial(new, 3, chance) * binomial(old, 3, chance) for new in range(4) for old in range(4) if new - old >= 2):.3f}")
print("   the same when 1.0 cm keeps it with 0.2 and 0.65 cm with 0.8: 'at least 2 more passes':",
      round(sum(binomial(new, 3, 0.8) * binomial(old, 3, 0.2) for new in range(4) for old in range(4) if new - old >= 2), 3))
