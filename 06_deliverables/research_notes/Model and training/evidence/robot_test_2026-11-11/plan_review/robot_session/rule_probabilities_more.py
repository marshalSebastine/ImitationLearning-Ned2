"""
Robot-session critic, second sheet of arithmetic on the plan's rules (the first is rule_probabilities.py):
    block A   how often the rule's outcome is one the plan does not cover; "pass alike"; "the two best settings";
              a rank rule on a measured number instead of pass counts;
    block C   the rule under a third truth (model E does what it learnt by heart only on the print it learnt it on),
              and the same eight runs spent as two per cell of shape x line style;
    block E   what the phone photo itself can be wrong by.
Plain binomial and rank arithmetic and one small simulation with a fixed seed. Calculated, nothing measured, except the
three offsets of 11.11.26 quoted from block_a_rule_on_the_three_runs.out.txt.

Run from this folder:  D:/AI_Project/.venv/Scripts/python.exe rule_probabilities_more.py > rule_probabilities_more.out.txt
"""

import itertools
import math
from math import comb

import numpy


def binomial(successes, runs, chance):
    return comb(runs, successes) * chance ** successes * (1.0 - chance) ** (runs - successes)


def at_least(least, runs, chance):
    return sum(binomial(count, runs, chance) for count in range(least, runs + 1))


def at_most(most, runs, chance):
    return sum(binomial(count, runs, chance) for count in range(0, most + 1))


print("BLOCK A, the plan's rule (new at least 3 of 4, fixed 1.0 cm at most 1 of 4), three settings with 4 runs each")
print("  what the rule leaves open: the fixed 1.0 cm passes 2 or more of 4 (then no new rule can be taken on, whatever it scores)")
for old_chance in (0.05, 0.2, 1 / 3, 0.5, 0.7):
    print(f"      true pass chance of fixed 1.0 cm {old_chance:.2f}: it passes 2 or more of 4 with chance {at_least(2, 4, old_chance):.2f}")
print("  'if the model's own length and fixed 0.65 cm pass alike': chance that two settings show the SAME count of 4")
for first, second in ((0.7, 0.7), (0.5, 0.5), (0.8, 0.5), (0.9, 0.5)):
    same = sum(binomial(count, 4, first) * binomial(count, 4, second) for count in range(5))
    print(f"      true chances {first:.1f} and {second:.1f}: same count {same:.2f}, different count {1 - same:.2f}")
print("  'the two best settings' go on to block B: chance that second and third place tie, so that 'the two best' is not defined")
for chances in ((0.2, 0.7, 0.7), (0.3, 0.5, 0.7), (0.5, 0.5, 0.5), (0.1, 0.8, 0.8)):
    tie = 0.0
    for counts in itertools.product(range(5), repeat = 3):
        chance = math.prod(binomial(count, 4, value) for count, value in zip(counts, chances))
        ordered = sorted(counts, reverse = True)
        if ordered[1] == ordered[2]:
            tie += chance
    print(f"      true pass chances {chances}: {tie:.2f}")

print("\nBLOCK A with a measured number per run instead of pass or fail (for example the puck's largest distance from the line's middle in the bend window, mm)")
print("  rule: the new setting's runs are ranked against the old one's; taken on when its rank sum is small enough. Exact chance that a setting that is NO better meets the rule:")
for runs, limit in ((4, 10), (4, 11), (4, 12), (5, 15), (5, 17), (5, 18), (5, 19), (6, 23), (6, 26), (6, 28)):
    total, met = 0, 0
    for chosen in itertools.combinations(range(1, 2 * runs + 1), runs):
        total += 1
        met += sum(chosen) <= limit
    print(f"      {runs} against {runs} runs, rank sum of the new setting at most {limit}: {met} of {total} orders = {met / total:.3f}")
generator = numpy.random.default_rng(11_11_26)
old_values = numpy.array([41.1, 32.7, 15.6])
spread = float(numpy.std(numpy.log(old_values), ddof = 1))
print(f"  the three sheet-03 runs of 11.11 read 41.1, 32.7 and 15.6 mm (largest offset in the window): median {numpy.median(old_values):.0f} mm, spread of the logarithm {spread:.2f}")
print("  simulation (20,000 sessions, runs drawn log-normal with that spread; a guess of the shape, not a measurement): chance that the rule takes the better setting on")
for runs, limit, pass_new, pass_old in ((4, 11, 3, 1), (4, 12, 3, 1), (5, 19, 4, 1)):
    for old_median, new_median in ((30.0, 20.0), (30.0, 15.0), (30.0, 10.0)):
        old = generator.lognormal(math.log(old_median), spread, size = (20_000, runs))
        new = generator.lognormal(math.log(new_median), spread, size = (20_000, runs))
        ranks = numpy.argsort(numpy.argsort(numpy.concatenate([new, old], axis = 1), axis = 1), axis = 1) + 1
        by_rank = float(numpy.mean(ranks[:, :runs].sum(axis = 1) <= limit))
        by_count = float(numpy.mean(((new <= 17.0).sum(axis = 1) >= pass_new) & ((old <= 17.0).sum(axis = 1) <= pass_old)))
        print(f"      {runs} runs each, old median {old_median:.0f} mm, new median {new_median:.0f} mm: rank rule (sum at most {limit}) {by_rank:.2f}; "
              f"pass-count rule at 17 mm (new at least {pass_new}, old at most {pass_old}) {by_count:.2f}")

print("\nBLOCK C, the plan's rule (solid print of shape 06 finished at least 2 of 3 AND dashed print of shape 05 fails at least 2 of 3 -> 'dashes';")
print("          solid print of shape 06 fails 'like run 8' -> 'shape'), under a third truth the check itself raised: model E finishes only the print it learnt")
for known, changed in ((0.8, 0.2), (0.7, 0.3)):
    says_shape = at_most(1, 3, changed)
    says_dashes = at_least(2, 3, changed) * at_most(1, 3, changed)
    print(f"      a learnt print is finished with chance {known}, any changed print (solid 06, dashed 05) with {changed}: the rule says 'shape' with chance {says_shape:.2f}, "
          f"'dashes' with {says_dashes:.2f}; both would be wrong")
print("  the same 8 runs as 2 per cell (shape 05 or 06) x (solid or dashed), all four printed by us: dashes judged on 4 dashed against 4 solid runs,")
print("  shape on 4 runs of shape 05 against 4 of shape 06. Rule 'dashes matter' = solid finished at least 3 of 4 and dashed at most 1 of 4:")
for solid, dashed, label in ((0.5, 0.5, "dashes do nothing"), (0.8, 0.2, "dashes matter a lot"), (0.7, 0.3, "dashes matter")):
    print(f"      {label} (solid {solid}, dashed {dashed}): rule met with chance {at_least(3, 4, solid) * at_most(1, 4, dashed):.2f}")

print("\nBLOCK E, a phone photo 'from straight above' of a puck 29 mm tall, ruler on the paper [calculated, plain geometry]")
for height_mm in (250.0, 350.0, 500.0):
    for off_centre_mm in (10.0, 30.0, 60.0):
        shift = off_centre_mm * 29.0 / (height_mm - 29.0)
        print(f"      lens {height_mm:.0f} mm above the paper, puck {off_centre_mm:.0f} mm from the picture's centre: the puck's top is drawn {shift:.1f} mm further out than its foot")
for tilt_deg in (3.0, 6.0, 10.0):
    print(f"      phone tilted {tilt_deg:.0f} degrees: the puck's top is drawn {29.0 * math.tan(math.radians(tilt_deg)):.1f} mm beside its foot")
