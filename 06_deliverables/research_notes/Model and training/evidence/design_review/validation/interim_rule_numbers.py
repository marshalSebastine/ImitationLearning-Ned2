"""
Exact properties of the two-stage choice rule for the robot comparison of two models (Team M, 9.11.26).

Rule: run 12 blocks (one trial of each model per block). If one model leads by at least "lead_to_stop" successes,
stop and choose it. Otherwise run 12 more blocks and choose the model with more successes over all 24 blocks; a tie
is settled by a coin (in the protocol: by the finish-then-time rule, which this calculation does not model).
Trials are treated as independent with one fixed success rate per model, as every source in the literature note does.

Run: D:/AI_Project/.venv/Scripts/python.exe -I -B interim_rule_numbers.py > interim_rule_numbers_output.txt
"""

import math


RULE_PARAMETERS = {
    "first_blocks": 12,
    "second_blocks": 12,
    "leads_to_stop": [3, 4, 5],
    "rate_pairs": [(0.9, 0.7), (0.9, 0.8), (0.7, 0.5), (0.8, 0.6), (0.7, 0.7)]
}



def probability_mass(trials: int, probability: float) -> list:
    """Return the binomial probabilities of 0 to ``trials`` successes."""

    return [math.comb(trials, count) * probability ** count * (1.0 - probability) ** (trials - count) for count in range(trials + 1)]



def two_stage_properties(better: float, worse: float, lead_to_stop: int, parameters: dict) -> dict:
    """
    Return the chance of choosing the better model, of stopping after the first stage, of choosing the worse model
    at the first stage, and the expected number of blocks.
    """

    first = parameters["first_blocks"]
    second = parameters["second_blocks"]
    better_first, worse_first = probability_mass(first, better), probability_mass(first, worse)
    better_second, worse_second = probability_mass(second, better), probability_mass(second, worse)
    choose_better = stop_first = wrong_at_first = 0.0
    for better_count, better_probability in enumerate(better_first):
        for worse_count, worse_probability in enumerate(worse_first):
            probability = better_probability * worse_probability
            lead = better_count - worse_count
            if lead >= lead_to_stop:
                choose_better += probability
                stop_first += probability
                continue
            if lead <= -lead_to_stop:
                stop_first += probability
                wrong_at_first += probability
                continue
            for more_better, more_better_probability in enumerate(better_second):
                for more_worse, more_worse_probability in enumerate(worse_second):
                    total_lead = lead + more_better - more_worse
                    weight = probability * more_better_probability * more_worse_probability
                    if total_lead > 0:
                        choose_better += weight
                    elif total_lead == 0:
                        choose_better += 0.5 * weight
    expected_blocks = first + (1.0 - stop_first) * second
    return {"choose_better": choose_better, "stop_first": stop_first, "wrong_at_first": wrong_at_first, "expected_blocks": expected_blocks}



def fixed_rule(better: float, worse: float, blocks: int) -> float:
    """Return the chance of choosing the better model after a fixed number of blocks (ties split)."""

    better_mass, worse_mass = probability_mass(blocks, better), probability_mass(blocks, worse)
    total = 0.0
    for better_count, better_probability in enumerate(better_mass):
        for worse_count, worse_probability in enumerate(worse_mass):
            if better_count > worse_count:
                total += better_probability * worse_probability
            elif better_count == worse_count:
                total += 0.5 * better_probability * worse_probability
    return total



def main() -> None:

    parameters = dict(RULE_PARAMETERS)
    for better, worse in parameters["rate_pairs"]:
        print(f"true rates {better:.0%} against {worse:.0%}: fixed 24 blocks chooses the better {fixed_rule(better, worse, 24):.3f}; "
              f"fixed 12 blocks {fixed_rule(better, worse, 12):.3f}")
        for lead in parameters["leads_to_stop"]:
            result = two_stage_properties(better, worse, lead, parameters)
            print(f"  stop after 12 at a lead of {lead}: chooses the better {result['choose_better']:.3f}; stops after 12 blocks "
                  f"{result['stop_first']:.3f}; chooses the worse at the stop {result['wrong_at_first']:.3f}; expected blocks {result['expected_blocks']:.1f}")



if __name__ == "__main__":
    main()
