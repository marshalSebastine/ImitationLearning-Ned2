"""
Independent re-checks for the evidence critique of the validation section (Team M, 9.11.26).

Reads only frames.csv of the v01 training index (never writes anywhere but standard output). Pure Python standard
library, so it starts quickly on a busy laptop:

    D:/AI_Project/.venv/Scripts/python.exe -I -B recheck_numbers.py > recheck_numbers_output.txt

Parts
-----
1. The two-stage choice rule (12 + 12 paired blocks, stop at a lead of 3), by convolving the per-block difference
   distribution (a different method from interim_rule_numbers.py, which enumerates both binomials).
2. Confidence interval of the six-sheet mean against the prediction interval for one new track (t, 5 degrees of
   freedom), from the per-sheet values printed in validation_numbers_output.txt part 1.
3. Seed noise of the "worst fold" measures, from the per-sheet values of four seeds in validation_numbers_output.txt
   part 2.
4. Standard-start demonstrations against the recording tool's own step budget (config.py: ceil(2.0 x line_length_cm),
   with the config lengths 24, 27, 55, 77, 57, 57 cm), and against 2 steps per cap-to-cap cm.
"""

import csv
import math


CHECK_PARAMETERS = {
    "frames_path": "D:/AI_Project/03_data/processed/v01_training_index/frames.csv",
    "t_quantile_5": 2.5706,
    "sheet_values": {
        "classes": [0.7557, 0.5299, 0.5651, 0.5595, 0.5832, 0.4671],
        "exact": [0.7469, 0.5835, 0.6238, 0.5481, 0.5633, 0.4831]
    },
    # validation_numbers_output.txt part 2: per sheet 01 to 06, seeds default, 11, 12, 13
    "seed_values": {
        "classes step": [[0.756, 0.777, 0.685, 0.731], [0.530, 0.471, 0.506, 0.500], [0.565, 0.555, 0.561, 0.570],
                         [0.559, 0.551, 0.554, 0.533], [0.583, 0.539, 0.522, 0.556], [0.467, 0.474, 0.444, 0.448]],
        "classes path": [[0.801, 0.832, 0.725, 0.793], [0.629, 0.553, 0.596, 0.582], [0.570, 0.557, 0.584, 0.585],
                         [0.652, 0.628, 0.619, 0.590], [0.621, 0.582, 0.565, 0.596], [0.526, 0.518, 0.508, 0.483]],
        "exact step": [[0.747, 0.747, 0.724, 0.752], [0.584, 0.578, 0.572, 0.576], [0.624, 0.627, 0.591, 0.613],
                       [0.548, 0.587, 0.572, 0.597], [0.563, 0.590, 0.550, 0.544], [0.483, 0.565, 0.549, 0.582]],
        "exact path": [[0.828, 0.808, 0.823, 0.840], [0.695, 0.713, 0.672, 0.696], [0.702, 0.701, 0.636, 0.689],
                       [0.633, 0.696, 0.674, 0.701], [0.648, 0.658, 0.599, 0.624], [0.534, 0.676, 0.639, 0.694]]
    },
    "config_line_length_cm": {"path_01": 24, "path_02": 27, "path_03": 55, "path_04": 77, "path_05": 57, "path_06": 57},
    "cap_to_cap_line_length_cm": {"path_01": 23.3, "path_02": 31.3, "path_03": 50.7, "path_04": 72.4, "path_05": 56.6,
                                  "path_06": 46.5},
    "step_budget_per_cm": 2.0
}



def binomial_mass(trials: int, probability: float) -> list:
    """Return the binomial probabilities of 0 to ``trials`` successes."""

    return [math.comb(trials, count) * probability ** count * (1.0 - probability) ** (trials - count)
            for count in range(trials + 1)]



def lead_distribution(blocks: int, better: float, worse: float) -> dict:
    """Return the distribution of (better successes minus worse successes) after ``blocks`` independent blocks."""

    per_block = {1: better * (1.0 - worse), -1: (1.0 - better) * worse,
                 0: better * worse + (1.0 - better) * (1.0 - worse)}
    distribution = {0: 1.0}
    for _ in range(blocks):
        updated = {}
        for lead, probability in distribution.items():
            for change, change_probability in per_block.items():
                updated[lead + change] = updated.get(lead + change, 0.0) + probability * change_probability
        distribution = updated
    return distribution



def part_one_interim_rule(parameters: dict) -> None:
    """Recompute the two-stage rule by convolution."""

    print("=== 1. Two-stage choice rule, by convolution of the per-block lead ===")
    for better, worse in ((0.9, 0.7), (0.9, 0.8), (0.7, 0.7)):
        first = lead_distribution(12, better, worse)
        second = lead_distribution(12, better, worse)
        choose_better = 0.0
        stop_first = 0.0
        for lead, probability in first.items():
            if lead >= 3:
                choose_better += probability
                stop_first += probability
            elif lead <= -3:
                stop_first += probability
            else:
                for more, more_probability in second.items():
                    total = lead + more
                    if total > 0:
                        choose_better += probability * more_probability
                    elif total == 0:
                        choose_better += 0.5 * probability * more_probability
        fixed = 0.0
        for lead, probability in lead_distribution(24, better, worse).items():
            fixed += probability if lead > 0 else (0.5 * probability if lead == 0 else 0.0)
        print(f"  {better:.0%} against {worse:.0%}: two-stage chooses the better {choose_better:.3f}, stops after 12 "
              f"{stop_first:.3f}, expected blocks {12 + 12 * (1.0 - stop_first):.1f}; fixed 24 blocks {fixed:.3f}")



def part_two_intervals(parameters: dict) -> None:
    """Confidence interval of the mean against the prediction interval for one new track."""

    print("\n=== 2. Six held-out sheets: interval for the mean against the interval for one new track ===")
    quantile = parameters["t_quantile_5"]
    for design, values in parameters["sheet_values"].items():
        mean = sum(values) / len(values)
        deviation = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        mean_half = quantile * deviation / math.sqrt(len(values))
        new_half = quantile * deviation * math.sqrt(1.0 + 1.0 / len(values))
        print(f"  {design}: mean {mean:.3f}, standard deviation {deviation:.4f}; interval for the mean "
              f"{mean - mean_half:.3f} to {mean + mean_half:.3f}; prediction interval for one new track "
              f"{mean - new_half:.3f} to {mean + new_half:.3f}")



def part_three_worst_fold(parameters: dict) -> None:
    """Seed noise of the worst-fold and the six-fold-mean measures."""

    print("\n=== 3. Worst fold and six-fold mean over four seeds (default, 11, 12, 13) ===")
    for label, per_sheet in parameters["seed_values"].items():
        worst = []
        means = []
        for seed in range(4):
            seed_values = [per_sheet[sheet][seed] for sheet in range(6)]
            lowest = min(seed_values)
            worst.append((lowest, seed_values.index(lowest) + 1))
            means.append(sum(seed_values) / 6.0)
        ranges = [100.0 * (max(values) - min(values)) for values in per_sheet]
        worst_text = ", ".join(f"{value:.3f} (sheet 0{sheet})" for value, sheet in worst)
        print(f"  {label}: worst fold per seed {worst_text}; worst-fold range "
              f"{100.0 * (max(value for value, _ in worst) - min(value for value, _ in worst)):.1f} points; six-fold mean range "
              f"{100.0 * (max(means) - min(means)):.1f} points; largest single-sheet range {max(ranges):.1f} points")



def part_four_step_budget(parameters: dict) -> None:
    """Standard-start demonstrations against the tool's own step budget."""

    print("\n=== 4. Standard-start demonstrations (score 1 to 3) against the tool's step budget ===")
    rows_per_episode = {}
    with open(parameters["frames_path"], newline = "", encoding = "utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("score") not in ("1", "2", "3", "1.0", "2.0", "3.0"):
                continue
            key = row["episode_id"]
            if key not in rows_per_episode:
                rows_per_episode[key] = {"sheet": row["sheet"], "start_kind": row["start_kind"], "rows": 0}
            rows_per_episode[key]["rows"] += 1
    standard = [episode for episode in rows_per_episode.values() if episode["start_kind"] == "STD"]
    over_tool = 0
    over_two_cap = 0
    for sheet, length in parameters["config_line_length_cm"].items():
        budget = math.ceil(parameters["step_budget_per_cm"] * length)
        chosen = [episode["rows"] for episode in standard if episode["sheet"] == sheet]
        over = sum(1 for count in chosen if count > budget)
        over_cap = sum(1 for count in chosen if count > 2.0 * parameters["cap_to_cap_line_length_cm"][sheet])
        over_tool += over
        over_two_cap += over_cap
        print(f"  {sheet}: tool budget {budget} steps; STD episodes {len(chosen)}, over the tool budget {over}, "
              f"over 2 per cap-to-cap cm {over_cap}")
    print(f"  all sheets: {len(standard)} STD episodes ({len(rows_per_episode)} scored episodes); over the tool budget "
          f"{over_tool}; over 2 per cap-to-cap cm {over_two_cap}")



def main() -> None:

    parameters = dict(CHECK_PARAMETERS)
    part_one_interim_rule(parameters)
    part_two_intervals(parameters)
    part_three_worst_fold(parameters)
    part_four_step_budget(parameters)



if __name__ == "__main__":
    main()
