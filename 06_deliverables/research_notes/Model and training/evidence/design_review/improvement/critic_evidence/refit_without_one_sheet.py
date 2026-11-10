"""
Robustness check of the improvement section's data-amount fit (read only; writes nothing outside this folder).

Rebuilds the 126 fold results exactly as data_amount_regression.py does (same kept-sheet draw, same picture counts),
then refits within_45_deg ~ fold + log2(pictures) + log2(sheets) on subsets: all runs, without the one-sheet runs,
and without the one-sheet and two-sheet runs. Bootstrap over runs as in the original.

Usage: python -I -B refit_without_one_sheet.py
"""

import json

import numpy

PARAMETERS = {
    "probe_file": "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes/data_amount.json",
    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"],
    "kept_share_after_validation": 0.85,
    "bootstrap_draws": 4000,
    "seed": 91126,
}


def kept_sheets(training_sheets, sheet_count, seed):
    """
    The sheets a sheet-count run kept, as training_probes.subsampled_training_rows draws them.
    """
    randomness_generator = numpy.random.default_rng(seed)
    sheets = sorted(training_sheets)
    if sheet_count:
        sheets = sorted(randomness_generator.permutation(sheets)[:sheet_count])
    return list(sheets)


def run_table(settings):
    """
    One row per (run, fold).
    """
    with open(settings["probe_file"], "r", encoding = "utf-8") as handle:
        results = json.load(handle)["results"]
    reference = [result for result in results if result["name"] == "episode_share_1.0_seed_1"][0]
    rows_per_sheet = {fold["fold"].replace("without_", ""): fold["test_rows"] for fold in reference["folds"]}
    table = []
    for result in results:
        run_settings = result["settings"]
        sheet_count = run_settings.get("train_sheet_count")
        share = run_settings.get("train_episode_share", 1.0)
        for fold_number, fold in enumerate(result["folds"]):
            held_out = fold["fold"].replace("without_", "")
            training_sheets = [sheet for sheet in settings["sheets"] if sheet != held_out]
            if sheet_count or share < 1.0:
                sheets = kept_sheets(training_sheets, sheet_count, run_settings["seed"] + fold_number)
            else:
                sheets = training_sheets
            pictures = settings["kept_share_after_validation"] * share * sum(rows_per_sheet[sheet] for sheet in sheets)
            table.append({"name": result["name"], "fold": held_out, "sheet_count": len(sheets), "pictures": pictures,
                          "within_45_deg": fold["within_45_deg"],
                          "tuned_two_episodes_stopped_right": fold.get("tuned_two_episodes_stopped_right")})
    return table


def fit_and_bootstrap(rows, measure, settings, randomness_generator):
    """
    Least squares with fold constants; returns the two slopes and their 95 % bootstrap ranges.
    """
    folds = settings["sheets"]
    design = numpy.array([[numpy.log2(row["pictures"]), numpy.log2(row["sheet_count"])]
                          + [1.0 if row["fold"] == fold else 0.0 for fold in folds] for row in rows])
    target = numpy.array([row[measure] for row in rows])
    coefficients, *_ = numpy.linalg.lstsq(design, target, rcond = None)
    draws = []
    for _ in range(settings["bootstrap_draws"]):
        picked = randomness_generator.integers(0, len(target), len(target))
        draw, *_ = numpy.linalg.lstsq(design[picked], target[picked], rcond = None)
        draws.append(draw[:2])
    draws = numpy.array(draws)
    return coefficients[:2], numpy.percentile(draws, [2.5, 97.5], axis = 0), len(target)


def main(parameters = None):
    """
    Print the slopes per subset.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    table = run_table(settings)
    randomness_generator = numpy.random.default_rng(settings["seed"])
    subsets = {
        "all runs": table,
        "without the one-sheet runs": [row for row in table if not row["name"].startswith("sheet_count_1_")],
        "without the one- and two-sheet runs": [row for row in table if not row["name"].startswith(("sheet_count_1_", "sheet_count_2_"))],
    }
    for measure in ("within_45_deg", "tuned_two_episodes_stopped_right"):
        print(f"=== {measure} ~ fold + log2(pictures) + log2(sheets) ===")
        for label, rows in subsets.items():
            (picture_slope, sheet_slope), ranges, count = fit_and_bootstrap(rows, measure, settings, randomness_generator)
            print(f"{label} (n {count}): per doubling of pictures {100 * picture_slope:+.2f} points "
                  f"({100 * ranges[0][0]:+.2f} to {100 * ranges[1][0]:+.2f}); per doubling of sheets {100 * sheet_slope:+.2f} "
                  f"({100 * ranges[0][1]:+.2f} to {100 * ranges[1][1]:+.2f})")
            new_tracks = picture_slope * numpy.log2(1.383) + sheet_slope * numpy.log2(10 / 6)
            more_episodes = picture_slope * numpy.log2(1.4785)
            print(f"    projected: 4 new tracks x 20 {100 * new_tracks:+.2f}; 100 more episodes on the six {100 * more_episodes:+.2f}; "
                  f"ratio {new_tracks / more_episodes:.2f}")
        print()


if __name__ == "__main__":
    main()
