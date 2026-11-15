"""
Separate "more pictures" from "more different sheets" in the data-amount experiment (read only).

The data-amount probes (frozen ResNet-18 grid features, classes head, leave one sheet out) trained each fold on a
share of the episodes of all five training sheets (0.25, 0.5, 1.0) or on all episodes of 1 to 4 of them, three
seeds each. This script rebuilds, for every run, which training sheets were kept (training_probes.subsampled_training_rows
draws them with numpy.random.default_rng(seed + fold_number).permutation(sorted sheets)[:count]) and how many
training pictures and episodes that gives (calculated: 0.85 of each kept sheet's rows and episodes, the share left
after the 15 percent of episodes held back for early stopping, times the episode share). It then fits, over all
126 fold results,

    measure = fold constant + b * log2(training pictures) + c * log2(training sheets)

and the same with log2(training episodes) for the stop measures, with a bootstrap over runs for the spread. Finally
it looks, per held-out sheet, at the mean residual when each other sheet was in the training set (weak: 12 runs per
fold).

Usage: python data_amount_regression.py
"""

import json
import statistics

import numpy

PARAMETERS = {
    "probe_file": "D:/AI_Project/.claude/worktrees/training-analysis/04_runs/2026-11-09_frozen_feature_probes/data_amount.json",
    "sheets": ["path_01", "path_02", "path_03", "path_04", "path_05", "path_06"],
    "kept_share_after_validation": 0.85,
    "bootstrap_draws": 4000,
    "seed": 91126,
    "measures": ["within_45_deg", "path_within_45_deg", "tuned_two_episodes_stopped_right", "episodes_stopped_right"],
}


def kept_sheets(training_sheets, sheet_count, seed):
    """
    Rebuild the sheets a sheet-count run kept, exactly as training_probes.subsampled_training_rows draws them.
    """
    randomness_generator = numpy.random.default_rng(seed)
    sheets = sorted(training_sheets)
    if sheet_count:
        sheets = sorted(randomness_generator.permutation(sheets)[:sheet_count])
    return list(sheets)


def run_table(parameters = None):
    """
    One row per (run, fold): fold, sheets kept, pictures, episodes and the measures.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    with open(settings["probe_file"], "r", encoding = "utf-8") as handle:
        results = json.load(handle)["results"]
    rows_per_sheet = {}
    episodes_per_sheet = {}
    reference = [result for result in results if result["name"] == "episode_share_1.0_seed_1"][0]
    for fold in reference["folds"]:
        sheet = fold["fold"].replace("without_", "")
        rows_per_sheet[sheet] = fold["test_rows"]
        episodes_per_sheet[sheet] = fold["test_episodes"]
    table = []
    for result in results:
        run_settings = result["settings"]
        sheet_count = run_settings.get("train_sheet_count")
        share = run_settings.get("train_episode_share", 1.0)
        for fold_number, fold in enumerate(result["folds"]):
            held_out = fold["fold"].replace("without_", "")
            assert held_out == settings["sheets"][fold_number]
            training_sheets = [sheet for sheet in settings["sheets"] if sheet != held_out]
            if sheet_count or share < 1.0:
                sheets = kept_sheets(training_sheets, sheet_count, run_settings["seed"] + fold_number)
            else:
                sheets = training_sheets
            pictures = settings["kept_share_after_validation"] * share * sum(rows_per_sheet[sheet] for sheet in sheets)
            episodes = settings["kept_share_after_validation"] * share * sum(episodes_per_sheet[sheet] for sheet in sheets)
            row = {
                "name": result["name"],
                "seed": run_settings["seed"],
                "fold": held_out,
                "sheet_count": len(sheets),
                "sheets": sheets,
                "share": share,
                "pictures": pictures,
                "episodes": episodes,
            }
            for measure in settings["measures"]:
                row[measure] = fold.get(measure)
            table.append(row)
    return table, rows_per_sheet, episodes_per_sheet


def fit(table, measure, size_key, folds):
    """
    Least squares with one constant per fold, log2(size) and log2(sheet count).

    Returns
    -------
    numpy.ndarray
        [b for log2(size), c for log2(sheet count), fold constants...].
    """
    usable = [row for row in table if row[measure] is not None]
    design = []
    target = []
    for row in usable:
        fold_columns = [1.0 if row["fold"] == fold else 0.0 for fold in folds]
        design.append([numpy.log2(row[size_key]), numpy.log2(row["sheet_count"])] + fold_columns)
        target.append(row[measure])
    design = numpy.array(design)
    target = numpy.array(target)
    coefficients, *_ = numpy.linalg.lstsq(design, target, rcond = None)
    return coefficients, design, target


def main(parameters = None):
    """
    Print the reconstruction, the fits and the per-sheet residuals.
    """
    settings = {**PARAMETERS, **(parameters or {})}
    table, rows_per_sheet, episodes_per_sheet = run_table(settings)
    folds = settings["sheets"]
    print("rows per sheet (target rules of the probes):", rows_per_sheet)
    print("episodes per sheet:", episodes_per_sheet)
    print()
    print("=== sheets kept per sheet-count run (fold: seed -> sheets) ===")
    for count in (1, 2, 3, 4):
        print(f"--- {count} sheet(s)")
        for fold in folds:
            draws = [(row["seed"], [sheet[-2:] for sheet in row["sheets"]], round(row["pictures"]), row["within_45_deg"])
                     for row in table if row["fold"] == fold and row["name"].startswith(f"sheet_count_{count}_")]
            print(f"  out {fold[-2:]}: " + "; ".join(f"seed {seed}: {'+'.join(sheets)} ({pictures} pictures, {value:.3f})"
                                                     for seed, sheets, pictures, value in draws))
    print()
    print("=== mean training pictures and fold-mean within_45 per setting ===")
    names = sorted({row["name"].rsplit("_seed_", 1)[0] for row in table})
    for name in names:
        rows = [row for row in table if row["name"].startswith(name + "_seed_")]
        print(f"{name}: pictures {statistics.mean(row['pictures'] for row in rows):.0f}, "
              f"episodes {statistics.mean(row['episodes'] for row in rows):.1f}, "
              f"within_45 {statistics.mean(row['within_45_deg'] for row in rows):.4f}")
    print()
    randomness_generator = numpy.random.default_rng(settings["seed"])
    for measure in settings["measures"]:
        for size_key in ("pictures", "episodes"):
            coefficients, design, target = fit(table, measure, size_key, folds)
            draws = []
            for _ in range(settings["bootstrap_draws"]):
                picked = randomness_generator.integers(0, len(target), len(target))
                draw, *_ = numpy.linalg.lstsq(design[picked], target[picked], rcond = None)
                draws.append(draw[:2])
            draws = numpy.array(draws)
            low_size, high_size = numpy.percentile(draws[:, 0], [2.5, 97.5])
            low_sheets, high_sheets = numpy.percentile(draws[:, 1], [2.5, 97.5])
            residual = target - design @ coefficients
            print(f"{measure} ~ log2({size_key}) + log2(sheets) + fold: per doubling of {size_key} "
                  f"{coefficients[0]:+.4f} (95 % bootstrap {low_size:+.4f} to {high_size:+.4f}); per doubling of sheets "
                  f"{coefficients[1]:+.4f} ({low_sheets:+.4f} to {high_sheets:+.4f}); residual sd {residual.std(ddof = 8):.4f}, "
                  f"n {len(target)}")
        print()
    print("=== per held-out sheet: mean residual of within_45 (fit on pictures) when each other sheet was kept ===")
    coefficients, design, target = fit(table, "within_45_deg", "pictures", folds)
    residuals = target - design @ coefficients
    usable = [row for row in table if row["within_45_deg"] is not None]
    for fold in folds:
        fold_rows = [(row, residual) for row, residual in zip(usable, residuals)
                     if row["fold"] == fold and row["name"].startswith("sheet_count_")]
        parts = []
        for sheet in folds:
            if sheet == fold:
                continue
            inside = [residual for row, residual in fold_rows if sheet in row["sheets"]]
            outside = [residual for row, residual in fold_rows if sheet not in row["sheets"]]
            if inside and outside:
                parts.append(f"{sheet[-2:]}: in {statistics.mean(inside):+.3f} (n {len(inside)}) vs out "
                             f"{statistics.mean(outside):+.3f} (n {len(outside)})")
        print(f"out {fold[-2:]}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
