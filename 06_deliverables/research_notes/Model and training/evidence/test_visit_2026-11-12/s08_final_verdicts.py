"""
Step 8: the verdicts of the rule with the three measures together, how far the measures agree, the decisions the rules
ask for, and a figure of the offsets of model E's five pairs.

The measures of the puck's offset from the line's centre line (signed, positive = left looking from end A to end B):
    A   against the line map voted by the last 12 pictures (the measure of 11.11.26);
    B   against the printed drawing laid onto all pictures of the run;
    C   against the printed drawing laid onto the ink of the same picture (the main measure: an error of the camera
        model shared by the puck and the ink beside it cancels; see s07).
How far along the line: from method B's lay (C's agrees to a few millimetres).

A verdict is "kept" or "not kept" when C (middle reading of the travel), A, B, C's shortest and longest readings and C
with the limit 2 mm lower and higher (10 and 14 mm, about the measures' disagreement) all agree; otherwise "doubtful",
with the main reading (C, middle) given and what disagrees named.

Run from this folder after s03 and s07:  python s08_final_verdicts.py
Writes s08_verdicts.csv, s08_method_agreement.txt (also printed) and s08_pairs_offsets.png.
"""

import numpy
import pandas
from matplotlib import pyplot

import visit_common


FINAL_PARAMETERS = {
    "limits_tried_mm": [10.0, 14.0],
    "mark_shifts_mm": [-10.0, 10.0],
    "model_e_pairs": [("R05", "R06"), ("R10", "R09"), ("R11", "R12"), ("R15", "R16"), ("R18", "R17")],
    "without_03_pairs": [("R07", "R08"), ("R13", "R14"), ("R20", "R19")],
    "team_runs": ["R21", "R23", "R25", "R27"],
    "own_length_runs": ["R22", "R24", "R26", "R28"],
    "sheet_04_runs": ["R29", "R30", "R31"]
}



def score_run(table: pandas.DataFrame, sheet: str, ending: str) -> dict:
    """All readings of the rule on one run's per-picture table."""

    parameters = visit_common.merged_visit_parameters()
    travel = table["tip_travel_mm"].to_numpy()
    arcs = table["arc_mm"].to_numpy()
    mark = parameters["mark_mm"][sheet]

    def last_index_for(shift: float) -> tuple:
        passed = numpy.flatnonzero(arcs >= mark + shift)
        return (int(passed[0]), True) if len(passed) else (len(table) - 1, False)

    last_index, is_reached = last_index_for(0.0)
    found = {}
    for method in ["a", "b", "c"]:
        found[method] = visit_common.stretch_verdict(table[f"offset_{method}_mm"].to_numpy(), travel, last_index)
    variants = {"A": found["a"]["is_kept"], "B": found["b"]["is_kept"],
                "C shortest travel": found["c"]["is_kept_shortest"], "C longest travel": found["c"]["is_kept_longest"]}
    for limit in FINAL_PARAMETERS["limits_tried_mm"]:
        variants[f"C at {int(limit)} mm"] = visit_common.stretch_verdict(table["offset_c_mm"].to_numpy(), travel, last_index,
                                                                         {"offset_limit_mm": limit})["is_kept"]
    main = found["c"]["is_kept"]
    disagreeing = [name for name, value in variants.items() if value != main]
    mark_variants = {}
    for shift in FINAL_PARAMETERS["mark_shifts_mm"]:
        shifted_index, _ = last_index_for(shift)
        mark_variants[shift] = visit_common.stretch_verdict(table["offset_c_mm"].to_numpy(), travel, shifted_index)["is_kept"]
    window = table["offset_c_mm"].to_numpy()[:last_index + 1]
    whole = visit_common.stretch_verdict(table["offset_c_mm"].to_numpy(), travel, len(table) - 1)
    left_index = whole["first_failing_from"]
    status = ("kept" if main else "not kept") if not disagreeing else "doubtful"
    return {
        "mark_reached": is_reached, "mark_picture": last_index if is_reached else None,
        "window": "completed" if is_reached else "window not completed",
        "kept_c": main, "kept_a": variants["A"], "kept_b": variants["B"], "status": status,
        "disagreeing": "; ".join(disagreeing),
        "kept_c_mark_minus_10": mark_variants[-10.0], "kept_c_mark_plus_10": mark_variants[10.0],
        "longest_stretch_c_mm": round(found["c"]["longest_stretch_mm"], 1),
        "stretch_c_pictures": None if found["c"]["stretch_from"] is None else f"{found['c']['stretch_from']}-{found['c']['stretch_to']}",
        "largest_offset_c_mm": round(float(window[numpy.nanargmax(numpy.abs(window))]), 1) if numpy.any(numpy.isfinite(window)) else None,
        "largest_offset_a_mm": round(float(numpy.nanmax(numpy.abs(table["offset_a_mm"].to_numpy()[:last_index + 1]))), 1),
        "largest_offset_b_mm": round(float(numpy.nanmax(numpy.abs(table["offset_b_mm"].to_numpy()[:last_index + 1]))), 1),
        "pictures_without_c": int(numpy.sum(~numpy.isfinite(window))),
        "left_line_picture_c": left_index,
        "left_line_arc_mm": None if left_index is None else round(float(arcs[left_index]), 0),
        "ending": ending
    }



def agreement_text(steps: pandas.DataFrame, verdicts: pandas.DataFrame) -> str:
    """How far the three measures agree, inside the scoring windows of sheets 03 and 04."""

    lines = []
    in_window = []
    for _, verdict in verdicts.iterrows():
        run = steps[steps["label"] == verdict["label"]]
        last = verdict["mark_picture"] if pandas.notna(verdict["mark_picture"]) else run["step"].max()
        in_window.append(run[run["step"] <= last])
    window = pandas.concat(in_window)
    for first, second in [("c", "a"), ("c", "b"), ("b", "a")]:
        difference = (window[f"offset_{first}_mm"] - window[f"offset_{second}_mm"]).dropna()
        size = difference.abs()
        lines.append(f"{first.upper()} - {second.upper()}: {len(difference)} pictures; mean {difference.mean():+.2f} mm, median size {size.median():.2f} mm, "
                     f"90th percentile {size.quantile(0.9):.2f} mm, largest {size.max():.1f} mm; over 5 mm in {int((size > 5).sum())}")
        both = window[[f"offset_{first}_mm", f"offset_{second}_mm"]].dropna()
        is_over_first = both[f"offset_{first}_mm"].abs() > 12.0
        is_over_second = both[f"offset_{second}_mm"].abs() > 12.0
        lines.append(f"      over 12 mm by {first.upper()} {int(is_over_first.sum())}, by {second.upper()} {int(is_over_second.sum())}, "
                     f"by one and not the other {int((is_over_first != is_over_second).sum())}")
    arc_difference = (window["arc_c_mm"] - window["arc_mm"]).dropna().abs()
    lines.append(f"along the line, C against B: median {arc_difference.median():.2f} mm, 90th percentile {arc_difference.quantile(0.9):.2f} mm")
    return "\n".join(lines)



def main() -> None:

    steps = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_steps.csv")
    method_c = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s07_steps_method_c.csv")
    steps = steps.merge(method_c[["label", "step", "offset_c_mm", "arc_c_mm"]], on = ["label", "step"], how = "left")
    old_verdicts = pandas.read_csv(visit_common.EVIDENCE_FOLDER / "s03_verdicts.csv")
    rows = []
    for label in [label for label in old_verdicts["label"] if visit_common.sheet_of(label) in ("03", "04") and label != "demo"]:
        table = steps[steps["label"] == label].sort_values("step").reset_index(drop = True)
        old = old_verdicts[old_verdicts["label"] == label].iloc[0]
        row = {"label": label, "sheet": visit_common.sheet_of(label), "step": old["step_cm"]}
        row.update(score_run(table, visit_common.sheet_of(label), old["ending"]))
        rows.append(row)
    verdicts = pandas.DataFrame(rows)
    verdicts.to_csv(visit_common.EVIDENCE_FOLDER / "s08_verdicts.csv", index = False)
    with pandas.option_context("display.width", 300, "display.max_columns", 40, "display.max_colwidth", 60):
        print(verdicts.to_string())

    text = agreement_text(steps, verdicts)
    (visit_common.EVIDENCE_FOLDER / "s08_method_agreement.txt").write_text(text + "\n", encoding = "utf-8")
    print("\n" + text)

    def status_of(label: str) -> str:
        return verdicts.loc[verdicts["label"] == label, "status"].iloc[0]

    def main_of(label: str) -> bool:
        return bool(verdicts.loc[verdicts["label"] == label, "kept_c"].iloc[0])

    print("\nmodel E's five pairs (1.0 cm, 0.65 cm):")
    counts = {"1.0": [0, 0, 0], "0.65": [0, 0, 0]}
    for long_run, short_run in FINAL_PARAMETERS["model_e_pairs"]:
        print(f"  {long_run} {status_of(long_run):10s} (main {main_of(long_run)})   {short_run} {status_of(short_run):10s} (main {main_of(short_run)})")
        for step, label in [("1.0", long_run), ("0.65", short_run)]:
            counts[step][0] += status_of(label) == "kept"
            counts[step][1] += status_of(label) == "doubtful"
            counts[step][2] += main_of(label)
    for step, (sure, doubtful, main) in counts.items():
        print(f"  {step} cm: kept surely {sure}, doubtful {doubtful}, kept by the main reading {main}")
    print(f"  difference 0.65 - 1.0 by the main reading: {counts['0.65'][2] - counts['1.0'][2]}; "
          f"at most {counts['0.65'][0] + counts['0.65'][1] - counts['1.0'][0]}, at least {counts['0.65'][0] - counts['1.0'][0] - counts['1.0'][1]}")
    for name in ["without_03_pairs"]:
        print("\nmodel without sheet 03 (1.0 cm, 0.65 cm):")
        for long_run, short_run in FINAL_PARAMETERS[name]:
            print(f"  {long_run} {status_of(long_run)}   {short_run} {status_of(short_run)}")
    for name in ["team_runs", "own_length_runs", "sheet_04_runs"]:
        print(f"\n{name}: " + ", ".join(f"{label} {status_of(label)}" for label in FINAL_PARAMETERS[name]))

    # the figure: model E's five pairs, offset against tip travel
    figure, axes = pyplot.subplots(5, 2, figsize = (11, 13), sharex = True, sharey = True)
    for row_index, pair in enumerate(FINAL_PARAMETERS["model_e_pairs"]):
        for column_index, label in enumerate(pair):
            axis = axes[row_index, column_index]
            table = steps[steps["label"] == label].sort_values("step")
            verdict = verdicts[verdicts["label"] == label].iloc[0]
            axis.axhspan(-12.0, 12.0, color = "#e8eef6", zorder = 0)
            axis.plot(table["tip_travel_mm"], table["offset_a_mm"], color = "#b0b0b0", linewidth = 1, label = "A (12-picture map)")
            axis.plot(table["tip_travel_mm"], table["offset_b_mm"], color = "#b0b0b0", linewidth = 1, linestyle = "--", label = "B (drawing on the whole run)")
            axis.plot(table["tip_travel_mm"], table["offset_c_mm"], color = "#1f4e8c", linewidth = 2, marker = "o", markersize = 3, label = "C (drawing on the same picture)")
            if pandas.notna(verdict["mark_picture"]):
                axis.axvline(table["tip_travel_mm"].iloc[int(verdict["mark_picture"])], color = "#555555", linewidth = 1, linestyle = ":")
            step_text = "1.0 cm" if column_index == 0 else "0.65 cm"
            axis.set_title(f"{label}, {step_text}: {verdict['status']}", fontsize = 10)
            axis.set_ylim(-30, 30)
            axis.grid(alpha = 0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("puck left of line (mm)")
    for axis in axes[-1, :]:
        axis.set_xlabel("tip travel (mm)")
    axes[0, 0].legend(fontsize = 8, loc = "lower left")
    figure.suptitle("Model E, sheet 03: the puck's centre against the line's centre line (band = 12 mm; dotted = the 119 mm mark)", fontsize = 11)
    figure.tight_layout()
    figure.savefig(visit_common.EVIDENCE_FOLDER / "s08_pairs_offsets.png", dpi = 110)



if __name__ == "__main__":
    main()
