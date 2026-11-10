# Evidence critique of "What the model should predict, and how it stops" (key targets)

Written 9.11.26 by a critic sub-agent for Team M. Lens: evidence. I opened every cited file and re-read every number I
could. Nothing here was run on the robot; nothing in D:/AI_Project was changed. My own re-computation is
`extra_checks.py` in this folder (read only; it writes only `extra_checks.out.txt` here). Where its numbers overlap the
section's own printouts (final candidate sheets 01 and 02: backing up 0.054 / 0.139, stop 0.65 / 0.10, 0.80 / 0.15,
0.556 / 0.089, 0.422 / 0.178, far no-line p90 0.002 / 0.026), they agree exactly.

## What checks out

Almost every number matches its source: the cleaned row counts and label counts (data_facts.out.txt), the audit figures
(motion_statistics-3, -4, -5, -7, -8, -13; class_balance-5, -9, -10; label_consistency-3; code_actions-3, -4, -5, -6, -8,
-9, -11, -14, -15; view_and_tracks-2, -7; notes_context-3, -4, -23), the probe and end-to-end tables (comparison.md,
designs.json, seeds.json, both metrics.json files, the final candidate's fold files), the whole stop table of section 2.6
(stop_analysis.out.txt), the weighting tables (exact_probe.out.txt, stop_probe.out.txt) and the step-time table. The
literature claims used are all marked supported by their checkers, and the Chi et al. idle-action claim uses the
corrected form ("one contributing cause").

## Problems, most serious first

1. **High. Twist gain 1 is not supported for the model that will run.** The slope 0.97 (twist_calibration.out.txt) is
   pooled over the earlier end-to-end run's three sheets. Per sheet it is 1.05 (03), 0.59 (05), 1.19 (06). On the final
   candidate it is 0.65 (01), 0.55 (02), 0.94 (03), and the sign at a predicted twist of 3 deg or more is right on 0.72,
   0.87 and 0.93 of the twisting rows, not "0.91 to 0.94" (extra_checks.out.txt, lines A). The backup "x 1.2 on confident
   twists" rests on the same earlier run and was not checked on the final candidate, whose overall slopes are all
   below 1.
2. **Medium. The stop-threshold rule "never on validation episodes" has no measured advantage.** End to end, two in a
   row: validation-tuned 0.414 right / 0.350 early against 0.324 / 0.422 from the other sheets (comparison.md,
   stop_analysis.out.txt). Final candidate, one picture, validation-tuned against chosen on the other finished sheets:
   01 0.650 / 0.100 against 0.600 / 0.200; 02 0.556 / 0.089 against 0.533 / 0.044; 03 0.689 / 0.133 against 0.667 / 0.222
   (extra_checks; two other sheets each, so noisier than the planned five). Thresholds that vary between folds (0.05
   to 0.6; the final candidate already 0.1 to 0.7) do not by themselves show that they fail. The audit's reason (twins inside recording blocks, 11.2 item 4) is a method argument,
   and should be given as such.
3. **Medium. "Holding the wrist keeps the camera views of training" overstates code_actions-4.** The 0.5 to 0.7 px holds
   only when the wrist is set to the demonstrated value at every step. Holding joint 5 at the run's start shifts the
   picture centre by more than 20 px in 23.3 % of rows, in over a quarter of the rows of 77 of 210 episodes. The
   "such views exist in the 52 wrist-held episodes" judgement meets the audit's confound: all 25 sheet-06 episodes moved
   the wrist 10 deg or more (motion_statistics-7), Marshal's median range was 0.7 deg against 8.0 for Emranul.
4. **Medium. The lean output is dismissed with the wrong number and a refuted audit point.** The 5.4 mm per deg is
   joint 2; a lean output acts through joint 5 (0.54 mm vertical per deg, code_actions-14) with joints 1 to 3 re-solved.
   "The lean follows the reach, which a fixed-wrist solve produces by itself" repeats the part of motion_statistics-7
   the checkers refuted (r 0.713 explains only 51 % of the lean change).
5. **Medium. The length-rescale backup is fitted on another model's test sheets.** Gain 1.81 makes the earlier run's
   own held-out median match by construction. For the final candidate the matching gains are 1.12 (01) and 1.64 (02)
   (calculated from final_fold_check.out.txt); 1.81 would push most straight-sheet steps to the 1.5-cm clamp.
6. **Medium. "More different tracks helped more than more episodes" is a direction result used for the stop.** The same
   data-amount runs (frozen classes probe, comparison.md, stopped right at 0.5, mean of three seeds): doubling the
   episodes of the same five sheets raised the stop from 0.197 to 0.297, while five half-sheets (0.197) did no better
   than two or three full sheets (0.141, 0.218) at about the same picture count. For the stop, more stop examples
   helped; more sheets did not show a gain.
7. **Medium. The loss-weighting verdict comes from frozen features only.** It is applied to the end-to-end candidate,
   whose far no-line pictures already score low (p90 0.002, 0.026, 0.009 on sheets 01 to 03).
8. **Medium. The step budget is to be switched on without a value.** config.py sets 2.0 steps per cm; the demonstrated
   median rows per cm are 2.13, 2.08 and 2.33 on sheets 02 to 04 (notes_context-13), and sheet 05's demonstrated tip path
   (109.7 cm, motion_statistics-15) would use about 96 % of its 114-step budget at 1 cm per step (my calculation).
9. **Medium. Literature cited one-sidedly.** The action_representation checker calls the picture-based evidence mixed and
   lists the closest real task, planar pushing from pictures (Implicit BC, Table 6), where MSE got 35.0, 55.0, 6.7 and
   19.6 % against 48.3 to 88.3 % for the implicit model, and robomimic Square with pictures, 76.7 to 61.3 without the
   mixture head. Only the pro-regression numbers are quoted.
10. **Medium. "5 to 8 points" in the summary is the path measure only.** On step 45 the end-to-end lead is 2.6 points
    over the most likely class and 1.3 over the mean direction (per sheet -1.1 to +3.8, one seed); in the probes, at the
    default seed the exact step beats the most likely class on 3 of 6 sheets (-2.0 to +5.9), over all four seeds on 19
    of 24 sheet-seed pairs.
11. **Medium. The final candidate's sheet 03 has finished** (fold_without_path_03.json): step 45 0.680, path 0.801,
    twist sign 0.848, backward moves 0.412 of 340 (not "no better": 0.054 and 0.139 rested on 37 and 280 rows), stop at
    0.5 33 of 45 right and 9 early, all 9 at no-line pictures a median 4.7 cm from the goal; 55.0 minutes.
12. **Low.** One picture against two in a row is decided on 3 to 6 episodes per sheet (13 of 20 is 0.43 to 0.82, 16 of
    20 is 0.58 to 0.92, Wilson), mixing validation-tuned and other-sheet results; offline "two in a row" means two
    demonstration pictures, not a second look after a 0.5-cm step.
13. **Low.** The "about 3 points" seed range is from frozen probes; no end-to-end run was repeated with another seed, yet
    it gates the end-to-end chunk run.
14. **Low.** "1.6 to 1.7 times the driving time" is a calculation; the only measured short moves (3 to 9 mm, twist under
    1 deg, 83 steps) took 1.27 to 1.35 s each, 1.6 to 2.3 s per cm (data_facts.out.txt).
15. **Low, small mismatches.** Length-cue ceiling used step thresholds 0.2 to 0.8 cm, not 0.5 to 0.8, and its gain with
    other-sheet thresholds came from sheet 06 alone; bend radius is 0.43 to 0.46 cm (0.84 on 06), not 0.43 to 0.6; the
    "demonstrated median 2.37 deg" is over all rows of the bin; the 90.0 % stop-picture figure is on raw rows (210 stops);
    end-to-end folds took 20 to 55 min (final candidate 34.6, 48.8, 55.0), so "3.5 h for six sheets" is a lower
    bound; "no-line pictures come several in a row" was not measured; "its edge comes from the class weights" and "small because it does not know which way to
    turn" are interpretations marked measured.
16. **Low.** "3 to 4 runs out of 10" averages sheets that differ widely: for the end-to-end exact run 0.10 to 0.23 on
    sheet 05 and 0.69 to 0.76 on sheet 03, whatever the threshold rule; the final candidate gets 0.53 to 0.73 at 0.5 on
    sheets 01 to 03. "Where the paper edge filled the view" is stronger than the measure (no ink in the top half). The goal-distance stop, with
    the fewest early stops from other-sheet thresholds (0.290 / 0.153, median 4.3 cm), is left out of the backup without
    a reason.

## Verdict

The recommendation stands with changes: the exact step plus its own stop output is supported by path 45 (+4.6 points
over the names read as a mean direction on 3 of 3 sheets, +8.3 over the most likely class) and the twist sign (0.77
against 0.50). The executor numbers (twist gain, rescale gain, wrist, step budget) and the stop-threshold rule need their
evidence restated or re-measured on the final candidate's six folds.
