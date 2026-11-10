# Evidence check of "Improving after the first model" (key improvement)

Critic, lens: evidence. Checked on 9.11.26 against the files the section cites. Read only: nothing under D:/AI_Project
was changed. The one script I ran, `refit_without_one_sheet.py` (this folder), reads only
`04_runs/2026-11-09_frozen_feature_probes/data_amount.json` and writes `refit_without_one_sheet.out.txt` here
(run with `python -I -B`, so no bytecode cache was written).

## What holds

- Every number I traced matches its file:
  - the end-to-end exact folds (`2026-11-09_resnet18_finetune_exact/metrics.json`: within_45_deg 0.6995 / 0.6673 / 0.675;
    stopped right at 0.5 31/45, 6/30, 5/25; tuned two 34/45, 5/30, 8/25; early 19/30; never 19/25);
  - the final candidate's folds 01 to 03 (0.7785 of 578, 0.6318 of 2,425, 0.6802 of 3,458; tuned two 16/20, 19/45, 31/45);
  - the situation tables (`situation_errors.csv`, `situation_errors_by_sheet.out.txt`), the stop and paper-edge tables
    (`stop_by_end.out.txt`), the data-amount table and fit, the recovery ablation, and the timings;
  - the robot hours: `robot_hours.py` reproduces `robot_hours.out.txt`, and steps B, C, D and E and the totals
    (24 to 37 h, or 26 to 44 h with E) add up from it;
  - the audit citations (tables 9.1, 10.1, 10.2 and 10.4, view_and_tracks-2 and -16, class_balance-10, notes_context-10,
    cleaning rule B2, 11.2 point 20, question 18) and the code citations (record_episodes.py 898-903 and 1172-1211);
  - the professor-question numbers (15 = printing tracks, 19 = exam light, in `05_meetings/requests_to_professor.md`).
- Every literature claim used was marked supported (or corrected, and then used in the corrected form) by its checker.
  Lin et al. 2025 and Xie et al. 2023 are the checkers' own additions ("evidence the researcher missed"), so the
  claims used from them were each read by one checker only. The evaluation checker also opened Lin et al., but only
  for its test design.
- My refit of the section's own regression: the sheet effect on direction survives dropping the one-sheet runs,
  at +2.13 points per doubling (bootstrap +0.90 to +3.32, n 108), and at +3.21 (+1.00 to +5.47) without the one- and
  two-sheet runs (n 90). With all runs it reproduces +1.70 and +2.51 exactly. So "ask question 15 now" stands on the
  frozen-probe evidence.
- The stop's sheet effect does not survive the same refit: +4.57 (-0.85 to +9.63) without the one-sheet runs and
  +2.09 (-7.71 to +11.75) without the one- and two-sheet runs.

## Problems (most important first)

1. **Weak spots are ranked by raw within-45 only.** The always-forward column in the same CSV is not used. Measured
   against it, bends are not a model weakness:
   - End-to-end, the margin over always-forward is +20.2 points in bends (0.633 against 0.431) and +17.6 on straight
     stretches (0.798 against 0.622).
   - Frozen exact probe: +8.6 in bends against +1.2 on straight stretches.
   - The probe's off-line margin equals its on-line margin (+6.7 against +5.8), so "the frozen probe shows the same"
     is true only in raw numbers. Per sheet, the probe has no off-line gap at all on sheet 06 (0.483 of 317 against
     0.481 of 1,193), and only 3.4 points on sheet 04 and 4.5 points on sheet 01 (68 pictures).
   - What survives the adjustment: end-to-end off-line (+15.6 against +22.1), radius under 1.5 cm (+8.4 against
     +13.7) and the last 3 cm (+11.0 against +19.7). For the probe: the crossing, where it scores below always-forward
     (0.590 against 0.616).
   - Picture counts are pooled with no episode grouping or intervals, although the audit asks for statistics grouped
     by episode (11.2 point 5).
2. **The mirror evidence is out of date.** Fold 06 of `2026-11-09_resnet18_finetune_mirror` finished at 20:37, before
   the section was saved at 20:45. It scored within 45 0.599 against 0.640 without mirroring (1,809 pictures): -4.1
   points. The three-fold mean is 0.643 against 0.654. So the result is "worse on the dashed sheet", not "no gain so far".
3. **"The three 100 % runs used identical training rows" is false.** The data-amount variants set "seed" (config.py
   1665-1668). That key also exists in TRAINING_TARGET_PARAMETERS (config.py 1436), so run_variant copies it into the
   target parameters (training_probes.py 602). The early-stopping episodes are then drawn with it (line 623). Each seed
   therefore held back a different 15 percent of each sheet's episodes, and the 3.3-point range is training noise plus
   split noise. Illustration: out 01, sheet 06 alone, seeds 1 and 3 scored 0.648 and 0.501.
4. **"2.6 times the direction gain" claims more precision than the fit has.**
   - Both projected gains are below the 3.3-point spread of single runs: +2.6 against +1.0 points (unrounded +2.65
     against +0.96).
   - They come from a frozen probe with a classes head, extrapolated beyond 5 training sheets. The recommendation
     drops these qualifiers.
   - Within the bootstrap ranges the ratio can be roughly 1.6 to 6 (my arithmetic).
   - The stop advantage of new tracks (+6.5 against +4.9) rests on a sheet term that crosses zero in my refit.
   - "The sheet curve does not yet flatten" rests on one step (4 to 5 sheets, +2.3) after steps of +1.3 and +1.2.
     "+3.2" is +3.1 unrounded.
5. **The literature is stretched in the recommendation.** "The literature (Lin et al. 2025; Xie et al. 2023) both
   favour more different tracks" goes beyond the sources:
   - Their "environments" are scenes, objects and visual factors, not path shapes.
   - The augmentation checker states that no opened paper tests new path shapes of the same appearance.
   - Both papers are single-read checker additions, and the link to tracks is the checkers' own judgement.
6. **The sealed tracks are used to choose a model.** Steps B and D both run the 10 sealed tracks, and D picks between
   two models; its 6.4 to 11.0 h equals two models times step B's trials. This goes against the cited
   evaluation_statistics checker on two counts: it advises choosing first and then one sealed run (Cawley and Talbot),
   and a second sealed set if the team retrains. It also contradicts the section's own "never for choosing settings".
7. **Robot hours leave out set-up for the 10 sealed tracks.** robot_hours.py charges 0.5 h per new training track
   (print, tape, teach) but nothing for the sealed tracks in B or D. On the section's own assumption that is up to
   about 5 h more in B, and part of it again in D (guess).
8. **The paper-edge result is framed one way.**
   - All eight rank correlations point the hypothesised way: never stopped +0.14 to +0.54, stopped right -0.09 to
     -0.49.
   - The end-to-end far ends (5 cm or more from the edge) were stopped right in 13 of 45 episodes, against 34 of 55
     for the near ends (tuned). This split is in the printout but not in the section.
   - At 0.5, sheet 06's far end B was never stopped in 9 of 15.
   - Only the tuned "never" split goes the other way (0.16 against 0.22).
   - The correct summary is "confounded with sheet; cannot be decided".
   - The guess for end A (a dashed end never seen) does not separate A from B: the whole sheet-06 line is dashed
     (audit view_and_tracks-6), and end B was stopped right 8 of 15 times.
9. **The recovery ablation reading is stronger than its precision.**
   - The key cell ("more than 2 cm off") has a seed range of 0.500 to 0.527 with all rows, which overlaps the range
     without corrections (0.504 to 0.510). The compared drops (-0.7 and -2.1 against -1.6) differ by about 1 point,
     inside that noise.
   - "Act like ordinary extra data, not like targeted recovery data" should read "no detectable difference".
10. **The round size is far below the studies cited for it.** 50 episodes give about 1,624 rows, 9 percent of 18,053
    (robot_hours.out.txt). The cited rounds were 20 to 33 percent of the initial data, and the interactive_correction
    checker (problem 1) flags this gap, but it is not carried into the plan. Claim 29's "barely used the small
    correction set" (-1.67 points on book flipping) is also dropped. That matters for step D's plain-mixing arm.
11. **Smaller literature points.**
    - "Human-gated take-over is the usual real-robot recipe (claim 5)": claim 5 is a definition, and "usual" is the
      researcher's unchecked summary.
    - ThriftyDAgger sits under "At equal data" although its checker says it was not equal data, and 12 against 10 of
      15 is within chance.
    - Laskey's labels were mouse labels on half-speed video, not typed, and the checker's counterpoint (Ross et al.'s
      drone) is dropped.
    - Codevilla's truck training also used two side cameras with adjusted labels, so "from one camera" overstates
      the match.
12. **Code reading is marked "measured", and part of the work is missing from the nine pieces.**
    - After "reguide", only `__exact_replay_decider` reads the new waypoints (record_episodes.py 1208-1211,
      1539-1545). A model decider must also hand over to the exact replay, and that piece is not among the nine.
    - So "version 1 reuses the take-over as it is" understates the work.
13. **A claim marked "Measured" is not measured.** "Early stops cluster where no line is visible ahead" cites
    picture counts, not where the model stopped. Sheet 06 has the most look-alike pictures (592) but had 1 early stop
    in 25 at 0.5.
14. **Small items.**
    - "Fell with practice" (80.7 s on sheet 01, 5 corrections, to 32.5 s on sheet 06) is confounded with sheet,
      because the sheets were recorded in order.
    - Step A's 13.9 h high end is not reproducible (13.6 h at the measured near-end cycle, 14.0 h with the sitting
      factor).
    - The final candidate's small margins over always-forward on sheets 01 and 02 (+3.2 and +6.7 points, slightly
      different rows) are not reported next to the older run's +15.8 to +27.3.

## Verdict

The recommendation stands with changes. Ask question 15 now, record new printed tracks if allowed, and run one pilot
and then one human-gated round. These rest on numbers that check out, and the direction effect of more sheets is
robust to my refit. Fix:

- the stale mirror result and the false "identical rows" statement;
- the always-forward-adjusted list of weak spots (off-line, radius under 1.5 cm, line ends; crossings for the probe;
  not bends in general);
- the precision of "2.6 times";
- the literature overreach on track variety;
- the sealed-track reuse;
- the missing sealed-track set-up hours;
- the paper-edge framing;
- the null-result wording of the recovery ablation.
