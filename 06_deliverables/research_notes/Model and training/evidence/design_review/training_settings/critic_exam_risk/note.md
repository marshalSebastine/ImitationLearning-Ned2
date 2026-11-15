# Exam-risk critique of "Training settings" (key training_settings, lens: exam risk), 9.11.26

Question: does following the section maximise the chance of finishing UNSEEN tracks at the exam, and then the
finishing time? Nothing here was run on the robot and nothing was trained. I re-counted the saved held-out-sheet
predictions of the end-to-end runs and ran the two saved final-candidate fold models (sheets 01 and 02, 2 threads) on
their own held-out pictures, clean and changed. Scripts and printouts are in this folder:

| Script | Printout | What it measures |
|---|---|---|
| stop_and_twist_risk.py | stop_and_twist_risk_output.txt | stop outcomes per sheet for every threshold and 1 to 3 pictures in a row, against the rule tuned on inner validation episodes; where early stops fire; twist bias and shrinkage |
| stop_rule_cross_sheet.py | stop_rule_cross_sheet_output.txt | a stop rule chosen on the other held-out sheets, applied to each sheet |
| stress_test_final_candidate.py | stress_test_final_candidate_output.txt | final candidate fold models 01 and 02 under darker and brighter light, colour casts, 4 and 8 px shifts, 3 degree roll, blur; predicted step length |
| recovery_stratum.py | recovery_stratum_output.txt | direction on pictures recorded after a hand correction (the closest thing to recovery states) against the rest |
| probe_stop_variants.py | probe_stop_variants_output.txt | stop outcomes of the frozen-probe stop variants, fold by fold |
| sheet05_early_stop_pictures.py | sheet05_early_stop_pictures_output.txt | the pictures where the sheet 05 early stops fire (2 of 17 looked at) |

Caveat on all of it: one seed per run; sheets 01 and 02 come from the final candidate, sheets 03, 05 and 06 from the
earlier exact run (same design and stop output, before the audit cleaning); the final candidate's sheets 03 to 06 were
not finished at 19:46.

## What I measured

1. **The recommended stop rule adds early stops on unseen sheets.** Against the same two-in-a-row rule at the default
   threshold 0.5, the threshold tuned on the inner validation episodes cut never-stops by 32 (65 to 33) but added only
   11 right stops against 17 early stops and 4 stops 2 to 3 cm short (five held-out sheets, 165 episodes: right 82
   against 71, early 44 against 27). On the final candidate's own folds: right 35 and 35, early 11 against 4 of 65 (sheet 02: tuned
   threshold 0.1, early 8 against 1 of 45). Frozen probes, exact head, 208 episodes: tuned two-in-a-row right 44, early
   87, against right 58, early 55 at 0.5 with one picture (probes stop.json, exact_stop_head_on_stop_picture). Tuned thresholds ranged 0.1 to 0.4 between folds of one
   recipe. A rule chosen on the other four sheets (right minus early) gave right 76, early 30, never 53 of 165.
2. **The loop sheet stops mid-track whatever the threshold.** Exact model, sheet 05: 17 of 30 episodes stop early at
   0.5 (19 with the tuned rule), a median 14.2 cm before the goal (4.9 to 19.8) at a median 34 % of the episode; a
   threshold strict enough to remove them (0.8, two in a row: 2 early) leaves 1 right stop in 30. The two pictures I
   looked at show the puck at the paper's edge, mostly table ahead and the line turning along the edge: the stop cue
   the audit named (paper edge, no line ahead) appears mid-track. Class head: 5 of 30, median 17.8 cm early.
3. **The predicted step shrinks on an unseen sheet and near the goal.** Final candidate, moving pictures: median
   predicted step 1.00 cm against 1.12 cm demonstrated on sheet 01 (ratio of means 0.81), 0.68 against 1.11 cm on
   sheet 02 (0.61); within 2 cm of the goal 0.22 and 0.36 cm, with 64 % and 32 % under 3 mm. No offline measure in the
   runs reports step length (only its angle is saved).
4. **Recovery-like pictures are much harder for the exact model.** Pictures after a hand correction: final candidate
   within 45 0.493 (1,142 moving; always-forward 0.389) against 0.762 elsewhere (1,861; always-forward 0.729);
   backward moves recalled 18 of 241. Sheets 03, 05, 06: exact and class head both 0.605 after corrections (2,921),
   exact 0.731 against 0.688 elsewhere; backward moves recalled 146 against 284 of 516.
5. **Light and camera changes hit the stop, not the direction.** Within 45 stayed within 2.0 points of clean on both
   sheets for every change. Sheet 02 at half brightness: early stops 2 to 10 of 45 at (0.5, one picture), 1 to 9 at
   (0.3, two), 8 to 21 with the tuned rule (0.1, two); 8 px shifts: 8 to 14 and 11 with the tuned rule. Sheet 01: early
   1 to 4 of 20 in every condition.
6. **The twist is under-predicted.** Exact models: mean predicted |twist| 0.50 to 0.68 of the demonstrated on sheets
   01, 03, 05, 06 (1.06 on 02), slope 0.15 to 0.45. Demonstrations reach 118.8 to 119.4 degrees of joint 4 on sheets
   02, 03, 05, 06 (limit 119.75). What this does in closed loop is unknown.

7. **Stop comparisons in the section count right stops only.** Same threshold 0.5, sheets 03, 05, 06: exact head 25
   early stops of 100, class head 11 (fold files). Sheets 03 and 05: mirrored class head right 35, early 11, never 27
   of 75; exact head right 37, early 24, never 12.

## Verdict and recommendation (judgement)

Stands with changes. The training recipe can go to the first robot tests unchanged, but the section's stop rule should
not, and its list of next experiments is ordered by offline within-45 effects inside the seed noise while the failures
that decide the exam (early stops, slow or stalled creeping, recovery, dim light) are not on it.

1. Stop rule: drop the inner-validation tuning; fix one rule from the six fold models' held-out predictions with an
   early stop counted as a lost run (until then two pictures in a row at 0.5); confirm with fold models 02 and 05.
2. Make the stop at paper edges the first retraining topic: printed practice tracks with lines along an edge and ends
   away from an edge (ask the professor), stop targets beyond the last picture tested in probes, early stops per
   sheet as the deciding number.
3. Execute the direction with a fixed or minimum step of about 1 cm, or test it against the predicted length; switch on
   the step budget and a stall counter; save the predicted step vector in every run.
4. Add to every candidate's offline report: early stops per sheet, a stress test (half brightness, 8 px shift, 3 degree
   roll) and the after-correction stratum. Keep the blur and noise margins.
5. Write the robot test before running it (after the team decides): fold models on held-out sheets 02, 03, 05, a dimmed
   room, puck 1 to 2 cm off the line, an end away from the paper edge; count finishes, early stops, stalls, dithering,
   joint 4 range. Clamp joint 4 at about 115 degrees first.
6. Average the two exam-model seeds offline instead of choosing between them on the robot; spend robot time on the
   failure tests and one human-gated correction round.
7. Ask the professor whether the robot must stop by itself and how an overshoot past the end is judged.
