# Critique of "What the model should predict, and how it stops" (key targets), lens: exam risk

Written 9.11.26 for Team M. Read only; nothing was run on the robot. New numbers come from
`exam_risk_checks.py` in this folder (printout `exam_risk_checks.out.txt`, sections A to J), which reads frames.csv, the
audit's view table and the saved per-picture outputs in `../` (end-to-end exact model on held-out sheets 03, 05, 06;
final candidate on 01, 02). Markers: [measured], [calculated], [literature] (checker marked supported or corrected),
[judgement].

## Verdict

Stands with changes. Predicting the exact step (tip forward, tip left, twist) with its own stop output, and executing
the direction with a fixed 1-cm step, is a sound first robot candidate. But the section optimises numbers scored on the
demonstrators' own pictures, and four things that decide finishing at the exam are left open or settled too early:
the stop threshold for the model that will really run, recovery (backing up), the joint 4 clamp, and the step budget.

## What my checks support in the section

- No twist-in-place rule: a rule "twist only when the predicted twist is 3 deg or more and the step is short" would
  fire mostly at the wrong pictures (15 to 18 % of its firings were demonstrated twists in place; it catches 3 to 33 %
  of them) [measured; H].
- No smoothing needed against dithering: consecutive predictions flip the twist sign on 0.5 to 1.2 % of pairs against
  3.8 to 16.7 % in the demonstrations, and reverse the direction (over 90 deg) on 0 to 0.7 % against 4.8 to 10.0 %
  [measured; E]. Closed-loop dithering must still be counted on the robot [judgement].
- The direction holds up near the joint limit: end-to-end exact within 45 deg 0.714 (|joint 4| under 30 deg, n 2,546)
  against 0.661 (105 deg or more, n 283) [measured; A]. Start pictures are fine: 0.850 against 0.717 for always forward
  (first 3 moving pictures, n 300) [measured; G].
- The section's threshold rule (most stops right, the later one on ties) beats an early-stop-averse rule (right minus
  2 x early): right 0.364 / early 0.265 / never 0.348 against 0.096 / 0.200 / 0.704 (one picture, threshold from the
  other two sheets) [measured; F].

## Problems, most severe first

1. **High. The threshold will not transfer to the exam model.** The exam model is retrained on all six sheets (audit
   11.2 point 8). Stop probability on the stop pictures moves with the seed (medians 0.301, 0.521, 0.494, 0.653 for
   four seeds of one probe; targets/stop_analysis.out.txt) and with the sheet (0.804 / 0.455 / 0.115 on 03 / 05 / 06;
   0.915 / 0.730 on the final candidate's 01 / 02) [measured; F]. On sheet 06 only thresholds 0.25 to 0.4 stopped
   right in 0.28 to 0.32; at 0.5 it was 0.20 right, 0.76 never [measured; F]. Fix: record demonstrations on two or more
   newly printed tracks that never enter training, and choose the exam model's threshold (and one picture or two) on
   those pictures; confirm on the robot.
2. **High. The stop caps the finish rate, so its data remedy is the top priority, not an optional extra.** Per held-out
   sheet the end-to-end exact model stopped right in 0.73 / 0.20 / 0.16 (one picture, threshold from the other sheets)
   while path 45 was 0.777 to 0.802; on sheet 05 even the best own threshold gave 0.23 right, 0.40 early [measured;
   stop_analysis, F]. Fix: schedule 3 to 5 printed tracks (line ending mid-paper, line along the edge mid-track, other
   dash rhythms, a sharper corner) before 20.12.26; train on some, seal at least two for calibration and the final test
   [literature: interactive_correction check, keep one practice track for evaluation only; evaluation_statistics
   check, many distinct tracks].
3. **High. Recovery is the closed-loop skill the offline trade-off cannot price.** The final candidate backs up within
   45 deg on 0.054 of 37 and 0.139 of 280 backward moves, never straight back (targets/final_backing_up.out.txt). The
   weakness sits where backing up is genuine: at |joint 4| under 30 deg 0.303 of 218 (end-to-end exact) and 0.123 of 57
   (final), against 0.523 of 44 at 105 deg or more [measured; A]. The demonstrators' first real move after a
   correction pause was backward in 148 of 245 cases (audit label_consistency-13); 130 of 210 demonstrations needed a
   correction (class_balance-10). Errors compound over a run [literature: Ross et al. 2011, Theorem 2.1, supported].
   The 4.5 points of path 45 lost by weighting are scored on the demonstrators' states, so "a bad trade" is a
   judgement, not a measurement. Fix: run the names-plus-exact model now (about 3.5 h CPU) and compare both on the
   robot from the first session, on the same starts, including starts with the puck offset from the block.
4. **Medium. The joint 4 clamp will bind, and nothing says what happens then.** Starting at a median 0.0 deg, the
   demonstrations passed 115 deg in 11 of 45 (sheet 03), 23 of 45 (04) and 10 of 25 (06) episodes [measured; B]. At
   |joint 4| of 105 deg or more the final candidate twists further outwards on 0.627 of 83 pictures (9 episodes, sheet
   02) against 0.265 demonstrated, and its summed twist (open loop, no feedback) passes 115 deg in 27 of 65 episodes
   against 3 for the demonstrated path [measured; C, D]. Fix: write the clamp rule (hold the twist, keep stepping the
   direction, log each binding); rehearse on sheets 04 and 06; re-run C and D on folds 03 to 06.
5. **Medium. The held wrist does not give the training views over a whole run.** Per-step reproduction (p90 0.5 to
   0.7 px) assumes the demonstrated wrist at each step; held at the episode's first value the picture centre lands a
   median 2.8, p90 37.7, max 115 px off at 640 px (over 20 px in 23.3 % of rows), and the audit refuted whole-run
   reproduction (AUDIT_SYNTHESIS table 8.3, Refuted, code_actions-4) [calculated]. p90 is about 9 px at 160 px against
   the final candidate's 4-px shift augmentation (config.py line 1713) [calculated]. At the exam a hand-taught start
   sets the wrist and hand for the whole run. Fix: fixed standard start values for joints 5 and 6 (the demonstrated
   median), shift augmentation to about 10 px at 160 px for the exam model [judgement].
6. **Medium. The safety budget can end a good run.** For a sheet not listed, config.step_budget_for falls back to 80
   steps (config.py lines 646 to 651, 854 to 864): every unseen track. Demonstrations without corrections took up to 184
   (sheet 04, 77 cm), 111 (05) and 101 (06) steps; even 2 steps per cm would have cut 2 of 16 (01), 2 of 9 (02) and
   2 of 15 (04) of them [measured; I]. Fix: about 3 steps per cm of a measured line, or a fixed cap near 250 steps (about
   5 min at 1.22 s); ask whether measuring the track before the run is allowed.
7. **Low. "Two in a row" offline is not the robot's second look.** Offline the second picture is the next demonstrated
   one (median 1.05 cm on, usually twisted), not a 0.5-cm look, so it overstates the filtering [judgement]. The case for
   two rests on 16 against 13 of 20 episodes on sheet 01 (Wilson 0.58 to 0.92 against 0.43 to 0.82) [calculated], while
   for the end-to-end exact model it raised early stops from 0.265 to 0.422 [measured]. Fix: one picture by default;
   the real second look only as a robot A/B.
8. **Low. The fixed 1 cm throws away the model's own slowing at the end.** Within 1 cm of the goal it predicts 0.39 cm
   against 0.35 demonstrated (targets/exact_analysis.out.txt), so 1-cm steps give the stop fewer looks and can push past
   the end, a view never trained on. Near the goal the final candidate's direction is weak: 0.537 of 257 within 45 deg
   against 0.502 for always forward, median error 41.2 deg [measured; G]. Fix: robot A/B of "step the predicted length
   (at least 0.3 cm) once the stop probability passes a low pre-threshold" [judgement].
9. **Low. Light is not considered for the stop.** In the darkest third of pictures the end-to-end exact model's stop
   probability reached 0.3 far from the goal on 0.041 of 2,508 pictures against 0.018 of 2,573 in the brightest third;
   the final candidate's direction was 0.596 against 0.710 [measured; J; confounded with sheet and place]. Fix: include
   the exam room's light in the stop calibration runs.
10. **Low. Literature chosen one-sidedly.** The closest real-robot result, planar pushing from pictures, had plain
    regression at 35.0, 55.0, 6.7 and 19.6 % against 85.0 to 48.3 % for an implicit model (Implicit BC, PMLR Table 6)
    [literature: action_representation check, claim 27]. DMD (wrist camera, predicted direction, fixed 1-cm moves;
    off-track data 20 to 80 % in 10 trials, with extra play data) supports the executor and recovery data
    [literature: interactive_correction check, claim 36, corrected]. SAIL measured faster execution in time, which
    bears on the 20 % speed cap more than on step length [literature: action_chunking check, claim 48].

## Better recommendation (short)

Keep the exact step with its own stop output and the fixed 1-cm direction executor as candidate one. Train the
names-plus-exact model now as candidate two and compare both on the robot with offset-puck starts. Do not carry a
threshold from the fold models: calibrate the exam model on sealed, newly printed tracks. Make those tracks (with
varied line ends) the top data item before 20.12.26. Define the joint 4 clamp behaviour and rehearse it on sheets 04
and 06. Fix standard start values for the wrist and hand and widen the shift augmentation. Replace the 80-step fallback
budget. One picture for the stop by default; two in a row, near-end step length and twist gain as robot A/B tests.
