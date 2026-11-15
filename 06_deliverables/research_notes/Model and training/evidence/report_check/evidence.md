# Evidence check of the training set-up analysis (report of 9.11.26)

Critic, lens: evidence. Written 9.11.26, about 23:45, for Emranul Hoque and Marshal Sebastine. Read only: nothing under
D:/AI_Project was changed, nothing was trained, nothing ran on the robot. The report itself was not edited.

Report checked: `D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/2026-11-09_training_set_up_analysis.md`
(1,645 lines).

## What I did

- Read the whole report, the five drafted sections and their ten critic notes, the three judge notes, the main
  session's parts (data_section, how_to_compare, code_set_up, robot_time, professor_questions, results_so_far,
  literature_takeaways) and `design/main_session_facts.md`.
- Read the audit synthesis where the report draws on it (sections 1, 2, 6, the picture_quality findings and section
  11), the action_representation literature check, `02_code/config.py` lines 1780 to 1815, `STATUS.md`,
  `05_meetings/requests_to_professor.md`, the queue script and its logs.
- Recomputed from `04_runs`: every frozen-probe JSON and every end-to-end fold file (means, ranges, per-sheet values,
  seed spreads, stop counts, best epochs, file times) with a read-only script:
  `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/report_check/extract_numbers.py`
  (printout `extract_numbers.out.txt` in the same folder). I also checked the situation table against
  `design/improvement/situation_errors.csv`, the round size against `design/improvement/robot_hours.out.txt` and the
  near-goal stratum against `design/targets/critic_exam_risk/exam_risk_checks.out.txt`.

## Verdict

The numbers are in very good shape. Every final-candidate number I recomputed matches its fold file (step within 45
0.632 to 0.779, mean 0.678; path within 45 0.735 to 0.856, mean 0.786; twist sign mean 0.760; stop at 0.5 with one
picture 101 / 41 / 50 of 209; the tuned rules 105 / 33 / 55 and 98 / 58 / 33). The frozen-probe tables (sections 4.1,
5.1, 5.4, A.2, A.3), the seed spreads of sections 6.1, 6.9 and 7.3, the end-to-end table of section 5.2 and A.4, the
data-audit section 3, the situation margins of 8.1, the Wilson intervals of 7.5 and the config line numbers all trace
to their sources. The problems below are a few wrong or stale statements, two misread numbers, a robot-time total that
adds two corrections wrongly, and places where one judge's view or a calculation is presented as the panel's view or
as a measurement.

## Problems, most important first

| # | Severity | Place | Problem (correct value and source) | Fix |
|---|---|---|---|---|
| 1 | medium | Section 1, "Backup"; section 2.1, "Main risks" | "Backed up far better after a drift (0.66 against 0.39)" and "backing up after a drift: 0.342 of 1,894". These numbers count all backward-labelled moves, not moves after a drift. The audit refuted reading them as recovery: about half are ordinary pushing seen through a turned camera, and only 15.6 % sit in the first six rows of a correction (audit class_balance-10). The 0.39 is the earlier uncleaned exact run; the first set-up got 0.426 of the same 1,054 moves (judge 2). Measured recovery evidence is mixed: a tie, 0.605 against 0.605, over all 2,921 pictures after a correction (validation V4), and 216 against 169 of 500 like for like in the first five rows (judge 1). | Say "on backward moves of sheets 03, 05, 06 the names read as a mean direction got 0.66, the first set-up 0.43; whether this helps recovery after a drift is not shown offline". |
| 2 | medium | Section 2 intro and 2.3, source line "[judges 1 to 3 ...]" | Ranks 3 to 7 are judge 1's order only. Judge 2: small network (exact) 3rd, chunk 4th, final candidate with the tuned stop 5th, names only 6th, mirror 7th, frozen 8th. Judge 3: chunk 2nd, final candidate with the tuned stop 3rd, small network 4th, names plus exact 5th, class head with or without mirror 6th, frozen 7th. | Label the table as judge 1's order and add one line on where judges 2 and 3 differ. |
| 3 | medium | Section 4.3, "Twist size" | "The final candidate under-twists on average", followed by least-squares gains of 0.65, 0.55, 0.94, 0.96, 0.66 and 1.24. A gain of the demonstrated twist on the predicted one below 1 means the demonstrators twisted less than predicted for a given prediction; it is not evidence of under-twisting (the targets draft reads a slope above 1 as "scale up"; judge 2 reads these gains as an over-twist risk; judge 3 as gain 1 being the upper end). The next bullet even shows the summed open-loop twist passing 115 degrees in 27 of 65 episodes against 3 demonstrated. The only measured under-twisting is the median size on sheets 01 and 02 (1.28 against 4.92 and 3.43 against 5.27 degrees; targets section 2.3). | Restate: median predicted twist is smaller than demonstrated on 01 and 02, but where the model twists, the demonstrations twisted 0.55 to 0.96 times as much on five sheets, so no gain above 1 is supported; gain 1 is a judgement. |
| 4 | medium | Section 7.4, sentence after the phases table; per-trial table; 11.1 "Guess" | "With the measured set-up time of 1.86 minutes per trial the total is about 8 hours." The two critics' corrections are separate: sheet and end changes add 1.4 to 2.9 h (validation evidence critic: 7.7 to 9.2 h), and the measured 1.86 min set-up (0.75 + 0.19 + 0.92 min, `design/improvement/episode_timing.out.txt`) instead of the guessed 1 min gives about 8 h on its own (validation exam-risk critic). Together, calculated: 6.3 h + about 1.7 h (120 trials x 0.86 min) + 1.4 to 2.9 h = about 9.4 to 10.9 h. The per-trial table, the phase times and 11.1 still rest on the guessed 1 minute although a measured value exists, and section 8.4 uses a third base (2.5 to 2.8 min per trial on sheet 01 in `robot_hours.out.txt`, against 1.6 min in the 7.4 table). | Use the measured 1.86 min, give the combined total (about 9.4 to 10.9 h, plus phase 1a), and say which base each robot-hour figure uses. |
| 5 | medium | Section 11.3, row for the mirror and chunk runs | "The remaining sheets 01, 02, 04 are presumably meant." Wrong. The queue script defaults to folds 03, 05, 06 (`run_end_to_end_queue.sh` line 6); `logs/e2e_queue_3.log` shows the mirror run restarted at 23:02; its fold 03 file was rewritten at 23:21 with identical numbers (within 45 0.6703, 26 / 3 / 14 of 45) and fold 05 reproduced as well (`logs/e2e_resnet18_finetune_mirror.log`). So the main session's description was right: these runs repeat folds already on disk, add no evidence, and hold up the backup and the second seed until about 02:30. The report lists these queue files in A.6. | Correct the row (re-runs of 03, 05, 06, duplicates of the folds used above) and tell the main session the reruns duplicate existing results. |
| 6 | medium | Section 4.1, second bullet | "0.9 and 2.2 points in the probes" are means over four seeds (exact 0.603 / 0.691 against the names read as a mean direction 0.594 / 0.669; seeds.json and designs.json, recomputed). The table just above shows only the default seed, where the names read as a mean direction lead (0.605 against 0.591; 0.682 against 0.673). A reader cannot trace the bullet from the table. | Add "four-seed means" with the values, or the four-seed ranges for the mean-direction row (step 0.578 to 0.605, path 0.654 to 0.682). |
| 7 | medium | Section 2.2, "Why this backup" and "Main risks" | 216 (names) against 151 (final candidate) of 500 in the first five rows after a correction compares an uncleaned run with a cleaned one; like for like (both uncleaned) the exact model got 169 (judge 1), so the head's gap is 47, not 65. The "tie (0.605 and 0.605)" is the most-likely-class reading (validation V4; judge 3 counts class recall), not the mean direction the backup would execute. | Give 216 against 169 like for like beside the 151, and say which reading the tie uses. |
| 8 | low | Section 2.1, third bullet | "The most right stops: 47 of 99 ... against 42 of 100" (and 41 for the mirrored names head). The gaps, 5.5 and 6.5 points, are below the report's own bar (6.1: under about 7 points of stopped right is not evidence). The caveat "the size of the gain is uncertain" understates this: even the order is not established. | Say the first set-up's stop lead on 03, 05, 06 is inside the seed noise. |
| 9 | low | Section 4.1, "The chunk"; 2.3 rank 3 | "Against the final candidate ... its path lead is 4.0 and 3.9 points on sheets 03 and 05." On sheet 06 the final candidate leads, 0.810 against 0.770 (fold files, not paired); over the three sheets the chunk leads by about 1.2 points (0.801 against 0.788). | Add sheet 06 and the three-sheet mean. |
| 10 | low | Section 3.3, second bullet (under "[measured]") | "Smoothing the outputs over time would lag behind real turns" is a guess in the audit (label_consistency-4); label_consistency-17 only says smoothing is not supported. Measured offline, averaging the direction over 2 or 3 pictures changed path within 45 by +1.4 and +1.8 points for the frozen ResNet-18 (0.617 to 0.631 and 0.635) and by +0.3 and -0.5 for the final candidate (0.772 to 0.775 and 0.767; `results_so_far.md`). | Mark it judgement, give the measured numbers, and keep the literature caution (temporal averaging lost in closed loop under noise). |
| 11 | low | Section 5.2, bullet on line-and-puck maps | "-0.4 to 0.0 points of step within 45." Correct: -0.4, -1.0 and 0.0 on sheets 03, 05, 06 (0.630 against 0.635, 0.625 against 0.635, 0.623 against 0.623; fold files; the models draft says the same). | "-1.0 to 0.0 points". |
| 12 | low | Section 6.7, last bullet | "The exact runs' validation loss was still creeping down at epoch 8 (0.487, 0.535, 0.572)" holds for the earlier exact run only. For the final candidate it rose at epoch 8 on sheet 01 (0.5503 to 0.5507) and sheet 04 (0.6754 to 0.6774, lowest at epoch 5, 0.6354) (fold files, history). | Report both runs; this weakens the case for the 16-epoch test. |
| 13 | low | Section 6.2, mirror test | "The plain run kept its best-validation weights from epochs 7 and 5." The comparison covers three sheets: plain epochs 7, 5 and 7, mirrored 8, 8 and 8 (fold files). | "Epochs 7, 5 and 7". |
| 14 | low | Section 9, paragraph under the table | "The training-settings draft listed 31 to 73 hours." The draft listed 31 to 53 hours; 31 to 73 is its evidence critic's correction (critic note, problem 6). | Attribute the 31 to 73 to the critic. |
| 15 | low | Section 8.4, totals paragraph | "About 5 to 8 hours a week until 20.12.26" spreads 31 to 46 h over six weeks (exam-risk critic), but the report's own plan (section 9) has no robot hours in 9.11 to 15.11, so five robot weeks: about 6 to 9 h a week (calculated). | Use the five-week figure. |
| 16 | low | Sections 5.5 and 6.9 | "One ResNet-18 configuration on all six held-out sheets takes 2.0 to 4.9 hours" is six times 20.4 to 48.8 min, computed before the 55.0 min (final candidate, sheet 03) and 68.2 min (mirror, sheet 06) folds; the same table gives 20.4 to 68.2 min per sheet, so a loaded six-sheet run can take about 5.5 to 6.8 h. | Widen the range or tie it to a quiet laptop. |
| 17 | low | Section 3.5, third bullet | Hands in 15 + 36 + 1 = 52 census pictures, then "53 of the 55 hand pictures in that census". The 55 are the challenge checker's census file (audit picture_quality-12, 11.1); the audit elsewhere counts 51 hands found. | "53 of the 55 pictures in the checker's hand-census file". |
| 18 | low | Markers in 2.1, 2.2, 4.4 | Calculated or judged statements marked [measured]: the path margin "5.5 to 38.0 points" (calculated); "lost 4.6 points" (calculated by judge 1); "so it keeps the camera turning on loops" (a prediction for a model never trained); "a waypoint was cut after 1 cm or 5 degrees" (read in the code). | Mark them calculated, judgement or read in the code. |
| 19 | low | Section 1, "Build first" and "In practice" | "Trained on all our pictures": every model so far is a fold model trained on five sheets; no all-sheets model exists (report 11.2). "Where the line goes next": path within 45 compares with the demonstrated tip three steps later, not with the line. | Reword both. |
| 20 | low | Section 4.4 | "A fixed 1.0 cm, the demonstrated median" against the table's demonstrated median of 1.09 cm. | "Close to the demonstrated median of 1.09 cm". |
| 21 | low | Section 2.2, third bullet | "Matched the exact step overall (0.602 against 0.591, path 0.677 against 0.673)": one seed, against the default exact seed, which was the lowest of the four exact seeds (four-seed means 0.603 and 0.691). | Say "at the same seed". |
| 22 | low | Section 2.4, cleaning row | "Idle actions made single-step policies stall": the checked form says not removing idle actions contributed (action_representation check, Diffusion Policy Section 6.1; targets draft 2.7: one contributing cause). | Use "contributed to". |
| 23 | low | Section 8.3, "The idea" | ThriftyDAgger is listed among real-robot successes; its checker notes the behaviour-cloning baseline had less data, and the improvement evidence critic notes the comparison is within chance. | Qualify or drop ThriftyDAgger. |
| 24 | low | Section 2.3, rank 5 | "Backs up best": on straight-forearm backward moves the plain names head did better (155 of 218, 0.71, against 148, 0.68; judge 1). The mirrored head leads only in the first five rows after a correction (221 against 216 of 500). | "Best after a correction, near the plain names head on backing up". |
| 25 | low | Section 2.5, first bullet | "Backward moves within 45 degrees reach about 0.45" is judge 2's condition for making the backup the first set-up, used here as the condition to keep it as backup. | Attribute it as judge 2 framed it. |
| 26 | low | Section 4.5, stop designs table | The final candidate row is pooled over 209 episodes (0.483 / 0.196 / 0.239); the other rows are means of fold shares. Its mean of fold shares is 0.460 / 0.199 / 0.262 (metrics.json summary). | Use the same aggregation in every row, or mark the difference. |
| 27 | low | Status table, last row | "None of their numbers is used or guessed here", while 11.3 and A.4 use the exact run's sheet-01 fold. | Add the exception. |
| 28 | low | Section 10.1, first sentence | "None of these is on requests_to_professor.md yet": the printing part of item 3 is question 15, the light and start parts of item 5 are 19 and 18, the offline part of item 6 is question 10, all open on that list. | "None of these is on the list in this form". |
| 29 | low | Sections 2.1, 5.5, A.4 (sources that changed after 22:36) | `resnet18_finetune_exact/metrics.json` was overwritten at 23:01 by the run on 01, 02, 04 (it now holds those folds and 16.7 ms per picture), so the 03, 05, 06 numbers and the 21.5 ms now trace to the fold files only; the mirror fold files are being overwritten by the rerun (fold 03 now says 19.2 minutes, A.4 says 28.4). Also "21.5 to 37.4 ms" is given for the final candidate, whose own measured time is 37.4 ms; 21.5 ms is the earlier exact run. | Cite the fold files, refresh the minutes after the queue, and give 37.4 ms for the final candidate. |

## Judgement presented as a measured fact (summary of the above)

- The backup's case as recovery after a drift (problem 1).
- "Under-twists on average" as a reading of the gains (problem 3).
- The ranking as the three judges' order (problem 2).
- Smoothing that would lag behind turns (problem 10).
- "So it keeps the camera turning on loops" for the untrained backup (problem 18).
- "The most right stops" inside the seed noise (problem 8).

## What checks out (spot list)

- Final candidate per sheet (fold files): direction, path and twist rows; step, path, twist and lean sign; stop counts at
  0.5 (12/4/1, 24/2/18, 33/9/1, 18/13/4, 12/11/5, 2/2/21); tuned thresholds 0.4, 0.1, 0.1, 0.1, 0.4, 0.05; minutes
  (223 in all); 37.4 ms per picture.
- Earlier end-to-end runs (fold files): every value in section 5.2 and A.4, the stop counts of every run, best epochs
  7 / 5 / 7 (names), 15 / 16 / 16 (small network), 7 on sheet 05 (chunk), the exact run's epoch-8 losses, the exact
  run's sheet-01 fold of 22:24.
- Frozen probes: all of 4.1, 5.1 (every per-sheet value and the "beats always forward" counts), 5.4, 6.3, 6.8 (lower on
  all six sheets for the linear head and for label smoothing, recomputed), A.2, A.3; the seed spreads 3.1 / 2.4, 3.4 /
  3.5, 5.9 / 4.3, the per-sheet seed ranges (9.1, 9.8, 10.7, 16.0, 12.4; 11 to 26 and 7 to 32) and the seeds-needed
  table (standard deviations 1.24 to 1.70 and 2.0 to 3.6).
- Data audit numbers of section 3 (tables 1.2 and 6.1, class_balance-4 to -13, label_consistency-1 to -12,
  picture_quality-1 to -12, section 11.1), the 469 dropped rows and 18,032 training rows.
- Situation margins of 8.1 (`situation_errors.csv`), the near-goal 0.537 of 257 (exam_risk_checks part G), the round of
  1,624 rows (`robot_hours.out.txt`), the refit +2.13 and +4.57 (improvement evidence critic).
- Config lines 1797 to 1804, 1806 to 1813, 1814; question numbers 8, 9a, 10, 15, 18, 19 in requests_to_professor.md;
  the dates in STATUS.md (meeting 2 on 9.11 at 09:50, PRIMUSS 9 to 17.12, freeze 8.1.27, hand-in 10.1.27 23:59).
- Literature figures quoted in 2.4, 5.3, 6.2, 6.6, 6.7, 7.2, 7.5 and 8.2 match the checked notes or the drafts that
  cite them (the two wording issues are problems 22 and 23).
