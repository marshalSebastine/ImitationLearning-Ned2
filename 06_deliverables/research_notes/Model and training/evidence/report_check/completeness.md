# Completeness check of the training set-up analysis (9.11.26)

**Checked:** 9.11.26, about 23:35, by a critic sub-agent for Team M (Emranul Hoque, Marshal Sebastine). Lens:
completeness against Emranul's request.

**Read only.** I did not edit the report. Nothing ran on the robot, nothing was trained, and nothing in `03_data/raw`
was touched.

**What I read:** the whole report
(`D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/2026-11-09_training_set_up_analysis.md`,
1,645 lines, last saved 23:14); the five design sections and their ten critic notes; the three judges' notes; the seven
report parts plus `report_parts/errors_resnet18_final_candidate.csv`; section 11 of `evidence/audit_synthesis.md` and
the findings named below; `05_meetings/requests_to_professor.md`; the run files in `04_runs` and the queue script and
logs in the scratchpad.

Short names used below: **scratchpad** is
`C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/`; **runs** is
`D:/AI_Project/.claude/worktrees/training-analysis/04_runs/`; **audit** is
`D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/audit_synthesis.md`.

---

## 1. Verdict

Every item of the request has a section, and most are done well. The data audit, the training settings, the case for
holding out whole sheets, the robot statistics and the professor questions are thorough, marked and sourced.

Before the report goes to Marshal, these gaps need closing:

- The stop is called the main risk, but no stop option other than more data reaches the plan, and the robot has no stop
  backup. Three measured options in the sources are left out or left without a conclusion (problem 1).
- Two results reached the disk before the report's last save and are missing or misdescribed: the earlier exact run on
  all six sheets, and the queue re-running the mirror and chunk runs (problems 2 and 4).
- Three named request items are thin: the lighting split's effect on the model, the 11 classes, and the at-goal label
  (problems 3, 6 and 7).
- Several measured quick experiments are left out: averaging the output over pictures, most of the stress test, and the
  final candidate's error analysis by situation (problems 9 to 11).
- The robot protocol and the plan miss working pieces: when to end a run, what counts as a stall or as leaving the line,
  a plan for fewer robot hours, an estimate for building the model mode, and a check that the model mode reproduces the
  saved predictions (problems 5, 8, 14, 15).

---

## 2. Coverage of the request, item by item

| Request item | Where in the report | State | Problems |
|---|---|---|---|
| (1) class balance per sheet | 3.2 | covered, but only before the cleaning | 29 |
| (1) label noise and consistency | 3.3 | covered | |
| (1) near-zero moves | 3.4 | covered | |
| (1) hands or blur | 3.5, 3.9 | covered; the look-by-eye list is incomplete | 22 |
| (1) duplicates | 3.6 | covered | |
| (1) lighting split | 3.7 | thin: how the data split, not how the model does in each light | 3 |
| (1) scores 1, 2 and 3 | 3.8, 6.4 | covered with frozen probes; the final candidate's result per score is missing | 10 |
| (1) measured, with guesses flagged | markers throughout | covered; one judgement sits under a measured marker (3.3) | 11 |
| (2) 11 classes | 4.1 | missing: the 11-class plan is never set against the 12 labels in the data | 6 |
| (2) exact step, chunks, mix | 4.1 | covered | |
| (2) stop and at-goal | 4.5 | stop covered; the at-goal label is never discussed; no stop backup | 1, 7 |
| (2) lean and forearm turns | 4.3 | covered | |
| (3) model options with sizes and speed | 5.1 to 5.5 | covered; a few table cells blank | 31 |
| (3) histories | 5.4 | picture history covered; averaging the output and the crossing question missing | 11, 30 |
| (4) training settings (all ten sub-items) | 6.1 to 6.9 | covered | 25 |
| (5) held-out episodes against held-out sheets | 7.1 | covered | |
| (5) which offline numbers predict closed-loop success | 7.2 | from the literature only; no plan to check it on our robot | 16, 21 |
| (5) robot test protocol, trials per sheet, success rule | 7.4, 7.5 | covered; working rules thin | 14, 15, 19, 20 |
| (6) DAgger or human-gated corrections | 8.3 | thin on why plain DAgger relabelling does not fit us | 13 |
| (6) what to record, how many episodes | 8.4 | covered; the round size is left unresolved | 24 |
| (7) quick offline experiments with real numbers | throughout, A.2 | several measured results left out | 2, 9, 10, 11 |
| Ranked recommendation, first set-up and backup | 2.1 to 2.3, 2.5 | covered | 21, 31 |
| Evidence for each choice | 2.4 | covered for the main choices; several choices have no row | 25 |
| Experiment plan with times up to the deadline | 9 | thin for the offline items and for building the model mode | 8, 15, 17, 18, 23 |
| Open questions for the professor | 10.1 | covered; two small additions | 27, 28 |
| Short plain summary at the top | 1 | present; one claim needs a qualifier; light is absent | 3, 12 |
| State of the runs on the laptop | Status table, 11.3 | out of date at 23:32 | 2, 4 |

---

## 3. Results that reached the disk after the report's cut-off

The report says it was written "from the results on disk at 22:36"; it was last saved at 23:14. File times checked at
23:32 [measured]:

| File (in runs) | Written | What the report says | What is on disk |
|---|---|---|---|
| `2026-11-09_resnet18_finetune_exact/fold_without_path_02.json` | 22:45 | pending, about 23:30 | step 45 0.631, path 45 0.726, stop at (0.5, one picture) 21 right / 2 early / 20 never of 44 |
| `2026-11-09_resnet18_finetune_exact/fold_without_path_04.json`, `metrics.json` | 23:01 | pending | 0.635, 0.729, 15 / 17 / 4 of 45 |
| `2026-11-09_resnet18_finetune_mirror/fold_without_path_03.json` | 23:21, rewritten | "presumably" sheets 01, 02, 04 | the queue restarted the mirror run at 23:02 on its default sheets 03, 05, 06; fold 03 repeats the earlier numbers exactly (0.6703, path 0.726, 26 / 3 / 14 of 45); only the minutes changed (19.2) |

Sources: `scratchpad/run_end_to_end_queue.sh` line 6 (default folds 03, 05, 06); `scratchpad/logs/e2e_queue_3.log`
lines 10 to 15.

Still pending at 23:32, with no numbers used or guessed here: the chunk re-run on 03, 05 and 06; then
`resnet18_backup_names_plus_exact` and `resnet18_final_candidate_seed_2` on six sheets (expected about 10:00 on 10.11);
then `resnet18_finetune_no_augmentation` on 03, 05 and 06 (about 11:30).

With the earlier exact run now on all six held-out sheets, the cleaning and last-weights recipe can be compared with
the old rules on six sheets. [calculated from the fold files' shares and episode counts; one seed each; the test rows
differ slightly because of the cleaning (for example 19 against 20 episodes on sheet 01), and so do the fold seeds]

| Run | Step 45, mean of six | Path 45, mean of six (worst sheet) | Stop at (0.5, one picture): right / early / never |
|---|---|---|---|
| Earlier exact run (no cleaning, best-validation weights) | 0.678 | 0.781 (0.726) | 90 / 47 / 59 of 208 |
| Final candidate (cleaning, last weights) | 0.678 | 0.786 (0.735) | 101 / 41 / 50 of 209 |

The gain of 5.0 points in right stops is below the report's own 7-point bar for one run against one run (sections 6.1
and 7.3).

---

## 4. Problems, most severe first

### High

**1. The stop has no backup and no laptop experiment other than more data.**

- *Where:* 4.5, 2.5 (robot bullet "failures mostly at the stop"), 8.4, 9, 11.2.
- *Missing:* the report rests the stop on line-end data alone. The sources hold three further options with measured
  numbers:
  - a stop output trained on every picture within 2 cm of the goal (the near-goal window): with tuned thresholds the
    frozen exact probe gave 0.318 right / 0.276 early against 0.221 / 0.414 for the stop-picture target; it needs a high
    threshold (0.7 to 0.9) and is then about as good as the stop picture. Table 4.5 shows only the names version, at 0.5
    (where it looks worst) and at a hindsight threshold; the exact version's tuned result sits in A.2; no text draws a
    conclusion;
  - a combined rule, stop only when the stop probability passes and the predicted step is short: on the earlier exact
    run it raised the best-threshold ceiling from 0.439 / 0.288 to 0.494 / 0.177 (two in a row). Its critic notes that
    the step thresholds were 0.2 to 0.8 cm and that the gain with other-sheet thresholds came from sheet 06 alone, so it
    is a hint. It needs no retraining, because the final candidate already predicts the step length;
  - extra weight on far pictures where a first model wrongly fired (mined hard negatives), started and not finished.
- *Sources:* `scratchpad/design/targets/section.md` 2.6 (near-goal window, length cue, variant E) and section 4 (last
  backup bullet); `targets/critic_evidence/note.md` point 15; `judge_3/note.md` section 5 (a stop variant run end to end);
  `models/critic_exam_risk/note.md` ("Stop gate"); the report's own A.2 (tuned stop numbers).
- *Fix:* add a short paragraph "stop options not yet run end to end" to 4.5 with these numbers, each marked as a
  frozen-probe or three-sheet hint. Add the combined rule and the near-goal output at a high threshold as robot A/B
  candidates under "failures mostly at the stop" in 2.5. Add one optional laptop run (the near-goal window on the final
  recipe, six sheets, about 2.0 to 4.9 h) to section 9 with a keep rule fixed in advance, and list hard-negative mining
  under "not run" in 11.2.

**2. The earlier exact run has finished on all six sheets, and the report still calls it pending.**

- *Where:* Status table, 2.1 (third "why" bullet), 2.4 (row "the audit cleaning"), 11.3 (first row), A.4.
- *Missing:* folds 02 and 04 were written at 22:45 and 23:01, before the report's last save at 23:14. They complete the
  only six-sheet comparison of the cleaning and last-weights recipe with the old rules (table in part 3 above): equal
  direction, and 101 against 90 right stops of about 209, a 5-point gain under the report's own 7-point bar. The report
  argues the cleaning's stop gain from three sheets only (47 of 99 against 42 of 100).
- *Sources:* runs `2026-11-09_resnet18_finetune_exact/fold_without_path_01.json`, `_02.json`, `_04.json`, `metrics.json`;
  `scratchpad/logs/e2e_queue_3.log` lines 12 to 14.
- *Fix:* add the 02 and 04 rows to A.4; put the six-sheet comparison into 2.1 and 2.4 (one seed each, test rows and fold
  seeds slightly different); say plainly that the stop gain is inside the noise bar; move the run from 11.3 to the
  measured results and update the Status table.

**3. The lighting split stops at the data: the model's results by light are missing, and light is absent from the
risk lists.**

- *Where:* 3.7, 1 (summary), 2.1 ("main risks"), 4.5 ("fragile under changed pictures"), 7.4 (light block).
- *Missing:*
  - the final candidate on its six held-out sheets, by light: natural light path 45 0.702 of 2,938 pictures against
    0.785 of 15,026 under the ceiling light; always forward 0.605 against 0.585; so the margin over always forward is
    about 9.7 points in natural light against 20.0 under the ceiling light [measured; margins calculated; natural light
    occurs only on sheets 02 and 04, so light and sheet are tied];
  - in the darkest third of pictures the earlier exact model's stop probability passed 0.3 far from the goal on 0.041
    of 2,508 pictures against 0.018 of 2,573 in the brightest third, and the final candidate's direction was 0.596
    against 0.710 (confounded with sheet and place);
  - half brightness, which is darker than any training picture (natural light reaches down to 0.674 of the median),
    raised sheet 02's early stops from 2 to 10 of 45.
- *Sources:* `scratchpad/report_parts/results_so_far.md` lines 172 to 198 and `errors_resnet18_final_candidate.csv`
  (rows "lighting"); `scratchpad/design/targets/critic_exam_risk/note.md` point 9; `judge_2/note.md` section 3.1 (last
  bullet).
- *Fix:* add a paragraph "how the model does in each light" to 3.7 with these numbers and the confounding; name light as
  a risk in the summary and in 2.1 (the exam light is question 19 on `requests_to_professor.md`, still open); link it to
  recording under varied light in step A of 8.4.

### Medium

**4. The queue is re-running finished work, and 11.3 guesses the opposite.**

- *Where:* 11.3 (second row), Status table, 9 (first week), A.4 (minutes column).
- *Missing:* the report presumes the mirror and chunk runs are on sheets 01, 02 and 04 and asks the reader to "check
  the queue". The disk answers it: they are re-runs of 03, 05 and 06 (part 3 above), they reproduce the numbers already
  used, and they take about 3.5 laptop hours (until about 02:30) before the backup and the second seed that the 10.11
  decision waits for. The re-run also rewrites the cited fold files (minutes now differ from A.4) and overwrote the first
  mirror run's log at 23:02.
- *Fix:* correct the 11.3 row; state the cost in hours; leave to Emranul whether to take the two runs out of the queue
  (read the terminal first; stopping a job is his decision); note in A.4 that the re-run reproduced the mirror's sheet-03
  numbers but rewrote its minutes.

**5. "Test it on the simulated robot" is the only check before the robot, and it cannot test the model.**

- *Where:* 1 (next step 2), 4.6, 5.5, 9 (first week).
- *Missing:*
  - the simulated robot draws a white page, a black line and a red disc (`02_code/robot_link.py`, `SimulatedRobotLink`,
    lines 428 to 444), so it checks signs, clamps and the software path, not the model's skill;
  - no check that the model mode reproduces the saved predictions from logged raw pictures. That is the cheap test for a
    decode or resize mismatch between `training_dataset.py` and the run-time decider (5.5 says this cost is not even
    timed);
  - no check that each decision uses a fresh picture: the audit lists camera latency as a robot test (audit 11.3 item
    23) and 863 rows whose fresh-picture wait exceeded 0.2 s, where stale pictures cluster (audit 11.1 D5).
- *Sources:* `scratchpad/design/improvement/critic_exam_risk/note.md` point 3 and its plan step 2;
  `scratchpad/design/models/section.md` section 5; audit lines 1173 and 1273.
- *Fix:* add a replay check (run the decider on the logged pictures of one held-out sheet; the angles and stop
  probabilities must equal the saved ones), a fresh-picture check, and one plain sentence on what the simulated robot can
  and cannot show, to 4.6 and to week 1 of section 9.

**6. The 11 classes of the request are never addressed.**

- *Where:* 4.1, 3.2, glossary ("Move names").
- *Missing:* the data hold 12 labels: 8 directions, stop, small_motion, turn_left and turn_right. The 2,278 small_motion
  and turn rows (12.3 %) lie outside the 11-class set of the 6.11 memory; the step-over classes never occur, because the
  replay's labeller cannot produce them; and the diagonals sit at 60 degrees (back diagonals 135) since 4.11, not 45.
- *Sources:* audit notes_context-2 (line 911) and table 9.1 (line 970).
- *Fix:* add a short paragraph to 4.1 that sets the 11-class plan against the 12 labels and says what each label becomes:
  in the first set-up, the exact step, with small_motion and the turns as twist and the stop as its own output; in the
  backup, the 8 names plus the stop output.

**7. The at-goal label is never discussed.**

- *Where:* 4.5.
- *Missing:* the recorded is_at_goal label (808 rows, 4.36 %, about 3.85 times the stop rows; added on 7.11 per the
  memory) and why it is not the stop target: in 30 of 210 episodes the tip enters the 2-cm zone and leaves it again (124
  at-goal rows before that, 99 of them inside correction replays); 111 of the 808 lie more than 10 rows before the stop;
  and the demonstrators keep moving inside the zone (523 non-stop rows within 2 cm of the goal after the cleaning).
- *Sources:* audit class_balance-11 (line 76), notes_context-4 (line 915), 11.2 item 13 (line 1210);
  `scratchpad/design/targets/section.md` 2.1 (last bullet).
- *Fix:* add a paragraph "stop picture or at-goal" to 4.5, joined to the near-goal window of problem 1.

**8. Building the model mode is the critical path, and it has no estimate, no owner and no slip rule.**

- *Where:* 9 (first week: "not estimated"), 4.6, 1 (next step 2).
- *Missing:* 4.6 lists nine missing pieces plus the hand-over to the exact-replay decider after "reguide"; session 1 in
  the week of 16.11 depends on all of them. The plan says neither how long this takes, nor who builds it (Emranul, Marshal,
  or which workstream session in `HOW_WE_WORK.md`), nor what moves if it is late.
- *Sources:* `scratchpad/audit/code_actions/note.md` section (f), as cited in 4.6;
  `scratchpad/design/improvement/critic_evidence/note.md` point 12.
- *Fix:* give a rough estimate marked as a guess, name the owner, and add one line to section 9: if the model mode is not
  ready by about 15.11, session 1 moves and the dates after it shift by the same amount.

**9. The stress test is reported by one number.**

- *Where:* 4.5 ("fragile under changed pictures"), 6.2 (shift row), 11.1.
- *Missing:* the test covered nine changes on the fold models of sheets 01 and 02. Direction stayed within 2.0 points for
  every change (0.766 to 0.791 on 01; 0.612 to 0.640 on 02). Sheet 02 stops at (0.5, one picture), against clean 24 / 2 /
  18: half brightness 21 / 10 / 11; 1.5 times brightness 24 / 0 / 19; 8-px shift 25 / 4 / 14 and 25 / 4 / 13; 3-degree
  roll 21 / 2 / 19; 1-px blur 21 / 2 / 20; colour casts 25 / 2 / 16 and 22 / 2 / 20. Sheet 01: 2 to 4 early of 20 in every
  condition. Also missing: that half brightness lies below the training range, and that no condition was scored under
  the proposed rule (0.5, two in a row).
- *Source:* `scratchpad/design/training_settings/critic_exam_risk/stress_test_final_candidate_output.txt`.
- *Fix:* add a small table to 4.5; cite the 8-px result in the shift row of 6.2 (it bears on the camera move of 8.11 and
  on the proposed 8-px test); keep the six-fold stress test in week 1 and score it under the fixed rule.

**10. Section 8.1 analyses the old model, while a six-sheet analysis of the final candidate exists.**

- *Where:* 8.1, 3.7, 3.8.
- *Missing:* the final candidate's error analysis by situation, pooled over its six held-out sheets (path 45, always
  forward in brackets): bend 0.735 (0.524), straight 0.860 (0.740), near the goal 0.690 (0.566), start 0.900 (0.780);
  camera view A 0.769 (0.629) against the later view B 0.777 (0.518), so the 8.11 camera move shows no loss here;
  forearm far turned 0.766, turned 0.754, straight 0.798; score 1 0.735, score 2 0.723, score 3 0.804. The puck-off-line,
  bend-radius and crossing splits exist only for the earlier model. The analysis counts slightly more pictures than the
  fold files (633 against 578 on sheet 01), which needs one sentence if quoted.
- *Sources:* `scratchpad/report_parts/results_so_far.md` lines 172 to 198; `report_parts/errors_resnet18_final_candidate.csv`;
  `scratchpad/design/improvement/situation_errors.csv` (earlier model).
- *Fix:* put the six-sheet table beside the old one in 8.1; say that the off-line and radius splits were not redone for
  the final candidate (`situation_errors.py` takes minutes); use the camera-view and score rows in 3.7 and 3.8.

**11. Averaging the output over the last pictures, and the offline dithering check, are left out.**

- *Where:* 5.4, 3.3, 10.1 (question 9).
- *Missing:*
  - measured: averaging the final candidate's direction over 1, 2, 3 and 5 pictures gives path 45 0.772, 0.775, 0.767
    and 0.731 (14,294 pictures pooled); for the frozen ResNet-18 probe 0.617, 0.631, 0.635 and 0.620;
  - measured: consecutive predictions flip the twist sign on 0.5 to 1.2 % of picture pairs against 3.8 to 16.7 % in the
    demonstrations, and reverse the direction on 0 to 0.7 % against 4.8 to 10.0 %, so there is no dithering to smooth
    offline;
  - literature: averaging the outputs of several pictures did poorly and oscillated on a real robot (Black et al. 2025,
    supported), and lost under noise (BID, 18.4 against 29.5);
  - 3.3 states "smoothing would lag behind real turns" under a measured marker; it is a judgement.
- *Sources:* `scratchpad/report_parts/results_so_far.md` lines 199 to 203; `scratchpad/design/main_session_facts.md`
  lines 89 to 90; `scratchpad/design/targets/critic_exam_risk/note.md` ("No smoothing needed"); `targets/section.md`
  2.7; `improvement/critic_exam_risk/note.md` point 11.
- *Fix:* add a paragraph to 5.4 with these numbers and the conclusion (no averaging); mark the 3.3 sentence as
  judgement; then question 9's averaging part can be dropped or kept as low priority.

**12. The plain summary needs two qualifiers.**

- *Where:* 1 (title, second and third bullets).
- *Missing:*
  - "the names backed up far better after a drift (0.66 against 0.39)" compares an earlier names-only model with the
    earlier exact model (old rules, sheets 03, 05, 06). The first set-up scored 0.426 on the same moves; the backup's own
    number is pending. And many backward moves are ordinary pushing seen through a turned camera (glossary), not
    recovery after a drift;
  - "within 45 degrees of where the line goes next" describes path 45 loosely: it compares with where the
    demonstrator's tip was three steps later;
  - the title names Emranul only, and "second seed" is used before any explanation.
- *Sources:* report 2.2; `scratchpad/design/targets/section.md` 2.2; `judge_2/note.md` rank 2.
- *Fix:* name the models behind 0.66 and 0.39, add the first set-up's 0.43, drop "after a drift", reword the path
  measure, title it "Summary for Emranul and Marshal", and explain "seed" in a few words.

**13. Corrections: why human-gated and not plain DAgger is not explained, and three procedure points are missing.**

- *Where:* 8.3.
- *Missing:*
  - why a person cannot label the model's own pictures afterwards, as in plain DAgger: the exact step with its twist
    cannot be typed; key-override corrections drop the twist and move the view about 56 px from the training views; and
    labels given afterwards were unreliable in Laskey et al. 2017 (correlations 0.60 and 0.22; about 40 against 60 %
    success; the critic notes these were mouse labels on half-speed video);
  - end the episode when the puck has left the sheet or the view, rather than record a long recovery; note the failure
    type in the review form;
  - use the correction pictures also as an offline validation set with recoveries (Codevilla et al. 2018, in its
    corrected form).
- *Sources:* `scratchpad/design/improvement/section.md` 3.1 (labels given afterwards) and 3.2 (steps 4 and 5, "why not
  the other ways to label", use as a validation set); `improvement/critic_evidence/note.md` point 11.
- *Fix:* add one paragraph and two procedure steps to 8.3.

**14. The robot protocol misses its working rules.**

- *Where:* 7.4 ("success rule", "recorded in every trial", "light and drift").
- *Missing:*
  - what counts as a stall or dithering (the validation draft: under 1 cm of progress over 10 steps);
  - when the operator may end a run early (the puck has left the paper, the arm has not moved for 5 steps, or safety);
  - what "left the line" means (bare paper visible between the puck and the line's course; on a dashed line, the
    course through the gaps; judged live, checked on the overhead video) and how the photo is scaled (the sheet's
    29.7-cm edge);
  - one uncounted warm-up run per session;
  - the camera check has no numeric tolerance and no action when it fails (stop the session and follow the camera-mount
    checklist; never touch the camera).
- *Sources:* `scratchpad/design/validation/section.md` 2.3; `validation/critic_exam_risk/note.md` point 7.
- *Fix:* add these five rules to 7.4.

**15. There is no plan for fewer robot hours, and robot availability is not asked.**

- *Where:* 7.4 (phases), 8.4 (totals), 9, 10.2.
- *Missing:* the plan needs about 31 to 46 robot hours (5 to 8 a week) but gives no reduced version. The validation
  draft has one: phase 1 with 12 blocks (it still picks the better of a 90 and a 70 % model 89 % of the time) and phase 2
  with the six known sheets x 3 and the ten sealed tracks x 2 (38 trials), reported as exploratory. The hours the team
  can book per week until 20.12.26 are unknown.
- *Sources:* `scratchpad/design/validation/section.md` 2.4 (first bullet); `improvement/section.md` section 7 (last
  bullet).
- *Fix:* add the reduced plan to 7.4 and section 9; add "how many robot hours a week can we book until 20.12.26?" to
  10.2.

**16. Which offline numbers predict finishing is answered from the literature only.**

- *Where:* 7.2, 7.4.
- *Missing:* a plan to check it on our robot: set each model's per-sheet offline numbers (path 45, stop counts, the rows
  after a correction) beside its phase 1a and phase 2 results in a calibration table, and say what pattern would change
  the order of measures in 7.2.
- *Sources:* `scratchpad/design/validation/section.md` 2.1 point 5, phase 2 analysis (last bullet), section 3 (one
  bullet on how well offline numbers track this robot).
- *Fix:* add the calibration table to the plan of 7.4 and name it in 7.2.

**17. The offline experiment plan is not itemised.**

- *Where:* 9 (last paragraph), 11.2.
- *Missing:* the laptop experiments are spread over the report (copy test 5.4, 8-px shift 6.2, sheet-balanced weights
  6.3, gentler fine-tuning 6.6, 16 epochs 6.7, leave-one-block-out 7.1, the extra 05 and 06 fold 10.3, decider timing 5.5,
  near-end-start check 8.4, chunk rerun 2.5), and section 9 gives hours for only some. The training-settings draft's
  ordered table with hours per item is reduced to one total ("31 to 73 hours").
- *Sources:* `scratchpad/design/training_settings/section.md` ("Experiments still needed"); `models/section.md` section
  8; `training_settings/critic_evidence/note.md` point 6 (the corrected total).
- *Fix:* one table in section 9: experiment, question it answers, laptop hours, priority (do, optional, drop) and the
  decision rule fixed in advance.

### Low

**18. The exam lay-out rule and the track-drawing rule are not scheduled.** 7.4 depends on a written random rule for the
sealed tracks (the validation draft suggests a small script) and on a written exam lay-out rule, but section 9 schedules
neither and names no author. Emranul expects 4 to 5 sheets in very different orientations (`requests_to_professor.md`
question 18, open). *Sources:* `scratchpad/design/validation/section.md` 2.3; `improvement/critic_exam_risk/note.md`
point 8 and plan step 1. *Fix:* add both to week 1 or 2 with an owner.

**19. Phase 1a leaves out held-out sheet 02.** Its fold model never stopped in 18 of 45 runs (one picture) and 24 of 45
(two in a row), and its line ends lie 6.2 and 6.4 cm from the paper edge: a far-from-edge stop test that needs no
printing (about 2.4 minutes a trial). *Sources:* `judge_1/note.md` section 2; `judge_2/note.md` section 3.1;
`training_settings/critic_exam_risk/note.md` recommendation 5. *Fix:* add `without_path_02` on sheet 02, about 5 runs.

**20. The backup's robot comparison has no starts that need backing up.** The backup exists for backing up, yet 2.5 and
7.4 compare it from standard starts; offset starts sit in a separate block "reported apart". Judge 1 and the targets
exam-risk critic ask for starts with the puck laid beside the block. *Fix:* include such starts in the backup comparison.

**21. The switch rule lacks its reference numbers, and there is no offline gate before the robot.** 2.5 asks for
"clearly better first rows after a hand correction" without the first set-up's six-sheet values (415 of 1,124 step 45,
573 of 1,062 path 45; `judge_1/note.md`, "The backup in full") or a threshold. Judge 2 proposes a gate every model must
pass before robot time: path 45 above always forward on every fold, twist sign 0.6 or more on every fold, early stops
under about 15 % pooled, all met by the final candidate (`judge_2/note.md` section 4, point 1). *Fix:* add both to 2.5
and 7.2.

**22. The look-by-eye list leaves out three audit items.** 3.9 omits audit D1 (27 first rows after a pause over 10 s that
the waypoint marker keeps), D2 (hands that persist into the next picture, frame_index 6459 and 12603) and D5 (863 rows
with a fresh-picture wait over 0.2 s). *Sources:* audit 11.1 D1, D2, D5 (lines 1169 to 1173);
`training_settings/critic_evidence/note.md` point 7. *Fix:* add them to the 3.9 row and to 11.2.

**23. A code fix due before more recording is not in the plan.** `record_episodes.py` line 1120 still appends the
duplicated final point, so every new demonstration, near-end start and correction episode will carry it. *Source:* audit
11.1 A12 (line 1148). *Fix:* add it to section 9 before step A (low, because the cleaning relabels these rows).

**24. The correction round size is left unresolved.** 8.3 notes that 50 episodes give about 9 % of the data against 20 to
33 % in the studies, but step C keeps 50 without saying whether to enlarge the round, rely on version 2, or expect a
smaller effect; with version 1 most added rows are ordinary pushing. The reason for 20 demonstrations per new track
(half the episodes already gave within 0.9 points) is also missing. *Sources:* `improvement/critic_evidence/note.md`
point 10; `improvement/critic_exam_risk/note.md` point 7; `improvement/section.md` 4.2 (last paragraph).

**25. Some choices of the first set-up have no evidence row.** 2.4 has no row for the 160 x 120 input (6.5: no comparison
on our data), the head (6.8), the Huber loss (no comparison anywhere), the stop weight 10 (6.3: untested), the
photometric set (6.2), or the twist clamp of 10 degrees per step, which lies below the demonstrated 99th percentile of
11.6 degrees per step (4.3) and is justified nowhere. *Fix:* one row each, marked measured, literature, judgement or
untested.

**26. Checked literature that bears on the choices is left out.** Black et al. 2025 on averaging outputs (problem 11);
DMD (wrist camera, predicted direction, fixed 1-cm moves, off-track data; interactive_correction check claim 36,
corrected) for the executor and the recovery starts; Codevilla et al. 2018 recovery data (88 and 64 % against 56 and 22 %
without; augmentation check claim 22) for 8.4 point 2; Xie et al. 2023 (a new camera position was the hardest factor,
a new table texture next; augmentation check claims 16 and 17) for the camera check; recovery pictures in the validation set
(Codevilla 2018, corrected form) and Allshire et al. 2026 (validation action error tracked success, the loss did not) for
7.2. *Sources:* `targets/section.md` 2.7; `targets/critic_exam_risk/note.md` point 10; `judge_1/note.md` section 3;
`improvement/critic_exam_risk/note.md` point 5; `validation/section.md` 1.5.

**27. Two professor questions are missing.** May the executor know an unseen track's line length (to set the budget at
about 4 steps per cm), or must it use the 250-step cap (`targets/critic_exam_risk/note.md` point 6)? May a mask computed
from the joints and taught corners run at the exam (audit 11.3 question 17; question 8 of 10.1 covers only a fixed
mask)?

**28. "Question 7" in section 9 points to the wrong list.** The row "21.12 to 8.1.27" means question 7 of
`requests_to_professor.md` (the documentation, still open), but the report's own question 7 is about training on all six
sheets. *Fix:* "(question 7 on `requests_to_professor.md`)".

**29. Class balance per sheet is shown only before the cleaning.** Training uses the cleaned rows, where small_motion
loses the 449 null rows and the 51 duplicated final points (2,265 to 1,765); `training_settings/cleaned_data_counts_output.txt`
has only the overall counts after the cleaning. *Fix:* a per-sheet recount after the cleaning (minutes), or one sentence saying the shares change little except
small_motion.

**30. One picture at the crossings and the dashed gaps is not discussed.** The literature takeaways and the models draft
note that a single picture may not be enough there (which branch at a crossing; no line in a gap), and that only the
robot can tell. 5.4 keeps one picture without naming this risk or saying that phase 1a on sheets 05 and 06 tests it.
*Sources:* `report_parts/literature_takeaways.md` point 7; `models/section.md` section 7, item 7.

**31. Small table gaps.** 2.3: the "not now" row has an empty "why this place" cell, and fine-tuned MobileNetV3 and
EfficientNet-B0 and the line-and-puck small network are not ranked. 5.5: multiply-adds are blank for four backbones and
for ACT and diffusion.

---

## 5. What is complete and needs no change

- The data audit's bookkeeping, label noise, near-zero moves, hands, blur, near-copies, score analysis and the cleaning
  table (3.1 to 3.6, 3.8 to 3.11), all marked and matching the audit.
- The case for holding out whole sheets (3.6, 7.1) and the noise bars (6.1, 7.3), with the seeds-needed table (6.9).
- The augmentation table with what keeps or breaks the label (6.2), class weights (6.3), score weighting (6.4),
  optimiser, epochs and regularisation (6.6 to 6.8).
- The robot statistics (7.5), the four track roles and the pre-registered plan (7.4), the session rules from the
  project's working rules.
- The professor questions and the facts only Emranul and Marshal can confirm (10.1, 10.2), with the already-answered
  questions correctly cited (question 10 on pretrained models, question 8 on grading, question 9a on the lay-out).
- The pending runs in 11.3 are marked as pending with their expected times, and no pending number is guessed.
