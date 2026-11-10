# Critique of "Improving after the first model" (key improvement), lens: exam risk

Written 9.11.26 by a critic sub-agent for Team M (Emranul Hoque, Marshal Sebastine). Read only: nothing under
D:/AI_Project was changed and nothing was run on the robot. Markers: **measured** (file and field), **literature**
(a claim its checker marked supported or corrected), **calculated**, **judgement**, **guess**.

Question asked: does following the section maximise the chance of finishing UNSEEN tracks at the exam, then the time?

## Verdict: stands with changes

The core holds: more distinct printed tracks (data-amount fit and Lin et al. 2025, checked), human-gated corrections
replayed with exactly the demonstrations' labels, retraining from scratch (CR-DAgger, checked), sealed unseen-track
tests. But the plan ranks the risks by offline direction accuracy, while the exam is lost mostly through things the
offline numbers cannot show or under-weight: early stops, the executor at the arm's limits, a moved camera, a new
table or light, and an untested exam procedure. It also spends its largest robot block before any closed-loop run.

## Problems, most severe first

1. **High. The stop is the main exam risk, and its tuned rule optimises the wrong thing.** Measured, early stops (more
   than 3 cm before the demonstration's stop) on held-out sheets: end-to-end exact model 33 of 100 episodes with the
   tuned two-in-a-row rule (03: 8/45, 05: 19/30, 06: 6/25) against 25 of 100 at 0.5 (7/45, 17/30, 1/25); final
   candidate 21 of 110 against 15 of 110 (01: 3/20 against 4/20, 02: 8/45 against 2/45, 03: 10/45 against 9/45)
   [04_runs/2026-11-09_resnet18_finetune_exact/metrics.json, folds; 2026-11-09_resnet18_final_candidate/
   fold_without_path_01 to _03.json; fields episodes_stopped_early, tuned_two_episodes_stopped_early]. The tuned rule
   (thresholds 0.10 to 0.40, tuned_two_stop_threshold) turned never-stops into early stops (never-stops fell from 31
   to 19 of 100 and from 20 to 15 of 110) while stopped right changed little (42 against 47 of 100, 69 against 66 of
   110). Judgement: an early stop ends a run for certain; a never-stop may still count as finished if pushing past
   the end counts. Nobody has asked: question 8 in 05_meetings/requests_to_professor.md records only that finishing
   ranks first and time second, and the audit's question 20 (how is finished judged) is not on that list. In correction rounds a model "stop" must wait for Enter, as the replay's
   stop does now (code_actions note, section (a)), or early stops are never corrected.
   Fix: put the stop first; ask now how a finish is judged; until answered, choose the threshold on held-out sheets
   to keep early stops low (judgement: at most 1 in 10) and let the step budget end never-stopping runs; log the stop
   distance in every robot run.

2. **High. "Our check does not confirm" the paper-edge cue understates it for the model the team will use.** Measured,
   end-to-end exact model: line ends 5 cm or more from a paper edge were stopped right in 13 of 45 episodes (tuned,
   two in a row) against 34 of 55 for nearer ends, 11 of 45 against 31 of 55 at 0.5; the rank correlation of edge
   distance with stopped right was -0.49 for both rules, not reported in the section [stop_by_end.out.txt,
   end_to_end_exact]. Confounded with the loop sheets, so undecided, not unconfirmed. The audit names a line end more
   than 8 cm from an edge as the strongest unseen-track risk (view_and_tracks-16).
   Fix: put line ends 8 to 12 cm from the edge into at least two training and two sealed tracks; in the first robot
   session lay a provided sheet on a larger white sheet (edge cue gone) and see whether the stop changes. No printing
   needed for that test.

3. **High. The largest robot block (A, 10.3 to 13.9 h) comes before any closed-loop run.** Compounding drift, stalls,
   early stops in closed loop, refusal livelock, guards refusing new lay-outs and a run-time preprocessing mismatch
   are invisible offline, and offline numbers rarely pick between good models of one family (offline_vs_closed_loop
   check, "Between runs versus within a run"). The simulated robot draws a white page, a black line and a red disc
   (02_code/robot_link.py, SimulatedRobotLink, lines 428 to 444): it tests the software, not the model. Held-out-sheet
   checkpoints already exist (final_candidate/checkpoints/without_path_01 to _03.pt; finetune_exact/checkpoints/
   without_path_03, _05, _06.pt), so an unseen-sheet closed-loop test needs neither printing nor question 15. The
   audit already asks for an early closed-loop check on sheet 01 (11.3, point 22).
   Fix: a step 0 of about 2 h (guess) as soon as the model mode exists (see the plan below), and split the rest of A's
   hours by what it shows.

4. **High. The executor is not frozen before the corrections, and the arm's limits and finishing time are absent.**
   Measured: joint 4 reached -119.35 and +118.92 degrees against the arm model's limit of 119.75; 211 of 18,522 rows
   lie within 5 degrees of it; joint 4 spans a median 155 and up to 224 of the 239.5 degrees available per episode
   (code_actions note, sections (d) and (f); 239.5 calculated); 27 of 45 sheet-04 episodes pass 110 degrees (audit
   11.2, point 29). The models' twist errs by 2.2 to 3.6 degrees per step on average, sign right in 0.67 to 0.85 of
   rows (twist_error_mean_deg, twist_sign_accuracy, six fold files). Joint 5 drifts 10 degrees or more in 89 of 210
   episodes and a held wrist moves the picture by over 20 px in 23.3 percent of rows (audit 11.2, point 10), so the
   wrist rule decides which views the model meets. No demonstration shows what to do at the forearm limit. Faster
   than demonstrated execution lowered success (SAIL, action_chunking check claim 48: Square 0.83 to 0.55), and
   stop-and-go control removed a latency effect (OpenVLA App. D.4, offline_vs_closed_loop check). Judgement:
   corrections describe the executor they were collected with; changing the wrist rule, clamps, speed cap or steps
   per picture afterwards makes C and D partly stale.
   Fix: settle and freeze the exam executor in B (wrist rule, joint-4 clamp and its fallback, refusal counter, step
   budget, speed cap, one step per picture); log joint 4 every step; add one training and one sealed track with more
   one-way turning than sheet 04; change speed only after a paired known-sheet test scored finish first, then time
   (evaluation_statistics check, section 3, point 3).

5. **Medium. Camera, table and light at the exam are not guarded.** Measured: the camera has already moved:
   camera_shift_px medians about 5 px on sheets 01 to 03 against 15.0 and 14.5 px on 05 and 06, max 28.3 (audit
   view_and_tracks-1), against the team's own 4-px rule (claude_memory/project-forearm-camera-mount.md); 4 of 239
   attempts were discarded for a camera move (episode_timing.out.txt, statuses). The tool's camera check makes the
   first picture at a new start position its own reference (record_episodes.py lines 1147 to 1155), so on every new
   printed track and at the exam it cannot see a move. Natural light only on sheets 02 and 04 (3,007 pictures).
   Literature: new camera positions were the hardest factor (45.8 percent) and new table textures 52.8 percent even
   for RT-1 with 115,000 demonstrations (Xie et al. 2023, augmentation check claims 16 and 17); after a lab
   reorganisation one policy went from 13/20 to 0/8 (Kress-Gazit et al. 2024, evaluation_statistics check claim 5).
   Fix: one fixed camera-check pose and reference picture at the start of every sitting and on exam day,
   independent of the track; record A and C sittings across lights and log them; a few known-sheet trials under a
   deliberately different light and background in step 0 or B.

6. **Medium. The sealed tracks are used for choosing, and the backup has no unseen-track test.** D's 6.4 to 11.0 h
   is two models times B's tests (1.9 + 1.3 = 3.2 h, 3.3 + 2.2 = 5.5 h; robot_hours.out.txt, calculated), so the same
   10 sealed tracks decide which model goes to the exam and whether E runs, against the section's "never for choosing".
   With 24 trials each, the better of 80 and 90 percent is chosen only 0.83 of the time (evaluation_statistics check,
   section 3, point 2); a separate untouched final test and a written rule for drawing tracks are advised (same check,
   section 4: Vincent et al. 2024; Cawley and Talbot 2010, abstract). If printing is refused, the plan has no
   unseen-track robot test at all.
   Fix: two sealed sets drawn by a written rule that includes the audit's unseen features (view_and_tracks-16): a
   development set for B and D, a final set used once; score each run also by the furthest point reached (partial
   credit, Lin et al. 2024, evaluation_statistics check, section 4); in the no-printing backup, run the held-out-sheet
   checkpoints on their own held-out sheets.

7. **Medium. The corrections' payoff cannot be measured as planned, and the safer recovery starts sit in the backup.**
   Round 1 adds about 35 corrections and 1,624 rows (9 percent of 18,053) at a guessed 30 percent base success
   (robot_hours.out.txt); with version 1 most of those rows are ordinary pushing from the failure point to the end
   (judgement). Rounds here are 5 to 8 times smaller than in the cited studies relative to the initial data, and no
   study showed carry-over to unseen layouts (interactive_correction check, section 2.2, points 1 and 5); the
   section's own ablation found the operators' corrections act like ordinary data (recovery_ablation.out.txt). The
   question whether corrections count as our own demonstrations is not on requests_to_professor.md (audit 11.3,
   question 18). Deliberate recovery starts (puck 1 to 2 cm beside the line, guided back) need no model mode and no
   permission, and recovery data is what the strongest evidence supports (Codevilla et al. 2018, augmentation check
   claim 22: 56 and 22 percent without against 88 and 64 with; DMD, interactive_correction check claim 36, corrected).
   Fix: record about 30 recovery starts in A (2.8 to 3.6 h at the measured 5.6 to 7.2 min cycles of offset starts,
   timing_by_start_kind.out.txt; calculated); add the corrections question to the list now; run C only if step 0 and
   B show that most failures are drift off the line or wrong turns.

8. **Medium. No exam-procedure rehearsal and no freeze date.** The professor allowed the team to choose start and
   orientation (requests_to_professor.md, 9a), but Emranul expects 4 to 5 sheets in very different orientations, and
   an unseen sheet whose line runs more than about 24 cm from its start does not fit the planned start mark
   (06_deliverables/research_notes/Recording tool/2026-11-06_exam_start_opinion.md). Fix: lay every new and sealed
   track by the written exam lay-out rule; freeze the exam model and executor by about 13.12.26 (judgement) and run one
   timed dress rehearsal with the exam procedure on the final sealed set.

9. **Low. The correction share is to be chosen offline, partly on the correction pictures.** Testing on corrections
   favours heavier correction weights (judgement), and offline numbers rarely pick the winner (problem 3). Plain
   aggregation helped in Kelly et al., BC-Z and ThriftyDAgger (interactive_correction check, section 2.1, point 6).
   Fix: fix one setting in advance (plain mixing) and spend the robot trials on model against model.

10. **Low. More near-end starts may not cure early stops.** Early stops happen mid-track where no ink is ahead
    (1,588 of 12,241 pictures more than 10 cm before the stop; clustered on sheets 06, 05 and 04 at a median step
    fraction of 0.46; audit view_and_tracks-2). Near-end starts add stop pictures and near-end negatives, not those
    look-alikes (judgement). Fix: before recording 40 more, train once without the 42 existing near-end-start episodes
    (timing_by_start_kind.out.txt) and compare early stops on held-out sheets (laptop only).

11. **Low. Oscillation and stalls are not measured.** Averaging the direction over 2 to 3 pictures helped offline
    (path within 45 0.617 to 0.631 and 0.635, main_session_facts.md), but under noise temporal ensembling lost in closed
    loop (BID Table 4, action_chunking check: 18.4 against 29.5) and oscillation tripped a protective stop (RTC, claim
    45); the human-gated policy was the jerkiest on a real car (Spencer et al. 2020, interactive_correction check claim
    34). Fix: count stalls, back-and-forth moves and steps per finished track in every robot test; decide smoothing
    only by a paired robot test.

## What I would do instead (robot steps only after the team decides, each with its own checklist)

1. Now, no robot: add three questions to requests_to_professor.md (corrections as our own demonstrations; how a
   finish is judged, including pushing past the end, the clock and attempts; exam room, table and light) and chase
   question 15. Set the stop rule against early stops. Run the near-end-start ablation. Write the track-drawing rule
   and the exam lay-out rule; print two sealed sets if allowed.
2. Step 0 (about 2 h, guess), once the model mode passes the simulated robot and reproduces the stored predictions
   from logged pictures: the final candidate on sheets 01 and 03; held-out-sheet checkpoints on their own held-out
   sheets; one provided sheet on a white underlay. Log finish, stop distance, furthest point, stalls, refusals and
   joint 4 every step.
3. A, re-aimed: new tracks with far-from-edge line ends, a long one-way turn, close parallels, a near-90-degree
   crossing and another dash rhythm, laid by the exam rule; near-end and recovery starts on every track; about 30
   recovery starts on the known sheets; varied light; the fixed camera check every sitting.
4. B as planned plus the held-out-sheet checkpoints; freeze the executor here.
5. C only for drift-type failures, with the model's stop waiting for Enter; D retrains from scratch with plain mixing
   fixed in advance; paired, interleaved test scored finish first, then time.
6. By about 13.12.26: freeze the model; one pre-registered final test on the untouched sealed set and a timed dress
   rehearsal; E only if D showed a clear gain before the freeze.

Extra robot time about 7 to 9 h (step 0, recovery starts, final test and rehearsal; calculated from measured cycle
times with guessed trial counts): 31 to 46 h in all, 33 to 53 with E, about 5 to 8 h a week to 20.12.26.

## Sources read

Section: S/design/improvement/section.md and its printouts (robot_hours, stop_by_end, situation_errors,
situation_errors_by_sheet, episode_timing, timing_by_start_kind, data_amount_regression .out.txt; robot_hours.py).
Runs: R/comparison.md, R/2026-11-09_resnet18_finetune_exact/metrics.json, R/2026-11-09_resnet18_final_candidate/
fold_without_path_01 to _03.json, checkpoint listings. Audit: S/audit/AUDIT_SYNTHESIS.md sections 10 and 11,
S/audit/code_actions/note.md. Literature checks: interactive_correction, evaluation_statistics, augmentation,
offline_vs_closed_loop, action_chunking, diffusion_and_generative (4.2), history_and_causal_confusion. Project:
02_code/record_episodes.py (camera check), 02_code/robot_link.py (simulated robot), 05_meetings/requests_to_professor.md,
the exam start opinion of 6.11.26, claude_memory/project-forearm-camera-mount.md, S/design/main_session_facts.md.
S = C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad;
R = D:/AI_Project/.claude/worktrees/training-analysis/04_runs.
