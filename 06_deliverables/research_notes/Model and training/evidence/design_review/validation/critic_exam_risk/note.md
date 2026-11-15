# Critique of "Validation that predicts the exam, and the robot test" (key validation), lens: exam risk

Critic sub-agent, 9.11.26. Read only; nothing under D:/AI_Project was changed, nothing ran on the robot.
Question: does following the section maximise the chance of finishing UNSEEN tracks at the exam, then the time?

## Verdict: stands with changes

The core is sound and should stay: choose offline only on leave-one-sheet-out, report every fold against its own
always-forward rule, call differences inside the seed noise a tie, send at most two candidates, decide on the robot
under a pre-registered, paired, blind plan, sealed tracks drawn by a written rule, Wilson intervals. Every number the
section quotes that I re-checked (V1, V2, V6, V7, I) matches its source.

But the section is built to measure well, not to raise the finish rate. The parts most likely to end an exam run
(the stop, the executor on hard sheets, recovery off the line, twist drift, a new table or light, a moved camera)
are measured late, loosely or not at all, and improvement is only a fallback.

## Problems, most severe first

1. **Stop (high).** Ranked third, scored only on the demonstrations' own pictures, threshold tuned on held-back
   episodes of the training sheets (training_probes.fold_measures), the split the section bans for settings. End-to-end
   exact on held-out sheets: early 19 of 30 on sheet 05, never 10 of 10 at sheet 06 end A (V6;
   design/improvement/stop_by_end.out.txt); right 34 of 55 where the line end is under 5 cm from the paper edge, 13 of
   45 where it is 5 cm or more (sheet and distance confounded). Thresholds 0.05 to 0.40 between runs (V6; final
   candidate fold 01: 0.4). The cue is the paper edge with the table beyond: 164 to 176 of 210 stop pictures show no
   ink ahead, and 1,588 to 1,812 pictures more than 10 cm before the stop look the same, clustered on 04 to 06 where
   the loop runs along the edge (A, view_and_tracks-2). The far-from-edge stop meets the robot only in Phase 1.
2. **Phase 0 on sheet 01 only (high).** Sheet 01: 0 of 671 tip positions outside the workspace box (02: 609 of 2,775;
   03: 505 of 4,072); 27 of 45 sheet-04 runs pass 110 deg of the 119.75 joint-4 limit; wrist drift shifts the picture
   centre over 20 px in 23.3 % of rows if held (A, 11.2 items 10, 29; code_actions note). The model executor does not
   exist yet, and Phase 0 runs the model, so executor and model faults cannot be told apart.
3. **Improvement only as a fallback (high).** Phase 1 (2.0 to 4.8 h) chooses between models the section calls equal
   within noise; corrections only below 50 %; no gate before a retrained model replaces the winner (CR-DAgger
   fine-tuning on corrections lost 30 and 15.6 %; seeds up to 42 % closed loop, Codevilla 2019; both supported);
   Phase 2 two days before the cut-off cannot lead to a fix.
4. **Which weights (medium).** Never stated. One seed moves one sheet by up to 9.8 points of within 45 and 32 points
   of stopped right (V2). The exam must run exactly the weights that passed Phase 2.
5. **No closed-loop proxy (medium).** Puck over 2 cm off the line: path within 45 0.708 (1,105 pictures) vs 0.841
   within 1 cm (4,155); last 3 cm of line 0.649 (461) vs 0.792 (end-to-end exact, 03/05/06;
   design/improvement/situation_errors.out.txt). No per-episode direction measure.
6. **Twist as a gate (medium).** Twist sign on held-out 05/06: end-to-end classes 0.37/0.42, small CNN 0.41/0.37,
   exact 0.67/0.80 (main_session_facts). The two named alternatives would mostly test their twist heads.
7. **Light, table, camera (medium).** Table fills a median 32 % of each picture; light block last; camera jump about
   15 px vs a training shift of 12 px at 640 x 480 (calculated); camera_shift_px missed the move in 23 episodes
   (A, metadata_starts-16); no numeric tolerance.
8. **Phase 1 rests on two unseen tracks (medium)**, against the section's own clustering argument; the far end at
   "at least 8 cm" stays inside the 15 to 17 cm the camera sees ahead; the sealed rule draws 3 of the audit's feature
   kinds.
9. **Exam unknowns that change the tests (medium):** self-stop or not, attempts, who lays the sheet (a config comment
   says square; square lay-outs are 16 episodes on sheet 02 plus 2 by mistake).
10. **Robot hours (low):** measured set-up 1.86 min a trial (0.75 + 0.19 + 0.92, design/improvement/episode_timing.out.txt),
    not 1 min: about 8 h, not 6.3 h.
11. **Training length (low):** no held-out-sheet curve exists; 8 epochs is a fixed schedule.
12. **Two in a row (low):** hold or step between the two pictures is unspecified (about 1 cm of puck travel against a
    2.0 cm tolerance).

## Changed recommendation (summary)

Stop and path within 45 as the joint first gate, stop split by line-end distance, threshold chosen across held-out
sheets or on practice tracks; closed-loop proxies and perturbation tests from saved predictions; twist and early stops
as pass-fail gates; six-sheet weights named and frozen for the exam, optionally a three-seed average; Phase 0 as an
executor shake-down (demonstration replay without a model on 01, 04, 02; forced refusal; joint-4 clamp; far-end and
along-edge practice tracks; light and surface); one correction round by default after Phase 1 with a 12-block gate;
six validation tracks; a dress rehearsal by about 6.12; ask the professor now; plan about 8 h plus 2 to 4 h of
corrections.
