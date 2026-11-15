---
name: project-final-model-search
description: "14.11.26 night: round 2 recorded to the end (55 episodes, 16 tracks); final model search on ki-014 with a 16-track held-out yardstick; the stop is the weak point (paper-edge cue), three fixes under test; model 9 ready"
metadata:
  node_type: memory
  type: project
  originSessionId: 997601d4-4a2e-4eb2-91d1-890ff53278fd
  modified: 2026-11-14T21:40:46.803Z
---

State on 14.11.26, 23:10 (session of Emranul's account, branch `emranul/training-analysis`; commits local, not pushed).

**Data.** Round 2 is recorded to the end by Emranul alone on 14.11.26 (Marshal came late): 55 scored episodes on the 16
new tracks T01 to T16 (3 or 4 per track, solid and dashed prints), 4,748 pictures. The professor said the data is
enough; Emranul and Marshal are open to record more if the model work shows it helps. Index of round 1 plus all of
round 2: `03_data/processed/v03_training_index` (264 episodes, 22,646 training pictures after the audit cleaning; caches
at 160 and 320 px). Checked: no unreadable picture, no step over 2 cm, no lost puck, every end picture shows the puck on
the line's end. Decided by Emranul on 14.11.26, 22:55: episode 14-02 is left out of training (it skipped a hairpin); the recording stays on disk, the setting is "weightless_episode_patterns" in config.py; model 10 is model 9 without it.

**Yardstick** (config comment above `FULL_INDEX_FOLDER`, fixed before the results): each of the 16 new tracks held out
once, 55 test episodes on lines the model has not seen; a change is kept when it stops right on 5 more episodes without
losing more than 1 point of path within 45, or gains 1.5 points of path without stopping right less often.

**Findings so far (measured).** The stop is the weak point, not the steering. The model learned the paper's edge as its
stop cue: every round 1 line ends 2.4 to 4.3 cm from the edge; six new tracks end in the middle of the paper (T03, T05,
T07, T10, T13, T15) and there the model almost never stops right, also with all of round 2 in training
(`2026-11-14_stop_check.md`, `2026-11-14_final_model_search.md` in `06_deliverables/research_notes/Model and training/`).
Reference, two seeds, all 16 folds: path within 45 0.832 and 0.844 (round 1 only: 0.809); stopped right 11 and 17 of 55
(round 1 only: 14); on mid-paper ends 0 and 1 of 19. Seed gap: 1.2 points and 6 stops, so a fix must reach 20 right stops.
The model's look-ahead and a lower stop threshold do not find mid-paper ends either.

**Under test on ki-014** (results in `04_runs/gpu_ki014/2026-11-14_resnet18_full_*`, scored by
`evidence/final_2026-11-14/score_final.py`): two seeds of the reference; two pictures; ResNet-34; 16 rounds; score-1
episodes without weight; against the edge cue: the table repainted as paper, heavier mid-paper stop pictures, made-up
line ends (the line ahead of the puck painted over, the picture becomes a stop: not a demonstrated stop, the report must
say so if kept); the reference on 4, 8 or 12 new tracks (would more tracks help?); 320-px pictures.

**Queue state.** Second login at 22:58; the reordered queue (50 jobs, then 4 at 320 px) started 23:08 on ki-014 and should
end about 05:00 on 15.11.26; it runs under nohup. Results come back while the laptop stays connected; otherwise a
fetch-only login (`evidence/final_2026-11-14/ki014_scripts/final3_fetch.cmd`, Emranul types the password). Model 10 =
model 9 without 14-02, first job of that queue.

**Third queue (15.11.26, about 01:00).** Emranul: "you are not utilising the data to its full potential, try a superior
model, take six more hours". Built and trial-run: zoomed-in view round the puck, turned and mirrored pictures (his ideas),
far-reach pictures counted three times, ConvNeXt-Tiny, ResNet-50, a third reference seed, three combinations; it starts by
itself on ki-014 when the second queue ends (about 05:00) and should end late morning. Scripts: `final3.cmd`,
`final_fetch.cmd` in `evidence/final_2026-11-14/ki014_scripts/`. Robot findings of the same night: [[project-reach-and-lay-out]].

**State on 15.11.26, 02:30.** Merged into main (pull request 26); Emranul's trainer is `train_policy_emranul.py`
(the copy on ki-014 is still `train_policy.py`; the next upload must send the new name and the queue scripts must call
it). Third queue uploaded at 01:44 (52 jobs behind the second, ends about 11:30); fetch with `final_fetch.cmd`
(password). Stop fixes scored on all 16 tracks: none reaches 20 right stops of 55 (made-up ends stop 5 to 20 cm early;
mid-paper weighting 19 right but 1.5 points less steering). Robot, same night: model 10 is worse than model E on sheet
05's loop (laid as on 11.11, both step sizes); the two-picture round 1 model drove the loop well and stopped 4.6 cm
early; both models follow the puck and use the line about half as much; the claim "only 280 pictures show a puck brought
back" was wrong (Emranul objected, 15.11.26: most episodes had h corrections): it counted only the 35 starts beside the
line. Measured: 304 corrections by hand in 169 of 266 episodes (64 %); 2,094 training pictures in steps 2 to 8 after an
h press (the robot brings the drifted puck back) and 2,369 in the 8 steps before one (the replay goes on while the puck
drifts; trained on as good examples, only 34 marked as mistakes). Untested guess: those before-pictures teach the model
to carry on beside the line; the test is to leave them out of training and of the scoring.
Free gain found: steering by the look-ahead output scores +1 point within 45 and +2 to 2.5 within 30 degrees (both
seeds), not yet in the driver. Emranul (15.11.26): no more recording, "make it better yourself"; he was disappointed
by the loop. Marshal's three round 2 policies were driven on sheet 01 (results file in `Policy runs/`).

**Hand-correction runs (15.11.26, 03:05; Emranul: "yes build it", "put more weight on the frames after h is pressed").**
Built, tested and queued on ki-014 at 03:09 behind the third queue (fetch later with `final5_fetch.cmd`, one password): "before_correction_weight" 0 for the 5 steps before an h press, "after_correction_weight"
3 for the 8 steps after it (`config.HAND_CORRECTIONS`), four runs on the 16 tracks, one on the six course sheets, robot
candidates models 13 (one picture) and 14 (two pictures). One login with `final5.cmd` (in the evidence folder's
`ki014_scripts`, working copy in the scratchpad) uploads into `02_code_5` on ki-014, queues 20 jobs behind the third queue
and fetches everything; it calls `train_policy_emranul.py`. New measure (`score_hand_corrections.py`): the reference agrees
with the demonstration on 83 % of all held-out pictures but on 58 % of those after an h press (two pictures: 65 %). Rule
fixed before results: after-h pictures +5 points, path without before-h pictures not more than 1 point lower, stops not
worse than the seed gap; then the robot decides (sheet 03 first).

**Ideas kept for the next queue, not built (Emranul, 15.11.26, "keep it in mind").** (1) More weight on the steering of
the last 10 to 20 steps of each episode, only in well-scored episodes (score 3, puck in the goal), so the puck covers the
line end; the error audit (`error_audit.py`) shows 73 to 75 % agreement in the last 4 to 20 steps against 86 % earlier.
The stop picture itself already counts 25 times and more weight there did not pass. (2) The model almost never steps
back (right on 60 of 530 backwards pictures); first check whether the driver carries out a backwards step. (3) After the
results: combine only the changes that pass their rule into one recipe and score and drive the mix as its own run; try
after-h weight 5 if 3 helps. No negative weights (Emranul asked; weight 0 is the honest choice, explained to him).

**Model 9** (reference recipe on all 264 episodes, for the robot): 
`04_runs/gpu_ki014/2026-11-14_resnet18_model_9_full_all_sheets/checkpoints/all_sheets.pt`, sha256 3eca52154e5a; loads
in the robot driver (`ModelPolicy`), not yet driven on the robot.

**How to apply:** read the note `2026-11-14_final_model_search.md` first; scripts of the two logins are in the session's
scratchpad (may be gone) and described in the note; see [[feedback-login-tab-last-action]], [[project-gpu-pool-access]],
[[project-stop-must-cover-line-end]], [[project-exam-rules-reported]].
