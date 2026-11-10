# Beginner-reader check: training set-up analysis of 9.11.26

Checked on 9.11.26, late evening (after the queue log's 23:02 entry), by a critic agent for the main session. Lens: read
the report as Emranul, a self-described beginner who must act on it, and as Marshal. Read only: the report was not
edited, nothing ran on the robot, nothing in `03_data/raw` was touched.

Report checked (read in full, 1,645 lines; line numbers below refer to it):
`D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/2026-11-09_training_set_up_analysis.md`

Also read: the three judges' notes; the targets section and its exam-risk critic; the training-settings critics (evidence
and exam risk); the validation exam-risk critic; the report parts (`how_to_compare.md`, `results_so_far.md`,
`robot_time.md`, `professor_questions.md`, `code_set_up.md`, `data_section.md`, `literature_takeaways.md`);
`evidence/audit_synthesis.md` section 11 and finding code_actions-11; `02_code/config.py` lines 1560 to 1815;
`train_policy.py` (functions and arguments), `compare_runs.py`, `training_dataset.py` (resize), `requirements_training.txt`;
the final candidate's fold files and `checkpoints/` folder; `04_runs/comparison.md`; the queue script and logs;
`STATUS.md`; the team memory.

## Verdict

The evidence is careful and honest about what offline numbers cannot show, and the writing rules are kept (no em-dashes
found, British spelling, no "the user", Emranul's plan of 6.11 correctly attributed and confirmed in `STATUS.md` line
103). But a beginner cannot yet act on it:

1. **The summary misstates what exists.** It says the first set-up was "trained on all our pictures" and heads it
   "Build first". What exists is six test copies, each trained on five sheets. The exam copy (all six sheets) and the
   robot side (the model mode) do not exist, and the summary does not say so.
2. **Section 2 is not enough to build the first set-up.** The robot side, the exam model, where the weights are, how to
   load them and prepare a picture, the wrist and hand values and the refusal fallback are all missing or sit in other
   sections (table B).
3. **Next step (1) cannot be done with the project's tools.** The numbers it needs (fixed 0.5 with two pictures in a
   row, backing up, first rows after a hand correction) are not in the fold files or `compare_runs.py`. They exist only
   in the judges' scripts in this session's temporary scratchpad, hard-coded to other runs. The advice to copy them is
   buried in 11.2.
4. **The decisions are scattered** over sections 1, 2.5, 4.5, 7.4, 9 and 10, with no single list of what Emranul and
   Marshal must decide.
5. **The summary's backup bullet claims more than section 2.2 shows.**

Sections 1, 2 and 9 need rewriting for the beginner. The rest needs a reading guide, a fuller glossary and a few
consistency fixes.

---

## A. Does the summary (section 1) stand alone?

Not yet.

| # | Place | Problem | Proposed fix |
|---|---|---|---|
| A1 | Line 35, "Build first" | The model is already trained (six fold copies). What is still to build is the exam copy and the robot side. "Build first" sends a beginner back to training. | "Use first: the final candidate. Six test copies are trained; the exam copy and the robot side are not built yet." |
| A2 | Line 36, "trained on all our pictures", and the status table at line 14, "trained and scored on all six held-out sheets" | Both read as if a model was trained on every sheet. In fact each copy is trained on five sheets and scored on the sixth. No copy trained on all six sheets exists, and `train_policy.py` has no option for one (its only arguments are `run_name`, `--folds`, `--epochs` and `--train-row-limit`). | "Six copies were trained, each on five sheets and scored on the sixth. The copy for the exam, trained on all six, does not exist yet; `train_policy.py` needs an 'all sheets' option first." |
| A3 | Lines 37 to 38, "The robot steps 1 cm ... and stops when ..." | This is in the present tense, as if the robot already does it. The model mode does not exist (4.6), and the summary never says so; only step (2) implies it. | "On the robot it *would* step 1 cm ...; the part of the recording tool that lets a model drive is not built yet (nine pieces missing, section 4.6)." |
| A4 | Lines 41 to 43, backup bullet: "the names backed up far better after a drift (0.66 against 0.39 ...)" | Three problems. (a) "Backed up" sits next to "Backup" and can be read as "served as a backup". (b) The 0.39 is the *earlier* exact run; the final candidate on the same three sheets scored 0.426 [calculated by judge 2], and 0.342 over all six sheets. (c) "After a drift" overclaims. The measure counts all backward moves, and about half of them are ordinary pushing seen through a turned camera (targets section, line 59). Section 2.2's own risk paragraph says that on all 2,921 pictures after a hand correction the two models tied (0.605 and 0.605). | "Backup: the same network with a second output for the 8 move names, steering by their average direction. It aims at the first set-up's weak spot, pushing backwards: an earlier names model pointed 0.66 of backward moves within 45 degrees, the final candidate 0.43, on the same three sheets (one run each). Whether that helps win back a puck that slid is not shown: over all pictures after a hand correction the two kinds tied." |
| A5 | Line 44, "In practice" | These are offline scores on recorded pictures, not practice on the robot. | Rename the bullet "Offline scores (recorded pictures, not the robot)". |
| A6 | Lines 45 to 46, "stops in the right place in 102 of 209 runs (25 early, 69 never)" | (a) The 209 are recorded episodes, not robot runs. A beginner will think the robot did 209 runs. (b) 102 + 25 + 69 = 196: the 13 that stop 2 to 3 cm short are missing. (c) "Right place" is undefined in the summary. | "Of the 209 recorded episodes it would stop within 2 cm of the demonstrator's stop in 102, more than 3 cm early in 25, 2 to 3 cm short in 13, and never in 69; on the dashed loop 21 of 25 never stop." |
| A7 | Line 44, "where the line goes next" | Path within 45 compares with where the *demonstrator's tip* is three steps later, not with the line itself. | "where the demonstrator's tip went in the next three steps". |
| A8 | Line 34, "18,053 rows"; line 71, "18,032 rows" | Two row counts with no link. The difference (21 rows the recording tool marked as mistakes) is explained only in the last row of the 3.9 table (line 350). "Rows" is never defined. | "One row is one picture with the move recorded after it. The cleaning keeps 18,053 rows; the first set-up trains on 18,032 (it also leaves out 21 rows marked as mistakes); 259 are stops." |
| A9 | Line 48, step (1), "compare ... under the fixed stop rule" | The summary does not say how. The fold files hold only the fixed one-picture counts and the tuned rules (checked in `fold_without_path_06.json`). `comparison.md` has no fixed two-in-a-row column and no backing-up column. The only code that computes them is the judges' scripts in the temporary scratchpad. `judge2_checks.py` hard-codes the final candidate's folder (line 25), and `judge_rescoring.py` lists fixed run names. | "10.11, after about 10:00: score the backup and the second seed with the fixed stop rule (0.5, two in a row) and the backing-up counts. These numbers come from the judges' scripts, not from `compare_runs.py`; the scripts must first be copied into `evidence/` and pointed at the new runs. Then decide with the rules in 2.5." |
| A10 | Line 48, "decide the set-up together" | Decide what? The criteria are in 2.5, but the summary does not point there. | "...and decide with the rules in section 2.5 (keep, promote or drop the backup)." |
| A11 | Line 50, "the simulated robot" | Not explained. A beginner may think of the cancelled Webots twin. | "with the recording tool's `--simulated` practice mode (the real robot does not move)". |
| A12 | Lines 49 to 51, step (2) | This is the biggest piece of work in the plan. It has no time estimate (section 9 says "not estimated"), no owner and no first concrete sub-step. In the provisional split Marshal builds the recording tool (team memory), but the weights are on Emranul's laptop and out of git. | Add: "No estimate exists. First sub-step: load `checkpoints/without_path_01.pt` and predict on saved sheet-01 pictures, and check that the result reproduces the fold file. Agree who builds it. The weights (51 MB each) are not in git, so Marshal needs another way to get them." |
| A13 | Missing from the summary | The things to do before 10.11: write down what meeting 2 answered (only in 10.1, line 1239, and 10.2 question 7); copy the scratchpad evidence into the repo (only in 11.2, line 1321). | Add both as step 0. |
| A14 | Missing from the summary | The robot time the plan needs (about 31 to 46 hours until 20.12.26, 5 to 8 a week; 8.4 and 9). | Add one line. |
| A15 | Missing from the summary | A reading guide for a 1,645-line report. | "Read 1, 2.1, 2.5 and 9 to act. Sections 3 to 8 are the evidence, 10 the open questions, 12 how to run the code and the glossary." |
| A16 | Summary bullets "The data", "Build first", "Backup" | No markers, although the choice of set-up is a judgement by the review agents, not a measurement. | Add [measured] and [judgement] as in the other bullets. |
| A17 | Line 30, "Summary for Emranul" | Marshal is a named reader (line 5). | "Summary for Emranul and Marshal". |

### Proposed rewrite of section 1 (about 40 lines; every number is already in the report or its sources)

> **Read first:** this section, 2.1 (the first set-up), 2.5 (when to switch) and 9 (the weekly plan). Sections 3 to 8 are
> the evidence, 10 the open questions, 12 how to run the code and the glossary.
>
> **What exists tonight** [measured]. Six test copies of the first set-up, each trained on five sheets and scored on the
> sixth, in `04_runs/2026-11-09_resnet18_final_candidate/checkpoints/` (not in git). Not built yet: the copy for the exam,
> trained on all six sheets (`train_policy.py` needs an "all sheets" option first), and the recording tool's model mode,
> the part that lets a trained model drive (nine pieces missing, section 4.6). Nothing has run on the robot.
>
> **The data** [measured]. 210 demonstrations, 18,522 pictures; one row is one picture and the move recorded after it.
> The bookkeeping is right, but the hand-guided moves are noisy, 449 rows are pauses where the robot did not move, and
> episodes from one sitting look almost alike. So only a sheet left out of training completely (a *held-out sheet*)
> tests an unseen track. The cleaning, already in the code, keeps 18,053 rows. The first set-up trains on 18,032 of them,
> because it also leaves out 21 rows marked as mistakes. 259 rows are stops.
>
> **First set-up: the "final candidate"** (only the run's name in `config.py`; it is the first set-up, not a final
> model) [judgement of the three review agents]. A ResNet-18 network, pretrained on everyday photos and then trained on
> our pictures, looks at one camera picture. It predicts the next step as numbers: tip forward, tip sideways and forearm
> twist, plus the chance that this picture is the stop. On the robot it would step 1 cm in the predicted direction,
> twist within limits, keep the wrist and hand at fixed values, and stop when that chance is 0.5 or more on two pictures
> in a row. This replaces Emranul's plan of 6.11 (move names first). That decision is Emranul's and Marshal's.
>
> **Offline scores** (recorded pictures of the sheet each copy never saw; not robot runs; one training seed)
> [measured]. The push points within 45 degrees of where the demonstrator's tip went next in 74 to 86 of 100 pictures
> (always pushing straight ahead: 43 to 80). Of the 209 recorded episodes, it would stop within 2 cm of the
> demonstrator's stop in 102, more than 3 cm early in 25, 2 to 3 cm short in 13 and never in 69. On the dashed loop, 21
> of 25 never stop. Only robot runs show how often it finishes, and no setting tried here fixed the stop [judgement].
>
> **Backup: names plus exact** (results about 10:00 on 10.11). The same network with a second output for the 8 move
> names, steering by their average direction. It aims at the first set-up's weak spot, pushing backwards. An earlier
> names model pointed 0.66 of backward moves within 45 degrees, against 0.43 for the final candidate on the same three
> sheets [measured and calculated; one run each]. Whether this helps win back a puck that slid is not shown: over all
> pictures after a hand correction, the names and exact models tied [measured].
>
> **Decisions for you two:** the table in section 1.1 (proposal C below).
>
> **Next steps, in order** (who does each is yours to agree):
> 0. Tonight: write down what meeting 2 (9.11) answered. Copy the judges' scripts and notes from the temporary
>    scratchpad into `evidence/`, because step 1 needs them.
> 1. 10.11, after about 10:00: score the backup and the second seed with the fixed stop rule and the backing-up counts
>    (the judges' scripts compute these; `compare_runs.py` does not yet). Then decide with the rules in 2.5.
> 2. Build the model mode (section 4.6) and test it with the recording tool's `--simulated` mode. Then write the
>    checklist for robot session 1 (phases 0a, 0b and 1a of section 7.4). No time estimate exists.
> 3. Send the professor questions 1, 3 and 5 of section 10.1 (one of you sends them).
>
> **Robot time to plan with:** about 31 to 46 hours until 20.12.26, or 5 to 8 a week (section 8.4) [calculated, partly
> from guesses].

---

## B. Can the first set-up be built from section 2 alone?

Training it, almost (the config entry exists, but the command is only in 12.1). Using it at the exam, no. A builder
needs the following:

| What a builder needs | In section 2? | Where it is now | Proposed fix in 2.1 |
|---|---|---|---|
| The command that trains it | no | 12.1 step 4 | Add under line 64: `..\.venv\Scripts\python.exe train_policy.py resnet18_final_candidate`, run from `D:\AI_Project\02_code` (see F9 on the path). |
| Where the trained copies are | no | A.6, without file names | Add: `04_runs/2026-11-09_resnet18_final_candidate/checkpoints/without_path_01.pt` to `_06.pt` (each the model that never saw that sheet). They are out of git and only on Emranul's laptop. |
| The exam model | no | 5.5, 6.9 rule 4, 9, 11.2 | Add a row "Exam model": the same recipe on all six sheets, with the same fixed 0.5 rule. Needs an "all sheets" option in `train_policy.py` (not written); about 1.5 hours per seed. |
| That the robot side does not exist | no; only the status table says "proposals" | 4.6 | Add a row "Not built yet": the nine missing pieces of 4.6 plus the hand-over to the exact-replay decider after a reguide. |
| How to load a saved copy and predict from one picture | no | Nowhere in `02_code`: no `torch.load` anywhere; `load_state_dict` only restores the in-memory best state during training. The only working example is the critic's `stress_test_final_candidate.py` (line 126) in the temporary scratchpad. | Say plainly that no loader exists, and point to that script as a starting point once it is copied into the repo. |
| How to prepare the camera picture | partly ("shrunk to 160 x 120 ... ImageNet normalisation") | 5.5, line 691 ("exactly as `training_dataset.py` does") | Add: the 640 x 480 picture converted to RGB, then `cv2.resize(..., (160, 120), interpolation = cv2.INTER_AREA)` as in `training_dataset.py` line 436, then the ImageNet mean and spread from `TRAINING_RUN_PARAMETERS["normalisation"]`. |
| The wrist (joint 5) and hand (joint 6) values | named only ("fixed standard start values") | 4.6, line 547 ("the demonstrated median"); no number anywhere. 10.1 question 1 (line 1244) asks instead whether the controller may hold them "where they are". Glossary line 1426: "Joints 5 and 6 kept". | Pick one rule and use it everywhere. For example: "before each run, set joints 5 and 6 to the median of their start values over the demonstrations (to compute from `frames.csv` before session 1)". |
| The guards and the refusal fallback | named only ("workspace, height and refusal guards with a fallback") | 4.6 (workspace from the sheet in use). Why a fallback is needed is only in audit code_actions-11: after a refused step the loop asks again from an almost identical picture, so the model can repeat the refused step for ever. | Define the guards. Mark the fallback action (stop, or hand over to the operator?) as a decision for the team; nothing proposes one yet. |
| Signs of the outputs | no | audit 11.2 point 28: "signs to test on the simulated robot first" | Add one line to the phase-0a check. |
| Which settings keys to change for the step budget | partly | 4.6 (`is_step_budget_enforced` is False; the 80-step fallback for sheets not listed) | Name the keys in the executor row. |
| A warning not to change the stop weight | no | 6.3, line 758 | Add: "Do not change the stop weight (10) without re-checking the stop, because it sets what 0.5 means." |
| Which seed's copies go to the robot | no | 6.9 rule 4 covers only the exam model | Add: "Phase 1a uses the first seed's copies unless the second seed is clearly better; do not choose between seeds on the robot." |

---

## C. The buried decisions: a proposed table to put right after the summary (as section 1.1)

| # | Decision for Emranul and Marshal | Proposal | Evidence | Already in the code? |
|---|---|---|---|---|
| 1 | The exact step instead of the move names first (changes Emranul's plan of 6.11) | exact step | 2.4, 4.1 | yes, in the final candidate |
| 2 | Stop rule | fixed 0.5, two pictures in a row, the same for every model | 4.5 | no: the fold files and `compare_runs.py` report only one picture at 0.5 and the tuned rules |
| 3 | Robot rules for the first tests | 1-cm step, twist clamped, wrist and hand at fixed values, joint 4 within 115 degrees, step budget on | 2.1, 4.3 to 4.6 | no: the model mode does not exist |
| 4 | What the robot does after a refused step | none yet | audit code_actions-11 | no |
| 5 | Keep, promote or drop the backup | rules of 2.5 | 2.5 | run pending |
| 6 | The audit cleaning (449 null rows dropped, 50 stops relabelled, score 1 kept) | as applied | 3.9, 10.3 | yes, applied before you decided; undoing it needs a retrain |
| 7 | Who builds the model mode, and by when | none yet | 4.6, 9 | no |
| 8 | Which professor questions go now, and who sends them | 10.1 questions 1, 3 and 5 first | 10.1 | no |
| 9 | The plan for robot session 1 (phases 0a, 0b, 1a) | 7.4 | 7.4 | needs a checklist first |
| 10 | The seven facts only you can confirm | answer them | 10.2 | no |

Section 10.3 marks some items "done in the cleaning" (lines 1278 to 1280). Say that "done" means applied by Claude in
the final candidate, still open to reversal, so that "Emranul and Marshal decide" (line 20) stays true.

---

## D. Places that do not say what to do next

| # | Place | Problem | Fix |
|---|---|---|---|
| D1 | 11.2, lines 1321 to 1322 | "Copy them into `evidence/` if they should be kept" is the last bullet of a "did not do" list. The scratchpad holds everything the 10.11 decision needs: the judges' scripts, the stress-test loader and the queue script and logs. It is temporary. | Move this to the summary as step 0, with the exact folders (`design/judge_1`, `design/judge_2`, `design/judge_3`, `design/training_settings/critic_exam_risk`, `run_end_to_end_queue.sh`, `logs/`). |
| D2 | 2.5, line 169, "with clearly better first rows after a hand correction" | No baseline and no tool. Judge 1 gives the baseline (415 of 1,124 step within 45 and 573 of 1,062 path within 45 over six sheets). The report does not, and only `judge_recovery_checks.py` computes it. | Quote the baseline and name the script. |
| D3 | 2.5, lines 168 to 175 | The rules leave gaps: "keep" needs at most about 3 points lower, "drop" needs more than about 5 points lower. Between 3 and 5 points, or between "about 95 right" and "clearly worse", nothing is said. | Add: "between these, keep it as the backup, mark it weak, and let the robot decide". Or close the gap. |
| D4 | 2.5, lines 187 to 189 | "The plan changes" if the professor forbids joint angles, but not how. | Say plainly: "then no tested plan exists; the model would have to output joint moves directly, which nothing here has tried". |
| D5 | 10.1, lines 1238 to 1240 | "Record the answer there first" (meeting 2): there is no owner and it is not in the summary. Claude cannot send messages to the professor; one of the students must. | Put it in the summary's step 0. Say who sends the questions and by which channel, and before which date (meeting 3 is on 23.11). |
| D6 | 11.3, line 1331, "check the queue" | It does not say how. The queue's log is `scratchpad/logs/e2e_queue_3.log`. It shows `resnet18_finetune_exact` finishing at 23:01 on sheets 01, 02 and 04 and `resnet18_finetune_mirror` starting at 23:02, which supports the report's guess that the remaining sheets are 01, 02 and 04. | Name the log file and how to read it. |
| D7 | Section 9, week 16.11 to 22.11 | Session 1 depends on the model mode, which has no estimate. Nothing says what happens if it is not ready. | Add: "if the model mode is not ready, move session 1; nothing else on the robot that week depends on it". Or say what does. |
| D8 | Section 9, every week | No owners. The memory's provisional split (Emranul runs robot sessions; Marshal builds the recording tool and sorts out the GPU pool) is not used. | Add an "Owner" column or one line per week. |
| D9 | 12.1 note, lines 1376 to 1377 | It warns to read path within 45 first. It does not warn that the first and third rows of `comparison.md` are held-out-*episode* probes (split "episodes", within 45 0.707 and 0.655), which the report says must never be used for choosing. | Add: "Ignore rows whose split is 'episodes'; they are optimistic (section 3.6)." |

---

## E. Jargon used before it is explained

Line 26 promises: "Technical words are explained where they first appear and in the glossary". In sections 1 and 2 this
is mostly not kept. Terms used there without an explanation nearby, with a proposed plain meaning (add each to 12.3):

| Term | First use | In the glossary? | Plain meaning to add |
|---|---|---|---|
| row | line 33 | no | one picture with the move recorded after it |
| head | line 41 | yes, but only in 12.3 | the last small layers that turn the network's numbers into a move, a twist and a stop chance |
| seed | line 48 | yes, but only in 12.3 | the random start of a training run; the same set-up with another seed gives somewhat different results |
| simulated robot | line 50 | no | the recording tool's `--simulated` practice mode; the real robot does not move |
| design panel, critics, judges | line 58 | no | Claude agents that drafted, attacked and ranked the options; not people |
| chunk target | line 59 | yes ("Chunk"), defined in 4.1 | the tip's place after each of the next 5 steps |
| step within 45, path within 45 | lines 74 and 78 | yes, but defined in 4.1 (line 377) | move the two definitions to the top of section 2 |
| ImageNet normalisation | line 68 | no | the picture's colours scaled as for the photos the network was pretrained on |
| `timm resnet18.tv_in1k` | line 69 | no | the exact pretrained ResNet-18 file, from the timm library |
| feature map, 3 x 4 grid | line 69 | no | the network's last picture-like layer, cut into 3 x 4 cells so the head knows where things are |
| hidden layer, dropout | line 69 | no | one middle layer of 256 numbers; dropout switches off a random 20 % of them in training, against learning by heart |
| parameters | line 69 | no | the numbers that training adjusts |
| the picture's frame, label frame | lines 70, 251 | no | directions measured in the picture: up is forward, left is left |
| targets | line 70 | no | what the network is trained to predict |
| "unweighted" (Huber loss) | line 71 | no | here: no extra weight for rare move directions (section 6.3) |
| positive weight 10 | line 71 | no | in training, a stop picture counts ten times as much as another picture |
| AdamW, learning rate, weight decay, batch, warm-up, cosine decay | line 71 | no | one line: "the training settings; set in `config.py`, nothing to change" |
| passes | line 71 | no | times every picture is seen once |
| guards, fallback | line 72 | no | see B; the fallback is not defined anywhere |
| class heads, names head, names model, names-only twist heads | lines 84, 100, 108, 111 | partly | one term for all of these, for example "move-name model" |
| dithering | line 122 | no | stepping back and forth |
| old rules | line 130 | no; defined only in a source line (line 398) | no audit cleaning, and the weights of the epoch with the lowest validation loss |
| best-validation weights | line 398 | no | the network as it was at the epoch with the lowest validation loss |
| ACT, diffusion policy, DINOv2, CLIP | line 135 | no | one line each; the professor may ask why they were not used |
| episode-bootstrap interval | line 143 | no | a range from redrawing the held-out episodes; it shows chance from which episodes were tested, not from the training seed |
| robomimic Square, Implicit BC, regression, bins | lines 145 to 148 | no | a short note: published robot benchmarks and two ways of predicting a move |
| most likely class | line 146 | no | the robot executes the single most likely of the 8 names |
| "least-squares gains of the demonstrated on the predicted twist" | lines 159, 436 | no | "the demonstrated twist was 0.55 to 1.24 times the predicted twist, depending on the sheet" |
| rules hedge | line 190 | no | a fallback in case the professor forbids something |
| the recipe, "the recipe's former rule" | line 499 | no | the training-settings draft's rule (threshold tuned on validation episodes) |
| decider | line 539 | no | the part of the recording tool that picks the next step from the latest picture |
| open loop | line 440 | no | adding up predictions without the robot moving or new pictures arriving |
| sign test, p, power, prediction interval, bootstrap | 6.1, 6.9, 7.3, 8.2 | no | a short plain sentence each, or one line saying "statistics checks; the conclusion is in the sentence after" |
| GPU pool | 5.5 and later | no | the university's graphics-card computers; access not yet confirmed |
| "frozen" | four meanings | partly | frozen features, a fixed table that must not be regenerated (line 209), locked executor settings, and freeze dates. Keep "frozen" for frozen features only; say "fixed" or "locked" for the rest. |

---

## F. Numbers and words that will trip a beginner

| # | Place | Problem | Fix |
|---|---|---|---|
| F1 | Lines 42, 91, 107 and 162 | Backing up appears as 0.66, 0.39, 0.342, 0.426 and 0.663 against 0.392 in four places, over different runs and sheets. | One small table in 2.2: names model (earlier run, 03/05/06) 0.663; exact (earlier) 0.392; final candidate (03/05/06) 0.426; final candidate (all six) 0.342. |
| F2 | Line 91, "0.342 of 1,894 backward moves" | A share and a count side by side read as "0.342 moves". | "647 of 1,894 (34 %)". |
| F3 | Lines 84 to 86 (why it comes first, bullet 3) | The stop evidence is for one picture at 0.5 (47 of 99), but the proposed rule is two in a row. Under that rule judge 1 gives 46 / 12 / 36 of 99. | Give both, or the proposed rule's numbers. |
| F4 | Lines 58 to 60 and 137 | "Section 2.3 gives the full order", sourced "[judges 1 to 3]". The order below rank 2 is judge 1's. Judge 2 put the small network third and the chunk fourth. Judge 3 put the chunk second and names plus exact fifth. | "All three judges agree on rank 1 and two of them on rank 2. Below that the order is judge 1's: judge 2 put the small network third, judge 3 put names plus exact fifth." |
| F5 | Line 160 (2.4, "Wrist and hand held") | The evidence listed argues against the choice: held at one value, the picture centre moves more than 20 px in 23.3 % of rows. The reason it is still chosen (the keyboard and gamepad solves are worse, and no executor for a predicted wrist exists) is in 4.6, not here. | Add the comparison and the widened-shift proposal to the row. |
| F6 | Lines 970 to 997; line 1303 | The phase table and the robot-time table use a guessed 1 minute to lay the puck. The validation exam-risk critic corrected this to a measured 1.86 minutes per trial ("about 8 h, not 6.3 h"). The report keeps the guess in the tables, names the measured value only in a sentence (line 985), and still lists "1 minute" under guesses in 11.1. A beginner planning a session gets two different answers. | Use 1.86 minutes in both tables, give 8 hours as the planning total, and drop "1 minute" from the guesses (or say it is superseded). |
| F7 | 3.9 last rows (line 350) against 10.3 question 12 (line 1282) | 3.9 drops "ordinary moves before a correction" (the 21 mistake rows); 10.3 says to keep "the rows just before corrections" and that this is in the final candidate. Both are true (the 21 lie inside the 1,186 window rows), but they read as a contradiction. | In 10.3: "keep the 1,186 rows before corrections, except the 21 the recording tool marked as mistakes". |
| F8 | The report's name for the model | "Final candidate", "first set-up", "S1" (in the judges' notes) and "the exam model" mean different things. A beginner will take "final candidate" to mean the final model. | Explain once in 2.1: "'final candidate' is only the run's name; it is the first set-up. The exam model is a seventh copy, trained later on all six sheets." |
| F9 | 12.1 commands (lines 1355 to 1361 and 1370) | `..\.venv\Scripts\python.exe` works only from `D:\AI_Project\02_code`. In the worktree, where this branch and report live, `..\.venv` does not exist (checked: only `D:\AI_Project\.venv` exists, and the queue script uses `../../../../.venv`). | Say "run from `D:\AI_Project\02_code` after this branch is merged or checked out there". Or give the worktree path. |
| F10 | 12.2, lines 1405 to 1407 | Marshal is told to run an unpinned `pip install torch torchvision timm pandas scikit-learn`, while `02_code/requirements_training.txt` pins the exact versions used. | "`python -m pip install -r 02_code/requirements_training.txt`". |
| F11 | Line 397 and 5.2 sources ("their `metrics.json`") | For a run trained in parts, `metrics.json` holds only the last part. The earlier exact run's file now holds sheets 01, 02 and 04 (rewritten at 23:01; within 45 mean 0.6748), not the 03/05/06 numbers quoted. `compare_runs.py` rebuilds from the fold files, so the comparison is safe, but a beginner checking a number in `metrics.json` will not find it. | Cite the fold files for per-sheet numbers, and add one sentence on this behaviour. |
| F12 | 4.5 (lines 482 to 489 against 496 to 500) | The first stop table gives shares, the second counts (0.483 is 101 of 209). | Use counts in both, or both forms. |
| F13 | 2.1 "Executor" row and 4.6 against 10.1 question 1 (line 1244) and the glossary (line 1426) | "Fixed standard start values", "where they are" and "kept" describe the wrist and hand three ways. | One wording everywhere (see B). |

---

## G. Smaller wording points

- Line 32, "bookkept correctly": "every picture has the right label and file" is plainer.
- Line 12, "two independent checkers", and line 196, "two more agents": say these are Claude agents, so the reader does
  not take them for people.
- Lines 91 and 118, "Main risks": the dashed-loop failure (21 of 25 never stop) is the most striking risk for an unseen
  track with dashes, but it is not in 2.1's risk line. Add it.
- 4.6, "Esc polling": "checking for the Esc key while the model decides".
- 7.4 phase 1a: say why sheet 02 is left out of the held-out-sheet runs (it is not explained).
- 10.2: say where to write the answers (for example `05_meetings`, or tell Claude), so they reach the shared memory.

---

## H. Ranked problems (the same list is returned to the main session)

1. High: the summary and status table misstate what exists (A1, A2, A3).
2. High: section 2 is not enough to build the first set-up: no exam model, no robot side, no weights' location, loader
   or picture preparation (table B).
3. High: next step (1) cannot be done with project tools; the needed scripts are temporary and hard-coded (A9, D1).
4. High: the summary's backup claim overstates the evidence and uses the wrong comparison run (A4).
5. High: no single list of the decisions for Emranul and Marshal (C).
6. Medium: jargon in sections 1 and 2 not explained where first used; the glossary lacks many terms (E).
7. Medium: "runs" for recorded episodes, "in practice" for offline, and the stop counts do not add up to 209 (A5, A6).
8. Medium: wrist and hand described three ways and with no values; the 2.4 evidence reads against the choice (B, F5,
   F13).
9. Medium: the refusal fallback and the guards are undefined, and the fallback action is an undeclared decision (B).
10. Medium: 2.3's "full order" is one judge's order (F4).
11. Medium: 2.5's switch rules have gaps and lack the after-correction baseline (D2, D3).
12. Medium: robot-time tables use the guess the critic replaced with the measured 1.86 minutes (F6).
13. Medium: no owners, no estimate and no first sub-step for the model mode; the weights are out of git (A12, D8).
14. Medium: meeting 2's answers and the professor questions have no owner or step (A13, D5).
15. Medium: the two row counts, 18,053 and 18,032, are not linked; "row" is undefined (A8).
16. Medium: the backing-up numbers are spread over four contexts (F1, F2).
17. Medium: "final candidate" reads as "final model" (F8).
18. Medium: no reading guide for a 1,645-line report (A15).
19. Medium: `comparison.md`'s top rows are held-out-episode probes, with no warning (D9).
20. Low: the 12.1 commands fail from the worktree (F9).
21. Low: the 2.1 stop evidence uses one picture, while the proposal is two in a row (F3).
22. Low: the 10.3 "keep" reads against the 3.9 "drop the 21 mistake rows"; "done" items were applied before the team
    decided (F7, C).
23. Low: "check the queue" without saying how (D6).
24. Low: `metrics.json` citations for runs trained in parts (F11).
25. Low: Marshal's unpinned install (F10).
26. Low: markers missing in the summary; "Summary for Emranul" leaves out Marshal (A16, A17).
27. Low: "the simulated robot" and "where the line goes next" in the summary (A11, A7).
