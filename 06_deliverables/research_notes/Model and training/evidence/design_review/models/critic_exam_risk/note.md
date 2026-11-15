# Exam-risk critique of "Which model, and does it fit the laptop and the robot loop" (key models)

Lens: does following the section maximise the chance of finishing UNSEEN tracks at the exam, and then the finishing
time? Written 9.11.26. Offline only: nothing was run on the robot and nothing under D:/AI_Project was changed.

Verdict: the main model stands, with changes. The fine-tuned ImageNet ResNet-18 is a sound lead candidate. But the
section chooses it on direction measures alone. It leaves the weakest part for finishing, the stop, unaddressed, and
it holds back the first robot evidence until after about 20 to 32 laptop hours. That evidence then goes on the least
important decision (two candidates 1.7 to 2.9 points apart).

## What I did

- I read the section, design/main_session_facts.md, AUDIT_SYNTHESIS.md section 11 (plus lines 600 to 640 and 1040 to
  1078), audit/code_actions/note.md, 04_runs/comparison.md and the fold files of every end-to-end run. That includes
  the final candidate's sheet 03, which finished after the section was written. I also read the literature check notes
  for offline_vs_closed_loop, interactive_correction, evaluation_statistics, augmentation, pretrained_representations,
  action_representation and action_chunking.
- I recomputed from the saved test predictions with three read-only scripts in this folder:
  - stop_and_error_runs.py (output .out.json). It gives the stop outcome per episode for thresholds 0.05 to 0.9 and
    for 1 to 5 pictures in a row, streaks of direction errors, and the twist bias. It reproduces every fold file's
    tuned_two_* numbers exactly.
  - early_stop_places.py (outputs .out.txt and .out.csv). It shows where each early stop fired.
  - stop_outcome_by_end.py (outputs .out.txt and .out.csv). It gives the stop outcome by the line end that was the
    goal.
- I looked at 6 pictures under 03_data/raw/main_data:
  - three false stops: 06-02 frame 032, 05-01 try 3 frame 049, and 03-38 frame 010;
  - two real ends where the model stopped: 05-01 try 3 frame 098 and 06-01 frame 104;
  - one real end where it never stopped: 06-14 frame 119.

## Findings the section does not show (measured unless marked)

1. **The stop fails in about half of the held-out episodes, for every model tested.** The run-time rule is the fold's
   tuned threshold with two pictures in a row. Under it, the exact-step ResNet-18 had these outcomes:

   | Sheet | Episodes | Stopped right | Stopped early | Never stopped |
   |---|---|---|---|---|
   | 01 | 20 | 16 | 3 | 1 |
   | 02 | 45 | 19 | 8 | 13 |
   | 03 | 45 | 34 | 8 | 2 |
   | 05 | 30 | 5 | 19 | 6 |
   | 06 | 25 | 8 | 6 | 11 |
   | All five sheets | 165 | 82 | 44 | 33 |

   Sheets 01 and 02 are the final candidate; 03, 05 and 06 are resnet18_finetune_exact. The final candidate's newly
   finished sheet 03 gave 31, 10 and 1 of 45. No rule on the stop probability fixes this:
   - Three pictures in a row gives 50 right, 33 early and 76 never.
   - The best single setting chosen with hindsight on the test sheets gives 83 right, 33 early and 40 never.

   The small CNN is no better. On sheets 03, 05 and 06, out of 100 episodes, it stopped right 44 times, early 22
   times and never 32 times. The ResNet-18 had 47, 33 and 19.
2. **Most early stops fire far from the end.** 27 of the 44 early stops fired more than 8 cm from the goal, 20 of them
   on the loop sheets. On sheet 05 the 19 early stops fired after 24 to 67 % of the episode's pictures, 12 of them 12
   to 19 cm from the goal. Of the pictures more than 3 cm from the goal, 343 of 11,091 (3.1 %) passed the tuned threshold; on
   sheet 05 it was 211 of 2,529.
3. **The stop seems to key on the paper edge, not the line end.**
   - All three false-stop pictures I viewed show the paper edge within about one to three puck widths of the puck,
     with the table filling the top of the picture. So do the two real ends where the model stopped.
   - Sheet 06: all 10 runs that ended at end A never stopped. Their stop probability over the last three pictures was
     0.000. In the end picture I viewed, the track and paper lie ahead, with no paper edge.
   - Sheet 05: none of the 9 runs that ended at end B stopped right (stop probability at the end 0.000 to 0.060).

   This matches the audit's guess (AUDIT_SYNTHESIS line 630; section 11.2 point 13). It also matches the risk the
   audit ranks highest for unseen tracks: a line end away from a paper edge (view_and_tracks-16).
4. **Direction errors come in streaks.** I counted episodes with 5 or more consecutive pictures more than 45 degrees off
   the 3-step path. With the exact-step ResNet-18 that happened in 96 of 165 episodes. On sheets 03, 05 and 06 (100
   episodes):

   | Model | Episodes with such a streak |
   |---|---|
   | Exact-step ResNet-18 | 66 |
   | Classes ResNet-18 | 66 |
   | Small CNN | 71 |
   | Small CNN with line-and-puck maps | 74 |
   | "Always forward" | 85 |

   The median longest streak was 5 to 6 pictures on sheets 02 to 06, so the backbone hardly changes the streaks.
5. **The twist has a bias that would add up over a track.** The signed error per picture is +0.21 to +0.34 degrees on
   sheets 01, 02, 03 and 06, and -0.37 on sheet 05. On sheet 05 the model predicts 0.06 degrees per picture where the
   demonstrations average 0.43. Summed over an episode's demonstrated pictures, the median error is -41.5 to +21.4
   degrees. The 90th percentile of the absolute sum is 54 to 168 degrees. The forearm reached -119.35 and +118.92
   degrees in the demonstrations, against a limit of 119.75 (code_actions note, section d). What this means in closed
   loop is judgement.
6. **The predicted step length is never saved or scored.** The prediction files hold only the angle, so the offline
   numbers say nothing about step length, stalls or finishing time.

## Problems with the section (details in the structured result)

1. High: the choice rests on direction measures; the stop failure above is not weighed, and no candidate fixes it.
2. High: robot evidence comes only after the seeds, the backup runs and the exam model, and it is spent on the
   closest pair. The exam model can be tested on unseen tracks only if new tracks are printed, and the plan prints
   none. The model mode of the recording tool (code_actions note, section f, items 1 to 9) is on the critical path
   but is not mentioned.
3. Medium: the outputs are added up by the executor (twist, lean, step length, joint 4), but the section never says
   how they are executed.
4. Medium: exam conditions (camera re-taped or knocked, light, a new table) are not tested. The shift augmentation of
   4 px at 160 px (2.5 %) only covers the one camera move seen; the literature's normal range is about 5 %. The
   line-and-puck maps were dropped before their purpose was tested.
5. Medium: compounding errors are invisible offline. There is no recovery validation set and no correction round.
6. Medium: track variety moved held-out-sheet accuracy about as much as the backbone gap. In the frozen probes, going
   from 4 to 5 training sheets added 2.3 points; doubling the episodes added 0.9. Yet the plan spends its laptop hours
   on seeds and the backup first.
7. Low: the speed argument is settled for the current executor only. Chunk blending and the speed cap are robot
   questions.
8. Low (in favour of the main model): ImageNet starts reduced the spread of closed-loop results between seeds
   (Codevilla et al. 2019). The section does not use this point.

## Better recommendation

Keep the fine-tuned ResNet-18 with the exact-step output as the lead candidate. Do not call the model chosen until it
passes four gates, in this order:

1. **Stop gate.** Report the stop outcome per sheet and per line end as a selection measure. Run end-to-end stop
   variants:
   - a near-goal window target (frozen probes: 0.318 right and 0.276 early, against 0.221 and 0.414);
   - a separate stop network, or a lower positive weight;
   - added stop examples at line ends away from the paper edge and inside loops.

   Fix one threshold and one pictures-in-a-row rule from the pooled held-out-sheet predictions. A fixed 0.3 with two
   in a row gives 82 right of 165, as good as tuning per fold.
2. **Robustness gate.** Run an offline stress test with the existing checkpoints:
   - shifts of 8 and 16 px;
   - rotations of 2 and 5 degrees;
   - brightness 0.6 and 1.3;
   - a colour cast;
   - another table border.

   Test the main model, the backup and the line-and-puck variant, and widen the shift augmentation if needed.
3. **Executor gate.** Save and report the predicted step length, the twist bias and the error streaks. Build the model
   mode: fixed-wrist solve, joint 4 clamped at about ±115 degrees, step clip, refusal fallback, step budget on.
4. **Early robot gate**, after the team decides and with a checklist. Run the existing fold checkpoints on their own
   held-out sheets, about 5 runs each. That is about 2 to 2.5 robot hours, calculated. Record finish, progress,
   failure type, steps and time.

Alongside the gates:
- Print at least 10 new exam-size tracks now and keep them sealed for one final 10 x 3 test.
- Pilot one human-gated correction round.
- Run the seeds and the backup afterwards. This reorders the plan; nothing is dropped.
- Use the 24-block main-against-backup comparison only if both models finish most smoke-test runs.
- Leave these to robot tests: step length (predicted or fixed), the wrist rule, output smoothing, chunk blending and
  the speed cap.
