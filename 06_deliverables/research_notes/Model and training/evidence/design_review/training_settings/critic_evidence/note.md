# Evidence check of "Training settings" (key training_settings)

Checked 9.11.26 (after 19:42) against the files the section names. Read only: nothing under D:/AI_Project was changed.
My own recounts are in this folder: `dump_probe_fields.py`, `dump_run_folds.py` (fold files and histories, outputs
`run_folds_final_mirror.txt`, `run_folds_classes_exact.txt`) and `back_move_errors.py` (output
`back_move_errors_output.txt`).

## What checks out

Nearly every number matches its source:

- the seed spreads and sign tests (S1 lines 203 to 214; S3 lines 2 to 13)
- the like-for-like re-scores (S2)
- the class-weight, head and label-smoothing differences (S1)
- the cleaning counts: 449 + 1 + 19 = 469 dropped, 18,053 then 18,032 rows, 259 stops (S4; audit 11.1)
- the per-class recall (S5)
- the end-to-end fold numbers, the loss histories and the final candidate's sheet 01
- the augmentation settings and the code: shift ±4 px with edge padding, blur sigma 0.3 to 0.8, gamma exp(±0.25)
  (train_policy.py lines 176 to 206)
- the seeding by fold position (lines 446 to 464), the stale comments (config.py 1566 to 1567, train_policy.py 12) and
  the seeds-needed table

The literature claims I traced are all marked supported or corrected by their checkers, and the corrected forms are
used: augmentation 4, 7, 8, 10, 11, 19, 20, 35, 42; offline_vs_closed_loop 12, 20, 23, 29, 44; pretrained 27, 38, 39,
44, 50, 51; action_representation 31 and 24; evaluation_statistics 15, 25, 26; interactive_correction 29.

## Problems found

1. **The stop rule in the recipe (tuned threshold, two pictures in a row) is not supported; the evidence is mixed.**
   - In the frozen probes the exact head does worse with it: 0.221 tuned against 0.291 fixed (designs.json
     exact_step). Over 4 seeds the mean is 0.285 against 0.303, and early stops rise from 0.330 to 0.396 (seeds.json).
   - End to end (exact 03, 05, 06 and final candidate 01, 02; 165 episodes), it stopped right in 82 against 78, with
     early stops 44 against 31 and never-stopped 33 against 50 (fold files).
   - Final candidate sheet 02, finished after the section was written: 19 of 45 tuned against 24 of 45 fixed.
   - The threshold is tuned on inner validation episodes. These are near-twins of training episodes
     (duplicates_leakage-3), which is the section's own reason for distrusting validation loss.
   - The tuned thresholds run from 0.10 to 0.40 across folds.
2. **The stop noise floor and every stop comparison use the fixed 0.5 threshold, but the recipe uses the tuned rule.**
   - Seed spread on the tuned two-in-a-row measure (S1 lines 206, 210, 214): exact 0.221 to 0.323 (range 10.2 points,
     sd 4.45); classes 0.304 to 0.361 (sd 2.48); classes seeds 1 to 3 0.268 to 0.356 (sd 4.61).
   - On that measure the stop "gains" fall inside the noise:
     - square-root against no class weights: 5.4 points, with no weights higher on 2 of 6 (S1 line 9)
     - score weights 0.4 / 0.7 / 1.0: +1.3 (line 64)
     - score 3 only: -1.1 (line 53)
3. **The mirror comparison is confounded by the checkpoint.** Runs before the final candidate kept the
   best-validation weights (config.py line 1801; train_policy.py lines 370 to 381).
   - The plain class run's weights came from epoch 7 (sheet 03) and epoch 5 (sheet 05). The mirror run's came from
     epoch 8 on both sheets (fold histories).
   - So 35 against 17 compares models trained for different lengths. It is also one seed, and the fixed rule only.
   - In favour: the threshold-free stop average precision also rose (0.353 to 0.446; 0.155 to 0.266).
   - The audit's condition (11.2 point 26: mask the robot's corner) is not mentioned.
4. **The backup class head was never measured with last weights.** The 2.6 and 8.3 point costs, the 26.6 % back recall
   and the 35 of 75 all come from epochs 5 to 7 of the class run.
   - The 2.6-point within-45 cost is inside the section's own 3.5-point floor.
5. **Back moves: two definitions are mixed.**
   - The 0.3 % recall is for straight back (3.9 % of moving pictures). The 11.8 % and the 60 % after a correction count
     all three back classes (S4; label_consistency-13).
   - The class head's "about their real rate (4.7 % against 3.9 %)" holds for straight back only. Across all three back
     classes it predicts 22.1 % against 13.6 %, and back_left 10.8 % against 4.3 % (S5).
   - My recount of within 45 on true back moves (03, 05, 06 pooled):

     | Move | Exact head | Class head |
     |---|---|---|
     | Straight back (305 moves) | 19.0 % (median error 69 deg) | 60.3 % (median error 36 deg) |
     | All three back classes (1,054 moves) | 39.2 % | 66.2 % |

   - So the exact head's weakness is real, and the class head's over-prediction of back moves (a dithering risk) is not
     reported.
6. **Training time is understated.** The section gives 20.4 to 34.6 minutes per sheet, but mirror fold 05 (written
   before the section) took 42.2 minutes and the final candidate's fold 02 took 48.8.
   - A six-sheet run is therefore 2.0 to 4.9 hours.
   - The 31 to 53 hours of planned experiments become about 31 to 73 hours.
7. **Smaller points**
   - 3.4 points should be 3.3 (data_amount.json: 0.5405 to 0.5737).
   - "Score 3 only hurts beyond the noise" is marginal: -4.3 on a 134-episode subset against a floor measured on all
     208 episodes.
   - The section's audit list D leaves out D2 (two hand pictures, frame_index 6459 and 12603), D1 and D5.
   - "The largest are recording stalls": the audit says only 4 of the 25 clipped steps are stalls. It recommended
     clipping 4 twists; the implementation clips 13.
   - The augmentation checker called shifts the best-supported part. It did not say that of the label table in note
     section 4 and 6.2.
   - "A fold on its held-out sheet is a test of an unseen track" is judgement: each fold also holds out a day, a light
     and a camera period (audit 11.2 point 2).
   - Doubling the episodes "added 0.9 points" is inside the seed range: 0.541 to 0.574 at full data.

## Verdict

The recommendation stands with changes:

- Keep the training recipe unchanged for the first robot tests.
- Do not freeze the stop rule. Score all six final-candidate folds with the fixed rule and the tuned rule (one and two
  pictures in a row), like for like, and report the tuned-rule noise floor.
- Re-measure the mirror and the backup class head with last weights.
- Report all three back classes for both heads.
- Update the time plan.
