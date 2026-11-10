# Evidence critique of "Which model, and does it fit the laptop and the robot loop" (key models)

Written 9.11.26 by the evidence critic. Read only: nothing under D:/AI_Project was changed. Every number below was read
from the JSON files or notes named, with the two read-only scripts in this folder (extract_probe_numbers.py and
best_probe_per_fold.py; printouts probe_numbers.txt and best_probe_per_fold.txt).

## What checks out

- Almost every number in the section matches its file, to the third decimal: the frozen backbone table (backbones.json,
  baselines.json), the end-to-end table (four metrics.json files), denominators (direction_rows, path_direction_rows,
  twist_rows, test_episodes), the history table (history.json), the seed ranges of the classes probe (3.1 points on the
  mean; 1.5 to 9.1 per sheet), the final candidate's two folds, epoch seconds and minutes, model sizes
  (count_model_size.out.json), checkpoint bytes, speed.json, extraction_summary.json, the audit timings (code_actions
  note, section (e)) and the config line numbers.
- Every literature claim cited is marked supported (or used in its corrected form, ACT claim 15, history claim 18) by
  its checker.
- The section labels most judgements and calculations.

## Problems, most important first

1. **The main-over-backup order is not a measured result.** The only like-for-like test (classes output, sheets 03,
   05, 06) gives the ResNet-18 a lead of 2.9 / 2.5 / 1.7 points of step within 45 (0.663 / 0.659 / 0.640 against 0.635
   / 0.635 / 0.623). Against the probe's per-sheet seed ranges on those sheets (1.5 / 6.1 / 2.9) only sheet 03 clears
   noise; one seed each; a 3-of-3 sign pattern has p = 0.125. The arms also differ in learning rate, epochs (8 against
   16), patience and kept weights: the ResNet-18 classes run kept its lowest-validation-loss epochs 7 / 5 / 7, the small
   CNN 15 / 16 / 16 (fold histories; train_policy.py lines 370 to 381). Neither candidate exists in final form on
   sheets 03 to 06. "It scored best wherever it was tested" should be presented as judgement.
2. **03 / 05 / 06 numbers belong to a sibling run.** 0.700 / 0.667 / 0.675 are resnet18_finetune_exact, trained and
   scored without the audit cleaning; the recommended resnet18_final_candidate has sheets 01 and 02 only. (In the exact
   run the lowest validation loss fell at epoch 8 in all three folds, so best and last weights coincide there.)
3. **"Repeat the previous move" beats every picture model on step within 45 on every sheet, not only the arc.** 0.796
   / 0.648 / 0.711 / 0.785 / 0.783 (sheets 01, 02, 03, 05, 06) against 0.779 / 0.632 / 0.700 / 0.667 / 0.675. On path
   within 45 the model wins everywhere: by 9.2 to 19.6 points over that rule and 5.0 to 35.8 over always forward. The
   arc is not special; the step measure rewards step-to-step copying. Lead with path within 45.
4. **Sheets 01 and 02 compare different test sets.** The final candidate was scored after the cleaning (578 / 2,425
   moving pictures, 20 / 45 episodes; 01-09 and 02-41 re-admitted), the probes and rules before it (569 / 2,372; 19 /
   44). "Best frozen probe" (0.782, exact_stop_on_goal_distance) is the maximum of about 70 probe variants. The
   3.2-point gain over always forward on sheet 01 is inside that sheet's 9.1-point probe seed range.
5. **The noise yardstick is too small for the recommended output.** The exact-step probe's per-sheet seed ranges
   reach 9.8 points of step within 45 and 16.0 of path within 45 (sheet 06), and 20.0 to 31.6 points of tuned
   two-in-a-row stopped right on sheets 01, 03, 04 and 06 (seeds.json, designs.json). End-to-end seed noise is
   unmeasured. So "16 of 20" and "19 of 45" stopped right are single draws; on sheet 02 the fixed rule gave 24 of 45 and
   the tuned one-picture rule 25 of 45.
6. **Pictures are not independent.** Denominators are pictures, but consecutive pictures are strongly dependent (audit
   section 11.2, point 5: about 1,100 to 1,700 independent views among 18,522); per-sheet results rest on 20 to 45
   episodes. Episode-bootstrap intervals from the saved predictions would size the per-sheet differences.
7. **Literature for the main model also supports the backup.** The pretrained checker found no opened study with a
   clear winner between ImageNet fine-tuning and training from scratch at about 200 demonstrations (Chi Table 5: 0.94
   from scratch, 0.92 fine-tuned; Hansen Table 5: 65.0 against 63.1), and Chi's 95 % real Push-T model had no
   pretraining. The checker's own verdict: a sensible default, not what the evidence proves.
8. **Diffusion time per step and ACT training time are overstated.** The diffusion checker calls "seconds per step"
   overstated (the 5,244 ms is a transformer variant of unknown size; a right-sized head might take tens of ms,
   unchecked). No number supports "days of laptop training" for ACT (LeRobot's indicative 6 to 14 h for 5 passes over
   about 45,000 frames on Apple MPS).
9. Smaller points: the backup's literature claim is stitched (Hansen compared with frozen models and up to 100
   demonstrations; Chi's from-scratch network was a ResNet-18; fine-tuned R3M beat from-scratch by about 17 points); "led
   frozen features by 7.6 to 17.3" and "beat every frozen network by 5.1 to 15.6" are against the frozen ResNet-18 only
   (against the best frozen backbone on each of sheets 03 / 05 / 06: 7.6 / 7.1 / 10.6 and 4.8 / 4.7 / 8.9 points); "at most 0.19 s" from chunking needs a new executor (today
   about 0.02 s, A-code (f)) and ignores untested blending; "a picture history" in de Haan Table 1 is a history of
   positions; the 24-block robot comparison resolves only gaps of about 10 points or more (0.70 at 85 against 90 %),
   while the offline gap is about 2 points; 16.7 ms is the backbone alone (whole model 21.5 to 26.9 ms); "3 to 5 times
   slower" mixes picture sizes (3.2 to 3.5 at 224 x 224); "did no worse on 05 and 06" while the numbers are 2.5 and
   3.3 points lower; the cosine schedule forces the epoch-8 levelling; the VPN fact is not in [M]; the exam model's
   20-minute lower end pairs the fastest fold with the largest fold's ratio (about 25 to 65 minutes when consistent).

## Verdict

Stands with changes. The laptop and robot-loop fit is well measured. The choice of ResNet-18 over the small CNN is a
judgement the offline data do not yet support; present it so, and settle it with three seeds of both candidates in
final form on all six sheets, the baselines re-scored on the cleaned rows, and the professor's answer on ImageNet
weights.
