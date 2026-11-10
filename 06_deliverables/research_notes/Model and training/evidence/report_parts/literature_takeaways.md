# Literature takeaways (training-literature-check workflow, 9.11.26; each claim checked against its source)

Claim checks: offline_vs_closed_loop 48 (45 supported, 3 corrected); action_representation 40 (34, 6); action_chunking
49 (44, 5); diffusion_and_generative 44 (42, 2); pretrained_representations 57 (51, 6); augmentation 43 (40, 3);
history_and_causal_confusion 38 (33, 4, 1 could not open); interactive_correction 36 (34, 2); evaluation_statistics 42
(37, 5). Notes: scratchpad/literature/<key>/note.md and check/note.md.

1. Offline numbers against robot success. Validation loss is the weakest predictor (robomimic, Mandlekar et al. 2021,
   Table 29: the lowest-validation-loss checkpoint 57 to 100 percent below the best, the last checkpoint 12 to 35 percent
   below, per the checker). Direction-type measures correlate better (Codevilla et al. 2018: quantised classification
   error about 0.65 against 0.39 for MSE, unseen town). Validation pictures with recoveries raised the MSE correlation
   from 0.39 to 0.77 in one condition. Among good models and between seeds the offline numbers say little. So: shortlist
   offline on held-out sheets with direction measures, never pick on validation loss, fix the training length once and
   take the last checkpoint, then decide on the robot. Score robot runs also by how far the path strays from the line
   (Hussenot et al. 2021) and count stalls and dithering.
2. Classes against continuous outputs. From pictures, MSE was not worse than bins in the one controlled comparison
   (Pearce et al., ResNet-18: MSE 17.8, per-number bins 14.7, k-means plus residual 16.8). Big multimodality gains came
   with memory, states or many operators (robomimic with pictures, multi-operator Transport: 42.0 against 41.0 without
   the GMM). Two operators are closer to one operator. Pauses and idle actions made single-step policies stall (Diffusion
   Policy real Push-T: LSTM-GMM stuck in 8 of 20 runs; the authors call removing idle actions common practice).
3. Action chunks. ACT's gain from chunking (1 to 44 percent) was at 50 Hz in simulation with joint state; our 1 cm steps
   already give about 89 decisions per track. Real Push-T ran 6 of 16 predicted steps open loop at 95 percent. Under
   noise, executing long chunks blind lost (Liu et al. 2025). Supported plan: train with a 5 to 10 step target, execute
   one step per picture by default, try 2 or 3 per picture only for speed, on held-out tracks.
4. Diffusion and generative policies. Strong on Push-T but trained for hours on GPUs; with the same network and chunks,
   regression matched flow policies on most benchmarks (Pan et al. 2026). Not realistic on the laptop's processor now;
   a later GPU comparison only with the same encoder and a held-out sheet. Lin et al. 2025 (ICLR): generalisation to
   unseen environments grew roughly as a power law in the number of training environments, not demonstrations.
5. Pretrained backbones. With 100 to 200 demonstrations a ResNet-18 trained end to end with shift augmentation matched
   or beat frozen features (Chi et al. 2023, Square: 0.94 from scratch, 0.92 fine-tuned, 0.58 frozen; real Push-T 95
   percent with a ResNet-18 trained from scratch). Fully fine-tuning large ViTs on small data hurt. No study tested
   unseen track shapes. Candidates worth adding: a small CNN from scratch and a ResNet-18 cut after layer 3.
6. Augmentation. Small shifts or crops are the best supported (robomimic real Can 22 of 30 against 8 of 30 without);
   photometric changes guard against light and table changes. Mirrors with swapped labels: no measured robot benefit;
   safe only if the camera view is left-right symmetric and every label is mirrored; test first. Recorded recovery
   demonstrations are better supported than synthetic corrective views. Track variety matters (Xie et al.: the
   generalisation gap fell from about 0.4 to under 0.1 as training environments went from 5 to 100).
7. Picture history. History helped only where the current picture lacked needed state; otherwise it hurt closed-loop
   results while offline scores improved (Wen et al. 2020: lower test loss, less reward). Default: current picture only;
   test one extra frame only in closed loop, with a frozen-history copy test first (Chuang 2022). The crossing and the
   dashed gaps might need memory: check on the robot.
8. Corrections after the first model. Human-gated corrections helped on real robots (HG-DAgger, ThriftyDAgger, BC-Z),
   mostly 1 to 3 rounds; corrections recorded with the same tool keep the same labels; keep the correction share near 10
   to 25 percent of a batch, not 50; pilot 5 episodes to measure the time per correction (no source reports hours per
   round); keep at least one practice track for evaluation only.
9. Robot test statistics. Report counts with Wilson intervals; 10 trials cannot tell 70 from 90 percent (7/10 is 0.40 to
   0.89, 9/10 is 0.60 to 0.98); a formal test of 70 against 90 percent needs about 50 trials per model one-sided; choose
   between two candidates with about 24 paired, blind, interleaved trials from matched starts; run one pre-registered
   final test; on unseen tracks prefer many distinct tracks with 2 to 3 trials each (at least 10 tracks) over few tracks
   with many trials, because trials on one track share its difficulty; time among successful runs only is biased when
   finish rates differ, so use finish first, then time per paired block.
