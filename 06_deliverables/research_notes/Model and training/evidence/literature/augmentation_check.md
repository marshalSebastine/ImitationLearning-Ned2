# Check of "Augmentation that keeps labels right" (key: augmentation)

Sceptical check of 43 claims and of the conclusion, 9 November 2026. Plain language, British spelling.

## How I checked

- I opened every cited source myself. PDFs were fetched with WebFetch and their text was pulled out with
  pdftotext (the copy that ships with Git for Windows) into `sources/` in this folder; I read the numbers
  from that text, not from summaries. Where I relied on an automated summary of a web page, I say so.
- Versions opened: DrQ arXiv 2004.13649 v4 (7 Mar 2021, printed as a preprint under review, no venue);
  RAD arXiv 2004.14990 v5 (NeurIPS 2020 line printed); robomimic CoRL 2021 PMLR v164 PDF (main text only,
  no appendix); Diffusion Policy arXiv 2303.04137 v5 (14 Mar 2024); Hansen et al. arXiv 2212.05749 v2
  (ICML 2023, PMLR 202 printed); Xie et al. arXiv 2307.03659 v1 (no venue printed); Codevilla et al. arXiv
  1710.02410 v2; Bojarski et al. arXiv 1604.07316 v1; Pomerleau thesis CMU-CS-92-115 (DTIC AD-A249 972,
  scanned with an OCR text layer); Giusti et al. RA-L accepted preprint (DOI 10.1109/LRA.2015.2509024);
  Smolyanskiy et al. arXiv 1705.02550 v3; Young et al. CoRL 2020 PMLR v155 PDF; Zhou et al. arXiv
  2301.08556 v1; Zeng et al. arXiv 2010.14406 v3 (CoRL 2020 printed); Quartey and Korsah arXiv 2006.11670
  (DOI printed); Wang et al. arXiv 2211.09231 v2 abstract page plus the ar5iv rendering.
- Small-sample tests (two-sided Fisher exact test) and percentages are my own arithmetic
  (`tools/small_sample_checks.py`). The project frame was measured with OpenCV colour thresholds
  (`tools/measure_frame.py`), read-only.

Summary: 40 supported (several with a caveat), 3 corrected (7, 35, 43), none unsupported, none unopened.

## Verdict per claim

Status "checked" means I read it in the source. "Measured" versus "authors' view" follows the paper.

### DrQ: Kostrikov, Yarats, Fergus (2020), arXiv 2004.13649 v4, https://arxiv.org/pdf/2004.13649

1. **Supported** (checked, definition). Section 3.1: pad each side by 4 px repeating border pixels, then a
   random 84 x 84 crop, a shift of up to plus or minus 4 px (4/84 = 4.8%). Applied only to pictures drawn from
   the replay buffer. "Actions unchanged" is not written; it follows from the Section 3.2 definition, where
   the same action appears on both sides.
2. **Supported** (checked, definition). Section 3.2: a transformation that preserves Q for every state,
   action and transformation parameter; random image translations are the example. The imitation analogue
   is correctly marked as the researcher's own.
3. **Supported** (checked, measured, simulated RL). Section 3 and Figure 1: SAC from pixels on Cartpole
   Swingup, Finger Spin and Walker Walk with five encoders (0.22M to 2.44M parameters). Without augmentation,
   bigger encoders did worse; with shifts all sizes were similar and clearly better, most of all on Walker Walk.
4. **Supported** (checked, measured). Appendix E and Figure 6: shift, cutout, horizontal and vertical flip
   (probability 0.1), rotation (text prints the range as [-5, -5]; the code uses RandomRotation(degrees=5.0)),
   intensity. Figure 6 is learning curves only (six DMC tasks); the caption calls shifts the most effective.
   A second text/code mismatch: intensity sigma 0.1 in the text, scale 0.05 in the code.
5. **Supported** (checked, authors' argument). Section 3.1 says flips and rotations keep object labels, but
   "in many cases the reward would not be preserved by these transformations". Appendix E then does measure
   flips and rotations, but only as curves.

### RAD: Laskin, Lee, Stooke, Pinto, Abbeel, Srinivas (2020), NeurIPS 2020, https://arxiv.org/pdf/2004.14990

6. **Supported, with a caveat** (checked, definition). Section 4 and Figure 1 give exactly these definitions
   (crop 100 x 100 to 84 x 84 in DMControl; translate 100 x 100 inside 108 x 108; flip across the vertical axis;
   rotate by 0, 90, 180 or 270 degrees). RAD never says the action is unchanged; that is implied because only
   observations are augmented.
7. **Corrected** (checked, measured). The six numbers are exactly the diagonal of Figure 2(a) (Walker Walk,
   500k steps: crop 920, rotate 268, flip 223, cutout 215, colour jitter 194, grayscale 175). Section 5.2 says
   SAC without augmentation scores well below 200; Table 1 gives pixel SAC 42 plus or minus 12. But flip and
   rotate were the 2nd and 3rd best single augmentations, not "among the weakest"; all five non-crop singles
   (175 to 268) were far below crop. Off-diagonal cells show that adding them to crop hurt: crop then flip 650,
   crop then rotate 635, flip then crop 828, rotate then crop 604 (crop alone 920). The number of runs per
   cell is not stated. RAD does not discuss actions.
8. **Supported** (checked; attention maps are qualitative). Section 5.2 with Appendix C (Figure 6: translate
   helps, window has little effect; Figure 5: in some cases two extra pixels suffice) and Figure 4: without
   augmentation the activation sits on the forward knee, with crop or translate on the body edges,
   "providing a more complete and robust representation of the state" (the authors' reading).
9. **Supported** (checked, measured). Section 5.3 and Table 2: 55.8% is the 200-training-level BigFish row
   (4.3 to 6.7, test on unseen levels, 3 runs); with 100 levels crop went 1.9 to 5.4. Figure 3(b): random
   convolution, colour jitter and cutout-colour beat crop on CoinRun with unseen themes. Not in the claim but
   relevant: on BigFish at 200 levels, flip (3.5) and rotate (1.5) were below plain PPO (4.3).

### robomimic: Mandlekar et al. (2021), CoRL 2021, https://proceedings.mlr.press/v164/mandlekar22a/mandlekar22a.pdf

10. **Supported** (checked, measured, simulation). Section 4.3 and Figure 2a: removing pixel-shift
    randomisation gave 47% (Square) and 35% (Transport) relative drops for image-based BC-RNN. These agents see
    an agent-view camera and a wrist camera plus proprioception, not one camera.
11. **Supported** (checked, measured). Section 4.7: Franka, 200 demonstrations from one operator, final
    checkpoint, 30 rollouts, no real-world tuning: Can 73.3%, without randomisation 26.7%, without wrist camera
    43.3%. 22/30 against 8/30 gives p about 0.0007 (my arithmetic), so this result is solid.
12. **Supported** (checked, authors' explanation). Section 4.3, last sentence: randomisation helps the policy
    become invariant to picture parts that do not matter for the action. Not measured.

### Diffusion Policy: Chi et al. (2023), RSS 2023, extended arXiv v5, https://arxiv.org/pdf/2303.04137

13. **Supported** (checked, definition). Appendix A.3: random crop in training following Mandlekar et al.,
    a fixed centre crop of the same size at inference. Table 7: 84 to 76 (Lift, Can, Square; Transport printed
    as 84 x 85 to 76 x 76), 96 to 84 (Push-T), 240 to 216 (Tool Hang), 320 x 240 to 288 x 216 (real tasks).
    Largest shift 4.8% to 6.3% of the width (my arithmetic). The word crop appears nowhere else in v5, so there
    is no crop ablation.

### Hansen et al. (2023), ICML 2023, https://arxiv.org/pdf/2212.05749

14. **Supported** (checked, measured). Table 2, Section 3, Appendix B.1: all numbers match. Shift padding is
    12 on 256 x 256 frame stacks. Simulation used 100 oracle (not human) demonstrations and reports the mean of
    the best 3 epochs, which flatters every method. The real xArm used a fixed third-person camera, 10 (reach)
    and 20 (pick) human demonstrations, 20 trials per method across 2 seeds. Pick 11/20 against 5/20 gives
    p about 0.11 and Reach 17/20 against 16/20 shows no difference (my arithmetic).
15. **Supported** (checked, measured, simulation). Section 4 and Table 3 (behaviour cloning policies): colour
    jitter 53.1 (random colours) and 33.0 (video backgrounds); random overlay 39.3 and 46.6; best frozen
    pre-trained representation 37.2 and 9.2.

### Xie, Lee, Xiao, Finn (2023), arXiv 2307.03659 v1, https://arxiv.org/pdf/2307.03659

16. **Supported, with a caveat** (checked, measured). Section 5.1, Figure 6 (policy with both augmentations) and Appendix
    A.2.1 (random 300 x 300 crops; brightness, contrast, saturation): 91.7, 88.9, 83.3, 80.6, 52.8 and 45.8
    all match. The paper says 12 trials per factor (2 per task, 6 tasks), but 88.9 = 32/36 and 45.8 = 11/24, so
    the factors with 2 or 3 new values pool 24 or 36 trials (12 per value). The policy is RT-1 trained on over
    115,000 demonstrations with 6 frames of history, far more data than ours.
17. **Supported** (checked, measured). Section 5.2: on the robot, crop helped most for new camera positions and
    new table textures; photometric distortion helped table texture on the robot and object, table and
    background textures in Factor World.
18. **Supported** (checked, measured, simulation). Section 5.1 (continuous factors) and Figure 5b: radii
    0.025, 0.050 and 0.075 m; for camera and table position the gap roughly doubles as the radius grows.
    Demonstrations come from scripted experts.

Venue: search results list ICRA 2024 and a NeurIPS 2023 workshop poster (not opened).

### Codevilla et al. (2018), ICRA 2018, https://arxiv.org/pdf/1710.02410

19. **Supported, with a caveat** (checked, definition). Section IV-D lists exactly these transformations with
    random subsets and magnitudes, and: "No geometric augmentations such as translation or rotation were
    applied". Caveat: region dropout is occlusion, not a photometric change; geometric variety came instead from
    two real side cameras with adjusted steering (Section V).
20. **Supported** (checked, measured, simulation). Table 1, Section VI-A: 50 start-goal pairs per town;
    full model 88% and 64%; without augmentation 80% and 0%.
21. **Supported** (checked, measured). Table 2 and caption, Section VI-B: tests in overcast weather, training
    mostly sunny; numbers match; the no-augmentation model ran once (to avoid breaking the truck), the others
    3 times.
22. **Supported, with a caveat** (checked, measured). Without noise-injected data: 56% and 22% (Table 1),
    truck interventions 0.67 to 8.67 (Table 2); side cameras at 30 degrees with adjusted steering (Section V,
    Figure 7). The 12 of about 120 minutes refers to the CARLA data only; the truck's share is not given.

### Bojarski et al. (2016), NVIDIA report, https://arxiv.org/pdf/1604.07316

23. **Supported** (checked, definition). Sections 2 and 5.2, as stated. The paper calls the system DAVE-2;
    PilotNet is a later name.
24. **Supported** (checked, authors' statement). Section 2 (distortions of cars, poles, trees, buildings said
    not to matter much) and Section 5.2 (artefacts grow with magnitude). Nothing measured.
25. **Supported** (checked, measured). Section 7.2: about 98% autonomy on a typical drive (lane changes and
    turns excluded) and 10 highway miles without intervention. No ablation, and no mirrored pictures anywhere.

### Pomerleau (1992), PhD thesis CMU-CS-92-115, https://rosap.ntl.bts.gov/view/dot/42152/dot_42152_DS1.pdf

26. **Supported** (checked, definition). Sections 3.2.2 to 3.2.4: as stated (known sensor pose, planar world,
    pure pursuit at 2 to 3 seconds of travel, 14 transformed copies, plus or minus 0.6 m and 6 degrees,
    transforms needing a sharper turn than representable, usually 30 m radius, are redrawn).
27. **Supported, with a caveat** (checked). Section 3.2.2, Figures 3.6 to 3.8: the nearest-point fill smears the
    lane line, the smear length correlates with the correct steering and the network learns to use it. The
    measured part is offline: 37% less steering error on 100 test images (3.5 against 2.2 output units). That
    the network then drives poorly is stated, not shown.
28. **Supported, with a caveat** (checked, measured). Section 3.3, Figure 3.11: 100 m bike path at 4 mph,
    trained on a separate 150 m stretch; numbers match; one drive each. The author blames the first network's
    failure on overlearning the final right turn, so transforms and left-right balance are mixed up in this
    comparison.

### Giusti et al. (2016), IEEE RA-L, https://rpg.ifi.uzh.ch/docs/RAL16_Giusti.pdf

29. **Supported** (checked, definition). Sections II, III-A, III-B, as stated. That the affine changes keep
    the label is implied, not written.
30. **Supported, with a caveat** (checked, measured). Table I: 85.2% on 7,355 test frames; the humans scored
    86.5% and 82.0% on 200 random test frames each. No result without augmentation. Section IV: the quadrotor
    corrected a sideways shift only when about one metre off the centreline.

### Smolyanskiy et al. (2017), arXiv 1705.02550 v3, https://arxiv.org/pdf/1705.02550

31. **Supported** (checked, definition). Section IV-A, as stated (cameras 0.5 m apart on a 1 m bar).
32. **Supported, with a caveat** (checked, measured). Table I: autonomy 80% (their re-implementation of
    Giusti's 3-class net), 88% (6-class ResNet-18), 100% (TrailNet); accuracy 79%, 92%, 84%; mini AlexNet 97%
    and SqueezeNet 98% autonomy. ResNet-18 and TrailNet share the 6-class labels and differ in loss and
    activations; the clean 3-class against 6-class contrast is only the qualitative disturbance test
    (Figure 9). The arXiv comments call it an IROS 2017 conference submission (acceptance not shown there).

### Young et al. (2020), CoRL 2020, https://proceedings.mlr.press/v155/young21a/young21a.pdf

33. **Supported** (checked, definition). Section 3.3 lists exactly these five augmentations; reflection
    mirrors the action labels; the labels are COLMAP relative motions between frames (Section 3.2). That
    rotation keeps the labels is implied.
34. **Supported, with a caveat** (checked, measured). Table 1: pushing 0.625 to 0.875, stacking 0.291 (29.2%
    in the text; 7/24 = 29.17%) to 0.625; half the data with augmentation 0.750 and 0.416. Pushing was tested on
    24 unseen objects, stacking on 24 unseen configurations. Pushing 21/24 against 15/24 gives p about 0.09
    (not significant at 5%); stacking 15/24 against 7/24 gives p about 0.04 (my arithmetic).
35. **Corrected** (checked, measured). Section 4.4 says "the best augmentation performs 0.7% better than naive
    behavioral cloning in pushing", but Figure 6 shows this is an absolute difference in MSE: pushing 0.0381 to
    0.0315 (17% lower), stacking 0.0695 to 0.0599 (14% lower). The same paper reports 0.081 against 0.067 as a 1.4%
    difference, confirming the convention. So the offline gain was not tiny. Mirroring: colour jitter plus reflection was the second-best
    stacking pair (0.0614); reflection plus cutout-colour ranked 4th of 15 for pushing (0.0341); reflection on
    its own was weak (0.0376 pushing, 0.0670 stacking). Cutout-colour on its own was worst and worse than none
    (0.0398, 0.0749), though rotation plus cutout-colour was the second-best pushing pair (0.0334). The robot
    changes of 25 and 33.4 points are right.

### SPARTN: Zhou, Kim, Wang, Florence, Finn (2023), CVPR 2023, https://arxiv.org/pdf/2301.08556

Venue: the CVF open-access PDF appeared in search results (not opened).

36. **Supported** (checked, definition). Sections 3.3, 3.4, 4.3, 5.1, as stated. Extra details: noise is
    uniform within those bounds; simulation augments only time steps 5 to 13; 100 (simulation) and 50 (real)
    augmented copies per frame; the real robot also took extra pictures from fixed poses before each
    demonstration to help COLMAP; frames are used only before the gripper closes.
37. **Supported** (checked, measured). Table 2 (4 seeds) and Table 4 (8 objects, 20 trials each). Real
    totals 98/160 against 62/160 give p about 0.0001 (my arithmetic). Note that the rotation-only homography
    with corrected actions gave 29.7% against 28.9% for plain cloning: essentially no gain.
38. **Supported** (checked, measured). Table 3 and Section 4.4, as stated; the authors call the effect small
    next to SPARTN's corrective augmentation.
39. **Supported** (checked, authors' statement). Section 6, as stated. The same section adds that training a
    NeRF for every demonstration is computationally expensive.

### Transporter Networks: Zeng et al. (2020), CoRL 2020, https://arxiv.org/pdf/2010.14406

40. **Supported, with caveats** (checked). Appendix 6.6: a random rotation and shift in the table plane,
    observations adjusted as if seen from the new frame, assuming absolute table position does not matter.
    Section 3 says spatially consistent (orthographic) views make augmentation simpler because rigid changes
    look like physical rearrangement, in contrast to perspective images. Caveats: that the pick and place labels
    move with the frame is implied, not written; "only in orthographic pictures" is stronger than the text; the
    authors do state that augmentation substantially helps generalisation, but give no numbers.

### Quartey and Korsah (2018), IEEE ICAST 2018, https://arxiv.org/pdf/2006.11670

41. **Supported** (checked, single observation). Learning-layer section and evaluation section, Figure 9:
    horizontal flips with negated steering plus random shadows; the car drifted left onto the lawn after a
    couple of metres; the authors blame left-favoured steering in the data that the flips only partly balanced.
    3,147 pictures, one run, no comparison.

### Wang et al. (2023), ICLR 2023, https://arxiv.org/abs/2211.09231v2

42. **Supported at abstract level.** The abstract says imposing incorrect symmetry can impede performance and
    imposing extrinsic symmetry can improve it, in domains with latent symmetry. An automated reading of the
    ar5iv page (https://ar5iv.labs.arxiv.org/html/2211.09231, not read in full by me, so partly checked)
    reports: Section 4.2 defines incorrect (labels do not transform as the model assumes) and extrinsic (the
    transformation sends inputs outside the data distribution); Section 6.3 found random reflections that the
    model treats incorrectly made equivariant SAC barely learn. The paper is about architectures; the link to
    mirrored training pictures stays an interpretation.

### Project frame (not literature)

43. **Corrected** (checked by viewing and by measurement). The gripper fingers, magnet block and red puck are
    at the bottom centre, as stated. Measured with colour thresholds: the puck is 60 px wide (x 262 to 321), not
    about 65, which is about 25 px per cm at the puck's top. The line is about 21 to 22 px wide near the puck
    and also near the top (ridge medians 22.0, 21.2, 22.4 and 22.0 px in four regions), so about 22 px per cm
    on the sheet and no clear narrowing towards the top in this frame. Still estimates from one frame. By eye,
    the tool's centre is 5 to 10 px left of the picture centre.

## Challenge to the conclusion

What holds up:

- Small whole-picture shifts are the best-supported augmentation for behaviour cloning at our data size.
  robomimic's real Can result (22/30 against 8/30, 200 demonstrations) is statistically solid, and the
  simulation drops (47% and 35%) agree. Shift sizes used in the papers range from 3.3% (Young et al.) to 6.3%
  (Diffusion Policy, Push-T); about 5% is in the normal range. With my measurement (about 22 px per cm on the
  sheet), 5% of 640 x 480 is about 32 by 24 px, roughly 1.5 by 1.1 cm.
- Shifts are cheap on a laptop CPU, and DrQ-style padding makes a border smear that does not depend on the
  label, so Pomerleau's shortcut does not arise.

What is weaker than stated:

1. "Offline loss barely predicted robot success (Young et al.)" rests on a misreading: their 0.7% and 0.9%
   are absolute MSE differences, about 17% and 14% relative. Better evidence for the point: robomimic Section
   4.5 (choosing the checkpoint by lowest validation loss gave 10% to 100% worse success than the best
   checkpoint) and Smolyanskiy Table I (accuracy order differs from autonomy order).
2. The photometric evidence (Codevilla 0% against 64%) is about an unseen town's look, with all augmentation,
   including region dropout, removed at once. The exam's unseen tracks change shape, not look. Photometric
   jitter guards against lighting and table changes (real for us: new table on 6.11), not against new curves.
3. Mirrors: no paper measured robot success with mirrored, label-swapped pictures. Young et al.'s offline data
   give mild support (jitter plus reflection was the second-best stacking pair). RAD shows that a
   transformation the labels do not follow hurts (crop then flip 650 against crop 920; flip below plain PPO on
   BigFish at 200 levels), and Wang et al. report that an incorrect reflection symmetry can stop learning. So
   mirrors are safe only if the camera sits on the arm's plane of symmetry and every label is mirrored
   consistently: move names, the left part of the step, forearm twist and hand turn. Test before training:
   mirror a held-out episode and check that the model's predictions mirror. A further risk (my reasoning, not
   measured): if anything in the picture reveals that a frame is mirrored, and the original data favour one
   side, the network can learn the original bias back, which may be what happened to Quartey and Korsah's car.
   A point in favour that the conclusion misses: mirrors double the number of track shapes (12 instead of 6),
   the one listed augmentation that targets unseen tracks.
4. Corrective homography views: the closest test (SPARTN's rotation-only homography with corrected actions)
   gave no gain (29.7% against 28.9%); only NeRF views with translations helped, and those need a still scene,
   per-demonstration NeRF training and a GPU-scale budget. ALVINN's ground-plane transforms did help, but its
   baseline also failed through left-right bias. In our task the pushed puck moves and both puck and tool
   stand above the sheet, so the flat, still-scene assumption fails. Rotated or homography views also leave
   filled areas whose size grows with the label change, which is the shortcut Pomerleau measured; crop them
   away. The better-supported route is real recovery demonstrations: guide the arm back from a puck placed
   beside the line. Codevilla's 12 minutes of recovery data (out of 2 hours) moved simulation success from
   56%/22% to 88%/64% and cut truck interventions from 8.67 to 0.67, and stop-and-go hand guiding makes such
   recordings cheap.
5. "Tool, puck and line are all in view" rests on one frame. The camera is on the forearm, so the tool's place
   in the picture changes with the wrist joints, and the gripper fingers sit at the bottom edge: a 24 px upward
   or downward shift can cut them off or smear them. Check, across all 18,500 frames, how often the puck or
   tool lies within 32 px of a border before choosing the shift size.
6. No opened paper uses a camera on an arm link whose view of the tool changes; robomimic, SPARTN and Young et
   al. use wrist or tool cameras with the gripper fixed in view, Hansen et al. a fixed camera. The transfer of
   their numbers to our mount is an assumption.
7. Several cited robot gains are small-sample: Young et al. pushing p about 0.09 and Hansen et al. Pick p
   about 0.11. Our own exam-style tests with 10 to 20 runs per setting will be just as noisy, so compare
   augmentation settings on many runs or on offline checks that track success.

Overall: shifts of about 5% plus mild photometric jitter are well supported and cheap. Mirrors are plausible
but must first pass a symmetry test on this mount. Synthetic corrective views are the weakest part; recorded
recovery demonstrations should come first.

## Evidence the researcher missed

- robomimic Section 4.5: selecting by lowest validation loss gave 10% to 100% worse success; Section 4.4:
  a shallow convolutional encoder lost 25% to 62% relative to ResNet-18 (relevant to a CPU-sized model).
- RAD Figure 2(a) off-diagonal cells and Table 2 flip and rotate rows (see claims 7 and 9).
- SPARTN Table 2: the homography baseline gave no gain; Section 6: per-demonstration NeRF cost.
- Codevilla Tables 1 and 2: recovery data was as important as augmentation; not carried into the judgement.
- Pomerleau Section 3.3: left-right balance of the training buffer mattered (28.9 cm to 2.7 cm mean offset);
  balance our left and right move names.
- Xie et al.: camera pose was the hardest real factor (45.8%) and crop helped it most; our camera is taped and
  was moved once; table texture (52.8%) matters after the move to a new table; in Factor World the gap fell
  from about 0.4 to under 0.1 as training environments went from 5 to 100 (diversity of tracks matters).
- Not opened, so not evidence yet: DART (Laskey et al. 2017), DAgger (Ross et al. 2011), Kitchenshift (Xing
  et al. 2021), DrAC (Raileanu et al. 2021), DrQ-v2 (Yarats et al. 2021), and equivariant policies with
  rotated actions. No opened paper tests generalisation to new path shapes of the same appearance, which is
  what the exam asks.
