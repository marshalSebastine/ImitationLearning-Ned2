# Augmentation that keeps labels right

Key: `augmentation`. Written 9.11.26 by a Claude research sub-agent for Team M (Emranul Hoque and Marshal Sebastine),
AI Project, Niryo Ned2 pushing a red puck along a black line, one forearm camera as the only model input.

Question (as given to me): data augmentation for visuomotor behaviour cloning (random shifts and crops, colour jitter,
blur and noise, and label-changing augmentation such as PilotNet's shifted and rotated views with corrected steering, or
flipped pictures with negated steering). When does a geometric augmentation break the label of a line-following or
pushing policy, and when can the label be corrected? Give measured effects where papers report them.

How to read the markers in this note:

- **[checked]** I read it in the source myself. **[unchecked]** I did not read it in the body of the source (for example
  only the abstract).
- **[measured]** the paper measured it. **[authors' view]** the authors explain or guess, without a measurement.
  **[definition]** how a method works. **[my judgement]** and **[calculated]** are my own reasoning or arithmetic, not
  from a paper.
- Quotes are at most 15 words and at most one per source. Everything else is paraphrased.
- The claim ids (augmentation-1 to augmentation-43) match the structured result returned to the orchestrator.

---

## 1. Short answer in plain language

1. An augmentation "keeps the label right" when the demonstrator would have chosen the same move in the changed
   picture. Changes of brightness, contrast, colour, blur and noise nearly always do. Geometric changes (shift, rotate,
   mirror, zoom) keep it only if the move does not depend on what the change alters.
2. Which physical change a picture change imitates decides this. For a forward-looking driving camera, a sideways
   shift of the picture looks like turning the car, so the steering label must change (PilotNet and ALVINN correct
   it; Codevilla et al. refused geometric changes for this reason). For our camera the tool (block and gripper), the
   puck and the line are all in the picture and the move names are given in the picture's own frame, so [my judgement]:
   - a small shift of the whole picture moves tool, puck and line together: the move name stays right;
   - a rotation of the whole picture turns the needed direction by the same angle: near the borders between the eight
     directions the name becomes wrong unless it is rotated too;
   - a left-right mirror turns left into right: the name is right only after swapping the left and right names;
   - an upside-down flip or a 90 degree turn puts the tool at the top of the picture: no correction makes sense.
3. Papers measured large gains from small shifts or crops in behaviour cloning (robomimic, Hansen et al. 2023), large
   gains from colour, blur and noise changes when lighting or scenery change at test time (Codevilla et al. 2018), and
   large gains from viewpoint changes with corrected labels (Pomerleau's ALVINN, SPARTN). Mirrors with swapped labels
   are widely used (trail following, pushing), but no paper I opened measured their effect on robot success.
4. A measured trap (Pomerleau 1992): when a shifted picture's filled-in border looks different for each shift and the
   label is corrected for that shift, the network learns to read the border instead of the road.

---

## 2. Words used in this note

- **Label**: the move the demonstrator made from this picture (our move name, or the exact step).
- **Label-preserving augmentation**: the changed picture still deserves the same label. DrQ formalises this for
  reinforcement learning as a transformation that leaves the value of every action unchanged (augmentation-2). For imitation the analogue is that the expert's action is unchanged [my analogy].
- **Label-correcting augmentation**: the changed picture deserves a different label, and a model (a mirror rule, a
  steering law, the robot's geometry) tells us which.
- **Shortcut**: a cue that exists only in augmented training pictures but predicts their label, so the network learns
  the cue instead of the task (augmentation-27).

---

## 3. What the papers did and measured

### 3.1 Shifts and crops

**Kostrikov, Yarats, Fergus (2020). Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from
Pixels (DrQ).** arXiv 2004.13649 v4; the PDF and arXiv record I opened name no venue (it is commonly cited as ICLR
2021, not checked). Opened: https://arxiv.org/pdf/2004.13649 and https://arxiv.org/abs/2004.13649. Setting:
reinforcement learning in simulation; actions are never changed by the augmentation.

- augmentation-1 [definition, checked] Section 3.1: 84 x 84 pictures are padded by 4 pixels (border pixels repeated)
  and a random 84 x 84 crop is taken, a shift of up to plus or minus 4 pixels (about 5 percent of the width).
- augmentation-2 [definition, checked] Section 3.2: an "optimality invariant" transformation keeps the value of every
  action; random translations are their example.
- augmentation-3 [measured, checked] Figure 1: without augmentation, SAC from pixels got worse as the image encoder got
  bigger (over-fitting); with random shifts all encoder sizes performed alike and better.
- augmentation-4 [measured, checked] Appendix E, Figure 6: random shift, cutout, horizontal or vertical flip
  (probability 0.1), rotation (plus or minus 5 degrees in the code listing; the text's range is misprinted) and
  intensity scaling were compared; random shift was the most effective on DeepMind Control. Curves only, no table.
- augmentation-5 [authors' view, checked] Section 3.1: flips and rotations keep labels in object recognition, but in
  control tasks they would often not preserve the reward.

**Laskin, Lee, Stooke, Pinto, Abbeel, Srinivas (2020). Reinforcement Learning with Augmented Data (RAD).** NeurIPS 2020
(printed on the PDF). Opened: https://arxiv.org/pdf/2004.14990 (v5) and https://arxiv.org/abs/2004.14990. Setting:
reinforcement learning; actions unchanged.

- augmentation-6 [definition, checked] Section 4: crop takes a random patch (100 x 100 renders cropped to 84 x 84 in
  DeepMind Control); translate places the whole picture at a random place in a larger empty frame (for example 100 x
  100 inside 108 x 108); flip mirrors across the vertical axis; rotate picks 0, 90, 180 or 270 degrees.
- augmentation-7 [measured, checked] Section 5.2, Figure 2(a), Walker Walk after 500k steps, one augmentation each:
  crop 920, rotate 268, flip 223, cutout 215, colour jitter 194, grayscale 175; plain SAC scores well below 200.
- augmentation-8 [measured, checked] Section 5.2 with Appendix C (Figures 5 and 6, summarised in the main text): the
  gain of crop comes from translation, not from hiding the border; in some cases a random placement within only two
  extra pixels was enough. Figure 4 shows attention spread over the whole body with crop or translate, and on a single
  point (the knee) without; the authors read this as a more robust representation [authors' view].
- augmentation-9 [measured, checked] Section 5.3: crop gave a 55.8 percent relative gain over plain PPO on ProcGen
  BigFish; on CoinRun with unseen visual themes (Figure 3b), random convolution, colour jitter and coloured cutout
  generalised better than crop.

**Mandlekar, Xu, Wong, Nasiriany, Wang, Kulkarni, Fei-Fei, Savarese, Zhu, Martín-Martín (2021). What Matters in
Learning from Offline Human Demonstrations for Robot Manipulation (robomimic).** CoRL 2021 (PMLR 164, printed on the
PDF). Opened: https://proceedings.mlr.press/v164/mandlekar22a/mandlekar22a.pdf. Setting: behaviour cloning from human
teleoperation, simulation and a real Franka arm.

- augmentation-10 [measured, checked] Section 4.3, Figure 2a: removing pixel-shift image randomisation (random shifts
  of the picture) from the image-based BC-RNN agent lowered success by 47 percent (Square) and 35 percent (Transport),
  relative.
- augmentation-11 [measured, checked] Section 4.7: real Can task, 200 demonstrations, 30 trials: 73.3 percent with
  randomisation, 26.7 percent without (43.3 percent without the wrist camera).
- augmentation-12 [authors' view, checked] Section 4.3: randomisation probably helps the policy ignore picture parts
  that do not matter for the action.

**Chi, Xu, Feng, Cousineau, Du, Burchfiel, Tedrake, Song (2023). Diffusion Policy: Visuomotor Policy Learning via
Action Diffusion.** RSS 2023 (the extended arXiv v5 I opened cites the RSS version). Opened:
https://arxiv.org/pdf/2303.04137 (v5, 14 March 2024).

- augmentation-13 [definition, checked] Appendix A.3 and Table 7: random crop during training, following Mandlekar et
  al. 2021, and a fixed centre crop of the same size at test time. Sizes: 84 x 84 to 76 x 76 (robomimic tasks), 96 x 96
  to 84 x 84 (Push-T), 240 x 240 to 216 x 216 (Tool Hang), 320 x 240 to 288 x 216 (all real tasks). That is a shift
  range of roughly plus or minus 5 to 6 percent of the picture [calculated]. I found no ablation of the crop.

**Hansen, Yuan, Ze, Mu, Rajeswaran, Su, Xu, Wang (2023). On Pre-Training for Visuo-Motor Control: Revisiting a
Learning-from-Scratch Baseline.** ICML 2023 (PMLR 202, printed on the PDF). Opened: https://arxiv.org/pdf/2212.05749
(v2).

- augmentation-14 [measured, checked] Table 2 and Appendix B.1: behaviour cloning with a small convolutional network,
  with and without random shift (padding 12 on 256 x 256 pictures, about plus or minus 4.7 percent [calculated]).
  Adroit Pen 22.0 to 74.8 percent success, Relocate 16.9 to 51.4 percent (100 demonstrations each); real xArm Reach 80
  to 85 percent and Pick 25 to 55 percent (10 to 20 demonstrations, 20 trials per method and task over 2 seeds);
  DeepMind Control mixed (Reacher Easy return 261.3 to 657.4, Cheetah Run 469.8 to 448.9). Section 3: the authors
  call augmentation underused in behaviour cloning but surprisingly effective [authors' view].
- augmentation-15 [measured, checked] Section 4 and Table 3: on the DeepMind Control generalisation benchmark, colour
  jitter added to shifts gave a normalised return of 53.1 under random colours (best frozen pre-trained network 37.2);
  random overlay did better on video backgrounds (46.6 against 33.0 for colour jitter).

**Xie, Lee, Xiao, Finn (2023). Decomposing the Generalization Gap in Imitation Learning for Visual Robotic
Manipulation.** arXiv preprint 2307.03659 v1; the arXiv record names no venue. Opened:
https://arxiv.org/pdf/2307.03659 and https://arxiv.org/abs/2307.03659.

- augmentation-16 [measured, checked] Section 5.1 (Figure 6 caption says these use both augmentations): the real
  robot policy (RT-1, trained with random crops and brightness, contrast and saturation changes, Appendix A.2) reached
  91.7 percent on the training scenes; 88.9 with new backgrounds, 83.3 with new lighting, 80.6 with new distractors,
  52.8 with new table textures and 45.8 with a moved camera (12 trials per factor).
- augmentation-17 [measured, checked] Section 5.2, Figure 6: crop helped most for the moved camera and new table
  textures; photometric changes helped texture factors.
- augmentation-18 [measured, checked] Section 5.1, Figure 5b (simulation): as the range of camera or table positions
  grows linearly (radius 0.025, 0.050, 0.075 m), the generalisation gap roughly doubles.

### 3.2 Colour, brightness, blur and noise

**Codevilla, Müller, López, Koltun, Dosovitskiy (2018). End-to-end Driving via Conditional Imitation Learning.** ICRA
2018 (venue from my knowledge; the arXiv v2 PDF I opened does not print it). Opened: https://arxiv.org/pdf/1710.02410.
Setting: behaviour cloning of steering and acceleration from one front camera, in the CARLA simulator and on a 1/5
scale truck.

- augmentation-19 [definition, checked] Section IV-D: a random subset of contrast, brightness and tone changes,
  Gaussian blur, Gaussian noise, salt-and-pepper noise and region dropout (rectangles of about 1 percent of the
  picture), with random strengths, applied online. Translation and rotation were not used because, in their words,
  "control commands are not invariant to these transformations".
- augmentation-20 [measured, checked] Table 1 and Section VI-A (50 start and goal pairs per town): full model 88
  percent success in the training town and 64 percent in the unseen town; without augmentation 80 and 0 percent.
- augmentation-21 [measured, checked] Table 2 (truck, 14 intersections, trained mostly in sunny weather, tested
  overcast): full model 0 percent missed turns, 0.67 interventions, 2:19 minutes; without augmentation 73 percent
  missed turns, 39 interventions, 10:41 minutes (that model ran once only, to avoid breaking the truck).
- augmentation-22 [measured, checked] Section IV-C, Tables 1 and 2: recovery data recorded while noise was injected
  into the steering (12 minutes of 2 hours) also mattered: without it 56 and 22 percent success in simulation, and 8.67
  instead of 0.67 truck interventions. Section V: side cameras turned 30 degrees were used with adjusted labels.

Hansen et al. 2023 (augmentation-15), RAD (augmentation-9), Xie et al. 2023 (augmentation-16, -17), Young et al. 2020
(augmentation-33, -34) and SPARTN (augmentation-38) add further colour results. No paper I opened reports blur or noise
on their own; they are always part of a bundle.

### 3.3 Mirrored pictures with swapped labels

**Giusti, Guzzi, Cireşan, He, Rodríguez, Fontana, Faessler, Forster, Schmidhuber, Di Caro, Scaramuzza, Gambardella
(2016). A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots.** IEEE Robotics and
Automation Letters 1(2) (accepted preprint opened). Opened: https://rpg.ifi.uzh.ch/docs/RAL16_Giusti.pdf.

- augmentation-29 [definition, checked] Sections II, III-A and III-B: three classes from the angle between camera
  heading and trail direction (turn left, go straight, turn right; border at 15 degrees), labelled by which of three
  head-mounted cameras (30 degrees left, straight, 30 degrees right) took the picture. Every training picture is
  mirrored: a mirrored turn-right picture becomes turn-left and the reverse, go-straight stays. Mild affine changes
  (plus or minus 10 percent shift, 15 degrees rotation, 10 percent scaling) are applied without changing the label.
- augmentation-30 [measured, checked] Table I: 85.2 percent accuracy on 7,355 test frames from unseen trail sections
  (two humans: 86.5 and 82.0 percent); no result without augmentation. Section IV: on the quadrotor, with the heading
  roughly right, the network corrected a sideways offset only when about one metre off the centreline.
- [my judgement] Their rotation keeps the label because rotating a forward-looking picture in its own plane imitates a
  camera roll, which does not change the heading. Their 10 percent shift, on a camera with about 120 degrees field of
  view, imitates a turn of about 12 degrees [calculated], which stays on the same side of the 15 degree border for
  pictures taken at 0 or 30 degrees. Coarse classes tolerate small label-changing shifts.

**Smolyanskiy, Kamenev, Smith, Birchfield (2017). Toward Low-Flying Autonomous MAV Trail Navigation using Deep Neural
Networks for Environmental Awareness (TrailNet).** IROS 2017 (the arXiv comments call it an IROS 2017 submission).
Opened: https://arxiv.org/pdf/1705.02550 and https://arxiv.org/abs/1705.02550.

- augmentation-31 [definition, checked] Section IV-A: six outputs, three for heading and three for sideways offset (the
  offset labels come from a rig of three cameras 0.5 m apart). Augmentation: "random horizontal flips (with
  appropriate label changes)", contrast, brightness, saturation and sharpness jitter, scale 0.9 to 1.2, rotations up
  to 15 degrees, random crops.
- augmentation-32 [measured, checked] Table I and Section V-A: share of time flown without human override on an unseen
  250 m trail: Giusti's 3-class network 80 percent, a 6-class ResNet-18 88 percent, TrailNet 100 percent, although
  TrailNet's offline accuracy (84 percent) was below ResNet-18's (92 percent). The comparison mixes network, loss and
  label set. Section IV-A [authors' view]: with heading classes only, a drone flying parallel to the trail near its edge
  is never turned back; they call the offset classes essential from experience.

**Young, Gandhi, Tulsiani, Gupta, Abbeel, Pinto (2020). Visual Imitation Made Easy.** CoRL 2020 (PMLR 155, printed on
the PDF). Opened: https://proceedings.mlr.press/v155/young21a/young21a.pdf. Setting: behaviour cloning for pushing and
stacking, camera on a reacher-grabber tool (the fingers are in view), labels are the camera's relative motion.

- augmentation-33 [definition, checked] Section 3.3: colour jitter up to plus or minus 20 percent on brightness,
  contrast and saturation; 224 x 224 crop from 240 x 240; coloured cutout; rotation of plus or minus 5 degrees (labels
  not changed); horizontal mirror, where "Action labels are reflected as well."
- augmentation-34 [measured, checked] Table 1 and Section 4.4 (24 unseen object configurations): pushing 62.5 percent
  with plain behaviour cloning against 87.5 percent with crop and colour jitter; stacking 29.2 against 62.5 percent
  with rotation and colour jitter. With augmentation and half the data: 75.0 and 41.6 percent, above plain behaviour
  cloning with all the data.
- augmentation-35 [measured, checked] Section 4.4, Figure 6: on the offline test error, the best augmentations were
  only about 0.7 percent (pushing) and 0.9 percent (stacking) better; coloured cutout did worst (the authors guess it
  hides the fingers or the object). The mirror was not among the best pairs. Section 4.3 [authors' view]: small offline
  gains can mean large robot gains in fine manipulation.

**Quartey, Korsah (2018). Affordable Modular Autonomous Vehicle Development Platform.** IEEE ICAST 2018 (DOI
10.1109/ICASTECH.2018.8506757 printed on the PDF), arXiv 2006.11670. Opened: https://arxiv.org/pdf/2006.11670.

- augmentation-41 [measured as a single observation, checked] Learning layer and results sections: random horizontal
  flips with negated steering plus random shadows; the small car still drifted left and left the path after a few
  metres, which the authors put down to left-biased demonstrations that the flips only partly balanced (Figure 9). No
  controlled comparison.

Search note: in driving, flip-and-negate appears mostly in course projects (Udacity write-ups seen only in search
results, **not opened**). PilotNet itself does not flip (augmentation-25). I found no peer-reviewed driving paper that
measures the effect of flipping.

### 3.4 Viewpoint changes with corrected labels

**Pomerleau (1992). Neural Network Perception for Mobile Robot Guidance.** PhD thesis, Carnegie Mellon University,
CMU-CS-92-115 (DTIC AD-A249 972). Opened (scanned copy, rough text): https://rosap.ntl.bts.gov/view/dot/42152/dot_42152_DS1.pdf.
The 1991 journal paper with the same method (Neural Computation 3(1), "Efficient Training of Artificial Neural
Networks for Autonomous Navigation") was **not opened**; the CMU and course links failed.

- augmentation-26 [definition, checked] Sections 3.2.2 to 3.2.4: each camera picture is shifted and rotated in software
  so the vehicle seems placed differently, using the known camera pose and a flat-ground assumption. The new steering
  label comes from pure pursuit: steer so as to reach the road-centre point a look-ahead distance ahead (2 to 3 seconds
  of travel). 14 changed copies per picture, shifts of minus to plus 0.6 m, rotations of minus to plus 6 degrees; a
  change is redrawn when the corrected turn would be sharper than the network can output (about 30 m radius).
- augmentation-27 [measured, checked] Section 3.2.2, Figures 3.6 to 3.8: border areas the camera never saw must be
  filled. Filling from the nearest seen ground point smeared the lane line sideways; the smear's length correlated with
  the correct steering and the network learned to use it, then drove poorly without the smear. Filling along the road
  direction gave 37 percent less steering error on 100 test pictures (2.2 against 3.5 output units).
- augmentation-28 [measured, checked] Section 3.3, Figure 3.11 (one 100 m test drive each on a bike path): trained
  without changed pictures, the network drove off the road (it had over-learnt the last right turn); with changed
  pictures it followed the whole path but sat 28.9 cm right of centre on average (standard deviation 62.7 cm); with
  changed pictures plus a balanced buffer of past patterns, 2.7 cm (standard deviation 14.8 cm). The last step adds the
  buffer and steering-bias balancing, not more augmentation.

**Bojarski, Del Testa, Dworakowski, Firner, Flepp, Goyal, Jackel, Monfort, Muller, Zhang, Zhang, Zhao, Zieba (2016). End
to End Learning for Self-Driving Cars (PilotNet).** arXiv 1604.07316 (NVIDIA report; no peer-reviewed venue printed).
Opened: https://arxiv.org/pdf/1604.07316.

- augmentation-23 [definition, checked] Sections 2 and 5.2: left and right cameras give two off-centre views; other
  shifts and all rotations come from viewpoint transformation of the nearest camera, assuming points below the horizon
  lie on flat ground and points above it are infinitely far away. The label becomes the steering that returns the car
  to the desired position and heading within two seconds. Shift and rotation sizes are drawn from a zero-mean normal
  distribution with twice the spread measured for human drivers.
- augmentation-24 [authors' view, checked] Section 2: the flat-ground assumption distorts cars, poles, trees and
  buildings, which the authors say does not pose a big problem for training (no measurement shown). Section 5.2: the
  artefacts grow with the size of the change.
- augmentation-25 [measured, checked] Section 7.2: about 98 percent of the time autonomous on a typical drive, and 10
  highway miles without intervention. No result without augmentation is reported, so the augmentation's own effect is
  not measured. The paper does not mention mirrored pictures.

**Zhou, Kim, Wang, Florence, Finn (2023). NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via
Novel-View Synthesis (SPARTN).** CVPR 2023 (venue from the CVF listing in search results; I opened arXiv v1). Opened:
https://arxiv.org/pdf/2301.08556 and https://arxiv.org/abs/2301.08556.

- augmentation-36 [definition, checked] Sections 3.3, 3.4, 4.3 and 5.1: one NeRF per demonstration from the wrist
  camera pictures; camera poses are perturbed (simulation up to 0.2 rad and 3 mm, real robot 0.05 rad and 0.4 mm); the
  corrective label combines the original relative action with the inverse perturbation, so the gripper returns to the
  demonstrated path; the gripper, which sits at the same pixels in every picture, is masked out and pasted back; only
  the part of a demonstration before the object is touched is used, because NeRF needs a still scene. Section 2: in
  their words, "standard image augmentation approaches do not modify the action labels".
- augmentation-37 [measured, checked] Table 2 (simulated 6-DoF grasping of unseen YCB objects, 4 seeds): behaviour
  cloning 28.9 percent, DART 51.2, homography augmentation (camera rotation only, labels corrected) 29.7, SPARTN 74.7.
  Table 4 (real Franka, 8 objects, 14 to 25 demonstrations each, 20 trials each): 38.75 percent average for behaviour
  cloning, 61.25 percent for SPARTN.
- augmentation-38 [measured, checked] Table 3: adding random crop and colour jitter changed little (behaviour cloning
  25.3 to 28.9, DART 51.2 to 48.6, homography 23.9 to 29.7, SPARTN 68.3 to 74.7 percent).
- augmentation-39 [authors' view, checked] Section 6: the method needs still scenes; tasks that move objects would need
  view synthesis of moving scenes.

**Zeng, Florence, Tompson, Welker, Chien, Attarian, Armstrong, Krasin, Duong, Wahid, Sindhwani, Lee (2020). Transporter
Networks: Rearranging the Visual World for Robotic Manipulation.** CoRL 2020 (printed on the PDF). Opened:
https://arxiv.org/pdf/2010.14406 (v3).

- augmentation-40 [definition and authors' view, checked] Appendix 6.6: a random rotation and shift in the table plane
  is applied to the whole observation, and the pick and place poses are expressed in the same new frame, assuming the
  absolute position on the table does not matter. Section 3: such rigid changes look like real rearrangements only in
  top-down orthographic pictures; in perspective pictures objects scale and distort. No measured effect of the
  augmentation alone is given in what I read.

### 3.5 Symmetry that is only approximate

**Wang, Park, Sortur, Wong, Walters, Platt (2023). The Surprising Effectiveness of Equivariant Models in Domains with
Latent Symmetry.** ICLR 2023 (arXiv comments). Opened only the abstract page: https://arxiv.org/abs/2211.09231v2 (the PDF
was too large for my fetch tool).

- augmentation-42 [measured, unchecked in the body] Abstract: building an incorrect symmetry into a model can hurt, an
  extrinsic one can help (their two terms), even when the true symmetry is only latent. I did not read their
  definitions. My reading, to be checked: a mirror that makes pictures which never occur but keeps the labels
  consistent is closer to their extrinsic case; a change that makes realistic pictures with wrong labels is closer to
  their incorrect case.

---

## 4. When does a geometric augmentation break the label?

The test [my judgement, built on DrQ's definition (augmentation-2) and Codevilla's reason (augmentation-19)]: ask which
physical change the picture change imitates, then ask whether the demonstrator's move would differ after that physical
change.

| Camera and label | Shift of the whole picture | Rotation in the picture plane | Left-right mirror |
|---|---|---|---|
| Forward-looking camera, vehicle out of view, steering relative to the vehicle (driving, trails) | imitates turning the vehicle: label changes (PilotNet, ALVINN correct it; Codevilla avoids it) | imitates rolling the camera: heading unchanged, label kept for small angles (Giusti, TrailNet use up to 15 degrees) | left and right swap: label must be negated or swapped |
| Camera fixed to the tool, tool and target in view (our forearm camera; Young et al.'s grabber camera; wrist cameras with the gripper in view, as in SPARTN) | tool and scene move together, their relation is unchanged: label kept | all directions in the picture turn: a label given in the picture's frame must turn by the same angle | label must be mirrored (Young et al. do this) |
| Same camera, but only the scene moves while the tool stays where it is in the picture (SPARTN's NeRF views with the gripper pasted back; a homography of our sheet with block and puck kept) | imitates the tool sitting elsewhere relative to the target: label changes; SPARTN computes the correction from hand-eye geometry | imitates a camera rotation: label changes; SPARTN's homography variant corrects it | not studied |
| Camera fixed to the tool, tool out of view (a fixed pixel stands for the tool) | imitates the tool sitting elsewhere: label changes | label changes | label must be mirrored |
| Top-down orthographic camera, action in table coordinates (Transporter) | correct the label by the same shift | correct it by the same rotation | not studied |

Further ways a geometric change breaks the label:

- **It moves the tool somewhere impossible.** Upside-down flips and 90 degree turns put the tool or vehicle where it
  never is. In RAD, flips and 90 degree turns without label changes scored 223 and 268 against 920 for crop
  (augmentation-7); DrQ's flips and small rotations were also worse than shifts (augmentation-4).
- **It hides what the decision needs.** A crop that removes the line ahead or the puck takes away the reason for the
  move [my judgement]; Young et al. saw coloured cutout do worst and guess it hid the fingers or the object
  (augmentation-35).
- **It is too large for the class width.** A coarse class tolerates a small label-changing shift (Giusti, section 3.3);
  the same shift breaks a finer label [my judgement].
- **Whole-picture shifts are safe when tool and target are both in view.** robomimic and Diffusion Policy shift their
  pictures (gripper and objects in view) by about 5 percent without touching the labels, and robomimic measured a large
  benefit (augmentation-10, -11, -13). Because gripper and objects move together, the relation that decides the action
  is unchanged [my judgement]. A whole-picture shift is risky only when a fixed pixel position stands for something
  that is not in view (the vehicle's heading, an out-of-view tool).

## 5. When can the label be corrected, and what goes wrong?

1. **Mirror.** Easy when the action has a matching mirror symmetry: left and right classes (Giusti, TrailNet), a signed
   steering value (Quartey and Korsah) or an action vector that can be reflected (Young et al.). The mirrored picture
   may show a world that never occurs (the robot and camera are rarely perfectly symmetric); for class labels this is
   probably harmless (augmentation-42, unchecked), for exact continuous labels the mirror rule is only approximate [my
   judgement]. Mirrors can also balance left-right bias, but Quartey and Korsah's car still drifted (augmentation-41) and
   Pomerleau balanced the buffer explicitly (augmentation-28).
2. **Viewpoint change with a geometric model.** ALVINN and PilotNet use a flat world, the known camera pose and a
   steering law; SPARTN uses a NeRF and hand-eye calibration. Measured benefits are large (augmentation-28, -37), but
   only when:
   - the world model holds where it matters (flat ground; a still scene). Objects off the plane are distorted
     (PilotNet, unmeasured) or kept out (SPARTN pastes the gripper back and skips frames after contact);
   - the unseen border is filled without a cue that correlates with the corrected label (augmentation-27);
   - the correction rule resembles what the demonstrator would do (pure pursuit with 2 to 3 seconds look-ahead in
     ALVINN; return within 2 seconds in PilotNet; return to the path in SPARTN);
   - the perturbations stay plausible (ALVINN redraws impossible ones; PilotNet uses twice the human spread; SPARTN tunes
     the size per platform);
   - the change moves the tool relative to the target: a camera rotation alone (SPARTN's homography variant) gave
     29.7 against 28.9 percent, no real gain (augmentation-37).
3. **Real recovery data instead.** Recording with injected noise and human correction (Codevilla, augmentation-22) or
   with side cameras (PilotNet, Codevilla, TrailNet) gives corrected labels without image synthesis. TrailNet's and
   Giusti's results show that heading labels alone do not teach sideways correction (augmentation-30, -32).

---

## 6. What this means for our project (judgement unless marked otherwise)

### 6.1 What our pictures show

I looked at one recorded picture (640 x 480):
`D:\AI_Project\03_data\raw\main_data\2026-11-08_session_01_emranul\path_03_two_curve_episode_004_plan_03-04_try_1_score_03\frames\004.jpg`
(augmentation-43, by eye, one picture only):

- the gripper's blue fingers at the bottom edge, the black magnet block above them, the red puck just above the block
  at the bottom centre;
- the puck is about 65 px wide, so about 27 px per cm at the puck [calculated from the 2.4 cm puck]; the line is about
  25 px wide near the puck and about 20 px near the top, so the view looks fairly steeply down with moderate perspective;
- the sheet's right edge at about x = 415, the wooden table to the right of it.

From the team's memory and config (not literature): the camera is fitted (`config.FOREARM_CAMERA_PARAMETERS`, focal
length 502 px, principal point about x = 334, 58 mm from the forearm axis, map error about 2 mm median); move names are
in the picture's frame; a forearm turn rotates the sheet in the picture but not the tool; pulls hide the puck behind the
block.

### 6.2 Each augmentation for our labels

| Change | Move name (picture frame) | Exact step (tip forward and left, forearm twist, lean, hand) | Suggestion |
|---|---|---|---|
| Brightness, contrast, saturation, small hue, blur, noise | kept | kept | use first (evidence: augmentation-19 to -21, -15, -16, -34); keep hue changes small so the red puck stays red |
| Whole-picture shift (pad and crop), about plus or minus 5 percent | kept: tool, puck and line move together | kept (small perspective error) | use; about the size of DrQ, robomimic, Diffusion Policy and Hansen et al. (4.7 to 6 percent) |
| Random crop that cuts parts away | kept while the line ahead and the puck stay in view | kept | keep it to about 5 percent; check that the far line is not cut |
| Whole-picture rotation | wrong near the borders between directions unless rotated: about 6 percent of labels at plus or minus 5 degrees, 11 percent at 10, 17 percent at 15 [calculated, assuming directions spread evenly within 45 degree sectors] | tip forward and left: rotate by the same angle; forearm twist: no clear rule | skip at first; if used, rotate the exact step and recompute the name |
| Left-right mirror | right after swapping names (forward-left with forward-right, left with right, back-left with back-right, turn left with turn right) | only approximate: camera 58 mm off the forearm axis, principal point about 14 px right of centre, hand-turn sign not yet checked | try for the move-name model; it also mirrors the track shapes, which may help on unseen exam tracks; measure it, since no paper isolates its effect |
| Upside-down flip, 90 degree turns | broken: the tool goes to the top, forward becomes back | broken | never |
| Corrective sideways or heading offset (warp the sheet by a homography, keep tool and puck pixels fixed) | can be corrected: pure pursuit towards the demonstrated path a few steps ahead | tip forward and left can be corrected | second stage, only if closed-loop tests show drift off the line |

### 6.3 Practical notes

- **Shift size.** Plus or minus 5 percent of the width is about plus or minus 32 px at full resolution, about 1.2 cm
  at the puck [calculated with 27 px per cm]. Because the tool moves with the picture, this does not change the move name;
  it stops the network relying on the tool's fixed pixel position, which also covers small tape drift (the camera check
  allows under 4 px). Use the full picture at test time with pad-and-crop, or the same centre crop with crop-only (as in
  Diffusion Policy, augmentation-13). If the picture is cropped before training (for example to drop the wooden table),
  crop first, then shift.
- **Photometric.** Brightness, contrast and saturation within about plus or minus 20 percent (Young et al.'s values),
  hue at most a few percent, Gaussian blur (the 6.11 hazy pictures and blur warnings show blur happens), small Gaussian
  noise. The hue limit is my judgement; no paper I opened tests hue on a red target.
- **Mirror.** Apply with probability 0.5 and check the class counts afterwards. The mirrored pictures show the table on
  the wrong side; for move names this is probably harmless (augmentation-42, unchecked).
- **Corrective homography.** Feasible because the sheet is flat and the camera is fitted. Warp only the sheet and keep
  the gripper, block and puck pixels where they are (SPARTN pastes its gripper back; our puck moves with the block
  while pushed). Crop all pictures, real and synthetic, to the same inner window so no filled border is ever seen
  (augmentation-27). Recompute the move name by pure pursuit towards the demonstrated path two or three steps ahead,
  keep offsets within what the demonstrations show (PilotNet's twice-the-human rule), and leave out the goal region and
  pulls. This is a sizeable job for a beginner, and no paper I found tests it on planar pushing.
- **How to judge.** Compare augmentation choices on the robot or at least on whole held-out sheets, not only on offline
  accuracy: Young et al. saw about 1 percent offline against 25 points on the robot (augmentation-34, -35), and the most
  accurate trail network was not the one that flew furthest (augmentation-32).

---

## 7. Gaps

- No paper I opened measures mirrors alone on robot success.
- No paper I opened reports blur or noise separately in behaviour cloning; they are always in a bundle.
- I found no study of corrective (label-changing) augmentation for planar pushing, where the pushed object moves with
  the tool.
- PilotNet's claim that flat-world distortions do not matter is unmeasured.
- Not opened: Pomerleau (1991) Neural Computation paper, ALVINN (1989), DART (Laskey et al. 2017), DAgger (Ross et al.
  2011), DrQ-v2 (Yarats et al. 2021), automatic augmentation (Raileanu et al. 2021), the Udacity write-ups, and the full
  text of Wang et al. (2023). None of their claims is used as evidence here.

## 8. Working files

The papers' extracted texts used while writing this note were deleted afterwards, so only this note remains in the
folder.
