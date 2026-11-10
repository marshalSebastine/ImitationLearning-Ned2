# Improving after the first model

## In short

The first model will make mistakes on the robot, and the data say where: it is weakest when the puck has drifted
more than 2 cm off the line, in tight bends and loops, in the last few centimetres of a line and when deciding to
stop, while the crossings are not a weak spot. We therefore plan one or two "correction rounds": the model drives, a
person watches, and when the model goes wrong the person stops it, guides the arm by hand with the motors off, and the
recording tool replays that guided path exactly as it does for a demonstration, so the correction gets the same labels
as all our other data. When we trained on fewer sheets or on fewer episodes per sheet, the result on an unseen sheet
fell more with fewer sheets than with fewer episodes, so new printed tracks should help more than more episodes on the
six sheets; this is why we need the professor's answer to question 15 (may we print our own tracks). A round of
50 correction episodes costs about 4 to 6 robot hours and four new tracks with 20 demonstrations each about 10 to 13
robot hours (calculated from our measured times). Nothing here has been run on the robot; every robot step waits for
the team's decision.

## 1. Evidence: where the first model goes wrong

Markers: **measured** (computed from our data or runs), **calculated** (derived from measured numbers with stated
assumptions), **literature** (a claim its checker marked supported or corrected), **judgement** or **guess**. File
names in brackets are listed with their full paths at the end of the section.

### 1.1 Per held-out sheet

Direction measures count only pictures after which the tip really moved ("moving pictures"); "within 45" is the share
whose predicted push direction lies within 45 degrees of the demonstrated one.

| Held-out sheet | Moving pictures (episodes) | End-to-end ResNet-18, exact step: within 45 | Always forward: within 45 | Frozen probe, exact step: within 45 |
|---|---|---|---|---|
| 01 straight | 569 (19) | not run (final candidate: 0.779 of 578) | 0.747 | 0.747 |
| 02 arc | 2,372 (44) | not run (final candidate: 0.632 of 2,425) | 0.565 | 0.583 |
| 03 two curves | 3,458 (45) | 0.700 (final candidate: 0.680) | 0.522 | 0.624 |
| 04 meander | 5,271 (45) | not run | 0.485 | 0.548 |
| 05 loop | 2,471 (30) | 0.667 | 0.509 | 0.563 |
| 06 dashed loop | 1,809 (25) | 0.675 | 0.402 | 0.483 |

Measured: [runs: 2026-11-09_resnet18_finetune_exact/metrics.json, field within_45_deg per fold;
2026-11-09_resnet18_final_candidate/fold_without_path_01.json to _03.json; situation_errors.csv, rows "sheet"]. The
end-to-end model beats always-forward by 17.8, 15.8 and 27.3 points on sheets 03, 05 and 06. The final candidate (exact
step, stop output, the audit's cleaning, 8 epochs, last weights; its rows differ slightly from the older runs' because
of the cleaning) had finished its folds for sheets 01, 02 and 03 when this was written; sheets 04 to 06 were still
training.

The stop is the weakest part. End-to-end exact model, episodes stopped right (within 2 cm of where the demonstration
stopped) at the fixed threshold 0.5: sheet 03: 31 of 45 (0.689), sheet 05: 6 of 30 (0.200), sheet 06: 5 of 25 (0.200);
with the threshold tuned on validation episodes and two pictures in a row: 34 of 45, 5 of 30 and 8 of 25. Sheet 05
mostly stopped too early (19 of 30, tuned); sheet 06 never stopped in 19 of 25 (at 0.5) [metrics.json, fields
episodes_stopped_* and tuned_two_*; stop_by_end.out.txt]. Measured. The final candidate stopped right in 16 of 20
episodes of sheet 01, 19 of 45 of sheet 02 and 31 of 45 of sheet 03 (tuned, two in a row); at 0.5 it never stopped in
18 of the 45 sheet-02 episodes [fold_without_path_01.json to _03.json].

### 1.2 Per situation

End-to-end exact model on its three held-out sheets (03, 05, 06; 7,738 moving pictures in 100 episodes), and the
frozen exact probe on all six held-out sheets (15,950 moving pictures in 208 episodes). Situations from the project's
error analysis (02_code/training_error_analysis.py) and from the audit's two line registrations (good fits only; "no
fit" pictures left out) [situation_errors.csv; situation_errors_by_sheet.out.txt]. Measured.

| Situation | End-to-end exact: within 45 (moving pictures) | Spread over sheets 03 / 05 / 06 | Frozen exact probe, six sheets |
|---|---|---|---|
| Puck within 1 cm of the line | 0.744 (4,155) | 0.753 / 0.743 / 0.731 | 0.599 (8,758) |
| Puck 1 to 2 cm off the line | 0.615 (1,360) | 0.673 / 0.555 / 0.595 | 0.537 (2,709) |
| Puck more than 2 cm off the line | 0.571 (1,105) | 0.598 / 0.567 / 0.539 | 0.503 (2,152) |
| Straight stretch | 0.798 (2,378) | 0.790 / 0.816 / 0.790 | 0.629 (4,972) |
| Bend (tip's direction over the next 3 steps differs by over 15 degrees from the last 3) | 0.633 (5,360) | 0.661 / 0.600 / 0.622 | 0.546 (10,978) |
| Line radius 10 cm or more | 0.735 (2,937) | 0.682 / 0.823 / 0.760 | 0.619 (7,183) |
| Line radius under 1.5 cm | 0.610 (390) | 0.625 / none / 0.510 | 0.487 (1,382) |
| Last 3 cm of line | 0.607 (461) | 0.551 / 0.562 / 0.800 | 0.497 (1,126) |
| More than 8 cm of line to go | 0.683 (6,163) | 0.719 / 0.662 / 0.639 | 0.576 (12,509) |
| Puck within 2 cm of a crossing (sheets 05, 06) | 0.799 (432) | (05 and 06 pooled) | 0.590 (432) |
| Loop sheets, away from the crossing | 0.613 (1,985) | (05 and 06 pooled) | 0.507 (1,985) |
| Forearm turned 75 degrees or more | 0.654 (1,538) | | 0.572 (3,574) |
| First 3 rows of an episode | 0.812 (191) | | 0.618 (400) |

What this says (measured, with the reading as judgement):

1. **Off the line is the clearest weakness**: 15 to 19 points lower than on the line on every one of the three sheets.
   On the robot the model's own small errors will push the puck off the line more often than in the demonstrations, so
   this weakness should grow in closed loop (judgement; in DAgger's game test, behaviour cloning did not improve with
   more expert data while letting the policy drive and labelling its states did, interactive_correction claim 3).
   These are exactly the states a correction round collects (section 3).
2. **Bends and tight loops** are 13 to 22 points below straight stretches on every sheet.
3. **The end of the line** is weak on two of three sheets, and the stop fails on both loop sheets (section 1.1).
4. **The crossings are not a weak spot offline** (0.799 within 2 cm), although only 2 crossings on 2 sheets are in the
   data, so closed-loop behaviour there is untested.

### 1.3 The stop and the paper edge

The audit found that every line end lies 2.35 to 8.03 cm from a paper edge and that stop pictures show no line ahead,
so a model might learn "stop where the paper ends" [audit, view_and_tracks-2 and table 10.1]. Our check does not
confirm this: sheet 06's end B, 8.0 cm from its paper edge (the farthest in the data), was stopped right in 8 of 15
episodes (tuned), while sheet 06's end A, 4.4 cm from the edge, was never stopped in 10 of 10. The rank correlation
between a line end's edge distance and the share of its episodes never stopped was 0.14 to 0.54 (6 sheet-and-end groups
for the end-to-end model, 12 for the frozen probe, fixed or tuned threshold). In the probe, the 89 episodes finishing
5 cm or more from an edge were stopped right in 0.18 (tuned) against 0.24 for the other 119, and never stopped in 0.28
against 0.30 (fixed threshold: 0.46 against 0.37) [stop_by_end.out.txt]. Measured: at most a weak tendency, and the
far ends are mostly the hard loop and arc sheets, so the sheet and the edge distance cannot be told apart in these
data (judgement). A guess at the reason for the 0 of 10: end A of sheet 06 is a straight dashed end,
and a model trained without sheet 06 has never seen a dashed line end, because sheet 06 is the only dashed sheet.
Early stops cluster where no line is visible ahead: in the audit's count, 1,588 of the 12,241 pictures taken more than
10 cm before the stop show no ink in the top half, against 164 of the 210 stop pictures [audit, view_and_tracks-2].
Measured.

### 1.4 Rare situations in the data

From the audit's table 10.4 and class_balance-14 (measured): puck within 2 cm of a crossing 453 to 519 pictures (2.8 to
2.9 percent; two crossings, sheets 05 and 06 only); stop rows 259 of the 18,053 cleaned rows (1.43 percent); line radius
under 2.5 cm almost only on sheets 03 and 04 (859 and 2,182 pictures; 102 and 72 on sheets 05 and 06); one dashed sheet
with one dash rhythm (1,982 pictures); moves towards the base at a visible puck about 1,400; natural light 3,007
pictures, sheets 02 and 04 only; forearm turned 115 degrees or more 207 pictures. The audit's list of what an unseen
track could contain that none of the six shows (a guess list): a line end more than 8 cm from a paper edge, a corner
sharper than about 140 degrees, a crossing near 90 degrees or two crossings, parallel stretches closer than 2.9 cm away
from a crossing, another dash rhythm, long stretches towards the robot, lines longer than 72 cm [audit,
view_and_tracks-16].

## 2. Evidence: more episodes or more different sheets?

### 2.1 Our data-amount experiment

Frozen ResNet-18 grid features with a classes head, leave one sheet out, three seeds per setting. Each fold trained on
a share of the episodes of all five training sheets, or on all episodes of 1 to 4 of them [data_amount.json]. The
number of training pictures is calculated from the probe code's own draw of sheets (0.85 of each kept sheet's rows,
after the 15 percent of episodes held back for early stopping) [data_amount_regression.out.txt].

| Training data per fold | Training pictures (mean) | Within 45, mean of 6 folds (seed range) | Per held-out sheet 01 / 02 / 03 / 04 / 05 / 06 | Stopped right, tuned, 2 in a row |
|---|---|---|---|---|
| 1 sheet, all episodes | 3,304 | 0.456 (0.431 to 0.471) | 0.588 / 0.404 / 0.486 / 0.468 / 0.442 / 0.350 | 0.097 |
| 2 sheets | 5,758 | 0.507 (0.491 to 0.523) | 0.663 / 0.437 / 0.533 / 0.507 / 0.500 / 0.404 | 0.182 |
| 3 sheets | 7,632 | 0.520 (0.514 to 0.533) | 0.724 / 0.452 / 0.523 / 0.498 / 0.483 / 0.443 | 0.255 |
| 4 sheets | 10,774 | 0.532 (0.531 to 0.533) | 0.740 / 0.437 / 0.550 / 0.542 / 0.545 / 0.378 | 0.323 |
| 5 sheets, 25 % of episodes | 3,264 | 0.514 (0.512 to 0.518) | 0.702 / 0.439 / 0.522 / 0.516 / 0.500 / 0.407 | 0.207 |
| 5 sheets, 50 % | 6,527 | 0.546 (0.541 to 0.549) | 0.739 / 0.493 / 0.538 / 0.520 / 0.547 / 0.437 | 0.257 |
| 5 sheets, 100 % | 13,055 | 0.555 (0.541 to 0.574) | 0.744 / 0.487 / 0.557 / 0.538 / 0.559 / 0.446 | 0.320 |

Measured (fields within_45_deg and tuned_two_episodes_stopped_right; seed range = range of the three seeds' fold
means). The three 100 % runs used identical training rows, so their 3.3-point range is training noise alone.

- **At equal pictures, more sheets win.** Five sheets with a quarter of their episodes (3,264 pictures) scored 0.514,
  against 0.456 for one sheet with all its episodes (3,304 pictures), better on all six folds by 3.5 to 11.4 points.
  Five sheets at half their episodes (6,527 pictures) scored 0.546, above four full sheets with 65 percent more
  pictures (10,774; 0.532); per fold this is mixed (better on 02 and 06, worse on 03 and 04, equal on 01 and 05).
- **Separating the two** with a least-squares fit over all 126 fold results (one constant per fold): per doubling of
  training pictures +1.7 points within 45 (95 % bootstrap +0.8 to +2.7), per doubling of different sheets at the same
  pictures +2.5 points (+1.6 to +3.4); path within 45 +1.8 (+0.6 to +3.1) and +3.4 (+2.2 to +4.5). Calculated.
- **The episode curve flattens, the sheet curve does not yet:** +3.2 points from 25 to 50 percent of the episodes,
  +0.9 from 50 to 100 percent (inside the seed range), against +2.3 from four to five sheets. Measured.
- **The stop is different: it grows with the amount of data.** Stopped right (tuned, two in a row) +8.7 points per
  doubling of pictures (+5.7 to +11.7) and +3.3 per doubling of sheets (+0.2 to +6.2), because each episode holds one
  stop picture. Calculated (residual spread per fold result 12 points: noisy).
- **Which sheet helps which** (residuals per held-out sheet, 12 draws per fold) stays within the noise, with one weak
  hint: held-out sheet 05 scored 4.7 points better when sheet 06 was among the training sheets, and sheet 06 3.3 points
  better with sheet 05, as if the two loops help each other. Calculated, weak.

Limits (judgement): frozen features with a classes head, not the end-to-end exact model we will use, whose curve is
not measured; the fit covers 1 to 5 training sheets, so anything beyond six sheets is an extrapolation (a guess); the
random draws over-represent sheet 06 (in 11 of 12 draws of some folds); early stopping used validation episodes of
all five training sheets even when fewer were trained on.

### 2.2 What the literature says

- Lin et al. (2025, ICLR): a diffusion policy generalised to unseen environments and objects roughly as a power law of
  the number of training environments and objects, while the number of demonstrations alone showed no clear power law;
  gains stopped at about 50 demonstrations per environment-object pair (Sec. 4.1 to 4.3). Literature, checked by the
  diffusion_and_generative checker (section 4.2).
- Xie et al. (2023), Factor World: the generalisation gap fell from about 0.4 to under 0.1 as training environments
  went from 5 to 100 (simulation). Literature, augmentation checker, evidence the researcher missed.
- No opened study tested new path shapes of the same appearance, which is what the exam asks (augmentation checker).
- Mirrored pictures with swapped labels would double the number of track shapes on the laptop alone (augmentation
  checker), but the audit advises against mirroring unless the robot's dark bottom corner is masked (the bottom right
  is darker than its mirror in 16,793 of 18,522 pictures, notes_context-10), and they cannot add features the six
  sheets lack. The end-to-end mirror run (classes model with mirroring at probability 0.5, otherwise the same settings
  as resnet18_finetune_classes) had finished two folds: within 45 0.670 on sheet 03 and 0.659 on sheet 05, against
  0.663 and 0.659 without mirroring, so no gain so far (measured, two folds; smaller than the about 3-point spread
  between seeds measured for the probes)
  [2026-11-09_resnet18_finetune_mirror/fold_without_path_03.json and _05.json].

## 3. Evidence: correction rounds

### 3.1 What the literature supports

Claim numbers are those of the interactive_correction note; every claim used here was marked supported by its checker,
unless another checker is named.

- **The idea** (DAgger, Ross et al. 2011, claim 1): let the trained policy drive, collect the states it really reaches,
  get the expert's action there, add them and retrain.
- **Human-gated take-over is the usual real-robot recipe** (HG-DAgger, Kelly et al. 2019, claim 5): the human takes
  over when the novice goes wrong and only the human's recovery is labelled. On a real car (low-dimensional input,
  small test): behaviour cloning 1 collision and 6 road departures, HG-DAgger 0 and 0 (claim 8).
- **At equal data, corrections beat more of the same demonstrations** on real robots in BC-Z (claim 32), at a far larger
  scale (25,877 episodes, 12 robots, 125 robot hours): half manual plus half HG-DAgger episodes 53 percent against 27
  percent manual only on one task, 47 against 23 percent over 8 tasks.
  ThriftyDAgger cable routing: behaviour cloning 0 of 15, HG-DAgger 10 of 15 (claim 14; the checker notes the
  behaviour-cloning baseline had less data).
- **Recovery data matter for driving from one camera** (Codevilla et al. 2018, augmentation checker claim 22,
  supported with a caveat): without the noise-injected recovery data, simulated success fell from 88 and 64 percent to
  56 and 22 percent, and interventions on the real truck rose from 0.67 to 8.67.
- **Rounds and sizes**: about half of the opened real-robot studies used 1 to 3 rounds (checker's correction of
  "mostly"); round sizes were 20 to 33 percent of the initial data (HG-DAgger 2,000 labels on 10,000, claim 7; Sirius
  and Mandlekar et al. one third, claims 18 and 23) or 50 episodes (CR-DAgger, claim 30).
- **Training on the corrections**: retraining from scratch on old plus correction data was stable; fine-tuning on the
  corrections alone was unstable (minus 30 and minus 15.6 points); one batch of 50 episodes was more stable than an
  update every 10 episodes; using the moments just before a take-over lowered success (CR-DAgger, claims 29 and 30;
  Sirius gives them weight zero, claim 22). Up-weighting the corrections helped in simulation (claims 17 and 19); the
  checker warns that drawing a correction set of a few percent of the data as half of every batch weighs each
  correction picture about 25 times a demonstration picture, close to the unstable fine-tuning, and suggests comparing
  10 to 25 percent with 50 percent.
- **Labels typed afterwards are unreliable**: five participants relabelling their own teleoperated runs afterwards
  agreed with what they had done at correlations of only 0.60 (rotation) and 0.22 (translation), and labelling the
  robot's runs afterwards reached about 40 percent success against 60 percent for plain demonstrations (Laskey et al.
  2017, claims 24 and 25).
- **What no study shows** (checker): that correction data carry over to layouts outside the training set, and how many
  robot hours a round costs.

### 3.2 How a round works with our recording tool

From the code-reading audit [code_actions note, sections (a), (c) and (f)]:

- The tool's loop already is the loop the model will run: fresh picture, decision, log, move, with a decider called
  once per picture, `decider(picture_bytes, state, step) -> (action, source)`. The model mode does not exist yet; it
  needs the nine pieces the audit lists (a mode, a decider that never hands `state` to the model, a fixed-wrist
  executor for the exact step, clamps, guards, a refusal counter, the step budget switched on, a separate raw folder,
  columns for the model's outputs, and Esc and x polling).
- The take-over already exists in `hand_guided_exact`: the key h saves a reference picture, the person guides a new
  path with the motors off, the arm lifts, returns to the correction's start from 5 cm above, the puck goes back with
  the ghost picture, and the new path is replayed waypoint by waypoint with exactly the demonstrations' labelling
  (record_episodes.py lines 898 to 903, 1172 to 1211, 1400 to 1421). A decider can request it by returning "reguide".

Proposed procedure for one correction episode (judgement; to be written as a robot checklist first):

1. Lay the puck at the start as for a demonstration; the model drives one step per picture (about 1.2 s per step, so
   the watcher has time to react). As with the replay now, g lets it run on and any key pauses it [code_actions note,
   (a)].
2. When the model goes wrong (puck drifting off the line, stalling, turning the wrong way, stopping early, or
   running past the end), pause and press h. Motors go off (FreeMotion) before anyone touches the arm; guide the arm
   by its body, never by the camera or its tape.
3. **Version 1 (reuses the take-over as it is)**: guide from the failure to the end of the track; the tool replays
   the guided path to the stop, as after a demonstration correction now. **Version 2 (small change)**: guide only until
   the puck is back on the line and moving well, about 5 to 15 cm; after the replay the model drives again. Version 2
   finds more failures per hour; start with version 1 in the pilot.
4. If the puck has left the sheet or the view, end the episode instead of recording a long recovery (literature note,
   section 4.2, judgement).
5. After the episode, note the failure type in the review form.

How the corrections are labelled (the point of using this tool):

- Every replayed correction row gets exactly the demonstrations' labels: the move name, the exact step (tip forward
  and left in the current picture's frame, forearm, lean and hand changes) from the same `__label_motion`, the waypoint
  number and source replay_exact [code_actions note, (a)]. So corrections join the 18,053 training rows without any
  conversion. The first replayed row (waypoint 0) is a null move and is dropped by the audit's cleaning rule B2, like
  the 239 correction starts of the demonstrations.
- Add tags (new step columns; the audit notes that STEP_COLUMNS must be extended or the writer raises): round,
  model version, source (model, correction replay), the take-over picture, and the model's raw and clamped output on
  every model step.
- **Train only on the human's corrections**, not on the model's own steps (HG-DAgger style): this keeps the course
  rule (the motion comes from our own demonstrations) and avoids learning the model's errors; the 1 to 3 model steps
  before a take-over are then excluded automatically. Keep the model steps for analysis, and use the correction
  pictures also as an offline validation set with recoveries (Codevilla et al. 2018 advise validation data with
  recoveries; that this raised the offline-to-driving correlation from 0.39 to 0.77 holds only in their main condition,
  offline_vs_closed_loop checker, claim 3, corrected).
- Why not the other ways to label (judgement from the code audit): `hand_guided_exact` has no direction keys, and the
  keyboard's fixed 1 cm moves drop the forearm twist and, when turned, re-solve for the least lean, which moves the
  camera by a median 16 mm and the picture by 56 px from the demonstrated views [code_actions note, summary 3 and 4],
  so key-override corrections would not match our labels or our training pictures. Exact steps cannot be typed
  afterwards, and labels given afterwards were unreliable in Laskey et al.

Measured from the demonstrations (the same operators already made 239 such take-overs inside their own replays): a
take-over costs a median 51.6 s from the last row before it to the first replayed row (quartiles 38.9 to 71.0 s,
10th to 90th percentile 28.2 to 87.4 s; 239 corrections in 130 of 210 kept episodes), of which 17.1 s is guiding and
33.4 s lifting, returning and putting the puck back; a correction replays a median 22 rows (quartiles 15 to 41.5). With
practice the pause fell from 80.7 s on sheet 01 (5 corrections) to 32.5 s on sheet 06 (21). A regression over the 210
episodes gives 1.01 minutes per correction (95 % bootstrap 0.83 to 1.19) beyond its steps [episode_timing.out.txt].
The puck went back more than 3 px off its reference in 24 of 179 checked corrections, a small source of label noise
[audit, section 11.2, point 20].

### 3.3 An offline check with our own correction replays

The demonstrations already hold 239 hand corrections; their replays are 6,983 of the 18,430 rows the probes use (37.9
percent) and end 130 of the 210 episodes, so they hold most stop pictures [audit, class_balance-10;
recovery_ablation.out.txt]. Do such recovery-like rows help where the model is weakest? The frozen exact probe (six
held-out sheets, three seeds) was trained with all rows, without the correction-replay rows, and, as a control for
having less data, without the first passes of randomly chosen training episodes holding as many rows
[recovery_ablation.py, recovery_ablation_situations.out.txt]. Measured; step within 45 pooled over the six held-out
sheets, mean of three seeds (seed range), change against all rows in points:

| Test pictures | All rows | Without correction replays | Without as many first-pass rows |
|---|---|---|---|
| All moving pictures (15,950) | 0.581 (0.579 to 0.584) | 0.565 (0.561 to 0.569), -1.6 | 0.552 (0.547 to 0.556), -2.9 |
| Puck within 1 cm of the line (8,758) | 0.616 | 0.600, -1.6 | 0.586, -3.0 |
| Puck 1 to 2 cm off (2,709) | 0.536 | 0.515, -2.1 | 0.505, -3.1 |
| Puck more than 2 cm off (2,152) | 0.513 (0.500 to 0.527) | 0.506 (0.504 to 0.510), -0.7 | 0.475 (0.454 to 0.486), -3.8 |
| Test pictures that lie inside correction replays (5,996) | 0.521 | 0.499, -2.2 | 0.501, -2.0 |
| Six-fold mean, within 45 | 0.600 (0.596 to 0.603) | 0.583 (0.581 to 0.587), -1.7 | 0.577 (0.571 to 0.580), -2.3 |
| Six-fold mean, stopped right (tuned, two in a row) | 0.287 (0.263 to 0.319) | 0.211 (0.171 to 0.234), -7.5 | 0.264 (0.249 to 0.283), -2.3 |

What it says: the operators' own corrections act like ordinary extra data for the push direction (leaving them out cost
a little less than leaving out the same number of first-pass rows, 1.7 against 2.3 points of the six-fold mean, both
beyond the seed ranges), not like targeted recovery data: without them the off-line pictures did not get worse than the
others (-0.7 and -2.1 against -1.6 on the line). They matter most for the stop (-7.5 points against -2.3 for the
control), because most stop pictures sit at their ends. The model is about 10 points weaker on pictures from correction
stretches than on first passes (0.521 against 0.617), another sign that recovery states are hard. Judgement: these corrections were made
where the operators' own replays went wrong, not where a model goes wrong, so this offline check neither supports nor
rules out on-policy corrections; it does argue for adding corrections to new episodes rather than replacing them, and
for keeping enough stop pictures. Only the paired robot test after round 1 can show the size of the gain.

## 4. Robot time

### 4.1 Measured

| What | Measured value | Source |
|---|---|---|
| One step (move, fresh picture, decision) | 1.22 s median (17,863 replayed steps); 1.24 s mean of 18,046 plain steps | code_actions note (e); episode_timing.out.txt |
| Model time per picture | 21.5 ms (end-to-end ResNet-18, exact, 160 x 120, laptop busy) | 2026-11-09_resnet18_finetune_exact/metrics.json, milliseconds_per_picture |
| Steps per demonstration, median | 01: 31.5, 02: 66, 03: 106, 04: 168, 05: 103.5, 06: 82 | code_actions note (e) |
| Kept demonstration, start to finish | 4.31 min median (quartiles 2.97 to 6.02), 210 episodes | episode_timing.out.txt |
| Initial hand guiding inside it | 34.5 s median (27.2 to 46.0) | episode_timing.out.txt |
| Start to the next start | 5.78 min median (4.12 to 7.93), 207 episodes | episode_timing.out.txt |
| Whole sittings | 28.5 h for 210 kept episodes, 8.13 min each; per sitting 13.4, 17.1, 7.8, 8.5 and 5.6 min (the last, practised one on sheets 04 to 06); 29 of 239 attempts with an episode file were not kept | episode_timing.out.txt |
| Near-end start (cheapest stop example) | 2.4 min start to next start, 19 rows median, 42 episodes | timing_by_start_kind.out.txt |
| Take-over (hand correction) | 51.6 s median, see 3.2 | episode_timing.out.txt |

### 4.2 Calculated per option

Assumptions (guesses, named in robot_hours.py): a model needs 1.0 to 1.5 times the demonstration's steps; the high
end adds the measured whole-sitting overhead (8.13 / 5.78 = 1.41); a first model finishes 30 percent of runs without
help and gets one take-over per failed run (version 1), at mid-track, so a correction replays half a track; a new
track has 93 steps (the mean of the six sheets' medians), costs 0.5 h to print, tape and teach, and gets 20
demonstrations. Expected gains apply the fit of section 2.1 to the 18,053 cleaned rows and six sheets; beyond six
sheets this is an extrapolation (a guess).

| Option | Robot hours | Data added | Expected offline gain (frozen-probe fit) |
|---|---|---|---|
| Baseline test of a model: 5 trials on each of the six sheets | 1.9 to 3.3 | none | none (finds the failures) |
| 10 sealed new tracks x 2 trials | 1.3 to 2.2 | none | none (unseen-track test) |
| Pilot: 5 correction episodes | 0.4 to 0.6 | about 160 rows | none (measures the take-over in model mode) |
| Correction round, 50 episodes | 3.7 to 6.3 | about 35 corrections, 1,600 rows (9 %) | not estimable from the fit (on-policy states) |
| Correction round, 100 episodes | 7.3 to 12.5 | about 70 corrections, 3,250 rows (18 %) | as above |
| 100 more demonstrations on the six sheets | 9.4 to 13.6 | pictures x 1.48 | within 45 +1.0, stopped right +4.9 points |
| 210 more on the six sheets (doubling) | 19.7 to 28.5 | pictures x 2.00 | +1.7 and +8.7 |
| 4 new tracks x 20 demonstrations | 9.5 to 12.8 | pictures x 1.38, sheets x 1.67 | +2.6 and +6.5 |
| 6 new tracks x 20 | 14.3 to 19.3 | pictures x 1.57, sheets x 2.00 | +3.6 and +8.9 |
| Near-end starts for the stop | 25 per hour at the median cycle | one stop picture each | part of the stop gain above |

At the same robot time (about 10 to 13 hours), four new tracks are expected to add about 2.6 times the direction gain
of 100 more episodes on the six sheets, and a little more stop gain (calculated, frozen probe). 20 demonstrations per
new track is a judgement from section 2.1: half the episodes of each training sheet (about 8 to 19 training episodes
per sheet) already gave within 0.9 points of the full result.

## 5. Recommendation

**What to record more of** (from section 1, in order of expected value; judgement on measured weaknesses):

1. **The model's own off-line states** (puck 1 to 2 cm or more off the line): the demonstrations already hold 2,152
   moving pictures with the puck more than 2 cm off (946 of them inside hand-correction replays, 1,206 in first passes;
   registered good fits) [recovery_ablation_situations.out.txt], but those lie where the operators' own paths went;
   only correction rounds collect them where the model's errors lead. The benefit is supported by the literature
   (section 3.1), not yet by our own data (section 3.3).
2. **Loops and tight bends** (line radius under 5 cm): new tracks with tight bends in new places and both turning
   directions; on the six sheets, the two loop sheets have 30 and 25 episodes (sheets 02 to 04 have 45 each) and only 3
   and 4 near-end starts [timing_by_start_kind.out.txt].
3. **Line ends and stops**: near-end starts (2.4 min each), about 10 more on each loop sheet and 5 of the 20
   demonstrations on every new track (the share the 6.11 plan used: 42 of 210), and one new track whose line end lies
   far from the paper edge (the only way to test the paper-edge cue).
4. **Dashed lines**: one new track with dashed stretches of another rhythm, because sheet 06 is the only dashed sheet.
5. **Coverage only**: one crossing near 90 degrees and close parallel stretches (not a measured weakness, but outside
   the data). Not needed: more of the straight sheet 01; more natural light only if the exam is in daylight (open
   professor question 19).

**The plan** (robot steps only after the team decides, each with its own robot checklist; one program connected to the
robot at a time):

| Step | What | Robot hours (calculated) |
|---|---|---|
| A | If printing is allowed: 4 new training tracks x 20 demonstrations (5 of them near-end starts), plus 10 near-end starts on each of sheets 05 and 06. Needs no model, so it can run while the laptop trains the first model. | 10.3 to 13.9 |
| B | Baseline test of the first model (final candidate trained on all sheets): 5 trials per sheet, 2 trials on each of 10 sealed new tracks; pilot of 5 correction episodes to measure the take-over in model mode | 3.6 to 6.1 |
| C | Correction round 1, if the model finishes some runs (CR-DAgger's authors start at 10 to 20 percent success; speculation, literature): 50 episodes, about half on the six sheets (weighted to the failures of step B) and half on the new training tracks; version 1 of the take-over first, version 2 once built and tested on the simulated robot | 3.7 to 6.3 |
| D | Retrain from scratch on everything (laptop; corrections at 10 to 25 percent of each batch against plain mixing, chosen offline on held-out tracks and the correction pictures); paired, interleaved robot test of the new model against the first on the same starts | 6.4 to 11.0 |
| E | Only if step C helped and time allows: correction round 2, 30 to 50 episodes on the remaining failures | 2.2 to 6.3 |
| | Total | 24 to 37 (A to D); 26 to 44 with E |

The total compares with 28.5 h of sittings for the 210 demonstrations, recorded in four days (measured); spread over
the six weeks to 20.12.26 it needs the robot for about 4 to 7 hours a week (calculated). The 10 sealed tracks are used
only for tests, never for training or for choosing settings (evaluation_statistics checker: many distinct tracks with 2
to 3 trials each, because trials on one track share its difficulty). The correction episodes on the six sheets also
serve the known-sheet half of the exam: they come from new sittings, and the audit found almost no picture twins across
recording blocks of the same sheet (duplicates_leakage-3), so new sittings add views the old ones lack (judgement).
Laptop time per retrain: about 1 to 1.5 hours for one model on all data (calculated from the final candidate's 34.6
to 55.0 minutes per fold of about 15,000 training rows on the busy laptop) and about 5 to 6 hours for a six-fold check
(a guess).

**Answer for question 15**: yes, ask now. Our measurement (+2.5 points per doubling of different sheets on top of +1.7
per doubling of pictures) and the literature (Lin et al. 2025; Xie et al. 2023) both favour more different tracks over
more episodes of the six, and the unseen-track tests need printed tracks anyway. Ask together whether correction rounds
count as our own demonstrations (question 2 of the report's list; question 18 of the audit's list).

## 6. Backup

- **If we may not print our own tracks**: spend the hours on correction rounds on the six sheets (they add the
  on-policy states the offline numbers say are weakest), near-end starts on all sheets for the stop, the extra track
  files promised on 26.10.26 when they arrive, and more demonstrations of the loops and tight bends. Expect a smaller
  gain on unseen tracks: +1.0 point within 45 per 100 episodes on the six sheets against +2.6 for four new tracks
  (frozen fit, calculated).
- **If correction rounds are not allowed**: record deliberate recovery demonstrations instead, starting with the puck
  1 to 2 cm beside the line mid-track and guiding back; these are ordinary demonstrations (the data already hold 60
  offset or turned starts, audit table 9.1).
- **If the first model finishes almost nothing**: a take-over would turn into a full demonstration from an odd state;
  record targeted demonstrations of the failing situations first and fix the executor before a correction round.

## 7. What is still unknown

- How fast the end-to-end exact model improves with more sheets or more episodes (only the frozen probe was measured).
- The take-over time and the number of take-overs per run in model mode (the 51.6 s is from demonstrations).
- Whether corrections carry over to unseen tracks: no study measured it; only our sealed tracks can tell.
- Whether the stop relies on the paper edge: our check neither confirms nor rules it out; one printed track with a
  line end far from the edge would.
- Whether on-policy corrections help the off-line states more than the operators' own corrections did offline
  (section 3.3).
- The final candidate's folds for sheets 04 to 06, and how offline gains of 1 to 4 points translate into finished runs.
- How many robot hours per week are available until 20.12.26, and the professor's answers to questions 15 and 2.

## Sources and scripts

Scratchpad folder S = C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad;
runs folder R = D:/AI_Project/.claude/worktrees/training-analysis/04_runs.

- Audit: S/audit/AUDIT_SYNTHESIS.md (findings by id, sections 10 and 11); code audit: S/audit/code_actions/note.md.
- Runs: R/comparison.md; R/2026-11-09_resnet18_finetune_exact/metrics.json and fold_without_path_03.json, _05.json
  and _06.json;
  R/2026-11-09_resnet18_final_candidate/fold_without_path_01.json to _03.json;
  R/2026-11-09_resnet18_finetune_mirror/fold_without_path_03.json and _05.json;
  R/2026-11-09_resnet18_finetune_classes/metrics.json; R/2026-11-09_frozen_feature_probes/data_amount.json and
  designs.json (with the predictions folder).
- Literature: S/literature/interactive_correction/note.md and S/literature/interactive_correction/check/note.md;
  S/literature/diffusion_and_generative/check/note.md (section 4.2); S/literature/augmentation/check/note.md;
  S/literature/offline_vs_closed_loop/check/note.md; S/literature/evaluation_statistics/check/note.md.
- This section's scripts and their printouts, all in S/design/improvement/: data_amount_table.py (.out.txt),
  data_amount_regression.py (.out.txt), situation_errors.py (.out.txt, situation_errors.csv),
  situation_errors_by_sheet.py (.out.txt), stop_by_end.py (.out.txt, stop_by_end_episodes.csv), episode_timing.py
  (.out.txt, episode_timing_episodes.csv, episode_timing_corrections.csv), timing_by_start_kind.py (.out.txt),
  robot_hours.py (.out.txt), recovery_ablation.py (.out.txt, recovery_ablation/recovery_ablation.json and predictions),
  recovery_ablation_situations.py (.out.txt, recovery_ablation_situations.csv). They only read the project's files
  (episode_timing.py also reads D:/AI_Project/03_data/raw/main_data); no source, data or result file under D:/AI_Project
  was changed. Importing the project's modules made Python refresh its own bytecode caches in
  D:/AI_Project/.claude/worktrees/training-analysis/02_code/__pycache__ (ignored by git, rebuilt automatically).
