# The search for the final model (from 14.11.26, 22:00)

Asked for by Emranul when round 2 was recorded to the end: "make sure you are giving the best effort and you are
reaching the ceiling ... start with data preprocessing and cleaning, don't compromise on anything". The professor said
the data is enough; Emranul and Marshal are open to record more if this work shows that it helps.

What is measured, calculated or guessed is said in each part. Results are filled in as the runs come back from ki-014;
the table in part 5 says how many of the 16 folds each number rests on.

## 1. The data

- Round 2, recorded by Emranul on 14.11.26: 55 scored episodes on the 16 new tracks T01 to T16 (3 or 4 per track; solid
  and dashed prints), 4,748 pictures. Scores: 14 with 3, 27 with 2, 10 with 1 (score 0, 14 folders, is left out as in
  round 1). One plan number more than planned (16-03, a near-end start that worked).
- Index of round 1 plus all of round 2: `03_data/processed/v03_training_index` (266 episodes, 23,270 pictures;
  `training_dataset.py`, round 1's and the mid-way index stay as they are). After the audit cleaning of 9.11.26, which
  applies to round 2 by the same rules (row 0 and the row after a hand correction dropped, stand-still rows before the
  stop relabelled as stop, mistake rows dropped): 264 episodes, 22,646 training pictures, 4,614 of them on new tracks.
- A second picture cache at 320 x 240 was added to the index (`training_dataset.add_image_cache`); shrunk back to 160 px
  it differs from the 160-px cache by at most 0.4 of 255 on five sampled pictures (measured).

## 2. Cleaning check of round 2 (measured; scripts in the session's scratchpad, results quoted here)

- No unreadable picture, no MD5 mismatch; no step longer than 2 cm (longest 2.0 cm); no picture where the puck was
  lost; 13 rows marked as mistakes by the operator (dropped by the usual rule).
- 136 flagged pictures: 126 "person / shadow or hand?", 10 "no puck seen", a few with the laptop in view. Kept, as in
  round 1 (`is_flagged_dropped` False).
- The last picture of all 55 episodes was looked at: in each the puck sits on the line's end, no line shows beyond it.
- The rod tip's path was drawn over the planned line for every episode. The tip runs 1 to 2 cm beside the line (median
  1.3 cm) and swings wide at bends and when it is moved round the puck; that is how the pushing works, not an error.
  A skipped stretch cannot be read from the tip's path at a hairpin (two automatic measures were tried and dropped).
- Decided by Emranul on 14.11.26, 22:55 ("remove episode 14-02 as it skipped the thing"): 14-02 gets no weight in the
  training of every run from the second queue on, and of model 10; the recording stays on disk. Before that: one
  episode of 264 cannot be judged by the yardstick; leaving it out is a decision of principle.
- Score-1 episodes (10 in round 2) are tested as a question of their own ("score_1_off", part 4).

## 3. The yardstick and its rule (fixed in `config.py` before the results)

Each of the 16 new tracks is held out once: trained on round 1 and the other 15 tracks, tested on both prints of the
held-out track (55 test episodes on lines the model has not seen), because the exam track is unseen. Measures: path
within 45 (share of path pictures whose predicted direction is within 45 degrees of where the line goes) and the stop
(fixed threshold 0.5 on two pictures in a row: right within 2 cm of the demonstrated stop, early more than 3 cm before
it, never). Reference: model E's recipe (chunk, stop weighted 25, ResNet-18, 8 rounds), two seeds.

Rule: a change of one thing is kept when it stops right on at least 5 more of the 55 episodes than the reference's mean
without losing more than 1 point of path within 45, or gains 1.5 points or more of path without stopping right less
often than the reference's worse seed. A difference smaller than the gap between the two seeds is no difference.

## 4. What is trained

First queue (started 22:21 on ki-014, three jobs at a time): model 9 (the reference recipe on all 264 episodes, for the
robot); the reference, two seeds; the same recipe on round 1 only; then one change each: two pictures, score-1 episodes
without weight, ResNet-34, 16 rounds.

Second queue (prepared, starts with the second login; put in front of what is left of the first):

| Run | What changes | Why |
|---|---|---|
| table_as_paper | in half of the training pictures the table is repainted in the paper's colour | an end near the edge then looks like an end in the middle of a sheet |
| made_up_ends_4, _10 | in 4 % or 10 % of the training pictures the line that leads on from the puck is painted over and the picture becomes a stop | every stretch of every track gives examples of a line ending under the puck |
| mid_paper_stop_4 | the stop pictures of the six mid-paper-end tracks count four times | they are 19 of about 260 episodes |
| made_up_ends_and_table, table_and_mid_paper | two of the above together | |
| 12_tracks, 8_tracks, 4_tracks | the reference trained on round 1 and only that many of the other new tracks | would more tracks still help? Reading rule in `config.py` |
| wide_320 | pictures of twice the width and height | are a dash, a gap or the line's end lost at 160 px? |

A made-up end is not a demonstrated stop: if it is kept, the report has to say so, and whether the course's rule
("imitation learning only") allows it is a question for Emranul and the professor.

## 5. Results (measured; `evidence/final_2026-11-14/score_final.py`, numbers in `final_scores.json`)

State of 14.11.26, 23:10 (both seeds of the reference complete; everything else still running):

| Run | Folds | Path within 45 | Stopped right | Too early | Never | Right on mid-paper ends |
|---|---|---|---|---|---|---|
| reference, seed 1 (round 1 plus all of round 2) | 16 of 16 | 0.832 | 11 of 55 | 16 | 25 | 0 of 19 |
| reference, seed 2 | 16 of 16 | 0.844 | 17 of 55 | 13 | 22 | 1 of 19 |
| the same recipe on round 1 only (seed 1) | all tracks | 0.809 | 14 of 55 | 20 | 19 | 1 of 19 |

(3 more of seed 1's 55 episodes stopped 2 to 3 cm short of the end, neither right nor early.)

The reference's mean, against which every change is judged: path within 45 0.838, 14 of 55 stopped right. The gap
between its two seeds: 1.2 points of path and 6 right stops. By the rule a difference under that gap is no difference,
so a change has to stop right on 20 or more of the 55 (5 above the mean and outside the gap), or gain 1.5 points of
path without stopping right on fewer than 11.

Read so far:

- Steering: round 2 adds about 3 points of path within 45 on unseen tracks (0.809 to a mean of 0.838; both seeds lie
  above the round 1 model, by 2.3 and 3.5 points). The round 1 model has one seed only.
- The stop is the weak point and round 2 did not move it: right on 11 and 17 of 55 unseen-line episodes with round 2
  (two seeds), 14 of 55 without. On the six tracks that end in the middle of the paper it is right on 0 and 1 of 19, with 15 other new tracks in
  training (5 of them ending mid-paper).
- The model's own look-ahead does not know the end either (`evidence/final_2026-11-14/chunk_stop_check.py`, no new
  training): stopping when the predicted reach of the next five steps falls under a limit gives 13 right of 55 and 0 of
  19 on mid-paper ends; lowering the stop output's limit to 0.3 gives 15 right but 27 too early. So the end of a line in
  the middle of the paper is not seen at all by this recipe; a better threshold does not fix it. The three changes of
  the second queue are aimed at exactly this.

## 6. How the runs are started and fetched

One login per queue (Emranul types his OTH password; VPN on): the laptop sends the code and the picture cache through
`tar | ssh`, the queue runs on ki-014 under `nohup` (it goes on when the connection ends), and a small script on ki-014
sends every finished result file back through the same connection once a minute (`final_stream.sh`; read with
`tar -x -i`). Results land in `04_runs/gpu_ki014/2026-11-14_resnet18_full_*` (not in git).

Second login at 22:58 (5.4 GB uploaded in 10 minutes): the first queue's runner was stopped at 23:08 with three jobs
running and 13 not started; the new list has 50 jobs (model 10; the six runs against the edge cue; what was left of the
first queue; 12, 8 and 4 tracks) and then four jobs at 320 px, two at a time (62 GB of memory, 43 free). Expected to
finish at about 05:00 on 15.11.26 (calculated from 14 minutes per four-fold job). The scripts of both logins and a
fetch-only login are kept in `evidence/final_2026-11-14/ki014_scripts/` (their paths point into the session's
scratchpad, which may be gone; the fetch-only login needs nothing but ki-014's own files).

## 7. First look at the robot, night of 14 to 15.11.26 (run list `2026-11-14_night_look_runs.csv`, logs in `03_data/model_runs/`)

Model 10 is the reference recipe on all 264 episodes without 14-02's weight (sha256 a47982c26cdf). Whole lines, fixed
0.65 cm steps. What the operator saw is marked (Emranul); the rest is read from the run logs.

| Run | Sheet, lay-out | Model | Result |
|---|---|---|---|
| S01 | 01, standard | 10 | stopped by itself after 44 steps; the puck 1 cm left of the end's centre (Emranul). On 12.11 the round 1 model stopped 2.1 to 2.7 cm left |
| S02, twice | 02, standard (plan 02-03's faint picture) | 10 | on the line until about step 51, then to the right of it (Emranul); ended at the rod's lean limit (59.7 against 58 degrees). The second time with nothing else in the camera's view |
| SE2 | 02, standard | round 1 model E | the same failure at the same place |
| S2L | 02, landscape (plan 02-37's faint picture) | 10 | "differed a lot", better, not finished (Emranul); stopped by itself after 51 steps, the rod at 42 degrees |

Measured from the logs and the recordings: in the three failed runs the rod's tip was on the line until 43 cm from
the base and then went out to 48 cm. Of the 22,646 recorded pictures 90 % have the tip within 40.3 cm, 4.9 % beyond
42 cm and 2.0 % beyond 44 cm. The recordings on sheet 02 in the standard lay-out stop at a median 40 cm (after 41 to
43 cm at the arc's far point); lying landscape they stop at 26 to 30 cm. Two guesses were checked and dropped: the
laptop and an arm in the camera's view (the repeat failed with a clear view), and the wrist held by the driver (the
recordings keep joint 5 near the same 13 degrees). Why the model steers right at that place is not known; the reach
is what the failed and the working lay-out differ in. An earlier reading of S01 from the log ("5 cm past the end") was
wrong: the log knows the rod's tip and the planned sheet place, not the puck on the real sheet.

Consequence for the exam, where the team lays every sheet (Emranul: there will be no time to try lay-outs): one rule
(start cap on the mark, then turn the sheet until the whole line is as near the robot as it can be, no part more than
about 43 cm from the base's centre; the 43 is calculated, not tried) and an aid on the table (tape or a string of that
length). The six known sheets can be decided beforehand on a card.

## 8. Third queue (15.11.26, behind the second; Emranul: "you are not utilising the data to its full potential")

One thing changed each, all without 14-02: the zoomed-in view (a 320 x 240 cut-out of the camera picture round the
puck, shrunk to 160 x 120), pictures turned within 12 degrees and mirrored pictures (Emranul's ideas), far-reach
pictures counted three times, ConvNeXt-Tiny and ResNet-50, a third seed of the reference; then three combinations fixed
before any single result was known (zoom with a second seed; zoom with turned and mirrored pictures; ConvNeXt-Tiny on
the zoomed-in view). Same rule as before. Every run type was trial-run for one round on the laptop.

## 9. Does the model look at the puck? (15.11.26, 01:30; `evidence/night_look_2026-11-14/puck_test.py`)

Emranul at the robot, after the loop (sheet 05) went worse with model 10 and 0.65 cm steps than with the round 1 model
and 1 cm steps on 11.11: "the model is taking decisions on what is ahead, not considering the position of the red puck".
Tested on the camera pictures of both loop runs (tonight's S05, 112 pictures; the run of 11.11, 81 pictures): the puck
is painted over, moved 40 pixels left or right, or the line is painted over, and each model is asked again
(`puck_test_examples.jpg` shows the changed pictures). Measured, median change of the predicted direction:

| Pictures | Model | Puck hidden | Puck 40 px left (mean shift) | Puck 40 px right (mean shift) | Line hidden |
|---|---|---|---|---|---|
| S05 (15.11) | 10 | 28.5 | 25.6 (+6.2, to the left) | 34.3 (-40.8, to the right) | 12.0 |
| S05 (15.11) | E | 42.0 | 19.3 (+15.9) | 25.8 (-34.1) | 16.6 |
| run of 11.11 | 10 | 75.6 | 22.2 (+9.0) | 34.1 (-39.7) | 22.4 |
| run of 11.11 | E | 91.7 | 34.2 (+25.5) | 29.7 (-32.8) | 22.9 |

Both models follow the puck: move it and the direction goes after it. Hiding the line changes the direction about half
as much or less. So the model does consider the puck; what it uses weakly is the line. That fits what was seen on the
robot: once the puck is beside the line, the model keeps pushing it where it is and is only weakly drawn back to the
line. The two models do not differ clearly in this. Whether model 10 is worse on the loop than the round 1 model is
not settled by one run each with two things changed (model and step); run lines SE5F, S05F and SE5 separate them.
A first version of this test had masks that left the puck and the line half visible; its numbers were thrown away.
