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
- Open, Emranul's decision: episode 14-02 (his note: "skipped a hairpin", score 2) is still in the training data. One
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

State of 14.11.26, 22:49 (the first queue is still running):

| Run | Folds | Path within 45 | Stopped right | Too early | Never | Right on mid-paper ends |
|---|---|---|---|---|---|---|
| reference (round 1 plus all of round 2) | 16 of 16 | 0.832 | 11 of 55 | 16 | 25 | 0 of 19 |
| the same recipe on round 1 only | all tracks | 0.809 | 14 of 55 | 20 | 19 | 1 of 19 |
| reference, second seed | 5 of 16 | 0.831 (reference on the same 5: 0.808) | 5 of 16 (3 of 16) | 3 | 7 | 0 of 6 |

(3 more of the reference's 55 episodes stopped 2 to 3 cm short of the end, neither right nor early.)

Read so far:

- Steering: round 2 adds 2.3 points of path within 45 on unseen tracks (0.809 to 0.832). The second seed's first five
  tracks lie 2.3 points above the first seed's on the same tracks, so one seed alone moves the number by about as much
  as round 2 did; the full second seed is needed before more is said.
- The stop is the weak point and round 2 did not move it: right on 11 of 55 unseen-line episodes with round 2, 14 of 55
  without. On the six tracks that end in the middle of the paper it is right on 0 of 19, with 15 other new tracks in
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
