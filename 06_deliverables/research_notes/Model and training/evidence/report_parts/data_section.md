## 2. The data: what the audit found

Ten auditors each checked one side of the data, and two more agents re-checked every finding: one recomputed the
numbers, often from the raw files, and one attacked the conclusions. Of 165 findings, 61 were confirmed as they stood,
102 were corrected in a number or a wording, 1 was refuted and 1 could not be tested. The full, checked text is in
`evidence/audit_synthesis.md`; the numbers below are from it unless marked otherwise.

**The bookkeeping is sound.** Each label belongs to the right picture: the label of a row is the move from its picture
to the next one (0.09 mm median error, measured). Every picture file still has the MD5 it was logged with (measured
when the index was built), and the recording tool's direction rule is reproduced on more than 99.9 percent of the
moving rows (test_training.py).

**The labels are noisy, and the noise is in the hand guiding, not in the code.**

- The 8 move names cut a continuous direction into uneven bins (forward covers 60 degrees, left and right 37.5). 18.8
  percent of the moves lie within 5 degrees of a bin edge, 38.1 percent within 10.
- From one 1 cm step to the next the direction changes by a median 21 degrees, and consecutive move names differ 55
  percent of the time.
- Near-identical pictures from different episodes of the same sheet carry different move names in 26 percent of the
  moving pairs (8 percent differ by more than 45 degrees).
- The forearm twist that comes with most steps is not in the move name: 56 percent of the non-stop rows twist by 3
  degrees or more, and the names keep only 16 percent of the twist's variance (against 81 to 85 percent of the tip
  move's).

**"small_motion" is three different things** (2,265 rows, 12 percent):

| Part | Rows | What it is | What to do |
|---|---|---|---|
| Row 0 of every episode, and the first row after every hand correction | 449 | The robot did not move; the picture is almost the next one, sometimes with a hand in it | Drop |
| The point just before the stop | 51 | A duplicated final waypoint (record_episodes.py line 1120); its picture is the stop picture | Relabel as stop |
| The rest | 1,765 | A forearm twist with the tip still, in both directions | Keep; the twist output learns it |

**Other fixes the audit found** (all measured): two episodes have their score only in review.json (01-09 try 1 and
02-41 try 1, both score 2); the start end is wrong for 30 episodes in the index (19 on sheets 05 and 06, 11 on sheet
02), so the audit's table `evidence/corrected_ends.csv` replaces it; the half run 05-16 and rows 169 to 175 of 04-01
try 6 (the puck pulled back by hand) are dropped; 25 steps over 2 cm and four very large twists come from recording
stalls and are clipped. With all rules, 469 rows are dropped and 18,053 rows in 209 episodes remain, with 259 stop
rows (`training_targets.audit_cleaned_frames`, checked to give exactly these counts).

**What should not be filtered:** the 520 "dark object" flags (mostly shadows and the printed line), the 524 "no puck"
flags (today's detector finds the puck in 377 of them), the 17 blur flags (none is blurred), score 1 (not
distinguishable from score 2 in motion, and it holds many of the hardest starts), the 24 dim natural-light episodes
(2,817 pictures, the only daylight examples), and the hand-correction replays (37.9 percent of rows, recovery data).

**Leakage decides how to validate.** 61.5 percent of the pictures have a near-twin in another episode of the same
sheet, but only 2.8 percent on another sheet, and the twins belong to the recording block: episodes recorded within 15
minutes of each other share many views, episodes recorded hours apart almost none. A "model" that only copies the
label of the most similar training picture, which cannot generalise at all, errs by a median 27 degrees when whole
episodes are held out and by 53 degrees when a whole sheet is held out, worse than always pushing forward on five of
the six sheets. So a score on held-out episodes measures memory of the recording block, and is optimistic even for a
known sheet laid out again in a new session, as at the exam. Only holding out a whole sheet measures following an
unseen track. The independent units are few: about 1,200 to 1,700 independent views of picture content, about 3,100
independent labels, 210 episodes and 6 sheets.

**Things that are tied together.** Sheet, recording day, light, operator and camera period go together, so no split
separates them. Natural light occurs only on sheets 02 and 04. The taped camera's view moved once, during the break
on 8.11.26 between 18:07 and 19:33 (about 15 pixels, mostly sideways, at 640 x 480; the cause is not known): sheets 05
and 06 and the last 16 episodes of sheet 04 are in the later view (71 episodes, 6,435 pictures). On sheet 04 every
end-B episode is in the later view and every natural-light one in the earlier.

**What the camera sees at the stop.** Every line end lies 2.35 to 8.03 cm from a paper edge, and the stop pictures show
no line ahead. A model may learn "stop near the paper edge" instead of "stop at the line's end". An unseen track
whose line ends far from the edge would expose this, so one printed test track should have such an end.

**How the robot can carry out a model's output** (code-reading audit, checked). The demonstrations are in effect a
four-joint motion: the tip's position plus the forearm twist, with the hand (joint 6) almost still and the wrist (joint
5) drifting slowly. An executor that keeps the hand, follows the wrist and solves the other joints for the tip shows
the camera the views it saw in training. The keyboard and gamepad paths of the recording tool choose the least rod
lean instead and would move the picture's content by a median 48 to 56 pixels: views the model never saw. The
demonstrations also reached the forearm's joint limit (plus or minus 119 degrees) and left the configured workspace
box in 7.7 percent of the tip positions, so the executor needs clamps, a workspace taken from the sheet in use, a
fallback after a refused step, and the step budget switched on.
