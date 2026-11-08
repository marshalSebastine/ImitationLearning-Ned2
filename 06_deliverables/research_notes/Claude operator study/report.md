# Can Claude be a third operator? Offline check, 8.11.26

Question (Emranul, 8.11.26): can Claude look at one forearm-camera picture and name the next move, the way Emranul
and Marshal do in `hand_guided_exact`? Done offline only. No robot was used, and no data was changed.

## How it was tested

- 90 pictures from 18 episodes (6 each from path_01, path_02 and path_03, scores 02 and 03, both operators). From each
  episode: the start, about 35 % and 65 % of the way, one step within 3 cm of the goal, and the stop step.
  `02_code/claude_operator_study.py select` picked them. It hid the labels in a file Claude did not open until
  after predicting.
- Claude looked at each picture in a shuffled order and named one move (and a series of 3 moves for every third
  picture). Then `... score` compared the answers with the operators' labels. Everything is in `test_set.csv`.
- "Close" means a moving action within 60 degrees of the label, for example forward and forward_left. The baseline
  always answers "forward".

## Results

26 of the 90 labels are `small_motion`. The tool writes this label when the tip moved less than 3 mm. It is on all
18 start pictures, where the first step mostly lowers the arm. Nobody can pick it from a picture, so the main numbers
leave those 26 out (64 pictures).

| 64 pictures | Claude exact | Claude close | "always forward" exact | "always forward" close |
|---|---|---|---|---|
| all | 55 % | 72 % | 33 % | 50 % |
| stop pictures (18) | **100 %** | 100 % | 0 % | 0 % |
| moving pictures only (46) | **37 %** | 61 % | **46 %** | 70 % |
| bends (23) | 22 % | 65 % | 35 % | 74 % |
| near goal, not yet stopped (16) | 44 % | 50 % | 50 % | 63 % |

- Per sheet (exact): path_01 67 %, path_02 48 %, path_03 50 %.
- Series of 3 moves from one picture (30 pictures): first move right 47 %, second 30 %, third 32 %. The whole
  series was right 27 % of the time.
- Time: about 2.5 s per picture, reading and deciding in one batch. A live call from the tool (sending the picture,
  waiting for the answer) would add an unknown few seconds. This was not measured.

## Where Claude fails

1. **It gets every stop, but it stops too early.** In 10 of the 72 pictures that are not stops, it said "stop". All
   were within 3 cm of the goal, where the operators were still pulling back or nudging sideways (back, back_left,
   back_right, right). From one picture, "nearly there" and "there" look the same.
2. **Bends.** Claude reads the line's direction ahead, but the operators' labels come from how the hand moved the block
   around the puck. It often got the side or the size of the turn wrong: forward_left for forward_right, or a diagonal where the
   label was plain forward.
3. **Moves that depend on what came before.** Operators sometimes move sideways to get behind the puck, or turn the
   forearm, and a single picture does not show why. Some pictures show only the table, hands, or no puck at all.
4. **Possible hint, to be fair.** Before predicting, Claude listed the episode folders. That list showed frame counts,
   and the picture names show the step number (000 = start). Claude did not consciously use either, but a cleaner
   test would hide both.

## Verdict

**Not worth adding as an operator now.** On the moves that matter (the 46 moving pictures), Claude is worse than
always answering "forward" (37 % against 46 %). The only thing it does well is spotting the final stop, and it pays
for that with early stops. A trained model will learn from the same pictures and labels anyway. Claude's guesses
would add noise in a different style from Emranul's and Marshal's.

**Open question for the professor:** the course allows imitation learning only, from the team's own demonstrations.
Claude-made demonstrations may not count as "our own" at all. Please ask before using any, even if a later version
gets better. A safer use of Claude would be as a checker after recording, flagging pictures whose label looks odd
(for example "stop" with line still ahead), without making labels.
