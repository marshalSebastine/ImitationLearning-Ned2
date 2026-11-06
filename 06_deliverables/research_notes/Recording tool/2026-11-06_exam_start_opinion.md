# How should the robot start on exam day? An opinion, not a decision

Written 6.11.26 by the Recording and data session, on Emranul's question of 5.11: "is it ideal to hardcode the initial
position for final testing? The professor will give us 4 to 5 sheets in very different orientations, and setting
coordinates for each sheet on the spot is a real issue. Can the robot find the initial position itself?" Emranul asked
for an opinion first; the team decides later, ideally after the professor's answers at meeting 2 (9.11.26).

Sources: three independent judgements (exam day, course rules, learning), the sheet measurements and the reach test of
6.11 (`2026-11-06_sheet_measurements.json`, `figures/2026-11-06_rule_a_placements.png`), the professor's answer to
question 9a (26.10: we may decide the start and the sheet's orientation, taping allowed), and the course text ("the
video stream will be the model's input and the robot movement will be the model's output").

## The short answer

- Do not type coordinates per sheet. Use one taped start mark on the table, about 17 cm in front of the robot's centre
  (8 cm in front of the middle of the base's front edge). Every sheet is laid with the start of its line on the mark and
  the first few centimetres of the line pointing straight away from the robot, then taped. The robot always goes to the
  same start pose behind the puck. Laying and taping a sheet takes about 1 to 2 minutes (guess); nothing is typed.
- The professor already allowed this (question 9a), so it needs no new permission.
- "The robot finds the start itself" is possible in two ways, and neither is worth it now:
  - a script that finds the puck with the camera and moves the rod behind it. Since 6.11 this is technically within
    reach (the puck detector and the calibrated camera place the puck within about 2 to 3 mm, 5 to 8 mm at worst), but a
    hand-written script that moves the robot goes against "the robot movement will be the model's output", so it needs
    the professor's explicit yes, and it adds things that can break on the day (other lamps, red objects in view, the
    taped camera creeping). It would also need the line's direction, not only the puck.
  - the model learns to approach the puck itself. This is the cleanest under the rules, but it is a second skill with
    its own demonstrations (about 1 to 2 minutes more per episode, about a third of the push data lost to it), and one
    mistake before the push ends the run. Only worth it if the professor demands it.
- Keep a fallback for a sheet that does not fit the mark: teach its start by one touch (motors off, touch the start,
  motors on; about 2 minutes). If we want it nicer, the tool can do this inside the run program and also learn the
  line's direction from a second touch (a small change; not built).

## Why one mark, and its limit

- The six provided sheets all fit the mark at 17 cm with the line leaving straight away from the robot, with under 1 cm
  to spare (calc, reach 13 to 44 cm, 3 cm safety band; sheet 04 is 0.3 cm short on paper but reached its goal on 5.11,
  episode 045). Sheet 05 must start from its lower end (end B) and sheet 06 from its dashed tail in the middle of the
  right side (end A); from the other ends their paper would lie under the robot's base.
- An unseen A4 sheet whose line runs more than about 24 cm away from its start does not fit when laid this way. It can
  be turned so the line leaves at an angle (then almost any mark works: 655 of 783 tested marks fit all six sheets), but
  then the first push is not along the line. The training data therefore include starts with the sheet turned by 10 to
  15 degrees (offset starts in the episode plan); larger angles would need a push direction per sheet in the tool
  (`--start-push-deg`, a 15-minute change, not built).
- The model sees the same start picture in almost every episode, so the first pushes become its best-learned part.

## What this means for the data (nothing has to wait)

Record every episode from the standard start (rod just behind the puck, puck on the start cap of the line, hands out of
the picture before Enter). That start is the same for the fixed mark, the taught start, a hand-placed start and a
script-found start, so the data stay useful whichever the team chooses. Do not record approach episodes yet.

## Questions for meeting 2 (9.11.26), for the orchestrator to add to `05_meetings/requests_to_professor.md`

1. On the evaluation day, do we lay out every sheet ourselves, the unseen ones included (which end is the start, where
   it lies on the table, taped down)? On 26.10 you said we may choose start and orientation; does that hold for unseen
   tracks?
2. Before each run, may the robot make one fixed, scripted move to the same start pose, with the pusher just behind the
   puck at the line's start, after which only the trained model drives? Or must the model also bring the pusher to the
   puck?
3. Instead, may we put the pusher behind the puck by hand (motors off), or touch the line's start once with the pusher so
   that a script moves the robot there?
4. May a hand-written red-colour detector find the puck in the camera picture and move the pusher behind it before the
   trained model takes over?
5. How many unseen sheets are there, when are they run, how much set-up time do we get per sheet, and when does the clock
   start and stop?
6. We record demonstrations by keyboard, by gamepad and by guiding the arm by hand with the motors off, after which the
   robot replays the guided path while the camera records (only the replay pictures are used). Do all three count as
   our own demonstrations?

## How the three judgements ranked the options

| Option | Exam day | Course rules | Learning |
|---|---|---|---|
| A. One taped start mark, fixed start pose | 2nd (fastest; needs B as fallback) | clearly allowed (9a) | 1st |
| B. Start taught by one touch per sheet | 1st (after a small tool change) | grey: ask | 3rd |
| C. Pusher placed by hand, motors off | 3rd | grey: ask | 2nd |
| D. Model learns the approach | 5th | clearly allowed | 5th |
| E. Script finds the puck with the camera | 4th | needs an explicit yes | 4th |

My recommendation combines A with B as the fallback. It is an opinion: the team decides, best after meeting 2.
