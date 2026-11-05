# Side-view test for pulling the puck (for Emranul, written 4.11.26)

**The idea (Marshal, 4.11.26):** when the puck is pulled towards the base, the block hides the puck and the path ahead is
out of view. Turning the forearm (joint 4) as far as its limit allows gave Marshal a side view of the pull path. This test
checks whether that view is usable. It records nothing for training.

**The catch:** the rod leans when the forearm is turned, and no joint can fully correct it, unless the forearm itself is
vertical. The script keeps the magnet tip on target and the lean as small as it can.

| Tip distance from base | Least lean, tip down, 90 degree turn (calculated) |
| --- | --- |
| 20 cm | 1.5 degrees |
| 25 cm | 9.5 degrees |
| 28 cm | 14 degrees |
| 32 cm | 20 degrees (the script refuses this, limit 16) |

Measured: nothing yet with the forearm turned. The arm model matched a robot reading of 30.10 with the forearm straight.
Calculated or guessed: all the lean numbers, the 16 degree limit, the safe clearances with bracket F and its cable.

## What the script protects, and what it cannot

- Before anything moves, it compares every planned position with the robot's own kinematics, and stops if they differ by
  more than 3 mm. It also refuses too much lean.
- It lifts before turning, and turns in 30 degree increments with the tip held in place.
- On Esc, `x`, Ctrl-C or an error, it lifts first, turns back in increments, then goes home.
- **It cannot see collisions.** The camera, bracket and cable are not in its model, so watch them on the first turn.

## Stop rules

- Stop button in reach, and a second person if possible. Press it for any contact, cable pull, strange noise, or a move
  you did not expect.
- Never touch the arm while the motors are on, and never push on a held arm. Do not touch the camera.
- The robot stays away from the table edge. Only one program connected: close NiryoStudio and any viewer first.

## Before you start

- [ ] `git pull` on main. In `02_code`: `..\.venv\Scripts\python.exe test_side_view_test.py` says "all tests passed" (under a minute).
- [ ] Ethernet connected, robot calibrated by the **automatic** calibration only. `check_hardware.py` says "Calibration needed: False".
- [ ] Bracket F on, magnet block on the rod, sheet 01 taped in place, sheet clear. **No puck on the sheet.**

## Part 1: the view, no puck (about 15 minutes)

```
..\.venv\Scripts\python.exe side_view_test.py --operator emranul --part view
```

All keys go to the picture window: click it first. Enter goes on, Esc ends the test.

- [ ] The terminal prints "plan checked" for 20, 25 and 28 cm. If it says the model and the robot disagree, **stop** and send
      that message to Marshal: nothing has moved.
- [ ] First turn, tip lifted: watch the camera, bracket and cable. If the cable pulls or anything comes close, press Stop.
- [ ] At each distance, press Enter to come down. Check:

| Distance | Tip clear of the paper, about 5 mm? | Puck spot and path towards the base in the picture? | Bracket or arm close to anything? |
| --- | --- | --- | --- |
| 20 cm | | | |
| 25 cm | | | |
| 28 cm | | | |

- [ ] Look at the saved pictures (the terminal names the folder). If the view is no better than the normal one at 20 cm,
      stop here: that answers the question.

## Part 2: pulling with the puck (about 20 minutes)

Only if part 1 looks good. Start at 25 cm, then 22 cm (`--distance 0.22`), then 28 cm (`--distance 0.28`).

```
..\.venv\Scripts\python.exe side_view_test.py --operator emranul --part pull --distance 0.25
```

1. Press Enter: the tip comes down with no puck on the sheet.
2. The robot waits. Set the puck on the line on the **base side** of the tip, touching it. Do not touch the arm.
3. Press Enter to start, then Enter for each 1 cm pull, one at a time. `x` stops; the tip lifts first.

**Main risk to watch:** when pulling, the wrist sits nearer the base than the tip, so the rod leans over the puck. That could
move the contact higher up the puck and tip it over (calculated about 9 mm over the puck's top at a 20 degree lean).

Repeat five times per distance. The pull stops by itself at 17 cm from the base.

| Start | Run | Steps before the puck slipped or tipped | Puck stayed on the line? | Puck visible in every picture? |
| --- | --- | --- | --- | --- |
| 25 cm | 1 to 5 | | | |
| 22 cm | 1 to 5 | | | |
| 28 cm | 1 to 5 | | | |

## How to read the result (suggested, for Emranul and Marshal to change)

- **Worth building into the recording tool:** the puck and the path ahead are visible, and at least 4 of 5 pulls finish with the
  puck on the line. The furthest distance where this holds sets where the side view may be used.
- **Not worth it:** the puck tips or slips within a few steps, or the view shows nothing the normal one does not.

## Afterwards

- Push your notes, and copy the folder `03_data/raw/camera_check/side_view_test/` to Marshal: `03_data` is not in git.
- Nothing is built into the recording tool yet. The tool's direction keys and the step-over assume the normal camera position.
