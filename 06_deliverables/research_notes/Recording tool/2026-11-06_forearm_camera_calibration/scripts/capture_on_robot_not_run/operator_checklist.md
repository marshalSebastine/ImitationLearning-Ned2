# Forearm camera calibration capture: operator checklist

For `02_code/calibrate_forearm_camera.py` (written 6.11.26, tested on the simulated robot only, **never run on the real
robot yet**). It runs only when Emranul decides to. One person at the robot, one at the laptop is easiest, but one
person can do it alone.

**What it does.** The robot holds its magnet block 1 cm above an empty spot. You slide the puck under it. The robot then
rises and photographs the puck from 12 arm poses (forearm turned up to 60 degrees each way, block always at least 3 cm
above the puck). That happens at 5 spots. The pictures and the arm's joint angles let us fit where the forearm camera
looks, so the orientation window can show the real puck. Nothing it records is training data.

**How long.** About 3.5 minutes of planning on the laptop (nothing moves), then about 15 to 20 minutes with the robot.
That second figure is an estimate: 292 moves at 15 % speed. The simulator does not time moves. You are needed 5 times,
for about half a minute each.

## Before you start

1. The robot stands on the table you will use, and **the table plane and workspace in `config.py` were taught on this
   table** (`teach_positions.py`). If the robot was moved since the last teaching, teach first. A wrong table plane
   makes every height wrong.
2. Run the plan check (no robot, nothing moves):
   `D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/calibrate_forearm_camera.py --operator emranul --plan-only`
   It must end with **`0 failing a check`** and **`0 skipped`**. If not, stop and ask Claude.
3. The robot did its **automatic calibration** after switching on (the script refuses otherwise). Never calibrate by hand.
4. **Only one program may be connected to the robot.** Close the recording tool, any viewer and Niryo Studio.
5. The camera is on its forearm mount as in `forearm_mount/README_forearm_mount.md`. **Do not touch the camera or its
   focus ring.** If you are unsure whether it moved, record one recording-tool episode first: its camera check should
   say under 4 px.
6. Clear the table in front of the robot: no cups, no cables, **no red objects** (the mouse with the red wheel goes
   away). Tape an empty white sheet flat in front of the robot, long side pointing away from the base, from about
   15 to 43 cm out, centred on the line straight out from the base.
7. Same lamps as for recording. Robot's **STOP button in reach**. **Keep the puck in your hand.**

## Run

8. Type in the Terminal panel:
   `D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/calibrate_forearm_camera.py --operator emranul`
   It plans for about 3.5 minutes, then compares the arm model with the robot (nothing moves). Then the picture window
   asks: **Enter = start**. The robot goes to its parking pose first.
9. **At each spot** the robot comes down, stops, and the window says `SPOT n: slide the puck under the block`:
   - Look at the gap between the block and the paper: about 3.9 cm. **Do not put your hand under the block before the
     window asks.**
   - Slide the puck under the block, centred by eye. Look once from the front and once from the side. The block's
     long sides (37 mm) should stick out equally past the puck at both ends. Along the short side (13 mm), the puck
     should stick out equally on both sides. There should be a gap of about 1 cm between the block and the puck's top.
   - **Take your hand away**, then press **Enter** in the picture window.
   - `s` = skip this spot (for example something is in the way). `Esc` = finish now.
10. Then the robot works alone for about 3 minutes per spot (estimate): up, 12 poses, a picture at each. Stand back
    and watch it. Each line in the terminal says where the block is and whether the puck was seen. The spots are, in centimetres from the base
    (x out, y to the left seen from behind the robot): (22, 4), (22, -6), (30, -1), (37, -6), (37, 4).
11. After spot 5 the robot goes to its parking pose. The last line says `run completed: 60 pictures, puck seen live in
    N`, and a camera-check line says how many pixels the hand moved between start and end.

## Stop rules (any of these: stop)

| What you see | What to do |
|---|---|
| The block touches the puck, or the gap at the hover is clearly under 5 mm or over 2 cm | **Esc** (the table plane is off; teach it again) |
| The puck is pushed, tips or moves while the robot works | **Esc** |
| The camera cable pulls tight or catches when the forearm turns | **Esc** (the camera must not move on its mount) |
| The arm moves somewhere unexpected, or towards a person, the table edge or a cable | **robot STOP button** |
| A strange sound, or the robot stops with an error light | **robot STOP button** |
| The terminal shows `ERROR` or `refused` | nothing: the script rises and goes home on its own; read the line |
| `the live finder saw the puck in` fewer than 6 of 12 at a spot | let it finish, then tell Claude |
| The camera-check line says `THE CAMERA MAY HAVE MOVED` | keep the files, do not record, tell Claude |

- **Esc** works when the picture window is in front. **Ctrl-C** works when the terminal is in front. Both make the
  robot rise straight up, turn the forearm back up high, and go to the parking pose. A second Ctrl-C stops even that,
  and the arm stays where it is with the motors on.
- If the terminal says **`THE SAFE RETREAT STOPPED`**, the arm stays where it is with the motors on. Do not push it.
  Hold the arm first, then switch the motors off with the robot's FreeMotion button. Guide it up and away from the
  puck by hand, then park it with the robot's own controls.
- Never touch the arm while the motors are on, apart from the puck slide in step 9 (that touches the puck, not the
  arm).

## After

12. Files: `03_data/raw/camera_calibration/<date_time>/`. You get `captures.csv`, `spots.csv`, `moves.csv`, `log.txt`,
    `calibration_run.json`, and `pictures/` (60 survey pictures and 5 hover pictures). Nothing in it is edited
    afterwards.
13. Open two or three pictures: the puck should be visible above the hand. Then tell Claude the folder name. The fit
    is the next step.
