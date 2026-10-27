# Mock-up session at the robot (checklist step 5): the list to follow

Written 27.10.26 for Emranul and Marshal. This is checklist step 5 of `checklist_camera_mount.md`: before anything is
designed or printed, a cardboard block of the camera's real size is held where the printed bracket would put the camera, and
we check two things: does it hit anything, and does the camera from there see the puck and the line? Time limit: 30 minutes
at the robot. Two people: one at the arm, one at the laptop and the sheet.

**The position being tested is calculated, not proven.** The design review (`design_review_one_print.md`, option F) puts the
lens about 5 cm ahead of the rod axis, 6 cm above the wrist axis (joint 5), tilted about 10 degrees back towards the robot.
The same review predicts that the camera body touches the forearm at the nearest pushing position (tip x = 0.14 m). So if
the block touches there, that is the expected result, not a mistake; the point of the session is to find out where it stops
touching.

## 0. Before you leave the desk

- [ ] Checklist step 3 (measurement hour) is done and the camera case size is on the filled sheet (rows D1 to D3). Without it
      the cardboard block has the wrong size and the session is worthless. If step 3 is not done, do step 3 first.
- [ ] Cardboard block cut to the measured camera size plus 5 mm each way (width, height, depth). Write "CAMERA" and the
      lens side on it; mark the lens centre with a cross.
- [ ] Bring: tape, a ruler (mm), the puck, sheet 01 (straight line) and sheet 03, the phone, this list printed, a pen.
- [ ] Optional: the hose clamp, the bent aluminium strip and the M3 screws from checklist step 2, if you want to clamp the
      real camera under the holder's screws instead of holding it. Only if the measurement hour showed the strip fits.
- [ ] The laptop with the project folder; `02_code/view_camera.py` is the only program that will connect.

## 1. Safety, every time (read aloud)

1. **Motors off before anyone touches the arm.** Press FreeMotion (learning mode) and wait until the arm goes soft. Never push
   or pull on an arm that is being held by its motors.
2. **The robot stays away from the table edge.** The base sits where it was taught (sheet 5 cm in front of the base, 7 cm from
   the side edge). Do not move the base.
3. **One program connected at a time.** If the camera viewer is open, nothing else is started. If another script is to run,
   close the viewer first (press q in its window).
4. **No manual calibration.** Only the automatic one after a restart, if a restart is ever needed.
5. **The camera and its focus ring are not touched.** The cardboard block stands in for the camera; the real camera stays
   where it is. (Exception only if you chose the clamp option in section 0 and step 3 showed it is safe.)
6. **Hands clear before any move.** The person at the laptop says "moving" before every command; the person at the arm
   answers "clear".
7. **Stop at 30 minutes** even if the list is not finished. Write down where you stopped.

## 2. Set up (5 minutes)

1. Put sheet 01 on the table as taught on 25.10 (near edge 5 cm in front of the base, line pointing away from the robot).
2. Switch the robot on if it is off, wait for the automatic calibration, then connect with the camera viewer:
   `..\.venv\Scripts\python.exe view_camera.py` from `02_code`. Check that the live picture appears and that the sharpness
   number is shown.
3. Say "moving", then bring the arm to the start pose of the straight push (rod vertical, tip at about x = 0.16 m, 5 mm above
   the paper). Use the way you used on 25.10 (teach_positions or push_straight's start pose); nothing new is run today.
4. Press FreeMotion. The arm rests on the rod.

## 3. The gap check: does the camera body fit? (10 minutes)

Do this at four tip positions, nearest first. At each position the arm is moved by a command, then motors off, then the
block is held, then measured.

| Position | Tip x (m) | Where on the sheet |
|---|---|---|
| P1 | 0.14 | just before the line start (the tightest spot) |
| P2 | 0.18 | 2 cm into the line |
| P3 | 0.22 | 6 cm into the line |
| P4 | 0.42 | at the far end of the sheet |

For each position:

1. Laptop: "moving". Move the tip to the position with the rod vertical, 5 mm above the paper, the same way as the pushes on
   25.10 (a straight-line pose command). Wait until the move has finished.
2. Press FreeMotion. Say "motors off".
3. Hold the block at the planned spot: lens cross 5 cm ahead of the rod axis (towards the far end of the sheet), lens 6 cm
   above the wrist axis (the joint-5 axis runs through the wrist between the fork arms), block tilted about 10 degrees
   back towards the robot. The block's back face is towards the robot, the lens face looks down the line.
4. Measure with the ruler the smallest gap between the block and (a) each fork arm, (b) the blue wrist cover, (c) the button
   box on the forearm. Write the smallest of the three in the table below. "Touches" counts as 0.
5. Shift the block 2 cm towards the robot (lens 3 cm ahead of the rod axis) and measure again. Then 2 cm further away (lens
   7 cm ahead) and measure again.
6. Photograph the block in the planned spot from the side with the ruler in the picture (photo list, section 5).
7. Laptop: "moving". Next position.

Results table (mm):

| Position | Gap at 5 cm ahead (planned) | Gap at 3 cm ahead | Gap at 7 cm ahead | What it touched first | Photo no. |
|---|---|---|---|---|---|
| P1, x 0.14 | | | | | |
| P2, x 0.18 | | | | | |
| P3, x 0.22 | | | | | |
| P4, x 0.42 | | | | | |

Expected [calc, design review M17]: about -1 mm (touching) at P1, about +6 mm at P2, +12 mm at P3, 17 mm or more at P4.

## 4. The picture check: does the camera see the puck and the line? (10 minutes)

The real camera is still in its stock holder, so a picture from the planned spot cannot be taken directly today unless you
chose the clamp option. What can be checked without touching the camera:

1. Move to P2 (x = 0.18 m). Motors off. Put the puck against the block, on the line, as when pushing.
2. Hold the cardboard block in the planned spot. From just behind it, at the lens cross, look along the block's tilt and
   note what you see: is the puck's leading edge in view, how many centimetres of line lie ahead before the block's own
   edges cut the view? Hold the ruler on the line and read it. Write it down. This is an eye estimate, not a measurement.
3. Take a phone photo from the lens cross, looking the same way, with the ruler on the line (photo list).
4. Turn the sheet under the resting arm so the line runs sideways (heading 90 degrees), puck in contact again, and repeat 2
   and 3.
5. If the clamp option is in use (real camera on the strip at the planned spot): save a viewer picture (press s in the
   viewer window) at heading 0 and at heading 90, and write down the sharpness number shown for each. Those pictures go to
   `03_data/raw/camera_check/` automatically.

Results:

| Heading | Puck leading edge in view? | Centimetres of line in view ahead of the puck | Sharpness number (clamp option only) | Photo no. |
|---|---|---|---|---|
| 0 degrees | | | | |
| 90 degrees | | | | |

## 5. What to photograph (phone, ruler in every picture)

| Photo no. | What | Why |
|---|---|---|
| 1 | Block at P1 from the side, planned spot | the tightest gap, the decision on the smallest usable x |
| 2 | Block at P1, close-up of the smallest gap with the ruler | the number in the table must be visible |
| 3 | Block at P2 from the side | |
| 4 | Block at P3 from the side | |
| 5 | Block at P4 from the side, showing the button box of the forearm | the forearm leans most here |
| 6 | From the lens cross at P2, heading 0, puck in contact, ruler on the line | what the camera would see |
| 7 | From the lens cross at P2, heading 90 | |
| 8 | The whole set-up from above, with sheet, puck and arm | for the report |
| 9, 10 | Clamp option only: viewer pictures at heading 0 and 90 (saved by the viewer, not the phone) | the real sharpness |

Afterwards copy the phone photos to `03_data/raw/camera_check/` named `2026-MM-DD_mockup_<no>.jpg`.

## 6. What counts as acceptance (from checklist step 5)

The mock-up is accepted, and step 6 (the decision) can be made at the desk, when all three hold:

1. **Gap**: at every tip position you intend to use, the block has at least 3 mm to the fork, the cover and the button box.
   If P1 (x 0.14) fails and P2 (x 0.18) passes, that is also acceptable, with one written decision: "the tip stays at
   x = 0.16 m or more; the sheet is laid accordingly". (The line on sheet 01 already starts at x 0.156 m.)
2. **View**: at headings 0 and 90 degrees the puck's leading edge and at least 5 cm of line ahead of it are in view from the
   lens cross.
3. **Sharpness**: only with the clamp option. The sharpness number from the planned spot is readable and written down, so
   step 4 of the main checklist (the focus sweep) can say whether a refocus is needed. Without the clamp option this item
   is "not measured" and stays open for step 4.

If item 1 fails at every position, or item 2 fails at both headings, the mock-up is not accepted: write down the gap and
view numbers anyway, and the position is moved (higher, further ahead, or less tilt) in a second short session. Do not
guess a new position at the robot; take the numbers back to the desk.

## 7. Writing it down (at the desk, 10 minutes)

1. Copy the two results tables and the acceptance verdict into a dated note in this folder,
   `mockup_results_2026-MM-DD.md`, with the photo numbers.
2. One line for the team log (`STATUS.md`, Robot and hardware section): "Mock-up done DD.MM: smallest gap X mm at tip
   x = 0.YY m; puck and Z cm of line in view at 0 and 90 degrees; accepted / not accepted."
3. Leave the robot as you found it: arm at home, motors off, viewer closed, sheet and puck put away.
4. Step 6 (the decision: lens height, distance ahead of the rod, tilt, each to +-2 mm and +-3 degrees) is made from the
   photos and the tables, at the desk, in the next session.
