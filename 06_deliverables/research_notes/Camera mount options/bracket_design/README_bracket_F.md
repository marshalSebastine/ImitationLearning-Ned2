# Camera bracket F: the print file (checklist step 7)

Made 29.10.26 by the Robot and hardware session for Emranul, from the numbers measured on 28.10 and the camera pose fitted
from the team's own pictures. Files for the Maker Space: `TeamM_Project6_ned2_camera_bracket_F.3mf` (preferred) and
`TeamM_Project6_ned2_camera_bracket_F.step`, both in millimetres. Paper test: `ned2_camera_bracket_F_paper_test.pdf`
(print at 100 %). The script that made everything: `bracket_f_design.py` (CadQuery, run with `..\..\..\..\.venv\Scripts\python.exe`).

## What it is

An L-shaped riser, one piece, about 41 x 34 x 86 mm, 30 cm3 of plastic (about 37 g in PLA):

1. **Foot**: two 3 mm thick side strips (the outer 7.5 mm of a 34 x 22 mm plate) that sit on the aluminium plate like
   the holder's ears, held by the two existing screw holes. The middle is open, 19 mm wide and 10 mm deep, for the raised
   rounded boss around the shaft in the plate's centre (seen 29.10 when pattern A would not lie flat; its size is not
   measured, the stock holder has a similar gap).
2. **Legs and cross-bar**: 24 mm forward, out of the cover window, only below the screw heads, so the hex key reaches the two
   foot screws straight from the front.
3. **Post**: 12 mm thick, 16 mm wide, 64 mm up. Narrow on purpose, so the hex key still reaches the two foot screws.
4. **Top plate**: a copy of the aluminium plate's four-hole pattern (26 mm across, 16 mm up and down), leaning forward by
   32 degrees. Niryo's black camera holder screws onto it exactly as it screws onto the robot today: two M2.5 screws in the
   upper holes (they cut their own thread in the 2.2 mm printed holes), its two pins in the lower holes.

Where the lens ends up (calculated from today's fitted pose plus the bracket): about 58 mm ahead of the rod axis, 60 mm above
the joint-5 axis, looking 10 degrees back towards the robot. Today: 25 mm ahead, 6 mm above, 22 degrees forward
(`scripts/cam_calibration.json`, pixel fit of 26.10). This is option F of `design_review_one_print.md` (lens 51 ahead, 60 up,
tilt -10); the extra 7 mm forward comes from the leaning plate and is accepted.

## Which numbers are measured, which are not

| Number | Value | Source |
|---|---|---|
| Plate 34 x 22, holes 26 apart sideways | measured; **Niryo's CAD of 30.10 says 27** | Emranul, calipers, 28.10; `niryo_support_cad_check_2026-10-30.md` |
| Screws M2.5 x 6, 2 mm hex key | measured; **Niryo's CAD says M2 x 8** | same; CAD check 30.10 |
| Camera case 37 x 53 x 31, cable 5.3 mm | measured | same |
| Holes 16 apart up and down | confirmed: paper pattern B fits the holder's back (Emranul, 29.10) | paper test |
| Boss in the plate's middle: under 19 mm wide, under 10 mm high | **assumed** from the 29.10 photo | the open middle of the foot |
| Pin diameter 2 mm | confirmed by Niryo's CAD (2 x 8 pin) | CAD check 30.10 |
| Ear thickness 3 mm | CAD: ear 6 mm, the screw passes 3 mm of it (counterbore for the head) | CAD check 30.10 |
| Lens today: 25 ahead, 6 up, 22 degrees forward | fitted from pictures, +-3 mm, +-3 degrees | `cam_calibration.json` |
| Lens 16 mm in front of the plate face at pattern-centre height | **assumed**; Niryo's CAD puts the camera board centre about 29 mm in front: the lens then ends about 7 mm lower than aimed | photo 02; CAD check 30.10 |
| Cover rim sticks out less than 12 mm beyond the plate face | **assumed** from photo 01 and 08 | the arm is 12 mm before the post |
| Target: 51 ahead, 60 up, -10 degrees | calculated by the reviewers, not tested on the robot | design review, M17 |
| Clearance to the forearm at tip x 0.14 m | calculated: about -1 mm (touches), +6 at 0.16, +12 at 0.18 | design review, M17; the real camera body is larger than the one modelled there, so **the mock-up is still the test** |

## Print settings to ask for

- PLA (or PETG), 0.4 mm nozzle, 0.2 mm layers, 3 walls, 30 % infill or more.
- **Lay the part on its side** (the 34 mm wide faces of the foot and top plate on the bed). The post is then strongest
  against bending. The narrow post then floats 9 mm above the bed: **supports needed** under it (tree supports are fine).
- No scaling: the file is 1:1 in millimetres; the bounding box must read about 41 x 34 x 86 mm in the slicer.

## Screws

The two stock M2.5 x 6 screws go into the top plate with the holder (as today). For the foot two more M2.5 screws are needed,
6 to 10 mm long (the foot is 3 mm thick like the holder's ear, so 6 mm works; 8 to 10 mm is safer). Ask the Maker Space or the
lab; nothing to buy.

## After the print, before it goes on the robot

0. Checks A and B of `niryo_support_cad_check_2026-10-30.md`: pins and screw holes of the holder on the top plate (26 or 27 mm), a stock screw in a 2.2 mm hole (M2.5 or M2).
1. Hold the foot on the paper pattern A, the top plate on pattern B: holes in line.
2. Screw the holder onto the top plate with the stock screws, camera in it, cable on. Hold the whole thing against the wrist
   where the holder was: the foot's two holes over the two screw holes, and look that nothing touches the cover rim.
3. Only then screw it on (motors off), with the camera's own two screws kept for the holder.
4. First check at the robot: `view_camera.py` at the start pose; the puck's leading edge and at least 5 cm of line in view.

## To change it

Edit `DEFAULT_BRACKET_PARAMETERS` in `bracket_f_design.py` (for example `lens_up_mm`, `lens_tilt_back_degrees`,
`hole_spacing_y`) and run the script; the 3MF, STEP, STL, the side-view picture and the parameter JSON are rewritten. Then run
`paper_test.py` for the PDF.

## F2: the corrective version of 2.11.26 (leaning post)

Why: F was fitted on 2.11.26 (Emranul and Marshal). It mounts and the pictures are good, but the back of its straight post
meets the forearm cover when the wrist (joint 5) is tilted up to -2 degrees (measured by hand with the motors off,
`02_code/wrist_limit_by_hand.py`). The pushing pose nearest the base (rod 14 cm out) needs the wrist at +14 degrees, so with F
the rod can push only from about 25 cm out to the table edge; sheet 01's line does not fit. The wrist motor tripped
(hardware error 32) three times before this was understood. Details: `03_data/raw/camera_check/2026-11-02_bracket_F_notes.md`.

What changed (parameters in `bracket_f_design.py`, files `ned2_camera_bracket_F2.*`):

- `hole_spacing_z` 27 instead of 26: Niryo's CAD of the holder (check of 30.10) gives 27.0 mm between the screw holes and
  between the pin holes; fitting F on 2.11 the screws were hard to get in (Emranul: 0.5 to 1.5 mm more would be better), so
  the CAD wins over the 28.10 caliper reading. Applies to the foot (brace holes) and the top plate (holder holes); pattern
  A and B of the F2 paper test are therefore new and must be held on the brace and the holder before sending.
- `post_lean_degrees` 25 (new): the post is a parallelogram leaning forward, its top 35 mm further forward than its foot. The
  foot, the legs, the cross-bar and both hole patterns are unchanged, so patterns A and B of the paper test still apply.
  Checked against Niryo's own forearm STL (`scripts/review_f2_clearance.py`, the design review's method): the model shows F's
  post touching the forearm at wrist +4 degrees (the robot showed -2, so the model is about 5 mm optimistic); with a 20 degree
  lean the modelled clearance at the near poses was 2 to 6 mm, with 25 degrees it is 6 to 10 mm (13 mm further out).
- `lens_tilt_back_degrees` 30 instead of 10 (22 in the version e-mailed at midday on 2.11): the meander test the same
  afternoon showed that a leg running back towards the robot must be pushed from the far side of the puck, where a camera aimed
  ahead of the rod sees neither puck nor line. At 30 degrees the picture covers from 7.8 cm behind the rod axis to 8.3 cm ahead
  (calc: lens 93 mm ahead of the rod axis, 120 mm above the paper, 25 degree vertical half-angle), about 5 cm of line beyond
  the puck in both pushing directions; at 22 it covered 3.6 cm behind to 9.9 ahead, half a centimetre of line behind the puck. The camera-mount study's variant with a 20
  degree lean (`G2_F_lean20` in `scripts/cam_options_results.json`) sees 100 / 100 / 100 / 91 / 89 / 89 percent of sheets 01
  to 06, slightly more than F.
- Bounding box about 81 x 34 x 93 mm (was 41 x 34 x 86), 33.1 cm3 (was 30.2). Still printed lying on its side, supports under the post.
- Expected wrist travel gained: about 25 degrees more before contact. Verify after fitting with `wrist_limit_by_hand.py`; the
  goal is contact at +18 degrees or later (the pose 14 cm from the base needs +14.6 in the model, +14.1 measured on 31.10).

What is NOT changed and still assumed: the cover rim's distance in front of the brace face (assumed under 12 mm; no calipers
were available on 2.11). The mesh check above stands in for that measurement; its 5 mm optimism is the reason for 25 instead of
20 degrees.

Compromises, honestly: the camera hangs 35 mm further forward (a little more load on the wrist motor, which held F without
error at every pose except the contact ones); the camera looks 20 degrees more towards the robot, so it sees about 1.6 cm
less far ahead and the puck sits nearer the middle of the picture and the puck sits a little higher in the picture; the longer post bends a little more (12 x 16 mm PLA, fine).
Focus: at 12 cm above the paper the lens is out of focus (sharpness 10 to 39 against 160 far away); the one planned refocus
is still to do, with the camera at pushing height.

Files for the Maker Space: `ned2_camera_bracket_F2.3mf` (preferred) and `.step`, millimetres, 1:1. Paper test:
`ned2_camera_bracket_F2_paper_test.pdf` (pattern C is the new side outline). Views: `ned2_camera_bracket_F2_views.png`,
`ned2_camera_bracket_F2_side_view.png`.
