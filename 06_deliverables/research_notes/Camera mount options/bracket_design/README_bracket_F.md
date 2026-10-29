# Camera bracket F: the print file (checklist step 7)

Made 29.10.26 by the Robot and hardware session for Emranul, from the numbers measured on 28.10 and the camera pose fitted
from the team's own pictures. Files for the Maker Space: `TeamM_Project6_ned2_camera_bracket_F.3mf` (preferred) and
`TeamM_Project6_ned2_camera_bracket_F.step`, both in millimetres. Paper test: `ned2_camera_bracket_F_paper_test.pdf`
(print at 100 %). The script that made everything: `bracket_f_design.py` (CadQuery, run with `..\..\..\..\.venv\Scripts\python.exe`).

## What it is

An L-shaped riser, one piece, about 41 x 34 x 86 mm, 29 cm3 of plastic (about 36 g in PLA):

1. **Foot**: a 34 x 22 x 3 mm plate that sits on the aluminium plate in the wrist, held by the two existing screw holes.
2. **Arm**: 24 mm forward, out of the cover window.
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
| Plate 34 x 22, holes 26 apart sideways | measured | Emranul, calipers, 28.10 |
| Screws M2.5 x 6, 2 mm hex key | measured | same |
| Camera case 37 x 53 x 31, cable 5.3 mm | measured | same |
| Holes 16 apart up and down | **assumed** (Ned v1 holder pattern, 27 x 16) | Niryo STL; the paper test checks it |
| Pin diameter 2 mm, ear thickness 3 mm | **assumed** from the photos | paper test and the fit of the real holder |
| Lens today: 25 ahead, 6 up, 22 degrees forward | fitted from pictures, +-3 mm, +-3 degrees | `cam_calibration.json` |
| Lens 16 mm in front of the plate face at pattern-centre height | **assumed** (+-5 mm) | photo 02 |
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

1. Hold the foot on the paper pattern A, the top plate on pattern B: holes in line.
2. Screw the holder onto the top plate with the stock screws, camera in it, cable on. Hold the whole thing against the wrist
   where the holder was: the foot's two holes over the two screw holes, and look that nothing touches the cover rim.
3. Only then screw it on (motors off), with the camera's own two screws kept for the holder.
4. First check at the robot: `view_camera.py` at the start pose; the puck's leading edge and at least 5 cm of line in view.

## To change it

Edit `DEFAULT_BRACKET_PARAMETERS` in `bracket_f_design.py` (for example `lens_up_mm`, `lens_tilt_back_degrees`,
`hole_spacing_y`) and run the script; the 3MF, STEP, STL, the side-view picture and the parameter JSON are rewritten. Then run
`paper_test.py` for the PDF.
