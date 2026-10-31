# Niryo support's CAD of the camera holder, checked against bracket F (30.10.26)

Niryo support answered Emranul's form request of 28.10 on 30.10.26 with `Assembly 1.zip`: STL files of the Vision Set camera
assembly (camera support = the black holder, camera cover, camera 1080P module, the screw, the locating pin, the cable, a
label). The five useful files and one cross-section picture are in
`../niryo_files/ned2_camera_assembly_from_niryo_support_2026-10-30/` (the 7.5 MB cable and the label are left out).
Each part sits in its own coordinates (no assembly positions). Measured on the PC from the meshes by this session; the robot
was not touched.

## What the CAD says, next to what we measured and assumed

| Item | Niryo CAD (30.10) | Our value in the print file | Verdict |
|---|---|---|---|
| Holder outer width across the ears | 34.0 mm | 34 (measured 28.10) | same |
| Ear height | 24.5 mm | plate 22 (measured) | fine, the plate is the smaller |
| Screw holes, centre distance across | **27.0 mm** | **26 (calipers 28.10; paper B "fits")** | **1 mm apart; check Monday** |
| Pin holes, centre distance across | 27.0 mm | 26 | the same 1 mm question |
| Screw row to pin row, up and down | 16.0 mm | 16 | same |
| Screw | **M2 x 8, head 3.8 mm, hex socket** | **M2.5 x 6, 2 mm key (calipers 28.10)** | **differ; check Monday** |
| Ear screw hole | 2.9 mm through 3 mm of ear, then a 5 mm counterbore for the head (ear 6 mm thick in total) | ear assumed 3 mm thick | the screw passes 3 mm of ear as assumed; the holder is 3 mm deeper than assumed |
| Locating pin | 2 x 8 mm, 2.0 mm holes in the ear | pin 2 mm assumed, 2.6 mm holes in the top plate | pins fit loosely, as planned |
| Gap between the ears (room for the plate's boss) | 19.0 mm wide; nothing of the holder closer than 9 mm to the plate face in the middle | foot pocket 19 wide, 10 deep | matches |
| Lens centre in front of the plate face | about 29 mm (camera board centre; lens hole centre at 30) | 16 mm assumed from photo 02 | **13 mm apart**; changes the predicted lens position, not the part |
| Camera board inside the holder | tilted 15 degrees relative to the mounting face | included in the 22.4 degree pose fitted from pictures | nothing to change |

## What this means for the printed part

1. **Hole spacing 26 or 27.** If the real holder has 27 mm, the top plate's holes are 0.5 mm off on each side. The pins (2 mm
   in 2.6 mm holes) have 0.3 mm of play, so a 0.5 mm offset would not go in without forcing, and the screws would not line up.
   The calipers said 26 and paper pattern B fitted the holder's back on 29.10, which speaks for 26; the CAD says 27. One of
   the two is wrong, and only the holder in the hand can tell.
2. **Screw M2 or M2.5.** The top plate's 2.2 mm holes grip an M2.5 thread, but an M2 screw (2.0 mm) would just drop through
   them. The foot's 3.0 mm holes take either. The calipers said M2.5 and the 2 mm hex key fitted (an M2 socket head takes a
   1.5 mm key, so a 2 mm key would not go in), which speaks for M2.5; the CAD says M2.
3. **Lens 29 mm in front of the plate face, not 16.** Recomputing `derived_geometry` with 29 instead of 16: the plate lean stays
   32.4 degrees, but the lens ends about 7 mm lower than aimed (about 53 mm above the joint-5 axis instead of 60) and about
   2 mm less forward. Still far above today's 6 mm; the field of view is a little smaller than the 15 x 20 cm estimate. No
   reason to change the part before the first picture on the robot.

## Monday checks added (printed part and holder in hand, robot off, nothing unscrewed on the robot)

- A. Hold the holder's back on the printed top plate: do both pins drop into the lower holes and do the two screw holes line up?
  Yes: spacing 26 is right. No: spacing is 27, the top plate needs a reprint with `hole_spacing_z` = 27 (one parameter).
- B. Turn one of the holder's own screws into a 2.2 mm hole of the top plate by hand: it bites = M2.5 (as measured, all good);
  it drops through = M2, then use M2 nuts behind the top plate (the upper holes are through holes; there is room beside the
  post) with M2 x 12 screws, or reprint with 1.7 mm holes.
- Nothing of this needs the robot; it takes two minutes with the part and the holder.

## Fallback prepared

`bracket_f_design.py` can produce the corrected version in one run (`hole_spacing_z` 27 and/or `self_tap_hole_diameter` 1.7);
not generated yet, because the print on hand may well be right. If check A or B fails on Monday, generate it and send it to the
Maker Space the same day.
