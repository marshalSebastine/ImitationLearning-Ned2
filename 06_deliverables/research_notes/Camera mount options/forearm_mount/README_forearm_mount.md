# Camera on the forearm cover: where it sits (for putting it back)

Recorded 5.11.26 from Emranul's description and three photos in this folder. The camera, still in its printed F3
bracket, has been taped to the robot since 4.11.26 evening; all recordings from test 5 (episode 037, 4.11.26) on were
made with it in this place.

## Position

- **Where:** on the black forearm cover just below the elbow, on the lower part that turns with the forearm (joint 4),
  **4 mm below the seam** between the upper black cover (the one with the screws) and the lower black ring.
- **Sideways:** the bracket's foot sits **between the two cover screws**, centred between them (photo
  `2026-11-05_forearm_mount_from_behind.jpg`).
- **Held by:** **three strips** of clear tape over the bracket's foot and round the cover
  (`2026-11-05_forearm_mount_tape_detail.jpg`).
- **Pointing:** the lens looks down the forearm towards the hand and the sheet, so the robot's hand sits in the
  bottom middle of the picture (`2026-11-05_forearm_mount_side.jpg`). The cable runs over the cover to the back.

## Putting it back

1. Same place: foot between the two screws, 4 mm below the seam, three strips of tape. Do not touch the focus ring.
2. Check the view against the reference picture
   `03_data/raw/camera_check/camera_reference/forearm_cover_tape_F3_bracket_x171_y37mm.jpg` (episode 042, 5.11.26,
   arm at the start of sheet 04). The recording tool does this at every episode start ("camera check: ... px from the
   reference"); under 4 px means the view matches the recorded data. A live alignment helper (overlay of the reference
   on the live picture) is planned but not built yet.
3. If it cannot be brought within a few pixels, the data before and after are from different camera views: give the
   mount a new name in config (`camera_mount`) so they are never mixed.

## What the pictures showed so far

The view tilted once, between tests 4 and 5 on 4.11.26 (the hand came into the bottom of the picture); from test 5 to
5.11.26 it stayed within about 2 px (camera check on the first picture of every episode).
