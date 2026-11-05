# Camera on the forearm cover: where it sits (for putting it back)

Recorded 5.11.26 from Emranul's description and three photos in this folder. The **bare camera** (the Vision Set
camera in its own housing, no printed bracket) is taped to the robot's forearm cover. Corrected the same day: the first
version of this note said "in its F3 bracket", which was the plan of 4.11.26 evening, not what is on the robot.

## Position

- **Where:** on the black forearm cover just below the elbow, on the lower part that turns with the forearm (joint 4),
  **4 mm below the seam** between the upper black cover (the one with the screws) and the lower black ring.
- **Sideways:** the camera's foot sits **between the two cover screws**, centred between them (photo
  `2026-11-05_forearm_mount_from_behind.jpg`).
- **Held by:** **three strips** of clear tape over the camera's foot and round the cover
  (`2026-11-05_forearm_mount_tape_detail.jpg`).
- **Pointing:** the lens looks down the forearm towards the hand and the sheet, so the robot's hand sits in the
  bottom middle of the picture (`2026-11-05_forearm_mount_side.jpg`). The cable runs over the cover to the back.

## Which recordings used it

The camera check (first picture of every episode compared at the start of sheet 04) shows one change of view, between
test 4 (episode 036) and test 5 (episode 037) on 4.11.26: from test 5 on, the hand is in the bottom of the picture and
the view stayed within about 2 px up to 5.11.26. Episodes 037 to 043 are therefore taken to be this bare-camera mount;
their episode.json still says `forearm_cover_tape_F3_bracket` (the tool's mount name at the time). Whether episodes 033
to 036 were the F3 bracket on the cover is to be confirmed by Emranul. From 5.11.26 the tool writes
`forearm_cover_tape_bare_camera`.

## Putting it back

1. Same place: foot between the two screws, 4 mm below the seam, three strips of tape. Do not touch the focus ring.
2. Check the view against the reference picture
   `03_data/raw/camera_check/camera_reference/forearm_cover_tape_bare_camera_x171_y37mm.jpg` (episode 042, 5.11.26,
   arm at the start of sheet 04). The recording tool does this at every episode start ("camera check: ... px from the
   reference"); under 4 px means the view matches the recorded data. A live alignment helper (overlay of the reference
   on the live picture) is planned but not built yet.
3. If it cannot be brought within a few pixels, the data before and after are from different camera views: give the
   mount a new name in config (`camera_mount`) so they are never mixed.
