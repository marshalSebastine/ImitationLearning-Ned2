# Stepping rule v1 (draft, 31.10.26)

The one rule both operators follow when choosing a key, so that the same picture always gets the same label. Drafted by
Marshal's Claude session on 31.10.26 from the data-collection analysis (section 2.12) and summary; **Emranul and Marshal
confirm or change it before the pilot**. Any change gets a new version number here and in the session notes, because
episodes recorded under two different rules must not be mixed without saying so.

Valid for the stock camera on its wrist holder (rehearsal on sheet 01). The pixel numbers are re-measured for bracket F.

## Look only at the picture window

Decide from the picture on screen, never from a look at the real sheet: the model will only ever see the picture.
The person at the robot does not call out directions.

## One key, then wait

Press one key, then wait until the picture has changed before deciding the next step. Take as long as needed per step;
speed does not matter, consistency does.

## Which key

Words used below: the **exit point** is where the black line leaves the bottom of the picture just ahead of the puck. The
**band** is the stretch of the picture where the exit point sits when the puck is pushed straight along the line (stock
camera, measured on the 25.10 push pictures: about columns 150 to 200 of 640, see analysis section 2.1). Until the band
is drawn on screen, judge it from a reference picture. The camera holder was unscrewed and refitted on 28 and 29.10, so
the 25.10 reference (`03_data/raw/camera_check/view_reference_rod_camera_start_pose.jpg`) may be off: at the start of the
pilot, put the puck centred on the line start, let the robot come down behind it, and save that first picture as today's
reference (both operators use the same one).

1. **forward (`w`)**: the exit point is inside the band, or less than half a line width (about half a centimetre) outside it.
2. **forward-left (`q`) / forward-right (`e`)**: the exit point is more than half a line width outside the band, on that
   side. Take **one** diagonal, then go back to forward and look again; take another diagonal only if the next picture
   still shows the exit point outside the band.
3. **left (`a`) / right (`d`)**: only when the rod is no longer behind the puck (the puck has slid sideways and a forward
   step would miss it or push it further off). Use them to get back behind the puck, then continue with forward.
4. **back (`s`)**: only when the puck is more than half a puck width off the rod's axis and needs room to get behind it
   again. Never as a "wait".
5. **stop (Space)**: when no line is visible ahead of the puck (the line's end has passed under the puck). Not earlier
   because the goal "looks close".
6. **Crossings** (later sheets): straight through, following the line you came in on.

7. **back-left (`z`) / back-right (`c`)**, added 4.11.26: when pulling the puck towards the base, one diagonal when the line
   leading into the block drifts to that side, then `back` again. Pull rules are a first draft and have not been tried.

## When something goes wrong

- Pressed the wrong key: press **`u`** straight away (marks the previous step as a mistake; the episode is kept), then
  continue by the rule.
- No line visible in the middle of the path: press **`l`** (that picture gets no label), then recover by the rule.
- **`x`** (discard) only when the episode cannot be saved: the puck tipped over or left the sheet, the robot stopped,
  someone touched the puck or the arm.

## Hand-guided replay

The replay proposes a step; press **Enter** to accept it only if this rule would choose the same key. Otherwise press the
key the rule chooses. Guide the puck by hand the way the rule would push it.

## Set-up kept the same between operators

- Same sheet position (do not move the tape), same puck, same lamps and blinds; write the light into `--lighting`.
- Each operator records under their own `--operator` name; alternate operators every five episodes.
- Write any moment you wished for back-left, back-right or a hop into `session_notes.md`, with what the picture showed.

## Agreement test (before real data with bracket F)

Both operators label the same 60 saved pictures with this rule, separately. Agreement of about 80 percent or more:
the rule is usable. Less: find the pictures you disagreed on, sharpen the rule, give it a new version number, repeat.
