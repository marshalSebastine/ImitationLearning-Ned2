# Measurement sheet for the camera mount (checklist step 3)

Written 27.10.26 for Emranul and Marshal, shortened the same day. Print this page and take it to the robot with the calipers
from the Maker Space (lent 8:00 to 13:00, back the same day).

**Part 1 is enough for the print.** Ten numbers, about 20 minutes, most of them measured at a desk with the camera holder
unscrewed. They are the numbers Niryo does not publish anywhere (see `niryo_data.md`, section 6). **Part 2 is optional**:
those values already exist from Niryo's 3D files, and the cardboard mock-up (step 5) checks the clearances anyway. Skip
Part 2 without harm if you are short of time or patience.

| Date | Measured by | Calipers (make, resolution) | Start | End |
|---|---|---|---|---|
| 28.10.26 | Emranul Hoque | digital calipers borrowed from the Maker Space, 0.1 mm, checked on a 20 cent coin (22.3 mm, true 22.25) | morning | |

**Result 28.10.26: Part 1 complete (rows 1 to 10), camera screwed back on with the same two screws. Part 2 rows 11, 12 and 21
done, the rest skipped on purpose (Niryo's STEP confirmed by rows 11 and 12). Photos in `03_data/raw/camera_check/`, named
`2026-10-28_measure_photo01` to `photo11`. Surprises: holder hole spacing 26 mm (not 40 to 45); screws M2.5 x 6 with a 2 mm hex
key; the brace is a 34 x 22 mm plate with four corner holes, the holder uses the top two for screws and the bottom two for
pins; camera 37 wide, 53 tall, 31 deep; the cover window is 37.3 wide, so the camera only just fits through it.**

## Plain steps for Part 1 (written 28.10.26 for Emranul)

Before: robot on, FreeMotion pressed (arm soft), two photos of the camera in place (front and side, ruler next to it).
Key: 2 mm hex key fits the two screws (measured 28.10). Screws into a cup. Lift the camera with its black frame out,
cable stays plugged in, carry it to the table.

1. Row 1. One screw, big jaws on the threaded part. Write the number.
2. Row 2. Same screw, big jaws from under the head to the tip. Write it.
3. Row 3. Middle of one frame hole to middle of the other. Write it.
4. Row 4. Small jaws inside one hole, open until they touch. Write it.
5. Row 5. Big jaws across the camera front, left to right, then top to bottom. Write both.
6. Row 6. Big jaws from the lens front to the back of the frame. Write it.
7. Row 7. Big jaws on the cable. Write it. Cable where it enters the case: "ok" or "split".
8. Photos of the camera: front, side, back, underneath, ruler next to it.
9. Row 8, at the wrist. In the opening: metal with two holes. Middle to middle. Then rod to the nearer hole. Write both.
10. Row 9. The opening in the blue cover: wide and tall. Write both.
11. Photo of the opening with the ruler.
12. Camera back in, same as photo 1. Both screws in, turn right, gently tight.

## Rules

1. Motors off (FreeMotion) before anyone touches the arm. Never push on a held arm. The robot stays away from the table edge.
2. No manual calibration. The lens and the focus ring are not touched; measuring the case from outside is fine.
3. Write what the calipers show, not what you expected. If something cannot be measured, write "not measured" and why.
4. One photo per row with a ruler in the picture, numbered in the last column.

## How to read the "Expected" column

None of these was measured on your robot with calipers. **[Niryo STEP]** is calculated from Niryo's 3D drawing. **[team]**
is your own ruler reading of 24. to 26.10. **[calc]** was computed by a reviewer from Niryo's files. **[guess]** is a guess
from a photo or a reseller page, written only so a surprise is noticed. **unknown** means nobody has a number yet.

---

## Part 1: the ten numbers the print needs (20 minutes)

### Taking the holder off (5 minutes, at the arm)

1. Motors off (FreeMotion). Let the arm rest on the rod.
2. Photo of the holder in place from the front and from the side, ruler in the picture (photo 1 and 2). This is also the
   picture you compare against when putting it back.
3. With the Allen key that fits, undo the two hex screws in the holder's ears. Put the screws in a cup. Lift the holder with
   the camera out of the cover opening; the cable stays plugged in, so carry the holder only as far as the cable allows, or
   unplug the USB cable at the robot's back panel first and plug it back in afterwards.
4. Take the holder and camera to the desk. Items 1 to 7 are measured there.

### The numbers

| # | What to measure | Expected | Tool | Measured | Photo |
|---|---|---|---|---|---|
| 1 | The two screws: outer diameter of the thread | unknown; M3 (3.0 mm) or M2.5 (2.5 mm) most likely [guess] | calipers | **2.5 to 2.6 mm: M2.5** (Emranul, 28.10.26) | |
| 2 | The two screws: length under the head, and the hex key size that fits | unknown | calipers, Allen key | **5.7 mm under the head** (whole screw 8.2, head 2.5; so M2.5 x 6); **2 mm hex key** (Emranul, 28.10.26) | |
| 3 | Holder ears: distance between the two holes, centre to centre | 40 to 45 mm [guess from photo scale] | calipers | **26 mm** (Emranul, 28.10.26). **Differs from the guess by 15 to 19 mm; the guess is dropped** | 3 |
| 4 | Holder ears: hole diameter, and whether the hole is round or a slot | unknown; 3.2 to 3.5 mm would fit an M3 screw [guess] | calipers | **2.4 to 2.5 mm, round** (Emranul, 28.10.26); the M2.5 screw cuts its own way through | 3 |
| 5 | Camera case: width and height of the face with the lens | about 37 x 37 mm [guess: old Ned holder, simulation stand-in] | calipers | **Upper part (lens) 37.0 mm wide; lower part (label) 34 mm wide; total height 53.1 mm bottom tabs included, 48 mm from the top to the notch at the bottom; the notch between the two bottom tabs is 18.5 mm wide, widening towards the ears** (Emranul, 28.10.26). Width as guessed; height 16 mm more than guessed | 5 |
| 6 | Camera case: depth from the lens front to the back, including the focus ring and the ears | 15 to 30 mm [guess] | calipers | **31 mm**, lens front to the back of the frame (Emranul, 28.10.26) | 6 |
| 7 | Cable: which side of the case it leaves, and its outer diameter. Also look at the sleeve near the case (it looked split in the 26.10 photo): write "intact" or describe | unknown | calipers, eye | **5.3 mm**; leaves at the back of the case near the bottom, through a small plug (Emranul, 28.10.26). Sleeve at the case: **ok**, not split (checked by eye) | 7 |
| 8 | At the arm, holder off: the two screw holes in the aluminium brace: distance between them, and distance from the centre of the rod to the holes (sideways and up/down) | hole spacing should equal item 3; rest unknown | calipers or ruler | **The brace is a square plate with four holes, one per corner, the rod through its middle. The M2.5 screws go in the top two holes, two small pins on the back of the holder go in the bottom two. Top holes about 26 mm apart, by eye (calipers do not fit inside the cover)** (Emranul, 28.10.26). **Plate 34.0 to 34.2 mm wide, 22.0 to 22.5 mm tall** (calipers, Emranul, 28.10.26); the four holes sit near the corners, the rod passes through the plate's middle, so each top hole is about 13 mm sideways from the rod axis (calculated from 26 mm spacing) | 8, 10, 11 |
| 9 | At the arm: the opening in the blue cover where the holder sits: width and height, and how far its lower edge is above the rod's top | unknown | calipers | **37.3 mm wide** (blue edge to blue edge at the plate's height, calipers); **77.5 mm tall** (top of the opening to the bottom of the blue, ruler) (Emranul, 28.10.26). Note: the camera's 37.0 mm upper part just fits the 37.3 mm window | 8 |
| 10 | Photos with a ruler: holder with camera from the front, side, back and underneath; the brace holes through the cover opening with the holder off | | phone | | |

### Putting the holder back (5 minutes, at the arm)

1. Motors still off. Seat the holder in the cover opening exactly as in photo 1, both screws in, tightened gently with the
   Allen key, not forced.
2. Plug the USB cable back in if it was unplugged. Route the cable as it was.
3. Later, at the laptop: one picture from the camera at the start pose, compared with
   `03_data/raw/camera_check/view_reference_rod_camera_start_pose.jpg`. If the view has shifted by more than a few
   pixels, loosen and reseat once.

**Part 1 done when:** rows 1 to 9 have a number, row 10 has its photos, and the holder is back on with the same two screws.

---

## Part 2: optional, only if there is time

These confirm numbers that already come from Niryo's 3D files. A difference of more than 2 mm would be worth knowing; agreement
changes nothing.

| # | What to measure | Expected | Tool | Measured | Photo |
|---|---|---|---|---|---|
| 11 | Rod diameter on the full-diameter part (one height, two directions) | 21.0 mm [Niryo STEP]; about 18 mm [team] | calipers | **21 mm** (Emranul, 28.10.26); matches the STEP, the earlier 18 was wrong | |
| 12 | Rod visible below the cover edge, cover to rod end | 24.5 mm [team]; 28.5 mm total below the rod's top [Niryo STEP] | calipers | **27 mm** (Emranul, 28.10.26) | |
| 13 | Is the rod magnetic? (hold a small magnet to its side) | unknown | small magnet | yes / no | |
| 14 | Fork arms: how far their lowest edge hangs below the rod's top, each arm | 5.3 mm below [calc] | calipers | | |
| 15 | Gap between the two fork arms, inside to inside | 46 to 48 mm [calc] | calipers | | |
| 16 | Wrist cover: lowest edge above the rod's top on the camera side | 8.7 mm above [calc] | calipers | | |
| 17 | Button box on the forearm: lowest edge above the rod's top with the forearm level, and which side | 7.7 mm above, 30 x 47 mm box [calc] | ruler, photo | | |
| 18 | Magnet block: width, depth, height from top face to magnet tip | 37 x 32 x 28.5 mm [Niryo STEP]; height 26 mm [team] | calipers | | |
| 19 | Magnet block: diameter of the face that touches the puck | 20.5 to 26 mm nose [Niryo STEP]; 13 mm [team] | calipers | | |
| 20 | Puck offset: block and puck in contact by hand, photo from above with a ruler: rod centre to puck centre | 27 to 33 mm [calc] | ruler, photo | | |
| 21 | Mass of camera plus holder plus cable | 30 to 50 g camera alone [guess] | kitchen scale | **58 g** (camera, frame and USB cable together; Emranul, 28.10.26) | |

---

## Photo list

Copy the phone photos afterwards to `03_data/raw/camera_check/`, named `2026-MM-DD_measure_<row>.jpg`.

| Photo no. | Row | What is in it |
|---|---|---|
| 1 | holder in place, front | |
| 2 | holder in place, side | |
| 3 | | |
| 4 | | |
| 5 | | |
| 6 | | |
| 7 | | |
| 8 | | |
| 9 | | |
| 10 | | |

## Afterwards (at the desk, ten minutes)

1. Type the values into a copy of this sheet, named `measurement_hour_sheet_filled_2026-MM-DD.md`, in this folder.
2. Mark every row where the measured value differs from the expected one by more than 2 mm.
3. Tell the Claude session that the numbers are in; the part design (step 7) uses Part 1 only.
