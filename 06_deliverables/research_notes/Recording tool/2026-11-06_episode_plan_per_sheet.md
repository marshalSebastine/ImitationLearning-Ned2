# Episode plan per sheet (after the trial day)

Written 6.11.26 by the Recording and data session, on Emranul's request: how many episodes per sheet, and what to do in
each one, as a checklist. It starts AFTER the trial day (`2026-11-06_first_data_day_checklist.md`) and after Emranul and
Marshal have chosen the recording mode (section "Decide before the real data" there). Every number here is a plan, not
a measurement; Claude recounts after the trials.

## How many

About 200 first episodes, as in the 30.10 plan (`2026-10-30_data_collection_analysis.md`, recommended set without
printed sheets), weighted to the hard shapes. About 30 percent start off the line on purpose (offset starts), about 25
percent start near the goal or before a hard spot. The correction rounds with a first model (about 120 episodes, 30.10
plan) come later and are not in this list.

| Sheet | Episodes | Starts from | Time per episode (guess) | Robot time (guess) |
|---|---|---|---|---|
| 01 straight line, 23 cm | 20 | either end | 3.5 min | 1.2 h |
| 02 arc, 31 cm | 35 | both ends (bends left, bends right) | 4 min | 2.3 h |
| 03 two curves, the "2", 51 cm | 45 | end A first, end B if it reaches | 5.5 min | 4.1 h |
| 04 meander, 72 cm | 45 | end A (as on 4 and 5.11), end B if it reaches | 7 min | 5.3 h |
| 05 loop with a crossing, 57 cm | 30 | the lower end (B) only | 6 min | 3.0 h |
| 06 dashed loop with a crossing, 47 cm | 25 | the dashed tail (A) only | 5 min | 2.1 h |
| Total | 200 | | | about 18 h, about 22 h with re-dos and sheet changes |

Order: Round 1 is episodes 1 to 12 of sheets 01, 02 and 04 (36 episodes, about 5 hours): enough for the first
"does the whole pipeline work" model (30.10 decision). Round 2 is everything else, two or three sheets per session, so
that every sheet is recorded on several days and in several lights.

## The kinds of start (use these codes in the notes)

Left and right are seen from behind the robot, looking away from it. Positive numbers mean left.

| Code | What you do | Notes for the command |
|---|---|---|
| STD | start cap on the START MARK, line's first centimetres straight away from the robot | `--offset-cm 0 --offset-deg 0` |
| OL1, OR1 | as STD, but the puck 1 cm LEFT or RIGHT of the start cap; bring it back onto the line in the first 5 to 10 cm | `--offset-cm 1` or `--offset-cm -1` |
| OL1.5, OR1.5 | the same with 1.5 cm | `--offset-cm 1.5` or `-1.5` |
| TL10, TR10, TL15, TR15 | the sheet turned 10 or 15 degrees LEFT or RIGHT about the start cap (the cap stays on the mark), so the first push meets the line at an angle | `--offset-deg 10`, `-10`, `15`, `-15` |
| NE | near the end: the puck on the line 4 to 8 cm before the end cap | start point taught once per sheet layout (below), `--start-index 1` |
| MID | a start before a hard spot: 3 cm before the sharp turn (03) or at the middle hairpin (04) | taught once, `--start-index 2` |
| X | 5 cm before the crossing (05, 06) | taught once, `--start-index 2` (first pass) or `3` (second pass) |
| end A / end B | which end of the line is the start (no sheet marks one) | write it in `--orientation` |

Teaching a start point for NE, MID and X: `D:/AI_Project/.venv/Scripts/python.exe D:/AI_Project/02_code/teach_sheet_start.py`,
touch the point on the line with the MAGNET TIP (rod upright), and paste the printed `--start-x ... --start-y ...` to
Claude at once. Claude prepares the commands. The robot comes down 1.85 cm behind the taught point (towards the robot),
which only just touches a puck standing exactly on it. So place the puck's centre half a centimetre FURTHER FROM THE
ROBOT than the point you touched (even where the line runs sideways or back towards the robot), never closer. For the
normal start the same is done by touching where the ink begins and centring the puck on the round end.

Write the plan number into every command's note, for example `--orientation "plan 02-07, end A, OL1, START MARK"`. The
tool numbers episodes per sheet by itself (sheet 04 goes on at 046); the plan number is ours.

## Sheet 01: straight line (20)

- [ ] 1 to 4: STD
- [ ] 5: OL1. 6: OR1. 7: TL15. 8: TR15
- [ ] 9 to 12: NE, puck 6 cm before the end cap
- [ ] 13 to 16: STD with the other end as the start (sheet turned round, the other cap on the mark)
- [ ] 17: OL1.5. 18: OR1.5
- [ ] 19 to 20: NE, puck 4 cm before the end cap

## Sheet 02: arc (35)

Laid as in the figure from end A, the arc bends to the robot's left; from end B it bends to the right. Both are needed.

- [ ] 1 to 3: STD, end A
- [ ] 4 to 6: STD, end B
- [ ] 7: OL1, end A. 8: TR10, end A. 9: OR1, end B. 10: TL10, end B
- [ ] 11 to 12: NE, end A
- [ ] 13 to 18: STD, three end A, three end B
- [ ] 19 to 21: end A: OL1.5, OR1.5, TL15
- [ ] 22 to 24: end B: OL1.5, OR1.5, TR15
- [ ] 25 to 28: NE, end B
- [ ] 29 to 32: STD, two end A, two end B
- [ ] 33 to 35: NE, end A, puck 4 cm before the end cap

## Sheet 03: two curves, the "2" (45)

End B is tight for reach (calc: 0.2 cm to spare); use it only if its trial reached the goal without strain, otherwise do
those episodes from end A.

- [ ] 1 to 5: STD, end A
- [ ] 6: OL1. 7: OR1. 8: TL10. 9: TR10 (end A)
- [ ] 10 to 12: NE, end A
- [ ] 13 to 17: STD, end B
- [ ] 18 to 21: OL1, OR1, TL10, TR10 (end B)
- [ ] 22 to 24: NE, end B
- [ ] 25 to 30: STD, three end A, three end B
- [ ] 31 to 36: OL1.5, OR1.5, TL15 (end A) and OL1.5, OR1.5, TR15 (end B)
- [ ] 37 to 41: MID, 3 cm before the sharp turn (the end you use most)
- [ ] 42 to 45: NE, two end A, two end B

## Sheet 04: meander (45)

- [ ] 1 to 5: STD, end A
- [ ] 6: OL1. 7: OR1. 8: TL10. 9: TR10
- [ ] 10 to 12: NE, end A
- [ ] 13 to 18: STD, end A
- [ ] 19 to 24: OL1.5, OR1.5, TL15, TR15, OL1, OR1 (end A)
- [ ] 25 to 29: MID, at the middle hairpin
- [ ] 30 to 34: STD, end B (only if its trial reached the goal; otherwise end A)
- [ ] 35 to 38: OL1, OR1, TL10, TR10 (end B, or end A)
- [ ] 39 to 45: NE, four end A (4 to 6 cm before the end cap), three end B

## Sheet 05: loop with a crossing (30)

Start from the lower end (B) only: from the other end the paper would lie under the robot's base. At the crossing the
puck goes straight across, never round the corner.

- [ ] 1 to 6: STD
- [ ] 7: OL1. 8: OR1. 9: TL10. 10: TR10
- [ ] 11 to 13: X, 5 cm before the crossing, first pass
- [ ] 14 to 19: STD
- [ ] 20 to 24: OL1.5, OR1.5, TL15, TR15, OL1
- [ ] 25 to 27: X, 5 cm before the crossing, second pass
- [ ] 28 to 30: NE

## Sheet 06: dashed loop with a crossing (25)

Start from end A only: the end that lies halfway up the sheet, about 4 cm from its right edge in the course photo (not the
end near the bottom edge). Follow the dashes' direction across the gaps; go straight across the crossing.

- [ ] 1 to 6: STD
- [ ] 7: OL1 (puck beside the first dash). 8: OR1. 9: TL10. 10: TR10
- [ ] 11 to 13: X, 5 cm before the crossing
- [ ] 14 to 18: STD
- [ ] 19: OL1.5. 20: OR1.5. 21: TL15
- [ ] 22 to 25: NE

## How to demonstrate (the same way every time)

The model copies what it sees us do, so the same situation should always get the same answer.

1. Push smoothly at a steady, unhurried speed. The block stays behind the puck and touching it; the puck's centre stays on
   the line.
2. Keep the rod upright; let it lean only where the arm cannot reach otherwise (the far end of 04).
3. Turn the forearm only when the line ahead would leave the picture, and turn it back on straight stretches. Watch the
   picture window, not only the sheet.
4. At a crossing go straight across. On the dashed sheet keep the direction across the gaps.
5. Finish with the puck's centre on the end cap (within about 1 cm), then stop.
6. A deliberate offset start is brought back onto the line within the first 5 to 10 cm. Anything else that slips more
   than about 1 cm off the line: discard (x) and do it again.
7. Before Enter at the start: puck placed, hands out of the picture. Never touch the puck once the robot is down.

## What gets labelled, and by whom

- Automatic, every step (`steps.csv`): the picture's name and checksum, sharpness, the action and who chose it, the
  forearm turn, the picture's direction on the table, the robot's pose and its six joints, the times.
- Automatic, every episode (`episode.json`): sheet, mode, operator, times, code version, camera mount, start pose, camera
  shift, status, corrections, counts.
- By you: the answer to "Puck in the goal?" (y or n, never Esc), the discard reason after x (t tipped, o off sheet, r robot,
  h hand, m other), u and l during the replay, and the command-line notes (`--path-id`, `--orientation` with the plan
  number, start end and code, `--lighting`, `--offset-cm`, `--offset-deg`, `--start-index`). Anything odd goes into
  `session_notes.md` (append only).
- Where it is stored: `03_data/raw/<date>_session_NN_<operator>/<sheet>_episode_NNN/` (pictures in `frames/`), one folder per
  day and operator. Never rename or delete anything there; episodes to leave out are listed later in the dataset
  builder's settings.

## Masking the surroundings later: what to keep in mind

- The forearm camera turns, so the sheet sits in a different place in every picture: a fixed crop or a fixed mask cannot
  remove the room. The mask must find the paper in each picture (or come from the camera model and the taught sheet
  corners).
- Whatever mask training uses, the robot must apply the very same function to its live pictures at the exam. One masking
  function in `02_code`, its settings in `config.py`, its version written into every dataset.
- The raw pictures stay unmasked, so a better mask can be applied later.
- A table clearly darker than the paper and nothing white or red near the sheet make a picture-only mask far more
  reliable: choose and clear the table with this in mind.
- A mask from the camera model uses the joints and taught corners as a second input: ask the professor before relying on
  it.

## Tool changes that would help (suggestions, nothing built yet; Emranul chooses)

| Change | Why | Effort (guess) |
|---|---|---|
| "Hands off? Enter" before the first picture, and a key to take it again | 3 of the last 12 first pictures show a hand, 4 show no puck | 20 min |
| Teach fixes: no overwriting of the day's teach file; the magnet tip instead of the rod end; Claude's converter to config values | taught numbers were lost or 9 mm off before | 45 min |
| `--purpose trial / training / check` and `--start-push-deg` | trials and checks marked in the data, not in memory; sheets may lie at any angle | 15 min |
| The episode checker runs by itself after each episode | problems are found while the set-up is still unchanged | 1 h |
| Puck and hand warnings during the replay (the new puck detector) | steps with no puck or with a hand in view are flagged at once | 1 h |
| Automatic backup to a USB drive at the end of a session | nothing in `03_data` is backed up today | 20 min |
| Camera check at one fixed pose at the start of every session | a knocked camera is caught before the first episode | 45 min |
