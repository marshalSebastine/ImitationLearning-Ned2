# Why the model stops wrongly on new lines (14.11.26, evening)

Asked for by Emranul after the mid-way check (`2026-11-14_midway_check.md`): steering on unseen new tracks improved, but
only 7 of 31 episodes stopped right. Question: where and why does the stop go wrong? Read-only on the data; no robot,
no GPU. Script, per-episode results and picture sheets: `evidence/stop_check_2026-11-14/`.

## What was done

For each of the 31 held-out new-track episodes, the scorer's fixed rule (stop chance at least 0.5 on two pictures in a
row) was replayed on the saved predictions of `2026-11-14_resnet18_round_2_part_new_shapes`. The picture where the
model would stop (or, if it never stops, where its stop chance peaks) is shown beside the picture where the
demonstration really stopped (`stops_early.jpg`, `stops_never.jpg`, `stops_2_to_3_cm_short.jpg`). Two end pictures
per course sheet and end are in `round_1_end_pictures.jpg`. Same counts as the scorer: 7 right, 9 too early, 12 never,
3 stopped 2 to 3 cm short.

## Finding: the model learned "stop near the paper's edge", not "stop at the line's end"

Measured from the set-up geometry (`setup_visuals.all_setups()`: planned paper outline and end cap, distance from the
end cap to the nearest paper edge):

- On every course sheet, both ends, the line ends 2.4 to 4.3 cm from the paper's edge. Round 1 (209 episodes) never
  shows a line ending in the middle of the paper.
- New tracks that also end near the edge (T01, T02, T04, T06, T08, T09: 2.2 to 3.2 cm): 21 episodes, 7 stopped right
  and 3 stopped 2 to 3 cm short.
- New tracks that end in the middle of the paper (T03, T05, T07: 7.7 to 9.8 cm from the edge): 10 episodes, **none**
  stopped right; the stop chance at the real end was 0.00 to 0.17 in all ten.

Measured from the rod tip's position (frames.csv) against the planned paper outline, on the 2,068 held-out pictures
more than 3 cm before the real end:

| Rod tip to the paper's edge | Pictures | Model says stop |
|---|---|---|
| 0 to 4 cm | 856 | 5.8 % |
| 4 to 6 cm | 500 | 2.0 % |
| 6 to 9 cm | 590 | 0.2 % |
| 9 cm or more | 122 | 0.0 % |

All 9 early stops happened with the tip 0.1 to 4.7 cm from the paper's edge (7 of them within 3 cm), mostly where a
hairpin or corner runs the line along the edge (T04, T06, T08).

Caveats: the paper outline is the planned placement, not measured on the table (the lay-out line puts it within about
a centimetre); the tip is the rod's tip, the puck sits just ahead of it. A first try with a pixel count of "white
paper in the top half of the picture" did not show the pattern (glare on the table counts as white; the top half is
mostly table in nearly all pictures), so the measure above uses the geometry instead.

## What it means

- More recordings like round 1 do not fix the stop: 244 of the about 260 episodes end near a paper edge.
- Ends in the middle of the paper: T03, T05, T07, T10 recorded (13 episodes), T13 and T15 still to record (6 episodes).
- Exam risk: if the unseen exam line ends far from its paper's edge, the current recipe is unlikely to stop there.

## Options (none decided)

1. Training: count the stop pictures of mid-paper ends more (weight or sample them more often), then repeat the same
   held-out check with T03, T05, T07, T10, T13 and T15 held out. No robot time.
2. Recording: lay a blank white sheet under the track sheet beyond the end, so the paper goes on past the end cap
   (for tracks that end near the edge). Not tested; asked of Emranul for T14 and T16 on 14.11.26.
3. Later: re-record some course-sheet episodes with such a sheet beyond the end.
