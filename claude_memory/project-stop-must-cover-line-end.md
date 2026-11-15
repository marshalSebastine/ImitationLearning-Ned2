---
name: project-stop-must-cover-line-end
description: Emranul (11.11.26, first robot runs) wants the puck to cover the line's end completely when the model stops; a visible bit of the end is "okay but not perfect"
metadata:
  type: project
---

First robot runs with the model mode, 11.11.26 (run folders under `03_data/model_runs/`): Model 4 (chunk, stop weighted
25) on sheet 01 stopped by itself with the puck on the line's end, but a little of the line's end still showed. Emranul:
"I want the puck to cover the whole thing ... keep this in mind when we retrain".

**Why:** the exam judges whether the puck reaches the end; a stop a little short may count as not finished.
**How to apply:** when retraining or recording corrections, aim for stops with the puck fully over the end cap. Options
to weigh (none tried yet): check where the demonstrations' stop pictures put the puck relative to the end cap; record
new end-of-line demonstrations that push until the end is covered; or a small fixed final push after the stop rule
fires, tested on the robot first ([[feedback-test-before-hardware-changes]]).

Second point (Emranul, 11.11.26, Model 4 on sheet 02 placed as in plan 02-01, 41 steps, stopped by itself): "just before
the middle the puck was not aligned in the middle of the black line" - the puck should stay centred on the line the whole
way, not only reach the end. And a third finding of the same session: with sheet 02 lying straight instead of turned as in
the recordings (line start pointing straight away from the robot at the start mark), Model 4 failed at once, so
retraining needs demonstrations whose line starts at a slant or with the sheet turned, if the exam may place lines so.
