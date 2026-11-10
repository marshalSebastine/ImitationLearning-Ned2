---
name: project-model-training
description: "Model and training analysis of 9.11.26 (Model and training session for Emranul): first set-up = fine-tuned ResNet-18 predicting the exact step and forearm twist with its own stop output; leave-one-sheet-out results; executor and stop rules; backup; validation rules; where code, results and report are"
metadata:
  type: project
---

On 9.11.26, at Emranul's request ("set up the perfect setting so we can make multiple models and compare"), the Model and
training session built a comparison set-up and analysed the data, the literature, 13 frozen-feature experiments and 5
end-to-end training runs (more were still running that night). Report:
`06_deliverables/research_notes/Model and training/2026-11-09_training_set_up_analysis.md`; code `02_code/training_*.py`,
`train_policy.py`, `compare_runs.py`; settings `TRAINING_*` in `config.py`; results `04_runs` (branch
`emranul/training-analysis`). Nothing ran on the robot.

**First set-up** (all three judges of the design review agreed): `TRAINING_RUNS["resnet18_final_candidate"]`, ImageNet
ResNet-18 fine-tuned end to end on one 160 x 120 picture, a 3 x 4 grid head; it predicts the exact step (tip forward
and left in the picture's frame, forearm twist; lean and hand trained but not executed) and has its own stop output
trained on the 259 stop pictures; the audit's cleaning; 8 epochs on half the pictures each; the last weights kept.
Leave-one-sheet-out, steps within 45 degrees: 0.779, 0.632, 0.680, 0.642, 0.649, 0.684 on sheets 01 to 06 (always
forward 0.747, 0.565, 0.522, 0.485, 0.509, 0.402). Stop at a fixed 0.5 with two pictures in a row: 102 right, 25 early,
69 never of 209 held-out episodes (the weakest part, worst on the dashed loop).

**Executor rules** (the model sees only the picture): a fixed 1 cm step in the predicted direction (the predicted
lengths are too short), the predicted twist clamped to 10 degrees per step and the forearm within plus or minus 115
degrees, wrist and hand held, joints 1 to 3 from `side_view_test.solve_with_wrist_fixed`, the step budget on. The
recording tool's keyboard and gamepad paths choose the least lean and would show the model views it never saw (a
median 48 to 56 px shift), so a new model mode is needed (code_actions note in the audit).

**Measured lessons:** fine-tuning beats frozen features by about 10 points (frozen DINOv2 and CLIP no better than
ResNet-18); a small CNN from scratch is about 2 points behind ResNet-18; the exact step beats the 8 move names (about 4
points over seeds, twist far better); class weights, picture history, a line-and-puck input instead of colour, and
dropping score 1 did not help (mirroring and chunks were still running); more different tracks help more than more
episodes of the same tracks. Only leave-one-sheet-out
measures unseen tracks: episode splits leak through recording blocks. Never pick checkpoints on validation loss; the
robot decides.

**Backup** (two of three judges): names-plus-exact (`resnet18_backup_names_plus_exact`), because the first set-up backs
up and recovers poorly after corrections; its six-fold run and a second seed of the first set-up were still running on
the night of 9 to 10.11.26.

**Why:** the exam grades finishing first and time second, on the six sheets and unseen tracks; the data audit and the
literature (`evidence/` beside the report) back each choice.

**How to apply:** build the first set-up and the tool's model mode before any new recording; ask the professor the
report's questions (joints in the controller, correction rounds, printing tracks); compare models only with
leave-one-sheet-out and on the robot. Related: [[project-recording-tool]], [[project-forearm-camera-mount]],
[[ai-project-workspace-setup]].
