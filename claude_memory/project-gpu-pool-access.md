---
name: project-gpu-pool-access
description: "GPU pool SSH state on 10.11.26 - VPN works, ki-001/003/004/005 answer, Emranul's e.hoque login refused; Tamara Graf on parental leave, Adel replaces her"
metadata:
  node_type: memory
  type: project
  originSessionId: 997601d4-4a2e-4eb2-91d1-890ff53278fd
  modified: 2026-11-10T08:59:37.122Z
---

10.11.26: with the OTH VPN on, ki-001, ki-003, ki-004 and ki-005 answer on port 22 (ki-002, 006 to 010 do not). ki-003 is
10.10.134.106, ED25519 key SHA256:hnOtnqAGx5e9juMxCZul1Mh5F5DoGQOQIHT4plwE+fs. Login `e.hoque@oth-aw.de@ki-003.oth-aw.de`
gave "Permission denied" although the same password works on Moodle (checked by Emranul), so SSH access is probably not
activated for him. Emranul says Tamara Graf (named in the OTH guide) is on parental leave; a colleague named Adel took her
place (e-mail not known yet). Emranul e-mailed Adel on 10.11.26 asking to activate SSH for e.hoque@oth-aw.de; waiting for his answer.

**Why:** the laptop CPU takes 20 to 30 min per training fold; the pool would make trainings minutes.
**How to apply:** don't retry the password repeatedly (lockout risk); ask Adel or Prof. Nierhoff to activate SSH for
e.hoque@oth-aw.de (and Marshal's account), then continue: read-only GPU check, copy code and data, install packages.

**Update 10.11.26 (Adel's answer):** ki-003 is closed to SSH; SSH works on ki-013 to 016 and ki-025 to 032 (at 10:40, ki-014, 015,
016, 028 to 032 answered; 013 and 025 to 027 did not). Login `ssh e.hoque@oth-aw.de@ki-014.oth-aw.de` works; the account on
the pool is `e7dc@oth-aw.de`, home `/home/e7dc`. ki-014: one NVIDIA RTX 2080 (8 GB, CUDA 13.0 driver, idle at 10:45), 24
cores, 62 GB RAM, 2.8 TB free on /home, Python 3.12.11, Ubuntu 22.04. Lab rules in /share/labor (the .md is the 2024
version; the PDFs of 01.11.2025 are valid); the 2024 text forbids changing the lab computers' configuration, copying
software or data, or installing programs, so read the current PDF before installing packages or copying data.
ki-014 already has PyTorch 2.8.0+cu129 with CUDA working (`torch.cuda.is_available()` True), conda at /opt/miniconda and
pip3. The valid rules (1.11.2025, LaboratoryRegulations.pdf) keep the sentence "It is forbidden to change the configuration
of laboratory computers, copy software or data, or install programs brought from home"; whether copying our own code and
data into /home/e7dc and installing packages there is allowed: Adel said yes (e-mail to Emranul, 10.11.26).

**Set up on ki-014 (10.11.26):** code and processed data in `~/ai_project` (copied with scp, Emranul types the password; no
login key yet, creating one was refused here), private environment `~/ai_project/.venv` (system PyTorch 2.8 cu129 plus the
laptop's timm, pandas, scikit-learn, numpy, scipy versions). `train_policy.py` got a "device" setting ("auto": GPU when
present). Check run, first set-up held out on sheet 03: GPU 0.703 within 45 deg in 1.7 min (12 s per round); laptop 0.680
(seed 1) and 0.673 (seed 2), about 17 min when running alone. GPU runs are not bit-identical to the laptop's, so compare
runs from the same device.

**GPU comparison, 10.11.26 (ki-014, all six held-out sheets, results copied to the worktree's `04_runs/gpu_ki014/`,
not committed by Emranul's wish):** within 45 deg / path within 45 deg / stop right-early-never of 209 (fixed 0.5, two in a
row): first set-up seed 1 0.681 / 0.787 / 92-30-82; seed 2 0.675 / 0.772 / 104-30-71; backup names plus exact 0.651 / 0.710
/ 91-24-91; "no augmentation" 0.612 / 0.659 / 50-6-151 (of 208), but that run is the plain classes set-up without
augmentation, not the first set-up without it, so it does not answer whether augmentation helps. The first set-up stays
first; stops vary by about 12 episodes between seeds. The laptop queue was stopped on Emranul's request at 11:05.
Fair augmentation test (ki-014, 10.11.26 about 12:40 to 13:05): the first set-up with only augmentation off scores
0.669 / 0.674 within 45 and 0.778 / 0.775 path (seeds 1 / 2) against 0.681 / 0.675 and 0.787 / 0.772 with it, so under 1
point on the means; on the 25 daylight episodes augmentation is ahead in all four pairings (about 2 points step, 3 path).
Augmentation stays. Correction: the sheet-03 check run (0.703) differs from the full run (0.689) partly because
`--folds` renumbers the folds, which changes the seed and check set, so it is not a pure GPU-noise measure. The report's
rule 2.5 now says drop the backup (path 6 to 8 points lower); the decision is Emranul's and Marshal's.
Model 1 (10.11.26, about 14:40): the first set-up trained on all six sheets (18,032 pictures, no check pictures, `split`
"all" added to train_policy.py, runs resnet18_model_1_all_sheets and _seed_2 in config.py, not committed), 2.3 min per
seed on ki-014, 10.7 ms per picture on the laptop; copied to the worktree's 04_runs/gpu_ki014/2026-11-10_*model_1*.
Loads on the laptop and gives sensible outputs (stop chance 0.94 and 0.89 at a stop picture); it has no offline score of
its own, the robot is its test.
Night of 10 to 11.11.26: a queue of 11 trainings started on ki-014 at about 23:55 (night_queue.sh: final-candidate seeds
3 to 5, chunk with the final recipe x2, 16 rounds, and all-sheets copies seeds 3 to 5, model 2 chunk, model 3 16 rounds);
log ~/ai_project/logs/night_queue.log. In the morning: copy ~/ai_project/04_runs/2026-11-10_* back, score the six-sheet
runs (5-seed team included), pick at most 3 models. model_mode.py (robot driver, not committed) and the checklist
06_deliverables/research_notes/Model and training/2026-11-11_first_robot_session_checklist.md exist; nothing ran on the
robot. Emranul's OTH password showed as plain text in a terminal tab on 10.11.26; he was advised to change it.
11.11.26 morning: night queue results copied back; chunk recipe steers best (path 0.817 / 0.802) but stops worse; team of
the 5 model-1 seeds path 0.807 (beats every single seed), weight soup of them failed (0.519); 16 rounds no better. Four
robot candidates (A model 1, B team of 5, C chunk model 2, D 16-round model 3) are in the robot checklist with commands.
Round 1 (five chunk ideas x 2 seeds) started on ki-014 at about 08:35 for information only (Emranul: no overfitting).
20 printable A4 test tracks (6 practice, 4 validation, 10 sealed test) made by make_test_tracks.py into
06_deliverables/research_notes/Model and training/test_tracks/ (not committed).
11.11.26 round 1 (ki-014, 08:39 to 09:45) by a rule fixed before the results: chunk + stop weight 25 (right stops 103 / 105,
path 0.808 / 0.800) and chunk + ResNet-34 (95 / 101, 0.813 / 0.807) passed; twist focus, 8-step chunk, EfficientNet-B0 did
not. Round 2 (10:56 to 11:25): their all-sheets copies are models 4 and 5 (robot candidates E and F, 15 and 22 ms per
picture on the laptop); the combination (model 6) failed the rule (too-early stops 55 / 47). Checklist lists A to F.
