# Model and training: what is in this folder

Written 9.11.26 by the Model and training session for Emranul and Marshal.

| File | What it is |
|---|---|
| `2026-11-09_training_set_up_analysis.md` | The report: what to build first, the backup, the evidence, the plan to the deadline and the questions for the professor. Start here. |
| `evidence/audit_synthesis.md` | The data audit of 9.11.26: ten areas, 165 findings, each re-checked by two independent agents (recompute and challenge). Section 11 lists what the data forces on training. |
| `evidence/corrected_ends.csv` | The start end (A or B) of every episode as the audit corrected it; `02_code/training_targets.py` reads it when `is_audit_cleaning` is on. |
| `evidence/literature/<question>.md` | The literature behind each design question (nine questions), every claim with its source and where in it; `<question>_check.md` is the independent check of those claims against the sources (360 of 397 claims supported as stated, 36 corrected, 1 source could not be opened). Useful for the literature part of the written documentation. |
| `evidence/design_review/` | The design panel of 9.11.26: five drafted sections (targets, models, training settings, validation, improvement), two critics per section, three judges who ranked whole set-ups, with the scripts and printouts behind their numbers (the scripts still name the scratch folder they ran in). |
| `evidence/report_parts/`, `evidence/report_check/` | The parts the report was built from, including every measured experiment number (`results_so_far.md`), and the three independent critiques of the report (evidence, beginner reader, completeness). |
| `evidence/laptop_queue/` | The script that ran the end-to-end trainings one after another, and the logs of every run. |

The code that produced the numbers is in `02_code` (`training_dataset.py`, `training_targets.py`, `training_metrics.py`,
`training_features.py`, `training_probes.py`, `train_policy.py`, `training_error_analysis.py`, `compare_runs.py`,
`test_training.py`); the settings are the `TRAINING_*` dictionaries in `02_code/config.py`; the results are in `04_runs`
(`04_runs/comparison.md` puts them side by side). How to run each step: `02_code/HOW_TO_RUN.md`, section "Train and
compare models".
