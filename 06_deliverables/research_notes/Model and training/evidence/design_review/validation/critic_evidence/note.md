# Evidence check of "Validation that predicts the exam, and the robot test" (key validation)

Critic sub-agent, lens EVIDENCE, 9.11.26. Read only: nothing in D:/AI_Project was changed, nothing ran on the robot.

## What I checked

- Every number in `../section.md` against its stated source: `../validation_numbers_output.txt` (V1 to V7, and the
  script `../validation_numbers.py`), `../interim_rule_numbers_output.txt` (I), the run files in
  `D:/AI_Project/.claude/worktrees/training-analysis/04_runs/` (comparison.md, baselines.json, the end-to-end fold files,
  `2026-11-09_resnet18_final_candidate/fold_without_path_01.json`), `../../main_session_facts.md` (F),
  `AUDIT_SYNTHESIS.md` sections 5, 9, 10 and 11 plus `audit/code_actions/note.md`, the literature notes and checks
  (offline_vs_closed_loop, evaluation_statistics, history_and_causal_confusion, action_chunking,
  diffusion_and_generative, augmentation), `02_code/config.py` and `02_code/training_metrics.py`.
- My own re-computations (standard-library Python): `recheck_numbers.py`, printout `recheck_numbers_output.txt`
  (this folder). Parts: the interim rule by a different method (convolution), the interval for one new track, the
  seed noise of the worst-fold measures, and the standard-start episodes against the tool's step budget.

## Verdict: the recommendation stands with changes

The core holds: choose on leave-one-sheet-out and never on held-out episodes (V1: 9.2 / 12.3 points on the six-sheet
mean, 106 against 54 and 119 against 58 right stops of 208), and let a pre-registered, blind, paired robot test decide.
Nearly every number matches its source digit for digit. The problems are in what some numbers are taken to mean, in
measures the noise rule does not cover, and in a few places where the section breaks its own rules.

## What checks out

- Table 1 (all per-sheet values, gaps, means), the stop counts, the dashed-loop 12 against 1 of 25: V1 lines 23 to 59;
  re-scoring equals the stored folds: V1 lines 7 to 17 and 35 to 45.
- Leak figures (80.6 to 81.2 %, 56.3 to 58.1 %, 2.8 %, 23.6 against 52.8 deg, 26.4 against 40.3 %): audit table 5.2;
  0.316 / 0.000 and 0.000 to 0.003: duplicates_leakage-3; 1,175 to 1,736 views and 3,100 labels: duplicates_leakage-7.
- Table 2 and the seed lessons: V2 and V3; the four-seed exact-minus-classes means (+0.5 to +8.7, mean +4.3) I
  recomputed from V2. End-to-end paired differences: V3. Validation-loss epochs: V6. Steps per cm: V5.
- Copying baseline: baselines.json within_45_deg_mean 0.5383 to 0.7358, path 0.6044 to 0.6483. History probe:
  comparison.md rows one_picture and plus_change_since_previous. Final candidate fold 01: 0.7785 of 578, 0.8563 of 529,
  16 / 3 / 1 of 20, 34.6 min (its fold file).
- Robot statistics: V7 and the statistics checker's claims 33 to 38 and 41. Interim rule reproduced independently:
  0.961, stops early 0.481, 18.2 blocks; fixed 24 blocks 0.962; equal models stop early 0.263 (recheck part 1).
- Robot hours 6.3 and 11.4 h: V7 (10 + 162 + 188 + 18 min; slow 21 + 290 + 340 + 33 min).
- Every [supported] or [corrected] literature tag matches its checker (robomimic 12; Codevilla 1 to 3 and 5; Cowling
  23 and 24; Haq corrected; Ross 41; de Haan 42 corrected; Kress-Gazit 1 and 2; Snyder 8 corrected and 10; TRI 13, 14,
  16; Bose and Kuang 24; Brown 27; Liu claim 40 of the chunking check).

## Problems, most severe first

1. **High. The noise rule does not cover the measure ranked first.** The rule gives "about 3 points on the six-fold
   mean, 10 on one sheet"; Table 2 lists per-sheet seed ranges for step within 45 only. V2: path within 45 per sheet
   moved up to 10.7 points (classes, sheet 01) and 16.0 points (exact, sheet 06). The worst fold, which measure (1)
   names, moved 12.4 points over four seeds of the exact probe (0.534 to 0.658, and the worst sheet switches between 06
   and 05) and 8.2 points on step (recheck part 3; comparison.md worst-fold columns). Fix: one threshold per measure
   (path on one sheet about 16, worst fold about 12 for single runs), or compare 3-seed means only.
2. **High. The "range for a new track like these six" is the interval for the six-sheet mean.** V2 prints mean plus or
   minus t x SD / sqrt(6) (validation_numbers.py line 274). For one new track the prediction interval is
   t x SD x sqrt(1 + 1/6): 0.308 to 0.845 (classes) and 0.344 to 0.839 (exact), about 2.6 times wider (recheck part 2).
   Fix: relabel 0.475 to 0.678 as the interval for the mean and add the one-track interval.
3. **High. Measure (3) tunes the stop threshold on held-out episodes of the training sheets.**
   `training_metrics.tuned_stop_threshold` chooses it "on validation episodes only", and those are whole episodes held
   back from the training sheets (config.py line 1735; train_policy.py line 454; F, "How the comparison is set up"). The section itself says held-out episodes must
   never choose "a model, a setting or an epoch" (1.3 point 4; audit 11.2 item 4) and shows they double the right stops.
   The tuned two-in-a-row thresholds run 0.05 to 0.40, not 0.30 (V6 plus the final candidate's fold 01 at 0.4).
   Fix: tune on sheet or block folds (nested), or fix one threshold; report the fixed-0.5 counts beside it.
4. **Medium. The recovery-stratum lesson is selective and inside the noise.** The end-to-end exact lead over classes
   "vanishes" after a correction (V4, one seed, sheets 03, 05, 06). V4 also shows the frozen probes on all six sheets:
   exact minus classes +1.1 points before and +2.4 after a correction, the opposite pattern. The stratum is every row
   after the first correction (5,996 of 15,950 moving rows), harder for always forward too (0.431 against 0.555), and
   the audit finds no measured drift before corrections and most pucks put back within 3 px (155 of 179;
   label_consistency-2, -13). Fix: keep (4) as a reported stratum; drop "which erased the 4.3-point lead" as its reason.
5. **Medium. Other "own traps" fail the section's own noise rule.** History probe: +2.6 points (step mean), +0.7 (path
   mean), worst folds -1.3 and -2.8 (comparison.md), against classes seed ranges of 3.1, 3.4, 2.7 and 4.3 points (V2,
   recheck part 3). "The gap depends on the design (+12.3 against +9.2)": one seed per split; with V2's four-seed
   held-out-sheet means the gaps are +10.9 (classes) and +11.2 (exact), and sheet 06 gives +23.2 and +20.4, not 22.3 and
   26.6 (the default exact seed was the lowest of four there, 0.483 against a mean of 0.545). Fix: call them single-seed
   observations and quote the four-seed versions.
6. **Medium. Track-feature statements.** "Bends under 2.5 cm radius only on 03 and 04" contradicts view_and_tracks-7
   (05: 102, 06: 72 pictures under 2.5 cm) and table 10.1 (smallest radius 2.0 cm on 05, 0.84 to 1.0 cm on 06); the
   audit says such bends are well represented only on 03 and 04. Sheet 03 holds no feature that no other sheet shows
   (pointed corners on 03 and 04; its 148-degree turn is below 04's hairpins; view_and_tracks-5, -16), yet its gap is
   +8.9 / +6.9, which points to remembered views or the held-out sitting, against "these runs cannot tell". The twin
   alignment holds for one of the three twin measures of table 5.1 only (within a step: 01 68.9 %, 04 55.8 %; close
   twin: 01 16.5 %, 04 6.1 %). Fix: correct the bend sentence, state that 03 has no unique feature, name the measure.
7. **Medium. Training length and validation loss.** No held-out-sheet curve exists: fold histories log only
   train_loss and validation_loss, and the exact runs' validation loss was still falling at epoch 8 (sheet 05: 0.5405 to
   0.5349). So "8 epochs" is today's setting, not the result of the stated rule. Worse for the evidence base: every
   end-to-end run except the final candidate kept the lowest-validation-loss weights (config.py line 1801 sets
   is_last_checkpoint_kept False; train_policy.train_fold restores best_state, with patience 3), so the class folds used
   epochs 7, 5 and 7 and the small CNN epoch 15 on sheet 03, all chosen on held-out episodes of the training sheets,
   which the section's own rule forbids; the probes also stop early on those episodes (F). "Validation loss has not
   misled us yet" was therefore never tested. Fix: say which runs chose epochs this way, label 8 epochs a judgement,
   and log held-out-sheet within 45 per epoch in the next runs.
8. **Medium. Exact against classes is read on one measure only.** The section says offline evidence says little here,
   using step within 45 (+0.8 to +3.6). On its own first measure, path within 45, end-to-end exact leads by +9.8, +6.7
   and +8.4 points on 03, 05, 06 (V6); twist sign 0.84 / 0.67 / 0.80 against 0.70 / 0.37 / 0.42 (F). The list drops
   balanced class accuracy, which the audit recommends for class heads (11.2 item 6) and on which classes lead (0.381
   against 0.284, comparison.md). Measure (3) weighs early stops most, and there exact is worse: 25 of 100 episodes
   early at fixed 0.5 and 33 of 100 tuned, against 11 of 100 for classes (V3, V6; never stopped 31 against 70). Under
   the section's success rule both an early stop and no stop fail, so the weighting is a judgement (taken from a
   docstring in training_metrics.py). Fix: show each measure's direction for the two candidates; label the weighting.
9. **Medium. Robot hours leave out sheet changes.** Trials are randomised over about 10 lay-outs (Phase 1) and 22
   (Phase 2), so nearly every block or trial needs a new sheet or end. F's guessed minute covers laying the puck and
   starting; the team's exam-start note (6.11.26) guesses 1 to 2 minutes to lay and tape a sheet. About 86 changes add
   roughly 1.4 to 2.9 h (my calculation from guesses): 7.7 to 9.2 h instead of 6.3. Fix: add the term, or randomise
   inside lay-out groups, and time a sheet change in Phase 0.
10. **Medium. The Phase-1 fallback contradicts the Phase-1 design.** Phase 1 is placed on hard tracks to sit near
    50 % success (TRI, Sec. 3), yet "fewer than half of the Phase 1 trials" cancels Phase 2; by design that fires about
    half the time for an acceptable model. Fix: tie it to a stated bar on the validation tracks or to the known sheets.
11. **Medium. The 2.0 cm puck tolerance is justified by a different quantity.** The rule measures puck centre to end
    cap; offline "stopped right" measures the tip to where the demonstrator stopped (goal_distance_cm). The audit warns
    they differ and lists four tolerances in the notes (1, 1.2, 1 to 1.5, 2 cm; notes_context-23; 11.2 item 7). No
    number shows how many demonstrations would pass 2.0 cm. Fix: give 2.0 cm its own reason (judgement) and check it on
    the Phase-0 photos.
12. **Low to medium. Half of Phase 1 runs on training sheets.** 12 of 24 blocks use sheets 04 to 06, which six-sheet
    candidates trained on; V1 shows such runs reward remembering. Fix: report the two halves apart.
13. **Low to medium. Leave-one-block-out is confounded.** Sheet 04's blocks split at a light change and the 15-px camera
    jump (15 natural and 14 ceiling episodes before it, all end A; 16 after it, 12 end B); sheet 02's 9-episode block is
    all natural light (duplicates_leakage further facts, table 5.4). A drop there need not mean lost twins. Fix: soften
    "can tell which of the two causes the gap".
14. **Low. Validation-track features sit at the edge of the data.** Sheet 06 end B is already 8.0 cm (8.03) from the
    edge and sheet 04 has parts 2.9 cm apart; view_and_tracks-16 asks for more than 8.0 and closer than 2.9. Fix: 10 cm
    or more, about 2 cm.
15. **Low. Step limit against the slow case.** Limit 4 steps per cm; standard starts reach 2.75 (90th percentile) and
    3.07 per cm (V5). At the slow case's 1.5 times the steps that is 4.1 and 4.6, so about one run in ten would be cut
    off. Fix: say so, or set the limit per track from its demonstrations.
16. **Low. Literature labels.** Lin et al. 2025 ("checker: supported"), Xie et al. 2023, Hussenot et al. 2021, Wen et al.
    2020 App. 10.6 and Abbas et al. 2026 App. F were found and read by the checkers, not checked claims with a verdict;
    say so. Xie's 0.4 to under 0.1 is a simulation (Factor World). Newcombe 1998 was not opened by the statistics note;
    Kress-Gazit and TRI use Bayesian posteriors, and TRI do test significance (claim 14). Codevilla 2019's 42 % is a
    table value labelled "variance". "0.56 to 0.83 over all models" mixes Town 1 (0.75 to 0.83) and Town 2 (0.56 to
    0.75). Abbas fixes the list before robot results, not before offline runs. "Cannot rank good ones" is stronger
    than the sources (0.21 to 0.55 among the better half).
17. **Low. Small mismatches.** The tool's budget cuts off 36 of 71 standard-start episodes with its own line lengths
    (config 24, 27, 55, 77, 57, 57 cm), 46 of 71 at 2 per cap-to-cap cm (recheck part 4): "more than half" holds only
    narrowly. "Both models start from the same picture": put-backs were more than about 5 mm off in 24 of 179
    corrections. "Counts of 208": the cleaned set of the final candidate has 209 episodes (20 and 578 moving rows on
    sheet 01, against 19 and 569). The primary Wilson interval on k/30 ignores the clustering the section quantifies
    (checker: 0.934 coverage at a guessed 0.15). "About 24 trials per model are enough to choose" assumes a 20-point
    gap (90 against 80: 0.834).

## Files

- This note: `C:/Users/e7dc/AppData/Local/Temp/claude/D--AI-Project/997601d4-4a2e-4eb2-91d1-890ff53278fd/scratchpad/design/validation/critic_evidence/note.md`
- Re-check script and printout: `.../critic_evidence/recheck_numbers.py`, `.../critic_evidence/recheck_numbers_output.txt`
