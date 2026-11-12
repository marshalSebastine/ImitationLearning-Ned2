# Robot test of 11.11.26: results of the check, complete

Eight analysts each took one question about the ten robot runs of 11.11.26; each analysis was then checked by an
independent verifier who recomputed the numbers with own code. This file holds every answer in full, unedited. The
analysts' and verifiers' scripts, outputs and figures are in the folders next to this file (one per question, the
verifier's in its `verify` subfolder). Nothing in it was run on the robot; nothing was trained.

Kinds of statement: measured (from logs, pictures or data), calculated (from a model or replay on paper), guessed.
A verdict is `confirmed`, `corrected` (the detail gives the right version), `refuted` or `unverifiable`.

## Step length (folder `step_length`)

**Analyst's answer.** Yes for sheet 03, no for sheet 04. At the first bend of sheet 03 (and at the start of run 3) the fixed 1 cm moved the tip 2 to 3 times as far as the model asked, so the forearm turned at half the demonstrated rate per centimetre and the tip swung 2 to 3.5 cm too wide; on sheet 04 the model itself hardly turned the forearm and then had no opinion at the hairpin. The driver should carry out the model's own length (gain 1.0, at most 1.2 cm, twist gain kept equal) with a stall guard; a threshold on the predicted length cannot detect twists in place (area under the curve 0.63).

**Verifier's overall verdict.** Partly. Every measured number I recomputed with my own code came out the same or within a few millimetres or degrees (asked against done, the halved forearm turn per centimetre, the 2 to 3 cm overshoot at the first bend of sheet 03 measured three independent ways, AUC 0.63, gains 1.52 and 1.56), so the mechanism is real and "no for sheet 04" stands. But "yes for sheet 03" is a plausible mechanism, not a shown cause: run 10 passed the same bend with the same 1 cm driver and even shorter asks, and everything about the fix is open loop. The fix is also weaker for the exam's unseen line than the answer says: on unseen sheets the model's own length keeps 0.63 to 0.79 of the demonstrated NET turn per centimetre (not 0.89 to 0.95), behaves almost like a plain fixed step of 0.67 cm, and makes the stop come about 0.8 cm earlier, against Emranul's wish that the puck covers the line's end.

### Verdicts on the claims

1. **confirmed**: The driver did 1.0 cm on every decision; over the 545 moves the model asked a median of 0.61 cm, under 0.5 cm on 34.5 %, over 1.0 cm on 14.3 %, largest 1.26 cm; per-run asked medians and paths.

   Measured again from the ten steps.csv (verify/v01_asked_against_done.py): done median 0.999 cm (5th to 95th percentile 0.973 to 1.024, 544 moves); asked median 0.606 cm, 34.5 % under 0.5 cm, 14.3 % over 1.0 cm, 1.10 % over 1.2 cm, largest 1.260. Per run, asked median (asked path / done path, cm): 0.98 (23.5/24.9), 1.14 (25.6/24.2), 0.53 (23.2/37.1), 0.87 (32.9/40.5), 0.55 (31.9/59.3), 0.31 (26.0/67.8), 0.83 (68.9/87.8), 0.82 (32.3/40.0), 0.49 (44.1/89.2), 0.55 (37.5/72.4). The logs themselves are sound: model E on the 60 logged pictures of run 5 and the team B on the 74 of run 10 give back the logged step, twist and stop chance to 0.0000 (v11_team_length.py). All my scripts and outputs are in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/step_length/verify/ .

2. **confirmed**: Where the puck left the line, read from the pictures: run 3 at decision 3, run 5 at 9, run 6 at 10, run 9 at 16, run 10 at 29, run 8 at 29; the held runs 1, 4 and 7 also drifted 1.2 to 2.0 cm.

   Checked by my own eyes on contact sheets of the logged pictures (verify/v02_*.png), not by their code: run 3 beside the arc from picture 3 to 4; run 5 on the outer side of the corner at 8 to 13, on the ink at 14 to 15, left of the line from 16; run 6 straight on at 9 to 11, in the white between the lines at 12 to 13, on the hairpin at 15 to 17; run 9 like run 5 (left of the line from 16 to 17); run 10 on the ink through the first bend and inside the first curve from 29; run 1 beside the line from about 20; run 4 beside the arc at 24 to 28. Run 8 (dashed line) cannot be judged to the decision by eye and stays uncertain, as the analyst says. I did not recompute the centimetre distances from the ink (they rest on the fitted camera model).

3. **confirmed**: Before the puck left the line the model asked short steps only in runs 3, 5 and 9 (median 0.41, 0.45, 0.47 cm, twists of 4 to 6 degrees), so the driver moved 2.0 to 2.9 times the asked distance; in runs 6, 8, 10 the asks were 0.81, 0.87, 0.53 cm.

   Measured (v10_small_checks.py, steps.csv split at their decisions): 0.41, 0.45, 0.47 cm; under 0.5 cm on 100, 67 and 62 % of those decisions; twist median 4.9, 5.4, 5.2 degrees; 1 cm is 2.1 to 2.4 times the median ask. Runs 6, 8, 10: 0.81, 0.87, 0.53 cm. Run 6 after the loss: median 0.23 cm over 58 moves.

4. **confirmed**: First bend of sheet 03: the eight demonstrations turned the forearm 60 degrees within 6.7 cm (4.65 to 7.68) in 11 steps (5 to 13), 2.5 of them twists in place (0 to 3); the robot needed 16.3, 11.2, 12.6 cm; 3.8, 5.0, 4.0 degrees per cm against about 9.0 demonstrated.

   Measured with joint 4 and tip positions only (v04_first_bend.py): demonstrations 6.68 cm (4.65 to 7.68), 11 steps (5 to 13), 2.5 twists in place (0 to 3; the same from labels and from positions); robot 16.34, 11.23, 12.60 cm; 3.85, 5.00, 4.00 degrees per cm over the first 15 decisions against 9.4 in the demonstrations at the 60-degree point (8.0 to 13.0) and 9.2 over their first 15 steps (6.9 to 10.3).

5. **confirmed**: The robot's tip overshot the first bend of sheet 03 by +3.5, +2.2 and +0.9 cm (runs 5, 9, 10; 7.9, 6.7, 5.3 cm against 4.4 demonstrated); in runs 5 and 9 the puck was pushed 0.8 to 1.4 cm beyond the bend; in run 10 it stayed on the ink.

   Measured again WITHOUT laying demonstrations onto runs (each episode in its own frame, puck found again with PuckTracker on every picture; v05_overshoot_own_frame.py): tip furthest along the first leg 7.90, 6.69, 5.49 cm against 4.67 cm in the demonstrations (4.31 to 5.38): overshoot +3.2, +2.0, +0.8 cm. Puck furthest 5.85, 5.31, 4.43 cm against 4.51 (3.24 to 4.86): pushed 1.3 and 0.8 cm too far in runs 5 and 9, not in run 10. A third measure that needs neither puck nor camera model (v14): at the forearm turn reached after 15 decisions the demonstrations' tips were 4.5, 5.6, 4.6 cm from their start (medians), the robot's 9.5, 8.6, 8.2 cm.

6. **confirmed**: What the model asked at that bend was right: the same answers carried out with the asked length turn 60 degrees within 7.4, 5.2, 5.1 cm and give 8.3, 10.4, 9.0 degrees per cm.

   Calculated (v04): 7.9, 5.3, 5.4 cm when the asked twists are summed (the analyst used the logged joint 4, hence 7.4, 5.2, 5.1) and 8.3, 10.4, 9.0 degrees per cm. Stronger support (v14, replay through the driver's own model_step on paper): the same answers with the asked length put the tip 4.5, 4.1, 3.6 cm from its start at the same forearm turn, where the demonstrations had it at 4.5, 5.6, 4.6 cm (medians): the directions were about right and the length was the driver's doing. Caveat the analyst understates: in runs 9 and 10 the as-asked tip is 1.0 to 1.5 cm SHORT of the demonstrated place, so gain 1.0 may cut the bend on the inside.

7. **unverifiable**: The fixed step is the main cause of the loss in runs 5 and 9, and by the same pattern in run 3.

   Plausible, not shown. For it: across all runs, decisions with an ask under 0.5 cm AND a twist of 4 degrees or more, taken while the puck was on the line, were followed by the puck leaving the centre within 4 decisions in 62 % of 26 cases, against 18 to 23 % for every other kind (v13_association.py, using their puck distances); the pictures were fresh, not one move late (v18: the view's turn fits the same move, misfit 0.4 to 0.5 against 0.6 degrees for a late picture; no repeated picture). Against it: run 10 had the same driver and even shorter asks at the bend (0.13 to 0.50 cm at decisions 3 to 11), turned slower per cm (4.0) than run 9 (5.0), and kept the puck on the line there (0 of its 8 such decisions were followed by a departure); what differed was the asked direction (run 10: -41 to -58 degrees at decisions 2 to 6, run 9: -31 to -44) and 1.2 to 2.4 cm less tip overshoot. Run 3 rests on three decisions on a sheet that lay unlike any plan. Short asks also mark an unsure direction (their own finding), so the association is not clean. Only the robot test decides.

8. **confirmed**: Sheet 04 (run 6) did not fail because of the step length: asks of 0.69 to 0.81 cm before the loss, forearm turned only 13.6 degrees where 8 of 11 demonstrations turn 60 degrees within 9.1 to 11.5 cm; at the hairpin a median ask of 0.16 cm turned into 32.0 cm of back and forth for a net 1.7 cm.

   Measured (v04, v10, v09): asks 0.69 to 0.81 cm in the six decisions before decision 10 (median 0.81 over decisions 0 to 9); joint 4 turned +13.6 degrees by decision 10 (largest 15.3 in the whole run; 14.5 as asked); 8 of 11 demonstrations reach 60 degrees within 9.10 to 11.54 cm, the other three after 33 to 73 cm (the analyst's 49 and 52 cm for 04-14 and 04-15 count one turning sense only). Hairpin, decisions 19 to 50: ask median 0.16 cm (largest 0.42), twist 0.85 degrees, 32.0 cm driven for a net 1.7 cm, direction flipped on 20 of 31 changes. By eye the puck runs straight on at pictures 9 to 11.

9. **confirmed**: Offline the predicted length does not separate twists in place from real moves on held-out sheets: AUC 0.63 for both recipes (0.51 to 0.74 per sheet); a 0.3 cm threshold misses 93 % (E) or 90 % (A) and freezes 4.3 % or 6.6 % of real moves; equal-error threshold 0.64 / 0.59 cm with 41 % errors.

   First I checked their fold_outputs npz against my own inference through a different input path (raw JPEG through model_mode.ModelPolicy, 60 random pictures per fold, 6 folds, 2 recipes: every difference 0.0000; A-recipe angles equal the saved predictions on all rows, E's chunk equals the saved chunk; v06). Then my own statistics on the cleaned labels (v07): 1,688 twists in place against 16,008 real moves; AUC 0.626 (E) and 0.626 (A), per sheet 0.51 to 0.74; medians 0.595 against 0.685 cm (E), 0.552 against 0.650 (A); at 0.3 cm missed 93.2 / 89.7 %, frozen 4.3 / 6.6 %; equal error at 0.64 / 0.59 cm with 40.9 / 40.8 %; 13 to 14 % of the rows under any threshold are twists in place (9.5 % of all rows).

10. **confirmed**: Median gain from predicted to demonstrated length for moving rows: 1.52 (E), 1.56 (A); ratio of medians 1.59, 1.68; 1.13 to 1.15 on sheet 01, 1.48 to 1.68 on sheets 02 to 06.

   Measured (v07): 1.515 and 1.561; ratio of medians 1.589 and 1.676 (predicted median 0.685 / 0.650 cm against 1.089 demonstrated); per held-out sheet 1.14 and 1.15 on sheet 01, 1.48 to 1.67 elsewhere.

11. **confirmed**: A short predicted step means an unsure direction, not a short demonstrated step: correlation of predicted with demonstrated length 0.22; direction within 45 degrees on 33 to 49 % under 0.3 cm and 84 to 98 % at 0.9 cm or more.

   Measured (v07): correlation 0.219 (E), 0.217 (A); where the model predicts under 0.3 cm the demonstrations moved a median 0.85 cm (E) and 0.90 cm (A); direction within 45 degrees of the demonstrated step: 44 % / 41 % under 0.3 cm, 90 % / 87 % at 0.9 cm or more. Near the line's end the short length does carry meaning: in all four runs that reached the end the asks fell to 0.5 and then 0.2 cm as the stop chance rose (v15).

12. **corrected**: Length and twist shrink together (0.71 and 0.64), so the model's own length keeps 0.89 (E) and 0.95 (A) of the demonstrated turn per cm against 0.60 and 0.62 for the fixed step and 0.56 and 0.60 for the backup of report 4.4; landing error on bend rows 0.59 against 0.78 cm.

   With their measure (sum of ABSOLUTE twists over sum of lengths, all rows) I get the same: shrink 0.714 and 0.636; shares 0.600/0.617 fixed, 0.892/0.953 own length, 0.685/0.736 gain 1.5 capped at 1.0, 0.563/0.603 backup; landing error on bend rows 0.587 against 0.775 (E) and 0.587 against 0.781 (A). But that measure counts twist jitter on straight stretches as turn. Counting the NET turn over windows of 10 rows in the demonstrated sense (v08): own length keeps 0.74 (E) and 0.79 (A) of the demonstrated turn per cm, and in sharp stretches (net turn of 30 degrees or more) 0.63 and 0.68; fixed 1 cm 0.50/0.51 and 0.42/0.43; backup 0.47/0.50; fixed 1 cm with twist gain 1.6 0.79/0.81 and 0.65/0.67. The ranking holds, but the own length removes about half of the deficit, not most of it. And (v20) on unseen sheets the predicted length is nearly the same at bends as on straight parts (0.63 against 0.70 cm for E, 0.60 against 0.68 for A; demonstrated 0.67 against 1.17), so a plain fixed step of 0.67 cm gives almost the same offline numbers as the own length (sharp-stretch share 0.62 against 0.63 for E, 0.66 against 0.68 for A; landing error on bend rows 0.61 against 0.59 cm).

13. **confirmed**: On sheet 03 as an unseen sheet the fold models ask a median 0.50 to 0.53 cm at the first bend; own length gives 0.69 (E) and 1.04 (A) of the demonstrated 8.7 degrees per cm, a fixed 1 cm 0.36 and 0.53.

   Calculated (v08, part B): demonstrated 8.68 degrees per cm; own length 5.95 (0.69) for E and 8.99 (1.04) for A; fixed 1 cm 3.15 (0.36) and 4.58 (0.53); predicted length medians 0.53 and 0.50 cm. Added: over those rows E asks for only 0.52 of the demonstrated total turn, A for 0.74.

14. **confirmed**: The proposed rule on the first 15 decisions of runs 5, 9, 10: steps 0.18 to 0.73, 0.34 to 0.71, 0.13 to 0.73 cm; travel 6.8, 7.1, 6.4 cm (net 4.4, 3.9, 3.4) instead of 15.0, 15.1, 14.8 (net 9.5, 8.6, 8.2); the same forearm turn; the stall guard acts only in run 6 (nudges at 23, 29, 35, stalled at 41), once in run 9 (87) and ends run 10 at decision 61 as stalled near the end.

   Replayed on paper through the driver's own model_step with step_length_rule predicted and a 1.2 cm cap, so with the headings the arm would really have (v09_rule_replay.py): same step ranges; travel 6.8, 7.1, 6.4 cm; net 4.5, 4.1, 3.6 cm (their logged headings are up to 5 to 7 degrees off, hence 4.4, 3.9, 3.4); turn -57, -74, -58 degrees; 8.3, 10.4, 9.0 degrees per cm. My own guard gives the same events: run 6 nudges at 23, 29, 35 and stalled at 41; run 9 one nudge at 87; run 10 stalled near the end at decision 60 (the 61st decision). Beyond the first decisions a whole-log replay means little: through model_step the replay of run 5 ends with a refused step at decision 27 (tip at the workspace edge), an artefact of the open loop.

15. **confirmed**: Numbers behind the stall guard and the clamp: demonstrations are under both limits in 0.24 % of 16,519 six-step windows, the models' own steps in 0.35 % (E) and 0.57 % (A); under 0.3 cm the current direction is within 45 degrees on 50 %, the last sure direction on 29 to 34 %; 74.5 % of demonstrated steps are 1.2 cm or shorter; real moves median 1.09, 75th percentile 1.23 cm; over 1.2 cm asked on 1.1 % of robot moves and 0.7 % of held-out pictures.

   Measured (v08 part C, v10, v17): 0.25 % of 16,728 windows (I count the windows slightly differently), 0.35 % and 0.57 %; 50 % against 34 % (E) and 29 % (A); 74.5 %; 1.088 and 1.227 cm; 1.10 % and 0.68 / 0.70 %; the kept direction was used 3 times, all in run 6. Added: per RUN the guard is not rare. With the models' own steps 11.5 % of the 209 demonstrations have at least one stall event (v16); in the guard's own situation the summed direction is within 45 degrees of the demonstrated path on 70 % (E) and 59 % (A) and more than 90 degrees off on 14 to 17 % (v17).

16. **corrected**: Pace: with the model's own length a run needs about 1.3 times the decisions; the longest demonstration (209 cm in 222 rows, sheet 04) would take about 317 decisions, so the budget of 250 is too small.

   For the runs that held the factor is right: 1.27 (run 7: 112 against 88), 1.23 (run 4: 50 against 41), 0.95 (run 2), 1.06 (run 1). On the sheets with sharp bends the same calculation gives 1.86, 2.02 and 1.93 (runs 5, 9, 10) and 2.6 (run 6), partly because a lost robot asks short steps; expect about twice the decisions on sheet 03, not 1.3 times. Longest demonstration plan 04-30: 209 cm in 222 rows, 317 decisions at 0.66 cm: confirmed (v10). Measured pace 1.08 s per decision (run 5: 64.9 s for 60 pictures), so 400 decisions are about 7 minutes.

17. **corrected**: The sheets of the robot test did not lie as in the demonstrations: run 5 turned -9.5 to +7 degrees, run 9 -8.5 to +10, run 10 -18 to +0.5, run 6 +15.5 to +30 degrees; the demonstrations of one sheet and end form two placement groups about 14 degrees (sheet 03) and 12 degrees (sheet 04) apart.

   The substance holds, the sizes are a little smaller. My own estimate shares only the fitted camera model with theirs (own ink mask, one or five pictures per episode, own search; v12, v12b): run 5 -11 to +1.5 degrees (four demonstrations), run 9 -9 to +4, run 10 -16 to -5; run 6 +22 to +25 against plans 04-01, 04-02, 04-05 and +15 to +17.5 against 04-13, 04-16, 04-17. Groups: sheet 03 11 to 11.5 degrees apart, sheet 04 7.5 to 8.5. Their single values scatter by about 6 degrees (they turn 03-04 by 7 degrees more than 03-01, yet the two sheets lay within 0.5 degrees of each other by my estimate). So the sheet of run 6 lay about 15 to 25 degrees off every standard-start placement of end A, not 'as in plan 04-01'; 30 degrees is about 5 too high. Run 6 is therefore a weak test of anything on sheet 04.

18. **confirmed**: Correction to the brief: the puck is 2.4 cm across, not 4 cm.

   config.PUCK_PARAMETERS['diameter_m'] is 0.024, and in the first pictures of six runs the red puck is 2.4 to 2.7 line widths across (median 2.57; v19_puck_size.py), which fits 2.4 cm with its top 2.9 cm nearer the camera; a 4 cm puck would be more than 4 line widths.

19. **confirmed**: Corrections to the brief: run 10 did not fail at the first bend (lost at decision 29 at the top of the first curve); run 6 was not fine for 15 moves (it left the line at decision 10); the [scout] demonstration statistics hold.

   By eye on the contact sheets (v02_run_10_*.png, v02_run_06_*.png) and by number: run 10's puck went 4.43 cm along the first leg against 4.51 in the demonstrations (v05). Raw non-stop rows of frames.csv (18,312): median step 1.053 cm, mean 0.919, 20.4 % under 0.5 cm, 8.0 % under 0.2 cm, 0.649 cm at twists of 6 degrees or more; demonstrated turn per cm per sheet (median over episodes) 2.55, 4.64, 5.65, 3.44, 4.43 for sheets 02 to 06 (v10).

20. **corrected**: Risk named with the length rule: the stop comes up to about 0.7 cm earlier because steps near the end are short.

   It is at least 0.8 cm and it hits Emranul's first wish. Measured (v15_end_of_run.py): in all four runs that reached the end (1, 2, 4, 7) the model's asks fell to 0.51 to 0.60 cm and then 0.21 to 0.25 cm, and the driver still moved a full 1.0 cm AFTER the first picture with a stop chance of 0.86 to 0.93 (the rule waits for two pictures). Under the own-length rule that last move is 0.2 cm, so the tip stops about 0.8 cm earlier from the last move alone; over the last six moves the asks add up to 1.2 to 2.0 cm less than was driven (open loop, an upper bound). Run 2 already left a bit of the line's end visible with the extra centimetre.

### The analyst's findings with their evidence

1. (measured, confidence high) The driver did 1.0 cm on every decision whatever the model asked. Over the 545 moves of the ten runs the model asked for a median of 0.61 cm, under 0.5 cm on 34.5 % and over 1.0 cm on 14.3 % of the moves (largest 1.26 cm).

   Evidence: 01_asked_against_done.py / .out.txt (from the ten steps.csv): done median 0.999 cm, 5th to 95th percentile 0.973 to 1.024. Asked median, share under 0.3 cm, asked path, done path per run: run 1: 0.98 cm, 4 %, 23.5, 24.9 cm; run 2: 1.14, 4 %, 25.6, 24.2; run 3: 0.53, 0 %, 23.1, 37.1; run 4: 0.87, 2 %, 32.9, 40.5; run 5: 0.55, 12 %, 31.9, 59.3; run 6: 0.31, 50 %, 26.0, 67.8; run 7: 0.83, 9 %, 68.9, 87.7; run 8: 0.82, 2 %, 32.3, 40.0; run 9: 0.49, 15 %, 44.1, 89.2; run 10: 0.55, 16 %, 37.5, 72.4. The brief's [scout] numbers are confirmed. Figure 01_asked_against_done.png; per-step table 01_asked_against_done_steps.csv.

2. (measured, confidence high) Where the puck left the line, read from each run's own pictures (so independent of where the sheet lay): run 3 at decision 3, run 5 at 9, run 6 at 10, run 9 at 16 (off centre from 8), run 10 at 29, run 8 at 29. The held runs also drifted: run 1 was 1.2 to 2.0 cm outside the ink from decision 19 to the end, run 4 1.3 to 1.6 cm at decisions 24 to 28 and then back, run 7 up to 1.4 cm between decisions 46 and 58 and then back, run 2 never more than 1.1 cm.

   Evidence: line_and_puck.py builds, per run, a line map on the table (ink pixels of all pictures put onto the table with the forearm camera model) and the puck's position per picture (PuckTracker); 01_asked_against_done.py marks 'off the line' when the puck's centre is more than 1.2 cm outside the ink (the puck's radius, config.PUCK_PARAMETERS diameter 2.4 cm) on 2 pictures in a row, 'off centre' at 0.6 cm. Checked by eye on run 5 frames 4, 8, 12, run 6 frames 5, 9, 12, 16, 19, run 9 frames 10, 18, run 10 frames 12, 31. Run 7's mark moves between decision 48 and 55 with small changes of the map (the puck sat within 0.2 cm of the limit), run 8's is uncertain because the nearest ink of a dashed loop can belong to another dash. Against the demonstrations' TIP paths (laid onto the run by the printed line) the tip leaves their corridor only at decisions 25 (run 5) and 23 (run 9) and never in runs 10 and 6: the demonstrated tip paths themselves spread by 2.7 to 3.5 cm on sheet 03, so the tip path is a poor yardstick (02_paths_over_demonstrations.out.txt).

3. (measured, confidence high) Before the puck left the line the model asked for short steps only in runs 3, 5 and 9: median 0.41, 0.45 and 0.47 cm (62 to 100 % of those decisions under 0.5 cm) with twists of 4 to 6 degrees per decision, so the driver moved 2.0 to 2.9 times the asked distance. In runs 6, 8 and 10 the asks before the loss were 0.81, 0.87 and 0.53 cm (median).

   Evidence: 01_asked_against_done_steps.csv split at each run's off-the-line decision (computed in the session; the six decisions before the mark are in 01_asked_against_done.out.txt: run 3: 0.36 to 0.49 cm, run 5: 0.42 to 0.47, run 9: 0.34 to 0.58, run 6: 0.69 to 0.81, run 8: 0.65 to 0.88, run 10: 0.53 to 0.73). Run 6's low run median (0.31 cm) comes from the 58 decisions after the loss (median 0.23 cm).

4. (measured, confidence high) First bend of sheet 03: the eight demonstrations (end A, standard start) turned the forearm its first 60 degrees within 6.7 cm of tip travel (4.65 to 7.68) in 11 steps (5 to 13), of which 2.5 (0 to 3) were twists in place; their mean step there was 0.59 cm (0.46 to 1.01). The robot needed 16.3, 11.2 and 12.6 cm (runs 5, 9, 10). Over the first 15 decisions the forearm turned 3.8, 5.0 and 4.0 degrees per cm of tip travel against about 9.0 demonstrated (7.8 to 12.9).

   Evidence: 02_paths_over_demonstrations.py / .out.txt / 02_first_bend.csv (frames.csv tip_x_m, tip_y_m, joint_4 for plans 03-01 to 03-05 and 03-25 to 03-27; steps.csv for the runs); 04_length_rule.out.txt for the turn per cm. Figures 02_turn_against_travel.png and 04_length_rule_first_15.png. Twist in place = tip move under 0.3 cm with a twist of 3 degrees or more.

5. (measured, confidence high) The robot's tip overshot the first bend of sheet 03: it went 7.9, 6.7 and 5.3 cm past the puck's start place along the line's first leg (runs 5, 9, 10) where the demonstrations' tips went 4.4 cm (4.2 to 5.6): overshoot +3.5, +2.2 and +0.9 cm. In runs 5 and 9 the puck was pushed 0.8 to 1.4 cm beyond the bend (decisions 8 to 13), came back onto the ink (decision 15; 13 to 14) and was then pushed across the line to the other side by the returning tip (up to 3.7 and 2.8 cm off). In run 10 (team B) the puck stayed on the ink through the first bend (0.0 cm outside up to decision 15) and was lost at decision 29 at the top of the first curve.

   Evidence: 02_paths_over_demonstrations.out.txt, 02_top_view_run_05.png, _09.png, _10.png (grey = the line read from the run's pictures, green = demonstrated tip paths laid onto it, red = robot tip, blue = puck); puck distances in 01_asked_against_done_steps.csv. The demonstrations were laid onto each run by fitting their puck tracks onto the run's line map: after the fit a median 79 to 88 % of the demonstrated puck-track points lie within 5 mm of the run's ink (02_registration.csv).

6. (calculated, confidence medium) What the model asked for at that bend was right: the same logged answers carried out with the asked length turn the forearm 60 degrees within 7.4, 5.2 and 5.1 cm (demonstrations 6.7 cm) and give 8.3, 10.4 and 9.0 degrees per cm over the first 15 decisions. The fixed step halved the turn per centimetre; that is the main cause of the loss in runs 5 and 9, and by the same pattern in run 3 (first three decisions asked 0.36 to 0.49 cm with twists of -4.2 to -5.3 degrees; puck 1.3 cm off at decision 3, 2.8 cm at decision 5).

   Evidence: 02_first_bend.csv rows 'as asked'; 04_length_rule.py replays the logged answers on paper (04_length_rule.out.txt, 04_length_rule_first_15.csv/.png). Caveats: open loop (every picture after the first would differ), sheet 03 was a training sheet for models E and B, and inside the heading window of 20 to 60 degrees the asked travel (1.7 to 2.8 cm) is shorter than the demonstrated (3.7 cm, 2.8 to 4.9), so the rule may turn a little tighter than demonstrated there. Run 9 repeated run 5 without the hoodie, so the hoodie is not the cause. A robot test is needed to prove the causal step.

7. (measured, confidence high) Sheet 04 (run 6) did not fail because of the step length. Before the loss the model asked for 0.69 to 0.81 cm; the forearm turned only 13.6 degrees in total where 8 of 11 demonstrations turned 60 degrees within 9.1 to 11.5 cm (the other three never turned 60 degrees there: two demonstrated ways through the S-bend); neither the done nor the as-asked path reaches 60 degrees. The puck ran straight on (2.6 cm off at decisions 11 to 12), met the line again at the hairpin at decision 16 and sat there. From decision 19 to 50 the model asked for a median of 0.16 cm (largest 0.42) and 0.85 degrees, its direction flipping by more than 90 degrees on 20 of 31 changes; the driver turned that into 32.0 cm of back-and-forth for a net 1.7 cm.

   Evidence: 02_paths_over_demonstrations.out.txt (run 6), 02_top_view_run_06.png, 02_turn_against_travel.png (right column), 04_length_rule.out.txt (hairpin table), 01_asked_against_done_steps.csv. Confound: the sheet of run 6 lay turned 15.5 to 30 degrees and shifted 2 to 4 cm against the demonstrations' placements (02_registration.csv).

8. (measured, confidence high) Offline, the predicted length does NOT separate demonstrated twists in place from real moves on held-out sheets: area under the curve 0.63 for both recipes (0.51 to 0.74 per sheet). Medians 0.59 against 0.69 cm (E recipe), 0.55 against 0.65 (A recipe). A threshold of 0.3 cm misses 93 % (E) or 90 % (A) of the twists in place and would freeze 4.3 % or 6.6 % of the real moves; the equal-error threshold (0.64 or 0.59 cm) is wrong on 41 % of both. Of the rows under any threshold only 13 to 14 % are twists in place (9.5 % of all rows are).

   Evidence: fold_model_outputs.py re-ran the six fold models of 2026-11-10_resnet18_chunk_stop_weight_25 (E recipe) and 2026-11-09_resnet18_final_candidate (A recipe) on their held-out sheets (18,032 pictures each; twist and stop chance reproduce the saved predictions to 0.0000). 03_predicted_length_offline.py / .out.txt section 1, figure 03_predicted_length_offline.png. 1,688 twists in place against 16,008 real moves (tip move of 0.3 cm or more). Twists in place are 13.9 % of the rows of sheet 04 and 10.5 % of sheet 03, 3.4 to 6.3 % elsewhere.

9. (measured, confidence high) The median gain from predicted to demonstrated length for moving rows is 1.52 (E recipe) and 1.56 (A recipe); as a ratio of medians 1.59 and 1.68. Per held-out sheet: 1.13 to 1.15 on sheet 01, 1.48 to 1.68 on sheets 02 to 06.

   Evidence: 03_predicted_length_offline.out.txt section 2 (predicted median 0.69 and 0.65 cm against 1.09 cm demonstrated, 16,008 rows).

10. (measured, confidence high) A short predicted step means an unsure direction, not a short demonstrated step. Where the model predicts under 0.3 cm the demonstrations still moved a median of 0.83 to 1.0 cm; the correlation of predicted with demonstrated length is 0.22. The predicted direction is within 45 degrees of the demonstrated one on 33 to 49 % of rows when the predicted length is under 0.3 cm and on 84 to 98 % at 0.9 cm or more (median error 46 to 74 against 9 to 17 degrees).

   Evidence: 03_predicted_length_offline.out.txt section 3, 03_predicted_length_bins_E_recipe.csv and _A_recipe.csv, right panels of 03_predicted_length_offline.png. The exact step is trained with a Huber loss on forward and left in cm (training_probes.head_losses), so the output is close to the mean step vector, which is short when directions scatter.

11. (measured, confidence high) Length and twist shrink together (0.71 and 0.64 of the demonstrated size on unseen sheets), so only the pair keeps the forearm's turn per centimetre. Predicted twist per cm of travel, as a share of the demonstrated: fixed 1 cm 0.60 (E) and 0.62 (A); the model's own length 0.89 and 0.95; gain 1.5 capped at 1.0 cm 0.69 and 0.74; the backup of report 4.4 (gain 1.6, clamp 0.5 to 1.5 cm) 0.56 and 0.60, no better than the fixed step. One step ahead, the own length lands 0.59 cm from the demonstrated next tip position on bend rows against 0.78 cm for the fixed step (0.52 to 0.55 against 0.95 on twists in place; 0.75 against 0.71 on straight rows, an undershoot along the path).

   Evidence: 03_predicted_length_offline.out.txt sections 4 and 5 (17,773 held-out rows per recipe; bend rows = demonstrated twist of 5 degrees or more, 7,985 rows). On the robot the held runs agree: asked twist per asked cm 2.40 (run 4) and 3.50 (run 7) degrees per cm against 2.55 and 3.44 demonstrated per episode, done 2.01 and 2.78 (01_asked_against_done.out.txt).

12. (calculated, confidence medium) On an unseen sheet the picture is the same but less complete: the fold models that never saw sheet 03, shown the demonstrations' pictures of its first bend, ask for a median of 0.50 to 0.53 cm per step; with their own length the turn per cm is 0.69 (E recipe) and 1.04 (A recipe) of the demonstrated 8.7 degrees per cm, with a fixed 1 cm 0.36 and 0.53. So the length rule removes most of the deficit, and recipe E keeps a twist deficit of about 30 % on an unseen bend.

   Evidence: 03_predicted_length_offline.out.txt section 7 (8 demonstrations, rows until the forearm has turned 60 degrees; teacher-forced on demonstration pictures, not a closed loop).

13. (calculated, confidence medium) The proposed rule on the first 15 decisions of runs 5, 9 and 10 (same logged answers, open loop): steps of 0.18 to 0.73, 0.34 to 0.71 and 0.13 to 0.73 cm instead of 1.0; tip travel 6.8, 7.1 and 6.4 cm (net 4.4, 3.9 and 3.4 cm from the start) instead of 15.0, 15.1 and 14.8 cm (net 9.5, 8.6, 8.2); the same forearm turn (-57, -74, -58 degrees); no stall nudge. On all ten logs the stall guard acts only in run 6 (nudges at decisions 23, 29, 35, run ended as stalled at 41 instead of wandering to a wrong stop at 68), once in run 9 (decision 87) and ends run 10 at decision 61 as 'stalled near the end' (stop chance 0.26 to 0.44 on five pictures with asks of 0.07 to 0.39 cm).

   Evidence: 04_length_rule.py / .out.txt / 04_length_rule_first_15.csv / .png. Later pictures would differ under the rule, so this shows what the rule does with the same answers, not how the runs would have gone.

14. (measured, confidence high) Numbers behind the stall guard and the clamp. Demonstrations almost never stand still: over 6 steps the net tip move is under 1.0 cm AND the net forearm turn under 10 degrees in 0.24 % of 16,519 windows (1st percentile of net move 1.00 cm, median 4.43 cm); the fold models' own steps on the same pictures in 0.35 % (E) and 0.57 % (A). When the predicted step is under 0.3 cm, the model's current direction is within 45 degrees of the demonstrated path over the next 5 rows on 50 % of rows, the last sure direction (asked 0.7 cm or more) on 29 to 34 %: a remembered direction is worse, so the driver's rule 'under 0.05 cm keep the last direction' (used 3 times, all in run 6) has no support. 74.5 % of demonstrated steps are 1.2 cm or shorter (real moves: median 1.09, 75th percentile 1.23 cm); the model asked for more than 1.2 cm on 1.1 % of the robot moves and 0.7 % of held-out pictures.

   Evidence: 03_predicted_length_offline.out.txt section 6; 04_length_rule.out.txt sections 'numbers behind the clamp and the budget' and 'which direction when the predicted step is short'.

15. (calculated, confidence medium) Pace: with the model's own length a run needs about 1.3 times the decisions on the sheets that held (run 7: about 112 instead of 88 for the same path, run 4: 50 instead of 41, run 2: 23 instead of 24). The longest demonstration (209 cm of tip path in 222 rows, sheet 04) would take about 317 decisions at the 0.65 to 0.67 cm the fold models predict on unseen sheets, so the budget of 250 is too small.

   Evidence: 04_length_rule.out.txt section PACE (mean rule step 0.78 to 1.07 cm on the held runs); frames.csv path lengths. At about 1.0 to 1.2 s per decision (report 4.6) 320 decisions are about 6 minutes.

16. (calculated, confidence medium) The sheets of the robot test did not lie as in the demonstrations, and the demonstrations of one sheet and end used two placements. Against the demonstrations the sheet of run 5 lay turned -9.5 to +7 degrees, run 9 -8.5 to +10, run 10 -18 to +0.5, run 6 +15.5 to +30 degrees, with shifts of 0.7 to 3.5 cm (medians). Demonstrations 03-01 to 03-05 and 03-25 to 03-27 differ by about 14 degrees, 04-01 to 04-05 and 04-13 to 04-18 by about 12 degrees.

   Evidence: 02_registration.csv (turn and shift that lay each demonstration's puck track onto the run's line map; checked by drawing the laid-on puck tracks over the maps: they lie on the line for all four runs).

### The analyst's recommendations

1. Change the driver's length rule in model_mode.model_step: tip move = length_gain x hypot(model forward, model left), length_gain 1.0, at most 1.2 cm, no smallest step; a predicted length under 0.05 cm gives a twist-only step (tip stays, joints 1 to 3 re-solved); twist as today (twist_gain x model twist, at most 10 degrees). Keep length_gain and twist_gain equal. Remove 'under 0.05 cm keep the last direction'. New keys in MODEL_MODE_PARAMETERS (length_gain, step_most_cm), new columns in steps.csv (step_cm, kind), tests in test_model_mode.py; the executor check must then give back the recorded step LENGTH as well.

   Why: At the first bend of sheet 03 the model's own length gives 8.3 to 10.4 degrees of forearm turn per cm against 3.8 to 5.0 done and about 9.0 demonstrated; offline the own length keeps 0.89 to 0.95 of the demonstrated turn per cm against 0.60 to 0.62 for the fixed step, and lands 0.59 instead of 0.78 cm from the demonstrated next tip position on bend rows. It is also the plain form of imitation: the robot does what the model answers. 1.2 cm: 75th percentile of demonstrated real moves is 1.23 cm and the model asks for more on about 1 % of decisions (a cap of 1.0 cm would keep straight parts exactly as on 11.11).

   Needs: code only. Effort: about half a day with tests (simulated run, executor check); no robot

   Expected effect: At sharp bends the forearm turns per cm about as demonstrated instead of half as fast; the tip no longer swings 2 to 3.5 cm past the demonstrated turning point. Whether the puck then stays on the line is not known until the robot test. Runs take about 1.3 times as many decisions.

   Risk: Untested in closed loop. Slower runs. Where the model is unsure it creeps (0.1 to 0.3 cm per decision). The stop comes up to about 0.7 cm earlier because steps near the end are short (the stop analysis must take this into account). With model E a twist deficit of about 30 % remains on an unseen bend.

2. Add a stall guard and raise the step budget to 400. Guard: after each decision, if over the last 6 decisions the tip's net move is under 1.0 cm AND joint 4's net turn under 10 degrees, the run is stalled. If the stop chance reached 0.25 on a picture of that window, end the run ('stalled near the end'). Otherwise one nudge of 0.5 cm along the sum of the last 6 predicted steps on the table (the picture's up direction if that sum is under 0.05 cm), twist as predicted, and the window starts again. The fourth stalled window in a row ends the run ('stalled'); a window that is not stalled resets the count.

   Why: With the own length a run of tiny steps could hang (run 6: 32 decisions with a median ask of 0.16 cm and 0.85 degrees). Demonstrations are below both limits in 0.24 % of their 6-step windows, the models' own steps on demonstration pictures in 0.35 to 0.57 %. The nudge follows the model's current answers because a remembered direction is worse (within 45 degrees on 29 to 34 % against 50 %). 0.5 cm is about the demonstrated mean step at the sheet-03 bend (0.59 cm); 0.25 and the three nudges are judgement. Worst case 27 decisions (about 30 s) before the run ends. Budget: the longest demonstration would need about 317 decisions.

   Needs: code only. Effort: 2 to 3 hours with tests (a small class next to StopRule, called in ModelRun.run)

   Expected effect: No run can hang or dither for long: on the log of run 6 the guard nudges at decisions 23, 29, 35 and ends the run at 41. It does not make the model right where it has no opinion.

   Risk: A nudge can push the puck up to 1.5 cm in a poorly known direction (50 % within 45 degrees). 'Stalled near the end' can end a run a little early when the stop chance hovers between 0.25 and 0.5 (run 10, decisions 56 to 60). The limits come from demonstrations and logs, not from a closed-loop test.

3. One robot session that compares driver settings with the same model (E), the same sheet placement and alternating order: (A) fixed 1 cm, twist gain 1 (as on 11.11); (B) own length, twist gain 1 (proposed); (C) fixed 1 cm with twist gain 1.6; (D) own length and twist both x 1.4. Sheets: 03 from end A, 05, and one printed practice track the model has never seen; 2 runs per setting and sheet (24 runs). Score every run from its pictures with line_and_puck.py and 01_asked_against_done.py: first off-centre and off-line decision, largest distance outside the ink, forearm turn per cm in the first 15 decisions, where it stopped. Fix the decision rule before the session: B is kept if the puck stays within 1.2 cm of the ink through the first bend of sheet 03 in both runs and sheet 05 does not get worse.

   Why: Everything about the new rule is calculated open loop; only the robot shows whether the puck stays on the line. Sheet 03 was a training sheet and flatters the model (the unseen-sheet estimate for model E is 0.69 of the demonstrated turn per cm), so an unseen track must be in the test. C is the cheapest alternative (no code: --twist-gain) and D restores the demonstrated pace (length shrinks to 0.71, twist to 0.64 of the demonstrated size on unseen sheets).

   Needs: a robot test. Effort: one session of 2 to 3 hours after the code change, with a written checklist and Emranul's go; about 1 hour of analysis with the existing scripts

   Expected effect: A measured answer on the first bend of sheet 03 (on 11.11: lost in 2 of 3 runs) and a first number for an unseen track; shows whether the pace factor 1.4 is safe.

   Risk: Two runs per setting cannot settle small differences; hand-laid sheets differ by centimetres and up to 30 degrees (measured), so lay the sheet against tape marks and keep it there for all settings of one sheet. C doubles the twist also where its sign is wrong (sign right on 68 to 84 % offline).

4. Drop the backup of report section 4.4 (predicted length rescaled by about 1.6 and clamped to 0.5 to 1.5 cm) and do not build a 'twist in place when the predicted length is under a threshold' rule.

   Why: Offline the backup gives 0.56 to 0.60 of the demonstrated turn per cm (the fixed step 0.60 to 0.62) and the same landing error on bend rows (0.76 to 0.79 against 0.78 cm). The predicted length cannot tell twists in place from moves (area under the curve 0.63; at 0.3 cm it misses 90 to 93 % of them).

   Needs: code only. Effort: none (a decision; one paragraph in the report)

   Expected effect: No time spent on two rules that the data do not support.

   Risk: None found.

5. Make the picture-based score a standard step after every robot run: line_and_puck.py (line map and puck track on the table) plus the table of 01_asked_against_done.py, so that 'puck in the middle' and 'left the line at decision N' are numbers instead of impressions.

   Why: It placed the loss of each run of 11.11 to the decision, corrected two impressions (run 6 lost the line at decision 10, not after 15 fine moves; run 10 passed the first bend), and showed drift in the held runs (1.2 to 2.0 cm in run 1, up to 1.4 to 1.6 cm in runs 4 and 7), which is what Emranul wants to remove.

   Needs: code only. Effort: 1 to 2 hours to wrap the two scripts into one command per run folder

   Expected effect: Every later robot test can be compared on the same measure within a minute.

   Risk: Valid only while the camera stays as taped (fitted camera model) and under similar light (ink and paper colour limits); dashed lines give uncertain distances (the nearest ink can belong to another dash).

6. Only if the robot test shows the own length too slow or stalling often: retrain with 'how far' separated from 'which way' (direction as a unit vector, the length as its own output trained on the demonstrated length, or a twist-in-place class). First a probe on frozen features on the laptop: can any head tell twist-in-place rows from moves on held-out sheets better than 0.63?

   Why: Today's length output is the length of a mean vector, so it mixes step length with direction doubt (correlation with the demonstrated length 0.22). A separate output could give the demonstrated pace without scaling the twist.

   Needs: retraining. Effort: probe: about an hour on the laptop; a full recipe: 1 to 2 days of code plus one run per recipe on ki-014 with six held-out sheets and two seeds

   Expected effect: Unknown; the probe decides whether it is worth a GPU run. No new demonstrations are needed for it.

   Risk: A single picture may not show whether the demonstrator twisted in place at that waypoint (the twists in place come from the recording rule '1 cm or 5 degrees on any joint'), in which case the new output is no better.

7. Ask the professor on 23.11: (1) may the driver scale the model's step by a constant and nudge after a stall, or must it carry out the model's output as it is (today's fixed 1 cm is itself such a rule; the proposal moves closer to the plain output); (2) is there a time limit per run at the exam.

   Why: The proposed rule makes runs about 1.3 times longer (an unseen track as long as sheet 04 could take about 320 decisions, about 6 minutes), and the stall guard is a hand-written rule.

   Needs: the professor's answer. Effort: two questions in the meeting

   Expected effect: Decides between gain 1.0 (slow, plain) and the common pace factor 1.4, and whether the nudge stays.

   Risk: If no hand-written guard is allowed, a stalled run must simply end.

### The verifier's challenges to the recommendations

1. Change the length rule in model_mode.model_step (own length, gain 1.0, at most 1.2 cm, new keys, about half a day): the rule already exists in the working tree. 02_code/model_mode.py (saved 18:18 on 11.11, before the analyst's answer) has --step-length predicted, --length-gain, --twist-gain and --step-budget, config keys step_length_rule, step_length_gain and step_most_m (0.01), and test_model_mode.py tests it, including a twist in place where the tip stays. New key names would duplicate these. What is left is one number (cap 0.010 to 0.012 m, of little weight: asks over 1.0 cm were 14 % of moves, over 1.2 cm 1 %), the executor check for length, and the decision about the guard. Settings B and D of the proposed session need no new code.

2. The same recommendation, risk on the robot: it makes the robot stop about 0.8 cm earlier (see the corrected verdict), against Emranul's wish that the puck covers the line's end completely. Do not take it to the robot without deciding what the driver does after the first stop picture, and score the stop place in every run. Also: in the open-loop replay the as-asked tip is 1.0 to 1.5 cm short of the demonstrated place in runs 9 and 10 at the same forearm turn, so gain 1.0 may cut a bend on the inside; and steps of 1 to 3 mm were never commanded by this driver on the real arm (11.11: always 1 cm), so a dry run without the puck belongs first on the checklist.

3. The reason given for 'own length' over other fixes is weaker than stated. On unseen sheets the fold models hardly shorten at bends (0.63 against 0.70 cm for E; the demonstrations 0.67 against 1.17), so offline the own length is nearly the same as a plain fixed step of 0.67 cm, and a fixed 0.5 cm would keep MORE of the turn in sharp stretches (0.83 to 0.85 against 0.63 to 0.68, at the cost of straight parts and pace). On the training sheets used on 11.11 the all-sheets models did shorten at bends (asks of 0.46 to 0.51 cm at twists of 4 degrees or more against 0.60 to 0.89 cm elsewhere in runs 3, 5, 7), so a win of B on sheet 03 would flatter the rule for the exam's unseen line. The robot comparison needs a plain shorter fixed step (step_m 0.006 to 0.007, a config number) as a control.

4. 'Keep length_gain and twist_gain equal' should not be fixed as a rule before robot data. Model E asks for only 0.52 of the demonstrated turn at an unseen sharp bend while its length there is about 0.75 of the demonstrated travel, and on the robot the forearm turned only 62, 76 and 62 degrees at the first bend of sheet 03 where the demonstrations turn 96 (90 to 110); the asked twists add up to no more (61, 74, 61; v21). That points to a twist gain above the length gain for E, which is setting C or a variant of D; the length rule alone does not add turn.

5. Stall guard with nudges: (a) the driver already has a stall rule (30 pictures within 3 cm and under 15 degrees ends the run); it would not have fired on any run of 11.11 as driven (v10). (b) The nudge is a hand-written move with no evidence of benefit: on run 6's log three nudges change nothing and the run still ends as stalled at 41; 14 to 17 % of nudges would point more than 90 degrees away from the demonstrated path; it needs the professor's yes on 23.11. Until then end the run instead of nudging. (c) The exit 'stalled near the end' at a stop chance of 0.25 is a second, weaker stop rule: teacher-forced it would have ended 1.9 % (E) and 2.4 % (A) of the demonstrations before their last tenth, and on sheet 05, the one with the crossing, 2 and 4 of 29 (7 and 14 %; v16). Drop that exit and let only the normal stop rule end a run as stopped. (d) The guard would fire at least once in about 1 healthy run in 9 (11.5 % of demonstrations). The budget of 400 is fine (about 7 minutes).

6. One robot session with four settings, three sheets, two runs each (24 runs): too thin to rank four settings, and the pass rule can be met by luck: under the old driver the first bend of sheet 03 was passed 1 time in 3 (run 10), so 'both runs pass' has a chance of about 1 in 9 for a setting that is no better. Cheaper and sharper first: only the first bend of sheet 03, sheet against tape marks, alternating A (fixed 1 cm), B (own length) and a shorter fixed step, 4 short runs each ended with Esc after the bend (about 30 to 40 decisions, 12 runs in about 20 minutes); then full runs for the best one or two settings on sheet 05 and the unseen track. The decision rule must include the stop place and the unseen track, because sheet 03 is a training sheet. Setting C (twist times 1.6) hits the 10-degree clamp on 3 to 4 % of held-out rows and enlarges wrong-signed twists (sign right on 78 %). Lay every sheet as in a set-up picture: run 6 lay 15 to 25 degrees off. All of it only with Emranul's go and a written checklist.

7. Retraining 'only if the robot test shows the own length too slow or stalling': the trigger is too narrow. The offline numbers already show that on unseen sharp bends the length output barely tells a bend from a straight part and that E's twist is about half the demonstrated, whatever the driver does. The one-hour probe on frozen features (can any head tell twist-in-place rows from moves better than 0.63, can a separate length output track the demonstrated length better than a correlation of 0.22) costs no robot time and should run before or beside the robot session, so that a retraining run on ki-014 is not waiting for it later.

8. Picture-based score as a standard step: reasonable, and its marks agreed with my eyes on seven runs, but its centimetre values rest on the fitted camera model (valid only for this taping of the camera) and I did not check them independently; on the dashed sheet it stays uncertain. Keep the pictures' contact sheet next to the number for every run.

### What the verifier says was missed

1. That the driver in the working tree already carries the predicted-length rule, both gains, a step budget switch, a stall rule and tests (02_code/model_mode.py, config.py MODEL_MODE_PARAMETERS, test_model_mode.py), and that a simulated run with it exists (03_data/model_runs/2026-11-11_181904_simulated).

2. The stop: what the length rule does at the line's end was named as a risk but not measured. In all four runs that reached the end the driver pushed a full 1.0 cm after the first stop picture where the model asked 0.21 to 0.25 cm (v15).

3. Net turn instead of summed absolute twists: the share of the demonstrated turn per cm that the own length keeps on unseen sheets is 0.74 to 0.79 overall and 0.63 to 0.68 in sharp stretches, not 0.89 to 0.95 (v08).

4. A control with a plain shorter fixed step: on unseen sheets the own length and a fixed 0.67 cm give almost the same offline numbers, because the predicted length changes little between bends (0.63 cm) and straight parts (0.70 cm) (v20). Without this control a robot win of 'own length' cannot be told from a win of 'shorter steps'.

5. How FAR the forearm turned, not only how fast: at the first bend of sheet 03 the model asked for 61, 74 and 61 degrees in total where the demonstrations turn 96 (90 to 110); the picture's heading ended 12 to 25 degrees short of the demonstrated one (v21). The length rule does not change the asked total.

6. Run 10 as a test of the causal chain: same driver, shorter asks, slower turn per cm than the failed run 9, and the puck stayed on the line through the bend. The difference lay in the asked directions and 1.2 to 2.4 cm less tip overshoot, so the direction answers count as much as the length (v05, v09, v13).

7. The guard's alarm rate per run and the early-end risk of its stop-chance exit on the sheet with the crossing (v16); the analyst gives shares of windows only.

8. Whether a stale picture (one move late) could explain the overshoot instead of the step length. I checked: the pictures were fresh in runs 4, 5, 7, 9 and 10 and no picture is repeated (v18).

9. Whether the team's averaging shortens the step further (it would matter for a driver that carries out the length). I checked on run 10: the team's length is a median 0.98 of its members' mean length, under 0.7 on 7 % of pictures; the members themselves ask short where they are unsure (v11). Not a problem.

10. Run 3: the claim 'by the same pattern' was not tested against the simpler reading that the sheet lay unlike any plan; no demonstration of sheet 02 was compared, and the model kept steering right long after the puck was gone.

11. For sheet 03 the as-asked tip goes only 1.6 to 2.0 cm past the puck's start against 4.4 demonstrated (their own table): they call this 'a little tighter', but it is the main open-loop sign that gain 1.0 may undershoot a bend, and it should be one of the things the first robot runs look at.

### The analyst's corrections to the brief

1. The puck is 2.4 cm across, not 4 cm: config.PUCK_PARAMETERS['diameter_m'] is 0.024 and 12 pictures measured through the camera model give 2.2 to 2.3 cm. Every 'off the line' limit depends on it (puck radius 1.2 cm, line half width 0.5 cm).

2. Run 10 (team B) did not fail 'the same way at the first bend': its tip overshot least (+0.9 cm against +3.5 and +2.2 cm in runs 5 and 9), the puck stayed on the ink through the first bend (0.0 cm outside up to decision 15) and was lost at decision 29 at the top of the first curve.

3. Run 6 was not 'fine for the first 15 moves': the puck left the line at decision 10 (2.6 cm outside the ink at decisions 11 to 12) by running straight on at the first S-bend, and met the line again at the hairpin at decision 16. Its sheet also lay turned 15.5 to 30 degrees and shifted 2 to 4 cm against the demonstrations' placements, not 'as in plan 04-01'.

4. The [scout] sentence 'the runs that failed at sharp bends are the ones where the model asked for short steps' holds before the loss only for runs 3, 5 and 9 (asked median 0.41 to 0.47 cm). In runs 6, 8 and 10 the asks before the loss were 0.81, 0.87 and 0.53 cm; run 6's median of 0.31 cm comes from the decisions after the loss. Short asks are partly a result of being lost.

5. The backup quoted from report 4.4 (length rescaled to the demonstrated median, clamped 0.5 to 1.5 cm) would not have helped: offline it turns the forearm 0.56 to 0.60 of the demonstrated amount per cm, the same as the fixed step.

6. A short predicted step is not a predicted twist in place. On held-out sheets the demonstrations moved a median of 0.83 to 1.0 cm where the model predicts under 0.3 cm; the short length mostly shows an unsure direction.

7. 'Aligned at the start' needs care: every standard-start demonstration begins with the tip on the taught start mark (14.0, -1.5 cm), 2.9 to 14 cm away from the puck (eight demonstrations checked), and first drives to the puck; in the robot runs the tip was placed 2.5 to 3.6 cm behind the puck's centre. The puck itself lies 2 to 3 cm after the place where the ink begins. The comparison here is therefore aligned by the printed line read from the pictures.

8. The demonstrations of one sheet and end are not one placement: sheet 03 end A has two groups about 14 degrees apart, sheet 04 end A two groups about 12 degrees apart.

9. Smaller points: the twist per cm done on sheet 03 is 2.9 to 3.3 degrees per cm with run 10 (brief: 3.1 to 3.2); the other [scout] numbers (asked medians and paths, demonstrated step statistics on the raw non-stop rows: median 1.05, mean 0.92, 20.4 % under 0.5 cm, 8.0 % under 0.2 cm, 0.65 cm for twists of 6 degrees or more; demonstrated twist per cm per sheet) are confirmed.

### The analyst's open questions

1. Does the model's own length keep the puck on the line at the first bend of sheet 03 in closed loop? The replay is open loop; only a robot test answers.

2. Is gain 1.0 the right pace? The gain that matches the demonstrated turn per cm lies between 0.7 and 1.3 depending on the measure (held-out sheets 0.89 to 0.95, the unseen-sheet bend 0.69 for E and 1.04 for A, the robot's sheet-03 bend 0.9 to 1.2); a common factor on length and twist (1.4) would restore the demonstrated pace but is untested.

3. How does the stop rule behave with short steps near the end, and is 0.25 the right stop chance for 'stalled near the end'? The stop place moves up to about 0.7 cm earlier; this must be settled together with the stop analysis.

4. Model E turns the forearm about 30 % too little per cm on an unseen sharp bend even with its own length (the twist shrinks more than the length there). Does that need a twist gain, the A recipe or the team of five, or retraining? This belongs to the twist analysis.

5. Why did run 6 turn the forearm only 14 degrees at the first S-bend of sheet 04: the two demonstrated ways through that bend (8 of 11 demonstrations turn 60 degrees or more, 3 do not), the sheet turned 15 to 30 degrees against every demonstration, or both?

6. Is a hand-written stall guard with a nudge allowed under 'imitation learning only', and is there a time limit per run at the exam?

7. Can any output learn from one picture that the demonstrator twisted in place at that waypoint (today 0.63), or is that an artefact of the recording rule that no model can see?

8. The line-and-puck measure is uncertain on the dashed sheet (run 8: off the line at decision 29 by the measure, 'outside the loop by step 15' by eye).

## Twist and heading (folder `twist_and_heading`)

**Analyst's answer.** Under-turning is real at sharp bends and comes before the failure there, but on 11.11.26 its main cause was the driver's fixed 1-cm step, not the size of the model's twist: at the first bend of sheet 03 the forearm reached the demonstrated angle after the same number of decisions (12 to 13 against 12 pictures) while the tip had travelled 12 to 13 cm instead of 7.8 cm. First fix: let the tip travel follow the model (asked length or a "unit step") and test it against a twist gain of 1.5; on unseen sheets the model's own twist is also weak (about half of the demonstrated turn), which needs more varied tracks and a target that separates re-aiming from pushing, not a hand-written rule.

**Verifier's overall verdict.** Partly. The numbers are right almost everywhere: I recomputed all 14 findings with my own code (scripts v00 to v22 with their printed output and four picture sheets in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/twist_and_heading/verify/). The core diagnosis for sheet 03 survived my attempts to refute it: the driver carried out the twist it was given, and at the same straight-line distance from the start (3 to 7 cm) the forearm was 31 to 64 degrees behind the demonstrations' median and behind every one of the eight demonstrations, because one decision moved the tip 1.0 cm where the model had asked for about 0.45 cm. But "not the size of the model's twist" is too strong: the twist per decision was itself only 0.54 to 0.93 of the demonstrated twist per picture (the two factors multiply, and the starts were pre-twisted 5 to 10 degrees towards the bend); the sheet-04 failure is under-turning by the model alone (0.29 of the demonstrated twist), which no step rule cures; and the place comparisons are solid for run 9 only. Two facts change the plan: the working copy of the driver already holds the asked-length rule, a length gain and a stall guard (model_mode.py and config.py saved 18:18 on 11.11.26), and a model that never saw sheet 03 gives almost the same answers on the robot's own pictures of that bend, so the next session can test the exam case with a fold checkpoint instead of only model E on a sheet it was trained on.

### Verdicts on the claims

1. **confirmed**: Finding 1: twist per cm recomputed; by the sum of |joint-4 change| per cm the robot was below the demonstrations on every sheet (0.24 to 0.96); net turn per 5 cm tells a different story

   Measured with my own code (v01, v21; frames.csv with the audit cleaning, 18,032 rows, 209 episodes; steps.csv). Demonstrations, median over episodes, degrees per cm: 1.57, 2.54, 4.64, 5.65, 3.39, 4.42 for sheets 01 to 06 (from joint 4 and tip positions instead of the labels: 1.53, 2.49, 4.69, 5.73, 3.43, 4.49). Robot runs 1 to 10: 1.05, 0.38, 2.45, 2.01, 3.16, 1.38, 2.78, 2.60, 3.28, 2.93; ratios 0.67, 0.24, 0.96, 0.79, 0.68, 0.24, 0.82, 0.59, 0.71, 0.63. Net per 5 cm, robot over demonstrations: 1.27 (run 4), 1.03 (run 7), 1.00 and 1.11 (runs 5, 9), 0.72 (run 10), 0.39 (run 8), 0.19 (run 6). Share of the demonstrated twisting still there at the end of a 5-cm stretch: 0.67 (sheet 01) to 0.90 (sheet 03). One thing to add: the demonstrated path length contains the hand's back-and-forth (a path smoothed over plus or minus 2 rows is 0.79 of the labelled path, median), so per cm of the smooth path a robot drives the demonstrated twist is about a quarter higher than these figures and the robot's shortfall is larger, not smaller.

2. **confirmed**: Finding 2: the driver carried out the twist it was given; no clamp ever bound

   Measured (v00, v01): 0 clamped steps and 0 refusals in all ten runs; largest asked twist 7.5 degrees (run 10); joint 4 between -80.9 (run 9) and +101.5 (run 8); used twist equals asked twist to 0.001 degrees; measured joint-4 change against the used twist on carried-out steps: median 0.03 to 0.10, largest 0.13 to 0.53 degrees. The tip step measured 0.94 to 1.05 cm for the 1.0 cm commanded. Joint 4 at the first picture: -9.7 to +9.8 degrees.

3. **corrected**: Finding 3 and the headline: at the first bend of sheet 03 the forearm reached -70 degrees after the same number of decisions as demonstrated pictures (12 to 13 against 12) but after 12 to 13 cm instead of 7.8 cm; the camera fell 42 to 47 degrees behind at the same place before the tip left the path; the reason was tip travel per decision, not twist per decision

   The raw numbers are right (v02, measured): demonstrations (8 standard starts from end A) reach -70 after a median of 12 pictures (6 to 13) and 7.8 cm of labelled path (5.3 to 10.2); robot: picture 12 after 12.1 cm (run 9), 13 after 12.8 cm (run 10), 17 after 17.0 cm (run 5); asked lengths up to there 5.5, 5.2, 7.7 cm. Three corrections. (1) The robot started with joint 4 at -5.2, -9.7 and -9.3 degrees (demonstrations 0.0), so it needed only 65, 61 and 61 degrees of turn against 71 to 86: twist per decision 5.4, 4.7 and 3.6 degrees against 6.6 per demonstrated picture, that is 0.82, 0.71, 0.54 (over the first 13 decisions 0.89, 0.77, 0.68 of my 6.06 per picture; the analyst's 0.93, 0.80, 0.71 use 5.7). The travel factor is 1.44 demonstrated pictures per cm against 1.0, that is 0.69. So the step is the larger factor for run 9, the two are about equal for run 10, and for run 5 the twist size weighs at least as much. "Same number of decisions" holds for 2 of 3 runs and partly thanks to the pre-twisted start. (2) The place comparison is solid for run 9 only: start tips were 3.1 (run 5), 1.3 (run 9) and 3.2 cm (run 10) from the demonstrated start tip (14.0, -1.5) cm, with start headings +11.7, +3.2, +12.3 against -6.1 degrees (v06); the analyst's "within 0.2 to 2.1 cm" is the distance to the nearest demonstrated tip position anywhere on the path, and at a bend where the demonstrator turns nearly in place a 1.5-cm disc holds headings from 0 to -70. (3) Measured without the table (v13): when the tip is 3, 4, 5, 6, 7 cm in a straight line from its start the demonstrations have turned -45, -55, -71, -86, -87 degrees (median), the robot -5/-14/-11, -11/-20/-19, -17/-26/-26, -23/-32/-33, -28/-37/-38 (runs 5/9/10): 31 to 64 degrees behind the median and behind every one of the eight demonstrations. So the size of the lag (the analyst's 42 to 47) is supported, and it came before the puck was lost. In the pictures (v07_bend_sheet_03.png): at picture 12 the forearm angle matches the demonstration (-70 against -68) but the tip stands beside the puck instead of behind it. Net tip move at -70 in the first picture's frame (v18): demonstrated 5.8 cm (5.0 to 7.4), robot 7.6, 9.6, 8.4 cm.

4. **confirmed**: Finding 4: runs 3, 5, 9, 10 all began with re-aiming answers (short tip move plus twist) that the driver stretched to 1 cm; the runs that began with about 1 cm got through their first stretch

   Measured (v02): first 10 decisions, mean asked length and mean |asked twist|: run 3 0.42 cm and 3.7, run 5 0.51 and 4.3, run 9 0.48 and 5.3, run 10 0.42 and 5.1; run 1 1.01 and 1.1, run 2 1.15 and 0.2, run 4 0.95 and 2.3, run 7 1.15 and 2.2, run 8 1.14 and 2.8, run 6 0.84 and 1.9. Pictures of run 3 looked at (v10_runs_3_4.png): the model asks 0.36 to 0.49 cm with -3 to -5 degrees of twist while the arc leaves up-left; with 1-cm steps the puck is clear of the line to its right by picture 6. First order only: with the asked length the puck would still have been pushed about 4 cm the same way before the answers turned left.

5. **confirmed**: Finding 5: the recording rule makes two kinds of step; every step is a unit step (5th percentile 1.02, median 1.22); steps under 0.7 cm are 27 % of steps, 11 % of the path and 44 % of all twist

   Measured (v09): size = max(length / 1 cm, largest joint change / 5 degrees) over 17,533 consecutive steps: 5th percentile 1.02, median 1.22. Steps under 0.7 cm: 27.4 % of the steps, 11.3 % of the path, 44.2 % of all |twist|; joint 4 moved most in 96.0 % of them; median 0.40 cm and 6.3 degrees; 17.2 degrees per cm pooled against 2.8 on longer steps. Brief's scout numbers corrected as the analyst says: median step 1.06 cm, mean 0.94, 18.2 % under 0.5 cm, 5.4 % under 0.2 cm, 0.65 cm for steps twisting 6 degrees or more. I also tested whether the short tip moves are only the sideways swing a forearm twist gives the tip (v12, arm model): no. The twist alone swings the tip 0.045 cm per degree the OTHER way; the recorded move is a median 124 degrees from that swing, so the demonstrator really moved the tip round while turning.

6. **corrected**: Finding 6: not every failure was under-turning: run 6 (sheet 04) the model asked for almost no twist; run 8 (sheet 06) turned more than the demonstrations; run 4 (sheet 02) had not under-turned where the puck drifted (whole-arc net turn 77 against 76 degrees)

   Run 6, measured (v08, v09): mean |asked twist| 1.37 degrees per decision against 4.69 per demonstrated picture (0.29); joint 4 stayed within -12.1 to +16.8; after 8, 12, 16 cm the robot had turned +11.9, +11.8, +7.6 degrees against demonstration medians +29.5, +68.8, +106.0, below every one of 11 demonstrations from 8 cm on; from picture 19 answers of 0.03 to 0.36 cm alternate by about 180 degrees. This IS under-turning, the strongest of the whole test, and it came before the failure; what the analyst means is that its cause is the model's own twist, not the driver's step. Its start tip was 5.0 cm from the demonstrated start tip, the furthest of all runs, and the first picture does not look like the first picture of demonstration 04-01 (v08_sheet_04.png). Run 8, confirmed as an over-turn: after 9, 12, 15, 18 cm it had turned +21.9, +40.5, +60.3, +75.4 against medians +15.5, +24.8, +40.8, +58.4 and above the largest of the six demonstrations each time; in the pictures the puck is inside the loop from picture 12 (v08_sheet_06.png). The "19 to 46 degrees further clockwise at the same place" cannot be checked (start 3.2 cm off); by path length plus the 7-degree pre-twist it is 13 to 27 degrees. Run 4: "77 against 76" is wrong as a whole-arc figure: the six demonstrations end at a median of +91.5 degrees (range +25.8 to +101.3); 76 is the sum of the per-5-cm medians over the first 40 cm. The substance holds: by path length the robot was at or above the median (+36.4 against +12.5 at 15 cm, +49.6 against +37.8 at 20 cm, +68.3 against +62.8 at 35 cm) and ended at +86.7 from a start of +9.8.

7. **confirmed**: Finding 7a: the demonstrator did not keep the direction of travel up in the picture (median |angle| 46.4 degrees, 49 % within 45, 24 % beyond 90); model_angle_deg far from 0 is not by itself a failure sign; in every failed run the 10-picture median rises to 70 to 113 degrees at the failure

   Measured (v03): move_angle_deg equals atan2(left, forward) to 0.005 degrees; 17,670 steps, median |angle| 46.4, 23.8 % within 15, 49.0 % within 45, 23.7 % beyond 90; pushing steps 37.9 and 54.1 %; sheet 01 15.0, sheet 06 65.3. Averaged over the next five steps, which removes the hand's jitter (v12): median 32.2 degrees, 60 % within 45, 16 % beyond 90, so the statement holds but less sharply. Robot: median |model angle| 5.6, 5.1, 15.8, 21.3 in runs 1, 2, 4, 7; blocks of 10 pictures reach 71 to 113 in runs 3, 5, 6, 8, 9 around the failure and 70 and 122 in run 7, which finished. One detail differs: run 10 was at 42 to 50 through its bend failure; its 87 and 89 blocks come after picture 50.

8. **corrected**: Finding 7b: what the demonstrator kept is the puck straight ahead of the tip; the robot's aim was outside the demonstrations' 5-to-95 % range (-28 to +40) in 35 to 77 % of the pictures of nine runs

   Recomputed from the analyst's cache of puck positions (v15, v17; I did not rerun the tracker, but in 12 random pictures the tracker's distances match what I see, v16_tracker_spot_check.png). The arithmetic is right: tip within 2.8 cm of the puck in 44 % of the demonstration pictures, their aim median +1.7, quartiles -11.1 to +16.2, 68 % within 20 degrees, 86 % within 30, 96 % within 45. But the 35 to 77 % compares ALL robot pictures (tip within 2.8 cm in only 46 to 70 % of the pictures of runs 3 to 10) with only that 44 % of demonstration pictures. Like for like the robot was outside in 0 to 53 % (near against near: run 8 0 %, run 4 5 %, run 3 18 %, runs 5 and 9 40 %, run 6 51 %, run 10 53 %) or in 0 to 21 % (all against all, range -90 to +104). Over all demonstration pictures 29 % are more than 45 degrees from the median aim and 42 % more than 30. The touching distance by config.py is 1.85 cm, the tracker's median is 2.94 cm, so the 2.8-cm line cuts through the middle of the data.

9. **confirmed**: Finding 8: offline, step by step, the fold models' twist is about two thirds of the demonstrated size (0.63 to 0.69), not half; the best step-by-step gain is below 1 on four of six sheets

   Measured from the saved predictions (v04), 17,773 non-stop rows, seven fold runs: mean |predicted| / mean |demonstrated| 0.63 to 0.69; slope of demonstrated on predicted 0.87 to 0.94 (E 0.93, A 0.90); predicted on demonstrated 0.35 to 0.39; correlation 0.55 to 0.59; sign right on steps of 2 degrees or more 0.77 to 0.79. E per sheet: size 0.56, 0.96, 0.70, 0.63, 0.45, 0.49; gain 0.65, 0.54, 1.02, 0.97, 0.70, 1.28; A gain 0.64, 0.54, 0.97, 0.97, 0.84, 1.16. Twist focus: size 0.67 and 0.69, correlation 0.59 and 0.58.

10. **confirmed**: Finding 9: summed along held-out demonstrations the predicted twists give back about half of the demonstrated turn after 10 steps and two thirds after 40, and drift widely

   Measured (v04), E recipe, 10 / 20 / 40 steps: median demonstrated turn 34.0 / 55.5 / 89.8 degrees; median |drift| 17.9 / 27.0 / 41.0 (90th percentile 48.5 / 78.3 / 122.5); share given back 0.52 / 0.59 / 0.65; correlation 0.72 / 0.76 / 0.75; best gain 1.01 / 0.99 / 0.86. A: drift 17.3 / 26.8 / 41.6, share 0.52 / 0.59 / 0.64, best gain 0.99 / 0.97 / 0.81. Share after 10 steps per sheet (E) 0.20, 0.64, 0.66, 0.52, 0.20, 0.47; sheet 05 correlation after 40 steps 0.14. My addition: a twist gain makes the summed drift worse, not better: with gain 1.5 the median |drift| is 19.8 / 33.3 / 61.6, with gain 2.0 28.8 / 53.2 / 95.4 degrees.

11. **confirmed**: Finding 10: the demonstrated twist is not noisy step to step, so a target smoothed over plus or minus 2 steps would help only a little

   Measured (v04): next-step correlation 0.78; 3.7 % of two twisting steps in a row reverse; the mean over plus or minus 2 steps keeps 74 % of the variance and removes a part of standard deviation 2.23 degrees (18 % of the variance; 32 % on sheet 01, 28 % on sheet 02, 16 to 19 % on sheets 03 to 06); step-to-step change 3.41 degrees raw. E predictions against the raw twist: correlation 0.59, size ratio 0.62; against the smoothed twist 0.66 and 0.70 (the analyst's 0.58 to 0.64 and 0.64 to 0.71 on a slightly different row set).

12. **corrected**: Finding 11: on unseen sheets the models answer a blend of re-aim and push; the robot's models had seen all six sheets and asked for much clearer re-aim steps, so the runs of 11.11 say little about an unseen line

   The offline part is right (v19; the analyst's fold_steps_E.npz equals my own inference on 87 pictures to 0.0000): on demonstrated re-aim steps fold E asks 0.61 cm and 3.4 degrees, on push steps 0.70 cm; told apart with an area under the curve of 0.62 by length, 0.65 by twist, 0.67 by twist per cm (A 0.69). The conclusion about the robot runs is wrong for sheet 03 and right for the loops. I ran the fold checkpoints that never saw the sheet on the pictures the robot really took (v05, v14; inference only; my pipeline gives back the logged answers of run 9 to 0.0000). First 13 pictures at the sheet-03 bend: net twist -56.6, -54.4, -50.1 degrees (unseen-sheet model) against the logged -52.8, -68.7, -59.6 (runs 5, 9, 10); asked length 0.42 to 0.51 cm against the logged 0.40 to 0.46; twist per asked cm 7.6 to 9.1. Whole runs on sheet 03: twist correlation with the logged answers 0.76 to 0.86. Sheet 02: the unseen-sheet model asks 1.2 to 1.4 times the logged twist. Sheet 04: more twist (+24.1 against +10.7 over 13 pictures). Sheet 05: unrelated (correlation 0.11; +0.8 against -31.0). Sheet 06: correlation 0.55; +9.1 against +45.5. First order only (the pictures come from runs steered by the all-sheets model).

13. **confirmed**: Finding 12: driver rules compared without the robot: any rule that lets the twist per cm follow the model brings the turn at a bend close to the demonstrated one (logged answers: now 4.1 / 5.3 / 4.6 degrees per cm, asked length 9.0 / 11.4 / 11.4; simulated walk: first big turn 0.47 / 0.55 now, 0.79 / 0.84 asked length, 0.81 / 0.85 unit step)

   Calculated, as arithmetic only. Logged answers, first 13 decisions (v02): now 4.13 / 5.34 / 4.74 degrees per cm, per asked cm 9.0 / 11.4 / 11.5. My own walk with a simpler picture choice (v11) gives the same table, E / A: now 0.47 / 0.55; gain 1.5 0.70 / 0.81; gain 2.0 0.87 / 1.00; asked length 0.79 / 0.86 with 1.57 / 1.70 times the decisions; asked length x 1.3 0.60 / 0.67; unit step 0.80 / 0.88 with 1.22 / 1.30; half step 0.94 / 1.07; one decision per picture 0.61 / 0.69; sheet-03 standard starts now 0.46 / 0.55, unit step 0.94 / 1.25. The rule already in the code (asked length, at most 1.0 cm, no floor) gives 0.80 / 0.88. Two cautions. Measured on a path without the hand's jitter every share drops: now 0.42 / 0.48, asked length 0.72 / 0.74, x 1.3 0.54 / 0.57, unit step 0.70 / 0.75, gain 1.5 0.63 / 0.70. And the walk has no picture feedback and keeps the tip on the demonstrated path, so it counts decisions per cm times twist per decision; it cannot say whether the robot then follows the line.

14. **confirmed**: Finding 13: a rule that turns the forearm towards the model's predicted direction of travel has the wrong sign against the demonstrations (correlation +0.43; camera turned away from the travel direction in 62 % against 18 % of the steps at 45 to 90 degrees)

   Measured (v03, v12, v13). A twist of +1 degree turns the picture heading by -0.90 degrees (opposite sign in 99.7 % of steps over 4 degrees), so turning the camera towards a travel direction on the picture-left needs a negative twist. Demonstrated: correlation of twist with travel direction +0.428 within 90 degrees, +0.049 degrees per degree, per sheet +0.08 to +0.58; travelling 45 to 90 degrees left the twist is above +1 in 65 % and below -1 in 20 %; against the net direction of the next five steps +0.35. It is not a geometric side effect: with the swing of the twist taken out of every step the correlation rises to +0.66. Direct check on the robot logs: in the first ten pictures of runs 3, 5, 9, 10 the rule would have turned the forearm against the model's own twist in 8, 9, 10 and 7 pictures (asked travel -11 to -27 degrees to the right, asked twist -3.7 to -5.3). The puck-rule and chunk-bend numbers I did not recompute.

15. **corrected**: Finding 14: as a correction of the aim the models on the robot turned the forearm 0.40 times as strongly per decision as the demonstrator per picture

   Recomputed from the analyst's puck cache (v15). Demonstrated slope -0.074 degrees of twist per degree of aim, correlation -0.25 (7,479 pictures): confirmed. The robot's figure depends on the window: the analyst's code centres the 45-degree window on the robot's own median aim (about -22) and gets -0.030 (0.40); centred on the demonstrations' median aim, as for the demonstrations, it is -0.046 (0.61), correlation -0.30, 295 pictures. Right statement: 0.4 to 0.6, from correlations of -0.2 to -0.3. Too weak to carry the gain estimate of 2.5 quoted in the gain recommendation.

16. **corrected**: Corrections to the brief: the puck is 2.4 cm across, not 4 cm; the models on the robot were trained on all six sheets; joint 4 started between -9.7 and +9.8 degrees; comparisons of place are solid for runs 5, 9 and 10

   Confirmed: PUCK_PARAMETERS diameter_m 0.024, height_m 0.029 in config.py; summary.json of run 9 names checkpoints/all_sheets.pt; joint 4 at the first picture -1.8, +6.1, -1.0, +9.8, -9.3, +3.2, -1.7, +7.2, -5.2, -9.7 degrees (v00). Not confirmed: the last part. Distance from the robot's first tip position to the demonstrated start tip (14.0, -1.5) cm, which is the same in every standard-start demonstration (v06): run 4 3.8, run 5 3.1, run 6 5.0, run 7 4.0, run 8 3.2, run 9 1.3, run 10 3.2 cm; first picture heading +0.5, +11.7, +1.5, +8.5, -2.3, +3.2, +12.3 degrees against the demonstrations' -6.1. Only run 9 began near the demonstrated start, and run 6 was the furthest although its tip lay 0.2 cm from a later demonstrated tip position.

17. **corrected**: Recommendation 1 says the driver needs about 40 new lines and 2 to 3 hours for the asked-length rule and a guard

   Read in the working copy (02_code/model_mode.py and config.py, both saved 18:18:34 on 11.11.26, while the analyst was working): model_step already has step_length_rule "predicted" (tip travel = step_length_gain x asked length, at most step_most_m = 1.0 cm, no floor), the command line has --step-length and --length-gain, and ModelRun.is_stalled ends a run whose tips stay within 3 cm and whose forearm turns under 15 degrees over 30 pictures. New would be only the 0.2-cm floor, the 1.5-cm cap, the unit step and the tighter guard. Two checks of mine on that code and proposal: on paper (model_mode.executor_check and model_step, nothing moves; v22) 1,200 recorded steps of any length and 500 recorded steps of 0.05 to 0.5 cm go through with the asked length without one refusal, the planned tip move within 0.02 cm of the recorded one in 95 %; and on the logged answers the proposed guard (8 decisions, 1 cm, 5 degrees) would not have ended runs 1, 2, 4, 7 and fires at decision 24 of run 6, where the guard in the code fires at 44 (v20, calculated, first order).

18. **unverifiable**: Expected effect of recommendation 1: the asked-length or unit-step rule cures the overshoot type of failure (runs 3, 5, 9, 10)

   Only the robot can show it. What I can add (calculated, first order, v18): the logged answers carried out with their own length along the logged headings give a net tip move, at the picture where joint 4 reaches -70 degrees, of 3.5 / 4.6 / 3.8 cm at factor 1.0 and 4.6 / 5.9 / 4.9 cm at factor 1.3 (runs 5 / 9 / 10) against 5.8 cm demonstrated (5.0 to 7.4) and 7.6 / 9.6 / 8.4 cm really driven. So the rule would put the tip about where the demonstrations had it; whether the model's answers then stay good on the pictures it sees is not known.

### The analyst's findings with their evidence

1. (measured, confidence high) Twist per cm recomputed: by the sum of |joint-4 change| per cm of tip path the robot was below the demonstrations on every sheet (0.24 to 0.96 of the sheet's median), as the scout said. But that measure hides where and why.

   Evidence: Demonstrations, median over episodes, degrees per cm: sheet 01 1.57, 02 2.54, 03 4.64, 04 5.65, 05 3.39, 06 4.42 (209 episodes, audit-cleaned frames.csv). Robot (steps.csv, measured joint-4 change and tip path): run 1 1.05, run 2 0.38, run 3 2.45, run 4 2.01, run 5 3.16, run 6 1.38, run 7 2.78, run 8 2.60, run 9 3.28, run 10 2.93. Of the demonstrated twisting inside a 5-cm stretch only 0.67 (sheet 01) to 0.90 (sheets 03, 06) is still there at the end of the stretch. By NET turn per 5 cm (median over stretches) robot/demonstrations was 1.27 (run 4), 1.03 (run 7), 1.00 and 1.11 (runs 5, 9, which include the wandering after the failure), 0.72 (run 10), 0.39 (run 8), 0.19 (run 6). Script 01_twist_per_cm.py, output 01_twist_per_cm.out.txt, figure 01_twist_along_path.png, table 01_twist_per_window.csv.

2. (measured, confidence high) The driver carried out the twist it was given: no clamp ever bound, so the 10-degree step limit and the 117-degree joint limit played no part.

   Evidence: All ten runs: 0 clamped steps; largest asked twist 7.5 degrees; joint 4 stayed within -80.9 to +101.5 degrees; measured joint-4 change equals the commanded twist within 0.13 to 0.53 degrees (one 1.97 in run 3). Inline check on steps.csv (numbers also in 01_twist_per_cm.out.txt, column 'clamped').

3. (measured, confidence high) At the first bend of sheet 03 (runs 5, 9, 10: 3 of 3 failed there) the camera fell 42 to 47 degrees behind the demonstrations at the same place BEFORE the tip left the demonstrated path, and the reason was tip travel per decision, not twist per decision.

   Evidence: Standard-start demonstrations of sheet 03 from end A (8 episodes): joint 4 reaches -70 degrees after a median of 12 pictures (range 6 to 13) and 7.8 cm of tip travel (5.3 to 10.2); first 10 cm of path: 16 pictures, -88 degrees (range -105 to -70) = 1.46 pictures per cm, 5.7 degrees per picture, 8.3 degrees per cm. Robot: -70 degrees at picture 12 after 12.1 cm (run 9), picture 13 after 12.8 cm (run 10), picture 17 after 17.0 cm (run 5); the lengths the model ASKED for up to there add up to 5.5, 5.2 and 7.7 cm. First 13 decisions: 1.0 decision per cm, 5.3 / 4.6 / 4.1 degrees per decision (0.93 / 0.80 / 0.71 of the demonstrated per picture), 5.3 / 4.6 / 4.1 degrees per cm (0.64 / 0.55 / 0.49). Heading gap to the demonstrations at the same place: run 9 +1.6 at picture 0, -41.7 at picture 8 with the tip within 0.2 cm of a demonstrated position, tip leaves the path at picture 20; run 5 -46.9 at picture 10, leaves at 19; run 10 -48.5 at picture 9. The demonstrations' own spread of heading at one place is 5 to 7 degrees (standard deviation). Scripts 13_same_turn_how_much_travel.py, 02_heading_against_demonstrations.py, 09_rules_on_logged_answers.py; pictures sheet_run_09.png against sheet_plan_03-01.png; figure 12_summary_figure.png.

4. (measured, confidence high) The four runs that failed within their first bend (3, 5, 9, 10) all began with the model asking for re-aiming steps (short tip move plus twist), which the driver stretched to 1 cm; the runs that began with the model asking for about 1 cm got through their first stretch.

   Evidence: First 10 decisions, mean asked length and mean |asked twist|: run 3 0.42 cm and 3.7 degrees, run 5 0.51 and 4.3, run 9 0.48 and 5.3, run 10 0.42 and 5.1; against run 1 1.01 and 1.1, run 2 1.15 and 0.2, run 4 0.95 and 2.3, run 7 1.15 and 2.2, run 8 1.14 and 2.8, run 6 0.84 and 1.9. In the pictures of runs 3 and 9 the tip swings 1 cm per step to the outside of the bend, passes beside the puck and then pushes it off the line. 13_same_turn_how_much_travel.out.txt, sheet_run_03.png, sheet_run_09.png.

5. (measured, confidence high) Why a fixed 1-cm step cannot follow a sharp bend: the recording rule (a waypoint after 1 cm of tip travel OR 5 degrees on a joint) makes two kinds of demonstrated step, and almost half of all forearm turning sits in the short kind.

   Evidence: Every demonstrated step is a 'unit step': max(length / 1 cm, largest joint change / 5 degrees) has its 5th percentile at 1.02 and median 1.22 (17,533 steps). Steps shorter than 0.7 cm are 27 % of the steps, 11 % of the path and 44 % of all |twist|; joint 4 is the joint that moved most in 96 % of them; median 0.40 cm and 6.3 degrees; pooled 17.2 degrees per cm against 2.8 on long steps. The models never asked for more than 7.5 degrees per decision on the robot, so at 1 cm per decision the robot cannot exceed about 6 to 7 degrees per cm where the demonstrations made 8.3 (first 10 cm of sheet 03) to 17 (re-aiming steps). Inline check on frames.csv; 03_picture_direction.out.txt part 1; 11_reaim_or_push.out.txt.

6. (measured, confidence medium) Not every failure was under-turning: on sheet 04 (run 6) the model itself asked for almost no twist; on sheet 06 (run 8) the forearm turned MORE than the demonstrations before the failure; and on the gentle arc of sheet 02 (run 4) the puck left the middle of the line although the forearm had not under-turned.

   Evidence: Run 6: net joint-4 change per 5 cm 0.5, 12.3, -5.2 degrees against the demonstrations' 15.7, 29.9, 42.6; joint 4 stayed within -12 to +17 degrees where demonstration 04-01 reached +118 by row 25; mean |asked twist| 1.37 degrees per decision against 4.77 per demonstrated picture (0.29); from picture 19 the answers alternate left and right by about 100 degrees with asked lengths of 0.03 to 0.36 cm and twists of about 1 degree, each carried out as a 1-cm move (the 30 moves back and forth). Run 8: 30.1 and 32.8 degrees per 5 cm at 5 to 15 cm against 17.6 and 22.5 (demonstration ranges 11 to 20 and 20 to 37); camera 19 to 46 degrees further clockwise than the demonstrations at pictures 8 to 11: a wrong turn at the crossing. Run 4: whole-arc net turn 77 degrees against 76; at the place where the puck drifted out (pictures 17 to 24) the robot had turned more than the demonstrations' median (38.3 against 20.8 degrees at 15 cm, 50.8 against 44.5 at 20 cm); on-path heading gap median 9.9 degrees, largest 28.3. 01_twist_per_cm.out.txt part 3, 02_heading_against_demonstrations.out.txt, sheet_run_06.png, sheet_run_06_hairpin.png, sheet_run_08.png, sheet_run_04.png. Weaker for runs 4 and 8 because their tips started 2.7 to 2.9 cm from every demonstrated tip position.

7. (measured, confidence medium) The demonstrator did NOT keep the direction of travel 'up in the picture'; what was kept is the puck straight ahead of the tip. So model_angle_deg far from 0 is not by itself a sign of trouble.

   Evidence: move_angle_deg over 17,670 demonstrated steps: median |angle| 46.4 degrees, 24 % within 15, 49 % within 45, 24 % beyond 90; pushing steps (0.7 cm or more) median 37.9, 54 % within 45; sheet 01 median 15.0, sheet 06 65.3. Aim (direction from tip to puck minus picture heading, team's puck tracker on all 18,032 pictures, puck found in 99.3 %): when the tip is at the puck (centre within 2.8 cm, 44 % of the pictures) median +1.8, quartiles -10.9 to +16.2, 68 % within 20 degrees of the median, 86 % within 30, 96 % within 45. The demonstrated tip travels 1.67 times as far as the puck (median per demonstration): much of its travel is moving round the puck. Robot: median |model angle| 5.6, 5.1, 15.8 in runs 1, 2, 4 and 21.3 in run 7; in every failed run the 10-picture median rises to 70 to 113 degrees at the failure, but run 7 also had blocks of 70 and 122 on the loop and finished. Aim outside the demonstrations' 5-to-95 % range (-28 to +40) in 35 to 77 % of the pictures of nine runs (0 % in run 2); median aim in the ten pictures before the sheet-03 failures +83 (run 5), +68 (run 9), +44 (run 10); +53 to +62 for 40 pictures at the hairpin of run 6. 03_picture_direction.out.txt, 08_aim_of_the_camera.out.txt, 08_aim_over_runs.png, 03_model_angle_over_runs.png.

8. (measured, confidence high) Offline, step by step, the fold models' twist is about two thirds of the demonstrated size (not half), and the gain that fits best step by step is below 1 on four of six sheets.

   Evidence: Seven fold runs, 17,773 non-stop rows of held-out sheets: mean |predicted| / mean |demonstrated| 0.63 to 0.69 over all sheets (E recipe per sheet 0.56, 0.96, 0.70, 0.63, 0.45, 0.49). Slope of demonstrated on predicted twist: E 0.65, 0.54, 1.02, 0.97, 0.70, 1.28 for sheets 01 to 06 (all 0.93); A 0.64, 0.54, 0.97, 0.97, 0.84, 1.16 (all 0.90). Slope of predicted on demonstrated 0.35 to 0.39; correlation 0.55 to 0.59; sign right on steps of 2 degrees or more 0.77 to 0.79 (per sheet 0.63 to 0.86). The trained 'twist focus' variant (angle scale 2.5) changes little: size ratio 0.67 and 0.69, correlation 0.59 and 0.58. 05_offline_twist.out.txt part 1.

9. (measured, confidence high) Summed along held-out demonstrations (one decision per demonstrated picture, the demonstration's own pictures) the predicted twists give back about half of the demonstrated turn after 10 steps and two thirds after 40, and drift widely.

   Evidence: E recipe, all sheets: after 10 / 20 / 40 steps the median demonstrated turn is 34.0 / 55.6 / 89.8 degrees, the median |drift| 17.9 / 27.0 / 41.0 degrees (90th percentile 48.5 / 78.3 / 122.5), the share of the demonstrated turn given back 0.52 / 0.59 / 0.65, the correlation 0.72 / 0.76 / 0.75, and the gain that fits the sums best 1.01 / 0.99 / 0.86. A recipe: drift 17.3 / 26.8 / 41.6, share 0.52 / 0.59 / 0.64, best gain 0.99 / 0.97 / 0.81. Share after 10 steps per sheet (E): 0.20, 0.64, 0.66, 0.52, 0.20, 0.47; on sheet 05 the 40-step sums are unrelated to the demonstrations (correlation 0.14). 05_offline_twist.out.txt part 2, 05_summed_twist_scatter.png, 05_twist_noise_example.png.

10. (measured, confidence high) The demonstrated twist is NOT noisy from step to step, so a target smoothed over plus or minus 2 steps would help only a little.

   Evidence: Correlation between the twist of one step and the next +0.78 (0.62 on sheet 01 to 0.81 on sheet 04); of two twisting steps in a row only 4 % reverse; smoothing over +-2 steps keeps 74 % of the variance and removes a part with standard deviation 2.2 degrees (18 % of the variance; 35 % on sheet 01, 16 to 17 % on sheets 03, 04, 06); the step-to-step change shrinks from 3.41 to 1.11 degrees (factor 0.33). The saved predictions already fit the smoothed twist better than the raw one (correlation 0.64 against 0.58, size ratio 0.71 against 0.64) and the mean of the next 5 steps about as well (0.63). 05_offline_twist.out.txt part 3.

11. (measured, confidence high) On unseen sheets the models cannot tell from one picture whether the demonstrator re-aimed or pushed on, and answer a blend of both: this, not noise, is why their twist is small there.

   Evidence: Fold models with their whole exact step recomputed (twist equal to the saved predictions to 0.0000 degrees): on demonstrated re-aim steps (0.40 cm, 6.4 degrees, 17.9 degrees per cm) recipe E asks 0.61 cm and 3.3 degrees (5.6 per cm), recipe A 0.57 cm and 3.6 degrees; on push steps (1.14 cm, 2.4 degrees) E asks 0.70 cm and 2.1 degrees. The answers separate the two kinds with an area under the curve of only 0.67 (E) and 0.69 (A) by asked twist per cm (0.58 to 0.61 on sheets 02, 05, 06). The models A and E on the robot had seen all six sheets and asked for much clearer re-aim steps (0.34 to 0.52 cm with 5 to 6 degrees at the sheet-03 bend), so the robot runs of 11.11 say little about an unseen line. 04_fold_model_steps.py (inference only), 11_reaim_or_push.out.txt, 11_reaim_or_push.png.

12. (calculated, confidence medium) Driver rules compared without the robot: any rule that lets the twist per cm follow the model brings the turn at a bend close to the demonstrated one; a gain alone does too but leaves the tip's over-travel.

   Evidence: (a) The answers logged at the sheet-03 bend (first 13 decisions of runs 5 / 9 / 10; demonstrations 8.3 degrees per cm): now 4.1 / 5.3 / 4.6; gain 1.5 6.1 / 7.9 / 6.7; gain 2.0 7.4 / 9.6 / 7.8; asked length x 1.3 7.0 / 8.7 / 8.8 with 7.6 / 7.9 / 6.7 cm of travel instead of 13; asked length 9.0 / 11.4 / 11.4; unit step (travel and twist scaled together to 1 cm or 5 degrees, factor at most 2) 7.6 / 10.8 / 9.7 with 6.5 / 6.0 / 5.6 cm (09_rules_on_logged_answers.out.txt, first order only). (b) Simulated robot walked along all 209 held-out demonstrations with the fold models' answers: share of each demonstration's first big turn (152 turns, median 74 degrees at 16 cm) made on arrival: now 0.47 (E) / 0.55 (A); gain 1.5 0.71 / 0.81; gain 2.0 0.86 / 0.98; asked length 0.79 / 0.84 with 1.56 / 1.67 times the decisions; asked length x 1.3 0.61 / 0.64; unit step 0.81 / 0.85 with 1.21 / 1.28 times the decisions; half step 0.5 cm 0.92 / 1.04 with 1.89 times; one decision per demonstration picture 0.61 / 0.69. For sheet-03 standard starts: now 0.48 / 0.55 (the robot measured 0.49 to 0.64 with the all-sheets models), unit step 0.95 / 1.22. The simulation assumes the tip stays on the demonstrated path and has no real picture feedback, so its open-loop gaps (median 25 to 50 degrees, larger with more twist) are not a ranking. 07_driver_rules.py, 07_driver_rules.out.txt, 07_driver_rules_example.png, 12_summary_figure.png.

13. (measured, confidence high) A hand-written rule that turns the forearm towards the model's predicted direction of travel would have the WRONG sign against the demonstrations; other rules made from the model's outputs or from the puck are no better than the model's own twist.

   Evidence: In the demonstrations the twist goes with the travel direction in the picture, not against it: correlation +0.43 within 90 degrees (+0.049 degrees of twist per degree; per sheet +0.08 to +0.59); with the tip travelling 45 to 90 degrees to the picture's left the camera is turned right in 62 % of the steps and left in 18 %. Reason seen in the pictures: at a bend the tip swings to the OUTSIDE to get behind the puck while the camera turns to the inside. Predicted travel direction of fold E: +0.33. Chunk end direction +0.32 (true), +0.24 (predicted). Chunk bend has the right sign: -0.38 (true), -0.48 (predicted), but the model's own twist output scores 0.58. A rule 'turn towards the puck' from the puck tracker: correlation 0.20 to 0.24 with the demonstrated twist against 0.66 for fold E on the same pictures, sign right 0.63 against 0.83, and it is not imitation learning. 03_picture_direction.out.txt part 2, 10_twist_from_the_path_ahead.out.txt, 08_aim_of_the_camera.out.txt part 4.

14. (measured, confidence medium) Measured as a correction of the aim, the models on the robot turned the forearm 0.40 times as strongly per decision as the demonstrator per picture; fold models on demonstration pictures 0.75 to 0.77. The demonstrator's own correction is slow and loose.

   Evidence: Least-squares twist per degree of aim within 45 degrees of the usual aim: demonstrated -0.074 (correlation -0.25, 7,480 pictures with the tip at the puck; -0.12 on sheet 03, -0.09 on sheet 04, about 0 on sheets 01 and 05); fold A -0.056, fold E -0.057 on the same pictures; robot, all ten runs -0.030 per decision (327 pictures). Aim 10 steps later against the aim now: 0.71 in the demonstrations, 0.81 on the robot (1 = no correction). 08_aim_of_the_camera.out.txt part 3, 08_twist_against_aim.png. Depends on the puck tracker's table positions, which were not checked here.

### The analyst's recommendations

1. Change the driver so that the tip travel follows the model instead of a fixed 1.0 cm, keeping the predicted twist. Put two rules behind one setting in MODEL_MODE_PARAMETERS: (a) 'asked length': travel = asked length x factor, kept within 0.2 to 1.5 cm (factor 1.3 to start); (b) 'unit step': travel and twist scaled together until the step is as large as a recorded one (1 cm of travel or 5 degrees of twist, factor at most 2). Add a guard that ends the run when the last 8 decisions moved the tip under 1 cm net and turned the forearm under 5 degrees net, and log the travel and factor used.

   Why: At the sheet-03 bend the forearm reached -70 degrees after 12 to 13 decisions like the demonstrations' 12 pictures, but after 12.1 to 12.8 cm of tip travel instead of 7.8 cm; the asked lengths added up to 5.2 to 5.5 cm. Runs 3, 5, 9, 10 all began with asked lengths of 0.42 to 0.51 cm that were stretched to 1 cm. 44 % of the demonstrated turning sits in steps of median 0.40 cm, which a fixed 1-cm step cannot reproduce.

   Needs: code only, then a robot test. Effort: 2 to 3 hours: about 40 lines in model_step, the settings, tests in test_model_mode.py, the executor and simulated checks run again

   Expected effect: Calculated from the logged answers: twist per cm at the sheet-03 bend 7.0 to 8.8 (length x 1.3) or 7.6 to 10.8 (unit step) against the demonstrated 8.3 and today's 4.1 to 5.3, with 5.6 to 7.9 cm of tip travel in the first 13 decisions instead of 13 cm. Simulation with held-out models: first big turn 0.79 to 0.85 of the demonstrated instead of 0.47 to 0.55. Expected to cure the overshoot type of failure (runs 3, 5, 9, 10); not the sheet-04 and sheet-06 failures.

   Risk: 1.2 to 1.6 times as many decisions (time per run up by a fifth to a half). The robot can stand still where the model asks for neither travel nor twist: 1 to 4 % of the decisions while on the line, 40 % (15 in a row) at the hairpin of run 6, 16 to 19 % within 3 cm of the goal, so the guard and the stop rule must be checked together. The factor 1.3 fits the all-sheets model on its own sheets; held-out models ask for less distinct steps.

2. Robot session on the twist: compare four driver settings at the places that failed, with model E: today's driver, --twist-gain 1.5, asked length x 1.3, unit step. Three runs each on sheet 03 from the standard start, then the best two on sheet 04 and on sheet 02 lying as in run 3. Put the sheet on the taped start mark and start with the forearm untwisted, then rerun scripts 02, 08 and 13 of this folder on the new logs.

   Why: Only the robot shows whether the picture feedback holds once the twist per cm is right; the offline sums drift by 41 degrees after 40 pictures, and the simulation has no real feedback. Sheet 03 failed 3 of 3 at the same bend, so 3 runs per setting already separate 'never' from 'always'.

   Needs: a robot test. Effort: about 16 to 20 runs, 1.5 hours of one session; 20 minutes to rerun the read-out

   Expected effect: A decision on the driver rule from three numbers per run: first bend passed or not, largest heading gap to the demonstrations before the bend (today 42 to 47 degrees), tip travel when joint 4 reaches -70 degrees (today 12 to 17 cm, demonstrated 7.8).

   Risk: Three runs per setting show only large differences. The place comparison fails if the sheet lies elsewhere: on 11.11 the tips of runs 4, 7 and 8 started 2.7 to 2.9 cm from every demonstrated tip position and joint 4 started between -9.7 and +9.8 degrees (demonstrated standard starts: 0.0).

3. Twist gain: keep 1.0 as the default; test 1.5 only as the cheap comparison in the session above (no code change); do not use 2.0 with the fixed 1-cm step.

   Why: Why 1.5: before the sheet-03 bend runs 9 and 10 made 0.61 to 0.64 of the demonstrated net turn per 5 cm (1 / 0.63 = 1.6); with the logged answers gain 1.5 gives 6.1 to 7.9 degrees per cm against the demonstrated 8.3; in the simulation the first big turn rises from 0.47 to 0.55 to 0.71 to 0.81. Why not more: the gain that fits summed turns best offline is 0.99 to 1.01 (per sheet 0.51 to 1.52), so a gain multiplies the wrong turns as well; run 4 already matched the demonstrated turn on sheet 02 (77 against 76 degrees) and would over-turn by half; and a gain leaves the tip travelling 13 cm where the demonstrations travel 8 to 9, which is what pushed the puck off in runs 3 and 9.

   Needs: a robot test. Effort: 3 runs inside the session above

   Expected effect: At best the sheet-03 bend is passed; it is the fallback if the step rule cannot be finished in time.

   Risk: Over-turning on gentle arcs and straight lines, and twice the twist jitter at gain 2; estimates of the 'right' gain range from 1.0 (least squares) over 1.6 (sheet 03) and 1.9 (10-step share 0.52) to 2.5 (reaction to the aim 0.40), so no single value is supported.

4. Do not write a twist feedback rule (towards the predicted direction of travel, towards the puck, or from the chunk), and so do not ask the professor for one. Add to the existing question 1 for the professor only: may the driver use the model's own step length (or scale the model's step to the size of a recorded step), apply a constant factor, and stop a run that stands still?

   Why: The rule on the direction of travel has the wrong sign against the demonstrations (correlation +0.43; camera turned away from the travel direction in 62 % against 18 % of the steps at 45 to 90 degrees). The puck rule explains 0.20 to 0.24 of the demonstrated twist against the model's 0.66 and is not imitation learning. The chunk bend (-0.48) carries no more than the twist output (0.58).

   Needs: the professor's answer (for the step rule and the guard only). Effort: one paragraph for the meeting of 23.11.26

   Expected effect: Saves the time a rule would cost, and settles early whether the driver change is allowed.

   Risk: If the professor allows no rescaling at all, the plain asked length (factor 1.0) is the purest form: the model's exact step carried out as predicted.

5. Before any new training run for the twist, check on the laptop with a frozen-feature probe whether 're-aim or push' can be told from the picture at all on a held-out sheet, with one picture and with the previous picture added. Only if it can (clearly above the 0.67 to 0.69 the present answers reach): train one variant on ki-014 that predicts the kind of step as a class and the step for each kind, two seeds, scored by the 10-step turn share (today 0.52) and the per-step twist correlation (0.58) next to path and stop.

   Why: On unseen sheets the models answer the same blend for both kinds (0.61 cm and 3.3 degrees on demonstrated re-aim steps of 0.40 cm and 6.4 degrees), which is why the exam model will turn too little per cm even with the new driver (simulation: 0.8 of the first big turn, 0.3 to 0.5 on sheet 05). Weighting the twist loss more did not help (twist focus: size ratio 0.67 to 0.69 against 0.64).

   Needs: retraining. Effort: probe 1 to 2 hours on the laptop; the variant one GPU night for six folds and two seeds

   Expected effect: Unknown until the probe is run; the target is a 10-step turn share of 0.7 or more on held-out sheets.

   Risk: The kind of step may simply not be visible in one picture (the demonstrator's timing varies), and a picture history can make a model copy its last move; then only more data helps.

6. Treat a smoothed twist target (plus or minus 2 steps, or the mean of the next 5 steps) as a low-priority extra: at most one run pair, and only if GPU time is free.

   Why: The demonstrated twist is not noisy step to step (next-step correlation +0.78; smoothing removes 18 % of the variance); the present predictions already correlate 0.64 with the smoothed twist against 0.58 with the raw one, so the ceiling of the gain is a few points and the turn per cm stays as it is.

   Needs: retraining. Effort: a few lines in training_targets.py and one GPU night

   Expected effect: Size ratio from about 0.64 to about 0.71 and a slightly steadier twist; no effect on the bend failures.

   Risk: Smoothing over twists in place spreads their twist onto neighbouring push steps and blurs the two kinds of step further.

7. Recording: (1) demonstrations on new, different tracks with bends of several sharpnesses in both directions (the printed practice tracks), before more episodes on the six sheets; (2) correction demonstrations that start from the states the robot reached on 11.11: camera 20 to 45 degrees off the push direction, and the tip beside the puck after an overshoot, at bends. Then test the chosen driver rule on a printed track the model has never seen.

   Why: The twist generalises worst of all outputs: on held-out sheets 0.20 to 0.66 of the demonstrated turn after 10 steps (0.20 on sheets 01 and 05), correlation 0.32 to 0.68. The demonstrations hold almost no mis-aimed states (4 % of the pictures with the tip at the puck are more than 45 degrees off the usual aim, 14 % more than 30), while 35 to 77 % of the robot's pictures were outside the demonstrated range; the existing turned starts are only 10 and 15 degrees.

   Needs: new demonstrations, then retraining and a robot test. Effort: 1 to 2 robot sessions of recording (about 30 to 40 episodes), one GPU night, one test session

   Expected effect: A stronger and better aimed twist on an unseen line; the only measure here that addresses the exam case directly.

   Risk: Recording time before the freeze of 8.1.27; correction episodes recorded in a different style can add a third kind of step; the benefit cannot be estimated offline.

### The verifier's challenges to the recommendations

1. Recommendation 1, factor 1.3 and the 1.5-cm cap. The factor is not settled and should not be fixed from three runs of a model that had seen the sheet. For that model at the sheet-03 bend my vector sum supports 1.3 (net tip move 4.6 to 5.9 cm against 5.8 demonstrated; 3.5 to 4.6 at factor 1.0). For models that never saw the sheet, which is the exam case, everything points to 1.0: the walk gives 0.79 to 0.86 of the first big turn at 1.0 and 0.60 to 0.67 at 1.3 (on a jitter-free path 0.72 to 0.74 against 0.54 to 0.57, with today's driver at 0.42 to 0.48), and on the robot's own bend pictures the unseen-sheet models' twist per cm would be 7.6 to 9.1 at 1.0 and 5.8 to 7.0 at 1.3 against 8.3 to 9.0 demonstrated. On unseen sheets the asked length is about 0.67 cm everywhere (0.70 cm even on demonstrated push steps of 1.14 cm), so x 1.3 nearly restores today's 1-cm step. Test 1.0 and 1.3. Keep the cap at the 1.0 cm that is in the code: with x 1.3 and a 1.5-cm cap the steps on straight stretches become 1.3 to 1.5 cm (asked 1.0 to 1.15 cm in runs 1 and 2), longer than anything driven on 11.11.26, exactly where the runs worked, and the stop rule of two pictures in a row then spans up to 3 cm.

2. Recommendation 1, unit step. It is a hand-written gain on the model's twist of about 0.7 to 2 that depends on the answer: it cuts every twist above 5 degrees and doubles small uncertain answers (the alternating 0.2 cm and 1 degree at the hairpin of run 6 would become 0.4 cm and 2 degrees back and forth). It needs the professor's yes more than the plain asked length does, and it sits oddly beside the advice not to write a twist rule. The plain asked length reaches the same share in the walk (0.79 to 0.86 against 0.80 to 0.88).

3. Recommendation 1, risk at the stop. With shorter steps near the goal two pictures in a row with a stop chance of 0.5 lie about 0.3 cm apart instead of 1 cm, so the robot is likely to stop earlier than today, against Emranul's wish that the puck covers the line's end completely (run 2 already left a bit of the end visible). The analyst names the interaction but puts no number on it; it has to be read from the first runs before the rule is adopted.

4. Recommendation 2, the robot session. As written it uses only model E, which was trained on the sheets it is tested on, so it cannot say anything about the exam's unseen line. The fold checkpoints exist (for example 04_runs/gpu_ki014/2026-11-10_resnet18_chunk_stop_weight_25/checkpoints/without_path_03.pt with --run-name resnet18_chunk_stop_weight_25) and on the robot's own pictures such a model asks nearly the same at the sheet-03 bend, so one setting should be run with it. Three runs per setting separate only 0 of 3 from 3 of 3 (Fisher exact, one-sided 0.05); four settings leave no room for the held-out model, so drop the unit step or the gain from the first session. "Sheet 02 lying as in run 3" is recorded nowhere except in that run's pictures. Sheet 04 needs a run with today's driver at the taped placement first, otherwise a pass or a fail there cannot be put down to the driver rule (on 11.11.26 that run began 5.0 cm from the demonstrated start).

5. Recommendation 3, twist gain. Keeping 1.0 follows from the evidence, and my own numbers strengthen it: offline the summed drift grows with a gain (median 19.8 / 33.3 / 61.6 degrees after 10 / 20 / 40 steps at gain 1.5 against 17.9 / 27.0 / 41.0), on sheet 02 unseen-sheet models already ask 1.2 to 1.4 times the twist of model E, and run 8 over-turned at gain 1.0. But the gain is the only option on the list that touches the sheet-04 kind of failure (twist 0.29 of the demonstrated, right sign in pictures 4 to 8), so if 1.5 is tried it belongs on sheet 04, not only on sheet 03 where the step rule is the better lever. The supporting estimate of 2.5 from the aim reaction is not robust (0.40 becomes 0.61 with the window centred as for the demonstrations).

6. Recommendation 5, the probe before retraining. Its premise is weaker than stated: on the robot's pictures of the sheet-03 bend the unseen-sheet models asked clear re-aim steps (0.42 to 0.51 cm with peaks of 5 to 6 degrees). The weakness on unseen sheets sits on the loops (sheet 05: twist unrelated to the all-sheets model's, correlation 0.11; 10-step share 0.20). And which picture gets the short step depends on where the recording rule cut the waypoint, so a low probe score would not show that the picture lacks what control needs. Low value for 1 to 2 hours; a held-out robot run costs ten minutes and answers more.

7. Recommendation 7, correction demonstrations. The reason given does not hold as stated: "almost no mis-aimed states (4 %)" and "35 to 77 % of the robot's pictures outside the demonstrated range" compare all robot pictures with the 44 % of demonstration pictures whose tip is within 2.8 cm of the puck. Like for like the robot was outside in 0 to 53 % or 0 to 21 %, and 29 % of all demonstration pictures are more than 45 degrees from the median aim: the demonstrations already hold many pictures with the tip beside the puck. To first order the overshoot states of 11.11.26 were made by the driver's step, so test the driver first and record corrections only for what still fails. The other half, new tracks of several sharpnesses, does follow from the evidence (sheets 05 and 06).

### What the verifier says was missed

1. A held-out test on the robot is possible today and was not proposed. On the robot's own pictures the fold checkpoint trained without the sheet answers almost like the all-sheets model on sheet 03 (first 13 pictures: net twist -50 to -57 against -53 to -69 degrees, asked length 0.42 to 0.51 against 0.40 to 0.46 cm), very differently on sheet 05 (twist correlation 0.11; the run that succeeded there would probably not succeed with an unseen-sheet model) and on sheet 06 (0.55). Inference only, v05 and v14.

2. The start state was not controlled and the analyst counted three runs as solid that are not. Every run placed as in a plan began 1.3 to 5.0 cm from the demonstrated start tip with the camera heading 4 to 18 degrees off, and joint 4 between -9.7 and +9.8 degrees; the driver takes the hand-placed pose as it is (held_joints_at keeps joint 4). The plan says to start untwisted on the taped mark but not how; a move to the recording tool's start pose before each run would make runs comparable. Until then no comparison of place on the table can be trusted except for run 9.

3. The code state. The asked-length rule, the length gain and a stall guard are already in the working copy, so the first test of the main recommendation needs no programming. The driver's own executor check, as configured, only takes recorded steps of 0.5 cm or more (executor_check_smallest_step_m 0.005), which are not the steps the new rule will command; I ran it on all steps and on 500 short ones (no refusal), but that should become part of the standing checks.

4. The split of the under-turn into its two factors. Twist per decision was 0.54 to 0.93 of the demonstrated twist per picture and decisions per cm 0.69 of the demonstrated pictures per cm; they multiply to the 0.37 to 0.64 seen. The pre-twisted starts (5 to 10 degrees towards the bend) hide part of the first factor in the analyst's "same number of decisions".

5. Sheet 04 has no remedy in the plan. It is the strongest under-turn of the test (below every demonstration from 8 cm on), on a sheet the model was trained on, with a start 5.0 cm from the demonstrated one and a first picture unlike the demonstrations'. An unseen-sheet model would have asked for more twist on the same pictures (+24.1 against +10.7 degrees over 13 pictures), which points at the state the run was in, not at a lack of training. A repeat with the standard placement and today's driver comes before any conclusion.

6. Whether the arm and the puck follow steps of 0.2 to 0.5 cm. On 11.11.26 the 1-cm steps measured 0.94 to 1.05 cm, an error of about 0.05 cm, which would be up to a quarter of a 0.2-cm step; and nobody has seen whether the puck moves cleanly under such small pushes. The first run with the new rule gives this from steps.csv (measured tip move against commanded) and should be read before settings are compared.

7. The demonstrated path length contains the guiding hand's back-and-forth (the smoothed path is 0.79 of it; at the sheet-03 bend 7.8 cm of path for 5.8 cm of net move). Every "degrees per cm" comparison with the demonstrations and every share in the simulated walk is therefore about a fifth too kind to the robot. This is also why a length factor above 1.0 can look right on training-sheet answers.

8. A fixed shorter step (0.5 or 0.7 cm, one setting, no rule on the model's output, no question for the professor) is the cleanest test of the headline and has the highest share in the analyst's own walk (0.92 / 1.04), yet it is not in the session. As a diagnostic at the sheet-03 start only: on straight stretches it doubles the twist per cm, so it is not a candidate for the final driver.

9. The reading of run 4 for Emranul's second wish (puck in the middle of the line): the forearm had not under-turned there, the puck drifted to the outside of the arc while the model asked 0.65 to 0.9 cm and the robot drove 1.0 cm. Nothing in the answer says what would keep the puck centred on a gentle arc; it is left to the centring analysis.

### The analyst's corrections to the brief

1. 'Offline about half the demonstrated size': measured, the mean |predicted twist| is 0.63 to 0.69 of the mean |demonstrated twist| over all held-out sheets (about half only on sheets 05 and 06: 0.32 to 0.57; about equal on sheet 02: 0.93 to 1.04). The number that is 'about half' is another one: summed over 10 pictures the predictions give back 0.52 of the demonstrated turn. Sign right on steps of 2 degrees or more: 0.63 to 0.86 per sheet, 0.77 to 0.79 over all.

2. 'The robot's twist per cm was below the demonstrations' on every sheet' holds only for the sum of |twist| over whole runs (0.24 to 0.96). That measure counts twisting that is turned back and includes the wandering after a failure. By net turn the runs that reached the end matched the demonstrations (run 4: 77 against 76 degrees over the arc; net ratio 1.27 for run 4 and 1.03 for run 7), and run 8 turned more than the demonstrations before it failed (30 and 33 against 18 and 23 degrees per 5 cm).

3. The question assumes the demonstrator kept the camera looking along the path, with travel 'up in the picture'. The demonstrations do not show that: the travel direction is a median 46 degrees away from the picture's top and beyond 90 degrees in 24 % of the steps, because the tip moves round the puck. What the demonstrator kept is the puck straight ahead of the tip (within 20 degrees in 68 %, within 30 in 86 % of the pictures with the tip at the puck). model_angle_deg far from 0 is therefore not a failure sign by itself.

4. The brief gives the puck as 4 cm. config.py (PUCK_PARAMETERS diameter_m 0.024) and puck_detector.py say 2.4 cm across and 2.9 cm tall, and the tip-to-puck distance while pushing in runs 1 and 2 (median 2.1 and 1.9 cm) fits 2.4 cm, not 4 cm.

5. [scout] step lengths: with the audit cleaning (the rows the models train on) the median demonstrated step is 1.06 cm and the mean 0.94; 18.2 % of the steps are under 0.5 cm and 5.4 % under 0.2 cm (the brief: 20.5 % and 8 %). The median length of steps twisting 6 degrees or more is 0.65 cm, as stated.

6. [scout] twist per cm: confirmed within rounding. Exact values: sheet 03 runs 3.16 and 3.28 (model E) and 2.93 (team B, just below the scout's 3.1 to 3.2) against 4.64; sheet 04 1.38 against 5.65; sheet 05 2.78 against 3.39 (scout 3.44); sheet 06 2.60 against 4.42; sheet 02 2.45 and 2.01 against 2.54.

7. 'As in plan XX-01' was only roughly true for runs 4, 7 and 8: their tips started 2.7 to 2.9 cm from every tip position of the standard-start demonstrations of that sheet and end, and joint 4 started between -9.7 and +9.8 degrees in the ten runs, where every demonstrated standard, offset and turned start began at 0.0 (range -0.7 to +1.5). Comparisons of place are solid for runs 5, 9 and 10 (start within 0.2 to 2.1 cm, first heading gap +1.6 and +4.9 degrees for runs 9 and 10).

8. The models A, B and E used on the robot had been trained on all six sheets, so the ten runs are tests on training sheets. The fold models show that on an unseen sheet the answers at a bend are much less distinct (0.61 cm and 3.3 degrees instead of the 0.34 to 0.52 cm and 5 to 6 degrees asked at the sheet-03 bend on the robot).

### The analyst's open questions

1. Does the picture feedback hold the forearm once the twist per cm is right? Offline sums drift 41 degrees after 40 pictures and the simulation has no real feedback; only the robot session can tell whether the new step rule over-turns.

2. Why did model E ask for almost no twist at the first bend of sheet 04 (0.03 to 0.48 of the demonstrated twist per picture, then about 0), a sheet it was trained on? Candidates: the hand-placed start (puck straight ahead, joint 4 at +3 degrees, unlike the demonstrations' first pictures), the blue hoodie in view, or the 1-cm swings at pictures 4 to 7. One more sheet-04 run without the hoodie and with the new driver rule would separate them.

3. How much of the demonstrations' travel round the puck is real? By the puck tracker the puck centre is further than 2.8 cm from the tip in 56 % of the demonstration pictures and the tip travels 1.67 times as far as the puck; the tracker's table positions were not validated here. If it is real, a cleaner demonstration style (less moving round the puck) may be worth more than more episodes.

4. How do shorter steps near the goal act together with the stop rule? Within 3 cm of the goal 16 to 19 % of the fold models' answers ask for under 0.3 cm and under 1.5 degrees, so a driver that follows the asked length slows down or stands there; whether that helps or hurts the stop is for the stop analysis.

5. Should every run start with the forearm untwisted (joint 4 = 0), as every demonstrated standard start did, or stay as Emranul places it by hand (-9.7 to +9.8 degrees on 11.11)? Not tested.

6. Will the professor accept a driver that uses the model's own step length, a constant factor and a stand-still guard as part of 'imitation learning only'?

7. Can 're-aim or push' be told from the picture on an unseen sheet at all? The present answers reach 0.67 to 0.69; no probe was trained here because training was not allowed in this task.

## Unfamiliar pictures (folder `view_novelty`)

**Analyst's answer.** No: in none of the six failed runs did the model see unfamiliar pictures before its first wrong move (0 pictures above the demonstrations' own 95 % point in the fair yardstick); the pictures turned unfamiliar only after the puck had left the line, so light, hoodie, camera and stale pictures are ruled out and the failures are decision and execution problems. Two runs have a set-up share: run 3 used the square lay-out that only six mutually inconsistent demonstrations show, and in run 6 the view drifted to the edge of the demonstrated range over the steps before the corner, through the run's own moves.

**Verifier's overall verdict.** Partly. The first half stands and I reproduced it with my own code (all files in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/view_novelty/verify/): no failed run saw a picture above the demonstrations' own 95 % point before its first wrong move, model E answers on the robot pictures exactly as on its nearest training pictures, and light, hoodie, stale or late pictures and the camera's shift are not the cause. The second half is wrong because four of the six failure steps were misread: the puck left the line at step 18, 17 and about 30 to 37 in runs 5, 9 and 10 (not 5), and run 6 cut its first corner at step 10 after a wrong direction from step 8 (not 14 and 16), so in runs 5, 9 and 10 the pictures turned unfamiliar 7 to 9 steps BEFORE the puck was lost, as a result of the over-long swing, not of the set-up. "As close as the demonstrations are to each other" also overstates: the robot's pictures sit in the upper half of the demonstrations' range all day (fair percentile median 65, about 1.6 to 1.8 times the local spacing), in good and failed runs alike.

### Verdicts on the claims

1. **confirmed**: Headline, first half: before its first wrong move no failed run saw a picture above the demonstrations' own 95 % point (fair yardstick, 0 pictures in all six)

   Measured, own descriptors (forward hooks on the whole model), own neighbour code. With the analyst's steps I get the same fair percentiles, median/largest: run 3 72/72, run 5 75/82, run 6 91/95, run 8 54/71, run 9 66/73, run 10 82/91; 0 above 95 %. With my corrected steps (first wrong move at step 0, 2, 8, 7, 2, 2) still 0 above 95 % in the fair yardstick and in model E's own (run 6, steps 0 to 8: fair 73 to 92, model E 60 to 91). It also holds with the single nearest picture, with Euclidean distance and against demonstration pictures of the same part of an episode; only the very first picture of run 10 crosses 95 % in two of those variants (fair percentile 91). The analyst's descriptor cache equals mine (training pictures: difference 0; robot pictures: at most 0.004 on values up to 16). Files: own_novelty.out.txt, own_novelty_per_picture.csv.

2. **refuted**: Headline, second half: the pictures turned unfamiliar only after the puck had left the line

   True for runs 3, 6 and 8, false for runs 5, 9 and 10. Measured with my own puck-to-line gap (puck_on_line.py: red puck, dark line on the paper, gap of 0.5 cm or more on three pictures in a row) and checked on enlarged raw pictures: the puck first left the line at step 5 (run 3), 18 (run 5), 10 (run 6), 17 (run 9), 37 (run 10; by eye it cuts across the hump from about step 30); run 8 cannot be measured on dashes, by eye about step 20. First picture above the fair 95 % point: step 12, 9, 16, 22, 10, 11 (run 10 lasting from 25). So pictures above 95 % while the puck still touched the line: run 5: 9 of 15 (steps 9 to 17), run 9: 7 of 14 (steps 10 to 16), run 10: 6 of 34; runs 3, 6, 8: none. In runs 5, 9, 10 the unfamiliar pictures show the tip several centimetres to the right of a puck that still sits on the bend: a state made by the run's own over-long swing. Files: puck_on_line.out.txt, own_novelty.out.txt (tables 2 and 4), figures/own_run_05_steps_012_027.png, own_run_09_steps_012_027.png, own_run_10_steps_012_027.png, own_run_10_steps_028_042.png.

3. **corrected**: Failure steps read by eye, good to about one step (first wrong decision / puck off line): run 3 0/5, run 5 2/5, run 6 14/16, run 8 7/11, run 9 2/5, run 10 2/5

   Run 3: 0/5 is right (gap 0.3, 0.8, 1.8 cm at steps 4, 5, 6). Run 5: 2/18 (touching to step 17, then 1.3, 2.0, 2.8 cm). Run 9: 2/17. Run 10: 2/about 30 to 37; it got past the first bend, the pictures were familiar again at steps 15 to 21 (fair percentile 33 to 75) and it lost the line at the NEXT turn. Run 6: 8/10: the model asked +8, +1, -11, -19 degrees at steps 8 to 11 where its nearest demonstrations did +47, +40, +31, +31, and the puck cut the first corner at steps 10 to 12 (gap 1.1, 1.8, 1.3 cm) before landing on the next piece of line at step 13. Run 8: the asks match the nearest demonstrations within 1 to 9 degrees up to step 16 and the puck sits on the dashed branch at steps 9 to 15 as in the demonstration pictures; it leaves on the outside at about steps 17 to 21. The analyst's two 'corrections to the brief' for runs 6 ('the puck stayed on the line') and 8 ('off the dashed line by step 11') are therefore themselves wrong. Files: puck_on_line.out.txt, own_key_frames.out.txt, figures/own_run_06_steps_008_013.png, own_run_08_steps_005_013.png, own_neighbours_run_08_step_009.png.

4. **confirmed**: The pictures and descriptors analysed are exactly what the robot's model saw (454 pictures within 0.00001; same kind of JPEG)

   Measured: ModelPolicy.decide on the saved JPEGs against steps.csv, runs 2 to 9, 454 pictures: largest difference 0.000005 in forward cm, left cm, twist and stop chance (extract_own_model_e.out.txt). Model A on run 1: 26 pictures, direction equal to 0.0000 degrees (own_model_a_and_joints.out.txt). All 554 robot files and 741 sampled training files are 640 x 480 RGB with identical quantisation tables (own_file_format_check.out.txt).

5. **confirmed**: Runs 5, 9, 10: familiar pictures; model E asked what the nearest demonstrations did (a turn in place of a few millimetres with 6 to 8 degrees of twist); the robot did 1.00 cm each time

   Measured: run 5 step 3, the five nearest training pictures are sheet 03 (plans 03-02 row 6, 03-01 row 6, 03-02 row 7, 03-01 row 5, 03-06 row 27) and did 0.35 cm at -65, 0.64 at -113, 0.63 at -60, 0.16 at -124, 0.30 at -110 with twists -5.0 to -7.0, exactly as quoted; model E asked 0.47 cm at -54, twist -5.7; the robot moved 1.00 cm. New and stronger: model E's own answer on those five training pictures is 0.46 cm at -56, twist -6.3, the same as on the robot picture (step 4: 0.47 at -55 against 0.45 at -50), so the robot's view did not change the answer. Tip path while the forearm turned its first 30 degrees at this bend: demonstrations 2.2 to 4.7 cm (median 3.4, 8 episodes), robot 6.0, 6.0, 5.0 cm, asked by the model 2.6, 3.0, 2.3 cm. Two corrections: the model's direction is not the demonstrations' (labels' mean 0.37 cm at -89 at step 3 and 0.29 cm at -124 at step 4: sideways and backwards; the model adds a forward part), and run 10 did not fail at this bend (see the failure steps). Files: own_key_frames.out.txt, own_answers_on_neighbours.out.txt, own_travel_per_turn.out.txt, figures/own_neighbours_run_05_step_003.png.

6. **corrected**: Run 8: the most familiar pictures; model E asked what the demonstrations did; the puck slid off to the left over steps 7 to 11 and the model did not react

   Familiar: confirmed (fair percentile 25 to 71 at steps 0 to 8, none above 95 % before step 22). Asks like the nearest demonstrations: confirmed for steps 4 to 16 (direction within 1 to 9 degrees of the neighbours' mean, twist +5 to +6.6 against +5.0 to +6.4). Not confirmed: the slide at steps 7 to 11; at steps 9 and 14 the puck sits under the dash of the loop's left branch in the robot picture and in its five nearest demonstration pictures alike. It leaves on the outside at about steps 17 to 21, where model E asked +35 and +54 degrees and the neighbours did +10 and +26; on those same training pictures model E answers +36 where the labels' mean is +10, so its fit is off there even without the robot. Also measured: the robot's forearm was 10 to 20 degrees further twisted than the nearest demonstrations at every step from 4 to 15 (step 14: +60.6 against +37 to +43). Files: own_key_frames.out.txt, own_answers_on_neighbours.out.txt, figures/own_neighbours_run_08_step_009.png, own_neighbours_run_08_step_014.png, own_run_08_steps_016_023.png.

7. **confirmed**: Run 3 used the square portrait lay-out from end B, present in six episodes whose starts disagree; model E put the first picture next to mid-turn pictures of sheets 03 and 04

   Measured: plans 02-13 to 02-15 are end A in frames.csv and end B in model E's training table (the audit's corrected ends), 02-16 to 02-18 are B in both; plan 02-01 is A in frames.csv and B in the training table. Fold model without sheet 02: the first three pictures of run 3 are nearest to plan 02-14 rows 4 to 6 at 0.083 to 0.092, every other kind of start at 0.31 or more; run 4 is nearest to STD end B (02-31 at 0.054). Forearm twist over the first 5 cm: -44.2, -20.9, -7.2, -24.8, +33.0, -21.2 degrees (the analyst: -49.9 to +41.5, a slightly different window). Model E's five nearest of the first picture: 04-32, 03-09, 04-31, 03-09, 03-25 (0 % sheet 02). Added: at step 3 model E asked 0.42 cm at -18 and answers 0.41 cm at -18 on its five nearest training pictures (plans 02-17 and 02-18), whose labels are 1.03 cm at +48: it has not fitted these episodes. Tip path for the first 30 degrees of turn: SQP demonstrations 2.5 to 7.7 cm (median 3.9), robot 8.0, asked 3.3. The local ratio 4.56 was not recomputed. Files: own_setup_checks.out.txt (part F), own_answers_on_neighbours.out.txt, figures/own_run_03_against_sqp_starts.png.

8. **corrected**: Run 6 is the one run where the pictures left the demonstrations' range BEFORE the visible failure, through the run's own moves (above 95 % at steps 12 to 14, first wrong decision at 14)

   The failure was visible earlier: wrong direction from step 8, corner cut at steps 10 to 12 (measured gap up to 1.8 cm), so the pictures above 95 % at steps 12 to 14 (model E 95, 97, 98) came after it. Before step 8 nothing is above 95 %. What is true: run 6 had the least familiar start of the day from its first picture (fair percentile 92, 92, 89 among the sheet's pictures; 86, 86, 77 among its starts; start 5.0 cm from the demonstrated one), so this is not a drift through the run's own moves. The cause is in the model: on the five nearest training pictures of step 8 (plans 04-08, 04-09, 04-14, 04-15, rows 9 to 14, labels 1.1 to 1.3 cm at +16 to +58, mean +47) model E itself answers 0.58 cm at 0 degrees, and at steps 9, 10, 12 it answers -18, -47, -34 where the labels' means are +40, +31, +41. The demonstrated steps there are full steps, so the 1-cm rule is not what failed here. Files: own_key_frames.out.txt, own_answers_on_neighbours.out.txt, figures/own_neighbours_run_06_step_008.png.

9. **confirmed**: The blue hoodie did not cause the failures of runs 5 and 6

   Measured with my own detector (large royal-blue region at the picture's border): run 6: first in view at step 13 (0.9 % of the picture, then up to 13.3 %), that is after the wrong direction (step 8) and the corner cut (step 10); painting it over changes model E's answer at steps 13 to 18 by at most 9.4 degrees and 0.10 cm (the analyst: 9.2 degrees, 0.08 cm). Run 5: nothing large and blue before step 16, and that is the robot's own base at the bottom left (the same in run 9). Run 4: hoodie and a hand in view from step 19 to 41, up to 18 % of the picture, and it reached the end. One caution: the stop chance at steps 40 and 41 of run 4 is 0.90 and 0.88 as seen and 0.22 and 0.37 with my paint-over (the analyst: 0.65 and 0.66), so the stop at the end of run 4 may have leaned on the hoodie. Training pictures (every 12th): 0.6 % show more than 2.6 % of such blue, at most 6.3 % (the robot's own base). Files: own_setup_checks.out.txt (part D), own_model_a_and_joints.out.txt (part C), figures/own_hoodie_first_views.png.

10. **confirmed**: Run 1 did not start with the puck 1.5 cm left of the line; all ten runs started 1.3 to 5.1 cm from the demonstrations' start with -9.7 to +9.8 degrees of forearm twist

   Measured from the logs: tip y at step 0 is +0.09 cm (run 1) and +0.11 cm (run 2) against -1.49 cm in all 71 standard-start demonstrations (row 0: x 14.00 to 14.04 cm, y -1.49 to -1.48, twist within 0.09 degrees). By eye the puck is centred on the line in the first pictures of both runs and right of the line at step 22 of run 1. Distances from the demonstrated start: 1.6, 1.8, 2.5, 3.8, 3.1, 5.0, 4.0, 3.2, 1.3, 3.3 cm; twist -9.7 to +9.8 degrees. Added: the offset is one-sided, nine of ten runs started 1.3 to 3.7 cm to the LEFT, and the picture's heading at step 0 was -4.4 to +12.3 degrees against exactly -6.1 in every start-mark demonstration. The puck measure in the picture (0.00, -0.05, -0.05 cm) was not recomputed. Files: own_setup_checks.out.txt (part B), figures/own_runs_01_02_starts.png.

11. **confirmed**: Light, colour, sharpness and file format of the robot pictures were inside the training pictures' range

   Brightness measured with one formula for both (mean of the 160 x 120 picture): training median 144.7 (1 % to 99 %: 106.7 to 162.6); run medians 139.7 to 163.4; run 2, the cleanest run, is the brightest (163.4, 99th percentile); 60 of 554 robot pictures lie outside the training pictures' 1 % to 99 % (the analyst counts 33 outside the middle 99 %), most of them in runs 2, 3 and 7, two of which reached the end. Colour and sharpness were not recomputed. Not covered by either of us in numbers: shadows; the sheet-04 demonstrations nearest to run 6 show a hard shadow of the arm that the robot pictures do not have (figures/own_neighbours_run_06_step_008.png). File: own_setup_checks.out.txt (part C).

12. **confirmed**: The camera has crept on its tape mount: the robot's own hand sits about 16 px (of 640) further right than when sheets 01 to 03 were recorded

   Measured with my own method (template of the arm from the robot day, matched in demonstration pictures with wrist and hand within 0.5 degrees of the held values): the arm sits 17 px further left on 6.11 (sheet 01), 18 px on 7.11 (sheet 02), 18 px in sheet 03 (8.11); sheet 04 is mixed (median 14.5 px, 56 % of the sampled pictures more than 8 px); sheet 05: 1 px; sheet 06 (9.11): 2 px. Self-check on the robot runs: -1 to +2 px. So it is one move on 8.11 during sheet 04 plus 2 to 3 px since, more a step than a steady creep. Files: own_camera_creep.out.txt, figures/own_camera_creep_examples.png.

13. **confirmed**: The creep biases the steering slightly to the right but did not cause any of the six failures (first wrong decisions stay as they were; up to about half of run 1's drift)

   Measured by sliding every robot picture 17 px to the left: the asked direction turns by a median of +1.3 to +3.6 degrees in runs 2 and 4 to 9 (run 3: -4.8); model A on run 1: +2.8 degrees (middle half +1.8 to +3.4; the analyst: +2.5). The first wrong decisions keep their character: run 5 step 3 0.47 cm at -54 becomes 0.42 at -50, run 9 step 3 0.49 at -31 becomes 0.46 at -35, run 6 step 8 +8 becomes +10, run 3 step 0 -28 becomes -31; the largest single change is -15.9 degrees at run 5 step 2. No pattern by sheet either: sheets 01 and 02 (earlier view) reached the end, sheet 06 (later view) failed. The 1.06 cm over 22 cm for run 1 is a calculation that assumes nothing corrects the bias. Files: own_setup_checks.out.txt (part E), own_model_a_and_joints.out.txt (part A).

14. **confirmed**: No stale and no late pictures in 544 pairs

   Measured: 544 pairs, 0 byte-identical, 0 identical after decoding; smallest mean change 2.24 grey levels (run 4, step 14); the tip moved 0.94 to 1.05 cm between pictures. Lag, own test without the camera model: in the 37 pairs where the tip's direction changed by 60 degrees or more from the step before, the scene's shift under the tip fits the move of the same step in 35 (median direction error 11 degrees against 114 degrees for 'one step late'). Small correction: the times between pictures range from 0.66 to 1.50 s; the analyst's 0.86 to 1.07 s are not the extremes. Files: own_setup_checks.out.txt (part A), own_lag_check.out.txt.

15. **corrected**: The held wrist and hand did not put the camera anywhere the demonstrations had not had it; the tip falls at pixel (313, 372) in every robot picture

   Joint ranges, measured: forearm twist -80.9 to +101.5 degrees on the robot against -105.2 to +114.0 (1 % to 99 %) in training; 54.9 % of the training pictures have wrist and hand within 3 degrees of the held values; shoulder outside the training pictures' 1 % to 99 % in 25 of 554 robot pictures, elbow in 5, reach in 7. The camera heights and tilts from the fitted camera model were not recomputed. Wrong as stated: pixel (313, 372) is what the old camera model predicts; since the camera moved, the rod and block sit about 17 px further right on the robot day (near column 330, figures/own_mean_robot_picture.png), about 1.2 cm at the tip's distance, which affects anything measured against column 313. File: own_model_a_and_joints.out.txt (part B).

16. **corrected**: Before the first wrong decision the robot's pictures were as close to the demonstrations as the demonstrations are to each other; unfamiliar pictures alone did not break a run

   Second part confirmed: above the fair 95 % point are 3 of 42 pictures of run 4, 12 of 89 of run 7, none of runs 1 and 2. First part too strong: over the 209 pictures of the four good runs plus the pictures before each first wrong move, the fair percentile has a median of 65, not 50 (71 % lie above the demonstrations' median), and a robot picture is 1.63 times (fair) or 1.79 times (model E) as far from its neighbours as those are from theirs; in the six failed runs the 'before' medians are 72, 75, 84, 54, 66, 82. Right wording: inside the demonstrations' range but in its upper half all day, in good and failed runs alike. File: own_novelty.out.txt (table 6).

17. **confirmed**: Model E asks for shorter steps than were demonstrated even on its own training pictures (median 0.75 cm against 1.06 cm)

   Measured on the 17,773 moving rows it trained on: label median 1.06 cm, answer 0.75 cm; median absolute difference 0.28 cm forward, 0.27 cm left, 1.66 degrees twist. Added, and it matters for the driver: on turn-in-place rows (label under 0.5 cm with 4 degrees of twist or more, 17 % of rows) the label median is 0.28 cm and the answer 0.51 cm, with the direction more than 45 degrees off the label in 44 % of them; over all moving rows the direction is more than 45 degrees off in 23 % (straight pushes 9 %, sideways pushes 26 %), on pictures the model was trained on. Files: own_key_frames.out.txt (top), own_fit_on_training_pictures.out.txt.

18. **confirmed**: Side observations: sheet 01 offset labels swapped (plans 01-17, 01-18); a stop rule 'distance above 0.18 on 4 pictures in a row' separates failed from good runs; 210 episode ids, model E trained on 18,032 pictures from 209; on a first stop picture the robot still moves 1 cm

   Plan 01-17 (labelled OL1.5) shows the puck right of the line and 01-18 (labelled OR1.5) left of it (figures/own_sheet_01_offset_labels.png). The stop rule fires at step 10, 19, 16, 34, 30, 39 in runs 3, 5, 6, 8, 9, 10 and never in runs 1, 2, 4, 7 (run 4 reaches 0.265 but not four in a row); against my steps that is 5, 1, 6, 14, 13, 2 steps after the puck left the line; it is tuned on these ten runs (own_misc_checks.out.txt). 18,522 pictures, 210 ids, 18,032 trained on from 209: measured (own_novelty.out.txt). The stop rule's extra centimetre: read from ModelRun.run and StopRule.update in D:/AI_Project/.claude/worktrees/training-analysis/02_code/model_mode.py. Not verified: 'puck 0.87 cm right of the tip's line of travel' (it may lean on the outdated tip pixel).

### The analyst's findings with their evidence

1. (measured, confidence high) Before the first wrong decision the robot's pictures were as close to the demonstrations as the demonstrations are to each other, in every failed run; they became unfamiliar only after the puck left the line.

   Evidence: Descriptor = the 3 x 4 grid-pooled backbone features (6,144 numbers, what the prediction head sees); distance = mean cosine distance to the 5 nearest demonstration pictures; yardstick = the same distance for every demonstration picture of that sheet against other episodes. Fair yardstick (fold model of model E's recipe that never saw the sheet), pictures up to the first wrong decision, percentile median / largest: run 3 72/72 (1 picture), run 5 75/82, run 6 91/95, run 8 54/71, run 9 66/73, run 10 82/91; pictures above the 95 % point: 0 in all six. In model E's own descriptors (18,032 training pictures as reference): medians 76, 80, 89, 39, 70, 83; above 95 %: 0 except run 6 (3 of 15, steps 12 to 14). After the puck left the line: fair median 87 to 96 with 15 to 58 % of pictures above the 95 % point; model E 84 to 99 with 27 to 92 %. Mean-pooled (512) and hidden-layer (256) descriptors give the same picture. Files: novelty_fold_models.out.txt, novelty_model_e.out.txt, per_run_summary.out.txt, figures/novelty_fold_models_along_runs.png, figures/novelty_model_e_along_runs.png. The failure steps (first wrong decision / puck off line: run 3 0/5, run 5 2/5, run 6 14/16, run 8 7/11, run 9 2/5, run 10 2/5) were read by eye off figures/failure_strip_run_NN.png, good to about one step.

2. (measured, confidence high) The pictures and descriptors analysed are exactly what the robot's model saw.

   Evidence: Model E's answers recomputed here from the saved JPEGs match steps.csv of runs 2 to 9 within 0.00001 (cm, degrees, stop chance) on all 454 pictures (extract_features_model_e.out.txt). The robot JPEGs are the same kind of file as the training JPEGs: 640 x 480, identical quantisation tables (setup_light_and_camera.out.txt).

3. (measured, confidence high) Runs 5, 9 and 10 (sheet 03, first bend): familiar pictures, and the model asked for what the nearest demonstrations did there, a turn in place with a few millimetres of sideways tip travel; the robot carried it out as 1-cm pushes. A decision and execution problem, not a view problem.

   Evidence: Pictures of steps 3 and 4: 100 % of the 5 nearest demonstration pictures are sheet 03 at the same place (plans 03-01, 03-02, 03-04, 03-05, 03-06, 03-09 and others, rows 4 to 8). Their demonstrated steps (medians) 0.36, 0.46, 0.61 cm with forearm twists of 6.2, 6.7, 7.6 degrees; model E asked 0.46, 0.47, 0.47 cm with 5.8, 6.1, 6.2 degrees; the robot moved 1.00 cm each time. Example run 5 step 3: the five demonstrations did 0.35 cm at -65 degrees, 0.64 at -113, 0.63 at -60, 0.16 at -124, 0.30 at -110 with twists -5.0 to -7.0; model E asked 0.47 cm at -54 with twist -5.7. Start pictures were ordinary: fair percentile among the demonstrations' start pictures 78/68/59 (run 5), 64/59/52 (run 9), 91/78/58 (run 10). Files: key_frames.out.txt, per_run_summary.out.txt, figures/key_frames_run_05.png, key_frames_run_09.png, key_frames_run_10.png. That the 1-cm rule is what took the puck off is my reading of asked 0.47 against done 1.00; the step-length analysts test it.

4. (measured, confidence medium) Run 8 (sheet 06): the most familiar pictures of all failed runs, and model E asked what the demonstrations did on the line; the puck slid off to the left over steps 7 to 11 and the model did not react to it.

   Evidence: Steps 0 to 10: all 5 nearest demonstration pictures are sheet 06 end A (plans 06-01 to 06-10, rows 2 to 13) at distances 0.020 to 0.067; model E's direction is within 1 to 9 degrees of the neighbours' mean direction, its step 0.82 to 1.24 cm against their 1.01 to 1.27 cm, its twist +2.5 to +6.6 against +2.5 to +5.4 degrees (key_frames.out.txt, figures/key_frames_run_08.png). Percentile before the first wrong decision: 39 (model E), 54 (fair). In the montage the robot's puck at steps 9 and 10 sits left of the dashed branch while in the nearest demonstration pictures it sits on it.

5. (measured, confidence high) Run 3 (sheet 02 lying straight) used the square portrait lay-out from end B. It IS in the data, but only in six episodes whose starts disagree, and model E placed the first picture next to mid-turn pictures of other sheets. Set-up weakness (thin, inconsistent coverage) answered with the turn-in-place pattern, which the driver stretched to 1-cm pushes.

   Evidence: In the descriptors of the fold model that never saw sheet 02, the first three pictures of run 3 are nearest to the SQP end B starts (plan 02-14 at 0.090; every other kind of start 0.32 or more); run 4 is nearest to STD end B (plan 02-31, 0.078). After the audit's corrected ends all six SQP episodes (02-13 to 02-18) are end B. Their forearm twist over the first 5 cm of tip travel: -49.9, -22.6, -9.7, -31.9, +41.5, -21.0 degrees (STD episodes: -11.3 to +13.0). In model E's descriptors the 5 nearest pictures of run 3's first picture are from sheets 03 and 04 (0 % sheet 02; plans 04-32, 03-09, 04-31, 03-25 at joint 4 of -19 to -41 degrees, demonstrated steps 0.17 to 0.89 cm with twists -5.5 to -7.9); its distance is unremarkable (percentile 76) but its local ratio is 4.56 (the yardstick's own 99 % point is 3.71). Model E asked 0.36, 0.41, 0.49, 0.42, 0.38 cm with twists -5.3 to -3.0 in the first five steps; each was done as 1.00 cm. Files: sheet02_square_check.out.txt, figures/sheet02_square_starts.png, key_frames.out.txt, figures/key_frames_run_03.png.

6. (measured, confidence medium) Run 6 (sheet 04) is the one run where the pictures left the demonstrations' range BEFORE the visible failure, but through the run's own moves, not through the set-up: the robot kept pushing straight where the demonstrators had already swung the tip round the puck and turned the forearm.

   Evidence: Distance (model E) rose from 0.042 at step 2 to 0.196 at step 14; percentile 89 or more from step 7, above the 95 % point at steps 12 to 14; fair yardstick median 91, largest 95 before the first wrong decision. At steps 8 to 13 the nearest demonstration pictures (plans 04-08, 04-09, 04-14, 04-15, rows 9 to 15) have the forearm at +23 to +50 degrees (robot +15 to +17) and tip steps of 1.1 to 1.3 cm at +50 to +58 degrees (four of the five nearest at step 8), where model E asked 0.80 cm at +8, 0.71 at -19, 0.53 at -18. At the corner (steps 14 and 15) the neighbours twisted 5.2 and 6.6 degrees per step, model E asked 1.3 and 1.8. The start was the most displaced of the day (5.1 cm from the demonstrations' start; first three pictures at percentile 86, 86, 77 of the demonstrations' starts in the fair yardstick). Files: novelty_model_e_per_picture.csv, key_frames.out.txt, figures/key_frames_run_06.png, figures/failure_strip_run_06.png.

7. (measured, confidence high) The blue hoodie did not cause the failures of runs 5 and 6.

   Evidence: Hoodie = strongly saturated royal blue (hue 113 or more) covering more than 0.5 % of the 160 x 120 picture outside the robot's own area. Run 5: in view in 2 pictures only (steps 23 and 24, 0.7 %), none before the first wrong decision at step 2; runs 5 and 9 asked the same pattern at steps 2 to 4 (0.58 at -22, 0.47 at -54, 0.45 at -50 against 0.52 at -31, 0.49 at -31, 0.46 at -38). The bright blue at the bottom left of runs 5 and 9 is the robot's own blue (in view in 4.8 % of the training pictures). Run 6: first in view at step 14 (0.9 %), the step of the first wrong direction, after the distance had already climbed to the 99th percentile; with the whole hoodie painted out, model E's answers at steps 14 to 18 change by at most 9.2 degrees and 0.08 cm, and over all 55 hoodie pictures the asked length stays 0.23 cm and the distance 0.21. Run 4, which reached the end, had the hoodie in 22 pictures (steps 20 to 41, up to 15.1 % of the picture, plus a hand); its stop chance at steps 40 and 41 is 0.90 and 0.88 with and 0.65 and 0.66 without the hoodie. Pasting run 6's hoodie (7.5 % of the picture) into 596 training pictures turns the asked direction by a median of 2.2 degrees (90 % point 7.9); run 4's larger one (18.9 %) by 5.2 degrees (16.3), shortens the step from 0.75 to 0.67 cm and raises the stop chance by up to 0.14. Royal blue is in view in 0.53 % of the training pictures, never above 2.6 % of a picture. Files: hoodie_check.out.txt, hoodie_check_per_picture.csv, figures/hoodie_painted_out.png, figures/hoodie_examples.png. Limit: the painted patch is flat table colour, not a real hoodie-free picture.

8. (measured, confidence high) Run 1 did not start with the puck 1.5 cm left of the line: the 1.5 cm is the arm's offset from the demonstrations' start position, which every run had. In the picture the puck was centred.

   Evidence: Puck beside line measured in the picture (puck width as the 4 cm scale): run 1 steps 0 to 2: 0.00, -0.05, -0.05 cm; run 2: -0.09, -0.15, -0.12 cm (zero of the measure uncertain by about 0.5 cm: STD demonstrations read +0.33, range -0.65 to +0.57). Tip y at step 0: +0.09 cm (run 1) and +0.12 cm (run 2) against -1.49 cm in every standard-start demonstration. Run 1's puck then drifted right: -0.74 cm median over steps 10 to 19, -1.8 cm at step 22 (matches what Emranul saw); run 2 stayed within 0.25 cm for 17 steps. All ten runs started 1.3 to 5.1 cm from the demonstrations' start (0.1 to 3.7 cm to the left, -1.8 to +4.2 cm forward) with the forearm twisted -9.7 to +9.8 degrees, where every standard-start demonstration began at exactly x 14.0 cm, y -1.5 cm, twist 0. The two runs with 3.9 and 4.0 cm offsets (4 and 7) reached the end. Files: run1_start_offset.out.txt, setup_light_and_camera.out.txt (section WHERE EACH RUN STARTED).

9. (measured, confidence high) Light, colour, sharpness and file format of the robot pictures were inside the training pictures' range.

   Evidence: Brightness (grey 0 to 255), run medians 138.0 to 163.0; training median 143.8, 5 % to 95 %: 119.2 to 157.6. Run 2, the cleanest run, was the brightest (163.0, above 99.5 % of training pictures); 33 of 554 robot pictures lie outside the training pictures' middle 99 %. Paper colour red over blue 0.71 to 0.78 (training median 0.78, 5 % to 95 %: 0.67 to 0.91). Sharpness (variance of the Laplacian) run medians 234 to 328, training median 213: no blur. Measured with the training index's own function (training_dataset.picture_statistics). Files: setup_light_and_camera.out.txt, figures/setup_light_and_colour.png.

10. (measured, confidence high) The camera has crept on its tape mount: on the robot day the robot's own hand sat about 16 px (of 640) further right in the picture than when sheets 01 to 03 were recorded.

   Evidence: Shift of the picture's bottom strip against the recording tool's reference of 6.11.26, by the recording tool's own phase-correlation check (record_episodes.camera_shift_px), first pictures of standard-start demonstrations, median x: 6.11: +0.2, 7.11: -0.1, 8.11: +2.9 (range -2.9 to +16.8, the day the audit says the camera moved), 9.11: +13.6 (13.3 to 14.6); robot day: +16.3 over all 554 pictures (run medians 15.2 to 17.7). Training pictures in the earlier view: sheets 01, 02, 03 entirely (644, 2,670, 3,955 pictures), sheet 04 72 %; sheets 05 and 06 entirely in the later view. The y shift is noisy (run medians -11.9 to +1.9). Files: setup_light_and_camera.out.txt, figures/setup_camera_strips.png, per_run_summary.out.txt.

11. (calculated, confidence medium) The creep is visible to the model and biases the steering slightly to the right, but it did not cause any of the six failures.

   Evidence: Sliding the robot pictures 16 px to the left before model E sees them (undoing the creep) brings them closer to the training pictures on the sheets recorded in the earlier view: run 1 0.0407 to 0.0261, run 2 0.0230 to 0.0179, run 4 0.0531 to 0.0457, run 10 0.0663 to 0.0510; for sheet 06 (later view) the distance is smallest with no slide (run 8). The asked direction turns by a median of +1.0 to +1.7 degrees to the left (model E, runs 1, 2, 4, 7) and +2.5 degrees for model A on run 1, the model that drove it (middle half +1.8 to +3.8). The first wrong decisions stay as they were (run 5 step 3: 0.47 cm at -54 as seen, 0.43 at -49 with the creep undone; run 9 step 3: 0.49 at -31 against 0.46 at -34). Calculated: a 2.5 degree bias over run 1's 22 cm is 0.96 cm to the right; the puck was measured 1.8 cm right at step 22, so the creep may explain up to about half of run 1's drift if nothing corrected it. File: camera_shift_counterfactual.out.txt. Limit: a sideways slide imitates a turned camera only approximately.

12. (measured, confidence high) No stale and no late pictures: the two-new-pictures rule of the driver works.

   Evidence: 544 pairs of consecutive pictures: 0 byte-identical; smallest change 1.70 grey levels (run 4 step 14) against 0.60 (median) and 1.01 (75 % point) for demonstration pairs with the arm standing still; the tip moved 0.94 to 1.05 cm between pictures; 0.86 to 1.07 s between pictures; decision time 16 to 18 ms (56 ms for the team of five). Lag test with table-fixed content redrawn through the camera model: assuming every picture one step late makes consecutive pictures fit worse (median misfit 4.83 against 3.37 in step) and fits best in 7 % of the 319 pairs that can tell (demonstrations: 2 %); in step or a quarter step late fits best in 72 % (demonstrations 87 %). The method cannot tell zero from a quarter step, in the demonstrations either. Files: stale_pictures.out.txt, picture_lag_check.out.txt.

13. (calculated, confidence high) The held wrist and hand did not put the camera anywhere the demonstrations had not had it.

   Evidence: From the joint angles through the fitted camera model: camera height 19.6 to 22.9 cm above the table (training pictures 1 % to 99 %: 18.4 to 23.6), forearm twist -80.9 to +101.5 degrees (training -105.2 to +114.0), picture roll -29.0 to +25.4 (training -33.7 to +35.7); tilt outside the training range in 25 of 554 pictures, 1 of them before a first wrong decision. The tip falls at pixel (313, 372) in every robot picture, the demonstrations' median. 55.1 % of the training pictures have wrist and hand within 3 degrees of the held 13.4 and -6.1. File: camera_pose_check.out.txt.

14. (calculated, confidence medium) Unfamiliar pictures alone did not break a run: the runs that reached the end met them too.

   Evidence: Pictures above the demonstrations' 95 % point: run 4: 5 of 42 (model E), 3 of 42 (fair), with hoodie and hand in view and a largest percentile of 99 to 100; run 7: 10 of 89 and 12 of 89; run 1: 1 of 26 and 0; run 2: none. Across runs 2, 4, 7 the robot pictures sit a median of 1.72 times farther from the demonstrations than those are from each other locally (local ratio; the yardstick's own median is 1.18). Scale for comparison: a whole unseen sheet sits 2.3 to 3.9 times farther from a fold model's training pictures than those are from each other (the situation of the exam, where the fold models still scored about 0.8 path within 45 degrees offline); the robot pictures seen before each failure were 0.6 to 2.7 times model E's own median. Files: per_run_summary.out.txt, novelty_model_e.out.txt, novelty_fold_models.out.txt (last table).

15. (measured, confidence high) Model E asks for shorter steps than were demonstrated even on its own training pictures.

   Evidence: On the 18,032 training pictures (moving rows): label step median 1.06 cm, model E's answer median 0.75 cm; median absolute difference 0.28 cm forward, 0.27 cm left, 1.66 degrees twist (key_frames.out.txt, first lines). Confirms the brief's scout number of 1.05 cm for the demonstrated median.

16. (measured, confidence medium) Side observations for the other workstreams.

   Evidence: (a) Sheet 01 offset labels look swapped: plan 01-17 labelled OL1.5 shows the puck 2.5 cm RIGHT of the line (measured -2.51 cm), plan 01-18 labelled OR1.5 shows it far left (by eye); OL1 reads +0.79 and OR1 -0.60 as planned (run1_start_offset.out.txt). (b) In the demonstrations' first pictures the block stands a few centimetres behind the puck (approach rows); the robot runs start touching the puck and their first pictures match demonstration rows 2 to 6 (figures/key_frames_starts.png, figures/sheet02_square_starts.png). (c) While pushing, the puck sat a median 0.87 cm right of the tip's line of travel on the robot against 0.24 cm right in the 9.11 demonstrations; run 1: 1.00 cm right, run 2: 0.37 (puck_beside_tip.out.txt). (d) A stop rule 'distance above 0.18 on 4 pictures in a row' would have stopped all six failed runs 0 to 34 steps after the puck left the line and none of the four that reached the end; tuned on these ten runs only. (e) frames.csv holds 210 episode ids; model E trained on 18,032 pictures from 209.

### The analyst's recommendations

1. In the driver, do not turn a 'turn in place' answer into a 1-cm push: when the model asks for a short step together with a forearm twist, travel the asked length (or stand still) and twist. Check the change first on the logged pictures of runs 3, 5, 9 and 10, then on the robot on sheet 03 and on sheet 02 lying square.

   Why: In runs 3, 5, 9 and 10 the pictures before the failure were familiar and model E asked what the nearest demonstrations did: steps of 0.36 to 0.61 cm with 6 to 8 degrees of twist (run 3: 0.36 to 0.49 cm). The robot did 1.00 cm each time, two to three times the demonstrated travel, in the sideways direction of a by-product motion.

   Needs: code only, then a robot test. Effort: half a day of code and replay checks, one robot session of 2 to 3 hours

   Expected effect: Removes the first wrong moves of four of the six failed runs. Whether the bend is then followed to its end is not shown by my data.

   Risk: Slower runs. Where the model asks for almost nothing without a twist (run 6 after step 19: 0.03 to 0.36 cm, twist under 2 degrees) the arm would stall, so a 'no progress' stop is needed.

2. Start every robot run from the demonstrations' start pose: let the driver move the arm to tip x 14.0 cm, y -1.5 cm, forearm twist 0, wrist 13.5, hand -6.1 degrees and have the person lay the sheet and puck under it, as the recording tool did; at the least print the offset and refuse beyond about 1 cm or 3 degrees.

   Why: Every run started 1.3 to 5.1 cm from that pose with up to 9.8 degrees of forearm twist, while every standard-start demonstration began exactly there. Run 6, the most displaced (5.1 cm), had the least familiar start (percentile 86 of the demonstrations' starts) and is the run whose view drifted out of range.

   Needs: code only. Effort: 1 to 2 hours

   Expected effect: Robot tests become comparable with each other and with the demonstrations; one uncontrolled variable less when the driver changes are judged.

   Risk: If the exam does not let the team choose the start, robustness to start offsets must come from data instead (see the demonstrations recommendation). Runs 4 and 7 show 4 cm is survivable.

3. Watch and neutralise the camera's creep: (a) run the recording tool's bottom-strip check at the start of every recording and every robot run and write the shift into summary.json; pause when it is more than about 8 px from the data the model was trained on; (b) at the next retraining, either widen the sideways shift augmentation from 4 to 8 px (at 160 px wide) or shift every picture so the robot's own hand sits at a fixed column, in training and on the robot.

   Why: The hand sits 16.3 px right of the 6.11 reference (13.6 px on the last recording day, 0 to 3 px when sheets 01 to 03 were recorded). Undoing it brings the pictures of sheets 01 to 03 14 to 36 % closer to the training pictures and turns the asked direction 1.0 to 2.5 degrees to the left, which is up to about 1 cm sideways over a 24 cm line.

   Needs: code only for (a); retraining for (b). Effort: (a) about 1 hour; (b) half a day plus one training run on ki-014

   Expected effect: Removes a small constant steering bias and so helps Emranul's 'puck in the middle of the line'; protects against the day the tape moves further.

   Risk: The strip check is disturbed by what lies behind the hand (5 % to 95 % range up to 80 px in runs 4, 5, 8), so use the median over several pictures at the start pose. A wider shift augmentation may cost a little centring precision. The camera itself must not be touched outside the camera-mount checklist.

4. Record demonstrations where the data are thin, with one agreed routine each: (a) the square lay-out for every sheet and both ends, if the exam uses it; (b) recoveries: the puck 1 to 2 cm beside the line at several places along each sheet, not only at the start; (c) starts spread by a few centimetres and up to 10 degrees of forearm twist.

   Why: (a) Sheet 02 has six square episodes, all from end B, whose first 5 cm disagree (forearm twist -50 to +42 degrees); no other sheet has any; run 3 failed on exactly this lay-out. (b) After the puck left the line 15 to 92 % of the pictures were outside the demonstrations' range and in run 8 the model kept asking the on-line move while the puck slid off. (c) The robot day's starts differed from the single demonstrated start in every run.

   Needs: new demonstrations, retraining, and the professor's answer on how the sheet lies at the exam. Effort: two robot sessions of 2 to 3 hours for about 60 to 80 short episodes, one retraining

   Expected effect: Makes the exam lay-out and the way back to the line demonstrated situations. This is the only lever my data show for the off-line states.

   Risk: Inconsistent demonstrations make it worse (the square episodes show this): agree and write down the routine first. Recoveries are awkward to guide by hand.

5. Decide how sharp corners are to be demonstrated and executed: first check offline whether model E's five-step chunk reproduces the swing round the puck that the demonstrators made at the first corner of sheet 04; if not, re-record the sharp corners of sheets 03 and 04 with one simple, repeatable routine.

   Why: In run 6 the nearest demonstrations at steps 8 to 13 moved the tip 1.1 to 1.3 cm at +50 to +58 degrees with the forearm at +23 to +50 degrees, a manoeuvre spread over several steps; model E asked 0.5 to 0.8 cm at +8 to -19 degrees and 1 to 2 degrees of twist where they twisted 5 to 7, and the view then drifted out of the demonstrated range.

   Needs: code only for the check; new demonstrations and retraining if it fails. Effort: 2 to 3 hours for the offline check; one robot session if re-recording

   Expected effect: Addresses the one failure that the step-length change does not obviously cover.

   Risk: A one-picture model may not be able to hold a multi-step manoeuvre at all; that would point to executing chunks or adding memory, a larger change.

6. Log each picture's familiarity in the driver: the distance of its descriptor to the training pictures' descriptors (a table of 18,032 x 256 numbers), with the demonstrations' 95 % point beside it in steps.csv. Leave an automatic 'lost the line' stop switched off until more runs exist.

   Why: On this day the number separated 'before' from 'after' in every failed run (0 pictures above the fair 95 % point before, 15 to 58 % after). Every future robot run would then say by itself whether it failed on familiar pictures, which the plan's many driver options need.

   Needs: code only. Effort: 2 to 3 hours

   Expected effect: Faster diagnosis of every later robot test; later perhaps a guard against wandering to the table's edge (runs 5, 6, 9, 10 wandered 53 to 85 pictures after losing the line).

   Risk: A stop rule tuned on ten runs can stop a good run: a looser rule (0.152 on 5 pictures in a row) would have stopped run 4 at step 29.

7. Keep people, clothing and hands out of the camera's view during robot runs, and add pasted coloured patches outside the paper to the training augmentation at the next retraining.

   Why: The hoodie was not the cause of any failure, but it is not free: it moved model E's direction by a median of 2 to 8 degrees and the stop chance by up to 0.25 (run 4), and training pictures never show more than 2.6 % royal blue.

   Needs: nothing for the first part; retraining for the second. Effort: none; 1 to 2 hours of code with the next retraining

   Expected effect: Small: steadier stop decision and direction when something enters the view at the exam.

   Risk: Low. Do not spend a robot session on it.

8. Leave the picture taking, the light and the held wrist as they are.

   Why: No identical or late pictures in 544 pairs, brightness and colour inside the training range (the brightest run was the cleanest), camera heights, tilts and rolls inside the demonstrated ranges.

   Needs: nothing. Effort: none

   Expected effect: Saves robot time for the changes that matter.

   Risk: None seen; the lag test cannot rule out a delay under about half a step.

### The verifier's challenges to the recommendations

1. Recommendation 1 (do not stretch a turn in place to 1 cm; travel the asked length): the direction is right for the first deviation of runs 3, 5, 9 and 10, but the expected effect ('removes the first wrong moves of four of the six failed runs') is overstated. (a) On its own training pictures model E asks 0.51 cm where a turn in place was demonstrated with 0.28 cm, and its direction there is more than 45 degrees off in 44 % of rows, with a forward part the demonstrations did not have (run 5 step 4: asked 0.45 cm at -50, demonstrated 0.29 cm at -124); 'stand still and twist' for short asks with a twist is at least as close to the demonstrations and should be one of the tested variants. (b) It does nothing for run 6 and run 8, where model E's answer is wrong on the training pictures themselves, and run 10 failed at the second turn. (c) 'Check it first on the logged pictures' cannot show that it works: the logged pictures come from the 1-cm policy and the loop is closed, so only a robot test counts. (d) The rule already exists untested in the working tree (config.py: "step_length_rule": "predicted", stall stop after 30 pictures within 3 cm), so the cost is a robot session with a checklist and Emranul's go-ahead, not half a day of code. Robot risk: no larger moves than today, but runs on straights become about 30 % longer and sheet 04 (asked median 0.31 cm) may stall; change one thing per run and keep the start fixed, or the result cannot be read.

2. Recommendation 2 (start from the demonstrations' start pose; refuse beyond about 1 cm or 3 degrees): the evidence shows that offsets existed, not that they caused failures (runs 4 and 7 started 3.8 and 4.0 cm away and reached the end, run 9 started 1.3 cm away and failed). As a hard refusal the limits would have refused all ten runs, the four good ones included, and it works against Emranul's wish for 'any orientation'; keep it as a test protocol and log the offset. An automatic move to the start pose is a new robot motion that needs the session checklist. Before that, check the cheap thing my numbers point to: nine of ten starts lay 1.3 to 3.7 cm to the LEFT of the demonstrated start, so the sheets were probably not laid where the start mark was during recording.

3. Recommendation 3 (watch and neutralise the camera's creep): (a) 'pause when more than about 8 px from the data the model was trained on' has no defined reference: the training data hold two views 17 px apart (sheets 01 to 03 and part of 04 against sheets 05, 06), the robot day is 17 px from one and 2 px from the other, and against the 6.11 reference the rule would have paused all ten runs. (b) Shifting every picture so that the hand sits at a fixed column is risky on the robot: the block's column also depends on wrist and hand in the demonstrations, and the analyst's own strip check scatters by up to 80 px in runs 4, 5 and 8; one wrong registration moves the model's input by tens of pixels. The wider shift augmentation is the safer half. The measured effect is 1 to 4 degrees of direction and none of the six failures, so this is low priority and must not become a reason to touch the camera.

4. Recommendation 4 (record more demonstrations: square lay-outs, recoveries, spread starts): (a) more square episodes do not address why run 3 failed (the same stretched turn), and the six existing ones are already fitted badly (model E answers 0.41 cm at -18 where their labels say 1.03 cm at +48), so one agreed routine matters more than the count; it also waits on the professor's answer. (b) 'Puck 1 to 2 cm beside the line' misses the state the model actually could not handle in runs 5, 9 and 10: the tip several centimetres away from a puck that still sits on the line. It costs two robot sessions and a retraining, so it should follow, not precede, the robot test of recommendation 1, which removes the cause of that state.

5. Recommendation 5 (decide how sharp corners are demonstrated; check offline whether model E reproduces the swing): supported and largely answered by my check: on the training pictures at the first corner of sheet 04 model E's own one-step answers are 47 to 78 degrees to the right of the labels (step 8: 0 degrees against +47), so a one-picture model with this target does not hold the swing; the chunk output was not checked by either of us and should be checked with the fold model that never saw sheet 04. Risk in the remedy: re-recording only the corners of sheets 03 and 04 with a new routine leaves two routines for the same places unless the old rows are dropped, and it tunes the data to the six course sheets, which Emranul ruled out.

6. Recommendation 6 (log each picture's familiarity in the driver): harmless as a log, but its supporting claim ('separated before from after in every failed run') rests on the misread steps: the number rose 7 to 9 steps before the puck left the line in runs 5 and 9 and 2 to 7 steps after it in runs 3, 6 and 8. The fair yardstick does not exist on the robot (the driving model has seen every sheet), and on an unseen exam sheet nearly every picture would exceed a 95 % point taken from seen sheets (the analyst's own table: 2.3 to 3.9 times farther). The evidence is for the 6,144-number descriptor; the recommendation names the 256-number one, which separated less in the analyst's own table.

7. Recommendation 7 (keep people and clothing out of view; patch augmentation): follows from the evidence and is low risk. The effect on the stop decision may be larger than stated (run 4, steps 40 and 41: stop chance 0.90 and 0.88 seen, 0.22 and 0.37 with my paint-over, 0.65 and 0.66 with the analyst's), and nobody can keep the view clear at the exam, so the augmentation is the part that counts.

8. Recommendation 8 (leave the picture taking, the light and the held wrist as they are): picture taking and light follow from the evidence. 'The held wrist' does not: a camera pose inside the demonstrated range says nothing about how the puck is pushed, run 8 failed on familiar pictures with asks that matched the demonstrations, the robot's forearm ran 10 to 20 degrees ahead of the demonstrations at matching pictures in that run, and the analyst's own side note has the puck sitting further to the right of the tip's line of travel than in the demonstrations. That part belongs to the wrist analysis and should stay open.

### What the verifier says was missed

1. An objective reading of the failure steps. The analyst listed it as an open question but built the before/after split and half the headline on steps read by eye; a simple puck-to-line gap per picture (puck_on_line.py) moves four of the six.

2. Model E's answers on the nearest TRAINING pictures themselves. This is the sharpest test of 'view or decision': at every key step model E answers on the robot picture as on its nearest training pictures (run 5 step 3: 0.47 cm at -54 against 0.46 at -56; run 3 step 3: 0.42 at -18 against 0.41 at -18; run 6 step 8: 0.80 at +8 against 0.58 at 0), and where the run went wrong that shared answer is far from the labels (run 6 steps 8 to 12, run 3, run 8 from step 17). The failures at those places are a fit problem that exists without the robot (own_answers_on_neighbours.out.txt, own_fit_on_training_pictures.out.txt).

3. Whether the asked length is itself right at turns in place. The analyst compared 'asked' with 'done' only; on training pictures the model asks 0.51 cm where 0.28 cm was demonstrated, in a direction that is often unrelated to the demonstrated one.

4. Run 10 has a different story from runs 5 and 9: it passed the first bend with the same detour, saw familiar pictures again at steps 15 to 21 and lost the line at the next turn at about step 30. Runs 1 and 10 were driven by models A and B, while every familiarity number is from model E's recipe; for those two runs the numbers describe a model that was not driving.

5. The start offset is one-sided: nine of ten runs began 1.3 to 3.7 cm to the left of the demonstrated start and with the picture turned 2 to 18 degrees against the demonstrations' fixed -6.1 degrees. The analyst reported distances, not the direction; a moved start mark or a different way of laying the sheet is a cheap thing to check before the next robot session.

6. The tip pixel (313, 372) comes from the camera model fitted before the camera moved; on the robot day the rod and block sit about 17 px further right. The analyst's own creep finding was not carried over to the scripts that use the camera model (camera_pose_check.py, picture_lag_check.py, puck_beside_tip.py).

7. Shadows. The light check covers mean brightness, contrast, colour and sharpness; the demonstration pictures nearest to run 6 show a hard shadow of the arm across the paper that the robot pictures lack.

8. The stop at the end of run 4 was not examined as possibly helped by the hoodie, although the analyst's own number (0.90 falling to 0.65) and mine (falling to 0.22) both show the hoodie raised the stop chance there; run 4 should not be counted as a clean stop.

9. In run 8 the robot's forearm was 10 to 20 degrees further twisted than the nearest demonstrations at every step from 4 to 15. The analyst listed the neighbours' forearm angle for run 6 but did not notice this lead, which is the best clue the view data hold for the one failure they leave unexplained.

### The analyst's corrections to the brief

1. Run 1, 'tip about 1.5 cm left of the line at the start': the 1.5 cm is the tip's offset from the demonstrations' start position in robot coordinates (tip y +0.09 cm against -1.49 cm), not from the line. In the picture the puck was centred on the line (0.00, -0.05, -0.05 cm at steps 0 to 2), and run 2 ('puck centred') started at the same tip y (+0.12 cm). All ten runs started 1.3 to 5.1 cm from the demonstrations' start and with -9.7 to +9.8 degrees of forearm twist.

2. Run 5, 'a blue hoodie may be in view': it was not in view before the run went wrong. Royal blue above 0.5 % of the picture appears in 2 pictures only (steps 23 and 24); the bright blue at the bottom left of runs 5 and 9 is the robot's own blue.

3. Run 4 is not listed as a hoodie run, but it had the hoodie in view for 22 pictures (steps 20 to 41, up to 15.1 % of the picture) and a hand from about step 32, and still reached the end. It is the strongest evidence that the hoodie is survivable.

4. Run 9, 'nothing blue in view': the hoodie's royal blue is indeed nearly absent (at most 1.3 % in 11 pictures after step 16), but the robot's own blue (48 pictures) and the dark navy object beyond the table's far edge (40 pictures) are in view, as in the runs that reached the end and in about 5 % of the training pictures each.

5. Run 3, 'sheet 02 lying straight (not as in plan 02-01)': it matches the square portrait lay-out from end B (plans 02-13 to 02-18), which is in the data. After the audit's corrected ends all six SQP episodes are end B (there is no SQP end A); the ten SQL episodes are five per end.

6. Run 6, 'first 15 moves fine': the puck stayed on the line, but the pictures had been leaving the demonstrations' range since about step 7 (percentile 89 or more, above the 95 % point at steps 12 to 14), and the hoodie only came into view at step 14.

7. Run 8, 'puck outside the loop by step 15': by eye the first wrong direction is at step 7 and the puck is off the dashed line by step 11.

8. 'The camera is taped on the forearm cover' should carry a warning: it has crept. The robot's own hand sits 16.3 px (of 640) right of the 6.11 reference, 13.6 px on the last recording day; sheets 01 to 03 were recorded entirely before the move of 8.11.

9. The set-up picture of plan 02-01 is titled 'end A', but the training index (audit-corrected ends) holds plan 02-01 as STD end B; run 4's first pictures are nearest to STD end B episodes (plan 02-31).

10. An implication of the stop rule that the brief does not spell out (read from model_mode.py): on the first picture with a stop chance of 0.5 or more the robot still moves 1 cm; it stops on the second such picture.

### The analyst's open questions

1. For the professor: how will the sheet lie at the exam (square to the robot or turned, on a marked start or anywhere), and who places the puck and the arm? This decides how much square-lay-out and varied-start data are needed.

2. For the professor: is it within 'imitation learning only' to record recovery demonstrations that start from the states where the model failed on the robot (arm and puck left where the run ended, then hand-guided back)?

3. Will the camera mount be re-fixed before more data are recorded, or do the team keep the present view and neutralise the creep in software? New demonstrations and robot runs should share one view either way.

4. Is the creep a turn or a sideways shift of the camera? My test slid the picture sideways; a turned camera also changes the direction 'up in the picture' on the table by up to about 1.8 degrees, which the driver's heading does not know.

5. Why did the puck slide left in run 8 between steps 7 and 11 although the pictures were familiar and the asked steps matched the demonstrations in direction, length and twist? Candidates outside my question: the wrist and hand are held while the demonstrators moved them; the puck sat a median 0.66 cm right of the tip's line of travel.

6. Can a one-picture model reproduce the demonstrators' several-step swing round the puck at sharp corners (run 6), or does that need chunks executed as a whole or some memory?

7. Are the OL1.5 and OR1.5 labels of sheet 01 (plans 01-17 and 01-18) swapped, and are other sheets' offset labels affected? It does not change training (labels are not inputs) but any analysis by kind of start.

8. The failure steps were read by eye, good to about one step. If another analyst has an objective puck-to-line distance per picture, the before and after split should be re-run with it (per_run_summary.py reads FAILURE_STEPS from common.py).

## Wrist and camera (folder `wrist_and_camera`)

**Analyst's answer.** No: the held wrist is not what made the robot's view differ where the runs failed. At the first bends of sheets 03 and 04 the demonstrations at the same places had the same wrist as the driver (13.5 degrees); what differed there was the forearm twist, 30 to 77 degrees behind. Holding the wrist does move the view against 27 % of all demonstration rows and 55 % of the stop pictures: that costs almost no direction (about 0.6 points) but, in an offline imitation, about 13 of 207 right stops, so the fix belongs in the data, the scoring and the start procedure, not in a moving wrist.

**Verifier's overall verdict.** Partly. The central answer stands and I reproduced it with my own code (scripts v01 to v20 and their outputs are in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/wrist_and_camera/verify/): at the first bends of sheets 03 and 04 the demonstrated wrist was 13.5 degrees like the driver's, the forearm twist lagged by 20 to 95 degrees even when robot and demonstrations are matched by travelled distance, and the view-shift numbers (27.1 % of rows over 20 px, 2.80 px per degree) are exact and hold on real picture pairs. What does not stand as stated is the cost in stops: my own imitation gives 103 to 93 or 94 right stops, not 90, and a control that moves nothing and only smears the edge rows a move would push out already gives 95, so "about 13 of 207" is mostly inside the imitation's own error. The recommendations that rest on that number (wrist sweep, driver's-view scoring column, retraining to the driver's view, re-solving new demonstrations) are therefore weaker than presented; keeping the wrist at 13.5, not following the lean output and the camera finding are sound.

### Verdicts on the claims

1. **confirmed**: Wrist held at 13.5 and hand at -6.1 against every demonstration row: camera moves a median 1.7 mm and turns 0.27 degrees; the tip and table shift a median 5.2 px at 640 (90th 41, 95th 59, most 132), over 20 px in 27.1 % of rows, over 16 px in 30.9 %, over 40 px in 10.6 %; the shift is vertical and follows the wrist alone at 2.80 px at 640 = 0.70 px at 160 per degree; tip height adds only a few px.

   Calculated by me with the project's own functions (side_view_test.solve_with_wrist_fixed, magnet_tip_and_lean, forearm_camera_model.project_points), all 18,522 rows, v02: tip shift median 5.20, 90th 41.2, 95th 58.5, most 131.5 px; over 20 px 27.1 %, over 16 px 30.9 %, over 40 px 10.6 %; camera move median 1.66 mm (90th 13.89, most 38.86), turn median 0.27 degrees (90th 2.58, most 10.07); slope 2.803 px per degree (r 0.9988, residual 1.5 px), sideways part median 0.22 px; all rows solvable, 4 rows over 58 degrees lean. Measured on real pictures in a way the analyst did not try (v13: two different episodes of the same placement, tip within 3 mm, twist within 1.5 degrees, wrists 8 to 14 degrees apart): the table content moves 2.83 px per degree against 2.98 calculated for the same pairs, same sign in 14 of 14 clean pairs, ratio median 0.96 (one sheet-06 pair locked onto the wrong dash). Tip height alone (v17, 400 rows): a table point 5 cm ahead shifts a median 2.4 px (90th 6.5). With only the wrist turning the rod's end moves 1.90 px per degree and the tip 2.71 (v19), so the block moves about a third less than the table: the analyst's imitation, which moves everything but the fork by the table's amount, misplaces the block by about 0.2 px at 160 per degree.

2. **confirmed**: The report's '23.3 % over 20 px' belongs to the wrist held at each episode's own first-row value; for the driver's constant 13.5 it is 27.1 %.

   Calculated, v02: first-row variant 23.2 % (tip), 23.0 % (table point under the tip), 23.4 % (5 cm ahead); constant 13.5: 27.1 %.

3. **confirmed**: The mismatch grows with reach and progress, hardly with twist: over 20 px in 24.1 / 27.1 / 28.7 / 33.2 % of rows for |joint 4| 0-30 / 30-60 / 60-90 / 90-120; 2.3 / 20.9 / 29.7 / 33.6 / 35.1 % by reach; by sheet 39.0, 17.4, 25.0, 23.4, 33.6, 43.8 %.

   Calculated, v02: identical to the first decimal (24.1, 27.1, 28.7, 33.2; 2.3, 20.9, 29.7, 33.6, 35.1; 39.0, 17.4, 25.0, 23.4, 33.6, 43.8).

4. **confirmed**: The demonstrated wrist is start and operator noise: 149 start-mark episodes begin at one pose (tip 14.0, -1.5 cm, twist 0, wrist 13.5, hand -6.1), 61 taught starts begin at -29.3 to +8.9 degrees; 582 of 18,312 steps (3.2 %) move the wrist over 1 degree and carry 68 % of its travel; 50.5 % of rows within 2 degrees of 13.5; over 5.7 degrees away in 22.5 % of start-mark rows and 88.4 % of taught-start rows.

   Measured from frames.csv, v01: 149 and 61 episodes; first rows of the start-mark episodes: tip 14.0, -1.5 cm, twist -0.17 to +0.09, wrist 13.53 to 13.97, hand -6.15 to -6.06; taught starts -29.27 to +8.87; 582 steps (3.18 %) carrying 68.3 %; 50.5 % within 2 degrees; 22.5 % and 88.4 %; wrist percentiles 1st -15.2, 5th -6.7, 95th 25.2, 99th 31.4, range -29.3 to 45.3. Small differences: last tenth of an episode 25 to 44 % (not 26 to 43), sheet 06 82 %; per-episode wrist range (largest minus smallest) median 8.0 degrees for Emranul (193 episodes) and 0.7 for Marshal (17), against the analyst's 7.4 and 3.3, same direction. Stronger form of the point (v04): within the first 20 cm of travel only 0.5 to 5.4 % of start-mark rows on sheets 03 and 04 and 0 % on sheets 05 and 06 have a wrist more than 5.7 degrees from 13.5.

5. **confirmed**: Robot runs against demonstrations at the same places: 493 of 554 pictures have 5 or more neighbours and in 91.7 % the neighbours' median wrist is within 5.7 degrees of 13.5; in the first 13 pictures of runs 5, 9, 10 (sheet 03) and 6 (sheet 04) the neighbours' wrist was 13.5 while the twist was 30 to 77 degrees behind (largest gaps 44, 39, 56, 77); run 7 finished the loop although the neighbours' wrist band was -13 to +28.

   Measured, three ways. (a) The analyst's family (same sheet and corrected end, start mark), v03b: 501 of 554 pictures, 91.6 %; first 13 pictures: run 5 median wrist 11.2 to 13.6, largest twist gap 44, outside the 10th-90th band in 6; run 9: 13.5 to 13.6, 39, 8; run 10: 13.5 to 13.6, 56, 9; run 6: 13.5 to 18.0, 77, 12. (b) A stricter family found from the paths themselves (STD only), v03: run 5 pictures 6 / 8 / 10: robot twist -27 / -38 / -48 against demonstrated medians -47 / -72 / -90, neighbours' wrist 13.6 to 13.8, 100 % in range; run 6 pictures 2 / 4 / 6 / 8: 0.4 / 0.8 / 7.9 / 15.0 against 24 / 37 / 66 / 81, wrist 13.5. (c) Matched by travelled distance, which does not depend on where the sheet lay, v18: at 6 / 8 / 10 cm run 5 had turned -17 / -28 / -39 degrees where the STD demonstrations had turned -53 to -88 / -77 to -95 / -89 to -95; run 6 stayed at 12 to 14 degrees where the demonstrations reached 30 to 108; demonstrated wrist 13.5 to 13.6 throughout. Run 7 (v03c): band -12.9 to +28.4, and at 36 of 85 pictures more than a third of the neighbours had a wrist over 5.7 degrees away. Addition: in run 8 the twist was not behind (17 / 27 / 40 / 53 degrees at 8 / 10 / 12 / 14 cm against 11 to 21 / 18 to 31 / 25 to 40 / 36 to 53), so the twist lag explains sheets 03 and 04 only.

6. **confirmed**: Sheet 06 is the one sheet where the demonstrated wrist differs later (median 17 to 28 degrees from 30 % of an episode), but run 8 left the line before that; from picture 2 its tip ran 1 to 2 cm beside every demonstrated tip.

   Measured: median wrist by tenth of the episode on sheet 06 (start-mark): 13.5, 13.5, 13.6, 16.7, 19.5, 14.7, 20.1, 22.5, 28.0, 27.9 (v17). No sheet-06 start-mark episode has its wrist more than 5.7 degrees from 13.5 before 22.5 cm of travel (median 37 cm, v04); run 8 had left the loop by about 15 cm. Against the STD demonstrations of plan 06-01's end only, the run's tip was 2.1 to 3.0 cm from every demonstrated tip already in pictures 0 to 7 (v03), more than the analyst's 1 to 2 cm with the wider family.

7. **corrected**: None of the ten runs started at the demonstrated start pose: start tips 1.3 to 5.0 cm (median 3.2) from the demonstrated one, forearm twisted -9.7 to +9.8 degrees, picture heading 1.7 to 18.3 degrees off.

   The numbers are right (measured, v03: gaps 1.6, 1.8, 2.5, 3.8, 3.1, 5.0, 4.0, 3.2, 1.3, 3.3 cm; twists -9.7 to +9.8; demonstrations -0.17 to +0.09). The reading needs a correction: the tip-to-tip gap mixes two things. Measured on the first pictures with the fitted camera (v20, same method on both sides): the demonstrations begin with the tip 2.8 to 5.3 cm behind the puck's centre (sheet 03 end A: puck at 16.8 cm; sheet 04: 19.3 cm; sheet 06: 18.8 cm; tip always at 14.0), the runs began 2.4 to 3.7 cm behind it; and the puck itself, so the start cap and the sheet, lay 1.9 cm (run 6), 2.4 cm (run 8) and 2.4 cm (run 10) from its place in the STD demonstrations of the same sheet and end, whose own spread is under 1 cm there. So the sheet was about 2 cm off its recorded place where this can be measured; it cannot be measured for sheets 01, 02, 05 or runs 5 and 9 (puck not found by colour, or the demonstrations themselves scatter). The start twist of up to 10 degrees is the clearest difference: no demonstration starts twisted.

8. **confirmed**: The camera stepped once on its tape on 8.11.26 between 17:23 (plan 04-24) and 19:33 (plan 04-30) by about 14.6 px right and 6 px down at 640 and has not moved since; 12,087 of 18,522 rows (65.3 %) are from before the step; a 4 px shift at 160 changes direction by under 1 point.

   Measured with my own method, not the recording tool's phase correlation (v05: blue-minus-red picture of the bottom strip, template of the fork from the first demonstration of 6.11, normalised cross-correlation): 101 first pictures before the step sit at x +0.3 px (5th -3.8, 95th +3.5), 48 after it at +14.8 (13.4 to 16.2); plan 04-24 at 17:23:28 is +3.1, plan 04-30 at 19:33:39 is +16.4; none back afterwards; the ten runs' medians are +15.5 to +17.6 (median +16.6), about 2 px right of the last recording day, which is within the measure's noise. Vertical part about +4.7 px down but noisy (single runs read -10 to +10). Rows before the step: 12,087 of 18,522 (65.3 %), sheet 04: 4,569 of 6,350. Extra check on the test's own pictures (v15, model E, my inference gives back the logged answers exactly): with the step undone (whole picture 15 px left, 5 px up at 640) the twist asked over the first 15 pictures is -56.4 against -56.6 (run 5), -70.0 against -74.0 (run 9), 8.7 against 8.6 (run 6); the direction turns by a median 1.6 to 11 degrees per picture, more than the analyst's 1.2 to 1.5 on demonstration rows but small against the twist lag. The step does not explain the failures at the bends.

9. **confirmed**: Following the model's lean output would make the mismatch worse: lean sign right on 82.5 % (E recipe) and 81.9 % (A recipe) of steps, a held wrist alone 98.5 % and 68.8 % of the variance; followed along the demonstrated paths the wrist ends a mean 15.6 (E) and 13.2 (A) degrees from the demonstrated wrist against 3.4 for the constant; the rod passes 58 degrees in 246 rows.

   From the saved fold predictions (v06, real pictures): sign right on 81.8 % (E), 80.4 % (E seed 2), 81.1 % (A) of the 11,419 steps with a lean label of 0.5 degrees or more; an answer of 'always forward' scores 73.5 %, so the output is 7 to 8 points above a constant. Correlation with the label 0.37 to 0.40; on the 582 wrist-jump steps the predicted lean correlates -0.07 to -0.13 with the wrist's own change. Kinematic part recomputed with the project's solver on 2,500 random steps (v14): explains 74 to 76 % of the variance (analyst 68.8 % on all steps), sign right on 98.7 %, 0.857 degrees of lean per degree of wrist. Drift recomputed in a simpler, linearised way (running sum of predicted minus demonstrated lean, divided by 0.867): mean distance to the demonstrated wrist on start-mark episodes 11.3 (E), 12.8 (E seed 2), 9.7 (A) degrees against 3.4 for the constant; last-row gap 5th to 95th percentile -43 to +29 (E). Smaller than the analyst's 15.6 and 13.2 from the full simulation, same conclusion: three to four times worse than holding. The 246 rows over 58 degrees I did not recompute; it is plausible, since a wrist of 25 to 35 degrees gives a lean of 51 to 59 degrees at 40 cm and 63 to 72 at 46 cm (v10). Runs 1 and 2: summed lean output +31.5 and +32.8 (v12), as stated.

10. **confirmed**: A wrist fitted to reach and twist is no better than the constant (3.5 and 4.1 against 3.4 degrees on start-mark episodes, fitted on five sheets and applied to the sixth).

   Measured, my own fits (v11): constant 13.5: 3.40 degrees mean distance; table of medians by reach and twist 3.44; least squares on reach, twist, squares and product 3.82; a random forest that also gets the tip's x and y and the travelled share of the episode 4.34. Nothing predicts the wrist on an unseen sheet better than the constant.

11. **confirmed**: Limits: wrist_pitch_max_rad is 1.5 rad (85.9 degrees), a dead check at 13.5; with the wrist held the lean is 0.4 / 26 / 42 / 54 degrees at 14 / 30 / 40 / 46 cm; 58 degrees is met at 47 cm (twist 0 to 30), 48 cm (60), 49 cm (90 to 117); demonstrations reached 48.5 cm; largest lean in the test 51.3 degrees at 45.0 cm.

   Read from config.py: 1.5 rad = 85.9 degrees, set on 4.11.26 when the bracket left the hand. Calculated with the project's solver at the driver's tip height (v10): lean 0.4 / 26.1 / 42.1 / 54.1 degrees at 14 / 30 / 40 / 46 cm, twist 0; 58 degrees is first passed at 47.75 cm (twist 0), 48.0 (30), 48.75 (60), 49.75 (90) and not up to 50 cm at 117; solutions exist from 12 to 50 cm at every twist. Measured: demonstrations reach 48.5 cm, 77 rows beyond 46 cm, lean over 58 degrees in 7 demonstrated rows and in 4 with the wrist held (v02); the test's largest lean 51.28 degrees at 44.9 cm in run 10, no refusal in any run, wrist 13.27 to 13.53 and hand -6.23 to -6.06 in every picture (v12).

12. **confirmed**: The hand (joint 6) moves neither the tip nor the camera; the camera sees the block turn (hand 29.6 degrees in five sheet-03 MID episodes); -6.1 matches 82.2 % of rows within 0.3 degrees, 0.0 in 9.8 %, over 10 degrees away in 3.6 % = 663 rows of six episodes; hand output on the robot -0.02 to +0.08 degrees per step.

   Calculated (v10): tip and camera centre identical to 0.001 mm for hand -6.1, 0 and 29.6 degrees. Measured (v01, v12): 82.2 %, 9.8 %, 663 rows (3.6 %), 65 steps over 1 degree, five sheet-03 MID episodes (plans 03-37 to 03-41) at 29.4 to 29.7 degrees; the model's hand output in the runs -0.02 to +0.08 per step, summed at most +1.22. One detail is wrong: the 663 rows lie in 13 episodes, not six (seven with 57 to 139 rows: plans 03-37 to 03-41, 04-29, 04-38; three with 11 to 27 rows; three with 1 or 2). The share of pushes with the rod leaning back I did not recompute.

13. **unverifiable**: How the block touches the puck is not settled: the puck seems to touch the square block body, and in run 5's pictures 6 and 10 it sits on the block's left corner where the nearest demonstrations show it in front of the middle.

   I looked at run_05_pictures_against_nearest_demonstration.jpg and at my own sheet of start pictures (v16): the description is right, the hand, rod and block sit at the same place in robot and demonstration pictures, and the puck is beside the block's left corner at pictures 6 and 10. Which part touches the puck cannot be seen from above or read from the logs; the analyst rightly calls it a guess. One fact for whoever follows this up: the puck is 2.4 cm across (config.PUCK_PARAMETERS; 2.1 cm measured on the pictures in v20), not 4 cm as the brief says, so the block is wider than the puck.

14. **confirmed**: Direction hardly cares about the held wrist: in the imitated driver's view the fold models' steps within 45 degrees go from 68.4 % to 67.8 % (-0.6 points, 14,473 rows), on rows moved by more than 4 px from 62.6 % to 60.7 %.

   My own imitation with the E-recipe fold models (v09: whole-pixel move of 0.701 px per degree, fork kept by my own mask, my inference gives back the saved stop chances to 0.000001): on the 8,206 moved rows that move the tip 5 mm or more, 64.7 % to 63.5 % (edge row repeated) or 63.2 % (mirrored edge), which is -0.7 points over all 14,473 rows; rows moved by more than 4 px: 62.6 % to 60.5 %. A control that moves nothing and only smears the rows a move would push out: 64.8 %, no loss. Saved predictions (v17): predicted twist is 0.50 (E) and 0.52 (A) of the demonstrated size on steps that twist by 2 degrees or more, without any shift.

15. **corrected**: Stops: 115 of 208 stop rows (55.3 %) were recorded with a wrist more than 5.7 degrees from 13.5; in the imitated driver's view the fold models' right stops fall from 103 to 90 of 207 and never-stopped rises from 69 to 83; the loss grows with the size of the move; model E on its own sheets is hardly touched.

   Confirmed: 115 of the 207 scored stop rows are out of range (83 below, 32 above; 116 of 210 before cleaning), median wrist at the stop 10.8, sheet 06 88 %; the baseline 103 right, 27 early, 69 never (project's stop_metrics on the saved predictions, v06). Not confirmed: the size of the loss. My own imitation (v09, v09b) gives 93 right (edge row repeated) or 94 (mirrored edge), never 79 or 78: 16 right stops lost and 6 won. The control that moves NOTHING and only smears the rows a move would push out gives 95 right (10 lost, 2 won, 5 of them new early stops). So about 8 of the 9 to 13 lost stops appear without any change of view, and the estimate depends on how the picture's edge is filled. Where the last picture moves by 0 to 4 px: 44 to 43 or 44 right of 95; 5 to 9 px: 27 to 25 of 49; 10 px or more: 32 to 25 of 63, exactly where the imitation is crudest. Real pictures only (v07, five fold-model sets): stop rows recorded with the wrist within 5.7 degrees of 13.5, which is the driver's own view, reach a stop chance of 0.5 in 73.0 % (92 rows), with a lower wrist in 78.5 % (83 rows), with a higher wrist in 42.5 % (32 rows, 21 of them sheet 06); inside same-goal groups the driver's-wrist rows are 6 points lower (4 of 5 groups, 3 to 7 rows in the smaller cell), within noise. Model E on all sheets (v19): both last rows at 0.5 or more in 187 of 208 episodes as recorded, 178 in my imitation, 187 in the control. Fair summary: a small stop penalty on unseen sheets is possible, concentrated at line ends where the demonstrated wrist was 14 degrees or more away; its size is not known and 13 of 207 is an upper figure, not an estimate.

### The analyst's findings with their evidence

1. (calculated, confidence high) Wrist held at 13.5 and hand at -6.1 degrees against every demonstration row (same tip point, same joint 4): the camera moves a median 1.7 mm (90th percentile 13.9, most 38.9 mm) and turns a median 0.27 degrees (90th 2.6, most 10.1). The magnet tip, and the table with it, shifts a median 5.2 px at 640 (90th 41, 95th 59, most 132 px), which is 1.3 / 10.3 / 14.6 / 32.9 px at the model's 160. Over 20 px in 27.1 % of rows (picture centre: 27.5 %), over 16 px (the training shift augmentation) in 30.9 %, over 40 px in 10.6 %. The shift is straight up or down the picture and follows the wrist alone: 2.80 px at 640 = 0.70 px at 160 per degree; the blue fork does not move. The driver's tip height (5.0 mm above the fitted table against a demonstrated median of 2.8 mm) adds only a median 3.9 px at 640.

   Evidence: held_wrist_views.py / .out.txt / .png (all 18,522 rows of frames.csv; numpy copies of ned2_fk, magnet_tip_and_lean, solve_with_wrist_fixed and the fitted camera, equal to the project functions within 2e-13 mm and 0.0003 degrees: wrist_geometry.out.txt). Checked on real pictures: across 75 demonstrated wrist jumps the block moves 1.83 px per degree, calculated 1.81 for the rod's end; 277 table patches land a median 0.96 px from the calculated place (check_shift_per_degree.out.txt).

2. (calculated, confidence high) The report's '23.3 % over 20 px' is reproduced (23.2 % tip, 23.5 % picture centre), but it belongs to the wrist held at each episode's own first-row value. For the driver's actual constant 13.5 degrees the share is 27.1 %.

   Evidence: held_wrist_views.out.txt, variants held_first_row and held_same_tip.

3. (calculated, confidence high) The mismatch grows with reach and with progress in the episode, hardly with the forearm twist. Rows with the tip over 20 px from its demonstrated place: |joint 4| 0-30: 24.1 %, 30-60: 27.1 %, 60-90: 28.7 %, 90-120: 33.2 %. Reach 12-20 cm: 2.3 %, 20-26: 20.9 %, 26-32: 29.7 %, 32-38: 33.6 %, 38-50: 35.1 %. By sheet: 01 39.0 %, 02 17.4 %, 03 25.0 %, 04 23.4 %, 05 33.6 %, 06 43.8 %.

   Evidence: held_wrist_views.out.txt (tables by sheet, twist, reach and twist x reach), held_wrist_views.png.

4. (measured, confidence high) The demonstrated wrist is start and operator noise, not a function of the state. All 149 start-mark episodes begin at one pose (tip 14.0, -1.5 cm; twist 0.0; wrist 13.5; hand -6.1). The 61 taught-start episodes (NE, MID, X1, X2; 2,465 rows) begin at the least-lean wrist for their start (-29.3 to +8.9 degrees, within 0.26 degrees of the calculated least-lean wrist) with the hand at 0.0. After the start the wrist changes in rare jumps while the arm is guided by hand: 582 of 18,312 steps (3.2 %) move it more than 1 degree and carry 68 % of its travel. Half of all rows (50.5 %) are within 2 degrees of 13.5. More than 5.7 degrees away (= outside the shift augmentation): 22.5 % of start-mark rows (0 to 3 % in the first tenth of an episode, 26 to 43 % in the last tenth, 82 % on sheet 06) and 88.4 % of taught-start rows.

   Evidence: frames.csv and episodes.csv; numbers printed by held_wrist_views.out.txt, wrist_options.out.txt, runs_against_demonstrations.out.txt (start pose), and the session's checks of the first rows.

5. (measured, confidence high) Robot runs against demonstrations at the same places (same sheet, same placement by the corrected line ends, tip within 2 cm): 493 of 554 run pictures have 5 or more such rows, and in 91.7 % of them the neighbours' median wrist is within 5.7 degrees of the held one. In the first 13 pictures of the failing runs 5, 9, 10 (sheet 03) and 6 (sheet 04) the neighbours' median wrist was 11.2-13.6, 13.5-13.6, 13.5-13.6 and 13.5-18.0 degrees, with 97 to 99 % of neighbour rows in range. The twist differed instead: run 5 had -27 / -38 / -48 degrees at pictures 6 / 8 / 10 against demonstrated medians of -40 / -66 / -90; run 6 had 0.4 / 0.8 / 7.9 / 15.0 at pictures 2 / 4 / 6 / 8 against 26 / 40 / 71 / 86; the largest gaps were 44, 39, 56 and 77 degrees, and 6, 8, 9 and 12 of 13 pictures lay outside the neighbours' 10th-90th percentile twist band. The rod's lean differed by -5.5 to +3.6 degrees (median per run). Run 7 (sheet 05) finished the whole loop with the held wrist although the neighbours' wrist band there was -13 to +28 degrees.

   Evidence: runs_against_demonstrations.py / .out.txt, runs_against_demonstrations_steps.csv, run_05_against_demonstrations.png, run_06_..., run_07_...; run_05_pictures_against_nearest_demonstration.jpg (robot and demonstration pictures side by side: the hand, rod and block look alike).

6. (measured, confidence medium) Sheet 06 (run 8) is the one sheet where the demonstrated wrist really differs later on (median 17 to 28 degrees from 30 % of an episode onwards, so the robot sees the block and table 8 to 10 px at 160 lower than those pictures). But the run left the line before that: up to picture 18 the same-pass demonstrations (twist within 20 degrees) had wrist 13.5 to 14.0, and from picture 2 the robot's tip ran 1 to 2 cm beside every demonstrated tip of that placement (only 8 of the first 13 pictures have neighbours at all).

   Evidence: runs_against_demonstrations.out.txt (run 8 table), run_08_against_demonstrations.png.

7. (measured, confidence high) None of the ten runs started at the demonstrated start pose. Start tips were 1.3 to 5.0 cm (median 3.2) from the demonstrated one, the forearm was twisted -9.7 to +9.8 degrees (demonstrations: -0.17 to +0.09), and the picture's heading was 1.7 to 18.3 degrees (median 7.7) off. The driver keeps whatever twist the guiding hand left.

   Evidence: run_starts_against_demonstrated_start.csv, runs_against_demonstrations.out.txt (first table); model_mode.held_joints_at keeps joint 4.

8. (measured, confidence high) The camera stepped once on its tape, on 8.11.26 between 17:23 (plan 04-24) and 19:33 (plan 04-30), by about 14.6 px to the right and 6 px down (at 640), and has not moved since: robot runs of 11.11 sit +16.3 px right of the reference picture, demonstrations after the step +14.4, before the step -0.2. 12,087 of 18,522 training rows (65.3 %: all of sheets 01 to 03 and 4,569 of 6,350 rows of sheet 04) were taken before the step, so against those the robot's view sits about 16 px (4 px at 160) to the right, the edge of the shift augmentation. Imitated on 600 held-out rows (whole picture 4 px right) this changes steps within 45 degrees by -0.7 points for the fold models (95 % interval -2.3 to +1.0) and +0.3 for model E (-1.2 to +1.8); the predicted direction turns by 1.2 to 1.5 degrees on average and the twist size is unchanged. Even 8 px costs at most 1 point.

   Evidence: camera_creep.py / .out.txt / .png (the recording tool's own camera_shift_px on the bottom strip, valid on every run picture because wrist and hand are held at the reference pose; agrees with episodes.csv camera_shift_px of 15.0 and 14.4 on sheets 05 and 06); shift_sensitivity.out.txt.

9. (calculated, confidence high) Following the model's lean output would make the mismatch worse, not better. The lean label is mostly kinematics: a held wrist alone explains 68.8 % of its variance and gets its sign right on 98.5 % of steps, the fold models' lean output on 82.5 % (E recipe) and 81.9 % (A recipe). The part that needs the wrist to move is not predicted at all (correlation -0.065 and -0.053; sign right on 37.6 % and 39.9 % of the 582 wrist-jump steps). Followed step after step along the demonstrated paths from 13.5, the wrist ends a mean 15.6 degrees (E) and 13.2 (A) from the demonstrated wrist on start-mark episodes, against 3.4 for the constant; the final wrist spreads from -54 to +26 degrees around 13.5 and the rod lean passes 58 degrees in 246 rows of 9 episodes (largest 82 to 86). Even the true lean labels drift when integrated (median 1.7, up to 21 degrees). On the straight sheet 01 the summed lean output of runs 1 and 2 (+31.5 and +32.8 degrees) is simply the lean the held wrist produced anyway (0.4 to 38.2 and 1.8 to 38.6).

   Evidence: wrist_options.py / .out.txt / .png, wrist_options_scores.csv (saved fold predictions of 2026-11-10_resnet18_chunk_stop_weight_25 and 2026-11-09_resnet18_final_candidate on sheets never seen; single-step check exact to 0.003 degrees); lean_and_reach.out.txt part 3.

10. (calculated, confidence high) A wrist fitted to reach and twist is no better than the constant. Fitted on five sheets and applied to the sixth, the mean distance to the demonstrated wrist on start-mark episodes is 3.5 degrees (table of medians) and 4.1 (least squares) against 3.4 for the constant 13.5; on taught-start episodes 16.0 and 14.9 against 16.5.

   Evidence: wrist_options.out.txt (table 'each wrist rule against the demonstrated wrist').

11. (calculated, confidence high) Limits. wrist_pitch_max_rad is 1.5 rad (85.9 degrees), so the driver's bracket check can never fire at 13.5 degrees and would not protect a moving wrist. With the wrist held, the rod's lean is fixed by reach and twist: 0.4 degrees at 14 cm, 26 at 30 cm, 42 at 40 cm, 54 at 46 cm (twist 0). The 58-degree limit is met at 47 cm reach (twist 0 to 30), 48 cm (60) and 49 cm (90 to 117), where the run would end with a refusal; the joint limits allow 12 to 49 cm at every twist. The demonstrations reached 48.5 cm (77 rows beyond 46 cm); the test's largest lean was 51.3 degrees at 45.0 cm (run 10). All 18,522 demonstrated tip points can be reached with the wrist held; 4 would pass 58 degrees. The demonstrated wrist range is -29.3 to +45.3 degrees (1st to 99th percentile -15.2 to 31.4).

   Evidence: lean_and_reach.py / .out.txt / .png (arm model, tip 5 mm above the fitted table); held_wrist_views.out.txt (last lines); config.py RECORDING_PARAMETERS and MODEL_MODE_PARAMETERS.

12. (measured, confidence high) The hand (joint 6) moves neither the tip nor the camera (0.000 mm for a 30-degree turn in the arm model, assuming the block is centred on the rod). The camera does see the block turn: in five sheet-03 MID episodes the hand is at 29.6 degrees and the block looks like a diamond. The held -6.1 equals the demonstrations within 0.3 degrees in 82.2 % of rows (0.0 in 9.8 %; more than 10 degrees away in 3.6 % = 663 rows of six episodes); the hand turned by more than 1 degree in 65 of 18,312 steps. The model's hand output on the robot was -0.02 to +0.08 degrees per step, at most +1.2 degrees summed over a run, so following it would change nothing. The rod leaned against the push much as demonstrated (leaning back by more than 10 degrees in 0 to 28 % of pushes per run against 4 to 27 % per sheet in the demonstrations).

   Evidence: lean_and_reach.out.txt parts 2 and 3, lean_against_push.csv; pictures 2026-11-08 path_03 episode 041 frames/000.jpg (hand 29.6) against path_04 episode 013 frames/079.jpg (hand -6.2).

13. (guess, confidence low) How the block touches the puck is not settled. In the pictures the puck touches the edge of the square black block body (about 3.3 cm wide), not a round 1.3 cm tip, and in run 5's pictures 6 and 10 it sits on the block's left corner while the nearest demonstrations show it in front of the block's middle. If a flat face pushes, its angle to the push (set by the forearm twist and the hand angle) decides whether the puck slides sideways.

   Evidence: run_05_pictures_against_nearest_demonstration.jpg; niryo_data.md (electromagnet support about 40 x 30 x 30 mm; 'worth re-measuring which face is meant').

14. (calculated, confidence medium) Direction hardly cares about the held wrist. With every held-out picture moved to the driver's view (its own shift, fork kept), the fold models' exact steps within 45 degrees of the demonstrated step go from 68.4 % to 67.8 % (-0.6 points, 14,473 rows); on the 4,281 rows moved by more than 4 px from 62.6 % to 60.7 % (-1.9 points). The twist output does not shrink (size ratio 0.40 to 0.41 on moved rows). The under-twisting exists without any shift: predicted twist is 0.47 of the demonstrated size on unseen sheets and 0.64 for model E on its own training sheets.

   Evidence: held_wrist_offline_scores.out.txt, held_wrist_offline_direction.csv; shift_sensitivity.out.txt (600 rows, paired: fold models -1.8 points, interval -4.5 to +0.5; model E -2.2, interval -4.3 to -0.2); inference path checked against the saved predictions to 0.0001 degrees.

15. (calculated, confidence medium) Stops are where the held wrist may cost something on unseen sheets. 115 of 208 stop rows (55.3 %) were recorded with a wrist more than 5.7 degrees from 13.5 (median wrist at the stop 10.8, 5th to 95th percentile -10.8 to 29.2; sheet 06: 88 % with median 27.3; taught starts: 95 %). In the imitated driver's view the fold models' right stops fall from 103 to 90 of 207 episodes and never-stopped rises from 69 to 83 (start-mark episodes 66 to 60 of 147; taught-start 37 to 30 of 60). The loss grows with how far the end view moves: 44 to 42 of 95 episodes (0 to 4 px), 27 to 24 of 49 (5 to 9 px), 32 to 24 of 63 (10 px or more). Controls: moving the whole picture by the same amounts loses the same (-7.8 points of out-of-range stop rows), moving it the other way doubles the loss (-17.4); sideways shifts up to 8 px change the stop share by at most 1.5 points, content 12 px up by -6.3. Model E on sheets it was trained on is hardly touched (100 % to 97.4 %), which fits the test: runs 1, 2, 4 and 7 stopped by themselves.

   Evidence: stop_views.py / .out.txt, held_wrist_offline_scores.out.txt, held_wrist_offline_stops.csv, held_wrist_offline_by_shift.out.txt (project stop rule: training_metrics.stop_metrics, 0.5 on two pictures in a row; the 'as recorded' line reproduces the known 103). The imitation is crude for large moves (bottom rows smeared), so the size of the effect is uncertain; the direction and the dose-response are the solid part.

### The analyst's recommendations

1. Keep the wrist at 13.5 and the hand at -6.1 degrees in the driver. Do not follow the lean or hand outputs and do not fit a wrist rule. Remove the dead 'bracket's safe limit' check or replace its text.

   Why: At the places where runs failed the demonstrated wrist was already 13.5; following the lean output moves the wrist a mean 13 to 16 degrees away from the demonstrations (constant: 3.4) and pushes the rod past 58 degrees; a fitted function gives 3.5 to 4.1 against 3.4.

   Needs: code only. Effort: none for the decision; 15 minutes to tidy the check

   Expected effect: Avoids a change that would add drift and refusals; no gain expected from it.

   Risk: On the later part of loops (sheet 06 most) the view stays 8 to 10 px at 160 from the demonstrated one; handled by the data and scoring items below.

2. Give the driver a taught start for baseline runs: move to the demonstrations' start pose (tip 14.0, -1.5 cm, twist 0, wrist 13.5, hand -6.1, as simulated_run already does) with the sheet's start cap on the taped mark; keep hand-placed starts as a separate, labelled robustness test.

   Why: All 149 start-mark demonstrations began at that one pose; the ten runs began 1.3 to 5.0 cm away with twists of -9.7 to +9.8 degrees and headings up to 18 degrees off, so the baseline was never tested as demonstrated.

   Needs: code only, then a robot test. Effort: 1 to 2 hours of code and simulated runs, 15 minutes at the robot with a checklist

   Expected effect: A clean baseline: failures can then be blamed on the model and not on the start. Size of the gain unknown (runs 2, 4 and 7 succeeded with offsets of 1.8 to 4.0 cm).

   Risk: The arm moves by itself to the start, so the approach must come from above and the puck be placed afterwards; needs Emranul's go and the checklist.

3. Wrist sweep at the robot, pictures only: at about four places on one sheet, including the puck on the line's end, set the wrist to -10, 0, 13.5 and 25 degrees with the same tip and twist (solve_with_wrist_fixed; lift 3 cm between poses), save a picture each, and read the model's stop chance and direction offline.

   Why: My estimate of the stop loss (103 to 90 right stops) rests on an imitation of the view; 16 real pictures settle whether the stop decision really depends on the wrist.

   Needs: a robot test. Effort: 1 hour for a small script and checklist, 20 to 30 minutes at the robot

   Expected effect: Decides whether the next three items are worth doing: if the stop chance at the line's end changes by less than about 0.1 across wrists, drop them.

   Risk: The block can nudge the puck when the rod's lean changes, hence the lift between poses; nothing is pushed.

4. Score every model offline in the driver's view as well: add the held-wrist view (each picture moved by 0.70 px at 160 per degree between its wrist and 13.5, fork kept) as a second column in compare_runs for the stop rule and the direction.

   Why: 55 % of the stop pictures and 31 % of all pictures were recorded with a wrist the driver never has, so today's offline stop score (103 of 207) is likely optimistic for the robot (about 90 in the driver's view).

   Needs: code only. Effort: 1 to 2 hours (fork_mask and held_wrist_pictures exist in shift_sensitivity.py; about 7 minutes of inference per run of six folds on the laptop)

   Expected effect: Model choices are made on a score closer to what the robot will see.

   Risk: The imitation is crude for moves over about 16 px; treat differences of a few episodes as noise.

5. One retraining experiment on ki-014: train model E's recipe on pictures brought to the driver's view (same transformation, dropping the 6 % of rows that would move more than 16 px), or with an up-and-down shift augmentation of about 12 to 16 px; judge it by the score of the previous item. In the same run test dropping the lean and hand outputs.

   Why: The stop decision reacts to where the table sits vertically in the picture (content 12 px up: -6.3 points of stop rows), and the wrist's scatter in the data is exactly that; the lean and hand outputs carry no information beyond kinematics.

   Needs: retraining. Effort: one config and a cache variant, then one six-fold run

   Expected effect: Should win back part of the roughly 13 lost stops in the driver's view; direction about unchanged. Unproven until run.

   Risk: A large shift can cut the puck or the line's end out of the picture; imitation artefacts could be learnt; hence only one experiment first.

6. For all new demonstrations use one wrist rule: in the exact replay re-solve every guided waypoint for the same tip and twist with wrist 13.5 and hand -6.1, and let taught starts settle to those values too. Until then, guide the arm so that the wrist does not bend.

   Why: It makes every new training picture, above all every new stop picture, the view the driver will produce; today 22.5 % of start-mark rows and 88 % of taught-start rows are outside the augmentation range, and the per-episode wrist range was a median 7.4 degrees for Emranul against 3.3 for Marshal.

   Needs: code, a robot test, then new demonstrations. Effort: half a day in the recording tool (the Recording workstream's file) plus a 20-minute robot check on three to five replays

   Expected effect: New data need no imitation and no wrist invariance; mid-line and near-end starts then match the driver.

   Risk: The replayed rod leans differently from the guided one (same tip path), so the puck may be touched differently; check with the tool's puck checker before recording in bulk.

7. Camera: do not touch it now. Run camera_creep.py on the frames of every robot session (no extra robot motion needed) and note the number. When retraining, either shift the pictures taken before 8.11.26 19:33 by (+4, +1.5) px at 160 or widen the sideways shift augmentation to 8 px. Raise a fixed camera clip with the professor only if the camera moves again.

   Why: The camera stepped once by about 15 px and has been steady for three days including the test; 65 % of the rows were taken before the step; the models hold direction within 1 point up to 8 px.

   Needs: code only. Effort: minutes per session; one config line when retraining

   Expected effect: Protects against a silent second step; removes a known 4 px offset for sheets 01 to 04.

   Risk: A new mount would change the view of all data; a pure camera turn can be undone by shifting old pictures, a real displacement only approximately.

8. Set-up rule for tests: keep the line within about 46 cm of the base, or decide on the lean limit (58 degrees now, 63.0 the largest demonstrated) after a bench check.

   Why: With the wrist held the rod reaches 58 degrees at 47 to 49 cm reach and the run ends at the first refused step.

   Needs: a robot test only if the limit is raised. Effort: none for the rule

   Expected effect: No surprise refusals on far placements when testing other orientations.

   Risk: Raising the limit lets the block lean further over the puck; only 7 demonstration rows were above 58 degrees.

9. Two-minute look at the hardware with the motors off: which part of the magnet block touches the puck at rod leans of about 0, 20 and 40 degrees, and is that face square to a forward push at hand -6.1?

   Why: The pictures suggest the square block body pushes the puck; then the forearm twist lagging behind turns the face away from the line and the puck slides to a corner, which bears on Emranul's wish to keep the puck in the middle.

   Needs: a robot test (looking only). Effort: 2 to 5 minutes

   Expected effect: Tells whether sideways slip is a contact matter that the twist (or a pusher shape) must solve, not the camera view.

   Risk: None if the arm is only held by hand with the motors off.

### The verifier's challenges to the recommendations

1. Wrist sweep at the robot (pictures at wrist -10, 0, 13.5, 25 at four places on one sheet, 'drop the next items if the stop chance changes by less than 0.1'): as designed it cannot decide anything. Model E has seen all six sheets and gives its own stop rows a stop chance of 0.5 or more in 97 to 100 % of cases whatever the move (v19), so one line end of a known sheet will pass the 0.1 rule by default; the question is about unseen tracks. It needs a newly printed track (make_test_tracks.py) or the fold model that never saw the sheet, and at least three line ends. Risk the analyst missed: the 25-degree pose at a line end leans the rod 51 degrees at 40 cm and 63.5 degrees at 46 cm (v10, calculated), over the driver's own 58-degree limit, and the block's edge comes down towards the paper; the -10 pose near the base leans the rod 18 degrees the other way. It is also a new motion program on the real robot for a problem the robot has not shown: all four runs that reached the line's end (1, 2, 4, 7) stopped by themselves, including sheet 01 where 11 of 19 demonstrated stop pictures had a wrist more than 5.7 degrees below 13.5. If it is done: lean check per pose, checklist, Emranul's go.

2. Scoring every model offline 'in the driver's view' (second column in compare_runs): the column would mostly measure the imitation. A control that changes no view at all loses 8 of 103 right stops, and two ways of filling the picture's edge differ by 1 to 4 episodes, so differences under about 10 episodes mean nothing and model choices could be bent towards models that tolerate smear. A column that needs no imitation exists: score the stop rule separately on the episodes whose last wrist was within 5.7 degrees of 13.5, which are real pictures in the driver's view (92 of 207 episodes; 42 right for the E recipe as recorded, v06).

3. Retraining on pictures brought to the driver's view, judged by that column: circular, because the model would be trained on the imitation and scored on the same imitation, so a gain is built in. Dropping the rows that would move more than 16 px removes mostly stop and taught-start material (88 % of taught-start rows are out of range; 63 of 207 episodes end 10 px or more away), so the stop head would see fewer end pictures. If an experiment is wanted, take only the plain up-and-down shift augmentation, and judge it on the as-recorded scores and on the robot. Dropping the lean and hand outputs in the same run mixes two changes; do it as its own run or not at all.

4. Re-solving all new demonstrations to wrist 13.5 and hand -6.1 in the exact replay: right as a principle (record what the driver will see), but the evidence that the free wrist hurts is weak (direction -0.6 to -0.7 points, stops unknown), so it only makes sense if new demonstrations are recorded anyway. Risks beyond the one named: the replay is then no longer what the hand guided, the rod leans up to about 6 to 10 degrees more at far reach (42 degrees at 40 cm with wrist 13.5 against 36 with wrist 5 and 32 with wrist 0, v10), so the puck is touched differently than in the 211 existing demonstrations; it changes the Recording workstream's file and needs its own robot check. It does nothing for the failures at the bends.

5. Taught start for baseline runs: sound as a clean baseline, but its gain is unproven (runs 2, 4 and 7 succeeded from the same kind of hand-placed start, and the twist lag at the bends is already in the offline predictions: predicted twist 0.50 of the demonstrated). Risks the analyst did not name: (1) a fixed tip pose only helps if the sheet lies on its mark; in runs 6, 8 and 10 the start cap lay about 2 cm from its recorded place (v20), and with a taught pose and a displaced sheet the tip starts beside the puck, worse than a hand-placed tip, so the mark must be checked or re-taped in the checklist first; (2) the demonstrated start has the tip 2.8 to 5.3 cm behind the puck, so the first two to four pictures are an approach, and the puck must not be closer than in the recordings; (3) it runs against Emranul's wish for a robot that works in any orientation and against an exam start that is not known (config notes say the exam sheet lies square to the robot, a lay-out recorded only on sheet 02), so it should stay a baseline, not the plan. The arm moving to the start by itself needs Emranul's go and an approach from above.

6. Keep the wrist at 13.5 and the hand at -6.1, do not follow the lean or hand outputs, do not fit a wrist rule: follows from the evidence and I reproduced each part. Two notes: removing the dead bracket check is an edit of the robot driver and should pass the executor and simulated checks before the next session; and the remaining mismatch (late parts of loops, 55 % of line ends, mid-line starts with wrists of -29 to +9) is untested on an unseen track, so 'keep' is the best choice available, not a proof that nothing is lost.

7. Camera: agree with not touching it and with measuring the fork's place after every session (my independent measure gives the same step). Shifting the 65 % of training pictures from before the step by 4 px is optional and has a small cost of its own: it puts a 4 px smeared column into exactly sheets 01 to 03 and most of 04, which a model could learn as a sheet marker; widening the sideways augmentation is the safer of the two. On the test's own pictures model E's twist did not react to undoing the step (v15), so this has low priority.

8. Set-up rule (line within about 46 cm, or decide on the lean limit after a bench check) and the two-minute look at how the block touches the puck: both follow from the evidence. For the look: motors off means the arm sags, so one person holds the arm the whole time; nothing else to add.

### What the verifier says was missed

1. A control for the stop estimate that changes no view: smearing only the rows a move would push out of the picture already costs 8 of 103 right stops (v09). The analyst's controls (whole picture moved, moved the other way) all move the content and so cannot separate the view from the damage at the picture's edge.

2. The stop question on real pictures: the driver's wrist is the most common stop wrist in the data (92 of 207 stop rows within 5.7 degrees of 13.5), and those real pictures are recognised as stops about as often as pictures with a lower wrist (73.0 % against 78.5 %, five fold-model sets, v07). No imitation is needed to see that the driver's stop view is inside the training data.

3. Real pictures of the same place with different wrists. The analyst's picture check used neighbouring pictures of one episode, where the camera hardly moves; the held-wrist case (camera moves, table shifts) was only calculated. My pairs from different episodes confirm it: 2.83 px per degree measured against 2.98 calculated (v13).

4. Where the sheets lay in the test, left as an open question although the pictures answer part of it: in runs 6, 8 and 10 the puck on the start cap lay 1.9 to 2.4 cm from its place in the STD demonstrations of the same sheet and end (v20). So 'as in plan 0X-01' was only roughly true, and every comparison 'at the same place in the robot's frame' is off by that much.

5. Matching robot and demonstrations by travelled distance instead of by place, which does not depend on the sheet's position: the twist lag on sheets 03 and 04 survives (20 to 95 degrees behind), and it shows that run 8 on sheet 06 was not behind in twist at all, so that run failed for another reason (v18).

6. The demonstrated start is an approach: the tip begins 2.8 to 5.3 cm behind the puck's centre (measured on sheets 03, 04, 06), while the runs began 2.4 to 3.7 cm behind it. The '1.3 to 5.0 cm from the demonstrated start' therefore overstates how far the test was from the recordings; the start twist of up to 10 degrees (demonstrations: 0) is the cleaner difference.

7. Model E's reaction to the camera step on the robot test's own pictures (v15): the analyst tested demonstration rows only. The twist asked at the first bends does not change when the step is undone.

8. A baseline for the lean sign: 73.5 % of the lean labels of 0.5 degrees or more are positive, so the model's 80 to 82 % is 7 to 8 points above answering 'forward' every time. This makes the case against following the lean output stronger than the analyst put it.

9. The lean of the proposed sweep poses and of any other constant wrist (v10): wrist 25 gives 51 degrees at 40 cm and 63.5 at 46 cm; wrist 0 gives 32 and 44; wrist -10 gives 24 and 36 but 18 degrees the other way at 14 cm.

10. The brief's puck size: the puck is 2.4 cm across (config.PUCK_PARAMETERS; 2.1 cm measured in the pictures), not 4 cm. At the puck's distance the picture has about 22 to 27 px per cm at 640, so the 16 px shift augmentation equals about 0.65 cm on the table and 13.5 degrees of wrist about 1.5 cm.

11. Still unchecked by anyone, and not checkable without the robot: whether a wrist of 13.5 at a far line end of an UNSEEN track still stops (the four self-stops of the test were all on trained sheets), and which part of the block pushes the puck.

### The analyst's corrections to the brief

1. 'More than 20 px in 23.3 % of rows when held' holds for the wrist held at each episode's own first-row value. For the driver's constant 13.5 degrees it is 27.1 % (tip) or 27.5 % (picture centre).

2. 13.5 and -6.1 degrees are not just medians: they are the one fixed start pose of all 149 start-mark episodes. The other 61 episodes (taught starts) begin at the least-lean wrist, -29.3 to +8.9 degrees, with the hand at 0.0 (five at 29.6).

3. The lean output's sign is right on 82.5 % (E recipe) and 81.9 % (A recipe) of steps, not about 85 %, and a held wrist alone scores 98.5 % on the same steps, so that number says nothing in favour of following it.

4. frames.csv 'start_end' is wrong for 30 episodes (sheets 02, 05, 06 most; plan 02-01 is really end B). Use evidence/corrected_ends.csv, as the training code does; otherwise 'same placement' comparisons mix the two directions of travel.

5. SQP and SQL on sheet 02 are 'sheet square to the robot, portrait' and 'landscape' (config.py lines 274 to 276 and 319 to 320); run 3 ('lying straight') is of that kind, so it has no single matching placement among the standard plans.

6. Not in the brief: the camera stepped on its tape on 8.11.26 between 17:23 and 19:33 (about 14.6 px right, 6 px down at 640). 65 % of the training rows are from before the step; the robot test saw the view from after it.

7. Not only run 1 started off the mark: all ten runs began 1.3 to 5.0 cm from the demonstrated start tip, with the forearm twisted -9.7 to +9.8 degrees (demonstrations: 0.0).

8. Minor: in chunk runs the saved 'angle_deg' is the chunk's first step while the driver carries out the exact step; on 600 rows they differ by a median 0.8 degrees (90th percentile 3.6), so this does not matter in practice.

### The analyst's open questions

1. Does the real model's stop chance at the line's end change with the wrist? Only the wrist sweep on the robot can replace my imitation (which says 103 to 90 right stops of 207).

2. Which part of the block touches the puck, and is that face square to the push? This decides whether the hand angle and the twist lag explain the puck sliding off the middle.

3. Why did the camera step on 8.11.26 between plans 04-24 and 04-30 (bumped, re-taped)? Emranul may remember; it decides how likely a second step is.

4. Was the sheet on its taped mark in the test, and how was the tip placed behind the puck? The starts were 1.3 to 5.0 cm from the demonstrated start.

5. For the professor: where will the exam sheet lie and from which start pose does the run begin? Beyond about 47 cm reach the held wrist ends the run at the lean limit, and a start away from the mark gives views the start-mark demonstrations do not have.

6. Sheet 06: the robot's tip ran 1 to 2 cm beside every demonstrated tip from picture 2, so my data cannot say whether the view or the start offset made run 8 leave the loop.

7. Would a mid-line or near-end start work with the present driver? Those demonstrations have wrists of -29 to +9 degrees, a mean 16.5 degrees (about 12 px at 160) from the driver's view; untested on the robot.

## Stop and centring (folder `stop_and_centring`)

**Analyst's answer.** The robot does not stop short: in all four runs that reached the end the puck's centre stood 1 to 8 mm PAST the centre of the line's round end, and the two visible misses (runs 1 and 2) were sideways, by 21 and 13 mm. The demonstrations have the same sideways weakness (end fully covered in 104 of 175 measured, puck within 7 mm of the line's middle in about 60 % of pictures), so a final forward push will not help; centring must be taught and measured, and this folder now holds an automatic measure for both requirements.

**Verifier's overall verdict.** Yes for what was measured; partly as a guide for the plan. My own measure (a different puck fit, the line's end taken from single earlier pictures, checked against the arm model's joint log) puts the four stops 1 to 9 mm past the end's centre and the two misses 21 and 13 mm sideways, and gives the same picture of the demonstrations (ends covered in 52 to 60 %, sideways spread 6.2 to 6.4 mm, offsets within 1.7 mm of the analyst's on sheets 01 and 02). But "a final push will not help" holds only while the driver keeps its fixed 1 cm step (that step already adds a full centimetre after the first stop-like picture; a driver that moves the predicted length would stop about 5 mm short by the analyst's own offline table), and the robot's sideways miss in run 2 is not copied from the demonstrations: there the last moves bring the puck towards the line in 44 of 58 cases. All my scripts, outputs and pictures are in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/stop_and_centring/verify/ (start with run_order.txt; file names below are in that folder).

### Verdicts on the claims

1. **confirmed**: Where the robot stopped (finding 5 and headline): runs 1, 2, 4, 7 ended 3.9, 1.9, 7.5 and 1.3 mm beyond the end's centre; run 1 was 21.3 mm right and run 2 13.1 mm left of the line; runs 4 and 7 within 2 mm sideways.

   Measured with my own tools (verify_robot_ends.out.txt: puck from its red top face, the end from 9 or 10 single pictures 3 to 12 steps before the stop; first number camera as fitted, second with the analyst's correction). Beyond the end's centre: run 1 4.3 / 3.8 mm, run 2 0.7 / 0.6, run 4 7.5 / 8.7, run 7 0.1 / 1.8. Sideways: run 1 20.9 / 21.0 mm right, run 2 13.0 / 13.8 left, run 4 5.0 / 1.7 right, run 7 4.7 / 1.3 right. Along the line the two measures agree within 1.5 mm; sideways within 1 mm for runs 1 and 2 and within 2 to 5 mm (the sign changes) for runs 4 and 7. My single views scatter by about plus or minus 3 mm along the line, and views from far back read 3 to 5 mm more 'beyond' than near ones, so run 1 may be only 1 to 4 mm past. No reading puts any stop short. The two puck fits agree to a median 1.0 to 1.4 mm (verify_robot_centring.out.txt). Both measures share the fitted camera, the taught table plane and the arm model; there is still no ruler check.

2. **confirmed**: The driver makes one more full 1.0 cm step between the two stop pictures; the first stop-like picture came 2 to 8 mm short (5.4, 6.9, 1.9, 7.9 mm) and the model asked for 0.21 to 0.25 cm.

   steps.csv of the four runs (verify_labels_and_logs.out.txt, part 2): at the first picture with a stop chance of 0.862, 0.884, 0.897, 0.928 the action is model_step, the planned step 10.0 mm, the tip really moved 9.8 to 10.0 mm, asked length 0.21, 0.25, 0.23, 0.22 cm. model_mode.StopRule counts pictures in a row, so the code agrees. My puck places at that picture: 6.0, 8.3, 0.6, 7.7 mm short; one step later 3.3, 0.5, 8.7, 1.7 mm beyond (verify_robot_last_steps.out.txt). The puck advanced 8.6 to 9.6 mm per step.

3. **confirmed**: The labels cannot say whether the end is covered (finding 1): goal_distance_cm 0.00 and is_at_goal 1 at all 210 stop rows; 808 of 18,522 rows at goal; 259 stop pictures (50 relabelled) in 18,032 training rows of 209 episodes, 1.44 %; operators marked 208 of 210 'puck_in_goal'; last demonstrated move median 0.37 cm.

   Recounted from frames.csv, the episode.json files and training_targets.load_training_frames with each model's own settings from metrics.json (verify_labels_and_logs.out.txt): 210 stop rows, all 0.0 and 1; 808 at-goal rows, median 3 per episode (2 to 13); last move median 0.37 cm, under 0.5 cm in 123, under 0.2 cm in 54; outcome 208 and 2; models A and E: 18,032 rows, 209 episodes, 259 stop rows, 50 relabelled, 1.44 %. record_episodes.goal_labels takes the tip's own stop place as the goal. One episode of episodes.csv (path_01 episode 016, plan 01-14) has no rows: 210, not 211.

4. **confirmed**: How the demonstrations end (finding 4): 175 of 210 measured; mean 2.0 mm short (spread 4.2), sideways spread 6.0 mm; covered (centres at most 7 mm apart) in 104 of 175; sheet 01 only 5 of 18 (median 8.1 mm sideways); within 12 mm in 161.

   The analyst's table recounts exactly (compare_demonstration_ends.out.txt). Independently (verify_demonstration_ends.py on all 210 episodes, compare_demonstration_ends_all.out.txt): on the 156 ends both measured, my place of the puck against the end differs from the analyst's by a median 1.8 mm (90th percentile 4.5 mm; on 9 my simple end finder jumps to another piece of line on the meander and the dashed loop). On the 143 that agree within 5 mm: short mean +2.2 mm, spread 4.3; sideways spread 6.4; covered 77 by mine (54 %) against 86 by the analyst's (60 %). All 156: 52 % against 60 %; with the camera as fitted 46 %. Sheet 01 by mine: 5 of 18 covered, median 8.3 mm sideways. Calculated (recount_claims.out.txt G, normal spreads assumed): a scatter of the measure of 2 mm in each direction would make the true share about 63 %, so 'a little too low rather than too high' is right.

5. **corrected**: The stop picture shows ink ahead of the puck in only 3 of 175, and none of the four robot stop pictures shows ink ahead, so neither the operators' verdict nor the model's picture can tell a covered end from an end 1 cm beside the puck.

   3 of 175 is right for the analyst's threshold of 2 square mm (6 are above zero). But the stop picture of run 1 plainly shows the line and its round end to the left of the puck beside the block (looked at 03_data/model_runs/2026-11-11_172021_check_no_puck/frames/025.jpg): 'ahead' is true, 'cannot tell' is not for a miss of 2 cm. Of the nine least covered demonstration ends (13 to 18 mm) a stub of line shows in the stop picture of 3 and nothing in 6 (look_uncovered_stop_pictures.png, looked at). So: a miss of about 1 cm is usually hidden by the puck and the block, a larger one on the open side is seen. The operators judged by eye from above, not through the camera; by config.py's 12 mm rule 161 of 175 ends pass, which fits their 208 of 210. My own automatic count for this (recount_claims check F) failed: my line mask is blind next to the hand; do not use it.

6. **corrected**: Why run 2 missed (finding 6): the puck went from 5.8 to 13.1 mm left in the last three moves; asked (0.85, 0.24), (0.54, 0.27), (0.20, 0.15) cm pointing 16, 27, 38 degrees left with the line 11 degrees left; the model copies last-moment centring nudges and the fixed 1 cm step turns them into kicks.

   The numbers stand: asked angles and lengths from steps.csv; picture heading -12.2 degrees against my line direction of -1.2 degrees; my puck places 4.7, 5.7, 6.9, 10.3, 13.7 mm left in pictures 20 to 24; and from the arm model alone the tip moved 0.8, 2.9 and 4.6 mm to the left of the line's direction in the last three steps, 8.3 mm (verify_robot_last_steps.out.txt). So the puck was carried about 8 mm further left, to 13 mm, not '13 mm to the left'. The explanation does not stand: in the 58 demonstrations whose puck is more than 7 mm beside the line three pictures before the stop, the tip then moves towards the line in 46 and away in 6 (median 7.2 mm towards) and the puck comes closer in 44 (check_last_moves_direction.out.txt). In run 2 the tip moved away from the line. These were wrong-way short answers made bigger by the fixed step, not copied centring nudges.

7. **confirmed**: The stop output cannot place the puck to 7 mm and a fixed extra push cannot repair that (finding 7): first stop-like picture a median 8.3 mm short (0.6 to 19.7), never in 42 of 175; pushes of 0, 5, 8, 10 mm give 46, 59, 62, 62 % within 7 mm along the line; thresholds 0.7 and 0.9 miss 59 and 89; camera looks down at about 67 degrees and hides about 12 mm beyond the puck.

   Recounted from the analyst's puck places and the saved fold predictions with my own code (recount_stop_table.out.txt): E recipe found in 133, never in 42, median +8.3 mm (0.6 to 19.7); pushes 45, 59, 61, 62 % along the line and 29, 35, 32, 29 % with the sideways offset; thresholds 0.7 and 0.9 never in 59 and 89; A recipe 123 and 52, +6.7 mm. Camera (calculated from the model at the four stops): 63 to 66 degrees, 13 to 15 mm hidden beyond the puck's far edge. My first check by the tip's distance (verify_stop_output.out.txt) read higher (median 11.3 mm) because in some demonstrations the tip moves round a puck that already stands at the end (looked at sheet_04_038_last.png); the analyst's puck-based count is the right one. Two things the same table says that the analyst did not draw out: the present full-centimetre confirming step is already the best fixed push (62 %), and with the next demonstrated picture also stop-like (103 episodes) the puck stands a median 4.6 mm short with 35 still more than 7 mm short.

8. **corrected**: The demonstrations are not centred either (finding 8): 36 sampled episodes, 3,260 pictures, median offset 5.4 mm, within 7 mm in 60 %, over 12 mm in 23 %, over 17 mm in 15 %.

   The count is exact and the measure holds: on sheets 01 and 02 my one-picture offset (puck and line from the same picture) differs from the analyst's by a median 1.7 mm on 273 pictures, and of 80 pictures the analyst puts over 12 mm mine puts 67 over 12 mm and 76 over 9 mm (verify_demonstration_centring.out.txt). By eye on a seeded sample with my own foot circle: of 15 pictures over 17 mm 12 are plainly off the line, 1 is wrong (sheet 06, -61 mm with the puck on the dashed loop), 2 unclear; all 9 'centred' ones sit on the line (look_far_off.png, look_off.png, look_centred.png). The reading needs a correction: only 1,593 of the 3,260 pictures show the puck being pushed; in those the median is 4.5 mm, within 7 mm 68 %, over 12 mm 16 %, over 17 mm 8 %. In 762 pictures the puck stands while the tip moves round it (30 % over 12 mm), as in episode 01-015 pictures 23 to 35, a correction with the rod's one contact point (recount_demonstration_centring.out.txt). 12 of the 36 sampled episodes are puck-aside starts against 35 of 210 in the data. The analyst's own one-picture offset says 51 % within 7 mm where the 12-picture one says 68 % on the same 1,348 pictures, so the share at 7 mm is good to about ten points either way.

9. **corrected**: Robot centring per run (finding 8): run 2 median 1.1 mm and better than all sampled sheet-01 demonstrations; run 4 over 12 mm from picture 20 to 29 with a peak of 20.9 mm at picture 26; run 7 off the line in 17 % (pictures 46 to 65); run 1 drifting right from picture 14 to 22 mm; the robot keeps the puck about as well as the demonstrations.

   Mine (verify_robot_centring.out.txt): run 2 median 0.9 mm on 19 usable pictures; run 1 from 0.9 mm right at picture 12 to 19.8 mm at picture 22 (analyst 1.6 to 19.2). Run 4, where the line is seen beside the puck: over 12 mm from picture 23 to 29, peak 18.1 mm at picture 25, so the stretch is shorter and the peak about 3 mm lower. Run 7: my one-picture tool fails inside the loop (it measures to the wrong branch), but by eye in pictures 52 and 56 the puck stands inside the loop clear of the line (try_measure_run7.png). Like for like (pushed pictures, the analyst's tables): over 12 mm in 29 % (run 1), 0 % (run 2), 24 % (run 4), 25 % (run 7) against 6 %, 6 %, 23 %, 15 % for the sampled demonstrations of the same sheets. 'About as well' holds for run 4, run 2 is better, runs 1 and 7 are worse.

10. **confirmed**: Recoveries (finding 9): for offsets of 7 mm or more the demonstrations change by a median -2.9 mm five steps later (55 % smaller, 23 % larger; 1,166 steps), the robot by 0.0 mm (45 %, 42 %; 64 steps); puck-aside starts go 18, 14, 4.6, 3.4 mm; 35 of 210 episodes start with the puck aside.

   Recounted exactly (recount_claims.out.txt C): 1,166 steps, -2.9 mm, 55 % and 23 %; robot 64 steps, +0.0 mm, 45 % and 42 %; 18.1, 17.8, 14.1, 4.6, 3.4 mm; start kinds STD 71, NE 42, puck aside 35, MID 10. Caveat on 'the robot not at all': the 64 steps mix run 1 (model A, a steady drift away) with model E; in run 4 model E brought the puck back from 20.9 mm to 1.0 mm in six steps (pictures 26 to 32; mine 17.9 to 0.4 mm). The analyst's own 'low confidence' is the right label.

11. **corrected**: The camera correction (finding 3): the fitted forearm camera no longer matches the pictures; shift (-3.1, -4.0) px before the break of 8.11.26, (+12.6, -1.5) after, (+9.9, -2.7) on 11.11.26, roll 1.1 to 1.35 degrees; the camera itself moved; with the correction two views 12 steps apart disagree by 1.8 and 2.5 mm instead of 3.6 and 4.6 mm.

   A test of my own that does not use the line (camera_shift_twist_test_4mm_15deg.out.txt: a standing puck while the forearm twists in place by 15 degrees or more; the puck should not move on the table). Before the move, 44 twists: the puck wanders a median 1.6 mm with the camera as fitted, 1.5 mm with the 'before' correction, 2.8 mm with the 'after' correction. After the move, 12 twists: 3.7 mm as fitted, 2.6 mm with its own correction, 3.5 mm with the other. So a shift did appear after the move and the two periods need different cameras (confirmed), but before the move the fitted camera already fits and its correction is not needed, and after the move the correction repairs only part (2.6 mm against 1.5 mm before). In my other test (camera_correction_check.out.txt: the range of the end's place over single views) the correction brings 6.8 to 5.3 mm after the move and 6.5 to 6.2 mm before; I do not reproduce a halving. The 14 px on the robot's hand from the data audit was not rechecked.

12. **corrected**: Reliability of the measure (finding 2): views agree to a median 1.8 to 3.3 mm; offsets over 12 mm are safely real; one picture alone agrees with the 12-picture measure to a median 1.6 mm; the sign is unreliable near loops.

   In size the measure holds against my independent one: puck places 1.0 to 1.6 mm apart at the median, end places 1.8 mm (90th percentile 4.5 mm), offsets 0.4 to 1.8 mm on runs 1, 2 and 4 and 1.7 mm on demonstration sheets 01 and 02. Corrections: the 1.6 mm is for the robot runs; on the demonstrations the analyst's own summary.out.txt gives 3.1 mm and over 5 mm apart in 449 of 1,298. 'Safely real' does not hold on sheet 06 (dashed; one reading of -61 mm with the puck on the loop), which was pooled into the totals. The sign also flips whenever the tip reverses round a standing puck (episode 01-015, pictures 23 to 30), so only sizes should be reported. The registration figures for three demonstrations (2.5 to 3.3 mm) came from a scratch run with no saved output; my range test fits about 2 mm per direction.

13. **confirmed**: The puck is 2.4 cm across and 2.9 cm tall, not 4 cm; 'covered completely' then means the two centres at most 7 mm apart.

   config.PUCK_PARAMETERS: 0.024 and 0.029 m; line 0.010 m. My top-face fit on eight pictures of runs 2 and 4 misfits by 1.0 to 1.8 px with a radius of 1.2 cm, 6.6 to 7.5 px with 1.6 cm and 12 to 14 px with 2.0 cm (recount_claims.out.txt E). 12 - 5 = 7 mm is right for a puck short of or beside the end; for a puck straight beyond the end the round end stays hidden up to 10.9 mm (calculated), so the 7 mm centre rule is a little strict for overshoot (run 4).

14. **corrected**: By config.py's 12 mm rule 161 of 175 demonstrations and three of the four robot stops pass (run 2 misses by 1 mm); by 'covered completely' 104 of 175 and two of four.

   161 of 175 is right. Of the robot stops two pass the 12 mm rule, runs 4 and 7; run 1 (21.7 mm) and run 2 (13.2 mm) fail (recount_claims.out.txt A). 'Two of four covered' is right by the analyst's rim measure; by the plain 7 mm centre rule only run 7 passes clearly (run 4: centres 7.5 mm apart by the analyst, 8.9 to 9.0 mm by mine).

15. **confirmed**: Calculated shares: part of the line's last centimetre shows in 78 of 175 demonstration stops and more than a third in 29; robot runs 100, 68, 11, 0 %; run 2 ends further from the end's centre than 94 % of demonstrations, run 4 than 63 %, run 7 than 10 %; of 66 episodes over 7 mm aside three pictures before the stop 44 are within 7 mm at the stop.

   My own grid on the analyst's distances (recount_claims.out.txt A and B): nothing visible in 97, some in 78, up to a tenth 21, a tenth to a third 28, more than a third 29; robot 100, 68, 11, 0 %; ranks 94, 63, 10 %. Last three pictures on the reliable ends: 67 over 7 mm aside, 45 of them within 7 mm at the stop; median sideways move 3.8 mm. Added: of 107 within 7 mm three pictures before the stop, 17 are more than 7 mm aside at the stop.

16. **corrected**: Retraining rule (2), dropping pictures over 12 mm whose offset is not shrinking, 'removes roughly a fifth of the pictures'.

   As written the rule takes 365 of the 3,260 sampled pictures, 11 % (check_drop_rule.out.txt); a fifth (23 %) only if every picture over 12 mm goes. Of the 365, 245 (67 %) are followed by the puck coming back within 7 mm during the next 15 pictures, and in 113 the puck stands while the tip goes round it. Only 120 pictures (4 %) are off the line and never brought back.

### The analyst's findings with their evidence

1. (measured, confidence high) The labels cannot say whether the puck covers the line's end. goal_distance_cm is 0.00 and is_at_goal is 1 at all 210 stop rows by construction: record_episodes.goal_labels takes the magnet tip's place at the episode's own stop row as the goal, and is_at_goal means 'tip within 2 cm of that place' (median 3 rows per episode, 808 of 18,522 rows). Models A and E learnt the stop from 259 stop pictures (209 plus 50 relabelled), 1.44 % of 18,032 training rows. The operators marked 208 of 210 episodes 'puck_in_goal'.

   Evidence: stop_and_centring/stop_labels.out.txt part 1 (from frames.csv and training_targets.load_training_frames with the audit cleaning); 02_code/record_episodes.py lines 183 to 214; config at_goal_within_cm 2.0.

2. (measured, confidence medium) An automatic measure now exists for both requirements, from pictures and logged joints only. The puck's foot on the table comes from fitting the outline of a 2.4 x 2.9 cm cylinder through the forearm camera model to the red blob of puck_detector; the line comes from a table map (0.5 mm cells) in which the last 12 pictures vote 'line' or 'paper'. Reliability: views 12 steps apart put the same piece of line a median 1.8 and 2.5 mm apart on robot runs 4 and 7 (worst tenth 2.7 and 5.1 mm) and 2.5 to 3.3 mm on three demonstrations (worst tenth 6.0 to 6.8 mm). So offsets over 12 mm are safely real, shares at the 7 mm limit are rough where the forearm turns, and the sign of the offset is not reliable where the line comes near itself (loops, crossing). One picture alone agrees with the 12-picture measure to a median 1.6 mm, but gives a clean value in only 114 of 165 robot steps (over 5 mm off in 15 of them) and in 1,298 of 2,635 demonstration steps (over 5 mm off in 449): the line under and just ahead of the puck is hidden in its own picture. No check against a ruler exists yet.

   Evidence: picture_measures.py, run_measures.py; measure_check.out.txt (checks 1 and 2), overlays_robot_runs.png and overlays_demonstrations.png (looked at: the fitted foot and the mapped line sit on the visible puck and line); robot_runs.out.txt checks A and B; summary.out.txt last block. The demonstration registration numbers were printed by a scratch run of run_measures.view_registration on plans 02-05, 03-13 and 05-03 and are quoted in summary.py.

3. (measured, confidence medium) The fitted forearm camera (config.FOREARM_CAMERA_PARAMETERS, fitted 6.11.26) no longer matches the pictures, and the camera itself moved on the evening of 8.11.26. The picture shift that makes the pictures of a run agree best on where the line lies is (-3.1, -4.0) px before the break, (+12.6, -1.5) px after it and (+9.9, -2.7) px on 11.11.26, with a roll of 1.1 to 1.35 degrees in all three periods. The far field (the table) therefore moved about 16 px, the same as the 14 px the data audit measured on the robot's own hand and could not attribute. With the correction, two views 12 steps apart disagree by 1.8 and 2.5 mm instead of 3.6 and 4.6 mm.

   Evidence: camera_shift.py; camera_shift_robot_test.out.txt, camera_shift_before_move.out.txt, camera_shift_after_move.out.txt, camera_shift.json (four runs or episodes with large forearm turns per period); measure_check.out.txt check 2; audit_synthesis.md picture_quality-10 for the hand's shift.

4. (measured, confidence medium) How the demonstrations end (175 of 210 ends measured reliably): along the line the puck's centre stops a mean 2.0 mm short of the centre of the round end (spread 4.2 mm; over 7 mm short in 17, over 7 mm beyond in 5); sideways the spread is 6.0 mm (median size 4.2 mm, over 7 mm in 39). The two centres are at most 7 mm apart, which is what the 12 mm puck needs to hide the 5 mm round end completely, in 104 of 175 (59 %); on sheet 01 in only 5 of 18 (median 8.1 mm sideways); within 12 mm in 161. Seen from above, part of the line's last centimetre shows in 78 of 175 and more than a third of it in 29. The stop picture itself shows ink ahead of the puck in only 3 of 175: the puck's own body (2.9 cm tall) and the magnet block hide the end from the camera, so neither the operators' verdict nor the model's picture can tell a covered end from an end 1 cm beside the puck. The measure's own scatter of 2 to 3 mm makes the 59 % a little too low rather than too high.

   Evidence: demonstration_ends.py; demonstration_ends_01_03.csv, demonstration_ends_04_06.csv; summary.out.txt first block; ends_scatter.png. The 'visible share of the last centimetre' is calculated from the two measured distances and the sizes of puck and line.

5. (measured, confidence medium) Where the robot stopped (short of the end's centre along the line / left of the line / centres apart / share of the line's last centimetre not under the puck): run 1 (A, sheet 01) -3.9 / -21.3 / 21.7 mm / 100 %; run 2 (E, sheet 01) -1.9 / +13.1 / 13.2 mm / 68 %; run 4 (E, sheet 02) -7.5 / -0.3 / 7.5 mm / 11 % (line behind the puck, the round end itself is hidden); run 7 (E, sheet 05) -1.3 / +1.9 / 2.3 mm / 0 %. Negative 'short' means beyond the end's centre. The stop chance first passed 0.5 when the puck was still 5.4, 6.9, 1.9 and 7.9 mm short; the driver then made one more full 1.0 cm step (the model asked for 0.21 to 0.25 cm) before the second stop picture. None of the four stop pictures shows any ink ahead of the puck. Run 2 ends further from the end's centre than 94 % of the demonstrations, run 4 than 63 %, run 7 than 10 %.

   Evidence: robot_runs.py; robot_runs.out.txt (the 'at the stop' and 'last steps' lines), robot_runs_ends.csv, robot_runs_ends.png; summary.out.txt. Sideways values moved by about 2 mm between two versions of the camera correction, so read them as plus or minus 2 to 3 mm.

6. (calculated, confidence medium) Why run 2 left part of the end visible: not a short stop but a sideways kick in the last moves. Three centimetres before the end the puck was 5.8 mm left of the line and the model was steering parallel to the line, not back to it. In the last three moves the model's answers were (forward, left) = (0.85, 0.24), (0.54, 0.27), (0.20, 0.15) cm, that is lengths 0.89, 0.60 and 0.25 cm pointing 16, 27 and 38 degrees left of the picture's forward (the line lies 11 degrees left there): the forward part shrinks towards the stop while the left part stays. The driver turned each into a full centimetre, and the puck went from 5.8 to 13.1 mm left. In the demonstrations the last move before the stop is a nudge (median 0.37 cm, under 0.5 cm in 123 of 210) and the puck moves a median 3.9 mm sideways over the last three pictures (of 66 episodes more than 7 mm aside three pictures before the stop, 44 are within 7 mm at the stop): the model copies last-moment centring nudges and the fixed 1 cm step turns them into kicks.

   Evidence: robot_runs.out.txt (run 2 last steps), 03_data/model_runs/2026-11-11_172421_E_sheet01_run1/steps.csv (model outputs and picture_heading_deg; angles calculated); stop_labels.out.txt part 1; demonstration_last_steps_*.csv (a scratch tally of the sideways move over the last three pictures, not kept as a script).

7. (measured, confidence medium) The stop output cannot place the puck to 7 mm, and a fixed extra push cannot repair that. On 175 held-out demonstrations (fold models, the E recipe) the share of pictures with a stop chance of 0.5 or more rises slowly: 1 % at 25 to 45 mm short, 18 % at 15 to 25, 39 % at 10 to 15, 56 % at 5 to 10, 69 % at 0 to 5, 82 % just beyond. The first stop-like picture finds the puck a median 8.3 mm short (10th to 90th percentile 0.6 to 19.7 mm; second seed 10.2; A recipe 6.7) and never comes within the demonstrated pictures in 42 of 175 (A: 52). A last straight push of 0, 5, 8 or 10 mm after that picture would leave the puck within 7 mm of the end's centre along the line in 46, 59, 62 and 62 % of cases, and with the sideways offset as it then was in only 29 to 35 %. A threshold of 0.7 or 0.9 misses 59 and 89 of the 175 ends. Part of the cause is geometry, not training: at the far end of sheet 01 the camera looks down at about 67 degrees, so the table is hidden up to about 12 mm beyond the puck's far edge (2.9 cm / tan 67) and the end vanishes from the picture before it is covered.

   Evidence: stop_labels.py; stop_labels.out.txt part 2; stop_probability_by_shortfall.png; predictions_without_path_0N.npz of 04_runs/gpu_ki014/2026-11-10_resnet18_chunk_stop_weight_25 (and _seed_2) and 2026-11-09_resnet18_final_candidate (and _seed_2), joined by frame_index with demonstration_last_steps_*.csv. The 67 degrees and 12 mm are calculated from the camera model at run 2's last step. This analysis follows the demonstrator's path, not the model's own.

8. (measured, confidence medium) The demonstrations are not centred either, and the robot keeps the puck about as well as they do. In 36 sampled demonstrations (6 per sheet, 3,260 measured pictures) the median sideways offset is 5.4 mm; the puck hides the line's full width (offset at most 7 mm) in 60 % of pictures, the middle of the line is no longer under the puck (over 12 mm) in 23 %, and the puck does not touch the line at all (over 17 mm) in 15 %; no sampled demonstration stays within 7 mm throughout; score 3 episodes: median 4.9 mm, score 2: 7.1 mm. Robot: run 2 (sheet 01) median 1.1 mm, within 7 mm in 92 %, over 12 mm only in its last picture, better than all six sampled sheet-01 demonstrations (medians 3.5 to 5.9 mm); run 4 (sheet 02) median 6.5 mm, 59 %, over 12 mm in pictures 20 to 29 with a peak of 20.9 mm at picture 26 (demonstrations of sheet 02: medians 5.3 to 7.9 mm); run 7 (sheet 05) median 5.0 mm, 53 %, off the line in 17 % (pictures 46 to 65, inside the loop); run 1 (model A) 5.3 mm, 56 %, drifting right from picture 14 to 22 mm at the stop.

   Evidence: demonstration_centring.py (seeded sample: per sheet 3 standard starts, 1 puck-left, 1 puck-right, 1 other); demonstration_centring_01.csv to _06.csv; robot_runs_steps.csv; summary.out.txt second block; centring_distribution.png; robot_runs_offsets.png; robot_runs_maps.png. A contact sheet of 20 demonstration pictures sorted by measured offset was looked at: the large offsets are plainly visible, single small ones can be wrong by about 5 mm.

9. (measured, confidence low) The demonstrations pull an off-centre puck back only slowly, and the robot not at all on average. For pictures with an offset of 7 mm or more, five steps later the offset has changed by a median -2.9 mm in the demonstrations (smaller by 2 mm or more in 55 %, larger in 23 %; 1,166 steps) and by 0.0 mm in the robot's four runs that reached the end (45 % smaller, 42 % larger; 64 steps, two models). In the 12 sampled demonstrations that start with the puck put 1 or 1.5 cm aside, the median offset is 18 mm in pictures 1 to 6, 14 mm in pictures 7 to 10, 4.6 mm in pictures 11 to 15 and 3.4 mm after that: a recovery takes 10 to 15 pictures. All such starts (35 of 210 episodes) are at the start of the line; none is in the last centimetres before the end.

   Evidence: summary.out.txt ('does an offset shrink again', 'the demonstrated recoveries'); start kinds from frames.csv (STD 71, NE 42, puck put aside 35, MID 10, others).

### The analyst's recommendations

1. Move the two measures into the tools: after every model run and after every recorded episode print and log 'end covered: yes or no, centres N mm apart (N short or beyond, N left or right)' and 'puck within 7 mm of the line's middle in N % of pictures, over 12 mm in N %'. Take picture_measures.py and run_measures.py as the start, with the camera correction per period in config.py.

   Why: Every statement about the two requirements has so far been an impression; the operators marked 208 of 210 demonstrations 'puck_in_goal' while 104 of 175 measured ends are fully covered, and the camera cannot see a missed end (ink ahead visible in 3 of 175 stop pictures). Every later change (driver, new demonstrations, retraining) needs this number to be judged.

   Needs: code only. Effort: About one day: move the code, add tests, wire it into model_mode.py's summary and the recording page's review form. It reads pictures and logged joints after the run; nothing changes while the robot moves.

   Expected effect: A number for both requirements for every run and every demonstration, in about 0.1 s per picture. Measured accuracy 2 to 3 mm, good enough to tell 'covered' from run 2's 13 mm.

   Risk: Wrong by more than 5 mm in about a tenth of pictures where the forearm turns; the sign is unreliable on loops; it depends on the camera staying as taped (rerun camera_shift.py after any touch); it fails when the puck is not found.

2. Check the measure once against a ruler at the robot: at five stops (model runs or demonstrations) Emranul measures with a ruler how far the puck's centre is from the centre of the line's end, along and across the line, before anything is moved; compare with the tool's two numbers.

   Why: The measure has only been checked against itself (views agree to 2 to 3 mm) and by eye on overlays. The headline numbers (13 mm in run 2, 59 % of demonstrations covered) rest on it.

   Needs: a robot test. Effort: 15 minutes inside the next robot session; no new code.

   Expected effect: Turns 'medium confidence' into a known error; if the ruler and the tool differ by more than 3 mm at the far end of sheet 01, the camera model needs a proper refit before the measure is used as a gate.

   Risk: None for the robot. A ruler reading on a 2.4 cm puck is itself only good to about 1 mm.

3. Change only the end game of the driver and test three variants with the new numbers, 5 runs each on sheets 01 and 02: (a) as now; (b) once the model's own step is under 0.5 cm, keep the direction of the previous step instead of the direction of the short answer; (c) as (b), with the confirming step 0.5 cm instead of 1.0 cm. Do NOT add an extra forward push and do NOT raise the stop threshold.

   Why: The robot already ends 1 to 8 mm beyond the end's centre, so more push makes it worse. The short answers near the stop point 27 and 38 degrees off (run 2) and 32 degrees off (run 4) and the driver stretches them from 0.2 to 1.0 cm; the demonstrated last move is a 0.37 cm nudge. Offline, a threshold of 0.7 or 0.9 misses 59 and 89 of 175 ends.

   Needs: code only, then a robot test. Effort: Half a day of code and tests in model_mode.py (two settings in MODEL_MODE_PARAMETERS), then one robot session of about 30 runs.

   Expected effect: Calculated from the four runs: with (c) the puck's centre would have ended +0.4, +1.9, -3.1 and +2.9 mm from the end's centre along the line instead of -3.9, -1.9, -7.5 and -1.3; the sideways kick of the last step goes (about 3 of run 2's 13 mm). It would NOT have covered the end in run 2 (still about 10 mm aside) or run 1.

   Risk: Four stops are a small basis: offline the first stop-like picture comes 8 to 10 mm short at the median for model E, so (c) may stop short where (a) does not; that is why all three are run. On a curved end 'the previous direction' can be a few degrees off.

4. Record a first batch of 10 centring demonstrations with the number shown after each episode, and only if they come out clearly better record the rest: per sheet 8 'end zone' episodes (start 6 to 8 cm before the end with the puck 0.5, 1.0 or 1.5 cm left or right; bring it to the middle within the first 3 to 4 cm while the end is still in the camera's view, then push straight and stop on the end) and 6 mid-line recoveries on the curved sheets. Keep an episode only when the tool reports the centres at most 5 mm apart at the stop and the puck within 7 mm of the line after the recovery.

   Why: A model can at best copy its demonstrations: 59 % covered ends, 60 % of pictures within 7 mm, recoveries that take 10 to 15 pictures and exist only at the start of the line. The last 1 to 2 cm before the end are blind for the camera, so the puck must already be centred when the end disappears.

   Needs: new demonstrations. Effort: First batch: 30 minutes. Full set of about 80 episodes (about 1,500 pictures): one to two robot sessions.

   Expected effect: Unknown in size (a guess that it is the largest lever for both requirements). The first batch answers the prior question whether hand guiding with one contact point can do better than the present 5 mm median at all.

   Risk: If the hand cannot centre better, more recording only adds more of the same; then the measure should select the best stretches of the existing data instead. New episodes on the six sheets do not prove anything for an unseen sheet.

5. Retrain with the measure, as separate runs judged on the robot by the new numbers: (1) compute offset and distance to the end for all 18,522 pictures; (2) drop or down-weight pictures where the middle of the line is not under the puck (over 12 mm) and the offset is not shrinking over the next steps; (3) take stop pictures only from episodes whose end is covered (104 of 175 measured), or learn a graded 'distance to the end's centre' in place of the yes or no stop picture; (4) add the sideways offset as an extra output so that the network must see it.

   Why: 23 % of sampled demonstration pictures have the line's middle outside the puck and are imitated as they are; the stop pictures scatter 4 mm along and 6 mm across the true end, and the yes or no stop output is blunt (it rises from 1 % to 69 % over 4 cm).

   Needs: retraining (and the professor's answer for 3 and 4). Effort: About 40 minutes of laptop time for step 1 (0.12 s per picture, light), one to two days of code, then runs on ki-014.

   Expected effect: Unknown; (2) and the covered-ends filter are cheap and stay plain imitation of our own demonstrations. The measure's 2 to 3 mm scatter limits how sharp a graded stop can become.

   Risk: (2) removes roughly a fifth of the pictures, many of them at bends where data are already thin; offline direction scores will not show a gain in centring, only robot runs with the new numbers will.

6. Ask the professor two things on 23.11.26: (1) how 'the puck stops at the end of the line' and 'follows the line' are judged at the exam (puck touching the end, covering it, or its centre within a distance; how far beside the line still counts); (2) whether a number computed from the pictures by ordinary image processing may be used to choose and weight our own demonstrations, and as an extra training target (distance to the end, sideways offset), while the running robot still uses only the camera picture and the trained model.

   Why: config.py assumes 'in the goal' means the puck's centre within one puck radius (12 mm) of the end: by that rule 161 of 175 demonstrations and three of the four robot stops pass (run 2 misses by 1 mm); by Emranul's rule (covered completely, 7 mm) 104 of 175 and two of four. The second question decides whether recommendation 5 parts 3 and 4 are allowed.

   Needs: the professor's answer. Effort: Five minutes in the meeting; bring ends_scatter.png and robot_runs_ends.png.

   Expected effect: Fixes the target the next eight weeks aim at and which training variants are allowed.

   Risk: A strict answer to (2) leaves only new demonstrations and selection by hand.

7. Put the camera correction into config.py (or refit the forearm camera on pictures of 9 to 11.11.26) and note the camera period per episode: shift (-3, -4) px before the 8.11.26 break, (+13, -2) px after it, (+10, -3) px on 11.11.26, roll about 1.2 degrees.

   Why: The recording tool's puck map and any picture measure use this model; uncorrected, two views 12 steps apart put the same piece of line 3.6 to 4.6 mm apart instead of 1.8 to 2.5 mm, and absolute table positions are off by about 5 to 7 mm.

   Needs: code only. Effort: One hour with camera_shift.json as the source; a proper refit with forearm_camera_model.fit_camera_parameters is half a day.

   Expected effect: Table positions from the picture right to about 2 to 3 mm again.

   Risk: config.py is shared and other sessions read FOREARM_CAMERA_PARAMETERS; the change needs Emranul's go and a new 'version' entry. It does not touch the camera or its focus ring.

### The verifier's challenges to the recommendations

1. Recommendation 1 (build the two measures into the tools in about a day, wired into model_mode.py and the recording page): the order is wrong and the wiring is risky. The measure has only been checked against a second picture measure that shares the same camera model, table plane and arm model; the ruler check (recommendation 2) should come before it is built in or used to accept and reject anything. Do not put it inside model_mode.py: a fault in new code at the end of a run sits next to the lift and the log; run it as a separate script on the run folder afterwards. The recording page belongs to the Recording and data session (HOW_WE_WORK.md), so that part goes through the orchestrator. Report sizes only (the sign flips), pushed pictures apart from standing ones, and leave sheet 06 out until the dashed line is handled. 'One day' is optimistic for about 2,000 lines that call a private method of PuckDetector.

2. Recommendation 2 (ruler at five stops, 'risk: none for the robot'): needed, but the end is hidden under the puck at a good stop, so a ruler cannot reach it. A method must be written first: a phone photo from straight above with a ruler in view, or a pencil line round the puck on a spare print before it is lifted. Nobody should reach under a held arm (project rule: motors off before anyone touches the arm); measure after the lift with the arm clear, and put it in the session checklist. Five stops can show a bias over 2 mm but not the scatter; take ten or more, on two sheets, with one stop at the far end of sheet 01 and one with the forearm turned.

3. Recommendation 3 (three end-game variants of the driver, 5 runs each on sheets 01 and 02): (b) has no support offline. Within three pictures of the stop the previous row's direction is no better than the model's current one (median error 25 degrees for both, better in 52 %; check_keep_previous_direction.out.txt), and as worded the rule would act on every asked step under 0.5 cm, which is 3,018 held-out rows, 2,742 of them at bends and twists far from the end, where six of ten runs failed. If tried, tie it to a raised stop chance. (c): by the analyst's own offline table a 5 mm step after the first stop-like picture gives 59 % within 7 mm along the line against 62 % for the present 10 mm; the gain 'calculated from the four runs' comes from sheets the model was trained on. Five runs per cell cannot separate differences of 2 to 4 mm when stops scatter by 4 to 6 mm. Above all, decide the step-length rule first: if the driver moves the predicted length (as the step-length analysis proposes), the confirming step is about 0.2 cm, the puck stops about 5 mm short at the median and 35 of 103 still more than 7 mm short (recount_stop_table.out.txt), and a short final push of about 5 mm becomes the thing to test. 'Do not raise the threshold' is supported.

4. Recommendation 4 (centring demonstrations, kept only when centres are at most 5 mm apart): the gate is tighter than the measure. Two independent measures of the same 147 stops disagree on pass or fail at 5 mm in 24 (16 %), and only 59 of 175 existing ends (34 %) pass, so two of three episodes would be thrown away at the present skill. Use 7 mm, the requirement itself, and only after the ruler check. Recoveries in which the tip goes round a standing puck cannot be driven by the present fixed 1 cm step; more of them help only once the driver can make small steps. The session needs a written checklist and a taught start for the new start places. The first batch of ten with the number shown is a sound first step.

5. Recommendation 5 (retrain with the measure): rule (2) as written would mostly remove the beginnings of corrections that work (245 of the 365 pictures it takes), exactly the behaviour Emranul wants more of; if anything is dropped, drop only the 4 % that are off the line and not brought back within about 15 pictures. Rule (3), stop pictures only from covered ends, cuts the stop examples to about 104 of 209 when stop rows are already 1.44 % of the data and the two-in-a-row rule fires at the end in only 106 of 209 held-out episodes and never in 71 (check_early_and_never_stops.out.txt); it risks more runs that never stop, and since a miss of about 1 cm is hidden in the stop picture it may not teach what is hoped: the miss is made sideways two or three pictures earlier. Rule (4) needs an offset for all 18,522 pictures from a measure that fails on sheet 06 and at crossings.

6. Recommendation 6 (questions for the professor): right to ask, but correct the numbers first: by the 12 mm rule two of the four robot stops pass (runs 4 and 7), not three. Bring a top-view photo of a stop so that 'covered' is judged on what a person sees, and ask how a stop that never comes, or comes early, is judged, because on unseen sheets that is the more frequent failure offline.

7. Recommendation 7 (put the camera correction into config.py in an hour): by my line-free test the fitted camera still fits every picture before the move (1.6 mm against 1.5 mm with the correction), so no correction belongs there; after the move the analyst's shift helps (3.7 to 2.6 mm) but leaves the camera worse than it was before the move, so it is half a repair. config.py is shared and the recording tool's puck map reads it; a patch would move every table position by about 5 mm for all sessions. Keep the correction inside the measuring script for now and do the proper refit on pictures from 9.11.26 on, with Emranul's go.

### What the verifier says was missed

1. The end result depends on the driver's fixed step. All four stops were made with a full centimetre added after the first stop-like picture; nothing was said about what happens under a predicted-length step, although config.py already carries that option and the step-length analysis proposes it. The stop tests must be repeated after any change of the step rule.

2. Pushing was not separated from standing. Half of the sampled demonstration pictures (1,667 of 3,260) show a puck that is not being pushed 4 mm or more; a third of the off-line pictures are corrections in which the tip goes round a standing puck. The robot's pictures are almost all pushes, so the comparison 'the robot keeps the puck about as well' mixed two different things.

3. Which way the demonstrated last moves go was not checked. In 46 of 58 demonstrations with the puck beside the line the tip moves towards the line before the stop; in run 2 the robot's tip moved away. The demonstrations therefore already show the wanted end-game behaviour; the model's short answers near the stop point the wrong way.

4. Stop behaviour over whole episodes on unseen sheets. The analysis looks at the last eight pictures only. Over all pictures the two-in-a-row rule (E recipe, held out) fires at the end in 106 of 209 episodes, never in 71 (29 of 45 on sheet 02, 15 of 25 on sheet 06) and earlier in 32 (tip a median 4.3 cm before its stop place, 11 of them on sheet 05). For the exam's unseen line, stopping at all is a bigger risk than covering to 7 mm.

5. The four robot stops were on sheets the models were trained on, and only three are model E (run 1 is model A, started 1.5 cm beside the line). One of the three missed sideways. There is no repeat of any run, so nothing is known about the scatter from run to run.

6. The measure fails on the dashed sheet 06 (no line in the gaps) and its numbers were pooled with the rest; the 35 ends that could not be measured and the 9 where my finder and the analyst's disagree by over 10 mm sit mostly on sheets 04 and 06.

7. The puck does not sit a fixed 18.5 mm ahead of the magnet tip. By the analyst's own puck places the distance falls from 20 mm to 13 mm along run 2 as the rod leans from about 26 to 40 degrees, and is about 40 mm at the end of run 4, where the leaning rod pushes the puck's top edge. The tip's place cannot stand in for the puck's, and a 'push N mm' rule or the driver's workspace guard that uses the tip is off by up to 2 cm.

8. How to measure a covered end by hand. The proposed ruler check has no method for an end that lies under the puck, and no top-view photo was planned, although 'seen from above' is exactly Emranul's requirement.

9. The camera's hidden zone was calculated for the far end of sheet 01 only. I get 63 to 66 degrees and 13 to 15 mm at all four stops, but demonstrations with the forearm turned or the arm nearer the base were not looked at; where the camera looks down more steeply the end is seen longer, which may matter for where new end-zone demonstrations are recorded.

### The analyst's corrections to the brief

1. The puck is not 4 cm: config.PUCK_PARAMETERS and the memory notes say 2.4 cm across and 2.9 cm tall, and the outline fit agrees (a radius of 1.2 or 1.3 cm fits best in 20 of 22 pictures; a radius of 2.0 cm fits about 7 times worse; measure_check.out.txt check 1). With a 2.4 cm puck on a 1 cm line with round ends, 'covers the end completely' means the two centres at most 7 mm apart.

2. The brief asks for goal_distance_cm and is_at_goal at the stop rows as if they measured the goal: they are 0.00 and 1 in all 210 stop rows by construction (the goal is the tip's own stop place), so they say nothing about the line's end.

3. The driver's stop rule is described correctly but one consequence is missing: between the two stop pictures the robot makes one more full 1.0 cm step in the direction of the model's short answer (0.21 to 0.25 cm asked in the four runs). The first stop-like picture came 2 to 8 mm short of the end's centre; the final place is 1 to 8 mm beyond it.

4. Run 2 ('on the line, a bit of the line's end still visible'): on the line for 23 of 24 measured pictures (median offset 1.1 mm), but the last three moves carried the puck 13 mm to the left; about two thirds of the line's last centimetre were not under the puck (calculated from the measured 13 mm, which is good to 2 to 3 mm). It was a sideways miss, not a short stop.

5. Run 1 ('drifted about 1 to 2 cm right'): 21 to 22 mm right of the line at the stop; the line's end lay entirely beside the puck.

6. Run 7 ('a little issue with keeping the puck in the middle'): the puck's foot did not touch the line at all in 17 % of the pictures (about pictures 46 to 65, inside the loop); it recovered and the end was covered.

7. Run 4: 'just before the middle' is confirmed and sized: over 12 mm off from picture 20 to 29 of 41, peak 20.9 mm at picture 26.

8. frames.csv holds 210 episodes with pictures, not 211: one episode of episodes.csv (path_01 episode 016, plan 01-14) has no rows.

9. The data audit left open whether the camera or the robot's hand moved on 8.11.26: the table's picture moved by about the same 16 px as the hand, so the camera moved (camera_shift_*.out.txt).

### The analyst's open questions

1. How will the exam judge 'at the end' and 'on the line'? By the 12 mm rule assumed in config.py most stops pass; by Emranul's 'covered completely' about 6 in 10 demonstrations pass.

2. May a number computed from the pictures be used to select or weight our own demonstrations and as an extra training target (question for the professor)?

3. Does the sideways miss at the end repeat? There is one run per sheet and model; 5 runs per sheet with the new numbers are needed before any driver change is judged.

4. How accurate is the measure really? It has no ruler check yet, and its residual error (views disagree by 2 to 3 mm, worst tenth up to 7 mm) is not explained by a picture shift, roll, focal length, lens term, camera position or table height (all tried).

5. Can a person guiding the arm by hand, with one contact point on the puck, centre better than the present 5 mm median at all, or is that the limit of the method? Ten test episodes with the number shown would tell.

6. 35 of 210 demonstration ends could not be measured reliably (no bare paper seen beyond the end, or the mapped line not 8 to 12.5 mm wide); whether these ends differ from the 175 is unknown.

7. Sheet 06's dashes are said to end square (config comment for the test tracks); the measure treats its last dash like a round end, so its 'covered' rule is slightly too kind there. Not checked on a picture.

8. The offline stop analysis follows the demonstrators' paths, on which the last picture spacing is a 0.37 cm nudge; how the stop output behaves on the model's own approach is known only from four robot stops.

9. The centring sample is 36 of 210 demonstrations (6 per sheet, seeded); the offset of all 18,522 pictures has not been computed yet (about 40 minutes of laptop time).

## Orientation coverage (folder `orientation_coverage`)

**Analyst's answer.** Orientation was not what failed on 11.11: all ten runs used lay-outs the demonstrations hold, and the demonstrations cover every travel direction at mid reach. The real orientation gap is the start (148 of 209 episodes begin from one arm pose with the line leaving within about 37 degrees of straight away, and with the puck about 2.7 cm clear of the block), and run 3 failed because the driver pushed 1 cm per step where the model, like the six square-portrait demonstrations, asked for a twist nearly in place.

**Verifier's overall verdict.** Partly. Three of the four parts of the headline hold on my own recomputation (the ten runs used lay-outs the demonstrations hold, every travel direction is covered at mid reach, the thin place is the start), but the fourth does not: run 3 did not fail because of the fixed 1-cm step. The model asked for 0.36 to 0.49 cm steps straight up the picture (4.06 cm straight away from the robot over ten pictures, measured from steps.csv) while the block already touched the puck and the line left 53 degrees to the left, so with the asked lengths the puck would still have left the line (calculated: about 1.2 to 1.6 cm off after four pictures, about 3 cm after ten); the cause is the touching start, which the demonstrations never show. All my scripts and their outputs are in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/orientation_coverage/verify/ (v1 to v16, each with its .out.txt).

### Verdicts on the claims

1. **refuted**: Headline and finding 14: run 3 failed because the driver pushed 1 cm per step where the model, like the six SQP demonstrations, asked for a twist nearly in place

   The numbers are right, the reading is not. Measured from D:/AI_Project/03_data/model_runs/2026-11-11_172743_E_sheet02_run1/steps.csv (v2_run3.out.txt): first ten pictures asked 0.36 to 0.49 cm (median 0.41), twist median -3.47 degrees, 30 degrees of twist reached at picture 8 after 8.0 cm done against 3.29 cm asked. But the asked direction was straight up the picture (median -7.6 degrees, range -27.6 to +2.2) while the line lay 57 degrees to the left; the ten asked steps sum to 4.06 cm straight away from the robot (+1.7 degrees on the table). Model E's own five-step forecast, re-run on the logged pictures (v12_after_the_start.out.txt), runs 1.6 to 2.9 cm straight up the picture (fifth point -11 to +20 degrees). That is a slow straight approach, not a twist in place. The block touched the puck (2.35 cm), so with the asked lengths the puck is carried straight ahead too: calculated 1.16 cm off the line at picture 3, 1.56 at picture 4, 3.2 at picture 9 (logged answers; this simple puck model reads about 0.5 to 1.5 cm high against the pictures). As done, the puck seen in the pictures was 2.2 cm off at picture 3 and 5.1 cm at picture 8. In the SQP demonstrations (raw index, own red-blob finder) the tip first went 1.0 to 3.4 cm to the right round the puck in free space; in plans 02-13, 02-14 and 02-16 it travelled 9.9 to 13.1 cm and the forearm turned 65 to 92 degrees before the puck moved 1 cm (02-15 about 45 degrees after overshooting past the puck, 02-18 about 21 degrees, 02-17 a false start with the forearm first turned to +45 the wrong way). The step length made the failure faster, it did not cause it.

2. **confirmed**: Finding 12: run 3 was not an unseen lay-out; the sheet lay as in the six SQP demonstrations from end B; line leaves the puck at +53 degrees

   Own line finder on the fitted camera (v2_run3.out.txt): run 3 +53.1 degrees (spread 4.2, 1,799 pixels); same finder on plan 02-18 +52.5 and plan 02-14 +66.2. An exactly square portrait sheet from end B gives about +70 at the cap and about +60 over 2.5 to 6 cm (calculated from the traced line). I looked at frames 000, 005 and 010 of run 3 and at first pictures of plans 02-13, 02-15 and 02-16: same lay, same paper edges. Puck at the start: run 3 (18.8, -1.5) cm, demonstrations (18.8 to 21.6, -0.6 to +1.4). Run 4: my finder gives +2.8 degrees (spread 19), near plan 02-01's -7.6; the analyst's printed +50 for run 4 (their spread 58) is noise and should not be quoted.

3. **confirmed**: Finding 1: start codes and counts (STD 72, NE 42, offsets 35, turned 25, MID 10, SQL 10, SQP 6, X1 6, X2 3; 209 episodes, 18,032 pictures); all six SQP episodes from end B

   Recounted from the audit-cleaned index (v1_basics.out.txt): identical counts. SQP/SQL meanings read in config.py lines 274 to 320. End B checked without the audit's table, from the shape of the tip path (v7_bends_flips_layouts.out.txt): all six SQP paths run 70 to 109 degrees to the left first and then bend right, ending at about (41 to 44, 1 to 3) cm; the set-up picture plan_02-13.png draws end A (line leaving 71 degrees right), which was not what was recorded. No square-portrait episode leaving to the right exists.

4. **confirmed**: Findings 2 and 3: 148 start-mark episodes from one arm pose (14.0, -1.5) cm with the forearm straight; line leaves within 20 degrees in 127, 20 to 40 in 15, more only in the six SQP; 37 lay-outs; sheets a median 5.9 degrees from the plan; caps at 19.2 cm where the pictures draw 16.4

   Pose measured (v1_basics.out.txt): tip x 14.00 to 14.06, y -1.49 in raw row 0 of all 148; joint 4 -0.7 to +1.5 in the first kept rows; taught starts 61, reach 16.0 to 37.7 cm, |joint 4| over 10 in 33. The 127/15/6 split is the analyst's fit; recounted from their CSV (largest outside SQP 36.9 degrees) and checked two ways: the fitted start cap lies a median 1.26 cm (90th percentile 2.34) from the puck my own finder sees in 84 first pictures (v3_start_gap.out.txt), and SQP/SQL agree with the traced line's geometry (+70 portrait from B, +19 and -23 landscape). Pucks' x median 19.2 cm by my measure. Two cautions: the tip's own first five steps go more than 45 degrees off radial in 34 of 148 episodes (sidesteps to offset pucks and SQL), and the lay-out groups are loose (fitted turn spans 32 to 51 degrees inside the five biggest groups), so '37 distinct lay-outs' is a rough count.

5. **confirmed**: Finding 4: demonstrations start with the puck a median 5.2 cm from the tip (about 2.7 cm of free travel), only 11 of 147 within 1 cm of touching; all ten robot runs started touching or nearly (2.3 to 3.6 cm)

   Own red-blob finder on the same fitted camera (v3_start_gap.out.txt): start-mark demonstrations median 5.47 cm (10th to 90th percentile 3.73 to 8.66; 106 of 148 found), distance while pushing straight 2.68 cm (546 pictures), so about 2.8 cm free; under 3.5 cm: 8 episodes; SQP 5.6 to 7.7 cm. Per episode my value differs from the analyst's by a median 0.08 cm, never more than 1 cm (161 episodes). Robot runs 2.35 to 3.65 cm (8 of 10 found). I also saw it by eye: clear gap in plans 02-13, 02-15, 02-16, 06-01; block on the puck in runs 3 and 8.

6. **confirmed**: Findings 5 to 7: travel direction from radial 42.0 / 40.3 / 17.7 %, left 3,323 against right 2,995, every mid-reach cell at least 134 steps; 19 thin cells of 60 with 4.2 % of steps; joint 4 -119.3 to +118.9, beyond 60 in 36.5 %, 90 in 10.3 %, 110 in 2.2 %; picture frame 45.7 / 20.4 / 26.0 / 8.0 %

   Own code on the audit-cleaned index (v1_basics.out.txt): 42.1 / 40.2 / 17.7 % over 15,697 three-step paths; left 3,323, right 2,995; the 5 x 12 cell table is identical, smallest mid-reach cell 134, 375 moving steps beyond 42.5 cm, 19 cells under 100, 4.2 %; joint 4 the same figures (123 pictures, 0.68 %, lie beyond the driver's 117); picture frame 45.7 / 20.4 / 25.9 / 8.0 %. Twisting rows: 5,085, median step 0.65 cm, 36 % under 0.5 cm.

7. **confirmed**: Finding 8: with joint 4 at 117 and the wrist held, the picture's top turns at most 106 / 112 / 116 / 123 degrees from radial at 12 / 25 / 35 / 45 cm; usable reach ends at about 47.5 cm with the forearm straight

   Calculated two ways (v4_physics.out.txt): recording tool's heading rule 106.3 / 112.1 / 116.0 / 123.2; fitted camera alone (which table direction runs straight up the picture at the tip's pixel) 106.0 / 111.5 / 115.5 / 122.0. Rod lean passes 58 degrees at 47.6 cm with the forearm straight; with joint 4 at 117 it stays under 58 up to 49.5 cm, so the 47.5 cm limit holds only for a nearly straight forearm. Measured in the demonstrations: 966 pictures at |joint 4| of 100 or more, median 105.0, largest 125.4, slope -0.933. Tip pixel (313, 367 to 380).

8. **confirmed**: Findings 9 and 10: robot's parts cover 8.5 % of the picture; a line straight back to the base lies 57 to 72 degrees off the picture's top at best; demonstrators looked along such lines in 0 to 1 % of rows, with |joint 4| only 69 to 73

   Paths beyond 135 degrees from radial (2,776 rows, v4_physics.out.txt): looking along 0.7 %, more than 90 degrees off the picture's top 67.5 %, |joint 4| median 71.0; lean there median 28 degrees, wrist median 13.5, so pushing towards the base with the driver's held wrist was demonstrated (893 rows at 35 to 42.5 cm, 111 beyond). 180 - 123 = 57 and 180 - 108 = 72 follow from the table above. Robot parts: 9.8 % by my median-picture mask (v11_robot_parts.out.txt; an upper estimate, it takes in the puck's blob), block top at about row 345 in the pictures I looked at. Not recomputed: the 105-degree visibility limit and the puck-detector shares.

9. **corrected**: Finding 11: flips and tight bends are common and need twisting nearly in place; sheets 03 to 06 bend 25 to 53 degrees per cm at the sharpest, 15 to 30 cm faster than 10 degrees per cm; practice tracks at most 8 to 15

   Counts right (v7_bends_flips_layouts.out.txt): 157 flips in 103 episodes, 87 on sheet 04, median 127 degrees over 24 rows and 19.4 cm. But that is 6.4 degrees per cm, inside the driver's 10 per cm; only 42 of 157 flips (27 %) were made over less than the 12.7 cm the driver would need. So tight bends need twisting in place, most flips do not; for flips the weak point is the model's small twist, not the step. Bends: 15.6 / 28.3 / 15.8 / 16.7 cm faster than 10 degrees per cm on sheets 03 to 06 (confirmed); with a 1-cm window the sharpest are sharper than stated: 77 degrees per cm on sheets 03 and 04, 27 on 05, 48 on 06. Practice tracks 10.5 to 18 degrees per cm by the same window (rule: radius 35 mm, tracks.json).

10. **corrected**: Finding 13: run 3's first 13 pictures match no demonstration picture well (best correlation median 0.75; run 4 0.84; SQP among themselves 0.96), which shows an unseen start state

   I get the same two numbers (0.751 and 0.840, v13_nearest_pictures.out.txt), but the same measure on all ten runs gives 0.776 for run 1, 0.815 for run 2 and 0.813 for run 7 (all reached the end), 0.800 for run 8, 0.749 for run 6 and 0.666 to 0.698 for runs 5, 9 and 10; demonstration episodes against all other episodes score 0.934 (10th to 90th percentile 0.898 to 0.979). So 0.75 does not single out run 3: every robot run looks less like the demonstrations than they look like each other. The unseen start state stands on the direct measures instead: puck 2.35 cm from the tip against 5.6 to 7.7 cm, forearm straight with the line 57 degrees off the picture's top. Pushing sideways in the picture near the base with a straight forearm appears in 243 pictures, nearly all free sidesteps before the puck is touched (v14_start_state_rarity.out.txt).

11. **confirmed**: Finding 15: offline, models trained without sheet 02 score 0.99 on standard starts, 0.96 to 1.00 on turned, 0.90 to 0.95 on SQL, 0.62 to 0.72 on SQP first rows (always-forward 0.74)

   Own scoring of the saved predictions (v5_offline.out.txt): standard 0.989 to 0.994, turned 0.965 to 1.000, SQL 0.895 to 0.947, SQP 0.615 / 0.718 / 0.718 / 0.718 on 39 pictures, always-forward 0.744; taught 0.751 to 0.818. Caveat: the SQP misses come mostly from plan 02-17 (1 or 2 hits of 6, the false start) and 02-18; those rows are the free approach, not line following. In the rows after the manoeuvre the same models do well: SQP rows 10 to 29 score 0.77 to 0.91, and the 42 SQP pictures with the forearm turned 45 degrees or more within 25 cm of the base score 0.74 to 0.98 (E recipe 0.98 and 0.88) (v12_after_the_start.out.txt).

12. **confirmed**: Finding 16: offline the score follows coverage weakly; reach matters more than direction (0.90 to 0.94 near the base, 0.73 to 0.76 at 35 to 42.5 cm, 0.56 to 0.63 beyond)

   Own numbers (v5_offline.out.txt): straight away 0.854 to 0.888, back towards the base 0.647 to 0.745; reach 0.905 to 0.939, then 0.726 to 0.760, then 0.564 to 0.629. I also answered the analyst's open question: the fall with reach is not an effect of the easy starts. It stays with the first ten rows of every episode left out (0.86 to 0.92, then 0.72 to 0.76, then 0.56 to 0.63) and inside paths that run straight away (0.95 to 0.97, then 0.80 to 0.83, then 0.50 to 0.61 beyond 42.5 cm). Rank correlations not recomputed.

13. **confirmed**: Finding 17: runs 1, 2 and 4 never left the demonstrated combinations; runs 5, 6 and 9 left them from step 16 to 18; run 8 stayed inside

   Same neighbour rule recomputed with own code (v9_runs_coverage.out.txt): the table is identical (run 3: 5 steps under 20 neighbours from step 23; run 5: 18 from 18; run 6: 15 from 16, joint 4 only -12 to +17; run 9: 13 from 16; runs 1, 2, 4, 8: 0). One selective point: run 7, which reached the end, also took 31 steps more than 60 degrees off the picture's top, so that count does not separate failed from good runs. The asked step lengths of the brief's scout also check out (run 6 median 0.31 cm, half under 0.3).

14. **corrected**: Findings 18 and 19: on the present start mark only 141 of 576 placements are possible and 18 of 24 shape ends by the demonstrations' rule; with the paper's centre on a cross at 31 cm all 576 are possible (tip 15.1 to 46.9 cm)

   Centre cross confirmed: 576 of 576, tip reach 14.7 to 47.4 cm by my calculation (v10_centre_cross_and_twist.out.txt); practice_03 comes within 0.1 cm of the 47.5 cm limit. The mark numbers are lower bounds: the rule 'paper on the base' marks sheet 03 from end A on the mark as impossible, yet that lay-out was recorded 25 times. In the analyst's own fits 12 recorded episodes have paper inside the base's square (9 of 25 for sheet 03 end A, 7 by more than 0.5 cm, 2 by more than 2 cm; v7_bends_flips_layouts.out.txt). At least 19 of 24 shape ends fit by the demonstrations' rule. The 298 comfortable and 211 flip-free counts were not recomputed.

15. **unverifiable**: Finding 20: Plan P (36 episodes, 6.4 robot hours) and Plan C (24 episodes, 4.6 hours) raise filled start cells from 15.8 to 21.2, 21.8 or 27.0 of 44; travel cells only from 44.1 to 45.0

   The arithmetic is right but the gain is true by construction: a start cell counts as full at 4 episodes and each new placement of 2 episodes adds 0.5, so 12 placements add 6 cells. It is not a measure of what the model learns. By the same rule the SQP start is already full (6 episodes), and it is the weakest start offline (0.62 to 0.72) and the one that failed on the robot. Time: my median start-to-start time inside one sheet is 5.8 minutes (4.4 to 7.3 by sheet, 192 gaps; v7_bends_flips_layouts.out.txt) against the analyst's 7; 36 x 5.8 + 18 x 10 minutes gives 6.5 hours, so the hours stand. Picks 5 and 9 of Plan C are the same lay-out (sheet 02 square portrait from end A) 6 cm apart (seen in 05_plan_course_sheets.png).

16. **confirmed**: Finding 21: a forearm twist turns the picture about the tip's pixel (313, 371), measured (309, 372), 33 px from the tip's pixel and 136 px from the centre, by 0.91 times the heading change

   Different method (ORB key points and a RANSAC rotation fit on 199 recorded pairs where the tip moved under 0.15 cm and joint 4 turned 5 degrees or more; v6_rotation.out.txt): the picture turns about (319, 369), a median 27 px from the tip's pixel (313, 376) and 132 px from the centre, nearer the tip's pixel in 98 % of pairs, by 0.92 times the calculated heading change, scale 0.998.

17. **confirmed**: Finding 22: a plain rotation about the tip's pixel is within 0.2 to 2.0 px near the tip but 10 to 35 px off at 12 cm; about the centre the tip moves 23 px at 10 degrees and 68 px at 30; 10 % of the table part is empty at 10 degrees, 25 % at 30

   Calculated with the fitted camera (v15_rotation_error.out.txt): within 5 cm of the tip 0.2 to 1.2 px up to 30 degrees and 35 cm reach (5.1 px at 45 cm); largest error at about 12 cm from the tip 10.7 / 16.0 / 37.4 px at 30 degrees for 15 / 25 / 35 cm reach. Tip shift 23 px and 68 px (plain geometry, 132 px lever). Empty share 12.6 % at 10 degrees and 30.6 % at 30 by an upper-bound count that treats all rows from 330 down as robot (v6_rotation.out.txt), in line with the analyst's 10 and 25 %.

18. **confirmed**: Finding 23: the step and chunk labels turn with the picture, but the forearm twist has no rule (correlation of the twist with the bend ahead -0.45)

   Measured (v10_centre_cross_and_twist.out.txt): correlation -0.45 over 14,745 rows; mean twist -3.10 degrees before a left bend of more than 30 degrees, +3.21 before a right bend. The reasoning follows: a turned picture would need a made-up twist label, which is no longer a demonstration.

19. **confirmed**: Corrections to the brief: puck 2.4 cm not 4 cm; tracks.json lists 7 of 20 tracks and disagrees with the SVG files; step_length_rule 'predicted' exists in config.py; run 4 lay as plan 02-01 was recorded (end B)

   config.py PUCK_PARAMETERS diameter_m 0.024. tracks.json lists 7 tracks while the folders hold 6 + 4 + 10 SVG files; practice_03 is 253.1 mm from (97.4, 244.7) in the SVG against 264 mm from (69.2, 251.4) in the JSON, practice_06 358.1 mm against 253 mm (own parse). config.py MODEL_MODE_PARAMETERS holds step_length_rule 'fixed' with 'predicted' available and a stall rule; no robot run is logged after 17:55. Plan 02-01's path bends right (end B) and run 4's joint 4 ran from 8 to 86 degrees, the same way.

### The analyst's findings with their evidence

1. (measured, confidence high) What the start codes mean and how many there are: SQP and SQL are sheet 02 laid with its edges square to the robot, portrait and landscape, start cap on the start mark, so the line leaves at its own angle. NE is a taught start 4 to 8 cm before the end, MID a taught start before a hard spot (3 cm before sheet 03's sharp turn, sheet 04's middle hairpin), X1 and X2 taught starts 5 cm before the crossing (first and second pass). In the cleaned training set (209 episodes, 18,032 pictures): STD 72, NE 42, puck-beside-the-line offsets 35, sheet turned 10 or 15 degrees 25, MID 10, SQL 10, SQP 6, X1 6, X2 3. After the audit's end corrections all six SQP episodes start from end B; no square-portrait episode from end A exists.

   Evidence: config.py RECORDING_PAGE_PARAMETERS start_codes and start_setups; orientation_coverage/01_placements.out.txt (tables per sheet, end and kind); evidence/corrected_ends.csv

2. (calculated, confidence medium) How the sheets really lay: a rigid fit of each sheet's traced line to the recorded tip path gives 37 distinct lay-outs in the 197 full-length episodes (sheet 01: 3, 02: 11, 03: 6, 04: 7, 05: 7, 06: 3). The set-up pictures show the plan, not the lay: the sheets lay within a median 5.9 degrees of the plan (20 of 146 start-mark episodes more than 15 degrees off, 05-01 and 05-02 about 33 degrees because they were laid square), and the start caps lay a median 19.2 cm in front of the base axis where the pictures draw 16.4 cm.

   Evidence: 01_placements.out.txt, 01_placements_groups.csv, 01_placements_per_episode.csv, 01_placements_map.png. Fit quality: median tip-to-line distance 0.96 cm per episode. Independent check: the puck seen by the puck detector in the first picture lies a median 1.3 cm (90th percentile 2.5) from the fitted start cap in 112 episodes (03_run3_first_pictures.out.txt, part D)

3. (measured, confidence high) How a line starts: every start-mark demonstration (148 of 209) starts from one arm pose, tip at (14.0, -1.5) cm, 14.1 cm from the base axis, forearm straight (joint 4 between -0.7 and +1.5 degrees). The line leaves the start cap within 20 degrees of straight away in 127 of them, 20 to 40 degrees in 15, and more only in the six SQP episodes (+66 to +72 degrees). No demonstration starts with the line leaving sideways beyond 72 degrees or back towards the robot. The 61 taught starts lie 16 to 38 cm out, 33 of them with the forearm already turned more than 10 degrees (-89 to +70).

   Evidence: 01_placements.out.txt (fitted line direction at the start cap, tip start place); 02_coverage.out.txt section 6; 02_first_five_steps.csv

4. (measured, confidence high) The demonstrations start with free travel, the robot runs did not: in the first picture the puck lay a median 5.2 cm from the tip (10th to 90th percentile 3.7 to 8.4; a touching puck lies about 2.5 cm away), so the block travels about 2.7 cm before it touches. Only 11 of 147 start-mark demonstrations began within 1 cm of touching. All ten robot runs began touching or nearly (2.3 to 3.6 cm).

   Evidence: 03_run3_first_pictures.out.txt part D and 03_start_gaps.csv (puck detector plus fitted camera on the first kept picture of every episode and run); touching distance from 04_physics.out.txt part C (median 2.56 cm on forward pushes)

5. (measured, confidence high) Travel direction on the table relative to the arm's radial direction is covered all round at mid reach: 42.0 % of the three-step paths run within 45 degrees of straight away from the base, 40.3 % sideways and 17.7 % back towards the base; left and right are balanced (3,323 against 2,995). Between 20 and 42.5 cm reach every 30-degree direction holds at least 134 moving steps from at least 20 lay-outs and five or six sheets.

   Evidence: 02_coverage.out.txt sections 1 and 4, 02_coverage_cells.csv, 02_coverage.png

6. (measured, confidence high) The thin places are the two ends of the reach: within 12 to 20 cm the seven sideways and backward directions hold only 20 to 75 moving steps each, and beyond 42.5 cm every direction holds 8 to 67 (375 in all). That is 19 of 60 cells and 4.2 % of all moving steps.

   Evidence: 02_coverage.out.txt section 4 (thin cells: under 100 pictures or under 3 lay-outs, a stated choice)

7. (measured, confidence high) Joint 4 and the picture frame: joint 4 runs from -119.3 to +118.9 degrees, median absolute value 46.5; beyond 60 degrees in 36.5 % of pictures, beyond 90 in 10.3 %, beyond 110 in 2.2 %. In the picture the path points within 30 degrees of the picture's top in 45.7 % of rows, 30 to 60 degrees off in 20.4 %, sideways (60 to 120) in 26.0 % and towards the bottom (beyond 120) in 8.0 %.

   Evidence: 02_coverage.out.txt sections 2, 3 and 5

8. (calculated, confidence high) Physics, how far the camera can be turned: with joint 4 at 117 degrees and the wrist held at 13.5 as the driver holds it, the picture's top can be turned from the radial direction by at most 106 degrees at 12 cm reach, 112 at 25 cm, 116 at 35 cm and 123 at 45 cm. The demonstrations agree (slope -0.93 degrees of picture turn per degree of joint 4; median 105 degrees where |joint 4| is 100 or more, largest 125). With the forearm straight the rod's lean passes the driver's 58-degree limit from 48 cm on, so the usable reach ends at about 47.5 cm, not 49.5.

   Evidence: 04_physics.out.txt part A, 04_heading_and_lean.csv, 04_physics.png (arm model through solve_with_wrist_fixed and forearm_camera_heading_rad; 966 demonstration pictures at |joint 4| of 100 or more)

9. (calculated, confidence medium) Physics, what the camera sees: round the tip the puck and at least 10 cm of line are in view for travel directions up to about 105 degrees off the picture's top; from about 120 to 135 degrees on the puck is behind the robot's own parts, which cover 8.5 % of the picture (nothing above row 340 of 480). So the camera can point straight along a line that runs up to 106 to 123 degrees from radial; a line straight back to the base lies 57 to 72 degrees off the picture's top at best, still in view. No table direction forces a blind push if the forearm is twisted far enough; a blind push arises when the forearm lags.

   Evidence: 04_physics.out.txt parts B and D, 04_view_by_direction.csv (fitted camera project_points; mask of the robot's parts from the median of 1,500 cached pictures)

10. (measured, confidence medium) What the demonstrators did with lines that run back to the base (150 to 180 degrees from radial): they looked along the line in 0 to 1 % of rows and travelled more than 90 degrees off the picture's top in 62 to 78 %, with a median |joint 4| of only 69 to 73 degrees. The puck detector still finds the puck in at least 97.5 % of the pictures of every direction, and where the tip moves towards the picture's bottom the puck is not in front of it (1.0 cm ahead at 4.1 cm distance): those rows move the block round the puck, they are not blind pushes. Blind pushing was never demonstrated.

   Evidence: 04_physics.out.txt part C (300 pictures per 30-degree bin through PuckDetector) and part D (three-step paths)

11. (measured, confidence high) Flips and tight bends are common and need twisting nearly in place: the forearm flips from beyond 60 degrees on one side to beyond 60 on the other 157 times in 103 episodes (87 times on sheet 04), a median 127 degrees over 24 rows and 19.5 cm of tip travel. On sheets 03 to 06, 15 to 30 cm of each line bend faster than 10 degrees per cm (sharpest 25 to 53 degrees per cm, radius 1.1 to 2.3 cm), more than the driver can twist while it moves 1 cm per step. The 5,085 rows that twist 6 degrees or more move the tip a median 0.65 cm (36 % under 0.5 cm), which confirms the scout's 0.65. The six practice tracks bend at most 8 to 15 degrees per cm (radius 3.9 to 7.0 cm): they hold no corner like sheet 03's or hairpin like sheet 04's.

   Evidence: 04_physics.out.txt parts D and E, 04_line_bends.csv (traced course sheets 2026-11-06_sheet_measurements.json; practice SVG files)

12. (measured, confidence medium) Run 3 was not an unseen lay-out: the sheet lay as in the six SQP demonstrations from end B (plans 02-13 to 02-18): square, portrait, line leaving the puck to the left and bending right. Measured in run 3's first picture the line leaves the puck at +53 degrees on the table (2.5 to 6 cm from the puck); the fitted SQP lay-outs start at +66 to +72 and bend right, while SQL starts at +11 to +19 (end A) or -16 to -31 (end B) and the standard lay-outs within about 23 degrees.

   Evidence: 03_run3_first_pictures.out.txt part B; 03_sqp_start_sequences.png; 01_placements_groups.csv (lay-out 02B-2, turn -88 degrees)

13. (measured, confidence high) What was unseen in run 3 is the start state: the block touched the puck (2.3 cm) with the forearm straight and the line 57 degrees off the picture's top. In the SQP demonstrations the puck lay 5.6 to 7.6 cm ahead, and four of the six turned the forearm 49 to 92 degrees within the first 16 rows, mostly before or while reaching the puck (the rows that twist 3 degrees or more move the tip a median 0.61 cm, 42 % under 0.5 cm). Run 3's first 13 pictures match no demonstration picture well (best correlation a median 0.75; run 4: 0.84; SQP demonstrations among themselves: 0.96).

   Evidence: 03_run3_first_pictures.out.txt parts A, C and D; 03_run3_nearest.csv; 03_run3_nearest.png; 03_sqp_start_sequences.png

14. (measured, confidence high) In run 3 the model asked for what the demonstrations show and the driver did something else: over the first ten pictures the model asked for a median step of 0.41 cm with -3.5 degrees of twist per step, so 30 degrees of twist over 3.3 cm of asked travel (9.4 degrees per cm). The driver moved 1 cm every step, 8 cm for the same 30 degrees (3.9 degrees per cm), and the puck was off the line by about picture 4. Four of the six SQP demonstrations reached 30 degrees of twist after 4.6 to 6.5 cm of tip travel, most of it before the block touched the puck.

   Evidence: D:/AI_Project/03_data/model_runs/2026-11-11_172743_E_sheet02_run1/steps.csv; 03_run3_first_pictures.out.txt part C

15. (measured, confidence medium) Offline, a model trained without sheet 02 copes with turned and landscape starts but not with the steep portrait start: over the first ten rows of an episode the fold models score (path within 45 degrees) 0.99 on standard starts, 0.96 to 1.00 on sheets turned 10 or 15 degrees, 0.90 to 0.95 on SQL, but 0.62 to 0.72 on SQP, where always-forward scores 0.74. The steep start is learnt only from the six SQP episodes themselves.

   Evidence: 07_offline_score_by_coverage.out.txt (saved predictions of four fold runs: 2026-11-09_resnet18_final_candidate and seed 2, 2026-11-10_resnet18_chunk_stop_weight_25 and seed 2; SQP sample is 39 pictures of 6 episodes)

16. (measured, confidence medium) Offline the score follows coverage only weakly, and reach matters more than direction: cells with at most 50 training pictures score 0.62 to 0.68 against 0.70 to 0.88 elsewhere (475 of 13,766 scored pictures; rank correlation +0.14 to +0.25). By direction from radial the models score 0.85 to 0.89 straight away and 0.63 to 0.77 back towards the base (always-forward 0.00 to 0.27 there). By reach the score falls from 0.90 to 0.94 at 12 to 20 cm to 0.73 to 0.76 at 35 to 42.5 cm and 0.56 to 0.63 beyond. Rare lay-outs score like common ones (0.75 to 0.78 against 0.76 to 0.80).

   Evidence: 07_offline_score_by_coverage.out.txt, 07_score_by_cell.csv (four fold runs, held-out sheets only; near reach is also where the easy starts are, so reach and difficulty are mixed)

17. (measured, confidence medium) The robot runs in coverage terms: runs 1, 2 and 4 never left the demonstrated combinations of forearm twist, travel direction and reach (0 steps with under 20 demonstrated neighbours, median 154 to 368 neighbours). Runs 5, 6 and 9 left them from step 16 to 18 on (13 to 18 steps), run 6 with joint 4 only between -12 and +17 degrees while it travelled up to 142 degrees from radial. Runs 3, 5, 6, 9 and 10 took 14 to 37 steps more than 60 degrees off the picture's top; runs 1, 2 and 4 none. Run 8 stayed inside (0 steps), so its failure is not an orientation matter.

   Evidence: 02_coverage.out.txt section 7 (neighbours: demonstrated moving steps within 10 degrees of joint 4, 15 degrees of direction from radial and 5 cm of reach; 10.7 % of the demonstrations' own steps have under 20)

18. (calculated, confidence medium) Where a sheet may lie: with the start cap on the present start mark only 141 of 576 placements (24 start directions, 12 shapes, 2 ends) are possible: the straight sheet only with its line within 30 degrees of straight away, curved sheets up to 90 to 135 degrees to one side, none leaving back towards the robot (a paper corner meets the base or the tip comes within 12 cm). With the paper's centre on a cross 31 cm in front of the base axis all 576 are possible (tip 15.1 to 46.9 cm). Of those, 298 are comfortable (line leaves within 120 degrees of radial, at most 35 % in the back sector beyond 140 degrees); a median 18 to 25 % of a curved line lies in the back sector, and only 211 need no forearm flip.

   Evidence: 05_recording_plan.out.txt sections 1, 2 and 4; 05_candidates.csv (traced lines laid by calculation; driver limits from MODEL_MODE_PARAMETERS; table size not checked)

19. (calculated, confidence medium) Laid by the demonstrations' own rule (start cap on the mark, line leaving straight away) 18 of the 24 shape ends of the six course sheets and six practice tracks are possible; the tip then stays between 17.2 and 40.8 cm from the base axis, inside the well-covered cells, with 15 % of the line length in the back sector. For the exam this rule needs no new orientation data.

   Evidence: 05_recording_plan.out.txt section 2 (table 'The lay-out rule of the demonstrations')

20. (calculated, confidence medium) The recording plan for orientation coverage (2 episodes per placement; only square or rule lay-outs, only comfortable ones; picked by coverage gain per robot hour): Plan P, the six practice tracks in 3 placements each, 36 episodes, about 1,260 pictures, 6.4 robot hours; Plan C, the six course sheets in 2 new placements each, 24 episodes, about 1,124 pictures, 4.6 hours. Filled start cells rise from 15.8 to 21.2 (P), 21.8 (C) or 27.0 (both) of 44 reachable; filled travel cells only from 44.1 to 45.0 of 60. Nearly all the new coverage is in starts that leave 23 to 115 degrees off straight away.

   Evidence: 05_recording_plan.out.txt sections 3 and 5 to 7, 05_plan.csv, 05_plan_course_sheets.png, 05_plan_practice_tracks.png (weights and the 10 minutes per new lay-out are choices or guesses; 7 minutes per episode is the measured median start-to-start time, 4.6 to 8.0 by sheet)

21. (measured, confidence high) Picture rotation, the right centre: a forearm twist turns the table in the picture about the tip's picture point, not the picture's centre. Calculated: (313, 371) of 640 x 480, that is (78, 93) of the network's 160 x 120, 131 pixels below the centre. Measured on 160 recorded pairs where the forearm twisted in place: the picture turns about (309, 372) (median), 33 px from the tip's pixel and 136 px from the centre, by 0.91 times the calculated heading change.

   Evidence: 06_rotation_augmentation.out.txt parts A and C, 06_twist_pairs.csv (ECC alignment of the picture above the robot's parts; all 160 pairs aligned at correlation 0.9 or more)

22. (calculated, confidence medium) Picture rotation, what goes wrong: a plain rotation about the tip's pixel is within 0.2 to 2.0 px (640-px scale) of the true picture for table points within 5 cm of the tip up to 30 degrees and 35 cm reach, but 10 to 35 px off at 12 cm from the tip. Rotated about the picture's centre instead, the tip itself moves 23 px at 10 degrees and 68 px at 30 degrees, and the block and jaws turn with it, which cannot occur. After turning only the table content, 10 % of the table part of the picture is empty at 10 degrees, 25 % at 30 and 32 % at 45; the robot's parts (8.5 % of the picture) must stay as recorded.

   Evidence: 06_rotation_augmentation.out.txt parts A and B, 06_rotation_example.png

23. (measured, confidence medium) Picture rotation, the labels: the step and the five chunk points turn with the picture by the rotation angle and the stop label stays, but the forearm twist has no rule. The demonstrators turn the camera into the bend ahead (correlation of the twist with the bend 5 to 10 rows ahead -0.45; on bends the picture's top runs 19 to 49 degrees ahead of the path), so a turned picture needs a twist that no demonstration supplies. The twist, lean and hand losses would have to be switched off for rotated pictures, and then rotation cannot teach the very thing a turned start needs, turning the camera towards the line.

   Evidence: 06_rotation_augmentation.out.txt part D

### The analyst's recommendations

1. Repeat run 3's lay-out (sheet 02 square, portrait, from end B) three ways with model E, one run each and a second where it fails: (a) as on 11.11; (b) with step_length_rule 'predicted' (already in config.py since 11.11, 18:18); (c) the same with the tip set about 5 cm behind the puck, as in the demonstrations. Then the six sheets as on 11.11 with the predicted rule.

   Why: In run 3 the model asked for 0.41 cm steps with 3.5 degrees of twist (30 degrees over 3.3 cm) and the driver made it 8 cm; the SQP demonstrations twist nearly in place and start 5.6 to 7.6 cm behind the puck. This separates the driver's fixed step from the touching start.

   Needs: robot test. Effort: about 1 hour of a session (12 to 15 runs at 2.5 to 3 minutes), with a checklist

   Expected effect: The start of the square lay-out should hold; it also shows whether sharp bends on sheets 03 and 04 improve. No offline number can stand in for it.

   Risk: The model may twist on the spot without progress (the stall rule ends a run after 30 pictures); its short steps are partly hand jitter, so runs get slower; a new rule changes what correction data would mean.

2. Set a start rule for sheets whose line does not leave straight away: place the tip by hand about 5 cm behind the puck on the line's extension with the forearm already turned so that the picture's top points along the line; write it into the robot checklist and the exam procedure.

   Why: The camera can be pointed along any line up to 106 to 116 degrees from radial; the driver keeps a hand-set twist; 33 of the 61 taught-start demonstrations begin with the forearm turned. It turns a sideways start into a picture of the kind the data hold in every direction (at least 134 steps per direction at 20 to 42.5 cm).

   Needs: robot test. Effort: no code; a minute per start; tested inside the orientation ladder

   Expected effect: Removes the need for the model to turn in place at the start, the one thing it has six examples of.

   Risk: A turned forearm close to the base with the puck ahead is shown only by taught starts, whose first rows score no better than always-forward offline (0.75 to 0.82 against 0.79); hand-set twists are inexact; motors off before anyone touches the arm.

3. Run an orientation ladder before recording anything for orientation: sheet 01 and sheet 02 with the paper's centre on a taped cross 31 cm in front of the base axis, every 45 degrees (16 placements, easiest first, listed in 05_recording_plan.out.txt section 8), one run each, stopping a sheet after two failures in a row.

   Why: It measures at which turn the present model stops following, which no offline score shows; the eight comfortable placements test sideways starts, the rest the back sector.

   Needs: robot test. Effort: about 2 hours, one session, with a checklist (a guess: 4 minutes to tape and set each start)

   Expected effect: Decides whether the orientation part of the recording plan is needed at all, and which turns to record.

   Risk: Back-sector placements start 42 to 44 cm out with the line running towards the robot: the puck can go under the arm or off the sheet (Esc at hand), and the far reach is the weakest zone offline (0.56 to 0.63). Table space round the cross is assumed, not checked.

4. Decide the exam lay-out rule now and have it confirmed: every sheet laid by the demonstrations' rule (start cap about 19 cm in front of the base axis where the caps really lay, line leaving straight away, the other end if a paper corner meets the base), and ask the professor on 23.11 whether Emranul and Marshal lay the unseen sheets and set the start themselves.

   Why: Under this rule the tip stays between 17.2 and 40.8 cm, inside the well-covered cells, and standard and 10 or 15 degree turned starts score 0.96 to 1.00 offline on unseen sheets. Start and orientation were allowed on 26.10; it has not been confirmed for the unseen sheets.

   Needs: professor. Effort: one question and one line in the checklist

   Expected effect: Takes orientation out of the exam risk; 'any orientation' becomes a robustness extra.

   Risk: If the professor lays the sheets, square or freely turned lay-outs become necessary and Plans P and C move up; 6 of 24 shape ends do not fit on the mark by this rule.

5. If printing is allowed, record Plan P first: the six practice tracks in 3 placements each, 2 episodes per placement (36 episodes): each track first by the demonstrations' rule, then the two square placements listed in 05_plan.csv (order by gain per hour). Start as in the demonstrations (puck about 5 cm ahead); where the line leaves at an angle, one episode turning in place and one with the forearm pre-turned.

   Why: New shapes help more than more episodes of known sheets (+2.5 points of step within 45 per doubling of sheets against +1.7 per doubling of pictures), and the 12 extra placements add starts leaving 23 to 115 degrees off straight away, which the data lack.

   Needs: new demonstrations. Effort: about 6.4 robot hours (3 sessions), then retraining (minutes on ki-014); also needs the professor's yes to printing (question 15)

   Expected effect: Filled start cells 15.8 to 21.2 of 44; six new shapes. The offline gain is likely within the 3.3-point spread of single runs; the ladder repeated afterwards is the real measure.

   Risk: Sideways starts need a taught start per lay-out and take longer; the practice tracks hold no sharp corners; 6 episodes per track is far below the 20 the earlier plan gave a new track.

6. Record Plan C (the six course sheets in 2 new square placements each, 24 episodes, order in 05_plan.csv) only where the ladder shows failures, or in place of Plan P if printing is refused. It includes the missing square-portrait lay-out of sheet 02 from end A and sheet 01 laid across in front of the robot in both directions.

   Why: Each course sheet was recorded in only 3 to 11 lay-outs, all with the line leaving within 37 degrees of straight away except the six SQP episodes.

   Needs: new demonstrations. Effort: about 4.6 robot hours (2 sessions), then retraining

   Expected effect: Filled start cells 15.8 to 21.8 (27.0 together with Plan P); travel cells hardly change (44.1 to 45.0 of 60).

   Risk: More episodes of known shapes are the weaker lever; pick 10 (sheet 04 from end B) has 13 cm in the back sector and two flips; time taken from correction rounds.

7. Before printing, draw two or three more practice tracks with corners as tight as the course sheets (radius about 1.2 to 2.5 cm instead of the rule's 3.5 cm) and write tracks.json again from the SVG files.

   Why: The practice tracks bend at most 8 to 15 degrees per cm; sheets 03 to 06 bend 25 to 53 degrees per cm over parts of 15 to 30 cm, and the 11.11 failures were at exactly such bends. tracks.json lists 7 of the 20 tracks and matches the SVG file for only one.

   Needs: code only. Effort: about an hour (one setting in make_test_tracks.py and a rerun)

   Expected effect: New-track demonstrations then also cover hairpins and corners, with twisting nearly in place.

   Risk: Validation and test tracks must be redrawn by the same rule or they no longer match the practice set; the sealed test tracks must stay unseen while doing so.

8. Do not use plain picture rotation. If rotation is tried at all, as a later GPU experiment only: turn the table content by at most about 10 degrees about the tip's pixel (78, 93 at 160 x 120), paste the robot's parts back, fill the empty part, turn the step and chunk labels by the same angle and switch the twist, lean and hand losses off for rotated pictures.

   Why: The measured centre of rotation is the tip's pixel, 136 px from the picture's centre; 25 % of the table part is empty at 30 degrees; the twist label cannot be turned.

   Needs: code only. Effort: half a day of code and checks, a GPU run of minutes; judged on held-out sheets and on the ladder

   Expected effect: Uncertain and probably small; it cannot replace demonstrations of turned starts.

   Risk: Fill and mask artefacts the model may latch onto; wrong twist habits if the twist loss stays on; whether changed pictures still count as own demonstrations is a question for the professor.

9. Keep the measured lay-out of every episode (01_placements_per_episode.csv: turn, start cap, lay-out group) beside the index and score robustness by measured lay-out and by the puck's start distance, not by start kind.

   Why: The sheets lay a median 5.9 degrees and 3.3 cm from the set-up pictures, 20 episodes more than 15 degrees; the puck's start distance differs between demonstrations (median 5.2 cm) and robot runs (2.3 to 3.6 cm).

   Needs: code only. Effort: an hour

   Expected effect: Future comparisons and the exam checklist rest on how the sheets really lay.

   Risk: The fits are good to about 5 degrees and 1 to 2.5 cm, less for short near-end episodes.

### The verifier's challenges to the recommendations

1. Recommendation 1 (repeat run 3 three ways): variant (b), the predicted step length with the block still touching the puck, does not follow from the evidence as a fix. By the logged answers and by model E's own five-step forecast the tip still goes straight ahead through the puck (calculated: 1.2 to 1.6 cm off the line after four pictures, about 3 cm after ten). The stated effect 'the start of the square lay-out should hold' is supported at best for variant (c). If the hour is short, skip (a), which is already known, and spend it on (c) and on the pre-turned start. Risk not named for (c): with the tip straight behind the puck the model must copy a go-round manoeuvre that the six SQP demonstrations show in at least four different ways (two of them scored 1 and 2), so (c) may fail too and that would not speak against the pre-turned start.

2. Recommendation 2 (pre-turned start rule): worth testing, and my offline check supports it (models that never saw sheet 02 score 0.74 to 0.98 on the 42 SQP pictures with the forearm turned 45 degrees or more within 25 cm of the base). But its stated reason is overstated: the 'at least 134 steps per direction' holds at 20 to 42.5 cm, while this start lies at 17.5 to 18.3 cm reach with joint 4 at -82 to -97 degrees (calculated for run 3's lay-out, v16_pre_turned_start_pose.out.txt), where the demonstrations hold only 85 to 165 pictures, and no demonstration starts with the forearm turned 40 degrees or more within 25 cm of the base. Risks on the robot: a hand-set twist of about 90 degrees per sheet with the motors off is slow and easy to get wrong at the exam (Emranul's own question 18 says so); a line that bends on in the same direction meets the 117-degree limit within about 20 to 35 degrees; the driver's settling move shifts the arm when the hand-set wrist differs from 13.5 degrees, so keep the 5 cm gap to the puck.

3. Recommendation 3 (orientation ladder on a centre cross): with the paper's centre fixed at 31 cm the turn and the reach change together. Every back-sector placement starts 41.5 to 44.4 cm out, where the offline score is lowest for any direction (0.50 to 0.61 even for paths straight away beyond 42.5 cm), only 428 pictures exist and the lean limit ends a run at 47.6 cm. A failure there cannot be read as an orientation limit. Run the eight comfortable placements first (about one hour) and treat the far ones as reach tests. The table edge was not checked (paper out to about x = 49 cm); the project rule is that the robot stays away from the table edge, so measure the table before taping.

4. Recommendation 4 (exam lay-out rule, line leaving straight away): follows from the evidence (standard starts 0.99 offline, run 4 good and run 3 bad on the same sheet). Two things are not said. It reverses the team's own note of 7.11.26 in config.py (start_codes comment: at the exam the sheet lies square to the robot, portrait or landscape), which Emranul and Marshal should decide knowingly. And question 9(a) in D:/AI_Project/05_meetings/requests_to_professor.md was asked for the exam day and answered yes on 26.10.26 (start and orientation are ours, taping allowed), so the question for 23.11 is a short confirmation, not an open risk.

5. Recommendation 5 (Plan P, 36 episodes, 6.4 hours): it does not address what failed on 11.11 (sharp bends and the dashed loop): the practice tracks bend at most 10.5 to 18 degrees per cm against 27 to 78 on sheets 03 to 06, so recommendation 7 must come first or the hours go into easy shapes. Its first six placements add no orientation coverage by its own table. Two episodes per new start is a third of what the SQP start has, and that start is the weakest. Recording 'one episode turning in place and one pre-turned' adds more of the manoeuvre that fails when the puck is touched: record one start procedure only, the one the robot will use, and write it down before recording.

6. Recommendation 6 (Plan C, 24 episodes, 4.6 hours): the gain in start cells is true by construction and travel cells hardly move (44.1 to 45.0 of 60), so by the analyst's own numbers this is 4.6 hours for starts that a start procedure can remove. Keep it strictly behind the ladder, and note that two of its twelve placements are one lay-out 6 cm apart.

7. Recommendation 9 (keep the measured lay-out of every episode): fine, but use the fitted turn of each episode, not the lay-out group label: the fitted turn spans 32 to 51 degrees inside the five biggest groups, and short near-end episodes have unreliable turns.

8. Order of the whole list: the analyst's own headline says orientation was not what failed, yet the list asks for about 14 robot hours on orientation (1 + 2 + 6.4 + 4.6). What the evidence supports for orientation is cheap: decide the lay-out rule (no robot time), test the pre-turned start and the start with a gap (inside one hour), and the eight comfortable ladder placements (about one hour). The recording plans should wait for those results and rank behind the bend and dashed-line failures.

### What the verifier says was missed

1. The asked direction in run 3. The analyst compared only the asked length and twist with what the driver did. The asked direction (straight ahead, through a puck the block already touched) and model E's five-step forecast (1.6 to 2.9 cm straight up the picture) were not looked at; they change the cause of the failure and the worth of the predicted-length rule for this start.

2. The six SQP demonstrations disagree with each other. Three sidestep 1.4 to 2.5 cm to the right and turn the forearm 65 to 92 degrees before the puck moves, one drives 9 cm past the puck and pushes it back towards the robot, one starts by turning the forearm 45 degrees the wrong way and returns to the start pose, one pushes with only about 21 degrees of twist. The model's 0.4 cm straight steps with a small twist look like the average of these. No written rule says how to approach a puck whose line leaves at an angle; one is needed before any new recording.

3. All ten robot runs, not only run 3, match the demonstrations poorly on the analyst's own nearest-picture measure: 0.67 to 0.84 against 0.93 between demonstration episodes (run 1, which reached the end, 0.78). This points to a general difference between the recording days and the test day (light, exposure, the touching start, the table), which bears on every run and was not examined here. The measure is crude (grey 40 x 30 correlation), so this is a pointer, not a finding.

4. The pre-turned start begins in a thinly demonstrated state: 153 pictures (0.8 %) lie within 20 cm of the base with |joint 4| of 60 or more, 38 with 80 or more, and none of the 209 episodes starts with the forearm turned 40 degrees or more within 25 cm of the base (the 29 episodes that start turned all start 26 to 38 cm out). The analyst names the risk but backs the rule with mid-reach numbers.

5. The fall of the offline score with reach is real and not an effect of easy starts (it stays with the first ten rows left out and inside straight-away paths). The analyst left this as an open question; it matters for the ladder, whose hard placements all start far out.

6. The feasibility rule for placements contradicts the recordings: it forbids sheet 03 from end A on the mark, recorded 25 times, and 12 recorded episodes have fitted paper inside the base's square. Either the paper really lay against or beside the base by up to about 2 cm or the fits are off by that much; in both cases the 'possible' counts on the mark are too strict by about 1 to 2 cm.

7. 'Start cells filled' was used to rank the recording plans without a check that a filled cell is a learnt start. The one full non-standard cell (SQP, six episodes) is the counter-example in the analyst's own data.

8. Most forearm flips do not need twisting in place (median 6.4 degrees per cm against the driver's 10); only 27 % were made faster than the driver can turn. The limit for flips is the model's twist being about half the demonstrated size (brief, scout numbers), which a step-length rule does not change.

9. The team's note of 7.11.26 in config.py (square sheets at the exam) and the professor's answer to question 9(a) of 26.10.26 were not weighed against the proposed exam rule; together they decide whether orientation is an exam risk at all.

10. Limits of my own check, so they are not taken for more than they are: every position on the table (puck, line, start gap) rests on the same fitted camera of 6.11.26 as the analyst's, only the finders differ; my line finder gave clean readings in only 15 of 131 demonstration first pictures, so I confirmed the lay-outs of runs 3, 4, 6 and 7 by measure and of run 8 by eye and took the other runs' lay-outs from the brief; the what-if with the asked step lengths uses the logged answers, and a real run would see other pictures. Nothing was run on the robot, nothing trained, no git change made.

### The analyst's corrections to the brief

1. The puck is not 4 cm: it measures 2.45 cm across in the pictures (median of 375 whole detections placed on the table with the fitted camera, 04_physics.out.txt part C), and config.py PUCK_PARAMETERS says 2.4 cm. The black magnet block is the part that is about 4 cm wide.

2. tracks.json does not describe the practice tracks in the folder: it lists 7 of the 20 tracks, and for practice_03 and practice_06 its start, end and length differ from the SVG files (practice_03: start (69.2, 251.4) mm and 264 mm long in the JSON, (97.4, 244.7) and 253 mm in the SVG; practice_06: 253 mm against 358 mm). The SVG files were used.

3. Run 3's sheet was not an unseen placement: it lay as in the six SQP demonstrations from end B (plans 02-13 to 02-18). What the demonstrations do not hold is its start state (block touching the puck, forearm straight, line 57 degrees off the picture's top).

4. Run 4 'as in plan 02-01': the set-up picture plan_02-01.png draws end A, but plan 02-01 was pushed from end B (audit) and run 4's first pictures match the end-B lay-out 02B-1. 'As plan 02-01 was recorded' is the accurate wording.

5. The set-up pictures show where a sheet was planned to lie, not where it lay: the start caps lay a median 19.2 cm in front of the base axis (drawn 16.4), the sheets a median 5.9 degrees from the plan, 20 of 146 start-mark episodes more than 15 degrees.

6. All six SQP episodes are from end B; plans 02-13 to 02-15 are labelled end A but were laid and pushed from end B (the audit's corrected ends). No square-portrait demonstration from end A exists.

7. The usable reach ends at about 47.5 cm, not 49 to 49.5: with the forearm straight and the wrist held, the rod's lean passes the driver's 58-degree limit at 48 cm (calculated, 04_physics.out.txt part A).

8. The driver is no longer only the fixed 1-cm step the brief describes: since 18:18 on 11.11 config.py and model_mode.py hold step_length_rule ('fixed' by default, 'predicted' available) and a stall rule. Not yet tried on the robot as far as the logs show.

9. Scout number confirmed on the cleaned set: rows that twist 6 degrees or more move the tip a median 0.65 cm (5,085 rows, 36 % under 0.5 cm).

### The analyst's open questions

1. Who lays the unseen sheets at the exam, and may Emranul and Marshal set the start by hand for each sheet (tip place and forearm twist)? The 26.10 answer allows choosing start and orientation; it was not asked for the unseen sheets.

2. May own tracks be printed (question 15), and do pictures changed by rotation still count as own demonstrations?

3. How much free table is there round the robot on the first-floor table? The centre-cross placements need paper out to about x = 49 cm and y = 18 cm either side, some mark placements out to y = -28 cm. The calculation checked the base and the driver's box, not the table edge.

4. Does the model turn in place when the driver obeys its step length, or does it stall? Only the robot test of recommendation 1 can tell; the what-if in the findings uses run 3's logged answers, and the pictures would differ in a real run.

5. Is the fall of the offline score with reach (0.90 to 0.94 near the base, 0.56 to 0.63 beyond 42.5 cm) a lack of pictures (375 beyond 42.5 cm) or the view with a far-leaning rod? The saved predictions cannot separate the two, and the easy starts all lie near the base.

6. Taught starts score no better than always-forward over their first ten rows offline (0.75 to 0.82 against 0.79). Is that the known weakness near the goal or the start with a turned forearm? It matters for the pre-turned start rule and was not separated here.

7. The fitted camera dates from 6.11 on the old desk. It was checked here on six pictures by eye and through the 1.3 cm agreement between seen pucks and fitted start caps, not by a fresh calibration; the view and visibility numbers rest on it.

8. The lay-out fits' turn accuracy is a guess of about 5 degrees (from the spread inside one lay-out group); short near-end episodes are less certain and were attached to the nearest full lay-out.

9. The coverage score of the recording plan uses stated choices (a cell full at 150 pictures or 4 starts, 100 new centimetres worth one cell, 10 minutes per new lay-out). Other choices would reorder picks of similar score.

## Bends and two ways (folder `bends_and_two_ways`)

**Analyst's answer.** At sharp bends and hairpins the demonstrations slow down and twist the forearm (hairpin: 4.9 pictures per cm of line, 30 % twists in place), and the model asks for that slow pace, but the driver pushed a full 1 cm per picture: about twice the asked distance at the first bend of sheet 03. The back-and-forth of run 6 was not at the hairpin itself: the puck had already left the line at the first bend, and in that unseen view the model's 0.16 cm answer was the average of look-alike demonstrations that go opposite ways, so neither the chunk's 3-step point nor a keep-the-last-direction rule would have cured it.

**Verifier's overall verdict.** Yes for the numbers, partly for the explanation of run 6. Every headline number I recomputed with my own code came out the same (hairpins 4.86 pictures per cm of line and 30 % of steps under 0.3 cm; sheet 03 first bend 7.1 to 7.7 cm asked against 16.0 cm driven; run 6: puck off the line at the first bend in the pictures, 32 answers with a median of 0.16 cm, chunk point 3 ahead reversing 18 times against 20), and my forward pass reproduced the saved fold predictions within 0.002 degrees. Two things must be added before a plan is built on it: (1) the back-and-forth of run 6 was the fixed 1 cm step jumping across a balance point the model was steering to (16 of 18 off-centre answers pointed back to the middle, forward part 0.00 cm), so with the asked length the robot would have stood still there (calculated, not tried), and sheet 04 is not a fair test of the step-length idea; (2) at sheet 03's first bend the model asked for about two thirds of the demonstrated travel (7.1 to 7.7 against 11.1 cm in 16 steps), so whether the asked pace carries the puck round is still a guess, and a length gain above 1 gives the turning back away. All my scripts and outputs are in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/bends_and_two_ways/verify/ (nothing connected to the robot, no git change, no training).

### Verdicts on the claims

1. **confirmed**: Demonstrations get round sharp bends and hairpins by slowing down and twisting, not by backing up (table per kind of place; hairpins only on sheet 04)

   Measured with my own turning measure (1 cm chords on the traced line) and my own counting, v04_demonstrations.py. Straight / gentle / sharp / hairpin: pictures per cm of line 1.90 / 2.01 / 3.03 / 4.86; median step 1.13 / 1.09 / 1.01 / 0.53 cm; under 0.3 cm 6.2 / 5.5 / 12.9 / 30.0 %; median twist 2.46 / 3.43 / 5.27 / 6.15 deg; twist per cm of line 6.0 / 7.4 / 14.8 / 29.5 deg; real moves against the line 8.3 / 14.1 / 12.8 / 5.5 %. Independent count for hairpins: 1,091 pictures over 35 episodes times 5.9 cm of hairpin line = 5.3 pictures per cm. Hairpin pictures: 1,091, all on sheet 04. I looked at 30 pictures drawn at random (v12_place_*.jpg): all 10 'hairpin' pictures show the puck at a U-turn, the 'straight' ones on straight stretches.

2. **corrected**: Line end: 8.53 pictures per cm of line and 17.3 degrees of twist per cm

   Not a robust number. The audit's line position saturates at the end cap (sheet 04: 221 pictures for 0.6 cm of progress in the analyst's own table). My values: 4.16 pictures per cm with the same clipping, 6.34 without clipping, twist 8.2 deg per cm. Do not use the line-end pace for anything; the other four kinds of place are unaffected.

3. **confirmed**: A hairpin takes about 28 pictures and 180 degrees of twist; within one direction of travel all passes go round the same way, the other direction is the mirror image

   Measured, v04_demonstrations.py. Hairpin 1 from end A: 26 passes (analyst 25), median 27.5 steps (8 to 33), tip travel 17.5 cm, 8 twists in place, twist 178 deg (net +175), same sign in 96 %. From end B: 9 passes, 24 steps, net -148 deg, same sign in all. Hairpin 2 from A: 26 passes, 17.5 steps (11 to 37), 98 deg. Sheet 03 first bend from A: 15 passes, 14 steps (9 to 21), 3 twists in place, net -79 deg (analyst -85), same sign in all. I looked at s01_hairpin_paths.png: the tip goes round the outside.

4. **confirmed**: Jitter sits mostly in tip moves under 3 mm made while the forearm twists; real moves seldom reverse

   Measured. Share of consecutive steps differing by more than 90 deg: 6.8 / 7.4 / 9.6 / 18.4 / 11.6 % (straight / gentle / sharp / hairpin / line end); both steps 0.3 cm or more: 3.8 / 5.2 / 5.2 / 8.2 / 9.4 % (over 120 deg: 0.9 to 2.1 %, line end 4.0 %); a step under 0.3 cm involved: 24 to 34 % (over 120 deg: 15 to 22 %). Per sheet within 0.2 points of the analyst. The values depend on counting only steps recorded one after the other; with pairs that bridge a dropped row they are 7.4 / 8.8 / 10.2 / 18.4 / 15.6 %.

5. **confirmed**: Two styles: sheets 03 and 04 twist the camera round, the loops of sheets 05 and 06 step back and sideways more

   Measured: real moves against the line at sharp bends 7.1 / 9.1 / 24.5 / 22.9 % on sheets 03 / 04 / 05 / 06 (analyst 7.0 / 9.2 / 24.7 / 20.0). But the median twist per step at sharp bends is nearly the same on all four sheets (5.1 / 5.2 / 4.6 / 4.9 deg), so 'the camera turns less on the loops' rests only on the pass sums of 18 and 13 passes. Medium, as the analyst rated it.

6. **confirmed**: Offline the predicted step gets shorter at bends but not tiny; it equals the demonstration's local pace

   Measured with my own forward pass of the six recipe E fold models (v05_offline_by_place.py; saved predictions reproduced within 0.002 deg). Predicted median 0.84 / 0.70 / 0.63 / 0.63 / 0.47 cm; under 0.3 cm 3.6 / 2.4 / 5.0 / 7.3 / 22.3 %; local pace (mean of 5 steps) 1.00 / 0.85 / 0.71 / 0.47 / 0.76 cm; model E on its own training pictures 0.96 / 0.80 / 0.69 / 0.46 / 0.44 cm. To add: the fold model for sheet 04 has never seen a hairpin and does not slow down there (0.63 asked against a pace of 0.47); over all unseen-sheet pictures the asked lengths sum to 0.71 of the demonstrated lengths and the twists to 0.64.

7. **confirmed**: Offline, bends are not where the direction is wrong (the line's last 2 cm is); predictions almost never reverse; the model twists into the bend with about two thirds of the demonstrated size

   Measured. Path within 45 deg, exact step: 0.900 / 0.793 / 0.790 / 0.800 / 0.462 (always forward 0.777 / 0.599 / 0.498 / 0.607 / 0.309). Predictions differing by more than 90 deg between consecutive pictures: 0.18 to 0.45 % against 6.8 to 18.4 % demonstrated. Hairpins: predicted twist into the bend on 95.5 % of pictures at a 1 deg limit (analyst 89.4 % at 2 deg), out of it 0.5 %; mean size 4.47 against 6.07 deg. Sum of predicted over demonstrated twist 0.66 / 0.68 / 0.57 / 0.74.

8. **confirmed**: A short answer means an unsure answer, so the predicted length works as a confidence signal

   Measured offline: 830 answers under 0.3 cm (4.7 %); the demonstration made a real move of 0.7 cm or more on 59 %, a twist in place on 14 % (hairpins 35 % of 80); direction more than 90 deg off on 28.1 % of those real moves. Path within 45 by predicted length: 0.453 / 0.675 / 0.831 / 0.964. Limit: on the robot in run 6 the short answers were not random in direction (see the verdict on the two opposite ways).

9. **confirmed**: Run 6 did not fail at the hairpin: the puck left the line at the first bend, crossed white paper and met the approach to hairpin 1 from the side, 3 to 4 cm before it; the back-and-forth happened there

   I looked at all 69 logged pictures (contact sheets v01_174108_*.jpg). The puck is over the first cross stretch at pictures 8 to 10, on white paper at 11 to 13, on the near branch of the hairpin-1 lobe from 14 to 17 and stays there until 50, with the U-turn in view to its left; after 50 it is pushed straight across both branches towards end B. Measured in picture 020: line 21 pixels wide = 1 cm, puck centre 114 pixels from the U-turn's outer point, so about 5 cm before the apex, about 3.5 cm before the stretch classed as hairpin (from 23.5 cm of line). Numbers (v03_robot_runs.py): first 16 moves asked 12.1 cm, driven 16.0, twist net +7 deg, 25 as a sum; demonstrations 12.5 cm (10.2 to 17.9) and 85 deg (32 to 112). Steps 19 to 50: 32 answers, median 0.16 cm, sum 5.2 cm, 20 reversals in 31, tip box x 31.5 to 33.2 and y -3.7 to -0.3 cm, net travel 1.0 cm, forearm -5.7 to +6.7 deg.

10. **corrected**: In the back-and-forth the answer was the average of two opposite demonstrated ways; it flipped with whichever group of look-alike pictures was larger

   The numbers reproduce with my code (v06_run6_mechanism.py, hidden layer): nearest distance 8.71 against 4.46 (90th percentile 7.90); resultant 0.32; both groups 53 %; neighbours' mean move 0.27 against 0.87 cm; larger group 24 of 27; left group twists +5.3 deg, right group -2.7 deg. Two corrections. (a) In that description 12 demonstration pictures drawn AT RANDOM give both groups in 46 to 54 % of sets and a resultant of 0.47 to 0.53, so the neighbours agree no better than random pictures (the analyst's own figure shows line ends of sheet 01 and pictures of sheets 05 and 06 among them). In a second description (pooled backbone features, 6,144 numbers) the neighbours are from sheet 04 (9.8 of 12), split in 41 % of pictures (normally 7 %), and the answer's sideways part follows their mean (r = +0.89): that supports a real split. (b) The flips were not chance: the tip's sideways place and the asked sideways move correlate at -0.71; 16 of the 18 pictures more than 0.5 cm from the middle ask to go back to it; asked forward part 0.000 cm on average, twist -0.24 deg, stop chance at most 0.007. The model was steering to a balance point near y = -1.9 cm with 0.15 cm per picture; the fixed step drove 1 cm, about 7 times as much, and jumped across it every time. Calculated, not tried: with the asked length the tip settles there and stands still.

11. **corrected**: At sheet 03's first bend the model asked for the demonstrated manoeuvre and the fixed 1 cm step drove it at more than twice that pace, so the tip went 2 to 5 cm further out than the demonstrations

   Confirmed (v03, v04, v09): asked 7.3 / 7.7 / 7.1 cm against 16.0 driven (2.1 to 2.3 times); twist -58 / -74 / -53 deg; demonstrations 11.1 cm (10.5 to 14.5) and 93 deg (88 to 106); twist per asked cm 5.81 / 6.51 / 5.56, per driven cm 3.14 / 3.23 / 2.85, demonstrated 4.64. Two corrections. (1) '2 to 5 cm further out' holds against the median (6.4 cm) only: the 8 standard starts advance 6.0, 6.1, 6.4, 6.4, 6.5, 8.3, 8.8, 8.9 cm; runs 5 and 9 (11.1 and 9.8 cm) are beyond all 15 demonstrations, run 10 (8.2 cm) is inside their range. (2) The model did not ask for the demonstrated pace but for about two thirds of it (7.1 to 7.7 against 11.1 cm in 16 steps). Along the logged directions the asked lengths would give an advance of 5.3 / 4.7 / 3.8 cm (calculated; the pictures would then have differed), below every standard-start demonstration, so a gain of 1.0 may turn too early.

12. **confirmed**: The asked step length separates the runs that worked from those that failed

   All ten rows identical (v03): medians 0.98 / 1.14 / 0.53 / 0.87 / 0.55 / 0.31 / 0.83 / 0.82 / 0.49 / 0.55 cm, shares under 0.3 cm 4 / 4 / 0 / 2 / 12 / 50 / 9 / 2.5 / 15 / 16 %, asked paths 23.5 / 25.6 / 23.1 / 32.9 / 31.9 / 26.0 / 68.9 / 32.3 / 44.1 / 37.5 cm. Model E on run 6's pictures gives the logged step back to 0.00001 cm. The wording is too wide: run 8 (sheet 06, failed) asked a median of 0.82 cm, the same as run 7 (0.83, worked), and run 3 failed with no short answers. The asked length marks the four sharp-bend failures (runs 5, 6, 9, 10), not every failure.

13. **confirmed**: Steering by the chunk's place 3 steps ahead is worth about 1 point offline and would not have calmed run 6; chunk-trained models are better at bends even through their first step

   Measured from saved predictions (v07) and my forward pass (v06). Three ahead minus first step: all places +0.6 to +1.0, sharp bends -0.1 to -0.6, hairpins +2.8 to +3.5 points. Five exact seeds: sharp 0.738 to 0.768, hairpins 0.682 to 0.767; chunk-trained first step 0.780 to 0.798 and 0.801 to 0.836. Run 6, steps 19 to 50: reversals exact 20, chunk 3 ahead 18, chunk 5 ahead 6 of 31; lengths 0.53 and 0.97 cm (steps 0 to 4: 2.95 and 4.69). Addition (v14): the TEAM of the five exact seeds, which is model B on the robot, scores 0.780 at sharp bends (approximated from the saved directions), level with E (0.780 to 0.794); E leads only at hairpins (0.744 against 0.801 to 0.836, one sheet, 544 scored pictures).

14. **confirmed**: Carrying out 2 or 3 chunk steps loses nothing on the demonstrations' pictures, but the camera would have to twist 10 to 19 degrees at bends without a new picture

   Measured (v11): chunk place error 0.99 cm at 2 steps and 1.27 at 3 (sharp bends 1.02 / 1.32, hairpins 0.87 / 1.16). Demonstrated twist summed over 2 and 3 steps, median: all places 7.8 and 11.3 deg (analyst 8.0 and 11.7), sharp bends 10.4 and 14.9, hairpins 12.5 and 19.0; 90th percentile at sharp bends for 3 steps 23.6. The 'looking again' figures (1.04 and 1.40 cm) were not recomputed.

15. **confirmed**: A names output does not pick one way better at bends: 8 to 12 points worse, not split at bends, a coin toss where the exact step is short

   Measured (v07 and my forward pass of the six backup fold models, v11). Path within 45, gentle / sharp / hairpin: most likely name 0.683 / 0.667 / 0.649; mean direction 0.735 / 0.707 / 0.691; the same model's exact step 0.780 / 0.755 / 0.761. Pictures with 25 % or more of the probability over 90 deg from the top name: 2.3 / 3.4 / 4.2 / 0.7 / 22.6 %. Where the same model's exact step is under 0.3 cm: 977 pictures, 43.2 % split, top probability 0.281. Top name reverses on 4.4 % of consecutive pictures (the same model's exact step 0.7 %, recipe E 0.35 %).

16. **confirmed**: Keep the last direction on a flip over 120 degrees: would not have stopped run 6 and would have broken run 7

   Calculated on the logged answers in order (v03): run 6 fires 17 times from step 19, never more than 2 in a row; run 7 fires 6 times in a row at steps 49 to 54 while the forearm goes from -17.4 to +55.0 deg over steps 42 to 60; run 9 once (step 30); the others never. Note: in run 7 the rule locks because it takes two answers of 0.14 and 0.20 cm (steps 47 and 48) as its reference. This shows that this version of the rule is unsafe, not that every damping rule is. The reason given in the recommendation ('answers under 0.3 cm, whose direction is close to chance') is wrong for run 6, where the direction was systematic.

17. **confirmed**: A short picture history gains little at bends on frozen features, and the best variant shows the signature of copying the last move

   Measured from the saved probe predictions (v07): one picture / plus previous / plus change / plus pictures 2 and 4 back: all places 0.672 / 0.676 / 0.695 / 0.682; sharp bends 0.650 / 0.662 / 0.678 / 0.679; hairpins 0.730 / 0.752 / 0.713 / 0.706; line end 0.436 / 0.497 / 0.530 / 0.538. 'Change since previous': +4.8 points where the demonstration kept its direction (0.717 to 0.765), -1.4 where it changed (0.597 to 0.583); answer within 30 deg of the previous demonstrated step on 45.8 % against 40.3 %.

18. **corrected**: Averaging the joint angles of the guided path moves the tip by up to 27 mm sideways and 7.5 mm down, so it must not be done that way; the place in record_episodes.py is known

   The code places are right: __record_guided_path line 2345, __exact_waypoints line 1105, called at lines 799 and 1208. The 'must not' rests on single points. My check (v10_smoothing_spot_check.py, 227 guided paths, 44,328 points, joints averaged over 5 points): the smoothed tip lies 0.49 mm (median), 1.78 mm (95th percentile), 2.47 mm (99th) and at most 9.3 mm from the hand's tip path; only 6 points are more than 5 mm off, and they sit where the guided path itself jumps (25 to 107 mm between neighbouring points against 5 mm normally). Height change -2.7 to +2.4 mm (1st to 99th percentile), -14 and +13 mm at the jumps. Their larger figures probably measure point to point, which also counts sliding along the path (a guess). The label statistics of the variants (jumps 8.9 % to 1.7 % and so on) were not recomputed.

19. **confirmed**: The stall rule now in the driver would not have ended run 6; a slightly wider rule fires in run 6 only

   Calculated on the logs (v03), with the box's DIAGONAL as model_mode.py measures it: the driver's rule never fires (smallest 30-picture diagonal in run 6: 3.75 cm against the limit of 3.0); 15 pictures within 4 cm and under 15 deg: picture 31 of run 6, no other run (closest other run 9.6 cm); net travel under 1.5 cm over 12 pictures: picture 28 of run 6 only. New (v08, v08b): on the 210 recorded demonstrations treated as runs, the driver's rule and the 4 cm box rule never fire; the net-travel rule fires in 57 (54 of them with a hand correction; 3 of the 79 without).

20. **confirmed**: Every standard-start demonstration began from the same tip position and forearm angle; the ten runs began 1.3 to 5.0 cm away with the forearm turned by up to 10 degrees

   Measured (v03): 71 standard starts: tip x 14.00 to 14.04, y -1.49 to -1.48 cm, joint 4 -0.08 to +0.09 deg. Runs: 1.6 / 1.8 / 2.5 / 3.8 / 3.1 / 5.0 / 4.0 / 3.2 / 1.3 / 3.3 cm away, forearm -9.7 to +9.8 deg. Addition: the offset, turned and square starts share exactly this pose too (170 of 210 episodes), and the picture's heading at the start is -6.1 deg in all of them against -4.4 to +12.3 deg in the runs.

21. **confirmed**: Corrections to the brief: scout step lengths; the driver already holds a predicted-length rule and a stall rule; in chunk runs 'angle_deg' is the chunk's first step

   Step lengths: all 18,312 non-stop rows 20.4 % under 0.5 cm and 8.0 % under 0.2 cm; the 17,773 cleaned rows 18.2 % and 5.4 %, median 1.06, mean 0.94; 0.65 cm for twists of 6 deg or more. config.py lines 1905 to 1910 and model_mode.py lines 260 and 447 to 469 hold the rules; test_model_mode.py has tests for both and a simulated run of 18:19 used 'predicted'; not tried on the robot. training_probes.py line 448 overwrites angle_deg with the chunk's first step; harmless for the scores (exact step 0.799, first chunk step 0.799).

### The analyst's findings with their evidence

1. (measured, confidence high) The demonstrations get round sharp bends and hairpins by slowing down and twisting the forearm, not by backing up: the tighter the place, the more pictures per cm of line, the shorter the steps and the more twists in place.

   Evidence: Kind of place = how much the traced line turns within 6 cm at the puck (straight under 15 deg, gentle 15 to 60, sharp from 60, hairpin = direction changes by 150 deg or more; last 2 cm of the line kept apart). Straight / gentle / sharp / hairpin / line end: pictures per cm of line 1.84 / 2.06 / 3.04 / 4.86 / 8.53; median step 1.13 / 1.09 / 1.01 / 0.53 / 1.08 cm; steps under 0.3 cm 5.6 / 5.6 / 12.9 / 30.0 / 10.4 %; median forearm twist per step 2.4 / 3.4 / 5.3 / 6.2 / 0.8 deg; twist per cm of line 5.7 / 7.6 / 14.9 / 29.5 / 17.3 deg; real moves against the line's direction 8.1 / 13.8 / 12.5 / 5.4 / 24.6 %. 17,177 pictures with a line position, 209 episodes. Files: s01_demonstrations_at_bends.out.txt, s01_place_table.csv (frames.csv through load_training_frames with the audit cleaning; puck positions along the line from the audit of 9.11.26, copied to inputs/, good to 1 to 2 cm). Hairpins exist only on sheet 04 (1,091 pictures, 35 episodes).

2. (measured, confidence high) A hairpin takes about 28 pictures and about 180 degrees of forearm twist; a straight 6 cm takes 5 pictures. Within one direction of travel every demonstration goes round the same way (tip on the outside, camera turning into the bend); the other direction of travel is its mirror image.

   Evidence: Complete passes through a window of 3 cm of line before and after the bend's middle. Sheet 04 hairpin 1 from end A: 25 passes, median 28 steps (8 to 33), tip travel 17.6 cm, 8 twists in place, 0 steps against the line, twist summed 178 deg, same sign in 96 % of passes, tip 1.35 cm outside the line (median). From end B: 9 passes, 24 steps, twist -148 deg, opposite sign in all 9. Hairpin 2 from A: 26 passes, 17.5 steps (11 to 37), twist 98 deg. Sheet 03 first bend from A: 15 passes, 14 steps (9 to 21), 3 twists in place, twist -85 deg, same sign in all 15. Straight controls (sheets 01 and 03): 5 to 7 steps, 7 to 14 deg. Move names at hairpin 1 from A: small motion 29 %, forward-left 27 %, left 20 %, forward 18 %, back names 4 %; from B: small motion 32 %, forward-right 24 %, forward 18 %, right 16 %. Files: s01_demonstrations_at_bends.out.txt, s01_feature_passes.csv, figure s01_hairpin_paths.png.

3. (measured, confidence high) The jitter in the demonstrations sits mostly in tip moves under 3 mm made while the forearm twists; real moves seldom reverse. The 'zig-zag' at the first bend of sheet 03 is of this kind.

   Evidence: Share of consecutive steps whose directions on the table differ by more than 90 deg: straight 6.6 %, gentle 7.4 %, sharp 9.7 %, hairpin 18.4 %, line end 11.6 %. Per sheet: 01 straight 6.9; 02 gentle 9.6; 03 gentle 8.0, sharp 11.1; 04 gentle 6.8, sharp 10.7, hairpin 18.4; 05 gentle 4.8, sharp 6.8; 06 gentle 5.0, sharp 6.3 %. Pairs where both steps are 0.3 cm or more: 3.8 / 5.1 / 5.3 / 8.2 / 9.2 % (over 120 deg: 1.0 to 2.1 %, line end 3.9 %); pairs with a step under 0.3 cm: 25 to 35 % (over 120 deg: 15 to 23 %). Plan 03-04, first steps: the jumping directions (-114, 12, -82 deg) belong to moves of 0.15 to 0.21 cm with twists of -4.8 to -6.2 deg; its real moves go 16, -36, -28, -35, -96 deg. Files: s01_demonstrations_at_bends.out.txt (table and 'direction jumps'), s08_scout_checks.out.txt section 3.

4. (measured, confidence medium) Two styles of rounding a bend are in the data: on sheets 03 and 04 the camera is twisted round with the line; on the loops of sheets 05 and 06 the tip steps back and sideways more and the camera turns less.

   Evidence: Real moves against the line's direction at sharp bends: 7.0 % (sheet 03), 9.2 % (04), 24.7 % (05), 20.0 % (06). Tight bend of the sheet 05 loop from end B (18 passes): 8 of 18 steps against the line, twist 82 deg for a bend of 114 deg; sheet 06 loop from end A (13 passes): twist 29 deg for 130 deg. Files: s01_place_table.csv, s01_demonstrations_at_bends.out.txt (passes and example passes).

5. (measured, confidence high) Offline, on sheets the model was not trained on, the predicted step gets shorter at bends but not tiny: it is about as long as the average of five neighbouring demonstrated steps, that is the demonstration's local pace.

   Evidence: Median predicted length, recipe E fold models, straight / gentle / sharp / hairpin / line end: 0.85 / 0.70 / 0.63 / 0.63 / 0.47 cm (second seed 0.83 / 0.67 / 0.62 / 0.61 / 0.41); under 0.3 cm: 3.8 / 2.4 / 5.0 / 7.3 / 21.7 %. Demonstrated median 1.13 / 1.09 / 1.01 / 0.53 / 1.08 cm. Length of the average of 5 neighbouring demonstrated steps: 0.99 / 0.85 / 0.71 / 0.47 / 0.69 cm; model E on its own training pictures: 0.97 / 0.80 / 0.69 / 0.46 / 0.55 cm. Files: s03_offline_by_place.out.txt sections 1 and 6b, s08_scout_checks.out.txt section 4 (the fold models were run again on the cached pictures by s02_model_outputs.py; every fold reproduced its saved angle within 0.002 deg).

6. (measured, confidence high) Offline, bends and hairpins are not where the direction is wrong: the weak place is the line's last 2 cm. Predictions almost never reverse from one picture to the next, and the model twists into the bend like the demonstrations, with about two thirds of their size.

   Evidence: Path within 45 deg (direction against where the demonstration's tip was 3 steps later), exact step of recipe E, held-out sheets: straight 0.897, gentle 0.795, sharp 0.790, hairpin 0.800, line end 0.478 (always forward: 0.779 / 0.601 / 0.498 / 0.607 / 0.329). Consecutive predictions differing by more than 90 deg: 0.2 to 0.4 % of pairs, against 6.6 to 18.4 % for the demonstrated steps. At hairpins the predicted twist goes into the bend on 89.4 % of pictures and out of it on 0.2 % (demonstrations 84.8 % and 2.4 %), mean 4.44 against 5.81 deg; sum of predicted over sum of demonstrated twist sizes: straight 0.66, gentle 0.68, sharp 0.57, hairpin 0.74. Files: s03_offline_by_place.out.txt sections 2, 3, 6, 7; s03_*.csv.

7. (measured, confidence high) A short answer means an unsure answer, not 'the demonstration twisted in place here': the predicted length works as a confidence signal.

   Evidence: Where recipe E's held-out step is under 0.3 cm (818 pictures, 4.8 %): the demonstration made a real move of 0.7 cm or more on 58 % and a twist in place on 14 % (at hairpins 35 %); for the real moves the predicted direction is more than 90 deg off on 28.5 %. Path within 45 by predicted length: under 0.3 cm 0.453; 0.3 to 0.6 cm 0.671; 0.6 to 0.9 cm 0.830; 0.9 cm or more 0.963. Files: s03_offline_by_place.out.txt section 5, s03_short_steps.csv.

8. (measured, confidence high) Run 6 did not fail at the hairpin. The puck left the line at the first bend (pictures 6 to 9), was pushed across about 5 cm of white paper and met the approach to hairpin 1 from the side (pictures 13 to 16), 3 to 4 cm before the hairpin; the back-and-forth happened there. At the first bend the model's twist came late and died: the asked step lengths were about right.

   Evidence: Seen in the run's own pictures, figure s04_run6_pictures.jpg (steps 4, 6, 8, 9, 11, 13, 16, 18, 19, 20, 22, 24 with the asked direction). First 16 moves of run 6: asked 12.1 cm (driven 16.0), twist asked +7 deg net, 25 deg as a sum of sizes; the 11 standard-start demonstrations of sheet 04 over their first 16 steps: 12.5 cm (10.2 to 17.9) and 85 deg, twisting from their third step on. Back-and-forth, steps 19 to 50: 32 answers with a median length of 0.16 cm (sum 5.2 cm), 20 reversals over 90 deg in 31, tip inside x 31.5 to 33.2 and y -3.7 to -0.3 cm, net travel 1.0 cm in 31 moves, forearm between -5.7 and +6.7 deg. Files: s05_options_on_robot_runs.out.txt (sheet 04 section), s04_run6_neighbours.out.txt, figure s04_run6_path.png (steps.csv of 2026-11-11_174108_E_sheet04_run1).

9. (measured, confidence medium) In the back-and-forth of run 6 the answer was the average of two opposite demonstrated ways: the look-alike demonstration pictures split into moves to the left with a twist to the right and moves to the right with a twist to the left, and the model's answer flipped with whichever group was larger. On ordinary demonstration pictures, hairpins included, this split is rare.

   Evidence: Look-alike = the 12 nearest of the 18,032 demonstration pictures in model E's own hidden layer. Steps 19 to 50: each neighbour's labelled move is 0.87 cm long on average, but the neighbours' mean move is 0.27 cm; resultant of their directions 0.32 (median); in 53 % of the pictures at least 3 neighbours go 0.3 cm or more left and at least 3 right; both twist signs in 50 %. Yardstick, 600 random demonstration pictures and their neighbours from other episodes: resultant 0.87, both groups in 9 % (hairpins 3 %, sharp bends 7 %). The answer went to the side of the larger group in 24 of 27 pictures and its twist had the majority's sign in 28 of 32; answers to the right came with left twists (93 %), answers to the left with right twists (82 %). The view was far from the data: nearest demonstration picture at a distance of 8.71 (median) against 4.46 (median) and 7.90 (90th percentile) between demonstration pictures of different episodes; steps 0 to 4 of the same run: 4.53. Files: s04_run6_neighbours.out.txt, s04_run6_neighbours.csv, figure s04_run6_neighbours.jpg.

10. (measured, confidence high) At the first bend of sheet 03 the model asked for the demonstrated manoeuvre (a swing to the outside of the bend in steps of 0.42 to 0.47 cm with twists of 5 to 6 degrees), and the fixed 1 cm step drove it at more than twice that pace, so the tip went 2 to 5 cm further out than the demonstrations before the bend was taken.

   Evidence: First 16 moves of runs 5, 9, 10: asked 7.3 / 7.7 / 7.1 cm, driven 16.0 cm; twist asked -58 / -74 / -53 deg. The 8 standard-start demonstrations over their first 16 steps: 11.1 cm (10.5 to 14.5) and 93 deg. Farthest advance in x before the tip was 2 cm left of its start: runs 11.1 / 9.8 / 8.2 cm, demonstrations 6.4 cm (median of 15, from 0.0 to 9.4). Twist per cm: demonstrations of sheet 03 4.64 deg; robot per cm driven 3.14 / 3.23 / 2.85; per cm ASKED 5.81 / 6.51 / 5.56. Other sheets, twist per asked cm against demonstrated: sheet 05 3.50 against 3.39; sheet 04 3.58 against 5.65; sheet 06 3.20 against 4.42; sheet 02 3.94 and 2.40 against 2.54. Files: s05_options_on_robot_runs.out.txt (sheet 03 section), s08_scout_checks.out.txt sections 2 and 5, figures s05_sheet03_first_bend.png and s05_run9_first_bend.jpg.

11. (measured, confidence high) The asked step length separates the runs that worked from those that failed, as the brief's quick numbers said.

   Evidence: Median asked length and share under 0.3 cm: run 1 0.98 cm, 4 %; run 2 1.14, 4 %; run 3 0.53, 0 %; run 4 0.87, 2 %; run 5 0.55, 12 %; run 6 0.31, 50 %; run 7 0.83, 9 %; run 8 0.82, 3 %; run 9 0.49, 15 %; run 10 0.55, 16 %. Asked path against moves of 1 cm: 25.6 / 24 (run 2), 32.9 / 41 (run 4), 68.9 / 88 (run 7), 31.9 / 59 (run 5), 44.1 / 89 (run 9), 37.5 / 73 (run 10), 26.0 / 68 (run 6), 32.3 / 40 (run 8). Model E run again on the logged pictures gave back the logged step exactly (largest difference 0.0000 cm in all eight E runs). Files: s05_run_table.csv, s02_model_outputs.robot_runs.out.txt.

12. (measured, confidence medium) Steering by the chunk's place 3 steps ahead is worth about 1 point offline and would not have calmed run 6; the place 5 steps ahead was steadier there. Models trained with the chunk are better at bends even through their first step.

   Evidence: Path within 45, 3-steps-ahead point minus first step of the same model, four held-out models (chunk recipe and recipe E, two seeds each): all places +0.6 to +1.0 points; sharp bends -0.1 to -0.6; hairpins +2.8 to +3.5; line end +2.8 to +6.2. Spread between five seeds of the exact recipe: 2.2 points overall, 8.5 at hairpins. First step alone, chunk-trained models against the five exact-only seeds: sharp bends 0.781 to 0.800 against 0.740 to 0.768; hairpins 0.801 to 0.836 against 0.682 to 0.767. On run 6's pictures, steps 19 to 50: reversals over 90 deg in 31 pairs: exact step 20, chunk 3 ahead 18, chunk 5 ahead 6; the chunk shrank as well (3 ahead 0.53 cm, 5 ahead 0.97 cm, against 2.95 and 4.69 cm on steps 0 to 4). At sheet 03's first bend the chunk points 10 to 20 deg less to the outside than the exact step (run 5, steps 3 to 7: exact -54 to -31, chunk 5 ahead -36 to -12 deg). Files: s09_seed_spread.out.txt, s03_offline_by_place.out.txt, s05_options_on_robot_runs.out.txt.

13. (measured, confidence medium) Carrying out 2 or 3 steps of one chunk loses nothing on the demonstrations' own pictures, but the camera would have to be twisted 10 to 19 degrees at bends without a new picture, and the model gives one twist per picture.

   Evidence: Recipe E, held-out sheets, distance of the predicted place from the true place: 2 steps ahead 0.99 cm (looking again after each step: 1.04), 3 steps 1.26 cm (1.40); at sharp bends 1.02 / 1.32 cm, at hairpins 0.87 / 1.16 cm. Direction of the chunk's own 1st, 2nd, 3rd step within 45 deg of the demonstrated one: 0.669, 0.720, 0.735. Demonstrated twist summed over 2 and 3 steps (median): all places 8.0 and 11.7 deg; sharp bends 10.5 and 15.0; hairpins 12.5 and 19.0 (90th percentile at sharp bends for 3 steps: 23.7). Offline both follow the demonstrator's pictures, so the loss from not correcting between looks cannot be seen. Files: s06_chunk_steps_and_history.out.txt part 1, s06_chunk_error.csv.

14. (measured, confidence high) A names output does not pick 'one way' better at bends: it is 8 to 12 points worse there, its probability is not split at bends, and where the exact step is short it is split and its top name is a coin toss.

   Evidence: Backup model (8 names plus exact), held-out sheets, path within 45: most likely name 0.686 gentle / 0.667 sharp / 0.649 hairpin; probability-weighted mean direction 0.738 / 0.708 / 0.691; exact step of recipe E 0.795 / 0.790 / 0.800. Pictures with 25 % or more of the probability on names over 90 deg from the top name: straight 2.4 %, gentle 3.3 %, sharp 4.2 %, hairpin 0.7 %, line end 22.1 %. Where the same model's exact step is under 0.3 cm (977 pictures): 43 % of pictures split like that, top probability 0.28 (median). The most likely name reverses on 4.4 % of consecutive pictures, the exact step on 0.4 %. Files: s03_offline_by_place.out.txt sections 2, 3, 4; s03_names_by_place.csv.

15. (calculated, confidence high) A rule that keeps the last direction when the new one flips by more than 120 degrees would not have stopped run 6 and would have broken the one run that went round a loop.

   Evidence: Applied to the logged answers in order (open loop): run 6: fires 17 times from step 19, never more than 2 in a row, and the driven direction still swings (-89, 17, 93, 107 deg pass the rule). Run 7 (sheet 05, completed): fires 6 times in a row at steps 49 to 54, where the tip went forward round the puck with twists of +5 to +6 deg per step after two answers of 0.14 and 0.20 cm had pointed backwards; the rule would have driven the tip backwards for 6 cm. Run 9: once; other runs: never. In the demonstrations 1.0 to 2.1 % of pairs of real moves differ by more than 120 deg (3.9 % at the line end). Files: s05_options_on_robot_runs.out.txt (run table, run 6 and run 7 listings), figure s05_run7_walk_round.jpg.

16. (measured, confidence medium) A short picture history gains little at bends on frozen features, and the variant that gains most shows the signature of copying the last move.

   Evidence: Frozen ResNet-18 names probes of 9.11.26, held-out sheets, path within 45 (mean direction), one picture / plus previous picture / plus change since previous / plus pictures 2 and 4 back: all places 0.672 / 0.676 / 0.695 / 0.682; sharp bends 0.651 / 0.663 / 0.678 / 0.680; hairpins 0.730 / 0.752 / 0.713 / 0.706; line end 0.442 / 0.490 / 0.534 / 0.538; straight 0.796 / 0.795 / 0.809 / 0.736. 'Change since previous': +5.0 points where the demonstration kept its direction (0.720 to 0.770, 8,017 rows) and -1.3 where it changed it (0.588 to 0.575, 4,652 rows); its answer is within 30 deg of the previous demonstrated step on 47.5 % of rows against 41.5 % for one picture. These are frozen-feature heads, not the fine-tuned model. Files: s06_chunk_steps_and_history.out.txt part 2, s06_history_by_place.csv (saved predictions in 04_runs/2026-11-09_frozen_feature_probes/predictions).

17. (calculated, confidence medium) Smoothing the hand-guided path before the replay would remove most direction jumps from new labels, but plain averaging of the joint angles moves the tip by up to 27 mm sideways and 7.5 mm down, so it must not be done that way. The place for it in the tool is known.

   Evidence: From the 209 recorded guided_path.csv files (40,331 points) and the arm model; the thinning rule was reproduced (waypoints here = recorded steps minus the stop row in 81 % of the 78 episodes without a hand correction, never off by more than 2). As recorded: 15,102 steps, 10.4 % under 0.3 cm, jumps over 90 deg on 8.9 % of pairs, median change between real moves 19.2 deg. Joints averaged over 5 guided points: jumps 1.7 %, median change 12.6 deg, 4 % fewer steps; tip off the hand's path 1.06 mm (median), 3.11 mm (95th percentile), 26.9 mm at worst; tip height -7.5 to +6.7 mm. Waypoint after 1 cm or 10 deg of forearm twist only (no smoothing): steps under 0.3 cm 2.4 %, 18 % fewer steps, twist per step up to 16.8 deg (99th percentile). In record_episodes.py the path comes from __record_guided_path (line 2345) and is thinned by __exact_waypoints (line 1105), called at lines 799 and 1208 (after a hand correction): a smoothing step belongs between them. Files: s07_smoothing_guided_paths.out.txt, s07_smoothing_variants.csv.

18. (calculated, confidence high) The stall rule now in the driver (30 pictures, box of 3 cm) would not have ended run 6 as it was driven; a slightly wider rule fires in run 6 only.

   Evidence: On the logged tips and forearm angles of the ten runs: driver's rule (config.py, added 11.11.26): never fires, because the tips of the back-and-forth span up to 3.8 cm. Box under 4 cm over 15 pictures with under 15 deg of forearm turn: fires at picture 31 of run 6, in no other run. Net travel under 1.5 cm over 12 pictures: picture 28 of run 6, no other run. With steps of the asked length the back-and-forth would have stayed within a few millimetres, so the driver's rule would fire then (calculated, not tried). File: s05_options_on_robot_runs.out.txt ('a stall rule').

19. (measured, confidence high) Every standard-start demonstration began from one and the same tip position and forearm angle; the ten runs began 1.3 to 5.0 cm away from it with the forearm turned by up to 10 degrees.

   Evidence: Demonstrations with a standard start, all sheets: tip at x 14.0, y -1.5 cm, forearm 0.0 deg (no spread). Runs: 1.6, 1.8, 2.5, 3.8, 3.1, 5.0 (run 6: x 18.2, y +1.3), 4.0, 3.2, 1.3, 3.2 cm away; forearm -9.7 to +9.8 deg. Runs 4 and 7 worked at 3.8 and 4.0 cm; run 9 failed at 1.3 cm, so the offset alone does not explain success. File: s08_scout_checks.out.txt section 6 (steps.csv of each run, frames.csv).

### The analyst's recommendations

1. Test the step length the model asks for against the fixed 1 cm on the robot. The switch is already in the driver (model_mode.py and config.py as changed on 11.11.26 after the test: --step-length predicted, --length-gain, at most 1.0 cm, no lower limit); it has not been tried. Sheets 03 and 04 from end A, same start each time, three settings: fixed 1.0 cm, predicted with gain 1.0, predicted with gain 1.3; 3 runs each; raise the step budget to about 400. Count: is the puck still on the line 10 cm after the first bend.

   Why: At sheet 03's first bend the model asked 7.1 to 7.7 cm over 16 moves with the demonstrated twist (53 to 74 deg), the robot drove 16 cm; the demonstrations move 11.1 cm with 93 deg. The predicted length follows the demonstrations' local pace per kind of place (0.97 / 0.80 / 0.69 / 0.46 cm against 0.99 / 0.85 / 0.71 / 0.47) and is short where the direction is unsure (path within 45: 0.45 under 0.3 cm, 0.96 from 0.9 cm).

   Needs: a robot test (the code exists; Emranul's go and the checklist first). Effort: one robot session of 2 to 3 hours (18 runs); half an hour to write the checklist and the run table beforehand

   Expected effect: Twist per cm back at the demonstrated level on sheet 03 (5.6 to 6.5 deg per asked cm against 4.64 demonstrated; 2.9 to 3.2 as driven). Guess: the first bend of sheet 03 is passed more often. Uncertain for sheet 04, where the asked length was already about right (12.1 against 12.5 cm) and the twist was missing (25 against 85 deg).

   Risk: Dithering on the spot where the model is undecided (run 6: 32 answers with a median of 0.16 cm); more pictures per run, so the budget of 250 can bind on sheet 04; creeping near the line's end, where the model asks 0.47 cm and 22 % of answers are under 0.3 cm, which changes what the stop sees. The right gain is not known: 1.0 fits the six known sheets, on unseen sheets the model asks about 15 % less.

2. Widen the driver's stall rule before that session: end the run as stalled when the box round the tips of the last 15 pictures is under 4 cm and the forearm turned over less than 15 degrees (or: net tip travel under 1.5 cm over 12 pictures).

   Why: On the ten logged runs the rule as set (30 pictures, 3 cm) never fires, not even in run 6 (the tips spanned 3.8 cm); both wider rules fire in run 6 only (pictures 31 and 28).

   Needs: code only (two numbers in MODEL_MODE_PARAMETERS and a line in test_model_mode.py). Effort: about 15 minutes

   Expected effect: A back-and-forth like run 6 ends after about 15 pictures instead of running on for 30 moves and then wandering across the sheet.

   Risk: A legitimate long twist in place could be cut short if the forearm limit of 15 degrees is set too high; in run 7 the walk round the puck turned the forearm by more than 40 degrees, so it is safe there. Checked on ten runs only.

3. Do not add the rule 'keep the last direction when the new one flips by more than 120 degrees'.

   Why: On the logged answers it fires 17 times in run 6 without stopping the swing, and 6 times in a row in run 7, the only run that went round a loop, where it would have driven the tip backwards for 6 cm in the middle of a correct walk round the puck. The flips come from answers under 0.3 cm, whose direction is close to chance; the asked-length step deals with those by hardly moving.

   Needs: nothing (a decision). Effort: none

   Expected effect: Avoids breaking the manoeuvre that worked on sheet 05.

   Risk: None from leaving it out.

4. Record recovery demonstrations at bends and retrain: short episodes that start in the states the model produced on 11.11.26 (puck 1 to 3 cm past a bend or beside the block, forearm not yet turned into the bend, puck meeting a line from the side), on sheets 03 and 04 first; and more passes of the two hairpins from end B.

   Why: Run 6's back-and-forth happened in a view farther from any demonstration (8.7) than 90 % of demonstration pictures are from other episodes (7.9), and there the look-alike demonstrations disagree (both ways among the neighbours in 53 % of pictures against 9 % normally). No driver rule supplies the missing answer. Each hairpin has 25 to 26 passes from end A and 9 from end B.

   Needs: new demonstrations and retraining; the professor's answer on whether such correction rounds count as our own demonstrations (already open in the report, section 10.1). Effort: 1 to 2 robot sessions for about 30 to 40 short episodes (near-start and MID starts exist in the recording page), half a day for the index, the audit rules and the retraining on ki-014 (about 2 minutes per held-out sheet, 2.3 minutes for an all-sheets model)

   Expected effect: Guess: the model has an answer where it now averages; the offline scores will hardly show it, only robot runs will.

   Risk: If the way INTO the bad state is recorded too, the model learns to leave the line; record only the way back. Forty episodes are a small share of 211, so the effect may be weak; the placement must be noted per episode as for the other plans.

5. Keep the chunk-trained recipe (model E) as the base model for bends, and drop the names output as a cure for bends.

   Why: With the first step alone, chunk-trained fold models beat all five exact-only seeds at sharp bends (0.781 to 0.800 against 0.740 to 0.768) and hairpins (0.801 to 0.836 against 0.682 to 0.767). The names output is 8 to 12 points worse at bends, its probability is not split there, and its top name reverses ten times as often between pictures (4.4 % against 0.4 %).

   Needs: nothing (a decision). Effort: none

   Expected effect: No time spent on a names model for this problem.

   Risk: The hairpin numbers come from one sheet; on the robot, model E and the team of five exact models failed alike on sheet 03, so the robot has not yet shown a difference.

6. Add 'steer by the chunk' as a driver option (direction from the chunk's place 3 or 5 steps ahead, length and twist as now) and try it only after the step-length test, as a second variable.

   Why: Offline gain +0.6 to +1.0 points overall on four models, +3 at hairpins, none at sharp bends; on run 6's pictures the 3-step point reversed as often as the exact step (18 against 20 of 31), the 5-step point 6 times.

   Needs: code only (ModelPolicy.decide must also return the chunk; one setting; tests), then a robot test. Effort: about 2 hours of code and tests; 6 to 9 runs inside a later session

   Expected effect: Small. Guess: a calmer direction where the model is unsure, most likely with the 5-step point.

   Risk: The far point cuts the swing to the outside of the bend that puts the block behind the puck (at sheet 03's first bend it points 10 to 20 degrees less outward), so the puck may be pushed straight on; the chunk carries no twist of its own.

7. Do not carry out 2 or 3 chunk steps per picture for now.

   Why: On the demonstrations' pictures the later chunk steps are as good as the first (place error 0.99 cm at 2 steps, 1.26 cm at 3), but the demonstrations twist the forearm by 10.5 and 15 degrees over 2 and 3 steps at sharp bends (12.5 and 19 at hairpins) and the model gives one twist per picture; errors of 1.0 to 1.3 cm could not be corrected between looks.

   Needs: nothing now (code and a robot test if it is ever wanted). Effort: none now

   Expected effect: Avoids a change whose only sure gain is speed.

   Risk: None from waiting.

8. If more demonstrations are recorded, decide the recording rule first, in this order: (1) leave the tool as it is, or (2) try 'a waypoint after 1 cm of tip travel or 10 degrees of forearm twist' or a smoothing of the TIP path in the plane with the height kept, in the simulator and then on 3 to 5 episodes of sheet 03. Do not average the joint angles.

   Why: Calculated on the 209 guided paths: joint averaging over 5 points cuts jumps over 90 degrees from 8.9 % to 1.7 % of step pairs but moves the tip up to 27 mm sideways and 7.5 mm down; the 10-degree forearm rule cuts steps under 0.3 cm from 10.4 % to 2.4 %. The models do not copy the jitter anyway (0.4 % reversals predicted against 9 % demonstrated), so the gain is cleaner labels and fewer near-empty steps, not a cure for the bends.

   Needs: code in record_episodes.py (between __record_guided_path and __exact_waypoints, lines 799 and 1208; the rule itself at line 1105), a robot test, then new demonstrations. Effort: about 1 day of code and simulator tests, 1 robot session to compare hand corrections with and without

   Expected effect: Median change of direction between real moves from 19 to about 13 degrees in new labels (calculated); 4 to 20 % fewer pictures per episode.

   Risk: The smoothed replay pushes the puck along a slightly different path than the hand did, so more hand corrections; twists up to 17 degrees per step with the 10-degree rule, above the driver's 10-degree limit and the 15-degree turn label; old and new episodes would then carry two label styles in one training set.

9. Treat a picture history as a later experiment on ki-014, not as the next step: the fine-tuned model with the previous picture as a second input, scored per kind of place with these scripts and with the copy test (same picture twice).

   Why: Frozen probes: +1.2 to +2.9 points at sharp bends, -2.4 to +2.2 at hairpins, inside the spread between seeds; the best variant gains only where the demonstration keeps its direction (+5.0) and loses where it changes (-1.3). It could tell the two halves of a manoeuvre apart, which one picture cannot, but it can also learn to repeat the last move.

   Needs: retraining and code (train_policy.py and model_mode.py), the professor's answer on earlier pictures of the same camera as input, then a robot test. Effort: about 1 day of code; 12 fold runs of about 2 minutes per seed on ki-014

   Expected effect: Unknown for the robot. Offline the clearest gain is at the line's end (+5 to +10 points on frozen features).

   Risk: Copying the last move in closed loop, which would turn a wrong direction into a lasting one.

10. Start every model run from the taught start pose of the demonstrations, or at least log the offset and keep it under 1 cm.

   Why: All standard-start demonstrations begin at tip x 14.0, y -1.5 cm with the forearm at 0.0 degrees; the ten runs began 1.3 to 5.0 cm away with the forearm at -9.7 to +9.8 degrees (run 6: 5.0 cm). In run 6 the model began to twist four steps later than the demonstrations do.

   Needs: code only (the model mode could go to start_pose() as the recording tool does), then a robot test. Effort: about 1 to 2 hours of code and a simulator check

   Expected effect: Removes one difference between the test and the training data; its size is unknown (runs 4 and 7 worked at 3.8 and 4.0 cm).

   Risk: The exam start may not be the taught start; then offset and turned starts must be trained, which the 35 offset and 25 turned episodes only partly cover.

11. Ask the professor on 23.11.26: (1) can the exam line contain hairpins or pointed corners (a U-turn of 150 degrees or more within 6 cm; only sheet 04 has two); (2) may the driver use the model's own step length and stop a run that makes no progress; (3) may the model see the previous picture of the same camera; (4) do hand-guided recoveries from the model's own mistakes count as our demonstrations.

   Why: The answers decide how much recording goes into hairpins (1,091 of 18,032 pictures, 9 passes from end B per hairpin) and whether the history experiment and the stall stop are allowed.

   Needs: the professor's answer. Effort: 10 minutes in the meeting; copy the questions into 05_meetings/requests_to_professor.md

   Expected effect: Fixes the scope of the next recording sessions.

   Risk: None.

### The verifier's challenges to the recommendations

1. Step-length test (fixed 1.0, predicted gain 1.0, predicted gain 1.3; sheets 03 and 04): the idea follows from the evidence and my numbers back it. Twist per cm of travel at sharp bends, on demonstration pictures (v13_twist_per_cm_held_out.py): with the fixed 1 cm step 50 % of the demonstrated for a model on an unseen sheet and 74 % for model E on its own sheets; with the asked length 79 % and 105 %. But a length gain of 1.3 gives most of that back (61 % and 81 %). The analyst's open question even suggests 1.15 to 1.3 for unseen sheets, which would lower the turning again. If a gain above 1 is tried, raise the twist by the same factor (the driver has twist_gain), otherwise the arm 'gain 1.3' is partly a repeat of the failure.

2. Same test, sheet 04: do not expect it to change. At sheet 04's first bend the fixed step was not the problem (asked 0.69 to 0.81 cm at steps 4 to 8 against 1.0 driven; twist 25 against 85 deg as a sum, below the lowest demonstration at 32), and at the spot of the back-and-forth the asked forward part was 0.00 cm, so with asked-length steps the robot would stand still there and only the stall rule would end the run (calculated). Nine of the 18 runs go to a sheet where the idea has little to say.

3. Same test, what it leaves out: every run so far and every proposed run uses the all-sheets model on sheets it was trained on; the exam line is unseen. Add at least 3 runs with a fold model on its unseen sheet (the checkpoint exists: 04_runs/gpu_ki014/2026-11-10_resnet18_chunk_stop_weight_25/checkpoints/without_path_03.pt) or on a newly printed track. Unseen-sheet models ask 0.71 of the demonstrated length, 0.64 of the twist, and do not slow down at a kind of place they have not seen (hairpin 0.63 cm against a pace of 0.47). With 3 runs per setting only 0 of 3 against 3 of 3 can be read; fix the start and tape the sheet, or the start differences of 11.11.26 (1.3 to 5.0 cm, up to 10 degrees) come back in.

4. Wider stall rule: take the box rule (15 pictures, diagonal under 4 cm, forearm under 15 degrees); it fires in run 6 only and in none of the 210 demonstrations treated as runs. Drop the offered alternative 'net travel under 1.5 cm over 12 pictures': it fires in 57 of 210 demonstrations (3 of the 79 without a hand correction, for example while the block walks round the puck at an offset start), and it is not 'two numbers in the settings' because the driver measures a box.

5. Do not add the keep-last rule: the decision is right, the stated reason is not. In run 6 the short answers were not close to chance (16 of 18 pointed back to the balance point), and run 7 breaks only because the rule takes two answers of 0.14 and 0.20 cm as its reference. What the evidence shows is that the fixed step, not the direction, made the back-and-forth.

6. Recovery demonstrations: the evidence shows the view was outside the data (distance 8.7 against 4.5), not that new episodes would fix it. Risk the analyst missed: for 'puck meeting a line from the side' one picture cannot say which way along the line is right (episodes run from both ends on every sheet), so such episodes can add contradicting labels and more averaging unless a fixed rule is written first. Keep to states where the line just left is still in view (puck 1 to 3 cm past a bend, forearm not yet turned). Do it after the step-length test, because the cheaper cure is not losing the line at the first bend; it costs 1 to 2 sessions plus index, audit and retraining, and needs the professor's answer.

7. Keep model E as the base for bends: the per-seed comparison is right, but the real alternative on the robot is the team of five (B), which scores 0.780 at sharp bends, level with E (0.780 to 0.794). E's lead is at hairpins only (0.744 against 0.801 to 0.836), one sheet and 544 pictures, and on the robot E and B failed alike. Fine as a default, not as a proven gain at bends.

8. 'Do not average the joint angles' rests on 6 of 44,328 guided points that sit at jumps of the guided path; elsewhere the smoothed tip stays within 2.5 mm of the hand's path (99th percentile). Joint averaging is not ruled out, it must only not run across such jumps. Since the models do not copy the jitter (0.2 to 0.45 % reversals), nothing here is urgent: leave the recording tool as it is, as option 1 says; a change would put two label styles into one training set.

9. Start every run from the taught start pose: right as a control for the next test, weak as a rule. The ten runs do not show that the offset matters (runs 4 and 7 worked at 3.8 and 4.0 cm, run 9 failed at 1.3 cm). It works against Emranul's wish for 'any orientation' if it becomes the standard; the real finding is that 170 of 210 demonstrations start from exactly one arm pose and one picture heading (-6.1 degrees). On the robot it is a new self-driven move: it needs Emranul's go, a checklist, and the puck and sheet placed only after the arm is at the start.

### What the verifier says was missed

1. The mechanism of run 6's back-and-forth: an overshoot round a balance point, not random flips (correlation -0.71 between the tip's sideways place and the asked sideways move; 16 of 18 off-centre answers point back; asked 0.15 cm, driven 1 cm). The fixed step caused the swinging; the asked length would give a standstill at the same spot (calculated). Files: v06_run6_mechanism.py and .out.txt.

2. What a length gain does to the turning: twist per cm at sharp bends falls from 79 % to 61 % of the demonstrated on unseen sheets and from 105 % to 81 % on the model's own sheets when the length gain goes from 1.0 to 1.3 (v13_twist_per_cm_held_out.out.txt). Length gain and twist gain have to be set together.

3. No robot run has yet used a model on a line it was not trained on, and none is proposed. Offline on its unseen sheet the fold model asks a median of 0.56 cm and a mean twist of -3.4 degrees at sheet 03's first bend against 0.77 cm and -5.1 degrees demonstrated (145 pictures of the standard starts).

4. The stall rules were tried on the ten fixed-step runs only, although they are meant for runs at the asked pace. On the demonstrations treated as runs the box rule is safe (0 of 210) and the net-travel alternative is not (57 of 210).

5. With the asked lengths the tip would turn earlier than the demonstrations at sheet 03's first bend (advance 3.8 to 5.3 cm against 6.0 to 8.9 cm for the 8 standard starts; calculated along the logged directions), so gain 1.0 carries a risk of cutting the bend on the inside. The analyst names the gain as an open question but not this number.

6. The two input files in inputs/ (the audit's line positions and traced lines) have no script and no original anywhere in the project; every kind of place rests on them. My spot check of 30 pictures and the independent hairpin count agree with them, but the copy cannot be traced.

7. The brief gives the puck as 4 cm; config.py has PUCK_PARAMETERS diameter 2.4 cm, and in the pictures the 1 cm line is 21 pixels wide against about 60 for the puck's top. Any distance read off a picture with the 4 cm figure is too large by two thirds; the analyst did not flag it.

8. Runs 8 (dashed loop, asked median 0.82 cm) and 3 (sheet lying differently) failed without short asked steps, so the step length does not cover the dashed line or the placement.

9. In run 6 a hand is in view at the top right from picture 55 on, while the puck is pushed across the lines; it may add to how unusual those views are (seen in v01_174108_E_sheet04_run1_steps_040_068.jpg).

### The analyst's corrections to the brief

1. Run 6, 'first 15 moves fine, then about 30 moves back and forth at the hairpin': the logged pictures show the puck leaving the line at the first bend between pictures 6 and 9, crossing about 5 cm of white paper and meeting the approach to hairpin 1 from the side at pictures 13 to 16. The back-and-forth (steps 19 to 50) happened 3 to 4 cm before the hairpin; the hairpin itself was never attempted (figure s04_run6_pictures.jpg).

2. 'At the first bend of sheet 03 the demonstrations themselves zig-zag': the jumping directions of plan 03-04 belong to tip moves of 0.15 to 0.21 cm made during twists in place; its real moves are consistent (16, -36, -28, -35, -96 degrees), and all 15 passes from end A twist the same way. Jumps over 90 degrees between two real moves are 3.8 to 8.2 % of pairs, against 25 to 35 % when a move under 0.3 cm is involved.

3. Scout step lengths: '20.5 % under 0.5 cm, 8 % under 0.2 cm' hold for all 18,312 non-stop rows of frames.csv (20.4 % and 8.0 %); for the 17,773 cleaned rows the models were trained on they are 18.2 % and 5.4 % (median 1.06 cm, mean 0.94). The 0.65 cm for steps with a twist of 6 degrees or more is confirmed.

4. 'Offline the models' twist was already about half the demonstrated size': per step it is 57 to 74 % (sum of predicted over sum of demonstrated twist: straight 0.66, gentle 0.68, sharp 0.57, hairpin 0.74). The low twist per cm on the robot comes mostly from the fixed step: per cm the model ASKED for, the twist was 5.6 to 6.5 degrees on sheet 03 (demonstrated 4.64) and 3.5 on sheet 05 (3.39); it was really low only on sheet 04 (3.58 against 5.65) and sheet 06 (3.20 against 4.42). The scout's per-cm numbers themselves are confirmed.

5. 'Sheet 04 as in plan 04-01' and the other placements: the tip started 5.0 cm from where the demonstrations of that plan start (x 18.2, y +1.3 against x 14.0, y -1.5 cm); over the ten runs the start was 1.3 to 5.0 cm away and the forearm at -9.7 to +9.8 degrees instead of 0.

6. The brief describes the driver as moving a fixed 1.0 cm. That is still the default, but model_mode.py and config.py in the worktree (file time 18:18 on 11.11.26) now also hold a 'step_length_rule' 'predicted' with 'step_length_gain' and 'step_most_m', and a stall rule (30 pictures, 3 cm, 15 degrees). Neither has been tried on the robot, and the stall rule as set would not have fired in run 6.

7. For anyone using the saved predictions: in the chunk runs 'angle_deg' is the direction of the chunk's FIRST step, not of the exact step the robot carries out, and no file holds the predicted length; both had to be computed again from the checkpoints (s02_model_outputs.py).

### The analyst's open questions

1. Will steps of the asked length carry the puck round the first bends of sheets 03 and 04? Only the robot can say; offline every measure follows the demonstrator's pictures.

2. Why did the model's twist come late and die at the first bend of sheet 04 in run 6 (25 degrees as a sum of sizes over 16 moves against 85 demonstrated), while on sheet 03 it twisted like the demonstrations? The start 5.0 cm from the demonstrated start is one candidate (a guess); a run from the taught start would test it.

3. Which gain for the asked length: 1.0 (fits the six known sheets, where model E's length equals the local pace) or about 1.15 to 1.3 (on unseen sheets the fold models ask about 15 % less; at sheet 03's first bend the asked pace was tighter than the demonstrations')?

4. Does the stop still work when the robot creeps near the line's end with asked-length steps (median 0.47 cm there, 22 % of answers under 0.3 cm)? That belongs to the stop analysis.

5. Can the exam line hold hairpins or pointed corners? All hairpin numbers here come from one sheet (1,091 pictures, 35 episodes, 9 passes from end B per hairpin).

6. Would a model trained with recovery episodes still average at a hairpin reached from the side? Not answerable without recording and a robot test.

7. Does a smoothed replay keep the puck on the line without more hand corrections? The calculation covers labels only, not pictures or the puck.

8. Is the difference between the twist-led bends of sheets 03 and 04 and the step-led loops of sheets 05 and 06 a matter of the line's shape (the forearm cannot follow a full loop past 119 degrees) or of the recording day? It matters for which style an unseen bend will get.

9. The kinds of place rest on the audit's line positions, good to 1 to 2 cm, and 653 pictures have no position; pictures near a boundary between two kinds can be in the wrong one.

## Dashed line (folder `dashed_line`)

**Analyst's answer.** Run 8 did not fail mainly because the line is dashed. The puck followed the dashed line through the crossing and into the loop, left the loop's hairpin about 1 cm wide, and then model E never made the "walk round the puck" (about 32 pictures, forearm unwound by about 119 degrees) that all 13 demonstrations make there and that the same model did make on sheet 05 in run 7; the driver stretched its short, undecided answers to full centimetres, pushed the puck 7 cm off the line and stopped with no line in view. The dashes themselves cost a few points of direction and make the models ask for 10 to 35 % less twist, no more. Both statements rest on one run per sheet.

**Verifier's overall verdict.** Partly. What happened in run 8 stands: with my own code (own ink and puck detection, own top view, the Recording session's course-photo centre line instead of the analyst's reference line) I get the same puck path (0.14 cm median difference), the same missing walk round the puck and the same stretched last steps. What does not stand is the causal half, "not mainly the dashes ... a few points, no more": it rests on one run per sheet and on demonstration pictures, while on the robot's own pictures the look of the line changes model E's answers a lot (19 degrees on average at steps 16 to 30 of run 8 when the gaps are painted solid, against 3 for a control repaint; 62 to 70 % less twist in the walk when sheet-06 dashes are cut into run 7's pictures). The dash question is open and needs a direct robot test (solid print of sheet 06's shape, dashed print of sheet 05's shape). All my scripts and outputs: D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/dashed_line/verify/ (nothing connected to the robot, no git change, no training, nothing written outside that folder).

### Verdicts on the claims

1. **confirmed**: Table positions taken from the pictures are good to about 0.5 cm, also with the forearm twisted beyond 60 degrees.

   Measured with my own pipeline (v04_run_tracks.py): the centre line placed from pictures twisted over 60 degrees lies 5.6 mm (run 8) and 7.4 mm (run 7, 10.2 at most) from the line placed from pictures under 45 degrees. My puck positions lie 0.12 cm (run 8) and 0.05 cm (run 7) from the analyst's (medians, 0.26 at most) and 0.07 cm on the 1,217 sheet-06 demonstration pictures. Taking the puck's red centre at 2.9 instead of 1.45 cm height moves it 0.2 to 0.3 cm. So about 0.5 cm, and 0.6 to 1.0 cm between little-twisted and far-twisted views.

2. **confirmed**: Sheets 05 and 06 are both loops with a crossing but differ: loop 24.5 against 31.5 cm, 10.0 x 5.5 against 11.8 x 7.1 cm, 40 against 70 cm2, crossing after 6.8 against at least 10.9 cm, right 270 against left 309 degrees, tightest corner 1.4 against 1.8 cm.

   From the Recording session's course-photo centre lines (v11_loop_shapes.py, calculated): loop 25.4 against 32.3 cm; 10.3 x 5.6 against 12.2 x 7.3 cm; 44 against 74 cm2 inside; start to crossing 6.2 against 11.5 cm; turning 278 right (06 from A) against 310 left (05 from B); tightest bend 1.3 cm against 2.2 cm radius.

3. **confirmed**: Sheet 06's dashes are about 2.0 cm long with gaps of about 1.0 cm; ink on 69 % of the line.

   Measured on my top view of run 8's pictures with under 30 degrees of twist, 1 mm cells (v09b_dash_lengths_plain.py): 12 dashes, median 2.1 cm (2.0 to 2.2); 11 gaps, median 0.9 cm (0.7 to 1.2); dash plus gap 3.0 cm; ink on 71 % of the seen line.

4. **confirmed**: The order of the hard parts differs: on sheet 06 from end A the tightest bend (10.7 cm after the crossing) comes before the walk round the puck (14 to 20 cm); on sheet 05 from end B the walk (15 to 19 cm) comes before the tightest bend (21 cm).

   Mine: tightest bend 11.3 cm (sheet 06) and 21.1 cm (sheet 05) after the crossing (v11); centimetres with 2.5 or more pictures per demonstration: 15 to 19 on sheet 06 from A, 15 to 18 on sheet 05 from B (v06; my zero lies about 0.6 cm earlier than the analyst's). That this order makes sheet 06 harder is a reading, not testable offline (the analyst says so).

5. **confirmed**: Run 8 step by step: puck within 0.4 cm of the line's middle up to step 15, 0.6 to 0.9 cm inside the loop at steps 16 to 19, outside from step 23, 1.0 cm out at step 25, 2.0 at step 30, 7.3 at the stop; further out than any demonstration from step 30. (Also: the brief's 'outside the loop by step 15' is wrong.)

   Mine (v04): 0.33 cm at most for steps 0 to 15; 0.5 to 1.0 cm inside at steps 16 to 19; 0.3 cm outside at step 23 (0.7 by the line placed from far-twisted pictures), 0.9 at step 25, 1.75 at step 30, 7.3 at step 40; about 21.6 cm of line left. By my placing the run leaves the demonstrations' 10th to 90th percentile and every demonstration picture at step 31 (analyst: 29 and 30). Difference to the analyst's 'beside the line': 0.14 cm median, 0.36 at most.

6. **confirmed**: Where the drift began (steps 23 to 26) model E's step pointed 64, 31, 25, 23 degrees to the outside of the line; it asked 0.78 to 0.88 cm and +0.6 to +1.7 degrees of twist; over the first 8 cm of the loop the run twisted as much as the demonstrations (57 against 60 degrees).

   Mine (v08, v09): asked direction against the line at steps 23 to 26: 70, 38, 29, 23 degrees outward; the demonstrations' median steps within 0.75 cm: 5 out, 11 in, 1 in, 11 out (quartiles from 40 in to 101 out, so wide). Asked lengths 0.88, 0.78, 0.87, 0.83 cm and twists +0.9, +0.6, +1.1, +1.7 (log). Twist over the first 8 cm of the loop: 53 degrees (run) against 58 per demonstration.

7. **confirmed**: All 13 demonstrations from end A walk round the puck 14 to 20 cm after the crossing (median 32 pictures, twist summed -119 degrees, 11 backward steps, forearm from +75 to -50); run 8 had 14 pictures, -2.6 degrees and no backward step there; in its last 14 answers it asked 7.2 cm and the driver moved 13.

   Mine (v06, v01): zone 14 to 20 cm: median 32 pictures (26 to 58), twist -119 degrees (-91 to -166), forearm from 74 to -51 (medians). From joint 4 alone, without any camera model, every one of the 13 unwinds by 147 to 216 degrees (median 168) over 26 to 61 pictures. Backward steps counted as 'beyond 90 degrees and longer than 0.05 cm': median 18 (12 to 35); the analyst's 11 (9 to 26) uses a stricter rule. Run 8: forearm 101.5 at most and 93.5 at the end, twist asked -2.6 degrees over steps 27 to 40, one answer just beyond 90 degrees (step 33: -90.2), 7.2 cm asked against 13.0 cm driven.

8. **confirmed**: The same model E did make the walk on sheet 05 in run 7 (27 pictures, +107 degrees, forearm -54 to +56).

   Mine: the puck stood at 16.3 to 16.7 cm after the crossing for 24 pictures (steps 33 to 56) while the forearm went from -54 to +55 degrees; twist asked +106 degrees over steps 31 to 56; 22 answers beyond 90 degrees in the run; the 18 demonstrations unwind by 141 to 186 degrees (median 160). New (v16): every all-sheets model (E, A, the team of five, the plain chunk model) asks for the walk on run 7's pictures (+102 to +140 degrees, 17 to 20 of 36 answers backward) and none does on run 8's (-0.4 to -3.7 degrees over steps 27 to 40), so the missing walk is not a quirk of model E.

9. **confirmed**: Run 8 reached the place of the walk in a state the demonstrations hardly show (puck 1.3 cm and more out, forearm 94 to 102, wrist 13.4 against the demonstrations' median 25); run 7 did not.

   Mine: at each demonstration's first backward step in the zone: puck -1.4 to +2.9 cm beside the line (median -0.4), forearm 59 to 96 degrees (median 82), wrist 13.4 to 31.6 (median 25.3; 12 of 13 above 18.5). Two things the analyst's wording hides: 2 of the 13 (06-02 at +2.9 cm with forearm 94 and wrist 13.4; 06-05 at +1.9 cm) do start the walk from as far out as run 8 was, and 4 of 13 (06-01, 02, 03, 05) carry the puck 0.7 to 1.8 cm outside (median) through the whole zone. And run 7's puck was 1.1 cm beside the line at its first backward answer (step 27), about as far as run 8's at steps 27 to 29 (1.1 to 1.4 cm): the puck's offset does not separate the two runs; the wrist and the forearm angle may. The reason cannot be tested offline.

10. **confirmed**: The held wrist (13.5 degrees) fits sheet 05's demonstrations but not the second half of sheet 06's (median 22 to 28 degrees from 14 cm after the crossing); sheet 06 has the most sideways and backward steps (53 %).

   Mine (v06, v09): sheet 06 from A: wrist median 22.3 to 28.4 degrees from 14 cm to the end, 67 to 93 % of those pictures more than 5 degrees from 13.5; sheet 05 from B: median 13.2 to 14.1 in nearly every centimetre. Per sheet (cleaned pictures): 52.2 % (06), 41.8 % (05), 21.8 to 30.3 % (02 to 04) more than 5 degrees off; 90th percentile 30.8 against 25.7; steps more than 60 degrees from forward: 52.6 % (06), 43.2 % (05), 21.3 to 41.1 % (01 to 04). Caveat left out by the analyst: sheet 01 has 41.1 % of pictures more than 5 degrees off (median 11.1) and was completed twice, and in sheet 05's loop 40 % of pictures are more than 5 degrees off (90th percentile 25.6). A wrist away from 13.5 does not by itself predict failure.

11. **confirmed**: The stop at step 40 was a false stop on blank paper and not caused by the dashes (0.27, 0.94, 0.90; with the line painted solid 0.31, 0.92, 0.90; the fold that never saw sheet 06 gives 0.71 and 0.81 at steps 32 and 33).

   Log: 0.27, 0.94, 0.90. My solid edit: 0.78, 0.87, 0.90. The two E-recipe folds that never saw sheet 06: 0.71 and 0.82 at step 32, 0.81 and 0.85 at step 33; models A, the team of five and the chunk model also reach 0.5 twice in a row at step 40 (v09, v16). On sheet 06's demonstration pictures the folds that never saw dashes give a stop chance of 0.5 or more in 0 to 0.3 % of pictures with a gap just ahead of the puck, against 3.4 to 7.9 % when no line ahead is in view (v12). One wording fix: dashes of the loop are still in view at the lower left of pictures 36 to 40; what is missing is line ahead of the puck.

12. **confirmed**: Offline, models that never saw sheet 06 steer in its loop as well as models that never saw sheet 05 steer in sheet 05's loop (0.78 to 0.80 against 0.73 to 0.77; twist sign 73 to 82 against 62 to 72 %; twist size 26 to 43 against 33 to 38 %; asked step 0.60 to 0.67 against a demonstrated 1.05 cm).

   Recomputed from the saved predictions with my own path target and my own placing (v07; the whole-sheet value reproduces the saved 0.788): in the loop 0.78 to 0.81 (sheet 06) against 0.73 to 0.76 (sheet 05) for the E recipe, its second seed and the A recipe; before the crossing 0.98 to 0.99 against 0.94 to 0.98; twist sign right 0.74 to 0.83 against 0.62 to 0.72; asked size 0.26 to 0.43 against 0.33 to 0.38 of the demonstrated; the chunk's first step 0.64 to 0.67 cm against a demonstrated median of 1.05 cm; demonstrated backward steps also asked backward 0.78 to 0.80 against 0.53 to 0.56.

13. **confirmed**: The weak, unstable part of the unseen dashed sheet is the tail after the loop (0.37 to 0.89 over 24 fold runs; E recipe folds 0.52 and 0.60).

   Mine, six fold runs (v07): 0.36 to 0.91 after the loop; the two E recipe folds 0.51 and 0.61 (222 scored pictures).

14. **confirmed**: A gap just ahead of the puck goes with only 3 to 7 points more direction errors; what matters is whether any line is in view ahead.

   Measured in the pictures themselves (v10): in the loop 0.79 to 0.83 within 45 degrees with a gap in the first centimetre beyond the puck (150 pictures) against 0.84 to 0.89 with a dash (225 pictures): 3 to 7 points, one standard error 3 to 4 points, so not clearly different from zero. No line ahead in view: 0.73 to 0.77 against 0.81 to 0.84 with ink in view (7 to 9 points by my definition, about 15 by the analyst's).

15. **corrected**: The dashes themselves cost a few points of direction and make the models ask for 10 to 35 % less twist, no more (headline; findings on dashes cut into sheet 05 and on other dash patterns).

   On demonstration pictures this is what the analyst measured (I did not repeat those 3,000-picture edits). On the robot's own pictures it is far more: with my own edit, sheet-06 dashes (20 / 10 mm, two positions of the pattern) or practice-track dashes (35 / 15 mm) cut into run 7's 89 pictures make model E ask for 62 to 70 % less twist in the walk round the puck (106 degrees down to 32 to 41) and give 8 to 11 instead of 22 answers beyond 90 degrees; its direction changes by 18 to 20 degrees on average (40 to 47 in the walk). Control, the same solid line repainted by the same tool: 12 % less twist, 22 of 22 backward answers kept, 3.4 degrees. No false stop appears. Calculated on edited copies, not a second run (v13_run_7_dashes_cut.py, v08_control_run_7_repainted.py).

16. **confirmed**: On run 8's own pictures, painting the line solid changes model E's direction by 4 / 21 / 8 degrees (steps 0 to 15 / 16 to 30 / 31 to 40); at steps 24 to 27 the outward push of 13 to 31 degrees becomes 2 to 11; still no walk; same stop.

   My own edit on my own placed line (v08): 4.8 / 19.3 (35 at most) / 9.1 degrees; at steps 24 to 27 the recorded answers point 38, 28, 23, 14 degrees outward, the solid ones 2, -2, -3, 3; twist 82.7 against 79.4 degrees; no answer beyond 90 degrees; stop unchanged. It is not an artefact of the edit: the control repaint of run 7 changes the direction by 3.4 degrees on average. Added: the two folds that never saw sheet 06 steer along the line at steps 24 to 27 on the dashed pictures as recorded (-34 to +12 degrees), so E's outward push there looks learned from sheet 06's own demonstrations, whose pucks are often outside the line at that place.

17. **unverifiable**: Run 8 did not fail mainly because the line is dashed.

   One run per sheet. For it: the dashed look matched training, models that never saw dashes steer the dashed loop as well offline, every all-sheets model refuses the walk on run 8's pictures, and painting the line solid does not bring the walk back. Against it: the same solid edit changes E's answers around the hairpin by 19 degrees on average, and dashes cut into run 7's pictures remove two thirds of the walk. Sheets 05 and 06 also differ in hairpin radius (2.2 against 1.3 cm), loop size, turning side, order of the manoeuvres and wrist use, all at once. Only a robot test separates these.

18. **corrected**: The driver stretched E's short, undecided answers to full centimetres and pushed the puck 7 cm off the line.

   The numbers are right (7.2 cm asked, 13.0 driven in the last 14 answers). But the puck left the line earlier, with nearly full asked lengths: at steps 19 to 28 E asked 0.65 to 0.96 cm (median 0.85), and at steps 23 to 30, while the puck went from 0.3 to 1.75 cm outside, it asked 5.8 cm against 8.0 driven. The stretching explains how far the puck was carried after step 30, not why it left the line.

19. **confirmed**: Dashed data: 25 episodes (scores 3: 17, 2: 5, 1: 3), 1,982 pictures, 1,936 after cleaning (10.7 % of 18,032), one night, one operator, one pattern; 15 from end A, 10 from end B; run 8's sheet 3 degrees and 3.3 cm from the main placement, run 7's 7 degrees and 3.9 cm from its nearest.

   Mine (v07, v05, v04): all counts identical; session 9.11.26, 00:22:33 to 02:14:14, operator emranul, ceiling light; 15 from A (1,230 cleaned pictures), 10 from B (706). Eight whole-line end-A demonstrations lie within 1.8 degrees and 0.5 cm of each other, 06-09 at +17 degrees, 06-10 to 06-13 at +7 to +8.5 degrees; sheet 05 from B: six placements over 51 degrees. Run 8: 3.5 degrees and 3.3 cm from the main placement, and 1.4 cm further to the left than any end-A demonstration (line centre at y = +0.45 cm against -2.8 to -0.9 cm). Run 7: 7.6 degrees and 3.9 cm from plans 05-20 and 05-21, 21.5 degrees from 05-01.

20. **confirmed**: Correction to the brief: sheet 05 has 21 episodes from end B and 9 from end A, sheet 06 has 15 from A and 10 from B; frames.csv's start_end column is stale.

   config.py's episode plan says so (sheet 05: plans 14 to 19 and 28 to 30 from A; sheet 06: plans 14 to 21, 24, 25 from B), and my own placing of every episode shows the puck travelling from the matching end in all 49 episodes I could place (the six short near-end starts 05-28, 05-30, 06-22 to 06-25 fitted poorly). frames.csv says B for all 30 of sheet 05 and A for all 25 of sheet 06.

21. **confirmed**: The demonstrations themselves do not keep the puck in the middle in the loops (on the ink in 60 % of pictures on sheet 06 from A, 47 % on sheet 05 from B); run 7 had the puck on the ink in 41 of 89 pictures and more than 1 cm off in 18.

   Mine (v15): sheet 06 from A: centre within 0.5 cm in 58 % of pictures, beyond 1 cm in 17 %, median 0.40 cm; sheet 05 from B: 45 %, 29 %, 0.59 cm. Run 7: 42 of 89 within 0.5 cm, 24 beyond 1 cm (the analyst's 18 is within the 0.5 cm accuracy), 2.1 cm at most, median 0.82 cm in the loop.

22. **confirmed**: The practice tracks do not match the course sheets: dashes 35 / 15 mm against 20 / 10 mm, tightest bend 3.5 cm against about 1.0 cm (sheets 03, 04), 1.4 to 1.5 (06), 1.8 to 2.2 (05); the robot completed sheets with bends of 1.8 cm or wider and failed on those of 1.5 cm or tighter.

   config.TEST_TRACK_PARAMETERS: dash 35 mm, gap 15 mm, smallest radius 35 mm. Course-photo centre lines, circle through three points 1 cm apart (v09): 0.8 cm (sheets 03 and 04), 1.2 to 1.3 (06), 2.1 to 2.2 (05), 9.7 (02); 9.2, 18.7, 5.4 and 3.4 cm of sheets 03 to 06 bend tighter than 3.5 cm (analyst: 9.3, 18.8, 5.2, 3.2). Sheets 03 and 04 are a little tighter than 'about 1.0 cm'.

23. **confirmed**: A dash-cutting augmentation is feasible on the real pictures (band within a median of 4 px of the line, 14 at most).

   My own cut works on run 7's 89 pictures (v13_cut_examples.jpg), but only after two fixes: the band must be shifted per picture (5 to 6 px on average, up to my limit of 14 px), and the robot's own parts need a proper mask (my first version painted paper onto the pusher block). The analyst's own example (plan 05-03, picture 31 in 05_edit_examples.png) shows long uncut stretches and white bars beside the line. Feasible, but every edited picture needs a check before it is trained on.

### The analyst's findings with their evidence

1. (measured, confidence high) How far the picture measurements can be trusted: table positions taken from the pictures with the fitted forearm camera model are good to about 0.5 cm, also with the forearm twisted beyond 60 degrees.

   Evidence: All files named in this answer are in D:/AI_Project/.claude/worktrees/training-analysis/06_deliverables/research_notes/Model and training/evidence/robot_test_2026-11-11/dashed_line/. 00_camera_model_check.out.txt: the line drawn from above from pictures with under 30 degrees of twist and from pictures with over 60 degrees fits best when shifted by 2.0 to 6.0 mm (runs 7 and 8, demonstrations 06-03, 06-06, 05-03, 05-06). 00_camera_model_check.png: the placed dashes drawn into run 8's pictures at 61 to 102 degrees of twist lie on the dashes the pictures show. The puck was found in 4,746 of 4,784 pictures (02_tracks.out.txt).

2. (measured, confidence high) Sheets 05 and 06 are both loops with a crossing but not the same shape. Sheet 06's loop is shorter (24.5 against 31.5 cm), smaller (10.0 x 5.5 against 11.8 x 7.1 cm, 40 against 70 cm2 inside), is reached sooner (6.8 against at least 10.9 cm from the start to the crossing) and turns the other way for the end the robot started from (right by 270 degrees from end A; sheet 05 left by 309 degrees from end B). Both have one tight corner of similar size: 1.4 cm radius on sheet 06, 1.8 cm on sheet 05.

   Evidence: 01_sheet_geometry.out.txt, 01_sheet_geometry.png, reference_line_05.csv, reference_line_06.csv: centre lines laid through the ink of the top views of runs 7 and 8 (bend measured over 2 cm of a line smoothed over 1.5 cm, so the true corners may be tighter). Line lengths 53.8 cm seen (sheet 05, its first 2 to 3 cm lay under the puck; course photo 56.6) and 45.6 cm (sheet 06; course photo 46.5). The course-photo centre lines give 2.2 and 1.5 cm for the same corners (08_dashed_data_and_tracks.out.txt).

3. (measured, confidence high) The dashes of sheet 06 are about 2.0 cm long with gaps of about 1.0 cm (16 dash passes from end A to end B, ink on 69 % of the line's length).

   Evidence: 01_sheet_geometry.out.txt: dashes 1.9 to 2.4 cm (median 2.0), gaps 0.7 to 1.2 cm (median 0.9); the dashes are twice as long as the 1 cm line is wide in the top view and in the course photo. The first dash lay under the puck in run 8 and was read from demonstration 06-15 (01_check_first_dash.png).

4. (measured, confidence medium) The order of the hard parts differs between the two sheets. On sheet 06 (from end A) the tightest bend comes 10.7 cm after the crossing, where the demonstrations already hold the forearm at 65 to 77 degrees (medians), and the walk round the puck comes after it, 14 to 20 cm after the crossing. On sheet 05 (from end B) the walk comes first (15 to 19 cm) and the tightest bend after it (21 cm), with the forearm freshly unwound.

   Evidence: Positions: 01_sheet_geometry_summary.json (tightest_at_cm_after_crossing 10.7 and 21.0). Walk zones and forearm angles per centimetre of line: 03_run_against_demonstrations.out.txt (pictures per demonstration per cm: sheet 06 6.0, 11.5 and 10.2 at 16, 17 and 18 cm; sheet 05 4.8, 8.6, 6.4 and 6.2 at 15 to 18 cm) and the third panels of 03_run_against_demonstrations.png. That this order makes sheet 06 harder for the robot is my reading, not a measurement.

5. (measured, confidence high) Run 8 step by step: the puck's centre stayed within 0.4 cm of the line's middle for pictures 0 to 15 (0.25 cm at most on the dashed straight before the crossing), cut the approach to the hairpin 0.6 to 0.9 cm on the inside at steps 16 to 19 (the demonstrations' median is 0.3 to 0.5 cm inside there), crossed to the outside at step 23 (0.6 cm, 12.3 cm after the crossing), was 1.0 cm out at step 25, 2.0 cm at step 30 and 7.3 cm at the stop (step 40), with 20.9 cm of line still to go. From step 29 it stays outside the demonstrations' 10th to 90th percentile (1.7 cm against 1.1 cm) and from step 30 it is further out than any of the 13 demonstrations (2.0 against 1.4 cm).

   Evidence: tracks.csv, 03_run_against_demonstrations.out.txt (step table and the line 'First step from which the run's puck stays outside ...: 29; outside every demonstration: 30'), 03_run_against_demonstrations.png (top row), 07_run_8_all_steps.jpg (all 41 pictures). Accuracy about 0.5 cm.

6. (measured, confidence high) Where the drift began (steps 23 to 26, just past the hairpin) model E's step pointed 64, 31, 25 and 23 degrees to the outside of the line's direction, where the demonstrations' median steps point 3 degrees out, then 19 and 12 degrees in, then 13 out. It asked for 0.78 to 0.88 cm (the driver did 1.0) and a twist of +0.6 to +1.7 degrees (demonstrations' median 0 to +2.9). Before that the run matched the demonstrations: 57 degrees of twist over the first 8 cm of the loop against 60 in the demonstrations, so the brief's 'twist per cm 2.58 against 4.43' is not a shortfall along the way but the missing walk.

   Evidence: 03_run_against_demonstrations.out.txt, columns step_vs_line and d_step_vs_line, asked_len, asked_twist, d_twist (demonstration steps within 0.75 cm along the line); twist sums from the per-centimetre table (run 8 steps 8 to 17: 57.1 degrees; demonstrations bins 1 to 8 cm: 59.8 degrees per demonstration).

7. (measured, confidence high) The walk round the puck is the step run 8 missed. Between 14 and 20 cm after the crossing every one of the 13 demonstrations from end A leaves the puck, walks the tip round it and unwinds the forearm: median 32 pictures (26 to 57), twist summed -119 degrees (-73 to -168), 11 backward steps (9 to 26), forearm from about +75 to about -50 degrees. Run 8 there: 14 pictures, -2.6 degrees of twist in total, no backward step in the whole run; it asked for 7.2 cm in its last 14 answers and the driver moved 13 cm.

   Evidence: 03_run_against_demonstrations.out.txt, section 'The walk round the puck' (per demonstration and 'the run there'); at steps 29 to 40 the demonstrations near the same place have a median twist of -4.4 to -5.6 degrees per picture and 38 to 45 % backward steps, the model asked +1.6 to -1.9 degrees and lengths of 0.13 to 0.77 cm. 07_demonstration_06-03.jpg shows the walk in pictures 35 to 56.

8. (measured, confidence high) The same model E did make this walk on sheet 05 in run 7: 27 pictures, +107 degrees of twist, 7 backward steps, forearm from -54 to +56 degrees, against the 18 demonstrations' median of 27.5 pictures, +119 degrees, 9.5 backward steps, -66 to +57 degrees. So the model knows the manoeuvre; it did not start it on sheet 06.

   Evidence: 03_run_against_demonstrations.out.txt (sheet 05 section), 07_run_7_walk_round.jpg (steps 28 to 62), 03_run_against_demonstrations.png (bottom row).

9. (measured, confidence medium) Run 8 reached the place of the walk in a state the demonstrations hardly show, run 7 did not. Sheet 06 demonstrations start the walk with the puck -1.0 to +2.8 cm beside the line (median 0.0), the forearm at 59 to 96 degrees (median 78) and the wrist at 13 to 32 degrees (median 25; 12 of 13 more than 5 degrees from the 13.5 the driver holds). Run 8 was there with the puck 1.3 to 7.3 cm out, the forearm at 94 to 102 degrees and the wrist at 13.4. Run 7 asked its first backward step with the puck 0.9 cm beside the line and the forearm at -54 degrees, inside the demonstrated ranges (-2.2 to +1.1 cm, -96 to -44 degrees, wrist median 14).

   Evidence: 08_dashed_data_and_tracks.out.txt, part 4. That this is the reason the walk did not start cannot be tested without the robot.

10. (measured, confidence medium) The driver's held wrist (13.5 degrees) fits sheet 05's demonstrations but not the second half of sheet 06's. On sheet 06 from end A the demonstrations' median wrist angle is 22 to 28 degrees from 14 cm after the crossing to the end of the line (80 to 90 % of those pictures are more than 5 degrees from 13.5); on sheet 05 from end B the median stays at 13.5 all along. Sheet 06 is also the sheet with the most sideways and backward steps: 53 % of its demonstrated steps point more than 60 degrees away from 'forward in the picture' (sheet 05 43 %, sheets 01 to 04 21 to 41 %).

   Evidence: 08_dashed_data_and_tracks.out.txt, part 2 (wrist more than 5 degrees from the held value in 52 % of sheet 06's pictures, 42 % of sheet 05's, 21 to 30 % on sheets 02 to 04; 90th percentile 30.8 against 25.7 degrees) and tracks.csv (joint_5_deg per 2 cm of line). Whether the held wrist caused the drift is not shown.

11. (measured, confidence high) The stop at step 40 was a false stop on blank paper and not caused by the dashes: stop chance 0.27, 0.94, 0.90 at steps 38 to 40 with the puck 5.6 to 7.3 cm beside the line and no line ahead of it in view since step 25; with the whole line painted solid in the same pictures it is 0.31, 0.92, 0.90.

   Evidence: 05_run_8_filled.csv and the run 8 table in 05_dash_edits_and_models.out.txt; ink_ahead_picture in tracks.csv. The fold model that never saw sheet 06 gives 0.71 and 0.81 at steps 32 and 33 on the same pictures.

12. (measured, confidence high) Offline, models that never saw sheet 06 (so never a dashed line) steer in its loop as well as models that never saw sheet 05 steer in sheet 05's loop: path within 45 degrees 0.78 to 0.80 against 0.73 to 0.77 in the loop, 0.98 against 0.94 to 0.98 before the crossing. Twist: right sign in 73 to 82 % against 62 to 72 %, asked size 26 to 43 % of the demonstrated size on sheet 06 and 33 to 38 % on sheet 05; asked step length 0.60 to 0.67 cm against a demonstrated 1.05 cm. The offline scores gave no warning that sheet 06 would fail and sheet 05 pass.

   Evidence: 04_fold_models_on_loops.out.txt and .png: E recipe folds (two seeds) and A recipe fold on the cleaned pictures (2,600 of sheet 05, 1,936 of sheet 06), scored on the exact step, which the driver carries out; the script's directions match the saved fold predictions within 0.001 degree. Backward steps of the demonstrations are also asked backward in 85 % (sheet 06) against 54 to 62 % (sheet 05) of pictures by the E recipe folds.

13. (measured, confidence medium) The weak, unstable part of the unseen dashed sheet is the tail after the loop: 0.37 to 0.89 within 45 degrees over 24 saved fold runs (median 0.70; the two E recipe folds 0.52 and 0.60), against 0.91 to 0.99 before the crossing and 0.60 to 0.83 in the loop. There the demonstrations push sideways in the picture (median step direction 80 and 96 degrees) with the wrist at a median of 27 to 28 degrees.

   Evidence: 10_parts_of_sheet_06_by_seed.out.txt (234 path pictures after the loop, 187 from end A and 47 from end B). Run 8 never got that far.

14. (measured, confidence medium) A gap just ahead of the puck goes with only slightly more direction errors: in the loop 0.75 to 0.79 within 45 degrees when the first centimetre beyond the puck is a gap (200 pictures) against 0.79 to 0.82 when it is a dash (485 pictures), a difference of 3 to 7 points in three models; whether the puck's centre is over a gap or a dash makes no difference (0.78 to 0.82 both ways). What does matter is whether any line is in view ahead of the puck: 0.66 to 0.72 with none against 0.83 to 0.87 with plenty, and the same on the solid sheet 05 (0.60 to 0.68 against 0.78 to 0.82), so that is the camera's view during the manoeuvres, not the printed gaps.

   Evidence: 04_fold_models_on_loops.out.txt (gap table, models that never saw a dashed line) and the control on sheet 05 computed from 04_fold_rows.csv (no ink in view ahead of the puck in 28 % of sheet 05's and 21 % of sheet 06's loop pictures).

15. (measured, confidence medium) Painting sheet 06's line solid in the real pictures does not help a model that never saw dashes: 0.786 to 0.790 on the whole sheet and 0.797 to 0.775 in the loop (second seed 0.794 to 0.785 and 0.789 to 0.773). It asks for 9 and 19 % more twist and 10 % longer steps on the solid version. Cutting dashes into sheet 05 (2.0 cm, gaps 1.0 cm) costs the model that never saw sheet 05 1.5 points (0.820 to 0.805) and model E 3.3 points (0.950 to 0.917), with 24 % less twist asked by E and fewer of the demonstrations' backward steps recognised (0.82 to 0.69; the model that never saw sheet 05: 0.63 to 0.44).

   Evidence: 05_dash_edits_and_models.out.txt, 05_edit_scores.csv, 05_dash_edits_and_models.png; the edits at the size the models see are in 05_edit_examples.png (1,879 pictures of sheet 06, 1,296 of sheet 05). By bend tightness the effect is within 4 points either way for unseen-sheet models (06_dashes_at_tight_bends.out.txt). The edits are not perfect (flat ink colour, cuts through the crossing, nothing painted next to the robot's own parts), so a few points of these changes may be the edit and not the dashes.

16. (calculated, confidence low) On run 8's own pictures the dashes may have weakened the turn out of the hairpin but did not cause the missing walk: with the line painted solid model E's asked direction changes by 4 degrees on average at steps 0 to 15, by 21 degrees at steps 16 to 30 (up to 46) and by 8 degrees at steps 31 to 40. At steps 24 to 27, where the recorded answers pushed 13 to 31 degrees to the outside of the line's direction, the solid version asks within 2 to 11 degrees of it. It still asks for no walk (72 against 83 degrees of twist in total, 2 against 1 answers beyond 90 degrees) and stops the same.

   Evidence: 05_run_8_filled.csv, run 8 table in 05_dash_edits_and_models.out.txt (the script reproduces the logged answers exactly on the pictures as recorded). This is the same model on edited copies of pictures that the dashed run produced, not a second run.

17. (measured, confidence medium) The look of the dashes matters moderately. Cut into sheet 05's pictures, dashes like sheet 06's (20 / 10 mm) or like the practice tracks' (35 / 15 mm) cost model E 2 to 3 points of direction (0.936 to 0.902 and 0.915), short or sparse dashes (10 / 10, 10 / 20 mm) 5 to 7 points (0.884, 0.864), and all patterns make it ask 20 to 37 % less twist. False stops do not rise (1.4 to 1.7 % of pictures). For the model that never saw sheet 05 the direction moves by 0 to 4 points.

   Evidence: 09_other_dash_patterns.out.txt (432 pictures of sheet 05, 389 path pictures; one standard error is about 2 points).

18. (measured, confidence high) Dashed data: 25 episodes of sheet 06 (scores 3: 17, 2: 5, 1: 3), 1,982 pictures recorded and 1,936 after cleaning, 10.7 % of the 18,032 cleaned pictures; all from one night (9.11.26, 00:22 to 02:14), one operator (Emranul), ceiling light, one dash pattern, one printed sheet. 15 episodes start at end A (13 whole-line, 2 near the end; 1,230 pictures) and 10 at end B (706 pictures). From end A the sheet lay in 3 placements found (8 of the 13 whole-line demonstrations in one, within 2 degrees and 1 cm; the others at +8 and +17 degrees from it) plus one not found; sheet 05 from end B was demonstrated in 6 placements spread over 51 degrees. Run 8's sheet lay 3 degrees and 3.3 cm from the main sheet 06 placement, run 7's 7 degrees and 3.9 cm from its nearest, so the placement does not explain why 06 failed and 05 passed.

   Evidence: 08_dashed_data_and_tracks.out.txt part 1, registrations.csv (the reference line laid onto each episode's own top view; 53 of 55 episodes placed, check picture 02_registration_check.png).

19. (measured, confidence medium) The demonstrations themselves do not keep the puck in the middle of the line in the loops: its centre is on the ink (within 0.5 cm) in 60 % of the pictures of the sheet 06 demonstrations from end A (more than 1 cm off in 17 %, median 0.37 cm) and in 47 % on sheet 05 from end B (more than 1 cm off in 27 %, median 0.54 cm). Run 7, the 'successful' loop, had the puck on the ink in 41 of 89 pictures and more than 1 cm off in 18 (2.0 cm at most, 0.79 cm median in the loop).

   Evidence: 03_run_against_demonstrations.out.txt (lines 'The run's puck' and 'The demonstrations' puck'); accuracy about 0.5 cm, so the 60 and 47 % include measurement scatter.

20. (measured, confidence high) The printable practice tracks do not match the course sheets in two settings: their dashes are 35 mm with 15 mm gaps (sheet 06: 20 and 10 mm) and their tightest allowed bend is 3.5 cm radius, while the course sheets bend down to about 1.0 cm (sheets 03 and 04), 1.4 to 1.5 cm (06) and 1.8 to 2.2 cm (05). In this robot test the sheets whose tightest bend is 1.8 cm or wider were completed (01, 02, 05) and those with 1.5 cm or tighter failed (03 three times, 04, 06), with one run per sheet apart from 03.

   Evidence: 08_dashed_data_and_tracks.out.txt part 3 (config.TEST_TRACK_PARAMETERS against step 1 and the course-photo centre lines in Recording tool/2026-11-06_sheet_measurements.json; 9.3, 18.8, 5.2 and 3.2 cm of sheets 03 to 06 are tighter than 3.5 cm radius). The pass or fail pattern is from the brief's table.

21. (measured, confidence high) A dash-cutting augmentation is feasible on the real pictures: with the sheet's centre line placed on the table per episode and the camera model, gaps were cut into 1,296 pictures of sheet 05 (and the line of 1,879 pictures of sheet 06 painted solid) automatically, the band landing within a median of 4 px of 640 (14 px at most) of the line, in 0.04 to 0.08 s per picture.

   Evidence: picture_edits.py, 05_edit_examples.png, 05_dash_edits_and_models.out.txt (first lines per sheet). Limits seen: no cut where the line passes the robot's own parts or the puck, cuts also pass through the crossing, and it needs each sheet's centre line (made here for 05 and 06 only; for 01 to 04 the course-photo centre lines exist).

### The analyst's recommendations

1. Before changing anything, repeat the two loop runs: sheet 06 from end A three times and sheet 05 from end B three times with model E and today's driver, logged as today.

   Why: Every conclusion about 'sheet 05 works, sheet 06 fails' rests on one run each, and run 7 itself had the puck more than 1 cm beside the line in 18 of 89 pictures.

   Needs: robot test. Effort: About 20 minutes of one session (a run takes under 2 minutes plus placing the sheet).

   Expected effect: A baseline to judge every later change against (for example 0 of 3 against 3 of 3, or something less clear).

   Risk: None beyond a normal run; write the checklist first and ask Emranul before anything moves.

2. Driver: stop turning short answers into full centimetres. Carry out the asked length within limits (for example 0.2 to 1.2 cm) with the asked twist, at least when the asked length is under about 0.6 cm. Check it first on the logged runs with the driver's own executor mode.

   Why: In run 8's last 14 answers the model asked for 7.2 cm in total and the driver moved 13 cm, which carried the puck from 1.3 to 7.3 cm off the line; in run 7's walk round the puck the model asked for 0.14 to 0.38 cm per step for nine steps.

   Needs: code only, then a robot test. Effort: 1 to 2 hours of code and offline checks, then part of a session.

   Expected effect: The puck is no longer pushed away while the model is undecided, which gives the walk round the puck a chance to start; it will not create the walk by itself (guess).

   Risk: The run can stall on very short answers, which is why the report chose fixed steps; it needs a smallest step or a limit on short steps in a row.

3. Decide how the wrist is handled on the loops: either let the driver follow the model's lean output with the wrist, or record the next loop demonstrations with the wrist kept near 13.5 degrees so that the driver's fixed wrist matches what the model saw.

   Why: On sheet 06 from end A the demonstrations hold the wrist at a median of 22 to 28 degrees from 14 cm after the crossing to the end and 12 of 13 start the walk with it more than 5 degrees from the held value; on sheet 05, which the robot completed, the median is 13.5.

   Needs: code and a robot test (following the lean), or new demonstrations and retraining (wrist kept still). Effort: 0.5 to 1 day of code with the bracket's wrist limit and the rod's lean limit checked; or no code and a rule for the next recording session.

   Expected effect: Unknown (guess): it removes one steady difference between the demonstrations and the robot in the second half of sheet 06.

   Risk: Wrist and lean limits near the camera bracket; nothing runs on the robot without Emranul's go. The link between the held wrist and the failure is not proven.

4. Record corrective demonstrations on sheet 06 aimed at the failure: about 12 short episodes that start with the puck 1 to 2 cm outside the line just after the hairpin (11 to 16 cm after the crossing, forearm at 80 to 100 degrees) and bring it back onto the line and through the walk, plus about 9 whole-line episodes from end A in two or three new placements (sheet turned by about 20 degrees either way and shifted by 3 to 5 cm). Then retrain the all-sheets model.

   Why: The demonstrations show the walk only from states with the puck on the line (median 0.0 cm beside it); run 8 arrived 1.3 cm and more outside. 8 of the 13 whole-line demonstrations from end A lay in one placement, against six placements over 51 degrees for sheet 05. The false stop also came from a 'puck on blank paper' view that training never showed as 'keep going'.

   Needs: new demonstrations, retraining, then a robot test. Effort: About 20 episodes, roughly 1,000 to 1,200 pictures, 1.5 to 2 hours at the robot with taught starts; one training run on ki-014.

   Expected effect: The model has seen the off-line state it met in run 8 and what to do there (guess; no number can be promised).

   Risk: If the driver still cannot carry out what is demonstrated (wrist, step length), more demonstrations of the same style will not help; the way back onto the line must be demonstrated the same way every time.

5. Try by hand at the robot (motors off, no model) whether the dashed loop can be driven in a simpler style: unwind the forearm before the hairpin, on the nearly straight 5 to 8 cm after the crossing, or enter the loop with the forearm already twisted the other way, with the wrist kept still. If it works, re-record sheet 06 in that style.

   Why: On sheet 06 the tightest bend (1.4 cm radius, 10.7 cm after the crossing) is taken with the forearm at 65 to 77 degrees, sideways pushes and a bent wrist, and the walk only comes after it; on sheet 05 the walk comes before the tight bend.

   Needs: robot time for a hand trial, then new demonstrations and retraining. Effort: Half a session for the trial; 2 to 3 hours to re-record 13 to 25 episodes if the style is adopted.

   Expected effect: A manoeuvre the driver can repeat (guess).

   Risk: Two styles for the same sheet in the data make the model average them; either replace the old sheet 06 episodes or keep one style.

6. Regenerate the practice tracks with dashes like sheet 06 (20 mm, gaps 10 mm) or a spread around it (dashes 15 to 35 mm, gaps 8 to 15 mm), and allow bends down to about 1.5 cm radius in some of them.

   Why: The tracks now use 35 / 15 mm dashes and no bend tighter than 3.5 cm, while the course sheets bend down to 1.0 to 2.2 cm and the robot failed exactly on the sheets with bends of 1.5 cm or tighter.

   Needs: code only (two settings in config.TEST_TRACK_PARAMETERS, changed by the session that owns config.py) and printing. Effort: 30 minutes and the printing.

   Expected effect: Practice results that carry over to the course sheets and to an exam line of the same kind.

   Risk: Tracks that are too hard waste robot sessions; keep some gentle ones.

7. Build the dash-cutting augmentation only as insurance against an exam line whose dashes or shape differ from sheet 06: a second copy of the solid sheets' pictures with dashes of random length (10 to 35 mm) and gaps (8 to 20 mm), used for part of the training pictures, then retrain and compare on the edited pictures and on a printed dashed practice track.

   Why: Dashes are now seen on one shape in one pattern only; on dashed versions of sheet 05 model E loses 2 to 7 points of direction and 20 to 37 % of its twist size. But painting sheet 06 solid gains a model that never saw dashes nothing, so little is to be won on sheet 06 itself.

   Needs: code, retraining, and a robot test on a printed dashed track. Effort: 1.5 to 2 days: place the line in the 156 episodes of sheets 01 to 04 (centre lines exist from the course photos), write the dashed picture cache, add the switch to training, one or two runs on ki-014.

   Expected effect: Keeps the twist size and direction on dashed versions of known shapes; on sheet 06 itself at most about 2 points (measured upper bound from the solid edit).

   Risk: The edit leaves marks (uncut pieces next to the robot, cuts through crossings) that a model could learn; it assumes a dashed line is driven exactly like the solid one; it fixes none of the driver problems.

8. Ask the professor on 23.11.26: can the exam line be dashed, and with what dash and gap lengths; can it cross itself or loop; how tightly may it bend; may the pusher leave the puck and walk round it; where does the sheet lie at the start and how many attempts are allowed.

   Why: These answers decide whether the days for corrective loop demonstrations, a new loop style and the dash augmentation are worth spending: all six course sheets are known, the exam line is not.

   Needs: the professor's answer. Effort: Five minutes in the meeting.

   Expected effect: The loop and dash work can be sized to what the exam needs.

   Risk: None.

### The verifier's challenges to the recommendations

1. Repeat sheet 06 from A and sheet 05 from B three times each before changing anything: follows from the evidence, but alone it only tells whether the result repeats (0 of 3 against 3 of 3 is still p = 0.10 by Fisher's exact test) and says nothing about dashes. In the same session, with no code change, add a solid print of sheet 06's shape and a dashed print (20 / 10 mm) of sheet 05's shape (both can be drawn from the course-photo centre lines with the practice-track generator), and one run each from the two ends that were demonstrated but never driven (sheet 06 from B, sheet 05 from A). Robot risk for those: config.py notes that from end B the paper may reach the base, and that sheet 05 from A puts the loop about 40 cm out (rod lean, reach limit); write the checklist and ask Emranul first.

2. Driver: use the asked length when it is short. The evidence supports it only for the end of run 8; the puck left the line while E asked 0.65 to 0.96 cm. Risk the analyst did not name: the one loop that worked relied on stretched steps. In run 7's walk E asked 0.14 to 0.38 cm for nine steps (47 to 55, 2.3 cm in total) while turning the forearm by 51 degrees, and the driver's 9 cm carried the tip round the puck; the demonstrations move the tip about 0.9 cm per picture in the walk. With the asked lengths the twist per centimetre would rise about fourfold and the tip may never get round the puck. Test it on sheet 05 first, on the same day as a fixed-step baseline, and keep the fixed step as fallback. The switch already exists in the working tree (MODEL_MODE_PARAMETERS 'step_length_rule': 'predicted'), so it is a setting plus a floor, not new code.

3. Let the driver follow the model's lean output with the wrist: the lean output is too weak for that. On held-out sheets it has the right sign in 80 to 86 % of rows with a lean change of 1 degree or more, 28 to 43 % of the demonstrated size and a correlation of 0.31 to 0.46; on held-out sheet 06 it adds up to 12 to 18 degrees per episode where the demonstrations add up to 36 (v09). That means new solver code near the lean limit, with a puck that is taller than wide and tips, for about a third of the demonstrated motion. The case against the held wrist is also thinner than presented (sheet 01: 41 % of pictures more than 5 degrees off, completed twice; sheet 05's loop: 40 %). Do the hand trial with the wrist kept still first.

4. Corrective demonstrations on sheet 06 (about 12 short and 9 whole-line episodes): (a) sheet-specific, while Emranul asked for no overfitting to the six sheets and the same kind of failure hit sheets 03 and 04 (bends of 0.8 cm radius); (b) its premise is overstated: 2 of 13 demonstrations already start the walk 1.9 and 2.9 cm outside and 4 of 13 carry the puck 0.7 to 1.8 cm outside through the zone, and no model walks on run 8's pictures; (c) 12 short episodes are roughly 350 pictures, 2 % of 18,032; (d) more dashed pictures on the one dashed shape tie 'dashed' to 'sheet 06's manoeuvres' even more; (e) the existing loop demonstrations have the puck's centre on the ink in only 58 % (06) and 45 % (05) of pictures, so new ones must be recorded and scored for centring or the model keeps learning to push beside the line. Rank it after the paper test and the driver test.

5. Practice tracks with sheet-06 dashes and bends down to about 1.5 cm: follows, with two gaps. Sheets 03 and 04 bend to about 0.8 to 1.0 cm radius, so 1.5 cm still leaves the practice tracks gentler than two of the six course sheets. And each track should be printed both solid and dashed; otherwise dashes and shape stay tied together as on the course sheets.

6. Dash-cutting augmentation 'only as insurance', worth 'at most about 2 points on sheet 06': underrated. The 2 points come from demonstration pictures. On the robot's own pictures the dashed look removes 62 to 70 % of the twist E asks for in the walk (run 7's pictures) and shifts E's direction by 19 degrees on average around run 8's hairpin (control 3). If the exam line may be dashed this is needed, not insurance; decide it straight after the professor's answer. Before any robot time a retrained model can be checked offline with the same test (dashed copies of run 7's pictures: does it still ask for the walk?). The edit needs a per-picture quality check (see the failed example in the analyst's own figure).

7. Ask the professor on 23.11.26: right questions, wrong date. The meeting is 12 days away and the answers decide two to four days of work; ask now in writing (e-mail or Moodle) and keep 23.11 for follow-up.

8. Hand trial of a simpler style for the dashed loop: no objection (motors off, no model), and it should come before the wrist work and before any new demonstrations. If the style is adopted it replaces 13 to 25 episodes and changes how the camera sees the approach (the camera turns with the forearm), so one style only, as the analyst says.

### What the verifier says was missed

1. No test that separates dashes from shape: a solid print of sheet 06's shape and a dashed print of sheet 05's shape, driven by the same model, would answer the headline question in about 15 minutes of robot time. The analyst lists the solid copy only as an open question.

2. The dash-cut test was run only on demonstration pictures of sheet 05 that model E had trained on (0.950 to 0.917). On run 7's own pictures, which E never trained on, the same kind of cut removes two thirds of the walk round the puck (verify/v13_run_7_dashes_cut.py, three patterns).

3. No control for the picture edits themselves. Repainting run 7's already solid line with the same tool changes E's direction by 3.4 degrees on average, so the 19 to 21 degrees around run 8's hairpin are an effect of the dashes on E's answers, not of the tool (verify/v08_control_run_7_repainted.py).

4. Other models on the same pictures: A, the team of five and the plain chunk model also ask for no walk on run 8's pictures (-0.4 to -3.7 degrees of twist over steps 27 to 40) and all ask for it on run 7's (+102 to +140 degrees). Switching model will not fix the dashed loop, and the missing walk is not chance in one model (verify/v16_other_models_on_the_runs.py).

5. The two start ends that were demonstrated but never driven: sheet 06 from end B (8 whole-line demonstrations) and sheet 05 from end A (5). In both the wrist is more than 5 degrees from 13.5 in 46 to 51 % of loop pictures and in 81 to 100 % after the loop (verify/v12_other_ends_and_stops.py), so runs from those ends also test the wrist explanation on a solid line.

6. The brief's puck size is wrong and was not corrected: the puck is 2.4 cm wide and 2.9 cm tall (config.PUCK_PARAMETERS; I measure 2.3 to 2.5 cm in the pictures, verify/v14_puck_size.py), not 4 cm. So from 1.7 cm beside the line the puck no longer touches it: run 8 from step 30, run 7 in 11 of 89 pictures, sheet 05's demonstrations in 15 % of their pictures.

7. No recommendation follows from the finding that the demonstrations themselves leave the line in the loops, although Emranul wants the puck in the middle the whole way: an imitation model cannot centre better than its teacher. Loop episodes could be scored or filtered by centring (worst on sheet 06 from A: 06-01, 06-02, 06-05 with the puck 1.0 to 1.8 cm outside through the walk zone).

8. E's outward push just past the hairpin (steps 24 to 27) is not shared by the models that never saw sheet 06, which steer along the line on the same dashed pictures; the numbers are in the analyst's own table but not in the answer. Together with the cut test it means E's manoeuvres are tied to the look and shape of the sheet they were learned on, which is the real risk for an unseen exam line.

9. The camera model's error grows with the forearm twist (5.6 and 7.4 mm between views under 45 and over 60 degrees, up to 10 mm), which is exactly where the walk happens; it limits every 'beside the line' number in the loop and is why any picture edit needs a per-picture shift.

### The analyst's corrections to the brief

1. The task's 'sheet 05 only from end B, sheet 06 only from end A' is wrong. It comes from the stale start_end column of frames.csv. By the audit's corrected ends, which the trained models used, sheet 05 has 21 episodes from end B and 9 from end A (plans 05-14 to 05-19 and 05-28 to 05-30; 8 after cleaning) and sheet 06 has 15 from end A and 10 from end B (plans 06-14 to 06-21, 06-24, 06-25). The puck's travel along the placed line confirms the corrected end in all 53 placed episodes.

2. Run 8: 'puck outside the loop by step 15' is not what the pictures show. The puck's centre was within 0.4 cm of the line's middle up to step 15, 0.6 to 0.9 cm inside the loop at steps 16 to 19, first outside at step 23 (0.6 cm), 1 cm out at step 25 and further out than any demonstration only from step 30.

3. 'The same model did the solid alpha loop' needs two cautions: it is one run, and in it the puck was more than 1 cm beside the line in 18 of 89 pictures (2.0 cm at most). What that run does show is that model E can make the walk round the puck (27 pictures, 107 degrees of twist).

4. The sheets did not lie 'as in plan 05-01' and 'as in plan 06-01' exactly: run 7's sheet lay 21.5 degrees from where it lay in demonstration 05-01 (nearest demonstrated placement 7 degrees and 3.9 cm away, plans 05-20 and 05-21), run 8's 3 degrees and 3.3 cm from the placement of plans 06-01 to 06-08. The 'STD' episodes of one sheet also do not share one placement (05-01 and 05-02 lie 31 degrees from 05-03 to 05-08), and the arms were set by hand at the start of the robot runs (tip start between x 12.2 and 18.2 cm).

5. The scout's 'twist per cm on sheet 06: 2.58 on the robot against 4.43 demonstrated' hides where the difference is: over the first 8 cm of the loop run 8 twisted as much as the demonstrations (57 against 60 degrees); the whole shortfall is the walk round the puck that never came (-2.6 against -119 degrees).

6. Not in the brief but in config.py: TEST_TRACK_PARAMETERS describes its dashes (35 mm, gaps 15 mm) as 'as sheet 06' and guesses the loops' radius at 40 to 50 mm. Measured: sheet 06's dashes are about 20 mm with 10 mm gaps, and the tightest bends of sheets 05 and 06 are about 1.8 and 1.4 cm.

7. A detail on the offline numbers: for the chunk recipe (model E) the saved 'path within 45' scores the chunk's first step, while the robot carries out the exact step. On sheets 05 and 06 the two agree within about 1 point (0.811 against 0.813, 0.788 against 0.782), so the brief's numbers stand.

### The analyst's open questions

1. Is run 8's failure repeatable, and run 7's success? One run each is all there is.

2. Would model E have started the walk round the puck on sheet 06 if the puck had still been on the line, or does the held wrist (13.5 degrees against the demonstrations' 22 to 28) stop it anyway? This cannot be answered from logs; it needs the robot.

3. For the professor: will the exam line be dashed and with what dash and gap lengths, can it loop or cross itself, how tight can it bend, and is leaving the puck to walk round it allowed?

4. Why is the tail after the loop of sheet 06 so unstable offline (0.37 to 0.89 between 24 fold runs)? Few pictures (234), sideways pushing and a bent wrist are the likely reasons, not checked further.

5. Where did the sheet lie in the two short episodes 06-22 and 06-23 (36 pictures)? Too few dashes were in view to place the line; they were left out of everything measured along the line.

6. The sheet rotations found for the turned starts of sheet 06 do not match their names (TL10 lies 17 degrees and TR10 8 degrees to the left of the standard placement). A question for the Recording session; it does not change the conclusions here.

7. The step directions on run 8's pictures change by 13 to 46 degrees around the hairpin when the line is painted solid. Is that a real weakness of model E on dashed tight bends or an effect of the edit? A printed solid copy of sheet 06's shape would settle it on the robot.
