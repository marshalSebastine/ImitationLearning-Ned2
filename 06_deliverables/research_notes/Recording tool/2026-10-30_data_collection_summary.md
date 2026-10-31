# Data collection: the two decisions and the plan in three pages

Short version of `2026-10-30_data_collection_analysis.md` (30 October 2026) for Emranul Hoque and Marshal Sebastine. The long note has the evidence, the references and the arithmetic; this page has only what you need to decide. Everything here is a recommendation. You decide.

## Decision 1: how do we collect hand-guided demonstrations?

**The question.** Marshal's tool has two modes. In keyboard mode a person looks at the camera picture and presses one of seven keys per 1 cm step. In hand-guided mode a person switches the motors off, pushes the puck to the goal by holding the arm while the tool records where the tip went, puts the puck back, and the robot replays that path in 1 cm steps; only the replay pictures become training data. Should hand-guiding stay, change, or go?

**Recommendation: keyboard stepping is the main method; hand-guiding stays only as a measured pilot, with one change.**

- Why keyboard first. The pictures the model trains on must look exactly like the pictures it sees when it runs. Keyboard stepping records through the very same loop the trained model will later run: the arm stands still, no hand, no blur, picture taken after the move finished. It is also the most consistent method (the person has unlimited time per step) and the safest (nobody touches the arm). The usual complaint about keyboards, that they are slow, does not apply here: the robot itself allows only one step per second.
- Why hand-guiding is doubtful. The replay follows the old tip path, not the puck. If the puck slides a little differently during the replay (identical pushes scatter by 3 to 12 mm in the one careful study), the picture shows an off-centre puck while the label says "follow the old path". That is a wrong label exactly where a right one matters most. In the only real-robot comparison of hand-guiding against teleoperation, hand-guiding was best on two tasks but worst on the pushing task. Each hand-guided episode also costs about 1.5 times a keyboard episode.
- The one change to make now. The tool gives the person 400 ms to override each replay step. Reacting to a seven-way choice takes longer than that, so overrides would land one step late or not at all. Change it to: Enter executes the proposed step, a direction key overrides, no timeout.
- How the pilot decides. This weekend on sheet 01 (rehearsal, stock camera) and once more on sheet 02 with the new bracket: 10 keyboard episodes and 10 hand-guided episodes per operator. Hand-guiding stays if overrides are under 15 percent of steps, discards under 20 percent of episodes, and the time per finished episode under 1.5 times keyboard. Above 30 percent overrides it survives only as an on-screen suggestion while a person still presses every key. If the safety test fails (Ctrl-C while someone holds the torque-free arm must not move the robot), it goes.

**What the professor said.** Both methods are allowed; in his thesis hand-guiding was faster. Question 1 for him: does a replay with a student confirming every step count as "pushing the puck ourselves"?

## Decision 2: which output classes (actions) does the model have?

**The question.** The tool has seven: forward, forward-left, forward-right, left, right, back, stop, all named in the picture's frame (forward is the top of the picture). Marshal proposed back-left, back-right and a sideways hop and deferred them on 29.10. Keep seven, or change?

**Recommendation: keep the seven. Add two tool keys, not classes.**

- Why not more classes. Every extra class splits the examples. With a few hundred episodes the rare moves (left, right, back, stop) are already the ones learnt worst; a model without memory can also oscillate on back moves, and two operators disagree more on near-equal classes. A hop to the puck's far side leaves the puck out of the picture, so the camera cannot even see what that action depends on.
- The two keys. `u` marks the previous step as a mistake (training drops it and the step that undid it, the episode survives instead of being discarded). `l` marks "lost" (the line left the picture, so this frame gets no label). Both fix real problems without touching the class set.
- What must come with the seven classes. A written stepping rule both of you follow, drawn as a reference band on the screen: step along the line; take one diagonal when the line is more than half a line width off the band; straight through crossings; stop when the line end is visible. A short agreement test: both of you label the same 60 saved pictures with the rule; if you agree on fewer than about 80 percent, the rule is rewritten before real recording. Quotas so the rare classes get examples: about 30 percent of episodes start with the puck or rod deliberately 0.5 to 1 cm off the line, about 25 percent start 5 to 8 cm before the line end.
- When to revisit. Add a class only if the recording notes show the same situation again and again and the camera can see its effect.

## The plan that follows from both

| Phase | When | What |
| --- | --- | --- |
| Rehearsal, stock camera | 31 Oct to 1 Nov | Marshal's checklist plus the pilot above; tool fixes (key buffer drain, mistake and lost keys, three timestamps, camera mount and sheet orientation in `episode.json`, laptop in the session name, puck-centre workspace guard, exception-safe guiding, Enter-to-confirm replay, step budget per sheet). No training data. |
| Bracket F fitted | 2 to 6 Nov | Direction test, reference picture, re-teach the six points, dry run over sheets 02 to 06 to count frames with no line in view. Sheets with more than about 10 percent blind frames do not enter the first model. |
| Seed and first model | 7 to 8 Nov | 30 to 40 keyboard episodes on sheets 01, 02 and the gentle part of 03; train Saturday evening; 5 trials Sunday. This shows whether the pipeline works before more time is spent. |
| Initial set | 14 to 22 Nov | Keyboard on the sheets that passed; printed sheets if the professor allows; retrain after every session. |
| Correction rounds | 22 Nov to 6 Dec | The model proposes each step, a person accepts or overrides; about 60 episodes per round, two rounds. This is the best-evidenced fix for drift. |
| Buffer and final evaluation | 12 to 20 Dec | Spare weekend; then 20 trials per sheet, pre-registered, same room and lamps. |

## Episode budget (recommended plan)

Steps per episode come from the traced line lengths (24, 27, 55, 77, 57, 57 cm; 1 cm steps plus about 15 percent corrections). Hours assume 2.7 s per keyboard step, 40 s per episode for the puck reset, 15 percent discards, and 36 productive robot hours until 20 December after 4 hours for fitting and tests.

| Sheet | Steps per episode | Keyboard episodes | Correction episodes |
| --- | --- | --- | --- |
| 01 straight | 29 | 20 | 15 |
| 02 arc | 32 | 35 | 20 |
| 03 two-curve | 64 | 45 | 25 |
| 04 meander | 90 | 45 | 25 |
| 05 loop | 67 | 30 | 20 |
| 06 dashed loop | 67 | 25 | 15 |
| Own printed sheets, if allowed | 40 | 80 | inside the rounds |
| Total | | 280 (about 17 h) | 120 (about 7 h) |

With about 6.5 hours of evaluation this is about 30 of the 36 hours; without printed sheets about 27. The minimum plan (255 episodes) needs about 21 hours, the stretch plan (650) does not fit. The tool's step budget of 80 must become per sheet: the meander needs 116.

## Three things to do before any real episode

1. Fix the tool as listed in the rehearsal row, above all the key buffer (a key pressed during a move must never label a picture the person never saw) and the Enter-to-confirm replay.
2. Write the stepping rule and pass the 60-picture agreement test.
3. Record nothing for training until bracket F is on and the direction test is repeated; stock-camera pictures and bracket-F pictures never go into one training set.

## Questions for the professor on 9 November (full list of 14 in the long note)

Does a confirmed replay count as "pushing the puck ourselves"? Are human-gated correction rounds with our own model allowed? May we print our own training sheets, and will the unseen sheets have a visible line end? How exactly is "finishing the track" judged?
