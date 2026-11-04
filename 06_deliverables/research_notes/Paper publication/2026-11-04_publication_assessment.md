# Could this project become a paper? Assessment of 4 November 2026

Written 4.11.26 for Emranul Hoque, Marshal Sebastine and Prof. Nierhoff, because Emranul asked on 3.11.26 whether the
project could lead to a publication. Everything here is an assessment, not a decision: the students and the professor
decide. Sources read: `STATUS.md`, the three data-collection notes of 30 and 31.10 (analysis, summary, decisions), the
bracket F README, and the web pages linked in section 2 (all checked on 4.11.26). Nothing was run on the robot and
nothing under `03_data` was touched. Dates marked **verified** were read on the organiser's own page on 4.11.26;
dates marked **pattern** are last year's dates and may shift; **unverified** means no source was found.
## 0. The short answer

- Yes, there is probably one publishable result in this project, but not before 10 January 2027. The course
  deliverables (dataset, model, 7-page documentation, presentation) must come first, and the evidence a paper needs
  is largely the same evidence the course needs, so nothing is lost by waiting.
- The most publishable piece is not the robot or the bracket but the **controlled comparison of two ways of
  collecting demonstrations** (keyboard stepping against hand-guided replay) **for a camera-only, one-picture,
  discrete-action policy**, with the occlusion-when-backing problem and the step-over action as a second finding. The
  analysis note of 30.10 found no paper that makes this comparison for a discrete classification policy (section 4
  and question 12 of that note).
- The realistic target is a **4-page workshop paper** at ICRA 2027 (Seoul, workshops on 24 or 28 May 2027; the
  workshop list appears on 31 January 2027, paper deadlines usually March to April) or at RSS 2027 (Athens, 6 to
  11 July 2027; workshop paper deadlines usually early June), plus a copy on arXiv. A full conference paper (IROS
  2027, deadline 1 March 2027) is only reachable if the course results are strong and the professor wants to push for it.
- The fastest archival paper is an education paper about the whole course project (idea K in section 1). It could go
  to Robotics in Education 2027 in Prague, whose deadline is probably late January 2027, right after the course
  submission. It reuses the course documentation and needs no extra experiment.
- Seven more research ideas for this project (E to K: camera placement, memory against occlusion, classify or regress,
  corrections against demonstrations, variety against repetition, two-operator agreement, education) are in section 1,
  grouped into three possible papers.
- What to do now: nothing extra on the robot. Only collect "paper-ready": pre-register the comparison, keep every
  measured number, and ask the professor the questions in section 4 at meeting 2 on 9 November.

## 1. Candidate contributions, judged against what exists

A paper needs one claim that is new, true and useful to someone else. The four candidates below are ordered by how
likely they are to pass a reviewer. "Work" is extra work beyond the course plan in `STATUS.md`; the course plan itself
(about 410 episodes, 20 trials per sheet, pre-registered evaluation) is assumed done.

### A. Keyboard stepping against hand-guided replay for a discrete, camera-only policy (strongest)

- **The claim.** For a policy that sees one wrist-camera picture and picks one of a few 1 cm steps, demonstrations
  recorded through the policy's own loop (keyboard stepping) give better or cheaper training data than hand-guiding
  followed by a replay, or they do not. Either outcome is reportable if measured properly.
- **Why it could be new.** The closest published comparison is Li, Cui and Sadigh (2025, "How to Train Your Robots?"),
  which compares kinesthetic teaching with replay, VR and SpaceMouse on a Franka with wrist and external cameras and
  found kinesthetic worst on the pushing task. Vanc et al. (2026) compare kinesthetic, joystick and gesture teaching
  with eight novices on eye-in-hand tasks. None of the 100 sources checked on 30.10 compares the two methods for a
  one-picture classifier at one decision per second, where the usual "keyboard is slow" argument does not apply
  because the robot allows only one step per second anyway. That gap is the paper, provided the search in February
  (section 3) does not find it closed.
- **Evidence needed before it is publishable.** (1) The pilot already planned: 10 keyboard and 10 hand-guided
  episodes per operator on sheets 01 and 02, with time per episode, override rate, discard rate and the Ctrl-C safety
  test (course plan, about 1 robot hour). (2) One model trained from each source at equal episode counts, evaluated
  with 20 pre-registered trials on two seen sheets and one unseen sheet, success counts with confidence intervals.
  (3) The two-operator agreement test (kappa) for the keyboard data, so a reviewer can see the labels are consistent.
- **Extra work.** About 3 to 4 robot hours (the hand-guided model's evaluation; the pilot and the keyboard model are
  in the course plan anyway), 2 to 3 days of analysis and writing. If the pilot drops hand-guiding on 31.10's
  thresholds, the paper shrinks to "why replay fails for pushing" with the pilot numbers only: still a short finding,
  but weaker, and it should then be folded into B.
- **Honest risk.** With 10 to 20 episodes per arm the result may be a null result that reviewers call underpowered.
  The counter is to report it as a measured pilot with intervals, not as a law.

### B. The task as a low-cost, reproducible benchmark for wrist-only imitation learning (useful, big)

- **The claim.** A classroom robot (Niryo Ned2), six printed sheets, a 24 mm puck with a magnet pusher, a printed
  camera bracket, a recording tool, a dataset and baseline numbers per sheet form a benchmark others can rebuild for a
  few hundred euros beyond the robot, and the benchmark is hard in an interesting way (the pusher hides the target
  when backing, hairpins leave the line out of view).
- **Why it could be useful, and why it is not clearly new.** Push-T (Chi et al., 2023), Duckietown lane following
  (Lorincz et al., 2022) and UniROS (Kapukotuwa et al., 2025, which has a Ned2 "push a cube to a goal" task for
  reinforcement learning) all exist. None combines line following, pushing and a wrist camera as the only input, so
  the combination is new; but benchmark papers are judged on whether anyone else uses them, and a benchmark from one
  lab on one robot is a workshop contribution, not a main-track paper, until a second team reproduces it.
- **Evidence needed.** Released data and code with a licence, per-sheet success numbers with intervals for at least
  one baseline (the course model) and preferably two (the optional ACT comparison), the unseen-sheet result, and a
  cost and time table (bracket print, sheets, hours of demonstrations). A second lab or a second robot reproducing one
  sheet would turn it from "we did this" into "you can do this".
- **Extra work.** The data and code release alone is 3 to 5 days (cleaning, manifest, licence, DOI on Zenodo) and
  needs the professor's permission (the sheets are his, the course owns the deliverables). Any second baseline is
  GPU-pool work of about a week. A second-lab reproduction is outside the team's control.

### C. Occlusion when backing: the pusher hides the line, and what a memoryless policy can do about it (finding)

- **The claim.** With a wrist camera behind the pusher, every pull towards the base hides the puck and the line ahead
  (Marshal's bracket F picture of 4.11); a model without memory cannot steer through that stretch from the picture
  alone. Candidate remedies are a step-over action (lift, cross 4.2 cm, lower, one decision), a hairpin strategy
  (reorient so that pulls become pushes), or a side-offset camera. Measuring how much each helps is a small, concrete
  finding.
- **Why it is only partly new.** Wrist-camera occlusion is a known limit: Chuang et al. (2024, "Active Vision Might
  Be All You Need") argue that eye-in-hand views depend on what the arm is doing, and Wang et al. (2025, Observer-Actor)
  add a second observer arm to see around occlusions. Our version is narrower (a pusher that hides its own target)
  and the remedy is a discrete recovery action, which is modest but clean.
- **Evidence needed.** The blind-frame fraction per sheet with bracket F (the course plan measures it in the dry run),
  success per sheet with and without the step-over classes at equal data, and the number of step-over decisions a
  policy actually learns to take. The step-over itself is untested on the robot (STATUS, 4.11).
- **Extra work.** 4 to 6 robot hours for the ablation, 1 day of analysis. A side-offset camera would need a second
  print and breaks the "one print, copy a photographed geometry" rule, so it is a question for the professor, not a
  plan. Best used as a section inside A or B, not as a paper of its own.

### D. The data-collection protocol itself (method section, not a paper)

- **What it is.** Actions named in the picture's frame, a written stepping rule drawn as a band on the screen, a
  60-picture agreement test between the two operators, tool keys for "previous step was a mistake" and "line lost"
  instead of extra classes, quotas of deliberately offset and near-end starts, a per-sheet step budget from traced line
  lengths, and a metadata record per episode (mount, mapping version, stepping-rule version, outcome).
- **Judgement.** Each piece has a published ancestor (DART-style offsets, robomimic's "one consistent operator",
  HG-DAgger's gating); the combination is sensible engineering, not a research result. It is the method section that
  makes A credible, and the agreement test gives the one number reviewers will ask for. The episode-budget rule is
  arithmetic from measured timings and belongs in the appendix.
- **Extra work.** None beyond the course plan; 1 day to write up.

### Not a contribution: the printed bracket

The CadQuery bracket (parametric script, STEP and 3MF, measured numbers, Niryo's CAD cross-check) is a reproducibility
asset, not a reviewable result. It belongs in a GitHub release with a DOI, one figure and two sentences in the paper.

### More ideas for this project (added 4.11.26 at Emranul's request)

Emranul's own idea is A: compare keyboard teleoperation with hand-guiding for this task. Three cheap ways to make A
stronger:

- **Make it three-way.** Compare keyboard, hand-guided replay, and the raw pictures taken while the hand guides. The 30.10
  analysis (section 2.3) recommended saving the guide-phase pictures, since they come for free. The tool does not save
  them yet (no `guided_frames` in `02_code` on 4.11), so this needs a small code change on Marshal's side.
- **Measure replay divergence.** After each hand-guided replay, measure how far the puck ends from where the hand left
  it, using a photo or the last camera picture. This is the effect that would make replay labels wrong, and no paper in
  the analysis measured it for pushing.
- **Compare at equal robot time, not equal episode counts.** "Success per robot hour" is the number another lab can use.

New ideas, each with its question, why it fits this project and its extra cost. Robot time is calculated from the
analysis's figure of 6.5 h for about 220 trials. So 40 extra trials (20 on each of two sheets) cost about 1.2 h.

**E. Where to put a wrist camera for pushing: a visibility model.**
- Question: from the track geometry and the camera pose alone, can we predict which stretches the policy sees blind,
  and do the failures happen exactly there?
- Why here: we have two camera poses (stock holder and bracket F), traced line shapes for six sheets, a fitted camera
  pose (`cam_calibration.json`) and a measured 9 x 7 cm view. A short script can compute the blind fraction per sheet
  and per pose. The bracket F dry run already in the plan counts the real blind frames to check it against.
- Cost: 2 to 3 days of desk work, almost no robot time. Stock-holder pictures from the rehearsal are fine for counting
  blind frames, though never for training.
- Strength: a design rule others can apply. Fits with C (occlusion when backing) in one paper.

**F. One picture or a short memory when the pusher hides the target.**
- Question: does a policy that sees the last 2 or 3 pictures, or the last picture plus its previous action, handle
  pulls, hairpins and crossings better than a one-picture policy?
- Why here: the 29.10 discussion and the step-over design both rest on "a model without memory cannot tell how far
  round it has got". This tests that claim on our own data.
- Cost: training only (same dataset, two or three model variants) plus about 1.2 h of robot evaluation per variant.
  Stacked pictures are still camera-only. Ask Prof. Nierhoff whether the previous action counts as an extra input.
- Strength: good. It turns C from a sketch into a measured remedy.

**G. Classify or regress the step.**
- Question: on the same data, does the 11-class head beat a head that regresses the step direction, as in Diffusion
  Meets DAgger (Zhang et al. 2024)?
- Why here: the 30.10 analysis notes that the closest pushing paper does not validate a class head. RT-1 found discrete
  outputs better than continuous ones, but not for pushing.
- Cost: training only plus about 1.2 h of evaluation. A clean section, not a paper of its own.

**H. Corrections against more demonstrations, at the same robot time.**
- Question: is one hour of human-gated corrections worth more than one hour of plain keyboard episodes? In a
  human-gated correction, the model proposes each step and a person accepts or overrides it.
- Why here: our blocking loop at one step per second removes reaction lag, the main weakness of earlier human DAgger
  studies, so the result may differ from IWR and HG-DAgger.
- Cost: the correction rounds are already in the course plan. The equal-time control arm adds about 3 to 4 robot hours.
  It needs Prof. Nierhoff's yes to question 2 of the 30.10 analysis.
- Strength: strong and easy for a reviewer to follow.

**I. Variety or repetition for an unseen track.**
- Question: for a fixed number of episodes, is it better to record many episodes on few sheets or few on many sheets?
- Why here: the course evaluates on an unseen path. Lin et al. (2025) found that variety beats repetition, but for
  grasping, not line following.
- Cost: needs permission to print our own sheets, and it needs the most robot time of all the ideas (several hours per
  arm). Realistic only after the course.
- Strength: the most "research" question, and the most expensive.

**J. Two novice operators: does a written rule make them label like one?**
- Question: how far do Emranul and Marshal agree (kappa) before and after the stepping rule and the on-screen band? How
  does a policy trained on Emranul's data only, Marshal's only, or both compare?
- Cost: the agreement test is already in the plan; the extra models need training plus about 1.2 h of evaluation each.
- Strength: modest alone, good next to A in a "data collection" paper.

**K. A full imitation-learning pipeline on a classroom robot in one semester (education paper).**
- What: the course project written up for teachers and students. It covers cost, hours, what went wrong (camera mount,
  occlusion, line ends), the tool, the protocol and the results. The Ned2 is sold as an education robot.
- Venue: Robotics in Education (RiE) 2027, Prague, 26 to 28 May 2027, short papers up to 6 pages, Springer proceedings
  (archival) ([call](https://easychair.org/cfp/rie2027)). The 2027 deadline is not posted yet. RiE 2026 closed on
  31 January 2026, extended from 15 January (pattern only). If the 2027 deadline is also late January, the 7-page course
  documentation due on 10 January can be cut down to it in about two weeks.
- Cost: the lowest of all, with no extra experiment and about one week of rewriting. It is the only idea that could
  give an archival paper right after the course.
- Strength: lower research novelty, but the highest chance of acceptance, and Prague is close to Amberg.

**How the ideas combine into papers:**

1. **Data-collection paper:** A in its three-way form, plus J, with D as the method section. Workshop, spring 2027.
2. **Occlusion paper:** C plus E plus F. Workshop, or the IROS or CASE main track if the numbers are strong.
3. **Education paper:** K, with whatever numbers the course produced. RiE 2027, deadline probably late January 2027.

G and H fit as extra sections into paper 1 or 2.

## 2. Venues, checked on 4.11.26

A **workshop paper** is a short (usually 4 pages plus references), lightly reviewed, usually non-archival paper
presented as a poster at a workshop on the day before or after a big conference. It is the normal first paper for a
master's project, it allows a later full paper on the same work, and it is what section 0 recommends. A **main-track
paper** is 6 to 8 pages, reviewed by three people, and archival.

| Venue | What | Deadline | Pages | Students and workshops welcome? | Fits the 10 Jan deadline? | Status |
|---|---|---|---|---|---|---|
| ICRA 2027, Seoul, 24 to 28 May 2027 ([site](https://2027.ieee-icra.org/contribute/)) | Main track | 15 Sep 2026 (passed) | not on the page (usually 6 + references) | yes | no, passed | verified |
| ICRA 2027 workshops ([site](https://2027.ieee-icra.org/contribute/)) | 4-page workshop papers | workshop list on 31 Jan 2027; each workshop sets its own deadline, in 2026 typically March to April (the "Beyond Teleoperation" workshop at ICRA 2026: 4 pages, 24 Apr 2026, non-archival, [page](https://icra-beyond-teleop.github.io/)) | 4 + references (2026 pattern) | yes, this is their purpose | yes: results frozen 8 Jan, writing in Feb and Mar | dates verified; 2027 workshop deadlines pattern |
| IROS 2027, Florence, 26 Sep to 1 Oct 2027 (place and dates from [aggregators](https://mldeadlines.com/conference/iros/), not the official site) | Main track | 1 Mar 2027 ([IEEE RAS dates](https://www.ieee-ras.org/events/category/important-dates/list/)) | usually 6 + references (unverified for 2027) | yes | barely: seven weeks after the presentation, no spare robot time | deadline verified; place, dates and pages unverified |
| IROS 2027 workshops and late-breaking posters | 4-page papers; 1-page posters | workshop calls usually June to July; late-breaking usually August (2022 [pattern](https://iros2022.org/contributing/call-for-late-breaking-results-posters/)) | 4; 1 | yes | yes | pattern, unverified for 2027 |
| RSS 2027, Athens, 6 to 11 Jul 2027 ([site](https://roboticsconference.org/)) | Main track, two stages | extended abstract 4 Dec 2026; full paper 16 Apr 2027 only if invited on 26 Feb 2027 | not on the page | yes, but RSS is the most selective robotics venue | no: the abstract is due before any model exists | verified |
| RSS 2027 workshops | 4-page papers | proposals in Feb (2026: 12 Feb); papers in 2026 were due 5 to 12 June, 4 pages, non-archival ([examples](https://rss2026-beyond-rigidity.github.io/)) | 4 | yes | yes, the most relaxed timing | pattern, unverified for 2027 |
| CoRL 2026, Austin, 9 to 12 Nov 2026 ([site](https://2026.corl.org/contributions)) | Main track | 28 May 2026 (passed) | not checked | yes | no | verified |
| CoRL 2027 | Main track; workshops | not announced; pattern: paper late May 2027, workshop papers September 2027 | usually 8 + references | yes; CoRL workshops are a good fit for data-collection studies | main track only with extra robot work in spring; workshops yes | unverified |
| German Robotics Conference 2027, Munich, 17 to 19 Mar 2027 ([RIG](https://robotics-institute-germany.de/event/grc2027/), [TUM](https://www.mirmi.tum.de/en/mirmi/news/article/3-german-robotics-conference-2027/); the RIG event page still lists the Cologne venue of 2026 lower down) | Papers, posters, videos | call not yet published; 2026 pattern: submission 15 Jan, notification 23 Jan, camera-ready 7 Feb | not stated | yes (2026 best-paper call addressed PhD students; master's students not excluded) | collides with the course deadline: a submission on 15 Jan would have to be written in December | dates verified; call pattern |
| KI 2027, German Conference on AI ([KI 2026 call](https://ki2026.gi.de/calls/call-for-papers)) | Main track, Springer LNAI; some years a student day | not announced; 2026 pattern: abstract late March, paper early April, conference August | 12 | yes | yes, but AI-general audience, robotics is a side topic | unverified for 2027 |
| IEEE CASE 2027, Linz, 23 to 27 Aug 2027 (place and dates from [beri.net](https://www.beri.net/events/ieee-case-2027)) | Main track | 1 Mar 2027 ([IEEE RAS dates](https://www.ieee-ras.org/events/category/important-dates/list/)) | not checked | yes | barely, same as IROS; Linz is close to Amberg | deadline verified; place and dates unverified |
| Applied Research Conference 2027 (Bavarian universities of applied sciences) ([2026 call](https://mediapool.hm.edu/media/dachmarke/dm_lokal/konferenzen/2026_arc_fk03/CfP_BF.pdf)) | Student paper sessions and posters; proceedings | 2026: paper 3 Jun, conference 3 Jul; 2027 host and dates not announced | not stated | **only students of the MAPR (Applied Research) master**; the team is in MAI, so probably not eligible; the Moodle course is hosted by OTH Amberg-Weiden, so the professor can ask | yes | 2026 verified; 2027 unverified |
| Robotics in Education (RiE) 2027, Prague, 26 to 28 May 2027 ([call](https://easychair.org/cfp/rie2027)) | Regular (12 pages) and short papers or posters (6 pages), Springer LNNS, archival | not posted; 2026 pattern: 15 Jan, extended to 31 Jan ([RiE 2026](https://easychair.org/cfp/rie2026)) | 6 (short) | yes; "exemplary robotics projects in classes" is a listed topic | yes, if late January again: right after the course submission | place, dates, pages verified; deadline pattern |
| IEEE EDUCON 2027, Timisoara, 5 to 9 Apr 2027 ([call](https://educon-conference.org/2027/authors/call-for-papers)) | Engineering-education papers | abstract 30 Oct 2026 (passed), paper 7 Dec 2026 | 6 | yes | no, abstract deadline passed | verified |
| OTH Amberg-Weiden research report (Forschungsbericht) | One-page project summary, not peer-reviewed | annual, date unverified | 1 | professors submit; students as co-authors unverified | yes | unverified |
| arXiv, cs.RO ([endorsement rules](https://info.arxiv.org/help/endorsement.html)) | Preprint, no review | none | none | yes, but since 21 Jan 2026 a university e-mail no longer endorses a new author: an established cs.RO author (the professor, if he has arXiv papers) must endorse the first submission | yes | verified |

Not fitting: ECMR 2027 (mobile robots, biennial, dates not announced), Humanoids and RO-MAN 2027 (wrong topic),
ICAR 2027 (deadline 21 Mar 2027 per IEEE RAS, location unknown, a possible second choice to CASE).

Costs to keep in mind: a workshop paper needs one author at the workshop (some allow remote posters, each workshop
decides). Seoul means a long-haul flight, roughly 1,000 to 1,500 euros all in (a guess, not checked); Athens and
Munich are cheaper. Registration for a
workshop day at ICRA or RSS is usually 100 to 300 euros for students (unverified for 2027). Ask the professor whether
the faculty funds student conference travel.

## 3. A realistic timeline

| When | What | Who |
|---|---|---|
| 9 Nov 2026 (meeting 2) | Ask the professor the questions in section 4; learn whether he wants a paper and would co-author | both |
| Nov to 20 Dec | Course plan only. "Paper-ready" habits: pre-register the keyboard-against-replay comparison and the 20-trial evaluation in a dated note before running them; keep the agreement-test numbers; store every time, override and discard count; take the figure photos (set-up, camera view, occlusion when backing) | both |
| 9 to 17 Dec | Exam registration (binding) | both |
| 20 Dec to 8 Jan | Documentation and presentation; freeze on 8 Jan; submission 10 Jan; presentation 11 Jan | both |
| 12 to 31 Jan 2027 | If the professor agreed to the education paper (idea K): cut the course documentation down to a 6-page RiE 2027 short paper, deadline probably late January (check rie.science in December). Then one meeting with the professor: go or no-go for a workshop paper, which contribution (A with C as a section is the recommendation), authorship order. Check the ICRA 2027 workshop list on 31 Jan and the GRC 2027 call (probably already closed) | both, professor |
| February | Half a day of dedicated literature search on line-following imitation learning and demonstration-interface comparisons (the 100-source check of 30.10 was about collection methods, not about this exact task); at most two extra robot sessions if one experiment is missing (likely the hand-guided model's 20-trial evaluation); draft v1, 4 pages, by 28 Feb; professor's comments | Emranul writes sections 1, 3, 6; Marshal sections 4, 5 (provisional, mirrors who built what) |
| March | Pick the workshop (ICRA: deadline likely March or April; RSS: June); draft v2; data and code release prepared if allowed | both |
| April to June | Submit; on acceptance, poster and arXiv copy; travel | one of the two presents |

Robot access after 11 January must be asked for; the course does not guarantee it. If no robot time exists after the
course, the paper can only contain what the course plan measured, which is why the pre-registration in November matters.

## 4. Proposed paper and questions for the professor

### Title candidates

1. "Keyboard or Hand? Comparing Demonstration Methods for a Camera-Only Line-Pushing Policy on a Classroom Robot"
2. "One Picture, One Step: Discrete Imitation Learning for Line-Following Pushing with a Wrist Camera"
3. "Pushing a Puck Along a Line with One Wrist Camera: A Low-Cost Imitation-Learning Benchmark" (only if B is chosen
   and data release is allowed)

### Sections (4-page workshop format)

1. Introduction: the task, the course constraint (wrist camera as the only input, the team's own demonstrations), why
   it is interesting (the pusher hides its target when backing; one decision per second; two novice operators), the
   one claim.
2. Related work: eye-in-hand imitation learning and its occlusion limit; pushing from pixels; demonstration interfaces
   (kinesthetic, replay, keyboard, teleoperation); interactive corrections. About 12 references, all from the
   verified list of the 30.10 analysis plus the three found on 4.11 (UniROS, Observer-Actor, Active Vision).
3. Set-up: Ned2, bracket F (one figure), six sheets, the 1 cm step set in the picture frame, the recording loop
   (move, settle, fresh picture, key, log), the stepping rule and the agreement test.
4. The collection study: keyboard against hand-guided replay; time per episode, override and discard rates, label
   agreement; episodes per hour.
5. Policy and results: the classifier, data counts per class, success per sheet with 20 trials and intervals for each
   data source, the unseen sheet, the occlusion stretch and the step-over ablation.
6. Limitations and conclusion: one robot, one room, two operators, small counts; what a second lab would need.

### Figures

- F1: the robot with bracket F and the magnet pusher on a sheet, the camera's 9 x 7 cm window drawn in.
- F2: one camera picture with the stepping band and the eleven step directions overlaid.
- F3: the six sheets with per-sheet success bars (keyboard model against replay model), intervals drawn.
- F4: the occlusion picture when backing (Marshal's 4.11 picture) next to the step-over manoeuvre sketch.
- F5: the collection comparison: minutes per finished episode, override and discard rates, per operator.
- Table 1: dataset statistics (episodes, steps, class counts, mistakes and lost flags, per sheet and source).

### Questions for Prof. Nierhoff (meeting 2 on 9 November, or after the presentation)

1. Would he support a short paper from this project, and would he co-author it? Who would be first author (the
   course grades each student's identifiable share; a paper should not disturb that)?
2. Does the course or the faculty allow publishing the dataset (pictures of his sheets), the code and the bracket
   files under an open licence, and does OTH have a rule on student publications or a publication fund?
3. May the end-effector-camera constraint be relaxed for a comparison experiment after the course (for example an
   overhead camera as an upper bound, or a side-offset wrist camera), clearly labelled as outside the course task?
4. Is robot access possible in February and March 2027 for two or three sessions, with the same room and lamps?
5. Would he endorse the students on arXiv for cs.RO (needed since 21 Jan 2026), or name someone who can?
6. Does the faculty fund student travel to a workshop (Seoul in May, Athens in July, Munich in March)?
7. Is the Applied Research Conference open to MAI students as guests, since OTH Amberg-Weiden hosts its Moodle course?
8. His own thesis used hand-guiding: would he see value in the keyboard-against-replay comparison, and does he know a
   paper that already did it for a discrete, camera-only policy (which would change contribution A)?
9. May the unseen evaluation sheet from the course be described and shown in a paper?
10. Would he support an education paper about the course project (idea K, RiE 2027), and may the course documentation
    be reused for it?
11. For idea F: does the previous action count as an input beyond the camera, or only stacked camera pictures?

## 5. Risks and verdict

Risks, in order of weight:

- **Time and grades.** The course deadline is fixed, the exam registration is binding, and the robot time until
  20 December is already fully planned (about 30 of 36 hours). Any paper work before January competes with the grade.
- **A thin result.** Ten to twenty episodes per collection method and 20 trials per sheet give wide intervals. A
  workshop accepts that if it is stated; a main track would not.
- **The gap may not be a gap.** The 30.10 check covered collection methods, not every line-following imitation paper.
  Half a day of search in February must come before any writing.
- **Permissions.** The sheets belong to the professor, the deliverables to the course; data release needs a yes.
- **Workshop papers are non-archival.** Good for a first paper and a CV, but not a "publication" in the sense of some
  scholarship or PhD rules; a later full paper remains possible.
- **Travel and money.** Seoul is expensive; Athens, Munich or a remote poster are the cheap options.
- **The professor's own view is unknown.** Everything above changes if he says no, or if he wants a main-track paper.

Verdict: **better planned for after the course, with the groundwork laid now.** Before 10 January 2027 the only paper
work worth doing is free: pre-register the comparison, keep the numbers, take the photos, ask the professor. From
February 2027 a 4-page workshop paper on contribution A (with C as a section and D as its method) at an ICRA 2027 or
RSS 2027 workshop, plus an arXiv copy, is realistic if the professor supports it and two or three robot sessions exist.
Contribution B (the benchmark) only if data release is allowed and someone outside the team wants to use it. The
education paper (K) at RiE 2027 is the low-risk extra: no new experiments, an archival paper, about one week of
writing in late January, if the deadline falls there and the professor agrees.

## Sources found on 4.11.26 (not in the 30.10 analysis)

All other papers named here are in the reference list of `../Recording tool/2026-10-30_data_collection_analysis.md`.

- Kapukotuwa, Lee, Devine, Qiao. UniROS: ROS-based reinforcement learning across simulated and real-world robotics.
  Sensors 25(18), 5679, 2025. https://doi.org/10.3390/s25185679 (Ned2 push task: https://uniros.readthedocs.io/en/latest/envs/ned2/push.html)
- Chuang et al. Active Vision Might Be All You Need. arXiv 2409.17435, 2024. https://arxiv.org/abs/2409.17435
- Wang, Qian, Fan, Johns. Observer-Actor: active vision imitation learning with sparse-view Gaussian splatting.
  arXiv 2511.18140, 2025. https://arxiv.org/abs/2511.18140
- Venue pages: as linked in the table of section 2, all opened on 4.11.26. Only the abstract of each paper above was
  read; none was checked in full, so treat them as "use with care" until read.
