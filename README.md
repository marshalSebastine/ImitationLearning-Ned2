# AI Project, Team M: imitation learning with a Niryo Ned2

- **Team M:** Emranul Hoque, Marshal Sebastine.
- **Supervisor:** Prof. Dr.-Ing. Thomas Nierhoff.
- **Course:** [MAI/MKI - AI Project (Moodle)](https://moodle.oth-aw.de/course/view.php?id=3755).
- **Requirements handout:** [AI Project – Team M Requirements Handout](https://claude.ai/artifact/W8ezA3EnxKZLTWdpZeZPy1), a living doc with every requirement, rule and deadline (Emranul's Claude account; the PDF copy `requirements_handout.pdf` in the root is the version to read).
- **NiryoStudio guide:** [NiryoStudio Guide](https://claude.ai/artifact/KKkAgtyNJ8SDsupgAQtR7j), every page and button of the robot app explained, and which ones we use (live doc in Emranul's account; PDF copy `niryostudio_guide.pdf`).
- **Project guide:** [AI Project Guide](https://claude.ai/artifact/VC9qSsuXC1teaRy3ghbuqR), the whole task end to end for a first-time robot learner: show, train, drive, and what the exam tests (live doc in Emranul's account; PDF copy `project_guide.pdf`).
- **Phase 2 guide:** [Phase 2 Guide: Recording](https://claude.ai/artifact/GSa3cW77MZ2Gu7UK4kayMz), the current phase only: workspace, recording script, checks, method trial and pilot, with the manual work per step (live doc in Emranul's account; PDF copy `phase_2_recording_guide.pdf`; its recorder design is superseded, see the handoff).

## Task (Project 6)

1. Push a puck along a printed black line to its goal using a Niryo Ned2 arm.
2. Record our own demonstrations: the end-effector webcam video is the input, and the robot motion is the output.
3. Train an offline policy that the final evaluation tests on a **new, unseen path**. The evaluation format has not been announced yet.

The professor expects the dataset, not the model, to be the hard part. Full course details are in `01_course/moodle_snapshot.md`.

## Deadlines

All four meetings are on BBB and mandatory: points are deducted for each absence without a medical certificate.

- [x] 19.10.26: hardware handout
- [ ] 26.10.26, 09:50–10:00: meeting 1
- [ ] 09.11.26, 09:50–10:00: meeting 2
- [ ] 23.11.26, 09:50–10:00: meeting 3
- [ ] 07.12.26, 09:50–10:00: meeting 4
- [ ] **09.12.26–17.12.26: exam registration in PRIMUSS, both of us. Binding: missed means no grade this semester.**
- [ ] 08.01.27: submission package frozen (our own buffer)
- [ ] **10.01.27, 23:59: submission**
- [ ] **11.01.27: presentation**

## Folder map

```
AI_Project/
├── README.md            start here
├── STATUS.md            the live state per workstream; every Claude session reads it first and updates it last
├── HOW_WE_WORK.md       the session system: orchestrator, workstream sessions, the open-and-close ritual
├── how_we_work.pdf      PDF copy of HOW_WE_WORK.md, to print
├── HANDOFF_2026-10-26.md  full background for a teammate taking over; read once
├── CLAUDE.md            working rules for Claude in this project
├── COLLABORATION.md     how the two of us (and our two Claude accounts) work through git: branches, session rhythm, owners
├── setup_claude.sh      links claude_memory/ and claude_skills/ into Claude Code; run once per laptop (setup_claude.ps1 on Windows)
├── claude_memory/       Claude's shared project memory, tracked in git so both Claude accounts share it (MEMORY.md is the index)
├── claude_skills/       Claude skills for this project (emran-code-style)
├── requirements_handout.pdf  PDF copy of the requirements handout: every requirement, rule and deadline with sources
├── mynotes.txt          Emranul's running questions for Claude
├── work_plan.pdf        the 8 work phases with sub-phases, tasks and done-when tests; no dates
├── niryostudio_guide.pdf  PDF copy of the NiryoStudio guide (the live doc is linked above)
├── project_guide.pdf    PDF copy of the project guide: how imitation learning gets the puck to the goal (live doc linked above)
├── phase_2_recording_guide.pdf  PDF copy of the phase 2 guide: recording the first demonstrations (live doc linked above)
├── .venv/               Python 3.11 environment for the robot code (made by `py -3.11 -m venv .venv`); never in git
├── 01_course/           course inputs; add new files, never edit them (exception: the snapshot)
│   ├── moodle_snapshot.md                 our summary of Moodle, kept current by rule 6
│   ├── <yyyy-mm-dd>_moodle_export.json    raw Moodle export, one per sync
│   ├── ai_project_kickoff_slides.pdf
│   ├── gpu_pool_ssh_guide_en.pdf
│   ├── mai_course_catalogue_2026-05-19.pdf  module handbook (source of the 7-page documentation rule)
│   └── path_sheets/     path_NN_<shape>.jpeg; the file stem is the path_id
├── 02_code/             all Python, flat; runs on the laptop and on the GPU pool
│   └── requirements.txt packages for .venv on the laptop (pip install -r)
├── 03_data/             large files, never in version control
│   ├── raw/             <yyyy-mm-dd>_session_NN/: session_notes.md plus <path_id>_episode_NNN.mp4/.csv (recording sessions, none yet)
│   │   ├── camera_check/   pictures, videos and notes from the set-up days (never renamed, even the WhatsApp video)
│   │   ├── robot_survey/   survey_robot.py results (JSON)
│   │   └── workspace/      taught positions, test pictures, push_straight/ runs, camera_tests/
│   └── processed/       v<NN>_<description>/: rebuilt from raw (created on first build)
├── 04_runs/             <yyyy-mm-dd>_<model>_<what_changed>/: training plus its evaluation/ on the robot
├── 05_meetings/         <yyyy-mm-dd>_meeting_N.md, one note per meeting
│   ├── requests_to_professor.md   running list of what we need from him, by meeting
│   └── <yyyy-mm-dd>_meeting_N_questions.pdf   that meeting's questions, printed for the call
└── 06_deliverables/     team_m_project6_* submission and presentation
    ├── team_m_project6_documentation/figures/  photos for the write-up
    ├── reports/         research reports (<title>.md and .pdf), e.g. the SOTA survey of 25.10.26
    └── research_notes/<title>/  the sourced notes behind each report, one file per angle; may hold figures/, scripts/, niryo_files/
```

## Path sheets

- `path_01` to `path_06` are the 6 sheets from Prof. Nierhoff, received as photos on 23.10.26 (not on Moodle) and numbered from easy to hard.
- Later sheets are numbered from `path_07` onwards, in the order they arrive. That includes the professor's unseen path, and our own sheets if he allows them (asked in meeting 1).

| path_id | Shape |
| --- | --- |
| `path_01_straight_line` | Straight line |
| `path_02_arc` | "("-shaped arc |
| `path_03_two_curve` | "2"-shaped curve with one sharp turn |
| `path_04_meander` | Long meander |
| `path_05_alpha_loop` | Self-crossing loop |
| `path_06_alpha_loop_dashed` | Dashed self-crossing loop (hardest) |

## Rules

1. **Names:**
   - lowercase snake_case, no spaces;
   - names start with `yyyy-mm-dd` when dated, and text writes dates as `dd.mm.yy`;
   - `README.md`, `CLAUDE.md`, `HANDOFF_<date>.md` and the root folder are exempt, and so are research-report and research-note titles under `06_deliverables` (the research tool names them with spaces and capitals) and raw files, which keep the name they arrived with;
   - never `misc`, `old`, `temp` or `final_v2`.
2. **Map first:**
   - a new kind of folder, one no pattern above covers, is added to the map before it is created;
   - folders we make by hand stay at most 3 levels below `AI_Project`;
   - folders that tools generate inside a dataset or run are exempt.
3. **Inputs are read-only:**
   - nothing in `01_course` or `03_data/raw` is edited, renamed or deleted;
   - the only exceptions are `moodle_snapshot.md` (rule 6) and each session's `session_notes.md`, which may be appended to;
   - a bad demonstration is noted in `session_notes.md` and excluded at dataset build;
   - each new raw session is copied to a second disk the same day.
4. **Episodes and splits:**
   - every episode and trial file starts with its path_id;
   - episode numbers count up per path_id across all sessions, so `path_02_arc_episode_004.mp4` exists only once;
   - splits and exclusions are set in the dataset-build config in `02_code` and written into each dataset's `manifest.json`, never by moving files.
5. **Outputs:**
   - `03_data/processed`, and the checkpoints and logs in `04_runs`, must be rebuildable from raw data, code and config;
   - each run's `config.json` records its dataset version and code version (the git commit, once `02_code` is in git);
   - the `evaluation/` recordings in a run are real robot trials, so they are kept;
   - after each meeting, delete runs that are cited neither in a meeting note nor in `06_deliverables`.
6. **Moodle sync every Monday:**
   - update `moodle_snapshot.md` (as-of date plus a changelog line) and add a new dated export;
   - file new downloads into `01_course` under descriptive names, with a row in the snapshot's file-name map.

## Resources

- **PyNiryo:**
  - [docs](https://niryorobotics.github.io/pyniryo/v1.2.5/index.html) · [API](https://niryorobotics.github.io/pyniryo/v1.2.5/api/api.html) · [compatibility](https://niryorobotics.github.io/pyniryo/v1.2.5/versions/compatibility.html)
  - install with `pip install pyniryo`; it needs robot software 5.5.0 or newer;
  - robot IP `10.10.10.10` on the robot's hotspot, `169.254.200.200` over direct Ethernet.
- **Ned2:**
  - [specifications](https://docs.niryo.com/robots/ned2/technical-specifications/) · [learning mode / FreeMotion](https://docs.niryo.com/robots/ned2/robot-operating-mode/) (move the arm by hand to demonstrate)
  - [NiryoStudio download](https://niryo.com/resources/download-center/)
- **Imitation learning:**
  - [LeRobot](https://github.com/huggingface/lerobot) (ACT and Diffusion policies, plus a dataset format)
  - [ACT paper](https://arxiv.org/abs/2304.13705)
  - [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) (its Push-T task is the closest published analogue)
  - [DAgger](https://arxiv.org/abs/1011.0686) (why small errors snowball on unseen paths)

## Open questions

- **For the professor:** everything we need from him is in `05_meetings/requests_to_professor.md`, grouped by meeting.
- **For us:** PyNiryo's PyPI classifiers stop at Python 3.11, so the robot environment should use 3.11 rather than the local 3.13.
