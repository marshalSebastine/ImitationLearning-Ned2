As of 2026-10-23. Source: [MAI/MKI - AI Project / KI Project (WS) (Nierhoff)](https://moodle.oth-aw.de/course/view.php?id=3755)

# Moodle snapshot

A readable summary of everything on the course page, as it was when it was last synced. The raw data behind it is in `2026-10-23_moodle_export.json`.

## Course

- **Supervisor:** Prof. Dr.-Ing. Thomas Nierhoff.
- **Format:** a larger AI project done in pairs, with no thesis write-up. It runs as:
  1. group and topic selection, then a hardware handout;
  2. biweekly progress meetings;
  3. a live presentation at the end of the semester.
- **Semester:** offered in the winter semester only, and grades are submitted only then.
- **Team M:** Emranul Hoque and Marshal Sebastine (spelling as on Moodle). Assigned **Project 6**.

## Project 6: imitation learning using a Niryo robot

Slide 11 caps this project at 5 teams, but 7 groups are actually assigned to it: G, I, M, N, S, AK and AZ.

- **Task (kickoff slide 11):** learn to push an object along a designated path (a black line on the table) to a goal position, using imitation learning. The slide's photo shows a Niryo **Ned2**; the text says only "Niryo robot".
- **Method (announcement of 05.10.26):**
  - demonstrate the push yourself by moving the robot, while recording the end-effector webcam;
  - the video stream is the model input, and the robot movement is the model output.
- **Evaluation:** a new, previously unseen path.
- **Main difficulty:** per the professor, obtaining a good dataset rather than training the model.
- **FAQ (as of 22.10.26):**
  - do not place the robot near or at the edge of the table;
  - pucks are on the table in front of the Maker Space (DC 1.04); take one per group.
  - FAQ update seen 26.10.26: final evaluation uses the tracks provided AND new, unseen ones with the same overall dimensions and line thickness; do NOT modify the end effector for easier pushing; taping the sheets to the table is allowed; the camera position may be changed by designing a modified camera mount (STEP, STL or 3MF file sent to the Maker Space with the group number; they print it and leave it on the table in front of the Maker Space).

## Dates

| Date | What |
| --- | --- |
| 12.10.26 | Group and project selection (done) |
| 19.10.26 | Hardware handout |
| 26.10.26, 09.11.26, 23.11.26, 07.12.26 | Team M meetings on BBB, 09:50–10:00. **Mandatory:** points are deducted for each absence without a medical certificate ("ärztliches Attest") |
| 10.01.27, 23:59 | **Submission deadline** |
| 11.01.27 | Presentation (live, per slide 3); the format of the Project 6 evaluation is not announced yet |

## Announcements

| Date | Summary |
| --- | --- |
| 05.10.26 | Teams of two. Selection is not first-come, first-served (a coin toss if oversubscribed). Projects done with another professor follow that professor's rules, but still need PRIMUSS registration for "AI Project". Gives the Project 6 method described above. |
| 05.10.26 | Minimum number of groups for a self-proposed project lowered from 3 to 2. |
| 08.10.26 | Single-member groups will be merged randomly and receive leftover projects; there is a grade penalty if more than one single-member group remains. |
| 12.10.26 | Group and topic matching published. Hardware handout the following Monday. SSH access to the GPU pool (DC 1.07) follows in a few days; usually only the last row of machines is switched on. |
| 22.10.26 | Meeting schedule published (attendance mandatory). Project 6 pucks are on the table in front of the Maker Space. New FAQ page, to be checked regularly. |

## Infrastructure

- **GPU pool, room DC 1.07:**
  - connect with `ssh <oth-email>@ki-0xx.oth-aw.de`: your full OTH e-mail address including `@oth-aw.de`, then `@ki-0xx.oth-aw.de`, where xx is the computer number. Example: `ssh ta.graf@oth-aw.de@ki-003.oth-aw.de`;
  - needs the OTH VPN (FortiClient), even on eduroam, and activation by the professor;
  - see `gpu_pool_ssh_guide_en.pdf`; contact Tamara Graf, DC 1.01.
- **Maker Space:** room DC 1.04. The pucks are on the table in front of it; take one per group.

## Still to watch

- The **"Project submission"** section is empty; the submission activity has not been published yet.
- The **FAQ page** is updated over time.

## File-name map

The 6 path-sheet photos are not on Moodle. They came from Prof. Nierhoff as photos on 23.10.26.

| Original | Here |
| --- | --- |
| `AI Project.pdf` | `ai_project_kickoff_slides.pdf` |
| `AnleitungSSH-en.pdf` | `gpu_pool_ssh_guide_en.pdf` |
| `WhatsApp Image 2026-10-23 at 5.36.55 PM (2).jpeg` | `path_sheets/path_01_straight_line.jpeg` |
| `WhatsApp Image 2026-10-23 at 5.36.55 PM (3).jpeg` | `path_sheets/path_02_arc.jpeg` |
| `WhatsApp Image 2026-10-23 at 5.36.55 PM (4).jpeg` | `path_sheets/path_03_two_curve.jpeg` |
| `WhatsApp Image 2026-10-23 at 5.36.55 PM (1).jpeg` | `path_sheets/path_04_meander.jpeg` |
| `WhatsApp Image 2026-10-23 at 5.36.55 PM.jpeg` | `path_sheets/path_05_alpha_loop.jpeg` |
| `WhatsApp Image 2026-10-23 at 5.36.55 PM (5).jpeg` | `path_sheets/path_06_alpha_loop_dashed.jpeg` |
| `CourseCatalogue_MAI_2026-05-19.pdf` (oth-aw.de, MAI programme page; not on Moodle) | `mai_course_catalogue_2026-05-19.pdf` |

## Changelog

- 2026-10-23: first snapshot (course page, FAQ, timetable, 5 announcements, 2 files).
- 2026-10-25: re-checked the course page, the announcements and the FAQ page: no change since 22.10.26, so no news on GPU-pool access. Added the MAI course catalogue (19.05.2026, from oth-aw.de) as the source of the "7-page documentation" rule for question 7.
- 2026-10-26: FAQ update for Project 6 copied in (evaluation on provided and unseen tracks, end effector not to be modified, taping allowed, printable camera mount).
