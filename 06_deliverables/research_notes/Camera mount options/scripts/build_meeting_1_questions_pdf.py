"""Build the one-page question sheet for meeting 1 with Prof. Nierhoff."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT_PATH = "D:/AI_Project/05_meetings/2026-10-26_meeting_1_questions.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent = styles["Normal"], fontSize = 8.4, leading = 10.3, alignment = TA_LEFT)
cell = ParagraphStyle("cell", parent = body, fontSize = 8.0, leading = 9.7)
heading = ParagraphStyle("heading", parent = styles["Heading2"], fontSize = 10.5, leading = 13, spaceBefore = 4, spaceAfter = 2)
title = ParagraphStyle("title", parent = styles["Title"], fontSize = 14, leading = 17, spaceAfter = 3)

QUESTIONS = [
    ("4",
     "For the demonstrations the robot learns from: may we drive the arm from the keyboard in small steps, "
     "or should we guide it by hand in learning mode? (The pusher is the magnet block on the rod.)",
     "Both are imitation learning but give different data: keyboard driving gives the same pictures the robot "
     "sees later; hand-guiding is quicker but shakier. Our default: keyboard."),
    ("9 + 2",
     "Our camera sits on the robot's wrist and always looks straight ahead, away from the robot: it sees about "
     "9 cm across and 7 cm ahead of the puck. It can follow a line that runs away from the robot and bends "
     "less than about 45 degrees (sheets 01 and most of 02), but not hairpins or loops (sheets 03 to 06), "
     "because there the line ahead leaves the picture. (a) On the exam day, may we choose how the sheet lies "
     "in front of the robot and where the start is, and may we tape it down? (b) May we re-mount the camera "
     "on the hand, with a printed holder from the Maker Space, so it turns with the pusher, or add a second "
     "camera? (c) Will the test path be like sheets 01 and 02, or can it loop back like 04 to 06? (d) Last "
     "semester's groups managed this task: which camera set-up and which kinds of paths did they use, did their "
     "test path loop back, and may we see one of their reports or set-ups?",
     "Decides our camera set-up and the whole recording plan. The course text says \"webcam mounted onto "
     "the end effector\", which a hand mount fits."),
    ("1",
     "May we have the original path-sheet files (SVG or PDF)? And may we make our own lines, for example "
     "generated SVG paths, print them and train on them for more variety?",
     "Exact reprints; more variety generalises better to the unseen path."),
    ("6",
     "Has SSH access to the GPU pool (DC 1.07) been activated for us?",
     "Training needs a GPU. Moodle (12.10) said access follows in a few days; checked on 25.10: nothing new "
     "since then, so we ask."),
    ("8",
     "If time allows: how will the final run be graded, only whether the puck reaches the goal, or also how "
     "closely it follows the line and how long it takes? Must the whole line be followed, or does reaching "
     "the goal on its own count?",
     "We train and test for what is graded."),
]

LATER = [
    ("7", "The MAI course catalogue (19.05.2026, entry \"AI project\", pages 15 to 16) lists the examination as "
          "\"Presentation in the team (15 minutes), project documentation (7 pages)\" with a literature review "
          "and a software part; the Moodle post of 05.10 says only Project 3 needs documentation. Which applies "
          "to us, and what exactly do we hand in by 10.01.27 (code, model, data, report, video), and where?",
     "2, 09.11.26"),
    ("10", "Must the final policy run offline or locally? May we use pretrained vision models (e.g. a ResNet "
           "backbone or LeRobot's ACT)?", "2, 09.11.26"),
    ("11", "Presentation on 11.01.27: 15-minute team presentation with a live robot demo; we assume yes, "
           "please confirm.", "3, 23.11.26"),
    ("12", "When and where is the unseen-path test run? We assume the test path is handed to us on the exam "
           "day; please confirm.", "4, 07.12.26"),
]


def build_table(
        header: list,
        rows: list,
        widths: list
) -> Table:
    """A ruled table whose cells wrap as paragraphs."""

    data = [[Paragraph(f"<b>{text}</b>", cell) for text in header]]
    data += [[Paragraph(text, cell) for text in row] for row in rows]
    table = Table(data, colWidths = widths, repeatRows = 1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


document = SimpleDocTemplate(OUTPUT_PATH, pagesize = A4, leftMargin = 1.5 * cm, rightMargin = 1.5 * cm,
                             topMargin = 1.1 * cm, bottomMargin = 1.1 * cm)
content_width = A4[0] - 3 * cm

story = [
    Paragraph("Meeting 1 with Prof. Nierhoff: questions", title),
    Paragraph("Monday 26.10.26, 09:50 to 10:00, BigBlueButton. Team M: Emranul Hoque and Marshal Sebastine. "
              "Project 6: imitation learning, Niryo Ned2 pushing a puck along a line.", body),
    Paragraph("1. Report first (one minute)", heading),
    Paragraph("Robot and camera work from Python; calibrated, moves and timings measured. The straight-line sheet "
              "lies in front of the robot with the line pointing away from it. Pusher: the Niryo magnet block on "
              "the rod (gripper removed). Camera in its original bracket; on the straight line the puck stays in "
              "every picture. Test pushes so far were scripted hardware checks, not training data; recording "
              "starts next.", body),
    Paragraph("2. Questions, in this order", heading),
    build_table(["#", "Question", "Why we ask"], QUESTIONS,
                [1.0 * cm, content_width * 0.64, content_width * 0.36 - 1.0 * cm]),
    Paragraph("3. Already settled, merged or moved", heading),
    Paragraph("Question 5 (robot software): 6.1.2-b374 and NiryoStudio 1.16.3, no update needed. Question 3 is "
              "merged into question 1, question 2 (camera set-up) into question 9. Question 7 (documentation) "
              "is moved to meeting 2.", body),
    Paragraph("4. For later meetings", heading),
    build_table(["#", "Question", "Meeting"], LATER,
                [1.0 * cm, content_width * 0.82, content_width * 0.18 - 1.0 * cm]),
    Spacer(1, 3),
    Paragraph("After the meeting: write each answer into 05_meetings/requests_to_professor.md and set its "
              "status to answered. Source for question 7: 01_course/mai_course_catalogue_2026-05-19.pdf.", body),
]

document.build(story)
print("written", OUTPUT_PATH)
