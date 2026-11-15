"""
Make the printable checklist of the robot test of 15.11.26 from its run list (asked for by Emranul).

Run from 02_code:  python "<this file>"
"""

from pathlib import Path

import pandas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


CHECKLIST_PARAMETERS = {
    "run_list": Path(__file__).resolve().parents[2] / "2026-11-15_robot_test_runs.csv",
    "output_path": Path(__file__).resolve().parents[2] / "2026-11-15_robot_test_checklist.pdf",
    "models": [
        ["E", "Model E (round 1 only)", "resnet18_model_4_chunk_stop_25_all_sheets", "The yardstick. Drove sheet 05's loop to the end on 11.11 and 15.11."],
        ["13", "Model 13 (hand corrections)", "resnet18_model_13_full_hand_corrections_all_sheets", "All data; 5 steps before an h press left out, 8 steps after it count 3 times."],
        ["14", "Model 14 (two pictures)", "resnet18_model_14_full_hand_corrections_two_pictures_all_sheets", "Model 13 that also sees the picture before."],
        ["15", "Model 15 (the mix)", "resnet18_model_15_full_mirror_after_h_16_all_sheets", "All data; mirrored pictures, steps after an h press 3 times, 16 rounds."],
    ],
    "watch": {
        "A1": "stays on the line through both bends", "A2": "same bends as A1", "A3": "same bends as A1",
        "A4": "turns into the loop, crossing, stop", "A5": "turns into the loop in time", "A6": "same with 1 cm steps",
        "A7": "loop, and whether it stops", "A8": "the far part of the arc", "A9": "same as A8",
        "A10": "where the stop lands",
        "R1": "does the puck come back onto the line", "R2": "same as R1", "R3": "same as R1", "R4": "other side",
    },
    "before": [
        "VPN OFF (it blocks the robot). Tell Claude first, so the results from ki-014 are unpacked.",
        "Only ONE program connected to the robot: recording page, NiryoStudio and any viewer closed.",
        "check_hardware.py says 'Calibration needed: False'. Only the automatic calibration, never manual.",
        "Stop button in reach. Robot away from the table edge.",
        "Motors off (FreeMotion) before anyone touches the arm. Never push on a held arm.",
        "Camera and its focus ring are NOT touched. Same table, taping and ceiling lights as the recordings.",
        "Printed: validation_01 to validation_04 from the test_tracks folder, A4, 100 % size (not 'fit to page').",
        "A ruler on the table for the recovery runs (puck 1.5 cm beside the line).",
    ],
    "each_run": [
        "In 02_code:  python visit_runs.py \"../06_deliverables/research_notes/Model and training/2026-11-15_robot_test_runs.csv\" LABEL   (LABEL = A1, A2, ...). Read the line, Enter.",
        "Lay the sheet and the puck as the faint picture shows; place the tip with the placing line; Enter to go.",
        "Recovery runs (R): after placing, move ONLY the puck 1.5 cm sideways by hand, hands out of the picture, then Enter.",
        "Unseen runs (U): no faint picture. Start cap on the start mark; turn the sheet so the whole line is as near the robot as it can be (nothing beyond about 43 cm from the base).",
        "End a run with Esc (never Ctrl+C, never the window's X). If it drives past the end or leaves the line for good: Esc.",
        "Fill in the row: line kept (yes / left at ... cm), stop (on the end / early ... cm / none), puck at the stop (covers the end? cm off).",
    ],
}


def checklist_story(parameters: dict) -> list:
    """Return the PDF's content: the rules, the models, and one table row per run with boxes to fill in."""

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent = styles["Normal"], fontSize = 8, leading = 10)
    normal = ParagraphStyle("normal", parent = styles["Normal"], fontSize = 9.5, leading = 12.5)
    runs = pandas.read_csv(parameters["run_list"])
    story = [Paragraph("Robot test, 15.11.26: checklist and run sheet (team M)", styles["Title"]),
             Paragraph("Goal: (1) do the new models drive better than model E on known sheets, (2) do they bring a puck back onto the line, "
                       "(3) how do they do on printed tracks no model has seen. All scores so far are from recorded pictures; this sheet is the test that counts.", normal),
             Spacer(1, 4 * mm), Paragraph("Before the first run (tick each)", styles["Heading3"])]
    story += [Paragraph(f"[&nbsp;&nbsp;]&nbsp; {text}", normal) for text in parameters["before"]]
    story += [Spacer(1, 3 * mm), Paragraph("Every run", styles["Heading3"])]
    story += [Paragraph(f"{number}.&nbsp; {text}", normal) for number, text in enumerate(parameters["each_run"], start = 1)]
    story += [Spacer(1, 3 * mm), Paragraph("Models", styles["Heading3"])]
    sha_of_key = {str(key): sha for key, sha in zip(runs["model_key"], runs["checkpoint_sha256"])}
    model_rows = [["Key", "Name", "Recipe (run name)", "File check (sha256)", "What it is"]]
    model_rows += [[key, Paragraph(name, small), Paragraph(run_name, small), sha_of_key.get(key, ""), Paragraph(what, small)]
                   for key, name, run_name, what in parameters["models"]]
    model_table = Table(model_rows, colWidths = [12 * mm, 42 * mm, 88 * mm, 30 * mm, 95 * mm], repeatRows = 1)
    model_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                     ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [model_table, Paragraph("Not on the list: model 16 (zoomed view) cannot be driven, the driver does not cut the zoomed picture; "
                                     "models 11 and 12 are superseded by 13 to 15.", small), PageBreak(),
              Paragraph("Run sheet (in this order; stop a part early if every model fails at the same place)", styles["Heading3"])]

    parts = {"A": "A: known sheets, new models against model E", "R": "R: recovery, puck set beside the line by hand",
             "U": "U: printed tracks no model has seen"}
    rows = [["No.", "Label", "Sheet and lay-out", "Model", "Step", "Watch", "Line kept?", "Stop", "Puck at stop", "Remark"]]
    part_rows = []
    last_part = None
    for number, run in enumerate(runs.itertuples(), start = 1):
        if run.label[0] != last_part:
            last_part = run.label[0]
            part_rows.append(len(rows))
            rows.append([Paragraph(f"<b>{parts[last_part]}</b>", small)] + [""] * 9)
        watch = parameters["watch"].get(run.label, "stays on the line; stops on the end")
        rows.append([str(number), run.label, Paragraph(f"{run.sheet}<br/><font size=6.5>{run.lay_out[:95]}</font>", small), str(run.model_key),
                     f"{run.step_cm:g} cm", Paragraph(watch, small), "", "", "", ""])
    table = Table(rows, colWidths = [9 * mm, 12 * mm, 78 * mm, 13 * mm, 14 * mm, 44 * mm, 26 * mm, 26 * mm, 24 * mm, 26 * mm], repeatRows = 1)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
             ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke])]
    for row in part_rows:
        style += [("SPAN", (0, row), (-1, row)), ("BACKGROUND", (0, row), (-1, row), colors.Color(0.85, 0.9, 1.0))]
    table.setStyle(TableStyle(style))
    story += [table, Spacer(1, 4 * mm),
              Paragraph("How to fill in. Line kept: 'yes', or the place where the puck's centre left the line by more than 2 cm and did not come back. "
                        "Stop: 'on the end', 'early' with the distance to the end, or 'none'. Puck at stop: does the puck cover the line end fully, and how far its centre is off (cm, left or right). "
                        "A run counts as finished only when the puck covers the line end and stayed on the line all the way (Emranul, 11.11.26).", small),
              Paragraph("Afterwards: tell Claude the results row by row; VPN on again only when the robot work is over.", small)]
    return story


def main() -> None:
    """Write the PDF."""

    parameters = dict(CHECKLIST_PARAMETERS)
    document = SimpleDocTemplate(str(parameters["output_path"]), pagesize = landscape(A4), leftMargin = 12 * mm, rightMargin = 12 * mm,
                                 topMargin = 10 * mm, bottomMargin = 10 * mm, title = "Robot test 15.11.26: checklist")
    document.build(checklist_story(parameters))
    print("written:", parameters["output_path"])


if __name__ == "__main__":
    main()
