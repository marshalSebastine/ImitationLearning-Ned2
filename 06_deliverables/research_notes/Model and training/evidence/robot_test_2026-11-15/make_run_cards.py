"""
One card per run of the robot test of 15.11.26, with the picture to lay the sheet by (asked for by Emranul at the robot:
"we don't remember the sheet and picture"). Known sheets show the faint picture the run tool also lays over the live
view; printed unseen tracks show the track's drawing with the start marked.

Run from 02_code:  python "<this file>"
"""

import tempfile
from pathlib import Path

import fitz
import pandas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


CARD_PARAMETERS = {
    "run_list": Path(__file__).resolve().parents[2] / "2026-11-15_robot_test_runs.csv",
    "track_folder": Path(__file__).resolve().parents[2] / "tracks_2026-11-12" / "development",
    "output_path": Path(__file__).resolve().parents[2] / "2026-11-15_robot_test_run_cards.pdf",
    "model_names": {"E": "Model E (round 1)", "13": "Model 13 (hand corrections)", "14": "Model 14 (two pictures)", "15": "Model 15 (the mix)"},
}


def track_picture(sheet_text: str, parameters: dict, folder: Path) -> Path:
    """Render the printed track named in the run list's sheet text (for example 'PRINTED D03 solid: ...') as a PNG."""

    name, style = sheet_text.split()[1], sheet_text.split()[2].rstrip(":")
    page = fitz.open(str(parameters["track_folder"] / f"{name}_{style}.pdf"))[0]
    path = folder / f"{name}_{style}.png"
    page.get_pixmap(dpi = 40).save(str(path))
    return path


def main(parameters: dict = CARD_PARAMETERS) -> None:
    """Write two cards per landscape A4 page."""

    runs = pandas.read_csv(parameters["run_list"])
    width, height = landscape(A4)
    sheet = canvas.Canvas(str(parameters["output_path"]), pagesize = landscape(A4))
    sheet.setTitle("Robot test 15.11.26: run cards")
    with tempfile.TemporaryDirectory() as folder:
        for index, run in enumerate(runs.itertuples()):
            left = 10 * mm if index % 2 == 0 else width / 2 + 5 * mm
            card_width = width / 2 - 15 * mm
            top = height - 12 * mm
            sheet.setFont("Helvetica-Bold", 22)
            sheet.drawString(left, top - 8 * mm, f"{index + 1}.  {run.label}")
            sheet.setFont("Helvetica-Bold", 12)
            sheet.drawString(left + 42 * mm, top - 8 * mm, parameters["model_names"].get(str(run.model_key), str(run.model_key)) + f",  step {run.step_cm:g} cm")
            sheet.setFont("Helvetica", 10)
            text = sheet.beginText(left, top - 17 * mm)
            text.setLeading(12)
            for line in (f"Sheet: {run.sheet}", f"Lay-out: {run.lay_out}"):
                words, current = line.split(), ""
                for word in words:
                    if sheet.stringWidth(current + " " + word, "Helvetica", 10) > card_width:
                        text.textLine(current)
                        current = word
                    else:
                        current = (current + " " + word).strip()
                text.textLine(current)
            sheet.drawText(text)
            if str(run.label).startswith("U"):
                picture, caption = track_picture(str(run.sheet), parameters, Path(folder)), "Printed track, start = bottom end. Tape once; both models on the same lay-out."
            elif isinstance(run.ghost, str) and Path(run.ghost).is_file():
                picture, caption = Path(run.ghost), "Camera view at the start: lay sheet and puck like this."
            else:
                picture, caption = None, "No picture found for this run."
            if picture is not None:
                picture_height = 120 * mm
                image = fitz.open(str(picture))
                ratio = image[0].rect.width / image[0].rect.height
                picture_width = min(card_width, picture_height * ratio)
                sheet.drawImage(str(picture), left, top - 50 * mm - picture_height, width = picture_width, height = picture_width / ratio)
            sheet.setFont("Helvetica-Oblique", 9)
            sheet.drawString(left, 12 * mm, caption[:110])
            if str(run.label).startswith("R"):
                sheet.setFont("Helvetica-Bold", 11)
                sheet.drawString(left, 18 * mm, "Then move ONLY the puck 1.5 cm sideways by hand (see Sheet), hands out of the picture.")
            if index % 2 == 1:
                sheet.showPage()
        sheet.save()
    print("written:", parameters["output_path"], len(runs), "cards")


if __name__ == "__main__":
    main()
