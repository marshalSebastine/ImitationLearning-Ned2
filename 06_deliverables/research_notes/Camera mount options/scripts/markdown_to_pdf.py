"""Render a markdown report (headings, paragraphs, links, emphasis, tables, bullet lists) to an A4 PDF."""

import html
import re
import pathlib
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

source_path, output_path = sys.argv[1], sys.argv[2]
footer_title = sys.argv[3] if len(sys.argv) > 3 else pathlib.Path(source_path).stem

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent = styles["Normal"], fontSize = 9.4, leading = 12.6, spaceAfter = 5)
cell = ParagraphStyle("cell", parent = body, fontSize = 8.0, leading = 10.0, spaceAfter = 0)
bullet = ParagraphStyle("bullet", parent = body, leftIndent = 12, bulletIndent = 2)
title_style = ParagraphStyle("title", parent = styles["Title"], fontSize = 18, leading = 22, spaceAfter = 8)
h2 = ParagraphStyle("h2", parent = styles["Heading2"], fontSize = 13, leading = 16, spaceBefore = 10, spaceAfter = 4)
h3 = ParagraphStyle("h3", parent = styles["Heading3"], fontSize = 10.8, leading = 13.5, spaceBefore = 7, spaceAfter = 3)
note = ParagraphStyle("note", parent = body, fontSize = 8.6, leading = 11.2, textColor = colors.HexColor("#444444"))


def inline(text: str) -> str:
    """Convert inline markdown (links, bold, italics, code) to reportlab paragraph markup."""

    text = html.escape(text, quote = False)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" color="#1a4f9c">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", text)
    return text


def table_rows(lines: list) -> Table:
    """Build a ruled table from markdown pipe rows, skipping the separator row."""

    rows = []
    for line in lines:
        if re.match(r"^\|?\s*-{3,}", line.strip("| ")):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        rows.append(cells)
    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]
    width = A4[0] - 3.0 * cm
    if column_count == 5:
        widths = [width * share for share in (0.24, 0.17, 0.22, 0.17, 0.20)]
    elif column_count == 3:
        widths = [width * share for share in (0.16, 0.56, 0.28)]
    else:
        widths = [width / column_count] * column_count
    data = [[Paragraph(f"<b>{inline(text)}</b>" if index == 0 else inline(text), cell) for text in row]
            for index, row in enumerate(rows)]
    table = Table(data, colWidths = widths, repeatRows = 1)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return table


lines = open(source_path, encoding = "utf-8").read().splitlines()
story = []
paragraph_lines = []
table_lines = []


def flush_paragraph() -> None:
    global paragraph_lines
    if paragraph_lines:
        text = " ".join(paragraph_lines).strip()
        if text.startswith("*") and text.endswith("*") and text.count("*") == 2:
            story.append(Paragraph(inline(text), note))
        else:
            story.append(Paragraph(inline(text), body))
        paragraph_lines = []


def flush_table() -> None:
    global table_lines
    if table_lines:
        story.append(table_rows(table_lines))
        story.append(Spacer(1, 6))
        table_lines = []


for line in lines:
    stripped = line.strip()
    if stripped.startswith("|"):
        flush_paragraph()
        table_lines.append(stripped)
        continue
    flush_table()
    if not stripped:
        flush_paragraph()
        continue
    if stripped.startswith("# "):
        flush_paragraph()
        story.append(Paragraph(inline(stripped[2:]), title_style))
    elif stripped.startswith("## "):
        flush_paragraph()
        story.append(Paragraph(inline(stripped[3:]), h2))
    elif stripped.startswith("### "):
        flush_paragraph()
        story.append(Paragraph(inline(stripped[4:]), h3))
    elif re.match(r"^[-*] ", stripped):
        flush_paragraph()
        story.append(Paragraph(inline(stripped[2:]), bullet, bulletText = "\u2022"))
    elif re.match(r"^\d+\. ", stripped):
        flush_paragraph()
        number, rest = stripped.split(". ", 1)
        story.append(Paragraph(inline(rest), bullet, bulletText = f"{number}."))
    else:
        paragraph_lines.append(stripped)
flush_paragraph()
flush_table()


def page_number(canvas, document) -> None:
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"{footer_title}, page {document.page}")


document = SimpleDocTemplate(output_path, pagesize = A4, leftMargin = 1.5 * cm, rightMargin = 1.5 * cm,
                             topMargin = 1.6 * cm, bottomMargin = 1.6 * cm,
                             title = footer_title, author = "Team M")
document.build(story, onFirstPage = page_number, onLaterPages = page_number)
print("written", output_path)
