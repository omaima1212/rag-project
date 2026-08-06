"""Generate a deliberately MESSY test PDF: two-column layout + a data table.

Why this exists: the README and SPECIFICATION both list "multi-column layouts
and tables extract poorly" as a known limitation, but until now the claim had
never been tested - every document that had actually gone through the
pipeline was single-column prose. This script builds a synthetic document
that exercises both weaknesses on purpose, with facts specific enough that a
correct answer proves nothing was memorised from training data.

    python tools\\make_messy_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (BaseDocTemplate, Frame, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = Path(__file__).parent.parent / "data" / "meridian_report.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5,
                      leading=13, spaceAfter=8)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12,
                    spaceBefore=6, spaceAfter=6)

# --- Page 1: two-column prose -----------------------------------------------
# Two Frames side by side on one page template. PyMuPDF extracts text in the
# order blocks appear in its own layout analysis, which is not guaranteed to
# respect the visual left-column-then-right-column reading order - this is
# the exact failure mode the README calls out and had never been tested.
LEFT_COL = [
    ("h1", "Meridian Robotics — Quarterly Field Report"),
    ("h2", "Deployment summary"),
    ("p", "Meridian Robotics shipped 214 units of the Falcon-9X inspection "
          "drone in the third quarter of 2026. Of these, 131 went to "
          "pipeline operators, 52 to offshore wind operators, and 31 to "
          "municipal bridge inspectors."),
    ("p", "The flagship customer this quarter was Calderon Midstream, which "
          "took delivery of 40 units for its Texas gathering-line network. "
          "Calderon reported a 63 percent reduction in manual inspection "
          "hours during the first six weeks of operation."),
    ("h2", "Battery performance"),
    ("p", "The Falcon-9X carries a 6,400 mAh battery rated for 47 minutes "
          "of continuous flight under standard conditions. In cold-weather "
          "trials at negative 15 degrees Celsius, observed flight time fell "
          "to 31 minutes, prompting the addition of a heated battery bay in "
          "the next hardware revision."),
]

RIGHT_COL = [
    ("h2", "Software release"),
    ("p", "Firmware version 3.4 shipped on 8 August 2026, adding automatic "
          "crack-width estimation to the bridge-inspection profile. The "
          "estimation model was trained on 2,900 annotated images collected "
          "during Q2 field trials and reports width in millimetres with a "
          "stated accuracy of plus or minus 0.4 millimetres."),
    ("p", "A known issue in 3.4: units operating above 3,000 metres altitude "
          "occasionally report a false low-battery warning. A fix is "
          "scheduled for version 3.5, targeted for October 2026."),
    ("h2", "Support incidents"),
    ("p", "Meridian's support desk logged 58 tickets in the quarter. The "
          "most common category, at 22 tickets, was gimbal calibration "
          "drift after shipping. The second most common, at 14 tickets, "
          "was Wi-Fi handoff failures during hangar docking."),
    ("p", "Average time to first response was 3.2 hours during business "
          "hours. Average time to resolution across all severities was "
          "1.6 days."),
]


def frame_flow(items):
    story = []
    for kind, text in items:
        style = {"h1": h1, "h2": h2, "p": body}[kind]
        story.append(Paragraph(text, style))
    return story


# --- Page 2: a data table ----------------------------------------------------
# A table of unit economics per drone model. When PyMuPDF flattens a table to
# plain text it reads cell-by-cell in whatever order the content stream puts
# them, typically row-major - but with no column separators preserved, digits
# from adjacent cells can run together into a single unreadable token.
TABLE_HEADER = ["Model", "Units sold", "Unit price (USD)",
                "Flight time (min)", "Warranty (months)"]
TABLE_ROWS = [
    ["Falcon-9X", "214", "18,400", "47", "24"],
    ["Falcon-7 Lite", "96", "11,250", "34", "18"],
    ["Kestrel-2 (tethered)", "18", "42,900", "n/a", "36"],
    ["Falcon-9X Arctic", "12", "21,600", "31", "24"],
]


def build():
    OUT.parent.mkdir(exist_ok=True)

    doc = BaseDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Meridian Robotics Quarterly Field Report",
    )

    page_w, page_h = A4
    margin = 1.8 * cm
    usable_w = page_w - 2 * margin
    gap = 0.8 * cm
    col_w = (usable_w - gap) / 2
    usable_h = page_h - 2 * margin

    left_frame = Frame(margin, margin, col_w, usable_h, id="left",
                       leftPadding=0, rightPadding=6, topPadding=0, bottomPadding=0)
    right_frame = Frame(margin + col_w + gap, margin, col_w, usable_h, id="right",
                        leftPadding=6, rightPadding=0, topPadding=0, bottomPadding=0)
    single_frame = Frame(margin, margin, usable_w, usable_h, id="single")

    doc.addPageTemplates([
        PageTemplate(id="TwoCol", frames=[left_frame, right_frame]),
        PageTemplate(id="Single", frames=[single_frame]),
    ])

    story = []
    story += frame_flow(LEFT_COL)
    # FrameBreak would normally separate columns explicitly, but that would
    # hand PyMuPDF an easier document than a real two-column PDF ever is -
    # real two-column layouts don't announce their own column breaks in the
    # extracted text either. Let natural flow fill the left frame, then the
    # right, exactly as reportlab's frame-filling logic decides.
    from reportlab.platypus import FrameBreak
    story.append(FrameBreak())
    story += frame_flow(RIGHT_COL)

    story.append(NextPageTemplate("Single"))
    story.append(PageBreak())

    story.append(Paragraph("Unit economics by model, Q3 2026", h2))
    table = Table([TABLE_HEADER] + TABLE_ROWS, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "The Kestrel-2 is tethered and has no rated free-flight time; its "
        "warranty is longer because it has no battery to degrade. The "
        "Falcon-9X Arctic costs more than the standard Falcon-9X because of "
        "its heated battery bay, described on the previous page.", body))

    doc.build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
