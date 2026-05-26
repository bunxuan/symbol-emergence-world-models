from __future__ import annotations

import re
from pathlib import Path

import markdown as mdlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "mini_report.md"
PDF_PATH = ROOT / "mini_report.pdf"


def register_fonts() -> None:
    try:
        import matplotlib

        font_dir = (
            Path(matplotlib.__file__).resolve().parent / "mpl-data" / "fonts" / "ttf"
        )
        pdfmetrics.registerFont(
            TTFont("DejaVuSerif", str(font_dir / "DejaVuSerif.ttf"))
        )
        pdfmetrics.registerFont(TTFont("DejaVuSans", str(font_dir / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(
            TTFont("DejaVuSansMono", str(font_dir / "DejaVuSansMono.ttf"))
        )
    except Exception:
        # Fallback to built-in fonts if the bundled DejaVu fonts are unavailable.
        pass


def styleset():
    base = getSampleStyleSheet()
    serif = (
        "DejaVuSerif"
        if "DejaVuSerif" in pdfmetrics.getRegisteredFontNames()
        else "Times-Roman"
    )
    sans = (
        "DejaVuSans"
        if "DejaVuSans" in pdfmetrics.getRegisteredFontNames()
        else "Helvetica"
    )
    mono = (
        "DejaVuSansMono"
        if "DejaVuSansMono" in pdfmetrics.getRegisteredFontNames()
        else "Courier"
    )

    base.add(
        ParagraphStyle(
            name="DocTitle",
            parent=base["Title"],
            fontName=sans,
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="DocSubtitle",
            parent=base["Italic"],
            fontName=serif,
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=10,
        )
    )
    base.add(
        ParagraphStyle(
            name="TopNote",
            parent=base["BodyText"],
            fontName=sans,
            fontSize=9.2,
            leading=11.5,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="H1X",
            parent=base["Heading1"],
            fontName=sans,
            fontSize=16,
            leading=19,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#0f172a"),
        )
    )
    base.add(
        ParagraphStyle(
            name="H2X",
            parent=base["Heading2"],
            fontName=sans,
            fontSize=13,
            leading=16,
            spaceBefore=8,
            spaceAfter=5,
            textColor=colors.HexColor("#0f172a"),
        )
    )
    base.add(
        ParagraphStyle(
            name="H3X",
            parent=base["Heading3"],
            fontName=sans,
            fontSize=11.5,
            leading=14,
            spaceBefore=6,
            spaceAfter=4,
            textColor=colors.HexColor("#0f172a"),
        )
    )
    base.add(
        ParagraphStyle(
            name="BodyX",
            parent=base["BodyText"],
            fontName=serif,
            fontSize=10.6,
            leading=14,
            spaceAfter=5,
        )
    )
    base.add(
        ParagraphStyle(
            name="QuoteX",
            parent=base["BodyText"],
            fontName=serif,
            fontSize=10.6,
            leading=14,
            leftIndent=14,
            rightIndent=8,
            textColor=colors.HexColor("#334155"),
            spaceAfter=5,
        )
    )
    base.add(
        ParagraphStyle(
            name="EqX",
            parent=base["BodyText"],
            fontName=serif,
            fontSize=10.4,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
            spaceBefore=3,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="CodeX",
            parent=base["Code"],
            fontName=mono,
            fontSize=8.3,
            leading=10.2,
            leftIndent=10,
            rightIndent=10,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#e5e7eb"),
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="TableCellX",
            parent=base["BodyText"],
            fontName=sans,
            fontSize=8.5,
            leading=10.2,
        )
    )
    base.add(
        ParagraphStyle(
            name="CaptionX",
            parent=base["Italic"],
            fontName=serif,
            fontSize=9.2,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
            spaceAfter=5,
        )
    )
    return base, serif, sans, mono


def inline_markup(text: str) -> str:
    html = mdlib.markdown(text, extensions=["extra"])
    html = html.strip()
    if html.startswith("<p>") and html.endswith("</p>"):
        html = html[3:-4]
    html = html.replace("<strong>", "<b>").replace("</strong>", "</b>")
    html = html.replace("<em>", "<i>").replace("</em>", "</i>")
    html = html.replace("<code>", "<font face='DejaVuSansMono'>").replace(
        "</code>", "</font>"
    )
    return html


def para(text: str, style: str, styles) -> Paragraph:
    return Paragraph(inline_markup(text), styles[style])


def split_md_table_row(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return cells


def is_table_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped))


def build_table(block: list[str], styles, width: float) -> Table:
    rows = [split_md_table_row(row) for row in block]
    header = rows[0]
    data = rows[2:] if len(rows) > 2 and is_table_separator(block[1]) else rows[1:]
    body = [header] + data
    col_count = len(header)
    if col_count == 3:
        col_widths = [0.22 * width, 0.38 * width, 0.40 * width]
    elif col_count == 2:
        col_widths = [0.35 * width, 0.65 * width]
    else:
        col_widths = [width / col_count] * col_count

    table_data = []
    for row in body:
        table_data.append(
            [Paragraph(inline_markup(cell), styles["TableCellX"]) for cell in row]
        )

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    (
                        "DejaVuSans"
                        if "DejaVuSans" in pdfmetrics.getRegisteredFontNames()
                        else "Helvetica-Bold"
                    ),
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (-1, -1),
                    (
                        "DejaVuSans"
                        if "DejaVuSans" in pdfmetrics.getRegisteredFontNames()
                        else "Helvetica"
                    ),
                ),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10.2),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def build_story(text: str, styles, doc_width: float):
    story = []
    lines = text.splitlines()
    i = 0

    def flush_paragraph(buf: list[str]):
        if not buf:
            return
        joined = " ".join(piece.strip() for piece in buf).strip()
        if joined:
            story.append(Paragraph(inline_markup(joined), styles["BodyX"]))
        buf.clear()

    # front matter notes before the title
    front_buf: list[str] = []
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("# "):
            break
        if stripped == '<p align="center">':
            flush_paragraph(front_buf)
            link_label = None
            while i < len(lines) and lines[i].strip() != "</p>":
                m = re.search(r">(.*?)</a>", lines[i])
                if m:
                    link_label = re.sub(r"\s+", " ", m.group(1).strip())
                i += 1
            if link_label:
                story.append(Paragraph(link_label, styles["TopNote"]))
            i += 1
            continue
        if stripped:
            front_buf.append(stripped)
        i += 1
    flush_paragraph(front_buf)

    # title and subtitle
    if i < len(lines) and lines[i].startswith("# "):
        story.append(Paragraph(lines[i][2:].strip(), styles["DocTitle"]))
        i += 1
        if (
            i < len(lines)
            and lines[i].strip().startswith("_")
            and lines[i].strip().endswith("_")
        ):
            story.append(Paragraph(lines[i].strip().strip("_"), styles["DocSubtitle"]))
            i += 1

    # the rest
    paragraph_buf: list[str] = []
    in_code = False
    code_buf: list[str] = []
    in_math_block = False
    math_buf: list[str] = []

    def flush_math():
        nonlocal math_buf
        for line in math_buf:
            cleaned = line.strip()
            if cleaned:
                story.append(Paragraph(cleaned, styles["EqX"]))
        math_buf = []

    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(paragraph_buf)
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                story.append(Preformatted("\n".join(code_buf), styles["CodeX"]))
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if stripped == "---":
            flush_paragraph(paragraph_buf)
            flush_math()
            story.append(Spacer(1, 5))
            story.append(Paragraph("&nbsp;", styles["BodyX"]))
            story.append(Spacer(1, 3))
            i += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph_buf)
            flush_math()
            story.append(Paragraph(stripped[3:].strip(), styles["H1X"]))
            i += 1
            continue

        if stripped.startswith("### "):
            flush_paragraph(paragraph_buf)
            flush_math()
            story.append(Paragraph(stripped[4:].strip(), styles["H2X"]))
            i += 1
            continue

        if stripped.startswith("#### "):
            flush_paragraph(paragraph_buf)
            flush_math()
            story.append(Paragraph(stripped[5:].strip(), styles["H3X"]))
            i += 1
            continue

        if stripped.startswith("> "):
            flush_paragraph(paragraph_buf)
            flush_math()
            story.append(
                Paragraph(inline_markup(stripped[2:].strip()), styles["QuoteX"])
            )
            i += 1
            continue

        if stripped.startswith("!"):
            flush_paragraph(paragraph_buf)
            flush_math()
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                rel_path = m.group(2)
                img_path = (ROOT / rel_path).resolve()
                if img_path.exists():
                    from reportlab.platypus import Image as RLImage

                    img = RLImage(str(img_path))
                    max_width = doc_width
                    scale = min(1.0, max_width / img.drawWidth)
                    img.drawWidth *= scale
                    img.drawHeight *= scale
                    img.hAlign = "CENTER"
                    story.append(img)
            i += 1
            continue

        # markdown table block
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and is_table_separator(lines[i + 1])
        ):
            flush_paragraph(paragraph_buf)
            flush_math()
            block = [stripped, lines[i + 1].strip()]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            story.append(build_table(block, styles, doc_width))
            story.append(Spacer(1, 5))
            continue

        # Display-math-like indented lines in the appendix.
        if line.startswith("    ") and stripped:
            flush_paragraph(paragraph_buf)
            math_buf.append(stripped)
            i += 1
            continue
        else:
            if math_buf:
                flush_math()

        if not stripped:
            flush_paragraph(paragraph_buf)
            story.append(Spacer(1, 3))
            i += 1
            continue

        paragraph_buf.append(line)
        i += 1

    flush_paragraph(paragraph_buf)
    flush_math()
    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def main():
    register_fonts()
    styles, _, _, _ = styleset()
    text = MD_PATH.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Symbol Emergence from Predictive Dynamics in a 1D World Model",
        author="Xu Wenxuan",
    )

    story = build_story(text, styles, doc.width)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
