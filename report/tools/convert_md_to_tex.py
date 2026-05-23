import re
from pathlib import Path

md = Path("mini_report.md").read_text(encoding="utf8")
lines = md.splitlines()
# extract title (first H1)
title = "Document"
out_lines = []
if lines and lines[0].startswith("# "):
    title = lines[0][2:].strip()
    lines = lines[1:]

in_code = False
in_table = False
table_rows = []
for i, ln in enumerate(lines):
    if ln.strip().startswith("```"):
        if not in_code:
            out_lines.append("\\begin{verbatim}")
            in_code = True
        else:
            out_lines.append("\\end{verbatim}")
            in_code = False
        continue
    if in_code:
        out_lines.append(ln)
        continue
    # headings
    if ln.startswith("## "):
        hdr = ln[3:].strip()
        if hdr.lower() == "abstract":
            out_lines.append("\\begin{abstract}")
            # gather following paragraph lines until blank or next heading
            j = i + 1
            while j < len(lines) and not re.match(r"^(#{1,4})\s", lines[j]):
                out_lines.append(lines[j])
                j += 1
            out_lines.append("\\end{abstract}")
            # skip consumed lines
            for _ in range(i + 1, j):
                lines[_] = ""
            continue
        out_lines.append("\\section{" + hdr.replace("\\", "\\\\") + "}")
        continue
    if ln.startswith("### "):
        out_lines.append("\\subsection{" + ln[4:].strip().replace("\\", "\\\\") + "}")
        continue
    if ln.startswith("#### "):
        out_lines.append(
            "\\subsubsection{" + ln[5:].strip().replace("\\", "\\\\") + "}"
        )
        continue
    # horizontal rule
    if ln.strip() == "---":
        out_lines.append("\\bigskip\\hrule\\bigskip")
        continue
    # images
    m = re.search(r"!\[(.*?)\]\((.*?)\)", ln)
    if m:
        alt, path = m.group(1), m.group(2)
        out_lines.append("\\begin{figure}[h]")
        out_lines.append("\\centering")
        out_lines.append("\\includegraphics[width=0.9\\linewidth]{" + path + "}")
        out_lines.append("\\caption{" + alt.replace("\\", "\\\\") + "}")
        out_lines.append("\\end{figure}")
        continue
    # tables: naive: if line contains | and next line contains --- treat as table
    if "|" in ln and re.search(r"\|", ln):
        # collect table
        # naive handling: output as verbatim
        out_lines.append("\\begin{verbatim}")
        out_lines.append(ln)
        out_lines.append("\\end{verbatim}")
        continue
    # inline code `x`
    ln2 = re.sub(
        r"`([^`]*)`", lambda m: "\\texttt{" + m.group(1).replace("\\", "\\\\") + "}", ln
    )
    # escape TeX special chars
    ln2 = ln2.replace("%", "\\%").replace("#", "\\#").replace("&", "\\&")
    ln2 = (
        ln2.replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("$", "\\$")
    )
    out_lines.append(ln2)

preamble = (
    "% Auto-converted LaTeX from mini_report.md\n"
    "\\documentclass[11pt]{article}\n"
    "\\usepackage[utf8]{inputenc}\n"
    "\\usepackage{microtype}\n"
    "\\usepackage{graphicx}\n"
    "\\usepackage{amsmath,amssymb}\n"
    "\\usepackage{hyperref}\n"
    "\\usepackage{booktabs}\n"
    "\\usepackage{verbatim}\n"
    "\\title{" + title + "}\n"
    "\\author{Author Name(s)}\n"
    "\\date{}\n"
    "\\begin{document}\n"
    "\\maketitle\n"
)
postamble = "\\end{document}\n"
out = preamble + "\n".join(out_lines) + "\n" + postamble
# replace some common unicode symbols with LaTeX equivalents
replacements = {
    "≈": "\\approx",
    "∈": "\\in",
    "ᵀ": "^{\\top}",
    "Σ": "\\Sigma",
    "²": "^{2}",
    "…": "...",
}
for k, v in replacements.items():
    out = out.replace(k, v)

Path("mini_report_converted.tex").write_text(out, encoding="utf8")
print("WROTE mini_report_converted.tex")
