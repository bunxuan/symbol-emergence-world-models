from pathlib import Path
import re

p = Path("mini_report_converted.tex")
s = p.read_text(encoding="utf8")


# replace HTML center block with LaTeX link
def _replace_html(m):
    return (
        "\\begin{center}"
        + "\n"
        + "\\vspace{-1em}"
        + "\n"
        + "\\href{../README.md}{Back to Project README}"
        + "\n"
        + "\\end{center}"
        + "\n\n"
    )


s = re.sub(r"<p\s+align=\"center\"[\s\S]*?</p>\s*\n", _replace_html, s)
# fix leftover markdown-ish title/author lines if present
s = s.replace("\\# Symbol Emergence from Predictive Dynamics in a 1D World Model", "")
# ensure document has title/author/maketitle at top (if not already)
if "\\maketitle" not in s.splitlines()[:30]:
    # insert maketitle after \begin{document}
    s = s.replace("\\begin{document}\n", "\\begin{document}\n\\maketitle\n")
# small cleanup: remove literal Markdown backslashes before hashes/underscores
s = s.replace("\\_", "_")
# write back
p.write_text(s, encoding="utf8")
print("patched")
