import re
from pathlib import Path

p = Path("mini_report_converted.tex")
s = p.read_text(encoding="utf8")
# replace backtick code spans `...` -> \texttt{...}
s = re.sub(
    r"`([^`]+)`",
    lambda m: r"\\texttt{"
    + m.group(1).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    + "}",
    s,
)
# replace N\approx variants
s = re.sub(r"N\\approx\s*20,000", r"$N\\approx 20{,}000$", s)
s = re.sub(r"N\\approx20,000", r"$N\\approx 20{,}000$", s)
# Wrap SVD line in math mode
s = s.replace("    X = U \\Sigma V^{\\top}", "    $X = U \\Sigma V^{\\top}$")
# Replace Sigma^2 occurrence
s = s.replace("\\Sigma^{2}", "$\\Sigma^{2}$")
# Some leftover patterns like \texttt\{...\} -> \texttt{...}
s = s.replace("\\texttt\\{", "\\texttt{").replace("\\}\)", "\\})")
# write back
p.write_text(s, encoding="utf8")
print("fixed")
