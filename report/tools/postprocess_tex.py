import re
from pathlib import Path

p = Path("mini_report_converted.tex")
text = p.read_text(encoding="utf8")

# Fix double-backslashes before common macros (\texttt, \emph, \href)
text = text.replace("\\\\texttt{", "\\texttt{")
text = text.replace("\\\\emph{", "\\emph{")
text = text.replace("\\\\href{", "\\href{")

# Remove leftover literal Markdown H1 lines like "# Title"
text = re.sub(r"^#\s+.*\n", "", text, flags=re.M)

# Strip manual numeric prefixes in section/subsection titles
text = re.sub(r"\\section\{\s*\d+\.\s*(.*?)\}", r"\\section{\1}", text)
text = re.sub(r"\\subsection\{\s*\d+(?:\.\d+)?\.?\s*(.*?)\}", r"\\subsection{\1}", text)

# Replace Implementation Notes verbatim blocks (if present) with a wrapped table
impl_start = text.find("\\subsection{Implementation Notes}")
if impl_start != -1:
    # find the next occurrence of the paragraph that begins the list (we look for the first \begin{verbatim} after it)
    m = re.search(
        r"\\subsection\{Implementation Notes\}.*?\\begin\{verbatim\}",
        text[impl_start:],
        flags=re.S,
    )
    if m:
        # find end of the block of consecutive verbatim blocks
        start_idx = impl_start + m.end() - len("\\begin{verbatim}")
        # look forward to the next section or references
        end_match = re.search(r"\\\\section\{|\\end\{document\}", text[start_idx:])
        if end_match:
            end_idx = start_idx + end_match.start()
        else:
            end_idx = len(text)
        # replacement table
        table = """\\subsection{Implementation Notes}

The repository contains the code used to generate the figures and the corresponding training scripts. The default settings are summarized below for readers who want to reproduce or extend the experiments.

\\begin{table}[h]
\\centering
\\small
\\begin{tabular}{p{0.22\\textwidth} p{0.36\\textwidth} p{0.36\\textwidth}}
\\toprule
\\textbf{Item} & \\\textbf{Value (repository)} & \\\textbf{Notes} \\\\\
\\midrule
Environment & 1D bouncing-ball (deterministic) & \\\texttt{data/generate_data.py} \\\\\
Latent dimension & 16 & Default CLI value used by \\\texttt{model/train.py} \\\\\
Encoder & Linear(1,32) -> ReLU -> Linear(32, latent\\_dim) & \\\texttt{model/world\\_model.py::WorldModel} \\\\\
Decoder & Linear(latent\\_dim,32) -> ReLU -> Linear(32,1) & \\\texttt{model/world\\_model.py::WorldModel} \\\\\
Optimizer & Adam & Used across trainers \\\\\
Learning rate & 1e-3 & Default training rate in \\\texttt{model/train.py} \\\\\
Batch size & 32 (world), 128 (flow/diffusion) & Dispatcher defaults \\\\\
Epochs & 50 (world), 300 (flow), 1000 (diffusion) & Default epoch schedule \\\\\
Flow model & RealNVP, num_layers=8, hidden_dim=128 & Invertible baseline \\\\\
Diffusion model & timesteps=1000, hidden_dim=256, time_embed_dim=128 & Generative baseline \\\\\
PCA & \\\texttt{sklearn.decomposition.PCA(n_components=2)} & Latent-geometry summary \\\\\
Clustering & \\\texttt{KMeans(n_clusters=4, n_init=20, random_state=0)} & Symbolic segmentation \\\\\
Jacobian computation & autograd & See the subsection above \\\\\
\\bottomrule
\\end{tabular}
\\end{table}
"""
        # perform replacement
        text = text[:impl_start] + table + text[end_idx:]

# Ensure single backslash before texttt/emph/href
text = text.replace("\\\\texttt{", "\\texttt{")
text = text.replace("\\\\emph{", "\\emph{")
text = text.replace("\\\\href{", "\\href{")

p.write_text(text, encoding="utf8")
print("postprocessed")
