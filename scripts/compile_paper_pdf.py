"""Compile paper.md (+ abstract, contributions, appendices) into a single PDF.

Uses markdown → HTML → PDF (weasyprint). LaTeX-style figures included as embedded images.
"""
from __future__ import annotations
import re
from pathlib import Path
import markdown as md
from weasyprint import HTML, CSS

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "reports" / "paper_draft"
FIGS = ROOT / "reports" / "phase3_figures"
OUT = DRAFT / "paper.pdf"

# Concatenate sections in the right order
sections = []

# Title page
sections.append("""# Architectural Returns in Deep Research Are Bounded:
## A Three-Judge Comparison of Eleven Agent Architectures

*Anonymous Authors*

*Affiliation withheld for review*

---

""")

# Abstract
sections.append("# Abstract\n\n" + (DRAFT / "abstract.md").read_text().split("\n", 2)[2] + "\n\n")

# Main paper (skip its abstract since we already have it)
paper_text = (DRAFT / "paper.md").read_text()
# Find first "## " heading to skip duplicate abstract
sections.append("\n\n---\n\n# Main Paper\n\n")
sections.append(paper_text)

# Future work protocols (appendix B)
sections.append("\n\n---\n\n# Appendix B — Future Work Protocols\n\n")
sections.append((DRAFT / "future_work_protocols.md").read_text())

# Reproducibility checklist
sections.append("\n\n---\n\n# Appendix C — Reproducibility Checklist\n\n")
sections.append((DRAFT / "reproducibility_checklist.md").read_text())

# Revision notes
if (DRAFT / "REVISION_NOTES.md").exists():
    sections.append("\n\n---\n\n# Appendix D — Revision Notes\n\n")
    sections.append((DRAFT / "REVISION_NOTES.md").read_text())

raw = "\n".join(sections)

# Resolve figure paths to absolute file URIs for weasyprint
def fix_image(match):
    path = match.group(1)
    if path.startswith("phase3_figures/"):
        abs_path = (FIGS / Path(path).name).resolve()
        if abs_path.exists():
            return f"![]({abs_path.as_uri()})"
    return match.group(0)

raw = re.sub(r"!\[\]\(([^)]+)\)", fix_image, raw)
# also handle latex \includegraphics calls in the markdown
def fix_latex_image(match):
    path = match.group(1)
    if path.startswith("phase3_figures/"):
        abs_path = (FIGS / Path(path).name).resolve()
        # convert .pdf to .png for weasyprint
        png_path = abs_path.with_suffix(".png")
        if png_path.exists():
            return f"![]({png_path.as_uri()})"
        if abs_path.exists():
            return f"![]({abs_path.as_uri()})"
    return ""
raw = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", fix_latex_image, raw)
# strip latex figure environments leaving the image
raw = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", lambda m: m.group(0), raw, flags=re.DOTALL)
raw = re.sub(r"\\caption\{([^}]*)\}", r"\n\n*Figure caption:* \1\n\n", raw)
raw = re.sub(r"\\label\{[^}]+\}", "", raw)
raw = re.sub(r"\\begin\{(figure|table)[^\}]*\}", "", raw)
raw = re.sub(r"\\end\{(figure|table)\}", "", raw)
raw = re.sub(r"\\autoref\{([^}]+)\}", r"Fig./Tab. \1", raw)

html_body = md.markdown(raw, extensions=["tables", "fenced_code", "toc"])

CSS_STR = """
@page {
    size: A4;
    margin: 2.0cm 2.5cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #555; }
}
body {
    font-family: "Charter", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #111;
}
h1 { font-size: 20pt; margin-top: 1.2em; border-bottom: 2px solid #333; padding-bottom: 0.2em; }
h2 { font-size: 14pt; margin-top: 1.1em; }
h3 { font-size: 12pt; margin-top: 1em; }
h4 { font-size: 11pt; margin-top: 1em; }
table { border-collapse: collapse; margin: 0.7em 0; font-size: 9.5pt; }
th, td { border: 1px solid #888; padding: 4px 8px; vertical-align: top; }
th { background: #eee; }
code { font-family: "Inconsolata", monospace; font-size: 9.5pt; background: #f5f5f5; padding: 1px 3px; }
pre { background: #f5f5f5; padding: 8px; border-left: 3px solid #888; overflow-x: auto; font-size: 9pt; }
img { max-width: 100%; margin: 8px 0; }
blockquote { border-left: 3px solid #aaa; padding-left: 10px; color: #444; font-style: italic; }
hr { border: none; border-top: 1px solid #aaa; margin: 1.5em 0; }
a { color: #0066cc; text-decoration: none; }
"""

html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Paper</title></head>
<body>
{html_body}
</body></html>
"""

(DRAFT / "paper.html").write_text(html_doc)
print(f"HTML written: {DRAFT / 'paper.html'}  ({len(html_doc):,} chars)")

print(f"Compiling to PDF...")
HTML(string=html_doc).write_pdf(str(OUT), stylesheets=[CSS(string=CSS_STR)])
print(f"PDF written: {OUT}  ({OUT.stat().st_size:,} bytes)")
