"""Compile paper.md + bibliography → PDF with numbered citations and References section.

Pipeline: markdown → HTML → PDF via python-markdown + weasyprint.
Improvements over v1:
  - Parses bibliography.bib and replaces [citekey] with numbered [1], [1,2] etc.
  - Appends a References section ordered by first-cite appearance
  - Skips label refs like [fig:pareto], [tab:fail], [sec:methods]
  - Handles LaTeX figure blocks (converts to img tags, swaps .pdf->.png)
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import markdown as md
from weasyprint import HTML, CSS
import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "reports" / "paper_draft"
REPORTS = ROOT / "reports"
FIGS = ROOT / "reports" / "phase3_figures"
BIB = DRAFT / "bibliography.bib"

cli = argparse.ArgumentParser(description="Compile a paper markdown file to PDF with numbered citations.")
cli.add_argument(
    "--paper-source",
    default=None,
    help="Markdown file under reports/paper_draft/ to compile. Defaults to paper_v9.md when present, else paper.md.",
)
cli.add_argument(
    "--out",
    default=None,
    help="Output PDF path. Defaults to <paper-source stem>.pdf under reports/paper_draft/.",
)
ARGS = cli.parse_args()
DEFAULT_SOURCE = "paper_v9.md" if (DRAFT / "paper_v9.md").exists() else "paper.md"
PAPER_SOURCE = DRAFT / (ARGS.paper_source or DEFAULT_SOURCE)
OUT = Path(ARGS.out) if ARGS.out else DRAFT / f"{PAPER_SOURCE.stem}.pdf"


def resolve_figure(path_str: str) -> Path | None:
    """Resolve a figure path that may be relative to paper_draft/ or to reports/."""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to paper_draft (handles "../phaseN/figures/foo.png")
    candidate = (DRAFT / p).resolve()
    if candidate.exists():
        return candidate
    # Try relative to reports/ (handles "phase3_figures/foo.png")
    candidate = (REPORTS / p).resolve()
    if candidate.exists():
        return candidate
    # Try basename in phase3_figures (legacy fallback)
    candidate = (FIGS / p.name).resolve()
    if candidate.exists():
        return candidate
    # Try basename swap to .png
    candidate = (FIGS / p.with_suffix(".png").name).resolve()
    if candidate.exists():
        return candidate
    return None

# ----- Parse bibliography ----------------------------------------------------
parser = BibTexParser(common_strings=True)
parser.customization = convert_to_unicode
parser.ignore_nonstandard_types = False
parser.homogenize_fields = False

bib_db = bibtexparser.loads(BIB.read_text(), parser=parser)
ENTRIES = {e["ID"]: e for e in bib_db.entries}
print(f"Loaded {len(ENTRIES):,} bibliography entries")

# ----- Concatenate sections --------------------------------------------------
print(f"Compiling {PAPER_SOURCE} -> {OUT}")
if PAPER_SOURCE.name == "paper_v9.md":
    raw = PAPER_SOURCE.read_text()
else:
    sections = []
    sections.append(
        "# Architectural Returns in Deep Research Are Bounded:\n"
        "## A Three-Judge Controlled Comparison of Eleven Agent Architectures\n\n"
        "*Anonymous Authors*\n\n*Affiliation withheld for review*\n\n---\n\n"
    )

    # Abstract (with title skip)
    abs_text = (DRAFT / "abstract.md").read_text()
    first_heading = abs_text.find("\n# ")
    sections.append("# Abstract\n\n" + (abs_text[first_heading:].split("\n", 2)[2] if first_heading >= 0 else abs_text) + "\n\n---\n\n")

    # Main paper
    sections.append("# Main Paper\n\n")
    sections.append(PAPER_SOURCE.read_text())

    # Future work protocols
    sections.append("\n\n---\n\n# Appendix B — Future Work Protocols\n\n")
    sections.append((DRAFT / "future_work_protocols.md").read_text())

    # Reproducibility checklist
    sections.append("\n\n---\n\n# Appendix C — Reproducibility Checklist\n\n")
    sections.append((DRAFT / "reproducibility_checklist.md").read_text())

    raw = "\n".join(sections)

# ----- Fix LaTeX figure blocks → markdown images -----------------------------
def fix_latex_image(match: re.Match) -> str:
    path = match.group(1)
    abs_path = resolve_figure(path)
    if abs_path is None:
        # Last resort: try swapping .pdf to .png
        png_alt = path.replace(".pdf", ".png")
        abs_path = resolve_figure(png_alt)
    if abs_path is None:
        return ""
    # Prefer PNG for weasyprint
    if abs_path.suffix.lower() == ".pdf":
        png = abs_path.with_suffix(".png")
        if png.exists():
            abs_path = png
    return f"![]({abs_path.as_uri()})"


raw = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", fix_latex_image, raw)
raw = re.sub(r"\\caption\{([^}]*)\}", r"\n\n*\1*\n\n", raw)
raw = re.sub(r"\\label\{[^}]+\}", "", raw)
raw = re.sub(r"\\begin\{(figure|table)[^\}]*\}", "", raw)
raw = re.sub(r"\\end\{(figure|table)\}", "", raw)
raw = re.sub(r"\\autoref\{fig:([^}]+)\}", r"Figure \1", raw)
raw = re.sub(r"\\autoref\{tab:([^}]+)\}", r"Table \1", raw)
raw = re.sub(r"\\autoref\{([^}]+)\}", r"\1", raw)
raw = re.sub(r"\\linewidth", "", raw)
raw = re.sub(r"\\centering", "", raw)

# ----- Citation handling -----------------------------------------------------
# Valid cite keys: alphanumeric + underscore. Exclude [fig:...], [tab:...], [sec:...], [eq:...]
LABEL_PREFIXES = ("fig:", "tab:", "sec:", "eq:", "Fig.", "Tab.")
CITE_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9_,;\s]*)\]")

cite_order: list[str] = []           # first-appearance order
cite_numbers: dict[str, int] = {}    # key -> number
missing_keys: set[str] = set()


def handle_cite(match: re.Match) -> str:
    inner = match.group(1).strip()
    # Multiple keys separated by commas, semicolons, or whitespace. The semicolon
    # case is common in paper_v9.md literature-review paragraphs.
    keys = [k.strip() for k in re.split(r"[,;\s]+", inner) if k.strip()]
    # filter out label references
    if any(k.startswith(LABEL_PREFIXES) for k in keys):
        return match.group(0)
    # all keys must be in the bibliography to treat as citation
    if not all(k in ENTRIES for k in keys):
        # Not a bibliography citation - could be inline text. Drop from citation
        # processing but flag the missing ones.
        for k in keys:
            if k not in ENTRIES:
                missing_keys.add(k)
        return match.group(0)
    nums = []
    for k in keys:
        if k not in cite_numbers:
            cite_order.append(k)
            cite_numbers[k] = len(cite_order)
        nums.append(str(cite_numbers[k]))
    return "[" + ", ".join(nums) + "]"


raw = CITE_PATTERN.sub(handle_cite, raw)

print(f"Citations resolved: {len(cite_order):,} unique refs")
if missing_keys:
    # Only report missing ones that look like citation keys (lowercase starts etc)
    likely = sorted([k for k in missing_keys if re.match(r"^[a-z][a-z0-9_]*[0-9]{4}", k) or re.match(r"^[a-z]+\d{4}[a-z]*$", k)])
    if likely:
        print(f"WARNING: {len(likely)} unresolved citation keys: {likely[:20]}{'...' if len(likely) > 20 else ''}")

# ----- Build References section ---------------------------------------------
def format_reference(entry: dict) -> str:
    authors = entry.get("author", "").replace("\n", " ").replace("  ", " ")
    # normalize "Last, First and Last, First and ..." to "First Last, First Last, ..."
    if authors:
        parts = [p.strip() for p in authors.split(" and ")]
        norm = []
        for p in parts:
            if "," in p:
                last, first = [s.strip() for s in p.split(",", 1)]
                norm.append(f"{first} {last}")
            else:
                norm.append(p)
        if len(norm) > 6:
            authors = ", ".join(norm[:6]) + ", et al."
        else:
            authors = ", ".join(norm)
    title = entry.get("title", "").strip().strip("{}").replace("{", "").replace("}", "")
    year = entry.get("year", "").strip()
    venue = entry.get("booktitle") or entry.get("journal") or entry.get("publisher") or entry.get("institution") or ""
    venue = venue.strip().strip("{}")
    # arxiv / doi / url
    note = entry.get("note", "")
    eprint = entry.get("eprint", "")
    doi = entry.get("doi", "")
    url = entry.get("url", "")

    tail = []
    if venue:
        tail.append(f"*{venue}*")
    if year:
        tail.append(year)
    if eprint:
        tail.append(f"arXiv:{eprint}")
    elif note and "arXiv:" in note:
        arxiv_id = re.search(r"arXiv:([0-9.]+)", note)
        if arxiv_id:
            tail.append(f"arXiv:{arxiv_id.group(1)}")
    if doi:
        tail.append(f"doi:{doi}")
    if url and not eprint:
        tail.append(url)

    pieces = []
    if authors:
        pieces.append(authors)
    if title:
        pieces.append(f'"{title}"')
    if tail:
        pieces.append(". ".join(tail) + ".")
    return ". ".join(pieces)


if cite_order:
    refs_md = ["\n\n---\n\n# References\n\n"]
    for i, key in enumerate(cite_order, 1):
        entry = ENTRIES[key]
        refs_md.append(f"{i}. {format_reference(entry)}\n")
    raw += "\n".join(refs_md)

# ----- Resolve image URIs for weasyprint -------------------------------------
def fix_md_image(match: re.Match) -> str:
    path = match.group(1)
    if path.startswith(("file" + "://", "http://", "https://")):
        return match.group(0)
    abs_path = resolve_figure(path)
    if abs_path is not None:
        return f"![]({abs_path.as_uri()})"
    return match.group(0)


raw = re.sub(r"!\[\]\(([^)]+)\)", fix_md_image, raw)

# ----- Render HTML -----------------------------------------------------------
html_body = md.markdown(raw, extensions=["tables", "fenced_code", "toc"])

CSS_STR = """
@page {
    size: A4;
    margin: 2.0cm 2.2cm;
    @bottom-center { content: counter(page); font-size: 9pt; color: #555; }
}
body {
    font-family: "Charter", "Georgia", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #111;
}
h1 { font-size: 18pt; margin-top: 1.2em; border-bottom: 2px solid #333; padding-bottom: 0.2em; page-break-before: auto; }
h2 { font-size: 13pt; margin-top: 1.1em; color: #1a1a1a; }
h3 { font-size: 11.5pt; margin-top: 0.9em; color: #222; }
h4 { font-size: 10.8pt; margin-top: 0.8em; }
table { border-collapse: collapse; margin: 0.7em 0; font-size: 9.2pt; width: 100%; }
th, td { border: 1px solid #888; padding: 3px 6px; vertical-align: top; }
th { background: #e8e8e8; font-weight: 600; }
code { font-family: "JetBrains Mono", "Inconsolata", monospace; font-size: 9pt; background: #f2f2f2; padding: 1px 3px; }
pre { background: #f4f4f4; padding: 8px; border-left: 3px solid #888; overflow-x: auto; font-size: 8.8pt; }
img { max-width: 100%; margin: 8px 0; display: block; }
blockquote { border-left: 3px solid #aaa; padding-left: 10px; color: #444; font-style: italic; }
hr { border: none; border-top: 1px solid #aaa; margin: 1.5em 0; }
a { color: #0066cc; text-decoration: none; }
/* References are tight */
h1:has(+ ol) + ol, h1#references + ol { font-size: 9pt; line-height: 1.35; }
"""

html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{PAPER_SOURCE.stem}</title></head>
<body>
{html_body}
</body></html>
"""

html_out = OUT.with_suffix(".html")
html_out.write_text(html_doc)
print(f"HTML written: {html_out}  ({len(html_doc):,} chars)")

print(f"Compiling PDF...")
HTML(string=html_doc).write_pdf(str(OUT), stylesheets=[CSS(string=CSS_STR)])
print(f"PDF written: {OUT}  ({OUT.stat().st_size:,} bytes)")
