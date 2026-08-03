"""Assemble paper.md from expanded sections_v2/*.md files.

Replaces §1, §2, §9-§11 in paper.md with the expanded versions, renumbering
sections so the new Discussion (previously absent) sits between Failure Analysis
and Limitations:

OLD layout:
  §1 Introduction
  §2 Related Work
  §3 Pattern Taxonomy
  §4 Experimental Setup
  §5 Main Results
  §6 The Retrieval Bottleneck
  §7 Ablations
  §8 Failure Analysis
  §9 Limitations
  §10 Conclusion

NEW layout:
  §1 Introduction          [expanded from sections_v2/01_introduction.md]
  §2 Related Work          [rewritten from sections_v2/02_related_work.md]
  §3 Pattern Taxonomy      [kept]
  §4 Experimental Setup    [kept]
  §5 Main Results          [kept]
  §6 The Retrieval Bottleneck [kept]
  §7 Ablations             [kept]
  §8 Failure Analysis      [kept]
  §9 Discussion            [NEW from sections_v2/09_discussion.md]
  §10 Limitations          [expanded from sections_v2/10_limitations.md]
  §11 Conclusion           [kept, re-numbered]
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "reports" / "paper_draft"
V2 = DRAFT / "sections_v2"

paper_md = (DRAFT / "paper.md").read_text()

# --- Extract preamble (title, abstract) up to "## 1 Introduction" -----------
intro_start = paper_md.find("\n## 1 Introduction")
if intro_start < 0:
    raise RuntimeError("Could not find '## 1 Introduction' in paper.md")
preamble = paper_md[:intro_start].rstrip() + "\n\n"

# --- Section 1 -> replace with expanded version ------------------------------
sec2_start = paper_md.find("\n## 2 Related Work")
if sec2_start < 0:
    raise RuntimeError("Could not find '## 2 Related Work' in paper.md")
new_sec1 = (V2 / "01_introduction.md").read_text().strip()
# Normalise heading if the file begins with "## 1 Introduction"
new_sec1 = re.sub(r"^#+\s*1?\s*Introduction\s*\n", "## 1 Introduction\n\n", new_sec1, count=1, flags=re.MULTILINE)
if not new_sec1.startswith("## 1 Introduction"):
    new_sec1 = "## 1 Introduction\n\n" + new_sec1

# --- Section 2 -> replace with expanded version ------------------------------
sec3_start = paper_md.find("\n## 3 Pattern Taxonomy")
if sec3_start < 0:
    raise RuntimeError("Could not find '## 3 Pattern Taxonomy' in paper.md")
new_sec2 = (V2 / "02_related_work.md").read_text().strip()
new_sec2 = re.sub(r"^#+\s*2?\s*Related Work\s*\n", "## 2 Related Work\n\n", new_sec2, count=1, flags=re.MULTILINE)
if not new_sec2.startswith("## 2 Related Work"):
    new_sec2 = "## 2 Related Work\n\n" + new_sec2

# --- Sections 3-8 keep unchanged ---------------------------------------------
sec9_start = paper_md.find("\n## 9 Limitations")
if sec9_start < 0:
    raise RuntimeError("Could not find '## 9 Limitations' in paper.md")
middle = paper_md[sec3_start:sec9_start].strip()

# --- Section 9 Discussion (NEW) ---------------------------------------------
new_sec9 = (V2 / "09_discussion.md").read_text().strip()
new_sec9 = re.sub(r"^#+\s*9?\s*Discussion\s*\n", "## 9 Discussion\n\n", new_sec9, count=1, flags=re.MULTILINE)
if not new_sec9.startswith("## 9 Discussion"):
    new_sec9 = "## 9 Discussion\n\n" + new_sec9

# --- Section 10 Limitations (was §9, expanded) ------------------------------
new_sec10 = (V2 / "10_limitations.md").read_text().strip()
new_sec10 = re.sub(r"^#+\s*(9|10)?\s*Limitations\s*\n", "## 10 Limitations\n\n", new_sec10, count=1, flags=re.MULTILINE)
if not new_sec10.startswith("## 10 Limitations"):
    new_sec10 = "## 10 Limitations\n\n" + new_sec10

# --- Section 11 Conclusion (was §10) ----------------------------------------
sec10_start = paper_md.find("\n## 10 Conclusion")
if sec10_start < 0:
    raise RuntimeError("Could not find '## 10 Conclusion' in paper.md")
# Extract old §10 Conclusion through the end-of-body, up to Figures/Tables appendices
# Find where conclusion ends (next "---" or "## Acknowledgments")
conc_body_start = sec10_start
# Find end of Conclusion body: look for "## Acknowledgments"
ack_start = paper_md.find("\n## Acknowledgments", conc_body_start)
if ack_start < 0:
    raise RuntimeError("Could not find '## Acknowledgments' after Conclusion")
old_conc = paper_md[conc_body_start:ack_start].strip()
# Renumber heading
new_sec11 = re.sub(r"^##\s*10\s*Conclusion", "## 11 Conclusion", old_conc, count=1)

# Tail: Acknowledgments + Figures + Tables appendix
tail = paper_md[ack_start:]

# --- Apply methods_citations_patch.md if it exists --------------------------
patch_path = V2 / "methods_citations_patch.md"
if patch_path.exists():
    # Patches are documentation - applied manually by a follow-up step.
    # For now, surface a note that patches exist so the build can proceed.
    print(f"NOTE: {patch_path} exists — Methods-body citations will be applied separately.")

# --- Assemble ----------------------------------------------------------------
assembled = [
    preamble.rstrip(),
    "",
    new_sec1,
    "",
    "---",
    "",
    new_sec2,
    "",
    "---",
    "",
    middle,
    "",
    "---",
    "",
    new_sec9,
    "",
    "---",
    "",
    new_sec10,
    "",
    "---",
    "",
    new_sec11,
    "",
    "---",
    "",
    tail.strip(),
    "",
]

output = "\n".join(assembled)
(DRAFT / "paper.md").write_text(output)
print(f"Assembled paper.md: {len(output):,} chars, {output.count(chr(10)):,} lines")

# Word count
word_count = len(output.split())
print(f"Word count: ~{word_count:,} words")
