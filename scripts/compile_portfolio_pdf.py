#!/usr/bin/env python3
"""Render the four-part portfolio into a single PDF.

Concatenates PORTFOLIO.md (index) + EXECUTIVE_SUMMARY + CV_BULLETS +
TECHNICAL_DEEP_DIVE + ARTEFACTS_INVENTORY into one HTML, then PDF.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = ROOT / "reports" / "portfolio"
ORDER = [
    "PORTFOLIO.md",
    "EXECUTIVE_SUMMARY.md",
    "CV_BULLETS.md",
    "TECHNICAL_DEEP_DIVE.md",
    "ARTEFACTS_INVENTORY.md",
]


def main():
    import markdown
    md_chunks = []
    for name in ORDER:
        path = PORT / name
        if not path.exists():
            print(f"WARN: missing {path}", file=sys.stderr)
            continue
        text = path.read_text()
        md_chunks.append(f"\n\n<!-- ===== {name} ===== -->\n\n{text}\n\n<div style='page-break-after: always;'></div>\n")
    combined = "\n".join(md_chunks)
    html_body = markdown.markdown(
        combined,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
    )
    style = """
    <style>
      @page { size: A4; margin: 1.2cm; }
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                          Helvetica, Arial, sans-serif;
             font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; max-width: 17.5cm; margin: 0 auto; }
      h1 { font-size: 22pt; border-bottom: 2px solid #1a1a1a; padding-bottom: 0.2em;
           margin-top: 1em; page-break-before: auto; }
      h2 { font-size: 14pt; margin-top: 1.4em; border-bottom: 1px solid #888; padding-bottom: 0.15em; }
      h3 { font-size: 12pt; margin-top: 1.2em; color: #333; }
      h4 { font-size: 11pt; margin-top: 1em; color: #555; }
      code { font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
             background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
      pre { background: #f4f4f4; padding: 8px 10px; border-radius: 4px; overflow-x: auto;
            font-size: 9pt; line-height: 1.35; }
      pre code { background: transparent; padding: 0; }
      table { border-collapse: collapse; margin: 0.6em 0; font-size: 9.5pt; width: 100%; }
      th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
      th { background: #f0f0f0; font-weight: 600; }
      blockquote { border-left: 3px solid #888; margin: 0.5em 0; padding-left: 0.8em;
                   color: #555; font-style: italic; }
      a { color: #0066cc; text-decoration: none; }
      hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
    </style>
    """
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Deep Research Projects — Portfolio</title>
{style}
</head><body>
{html_body}
</body></html>
"""
    out_html = PORT / "PORTFOLIO.html"
    out_html.write_text(html)
    print(f"HTML: {out_html} ({len(html):,} chars)")

    # Render PDF via weasyprint or playwright
    try:
        from weasyprint import HTML
        out_pdf = PORT / "PORTFOLIO.pdf"
        HTML(string=html, base_url=str(PORT)).write_pdf(str(out_pdf))
        print(f"PDF:  {out_pdf} ({out_pdf.stat().st_size:,} bytes)")
    except ImportError:
        # fall back to pdfkit / wkhtmltopdf if weasyprint unavailable
        try:
            import pdfkit
            out_pdf = PORT / "PORTFOLIO.pdf"
            pdfkit.from_string(html, str(out_pdf))
            print(f"PDF (pdfkit): {out_pdf}")
        except Exception as e:
            print(f"WARN: PDF render failed: {e}", file=sys.stderr)
            print("HTML available; install weasyprint to enable PDF render.")


if __name__ == "__main__":
    main()
