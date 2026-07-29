#!/usr/bin/env python3
"""SPEECH.md -> lectern-oriented HTML -> PDF (Chromium print).

Design brief: this is read while standing and talking, so it optimises for
finding your place at a glance — big numbered slide markers, open serif at a
generous leading, pause marks visible but not interrupting, and nothing that
is spoken sharing a style with anything that isn't.
"""
import re, sys, asyncio, markdown
from pathlib import Path

SRC  = Path("/home/chjin/lab_internship/appt_spgemm/report/SPEECH.md")
HTML = Path("/home/chjin/lab_internship/appt_spgemm/report/SPEECH.html")
PDF  = Path("/home/chjin/lab_internship/appt_spgemm/report/SPEECH.pdf")

CSS = r"""
:root{
  --ink:#16191f; --ink-2:#48525f; --ink-3:#7c8695;
  --accent:#17509e; --accent-soft:#eaf0fa; --rule:#d8dee8; --panel:#f4f6fa;
  --serif:"DejaVu Serif","Liberation Serif","WenQuanYi Micro Hei",Georgia,serif;
  --sans:"DejaVu Sans","Liberation Sans","WenQuanYi Micro Hei",Arial,sans-serif;
  --mono:"DejaVu Sans Mono","Liberation Mono",monospace;
}
@page{ size:A4; margin:19mm 17mm 17mm; }
*{box-sizing:border-box}
body{font-family:var(--serif); font-size:11.4pt; line-height:1.72; color:var(--ink);
     margin:0; -webkit-print-color-adjust:exact; print-color-adjust:exact}

/* ---- masthead ---- */
h1{font-family:var(--sans); font-size:19pt; line-height:1.2; font-weight:700;
   letter-spacing:-.2pt; margin:0 0 2mm}
h3{font-family:var(--sans); font-size:9.5pt; font-weight:400; color:var(--ink-2);
   margin:0 0 5mm; letter-spacing:.02em}

/* ---- production notes: everything NOT spoken lives in this box ---- */
.notes{background:var(--panel); border-left:3pt solid var(--accent);
  padding:4mm 5mm; margin:0 0 9mm; font-family:var(--sans); font-size:8.9pt;
  line-height:1.55; color:var(--ink-2); break-inside:avoid}
.notes p{margin:0 0 2.4mm} .notes p:last-child{margin:0}
.notes em{font-style:normal} .notes strong{color:var(--ink)}
.notes::before{content:"Production notes — not spoken"; display:block;
  font-size:7.6pt; font-weight:700; letter-spacing:.11em; text-transform:uppercase;
  color:var(--accent); margin-bottom:2.2mm}

/* ---- slide sections: the number is the wayfinding device ---- */
h2{font-family:var(--sans); font-size:12.4pt; font-weight:700; color:var(--ink);
   margin:9mm 0 3.5mm; padding-bottom:2mm; border-bottom:.6pt solid var(--rule);
   break-after:avoid; break-inside:avoid}
h2 .n{display:inline-block; min-width:9.4mm; margin-right:2.6mm; padding:.4mm 0;
   text-align:center; background:var(--accent); color:#fff; border-radius:1.2mm;
   font-size:10pt; font-variant-numeric:tabular-nums}
h2.plain{margin-top:11mm}
h2.plain::before{content:""}

/* ---- spoken prose ---- */
p{margin:0 0 3.6mm; orphans:2; widows:2}
strong{font-weight:700}
code{font-family:var(--mono); font-size:.86em; background:var(--accent-soft);
  color:var(--accent); padding:.3mm 1mm; border-radius:.8mm}

/* pause mark: visible when scanning, silent when reading. Rendered as the
   literal "//" of the source so the production note stays true, and glued to
   the preceding word by an nbsp so it can never orphan onto the next line. */
.pause{font-family:var(--sans); color:var(--accent); font-size:.8em;
  opacity:.5; letter-spacing:-.3pt}

/* ---- Q&A ---- */
ul{margin:0; padding:0; list-style:none}
ul li{margin:0 0 4mm; padding-left:5mm; border-left:1.6pt solid var(--rule);
  break-inside:avoid; font-size:10.4pt; line-height:1.62; color:var(--ink-2)}
ul li strong:first-child{display:block; color:var(--ink); margin-bottom:1mm;
  font-family:var(--sans); font-size:9.9pt}

/* ---- pronunciation tables ---- */
table{width:100%; border-collapse:collapse; margin:2mm 0 6mm; font-size:9.6pt;
  break-inside:auto}
thead th{font-family:var(--sans); font-size:7.8pt; font-weight:700; text-align:left;
  text-transform:uppercase; letter-spacing:.09em; color:var(--ink-3);
  border-bottom:.6pt solid var(--rule); padding:1.4mm 2mm}
td{padding:1.5mm 2mm; border-bottom:.35pt solid var(--rule); vertical-align:top;
  line-height:1.45}
tbody tr{break-inside:avoid}
td:nth-child(2){font-family:var(--sans); color:var(--ink-2); white-space:nowrap}
h4{font-family:var(--sans); font-size:9.6pt; font-weight:700; color:var(--accent);
  margin:6mm 0 1mm; break-after:avoid}
hr{display:none}
"""

def build_html() -> str:
    raw = SRC.read_text(encoding="utf-8")

    # split the un-spoken preamble from the script proper
    head, _, rest = raw.partition("\n---\n")
    title_lines = [l for l in head.split("\n") if l.startswith("#")]
    notes_md = "\n\n".join(
        l for l in head.split("\n") if l.strip().startswith("*") and l.strip()
    )

    # pause marks -> a real glyph (they are stage direction, not punctuation)
    rest = re.sub(r"[ \t]+//[ \t]*$",
                  '<span class="pause">  //</span>', rest, flags=re.M)

    md = markdown.Markdown(extensions=["tables", "sane_lists"])
    head_html  = md.convert("\n".join(title_lines))
    md.reset(); notes_html = md.convert(notes_md)
    md.reset(); body_html  = md.convert(rest)

    # number the slide headings; leave the other h2s (Q&A) unnumbered
    def numberise(m):
        inner = m.group(1)
        sm = re.match(r"Slide (\d+)\s*—\s*(.+)", inner)
        if sm:
            return f'<h2><span class="n">{sm.group(1)}</span>{sm.group(2)}</h2>'
        return f'<h2 class="plain">{inner}</h2>'
    body_html = re.sub(r"<h2>(.*?)</h2>", numberise, body_html, flags=re.S)

    # the pronunciation guide's "# ..." became h1; demote it to a section head
    body_html = body_html.replace("<h1>", '<h2 class="plain">').replace("</h1>", "</h2>")

    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Speech Script — Structure-Aware SpGEMM</title>"
            f"<style>{CSS}</style></head><body>"
            f"{head_html}<div class='notes'>{notes_html}</div>{body_html}"
            f"</body></html>")


async def to_pdf():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page()
        await pg.goto(HTML.as_uri())
        await pg.wait_for_timeout(400)
        await pg.pdf(
            path=str(PDF), format="A4", print_background=True,
            display_header_footer=True,
            margin={"top": "19mm", "bottom": "17mm", "left": "17mm", "right": "17mm"},
            header_template="<div style='width:100%;font:7.5pt DejaVu Sans;color:#9aa3b0;"
                            "padding:0 17mm;letter-spacing:.08em'>"
                            "STRUCTURE-AWARE SpGEMM &middot; APPT 2026 &middot; SPEECH SCRIPT</div>",
            footer_template="<div style='width:100%;font:7.5pt DejaVu Sans;color:#9aa3b0;"
                            "padding:0 17mm;text-align:right'>"
                            "<span class='pageNumber'></span> / <span class='totalPages'></span></div>",
        )
        await b.close()

HTML.write_text(build_html(), encoding="utf-8")
asyncio.run(to_pdf())
print(f"wrote {HTML}\nwrote {PDF}")
