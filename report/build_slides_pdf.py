#!/usr/bin/env python3
"""slides.html -> slides.pdf (8 pages, 16:9, vector text).

Unlike slides.pptx — which embeds 2x screenshots — this prints the deck
through Chromium so the text stays real text: selectable, searchable, and
crisp at any zoom. That is what a conference upload form usually wants.

The deck shows one slide at a time (position:absolute + display:none), so a
print stylesheet is injected to lay all eight out as consecutive pages. The
root font-size is pinned rather than left on min(1vw,1.778vh), because vw/vh
resolution against the print page box is not worth depending on: at
13.333in = 1280 CSS px and 16:9, 1rem is exactly 12.8px.
"""
import asyncio
from pathlib import Path

REPORT = Path("/home/chjin/lab_internship/appt_spgemm/report")
DECK   = REPORT / "slides.html"
OUT    = REPORT / "slides.pdf"

PAGE_W, PAGE_H = 1280, 720          # CSS px == 13.333in x 7.5in at 96dpi

PRINT_CSS = f"""
@media print {{
  @page {{ size: {PAGE_W}px {PAGE_H}px; margin: 0; }}
  html {{ font-size: {PAGE_W/100}px; }}      /* 1rem == 1vw of a 16:9 page */
  body {{ background: #fff !important; overflow: visible !important; height: auto !important; }}
  .progress {{ display: none !important; }}
  .deck {{ position: static !important; inset: auto !important; }}
  .slide {{
    display: flex !important;
    position: relative !important;
    inset: auto !important;
    width: {PAGE_W}px !important;
    height: {PAGE_H}px !important;
    animation: none !important;
    break-after: page;
    overflow: hidden;
  }}
  .slide:last-child {{ break-after: auto; }}
}}
"""


async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": PAGE_W, "height": PAGE_H})
        await pg.goto(DECK.as_uri())
        await pg.wait_for_timeout(700)          # let the SVG spy plots draw
        await pg.add_style_tag(content=PRINT_CSS)
        await pg.pdf(path=str(OUT),
                     width=f"{PAGE_W}px", height=f"{PAGE_H}px",
                     print_background=True, prefer_css_page_size=True,
                     margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        await b.close()
    print(f"wrote {OUT}")

asyncio.run(main())
