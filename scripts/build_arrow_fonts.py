"""Build the two tiny arrow fonts in web/assets/fonts/.

Press Start 2P and the Inter latin subset have no U+2190 or U+2192, and Press
Start 2P has no U+25B6, so browsers fell back to a system font for them. These
faces supply only those glyphs. See the GRIN arrows block in fonts.css.

Run from the repo root:
    uv run --no-project --with fonttools --with brotli --with skia-pathops \
        python scripts/build_arrow_fonts.py
"""

from pathlib import Path

import pathops
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.transformPen import TransformPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

FONTS = Path(__file__).resolve().parent.parent / "web" / "assets" / "fonts"

# Press Start 2P: 1000 units per em, 125-unit pixels, capitals sit in y 125..1000.
RIGHT = ["...#...", "...##..", "...###.", "#######", "...###.", "...##..", "...#..."]
LEFT = [row[::-1] for row in RIGHT]
TRIANGLE = ["#...", "##..", "###.", "####", "###.", "##..", "#..."]


def bitmap_glyph(rows, px=125, top=1000):
    path = pathops.Path()
    pen = path.getPen()
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                x0, y1 = c * px, top - r * px
                pen.moveTo((x0, y1 - px))
                pen.lineTo((x0, y1))
                pen.lineTo((x0 + px, y1))
                pen.lineTo((x0 + px, y1 - px))
                pen.closePath()
    path.simplify(fix_winding=True, keep_starting_points=False)
    out = TTGlyphPen(None)
    path.draw(out)
    return out.glyph()


def build(family, glyphs, cmap, upem, ascent, descent, advances, out, notice):
    fb = FontBuilder(upem, isTTF=True)
    order = [".notdef", *glyphs]
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf({".notdef": TTGlyphPen(None).glyph(), **glyphs})
    glyf = fb.font["glyf"]
    metrics = {}
    for name in order:
        g = glyf[name]
        g.recalcBounds(glyf)
        metrics[name] = (advances.get(name, upem // 2), getattr(g, "xMin", 0) or 0)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupNameTable(
        {
            "familyName": family,
            "styleName": "Regular",
            "copyright": notice,
            "licenseDescription": "SIL Open Font License 1.1",
            "licenseInfoURL": "https://openfontlicense.org",
        }
    )
    fb.setupOS2(sTypoAscender=ascent, sTypoDescender=descent,
                usWinAscent=ascent, usWinDescent=-descent)
    fb.setupPost()
    fb.font.flavor = "woff2"
    fb.save(out)


def main():
    build(
        "GRIN Arrows Pixel",
        {"arrowleft": bitmap_glyph(LEFT), "arrowright": bitmap_glyph(RIGHT),
         "triagrt": bitmap_glyph(TRIANGLE)},
        {0x2190: "arrowleft", 0x2192: "arrowright", 0x25B6: "triagrt"},
        1000, 1000, 0,
        {"arrowleft": 1000, "arrowright": 1000, "triagrt": 625},
        FONTS / "grin-arrows-pixel.woff2",
        "Pixel arrows drawn for GRIN to match Press Start 2P. "
        "SIL Open Font License 1.1.",
    )

    # Inter: rotate its own up arrow so the stroke weight and head match.
    up = TTFont(FONTS / "inter-latin.woff2").getGlyphSet()["arrowup"]

    def rotated(matrix):
        pen = TTGlyphPen(None)
        up.draw(TransformPen(pen, matrix))
        return pen.glyph()

    build(
        "GRIN Arrows Sans",
        {"arrowleft": rotated((0, -1, -1, 0, 1664, 1437)),   # (x, y) -> (1664 - y, 1437 - x)
         "arrowright": rotated((0, -1, 1, 0, 150, 1437))},   # (x, y) -> (y + 150, 1437 - x)
        {0x2190: "arrowleft", 0x2192: "arrowright"},
        2048, 1984, -494,
        {"arrowleft": 1814, "arrowright": 1814},
        FONTS / "grin-arrows-sans.woff2",
        "Copyright 2016 The Inter Project Authors (https://github.com/rsms/inter). "
        "Arrows rotated from Inter's up arrow for GRIN. SIL Open Font License 1.1.",
    )


if __name__ == "__main__":
    main()
