#!/usr/bin/env python3
"""Generate social.jpg (1280x640) from the woodcut hero by scaling it to fill
the card, darkening the lower-right corner with a localized scrim (so the
triremes on the left stay visible), and setting the wordmark and tagline
right-aligned in the lower-right corner.

All assets live alongside this script in scripts/.
"""

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ASSETS = Path(__file__).parent
HERO = ASSETS / "dilemma_highres_woodcut.png"
OUT = ASSETS / "social.jpg"

CARD_W, CARD_H = 1280, 640
WORDMARK = "Dilemma"
TAGLINE = "Ancient, Medieval, and Modern Greek NLP"

FONT_PATH = "/Library/Fonts/SF-Pro.ttf"
WORDMARK_VARIATION = b"Bold"
TAGLINE_VARIATION = b"Regular"
WORDMARK_SIZE = 128
TAGLINE_SIZE = 38

# Warm off-white text to sit with the sepia woodcut, with a dark sepia stroke so
# the glyphs read cleanly over the bright engraved wave-lines.
WORDMARK_FILL = (247, 240, 224, 255)
TAGLINE_FILL = (228, 219, 200, 255)
STROKE_FILL = (18, 14, 8, 255)
WORDMARK_STROKE = 2
TAGLINE_STROKE = 1

PAD_X = 56
PAD_BOTTOM = 18
WORDMARK_TAGLINE_GAP = -10

# Scrim peaks at the bottom-right corner and fades out toward the top and left,
# leaving the owl/Athena triremes on the lower-left untouched.
SCRIM_MAX_ALPHA = 240
SCRIM_TOP = 120          # rows above this stay fully clear
SCRIM_LEFT = 300         # columns left of this stay fully clear
SCRIM_RIGHT_FULL = 720   # columns at/after this get the full horizontal weight
SCRIM_VFADE_POWER = 0.75  # <1 darkens the mid-height band faster


def cover(img: Image.Image, w: int, h: int) -> Image.Image:
    src_ratio = img.width / img.height
    dst_ratio = w / h
    if src_ratio > dst_ratio:
        new_h = h
        new_w = round(h * src_ratio)
    else:
        new_w = w
        new_h = round(w / src_ratio)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return resized.crop((left, top, left + w, top + h))


def corner_scrim(w: int, h: int) -> Image.Image:
    """Black RGBA scrim whose opacity peaks at the bottom-right corner.

    Built from separable 1-D ramps (vertical * horizontal) so the darkening is
    concentrated under the wordmark/tagline and fades to nothing on the left.
    """
    vgrad = Image.new("L", (1, h))
    for y in range(h):
        t = max(0.0, (y - SCRIM_TOP) / max(1, (h - 1 - SCRIM_TOP)))
        vgrad.putpixel((0, y), round(255 * (t ** SCRIM_VFADE_POWER)))

    hgrad = Image.new("L", (w, 1))
    for x in range(w):
        t = (x - SCRIM_LEFT) / max(1, (SCRIM_RIGHT_FULL - SCRIM_LEFT))
        hgrad.putpixel((x, 0), round(255 * min(1.0, max(0.0, t))))

    weight = ImageChops.multiply(vgrad.resize((w, h)), hgrad.resize((w, h)))
    alpha = weight.point(lambda v: round(v * SCRIM_MAX_ALPHA / 255))

    scrim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    scrim.putalpha(alpha)
    return scrim


def main() -> None:
    hero = Image.open(HERO).convert("RGBA")
    canvas = cover(hero, CARD_W, CARD_H)

    canvas.alpha_composite(corner_scrim(CARD_W, CARD_H))

    draw = ImageDraw.Draw(canvas)
    wordmark_font = ImageFont.truetype(FONT_PATH, WORDMARK_SIZE)
    wordmark_font.set_variation_by_name(WORDMARK_VARIATION)
    tagline_font = ImageFont.truetype(FONT_PATH, TAGLINE_SIZE)
    tagline_font.set_variation_by_name(TAGLINE_VARIATION)

    # Right-align both lines: position so each text's right edge sits at
    # CARD_W - PAD_X. Use textlength (advance width) which excludes the
    # left-side bearing that textbbox carries.
    right_edge = CARD_W - PAD_X
    word_w = draw.textlength(WORDMARK, font=wordmark_font)
    tag_w = draw.textlength(TAGLINE, font=tagline_font)
    word_x = round(right_edge - word_w)
    tag_x = round(right_edge - tag_w)

    # Bottom-anchor: tagline baseline sits PAD_BOTTOM above the canvas edge.
    tag_ascent, tag_descent = tagline_font.getmetrics()
    word_ascent, word_descent = wordmark_font.getmetrics()
    tag_y = CARD_H - PAD_BOTTOM - (tag_ascent + tag_descent)
    word_y = tag_y - WORDMARK_TAGLINE_GAP - (word_ascent + word_descent)

    draw.text(
        (word_x, word_y), WORDMARK, font=wordmark_font, fill=WORDMARK_FILL,
        stroke_width=WORDMARK_STROKE, stroke_fill=STROKE_FILL,
    )
    draw.text(
        (tag_x, tag_y), TAGLINE, font=tagline_font, fill=TAGLINE_FILL,
        stroke_width=TAGLINE_STROKE, stroke_fill=STROKE_FILL,
    )

    canvas.convert("RGB").save(OUT, "JPEG", quality=85, optimize=True, progressive=True)
    print(f"wrote {OUT} ({CARD_W}x{CARD_H}, {OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
