"""Compose a query + results contact sheet PNG."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .index import IMAGES

_TH = 240


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _tile(path: str, caption: str, sub: str = "") -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((_TH, _TH))
    pad_b = 46 if sub else 28
    canvas = Image.new("RGB", (_TH, _TH + pad_b), "white")
    canvas.paste(im, ((_TH - im.width) // 2, (_TH - im.height) // 2))
    d = ImageDraw.Draw(canvas)
    d.text((6, _TH + 4), caption, fill="black", font=_font(15))
    if sub:
        d.text((6, _TH + 24), sub, fill=(110, 110, 110), font=_font(12))
    return canvas


def contact_sheet(result: dict, out_path: str | Path) -> Path:
    q = result["query"]
    q_img = IMAGES / q["image_file"]
    tiles = [_tile(str(q_img), f"QUERY  {q['id']}")]
    for r in result["results"]:
        b = r["breakdown"]
        tiles.append(_tile(
            str(IMAGES / r["image_file"]),
            f"#{r['rank']}  {r['id']}  {r['score']:.3f}",
            f"clip {b['clip']:.2f} | col {b['colour']:.2f} | att {b['attr']:.2f}",
        ))
    gap = 12
    W = sum(t.width for t in tiles) + gap * (len(tiles) + 1)
    H = max(t.height for t in tiles) + gap * 2
    sheet = Image.new("RGB", (W, H), (245, 245, 245))
    x = gap
    for t in tiles:
        sheet.paste(t, (x, gap))
        x += t.width + gap
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
