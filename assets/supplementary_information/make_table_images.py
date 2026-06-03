from __future__ import annotations

import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "calibri.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text).split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        test = " ".join(current + [word])
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            if draw.textbbox((0, 0), word, font=font)[2] <= max_width:
                current = [word]
            else:
                lines.extend(textwrap.wrap(word, width=18))
                current = []
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def render_table(csv_path: Path, out_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    cols = ["quantity", "value", "note"]
    widths = [620, 1320, 760]
    margin = 80
    pad_x = 22
    pad_y = 16
    header_font = load_font(31, bold=True)
    body_font = load_font(29)
    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)

    wrapped_rows = []
    row_heights = []
    for row in rows:
        wrapped = [wrap_text(draw, row[col], body_font, widths[i] - 2 * pad_x) for i, col in enumerate(cols)]
        max_lines = max(len(part) for part in wrapped)
        wrapped_rows.append(wrapped)
        row_heights.append(max(62, max_lines * 35 + 2 * pad_y))

    header_h = 70
    table_w = sum(widths)
    image_w = table_w + 2 * margin
    image_h = margin + header_h + sum(row_heights) + margin
    image = Image.new("RGB", (image_w, image_h), "white")
    draw = ImageDraw.Draw(image)

    x0 = margin
    y = margin
    header_fill = "#2f5f6b"
    grid = "#c7ced1"
    x = x0
    for col, width in zip(cols, widths):
        draw.rectangle([x, y, x + width, y + header_h], fill=header_fill, outline=grid, width=2)
        draw.text((x + pad_x, y + 18), col, fill="white", font=header_font)
        x += width
    y += header_h

    for row_index, (wrapped, height) in enumerate(zip(wrapped_rows, row_heights)):
        fill = "#f5f7f8" if row_index % 2 == 0 else "white"
        x = x0
        for col_index, width in enumerate(widths):
            draw.rectangle([x, y, x + width, y + height], fill=fill, outline=grid, width=2)
            ty = y + pad_y
            for line in wrapped[col_index]:
                draw.text((x + pad_x, ty), line, fill="#1c1c1c", font=body_font)
                ty += 35
            x += width
        y += height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    render_table(
        ROOT / "table_S01_groel_parameters" / "data" / "table_S01_groel_parameters.csv",
        ROOT / "table_S01_groel_parameters" / "images" / "table_S01_groel_parameters.png",
    )
    render_table(
        ROOT / "table_S02_id57_parameters" / "data" / "table_S02_id57_parameters.csv",
        ROOT / "table_S02_id57_parameters" / "images" / "table_S02_id57_parameters.png",
    )


if __name__ == "__main__":
    main()
