from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "validation_supporting_package" / "section5_groel" / "polyprism_preview" / "groel_model_ab_preview_montage.png"
OUT_DIR = Path(__file__).resolve().parent
IMG_DIR = OUT_DIR / "images"
DATA_DIR = OUT_DIR / "data"


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    src = Image.open(SRC).convert("RGB")
    width, height = src.size
    n_cols = 16
    tile_w = width // n_cols
    selected = [0, 4, 8, 12]
    labels = ["(a) Model A", "(b) Model B", "(c) R35 variant", "(d) R55 variant"]

    crops = [src.crop((idx * tile_w, 0, (idx + 1) * tile_w, height)) for idx in selected]
    scale = 2
    crops = [c.resize((c.width * scale, c.height * scale), Image.Resampling.LANCZOS) for c in crops]
    gap = 24
    top = 8
    label_h = 30
    out_w = sum(c.width for c in crops) + gap * (len(crops) - 1)
    out_h = top + label_h + max(c.height for c in crops)
    out = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    x = 0
    for crop, label in zip(crops, labels):
        lw = draw.textbbox((0, 0), label, font=font)[2]
        draw.text((x + (crop.width - lw) / 2, top), label, fill="black", font=font)
        out.paste(crop, (x, top + label_h))
        x += crop.width + gap

    out.save(IMG_DIR / "figure_S03_groel_model_variants_and_preview_patterns.png")


if __name__ == "__main__":
    main()
