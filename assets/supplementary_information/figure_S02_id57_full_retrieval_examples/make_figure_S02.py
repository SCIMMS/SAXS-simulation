from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "validation_supporting_package" / "section5_id57" / "retrieval_real"
OUT_DIR = Path(__file__).resolve().parent
IMG_DIR = OUT_DIR / "images"


def crop_light_border(image: Image.Image, threshold: int = 248, pad: int = 8) -> Image.Image:
    rgb = image.convert("RGB")
    px = rgb.load()
    w, h = rgb.size
    xs = []
    ys = []
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            if r < threshold or g < threshold or b < threshold:
                xs.append(x)
                ys.append(y)
    if not xs:
        return rgb
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(w, max(xs) + 1 + pad)
    y1 = min(h, max(ys) + 1 + pad)
    return rgb.crop((x0, y0, x1, y1))


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in ["arialbd.ttf" if bold else "arial.ttf", "calibri.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(SRC_DIR.glob("retrieval_exp_*_best_*.png"))[:4]
    crops = [crop_light_border(Image.open(path)) for path in sources]
    if not crops:
        raise FileNotFoundError(f"No retrieval images found in {SRC_DIR}")

    target_w = max(c.width for c in crops)
    scaled = []
    for crop in crops:
        if crop.width != target_w:
            h = int(crop.height * target_w / crop.width)
            crop = crop.resize((target_w, h), Image.Resampling.LANCZOS)
        scaled.append(crop)

    gap = 20
    margin = 14
    out_w = target_w + 2 * margin
    out_h = sum(c.height for c in scaled) + gap * (len(scaled) - 1) + 2 * margin
    out = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(out)
    font = load_font(28, bold=True)

    y = margin
    for idx, crop in enumerate(scaled):
        out.paste(crop, (margin, y))
        label = f"({chr(ord('a') + idx)})"
        box = draw.textbbox((0, 0), label, font=font)
        draw.rectangle(
            [margin + 8, y + 8, margin + 18 + box[2] - box[0], y + 18 + box[3] - box[1]],
            fill="white",
        )
        draw.text((margin + 13, y + 11), label, fill="black", font=font)
        y += crop.height + gap

    out.save(IMG_DIR / "figure_S02_id57_full_retrieval_examples.png")


if __name__ == "__main__":
    main()
