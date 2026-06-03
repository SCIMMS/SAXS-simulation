#!/usr/bin/env python
"""Retrieve GroEL model/orientation from experimental assembled hit patterns.

Expected future input is an assembled-hit HDF5 file similar to the public GroEL
analysis pipeline output:

  Results/AssembledHits/hits_r0151.h5
    patterns_ph: (n_hits, 1024, 1024) photon counts
    mask:        (1024, 1024) valid-pixel mask

The script also accepts already-downsampled 128 x 128 patterns. Matching follows
the Ekeberg-style template fit: downsample to 128 x 128, apply an annular mask,
search small center shifts, and fit signal/background coefficients for each
template.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np


def parse_indices(value: str | None) -> List[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def downsample_sum_2d(image: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.shape == (128, 128):
        return arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D image, got shape {arr.shape}")
    if arr.shape[0] % factor != 0 or arr.shape[1] % factor != 0:
        raise ValueError(f"image shape {arr.shape} is not divisible by downsample factor {factor}")
    return arr.reshape(arr.shape[0] // factor, factor, arr.shape[1] // factor, factor).sum(axis=(1, 3)).astype(np.float32)


def downsample_mask_2d(mask: np.ndarray, factor: int) -> np.ndarray:
    arr = np.asarray(mask).astype(bool)
    if arr.shape == (128, 128):
        return arr
    if arr.ndim != 2:
        raise ValueError(f"expected 2D mask, got shape {arr.shape}")
    if arr.shape[0] % factor != 0 or arr.shape[1] % factor != 0:
        raise ValueError(f"mask shape {arr.shape} is not divisible by downsample factor {factor}")
    return arr.reshape(arr.shape[0] // factor, factor, arr.shape[1] // factor, factor).sum(axis=(1, 3)) == factor * factor


def build_annular_mask(shape: Tuple[int, int], inner_radius_px: float, outer_radius_px: float) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    center_y = (shape[0] - 1.0) * 0.5
    center_x = (shape[1] - 1.0) * 0.5
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return (r >= float(inner_radius_px)) & (r <= float(outer_radius_px))


def choose_top_pattern_indices(dataset: h5py.Dataset, count: int, downsample: int) -> List[int]:
    totals = []
    for idx in range(dataset.shape[0]):
        image = downsample_sum_2d(dataset[idx], downsample)
        totals.append(float(np.nansum(np.clip(image, 0.0, None))))
    return np.argsort(np.asarray(totals))[::-1][: int(count)].astype(int).tolist()


def choose_top_pattern_indices_filtered(
    dataset: h5py.Dataset,
    count: int,
    downsample: int,
    allowed_pattern_indices: Sequence[int] | None,
) -> List[int]:
    if allowed_pattern_indices is None:
        return choose_top_pattern_indices(dataset, count, downsample)
    totals = []
    indices = [int(v) for v in allowed_pattern_indices]
    for idx in indices:
        image = downsample_sum_2d(dataset[idx], downsample)
        totals.append(float(np.nansum(np.clip(image, 0.0, None))))
    order = np.argsort(np.asarray(totals))[::-1][: int(count)]
    return [indices[int(i)] for i in order]


def load_mask_from_pattern_file(
    handle: h5py.File,
    mask_dataset: str | None,
    image_shape: Tuple[int, int],
    downsample: int,
) -> np.ndarray:
    if mask_dataset and mask_dataset in handle:
        return downsample_mask_2d(handle[mask_dataset][...], downsample)
    return np.ones(image_shape, dtype=bool)


def load_background(
    background_file: Path | None,
    background_dataset: str,
    image_shape: Tuple[int, int],
    downsample: int,
) -> np.ndarray:
    if background_file is None:
        return np.ones(image_shape, dtype=np.float32)
    with h5py.File(background_file, "r") as h5:
        if background_dataset not in h5:
            raise KeyError(f"background dataset not found: {background_dataset}")
        bg = downsample_sum_2d(h5[background_dataset][...], downsample)
    return np.clip(bg.astype(np.float32), 0.0, None)


def nonnegative_two_parameter_fit(
    y: np.ndarray,
    templates: np.ndarray,
    background: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit y ~= signal * template + bg_scale * background for all templates."""
    y64 = y.astype(np.float64, copy=False)
    t64 = templates.astype(np.float64, copy=False)
    b64 = background.astype(np.float64, copy=False)

    yy = float(np.dot(y64, y64))
    tb = t64 @ b64
    tt = np.einsum("ij,ij->i", t64, t64)
    ty = t64 @ y64
    bb = float(np.dot(b64, b64))
    by = float(np.dot(b64, y64))

    denom = tt * bb - tb * tb
    signal = np.divide(ty * bb - by * tb, denom, out=np.zeros_like(ty), where=np.abs(denom) > 1e-12)
    bg_scale = np.divide(tt * by - tb * ty, denom, out=np.zeros_like(ty), where=np.abs(denom) > 1e-12)
    sse = yy - 2.0 * signal * ty - 2.0 * bg_scale * by + signal * signal * tt + 2.0 * signal * bg_scale * tb + bg_scale * bg_scale * bb

    signal_only = np.divide(ty, tt, out=np.zeros_like(ty), where=tt > 1e-12)
    signal_only = np.maximum(signal_only, 0.0)
    sse_signal_only = yy - 2.0 * signal_only * ty + signal_only * signal_only * tt

    bg_only = by / bb if bb > 1e-12 else 0.0
    bg_only = max(float(bg_only), 0.0)
    sse_bg_only = yy - 2.0 * bg_only * by + bg_only * bg_only * bb

    bad = (signal < 0.0) | (bg_scale < 0.0) | ~np.isfinite(sse)
    use_signal_only = bad & (sse_signal_only <= sse_bg_only)
    use_bg_only = bad & ~use_signal_only
    signal = signal.copy()
    bg_scale = bg_scale.copy()
    sse = sse.copy()
    signal[use_signal_only] = signal_only[use_signal_only]
    bg_scale[use_signal_only] = 0.0
    sse[use_signal_only] = sse_signal_only[use_signal_only]
    signal[use_bg_only] = 0.0
    bg_scale[use_bg_only] = bg_only
    sse[use_bg_only] = sse_bg_only
    return signal.astype(np.float64), bg_scale.astype(np.float64), np.maximum(sse, 0.0).astype(np.float64)


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std())
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def logcorr_score(y: np.ndarray, fitted: np.ndarray) -> float:
    return float(np.mean(zscore(np.log1p(np.clip(y, 0.0, None))) * zscore(np.log1p(np.clip(fitted, 0.0, None)))))


def iter_template_files(values: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(value))
    missing = [str(p) for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError("missing template file(s): " + ", ".join(missing))
    return out


def scan_template_file(
    template_file: Path,
    pattern: np.ndarray,
    valid_mask: np.ndarray,
    annular_mask: np.ndarray,
    background: np.ndarray,
    shift_radius: int,
    top_k: int,
) -> List[Dict[str, object]]:
    hits: List[Dict[str, object]] = []
    with h5py.File(template_file, "r") as h5:
        templates = h5["templates"][...].astype(np.float32)
        rotations = h5["rotations"][...].astype(np.float32)
        rotation_indices = h5["rotation_indices"][...].astype(np.int64)
        model_names = [decode_h5_string(x) for x in h5["model_names"][...]]
        template_annular = h5["annular_mask"][...].astype(bool) if "annular_mask" in h5 else annular_mask

    base_roi = annular_mask & template_annular
    for shift_y in range(-int(shift_radius), int(shift_radius) + 1):
        for shift_x in range(-int(shift_radius), int(shift_radius) + 1):
            shifted_pattern = np.roll(pattern, (shift_y, shift_x), axis=(0, 1))
            shifted_valid = np.roll(valid_mask, (shift_y, shift_x), axis=(0, 1))
            shifted_background = np.roll(background, (shift_y, shift_x), axis=(0, 1))
            roi = base_roi & shifted_valid
            if int(roi.sum()) < 20:
                continue
            y = np.clip(shifted_pattern[roi].astype(np.float32), 0.0, None)
            b = np.clip(shifted_background[roi].astype(np.float32), 0.0, None)
            t = np.clip(templates[:, roi].astype(np.float32), 0.0, None)
            signal, bg_scale, sse = nonnegative_two_parameter_fit(y, t, b)
            rms = np.sqrt(sse / float(y.size))
            order = np.argsort(rms)[: max(int(top_k), 1)]
            for idx in order:
                fitted = signal[idx] * t[idx] + bg_scale[idx] * b
                hits.append(
                    {
                        "template_file": str(template_file),
                        "template_index": int(idx),
                        "model_name": model_names[int(idx)],
                        "rotation_index": int(rotation_indices[int(idx)]),
                        "qw": float(rotations[int(idx), 0]),
                        "qx": float(rotations[int(idx), 1]),
                        "qy": float(rotations[int(idx), 2]),
                        "qz": float(rotations[int(idx), 3]),
                        "shift_y": int(shift_y),
                        "shift_x": int(shift_x),
                        "signal": float(signal[int(idx)]),
                        "background_scale": float(bg_scale[int(idx)]),
                        "fit_rms": float(rms[int(idx)]),
                        "logcorr": logcorr_score(y, fitted),
                        "roi_pixels": int(y.size),
                    }
                )
    hits.sort(key=lambda row: (float(row["fit_rms"]), -float(row["logcorr"])))
    return hits[: int(top_k)]


def load_template_by_row(row: Dict[str, object]) -> np.ndarray:
    with h5py.File(str(row["template_file"]), "r") as h5:
        return h5["templates"][int(row["template_index"])].astype(np.float32)


def get_optional_indexed_value(handle: h5py.File, dataset_name: str, index: int) -> int | None:
    if dataset_name not in handle:
        return None
    dataset = handle[dataset_name]
    if dataset.ndim != 1 or int(index) >= int(dataset.shape[0]):
        return None
    return int(dataset[int(index)])


def render_log(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.log1p(np.clip(np.asarray(image, dtype=np.float64), 0.0, None))
    out[~valid_mask] = np.nan
    return out


def save_comparison_figure(
    pattern: np.ndarray,
    template: np.ndarray,
    background: np.ndarray,
    row: Dict[str, object],
    valid_mask: np.ndarray,
    annular_mask: np.ndarray,
    out_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(out_path.parent / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shift_y = int(row["shift_y"])
    shift_x = int(row["shift_x"])
    shifted_pattern = np.roll(pattern, (shift_y, shift_x), axis=(0, 1))
    shifted_valid = np.roll(valid_mask, (shift_y, shift_x), axis=(0, 1))
    shifted_background = np.roll(background, (shift_y, shift_x), axis=(0, 1))
    fitted = float(row["signal"]) * template + float(row["background_scale"]) * shifted_background
    valid = shifted_valid & annular_mask

    exp_log = render_log(shifted_pattern, shifted_valid)
    fit_log = render_log(fitted, shifted_valid)
    combined = np.concatenate((exp_log[np.isfinite(exp_log)], fit_log[np.isfinite(fit_log)]))
    vmin, vmax = np.nanpercentile(combined, [1.0, 99.7])
    residual = np.full_like(exp_log, np.nan, dtype=np.float64)
    residual[valid] = exp_log[valid] - fit_log[valid]
    rlim = max(float(np.nanpercentile(np.abs(residual[valid]), 97.5)), 1e-6)
    split = pattern.shape[1] // 2
    half = fit_log.copy()
    half[:, :split] = exp_log[:, :split]

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), dpi=170, constrained_layout=True)
    intensity_cmap = plt.get_cmap("viridis").copy()
    intensity_cmap.set_bad("black")
    residual_cmap = plt.get_cmap("coolwarm").copy()
    residual_cmap.set_bad("white")
    ims = [
        axes[0].imshow(exp_log, origin="lower", cmap=intensity_cmap, vmin=vmin, vmax=vmax),
        axes[1].imshow(fit_log, origin="lower", cmap=intensity_cmap, vmin=vmin, vmax=vmax),
        axes[2].imshow(half, origin="lower", cmap=intensity_cmap, vmin=vmin, vmax=vmax),
        axes[3].imshow(residual, origin="lower", cmap=residual_cmap, vmin=-rlim, vmax=rlim),
    ]
    titles = ["experimental", "fit", "left exp / right fit", "log residual"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[2].axvline(split - 0.5, color="white", lw=0.8)
    fig.suptitle(
        f"{row['model_name']} rot={row['rotation_index']} shift=({shift_y},{shift_x}) "
        f"rms={float(row['fit_rms']):.4g} corr={float(row['logcorr']):.3f}",
        fontsize=9,
    )
    fig.colorbar(ims[2], ax=axes[:3], shrink=0.80, label="log1p intensity")
    fig.colorbar(ims[3], ax=axes[3], shrink=0.80, label="delta log1p")
    fig.savefig(out_path)
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern-file", type=Path, required=True)
    parser.add_argument("--pattern-dataset", default="patterns_ph")
    parser.add_argument("--mask-dataset", default="mask")
    parser.add_argument("--background-file", type=Path, default=None)
    parser.add_argument("--background-dataset", default="photon_background")
    parser.add_argument("--template-files", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("validation") / "groel_retrieval_results")
    parser.add_argument("--pattern-indices", default=None)
    parser.add_argument("--exclude-source-frame-indices", default="")
    parser.add_argument("--top-pattern-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-comparison-png", type=int, default=3)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--inner-radius-px", type=float, default=5.0)
    parser.add_argument("--outer-radius-px", type=float, default=40.0)
    parser.add_argument("--shift-radius", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template_files = iter_template_files(args.template_files)

    all_top_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    with h5py.File(args.pattern_file, "r") as pattern_h5:
        if args.pattern_dataset not in pattern_h5:
            raise KeyError(f"pattern dataset not found: {args.pattern_dataset}")
        dataset = pattern_h5[args.pattern_dataset]
        if dataset.ndim == 2:
            pattern_indices = [0]
        else:
            excluded_source_frames = set(parse_indices(args.exclude_source_frame_indices) or [])
            allowed_pattern_indices = None
            if excluded_source_frames and "selected_frame_indices" in pattern_h5:
                selected_frames = pattern_h5["selected_frame_indices"][...].astype(np.int64)
                allowed_pattern_indices = [
                    idx for idx, frame in enumerate(selected_frames.tolist()) if int(frame) not in excluded_source_frames
                ]
            pattern_indices = parse_indices(args.pattern_indices)
            if pattern_indices is None:
                pattern_indices = choose_top_pattern_indices_filtered(
                    dataset,
                    args.top_pattern_count,
                    args.downsample,
                    allowed_pattern_indices,
                )
            elif excluded_source_frames and "selected_frame_indices" in pattern_h5:
                selected_frames = pattern_h5["selected_frame_indices"][...].astype(np.int64)
                pattern_indices = [
                    int(idx) for idx in pattern_indices if int(selected_frames[int(idx)]) not in excluded_source_frames
                ]

        first_image = dataset[...] if dataset.ndim == 2 else dataset[int(pattern_indices[0])]
        image_shape = downsample_sum_2d(first_image, args.downsample).shape
        valid_mask = load_mask_from_pattern_file(pattern_h5, args.mask_dataset, image_shape, args.downsample)

        background = load_background(args.background_file, args.background_dataset, image_shape, args.downsample)
        annular_mask = build_annular_mask(image_shape, args.inner_radius_px, args.outer_radius_px)

        for output_index, pattern_index in enumerate(pattern_indices):
            raw = dataset[...] if dataset.ndim == 2 else dataset[int(pattern_index)]
            pattern = downsample_sum_2d(raw, args.downsample)
            pattern_rows: List[Dict[str, object]] = []
            for template_file in template_files:
                pattern_rows.extend(
                    scan_template_file(
                        template_file=template_file,
                        pattern=pattern,
                        valid_mask=valid_mask,
                        annular_mask=annular_mask,
                        background=background,
                        shift_radius=args.shift_radius,
                        top_k=args.top_k,
                    )
                )
            pattern_rows.sort(key=lambda row: (float(row["fit_rms"]), -float(row["logcorr"])))
            pattern_rows = pattern_rows[: args.top_k]
            for rank, row in enumerate(pattern_rows, start=1):
                row["pattern_index"] = int(pattern_index)
                source_frame = get_optional_indexed_value(pattern_h5, "selected_frame_indices", int(pattern_index))
                lit_pixels = get_optional_indexed_value(pattern_h5, "selected_lit_pixels", int(pattern_index))
                if source_frame is not None:
                    row["source_frame_index"] = source_frame
                if lit_pixels is not None:
                    row["selected_lit_pixels"] = lit_pixels
                row["rank"] = int(rank)
                all_top_rows.append(row)
            if pattern_rows:
                best_rows.append(pattern_rows[0])
                if output_index < args.max_comparison_png:
                    best = pattern_rows[0]
                    template = load_template_by_row(best)
                    out_png = args.output_dir / f"groel_retrieval_pattern_{int(pattern_index):04d}_best.png"
                    save_comparison_figure(pattern, template, background, best, valid_mask, annular_mask, out_png)

    write_csv(args.output_dir / "groel_retrieval_topk.csv", all_top_rows)
    write_csv(args.output_dir / "groel_retrieval_best.csv", best_rows)
    summary = {
        "pattern_file": str(args.pattern_file),
        "pattern_dataset": args.pattern_dataset,
        "template_files": [str(p) for p in template_files],
        "output_dir": str(args.output_dir),
        "pattern_indices": [int(row["pattern_index"]) for row in best_rows],
        "top_k": int(args.top_k),
        "shift_radius": int(args.shift_radius),
        "roi": {
            "inner_radius_px": float(args.inner_radius_px),
            "outer_radius_px": float(args.outer_radius_px),
        },
        "score": "primary ranking is nonnegative least-squares RMS over shifted annular ROI; logcorr is reported for diagnostics",
        "best": best_rows,
    }
    if best_rows and "source_frame_index" in best_rows[0]:
        summary["source_frame_indices"] = [int(row["source_frame_index"]) for row in best_rows]
    (args.output_dir / "groel_retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
