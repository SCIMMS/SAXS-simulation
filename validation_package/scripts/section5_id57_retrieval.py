"""Retrieve ID 57 model orientation by matching experimental CXI images.

The retrieval target is the best matching entry in a synthetic core-shell
template bank. The primary score is masked, annular-ROI log-intensity
normalized cross-correlation after per-template scale/background fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

MPL_CACHE_DIR = Path(__file__).resolve().parents[1] / ".runtime_cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DATA_LINK_PATH = "/entry_1/data_1/data"
DETECTOR_DATA_PATH = "/entry_1/instrument_1/detector_1/data"


def parse_indices(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def read_template_parameters(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[int(row["index"])] = row
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_annular_mask(shape: tuple[int, int], inner_radius_px: float, outer_radius_px: float) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    cx = (shape[1] - 1) / 2.0
    cy = (shape[0] - 1) / 2.0
    radius = np.hypot(x - cx, y - cy)
    return (radius >= inner_radius_px) & (radius <= outer_radius_px)


def choose_top_experimental_indices(data: h5py.Dataset, count: int) -> list[int]:
    totals = []
    for index in range(data.shape[0]):
        image = data[index]
        totals.append(float(np.nansum(np.clip(image, 0.0, None))))
    return np.argsort(np.asarray(totals))[::-1][:count].astype(int).tolist()


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1.0e-12:
        return values * 0.0
    return (values - mean) / std


def fit_scale_background_and_score(
    experimental: np.ndarray,
    templates: np.ndarray,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return score, fitted scale, fitted background, fitted log residual RMS."""
    y_raw = np.clip(experimental[roi_mask].astype(np.float64), 0.0, None)
    x_raw = np.clip(templates[:, roi_mask].astype(np.float64), 0.0, None)
    n = float(y_raw.size)

    sum_x = x_raw.sum(axis=1)
    sum_xx = (x_raw * x_raw).sum(axis=1)
    sum_y = float(y_raw.sum())
    sum_xy = x_raw @ y_raw
    denom = n * sum_xx - sum_x * sum_x
    scale = np.divide(n * sum_xy - sum_x * sum_y, denom, out=np.zeros_like(sum_x), where=np.abs(denom) > 1.0e-12)
    background = (sum_y - scale * sum_x) / n
    scale = np.maximum(scale, 0.0)
    background = np.maximum(background, 0.0)

    y_log_z = zscore(np.log1p(y_raw))
    scores = np.empty(templates.shape[0], dtype=np.float64)
    residual_rms = np.empty(templates.shape[0], dtype=np.float64)
    for template_index in range(templates.shape[0]):
        fitted = np.clip(scale[template_index] * x_raw[template_index] + background[template_index], 0.0, None)
        sim_log = np.log1p(fitted)
        sim_log_z = zscore(sim_log)
        scores[template_index] = float(np.mean(y_log_z * sim_log_z))
        residual_rms[template_index] = float(np.sqrt(np.mean((y_log_z - sim_log_z) ** 2)))

    return scores, scale, background, residual_rms


def render_log(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.log1p(np.clip(image.astype(np.float64), 0.0, None))
    out[~valid_mask] = np.nan
    return out


def save_comparison_figure(
    experimental: np.ndarray,
    fitted_simulation: np.ndarray,
    valid_mask: np.ndarray,
    roi_mask: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    exp_log = render_log(experimental, valid_mask)
    sim_log = render_log(fitted_simulation, valid_mask)
    combined_values = np.concatenate([exp_log[np.isfinite(exp_log)], sim_log[np.isfinite(sim_log)]])
    vmin = float(np.nanpercentile(combined_values, 1.0))
    vmax = float(np.nanpercentile(combined_values, 99.7))

    split = experimental.shape[1] // 2
    half = sim_log.copy()
    half[:, :split] = exp_log[:, :split]
    half[~valid_mask] = np.nan

    residual = np.full_like(exp_log, np.nan, dtype=np.float64)
    residual[roi_mask] = exp_log[roi_mask] - sim_log[roi_mask]
    residual_limit = float(np.nanpercentile(np.abs(residual[roi_mask]), 97.5))
    residual_limit = max(residual_limit, 1.0e-6)

    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), dpi=170)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    intensity_cmap = plt.get_cmap("magma").copy()
    intensity_cmap.set_bad("black")
    residual_cmap = plt.get_cmap("coolwarm").copy()
    residual_cmap.set_bad("white")

    images = [
        axes[0].imshow(exp_log, origin="lower", cmap=intensity_cmap, vmin=vmin, vmax=vmax),
        axes[1].imshow(sim_log, origin="lower", cmap=intensity_cmap, vmin=vmin, vmax=vmax),
        axes[2].imshow(half, origin="lower", cmap=intensity_cmap, vmin=vmin, vmax=vmax),
        axes[3].imshow(residual, origin="lower", cmap=residual_cmap, vmin=-residual_limit, vmax=residual_limit),
    ]
    axes[0].set_title("experimental")
    axes[1].set_title("retrieved simulation")
    axes[2].set_title("left exp / right sim")
    axes[3].set_title("ROI log residual")
    axes[2].axvline(split - 0.5, color="white", lw=0.8, alpha=0.85)
    fig.suptitle(title, fontsize=10)
    fig.colorbar(images[2], ax=axes[:3], fraction=0.025, pad=0.02, label="log1p intensity")
    fig.colorbar(images[3], ax=axes[3], fraction=0.046, pad=0.04, label="delta log1p")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public_data" / "CXIDB_ID_57",
    )
    parser.add_argument("--cxi-name", default="normal-incidence-pattern.cxi")
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "public_data"
        / "CXIDB_ID_57"
        / "synthetic_id57_core_shell_v3_uniform_n0120",
    )
    parser.add_argument("--patterns-name", default="synthetic_id57_patterns.npz")
    parser.add_argument("--parameters-name", default="synthetic_id57_parameters.csv")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--experimental-indices", default=None)
    parser.add_argument("--top-experimental-count", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--inner-radius-px", type=float, default=45.0)
    parser.add_argument("--outer-radius-px", type=float, default=190.0)
    parser.add_argument("--template-kind", choices=("clean", "noisy"), default="clean")
    parser.add_argument("--max-comparison-png", type=int, default=None)
    args = parser.parse_args()

    cxi_path = args.data_dir / args.cxi_name
    synthetic_dir = args.synthetic_dir
    out_dir = args.out_dir or args.data_dir / "retrieval_v1_uniform_n0120"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(synthetic_dir / args.patterns_name) as synthetic:
        key = "clean_patterns" if args.template_kind == "clean" else "noisy_patterns"
        templates = synthetic[key].astype(np.float32)
        valid_mask = synthetic["valid_mask"].astype(bool)
        rotations = synthetic["rotation_matrices"].astype(np.float32) if "rotation_matrices" in synthetic else None
        quaternions = synthetic["quaternions_wxyz"].astype(np.float32) if "quaternions_wxyz" in synthetic else None

    annular_mask = build_annular_mask(valid_mask.shape, args.inner_radius_px, args.outer_radius_px)
    roi_mask = valid_mask & annular_mask
    template_parameters = read_template_parameters(synthetic_dir / args.parameters_name)

    top_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    with h5py.File(cxi_path, "r") as handle:
        data_path = DATA_LINK_PATH if DATA_LINK_PATH in handle else DETECTOR_DATA_PATH
        data = handle[data_path]
        experimental_indices = parse_indices(args.experimental_indices)
        if experimental_indices is None:
            experimental_indices = choose_top_experimental_indices(data, args.top_experimental_count)

        png_limit = len(experimental_indices) if args.max_comparison_png is None else max(args.max_comparison_png, 0)
        for output_index, experimental_index in enumerate(experimental_indices):
            experimental = data[experimental_index].astype(np.float32)
            scores, scales, backgrounds, residual_rms = fit_scale_background_and_score(
                experimental=experimental,
                templates=templates,
                roi_mask=roi_mask,
            )
            order = np.argsort(scores)[::-1][: args.top_k]
            best_template = int(order[0])
            fitted_best = np.clip(scales[best_template] * templates[best_template] + backgrounds[best_template], 0.0, None)
            fitted_best[~valid_mask] = 0.0

            for rank, template_index_raw in enumerate(order, start=1):
                template_index = int(template_index_raw)
                q = quaternions[template_index] if quaternions is not None else [math.nan] * 4
                row: dict[str, Any] = {
                    "experimental_index": experimental_index,
                    "rank": rank,
                    "template_index": template_index,
                    "score_log_corr": float(scores[template_index]),
                    "residual_log_z_rms": float(residual_rms[template_index]),
                    "scale": float(scales[template_index]),
                    "background": float(backgrounds[template_index]),
                    "qw": float(q[0]),
                    "qx": float(q[1]),
                    "qy": float(q[2]),
                    "qz": float(q[3]),
                }
                for key, value in template_parameters.get(template_index, {}).items():
                    if key not in row:
                        row[key] = value
                top_rows.append(row)
                if rank == 1:
                    best_rows.append(row)

            if output_index < png_limit:
                comparison_path = out_dir / f"retrieval_exp_{experimental_index:04d}_best_{best_template:04d}.png"
                title = (
                    f"exp {experimental_index}, template {best_template}, "
                    f"score={scores[best_template]:.4f}, rms={residual_rms[best_template]:.3f}"
                )
                save_comparison_figure(experimental, fitted_best, valid_mask, roi_mask, comparison_path, title)

    write_csv(out_dir / "retrieval_topk.csv", top_rows)
    write_csv(out_dir / "retrieval_best.csv", best_rows)
    summary = {
        "cxi_path": str(cxi_path),
        "synthetic_dir": str(synthetic_dir),
        "out_dir": str(out_dir),
        "template_kind": args.template_kind,
        "template_count": int(templates.shape[0]),
        "experimental_indices": [int(row["experimental_index"]) for row in best_rows],
        "top_k": args.top_k,
        "max_comparison_png": args.max_comparison_png,
        "roi": {
            "inner_radius_px": args.inner_radius_px,
            "outer_radius_px": args.outer_radius_px,
            "valid_pixels": int(valid_mask.sum()),
            "roi_pixels": int(roi_mask.sum()),
        },
        "score": "masked annular ROI normalized cross-correlation of log1p intensities after linear scale/background fit",
        "best": best_rows,
    }
    (out_dir / "retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
