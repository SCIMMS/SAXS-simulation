"""Initial CXIDB ID 57 data probe for Section 5 public validation.

This script reads the normal-incidence CXI file with h5py only, extracts
metadata needed for detector-matched simulation, and writes small diagnostic
artifacts for model-planning.
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


HC_J_M = 6.62607015e-34 * 299792458.0
DATA_LINK_PATH = "/entry_1/data_1/data"
DETECTOR_DATA_PATH = "/entry_1/instrument_1/detector_1/data"


def to_python(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_scalar(handle: h5py.File, path: str) -> Any:
    if path not in handle:
        return None
    return to_python(handle[path][()])


def read_pixel_size_from_geom(geom_path: Path) -> float | None:
    if not geom_path.exists():
        return None

    for line in geom_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key == "res" or key.endswith("/res"):
            try:
                pixels_per_meter = float(raw_value.strip())
                return 1.0 / pixels_per_meter
            except ValueError:
                continue
    return None


def build_q_map(shape: tuple[int, int], pixel_size_m: float, distance_m: float, wavelength_m: float) -> np.ndarray:
    ny, nx = shape
    y, x = np.indices((ny, nx), dtype=np.float64)
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    radius_m = np.hypot((x - cx) * pixel_size_m, (y - cy) * pixel_size_m)
    two_theta = np.arctan2(radius_m, distance_m)
    q_m_inv = (4.0 * math.pi / wavelength_m) * np.sin(0.5 * two_theta)
    return q_m_inv * 1.0e-9


def radial_mean(image: np.ndarray, q_nm_inv: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.clip(image.astype(np.float64), 0.0, None)
    finite = np.isfinite(values) & np.isfinite(q_nm_inv)
    q = q_nm_inv[finite]
    v = values[finite]
    edges = np.linspace(0.0, float(q.max()), bins + 1)
    indices = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, bins - 1)
    sums = np.bincount(indices, weights=v, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    mean = np.divide(sums, counts, out=np.full(bins, np.nan), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, mean, counts


def save_log_image(image: np.ndarray, out_path: Path, title: str) -> None:
    view = np.log1p(np.clip(image.astype(np.float64), 0.0, None))
    vmax = float(np.nanpercentile(view, 99.7))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(view, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.colorbar(im, ax=ax, label="log1p(clipped ADU)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_radial_plot(q: np.ndarray, intensity: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(q, intensity, lw=1.3)
    ax.set_xlabel("q (nm^-1)")
    ax.set_ylabel("mean clipped intensity (ADU)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public_data" / "CXIDB_ID_57",
    )
    parser.add_argument("--cxi-name", default="normal-incidence-pattern.cxi")
    parser.add_argument("--geom-name", default="new.geom")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--radial-bins", type=int, default=220)
    args = parser.parse_args()

    data_dir = args.data_dir
    cxi_path = data_dir / args.cxi_name
    geom_path = data_dir / args.geom_name
    out_dir = args.out_dir or data_dir / "analysis_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_size_m = read_pixel_size_from_geom(geom_path)

    with h5py.File(cxi_path, "r") as handle:
        data_path = DATA_LINK_PATH if DATA_LINK_PATH in handle else DETECTOR_DATA_PATH
        data = handle[data_path]
        stack = data[...]

        energy_j = read_scalar(handle, "/entry_1/instrument_1/source_1/energy")
        wavelength_m = HC_J_M / energy_j if energy_j else None
        distance_m = read_scalar(handle, "/entry_1/instrument_1/detector_1/distance")

        pattern_rows: list[dict[str, Any]] = []
        for index, image in enumerate(stack):
            finite = np.isfinite(image)
            clipped = np.clip(image, 0.0, None)
            pattern_rows.append(
                {
                    "pattern_index": index,
                    "raw_total": float(np.nansum(image)),
                    "positive_total": float(np.nansum(clipped)),
                    "min": float(np.nanmin(image)),
                    "max": float(np.nanmax(image)),
                    "mean": float(np.nanmean(image)),
                    "median": float(np.nanmedian(image)),
                    "std": float(np.nanstd(image)),
                    "finite_pixels": int(finite.sum()),
                    "negative_pixels": int((image < 0.0).sum()),
                }
            )

        best_index = int(max(pattern_rows, key=lambda row: row["positive_total"])["pattern_index"])
        mean_pattern = np.nanmean(stack, axis=0)
        median_pattern = np.nanmedian(stack, axis=0)

        q_summary = None
        if pixel_size_m is not None and distance_m and wavelength_m:
            q_nm_inv = build_q_map(stack.shape[1:], pixel_size_m, float(distance_m), float(wavelength_m))
            q_centers, radial, counts = radial_mean(mean_pattern, q_nm_inv, args.radial_bins)
            radial_rows = [
                {"q_nm_inv": float(q), "mean_intensity": float(i), "pixel_count": int(c)}
                for q, i, c in zip(q_centers, radial, counts)
            ]
            write_csv(out_dir / "id57_mean_radial_profile.csv", radial_rows)
            save_radial_plot(q_centers, radial, out_dir / "id57_mean_radial_profile.png")
            q_summary = {
                "q_min_nm_inv": float(np.nanmin(q_nm_inv)),
                "q_max_nm_inv": float(np.nanmax(q_nm_inv)),
                "q_center_pixel": float(q_nm_inv[stack.shape[1] // 2, stack.shape[2] // 2]),
                "radial_bins": args.radial_bins,
            }

        write_csv(out_dir / "id57_pattern_stats.csv", pattern_rows)
        save_log_image(mean_pattern, out_dir / "id57_mean_pattern_log.png", "CXIDB ID 57 mean pattern")
        save_log_image(median_pattern, out_dir / "id57_median_pattern_log.png", "CXIDB ID 57 median pattern")
        save_log_image(stack[best_index], out_dir / f"id57_pattern_{best_index:04d}_log.png", f"CXIDB ID 57 pattern {best_index}")

        summary = {
            "cxi_path": str(cxi_path),
            "geom_path": str(geom_path),
            "data_path": data_path,
            "data_shape": list(stack.shape),
            "data_dtype": str(stack.dtype),
            "cxi_version": read_scalar(handle, "/cxi_version"),
            "experimental_identifier": read_scalar(handle, "/entry_1/experimental_identifier"),
            "start_time": read_scalar(handle, "/entry_1/start_time"),
            "instrument_name": read_scalar(handle, "/entry_1/instrument_1/name"),
            "sample_name": read_scalar(handle, "/entry_1/sample_1/name"),
            "detector_distance_m": to_python(distance_m),
            "source_energy_j": to_python(energy_j),
            "source_energy_ev": float(energy_j / 1.602176634e-19) if energy_j else None,
            "wavelength_m": wavelength_m,
            "wavelength_nm": float(wavelength_m * 1.0e9) if wavelength_m else None,
            "pulse_width_s": read_scalar(handle, "/entry_1/instrument_1/source_1/pulse_width"),
            "pixel_size_m_from_geom": pixel_size_m,
            "best_pattern_index_by_positive_total": best_index,
            "raw_total_min": min(row["raw_total"] for row in pattern_rows),
            "raw_total_median": float(np.median([row["raw_total"] for row in pattern_rows])),
            "raw_total_max": max(row["raw_total"] for row in pattern_rows),
            "positive_total_min": min(row["positive_total"] for row in pattern_rows),
            "positive_total_median": float(np.median([row["positive_total"] for row in pattern_rows])),
            "positive_total_max": max(row["positive_total"] for row in pattern_rows),
            "negative_pixels_total": int(sum(row["negative_pixels"] for row in pattern_rows)),
            "q_summary": q_summary,
            "outputs": {
                "summary_json": str(out_dir / "id57_cxi_summary.json"),
                "pattern_stats_csv": str(out_dir / "id57_pattern_stats.csv"),
                "mean_pattern_png": str(out_dir / "id57_mean_pattern_log.png"),
                "median_pattern_png": str(out_dir / "id57_median_pattern_log.png"),
                "best_pattern_png": str(out_dir / f"id57_pattern_{best_index:04d}_log.png"),
                "radial_profile_csv": str(out_dir / "id57_mean_radial_profile.csv"),
                "radial_profile_png": str(out_dir / "id57_mean_radial_profile.png"),
            },
        }

    (out_dir / "id57_cxi_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
