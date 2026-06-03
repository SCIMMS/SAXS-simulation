#!/usr/bin/env python
"""Generate quick GroEL Model A/B diffraction-preview images.

This is a CPU preview, not a replacement for the final GPU signed-raster/CZT
pipeline. It reuses the WK electron-density grid, projects selected orientations
from the Ekeberg GroEL quaternion library, and samples the 2D FFT on the
128 x 128 detector grid used by the public GroEL analysis code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np


HC_EV_ANGSTROM = 12398.419843320026


@dataclass(frozen=True)
class DensityPoints:
    density: np.ndarray
    origin_A: np.ndarray
    coords_A: np.ndarray
    charge_e: np.ndarray
    retained_charge_e: float
    total_charge_e: float
    spacing_A: float
    density_max: float


def next_even(value: int) -> int:
    return int(value + (value % 2))


def load_density_points(npz_path: Path, min_fraction: float) -> DensityPoints:
    data = np.load(npz_path)
    density = np.asarray(data["density"], dtype=np.float32)
    origin_A = np.asarray(data["origin_A"], dtype=np.float64)
    spacing_A = float(np.asarray(data["spacing_A"]).item())

    cutoff = float(density.max()) * float(min_fraction)
    keep = density > cutoff
    i, j, k = np.nonzero(keep)
    coords = np.column_stack(
        (
            origin_A[0] + i.astype(np.float64) * spacing_A,
            origin_A[1] + j.astype(np.float64) * spacing_A,
            origin_A[2] + k.astype(np.float64) * spacing_A,
        )
    ).astype(np.float32)
    charge = (density[keep].astype(np.float64) * spacing_A**3).astype(np.float32)
    return DensityPoints(
        density=density,
        origin_A=origin_A.astype(np.float32),
        coords_A=coords,
        charge_e=charge,
        retained_charge_e=float(charge.sum(dtype=np.float64)),
        total_charge_e=float(density.sum(dtype=np.float64) * spacing_A**3),
        spacing_A=spacing_A,
        density_max=float(density.max()),
    )


def weighted_center(coords: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(coords.astype(np.float64), axis=0, weights=weights.astype(np.float64)).astype(np.float32)


def make_model_points(points: DensityPoints, model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = points.coords_A
    weights = points.charge_e
    if model == "A_heptamer_C7":
        out_coords = coords.copy()
        out_weights = weights.copy()
    elif model == "B_tetradecamer_D7":
        copy_coords = coords.copy()
        copy_coords[:, 1] *= -1.0
        copy_coords[:, 2] *= -1.0
        out_coords = np.vstack((coords, copy_coords)).astype(np.float32)
        out_weights = np.concatenate((weights, weights)).astype(np.float32)
    else:
        raise ValueError(f"unknown model: {model}")

    center = weighted_center(out_coords, out_weights)
    out_coords -= center[np.newaxis, :]
    return out_coords, out_weights, center


def regular_polygon_mask(y: np.ndarray, z: np.ndarray, radius_A: float, sides: int) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(int(sides), dtype=np.float64) / float(sides) + np.pi / float(sides)
    vertices = np.column_stack((float(radius_A) * np.cos(angles), float(radius_A) * np.sin(angles)))
    mask = np.ones(y.shape, dtype=bool)
    for idx in range(int(sides)):
        p0 = vertices[idx]
        p1 = vertices[(idx + 1) % int(sides)]
        edge = p1 - p0
        cross = edge[0] * (z - p0[1]) - edge[1] * (y - p0[0])
        mask &= cross >= -1e-7
    return mask


def make_polygonal_prism_points(
    radius_A: float,
    length_A: float,
    sides: int,
    spacing_A: float,
    density_e_A3: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x_vals = np.arange(-0.5 * float(length_A) + 0.5 * spacing_A, 0.5 * float(length_A), float(spacing_A))
    yz_vals = np.arange(-float(radius_A) + 0.5 * spacing_A, float(radius_A), float(spacing_A))
    y, z = np.meshgrid(yz_vals, yz_vals, indexing="ij")
    keep_yz = regular_polygon_mask(y, z, radius_A=float(radius_A), sides=int(sides))
    yz = np.column_stack((y[keep_yz], z[keep_yz]))
    coords = np.empty((len(x_vals) * len(yz), 3), dtype=np.float32)
    for idx, x_val in enumerate(x_vals):
        start = idx * len(yz)
        stop = start + len(yz)
        coords[start:stop, 0] = float(x_val)
        coords[start:stop, 1:] = yz
    charge = np.full(coords.shape[0], float(density_e_A3) * float(spacing_A) ** 3, dtype=np.float32)
    return coords, charge


def sample_density_nearest(points: DensityPoints, coords_abs_A: np.ndarray) -> np.ndarray:
    idx = np.rint((coords_abs_A.astype(np.float64) - points.origin_A.astype(np.float64)) / points.spacing_A).astype(np.int64)
    shape = np.asarray(points.density.shape, dtype=np.int64)
    valid = np.all((idx >= 0) & (idx < shape[np.newaxis, :]), axis=1)
    out = np.zeros(coords_abs_A.shape[0], dtype=np.float32)
    good = idx[valid]
    out[valid] = points.density[good[:, 0], good[:, 1], good[:, 2]]
    return out


def exclude_prism_protein_overlap(
    points: DensityPoints,
    prism_coords_centered_A: np.ndarray,
    model_b_center_A: np.ndarray,
    exclude_fraction: float,
) -> np.ndarray:
    if float(exclude_fraction) <= 0.0:
        return np.ones(prism_coords_centered_A.shape[0], dtype=bool)
    coords_abs = prism_coords_centered_A.astype(np.float32) + model_b_center_A[np.newaxis, :].astype(np.float32)
    density_a = sample_density_nearest(points, coords_abs)
    copy_query = coords_abs.copy()
    copy_query[:, 1] *= -1.0
    copy_query[:, 2] *= -1.0
    density_b = sample_density_nearest(points, copy_query)
    density_model_b = np.maximum(density_a, density_b)
    return density_model_b < float(exclude_fraction) * points.density_max


def add_polygonal_prism_to_model_b(
    points: DensityPoints,
    protein_coords: np.ndarray,
    protein_weights: np.ndarray,
    model_b_center_A: np.ndarray,
    radius_A: float,
    length_A: float,
    sides: int,
    spacing_A: float,
    density_e_A3: float,
    exclude_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | int]]:
    prism_coords, prism_weights = make_polygonal_prism_points(
        radius_A=radius_A,
        length_A=length_A,
        sides=sides,
        spacing_A=spacing_A,
        density_e_A3=density_e_A3,
    )
    keep = exclude_prism_protein_overlap(
        points=points,
        prism_coords_centered_A=prism_coords,
        model_b_center_A=model_b_center_A,
        exclude_fraction=exclude_fraction,
    )
    prism_coords = prism_coords[keep]
    prism_weights = prism_weights[keep]
    coords = np.vstack((protein_coords, prism_coords)).astype(np.float32)
    weights = np.concatenate((protein_weights, prism_weights)).astype(np.float32)
    meta = {
        "radius_A": float(radius_A),
        "length_A": float(length_A),
        "sides": int(sides),
        "spacing_A": float(spacing_A),
        "density_e_A3": float(density_e_A3),
        "exclude_fraction": float(exclude_fraction),
        "candidate_prism_points": int(len(keep)),
        "retained_prism_points": int(np.count_nonzero(keep)),
        "retained_prism_charge_e": float(prism_weights.sum(dtype=np.float64)),
    }
    return coords, weights, meta


def quaternion_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def detector_q_grid(
    image_size: int,
    downsample: int,
    photon_energy_eV: float,
    detector_distance_m: float,
    native_pixel_size_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength_A = HC_EV_ANGSTROM / float(photon_energy_eV)
    k_Ainv = 2.0 * np.pi / wavelength_A
    pixel_size_m = float(downsample) * float(native_pixel_size_m)
    center = image_size / 2.0 - 0.5
    pix = np.arange(image_size, dtype=np.float64) - center
    x_m, y_m = np.meshgrid(pix * pixel_size_m, pix * pixel_size_m, indexing="xy")
    z_m = np.full_like(x_m, float(detector_distance_m))
    norm = np.sqrt(x_m * x_m + y_m * y_m + z_m * z_m)
    qx = k_Ainv * x_m / norm
    qy = k_Ainv * y_m / norm
    qr = np.sqrt(qx * qx + qy * qy)
    return qx.astype(np.float32), qy.astype(np.float32), qr.astype(np.float32)


def project_points(
    coords: np.ndarray,
    weights: np.ndarray,
    rotation: np.ndarray,
    pixel_size_A: float,
    margin_A: float,
) -> Tuple[np.ndarray, float]:
    u = coords @ rotation[0, :]
    v = coords @ rotation[1, :]
    max_abs = float(max(np.max(np.abs(u)), np.max(np.abs(v)))) + float(margin_A)
    n = next_even(int(math.ceil(2.0 * max_abs / float(pixel_size_A))))
    half_width = 0.5 * n * float(pixel_size_A)
    edges = np.linspace(-half_width, half_width, n + 1, dtype=np.float64)
    hist, _, _ = np.histogram2d(v, u, bins=(edges, edges), weights=weights)
    projection = (hist / (float(pixel_size_A) ** 2)).astype(np.float32)
    return projection, float(pixel_size_A)


def sample_complex_bilinear(
    values: np.ndarray,
    q_axis: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
) -> np.ndarray:
    flat_x = qx.ravel()
    flat_y = qy.ravel()
    ix = np.searchsorted(q_axis, flat_x, side="right") - 1
    iy = np.searchsorted(q_axis, flat_y, side="right") - 1
    ix = np.clip(ix, 0, len(q_axis) - 2)
    iy = np.clip(iy, 0, len(q_axis) - 2)
    x0 = q_axis[ix]
    x1 = q_axis[ix + 1]
    y0 = q_axis[iy]
    y1 = q_axis[iy + 1]
    tx = (flat_x - x0) / (x1 - x0)
    ty = (flat_y - y0) / (y1 - y0)

    v00 = values[iy, ix]
    v10 = values[iy, ix + 1]
    v01 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]
    out = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return out.reshape(qx.shape)


def diffraction_from_projection(
    projection: np.ndarray,
    pixel_size_A: float,
    qx: np.ndarray,
    qy: np.ndarray,
    fft_size: int,
) -> np.ndarray:
    if max(projection.shape) > fft_size:
        raise ValueError(f"projection shape {projection.shape} exceeds fft size {fft_size}")
    padded = np.zeros((fft_size, fft_size), dtype=np.float32)
    y0 = (fft_size - projection.shape[0]) // 2
    x0 = (fft_size - projection.shape[1]) // 2
    padded[y0 : y0 + projection.shape[0], x0 : x0 + projection.shape[1]] = projection
    amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))) * float(pixel_size_A) ** 2
    q_axis = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(fft_size, d=float(pixel_size_A))).astype(np.float32)
    sampled = sample_complex_bilinear(amp, q_axis, qx, qy)
    intensity = np.abs(sampled) ** 2
    return intensity.astype(np.float32)


def choose_rotation_indices(rotations: np.ndarray, count: int) -> np.ndarray:
    if count >= len(rotations):
        return np.arange(len(rotations), dtype=np.int64)
    return np.unique(np.linspace(0, len(rotations) - 1, int(count), dtype=np.int64))


def make_poisson_like_counts(
    intensities: np.ndarray,
    annulus_mask: np.ndarray,
    peak_percentile_photons: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = np.zeros_like(intensities, dtype=np.float32)
    for idx, image in enumerate(intensities):
        ref = np.percentile(image[annulus_mask], 99.7)
        scale = float(peak_percentile_photons) / max(float(ref), np.finfo(float).eps)
        lam = np.clip(image * scale, 0.0, 200.0)
        counts[idx] = rng.poisson(lam).astype(np.float32)
    return counts


def save_montage(
    output_dir: Path,
    smooth: np.ndarray,
    poisson: np.ndarray,
    labels: Sequence[str],
    annulus_mask: np.ndarray,
    filename: str,
) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = smooth.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#bdbdbd")

    smooth_disp = np.log10(np.where(annulus_mask[np.newaxis, :, :], smooth, np.nan) + 1e-3)
    count_disp = np.log10(np.where(annulus_mask[np.newaxis, :, :], poisson, np.nan) + 0.3)
    vmin_s, vmax_s = np.nanpercentile(smooth_disp, [5.0, 99.7])
    vmin_c, vmax_c = 0.0, max(2.0, float(np.nanpercentile(count_disp, 99.7)))

    for i in range(n):
        axes[0, i].imshow(smooth_disp[i], origin="lower", cmap=cmap, vmin=vmin_s, vmax=vmax_s)
        axes[1, i].imshow(count_disp[i], origin="lower", cmap=cmap, vmin=vmin_c, vmax=vmax_c)
        axes[0, i].set_title(labels[i], fontsize=8)
        axes[1, i].set_title("Poisson-like", fontsize=8)
        for row in range(2):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def radial_profile(image: np.ndarray, q_radius: np.ndarray, bins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = image[mask]
    radii = q_radius[mask]
    sums, _ = np.histogram(radii, bins=bins, weights=values)
    counts, _ = np.histogram(radii, bins=bins)
    return sums / np.maximum(counts, 1)


def save_radial_profiles(
    output_dir: Path,
    smooth: np.ndarray,
    model_labels: Sequence[str],
    q_radius: np.ndarray,
    annulus_mask: np.ndarray,
    filename: str,
) -> Tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, float(q_radius[annulus_mask].max()), 70)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rows: List[Dict[str, float | str | int]] = []
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    for idx, image in enumerate(smooth):
        prof = radial_profile(image, q_radius, bins, annulus_mask)
        norm = np.percentile(prof[prof > 0], 95) if np.any(prof > 0) else 1.0
        ax.plot(centers, prof / max(float(norm), np.finfo(float).eps), linewidth=1.0, alpha=0.85, label=model_labels[idx])
        for q, value in zip(centers, prof):
            rows.append({"image_index": idx, "label": model_labels[idx], "q_Ainv": float(q), "mean_intensity": float(value)})
    ax.set_yscale("log")
    ax.set_xlabel("q (1/A)")
    ax.set_ylabel("normalized radial mean")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    png_path = output_dir / filename
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    csv_path = output_dir / "groel_model_ab_preview_radial_profiles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_index", "label", "q_Ainv", "mean_intensity"])
        writer.writeheader()
        writer.writerows(rows)
    return png_path, csv_path


def save_model_b_paper_style_panel(
    output_dir: Path,
    poisson: np.ndarray,
    labels: Sequence[str],
    model_names: Sequence[str],
    annulus_mask: np.ndarray,
    filename: str,
) -> Path | None:
    b_indices = [idx for idx, model in enumerate(model_names) if model == "B_tetradecamer_D7"]
    if not b_indices:
        return None
    chosen = b_indices[: min(2, len(b_indices))]

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n + 0.6, 3.8), constrained_layout=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#2b0b58")

    h, w = poisson.shape[1:]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    center = h / 2.0 - 0.5
    display_mask = annulus_mask.copy()
    display_mask &= np.abs(yy - center) > 4.0
    display_mask &= ~((np.abs(xx - center) < 5.0) & (np.abs(yy - center) < 6.0))

    disp_all = np.log10(np.where(display_mask[np.newaxis, :, :], poisson, np.nan) + 0.3)
    vmax = max(2.0, float(np.nanpercentile(disp_all[chosen], 99.7)))
    last_im = None
    for ax, idx in zip(axes, chosen):
        disp = disp_all[idx]
        last_im = ax.imshow(disp, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(f"Model B {labels[idx]}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="log10(count + 0.3)")
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_first_rotation_model_comparison_panel(
    output_dir: Path,
    poisson: np.ndarray,
    labels: Sequence[str],
    model_names: Sequence[str],
    rotation_indices: Sequence[int],
    annulus_mask: np.ndarray,
    filename: str,
) -> Path | None:
    if not rotation_indices:
        return None
    first_rotation = int(rotation_indices[0])
    chosen = [
        idx
        for idx, (model, ridx) in enumerate(zip(model_names, rotation_indices))
        if int(ridx) == first_rotation and not str(model).startswith("A_")
    ]
    if len(chosen) < 2:
        return None

    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.6, 3.6), constrained_layout=True)
    if n == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#2b0b58")
    disp_all = np.log10(np.where(annulus_mask[np.newaxis, :, :], poisson, np.nan) + 0.3)
    vmax = max(2.0, float(np.nanpercentile(disp_all[chosen], 99.7)))
    last_im = None
    for ax, idx in zip(axes, chosen):
        last_im = ax.imshow(disp_all[idx], origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(labels[idx], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.82, label="log10(count + 0.3)")
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density-npz",
        type=Path,
        default=Path("validation") / "protein_1ss8_wk_density_meshes_sigma6" / "1ss8_wk_density_grid.npz",
    )
    parser.add_argument(
        "--rotations-h5",
        type=Path,
        default=Path("public_data") / "CXIDB_ID_187" / "rotations.h5",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("validation") / "groel_model_ab_preview")
    parser.add_argument("--num-rotations", type=int, default=6)
    parser.add_argument("--density-min-fraction", type=float, default=1e-4)
    parser.add_argument("--projection-pixel-size-A", type=float, default=1.0)
    parser.add_argument("--projection-margin-A", type=float, default=18.0)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--photon-energy-eV", type=float, default=1200.0)
    parser.add_argument("--detector-distance-m", type=float, default=0.15)
    parser.add_argument("--native-pixel-size-m", type=float, default=75e-6)
    parser.add_argument("--display-r-min-px", type=float, default=5.0)
    parser.add_argument("--display-r-max-px", type=float, default=60.0)
    parser.add_argument("--poisson-peak-percentile-photons", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=187)
    parser.add_argument("--include-prism-models", action="store_true")
    parser.add_argument("--prism-radii-A", type=parse_float_list, default=[35.0, 55.0])
    parser.add_argument("--prism-length-A", type=float, default=135.0)
    parser.add_argument("--prism-sides", type=int, default=14)
    parser.add_argument("--prism-spacing-A", type=float, default=2.0)
    parser.add_argument(
        "--prism-density-e-A3",
        type=float,
        default=0.267,
        help="Default is oxygen-only water electron density, roughly matching the public GroEL water-model script.",
    )
    parser.add_argument("--prism-exclude-protein-density-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points = load_density_points(args.density_npz, args.density_min_fraction)
    with h5py.File(args.rotations_h5, "r") as f:
        rotations = np.asarray(f["rotations"][...], dtype=np.float32)
    selected = choose_rotation_indices(rotations, args.num_rotations)

    qx, qy, qr = detector_q_grid(
        image_size=args.image_size,
        downsample=args.downsample,
        photon_energy_eV=args.photon_energy_eV,
        detector_distance_m=args.detector_distance_m,
        native_pixel_size_m=args.native_pixel_size_m,
    )
    y, x = np.meshgrid(np.arange(args.image_size), np.arange(args.image_size), indexing="ij")
    center = args.image_size / 2.0 - 0.5
    r_px = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    annulus_mask = (r_px >= float(args.display_r_min_px)) & (r_px <= float(args.display_r_max_px))

    model_specs: List[Tuple[str, Dict[str, float | int]]] = [
        ("A_heptamer_C7", {}),
        ("B_tetradecamer_D7", {}),
    ]
    if args.include_prism_models:
        for radius in args.prism_radii_A:
            label_radius = f"{float(radius):g}".replace(".", "p")
            model_specs.append(
                (
                    f"B_tetradecamer_D7_polyprism_R{label_radius}_N{int(args.prism_sides)}",
                    {
                        "radius_A": float(radius),
                        "length_A": float(args.prism_length_A),
                        "sides": int(args.prism_sides),
                        "spacing_A": float(args.prism_spacing_A),
                        "density_e_A3": float(args.prism_density_e_A3),
                        "exclude_fraction": float(args.prism_exclude_protein_density_fraction),
                    },
                )
            )

    all_smooth: List[np.ndarray] = []
    all_labels: List[str] = []
    all_model_names: List[str] = []
    all_rotation_indices: List[int] = []
    all_projection_shapes: List[Tuple[int, int]] = []
    model_metadata: Dict[str, Dict[str, float | int]] = {}
    for model, prism_meta in model_specs:
        if model.startswith("B_tetradecamer_D7_polyprism"):
            protein_coords, protein_weights, model_b_center = make_model_points(points, "B_tetradecamer_D7")
            coords, weights, prism_result_meta = add_polygonal_prism_to_model_b(
                points=points,
                protein_coords=protein_coords,
                protein_weights=protein_weights,
                model_b_center_A=model_b_center,
                radius_A=float(prism_meta["radius_A"]),
                length_A=float(prism_meta["length_A"]),
                sides=int(prism_meta["sides"]),
                spacing_A=float(prism_meta["spacing_A"]),
                density_e_A3=float(prism_meta["density_e_A3"]),
                exclude_fraction=float(prism_meta["exclude_fraction"]),
            )
            model_metadata[model] = prism_result_meta
        else:
            coords, weights, _ = make_model_points(points, model)
            model_metadata[model] = {}
        print(
            f"{model}: points={len(weights)} charge={weights.sum(dtype=np.float64):.3f} e",
            flush=True,
        )
        for ridx in selected:
            rot = quaternion_wxyz_to_matrix(rotations[int(ridx)])
            projection, projection_pixel_size_A = project_points(
                coords=coords,
                weights=weights,
                rotation=rot,
                pixel_size_A=args.projection_pixel_size_A,
                margin_A=args.projection_margin_A,
            )
            intensity = diffraction_from_projection(
                projection=projection,
                pixel_size_A=projection_pixel_size_A,
                qx=qx,
                qy=qy,
                fft_size=args.fft_size,
            )
            all_smooth.append(intensity)
            if model.startswith("A_"):
                short_model = "A"
            elif "polyprism" in model:
                short_model = "P" + str(prism_meta.get("radius_A", "")).replace(".0", "")
            else:
                short_model = "B"
            all_labels.append(f"{short_model} r{int(ridx)}")
            all_model_names.append(model)
            all_rotation_indices.append(int(ridx))
            all_projection_shapes.append(tuple(int(v) for v in projection.shape))
            print(f"  rotation {int(ridx):06d}: projection={projection.shape}", flush=True)

    smooth = np.stack(all_smooth, axis=0).astype(np.float32)
    poisson = make_poisson_like_counts(
        smooth,
        annulus_mask=annulus_mask,
        peak_percentile_photons=args.poisson_peak_percentile_photons,
        seed=args.seed,
    )

    montage_path = save_montage(
        args.output_dir,
        smooth=smooth,
        poisson=poisson,
        labels=all_labels,
        annulus_mask=annulus_mask,
        filename="groel_model_ab_preview_montage.png",
    )
    radial_png, radial_csv = save_radial_profiles(
        args.output_dir,
        smooth=smooth,
        model_labels=all_labels,
        q_radius=qr,
        annulus_mask=annulus_mask,
        filename="groel_model_ab_preview_radial_profiles.png",
    )
    paper_style_png = save_model_b_paper_style_panel(
        args.output_dir,
        poisson=poisson,
        labels=all_labels,
        model_names=all_model_names,
        annulus_mask=annulus_mask,
        filename="groel_model_b_paper_style_2panel.png",
    )
    first_rotation_comparison_png = save_first_rotation_model_comparison_panel(
        args.output_dir,
        poisson=poisson,
        labels=all_labels,
        model_names=all_model_names,
        rotation_indices=all_rotation_indices,
        annulus_mask=annulus_mask,
        filename="groel_model_b_prism_first_rotation_comparison.png",
    )

    arrays_path = args.output_dir / "groel_model_ab_preview_patterns.npz"
    np.savez_compressed(
        arrays_path,
        smooth_intensity=smooth,
        poisson_counts=poisson,
        qx_Ainv=qx,
        qy_Ainv=qy,
        q_radius_Ainv=qr,
        annulus_mask=annulus_mask,
        rotation_indices=np.asarray(all_rotation_indices, dtype=np.int64),
        rotations=rotations[np.asarray(all_rotation_indices, dtype=np.int64)],
        labels=np.asarray(all_labels),
        model_names=np.asarray(all_model_names),
    )

    summary = {
        "purpose": "Quick GroEL Model A/B dry-density diffraction preview for comparison with paper figures.",
        "density_npz": str(args.density_npz),
        "rotations_h5": str(args.rotations_h5),
        "output_dir": str(args.output_dir),
        "density_min_fraction": float(args.density_min_fraction),
        "retained_charge_fraction": float(points.retained_charge_e / points.total_charge_e),
        "selected_rotation_indices": selected.astype(int).tolist(),
        "models": ["A_heptamer_C7", "B_tetradecamer_D7"],
        "model_specs": [
            {"name": name, "metadata": model_metadata.get(name, {})}
            for name, _ in model_specs
        ],
        "detector": {
            "image_size": int(args.image_size),
            "downsample": int(args.downsample),
            "photon_energy_eV": float(args.photon_energy_eV),
            "detector_distance_m": float(args.detector_distance_m),
            "native_pixel_size_m": float(args.native_pixel_size_m),
            "q_max_Ainv": float(qr.max()),
        },
        "projection_shapes": [list(shape) for shape in all_projection_shapes],
        "outputs": {
            "montage_png": str(montage_path),
            "radial_profiles_png": str(radial_png),
            "radial_profiles_csv": str(radial_csv),
            "patterns_npz": str(arrays_path),
            "model_b_paper_style_png": str(paper_style_png) if paper_style_png else None,
            "first_rotation_model_comparison_png": str(first_rotation_comparison_png)
            if first_rotation_comparison_png
            else None,
        },
        "notes": [
            "Preview uses direct density-point projection plus FFT sampling.",
            "This intentionally matches the public code's 128 x 128 downsampled detector geometry.",
            "Poisson-like panels are display aids only; no experimental hit or calibrated background is used.",
        ],
    }
    summary_path = args.output_dir / "groel_model_ab_preview_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
