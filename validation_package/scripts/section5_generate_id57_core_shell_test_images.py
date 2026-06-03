"""Generate detector-matched CXIDB ID 57 core-shell test images.

The model is a public, geometry-driven approximation of the ID 57 sample:
an outer Pd cube shell with an inner Au octahedral core. The forward model
uses ray-marched projected density followed by a 2D FFT and interpolation onto
the 401 x 401 CXI detector crop.
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

from section5_orientation_utils import normal_incidence_orientation_table  # noqa: E402


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
                return 1.0 / float(raw_value.strip())
            except ValueError:
                continue
    return None


def build_detector_q_maps(
    shape: tuple[int, int],
    pixel_size_m: float,
    distance_m: float,
    wavelength_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = shape
    y, x = np.indices((ny, nx), dtype=np.float32)
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0
    k = np.float32(2.0 * math.pi / wavelength_nm)
    qx = k * ((x - cx) * pixel_size_m / distance_m)
    qy = k * ((y - cy) * pixel_size_m / distance_m)
    return qx.astype(np.float32), qy.astype(np.float32)


def project_core_shell_density(
    rotation_obj_to_lab: np.ndarray,
    grid_size: int,
    extent_nm: float,
    depth_steps: int,
    depth_extent_nm: float,
    outer_cube_side_nm: float,
    core_octa_vertex_radius_nm: float,
    rho_pd: float,
    rho_au: float,
    chunk_rows: int,
) -> tuple[np.ndarray, float]:
    dx_nm = extent_nm / grid_size
    half_cube = np.float32(outer_cube_side_nm / 2.0)
    core_r = np.float32(core_octa_vertex_radius_nm)
    density_delta = np.float32(rho_au - rho_pd)
    rho_pd = np.float32(rho_pd)

    coords = (np.arange(grid_size, dtype=np.float32) - (grid_size - 1) / 2.0) * np.float32(dx_nm)
    t_values = (np.arange(depth_steps, dtype=np.float32) - (depth_steps - 1) / 2.0)
    t_values *= np.float32(depth_extent_nm / depth_steps)
    dt_nm = float(depth_extent_nm / depth_steps)

    # Convert lab ray coordinates into object coordinates.
    object_from_lab = rotation_obj_to_lab.T.astype(np.float32)
    projection = np.zeros((grid_size, grid_size), dtype=np.float32)
    t = t_values[None, None, :]
    x_lab = coords[None, :, None]

    for row0 in range(0, grid_size, chunk_rows):
        row1 = min(row0 + chunk_rows, grid_size)
        y_lab = coords[row0:row1, None, None]

        xo = object_from_lab[0, 0] * x_lab + object_from_lab[0, 1] * y_lab + object_from_lab[0, 2] * t
        yo = object_from_lab[1, 0] * x_lab + object_from_lab[1, 1] * y_lab + object_from_lab[1, 2] * t
        zo = object_from_lab[2, 0] * x_lab + object_from_lab[2, 1] * y_lab + object_from_lab[2, 2] * t

        ax = np.abs(xo)
        ay = np.abs(yo)
        az = np.abs(zo)
        cube = (ax <= half_cube) & (ay <= half_cube) & (az <= half_cube)
        core = cube & ((ax + ay + az) <= core_r)
        density = rho_pd * cube.astype(np.float32) + density_delta * core.astype(np.float32)
        projection[row0:row1, :] = density.sum(axis=2) * np.float32(dt_nm)

    return projection, dx_nm


def bilinear_interpolate_regular_grid(
    image: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    x = target_x.astype(np.float64)
    y = target_y.astype(np.float64)
    ix = np.searchsorted(axis_x, x, side="right") - 1
    iy = np.searchsorted(axis_y, y, side="right") - 1
    valid = (ix >= 0) & (iy >= 0) & (ix < len(axis_x) - 1) & (iy < len(axis_y) - 1)
    ix = np.clip(ix, 0, len(axis_x) - 2)
    iy = np.clip(iy, 0, len(axis_y) - 2)

    x0 = axis_x[ix]
    x1 = axis_x[ix + 1]
    y0 = axis_y[iy]
    y1 = axis_y[iy + 1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 != x0))
    ty = np.divide(y - y0, y1 - y0, out=np.zeros_like(y), where=(y1 != y0))

    i00 = image[iy, ix]
    i10 = image[iy, ix + 1]
    i01 = image[iy + 1, ix]
    i11 = image[iy + 1, ix + 1]
    out = (1.0 - tx) * (1.0 - ty) * i00 + tx * (1.0 - ty) * i10 + (1.0 - tx) * ty * i01 + tx * ty * i11
    out[~valid] = 0.0
    return out.astype(np.float32)


def projected_density_to_detector_intensity(
    projection: np.ndarray,
    dx_nm: float,
    qx_nm_inv: np.ndarray,
    qy_nm_inv: np.ndarray,
) -> np.ndarray:
    amp = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(projection))) * (dx_nm * dx_nm)
    intensity_fft = np.abs(amp) ** 2
    q_axis = np.fft.fftshift(2.0 * math.pi * np.fft.fftfreq(projection.shape[0], d=dx_nm))
    return bilinear_interpolate_regular_grid(intensity_fft, q_axis, q_axis, qx_nm_inv, qy_nm_inv)


def save_log_image(image: np.ndarray, out_path: Path, title: str) -> None:
    view = np.log1p(np.clip(image.astype(np.float64), 0.0, None))
    vmax = float(np.nanpercentile(view, 99.7))
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=160)
    im = ax.imshow(view, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    fig.colorbar(im, ax=ax, label="log1p(ADU)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_montage(images: np.ndarray, out_path: Path, cols: int = 5) -> None:
    rows = int(math.ceil(images.shape[0] / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3), dpi=150)
    axes = np.asarray(axes).reshape(rows, cols)
    vmax = float(np.nanpercentile(np.log1p(np.clip(images, 0.0, None)), 99.5))
    for i, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= images.shape[0]:
            ax.axis("off")
            continue
        view = np.log1p(np.clip(images[i], 0.0, None))
        ax.imshow(view, origin="lower", cmap="magma", vmin=0.0, vmax=vmax)
        ax.set_title(f"{i:02d}", fontsize=8)
    fig.tight_layout(pad=0.35)
    fig.savefig(out_path)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rotation_library(path: Path, count: int | None) -> list[dict[str, Any]]:
    with np.load(path) as data:
        rotations = data["rotation_matrices"].astype(np.float32)
        quaternions = data["quaternions_wxyz"].astype(np.float32)
        tilt_deg = data["tilt_deg"].astype(np.float32) if "tilt_deg" in data else np.full(rotations.shape[0], np.nan)
        azimuth_deg = (
            data["azimuth_deg"].astype(np.float32) if "azimuth_deg" in data else np.full(rotations.shape[0], np.nan)
        )
        roll_deg = data["roll_deg"].astype(np.float32) if "roll_deg" in data else np.full(rotations.shape[0], np.nan)
        halton_index = data["halton_index"].astype(np.int32) if "halton_index" in data else np.arange(rotations.shape[0])

    n = rotations.shape[0] if count is None else min(count, rotations.shape[0])
    rows: list[dict[str, Any]] = []
    for index in range(n):
        rows.append(
            {
                "index": index,
                "library_index": index,
                "halton_index": int(halton_index[index]),
                "tilt_deg": float(tilt_deg[index]),
                "azimuth_deg": float(azimuth_deg[index]),
                "roll_deg": float(roll_deg[index]),
                "rotation_matrix": rotations[index],
                "quaternion_wxyz": quaternions[index],
            }
        )
    return rows


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
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--rotation-library", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--outer-cube-side-nm", type=float, default=52.0)
    parser.add_argument("--core-octa-vertex-radius-nm", type=float, default=26.2)
    parser.add_argument("--rho-pd", type=float, default=1.0)
    parser.add_argument("--rho-au", type=float, default=1.5)
    parser.add_argument("--tilt-max-deg", type=float, default=8.0)
    parser.add_argument("--projection-grid", type=int, default=384)
    parser.add_argument("--projection-extent-nm", type=float, default=144.0)
    parser.add_argument("--depth-steps", type=int, default=256)
    parser.add_argument("--depth-extent-nm", type=float, default=120.0)
    parser.add_argument("--chunk-rows", type=int, default=24)
    parser.add_argument("--background-adu", type=float, default=8.0)
    parser.add_argument("--save-individual-png-count", type=int, default=None)
    parser.add_argument("--montage-count", type=int, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    cxi_path = data_dir / args.cxi_name
    geom_path = data_dir / args.geom_name
    out_dir = args.out_dir or data_dir / "synthetic_id57_core_shell_uniform"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    if args.rotation_library is not None:
        orientation_rows = load_rotation_library(args.rotation_library, args.count)
        orientation_sampling = "loaded_rotation_library"
    else:
        count = args.count if args.count is not None else 20
        orientation_rows = normal_incidence_orientation_table(count, args.tilt_max_deg)
        orientation_sampling = "halton_normal_incidence_generated_inline"

    pixel_size_m = read_pixel_size_from_geom(geom_path)
    if pixel_size_m is None:
        raise RuntimeError(f"Could not read detector pixel size from {geom_path}")

    with h5py.File(cxi_path, "r") as handle:
        data_path = DATA_LINK_PATH if DATA_LINK_PATH in handle else DETECTOR_DATA_PATH
        data = handle[data_path]
        detector_shape = tuple(int(x) for x in data.shape[1:])
        mean_exp = np.nanmean(data[...], axis=0)
        positive_totals = [float(np.nansum(np.clip(data[i], 0.0, None))) for i in range(data.shape[0])]
        target_total = float(np.median(positive_totals))
        distance_m = float(read_scalar(handle, "/entry_1/instrument_1/detector_1/distance"))
        energy_j = float(read_scalar(handle, "/entry_1/instrument_1/source_1/energy"))

    wavelength_nm = (HC_J_M / energy_j) * 1.0e9
    qx_nm_inv, qy_nm_inv = build_detector_q_maps(detector_shape, pixel_size_m, distance_m, wavelength_nm)

    valid_mask = np.isfinite(mean_exp) & (mean_exp != 0.0)
    valid_count = int(valid_mask.sum())
    structural_target = max(target_total - args.background_adu * valid_count, 0.2 * target_total)

    clean_patterns = []
    noisy_patterns = []
    rows: list[dict[str, Any]] = []
    png_limit = len(orientation_rows) if args.save_individual_png_count is None else max(args.save_individual_png_count, 0)

    for index, orientation in enumerate(orientation_rows):
        rotation = orientation["rotation_matrix"]

        projection, dx_nm = project_core_shell_density(
            rotation_obj_to_lab=rotation,
            grid_size=args.projection_grid,
            extent_nm=args.projection_extent_nm,
            depth_steps=args.depth_steps,
            depth_extent_nm=args.depth_extent_nm,
            outer_cube_side_nm=args.outer_cube_side_nm,
            core_octa_vertex_radius_nm=args.core_octa_vertex_radius_nm,
            rho_pd=args.rho_pd,
            rho_au=args.rho_au,
            chunk_rows=args.chunk_rows,
        )
        intensity = projected_density_to_detector_intensity(projection, dx_nm, qx_nm_inv, qy_nm_inv)
        intensity = np.clip(intensity, 0.0, None)
        intensity[~valid_mask] = 0.0
        scale = structural_target / max(float(intensity[valid_mask].sum()), 1.0)
        clean = intensity * scale
        clean[valid_mask] += args.background_adu
        clean[~valid_mask] = 0.0
        noisy = rng.poisson(np.clip(clean, 0.0, None)).astype(np.float32)
        noisy[~valid_mask] = 0.0

        clean_patterns.append(clean.astype(np.float32))
        noisy_patterns.append(noisy)
        if index < png_limit:
            save_log_image(noisy, out_dir / f"synthetic_id57_{index:04d}.png", f"synthetic ID57 {index:04d}")
        q = orientation["quaternion_wxyz"]

        rows.append(
            {
                "index": index,
                "library_index": orientation.get("library_index", index),
                "halton_index": orientation.get("halton_index", index + 1),
                "tilt_deg": orientation.get("tilt_deg"),
                "azimuth_deg": orientation.get("azimuth_deg"),
                "roll_deg": orientation.get("roll_deg"),
                "qw": float(q[0]),
                "qx": float(q[1]),
                "qy": float(q[2]),
                "qz": float(q[3]),
                "scale": float(scale),
                "clean_total": float(clean.sum()),
                "noisy_total": float(noisy.sum()),
                "max_adu": float(noisy.max()),
            }
        )

    clean_stack = np.stack(clean_patterns, axis=0)
    noisy_stack = np.stack(noisy_patterns, axis=0)
    montage_count = noisy_stack.shape[0] if args.montage_count is None else max(args.montage_count, 0)
    if montage_count > 0:
        save_montage(noisy_stack[:montage_count], out_dir / "synthetic_id57_montage.png")
    write_csv(out_dir / "synthetic_id57_parameters.csv", rows)

    summary = {
        "model": "outer Pd cube shell plus inner Au octahedral core",
        "count": len(orientation_rows),
        "seed": args.seed,
        "cxi_path": str(cxi_path),
        "geom_path": str(geom_path),
        "out_dir": str(out_dir),
        "detector_shape": list(detector_shape),
        "wavelength_nm": wavelength_nm,
        "detector_distance_m": distance_m,
        "pixel_size_m": pixel_size_m,
        "target_positive_total_from_experiment": target_total,
        "valid_detector_pixels": valid_count,
        "outer_cube_side_nm": args.outer_cube_side_nm,
        "core_octa_vertex_radius_nm": args.core_octa_vertex_radius_nm,
        "rho_pd": args.rho_pd,
        "rho_au": args.rho_au,
        "tilt_max_deg": args.tilt_max_deg,
        "projection_grid": args.projection_grid,
        "projection_extent_nm": args.projection_extent_nm,
        "depth_steps": args.depth_steps,
        "depth_extent_nm": args.depth_extent_nm,
        "background_adu": args.background_adu,
        "save_individual_png_count": png_limit,
        "montage_count": montage_count,
        "rotation_library": str(args.rotation_library) if args.rotation_library is not None else None,
        "orientation_sampling": orientation_sampling,
        "symmetry_reduction": "+z face representative; roll restricted to [0, 90 deg)",
        "quaternion_convention": "wxyz",
    }

    rotation_matrices = np.stack([row["rotation_matrix"] for row in orientation_rows], axis=0).astype(np.float32)
    quaternions_wxyz = np.stack([row["quaternion_wxyz"] for row in orientation_rows], axis=0).astype(np.float32)
    np.savez_compressed(
        out_dir / "synthetic_id57_patterns.npz",
        noisy_patterns=noisy_stack,
        clean_patterns=clean_stack,
        valid_mask=valid_mask.astype(np.uint8),
        qx_nm_inv=qx_nm_inv.astype(np.float32),
        qy_nm_inv=qy_nm_inv.astype(np.float32),
        rotation_matrices=rotation_matrices,
        quaternions_wxyz=quaternions_wxyz,
    )
    (out_dir / "synthetic_id57_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
