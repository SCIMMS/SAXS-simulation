#!/usr/bin/env python
"""Validate 1SS8 multi-isosurface layer approximation along z projection.

This compares:
  1. Direct density projection: sum_z rho(x,y,z) dz.
  2. Voxel threshold layers: sum_j w_j I[rho >= tau_j], then project.
  3. VTK mesh-stencil layers from Flying-Edges .vtp meshes, then project.

The first pass intentionally uses the z direction only so projection errors can
be diagnosed before adding arbitrary rotations or CZT diffraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy


def load_density(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    data = np.load(npz_path)
    density = np.asarray(data["density"], dtype=np.float32)
    origin_A = np.asarray(data["origin_A"], dtype=np.float32)
    spacing_A = float(np.asarray(data["spacing_A"]).item())
    return density, origin_A, spacing_A


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def threshold_values(density: np.ndarray, fractions: Sequence[float]) -> np.ndarray:
    return np.asarray(fractions, dtype=np.float64) * float(density.max())


def lower_bound_weights(thresholds: np.ndarray) -> np.ndarray:
    weights = np.empty_like(thresholds, dtype=np.float64)
    weights[0] = thresholds[0]
    weights[1:] = np.diff(thresholds)
    return weights


def midpoint_weights(thresholds: np.ndarray, density_max: float) -> np.ndarray:
    boundaries = np.asarray(thresholds, dtype=np.float64)
    levels = np.empty(boundaries.size + 1, dtype=np.float64)
    levels[0] = 0.0
    if boundaries.size > 1:
        levels[1:-1] = 0.5 * (boundaries[:-1] + boundaries[1:])
    levels[-1] = float(density_max)
    weights = np.diff(levels)
    return weights


def least_squares_weights(density: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Solve min_w ||rho - sum_j w_j I[rho>=tau_j]||_2 over all voxels."""
    masks = [(density >= threshold) for threshold in thresholds]
    n = len(masks)
    xtx = np.zeros((n, n), dtype=np.float64)
    xty = np.zeros(n, dtype=np.float64)
    for i, mask_i in enumerate(masks):
        xty[i] = float(density[mask_i].sum(dtype=np.float64))
        for j, mask_j in enumerate(masks):
            if i <= j:
                value = float(np.logical_and(mask_i, mask_j).sum())
                xtx[i, j] = value
                xtx[j, i] = value
    weights, *_ = np.linalg.lstsq(xtx, xty, rcond=None)
    return np.maximum(weights, 0.0)


def project_direct_density_z(density: np.ndarray, spacing_A: float) -> np.ndarray:
    return density.sum(axis=2, dtype=np.float64) * float(spacing_A)


def project_voxel_layers_z(
    density: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
    spacing_A: float,
) -> np.ndarray:
    projection = np.zeros(density.shape[:2], dtype=np.float64)
    for threshold, weight in zip(thresholds, weights):
        projection += (density >= float(threshold)).sum(axis=2, dtype=np.float64) * float(spacing_A) * float(weight)
    return projection


def polydata_to_mask(
    polydata: vtk.vtkPolyData,
    shape: Tuple[int, int, int],
    spacing_A: float,
    origin_A: Sequence[float],
) -> np.ndarray:
    white_image = vtk.vtkImageData()
    white_image.SetDimensions(int(shape[0]), int(shape[1]), int(shape[2]))
    white_image.SetSpacing(float(spacing_A), float(spacing_A), float(spacing_A))
    white_image.SetOrigin(float(origin_A[0]), float(origin_A[1]), float(origin_A[2]))
    white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    white_scalars = white_image.GetPointData().GetScalars()
    white_scalars.Fill(1)

    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(polydata)
    poly_to_stencil.SetOutputOrigin(white_image.GetOrigin())
    poly_to_stencil.SetOutputSpacing(white_image.GetSpacing())
    poly_to_stencil.SetOutputWholeExtent(white_image.GetExtent())
    poly_to_stencil.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(white_image)
    stencil.SetStencilConnection(poly_to_stencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()

    mask = vtk_to_numpy(stencil.GetOutput().GetPointData().GetScalars()).reshape(shape, order="F")
    return mask.astype(bool, copy=False)


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = reader.GetOutput()
    if polydata.GetNumberOfCells() == 0:
        raise RuntimeError(f"mesh has no cells: {path}")
    return polydata


def project_mesh_layers_z(
    mesh_dir: Path,
    mesh_suffix: str,
    fractions: Sequence[float],
    weights: np.ndarray,
    shape: Tuple[int, int, int],
    spacing_A: float,
    origin_A: Sequence[float],
) -> np.ndarray:
    projection = np.zeros(shape[:2], dtype=np.float64)
    for fraction, weight in zip(fractions, weights):
        label = f"thr{int(round(float(fraction) * 100)):03d}"
        mesh_path = mesh_dir / f"1ss8_wk_{label}_{mesh_suffix}.vtp"
        polydata = read_polydata(mesh_path)
        mask = polydata_to_mask(polydata, shape=shape, spacing_A=spacing_A, origin_A=origin_A)
        projection += mask.sum(axis=2, dtype=np.float64) * float(spacing_A) * float(weight)
    return projection


def metrics(reference: np.ndarray, candidate: np.ndarray, name: str) -> Dict[str, float | str]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    diff = cand - ref
    support = np.logical_or(ref > 0.0, cand > 0.0)
    if not np.any(support):
        support = np.ones(ref.shape, dtype=bool)
    r = ref[support]
    c = cand[support]
    d = diff[support]
    rmse = float(np.sqrt(np.mean(d * d)))
    mae = float(np.mean(np.abs(d)))
    ref_rms = float(np.sqrt(np.mean(r * r)))
    ref_mean = float(np.mean(np.abs(r)))
    total_ref = float(ref.sum())
    total_cand = float(cand.sum())
    r0 = r - float(r.mean())
    c0 = c - float(c.mean())
    denom_corr = float(np.sqrt(np.sum(r0 * r0) * np.sum(c0 * c0)))
    denom_cos = float(np.sqrt(np.sum(r * r) * np.sum(c * c)))
    return {
        "method": name,
        "total_ref": total_ref,
        "total_candidate": total_cand,
        "relative_total_error": float((total_cand - total_ref) / total_ref) if total_ref else 0.0,
        "rmse": rmse,
        "relative_rmse_over_ref_rms": float(rmse / ref_rms) if ref_rms else 0.0,
        "mae": mae,
        "relative_mae_over_ref_mean": float(mae / ref_mean) if ref_mean else 0.0,
        "max_abs_error": float(np.max(np.abs(d))),
        "pearson": float(np.sum(r0 * c0) / denom_corr) if denom_corr else 0.0,
        "cosine": float(np.sum(r * c) / denom_cos) if denom_cos else 0.0,
        "support_pixels": int(np.count_nonzero(support)),
    }


def save_projection_arrays(output_dir: Path, arrays: Dict[str, np.ndarray]) -> Path:
    path = output_dir / "1ss8_z_projection_validation_arrays.npz"
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    return path


def save_comparison_figure(output_dir: Path, arrays: Dict[str, np.ndarray], filename: str) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = arrays["direct"]
    names = [name for name in arrays if name != "direct"]
    cols = len(names) + 1
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6.4), constrained_layout=True)
    vmax = np.percentile(ref[ref > 0], 99.5) if np.any(ref > 0) else float(ref.max())
    first = axes[0, 0].imshow(ref.T, origin="lower", cmap="viridis", vmax=vmax)
    axes[0, 0].set_title("direct")
    axes[1, 0].axis("off")
    for ax in axes[:, 0]:
        ax.set_xticks([])
        ax.set_yticks([])

    err_abs_max = 0.0
    for name in names:
        err_abs_max = max(err_abs_max, float(np.max(np.abs(arrays[name] - ref))))
    err_abs_max = max(err_abs_max, np.finfo(float).eps)

    for col, name in enumerate(names, start=1):
        candidate = arrays[name]
        err = candidate - ref
        axes[0, col].imshow(candidate.T, origin="lower", cmap="viridis", vmax=vmax)
        axes[0, col].set_title(name)
        axes[1, col].imshow(err.T, origin="lower", cmap="coolwarm", vmin=-err_abs_max, vmax=err_abs_max)
        axes[1, col].set_title(f"{name} - direct")
        for ax in axes[:, col]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(first, ax=axes[0, :], shrink=0.8, label="projected e / A^2")
    path = output_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_metrics_csv(output_dir: Path, rows: List[Dict[str, float | str]]) -> Path:
    path = output_dir / "1ss8_z_projection_validation_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density-npz",
        type=Path,
        default=Path("validation")
        / "protein_1ss8_wk_density_meshes_sigma6"
        / "1ss8_wk_density_grid.npz",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("validation") / "protein_1ss8_wk_density_meshes_sigma6",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation") / "protein_1ss8_z_projection_validation_sigma6",
    )
    parser.add_argument("--threshold-fractions", type=parse_float_list, default=[0.10, 0.30, 0.50, 0.70, 0.90])
    parser.add_argument("--skip-mesh-stencil", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    density, origin_A, spacing_A = load_density(args.density_npz)
    fractions = list(args.threshold_fractions)
    thresholds = threshold_values(density, fractions)

    direct = project_direct_density_z(density, spacing_A)
    weights = {
        "lower": lower_bound_weights(thresholds),
        "midpoint": midpoint_weights(thresholds, float(density.max())),
        "lstsq": least_squares_weights(density, thresholds),
    }

    arrays: Dict[str, np.ndarray] = {"direct": direct}
    rows: List[Dict[str, float | str]] = []
    weight_meta = {}
    for name, w in weights.items():
        projection = project_voxel_layers_z(density, thresholds, w, spacing_A)
        method = f"voxel_{name}"
        arrays[method] = projection
        rows.append(metrics(direct, projection, method))
        weight_meta[name] = [float(x) for x in w]

    if not bool(args.skip_mesh_stencil):
        for suffix in ("original", "quadric_r80"):
            for weight_name in ("midpoint", "lstsq"):
                projection = project_mesh_layers_z(
                    mesh_dir=args.mesh_dir,
                    mesh_suffix=suffix,
                    fractions=fractions,
                    weights=weights[weight_name],
                    shape=density.shape,
                    spacing_A=spacing_A,
                    origin_A=origin_A,
                )
                method = f"mesh_stencil_{suffix}_{weight_name}"
                arrays[method] = projection
                rows.append(metrics(direct, projection, method))

    metrics_csv = write_metrics_csv(args.output_dir, rows)
    arrays_npz = save_projection_arrays(args.output_dir, arrays)
    voxel_fig = save_comparison_figure(
        args.output_dir,
        {k: arrays[k] for k in ("direct", "voxel_lower", "voxel_midpoint", "voxel_lstsq")},
        "1ss8_z_projection_voxel_layer_comparison.png",
    )
    mesh_fig = None
    if not bool(args.skip_mesh_stencil):
        mesh_fig = save_comparison_figure(
            args.output_dir,
            {
                "direct": arrays["direct"],
                "mesh_original_mid": arrays["mesh_stencil_original_midpoint"],
                "mesh_quadric_r80_mid": arrays["mesh_stencil_quadric_r80_midpoint"],
            },
            "1ss8_z_projection_mesh_stencil_comparison.png",
        )

    summary = {
        "density_npz": str(args.density_npz),
        "mesh_dir": str(args.mesh_dir),
        "shape": list(density.shape),
        "spacing_A": float(spacing_A),
        "origin_A": [float(x) for x in origin_A],
        "threshold_fractions": [float(x) for x in fractions],
        "threshold_values": [float(x) for x in thresholds],
        "weights": weight_meta,
        "metrics_csv": str(metrics_csv),
        "arrays_npz": str(arrays_npz),
        "voxel_figure": str(voxel_fig),
        "mesh_figure": str(mesh_fig) if mesh_fig is not None else None,
        "metrics": rows,
    }
    summary_path = args.output_dir / "1ss8_z_projection_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "metrics_csv": str(metrics_csv)}, indent=2))


if __name__ == "__main__":
    main()
