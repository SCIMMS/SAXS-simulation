#!/usr/bin/env python
"""Build a WK-derived electron-density grid and isosurface meshes for 1SS8.

The real-space density is constructed from non-hydrogen PDB atoms. Element
form-factor coefficients are loaded from periodictable's Waasmaier-Kirfel
Cromer-Mann table. Positive finite-width Gaussian terms are converted to
real-space Gaussians; near-constant residual terms are collected into the
resolution-limited smoothing Gaussian. This avoids non-physical negative
delta-like density artifacts while preserving each atom's f(0) electron count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import vtk
from periodictable.cromermann import getCMformula
from vtkmodules.util.numpy_support import numpy_to_vtk


@dataclass(frozen=True)
class AtomTable:
    xyz: np.ndarray
    element: np.ndarray
    occupancy: np.ndarray
    bfactor: np.ndarray


@dataclass(frozen=True)
class GaussianTerm:
    weight_e: float
    sigma_base_A: float
    source: str


def infer_element(line: str) -> str:
    if len(line) >= 78:
        element = line[76:78].strip()
        if element:
            return element.upper()
    name = line[12:16].strip()
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].title() in {"CL", "BR", "FE", "ZN", "MG", "CA", "MN"}:
        return letters[:2].title().upper()
    return letters[0].upper()


def parse_pdb_atoms(path: Path, include_hydrogen: bool = False) -> AtomTable:
    xyz: List[Tuple[float, float, float]] = []
    elements: List[str] = []
    occupancy: List[float] = []
    bfactor: List[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            element = infer_element(line)
            if not include_hydrogen and element in {"H", "D"}:
                continue
            try:
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
            try:
                occupancy.append(float(line[54:60]))
            except ValueError:
                occupancy.append(1.0)
            try:
                bfactor.append(float(line[60:66]))
            except ValueError:
                bfactor.append(0.0)
            elements.append(element)

    return AtomTable(
        xyz=np.asarray(xyz, dtype=np.float32),
        element=np.asarray(elements),
        occupancy=np.asarray(occupancy, dtype=np.float32),
        bfactor=np.asarray(bfactor, dtype=np.float32),
    )


def wk_terms_for_element(
    element: str,
    min_b_A2: float = 1.0e-3,
) -> Tuple[List[GaussianTerm], Dict[str, object]]:
    """Return positive real-space Gaussian terms derived from WK f0(s).

    The WK formula uses f(s)=c+sum_i a_i exp(-b_i s^2), s=q/(4*pi). A 3D
    Gaussian density with standard deviation sigma has transform
    exp(-8*pi^2*sigma^2*s^2), so sigma=sqrt(b/(8*pi^2)).
    """
    formula = getCMformula(element)
    a = np.asarray(formula.a, dtype=np.float64)
    b = np.asarray(formula.b, dtype=np.float64)
    c = float(formula.c)

    terms: List[GaussianTerm] = []
    residual = c
    raw_terms = []
    for i, (ai, bi) in enumerate(zip(a, b), start=1):
        raw_terms.append({"i": i, "a": float(ai), "b": float(bi)})
        if ai > 0.0 and bi > float(min_b_A2):
            terms.append(
                GaussianTerm(
                    weight_e=float(ai),
                    sigma_base_A=float(math.sqrt(float(bi) / (8.0 * math.pi * math.pi))),
                    source=f"a{i}",
                )
            )
        else:
            residual += float(ai)

    if residual > 1.0e-6:
        terms.append(GaussianTerm(weight_e=float(residual), sigma_base_A=0.0, source="residual"))
    elif residual < -1.0e-6:
        raise ValueError(f"negative WK residual for {element}: {residual}")

    total_f0 = float(c + a.sum())
    positive_total = float(sum(term.weight_e for term in terms))
    meta = {
        "element": element,
        "raw_terms": raw_terms,
        "c": c,
        "min_b_A2": float(min_b_A2),
        "total_f0_e": total_f0,
        "positive_model_total_e": positive_total,
        "terms": [
            {"weight_e": term.weight_e, "sigma_base_A": term.sigma_base_A, "source": term.source}
            for term in terms
        ],
    }
    return terms, meta


def bfactor_sigma_A(bfactor_A2: float) -> float:
    return float(math.sqrt(max(float(bfactor_A2), 0.0) / (8.0 * math.pi * math.pi)))


def grid_for_atoms(
    atoms_xyz: np.ndarray,
    spacing_A: float,
    margin_A: float,
) -> Tuple[np.ndarray, Tuple[int, int, int], Dict[str, object]]:
    xyz_min = atoms_xyz.min(axis=0).astype(np.float32)
    xyz_max = atoms_xyz.max(axis=0).astype(np.float32)
    origin = xyz_min - np.float32(margin_A)
    extent = xyz_max - xyz_min + np.float32(2.0 * margin_A)
    shape = tuple(int(math.ceil(float(v) / float(spacing_A))) + 1 for v in extent)
    meta = {
        "xyz_min_A": [float(x) for x in xyz_min],
        "xyz_max_A": [float(x) for x in xyz_max],
        "origin_A": [float(x) for x in origin],
        "shape": list(shape),
        "spacing_A": float(spacing_A),
        "margin_A": float(margin_A),
        "grid_voxels": int(np.prod(shape)),
        "grid_float32_MB": float(np.prod(shape) * np.dtype(np.float32).itemsize / (1024.0 * 1024.0)),
    }
    return origin.astype(np.float32), shape, meta


def add_gaussian_to_grid(
    density: np.ndarray,
    atom_xyz: np.ndarray,
    origin: np.ndarray,
    spacing_A: float,
    sigma_A: float,
    weight_e: float,
    clip_sigma: float,
) -> int:
    shape = density.shape
    radius = int(math.ceil(float(sigma_A) * float(clip_sigma) / float(spacing_A)))
    radius = max(radius, 1)
    center_grid = (atom_xyz - origin) / np.float32(spacing_A)
    cx, cy, cz = [int(round(float(v))) for v in center_grid]

    x0 = max(0, cx - radius)
    x1 = min(shape[0], cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(shape[1], cy + radius + 1)
    z0 = max(0, cz - radius)
    z1 = min(shape[2], cz + radius + 1)
    if x0 >= x1 or y0 >= y1 or z0 >= z1:
        return 0

    xs = origin[0] + np.arange(x0, x1, dtype=np.float32) * np.float32(spacing_A)
    ys = origin[1] + np.arange(y0, y1, dtype=np.float32) * np.float32(spacing_A)
    zs = origin[2] + np.arange(z0, z1, dtype=np.float32) * np.float32(spacing_A)
    dx2 = ((xs - atom_xyz[0]) / np.float32(sigma_A)) ** 2
    dy2 = ((ys - atom_xyz[1]) / np.float32(sigma_A)) ** 2
    dz2 = ((zs - atom_xyz[2]) / np.float32(sigma_A)) ** 2
    kernel = np.exp(-0.5 * (dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :])).astype(np.float32)

    voxel_volume = float(spacing_A) ** 3
    scale = float(weight_e) / max(float(kernel.sum()) * voxel_volume, np.finfo(np.float32).tiny)
    density[x0:x1, y0:y1, z0:z1] += kernel * np.float32(scale)
    return int(kernel.size)


def build_density(
    atoms: AtomTable,
    spacing_A: float,
    smooth_sigma_A: float,
    clip_sigma: float,
    use_bfactor: bool,
    min_b_A2: float,
    output_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    element_terms: Dict[str, List[GaussianTerm]] = {}
    element_meta: Dict[str, object] = {}
    for element in sorted(set(str(x) for x in atoms.element)):
        terms, meta = wk_terms_for_element(element, min_b_A2=min_b_A2)
        element_terms[element] = terms
        element_meta[element] = meta

    max_sigma_base = max(term.sigma_base_A for terms in element_terms.values() for term in terms)
    max_b_sigma = bfactor_sigma_A(float(np.nanmax(atoms.bfactor))) if use_bfactor else 0.0
    max_sigma_total = math.sqrt(max_sigma_base**2 + float(smooth_sigma_A) ** 2 + max_b_sigma**2)
    margin_A = max(4.0, float(clip_sigma) * max_sigma_total + 2.0 * float(spacing_A))
    origin, shape, grid_meta = grid_for_atoms(atoms.xyz, spacing_A=spacing_A, margin_A=margin_A)

    density = np.zeros(shape, dtype=np.float32)
    touched_voxels = 0
    term_count = 0
    expected_electrons = 0.0
    t0 = time.perf_counter()
    n_atoms = int(atoms.xyz.shape[0])
    progress_every = max(n_atoms // 10, 1)

    for idx, (xyz, element, occ, bfactor) in enumerate(
        zip(atoms.xyz, atoms.element, atoms.occupancy, atoms.bfactor),
        start=1,
    ):
        b_sigma = bfactor_sigma_A(float(bfactor)) if use_bfactor else 0.0
        for term in element_terms[str(element)]:
            sigma_total = math.sqrt(term.sigma_base_A**2 + float(smooth_sigma_A) ** 2 + b_sigma**2)
            weight = float(term.weight_e) * float(occ)
            touched_voxels += add_gaussian_to_grid(
                density=density,
                atom_xyz=xyz,
                origin=origin,
                spacing_A=spacing_A,
                sigma_A=sigma_total,
                weight_e=weight,
                clip_sigma=clip_sigma,
            )
            expected_electrons += weight
            term_count += 1
        if output_progress and (idx % progress_every == 0 or idx == n_atoms):
            print(f"density {idx}/{n_atoms} atoms elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    elapsed = time.perf_counter() - t0
    voxel_volume = float(spacing_A) ** 3
    meta = {
        "density_elapsed_s": float(elapsed),
        "spacing_A": float(spacing_A),
        "smooth_sigma_A": float(smooth_sigma_A),
        "clip_sigma": float(clip_sigma),
        "use_bfactor": bool(use_bfactor),
        "min_b_A2": float(min_b_A2),
        "term_count": int(term_count),
        "touched_voxels": int(touched_voxels),
        "expected_electrons": float(expected_electrons),
        "grid_integral_electrons": float(density.sum(dtype=np.float64) * voxel_volume),
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "density_mean": float(density.mean()),
        "density_nonzero_voxels": int(np.count_nonzero(density)),
        "grid": grid_meta,
        "element_terms": element_meta,
    }
    return density, meta


def numpy_to_image_data(field: np.ndarray, spacing_A: float, origin_A: Sequence[float]) -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    image.SetDimensions(int(field.shape[0]), int(field.shape[1]), int(field.shape[2]))
    image.SetSpacing(float(spacing_A), float(spacing_A), float(spacing_A))
    image.SetOrigin(float(origin_A[0]), float(origin_A[1]), float(origin_A[2]))
    vtk_scalars = numpy_to_vtk(field.astype(np.float32, copy=False).ravel(order="F"), deep=True)
    vtk_scalars.SetName("electron_density")
    image.GetPointData().SetScalars(vtk_scalars)
    return image


def extract_isosurface(image: vtk.vtkImageData, threshold: float) -> vtk.vtkPolyData:
    flying_edges = vtk.vtkFlyingEdges3D()
    flying_edges.SetInputData(image)
    flying_edges.SetValue(0, float(threshold))
    flying_edges.ComputeNormalsOn()
    flying_edges.ComputeGradientsOff()
    flying_edges.Update()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(flying_edges.GetOutputPort())
    triangles.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def count_components(polydata: vtk.vtkPolyData) -> int:
    if polydata.GetNumberOfCells() == 0:
        return 0
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    return int(connectivity.GetNumberOfExtractedRegions())


def polydata_stats(polydata: vtk.vtkPolyData) -> Dict[str, object]:
    bounds = tuple(float(x) for x in polydata.GetBounds()) if polydata.GetNumberOfPoints() else (0.0,) * 6
    return {
        "points": int(polydata.GetNumberOfPoints()),
        "cells": int(polydata.GetNumberOfCells()),
        "components": int(count_components(polydata)),
        "bounds": list(bounds),
    }


def quadric_decimate(polydata: vtk.vtkPolyData, target_reduction: float) -> vtk.vtkPolyData:
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(float(target_reduction))
    decimate.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(decimate.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def write_polydata(polydata: vtk.vtkPolyData, path: Path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write mesh: {path}")


def build_meshes(
    density: np.ndarray,
    spacing_A: float,
    origin_A: Sequence[float],
    output_dir: Path,
    threshold_fractions: Iterable[float],
    decimate_reduction: float,
    save_meshes: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    image = numpy_to_image_data(density, spacing_A=spacing_A, origin_A=origin_A)
    rows: List[Dict[str, object]] = []
    timings: Dict[str, float] = {}
    density_max = float(density.max())
    for fraction in threshold_fractions:
        threshold = density_max * float(fraction)
        label = f"thr{int(round(float(fraction) * 100)):03d}"

        t0 = time.perf_counter()
        mesh = extract_isosurface(image, threshold)
        timings[f"{label}_flying_edges_s"] = float(time.perf_counter() - t0)
        original_stats = polydata_stats(mesh)
        if save_meshes:
            write_polydata(mesh, output_dir / f"1ss8_wk_{label}_original.vtp")

        t1 = time.perf_counter()
        decimated = quadric_decimate(mesh, target_reduction=decimate_reduction)
        timings[f"{label}_quadric_decimation_s"] = float(time.perf_counter() - t1)
        decimated_stats = polydata_stats(decimated)
        if save_meshes:
            write_polydata(
                decimated,
                output_dir / f"1ss8_wk_{label}_quadric_r{int(round(decimate_reduction * 100)):02d}.vtp",
            )

        rows.append(
            {
                "threshold_fraction_of_max": float(fraction),
                "threshold_value": float(threshold),
                "original_points": original_stats["points"],
                "original_cells": original_stats["cells"],
                "original_components": original_stats["components"],
                "decimated_target_reduction": float(decimate_reduction),
                "decimated_points": decimated_stats["points"],
                "decimated_cells": decimated_stats["cells"],
                "decimated_components": decimated_stats["components"],
                "original_bounds": original_stats["bounds"],
                "decimated_bounds": decimated_stats["bounds"],
            }
        )
        print(
            f"{label}: original cells={original_stats['cells']} components={original_stats['components']} "
            f"decimated cells={decimated_stats['cells']} components={decimated_stats['components']}",
            flush=True,
        )
    return rows, timings


def save_preview_slices(density: np.ndarray, output_dir: Path) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mids = [axis // 2 for axis in density.shape]
    slices = [
        ("x", density[mids[0], :, :]),
        ("y", density[:, mids[1], :]),
        ("z", density[:, :, mids[2]]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    vmax = np.percentile(density[density > 0], 99.5) if np.any(density > 0) else density.max()
    for ax, (name, arr) in zip(axes, slices):
        im = ax.imshow(np.log10(np.maximum(arr.T, 1.0e-9)), origin="lower", cmap="viridis")
        ax.set_title(f"{name}-mid")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.75, label="log10 density")
    path = output_dir / "1ss8_wk_density_mid_slices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=Path(r"C:\Users\tnvlaos456\Downloads\1SS8.pdb"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation") / "protein_1ss8_wk_density_meshes",
    )
    parser.add_argument("--spacing-A", type=float, default=1.0)
    parser.add_argument("--smooth-sigma-A", type=float, default=4.0)
    parser.add_argument("--clip-sigma", type=float, default=3.0)
    parser.add_argument("--min-b-A2", type=float, default=1.0e-3)
    parser.add_argument("--threshold-fractions", type=parse_float_list, default=[0.10, 0.30, 0.50, 0.70, 0.90])
    parser.add_argument("--decimate-reduction", type=float, default=0.80)
    parser.add_argument("--include-hydrogen", action="store_true")
    parser.add_argument("--use-bfactor", action="store_true")
    parser.add_argument("--save-density", action="store_true")
    parser.add_argument("--save-meshes", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atoms = parse_pdb_atoms(args.pdb, include_hydrogen=bool(args.include_hydrogen))
    if atoms.xyz.size == 0:
        raise RuntimeError(f"No atoms parsed from {args.pdb}")

    total_t0 = time.perf_counter()
    density, density_meta = build_density(
        atoms=atoms,
        spacing_A=float(args.spacing_A),
        smooth_sigma_A=float(args.smooth_sigma_A),
        clip_sigma=float(args.clip_sigma),
        use_bfactor=bool(args.use_bfactor),
        min_b_A2=float(args.min_b_A2),
    )

    if bool(args.save_density):
        np.savez_compressed(
            args.output_dir / "1ss8_wk_density_grid.npz",
            density=density,
            origin_A=np.asarray(density_meta["grid"]["origin_A"], dtype=np.float32),
            spacing_A=np.float32(args.spacing_A),
        )

    preview_path = None
    if not bool(args.no_preview):
        preview_path = save_preview_slices(density, args.output_dir)

    mesh_rows, mesh_timings = build_meshes(
        density=density,
        spacing_A=float(args.spacing_A),
        origin_A=density_meta["grid"]["origin_A"],
        output_dir=args.output_dir,
        threshold_fractions=args.threshold_fractions,
        decimate_reduction=float(args.decimate_reduction),
        save_meshes=bool(args.save_meshes),
    )

    csv_path = args.output_dir / "1ss8_wk_isosurface_mesh_stats.csv"
    flat_rows = []
    for row in mesh_rows:
        flat = dict(row)
        flat["original_bounds"] = json.dumps(flat["original_bounds"])
        flat["decimated_bounds"] = json.dumps(flat["decimated_bounds"])
        flat_rows.append(flat)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    element_counts = {str(k): int(v) for k, v in zip(*np.unique(atoms.element, return_counts=True))}
    summary = {
        "pdb": str(args.pdb),
        "atom_count": int(atoms.xyz.shape[0]),
        "element_counts": element_counts,
        "density": density_meta,
        "mesh_stats_csv": str(csv_path),
        "mesh_rows": mesh_rows,
        "mesh_timings_s": mesh_timings,
        "preview_png": str(preview_path) if preview_path is not None else None,
        "total_elapsed_s": float(time.perf_counter() - total_t0),
    }
    summary_path = args.output_dir / "1ss8_wk_density_mesh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "mesh_stats_csv": str(csv_path), "total_elapsed_s": summary["total_elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
