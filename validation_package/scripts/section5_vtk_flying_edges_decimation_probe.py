#!/usr/bin/env python
"""Probe VTK Flying Edges and decimation on disconnected isosurfaces.

This script builds a small synthetic scalar field with two separated Gaussian
lobes, extracts isosurfaces with vtkFlyingEdges3D, then checks whether mesh
decimation changes the number of disconnected surface components.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


def make_disconnected_gaussian_field(
    shape: Tuple[int, int, int] = (96, 96, 96),
    extent: float = 3.0,
    sigma: float = 0.42,
    secondary_weight: float = 1.0,
) -> np.ndarray:
    """Return two separated Gaussian blobs plus a tiny smooth pedestal."""
    nx, ny, nz = shape
    x = np.linspace(-extent, extent, nx, dtype=np.float32)
    y = np.linspace(-extent, extent, ny, dtype=np.float32)
    z = np.linspace(-extent, extent, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    centers = np.array(
        [
            [-1.05, -0.25, 0.00],
            [1.05, 0.25, 0.08],
        ],
        dtype=np.float32,
    )
    weights = np.array([1.0, float(secondary_weight)], dtype=np.float32)
    field = np.zeros(shape, dtype=np.float32)
    for center, weight in zip(centers, weights):
        rr2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
        field += weight * np.exp(-0.5 * rr2 / (sigma * sigma))

    # Keep this small enough not to bridge the two blobs at tested thresholds.
    rr2 = xx**2 + yy**2 + zz**2
    field += np.float32(0.015) * np.exp(-0.5 * rr2 / (1.8 * 1.8))
    return field


def numpy_to_image_data(field: np.ndarray, spacing: Tuple[float, float, float]) -> vtk.vtkImageData:
    """Convert a 3D numpy array indexed as x, y, z into vtkImageData."""
    if field.ndim != 3:
        raise ValueError(f"expected a 3D field, got shape={field.shape}")

    image = vtk.vtkImageData()
    image.SetDimensions(int(field.shape[0]), int(field.shape[1]), int(field.shape[2]))
    image.SetSpacing(*map(float, spacing))
    image.SetOrigin(0.0, 0.0, 0.0)

    vtk_scalars = numpy_to_vtk(field.astype(np.float32, copy=False).ravel(order="F"), deep=True)
    vtk_scalars.SetName("density")
    image.GetPointData().SetScalars(vtk_scalars)
    return image


def extract_isosurface(image: vtk.vtkImageData, threshold: float) -> vtk.vtkPolyData:
    flying_edges = vtk.vtkFlyingEdges3D()
    flying_edges.SetInputData(image)
    flying_edges.SetValue(0, float(threshold))
    flying_edges.ComputeNormalsOff()
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


def mesh_stats(polydata: vtk.vtkPolyData) -> Dict[str, float]:
    bounds = tuple(float(x) for x in polydata.GetBounds()) if polydata.GetNumberOfPoints() else (0.0,) * 6
    return {
        "points": int(polydata.GetNumberOfPoints()),
        "cells": int(polydata.GetNumberOfCells()),
        "components": count_components(polydata),
        "bounds_x_min": bounds[0],
        "bounds_x_max": bounds[1],
        "bounds_y_min": bounds[2],
        "bounds_y_max": bounds[3],
        "bounds_z_min": bounds[4],
        "bounds_z_max": bounds[5],
    }


def quadric_decimate(polydata: vtk.vtkPolyData, target_reduction: float) -> vtk.vtkPolyData:
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(float(target_reduction))
    decimate.Update()
    return decimate.GetOutput()


def topology_preserving_decimate(polydata: vtk.vtkPolyData, target_reduction: float) -> vtk.vtkPolyData:
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(float(target_reduction))
    decimate.PreserveTopologyOn()
    decimate.SplittingOff()
    decimate.BoundaryVertexDeletionOff()
    decimate.Update()
    return decimate.GetOutput()


def write_polydata(polydata: vtk.vtkPolyData, path: Path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write {path}")


def run_probe(
    output_dir: Path,
    shape: Tuple[int, int, int],
    thresholds: Iterable[float],
    target_reductions: Iterable[float],
    secondary_weight: float,
    save_meshes: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    field = make_disconnected_gaussian_field(shape=shape, secondary_weight=secondary_weight)
    spacing = tuple(6.0 / (n - 1) for n in shape)
    image = numpy_to_image_data(field, spacing=spacing)

    rows = []
    for threshold_fraction in thresholds:
        threshold = float(field.max() * float(threshold_fraction))
        original = extract_isosurface(image, threshold)
        original_stats = mesh_stats(original)
        case_name = f"thr{int(round(threshold_fraction * 100)):03d}"

        if save_meshes:
            write_polydata(original, output_dir / f"{case_name}_original.vtp")

        rows.append(
            {
                "threshold_fraction": float(threshold_fraction),
                "threshold_value": threshold,
                "method": "original_flying_edges",
                "target_reduction": 0.0,
                **original_stats,
            }
        )

        for target_reduction in target_reductions:
            quadric = quadric_decimate(original, target_reduction)
            quadric_stats = mesh_stats(quadric)
            if save_meshes:
                write_polydata(
                    quadric,
                    output_dir / f"{case_name}_quadric_r{int(round(target_reduction * 100)):02d}.vtp",
                )
            rows.append(
                {
                    "threshold_fraction": float(threshold_fraction),
                    "threshold_value": threshold,
                    "method": "vtkQuadricDecimation",
                    "target_reduction": float(target_reduction),
                    **quadric_stats,
                }
            )

            decimate_pro = topology_preserving_decimate(original, target_reduction)
            decimate_pro_stats = mesh_stats(decimate_pro)
            if save_meshes:
                write_polydata(
                    decimate_pro,
                    output_dir / f"{case_name}_decimate_pro_r{int(round(target_reduction * 100)):02d}.vtp",
                )
            rows.append(
                {
                    "threshold_fraction": float(threshold_fraction),
                    "threshold_value": threshold,
                    "method": "vtkDecimatePro_preserve_topology",
                    "target_reduction": float(target_reduction),
                    **decimate_pro_stats,
                }
            )

    csv_path = output_dir / "flying_edges_decimation_probe.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "vtk_version": vtk.vtkVersion.GetVTKVersion(),
        "field_shape": list(shape),
        "field_min": float(field.min()),
        "field_max": float(field.max()),
        "secondary_weight": float(secondary_weight),
        "threshold_fractions": [float(x) for x in thresholds],
        "target_reductions": [float(x) for x in target_reductions],
        "csv": str(csv_path),
        "rows": rows,
    }
    json_path = output_dir / "flying_edges_decimation_probe_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation") / "vtk_flying_edges_decimation_probe",
    )
    parser.add_argument("--shape", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument(
        "--secondary-weight",
        type=float,
        default=1.0,
        help="Relative height of the second Gaussian lobe.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.10, 0.30, 0.50, 0.70, 0.90),
        help="Fractions of the scalar-field maximum.",
    )
    parser.add_argument(
        "--target-reductions",
        type=float,
        nargs="+",
        default=(0.50, 0.80, 0.90),
    )
    parser.add_argument("--save-meshes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_probe(
        output_dir=args.output_dir,
        shape=tuple(args.shape),
        thresholds=args.thresholds,
        target_reductions=args.target_reductions,
        secondary_weight=float(args.secondary_weight),
        save_meshes=bool(args.save_meshes),
    )
    print(json.dumps({k: summary[k] for k in ("vtk_version", "field_shape", "csv")}, indent=2))


if __name__ == "__main__":
    main()
