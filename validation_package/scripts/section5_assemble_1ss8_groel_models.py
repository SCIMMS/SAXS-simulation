#!/usr/bin/env python
"""Assemble GroEL heptamer and tetradecamer mesh models from 1SS8 meshes.

Model A is the existing 1SS8 chain A-G heptamer mesh.
Model B is assembled by appending Model A and a BIOMT-generated copy using the
PDB REMARK 350 transform (x, y, z) -> (x, -y, -z). This is a proper 180 degree
rotation around the x-axis and corresponds to the tetradecamer biological unit
described in 1SS8.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import vtk


BIOMT_TETRADECAMER_COPY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(reader.GetOutput())
    if polydata.GetNumberOfCells() == 0:
        raise RuntimeError(f"mesh has no cells: {path}")
    return polydata


def write_polydata(polydata: vtk.vtkPolyData, path: Path) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if writer.Write() != 1:
        raise RuntimeError(f"failed to write mesh: {path}")


def transform_polydata(polydata: vtk.vtkPolyData, matrix_rows: Tuple[Tuple[float, ...], ...]) -> vtk.vtkPolyData:
    matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            matrix.SetElement(i, j, float(matrix_rows[i][j]))

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)

    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputData(polydata)
    filt.SetTransform(transform)
    filt.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(filt.GetOutput())
    return out


def append_polydata(parts: Iterable[vtk.vtkPolyData], clean: bool = False) -> vtk.vtkPolyData:
    appender = vtk.vtkAppendPolyData()
    for part in parts:
        appender.AddInputData(part)
    appender.Update()

    if clean:
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(appender.GetOutputPort())
        cleaner.Update()
        out = vtk.vtkPolyData()
        out.DeepCopy(cleaner.GetOutput())
        return out

    out = vtk.vtkPolyData()
    out.DeepCopy(appender.GetOutput())
    return out


def count_components(polydata: vtk.vtkPolyData) -> int:
    if polydata.GetNumberOfCells() == 0:
        return 0
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    return int(connectivity.GetNumberOfExtractedRegions())


def signed_volume(polydata: vtk.vtkPolyData) -> float:
    mass = vtk.vtkMassProperties()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polydata)
    triangles.Update()
    mass.SetInputConnection(triangles.GetOutputPort())
    try:
        return float(mass.GetVolume())
    except Exception:
        return float("nan")


def stats(polydata: vtk.vtkPolyData) -> Dict[str, object]:
    bounds = tuple(float(x) for x in polydata.GetBounds()) if polydata.GetNumberOfPoints() else (0.0,) * 6
    return {
        "points": int(polydata.GetNumberOfPoints()),
        "cells": int(polydata.GetNumberOfCells()),
        "components": int(count_components(polydata)),
        "volume_A3": signed_volume(polydata),
        "bounds": list(bounds),
    }


def expected_input_names(fractions: Iterable[float], suffixes: Iterable[str]) -> List[Tuple[float, str, str]]:
    rows = []
    for fraction in fractions:
        label = f"thr{int(round(float(fraction) * 100)):03d}"
        for suffix in suffixes:
            rows.append((float(fraction), label, suffix))
    return rows


def assemble_models(
    mesh_dir: Path,
    output_dir: Path,
    fractions: Iterable[float],
    suffixes: Iterable[str],
    copy_model_a: bool = True,
) -> Dict[str, object]:
    model_a_dir = output_dir / "model_A_heptamer_C7"
    model_b_dir = output_dir / "model_B_tetradecamer_D7"
    model_a_dir.mkdir(parents=True, exist_ok=True)
    model_b_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for fraction, label, suffix in expected_input_names(fractions, suffixes):
        input_path = mesh_dir / f"1ss8_wk_{label}_{suffix}.vtp"
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        heptamer = read_polydata(input_path)
        heptamer_stats = stats(heptamer)
        model_a_name = f"1ss8_model_A_heptamer_C7_{label}_{suffix}.vtp"
        model_a_path = model_a_dir / model_a_name
        if copy_model_a:
            shutil.copy2(input_path, model_a_path)
        else:
            write_polydata(heptamer, model_a_path)

        copied = transform_polydata(heptamer, BIOMT_TETRADECAMER_COPY)
        tetradecamer = append_polydata((heptamer, copied), clean=False)
        tetradecamer_stats = stats(tetradecamer)
        model_b_name = f"1ss8_model_B_tetradecamer_D7_{label}_{suffix}.vtp"
        model_b_path = model_b_dir / model_b_name
        write_polydata(tetradecamer, model_b_path)

        for model_name, path, row_stats in (
            ("A_heptamer_C7", model_a_path, heptamer_stats),
            ("B_tetradecamer_D7", model_b_path, tetradecamer_stats),
        ):
            flat = {
                "model": model_name,
                "threshold_fraction": fraction,
                "threshold_label": label,
                "mesh_suffix": suffix,
                "path": str(path),
                "points": row_stats["points"],
                "cells": row_stats["cells"],
                "components": row_stats["components"],
                "volume_A3": row_stats["volume_A3"],
                "bounds": json.dumps(row_stats["bounds"]),
            }
            rows.append(flat)
            print(
                f"{model_name} {label} {suffix}: cells={flat['cells']} "
                f"components={flat['components']} bounds={row_stats['bounds']}",
                flush=True,
            )

    csv_path = output_dir / "1ss8_groel_model_mesh_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source_mesh_dir": str(mesh_dir),
        "output_dir": str(output_dir),
        "model_A": {
            "name": "heptamer",
            "symmetry": "C7",
            "description": "Original 1SS8 chain A-G mesh.",
            "directory": str(model_a_dir),
        },
        "model_B": {
            "name": "tetradecamer",
            "symmetry": "D7",
            "description": "Append original heptamer mesh with BIOMT copy (x, -y, -z).",
            "directory": str(model_b_dir),
            "biomt_copy_matrix": [list(row) for row in BIOMT_TETRADECAMER_COPY],
        },
        "stats_csv": str(csv_path),
        "rows": rows,
    }
    summary_path = output_dir / "1ss8_groel_model_assembly_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("validation") / "protein_1ss8_wk_density_meshes_sigma6",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation") / "protein_1ss8_groel_models_sigma6",
    )
    parser.add_argument("--threshold-fractions", type=parse_float_list, default=[0.10, 0.30, 0.50, 0.70, 0.90])
    parser.add_argument("--suffixes", nargs="+", default=["original", "quadric_r80"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = assemble_models(
        mesh_dir=args.mesh_dir,
        output_dir=args.output_dir,
        fractions=args.threshold_fractions,
        suffixes=args.suffixes,
    )
    print(json.dumps({"summary": str(Path(summary["output_dir"]) / "1ss8_groel_model_assembly_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
