#!/usr/bin/env python
"""Estimate runtime for atom-centered Gaussian protein density grids.

The intended production model uses Waasmaier-Kirfel element form factors. For
timing, the expensive part is independent of exact coefficient values: each atom
contributes several local Gaussian stencils clipped at n*sigma_eff. This probe
therefore runs the same local-stencil accumulation pattern on atom subsets and
extrapolates to all non-hydrogen atoms and five WK Gaussian terms per atom.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class AtomTable:
    xyz: np.ndarray
    element: np.ndarray
    occupancy: np.ndarray
    bfactor: np.ndarray


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
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            try:
                occ = float(line[54:60])
            except ValueError:
                occ = 1.0
            try:
                b = float(line[60:66])
            except ValueError:
                b = 0.0
            xyz.append((x, y, z))
            elements.append(element)
            occupancy.append(occ)
            bfactor.append(b)

    return AtomTable(
        xyz=np.asarray(xyz, dtype=np.float32),
        element=np.asarray(elements),
        occupancy=np.asarray(occupancy, dtype=np.float32),
        bfactor=np.asarray(bfactor, dtype=np.float32),
    )


def gaussian_stencil_radius_voxels(sigma_angstrom: float, clip_sigma: float, spacing: float) -> int:
    return int(math.ceil(float(sigma_angstrom) * float(clip_sigma) / float(spacing)))


def accumulate_subset_density(
    atoms_xyz: np.ndarray,
    origin: np.ndarray,
    shape: Tuple[int, int, int],
    spacing: float,
    sigma_eff: float,
    clip_sigma: float,
    terms_per_atom: int,
) -> Dict[str, float]:
    """Accumulate a WK-like multi-Gaussian density for a subset of atoms."""
    density = np.zeros(shape, dtype=np.float32)
    radius = gaussian_stencil_radius_voxels(sigma_eff, clip_sigma, spacing)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32) * np.float32(spacing)
    gx1 = np.exp(-0.5 * (offsets / np.float32(sigma_eff)) ** 2).astype(np.float32)

    touched_voxels = 0
    skipped_terms = 0
    t0 = time.perf_counter()
    for atom in atoms_xyz:
        center_f = (atom - origin) / np.float32(spacing)
        cx, cy, cz = [int(round(float(v))) for v in center_f]

        x0 = max(0, cx - radius)
        x1 = min(shape[0], cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(shape[1], cy + radius + 1)
        z0 = max(0, cz - radius)
        z1 = min(shape[2], cz + radius + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            skipped_terms += terms_per_atom
            continue

        sx0 = x0 - (cx - radius)
        sy0 = y0 - (cy - radius)
        sz0 = z0 - (cz - radius)
        gx = gx1[sx0 : sx0 + (x1 - x0)]
        gy = gx1[sy0 : sy0 + (y1 - y0)]
        gz = gx1[sz0 : sz0 + (z1 - z0)]
        kernel = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        touched_voxels += int(kernel.size) * int(terms_per_atom)

        # Repeat the same stencil add pattern five times to emulate WK terms.
        # We intentionally keep weights simple because they do not control cost.
        for _ in range(int(terms_per_atom)):
            density[x0:x1, y0:y1, z0:z1] += kernel

    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": float(elapsed),
        "radius_voxels": int(radius),
        "stencil_width_voxels": int(2 * radius + 1),
        "touched_voxels": int(touched_voxels),
        "skipped_terms": int(skipped_terms),
        "density_sum": float(density.sum()),
    }


def grid_shape_for_atoms(
    atoms_xyz: np.ndarray,
    spacing: float,
    margin_angstrom: float,
) -> Tuple[np.ndarray, Tuple[int, int, int], np.ndarray, np.ndarray]:
    xyz_min = atoms_xyz.min(axis=0)
    xyz_max = atoms_xyz.max(axis=0)
    origin = xyz_min - np.float32(margin_angstrom)
    extent = xyz_max - xyz_min + np.float32(2.0 * margin_angstrom)
    shape = tuple(int(math.ceil(float(v) / float(spacing))) + 1 for v in extent)
    return origin.astype(np.float32), shape, xyz_min.astype(np.float32), xyz_max.astype(np.float32)


def estimate_cases(
    atoms: AtomTable,
    spacings: Iterable[float],
    sigma_effs: Iterable[float],
    clip_sigmas: Iterable[float],
    subset_atoms: int,
    repeats: int,
    margin_extra: float,
    terms_per_atom: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(12345)
    n_atoms = int(atoms.xyz.shape[0])
    subset_n = min(int(subset_atoms), n_atoms)
    subset_idx = rng.choice(n_atoms, size=subset_n, replace=False)
    subset_xyz = atoms.xyz[subset_idx]

    element_counts = {str(k): int(v) for k, v in zip(*np.unique(atoms.element, return_counts=True))}
    bfactor_stats = {
        "min": float(np.nanmin(atoms.bfactor)),
        "median": float(np.nanmedian(atoms.bfactor)),
        "mean": float(np.nanmean(atoms.bfactor)),
        "max": float(np.nanmax(atoms.bfactor)),
    }

    for spacing in spacings:
        for sigma_eff in sigma_effs:
            margin = float(margin_extra) + float(clip_sigmas[-1]) * float(sigma_eff)
            origin, shape, xyz_min, xyz_max = grid_shape_for_atoms(atoms.xyz, spacing, margin)
            grid_voxels = int(np.prod(shape))
            grid_mb = grid_voxels * np.dtype(np.float32).itemsize / (1024.0 * 1024.0)

            for clip_sigma in clip_sigmas:
                repeat_times = []
                repeat_touched = []
                radius = gaussian_stencil_radius_voxels(sigma_eff, clip_sigma, spacing)
                for _ in range(int(repeats)):
                    result = accumulate_subset_density(
                        subset_xyz,
                        origin=origin,
                        shape=shape,
                        spacing=float(spacing),
                        sigma_eff=float(sigma_eff),
                        clip_sigma=float(clip_sigma),
                        terms_per_atom=int(terms_per_atom),
                    )
                    repeat_times.append(result["elapsed_s"])
                    repeat_touched.append(result["touched_voxels"])

                median_elapsed = float(np.median(repeat_times))
                touched = int(np.median(repeat_touched))
                terms_subset = subset_n * int(terms_per_atom)
                terms_full = n_atoms * int(terms_per_atom)
                seconds_per_term = median_elapsed / max(terms_subset, 1)
                seconds_per_touched_voxel = median_elapsed / max(touched, 1)
                estimated_full_s_by_terms = seconds_per_term * terms_full
                estimated_full_s_by_touched = seconds_per_touched_voxel * touched * (n_atoms / subset_n)
                rows.append(
                    {
                        "spacing_A": float(spacing),
                        "sigma_eff_A": float(sigma_eff),
                        "clip_sigma": float(clip_sigma),
                        "shape_x": int(shape[0]),
                        "shape_y": int(shape[1]),
                        "shape_z": int(shape[2]),
                        "grid_voxels": grid_voxels,
                        "grid_float32_MB": float(grid_mb),
                        "radius_voxels": int(radius),
                        "stencil_width_voxels": int(2 * radius + 1),
                        "subset_atoms": int(subset_n),
                        "total_atoms": int(n_atoms),
                        "terms_per_atom": int(terms_per_atom),
                        "repeat_median_s": median_elapsed,
                        "repeat_min_s": float(np.min(repeat_times)),
                        "repeat_max_s": float(np.max(repeat_times)),
                        "estimated_full_s_by_terms": float(estimated_full_s_by_terms),
                        "estimated_full_s_by_touched": float(estimated_full_s_by_touched),
                        "estimated_full_min": float(estimated_full_s_by_touched / 60.0),
                    }
                )

    summary = {
        "total_atoms": int(n_atoms),
        "element_counts": element_counts,
        "bfactor_stats": bfactor_stats,
        "xyz_min_A": [float(x) for x in atoms.xyz.min(axis=0)],
        "xyz_max_A": [float(x) for x in atoms.xyz.max(axis=0)],
        "subset_atoms": int(subset_n),
        "terms_per_atom": int(terms_per_atom),
        "note": "Timing emulates five WK local Gaussian stencils per atom; exact WK weights do not affect runtime materially.",
    }
    return rows, summary


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, default=Path(r"C:\Users\tnvlaos456\Downloads\1SS8.pdb"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation") / "protein_density_timing_probe")
    parser.add_argument("--spacings", type=parse_float_list, default=[1.5, 2.0, 2.5])
    parser.add_argument("--sigma-effs", type=parse_float_list, default=[1.5, 2.5, 4.0])
    parser.add_argument("--clip-sigmas", type=parse_float_list, default=[2.5, 3.0])
    parser.add_argument("--subset-atoms", type=int, default=600)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--margin-extra", type=float, default=4.0)
    parser.add_argument("--terms-per-atom", type=int, default=5)
    parser.add_argument("--include-hydrogen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atoms = parse_pdb_atoms(args.pdb, include_hydrogen=bool(args.include_hydrogen))
    if atoms.xyz.size == 0:
        raise RuntimeError(f"No atoms parsed from {args.pdb}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = estimate_cases(
        atoms,
        spacings=args.spacings,
        sigma_effs=args.sigma_effs,
        clip_sigmas=args.clip_sigmas,
        subset_atoms=int(args.subset_atoms),
        repeats=int(args.repeats),
        margin_extra=float(args.margin_extra),
        terms_per_atom=int(args.terms_per_atom),
    )

    csv_path = args.output_dir / "protein_density_timing_probe.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary["csv"] = str(csv_path)
    summary["cases"] = rows
    json_path = args.output_dir / "protein_density_timing_probe_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("total_atoms", "element_counts", "xyz_min_A", "xyz_max_A", "csv")}, indent=2))


if __name__ == "__main__":
    main()
