#!/usr/bin/env python
"""Generate GroEL template libraries for later experimental retrieval.

The output HDF5 is intentionally compatible with chunked generation. A full
25,680-orientation library can be generated in multiple files by varying
``--rotation-start``/``--rotation-stop``/``--rotation-step``; the retrieval
script can scan one or more template files without merging them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np

import section5_generate_groel_model_ab_preview as groel


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_model_list(value: str) -> List[str]:
    aliases = {
        "A": "A_heptamer_C7",
        "B": "B_tetradecamer_D7",
        "heptamer": "A_heptamer_C7",
        "tetradecamer": "B_tetradecamer_D7",
    }
    out = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        out.append(aliases.get(item, item))
    return out


def choose_rotation_indices(
    rotations_count: int,
    *,
    rotation_indices: str | None,
    rotation_start: int,
    rotation_stop: int | None,
    rotation_step: int,
    max_rotations: int | None,
) -> np.ndarray:
    if rotation_indices:
        indices = np.asarray([int(x.strip()) for x in rotation_indices.split(",") if x.strip()], dtype=np.int64)
    else:
        stop = rotations_count if rotation_stop is None else min(int(rotation_stop), rotations_count)
        indices = np.arange(int(rotation_start), stop, int(rotation_step), dtype=np.int64)
    if max_rotations is not None:
        indices = indices[: int(max_rotations)]
    if indices.size == 0:
        raise ValueError("no rotations selected")
    if int(indices.min()) < 0 or int(indices.max()) >= rotations_count:
        raise ValueError(f"rotation index out of range for {rotations_count} rotations")
    return indices


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    out = np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )
    return out


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def cyclic_symmetry_quaternions(order: int, axis: str = "x") -> np.ndarray:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    rows = []
    for k in range(int(order)):
        theta = 2.0 * np.pi * k / float(order)
        q = np.zeros(4, dtype=np.float64)
        q[0] = np.cos(0.5 * theta)
        q[1 + axis_index] = np.sin(0.5 * theta)
        rows.append(q)
    return np.stack(rows, axis=0)


def d7_symmetry_quaternions() -> np.ndarray:
    rows = [q for q in cyclic_symmetry_quaternions(7, axis="x")]
    for k in range(7):
        phi = np.pi * k / 7.0
        rows.append(np.asarray([0.0, 0.0, np.cos(phi), np.sin(phi)], dtype=np.float64))
    return np.stack(rows, axis=0)


def symmetry_group_quaternions(name: str) -> np.ndarray:
    normalized = name.strip().upper()
    if normalized in ("NONE", "IDENTITY"):
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    if normalized == "C7":
        return cyclic_symmetry_quaternions(7, axis="x")
    if normalized == "D7":
        return d7_symmetry_quaternions()
    raise ValueError(f"unsupported symmetry: {name}")


def min_symmetry_distance_deg(candidate: np.ndarray, kept: np.ndarray, symmetry: np.ndarray) -> float:
    if kept.size == 0:
        return 180.0
    cand = np.asarray(candidate, dtype=np.float64)
    cand /= np.linalg.norm(cand)
    kept_norm = np.asarray(kept, dtype=np.float64)
    kept_norm = kept_norm / np.linalg.norm(kept_norm, axis=1, keepdims=True)
    rel_left = quaternion_multiply(quaternion_conjugate(kept_norm), cand[np.newaxis, :])
    rel = quaternion_multiply(rel_left[:, np.newaxis, :], symmetry[np.newaxis, :, :])
    dots = np.clip(np.abs(rel[..., 0]), -1.0, 1.0)
    best_dot = float(np.max(dots))
    return float(np.degrees(2.0 * np.arccos(best_dot)))


def apply_symmetry_min_separation(
    rotations: np.ndarray,
    indices: np.ndarray,
    *,
    symmetry_name: str,
    min_separation_deg: float,
    max_kept: int | None,
) -> np.ndarray:
    if min_separation_deg <= 0.0 or symmetry_name.strip().upper() in ("NONE", "IDENTITY"):
        return indices[: int(max_kept)] if max_kept is not None else indices
    symmetry = symmetry_group_quaternions(symmetry_name)
    kept_indices: List[int] = []
    kept_quaternions: List[np.ndarray] = []
    for index in indices:
        q = np.asarray(rotations[int(index)], dtype=np.float64)
        distance = min_symmetry_distance_deg(
            q,
            np.asarray(kept_quaternions, dtype=np.float64) if kept_quaternions else np.zeros((0, 4), dtype=np.float64),
            symmetry,
        )
        if distance >= float(min_separation_deg):
            kept_indices.append(int(index))
            kept_quaternions.append(q)
            if max_kept is not None and len(kept_indices) >= int(max_kept):
                break
    if not kept_indices:
        raise RuntimeError("symmetry filtering removed every rotation")
    return np.asarray(kept_indices, dtype=np.int64)


def model_specs_from_args(args: argparse.Namespace) -> List[Tuple[str, Dict[str, float | int]]]:
    specs: List[Tuple[str, Dict[str, float | int]]] = []
    for model in parse_model_list(args.models):
        specs.append((model, {}))
    if args.include_prism_models:
        for radius in args.prism_radii_A:
            label_radius = f"{float(radius):g}".replace(".", "p")
            specs.append(
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
    return specs


def build_model_points(
    density_points: groel.DensityPoints,
    model_name: str,
    prism_meta: Dict[str, float | int],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float | int]]:
    if model_name.startswith("B_tetradecamer_D7_polyprism"):
        protein_coords, protein_weights, model_b_center = groel.make_model_points(density_points, "B_tetradecamer_D7")
        coords, weights, meta = groel.add_polygonal_prism_to_model_b(
            points=density_points,
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
        return coords, weights, meta
    coords, weights, _ = groel.make_model_points(density_points, model_name)
    return coords, weights, {}


def build_annular_mask(image_size: int, inner_radius_px: float, outer_radius_px: float) -> np.ndarray:
    y, x = np.indices((image_size, image_size), dtype=np.float32)
    center = image_size / 2.0 - 0.5
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    return (r >= float(inner_radius_px)) & (r <= float(outer_radius_px))


def write_parameters_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--density-npz",
        type=Path,
        default=Path("validation") / "protein_1ss8_wk_density_meshes_sigma6" / "1ss8_wk_density_grid.npz",
    )
    parser.add_argument("--rotations-h5", type=Path, default=Path("public_data") / "CXIDB_ID_187" / "rotations.h5")
    parser.add_argument("--output-dir", type=Path, default=Path("validation") / "groel_template_library")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--models", default="A,B")
    parser.add_argument("--include-prism-models", action="store_true")
    parser.add_argument("--prism-radii-A", type=parse_float_list, default=[35.0, 55.0])
    parser.add_argument("--prism-length-A", type=float, default=135.0)
    parser.add_argument("--prism-sides", type=int, default=14)
    parser.add_argument("--prism-spacing-A", type=float, default=2.0)
    parser.add_argument("--prism-density-e-A3", type=float, default=0.267)
    parser.add_argument("--prism-exclude-protein-density-fraction", type=float, default=0.20)
    parser.add_argument("--density-min-fraction", type=float, default=1e-4)
    parser.add_argument("--projection-pixel-size-A", type=float, default=1.0)
    parser.add_argument("--projection-margin-A", type=float, default=18.0)
    parser.add_argument("--fft-size", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--photon-energy-eV", type=float, default=1200.0)
    parser.add_argument("--detector-distance-m", type=float, default=0.15)
    parser.add_argument("--native-pixel-size-m", type=float, default=75e-6)
    parser.add_argument("--inner-radius-px", type=float, default=5.0)
    parser.add_argument("--outer-radius-px", type=float, default=40.0)
    parser.add_argument("--rotation-indices", default=None)
    parser.add_argument("--rotation-start", type=int, default=0)
    parser.add_argument("--rotation-stop", type=int, default=None)
    parser.add_argument("--rotation-step", type=int, default=1)
    parser.add_argument("--max-rotations", type=int, default=None)
    parser.add_argument("--symmetry", choices=("none", "C7", "D7"), default="none")
    parser.add_argument("--symmetry-min-separation-deg", type=float, default=0.0)
    parser.add_argument("--symmetry-max-rotations", type=int, default=None)
    parser.add_argument("--compression", choices=("none", "gzip", "lzf"), default="lzf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    density_points = groel.load_density_points(args.density_npz, args.density_min_fraction)
    with h5py.File(args.rotations_h5, "r") as f:
        rotations = np.asarray(f["rotations"][...], dtype=np.float32)
    selected_indices = choose_rotation_indices(
        len(rotations),
        rotation_indices=args.rotation_indices,
        rotation_start=args.rotation_start,
        rotation_stop=args.rotation_stop,
        rotation_step=args.rotation_step,
        max_rotations=args.max_rotations,
    )
    selected_indices_before_symmetry = selected_indices.copy()
    selected_indices = apply_symmetry_min_separation(
        rotations,
        selected_indices,
        symmetry_name=args.symmetry,
        min_separation_deg=args.symmetry_min_separation_deg,
        max_kept=args.symmetry_max_rotations,
    )
    selected_rotations = rotations[selected_indices]
    model_specs = model_specs_from_args(args)

    total_templates = len(model_specs) * len(selected_indices)
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = (
            f"groel_templates_rot{int(selected_indices[0]):06d}-"
            f"{int(selected_indices[-1]):06d}_n{len(selected_indices):05d}_m{len(model_specs):02d}.h5"
        )
    output_path = args.output_dir / output_name
    parameters_path = output_path.with_suffix(".csv")

    qx, qy, qr = groel.detector_q_grid(
        image_size=args.image_size,
        downsample=args.downsample,
        photon_energy_eV=args.photon_energy_eV,
        detector_distance_m=args.detector_distance_m,
        native_pixel_size_m=args.native_pixel_size_m,
    )
    annular_mask = build_annular_mask(args.image_size, args.inner_radius_px, args.outer_radius_px)

    compression = None if args.compression == "none" else args.compression
    string_dtype = h5py.string_dtype(encoding="utf-8")
    rows: List[Dict[str, object]] = []
    model_metadata: Dict[str, Dict[str, float | int]] = {}

    with h5py.File(output_path, "w") as h5:
        templates_ds = h5.create_dataset(
            "templates",
            shape=(total_templates, args.image_size, args.image_size),
            dtype="f4",
            chunks=(1, args.image_size, args.image_size),
            compression=compression,
        )
        rotations_ds = h5.create_dataset("rotations", shape=(total_templates, 4), dtype="f4")
        rotation_indices_ds = h5.create_dataset("rotation_indices", shape=(total_templates,), dtype="i8")
        model_names_ds = h5.create_dataset("model_names", shape=(total_templates,), dtype=string_dtype)
        model_ids_ds = h5.create_dataset("model_ids", shape=(total_templates,), dtype="i4")
        h5.create_dataset("qx_Ainv", data=qx.astype(np.float32))
        h5.create_dataset("qy_Ainv", data=qy.astype(np.float32))
        h5.create_dataset("q_radius_Ainv", data=qr.astype(np.float32))
        h5.create_dataset("annular_mask", data=annular_mask.astype(bool))
        h5.create_dataset("selected_rotation_indices", data=selected_indices.astype(np.int64))
        h5.create_dataset("selected_rotations", data=selected_rotations.astype(np.float32))

        template_index = 0
        for model_id, (model_name, prism_meta) in enumerate(model_specs):
            coords, weights, meta = build_model_points(density_points, model_name, prism_meta)
            model_metadata[model_name] = meta
            print(
                f"{model_name}: points={len(weights)} charge={weights.sum(dtype=np.float64):.3f} e",
                flush=True,
            )
            for local_rotation_index, rotation_index in enumerate(selected_indices):
                rotation = groel.quaternion_wxyz_to_matrix(rotations[int(rotation_index)])
                projection, projection_pixel_size_A = groel.project_points(
                    coords=coords,
                    weights=weights,
                    rotation=rotation,
                    pixel_size_A=args.projection_pixel_size_A,
                    margin_A=args.projection_margin_A,
                )
                intensity = groel.diffraction_from_projection(
                    projection=projection,
                    pixel_size_A=projection_pixel_size_A,
                    qx=qx,
                    qy=qy,
                    fft_size=args.fft_size,
                )
                templates_ds[template_index] = intensity.astype(np.float32)
                rotations_ds[template_index] = rotations[int(rotation_index)]
                rotation_indices_ds[template_index] = int(rotation_index)
                model_names_ds[template_index] = model_name
                model_ids_ds[template_index] = model_id
                rows.append(
                    {
                        "template_index": template_index,
                        "model_id": model_id,
                        "model_name": model_name,
                        "rotation_index": int(rotation_index),
                        "rotation_local_index": int(local_rotation_index),
                        "qw": float(rotations[int(rotation_index), 0]),
                        "qx": float(rotations[int(rotation_index), 1]),
                        "qy": float(rotations[int(rotation_index), 2]),
                        "qz": float(rotations[int(rotation_index), 3]),
                        "projection_shape_y": int(projection.shape[0]),
                        "projection_shape_x": int(projection.shape[1]),
                    }
                )
                if template_index % 25 == 0:
                    print(
                        f"  template {template_index + 1}/{total_templates}: "
                        f"{model_name} rot={int(rotation_index)}",
                        flush=True,
                    )
                template_index += 1

        metadata = {
            "purpose": "GroEL template library for later experimental retrieval.",
            "density_npz": str(args.density_npz),
            "rotations_h5": str(args.rotations_h5),
            "output_path": str(output_path),
            "parameters_csv": str(parameters_path),
            "density_min_fraction": float(args.density_min_fraction),
            "retained_charge_fraction": float(density_points.retained_charge_e / density_points.total_charge_e),
            "models": [{"id": idx, "name": name, "metadata": model_metadata.get(name, {})} for idx, (name, _) in enumerate(model_specs)],
            "template_count": int(total_templates),
            "rotation_count": int(len(selected_indices)),
            "rotation_count_before_symmetry": int(len(selected_indices_before_symmetry)),
            "selected_rotation_first": int(selected_indices[0]),
            "selected_rotation_last": int(selected_indices[-1]),
            "symmetry_sampling": {
                "symmetry": args.symmetry,
                "min_separation_deg": float(args.symmetry_min_separation_deg),
                "max_rotations": int(args.symmetry_max_rotations) if args.symmetry_max_rotations is not None else None,
            },
            "detector": {
                "image_size": int(args.image_size),
                "downsample": int(args.downsample),
                "photon_energy_eV": float(args.photon_energy_eV),
                "detector_distance_m": float(args.detector_distance_m),
                "native_pixel_size_m": float(args.native_pixel_size_m),
                "q_max_Ainv": float(qr.max()),
            },
            "roi": {
                "inner_radius_px": float(args.inner_radius_px),
                "outer_radius_px": float(args.outer_radius_px),
                "annular_pixels": int(annular_mask.sum()),
            },
            "projection": {
                "projection_pixel_size_A": float(args.projection_pixel_size_A),
                "projection_margin_A": float(args.projection_margin_A),
                "fft_size": int(args.fft_size),
            },
        }
        h5.attrs["metadata_json"] = json.dumps(metadata)

    write_parameters_csv(parameters_path, rows)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
