"""Build a reusable, symmetry-reduced rotation library for CXIDB ID 57."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from section5_orientation_utils import (
    deduplicate_by_octahedral_symmetry,
    normal_incidence_orientation_table,
    octahedral_group_matrices,
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "index",
        "halton_index",
        "tilt_deg",
        "azimuth_deg",
        "roll_deg",
        "qw",
        "qx",
        "qy",
        "qz",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            q = row["quaternion_wxyz"]
            writer.writerow(
                {
                    "index": row["index"],
                    "halton_index": row["halton_index"],
                    "tilt_deg": row["tilt_deg"],
                    "azimuth_deg": row["azimuth_deg"],
                    "roll_deg": row["roll_deg"],
                    "qw": float(q[0]),
                    "qx": float(q[1]),
                    "qy": float(q[2]),
                    "qz": float(q[3]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public_data" / "CXIDB_ID_57" / "rotation_libraries",
    )
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--tilt-max-deg", type=float, default=8.0)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--min-separation-deg", type=float, default=0.05)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"id57_normal_incidence_n{args.count:04d}_tilt{args.tilt_max_deg:g}"
    npz_path = args.out_dir / f"{name}.npz"
    csv_path = args.out_dir / f"{name}.csv"
    json_path = args.out_dir / f"{name}_summary.json"

    candidate_count = max(args.count * 2, args.count + 32)
    rows = normal_incidence_orientation_table(candidate_count, args.tilt_max_deg, start_index=args.start_index)
    rows = deduplicate_by_octahedral_symmetry(rows, min_separation_deg=args.min_separation_deg)
    if len(rows) < args.count:
        raise RuntimeError(
            f"Only {len(rows)} unique orientations remained; reduce --min-separation-deg or increase candidate_count."
        )
    rows = rows[: args.count]
    for i, row in enumerate(rows):
        row["index"] = i

    rotation_matrices = np.stack([row["rotation_matrix"] for row in rows], axis=0).astype(np.float32)
    quaternions_wxyz = np.stack([row["quaternion_wxyz"] for row in rows], axis=0).astype(np.float32)
    tilt_deg = np.array([row["tilt_deg"] for row in rows], dtype=np.float32)
    azimuth_deg = np.array([row["azimuth_deg"] for row in rows], dtype=np.float32)
    roll_deg = np.array([row["roll_deg"] for row in rows], dtype=np.float32)
    halton_index = np.array([row["halton_index"] for row in rows], dtype=np.int32)

    np.savez_compressed(
        npz_path,
        rotation_matrices=rotation_matrices,
        quaternions_wxyz=quaternions_wxyz,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        roll_deg=roll_deg,
        halton_index=halton_index,
        octahedral_group_matrices=octahedral_group_matrices().astype(np.float32),
    )
    write_csv(csv_path, rows)

    summary = {
        "library_path": str(npz_path),
        "csv_path": str(csv_path),
        "count": int(rotation_matrices.shape[0]),
        "sampling": "Halton quasi-uniform spherical cap plus roll",
        "symmetry_reduction": "+z face representative; roll in [0, 90 deg); proper octahedral duplicate check",
        "tilt_max_deg": args.tilt_max_deg,
        "start_index": args.start_index,
        "min_separation_deg": args.min_separation_deg,
        "quaternion_convention": "wxyz",
        "rotation_matrix_convention": "object coordinates to lab coordinates",
        "octahedral_group_size": 24,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
