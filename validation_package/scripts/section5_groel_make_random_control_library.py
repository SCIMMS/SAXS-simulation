#!/usr/bin/env python
"""Build a radial-profile-preserving random control template library.

The control is intended for GroEL retrieval significance checks. It keeps the
same template count, detector grid, annular mask, and per-template radial
intensity envelope as a real library, but destroys orientation-dependent angular
structure by randomly permuting pixels within radius bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def radius_bin_map(shape: tuple[int, int], bin_width_px: float) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    center_y = (shape[0] - 1.0) * 0.5
    center_x = (shape[1] - 1.0) * 0.5
    radius = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return np.floor(radius / float(bin_width_px)).astype(np.int32)


def radial_scramble_template(
    template: np.ndarray,
    bins: np.ndarray,
    active_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(template, dtype=np.float32).copy()
    active_bins = np.unique(bins[active_mask])
    for bin_id in active_bins:
        members = np.flatnonzero((bins == int(bin_id)) & active_mask)
        if members.size < 2:
            continue
        values = out.ravel()[members].copy()
        rng.shuffle(values)
        out.ravel()[members] = values
    return out


def decode_h5_string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-library", type=Path, required=True)
    parser.add_argument("--output-library", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--bin-width-px", type=float, default=1.0)
    parser.add_argument(
        "--active-mask",
        choices=("annular", "all-valid"),
        default="annular",
        help="Pixels to scramble. annular is stricter for the retrieval ROI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_library.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    with h5py.File(args.input_library, "r") as src:
        templates = src["templates"][...].astype(np.float32)
        annular_mask = src["annular_mask"][...].astype(bool)
        if templates.ndim != 3:
            raise ValueError(f"expected templates with shape (n, y, x), got {templates.shape}")
        image_shape = templates.shape[1:]
        active_mask = annular_mask if args.active_mask == "annular" else np.ones(image_shape, dtype=bool)
        bins = radius_bin_map(image_shape, args.bin_width_px)

        randomized = np.empty_like(templates, dtype=np.float32)
        for idx in range(templates.shape[0]):
            randomized[idx] = radial_scramble_template(templates[idx], bins, active_mask, rng)

        model_names = np.asarray(
            [f"radial_random_control::{decode_h5_string(name)}".encode("utf-8") for name in src["model_names"][...]],
            dtype=object,
        )

        with h5py.File(args.output_library, "w") as out:
            for key in src.keys():
                if key == "templates":
                    out.create_dataset("templates", data=randomized, compression="gzip", compression_opts=4, shuffle=True)
                elif key == "model_names":
                    out.create_dataset("model_names", data=model_names, dtype=h5py.string_dtype("utf-8"))
                else:
                    data = src[key][...]
                    if data.dtype.kind in ("f", "i", "u", "b"):
                        out.create_dataset(key, data=data, compression="gzip", compression_opts=4, shuffle=True)
                    else:
                        out.create_dataset(key, data=data)

            source_meta = {}
            raw_meta = src.attrs.get("metadata_json")
            if raw_meta is not None:
                if isinstance(raw_meta, bytes):
                    raw_meta = raw_meta.decode("utf-8")
                try:
                    source_meta = json.loads(str(raw_meta))
                except json.JSONDecodeError:
                    source_meta = {"raw_metadata_json": str(raw_meta)}
            metadata = {
                "purpose": "Radial-profile-preserving random control library for GroEL retrieval.",
                "input_library": str(args.input_library),
                "output_library": str(args.output_library),
                "randomization": "pixel permutation within radius bins",
                "seed": int(args.seed),
                "bin_width_px": float(args.bin_width_px),
                "active_mask": args.active_mask,
                "template_count": int(templates.shape[0]),
                "image_shape": list(map(int, image_shape)),
                "source_metadata": source_meta,
            }
            out.attrs["metadata_json"] = json.dumps(metadata)

    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
