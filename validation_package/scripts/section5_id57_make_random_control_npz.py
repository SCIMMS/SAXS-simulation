#!/usr/bin/env python
"""Create an ID57 radial-profile-preserving random control NPZ library."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def build_radius_bins(shape: tuple[int, int], bin_width_px: float) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    center_y = (shape[0] - 1.0) * 0.5
    center_x = (shape[1] - 1.0) * 0.5
    radius = np.hypot(x - center_x, y - center_y)
    return np.floor(radius / float(bin_width_px)).astype(np.int32)


def build_annular_mask(shape: tuple[int, int], inner_radius_px: float, outer_radius_px: float) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    center_y = (shape[0] - 1.0) * 0.5
    center_x = (shape[1] - 1.0) * 0.5
    radius = np.hypot(x - center_x, y - center_y)
    return (radius >= float(inner_radius_px)) & (radius <= float(outer_radius_px))


def scramble_stack(
    stack: np.ndarray,
    bins: np.ndarray,
    active_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.asarray(stack, dtype=np.float32).copy()
    active_bins = np.unique(bins[active_mask])
    flat_bins = bins.ravel()
    flat_active = active_mask.ravel()
    for image_index in range(out.shape[0]):
        flat = out[image_index].ravel()
        for bin_id in active_bins:
            members = np.flatnonzero((flat_bins == int(bin_id)) & flat_active)
            if members.size < 2:
                continue
            values = flat[members].copy()
            rng.shuffle(values)
            flat[members] = values
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patterns-name", default="synthetic_id57_patterns.npz")
    parser.add_argument("--parameters-name", default="synthetic_id57_parameters.csv")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--bin-width-px", type=float, default=1.0)
    parser.add_argument("--inner-radius-px", type=float, default=45.0)
    parser.add_argument("--outer-radius-px", type=float, default=190.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    input_npz = args.input_dir / args.patterns_name
    output_npz = args.output_dir / args.patterns_name
    with np.load(input_npz) as source:
        clean = source["clean_patterns"].astype(np.float32)
        noisy = source["noisy_patterns"].astype(np.float32)
        valid_mask = source["valid_mask"].astype(bool)
        bins = build_radius_bins(clean.shape[1:], args.bin_width_px)
        roi_mask = valid_mask & build_annular_mask(clean.shape[1:], args.inner_radius_px, args.outer_radius_px)
        clean_random = scramble_stack(clean, bins, roi_mask, rng)
        noisy_random = scramble_stack(noisy, bins, roi_mask, rng)
        payload = {key: source[key] for key in source.files if key not in ("clean_patterns", "noisy_patterns")}
        payload["clean_patterns"] = clean_random
        payload["noisy_patterns"] = noisy_random
        np.savez_compressed(output_npz, **payload)

    params_src = args.input_dir / args.parameters_name
    if params_src.exists():
        shutil.copy2(params_src, args.output_dir / args.parameters_name)

    metadata = {
        "purpose": "ID57 radial-profile-preserving random control synthetic library.",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_npz": str(input_npz),
        "output_npz": str(output_npz),
        "seed": int(args.seed),
        "bin_width_px": float(args.bin_width_px),
        "active_mask": "valid pixels in annular retrieval ROI",
        "inner_radius_px": float(args.inner_radius_px),
        "outer_radius_px": float(args.outer_radius_px),
        "template_count": int(clean.shape[0]),
        "image_shape": list(map(int, clean.shape[1:])),
        "roi_pixels": int(roi_mask.sum()),
    }
    (args.output_dir / "random_control_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
