#!/usr/bin/env python
"""Process CXIDB ID 187 GroEL run 151 into validation-ready assembled hits.

This is a compact local adaptation of the Ekeberg2022GroEL processing flow for
one run:

1. dark mean from r0149
2. approximate gain map from histograms.h5
3. lit-pixel scan for r0151
4. export top hit photon patterns
5. assemble PNCCD modules to 1024 x 1024 and optionally downsample to 128 x 128

It avoids writing full raw/common-mode intermediate arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np


PNCCD_DATASET = "INSTRUMENT/SQS_NQS_PNCCD1MP/CAL/PNCCD_FMT-0:output/data/image"


def raw_files(base_dir: Path, run: int) -> List[Path]:
    run_dir = base_dir / f"r{run:04d}"
    files = sorted(run_dir.glob(f"RAW-R{run:04d}-PNCCD01-S*.h5"))
    if not files:
        raise FileNotFoundError(f"no raw files found in {run_dir}")
    return files


def read_image_count(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        return int(h5[PNCCD_DATASET].shape[0])


def file_frame_offsets(files: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([read_image_count(p) for p in files], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return counts, offsets


def compute_dark_mean(files: Sequence[Path], output_path: Path, chunk_size: int) -> np.ndarray:
    if output_path.exists():
        with h5py.File(output_path, "r") as h5:
            return h5["dark"][...].astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = np.zeros((1024, 1024), dtype=np.float64)
    nimages = 0
    for path in files:
        print(f"dark: {path.name}", flush=True)
        with h5py.File(path, "r") as h5:
            data = h5[PNCCD_DATASET]
            for start in range(0, data.shape[0], chunk_size):
                stop = min(start + chunk_size, data.shape[0])
                total += data[start:stop].sum(axis=0, dtype=np.float64)
                nimages += stop - start
    dark = (total / float(nimages)).astype(np.float32)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("dark", data=dark, compression="lzf")
        h5.create_dataset("nimages", data=int(nimages))
    return dark


def weighted_peak_center(hist: np.ndarray, bins: np.ndarray, baseline_quantile: float = 0.10) -> np.ndarray:
    h = hist.astype(np.float64, copy=False)
    baseline = np.quantile(h, baseline_quantile, axis=2, keepdims=True)
    weights = np.clip(h - baseline, 0.0, None)
    denom = weights.sum(axis=2)
    numerator = np.tensordot(weights, bins.astype(np.float64), axes=([2], [0]))
    centers = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0.0)
    fallback = bins[np.argmax(h, axis=2)]
    centers[denom <= 0.0] = fallback[denom <= 0.0]
    return centers.astype(np.float32)


def compute_gain_map(histograms_path: Path, mask_path: Path, output_path: Path) -> Dict[str, np.ndarray]:
    if output_path.exists():
        with h5py.File(output_path, "r") as h5:
            return {
                "peak_0_pos": h5["peak_0_pos"][...].astype(np.float32),
                "peak_1_pos": h5["peak_1_pos"][...].astype(np.float32),
                "adu_per_photon": h5["adu_per_photon"][...].astype(np.float32),
                "mask": h5["mask"][...].astype(bool),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(histograms_path, "r") as h5:
        bins0 = h5["bin_centers_0"][...]
        bins1 = h5["bin_centers_1"][...]
        hist0 = h5["histograms_0"][...]
        hist1 = h5["histograms_1"][...]
    peak0 = weighted_peak_center(hist0, bins0)
    peak1 = weighted_peak_center(hist1, bins1)
    adu = peak1 - peak0
    with h5py.File(mask_path, "r") as h5:
        cxidb_mask = h5["mask"][...].astype(bool)
    good_gain = (adu > 3500.0) & (adu < 6500.0) & np.isfinite(adu)
    mask = cxidb_mask & good_gain
    adu_safe = adu.astype(np.float32)
    median_adu = float(np.median(adu_safe[mask]))
    adu_safe[~good_gain] = median_adu
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("peak_0_pos", data=peak0, compression="lzf")
        h5.create_dataset("peak_1_pos", data=peak1, compression="lzf")
        h5.create_dataset("adu_per_photon", data=adu_safe, compression="lzf")
        h5.create_dataset("mask", data=mask, compression="lzf")
        h5.attrs["method"] = "weighted histogram peak centers, no scipy curve_fit"
        h5.attrs["median_adu_per_photon"] = median_adu
    return {"peak_0_pos": peak0, "peak_1_pos": peak1, "adu_per_photon": adu_safe, "mask": mask}


def common_mode_and_photon_convert(raw: np.ndarray, dark: np.ndarray, peak0: np.ndarray, adu: np.ndarray, mask: np.ndarray) -> np.ndarray:
    patterns = raw.astype(np.float32) - dark[np.newaxis, :, :]
    reshaped = patterns.reshape((patterns.shape[0], 2, 512, 1024))
    med = np.median(reshaped, axis=2).astype(np.float32)
    patterns_cm = (reshaped - med[:, :, np.newaxis, :]).reshape(patterns.shape)
    ph = patterns_cm
    ph -= peak0[np.newaxis, :, :]
    ph /= adu[np.newaxis, :, :]
    ph[:, ~mask] = 0.0
    ph[ph < 0.75] = 0.0
    np.round(ph, out=ph)
    ph[:, ~mask] = 0.0
    return ph.astype(np.float32, copy=False)


def scan_lit_pixels(
    files: Sequence[Path],
    dark: np.ndarray,
    gain: Dict[str, np.ndarray],
    output_path: Path,
    chunk_size: int,
) -> np.ndarray:
    if output_path.exists():
        with h5py.File(output_path, "r") as h5:
            return h5["lit_pixels"][...].astype(np.int32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lit_chunks: List[np.ndarray] = []
    for path in files:
        print(f"lit: {path.name}", flush=True)
        with h5py.File(path, "r") as h5:
            data = h5[PNCCD_DATASET]
            for start in range(0, data.shape[0], chunk_size):
                stop = min(start + chunk_size, data.shape[0])
                ph = common_mode_and_photon_convert(data[start:stop], dark, gain["peak_0_pos"], gain["adu_per_photon"], gain["mask"])
                lit_chunks.append((ph > 2.0).sum(axis=(1, 2)).astype(np.int32))
    lit = np.concatenate(lit_chunks)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("lit_pixels", data=lit, compression="lzf")
    return lit


def scan_lit_pixels_range(
    files: Sequence[Path],
    dark: np.ndarray,
    gain: Dict[str, np.ndarray],
    output_path: Path,
    chunk_size: int,
    *,
    file_start: int,
    file_stop: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path
    lit_chunks: List[np.ndarray] = []
    counts, offsets = file_frame_offsets(files)
    for file_index in range(file_start, file_stop):
        path = files[file_index]
        print(f"lit range: file {file_index}/{len(files)} {path.name}", flush=True)
        with h5py.File(path, "r") as h5:
            data = h5[PNCCD_DATASET]
            for start in range(0, data.shape[0], chunk_size):
                stop = min(start + chunk_size, data.shape[0])
                ph = common_mode_and_photon_convert(data[start:stop], dark, gain["peak_0_pos"], gain["adu_per_photon"], gain["mask"])
                lit_chunks.append((ph > 2.0).sum(axis=(1, 2)).astype(np.int32))
    lit = np.concatenate(lit_chunks) if lit_chunks else np.zeros(0, dtype=np.int32)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("lit_pixels", data=lit, compression="lzf")
        h5.create_dataset("file_counts", data=counts[file_start:file_stop].astype(np.int64))
        h5.attrs["file_start"] = int(file_start)
        h5.attrs["file_stop"] = int(file_stop)
        h5.attrs["global_frame_start"] = int(offsets[file_start])
        h5.attrs["global_frame_stop"] = int(offsets[file_stop - 1] + counts[file_stop - 1])
    return output_path


def combine_lit_pixel_chunks(files: Sequence[Path], chunks_dir: Path, output_path: Path) -> np.ndarray:
    if output_path.exists():
        with h5py.File(output_path, "r") as h5:
            return h5["lit_pixels"][...].astype(np.int32)
    counts, _ = file_frame_offsets(files)
    total = int(counts.sum())
    lit = np.full(total, -1, dtype=np.int32)
    chunk_files = sorted(chunks_dir.glob("lit_pixels_r0151_files_*.h5"))
    if not chunk_files:
        raise FileNotFoundError(f"no lit-pixel chunks found in {chunks_dir}")
    for path in chunk_files:
        with h5py.File(path, "r") as h5:
            start = int(h5.attrs["global_frame_start"])
            stop = int(h5.attrs["global_frame_stop"])
            values = h5["lit_pixels"][...].astype(np.int32)
        if stop - start != values.size:
            raise ValueError(f"chunk length mismatch: {path}")
        lit[start:stop] = values
    missing = int(np.count_nonzero(lit < 0))
    if missing:
        raise RuntimeError(f"missing lit-pixel values for {missing} frames; run more chunks first")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("lit_pixels", data=lit, compression="lzf")
    return lit


def assemble(patterns: np.ndarray) -> np.ndarray:
    c = [536, 530]
    if patterns.ndim == 3:
        panel_1 = patterns[:, c[0] - 512 : 512, 8:]
        panel_2 = patterns[:, 512 : -(512 + 49 - c[0]), (c[1] - 512) :]
        out = np.zeros((patterns.shape[0], 1024, 1024), dtype=patterns.dtype)
        out[:, : panel_1.shape[1], : panel_1.shape[2]] = panel_1
        out[:, -panel_2.shape[1] :, : panel_2.shape[2]] = panel_2
        return out
    panel_1 = patterns[c[0] - 512 : 512, 8:]
    panel_2 = patterns[512 : -(512 + 49 - c[0]), (c[1] - 512) :]
    out = np.zeros((1024, 1024), dtype=patterns.dtype)
    out[: panel_1.shape[0], : panel_1.shape[1]] = panel_1
    out[-panel_2.shape[0] :, : panel_2.shape[1]] = panel_2
    return out


def downsample_sum(patterns: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return patterns
    return patterns.reshape(patterns.shape[0], patterns.shape[1] // factor, factor, patterns.shape[2] // factor, factor).sum(axis=(2, 4))


def downsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return mask.astype(bool)
    return mask.reshape(mask.shape[0] // factor, factor, mask.shape[1] // factor, factor).sum(axis=(1, 3)) == factor * factor


def append_dataset(dataset: h5py.Dataset, data: np.ndarray) -> None:
    old = dataset.shape[0]
    dataset.resize((old + data.shape[0],) + dataset.shape[1:])
    dataset[old:] = data


def export_hits(
    files: Sequence[Path],
    dark: np.ndarray,
    gain: Dict[str, np.ndarray],
    lit_pixels: np.ndarray,
    output_path: Path,
    *,
    percentile: float,
    max_hits: int,
    chunk_size: int,
    downsample_factor: int,
    exclude_frame_indices: Sequence[int],
) -> Dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = float(np.percentile(lit_pixels, percentile))
    hit_indices = np.flatnonzero(lit_pixels > threshold)
    if exclude_frame_indices:
        excluded = set(int(v) for v in exclude_frame_indices)
        hit_indices = np.asarray([int(idx) for idx in hit_indices if int(idx) not in excluded], dtype=np.int64)
    if max_hits > 0 and hit_indices.size > max_hits:
        order = np.argsort(lit_pixels[hit_indices])[::-1][:max_hits]
        hit_indices = np.sort(hit_indices[order])
    hit_set = set(int(i) for i in hit_indices.tolist())

    assembled_mask = assemble(gain["mask"].astype(np.uint8)).astype(bool)
    assembled_mask_ds = downsample_mask(assembled_mask, downsample_factor)
    image_shape = (1024 // downsample_factor, 1024 // downsample_factor)
    dtype = np.uint16

    counts, offsets = file_frame_offsets(files)
    with h5py.File(output_path, "w") as h5:
        patterns_ds = h5.create_dataset(
            "patterns_ph",
            shape=(0,) + image_shape,
            maxshape=(None,) + image_shape,
            dtype=dtype,
            chunks=(1,) + image_shape,
            compression="lzf",
        )
        h5.create_dataset("mask", data=assembled_mask_ds, compression="lzf")
        h5.create_dataset("lit_pixels_all", data=lit_pixels.astype(np.int32), compression="lzf")
        h5.create_dataset("selected_frame_indices", data=hit_indices.astype(np.int64), compression="lzf")
        h5.create_dataset("selected_lit_pixels", data=lit_pixels[hit_indices].astype(np.int32), compression="lzf")
        h5.create_dataset("excluded_frame_indices", data=np.asarray(list(exclude_frame_indices), dtype=np.int64), compression="lzf")
        source_file_rows: List[int] = []
        source_local_rows: List[int] = []

        global_base = 0
        for file_index, path in enumerate(files):
            nframes = int(counts[file_index])
            file_hit_indices = [idx - global_base for idx in hit_set if global_base <= idx < global_base + nframes]
            file_hit_indices.sort()
            global_base += nframes
            if not file_hit_indices:
                continue
            print(f"export: {path.name} hits={len(file_hit_indices)}", flush=True)
            with h5py.File(path, "r") as raw_h5:
                data = raw_h5[PNCCD_DATASET]
                for start in range(0, len(file_hit_indices), chunk_size):
                    local = np.asarray(file_hit_indices[start : start + chunk_size], dtype=np.int64)
                    raw = data[local, :, :]
                    ph = common_mode_and_photon_convert(raw, dark, gain["peak_0_pos"], gain["adu_per_photon"], gain["mask"])
                    assembled = assemble(ph)
                    assembled *= assembled_mask[np.newaxis, :, :]
                    if downsample_factor > 1:
                        assembled = downsample_sum(assembled, downsample_factor)
                    assembled = np.clip(assembled, 0, np.iinfo(dtype).max).astype(dtype)
                    append_dataset(patterns_ds, assembled)
                    source_file_rows.extend([file_index] * len(local))
                    source_local_rows.extend(local.astype(int).tolist())
        h5.create_dataset("source_file_index", data=np.asarray(source_file_rows, dtype=np.int32), compression="lzf")
        h5.create_dataset("source_local_frame_index", data=np.asarray(source_local_rows, dtype=np.int32), compression="lzf")
        h5.attrs["lit_pixel_threshold"] = threshold
        h5.attrs["lit_pixel_percentile"] = float(percentile)
        h5.attrs["downsample_factor"] = int(downsample_factor)
        h5.attrs["source_run"] = 151
    return {
        "output_path": str(output_path),
        "threshold": threshold,
        "selected_hits": int(hit_indices.size),
        "excluded_frame_indices": [int(v) for v in exclude_frame_indices],
        "downsample_factor": int(downsample_factor),
        "image_shape": list(image_shape),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(r"E:\GroEL"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation") / "groel_run151_processed")
    parser.add_argument("--run", type=int, default=151)
    parser.add_argument("--dark-run", type=int, default=149)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--export-chunk-size", type=int, default=8)
    parser.add_argument("--hit-percentile", type=float, default=99.0)
    parser.add_argument("--max-hits", type=int, default=500)
    parser.add_argument("--downsample-factor", type=int, default=8)
    parser.add_argument("--exclude-frame-indices", default="")
    parser.add_argument("--stage", choices=("all", "dark", "gain", "lit-range", "combine-lit", "export"), default="all")
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-stop", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dark_files = raw_files(args.base_dir, args.dark_run)
    run_files = raw_files(args.base_dir, args.run)

    dark_path = args.output_dir / f"dark_r{args.dark_run:04d}.h5"
    gain_path = args.output_dir / "gain_map_approx.h5"
    lit_path = args.output_dir / f"lit_pixels_r{args.run:04d}.h5"
    hits_path = args.output_dir / "Results" / "AssembledHits" / f"hits_r{args.run:04d}_top{args.max_hits}_ds{args.downsample_factor}.h5"
    summary_path = args.output_dir / f"run{args.run:04d}_processing_summary.json"

    if args.force and args.stage in ("all", "dark"):
        for path in (dark_path, gain_path, lit_path, hits_path, summary_path):
            if path.exists():
                path.unlink()

    dark = compute_dark_mean(dark_files, dark_path, args.chunk_size)
    if args.stage == "dark":
        print(json.dumps({"dark_path": str(dark_path)}, indent=2), flush=True)
        return

    gain = compute_gain_map(args.base_dir / "histograms.h5", args.base_dir / "mask.h5", gain_path)
    if args.stage == "gain":
        print(json.dumps({"gain_path": str(gain_path)}, indent=2), flush=True)
        return

    lit_chunks_dir = args.output_dir / "lit_pixel_chunks"
    if args.stage == "lit-range":
        file_stop = len(run_files) if args.file_stop is None else min(int(args.file_stop), len(run_files))
        if args.file_start < 0 or args.file_start >= file_stop:
            raise ValueError("--file-start/--file-stop select no files")
        chunk_path = lit_chunks_dir / f"lit_pixels_r{args.run:04d}_files_{args.file_start:03d}_{file_stop:03d}.h5"
        path = scan_lit_pixels_range(
            run_files,
            dark,
            gain,
            chunk_path,
            args.chunk_size,
            file_start=int(args.file_start),
            file_stop=int(file_stop),
        )
        print(json.dumps({"lit_chunk_path": str(path)}, indent=2), flush=True)
        return

    if args.stage == "combine-lit":
        lit = combine_lit_pixel_chunks(run_files, lit_chunks_dir, lit_path)
        print(json.dumps({"lit_path": str(lit_path), "frames": int(lit.size)}, indent=2), flush=True)
        return

    if args.stage == "all":
        lit = scan_lit_pixels(run_files, dark, gain, lit_path, args.chunk_size)
    else:
        lit = combine_lit_pixel_chunks(run_files, lit_chunks_dir, lit_path)

    export_summary = export_hits(
        run_files,
        dark,
        gain,
        lit,
        hits_path,
        percentile=args.hit_percentile,
        max_hits=args.max_hits,
        chunk_size=args.export_chunk_size,
        downsample_factor=args.downsample_factor,
        exclude_frame_indices=[int(x.strip()) for x in args.exclude_frame_indices.split(",") if x.strip()],
    )

    summary = {
        "base_dir": str(args.base_dir),
        "run": int(args.run),
        "dark_run": int(args.dark_run),
        "dark_files": len(dark_files),
        "run_files": len(run_files),
        "dark_path": str(dark_path),
        "gain_path": str(gain_path),
        "lit_path": str(lit_path),
        "hits": export_summary,
        "lit_pixels": {
            "count": int(lit.size),
            "min": int(lit.min()),
            "median": float(np.median(lit)),
            "p95": float(np.percentile(lit, 95)),
            "p99": float(np.percentile(lit, 99)),
            "max": int(lit.max()),
        },
        "notes": [
            "Gain map uses approximate histogram peak centers because scipy curve_fit is not installed in the local validation venv.",
            "Output contains downsampled assembled photon-count hits for GroEL template retrieval.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
