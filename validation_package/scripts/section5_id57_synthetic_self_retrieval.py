#!/usr/bin/env python
"""Synthetic self-retrieval for ID57 with symmetry-aware angular errors.

Noisy synthetic patterns are retrieved against the clean synthetic template
library. The known template quaternion is used as ground truth.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def build_annular_mask(shape: tuple[int, int], inner_radius_px: float, outer_radius_px: float) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float32)
    center_y = (shape[0] - 1.0) * 0.5
    center_x = (shape[1] - 1.0) * 0.5
    radius = np.hypot(x - center_x, y - center_y)
    return (radius >= float(inner_radius_px)) & (radius <= float(outer_radius_px))


def zscore_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return np.divide(arr - mean, std, out=np.zeros_like(arr), where=std > 1.0e-12)


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std())
    if std < 1.0e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def score_query_against_templates(y_raw: np.ndarray, x_raw: np.ndarray, sum_x: np.ndarray, sum_xx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = float(y_raw.size)
    sum_y = float(y_raw.sum())
    sum_xy = x_raw @ y_raw
    denom = n * sum_xx - sum_x * sum_x
    scale = np.divide(n * sum_xy - sum_x * sum_y, denom, out=np.zeros_like(sum_x), where=np.abs(denom) > 1.0e-12)
    background = (sum_y - scale * sum_x) / n
    scale = np.maximum(scale, 0.0)
    background = np.maximum(background, 0.0)
    fitted = np.clip(scale[:, np.newaxis] * x_raw + background[:, np.newaxis], 0.0, None)
    sim_log_z = zscore_rows(np.log1p(fitted))
    y_log_z = zscore(np.log1p(y_raw))
    scores = (sim_log_z @ y_log_z) / float(y_log_z.size)
    residual = np.sqrt(np.mean((sim_log_z - y_log_z[np.newaxis, :]) ** 2, axis=1))
    return scores, scale, residual


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        idx = int(np.argmax(np.diag(m)))
        if idx == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif idx == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q


def octahedral_symmetry_quaternions() -> np.ndarray:
    rows = []
    for perm in itertools.permutations(range(3)):
        base = np.zeros((3, 3), dtype=np.float64)
        for col, row in enumerate(perm):
            base[row, col] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mat = base @ np.diag(signs)
            if round(np.linalg.det(mat)) == 1:
                rows.append(matrix_to_quaternion_wxyz(mat))
    unique = []
    for q in rows:
        if not any(np.allclose(np.abs(np.dot(q, prev)), 1.0, atol=1e-7) for prev in unique):
            unique.append(q)
    return np.stack(unique, axis=0)


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[1:] *= -1.0
    return out


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    qa = np.asarray(a, dtype=np.float64)
    qb = np.asarray(b, dtype=np.float64)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * math.acos(dot)))


def symmetry_aware_angle_deg(q_true: np.ndarray, q_pred: np.ndarray, symmetry: np.ndarray) -> float:
    q_true = np.asarray(q_true, dtype=np.float64)
    q_pred = np.asarray(q_pred, dtype=np.float64)
    q_true /= np.linalg.norm(q_true)
    q_pred /= np.linalg.norm(q_pred)
    best = 180.0
    inv_pred = quaternion_conjugate(q_pred)
    for sym in symmetry:
        equivalent_true = quaternion_multiply(q_true, sym)
        relative = quaternion_multiply(inv_pred, equivalent_true)
        dot = float(np.clip(abs(relative[0]), -1.0, 1.0))
        angle = float(np.degrees(2.0 * math.acos(dot)))
        best = min(best, angle)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-dir", type=Path, required=True)
    parser.add_argument("--patterns-name", default="synthetic_id57_patterns.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-radius-px", type=float, default=45.0)
    parser.add_argument("--outer-radius-px", type=float, default=190.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.synthetic_dir / args.patterns_name) as data:
        clean = data["clean_patterns"].astype(np.float32)
        noisy = data["noisy_patterns"].astype(np.float32)
        valid_mask = data["valid_mask"].astype(bool)
        quaternions = data["quaternions_wxyz"].astype(np.float64)

    roi_mask = valid_mask & build_annular_mask(valid_mask.shape, args.inner_radius_px, args.outer_radius_px)
    x_raw = np.clip(clean[:, roi_mask].astype(np.float64), 0.0, None)
    sum_x = x_raw.sum(axis=1)
    sum_xx = np.einsum("ij,ij->i", x_raw, x_raw)
    symmetry = octahedral_symmetry_quaternions()

    rows = []
    for query_index in range(noisy.shape[0]):
        y = np.clip(noisy[query_index, roi_mask].astype(np.float64), 0.0, None)
        scores, scales, residual = score_query_against_templates(y, x_raw, sum_x, sum_xx)
        best_index = int(np.argmax(scores))
        rows.append(
            {
                "query_index": int(query_index),
                "retrieved_index": best_index,
                "is_exact_index": bool(best_index == query_index),
                "score_log_corr": float(scores[best_index]),
                "second_score_log_corr": float(np.partition(scores, -2)[-2]),
                "score_margin": float(scores[best_index] - np.partition(scores, -2)[-2]),
                "residual_log_z_rms": float(residual[best_index]),
                "scale": float(scales[best_index]),
                "raw_quaternion_error_deg": quaternion_angle_deg(quaternions[query_index], quaternions[best_index]),
                "octahedral_symmetry_error_deg": symmetry_aware_angle_deg(quaternions[query_index], quaternions[best_index], symmetry),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "id57_synthetic_self_retrieval.csv", index=False)
    summary = {
        "synthetic_dir": str(args.synthetic_dir),
        "query": "noisy_patterns",
        "library": "clean_patterns",
        "n_queries": int(len(frame)),
        "roi_pixels": int(roi_mask.sum()),
        "symmetry": "proper octahedral/cubic group, 24 rotations",
        "exact_index_accuracy": float(frame["is_exact_index"].mean()),
        "score_log_corr": {
            "median": float(frame["score_log_corr"].median()),
            "min": float(frame["score_log_corr"].min()),
            "max": float(frame["score_log_corr"].max()),
        },
        "score_margin": {
            "median": float(frame["score_margin"].median()),
            "min": float(frame["score_margin"].min()),
            "max": float(frame["score_margin"].max()),
        },
        "raw_quaternion_error_deg": {
            "median": float(frame["raw_quaternion_error_deg"].median()),
            "p95": float(frame["raw_quaternion_error_deg"].quantile(0.95)),
            "max": float(frame["raw_quaternion_error_deg"].max()),
        },
        "octahedral_symmetry_error_deg": {
            "median": float(frame["octahedral_symmetry_error_deg"].median()),
            "p95": float(frame["octahedral_symmetry_error_deg"].quantile(0.95)),
            "max": float(frame["octahedral_symmetry_error_deg"].max()),
        },
    }
    (args.output_dir / "id57_synthetic_self_retrieval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mplconfig = args.output_dir / "mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=170, constrained_layout=True)
    axes[0].hist(frame["score_log_corr"], bins=24, alpha=0.8)
    axes[0].set_xlabel("best score_log_corr")
    axes[0].set_ylabel("query count")
    axes[0].set_title(f"exact index {summary['exact_index_accuracy']:.1%}")
    axes[1].hist(frame["score_margin"], bins=24, alpha=0.8, color="tab:green")
    axes[1].set_xlabel("best - second score")
    axes[1].set_ylabel("query count")
    axes[2].hist(frame["octahedral_symmetry_error_deg"], bins=24, alpha=0.8, color="tab:orange")
    axes[2].set_xlabel("symmetry-aware angular error (deg)")
    axes[2].set_ylabel("query count")
    fig.suptitle("ID57 synthetic noisy-to-clean self retrieval")
    fig.savefig(args.output_dir / "id57_synthetic_self_retrieval_summary.png")
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
