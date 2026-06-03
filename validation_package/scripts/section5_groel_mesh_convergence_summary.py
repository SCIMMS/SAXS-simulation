#!/usr/bin/env python
"""Summarize GroEL mesh decimation and projection-convergence metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-stats-csv",
        type=Path,
        default=Path("validation") / "protein_1ss8_groel_models_sigma6" / "1ss8_groel_model_mesh_stats.csv",
    )
    parser.add_argument(
        "--projection-summary-json",
        type=Path,
        default=Path("validation")
        / "protein_1ss8_z_projection_validation_sigma6"
        / "1ss8_z_projection_validation_summary.json",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("validation") / "groel_mesh_convergence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh = pd.read_csv(args.mesh_stats_csv)
    original = mesh[mesh["mesh_suffix"] == "original"].copy()
    decimated = mesh[mesh["mesh_suffix"] == "quadric_r80"].copy()
    paired = original.merge(
        decimated,
        on=["model", "threshold_fraction", "threshold_label"],
        suffixes=("_original", "_quadric_r80"),
        validate="one_to_one",
    )
    paired["point_ratio_quadric_over_original"] = paired["points_quadric_r80"] / paired["points_original"]
    paired["cell_ratio_quadric_over_original"] = paired["cells_quadric_r80"] / paired["cells_original"]
    paired["relative_volume_error_quadric_minus_original"] = (
        paired["volume_A3_quadric_r80"] - paired["volume_A3_original"]
    ) / paired["volume_A3_original"]
    paired["component_delta"] = paired["components_quadric_r80"] - paired["components_original"]
    keep = [
        "model",
        "threshold_fraction",
        "threshold_label",
        "points_original",
        "points_quadric_r80",
        "point_ratio_quadric_over_original",
        "cells_original",
        "cells_quadric_r80",
        "cell_ratio_quadric_over_original",
        "volume_A3_original",
        "volume_A3_quadric_r80",
        "relative_volume_error_quadric_minus_original",
        "components_original",
        "components_quadric_r80",
        "component_delta",
    ]
    paired[keep].to_csv(args.output_dir / "groel_mesh_decimation_convergence.csv", index=False)

    projection_summary = json.loads(args.projection_summary_json.read_text(encoding="utf-8"))
    projection_rows = pd.DataFrame(projection_summary["metrics"])
    projection_rows.to_csv(args.output_dir / "groel_z_projection_metrics.csv", index=False)

    original_mid = projection_rows[projection_rows["method"] == "mesh_stencil_original_midpoint"].iloc[0]
    decimated_mid = projection_rows[projection_rows["method"] == "mesh_stencil_quadric_r80_midpoint"].iloc[0]
    original_lstsq = projection_rows[projection_rows["method"] == "mesh_stencil_original_lstsq"].iloc[0]
    decimated_lstsq = projection_rows[projection_rows["method"] == "mesh_stencil_quadric_r80_lstsq"].iloc[0]
    summary = {
        "mesh_stats_csv": str(args.mesh_stats_csv),
        "projection_summary_json": str(args.projection_summary_json),
        "decimation": {
            "rows": int(len(paired)),
            "median_point_ratio": float(paired["point_ratio_quadric_over_original"].median()),
            "median_cell_ratio": float(paired["cell_ratio_quadric_over_original"].median()),
            "max_abs_relative_volume_error": float(paired["relative_volume_error_quadric_minus_original"].abs().max()),
            "max_abs_component_delta": int(paired["component_delta"].abs().max()),
        },
        "projection_midpoint_original": {
            "relative_total_error": float(original_mid["relative_total_error"]),
            "relative_rmse_over_ref_rms": float(original_mid["relative_rmse_over_ref_rms"]),
            "pearson": float(original_mid["pearson"]),
            "cosine": float(original_mid["cosine"]),
        },
        "projection_midpoint_quadric_r80": {
            "relative_total_error": float(decimated_mid["relative_total_error"]),
            "relative_rmse_over_ref_rms": float(decimated_mid["relative_rmse_over_ref_rms"]),
            "pearson": float(decimated_mid["pearson"]),
            "cosine": float(decimated_mid["cosine"]),
        },
        "projection_lstsq_original": {
            "relative_total_error": float(original_lstsq["relative_total_error"]),
            "relative_rmse_over_ref_rms": float(original_lstsq["relative_rmse_over_ref_rms"]),
            "pearson": float(original_lstsq["pearson"]),
            "cosine": float(original_lstsq["cosine"]),
        },
        "projection_lstsq_quadric_r80": {
            "relative_total_error": float(decimated_lstsq["relative_total_error"]),
            "relative_rmse_over_ref_rms": float(decimated_lstsq["relative_rmse_over_ref_rms"]),
            "pearson": float(decimated_lstsq["pearson"]),
            "cosine": float(decimated_lstsq["cosine"]),
        },
    }
    (args.output_dir / "groel_mesh_convergence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mplconfig = args.output_dir / "mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=170, constrained_layout=True)
    for model, group in paired.groupby("model"):
        axes[0].plot(group["threshold_fraction"], group["point_ratio_quadric_over_original"], marker="o", label=model)
        axes[1].plot(group["threshold_fraction"], group["relative_volume_error_quadric_minus_original"], marker="o", label=model)
        axes[2].plot(group["threshold_fraction"], group["components_original"], marker="o", label=f"{model} original")
        axes[2].plot(group["threshold_fraction"], group["components_quadric_r80"], marker="x", linestyle="--", label=f"{model} decimated")
    axes[0].set_xlabel("threshold fraction")
    axes[0].set_ylabel("point ratio decimated/original")
    axes[0].set_ylim(0.0, 0.25)
    axes[1].set_xlabel("threshold fraction")
    axes[1].set_ylabel("relative volume error")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[2].set_xlabel("threshold fraction")
    axes[2].set_ylabel("connected components")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=6)
    fig.suptitle("GroEL mesh decimation convergence")
    fig.savefig(args.output_dir / "groel_mesh_decimation_convergence.png")
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
