#!/usr/bin/env python
"""Compare ID57 real-template retrieval scores against radial random controls."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import pandas as pd


def sign_test_pvalue(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    p = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * p))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-best-csv", type=Path, required=True)
    parser.add_argument("--control-best-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real = pd.read_csv(args.real_best_csv)
    control = pd.read_csv(args.control_best_csv)
    paired = real.merge(control, on="experimental_index", suffixes=("_real", "_control"), validate="one_to_one")
    paired["delta_score_log_corr"] = paired["score_log_corr_real"] - paired["score_log_corr_control"]
    paired["delta_residual_log_z_rms"] = paired["residual_log_z_rms_real"] - paired["residual_log_z_rms_control"]
    paired["real_better_corr"] = paired["delta_score_log_corr"] > 0.0
    paired["real_better_rms"] = paired["delta_residual_log_z_rms"] < 0.0
    paired = paired.sort_values("experimental_index")
    paired.to_csv(args.output_dir / "id57_real_vs_random_control_paired.csv", index=False)

    wins_corr = int(paired["real_better_corr"].sum())
    losses_corr = int((paired["delta_score_log_corr"] < 0.0).sum())
    wins_rms = int(paired["real_better_rms"].sum())
    losses_rms = int((paired["delta_residual_log_z_rms"] > 0.0).sum())
    summary = {
        "n_paired": int(len(paired)),
        "real_score_log_corr": {
            "median": float(paired["score_log_corr_real"].median()),
            "mean": float(paired["score_log_corr_real"].mean()),
            "min": float(paired["score_log_corr_real"].min()),
            "max": float(paired["score_log_corr_real"].max()),
        },
        "control_score_log_corr": {
            "median": float(paired["score_log_corr_control"].median()),
            "mean": float(paired["score_log_corr_control"].mean()),
            "min": float(paired["score_log_corr_control"].min()),
            "max": float(paired["score_log_corr_control"].max()),
        },
        "delta_score_log_corr_real_minus_control": {
            "median": float(paired["delta_score_log_corr"].median()),
            "mean": float(paired["delta_score_log_corr"].mean()),
            "min": float(paired["delta_score_log_corr"].min()),
            "max": float(paired["delta_score_log_corr"].max()),
            "real_better_count": wins_corr,
            "control_better_count": losses_corr,
            "two_sided_sign_test_pvalue": sign_test_pvalue(wins_corr, losses_corr),
        },
        "real_residual_log_z_rms": {
            "median": float(paired["residual_log_z_rms_real"].median()),
            "mean": float(paired["residual_log_z_rms_real"].mean()),
            "min": float(paired["residual_log_z_rms_real"].min()),
            "max": float(paired["residual_log_z_rms_real"].max()),
        },
        "control_residual_log_z_rms": {
            "median": float(paired["residual_log_z_rms_control"].median()),
            "mean": float(paired["residual_log_z_rms_control"].mean()),
            "min": float(paired["residual_log_z_rms_control"].min()),
            "max": float(paired["residual_log_z_rms_control"].max()),
        },
        "delta_residual_log_z_rms_real_minus_control": {
            "median": float(paired["delta_residual_log_z_rms"].median()),
            "mean": float(paired["delta_residual_log_z_rms"].mean()),
            "min": float(paired["delta_residual_log_z_rms"].min()),
            "max": float(paired["delta_residual_log_z_rms"].max()),
            "real_better_count": wins_rms,
            "control_better_count": losses_rms,
            "two_sided_sign_test_pvalue": sign_test_pvalue(wins_rms, losses_rms),
        },
    }
    (args.output_dir / "id57_real_vs_random_control_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    mplconfig = args.output_dir / "mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=170, constrained_layout=True)
    lo = min(float(paired["score_log_corr_real"].min()), float(paired["score_log_corr_control"].min()))
    hi = max(float(paired["score_log_corr_real"].max()), float(paired["score_log_corr_control"].max()))
    bins = pd.Series([lo, hi]).pipe(lambda s: __import__("numpy").linspace(s.iloc[0], s.iloc[1], 22))
    axes[0].hist(paired["score_log_corr_control"], bins=bins, alpha=0.65, label="radial random control")
    axes[0].hist(paired["score_log_corr_real"], bins=bins, alpha=0.65, label="ID57 template")
    axes[0].set_xlabel("best score_log_corr")
    axes[0].set_ylabel("shot count")
    axes[0].legend(fontsize=8)

    axes[1].scatter(paired["score_log_corr_control"], paired["score_log_corr_real"], s=16, alpha=0.7)
    axes[1].plot([lo, hi], [lo, hi], color="black", lw=0.8)
    axes[1].set_xlabel("control score")
    axes[1].set_ylabel("real score")
    axes[1].set_title(f"real better: {wins_corr}/{len(paired)}")

    axes[2].hist(paired["delta_score_log_corr"], bins=22, color="tab:green", alpha=0.75)
    axes[2].axvline(0.0, color="black", lw=0.8)
    axes[2].set_xlabel("score(real) - score(control)")
    axes[2].set_ylabel("shot count")
    axes[2].set_title(f"median delta {summary['delta_score_log_corr_real_minus_control']['median']:.3f}")
    fig.suptitle("ID57 retrieval vs radial-profile random control")
    fig.savefig(args.output_dir / "id57_real_vs_random_control_score.png")
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
