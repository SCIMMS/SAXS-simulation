#!/usr/bin/env python
"""Compare GroEL retrieval scores against a random control library."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def two_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n == 0:
        return 1.0
    k = min(int(wins), int(losses))
    probability = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return float(min(1.0, 2.0 * probability))


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
    required = {"source_frame_index", "logcorr", "fit_rms"}
    for name, frame in (("real", real), ("control", control)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")

    paired = real.merge(
        control,
        on="source_frame_index",
        suffixes=("_real", "_control"),
        validate="one_to_one",
    )
    paired["delta_logcorr"] = paired["logcorr_real"] - paired["logcorr_control"]
    paired["delta_fit_rms"] = paired["fit_rms_real"] - paired["fit_rms_control"]
    paired["real_better_logcorr"] = paired["delta_logcorr"] > 0.0
    paired["real_better_rms"] = paired["delta_fit_rms"] < 0.0
    paired = paired.sort_values("source_frame_index")
    paired.to_csv(args.output_dir / "groel_real_vs_random_control_paired.csv", index=False)

    wins_corr = int(paired["real_better_logcorr"].sum())
    losses_corr = int((paired["delta_logcorr"] < 0.0).sum())
    wins_rms = int(paired["real_better_rms"].sum())
    losses_rms = int((paired["delta_fit_rms"] > 0.0).sum())
    summary = {
        "n_paired": int(len(paired)),
        "real_logcorr": {
            "median": float(paired["logcorr_real"].median()),
            "mean": float(paired["logcorr_real"].mean()),
            "min": float(paired["logcorr_real"].min()),
            "max": float(paired["logcorr_real"].max()),
        },
        "control_logcorr": {
            "median": float(paired["logcorr_control"].median()),
            "mean": float(paired["logcorr_control"].mean()),
            "min": float(paired["logcorr_control"].min()),
            "max": float(paired["logcorr_control"].max()),
        },
        "delta_logcorr_real_minus_control": {
            "median": float(paired["delta_logcorr"].median()),
            "mean": float(paired["delta_logcorr"].mean()),
            "min": float(paired["delta_logcorr"].min()),
            "max": float(paired["delta_logcorr"].max()),
            "real_better_count": wins_corr,
            "control_better_count": losses_corr,
            "two_sided_sign_test_pvalue": two_sided_sign_test_pvalue(wins_corr, losses_corr),
        },
        "real_fit_rms": {
            "median": float(paired["fit_rms_real"].median()),
            "mean": float(paired["fit_rms_real"].mean()),
            "min": float(paired["fit_rms_real"].min()),
            "max": float(paired["fit_rms_real"].max()),
        },
        "control_fit_rms": {
            "median": float(paired["fit_rms_control"].median()),
            "mean": float(paired["fit_rms_control"].mean()),
            "min": float(paired["fit_rms_control"].min()),
            "max": float(paired["fit_rms_control"].max()),
        },
        "delta_fit_rms_real_minus_control": {
            "median": float(paired["delta_fit_rms"].median()),
            "mean": float(paired["delta_fit_rms"].mean()),
            "min": float(paired["delta_fit_rms"].min()),
            "max": float(paired["delta_fit_rms"].max()),
            "real_better_count": wins_rms,
            "control_better_count": losses_rms,
            "two_sided_sign_test_pvalue": two_sided_sign_test_pvalue(wins_rms, losses_rms),
        },
    }
    (args.output_dir / "groel_real_vs_random_control_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    import os

    mplconfig = args.output_dir / "mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=170, constrained_layout=True)
    bins = np.linspace(
        min(float(paired["logcorr_real"].min()), float(paired["logcorr_control"].min())),
        max(float(paired["logcorr_real"].max()), float(paired["logcorr_control"].max())),
        18,
    )
    axes[0].hist(paired["logcorr_control"], bins=bins, alpha=0.65, label="radial random control")
    axes[0].hist(paired["logcorr_real"], bins=bins, alpha=0.65, label="GroEL template")
    axes[0].set_xlabel("best logcorr")
    axes[0].set_ylabel("shot count")
    axes[0].legend(fontsize=8)

    axes[1].scatter(paired["logcorr_control"], paired["logcorr_real"], s=18, alpha=0.75)
    lo = min(float(paired["logcorr_control"].min()), float(paired["logcorr_real"].min()))
    hi = max(float(paired["logcorr_control"].max()), float(paired["logcorr_real"].max()))
    axes[1].plot([lo, hi], [lo, hi], color="black", lw=0.8)
    axes[1].set_xlabel("control logcorr")
    axes[1].set_ylabel("GroEL logcorr")
    axes[1].set_title(f"real better: {wins_corr}/{len(paired)}")

    axes[2].hist(paired["delta_logcorr"], bins=18, color="tab:green", alpha=0.75)
    axes[2].axvline(0.0, color="black", lw=0.8)
    axes[2].set_xlabel("logcorr(real) - logcorr(control)")
    axes[2].set_ylabel("shot count")
    axes[2].set_title(f"median delta {summary['delta_logcorr_real_minus_control']['median']:.4f}")

    fig.suptitle("GroEL template retrieval vs radial-profile random control")
    fig.savefig(args.output_dir / "groel_real_vs_random_control_logcorr.png")
    plt.close(fig)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
