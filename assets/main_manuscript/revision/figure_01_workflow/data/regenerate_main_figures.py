"""Redraw the current main Figures 1, 3, 5 and 6 from frozen public inputs.

Requires Python, NumPy and Matplotlib. This is a plotting-only, CPU-only helper;
it does not fit models, select candidates, or require CuPy/VTK/private files.
The two Figure 5 records include a saved zero-frequency density projection.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os


def repository_root(explicit=None):
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "validation_package").is_dir():
            raise ValueError("--repo-root must contain validation_package/")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "validation_package").is_dir() and (root / "assets").is_dir():
            return root
    raise RuntimeError("Cannot find the repository; use --repo-root")


def main(default_figure=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure", type=int, choices=[1, 3, 5, 6], default=default_figure,
                        help="Select a figure; the shared helper alone defaults to all four")
    args = parser.parse_args()
    root = repository_root(args.repo_root)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "surface-mesh-projection",
                         "svg.fonttype": "none"})
    sources = {}
    outputs = []

    def source(relative):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required public input is missing: {relative}")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def save(fig, number, **kwargs):
        stem = f"figure_{number:02d}"
        for extension in ["png", "svg"]:
            path = output / f"{stem}.{extension}"
            metadata = {"Date": None} if extension == "svg" else None
            fig.savefig(path, dpi=600, metadata=metadata, **kwargs)
            outputs.append({"filename": path.name, "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)

    figures = [args.figure] if args.figure is not None else [1, 3, 5, 6]
    checks = {}
    for number in figures:
        if number == 1:
            fig, ax = plt.subplots(figsize=(10.5, 3.4))
            ax.axis("off")

            def box(x, y, w, h, label, color="#e5eef5"):
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.008",
                             fc=color, ec="#607d8b", transform=ax.transAxes))
                ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8.5,
                        transform=ax.transAxes)

            def arrow(x, y, a, b):
                ax.annotate("", xy=(a, b), xytext=(x, y), xycoords="axes fraction",
                            arrowprops={"arrowstyle": "->", "color": "#526572", "lw": 1.3})

            box(.01, .38, .21, .2, "Closed density boundaries\nPrescribed orientation")
            box(.32, .69, .23, .20, "K = 1, κ = 0\nReal projected density")
            box(.32, .13, .23, .24, "K longitudinal frequencies\nComplex weighted\nprojections")
            box(.64, .69, .23, .20, "2D CZT\nPlanar amplitude")
            box(.64, .13, .23, .24, "K × 2D CZT\nComplex interpolation\non Ewald")
            arrow(.23, .53, .31, .79); arrow(.23, .43, .31, .25)
            arrow(.56, .79, .63, .79); arrow(.56, .25, .63, .25)
            ax.text(.96, .48, "|A|²\n↓\nTemplate\ncomparison", ha="center", va="center",
                    fontsize=9, transform=ax.transAxes)
            arrow(.88, .79, .94, .58); arrow(.88, .25, .94, .38)
            save(fig, number, bbox_inches="tight")
            checks["figure_01"] = {"type": "workflow diagram", "numerical_inputs": "none"}
        elif number == 3:
            relative = "validation_package/revision_2026_10_17/data/output/complex_projection_2026-10-15/comparison_summary.csv"
            with source(relative).open(encoding="utf-8-sig", newline="") as f:
                bench = list(csv.DictReader(f))
            fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
            for radius, color, label in [(0, "#355f8d", "Sharp"),
                                         (4, "#d28336", "Rounded 4 nm"),
                                         (12, "#329479", "Rounded 12 nm")]:
                a = [next(r for r in bench if float(r["radius_nm"]) == radius and r["method"] == method)
                     for method in ["qz3", "qz5", "qz9"]]
                for ax, key in [(axs[0], "sweep_mean_intensity_relative_L2_percent"),
                                (axs[1], "sweep_max_intensity_relative_L2_percent")]:
                    ax.plot([3, 5, 9], [float(r[key]) for r in a], "o-", color=color, label=label, ms=4)
                b = [next(r for r in bench if float(r["radius_nm"]) == radius and r["method"] == method)
                     for method in ["resident_planar", "qz3", "qz5", "qz9"]]
                axs[2].plot(range(4), [float(r["median_ms_per_orientation"]) for r in b],
                            "o-", color=color, ms=4)
            axs[0].set_ylabel("Mean intensity relative L2 / %")
            axs[1].set_ylabel("Maximum intensity relative L2 / %")
            for ax in axs[:2]:
                ax.set_xlabel("Longitudinal planes K"); ax.set_xticks([3, 5, 9]); ax.set_yscale("log")
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.),
                       ncol=3, frameon=False, fontsize=8)
            axs[2].set_xticks(range(4), ["Planar", "3", "5", "9"])
            axs[2].set_xlabel("Longitudinal planes K")
            axs[2].set_ylabel("Warm time / ms per orientation")
            for ax, label in zip(axs, ["(a)", "(b)", "(c)"]):
                ax.text(.02, .98, label, transform=ax.transAxes, va="top", fontweight="bold")
                ax.grid(alpha=.15)
            fig.tight_layout(rect=(0, 0, 1, .89)); save(fig, number)
            checks["figure_03"] = {"radii_nm": [0, 4, 12], "planes": [3, 5, 9],
                                    "accuracy_points": 18, "timing_points": 12}
        elif number == 5:
            relative = "assets/main_manuscript/revision/figure_05_id57_five_plane_examples/data/selected_frames.npz"
            metadata_relative = "assets/main_manuscript/revision/figure_05_id57_five_plane_examples/data/selected_frames_provenance.json"
            metadata = json.loads(source(metadata_relative).read_text(encoding="utf-8"))
            path = source(relative)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["archive"]["sha256"]
            fig, axs = plt.subplots(2, 4, figsize=(10.3, 5.8))
            with np.load(path, allow_pickle=False) as z:
                mask = z["mask"]; qaxis = z["qaxis_nm_inv"]
                assert z["frame_indices"].tolist() == [11, 129]
                for row, frame in enumerate([11, 129]):
                    record = metadata["records"][row]
                    for key in ["measured_roi", "template_roi", "prediction_roi", "projection", "rotation"]:
                        a = z[f"frame_{frame}_{key}"]
                        assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() == record["arrays"][key]["sha256"]
                    y = z[f"frame_{frame}_measured_roi"]
                    prediction = z[f"frame_{frame}_prediction_roi"]
                    raw = z[f"frame_{frame}_template_roi"]
                    assert np.array_equal(prediction, record["signal_normalized"]*raw/record["train_template_mean"]+record["background"])
                    mats = []
                    for a in [np.log1p(y), np.log1p(prediction), np.log1p(y)-np.log1p(prediction)]:
                        image = np.full(mask.shape, np.nan); image[mask] = a; mats.append(image)
                    vmax = float(np.quantile(mats[0][mask], .995))
                    limit = float(np.quantile(np.abs(mats[2][mask]), .995))
                    for col in range(3):
                        im = axs[row, col].imshow(mats[col], origin="lower",
                            extent=[qaxis[0], qaxis[-1], qaxis[0], qaxis[-1]],
                            cmap="RdBu_r" if col == 2 else "viridis",
                            vmin=-limit if col == 2 else 0, vmax=limit if col == 2 else vmax)
                        fig.colorbar(im, ax=axs[row, col], shrink=.65, pad=.02)
                        axs[row, col].set_xlabel(r"$q_u$ / nm$^{-1}$")
                        axs[row, col].set_ylabel(r"$q_v$ / nm$^{-1}$")
                    projection = z[f"frame_{frame}_projection"]
                    scale = float(z[f"frame_{frame}_scale"])
                    im = axs[row, 3].imshow(projection*scale, origin="lower",
                         extent=np.array([-72, 72, -72, 72])*scale, cmap="magma", vmin=0)
                    axs[row, 3].set_xlim(-37, 37); axs[row, 3].set_ylim(-37, 37)
                    axs[row, 3].set_xlabel("u / nm"); axs[row, 3].set_ylabel("v / nm")
                    fig.colorbar(im, ax=axs[row, 3], shrink=.65, pad=.02)
                    for ax, title in zip(axs[row], ["Measured", "Five-plane model", "Log residual", "Projected density"]):
                        ax.set_title(("Frame "+str(frame)+"\n" if title == "Measured" else "")+title, fontsize=9)
            fig.tight_layout(); save(fig, number)
            checks["figure_05"] = {"frames": [11, 129], "candidates": [7032, 6127],
                                    "refitting": False, "arrays_and_fitted_prediction_verified": True}
        elif number == 6:
            relative = "validation_package/revision_CZT_R19_2026_10_17/data/output/groel_czt_primary_2026-10-17/pattern67_primary.npz"
            with np.load(source(relative), allow_pickle=False) as z:
                y = z["measured"]; roi = z["roi"]
                prediction = z["structured"]; projection = z["projection"]
            fig, axs = plt.subplots(2, 2, figsize=(8.36, 7.6))
            vmax = float(np.quantile(np.log1p(y[roi]), .995))
            residual = np.log1p(y)-np.log1p(prediction)
            limit = float(np.quantile(np.abs(residual[roi]), .995))
            panels = [(axs[0, 0], np.log1p(y), "Measured log1p intensity", "viridis", 0, vmax),
                      (axs[0, 1], np.log1p(prediction), "Mesh/CZT log1p intensity", "viridis", 0, vmax),
                      (axs[1, 1], residual, "Measured minus model log1p", "RdBu_r", -limit, limit)]
            for ax, a, title, cmap, vmin, vmax2 in panels:
                im = ax.imshow(np.where(roi, a, np.nan), origin="lower", extent=[-64, 64, -64, 64],
                               cmap=cmap, vmin=vmin, vmax=vmax2)
                ax.set_xlim(-42, 42); ax.set_ylim(-42, 42)
                ax.set_xlabel("Detector x / pixel"); ax.set_ylabel("Detector y / pixel")
                ax.set_title(title); fig.colorbar(im, ax=ax, shrink=.8, pad=.03)
            n = len(projection)
            im = axs[1, 0].imshow(projection, origin="lower", extent=[-n/2, n/2, -n/2, n/2],
                                   cmap="magma", vmin=0)
            axs[1, 0].set_xlabel("u / Å"); axs[1, 0].set_ylabel("v / Å")
            axs[1, 0].set_title("Projected mesh density / e Å⁻²")
            fig.colorbar(im, ax=axs[1, 0], shrink=.8, pad=.03)
            fig.tight_layout(); save(fig, number)
            checks["figure_06"] = {"pattern": 67, "forward_path": "mesh+CZT, planar", "refitting": False}
    audit = {"status": "pass", "purpose": "Plot-only regeneration from frozen public data",
             "png_dpi": 600, "svg": "Vector labels and axes; detector/projection panels retain native raster data.",
             "source_hashes": sources, "checks": checks, "outputs": outputs}
    audit_name = f"figure_{args.figure:02d}_regeneration_audit.json" if args.figure else "regeneration_audit.json"
    (output / audit_name).write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "figures": figures, "outputs": [x["filename"] for x in outputs]}))


if __name__ == "__main__":
    main()
