from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".runtime_cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PDB_PATH = ROOT / "public_data" / "CXIDB_ID_187" / "1SS8.pdb"
OUT_DIR = Path(__file__).resolve().parent
IMG_DIR = OUT_DIR / "images"
DATA_DIR = OUT_DIR / "data"


def parse_ca_atoms(path: Path) -> list[dict[str, object]]:
    atoms: list[dict[str, object]] = []
    with path.open("r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            element = line[76:78].strip() or atom_name[0]
            if element.upper() == "H":
                continue
            atoms.append(
                {
                    "chain": line[21].strip() or "?",
                    "residue": line[17:20].strip(),
                    "residue_index": int(line[22:26]),
                    "x_A": float(line[30:38]),
                    "y_A": float(line[38:46]),
                    "z_A": float(line[46:54]),
                    "copy": "original",
                }
            )
    return atoms


def write_ca_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = ["copy", "chain", "residue", "residue_index", "x_A", "y_A", "z_A"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_array(rows: list[dict[str, object]]) -> np.ndarray:
    return np.array([[float(r["x_A"]), float(r["y_A"]), float(r["z_A"])] for r in rows], dtype=float)


def centered(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    return points - center[None, :]


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
        zorder=20,
    )


def plot_view(ax, rows: list[dict[str, object]], center: np.ndarray, axes: tuple[int, int]) -> None:
    original = [r for r in rows if r["copy"] == "original"]
    biomt = [r for r in rows if r["copy"] == "biomt_copy"]
    colors = plt.get_cmap("tab10")

    for chain_index, chain in enumerate(sorted({str(r["chain"]) for r in original})):
        chain_rows = [r for r in original if r["chain"] == chain]
        pts = centered(as_array(chain_rows), center)
        order = np.argsort([int(r["residue_index"]) for r in chain_rows])
        pts = pts[order]
        ax.plot(pts[:, axes[0]], pts[:, axes[1]], color=colors(chain_index % 10), linewidth=1.1, alpha=0.9)
        ax.scatter(pts[:, axes[0]], pts[:, axes[1]], s=5, color=colors(chain_index % 10), alpha=0.85)

    pts_copy = centered(as_array(biomt), center)
    ax.scatter(pts_copy[:, axes[0]], pts_copy[:, axes[1]], s=4, color="#c77722", alpha=0.34, label="BIOMT copy")
    ax.set_xlabel(("x", "y", "z")[axes[0]] + " (A)")
    ax.set_ylabel(("x", "y", "z")[axes[1]] + " (A)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e6e6e6", linewidth=0.6)


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    original = parse_ca_atoms(PDB_PATH)
    biomt = []
    for row in original:
        copied = dict(row)
        copied["copy"] = "biomt_copy"
        copied["y_A"] = -float(row["y_A"])
        copied["z_A"] = -float(row["z_A"])
        biomt.append(copied)
    assembled = original + biomt
    center = as_array(assembled).mean(axis=0)

    write_ca_csv(assembled, DATA_DIR / "groel_1ss8_ca_original_and_biomt_copy.csv")

    summary = {
        "source_pdb": str(PDB_PATH),
        "ca_atoms_original": len(original),
        "ca_atoms_biomt_copy": len(biomt),
        "biomt_transform": "(x, y, z) -> (x, -y, -z)",
        "purpose": "Supplementary structural preview for the PDB-derived GroEL density/mesh model.",
    }
    (DATA_DIR / "figure_S01_groel_structure_preview_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
    plot_view(axes[0], assembled, center, (0, 2))
    plot_view(axes[1], assembled, center, (0, 1))
    plot_view(axes[2], assembled, center, (1, 2))
    for idx, ax in enumerate(axes):
        ax.tick_params(labelsize=9)
        add_panel_label(ax, f"({chr(ord('a') + idx)})")

    fig.savefig(IMG_DIR / "figure_S01_groel_structure_preview.png", dpi=300)
    fig.savefig(IMG_DIR / "figure_S01_groel_structure_preview.pdf")


if __name__ == "__main__":
    main()
