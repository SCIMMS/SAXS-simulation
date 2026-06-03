"""
Integrated SAXS simulator launcher with dropdown selector.
Usage:
    python gui_all_simulators_two_row_dropdown.py
Notes:
- Keep this file in the same folder as the individual gui_*.py files,
  saxs_module.py, SAXS_mask.npy, and post_beamtime_tmp_saxs.poni.
- Each simulator is loaded lazily when selected.
Display controls:
- Matplotlib toolbar is shown above each figure.
- Use Pan to move the view and Zoom to box-zoom.
- Mouse wheel over a plot zooms around the cursor without scrolling the parameter panel.
- Reset View restores the simulator-generated full-view axis limits.
"""
from __future__ import annotations
import importlib
import os
import sys
from pathlib import Path
import numpy as np
import csv
import json
import hashlib
import time
import itertools
import io
import math
import tempfile
import shutil
import re
import webbrowser

if getattr(sys, "frozen", False):
    BASE_DIR_BOOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
else:
    BASE_DIR_BOOT = Path(__file__).resolve().parent

_RUNTIME_CACHE_DIR = BASE_DIR_BOOT / ".runtime_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("CUPY_CACHE_DIR", str(_RUNTIME_CACHE_DIR / "cupy"))
for _cache_dir in (Path(os.environ["MPLCONFIGDIR"]), Path(os.environ["CUPY_CACHE_DIR"])):
    try:
        _cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

_DLL_DIR_HANDLES = []
if getattr(sys, "frozen", False):
    _nvidia_root = BASE_DIR_BOOT / "nvidia"
    if _nvidia_root.exists():
        for _bin_dir in sorted(_nvidia_root.glob("*/bin")):
            if _bin_dir.is_dir():
                try:
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(str(_bin_dir)))
                except (AttributeError, OSError):
                    pass
                os.environ["PATH"] = str(_bin_dir) + os.pathsep + os.environ.get("PATH", "")

import tkinter as tk
import zipfile
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg as _OriginalFigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
    ImageTk = None
BASE_DIR = BASE_DIR_BOOT
DEFAULT_POISSON_REFERENCE_NPZ = BASE_DIR / "cimg001.npz"
DEFAULT_POISSON_REFERENCE_PERCENTILE = 99.9
# Fixed from bundled cimg001.npz: strongest local image #262, p99.9 positive intensity.
# Keep this value constant so Poisson scaling does not reload or recompute the dataset.
DEFAULT_POISSON_REFERENCE_IMAGE_NUMBER = 262
DEFAULT_POISSON_REFERENCE_COUNT = 582.0265400390657
# Fixed simulation-side reference used to map arbitrary simulation intensities to
# the experimental count scale for Poisson noise.  This was calibrated once from
# the Ellipse 1 simulator with: b=1, object_scale=0.5, scale_factor=5000, and
# angle arbitrary (0 deg used numerically because the b=1 case is rotationally
# symmetric in-plane).  The value below is the p99.9 positive intensity of the
# resulting clean I_total image.
DEFAULT_SIMULATION_POISSON_REFERENCE_MODE = "fixed_ellipse1_b1_objectscale0.5_scalefactor5000_p99.9"
DEFAULT_SIMULATION_POISSON_REFERENCE_INTENSITY = 1.7760686539174396e20
DEFAULT_SIMULATION_POISSON_REFERENCE_METADATA = {
    "simulator": "Ellipse 1",
    "b": 1.0,
    "angle_deg": 0.0,
    "object_scale": 0.5,
    "scale_factor": 5000.0,
    "percentile": float(DEFAULT_POISSON_REFERENCE_PERCENTILE),
    "reference_intensity": float(DEFAULT_SIMULATION_POISSON_REFERENCE_INTENSITY),
    "mode": DEFAULT_SIMULATION_POISSON_REFERENCE_MODE,
}
class _HoverToolTip:
    """Small non-invasive tooltip used by Auto Screen threshold controls."""
    def __init__(self, widget, text, delay_ms=450, wraplength=460):
        self.widget = widget
        self.text = text
        self.delay_ms = int(delay_ms)
        self.wraplength = int(wraplength)
        self._after_id = None
        self._tip = None
        if widget is not None:
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")
            widget.bind("<Destroy>", self._hide, add="+")
    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)
    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
    def _show(self):
        self._after_id = None
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_pointerx() + 14
            y = self.widget.winfo_pointery() + 16
        except Exception:
            x = y = 0
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.attributes("-topmost", True)
        except Exception:
            pass
        tip.wm_geometry(f"+{int(x)}+{int(y)}")
        label = tk.Label(
            tip,
            text=self.text,
            justify="left",
            background="#fffde7",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            wraplength=self.wraplength,
            font=("Arial", 9),
        )
        label.pack()
        self._tip = tip
    def _hide(self, _event=None):
        self._cancel()
        tip = self._tip
        self._tip = None
        if tip is not None:
            try:
                tip.destroy()
            except Exception:
                pass
DEFAULT_1D_SCALE_FIT_Q_MIN = 0.005
DEFAULT_1D_SCALE_FIT_Q_MAX = 0.03
# Harmonic anisotropy profiles are deliberately evaluated on a slightly
# narrower q window than the full 2D/radial feature comparison. This avoids
# low-q beamstop/mask artifacts while keeping the physically informative
# SAXS fringe/anisotropy region.
DEFAULT_HARMONIC_Q_MIN = 0.007
DEFAULT_HARMONIC_Q_MAX = 0.03
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
# The simulator source files use relative paths for SAXS_mask.npy and .poni.
try:
    os.chdir(BASE_DIR)
except Exception:
    pass
PLOT_WIDGET_PATHS = set()
def _safe_filename_fragment(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "value"
def _format_param_value(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.6g}"
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"
def _inclusive_float_range(start, stop, step):
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step == 0:
        raise ValueError("Step must be non-zero.")
    if stop < start and step > 0:
        step = -step
    if stop > start and step < 0:
        step = -step
    values = []
    current = start
    tolerance = abs(step) * 1e-9 + 1e-12
    max_iter = 100000
    for _ in range(max_iter):
        values.append(float(current))
        current += step
        if step > 0 and current > stop + tolerance:
            break
        if step < 0 and current < stop - tolerance:
            break
    else:
        raise ValueError("Too many sweep points. Please use a larger step or narrower range.")
    if values and abs(values[-1] - stop) > tolerance:
        if (step > 0 and values[-1] < stop - tolerance) or (step < 0 and values[-1] > stop + tolerance):
            values.append(float(stop))
        elif abs(start - stop) <= tolerance:
            values = [float(start)]
    return values
class BatchSweepDialog(tk.Toplevel):
    def __init__(self, parent, simulator_name, app, root):
        super().__init__(parent)
        self.parent = parent
        self.simulator_name = simulator_name
        self.app = app
        self.root = root
        self.param_rows = {}
        self.export_vars = {}
        self.cancel_requested = False
        self.is_running = False
        self.title(f"Batch Auto Save - {simulator_name}")
        self.geometry("1040x840")
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.output_dir_var = tk.StringVar(value=str((BASE_DIR / "batch_outputs" / _safe_filename_fragment(simulator_name)).resolve()))
        self.prefix_var = tk.StringVar(value=_safe_filename_fragment(simulator_name))
        self.dpi_var = tk.StringVar(value="150")
        self.save_mode_var = tk.StringVar(value="individual")
        self.grid_size_var = tk.StringVar(value="3")
        self.save_manifest_var = tk.BooleanVar(value=True)
        self.progress_var = tk.StringVar(value="Set sweep ranges and click 'Run batch'.")
        self.total_combinations_var = tk.StringVar(value="Total combinations: 1")
        self._build_ui()
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        top = ttk.Frame(self, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text=f"Batch auto-save for: {self.simulator_name}", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(top, text="Output folder").grid(row=1, column=0, sticky="w", pady=(10, 4))
        ttk.Entry(top, textvariable=self.output_dir_var).grid(row=1, column=1, sticky="ew", pady=(10, 4), padx=(8, 8))
        ttk.Button(top, text="Browse...", command=self._choose_folder).grid(row=1, column=2, sticky="e", pady=(10, 4))
        ttk.Label(top, text="File prefix").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.prefix_var, width=24).grid(row=2, column=1, sticky="w", pady=4, padx=(8, 8))
        extra = ttk.Frame(top)
        extra.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Label(extra, text="DPI").pack(side=tk.LEFT)
        ttk.Entry(extra, textvariable=self.dpi_var, width=8).pack(side=tk.LEFT, padx=(6, 20))
        ttk.Label(extra, text="Save mode").pack(side=tk.LEFT)
        ttk.Radiobutton(extra, text="Individual files", variable=self.save_mode_var, value="individual").pack(side=tk.LEFT, padx=(6, 4))
        ttk.Radiobutton(extra, text="Square grid per image type", variable=self.save_mode_var, value="grid").pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(extra, text="Grid size (N×N)").pack(side=tk.LEFT)
        ttk.Entry(extra, textvariable=self.grid_size_var, width=6).pack(side=tk.LEFT, padx=(6, 20))
        ttk.Checkbutton(extra, text="Save CSV manifest", variable=self.save_manifest_var).pack(side=tk.LEFT)
        ttk.Label(extra, textvariable=self.total_combinations_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(24, 0))
        export_frame = ttk.LabelFrame(top, text="Images to save", padding=(8, 6))
        export_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        export_frame.columnconfigure(0, weight=1)
        export_options = self._get_export_options()
        for idx, option in enumerate(export_options):
            key = option["key"]
            label = option["label"]
            default = option["default"]
            var = tk.BooleanVar(value=default)
            self.export_vars[key] = {"var": var, "option": option}
            ttk.Checkbutton(export_frame, text=label, variable=var).grid(row=idx // 3, column=idx % 3, sticky="w", padx=(0, 18), pady=2)
        mid = ttk.Frame(self, padding=(12, 0, 12, 0))
        mid.grid(row=1, column=0, sticky="nsew")
        mid.columnconfigure(0, weight=1)
        mid.rowconfigure(0, weight=1)
        canvas = tk.Canvas(mid, highlightthickness=0)
        scrollbar = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        hdr = ttk.Frame(inner)
        hdr.pack(fill=tk.X, pady=(8, 6))
        headers = [("Use", 6), ("Parameter", 22), ("Current", 12), ("Start", 12), ("Stop", 12), ("Step", 12), ("Points", 10)]
        for idx, (label, width) in enumerate(headers):
            ttk.Label(hdr, text=label, width=width, font=("Arial", 10, "bold")).grid(row=0, column=idx, sticky="w", padx=2)
        vars_dict = getattr(self.app, "vars", {}) or {}
        control_specs = getattr(self.app, "control_specs", {}) or {}
        for name, var in vars_dict.items():
            try:
                current = float(var.get())
            except Exception:
                continue
            spec = control_specs.get(name, {})
            default_step = spec.get("step", None)
            if default_step in (None, 0, ""):
                span = abs(spec.get("max", current) - spec.get("min", current))
                default_step = span / 10.0 if span > 0 else 1.0
            default_step = float(default_step)
            use_var = tk.BooleanVar(value=False)
            start_var = tk.StringVar(value=_format_param_value(current))
            stop_var = tk.StringVar(value=_format_param_value(current))
            step_var = tk.StringVar(value=_format_param_value(default_step))
            points_var = tk.StringVar(value="1")
            frame = ttk.Frame(inner)
            frame.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(frame, variable=use_var, command=lambda n=name: self._update_points(n)).grid(row=0, column=0, sticky="w", padx=2)
            ttk.Label(frame, text=name, width=22).grid(row=0, column=1, sticky="w", padx=2)
            ttk.Label(frame, text=_format_param_value(current), width=12).grid(row=0, column=2, sticky="w", padx=2)
            ttk.Entry(frame, textvariable=start_var, width=12).grid(row=0, column=3, sticky="w", padx=2)
            ttk.Entry(frame, textvariable=stop_var, width=12).grid(row=0, column=4, sticky="w", padx=2)
            ttk.Entry(frame, textvariable=step_var, width=12).grid(row=0, column=5, sticky="w", padx=2)
            ttk.Label(frame, textvariable=points_var, width=10).grid(row=0, column=6, sticky="w", padx=2)
            for sv in (start_var, stop_var, step_var):
                sv.trace_add("write", lambda *_args, n=name: self._update_points(n))
            self.param_rows[name] = {
                "use": use_var,
                "start": start_var,
                "stop": stop_var,
                "step": step_var,
                "points": points_var,
                "current": current,
                "spec": spec,
            }
            self._update_points(name)
        self._update_total_combinations()
        bottom = ttk.Frame(self, padding=12)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.progress_var).grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(bottom)
        actions.grid(row=0, column=1, sticky="e")
        self.close_button = ttk.Button(actions, text="Close", command=self._on_close)
        self.close_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.request_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.run_button = ttk.Button(actions, text="Run batch", command=self.run_batch)
        self.run_button.pack(side=tk.RIGHT)
    def _get_export_options(self):
        options = [{"key": "full_figure", "label": "Full figure", "default": True, "kind": "full"}]
        fig = getattr(self.app, "fig", None)
        axes = list(getattr(fig, "axes", [])) if fig is not None else []
        panel_index = 1
        for ax in axes:
            if not getattr(ax, "get_visible", lambda: True)():
                continue
            title = (ax.get_title() or "").strip()
            xlabel = (ax.get_xlabel() or "").strip()
            ylabel = (ax.get_ylabel() or "").strip()
            has_content = bool(getattr(ax, "images", [])) or bool(getattr(ax, "lines", [])) or bool(getattr(ax, "patches", [])) or bool(getattr(ax, "collections", []))
            if not has_content and not title and not xlabel and not ylabel:
                continue
            label = title or f"Panel {panel_index}"
            key = f"axis_{panel_index}"
            options.append({"key": key, "label": label, "default": False, "kind": "axis", "axis_index": panel_index - 1})
            panel_index += 1
        return options
    def _choose_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Choose batch output folder", mustexist=False)
        if folder:
            self.output_dir_var.set(folder)
    def _update_points(self, name):
        row = self.param_rows[name]
        if not row["use"].get():
            row["points"].set("1")
            self._update_total_combinations()
            return
        try:
            values = _inclusive_float_range(row["start"].get(), row["stop"].get(), row["step"].get())
            row["points"].set(str(len(values)))
        except Exception:
            row["points"].set("?")
        self._update_total_combinations()
    def _update_total_combinations(self):
        total = 1
        invalid = False
        active = 0
        for row in self.param_rows.values():
            if row["use"].get():
                active += 1
                try:
                    points = int(row["points"].get())
                except Exception:
                    invalid = True
                    break
                total *= points
        if invalid:
            self.total_combinations_var.set("Total combinations: ?")
        elif active == 0:
            self.total_combinations_var.set("Total combinations: 1 (current settings only)")
        else:
            self.total_combinations_var.set(f"Total combinations: {total}")
    def _collect_sweep(self):
        param_values = {}
        total = 1
        for name, row in self.param_rows.items():
            if row["use"].get():
                values = _inclusive_float_range(row["start"].get(), row["stop"].get(), row["step"].get())
            else:
                values = [row["current"]]
            if len(values) == 0:
                raise ValueError(f"No values generated for parameter '{name}'.")
            param_values[name] = values
            total *= len(values)
        return param_values, total
    def _selected_export_options(self):
        selected = []
        for item in self.export_vars.values():
            if item["var"].get():
                selected.append(item["option"])
        return selected
    def _set_running_state(self, running):
        self.is_running = bool(running)
        if running:
            self.run_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.close_button.configure(state=tk.DISABLED)
        else:
            self.run_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.close_button.configure(state=tk.NORMAL)
    def request_stop(self):
        if self.is_running:
            self.cancel_requested = True
            self.progress_var.set("Stopping after the current save finishes...")
            self.update_idletasks()
    def _on_close(self):
        if self.is_running:
            self.request_stop()
            return
        self.destroy()
    def _axis_bbox_inches(self, fig, ax):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer)
        bbox = bbox.expanded(1.03, 1.05)
        return bbox.transformed(fig.dpi_scale_trans.inverted())
    def _save_axis_image(self, fig, ax, path, dpi):
        fig.savefig(path, dpi=dpi, bbox_inches=self._axis_bbox_inches(fig, ax))
    def _fig_to_pil_image(self, fig, dpi):
        if Image is None:
            raise RuntimeError("Pillow is required for square-grid saving. Please install pillow or use Individual files mode.")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    def _axis_to_pil_image(self, fig, ax, dpi):
        if Image is None:
            raise RuntimeError("Pillow is required for square-grid saving. Please install pillow or use Individual files mode.")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches=self._axis_bbox_inches(fig, ax))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    def _save_square_grid_image(self, images, path, labels=None, background=(255, 255, 255), grid_n=None):
        if Image is None:
            raise RuntimeError("Pillow is required for square-grid saving. Please install pillow or use Individual files mode.")
        if not images:
            return
        n = len(images)
        if grid_n is None:
            grid_n = int(math.ceil(math.sqrt(n)))
        try:
            grid_n = int(grid_n)
        except Exception:
            raise ValueError("Grid size must be an integer.")
        if grid_n <= 0:
            raise ValueError("Grid size must be at least 1.")
        max_w = max(img.width for img in images)
        max_h = max(img.height for img in images)
        tile_size = max(max_w, max_h)
        canvas = Image.new("RGB", (grid_n * tile_size, grid_n * tile_size), background)
        draw = ImageDraw.Draw(canvas) if labels and ImageDraw is not None else None
        try:
            font = ImageFont.load_default() if ImageFont is not None else None
        except Exception:
            font = None
        for idx, img in enumerate(images):
            row = idx // grid_n
            col = idx % grid_n
            x0 = col * tile_size
            y0 = row * tile_size
            scale = min(tile_size / img.width, tile_size / img.height)
            new_w = max(1, int(round(img.width * scale)))
            new_h = max(1, int(round(img.height * scale)))
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            paste_x = x0 + (tile_size - new_w) // 2
            paste_y = y0 + (tile_size - new_h) // 2
            canvas.paste(resized, (paste_x, paste_y))
            if draw is not None:
                label = str(labels[idx]) if idx < len(labels) else str(idx + 1)
                draw.text((x0 + 6, y0 + 6), label, fill=(0, 0, 0), font=font)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
    def _grid_output_path(self, output_dir, label_slug, prefix, page_index=0, multipart=False):
        if multipart or page_index > 0:
            filename = f"{prefix}__{label_slug}__square_grid_part{page_index + 1:02d}.png"
        else:
            filename = f"{prefix}__{label_slug}__square_grid.png"
        return output_dir / label_slug / filename
    def run_batch(self):
        app = self.app
        if not hasattr(app, "update_plot") or not hasattr(app, "fig") or not hasattr(app, "vars"):
            messagebox.showerror("Batch auto save", "This simulator does not expose the required API (vars, update_plot, fig).", parent=self)
            return
        try:
            param_values, total = self._collect_sweep()
        except Exception as exc:
            messagebox.showerror("Invalid sweep", str(exc), parent=self)
            return
        selected_exports = self._selected_export_options()
        if not selected_exports:
            messagebox.showerror("Batch auto save", "Please select at least one image type to save.", parent=self)
            return
        output_dir = Path(self.output_dir_var.get()).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = _safe_filename_fragment(self.prefix_var.get() or self.simulator_name)
        try:
            dpi = int(float(self.dpi_var.get()))
        except Exception:
            messagebox.showerror("Invalid DPI", "Please enter a valid numeric DPI.", parent=self)
            return
        original_values = {}
        for name, var in app.vars.items():
            try:
                original_values[name] = var.get()
            except Exception:
                pass
        save_mode = self.save_mode_var.get()
        if save_mode not in {"individual", "grid"}:
            save_mode = "individual"
        grid_size = None
        if save_mode == "grid":
            try:
                grid_size = int(float(self.grid_size_var.get()))
            except Exception:
                messagebox.showerror("Invalid grid size", "Please enter a valid integer for Grid size (N×N).", parent=self)
                return
            if grid_size <= 0:
                messagebox.showerror("Invalid grid size", "Grid size (N×N) must be at least 1.", parent=self)
                return
        manifest_rows = []
        grid_states = {}
        grid_page_capacity = max(1, grid_size * grid_size) if save_mode == "grid" else None
        active_names = [name for name, row in self.param_rows.items() if row["use"].get()]
        if not active_names:
            active_names = [name for name in app.vars.keys() if name in param_values]
        self.cancel_requested = False
        self._set_running_state(True)
        stopped_early = False
        try:
            keys = list(param_values.keys())
            value_lists = [param_values[k] for k in keys]
            total_digits = max(3, len(str(total)))
            for idx, values in enumerate(itertools.product(*value_lists), start=1):
                if self.cancel_requested:
                    stopped_early = True
                    break
                combo = dict(zip(keys, values))
                for name, value in combo.items():
                    try:
                        app.vars[name].set(value)
                    except Exception:
                        pass
                app.update_plot()
                self.root.update_idletasks()
                self.root.update()
                selected_parts = [
                    f"{_safe_filename_fragment(name)}-{_safe_filename_fragment(_format_param_value(combo[name]))}"
                    for name in active_names if name in combo
                ]
                if not selected_parts:
                    selected_parts = [f"run-{idx:0{total_digits}d}"]
                stem = f"{prefix}_{idx:0{total_digits}d}__" + "__".join(selected_parts)
                saved_files = []
                fig = app.fig
                fig.canvas.draw()
                visible_axes = [ax for ax in fig.axes if getattr(ax, "get_visible", lambda: True)()]
                combo_label = f"{idx:0{total_digits}d}"
                if active_names:
                    combo_label += " | " + ", ".join(
                        f"{name}={_format_param_value(combo[name])}" for name in active_names if name in combo
                    )
                for export_option in selected_exports:
                    if self.cancel_requested:
                        stopped_early = True
                        break
                    kind = export_option["kind"]
                    label_slug = _safe_filename_fragment(export_option["label"])
                    subfolder = output_dir / label_slug
                    subfolder.mkdir(parents=True, exist_ok=True)
                    if save_mode == "individual":
                        if kind == "full":
                            path = subfolder / f"{stem}__full_figure.png"
                            fig.savefig(path, dpi=dpi, bbox_inches="tight")
                            saved_files.append(str(path.relative_to(output_dir)))
                        elif kind == "axis":
                            axis_index = export_option["axis_index"]
                            if 0 <= axis_index < len(visible_axes):
                                path = subfolder / f"{stem}__{label_slug}.png"
                                self._save_axis_image(fig, visible_axes[axis_index], path, dpi)
                                saved_files.append(str(path.relative_to(output_dir)))
                    else:
                        if kind == "full":
                            image = self._fig_to_pil_image(fig, dpi)
                        elif kind == "axis":
                            axis_index = export_option["axis_index"]
                            if not (0 <= axis_index < len(visible_axes)):
                                continue
                            image = self._axis_to_pil_image(fig, visible_axes[axis_index], dpi)
                        else:
                            continue
                        state = grid_states.setdefault(label_slug, {
                            "count": 0,
                            "images": [],
                            "labels": [],
                            "multipart": False,
                            "files": [],
                            "page_saved": False,
                        })
                        count = state["count"]
                        page_index = count // grid_page_capacity
                        slot_index = count % grid_page_capacity
                        if slot_index == 0 and count > 0:
                            state["images"] = []
                            state["labels"] = []
                            state["page_saved"] = False
                            if page_index == 1 and not state["multipart"]:
                                old_path = self._grid_output_path(output_dir, label_slug, prefix, page_index=0, multipart=False)
                                new_path = self._grid_output_path(output_dir, label_slug, prefix, page_index=0, multipart=True)
                                old_path.parent.mkdir(parents=True, exist_ok=True)
                                if old_path.exists() and old_path != new_path:
                                    try:
                                        if new_path.exists():
                                            new_path.unlink()
                                        old_path.rename(new_path)
                                    except Exception:
                                        import shutil
                                        shutil.copy2(old_path, new_path)
                                        try:
                                            old_path.unlink()
                                        except Exception:
                                            pass
                                old_rel = str(old_path.relative_to(output_dir))
                                new_rel = str(new_path.relative_to(output_dir))
                                state["files"] = [new_rel if rel == old_rel else rel for rel in state["files"]]
                                state["multipart"] = True
                        state["images"].append(image)
                        state["labels"].append(combo_label)
                        current_path = self._grid_output_path(
                            output_dir,
                            label_slug,
                            prefix,
                            page_index=page_index,
                            multipart=state["multipart"],
                        )
                        current_rel = str(current_path.relative_to(output_dir))
                        if len(state["images"]) >= grid_page_capacity:
                            self._save_square_grid_image(state["images"], current_path, labels=state["labels"], grid_n=grid_size)
                            if current_rel not in state["files"]:
                                state["files"].append(current_rel)
                            saved_files.append(current_rel)
                            state["page_saved"] = True
                        state["count"] = count + 1
                if save_mode == "grid" and not saved_files:
                    saved_files = ["<no grid image updated>"]
                if saved_files:
                    row = {"index": idx, "image_files": "; ".join(saved_files)}
                    for name in keys:
                        row[name] = combo[name]
                    manifest_rows.append(row)
                if stopped_early:
                    break
                self.progress_var.set(f"Saved {idx}/{total}: {len(saved_files)} file(s)")
                self.update_idletasks()
                self.root.update()
        except Exception as exc:
            messagebox.showerror("Batch auto save", f"Batch run failed\n\n{type(exc).__name__}: {exc}", parent=self)
            return
        finally:
            self._set_running_state(False)
            for name, value in original_values.items():
                try:
                    app.vars[name].set(value)
                except Exception:
                    pass
            try:
                app.update_plot()
                self.root.update_idletasks()
            except Exception:
                pass
        grid_files = []
        if save_mode == "grid" and grid_states:
            for label_slug, state in grid_states.items():
                if state.get("images") and not state.get("page_saved", False):
                    page_index = max(0, (state.get("count", 1) - 1) // grid_page_capacity)
                    current_path = self._grid_output_path(
                        output_dir,
                        label_slug,
                        prefix,
                        page_index=page_index,
                        multipart=state.get("multipart", False),
                    )
                    self._save_square_grid_image(state["images"], current_path, labels=state["labels"], grid_n=grid_size)
                    current_rel = str(current_path.relative_to(output_dir))
                    if current_rel not in state["files"]:
                        state["files"].append(current_rel)
                    state["page_saved"] = True
                for relpath in state.get("files", []):
                    if relpath not in grid_files:
                        grid_files.append(relpath)
            if manifest_rows:
                grid_summary = "; ".join(grid_files)
                for row in manifest_rows:
                    row["image_files"] = grid_summary
        if self.save_manifest_var.get() and manifest_rows:
            csv_path = output_dir / f"{prefix}_manifest.csv"
            fieldnames = list(manifest_rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(manifest_rows)
        if stopped_early:
            messagebox.showinfo(
                "Batch auto save",
                f"Batch stopped early. Saved {len(manifest_rows)} parameter sets to\n{output_dir}",
                parent=self,
            )
            self.progress_var.set(f"Stopped early. Saved {len(manifest_rows)} parameter sets to {output_dir}")
        else:
            messagebox.showinfo(
                "Batch auto save",
                f"Finished. Saved {len(manifest_rows)} parameter sets to\n{output_dir}",
                parent=self,
            )
            self.progress_var.set(f"Finished. Saved {len(manifest_rows)} parameter sets to {output_dir}")
def _parse_index_list_spec(text, min_value=1, max_value=None):
    """Parse image indices: 561, 600-610, 700:740:5."""
    items = []
    for token in re.split(r"[, \t]+", str(text).strip()):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = [p for p in token.split(":") if p != ""]
            if len(parts) not in (2, 3):
                raise ValueError(f"Invalid range token: {token}")
            start = int(float(parts[0]))
            stop = int(float(parts[1]))
            step = int(float(parts[2])) if len(parts) == 3 else 1
            if step == 0:
                raise ValueError(f"Step cannot be zero: {token}")
            if start <= stop and step < 0:
                step = abs(step)
            if start > stop and step > 0:
                step = -step
            cur = start
            if step > 0:
                while cur <= stop:
                    items.append(cur)
                    cur += step
            else:
                while cur >= stop:
                    items.append(cur)
                    cur += step
        elif "-" in token and re.match(r"^\d+(\.\d+)?-\d+(\.\d+)?$", token):
            start_s, stop_s = token.split("-", 1)
            start = int(float(start_s))
            stop = int(float(stop_s))
            step = 1 if start <= stop else -1
            cur = start
            if step > 0:
                while cur <= stop:
                    items.append(cur)
                    cur += step
            else:
                while cur >= stop:
                    items.append(cur)
                    cur += step
        else:
            items.append(int(float(token)))
    out = []
    seen = set()
    for value in items:
        if value < min_value:
            raise ValueError(f"Index {value} is below {min_value}.")
        if max_value is not None and value > max_value:
            raise ValueError(f"Index {value} is above {max_value}.")
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
def _parse_experimental_dataset_spec(spec_text, default_base_dir):
    """Parse multi-run experimental dataset specification.
    Supported lines
    ---------------
    run=245; images=561,600-610; base=/path/to/runs
    245:561,600-610
    245,561,600-610
    Empty lines and lines starting with # are ignored.
    """
    records = []
    default_base = str(default_base_dir).strip()
    for raw_line in str(spec_text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        base_dir = default_base
        run_text = None
        images_text = None
        if "=" in line:
            parts = [p.strip() for p in line.split(";") if p.strip()]
            kv = {}
            for part in parts:
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                kv[k.strip().lower()] = v.strip()
            run_text = kv.get("run") or kv.get("runs")
            images_text = kv.get("image") or kv.get("images") or kv.get("img") or kv.get("imgs")
            base_dir = kv.get("base") or kv.get("base_dir") or kv.get("dir") or base_dir
            if run_text is None or images_text is None:
                raise ValueError(f"Could not parse experimental spec line: {raw_line}")
        elif ":" in line:
            run_text, images_text = line.split(":", 1)
            run_text = run_text.strip()
            images_text = images_text.strip()
        else:
            tokens = [t.strip() for t in line.split(",") if t.strip()]
            if len(tokens) < 2:
                raise ValueError(f"Could not parse experimental spec line: {raw_line}")
            run_text = tokens[0]
            images_text = ",".join(tokens[1:])
        runs = _parse_index_list_spec(run_text, min_value=1)
        images = _parse_index_list_spec(images_text, min_value=1, max_value=3000)
        for run_number in runs:
            for image_number in images:
                records.append({
                    "base_dir": base_dir,
                    "run_number": int(run_number),
                    "image_number": int(image_number),
                })
    if not records:
        raise ValueError("No experimental images were specified.")
    return records
def _parse_experimental_csv_file(csv_path, default_base_dir):
    """Parse an experimental list CSV with no header.
    Required format, starting from row 1:
        column 1 = run number
        column 2 = image number
    Optional:
        column 3 = base directory, if different from the default experimental base.
    Blank rows and comment rows beginning with # are ignored.
    """
    path = Path(str(csv_path).strip()).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Experimental CSV file not found: {path}")
    records = []
    default_base = str(default_base_dir).strip()
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            row = [str(x).strip() for x in row]
            if not any(row):
                continue
            if row[0].startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(
                    f"Experimental CSV row {row_number} must contain at least two columns: run_number,image_number"
                )
            try:
                run_number = int(float(row[0]))
                image_number = int(float(row[1]))
            except Exception as exc:
                raise ValueError(
                    f"Experimental CSV row {row_number} has invalid run/image values: {row[:2]}"
                ) from exc
            if run_number < 1:
                raise ValueError(f"Experimental CSV row {row_number}: run number must be >= 1.")
            if image_number < 1 or image_number > 3000:
                raise ValueError(f"Experimental CSV row {row_number}: image number must be 1..3000.")
            base_dir = row[2] if len(row) >= 3 and row[2] else default_base
            records.append({
                "base_dir": base_dir,
                "run_number": run_number,
                "image_number": image_number,
                "source": "csv",
                "csv_path": str(path),
                "csv_row": row_number,
            })
    if not records:
        raise ValueError(f"No experimental rows were found in CSV file: {path}")
    return records
def _screening_safe_float(value, default=float("nan")):
    try:
        v = float(value)
    except Exception:
        return default
    return v if np.isfinite(v) else default
def _format_elapsed_seconds(seconds):
    try:
        total = int(max(0, float(seconds)))
    except Exception:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
def _jsonify_params(params):
    out = {}
    for k, v in params.items():
        try:
            fv = float(v)
            out[str(k)] = fv
        except Exception:
            out[str(k)] = str(v)
    return out
class AutoScreenDialog(tk.Toplevel):
    """Automatic parameter sweep + experimental SAXS screening.
    This dialog intentionally does not export visualization images.  It computes
    simulations, caches the simulated 2D arrays, compares them against one or more
    experimental images, and writes CSV score tables.
    """
    METRIC_FIELDS = [
        "accepted",
        "total_score",
        "total_score_formula",
        "total_score_active_terms",
        "total_score_weight_sum",
        "score_contribution_summary",
        "radial_profile_corr",
        "radial_profile_nrmse",
        "integration_curve_corr",
        "integration_curve_nrmse",
        "integration_curve_scale_factor",
        "integration_chi2_reduced",
        "integration_chi2_n_points",
        "integration_chi2_dof",
        "integration_chi2_scale_factor",
        "integration_chi2_threshold_enabled",
        "integration_chi2_threshold_max",
        "integration_chi2_threshold_pass",
        "fringe_position_rel_error",
        "fringe_peak_rel_error",
        "fringe_min_rel_error",
        "fringe_feature_count_exp",
        "fringe_feature_count_sim",
        "fringe_feature_count_matched",
        "fringe_peak_count_exp",
        "fringe_peak_count_sim",
        "fringe_min_count_exp",
        "fringe_min_count_sim",
        "radial_chi2_reduced",
        "radial_chi2_n_points",
        "radial_chi2_scale_factor",
        "radial_weighted_nrmse",
        "integration_n_bins",
        "pixel_pearson_corr_asinh_z",
        "pixel_nrmse_asinh_z",
        "A2_anisotropy_corr",
        "A4_anisotropy_corr",
        "A2_phase_error_deg",
        "A2_phase_qmean_abs_error_deg",
        "A2_phase_global_error_deg",
        "A2_phase_mean_exp_deg",
        "A2_phase_mean_sim_deg",
        "A2_phase_n_points",
        "A3_phase_error_deg",
        "A3_phase_qmean_abs_error_deg",
        "A3_phase_global_error_deg",
        "A3_phase_mean_exp_deg",
        "A3_phase_mean_sim_deg",
        "A3_phase_n_points",
        "A4_phase_error_deg",
        "A4_phase_qmean_abs_error_deg",
        "A4_phase_global_error_deg",
        "A4_phase_mean_exp_deg",
        "A4_phase_mean_sim_deg",
        "A4_phase_n_points",
        "A5_phase_error_deg",
        "A5_phase_qmean_abs_error_deg",
        "A5_phase_global_error_deg",
        "A5_phase_mean_exp_deg",
        "A5_phase_mean_sim_deg",
        "A5_phase_n_points",
        "A6_phase_error_deg",
        "A6_phase_qmean_abs_error_deg",
        "A6_phase_global_error_deg",
        "A6_phase_mean_exp_deg",
        "A6_phase_mean_sim_deg",
        "A6_phase_n_points",
        "A2_phase_threshold_enabled",
        "A2_phase_threshold_max_deg",
        "A2_phase_threshold_pass",
        "harmonic_phase_threshold_enabled",
        "harmonic_phase_threshold_max_deg",
        "harmonic_phase_threshold_pass",
        "angular_profile_corr_zero_shift",
        "angular_xcorr_best_corr",
        "angular_xcorr_best_shift_deg",
        "angular_xcorr_best_shift_abs_deg_axial",
        "angular_xcorr_n_phi_bins",
        "angular_xcorr_valid_bins",
        "polar_corr_zero_shift",
        "polar_nrmse_zero_shift",
        "polar_xcorr_best_corr",
        "polar_xcorr_best_shift_deg",
        "polar_nrmse_best_shift",
        "polar_n_q_bins",
        "polar_n_phi_bins",
        "polar_valid_bins",
        "harmonics_mean_corr_A2_A6",
        "harmonics_mean_nrmse_A2_A6",
        "harmonic_phase_error_mean_A2_A6_deg",
        "harmonic_phase_error_weighted_A2_A6_deg",
        "harmonic_phase_score_n_terms",
        "angular_corr_threshold_enabled",
        "angular_corr_threshold_min",
        "angular_corr_threshold_pass",
        "harmonics_mean_corr_A1_A12",
        "harmonics_mean_nrmse_A1_A12",
        "pixel_ssim_asinh_z",
        "valid_pixels",
        "feature_q_min",
        "feature_q_max",
        "harmonic_q_min",
        "harmonic_q_max",
        "simulation_scale_factor_raw_pixel_LS",
    ]
    SCORE_TERMS = [
        {"key": "radial_corr", "label": "2D radial corr", "metric": "radial_profile_corr", "mode": "corr", "enabled": True, "weight": 0.08},
        {"key": "radial_nrmse", "label": "2D radial NRMSE", "metric": "radial_profile_nrmse", "mode": "error", "enabled": True, "weight": 0.05},
        {"key": "radial_chi2", "label": "2D radial χ²red", "metric": "radial_chi2_reduced", "mode": "log_chi2", "enabled": True, "weight": 0.10},
        {"key": "int_corr", "label": "1D integration corr", "metric": "integration_curve_corr", "mode": "corr", "enabled": True, "weight": 0.10},
        {"key": "int_nrmse", "label": "1D integration NRMSE", "metric": "integration_curve_nrmse", "mode": "error", "enabled": True, "weight": 0.06},
        {"key": "int_chi2", "label": "1D integration χ²red", "metric": "integration_chi2_reduced", "mode": "log_chi2", "enabled": True, "weight": 0.08},
        {"key": "fringe_pos", "label": "Fringe position error", "metric": "fringe_position_rel_error", "mode": "scaled_error", "scale": 0.15, "cap": 1.5, "enabled": True, "weight": 0.08},
        {"key": "pixel_corr", "label": "2D signed pixel corr", "metric": "pixel_pearson_corr_asinh_z", "mode": "corr", "enabled": True, "weight": 0.08},
        {"key": "pixel_nrmse", "label": "2D signed pixel NRMSE", "metric": "pixel_nrmse_asinh_z", "mode": "error", "enabled": True, "weight": 0.04},
        {"key": "harmonic_corr", "label": "A2–A6 harmonic corr", "metric": "harmonics_mean_corr_A2_A6", "mode": "corr", "enabled": True, "weight": 0.08},
        {"key": "harmonic_nrmse", "label": "A2–A6 harmonic NRMSE", "metric": "harmonics_mean_nrmse_A2_A6", "mode": "error", "enabled": False, "weight": 0.04},
        {"key": "harmonic_phase", "label": "A2–A6 phase error", "metric": "harmonic_phase_error_weighted_A2_A6_deg", "mode": "scaled_error", "scale": 30.0, "cap": 1.5, "enabled": True, "weight": 0.06},
        {"key": "angular_zero", "label": "Angular zero-shift corr", "metric": "angular_profile_corr_zero_shift", "mode": "corr", "enabled": True, "weight": 0.05},
        {"key": "angular_best", "label": "Angular best-shift corr", "metric": "angular_xcorr_best_corr", "mode": "corr", "enabled": False, "weight": 0.04},
        {"key": "polar_zero", "label": "Polar q–φ zero corr", "metric": "polar_corr_zero_shift", "mode": "corr", "enabled": True, "weight": 0.05},
        {"key": "polar_best", "label": "Polar q–φ best corr", "metric": "polar_xcorr_best_corr", "mode": "corr", "enabled": True, "weight": 0.06},
        {"key": "polar_best_nrmse", "label": "Polar q–φ best NRMSE", "metric": "polar_nrmse_best_shift", "mode": "error", "enabled": False, "weight": 0.04},
    ]
    def __init__(self, parent, simulator_name, app, root, noise_source=None):
        super().__init__(parent)
        self.parent = parent
        self.simulator_name = simulator_name
        self.app = app
        self.root = root
        # Poisson-noise controls can be configured directly in this Auto Screen
        # window.  The main GUI is kept only as a delegate for the actual
        # poissonization helper and as the source of initial values.
        self.noise_source = noise_source if noise_source is not None else app
        initial_noise_source = self.noise_source
        try:
            initial_poisson_enabled = bool(initial_noise_source.poisson_noise_var.get())
        except Exception:
            initial_poisson_enabled = False
        try:
            initial_poisson_peak_counts = str(initial_noise_source.poisson_peak_counts_var.get())
        except Exception:
            initial_poisson_peak_counts = "10000"
        try:
            initial_poisson_seed = str(initial_noise_source.poisson_seed_var.get())
        except Exception:
            initial_poisson_seed = ""
        try:
            initial_use_dataset_ref = bool(initial_noise_source.poisson_use_dataset_reference_var.get())
        except Exception:
            initial_use_dataset_ref = False
        try:
            initial_poisson_multiplier = str(initial_noise_source.poisson_count_multiplier_var.get())
        except Exception:
            initial_poisson_multiplier = "1.0"
        try:
            initial_poisson_percentile = str(initial_noise_source.poisson_reference_percentile_var.get())
        except Exception:
            initial_poisson_percentile = "99.9"
        self.poisson_noise_var = tk.BooleanVar(value=initial_poisson_enabled)
        self.poisson_peak_counts_var = tk.StringVar(value=initial_poisson_peak_counts)
        self.poisson_seed_var = tk.StringVar(value=initial_poisson_seed)
        self.poisson_use_dataset_reference_var = tk.BooleanVar(value=initial_use_dataset_ref)
        self.poisson_count_multiplier_var = tk.StringVar(value=initial_poisson_multiplier)
        self.poisson_reference_percentile_var = tk.StringVar(value=initial_poisson_percentile)
        self.poisson_reference_status_var = tk.StringVar(value="Poisson ref: inherited or not calibrated")
        self.poisson_reference_intensity = getattr(initial_noise_source, "poisson_reference_intensity", None)
        self.poisson_reference_metadata = dict(getattr(initial_noise_source, "poisson_reference_metadata", {}) or {})
        self.param_rows = {}
        self.cancel_requested = False
        self.is_running = False
        self.title(f"Auto Screen - {simulator_name}")
        self.geometry("1120x780")
        try:
            self.minsize(940, 640)
        except Exception:
            pass
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.output_dir_var = tk.StringVar(value=str((BASE_DIR / "screening_outputs" / _safe_filename_fragment(simulator_name)).resolve()))
        self.cache_dir_var = tk.StringVar(value=str((BASE_DIR / "simulation_cache").resolve()))
        self.prefix_var = tk.StringVar(value=_safe_filename_fragment(simulator_name))
        self.default_exp_base_dir_var = tk.StringVar(value=str(BASE_DIR.resolve()))
        self.exp_csv_path_var = tk.StringVar(value="")
        self.n_bins_var = tk.StringVar(value="500")
        self.radial_corr_min_var = tk.StringVar(value="0.95")
        self.radial_nrmse_max_var = tk.StringVar(value="0.25")
        self.radial_chi2_max_var = tk.StringVar(value="5.0")
        self.integration_curve_corr_min_var = tk.StringVar(value="0.97")
        self.integration_curve_nrmse_max_var = tk.StringVar(value="0.20")
        self.integration_chi2_max_var = tk.StringVar(value="100.0")
        self.fringe_position_error_max_var = tk.StringVar(value="0.15")
        self.pixel_corr_min_var = tk.StringVar(value="0.65")
        self.harmonic_corr_min_var = tk.StringVar(value="0.45")
        self.a2_phase_error_max_var = tk.StringVar(value="30.0")
        self.angular_corr_min_var = tk.StringVar(value="0.30")
        self.total_score_max_var = tk.StringVar(value="0.50")
        # Acceptance hard-gate switches.  Score terms are controlled separately below.
        self.threshold_gate_vars = {
            "radial_corr": tk.BooleanVar(value=True),
            "radial_nrmse": tk.BooleanVar(value=True),
            "radial_chi2": tk.BooleanVar(value=True),
            "int_corr": tk.BooleanVar(value=True),
            "int_nrmse": tk.BooleanVar(value=True),
            "int_chi2": tk.BooleanVar(value=True),
            "fringe": tk.BooleanVar(value=True),
            "pixel_corr": tk.BooleanVar(value=True),
            "harmonic_corr": tk.BooleanVar(value=True),
            "harmonic_phase": tk.BooleanVar(value=True),
            "angular_corr": tk.BooleanVar(value=True),
            "total_score": tk.BooleanVar(value=True),
        }
        self.score_term_vars = {}
        self.score_formula_var = tk.StringVar(value="")
        self.local_scan_percent_var = tk.StringVar(value="10")
        self.local_scan_angle_span_var = tk.StringVar(value="10")
        self.local_scan_points_var = tk.StringVar(value="5")
        self.local_scan_status_var = tk.StringVar(value="Check parameters, then fill local ranges from the current simulator values.")
        self.local_scan_estimate_var = tk.StringVar(value="Local estimate: no checked parameter (1 simulation).")
        for term in self.SCORE_TERMS:
            key = str(term["key"])
            self.score_term_vars[key] = {
                "enabled": tk.BooleanVar(value=bool(term.get("enabled", True))),
                "weight": tk.StringVar(value=str(term.get("weight", 1.0))),
            }
        self.save_all_var = tk.BooleanVar(value=True)
        self.skip_cached_comparison_var = tk.BooleanVar(value=False)
        self.progress_var = tk.StringVar(value="Set experimental images and sweep ranges, then click 'Run screening'.")
        self.total_combinations_var = tk.StringVar(value="Total combinations: 1")
        for _v in (self.local_scan_percent_var, self.local_scan_angle_span_var, self.local_scan_points_var, self.exp_csv_path_var):
            try:
                _v.trace_add("write", lambda *_a: self._update_local_scan_estimate())
            except Exception:
                pass
        self._tooltips = []
        self._build_ui()
    def _make_tooltip(self, widget, text):
        tip = _HoverToolTip(widget, text)
        self._tooltips.append(tip)
        return tip
    def _threshold_tooltip_text(self, key):
        tips = {
            "n_bins": (
                "2D feature에서 q 방향을 몇 개의 bin으로 나누어 radial/angular feature를 계산할지 정합니다.\n\n"
                "현재 workflow에서는 SAXS 1D integration과 해상도를 맞추기 위해 보통 500을 사용합니다.\n\n"
                "값을 키우면 q에 따른 미세한 oscillation이나 anisotropy 변화를 더 잘 분해할 수 있지만, bin 하나에 들어가는 pixel 수가 줄어 noise에 더 민감해질 수 있습니다.\n"
                "값을 줄이면 feature가 더 부드럽고 관대하게 계산되지만, 좁은 fringe나 작은 구조 차이를 놓칠 수 있습니다."
            ),
            "radial_corr": (
                "2D feature q range에서 experimental radial profile과 simulated radial profile 사이의 최소 Pearson correlation 기준입니다.\n\n"
                "주로 ring 위치, q 방향 intensity modulation, 전체 radial shape이 비슷한지를 봅니다.\n\n"
                "값을 높이면 radial shape이 매우 비슷한 후보만 통과합니다.\n"
                "값을 낮추면 radial profile이 어느 정도 달라도 후보로 남을 수 있습니다."
            ),
            "radial_nrmse": (
                "experimental radial profile과 simulated radial profile 사이의 최대 normalized RMSE 기준입니다.\n\n"
                "radial intensity profile의 크기와 모양이 얼마나 어긋나는지를 error 관점에서 봅니다.\n\n"
                "값을 낮추면 radial profile amplitude/shape 차이에 더 엄격해집니다.\n"
                "값을 높이면 radial profile error가 더 큰 후보도 허용합니다."
            ),
            "radial_chi2": (
                "2D feature용 robust radial reduced χ²의 최대 허용값입니다.\n\n"
                "이 값은 SAXS integrate 창의 1D χ²와는 의도적으로 분리된 지표입니다.\n"
                "radial feature 비교에서 특정 bin의 작은 uncertainty가 전체 판정을 과도하게 지배하지 않도록 robust uncertainty floor를 사용합니다.\n\n"
                "값을 낮추면 radial mismatch에 더 엄격해지고, 값을 높이면 더 관대해집니다."
            ),
            "int_corr": (
                "SAXS 1D integration curve의 최소 correlation 기준입니다.\n\n"
                "500-bin으로 적분한 experimental 1D curve와 auto-scaled theoretical 1D curve를 q = 0.005–0.03 Å⁻¹ 범위에서 비교합니다.\n\n"
                "값을 높이면 1D curve의 overall shape과 oscillation이 더 잘 겹치는 경우만 통과합니다."
            ),
            "int_nrmse": (
                "SAXS 1D integration curve의 최대 normalized RMSE 기준입니다.\n\n"
                "experimental 1D curve와 scaled theoretical 1D curve의 intensity 차이를 normalize해서 평가합니다.\n\n"
                "값을 낮추면 1D curve intensity 차이에 더 엄격합니다.\n"
                "값을 높이면 curve overlap이 다소 부족해도 후보로 남을 수 있습니다."
            ),
            "int_chi2": (
                "SAXS 1D integration curve의 reduced χ² 최대 허용값입니다.\n\n"
                "experimental 1D curve와 scaled theoretical 1D curve를 q = 0.005–0.03 Å⁻¹ 범위에서 비교하며, experimental SEM과 robust uncertainty floor를 함께 사용합니다.\n\n"
                "값을 낮추면 1D curve의 intensity mismatch에 더 엄격해지고, 값을 높이면 rough screening에 더 관대해집니다."
            ),
            "fringe": (
                "1D integration curve에서 검출된 fringe peak/minimum 위치의 최대 상대 오차 기준입니다.\n\n"
                "전체 correlation이 높더라도 peak 위치나 minimum 위치가 밀려 있으면 이 기준에서 점수가 나빠집니다.\n\n"
                "값을 낮추면 fringe spacing과 peak 위치가 정확히 맞아야 통과합니다.\n"
                "값을 높이면 peak 위치가 어느 정도 shift되어도 허용합니다."
            ),
            "pixel_corr": (
                "selected q range에서 signed-asinh 변환된 2D feature image 사이의 최소 pixel-wise Pearson correlation 기준입니다.\n\n"
                "밝고 어두운 2D pattern의 위치가 서로 잘 대응되는지를 봅니다.\n"
                "signed-asinh 변환은 강한 intensity를 압축하여 약한 feature도 비교에 반영되도록 합니다.\n\n"
                "값을 높이면 2D spatial pattern이 더 비슷한 후보만 통과합니다."
            ),
            "harmonic_corr": (
                "A2–A6 angular harmonic feature들의 평균 correlation 최소 기준입니다.\n\n"
                "2D pattern의 anisotropy, 방향성, angular symmetry가 q에 따라 얼마나 비슷한지 평가합니다.\n\n"
                "값을 높이면 orientation/symmetry pattern이 잘 맞는 후보만 통과합니다.\n"
                "값을 낮추면 angular mismatch에 더 관대해집니다."
            ),
            "a2_phase": (
                "A2–A6 complex harmonic phase error의 weighted mean 최대 허용값입니다.\n\n"
                "A2만 보는 기준이 아니라, A2, A3, A4, A5, A6에서 계산한 phase/orientation mismatch를 종합합니다.\n"
                "각 harmonic의 q-averaged amplitude와 유효 bin 수를 반영해 weighted mean을 만들고, 이 값이 threshold 이하일 때 phase gate를 통과합니다.\n\n"
                "값을 낮추면 higher-order angular symmetry와 orientation mismatch에 더 엄격해집니다. 90 이상으로 두면 사실상 비활성화됩니다."
            ),
            "angular_corr": (
                "harmonic q-range에서 φ 방향 angular intensity profile의 zero-shift correlation 최소 기준입니다.\n\n"
                "현재 simulation orientation이 experimental orientation과 그대로 겹치는지를 봅니다. best-shift cross-correlation과 best rotation shift도 CSV에 함께 저장됩니다.\n"
                "값을 높이면 angular orientation/shape mismatch에 더 엄격해집니다. -1 이하로 두면 사실상 비활성화됩니다."
            ),
            "total_score": (
                "여러 판별 기준을 합친 최종 screening score의 최대 허용값입니다.\n\n"
                "radial profile, 1D integration, fringe position, 2D pixel similarity, harmonic similarity 등을 종합합니다.\n"
                "작을수록 experimental image와 simulation이 더 잘 맞는 것으로 해석합니다.\n\n"
                "값을 낮추면 개별 기준을 통과하더라도 최종 accepted 판정이 더 엄격해집니다.\n"
                "값을 높이면 더 많은 후보가 accepted로 남을 수 있습니다."
            ),
            "poisson_noise": (
                "Auto Screen에서 theoretical simulation image에 Poisson shot noise를 적용한 뒤 experimental image와 비교합니다.\n\n"
                "기본 방식은 기존과 같이 simulation max를 Peak counts에 맞춥니다. Dataset max reference를 켜면 선택한 run에서 total intensity가 가장 큰 experimental image를 찾고, 그 image의 robust percentile count를 고정 reference로 사용합니다.\n"
                "Multiplier는 추출된 reference count에 곱해지는 배수입니다. 값이 클수록 상대적인 shot noise가 작아집니다."
            ),
        }
        return tips.get(key, "")
    def _score_term_default_map(self):
        return {str(term.get("key")): dict(term) for term in self.SCORE_TERMS}
    def _score_terms_from_controls(self):
        terms = []
        var_map = getattr(self, "score_term_vars", {}) or {}
        for term in self.SCORE_TERMS:
            t = dict(term)
            key = str(t.get("key"))
            if key in var_map:
                try:
                    t["enabled"] = bool(var_map[key]["enabled"].get())
                except Exception:
                    t["enabled"] = bool(term.get("enabled", True))
                try:
                    t["weight"] = _screening_safe_float(var_map[key]["weight"].get(), float(term.get("weight", 1.0)))
                except Exception:
                    t["weight"] = float(term.get("weight", 1.0))
            else:
                t["enabled"] = bool(term.get("enabled", True))
                t["weight"] = float(term.get("weight", 1.0))
            if not np.isfinite(float(t.get("weight", 0.0))) or float(t.get("weight", 0.0)) < 0:
                t["weight"] = 0.0
            terms.append(t)
        return terms
    def _threshold_gate_enabled_map(self):
        vars_map = getattr(self, "threshold_gate_vars", {}) or {}
        defaults = {
            "radial_corr": True, "radial_nrmse": True, "radial_chi2": True,
            "int_corr": True, "int_nrmse": True, "int_chi2": True,
            "fringe": True, "pixel_corr": True, "harmonic_corr": True,
            "harmonic_phase": True, "angular_corr": True, "total_score": True,
        }
        out = {}
        for key, default in defaults.items():
            var = vars_map.get(key)
            try:
                out[key] = bool(var.get()) if var is not None else bool(default)
            except Exception:
                out[key] = bool(default)
        return out

    def _threshold_gate_is_enabled(self, thresholds, key, default=True):
        if isinstance(thresholds, dict):
            gate_map = thresholds.get("gate_enabled", {}) or {}
            if key in gate_map:
                return bool(gate_map.get(key))
        return bool(default)

    def _set_all_threshold_gates(self, value=True):
        """Enable/disable all hard acceptance gates in the threshold panel."""
        for var in (getattr(self, "threshold_gate_vars", {}) or {}).values():
            try:
                var.set(bool(value))
            except Exception:
                pass

    def _set_all_score_terms(self, value=True):
        """Enable/disable all score terms used in the total-score formula."""
        for item in (getattr(self, "score_term_vars", {}) or {}).values():
            try:
                item["enabled"].set(bool(value))
            except Exception:
                pass
        try:
            self._update_score_formula_text()
        except Exception:
            pass

    def _set_all_scan_parameters(self, value=True):
        """Enable/disable all parameter rows used for parameter sweeping."""
        for name, row in (getattr(self, "param_rows", {}) or {}).items():
            try:
                row["use"].set(bool(value))
                self._update_points(name)
            except Exception:
                pass
        try:
            self.local_scan_status_var.set(
                "All scan parameters selected." if value else "All scan parameters cleared."
            )
        except Exception:
            pass
        self._update_total_combinations()
        self._update_local_scan_estimate()

    def _score_formula_symbol(self, term):
        """Very short score-term symbols shown in the Auto Screen UI.

        The symbols are intentionally plain uppercase letters so they remain
        readable in Tkinter.  They correspond to the order of SCORE_TERMS.
        """
        key = str(term.get("key", term.get("metric", "M")))
        for idx, ref in enumerate(self.SCORE_TERMS):
            if str(ref.get("key")) == key:
                return chr(ord("A") + idx) if idx < 26 else f"T{idx + 1}"
        return "?"

    def _score_formula_label(self, term):
        sym = self._score_formula_symbol(term)
        label = str(term.get("label", term.get("key", sym)))
        return f"{sym}. {label}"

    def _update_score_formula_text(self, *_args):
        terms = [t for t in self._score_terms_from_controls() if bool(t.get("enabled", True)) and float(t.get("weight", 0.0)) > 0]
        if not terms:
            msg = "S = nan  (no active term)"
        else:
            pieces = [f"{float(t.get('weight', 0.0)):g}*M{self._score_formula_symbol(t)}" for t in terms]
            denom = sum(float(t.get("weight", 0.0)) for t in terms)
            msg = "S = sum(W*M)/sum(W) = [" + " + ".join(pieces) + f"] / {denom:g}"
            msg += "\nTerm symbols: A, B, C...  M = converted mismatch used in the score.  corr: M=1-r, error: M=e, chi2: M=log, scaled: M=min(cap,e/scale)."
        try:
            self.score_formula_var.set(msg)
        except Exception:
            pass
    def _build_score_terms_panel(self, parent, row, col, colspan=3):
        box = ttk.LabelFrame(parent, text="Total score terms", padding=(8, 6))
        box.grid(row=row, column=col, columnspan=colspan, sticky="ew", pady=(10, 0))
        for c in range(6):
            box.columnconfigure(c, weight=1 if c in (1, 4) else 0)
        score_bar = ttk.Frame(box)
        score_bar.grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 6))
        ttk.Label(score_bar, text="Score-term Use:").pack(side=tk.LEFT)
        ttk.Button(score_bar, text="Select all terms", command=lambda: self._set_all_score_terms(True)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(score_bar, text="Clear all terms", command=lambda: self._set_all_score_terms(False)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            score_bar,
            text="Only checked terms enter S = sum(W*M)/sum(W).",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(box, text="Use", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky="w")
        ttk.Label(box, text="T. Term", font=("Arial", 9, "bold")).grid(row=1, column=1, sticky="w")
        ttk.Label(box, text="Weight", font=("Arial", 9, "bold")).grid(row=1, column=2, sticky="w")
        ttk.Label(box, text="Use", font=("Arial", 9, "bold")).grid(row=1, column=3, sticky="w", padx=(18, 0))
        ttk.Label(box, text="T. Term", font=("Arial", 9, "bold")).grid(row=1, column=4, sticky="w")
        ttk.Label(box, text="Weight", font=("Arial", 9, "bold")).grid(row=1, column=5, sticky="w")
        half = (len(self.SCORE_TERMS) + 1) // 2
        for idx, term in enumerate(self.SCORE_TERMS):
            side = 0 if idx < half else 1
            local_row = 2 + (idx if side == 0 else idx - half)
            base_col = 0 if side == 0 else 3
            key = str(term["key"])
            vars_for_key = self.score_term_vars[key]
            chk = ttk.Checkbutton(box, variable=vars_for_key["enabled"], command=self._update_score_formula_text)
            chk.grid(row=local_row, column=base_col, sticky="w")
            lbl = ttk.Label(box, text=self._score_formula_label(term), font=("Arial", 10))
            lbl.grid(row=local_row, column=base_col + 1, sticky="w", padx=(2, 4))
            ent = ttk.Entry(box, textvariable=vars_for_key["weight"], width=7)
            ent.grid(row=local_row, column=base_col + 2, sticky="w", padx=(2, 8))
            try:
                vars_for_key["weight"].trace_add("write", lambda *_a: self._update_score_formula_text())
            except Exception:
                pass
        formula = ttk.Label(box, textvariable=self.score_formula_var, wraplength=1040, justify="left", font=("Consolas", 10))
        formula.grid(row=2 + half, column=0, columnspan=6, sticky="ew", pady=(8, 0))
        self._update_score_formula_text()
        return box

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        # The Auto Screen dialog contains many setup/threshold controls.  Put the
        # full upper body in one scrollable region so smaller displays do not crop
        # the top controls while keeping the action buttons fixed at the bottom.
        body_canvas = tk.Canvas(self, highlightthickness=0)
        body_scrollbar = ttk.Scrollbar(self, orient="vertical", command=body_canvas.yview)
        body = ttk.Frame(body_canvas)
        body_window = body_canvas.create_window((0, 0), window=body, anchor="nw")
        body_canvas.configure(yscrollcommand=body_scrollbar.set)
        body_canvas.grid(row=0, column=0, sticky="nsew")
        body_scrollbar.grid(row=0, column=1, sticky="ns")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        def _sync_body_scrollregion(_event=None):
            body_canvas.configure(scrollregion=body_canvas.bbox("all"))

        def _resize_body_width(event):
            body_canvas.itemconfigure(body_window, width=event.width)
            _sync_body_scrollregion()

        def _on_body_mousewheel(event):
            # Let text widgets keep their own native scrolling behaviour.
            if isinstance(getattr(event, "widget", None), tk.Text):
                return None
            if getattr(event, "num", None) == 4:
                step = -1
            elif getattr(event, "num", None) == 5:
                step = 1
            else:
                delta = getattr(event, "delta", 0)
                step = -int(delta / 120) if delta else 0
            if step:
                body_canvas.yview_scroll(step, "units")
                return "break"
            return None

        def _bind_body_mousewheel_tree(widget):
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, _on_body_mousewheel, add="+")
            for child in widget.winfo_children():
                _bind_body_mousewheel_tree(child)

        body.bind("<Configure>", _sync_body_scrollregion)
        body_canvas.bind("<Configure>", _resize_body_width)
        self._bind_autoscreen_mousewheel_tree = _bind_body_mousewheel_tree
        self._autoscreen_body_canvas = body_canvas
        self._autoscreen_body_scrollbar = body_scrollbar

        top = ttk.Frame(body, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text=f"Automatic experimental screening for: {self.simulator_name}", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(top, text="Output folder").grid(row=1, column=0, sticky="w", pady=(10, 4))
        ttk.Entry(top, textvariable=self.output_dir_var).grid(row=1, column=1, sticky="ew", pady=(10, 4), padx=(8, 8))
        ttk.Button(top, text="Browse...", command=self._choose_output_folder).grid(row=1, column=2, sticky="e", pady=(10, 4))
        ttk.Label(top, text="Simulation cache root").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.cache_dir_var).grid(row=2, column=1, sticky="ew", pady=4, padx=(8, 8))
        ttk.Button(top, text="Browse...", command=self._choose_cache_folder).grid(row=2, column=2, sticky="e", pady=4)
        ttk.Label(top, text="Default experimental base").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.default_exp_base_dir_var).grid(row=3, column=1, sticky="ew", pady=4, padx=(8, 8))
        ttk.Button(top, text="Browse...", command=self._choose_exp_base_folder).grid(row=3, column=2, sticky="e", pady=4)
        ttk.Label(top, text="Experimental CSV").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(top, textvariable=self.exp_csv_path_var).grid(row=4, column=1, sticky="ew", pady=4, padx=(8, 8))
        csv_buttons = ttk.Frame(top)
        csv_buttons.grid(row=4, column=2, sticky="e", pady=4)
        ttk.Button(csv_buttons, text="Browse...", command=self._choose_exp_csv_file).pack(side=tk.LEFT)
        ttk.Button(csv_buttons, text="Clear", command=lambda: self.exp_csv_path_var.set("")).pack(side=tk.LEFT, padx=(4, 0))
        prefix_row = ttk.Frame(top)
        prefix_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Label(prefix_row, text="File prefix").pack(side=tk.LEFT)
        ttk.Entry(prefix_row, textvariable=self.prefix_var, width=22).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Checkbutton(prefix_row, text="Save all rows to all_scores.csv", variable=self.save_all_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Checkbutton(prefix_row, text="Skip comparison if simulation cache exists", variable=self.skip_cached_comparison_var).pack(side=tk.LEFT, padx=(0, 18))
        ttk.Label(prefix_row, textvariable=self.total_combinations_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(12, 0))
        noise_box = ttk.LabelFrame(top, text="Poisson noise for scan/screening", padding=(8, 6))
        noise_box.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        noise_tip = self._threshold_tooltip_text("poisson_noise")
        chk = ttk.Checkbutton(
            noise_box,
            text="Apply Poisson noise before comparison",
            variable=self.poisson_noise_var,
        )
        chk.pack(side=tk.LEFT)
        ttk.Label(noise_box, text="Reference").pack(side=tk.LEFT, padx=(14, 4))
        note = ttk.Label(noise_box, textvariable=self.poisson_reference_status_var)
        note.pack(side=tk.LEFT)
        ttk.Label(noise_box, text="Multiplier").pack(side=tk.LEFT, padx=(14, 4))
        mult_entry = ttk.Entry(noise_box, textvariable=self.poisson_count_multiplier_var, width=7)
        mult_entry.pack(side=tk.LEFT)
        ttk.Label(noise_box, text="Seed").pack(side=tk.LEFT, padx=(14, 4))
        seed_entry = ttk.Entry(noise_box, textvariable=self.poisson_seed_var, width=8)
        seed_entry.pack(side=tk.LEFT)
        for widget in (chk, mult_entry, seed_entry, note):
            self._make_tooltip(widget, noise_tip)
        exp_box = ttk.LabelFrame(top, text="Manual experimental data spec (optional; CSV can be used instead)", padding=(8, 6))
        exp_box.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        exp_box.columnconfigure(0, weight=1)
        ttk.Label(
            exp_box,
            text="Manual examples: 245:561,600-610    or    run=245; images=561,600-610; base=/path/to/runs. CSV format: col1=run, col2=image, no header.",
        ).grid(row=0, column=0, sticky="w")
        self.exp_text = tk.Text(exp_box, height=5, wrap="none")
        self.exp_text.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.exp_text.insert("1.0", "245:561\n")
        try:
            self.exp_text.bind("<KeyRelease>", lambda _e: self._update_local_scan_estimate(), add="+")
            self.exp_text.bind("<<Paste>>", lambda _e: self.after(50, self._update_local_scan_estimate), add="+")
        except Exception:
            pass
        thresh = ttk.LabelFrame(top, text="Acceptance thresholds by criterion type", padding=(8, 6))
        thresh.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        for col in range(4):
            thresh.columnconfigure(col, weight=1)
        gate_bar = ttk.Frame(thresh)
        gate_bar.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 6))
        ttk.Label(gate_bar, text="Hard-gate Use:").pack(side=tk.LEFT)
        ttk.Button(gate_bar, text="Select all gates", command=lambda: self._set_all_threshold_gates(True)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(gate_bar, text="Clear all gates", command=lambda: self._set_all_threshold_gates(False)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            gate_bar,
            text="Only checked gates are used for accepted/rejected. Cutoff values remain saved even when a gate is off.",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(12, 0))
        def _add_threshold_group(parent, title, items, row, col, colspan=1):
            group = ttk.LabelFrame(parent, text=title, padding=(8, 6))
            group.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=(0 if col == 0 else 8, 0), pady=(0 if row == 0 else 8, 0))
            ttk.Label(group, text="Use", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 4), pady=(0, 2))
            ttk.Label(group, text="Gate", font=("Arial", 8, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 4), pady=(0, 2))
            ttk.Label(group, text="Cutoff", font=("Arial", 8, "bold")).grid(row=0, column=2, sticky="w", padx=(4, 0), pady=(0, 2))
            for idx, item in enumerate(items, start=1):
                if len(item) == 5:
                    label, var, width, tip_key, gate_key = item
                elif len(item) == 4:
                    label, var, width, tip_key = item
                    gate_key = ""
                else:
                    label, var, width = item
                    tip_key = ""
                    gate_key = ""
                gate_var = (getattr(self, "threshold_gate_vars", {}) or {}).get(gate_key)
                if gate_var is not None:
                    chk = ttk.Checkbutton(group, variable=gate_var)
                    chk.grid(row=idx, column=0, sticky="w", padx=(0, 4), pady=2)
                else:
                    chk = None
                    ttk.Label(group, text="").grid(row=idx, column=0, sticky="w", padx=(0, 4), pady=2)
                lbl = ttk.Label(group, text=label)
                lbl.grid(row=idx, column=1, sticky="w", padx=(0, 4), pady=2)
                ent = ttk.Entry(group, textvariable=var, width=width)
                ent.grid(row=idx, column=2, sticky="w", padx=(4, 0), pady=2)
                tip_text = self._threshold_tooltip_text(tip_key)
                if tip_text:
                    if chk is not None:
                        self._make_tooltip(chk, tip_text)
                    self._make_tooltip(lbl, tip_text)
                    self._make_tooltip(ent, tip_text)
            return group
        _add_threshold_group(
            thresh,
            "1D radial",
            [
                ("radial corr ≥", self.radial_corr_min_var, 8, "radial_corr", "radial_corr"),
                ("radial NRMSE ≤", self.radial_nrmse_max_var, 8, "radial_nrmse", "radial_nrmse"),
                ("radial χ²red ≤", self.radial_chi2_max_var, 8, "radial_chi2", "radial_chi2"),
            ],
            row=1,
            col=0,
        )
        _add_threshold_group(
            thresh,
            "1D integration / oscillation",
            [
                ("1D curve corr ≥", self.integration_curve_corr_min_var, 8, "int_corr", "int_corr"),
                ("1D curve NRMSE ≤", self.integration_curve_nrmse_max_var, 8, "int_nrmse", "int_nrmse"),
                ("1D curve χ²red ≤", self.integration_chi2_max_var, 8, "int_chi2", "int_chi2"),
                ("fringe pos err ≤", self.fringe_position_error_max_var, 8, "fringe", "fringe"),
            ],
            row=1,
            col=1,
        )
        _add_threshold_group(
            thresh,
            "2D anisotropy",
            [
                ("2D pixel corr ≥", self.pixel_corr_min_var, 8, "pixel_corr", "pixel_corr"),
                ("harmonic corr ≥", self.harmonic_corr_min_var, 8, "harmonic_corr", "harmonic_corr"),
                ("A2–A6 phase err ≤ deg", self.a2_phase_error_max_var, 8, "a2_phase", "harmonic_phase"),
                ("angular corr ≥", self.angular_corr_min_var, 8, "angular_corr", "angular_corr"),
            ],
            row=1,
            col=2,
        )
        _add_threshold_group(
            thresh,
            "Overall decision",
            [
                ("total score ≤", self.total_score_max_var, 8, "total_score", "total_score"),
            ],
            row=1,
            col=3,
        )
        self._build_score_terms_panel(top, row=9, col=0, colspan=3)
        mid = ttk.Frame(body, padding=(12, 0, 12, 0))
        mid.grid(row=1, column=0, sticky="nsew")
        mid.columnconfigure(0, weight=1)
        mid.rowconfigure(0, weight=1)
        canvas = tk.Canvas(mid, highlightthickness=0, height=320)
        scrollbar = ttk.Scrollbar(mid, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        inner_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        def _resize_param_inner(event):
            try:
                canvas.itemconfigure(inner_window, width=event.width)
            except Exception:
                pass
        canvas.bind("<Configure>", _resize_param_inner)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        # Local-scan helper is placed above the Auto Screen parameter table.
        # It fills start/stop/step for rows checked in the table below.
        self._build_inline_local_scan_panel(inner)
        # Select/Clear controls are placed immediately above the actual parameter rows
        # so their target is visually unambiguous.
        self._build_scan_parameter_select_bar(inner)
        hdr = ttk.Frame(inner)
        hdr.pack(fill=tk.X, pady=(4, 6))
        headers = [("Use", 6), ("Parameter", 22), ("Current", 12), ("Start", 12), ("Stop", 12), ("Step", 12), ("Points", 10)]
        for idx, (label, width) in enumerate(headers):
            ttk.Label(hdr, text=label, width=width, font=("Arial", 10, "bold")).grid(row=0, column=idx, sticky="w", padx=2)
        vars_dict = getattr(self.app, "vars", {}) or {}
        control_specs = getattr(self.app, "control_specs", {}) or {}
        for name, var in vars_dict.items():
            try:
                current = float(var.get())
            except Exception:
                continue
            spec = control_specs.get(name, {})
            default_step = spec.get("step", None)
            if default_step in (None, 0, ""):
                span = abs(spec.get("max", current) - spec.get("min", current))
                default_step = span / 10.0 if span > 0 else 1.0
            default_step = float(default_step)
            use_var = tk.BooleanVar(value=False)
            current_var = tk.StringVar(value=_format_param_value(current))
            start_var = tk.StringVar(value=_format_param_value(current))
            stop_var = tk.StringVar(value=_format_param_value(current))
            step_var = tk.StringVar(value=_format_param_value(default_step))
            points_var = tk.StringVar(value="1")
            frame = ttk.Frame(inner)
            frame.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(frame, variable=use_var, command=lambda n=name: self._on_param_use_changed(n)).grid(row=0, column=0, sticky="w", padx=2)
            ttk.Label(frame, text=name, width=22).grid(row=0, column=1, sticky="w", padx=2)
            ttk.Label(frame, textvariable=current_var, width=12).grid(row=0, column=2, sticky="w", padx=2)
            ttk.Entry(frame, textvariable=start_var, width=12).grid(row=0, column=3, sticky="w", padx=2)
            ttk.Entry(frame, textvariable=stop_var, width=12).grid(row=0, column=4, sticky="w", padx=2)
            ttk.Entry(frame, textvariable=step_var, width=12).grid(row=0, column=5, sticky="w", padx=2)
            ttk.Label(frame, textvariable=points_var, width=10).grid(row=0, column=6, sticky="w", padx=2)
            for sv in (start_var, stop_var, step_var):
                sv.trace_add("write", lambda *_args, n=name: self._update_points(n))
            self.param_rows[name] = {
                "use": use_var,
                "start": start_var,
                "stop": stop_var,
                "step": step_var,
                "points": points_var,
                "current": current,
                "current_var": current_var,
                "default_step": default_step,
                "spec": spec,
            }
            self._update_points(name)
        self._update_total_combinations()
        self._update_local_scan_estimate()
        bottom = ttk.Frame(self, padding=(12, 8, 12, 10))
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.progress_var, wraplength=760).grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(bottom, textvariable=self.total_combinations_var, font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=(6, 0))
        actions = ttk.Frame(bottom)
        actions.grid(row=1, column=1, sticky="e", pady=(6, 0))
        self.close_button = ttk.Button(actions, text="Close", command=self._on_close)
        self.close_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.request_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.test_output_button = ttk.Button(actions, text="Preview one image", command=self.test_output_one_image)
        self.test_output_button.pack(side=tk.RIGHT, padx=(8, 0))
        self.run_button = ttk.Button(actions, text="Run screening", command=self.run_screening)
        self.run_button.pack(side=tk.RIGHT)
        self._bind_autoscreen_mousewheel_tree(body)
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            body_canvas.bind(sequence, _on_body_mousewheel, add="+")
    def _choose_output_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Choose screening output folder", mustexist=False)
        if folder:
            self.output_dir_var.set(folder)
    def _choose_cache_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Choose simulation cache root folder", mustexist=False)
        if folder:
            self.cache_dir_var.set(folder)
    def _resolve_simulator_cache_dir(self):
        """Return cache_root/<simulator_name>/, creating a simulator-specific cache folder."""
        cache_root = Path(self.cache_dir_var.get()).expanduser()
        simulator_slug = _safe_filename_fragment(self.simulator_name)
        return cache_root / simulator_slug
    def _choose_exp_base_folder(self):
        folder = filedialog.askdirectory(parent=self, title="Choose experimental base folder", mustexist=False)
        if folder:
            self.default_exp_base_dir_var.set(folder)
    def _choose_exp_csv_file(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Choose experimental CSV file",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.exp_csv_path_var.set(path)
    def _build_inline_local_scan_panel(self, parent):
        """Build the local-scan range helper directly above the parameter table."""
        box = ttk.LabelFrame(parent, text="Local scan from current parameters", padding=(10, 8))
        box.pack(fill=tk.X, pady=(4, 10))
        box.columnconfigure(1, weight=1)
        ttk.Label(
            box,
            text="Set the experimental target above, choose local range rules here, then check parameter rows below and fill ranges around the current simulator values.",
            foreground="#444444",
        ).grid(row=0, column=0, columnspan=8, sticky="w", pady=(0, 6))
        ttk.Label(box, text="Normal ±% ").grid(row=1, column=0, sticky="w", padx=(0, 4))
        ttk.Entry(box, textvariable=self.local_scan_percent_var, width=7).grid(row=1, column=1, sticky="w", padx=(0, 12))
        ttk.Label(box, text="Angle ±deg ").grid(row=1, column=2, sticky="w", padx=(0, 4))
        ttk.Entry(box, textvariable=self.local_scan_angle_span_var, width=7).grid(row=1, column=3, sticky="w", padx=(0, 12))
        ttk.Label(box, text="Points ").grid(row=1, column=4, sticky="w", padx=(0, 4))
        ttk.Entry(box, textvariable=self.local_scan_points_var, width=6).grid(row=1, column=5, sticky="w", padx=(0, 12))
        self.local_scan_button = ttk.Button(
            box,
            text="Fill checked rows from current",
            command=self.fill_local_scan_from_current,
        )
        self.local_scan_button.grid(row=1, column=6, sticky="w", padx=(0, 6))
        ttk.Button(
            box,
            text="Refresh current values",
            command=self._refresh_current_values_for_local_scan,
        ).grid(row=1, column=7, sticky="w", padx=(0, 6))
        self.local_scan_run_button = ttk.Button(
            box,
            text="Run local scan + scoring summary",
            command=self.run_local_scan,
        )
        self.local_scan_run_button.grid(row=1, column=8, sticky="w", padx=(0, 6))
        ttk.Label(
            box,
            text="Type rule: normal → current ±%; angle/theta/phi/roll/... → current ±deg. Fill applies only to checked rows in the parameter table below.",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=10, sticky="w", pady=(6, 0))
        ttk.Label(
            box,
            text="Run local scan fills checked rows, runs screening, and writes extra local-scan scoring summary files.",
            foreground="#555555",
        ).grid(row=3, column=0, columnspan=10, sticky="w", pady=(3, 0))
        ttk.Label(box, textvariable=self.local_scan_status_var, foreground="#555555").grid(row=4, column=0, columnspan=10, sticky="w", pady=(4, 0))
        ttk.Label(box, textvariable=self.local_scan_estimate_var, foreground="#003366", font=("Arial", 10, "bold")).grid(row=5, column=0, columnspan=10, sticky="w", pady=(4, 0))

    def _build_scan_parameter_select_bar(self, parent):
        """Build select/clear controls immediately above the parameter Use checkboxes."""
        scan_select_bar = ttk.Frame(parent)
        scan_select_bar.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(scan_select_bar, text="Scan-row Use:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        ttk.Button(
            scan_select_bar,
            text="Select all parameters",
            command=lambda: self._set_all_scan_parameters(True),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(
            scan_select_bar,
            text="Clear all parameters",
            command=lambda: self._set_all_scan_parameters(False),
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(
            scan_select_bar,
            text="Applies to the parameter Use checkboxes below.",
            foreground="#555555",
        ).pack(side=tk.LEFT, padx=(12, 0))
        return scan_select_bar

    def _experimental_spec_list_for_screening(self):
        """Return parsed experimental run/image specs without loading image arrays."""
        default_base = self.default_exp_base_dir_var.get()
        specs = []
        csv_path = str(self.exp_csv_path_var.get()).strip()
        if csv_path:
            specs.extend(_parse_experimental_csv_file(csv_path, default_base))
        manual_text = self.exp_text.get("1.0", "end")
        if manual_text.strip() and not (csv_path and manual_text.strip() == "245:561"):
            specs.extend(_parse_experimental_dataset_spec(manual_text, default_base))
        unique_specs = []
        seen = set()
        for spec in specs:
            try:
                key = (str(spec["base_dir"]), int(spec["run_number"]), int(spec["image_number"]))
            except Exception:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique_specs.append(spec)
        return unique_specs

    def _experimental_target_summary_for_local_scan(self):
        """Compact user-facing summary of the target images used by Auto Screen."""
        specs = self._experimental_spec_list_for_screening()
        if not specs:
            return specs, "No experimental target set."
        first = specs[0]
        source = "CSV" if str(first.get("source", "")).lower() == "csv" or str(self.exp_csv_path_var.get()).strip() else "manual"
        try:
            first_text = f"run {int(first['run_number'])}, image {int(first['image_number'])}"
        except Exception:
            first_text = "first target unknown"
        if len(specs) == 1:
            return specs, f"Target: {first_text} ({source})"
        return specs, f"Targets: {len(specs)} images; first = {first_text} ({source})"

    def open_local_scan_setup_dialog(self):
        """Open a separate local-scan setup dialog.

        The Auto Screen table keeps the final scan ranges and active Use flags.
        This dialog has its own temporary parameter checkboxes so the user can
        decide which parameters should be filled from the current simulator state
        without confusing them with the main screening table.
        """
        try:
            target_specs, target_summary = self._experimental_target_summary_for_local_scan()
        except Exception as exc:
            messagebox.showerror(
                "Local scan setup",
                f"Experimental target specification is not valid.\n\n{type(exc).__name__}: {exc}",
                parent=self,
            )
            return
        if not target_specs:
            messagebox.showwarning(
                "Local scan setup",
                "Set the experimental target in the Auto Screen window before building a local scan.\n\n"
                "Use either Experimental CSV or Manual experimental data spec, then open Local scan setup again.",
                parent=self,
            )
            self.local_scan_status_var.set("Local scan needs an experimental target first.")
            return
        self._refresh_param_current_values()
        dialog = tk.Toplevel(self)
        dialog.title("Local scan setup")
        dialog.geometry("1040x710")
        dialog.transient(self)
        try:
            dialog.grab_set()
        except Exception:
            pass
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(2, weight=1)

        target_box = ttk.LabelFrame(dialog, text="Experimental target for local scan", padding=(10, 8))
        target_box.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        target_box.columnconfigure(1, weight=1)
        ttk.Label(target_box, text="Target", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(target_box, text=target_summary, foreground="#333333").grid(row=0, column=1, sticky="w")
        ttk.Label(
            target_box,
            text="Local scan will compare candidates against this experimental target when Preview/Run screening is executed.",
            foreground="#666666",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        top = ttk.LabelFrame(dialog, text="Range settings", padding=(10, 8))
        top.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        top.columnconfigure(11, weight=1)
        percent_var = tk.StringVar(master=dialog, value=str(self.local_scan_percent_var.get()))
        angle_var = tk.StringVar(master=dialog, value=str(self.local_scan_angle_span_var.get()))
        points_var = tk.StringVar(master=dialog, value=str(self.local_scan_points_var.get()))
        selected_only_var = tk.BooleanVar(master=dialog, value=True)
        status_var = tk.StringVar(master=dialog, value="Select parameters below, then apply ranges to Auto Screen.")

        ttk.Label(top, text="Normal ±%", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        normal_entry = ttk.Entry(top, textvariable=percent_var, width=7)
        normal_entry.grid(row=0, column=1, sticky="w", padx=(4, 14))
        ttk.Label(top, text="Angle ±deg", font=("Arial", 10)).grid(row=0, column=2, sticky="w")
        angle_entry = ttk.Entry(top, textvariable=angle_var, width=7)
        angle_entry.grid(row=0, column=3, sticky="w", padx=(4, 14))
        ttk.Label(top, text="Points", font=("Arial", 10)).grid(row=0, column=4, sticky="w")
        points_entry = ttk.Entry(top, textvariable=points_var, width=6)
        points_entry.grid(row=0, column=5, sticky="w", padx=(4, 14))
        ttk.Checkbutton(
            top,
            text="Use selected only in Auto Screen",
            variable=selected_only_var,
        ).grid(row=0, column=6, columnspan=3, sticky="w", padx=(0, 14))
        ttk.Label(
            top,
            textvariable=status_var,
            foreground="#555555",
        ).grid(row=0, column=9, columnspan=3, sticky="ew")

        guide = ttk.Label(
            dialog,
            text=(
                "Purpose: refine the current simulation around the experimental target shown above. "
                "After Apply, selected parameters become active scan variables in the main Auto Screen table."
            ),
            foreground="#555555",
        )
        guide.grid(row=3, column=0, sticky="ew", padx=12, pady=(2, 0))

        table_outer = ttk.Frame(dialog, padding=(10, 0, 10, 0))
        table_outer.grid(row=2, column=0, sticky="nsew")
        table_outer.columnconfigure(0, weight=1)
        table_outer.rowconfigure(0, weight=1)
        canvas = tk.Canvas(table_outer, highlightthickness=0)
        yscroll = ttk.Scrollbar(table_outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        def _resize_inner(event):
            try:
                canvas.itemconfigure(canvas_window, width=event.width)
            except Exception:
                pass
        canvas.bind("<Configure>", _resize_inner)

        header = ttk.Frame(inner)
        header.pack(fill=tk.X, pady=(4, 6))
        headers = [
            ("Local", 7),
            ("Parameter", 26),
            ("Current", 12),
            ("Type", 10),
            ("Start", 12),
            ("Stop", 12),
            ("Step", 12),
            ("N", 6),
        ]
        for col, (label, width) in enumerate(headers):
            ttk.Label(header, text=label, width=width, font=("Arial", 10, "bold")).grid(
                row=0, column=col, sticky="w", padx=2
            )

        selection_vars = {}
        preview_vars = {}
        for name, row in self.param_rows.items():
            use_var = tk.BooleanVar(master=dialog, value=bool(row.get("use").get()))
            selection_vars[name] = use_var
            start_preview = tk.StringVar(master=dialog, value="")
            stop_preview = tk.StringVar(master=dialog, value="")
            step_preview = tk.StringVar(master=dialog, value="")
            n_preview = tk.StringVar(master=dialog, value="")
            preview_vars[name] = {
                "start": start_preview,
                "stop": stop_preview,
                "step": step_preview,
                "n": n_preview,
            }
            frame = ttk.Frame(inner)
            frame.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(frame, variable=use_var, command=lambda: recalc_all()).grid(row=0, column=0, sticky="w", padx=2)
            ttk.Label(frame, text=name, width=26).grid(row=0, column=1, sticky="w", padx=2)
            ttk.Label(frame, text=row.get("current_var").get(), width=12).grid(row=0, column=2, sticky="w", padx=2)
            kind = "angle" if self._is_angle_like_parameter(name) else "normal"
            ttk.Label(frame, text=kind, width=10).grid(row=0, column=3, sticky="w", padx=2)
            ttk.Label(frame, textvariable=start_preview, width=12).grid(row=0, column=4, sticky="w", padx=2)
            ttk.Label(frame, textvariable=stop_preview, width=12).grid(row=0, column=5, sticky="w", padx=2)
            ttk.Label(frame, textvariable=step_preview, width=12).grid(row=0, column=6, sticky="w", padx=2)
            ttk.Label(frame, textvariable=n_preview, width=6).grid(row=0, column=7, sticky="w", padx=2)

        def _parse_dialog_values():
            percent_span = self._safe_float_from_var(percent_var, 10.0)
            angle_span = self._safe_float_from_var(angle_var, 10.0)
            try:
                n_points = int(round(self._safe_float_from_var(points_var, 5)))
            except Exception:
                n_points = 5
            n_points = max(1, min(101, n_points))
            return percent_span, angle_span, n_points

        def recalc_all():
            percent_span, angle_span, n_points = _parse_dialog_values()
            selected_count = 0
            total = 1
            for pname, prow in self.param_rows.items():
                pv = preview_vars[pname]
                try:
                    current = float(prow.get("current", 0.0))
                    start, stop, step = self._local_scan_range_for_parameter(
                        name=pname,
                        current=current,
                        row=prow,
                        percent_span=percent_span,
                        angle_span=angle_span,
                        n_points=n_points,
                    )
                    pv["start"].set(_format_param_value(start))
                    pv["stop"].set(_format_param_value(stop))
                    pv["step"].set(_format_param_value(step))
                    try:
                        n_val = len(_inclusive_float_range(start, stop, step))
                    except Exception:
                        n_val = n_points if bool(selection_vars[pname].get()) else 1
                    pv["n"].set(str(n_val))
                    if bool(selection_vars[pname].get()):
                        selected_count += 1
                        total *= max(1, int(n_val))
                except Exception:
                    pv["start"].set("?")
                    pv["stop"].set("?")
                    pv["step"].set("?")
                    pv["n"].set("?")
            status_var.set(f"Selected {selected_count} parameter(s); estimated combinations: {total:,}")

        def _set_all(value):
            for v in selection_vars.values():
                v.set(bool(value))
            recalc_all()

        def _set_from_active():
            for pname, v in selection_vars.items():
                try:
                    v.set(bool(self.param_rows[pname]["use"].get()))
                except Exception:
                    v.set(False)
            recalc_all()

        def _refresh_current():
            self._refresh_param_current_values()
            # Rebuild current labels would require recreating rows; preview ranges are
            # still updated from refreshed current values, which is the important part.
            recalc_all()
            status_var.set(status_var.get() + " | current values refreshed")

        def _apply_and_close():
            try:
                latest_specs, latest_summary = self._experimental_target_summary_for_local_scan()
            except Exception as exc:
                messagebox.showerror(
                    "Local scan setup",
                    f"Experimental target specification is not valid.\n\n{type(exc).__name__}: {exc}",
                    parent=dialog,
                )
                return
            if not latest_specs:
                messagebox.showwarning(
                    "Local scan setup",
                    "Experimental target was cleared. Set Experimental CSV or Manual experimental data spec before applying local scan ranges.",
                    parent=dialog,
                )
                return
            percent_span, angle_span, n_points = _parse_dialog_values()
            selected_names = [pname for pname, v in selection_vars.items() if bool(v.get())]
            if not selected_names:
                messagebox.showinfo(
                    "Local scan setup",
                    "Select at least one parameter for local scanning.",
                    parent=dialog,
                )
                return
            self.local_scan_percent_var.set(_format_param_value(percent_span))
            self.local_scan_angle_span_var.set(_format_param_value(angle_span))
            self.local_scan_points_var.set(str(n_points))
            if bool(selected_only_var.get()):
                for prow in self.param_rows.values():
                    try:
                        prow["use"].set(False)
                    except Exception:
                        pass
            filled = 0
            skipped = []
            for pname in selected_names:
                prow = self.param_rows[pname]
                try:
                    current = float(prow.get("current", 0.0))
                    start, stop, step = self._local_scan_range_for_parameter(
                        name=pname,
                        current=current,
                        row=prow,
                        percent_span=percent_span,
                        angle_span=angle_span,
                        n_points=n_points,
                    )
                    prow["use"].set(True)
                    prow["start"].set(_format_param_value(start))
                    prow["stop"].set(_format_param_value(stop))
                    prow["step"].set(_format_param_value(step))
                    self._update_points(pname)
                    filled += 1
                except Exception as exc:
                    skipped.append(f"{pname}: {exc}")
            self._update_total_combinations()
            self.local_scan_status_var.set(
                f"Local scan applied to {filled} parameter(s) for {latest_summary}: normal ±{percent_span:g}%, angle ±{angle_span:g} deg, {n_points} points."
            )
            if skipped:
                messagebox.showwarning(
                    "Local scan setup",
                    "Some parameters could not be filled:\n" + "\n".join(skipped[:8]),
                    parent=dialog,
                )
            dialog.destroy()

        for trace_var in (percent_var, angle_var, points_var):
            trace_var.trace_add("write", lambda *_args: recalc_all())
        recalc_all()

        bottom = ttk.Frame(dialog, padding=(10, 8))
        bottom.grid(row=4, column=0, sticky="ew")
        ttk.Button(bottom, text="Select all", command=lambda: _set_all(True)).pack(side=tk.LEFT)
        ttk.Button(bottom, text="Select none", command=lambda: _set_all(False)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bottom, text="Select active in Auto Screen", command=_set_from_active).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bottom, text="Refresh current values", command=_refresh_current).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(bottom, text="Close", command=dialog.destroy).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(bottom, text="Apply to Auto Screen", command=_apply_and_close).pack(side=tk.RIGHT)

        def _on_mousewheel(event):
            try:
                delta = -1 * int(event.delta / 120)
            except Exception:
                delta = 1 if getattr(event, "num", None) == 5 else -1
            canvas.yview_scroll(delta, "units")
            return "break"
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind(seq, _on_mousewheel, add="+")
            inner.bind(seq, _on_mousewheel, add="+")

    def _refresh_current_values_for_local_scan(self):
        refreshed = self._refresh_param_current_values()
        self.local_scan_status_var.set(f"Current values refreshed from simulator controls ({refreshed} parameter(s)).")

    def _uncheck_all_parameters(self):
        self._set_all_scan_parameters(False)

    @staticmethod
    def _is_angle_like_parameter(name):
        text = str(name).lower()
        angle_tokens = ("angle", "theta", "phi", "roll", "pitch", "yaw", "rotation", "azimuth", "tilt")
        return any(tok in text for tok in angle_tokens)

    @staticmethod
    def _safe_float_from_var(var, default):
        try:
            value = float(str(var.get()).strip())
            if not math.isfinite(value):
                return float(default)
            return value
        except Exception:
            return float(default)

    @staticmethod
    def _spec_bound(spec, key):
        try:
            value = float(spec.get(key))
            return value if math.isfinite(value) else None
        except Exception:
            return None

    def _refresh_param_current_values(self):
        """Refresh the table's current column from the live simulator controls."""
        vars_dict = getattr(self.app, "vars", {}) or {}
        refreshed = 0
        for name, row in self.param_rows.items():
            var = vars_dict.get(name)
            if var is None:
                continue
            try:
                current = float(var.get())
                if not math.isfinite(current):
                    continue
            except Exception:
                continue
            row["current"] = current
            try:
                row["current_var"].set(_format_param_value(current))
            except Exception:
                pass
            refreshed += 1
        return refreshed

    def _local_scan_range_for_parameter(self, name, current, row, percent_span, angle_span, n_points):
        spec = row.get("spec", {}) or {}
        default_step = row.get("default_step", None)
        min_bound = self._spec_bound(spec, "min")
        max_bound = self._spec_bound(spec, "max")
        if self._is_angle_like_parameter(name):
            half_span = abs(float(angle_span))
        else:
            pct = abs(float(percent_span)) / 100.0
            half_span = abs(float(current)) * pct
            if half_span <= 0:
                if min_bound is not None and max_bound is not None and max_bound > min_bound:
                    half_span = abs(max_bound - min_bound) * pct * 0.5
                else:
                    try:
                        half_span = abs(float(default_step)) * max(1.0, (max(int(n_points), 2) - 1) / 2.0)
                    except Exception:
                        half_span = 1.0
            if half_span <= 0:
                half_span = 1.0
        start = float(current) - half_span
        stop = float(current) + half_span
        if min_bound is not None:
            start = max(start, min_bound)
            stop = max(stop, min_bound)
        if max_bound is not None:
            start = min(start, max_bound)
            stop = min(stop, max_bound)
        if stop < start:
            start, stop = stop, start
        if int(n_points) <= 1 or abs(stop - start) <= 0:
            step = abs(float(default_step)) if default_step not in (None, 0, "") else 1.0
            if step <= 0 or not math.isfinite(step):
                step = 1.0
            start = stop = float(current)
        else:
            step = abs(stop - start) / float(int(n_points) - 1)
            if step <= 0 or not math.isfinite(step):
                step = abs(float(default_step)) if default_step not in (None, 0, "") else 1.0
        return start, stop, step

    def fill_local_scan_from_current(self):
        """Fill start/stop/step for checked parameters around the live current values."""
        try:
            _target_specs, target_summary = self._experimental_target_summary_for_local_scan()
        except Exception:
            messagebox.showinfo(
                "Local scan",
                "Set the experimental target in the Auto Screen window before building a local scan.\n\n"
                "Use either Experimental CSV or Manual experimental data spec above, then try again.",
                parent=self,
            )
            self.local_scan_status_var.set("Local scan needs an experimental target first.")
            return 0
        self._refresh_param_current_values()
        checked = [(name, row) for name, row in self.param_rows.items() if bool(row["use"].get())]
        if not checked:
            messagebox.showinfo(
                "Local scan",
                "Check one or more parameters first, then click 'Fill checked from current'.\n\n"
                "This avoids accidentally creating a huge full-dimensional sweep.",
                parent=self,
            )
            self.local_scan_status_var.set("No parameter checked. Check parameters to scan first.")
            return 0
        percent_span = self._safe_float_from_var(self.local_scan_percent_var, 10.0)
        angle_span = self._safe_float_from_var(self.local_scan_angle_span_var, 10.0)
        try:
            n_points = int(round(self._safe_float_from_var(self.local_scan_points_var, 5)))
        except Exception:
            n_points = 5
        n_points = max(1, min(101, n_points))
        self.local_scan_points_var.set(str(n_points))
        filled = 0
        skipped = []
        for name, row in checked:
            try:
                current = float(row.get("current", 0.0))
                start, stop, step = self._local_scan_range_for_parameter(
                    name=name,
                    current=current,
                    row=row,
                    percent_span=percent_span,
                    angle_span=angle_span,
                    n_points=n_points,
                )
                row["start"].set(_format_param_value(start))
                row["stop"].set(_format_param_value(stop))
                row["step"].set(_format_param_value(step))
                self._update_points(name)
                filled += 1
            except Exception as exc:
                skipped.append(f"{name}: {exc}")
        self._update_total_combinations()
        if skipped:
            messagebox.showwarning(
                "Local scan",
                "Some parameters could not be filled:\n" + "\n".join(skipped[:8]),
                parent=self,
            )
        self.local_scan_status_var.set(
            f"Filled {filled} checked parameter(s) for {target_summary}: normal ±{percent_span:g}%, angle ±{angle_span:g} deg, {n_points} points."
        )
        self._update_local_scan_estimate()
        return int(filled)

    def _on_param_use_changed(self, name):
        self._update_points(name)
        self._update_local_scan_estimate()

    def _format_large_int(self, value):
        try:
            value = int(value)
        except Exception:
            return "?"
        return f"{value:,}"

    def _update_local_scan_estimate(self, *_args):
        """Show how many simulations/comparisons the local scan would create.

        The estimate is based on the currently checked parameter rows and the
        Points value in the local-scan panel, so the user can see the
        combinatorial growth before pressing Fill/Run.
        """
        try:
            checked = [(name, row) for name, row in self.param_rows.items() if bool(row["use"].get())]
        except Exception:
            checked = []
        active = len(checked)
        try:
            n_points = int(round(self._safe_float_from_var(self.local_scan_points_var, 5)))
        except Exception:
            n_points = 5
        n_points = max(1, min(101, n_points))
        combos = None
        if active <= 0:
            combos = 1
            combo_part = "no checked parameter -> 1 simulation"
        else:
            try:
                combos = int(n_points) ** int(active)
            except Exception:
                combos = None
            if combos is None:
                combo_part = f"{active} checked x {n_points} points each -> ? simulations"
            else:
                combo_part = f"{active} checked x {n_points} points each -> {self._format_large_int(combos)} simulations"
        try:
            target_count = len(self._experimental_spec_list_for_screening())
        except Exception:
            target_count = 0
        if target_count > 0 and combos is not None:
            compare_part = f"; targets {target_count} -> {self._format_large_int(combos * target_count)} comparisons"
        elif target_count > 0:
            compare_part = f"; targets {target_count} -> ? comparisons"
        else:
            compare_part = "; set experimental target to estimate comparisons"
        try:
            self.local_scan_estimate_var.set("Local estimate: " + combo_part + compare_part)
        except Exception:
            pass

    def _update_points(self, name):
        row = self.param_rows[name]
        if not row["use"].get():
            row["points"].set("1")
            self._update_total_combinations()
            self._update_local_scan_estimate()
            return
        try:
            values = _inclusive_float_range(row["start"].get(), row["stop"].get(), row["step"].get())
            row["points"].set(str(len(values)))
        except Exception:
            row["points"].set("?")
        self._update_total_combinations()
        self._update_local_scan_estimate()
    def _update_total_combinations(self):
        total = 1
        invalid = False
        active = 0
        for row in self.param_rows.values():
            if row["use"].get():
                active += 1
                try:
                    points = int(row["points"].get())
                except Exception:
                    invalid = True
                    break
                total *= points
        if invalid:
            self.total_combinations_var.set("Total combinations: ?")
        elif active == 0:
            self.total_combinations_var.set("Total combinations: 1 (current settings only)")
        else:
            self.total_combinations_var.set(f"Total combinations: {self._format_large_int(total)}")
        try:
            self._update_local_scan_estimate()
        except Exception:
            pass
    def _collect_sweep(self):
        param_values = {}
        total = 1
        for name, row in self.param_rows.items():
            if row["use"].get():
                values = _inclusive_float_range(row["start"].get(), row["stop"].get(), row["step"].get())
            else:
                values = [row["current"]]
            if len(values) == 0:
                raise ValueError(f"No values generated for parameter '{name}'.")
            param_values[name] = values
            total *= len(values)
        return param_values, total
    def _set_running_state(self, running):
        self.is_running = bool(running)
        if running:
            self.run_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.close_button.configure(state=tk.DISABLED)
            if hasattr(self, "test_output_button"):
                self.test_output_button.configure(state=tk.DISABLED)
            if hasattr(self, "local_scan_button"):
                self.local_scan_button.configure(state=tk.DISABLED)
            if hasattr(self, "local_scan_run_button"):
                self.local_scan_run_button.configure(state=tk.DISABLED)
        else:
            self.run_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.close_button.configure(state=tk.NORMAL)
            if hasattr(self, "test_output_button"):
                self.test_output_button.configure(state=tk.NORMAL)
            if hasattr(self, "local_scan_button"):
                self.local_scan_button.configure(state=tk.NORMAL)
            if hasattr(self, "local_scan_run_button"):
                self.local_scan_run_button.configure(state=tk.NORMAL)
    def request_stop(self):
        if self.is_running:
            self.cancel_requested = True
            self.progress_var.set("Stopping after the current comparison finishes...")
            self.update_idletasks()
    def _on_close(self):
        if self.is_running:
            self.request_stop()
            return
        self.destroy()
    def _load_experimental_records(self):
        default_base = self.default_exp_base_dir_var.get()
        specs = []
        csv_path = str(self.exp_csv_path_var.get()).strip()
        if csv_path:
            specs.extend(_parse_experimental_csv_file(csv_path, default_base))
        manual_text = self.exp_text.get("1.0", "end")
        # Keep manual input available, but ignore the default placeholder when a CSV
        # has been selected and the user has not edited the manual field.
        if manual_text.strip() and not (csv_path and manual_text.strip() == "245:561"):
            specs.extend(_parse_experimental_dataset_spec(manual_text, default_base))
        if not specs:
            raise ValueError(
                "No experimental data were specified. Provide a CSV file or enter manual run:image values."
            )
        # De-duplicate repeated run/image/base entries while preserving order.
        unique_specs = []
        seen = set()
        for spec in specs:
            key = (str(spec["base_dir"]), int(spec["run_number"]), int(spec["image_number"]))
            if key in seen:
                continue
            seen.add(key)
            unique_specs.append(spec)
        loaded = []
        for spec in unique_specs:
            run_source, run_name = _find_run_source(spec["base_dir"], spec["run_number"])
            image, npz_name, local_number = _load_run_image_number(run_source, spec["image_number"])
            image = np.asarray(image, dtype=float)
            valid_mask = _load_default_saxs_valid_mask(image.shape)
            q_u, q_v, _meta = _load_default_q_axes_from_poni(image.shape)
            loaded.append({
                "base_dir": str(spec["base_dir"]),
                "run_number": int(spec["run_number"]),
                "run_name": run_name,
                "run_source": str(run_source),
                "image_number": int(spec["image_number"]),
                "npz_name": npz_name,
                "local_number": int(local_number),
                "id": f"{run_name}_img{int(spec['image_number']):04d}",
                "image": image,
                "valid_mask": valid_mask,
                "q_u": q_u,
                "q_v": q_v,
                "source": spec.get("source", "manual"),
                "csv_path": spec.get("csv_path", ""),
                "csv_row": spec.get("csv_row", ""),
            })
        return loaded
    def _thresholds(self):
        def parse(var, default):
            return _screening_safe_float(var.get(), default)
        # Feature extraction bins are fixed at 500 in the current workflow.
        # The former n_bins cutoff control was removed from the UI to avoid
        # treating this as an acceptance threshold.
        n_bins = 500
        return {
            "n_bins": n_bins,
            "gate_enabled": self._threshold_gate_enabled_map(),
            "radial_corr_min": parse(self.radial_corr_min_var, 0.90),
            "radial_nrmse_max": parse(self.radial_nrmse_max_var, 0.35),
            "radial_chi2_max": parse(self.radial_chi2_max_var, 10.0),
            "integration_curve_corr_min": parse(self.integration_curve_corr_min_var, 0.95),
            "integration_curve_nrmse_max": parse(self.integration_curve_nrmse_max_var, 0.25),
            "integration_chi2_max": parse(self.integration_chi2_max_var, 100.0),
            "fringe_position_error_max": parse(self.fringe_position_error_max_var, 0.15),
            "pixel_corr_min": parse(self.pixel_corr_min_var, 0.50),
            "harmonic_corr_min": parse(self.harmonic_corr_min_var, 0.30),
            "a2_phase_error_max_deg": parse(self.a2_phase_error_max_var, 30.0),
            "angular_corr_min": parse(self.angular_corr_min_var, 0.30),
            "total_score_max": parse(self.total_score_max_var, 0.65),
            "score_terms": self._score_terms_from_controls(),
        }
    def _cache_key(self, params):
        payload = {
            "version": "auto_screen_v1",
            "simulator": str(self.simulator_name),
            "params": _jsonify_params(params),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]
    def _extract_simulation_from_app(self):
        result = getattr(self.app, "last_result", None)
        if not isinstance(result, dict):
            raise ValueError("Current simulator does not expose last_result after update_plot().")
        # Auto-screen cache must always store the clean/noiseless simulation, even if
        # the Poisson-noise checkbox is currently enabled for interactive display.
        image = None
        image_key = None
        for key in ("I_masked_noiseless", "I_masked", "masked", "masked_theoretical"):
            if key in result:
                arr = np.asarray(result[key], dtype=float)
                if arr.ndim == 2:
                    image = arr.copy()
                    image_key = key
                    break
        if image is None:
            # Fall back to total intensity only if a dedicated masked image is not available.
            for key in ("I_total_noiseless", "I_total"):
                if key in result:
                    arr = np.asarray(result[key], dtype=float)
                    if arr.ndim == 2:
                        image = arr.copy()
                        image_key = key
                        break
        if image is None:
            raise ValueError("Could not find a 2D simulated intensity image in last_result.")
        total_clean = None
        for key in ("I_total_noiseless", "I_total"):
            if key in result:
                arr = np.asarray(result[key], dtype=float)
                if arr.ndim == 2:
                    total_clean = arr.copy()
                    break
        mask = None
        app_mask = getattr(self.app, "mask", None)
        if app_mask is not None:
            try:
                mask_arr = np.asarray(app_mask, dtype=float)
                if mask_arr.shape == image.shape:
                    mask = mask_arr.copy()
            except Exception:
                mask = None
        q_u = result.get("q_u", None)
        q_v = result.get("q_v", None)
        if q_u is not None:
            try:
                q_u = np.asarray(q_u, dtype=float)
            except Exception:
                q_u = None
        if q_v is not None:
            try:
                q_v = np.asarray(q_v, dtype=float)
            except Exception:
                q_v = None
        if q_u is None or q_v is None:
            q_u_default, q_v_default, _meta = _load_default_q_axes_from_poni(image.shape)
            q_u = q_u_default
            q_v = q_v_default
        return {
            "image": image,
            "image_total_clean": total_clean,
            "mask": mask,
            "q_u": q_u,
            "q_v": q_v,
            "image_key": image_key,
            "is_clean_cache": True,
        }
    def _load_simulation_cache(self, cache_path):
        with np.load(cache_path, allow_pickle=False) as data:
            image = data["I_masked"].astype(float)
            total_clean = data["I_total"].astype(float) if "I_total" in data else None
            mask = data["mask"].astype(float) if "mask" in data else None
            q_u = data["q_u"].astype(float) if "q_u" in data else None
            q_v = data["q_v"].astype(float) if "q_v" in data else None
            params_json = str(data["params_json"]) if "params_json" in data else "{}"
        return {
            "image": image.copy(),
            "image_total_clean": total_clean.copy() if isinstance(total_clean, np.ndarray) else total_clean,
            "mask": mask.copy() if isinstance(mask, np.ndarray) else mask,
            "q_u": q_u,
            "q_v": q_v,
            "params_json": params_json,
            "from_cache": True,
            "is_clean_cache": True,
        }
    def _save_simulation_cache(self, cache_path, sim_data, params):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "I_masked": np.asarray(sim_data["image"], dtype=np.float32),
            "params_json": np.array(json.dumps(_jsonify_params(params), sort_keys=True)),
            "simulator": np.array(str(self.simulator_name)),
            "created_unix": np.array(time.time()),
        }
        if sim_data.get("image_total_clean") is not None:
            arrays["I_total"] = np.asarray(sim_data["image_total_clean"], dtype=np.float32)
        if sim_data.get("mask") is not None:
            arrays["mask"] = np.asarray(sim_data["mask"], dtype=np.float32)
        if sim_data.get("q_u") is not None:
            arrays["q_u"] = np.asarray(sim_data["q_u"], dtype=np.float64)
        if sim_data.get("q_v") is not None:
            arrays["q_v"] = np.asarray(sim_data["q_v"], dtype=np.float64)
        np.savez_compressed(cache_path, **arrays)
    def _simulation_cache_path_for_combo(self, combo, cache_dir):
        """Return the deterministic simulation cache key/path for a parameter set."""
        key = self._cache_key(combo)
        return key, Path(cache_dir) / f"sim_{key}.npz"
    def _run_clean_update_plot_for_screening(self):
        """Run simulator update_plot while keeping app.last_result clean.
        AutoScreenDialog now owns its own scan-window Poisson controls. Therefore
        the object returned by _screening_poisson_source() can be this dialog
        itself. The clean-update delegate, however, must be the main controller
        or another external object that can temporarily suspend display-side
        Poisson mutation. Do not call this method on self with an extra app
        argument; otherwise screening can raise "takes 1 positional argument
        but 2 were given".
        """
        # Prefer external delegates that know how to suspend the GUI's display-side
        # Poisson update. Skip self, because this method accepts no extra app arg.
        delegate_sources = (
            getattr(self, "noise_source", None),
            getattr(self, "parent_controller", None),
            self.app,
        )
        for source in delegate_sources:
            if source is None or source is self:
                continue
            delegate = getattr(source, "_run_clean_update_plot_for_screening", None)
            if callable(delegate):
                try:
                    return delegate(self.app)
                except TypeError:
                    # Be tolerant of older helper signatures.
                    return delegate()
        # Fallback: directly suppress display-side Poisson processing on the best
        # available external controller while update_plot regenerates clean data.
        source = next((obj for obj in delegate_sources if obj is not None and obj is not self), None)
        old_flag = getattr(source, "_suspend_poisson_processing_update", False) if source is not None else False
        if source is not None:
            try:
                source._suspend_poisson_processing_update = True
            except Exception:
                pass
        try:
            return self.app.update_plot()
        finally:
            if source is not None:
                try:
                    source._suspend_poisson_processing_update = old_flag
                except Exception:
                    pass
    def _get_or_compute_simulation(self, combo, cache_dir):
        key, cache_path = self._simulation_cache_path_for_combo(combo, cache_dir)
        if cache_path.exists():
            sim_data = self._load_simulation_cache(cache_path)
            # Older cache files may contain only the masked image. When Poisson
            # noise is enabled, upgrade such cache entries by recomputing the clean
            # total-theoretical image + mask so the noisy processing matches the GUI.
            noise_enabled, _peak_counts, _seed, _ref_intensity, _ref_mode, _multiplier = self._screening_poisson_state()
            if noise_enabled and sim_data.get("image_total_clean") is None:
                for name, value in combo.items():
                    try:
                        self.app.vars[name].set(value)
                    except Exception:
                        pass
                self._run_clean_update_plot_for_screening()
                self.root.update_idletasks()
                sim_data = self._extract_simulation_from_app()
                self._save_simulation_cache(cache_path, sim_data, combo)
                sim_data["from_cache"] = False
            sim_data["cache_key"] = key
            sim_data["cache_path"] = str(cache_path)
            return sim_data
        for name, value in combo.items():
            try:
                self.app.vars[name].set(value)
            except Exception:
                pass
        self._run_clean_update_plot_for_screening()
        self.root.update_idletasks()
        sim_data = self._extract_simulation_from_app()
        self._save_simulation_cache(cache_path, sim_data, combo)
        sim_data["cache_key"] = key
        sim_data["cache_path"] = str(cache_path)
        sim_data["from_cache"] = False
        return sim_data
    def _compute_radial_chi_square_metrics(self, exp_img, sim_img, q_u=None, q_v=None, valid_mask=None, n_bins=80):
        """Compute 1D radial reduced chi-square after least-squares scaling."""
        _q_exp, I_exp, counts_exp, _qlabel, _std_exp, sem_exp = _radial_integrate_on_q_grid(
            exp_img, q_u=q_u, q_v=q_v, valid_mask=valid_mask, npt=n_bins
        )
        _q_sim, I_sim, _counts_sim, _qlabel2, _std_sim, _sem_sim = _radial_integrate_on_q_grid(
            sim_img, q_u=q_u, q_v=q_v, valid_mask=valid_mask, npt=n_bins
        )
        return _compute_radial_chi_square_metrics_from_integrated(I_exp, I_sim, sem_exp, counts_exp)
    def _compute_integration_curve_match_metrics(self, exp_img, sim_img, q_u=None, q_v=None, valid_mask=None, n_bins=500, q_lower_display_limit=DEFAULT_1D_SCALE_FIT_Q_MIN, q_upper_display_limit=DEFAULT_1D_SCALE_FIT_Q_MAX):
        """Similarity metrics for the displayed 1D integrated curves after scale fitting.
        The 1D integration reduced chi-square is computed on the same q-window used
        for the displayed 1D curve comparison.  It uses the experimental SEM when
        available, with a robust intensity-dynamic-range floor so that bins with
        nearly zero SEM do not dominate the accept/reject decision.
        """
        q_exp, I_exp, counts_exp, _qlabel, _std_exp, sem_exp = _radial_integrate_on_q_grid(
            exp_img, q_u=q_u, q_v=q_v, valid_mask=valid_mask, npt=n_bins
        )
        q_sim, I_sim, _counts_sim, _qlabel2, _std_sim, _sem_sim = _radial_integrate_on_q_grid(
            sim_img, q_u=q_u, q_v=q_v, valid_mask=valid_mask, npt=n_bins
        )
        q_exp = np.asarray(q_exp, dtype=float)
        exp = np.asarray(I_exp, dtype=float)
        q_sim = np.asarray(q_sim, dtype=float)
        sim = np.asarray(I_sim, dtype=float)
        sem = np.asarray(sem_exp, dtype=float)
        counts = np.asarray(counts_exp, dtype=float)
        if q_exp.shape != q_sim.shape or not np.allclose(q_exp, q_sim, equal_nan=True):
            sim_on_exp = np.full_like(exp, np.nan, dtype=float)
            good_src = np.isfinite(q_sim) & np.isfinite(sim)
            good_dst = np.isfinite(q_exp)
            if np.count_nonzero(good_src) >= 2 and np.count_nonzero(good_dst) >= 1:
                order = np.argsort(q_sim[good_src])
                sim_on_exp[good_dst] = np.interp(q_exp[good_dst], q_sim[good_src][order], sim[good_src][order], left=np.nan, right=np.nan)
            else:
                sim_on_exp = sim.copy()
        else:
            sim_on_exp = sim.copy()
        fit_mask = np.isfinite(q_exp) & np.isfinite(exp) & np.isfinite(sim_on_exp) & (counts > 0)
        if np.isfinite(q_lower_display_limit):
            fit_mask &= (q_exp >= q_lower_display_limit)
        if np.isfinite(q_upper_display_limit):
            fit_mask &= (q_exp <= q_upper_display_limit)
        if np.count_nonzero(fit_mask) < 5:
            fit_mask = np.isfinite(q_exp) & np.isfinite(exp) & np.isfinite(sim_on_exp) & (counts > 0)
        if np.count_nonzero(fit_mask) < 5:
            return {
                "integration_curve_corr": float("nan"),
                "integration_curve_nrmse": float("nan"),
                "integration_curve_scale_factor": float("nan"),
                "integration_chi2_reduced": float("nan"),
                "integration_chi2_n_points": int(np.count_nonzero(fit_mask)),
                "integration_chi2_dof": 0,
                "integration_chi2_scale_factor": float("nan"),
            }
        scale = _compute_overlay_scale(exp[fit_mask], sim_on_exp[fit_mask])
        if (not np.isfinite(scale)) or scale <= 0:
            scale = 1.0
        sim_scaled = sim_on_exp * scale
        cmp_mask = np.isfinite(q_exp) & np.isfinite(exp) & np.isfinite(sim_scaled) & (counts > 0)
        if np.isfinite(q_lower_display_limit):
            cmp_mask &= (q_exp >= q_lower_display_limit)
        if np.isfinite(q_upper_display_limit):
            cmp_mask &= (q_exp <= q_upper_display_limit)
        if np.count_nonzero(cmp_mask) < 5:
            cmp_mask = np.isfinite(q_exp) & np.isfinite(exp) & np.isfinite(sim_scaled) & (counts > 0)
        if np.count_nonzero(cmp_mask) < 5:
            return {
                "integration_curve_corr": float("nan"),
                "integration_curve_nrmse": float("nan"),
                "integration_curve_scale_factor": float(scale),
                "integration_chi2_reduced": float("nan"),
                "integration_chi2_n_points": int(np.count_nonzero(cmp_mask)),
                "integration_chi2_dof": 0,
                "integration_chi2_scale_factor": float(scale),
            }
        corr = _safe_corrcoef(exp, sim_scaled, cmp_mask)
        nrmse = _safe_nrmse(exp, sim_scaled, cmp_mask)
        sem_cmp = sem[cmp_mask]
        exp_cmp = exp[cmp_mask]
        sim_cmp = sim_scaled[cmp_mask]
        dynamic = float(np.nanpercentile(exp_cmp, 95) - np.nanpercentile(exp_cmp, 5)) if exp_cmp.size else float("nan")
        if (not np.isfinite(dynamic)) or dynamic <= 0:
            dynamic = float(np.nanstd(exp_cmp)) if exp_cmp.size else float("nan")
        if (not np.isfinite(dynamic)) or dynamic <= 0:
            dynamic = max(float(np.nanmax(np.abs(exp_cmp))) if exp_cmp.size else 1.0, 1.0)
        sigma_floor = max(1.0e-12, 0.02 * dynamic)
        sigma = np.where(np.isfinite(sem_cmp) & (sem_cmp > 0), sem_cmp, sigma_floor)
        sigma = np.sqrt(sigma ** 2 + sigma_floor ** 2)
        chi_good = np.isfinite(exp_cmp) & np.isfinite(sim_cmp) & np.isfinite(sigma) & (sigma > 0)
        n_chi = int(np.count_nonzero(chi_good))
        dof = max(n_chi - 1, 1)
        if n_chi > 0:
            chi2 = float(np.nansum(((exp_cmp[chi_good] - sim_cmp[chi_good]) / sigma[chi_good]) ** 2))
            reduced_chi2 = chi2 / dof
        else:
            reduced_chi2 = float("nan")
            dof = 0
        return {
            "integration_curve_corr": float(corr) if np.isfinite(corr) else float("nan"),
            "integration_curve_nrmse": float(nrmse) if np.isfinite(nrmse) else float("nan"),
            "integration_curve_scale_factor": float(scale),
            "integration_chi2_reduced": float(reduced_chi2) if np.isfinite(reduced_chi2) else float("nan"),
            "integration_chi2_n_points": int(n_chi),
            "integration_chi2_dof": int(dof),
            "integration_chi2_scale_factor": float(scale),
        }
    def _nan_moving_average_1d(self, y, window=9):
        arr = np.asarray(y, dtype=float)
        if window is None or int(window) <= 1 or arr.size < 3:
            return arr.copy()
        w = int(window)
        if w % 2 == 0:
            w += 1
        w = max(3, min(w, arr.size if arr.size % 2 == 1 else arr.size - 1))
        if w < 3:
            return arr.copy()
        finite = np.isfinite(arr)
        vals = np.where(finite, arr, 0.0)
        weights = finite.astype(float)
        kernel = np.ones(w, dtype=float)
        num = np.convolve(vals, kernel, mode="same")
        den = np.convolve(weights, kernel, mode="same")
        out = np.full_like(arr, np.nan, dtype=float)
        good = den > 0
        out[good] = num[good] / den[good]
        return out
    def _find_fringe_extrema_positions(self, q, intensity, q_lower_display_limit=DEFAULT_1D_SCALE_FIT_Q_MIN, q_upper_display_limit=DEFAULT_1D_SCALE_FIT_Q_MAX, exclude_first_n=10):
        """Find q positions of maxima/minima in a smoothed q^4-weighted 1D curve.

        The raw SAXS profile usually decays strongly with q, so local oscillatory
        fringes can be difficult to identify directly on I(q).  The profile is
        flattened as q^4 I(q), robustly normalized, smoothed with a moving
        average, and searched for alternating local extrema.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        I = np.asarray(intensity, dtype=float).reshape(-1)
        if q.size == 0 or I.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty
        if q.size != I.size:
            n = min(q.size, I.size)
            q = q[:n]
            I = I[:n]

        good = np.isfinite(q) & np.isfinite(I) & (q > 0)
        if np.isfinite(q_lower_display_limit):
            good &= (q >= float(q_lower_display_limit))
        if np.isfinite(q_upper_display_limit):
            good &= (q <= float(q_upper_display_limit))
        idx = np.where(good)[0]
        if idx.size <= max(12, int(exclude_first_n) + 4):
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty
        if exclude_first_n and idx.size > int(exclude_first_n):
            idx = idx[int(exclude_first_n):]

        qg = q[idx]
        yg = I[idx] * (qg ** 4)
        order = np.argsort(qg)
        qg = qg[order]
        yg = yg[order]
        finite = np.isfinite(yg)
        if np.count_nonzero(finite) < 8:
            empty = np.array([], dtype=float)
            return empty, empty, qg, np.full_like(qg, np.nan, dtype=float)

        y_fill = yg.astype(float, copy=True)
        if not np.all(finite):
            x = np.arange(y_fill.size, dtype=float)
            if np.count_nonzero(finite) >= 2:
                y_fill[~finite] = np.interp(x[~finite], x[finite], y_fill[finite])
            else:
                y_fill[~finite] = np.nanmedian(y_fill[finite])

        lo = np.nanpercentile(y_fill, 2)
        hi = np.nanpercentile(y_fill, 98)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = np.nanmin(y_fill)
            hi = np.nanmax(y_fill)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            y_norm = (y_fill - lo) / (hi - lo)
        else:
            empty = np.array([], dtype=float)
            return empty, empty, qg, np.full_like(qg, np.nan, dtype=float)

        window = 9
        if y_norm.size >= 700:
            window = 13
        if y_norm.size >= 1200:
            window = 17
        ys = self._nan_moving_average_1d(y_norm, window=window)
        finite_s = np.isfinite(ys)
        if np.count_nonzero(finite_s) < 8:
            empty = np.array([], dtype=float)
            return empty, empty, qg, ys
        if not np.all(finite_s):
            x = np.arange(ys.size, dtype=float)
            ys_filled = ys.copy()
            ys_filled[~finite_s] = np.interp(x[~finite_s], x[finite_s], ys[finite_s])
        else:
            ys_filled = ys

        def _score_extrema(indices, kind):
            indices = np.asarray(indices, dtype=int)
            if indices.size == 0:
                return indices
            half_width = max(4, int(round(ys_filled.size / 100)))
            keep = []
            for i in indices:
                if i <= 1 or i >= ys_filled.size - 2:
                    continue
                left = ys_filled[max(0, i - half_width):i]
                right = ys_filled[i + 1:min(ys_filled.size, i + 1 + half_width)]
                if left.size == 0 or right.size == 0:
                    continue
                if kind == "max":
                    baseline = max(np.nanmin(left), np.nanmin(right))
                    score = ys_filled[i] - baseline
                else:
                    baseline = min(np.nanmax(left), np.nanmax(right))
                    score = baseline - ys_filled[i]
                if np.isfinite(score) and score >= 0.003:
                    keep.append(int(i))
            return np.asarray(keep, dtype=int)

        try:
            from scipy.signal import find_peaks
            distance = max(3, int(round(ys_filled.size / 120)))
            # Low prominence is intentional because, in the combined pattern,
            # the ellipsoid fringe may survive only as a weak residual feature.
            peaks, _ = find_peaks(ys_filled, prominence=0.008, distance=distance, width=1)
            mins, _ = find_peaks(-ys_filled, prominence=0.008, distance=distance, width=1)
        except Exception:
            peaks = np.array([], dtype=int)
            mins = np.array([], dtype=int)

        # Derivative sign-change supplement catches shallow but smooth fringes.
        dy = np.diff(ys_filled)
        sign = np.sign(dy)
        for i in range(1, sign.size):
            if sign[i] == 0:
                sign[i] = sign[i - 1]
        cand_peaks = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
        cand_mins = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1
        peaks = np.unique(np.concatenate([np.asarray(peaks, dtype=int), cand_peaks.astype(int)]))
        mins = np.unique(np.concatenate([np.asarray(mins, dtype=int), cand_mins.astype(int)]))
        peaks = _score_extrema(peaks, "max")
        mins = _score_extrema(mins, "min")
        peaks = np.sort(peaks[(peaks >= 0) & (peaks < qg.size)])
        mins = np.sort(mins[(mins >= 0) & (mins < qg.size)])
        return qg[peaks], qg[mins], qg, ys

    def _estimate_sphere_radius_from_first_minimum(self, min_q_values, q_threshold=0.005):
        """Estimate droplet radius from the first visible minimum, assuming a sphere.

        Uses the first minimum of the homogeneous sphere form factor, qR ≈ 4.493.
        q is assumed to be in Å^-1, so the returned radius is reported in both Å and nm.
        The intended use here is the sweep preview for cases where the first droplet
        minimum appears at q >= 0.005 Å^-1, corresponding to R ≲ 90 nm.
        """
        qvals = np.asarray(min_q_values, dtype=float).reshape(-1)
        qvals = qvals[np.isfinite(qvals) & (qvals > 0)]
        if qvals.size == 0:
            return {
                "found": False,
                "first_min_q": np.nan,
                "radius_A": np.nan,
                "radius_nm": np.nan,
                "within_visible_first_min_window": False,
            }
        first_q = float(qvals[0])
        qr_first_min = 4.493409457909064
        radius_A = qr_first_min / first_q
        radius_nm = radius_A * 0.1
        return {
            "found": True,
            "first_min_q": first_q,
            "radius_A": float(radius_A),
            "radius_nm": float(radius_nm),
            "within_visible_first_min_window": bool(first_q >= float(q_threshold)),
        }


    def _compute_oscillation_fft_spectrum(
        self,
        q,
        intensity,
        *,
        q_min=0.005,
        q_max=0.03,
        n_grid=768,
        baseline_q_width=0.0035,
    ):
        """Return FFT spectrum of the oscillatory part of a 1D SAXS curve.

        The curve is first restricted to the visible q-window, interpolated onto
        a uniform q-grid, transformed to log10 intensity, and detrended by a
        broad smooth baseline.  The FFT is then applied to the residual using a
        Hann window.  The frequency axis is in cycles per q-unit; when q is in
        Å^-1, the frequency has units of Å.
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        y_arr = np.asarray(intensity, dtype=float).reshape(-1)
        if q_arr.size == 0 or y_arr.size == 0:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}
        if q_arr.size != y_arr.size:
            n = min(q_arr.size, y_arr.size)
            q_arr = q_arr[:n]
            y_arr = y_arr[:n]

        good = np.isfinite(q_arr) & np.isfinite(y_arr) & (q_arr > 0)
        good &= (q_arr >= float(q_min)) & (q_arr <= float(q_max))
        if np.count_nonzero(good) < 16:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}

        qg = q_arr[good]
        yg = y_arr[good]
        order = np.argsort(qg)
        qg = qg[order]
        yg = yg[order]

        # Collapse duplicate q values, if present.
        uq, inv = np.unique(qg, return_inverse=True)
        if uq.size < 16:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}
        if uq.size != qg.size:
            y_acc = np.zeros_like(uq, dtype=float)
            counts = np.zeros_like(uq, dtype=float)
            for ii, val in zip(inv, yg):
                if np.isfinite(val):
                    y_acc[ii] += float(val)
                    counts[ii] += 1.0
            ok = counts > 0
            uq = uq[ok]
            yg = y_acc[ok] / counts[ok]
        else:
            uq = qg

        n_grid = int(max(128, min(int(n_grid), 4096)))
        if uq.size < 16 or float(uq[-1]) <= float(uq[0]):
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}

        q_uniform = np.linspace(float(uq[0]), float(uq[-1]), n_grid)
        y_interp = np.interp(q_uniform, uq, yg)

        positive = np.isfinite(y_interp) & (y_interp > 0)
        if np.count_nonzero(positive) >= 8:
            floor = np.nanpercentile(y_interp[positive], 0.5)
            if not np.isfinite(floor) or floor <= 0:
                floor = np.nanmin(y_interp[positive])
            floor = max(float(floor), np.finfo(float).tiny)
            y_work = np.log10(np.maximum(y_interp, floor))
        else:
            y_work = y_interp.astype(float, copy=True)

        finite = np.isfinite(y_work)
        if np.count_nonzero(finite) < 16:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": q_uniform, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}
        if not np.all(finite):
            x = np.arange(y_work.size, dtype=float)
            y_work[~finite] = np.interp(x[~finite], x[finite], y_work[finite])

        dq = float(np.nanmedian(np.diff(q_uniform)))
        baseline_window = int(round(float(baseline_q_width) / dq)) if dq > 0 else 101
        baseline_window = max(31, min(baseline_window, max(31, n_grid // 2 - 1)))
        if baseline_window % 2 == 0:
            baseline_window += 1
        if baseline_window >= n_grid:
            baseline_window = n_grid - 1 if (n_grid - 1) % 2 == 1 else n_grid - 2
        baseline_window = max(7, baseline_window)

        try:
            from scipy.signal import savgol_filter
            polyorder = 3 if baseline_window >= 9 else 2
            baseline = savgol_filter(y_work, window_length=baseline_window, polyorder=polyorder, mode="interp")
        except Exception:
            pad = baseline_window // 2
            padded = np.pad(y_work, pad_width=pad, mode="edge")
            kernel = np.ones(baseline_window, dtype=float) / float(baseline_window)
            baseline = np.convolve(padded, kernel, mode="valid")

        residual = y_work - baseline
        residual = residual - np.nanmean(residual)
        if not np.all(np.isfinite(residual)) or np.nanstd(residual) <= 0:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": q_uniform, "residual": residual, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}

        window = np.hanning(residual.size)
        fft_vals = np.fft.rfft(residual * window)
        freq = np.fft.rfftfreq(residual.size, d=dq)
        amp = np.abs(fft_vals)
        if window.sum() > 0:
            amp = amp / window.sum()
        if amp.size:
            amp[0] = 0.0
        if np.nanmax(amp) > 0:
            amp = amp / np.nanmax(amp)

        # Dominant frequency in a conservative band. Very low frequencies mostly
        # represent imperfect baseline removal; very high frequencies are usually
        # small residual ripples/noise.
        freq_min = 80.0
        freq_max = min(float(np.nanmax(freq)) if freq.size else 0.0, 1600.0)
        band = np.isfinite(freq) & np.isfinite(amp) & (freq >= freq_min) & (freq <= freq_max)
        if np.count_nonzero(band) >= 2:
            band_idx = np.where(band)[0]
            best_idx = int(band_idx[np.nanargmax(amp[band])])
            dominant_frequency = float(freq[best_idx])
            # For sphere-like oscillations, the large-q fringe period is roughly Δq ≈ π/R.
            # Thus frequency = 1/Δq ≈ R/π, giving R[nm] ≈ 0.1*pi*frequency for q in Å^-1.
            dominant_radius_nm = float(0.1 * np.pi * dominant_frequency)
        else:
            dominant_frequency = np.nan
            dominant_radius_nm = np.nan

        return {
            "frequency": freq,
            "amplitude": amp,
            "q_uniform": q_uniform,
            "residual": residual,
            "dominant_frequency": dominant_frequency,
            "dominant_radius_nm": dominant_radius_nm,
        }

    def _plot_smoothed_extrema_markers(
        self,
        ax,
        q,
        intensity,
        *,
        q_min=None,
        q_max=None,
        color=None,
        label_prefix="curve",
        log_y=False,
        max_features=10,
        marker_size=38,
        legend=True,
        draw_guides=True,
    ):
        """Overlay extrema positions detected from the smoothed q^4-weighted curve."""
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        y_arr = np.asarray(intensity, dtype=float).reshape(-1)
        if q_arr.size == 0 or y_arr.size == 0:
            return {"max_q": np.array([], dtype=float), "min_q": np.array([], dtype=float), "sphere_radius": None}
        if q_arr.size != y_arr.size:
            n = min(q_arr.size, y_arr.size)
            q_arr = q_arr[:n]
            y_arr = y_arr[:n]
        q_lo = DEFAULT_1D_SCALE_FIT_Q_MIN if q_min is None else float(q_min)
        q_hi = DEFAULT_1D_SCALE_FIT_Q_MAX if q_max is None else float(q_max)
        max_q, min_q, _q_smooth, _y_smooth = self._find_fringe_extrema_positions(
            q_arr,
            y_arr,
            q_lower_display_limit=q_lo,
            q_upper_display_limit=q_hi,
            exclude_first_n=0,
        )
        max_q = np.asarray(max_q, dtype=float)
        min_q = np.asarray(min_q, dtype=float)
        if max_features is not None and int(max_features) > 0:
            max_q = max_q[: int(max_features)]
            min_q = min_q[: int(max_features)]

        finite = np.isfinite(q_arr) & np.isfinite(y_arr)
        if log_y:
            finite &= y_arr > 0
        if np.count_nonzero(finite) < 2:
            return {"max_q": max_q, "min_q": min_q, "sphere_radius": None}
        order = np.argsort(q_arr[finite])
        q_good = q_arr[finite][order]
        y_good = y_arr[finite][order]

        def _interp_y(q_points):
            q_points = np.asarray(q_points, dtype=float)
            if q_points.size == 0:
                return np.array([], dtype=float)
            inside = np.isfinite(q_points) & (q_points >= q_good[0]) & (q_points <= q_good[-1])
            out = np.full(q_points.shape, np.nan, dtype=float)
            if np.any(inside):
                out[inside] = np.interp(q_points[inside], q_good, y_good)
            if log_y:
                out[out <= 0] = np.nan
            return out

        max_y = _interp_y(max_q)
        min_y = _interp_y(min_q)
        max_good = np.isfinite(max_q) & np.isfinite(max_y)
        min_good = np.isfinite(min_q) & np.isfinite(min_y)
        if bool(draw_guides):
            for qv in max_q[max_good]:
                ax.axvline(qv, color=color, linestyle=":", linewidth=0.65, alpha=0.28, zorder=1)
            for qv in min_q[min_good]:
                ax.axvline(qv, color=color, linestyle="--", linewidth=0.55, alpha=0.20, zorder=1)
        if np.any(max_good):
            ax.scatter(
                max_q[max_good],
                max_y[max_good],
                marker="^",
                s=float(marker_size),
                facecolors="none",
                edgecolors=color,
                linewidths=1.15,
                label=(f"{label_prefix} maxima" if legend else None),
                zorder=6,
                clip_on=False,
            )
        if np.any(min_good):
            ax.scatter(
                min_q[min_good],
                min_y[min_good],
                marker="v",
                s=float(marker_size),
                facecolors="none",
                edgecolors=color,
                linewidths=1.15,
                label=(f"{label_prefix} minima" if legend else None),
                zorder=6,
            )
        return {"max_q": max_q[max_good], "min_q": min_q[min_good]}
    def _mean_matched_q_error(self, exp_q, sim_q, q_span):
        """Return mean one-to-one q-position mismatch normalized by q_span.

        The fringe scorer compares local maxima and minima detected from the
        experimental and simulated 1D curves.  Earlier builds called this helper
        from ``_compute_fringe_peak_match_metrics`` but did not define it, which
        made the shared Single-shot output path fail for every simulator mode.

        Matching is order-preserving and one-to-one: all extrema from the smaller
        set are matched to an ordered subset of extrema from the larger set with
        the minimum total absolute q displacement.  This prevents one simulated
        peak from being reused for several experimental peaks while still giving
        a meaningful large error when the fringe spacing is shifted.
        """
        exp = np.asarray(exp_q, dtype=float).reshape(-1)
        sim = np.asarray(sim_q, dtype=float).reshape(-1)
        exp = np.sort(exp[np.isfinite(exp)])
        sim = np.sort(sim[np.isfinite(sim)])
        n_exp = int(exp.size)
        n_sim = int(sim.size)
        if n_exp == 0 or n_sim == 0:
            return float("nan"), 0

        try:
            span = float(q_span)
        except Exception:
            span = float("nan")
        if not np.isfinite(span) or span <= 0:
            combined = np.concatenate([exp, sim])
            if combined.size >= 2:
                span = float(np.nanmax(combined) - np.nanmin(combined))
            if not np.isfinite(span) or span <= 0:
                span = 1.0

        # Match every point in the smaller array to an ordered subset of the
        # larger array.  A small dynamic-programming table is used rather than a
        # nearest-neighbour loop so that the total displacement is minimized.
        swapped = False
        a = exp
        b = sim
        if a.size > b.size:
            a, b = b, a
            swapped = True
        n = int(a.size)
        m = int(b.size)

        dp = np.full((n + 1, m + 1), np.inf, dtype=float)
        take = np.zeros((n + 1, m + 1), dtype=bool)
        dp[0, :] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                skip_cost = dp[i, j - 1]
                take_cost = dp[i - 1, j - 1] + abs(float(a[i - 1] - b[j - 1]))
                if take_cost <= skip_cost:
                    dp[i, j] = take_cost
                    take[i, j] = True
                else:
                    dp[i, j] = skip_cost

        if not np.isfinite(dp[n, m]):
            return float("nan"), 0

        distances = []
        i, j = n, m
        while i > 0 and j > 0:
            if take[i, j]:
                distances.append(abs(float(a[i - 1] - b[j - 1])))
                i -= 1
                j -= 1
            else:
                j -= 1
        if len(distances) != n:
            # Defensive fallback; this should not be reached, but keeps the
            # scorer from raising during GUI export if an unexpected numerical
            # edge case appears.
            distances = []
            used = np.zeros(m, dtype=bool)
            for value in a:
                available = np.where(~used)[0]
                if available.size == 0:
                    break
                idx = available[int(np.argmin(np.abs(b[available] - value)))]
                used[idx] = True
                distances.append(abs(float(value - b[idx])))

        mean_error = float(np.nanmean(distances) / span) if distances else float("nan")
        if not np.isfinite(mean_error):
            mean_error = float("nan")
        return mean_error, int(len(distances))

    def _compute_fringe_peak_match_metrics(self, exp_img, sim_img, q_u=None, q_v=None, valid_mask=None, n_bins=500, q_lower_display_limit=DEFAULT_1D_SCALE_FIT_Q_MIN, q_upper_display_limit=DEFAULT_1D_SCALE_FIT_Q_MAX):
        """Compare fringe peak/minimum positions in q^4-weighted 1D curves.
        This is not a full structural least-squares fit.  It is a screening gate
        designed to reject candidates whose overall 2D envelope/orientation looks
        acceptable but whose fringe spacing is shifted.
        """
        q_exp, I_exp, _counts_exp, _qlabel, _std_exp, _sem_exp = _radial_integrate_on_q_grid(
            exp_img, q_u=q_u, q_v=q_v, valid_mask=valid_mask, npt=n_bins
        )
        q_sim, I_sim, _counts_sim, _qlabel2, _std_sim, _sem_sim = _radial_integrate_on_q_grid(
            sim_img, q_u=q_u, q_v=q_v, valid_mask=valid_mask, npt=n_bins
        )
        q_exp = np.asarray(q_exp, dtype=float)
        q_sim = np.asarray(q_sim, dtype=float)
        I_exp = np.asarray(I_exp, dtype=float)
        I_sim = np.asarray(I_sim, dtype=float)
        # Use the same positive 1D scale fitting used for the visual overlay.
        good = np.isfinite(q_exp) & np.isfinite(I_exp)
        if q_exp.shape == q_sim.shape and np.allclose(q_exp, q_sim, equal_nan=True):
            I_sim_on_exp = I_sim.copy()
        else:
            I_sim_on_exp = np.full_like(I_exp, np.nan, dtype=float)
            good_src = np.isfinite(q_sim) & np.isfinite(I_sim)
            good_dst = np.isfinite(q_exp)
            if np.count_nonzero(good_src) >= 2:
                order = np.argsort(q_sim[good_src])
                I_sim_on_exp[good_dst] = np.interp(q_exp[good_dst], q_sim[good_src][order], I_sim[good_src][order], left=np.nan, right=np.nan)
        fit_good = good & np.isfinite(I_sim_on_exp)
        if np.isfinite(q_lower_display_limit):
            fit_good &= (q_exp >= q_lower_display_limit)
        if np.isfinite(q_upper_display_limit):
            fit_good &= (q_exp <= q_upper_display_limit)
        scale = _compute_overlay_scale(I_exp[fit_good], I_sim_on_exp[fit_good]) if np.count_nonzero(fit_good) >= 3 else 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        I_sim_scaled = I_sim_on_exp * scale
        exp_peak_q, exp_min_q, q_curve, y_exp = self._find_fringe_extrema_positions(
            q_exp, I_exp, q_lower_display_limit=q_lower_display_limit, q_upper_display_limit=q_upper_display_limit
        )
        sim_peak_q, sim_min_q, _q_curve_sim, y_sim = self._find_fringe_extrema_positions(
            q_exp, I_sim_scaled, q_lower_display_limit=q_lower_display_limit, q_upper_display_limit=q_upper_display_limit
        )
        q_fit = q_exp[np.isfinite(q_exp) & (q_exp > 0)]
        if np.isfinite(q_lower_display_limit):
            q_fit = q_fit[q_fit >= q_lower_display_limit]
        if np.isfinite(q_upper_display_limit):
            q_fit = q_fit[q_fit <= q_upper_display_limit]
        q_span = float(np.nanmax(q_fit) - np.nanmin(q_fit)) if q_fit.size else float("nan")
        peak_err, n_peak = self._mean_matched_q_error(exp_peak_q, sim_peak_q, q_span)
        min_err, n_min = self._mean_matched_q_error(exp_min_q, sim_min_q, q_span)
        errs = [v for v in (peak_err, min_err) if np.isfinite(v)]
        if errs:
            pos_err = float(np.nanmean(errs))
            # Penalize feature-count mismatch mildly.  This follows the logic of
            # rejecting curves that do not show the same characteristic maxima/minima.
            exp_count = int(exp_peak_q.size + exp_min_q.size)
            sim_count = int(sim_peak_q.size + sim_min_q.size)
            denom = max(exp_count, sim_count, 1)
            pos_err += 0.03 * abs(exp_count - sim_count) / denom
        else:
            pos_err = float("nan")
            exp_count = int(exp_peak_q.size + exp_min_q.size)
            sim_count = int(sim_peak_q.size + sim_min_q.size)
        return {
            "fringe_position_rel_error": float(pos_err) if np.isfinite(pos_err) else float("nan"),
            "fringe_peak_rel_error": float(peak_err) if np.isfinite(peak_err) else float("nan"),
            "fringe_min_rel_error": float(min_err) if np.isfinite(min_err) else float("nan"),
            "fringe_feature_count_exp": int(exp_count),
            "fringe_feature_count_sim": int(sim_count),
            "fringe_feature_count_matched": int(n_peak + n_min),
            "fringe_peak_count_exp": int(exp_peak_q.size),
            "fringe_peak_count_sim": int(sim_peak_q.size),
            "fringe_min_count_exp": int(exp_min_q.size),
            "fringe_min_count_sim": int(sim_min_q.size),
            "fringe_peak_q_exp": ";".join(f"{x:.8g}" for x in np.asarray(exp_peak_q, dtype=float)[:8]),
            "fringe_peak_q_sim": ";".join(f"{x:.8g}" for x in np.asarray(sim_peak_q, dtype=float)[:8]),
            "fringe_min_q_exp": ";".join(f"{x:.8g}" for x in np.asarray(exp_min_q, dtype=float)[:8]),
            "fringe_min_q_sim": ";".join(f"{x:.8g}" for x in np.asarray(sim_min_q, dtype=float)[:8]),
        }
    def _score_cost_from_metric(self, metrics, term):
        mode = str(term.get("mode", "error"))
        metric = str(term.get("metric", ""))
        value = _screening_safe_float(metrics.get(metric), np.nan)
        if mode == "corr":
            value = _screening_safe_float(value, 0.0)
            return 1.0 - max(-1.0, min(1.0, value))
        if mode == "log_chi2":
            value = _screening_safe_float(value, 1.0e6)
            return min(2.0, np.log10(max(value, 1.0)) / 2.0)
        if mode == "scaled_error":
            value = _screening_safe_float(value, np.inf)
            scale = float(term.get("scale", 1.0) or 1.0)
            cap = float(term.get("cap", 1.5) or 1.5)
            if not np.isfinite(value):
                return cap
            return min(cap, max(0.0, value) / max(scale, 1.0e-12))
        # default error/NRMSE-like term
        value = _screening_safe_float(value, 1.0)
        if not np.isfinite(value):
            value = 1.0
        return max(0.0, value)
    def _compute_total_score(self, metrics, thresholds=None):
        terms = None
        if isinstance(thresholds, dict):
            terms = thresholds.get("score_terms", None)
        if not terms:
            terms = self._score_terms_from_controls()
        active = []
        total_weight = 0.0
        weighted_sum = 0.0
        pieces = []
        for term in terms:
            if not bool(term.get("enabled", True)):
                continue
            weight = _screening_safe_float(term.get("weight", 0.0), 0.0)
            if not np.isfinite(weight) or weight <= 0:
                continue
            cost = self._score_cost_from_metric(metrics, term)
            if not np.isfinite(cost):
                cost = 1.0
            weighted_sum += weight * cost
            total_weight += weight
            key = str(term.get("key", term.get("metric", "term")))
            active.append(key)
            pieces.append(f"{key}:{weight:.4g}×{cost:.4g}")
        if total_weight <= 0:
            metrics["total_score_formula"] = "nan; no active score terms"
            metrics["total_score_active_terms"] = ""
            metrics["total_score_weight_sum"] = 0.0
            metrics["score_contribution_summary"] = ""
            return float("nan")
        score = float(weighted_sum / total_weight)
        metrics["total_score_formula"] = "(" + " + ".join(pieces) + f") / {total_weight:.4g}"
        metrics["total_score_active_terms"] = ";".join(active)
        metrics["total_score_weight_sum"] = float(total_weight)
        metrics["score_contribution_summary"] = "; ".join(pieces)
        return score
    def _is_accepted(self, metrics, thresholds, total_score):
        radial_corr = _screening_safe_float(metrics.get("radial_profile_corr"), -np.inf)
        radial_nrmse = _screening_safe_float(metrics.get("radial_profile_nrmse"), np.inf)
        radial_chi2 = _screening_safe_float(metrics.get("radial_chi2_reduced"), np.inf)
        integration_corr = _screening_safe_float(metrics.get("integration_curve_corr"), -np.inf)
        integration_nrmse = _screening_safe_float(metrics.get("integration_curve_nrmse"), np.inf)
        integration_chi2 = _screening_safe_float(metrics.get("integration_chi2_reduced"), np.inf)
        fringe_err = _screening_safe_float(metrics.get("fringe_position_rel_error"), np.inf)
        fringe_threshold = _screening_safe_float(thresholds.get("fringe_position_error_max"), np.inf)
        fringe_ok = True if (not np.isfinite(fringe_threshold) or fringe_threshold >= 1.0) else (fringe_err <= fringe_threshold)
        pixel_corr = _screening_safe_float(metrics.get("pixel_pearson_corr_asinh_z"), -np.inf)
        harmonic_corr = _screening_safe_float(metrics.get("harmonics_mean_corr_A2_A6", metrics.get("harmonics_mean_corr_A1_A12")), -np.inf)
        harmonic_phase_err = _screening_safe_float(
            metrics.get("harmonic_phase_error_weighted_A2_A6_deg",
                        metrics.get("harmonic_phase_error_mean_A2_A6_deg",
                                    metrics.get("A2_phase_error_deg"))),
            np.inf,
        )
        a2_phase_threshold = _screening_safe_float(thresholds.get("a2_phase_error_max_deg"), np.inf)
        a2_phase_ok = True if (not np.isfinite(a2_phase_threshold) or a2_phase_threshold >= 90.0) else (harmonic_phase_err <= a2_phase_threshold)
        angular_corr = _screening_safe_float(metrics.get("angular_profile_corr_zero_shift"), -np.inf)
        angular_corr_threshold = _screening_safe_float(thresholds.get("angular_corr_min"), -np.inf)
        angular_corr_ok = True if (not np.isfinite(angular_corr_threshold) or angular_corr_threshold <= -1.0) else (angular_corr >= angular_corr_threshold)
        def gate_ok(key, condition):
            return True if not self._threshold_gate_is_enabled(thresholds, key, True) else bool(condition)

        total_score_ok = total_score <= thresholds["total_score_max"]
        return (
            gate_ok("radial_corr", radial_corr >= thresholds["radial_corr_min"]) and
            gate_ok("radial_nrmse", radial_nrmse <= thresholds["radial_nrmse_max"]) and
            gate_ok("radial_chi2", radial_chi2 <= thresholds["radial_chi2_max"]) and
            gate_ok("int_corr", integration_corr >= thresholds["integration_curve_corr_min"]) and
            gate_ok("int_nrmse", integration_nrmse <= thresholds["integration_curve_nrmse_max"]) and
            gate_ok("int_chi2", integration_chi2 <= thresholds["integration_chi2_max"]) and
            gate_ok("fringe", fringe_ok) and
            gate_ok("pixel_corr", pixel_corr >= thresholds["pixel_corr_min"]) and
            gate_ok("harmonic_corr", harmonic_corr >= thresholds["harmonic_corr_min"]) and
            gate_ok("harmonic_phase", a2_phase_ok) and
            gate_ok("angular_corr", angular_corr_ok) and
            gate_ok("total_score", total_score_ok)
        )
    def _screening_poisson_source(self):
        """Return the object that owns the Poisson-noise controls for screening.
        AutoScreenDialog has its own Poisson-noise controls so scan runs can be
        configured independently of the main interactive display.  Single-shot
        helper objects also receive copied controls.  The main GUI remains a
        delegate for the poissonization implementation when needed.
        """
        for obj in (self, getattr(self, "noise_source", None), getattr(self, "parent_controller", None), self.app):
            if obj is not None and hasattr(obj, "poisson_noise_var"):
                return obj
        return None
    def _parse_poisson_peak_counts(self):
        try:
            value = float(self.poisson_peak_counts_var.get().strip())
        except Exception:
            value = 10000.0
        if not np.isfinite(value) or value <= 0:
            value = 10000.0
            try:
                self.poisson_peak_counts_var.set("10000")
            except Exception:
                pass
        return value
    def _parse_poisson_seed(self):
        try:
            text = str(self.poisson_seed_var.get()).strip()
        except Exception:
            text = ""
        if text == "":
            return None
        try:
            return int(text)
        except Exception:
            # Allow a non-integer string to act as a reproducible seed.
            return abs(hash(text)) % (2 ** 32)
    def _parse_poisson_count_multiplier(self):
        try:
            value = float(self.poisson_count_multiplier_var.get().strip())
        except Exception:
            value = 1.0
        if not np.isfinite(value) or value <= 0:
            value = 1.0
            try:
                self.poisson_count_multiplier_var.set("1.0")
            except Exception:
                pass
        return value
    def _parse_poisson_reference_percentile(self):
        try:
            value = float(self.poisson_reference_percentile_var.get().strip())
        except Exception:
            value = 99.9
        if not np.isfinite(value):
            value = 99.9
        value = min(100.0, max(0.0, value))
        try:
            self.poisson_reference_percentile_var.set(f"{value:g}")
        except Exception:
            pass
        return value
    def _poisson_reference_state(self):
        """Return effective counts and optional fixed intensity reference."""
        peak_counts = self._parse_poisson_peak_counts()
        multiplier = self._parse_poisson_count_multiplier() if hasattr(self, "poisson_count_multiplier_var") else 1.0
        use_dataset_ref = False
        try:
            use_dataset_ref = bool(self.poisson_use_dataset_reference_var.get())
        except Exception:
            use_dataset_ref = False
        ref = getattr(self, "poisson_reference_intensity", None)
        try:
            ref = float(ref)
        except Exception:
            ref = None
        if use_dataset_ref and ref is not None and np.isfinite(ref) and ref > 0:
            return float(ref * multiplier), float(ref), "dataset_max", float(multiplier)
        return float(peak_counts), None, "simulation_max", float(multiplier)
    def _calibrate_poisson_reference_from_default_npz(self, show_errors=True):
        """Use the fixed bundled Poisson reference; do not reload cimg001.npz."""
        ref = float(DEFAULT_POISSON_REFERENCE_COUNT)
        meta = {
            "mode": "fixed_cimg001_img262_p99.9",
            "npz_name": "cimg001.npz",
            "image_number": int(DEFAULT_POISSON_REFERENCE_IMAGE_NUMBER),
            "local_number": int(DEFAULT_POISSON_REFERENCE_IMAGE_NUMBER),
            "percentile": float(DEFAULT_POISSON_REFERENCE_PERCENTILE),
            "reference_intensity": ref,
        }
        self.poisson_reference_intensity = ref
        self.poisson_reference_metadata = dict(meta)
        try:
            self.poisson_use_dataset_reference_var.set(True)
            self.poisson_peak_counts_var.set(f"{ref:.6g}")
            self.poisson_reference_status_var.set(
                f"fixed cimg001 img {DEFAULT_POISSON_REFERENCE_IMAGE_NUMBER}, p{DEFAULT_POISSON_REFERENCE_PERCENTILE:g}={ref:.6g}"
            )
        except Exception:
            pass
        try:
            self._update_poisson_noise_only()
        except Exception:
            pass
        return self.poisson_reference_metadata
    def _calibrate_poisson_reference_from_dataset(self):
        delegate_source = getattr(self, "noise_source", None)
        delegate = getattr(delegate_source, "_calibrate_poisson_reference_from_dataset", None) if delegate_source is not None else None
        if callable(delegate) and delegate is not self._calibrate_poisson_reference_from_dataset:
            result = delegate()
            self.poisson_reference_intensity = getattr(delegate_source, "poisson_reference_intensity", None)
            self.poisson_reference_metadata = dict(getattr(delegate_source, "poisson_reference_metadata", {}) or {})
            try:
                self.poisson_reference_status_var.set(getattr(delegate_source, "poisson_reference_status_var").get())
            except Exception:
                pass
            return result
        return self._calibrate_poisson_reference_from_default_npz(show_errors=True)
    def _poissonize_image(self, image, peak_counts, rng, reference_intensity=None):
        # Prefer the main GUI implementation so scan/single-shot noise behaves
        # identically to the interactive display.  Keep a local fallback so this
        # dialog remains functional if opened from a different controller.
        for obj in (getattr(self, "noise_source", None), getattr(self, "parent_controller", None), self.app):
            delegate = getattr(obj, "_poissonize_image", None) if obj is not None else None
            if callable(delegate) and delegate is not self._poissonize_image:
                return delegate(image, peak_counts, rng, reference_intensity=reference_intensity)
        arr = np.asarray(image, dtype=float)
        out = arr.copy()
        finite = np.isfinite(arr)
        if not np.any(finite):
            return out
        nonnegative = np.zeros_like(arr, dtype=float)
        nonnegative[finite] = np.maximum(arr[finite], 0.0)
        max_val = float(np.nanmax(nonnegative[finite])) if np.any(finite) else 0.0
        if not np.isfinite(max_val) or max_val <= 0:
            return out
        try:
            ref = float(reference_intensity) if reference_intensity is not None else None
        except Exception:
            ref = None
        if ref is None or not np.isfinite(ref) or ref <= 0:
            ref = max_val
        alpha = float(peak_counts) / float(ref)
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = float(peak_counts) / max_val
        lam = nonnegative * alpha
        lam[~np.isfinite(lam)] = 0.0
        lam = np.maximum(lam, 0.0)
        counts = rng.poisson(lam)
        noisy = counts.astype(float) / alpha
        out[finite] = noisy[finite]
        return out
    def _screening_poisson_state(self):
        source = self._screening_poisson_source()
        if source is None:
            return False, None, None, None, "simulation_max", 1.0
        try:
            enabled = bool(source.poisson_noise_var.get())
        except Exception:
            enabled = False
        if not enabled:
            return False, None, None, None, "simulation_max", 1.0
        try:
            if hasattr(source, "_poisson_reference_state"):
                peak_counts, reference_intensity, reference_mode, multiplier = source._poisson_reference_state()
            elif hasattr(source, "_parse_poisson_peak_counts"):
                peak_counts = float(source._parse_poisson_peak_counts())
                reference_intensity = None
                reference_mode = "simulation_max"
                multiplier = 1.0
            else:
                peak_counts = float(getattr(source, "poisson_peak_counts_var").get())
                reference_intensity = None
                reference_mode = "simulation_max"
                multiplier = 1.0
        except Exception:
            peak_counts = 10000.0
            reference_intensity = None
            reference_mode = "simulation_max"
            multiplier = 1.0
        if not np.isfinite(peak_counts) or peak_counts <= 0:
            peak_counts = 10000.0
            reference_intensity = None
            reference_mode = "simulation_max"
        try:
            if hasattr(source, "_parse_poisson_seed"):
                seed = source._parse_poisson_seed()
            else:
                text = str(getattr(source, "poisson_seed_var").get()).strip()
                seed = None if text == "" else int(text)
        except Exception:
            seed = None
        return True, float(peak_counts), seed, reference_intensity, reference_mode, float(multiplier)
    def _get_screening_simulation_image(self, sim_data):
        """Return the simulation image actually used for screening metrics.
        ``sim_data["image"]`` is intentionally clean/noiseless so the simulation
        cache remains reusable.  Poisson noise is injected only at the processing
        stage when the GUI checkbox is enabled.
        For single-shot output, ``sim_data`` may instead carry the exact image that
        is currently displayed in the live simulator window. In that case we must
        use that array as-is, otherwise an empty/random Poisson seed would produce
        a different noisy realization from the one shown to the user.
        When possible, the noise is applied to the clean total-theoretical image
        first and the detector mask is applied afterwards, matching the live GUI
        behaviour.  This keeps auto-screen and single-shot outputs visually and
        numerically consistent with the simulator window.
        """
        if bool(sim_data.get("use_image_as_is", False)):
            return (
                np.asarray(sim_data["image"], dtype=float).copy(),
                bool(sim_data.get("screening_noise_enabled", False)),
                sim_data.get("screening_peak_counts", None),
                sim_data.get("screening_seed", None),
            )
        clean_masked = np.asarray(sim_data["image"], dtype=float)
        enabled, peak_counts, seed, reference_intensity, reference_mode, multiplier = self._screening_poisson_state()
        if not enabled:
            return clean_masked.copy(), False, None, None
        source = self._screening_poisson_source()
        poissonize = getattr(source, "_poissonize_image", None) if source is not None else None
        if not callable(poissonize):
            poissonize = getattr(self.app, "_poissonize_image", None)
        if not callable(poissonize):
            return clean_masked.copy(), False, None, None
        rng = np.random.default_rng(seed)
        clean_total = sim_data.get("image_total_clean", None)
        mask = sim_data.get("mask", None)
        # Convert arbitrary simulation units to the fixed experimental reference
        # count scale.  Screening dialogs own their own Poisson checkbox/multiplier,
        # but the simulation-side reference must still come from the main
        # controller's global Ellipse-1 calibration.  Therefore search across the
        # dialog, its noise_source, parent controller, and app before falling back.
        sim_reference_intensity = None
        ref_owner_candidates = (
            source,
            getattr(self, "noise_source", None),
            getattr(self, "parent_controller", None),
            getattr(self, "app", None),
        )
        if clean_total is not None:
            for owner in ref_owner_candidates:
                if owner is None:
                    continue
                get_ref = getattr(owner, "_get_or_set_simulation_poisson_reference", None)
                if callable(get_ref):
                    try:
                        sim_reference_intensity = get_ref(getattr(self, "app", None), np.asarray(clean_total, dtype=float))
                    except Exception:
                        sim_reference_intensity = None
                    if sim_reference_intensity is not None:
                        break
        if sim_reference_intensity is None:
            for owner in ref_owner_candidates:
                if owner is None:
                    continue
                estimate = getattr(owner, "_estimate_array_reference_intensity", None)
                if callable(estimate):
                    try:
                        sim_reference_intensity = estimate(clean_total if clean_total is not None else clean_masked, DEFAULT_POISSON_REFERENCE_PERCENTILE)
                    except Exception:
                        sim_reference_intensity = None
                    if sim_reference_intensity is not None:
                        break
        if sim_reference_intensity is None:
            sim_reference_intensity = reference_intensity
        if clean_total is not None:
            try:
                noisy_total = poissonize(np.asarray(clean_total, dtype=float), peak_counts, rng, reference_intensity=sim_reference_intensity)
                if mask is not None:
                    mask_arr = np.asarray(mask, dtype=float)
                    if mask_arr.shape == noisy_total.shape:
                        noisy_masked = np.asarray(noisy_total, dtype=float) * mask_arr
                        return np.asarray(noisy_masked, dtype=float), True, float(peak_counts), seed
                return np.asarray(noisy_total, dtype=float), True, float(peak_counts), seed
            except Exception:
                pass
        noisy = poissonize(clean_masked, peak_counts, rng, reference_intensity=sim_reference_intensity)
        return np.asarray(noisy, dtype=float), True, float(peak_counts), seed
    def _add_derived_2d_score_metrics(self, metrics):
        phase_vals = []
        phase_weights = []
        for n in range(2, 7):
            err = _screening_safe_float(metrics.get(f"A{n}_phase_error_deg"), np.nan)
            amp = _screening_safe_float(metrics.get(f"A{n}_anisotropy_corr"), np.nan)
            # Use profile correlation as a soft reliability proxy when explicit amplitudes are not available in metrics.
            weight = max(0.0, 1.0 + max(-1.0, min(1.0, amp))) if np.isfinite(amp) else 1.0
            if np.isfinite(err):
                phase_vals.append(err)
                phase_weights.append(weight)
        if phase_vals:
            arr = np.asarray(phase_vals, dtype=float)
            w = np.asarray(phase_weights, dtype=float)
            metrics["harmonic_phase_error_mean_A2_A6_deg"] = float(np.nanmean(arr))
            metrics["harmonic_phase_error_weighted_A2_A6_deg"] = float(np.nansum(arr * w) / np.nansum(w)) if np.nansum(w) > 0 else float(np.nanmean(arr))
            metrics["harmonic_phase_score_n_terms"] = int(arr.size)
        else:
            metrics["harmonic_phase_error_mean_A2_A6_deg"] = float("nan")
            metrics["harmonic_phase_error_weighted_A2_A6_deg"] = float("nan")
            metrics["harmonic_phase_score_n_terms"] = 0
        # New clearer aliases.  The legacy A1_A12 field is kept for backward compatibility,
        # but in the current workflow harmonic_orders=(2,3,4,5,6), so these represent A2–A6.
        if "harmonics_mean_corr_A2_A6" not in metrics:
            metrics["harmonics_mean_corr_A2_A6"] = metrics.get("harmonics_mean_corr_A1_A12", float("nan"))
        if "harmonics_mean_nrmse_A2_A6" not in metrics:
            metrics["harmonics_mean_nrmse_A2_A6"] = metrics.get("harmonics_mean_nrmse_A1_A12", float("nan"))
        return metrics

    def _comparison_row(self, combo_index, total_combos, combo, exp, sim_data, thresholds):
        exp_img = exp["image"]
        sim_img, screening_noise_enabled, screening_peak_counts, screening_seed = self._get_screening_simulation_image(sim_data)
        sim_img = np.asarray(sim_img, dtype=float)
        if exp_img.shape != sim_img.shape:
            raise ValueError(f"Shape mismatch: experimental {exp_img.shape}, simulation {sim_img.shape}")
        valid_mask = exp["valid_mask"]
        if valid_mask is None or np.asarray(valid_mask).shape != exp_img.shape:
            valid_mask = _load_default_saxs_valid_mask(exp_img.shape)
        # Use the same q-grid preference as the interactive "2D features" window:
        # prefer experimental/.poni geometry first, then fall back to any simulator
        # q-grid bundled with the simulation data.
        q_u = exp.get("q_u", None)
        q_v = exp.get("q_v", None)
        if q_u is None or q_v is None:
            q_u = sim_data.get("q_u", None)
            q_v = sim_data.get("q_v", None)
        comparison = _compare_2d_saxs_features(
            exp_img,
            sim_img,
            q_u=q_u,
            q_v=q_v,
            valid_mask=valid_mask,
            n_bins=thresholds["n_bins"],
            harmonic_orders=tuple(range(2, 7)),
        )
        metrics = comparison["metrics"]
        # Keep feature extraction n_bins user-controlled, but use the same
        # 1D integration density as the interactive "SAXS integrate" window
        # for oscillation-sensitive chi-square/radial comparison.
        integration_n_bins = 500
        chi_metrics = self._compute_radial_chi_square_metrics(
            exp_img,
            sim_img,
            q_u=q_u,
            q_v=q_v,
            valid_mask=valid_mask,
            n_bins=integration_n_bins,
        )
        integration_curve_metrics = self._compute_integration_curve_match_metrics(
            exp_img,
            sim_img,
            q_u=q_u,
            q_v=q_v,
            valid_mask=valid_mask,
            n_bins=integration_n_bins,
            q_lower_display_limit=DEFAULT_1D_SCALE_FIT_Q_MIN,
            q_upper_display_limit=DEFAULT_1D_SCALE_FIT_Q_MAX,
        )
        fringe_metrics = self._compute_fringe_peak_match_metrics(
            exp_img,
            sim_img,
            q_u=q_u,
            q_v=q_v,
            valid_mask=valid_mask,
            n_bins=integration_n_bins,
            q_lower_display_limit=DEFAULT_1D_SCALE_FIT_Q_MIN,
            q_upper_display_limit=DEFAULT_1D_SCALE_FIT_Q_MAX,
        )
        metrics.update(chi_metrics)
        metrics.update(integration_curve_metrics)
        metrics.update(fringe_metrics)
        self._add_derived_2d_score_metrics(metrics)
        gate_enabled = thresholds.get("gate_enabled", {}) if isinstance(thresholds, dict) else {}
        def _gate_on(key, default=True):
            try:
                return bool(gate_enabled.get(key, default))
            except Exception:
                return bool(default)
        integration_chi2_limit = _screening_safe_float(thresholds.get("integration_chi2_max"), np.inf)
        integration_chi2_value = _screening_safe_float(metrics.get("integration_chi2_reduced"), np.inf)
        metrics["integration_chi2_threshold_enabled"] = bool(_gate_on("int_chi2"))
        metrics["integration_chi2_threshold_max"] = float(integration_chi2_limit) if np.isfinite(integration_chi2_limit) else float("inf")
        metrics["integration_chi2_threshold_pass"] = bool(True if not _gate_on("int_chi2") else (integration_chi2_value <= integration_chi2_limit))
        a2_phase_limit = _screening_safe_float(thresholds.get("a2_phase_error_max_deg"), np.inf)
        a2_phase_value = _screening_safe_float(metrics.get("A2_phase_error_deg"), np.inf)
        harmonic_phase_value = _screening_safe_float(
            metrics.get("harmonic_phase_error_weighted_A2_A6_deg",
                        metrics.get("harmonic_phase_error_mean_A2_A6_deg",
                                    metrics.get("A2_phase_error_deg"))),
            np.inf,
        )
        metrics["A2_phase_threshold_enabled"] = bool(_gate_on("harmonic_phase"))
        metrics["A2_phase_threshold_max_deg"] = float(a2_phase_limit) if np.isfinite(a2_phase_limit) else float("inf")
        metrics["A2_phase_threshold_pass"] = bool(True if (not _gate_on("harmonic_phase") or not np.isfinite(a2_phase_limit) or a2_phase_limit >= 90.0) else (a2_phase_value <= a2_phase_limit))
        metrics["harmonic_phase_threshold_enabled"] = bool(_gate_on("harmonic_phase"))
        metrics["harmonic_phase_threshold_max_deg"] = float(a2_phase_limit) if np.isfinite(a2_phase_limit) else float("inf")
        metrics["harmonic_phase_threshold_pass"] = bool(True if (not _gate_on("harmonic_phase") or not np.isfinite(a2_phase_limit) or a2_phase_limit >= 90.0) else (harmonic_phase_value <= a2_phase_limit))
        angular_corr_limit = _screening_safe_float(thresholds.get("angular_corr_min"), -np.inf)
        angular_corr_value = _screening_safe_float(metrics.get("angular_profile_corr_zero_shift"), -np.inf)
        metrics["angular_corr_threshold_enabled"] = bool(_gate_on("angular_corr"))
        metrics["angular_corr_threshold_min"] = float(angular_corr_limit) if np.isfinite(angular_corr_limit) else float("-inf")
        metrics["angular_corr_threshold_pass"] = bool(True if (not _gate_on("angular_corr") or not np.isfinite(angular_corr_limit) or angular_corr_limit <= -1.0) else (angular_corr_value >= angular_corr_limit))
        metrics["n_bins"] = int(thresholds["n_bins"])
        metrics["integration_n_bins"] = int(integration_n_bins)
        comparison["exp_img_raw"] = np.asarray(exp_img, dtype=float)
        comparison["sim_img_raw"] = np.asarray(sim_img, dtype=float)
        comparison["q_u"] = q_u
        comparison["q_v"] = q_v
        comparison["valid_mask_input"] = np.asarray(valid_mask, dtype=bool)
        comparison["screening_noise_enabled"] = bool(screening_noise_enabled)
        comparison["screening_peak_counts"] = screening_peak_counts
        comparison["screening_seed"] = screening_seed
        total_score = self._compute_total_score(metrics, thresholds)
        accepted = self._is_accepted(metrics, thresholds, total_score)
        row = {
            "combo_index": combo_index,
            "total_combos": total_combos,
            "experimental_id": exp["id"],
            "run_name": exp["run_name"],
            "run_number": exp["run_number"],
            "image_number": exp["image_number"],
            "npz_name": exp["npz_name"],
            "local_number": exp["local_number"],
            "simulator": self.simulator_name,
            "cache_key": sim_data.get("cache_key", ""),
            "cache_path": sim_data.get("cache_path", ""),
            "from_cache": bool(sim_data.get("from_cache", False)),
            "screening_poisson_noise_enabled": bool(screening_noise_enabled),
            "screening_poisson_peak_counts": screening_peak_counts if screening_peak_counts is not None else np.nan,
            "screening_poisson_seed": screening_seed if screening_seed is not None else "",
            "screening_poisson_reference_mode": getattr(self._screening_poisson_source(), "poisson_reference_metadata", {}).get("mode", "dataset_max" if screening_noise_enabled and getattr(self._screening_poisson_source(), "poisson_reference_intensity", None) is not None else "simulation_max"),
            "screening_poisson_reference_intensity": getattr(self._screening_poisson_source(), "poisson_reference_intensity", np.nan),
            "screening_poisson_count_multiplier": getattr(self._screening_poisson_source(), "_parse_poisson_count_multiplier", lambda: np.nan)(),
            "accepted": bool(accepted),
            "total_score": total_score,
            "summary_figure_path": "",
            "harmonic_heatmap_path": "",
            "experimental_masked_figure_path": "",
            "theoretical_masked_figure_path": "",
        }
        for name, value in combo.items():
            row[f"param_{name}"] = value
        for field in self.METRIC_FIELDS:
            if field in ("accepted", "total_score"):
                continue
            row[field] = metrics.get(field, np.nan)
        return row, comparison
    def _comparison_figure_slug(self, row):
        combo_index = int(row.get("combo_index", 0))
        exp_slug = _safe_filename_fragment(row.get("experimental_id", "exp"))
        cache_key = _safe_filename_fragment(row.get("cache_key", "cache"))
        return f"combo{combo_index:05d}_{exp_slug}_{cache_key}"
    def _save_comparison_figures(self, row, comparison, output_dir, bucket="accepted"):
        figures_dir = Path(output_dir) / "figures" / _safe_filename_fragment(bucket)
        figures_dir.mkdir(parents=True, exist_ok=True)
        slug = self._comparison_figure_slug(row)
        summary_path = figures_dir / f"{slug}_summary.png"
        metrics = dict(comparison.get("metrics", {}))
        exp_img_raw = np.asarray(comparison.get("exp_img_raw", np.zeros((4, 4), dtype=float)), dtype=float)
        sim_img_raw = np.asarray(comparison.get("sim_img_raw", np.zeros_like(exp_img_raw)), dtype=float)
        q_u = comparison.get("q_u", None)
        q_v = comparison.get("q_v", None)
        valid_mask_plot = comparison.get("valid_mask_input", None)
        if valid_mask_plot is not None:
            valid_mask_plot = np.asarray(valid_mask_plot, dtype=bool)
            if valid_mask_plot.shape != exp_img_raw.shape:
                valid_mask_plot = None
        # Recompute feature/display products directly from the raw images that were
        # actually used for screening. This keeps the saved summary fully aligned
        # with the noisy theoretical image when Poisson noise is enabled.
        harmonic_orders = tuple(range(1, 13))
        feature_n_bins = int(metrics.get("n_bins", 500)) if np.isfinite(_screening_safe_float(metrics.get("n_bins", np.nan), np.nan)) else 500
        try:
            figure_comp = _compare_2d_saxs_features(
                exp_img_raw,
                sim_img_raw,
                q_u=q_u,
                q_v=q_v,
                valid_mask=valid_mask_plot,
                n_bins=feature_n_bins,
                harmonic_orders=harmonic_orders,
            )
        except Exception:
            figure_comp = comparison
        # Keep the original screening metrics (which already include 1D/fringe
        # calculations) but ensure any display-specific values are consistent with
        # the figure recomputation above.
        metrics.update(figure_comp.get("metrics", {}))
        for _key in ("radial_chi2_reduced", "radial_chi2_n_points", "radial_chi2_scale_factor", "radial_weighted_nrmse"):
            _row_val = row.get(_key, None)
            if _row_val is not None:
                try:
                    _row_float = float(_row_val)
                    if np.isfinite(_row_float):
                        metrics[_key] = _row_float
                except Exception:
                    pass
        exp_profiles = figure_comp.get("exp_profiles", comparison.get("exp_profiles", {}))
        sim_profiles = figure_comp.get("sim_profiles", comparison.get("sim_profiles", {}))
        common_mask = np.asarray(
            figure_comp.get(
                "common_mask",
                comparison.get("common_mask", np.ones_like(exp_img_raw, dtype=bool)),
            ),
            dtype=bool,
        )
        if common_mask.shape != exp_img_raw.shape:
            common_mask = np.ones_like(exp_img_raw, dtype=bool)
        scale_raw = _screening_safe_float(metrics.get("simulation_scale_factor_raw_pixel_LS", 1.0), 1.0)
        if not np.isfinite(scale_raw):
            scale_raw = 1.0
        # Match the simulator window more closely for the image panels:
        #   * display the raw masked simulation (not LS-scaled for display),
        #   * use the q-axis extent when available,
        #   * use the same vertical orientation convention as the GUI modules
        #     (flipud + origin="lower"),
        #   * do not percentile-compress the color scale.
        exp_display = np.asarray(exp_img_raw, dtype=float)
        sim_display = float(scale_raw) * np.asarray(sim_img_raw, dtype=float)
        residual_z = np.asarray(exp_img_raw, dtype=float) - sim_display
        def _image_extent_from_axes(x_axis, y_axis):
            try:
                x = np.asarray(x_axis, dtype=np.float64)
                y = np.asarray(y_axis, dtype=np.float64)
                if x.ndim == 1 and y.ndim == 1 and x.size > 1 and y.size > 1:
                    return (float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y)))
            except Exception:
                pass
            return None
        q_extent = _image_extent_from_axes(q_u, q_v)
        def _panel_display_array(data, apply_common_mask=False):
            arr = np.asarray(data, dtype=float).copy()
            if apply_common_mask and arr.shape == common_mask.shape:
                arr[~common_mask] = np.nan
            return np.flipud(arr)
        def _imshow_like_gui(ax, data, title, cmap=None, apply_common_mask=False):
            disp = _panel_display_array(data, apply_common_mask=apply_common_mask)
            kwargs = {
                "origin": "lower",
                "interpolation": "nearest",
                "aspect": "equal",
                "cmap": (cmap or "viridis"),
            }
            if q_extent is not None:
                kwargs["extent"] = q_extent
            if cmap == "coolwarm":
                finite = np.isfinite(disp)
                if np.any(finite):
                    vmax = float(np.nanmax(np.abs(disp[finite])))
                    if np.isfinite(vmax) and vmax > 0:
                        kwargs["vmin"] = -vmax
                        kwargs["vmax"] = vmax
            im = ax.imshow(disp, **kwargs)
            ax.set_title(title)
            if q_extent is not None:
                ax.set_xlabel("q_u")
                ax.set_ylabel("q_v")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            return im
        def _imshow_feature_with_mask(ax, data, title, cmap=None):
            return _imshow_signed_feature_panel(ax, data, common_mask, title, cmap=cmap)
        # Match the interactive "SAXS integrate" window:
        #   1) prefer bundled PONI q axes,
        #   2) use the default detector-valid mask rather than the stricter
        #      feature-comparison common_mask,
        #   3) use 500 radial bins,
        #   4) fit the simulation scale over the displayed low-q window.
        try:
            q_u_real, q_v_real, _q_meta = _load_default_q_axes_from_poni(exp_img_raw.shape)
            if q_u_real is not None and q_v_real is not None:
                q_u_int_use, q_v_int_use = q_u_real, q_v_real
            else:
                q_u_int_use, q_v_int_use = q_u, q_v
        except Exception:
            q_u_int_use, q_v_int_use = q_u, q_v
        try:
            integration_mask = _load_default_saxs_valid_mask(exp_img_raw.shape)
        except Exception:
            integration_mask = common_mask
        integration_n_bins = int(metrics.get("integration_n_bins", 500)) if np.isfinite(_screening_safe_float(metrics.get("integration_n_bins", np.nan), np.nan)) else 500
        q_exp_int, I_exp_int, counts_exp, q_label, std_exp, sem_exp = _radial_integrate_on_q_grid(
            exp_img_raw, q_u=q_u_int_use, q_v=q_v_int_use, valid_mask=integration_mask, npt=integration_n_bins
        )
        q_sim_int, I_sim_int, counts_sim, _q_label2, _std_sim, _sem_sim = _radial_integrate_on_q_grid(
            sim_img_raw, q_u=q_u_int_use, q_v=q_v_int_use, valid_mask=integration_mask, npt=integration_n_bins
        )
        finite_plot = np.isfinite(q_exp_int) & (np.isfinite(I_exp_int) | np.isfinite(I_sim_int))
        if np.any(finite_plot):
            q_min_data = float(np.nanmin(q_exp_int[finite_plot]))
            q_max_data = float(np.nanmax(q_exp_int[finite_plot]))
        else:
            q_min_data = float(np.nanmin(q_exp_int))
            q_max_data = float(np.nanmax(q_exp_int))
        q_lower_display_limit = DEFAULT_1D_SCALE_FIT_Q_MIN
        q_upper_display_limit = DEFAULT_1D_SCALE_FIT_Q_MAX
        q_min_plot = max(q_min_data, q_lower_display_limit) if np.isfinite(q_lower_display_limit) else q_min_data
        q_max_plot = min(q_max_data, q_upper_display_limit) if q_min_plot < q_upper_display_limit else q_max_data
        if not np.isfinite(q_max_plot) or q_max_plot <= q_min_plot:
            q_max_plot = q_max_data
        def _scan_plot_mask(q_values, y_values=None):
            q_arr = np.asarray(q_values, dtype=float)
            good = np.isfinite(q_arr) & (q_arr >= q_min_plot) & (q_arr <= q_max_plot)
            if y_values is not None:
                good &= np.isfinite(np.asarray(y_values, dtype=float))
            if np.count_nonzero(good) < 3:
                good = np.isfinite(q_arr)
                if y_values is not None:
                    good &= np.isfinite(np.asarray(y_values, dtype=float))
            return good
        def _scan_fit_mask(y_a=None, y_b=None):
            good = np.isfinite(q_exp_int) & (q_exp_int >= q_min_plot) & (q_exp_int <= q_max_plot)
            if y_a is not None:
                good &= np.isfinite(np.asarray(y_a, dtype=float))
            if y_b is not None:
                good &= np.isfinite(np.asarray(y_b, dtype=float))
            if np.count_nonzero(good) < 3:
                good = np.isfinite(q_exp_int)
                if y_a is not None:
                    good &= np.isfinite(np.asarray(y_a, dtype=float))
                if y_b is not None:
                    good &= np.isfinite(np.asarray(y_b, dtype=float))
            return good
        fit_good = _scan_fit_mask(I_exp_int, I_sim_int)
        scale_1d = _compute_overlay_scale(np.asarray(I_exp_int)[fit_good], np.asarray(I_sim_int)[fit_good])
        I_sim_scaled = scale_1d * np.asarray(I_sim_int, dtype=float)
        def _interp_single_shot_sim_to_exp_grid(q_src, y_src):
            q_src = np.asarray(q_src, dtype=float)
            y_src = np.asarray(y_src, dtype=float)
            if q_src.shape == q_exp_int.shape and np.allclose(q_src, q_exp_int, equal_nan=True, rtol=1e-10, atol=1e-12):
                return y_src.astype(float, copy=False)
            out = np.full_like(np.asarray(q_exp_int, dtype=float), np.nan, dtype=float)
            good = np.isfinite(q_src) & np.isfinite(y_src)
            if np.count_nonzero(good) >= 2:
                out = np.interp(np.asarray(q_exp_int, dtype=float), q_src[good], y_src[good], left=np.nan, right=np.nan)
            return out
        I_sim_scaled_on_exp = _interp_single_shot_sim_to_exp_grid(q_sim_int, I_sim_scaled)
        good_int = np.isfinite(I_exp_int) & np.isfinite(I_sim_scaled_on_exp)
        q_int = np.asarray(q_exp_int, dtype=float)
        res_int = np.full_like(np.asarray(I_exp_int, dtype=float), np.nan)
        res_int[good_int] = np.asarray(I_exp_int, dtype=float)[good_int] - I_sim_scaled_on_exp[good_int]
        # Use the same robust uncertainty floor as the screening-time
        # integration_chi2_reduced calculation.  The previous figure-only
        # calculation used an extremely small SEM floor, so accepted rows could
        # display a much larger 1D chi-square than the value actually used by
        # the threshold gate.
        sigma_exp_int = np.asarray(sem_exp, dtype=float).copy()
        chi1d = np.full_like(np.asarray(I_exp_int, dtype=float), np.nan)
        chi_range_good = _scan_fit_mask(I_exp_int, I_sim_scaled_on_exp)
        exp_chi_cmp = np.asarray(I_exp_int, dtype=float)[chi_range_good]
        sim_chi_cmp = np.asarray(I_sim_scaled_on_exp, dtype=float)[chi_range_good]
        sigma_chi_cmp = sigma_exp_int[chi_range_good]
        dynamic = float(np.nanpercentile(exp_chi_cmp, 95) - np.nanpercentile(exp_chi_cmp, 5)) if exp_chi_cmp.size else float("nan")
        if (not np.isfinite(dynamic)) or dynamic <= 0:
            dynamic = float(np.nanstd(exp_chi_cmp)) if exp_chi_cmp.size else float("nan")
        if (not np.isfinite(dynamic)) or dynamic <= 0:
            dynamic = max(float(np.nanmax(np.abs(exp_chi_cmp))) if exp_chi_cmp.size else 1.0, 1.0)
        sigma_floor = max(1.0e-12, 0.02 * dynamic)
        sigma_chi_cmp = np.where(np.isfinite(sigma_chi_cmp) & (sigma_chi_cmp > 0), sigma_chi_cmp, sigma_floor)
        sigma_chi_cmp = np.sqrt(sigma_chi_cmp ** 2 + sigma_floor ** 2)
        chi_good_local = (
            np.isfinite(exp_chi_cmp)
            & np.isfinite(sim_chi_cmp)
            & np.isfinite(sigma_chi_cmp)
            & (sigma_chi_cmp > 0)
        )
        n_chi_1d = int(np.count_nonzero(chi_good_local))
        dof_chi_1d = max(n_chi_1d - 1, 1)
        if n_chi_1d > 0:
            chi_vals = ((exp_chi_cmp[chi_good_local] - sim_chi_cmp[chi_good_local]) / sigma_chi_cmp[chi_good_local]) ** 2
            reduced_chi2_1d = float(np.nansum(chi_vals) / dof_chi_1d)
            full_idx = np.where(chi_range_good)[0]
            chi1d[full_idx[chi_good_local]] = chi_vals
        else:
            reduced_chi2_1d = float("nan")
            dof_chi_1d = 0
        # Keep the saved figure text synchronized with the screening row.
        # If a row value exists, it is the authoritative threshold-gate value.
        row_chi2 = _screening_safe_float(row.get("integration_chi2_reduced"), np.nan)
        if np.isfinite(row_chi2):
            reduced_chi2_1d = row_chi2
        metrics["integration_chi2_reduced"] = reduced_chi2_1d
        metrics["integration_chi2_n_points"] = int(row.get("integration_chi2_n_points", n_chi_1d)) if str(row.get("integration_chi2_n_points", "")).strip() != "" else n_chi_1d
        metrics["integration_chi2_dof"] = int(row.get("integration_chi2_dof", dof_chi_1d)) if str(row.get("integration_chi2_dof", "")).strip() != "" else dof_chi_1d
        metrics["integration_chi2_scale_factor"] = float(row.get("integration_chi2_scale_factor", scale_1d)) if np.isfinite(_screening_safe_float(row.get("integration_chi2_scale_factor", np.nan), np.nan)) else float(scale_1d)
        fig = Figure(figsize=(14.2, 18.0), dpi=120)
        gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.0, 0.86, 0.86, 0.86], hspace=0.44, wspace=0.30)
        ax_exp = fig.add_subplot(gs[0, 0])
        ax_sim = fig.add_subplot(gs[0, 1])
        ax_res = fig.add_subplot(gs[0, 2])
        ax_feat_exp = fig.add_subplot(gs[1, 0])
        ax_feat_sim = fig.add_subplot(gs[1, 1])
        ax_feat_res = fig.add_subplot(gs[1, 2])
        ax_rad = fig.add_subplot(gs[2, 0])
        ax_int = fig.add_subplot(gs[2, 1])
        ax_int_res = fig.add_subplot(gs[2, 2])
        ax_a2 = fig.add_subplot(gs[3, 0])
        ax_a3 = fig.add_subplot(gs[3, 1])
        ax_a4 = fig.add_subplot(gs[3, 2])
        ax_a5 = fig.add_subplot(gs[4, 0])
        ax_a6 = fig.add_subplot(gs[4, 1])
        ax_hblank = fig.add_subplot(gs[4, 2])
        im0 = _imshow_like_gui(ax_exp, exp_display, "experimental masked SAXS")
        sim_title = f"theoretical masked SAXS × {scale_raw:.3g}"
        im1 = _imshow_like_gui(ax_sim, sim_display, sim_title)
        im2 = _imshow_like_gui(ax_res, residual_z, "residual (display scale)", cmap="coolwarm", apply_common_mask=True)
        fig.colorbar(im0, ax=ax_exp, fraction=0.046, pad=0.03)
        fig.colorbar(im1, ax=ax_sim, fraction=0.046, pad=0.03)
        fig.colorbar(im2, ax=ax_res, fraction=0.046, pad=0.03)
        feat_exp_display = figure_comp.get("exp_display", comparison.get("exp_display", comparison.get("exp_z", exp_img_raw)))
        feat_sim_display = figure_comp.get("sim_display", comparison.get("sim_display", comparison.get("sim_z", sim_img_raw)))
        feat_residual = figure_comp.get("residual_z", comparison.get("residual_z", residual_z))
        imf0 = _imshow_feature_with_mask(ax_feat_exp, feat_exp_display, "Exp")
        imf1 = _imshow_feature_with_mask(ax_feat_sim, feat_sim_display, f"Sim × {scale_raw:.3g}")
        imf2 = _imshow_feature_with_mask(ax_feat_res, feat_residual, "Residual", cmap="coolwarm")
        fig.colorbar(imf0, ax=ax_feat_exp, fraction=0.046, pad=0.03)
        fig.colorbar(imf1, ax=ax_feat_sim, fraction=0.046, pad=0.03)
        fig.colorbar(imf2, ax=ax_feat_res, fraction=0.046, pad=0.03)
        q_vals = np.asarray(exp_profiles.get("q_raw", exp_profiles.get("q", np.array([]))), dtype=float)
        ax_rad.plot(q_vals, np.asarray(exp_profiles.get("radial_raw", exp_profiles.get("radial", np.array([]))), dtype=float), label="experimental")
        ax_rad.plot(np.asarray(sim_profiles.get("q_raw", sim_profiles.get("q", np.array([]))), dtype=float), np.asarray(sim_profiles.get("radial_raw", sim_profiles.get("radial", np.array([]))), dtype=float), label="simulation")
        ax_rad.set_title(f"2D radial | corr={_format_feature_metric(metrics.get('radial_profile_corr'))}, χ²red={_format_feature_metric(metrics.get('radial_chi2_reduced'))}")
        ax_rad.set_xlabel("q (Å⁻¹)")
        ax_rad.set_ylabel("raw radial intensity")
        ax_rad.grid(True, alpha=0.25)
        ax_rad.legend(loc="best", fontsize=8)
        m_exp_int = _scan_plot_mask(q_int, I_exp_int)
        m_sim_int = _scan_plot_mask(q_sim_int, I_sim_scaled)
        m_res_int = _scan_plot_mask(q_int, res_int)
        ax_int.plot(q_int[m_exp_int], np.asarray(I_exp_int)[m_exp_int], label="experimental 1D")
        ax_int.plot(np.asarray(q_sim_int)[m_sim_int], np.asarray(I_sim_scaled)[m_sim_int], '--', label=f"scaled theoretical 1D × {scale_1d:.3g}")
        ax_int.set_title(
            f"1D integration | χ²red={_format_feature_metric(reduced_chi2_1d)}, "
            f"corr={_format_feature_metric(metrics.get('integration_curve_corr'))}"
        )
        ax_int.set_xlabel(q_label)
        ax_int.set_ylabel("mean intensity")
        ax_int.set_xlim(q_min_plot, q_max_plot)
        ax_int.grid(True, alpha=0.25)
        ax_int.legend(loc="best", fontsize=8)
        ax_int_res.plot(q_int[m_res_int], res_int[m_res_int], label="I_exp - scaled I_theory")
        zero = np.zeros_like(q_int)
        ax_int_res.plot(q_int[m_res_int], zero[m_res_int], ':', linewidth=1)
        if np.any(np.isfinite(sem_exp)):
            sem = np.asarray(sem_exp, dtype=float)
            ax_int_res.fill_between(q_int[m_res_int], -sem[m_res_int], sem[m_res_int], alpha=0.25, label="± experimental SEM")
        ax_int_res.set_title("1D residual: exp - scaled theory")
        ax_int_res.set_xlabel(q_label)
        ax_int_res.set_ylabel("I_exp - scaled I_theory")
        ax_int_res.set_xlim(q_min_plot, q_max_plot)
        ax_int_res.grid(True, alpha=0.25)
        ax_int_res.legend(loc="best", fontsize=8)
        _plot_selected_harmonic_profile(ax_a2, exp_profiles, sim_profiles, metrics, 2, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a3, exp_profiles, sim_profiles, metrics, 3, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a4, exp_profiles, sim_profiles, metrics, 4, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a5, exp_profiles, sim_profiles, metrics, 5, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a6, exp_profiles, sim_profiles, metrics, 6, show_phase_error=True, fontsize=8)
        _plot_polar_match_summary(ax_hblank, figure_comp.get("polar_data", comparison.get("polar_data", {})), metrics, fontsize=8)
        param_items = [(k[6:], v) for k, v in row.items() if isinstance(k, str) and k.startswith('param_')]
        param_str = ', '.join(f"{k}={_format_param_value(v)}" for k, v in param_items[:8])
        fig.suptitle(
            f"{row.get('experimental_id','')} | accepted={row.get('accepted', False)} | total_score={_format_feature_metric(row.get('total_score', np.nan))} | radial χ²red={_format_feature_metric(metrics.get('radial_chi2_reduced'))} | 1D χ²red={_format_feature_metric(metrics.get('integration_chi2_reduced'))} | A2–A6 phase err={_format_feature_metric(metrics.get('harmonic_phase_error_weighted_A2_A6_deg'))}° | angular corr={_format_feature_metric(metrics.get('angular_profile_corr_zero_shift'))}\n{param_str}",
            fontsize=11,
            y=0.99,
        )
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass
        return str(summary_path), "", "", ""
    def _format_autoscreen_score_calculation(self, row, thresholds=None):
        """Return a readable symbolic explanation of the Auto Screen total score.

        Symbols A, B, C... are term identifiers.  The actual mismatch value used
        in the weighted average is written as M_A, M_B, ...
        """
        thresholds = thresholds if isinstance(thresholds, dict) else {}
        terms = thresholds.get("score_terms", None) or self._score_terms_from_controls()
        total_score = _screening_safe_float(row.get("total_score", np.nan), np.nan)
        total_score_max = _screening_safe_float(
            thresholds.get("total_score_max", row.get("total_score_threshold_max", np.nan)),
            np.nan,
        )

        def _fmt(value):
            return _format_feature_metric(value)

        def _sym(term):
            try:
                return self._score_formula_symbol(term)
            except Exception:
                return str(term.get("key", term.get("metric", "?")))

        def _rule_text(term):
            mode = str(term.get("mode", "error"))
            if mode == "corr":
                return "M = 1 - raw"
            if mode == "log_chi2":
                return "M = min(2, log10(max(raw,1))/2)"
            if mode == "scaled_error":
                scale = _fmt(term.get("scale", 1.0))
                cap = _fmt(term.get("cap", 1.5))
                return f"M = min({cap}, raw/{scale})"
            return "M = raw"

        def _short_rule(term):
            mode = str(term.get("mode", "error"))
            if mode == "corr":
                return "1-raw"
            if mode == "log_chi2":
                return "logchi"
            if mode == "scaled_error":
                return "scaled"
            return "raw"

        active_rows = []
        weighted_sum = 0.0
        weight_sum = 0.0
        key_to_sym = {}
        formula_terms = []
        for term in terms:
            sym = _sym(term)
            key_to_sym[str(term.get("key", ""))] = sym
            if not bool(term.get("enabled", True)):
                continue
            weight = _screening_safe_float(term.get("weight", 0.0), 0.0)
            if not np.isfinite(weight) or weight <= 0:
                continue
            metric = str(term.get("metric", ""))
            raw_value = row.get(metric, np.nan)
            mismatch = self._score_cost_from_metric(row, term)
            if not np.isfinite(_screening_safe_float(mismatch, np.nan)):
                mismatch = 1.0
            weighted = float(weight) * float(mismatch)
            weighted_sum += weighted
            weight_sum += float(weight)
            formula_terms.append(f"{_fmt(weight)}*M{sym}")
            active_rows.append({
                "sym": sym,
                "label": str(term.get("label", term.get("key", metric))),
                "metric": metric,
                "raw": raw_value,
                "rule": _rule_text(term),
                "short_rule": _short_rule(term),
                "m": float(mismatch),
                "w": float(weight),
                "wm": float(weighted),
            })

        for item in active_rows:
            item["dS"] = item["wm"] / weight_sum if weight_sum > 0 else float("nan")
        ranked = sorted(
            active_rows,
            key=lambda x: np.nan_to_num(x.get("dS", 0.0), nan=0.0),
            reverse=True,
        )

        lines = []
        lines.append("TOTAL SCORE")
        lines.append("S = sum(W*M)/sum(W)")
        lines.append("  W : term weight")
        lines.append("  M : converted mismatch used for scoring; smaller M is better")
        lines.append("  dS: contribution to S = W*M/sum(W)")
        lines.append("")
        if weight_sum > 0:
            recomputed = weighted_sum / weight_sum
            if formula_terms:
                formula_line = " + ".join(formula_terms)
                # Keep the formula readable in a fixed-width Text box.
                if len(formula_line) > 100:
                    formula_line = formula_line[:97] + "..."
                lines.append(f"S = ({formula_line}) / {_fmt(weight_sum)}")
            lines.append(f"S = {_fmt(weighted_sum)} / {_fmt(weight_sum)} = {_fmt(recomputed)}")
            gate = "n/a"
            if not self._threshold_gate_is_enabled(thresholds, "total_score", True):
                gate = "off"
            elif np.isfinite(total_score) and np.isfinite(total_score_max):
                gate = "PASS" if total_score <= total_score_max else "FAIL"
            if np.isfinite(total_score_max):
                lines.append(f"Total gate: S <= {_fmt(total_score_max)}  ->  {_fmt(total_score)}  [{gate}]")
        else:
            lines.append("S = nan  (no active term)")
        lines.append(f"Accepted: {row.get('accepted', False)}")
        lines.append("")

        lines.append("CONVERSION RULES")
        lines.append("corr term : raw close to 1 is good, so M = 1 - raw")
        lines.append("error/NRMSE: raw close to 0 is good, so M = raw")
        lines.append("chi2 term : compressed with log scale to avoid one huge value dominating")
        lines.append("scaled err: error divided by its cutoff-like scale, then capped")
        lines.append("")

        if ranked:
            lines.append("MAIN CONTRIBUTORS  (larger dS means larger effect on total score)")
            lines.append("T  term                         raw        M        W       dS    rule")
            lines.append("-- -------------------------- ---------- -------- -------- -------- --------")
            for item in ranked:
                label = item["label"]
                if len(label) > 26:
                    label = label[:23] + "..."
                lines.append(
                    f"{item['sym']:<2} {label:<26} {_fmt(item['raw']):>10} "
                    f"{_fmt(item['m']):>8} {_fmt(item['w']):>8} {_fmt(item['dS']):>8} {item['short_rule']:<8}"
                )
            lines.append("")

        lines.append("TERM MAP")
        for item in active_rows:
            lines.append(f"{item['sym']}: {item['label']}  |  {item['rule']}")
        lines.append("")

        lines.append("HARD GATES  (must pass in addition to total score, when cutoff is on)")
        gate_items = [
            ("radial_corr", key_to_sym.get("radial_corr", "A"), "2D radial corr", row.get("radial_profile_corr"), ">=", thresholds.get("radial_corr_min", np.nan)),
            ("radial_nrmse", key_to_sym.get("radial_nrmse", "B"), "2D radial NRMSE", row.get("radial_profile_nrmse"), "<=", thresholds.get("radial_nrmse_max", np.nan)),
            ("radial_chi2", key_to_sym.get("radial_chi2", "C"), "2D radial chi2", row.get("radial_chi2_reduced"), "<=", thresholds.get("radial_chi2_max", np.nan)),
            ("int_corr", key_to_sym.get("int_corr", "D"), "1D corr", row.get("integration_curve_corr"), ">=", thresholds.get("integration_curve_corr_min", np.nan)),
            ("int_nrmse", key_to_sym.get("int_nrmse", "E"), "1D NRMSE", row.get("integration_curve_nrmse"), "<=", thresholds.get("integration_curve_nrmse_max", np.nan)),
            ("int_chi2", key_to_sym.get("int_chi2", "F"), "1D chi2", row.get("integration_chi2_reduced"), "<=", thresholds.get("integration_chi2_max", np.nan)),
            ("fringe", key_to_sym.get("fringe_pos", "G"), "fringe pos", row.get("fringe_position_rel_error"), "<=", thresholds.get("fringe_position_error_max", np.nan)),
            ("pixel_corr", key_to_sym.get("pixel_corr", "H"), "2D pixel corr", row.get("pixel_pearson_corr_asinh_z"), ">=", thresholds.get("pixel_corr_min", np.nan)),
            ("harmonic_corr", key_to_sym.get("harmonic_corr", "J"), "harmonic corr", row.get("harmonics_mean_corr_A2_A6"), ">=", thresholds.get("harmonic_corr_min", np.nan)),
            ("harmonic_phase", key_to_sym.get("harmonic_phase", "L"), "A2–A6 phase", row.get("harmonic_phase_error_weighted_A2_A6_deg"), "<=", thresholds.get("a2_phase_error_max_deg", np.nan)),
            ("angular_corr", key_to_sym.get("angular_zero", "M"), "angular corr", row.get("angular_profile_corr_zero_shift"), ">=", thresholds.get("angular_corr_min", np.nan)),
        ]
        lines.append("T  gate               value      cutoff      result")
        lines.append("-- ---------------- ---------- ---------- ------")
        for gate_key, sym, label, value, op, threshold in gate_items:
            value_f = _screening_safe_float(value, np.nan)
            thresh_f = _screening_safe_float(threshold, np.nan)
            if not self._threshold_gate_is_enabled(thresholds, gate_key, True):
                result = "off"
            elif not np.isfinite(thresh_f):
                result = "off"
            elif op == ">=":
                result = "PASS" if value_f >= thresh_f else "FAIL"
            else:
                result = "PASS" if value_f <= thresh_f else "FAIL"
            cut_text = f"{op}{_fmt(threshold)}" if (np.isfinite(thresh_f) and result != "off") else "off"
            if len(label) > 16:
                label = label[:13] + "..."
            lines.append(f"{sym:<2} {label:<16} {_fmt(value):>10} {cut_text:>10} {result:>6}")
        return "\n".join(lines)


    def _show_autoscreen_preview_image(self, image_path, title="Auto Screen preview", info_text="", score_text=""):
        """Show a generated one-image Auto Screen summary in a fit/zoom preview window."""
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(str(img_path))
        preview = tk.Toplevel(self)
        preview.title(title)
        preview.geometry("1350x850")
        preview.minsize(960, 600)
        preview.resizable(True, True)
        preview.transient(self)

        outer = ttk.Frame(preview, padding=(8, 8))
        outer.pack(fill=tk.BOTH, expand=True)
        header = ttk.Frame(outer)
        header.pack(fill=tk.X, side=tk.TOP)
        ttk.Label(
            header,
            text=info_text or "Preview generated from the first Auto Screen parameter set and first experimental image.",
            anchor="w",
            justify=tk.LEFT,
            font=("Arial", 10, "bold"),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(header, text="Close", command=preview.destroy).pack(side=tk.RIGHT, padx=(8, 0))

        zoom_bar = ttk.Frame(outer)
        zoom_bar.pack(fill=tk.X, side=tk.TOP, pady=(6, 6))
        zoom_label_var = tk.StringVar(value="Zoom: auto-fit")

        # Keep the score calculation visible.  A simple two-column grid is more
        # reliable than a Panedwindow on older Tk builds, where the right pane can
        # collapse and make the score text appear to be missing.
        body_frame = ttk.Frame(outer)
        body_frame.pack(fill=tk.BOTH, expand=True)
        body_frame.columnconfigure(0, weight=4, minsize=520)
        body_frame.columnconfigure(1, weight=1, minsize=420)
        body_frame.rowconfigure(0, weight=1)
        image_panel = ttk.Frame(body_frame)
        score_panel = ttk.LabelFrame(body_frame, text="Score", padding=(8, 6))
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        score_panel.grid(row=0, column=1, sticky="nsew")

        image_panel.rowconfigure(0, weight=1)
        image_panel.columnconfigure(0, weight=1)
        canvas = tk.Canvas(image_panel, highlightthickness=0)
        yscroll = ttk.Scrollbar(image_panel, orient=tk.VERTICAL, command=canvas.yview)
        xscroll = ttk.Scrollbar(image_panel, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        ttk.Label(score_panel, text="Total score calculation", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 2))
        formula_hint = (
            "S = sum(W*M)/sum(W)\n"
            "A, B, C... = score terms  |  M = converted mismatch  |  dS = contribution"
        )
        ttk.Label(
            score_panel,
            text=formula_hint,
            foreground="#333333",
            wraplength=460,
            justify=tk.LEFT,
            font=("Arial", 10),
        ).pack(anchor="w", pady=(0, 8))
        score_text_frame = ttk.Frame(score_panel)
        score_text_frame.pack(fill=tk.BOTH, expand=True)
        score_box = tk.Text(score_text_frame, wrap="none", width=74, height=28, font=("Consolas", 10))
        score_scroll = ttk.Scrollbar(score_text_frame, orient=tk.VERTICAL, command=score_box.yview)
        score_xscroll = ttk.Scrollbar(score_text_frame, orient=tk.HORIZONTAL, command=score_box.xview)
        score_box.configure(yscrollcommand=score_scroll.set, xscrollcommand=score_xscroll.set)
        score_box.grid(row=0, column=0, sticky="nsew")
        score_scroll.grid(row=0, column=1, sticky="ns")
        score_xscroll.grid(row=1, column=0, sticky="ew")
        score_text_frame.rowconfigure(0, weight=1)
        score_text_frame.columnconfigure(0, weight=1)
        score_box.insert("1.0", score_text or "No score calculation text was generated.")
        score_box.configure(state="disabled")

        photo_ref = {"image": None, "pil": None, "scale": 1.0, "item": None}

        def _render_preview(scale=None, center=False):
            if Image is None or ImageTk is None or photo_ref.get("pil") is None:
                return
            if scale is not None:
                photo_ref["scale"] = max(0.04, min(8.0, float(scale)))
            pil_img = photo_ref["pil"]
            s = float(photo_ref["scale"])
            new_w = max(1, int(round(pil_img.width * s)))
            new_h = max(1, int(round(pil_img.height * s)))
            if abs(s - 1.0) < 1e-9:
                show_img = pil_img
            else:
                resample = getattr(Image, "Resampling", Image).LANCZOS
                show_img = pil_img.resize((new_w, new_h), resample)
            photo = ImageTk.PhotoImage(show_img)
            photo_ref["image"] = photo
            if photo_ref.get("item") is None:
                photo_ref["item"] = canvas.create_image(0, 0, image=photo, anchor="nw")
            else:
                canvas.itemconfigure(photo_ref["item"], image=photo)
            canvas.configure(scrollregion=(0, 0, new_w, new_h))
            zoom_label_var.set(f"Zoom: {int(round(s * 100))}%")
            if center:
                canvas.xview_moveto(0.0)
                canvas.yview_moveto(0.0)

        def _fit_scale(mode="window"):
            if Image is None or ImageTk is None or photo_ref.get("pil") is None:
                return None
            canvas.update_idletasks()
            available_w = max(120, canvas.winfo_width() - 8)
            available_h = max(120, canvas.winfo_height() - 8)
            pil = photo_ref["pil"]
            scale_w = available_w / max(1, pil.width)
            scale_h = available_h / max(1, pil.height)
            if mode == "width":
                return scale_w
            return min(scale_w, scale_h)

        def _zoom_by(factor):
            old_scale = float(photo_ref.get("scale", 1.0))
            new_scale = old_scale * float(factor)
            _render_preview(new_scale)

        def _zoom_reset():
            _render_preview(1.0, center=True)

        def _zoom_fit_width():
            scale = _fit_scale("width")
            if scale is not None:
                _render_preview(scale, center=True)

        def _zoom_fit_window():
            scale = _fit_scale("window")
            if scale is not None:
                _render_preview(scale, center=True)

        ttk.Button(zoom_bar, text="Zoom −", command=lambda: _zoom_by(0.80)).pack(side=tk.LEFT)
        ttk.Button(zoom_bar, text="Zoom +", command=lambda: _zoom_by(1.25)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(zoom_bar, text="100%", command=_zoom_reset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(zoom_bar, text="Fit width", command=_zoom_fit_width).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(zoom_bar, text="Fit window", command=_zoom_fit_window).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(zoom_bar, textvariable=zoom_label_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(zoom_bar, text="Mouse wheel: zoom | Shift+wheel: horizontal scroll", foreground="#555555").pack(side=tk.LEFT, padx=(12, 0))

        if Image is not None and ImageTk is not None:
            with Image.open(img_path) as _im:
                photo_ref["pil"] = _im.convert("RGB")
            preview.after(150, _zoom_fit_window)
        else:
            fallback_photo = tk.PhotoImage(file=str(img_path))
            photo_ref["image"] = fallback_photo
            photo_ref["item"] = canvas.create_image(0, 0, image=fallback_photo, anchor="nw")
            canvas.configure(scrollregion=(0, 0, int(fallback_photo.width()), int(fallback_photo.height())))
            zoom_label_var.set("Zoom: unavailable without Pillow")

        def _on_mousewheel(event):
            # Default wheel zooms the image. Use Shift+wheel for horizontal scroll.
            if getattr(event, "state", 0) & 0x0001:
                delta = getattr(event, "delta", 0)
                if delta:
                    canvas.xview_scroll(int(-1 * (delta / 120)), "units")
                return "break"
            delta = getattr(event, "delta", 0)
            num = getattr(event, "num", None)
            if delta > 0 or num == 4:
                _zoom_by(1.10)
            elif delta < 0 or num == 5:
                _zoom_by(1.0 / 1.10)
            return "break"

        def _bind_zoom_targets():
            for target in (preview, canvas, image_panel):
                try:
                    target.bind("<MouseWheel>", _on_mousewheel, add="+")
                except Exception:
                    pass
                try:
                    target.bind("<Button-4>", _on_mousewheel, add="+")
                    target.bind("<Button-5>", _on_mousewheel, add="+")
                except Exception:
                    pass
            try:
                canvas.bind("<Shift-MouseWheel>", _on_mousewheel, add="+")
            except Exception:
                pass

        preview._autoscreen_preview_photo_ref = photo_ref
        preview._autoscreen_autofit_done = False

        def _initial_fit_once(_event=None):
            if getattr(preview, "_autoscreen_autofit_done", False):
                return
            if Image is None or ImageTk is None or photo_ref.get("pil") is None:
                return
            if canvas.winfo_width() < 50 or canvas.winfo_height() < 50:
                return
            _zoom_fit_window()
            preview._autoscreen_autofit_done = True
            _bind_zoom_targets()

        canvas.bind("<Configure>", _initial_fit_once, add="+")
        preview.after(250, _initial_fit_once)
        _bind_zoom_targets()
        canvas.focus_set()
        preview.lift()
        try:
            preview.grab_set()
            preview.after(100, preview.grab_release)
        except Exception:
            pass
        return preview
    def test_output_one_image(self):
        """Generate one comparison summary and show it in a preview window.

        The preview uses the first sweep condition and the first experimental image.
        It no longer writes the test summary/CSV/config into the selected output
        folder; full Auto Screen runs still save their normal outputs.
        """
        app = self.app
        if not hasattr(app, "update_plot") or not hasattr(app, "vars"):
            messagebox.showerror("Auto screen", "This simulator does not expose the required API (vars, update_plot).", parent=self)
            return
        original_values = {}
        for name, var in getattr(app, "vars", {}).items():
            try:
                original_values[name] = var.get()
            except Exception:
                pass
        temp_dir_obj = None
        try:
            param_values, total = self._collect_sweep()
            experiments = self._load_experimental_records()
            thresholds = self._thresholds()
            if not experiments:
                raise ValueError("No experimental image was loaded for the preview.")
            combo = {name: values[0] for name, values in param_values.items() if values}
            cache_dir = self._resolve_simulator_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)

            self.cancel_requested = False
            self._set_running_state(True)
            self.progress_var.set("Generating one Auto Screen preview image...")
            self.update_idletasks()
            self.root.update_idletasks()

            sim_data = self._get_or_compute_simulation(combo, cache_dir)
            exp = experiments[0]
            row, comparison = self._comparison_row(1, max(1, int(total)), combo, exp, sim_data, thresholds)
            temp_dir_obj = tempfile.TemporaryDirectory(prefix="saxs_autoscreen_preview_")
            preview_dir = Path(temp_dir_obj.name)
            summary_path, _heatmap_path, _exp_mask_path, _sim_mask_path = self._save_comparison_figures(
                row, comparison, preview_dir, bucket="preview"
            )
            param_items = [(k[6:], v) for k, v in row.items() if isinstance(k, str) and k.startswith("param_")]
            param_str = ", ".join(f"{k}={_format_param_value(v)}" for k, v in param_items[:6])
            info = (
                f"Preview only — not saved to the Auto Screen output folder.  "
                f"Experimental: {row.get('experimental_id', '')} | "
                f"accepted={row.get('accepted', False)} | "
                f"total_score={_format_feature_metric(row.get('total_score', np.nan))}"
            )
            if param_str:
                info += f" | {param_str}"
            score_text = self._format_autoscreen_score_calculation(row, thresholds)
            preview = self._show_autoscreen_preview_image(
                summary_path,
                title="Auto Screen one-image preview",
                info_text=info,
                score_text=score_text,
            )
            # Keep the temporary PNG alive only while the preview window exists.
            preview._autoscreen_tempdir = temp_dir_obj
            temp_dir_obj = None
            def _cleanup_tempdir(_event=None, _preview=preview):
                tmp = getattr(_preview, "_autoscreen_tempdir", None)
                if tmp is not None:
                    try:
                        tmp.cleanup()
                    except Exception:
                        pass
                    _preview._autoscreen_tempdir = None
            preview.bind("<Destroy>", _cleanup_tempdir, add="+")
            self.progress_var.set("One-image Auto Screen preview generated. No test output was saved to the output folder.")
        except Exception as exc:
            messagebox.showerror(
                "Auto screen preview",
                f"Could not generate one-image preview.\n\n{type(exc).__name__}: {exc}",
                parent=self,
            )
        finally:
            if temp_dir_obj is not None:
                try:
                    temp_dir_obj.cleanup()
                except Exception:
                    pass
            self._set_running_state(False)
            for name, value in original_values.items():
                try:
                    app.vars[name].set(value)
                except Exception:
                    pass
            try:
                app.update_plot()
                self.root.update_idletasks()
            except Exception:
                pass

    def run_local_scan(self):
        """Fill checked rows around current parameters, run screening, and export a compact local-scan score summary."""
        try:
            target_specs, _target_summary = self._experimental_target_summary_for_local_scan()
        except Exception as exc:
            messagebox.showerror("Run local scan", f"Experimental target specification is not valid.\n\n{type(exc).__name__}: {exc}", parent=self)
            return
        if not target_specs:
            messagebox.showinfo("Run local scan", "Set the experimental target first.\n\nUse Experimental CSV or Manual experimental data spec in the Auto Screen window.", parent=self)
            self.local_scan_status_var.set("Local scan needs an experimental target first.")
            return
        checked = [(name, row) for name, row in self.param_rows.items() if bool(row["use"].get())]
        if not checked:
            messagebox.showinfo("Run local scan", "Check one or more parameter rows first.\n\nRun local scan refines only checked parameters around the current simulator values.", parent=self)
            self.local_scan_status_var.set("No parameter checked. Check parameters to scan first.")
            return
        filled = self.fill_local_scan_from_current()
        if not filled:
            return
        self.run_screening(local_scan_run=True)

    def _local_scan_settings_snapshot(self):
        checked = []
        for name, row in getattr(self, "param_rows", {}).items():
            try:
                if bool(row["use"].get()):
                    checked.append({
                        "parameter": name,
                        "current": _screening_safe_float(row.get("current"), float("nan")),
                        "start": str(row["start"].get()),
                        "stop": str(row["stop"].get()),
                        "step": str(row["step"].get()),
                        "points": str(row["points"].get()),
                        "type": "angle" if self._is_angle_like_parameter(name) else "normal",
                    })
            except Exception:
                continue
        return {
            "normal_percent_span": str(self.local_scan_percent_var.get()),
            "angle_degree_span": str(self.local_scan_angle_span_var.get()),
            "points_requested": str(self.local_scan_points_var.get()),
            "checked_parameters": checked,
        }

    def _row_float(self, row, key, default=float("nan")):
        try:
            value = row.get(key, default)
            if value in (None, ""):
                return default
            return float(value)
        except Exception:
            return default

    def _row_bool(self, row, key):
        value = row.get(key, False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    def _compact_row_for_local_summary(self, row, rank, param_fields):
        metric_fields = [
            "radial_profile_corr", "radial_profile_nrmse", "radial_chi2_reduced",
            "integration_curve_corr", "integration_curve_nrmse", "integration_chi2_reduced", "fringe_position_rel_error",
            "pixel_pearson_corr_asinh_z", "harmonics_mean_corr_A2_A6", "harmonic_phase_error_weighted_A2_A6_deg",
            "angular_profile_corr_zero_shift", "polar_corr_zero_shift", "polar_xcorr_best_corr",
        ]
        out = {
            "rank": int(rank),
            "accepted": row.get("accepted", ""),
            "total_score": row.get("total_score", ""),
            "experimental_id": row.get("experimental_id", ""),
            "run_name": row.get("run_name", ""),
            "run_number": row.get("run_number", ""),
            "image_number": row.get("image_number", ""),
            "combo_index": row.get("combo_index", ""),
            "summary_figure_path": row.get("summary_figure_path", ""),
            "score_contribution_summary": row.get("score_contribution_summary", ""),
        }
        for field in metric_fields:
            out[field] = row.get(field, "")
        for field in param_fields:
            out[field] = row.get(field, "")
        return out

    def _write_local_scan_scoring_summary(self, rows, output_dir, prefix, param_fields, total_combos, experiments, thresholds, elapsed_seconds, stopped_early=False):
        output_dir = Path(output_dir)
        summary_dir = output_dir / "local_scan_summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        clean_rows = [dict(r) for r in rows if np.isfinite(self._row_float(r, "total_score", float("nan")))]
        clean_rows.sort(key=lambda r: self._row_float(r, "total_score", float("inf")))
        accepted_rows = [r for r in clean_rows if self._row_bool(r, "accepted")]
        top_n = min(50, len(clean_rows))
        top_path = summary_dir / f"{prefix}_local_scan_top{top_n if top_n else 0}.csv"
        compact_fieldnames = [
            "rank", "accepted", "total_score", "experimental_id", "run_name", "run_number", "image_number", "combo_index",
            "summary_figure_path", "score_contribution_summary",
            "radial_profile_corr", "radial_profile_nrmse", "radial_chi2_reduced",
            "integration_curve_corr", "integration_curve_nrmse", "integration_chi2_reduced", "fringe_position_rel_error",
            "pixel_pearson_corr_asinh_z", "harmonics_mean_corr_A2_A6", "harmonic_phase_error_weighted_A2_A6_deg",
            "angular_profile_corr_zero_shift", "polar_corr_zero_shift", "polar_xcorr_best_corr",
            *param_fields,
        ]
        with top_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=compact_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rank, row in enumerate(clean_rows[:top_n], start=1):
                writer.writerow(self._compact_row_for_local_summary(row, rank, param_fields))
        accepted_top_n = min(50, len(accepted_rows))
        accepted_top_path = summary_dir / f"{prefix}_local_scan_accepted_top{accepted_top_n if accepted_top_n else 0}.csv"
        with accepted_top_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=compact_fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rank, row in enumerate(accepted_rows[:accepted_top_n], start=1):
                writer.writerow(self._compact_row_for_local_summary(row, rank, param_fields))
        best = clean_rows[0] if clean_rows else {}
        best_accepted = accepted_rows[0] if accepted_rows else {}
        settings = self._local_scan_settings_snapshot()
        summary_payload = {
            "mode": "local_scan",
            "simulator": self.simulator_name,
            "created_unix": time.time(),
            "stopped_early": bool(stopped_early),
            "output_dir": str(output_dir),
            "summary_dir": str(summary_dir),
            "n_simulation_combinations": int(total_combos),
            "n_experimental_targets": int(len(experiments)),
            "n_comparisons_written": int(len(rows)),
            "n_scored_rows": int(len(clean_rows)),
            "n_accepted_rows": int(len(accepted_rows)),
            "elapsed_seconds": float(elapsed_seconds),
            "local_scan_settings": settings,
            "thresholds": thresholds,
            "best_overall": self._compact_row_for_local_summary(best, 1, param_fields) if best else {},
            "best_accepted": self._compact_row_for_local_summary(best_accepted, 1, param_fields) if best_accepted else {},
            "files": {"top_candidates_csv": str(top_path), "accepted_top_csv": str(accepted_top_path)},
        }
        json_path = summary_dir / f"{prefix}_local_scan_summary.json"
        json_path.write_text(json.dumps(_jsonify_params(summary_payload), indent=2, sort_keys=True), encoding="utf-8")
        txt_path = summary_dir / f"{prefix}_local_scan_summary.txt"
        lines = [
            "Local scan scoring summary",
            "==========================",
            f"Simulator: {self.simulator_name}",
            f"Status: {'stopped early' if stopped_early else 'finished'}",
            f"Simulation combinations: {total_combos}",
            f"Experimental targets: {len(experiments)}",
            f"Comparisons written: {len(rows)}",
            f"Scored rows: {len(clean_rows)}",
            f"Accepted rows: {len(accepted_rows)}",
            f"Elapsed: {_format_elapsed_seconds(elapsed_seconds)}",
            "", "Local scan settings:",
            f"  normal +/- %: {settings.get('normal_percent_span')}",
            f"  angle +/- deg: {settings.get('angle_degree_span')}",
            f"  points requested: {settings.get('points_requested')}",
        ]
        for item in settings.get("checked_parameters", []):
            lines.append(f"  - {item.get('parameter')} ({item.get('type')}): current={item.get('current')}, start={item.get('start')}, stop={item.get('stop')}, step={item.get('step')}, points={item.get('points')}")
        if best:
            lines.extend(["", "Best overall:", f"  total_score: {best.get('total_score')}", f"  accepted: {best.get('accepted')}", f"  experimental_id: {best.get('experimental_id')}", f"  combo_index: {best.get('combo_index')}", f"  summary_figure_path: {best.get('summary_figure_path')}"])
        if best_accepted:
            lines.extend(["", "Best accepted:", f"  total_score: {best_accepted.get('total_score')}", f"  experimental_id: {best_accepted.get('experimental_id')}", f"  combo_index: {best_accepted.get('combo_index')}", f"  summary_figure_path: {best_accepted.get('summary_figure_path')}"])
        lines.extend(["", "Files:", f"  top candidates: {top_path}", f"  accepted top candidates: {accepted_top_path}", f"  JSON summary: {json_path}"])
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {
            "summary_dir": str(summary_dir),
            "top_candidates_csv": str(top_path),
            "accepted_top_csv": str(accepted_top_path),
            "summary_json": str(json_path),
            "summary_txt": str(txt_path),
            "n_scored_rows": len(clean_rows),
            "n_accepted_rows": len(accepted_rows),
        }

    def run_screening(self, local_scan_run=False):
        app = self.app
        if not hasattr(app, "update_plot") or not hasattr(app, "vars"):
            messagebox.showerror("Auto screen", "This simulator does not expose the required API (vars, update_plot).", parent=self)
            return
        try:
            param_values, total = self._collect_sweep()
            experiments = self._load_experimental_records()
            thresholds = self._thresholds()
        except Exception as exc:
            messagebox.showerror("Auto screen setup error", str(exc), parent=self)
            return
        output_dir = Path(self.output_dir_var.get()).expanduser()
        cache_root = Path(self.cache_dir_var.get()).expanduser()
        cache_dir = self._resolve_simulator_cache_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        prefix = _safe_filename_fragment(self.prefix_var.get() or self.simulator_name)
        original_values = {}
        for name, var in app.vars.items():
            try:
                original_values[name] = var.get()
            except Exception:
                pass
        keys = list(param_values.keys())
        value_lists = [param_values[k] for k in keys]
        active_names = [name for name, row in self.param_rows.items() if row["use"].get()]
        if not active_names:
            active_names = keys
        param_fields = [f"param_{name}" for name in keys]
        fieldnames = [
            "combo_index", "total_combos", "experimental_id", "run_name", "run_number", "image_number",
            "npz_name", "local_number", "simulator", "cache_key", "cache_path", "from_cache",
            "screening_poisson_noise_enabled", "screening_poisson_peak_counts", "screening_poisson_seed",
            "screening_poisson_reference_mode", "screening_poisson_reference_intensity", "screening_poisson_count_multiplier",
            "summary_figure_path", "harmonic_heatmap_path", "experimental_masked_figure_path", "theoretical_masked_figure_path",
            *param_fields,
            *self.METRIC_FIELDS,
        ]
        all_path = output_dir / f"{prefix}_all_scores.csv"
        accepted_path = output_dir / f"{prefix}_accepted.csv"
        exp_manifest_path = output_dir / f"{prefix}_experiments.csv"
        config_path = output_dir / f"{prefix}_screening_config.json"
        with exp_manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "experimental_id", "base_dir", "run_name", "run_number", "image_number",
                    "npz_name", "local_number", "shape", "source", "csv_path", "csv_row",
                ],
            )
            writer.writeheader()
            for exp in experiments:
                writer.writerow({
                    "experimental_id": exp["id"],
                    "base_dir": exp["base_dir"],
                    "run_name": exp["run_name"],
                    "run_number": exp["run_number"],
                    "image_number": exp["image_number"],
                    "npz_name": exp["npz_name"],
                    "local_number": exp["local_number"],
                    "shape": "x".join(str(x) for x in exp["image"].shape),
                    "source": exp.get("source", ""),
                    "csv_path": exp.get("csv_path", ""),
                    "csv_row": exp.get("csv_row", ""),
                })
        noise_enabled_cfg, noise_peak_cfg, noise_seed_cfg, noise_ref_cfg, noise_ref_mode_cfg, noise_multiplier_cfg = self._screening_poisson_state()
        config = {
            "simulator": self.simulator_name,
            "created_unix": time.time(),
            "output_dir": str(output_dir),
            "cache_root": str(cache_root),
            "cache_dir": str(cache_dir),
            "cache_subfolder_by_simulator": True,
            "thresholds": thresholds,
            "param_values": {k: [float(v) for v in vals] for k, vals in param_values.items()},
            "n_experimental_images": len(experiments),
            "screening_poisson_noise_enabled": bool(noise_enabled_cfg),
            "screening_poisson_peak_counts": noise_peak_cfg,
            "screening_poisson_seed": noise_seed_cfg,
            "screening_poisson_reference_mode": noise_ref_mode_cfg,
            "screening_poisson_reference_intensity": noise_ref_cfg,
            "screening_poisson_count_multiplier": noise_multiplier_cfg,
            "cache_stores_clean_simulation_only": True,
            "poisson_noise_applied_after_cache_load_before_comparison": True,
            "skip_comparison_if_simulation_cache_exists": bool(self.skip_cached_comparison_var.get()),
            "run_mode": "local_scan" if local_scan_run else "screening",
            "local_scan_settings": self._local_scan_settings_snapshot() if local_scan_run else {},
        }
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        self.cancel_requested = False
        self._set_running_state(True)
        stopped_early = False
        n_rows = 0
        n_accept = 0
        n_cached = 0
        n_cache_skipped_combos = 0
        n_cache_skipped_rows = 0
        skip_cached_comparison = bool(self.skip_cached_comparison_var.get())
        start_time = time.time()
        total_rows_expected = max(1, int(total) * max(1, len(experiments)))
        local_scan_rows = [] if local_scan_run else None
        local_scan_summary = None
        try:
            with all_path.open("w", newline="", encoding="utf-8") as f_all, accepted_path.open("w", newline="", encoding="utf-8") as f_acc:
                all_writer = csv.DictWriter(f_all, fieldnames=fieldnames, extrasaction="ignore")
                acc_writer = csv.DictWriter(f_acc, fieldnames=fieldnames, extrasaction="ignore")
                all_writer.writeheader()
                acc_writer.writeheader()
                for combo_index, values in enumerate(itertools.product(*value_lists), start=1):
                    if self.cancel_requested:
                        stopped_early = True
                        break
                    combo = dict(zip(keys, values))
                    cache_key, cache_path = self._simulation_cache_path_for_combo(combo, cache_dir)
                    if skip_cached_comparison and cache_path.exists():
                        n_cached += 1
                        n_cache_skipped_combos += 1
                        n_cache_skipped_rows += len(experiments)
                        elapsed = time.time() - start_time
                        rows_done_for_eta = max(1, n_rows + n_cache_skipped_rows)
                        rows_per_sec = rows_done_for_eta / max(elapsed, 1.0e-9)
                        rows_remaining = max(0, total_rows_expected - n_rows - n_cache_skipped_rows)
                        eta_seconds = rows_remaining / rows_per_sec if rows_per_sec > 0 else float("nan")
                        self.progress_var.set(
                            f"Screened {combo_index}/{total} simulations × {len(experiments)} experiments "
                            f"({n_rows}/{total_rows_expected} comparisons, skipped={n_cache_skipped_rows}) | "
                            f"accepted={n_accept}, cache_hits={n_cached}, cache_skipped_combos={n_cache_skipped_combos} | "
                            f"elapsed={_format_elapsed_seconds(elapsed)}, ETA={_format_elapsed_seconds(eta_seconds)}"
                        )
                        self.update_idletasks()
                        self.root.update()
                        continue
                    sim_data = self._get_or_compute_simulation(combo, cache_dir)
                    if sim_data.get("from_cache", False):
                        n_cached += 1
                    for exp in experiments:
                        if self.cancel_requested:
                            stopped_early = True
                            break
                        try:
                            row, comparison = self._comparison_row(combo_index, total, combo, exp, sim_data, thresholds)
                            if bool(row.get("accepted", False)):
                                summary_path_acc, heatmap_path_acc, exp_mask_path_acc, sim_mask_path_acc = self._save_comparison_figures(row, comparison, output_dir, bucket="accepted")
                                row["summary_figure_path"] = summary_path_acc
                                row["harmonic_heatmap_path"] = heatmap_path_acc
                                row["experimental_masked_figure_path"] = exp_mask_path_acc
                                row["theoretical_masked_figure_path"] = sim_mask_path_acc
                        except Exception as exc:
                            row = {
                                "combo_index": combo_index,
                                "total_combos": total,
                                "experimental_id": exp.get("id", ""),
                                "run_name": exp.get("run_name", ""),
                                "run_number": exp.get("run_number", ""),
                                "image_number": exp.get("image_number", ""),
                                "npz_name": exp.get("npz_name", ""),
                                "local_number": exp.get("local_number", ""),
                                "simulator": self.simulator_name,
                                "cache_key": sim_data.get("cache_key", ""),
                                "cache_path": sim_data.get("cache_path", ""),
                                "from_cache": bool(sim_data.get("from_cache", False)),
                                "screening_poisson_noise_enabled": False,
                                "screening_poisson_peak_counts": np.nan,
                                "screening_poisson_seed": "",
                                "summary_figure_path": "",
                                "harmonic_heatmap_path": "",
                                "experimental_masked_figure_path": "",
                                "theoretical_masked_figure_path": "",
                                "accepted": False,
                                "total_score": np.nan,
                            }
                            for name, value in combo.items():
                                row[f"param_{name}"] = value
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        n_rows += 1
                        if local_scan_rows is not None:
                            local_scan_rows.append(dict(row))
                        if self.save_all_var.get():
                            all_writer.writerow(row)
                        if bool(row.get("accepted", False)):
                            n_accept += 1
                            acc_writer.writerow(row)
                    elapsed = time.time() - start_time
                    rows_done_for_eta = max(1, n_rows + n_cache_skipped_rows)
                    rows_per_sec = rows_done_for_eta / max(elapsed, 1.0e-9)
                    rows_remaining = max(0, total_rows_expected - n_rows - n_cache_skipped_rows)
                    eta_seconds = rows_remaining / rows_per_sec if rows_per_sec > 0 else float("nan")
                    self.progress_var.set(
                        f"Screened {combo_index}/{total} simulations × {len(experiments)} experiments "
                        f"({n_rows}/{total_rows_expected} comparisons, skipped={n_cache_skipped_rows}) | "
                        f"accepted={n_accept}, cache_hits={n_cached}, cache_skipped_combos={n_cache_skipped_combos} | "
                        f"elapsed={_format_elapsed_seconds(elapsed)}, ETA={_format_elapsed_seconds(eta_seconds)}"
                    )
                    self.update_idletasks()
                    self.root.update()
        except Exception as exc:
            messagebox.showerror("Auto screen", f"Screening failed\n\n{type(exc).__name__}: {exc}", parent=self)
            return
        finally:
            self._set_running_state(False)
            for name, value in original_values.items():
                try:
                    app.vars[name].set(value)
                except Exception:
                    pass
            try:
                app.update_plot()
                self.root.update_idletasks()
            except Exception:
                pass
        total_elapsed = time.time() - start_time
        if local_scan_run:
            try:
                local_scan_summary = self._write_local_scan_scoring_summary(
                    local_scan_rows or [],
                    output_dir=output_dir,
                    prefix=prefix,
                    param_fields=param_fields,
                    total_combos=total,
                    experiments=experiments,
                    thresholds=thresholds,
                    elapsed_seconds=total_elapsed,
                    stopped_early=stopped_early,
                )
            except Exception as exc:
                local_scan_summary = {"error": f"{type(exc).__name__}: {exc}"}
        avg_rate = n_rows / max(total_elapsed, 1.0e-9)
        msg = (
            f"Screening {'stopped early' if stopped_early else 'finished'}.\n"
            f"Compared rows: {n_rows}\n"
            f"Accepted rows: {n_accept}\n"
            f"Simulation cache hits: {n_cached}\n"
            f"Cache-skipped simulations: {n_cache_skipped_combos}\n"
            f"Cache-skipped comparison rows: {n_cache_skipped_rows}\n"
            f"Elapsed time: {_format_elapsed_seconds(total_elapsed)}\n"
            f"Average comparison rate: {avg_rate:.2f} rows/s\n\n"
            f"Output:\n{output_dir}\n\n"
            f"Cache:\n{cache_dir}"
        )
        if local_scan_run:
            if isinstance(local_scan_summary, dict) and local_scan_summary.get("summary_dir"):
                msg += (
                    f"\n\nLocal scan scoring summary:\n{local_scan_summary.get('summary_dir')}\n"
                    f"Top candidates CSV:\n{local_scan_summary.get('top_candidates_csv')}\n"
                    f"Accepted top CSV:\n{local_scan_summary.get('accepted_top_csv')}"
                )
            elif isinstance(local_scan_summary, dict) and local_scan_summary.get("error"):
                msg += f"\n\nLocal scan scoring summary failed:\n{local_scan_summary.get('error')}"
        messagebox.showinfo("Auto screen" if not local_scan_run else "Local scan", msg, parent=self)
        self.progress_var.set(msg.replace("\n", " | "))
def _is_descendant_of_registered_plot(widget):
    """Return True when a Tk event originates from a matplotlib display widget."""
    try:
        current = widget
        while current is not None:
            if str(current) in PLOT_WIDGET_PATHS:
                return True
            current = getattr(current, "master", None)
    except Exception:
        return False
    return False
def _block_parameter_panel_mousewheel_when_over_plot(event):
    """Stop global MouseWheel handlers in embedded simulator panels."""
    if _is_descendant_of_registered_plot(getattr(event, "widget", None)):
        return "break"
    return None
class ZoomableFigureCanvasTkAgg:
    """FigureCanvasTkAgg wrapper with toolbar, mouse-wheel zoom, and Reset View."""
    def __init__(self, figure, master=None, *args, **kwargs):
        self.figure = figure
        self.container = ttk.Frame(master)
        self.canvas = _OriginalFigureCanvasTkAgg(figure, master=self.container, *args, **kwargs)
        toolbar_row = ttk.Frame(self.container)
        toolbar_row.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_row, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.reset_button = ttk.Button(toolbar_row, text="Reset View", command=self.reset_view)
        self.reset_button.pack(side=tk.RIGHT, padx=(6, 2))
        self._initial_views = None
        self._capture_after_draw_id = None
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        PLOT_WIDGET_PATHS.add(str(widget))
        # Handle mouse-wheel zoom directly at the Tk-canvas level.
        # Returning "break" here prevents simulator-side bind_all("<MouseWheel>")
        # handlers from scrolling the parameter panel, while keeping zoom active.
        widget.bind("<MouseWheel>", self._on_tk_mousewheel_zoom, add="+")
        widget.bind("<Button-4>", self._on_tk_button4_zoom, add="+")
        widget.bind("<Button-5>", self._on_tk_button5_zoom, add="+")
    def get_tk_widget(self):
        return self.container
    def draw(self, *args, **kwargs):
        result = self.canvas.draw(*args, **kwargs)
        self._schedule_capture_initial_views()
        return result
    def draw_idle(self, *args, **kwargs):
        result = self.canvas.draw_idle(*args, **kwargs)
        self._schedule_capture_initial_views()
        return result
    def mpl_connect(self, *args, **kwargs):
        return self.canvas.mpl_connect(*args, **kwargs)
    def mpl_disconnect(self, *args, **kwargs):
        return self.canvas.mpl_disconnect(*args, **kwargs)
    def _schedule_capture_initial_views(self):
        try:
            if self._capture_after_draw_id is not None:
                self.container.after_cancel(self._capture_after_draw_id)
            self._capture_after_draw_id = self.container.after_idle(self._capture_initial_views)
        except Exception:
            self._capture_initial_views()
    def _capture_initial_views(self):
        self._capture_after_draw_id = None
        views = []
        for ax in self.figure.axes:
            try:
                views.append((ax, ax.get_xlim(), ax.get_ylim()))
            except Exception:
                pass
        self._initial_views = views
    def reset_view(self):
        if not self._initial_views:
            self._capture_initial_views()
        for ax, xlim, ylim in self._initial_views or []:
            try:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            except Exception:
                pass
        try:
            self.toolbar.update()
        except Exception:
            pass
        self.canvas.draw_idle()
    def _on_tk_button4_zoom(self, event):
        return self._on_tk_mousewheel_zoom(event, forced_direction="up")
    def _on_tk_button5_zoom(self, event):
        return self._on_tk_mousewheel_zoom(event, forced_direction="down")
    def _on_tk_mousewheel_zoom(self, event, forced_direction=None):
        """Zoom the axes under the cursor using the native Tk mouse event."""
        try:
            if getattr(self.toolbar, "mode", ""):
                return "break"
        except Exception:
            pass
        ax, xdata, ydata = self._axes_and_data_from_tk_event(event)
        if ax is None or xdata is None or ydata is None:
            return "break"
        if forced_direction is not None:
            direction = forced_direction
        else:
            delta = getattr(event, "delta", 0)
            direction = "up" if delta > 0 else "down"
        base_scale = 1.2
        scale_factor = 1.0 / base_scale if direction == "up" else base_scale
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        if cur_xlim[1] == cur_xlim[0] or cur_ylim[1] == cur_ylim[0]:
            return "break"
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()
        return "break"
    def _axes_and_data_from_tk_event(self, event):
        """Return (axes, xdata, ydata) below a Tk event, or (None, None, None)."""
        widget = self.canvas.get_tk_widget()
        try:
            x_disp = float(event.x)
            y_disp = float(widget.winfo_height() - event.y)
        except Exception:
            return None, None, None
        for ax in reversed(self.figure.axes):
            try:
                if not ax.get_visible():
                    continue
                if ax.bbox.contains(x_disp, y_disp):
                    xdata, ydata = ax.transData.inverted().transform((x_disp, y_disp))
                    return ax, xdata, ydata
            except Exception:
                continue
        return None, None, None
def _normalize_run_number(run_number):
    digits = "".join(ch for ch in str(run_number).strip() if ch.isdigit())
    if not digits:
        raise ValueError("Run number must contain digits, e.g. 245 or run00245.")
    return f"run{int(digits):05d}"
def _find_run_source(base_dir, run_number):
    """Find run000xxx folder or run000xxx.zip.
    base_dir may be either:
    - a parent directory containing run00245/ or run00245.zip
    - the run00245 directory itself
    - the run00245.zip file itself
    """
    base = Path(base_dir).expanduser()
    run_name = _normalize_run_number(run_number)
    if base.exists() and base.is_dir() and base.name.lower() == run_name.lower():
        return base, run_name
    if base.exists() and base.is_file() and base.name.lower() == f"{run_name}.zip".lower():
        return base, run_name
    candidates = [
        base / run_name,
        base / f"{run_name}.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, run_name
    raise FileNotFoundError(f"Could not find {run_name}/ or {run_name}.zip under:\n{base}")
def _load_run_image_number(run_source, image_number):
    """Load one 1-based image number from a run.
    image_number is 1..3000.
    cimg001.npz contains images 1..300, cimg002.npz 301..600, etc.
    Each npz must contain key 'img' with shape (300, H, W).
    """
    run_source = Path(run_source)
    image_number = int(image_number)
    if image_number < 1 or image_number > 3000:
        raise ValueError("Image number must be between 1 and 3000.")
    zero_based = image_number - 1
    file_index = zero_based // 300 + 1
    local_index = zero_based % 300
    npz_name = f"cimg{file_index:03d}.npz"
    if run_source.is_dir():
        npz_path = run_source / npz_name
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        with np.load(npz_path) as data:
            arr = data["img"]
            if arr.ndim != 3:
                raise ValueError(f"{npz_name} has shape {arr.shape}; expected (300, H, W).")
            if local_index >= arr.shape[0]:
                raise IndexError(f"Local image index {local_index + 1} is out of range for {npz_name}.")
            image = arr[local_index].astype(float)
    elif run_source.is_file() and run_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(run_source, "r") as z:
            match = None
            for name in z.namelist():
                if Path(name).name == npz_name:
                    match = name
                    break
            if match is None:
                raise FileNotFoundError(f"Missing file in zip: {npz_name}")
            with z.open(match) as f:
                with np.load(f) as data:
                    arr = data["img"]
                    if arr.ndim != 3:
                        raise ValueError(f"{npz_name} has shape {arr.shape}; expected (300, H, W).")
                    if local_index >= arr.shape[0]:
                        raise IndexError(f"Local image index {local_index + 1} is out of range for {npz_name}.")
                    image = arr[local_index].astype(float)
    else:
        raise FileNotFoundError(f"Invalid run source: {run_source}")
    return image, npz_name, local_index + 1
def _find_max_intensity_image_number(run_source):
    """Find the 1-based image number with the largest total intensity."""
    run_source = Path(run_source)
    best_sum = None
    best_image_number = None
    best_file = None
    best_local = None
    for file_index in range(1, 11):
        npz_name = f"cimg{file_index:03d}.npz"
        try:
            if run_source.is_dir():
                npz_path = run_source / npz_name
                if not npz_path.exists():
                    continue
                with np.load(npz_path) as data:
                    arr = data["img"]
                    sums = arr.reshape(arr.shape[0], -1).sum(axis=1)
            elif run_source.is_file() and run_source.suffix.lower() == ".zip":
                with zipfile.ZipFile(run_source, "r") as z:
                    match = None
                    for name in z.namelist():
                        if Path(name).name == npz_name:
                            match = name
                            break
                    if match is None:
                        continue
                    with z.open(match) as f:
                        with np.load(f) as data:
                            arr = data["img"]
                            sums = arr.reshape(arr.shape[0], -1).sum(axis=1)
            else:
                raise FileNotFoundError(f"Invalid run source: {run_source}")
        except Exception:
            raise
        local_zero = int(np.argmax(sums))
        value = float(sums[local_zero])
        if best_sum is None or value > best_sum:
            best_sum = value
            best_image_number = (file_index - 1) * 300 + local_zero + 1
            best_file = npz_name
            best_local = local_zero + 1
    if best_sum is None:
        raise FileNotFoundError("No cimg001.npz ... cimg010.npz files were found in the selected run.")
    return best_image_number, best_file, best_local, best_sum
def _estimate_poisson_reference_intensity(image, valid_mask=None, percentile=99.9):
    """Return a robust positive intensity level for Poisson count calibration."""
    arr = np.asarray(image, dtype=float)
    good = np.isfinite(arr) & (arr > 0)
    if valid_mask is not None:
        try:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape == arr.shape:
                good &= mask
        except Exception:
            pass
    if np.count_nonzero(good) < 10:
        return None
    try:
        pct = float(percentile)
    except Exception:
        pct = 99.9
    pct = min(100.0, max(0.0, pct))
    ref = float(np.nanpercentile(arr[good], pct))
    if not np.isfinite(ref) or ref <= 0:
        return None
    return ref
def _find_dataset_max_poisson_reference(run_source, percentile=99.9, valid_mask=None):
    """Find the strongest image in a run and estimate its robust Poisson reference count."""
    image_number, npz_name, local_number, total_intensity = _find_max_intensity_image_number(run_source)
    image, _, _ = _load_run_image_number(run_source, image_number)
    ref = _estimate_poisson_reference_intensity(image, valid_mask=valid_mask, percentile=percentile)
    if ref is None:
        raise ValueError("Could not estimate a positive Poisson reference intensity from the dataset max image.")
    return image_number, npz_name, local_number, total_intensity, ref
def _find_default_npz_max_poisson_reference(npz_path=None, percentile=DEFAULT_POISSON_REFERENCE_PERCENTILE, valid_mask=None):
    """Estimate the built-in Poisson reference from the strongest image in cimg001.npz.
    The bundled cimg001.npz is treated as the fixed detector-count reference.
    The image with the largest total intensity is selected, and a robust high
    percentile of that image is used as the base Poisson count level.
    """
    npz_path = Path(npz_path) if npz_path is not None else DEFAULT_POISSON_REFERENCE_NPZ
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing default Poisson reference dataset: {npz_path}")
    with np.load(npz_path) as data:
        if "img" not in data:
            raise KeyError(f"{npz_path.name} does not contain key 'img'.")
        arr = np.asarray(data["img"], dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{npz_path.name} has shape {arr.shape}; expected (N, H, W).")
    if arr.shape[0] < 1:
        raise ValueError(f"{npz_path.name} contains no images.")
    finite_arr = np.where(np.isfinite(arr), arr, 0.0)
    sums = finite_arr.reshape(finite_arr.shape[0], -1).sum(axis=1)
    local_zero = int(np.argmax(sums))
    image = arr[local_zero]
    total_intensity = float(sums[local_zero])
    ref = _estimate_poisson_reference_intensity(image, valid_mask=valid_mask, percentile=percentile)
    if ref is None:
        raise ValueError("Could not estimate a positive Poisson reference intensity from the bundled cimg001.npz.")
    return {
        "mode": "bundled_cimg001_max",
        "npz_path": str(npz_path),
        "npz_name": npz_path.name,
        "image_number": int(local_zero + 1),
        "local_number": int(local_zero + 1),
        "total_intensity": total_intensity,
        "percentile": float(percentile),
        "reference_intensity": float(ref),
    }
class SimulatorHost(ttk.Frame):
    """Frame passed as a pseudo-root to legacy simulator GUI classes."""
    def title(self, *_args, **_kwargs):
        return None
    def geometry(self, *_args, **_kwargs):
        return None
    def resizable(self, *_args, **_kwargs):
        return None
def _load_default_saxs_valid_mask(shape):
    """Load SAXS_mask.npy as valid-pixel mask if possible."""
    try:
        mask = np.load(BASE_DIR / "SAXS_mask.npy")
        mask = np.squeeze(mask).astype(bool)
        if mask.shape == tuple(shape):
            return mask
    except Exception:
        pass
    return np.ones(tuple(shape), dtype=bool)
def _load_default_q_axes_from_poni(shape):
    """Load q_u/q_v axes from the bundled .poni file for the given image shape.
    Returns
    -------
    (q_u, q_v, meta) or (None, None, None) on failure.
    """
    poni_candidates = [
        BASE_DIR / "post_beamtime_tmp_saxs.poni",
        BASE_DIR / "post_beamtime_tmp_saxs.PONI",
    ]
    poni_path = None
    for candidate in poni_candidates:
        if candidate.exists():
            poni_path = candidate
            break
    if poni_path is None:
        return None, None, None
    try:
        from saxs_module import q_axes_from_poni_small_angle
        q_u, q_v, meta = q_axes_from_poni_small_angle(
            poni_path,
            image_shape=tuple(shape),
            q_unit="A^-1",
            center_mode="beam",
            row_origin="upper",
        )
        return q_u, q_v, meta
    except Exception:
        return None, None, None
def _radial_integrate_on_q_grid(image, q_u=None, q_v=None, valid_mask=None, npt=500, q_min=None, q_max=None):
    """Simple 1D SAXS-style radial integration on an existing q grid.
    This is intended for the already masked 500 x 500 experimental/simulation images.
    It performs azimuthal averaging in q = sqrt(q_u^2 + q_v^2).
    Accepted q-axis inputs
    ----------------------
    - 2D q_u, q_v arrays with the same shape as image
    - 1D q_u (length = image width) and 1D q_v (length = image height)
    If q axes are unavailable, a pixel-radius grid is used as a fallback.
    """
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}.")
    h, w = img.shape
    q = None
    if q_u is not None and q_v is not None:
        q_u = np.asarray(q_u, dtype=float)
        q_v = np.asarray(q_v, dtype=float)
        # Case 1: full 2D q grids already provided.
        if q_u.shape == img.shape and q_v.shape == img.shape:
            q = np.sqrt(q_u ** 2 + q_v ** 2)
        # Case 2: separate 1D q axes for horizontal/vertical directions.
        elif q_u.ndim == 1 and q_v.ndim == 1 and q_u.size == w and q_v.size == h:
            q = np.sqrt(q_v[:, None] ** 2 + q_u[None, :] ** 2)
    if q is None:
        yy, xx = np.indices(img.shape)
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        q = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        q_label = "pixel radius"
    else:
        q_label = r"$q$ ($\mathrm{\AA}^{-1}$)"
    if valid_mask is None:
        valid_mask = np.ones_like(img, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask).astype(bool)
        if valid_mask.shape != img.shape:
            valid_mask = np.ones_like(img, dtype=bool)
    finite = np.isfinite(img) & np.isfinite(q)
    good = finite & valid_mask
    # Exclude obvious masked sentinel pixels if present.
    good &= img > -1.0e5
    if not np.any(good):
        raise ValueError("No valid pixels available for integration.")
    q_good = q[good]
    i_good = img[good]
    if q_min is None:
        q_min = float(np.nanmin(q_good))
    if q_max is None:
        q_max = float(np.nanmax(q_good))
    if q_max <= q_min:
        raise ValueError("Invalid q range for integration.")
    npt = int(npt)
    if npt < 4:
        npt = 4
    edges = np.linspace(q_min, q_max, npt + 1)
    sums, _ = np.histogram(q_good, bins=edges, weights=i_good)
    sums_sq, _ = np.histogram(q_good, bins=edges, weights=i_good ** 2)
    counts, _ = np.histogram(q_good, bins=edges)
    q_cent = 0.5 * (edges[:-1] + edges[1:])
    intensity = np.full(npt, np.nan, dtype=float)
    std = np.full(npt, np.nan, dtype=float)
    sem = np.full(npt, np.nan, dtype=float)
    nz = counts > 0
    intensity[nz] = sums[nz] / counts[nz]
    gt1 = counts > 1
    if np.any(gt1):
        var_num = sums_sq[gt1] - (sums[gt1] ** 2) / counts[gt1]
        var_num = np.maximum(var_num, 0.0)
        variance = var_num / (counts[gt1] - 1)
        std[gt1] = np.sqrt(variance)
        sem[gt1] = std[gt1] / np.sqrt(counts[gt1])
    return q_cent, intensity, counts, q_label, std, sem
def _normalize_curve_for_overlay(y):
    y = np.asarray(y, dtype=float)
    out = y.copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    lo, hi = np.nanpercentile(out[finite], [1, 99])
    if hi <= lo:
        hi = np.nanmax(out[finite])
        lo = np.nanmin(out[finite])
    if hi > lo:
        out = (out - lo) / (hi - lo)
    return out
def _compute_overlay_scale(y_reference, y_to_scale):
    """Least-squares positive scale factor so scaled experimental overlays theoretical better."""
    yr = np.asarray(y_reference, dtype=float)
    ys = np.asarray(y_to_scale, dtype=float)
    good = np.isfinite(yr) & np.isfinite(ys)
    if np.count_nonzero(good) < 3:
        return 1.0
    yr = yr[good]
    ys = ys[good]
    denom = float(np.dot(ys, ys))
    if denom <= 0:
        return 1.0
    scale = float(np.dot(yr, ys) / denom)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return scale
def _make_q_phi_grid(shape, q_u=None, q_v=None):
    """Return q, phi grids for 2D SAXS feature extraction.
    q_u/q_v may be 2D grids or 1D horizontal/vertical axes.  If unavailable,
    pixel-radius coordinates centered on the image are used.
    """
    h, w = tuple(shape)
    q = None
    phi = None
    if q_u is not None and q_v is not None:
        q_u_arr = np.asarray(q_u, dtype=float)
        q_v_arr = np.asarray(q_v, dtype=float)
        if q_u_arr.shape == (h, w) and q_v_arr.shape == (h, w):
            qx = q_u_arr
            qy = q_v_arr
            q = np.sqrt(qx ** 2 + qy ** 2)
            phi = np.arctan2(qy, qx)
        elif q_u_arr.ndim == 1 and q_v_arr.ndim == 1 and q_u_arr.size == w and q_v_arr.size == h:
            qx = q_u_arr[None, :]
            qy = q_v_arr[:, None]
            q = np.sqrt(qy ** 2 + qx ** 2)
            phi = np.arctan2(qy, qx)
    if q is None or phi is None:
        yy, xx = np.indices((h, w))
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        dy = yy - cy
        dx = xx - cx
        q = np.sqrt(dx ** 2 + dy ** 2)
        phi = np.arctan2(dy, dx)
    return q, phi
def _valid_feature_mask(image, valid_mask=None):
    img = np.asarray(image, dtype=float)
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {img.shape}.")
    if valid_mask is None:
        mask = np.ones_like(img, dtype=bool)
    else:
        mask = np.asarray(valid_mask).astype(bool)
        if mask.shape != img.shape:
            mask = np.ones_like(img, dtype=bool)
    # The experimental loader and simulator may carry masked sentinel values.
    mask = mask & np.isfinite(img) & (img > -1.0e5)
    return mask
def _robust_positive_feature_image(image, valid_mask=None, low_pct=1.0, high_pct=99.5, smooth_sigma=0.0):
    """Return a robust, non-negative feature image for morphology/anisotropy analysis.
    Why this is stricter than a simple percentile clip
    -----------------------------------------------
    Experimental SAXS images can be background-subtracted and therefore may contain
    many small positive/negative fluctuations around zero.  If we simply map the
    1st percentile to zero, a negative lower percentile can shift almost the whole
    detector to positive values, making the pattern look like several overlapping
    blobs.  Here we instead estimate a background level near the median, clip values
    below that background to zero, and optionally smooth lightly so the global shape
    (e.g. nearly spherical vs elongated) is emphasized over pixel noise.
    """
    img = np.asarray(image, dtype=float)
    mask = _valid_feature_mask(img, valid_mask)
    out = np.zeros_like(img, dtype=float)
    if np.count_nonzero(mask) < 3:
        return out, mask, {"background": np.nan, "high": np.nan, "smoothed": False}
    vals = img[mask]
    lo, med, hi = np.nanpercentile(vals, [low_pct, 50.0, high_pct])
    if not np.isfinite(lo):
        lo = float(np.nanmin(vals))
    if not np.isfinite(med):
        med = float(np.nanmedian(vals))
    if (not np.isfinite(hi)):
        hi = float(np.nanmax(vals))
    # Prefer a background level near the median.  This suppresses weak bipolar noise
    # in background-subtracted experimental data and avoids turning most of the image
    # into positive signal when the lower percentile is negative.
    background = med
    if not np.isfinite(background):
        background = lo
    shifted = img - background
    shifted[~mask] = 0.0
    shifted = np.maximum(shifted, 0.0)
    smoothed = False
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            shifted = gaussian_filter(shifted, sigma=float(smooth_sigma), mode="nearest")
            shifted[~mask] = 0.0
            smoothed = True
        except Exception:
            smoothed = False
    pos_vals = shifted[mask]
    if np.count_nonzero(np.isfinite(pos_vals)) < 3:
        return out, mask, {"background": background, "high": np.nan, "smoothed": smoothed}
    hi_pos = float(np.nanpercentile(pos_vals, high_pct))
    if (not np.isfinite(hi_pos)) or hi_pos <= 0:
        hi_pos = float(np.nanmax(pos_vals))
    if (not np.isfinite(hi_pos)) or hi_pos <= 0:
        return out, mask, {"background": background, "high": hi_pos, "smoothed": smoothed}
    out[mask] = np.clip(shifted[mask] / hi_pos, 0.0, 1.0)
    out[~mask] = 0.0
    return out, mask, {"background": background, "high": hi_pos, "smoothed": smoothed}
def _zscore_asinh_feature_image(image, valid_mask=None):
    """Compress SAXS dynamic range with asinh and z-score only over valid pixels."""
    pos, mask, meta = _robust_positive_feature_image(image, valid_mask=valid_mask)
    out = np.zeros_like(pos, dtype=float)
    if np.count_nonzero(mask) < 3:
        return out, mask, meta
    vals = pos[mask]
    positive = vals[vals > 0]
    if positive.size > 0:
        scale = float(np.nanpercentile(positive, 90.0))
    else:
        scale = 1.0
    if (not np.isfinite(scale)) or scale <= 0:
        scale = 1.0
    transformed = np.arcsinh(pos / scale)
    mean = float(np.nanmean(transformed[mask]))
    std = float(np.nanstd(transformed[mask]))
    if (not np.isfinite(std)) or std <= 0:
        std = 1.0
    out[mask] = (transformed[mask] - mean) / std
    out[~mask] = 0.0
    meta = dict(meta)
    meta.update({"asinh_scale": scale, "mean": mean, "std": std})
    return out, mask, meta
def _signed_asinh_display_image(image, valid_mask=None, clip_pct=99.5, zscore=False):
    """Oscillation-preserving display transform for experimental/simulation SAXS images.
    Unlike the morphology-oriented positive feature image, this keeps the sign of the
    signal so ring oscillations and positive/negative background-subtracted structure
    remain visible in 2D.
    """
    img = np.asarray(image, dtype=float)
    mask = _valid_feature_mask(img, valid_mask)
    out = np.zeros_like(img, dtype=float)
    if np.count_nonzero(mask) < 3:
        return out, mask, {"scale": np.nan, "zscore": bool(zscore)}
    vals = img[mask]
    abs_vals = np.abs(vals[np.isfinite(vals)])
    if abs_vals.size > 0:
        scale = float(np.nanpercentile(abs_vals, clip_pct))
    else:
        scale = 1.0
    if (not np.isfinite(scale)) or scale <= 0:
        scale = float(np.nanmax(abs_vals)) if abs_vals.size else 1.0
    if (not np.isfinite(scale)) or scale <= 0:
        scale = 1.0
    transformed = np.arcsinh(img / scale)
    if zscore:
        mean = float(np.nanmean(transformed[mask]))
        std = float(np.nanstd(transformed[mask]))
        if (not np.isfinite(std)) or std <= 0:
            std = 1.0
        out[mask] = (transformed[mask] - mean) / std
        meta = {"scale": scale, "mean": mean, "std": std, "zscore": True}
    else:
        out[mask] = transformed[mask]
        meta = {"scale": scale, "zscore": False}
    out[~mask] = 0.0
    return out, mask, meta

def _imshow_signed_feature_panel(ax, data, mask, title, *, cmap=None):
    """Unified renderer for signed/asinh feature panels in both UI paths."""
    arr = np.asarray(data, dtype=float).copy()
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape == arr.shape:
            arr[~mask_arr] = np.nan
    finite = np.isfinite(arr)
    vmin = vmax = None
    if np.any(finite):
        vals = arr[finite]
        if cmap == "coolwarm":
            vmax = float(np.nanpercentile(np.abs(vals), 99.0))
            if (not np.isfinite(vmax)) or vmax <= 0:
                vmax = float(np.nanmax(np.abs(vals))) if vals.size else float("nan")
            if np.isfinite(vmax) and vmax > 0:
                vmin = -vmax
        else:
            try:
                vmin, vmax = np.nanpercentile(vals, [1.0, 99.0])
            except Exception:
                vmin = vmax = None
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmax <= vmin:
                vmin = vmax = None
    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _get_harmonic_profile_by_order(profiles, order):
    """Return q and A_n(q), falling back to the generic harmonics array.

    Older code stored only A2/A4 as direct keys, while the generalized harmonic
    calculation stores all requested orders in profiles["harmonics"].  This helper
    keeps A3/A5/A6 plotting from failing when those direct keys are absent.
    """
    order = int(order)
    q = np.asarray(profiles.get("q", np.array([])), dtype=float)
    direct = profiles.get(f"A{order}", None)
    if direct is not None:
        y = np.asarray(direct, dtype=float)
        if y.shape == q.shape:
            return q, y
    orders = np.asarray(profiles.get("harmonic_orders", []), dtype=int)
    harmonics = np.asarray(profiles.get("harmonics", np.empty((0, 0))), dtype=float)
    if orders.size and harmonics.ndim == 2:
        hit = np.where(orders == order)[0]
        if hit.size:
            y = np.asarray(harmonics[int(hit[0])], dtype=float)
            if q.size == y.size:
                return q, y
            if q.size == 0:
                return np.arange(y.size, dtype=float), y
    if q.size:
        return q, np.full(q.shape, np.nan, dtype=float)
    return np.array([], dtype=float), np.array([], dtype=float)


def _plot_selected_harmonic_profile(ax, exp_profiles, sim_profiles, metrics, order, *, show_phase_error=False, fontsize=8):
    """Plot a single harmonic profile A_n(q) for experimental vs simulation."""
    order = int(order)
    q_exp, y_exp = _get_harmonic_profile_by_order(exp_profiles, order)
    q_sim, y_sim = _get_harmonic_profile_by_order(sim_profiles, order)
    if q_exp.size and y_exp.size:
        ax.plot(q_exp, y_exp, label="experimental")
    if q_sim.size and y_sim.size:
        ax.plot(q_sim, y_sim, label="simulation")
    corr = _format_feature_metric(metrics.get(f"A{order}_anisotropy_corr"))
    title = f"A{order} | corr={corr}"
    if show_phase_error:
        title += f", phase err={_format_feature_metric(metrics.get(f'A{order}_phase_error_deg'))}°"
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("q (Å⁻¹)")
    ax.set_ylabel(f"A{order}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=fontsize)


def _plot_polar_match_summary(ax, polar_data, metrics, *, fontsize=8):
    shifts = np.asarray((polar_data or {}).get("shift_degs", np.array([])), dtype=float)
    corrs = np.asarray((polar_data or {}).get("corrs", np.array([])), dtype=float)
    finite = np.isfinite(shifts) & np.isfinite(corrs)
    if np.count_nonzero(finite) >= 2:
        x = np.where(shifts[finite] > 180.0, shifts[finite] - 360.0, shifts[finite])
        y = corrs[finite]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        ax.plot(x, y, label="polar q–φ corr")
        best_shift = _screening_safe_float(metrics.get("polar_xcorr_best_shift_deg"), np.nan)
        best_corr = _screening_safe_float(metrics.get("polar_xcorr_best_corr"), np.nan)
        if np.isfinite(best_shift) and np.isfinite(best_corr):
            ax.axvline(best_shift, linestyle=':', linewidth=1)
            ax.plot([best_shift], [best_corr], marker='o')
        ax.legend(loc="best", fontsize=fontsize)
    else:
        ax.text(0.5, 0.5, "Polar q–φ match unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(
        f"Polar q–φ | zero={_format_feature_metric(metrics.get('polar_corr_zero_shift'))}, "
        f"best={_format_feature_metric(metrics.get('polar_xcorr_best_corr'))} @ "
        f"{_format_feature_metric(metrics.get('polar_xcorr_best_shift_deg'))}°",
        fontsize=fontsize,
    )
    ax.set_xlabel("φ shift / deg")
    ax.set_ylabel("corr")
    ax.grid(True, alpha=0.25)

def _safe_corrcoef(a, b, mask=None):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    good = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        good &= np.asarray(mask).astype(bool)
    if np.count_nonzero(good) < 3:
        return float("nan")
    av = aa[good]
    bv = bb[good]
    av = av - np.nanmean(av)
    bv = bv - np.nanmean(bv)
    denom = float(np.sqrt(np.dot(av, av) * np.dot(bv, bv)))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(av, bv) / denom)
def _safe_cosine_similarity(a, b, mask=None):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    good = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        good &= np.asarray(mask).astype(bool)
    if np.count_nonzero(good) < 3:
        return float("nan")
    av = aa[good]
    bv = bb[good]
    denom = float(np.sqrt(np.dot(av, av) * np.dot(bv, bv)))
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float(np.dot(av, bv) / denom)
def _safe_nrmse(a, b, mask=None):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    good = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        good &= np.asarray(mask).astype(bool)
    if np.count_nonzero(good) < 3:
        return float("nan")
    diff = aa[good] - bb[good]
    rmse = float(np.sqrt(np.nanmean(diff ** 2)))
    denom = float(np.nanpercentile(aa[good], 95) - np.nanpercentile(aa[good], 5))
    if (not np.isfinite(denom)) or denom <= 0:
        denom = float(np.nanstd(aa[good]))
    if (not np.isfinite(denom)) or denom <= 0:
        denom = 1.0
    return rmse / denom

def _compute_radial_chi_square_metrics_from_integrated(exp_profile, sim_profile, sem_profile=None, counts_profile=None):
    """Compute reduced chi-square for raw radial profiles after weighted LS scaling.

    This mirrors the Auto Screen / Single-shot radial chi-square calculation, but
    is available to the interactive 2D-feature window as well.
    """
    exp = np.asarray(exp_profile, dtype=float)
    sim = np.asarray(sim_profile, dtype=float)
    if sem_profile is None:
        sem = np.full_like(exp, np.nan, dtype=float)
    else:
        sem = np.asarray(sem_profile, dtype=float)
        if sem.shape != exp.shape:
            sem = np.full_like(exp, np.nan, dtype=float)
    if counts_profile is None:
        counts = np.ones_like(exp, dtype=float)
    else:
        counts = np.asarray(counts_profile, dtype=float)
        if counts.shape != exp.shape:
            counts = np.ones_like(exp, dtype=float)
    good = np.isfinite(exp) & np.isfinite(sim) & (counts > 0)
    if np.count_nonzero(good) < 5:
        return {
            "radial_chi2_reduced": float("nan"),
            "radial_chi2_n_points": int(np.count_nonzero(good)),
            "radial_chi2_scale_factor": float("nan"),
            "radial_weighted_nrmse": float("nan"),
        }
    exp_good = exp[good]
    sim_good = sim[good]
    sem_good = sem[good]
    dynamic = float(np.nanpercentile(exp_good, 95) - np.nanpercentile(exp_good, 5))
    if (not np.isfinite(dynamic)) or dynamic <= 0:
        dynamic = float(np.nanstd(exp_good))
    if (not np.isfinite(dynamic)) or dynamic <= 0:
        dynamic = max(float(np.nanmax(np.abs(exp_good))), 1.0)
    sigma_floor = max(1.0e-12, 0.02 * dynamic)
    sigma = np.where(np.isfinite(sem_good) & (sem_good > 0), sem_good, sigma_floor)
    sigma = np.sqrt(sigma ** 2 + sigma_floor ** 2)
    w = 1.0 / np.maximum(sigma, 1.0e-12) ** 2
    denom = float(np.nansum(w * sim_good * sim_good))
    if (not np.isfinite(denom)) or denom <= 0:
        scale = 1.0
    else:
        scale = float(np.nansum(w * exp_good * sim_good) / denom)
    if (not np.isfinite(scale)) or scale <= 0:
        scale = 1.0
    residual = exp_good - scale * sim_good
    chi2 = float(np.nansum((residual / sigma) ** 2))
    dof = max(1, int(np.count_nonzero(good)) - 1)
    reduced = chi2 / dof
    rms = float(np.sqrt(np.nansum(w * residual ** 2) / np.nansum(w)))
    amp = float(np.nanpercentile(exp_good, 95) - np.nanpercentile(exp_good, 5))
    if (not np.isfinite(amp)) or amp <= 0:
        amp = max(float(np.nanstd(exp_good)), 1.0e-12)
    wnrmse = rms / amp
    return {
        "radial_chi2_reduced": float(reduced),
        "radial_chi2_n_points": int(np.count_nonzero(good)),
        "radial_chi2_scale_factor": float(scale),
        "radial_weighted_nrmse": float(wnrmse),
    }

def _compute_ssim_optional(exp_z, sim_z, mask=None):
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")
    a = np.asarray(exp_z, dtype=float).copy()
    b = np.asarray(sim_z, dtype=float).copy()
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        a[~m] = 0.0
        b[~m] = 0.0
    good = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(good) < 16:
        return float("nan")
    a[~np.isfinite(a)] = 0.0
    b[~np.isfinite(b)] = 0.0
    data_range = float(max(np.nanmax(a), np.nanmax(b)) - min(np.nanmin(a), np.nanmin(b)))
    if (not np.isfinite(data_range)) or data_range <= 0:
        data_range = 1.0
    try:
        return float(structural_similarity(a, b, data_range=data_range))
    except Exception:
        return float("nan")
def _weighted_nanmean(values, weights=None):
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return float("nan")
    good = np.isfinite(vals)
    if weights is None:
        if np.count_nonzero(good) == 0:
            return float("nan")
        return float(np.nanmean(vals[good]))
    w = np.asarray(weights, dtype=float)
    if w.shape != vals.shape:
        w = np.broadcast_to(w, vals.shape)
    good = good & np.isfinite(w) & (w > 0)
    if np.count_nonzero(good) == 0:
        return float("nan")
    return float(np.nansum(vals[good] * w[good]) / np.nansum(w[good]))
def _nfold_angle_diff_rad(a, b, n=2):
    """Smallest signed difference between two n-fold-periodic angles."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(int(n), 1)
    return (1.0 / n) * np.arctan2(np.sin(float(n) * (a - b)), np.cos(float(n) * (a - b)))
def _axial_angle_diff_rad(a, b):
    """Smallest signed difference between two 180°-periodic angles."""
    return _nfold_angle_diff_rad(a, b, n=2)
def _nfold_angle_to_deg(theta, n=2):
    try:
        v = float(theta)
    except Exception:
        return float("nan")
    if not np.isfinite(v):
        return float("nan")
    n = max(int(n), 1)
    period = 360.0 / float(n)
    half = 0.5 * period
    deg = float(np.degrees(v))
    return ((deg + half) % period) - half
def _axial_angle_to_deg(theta):
    return _nfold_angle_to_deg(theta, n=2)
def _weighted_nfold_mean_angle(theta, weights=None, n=2):
    theta = np.asarray(theta, dtype=float)
    good = np.isfinite(theta)
    if weights is None:
        w = np.ones(theta.shape, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != theta.shape:
            try:
                w = np.broadcast_to(w, theta.shape).astype(float)
            except Exception:
                w = np.ones(theta.shape, dtype=float)
    good &= np.isfinite(w) & (w > 0)
    if np.count_nonzero(good) == 0:
        return float("nan")
    n = max(int(n), 1)
    z = np.nansum(w[good] * np.exp((1j * float(n)) * theta[good]))
    if not np.isfinite(z.real) or not np.isfinite(z.imag) or abs(z) <= 0:
        return float("nan")
    return float((1.0 / n) * np.angle(z))
def _weighted_axial_mean_angle(theta, weights=None):
    return _weighted_nfold_mean_angle(theta, weights=weights, n=2)
def _get_harmonic_theta_by_order(profiles, order):
    order = int(order)
    if order == 2:
        theta2 = np.asarray(profiles.get("theta2", []), dtype=float)
        if theta2.size:
            return theta2
    orders = np.asarray(profiles.get("harmonic_orders", []), dtype=int)
    thetas = np.asarray(profiles.get("harmonic_thetas", np.empty((0, 0))), dtype=float)
    q = np.asarray(profiles.get("q", np.array([])), dtype=float)
    if orders.size and thetas.ndim == 2:
        hit = np.where(orders == order)[0]
        if hit.size:
            out = np.asarray(thetas[int(hit[0])], dtype=float)
            if q.size == 0 or out.size == q.size:
                return out
    return np.full(q.shape, np.nan, dtype=float) if q.size else np.array([], dtype=float)
def _compute_single_harmonic_phase_metrics(exp_profiles, sim_profiles, order):
    """Compare the phase/orientation of the complex A_n harmonic."""
    order = int(order)
    prefix = f"A{order}_phase"
    theta_exp = np.asarray(_get_harmonic_theta_by_order(exp_profiles, order), dtype=float)
    theta_sim = np.asarray(_get_harmonic_theta_by_order(sim_profiles, order), dtype=float)
    empty = {
        f"{prefix}_error_deg": float("nan"),
        f"{prefix}_qmean_abs_error_deg": float("nan"),
        f"{prefix}_global_error_deg": float("nan"),
        f"{prefix}_mean_exp_deg": float("nan"),
        f"{prefix}_mean_sim_deg": float("nan"),
        f"{prefix}_n_points": 0,
    }
    if theta_exp.shape != theta_sim.shape or theta_exp.size == 0:
        return empty
    _q_exp, amp_exp = _get_harmonic_profile_by_order(exp_profiles, order)
    _q_sim, amp_sim = _get_harmonic_profile_by_order(sim_profiles, order)
    amp_exp = np.asarray(amp_exp, dtype=float)
    amp_sim = np.asarray(amp_sim, dtype=float)
    if amp_exp.shape != theta_exp.shape:
        amp_exp = np.full(theta_exp.shape, np.nan, dtype=float)
    if amp_sim.shape != theta_sim.shape:
        amp_sim = np.full(theta_sim.shape, np.nan, dtype=float)
    cnt_exp = np.asarray(exp_profiles.get("harmonic_counts", np.zeros(theta_exp.shape, dtype=float)), dtype=float)
    cnt_sim = np.asarray(sim_profiles.get("harmonic_counts", np.zeros(theta_sim.shape, dtype=float)), dtype=float)
    if cnt_exp.shape != theta_exp.shape:
        cnt_exp = np.ones(theta_exp.shape, dtype=float)
    if cnt_sim.shape != theta_sim.shape:
        cnt_sim = np.ones(theta_sim.shape, dtype=float)
    amp_weight = np.sqrt(np.maximum(amp_exp, 0.0) * np.maximum(amp_sim, 0.0))
    count_weight = np.minimum(np.maximum(cnt_exp, 0.0), np.maximum(cnt_sim, 0.0))
    weights = count_weight * amp_weight
    good = np.isfinite(theta_exp) & np.isfinite(theta_sim) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(good) < 3:
        weights = count_weight
        good = np.isfinite(theta_exp) & np.isfinite(theta_sim) & np.isfinite(weights) & (weights > 0)
    if np.count_nonzero(good) == 0:
        return empty
    diff = _nfold_angle_diff_rad(theta_exp[good], theta_sim[good], n=order)
    abs_diff_deg = np.abs(np.degrees(diff))
    qmean_abs = float(np.nansum(abs_diff_deg * weights[good]) / np.nansum(weights[good]))
    mean_exp = _weighted_nfold_mean_angle(theta_exp[good], weights[good], n=order)
    mean_sim = _weighted_nfold_mean_angle(theta_sim[good], weights[good], n=order)
    if np.isfinite(mean_exp) and np.isfinite(mean_sim):
        global_err = float(abs(np.degrees(_nfold_angle_diff_rad(mean_exp, mean_sim, n=order))))
    else:
        global_err = float("nan")
    return {
        f"{prefix}_error_deg": qmean_abs,
        f"{prefix}_qmean_abs_error_deg": qmean_abs,
        f"{prefix}_global_error_deg": global_err,
        f"{prefix}_mean_exp_deg": _nfold_angle_to_deg(mean_exp, n=order),
        f"{prefix}_mean_sim_deg": _nfold_angle_to_deg(mean_sim, n=order),
        f"{prefix}_n_points": int(np.count_nonzero(good)),
    }
def _compute_a2_phase_metrics(exp_profiles, sim_profiles):
    """Backward-compatible A2 wrapper around the generalized harmonic phase metric."""
    return _compute_single_harmonic_phase_metrics(exp_profiles, sim_profiles, 2)
def _compute_selected_harmonic_phase_metrics(exp_profiles, sim_profiles, orders=None):
    if orders is None:
        orders = (2, 3, 4, 5, 6)
    out = {}
    for n in orders:
        try:
            out.update(_compute_single_harmonic_phase_metrics(exp_profiles, sim_profiles, int(n)))
        except Exception:
            pass
    return out
def _angular_profile_cross_correlation_metrics(exp_pos, sim_pos, q, phi, valid_mask=None, q_min=None, q_max=None, n_phi_bins=180):
    """Compare azimuthal intensity profiles in the harmonic q-window.
    zero-shift correlation is orientation-sensitive. best-shift correlation and
    shift quantify how much the simulation would need to rotate to align with the
    experimental angular profile.
    """
    exp = np.asarray(exp_pos, dtype=float)
    sim = np.asarray(sim_pos, dtype=float)
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mask = _valid_feature_mask(exp, valid_mask) & _valid_feature_mask(sim, valid_mask) & np.isfinite(q) & np.isfinite(phi)
    if q_min is not None:
        try:
            qmin = float(q_min)
            if np.isfinite(qmin):
                mask &= q >= qmin
        except Exception:
            pass
    if q_max is not None:
        try:
            qmax = float(q_max)
            if np.isfinite(qmax):
                mask &= q <= qmax
        except Exception:
            pass
    n_phi_bins = int(max(24, n_phi_bins))
    out = {
        "angular_profile_corr_zero_shift": float("nan"),
        "angular_xcorr_best_corr": float("nan"),
        "angular_xcorr_best_shift_deg": float("nan"),
        "angular_xcorr_best_shift_abs_deg_axial": float("nan"),
        "angular_xcorr_n_phi_bins": int(n_phi_bins),
        "angular_xcorr_valid_bins": 0,
    }
    if np.count_nonzero(mask) < max(32, n_phi_bins // 2):
        return out
    phi_norm = (phi[mask] + np.pi) % (2.0 * np.pi)
    edges = np.linspace(0.0, 2.0 * np.pi, n_phi_bins + 1)
    which = np.clip(np.digitize(phi_norm, edges) - 1, 0, n_phi_bins - 1)
    def _binned_mean(vals):
        vals = np.asarray(vals, dtype=float)
        sums = np.bincount(which, weights=vals, minlength=n_phi_bins).astype(float)
        counts = np.bincount(which, minlength=n_phi_bins).astype(float)
        prof = np.full(n_phi_bins, np.nan, dtype=float)
        good_bins = counts > 0
        prof[good_bins] = sums[good_bins] / counts[good_bins]
        return prof, counts
    pe, ce = _binned_mean(np.maximum(exp[mask], 0.0))
    ps, cs = _binned_mean(np.maximum(sim[mask], 0.0))
    good = np.isfinite(pe) & np.isfinite(ps) & (ce > 0) & (cs > 0)
    out["angular_xcorr_valid_bins"] = int(np.count_nonzero(good))
    if np.count_nonzero(good) < max(12, n_phi_bins // 6):
        return out
    # Fill missing angular sectors with their own mean before z-normalization so
    # mask holes do not create artificial extrema in the circular correlation.
    pe_fill = pe.copy()
    ps_fill = ps.copy()
    pe_mean = float(np.nanmean(pe[good]))
    ps_mean = float(np.nanmean(ps[good]))
    pe_fill[~np.isfinite(pe_fill)] = pe_mean
    ps_fill[~np.isfinite(ps_fill)] = ps_mean
    ae = pe_fill - float(np.nanmean(pe_fill))
    bs = ps_fill - float(np.nanmean(ps_fill))
    ae_std = float(np.sqrt(np.nanmean(ae ** 2)))
    bs_std = float(np.sqrt(np.nanmean(bs ** 2)))
    if not np.isfinite(ae_std) or not np.isfinite(bs_std) or ae_std <= 0 or bs_std <= 0:
        return out
    ae = ae / ae_std
    bs = bs / bs_std
    zero_corr = float(np.nanmean(ae * bs))
    corrs = np.array([float(np.nanmean(ae * np.roll(bs, shift))) for shift in range(n_phi_bins)], dtype=float)
    if not np.any(np.isfinite(corrs)):
        return out
    best_idx = int(np.nanargmax(corrs))
    best_corr = float(corrs[best_idx])
    # np.roll(bs, shift) is the profile used for comparison, so this is the
    # circular shift applied to the simulation profile in display/bin units.
    if best_idx > n_phi_bins // 2:
        signed_shift_bins = best_idx - n_phi_bins
    else:
        signed_shift_bins = best_idx
    shift_deg = float(signed_shift_bins * 360.0 / n_phi_bins)
    axial_abs = abs(((shift_deg + 90.0) % 180.0) - 90.0)
    out.update({
        "angular_profile_corr_zero_shift": zero_corr,
        "angular_xcorr_best_corr": best_corr,
        "angular_xcorr_best_shift_deg": shift_deg,
        "angular_xcorr_best_shift_abs_deg_axial": float(axial_abs),
    })
    return out
def _fill_nan_rows_with_rowmean(arr):
    data = np.asarray(arr, dtype=float).copy()
    if data.ndim != 2:
        return data
    finite_any = np.isfinite(data)
    global_mean = float(np.nanmean(data[finite_any])) if np.any(finite_any) else 0.0
    for i in range(data.shape[0]):
        row = data[i]
        good = np.isfinite(row)
        fill = float(np.nanmean(row[good])) if np.any(good) else global_mean
        data[i, ~good] = fill
    return data

def _polar_bin_image(image, q, phi, valid_mask=None, q_min=None, q_max=None, n_q_bins=84, n_phi_bins=180):
    img = np.asarray(image, dtype=float)
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mask = np.isfinite(img) & np.isfinite(q) & np.isfinite(phi)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape == img.shape:
            mask &= vm
    try:
        q_min = float(q_min) if q_min is not None else None
    except Exception:
        q_min = None
    try:
        q_max = float(q_max) if q_max is not None else None
    except Exception:
        q_max = None
    if q_min is not None and np.isfinite(q_min):
        mask &= q >= q_min
    else:
        q_min = None
    if q_max is not None and np.isfinite(q_max):
        mask &= q <= q_max
    else:
        q_max = None
    n_q_bins = int(max(16, n_q_bins))
    n_phi_bins = int(max(36, n_phi_bins))
    empty = {
        'image': np.full((n_q_bins, n_phi_bins), np.nan, dtype=float),
        'counts': np.zeros((n_q_bins, n_phi_bins), dtype=int),
        'q_centers': np.linspace(0.0, 1.0, n_q_bins, dtype=float),
        'phi_centers_deg': np.linspace(-180.0, 180.0, n_phi_bins, endpoint=False, dtype=float),
        'q_min': float('nan'),
        'q_max': float('nan'),
    }
    if np.count_nonzero(mask) < 16:
        return empty
    qv = q[mask]
    pv = np.mod(phi[mask] + np.pi, 2.0 * np.pi)
    iv = img[mask]
    q_lo = float(np.nanmin(qv)) if q_min is None else float(q_min)
    q_hi = float(np.nanmax(qv)) if q_max is None else float(q_max)
    if (not np.isfinite(q_lo)) or (not np.isfinite(q_hi)) or q_hi <= q_lo:
        return empty
    q_edges = np.linspace(q_lo, q_hi, n_q_bins + 1)
    p_edges = np.linspace(0.0, 2.0 * np.pi, n_phi_bins + 1)
    qi = np.digitize(qv, q_edges) - 1
    pi = np.digitize(pv, p_edges) - 1
    keep = (qi >= 0) & (qi < n_q_bins) & (pi >= 0) & (pi < n_phi_bins) & np.isfinite(iv)
    sums = np.zeros((n_q_bins, n_phi_bins), dtype=float)
    counts = np.zeros((n_q_bins, n_phi_bins), dtype=int)
    np.add.at(sums, (qi[keep], pi[keep]), iv[keep])
    np.add.at(counts, (qi[keep], pi[keep]), 1)
    out = np.full((n_q_bins, n_phi_bins), np.nan, dtype=float)
    good = counts > 0
    out[good] = sums[good] / counts[good]
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    phi_centers_deg = np.degrees(0.5 * (p_edges[:-1] + p_edges[1:]) - np.pi)
    return {
        'image': out,
        'counts': counts,
        'q_centers': q_centers,
        'phi_centers_deg': phi_centers_deg,
        'q_min': float(q_lo),
        'q_max': float(q_hi),
    }

def _polar_image_matching_metrics(exp_img, sim_img, q, phi, valid_mask=None, q_min=None, q_max=None, n_q_bins=84, n_phi_bins=180):
    exp_pol = _polar_bin_image(exp_img, q, phi, valid_mask=valid_mask, q_min=q_min, q_max=q_max, n_q_bins=n_q_bins, n_phi_bins=n_phi_bins)
    sim_pol = _polar_bin_image(sim_img, q, phi, valid_mask=valid_mask, q_min=q_min, q_max=q_max, n_q_bins=n_q_bins, n_phi_bins=n_phi_bins)
    exp_mat = np.asarray(exp_pol['image'], dtype=float)
    sim_mat = np.asarray(sim_pol['image'], dtype=float)
    good = np.isfinite(exp_mat) & np.isfinite(sim_mat)
    out = {
        'polar_corr_zero_shift': float('nan'),
        'polar_nrmse_zero_shift': float('nan'),
        'polar_xcorr_best_corr': float('nan'),
        'polar_xcorr_best_shift_deg': float('nan'),
        'polar_nrmse_best_shift': float('nan'),
        'polar_n_q_bins': int(exp_mat.shape[0]) if exp_mat.ndim == 2 else 0,
        'polar_n_phi_bins': int(exp_mat.shape[1]) if exp_mat.ndim == 2 else 0,
        'polar_valid_bins': int(np.count_nonzero(good)),
    }
    plot = {
        'exp_polar': exp_mat,
        'sim_polar': sim_mat,
        'q_centers': np.asarray(exp_pol['q_centers'], dtype=float),
        'phi_centers_deg': np.asarray(exp_pol['phi_centers_deg'], dtype=float),
        'shift_degs': np.array([], dtype=float),
        'corrs': np.array([], dtype=float),
        'best_shift_deg': float('nan'),
        'best_corr': float('nan'),
        'best_sim_polar': sim_mat,
    }
    if np.count_nonzero(good) < 16 or exp_mat.ndim != 2 or sim_mat.ndim != 2:
        return out, plot
    # Use the same filled polar matrices for the zero-shift value and the
    # rotation-scan values.  The previous implementation used the sparse native
    # mask for zero-shift but row-mean-filled matrices for the shift scan, so
    # zero could appear larger than the reported best value even though the scan
    # includes shift = 0.  With one common basis, best_corr is guaranteed to be
    # at least the zero-shift correlation up to numerical precision.
    exp_fill = _fill_nan_rows_with_rowmean(exp_mat)
    sim_fill = _fill_nan_rows_with_rowmean(sim_mat)
    gm0 = np.isfinite(exp_fill) & np.isfinite(sim_fill)
    out['polar_corr_zero_shift'] = _safe_corrcoef(exp_fill, sim_fill, gm0)
    out['polar_nrmse_zero_shift'] = _safe_nrmse(exp_fill, sim_fill, gm0)
    n_phi = exp_fill.shape[1]
    corrs = np.full(n_phi, np.nan, dtype=float)
    nrmses = np.full(n_phi, np.nan, dtype=float)
    best_sim = sim_fill.copy()
    for shift in range(n_phi):
        rolled = np.roll(sim_fill, shift, axis=1)
        gm = np.isfinite(exp_fill) & np.isfinite(rolled)
        if np.count_nonzero(gm) < 16:
            continue
        corrs[shift] = _safe_corrcoef(exp_fill, rolled, gm)
        nrmses[shift] = _safe_nrmse(exp_fill, rolled, gm)
    if np.any(np.isfinite(corrs)):
        best_idx = int(np.nanargmax(corrs))
        signed_shift_bins = best_idx - n_phi if best_idx > n_phi // 2 else best_idx
        shift_deg = float(signed_shift_bins * 360.0 / n_phi)
        out['polar_xcorr_best_corr'] = float(corrs[best_idx])
        out['polar_xcorr_best_shift_deg'] = shift_deg
        out['polar_nrmse_best_shift'] = float(nrmses[best_idx]) if np.isfinite(nrmses[best_idx]) else float('nan')
        best_sim = np.roll(sim_fill, best_idx, axis=1)
    plot.update({
        'shift_degs': np.arange(n_phi, dtype=float) * 360.0 / max(n_phi, 1),
        'corrs': corrs,
        'best_shift_deg': out['polar_xcorr_best_shift_deg'],
        'best_corr': out['polar_xcorr_best_corr'],
        'best_sim_polar': best_sim,
    })
    return out, plot

def _radial_angular_feature_profiles(image_pos, q, phi, valid_mask=None, n_bins=500, harmonic_orders=None, q_min=None, q_max=None, harmonic_q_min=None, harmonic_q_max=None):
    """Compute radial intensity and angular Fourier anisotropy profiles.
    This generalizes the earlier A2/A4-only implementation to an arbitrary set of
    harmonic orders ``A_n(q)`` where
        A_n(q) = |Σ I(q, φ) exp(i n φ)| / Σ I(q, φ)
    The defaults retain A2/A4 while also adding higher-order harmonics that are
    useful for cone/double-cone and other non-ellipsoidal SAXS patterns.
    """
    if harmonic_orders is None:
        harmonic_orders = tuple(range(1, 13))
    orders = np.array(sorted({int(n) for n in harmonic_orders if int(n) >= 1}), dtype=int)
    if orders.size == 0:
        orders = np.array([2, 4], dtype=int)
    img = np.asarray(image_pos, dtype=float)
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mask = _valid_feature_mask(img, valid_mask) & np.isfinite(q) & np.isfinite(phi)
    # q_min/q_max are treated as the evaluation/display window, not as binning limits.
    # This keeps q-bin meaning consistent with full-range 1D integration and then crops
    # metrics/plots by setting values outside the window to NaN.
    feature_q_min_use = None
    feature_q_max_use = None
    if q_min is not None:
        try:
            feature_q_min_use = float(q_min)
        except Exception:
            feature_q_min_use = None
    if q_max is not None:
        try:
            feature_q_max_use = float(q_max)
        except Exception:
            feature_q_max_use = None
    if feature_q_min_use is not None and not np.isfinite(feature_q_min_use):
        feature_q_min_use = None
    if feature_q_max_use is not None and not np.isfinite(feature_q_max_use):
        feature_q_max_use = None
    if feature_q_min_use is not None and feature_q_max_use is not None and feature_q_max_use <= feature_q_min_use:
        feature_q_min_use = None
        feature_q_max_use = None
    harmonic_q_min_use = None
    harmonic_q_max_use = None
    if harmonic_q_min is not None:
        try:
            harmonic_q_min_use = float(harmonic_q_min)
        except Exception:
            harmonic_q_min_use = None
    if harmonic_q_max is not None:
        try:
            harmonic_q_max_use = float(harmonic_q_max)
        except Exception:
            harmonic_q_max_use = None
    if harmonic_q_min_use is not None and not np.isfinite(harmonic_q_min_use):
        harmonic_q_min_use = None
    if harmonic_q_max_use is not None and not np.isfinite(harmonic_q_max_use):
        harmonic_q_max_use = None
    if harmonic_q_min_use is not None and harmonic_q_max_use is not None and harmonic_q_max_use <= harmonic_q_min_use:
        harmonic_q_min_use = None
        harmonic_q_max_use = None
    if np.count_nonzero(mask) < 16:
        empty = np.full(int(n_bins), np.nan, dtype=float)
        empty_h = np.full((orders.size, int(n_bins)), np.nan, dtype=float)
        empty_theta = np.full((orders.size, int(n_bins)), np.nan, dtype=float)
        return {
            "q": empty.copy(),
            "radial": empty.copy(),
            "A2": empty.copy(),
            "A4": empty.copy(),
            "theta2": empty.copy(),
            "counts": np.zeros(int(n_bins), dtype=int),
            "harmonic_counts": np.zeros(int(n_bins), dtype=int),
            "harmonic_orders": orders,
            "harmonics": empty_h,
            "harmonic_thetas": empty_theta,
            "harmonic_qavg": np.full(orders.size, np.nan, dtype=float),
            "harmonic_q_min": float(harmonic_q_min_use) if harmonic_q_min_use is not None else float("nan"),
            "harmonic_q_max": float(harmonic_q_max_use) if harmonic_q_max_use is not None else float("nan"),
        }
    qv = q[mask]
    # Full-range bin edges preserve the same q-bin interpretation across 1D,
    # 2D feature, Auto Screen, and Single-shot outputs.
    q_lo, q_hi = float(np.nanmin(qv)), float(np.nanmax(qv))
    if (not np.isfinite(q_lo)) or (not np.isfinite(q_hi)) or q_hi <= q_lo:
        q_lo, q_hi = np.nanpercentile(qv, [0.5, 99.5])
    n_bins = int(max(n_bins, 8))
    edges = np.linspace(q_lo, q_hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial = np.full(n_bins, np.nan, dtype=float)
    theta2 = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    harmonic_counts = np.zeros(n_bins, dtype=int)
    harmonics = np.full((orders.size, n_bins), np.nan, dtype=float)
    harmonic_thetas = np.full((orders.size, n_bins), np.nan, dtype=float)
    im = img[mask]
    ph = phi[mask]
    which = np.digitize(qv, edges) - 1
    order_to_index = {int(n): i for i, n in enumerate(orders)}
    for idx in range(n_bins):
        sel = which == idx
        counts[idx] = int(np.count_nonzero(sel))
        if counts[idx] < 8:
            continue
        radial[idx] = float(np.nanmean(im[sel]))
        # Harmonic anisotropy is evaluated only within the dedicated q window.
        # Radial/pixel features still use the broader feature q range above.
        hsel = sel.copy()
        if harmonic_q_min_use is not None:
            hsel &= qv >= harmonic_q_min_use
        if harmonic_q_max_use is not None:
            hsel &= qv <= harmonic_q_max_use
        harmonic_counts[idx] = int(np.count_nonzero(hsel))
        if harmonic_counts[idx] < 8:
            continue
        weights = np.maximum(im[hsel], 0.0)
        wsum = float(np.nansum(weights))
        if wsum <= 0 or not np.isfinite(wsum):
            continue
        ph_sel = ph[hsel]
        for j, n in enumerate(orders):
            cn = np.nansum(weights * np.exp((1j * int(n)) * ph_sel)) / wsum
            harmonics[j, idx] = float(np.abs(cn))
            harmonic_thetas[j, idx] = float((1.0 / float(int(n))) * np.angle(cn))
            if int(n) == 2:
                theta2[idx] = harmonic_thetas[j, idx]
    feature_bin_mask = np.isfinite(centers)
    if feature_q_min_use is not None:
        feature_bin_mask &= centers >= feature_q_min_use
    if feature_q_max_use is not None:
        feature_bin_mask &= centers <= feature_q_max_use
    radial_eval = radial.copy()
    radial_eval[~feature_bin_mask] = np.nan

    qavg = np.array([_weighted_nanmean(harmonics[j], harmonic_counts) for j in range(orders.size)], dtype=float)
    A2 = harmonics[order_to_index[2]].copy() if 2 in order_to_index else np.full(n_bins, np.nan, dtype=float)
    A4 = harmonics[order_to_index[4]].copy() if 4 in order_to_index else np.full(n_bins, np.nan, dtype=float)
    return {
        "q": centers,
        "radial": radial_eval,
        "radial_full": radial,
        "A2": A2,
        "A4": A4,
        "theta2": theta2,
        "counts": counts,
        "harmonic_counts": harmonic_counts,
        "harmonic_orders": orders,
        "harmonics": harmonics,
        "harmonic_thetas": harmonic_thetas,
        "harmonic_qavg": qavg,
        "feature_q_min": float(feature_q_min_use) if feature_q_min_use is not None else float("nan"),
        "feature_q_max": float(feature_q_max_use) if feature_q_max_use is not None else float("nan"),
        "binning_q_min": float(q_lo),
        "binning_q_max": float(q_hi),
        "binning_mode": "full_range_then_crop",
        "harmonic_q_min": float(harmonic_q_min_use) if harmonic_q_min_use is not None else float("nan"),
        "harmonic_q_max": float(harmonic_q_max_use) if harmonic_q_max_use is not None else float("nan"),
    }
def _global_2d_moment_features(image_pos, q, phi, valid_mask=None):
    img = np.asarray(image_pos, dtype=float)
    q = np.asarray(q, dtype=float)
    phi = np.asarray(phi, dtype=float)
    mask = _valid_feature_mask(img, valid_mask) & np.isfinite(q) & np.isfinite(phi)
    yy, xx = np.indices(img.shape)
    cy = (img.shape[0] - 1) / 2.0
    cx = (img.shape[1] - 1) / 2.0
    out = {
        "centroid_x_px": float("nan"),
        "centroid_y_px": float("nan"),
        "eccentricity": float("nan"),
        "orientation_rad": float("nan"),
        "mean_q": float("nan"),
        "std_q": float("nan"),
        "global_A2": float("nan"),
        "global_A4": float("nan"),
        "sparsity_cv": float("nan"),
        "entropy": float("nan"),
    }
    if np.count_nonzero(mask) < 16:
        return out
    w = np.maximum(img[mask], 0.0)
    wsum = float(np.nansum(w))
    if wsum <= 0 or not np.isfinite(wsum):
        return out
    x = xx[mask] - cx
    y = yy[mask] - cy
    mx = float(np.nansum(w * x) / wsum)
    my = float(np.nansum(w * y) / wsum)
    x0 = x - mx
    y0 = y - my
    cxx = float(np.nansum(w * x0 * x0) / wsum)
    cyy = float(np.nansum(w * y0 * y0) / wsum)
    cxy = float(np.nansum(w * x0 * y0) / wsum)
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)
    try:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vec = vecs[:, order[0]]
        if vals[0] > 0:
            ecc = float(np.sqrt(max(0.0, 1.0 - vals[-1] / vals[0])))
        else:
            ecc = 0.0
        orient = float(np.arctan2(vec[1], vec[0]))
    except Exception:
        ecc = float("nan")
        orient = float("nan")
    ph = phi[mask]
    c2 = np.nansum(w * np.exp(2j * ph)) / wsum
    c4 = np.nansum(w * np.exp(4j * ph)) / wsum
    p = w / wsum
    p = p[p > 0]
    entropy = float(-np.nansum(p * np.log(p)) / np.log(max(p.size, 2))) if p.size else float("nan")
    mean_int = float(np.nanmean(img[mask]))
    std_int = float(np.nanstd(img[mask]))
    out.update({
        "centroid_x_px": mx,
        "centroid_y_px": my,
        "eccentricity": ecc,
        "orientation_rad": orient,
        "mean_q": float(np.nansum(w * q[mask]) / wsum),
        "std_q": float(np.sqrt(max(0.0, np.nansum(w * (q[mask] - np.nansum(w * q[mask]) / wsum) ** 2) / wsum))),
        "global_A2": float(np.abs(c2)),
        "global_A4": float(np.abs(c4)),
        "sparsity_cv": float(std_int / mean_int) if mean_int > 0 else float("nan"),
        "entropy": entropy,
    })
    return out
def _compare_2d_saxs_features(exp_img, sim_img, q_u=None, q_v=None, valid_mask=None, n_bins=500, harmonic_orders=None, q_min=DEFAULT_1D_SCALE_FIT_Q_MIN, q_max=DEFAULT_1D_SCALE_FIT_Q_MAX, harmonic_q_min=DEFAULT_HARMONIC_Q_MIN, harmonic_q_max=DEFAULT_HARMONIC_Q_MAX):
    """Compute pixel-level and SAXS-specific 2D feature comparison metrics."""
    exp = np.asarray(exp_img, dtype=float)
    sim = np.asarray(sim_img, dtype=float)
    if exp.shape != sim.shape:
        raise ValueError(f"Image shapes differ: experimental={exp.shape}, simulation={sim.shape}")
    if valid_mask is None:
        valid_mask = np.ones_like(exp, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask).astype(bool)
        if valid_mask.shape != exp.shape:
            valid_mask = np.ones_like(exp, dtype=bool)
    q, phi = _make_q_phi_grid(exp.shape, q_u=q_u, q_v=q_v)
    q_range_mask = np.isfinite(q)
    q_min_use = None
    q_max_use = None
    if q_min is not None:
        try:
            q_min_use = float(q_min)
        except Exception:
            q_min_use = None
    if q_max is not None:
        try:
            q_max_use = float(q_max)
        except Exception:
            q_max_use = None
    harmonic_q_min_use = None
    harmonic_q_max_use = None
    if harmonic_q_min is not None:
        try:
            harmonic_q_min_use = float(harmonic_q_min)
        except Exception:
            harmonic_q_min_use = None
    if harmonic_q_max is not None:
        try:
            harmonic_q_max_use = float(harmonic_q_max)
        except Exception:
            harmonic_q_max_use = None
    if harmonic_q_min_use is not None and not np.isfinite(harmonic_q_min_use):
        harmonic_q_min_use = None
    if harmonic_q_max_use is not None and not np.isfinite(harmonic_q_max_use):
        harmonic_q_max_use = None
    if harmonic_q_min_use is not None and harmonic_q_max_use is not None and harmonic_q_max_use <= harmonic_q_min_use:
        harmonic_q_min_use = None
        harmonic_q_max_use = None
    if q_min_use is not None and np.isfinite(q_min_use):
        q_range_mask &= q >= q_min_use
    if q_max_use is not None and np.isfinite(q_max_use):
        q_range_mask &= q <= q_max_use
    good = _valid_feature_mask(exp, valid_mask) & _valid_feature_mask(sim, valid_mask) & q_range_mask
    if np.count_nonzero(good) < 16:
        raise ValueError("Not enough valid overlapping pixels for 2D feature comparison in the selected q range.")
    sim_scale = _compute_overlay_scale(exp[good], sim[good])
    sim_scaled = sim * sim_scale
    # Signed/asinh images preserve 2D ring oscillations for display and pixelwise comparison.
    exp_disp, mask_exp_disp, exp_disp_meta = _signed_asinh_display_image(exp, valid_mask=good, zscore=False)
    sim_disp, mask_sim_disp, sim_disp_meta = _signed_asinh_display_image(sim_scaled, valid_mask=good, zscore=False)
    exp_z, mask_exp, exp_meta = _signed_asinh_display_image(exp, valid_mask=good, zscore=True)
    sim_z, mask_sim, sim_meta = _signed_asinh_display_image(sim_scaled, valid_mask=good, zscore=True)
    common = good & mask_exp & mask_sim & mask_exp_disp & mask_sim_disp

    # Pixelwise/2D feature comparisons intentionally stay inside the selected q-window.
    # Radial and harmonic profiles, however, should keep the same q-bin meaning as
    # ordinary 1D integration: bin over the full valid detector q range first, then
    # evaluate/crop the resulting profiles over the requested q window.
    radial_common = (
        _valid_feature_mask(exp, valid_mask)
        & _valid_feature_mask(sim_scaled, valid_mask)
        & np.isfinite(q)
        & np.isfinite(phi)
    )
    if np.count_nonzero(radial_common) < 16:
        radial_common = common.copy()

    # Separate morphology-oriented positive image kept for anisotropy/global-shape features.
    # exp_pos/sim_pos are q-windowed for 2D morphology, while exp_pos_profile/sim_pos_profile
    # are full-range for consistent q-bin radial/harmonic profiles.
    exp_pos, _, exp_pos_meta = _robust_positive_feature_image(exp, valid_mask=common)
    sim_pos, _, sim_pos_meta = _robust_positive_feature_image(sim_scaled, valid_mask=common)
    exp_pos_profile, _, _exp_pos_profile_meta = _robust_positive_feature_image(exp, valid_mask=radial_common)
    sim_pos_profile, _, _sim_pos_profile_meta = _robust_positive_feature_image(sim_scaled, valid_mask=radial_common)
    if harmonic_orders is None:
        harmonic_orders = tuple(range(1, 13))
    exp_profiles = _radial_angular_feature_profiles(
        exp_pos_profile, q, phi, valid_mask=radial_common, n_bins=n_bins, harmonic_orders=harmonic_orders,
        q_min=q_min_use, q_max=q_max_use, harmonic_q_min=harmonic_q_min_use, harmonic_q_max=harmonic_q_max_use,
    )
    sim_profiles = _radial_angular_feature_profiles(
        sim_pos_profile, q, phi, valid_mask=radial_common, n_bins=n_bins, harmonic_orders=harmonic_orders,
        q_min=q_min_use, q_max=q_max_use, harmonic_q_min=harmonic_q_min_use, harmonic_q_max=harmonic_q_max_use,
    )
    # Preserve oscillatory radial information using the raw images as well.
    # Important: no q_min/q_max is passed here. Full-range binning is performed first;
    # radial metrics below then use a q-window mask on the already-binned curve.
    radial_chi_metrics = {
        "radial_chi2_reduced": float("nan"),
        "radial_chi2_n_points": 0,
        "radial_chi2_scale_factor": float("nan"),
        "radial_weighted_nrmse": float("nan"),
    }
    try:
        q_rad_exp, I_rad_exp, counts_rad_exp, _q_label_rad, _std_rad_exp, sem_rad_exp = _radial_integrate_on_q_grid(
            exp, q_u=q_u, q_v=q_v, valid_mask=radial_common, npt=n_bins
        )
        q_rad_sim, I_rad_sim_unscaled, _counts_rad_sim, _q_label_rad2, _std_rad_sim, _sem_rad_sim = _radial_integrate_on_q_grid(
            sim, q_u=q_u, q_v=q_v, valid_mask=radial_common, npt=n_bins
        )
        radial_chi_metrics = _compute_radial_chi_square_metrics_from_integrated(
            I_rad_exp, I_rad_sim_unscaled, sem_rad_exp, counts_rad_exp
        )
        I_rad_sim = np.asarray(I_rad_sim_unscaled, dtype=float) * float(sim_scale)
        exp_profiles["q_raw"] = q_rad_exp
        exp_profiles["radial_raw_full"] = I_rad_exp
        sim_profiles["q_raw"] = q_rad_sim
        sim_profiles["radial_raw_full"] = I_rad_sim
        radial_metric_mask = np.isfinite(q_rad_exp)
        if q_min_use is not None and np.isfinite(q_min_use):
            radial_metric_mask &= q_rad_exp >= q_min_use
        if q_max_use is not None and np.isfinite(q_max_use):
            radial_metric_mask &= q_rad_exp <= q_max_use
        I_rad_exp_eval = np.asarray(I_rad_exp, dtype=float).copy()
        I_rad_sim_eval = np.asarray(I_rad_sim, dtype=float).copy()
        I_rad_exp_eval[~radial_metric_mask] = np.nan
        I_rad_sim_eval[~radial_metric_mask] = np.nan
        exp_profiles["radial_raw"] = I_rad_exp_eval
        sim_profiles["radial_raw"] = I_rad_sim_eval
        exp_profiles["radial_raw_binning_mode"] = "full_range_then_crop"
        sim_profiles["radial_raw_binning_mode"] = "full_range_then_crop"
    except Exception:
        exp_profiles["q_raw"] = exp_profiles["q"]
        exp_profiles["radial_raw"] = exp_profiles["radial"]
        exp_profiles["radial_raw_full"] = exp_profiles.get("radial_full", exp_profiles["radial"])
        sim_profiles["q_raw"] = sim_profiles["q"]
        sim_profiles["radial_raw"] = sim_profiles["radial"]
        sim_profiles["radial_raw_full"] = sim_profiles.get("radial_full", sim_profiles["radial"])
    exp_global = _global_2d_moment_features(exp_pos, q, phi, valid_mask=common)
    sim_global = _global_2d_moment_features(sim_pos, q, phi, valid_mask=common)
    metrics = {
        "valid_pixels": int(np.count_nonzero(common)),
        "feature_q_min": float(q_min_use) if q_min_use is not None and np.isfinite(q_min_use) else float("nan"),
        "feature_q_max": float(q_max_use) if q_max_use is not None and np.isfinite(q_max_use) else float("nan"),
        "harmonic_q_min": float(harmonic_q_min_use) if harmonic_q_min_use is not None and np.isfinite(harmonic_q_min_use) else float("nan"),
        "harmonic_q_max": float(harmonic_q_max_use) if harmonic_q_max_use is not None and np.isfinite(harmonic_q_max_use) else float("nan"),
        "simulation_scale_factor_raw_pixel_LS": float(sim_scale),
        "pixel_pearson_corr_asinh_z": _safe_corrcoef(exp_z, sim_z, common),
        "pixel_cosine_similarity_asinh_z": _safe_cosine_similarity(exp_z, sim_z, common),
        "pixel_nrmse_asinh_z": _safe_nrmse(exp_z, sim_z, common),
        "pixel_ssim_asinh_z": _compute_ssim_optional(exp_z, sim_z, common),
        "radial_profile_corr": _safe_corrcoef(
            exp_profiles.get("radial_raw", exp_profiles["radial"]),
            sim_profiles.get("radial_raw", sim_profiles["radial"]),
        ),
        "radial_profile_nrmse": _safe_nrmse(
            exp_profiles.get("radial_raw", exp_profiles["radial"]),
            sim_profiles.get("radial_raw", sim_profiles["radial"]),
        ),
        "radial_profile_binning_mode": "full_range_then_crop",
        "radial_chi2_reduced": radial_chi_metrics.get("radial_chi2_reduced", float("nan")),
        "radial_chi2_n_points": radial_chi_metrics.get("radial_chi2_n_points", 0),
        "radial_chi2_scale_factor": radial_chi_metrics.get("radial_chi2_scale_factor", float("nan")),
        "radial_weighted_nrmse": radial_chi_metrics.get("radial_weighted_nrmse", float("nan")),
        "A2_anisotropy_corr": _safe_corrcoef(exp_profiles["A2"], sim_profiles["A2"]),
        "A2_anisotropy_nrmse": _safe_nrmse(exp_profiles["A2"], sim_profiles["A2"]),
        "A4_anisotropy_corr": _safe_corrcoef(exp_profiles["A4"], sim_profiles["A4"]),
        "A4_anisotropy_nrmse": _safe_nrmse(exp_profiles["A4"], sim_profiles["A4"]),
        "anisotropy_profile_binning_mode": "full_range_then_crop",
    }
    metrics.update(_compute_selected_harmonic_phase_metrics(exp_profiles, sim_profiles, orders=(2, 3, 4, 5, 6)))
    metrics.update(_angular_profile_cross_correlation_metrics(
        exp_pos, sim_pos, q, phi, valid_mask=common,
        q_min=harmonic_q_min_use, q_max=harmonic_q_max_use, n_phi_bins=180,
    ))
    polar_n_q_bins = int(min(120, max(48, int(max(n_bins, 48)) // 6)))
    polar_metrics, polar_data = _polar_image_matching_metrics(
        exp_z, sim_z, q, phi, valid_mask=common,
        q_min=q_min_use, q_max=q_max_use, n_q_bins=polar_n_q_bins, n_phi_bins=180,
    )
    metrics.update(polar_metrics)
    harmonic_metric_rows = []
    orders = np.asarray(exp_profiles.get("harmonic_orders", []), dtype=int)
    exp_h = np.asarray(exp_profiles.get("harmonics", np.empty((0, 0))), dtype=float)
    sim_h = np.asarray(sim_profiles.get("harmonics", np.empty((0, 0))), dtype=float)
    exp_qavg = np.asarray(exp_profiles.get("harmonic_qavg", np.full(orders.size, np.nan, dtype=float)), dtype=float)
    sim_qavg = np.asarray(sim_profiles.get("harmonic_qavg", np.full(orders.size, np.nan, dtype=float)), dtype=float)
    harmonic_corrs = []
    harmonic_nrmses = []
    for j, n in enumerate(orders):
        corr = _safe_corrcoef(exp_h[j], sim_h[j]) if j < exp_h.shape[0] and j < sim_h.shape[0] else float("nan")
        nrmse = _safe_nrmse(exp_h[j], sim_h[j]) if j < exp_h.shape[0] and j < sim_h.shape[0] else float("nan")
        metrics[f"A{int(n)}_anisotropy_corr"] = corr
        metrics[f"A{int(n)}_anisotropy_nrmse"] = nrmse
        harmonic_metric_rows.append({
            "order": int(n),
            "corr": corr,
            "nrmse": nrmse,
            "qavg_experimental": float(exp_qavg[j]) if j < exp_qavg.size else float("nan"),
            "qavg_simulation": float(sim_qavg[j]) if j < sim_qavg.size else float("nan"),
            "qavg_abs_difference": abs(float(exp_qavg[j]) - float(sim_qavg[j])) if j < exp_qavg.size and j < sim_qavg.size and np.isfinite(exp_qavg[j]) and np.isfinite(sim_qavg[j]) else float("nan"),
        })
        if np.isfinite(corr):
            harmonic_corrs.append(corr)
        if np.isfinite(nrmse):
            harmonic_nrmses.append(nrmse)
    metrics["harmonics_mean_corr_A1_A12"] = float(np.nanmean(harmonic_corrs)) if harmonic_corrs else float("nan")
    metrics["harmonics_mean_nrmse_A1_A12"] = float(np.nanmean(harmonic_nrmses)) if harmonic_nrmses else float("nan")
    metrics["harmonics_mean_corr_A2_A6"] = metrics["harmonics_mean_corr_A1_A12"]
    metrics["harmonics_mean_nrmse_A2_A6"] = metrics["harmonics_mean_nrmse_A1_A12"]
    global_names = list(exp_global.keys())
    global_abs_diff = {}
    for name in global_names:
        ev = float(exp_global.get(name, float("nan")))
        sv = float(sim_global.get(name, float("nan")))
        if np.isfinite(ev) and np.isfinite(sv):
            global_abs_diff[name] = abs(ev - sv)
        else:
            global_abs_diff[name] = float("nan")
    residual_z = np.full_like(exp_z, np.nan, dtype=float)
    residual_z[common] = exp_z[common] - sim_z[common]
    return {
        "metrics": metrics,
        "global_experimental": exp_global,
        "global_simulation": sim_global,
        "global_abs_difference": global_abs_diff,
        "harmonic_metric_rows": harmonic_metric_rows,
        "exp_profiles": exp_profiles,
        "sim_profiles": sim_profiles,
        "exp_display": exp_disp,
        "sim_display": sim_disp,
        "exp_z": exp_z,
        "sim_z": sim_z,
        "exp_pos": exp_pos,
        "sim_pos": sim_pos,
        "residual_z": residual_z,
        "common_mask": common,
        "exp_meta": exp_meta,
        "sim_meta": sim_meta,
        "exp_display_meta": exp_disp_meta,
        "sim_display_meta": sim_disp_meta,
        "exp_pos_meta": exp_pos_meta,
        "sim_pos_meta": sim_pos_meta,
        "polar_data": polar_data,
    }
def _format_feature_metric(value):
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "nan"
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-3):
        return f"{v:.4e}"
    return f"{v:.5g}"
SIMULATORS = [
    ("Ellipsoid", "gui_ellipsoid1", "EllipsoidSAXSGUI"),
    ("Ellipsoid OOP Revolved", "gui_ellipsoid1_oop", "EllipsoidOutOfPlaneSAXSGUI"),
    ("Two ellipsoids", "gui_ellipsoid2", "TwoEllipsoidsGUI"),
    ("Three ellipsoids", "gui_ellipsoid3", "ThreeEllipsoidsGUI"),
    ("Polyhedron", "gui_polyhedron", "PolyhedronGUI"),
    ("Polyhedron + 1 ellipsoid", "gui_polyhedron_ellipsoid1", "MergedObjectsGUI"),
    ("Polyhedron + 2 ellipsoids", "gui_polyhedron_ellipsoid2", "PolyhedronTwoEllipsoidsGUI"),
    ("Random blob", "gui_random", "BlobGUI"),
    ("Random + 1 ellipsoid", "gui_random_ellipsoid1", "RandomBlobEllipsoidGUI"),
    ("Random + 2 ellipsoids", "gui_random_ellipsoid2", "RandomBlobTwoEllipsoidsGUI"),
    ("Coned droplet", "gui_coned_droplet", "ConedDropletGUI"),
    ("Coned + 1 ellipsoid", "gui_coned_droplet_ellipsoid1", "ConedDropletEllipsoidGUI"),
    ("Coned + 2 ellipsoids", "gui_coned_droplet_ellipsoid2", "ConedDropletTwoEllipsoidsGUI"),
    ("Coned + polyhedron + ellipsoid", "gui_coned_droplet_polyhedron_ellipsoid", "ConedDropletPolyhedronEllipsoidGUI"),
    ("Double coned droplet", "gui_double_coned_droplet", "DoubleConedDropletGUI"),
    ("Double coned + 1 ellipsoid", "gui_double_coned_droplet_ellipsoid1", "DoubleConedDropletEllipsoidGUI"),
    ("Double coned + 2 ellipsoids", "gui_double_coned_droplet_ellipsoid2", "DoubleConedDropletTwoEllipsoidsGUI"),
    ("Double coned + polyhedron", "gui_double_coned_droplet_polyhedron", "DoubleConedDropletPolyhedronGUI"),
    ("Double coned + polyhedron + ellipsoid", "gui_double_coned_droplet_polyhedron_ellipsoid", "DoubleConedDropletPolyhedronEllipsoidGUI"),
]
class IntegratedSAXSSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated SAXS Simulator")
        self.root.geometry("1850x1000")
        self.loaded_apps = {}
        self.host_frames = {}
        self.current_index = None
        self._dropdown_syncing = False
        self.exp_base_dir_var = tk.StringVar(value=str(BASE_DIR))
        self.exp_run_number_var = tk.StringVar(value="")
        self.exp_image_number_var = tk.StringVar(value="1")
        self.exp_status_var = tk.StringVar(value="No experimental image loaded.")
        self.exp_last_run_source = None
        self.exp_last_run_name = None
        self.exp_image_data = None
        self.exp_image_title = ""
        self.exp_inset_ax = None
        self._exp_layout_reference = {}
        self.poisson_noise_var = tk.BooleanVar(value=False)
        self.poisson_peak_counts_var = tk.StringVar(value="10000")
        self.poisson_seed_var = tk.StringVar(value="")
        self.poisson_use_dataset_reference_var = tk.BooleanVar(value=True)
        self.poisson_count_multiplier_var = tk.StringVar(value="1.0")
        self.poisson_reference_percentile_var = tk.StringVar(value=f"{DEFAULT_POISSON_REFERENCE_PERCENTILE:g}")
        self.poisson_reference_status_var = tk.StringVar(value="Poisson ref: loading bundled cimg001.npz...")
        self.poisson_reference_intensity = None
        self.poisson_reference_metadata = {}
        self._calibrate_poisson_reference_from_default_npz(show_errors=False)
        self._build_ui()
        self.select_simulator(0)
    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        top = ttk.Frame(self.root, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        title = ttk.Label(top, text="Integrated SAXS Simulator", font=("Arial", 14, "bold"))
        title.grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.combo_var = tk.StringVar(value=SIMULATORS[0][0])
        self.combo = ttk.Combobox(
            top,
            textvariable=self.combo_var,
            values=[item[0] for item in SIMULATORS],
            state="readonly",
            width=58,
        )
        self.combo.grid(row=0, column=1, sticky="w")
        self.combo.bind("<<ComboboxSelected>>", self._on_dropdown_selected)
        right_controls = ttk.Frame(top)
        right_controls.grid(row=0, column=2, sticky="e", padx=(16, 0))
        hint = ttk.Label(
            right_controls,
            text="Select a simulator from the dropdown  |  Pan / Zoom / Mouse wheel zoom  |  Reset View",
        )
        hint.pack(side=tk.LEFT)
        ttk.Button(
            right_controls,
            text="Batch Auto Save",
            command=self.open_batch_dialog,
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(
            right_controls,
            text="Auto Screen",
            command=self.open_auto_screen_dialog,
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            right_controls,
            text="Electron density / SLD",
            command=self._open_electron_density_reference_window,
        ).pack(side=tk.LEFT, padx=(8, 0))
        self.exp_frame = ttk.LabelFrame(self.root, text="Experimental image loader", padding=(8, 6))
        self.exp_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.exp_frame.columnconfigure(1, weight=1)
        ttk.Label(self.exp_frame, text="Run parent/folder/zip").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.exp_frame, textvariable=self.exp_base_dir_var).grid(row=0, column=1, columnspan=4, sticky="ew", padx=(6, 6))
        ttk.Button(self.exp_frame, text="Browse folder...", command=self._choose_exp_base_dir).grid(row=0, column=5, sticky="e")
        ttk.Label(self.exp_frame, text="Run number").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(self.exp_frame, textvariable=self.exp_run_number_var, width=14).grid(row=1, column=1, sticky="w", padx=(6, 18), pady=(6, 0))
        ttk.Label(self.exp_frame, text="Image number (1–3000)").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(self.exp_frame, textvariable=self.exp_image_number_var, width=10).grid(row=1, column=3, sticky="w", padx=(6, 18), pady=(6, 0))
        ttk.Button(self.exp_frame, text="Load image", command=self._load_experimental_image).grid(row=1, column=4, sticky="e", pady=(6, 0))
        ttk.Button(self.exp_frame, text="Find max image", command=self._load_experimental_max_image).grid(row=1, column=5, sticky="e", padx=(6, 0), pady=(6, 0))
        ttk.Button(self.exp_frame, text="SAXS integrate", command=self._open_saxs_integration_window).grid(row=1, column=6, sticky="e", padx=(6, 0), pady=(6, 0))
        ttk.Button(self.exp_frame, text="2D features", command=self._open_saxs_2d_feature_window).grid(row=1, column=7, sticky="e", padx=(6, 0), pady=(6, 0))
        ttk.Button(self.exp_frame, text="Single-shot output", command=self._export_current_single_shot_screening).grid(row=1, column=8, sticky="e", padx=(6, 0), pady=(6, 0))
        ttk.Button(self.exp_frame, text="Polyhedron+Ellipsoid sweep 1D", command=self._open_polyhedron_ellipsoid1_sweep_dialog).grid(row=1, column=9, sticky="e", padx=(6, 0), pady=(6, 0))
        ttk.Label(self.exp_frame, textvariable=self.exp_status_var).grid(row=2, column=0, columnspan=10, sticky="w", pady=(6, 0))
        noise_bar = ttk.Frame(self.exp_frame)
        noise_bar.grid(row=3, column=0, columnspan=10, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(
            noise_bar,
            text="Add Poisson noise to simulation",
            variable=self.poisson_noise_var,
            command=self._update_poisson_noise_only,
        ).pack(side=tk.LEFT)
        ttk.Label(noise_bar, text="Reference").pack(side=tk.LEFT, padx=(14, 4))
        ttk.Label(noise_bar, textvariable=self.poisson_reference_status_var).pack(side=tk.LEFT)
        ttk.Label(noise_bar, text="Multiplier").pack(side=tk.LEFT, padx=(14, 4))
        mult_entry = ttk.Entry(noise_bar, textvariable=self.poisson_count_multiplier_var, width=7)
        mult_entry.pack(side=tk.LEFT)
        mult_entry.bind("<Return>", lambda _event: self._update_poisson_noise_only())
        ttk.Label(noise_bar, text="Seed").pack(side=tk.LEFT, padx=(14, 4))
        seed_entry = ttk.Entry(noise_bar, textvariable=self.poisson_seed_var, width=8)
        seed_entry.pack(side=tk.LEFT)
        seed_entry.bind("<Return>", lambda _event: self._update_poisson_noise_only())
        ttk.Button(noise_bar, text="Apply / replot", command=self._update_poisson_noise_only).pack(side=tk.LEFT, padx=(8, 0))
        self.content = ttk.Frame(self.root)
        self.content.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)
        self._single_shot_preview_windows = []
    def _run_clean_update_plot_for_screening(self, app):
        """Regenerate a clean simulator result for screening-style processing.
        This bypasses the display-side Poisson mutation that is normally applied
        by the wrapped update_plot().  The interactive display can be noisy; the
        saved cache and the raw sim_data used by Auto Screen/Single-shot remain
        clean until the explicit comparison-processing step.
        """
        if app is None or not hasattr(app, "update_plot"):
            return None
        old_flag = getattr(self, "_suspend_poisson_processing_update", False)
        self._suspend_poisson_processing_update = True
        try:
            result = app.update_plot()
        finally:
            self._suspend_poisson_processing_update = old_flag
        return result
    def _create_single_shot_screening_helper(self, simulator_name, app):
        helper = object.__new__(AutoScreenDialog)
        helper.parent = self.root
        helper.simulator_name = simulator_name
        helper.app = app
        helper.root = self.root
        helper.noise_source = self
        helper.param_rows = {}
        helper.cancel_requested = False
        helper.is_running = False
        helper.METRIC_FIELDS = list(AutoScreenDialog.METRIC_FIELDS)
        helper.n_bins_var = tk.StringVar(master=self.root, value="500")
        helper.radial_corr_min_var = tk.StringVar(master=self.root, value="0.95")
        helper.radial_nrmse_max_var = tk.StringVar(master=self.root, value="0.25")
        helper.radial_chi2_max_var = tk.StringVar(master=self.root, value="5.0")
        helper.integration_curve_corr_min_var = tk.StringVar(master=self.root, value="0.97")
        helper.integration_curve_nrmse_max_var = tk.StringVar(master=self.root, value="0.20")
        helper.integration_chi2_max_var = tk.StringVar(master=self.root, value="100.0")
        helper.fringe_position_error_max_var = tk.StringVar(master=self.root, value="0.15")
        helper.pixel_corr_min_var = tk.StringVar(master=self.root, value="0.65")
        helper.harmonic_corr_min_var = tk.StringVar(master=self.root, value="0.45")
        helper.a2_phase_error_max_var = tk.StringVar(master=self.root, value="30.0")
        helper.angular_corr_min_var = tk.StringVar(master=self.root, value="0.30")
        helper.total_score_max_var = tk.StringVar(master=self.root, value="0.50")
        helper.threshold_gate_vars = {
            "radial_corr": tk.BooleanVar(master=self.root, value=True),
            "radial_nrmse": tk.BooleanVar(master=self.root, value=True),
            "radial_chi2": tk.BooleanVar(master=self.root, value=True),
            "int_corr": tk.BooleanVar(master=self.root, value=True),
            "int_nrmse": tk.BooleanVar(master=self.root, value=True),
            "int_chi2": tk.BooleanVar(master=self.root, value=True),
            "fringe": tk.BooleanVar(master=self.root, value=True),
            "pixel_corr": tk.BooleanVar(master=self.root, value=True),
            "harmonic_corr": tk.BooleanVar(master=self.root, value=True),
            "harmonic_phase": tk.BooleanVar(master=self.root, value=True),
            "angular_corr": tk.BooleanVar(master=self.root, value=True),
            "total_score": tk.BooleanVar(master=self.root, value=True),
        }
        helper.poisson_noise_var = tk.BooleanVar(master=self.root, value=bool(self.poisson_noise_var.get()))
        helper.poisson_peak_counts_var = tk.StringVar(master=self.root, value=str(self.poisson_peak_counts_var.get()))
        helper.poisson_seed_var = tk.StringVar(master=self.root, value=str(self.poisson_seed_var.get()))
        helper.poisson_use_dataset_reference_var = tk.BooleanVar(master=self.root, value=bool(self.poisson_use_dataset_reference_var.get()))
        helper.poisson_count_multiplier_var = tk.StringVar(master=self.root, value=str(self.poisson_count_multiplier_var.get()))
        helper.poisson_reference_percentile_var = tk.StringVar(master=self.root, value=str(self.poisson_reference_percentile_var.get()))
        helper.poisson_reference_intensity = self.poisson_reference_intensity
        helper.poisson_reference_metadata = dict(self.poisson_reference_metadata or {})
        helper.score_formula_var = tk.StringVar(master=self.root, value="")
        helper.score_term_vars = {}
        for term in AutoScreenDialog.SCORE_TERMS:
            key = str(term["key"])
            helper.score_term_vars[key] = {
                "enabled": tk.BooleanVar(master=self.root, value=bool(term.get("enabled", True))),
                "weight": tk.StringVar(master=self.root, value=str(term.get("weight", 1.0))),
            }
        return helper
    def _build_current_experimental_record(self):
        if getattr(self, "exp_image_data", None) is None:
            raise ValueError("No experimental image is currently loaded.")
        image = np.asarray(self.exp_image_data, dtype=float)
        run_number_text = str(self.exp_run_number_var.get()).strip()
        try:
            run_number = int(float(run_number_text)) if run_number_text != "" else -1
        except Exception:
            run_number = -1
        image_number_text = str(self.exp_image_number_var.get()).strip()
        try:
            image_number = int(float(image_number_text)) if image_number_text != "" else -1
        except Exception:
            image_number = -1
        run_name = str(self.exp_last_run_name or (f"run{run_number:05d}" if run_number >= 0 else "current_run"))
        exp_id = f"{_safe_filename_fragment(run_name)}_img{max(0, image_number):04d}"
        return {
            "base_dir": str(self.exp_base_dir_var.get()),
            "run_number": run_number,
            "run_name": run_name,
            "run_source": str(self.exp_last_run_source) if self.exp_last_run_source is not None else "",
            "image_number": image_number,
            "npz_name": "",
            "local_number": int(image_number) if image_number >= 0 else -1,
            "id": exp_id,
            "image": image,
            "valid_mask": _load_default_saxs_valid_mask(image.shape),
            "q_u": _load_default_q_axes_from_poni(image.shape)[0],
            "q_v": _load_default_q_axes_from_poni(image.shape)[1],
            "source": "current_gui",
            "csv_path": "",
            "csv_row": "",
        }
    def _collect_current_simulation_combo(self, app=None):
        if app is None:
            app = self._get_current_app()
        if app is None or not hasattr(app, "vars"):
            raise ValueError("No simulator is currently loaded.")
        combo = {}
        for name, var in getattr(app, "vars", {}).items():
            try:
                combo[name] = float(var.get())
            except Exception:
                try:
                    combo[name] = var.get()
                except Exception:
                    pass
        return combo
    def _extract_current_displayed_single_shot_sim_data(self, app=None):
        if app is None:
            app = self._get_current_app()
        if app is None:
            raise ValueError("No simulator is currently loaded.")
        # Use the exact image currently shown in the simulator window so the
        # single-shot output matches the live display and the SAXS integrate
        # window even when Poisson noise is enabled with an empty/random seed.
        sim_img, q_u, q_v = self._get_current_simulation_masked_image_and_q()
        q_u_arr = None if q_u is None else np.asarray(q_u, dtype=float)
        q_v_arr = None if q_v is None else np.asarray(q_v, dtype=float)
        enabled = False
        peak_counts = None
        seed = None
        try:
            enabled = bool(self.poisson_noise_var.get())
        except Exception:
            enabled = False
        if enabled:
            try:
                peak_counts, _ref_intensity, _ref_mode, _multiplier = self._poisson_reference_state()
                peak_counts = float(peak_counts)
            except Exception:
                peak_counts = None
            try:
                seed = self._parse_poisson_seed()
            except Exception:
                seed = None
        return {
            "image": np.asarray(sim_img, dtype=float).copy(),
            "q_u": q_u_arr,
            "q_v": q_v_arr,
            "image_key": "I_masked",
            "is_clean_cache": False,
            "from_cache": False,
            "use_image_as_is": True,
            "screening_noise_enabled": bool(enabled),
            "screening_peak_counts": peak_counts,
            "screening_seed": seed,
        }
    def _compute_current_single_shot_payload(self):
        app = self._get_current_app()
        if app is None:
            raise ValueError("No simulator is currently loaded.")
        if getattr(self, "exp_image_data", None) is None:
            raise ValueError("Load an experimental image first.")
        helper = self._create_single_shot_screening_helper(self.combo_var.get(), app)
        thresholds = helper._thresholds()
        exp = self._build_current_experimental_record()
        combo = self._collect_current_simulation_combo(app)
        sim_data = self._extract_current_displayed_single_shot_sim_data(app)
        sim_data["cache_key"] = helper._cache_key(combo)
        sim_data["cache_path"] = ""
        sim_data["from_cache"] = False
        row, comparison = helper._comparison_row(1, 1, combo, exp, sim_data, thresholds)
        return {
            "helper": helper,
            "thresholds": thresholds,
            "exp": exp,
            "combo": combo,
            "sim_data": sim_data,
            "row": row,
            "comparison": comparison,
        }
    def _save_single_shot_output_bundle(self, payload, output_dir):
        helper = payload["helper"]
        thresholds = payload["thresholds"]
        exp = payload["exp"]
        combo = payload["combo"]
        row = dict(payload["row"])
        comparison = payload["comparison"]
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path, heatmap_path, exp_mask_path, sim_mask_path = helper._save_comparison_figures(
            row, comparison, output_path, bucket="single_shot"
        )
        row["summary_figure_path"] = summary_path
        row["harmonic_heatmap_path"] = heatmap_path
        row["experimental_masked_figure_path"] = exp_mask_path
        row["theoretical_masked_figure_path"] = sim_mask_path
        prefix = _safe_filename_fragment(self.combo_var.get() or "single_shot")
        csv_path = output_path / f"{prefix}_single_shot_scores.csv"
        config_path = output_path / f"{prefix}_single_shot_config.json"
        fieldnames = [
            "combo_index", "total_combos", "experimental_id", "run_name", "run_number", "image_number",
            "npz_name", "local_number", "simulator", "cache_key", "cache_path", "from_cache",
            "screening_poisson_noise_enabled", "screening_poisson_peak_counts", "screening_poisson_seed",
            "summary_figure_path", "harmonic_heatmap_path", "experimental_masked_figure_path", "theoretical_masked_figure_path",
            *[f"param_{name}" for name in combo.keys()],
            *helper.METRIC_FIELDS,
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
        config = {
            "mode": "single_shot_output",
            "created_unix": time.time(),
            "simulator": str(self.combo_var.get()),
            "output_dir": str(output_path),
            "experimental_id": exp.get("id", ""),
            "experimental_run_name": exp.get("run_name", ""),
            "experimental_run_number": exp.get("run_number", ""),
            "experimental_image_number": exp.get("image_number", ""),
            "thresholds": thresholds,
            "params": _jsonify_params(combo),
            "screening_poisson_noise_enabled": bool(row.get("screening_poisson_noise_enabled", False)),
            "screening_poisson_peak_counts": row.get("screening_poisson_peak_counts", np.nan),
            "screening_poisson_seed": row.get("screening_poisson_seed", ""),
        }
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        return row, summary_path, csv_path, config_path
    def _show_single_shot_preview_window(self, payload):
        helper = payload["helper"]
        row = payload["row"]
        comparison = payload["comparison"]
        temp_dir = Path(tempfile.mkdtemp(prefix="saxs_single_shot_preview_"))
        summary_path, _heatmap_path, _exp_mask_path, _sim_mask_path = helper._save_comparison_figures(
            row, comparison, temp_dir, bucket="preview"
        )
        win = tk.Toplevel(self.root)
        win.title("Single-shot output preview")
        win.geometry("1300x950")
        win.resizable(True, True)
        top = ttk.Frame(win, padding=(10, 8))
        top.pack(side=tk.TOP, fill=tk.X)
        status_text = (
            f"Accepted: {bool(row.get('accepted', False))}    "
            f"Total score: {_format_feature_metric(row.get('total_score'))}    "
            f"1D corr: {_format_feature_metric(row.get('integration_curve_corr'))}    "
            f"1D χ²red: {_format_feature_metric(row.get('integration_chi2_reduced'))} "
            f"≤ {_format_feature_metric(row.get('integration_chi2_threshold_max'))} "
            f"pass={row.get('integration_chi2_threshold_pass', '')}    "
            f"Fringe err: {_format_feature_metric(row.get('fringe_position_rel_error'))}    "
            f"A2–A6 phase err: {_format_feature_metric(row.get('harmonic_phase_error_weighted_A2_A6_deg'))}°    "
            f"Angular corr: {_format_feature_metric(row.get('angular_profile_corr_zero_shift'))}"
        )
        ttk.Label(top, text=status_text, font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        zoom_bar = ttk.Frame(win, padding=(10, 0, 10, 6))
        zoom_bar.pack(side=tk.TOP, fill=tk.X)
        zoom_label_var = tk.StringVar(value="Zoom: auto-fit")
        body = ttk.Frame(win, padding=(10, 0, 10, 0))
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        canvas = tk.Canvas(body, highlightthickness=0)
        xbar = ttk.Scrollbar(body, orient="horizontal", command=canvas.xview)
        ybar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        image_holder = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=image_holder, anchor="nw")
        def _on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        image_holder.bind("<Configure>", _on_inner_configure)
        photo_ref = {"image": None, "pil": None, "label": None, "scale": 1.0}
        def _render_preview(scale=None):
            if Image is None or ImageTk is None or photo_ref.get("pil") is None:
                return
            if scale is not None:
                photo_ref["scale"] = max(0.05, min(8.0, float(scale)))
            pil_img = photo_ref["pil"]
            s = float(photo_ref["scale"])
            new_w = max(1, int(round(pil_img.width * s)))
            new_h = max(1, int(round(pil_img.height * s)))
            if abs(s - 1.0) < 1e-9:
                show_img = pil_img
            else:
                resample = getattr(Image, "Resampling", Image).LANCZOS
                show_img = pil_img.resize((new_w, new_h), resample)
            photo = ImageTk.PhotoImage(show_img)
            photo_ref["image"] = photo
            if photo_ref.get("label") is None:
                lbl = ttk.Label(image_holder, image=photo)
                lbl.pack(anchor="nw")
                photo_ref["label"] = lbl
            else:
                photo_ref["label"].configure(image=photo)
            zoom_label_var.set(f"Zoom: {int(round(s * 100))}%")
            win.after_idle(lambda: canvas.configure(scrollregion=canvas.bbox("all")))
        def _zoom_by(factor):
            _render_preview(photo_ref.get("scale", 1.0) * float(factor))
        def _zoom_reset():
            _render_preview(1.0)
        def _fit_scale(mode="window"):
            if Image is None or ImageTk is None or photo_ref.get("pil") is None:
                return None
            canvas.update_idletasks()
            available_w = max(100, canvas.winfo_width() - 6)
            available_h = max(100, canvas.winfo_height() - 6)
            pil = photo_ref["pil"]
            scale_w = available_w / max(1, pil.width)
            scale_h = available_h / max(1, pil.height)
            if mode == "width":
                return scale_w
            return min(scale_w, scale_h)
        def _zoom_fit_width():
            scale = _fit_scale("width")
            if scale is not None:
                _render_preview(scale)
        def _zoom_fit_window():
            scale = _fit_scale("window")
            if scale is not None:
                _render_preview(scale)
        ttk.Button(zoom_bar, text="Zoom -", command=lambda: _zoom_by(0.80)).pack(side=tk.LEFT)
        ttk.Button(zoom_bar, text="Zoom +", command=lambda: _zoom_by(1.25)).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(zoom_bar, text="100%", command=_zoom_reset).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(zoom_bar, text="Fit width", command=_zoom_fit_width).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(zoom_bar, text="Fit window", command=_zoom_fit_window).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(zoom_bar, textvariable=zoom_label_var).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(zoom_bar, text="Mouse wheel: zoom image").pack(side=tk.LEFT, padx=(12, 0))
        if Image is not None and ImageTk is not None:
            try:
                pil_img = Image.open(summary_path).convert("RGB")
                photo_ref["pil"] = pil_img
                win.after(150, _zoom_fit_window)
            except Exception as exc:
                ttk.Label(image_holder, text=f"Could not render preview image: {type(exc).__name__}: {exc}").pack(anchor="w")
                ttk.Label(image_holder, text=str(summary_path)).pack(anchor="w")
        else:
            ttk.Label(image_holder, text="Preview image rendering requires Pillow.").pack(anchor="w")
            ttk.Label(image_holder, text=str(summary_path)).pack(anchor="w")
        def _preview_zoom_step(direction):
            if direction == "up":
                _zoom_by(1.10)
            elif direction == "down":
                _zoom_by(1.0 / 1.10)
            return "break"
        def _on_preview_mousewheel(event):
            delta = getattr(event, "delta", 0)
            if delta > 0:
                return _preview_zoom_step("up")
            if delta < 0:
                return _preview_zoom_step("down")
            num = getattr(event, "num", None)
            if num == 4:
                return _preview_zoom_step("up")
            if num == 5:
                return _preview_zoom_step("down")
            return "break"
        def _bind_preview_zoom_targets():
            targets = [win, canvas, image_holder]
            if photo_ref.get("label") is not None:
                targets.append(photo_ref["label"])
            for target in targets:
                try:
                    target.bind("<MouseWheel>", _on_preview_mousewheel, add="+")
                except Exception:
                    pass
                try:
                    target.bind("<Button-4>", _on_preview_mousewheel, add="+")
                    target.bind("<Button-5>", _on_preview_mousewheel, add="+")
                except Exception:
                    pass
        win._single_shot_autofit_done = False
        def _initial_fit_once(_event=None):
            if getattr(win, "_single_shot_autofit_done", False):
                return
            if photo_ref.get("pil") is None:
                return
            if canvas.winfo_width() < 50 or canvas.winfo_height() < 50:
                return
            _zoom_fit_window()
            win._single_shot_autofit_done = True
            _bind_preview_zoom_targets()
        canvas.bind("<Configure>", _initial_fit_once, add="+")
        win.after(250, _initial_fit_once)
        bottom = ttk.Frame(win, padding=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        def _save_now():
            default_dir = (BASE_DIR / "single_shot_outputs" / _safe_filename_fragment(self.combo_var.get())).resolve()
            folder = filedialog.askdirectory(
                parent=win,
                title="Choose output folder for single-shot screening output",
                mustexist=False,
                initialdir=str(default_dir.parent if default_dir.parent.exists() else BASE_DIR),
            )
            if not folder:
                return
            try:
                saved_row, saved_summary, csv_path, _config_path = self._save_single_shot_output_bundle(payload, folder)
                messagebox.showinfo(
                    "Single-shot output",
                    f"Single-shot screening output saved.\n\n"
                    f"Accepted: {bool(saved_row.get('accepted', False))}\n"
                    f"Total score: {_format_feature_metric(saved_row.get('total_score'))}\n"
                    f"1D χ²red: {_format_feature_metric(saved_row.get('integration_chi2_reduced'))}\n"
                    f"Summary figure: {saved_summary}\n"
                    f"CSV: {csv_path}",
                    parent=win,
                )
            except Exception as exc:
                messagebox.showerror(
                    "Single-shot output",
                    f"Could not save single-shot screening output.\n\n{type(exc).__name__}: {exc}",
                    parent=win,
                )
        def _close_preview():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                self._single_shot_preview_windows.remove(win)
            except Exception:
                pass
            win.destroy()
        ttk.Button(bottom, text="Save...", command=_save_now).pack(side=tk.RIGHT)
        ttk.Button(bottom, text="Close", command=_close_preview).pack(side=tk.RIGHT, padx=(0, 8))
        ttk.Label(bottom, text=f"Preview file: {summary_path}").pack(side=tk.LEFT)
        win.protocol("WM_DELETE_WINDOW", _close_preview)
        win._single_shot_preview_photo = photo_ref
        self._single_shot_preview_windows.append(win)
    def _export_current_single_shot_screening(self):
        try:
            payload = self._compute_current_single_shot_payload()
            self._show_single_shot_preview_window(payload)
        except Exception as exc:
            messagebox.showerror(
                "Single-shot output",
                f"Could not generate single-shot preview.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
    def _parse_poisson_peak_counts(self):
        try:
            value = float(self.poisson_peak_counts_var.get().strip())
        except Exception:
            value = 10000.0
        if not np.isfinite(value) or value <= 0:
            value = 10000.0
            self.poisson_peak_counts_var.set("10000")
        return value
    def _parse_poisson_seed(self):
        text = self.poisson_seed_var.get().strip()
        if text == "":
            return None
        try:
            return int(text)
        except Exception:
            # Allow a non-integer string to act as a reproducible seed.
            return abs(hash(text)) % (2 ** 32)
    def _parse_poisson_count_multiplier(self):
        try:
            value = float(self.poisson_count_multiplier_var.get().strip())
        except Exception:
            value = 1.0
        if not np.isfinite(value) or value <= 0:
            value = 1.0
            self.poisson_count_multiplier_var.set("1.0")
        return value
    def _parse_poisson_reference_percentile(self):
        try:
            value = float(self.poisson_reference_percentile_var.get().strip())
        except Exception:
            value = 99.9
        if not np.isfinite(value):
            value = 99.9
        value = min(100.0, max(0.0, value))
        self.poisson_reference_percentile_var.set(f"{value:g}")
        return value
    def _poisson_reference_state(self):
        peak_counts = self._parse_poisson_peak_counts()
        multiplier = self._parse_poisson_count_multiplier()
        use_dataset_ref = bool(self.poisson_use_dataset_reference_var.get())
        ref = self.poisson_reference_intensity
        try:
            ref = float(ref)
        except Exception:
            ref = None
        if use_dataset_ref and ref is not None and np.isfinite(ref) and ref > 0:
            return float(ref * multiplier), float(ref), "dataset_max", float(multiplier)
        return float(peak_counts), None, "simulation_max", float(multiplier)
    def _calibrate_poisson_reference_from_default_npz(self, show_errors=True):
        """Use the fixed Poisson reference calibrated once from cimg001.npz.
        The bundled reference dataset was already inspected: the strongest image is
        local image #262 and its p99.9 positive intensity is
        DEFAULT_POISSON_REFERENCE_COUNT.  Runtime noise updates must not reload the
        npz file; they only multiply this fixed count by the GUI multiplier.
        """
        ref = float(DEFAULT_POISSON_REFERENCE_COUNT)
        meta = {
            "mode": "fixed_cimg001_img262_p99.9",
            "npz_name": "cimg001.npz",
            "image_number": int(DEFAULT_POISSON_REFERENCE_IMAGE_NUMBER),
            "local_number": int(DEFAULT_POISSON_REFERENCE_IMAGE_NUMBER),
            "percentile": float(DEFAULT_POISSON_REFERENCE_PERCENTILE),
            "reference_intensity": ref,
        }
        self.poisson_reference_intensity = ref
        self.poisson_reference_metadata = dict(meta)
        self.poisson_use_dataset_reference_var.set(True)
        self.poisson_peak_counts_var.set(f"{ref:.6g}")
        self.poisson_reference_status_var.set(
            f"fixed cimg001 img {DEFAULT_POISSON_REFERENCE_IMAGE_NUMBER}, p{DEFAULT_POISSON_REFERENCE_PERCENTILE:g}={ref:.6g}"
        )
        try:
            self._update_poisson_noise_only()
        except Exception:
            pass
        return self.poisson_reference_metadata
    def _calibrate_poisson_reference_from_dataset(self):
        # Backward-compatible method name.  The reference now always comes from
        # the bundled cimg001.npz shipped with the program, not from the currently
        # selected experimental run.
        return self._calibrate_poisson_reference_from_default_npz(show_errors=True)
    def _poissonize_image(self, image, peak_counts, rng, reference_intensity=None):
        """Apply Poisson shot noise using an explicit intensity-to-count map.
        ``reference_intensity`` is the simulation-side intensity that should
        correspond to ``peak_counts`` detector counts.  This is deliberately not
        the experimental reference count itself; otherwise the reference count
        cancels out and only the multiplier remains.
        """
        arr = np.asarray(image, dtype=float)
        out = arr.copy()
        finite = np.isfinite(arr)
        if not np.any(finite):
            return out
        signal = np.zeros_like(arr, dtype=float)
        signal[finite] = np.maximum(arr[finite], 0.0)
        max_val = float(np.nanmax(signal[finite]))
        if not np.isfinite(max_val) or max_val <= 0:
            return out
        try:
            ref = float(reference_intensity) if reference_intensity is not None else None
        except Exception:
            ref = None
        if ref is None or not np.isfinite(ref) or ref <= 0:
            ref = max_val
        alpha = float(peak_counts) / float(ref)
        if not np.isfinite(alpha) or alpha <= 0:
            return out
        lam = signal * alpha
        lam[~np.isfinite(lam)] = 0.0
        lam = np.maximum(lam, 0.0)
        counts = rng.poisson(lam)
        noisy = counts.astype(float) / alpha
        out[finite] = noisy[finite]
        return out
    def _estimate_array_reference_intensity(self, array, percentile=99.9):
        """Robust positive-intensity reference for a simulation array."""
        try:
            arr = np.asarray(array, dtype=float)
        except Exception:
            return None
        good = np.isfinite(arr) & (arr > 0)
        if np.count_nonzero(good) < 10:
            return None
        try:
            ref = float(np.nanpercentile(arr[good], float(percentile)))
        except Exception:
            return None
        if not np.isfinite(ref) or ref <= 0:
            return None
        return ref
    def _get_or_set_simulation_poisson_reference(self, app, clean_total):
        """Return the globally fixed simulation-side Poisson reference intensity.
        For consistency across all current and future simulator images, the
        simulation-side reference is no longer taken from the first image that
        happened to receive Poisson noise.  Instead we use one explicit reference
        condition calibrated once from the Ellipse 1 simulator with
        b=1, object_scale=0.5, scale_factor=5000 (angle arbitrary; 0 deg used in
        the numerical calibration because the b=1 case is rotationally symmetric).
        This keeps the intensity-to-count mapping fixed when object size or other
        parameters are changed, so smaller objects naturally yield lower expected
        counts and therefore larger relative Poisson noise.
        """
        ref = float(DEFAULT_SIMULATION_POISSON_REFERENCE_INTENSITY)
        if not np.isfinite(ref) or ref <= 0:
            return None
        if app is not None:
            try:
                app._poisson_fixed_sim_reference_intensity = ref
                app._poisson_fixed_sim_reference_metadata = dict(DEFAULT_SIMULATION_POISSON_REFERENCE_METADATA)
            except Exception:
                pass
        return ref
    def _invalidate_poisson_result_cache(self, app):
        """Drop cached clean/noisy arrays when simulator parameters are rebuilt."""
        try:
            for name in (
                "_poisson_clean_last_result",
                "_poisson_clean_artist_arrays",
                "_poisson_last_state",
                "_poisson_last_applied_state",
            ):
                if hasattr(app, name):
                    delattr(app, name)
        except Exception:
            pass
    def _find_simulation_image_artists(self, app):
        """Return matplotlib image artists corresponding to theoretical panels."""
        fig = getattr(app, "fig", None)
        if fig is None:
            return []
        pairs = []
        for ax in list(getattr(fig, "axes", [])):
            if getattr(ax, "get_label", lambda: "")() == "experimental_inset":
                continue
            title = (ax.get_title() or "").strip().lower()
            if not title:
                continue
            # Avoid adding noise to real-space projected density.  Apply only to
            # reciprocal-space theoretical panels that users inspect.
            if "masked theoretical" in title or "log theoretical" in title:
                imgs = list(getattr(ax, "images", []))
                if imgs:
                    pairs.append((title, imgs[0]))
        return pairs
    def _display_array_matching_artist(self, clean_artist_array, clean_raw, noisy_raw, *, log_display=False):
        """Return noisy_raw transformed in the same display convention as clean_artist_array.
        Individual simulator modules do not all draw q-space images with the same
        orientation.  Ellipse 1 uses flipud(...)+origin="lower", whereas the
        multi-ellipsoid modules use matshow(...) with the raw array.  The Poisson
        updater therefore must not impose a universal flipud.  Instead, infer the
        artist's display transform from the clean artist data and apply the same
        transform to the noisy data.
        """
        try:
            clean_disp = np.asarray(clean_artist_array, dtype=float)
            clean = np.asarray(clean_raw, dtype=float)
            noisy = np.asarray(noisy_raw, dtype=float)
        except Exception:
            if log_display:
                return np.log(np.asarray(noisy_raw, dtype=float) + 1.0e-12)
            return np.asarray(noisy_raw, dtype=float)
        if clean_disp.shape != clean.shape or noisy.shape != clean.shape:
            if log_display:
                return np.log(np.maximum(noisy, 0.0) + 1.0e-12)
            return noisy
        if log_display:
            clean_base = np.log(np.maximum(clean, 0.0) + 1.0e-12)
            noisy_base = np.log(np.maximum(noisy, 0.0) + 1.0e-12)
        else:
            clean_base = clean
            noisy_base = noisy
        candidates = [
            (clean_base, noisy_base),
            (np.flipud(clean_base), np.flipud(noisy_base)),
            (np.fliplr(clean_base), np.fliplr(noisy_base)),
            (np.flipud(np.fliplr(clean_base)), np.flipud(np.fliplr(noisy_base))),
            (clean_base.T, noisy_base.T) if clean_base.T.shape == clean_disp.shape else None,
            (np.flipud(clean_base.T), np.flipud(noisy_base.T)) if clean_base.T.shape == clean_disp.shape else None,
        ]
        best = None
        best_err = None
        for item in candidates:
            if item is None:
                continue
            c, n = item
            if c.shape != clean_disp.shape:
                continue
            good = np.isfinite(clean_disp) & np.isfinite(c)
            if np.count_nonzero(good) < 10:
                continue
            diff = clean_disp[good] - c[good]
            scale = np.nanmedian(np.abs(clean_disp[good])) + np.nanmedian(np.abs(c[good])) + 1.0e-12
            err = float(np.nanmean((diff / scale) ** 2))
            if best_err is None or err < best_err:
                best_err = err
                best = n
        if best is not None:
            return best
        return noisy_base
    def _apply_poisson_noise_to_current_result(self, app):
        """Apply or remove display-side Poisson noise for the current simulator.
        The clean simulator output is cached once.  Multiplier changes re-use the
        clean cache, generate a fresh noisy image with the new effective count
        level, update app.last_result, and replace the displayed image arrays.
        """
        result = getattr(app, "last_result", None)
        if not isinstance(result, dict):
            return
        # Cache the clean result only once per simulator update.
        if not hasattr(app, "_poisson_clean_last_result"):
            clean = {}
            for k, v in result.items():
                if isinstance(v, np.ndarray):
                    clean[k] = np.array(v, copy=True)
                elif isinstance(v, dict):
                    clean[k] = dict(v)
                else:
                    clean[k] = v
            app._poisson_clean_last_result = clean
            artist_arrays = []
            for title, artist in self._find_simulation_image_artists(app):
                try:
                    artist_arrays.append((title, artist, np.array(artist.get_array(), copy=True)))
                except Exception:
                    pass
            app._poisson_clean_artist_arrays = artist_arrays
        clean_result = getattr(app, "_poisson_clean_last_result", result)
        clean_artists = getattr(app, "_poisson_clean_artist_arrays", [])
        enabled = bool(self.poisson_noise_var.get())
        peak_counts, reference_intensity, _mode, _multiplier = self._poisson_reference_state()
        seed = self._parse_poisson_seed()
        state = (enabled, float(peak_counts), float(reference_intensity) if reference_intensity else None, seed)
        # If the current noisy realization for this exact state is already applied,
        # do not draw a new Poisson sample. This keeps the live simulator window,
        # single-shot output, integration window, and feature-comparison windows
        # visually consistent when they all query the current image.
        if enabled and getattr(app, "_poisson_last_applied_state", None) == state:
            return
        # Restore clean state when disabled.
        if not enabled:
            restored = {}
            for k, v in clean_result.items():
                restored[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else (dict(v) if isinstance(v, dict) else v)
            app.last_result = restored
            for _title, artist, clean_arr in clean_artists:
                try:
                    artist.set_data(clean_arr)
                except Exception:
                    pass
            canvas = getattr(app, "canvas", None)
            if canvas is not None:
                try:
                    canvas.draw_idle()
                except Exception:
                    pass
            app._poisson_last_state = state
            app._poisson_last_applied_state = None
            return
        rng = np.random.default_rng(seed)
        # Generate noisy reciprocal-space arrays from the clean physical arrays.
        clean_total = clean_result.get("I_total", None)
        clean_masked = clean_result.get("I_masked", None)
        mask = getattr(app, "mask", None)
        if clean_masked is None:
            for alt in ("masked", "masked_theoretical"):
                if alt in clean_result:
                    clean_masked = clean_result[alt]
                    break
        if clean_total is None and clean_masked is not None:
            clean_total = clean_masked
        if clean_total is None:
            return
        # Convert arbitrary simulation units to the experimental reference count
        # scale.  The fixed experimental reference determines peak_counts, and the
        # fixed Ellipse-1 simulation reference provides one global bridge from
        # arbitrary simulation units to the experimental count scale.
        sim_reference_intensity = self._get_or_set_simulation_poisson_reference(app, clean_total)
        if sim_reference_intensity is None:
            sim_reference_intensity = reference_intensity
        noisy_total = self._poissonize_image(clean_total, peak_counts, rng, reference_intensity=sim_reference_intensity)
        if clean_masked is not None:
            if mask is not None and np.asarray(mask).shape == np.asarray(noisy_total).shape:
                noisy_masked = noisy_total * np.asarray(mask, dtype=float)
            else:
                # Use a second draw only if no mask relationship is available.
                noisy_masked = self._poissonize_image(clean_masked, peak_counts, rng, reference_intensity=sim_reference_intensity)
        else:
            noisy_masked = None
        updated = {}
        for k, v in clean_result.items():
            updated[k] = np.array(v, copy=True) if isinstance(v, np.ndarray) else (dict(v) if isinstance(v, dict) else v)
        updated["I_total"] = noisy_total
        if noisy_masked is not None:
            updated["I_masked"] = noisy_masked
        app.last_result = updated
        # Update visible artists using each simulator module's own display
        # convention.  Some modules draw q-space panels as flipud(...)+imshow,
        # whereas two-/three-ellipsoid modules draw raw arrays with matshow.  The
        # transform is inferred from the cached clean artist array to avoid a
        # spurious vertical flip when only Poisson noise should change.
        for title, artist, clean_arr in clean_artists:
            try:
                if "log theoretical" in title:
                    display = self._display_array_matching_artist(
                        clean_arr,
                        clean_total,
                        noisy_total,
                        log_display=True,
                    )
                    artist.set_data(display)
                elif "masked theoretical" in title and noisy_masked is not None:
                    clean_for_match = clean_masked if clean_masked is not None else clean_total
                    display = self._display_array_matching_artist(
                        clean_arr,
                        clean_for_match,
                        noisy_masked,
                        log_display=False,
                    )
                    artist.set_data(display)
            except Exception:
                pass
        canvas = getattr(app, "canvas", None)
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass
        app._poisson_last_state = state
        app._poisson_last_applied_state = state
    def _update_poisson_noise_only(self):
        """Apply/remove Poisson noise without rebuilding the whole simulator figure.
        Re-running app.update_plot() can trigger repeated tight_layout/axes-position
        recalculation inside the simulator modules, which makes the displayed image
        appear to shift slightly in position or size when the Poisson checkbox is
        toggled.  For noise-only changes, we keep the existing artists and replace
        just their image data.
        """
        app = self._get_current_app()
        if app is None:
            return
        try:
            self._apply_poisson_noise_to_current_result(app)
        except Exception:
            try:
                app.update_plot()
            except Exception:
                pass
    def _replot_current_simulator(self):
        app = self._get_current_app()
        if app is None:
            return
        try:
            app.update_plot()
        except Exception:
            try:
                self._apply_poisson_noise_to_current_result(app)
            except Exception:
                pass
    def _choose_exp_base_dir(self):
        folder = filedialog.askdirectory(
            parent=self.root,
            title="Choose run parent folder or run000xxx folder",
            mustexist=True,
        )
        if folder:
            self.exp_base_dir_var.set(folder)
    def _display_experimental_image(self, image, title):
        self.exp_image_data = image
        self.exp_image_title = title
        self._draw_experimental_inset()
    def _get_current_simulator_figure(self):
        if self.current_index is None:
            return None
        app = self.loaded_apps.get(self.current_index)
        return getattr(app, "fig", None) if app is not None else None
    def _get_current_simulator_canvas(self):
        if self.current_index is None:
            return None
        app = self.loaded_apps.get(self.current_index)
        canvas = getattr(app, "canvas", None) if app is not None else None
        if canvas is not None:
            return canvas
        fig = self._get_current_simulator_figure()
        return getattr(fig, "canvas", None) if fig is not None else None
    def _draw_experimental_inset(self):
        if getattr(self, "exp_image_data", None) is None:
            return
        fig = self._get_current_simulator_figure()
        if fig is None:
            return
        old_ax = getattr(self, "exp_inset_ax", None)
        if old_ax is not None:
            try:
                old_ax.remove()
            except Exception:
                pass
            self.exp_inset_ax = None
        sim_axes = []
        for ax in list(fig.axes):
            if ax is old_ax:
                continue
            if getattr(ax, "get_label", lambda: "")() == "experimental_inset":
                continue
            sim_axes.append(ax)
        if not sim_axes:
            return
        # Find the masked-theoretical axis. The experimental inset will match this panel's
        # displayed image size and horizontal position.
        masked_ax = None
        for ax in sim_axes:
            title = (ax.get_title() or "").strip().lower()
            if "masked theoretical" in title:
                masked_ax = ax
                break
        if masked_ax is None:
            masked_ax = sim_axes[-1]
        # Restore a stable baseline layout before applying the experimental inset layout.
        fig_key = id(fig)
        signature = tuple((ax.get_title() or "", getattr(ax, "get_label", lambda: "")()) for ax in sim_axes)
        ref = self._exp_layout_reference.get(fig_key)
        if ref is None or ref.get("signature") != signature or len(ref.get("positions", [])) != len(sim_axes):
            ref = {
                "signature": signature,
                "positions": [list(ax.get_position().bounds) for ax in sim_axes],
            }
            self._exp_layout_reference[fig_key] = ref
        originals = []
        for ax, pos in zip(sim_axes, ref["positions"]):
            ax.set_position(pos)
            originals.append((ax, list(pos)))
        orig_min_y0 = min(y0 for _, (x0, y0, w, h) in originals)
        orig_max_y1 = max(y0 + h for _, (x0, y0, w, h) in originals)
        band_h = orig_max_y1 - orig_min_y0
        masked_index = sim_axes.index(masked_ax)
        _, (_, _, _, masked_h_orig) = originals[masked_index]
        # Reserve room above the simulation band for an experimental inset whose displayed
        # data rectangle matches the masked-theoretical panel after layout adjustment.
        # Three-panel layouts need extra clearance, otherwise the experimental inset can
        # visually clash with the simulation row.
        n_axes = len(sim_axes)
        if n_axes <= 3:
            gap = 0.075
            top_limit = 0.95
            min_bottom = 0.03
            reserve_margin = 0.03
        else:
            gap = 0.03
            top_limit = 0.96
            min_bottom = 0.04
            reserve_margin = 0.0
        # Uniformly compress the simulation band vertically if needed so that:
        #   compressed_band_height + gap + compressed_masked_height <= available_height
        available_h = max(0.10, top_limit - min_bottom - gap - reserve_margin)
        denom = band_h + masked_h_orig
        scale = min(1.0, available_h / denom) if denom > 0 else 1.0
        for ax, (x0, y0, w, h) in originals:
            new_y0 = min_bottom + (y0 - orig_min_y0) * scale
            new_h = h * scale
            ax.set_position([x0, new_y0, w, new_h])
        masked_pos = masked_ax.get_position()
        top_after_layout = max(ax.get_position().y1 for ax in sim_axes)
        inset_w = masked_pos.width
        inset_h = masked_pos.height
        x0 = masked_pos.x0
        y0 = top_after_layout + gap
        # Safety clamp.
        if y0 + inset_h > top_limit:
            y0 = max(0.02, top_limit - inset_h)
        if x0 + inset_w > 0.98:
            x0 = 0.98 - inset_w
        x0 = max(0.02, x0)
        inset_ax = fig.add_axes([x0, y0, inset_w, inset_h], label="experimental_inset")
        inset_ax.imshow(self.exp_image_data)
        inset_ax.set_title("Experimental", fontsize=9, pad=2)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_linewidth(0.8)
        self.exp_inset_ax = inset_ax
        canvas = self._get_current_simulator_canvas()
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass
    def _refresh_experimental_inset_after_simulator_change(self):
        if self.exp_image_data is not None:
            self.root.after(100, self._draw_experimental_inset)
    def _load_experimental_image(self):
        try:
            run_source, run_name = _find_run_source(
                self.exp_base_dir_var.get(),
                self.exp_run_number_var.get(),
            )
            image_number = int(float(self.exp_image_number_var.get()))
            image, npz_name, local_number = _load_run_image_number(run_source, image_number)
        except Exception as exc:
            messagebox.showerror(
                "Experimental image loader",
                f"Could not load experimental image.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        self.exp_last_run_source = run_source
        self.exp_last_run_name = run_name
        self._display_experimental_image(image, f"{run_name}, image {image_number} ({npz_name} #{local_number})")
        self.exp_status_var.set(
            f"Loaded {run_name}: image {image_number}/3000, {npz_name}, local image {local_number}/300, shape={image.shape}"
        )
    def _load_experimental_max_image(self):
        try:
            run_source, run_name = _find_run_source(
                self.exp_base_dir_var.get(),
                self.exp_run_number_var.get(),
            )
            image_number, npz_name, local_number, total_intensity = _find_max_intensity_image_number(run_source)
            image, _, _ = _load_run_image_number(run_source, image_number)
        except Exception as exc:
            messagebox.showerror(
                "Experimental image loader",
                f"Could not find/load max-intensity image.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        self.exp_image_number_var.set(str(image_number))
        self.exp_last_run_source = run_source
        self.exp_last_run_name = run_name
        self._display_experimental_image(image, f"{run_name}, max image {image_number} ({npz_name} #{local_number})")
        self.exp_status_var.set(
            f"Loaded max image from {run_name}: image {image_number}/3000, {npz_name}, local image {local_number}/300, total intensity={total_intensity:.6g}"
        )
    def _get_current_app(self):
        if self.current_index is None:
            return None
        return self.loaded_apps.get(self.current_index)
    def _get_current_simulation_masked_image_and_q(self):
        app = self._get_current_app()
        if app is None:
            raise ValueError("No simulator is currently loaded.")
        # Most GUI modules store the current result in app.last_result.
        result = getattr(app, "last_result", None)
        if result is None:
            try:
                app.update_plot()
                result = getattr(app, "last_result", None)
            except Exception:
                result = None
        if not isinstance(result, dict):
            raise ValueError("Current simulator does not expose last_result.")
        try:
            self._apply_poisson_noise_to_current_result(app)
            result = getattr(app, "last_result", result)
        except Exception:
            pass
        image = None
        for key in ("I_masked", "masked", "masked_theoretical", "I_total"):
            if key in result:
                image = result[key]
                break
        if image is None:
            raise ValueError("Could not find masked theoretical image in current simulator result.")
        q_u = result.get("q_u", None)
        q_v = result.get("q_v", None)
        image = np.asarray(image, dtype=float)
        if image.ndim != 2:
            raise ValueError(f"Simulation image must be 2D, got shape {image.shape}.")
        return image, q_u, q_v
    def _open_electron_density_reference_window(self):
        """Open a compact SAXS electron-density / X-ray SLD reference table."""
        win = tk.Toplevel(self.root)
        win.title("SAXS electron density / X-ray SLD reference")
        win.geometry("1020x700")
        win.transient(self.root)

        outer = ttk.Frame(win, padding=(12, 10))
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        title = ttk.Label(
            outer,
            text="SAXS electron density / X-ray SLD reference",
            font=("Arial", 13, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        intro = (
            "Use Δρ = ρobject − ρmedium for SAXS contrast. "
            "The values below are a compact tabulated reference set for common SAXS materials/solvents. "
            "For quantitative work, prefer tabulated SLD/electron-density values measured or reported for the same temperature, composition, hydration, and X-ray energy."
        )
        ttk.Label(outer, text=intro, wraplength=960, justify=tk.LEFT).grid(
            row=1, column=0, sticky="ew", pady=(4, 10)
        )

        table_frame = ttk.LabelFrame(outer, text="Approximate material reference values", padding=(8, 6))
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        columns = ("material", "formula", "density", "electron_density", "xray_sld", "note")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=13)
        headings = {
            "material": "Material",
            "formula": "Formula / approximation",
            "density": "Density (g cm⁻³)",
            "electron_density": "ρe (e Å⁻³)",
            "xray_sld": "X-ray SLD (10⁻⁶ Å⁻²)",
            "note": "Comment",
        }
        widths = {
            "material": 140,
            "formula": 165,
            "density": 115,
            "electron_density": 120,
            "xray_sld": 145,
            "note": 330,
        }
        for col in columns:
            tree.heading(col, text=headings[col])
            tree.column(col, width=widths[col], anchor=tk.W, stretch=(col == "note"))

        # Approximate values from formula/density using rho_e = density * N_A * Ztot / M.
        rows = [
            ("Water", "H2O", "1.00", "0.334", "9.41", "Common SAXS solvent reference"),
            ("D2O", "D2O", "1.107", "0.370", "10.43", "X-ray contrast close to H2O; neutron contrast differs strongly"),
            ("Methanol / MeOH", "CH3OH", "0.792", "0.268", "7.55", "Approx. liquid value near 20 °C; useful solvent reference"),
            ("Silica", "SiO2", "2.20", "0.661", "18.6", "Use density appropriate to amorphous/porous silica"),
            ("Polystyrene", "C8H8", "1.03–1.05", "0.338–0.345", "9.5–9.7", "Polymer density depends on sample state"),
            ("PMMA", "C5H8O2", "1.18", "0.382", "10.8", "Typical bulk polymer value"),
            ("Polyethylene", "CH2", "0.92–0.96", "0.315–0.329", "8.9–9.3", "Density depends on crystallinity"),
            ("Protein", "average biomolecule", "~1.35", "~0.43–0.45", "~12.1–12.7", "Hydration and partial specific volume matter"),
            ("Silicon", "Si", "2.33", "0.700", "19.7", "Common substrate/reference material"),
            ("Gold", "Au", "19.32", "4.66", "131", "Very high X-ray contrast and absorption"),
        ]
        for row in rows:
            tree.insert("", tk.END, values=row)

        ybar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        xbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")

        calc_frame = ttk.LabelFrame(outer, text="How to use tabulated values in this simulator", padding=(8, 6))
        calc_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        calc_frame.columnconfigure(0, weight=1)

        eq_text = tk.Text(calc_frame, height=7, wrap=tk.WORD)
        eq_text.grid(row=0, column=0, sticky="ew")
        eq_text.insert(
            tk.END,
            "Recommended use:\n"
            "  1. Pick tabulated electron density or X-ray SLD for the object and surrounding medium.\n"
            "  2. Use contrast, Δρ = ρobject − ρmedium, rather than absolute density whenever possible.\n"
            "  3. Normalize contrasts to a convenient relative scale for this simulator.\n\n"
            "Example using tabulated electron densities in water:\n"
            "  silica contrast ≈ 0.661 − 0.334 = 0.327 e Å⁻³\n"
            "  MeOH contrast ≈ 0.268 − 0.334 = −0.066 e Å⁻³\n"
            "  gold contrast ≈ 4.66 − 0.334 = 4.33 e Å⁻³\n\n"
            "Note: solvent mixtures, hydration, porosity, and polymer crystallinity can shift tabulated values noticeably.\n"
        )
        eq_text.configure(state=tk.DISABLED)

        link_frame = ttk.LabelFrame(outer, text="External calculators / references", padding=(8, 6))
        link_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        for c in range(4):
            link_frame.columnconfigure(c, weight=1)

        def open_url(url):
            try:
                webbrowser.open(url, new=2)
            except Exception as exc:
                messagebox.showerror(
                    "Open link",
                    f"Could not open link.\n\n{type(exc).__name__}: {exc}",
                    parent=win,
                )

        links = [
            ("SAS Tutorials SLD table", "https://sastutorials.org/SLD/SLD.html"),
            ("GISAXS SLD overview", "https://gisaxs.com/index.php/Scattering_Length_Density"),
            ("NIST X-ray Tables", "https://www.nist.gov/physical-measurement-laboratory/x-ray-form-factor-atten-scattering-tables"),
            ("SasView SLD Calculator", "https://www.sasview.org/docs/user/qtgui/Calculators/sld_calculator_help.html"),
        ]
        for idx, (label, url) in enumerate(links):
            ttk.Button(link_frame, text=label, command=lambda u=url: open_url(u)).grid(
                row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 6, 0)
            )

        button_bar = ttk.Frame(outer)
        button_bar.grid(row=5, column=0, sticky="e", pady=(10, 0))

        def copy_table():
            header = "\t".join([headings[c] for c in columns])
            body = "\n".join("\t".join(map(str, row)) for row in rows)
            try:
                win.clipboard_clear()
                win.clipboard_append(header + "\n" + body)
            except Exception:
                pass

        ttk.Button(button_bar, text="Copy table", command=copy_table).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(button_bar, text="Close", command=win.destroy).pack(side=tk.LEFT)

    def _select_polyhedron_ellipsoid1_simulator(self):
        """Select the Polyhedron + 1 ellipsoid simulator without altering shape.
        The dedicated simulation-only 1D button reuses the existing
        ``Polyhedron + 1 ellipsoid`` simulator and preserves the currently
        selected ``polyhedron_name`` (tetra, cube, octa, dodeca, or icosa).
        All other user-controlled parameters also remain unchanged.
        """
        target_name = "Polyhedron + 1 ellipsoid"
        target_index = None
        for idx, (tab_name, _module_name, _class_name) in enumerate(SIMULATORS):
            if tab_name == target_name:
                target_index = idx
                break
        if target_index is None:
            raise ValueError(f"Could not find simulator: {target_name}")
        if self.current_index != target_index:
            self.select_simulator(target_index)
        app = self.loaded_apps.get(target_index)
        if app is None:
            raise ValueError(f"{target_name} is not loaded.")
        name_var = getattr(app, "name_var", None)
        if name_var is None:
            raise ValueError("The Polyhedron + 1 ellipsoid simulator does not expose polyhedron_name.")
        update_plot = getattr(app, "update_plot", None)
        if callable(update_plot):
            update_plot()
        try:
            poly_name = str(name_var.get()).strip() or "(unknown)"
        except Exception:
            poly_name = "(unknown)"
        self.exp_status_var.set(
            f"Polyhedron+Ellipsoid 1D: selected 'Polyhedron + 1 ellipsoid' with polyhedron_name={poly_name}."
        )
        return app
    # Backward-compatible alias used by older builds / callbacks.
    def _select_cube_polyhedron_ellipsoid1_simulator(self):
        return self._select_polyhedron_ellipsoid1_simulator()
    def _open_cube_polyhedron_ellipsoid1_integration_window(self):
        return self._open_polyhedron_ellipsoid1_integration_window()
    def _polyhedron_ellipsoid1_numeric_parameter_names(self, app):
        """Return float-compatible parameter names exposed by the current app."""
        names = []
        vars_dict = getattr(app, "vars", {}) or {}
        for name, var in vars_dict.items():
            try:
                float(var.get())
            except Exception:
                continue
            names.append(str(name))
        # Keep a predictable, UI-friendly order while preserving the simulator's
        # native order as much as possible.
        return names
    def _polyhedron_ellipsoid1_current_value_text(self, app, name):
        vars_dict = getattr(app, "vars", {}) or {}
        var = vars_dict.get(name)
        if var is None:
            return ""
        try:
            return str(var.get())
        except Exception:
            return ""
    def _linspace_from_text(self, start_text, stop_text, n_text, *, label="parameter"):
        try:
            start = float(str(start_text).strip())
            stop = float(str(stop_text).strip())
            n = int(float(str(n_text).strip()))
        except Exception as exc:
            raise ValueError(f"Invalid sweep range for {label}: start, stop, and points must be numeric.") from exc
        if n < 1:
            raise ValueError(f"Invalid sweep range for {label}: points must be at least 1.")
        if n == 1:
            return np.asarray([start], dtype=float)
        return np.linspace(start, stop, n, dtype=float)
    def _parse_sweep_values_text(self, values_text, *, label="parameter"):
        """Parse one parameter sweep entry into a 1D float array.
        Accepted forms:
        - single value: ``12``
        - comma/space/semicolon separated values: ``10, 12, 15``
        - linspace shorthand: ``start:stop:n`` such as ``10:20:6``
        """
        text = str(values_text or "").strip()
        if not text:
            raise ValueError(f"No values were provided for {label}.")
        # MATLAB/Python-like linspace shorthand: start:stop:n
        if ":" in text and "," not in text and ";" not in text:
            parts = [p.strip() for p in text.split(":") if p.strip() != ""]
            if len(parts) == 3:
                try:
                    start = float(parts[0])
                    stop = float(parts[1])
                    n = int(float(parts[2]))
                except Exception as exc:
                    raise ValueError(f"Invalid range for {label}. Use start:stop:n, e.g. 10:20:6.") from exc
                if n < 1:
                    raise ValueError(f"Invalid range for {label}: number of points must be at least 1.")
                if n == 1:
                    return np.asarray([start], dtype=float)
                return np.linspace(start, stop, n, dtype=float)
        tokens = [tok for tok in re.split(r"[,;\s]+", text) if tok.strip()]
        values = []
        for tok in tokens:
            try:
                values.append(float(tok))
            except Exception as exc:
                raise ValueError(
                    f"Invalid numeric value for {label}: {tok!r}. "
                    "Use comma-separated values or start:stop:n."
                ) from exc
        if not values:
            raise ValueError(f"No valid values were provided for {label}.")
        # Preserve order while removing exact duplicates.
        unique = []
        seen = set()
        for value in values:
            key = repr(float(value))
            if key not in seen:
                unique.append(float(value))
                seen.add(key)
        return np.asarray(unique, dtype=float)
    def _parse_polyhedron_sweep_names(self, mode, custom_text, current_name):
        allowed = ["tetra", "cube", "octa", "dodeca", "icosa"]
        mode = str(mode or "current").strip().lower()
        current_name = str(current_name or "cube").strip().lower()
        if mode.startswith("all"):
            return allowed
        if mode.startswith("custom"):
            raw = str(custom_text or "").replace(";", ",").split(",")
            names = []
            for item in raw:
                name = item.strip().lower()
                if not name:
                    continue
                if name not in allowed:
                    raise ValueError(f"Unknown polyhedron_name in custom list: {name}. Use one of {', '.join(allowed)}.")
                if name not in names:
                    names.append(name)
            if not names:
                raise ValueError("Custom polyhedron list is empty.")
            return names
        if current_name not in allowed:
            current_name = "cube"
        return [current_name]
    def _compute_polyhedron_ellipsoid1_radial_curves_from_result(
        self,
        app,
        result,
        *,
        npt=500,
        rng=None,
        apply_poisson=None,
    ):
        """Compute combined / ellipsoid-only / polyhedron-only radial profiles.
        The returned curves are generated from the current Polyhedron + 1
        ellipsoid simulator result.  If Poisson is enabled, noise is applied at
        the 2D detector-image level before radial integration, matching the live
        simulation display convention.
        """
        if not isinstance(result, dict):
            raise ValueError("The polyhedron + ellipsoid simulator did not return a valid result.")
        params = dict(result.get("parameters", {}))
        A_ell = np.asarray(result["A1"], dtype=np.complex128)
        A_poly = np.asarray(result["A2"], dtype=np.complex128)
        if A_ell.shape != A_poly.shape:
            raise ValueError(f"A1 and A2 shapes do not match: {A_ell.shape} vs {A_poly.shape}")
        image_shape = A_ell.shape
        app_mask = getattr(app, "mask", None)
        if app_mask is None:
            mask = np.ones(image_shape, dtype=float)
        else:
            mask = np.asarray(app_mask, dtype=float)
            if mask.shape != image_shape:
                try:
                    mask = mask.reshape(image_shape)
                except Exception:
                    mask = np.ones(image_shape, dtype=float)
        ell_weight = float(params.get("ell_weight", 1.0))
        poly_weight = float(params.get("poly_weight", 1.0))
        combined_total_img = np.abs(ell_weight * A_ell + poly_weight * A_poly) ** 2
        ellipse_total_img = np.abs(ell_weight * A_ell) ** 2
        poly_total_img = np.abs(poly_weight * A_poly) ** 2
        poisson_enabled = False
        poisson_peak_counts = None
        poisson_seed = None
        poisson_reference_mode = "off"
        poisson_reference_intensity_used = None
        if apply_poisson is None:
            try:
                apply_poisson = bool(self.poisson_noise_var.get())
            except Exception:
                apply_poisson = False
        if bool(apply_poisson):
            try:
                peak_counts, reference_intensity, reference_mode, _multiplier = self._poisson_reference_state()
                if rng is None:
                    seed = self._parse_poisson_seed()
                    rng = np.random.default_rng(seed)
                    poisson_seed = seed
                else:
                    try:
                        poisson_seed = self._parse_poisson_seed()
                    except Exception:
                        poisson_seed = None
                sim_reference_intensity = self._get_or_set_simulation_poisson_reference(app, combined_total_img)
                if sim_reference_intensity is None:
                    sim_reference_intensity = reference_intensity
                combined_total_img = self._poissonize_image(
                    combined_total_img,
                    peak_counts,
                    rng,
                    reference_intensity=sim_reference_intensity,
                )
                ellipse_total_img = self._poissonize_image(
                    ellipse_total_img,
                    peak_counts,
                    rng,
                    reference_intensity=sim_reference_intensity,
                )
                poly_total_img = self._poissonize_image(
                    poly_total_img,
                    peak_counts,
                    rng,
                    reference_intensity=sim_reference_intensity,
                )
                poisson_enabled = True
                poisson_peak_counts = float(peak_counts)
                poisson_reference_mode = str(reference_mode)
                try:
                    poisson_reference_intensity_used = float(sim_reference_intensity)
                except Exception:
                    poisson_reference_intensity_used = None
            except Exception:
                # Keep the sweep usable even if noise parameters are invalid.
                poisson_enabled = False
                poisson_peak_counts = None
                poisson_seed = None
                poisson_reference_mode = "failed; clean curves shown"
                poisson_reference_intensity_used = None
        combined_img = np.asarray(combined_total_img, dtype=float)
        ellipse_img = np.asarray(ellipse_total_img, dtype=float)
        poly_img = np.asarray(poly_total_img, dtype=float)
        if mask.shape == combined_img.shape:
            combined_img = combined_img * mask
            ellipse_img = ellipse_img * mask
            poly_img = poly_img * mask
        q_u = result.get("q_u", None)
        q_v = result.get("q_v", None)
        valid_mask = _load_default_saxs_valid_mask(combined_img.shape)
        q_u_real, q_v_real, _q_meta = _load_default_q_axes_from_poni(combined_img.shape)
        q_u_use = q_u_real if q_u_real is not None else q_u
        q_v_use = q_v_real if q_v_real is not None else q_v
        q_combo, I_combo, counts_combo, q_label, _std_combo, _sem_combo = _radial_integrate_on_q_grid(
            combined_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=int(npt)
        )
        q_ell, I_ell, _counts_ell, _q_label2, _std_ell, _sem_ell = _radial_integrate_on_q_grid(
            ellipse_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=int(npt)
        )
        q_poly, I_poly, _counts_poly, _q_label3, _std_poly, _sem_poly = _radial_integrate_on_q_grid(
            poly_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=int(npt)
        )
        q_source = ".poni geometry" if q_u_real is not None else "simulator q grid"
        poly_name = str(params.get("polyhedron_name", "cube"))

        def _density_extent(proj, du, dv, scale_factor, translation_u, translation_v):
            n_v, n_u = np.asarray(proj).shape
            width_u = float(n_u) * float(du) * float(scale_factor)
            width_v = float(n_v) * float(dv) * float(scale_factor)
            left = float(translation_u) - 0.5 * width_u
            right = float(translation_u) + 0.5 * width_u
            bottom = float(translation_v) - 0.5 * width_v
            top = float(translation_v) + 0.5 * width_v
            return (left, right, bottom, top)

        def _combined_density_limits(extents, margin_fraction=0.08):
            left = min(e[0] for e in extents)
            right = max(e[1] for e in extents)
            bottom = min(e[2] for e in extents)
            top = max(e[3] for e in extents)
            span = max(right - left, top - bottom, 1.0)
            margin = span * float(margin_fraction)
            cx = 0.5 * (left + right)
            cy = 0.5 * (bottom + top)
            half = 0.5 * span + margin
            return (cx - half, cx + half, cy - half, cy + half)

        density_proj1 = None
        density_proj2 = None
        density_extent1 = None
        density_extent2 = None
        density_limits = None
        try:
            density_proj1 = float(ell_weight) * np.asarray(result["proj1"], dtype=float).copy()
            density_proj2 = float(poly_weight) * np.asarray(result["proj2"], dtype=float).copy()
            density_extent1 = _density_extent(
                result["proj1"], result["du1"], result["dv1"],
                params.get("ell_scale", 1.0), params.get("ell_tu", 0.0), params.get("ell_tv", 0.0)
            )
            density_extent2 = _density_extent(
                result["proj2"], result["du2"], result["dv2"],
                params.get("poly_scale", 1.0), params.get("poly_tu", 0.0), params.get("poly_tv", 0.0)
            )
            density_limits = _combined_density_limits([density_extent1, density_extent2])
        except Exception:
            density_proj1 = None
            density_proj2 = None
            density_extent1 = None
            density_extent2 = None
            density_limits = None

        return {
            "q": q_combo,
            "q_ellipsoid": q_ell,
            "q_polyhedron": q_poly,
            "I_combined": I_combo,
            "I_ellipsoid": I_ell,
            "I_polyhedron": I_poly,
            "counts": counts_combo,
            "q_label": q_label,
            "q_source": q_source,
            "valid_pixels": int(np.sum(valid_mask)),
            "polyhedron_name": poly_name,
            "parameters": params,
            "proj_sum": np.asarray(result.get("proj_sum"), dtype=float).copy() if result.get("proj_sum") is not None else None,
            "density_proj1": density_proj1,
            "density_proj2": density_proj2,
            "density_extent1": density_extent1,
            "density_extent2": density_extent2,
            "density_limits": density_limits,
            "masked_theoretical": combined_img.copy(),
            "poisson_enabled": bool(poisson_enabled),
            "poisson_peak_counts": np.nan if poisson_peak_counts is None else float(poisson_peak_counts),
            "poisson_seed": "" if poisson_seed is None else str(poisson_seed),
            "poisson_reference_mode": poisson_reference_mode,
            "poisson_reference_intensity": np.nan if poisson_reference_intensity_used is None else float(poisson_reference_intensity_used),
        }
    def _nan_moving_average_1d(self, y, window=9):
        arr = np.asarray(y, dtype=float)
        if window is None or int(window) <= 1 or arr.size < 3:
            return arr.copy()
        w = int(window)
        if w % 2 == 0:
            w += 1
        w = max(3, min(w, arr.size if arr.size % 2 == 1 else arr.size - 1))
        if w < 3:
            return arr.copy()
        finite = np.isfinite(arr)
        vals = np.where(finite, arr, 0.0)
        weights = finite.astype(float)
        kernel = np.ones(w, dtype=float)
        num = np.convolve(vals, kernel, mode="same")
        den = np.convolve(weights, kernel, mode="same")
        out = np.full_like(arr, np.nan, dtype=float)
        good = den > 0
        out[good] = num[good] / den[good]
        return out
    def _find_fringe_extrema_positions(self, q, intensity, q_lower_display_limit=DEFAULT_1D_SCALE_FIT_Q_MIN, q_upper_display_limit=DEFAULT_1D_SCALE_FIT_Q_MAX, exclude_first_n=0):
        """Find visible local extrema in a lightly smoothed 1D sweep curve.

        This version is used for the Polyhedron + 1 ellipse sweep preview/CSV.
        It intentionally detects extrema on a smoothed copy of the displayed
        curve, rather than on q^4 I(q).  q^4 weighting is useful for fringe
        scoring, but it can shift the apparent peak position at low q because
        the q^4 factor changes the derivative most strongly near the left edge.

        The first q point is kept by default, and a one-sided boundary test is
        included so that a real maximum close to q = 0.005 is not discarded.
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        I = np.asarray(intensity, dtype=float).reshape(-1)
        if q.size == 0 or I.size == 0:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty
        if q.size != I.size:
            n = min(q.size, I.size)
            q = q[:n]
            I = I[:n]

        good = np.isfinite(q) & np.isfinite(I) & (q > 0)
        if np.isfinite(q_lower_display_limit):
            good &= (q >= float(q_lower_display_limit))
        if np.isfinite(q_upper_display_limit):
            good &= (q <= float(q_upper_display_limit))
        idx = np.where(good)[0]
        if idx.size < 8:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty
        # For sweep marker overlay we do not skip low-q points.  The first
        # oscillation is often the most informative for droplet-size estimation.
        if exclude_first_n and int(exclude_first_n) > 0 and idx.size > int(exclude_first_n) + 8:
            idx = idx[int(exclude_first_n):]

        qg = q[idx]
        yg = I[idx]
        order = np.argsort(qg)
        qg = qg[order]
        yg = yg[order]
        finite = np.isfinite(yg)
        if np.count_nonzero(finite) < 8:
            empty = np.array([], dtype=float)
            return empty, empty, qg, np.full_like(qg, np.nan, dtype=float)

        # Fill rare NaNs without changing the q grid.
        y_fill = yg.astype(float, copy=True)
        if not np.all(finite):
            x = np.arange(y_fill.size, dtype=float)
            if np.count_nonzero(finite) >= 2:
                y_fill[~finite] = np.interp(x[~finite], x[finite], y_fill[finite])
            else:
                y_fill[~finite] = np.nanmedian(y_fill[finite])

        # Work in log-intensity when possible.  log(I) has the same exact
        # extrema as I for positive data, but smoothing in log space is less
        # dominated by the steep SAXS decay and gives better low-q alignment
        # than q^4-weighted smoothing.
        positive = np.isfinite(y_fill) & (y_fill > 0)
        if np.count_nonzero(positive) >= 8:
            floor = np.nanpercentile(y_fill[positive], 0.5)
            if not np.isfinite(floor) or floor <= 0:
                floor = np.nanmin(y_fill[positive])
            floor = max(float(floor), np.finfo(float).tiny)
            y_work = np.log10(np.maximum(y_fill, floor))
        else:
            y_work = y_fill.copy()

        lo = np.nanpercentile(y_work, 1)
        hi = np.nanpercentile(y_work, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = np.nanmin(y_work)
            hi = np.nanmax(y_work)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            empty = np.array([], dtype=float)
            return empty, empty, qg, np.full_like(qg, np.nan, dtype=float)
        y_norm = (y_work - lo) / (hi - lo)

        diffs = np.diff(qg)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        dq = float(np.nanmedian(diffs)) if diffs.size else 0.0
        if not np.isfinite(dq) or dq <= 0:
            dq = 1.0 / max(1, y_norm.size - 1)

        # A narrow smoothing window keeps the first fringe from being displaced.
        # Target width is about 0.00025 in q; clamp to avoid over-smoothing.
        window = int(round(0.00025 / dq)) if dq > 0 else 5
        window = max(5, min(window, 11))
        if window % 2 == 0:
            window += 1
        if window >= y_norm.size:
            window = y_norm.size if y_norm.size % 2 == 1 else y_norm.size - 1
        if window < 5:
            window = 3 if y_norm.size >= 3 else 1

        try:
            from scipy.signal import savgol_filter
            if window >= 5:
                ys_filled = savgol_filter(y_norm, window_length=window, polyorder=2, mode="interp")
            else:
                ys_filled = self._nan_moving_average_1d(y_norm, window=window)
        except Exception:
            # Edge-padded moving average avoids the artificial depression that
            # zero-padded convolution can create at the left boundary.
            if window > 1:
                pad = window // 2
                padded = np.pad(y_norm, pad_width=pad, mode="edge")
                kernel = np.ones(window, dtype=float) / float(window)
                ys_filled = np.convolve(padded, kernel, mode="valid")
            else:
                ys_filled = y_norm.copy()

        finite_s = np.isfinite(ys_filled)
        if np.count_nonzero(finite_s) < 8:
            empty = np.array([], dtype=float)
            return empty, empty, qg, ys_filled
        if not np.all(finite_s):
            x = np.arange(ys_filled.size, dtype=float)
            ys_tmp = ys_filled.copy()
            ys_tmp[~finite_s] = np.interp(x[~finite_s], x[finite_s], ys_filled[finite_s])
            ys_filled = ys_tmp

        def _local_score(i, kind):
            i = int(i)
            half_width = int(round(0.00065 / dq)) if dq > 0 else max(4, ys_filled.size // 90)
            half_width = max(4, min(half_width, 28))
            left = ys_filled[max(0, i - half_width):i]
            right = ys_filled[i + 1:min(ys_filled.size, i + 1 + half_width)]
            vals = []
            if kind == "max":
                if left.size:
                    vals.append(ys_filled[i] - np.nanmin(left))
                if right.size:
                    vals.append(ys_filled[i] - np.nanmin(right))
            else:
                if left.size:
                    vals.append(np.nanmax(left) - ys_filled[i])
                if right.size:
                    vals.append(np.nanmax(right) - ys_filled[i])
            vals = [float(v) for v in vals if np.isfinite(v)]
            if not vals:
                return -np.inf
            # Interior extrema should be visible on both sides; boundary extrema
            # are evaluated one-sided so that q~0.005 maxima can survive.
            return min(vals) if len(vals) >= 2 else vals[0]

        try:
            from scipy.signal import find_peaks
            distance = int(round(0.00045 / dq)) if dq > 0 else max(3, int(round(ys_filled.size / 140)))
            distance = max(2, min(distance, 40))
            peaks, _ = find_peaks(ys_filled, prominence=0.0025, distance=distance, width=1)
            mins, _ = find_peaks(-ys_filled, prominence=0.0025, distance=distance, width=1)
        except Exception:
            peaks = np.array([], dtype=int)
            mins = np.array([], dtype=int)

        # Add derivative sign-change candidates for shallow but visually smooth fringes.
        dy = np.diff(ys_filled)
        sign = np.sign(dy)
        for i in range(1, sign.size):
            if sign[i] == 0:
                sign[i] = sign[i - 1]
        cand_peaks = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
        cand_mins = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1

        # Explicit one-sided boundary test for the first displayed feature.
        edge_span = max(3, min(10, int(round(0.00055 / dq)) if dq > 0 else 6, ys_filled.size - 1))
        boundary_peaks = []
        boundary_mins = []
        if edge_span >= 2:
            first_right = ys_filled[1:edge_span + 1]
            if first_right.size:
                if ys_filled[0] >= np.nanmax(first_right) - 5e-4 and (ys_filled[0] - np.nanmin(first_right)) >= 0.0010:
                    boundary_peaks.append(0)
                if ys_filled[0] <= np.nanmin(first_right) + 5e-4 and (np.nanmax(first_right) - ys_filled[0]) >= 0.0010:
                    boundary_mins.append(0)

        peaks = np.unique(np.concatenate([np.asarray(peaks, dtype=int), cand_peaks.astype(int), np.asarray(boundary_peaks, dtype=int)]))
        mins = np.unique(np.concatenate([np.asarray(mins, dtype=int), cand_mins.astype(int), np.asarray(boundary_mins, dtype=int)]))

        def _filter(indices, kind):
            keep = []
            for i in np.asarray(indices, dtype=int):
                if i < 0 or i >= ys_filled.size:
                    continue
                if i == ys_filled.size - 1:
                    continue
                score = _local_score(i, kind)
                if np.isfinite(score) and score >= 0.0015:
                    keep.append(int(i))
            return np.asarray(sorted(set(keep)), dtype=int)

        peaks = _filter(peaks, "max")
        mins = _filter(mins, "min")

        # Suppress duplicated extrema caused by tiny residual ripples on the
        # smoothed curve.  The retained feature is the strongest local maximum
        # or minimum within a small q-neighborhood.
        min_sep_q = 0.00055

        def _deduplicate_same_kind(indices, kind):
            indices = np.asarray(indices, dtype=int)
            if indices.size <= 1:
                return indices
            indices = indices[np.argsort(qg[indices])]
            selected = []
            group = [int(indices[0])]
            for ii in indices[1:]:
                ii = int(ii)
                if float(qg[ii]) - float(qg[group[-1]]) <= min_sep_q:
                    group.append(ii)
                else:
                    best = max(group, key=lambda jj: _local_score(jj, kind))
                    selected.append(int(best))
                    group = [ii]
            if group:
                best = max(group, key=lambda jj: _local_score(jj, kind))
                selected.append(int(best))
            return np.asarray(sorted(set(selected)), dtype=int)

        peaks = _deduplicate_same_kind(peaks, "max")
        mins = _deduplicate_same_kind(mins, "min")

        def _deduplicate_across_kinds(max_indices, min_indices):
            combined = []
            for ii in np.asarray(max_indices, dtype=int):
                combined.append((int(ii), "max", _local_score(ii, "max")))
            for ii in np.asarray(min_indices, dtype=int):
                combined.append((int(ii), "min", _local_score(ii, "min")))
            combined = [(ii, kind, score) for ii, kind, score in combined if np.isfinite(score)]
            if len(combined) <= 1:
                return np.asarray(max_indices, dtype=int), np.asarray(min_indices, dtype=int)
            combined.sort(key=lambda item: float(qg[item[0]]))
            kept = []
            group = [combined[0]]
            for item in combined[1:]:
                if float(qg[item[0]]) - float(qg[group[-1][0]]) <= min_sep_q:
                    group.append(item)
                else:
                    kept.append(max(group, key=lambda x: x[2]))
                    group = [item]
            if group:
                kept.append(max(group, key=lambda x: x[2]))
            max_keep = sorted({ii for ii, kind, _score in kept if kind == "max"})
            min_keep = sorted({ii for ii, kind, _score in kept if kind == "min"})
            return np.asarray(max_keep, dtype=int), np.asarray(min_keep, dtype=int)

        peaks, mins = _deduplicate_across_kinds(peaks, mins)

        return qg[peaks], qg[mins], qg, ys_filled

    def _first_local_minimum_from_smoothed_curve(self, q_smooth, y_smooth, q_threshold=0.005):
        """Return the first usable local minimum from a smoothed 1D curve.

        This routine is used only for 1st-minimum radius assignment.  It is
        intentionally more permissive than the visual extrema guide finder,
        because the first droplet minimum in a small-droplet + polyhedron curve
        can be shallow after log-smoothing and can be removed by guide-label
        de-duplication.  The order of preference is:

        1. scipy peak search on -y,
        2. derivative negative-to-positive sign changes,
        3. first windowed trough with a small right-side recovery,
        4. conservative early-window trough fallback.
        """
        qg = np.asarray(q_smooth, dtype=float).reshape(-1)
        ys = np.asarray(y_smooth, dtype=float).reshape(-1)
        if qg.size == 0 or ys.size == 0:
            return np.nan
        if qg.size != ys.size:
            n = min(qg.size, ys.size)
            qg = qg[:n]
            ys = ys[:n]
        good = np.isfinite(qg) & np.isfinite(ys) & (qg >= float(q_threshold))
        if np.count_nonzero(good) < 5:
            return np.nan
        qg = qg[good]
        ys = ys[good]
        order = np.argsort(qg)
        qg = qg[order]
        ys = ys[order]
        if qg.size < 5:
            return np.nan

        # Fill any rare non-finite values after sorting.
        finite = np.isfinite(ys)
        if np.count_nonzero(finite) < 5:
            return np.nan
        if not np.all(finite):
            x = np.arange(ys.size, dtype=float)
            ys = ys.copy()
            ys[~finite] = np.interp(x[~finite], x[finite], ys[finite])

        diffs = np.diff(qg)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        dq = float(np.nanmedian(diffs)) if diffs.size else 0.0
        if not np.isfinite(dq) or dq <= 0:
            dq = 1.0 / max(1, qg.size - 1)

        # A tiny score is sufficient here because q_smooth/y_smooth is already
        # the smoothed residual used for feature detection.  The purpose is not
        # to label every guide, but to avoid missing a visually present first
        # trough such as Re=40 nm, Rp=15 nm.
        min_score_threshold = 8e-5
        candidates = []

        try:
            from scipy.signal import find_peaks
            distance = int(round(0.00025 / dq)) if dq > 0 else 2
            distance = max(1, min(distance, 24))
            cand, _props = find_peaks(-ys, prominence=min_score_threshold, distance=distance, width=1)
            candidates.extend([int(i) for i in cand])
        except Exception:
            pass

        # Derivative sign-change candidates.  Smooth very lightly to suppress
        # single-bin numerical jitter without moving the first fringe.
        ys_d = ys.copy()
        try:
            w = int(round(0.00015 / dq)) if dq > 0 else 3
            w = max(3, min(w, 7))
            if w % 2 == 0:
                w += 1
            if w >= 3 and w < ys_d.size:
                pad = w // 2
                ys_d = np.convolve(np.pad(ys_d, pad, mode="edge"), np.ones(w, dtype=float) / float(w), mode="valid")
        except Exception:
            ys_d = ys.copy()
        dy = np.diff(ys_d)
        # Treat very small slopes as zero, then propagate neighboring signs.
        slope_eps = max(1e-7, 0.02 * min_score_threshold)
        sign = np.sign(np.where(np.abs(dy) < slope_eps, 0.0, dy))
        for i in range(1, sign.size):
            if sign[i] == 0:
                sign[i] = sign[i - 1]
        for i in range(sign.size - 2, -1, -1):
            if sign[i] == 0:
                sign[i] = sign[i + 1]
        cand2 = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1
        candidates.extend([int(i) for i in cand2])

        # Boundary trough, retained for completeness.
        edge_span = max(3, min(10, int(round(0.00055 / dq)) if dq > 0 else 6, ys.size - 1))
        if edge_span >= 2:
            right = ys[1:edge_span + 1]
            if right.size and ys[0] <= np.nanmin(right) + 5e-4 and (np.nanmax(right) - ys[0]) >= min_score_threshold:
                candidates.append(0)

        def _min_score(i):
            i = int(i)
            half_width = int(round(0.00075 / dq)) if dq > 0 else 5
            half_width = max(3, min(half_width, 32))
            left = ys[max(0, i - half_width):i]
            right = ys[i + 1:min(ys.size, i + 1 + half_width)]
            vals = []
            if left.size:
                vals.append(np.nanmax(left) - ys[i])
            if right.size:
                vals.append(np.nanmax(right) - ys[i])
            vals = [float(v) for v in vals if np.isfinite(v)]
            if not vals:
                return -np.inf
            return min(vals) if len(vals) >= 2 else vals[0]

        kept = []
        for i in sorted(set(candidates), key=lambda j: float(qg[j]) if 0 <= int(j) < qg.size else np.inf):
            i = int(i)
            if i < 0 or i >= ys.size - 1:
                continue
            score = _min_score(i)
            if np.isfinite(score) and score >= min_score_threshold:
                kept.append(i)
        if kept:
            return float(qg[min(kept, key=lambda j: float(qg[j]))])

        # Windowed trough fallback: scan for the first point that is the minimum
        # within a small q-neighborhood and has at least a slight recovery to
        # the right.  This catches shallow first fringes that fail peak
        # prominence tests but are still visible in the smoothed curve.
        half = int(round(0.0010 / dq)) if dq > 0 else 5
        half = max(3, min(half, max(3, ys.size // 6)))
        recovery_span = int(round(0.0015 / dq)) if dq > 0 else half * 2
        recovery_span = max(half + 1, min(recovery_span, max(half + 1, ys.size // 3)))
        for i in range(1, ys.size - 1):
            left_i = max(0, i - half)
            right_i = min(ys.size, i + half + 1)
            window = ys[left_i:right_i]
            if window.size < 3:
                continue
            if ys[i] <= np.nanmin(window) + 2e-5:
                right = ys[i + 1:min(ys.size, i + 1 + recovery_span)]
                left = ys[max(0, i - recovery_span):i]
                right_recovery = (np.nanmax(right) - ys[i]) if right.size else np.nan
                left_drop = (np.nanmax(left) - ys[i]) if left.size else np.nan
                if (np.isfinite(right_recovery) and right_recovery >= 2e-5) and (not np.isfinite(left_drop) or left_drop >= 0.0):
                    return float(qg[i])

        # Last conservative fallback: take the first broad trough in the low-q
        # analysis region, avoiding endpoints.  This is used only when all
        # stricter local-minimum tests failed.
        q_hi = min(float(qg[-1]), max(float(q_threshold) + 0.018, 0.018))
        region = np.where((qg > float(q_threshold) + 2.0 * dq) & (qg <= q_hi))[0]
        if region.size >= 5:
            j = int(region[np.nanargmin(ys[region])])
            if 0 < j < ys.size - 1:
                right = ys[j + 1:min(ys.size, j + 1 + max(3, half))]
                left = ys[max(0, j - max(3, half)):j]
                if right.size and left.size:
                    if (np.nanmax(right) - ys[j]) >= 1e-5 or (np.nanmax(left) - ys[j]) >= 1e-5:
                        return float(qg[j])
        return np.nan

    def _estimate_sphere_radius_from_first_minimum(self, min_q_values, q_threshold=0.005):
        """Estimate droplet radius from the first visible minimum, assuming a sphere.

        Uses the first minimum of the homogeneous sphere form factor, qR ≈ 4.493.
        q is assumed to be in Å^-1, so the returned radius is reported in both Å and nm.
        The intended use here is the sweep preview for cases where the first droplet
        minimum appears at q >= 0.005 Å^-1, corresponding to R ≲ 90 nm.
        """
        qvals = np.asarray(min_q_values, dtype=float).reshape(-1)
        qvals = qvals[np.isfinite(qvals) & (qvals > 0)]
        if qvals.size == 0:
            return {
                "found": False,
                "first_min_q": np.nan,
                "radius_A": np.nan,
                "radius_nm": np.nan,
                "within_visible_first_min_window": False,
            }
        first_q = float(qvals[0])
        qr_first_min = 4.493409457909064
        radius_A = qr_first_min / first_q
        radius_nm = radius_A * 0.1
        return {
            "found": True,
            "first_min_q": first_q,
            "radius_A": float(radius_A),
            "radius_nm": float(radius_nm),
            "within_visible_first_min_window": bool(first_q >= float(q_threshold)),
        }


    def _compute_oscillation_fft_spectrum(
        self,
        q,
        intensity,
        *,
        q_min=0.005,
        q_max=0.03,
        n_grid=768,
        baseline_q_width=0.0035,
    ):
        """Return FFT spectrum of the oscillatory part of a 1D SAXS curve.

        The curve is first restricted to the visible q-window, interpolated onto
        a uniform q-grid, transformed to log10 intensity, and detrended by a
        broad smooth baseline.  The FFT is then applied to the residual using a
        Hann window.  The frequency axis is in cycles per q-unit; when q is in
        Å^-1, the frequency has units of Å.
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        y_arr = np.asarray(intensity, dtype=float).reshape(-1)
        if q_arr.size == 0 or y_arr.size == 0:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}
        if q_arr.size != y_arr.size:
            n = min(q_arr.size, y_arr.size)
            q_arr = q_arr[:n]
            y_arr = y_arr[:n]

        good = np.isfinite(q_arr) & np.isfinite(y_arr) & (q_arr > 0)
        good &= (q_arr >= float(q_min)) & (q_arr <= float(q_max))
        if np.count_nonzero(good) < 16:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}

        qg = q_arr[good]
        yg = y_arr[good]
        order = np.argsort(qg)
        qg = qg[order]
        yg = yg[order]

        uq, inv = np.unique(qg, return_inverse=True)
        if uq.size < 16:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}
        if uq.size != qg.size:
            y_acc = np.zeros_like(uq, dtype=float)
            counts = np.zeros_like(uq, dtype=float)
            for ii, val in zip(inv, yg):
                if np.isfinite(val):
                    y_acc[ii] += float(val)
                    counts[ii] += 1.0
            ok = counts > 0
            uq = uq[ok]
            yg = y_acc[ok] / counts[ok]
        else:
            uq = qg

        n_grid = int(max(128, min(int(n_grid), 4096)))
        if uq.size < 16 or float(uq[-1]) <= float(uq[0]):
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": empty, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}

        q_uniform = np.linspace(float(uq[0]), float(uq[-1]), n_grid)
        y_interp = np.interp(q_uniform, uq, yg)

        positive = np.isfinite(y_interp) & (y_interp > 0)
        if np.count_nonzero(positive) >= 8:
            floor = np.nanpercentile(y_interp[positive], 0.5)
            if not np.isfinite(floor) or floor <= 0:
                floor = np.nanmin(y_interp[positive])
            floor = max(float(floor), np.finfo(float).tiny)
            y_work = np.log10(np.maximum(y_interp, floor))
        else:
            y_work = y_interp.astype(float, copy=True)

        finite = np.isfinite(y_work)
        if np.count_nonzero(finite) < 16:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": q_uniform, "residual": empty, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}
        if not np.all(finite):
            x = np.arange(y_work.size, dtype=float)
            y_work[~finite] = np.interp(x[~finite], x[finite], y_work[finite])

        dq = float(np.nanmedian(np.diff(q_uniform)))
        baseline_window = int(round(float(baseline_q_width) / dq)) if dq > 0 else 101
        baseline_window = max(31, min(baseline_window, max(31, n_grid // 2 - 1)))
        if baseline_window % 2 == 0:
            baseline_window += 1
        if baseline_window >= n_grid:
            baseline_window = n_grid - 1 if (n_grid - 1) % 2 == 1 else n_grid - 2
        baseline_window = max(7, baseline_window)

        try:
            from scipy.signal import savgol_filter
            polyorder = 3 if baseline_window >= 9 else 2
            baseline = savgol_filter(y_work, window_length=baseline_window, polyorder=polyorder, mode="interp")
        except Exception:
            pad = baseline_window // 2
            padded = np.pad(y_work, pad_width=pad, mode="edge")
            kernel = np.ones(baseline_window, dtype=float) / float(baseline_window)
            baseline = np.convolve(padded, kernel, mode="valid")

        residual = y_work - baseline
        residual = residual - np.nanmean(residual)
        if not np.all(np.isfinite(residual)) or np.nanstd(residual) <= 0:
            empty = np.array([], dtype=float)
            return {"frequency": empty, "amplitude": empty, "q_uniform": q_uniform, "residual": residual, "dominant_frequency": np.nan, "dominant_radius_nm": np.nan}

        window = np.hanning(residual.size)
        fft_vals = np.fft.rfft(residual * window)
        freq = np.fft.rfftfreq(residual.size, d=dq)
        amp_raw = np.abs(fft_vals)
        if window.sum() > 0:
            amp_raw = amp_raw / window.sum()
        if amp_raw.size:
            amp_raw[0] = 0.0
        amp = amp_raw.copy()
        if amp.size and np.nanmax(amp) > 0:
            amp = amp / np.nanmax(amp)

        freq_min = 80.0
        freq_max = min(float(np.nanmax(freq)) if freq.size else 0.0, 1600.0)
        band = np.isfinite(freq) & np.isfinite(amp) & (freq >= freq_min) & (freq <= freq_max)
        if np.count_nonzero(band) >= 2:
            band_idx = np.where(band)[0]
            best_idx = int(band_idx[np.nanargmax(amp[band])])
            dominant_frequency = float(freq[best_idx])
            dominant_radius_nm = float(0.1 * np.pi * dominant_frequency)
            dominant_amplitude = float(amp[best_idx]) if np.isfinite(amp[best_idx]) else np.nan
            dominant_amplitude_raw = float(amp_raw[best_idx]) if np.isfinite(amp_raw[best_idx]) else np.nan
        else:
            dominant_frequency = np.nan
            dominant_radius_nm = np.nan
            dominant_amplitude = np.nan
            dominant_amplitude_raw = np.nan

        return {
            "frequency": freq,
            "amplitude": amp,
            "raw_amplitude": amp_raw,
            "q_uniform": q_uniform,
            "residual": residual,
            "dominant_frequency": dominant_frequency,
            "dominant_radius_nm": dominant_radius_nm,
            "dominant_amplitude": dominant_amplitude,
            "dominant_amplitude_raw": dominant_amplitude_raw,
        }

    def _evaluate_radius_fft_gate(
        self,
        ellipse_spec,
        combined_spec,
        *,
        max_combined_frequency=300.0,
        max_frequency_delta=50.0,
        use_combined_frequency_cutoff=True,
        use_frequency_delta_gate=True,
    ):
        """Return whether a poly+ellipse curve passes the FFT gate for radius assignment.

        The gate is an AND condition among the enabled frequency criteria:
        combined f <= cutoff and |delta f| <= cutoff.  The FFT amplitude ratio
        is still reported as a diagnostic value, but it is not used for pass/fail.
        """
        def _finite_float(obj, key):
            try:
                value = float((obj or {}).get(key, np.nan))
            except Exception:
                value = np.nan
            return value if np.isfinite(value) else np.nan

        f_ell = _finite_float(ellipse_spec, "dominant_frequency")
        f_comb = _finite_float(combined_spec, "dominant_frequency")
        a_ell = _finite_float(ellipse_spec, "dominant_amplitude_raw")
        a_comb = _finite_float(combined_spec, "dominant_amplitude_raw")
        freq_delta = abs(f_comb - f_ell) if np.isfinite(f_comb) and np.isfinite(f_ell) else np.nan
        amp_ratio = a_comb / a_ell if np.isfinite(a_comb) and np.isfinite(a_ell) and a_ell > 0 else np.nan

        use_f_cut = bool(use_combined_frequency_cutoff)
        use_df_gate = bool(use_frequency_delta_gate)

        pass_f_cut = True
        pass_df = True
        reasons = []
        condition_details = []

        # The comparison directions are intentionally explicit here.
        # A condition only participates in the AND gate when its own checkbox is ON.
        if use_f_cut:
            if not np.isfinite(f_comb):
                pass_f_cut = False
                reasons.append("combined_fft_frequency_not_found")
                condition_details.append(f"combined_f=nan <= {float(max_combined_frequency):.6g}: FAIL")
            elif f_comb <= float(max_combined_frequency):
                condition_details.append(f"combined_f={f_comb:.6g} <= {float(max_combined_frequency):.6g}: PASS")
            else:
                pass_f_cut = False
                reasons.append(f"combined_frequency_above_cutoff({f_comb:.6g}>{float(max_combined_frequency):.6g})")
                condition_details.append(f"combined_f={f_comb:.6g} <= {float(max_combined_frequency):.6g}: FAIL")
        else:
            condition_details.append("combined_f cutoff: IGNORED")

        if use_df_gate:
            if not np.isfinite(f_ell):
                pass_df = False
                reasons.append("ellipse_fft_frequency_not_found")
                condition_details.append(f"abs_delta_f=nan <= {float(max_frequency_delta):.6g}: FAIL")
            elif not np.isfinite(f_comb):
                pass_df = False
                if "combined_fft_frequency_not_found" not in reasons:
                    reasons.append("combined_fft_frequency_not_found")
                condition_details.append(f"abs_delta_f=nan <= {float(max_frequency_delta):.6g}: FAIL")
            elif not np.isfinite(freq_delta):
                pass_df = False
                reasons.append("frequency_delta_not_available")
                condition_details.append(f"abs_delta_f=nan <= {float(max_frequency_delta):.6g}: FAIL")
            elif freq_delta <= float(max_frequency_delta):
                condition_details.append(f"abs_delta_f={freq_delta:.6g} <= {float(max_frequency_delta):.6g}: PASS")
            else:
                pass_df = False
                reasons.append(f"frequency_mismatch_too_large({freq_delta:.6g}>{float(max_frequency_delta):.6g})")
                condition_details.append(f"abs_delta_f={freq_delta:.6g} <= {float(max_frequency_delta):.6g}: FAIL")
        else:
            condition_details.append("abs_delta_f cutoff: IGNORED")

        # If the FFT gate itself is enabled but all subconditions are unchecked,
        # pass-by-default is too easy to misread. Treat it as fail with a clear reason.
        if not use_f_cut and not use_df_gate:
            passes = False
            reasons.append("no_fft_gate_condition_enabled")
        else:
            passes = bool((not use_f_cut or pass_f_cut) and (not use_df_gate or pass_df))

        return {
            "passes": passes,
            "reason": "pass" if passes else ";".join(reasons),
            "gate_expression": " AND ".join(condition_details),
            "ellipse_frequency": f_ell,
            "combined_frequency": f_comb,
            "frequency_delta": freq_delta,
            "ellipse_dominant_amplitude_raw": a_ell,
            "combined_dominant_amplitude_raw": a_comb,
            "amplitude_ratio_combined_over_ellipse": amp_ratio,
            "max_combined_frequency": float(max_combined_frequency),
            "max_frequency_delta": float(max_frequency_delta),
            "use_combined_frequency_cutoff": int(use_f_cut),
            "use_frequency_delta_gate": int(use_df_gate),
            "pass_combined_frequency_cutoff": int(pass_f_cut),
            "pass_frequency_delta": int(pass_df),
            "active_fft_gate_condition_count": int(use_f_cut) + int(use_df_gate),
        }

    def _plot_smoothed_extrema_markers(
        self,
        ax,
        q,
        intensity,
        *,
        q_min=None,
        q_max=None,
        color=None,
        label_prefix="curve",
        log_y=False,
        max_features=10,
        marker_size=38,
        legend=True,
        draw_guides=True,
        label_y_frac=0.96,
        show_sphere_radius=False,
        sphere_radius_q_threshold=0.005,
        radius_error_reference_nm=None,
        radius_error_reference_label="Re",
    ):
        """Overlay smoothed-curve extrema as vertical dashed q-position guides.

        Extrema are detected on a lightly smoothed copy of the displayed 1D
        curve.  Nothing is marked on the raw curve points themselves; the
        overlay is only a q-position annotation for comparing fringe survival
        between 1 ellipse and 1 ellipse + polyhedron.  Optionally, the first
        visible minimum can also be converted to a sphere-equivalent droplet
        radius using qR ≈ 4.493.
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        y_arr = np.asarray(intensity, dtype=float).reshape(-1)
        if q_arr.size == 0 or y_arr.size == 0:
            return {"max_q": np.array([], dtype=float), "min_q": np.array([], dtype=float)}
        if q_arr.size != y_arr.size:
            n = min(q_arr.size, y_arr.size)
            q_arr = q_arr[:n]
            y_arr = y_arr[:n]
        q_lo = DEFAULT_1D_SCALE_FIT_Q_MIN if q_min is None else float(q_min)
        q_hi = DEFAULT_1D_SCALE_FIT_Q_MAX if q_max is None else float(q_max)
        max_q, min_q, _q_smooth, _y_smooth = self._find_fringe_extrema_positions(
            q_arr,
            y_arr,
            q_lower_display_limit=q_lo,
            q_upper_display_limit=q_hi,
            exclude_first_n=0,
        )
        max_q = np.asarray(max_q, dtype=float)
        min_q = np.asarray(min_q, dtype=float)
        if max_features is not None and int(max_features) > 0:
            max_q = max_q[: int(max_features)]
            min_q = min_q[: int(max_features)]

        # Keep returned values restricted to the visible data range.
        finite = np.isfinite(q_arr) & np.isfinite(y_arr)
        if log_y:
            finite &= y_arr > 0
        if np.count_nonzero(finite) < 2:
            return {"max_q": max_q, "min_q": min_q}
        order = np.argsort(q_arr[finite])
        q_good = q_arr[finite][order]
        max_good = np.isfinite(max_q) & (max_q >= q_good[0]) & (max_q <= q_good[-1])
        min_good = np.isfinite(min_q) & (min_q >= q_good[0]) & (min_q <= q_good[-1])
        max_q = max_q[max_good]
        min_q = min_q[min_good]

        if bool(show_sphere_radius):
            fallback_min_q = self._first_local_minimum_from_smoothed_curve(_q_smooth, _y_smooth, q_threshold=sphere_radius_q_threshold)
            if np.isfinite(fallback_min_q):
                if min_q.size == 0 or float(fallback_min_q) < float(np.nanmin(min_q)) - 1e-9:
                    min_q = np.asarray(sorted(np.concatenate([min_q, np.asarray([fallback_min_q], dtype=float)])), dtype=float)

        if not bool(draw_guides):
            radius_info = self._estimate_sphere_radius_from_first_minimum(min_q, q_threshold=sphere_radius_q_threshold) if bool(show_sphere_radius) else None
            return {"max_q": max_q, "min_q": min_q, "sphere_radius": radius_info}

        # Axis-fraction y coordinates keep labels readable on both linear and
        # log intensity axes.  The two curve calls pass different y positions,
        # reducing overlap when their extrema are nearly coincident in q.
        text_transform = ax.get_xaxis_transform()
        color = color if color is not None else "black"
        base_y = max(0.05, min(0.98, float(label_y_frac)))

        def _draw(q_values, kind, y_offset, line_alpha, text_alpha):
            if q_values.size == 0:
                return
            for j, qv in enumerate(np.asarray(q_values, dtype=float)):
                if not np.isfinite(qv):
                    continue
                y_frac = base_y - y_offset - 0.055 * (j % 2)
                y_frac = max(0.06, min(0.98, y_frac))
                ax.axvline(
                    float(qv),
                    color=color,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=line_alpha,
                    zorder=2,
                )
                prefix = "max" if kind == "max" else "min"
                ax.text(
                    float(qv),
                    y_frac,
                    f"{prefix} {float(qv):.4f}",
                    transform=text_transform,
                    rotation=90,
                    ha="right",
                    va="top",
                    fontsize=5.8,
                    color=color,
                    alpha=text_alpha,
                    clip_on=False,
                    zorder=7,
                )

        _draw(max_q, "max", 0.00, 0.42, 0.95)
        _draw(min_q, "min", 0.17, 0.30, 0.82)

        radius_info = None
        if bool(show_sphere_radius):
            radius_info = self._estimate_sphere_radius_from_first_minimum(min_q, q_threshold=sphere_radius_q_threshold)
            summary_y = max(0.04, min(0.98, base_y - 0.30))
            if radius_info.get("found", False):
                msg = (
                    f"{label_prefix}: 1st min {radius_info['first_min_q']:.4f} -> "
                    f"R~{radius_info['radius_nm']:.1f} nm"
                )
                try:
                    ref_nm = float(radius_error_reference_nm)
                except Exception:
                    ref_nm = np.nan
                if np.isfinite(ref_nm) and ref_nm > 0 and np.isfinite(radius_info.get("radius_nm", np.nan)):
                    err_nm = float(radius_info["radius_nm"]) - ref_nm
                    err_pct = 100.0 * err_nm / ref_nm
                    msg += f"; err={err_nm:+.1f} nm ({err_pct:+.1f}%)"
            else:
                msg = f"{label_prefix}: 1st min not found in q >= {float(sphere_radius_q_threshold):.4f}"
            ax.text(
                0.015,
                summary_y,
                msg,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.1,
                color=color,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, alpha=0.65),
                zorder=8,
            )
        return {"max_q": max_q, "min_q": min_q, "sphere_radius": radius_info}

    def _polyhedron_ellipsoid1_ellipse_size_reference_nm(self, params):
        """Return ellipse size parameters in nm for radius-error reporting.

        ell_scale is interpreted as the semi-major axis in Angstrom.  ell_b is
        the minor/major ratio.  For non-circular ellipses, the reference radius
        used for first-minimum error is the area-equivalent projected radius,
        R_ref = sqrt(a*b).  For circular droplets this is simply Re.
        """
        params = params or {}
        def _to_float(value, default=np.nan):
            try:
                out = float(value)
                return out if np.isfinite(out) else default
            except Exception:
                return default
        ell_scale = _to_float(params.get("ell_scale", np.nan))
        ell_b = _to_float(params.get("ell_b", 1.0), 1.0)
        if not np.isfinite(ell_scale):
            return {"a_nm": np.nan, "b_nm": np.nan, "reference_radius_nm": np.nan, "reference_mode": "unavailable"}
        a_nm = abs(float(ell_scale)) / 10.0
        b_nm = abs(float(ell_scale) * float(ell_b)) / 10.0 if np.isfinite(ell_b) else np.nan
        if not np.isfinite(b_nm):
            b_nm = a_nm
        ref_nm = float(np.sqrt(max(a_nm, 0.0) * max(b_nm, 0.0))) if a_nm > 0 and b_nm > 0 else np.nan
        mode = "radius" if np.isclose(a_nm, b_nm, rtol=1e-4, atol=1e-9) else "area_equivalent_radius_sqrt_ab"
        return {"a_nm": float(a_nm), "b_nm": float(b_nm), "reference_radius_nm": float(ref_nm), "reference_mode": mode}

    def _format_polyhedron_ellipsoid1_size_title(self, params, selected_params=None, swept_values=None, max_extra=3, multiline=True):
        """Return compact nm-size labels for Polyhedron + 1 ellipsoid sweep preview titles.

        The simulator's real-space scale parameters are interpreted in Å.
        Therefore, ell_scale / 10 gives the ellipsoid semi-major axis in nm,
        ell_scale * ell_b / 10 gives the semi-minor axis in nm, and
        poly_scale * poly_radius / 10 gives the polyhedron circumradius in nm.
        The displayed labels are intentionally short for presentation figures:
        Re=a/b nm and Rp=... nm.
        """
        params = params or {}
        selected_params = list(selected_params or [])
        swept_values = swept_values or {}

        def _to_float(value, default=np.nan):
            try:
                out = float(value)
                return out if np.isfinite(out) else default
            except Exception:
                return default

        def _fmt_nm(value):
            value = _to_float(value)
            if not np.isfinite(value):
                return "nan"
            value = abs(value)
            if value >= 100:
                return f"{value:.0f}"
            if value >= 10:
                return f"{value:.1f}"
            if value >= 1:
                return f"{value:.2f}"
            return f"{value:.3g}"

        def _fmt_param_value(value):
            try:
                return f"{float(value):.4g}"
            except Exception:
                return str(value)

        size_parts = []
        ell_scale = _to_float(params.get("ell_scale", np.nan))
        ell_b = _to_float(params.get("ell_b", 1.0), 1.0)
        if np.isfinite(ell_scale):
            ell_a_nm = abs(ell_scale) / 10.0
            ell_b_nm = abs(ell_scale * ell_b) / 10.0
            if np.isclose(ell_a_nm, ell_b_nm, rtol=1e-4, atol=1e-9):
                size_parts.append(f"Re={_fmt_nm(ell_a_nm)} nm")
            else:
                size_parts.append(f"Re={_fmt_nm(ell_a_nm)}/{_fmt_nm(ell_b_nm)} nm")

        poly_scale = _to_float(params.get("poly_scale", np.nan))
        poly_radius = _to_float(params.get("poly_radius", 1.0), 1.0)
        if np.isfinite(poly_scale):
            poly_rcirc_nm = abs(poly_scale * poly_radius) / 10.0
            size_parts.append(f"Rp={_fmt_nm(poly_rcirc_nm)} nm")

        if not size_parts:
            size_parts = ["current sizes"]

        # Parameters already represented in the size labels are skipped here.
        size_params = {"ell_scale", "ell_b", "poly_scale", "poly_radius"}
        extra_parts = []
        for name in selected_params:
            if name in size_params:
                continue
            if name in swept_values:
                extra_parts.append(f"{name}={_fmt_param_value(swept_values[name])}")

        first_line = "; ".join(size_parts)
        if not extra_parts:
            return first_line
        extra_text = ", ".join(extra_parts[:max_extra])
        if len(extra_parts) > max_extra:
            extra_text += ", ..."
        if multiline:
            return first_line + "\n" + extra_text
        return first_line + "; " + extra_text

    def _save_polyhedron_ellipsoid1_sweep_results(
        self,
        output_dir,
        *,
        curves,
        conditions,
        sweep_config,
        q_label,
        q_source,
        npt,
        poisson_metadata,
        show_extrema_markers=True,
        show_first_min_radius=False,
        show_fft_spectrum=False,
        radius_fft_gate_config=None,
    ):
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = f"polyhedron_ellipsoid1_sweep_{timestamp}"
        npz_path = out / f"{stem}.npz"
        csv_path = out / f"{stem}.csv"
        json_path = out / f"{stem}_summary.json"
        extrema_csv_path = out / f"{stem}_extrema_positions.csv"
        radius_csv_path = out / f"{stem}_droplet_radius_estimates.csv"
        fft_gate_csv_path = out / f"{stem}_radius_fft_gate.csv"
        fft_csv_path = out / f"{stem}_oscillation_frequency_spectra.csv"
        png_path = out / f"{stem}_bundle_preview.png"
        q = np.asarray(curves[0]["q"], dtype=float)
        I_combined = np.vstack([np.asarray(c["I_combined"], dtype=float) for c in curves])
        I_ellipsoid = np.vstack([np.asarray(c["I_ellipsoid"], dtype=float) for c in curves])
        I_polyhedron = np.vstack([np.asarray(c["I_polyhedron"], dtype=float) for c in curves])
        counts = np.asarray(curves[0]["counts"], dtype=float)
        conditions_json = json.dumps(conditions, ensure_ascii=False, indent=2)
        config_json = json.dumps(sweep_config, ensure_ascii=False, indent=2)
        selected_params = list(sweep_config.get("sweep_parameters", []))
        np.savez(
            npz_path,
            q=q,
            I_1ellipse_plus_polyhedron=I_combined,
            I_1ellipse_only=I_ellipsoid,
            I_polyhedron_only=I_polyhedron,
            counts=counts,
            q_label=str(q_label),
            q_source=str(q_source),
            npt=int(npt),
            conditions_json=conditions_json,
            sweep_config_json=config_json,
            poisson_metadata_json=json.dumps(poisson_metadata, ensure_ascii=False, indent=2),
        )

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["condition_index", "polyhedron_name", "sweep_parameters_json"]
            header.extend([f"param_{name}" for name in selected_params])
            header.extend([q_label, "I_1ellipse_plus_polyhedron", "I_1ellipse_only", "I_polyhedron_only", "counts", "poisson_enabled"])
            writer.writerow(header)
            for idx, (condition, curve) in enumerate(zip(conditions, curves)):
                poly = condition.get("polyhedron_name", "")
                swept_values = condition.get("sweep_values", {}) or {}
                swept_json = json.dumps(swept_values, ensure_ascii=False, sort_keys=True)
                param_columns = [swept_values.get(name, "") for name in selected_params]
                for a, b, c, d, n in zip(curve["q"], curve["I_combined"], curve["I_ellipsoid"], curve["I_polyhedron"], curve["counts"]):
                    writer.writerow([idx, poly, swept_json, *param_columns, a, b, c, d, n, int(bool(curve.get("poisson_enabled", False)))])

        extrema_rows = []
        if bool(show_extrema_markers):
            for idx, (condition, curve) in enumerate(zip(conditions, curves)):
                for curve_key, curve_label in (("I_ellipsoid", "1 ellipse only"), ("I_combined", "1 ellipse + polyhedron")):
                    try:
                        max_q, min_q, _q_smooth, _y_smooth = self._find_fringe_extrema_positions(
                            curve["q"],
                            curve[curve_key],
                            q_lower_display_limit=0.005,
                            q_upper_display_limit=0.03,
                            exclude_first_n=0,
                        )
                    except Exception:
                        max_q = np.array([], dtype=float)
                        min_q = np.array([], dtype=float)
                    for order_i, qv in enumerate(np.asarray(max_q, dtype=float), start=1):
                        if np.isfinite(qv):
                            extrema_rows.append([idx, condition.get("polyhedron_name", ""), curve_label, "max", order_i, float(qv)])
                    for order_i, qv in enumerate(np.asarray(min_q, dtype=float), start=1):
                        if np.isfinite(qv):
                            extrema_rows.append([idx, condition.get("polyhedron_name", ""), curve_label, "min", order_i, float(qv)])
        with extrema_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["condition_index", "polyhedron_name", "curve", "extremum", "feature_order", str(q_label)])
            writer.writerows(extrema_rows)

        radius_fft_gate_config = dict(radius_fft_gate_config or {})
        use_radius_fft_gate = bool(radius_fft_gate_config.get("enabled", False))
        gate_max_freq = float(radius_fft_gate_config.get("max_combined_frequency", 300.0))
        gate_max_delta = float(radius_fft_gate_config.get("max_frequency_delta", 50.0))
        gate_use_f_cut = bool(radius_fft_gate_config.get("use_combined_frequency_cutoff", True))
        gate_use_df = bool(radius_fft_gate_config.get("use_frequency_delta_gate", True))
        need_fft_for_gate_or_plot = True

        fft_cache = {}
        if need_fft_for_gate_or_plot:
            for idx, (condition, curve) in enumerate(zip(conditions, curves)):
                for curve_key, curve_label in (("I_ellipsoid", "1 ellipse only"), ("I_combined", "1 ellipse + polyhedron")):
                    try:
                        spec = self._compute_oscillation_fft_spectrum(
                            curve["q"],
                            curve[curve_key],
                            q_min=0.005,
                            q_max=0.03,
                        )
                    except Exception:
                        spec = {
                            "frequency": np.array([], dtype=float),
                            "amplitude": np.array([], dtype=float),
                            "raw_amplitude": np.array([], dtype=float),
                            "dominant_frequency": np.nan,
                            "dominant_radius_nm": np.nan,
                            "dominant_amplitude": np.nan,
                            "dominant_amplitude_raw": np.nan,
                        }
                    fft_cache[(idx, curve_key)] = spec

        fft_gate_by_idx = {}
        fft_gate_rows = []
        if use_radius_fft_gate and bool(show_first_min_radius):
            for idx, (condition, curve) in enumerate(zip(conditions, curves)):
                gate = self._evaluate_radius_fft_gate(
                    fft_cache.get((idx, "I_ellipsoid"), {}),
                    fft_cache.get((idx, "I_combined"), {}),
                    max_combined_frequency=gate_max_freq,
                    max_frequency_delta=gate_max_delta,
                    use_combined_frequency_cutoff=gate_use_f_cut,
                    use_frequency_delta_gate=gate_use_df,
                )
                fft_gate_by_idx[idx] = gate
                fft_gate_rows.append([
                    idx,
                    condition.get("polyhedron_name", ""),
                    int(bool(gate.get("passes", False))),
                    gate.get("reason", ""),
                    gate.get("combined_frequency", np.nan) if np.isfinite(gate.get("combined_frequency", np.nan)) else "",
                    gate.get("ellipse_frequency", np.nan) if np.isfinite(gate.get("ellipse_frequency", np.nan)) else "",
                    gate.get("frequency_delta", np.nan) if np.isfinite(gate.get("frequency_delta", np.nan)) else "",
                    gate.get("amplitude_ratio_combined_over_ellipse", np.nan) if np.isfinite(gate.get("amplitude_ratio_combined_over_ellipse", np.nan)) else "",
                    gate_max_freq,
                    gate_max_delta,
                    int(gate_use_f_cut),
                    int(gate_use_df),
                    gate.get("pass_combined_frequency_cutoff", ""),
                    gate.get("pass_frequency_delta", ""),
                    gate.get("active_fft_gate_condition_count", ""),
                    gate.get("gate_expression", ""),
                ])
        with fft_gate_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "condition_index",
                "polyhedron_name",
                "fft_gate_pass",
                "fft_gate_reason",
                "combined_dominant_frequency",
                "ellipse_dominant_frequency",
                "abs_frequency_delta",
                "dominant_raw_amplitude_ratio_combined_over_ellipse",
                "max_allowed_combined_frequency",
                "max_allowed_frequency_delta",
                "use_combined_frequency_cutoff",
                "use_frequency_delta_gate",
                "pass_combined_frequency_cutoff",
                "pass_frequency_delta",
                "active_fft_gate_condition_count",
                "gate_expression",
            ])
            writer.writerows(fft_gate_rows)

        def _strict_gate_pass(gate):
            if not use_radius_fft_gate:
                return True
            if not bool((gate or {}).get("passes", False)):
                return False
            if gate_use_f_cut and int((gate or {}).get("pass_combined_frequency_cutoff", 0)) != 1:
                return False
            if gate_use_df and int((gate or {}).get("pass_frequency_delta", 0)) != 1:
                return False
            if (not gate_use_f_cut) and (not gate_use_df):
                return False
            return True

        def _first_min_radius_info_for_curve(curve, curve_key):
            try:
                _max_q, min_q, _q_smooth, _y_smooth = self._find_fringe_extrema_positions(
                    curve["q"],
                    curve[curve_key],
                    q_lower_display_limit=0.005,
                    q_upper_display_limit=0.03,
                    exclude_first_n=0,
                )
                fallback_min_q = self._first_local_minimum_from_smoothed_curve(_q_smooth, _y_smooth, q_threshold=0.005)
                min_q_for_radius = np.asarray(min_q, dtype=float)
                if np.isfinite(fallback_min_q):
                    if min_q_for_radius.size == 0 or float(fallback_min_q) < float(np.nanmin(min_q_for_radius)) - 1e-9:
                        min_q_for_radius = np.asarray(sorted(np.concatenate([min_q_for_radius, np.asarray([fallback_min_q], dtype=float)])), dtype=float)
                return self._estimate_sphere_radius_from_first_minimum(min_q_for_radius, q_threshold=0.005)
            except Exception:
                return {"found": False, "first_min_q": np.nan, "radius_A": np.nan, "radius_nm": np.nan, "within_visible_first_min_window": False}

        def _radius_delta_vs_ellipse_calculated(curve, radius_allowed=True):
            if not bool(radius_allowed):
                return {"found": False, "delta_nm": np.nan, "delta_percent": np.nan, "ellipse_radius_nm": np.nan, "combined_radius_nm": np.nan}
            ell_info = _first_min_radius_info_for_curve(curve, "I_ellipsoid")
            comb_info = _first_min_radius_info_for_curve(curve, "I_combined")
            ell_r = float(ell_info.get("radius_nm", np.nan)) if ell_info.get("found", False) else np.nan
            comb_r = float(comb_info.get("radius_nm", np.nan)) if comb_info.get("found", False) else np.nan
            if np.isfinite(ell_r) and ell_r > 0 and np.isfinite(comb_r):
                delta_nm = comb_r - ell_r
                delta_pct = 100.0 * delta_nm / ell_r
                return {"found": True, "delta_nm": float(delta_nm), "delta_percent": float(delta_pct), "ellipse_radius_nm": float(ell_r), "combined_radius_nm": float(comb_r)}
            return {"found": False, "delta_nm": np.nan, "delta_percent": np.nan, "ellipse_radius_nm": ell_r, "combined_radius_nm": comb_r}

        radius_rows = []
        if bool(show_first_min_radius):
            for idx, (condition, curve) in enumerate(zip(conditions, curves)):
                gate = fft_gate_by_idx.get(idx, {"passes": True, "reason": "gate_disabled"})
                radius_allowed_by_gate = _strict_gate_pass(gate)
                params_for_error = curve.get("parameters", {}) or condition.get("parameters", {}) or {}
                ellipse_ref_info = self._polyhedron_ellipsoid1_ellipse_size_reference_nm(params_for_error)
                ellipse_a_nm = ellipse_ref_info.get("a_nm", np.nan)
                ellipse_b_nm = ellipse_ref_info.get("b_nm", np.nan)
                ellipse_ref_nm = ellipse_ref_info.get("reference_radius_nm", np.nan)
                ellipse_ref_mode = ellipse_ref_info.get("reference_mode", "")
                for curve_key, curve_label in (("I_ellipsoid", "1 ellipse only"), ("I_combined", "1 ellipse + polyhedron")):
                    radius_info = _first_min_radius_info_for_curve(curve, curve_key)
                    radius_applied = bool(radius_allowed_by_gate and radius_info.get("found", False))
                    radius_nm_applied = float(radius_info.get("radius_nm", np.nan)) if radius_applied and np.isfinite(radius_info.get("radius_nm", np.nan)) else np.nan
                    radius_error_nm = radius_nm_applied - float(ellipse_ref_nm) if np.isfinite(radius_nm_applied) and np.isfinite(ellipse_ref_nm) and float(ellipse_ref_nm) > 0 else np.nan
                    radius_error_pct = 100.0 * radius_error_nm / float(ellipse_ref_nm) if np.isfinite(radius_error_nm) and np.isfinite(ellipse_ref_nm) and float(ellipse_ref_nm) > 0 else np.nan
                    delta_vs_calc = _radius_delta_vs_ellipse_calculated(curve, radius_allowed=radius_allowed_by_gate)
                    calc_ell_r = delta_vs_calc.get("ellipse_radius_nm", np.nan)
                    calc_comb_r = delta_vs_calc.get("combined_radius_nm", np.nan)
                    calc_delta_nm = delta_vs_calc.get("delta_nm", np.nan)
                    calc_delta_pct = delta_vs_calc.get("delta_percent", np.nan)
                    radius_rows.append([
                        idx,
                        condition.get("polyhedron_name", ""),
                        curve_label,
                        float(radius_info.get("first_min_q", np.nan)) if np.isfinite(radius_info.get("first_min_q", np.nan)) else "",
                        float(radius_info.get("radius_A", np.nan)) if radius_applied and np.isfinite(radius_info.get("radius_A", np.nan)) else "",
                        radius_nm_applied if np.isfinite(radius_nm_applied) else "",
                        float(calc_ell_r) if np.isfinite(calc_ell_r) else "",
                        float(calc_comb_r) if np.isfinite(calc_comb_r) else "",
                        float(calc_delta_nm) if np.isfinite(calc_delta_nm) else "",
                        float(calc_delta_pct) if np.isfinite(calc_delta_pct) else "",
                        float(ellipse_a_nm) if np.isfinite(ellipse_a_nm) else "",
                        float(ellipse_b_nm) if np.isfinite(ellipse_b_nm) else "",
                        float(ellipse_ref_nm) if np.isfinite(ellipse_ref_nm) else "",
                        ellipse_ref_mode,
                        float(radius_error_nm) if np.isfinite(radius_error_nm) else "",
                        float(radius_error_pct) if np.isfinite(radius_error_pct) else "",
                        int(bool(radius_info.get("found", False))),
                        int(bool(radius_info.get("within_visible_first_min_window", False))),
                        int(radius_applied),
                        int(use_radius_fft_gate),
                        int(bool(gate.get("passes", False))) if use_radius_fft_gate else "",
                        gate.get("reason", "gate_disabled"),
                        gate.get("combined_frequency", np.nan) if np.isfinite(gate.get("combined_frequency", np.nan)) else "",
                        gate.get("ellipse_frequency", np.nan) if np.isfinite(gate.get("ellipse_frequency", np.nan)) else "",
                        gate.get("frequency_delta", np.nan) if np.isfinite(gate.get("frequency_delta", np.nan)) else "",
                        gate.get("amplitude_ratio_combined_over_ellipse", np.nan) if np.isfinite(gate.get("amplitude_ratio_combined_over_ellipse", np.nan)) else "",
                        gate.get("pass_combined_frequency_cutoff", ""),
                        gate.get("pass_frequency_delta", ""),
                        gate.get("gate_expression", ""),
                    ])
        with radius_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "condition_index",
                "polyhedron_name",
                "curve",
                f"first_min_{str(q_label)}",
                "sphere_radius_A_applied",
                "sphere_radius_nm_applied",
                "ellipse_calculated_radius_nm_from_first_min",
                "combined_calculated_radius_nm_from_first_min",
                "combined_minus_ellipse_calculated_radius_nm",
                "combined_minus_ellipse_calculated_radius_percent",
                "ellipse_a_nm",
                "ellipse_b_nm",
                "ellipse_reference_radius_nm",
                "ellipse_reference_mode",
                "radius_error_nm_vs_ellipse_reference",
                "radius_error_percent_vs_ellipse_reference",
                "first_min_found",
                "first_min_q_ge_0.005",
                "radius_applied",
                "fft_gate_enabled",
                "fft_gate_pass",
                "fft_gate_reason",
                "combined_dominant_frequency",
                "ellipse_dominant_frequency",
                "abs_frequency_delta",
                "dominant_raw_amplitude_ratio_combined_over_ellipse",
                "pass_combined_frequency_cutoff",
                "pass_frequency_delta",
                "gate_expression",
            ])
            writer.writerows(radius_rows)

        fft_rows = []
        if need_fft_for_gate_or_plot:
            for idx, (condition, curve) in enumerate(zip(conditions, curves)):
                for curve_key, curve_label in (("I_ellipsoid", "1 ellipse only"), ("I_combined", "1 ellipse + polyhedron")):
                    spec = fft_cache.get((idx, curve_key), {})
                    freq = np.asarray(spec.get("frequency", []), dtype=float)
                    amp = np.asarray(spec.get("amplitude", []), dtype=float)
                    raw_amp = np.asarray(spec.get("raw_amplitude", np.full_like(amp, np.nan)), dtype=float)
                    if raw_amp.size != amp.size:
                        raw_amp = np.full_like(amp, np.nan, dtype=float)
                    dom_f = spec.get("dominant_frequency", np.nan)
                    dom_r = spec.get("dominant_radius_nm", np.nan)
                    dom_amp = spec.get("dominant_amplitude", np.nan)
                    dom_amp_raw = spec.get("dominant_amplitude_raw", np.nan)
                    for fval, aval, raw_aval in zip(freq, amp, raw_amp):
                        if np.isfinite(fval) and np.isfinite(aval):
                            fft_rows.append([
                                idx,
                                condition.get("polyhedron_name", ""),
                                curve_label,
                                float(fval),
                                float(aval),
                                float(raw_aval) if np.isfinite(raw_aval) else "",
                                float(dom_f) if np.isfinite(dom_f) else "",
                                float(dom_amp) if np.isfinite(dom_amp) else "",
                                float(dom_amp_raw) if np.isfinite(dom_amp_raw) else "",
                                float(dom_r) if np.isfinite(dom_r) else "",
                            ])
        with fft_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "condition_index",
                "polyhedron_name",
                "curve",
                "oscillation_frequency_cycles_per_q_unit",
                "normalized_amplitude",
                "raw_amplitude",
                "dominant_frequency_cycles_per_q_unit",
                "dominant_normalized_amplitude",
                "dominant_raw_amplitude",
                "rough_radius_nm_from_frequency",
            ])
            writer.writerows(fft_rows)

        preview_png_paths = []
        try:
            bundles_per_page = 3
            total_conditions = len(curves)
            noise_label = "Poisson ON" if bool(poisson_metadata.get("enabled", False)) else "clean"
            for page_idx, start_idx in enumerate(range(0, total_conditions, bundles_per_page), start=1):
                stop_idx = min(start_idx + bundles_per_page, total_conditions)
                chunk_curves = curves[start_idx:stop_idx]
                chunk_conditions = conditions[start_idx:stop_idx]
                current_png_path = png_path if total_conditions <= bundles_per_page else out / f"{stem}_bundle_preview_page{page_idx:02d}.png"
                include_fft_row = True
                if include_fft_row:
                    fig = Figure(figsize=(18.0, 11.3), dpi=130)
                    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.25, 1.05], hspace=0.34, wspace=0.34)
                    page_title = f"Polyhedron + 1 ellipse sweep bundles + FFT ({noise_label}) | conditions {start_idx + 1}-{stop_idx} of {total_conditions}"
                else:
                    fig = Figure(figsize=(18.0, 8.5), dpi=130)
                    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.25], hspace=0.26, wspace=0.34)
                    page_title = f"Polyhedron + 1 ellipse sweep bundles ({noise_label}) | conditions {start_idx + 1}-{stop_idx} of {total_conditions}"
                fig.suptitle(page_title, fontsize=13)

                for local_idx, (curve, cond) in enumerate(zip(chunk_curves, chunk_conditions)):
                    col0 = 2 * local_idx
                    ax_density = fig.add_subplot(gs[0, col0])
                    ax_masked = fig.add_subplot(gs[0, col0 + 1])
                    ax_graph = fig.add_subplot(gs[1, col0:col0 + 2])
                    ax_fft = fig.add_subplot(gs[2, col0:col0 + 2]) if include_fft_row else None

                    density_proj1 = curve.get("density_proj1")
                    density_proj2 = curve.get("density_proj2")
                    density_extent1 = curve.get("density_extent1")
                    density_extent2 = curve.get("density_extent2")
                    density_limits = curve.get("density_limits")
                    masked_img = np.asarray(curve.get("masked_theoretical"), dtype=float) if curve.get("masked_theoretical") is not None else None

                    if density_proj1 is not None and density_proj2 is not None and density_extent1 is not None and density_extent2 is not None:
                        ax_density.imshow(
                            np.asarray(density_proj1, dtype=float),
                            extent=tuple(density_extent1),
                            origin="lower",
                            interpolation="nearest",
                        )
                        ax_density.imshow(
                            np.asarray(density_proj2, dtype=float),
                            extent=tuple(density_extent2),
                            origin="lower",
                            interpolation="nearest",
                            alpha=0.65,
                        )
                        if density_limits is not None:
                            xmin, xmax, ymin, ymax = tuple(density_limits)
                            ax_density.set_xlim(xmin, xmax)
                            ax_density.set_ylim(ymin, ymax)
                        ax_density.set_aspect("equal", adjustable="box")
                    else:
                        density_img = np.asarray(curve.get("proj_sum"), dtype=float) if curve.get("proj_sum") is not None else None
                        if density_img is not None:
                            ax_density.matshow(density_img)
                    ax_density.set_title("translated density sum", fontsize=9)
                    ax_density.set_xticks([])
                    ax_density.set_yticks([])

                    if masked_img is not None:
                        ax_masked.matshow(masked_img)
                    ax_masked.set_title("masked theoretical", fontsize=9)
                    ax_masked.set_xticks([])
                    ax_masked.set_yticks([])

                    line_combined, = ax_graph.plot(curve["q"], curve["I_combined"], linewidth=1.1, alpha=0.9, label="1 ellipse + polyhedron")
                    line_ellipsoid, = ax_graph.plot(curve["q"], curve["I_ellipsoid"], linewidth=0.95, alpha=0.9, label="1 ellipse only")
                    ax_graph.plot(curve["q"], curve["I_polyhedron"], linewidth=0.95, alpha=0.9, label="polyhedron only")
                    global_condition_idx = start_idx + local_idx
                    gate_for_plot = fft_gate_by_idx.get(global_condition_idx, {"passes": True, "reason": "gate_disabled"})
                    radius_allowed_for_plot = bool(show_first_min_radius) and _strict_gate_pass(gate_for_plot)
                    params_for_error_plot = curve.get("parameters", {}) or cond.get("parameters", {}) or {}
                    ellipse_ref_for_plot = self._polyhedron_ellipsoid1_ellipse_size_reference_nm(params_for_error_plot)
                    ellipse_ref_nm_for_plot = ellipse_ref_for_plot.get("reference_radius_nm", np.nan)
                    if bool(show_first_min_radius) and bool(radius_allowed_for_plot):
                        try:
                            delta_calc = _radius_delta_vs_ellipse_calculated(curve, radius_allowed=True)
                            if bool(delta_calc.get("found", False)):
                                ax_graph.text(
                                    0.015,
                                    0.985,
                                    (
                                        f"ΔR(comb−ell) = {float(delta_calc['delta_nm']):+.1f} nm "
                                        f"({float(delta_calc['delta_percent']):+.1f}%)\n"
                                        f"Rell={float(delta_calc['ellipse_radius_nm']):.1f} nm, "
                                        f"Rcomb={float(delta_calc['combined_radius_nm']):.1f} nm"
                                    ),
                                    transform=ax_graph.transAxes,
                                    ha="left",
                                    va="top",
                                    fontsize=6.5,
                                    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="0.25", alpha=0.75),
                                    zorder=9,
                                )
                            else:
                                ax_graph.text(
                                    0.015,
                                    0.985,
                                    "ΔR(comb−ell): not available",
                                    transform=ax_graph.transAxes,
                                    ha="left",
                                    va="top",
                                    fontsize=6.5,
                                    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="0.25", alpha=0.70),
                                    zorder=9,
                                )
                        except Exception:
                            pass
                    if bool(show_extrema_markers):
                        try:
                            self._plot_smoothed_extrema_markers(
                                ax_graph,
                                curve["q"],
                                curve["I_combined"],
                                q_min=0.005,
                                q_max=0.03,
                                color=line_combined.get_color(),
                                label_prefix="combined",
                                max_features=8,
                                marker_size=28,
                                legend=False,
                                draw_guides=True,
                                label_y_frac=0.96,
                                show_sphere_radius=bool(radius_allowed_for_plot),
                                sphere_radius_q_threshold=0.005,
                                radius_error_reference_nm=None,
                                radius_error_reference_label="Re",
                            )
                            self._plot_smoothed_extrema_markers(
                                ax_graph,
                                curve["q"],
                                curve["I_ellipsoid"],
                                q_min=0.005,
                                q_max=0.03,
                                color=line_ellipsoid.get_color(),
                                label_prefix="1 ellipse",
                                max_features=8,
                                marker_size=28,
                                legend=False,
                                draw_guides=True,
                                label_y_frac=0.72,
                                show_sphere_radius=bool(radius_allowed_for_plot),
                                sphere_radius_q_threshold=0.005,
                                radius_error_reference_nm=None,
                                radius_error_reference_label="Re",
                            )
                        except Exception:
                            pass
                    if bool(show_first_min_radius) and use_radius_fft_gate and not _strict_gate_pass(gate_for_plot):
                        try:
                            ax_graph.text(
                                0.015,
                                0.08,
                                "radius not applied: FFT gate failed\n" + str(gate_for_plot.get("gate_expression", gate_for_plot.get("reason", ""))),
                                transform=ax_graph.transAxes,
                                ha="left",
                                va="bottom",
                                fontsize=6.2,
                                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.35", alpha=0.70),
                                zorder=8,
                            )
                        except Exception:
                            pass
                    if ax_fft is not None:
                        idx_global = start_idx + local_idx
                        spec_ell = fft_cache.get((idx_global, "I_ellipsoid"))
                        spec_comb = fft_cache.get((idx_global, "I_combined"))
                        if spec_ell is None:
                            spec_ell = self._compute_oscillation_fft_spectrum(curve["q"], curve["I_ellipsoid"], q_min=0.005, q_max=0.03)
                        if spec_comb is None:
                            spec_comb = self._compute_oscillation_fft_spectrum(curve["q"], curve["I_combined"], q_min=0.005, q_max=0.03)
                        for spec, label in ((spec_comb, "1 ellipse + polyhedron"), (spec_ell, "1 ellipse only")):
                            freq = np.asarray(spec.get("frequency", []), dtype=float)
                            amp = np.asarray(spec.get("amplitude", []), dtype=float)
                            good = np.isfinite(freq) & np.isfinite(amp) & (freq > 0)
                            good &= (freq <= 1800.0)
                            if np.count_nonzero(good) >= 2:
                                ax_fft.plot(freq[good], amp[good], linewidth=1.05, alpha=0.9, label=label)
                                dom_f = spec.get("dominant_frequency", np.nan)
                                if np.isfinite(dom_f):
                                    ax_fft.axvline(float(dom_f), linestyle="--", linewidth=0.8, alpha=0.55)
                                    txt = f"{label.split()[0]} f={float(dom_f):.0f}"
                                    ax_fft.text(
                                        float(dom_f),
                                        0.94 if "polyhedron" in label else 0.72,
                                        txt,
                                        transform=ax_fft.get_xaxis_transform(),
                                        rotation=90,
                                        ha="right",
                                        va="top",
                                        fontsize=6,
                                        clip_on=False,
                                    )
                        ax_fft.set_xlabel(f"oscillation frequency (cycles per {str(q_label)})", fontsize=8)
                        ax_fft.set_ylabel("normalized FFT amplitude", fontsize=8)
                        ax_fft.tick_params(axis="both", which="major", labelsize=7)
                        ax_fft.grid(True, alpha=0.25)
                        ax_fft.legend(loc="upper right", fontsize=7)

                    params_full = curve.get("parameters", {}) or {}
                    swept_values = cond.get("sweep_values", {}) or {}

                    title_text = self._format_polyhedron_ellipsoid1_size_title(
                        params_full,
                        selected_params=selected_params,
                        swept_values=swept_values,
                        max_extra=3,
                        multiline=True,
                    )
                    poly_title_name = str(cond.get("polyhedron_name", "")).strip()
                    if poly_title_name:
                        title_text = f"{poly_title_name}: {title_text}"
                    ax_graph.set_title(title_text, fontsize=9)
                    ax_graph.set_xlim(0.005, 0.03)
                    ax_graph.set_xlabel(str(q_label), fontsize=8)
                    ax_graph.set_ylabel("azimuthal average intensity", fontsize=8, labelpad=2)
                    ax_graph.tick_params(axis="both", which="major", labelsize=7, pad=1)
                    try:
                        ax_graph.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
                    except Exception:
                        pass
                    try:
                        ax_graph.yaxis.get_offset_text().set_fontsize(7)
                        ax_graph.xaxis.get_offset_text().set_fontsize(7)
                    except Exception:
                        pass
                    ax_graph.grid(True, alpha=0.25)
                    ax_graph.xaxis.get_offset_text().set_visible(False)
                    ax_graph.legend(loc="upper right", fontsize=7)
                    if ax_fft is not None:
                        ax_fft.set_title(title_text, fontsize=9)

                # hide unused bundle slots if page not full
                for local_idx in range(len(chunk_curves), bundles_per_page):
                    col0 = 2 * local_idx
                    specs_to_hide = [gs[0, col0], gs[0, col0 + 1], gs[1, col0:col0 + 2]]
                    if include_fft_row:
                        specs_to_hide.append(gs[2, col0:col0 + 2])
                    for spec in specs_to_hide:
                        ax = fig.add_subplot(spec)
                        ax.axis("off")

                fig.tight_layout(rect=(0.02, 0.02, 0.985, 0.94), w_pad=1.1)
                fig.savefig(current_png_path)
                preview_png_paths.append(current_png_path)
        except Exception:
            preview_png_paths = []

        fft_preview_png_paths = []


        summary = {
            "mode": "polyhedron_ellipsoid1_simulation_only_1d_sweep",
            "created_unix": time.time(),
            "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_conditions": len(conditions),
            "npt": int(npt),
            "q_label": str(q_label),
            "q_source": str(q_source),
            "plot_q_range": {"min": 0.005, "max": 0.03},
            "sweep_config": sweep_config,
            "poisson": poisson_metadata,
            "extrema_q_guides": {
                "enabled_for_preview": bool(show_extrema_markers),
                "method": "log-intensity smoothing of the displayed curve with low-q boundary retention and minimum q-separation pruning; extrema reported for 1 ellipse only and 1 ellipse + polyhedron",
                "q_range": {"min": 0.005, "max": 0.03},
            },
            "radius_fft_gate": {
                "enabled": bool(use_radius_fft_gate),
                "max_combined_frequency": gate_max_freq,
                "max_frequency_delta": gate_max_delta,
                "use_combined_frequency_cutoff": bool(gate_use_f_cut),
                "use_frequency_delta_gate": bool(gate_use_df),
                "amplitude_ratio": "reported as diagnostic only; not used for pass/fail",
                "gate_csv": str(fft_gate_csv_path),
            },
            "conditions": conditions,
            "files": {
                "npz": str(npz_path),
                "csv": str(csv_path),
                "extrema_csv": str(extrema_csv_path),
                "radius_csv": str(radius_csv_path),
                "fft_gate_csv": str(fft_gate_csv_path),
                "fft_csv": str(fft_csv_path),
                "preview_png": str(preview_png_paths[0]) if preview_png_paths else None,
                "preview_png_pages": [str(p) for p in preview_png_paths],
            },
        }
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"npz": npz_path, "csv": csv_path, "extrema_csv": extrema_csv_path, "radius_csv": radius_csv_path, "fft_gate_csv": fft_gate_csv_path, "fft_csv": fft_csv_path, "json": json_path, "png": preview_png_paths[0] if preview_png_paths else None, "png_pages": preview_png_paths}

    def _open_polyhedron_ellipsoid1_sweep_dialog(self):
        """Open a checkbox-based parameter-sweep dialog for simulation-only 1D curves."""
        try:
            app = self._select_polyhedron_ellipsoid1_simulator()
        except Exception as exc:
            messagebox.showerror(
                "Polyhedron+Ellipsoid sweep 1D",
                f"Could not open the sweep dialog.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        param_names = self._polyhedron_ellipsoid1_numeric_parameter_names(app)
        if not param_names:
            messagebox.showerror(
                "Polyhedron+Ellipsoid sweep 1D",
                "No numeric parameters were found in the Polyhedron + 1 ellipsoid simulator.",
                parent=self.root,
            )
            return
        allowed_polyhedra = ["tetra", "cube", "octa", "dodeca", "icosa"]
        try:
            current_poly_name = str(getattr(app, "name_var").get()).strip().lower()
        except Exception:
            current_poly_name = "cube"
        if current_poly_name not in allowed_polyhedra:
            current_poly_name = "cube"
        win = tk.Toplevel(self.root)
        win.title("Polyhedron + 1 ellipsoid: checkbox parameter sweep")
        win.geometry("1280x840")
        win.resizable(True, True)
        win.transient(self.root)
        outer = ttk.Frame(win, padding=(12, 10))
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)
        ttk.Label(
            outer,
            text="Simulation-only 1D sweep",
            font=("Arial", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            outer,
            text=(
                "Select one or more polyhedron types and check any numeric parameters to scan. "
                "For each condition, combined / ellipse-only / polyhedron-only 1D curves are saved."
            ),
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))
        poly_frame = ttk.LabelFrame(outer, text="Polyhedron", padding=(10, 8))
        poly_frame.grid(row=2, column=0, sticky="ew")
        poly_frame.columnconfigure(1, weight=1)
        poly_type_vars = {}
        ttk.Label(poly_frame, text="polyhedron types").grid(row=0, column=0, sticky="w")
        poly_checks_frame = ttk.Frame(poly_frame)
        poly_checks_frame.grid(row=0, column=1, sticky="w", padx=(8, 0))
        for c, name in enumerate(allowed_polyhedra):
            var = tk.BooleanVar(value=(name == current_poly_name))
            poly_type_vars[name] = var
            ttk.Checkbutton(poly_checks_frame, text=name, variable=var).grid(row=0, column=c, sticky="w", padx=(0, 10))
        def _set_all_polyhedron_types(value):
            for var in poly_type_vars.values():
                var.set(bool(value))
            try:
                update_total_label()
            except Exception:
                pass
        ttk.Button(poly_checks_frame, text="All", command=lambda: _set_all_polyhedron_types(True)).grid(row=0, column=len(allowed_polyhedra), sticky="w", padx=(4, 0))
        def _set_current_polyhedron_type_only():
            for poly_name_tmp, var in poly_type_vars.items():
                var.set(poly_name_tmp == current_poly_name)
            try:
                update_total_label()
            except Exception:
                pass
        ttk.Button(poly_checks_frame, text="Current only", command=_set_current_polyhedron_type_only).grid(row=0, column=len(allowed_polyhedra) + 1, sticky="w", padx=(4, 0))

        position_coupling_var = tk.StringVar(value="Fixed centers")
        ttk.Label(poly_frame, text="position coupling").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            poly_frame,
            textvariable=position_coupling_var,
            values=["Fixed centers", "Normalized separation fixed"],
            state="readonly",
            width=28,
        ).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Label(
            poly_frame,
            text="Normalized mode keeps d/(ell_scale + poly_scale×poly_radius) fixed and derives poly_tu/poly_tv.",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        param_frame = ttk.LabelFrame(outer, text="Parameters to scan", padding=(10, 8))
        param_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        param_frame.columnconfigure(0, weight=1)
        param_frame.rowconfigure(1, weight=1)
        help_text = (
            "Check parameters to vary. Use start / stop / step, or enter explicit comma/space-separated values. "
            "If Values is non-empty, it overrides start / stop / step for that parameter."
        )
        ttk.Label(param_frame, text=help_text).grid(row=0, column=0, sticky="w", pady=(0, 6))
        canvas = tk.Canvas(param_frame, height=260, highlightthickness=0)
        ybar = ttk.Scrollbar(param_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=ybar.set)
        canvas.grid(row=1, column=0, sticky="nsew")
        ybar.grid(row=1, column=1, sticky="ns")
        list_holder = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=list_holder, anchor="nw")
        def _on_list_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        list_holder.bind("<Configure>", _on_list_configure)
        def _on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)
        for col, weight in enumerate([0, 0, 1, 1, 1, 2, 0]):
            list_holder.columnconfigure(col, weight=weight)
        ttk.Label(list_holder, text="Use").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Label(list_holder, text="Parameter").grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Label(list_holder, text="Start").grid(row=0, column=2, sticky="ew", padx=(0, 8))
        ttk.Label(list_holder, text="Stop").grid(row=0, column=3, sticky="ew", padx=(0, 8))
        ttk.Label(list_holder, text="Step").grid(row=0, column=4, sticky="ew", padx=(0, 8))
        ttk.Label(list_holder, text="Values (optional)").grid(row=0, column=5, sticky="ew", padx=(0, 10))
        ttk.Label(list_holder, text="Current").grid(row=0, column=6, sticky="w")
        param_rows = {}
        for r, name in enumerate(param_names, start=1):
            current_text = self._polyhedron_ellipsoid1_current_value_text(app, name)
            use_var = tk.BooleanVar(value=False)
            start_var = tk.StringVar(value=current_text)
            stop_var = tk.StringVar(value=current_text)
            step_var = tk.StringVar(value="1")
            values_var = tk.StringVar(value="")
            ttk.Checkbutton(list_holder, variable=use_var).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Label(list_holder, text=name).grid(row=r, column=1, sticky="w", padx=(0, 10), pady=2)
            ttk.Entry(list_holder, textvariable=start_var, width=12).grid(row=r, column=2, sticky="ew", padx=(0, 8), pady=2)
            ttk.Entry(list_holder, textvariable=stop_var, width=12).grid(row=r, column=3, sticky="ew", padx=(0, 8), pady=2)
            ttk.Entry(list_holder, textvariable=step_var, width=12).grid(row=r, column=4, sticky="ew", padx=(0, 8), pady=2)
            ttk.Entry(list_holder, textvariable=values_var, width=24).grid(row=r, column=5, sticky="ew", padx=(0, 10), pady=2)
            ttk.Label(list_holder, text=current_text).grid(row=r, column=6, sticky="w", pady=2)
            param_rows[name] = {
                "use": use_var,
                "start": start_var,
                "stop": stop_var,
                "step": step_var,
                "values": values_var,
                "current": current_text,
            }
        select_bar = ttk.Frame(param_frame)
        select_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        def set_all_checks(value):
            for row in param_rows.values():
                row["use"].set(bool(value))
            update_total_label()
        ttk.Button(select_bar, text="Select all", command=lambda: set_all_checks(True)).pack(side=tk.LEFT)
        ttk.Button(select_bar, text="Clear all", command=lambda: set_all_checks(False)).pack(side=tk.LEFT, padx=(6, 0))
        output_frame = ttk.LabelFrame(outer, text="Output", padding=(10, 8))
        output_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        output_frame.columnconfigure(1, weight=1)
        output_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Label(output_frame, text="Save folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_frame, textvariable=output_dir_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        def browse_output():
            folder = filedialog.askdirectory(parent=win, title="Choose sweep output folder")
            if folder:
                output_dir_var.set(folder)
        ttk.Button(output_frame, text="Browse...", command=browse_output).grid(row=0, column=2, sticky="e")

        sweep_poisson_var = tk.BooleanVar(value=bool(self.poisson_noise_var.get()))
        ttk.Checkbutton(
            output_frame,
            text="Add Poisson noise to sweep",
            variable=sweep_poisson_var,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(output_frame, text="Multiplier").grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        ttk.Entry(output_frame, textvariable=self.poisson_count_multiplier_var, width=9).grid(
            row=1, column=1, sticky="w", padx=(78, 0), pady=(8, 0)
        )
        ttk.Label(output_frame, text="Seed").grid(row=1, column=1, sticky="w", padx=(160, 0), pady=(8, 0))
        ttk.Entry(output_frame, textvariable=self.poisson_seed_var, width=10).grid(
            row=1, column=1, sticky="w", padx=(205, 0), pady=(8, 0)
        )
        ttk.Label(
            output_frame,
            text="Poisson is applied to each 2D simulated detector image before radial integration.",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))
        sweep_extrema_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            output_frame,
            text="Show extrema q-guides on sweep preview",
            variable=sweep_extrema_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        radius_options_frame = ttk.LabelFrame(output_frame, text="1st-minimum radius assumption", padding=(8, 6))
        radius_options_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        radius_options_frame.columnconfigure(4, weight=1)

        sweep_radius_var = tk.BooleanVar(value=False)
        sweep_radius_gate_var = tk.BooleanVar(value=True)
        radius_check = ttk.Checkbutton(
            radius_options_frame,
            text="Assume 1st minimum is droplet oscillation and estimate radius",
            variable=sweep_radius_var,
        )
        radius_check.grid(row=0, column=0, columnspan=5, sticky="w")
        radius_help = ttk.Label(
            radius_options_frame,
            text="R = 4.493 / q_first_min, reported in nm when q is in Å⁻¹.",
        )
        radius_help.grid(row=1, column=0, columnspan=5, sticky="w", padx=(18, 0), pady=(2, 0))

        gate_check = ttk.Checkbutton(
            radius_options_frame,
            text="Use FFT gate before applying the 1st-minimum radius",
            variable=sweep_radius_gate_var,
        )
        gate_check.grid(row=2, column=0, columnspan=5, sticky="w", padx=(18, 0), pady=(5, 0))

        fft_gate_fmax_var = tk.StringVar(value="300")
        fft_gate_df_var = tk.StringVar(value="50")
        fft_gate_use_fmax_var = tk.BooleanVar(value=True)
        fft_gate_use_df_var = tk.BooleanVar(value=True)
        gate_widgets = []
        gate_input_pairs = []

        chk_fmax = ttk.Checkbutton(radius_options_frame, text="combined f ≤", variable=fft_gate_use_fmax_var)
        chk_fmax.grid(row=3, column=0, sticky="w", padx=(36, 0), pady=(3, 0))
        ent_fmax = ttk.Entry(radius_options_frame, textvariable=fft_gate_fmax_var, width=8)
        ent_fmax.grid(row=3, column=1, sticky="w", padx=(4, 14), pady=(3, 0))
        gate_widgets.extend([chk_fmax, ent_fmax])
        gate_input_pairs.append((fft_gate_use_fmax_var, ent_fmax))

        chk_df = ttk.Checkbutton(radius_options_frame, text="|Δf| ≤", variable=fft_gate_use_df_var)
        chk_df.grid(row=3, column=2, sticky="w", pady=(3, 0))
        ent_df = ttk.Entry(radius_options_frame, textvariable=fft_gate_df_var, width=8)
        ent_df.grid(row=3, column=3, sticky="w", padx=(4, 14), pady=(3, 0))
        gate_widgets.extend([chk_df, ent_df])
        gate_input_pairs.append((fft_gate_use_df_var, ent_df))

        gate_help = ttk.Label(
            radius_options_frame,
            text="Amplitude ratio is diagnostic only; it is not used for gate pass/fail.",
        )
        gate_help.grid(row=4, column=0, columnspan=5, sticky="w", padx=(36, 0), pady=(2, 0))

        def _update_radius_gate_controls(*_args):
            radius_on = bool(sweep_radius_var.get())
            gate_on = bool(sweep_radius_gate_var.get())
            radius_help.configure(state=("normal" if radius_on else "disabled"))
            gate_check.configure(state=("normal" if radius_on else "disabled"))
            gate_help.configure(state=("normal" if (radius_on and gate_on) else "disabled"))
            condition_state = "normal" if (radius_on and gate_on) else "disabled"
            for use_var, entry_widget in gate_input_pairs:
                # Condition checkboxes are enabled only when the parent radius
                # assumption and FFT gate are both enabled.  The numeric entry
                # is additionally disabled when its own condition is unchecked.
                for widget in gate_widgets:
                    if widget is entry_widget:
                        continue
                entry_widget.configure(state=("normal" if (radius_on and gate_on and bool(use_var.get())) else "disabled"))
            chk_fmax.configure(state=condition_state)
            chk_df.configure(state=condition_state)
            if not (radius_on and gate_on):
                ent_fmax.configure(state="disabled")
                ent_df.configure(state="disabled")

        sweep_radius_var.trace_add("write", _update_radius_gate_controls)
        sweep_radius_gate_var.trace_add("write", _update_radius_gate_controls)
        fft_gate_use_fmax_var.trace_add("write", _update_radius_gate_controls)
        fft_gate_use_df_var.trace_add("write", _update_radius_gate_controls)
        _update_radius_gate_controls()

        ttk.Label(
            output_frame,
            text="FFT of detrended log I(q) is automatically included in the 3rd row of the sweep preview for 1 ellipse only and 1 ellipse + polyhedron.",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))
        status_var = tk.StringVar(value="Ready. Radial bins are fixed at 500. Use start/stop/step or explicit Values for checked parameters.")
        total_var = tk.StringVar(value="Calculation count: 1")
        count_detail_var = tk.StringVar(value="Selected parameters: none; current parameter set only")
        progress_text_var = tk.StringVar(value="Progress: idle")
        current_condition_var = tk.StringVar(value="Current condition: —")
        status_row = ttk.Frame(outer)
        status_row.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        status_row.columnconfigure(0, weight=1)
        ttk.Label(status_row, textvariable=status_var, anchor=tk.W).grid(row=0, column=0, sticky="ew")
        ttk.Label(status_row, textvariable=total_var, anchor=tk.E).grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Label(status_row, textvariable=count_detail_var, anchor=tk.W).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(status_row, textvariable=progress_text_var, anchor=tk.W).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(status_row, textvariable=current_condition_var, anchor=tk.E).grid(row=2, column=1, sticky="e", padx=(12, 0), pady=(4, 0))
        progress = ttk.Progressbar(outer, mode="determinate")
        progress.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        def _parse_start_stop_step_values(start_text, stop_text, step_text, label="parameter"):
            start_text = str(start_text).strip()
            stop_text = str(stop_text).strip()
            step_text = str(step_text).strip()
            if start_text == "" or stop_text == "" or step_text == "":
                raise ValueError(f"{label}: start, stop, and step must all be provided.")
            try:
                start_val = float(start_text)
                stop_val = float(stop_text)
                step_val = float(step_text)
            except Exception as exc:
                raise ValueError(f"{label}: could not parse start/stop/step as numbers.") from exc
            if np.isclose(start_val, stop_val, rtol=0.0, atol=1e-15):
                return np.asarray([start_val], dtype=float)
            if np.isclose(step_val, 0.0, rtol=0.0, atol=1e-15):
                raise ValueError(f"{label}: step cannot be 0.")
            delta = stop_val - start_val
            if delta > 0 and step_val < 0:
                raise ValueError(f"{label}: step must be positive when stop > start.")
            if delta < 0 and step_val > 0:
                raise ValueError(f"{label}: step must be negative when stop < start.")
            step_abs = abs(step_val)
            if delta > 0:
                values = np.arange(start_val, stop_val + 0.5 * step_abs, step_abs, dtype=float)
            else:
                values = np.arange(start_val, stop_val - 0.5 * step_abs, -step_abs, dtype=float)
            if values.size == 0:
                raise ValueError(f"{label}: no sweep values were generated.")
            return values
        def _parse_explicit_sweep_values(values_text, label="parameter"):
            values_text = str(values_text or "").strip()
            if values_text == "":
                return None
            import re
            tokens = [tok for tok in re.split(r"[,;\s]+", values_text) if tok]
            if not tokens:
                return None
            parsed = []
            for tok in tokens:
                try:
                    parsed.append(float(tok))
                except Exception as exc:
                    raise ValueError(f"{label}: could not parse explicit value '{tok}'. Use values like 150, 300, 400.") from exc
            return np.asarray(parsed, dtype=float)
        def selected_polyhedron_names():
            names = [name for name in allowed_polyhedra if bool(poly_type_vars.get(name, tk.BooleanVar(value=False)).get())]
            if not names:
                raise ValueError("Select at least one polyhedron type.")
            return names

        def selected_parameter_values():
            selected = []
            value_lists = []
            for name, row in param_rows.items():
                if bool(row["use"].get()):
                    explicit_values = _parse_explicit_sweep_values(row["values"].get(), label=name)
                    if explicit_values is not None:
                        values = explicit_values
                    else:
                        values = _parse_start_stop_step_values(
                            row["start"].get(),
                            row["stop"].get(),
                            row["step"].get(),
                            label=name,
                        )
                    selected.append(name)
                    value_lists.append(values)
            return selected, value_lists
        def update_total_label():
            try:
                poly_names = selected_polyhedron_names()
                selected, value_lists = selected_parameter_values()
                total = max(1, len(poly_names))
                detail_parts = [f"polyhedron: {len(poly_names)} type(s) ({', '.join(poly_names)})"]
                formula_parts = [str(max(1, len(poly_names)))]
                for name, values in zip(selected, value_lists):
                    n_values = max(1, len(values))
                    total *= n_values
                    formula_parts.append(str(n_values))
                    first_val = float(values[0]) if len(values) else float("nan")
                    last_val = float(values[-1]) if len(values) else float("nan")
                    if n_values == 1:
                        detail_parts.append(f"{name}: 1 value ({first_val:.6g})")
                    else:
                        detail_parts.append(f"{name}: {n_values} values ({first_val:.6g} → {last_val:.6g})")
                formula = " × ".join(formula_parts)
                total_var.set(f"Calculation count: {total} condition(s)")
                if not selected:
                    count_detail_var.set(f"Selected parameters: none; polyhedron types {len(poly_names)} | {formula} = {total} | " + "; ".join(detail_parts))
                else:
                    count_detail_var.set(f"Selected parameters: {len(selected)} | {formula} = {total} | " + "; ".join(detail_parts))
                progress.configure(maximum=max(total, 1), value=0)
                progress_text_var.set(f"Progress: 0/{total} (0.0%)")
                current_condition_var.set("Current condition: —")
            except Exception as exc:
                total_var.set("Calculation count: invalid")
                count_detail_var.set(f"Invalid sweep settings: {exc}")
                progress_text_var.set("Progress: idle")
                current_condition_var.set("Current condition: —")
        for row in param_rows.values():
            row["use"].trace_add("write", lambda *_: update_total_label())
            row["start"].trace_add("write", lambda *_: update_total_label())
            row["stop"].trace_add("write", lambda *_: update_total_label())
            row["step"].trace_add("write", lambda *_: update_total_label())
            row["values"].trace_add("write", lambda *_: update_total_label())
        for var in poly_type_vars.values():
            var.trace_add("write", lambda *_: update_total_label())
        update_total_label()
        button_bar = ttk.Frame(outer)
        button_bar.grid(row=7, column=0, sticky="ew", pady=(12, 0))
        def plot_current_once():
            self._open_polyhedron_ellipsoid1_integration_window()
        ttk.Button(button_bar, text="Plot current single curve", command=plot_current_once).pack(side=tk.LEFT)
        def run_sweep_and_save():
            vars_dict = getattr(app, "vars", {}) or {}
            try:
                selected_params, value_lists = selected_parameter_values()
                poly_names_selected = selected_polyhedron_names()
                npt = 500
                for name in selected_params:
                    if name not in vars_dict:
                        raise ValueError(f"Unknown parameter: {name}")
                total = max(1, len(poly_names_selected))
                for values in value_lists:
                    total *= max(1, len(values))
            except Exception as exc:
                messagebox.showerror("Polyhedron+Ellipsoid sweep 1D", str(exc), parent=win)
                return
            output_dir = Path(output_dir_var.get()).expanduser()
            if not str(output_dir):
                messagebox.showerror("Polyhedron+Ellipsoid sweep 1D", "Choose a save folder first.", parent=win)
                return
            if total > 1000:
                proceed = messagebox.askyesno(
                    "Polyhedron+Ellipsoid sweep 1D",
                    f"This will compute {total} conditions. Continue?",
                    parent=win,
                )
                if not proceed:
                    return
            original_values = {}
            for name, var in vars_dict.items():
                try:
                    original_values[name] = var.get()
                except Exception:
                    pass
            try:
                original_poly = getattr(app, "name_var").get()
            except Exception:
                original_poly = current_poly_name

            position_coupling_mode = str(position_coupling_var.get()).strip()
            normalized_sep_state = None
            if position_coupling_mode == "Normalized separation fixed":
                def _float_original(name, default=0.0):
                    try:
                        return float(original_values.get(name, vars_dict[name].get()))
                    except Exception:
                        return float(default)

                base_ell_tu = _float_original("ell_tu", 0.0)
                base_ell_tv = _float_original("ell_tv", 0.0)
                base_poly_tu = _float_original("poly_tu", 0.0)
                base_poly_tv = _float_original("poly_tv", 0.0)
                base_dx = base_poly_tu - base_ell_tu
                base_dy = base_poly_tv - base_ell_tv
                base_distance = float(np.hypot(base_dx, base_dy))
                if (not np.isfinite(base_distance)) or base_distance <= 1e-12:
                    messagebox.showerror(
                        "Polyhedron+Ellipsoid sweep 1D",
                        "Normalized separation fixed mode requires a non-zero initial center-to-center distance.",
                        parent=win,
                    )
                    return
                base_ell_scale = abs(_float_original("ell_scale", 1.0))
                base_poly_scale = abs(_float_original("poly_scale", 1.0))
                base_poly_radius = abs(_float_original("poly_radius", 1.0))
                base_size_sum = base_ell_scale + base_poly_scale * base_poly_radius
                if (not np.isfinite(base_size_sum)) or base_size_sum <= 1e-12:
                    messagebox.showerror(
                        "Polyhedron+Ellipsoid sweep 1D",
                        "Normalized separation fixed mode requires a positive initial size sum: ell_scale + poly_scale×poly_radius.",
                        parent=win,
                    )
                    return
                normalized_sep_state = {
                    "unit_x": float(base_dx / base_distance),
                    "unit_y": float(base_dy / base_distance),
                    "ratio": float(base_distance / base_size_sum),
                    "base_distance": float(base_distance),
                    "base_size_sum": float(base_size_sum),
                    "base_ell_scale": float(base_ell_scale),
                    "base_poly_scale": float(base_poly_scale),
                    "base_poly_radius": float(base_poly_radius),
                }

            progress.configure(maximum=max(total, 1), value=0)
            progress_text_var.set(f"Progress: 0/{total} (0.0%)")
            current_condition_var.set("Current condition: preparing sweep")
            noise_state = "ON" if sweep_poisson_var.get() else "OFF"
            status_var.set(f"Preparing {total} condition(s) for polyhedron types={', '.join(poly_names_selected)}; Poisson={noise_state}.")
            try:
                win.update()
            except Exception:
                win.update_idletasks()
            curves = []
            conditions = []
            try:
                try:
                    base_seed = self._parse_poisson_seed()
                except Exception:
                    base_seed = None
                rng = np.random.default_rng(base_seed)
                def _apply_position_coupling_if_needed():
                    if normalized_sep_state is None:
                        return None
                    try:
                        ell_tu_now = float(vars_dict["ell_tu"].get())
                        ell_tv_now = float(vars_dict["ell_tv"].get())
                        ell_scale_now = abs(float(vars_dict["ell_scale"].get()))
                        poly_scale_now = abs(float(vars_dict["poly_scale"].get()))
                        poly_radius_now = abs(float(vars_dict["poly_radius"].get()))
                    except Exception as exc:
                        raise ValueError(f"Could not evaluate normalized separation variables: {exc}") from exc
                    size_sum_now = ell_scale_now + poly_scale_now * poly_radius_now
                    if (not np.isfinite(size_sum_now)) or size_sum_now <= 1e-12:
                        raise ValueError("Normalized separation fixed mode produced a non-positive current size sum.")
                    distance_now = normalized_sep_state["ratio"] * size_sum_now
                    poly_tu_now = ell_tu_now + normalized_sep_state["unit_x"] * distance_now
                    poly_tv_now = ell_tv_now + normalized_sep_state["unit_y"] * distance_now
                    vars_dict["poly_tu"].set(str(float(poly_tu_now)))
                    vars_dict["poly_tv"].set(str(float(poly_tv_now)))
                    return {
                        "poly_tu": float(poly_tu_now),
                        "poly_tv": float(poly_tv_now),
                        "distance": float(distance_now),
                        "size_sum": float(size_sum_now),
                        "normalized_separation": float(normalized_sep_state["ratio"]),
                    }

                if selected_params:
                    combo_list = list(itertools.product(*value_lists))
                else:
                    combo_list = [tuple()]
                idx = 0
                for poly_name_selected in poly_names_selected:
                    try:
                        getattr(app, "name_var").set(poly_name_selected)
                    except Exception:
                        pass
                    for combo_values in combo_list:
                        sweep_values = {}
                        for name, value in zip(selected_params, combo_values):
                            vars_dict[name].set(str(float(value)))
                            sweep_values[name] = float(value)
                        derived_position = _apply_position_coupling_if_needed()
                        if derived_position is not None:
                            sweep_values["derived_poly_tu"] = float(derived_position["poly_tu"])
                            sweep_values["derived_poly_tv"] = float(derived_position["poly_tv"])
                            sweep_values["derived_center_distance"] = float(derived_position["distance"])
                            sweep_values["normalized_separation_ratio"] = float(derived_position["normalized_separation"])
                        if sweep_values:
                            desc = ", ".join([f"{k}={v:.6g}" for k, v in sweep_values.items()])
                        else:
                            desc = "current parameters"
                        percent_before = 100.0 * idx / max(total, 1)
                        progress.configure(value=idx)
                        progress_text_var.set(f"Progress: {idx}/{total} ({percent_before:.1f}%)")
                        current_condition_var.set(f"Current condition: {idx + 1}/{total}")
                        noise_state = "ON" if sweep_poisson_var.get() else "OFF"
                        status_var.set(f"Running {idx + 1}/{total}: polyhedron={poly_name_selected}, {desc}; Poisson={noise_state}")
                        try:
                            win.update()
                        except Exception:
                            win.update_idletasks()
                        result = app.update_plot()
                        if result is None:
                            result = getattr(app, "last_result", None)
                        curve = self._compute_polyhedron_ellipsoid1_radial_curves_from_result(
                            app,
                            result,
                            npt=npt,
                            rng=rng,
                            apply_poisson=bool(sweep_poisson_var.get()),
                        )
                        curves.append(curve)
                        conditions.append({
                            "condition_index": idx,
                            "polyhedron_name": poly_name_selected,
                            "polyhedron_type_sweep": list(poly_names_selected),
                            "position_coupling_mode": position_coupling_mode,
                            "normalized_separation_state": _jsonify_params(normalized_sep_state or {}),
                            "sweep_values": dict(sweep_values),
                            "parameters": _jsonify_params(curve.get("parameters", {})),
                        })
                        idx += 1
                        percent_after = 100.0 * idx / max(total, 1)
                        progress.configure(value=idx)
                        progress_text_var.set(f"Progress: {idx}/{total} ({percent_after:.1f}%)")
                        current_condition_var.set(f"Finished condition: {idx}/{total}")
                        try:
                            win.update()
                        except Exception:
                            win.update_idletasks()
                if not curves:
                    raise ValueError("No curves were generated.")
                poisson_metadata = {
                    "requested": bool(sweep_poisson_var.get()),
                    "enabled": bool(curves[0].get("poisson_enabled", False)),
                    "peak_counts": curves[0].get("poisson_peak_counts", np.nan),
                    "seed": curves[0].get("poisson_seed", ""),
                    "reference_mode": curves[0].get("poisson_reference_mode", ""),
                    "reference_intensity": curves[0].get("poisson_reference_intensity", np.nan),
                    "count_multiplier": str(self.poisson_count_multiplier_var.get()),
                }
                def _parse_gate_float(value, label):
                    try:
                        parsed = float(str(value).strip())
                    except Exception as exc:
                        raise ValueError(f"{label} must be a number.") from exc
                    if not np.isfinite(parsed):
                        raise ValueError(f"{label} must be finite.")
                    return parsed

                radius_fft_gate_enabled = bool(sweep_radius_var.get()) and bool(sweep_radius_gate_var.get())
                if radius_fft_gate_enabled:
                    radius_fft_gate_config = {
                        "enabled": True,
                        "max_combined_frequency": _parse_gate_float(fft_gate_fmax_var.get(), "FFT gate combined f cutoff"),
                        "max_frequency_delta": _parse_gate_float(fft_gate_df_var.get(), "FFT gate |delta f| cutoff"),
                        "use_combined_frequency_cutoff": bool(fft_gate_use_fmax_var.get()),
                        "use_frequency_delta_gate": bool(fft_gate_use_df_var.get()),
                    }
                else:
                    radius_fft_gate_config = {
                        "enabled": False,
                        "max_combined_frequency": 300.0,
                        "max_frequency_delta": 50.0,
                        "use_combined_frequency_cutoff": True,
                        "use_frequency_delta_gate": True,
                    }
                sweep_config = {
                    "simulator": "Polyhedron + 1 ellipsoid",
                    "polyhedron_names": list(poly_names_selected),
                    "polyhedron_name": poly_names_selected[0] if len(poly_names_selected) == 1 else "multiple",
                    "position_coupling_mode": position_coupling_mode,
                    "normalized_separation_state": _jsonify_params(normalized_sep_state or {}),
                    "sweep_parameters": list(selected_params),
                    "sweep_values": {name: [float(x) for x in values] for name, values in zip(selected_params, value_lists)},
                    "npt": int(npt),
                    "plot_q_range_min": 0.005,
                    "plot_q_range_max": 0.03,
                    "output_dir": str(output_dir),
                    "poisson_requested": bool(sweep_poisson_var.get()),
                    "poisson_count_multiplier": str(self.poisson_count_multiplier_var.get()),
                    "poisson_seed": str(self.poisson_seed_var.get()),
                    "show_extrema_q_guides_on_preview": bool(sweep_extrema_var.get()),
                    "estimate_droplet_radius_from_first_minimum": bool(sweep_radius_var.get()),
                    "save_oscillation_frequency_fft_plot": True,
                    "radius_fft_gate": radius_fft_gate_config,
                }
                paths = self._save_polyhedron_ellipsoid1_sweep_results(
                    output_dir,
                    curves=curves,
                    conditions=conditions,
                    sweep_config=sweep_config,
                    q_label=curves[0].get("q_label", "q"),
                    q_source=curves[0].get("q_source", ""),
                    npt=npt,
                    poisson_metadata=poisson_metadata,
                    show_extrema_markers=bool(sweep_extrema_var.get()),
                    show_first_min_radius=bool(sweep_radius_var.get()),
                    show_fft_spectrum=True,
                    radius_fft_gate_config=radius_fft_gate_config,
                )
                progress.configure(value=max(total, 1))
                progress_text_var.set(f"Progress: {len(curves)}/{total} (100.0%)")
                current_condition_var.set("Current condition: complete")
                status_var.set(f"Saved {len(curves)} conditions: {paths['npz'].name}, {paths['csv'].name}")
                try:
                    win.update()
                except Exception:
                    win.update_idletasks()
                extra_saved_lines = []
                if bool(sweep_extrema_var.get()):
                    extra_saved_lines.append(f"Extrema CSV: {paths.get('extrema_csv')}")
                if bool(sweep_radius_var.get()):
                    extra_saved_lines.append(f"Radius CSV: {paths.get('radius_csv')}")
                    extra_saved_lines.append(f"FFT gate CSV: {paths.get('fft_gate_csv')}")
                extra_saved_lines.append(f"FFT CSV: {paths.get('fft_csv')}")
                extra_saved_text = ("\n" + "\n".join(extra_saved_lines)) if extra_saved_lines else ""
                messagebox.showinfo(
                    "Polyhedron+Ellipsoid sweep 1D",
                    "Sweep saved successfully.\n\n"
                    f"Conditions: {len(curves)}\n"
                    f"NPZ: {paths['npz']}\n"
                    f"CSV: {paths['csv']}\n"
                    f"Summary: {paths['json']}"
                    f"{extra_saved_text}",
                    parent=win,
                )
            except Exception as exc:
                messagebox.showerror(
                    "Polyhedron+Ellipsoid sweep 1D",
                    f"Sweep failed.\n\n{type(exc).__name__}: {exc}",
                    parent=win,
                )
                status_var.set(f"Failed: {type(exc).__name__}: {exc}")
                current_condition_var.set("Current condition: failed")
            finally:
                for name, value in original_values.items():
                    try:
                        vars_dict[name].set(value)
                    except Exception:
                        pass
                try:
                    getattr(app, "name_var").set(original_poly)
                except Exception:
                    pass
                try:
                    app.update_plot()
                except Exception:
                    pass
        ttk.Button(button_bar, text="Run sweep and save", command=run_sweep_and_save).pack(side=tk.RIGHT)
        ttk.Button(button_bar, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=(0, 8))
    def _open_polyhedron_ellipsoid1_integration_window(self):
        """Open a simulation-only 1D SAXS plot for one polyhedron + one ellipsoid.
        Unlike the general SAXS integrate window, this dedicated button does not
        require an experimental image.  It uses the current parameters of the
        Polyhedron + 1 ellipsoid simulator and preserves the currently selected
        polyhedron type.  It displays three simulated radial profiles on the
        same q axis:
        1. coherent polyhedron + ellipsoid pattern: |w_ell A_ell + w_poly A_poly|^2
        2. ellipsoid-only pattern: |w_ell A_ell|^2
        3. polyhedron-only pattern: |w_poly A_poly|^2
        """
        try:
            app = self._select_polyhedron_ellipsoid1_simulator()
            result = getattr(app, "last_result", None)
            if result is None:
                app.update_plot()
                result = getattr(app, "last_result", None)
            if not isinstance(result, dict):
                raise ValueError("The polyhedron + ellipsoid simulator did not return a valid result.")
            params = dict(result.get("parameters", {}))
            A_ell = np.asarray(result["A1"], dtype=np.complex128)
            A_poly = np.asarray(result["A2"], dtype=np.complex128)
            if A_ell.shape != A_poly.shape:
                raise ValueError(f"A1 and A2 shapes do not match: {A_ell.shape} vs {A_poly.shape}")
            image_shape = A_ell.shape
            app_mask = getattr(app, "mask", None)
            if app_mask is None:
                mask = np.ones(image_shape, dtype=float)
            else:
                mask = np.asarray(app_mask, dtype=float)
                if mask.shape != image_shape:
                    try:
                        mask = mask.reshape(image_shape)
                    except Exception:
                        mask = np.ones(image_shape, dtype=float)
            ell_weight = float(params.get("ell_weight", 1.0))
            poly_weight = float(params.get("poly_weight", 1.0))
            combined_total_img = np.abs(ell_weight * A_ell + poly_weight * A_poly) ** 2
            ellipse_total_img = np.abs(ell_weight * A_ell) ** 2
            poly_total_img = np.abs(poly_weight * A_poly) ** 2
            poisson_enabled = False
            poisson_peak_counts = None
            poisson_seed = None
            poisson_reference_mode = "off"
            poisson_reference_intensity_used = None
            if bool(self.poisson_noise_var.get()):
                try:
                    peak_counts, reference_intensity, reference_mode, _multiplier = self._poisson_reference_state()
                    seed = self._parse_poisson_seed()
                    rng = np.random.default_rng(seed)
                    # Use the same fixed simulation-side reference as the live 2D display.
                    # This preserves the current Poisson convention: the checkbox changes
                    # the simulated detector-count realization, while the underlying SAXS
                    # intensity model remains unchanged.
                    sim_reference_intensity = self._get_or_set_simulation_poisson_reference(app, combined_total_img)
                    if sim_reference_intensity is None:
                        sim_reference_intensity = reference_intensity
                    combined_total_img = self._poissonize_image(
                        combined_total_img,
                        peak_counts,
                        rng,
                        reference_intensity=sim_reference_intensity,
                    )
                    ellipse_total_img = self._poissonize_image(
                        ellipse_total_img,
                        peak_counts,
                        rng,
                        reference_intensity=sim_reference_intensity,
                    )
                    poly_total_img = self._poissonize_image(
                        poly_total_img,
                        peak_counts,
                        rng,
                        reference_intensity=sim_reference_intensity,
                    )
                    poisson_enabled = True
                    poisson_peak_counts = float(peak_counts)
                    poisson_seed = seed
                    poisson_reference_mode = str(reference_mode)
                    try:
                        poisson_reference_intensity_used = float(sim_reference_intensity)
                    except Exception:
                        poisson_reference_intensity_used = None
                except Exception:
                    # Keep the window usable even if noise parameters are invalid.
                    poisson_enabled = False
                    poisson_peak_counts = None
                    poisson_seed = None
                    poisson_reference_mode = "failed; clean curves shown"
                    poisson_reference_intensity_used = None
            combined_img = np.asarray(combined_total_img, dtype=float)
            ellipse_img = np.asarray(ellipse_total_img, dtype=float)
            poly_img = np.asarray(poly_total_img, dtype=float)
            if mask.shape == combined_img.shape:
                combined_img = combined_img * mask
                ellipse_img = ellipse_img * mask
                poly_img = poly_img * mask
            q_u = result.get("q_u", None)
            q_v = result.get("q_v", None)
            valid_mask = _load_default_saxs_valid_mask(combined_img.shape)
            q_u_real, q_v_real, _q_meta = _load_default_q_axes_from_poni(combined_img.shape)
            q_u_use = q_u_real if q_u_real is not None else q_u
            q_v_use = q_v_real if q_v_real is not None else q_v
            q_combo, I_combo, counts_combo, q_label, _std_combo, _sem_combo = _radial_integrate_on_q_grid(
                combined_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=500
            )
            q_ell, I_ell, _counts_ell, _q_label2, _std_ell, _sem_ell = _radial_integrate_on_q_grid(
                ellipse_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=500
            )
            q_poly, I_poly, _counts_poly, _q_label3, _std_poly, _sem_poly = _radial_integrate_on_q_grid(
                poly_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=500
            )
        except Exception as exc:
            messagebox.showerror(
                "Polyhedron+Ellipsoid simulation 1D",
                f"Could not compute the simulation-only 1D SAXS curves.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        win = tk.Toplevel(self.root)
        win.title("Simulation-only SAXS 1D: polyhedron + ellipsoid")
        win.geometry("1050x760")
        fig = Figure(figsize=(8.8, 6.8), dpi=100)
        ax = fig.add_subplot(111)
        finite_plot = (
            np.isfinite(q_combo)
            & (np.isfinite(I_combo) | np.isfinite(I_ell) | np.isfinite(I_poly))
        )
        if np.any(finite_plot):
            q_min_data = float(np.nanmin(q_combo[finite_plot]))
            q_max_data = float(np.nanmax(q_combo[finite_plot]))
        else:
            q_min_data = float(np.nanmin(q_combo))
            q_max_data = float(np.nanmax(q_combo))
        q_min_plot = 0.005
        q_max_plot = 0.03
        if q_max_plot <= q_min_plot:
            q_min_plot, q_max_plot = q_min_data, q_max_data
        def _plot_mask(q_arr, y_arr):
            q_arr = np.asarray(q_arr, dtype=float)
            y_arr = np.asarray(y_arr, dtype=float)
            return (
                np.isfinite(q_arr)
                & np.isfinite(y_arr)
                & (q_arr >= q_min_plot)
                & (q_arr <= q_max_plot)
            )
        def _redraw(log_y=False):
            ax.clear()
            m_combo = _plot_mask(q_combo, I_combo)
            m_ell = _plot_mask(q_ell, I_ell)
            m_poly = _plot_mask(q_poly, I_poly)
            ax.plot(q_combo[m_combo], I_combo[m_combo], label=f"1 ellipse + {poly_name}")
            ax.plot(q_ell[m_ell], I_ell[m_ell], label="1 ellipse only")
            ax.plot(q_poly[m_poly], I_poly[m_poly], label=f"{poly_name} only")
            ax.set_xlabel(q_label)
            ax.set_ylabel("azimuthal average intensity")
            noise_suffix = " with Poisson noise" if poisson_enabled else " (clean)"
            ax.set_title(f"Simulation-only SAXS 1D integration: {poly_name} + 1 ellipse" + noise_suffix)
            ax.set_xlim(q_min_plot, q_max_plot)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            ax.xaxis.get_offset_text().set_visible(False)
            if log_y:
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
            canvas.draw_idle()
        canvas = _OriginalFigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(win, padding=(8, 6))
        controls.pack(side=tk.BOTTOM, fill=tk.X)
        q_source = ".poni geometry" if 'q_u_real' in locals() and q_u_real is not None else "simulator q grid"
        poly_name = str(params.get("polyhedron_name", "cube"))
        if poisson_enabled:
            seed_text = "random" if poisson_seed is None else str(poisson_seed)
            ref_text = "n/a" if poisson_reference_intensity_used is None else f"{poisson_reference_intensity_used:.6g}"
            noise_text = f"Poisson=ON, counts={poisson_peak_counts:.6g}, seed={seed_text}, sim-ref={ref_text}"
        else:
            noise_text = f"Poisson=OFF ({poisson_reference_mode})"
        status = tk.StringVar(
            value=(
                f"Simulation-only radial integration | polyhedron={poly_name} | "
                f"valid pixels={int(np.sum(valid_mask))} | bins=500 | "
                f"x-range={q_min_plot:.4g} to {q_max_plot:.4g} {q_label} | source={q_source} | "
                f"{noise_text}"
            )
        )
        ttk.Label(controls, textvariable=status, anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        toolbar = ttk.Frame(controls)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        log_var = tk.BooleanVar(value=False)
        def on_toggle():
            _redraw(log_y=log_var.get())
        ttk.Checkbutton(toolbar, text="Log y", variable=log_var, command=on_toggle).pack(side=tk.LEFT)
        def save_curves():
            save_path = filedialog.asksaveasfilename(
                parent=win,
                title="Save simulation-only 1D curves",
                defaultextension=".npz",
                filetypes=[("NumPy archive", "*.npz"), ("CSV", "*.csv"), ("All files", "*.*")],
            )
            if not save_path:
                return
            save_path = Path(save_path)
            try:
                if save_path.suffix.lower() == ".csv":
                    import csv as _csv
                    with save_path.open("w", newline="", encoding="utf-8") as f:
                        writer = _csv.writer(f)
                        writer.writerow([q_label, f"I_1ellipse_plus_{poly_name}", "I_1ellipse_only", f"I_{poly_name}_only", "counts", "poisson_enabled"])
                        for a, b, c, d, n in zip(q_combo, I_combo, I_ell, I_poly, counts_combo):
                            writer.writerow([a, b, c, d, n, int(bool(poisson_enabled))])
                else:
                    np.savez(
                        save_path,
                        q=q_combo,
                        **{f"I_1ellipse_plus_{poly_name}": I_combo},
                        I_1ellipse_only=I_ell,
                        **{f"I_{poly_name}_only": I_poly},
                        counts=counts_combo,
                        q_label=q_label,
                        q_source=q_source,
                        poisson_enabled=bool(poisson_enabled),
                        poisson_peak_counts=np.nan if poisson_peak_counts is None else float(poisson_peak_counts),
                        poisson_seed="" if poisson_seed is None else str(poisson_seed),
                        poisson_reference_mode=poisson_reference_mode,
                        poisson_reference_intensity=np.nan if poisson_reference_intensity_used is None else float(poisson_reference_intensity_used),
                        parameters=params,
                    )
                status.set(f"Saved simulation-only curves to {save_path}")
            except Exception as exc:
                messagebox.showerror(
                    "Polyhedron+Ellipsoid simulation 1D",
                    f"Could not save curves.\n\n{type(exc).__name__}: {exc}",
                    parent=win,
                )
        ttk.Button(toolbar, text="Save curves...", command=save_curves).pack(side=tk.RIGHT)
        _redraw(log_y=False)
        canvas.draw()
    def _open_saxs_integration_window(self):
        try:
            sim_img, q_u, q_v = self._get_current_simulation_masked_image_and_q()
        except Exception as exc:
            messagebox.showerror(
                "SAXS integration",
                f"Could not read the current simulation image.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        exp_img = getattr(self, "exp_image_data", None)
        if exp_img is None:
            messagebox.showwarning(
                "SAXS integration",
                "Load an experimental image first.",
                parent=self.root,
            )
            return
        exp_img = np.asarray(exp_img, dtype=float)
        if exp_img.ndim != 2:
            messagebox.showerror(
                "SAXS integration",
                f"Experimental image must be 2D, got shape {exp_img.shape}.",
                parent=self.root,
            )
            return
        if exp_img.shape != sim_img.shape:
            messagebox.showerror(
                "SAXS integration",
                f"Experimental and simulation images have different shapes:\n"
                f"experimental={exp_img.shape}, simulation={sim_img.shape}",
                parent=self.root,
            )
            return
        try:
            valid_mask = _load_default_saxs_valid_mask(exp_img.shape)
            # Prefer q axes derived from the bundled .poni file so the q-axis matches the
            # real detector geometry. Fall back to the current simulator q grid if needed.
            q_u_real, q_v_real, q_meta = _load_default_q_axes_from_poni(exp_img.shape)
            q_u_use = q_u_real if q_u_real is not None else q_u
            q_v_use = q_v_real if q_v_real is not None else q_v
            q_exp, I_exp, counts_exp, q_label, std_exp, sem_exp = _radial_integrate_on_q_grid(
                exp_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=500
            )
            q_sim, I_sim, counts_sim, q_label, std_sim, sem_sim = _radial_integrate_on_q_grid(
                sim_img, q_u=q_u_use, q_v=q_v_use, valid_mask=valid_mask, npt=500
            )
        except Exception as exc:
            messagebox.showerror(
                "SAXS integration",
                f"Could not integrate images.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        win = tk.Toplevel(self.root)
        win.title("SAXS integration: experimental vs masked theoretical")
        win.geometry("1050x900")
        fig = Figure(figsize=(8.8, 8.2), dpi=100)
        gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 1.25, 1.25], hspace=0.18)
        ax = fig.add_subplot(gs[0, 0])
        ax_diff = fig.add_subplot(gs[1, 0], sharex=ax)
        ax_chi = fig.add_subplot(gs[2, 0], sharex=ax)
        finite_plot = np.isfinite(q_exp) & (np.isfinite(I_exp) | np.isfinite(I_sim))
        if np.any(finite_plot):
            q_min_data = float(np.nanmin(q_exp[finite_plot]))
            q_max_data = float(np.nanmax(q_exp[finite_plot]))
        else:
            q_min_data = float(np.nanmin(q_exp))
            q_max_data = float(np.nanmax(q_exp))
        # Show the low-q region by default.  This keeps the SAXS overlay focused on
        # the experimentally relevant range requested by the user while preserving
        # the full integrated arrays for saving.
        q_lower_display_limit = DEFAULT_1D_SCALE_FIT_Q_MIN
        q_upper_display_limit = DEFAULT_1D_SCALE_FIT_Q_MAX
        q_min_plot = max(q_min_data, q_lower_display_limit) if np.isfinite(q_lower_display_limit) else q_min_data
        q_max_plot = min(q_max_data, q_upper_display_limit) if q_min_plot < q_upper_display_limit else q_max_data
        if not np.isfinite(q_max_plot) or q_max_plot <= q_min_plot:
            q_max_plot = q_max_data
        def _fit_range_mask(y_a=None, y_b=None):
            good = np.isfinite(q_exp) & (q_exp >= q_min_plot) & (q_exp <= q_max_plot)
            if y_a is not None:
                good &= np.isfinite(np.asarray(y_a, dtype=float))
            if y_b is not None:
                good &= np.isfinite(np.asarray(y_b, dtype=float))
            if np.count_nonzero(good) < 3:
                good = np.isfinite(q_exp)
                if y_a is not None:
                    good &= np.isfinite(np.asarray(y_a, dtype=float))
                if y_b is not None:
                    good &= np.isfinite(np.asarray(y_b, dtype=float))
            return good
        def _compute_sim_scale_to_exp(y_exp_base, y_sim_base):
            # Least-squares scale factor a in I_exp(q) ~= a * I_sim(q),
            # evaluated over the displayed q range so the auto-scaled overlay
            # matches what the user is actually viewing.  This returns only a
            # display multiplier; it never changes the integrated simulation array.
            good = _fit_range_mask(y_exp_base, y_sim_base)
            return _compute_overlay_scale(np.asarray(y_exp_base)[good], np.asarray(y_sim_base)[good])
        def _plot_mask(q_values, y_values=None):
            # Plot only the visible q range instead of plotting the full curve and
            # then hiding it with set_xlim.  Matplotlib autoscaled the y-axis from
            # hidden high-q points in the previous version, which could make the
            # displayed low-q simulation curve look distorted or nearly flat.
            q_arr = np.asarray(q_values, dtype=float)
            good = np.isfinite(q_arr) & (q_arr >= q_min_plot) & (q_arr <= q_max_plot)
            if y_values is not None:
                good &= np.isfinite(np.asarray(y_values, dtype=float))
            if np.count_nonzero(good) < 3:
                good = np.isfinite(q_arr)
                if y_values is not None:
                    good &= np.isfinite(np.asarray(y_values, dtype=float))
            return good
        def _interp_to_exp_grid(q_src, y_src):
            q_src = np.asarray(q_src, dtype=float)
            y_src = np.asarray(y_src, dtype=float)
            if q_src.shape == q_exp.shape and np.allclose(q_src, q_exp, equal_nan=True, rtol=1e-10, atol=1e-12):
                return y_src.astype(float, copy=False)
            out = np.full_like(q_exp, np.nan, dtype=float)
            good = np.isfinite(q_src) & np.isfinite(y_src)
            if np.count_nonzero(good) >= 2:
                out = np.interp(q_exp, q_src[good], y_src[good], left=np.nan, right=np.nan)
            return out
        def _sigma_for_reduced_chi_square():
            sigma = np.asarray(sem_exp, dtype=float).copy()
            positive = np.isfinite(sigma) & (sigma > 0)
            if np.any(positive):
                sigma_floor = max(np.nanmedian(sigma[positive]) * 1.0e-6, 1.0e-12)
                sigma[positive] = np.maximum(sigma[positive], sigma_floor)
            return sigma
        def _compute_chi_square_contribution(y_exp, y_sim_on_exp, sigma):
            y_exp = np.asarray(y_exp, dtype=float)
            y_sim_on_exp = np.asarray(y_sim_on_exp, dtype=float)
            sigma = np.asarray(sigma, dtype=float)
            out = np.full_like(y_exp, np.nan, dtype=float)
            good = np.isfinite(y_exp) & np.isfinite(y_sim_on_exp) & np.isfinite(sigma) & (sigma > 0)
            if np.any(good):
                out[good] = ((y_exp[good] - y_sim_on_exp[good]) / sigma[good]) ** 2
            return out
        def _compute_reduced_chi_square(y_exp, y_sim_on_exp, sigma, n_fit_params=1):
            chi = _compute_chi_square_contribution(y_exp, y_sim_on_exp, sigma)
            good = _fit_range_mask(chi, sigma)
            n_points = int(np.count_nonzero(good))
            dof = max(n_points - int(n_fit_params), 1)
            if n_points == 0:
                return float('nan'), chi, n_points, dof
            red = float(np.nansum(chi[good]) / dof)
            return red, chi, n_points, dof
        sim_scale_default = _compute_sim_scale_to_exp(I_exp, I_sim)
        I_sim_scaled_default = I_sim * sim_scale_default
        I_sim_scaled_default_on_exp = _interp_to_exp_grid(q_sim, I_sim_scaled_default)
        sigma_exp_chi = _sigma_for_reduced_chi_square()
        I_diff_default = I_exp - I_sim_scaled_default_on_exp
        reduced_chi2_default, I_chi_default, n_chi_default, dof_default = _compute_reduced_chi_square(I_exp, I_sim_scaled_default_on_exp, sigma_exp_chi, n_fit_params=1)
        m_exp0 = _plot_mask(q_exp, I_exp)
        m_sim0 = _plot_mask(q_sim, I_sim_scaled_default)
        m_diff0 = _plot_mask(q_exp, I_diff_default)
        m_chi0 = _plot_mask(q_exp, I_chi_default)
        ax.plot(q_exp[m_exp0], I_exp[m_exp0], label="experimental")
        ax.plot(q_sim[m_sim0], I_sim_scaled_default[m_sim0], label=f"masked theoretical × {sim_scale_default:.3g}")
        ax.set_ylabel("azimuthal average intensity")
        ax.set_title(f"SAXS 1D integration (simulation auto-scaled, q fit={q_min_plot:.3f}–{q_max_plot:.3f})")
        ax.set_xlim(q_min_plot, q_max_plot)
        ax.legend()
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.xaxis.get_offset_text().set_visible(False)
        ax_diff.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
        ax_diff.plot(q_exp[m_diff0], I_diff_default[m_diff0], label="experimental - scaled simulation")
        ax_diff.set_ylabel("difference")
        ax_diff.grid(True, alpha=0.25)
        ax_diff.legend(loc="best")
        ax_diff.set_xlim(q_min_plot, q_max_plot)
        ax_diff.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax_diff.xaxis.get_offset_text().set_visible(False)
        ax_chi.plot(q_exp[m_chi0], I_chi_default[m_chi0], label=rf"$\chi^2$ contribution (reduced $\chi^2$={reduced_chi2_default:.4g})")
        ax_chi.set_xlabel(q_label)
        ax_chi.set_ylabel(r"$\chi^2$ contribution")
        ax_chi.grid(True, alpha=0.25)
        ax_chi.legend(loc="best")
        ax_chi.set_xlim(q_min_plot, q_max_plot)
        canvas = _OriginalFigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        controls = ttk.Frame(win, padding=(8, 6))
        controls.pack(side=tk.BOTTOM, fill=tk.X)
        q_source = ".poni geometry" if 'q_u_real' in locals() and q_u_real is not None else "simulator q grid"
        def _status_text(current_sim_scale=sim_scale_default, auto_scale=True, reduced_chi2=reduced_chi2_default, n_points=n_chi_default, dof=dof_default):
            auto_text = "on" if auto_scale else "off"
            return (
                f"Integrated {exp_img.shape[0]}x{exp_img.shape[1]} images | "
                f"valid pixels={int(np.sum(valid_mask))} | bins=500 | "
                f"x-range={q_min_plot:.4g} to {q_max_plot:.4g} {q_label} | source={q_source} | "
                f"auto-scale sim={auto_text} | sim scale={current_sim_scale:.6g} | reduced χ²={reduced_chi2:.6g} | N={n_points} | dof={dof}"
            )
        status = tk.StringVar(value=_status_text())
        # Keep the long status text on its own row.  If it is packed into the
        # same row as the controls, it can consume the whole width and push
        # later widgets such as the simulation scale entry outside the window.
        ttk.Label(controls, textvariable=status, anchor=tk.W).pack(side=tk.TOP, fill=tk.X)
        toolbar = ttk.Frame(controls)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))
        log_var = tk.BooleanVar(value=False)
        autoscale_var = tk.BooleanVar(value=True)
        sim_scale_var = tk.StringVar(value=f"{sim_scale_default:.6g}")
        def _get_sim_scale(show_warning=False):
            text = sim_scale_var.get().strip()
            try:
                value = float(text)
            except Exception:
                value = 1.0
            if (not np.isfinite(value)) or value <= 0:
                if show_warning:
                    messagebox.showwarning(
                        "SAXS integration",
                        "Simulation scale factor must be a positive finite number. Using 1.0 instead.",
                        parent=win,
                    )
                value = 1.0
                sim_scale_var.set("1")
            return value
        def redraw(log_y=False, auto_scale=True):
            ax.clear()
            ax_diff.clear()
            ax_chi.clear()
            y_exp = I_exp
            y_sim_base = I_sim
            ylabel = "azimuthal average intensity"
            title = "SAXS 1D integration"
            if auto_scale:
                sim_scale = _compute_sim_scale_to_exp(y_exp, y_sim_base)
                sim_scale_var.set(f"{sim_scale:.6g}")
                title += " (simulation auto-scaled)"
            else:
                sim_scale = _get_sim_scale(show_warning=False)
                title += " (manual simulation scale)"
            y_sim = y_sim_base * sim_scale
            sim_label = "masked theoretical" if abs(sim_scale - 1.0) < 1e-12 else f"masked theoretical × {sim_scale:.3g}"
            y_sim_on_exp = _interp_to_exp_grid(q_sim, y_sim)
            y_diff = y_exp - y_sim_on_exp
            n_fit_params = 1 if auto_scale else 0
            reduced_chi2, y_chi, n_points_chi, dof_chi = _compute_reduced_chi_square(y_exp, y_sim_on_exp, sigma_exp_chi, n_fit_params=n_fit_params)
            m_exp = _plot_mask(q_exp, y_exp)
            m_sim = _plot_mask(q_sim, y_sim)
            m_diff = _plot_mask(q_exp, y_diff)
            m_chi = _plot_mask(q_exp, y_chi)
            ax.plot(q_exp[m_exp], y_exp[m_exp], label="experimental")
            ax.plot(q_sim[m_sim], y_sim[m_sim], label=sim_label)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlim(q_min_plot, q_max_plot)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.xaxis.get_offset_text().set_visible(False)
            if log_y:
                ax.set_yscale("log")
            else:
                ax.set_yscale("linear")
            ax_diff.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
            ax_diff.plot(q_exp[m_diff], y_diff[m_diff], label="experimental - scaled simulation")
            ax_diff.set_ylabel("difference")
            ax_diff.set_xlim(q_min_plot, q_max_plot)
            ax_diff.grid(True, alpha=0.25)
            ax_diff.legend(loc="best")
            ax_diff.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax_diff.xaxis.get_offset_text().set_visible(False)
            ax_chi.plot(q_exp[m_chi], y_chi[m_chi], label=rf"$\chi^2$ contribution (reduced $\chi^2$={reduced_chi2:.4g})")
            ax_chi.set_xlabel(q_label)
            ax_chi.set_ylabel(r"$\chi^2$ contribution")
            ax_chi.set_xlim(q_min_plot, q_max_plot)
            ax_chi.grid(True, alpha=0.25)
            ax_chi.legend(loc="best")
            status.set(_status_text(current_sim_scale=sim_scale, auto_scale=auto_scale, reduced_chi2=reduced_chi2, n_points=n_points_chi, dof=dof_chi))
            canvas.draw_idle()
        def on_toggle():
            redraw(log_y=log_var.get(), auto_scale=autoscale_var.get())
        def on_manual_scale_apply():
            _get_sim_scale(show_warning=True)
            # Applying a typed value should switch to manual mode; otherwise the
            # auto-scale calculation would immediately overwrite the entry.
            if autoscale_var.get():
                autoscale_var.set(False)
            on_toggle()
        ttk.Checkbutton(toolbar, text="Auto-scale sim", variable=autoscale_var, command=on_toggle).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="Log y", variable=log_var, command=on_toggle).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(toolbar, text="Sim scale").pack(side=tk.LEFT, padx=(18, 4))
        sim_scale_entry = ttk.Entry(toolbar, textvariable=sim_scale_var, width=10)
        sim_scale_entry.pack(side=tk.LEFT)
        sim_scale_entry.bind("<Return>", lambda _event: on_manual_scale_apply())
        ttk.Button(toolbar, text="Apply", command=on_manual_scale_apply).pack(side=tk.LEFT, padx=(4, 0))
        def save_curves():
            save_path = filedialog.asksaveasfilename(
                parent=win,
                title="Save integrated curves",
                defaultextension=".npz",
                filetypes=[("NumPy archive", "*.npz"), ("CSV", "*.csv"), ("All files", "*.*")],
            )
            if not save_path:
                return
            save_path = Path(save_path)
            try:
                if autoscale_var.get():
                    sim_scale_to_save = _compute_sim_scale_to_exp(I_exp, I_sim)
                    sim_scale_var.set(f"{sim_scale_to_save:.6g}")
                else:
                    sim_scale_to_save = _get_sim_scale(show_warning=False)
                I_sim_scaled = I_sim * sim_scale_to_save
                I_sim_scaled_on_exp = _interp_to_exp_grid(q_sim, I_sim_scaled)
                I_difference = I_exp - I_sim_scaled_on_exp
                n_fit_params_save = 1 if autoscale_var.get() else 0
                reduced_chi2_save, I_chi, n_points_save, dof_save = _compute_reduced_chi_square(I_exp, I_sim_scaled_on_exp, sigma_exp_chi, n_fit_params=n_fit_params_save)
                poisson_enabled_save = bool(self.poisson_noise_var.get())
                poisson_peak_counts_save = self._parse_poisson_peak_counts()
                poisson_seed_save = self._parse_poisson_seed()
                if save_path.suffix.lower() == ".csv":
                    import csv as _csv
                    with save_path.open("w", newline="", encoding="utf-8") as f:
                        writer = _csv.writer(f)
                        writer.writerow([q_label, "I_experimental", "I_masked_theoretical", "I_masked_theoretical_scaled", "I_difference_experimental_minus_scaled_theoretical", "I_chi_square_contribution", "sigma_experimental", "simulation_scale_factor", "reduced_chi_square", "n_points_chi", "dof_chi", "poisson_noise_enabled", "poisson_peak_counts", "poisson_seed", "counts"])
                        for a, b, c, c_scaled, diff_val, chi_val, sigma_val, d in zip(q_exp, I_exp, I_sim, I_sim_scaled, I_difference, I_chi, sigma_exp_chi, counts_exp):
                            writer.writerow([a, b, c, c_scaled, diff_val, chi_val, sigma_val, sim_scale_to_save, reduced_chi2_save, n_points_save, dof_save, poisson_enabled_save, poisson_peak_counts_save, poisson_seed_save, d])
                else:
                    np.savez(
                        save_path,
                        q=q_exp,
                        I_experimental=I_exp,
                        I_masked_theoretical=I_sim,
                        I_masked_theoretical_scaled=I_sim_scaled,
                        I_difference_experimental_minus_scaled_theoretical=I_difference,
                        I_chi_square_contribution=I_chi,
                        sigma_experimental=sigma_exp_chi,
                        reduced_chi_square=reduced_chi2_save,
                        n_points_chi=n_points_save,
                        dof_chi=dof_save,
                        simulation_scale_factor=sim_scale_to_save,
                        poisson_noise_enabled=poisson_enabled_save,
                        poisson_peak_counts=poisson_peak_counts_save,
                        poisson_seed=poisson_seed_save,
                        counts=counts_exp,
                        q_label=q_label,
                    )
                status.set(f"Saved curves to {save_path}")
            except Exception as exc:
                messagebox.showerror("SAXS integration", f"Could not save curves.\n\n{type(exc).__name__}: {exc}", parent=win)
        ttk.Button(toolbar, text="Save curves...", command=save_curves).pack(side=tk.RIGHT)
    def _open_saxs_2d_feature_window(self):
        try:
            sim_img, q_u, q_v = self._get_current_simulation_masked_image_and_q()
        except Exception as exc:
            messagebox.showerror(
                "SAXS 2D feature comparison",
                f"Could not read the current simulation image.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        exp_img = getattr(self, "exp_image_data", None)
        if exp_img is None:
            messagebox.showwarning(
                "SAXS 2D feature comparison",
                "Load an experimental image first.",
                parent=self.root,
            )
            return
        exp_img = np.asarray(exp_img, dtype=float)
        sim_img = np.asarray(sim_img, dtype=float)
        if exp_img.ndim != 2 or sim_img.ndim != 2:
            messagebox.showerror(
                "SAXS 2D feature comparison",
                f"Both images must be 2D. experimental={exp_img.shape}, simulation={sim_img.shape}",
                parent=self.root,
            )
            return
        if exp_img.shape != sim_img.shape:
            messagebox.showerror(
                "SAXS 2D feature comparison",
                f"Experimental and simulation images have different shapes:\nexperimental={exp_img.shape}, simulation={sim_img.shape}",
                parent=self.root,
            )
            return
        try:
            valid_mask = _load_default_saxs_valid_mask(exp_img.shape)
            q_u_real, q_v_real, q_meta = _load_default_q_axes_from_poni(exp_img.shape)
            q_u_use = q_u_real if q_u_real is not None else q_u
            q_v_use = q_v_real if q_v_real is not None else q_v
            comparison = _compare_2d_saxs_features(
                exp_img,
                sim_img,
                q_u=q_u_use,
                q_v=q_v_use,
                valid_mask=valid_mask,
                n_bins=500,
                harmonic_orders=tuple(range(2, 7)),
            )
        except Exception as exc:
            messagebox.showerror(
                "SAXS 2D feature comparison",
                f"Could not compute 2D features.\n\n{type(exc).__name__}: {exc}",
                parent=self.root,
            )
            return
        metrics = comparison["metrics"]
        exp_profiles = comparison["exp_profiles"]
        sim_profiles = comparison["sim_profiles"]
        q_source = ".poni geometry" if 'q_u_real' in locals() and q_u_real is not None else "simulator/pixel q grid"
        win = tk.Toplevel(self.root)
        win.title("SAXS 2D feature comparison: experimental vs masked theoretical")
        win.geometry("1320x1180")
        fig = Figure(figsize=(12.8, 13.2), dpi=100)
        gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.88, 0.88, 0.60], hspace=0.38, wspace=0.28)
        ax_exp = fig.add_subplot(gs[0, 0])
        ax_sim = fig.add_subplot(gs[0, 1])
        ax_res = fig.add_subplot(gs[0, 2])
        ax_rad = fig.add_subplot(gs[1, 0])
        ax_a2 = fig.add_subplot(gs[1, 1])
        ax_a3 = fig.add_subplot(gs[1, 2])
        ax_a4 = fig.add_subplot(gs[2, 0])
        ax_a5 = fig.add_subplot(gs[2, 1])
        ax_a6 = fig.add_subplot(gs[2, 2])
        ax_polar = fig.add_subplot(gs[3, :])
        def _imshow_with_mask(ax, data, title, cmap=None):
            im = _imshow_signed_feature_panel(ax, data, comparison["common_mask"], title, cmap=cmap)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        _imshow_with_mask(ax_exp, comparison.get("exp_display", comparison["exp_z"]), "Exp")
        _imshow_with_mask(ax_sim, comparison.get("sim_display", comparison["sim_z"]), f"Sim × {metrics['simulation_scale_factor_raw_pixel_LS']:.3g}")
        _imshow_with_mask(ax_res, comparison["residual_z"], "Residual", cmap="coolwarm")
        q_vals = np.asarray(exp_profiles.get("q_raw", exp_profiles.get("q", np.array([]))), dtype=float)
        ax_rad.plot(q_vals, np.asarray(exp_profiles.get("radial_raw", exp_profiles.get("radial", np.array([]))), dtype=float), label="experimental")
        ax_rad.plot(np.asarray(sim_profiles.get("q_raw", sim_profiles.get("q", np.array([]))), dtype=float), np.asarray(sim_profiles.get("radial_raw", sim_profiles.get("radial", np.array([]))), dtype=float), label="simulation")
        ax_rad.set_title(f"2D radial | corr={_format_feature_metric(metrics.get('radial_profile_corr'))}, χ²red={_format_feature_metric(metrics.get('radial_chi2_reduced'))}")
        ax_rad.set_xlabel("q (Å⁻¹)")
        ax_rad.set_ylabel("raw radial intensity")
        ax_rad.grid(True, alpha=0.25)
        ax_rad.legend(loc="best", fontsize=8)
        _plot_selected_harmonic_profile(ax_a2, exp_profiles, sim_profiles, metrics, 2, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a3, exp_profiles, sim_profiles, metrics, 3, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a4, exp_profiles, sim_profiles, metrics, 4, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a5, exp_profiles, sim_profiles, metrics, 5, show_phase_error=True, fontsize=8)
        _plot_selected_harmonic_profile(ax_a6, exp_profiles, sim_profiles, metrics, 6, show_phase_error=True, fontsize=8)
        _plot_polar_match_summary(ax_polar, comparison.get("polar_data", {}), metrics, fontsize=8)
        canvas = _OriginalFigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        lower = ttk.Frame(win, padding=(8, 6))
        lower.pack(side=tk.BOTTOM, fill=tk.X)
        text = tk.Text(lower, height=11, wrap="word")
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(lower, orient="vertical", command=text.yview)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        text.configure(yscrollcommand=scrollbar.set)
        summary_lines = [
            "2D feature comparison summary",
            f"source q-grid: {q_source}",
            f"valid pixels: {metrics['valid_pixels']}",
            f"feature q-range: {_format_feature_metric(metrics.get('feature_q_min', np.nan))}–{_format_feature_metric(metrics.get('feature_q_max', np.nan))} Å⁻¹",
            f"harmonic q-range: {_format_feature_metric(metrics.get('harmonic_q_min', np.nan))}–{_format_feature_metric(metrics.get('harmonic_q_max', np.nan))} Å⁻¹",
            "",
            "Pixel-level metrics, after robust asinh + z-score normalization:",
            f"  Pearson correlation       = {_format_feature_metric(metrics['pixel_pearson_corr_asinh_z'])}",
            f"  Cosine similarity         = {_format_feature_metric(metrics['pixel_cosine_similarity_asinh_z'])}",
            f"  NRMSE                     = {_format_feature_metric(metrics['pixel_nrmse_asinh_z'])}",
            f"  SSIM, if scikit-image OK  = {_format_feature_metric(metrics['pixel_ssim_asinh_z'])}",
            "",
            "SAXS-specific features:",
            f"  Radial profile corr / NRMSE = {_format_feature_metric(metrics['radial_profile_corr'])} / {_format_feature_metric(metrics['radial_profile_nrmse'])}",
            f"  A2 anisotropy corr / NRMSE  = {_format_feature_metric(metrics['A2_anisotropy_corr'])} / {_format_feature_metric(metrics['A2_anisotropy_nrmse'])}",
            f"  A4 anisotropy corr / NRMSE  = {_format_feature_metric(metrics['A4_anisotropy_corr'])} / {_format_feature_metric(metrics['A4_anisotropy_nrmse'])}",
            f"  A2 phase error               = {_format_feature_metric(metrics.get('A2_phase_error_deg', np.nan))} deg  (exp/sim mean = {_format_feature_metric(metrics.get('A2_phase_mean_exp_deg', np.nan))} / {_format_feature_metric(metrics.get('A2_phase_mean_sim_deg', np.nan))} deg)",
            f"  A2–A6 weighted phase error   = {_format_feature_metric(metrics.get('harmonic_phase_error_weighted_A2_A6_deg', np.nan))} deg  (n = {_format_feature_metric(metrics.get('harmonic_phase_score_n_terms', np.nan))})",
            f"  A3 phase error               = {_format_feature_metric(metrics.get('A3_phase_error_deg', np.nan))} deg  (exp/sim mean = {_format_feature_metric(metrics.get('A3_phase_mean_exp_deg', np.nan))} / {_format_feature_metric(metrics.get('A3_phase_mean_sim_deg', np.nan))} deg)",
            f"  A4 phase error               = {_format_feature_metric(metrics.get('A4_phase_error_deg', np.nan))} deg  (exp/sim mean = {_format_feature_metric(metrics.get('A4_phase_mean_exp_deg', np.nan))} / {_format_feature_metric(metrics.get('A4_phase_mean_sim_deg', np.nan))} deg)",
            f"  A5 phase error               = {_format_feature_metric(metrics.get('A5_phase_error_deg', np.nan))} deg  (exp/sim mean = {_format_feature_metric(metrics.get('A5_phase_mean_exp_deg', np.nan))} / {_format_feature_metric(metrics.get('A5_phase_mean_sim_deg', np.nan))} deg)",
            f"  A6 phase error               = {_format_feature_metric(metrics.get('A6_phase_error_deg', np.nan))} deg  (exp/sim mean = {_format_feature_metric(metrics.get('A6_phase_mean_exp_deg', np.nan))} / {_format_feature_metric(metrics.get('A6_phase_mean_sim_deg', np.nan))} deg)",
            f"  Angular profile corr         = zero-shift {_format_feature_metric(metrics.get('angular_profile_corr_zero_shift', np.nan))}; best {_format_feature_metric(metrics.get('angular_xcorr_best_corr', np.nan))} at shift {_format_feature_metric(metrics.get('angular_xcorr_best_shift_deg', np.nan))} deg",
            f"  Polar q–φ corr / NRMSE      = zero-shift {_format_feature_metric(metrics.get('polar_corr_zero_shift', np.nan))} / {_format_feature_metric(metrics.get('polar_nrmse_zero_shift', np.nan))}; best {_format_feature_metric(metrics.get('polar_xcorr_best_corr', np.nan))} at shift {_format_feature_metric(metrics.get('polar_xcorr_best_shift_deg', np.nan))} deg (NRMSE {_format_feature_metric(metrics.get('polar_nrmse_best_shift', np.nan))})",
            f"  A2–A6 harmonics mean corr / NRMSE       = {_format_feature_metric(metrics.get('harmonics_mean_corr_A2_A6', np.nan))} / {_format_feature_metric(metrics.get('harmonics_mean_nrmse_A2_A6', np.nan))}",
            "",
            "Higher-order harmonic summary, q-averaged amplitude and profile agreement:",
        ]
        for row in comparison.get("harmonic_metric_rows", []):
            summary_lines.append(
                f"  A{int(row['order']):<2d}: qavg exp/sim/|Δ| = {_format_feature_metric(row['qavg_experimental'])} / {_format_feature_metric(row['qavg_simulation'])} / {_format_feature_metric(row['qavg_abs_difference'])};  corr / NRMSE = {_format_feature_metric(row['corr'])} / {_format_feature_metric(row['nrmse'])}"
            )
        summary_lines.extend(["", "Global moment features, experimental | simulation | abs difference:"])
        for key in comparison["global_experimental"].keys():
            ev = comparison["global_experimental"].get(key, float("nan"))
            sv = comparison["global_simulation"].get(key, float("nan"))
            dv = comparison["global_abs_difference"].get(key, float("nan"))
            summary_lines.append(f"  {key:18s}: {_format_feature_metric(ev):>10s} | {_format_feature_metric(sv):>10s} | {_format_feature_metric(dv):>10s}")
        text.insert("1.0", "\n".join(summary_lines))
        text.configure(state="disabled")
        def open_harmonic_heatmaps():
            orders = np.asarray(exp_profiles.get("harmonic_orders", []), dtype=int)
            exp_h = np.asarray(exp_profiles.get("harmonics", np.empty((0, 0))), dtype=float)
            sim_h = np.asarray(sim_profiles.get("harmonics", np.empty((0, 0))), dtype=float)
            if orders.size == 0 or exp_h.size == 0 or sim_h.size == 0:
                messagebox.showinfo("Higher-order harmonic heatmaps", "No harmonic data are available.", parent=win)
                return
            heat = tk.Toplevel(win)
            heat.title("Higher-order harmonic maps A_n(q), q=0.007–0.03")
            heat.geometry("1280x500")
            fig_h = Figure(figsize=(12.0, 4.2), dpi=100)
            gs_h = fig_h.add_gridspec(1, 3, wspace=0.28)
            ax1 = fig_h.add_subplot(gs_h[0, 0])
            ax2 = fig_h.add_subplot(gs_h[0, 1])
            ax3 = fig_h.add_subplot(gs_h[0, 2])
            q_prof = np.asarray(exp_profiles.get("q", np.arange(exp_h.shape[1], dtype=float)), dtype=float)
            if q_prof.size == 0:
                q_prof = np.arange(exp_h.shape[1], dtype=float)
            q0 = float(np.nanmin(q_prof)) if np.any(np.isfinite(q_prof)) else 0.0
            q1 = float(np.nanmax(q_prof)) if np.any(np.isfinite(q_prof)) else float(max(exp_h.shape[1] - 1, 1))
            extent = [q0, q1, float(orders[0]) - 0.5, float(orders[-1]) + 0.5]
            diff_h = np.abs(exp_h - sim_h)
            def _heat(ax, arr, title):
                arr = np.asarray(arr, dtype=float)
                if np.any(np.isfinite(arr)):
                    vmin, vmax = np.nanpercentile(arr[np.isfinite(arr)], [1, 99])
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                        vmin = vmax = None
                else:
                    vmin = vmax = None
                im = ax.imshow(arr, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.set_xlabel("q (Å⁻¹)")
                ax.set_ylabel("harmonic order n")
                ax.set_yticks(orders)
                fig_h.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            _heat(ax1, exp_h, "experimental A_n(q)")
            _heat(ax2, sim_h, "simulation A_n(q)")
            _heat(ax3, diff_h, "|Δ A_n(q)|")
            can_h = _OriginalFigureCanvasTkAgg(fig_h, master=heat)
            can_h.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            can_h.draw()
        def save_features():
            save_path = filedialog.asksaveasfilename(
                parent=win,
                title="Save 2D feature comparison",
                defaultextension=".npz",
                filetypes=[("NumPy archive", "*.npz"), ("CSV summary", "*.csv"), ("All files", "*.*")],
            )
            if not save_path:
                return
            save_path = Path(save_path)
            try:
                if save_path.suffix.lower() == ".csv":
                    with save_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["section", "name", "experimental", "simulation", "value_or_abs_difference"])
                        for k, v in metrics.items():
                            writer.writerow(["metric", k, "", "", v])
                        for k in comparison["global_experimental"].keys():
                            writer.writerow([
                                "global_feature",
                                k,
                                comparison["global_experimental"].get(k, np.nan),
                                comparison["global_simulation"].get(k, np.nan),
                                comparison["global_abs_difference"].get(k, np.nan),
                            ])
                        writer.writerow([])
                        harmonic_orders = np.asarray(exp_profiles.get("harmonic_orders", []), dtype=int)
                        header = ["q", "radial_exp", "radial_sim", "radial_counts_exp", "harmonic_counts_exp", "A2_exp", "A2_sim", "A4_exp", "A4_sim"]
                        for n in harmonic_orders:
                            header.extend([f"A{int(n)}_exp", f"A{int(n)}_sim"])
                        writer.writerow(header)
                        exp_h = np.asarray(exp_profiles.get("harmonics", np.empty((0, 0))), dtype=float)
                        sim_h = np.asarray(sim_profiles.get("harmonics", np.empty((0, 0))), dtype=float)
                        n_rows = len(exp_profiles["q"])
                        for i in range(n_rows):
                            row = [
                                exp_profiles["q"][i],
                                exp_profiles["radial"][i],
                                sim_profiles["radial"][i],
                                exp_profiles.get("counts", np.full(n_rows, np.nan))[i],
                                exp_profiles.get("harmonic_counts", np.full(n_rows, np.nan))[i],
                                exp_profiles["A2"][i],
                                sim_profiles["A2"][i],
                                exp_profiles["A4"][i],
                                sim_profiles["A4"][i],
                            ]
                            for j, _n in enumerate(harmonic_orders):
                                row.extend([
                                    exp_h[j, i] if j < exp_h.shape[0] else np.nan,
                                    sim_h[j, i] if j < sim_h.shape[0] else np.nan,
                                ])
                            writer.writerow(row)
                else:
                    np.savez(
                        save_path,
                        metrics=np.array(list(metrics.items()), dtype=object),
                        global_experimental=np.array(list(comparison["global_experimental"].items()), dtype=object),
                        global_simulation=np.array(list(comparison["global_simulation"].items()), dtype=object),
                        global_abs_difference=np.array(list(comparison["global_abs_difference"].items()), dtype=object),
                        harmonic_metric_rows=np.array(comparison.get("harmonic_metric_rows", []), dtype=object),
                        q=exp_profiles["q"],
                        q_raw=exp_profiles.get("q_raw", exp_profiles["q"]),
                        radial_experimental=exp_profiles["radial"],
                        radial_simulation=sim_profiles["radial"],
                        radial_raw_experimental=exp_profiles.get("radial_raw", exp_profiles["radial"]),
                        radial_raw_simulation=sim_profiles.get("radial_raw", sim_profiles["radial"]),
                        A2_experimental=exp_profiles["A2"],
                        A2_simulation=sim_profiles["A2"],
                        A4_experimental=exp_profiles["A4"],
                        A4_simulation=sim_profiles["A4"],
                        harmonic_orders=exp_profiles.get("harmonic_orders", np.array([], dtype=int)),
                        harmonics_experimental=exp_profiles.get("harmonics", np.empty((0, 0), dtype=float)),
                        harmonics_simulation=sim_profiles.get("harmonics", np.empty((0, 0), dtype=float)),
                        harmonic_qavg_experimental=exp_profiles.get("harmonic_qavg", np.array([], dtype=float)),
                        harmonic_qavg_simulation=sim_profiles.get("harmonic_qavg", np.array([], dtype=float)),
                        harmonic_counts_experimental=exp_profiles.get("harmonic_counts", np.array([], dtype=int)),
                        harmonic_counts_simulation=sim_profiles.get("harmonic_counts", np.array([], dtype=int)),
                        harmonic_q_min=metrics.get("harmonic_q_min", np.nan),
                        harmonic_q_max=metrics.get("harmonic_q_max", np.nan),
                        exp_display=comparison.get("exp_display", comparison["exp_z"]),
                        sim_display=comparison.get("sim_display", comparison["sim_z"]),
                        exp_z=comparison["exp_z"],
                        sim_z=comparison["sim_z"],
                        residual_z=comparison["residual_z"],
                        common_mask=comparison["common_mask"],
                        simulation_scale_factor=metrics["simulation_scale_factor_raw_pixel_LS"],
                    )
            except Exception as exc:
                messagebox.showerror("SAXS 2D feature comparison", f"Could not save features.\n\n{type(exc).__name__}: {exc}", parent=win)
        ttk.Button(lower, text="Save features...", command=save_features).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(lower, text="Show harmonic heatmaps...", command=open_harmonic_heatmaps).pack(side=tk.RIGHT, padx=(8, 0))
    def _on_dropdown_selected(self, _event=None):
        if self._dropdown_syncing:
            return
        name = self.combo_var.get()
        for idx, (tab_name, _module_name, _class_name) in enumerate(SIMULATORS):
            if tab_name == name:
                self.select_simulator(idx)
                return
    def _update_selector_state(self, selected_index):
        self._dropdown_syncing = True
        try:
            self.combo_var.set(SIMULATORS[selected_index][0])
        finally:
            self._dropdown_syncing = False
    def open_batch_dialog(self):
        if self.current_index is None or self.current_index not in self.loaded_apps:
            messagebox.showwarning("Batch auto save", "Load a simulator first.", parent=self.root)
            return
        app = self.loaded_apps[self.current_index]
        simulator_name = SIMULATORS[self.current_index][0]
        BatchSweepDialog(self.root, simulator_name, app, self.root)
    def open_auto_screen_dialog(self):
        if self.current_index is None or self.current_index not in self.loaded_apps:
            messagebox.showwarning("Auto screen", "Load a simulator first.", parent=self.root)
            return
        app = self.loaded_apps[self.current_index]
        simulator_name = SIMULATORS[self.current_index][0]
        AutoScreenDialog(self.root, simulator_name, app, self.root, noise_source=self)
    def select_simulator(self, index):
        if index < 0 or index >= len(SIMULATORS):
            return
        if self.current_index == index:
            return
        # Hide the previously selected simulator frame.
        if self.current_index is not None and self.current_index in self.host_frames:
            self.host_frames[self.current_index].grid_remove()
        self.current_index = index
        self._update_selector_state(index)
        if index not in self.host_frames:
            host = SimulatorHost(self.content)
            host.grid(row=0, column=0, sticky="nsew")
            self.host_frames[index] = host
            self._load_simulator(index, host)
        else:
            self.host_frames[index].grid(row=0, column=0, sticky="nsew")
        self._refresh_experimental_inset_after_simulator_change()
    def _load_simulator(self, index, host):
        tab_name, module_name, class_name = SIMULATORS[index]
        placeholder = ttk.Frame(host, padding=20)
        placeholder.pack(fill=tk.BOTH, expand=True)
        label = ttk.Label(placeholder, text=f"Loading {tab_name}...", font=("Arial", 12))
        label.pack(expand=True)
        self.root.update_idletasks()
        try:
            module = importlib.import_module(module_name)
            module.FigureCanvasTkAgg = ZoomableFigureCanvasTkAgg
            # Some simulator files implement their parameter panel with
            # canvas.bind_all("<MouseWheel>", ...). Without this guard,
            # wheel-zooming a plot can also scroll the parameter panel.
            scrollable_cls = getattr(module, "ScrollableFrame", None)
            if scrollable_cls is not None and not getattr(scrollable_cls, "_saxs_wheel_guarded", False):
                original_mousewheel = getattr(scrollable_cls, "_on_mousewheel", None)
                if original_mousewheel is not None:
                    def guarded_mousewheel(self_scroll, event, _orig=original_mousewheel):
                        if _is_descendant_of_registered_plot(getattr(event, "widget", None)):
                            return "break"
                        return _orig(self_scroll, event)
                    scrollable_cls._on_mousewheel = guarded_mousewheel
                    scrollable_cls._saxs_wheel_guarded = True
            cls = getattr(module, class_name)
            placeholder.destroy()
            app = cls(host)
            # Wrap update_plot so that every parameter change re-applies the stable
            # experimental-image layout. Without this, some simulator modules can
            # change subplot sizes/positions during update_plot, which causes the
            # experimental inset to overlap the simulation panels.
            original_update_plot = getattr(app, "update_plot", None)
            if callable(original_update_plot) and not getattr(app, "_experimental_update_wrapped", False):
                def wrapped_update_plot(*args, _orig=original_update_plot, _self=self, _app=app, **kwargs):
                    result = _orig(*args, **kwargs)
                    try:
                        _self._invalidate_poisson_result_cache(_app)
                    except Exception:
                        pass
                    try:
                        if not getattr(_self, "_suspend_poisson_processing_update", False):
                            _self._apply_poisson_noise_to_current_result(_app)
                    except Exception:
                        pass
                    try:
                        _self._refresh_experimental_inset_after_simulator_change()
                    except Exception:
                        pass
                    return result
                app.update_plot = wrapped_update_plot
                app._experimental_update_wrapped = True
            self.loaded_apps[index] = app
            try:
                self._apply_poisson_noise_to_current_result(app)
            except Exception:
                pass
            self._refresh_experimental_inset_after_simulator_change()
        except Exception as exc:
            placeholder.destroy()
            err = ttk.Frame(host, padding=20)
            err.pack(fill=tk.BOTH, expand=True)
            ttk.Label(
                err,
                text=f"Failed to load {tab_name}\n\n{type(exc).__name__}: {exc}",
                foreground="red",
                justify="left",
                wraplength=1000,
            ).pack(anchor="nw")
            messagebox.showerror("Load error", f"Failed to load {tab_name}:\n\n{exc}", parent=self.root)
def main():
    root = tk.Tk()
    IntegratedSAXSSimulator(root)
    root.mainloop()

def _run_gpu_smoke_test():
    log_path = os.environ.get("SAXS_SMOKE_LOG")
    if not log_path and getattr(sys, "frozen", False):
        log_path = str(Path.cwd() / "saxs_gpu_smoke.log")

    def _emit(message: str):
        print(message)
        if log_path:
            try:
                Path(log_path).write_text(message + "\n", encoding="utf-8")
            except Exception:
                pass

    try:
        import saxs_module as sm

        geom = sm.build_bvh("polyhedron", name="cube", radius=1.0)
        proj, du, dv, meta = sm.project_density_scaled(
            geom,
            ray_vector=(0.0, 0.0, 1.0),
            image_shape=(64, 64),
            object_scale=1.0,
            out_dtype="float32",
        )
        method = str(meta.get("projection_method", ""))
        nonzero = int(np.count_nonzero(proj))
        message = (
            "SAXS_GPU_SMOKE "
            f"method={method} shape={tuple(proj.shape)} nonzero={nonzero} "
            f"du={float(du):.8g} dv={float(dv):.8g}"
        )
        _emit(message)
        return 0 if method == "signed_raster_gpu" and nonzero > 0 else 2
    except Exception as exc:
        import traceback

        _emit(f"SAXS_GPU_SMOKE_ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        return 2

if __name__ == "__main__":
    if "--smoke-gpu" in sys.argv:
        sys.exit(_run_gpu_smoke_test())
    main()
