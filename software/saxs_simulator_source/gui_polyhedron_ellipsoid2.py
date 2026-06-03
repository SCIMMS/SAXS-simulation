
from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse

from saxs_module import (
    build_bvh,
    project_density,
    proj_ellipsoid,
    complex_amplitude_from_projected_density,
)

PONI_PATH = "post_beamtime_tmp_saxs.poni"
MASK_PATH = "SAXS_mask.npy"
IMG_SHAPE = (500, 500)
EPS = 1e-12


class ToolTip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.after_id = None
        widget.bind("<Enter>", self.schedule)
        widget.bind("<Leave>", self.hide)
        widget.bind("<ButtonPress>", self.hide)

    def schedule(self, event=None):
        self.unschedule()
        self.after_id = self.widget.after(self.delay, self.show)

    def unschedule(self):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show(self):
        if self.tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left", background="#ffffe0",
            relief="solid", borderwidth=1, padx=8, pady=5,
            font=("맑은 고딕", 9), wraplength=380,
        )
        label.pack()

    def hide(self, event=None):
        self.unschedule()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=510):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        if isinstance(event.widget, ttk.Entry) or event.widget.winfo_class() in ("TEntry", "Entry"):
            return
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            pass


class SaveFolderDialog:
    def __init__(self, parent, default_folder_name):
        self.parent = parent
        self.result = None
        self.top = tk.Toplevel(parent)
        self.top.title("Save result")
        self.top.transient(parent)
        self.top.grab_set()
        self.top.resizable(False, False)

        self.parent_dir = tk.StringVar(value="")
        self.folder_name = tk.StringVar(value=default_folder_name)
        self.full_path = tk.StringVar(value="")

        outer = ttk.Frame(self.top, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)

        ttk.Label(outer, text="Parent directory").grid(row=0, column=0, sticky="w")
        parent_row = ttk.Frame(outer)
        parent_row.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        parent_row.columnconfigure(0, weight=1)
        ttk.Entry(parent_row, textvariable=self.parent_dir, width=58).grid(row=0, column=0, sticky="ew")
        ttk.Button(parent_row, text="Browse", command=self.browse).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(outer, text="Result folder name").grid(row=2, column=0, sticky="w")
        folder_entry = ttk.Entry(outer, textvariable=self.folder_name, width=58)
        folder_entry.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(outer, text="Full save path").grid(row=4, column=0, sticky="w")
        ttk.Entry(outer, textvariable=self.full_path, width=58, state="readonly").grid(row=5, column=0, sticky="ew", pady=(2, 10))

        button_row = ttk.Frame(outer)
        button_row.grid(row=6, column=0, sticky="e")
        ttk.Button(button_row, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(button_row, text="Save", command=self.ok).pack(side=tk.RIGHT)

        self.parent_dir.trace_add("write", lambda *_: self.update_full_path())
        self.folder_name.trace_add("write", lambda *_: self.update_full_path())
        self.update_full_path()

        folder_entry.focus_set()
        folder_entry.selection_range(0, tk.END)
        self.top.bind("<Return>", lambda _event: self.ok())
        self.top.bind("<Escape>", lambda _event: self.cancel())
        self.center()
        parent.wait_window(self.top)

    def center(self):
        self.top.update_idletasks()
        x = self.parent.winfo_rootx() + max(0, (self.parent.winfo_width() - self.top.winfo_width()) // 2)
        y = self.parent.winfo_rooty() + max(0, (self.parent.winfo_height() - self.top.winfo_height()) // 2)
        self.top.geometry(f"+{x}+{y}")

    def browse(self):
        selected = filedialog.askdirectory(parent=self.top, title="Choose parent directory")
        if selected:
            self.parent_dir.set(selected)
            self.top.lift()
            self.top.focus_force()

    def sanitized_folder_name(self):
        name = self.folder_name.get().strip()
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        return name.strip(" .")

    def update_full_path(self):
        parent = self.parent_dir.get().strip()
        name = self.sanitized_folder_name()
        if parent and name:
            self.full_path.set(str(Path(parent) / name))
        elif parent:
            self.full_path.set(str(Path(parent)))
        else:
            self.full_path.set("")

    def ok(self):
        parent = self.parent_dir.get().strip()
        name = self.sanitized_folder_name()
        if not parent:
            messagebox.showerror("Save error", "Parent directory를 선택해주세요.", parent=self.top)
            return
        parent_path = Path(parent)
        if not parent_path.exists() or not parent_path.is_dir():
            messagebox.showerror("Save error", "선택한 parent directory가 존재하지 않습니다.", parent=self.top)
            return
        if not name:
            messagebox.showerror("Save error", "Result folder name을 입력해주세요.", parent=self.top)
            return
        target = parent_path / name
        if target.exists():
            ok = messagebox.askyesno(
                "Folder exists",
                f"이미 같은 이름의 폴더가 있습니다.\n계속 저장하시겠습니까?\n\n{target}",
                parent=self.top,
            )
            if not ok:
                return
        self.result = target
        self.top.destroy()

    def cancel(self):
        self.result = None
        self.top.destroy()


def safe_log(x):
    return np.log(np.asarray(x, dtype=np.float64) + EPS)


def get_amplitude(proj, du, dv, scale_factor, translation_u, translation_v):
    A, q_u, q_v, meta = complex_amplitude_from_projected_density(
        projected_density=proj, du=du, dv=dv, scale_factor=scale_factor,
        translation_u=translation_u, translation_v=translation_v,
        poni_path=PONI_PATH, detector_image_shape=IMG_SHAPE,
        q_unit="A^-1", center_mode="beam", row_origin="upper",
    )
    return A, q_u, q_v, meta


def projection_extent(proj, du, dv, scale_factor, translation_u, translation_v):
    n_v, n_u = proj.shape
    width_u = float(n_u) * float(du) * float(scale_factor)
    width_v = float(n_v) * float(dv) * float(scale_factor)
    return (
        float(translation_u) - 0.5 * width_u,
        float(translation_u) + 0.5 * width_u,
        float(translation_v) - 0.5 * width_v,
        float(translation_v) + 0.5 * width_v,
    )


def combined_real_space_limits(extents, margin_fraction=0.08):
    left = min(e[0] for e in extents)
    right = max(e[1] for e in extents)
    bottom = min(e[2] for e in extents)
    top = max(e[3] for e in extents)
    span = max(right - left, top - bottom, 1.0)
    margin = span * margin_fraction
    cx = 0.5 * (left + right)
    cy = 0.5 * (bottom + top)
    half = 0.5 * span + margin
    return (cx - half, cx + half, cy - half, cy + half)


def make_polyhedron(params):
    axis = np.array([params["poly_axis_x"], params["poly_axis_y"], params["poly_axis_z"]], dtype=np.float64)
    if np.linalg.norm(axis) < 1e-12:
        raise ValueError("polyhedron rotation axis cannot be zero vector.")
    bvh = build_bvh(
        "polyhedron",
        name=params["polyhedron_name"],
        radius=params["poly_radius"],
        rotations=[(axis.tolist(), np.deg2rad(params["poly_angle_deg"]))],
    )
    proj, du, dv, meta_proj = project_density(
        bvh, ray_vector=(0.0, 0.0, 1.0), roll_rad=0.0,
        image_shape=IMG_SHAPE, out_dtype="float32",
    )
    A, q_u, q_v, meta = get_amplitude(
        proj=proj, du=du, dv=dv,
        scale_factor=params["poly_scale"],
        translation_u=params["poly_tu"],
        translation_v=params["poly_tv"],
    )
    return proj, du, dv, A, q_u, q_v, meta_proj, meta


def make_ellipsoid(prefix, params):
    proj, du, dv = proj_ellipsoid(
        b=params[f"{prefix}_b"],
        angle=np.deg2rad(params[f"{prefix}_angle_deg"]),
        img_shape=IMG_SHAPE,
    )
    A, q_u, q_v, meta = get_amplitude(
        proj=proj, du=du, dv=dv,
        scale_factor=params[f"{prefix}_scale"],
        translation_u=params[f"{prefix}_tu"],
        translation_v=params[f"{prefix}_tv"],
    )
    return proj, du, dv, A, q_u, q_v, meta


def run_simulation(params):
    poly_proj, poly_du, poly_dv, A_poly, q_u, q_v, meta_proj_poly, meta_poly = make_polyhedron(params)
    ell1_proj, ell1_du, ell1_dv, A_ell1, _, _, meta_ell1 = make_ellipsoid("ell1", params)
    ell2_proj, ell2_du, ell2_dv, A_ell2, _, _, meta_ell2 = make_ellipsoid("ell2", params)

    A_total = (
        params["poly_weight"] * A_poly
        + params["ell1_weight"] * A_ell1
        + params["ell2_weight"] * A_ell2
    )
    I_total = np.abs(A_total) ** 2

    return {
        "poly_proj": poly_proj, "ell1_proj": ell1_proj, "ell2_proj": ell2_proj,
        "poly_du": poly_du, "poly_dv": poly_dv,
        "ell1_du": ell1_du, "ell1_dv": ell1_dv,
        "ell2_du": ell2_du, "ell2_dv": ell2_dv,
        "A_poly": A_poly, "A_ell1": A_ell1, "A_ell2": A_ell2,
        "A_total": A_total, "I_total": I_total,
        "q_u": q_u, "q_v": q_v,
        "meta_proj_poly": meta_proj_poly, "meta_poly": meta_poly,
        "meta_ell1": meta_ell1, "meta_ell2": meta_ell2,
    }


class PolyhedronTwoEllipsoidsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Polyhedron + Two Ellipsoids SAXS Simulator")
        self.root.geometry("1750x900")
        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.vars = {}
        self.entries = {}
        self.slider_meta = {}
        self.sliders = {}
        self._setting_value = set()

        self._build_ui()
        self.update_plot()

    def defaults(self):
        return {
            "poly_radius": 1.0, "poly_axis_x": 1.0, "poly_axis_y": 1.0, "poly_axis_z": 1.0,
            "poly_angle_deg": np.rad2deg(0.6), "poly_scale": 5000.0,
            "poly_tu": -1500.0, "poly_tv": 0.0, "poly_weight": 1.0,
            "ell1_b": 0.5, "ell1_angle_deg": 35.0, "ell1_scale": 1200.0,
            "ell1_tu": 1500.0, "ell1_tv": 500.0, "ell1_weight": 3.0,
            "ell2_b": 0.75, "ell2_angle_deg": -35.0, "ell2_scale": 900.0,
            "ell2_tu": 2500.0, "ell2_tv": -700.0, "ell2_weight": 2.0,
        }

    def tooltip(self, name):
        tips = {
            "polyhedron_name": "Object 1으로 사용할 정다면체 종류입니다. tetra, cube, octa, dodeca, icosa 중 선택합니다.",
            "poly_radius": "polyhedron의 크기입니다. 값이 커지면 real-space object가 커지고 q-space feature는 좁아집니다.",
            "poly_axis_x": "polyhedron 3D 회전축의 x 성분입니다.",
            "poly_axis_y": "polyhedron 3D 회전축의 y 성분입니다.",
            "poly_axis_z": "polyhedron 3D 회전축의 z 성분입니다.",
            "poly_angle_deg": "polyhedron을 지정한 회전축 기준으로 회전시키는 각도입니다. 단위는 degree입니다.",
            "poly_scale": "polyhedron의 실공간 크기 scale입니다.",
            "poly_tu": "polyhedron의 real-space u 방향 translation입니다.",
            "poly_tv": "polyhedron의 real-space v 방향 translation입니다.",
            "poly_weight": "polyhedron complex amplitude의 가중치입니다.",
        }
        for prefix in ("ell1", "ell2"):
            label = "ellipsoid 1" if prefix == "ell1" else "ellipsoid 2"
            tips.update({
                f"{prefix}_b": f"{label}의 단축 비율입니다. b=1이면 원형 projection에 가깝습니다.",
                f"{prefix}_angle_deg": f"{label} projected density의 detector plane 내 회전 각도입니다.",
                f"{prefix}_scale": f"{label}의 실공간 크기 scale입니다.",
                f"{prefix}_tu": f"{label}의 real-space u 방향 translation입니다.",
                f"{prefix}_tv": f"{label}의 real-space v 방향 translation입니다.",
                f"{prefix}_weight": f"{label} complex amplitude의 가중치입니다.",
            })
        return tips.get(name, "")

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        scroll = ScrollableFrame(main, width=510)
        scroll.pack(side=tk.LEFT, fill=tk.Y)
        controls = scroll.inner

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(controls, text="Polyhedron + Two Ellipsoids Parameters", font=("Arial", 13, "bold")).pack(anchor="w", pady=(4, 10), padx=8)

        for name, value in self.defaults().items():
            self.vars[name] = tk.DoubleVar(value=value)

        poly_frame = ttk.LabelFrame(controls, text="Object 1: Polyhedron", padding=6)
        poly_frame.pack(fill=tk.X, padx=8, pady=4)

        name_label = ttk.Label(poly_frame, text="polyhedron_name")
        name_label.pack(anchor="w")
        ToolTip(name_label, self.tooltip("polyhedron_name"))

        self.name_var = tk.StringVar(value="dodeca")
        combo = ttk.Combobox(poly_frame, textvariable=self.name_var, values=["tetra", "cube", "octa", "dodeca", "icosa"], state="readonly")
        combo.pack(fill=tk.X, pady=(2, 4))
        combo.bind("<<ComboboxSelected>>", lambda _event: self.schedule())

        self._add(poly_frame, "poly_radius", 0.2, 3.0, 0.1)
        self._add(poly_frame, "poly_axis_x", -2.0, 2.0, 0.1)
        self._add(poly_frame, "poly_axis_y", -2.0, 2.0, 0.1)
        self._add(poly_frame, "poly_axis_z", -2.0, 2.0, 0.1)
        self._add(poly_frame, "poly_angle_deg", -180.0, 180.0, 1.0)
        self._add(poly_frame, "poly_scale", 50.0, 20000.0, 50.0)
        self._add(poly_frame, "poly_tu", -8000.0, 8000.0, 50.0)
        self._add(poly_frame, "poly_tv", -8000.0, 8000.0, 50.0)
        self._add(poly_frame, "poly_weight", 0.0, 50.0, 0.5)

        for prefix, title in [("ell1", "Object 2: Ellipsoid 1"), ("ell2", "Object 3: Ellipsoid 2")]:
            frame = ttk.LabelFrame(controls, text=title, padding=6)
            frame.pack(fill=tk.X, padx=8, pady=4)
            self._add(frame, f"{prefix}_b", 0.05, 1.0, 0.01)
            self._add(frame, f"{prefix}_angle_deg", -180.0, 180.0, 1.0)
            self._add(frame, f"{prefix}_scale", 50.0, 10000.0, 50.0)
            self._add(frame, f"{prefix}_tu", -8000.0, 8000.0, 50.0)
            self._add(frame, f"{prefix}_tv", -8000.0, 8000.0, 50.0)
            self._add(frame, f"{prefix}_weight", 0.0, 50.0, 0.5)

        ttk.Button(controls, text="Save current result", command=self.save).pack(fill=tk.X, padx=8, pady=(10, 4))
        ttk.Button(controls, text="Reset", command=self.reset).pack(fill=tk.X, padx=8, pady=4)

        self.info = tk.Text(controls, width=54, height=10)
        self.info.pack(fill=tk.X, padx=8, pady=(10, 10))

        self.fig, self.ax = plt.subplots(1, 4, figsize=(15.8, 4.4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add(self, parent, name, vmin, vmax, resolution):
        self.slider_meta[name] = {"vmin": float(vmin), "vmax": float(vmax), "step": float(resolution)}
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        f.columnconfigure(2, weight=1)

        label = ttk.Label(f, text=name, width=14)
        label.grid(row=0, column=0, sticky="w")
        ToolTip(label, self.tooltip(name))

        ttk.Button(f, text="◀", width=3, command=lambda key=name: self.step_value(key, -1)).grid(row=0, column=1, padx=(0, 2))
        slider = ttk.Scale(f, from_=vmin, to=vmax, orient=tk.HORIZONTAL, variable=self.vars[name], command=lambda _value, key=name: self.on_slider_change(key))
        slider.grid(row=0, column=2, sticky="ew")
        self.sliders[name] = slider
        ttk.Button(f, text="▶", width=3, command=lambda key=name: self.step_value(key, 1)).grid(row=0, column=3, padx=(2, 4))

        entry = ttk.Entry(f, width=10)
        entry.insert(0, f"{self.vars[name].get():.6g}")
        entry.grid(row=0, column=4, sticky="e")
        self.entries[name] = entry

        entry.bind("<Return>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<FocusOut>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<Up>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Right>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Down>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<Left>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<MouseWheel>", lambda event, key=name: self.on_entry_mousewheel(event, key))
        entry.bind("<Button-4>", lambda event, key=name: self.on_entry_step_key(event, key, 1))
        entry.bind("<Button-5>", lambda event, key=name: self.on_entry_step_key(event, key, -1))

    def clamp_to_range(self, name, value):
        meta = self.slider_meta[name]
        return min(max(float(value), meta["vmin"]), meta["vmax"])

    def round_to_step(self, name, value):
        meta = self.slider_meta[name]
        step = meta["step"]
        vmin = meta["vmin"]
        value = self.clamp_to_range(name, value)
        if step > 0:
            value = vmin + round((value - vmin) / step) * step
            value = self.clamp_to_range(name, value)
            decimals = max(0, int(np.ceil(-np.log10(step))) + 2) if step < 1 else 6
            value = round(value, decimals)
        return value

    def set_value(self, name, value, schedule=True):
        value = self.round_to_step(name, value)
        self._setting_value.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._setting_value.discard(name)
        self.sync_entry(name)
        if schedule:
            self.schedule()

    def step_value(self, name, direction):
        meta = self.slider_meta[name]
        try:
            current = float(self.entries[name].get())
        except Exception:
            current = float(self.vars[name].get())
        self.set_value(name, current + float(direction) * meta["step"])
        self.entries[name].focus_set()
        self.entries[name].selection_range(0, tk.END)

    def on_entry_step_key(self, event, name, direction):
        self.step_value(name, direction)
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = 1 if event.delta > 0 else -1
        self.step_value(name, direction)
        return "break"

    def on_slider_change(self, name):
        if name in self._setting_value:
            return
        self.set_value(name, self.vars[name].get(), schedule=True)

    def on_entry_change(self, name):
        try:
            value = float(self.entries[name].get())
        except ValueError:
            messagebox.showerror("Input error", f"{name} must be numeric.")
            self.sync_entry(name)
            return
        self.set_value(name, value)

    def sync_entry(self, name):
        entry = self.entries[name]
        entry.delete(0, tk.END)
        entry.insert(0, f"{self.vars[name].get():.6g}")

    def schedule(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(700, self.update_plot)

    def params(self):
        p = {k: float(v.get()) for k, v in self.vars.items()}
        p["polyhedron_name"] = self.name_var.get()
        return p

    def reset(self):
        self.name_var.set("dodeca")
        for name, value in self.defaults().items():
            self.vars[name].set(value)
            self.sync_entry(name)
        self.update_plot()

    def draw_translated_density_sum(self, ax, result, p):
        ax.clear()
        ax.set_title("translated projected density sum")
        ax.set_aspect("equal", adjustable="box")

        extent_poly = projection_extent(result["poly_proj"], result["poly_du"], result["poly_dv"], p["poly_scale"], p["poly_tu"], p["poly_tv"])
        extent_ell1 = projection_extent(result["ell1_proj"], result["ell1_du"], result["ell1_dv"], p["ell1_scale"], p["ell1_tu"], p["ell1_tv"])
        extent_ell2 = projection_extent(result["ell2_proj"], result["ell2_du"], result["ell2_dv"], p["ell2_scale"], p["ell2_tu"], p["ell2_tv"])

        ax.imshow(p["poly_weight"] * result["poly_proj"], extent=extent_poly, origin="lower", interpolation="nearest")
        ax.imshow(p["ell1_weight"] * result["ell1_proj"], extent=extent_ell1, origin="lower", interpolation="nearest", alpha=0.65)
        ax.imshow(p["ell2_weight"] * result["ell2_proj"], extent=extent_ell2, origin="lower", interpolation="nearest", alpha=0.65)

        xmin, xmax, ymin, ymax = combined_real_space_limits([extent_poly, extent_ell1, extent_ell2])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("real-space u")
        ax.set_ylabel("real-space v")
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_position_preview(self, ax, p):
        ax.clear()
        ax.set_title("real-space placement preview")
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, linewidth=0.6)
        ax.axvline(0, linewidth=0.6)

        result = getattr(self, "last_result", None)
        extents = []
        if result is not None:
            extents = [
                projection_extent(result["poly_proj"], result["poly_du"], result["poly_dv"], p["poly_scale"], p["poly_tu"], p["poly_tv"]),
                projection_extent(result["ell1_proj"], result["ell1_du"], result["ell1_dv"], p["ell1_scale"], p["ell1_tu"], p["ell1_tv"]),
                projection_extent(result["ell2_proj"], result["ell2_du"], result["ell2_dv"], p["ell2_scale"], p["ell2_tu"], p["ell2_tv"]),
            ]

        if extents:
            poly_size = max(extents[0][1] - extents[0][0], extents[0][3] - extents[0][2])
        else:
            poly_size = 2.0 * p["poly_scale"]
        poly_patch = Ellipse((p["poly_tu"], p["poly_tv"]), width=poly_size, height=poly_size,
                             angle=0.0, fill=False, linewidth=1.5)
        ax.add_patch(poly_patch)
        ax.text(p["poly_tu"], p["poly_tv"], "Poly", ha="center", va="center")

        for idx, (prefix, label) in enumerate([("ell1", "Ell1"), ("ell2", "Ell2")], start=1):
            if extents:
                width = extents[idx][1] - extents[idx][0]
                height = width * p[f"{prefix}_b"]
            else:
                width = 2.0 * p[f"{prefix}_scale"]
                height = 2.0 * p[f"{prefix}_scale"] * p[f"{prefix}_b"]
            patch = Ellipse((p[f"{prefix}_tu"], p[f"{prefix}_tv"]), width=width, height=height,
                            angle=p[f"{prefix}_angle_deg"], fill=False, linewidth=1.5)
            ax.add_patch(patch)
            ax.text(p[f"{prefix}_tu"], p[f"{prefix}_tv"], label, ha="center", va="center")

        if extents:
            xmin, xmax, ymin, ymax = combined_real_space_limits(extents)
        else:
            centers = np.array([[p["poly_tu"], p["poly_tv"]], [p["ell1_tu"], p["ell1_tv"]], [p["ell2_tu"], p["ell2_tv"]]], dtype=float)
            max_abs = max(1.0, float(np.max(np.abs(centers))) * 1.25)
            xmin, xmax, ymin, ymax = -max_abs, max_abs, -max_abs, max_abs
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("u translation")
        ax.set_ylabel("v translation")

    def update_plot(self):
        self.update_job = None
        try:
            p = self.params()
            result = run_simulation(p)
        except Exception as e:
            self.show_error(e)
            return

        I_total = result["I_total"]
        I_masked = I_total * self.mask
        result["I_masked"] = I_masked
        result["parameters"] = p
        self.last_result = result

        self.draw_translated_density_sum(self.ax[0], result, p)

        images = [safe_log(I_total), I_masked]
        titles = ["log theoretical: |w_poly A_poly + w_ell1 A_ell1 + w_ell2 A_ell2|²", "masked theoretical"]
        for ax, im, title in zip(self.ax[1:3], images, titles):
            ax.clear()
            ax.matshow(im)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.draw_position_preview(self.ax[3], p)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        q_u = result["q_u"]
        q_v = result["q_v"]
        d1u = p["ell1_tu"] - p["poly_tu"]
        d1v = p["ell1_tv"] - p["poly_tv"]
        d2u = p["ell2_tu"] - p["poly_tu"]
        d2v = p["ell2_tv"] - p["poly_tv"]
        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"polyhedron = {p['polyhedron_name']}\n"
            f"A_total = {p['poly_weight']}*A_poly + {p['ell1_weight']}*A_ell1 + {p['ell2_weight']}*A_ell2\n\n"
            f"poly position = ({p['poly_tu']:.6g}, {p['poly_tv']:.6g})\n"
            f"ell1 position = ({p['ell1_tu']:.6g}, {p['ell1_tv']:.6g})\n"
            f"ell2 position = ({p['ell2_tu']:.6g}, {p['ell2_tv']:.6g})\n"
            f"poly→ell1 Δu, Δv = ({d1u:.6g}, {d1v:.6g})\n"
            f"poly→ell2 Δu, Δv = ({d2u:.6g}, {d2v:.6g})\n\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dialog = SaveFolderDialog(self.root, f"polyhedron_plus_two_ellipsoids_sim_{timestamp}")
        d = dialog.result
        if d is None:
            return
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save error", f"Could not create save folder:\n{d}\n\n{e}")
            return
        try:
            self.save_translated_density_sum(d / "01_translated_projected_density_sum.png")
            self.save_image(d / "02_log_theoretical.png", safe_log(self.last_result["I_total"]), "log theoretical")
            self.save_image(d / "03_masked_theoretical.png", self.last_result["I_masked"], "masked theoretical")
            self.save_position_preview(d / "04_real_space_placement_preview.png")
            self.save_params(d / "parameters.txt")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save result:\n{d}\n\n{e}")
            return
        messagebox.showinfo("Saved", f"Saved result to:\n{d}")

    def save_translated_density_sum(self, path):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.draw_translated_density_sum(ax, self.last_result, self.last_result["parameters"])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def save_position_preview(self, path):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.draw_position_preview(ax, self.last_result["parameters"])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def save_image(self, path, img, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def save_params(self, path):
        p = self.last_result["parameters"]
        q_u = self.last_result["q_u"]
        q_v = self.last_result["q_v"]
        lines = [
            "Polyhedron + Two Ellipsoids SAXS simulation parameters",
            "=" * 64,
            "",
            "[Merge equation]",
            "A_total = poly_weight * A_polyhedron + ell1_weight * A_ellipsoid1 + ell2_weight * A_ellipsoid2",
            "I_total = |A_total|^2",
            "",
            "[Object 1: Polyhedron]",
            f"polyhedron_name = {p['polyhedron_name']}",
            f"poly_radius = {p['poly_radius']}",
            f"poly_rotation_axis = [{p['poly_axis_x']}, {p['poly_axis_y']}, {p['poly_axis_z']}]",
            f"poly_angle_deg = {p['poly_angle_deg']}",
            f"poly_scale = {p['poly_scale']}",
            f"poly_translation_u = {p['poly_tu']}",
            f"poly_translation_v = {p['poly_tv']}",
            f"poly_weight = {p['poly_weight']}",
            "",
        ]
        for prefix, title in [("ell1", "Object 2: Ellipsoid 1"), ("ell2", "Object 3: Ellipsoid 2")]:
            lines.extend([
                f"[{title}]",
                f"{prefix}_b = {p[f'{prefix}_b']}",
                f"{prefix}_angle_deg = {p[f'{prefix}_angle_deg']}",
                f"{prefix}_scale = {p[f'{prefix}_scale']}",
                f"{prefix}_translation_u = {p[f'{prefix}_tu']}",
                f"{prefix}_translation_v = {p[f'{prefix}_tv']}",
                f"{prefix}_weight = {p[f'{prefix}_weight']}",
                "",
            ])
        lines.extend([
            "[Files]",
            f"poni_path = {PONI_PATH}",
            f"mask_path = {MASK_PATH}",
            "",
            "[q axes]",
            f"q_u_min = {float(q_u.min())}",
            f"q_u_max = {float(q_u.max())}",
            f"q_v_min = {float(q_v.min())}",
            f"q_v_max = {float(q_v.max())}",
        ])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def show_error(self, e):
        for ax in self.ax:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax[1].text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center", transform=self.ax[1].transAxes, wrap=True)
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = PolyhedronTwoEllipsoidsGUI(root)
    root.mainloop()
