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
    proj_ellipsoid,
    build_bvh,
    project_density,
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
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=5,
            font=("맑은 고딕", 9),
            wraplength=360,
        )
        label.pack()

    def hide(self, event=None):
        self.unschedule()
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None



class SaveFolderDialog:
    def __init__(self, parent, default_folder_name):
        self.parent = parent
        self.default_folder_name = default_folder_name
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

        ttk.Label(outer, text="Parent directory").grid(row=0, column=0, sticky="w")
        parent_row = ttk.Frame(outer)
        parent_row.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        parent_row.columnconfigure(0, weight=1)

        self.parent_entry = ttk.Entry(parent_row, textvariable=self.parent_dir, width=56)
        self.parent_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(parent_row, text="Browse", command=self.browse).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(outer, text="Result folder name").grid(row=2, column=0, sticky="w")
        folder_entry = ttk.Entry(outer, textvariable=self.folder_name, width=56)
        folder_entry.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(outer, text="Full save path").grid(row=4, column=0, sticky="w")
        full_entry = ttk.Entry(outer, textvariable=self.full_path, width=56, state="readonly")
        full_entry.grid(row=5, column=0, sticky="ew", pady=(2, 10))

        button_row = ttk.Frame(outer)
        button_row.grid(row=6, column=0, sticky="e")
        ttk.Button(button_row, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(button_row, text="Save", command=self.ok).pack(side=tk.RIGHT)

        outer.columnconfigure(0, weight=1)

        self.parent_dir.trace_add("write", lambda *_: self.update_full_path())
        self.folder_name.trace_add("write", lambda *_: self.update_full_path())
        self.update_full_path()

        folder_entry.focus_set()
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

    def sanitized_folder_name(self):
        name = self.folder_name.get().strip()
        for ch in '<>:"/\\|?*':
            name = name.replace(ch, "_")
        return name

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
        if not Path(parent).exists():
            messagebox.showerror("Save error", "선택한 parent directory가 존재하지 않습니다.", parent=self.top)
            return
        if not name:
            messagebox.showerror("Save error", "Result folder name을 입력해주세요.", parent=self.top)
            return

        target = Path(parent) / name
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
        projected_density=proj,
        du=du,
        dv=dv,
        scale_factor=scale_factor,
        translation_u=translation_u,
        translation_v=translation_v,
        poni_path=PONI_PATH,
        detector_image_shape=IMG_SHAPE,
        q_unit="A^-1",
        center_mode="beam",
        row_origin="upper",
    )
    return A, q_u, q_v, meta


def projection_extent(proj, du, dv, scale_factor, translation_u, translation_v):
    """Return imshow extent for projected density after real-space scale and translation."""
    n_v, n_u = proj.shape
    width_u = float(n_u) * float(du) * float(scale_factor)
    width_v = float(n_v) * float(dv) * float(scale_factor)
    left = float(translation_u) - 0.5 * width_u
    right = float(translation_u) + 0.5 * width_u
    bottom = float(translation_v) - 0.5 * width_v
    top = float(translation_v) + 0.5 * width_v
    return (left, right, bottom, top)


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


def run_simulation(params):
    # --------------------------------------------------
    # Object 1: ellipsoid
    # --------------------------------------------------
    proj1, du1, dv1 = proj_ellipsoid(
        b=params["ell_b"],
        angle=np.deg2rad(params["ell_angle_deg"]),
        img_shape=IMG_SHAPE,
    )

    A1, q_u, q_v, meta1 = get_amplitude(
        proj=proj1,
        du=du1,
        dv=dv1,
        scale_factor=params["ell_scale"],
        translation_u=params["ell_tu"],
        translation_v=params["ell_tv"],
    )

    # --------------------------------------------------
    # Object 2: polyhedron
    # --------------------------------------------------
    axis = np.array(
        [params["poly_axis_x"], params["poly_axis_y"], params["poly_axis_z"]],
        dtype=np.float64,
    )

    if np.linalg.norm(axis) < 1e-12:
        raise ValueError("polyhedron rotation axis cannot be zero vector.")

    bvh = build_bvh(
        "polyhedron",
        name=params["polyhedron_name"],
        radius=params["poly_radius"],
        rotations=[
            (axis.tolist(), np.deg2rad(params["poly_angle_deg"])),
        ],
    )

    proj2, du2, dv2, meta_proj2 = project_density(
        bvh,
        ray_vector=(0.0, 0.0, 1.0),
        roll_rad=0.0,
        image_shape=IMG_SHAPE,
        out_dtype="float32",
    )

    A2, _, _, meta2 = get_amplitude(
        proj=proj2,
        du=du2,
        dv=dv2,
        scale_factor=params["poly_scale"],
        translation_u=params["poly_tu"],
        translation_v=params["poly_tv"],
    )

    # --------------------------------------------------
    # Coherent merging
    # --------------------------------------------------
    A_total = params["ell_weight"] * A1 + params["poly_weight"] * A2
    I_total = np.abs(A_total) ** 2

    proj_sum = params["ell_weight"] * proj1 + params["poly_weight"] * proj2

    return {
        "proj1": proj1,
        "proj2": proj2,
        "proj_sum": proj_sum,
        "du1": du1,
        "dv1": dv1,
        "du2": du2,
        "dv2": dv2,
        "A1": A1,
        "A2": A2,
        "A_total": A_total,
        "I_total": I_total,
        "q_u": q_u,
        "q_v": q_v,
        "meta1": meta1,
        "meta2": meta2,
        "meta_proj2": meta_proj2,
    }


class MergedObjectsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Merged Objects SAXS Simulator")
        self.root.geometry("1600x850")

        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.vars = {}
        self.entries = {}
        self.sliders = {}
        self.slider_meta = {}
        self._slider_updating = set()

        self._build_ui()
        self.update_plot()

    def tooltip(self, name):
        tips = {
            "ell_b": (
                "ellipsoid의 단축 비율입니다.\n"
                "b=1이면 원형/sphere projection이고, 작아질수록 더 납작해집니다."
            ),
            "ell_angle_deg": (
                "ellipsoid projected density의 detector plane 내 회전 각도입니다.\n"
                "3D tilt가 아니라 in-plane rotation입니다."
            ),
            "ell_scale": (
                "ellipsoid object의 실공간 크기 스케일입니다.\n"
                "값이 커질수록 q-space 패턴은 중심부로 좁게 모입니다."
            ),
            "ell_tu": (
                "ellipsoid의 real-space u 방향 translation입니다.\n"
                "단일 object intensity에는 안 보이지만, coherent merge에서는 interference phase를 바꿉니다."
            ),
            "ell_tv": (
                "ellipsoid의 real-space v 방향 translation입니다.\n"
                "polyhedron과의 상대 위치 차이를 만들며 interference fringe에 영향을 줍니다."
            ),
            "ell_weight": (
                "ellipsoid complex amplitude의 가중치입니다.\n"
                "A_total = weight1*A1 + weight2*A2에서 weight1에 해당합니다."
            ),
            "polyhedron_name": (
                "두 번째 object로 사용할 정다면체 종류입니다.\n"
                "tetra, cube, octa, dodeca, icosa 중 선택합니다."
            ),
            "poly_radius": (
                "polyhedron의 크기입니다.\n"
                "값이 커지면 real-space object가 커지고 q-space feature는 좁아집니다."
            ),
            "poly_axis_x": "polyhedron 3D 회전축의 x 성분입니다.",
            "poly_axis_y": "polyhedron 3D 회전축의 y 성분입니다.",
            "poly_axis_z": "polyhedron 3D 회전축의 z 성분입니다.",
            "poly_angle_deg": (
                "polyhedron을 지정한 회전축 기준으로 회전시키는 각도입니다.\n"
                "단위는 degree입니다."
            ),
            "poly_scale": (
                "polyhedron object의 실공간 크기 스케일입니다.\n"
                "ellipsoid scale과 다르게 주면 서로 다른 q-space feature가 겹칩니다."
            ),
            "poly_tu": (
                "polyhedron의 real-space u 방향 translation입니다.\n"
                "ellipsoid와의 상대 위치를 바꾸므로 coherent interference가 달라집니다."
            ),
            "poly_tv": (
                "polyhedron의 real-space v 방향 translation입니다.\n"
                "상대 phase factor를 바꿔 interference pattern을 변화시킵니다."
            ),
            "poly_weight": (
                "polyhedron complex amplitude의 가중치입니다.\n"
                "A_total = weight1*A1 + weight2*A2에서 weight2에 해당합니다."
            ),
        }
        return tips.get(name, "")

    def defaults(self):
        return {
            "ell_b": 1.0,
            "ell_angle_deg": 0.0,
            "ell_scale": 1000.0,
            "ell_tu": 0.0,
            "ell_tv": 0.0,
            "ell_weight": 1.0,

            "poly_radius": 1.0,
            "poly_axis_x": 1.0,
            "poly_axis_y": 1.0,
            "poly_axis_z": 1.0,
            "poly_angle_deg": 0.0,
            "poly_scale": 200.0,
            "poly_tu": 0.0,
            "poly_tv": 600.0,
            "poly_weight": 15.0,
        }

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            left,
            text="Merged Objects Parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        for name, value in self.defaults().items():
            self.vars[name] = tk.DoubleVar(value=value)

        # ---------------- Ellipsoid ----------------
        ell_frame = ttk.LabelFrame(left, text="Object 1: Ellipsoid", padding=6)
        ell_frame.pack(fill=tk.X, pady=5)

        self._add(ell_frame, "ell_b", 0.05, 1.0, 0.01)
        self._add(ell_frame, "ell_angle_deg", 0.0, 180.0, 1.0)
        self._add(ell_frame, "ell_scale", 50.0, 10000.0, 50.0)
        self._add(ell_frame, "ell_tu", -5000.0, 5000.0, 50.0)
        self._add(ell_frame, "ell_tv", -5000.0, 5000.0, 50.0)
        self._add(ell_frame, "ell_weight", 0.0, 50.0, 0.5)

        # ---------------- Polyhedron ----------------
        poly_frame = ttk.LabelFrame(left, text="Object 2: Polyhedron", padding=6)
        poly_frame.pack(fill=tk.X, pady=5)

        name_label = ttk.Label(poly_frame, text="polyhedron_name")
        name_label.pack(anchor="w")
        ToolTip(name_label, self.tooltip("polyhedron_name"))

        self.name_var = tk.StringVar(value="cube")
        combo = ttk.Combobox(
            poly_frame,
            textvariable=self.name_var,
            values=["tetra", "cube", "octa", "dodeca", "icosa"],
            state="readonly",
        )
        combo.pack(fill=tk.X, pady=(2, 4))
        combo.bind("<<ComboboxSelected>>", lambda _event: self.schedule())

        self._add(poly_frame, "poly_radius", 0.2, 3.0, 0.1)
        self._add(poly_frame, "poly_axis_x", -2.0, 2.0, 0.1)
        self._add(poly_frame, "poly_axis_y", -2.0, 2.0, 0.1)
        self._add(poly_frame, "poly_axis_z", -2.0, 2.0, 0.1)
        self._add(poly_frame, "poly_angle_deg", -180.0, 180.0, 1.0)
        self._add(poly_frame, "poly_scale", 50.0, 20000.0, 50.0)
        self._add(poly_frame, "poly_tu", -5000.0, 5000.0, 50.0)
        self._add(poly_frame, "poly_tv", -5000.0, 5000.0, 50.0)
        self._add(poly_frame, "poly_weight", 0.0, 50.0, 0.5)

        ttk.Button(left, text="Save current result", command=self.save).pack(fill=tk.X, pady=(10, 4))
        ttk.Button(left, text="Reset", command=self.reset).pack(fill=tk.X, pady=4)

        self.info = tk.Text(left, width=44, height=8)
        self.info.pack(fill=tk.X, pady=(10, 0))

        self.fig, self.ax = plt.subplots(1, 4, figsize=(15.5, 4.4))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add(self, parent, name, vmin, vmax, resolution):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)

        self.slider_meta[name] = {
            "vmin": float(vmin),
            "vmax": float(vmax),
            "step": float(resolution),
        }

        label = ttk.Label(f, text=name, width=14)
        label.pack(side=tk.LEFT)
        ToolTip(label, self.tooltip(name))

        left_btn = ttk.Button(
            f,
            text="◀",
            width=2,
            command=lambda key=name: self.step_value(key, -1),
        )
        left_btn.pack(side=tk.LEFT, padx=(0, 2))

        slider = ttk.Scale(
            f,
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            variable=self.vars[name],
            command=lambda _value, key=name: self.on_slider_change(key),
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.sliders[name] = slider

        right_btn = ttk.Button(
            f,
            text="▶",
            width=2,
            command=lambda key=name: self.step_value(key, 1),
        )
        right_btn.pack(side=tk.LEFT, padx=(2, 4))

        entry = ttk.Entry(f, width=9)
        entry.insert(0, f"{self.vars[name].get():.6g}")
        entry.pack(side=tk.RIGHT, padx=(4, 0))
        self.entries[name] = entry

        entry.bind("<Return>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<FocusOut>", lambda _event, key=name: self.on_entry_change(key))
        entry.bind("<Up>", lambda event, key=name: self.on_entry_step_key(event, key, +1))
        entry.bind("<Right>", lambda event, key=name: self.on_entry_step_key(event, key, +1))
        entry.bind("<Down>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<Left>", lambda event, key=name: self.on_entry_step_key(event, key, -1))
        entry.bind("<MouseWheel>", lambda event, key=name: self.on_entry_mousewheel(event, key))
        entry.bind("<Button-4>", lambda event, key=name: self.on_entry_step_key(event, key, +1))
        entry.bind("<Button-5>", lambda event, key=name: self.on_entry_step_key(event, key, -1))

    def clamp_to_range(self, name, value):
        meta = self.slider_meta.get(name)
        if not meta:
            return float(value)
        return min(max(float(value), meta["vmin"]), meta["vmax"])

    def round_to_step(self, name, value):
        meta = self.slider_meta.get(name)
        if not meta:
            return float(value)
        step = meta["step"]
        vmin = meta["vmin"]
        if step <= 0:
            return self.clamp_to_range(name, value)
        rounded = vmin + round((float(value) - vmin) / step) * step
        rounded = self.clamp_to_range(name, rounded)

        if step < 1:
            decimals = max(0, int(np.ceil(-np.log10(step))) + 2)
            rounded = round(rounded, decimals)
        else:
            rounded = round(rounded, 6)
        return rounded

    def set_value(self, name, value, schedule=True):
        value = self.round_to_step(name, value)
        self._slider_updating.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._slider_updating.discard(name)
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

    def on_entry_step_key(self, event, name, direction):
        self.step_value(name, direction)
        self.entries[name].selection_range(0, tk.END)
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = +1 if getattr(event, "delta", 0) > 0 else -1
        self.step_value(name, direction)
        self.entries[name].selection_range(0, tk.END)
        return "break"

    def on_slider_change(self, name):
        if name in self._slider_updating:
            return
        value = self.round_to_step(name, self.vars[name].get())
        self._slider_updating.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._slider_updating.discard(name)
        self.sync_entry(name)
        self.schedule()

    def on_entry_change(self, name):
        try:
            value = float(self.entries[name].get())
        except ValueError:
            messagebox.showerror("Input error", f"{name} must be numeric.")
            self.sync_entry(name)
            return

        self.set_value(name, value)

    def sync_entry(self, name):
        self.entries[name].delete(0, tk.END)
        self.entries[name].insert(0, f"{self.vars[name].get():.6g}")

    def schedule(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(400, self.update_plot)

    def params(self):
        p = {k: float(v.get()) for k, v in self.vars.items()}
        p["polyhedron_name"] = self.name_var.get()
        return p

    def reset(self):
        self.name_var.set("cube")
        for name, value in self.defaults().items():
            self.vars[name].set(value)
            self.sync_entry(name)
        self.update_plot()

    def draw_translated_density_sum(self, ax, result, p):
        ax.clear()
        ax.set_title("translated projected density sum")
        ax.set_aspect("equal", adjustable="box")

        extent1 = projection_extent(
            result["proj1"], result["du1"], result["dv1"],
            p["ell_scale"], p["ell_tu"], p["ell_tv"]
        )
        extent2 = projection_extent(
            result["proj2"], result["du2"], result["dv2"],
            p["poly_scale"], p["poly_tu"], p["poly_tv"]
        )

        img1 = p["ell_weight"] * result["proj1"]
        img2 = p["poly_weight"] * result["proj2"]

        ax.imshow(img1, extent=extent1, origin="lower", interpolation="nearest")
        ax.imshow(img2, extent=extent2, origin="lower", interpolation="nearest", alpha=0.65)

        xmin, xmax, ymin, ymax = combined_real_space_limits([extent1, extent2])
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
                projection_extent(result["proj1"], result["du1"], result["dv1"], p["ell_scale"], p["ell_tu"], p["ell_tv"]),
                projection_extent(result["proj2"], result["du2"], result["dv2"], p["poly_scale"], p["poly_tu"], p["poly_tv"]),
            ]

        if extents:
            ell_width = extents[0][1] - extents[0][0]
            ell_height = ell_width * p["ell_b"]
            poly_size = max(extents[1][1] - extents[1][0], extents[1][3] - extents[1][2])
        else:
            ell_width = 2.0 * p["ell_scale"]
            ell_height = 2.0 * p["ell_scale"] * p["ell_b"]
            poly_size = 2.0 * p["poly_scale"] * p["poly_radius"] * 0.2

        ell_patch = Ellipse((p["ell_tu"], p["ell_tv"]), width=ell_width, height=ell_height,
                            angle=p["ell_angle_deg"], fill=False, linewidth=1.5)
        ax.add_patch(ell_patch)
        ax.text(p["ell_tu"], p["ell_tv"], "Ell", ha="center", va="center")

        poly_patch = Ellipse((p["poly_tu"], p["poly_tv"]), width=poly_size, height=poly_size,
                             angle=0.0, fill=False, linewidth=1.5)
        ax.add_patch(poly_patch)
        ax.text(p["poly_tu"], p["poly_tv"], "Poly", ha="center", va="center")

        if extents:
            xmin, xmax, ymin, ymax = combined_real_space_limits(extents)
        else:
            centers = np.array([[p["ell_tu"], p["ell_tv"]], [p["poly_tu"], p["poly_tv"]]], dtype=float)
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

        images = [
            safe_log(I_total),
            I_masked,
        ]

        titles = [
            "log theoretical: |w1 A1 + w2 A2|²",
            "masked theoretical",
        ]

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

        self.info.delete("1.0", tk.END)
        du_rel = p["poly_tu"] - p["ell_tu"]
        dv_rel = p["poly_tv"] - p["ell_tv"]
        dr_rel = float(np.hypot(du_rel, dv_rel))

        self.info.insert(
            tk.END,
            f"polyhedron = {p['polyhedron_name']}\n"
            f"A_total = {p['ell_weight']}*A_ell + {p['poly_weight']}*A_poly\n\n"
            f"ell position = ({p['ell_tu']:.6g}, {p['ell_tv']:.6g})\n"
            f"poly position = ({p['poly_tu']:.6g}, {p['poly_tv']:.6g})\n"
            f"relative Δu, Δv = ({du_rel:.6g}, {dv_rel:.6g})\n"
            f"relative distance = {dr_rel:.6g}\n\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"merged_objects_sim_{timestamp}"

        dialog = SaveFolderDialog(self.root, default_name)
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
            self.save_image(d / "02_log_theoretical.png", safe_log(self.last_result["I_total"]),
                            "log theoretical")
            self.save_image(d / "03_masked_theoretical.png", self.last_result["I_masked"],
                            "masked theoretical")
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
            "Merged Objects SAXS simulation parameters",
            "=" * 48,
            "",
            "[Merge equation]",
            "A_total = ell_weight * A_ellipsoid + poly_weight * A_polyhedron",
            "I_total = |A_total|^2",
            "",
            "[Object 1: Ellipsoid]",
            f"ell_b = {p['ell_b']}",
            f"ell_angle_deg = {p['ell_angle_deg']}",
            f"ell_scale = {p['ell_scale']}",
            f"ell_translation_u = {p['ell_tu']}",
            f"ell_translation_v = {p['ell_tv']}",
            f"ell_weight = {p['ell_weight']}",
            "",
            "[Object 2: Polyhedron]",
            f"polyhedron_name = {p['polyhedron_name']}",
            f"poly_radius = {p['poly_radius']}",
            f"poly_rotation_axis = [{p['poly_axis_x']}, {p['poly_axis_y']}, {p['poly_axis_z']}]",
            f"poly_angle_deg = {p['poly_angle_deg']}",
            f"poly_scale = {p['poly_scale']}",
            f"poly_translation_u = {p['poly_tu']}",
            f"poly_translation_v = {p['poly_tv']}",
            f"poly_weight = {p['poly_weight']}",
            "",
            "[Relative placement]",
            f"delta_u = {p['poly_tu'] - p['ell_tu']}",
            f"delta_v = {p['poly_tv'] - p['ell_tv']}",
            f"delta_r = {float(np.hypot(p['poly_tu'] - p['ell_tu'], p['poly_tv'] - p['ell_tv']))}",
            "",
            "[Files]",
            f"poni_path = {PONI_PATH}",
            f"mask_path = {MASK_PATH}",
            "",
            "[q axes]",
            f"q_u_min = {float(q_u.min())}",
            f"q_u_max = {float(q_u.max())}",
            f"q_v_min = {float(q_v.min())}",
            f"q_v_max = {float(q_v.max())}",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def show_error(self, e):
        for ax in self.ax:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax[1].text(
            0.5,
            0.5,
            f"Error:\n{e}",
            ha="center",
            va="center",
            transform=self.ax[1].transAxes,
            wrap=True,
        )
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = MergedObjectsGUI(root)
    root.mainloop()