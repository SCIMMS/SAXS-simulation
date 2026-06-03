from datetime import datetime
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from saxs_module import (
    build_bvh,
    project_density,
    project_density_scaled,
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


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=430):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, width=width, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except tk.TclError:
            pass


class SaveFolderDialog(tk.Toplevel):
    def __init__(self, parent, default_name):
        super().__init__(parent)
        self.title("Save result")
        self.resizable(False, False)
        self.result = None

        self.parent_dir = tk.StringVar(value="")
        self.folder_name = tk.StringVar(value=default_name)
        self.full_path = tk.StringVar(value="")

        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="Parent directory").grid(row=0, column=0, sticky="w")
        parent_row = ttk.Frame(body)
        parent_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 8))
        parent_row.columnconfigure(0, weight=1)
        ttk.Entry(parent_row, textvariable=self.parent_dir, width=52).grid(row=0, column=0, sticky="ew")
        ttk.Button(parent_row, text="Browse...", command=self.browse).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(body, text="Result folder name").grid(row=2, column=0, sticky="w")
        name_entry = ttk.Entry(body, textvariable=self.folder_name, width=52)
        name_entry.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        ttk.Label(body, text="Full save path").grid(row=4, column=0, sticky="w")
        ttk.Entry(body, textvariable=self.full_path, width=60, state="readonly").grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(2, 10)
        )

        buttons = ttk.Frame(body)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Save", command=self.accept).pack(side=tk.RIGHT)

        self.parent_dir.trace_add("write", lambda *_: self.update_full_path())
        self.folder_name.trace_add("write", lambda *_: self.update_full_path())
        self.update_full_path()

        self.transient(parent)
        self.grab_set()
        name_entry.focus_set()
        self.bind("<Return>", lambda _event: self.accept())
        self.bind("<Escape>", lambda _event: self.cancel())
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.wait_window(self)

    def browse(self):
        selected = filedialog.askdirectory(title="Choose parent directory", parent=self)
        if selected:
            self.parent_dir.set(selected)

    def update_full_path(self):
        parent = self.parent_dir.get().strip()
        name = self.folder_name.get().strip()
        self.full_path.set(str(Path(parent) / name) if parent and name else "")

    def accept(self):
        parent = self.parent_dir.get().strip()
        name = self.folder_name.get().strip()
        if not parent:
            messagebox.showerror("Save error", "Please choose a parent directory.", parent=self)
            return
        if not name:
            messagebox.showerror("Save error", "Please enter a result folder name.", parent=self)
            return

        invalid = set('<>:"/\\|?*')
        if any(ch in invalid for ch in name):
            messagebox.showerror(
                "Save error",
                'Folder name contains an invalid character: < > : " / \\ | ? *',
                parent=self,
            )
            return

        target = Path(parent) / name
        if target.exists():
            ok = messagebox.askyesno(
                "Folder exists",
                f"This folder already exists:\n{target}\n\nSave into this folder anyway?",
                parent=self,
            )
            if not ok:
                return

        self.result = target
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()


def safe_log(x):
    return np.log(np.asarray(x, dtype=np.float64) + EPS)


def run_simulation(params):
    centers = [
        [params[f"c{i}_x"], params[f"c{i}_y"], params[f"c{i}_z"]]
        for i in range(3)
    ]

    sigmas = [
        [params[f"s{i}_x"], params[f"s{i}_y"], params[f"s{i}_z"]]
        for i in range(3)
    ]

    weights = [params[f"w{i}"] for i in range(3)]
    axis = [params["axis_x"], params["axis_y"], params["axis_z"]]

    if np.linalg.norm(axis) < 1e-12:
        raise ValueError("rotation axis cannot be zero vector.")

    bvh = build_bvh(
        "blob",
        kernel_centers=centers,
        kernel_sigmas=sigmas,
        kernel_weights=weights,
        threshold=params["threshold"],
        rotations=[(axis, np.deg2rad(params["angle_deg"]))],
        field_shape=(
            int(params["field_shape"]),
            int(params["field_shape"]),
            int(params["field_shape"]),
        ),
    )

    proj, du, dv, meta_proj = project_density_scaled(
        bvh,
        ray_vector=(0, 0, 1),
        roll_rad=0,
        image_shape=IMG_SHAPE,
        object_scale=params["object_scale"],
        out_dtype="float32",
    )

    A, q_u, q_v, meta = complex_amplitude_from_projected_density(
        projected_density=proj,
        du=du,
        dv=dv,
        scale_factor=params["scale_factor"],
        translation_u=0,
        translation_v=0,
        poni_path=PONI_PATH,
        detector_image_shape=IMG_SHAPE,
        q_unit="A^-1",
        center_mode="beam",
        row_origin="upper",
    )

    I_total = np.abs(A) ** 2
    return proj, du, dv, I_total, q_u, q_v, meta_proj, meta


class BlobGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Random Blob SAXS Simulator")
        self.root.geometry("1600x820")

        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.vars = {}
        self.entries = {}
        self.control_specs = {}
        self.sliders = {}
        self._slider_updating = set()

        self._build_ui()
        self.update_plot()

    def tooltip(self, name):
        if name.startswith("c"):
            idx = name[1]
            axis = name[-1]
            return (
                f"Kernel {idx}의 중심 위치 {axis} 좌표입니다.\n"
                "Gaussian blob을 구성하는 한 덩어리의 중심을 이동합니다.\n"
                "값을 바꾸면 전체 random object의 비대칭성, elongation, projected density가 달라집니다."
            )

        if name.startswith("s"):
            idx = name[1]
            axis = name[-1]
            return (
                f"Kernel {idx}의 {axis} 방향 폭(sigma)입니다.\n"
                "값이 커지면 해당 방향으로 더 넓고 퍼진 덩어리가 됩니다.\n"
                "값이 작아지면 더 국소적이고 뾰족한 feature가 생깁니다."
            )

        if name.startswith("w"):
            idx = name[1]
            return (
                f"Kernel {idx}의 weight입니다.\n"
                "값이 커지면 이 Gaussian 덩어리가 전체 blob shape에 더 강하게 기여합니다.\n"
                "높은 weight는 해당 kernel 주변 부피를 키우는 효과가 있습니다."
            )

        tips = {
            "threshold": (
                "Gaussian mixture field에서 표면을 자르는 cutoff 값입니다.\n"
                "field 값이 threshold보다 큰 영역을 object 내부로 간주합니다.\n"
                "낮추면 부피가 커지고 kernel들이 더 쉽게 연결됩니다.\n"
                "높이면 object가 작아지거나 여러 조각으로 분리될 수 있습니다."
            ),
            "axis_x": (
                "3D 회전축의 x 성분입니다.\n"
                "axis_x, axis_y, axis_z가 함께 하나의 회전축 벡터를 이룹니다."
            ),
            "axis_y": (
                "3D 회전축의 y 성분입니다.\n"
                "회전축 방향을 바꾸면 같은 blob이라도 projection 방향에서 다르게 보입니다."
            ),
            "axis_z": (
                "3D 회전축의 z 성분입니다.\n"
                "axis=(0,0,1)에 가까우면 detector plane 안에서의 in-plane rotation에 가깝습니다."
            ),
            "angle_deg": (
                "지정한 회전축(axis)을 기준으로 blob object 전체를 회전시키는 각도입니다.\n"
                "단위는 degree입니다."
            ),
            "scale_factor": (
                "실공간 object의 크기 스케일입니다.\n"
                "값이 커지면 reciprocal-space scattering pattern이 중심부로 더 좁게 모이고,\n"
                "값이 작아지면 패턴이 더 넓게 퍼집니다."
            ),
            "field_shape": (
                "3D Gaussian field를 mesh로 변환할 때 쓰는 격자 해상도입니다.\n"
                "값이 크면 표면이 더 정교해지지만 계산 시간이 길어집니다.\n"
                "빠른 탐색은 70~100, 정밀 저장은 120 이상 정도가 적당합니다."
            ),
        }
        return tips.get(name, "")

    def default_values(self):
        values = {}

        centers = [
            [0.0, 0.0, 0.0],
            [0.3, 0.1, 0.2],
            [-0.2, 0.2, -0.1],
        ]

        sigmas = [
            [0.35, 0.30, 0.25],
            [0.20, 0.18, 0.70],
            [0.22, 0.50, 0.20],
        ]

        weights = [1.0, 0.8, 2.7]

        for i in range(3):
            values[f"c{i}_x"] = centers[i][0]
            values[f"c{i}_y"] = centers[i][1]
            values[f"c{i}_z"] = centers[i][2]
            values[f"s{i}_x"] = sigmas[i][0]
            values[f"s{i}_y"] = sigmas[i][1]
            values[f"s{i}_z"] = sigmas[i][2]
            values[f"w{i}"] = weights[i]

        values.update(
            {
                "threshold": 0.55,
                "axis_x": 0.0,
                "axis_y": 0.0,
                "axis_z": 1.0,
                "angle_deg": np.rad2deg(0.5),
                "object_scale": 0.7,
                "scale_factor": 5000.0,
                "field_shape": 100.0,
            }
        )
        return values

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        scroll = ScrollableFrame(main, width=430)
        scroll.pack(side=tk.LEFT, fill=tk.Y)

        controls = scroll.inner

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="Random Blob parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(4, 10), padx=8)

        defaults = self.default_values()

        for name, value in defaults.items():
            self.vars[name] = tk.DoubleVar(value=value)

        for i in range(3):
            k_frame = ttk.LabelFrame(controls, text=f"Kernel {i}", padding=6)
            k_frame.pack(fill=tk.X, padx=8, pady=4)

            self._add(k_frame, f"c{i}_x", -1.5, 1.5, 0.01)
            self._add(k_frame, f"c{i}_y", -1.5, 1.5, 0.01)
            self._add(k_frame, f"c{i}_z", -1.5, 1.5, 0.01)

            self._add(k_frame, f"s{i}_x", 0.05, 1.2, 0.01)
            self._add(k_frame, f"s{i}_y", 0.05, 1.2, 0.01)
            self._add(k_frame, f"s{i}_z", 0.05, 1.2, 0.01)

            self._add(k_frame, f"w{i}", 0.1, 4.0, 0.05)

        shape_frame = ttk.LabelFrame(controls, text="Shape threshold", padding=6)
        shape_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add(shape_frame, "threshold", 0.1, 1.5, 0.01)

        rot_frame = ttk.LabelFrame(controls, text="3D rotation", padding=6)
        rot_frame.pack(fill=tk.X, padx=8, pady=4)

        self._add(rot_frame, "axis_x", -1.0, 1.0, 0.1)
        self._add(rot_frame, "axis_y", -1.0, 1.0, 0.1)
        self._add(rot_frame, "axis_z", -1.0, 1.0, 0.1)
        self._add(rot_frame, "angle_deg", -180.0, 180.0, 1.0)

        calc_frame = ttk.LabelFrame(controls, text="Scattering / mesh", padding=6)
        calc_frame.pack(fill=tk.X, padx=8, pady=4)

        self._add(calc_frame, "object_scale", 0.2, 1.0, 0.01)
        self._add(calc_frame, "scale_factor", 100.0, 20000.0, 100.0)
        self._add(calc_frame, "field_shape", 50.0, 180.0, 10.0)

        ttk.Button(
            controls,
            text="Save current result",
            command=self.save,
        ).pack(fill=tk.X, padx=8, pady=(10, 4))

        ttk.Button(
            controls,
            text="Reset",
            command=self.reset,
        ).pack(fill=tk.X, padx=8, pady=4)

        self.info = tk.Text(controls, width=44, height=10)
        self.info.pack(fill=tk.X, padx=8, pady=(10, 10))

        self.fig, self.ax = plt.subplots(1, 3, figsize=(12.5, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add(self, parent, name, vmin, vmax, resolution):
        """Add one parameter control row.

        Layout:
            label | left step button | slider | right step button | numeric entry

        Keyboard/mouse interactions on the numeric entry:
            Up / Right   : +1 step
            Down / Left  : -1 step
            Mouse wheel  : +/-1 step
        """
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=2)
        f.columnconfigure(2, weight=1)

        self.control_specs[name] = {
            "min": float(vmin),
            "max": float(vmax),
            "step": float(resolution),
        }

        label = ttk.Label(f, text=name, width=10)
        label.grid(row=0, column=0, sticky="w")
        ToolTip(label, self.tooltip(name))

        left_button = ttk.Button(
            f,
            text="◀",
            width=3,
            command=lambda key=name: self.step_value(key, -1),
        )
        left_button.grid(row=0, column=1, padx=(2, 2))

        slider = ttk.Scale(
            f,
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            variable=self.vars[name],
            command=lambda _value, key=name: self.on_slider_change(key),
        )
        slider.grid(row=0, column=2, sticky="ew")
        self.sliders[name] = slider

        right_button = ttk.Button(
            f,
            text="▶",
            width=3,
            command=lambda key=name: self.step_value(key, 1),
        )
        right_button.grid(row=0, column=3, padx=(2, 4))

        entry = ttk.Entry(f, width=9)
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

    def normalize_value(self, name, value, snap_to_step=True):
        spec = self.control_specs[name]
        vmin = spec["min"]
        vmax = spec["max"]
        step = spec["step"]

        value = min(max(float(value), vmin), vmax)

        if snap_to_step and step > 0:
            n_step = round((value - vmin) / step)
            value = vmin + n_step * step
            value = min(max(value, vmin), vmax)

        if step > 0:
            decimals = max(0, int(np.ceil(-np.log10(step))) + 2) if step < 1 else 6
            value = round(value, decimals)

        return value

    def current_entry_or_var_value(self, name):
        """Use the active entry text as the starting value when possible."""
        try:
            return float(self.entries[name].get())
        except Exception:
            return float(self.vars[name].get())

    def set_parameter_value(self, name, value, snap_to_step=True, schedule_update=True, select_entry=False):
        value = self.normalize_value(name, value, snap_to_step=snap_to_step)

        self._slider_updating.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._slider_updating.discard(name)

        self.sync_entry(name)
        if select_entry:
            entry = self.entries[name]
            entry.focus_set()
            entry.selection_range(0, tk.END)
            entry.icursor(tk.END)

        if schedule_update:
            self.schedule()

    def step_value(self, name, direction):
        spec = self.control_specs[name]
        value = float(self.vars[name].get()) + float(direction) * spec["step"]
        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True)

    def on_entry_step_key(self, event, name, direction):
        spec = self.control_specs[name]
        value = self.current_entry_or_var_value(name) + float(direction) * spec["step"]
        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True, select_entry=True)
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = 1 if event.delta > 0 else -1
        return self.on_entry_step_key(event, name, direction)

    def on_slider_change(self, name):
        if name in self._slider_updating:
            return

        value = self.normalize_value(name, self.vars[name].get(), snap_to_step=True)
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

        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True)

    def sync_entry(self, name):
        self.entries[name].delete(0, tk.END)
        self.entries[name].insert(0, f"{self.vars[name].get():.6g}")

    def schedule(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(600, self.update_plot)

    def params(self):
        p = {k: float(v.get()) for k, v in self.vars.items()}
        p["field_shape"] = int(round(p["field_shape"]))
        p["field_shape"] = max(30, p["field_shape"])
        return p

    def reset(self):
        defaults = self.default_values()
        for name, value in defaults.items():
            self.vars[name].set(value)
            self.sync_entry(name)

        self.update_plot()

    def update_plot(self):
        self.update_job = None

        try:
            p = self.params()
            proj, du, dv, I, q_u, q_v, meta_proj, meta = run_simulation(p)
        except Exception as e:
            self.show_error(e)
            return

        masked = I * self.mask

        self.last_result = {
            "parameters": p,
            "proj": proj,
            "I": I,
            "masked": masked,
            "du": du,
            "dv": dv,
            "q_u": q_u,
            "q_v": q_v,
            "meta_proj": meta_proj,
            "meta": meta,
        }

        images = [proj, safe_log(I), masked]
        titles = ["projected density", "log theoretical", "masked theoretical"]

        for ax, im, title in zip(self.ax, images, titles):
            ax.clear()
            ax.matshow(im)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"threshold = {p['threshold']:.6g}\n"
            f"object_scale = {p.get('object_scale', 1.0):.6g}\n"
            f"scale_factor = {p['scale_factor']:.6g}\n"
            f"field_shape = {p['field_shape']}\n\n"
            f"du = {du:.6g}\n"
            f"dv = {dv:.6g}\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dialog = SaveFolderDialog(self.root, f"blob_sim_{timestamp}")
        d = dialog.result
        if d is None:
            return

        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save error", f"Could not create save folder:\n{d}\n\n{e}")
            return

        proj = self.last_result["proj"]
        I = self.last_result["I"]
        masked = self.last_result["masked"]

        try:
            self.save_image(d / "01_projected_density.png", proj, "projected density")
            self.save_image(d / "02_log_theoretical.png", safe_log(I), "log theoretical")
            self.save_image(d / "03_masked_theoretical.png", masked, "masked theoretical")
            self.save_params(d / "parameters.txt")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save result files:\n{d}\n\n{e}")
            return

        messagebox.showinfo("Saved", f"Saved result to:\n{d}")

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

        lines = [
            "Random Blob SAXS simulation parameters",
            "=" * 42,
            "",
            "[Shape model]",
            "shape = blob",
            "model = Gaussian mixture isosurface",
            f"threshold = {p['threshold']}",
            "",
        ]

        for i in range(3):
            lines.extend(
                [
                    f"[Kernel {i}]",
                    f"center = [{p[f'c{i}_x']}, {p[f'c{i}_y']}, {p[f'c{i}_z']}]",
                    f"sigma = [{p[f's{i}_x']}, {p[f's{i}_y']}, {p[f's{i}_z']}]",
                    f"weight = {p[f'w{i}']}",
                    "",
                ]
            )

        lines.extend(
            [
                "[Rotation]",
                f"axis = [{p['axis_x']}, {p['axis_y']}, {p['axis_z']}]",
                f"angle_deg = {p['angle_deg']}",
                f"angle_rad = {np.deg2rad(p['angle_deg'])}",
                "",
                "[Scattering / mesh]",
                f"object_scale = {p.get('object_scale', 1.0)}",
                f"scale_factor = {p['scale_factor']}",
                f"field_shape = {p['field_shape']}",
                "",
                "[Files]",
                f"poni_path = {PONI_PATH}",
                f"mask_path = {MASK_PATH}",
                "",
                "[Grid]",
                f"image_shape = {IMG_SHAPE}",
                f"du = {self.last_result['du']}",
                f"dv = {self.last_result['dv']}",
                "",
                "[q axes]",
                f"q_u_min = {float(self.last_result['q_u'].min())}",
                f"q_u_max = {float(self.last_result['q_u'].max())}",
                f"q_v_min = {float(self.last_result['q_v'].min())}",
                f"q_v_max = {float(self.last_result['q_v'].max())}",
            ]
        )

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
    app = BlobGUI(root)
    root.mainloop()