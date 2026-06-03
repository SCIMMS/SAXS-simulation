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

        ttk.Entry(parent_row, textvariable=self.parent_dir, width=56).grid(row=0, column=0, sticky="ew")
        ttk.Button(parent_row, text="Browse", command=self.browse).grid(row=0, column=1, padx=(6, 0))

        ttk.Label(outer, text="Result folder name").grid(row=2, column=0, sticky="w")
        folder_entry = ttk.Entry(outer, textvariable=self.folder_name, width=56)
        folder_entry.grid(row=3, column=0, sticky="ew", pady=(2, 8))

        ttk.Label(outer, text="Full save path").grid(row=4, column=0, sticky="w")
        ttk.Entry(outer, textvariable=self.full_path, width=56, state="readonly").grid(
            row=5, column=0, sticky="ew", pady=(2, 10)
        )

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
        self.top.protocol("WM_DELETE_WINDOW", self.cancel)

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

def draw_projected_density_physical(ax, img, meta_proj, title="projected density"):
    """Draw projected density in physical u/v coordinates, not raw pixel indices.

    This prevents apparent squashing when the projected bounding box is not square
    (for example, when a cone/tail changes the total object extent during rotation).
    """
    bounds = None
    if isinstance(meta_proj, dict):
        bounds = meta_proj.get("uv_bounds")
    ax.clear()
    if bounds is not None and len(bounds) == 4:
        ax.imshow(img, extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
                  origin="lower", interpolation="nearest")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
    else:
        ax.matshow(img)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])



def run_simulation(params):
    bvh = build_bvh(
        "coned-droplet",
        body_axes=[
            params["body_x"],
            params["body_y"],
            params["body_z"],
        ],
        cone_length=params["cone_length"],
        neck_radius=params["neck_radius"],
        tip_radius=params["tip_radius"],
        z_attach=params["z_attach"],
        alpha=params["alpha"],
        beta=params["beta"],
        rotations=[
            ([1, 0, 0], np.deg2rad(params["rot_x_deg"])),
            ([0, 1, 0], np.deg2rad(params["rot_y_deg"])),
            ([0, 0, 1], np.deg2rad(params["rot_z_deg"])),
        ],
        field_shape=(
            int(params["field_shape"]),
            int(params["field_shape"]),
            int(params["field_shape"]),
        ),
    )

    proj, du, dv, meta_proj = project_density_scaled(
        bvh,
        ray_vector=(0.0, 0.0, 1.0),
        roll_rad=0.0,
        image_shape=IMG_SHAPE,
        object_scale=params["object_scale"],
        out_dtype="float32",
    )

    A, q_u, q_v, meta = complex_amplitude_from_projected_density(
        projected_density=proj,
        du=du,
        dv=dv,
        scale_factor=params["scale_factor"],
        translation_u=0.0,
        translation_v=0.0,
        poni_path=PONI_PATH,
        detector_image_shape=IMG_SHAPE,
        q_unit="A^-1",
        center_mode="beam",
        row_origin="upper",
    )

    I_total = np.abs(A) ** 2
    return proj, du, dv, I_total, q_u, q_v, meta_proj, meta


class ConedDropletGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Coned Droplet SAXS Simulator")
        self.root.geometry("1550x780")

        self.update_job = None
        self.last_result = None

        self.mask = np.load(MASK_PATH)
        if self.mask.shape != IMG_SHAPE:
            self.mask = self.mask.reshape(IMG_SHAPE)

        self.vars = {}
        self.entries = {}
        self.sliders = {}
        self.control_specs = {}
        self._slider_updating = set()

        self._build_ui()
        self.update_plot()

    def get_tooltip_text(self, name):
        texts = {
            "body_x": "몸통 ellipsoid의 local x축 반축 길이입니다. 값이 커지면 x 방향으로 더 넓은 body가 됩니다.",
            "body_y": "몸통 ellipsoid의 local y축 반축 길이입니다. 기본값 2.0이면 y 방향으로 길쭉한 body가 됩니다.",
            "body_z": "몸통 ellipsoid의 local z축 반축 길이입니다. cone/tail은 local +z 방향으로 붙으므로 body_z는 tail 연결 위치와 두께에도 영향을 줍니다.",
            "cone_length": "tail의 local +z 방향 길이입니다. 값이 커지면 더 길게 늘어진 droplet 형태가 됩니다.",
            "neck_radius": "body와 tail이 만나는 연결부의 반지름입니다. 값이 작으면 neck이 가늘어지고, 값이 크면 body와 tail이 더 두껍게 연결됩니다.",
            "tip_radius": "tail 끝부분의 반지름입니다. 값이 작으면 sharp tip, 값이 크면 blunt tip에 가까워집니다.",
            "z_attach": "tail이 body의 local z축 어느 위치에서 시작되는지 정합니다. 값이 클수록 body 끝부분에 더 가깝게 tail이 붙습니다.",
            "alpha": "tail 반지름이 z 방향으로 줄어드는 초기 형태를 조절합니다. 값이 커지면 neck 근처 변화가 더 급격해질 수 있습니다.",
            "beta": "tail tapering의 곡률을 조절합니다. 값이 커지면 concave한 sidewall과 sharp한 tip 성향이 강해집니다.",
            "rot_x_deg": "3D object 전체를 x축 기준으로 회전시키는 각도입니다. projected density와 scattering pattern의 보이는 방향이 바뀝니다.",
            "rot_y_deg": "3D object 전체를 y축 기준으로 회전시키는 각도입니다. tail이 관측 방향에 대해 기울어지는 효과가 커질 수 있습니다.",
            "rot_z_deg": "3D object 전체를 z축 기준으로 회전시키는 각도입니다. detector plane 안에서의 in-plane rotation에 가깝습니다.",
            "object_scale": "projected density image 안에서 전체 coned droplet이 차지하는 크기 비율입니다. 1.0이면 기존과 같고, 값을 줄이면 object 주변에 margin이 생겨 box-filling artifact를 줄일 수 있습니다.",
            "scale_factor": "실공간 크기 스케일입니다. 값이 커지면 reciprocal-space 패턴이 중심부로 더 좁게 모이고, 값이 작으면 패턴이 더 넓게 퍼집니다.",
            "field_shape": "3D implicit field를 mesh로 변환할 때 쓰는 격자 해상도입니다. 값이 크면 표면이 정교해지지만 계산이 느려집니다.",
        }
        return texts.get(name, "")

    def defaults(self):
        return {
            "body_x": 1.0,
            "body_y": 2.0,
            "body_z": 1.0,
            "cone_length": 1.8,
            "neck_radius": 0.34,
            "tip_radius": 0.025,
            "z_attach": 0.72,
            "alpha": 1.0,
            "beta": 2.5,
            "rot_x_deg": -18.0,
            "rot_y_deg": 22.0,
            "rot_z_deg": 12.0,
            "object_scale": 0.7,
            "scale_factor": 5000.0,
            "field_shape": 120.0,
        }

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(main, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        plot_area = ttk.Frame(main, padding=8)
        plot_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            controls,
            text="Coned droplet parameters",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        for name, value in self.defaults().items():
            self.vars[name] = tk.DoubleVar(value=value)

        body_frame = ttk.LabelFrame(controls, text="Ellipsoidal body", padding=6)
        body_frame.pack(fill=tk.X, pady=4)
        self._add_slider_entry(body_frame, "body_x", 0.2, 3.0, 0.1)
        self._add_slider_entry(body_frame, "body_y", 0.2, 3.0, 0.1)
        self._add_slider_entry(body_frame, "body_z", 0.2, 3.0, 0.1)

        cone_frame = ttk.LabelFrame(controls, text="Cone / tail", padding=6)
        cone_frame.pack(fill=tk.X, pady=4)
        self._add_slider_entry(cone_frame, "cone_length", 0.2, 4.0, 0.1)
        self._add_slider_entry(cone_frame, "neck_radius", 0.05, 1.0, 0.01)
        self._add_slider_entry(cone_frame, "tip_radius", 0.005, 0.2, 0.005)
        self._add_slider_entry(cone_frame, "z_attach", 0.0, 1.0, 0.01)
        self._add_slider_entry(cone_frame, "alpha", 0.5, 3.0, 0.1)
        self._add_slider_entry(cone_frame, "beta", 1.0, 5.0, 0.1)

        rot_frame = ttk.LabelFrame(controls, text="3D rotation", padding=6)
        rot_frame.pack(fill=tk.X, pady=4)
        self._add_slider_entry(rot_frame, "rot_x_deg", -90.0, 90.0, 1.0)
        self._add_slider_entry(rot_frame, "rot_y_deg", -90.0, 90.0, 1.0)
        self._add_slider_entry(rot_frame, "rot_z_deg", -180.0, 180.0, 1.0)

        calc_frame = ttk.LabelFrame(controls, text="Scattering / mesh", padding=6)
        calc_frame.pack(fill=tk.X, pady=4)
        self._add_slider_entry(calc_frame, "object_scale", 0.2, 1.0, 0.01)
        self._add_slider_entry(calc_frame, "scale_factor", 100.0, 20000.0, 100.0)
        self._add_slider_entry(calc_frame, "field_shape", 50.0, 180.0, 10.0)

        ttk.Button(
            controls,
            text="Save current result",
            command=self.save_current_result,
        ).pack(fill=tk.X, pady=(10, 4))

        ttk.Button(
            controls,
            text="Reset",
            command=self.reset,
        ).pack(fill=tk.X, pady=4)

        self.info = tk.Text(controls, width=46, height=10)
        self.info.pack(fill=tk.X, pady=(12, 0))

        self.fig, self.axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _add_slider_entry(self, parent, name, vmin, vmax, resolution):
        self.control_specs[name] = {
            "min": float(vmin),
            "max": float(vmax),
            "step": float(resolution),
        }

        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        frame.columnconfigure(2, weight=1)

        label_widget = ttk.Label(frame, text=name, width=13)
        label_widget.grid(row=0, column=0, sticky="w")
        ToolTip(label_widget, self.get_tooltip_text(name))

        left_btn = ttk.Button(
            frame,
            text="◀",
            width=3,
            command=lambda key=name: self.step_value(key, -1),
        )
        left_btn.grid(row=0, column=1, padx=(2, 2))

        slider = ttk.Scale(
            frame,
            variable=self.vars[name],
            from_=vmin,
            to=vmax,
            orient=tk.HORIZONTAL,
            command=lambda _value, key=name: self.on_slider_change(key),
        )
        slider.grid(row=0, column=2, sticky="ew")
        self.sliders[name] = slider

        right_btn = ttk.Button(
            frame,
            text="▶",
            width=3,
            command=lambda key=name: self.step_value(key, 1),
        )
        right_btn.grid(row=0, column=3, padx=(2, 4))

        entry = ttk.Entry(frame, width=10)
        entry.grid(row=0, column=4, sticky="e")
        entry.insert(0, f"{self.vars[name].get():.6g}")
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

    def set_parameter_value(self, name, value, snap_to_step=True, schedule_update=True, select_entry=False):
        value = self.normalize_value(name, value, snap_to_step=snap_to_step)

        self._slider_updating.add(name)
        try:
            self.vars[name].set(value)
        finally:
            self._slider_updating.discard(name)

        self._sync_entry(name)
        if select_entry and name in self.entries:
            self.entries[name].selection_range(0, tk.END)
            self.entries[name].icursor(tk.END)
        if schedule_update:
            self.schedule_update()

    def get_entry_or_var_value(self, name):
        try:
            return float(self.entries[name].get())
        except Exception:
            return float(self.vars[name].get())

    def step_value(self, name, direction, select_entry=False):
        spec = self.control_specs[name]
        current = self.get_entry_or_var_value(name)
        self.set_parameter_value(
            name,
            current + float(direction) * spec["step"],
            snap_to_step=True,
            schedule_update=True,
            select_entry=select_entry,
        )

    def on_entry_step_key(self, event, name, direction):
        self.step_value(name, direction, select_entry=True)
        return "break"

    def on_entry_mousewheel(self, event, name):
        direction = 1 if event.delta > 0 else -1
        self.step_value(name, direction, select_entry=True)
        return "break"

    def on_slider_change(self, name):
        if name in self._slider_updating:
            return
        value = self.normalize_value(name, self.vars[name].get(), snap_to_step=True)
        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True)

    def on_entry_change(self, name):
        try:
            value = float(self.entries[name].get())
        except ValueError:
            messagebox.showerror("Input error", f"{name} must be numeric.")
            self._sync_entry(name)
            return
        self.set_parameter_value(name, value, snap_to_step=True, schedule_update=True)

    def _sync_entry(self, name):
        entry = self.entries[name]
        entry.delete(0, tk.END)
        entry.insert(0, f"{self.vars[name].get():.6g}")

    def schedule_update(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(700, self.update_plot)

    def get_params(self):
        params = {name: float(var.get()) for name, var in self.vars.items()}
        params["field_shape"] = int(round(params["field_shape"]))
        params["field_shape"] = max(30, params["field_shape"])
        return params

    def reset(self):
        for name, value in self.defaults().items():
            self.set_parameter_value(name, value, snap_to_step=True, schedule_update=False)
        self.update_plot()

    def update_plot(self):
        self.update_job = None

        try:
            params = self.get_params()
            proj, du, dv, I_total, q_u, q_v, meta_proj, meta = run_simulation(params)
        except Exception as e:
            self._show_error(e)
            return

        I_masked = I_total * self.mask

        self.last_result = {
            "parameters": params,
            "proj": proj,
            "du": du,
            "dv": dv,
            "I_total": I_total,
            "I_masked": I_masked,
            "q_u": q_u,
            "q_v": q_v,
            "meta_proj": meta_proj,
            "meta": meta,
        }

        draw_projected_density_physical(self.axes[0], proj, meta_proj, "projected density")

        images = [safe_log(I_total), I_masked]
        titles = ["log theoretical", "masked theoretical"]
        for ax, img, title in zip(self.axes[1:], images, titles):
            ax.clear()
            ax.matshow(img)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._update_info(du, dv, q_u, q_v)

    def _update_info(self, du, dv, q_u, q_v):
        self.info.delete("1.0", tk.END)
        self.info.insert(
            tk.END,
            f"body_axes = [{self.vars['body_x'].get():.6g}, {self.vars['body_y'].get():.6g}, {self.vars['body_z'].get():.6g}]\n"
            f"cone_length = {self.vars['cone_length'].get():.6g}\n"
            f"object_scale = {self.vars['object_scale'].get():.6g}\n"
            f"scale_factor = {self.vars['scale_factor'].get():.6g}\n"
            f"field_shape = {int(round(self.vars['field_shape'].get()))}\n\n"
            f"du = {du:.6g}\n"
            f"dv = {dv:.6g}\n"
            f"q_u = [{q_u.min():.6g}, {q_u.max():.6g}]\n"
            f"q_v = [{q_v.min():.6g}, {q_v.max():.6g}]\n"
        )

    def save_current_result(self):
        if self.last_result is None:
            messagebox.showerror("Save error", "No simulation result to save.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dialog = SaveFolderDialog(self.root, f"coned_droplet_sim_{timestamp}")
        save_dir = dialog.result
        if save_dir is None:
            return

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Save error", f"Could not create save folder:\n{save_dir}\n\n{e}")
            return

        try:
            proj = self.last_result["proj"]
            I_total = self.last_result["I_total"]
            I_masked = self.last_result["I_masked"]
            self._save_image(save_dir / "01_projected_density.png", proj, title="projected density")
            self._save_image(save_dir / "02_log_theoretical.png", safe_log(I_total), title="log theoretical")
            self._save_image(save_dir / "03_masked_theoretical.png", I_masked, title="masked theoretical")
            self._save_parameters_txt(save_dir / "parameters.txt")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save result:\n{save_dir}\n\n{e}")
            return

        messagebox.showinfo("Saved", f"Saved result to:\n{save_dir}")

    def _save_image(self, path, img, title=""):
        fig, ax = plt.subplots(figsize=(6, 6))
        if title == "projected density":
            draw_projected_density_physical(ax, img, self.last_result.get("meta_proj"), title)
        else:
            ax.matshow(img)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _save_parameters_txt(self, path):
        result = self.last_result
        p = result["parameters"]

        lines = [
            "Coned droplet SAXS simulation parameters",
            "=" * 45,
            "",
            "[Shape]",
            "shape = coned-droplet",
            f"body_axes = [{p['body_x']}, {p['body_y']}, {p['body_z']}]",
            f"cone_length = {p['cone_length']}",
            f"neck_radius = {p['neck_radius']}",
            f"tip_radius = {p['tip_radius']}",
            f"z_attach = {p['z_attach']}",
            f"alpha = {p['alpha']}",
            f"beta = {p['beta']}",
            "",
            "[Rotation]",
            f"rot_x_deg = {p['rot_x_deg']}",
            f"rot_y_deg = {p['rot_y_deg']}",
            f"rot_z_deg = {p['rot_z_deg']}",
            "",
            "[Scattering / mesh]",
            f"object_scale = {p.get('object_scale', 1.0)}",
            f"scale_factor = {p['scale_factor']}",
            f"field_shape = {int(round(p['field_shape']))}",
            f"translation_u = 0.0",
            f"translation_v = 0.0",
            "",
            "[Files]",
            f"poni_path = {PONI_PATH}",
            f"mask_path = {MASK_PATH}",
            "",
            "[Grid]",
            f"image_shape = {IMG_SHAPE}",
            f"du = {result['du']}",
            f"dv = {result['dv']}",
            "",
            "[q axes]",
            f"q_u_min = {float(result['q_u'].min())}",
            f"q_u_max = {float(result['q_u'].max())}",
            f"q_v_min = {float(result['q_v'].min())}",
            f"q_v_max = {float(result['q_v'].max())}",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _show_error(self, e):
        for ax in self.axes:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.axes[1].text(
            0.5,
            0.5,
            f"Error:\n{e}",
            ha="center",
            va="center",
            transform=self.axes[1].transAxes,
            wrap=True,
        )
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConedDropletGUI(root)
    root.mainloop()
