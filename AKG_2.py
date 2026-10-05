import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ------------------------------------------------------------------
# Допустимые диапазоны параметров (по тексту задания)
# ------------------------------------------------------------------
PARAM_LIMITS = {
    "W":    (100.0, 10000.0),
    "H":    (100.0, 10000.0),
    "Wres": (200, 800),
    "xL":   (-10000.0, 10000.0),
    "yL":   (-10000.0, 10000.0),
    "zL":   (100.0, 10000.0),
    "xC":   (-10000.0, 10000.0),
    "yC":   (-10000.0, 10000.0),
    "zC":   (100.0, 10000.0),
    "I0":   (0.01, 10000.0),
}

LAMP_AXIS = np.array([0.0, 0.0, -1.0])  # лампа светит вертикально вниз (как в ЛР_1)
EPS = 1e-9


class SphereBrightnessModel:
    """Расчет распределения яркости на сфере, видимой с точки наблюдения."""

    def __init__(self, W, H, Wres, zO, xC, yC, zC, R,
                 lights, kd, ks, ke):
        """
        lights: список словарей {"pos": (xL, yL, zL), "I0": I0}
        """
        self.W, self.H, self.Wres = W, H, Wres
        px = W / Wres
        self.Hres = max(1, round(H / px))
        self.px = px

        self.eye = np.array([0.0, 0.0, zO])
        self.C = np.array([xC, yC, zC])
        self.R = R
        self.lights = lights
        self.kd, self.ks, self.ke = kd, ks, ke

        xs = (np.arange(self.Wres) + 0.5) * px - W / 2.0
        ys = (np.arange(self.Hres) + 0.5) * px - H / 2.0
        X, Y = np.meshgrid(xs, ys)  # shape (Hres, Wres)
        self.X, self.Y = X, Y
        self.S = np.stack([X, Y, np.zeros_like(X)], axis=-1)  # (Hres, Wres, 3)

    # --------------------------------------------------------- геометрия --
    def _intersect_sphere(self):
        """Пересечение лучей (из eye через каждый пиксель экрана S) со сферой.
        Возвращает (hit_mask, P) - точку пересечения, ближайшую к наблюдателю."""
        d = self.S - self.eye                      # (Hres,Wres,3) направление луча (не нормировано)
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)

        oc = self.eye - self.C                      # (3,)
        b = 2.0 * np.sum(d * oc, axis=-1)            # (Hres,Wres)
        c = np.dot(oc, oc) - self.R ** 2
        disc = b ** 2 - 4.0 * c

        hit_mask = disc >= 0
        sqrt_disc = np.sqrt(np.clip(disc, 0, None))
        t_near = (-b - sqrt_disc) / 2.0
        hit_mask &= t_near > EPS

        P = self.eye + t_near[..., None] * d
        return hit_mask, P, d

    # --------------------------------------------------------- яркость --
    def _shade_point(self, P, N, d):
        """Яркость L(P, v) в точках P (произвольной формы (..., 3)) с нормалью N."""
        v = -d  # направление P -> наблюдатель (луч шел от eye к P, поэтому наблюдатель = -d)
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)

        L_total = np.zeros(P.shape[:-1])
        for light in self.lights:
            Pl = np.array(light["pos"])
            I0 = light["I0"]

            s = P - Pl                                   # от источника к точке
            r = np.linalg.norm(s, axis=-1)
            r_safe = np.where(r > EPS, r, EPS)

            cos_theta = np.clip(np.sum(s * LAMP_AXIS, axis=-1) / r_safe, 0.0, None)
            I_s = I0 * cos_theta

            l = -s / r_safe[..., None]                   # направление точка -> источник
            cos_sigma = np.clip(np.sum(N * l, axis=-1), 0.0, None)

            E = I_s * cos_sigma / (r_safe ** 2)

            h = v + l
            h_norm = np.linalg.norm(h, axis=-1, keepdims=True)
            h_norm = np.where(h_norm > EPS, h_norm, EPS)
            h = h / h_norm
            spec_base = np.clip(np.sum(h * N, axis=-1), 0.0, None)
            f = self.kd + self.ks * np.power(spec_base, self.ke)

            L_total = L_total + E * f

        return L_total / np.pi

    def compute(self):
        hit_mask, P, d = self._intersect_sphere()
        N = np.zeros_like(P)
        N[hit_mask] = (P[hit_mask] - self.C) / self.R

        L = np.zeros(P.shape[:-1])
        L[hit_mask] = self._shade_point(P[hit_mask], N[hit_mask], d[hit_mask])

        self.hit_mask, self.P, self.N, self.L = hit_mask, P, N, L

        if hit_mask.any():
            self.L_max = float(L[hit_mask].max())
            self.L_min = float(L[hit_mask].min())
        else:
            self.L_max = self.L_min = 0.0

        if self.L_max > 0:
            img = np.clip(L / self.L_max * 255.0, 0, 255)
        else:
            img = np.zeros_like(L)
        img = np.where(hit_mask, img, 0.0)
        self.image_u8 = img.astype(np.uint8)
        return self.image_u8

    # --------------------------------------------------------- точки --
    def brightness_at_pixel(self, row, col):
        if not self.hit_mask[row, col]:
            return None
        return float(self.L[row, col])

    def sample_points(self):
        """3 характерные точки: ближайшая к наблюдателю (полюс сферы),
        точка блика (максимум L) и случайная освещенная боковая точка,
        плюс max/min по всей сфере."""
        result = {}

        # 1) Точка сферы, ближайшая к наблюдателю (апекс, видимый в центре эк.)
        apex_world = self.C + np.array([0, 0, self.R])
        # Проецируем апекс в пиксель экрана (приближенно, через центр изображения)
        apex_row = int(round((0 - (-self.H / 2)) / self.px - 0.5))
        apex_col = int(round((0 - (-self.W / 2)) / self.px - 0.5))
        apex_row = min(max(apex_row, 0), self.Hres - 1)
        apex_col = min(max(apex_col, 0), self.Wres - 1)
        val = self.brightness_at_pixel(apex_row, apex_col)
        result["Точка в центре изображения (под наблюдателем)"] = val if val is not None else 0.0

        # 2) Точка максимальной яркости (блик)
        if self.hit_mask.any():
            idx = np.unravel_index(np.argmax(np.where(self.hit_mask, self.L, -1)), self.L.shape)
            result["Точка блика (максимум L)"] = float(self.L[idx])
        else:
            result["Точка блика (максимум L)"] = 0.0

        # 3) Произвольная освещенная боковая точка (первая найденная слева от центра)
        side_val = None
        if self.hit_mask.any():
            rows, cols = np.where(self.hit_mask)
            left_idx = np.argmin(cols)
            side_val = float(self.L[rows[left_idx], cols[left_idx]])
        result["Боковая точка сферы (край диска)"] = side_val if side_val is not None else 0.0

        result["Максимум яркости на сфере"] = self.L_max
        result["Минимум яркости на сфере (среди видимых точек)"] = self.L_min
        return result


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР_2. Яркость сферы от точечных источников света (Блинн-Фонг)")
        self.geometry("1260x800")
        self.model = None
        self._build_ui()

    # ---------------------------------------------------------- UI --
    def _build_ui(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=(0, 8))
        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True)

        self.vars = {}

        def add_field(parent, row, key, label, default):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=str(default))
            ttk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, pady=1)
            self.vars[key] = var

        scene = ttk.LabelFrame(left, text="Экран / наблюдатель", padding=6)
        scene.pack(fill="x")
        add_field(scene, 0, "W", "W, мм [100..10000]", 2000)
        add_field(scene, 1, "H", "H, мм [100..10000]", 2000)
        add_field(scene, 2, "Wres", "Wres, px [200..800]", 400)
        add_field(scene, 3, "zO", "zO наблюдателя, мм", 4000)

        sphere = ttk.LabelFrame(left, text="Сфера", padding=6)
        sphere.pack(fill="x", pady=(6, 0))
        add_field(sphere, 0, "xC", "xC, мм", 0)
        add_field(sphere, 1, "yC", "yC, мм", 0)
        add_field(sphere, 2, "zC", "zC, мм", 1500)
        add_field(sphere, 3, "R", "R (радиус), мм", 600)

        l1 = ttk.LabelFrame(left, text="Источник света 1", padding=6)
        l1.pack(fill="x", pady=(6, 0))
        add_field(l1, 0, "xL1", "xL1, мм", -1500)
        add_field(l1, 1, "yL1", "yL1, мм", -1000)
        add_field(l1, 2, "zL1", "zL1, мм", 3000)
        add_field(l1, 3, "I01", "I0_1, Вт/ср", 3000)

        l2 = ttk.LabelFrame(left, text="Источник света 2", padding=6)
        l2.pack(fill="x", pady=(6, 0))
        add_field(l2, 0, "xL2", "xL2, мм", 1500)
        add_field(l2, 1, "yL2", "yL2, мм", 1000)
        add_field(l2, 2, "zL2", "zL2, мм", 3000)
        add_field(l2, 3, "I02", "I0_2, Вт/ср", 1500)

        phong = ttk.LabelFrame(left, text="Модель Блинна-Фонга", padding=6)
        phong.pack(fill="x", pady=(6, 0))
        add_field(phong, 0, "kd", "kd (диффузная составл.)", 0.7)
        add_field(phong, 1, "ks", "ks (зеркальная составл.)", 0.9)
        add_field(phong, 2, "ke", "ke (резкость блика)", 40)

        self.hres_label = ttk.Label(left, text="Hres (авт., квадратные пиксели): -")
        self.hres_label.pack(anchor="w", pady=(6, 0))

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Рассчитать", command=self.calculate).pack(fill="x", pady=2)
        ttk.Button(btns, text="Сохранить изображение...", command=self.save_image).pack(fill="x", pady=2)

        table_frame = ttk.LabelFrame(left, text="Контрольные значения яркости L, Вт/(ср*мм²)", padding=6)
        table_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.result_text = tk.Text(table_frame, width=40, height=10, state="disabled")
        self.result_text.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(7.0, 7.0), dpi=100)
        self.ax_img = self.fig.add_subplot(111)
        self.ax_img.set_title("Распределение яркости на сфере (0-255)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

    # ---------------------------------------------------------- logic --
    def _read_params(self):
        try:
            values = {k: float(v.get()) for k, v in self.vars.items()}
        except ValueError:
            raise ValueError("Все параметры должны быть числами.")

        checks = {
            "W": "W", "H": "H", "Wres": "Wres",
            "xL1": "xL", "yL1": "yL", "zL1": "zL", "I01": "I0",
            "xL2": "xL", "yL2": "yL", "zL2": "zL", "I02": "I0",
            "xC": "xC", "yC": "yC", "zC": "zC",
        }
        for key, limit_key in checks.items():
            lo, hi = PARAM_LIMITS[limit_key]
            if not (lo <= values[key] <= hi):
                raise ValueError(f"Параметр {key} должен быть в диапазоне [{lo}, {hi}].")

        values["Wres"] = int(values["Wres"])
        if values["R"] <= 0:
            raise ValueError("Радиус сферы R должен быть положительным.")
        if values["ks"] < 0 or values["kd"] < 0:
            raise ValueError("Коэффициенты kd, ks должны быть неотрицательными.")
        if values["ke"] <= 0:
            raise ValueError("Показатель ke должен быть положительным.")
        return values

    def calculate(self):
        try:
            v = self._read_params()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return

        lights = [
            {"pos": (v["xL1"], v["yL1"], v["zL1"]), "I0": v["I01"]},
            {"pos": (v["xL2"], v["yL2"], v["zL2"]), "I0": v["I02"]},
        ]
        self.model = SphereBrightnessModel(
            W=v["W"], H=v["H"], Wres=v["Wres"], zO=v["zO"],
            xC=v["xC"], yC=v["yC"], zC=v["zC"], R=v["R"],
            lights=lights, kd=v["kd"], ks=v["ks"], ke=v["ke"],
        )
        self.model.compute()
        self.hres_label.config(text=f"Hres (авт., квадратные пиксели): {self.model.Hres}")

        self._draw_image()
        self._show_points()

    def _draw_image(self):
        self.ax_img.clear()
        m = self.model
        extent = [-m.W / 2, m.W / 2, -m.H / 2, m.H / 2]
        self.ax_img.imshow(m.image_u8, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=255)
        self.ax_img.set_title("Распределение яркости на сфере (0-255)")
        self.ax_img.set_xlabel("X, мм")
        self.ax_img.set_ylabel("Y, мм")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _show_points(self):
        pts = self.model.sample_points()
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        for name, val in pts.items():
            self.result_text.insert("end", f"{name}:\n    {val:.6e}\n")
        self.result_text.config(state="disabled")

    def save_image(self):
        if self.model is None:
            messagebox.showwarning("Нет данных", "Сначала выполните расчет.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG image", "*.png")])
        if not path:
            return
        Image.fromarray(self.model.image_u8, mode="L").save(path)
        messagebox.showinfo("Готово", f"Изображение сохранено:\n{path}")


if __name__ == "__main__":
    App().mainloop()