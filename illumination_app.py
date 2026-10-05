"""
ЛР_1. Расчет освещенности на плоскости от точечного источника света.
Курс "Алгоритмы компьютерной графики" (3й курс, 1й семестр).

Физическая модель:
    Точечный источник L(xL, yL, zL) [мм] над плоскостью z = 0 с
    ламбертовской диаграммой излучения силы I0 [Вт/ср] (сила излучения
    вдоль нормали к источнику theta = 0).
    Для точки P(x, y, 0) плоскости:
        r      = |L - P| = sqrt((xL-x)^2 + (yL-y)^2 + zL^2)   [мм]
        cos(theta) = zL / r         (theta - угол между направлением
                                      на источник и нормалью к плоскости)
        I(theta) = I0 * cos(theta)               (сила излучения, Вт/ср)
        E(P)     = I(theta) / r^2 = I0 * zL / r^3   (освещенность, Вт/мм^2,
                                                       закон обратных квадратов)
    Расчет ведется только внутри круга радиуса R с центром (cx, cy),
    лежащего в пределах прямоугольной области W x H с центром в начале
    координат (см. приложенную к заданию схему).

Приложение (Tkinter + Matplotlib):
    - поля ввода всех параметров с проверкой на допустимые диапазоны;
    - кнопка "Рассчитать" - строит карту освещенности (нормировка 0-255)
      и график сечения через центр области;
    - кнопка "Сохранить изображение..." - сохраняет карту освещенности в PNG;
    - кнопка "Сохранить график сечения..." - сохраняет график сечения в PNG;
    - таблица контрольных значений освещенности в 5 точках.
"""

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
    "W":    (100.0, 10000.0),    # ширина области, мм
    "H":    (100.0, 10000.0),    # высота области, мм
    "Wres": (200, 800),          # разрешение по ширине, px
    "xL":   (-10000.0, 10000.0), # координата источника, мм
    "yL":   (-10000.0, 10000.0),
    "zL":   (100.0, 10000.0),
    "I0":   (0.01, 10000.0),     # сила излучения, Вт/ср
}
# Круг задается дополнительно: центр (cx, cy) должен лежать в пределах
# области, а радиус R - таким, чтобы круг помещался в прямоугольник W x H.


class IlluminationModel:
    """Расчет распределения освещенности внутри заданного круга."""

    def __init__(self, W, H, Wres, xL, yL, zL, I0, cx, cy, R):
        self.W, self.H = W, H
        self.Wres = Wres
        # Квадратные пиксели: шаг сетки одинаков по X и Y.
        px = W / Wres
        self.Hres = max(1, round(H / px))
        self.px = px
        self.xL, self.yL, self.zL, self.I0 = xL, yL, zL, I0
        self.cx, self.cy, self.R = cx, cy, R

        # Координаты центров пикселей в мм, начало координат - в центре
        # прямоугольной области (совпадает с центром круга по умолчанию).
        xs = (np.arange(self.Wres) + 0.5) * px - W / 2.0
        ys = (np.arange(self.Hres) + 0.5) * px - H / 2.0
        self.X, self.Y = np.meshgrid(xs, ys)  # shape (Hres, Wres)

        self.mask = (self.X - cx) ** 2 + (self.Y - cy) ** 2 <= R ** 2

    def illuminance(self, x, y):
        """E(x, y) по закону Ламберта с ослаблением 1/r^2."""
        r2 = (self.xL - x) ** 2 + (self.yL - y) ** 2 + self.zL ** 2
        r = np.sqrt(r2)
        cos_theta = self.zL / r
        return self.I0 * cos_theta / r2

    def compute(self):
        E = self.illuminance(self.X, self.Y)
        E_masked = np.where(self.mask, E, 0.0)
        self.E = E_masked
        self.E_max = float(E_masked[self.mask].max()) if self.mask.any() else 0.0
        self.E_min = float(E_masked[self.mask].min()) if self.mask.any() else 0.0
        self.E_mean = float(E_masked[self.mask].mean()) if self.mask.any() else 0.0

        # Изображение 0-255, нормировка на максимум внутри круга.
        if self.E_max > 0:
            img = np.clip(E_masked / self.E_max * 255.0, 0, 255)
        else:
            img = np.zeros_like(E_masked)
        self.image_u8 = img.astype(np.uint8)
        return self.image_u8

    def control_points(self):
        """5 контрольных точек: центр круга, пересечения круга с осями
        X и Y (проходящими через центр круга), max/min/mean уже посчитаны."""
        cx, cy, R = self.cx, self.cy, self.R
        pts = {
            "Центр круга (cx, cy)": (cx, cy),
            "Пересечение с осью X (cx+R, cy)": (cx + R, cy),
            "Пересечение с осью Y (cx, cy+R)": (cx, cy + R),
        }
        result = {}
        for name, (x, y) in pts.items():
            result[name] = float(self.illuminance(x, y))
        result["Максимум в круге"] = self.E_max
        result["Минимум в круге"] = self.E_min
        result["Среднее в круге"] = self.E_mean
        return result

    def cross_section(self, axis="X"):
        """Сечение, проходящее через центр круга (cx, cy)."""
        px = self.px
        if axis == "X":
            row = int(round((self.cy + self.H / 2.0) / px - 0.5))
            row = min(max(row, 0), self.Hres - 1)
            coord = self.X[row, :]
            values = self.E[row, :]
        else:
            col = int(round((self.cx + self.W / 2.0) / px - 0.5))
            col = min(max(col, 0), self.Wres - 1)
            coord = self.Y[:, col]
            values = self.E[:, col]
        return coord, values


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР_1. Освещенность плоскости от точечного источника света")
        self.geometry("1200x760")
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

        # ---- поля ввода параметров ------------------------------------
        params = ttk.LabelFrame(left, text="Параметры", padding=8)
        params.pack(fill="x")

        self.vars = {}

        def add_field(row, key, label, default):
            ttk.Label(params, text=label).grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(default))
            ttk.Entry(params, textvariable=var, width=12).grid(row=row, column=1, pady=2)
            self.vars[key] = var

        add_field(0, "W", "W - ширина области, мм [100..10000]", 2000)
        add_field(1, "H", "H - высота области, мм [100..10000]", 2000)
        add_field(2, "Wres", "Wres - разрешение по ширине, px [200..800]", 400)
        add_field(3, "xL", "xL - X источника света, мм [-10000..10000]", 0)
        add_field(4, "yL", "yL - Y источника света, мм [-10000..10000]", 0)
        add_field(5, "zL", "zL - Z источника света, мм [100..10000]", 1000)
        add_field(6, "I0", "I0 - сила излучения, Вт/ср [0.01..10000]", 1000)

        ttk.Separator(params).grid(row=7, column=0, columnspan=2, sticky="ew", pady=6)

        add_field(8, "cx", "cx - центр круга, X, мм", 0)
        add_field(9, "cy", "cy - центр круга, Y, мм", 0)
        add_field(10, "R", "R - радиус круга, мм", 800)

        self.hres_label = ttk.Label(params, text="Hres (авт., квадратные пиксели): -")
        self.hres_label.grid(row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(params, text="Сечение через центр вдоль оси:").grid(
            row=12, column=0, columnspan=2, sticky="w", pady=(8, 2)
        )
        self.axis_var = tk.StringVar(value="X")
        axis_frame = ttk.Frame(params)
        axis_frame.grid(row=13, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(axis_frame, text="X", variable=self.axis_var, value="X",
                         command=self._redraw_section).pack(side="left")
        ttk.Radiobutton(axis_frame, text="Y", variable=self.axis_var, value="Y",
                         command=self._redraw_section).pack(side="left")

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="Рассчитать", command=self.calculate).pack(fill="x", pady=2)
        ttk.Button(btns, text="Сохранить изображение...", command=self.save_image).pack(fill="x", pady=2)
        ttk.Button(btns, text="Сохранить график сечения...", command=self.save_section).pack(fill="x", pady=2)

        # ---- таблица контрольных значений ------------------------------
        table_frame = ttk.LabelFrame(left, text="Контрольные значения освещенности, Вт/мм²", padding=8)
        table_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.result_text = tk.Text(table_frame, width=38, height=12, state="disabled")
        self.result_text.pack(fill="both", expand=True)

        # ---- график/изображение -----------------------------------------
        self.fig = Figure(figsize=(7.5, 6.5), dpi=100)
        self.ax_img = self.fig.add_subplot(211)
        self.ax_section = self.fig.add_subplot(212)
        self.ax_img.set_title("Распределение освещенности (0-255)")
        self.ax_section.set_title("Сечение через центр области")

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

        for key in ("W", "H", "Wres", "xL", "yL", "zL", "I0"):
            lo, hi = PARAM_LIMITS[key]
            if not (lo <= values[key] <= hi):
                raise ValueError(f"Параметр {key} должен быть в диапазоне [{lo}, {hi}].")

        values["Wres"] = int(values["Wres"])

        W, H, R = values["W"], values["H"], values["R"]
        cx, cy = values["cx"], values["cy"]
        if R <= 0:
            raise ValueError("Радиус круга R должен быть положительным.")
        if (abs(cx) + R > W / 2) or (abs(cy) + R > H / 2):
            raise ValueError("Круг (центр + радиус) должен целиком помещаться в область W x H.")

        return values

    def calculate(self):
        try:
            v = self._read_params()
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return

        self.model = IlluminationModel(
            W=v["W"], H=v["H"], Wres=v["Wres"],
            xL=v["xL"], yL=v["yL"], zL=v["zL"], I0=v["I0"],
            cx=v["cx"], cy=v["cy"], R=v["R"],
        )
        self.model.compute()
        self.hres_label.config(text=f"Hres (авт., квадратные пиксели): {self.model.Hres}")

        self._draw_image()
        self._redraw_section()
        self._show_control_points()

    def _draw_image(self):
        self.ax_img.clear()
        m = self.model
        extent = [-m.W / 2, m.W / 2, -m.H / 2, m.H / 2]
        self.ax_img.imshow(m.image_u8, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=255)
        circle = matplotlib.patches.Circle((m.cx, m.cy), m.R, fill=False, edgecolor="red", linewidth=1)
        self.ax_img.add_patch(circle)
        self.ax_img.set_title("Распределение освещенности (0-255)")
        self.ax_img.set_xlabel("X, мм")
        self.ax_img.set_ylabel("Y, мм")
        self.canvas.draw_idle()

    def _redraw_section(self):
        if self.model is None:
            return
        axis = self.axis_var.get()
        coord, values = self.model.cross_section(axis=axis)
        self.ax_section.clear()
        self.ax_section.plot(coord, values)
        self.ax_section.set_title(f"Сечение через центр области (вдоль {axis})")
        self.ax_section.set_xlabel(f"{axis}, мм")
        self.ax_section.set_ylabel("E, Вт/мм²")
        self.ax_section.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _show_control_points(self):
        pts = self.model.control_points()
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        for name, val in pts.items():
            self.result_text.insert("end", f"{name}:\n    {val:.6e}\n")
        self.result_text.config(state="disabled")

    # ---------------------------------------------------------- I/O --
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

    def save_section(self):
        if self.model is None:
            messagebox.showwarning("Нет данных", "Сначала выполните расчет.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG image", "*.png")])
        if not path:
            return
        section_fig = Figure(figsize=(6, 4), dpi=100)
        ax = section_fig.add_subplot(111)
        axis = self.axis_var.get()
        coord, values = self.model.cross_section(axis=axis)
        ax.plot(coord, values)
        ax.set_title(f"Сечение через центр области (вдоль {axis})")
        ax.set_xlabel(f"{axis}, мм")
        ax.set_ylabel("E, Вт/мм²")
        ax.grid(True, alpha=0.3)
        section_fig.tight_layout()
        section_fig.savefig(path)
        messagebox.showinfo("Готово", f"График сечения сохранен:\n{path}")


if __name__ == "__main__":
    import matplotlib.patches 
    App().mainloop()
