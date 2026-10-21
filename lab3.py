import csv
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
from PIL import Image

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CIE_FILE = os.path.join(BASE_DIR, "cie_xyz_1931_10nm.csv")
SPECTRA_FILE = os.path.join(BASE_DIR, "spectra_library.csv")

# ------------------------------------------------------------------
# Допустимые диапазоны параметров (по тексту задания)
# ------------------------------------------------------------------
PARAM_LIMITS = {
    "W":    (100.0, 10000.0),
    "H":    (100.0, 10000.0),
    "Wres": (200, 800),
    "xL":   (-10000.0, 10000.0),
    "yL":   (-10000.0, 10000.0),
    "zL":   (-10000.0, 10000.0),
    "xC":   (-10000.0, 10000.0),
    "yC":   (-10000.0, 10000.0),
    "zC":   (0.0, 10000.0),
    "zO":   (100.0, 10000.0),
    "I0":   (0.01, 10000.0),
}

LAMBDA = np.arange(380.0, 781.0, 10.0)   # длины волн, нм (41 отсчет)
DLAMBDA = 10.0                            # шаг интегрирования, нм
LAMP_AXIS = np.array([0.0, 0.0, -1.0])    # лампы светят вертикально вниз
EPS = 1e-9
WHITE_NAME = "D65 (белый)"                # опорный источник для Kd_RGB

# Матрицы CIE RGB 1931. Множитель 1/0.17697 из задания опущен: без него прямая
# матрица согласована с обратной (M_RGB2XYZ @ M_XYZ2RGB = E), и светимость Y по
# RGB совпадает с Y, вычисленной напрямую из спектра.
M_RGB2XYZ = np.array([[0.49000, 0.31000, 0.20000],
                      [0.17697, 0.81240, 0.01063],
                      [0.00000, 0.01000, 0.99000]])
M_XYZ2RGB = np.array([[ 2.36461, -0.89654, -0.46807],
                      [-0.51517,  1.42641,  0.08876],
                      [ 0.00520, -0.01441,  1.00920]])
Y_ROW = M_RGB2XYZ[1]    # вторая строка матрицы RGB->XYZ: светимость Y по RGB


# ------------------------------------------------------------------
# Таблицы: CIE XYZ и библиотека спектров
# ------------------------------------------------------------------
def load_cmf(path=CIE_FILE):
    """Функции сложения цветов: массив (41, 3) - x(l), y(l), z(l)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if not np.allclose(data[:, 0], LAMBDA):
        raise ValueError("Таблица CIE должна быть на 380..780 нм с шагом 10 нм.")
    return data[:, 1:4]


def load_library(path=SPECTRA_FILE):
    """Библиотека спектров: (источники, отражения) - словари имя -> массив (41,)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:]])
    if not np.allclose(data[:, 0], LAMBDA):
        raise ValueError("Спектры должны быть на 380..780 нм с шагом 10 нм.")
    sources, kds = {}, {}
    for j, name in enumerate(header[1:], start=1):
        if name.startswith("S:"):
            sources[name[2:]] = data[:, j]
        elif name.startswith("K:"):
            kds[name[2:]] = data[:, j]
    return sources, kds


def read_spectrum_file(path):
    """Читает спектр из CSV/TXT (две колонки: длина волны, значение; разделитель
    запятая, точка с запятой или пробел) и приводит к сетке 380..780 с шагом 10 нм."""
    lam, val = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.replace(";", ",").replace("\t", ",").replace(" ", ",").split(",")
            parts = [p for p in parts if p.strip() != ""]
            try:
                lam.append(float(parts[0]))
                val.append(float(parts[1]))
            except (ValueError, IndexError):
                continue  # заголовок или пустая строка
    if len(lam) < 2:
        raise ValueError("В файле не найдено двух числовых колонок (длина волны, значение).")
    lam, val = np.array(lam), np.array(val)
    order = np.argsort(lam)
    return np.interp(LAMBDA, lam[order], val[order], left=0.0, right=0.0)


# ------------------------------------------------------------------
# Колориметрия
# ------------------------------------------------------------------
def spectrum_to_xyz(spec, cmf):
    """XYZ = интеграл spec(l) * {x,y,z}(l) dl (сумма по 41 отсчету, шаг 10 нм).
    spec: (..., 41) -> (..., 3)."""
    return np.tensordot(spec, cmf, axes=([-1], [0])) * DLAMBDA


def xyz_to_rgb(xyz):
    """XYZ -> линейный RGB (обратная матрица из задания). (..., 3) -> (..., 3)."""
    return np.tensordot(xyz, M_XYZ2RGB, axes=([-1], [1]))


def kd_rgb_from_spectrum(kd, white, cmf):
    """Kd_RGB = RGB(D65 * Kd) / RGB(D65) по каналам (определение из задания)."""
    rgb_w = xyz_to_rgb(spectrum_to_xyz(white, cmf))
    rgb_k = xyz_to_rgb(spectrum_to_xyz(white * kd, cmf))
    return rgb_k / rgb_w


def to_monitor(rgb_abs, mask, gamma=1.0):
    """RGB (абсолютные линейные единицы) -> мониторные 0..255 (uint8).
    Нормировка на максимальное значение по всему изображению (один общий
    коэффициент, чтобы не менять цветность), отрицательные значения
    (цвета вне гаммы) отсекаются в 0 (gamut mapping по лекции)."""
    peak = rgb_abs[mask].max()
    v = np.clip(rgb_abs / peak, 0.0, 1.0) ** (1.0 / gamma)
    v = np.where(mask[..., None], v, 0.0)
    return np.round(v * 255.0).astype(np.uint8)


def to_gray(rgb_abs, mask, gamma=1.0):
    """Тона серого: яркость Y (вторая строка матрицы RGB->XYZ из задания),
    нормировка на максимум и перевод в 0..255."""
    Y = rgb_abs @ Y_ROW
    peak = Y[mask].max()
    v = np.clip(Y / peak, 0.0, 1.0) ** (1.0 / gamma)
    v = np.where(mask, v, 0.0)
    return np.round(v * 255.0).astype(np.uint8)


def diff_stats(a, b, mask):
    """Различие двух изображений (в мониторных единицах) по пикселям сферы."""
    d = np.abs(a.astype(float) - b.astype(float))[mask]
    return {"mean": float(d.mean()), "max": float(d.max()),
            "rms": float(np.sqrt((d ** 2).mean()))}


# ------------------------------------------------------------------
# Модель сцены
# ------------------------------------------------------------------
class ColorSphereModel:
    """Расчет изображения цветной ламбертовской сферы двумя способами."""

    def __init__(self, W, H, Wres, zO, xC, yC, zC, R, lights, kd_spec,
                 cmf, white_spec, gamma=1.0, cos_in_f=True):
        """
        lights: список {"pos": (xL, yL, zL), "I0": I0, "spec": массив (41,)}
        kd_spec: спектр отражения сферы (41,), значения 0..1
        white_spec: спектр D65 (41,), опорный для Kd_RGB
        cos_in_f: True - f = Kd * cos(sigma) (как в задании), False - f = Kd
        """
        self.W, self.H, self.Wres = W, H, Wres
        px = W / Wres
        self.Hres = max(1, round(H / px))
        self.px = px

        self.eye = np.array([0.0, 0.0, zO])
        self.C = np.array([xC, yC, zC])
        self.R = R
        self.lights = lights
        self.kd_spec = kd_spec
        self.cmf = cmf
        self.white = white_spec
        self.gamma = gamma
        self.cos_in_f = cos_in_f

        xs = (np.arange(self.Wres) + 0.5) * px - W / 2.0
        ys = (np.arange(self.Hres) + 0.5) * px - H / 2.0
        X, Y = np.meshgrid(xs, ys)
        self.S = np.stack([X, Y, np.zeros_like(X)], axis=-1)   # точки экрана

    # ----------------------------------------------------------- геометрия --
    def _intersect_sphere(self):
        """Ближайшее к наблюдателю пересечение лучей (eye -> пиксель) со сферой."""
        d = self.S - self.eye
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        oc = self.eye - self.C
        b = 2.0 * np.sum(d * oc, axis=-1)
        c = np.dot(oc, oc) - self.R ** 2
        disc = b ** 2 - 4.0 * c
        hit = disc >= 0
        t = (-b - np.sqrt(np.clip(disc, 0, None))) / 2.0
        hit &= t > EPS
        P = self.eye + t[..., None] * d
        return hit, P

    def _check_scene(self, hit):
        if np.linalg.norm(self.eye - self.C) <= self.R:
            raise ValueError("Наблюдатель находится внутри сферы: увеличьте zO "
                             "или уменьшите радиус.")
        if not hit.any():
            raise ValueError("Сфера не видна наблюдателю через экран (проверьте "
                             "zO, zC, положение и размер экрана).")
        if hit[0, :].any() or hit[-1, :].any() or hit[:, 0].any() or hit[:, -1].any():
            raise ValueError("Сфера не помещается в область видимости: увеличьте "
                             "размер экрана W, H или уменьшите радиус сферы.")

    def _geometry_factors(self, P, N):
        """Спектрально-независимая часть яркости от каждого источника:
        g_i = I0 * cos(theta) * cos(sigma) / R^2 * [cos(sigma)] / pi."""
        g = []
        for light in self.lights:
            s = P - np.array(light["pos"])                 # от источника к точке
            r = np.linalg.norm(s, axis=-1)
            r = np.where(r > EPS, r, EPS)
            cos_theta = np.clip(np.sum(s * LAMP_AXIS, axis=-1) / r, 0.0, None)
            # Вектор s в задании идет от источника к точке, а угол sigma - между
            # нормалью и направлением на источник; берем l = -s/R (как на рисунке).
            l = -s / r[..., None]
            cos_sigma = np.clip(np.sum(N * l, axis=-1), 0.0, None)
            val = light["I0"] * cos_theta * cos_sigma / r ** 2
            if self.cos_in_f:
                val = val * cos_sigma                       # f = Kd * cos(sigma)
            g.append(val / np.pi)
        return g

    # -------------------------------------------------------------- расчет --
    def compute(self):
        hit, P = self._intersect_sphere()
        self._check_scene(hit)
        N = (P[hit] - self.C) / self.R
        g = self._geometry_factors(P[hit], N)

        # а) спектральное изображение (в памяти, абсолютные единицы): (Nhit, 41)
        spec = np.zeros((hit.sum(), len(LAMBDA)))
        for gi, light in zip(g, self.lights):
            spec += gi[:, None] * (light["spec"] * self.kd_spec)[None, :]
        rgb1_hit = xyz_to_rgb(spectrum_to_xyz(spec, self.cmf))

        # б) RGB-сцена: источники -> RGB, Kd -> Kd_RGB, расчет по трем каналам
        kd_rgb = kd_rgb_from_spectrum(self.kd_spec, self.white, self.cmf)
        rgb2_hit = np.zeros_like(rgb1_hit)
        for gi, light in zip(g, self.lights):
            src_rgb = xyz_to_rgb(spectrum_to_xyz(light["spec"], self.cmf))
            rgb2_hit += gi[:, None] * (src_rgb * kd_rgb)[None, :]

        shape = hit.shape
        self.hit = hit
        self.spec_hit = spec
        self.kd_rgb = kd_rgb
        self.rgb1 = np.zeros(shape + (3,)); self.rgb1[hit] = rgb1_hit
        self.rgb2 = np.zeros(shape + (3,)); self.rgb2[hit] = rgb2_hit
        self.Y1 = self.rgb1 @ Y_ROW
        self.Y2 = self.rgb2 @ Y_ROW

        # мониторные изображения и тона серого
        self.image_1 = to_monitor(self.rgb1, hit, self.gamma)
        self.image_2 = to_monitor(self.rgb2, hit, self.gamma)
        self.gray_1 = to_gray(self.rgb1, hit, self.gamma)
        self.gray_2 = to_gray(self.rgb2, hit, self.gamma)
        self.diff_rgb = diff_stats(self.image_1, self.image_2,
                                   np.repeat(hit[..., None], 3, axis=-1))
        self.diff_gray = diff_stats(self.gray_1, self.gray_2, hit)
        return self

    # --------------------------------------------------------------- точки --
    def sample_points(self):
        """Три точки сферы (центр диска, блик/максимум яркости, боковая точка) и
        максимум/минимум яркости Y по сфере. Значения - абсолютные (линейные)."""
        rows, cols = np.where(self.hit)
        r0 = (rows.min() + rows.max()) // 2
        c_mid = (cols.min() + cols.max()) // 2
        c_side = c_mid - int(0.7 * (cols.max() - cols.min()) / 2)
        if not self.hit[r0, c_side]:
            c_side = cols[rows == r0].min()
        idx_max = np.unravel_index(np.argmax(np.where(self.hit, self.Y1, -np.inf)),
                                   self.Y1.shape)
        pts = [("Центр диска сферы (под наблюдателем)", (r0, c_mid)),
               ("Максимум яркости (блик)", idx_max),
               ("Боковая точка сферы", (r0, c_side))]
        result = []
        for name, (r, c) in pts:
            result.append({"name": name, "pixel": (int(r), int(c)),
                           "rgb1": self.rgb1[r, c], "rgb2": self.rgb2[r, c],
                           "Y1": float(self.Y1[r, c]), "Y2": float(self.Y2[r, c])})
        extremes = {"Y1_max": float(self.Y1[self.hit].max()),
                    "Y1_min": float(self.Y1[self.hit].min()),
                    "Y2_max": float(self.Y2[self.hit].max()),
                    "Y2_min": float(self.Y2[self.hit].min())}
        return result, extremes


# ------------------------------------------------------------------
# Приложение
# ------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР_3. Цветное изображение сферы: спектральный и RGB расчеты")
        self.geometry("1500x900")
        self.model = None

        try:
            self.cmf = load_cmf()
            self.sources, self.kds = load_library()
        except (OSError, ValueError) as e:
            messagebox.showerror("Ошибка загрузки таблиц", str(e))
            raise SystemExit(1)
        self.white = self.sources[WHITE_NAME]
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

        def add_row(parent, row, fields):
            """Строка из нескольких полей ввода: fields = [(ключ, подпись, значение)]."""
            for i, (key, label, default) in enumerate(fields):
                ttk.Label(parent, text=label).grid(row=row, column=2 * i, sticky="w",
                                                   padx=(0 if i == 0 else 8, 2), pady=1)
                var = tk.StringVar(value=str(default))
                ttk.Entry(parent, textvariable=var, width=9).grid(row=row, column=2 * i + 1, pady=1)
                self.vars[key] = var

        def add_combo(parent, row, key, label, values, default, kind):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=1)
            var = tk.StringVar(value=default)
            cb = ttk.Combobox(parent, textvariable=var, values=values,
                              state="readonly", width=24)
            cb.grid(row=row, column=1, columnspan=2, pady=1, sticky="w")
            ttk.Button(parent, text="CSV...", width=6,
                       command=lambda: self._load_spectrum(key, kind, cb)).grid(
                row=row, column=3, padx=2)
            self.vars[key] = var
            return cb

        screen = ttk.LabelFrame(left, text="Экран и наблюдатель", padding=5)
        screen.pack(fill="x")
        add_row(screen, 0, [("W", "W, мм", 2000), ("H", "H, мм", 2000)])
        add_row(screen, 1, [("Wres", "Wres, px", 400), ("zO", "zO, мм", 4000)])

        sphere = ttk.LabelFrame(left, text="Сфера", padding=5)
        sphere.pack(fill="x", pady=(5, 0))
        add_row(sphere, 0, [("xC", "xC, мм", 0), ("yC", "yC, мм", 0)])
        add_row(sphere, 1, [("zC", "zC, мм", 1500), ("R", "R, мм", 600)])
        self.cb_kd = add_combo(sphere, 2, "kd", "Kd (спектр)", sorted(self.kds),
                               "Жёлтый", "kd")

        l1 = ttk.LabelFrame(left, text="Источник света 1", padding=5)
        l1.pack(fill="x", pady=(5, 0))
        add_row(l1, 0, [("xL1", "xL, мм", -1500), ("yL1", "yL, мм", -1000)])
        add_row(l1, 1, [("zL1", "zL, мм", 3000), ("I01", "I0, Вт/ср", 3000)])
        self.cb_s1 = add_combo(l1, 2, "spec1", "Спектр", sorted(self.sources),
                               WHITE_NAME, "src")

        l2 = ttk.LabelFrame(left, text="Источник света 2", padding=5)
        l2.pack(fill="x", pady=(5, 0))
        add_row(l2, 0, [("xL2", "xL, мм", 1500), ("yL2", "yL, мм", 1000)])
        add_row(l2, 1, [("zL2", "zL, мм", 3000), ("I02", "I0, Вт/ср", 3000)])
        self.cb_s2 = add_combo(l2, 2, "spec2", "Спектр", sorted(self.sources),
                               WHITE_NAME, "src")

        mon = ttk.LabelFrame(left, text="Модель и монитор", padding=5)
        mon.pack(fill="x", pady=(5, 0))
        add_row(mon, 0, [("gamma", "Гамма монитора", 1.0)])

        quick = ttk.LabelFrame(left, text="Варианты источников", padding=5)
        quick.pack(fill="x", pady=(5, 0))
        ttk.Button(quick, text="Белые источники (D65)",
                   command=lambda: self._set_variant("white")).pack(fill="x", pady=1)
        ttk.Button(quick, text="Цветные источники (красный + синий)",
                   command=lambda: self._set_variant("colored")).pack(fill="x", pady=1)

        self.hres_label = ttk.Label(left, text="Hres (авт., квадратные пиксели): -")
        self.hres_label.pack(anchor="w", pady=(5, 0))

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="Рассчитать", command=self.calculate).pack(fill="x", pady=1)
        ttk.Button(btns, text="Сохранить image_1, image_2 и серые...",
                   command=self.save_images).pack(fill="x", pady=1)
        ttk.Button(btns, text="Сравнить белые и цветные источники...",
                   command=self.compare_variants).pack(fill="x", pady=1)

        # --- результаты справа: 2x2 изображения + таблица значений ---
        self.fig = Figure(figsize=(7.4, 5.4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.result_text = tk.Text(right, height=14, state="disabled",
                                   font=("Courier", 9))
        self.result_text.pack(side="bottom", fill="x", pady=(4, 0))
        NavigationToolbar2Tk(self.canvas, right).pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # ---------------------------------------------------------- спектры --
    def _load_spectrum(self, key, kind, combo):
        path = filedialog.askopenfilename(
            title="Спектр: CSV/TXT (длина волны, значение)",
            filetypes=[("Таблица", "*.csv *.txt"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            spec = read_spectrum_file(path)
        except (OSError, ValueError) as e:
            messagebox.showerror("Ошибка чтения спектра", str(e))
            return
        name = "файл: " + os.path.basename(path)
        target = self.kds if kind == "kd" else self.sources
        target[name] = spec
        for cb in ((self.cb_kd,) if kind == "kd" else (self.cb_s1, self.cb_s2)):
            cb["values"] = sorted(target)
        self.vars[key].set(name)

    def _set_variant(self, variant):
        if variant == "white":
            self.vars["spec1"].set(WHITE_NAME)
            self.vars["spec2"].set(WHITE_NAME)
        else:
            self.vars["spec1"].set("Красный")
            self.vars["spec2"].set("Синий")

    # ---------------------------------------------------------- логика --
    def _read_params(self):
        try:
            keys = [k for k in self.vars if k not in ("kd", "spec1", "spec2")]
            v = {k: float(self.vars[k].get().replace(",", ".")) for k in keys}
        except ValueError:
            raise ValueError("Все числовые параметры должны быть числами.")
        checks = {"W": "W", "H": "H", "Wres": "Wres", "zO": "zO",
                  "xC": "xC", "yC": "yC", "zC": "zC",
                  "xL1": "xL", "yL1": "yL", "zL1": "zL", "I01": "I0",
                  "xL2": "xL", "yL2": "yL", "zL2": "zL", "I02": "I0"}
        for key, limit_key in checks.items():
            lo, hi = PARAM_LIMITS[limit_key]
            if not (lo <= v[key] <= hi):
                raise ValueError(f"Параметр {key} должен быть в диапазоне [{lo}, {hi}].")
        v["Wres"] = int(v["Wres"])
        if v["R"] <= 0:
            raise ValueError("Радиус сферы R должен быть положительным.")
        if v["gamma"] <= 0:
            raise ValueError("Гамма монитора должна быть положительной.")
        return v

    def _build_model(self, v, spec1, spec2):
        lights = [
            {"pos": (v["xL1"], v["yL1"], v["zL1"]), "I0": v["I01"], "spec": spec1},
            {"pos": (v["xL2"], v["yL2"], v["zL2"]), "I0": v["I02"], "spec": spec2},
        ]
        return ColorSphereModel(
            W=v["W"], H=v["H"], Wres=v["Wres"], zO=v["zO"],
            xC=v["xC"], yC=v["yC"], zC=v["zC"], R=v["R"], lights=lights,
            kd_spec=self.kds[self.vars["kd"].get()], cmf=self.cmf,
            white_spec=self.white, gamma=v["gamma"])

    def calculate(self):
        try:
            v = self._read_params()
            m = self._build_model(v, self.sources[self.vars["spec1"].get()],
                                  self.sources[self.vars["spec2"].get()])
            m.compute()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.model = m
        self.hres_label.config(text=f"Hres (авт., квадратные пиксели): {m.Hres}")
        self._draw()
        self._show_points()

    def _draw(self):
        m = self.model
        extent = [-m.W / 2, m.W / 2, -m.H / 2, m.H / 2]
        self.fig.clear()
        items = [("image_1: спектральный расчет", m.image_1, None),
                 ("image_2: расчет в RGB", m.image_2, None),
                 ("image_1, тона серого", m.gray_1, "gray"),
                 ("image_2, тона серого", m.gray_2, "gray")]
        for k, (title, img, cmap) in enumerate(items, start=1):
            ax = self.fig.add_subplot(2, 2, k)
            ax.imshow(img, cmap=cmap, origin="lower", extent=extent, vmin=0, vmax=255)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _show_points(self):
        m = self.model
        pts, ext = m.sample_points()
        lines = ["Яркость в абсолютных единицах (линейный RGB; Y - светимость):"]
        for p in pts:
            r1, r2 = p["rgb1"], p["rgb2"]
            lines.append(f"{p['name']}, пиксель {p['pixel']}:")
            lines.append(f"  спектр.: RGB=({r1[0]:.3e}, {r1[1]:.3e}, {r1[2]:.3e})  Y={p['Y1']:.3e}")
            lines.append(f"  RGB-сц.: RGB=({r2[0]:.3e}, {r2[1]:.3e}, {r2[2]:.3e})  Y={p['Y2']:.3e}")
        lines.append(f"Y max / min по сфере: спектр. {ext['Y1_max']:.3e} / {ext['Y1_min']:.3e};"
                     f"  RGB-сц. {ext['Y2_max']:.3e} / {ext['Y2_min']:.3e}")
        lines.append(f"Различие image_1 и image_2 (0..255): "
                     f"среднее {m.diff_rgb['mean']:.2f}, максимум {m.diff_rgb['max']:.0f}; "
                     f"серые: среднее {m.diff_gray['mean']:.2f}, максимум {m.diff_gray['max']:.0f}")
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", "\n".join(lines))
        self.result_text.config(state="disabled")

    # ---------------------------------------------------- сохранение/сравнение --
    @staticmethod
    def _save_set(m, folder, suffix=""):
        Image.fromarray(m.image_1, mode="RGB").save(os.path.join(folder, f"image_1{suffix}.png"))
        Image.fromarray(m.image_2, mode="RGB").save(os.path.join(folder, f"image_2{suffix}.png"))
        Image.fromarray(m.gray_1, mode="L").save(os.path.join(folder, f"image_1_gray{suffix}.png"))
        Image.fromarray(m.gray_2, mode="L").save(os.path.join(folder, f"image_2_gray{suffix}.png"))

    def save_images(self):
        if self.model is None:
            messagebox.showwarning("Нет данных", "Сначала выполните расчет.")
            return
        folder = filedialog.askdirectory(title="Папка для image_1, image_2 и серых изображений")
        if not folder:
            return
        self._save_set(self.model, folder)
        messagebox.showinfo("Готово", "Сохранены: image_1.png, image_2.png, "
                                      "image_1_gray.png, image_2_gray.png\n" + folder)

    def compare_variants(self):
        """Вариант 1 - белые источники (D65), вариант 2 - цветные (красный и синий)
        при остальных текущих параметрах; сравнение image_1 и image_2."""
        try:
            v = self._read_params()
            models = {}
            for name, (s1, s2) in {"белые": (WHITE_NAME, WHITE_NAME),
                                   "цветные": ("Красный", "Синий")}.items():
                models[name] = self._build_model(v, self.sources[s1],
                                                 self.sources[s2]).compute()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
            return
        self.compare_models = models

        win = tk.Toplevel(self)
        win.title("Сравнение image_1 (спектр) и image_2 (RGB): белые и цветные источники")
        win.geometry("1100x820")
        fig = Figure(figsize=(10.5, 6.2), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=win)
        m0 = next(iter(models.values()))
        extent = [-m0.W / 2, m0.W / 2, -m0.H / 2, m0.H / 2]
        stats = []
        for row, (name, m) in enumerate(models.items()):
            diff = np.abs(m.image_1.astype(float) - m.image_2.astype(float))
            for col, (title, img) in enumerate([("image_1 (спектр)", m.image_1),
                                                ("image_2 (RGB)", m.image_2),
                                                ("|разность| ×4", np.clip(diff * 4, 0, 255).astype(np.uint8))]):
                ax = fig.add_subplot(2, 3, row * 3 + col + 1)
                ax.imshow(img, origin="lower", extent=extent)
                ax.set_title(f"{name} источники: {title}", fontsize=9)
                ax.tick_params(labelsize=7)
            stats.append(f"{name}: среднее различие {m.diff_rgb['mean']:.2f}, "
                         f"максимум {m.diff_rgb['max']:.0f} (из 255); серые: "
                         f"среднее {m.diff_gray['mean']:.2f}, максимум {m.diff_gray['max']:.0f}")
        fig.tight_layout()

        def save_all():
            folder = filedialog.askdirectory(title="Папка для изображений сравнения")
            if folder:
                self._save_set(models["белые"], folder, "_white")
                self._save_set(models["цветные"], folder, "_colored")
                messagebox.showinfo("Готово", "Изображения сохранены в папку:\n" + folder)

        # элементы снизу вверх, затем холст занимает оставшееся место
        ttk.Button(win, text="Сохранить все изображения сравнения...",
                   command=save_all).pack(side="bottom", pady=6)
        ttk.Label(win, text="\n".join(stats), justify="left").pack(
            side="bottom", anchor="w", padx=8)
        NavigationToolbar2Tk(canvas, win).pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)


if __name__ == "__main__":
    App().mainloop()