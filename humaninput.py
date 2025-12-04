
##### Windows API
import ctypes
from ctypes import wintypes

if hasattr(wintypes, "ULONG_PTR"):
    ULONG_PTR = wintypes.ULONG_PTR
else:
    ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

INPUT_MOUSE      = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx",          wintypes.LONG),
        ("dy",          wintypes.LONG),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )

class _INPUTunion(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT),)

class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTunion),)

SendInput = ctypes.windll.user32.SendInput

def move_relative(dx: int, dy: int):
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0,
                        dwFlags=MOUSEEVENTF_MOVE,
                        time=0, dwExtraInfo=ULONG_PTR(0))
    sent = SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    if sent != 1:
        raise OSError("SendInput fehlgeschlagen")

####methoden TODO

from typing import List, Tuple, Optional
import math, random

def human_path_to_origin_pro(
        x0: float, y0: float,
        steps: int = 64,
        curvature: float = 0.12,
        jitter_px: float = 0.25,
        tremor_px: float = 0.08,
        overshoot_px: float = 0.6,
        stall_prob: float = 0.15,
        seed: Optional[int] = None
) -> List[Tuple[float, float]]:
    # --- harte Absicherung gegen Float-Schritte ---
    steps = int(steps)
    if steps <= 1:
        return [(0.0, 0.0)]

    if seed is not None:
        random.seed(seed)

    dx, dy = -x0, -y0
    L = math.hypot(dx, dy)
    if L == 0:
        return [(0.0, 0.0)] * steps

    ux, uy = dx / L, dy / L
    vx, vy = -uy, ux

    def s_curve(t: float) -> float:
        return 10*t**3 - 15*t**4 + 6*t**5

    # Fortschritt 0..1, normiert
    t_vals = [(i + 1) / steps for i in range(steps)]
    prog = [s_curve(t) for t in t_vals]
    last = prog[-1]
    prog = [p / last for p in prog]

    c = curvature * 0.25 * L
    trem_cycles = random.uniform(2.0, 5.0)

    a = 0.82
    jx = jy = 0.0
    jitter_kick = jitter_px / max(1, steps // 8)

    do_stall = (random.random() < stall_prob) and steps >= 6
    if do_stall:
        stall_idx = random.randint(steps // 3, 2 * steps // 3)  # beide sind int
    else:
        stall_idx = -1

    points: List[Tuple[float, float]] = []
    for i, p in enumerate(prog, start=1):
        base_x = x0 + ux * (p * L)
        base_y = y0 + uy * (p * L)

        offset = math.sin(math.pi * p) * c

        trem = math.sin(2 * math.pi * trem_cycles * p)
        tx = tremor_px * trem
        ty = tremor_px * trem

        jx = a * jx + (1 - a) * random.uniform(-jitter_kick, jitter_kick)
        jy = a * jy + (1 - a) * random.uniform(-jitter_kick, jitter_kick)
        jx_aniso = jx * 1.1
        jy_aniso = jy * 0.9

        px = base_x + vx * offset + (jx_aniso + tx) * 0.85 + ux * (jx_aniso * 0.15)
        py = base_y + vy * offset + (jy_aniso + ty) * 0.85 + uy * (jy_aniso * 0.15)

        points.append((px, py))

    if do_stall and 1 <= stall_idx < steps - 1:
        points[stall_idx] = points[stall_idx - 1]

    pen = steps - 2
    if pen >= 0 and overshoot_px > 0:
        ox = points[pen][0] + ux * overshoot_px
        oy = points[pen][1] + uy * overshoot_px
        points[pen] = (ox, oy)

    points[-1] = (0.0, 0.0)
    return points

from typing import List, Tuple

def positions_to_int_deltas(points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    """Wandelt absolute Punkte in int-Relative (dx,dy) um – driftfrei via Fehlerakku."""
    deltas: List[Tuple[int, int]] = []
    if not points:
        return deltas

    prev_x, prev_y = points[0]  # erster Punkt als Start
    err_x = 0.0
    err_y = 0.0

    for x, y in points[1:]:
        fx = (x - prev_x) + err_x
        fy = (y - prev_y) + err_y
        dx = int(round(fx))
        dy = int(round(fy))
        err_x = fx - dx
        err_y = fy - dy
        deltas.append((dx, dy))
        prev_x, prev_y = x, y

    return deltas


def mov(x: float, y: float):
    points = human_path_to_origin_pro(x, y, steps=64, seed=42)  # 64 absolute Punkte
    # sichere Reihenfolge: [Startpunkt] + Punkte → damit der erste Delta-Schritt korrekt ist
    points_with_start = [(x, y)] + points
    deltas = positions_to_int_deltas(points_with_start)

    for i, (dx, dy) in enumerate(deltas, start=1):
        move_relative(-1 *dx,-1*dy)









