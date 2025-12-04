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

# overlay_trt_threaded_hotkeys.py
# ---------------------------------------------------------
# Multithread-Overlay (nur Visualisierung) für TensorRT 10.x
# - Thread 1: Capture (dxcam)  -> in_q
# - Thread 2: Inferenz (TRT)   -> out_q
# - Main:     Zeichnen + Anzeige + sx/sy (Head/Center)
# - Globale Hotkeys (keyboard):
#     8 = ACTIVE an/aus   |  0 = CENTER  |  9 = HEAD  |  c = nearest <-> highest_conf
# - Center = exakte Boxmitte, Head = 31% unter Top-Kante
# ---------------------------------------------------------

import os
import time
import threading
import queue
from typing import Tuple, Optional

import cv2
import dxcam
import numpy as np
import keyboard  # globale Hotkeys

from engine import TRTRunnerV10  # -> deine TRT-Klasse (mit RGB-Preprocess!)
# ---------- Pfade & Parameter ----------
ENGINE_PATH = r"urPath"
IMGSZ       = 256      # zur Engine passend bauen!
ROI_SIZE    = 256      # Anzeige-ROI (zentriert vom Bildschirm)
CONF_THRES  = 0.6
TARGET_FPS  = 70
SHOW_FPS    = True

# Head-Offset: 31% der Boxhöhe unterhalb der oberen Kante
HEAD_ALPHA = 0.20

# ---------- Utils ----------
def center_crop(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.shape[0] != size or roi.shape[1] != size:
        roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_LINEAR)
    return roi

def draw_boxes(roi: np.ndarray, det: np.ndarray, scale_xy: float, conf_thres: float):
    if det is None or det.size == 0:
        return
    for x1, y1, x2, y2, conf, cls in det:
        if conf < conf_thres:
            continue
        x1 = int(x1 * scale_xy); y1 = int(y1 * scale_xy)
        x2 = int(x2 * scale_xy); y2 = int(y2 * scale_xy)
        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(roi, f"{int(cls)}:{conf:.2f}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

def pick_det(det: np.ndarray, rcx: int, rcy: int, mode: str, conf_thres: float) -> Tuple[int, Optional[np.ndarray]]:
    """
    Wähle eine Detection:
      - 'nearest': Box deren Mittelpunkt am nächsten am Crosshair liegt (Conf leichte Gewichtung)
      - 'highest_conf': höchste Confidence
    Rückgabe: (index, row) oder (-1, None)
    """
    if det is None or det.size == 0:
        return -1, None

    m = det[det[:, 4] >= conf_thres]
    if m.size == 0:
        return -1, None

    if mode == "highest_conf":
        idx_local = int(np.argmax(m[:, 4]))
        return idx_local, m[idx_local]

    # nearest: rc auf IMGSZ-Skala umrechnen (det ist in IMGSZ)
    scale_xy = IMGSZ / float(ROI_SIZE)
    rcx_s = rcx * scale_xy
    rcy_s = rcy * scale_xy
    centers_x = (m[:, 0] + m[:, 2]) * 0.5
    centers_y = (m[:, 1] + m[:, 3]) * 0.5
    dx = centers_x - rcx_s
    dy = centers_y - rcy_s
    dist2 = dx * dx + dy * dy
    score = dist2 * (1.0 - 0.05 * m[:, 4])  # Conf bevorzugen, aber Distanz dominiert
    idx_local = int(np.argmin(score))
    return idx_local, m[idx_local]

def compute_offsets_for_modes(box, scale_xy: float, roi_size: int, head_alpha: float):
    """
    Box [x1,y1,x2,y2,conf,cls] (IMGSZ-Skala) -> Punkte:
      - Center = exakte Boxmitte
      - Head   = Top-Center + head_alpha * Boxhöhe
    gibt Offsets (sx,sy) zu ROI-Mitte + Pixel-Koords (im ROI) zurück.
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    # zurück auf ROI-Skala
    x1 *= scale_xy; y1 *= scale_xy; x2 *= scale_xy; y2 *= scale_xy

    h = max(1.0, (y2 - y1))
    rcx, rcy = roi_size * 0.5, roi_size * 0.5

    # CENTER: geometrische Mitte
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    # HEAD: 31% unter Top-Kante
    hx = (x1 + x2) * 0.5
    hy = y1 + head_alpha * h

    sx_center = cx - rcx
    sy_center = cy - rcy
    sx_head   = hx - rcx
    sy_head   = hy - rcy

    return {
        "center": (sx_center, sy_center, (int(cx), int(cy))),
        "head":   (sx_head,   sy_head,   (int(hx), int(hy))),
    }

def main_pipelined():
    in_q  = queue.Queue(maxsize=2)
    out_q = queue.Queue(maxsize=2)
    stop  = threading.Event()

    # --- shared state für Hotkeys ---
    state_lock = threading.Lock()
    active = False
    target_mode = "off"       # 'off' | 'center' | 'head'
    select_mode = "nearest"   # oder 'highest_conf'

    # Hotkey-Callbacks (global)
    def hk_toggle_active():
        nonlocal active, target_mode
        with state_lock:
            active = not active
            if not active:
                target_mode = "off"
        print(f"[ACTIVE] {'ON' if active else 'OFF'}")

    def hk_set_center():
        nonlocal active, target_mode
        with state_lock:
            if not active:
                print("[IGNORED] not active")
                return
            target_mode = "center"
        print("[MODE] CENTER")

    def hk_set_head():
        nonlocal active, target_mode
        with state_lock:
            if not active:
                print("[IGNORED] not active")
                return
            target_mode = "head"
        print("[MODE] HEAD")

    def hk_toggle_select():
        nonlocal select_mode
        with state_lock:
            select_mode = "highest_conf" if select_mode == "nearest" else "nearest"
        print(f"[SELECT] {select_mode}")

    # Hotkeys registrieren
    keyboard.add_hotkey("8", hk_toggle_active)   # global
    keyboard.add_hotkey("0", hk_set_center)      # global
    keyboard.add_hotkey("9", hk_set_head)        # global
    keyboard.add_hotkey("c", hk_toggle_select)   # global

    # --- Capture-Thread ---
    def t_capture():
        cam = dxcam.create(output_idx=0)
        cam.start(target_fps=TARGET_FPS)
        try:
            while not stop.is_set():
                frame = cam.get_latest_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                # BGRA -> BGR  oder RGB -> BGR
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                roi = center_crop(frame, ROI_SIZE)
                try:
                    if in_q.full():
                        _ = in_q.get_nowait()
                    in_q.put_nowait(roi)
                except queue.Full:
                    pass
        finally:
            cam.stop()

    # --- Inferenz-Thread ---
    def t_infer():
        import pycuda.driver as cuda
        cuda.init()
        dev = cuda.Device(0)
        ctx = dev.retain_primary_context()
        ctx.push()
        try:
            runner = TRTRunnerV10(ENGINE_PATH, imgsz=IMGSZ)  # Runner muss RGB-Preprocess haben!
            scale_back = ROI_SIZE / float(IMGSZ)
            while not stop.is_set():
                try:
                    roi = in_q.get(timeout=0.02)
                except queue.Empty:
                    continue
                det = runner.infer(roi)  # (M,6) in IMGSZ-Skala
                try:
                    if out_q.full():
                        _ = out_q.get_nowait()
                    out_q.put_nowait((roi, det, scale_back))
                except queue.Full:
                    pass
        finally:
            ctx.pop()
            ctx.detach()

    th_cap = threading.Thread(target=t_capture, name="Capture", daemon=True)
    th_inf = threading.Thread(target=t_infer,   name="Infer",   daemon=True)
    th_cap.start(); th_inf.start()

    # --- Anzeige / Render ---
    cv2.namedWindow("ROI (Analyse)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI (Analyse)", ROI_SIZE, ROI_SIZE)

    t0 = time.time(); frames = 0; fps_est = 0.0

    try:
        while True:
            try:
                roi, det, scale_back = out_q.get(timeout=0.05)
            except queue.Empty:
                # ESC lokal weiterhin zulassen
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break
                continue

            rcx, rcy = ROI_SIZE // 2, ROI_SIZE // 2

            # aktuellen State thread-sicher lesen
            with state_lock:
                _active = active
                _target_mode = target_mode
                _select_mode = select_mode

            idx, best = pick_det(det, rcx, rcy, _select_mode, CONF_THRES)

            if idx >= 0:
                offsets = compute_offsets_for_modes(best, scale_back, ROI_SIZE, HEAD_ALPHA)

                # Punkte & Offsets
                (sx_c, sy_c, (cx, cy)) = offsets["center"]
                (sx_h, sy_h, (hx, hy)) = offsets["head"]

                if target_mode == "center":
                    move_relative(int(sx_c),int(sy_c))
                if target_mode == "head":
                    move_relative(int(sx_h),int(sy_h))

                # Visuals: Center=gelb, Head=magenta
                cv2.circle(roi, (cx, cy), 3, (0, 255, 255), -1)  # center
                cv2.circle(roi, (hx, hy), 3, (255, 0, 255), -1)  # head

                # aktiven Punkt (wenn active) hervorheben
                if _active:
                    if _target_mode == "center":
                        cv2.drawMarker(roi, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
                    elif _target_mode == "head":
                        cv2.drawMarker(roi, (hx, hy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)

                # sx/sy anzeigen (beide)
                cv2.putText(roi, f"Center sx={sx_c:+.1f} sy={sy_c:+.1f}",
                            (8, ROI_SIZE - 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
                cv2.putText(roi, f"Head   sx={sx_h:+.1f} sy={sy_h:+.1f}  (31%)",
                            (8, ROI_SIZE - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(roi, "No detections", (8, ROI_SIZE - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 1, cv2.LINE_AA)

            # Boxen zeichnen
            draw_boxes(roi, det, scale_xy=scale_back, conf_thres=CONF_THRES)

            # FPS + HUD
            frames += 1
            now = time.time()
            if SHOW_FPS and (now - t0) >= 0.5:
                fps_est = frames / (now - t0)
                frames = 0
                t0 = now
            hud_active, hud_mode, hud_sel = ("ON" if _active else "OFF"), (_target_mode.upper() if _active else "OFF"), _select_mode
            if SHOW_FPS:
                cv2.putText(roi, f"FPS~{fps_est:4.1f}  imgsz={IMGSZ}  Sel:{hud_sel}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # Fadenkreuz + Statuszeile
            cv2.line(roi, (rcx - 6, rcy), (rcx + 6, rcy), (255, 255, 255), 1)
            cv2.line(roi, (rcx, rcy - 6), (rcx, rcy + 6), (255, 255, 255), 1)
            cv2.putText(roi, f"[8] ACTIVE: {hud_active}   Mode: {hud_mode}   (0=CENTER, 9=HEAD, c=toggle select)",
                        (8, ROI_SIZE - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1, cv2.LINE_AA)

            # Anzeige
            cv2.imshow("ROI (Analyse)", roi)

            # zusätzlich ESC lokal
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    finally:
        stop.set()
        for th in (th_cap, th_inf):
            th.join(timeout=1.0)
        # Hotkeys freigeben
        try:
            keyboard.clear_all_hotkeys()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.isfile(ENGINE_PATH):
        raise FileNotFoundError(f"Engine nicht gefunden: {ENGINE_PATH}")
    main_pipelined()
