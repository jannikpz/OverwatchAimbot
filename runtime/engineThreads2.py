

# engineThreads2.py
# ---------------------------------------------------------
# Multithread-Overlay (nur Visualisierung) für TensorRT 10.x
# - Thread 1: Capture (dxcam)  -> in_q
# - Thread 2: Inferenz (TRT)   -> out_q
# - Main:     Anzeige + grüne Boxen + Pfeil (Crosshair→Center) + HUD/FPS
# - Hotkeys: 8 = Trigger OFF/HOLD, 0 = halten für Aim, 9 = Beenden
# ---------------------------------------------------------

import ctypes
from ctypes import wintypes
import os
import time
import threading
import queue
from typing import Tuple, Optional

import cv2
import dxcam
import numpy as np
import keyboard  # globale Hotkeys

from engine import TRTRunnerV10  # deine TRT-Klasse (mit RGB-Preprocess!)

##### Windows API
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

# ---------- Pfade & Parameter ----------
ENGINE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "best.engine")
IMGSZ       = 256      # zur Engine passend builden
ROI_SIZE    = 256      # sichtbares ROI (zentriert)
CONF_THRES  = 0.6
TARGET_FPS  = 70
SHOW_FPS    = True

cv2.setNumThreads(1)

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

def draw_boxes(roi: np.ndarray, det: np.ndarray, scale_xy: float, conf_thres: float, highlight_idx: int = -1):
    """
    Zeichnet grüne Boxen für alle Detections (CONF >= conf_thres).
    highlight_idx referenziert die gefilterte Liste m (nicht det) und wird dicker gezeichnet.
    """
    if det is None or det.size == 0:
        return
    m = det[det[:, 4] >= conf_thres]
    if m.size == 0:
        return
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(m):
        x1 = int(x1 * scale_xy); y1 = int(y1 * scale_xy)
        x2 = int(x2 * scale_xy); y2 = int(y2 * scale_xy)
        thickness = 2 if i == highlight_idx else 1
        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), thickness)

def pick_det(det: np.ndarray, rcx: int, rcy: int, conf_thres: float) -> Tuple[int, Optional[np.ndarray], int]:
    """
    Wähle die Box, deren Mittelpunkt am nächsten am Crosshair liegt (Conf leicht gewichtet).
    Rückgabe: (idx_in_filtered, row, idx_in_filtered) oder (-1, None, -1)
    idx_in_filtered bezieht sich auf m = det[conf>=thres] (für highlight).
    """
    if det is None or det.size == 0:
        return -1, None, -1

    m = det[det[:, 4] >= conf_thres]
    if m.size == 0:
        return -1, None, -1

    # rc auf IMGSZ-Skala umrechnen (det ist in IMGSZ)
    scale_xy = IMGSZ / float(ROI_SIZE)
    rcx_s = rcx * scale_xy
    rcy_s = rcy * scale_xy
    centers_x = (m[:, 0] + m[:, 2]) * 0.5
    centers_y = (m[:, 1] + m[:, 3]) * 0.5
    dx = centers_x - rcx_s
    dy = centers_y - rcy_s
    dist2 = dx * dx + dy * dy
    score = dist2 * (1.0 - 0.05 * m[:, 4])  # Conf bevorzugt, aber Distanz dominiert
    i_f = int(np.argmin(score))
    return i_f, m[i_f], i_f

def compute_center_offset(box, scale_xy: float, roi_size: int):
    """
    Box [x1,y1,x2,y2,conf,cls] (IMGSZ-Skala) -> Boxmitte.
    gibt Offset (sx,sy) zu ROI-Mitte + Pixel-Koords (im ROI) zurück.
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    # zurück auf ROI-Skala
    x1 *= scale_xy; y1 *= scale_xy; x2 *= scale_xy; y2 *= scale_xy

    rcx, rcy = roi_size * 0.5, roi_size * 0.5

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    return cx - rcx, cy - rcy, (int(cx), int(cy))

def main_pipelined():
    in_q  = queue.Queue(maxsize=2)
    out_q = queue.Queue(maxsize=2)
    stop  = threading.Event()

    # --- shared state / hotkeys ---
    state_lock = threading.Lock()
    quit_event = threading.Event()

    # Trigger via 8: OFF <-> HOLD
    TRIGGER_OFF, TRIGGER_HOLD = 0, 1
    trigger_mode = TRIGGER_OFF

    # Hotkeys:
    def hk_toggle_trigger():
        nonlocal trigger_mode
        trigger_mode = TRIGGER_HOLD if trigger_mode == TRIGGER_OFF else TRIGGER_OFF
        print(f"[TriggerMode] -> {['OFF','HOLD'][trigger_mode]}")

    def hk_quit():
        quit_event.set()

    keyboard.add_hotkey("8", hk_toggle_trigger)
    keyboard.add_hotkey("9", hk_quit)

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
            runner = TRTRunnerV10(ENGINE_PATH, imgsz=IMGSZ)
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
        while not quit_event.is_set():
            try:
                roi, det, scale_back = out_q.get(timeout=0.05)
            except queue.Empty:
                cv2.waitKey(1)
                continue

            rcx, rcy = ROI_SIZE // 2, ROI_SIZE // 2

            with state_lock:
                _trigger_mode = trigger_mode

            # Sub-Mode ermitteln aus TriggerMode + 0
            if _trigger_mode == TRIGGER_HOLD and keyboard.is_pressed("0"):
                mode = "center"
            else:
                mode = "off"

            # Beste Detection für Visualisierung + Pfeile
            idx_f, best, highlight_idx = pick_det(det, rcx, rcy, CONF_THRES)

            # Immer: grüne Boxen (Highlight für gewählte Box)
            draw_boxes(roi, det, scale_back, CONF_THRES, highlight_idx=highlight_idx)

            # Pfeil/Marker nur, wenn aktiv & Box vorhanden
            if idx_f >= 0 and best is not None and mode == "center":
                sx_c, sy_c, (cx, cy) = compute_center_offset(best, scale_back, ROI_SIZE)

                move_relative(int(sx_c), int(sy_c))

                cv2.arrowedLine(roi, (rcx, rcy), (int(cx), int(cy)), (0, 255, 255), 1, tipLength=0.25)  # gelb
                cv2.circle(roi, (int(cx), int(cy)), 2, (0, 255, 255), -1)

            # FPS + HUD
            frames += 1
            now = time.time()
            if SHOW_FPS and (now - t0) >= 0.5:
                fps_est = frames / (now - t0)
                frames = 0
                t0 = now
            if SHOW_FPS:
                trig_name = ['OFF','HOLD'][_trigger_mode]
                line1 = f"FPS~{fps_est:4.1f}  imgsz={IMGSZ}"
                line2 = f"Trig:{trig_name}  Mode:{mode.upper()}"

                cv2.putText(roi, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(roi, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200,200,255), 1, cv2.LINE_AA)


    # Fadenkreuz
            cv2.line(roi, (rcx - 6, rcy), (rcx + 6, rcy), (255, 255, 255), 1)
            cv2.line(roi, (rcx, rcy - 6), (rcx, rcy + 6), (255, 255, 255), 1)

            cv2.imshow("ROI (Analyse)", roi)
            cv2.waitKey(1)

    finally:
        stop.set()
        th_cap.join(timeout=1.0)
        th_inf.join(timeout=1.0)
        try:
            keyboard.clear_all_hotkeys()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.isfile(ENGINE_PATH):
        raise FileNotFoundError(f"Engine nicht gefunden: {ENGINE_PATH}")
    main_pipelined()

