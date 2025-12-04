import cv2
import dxcam
import numpy as np
from collections import deque
from ultralytics import YOLO
import time

# ================== Einstellungen ==================
MODEL_PATH = r"C:\Users\gtvgp\Desktop\pkt2\OverwatchML\runs\detect\train13b\weights\best.pt"
ROI_SIZE   = 640              # 640x640 aus der Bildschirmmitte
CONF_THRES = 0.25             # Konfidenz-Schwelle
IOU_THRES  = 0.45             # NMS IOU (nur falls kein GPU-NMS in deinem .pt)
IMGSZ      = 640              # Inferenz-Bildgröße (an dein Training anpassen)
USE_GPU    = True             # wenn CUDA vorhanden -> True
USE_FP16   = True             # FP16 (half) für schnellere Inferenz auf GPU
BATCH_SIZE = 1               # 1 = keine Batch-Sammlung (min. Latenz). >1 = Frames sammeln und gemeinsam inferieren
SHOW_FPS   = True

# ================== Hilfen ==================
def center_crop_640(frame_bgr, size=ROI_SIZE):
    """Schneidet ein zentriertes Quadrat size×size aus. Clamped an Bildränder."""
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)

    roi = frame_bgr[y1:y2, x1:x2]
    # falls am Rand kleiner geworden -> auf exakt size×size resizen (letterbox vermeiden, hier einfach Resize)
    if roi.shape[0] != size or roi.shape[1] != size:
        roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_LINEAR)
    return roi

def draw_boxes(roi, yolo_result):
    """
    Zeichnet xyxy-Boxes + Konf auf roi (BGR).
    yolo_result: ein einzelnes Result-Objekt von Ultralytics.
    """
    if yolo_result is None or yolo_result.boxes is None:
        return roi

    boxes = yolo_result.boxes
    if boxes.xyxy is None:
        return roi

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((xyxy.shape[0],), dtype=np.float32)
    cls  = boxes.cls.cpu().numpy()  if boxes.cls is not None  else np.zeros((xyxy.shape[0],), dtype=np.float32)

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        c = float(conf[i]) if i < len(conf) else 0.0
        label = f"{int(cls[i]) if i < len(cls) else -1}:{c:.2f}"
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(roi, p1, p2, (0, 255, 0), 2)
        cv2.putText(roi, label, (p1[0], max(15, p1[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return roi

# ================== Setup ==================
print("[INFO] Lade YOLO Modell…")
model = YOLO(MODEL_PATH)

device = 0 if USE_GPU else "cpu"
half   = USE_FP16 and USE_GPU

# Kamera (Screenscraper)
cam = dxcam.create(output_idx=0)  # ggf. Monitor-Index anpassen
cam.start(target_fps=120)

cv2.namedWindow("ROI 640x640", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI 640x640", ROI_SIZE, ROI_SIZE)

# Batch-Puffer
frames_buf = deque()
times_buf  = deque()  # für FPS-Messung

last_infer_time = time.time()
frames_shown = 0
t0_fps = time.time()

print("[INFO] ESC zum Beenden.")
while True:
    frame = cam.get_latest_frame()
    if frame is None:
        if cv2.waitKey(1) == 27:
            break
        continue

    # dxcam liefert RGB/BGRA -> sicherheitshalber zu BGR konvertieren
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    roi = center_crop_640(frame, ROI_SIZE)

    # ---- Batching-Logik ----
    if BATCH_SIZE <= 1:
        # direkte Inferenz (niedrigste Latenz)
        results = model.predict(
            source=[roi],              # Liste mit einem Frame
            imgsz=IMGSZ,
            conf=CONF_THRES,
            iou=IOU_THRES,
            device=device,
            half=half,
            verbose=False
        )
        yres = results[0] if len(results) else None
        out = draw_boxes(roi, yres)

    else:
        # Frames sammeln, dann gemeinsam inferieren
        frames_buf.append(roi.copy())
        if len(frames_buf) >= BATCH_SIZE:
            batch = list(frames_buf)
            frames_buf.clear()

            results = model.predict(
                source=batch,          # Liste von BGR-Frames
                imgsz=IMGSZ,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device=device,
                half=half,
                verbose=False
            )

            # Beim Anzeigen nehmen wir das letzte (aktuellste) Result,
            # alternativ könntest du auch alle nacheinander zeigen.
            yres_latest = results[-1] if len(results) else None
            out = draw_boxes(roi, yres_latest)
        else:
            # noch nicht genug gesammelt -> zeige einfach aktuellen ROI ohne neue Inferenz
            out = roi

    # ---- HUD ----
    if SHOW_FPS:
        now = time.time()
        frames_shown += 1
        dt = now - t0_fps
        if dt >= 0.5:
            fps = frames_shown / dt
            frames_shown = 0
            t0_fps = now
            cv2.putText(out, f"FPS ~ {fps:5.1f}  (batch={BATCH_SIZE}, half={half}, dev={device})",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("ROI 640x640", out)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC
        break

cv2.destroyAllWindows()
