# overlay_trt_256_trt10.py
# ---------------------------------------------------------
# Analyse-Overlay (nur Visualisierung) für TensorRT 10.x
# - Lädt eine FP16-Engine mit NMS (imgsz=256) direkt (ohne trtexec)
# - DXCam Screen-Grab -> zentriertes 640x640 ROI -> Inferenz 256x256
# - Zeichnet Boxen [x1,y1,x2,y2,conf,cls] zurück ins 640er ROI
# ---------------------------------------------------------

import os
import time
from typing import List

import cv2
import dxcam
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # CUDA-Kontext für PyCUDA

# ---------- Pfade & Parameter ----------
ENGINE_PATH = r"urPath"  # <- anpassen
IMGSZ       = 640      # muss zur Engine passen (imgsz=256 beim Export/Build)
ROI_SIZE    = 640      # sichtbares ROI in der Bildschirmmitte
CONF_THRES  = 0.1
SHOW_FPS    = True

# ---------- Utils ----------
def center_crop(frame_bgr: np.ndarray, size: int = ROI_SIZE) -> np.ndarray:
    """Zentriertes Quadrat size×size ausschneiden. Wenn Rand -> auf size skalieren."""
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    half = size // 2
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.shape[0] != size or roi.shape[1] != size:
        roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_LINEAR)
    return roi

def _np_dtype(dt: trt.DataType):
    return {
        trt.DataType.FLOAT:  np.float32,
        trt.DataType.HALF:   np.float16,
        trt.DataType.INT32:  np.int32,
        trt.DataType.INT8:   np.int8,
        trt.DataType.BOOL:   np.bool_,
    }.get(dt, np.float32)

# ---------- TensorRT 10.x Runner (Tensor-API, enqueue_v3) ----------
class TRTRunnerV10:
    """
    Minimaler TensorRT-Runner für Engines mit NMS im Graph.
    Erwarteter Output-Shape: (1, max_det, 6) -> [x1,y1,x2,y2,conf,cls] in IMGSZ-Koordinaten.
    """
    def __init__(self, engine_path: str, imgsz: int = 256):
        self.imgsz = imgsz

        logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"Engine nicht gefunden: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Konnte Engine nicht deserialisieren (Version/Datei prüfen).")

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # IO-Tensoren über neue API einsammeln
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            (self.inputs if mode == trt.TensorIOMode.INPUT else self.outputs).append(name)
        if not self.inputs or not self.outputs:
            raise RuntimeError("Keine IO-Tensoren gefunden.")

        # Wähle Input/Output
        self.in_name = self.inputs[0]
        self.out_name = None
        for name in self.outputs:
            shp = self.engine.get_tensor_shape(name)
            # YOLO-Ausgang hat meist letzte Dim 6 (x1,y1,x2,y2,conf,cls)
            if len(shp) >= 2 and (shp[-1] == 6 or shp[-1] == -1):
                self.out_name = name
                break
        if self.out_name is None:
            self.out_name = self.outputs[0]

        # Dtypes
        self.in_dtype  = _np_dtype(self.engine.get_tensor_dtype(self.in_name))
        self.out_dtype = _np_dtype(self.engine.get_tensor_dtype(self.out_name))

        # Feste Eingabeform (Batch=1)
        self.context.set_input_shape(self.in_name, (1, 3, self.imgsz, self.imgsz))

        # Output-Shape abfragen (kann dynamisch sein)
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        if any(d < 0 for d in out_shape):
            # Fallback, wenn max_det dynamisch ist
            out_shape = (1, 300, 6)
        self.out_shape = out_shape

        # Host/GPU-Puffer (pagelocked Host für schnelle Transfers)
        self.h_in  = cuda.pagelocked_empty(1 * 3 * self.imgsz * self.imgsz, dtype=self.in_dtype)
        self.h_out = cuda.pagelocked_empty(int(np.prod(self.out_shape)),   dtype=self.out_dtype)
        self.d_in  = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)

        # Tensoradressen einmalig binden
        self.context.set_tensor_address(self.in_name,  int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """Resize -> NCHW -> [0..1] -> dtype (FP16/FP32), Batch=1 flach."""
        if bgr.shape[:2] != (self.imgsz, self.imgsz):
            bgr = cv2.resize(bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        rgb= cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
        if self.in_dtype == np.float16:
            x = x.astype(np.float16, copy=False)
        return x.reshape(-1)

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        # --- Preprocess -> Host (pagelocked) ---
        self.h_in[:] = self._preprocess(bgr)

        # --- H2D ---
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)

        # --- Inferenz: probiere mehrere Namen (je nach TRT-Build) ---
        called = False
        if hasattr(self.context, "enqueue_v3"):
            self.context.enqueue_v3(self.stream.handle)
            called = True
        elif hasattr(self.context, "enqueueV3"):
            self.context.enqueueV3(self.stream.handle)
            called = True
        elif hasattr(self.context, "execute_async_v3"):
            # gleiche Semantik wie enqueue_v3, aber anderer Name
            self.context.execute_async_v3(self.stream.handle)
            called = True
        else:
            # Letzter Fallback: alte API (V2) mit Bindings-Liste
            # Nur verwenden, wenn Engine die Bindings-API anbietet.
            if hasattr(self.engine, "get_binding_index") and hasattr(self.context, "execute_async_v2"):
                try:
                    # Bindings-Liste in Index-Reihenfolge bauen
                    num_bindings = getattr(self.engine, "num_bindings", None)
                    if num_bindings is None:
                        # einige Builds haben num_io_tensors, aber trotzdem Binding-Index-API
                        # Wir verwenden die beiden bekannten Tensors via get_binding_index:
                        bindings = [None] * 2
                    else:
                        bindings = [None] * num_bindings

                    # Eingabe und Ausgabe per Binding-Name/-Index zuweisen
                    in_idx  = self.engine.get_binding_index(self.in_name)
                    out_idx = self.engine.get_binding_index(self.out_name)
                    # Bindings erwartet INT (Adresse)
                    if in_idx is not None:
                        # ggf. Liste groß genug machen
                        if in_idx >= len(bindings):
                            bindings.extend([None] * (in_idx - len(bindings) + 1))
                        bindings[in_idx] = int(self.d_in)
                    if out_idx is not None:
                        if out_idx >= len(bindings):
                            bindings.extend([None] * (out_idx - len(bindings) + 1))
                        bindings[out_idx] = int(self.d_out)

                    # Für dynamische Shapes sicherstellen:
                    if hasattr(self.context, "set_binding_shape"):
                        self.context.set_binding_shape(in_idx, (1, 3, self.imgsz, self.imgsz))

                    # ausführen
                    self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                    called = True
                except Exception as e:
                    # Debug-Hilfe: zeig verfügbare Methoden an
                    avail = [m for m in dir(self.context) if ("enqueue" in m or "execute" in m)]
                    raise RuntimeError(f"Kein passender Inferenz-Call gefunden. context-methods={avail}\nGrund: {e}")
            else:
                avail = [m for m in dir(self.context) if ("enqueue" in m or "execute" in m)]
                raise RuntimeError(f"TensorRT ExecutionContext hat keinen enqueue_v3/enqueueV3/execute_async_v3 "
                                   f"und V2-Fallback ist nicht möglich. Verfügbar: {avail}")

        # --- D2H ---
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()

        # --- Postprocess ---
        out = np.array(self.h_out, dtype=np.float32).reshape(self.out_shape)  # (1, M, 6)
        det = out[0]
        det = det[det[:, 4] > 0.0]
        return det




# ---------- Zeichnen ----------
def draw_boxes(roi: np.ndarray, det: np.ndarray, scale_xy: float, conf_thres: float = CONF_THRES):
    if det is None or det.size == 0:
        return
    for x1, y1, x2, y2, conf, cls in det:
        if conf < conf_thres:
            continue
        x1 = int(x1 * scale_xy); y1 = int(y1 * scale_xy)
        x2 = int(x2 * scale_xy); y2 = int(y2 * scale_xy)
        cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(roi, f"{int(cls)}:{conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

def main():
    print("[INFO] Lade TensorRT-Engine…")
    runner = TRTRunnerV10(ENGINE_PATH, imgsz=IMGSZ)
    scale_back = ROI_SIZE / float(IMGSZ)

    cam = dxcam.create(output_idx=0)
    cam.start(target_fps=120)

    cv2.namedWindow("ROI (Analyse 640x640)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI (Analyse 640x640)", ROI_SIZE, ROI_SIZE)

    t0 = time.time()
    frames = 0
    fps_est = 0.0

    print("[INFO] ESC zum Beenden.")
    while True:
        frame = cam.get_latest_frame()
        if frame is None:
            if cv2.waitKey(1) == 27:
                break
            continue

        # BGRA -> BGR (Alpha droppen) ODER RGB -> BGR
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        roi = center_crop(frame, ROI_SIZE)

        det = runner.infer(roi)                    # arbeitet intern auf 256×256
        draw_boxes(roi, det, scale_xy=scale_back)  # skaliert zurück auf 640×640

        # HUD
        frames += 1
        now = time.time()
        if SHOW_FPS and (now - t0) >= 0.5:
            fps_est = frames / (now - t0)
            frames = 0
            t0 = now
        if SHOW_FPS:
            cv2.putText(roi, f"FPS~{fps_est:4.1f}  imgsz={IMGSZ}  engine=TRT10",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        # Fadenkreuz
        rcx, rcy = ROI_SIZE // 2, ROI_SIZE // 2
        cv2.line(roi, (rcx - 6, rcy), (rcx + 6, rcy), (255, 255, 255), 1)
        cv2.line(roi, (rcx, rcy - 6), (rcx, rcy + 6), (255, 255, 255), 1)

        cv2.imshow("ROI (Analyse 640x640)", roi)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Kurzer Existenz-Check, damit Pfadprobleme sofort sichtbar sind
    print("[DEBUG] ENGINE_PATH =", ENGINE_PATH, "| exists:", os.path.exists(ENGINE_PATH))
    main()
