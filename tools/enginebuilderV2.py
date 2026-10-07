import os
import subprocess
import sys

import tensorrt as trt

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(REPO_ROOT, "model")
PT_PATH     = os.path.join(MODEL_DIR, "best.pt")
ONNX_PATH   = os.path.join(MODEL_DIR, "best.onnx")
ENGINE_PATH = os.path.join(MODEL_DIR, "best.engine")

INPUT_SHAPE = (1, 3, 256, 256)   # (batch, rgb, h, w)
INPUT_NAME  = "images"
IMGSZ       = INPUT_SHAPE[-1]


def export_onnx(pt_path: str, imgsz: int) -> str:
    from ultralytics import YOLO
    print(f"[INFO] Exporting {pt_path} -> ONNX (imgsz={imgsz})", flush=True)
    model = YOLO(pt_path)
    # Ohne NMS: Ausgabe (1, 6, 1344). Die NMS passiert im Runner.
    onnx_path = str(model.export(format="onnx", imgsz=imgsz, dynamic=False,
                                 opset=17, simplify=False))
    print(f"[OK] ONNX: {onnx_path}", flush=True)
    return onnx_path


def build_engine(onnx_path: str, engine_path: str):
    logger  = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    print("[OK] ONNX geparst", flush=True)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)

    print(f"[INFO] Baue Engine (FP16={config.get_flag(trt.BuilderFlag.FP16)}), "
          "das dauert einige Minuten...", flush=True)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network failed")

    tmp_path = engine_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(plan)
    os.replace(tmp_path, engine_path)      # Datei erst komplett, dann umbenannt
    print(f"[OK] Engine gespeichert: {engine_path} ({os.path.getsize(engine_path)} bytes)", flush=True)

    with trt.Runtime(logger) as rt:
        ok = rt.deserialize_cuda_engine(plan) is not None
    print("[OK] Deserialisierung erfolgreich:", ok, flush=True)


def _build_once():
    import pycuda.autoinit            # CUDA vor TensorRT initialisieren
    import pycuda.driver as cuda
    print("[INFO] GPU:", cuda.Device(0).name(), flush=True)
    print("[INFO] TRT version:", trt.__version__, flush=True)

    if not os.path.isfile(PT_PATH):
        raise FileNotFoundError(f"best.pt nicht gefunden: {PT_PATH}")
    onnx_path = ONNX_PATH if os.path.isfile(ONNX_PATH) else export_onnx(PT_PATH, IMGSZ)
    build_engine(onnx_path, ENGINE_PATH)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _build_once()
    else:
        for attempt in range(1, 4):
            print(f"[INFO] Build-Versuch {attempt}/3", flush=True)
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child"])
            if r.returncode == 0 and os.path.isfile(ENGINE_PATH):
                print("[DONE] Engine fertig.")
                break
        else:
            raise SystemExit("Engine-Build fehlgeschlagen")