#für pt datei -> engine (onnx passiert intern), ggf PATH anpassen
#
# enginebuilder.py  (TensorRT 10.x)
import os
import subprocess
import sys

def _ensure_requirements():
    req_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")
    if os.path.isfile(req_path):
        print(f"[INFO] Installiere Requirements aus {req_path} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])

_ensure_requirements()

import tensorrt as trt
from ultralytics import YOLO

# --- Pfade anpassen ---
PT_PATH     = r"urpath"        # trainiertes YOLO-Modell (.pt)
ENGINE_PATH = r"urpath"
INPUT_NAME  = "images"                # aus deiner ONNX geprüft
INPUT_SHAPE = (1, 3, 256, 256)        # Batch=1, 256x256
IMGSZ       = INPUT_SHAPE[-1]

def export_onnx(pt_path: str, imgsz: int) -> str:
    print(f"[INFO] Exportiere {pt_path} -> ONNX (imgsz={imgsz})")
    model = YOLO(pt_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=False)
    print(f"[OK] ONNX exportiert (temporär): {onnx_path}")
    return str(onnx_path)

def build_engine(onnx_path: str):
    logger  = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    flags = 0
    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX Parse Error(s):")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
 
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

    # FP16 falls verfügbar
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Fixes Profil (min=opt=max) für 1x3x256x256
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)

    # Build serialized plan (bytes in HostMemory)
    print("[INFO] Building serialized network (FP16={}, shape={})"
          .format(config.get_flag(trt.BuilderFlag.FP16), INPUT_SHAPE))
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network failed")

    # Speichern
    with open(ENGINE_PATH, "wb") as f:
        f.write(plan)
    print(f"[OK] Engine gespeichert: {ENGINE_PATH}  ({os.path.getsize(ENGINE_PATH)} Bytes)")

    # Optionaler Deserialisierungs-Test
    print("[INFO] Deserialisiere Testweise…")
    with trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(plan)
    print("[OK] Deserialisierung erfolgreich:", engine is not None)

if __name__ == "__main__":
    print("[INFO] TRT Version:", trt.__version__)
    if not os.path.isfile(PT_PATH):
        raise FileNotFoundError(f"PT_PATH nicht gefunden: {PT_PATH}")
    onnx_path = export_onnx(PT_PATH, IMGSZ)
    try:
        build_engine(onnx_path)
    finally:
        os.remove(onnx_path)
        print(f"[INFO] Temporäre ONNX-Datei entfernt: {onnx_path}")

