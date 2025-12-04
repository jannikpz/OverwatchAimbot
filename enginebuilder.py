#python -c "from ultralytics import YOLO; YOLO(r'C:\Users\gtvgp\Desktop\pkt2\OverwatchML\runs\detect\train13b\weights\best.pt').export(format='onnx', imgsz=256, simplify=True, nms=True, dynamic=False)"
#für onnx datei und dann enginebuilder für die engine ggf PATH anpassen
#
# enginebuilder.py  (TensorRT 10.x)
import os
import tensorrt as trt

# --- Pfade anpassen ---
ONNX_PATH   = r"/runs/detect/train13b/weights/batch2.onnx"
ENGINE_PATH = r"/runs/detect/train13b/weights/batch2.engine"
INPUT_NAME  = "images"                # aus deiner ONNX geprüft
INPUT_SHAPE = (2, 3, 256, 256)        # Batch=1, 256x256

def build_engine():
    logger  = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    flags = 0
    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            print("ONNX Parse Error(s):")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # ~3 GB Workspace (anpassen, wenn nötig) == WIe viel platz TENSOR bei dem erstellen von der engine hat
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
    build_engine()

