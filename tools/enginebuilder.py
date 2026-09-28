import os
import tensorrt as trt
from ultralytics import YOLO

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PT_PATH     = os.path.join(REPO_ROOT, "model", "best.pt")
ENGINE_PATH = os.path.join(REPO_ROOT, "model", "best.engine")
INPUT_SHAPE = (1, 3, 256, 256)  #format (batch,rgb,256x256 pixels)
IMGSZ       = INPUT_SHAPE[-1]

def export_onnx(pt_path: str, imgsz: int) -> str: #path to best.pt & picture size
    print(f"[INFO] Exporting {pt_path} -> ONNX (imgsz={imgsz})")
    model = YOLO(pt_path)
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=False) #onnx
    print(f"[OK] ONNX exported (temporary): {onnx_path}")
    return str(onnx_path)  # return path to onnx

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

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30) # memory for workspace 3 * 2^30 Byte = 3GiB

    # switch to floatingpoint 16  (from 32) if available
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # fixed profile for 1x3x256x256
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
    config.add_optimization_profile(profile)


    print("[INFO] Building serialized network (FP16={}, shape={}) -- this can take a few minutes..."
          .format(config.get_flag(trt.BuilderFlag.FP16), INPUT_SHAPE))
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network failed")

    # save
    with open(ENGINE_PATH, "wb") as f:
        f.write(plan)
    print(f"[OK] Engine saved: {ENGINE_PATH}  ({os.path.getsize(ENGINE_PATH)} bytes)")

    # optional deserialization test
    print("[INFO] Test-deserializing...")
    with trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(plan)
    print("[OK] Deserialization successful:", engine is not None)

if __name__ == "__main__":
    print("[INFO] TRT version:", trt.__version__)
    if not os.path.isfile(PT_PATH):
        raise FileNotFoundError(f"PT_PATH not found: {PT_PATH}")
    onnx_path = export_onnx(PT_PATH, IMGSZ)
    #build engine
    try:
        build_engine(onnx_path)
    # if it fails remove onnx

    finally:
        os.remove(onnx_path)
        print(f"[INFO] Removed temporary ONNX file: {onnx_path}")

