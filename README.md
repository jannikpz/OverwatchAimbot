# OverwatchAimbot (YOLO + TensorRT)

## Requirements

- **Python 3.11** (tested with 3.11.5)
- NVIDIA GPU + current driver (TensorRT/PyCUDA do not run on AMD/Intel)
- NVIDIA CUDA Toolkit installed beforehand (needed to build `pycuda` during `pip install`)

## Structure

- `model/best.pt` – trained YOLO model (nano).
- `builder/enginebuilder.py` – builds a TensorRT `.engine` file (`model/best.engine`) from `model/best.pt` (ONNX export happens internally).
- `runtime/engine.py` – `TRTRunnerV10`, loads the `.engine` and runs inference.
- `runtime/engineThreads2.py` – actual entry point: capture + inference + aiming.
- `requirements.txt` / `installdependencies.py` – Python dependencies.

## Setup

1. Install `tensorrt` manually (needs a matching NVIDIA CUDA Toolkit already installed to build/link against):
   ```bash
   pip install tensorrt
   ```

2. Install the remaining Python dependencies (includes `pycuda`, which compiles locally against your CUDA Toolkit and can take several minutes):
   ```bash
   python installdependencies.py
   ```

3. Build the engine (automatically reads `model/best.pt`, writes `model/best.engine`). This step can also take a few minutes, TensorRT benchmarks multiple kernel implementations while building:
   ```bash
   python builder/enginebuilder.py
   ```

4. Run (`ENGINE_PATH` is resolved automatically to `model/best.engine`):
   ```bash
   python runtime/engineThreads2.py
   ```

## Hotkeys (runtime/engineThreads2.py)

- `8` – toggle trigger: OFF ↔ HOLD
- `0` – hold to aim (only active when trigger = HOLD)
- `9` – quit
