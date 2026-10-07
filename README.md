Realtime inference with a YOLO model, accelerated by a TensorRT engine that is built for your GPU.

## Setup

1. **Create a virtual environment** with Python 3.11:
   ```
   py -3.11 -m venv .venv
   ```
2. **Activate it:**
   ```
   .venv\Scripts\Activate.ps1
   ```
3. **Upgrade pip:**
   ```
   python -m pip install --upgrade pip
   ```
4. **Install the dependencies:**
   ```
   pip install -r requirements.txt
   ```
5. **Build the TensorRT engine** (first run may take a while):
   ```
   python tools/enginebuilderV2.py
   ```
6. **Run the program:**
   ```
   python runtime/engineThreads.py
   ```

> **Requirements:** NVIDIA GPU with at least CUDA 12.x and 3 GiB of VRAM.