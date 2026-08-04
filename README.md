# OverwatchAimbot (YOLO + TensorRT)

## Struktur

- `model/best.pt` – trainiertes YOLO-Modell (nano).
- `builder/enginebuilder.py` – baut aus `model/best.pt` automatisch (intern über ONNX) eine TensorRT-`.engine`-Datei.
- `runtime/engine.py` – `TRTRunnerV10`, lädt die `.engine` und macht die Inferenz.
- `runtime/engineThreads2.py` – eigentlicher Einstiegspunkt: Capture + Inferenz + Aiming.
- `requirements.txt` / `installdependencies.py` – Python-Abhängigkeiten.

## Setup

1. Python-Abhängigkeiten installieren:
   ```bash
   python installdependencies.py
   ```
   `tensorrt` und `pycuda` brauchen zusätzlich eine passende NVIDIA-CUDA-Installation (nur auf NVIDIA-GPUs lauffähig).

2. Engine bauen (liest automatisch `model/best.pt`, `ENGINE_PATH` in [builder/enginebuilder.py](builder/enginebuilder.py) anpassen):
   ```bash
   python builder/enginebuilder.py
   ```

3. `ENGINE_PATH` in [runtime/engineThreads2.py](runtime/engineThreads2.py) auf denselben Pfad wie oben setzen, dann starten:
   ```bash
   python runtime/engineThreads2.py
   ```

## Hotkeys (runtime/engineThreads2.py)

- `8` – Trigger-Mode durchschalten: OFF → HOLD → TOGGLE
- `9` – HOLD: Head-Ziel solange gehalten | TOGGLE: Head-Ziel an/aus
- `0` – HOLD: Center-Ziel solange gehalten | TOGGLE: Center-Ziel an/aus
- `!` – Auswahlmodus wechseln (nearest / highest_conf)
- `ESC` – Beenden
