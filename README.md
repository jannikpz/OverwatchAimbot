# OverwatchAimbot (YOLO + TensorRT)

## Voraussetzungen

- **Python 3.11** (getestet mit 3.11.5)
- NVIDIA-GPU + aktueller Treiber (TensorRT/PyCUDA laufen nicht auf AMD/Intel)

## Struktur

- `model/best.pt` – trainiertes YOLO-Modell (nano).
- `builder/enginebuilder.py` – baut aus `model/best.pt` automatisch (intern über ONNX) eine TensorRT-`.engine`-Datei (`model/best.engine`).
- `runtime/engine.py` – `TRTRunnerV10`, lädt die `.engine` und macht die Inferenz.
- `runtime/engineThreads2.py` – eigentlicher Einstiegspunkt: Capture + Inferenz + Aiming.
- `requirements.txt` / `installdependencies.py` – Python-Abhängigkeiten.

## Setup

1. `tensorrt` und `pycuda` manuell installieren (brauchen ein passendes, vorher installiertes NVIDIA CUDA Toolkit zum Bauen/Verlinken):
   ```bash
   pip install tensorrt pycuda
   ```

2. Restliche Python-Abhängigkeiten installieren:
   ```bash
   python installdependencies.py
   ```

3. Engine bauen (liest automatisch `model/best.pt`, schreibt `model/best.engine`):
   ```bash
   python builder/enginebuilder.py
   ```

4. Starten (`ENGINE_PATH` wird automatisch auf `model/best.engine` aufgelöst):
   ```bash
   python runtime/engineThreads2.py
   ```

## Hotkeys (runtime/engineThreads2.py)

- `8` – Trigger umschalten: OFF ↔ HOLD
- `0` – halten für Aim (nur wirksam wenn Trigger = HOLD)
- `9` – Beenden
