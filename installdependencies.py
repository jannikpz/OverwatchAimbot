# installdependencies.py
# Installiert alle Pakete aus requirements.txt in der aktiven Umgebung.
import os
import subprocess
import sys

def main():
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if not os.path.isfile(req_path):
        raise FileNotFoundError(f"requirements.txt nicht gefunden: {req_path}")
    print(f"[INFO] Installiere Requirements aus {req_path} …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("[OK] Fertig.")

if __name__ == "__main__":
    main()
