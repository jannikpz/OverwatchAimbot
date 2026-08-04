# installdependencies.py
# Installs all packages from requirements.txt into the active environment.
import os
import subprocess
import sys

def main():
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    if not os.path.isfile(req_path):
        raise FileNotFoundError(f"requirements.txt not found: {req_path}")
    print(f"[INFO] Installing requirements from {req_path} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("[OK] Done.")

if __name__ == "__main__":
    main()
