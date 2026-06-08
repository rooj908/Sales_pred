"""
setup.py — One-click setup: generate data, train models, launch app
Run: python setup.py
"""
import subprocess, sys, os

def run(cmd, cwd=None):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[WARNING] Command returned non-zero: {cmd}")

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/saved", exist_ok=True)

    print("=" * 60)
    print("  AI-Powered BI & Sales Forecasting Platform — Setup")
    print("=" * 60)

    print("\n[1/3] Generating synthetic data...")
    run(f"{sys.executable} data/generate_data.py")

    print("\n[2/3] Preprocessing data...")
    run(f"{sys.executable} utils/preprocessing.py")

    print("\n[3/3] Training ML models...")
    run(f"{sys.executable} models/train_models.py")

    print("\n" + "=" * 60)
    print("  Setup complete! To launch the platform:")
    print()
    print("  Dashboard:  streamlit run dashboard/app.py")
    print("  API Server: uvicorn api.main:app --reload --port 8000")
    print("  API Docs:   http://localhost:8000/docs")
    print("=" * 60)
