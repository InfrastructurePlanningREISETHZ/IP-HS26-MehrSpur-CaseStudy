"""verify_setup.py — Run this to check if your environment and data are ready."""
import sys
from pathlib import Path

# Force immediate console output on Windows
sys.stdout.reconfigure(encoding='utf-8')

print("--- Checking environment for Infrastructure Planning HS26 ---", flush=True)

# 1. Python version check
if sys.version_info < (3, 10):
    print(f"[FAIL] Python 3.10+ required. You are using {sys.version.split()[0]}", flush=True)
else:
    print(f"[OK] Python version: {sys.version.split()[0]}", flush=True)

# 2. Key library check
packages = ["numpy", "pandas", "geopandas", "shapely", "matplotlib", "seaborn", "ema_workbench"]
missing = []

print("Checking installed packages (this may take a few seconds)...", flush=True)
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}", flush=True)
    except ImportError:
        print(f"  [FAIL] {pkg} (NOT FOUND)", flush=True)
        missing.append(pkg)

if missing:
    print(f"\n[FAIL] Missing packages: {', '.join(missing)}")
    print("       Run: pip install -r requirements.txt\n")
else:
    print("[OK] All core packages imported successfully.\n", flush=True)

# 3. Data & Model check
# Automatically finds repo root whether run from root or inside code/
script_dir = Path(__file__).resolve().parent
root = script_dir if (script_dir / "code").exists() else script_dir.parent

checks = [
    root / "code" / "parameters.py",
    root / "data" / "processed" / "mehrspur_detailed_network.pkl",
    root / "IP_course_FSM-main" / "input_data" / "prepared" / "skims.pkl.gz",
    root / "notebooks" / "phase2_system_modeling.ipynb"
]

all_files_exist = True
print("Checking course files...", flush=True)
for path in checks:
    if path.exists():
        print(f"  [OK] Found: {path.relative_to(root)}", flush=True)
    else:
        print(f"  [FAIL] Missing: {path.relative_to(root)}", flush=True)
        all_files_exist = False

if all_files_exist and not missing:
    print("\nSUCCESS: Everything is set up correctly! You are ready for Phase 2.", flush=True)
else:
    print("\nWARNING: Some requirements or files are missing. Check above.", flush=True)
