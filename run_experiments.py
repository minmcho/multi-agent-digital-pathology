#!/usr/bin/env python3
"""
Interactive Menu Script — Multi-Agent Digital Pathology Experiments
Run on your RTX 5070 machine. Guides you through every phase.

Usage:
  python run_experiments.py
"""
import os
import sys
import subprocess
import shutil
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(REPO_DIR, "scripts")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(REPO_DIR, "data"))
FEATURES_DIR = os.environ.get("OUTPUT_DIR", os.path.join(REPO_DIR, "features"))
RESULTS_DIR = os.environ.get("E2E_RESULTS_DIR", os.path.join(REPO_DIR, "e2e_results"))

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header(title):
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"{name} ({mem:.1f} GB)"
        return False, "CUDA not available"
    except ImportError:
        return False, "PyTorch not installed"


def check_data():
    datasets = {
        "CAMELYON17 (WILDS)": os.path.join(DATA_DIR, "camelyon17_wilds"),
        "CAMELYON17 (splits)": os.path.join(DATA_DIR, "camelyon17"),
        "PCam": os.path.join(DATA_DIR, "pcam"),
        "Kather2016": os.path.join(DATA_DIR, "kather2016"),
        "LC25000": os.path.join(DATA_DIR, "lc25000"),
    }
    status = {}
    for name, path in datasets.items():
        if os.path.isdir(path):
            n_files = sum(len(f) for _, _, f in os.walk(path))
            status[name] = f"OK ({n_files} files)" if n_files > 0 else "EMPTY"
        else:
            status[name] = "MISSING"
    return status


def check_packages():
    required = ["torch", "torchvision", "transformers", "timm", "pyarrow",
                "sklearn", "PIL", "numpy", "pandas"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def run_script(script_path, env_extra=None, args=None):
    """Run a script with streaming output."""
    if not os.path.exists(script_path):
        print(f"{RED}Script not found: {script_path}{RESET}")
        input("Press Enter to continue...")
        return

    env = os.environ.copy()
    env.setdefault("DATA_DIR", DATA_DIR)
    env.setdefault("OUTPUT_DIR", FEATURES_DIR)
    env.setdefault("E2E_RESULTS_DIR", RESULTS_DIR)
    if env_extra:
        env.update(env_extra)

    cmd = [sys.executable, "-u", script_path] + (args or [])
    print(f"\n{YELLOW}Running: {' '.join(cmd)}{RESET}\n")

    start = time.time()
    try:
        proc = subprocess.run(cmd, env=env, cwd=REPO_DIR)
        elapsed = time.time() - start
        if proc.returncode == 0:
            print(f"\n{GREEN}Completed in {elapsed / 3600:.1f}h ({elapsed / 60:.0f} min){RESET}")
        else:
            print(f"\n{RED}Failed with exit code {proc.returncode}{RESET}")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted by user{RESET}")

    input(f"\nPress Enter to return to menu...")


def menu_environment():
    header("ENVIRONMENT CHECK")
    ok, gpu = check_gpu()
    status = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"\nGPU:          {status} — {gpu}")

    missing = check_packages()
    if missing:
        print(f"Packages:     {RED}MISSING: {', '.join(missing)}{RESET}")
        print(f"              Install with: pip install {' '.join(missing)}")
    else:
        print(f"Packages:     {GREEN}All required packages installed{RESET}")

    print(f"\n{BOLD}Datasets:{RESET}")
    for name, s in check_data().items():
        color = GREEN if s.startswith("OK") else (YELLOW if s == "EMPTY" else RED)
        print(f"  {name:25s} {color}{s}{RESET}")

    print(f"\n{BOLD}Output directories:{RESET}")
    for name, path in [("Features", FEATURES_DIR), ("E2E results", RESULTS_DIR)]:
        exists = os.path.isdir(path)
        color = GREEN if exists else YELLOW
        n = sum(len(f) for _, _, f in os.walk(path)) if exists else 0
        print(f"  {name:25s} {color}{'exists' if exists else 'not created'} ({n} files){RESET}")

    input("\nPress Enter to return to menu...")


def menu_download_data():
    header("DOWNLOAD DATA")
    print("""
This downloads all datasets to ./data/ (~12 GB total):
  1. CAMELYON17 WILDS  (~9 GB)  — HuggingFace: wltjr1007/Camelyon17-WILDS
  2. PCam              (~1 GB)  — HuggingFace: p1atdev/parquet-imgnet or similar
  3. Kather2016        (~100 MB) — Zenodo: CRC-VAL-HE-7K
  4. LC25000           (~2 GB)  — manual download required (see instructions)
""")
    choice = input("Download which? [1/2/3/all/q]: ").strip().lower()
    if choice == "q":
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if choice in ("1", "all"):
        print(f"\n{YELLOW}Downloading CAMELYON17 WILDS...{RESET}")
        code = ("from huggingface_hub import snapshot_download;"
                "snapshot_download('wltjr1007/Camelyon17-WILDS', repo_type='dataset',"
                f"local_dir='{os.path.join(DATA_DIR, 'camelyon17_wilds')}')")
        subprocess.run([sys.executable, "-c", code])

    if choice in ("2", "all"):
        print(f"\n{YELLOW}Downloading PCam...{RESET}")
        print("PCam parquet files: place them in data/pcam/ manually if this fails.")
        code = ("from huggingface_hub import snapshot_download;"
                "snapshot_download('bryanluk/pcam', repo_type='dataset',"
                f"local_dir='{os.path.join(DATA_DIR, 'pcam')}')")
        subprocess.run([sys.executable, "-c", code])

    if choice in ("3", "all"):
        print(f"\n{YELLOW}Downloading Kather2016...{RESET}")
        url = "https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip"
        subprocess.run(["wget", "-P", DATA_DIR, url])

    input("\nPress Enter to return to menu...")


def menu_feature_extraction():
    header("PHASE 2 — FEATURE EXTRACTION (RTX 5070, ~5h)")
    print("""
Extracts frozen features from foundation models for all datasets.
  1. UNI      (ViT-L, 1024-dim)   ~1.5h
  2. Virchow  (ViT-L, 1024-dim)   ~1.5h
  3. CONCH    (ViT-B, 512-dim)    ~2h  (448x448 input)
  4. All three (~5h total)
""")
    choice = input("Backbone [1/2/3/4/q]: ").strip()
    mapping = {"1": "uni", "2": "virchow", "3": "conch", "4": "all"}
    if choice == "q" or choice not in mapping:
        return
    run_script(os.path.join(SCRIPTS, "rtx5070_feat_extraction.py"),
               args=["--backbone", mapping[choice]])


def menu_conch_e2e():
    header("PHASE 3 — CONCH END-TO-END FINE-TUNING (RTX 5070, ~28h)")
    print("""
Fine-tunes CONCH end-to-end (15 epochs, 5 seeds per model).
  1. Baseline          (~9h)
  2. DANN              (~10h)
  3. MixStyle          (~9h)
  4. All three         (~28h — run overnight)
""")
    choice = input("Model [1/2/3/4/q]: ").strip()
    mapping = {"1": "baseline", "2": "dann", "3": "mixstyle", "4": "all"}
    if choice == "q" or choice not in mapping:
        return
    run_script(os.path.join(SCRIPTS, "rtx5070_conch_e2e.py"),
               args=["--model", mapping[choice], "--seeds", "5"])


def menu_lc25000():
    header("PHASE 4 — LC25000 LUNG/COLON EXPERIMENTS (RTX 5070, ~1.5h)")
    print("""
  1. Extract features only
  2. Frozen-feature experiments only
  3. Both
""")
    choice = input("Mode [1/2/3/q]: ").strip()
    mapping = {"1": "features", "2": "frozen", "3": "all"}
    if choice == "q" or choice not in mapping:
        return
    run_script(os.path.join(SCRIPTS, "lc25000_experiments.py"),
               args=["--mode", mapping[choice]])


def menu_frozen_features():
    header("PHASE 5 — FROZEN-FEATURE EXPERIMENTS (CPU, ~6h)")
    print("""
Runs 5 models x 10 seeds on pre-extracted features.
Requires Phase 2 (feature extraction) to be complete.
  1. UNI
  2. Virchow
  3. CONCH
  4. All
""")
    choice = input("Backbone [1/2/3/4/q]: ").strip()
    mapping = {"1": "uni", "2": "virchow", "3": "conch", "4": "all"}
    if choice == "q" or choice not in mapping:
        return
    run_script(os.path.join(SCRIPTS, "run_frozen_features.py"),
               args=["--backbone", mapping[choice]])


def menu_mil():
    header("PHASE 6 — WSI-LEVEL MIL (CPU, ~4h)")
    print("""
ABMIL + CLAM on WILDS CAMELYON17. Requires MIL features
(extracted by extract_mil_features_cpu.py or rtx5070_feat_extraction.py).
  1. Extract MIL features (GPU, ~1h)
  2. Train MIL models (CPU, ~4h)
""")
    choice = input("Option [1/2/q]: ").strip()
    if choice == "1":
        run_script(os.path.join(REPO_DIR, "extract_mil_features_cpu.py"))
    elif choice == "2":
        run_script(os.path.join(SCRIPTS, "mil_pipeline.py"),
                   args=["--backbone", "all", "--model", "all", "--seeds", "10"])


def menu_package_results():
    header("PACKAGE RESULTS FOR UPLOAD")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(REPO_DIR, f"results_upload_{stamp}.tar.gz")
    print(f"\nCreating {os.path.basename(out)}...")
    subprocess.run([
        "tar", "czf", out,
        "-C", REPO_DIR,
        *(["features"] if os.path.isdir(FEATURES_DIR) else []),
        *(["e2e_results"] if os.path.isdir(RESULTS_DIR) else []),
        *(["frozen_results"] if os.path.isdir(os.path.join(REPO_DIR, "frozen_results")) else []),
        *(["lc25000_results"] if os.path.isdir(os.path.join(REPO_DIR, "lc25000_results")) else []),
        *(["mil_results"] if os.path.isdir(os.path.join(REPO_DIR, "mil_results")) else []),
    ])
    size = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
    print(f"{GREEN}Created {out} ({size:.1f} MB){RESET}")
    print("\nUpload this file to Google Drive and share the link,")
    print("or attach it when you return to the Biomni session.")
    input("\nPress Enter to return to menu...")


def main():
    while True:
        clear()
        ok, gpu = check_gpu()
        gpu_status = f"{GREEN}{gpu}{RESET}" if ok else f"{RED}{gpu}{RESET}"
        print(f"""{BOLD}{CYAN}
╔══════════════════════════════════════════════════════════╗
║   Multi-Agent Digital Pathology — Experiment Runner      ║
╚══════════════════════════════════════════════════════════╝{RESET}
GPU: {gpu_status}

{BOLD}  0.{RESET} Environment check
{BOLD}  1.{RESET} Download data (~12 GB)
{BOLD}  2.{RESET} Feature extraction — UNI/Virchow/CONCH  {YELLOW}(RTX 5070, ~5h){RESET}
{BOLD}  3.{RESET} CONCH end-to-end fine-tuning            {YELLOW}(RTX 5070, ~28h){RESET}
{BOLD}  4.{RESET} LC25000 lung/colon experiments          {YELLOW}(RTX 5070, ~1.5h){RESET}
{BOLD}  5.{RESET} Frozen-feature experiments (new FMs)    {YELLOW}(CPU, ~6h){RESET}
{BOLD}  6.{RESET} WSI-level MIL (ABMIL + CLAM)            {YELLOW}(CPU, ~4h){RESET}
{BOLD}  7.{RESET} Package results for upload
{BOLD}  q.{RESET} Quit
""")
        choice = input(f"{BOLD}Select an option: {RESET}").strip().lower()
        if choice == "0":
            menu_environment()
        elif choice == "1":
            menu_download_data()
        elif choice == "2":
            menu_feature_extraction()
        elif choice == "3":
            menu_conch_e2e()
        elif choice == "4":
            menu_lc25000()
        elif choice == "5":
            menu_frozen_features()
        elif choice == "6":
            menu_mil()
        elif choice == "7":
            menu_package_results()
        elif choice == "q":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
