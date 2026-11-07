#!/usr/bin/env python3
"""
Download and prepare all datasets into the folders the pipeline scripts read.

    python scripts/download_data.py                 # all datasets
    python scripts/download_data.py camelyon17 pcam # selected datasets
    python scripts/download_data.py --check         # only verify what is on disk

Works on Windows, macOS and Linux (pure Python, no wget/unzip needed).
Writes data/MANIFEST.json: source, retrieval date, files, sizes and SHA-256 of
downloaded archives — use it for the manuscript's Data Availability statement.

Layout produced (DATA_DIR defaults to ./data):
  camelyon17/   CAMELYON17-WILDS parquet files (splits: id_train, id_val, ood_test)
  pcam/         PCam parquet files (splits: train, validation, test)
  kather2016/   crc_val_he_7k.parquet  — NORM (label 6) and TUM (label 8) tiles of
                CRC-VAL-HE-7K, converted from the official Zenodo archive
  lc25000/      LC25000 (Kaggle; needs a Kaggle API token)
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1] / "data"))

SOURCES = {
    "camelyon17": {
        "kind": "huggingface",
        "repo": "wltjr1007/Camelyon17-WILDS",
        "expect_splits": ["id_train", "id_val", "ood_test"],
        "citation": "Bandi P, et al. IEEE TMI 2019; Koh PW, et al. WILDS, ICML 2021",
        "license": "CC0 1.0 (CAMELYON17); see dataset card",
        "note": "Third-party Hugging Face mirror of the WILDS patch version; verify "
                "against the official WILDS release before submission.",
    },
    "pcam": {
        "kind": "huggingface",
        "repo": "bryanluk/pcam",
        "expect_splits": ["train", "valid", "test"],
        "citation": "Veeling BS, et al. MICCAI 2018",
        "license": "CC0 1.0 (PCam); see dataset card",
        "note": "Third-party Hugging Face mirror; official source is "
                "https://github.com/basveeling/pcam (Zenodo 2546921).",
    },
    "kather2016": {
        "kind": "zenodo_zip",
        "url": "https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip",
        "classes": {"NORM": 6, "TUM": 8},  # NCT-CRC-HE 9-class index (alphabetical)
        "expected_counts": {"NORM": 741, "TUM": 1233},
        "citation": "Kather JN, et al. PLoS Med 2019;16(1):e1002730 (CRC-VAL-HE-7K, "
                    "Zenodo 1214456)",
        "license": "CC BY 4.0",
        "note": "Official source. NB: this is the 2019 NCT dataset, not the 2016 "
                "Kather texture dataset (Sci Rep 2016).",
    },
    "lc25000": {
        "kind": "kaggle",
        "dataset": "andrewmvd/lung-and-colon-cancer-histopathological-images",
        "citation": "Borkowski AA, et al. arXiv:1912.12142 (2019)",
        "license": "See Kaggle dataset page",
        "note": "Needs a Kaggle API token (~/.kaggle/kaggle.json).",
    },
}


# ------------------------------------------------------------------ helpers

def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def folder_summary(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    return {"n_files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already downloaded: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def split_report(path: Path, splits: list[str]) -> dict:
    parquet = [p.name for p in path.rglob("*.parquet")]
    return {s: sum(s in n for n in parquet) for s in splits}


# ------------------------------------------------------------------ datasets

def get_huggingface(name: str, src: dict) -> dict:
    from huggingface_hub import snapshot_download

    out = DATA_DIR / name
    snapshot_download(src["repo"], repo_type="dataset", local_dir=str(out))
    # The pipeline globs parquet files directly in the folder: flatten if nested.
    for p in list(out.rglob("*.parquet")):
        if p.parent != out:
            target = out / p.name
            if not target.exists():
                shutil.move(str(p), str(target))
    splits = split_report(out, src["expect_splits"])
    missing = [s for s, n in splits.items() if n == 0]
    if missing:
        print(f"  WARNING: no parquet files matching split(s) {missing} in {out}. "
              "Check the mirror's file names before running feature extraction.")
    return {"parquet_files_per_split": splits}


def tiles_to_parquet(zip_path: Path, classes: dict, out_file: Path) -> dict:
    """Convert selected class folders of a tile ZIP into one parquet file with
    columns image ({'bytes': ...}) and label (int), as the pipeline expects."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    images, labels, counts = [], [], {c: 0 for c in classes}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            cls = next((p for p in parts[:-1] if p in classes), None)
            if cls is None or not info.filename.lower().endswith((".tif", ".tiff", ".png", ".jpg")):
                continue
            images.append({"bytes": z.read(info), "path": info.filename})
            labels.append(classes[cls])
            counts[cls] += 1
    out_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"image": images, "label": pa.array(labels, pa.int64())})
    pq.write_table(table, out_file)
    return counts


def get_kather(name: str, src: dict) -> dict:
    out = DATA_DIR / name
    zip_path = download(src["url"], DATA_DIR / "_downloads" / Path(src["url"]).name)
    pq_file = out / "crc_val_he_7k.parquet"
    if pq_file.exists():
        print(f"  already prepared: {pq_file}")
        import pyarrow.parquet as pq

        labels = pq.read_table(pq_file, columns=["label"]).column("label").to_pylist()
        inv = {v: k for k, v in src["classes"].items()}
        counts = {inv[v]: labels.count(v) for v in set(labels)}
    else:
        counts = tiles_to_parquet(zip_path, src["classes"], pq_file)
    for cls, exp in src["expected_counts"].items():
        if counts.get(cls) != exp:
            print(f"  WARNING: {cls} has {counts.get(cls)} tiles, expected {exp}")
    print(f"  {name}: {counts}")
    return {"archive": zip_path.name, "archive_sha256": sha256(zip_path), "tiles": counts}


def get_kaggle(name: str, src: dict) -> dict:
    out = DATA_DIR / name
    if folder_summary(out)["n_files"] > 0:
        print(f"  already present: {out}")
        return {}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(src["dataset"], path=str(out), unzip=True, quiet=False)
    except Exception as e:  # noqa: BLE001 - report and continue
        print(f"  Kaggle download failed ({e}).\n  Manual option: download "
              f"https://www.kaggle.com/datasets/{src['dataset']} and extract to {out}")
    return {}


HANDLERS = {"huggingface": get_huggingface, "zenodo_zip": get_kather, "kaggle": get_kaggle}


# ------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help=f"which datasets (default: all of {list(SOURCES)})")
    p.add_argument("--check", action="store_true", help="only report what is on disk")
    args = p.parse_args()
    names = args.datasets or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        p.error(f"unknown dataset(s) {unknown}; choose from {list(SOURCES)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for name in names:
        src = SOURCES[name]
        print(f"\n== {name} ==")
        entry = {k: v for k, v in src.items() if k not in ("kind",)}
        if not args.check:
            try:
                entry.update(HANDLERS[src["kind"]](name, src))
                entry["retrieved"] = dt.date.today().isoformat()
            except Exception as e:  # noqa: BLE001 - keep going with other datasets
                print(f"  FAILED: {type(e).__name__}: {e}")
                entry["error"] = f"{type(e).__name__}: {e}"
        entry.update(folder_summary(DATA_DIR / name))
        print(f"  on disk: {entry['n_files']} files, {entry['bytes'] / 1e9:.2f} GB")
        manifest[name] = {**manifest.get(name, {}), **entry}

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written: {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
