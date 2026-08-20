"""
Drishti-Kavach: Kaggle Dataset Packager

Packages `dataset_segmentation/` into a single zip archive for fast upload to Kaggle.

Usage:
  python src/3_kaggle_train/package_for_kaggle.py
"""

import os
import sys
import glob
import zipfile
from pathlib import Path
from tqdm import tqdm


def package_segmentation_dataset(
    dataset_dir: str = "dataset_segmentation",
    output_zip: str = "src/3_kaggle_train/raildrishti_segmentation_kaggle.zip"
):
    ds_path = Path(dataset_dir)
    out_zip_path = Path(output_zip)
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not ds_path.exists():
        print(f"[!] Error: {ds_path} not found.")
        return

    # Gather all files in dataset_segmentation
    all_files = []
    for root, _, files in os.walk(ds_path):
        for f in files:
            if not f.startswith(".") and not f.endswith(".tmp"):
                all_files.append(os.path.join(root, f))

    print("=" * 70)
    print(" 📦 PACKAGING RAILSEM19 SEGMENTATION DATASET FOR KAGGLE")
    print("=" * 70)
    print(f" • Source Folder: {ds_path.resolve()}")
    print(f" • Total Files:   {len(all_files):,}")
    print(f" • Output Zip:    {out_zip_path.resolve()}")
    print("=" * 70)

    # Create Zip Archive with progress bar
    with zipfile.ZipFile(str(out_zip_path), "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for file_path in tqdm(all_files, desc="Compressing dataset"):
            rel_path = os.path.relpath(file_path, start=ds_path.parent)
            zf.write(file_path, arcname=rel_path)

    zip_size_mb = os.path.getsize(str(out_zip_path)) / (1024 * 1024)
    print("=" * 70)
    print(f" ✅ PACKAGING COMPLETE!")
    print(f" • Zip Archive:   {out_zip_path.name}")
    print(f" • Size on Disk:  {zip_size_mb:.2f} MB (~{zip_size_mb/1024:.2f} GB)")
    print("=" * 70)
    print("\n📋 Next Steps to Train on Kaggle:")
    print("  1. Open https://www.kaggle.com/datasets -> Click '+ New Dataset'")
    print(f"  2. Upload: '{out_zip_path.resolve()}'")
    print("  3. Set Dataset Title to 'raildrishti-segmentation' and click 'Create'.")
    print("  4. Open 'src/3_kaggle_train/train_bisenetv2_kaggle.ipynb' in a new Kaggle Notebook, attach the dataset, and click 'Run All'!")


if __name__ == "__main__":
    package_segmentation_dataset()
