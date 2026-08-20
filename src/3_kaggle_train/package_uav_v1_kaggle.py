"""
Drishti-Kavach: UAV-RSOD V1 Segmentation Dataset Packager for Kaggle

Packages `dataset_segmentation_uav_v1/` into a single zip archive for fast upload to Kaggle.

Usage:
  python src/3_kaggle_train/package_uav_v1_kaggle.py
"""

import os
import sys
import glob
import zipfile
from pathlib import Path
from tqdm import tqdm


def package_uav_v1_dataset(
    dataset_dir: str = "dataset_segmentation_uav_v1",
    output_zip: str = "src/3_kaggle_train/raildrishti_uav_v1_kaggle.zip"
):
    ds_path = Path(dataset_dir)
    out_zip_path = Path(output_zip)
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)

    if not ds_path.exists():
        print(f"[!] Error: {ds_path} not found. Run prep_uav_v1_segmentation.py first.")
        return

    all_files = []
    for root, _, files in os.walk(ds_path):
        for f in files:
            if not f.startswith(".") and not f.endswith(".tmp"):
                all_files.append(os.path.join(root, f))

    print("=" * 70)
    print(" 📦 PACKAGING UAV-RSOD V1 SEGMENTATION DATASET FOR KAGGLE")
    print("=" * 70)
    print(f" • Source Folder: {ds_path.resolve()}")
    print(f" • Total Files:   {len(all_files):,}")
    print(f" • Output Zip:    {out_zip_path.resolve()}")
    print("=" * 70)

    with zipfile.ZipFile(str(out_zip_path), "w", zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for file_path in tqdm(all_files, desc="Compressing UAV-RSOD V1"):
            rel_path = os.path.relpath(file_path, start=ds_path.parent)
            zf.write(file_path, arcname=rel_path)

    zip_size_mb = os.path.getsize(str(out_zip_path)) / (1024 * 1024)
    print("=" * 70)
    print(f" ✅ PACKAGING COMPLETE!")
    print(f" • Zip Archive:   {out_zip_path.name}")
    print(f" • Size on Disk:  {zip_size_mb:.2f} MB")
    print("=" * 70)
    print("\n📋 Next Steps to Fine-Tune on Kaggle:")
    print("  1. Open https://www.kaggle.com/datasets -> Click '+ New Dataset'")
    print(f"  2. Upload: '{out_zip_path.resolve()}'")
    print("  3. Set Dataset Title to 'raildrishti-uav-v1' and click 'Create'.")
    print("  4. Open 'src/3_kaggle_train/finetune_bisenetv2_kaggle.ipynb' in Kaggle, attach the dataset and your base weights, and click 'Run All'!")


if __name__ == "__main__":
    package_uav_v1_dataset()
