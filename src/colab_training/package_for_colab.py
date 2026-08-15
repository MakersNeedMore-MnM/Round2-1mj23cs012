"""
Drishti Kavach: Google Colab Packaging Utility

Usage:
  python colab_training/package_for_colab.py

Features:
  - Zips the processed 'dataset_rail-drishti' and configuration files into 'raildrishti_colab.zip'
  - Generates ready-to-upload archive for training on Google Colab GPU (Tesla T4 / A100)
"""

import os
import zipfile
from tqdm import tqdm


def find_project_root():
    """Finds workspace root directory."""
    current = os.path.abspath(os.getcwd())
    if os.path.exists(os.path.join(current, "dataset_rail-drishti")):
        return current
    parent = os.path.abspath(os.path.join(current, ".."))
    if os.path.exists(os.path.join(parent, "dataset_rail-drishti")):
        return parent
    return current


def package_dataset_for_colab():
    root = find_project_root()
    dataset_dir = os.path.join(root, "dataset_rail-drishti")
    output_zip = os.path.join(root, "raildrishti_colab.zip")

    if not os.path.exists(dataset_dir):
        print(f"[!] Error: {dataset_dir} not found. Please run unified_dataset_builder.py first.")
        return

    print("=" * 75)
    print(" 📦 PACKAGING 'RailDrishti' DATASET FOR GOOGLE COLAB TRAINING")
    print("=" * 75)
    print(f" Source Directory: {dataset_dir}")
    print(f" Target Archive:   {output_zip}\n")

    file_list = []
    for cur_root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if not f.startswith("."):
                file_list.append(os.path.join(cur_root, f))

    print(f"[+] Found {len(file_list)} files to package. Creating zip archive...")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(file_list, desc="Compressing Dataset"):
            arcname = os.path.relpath(file_path, start=root)
            zipf.write(file_path, arcname)

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f" 🎉 SUCCESS: Archive created at: {output_zip} ({size_mb:.1f} MB)")
    print("=" * 75)


if __name__ == "__main__":
    package_dataset_for_colab()
