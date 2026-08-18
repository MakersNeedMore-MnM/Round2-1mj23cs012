"""
Drishti Kavach: Fast High-Speed Dataset Packager for Google Colab GPU Training

Usage:
  python src/colab_training/package_for_colab.py

Features:
  - Compresses 'dataset_rail-drishti' into 'raildrishti_colab.zip'
  - Displays a clean progress bar and compression ratio
  - Provides clear upload instructions for Google Drive root folder (MyDrive)
"""

import os
import zipfile
from tqdm import tqdm


def package_dataset_for_colab():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset_dir = os.path.join(project_root, "dataset_rail-drishti")
    output_zip = os.path.join(project_root, "raildrishti_colab.zip")

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset folder not found at: {dataset_dir}")
        print("        Please run Step 1 & 2 first:")
        print("        1. python src/preprocessing/unified_dataset_builder.py")
        print("        2. python src/augmentation/night_cctv_converter.py --convert-all --workers 8")
        return

    print("=" * 75)
    print(" 📦 PACKAGING 'dataset_rail-drishti' FOR GOOGLE COLAB TRAINING")
    print("=" * 75)
    print(f" • Source Folder : {dataset_dir}")
    print(f" • Target Archive: {output_zip}\n")

    file_list = []
    for cur_root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if not f.startswith("."):
                file_list.append(os.path.join(cur_root, f))

    if not file_list:
        print(f"[!] No files found in {dataset_dir}")
        return

    print(f"[+] Found {len(file_list)} images & annotation files to compress...")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file_path in tqdm(file_list, desc="Creating raildrishti_colab.zip"):
            arcname = os.path.relpath(file_path, start=project_root)
            zipf.write(file_path, arcname)

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(f" 🎉 SUCCESS: Archive created successfully!")
    print(f" • Location : {output_zip}")
    print(f" • File Size: {size_mb:.1f} MB")
    print("=" * 75)
    print("\n🚀 Next Steps for Google Colab:")
    print(" 1. Upload 'raildrishti_colab.zip' to your Google Drive (MyDrive) root directory.")
    print(" 2. Open 'src/colab_training/train_on_colab.ipynb' in Google Colab.")
    print(" 3. Set Runtime -> Change runtime type -> T4 GPU, and click 'Run All'!\n")


if __name__ == "__main__":
    package_dataset_for_colab()
