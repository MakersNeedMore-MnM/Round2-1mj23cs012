"""
Drishti Kavach: High-Speed Dataset Packager for Google Colab GPU Training

Compresses 'dataset_rail-drishti' and configuration files into 'raildrishti_colab.zip'
for uploading to Google Drive root (MyDrive) when training on Colab Cloud GPUs (A100 / L4 / T4).

Usage:
  python src/4_colab_training/package_for_colab.py
"""

import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm


def package_dataset_for_colab():
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root / "dataset_rail-drishti"
    config_file = project_root / "configs" / "raildrishti_dataset.yaml"
    output_zip = project_root / "raildrishti_colab.zip"

    if not dataset_dir.exists():
        print(f"[!] Error: Dataset directory not found at: {dataset_dir}")
        return False

    print("=" * 75)
    print(" 📦 PACKAGING 'dataset_rail-drishti' FOR GOOGLE COLAB TRAINING")
    print("=" * 75)
    print(f" • Source Folder : {dataset_dir}")
    print(f" • Target Archive: {output_zip}\n")

    # Collect all dataset files
    file_list = []
    for cur_root, _, files in os.walk(dataset_dir):
        for f in files:
            if not f.startswith(".") and not f.endswith(".DS_Store"):
                file_list.append(Path(cur_root) / f)

    if config_file.exists():
        file_list.append(config_file)

    if not file_list:
        print(f"[!] No valid files found to compress in {dataset_dir}")
        return False

    print(f"[*] Found {len(file_list)} files to package...")

    # Fast compression
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as zipf:
        for file_path in tqdm(file_list, desc="Packaging raildrishti_colab.zip", unit="file"):
            arcname = file_path.relative_to(project_root)
            zipf.write(str(file_path), str(arcname))

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(" 🎉 ARCHIVE CREATED SUCCESSFULLY!")
    print("=" * 75)
    print(f" • File Path : {output_zip}")
    print(f" • File Size : {size_mb:.1f} MB")
    print("=" * 75)
    print("\n🚀 Next Steps for Google Colab Cloud GPU Training:")
    print(" 1. Upload 'raildrishti_colab.zip' to your Google Drive root folder (MyDrive).")
    print(" 2. Open Google Colab and run 'src/4_colab_training/enhanced_train_on_colab.py'.")
    print(" 3. Choose your desired action (Start Fresh Training or Resume Paused Epochs)!\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Package dataset_rail-drishti for Google Colab GPU training")
    parser.add_argument("--output", type=str, default="raildrishti_colab.zip", help="Output zip filename")
    parser.add_argument("--fast", action="store_true", help="Use fastest compression level")
    args = parser.parse_args()
    package_dataset_for_colab()
