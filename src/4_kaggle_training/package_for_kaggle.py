"""
Drishti Kavach: High-Speed Dataset Packager for Kaggle Dataset Upload

Compresses 'dataset_rail-drishti' and dataset configuration into 'raildrishti_kaggle.zip'
ready to be uploaded as a new private dataset on Kaggle (https://www.kaggle.com/datasets).

Usage:
  python src/4_kaggle_training/package_for_kaggle.py
"""

import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm


def package_dataset_for_kaggle(output_name: str = "raildrishti_kaggle.zip"):
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = project_root / "dataset_rail-drishti"
    config_file = project_root / "configs" / "raildrishti_dataset.yaml"
    output_zip = project_root / output_name

    if not dataset_dir.exists():
        print(f"[!] Error: Dataset directory not found at: {dataset_dir}")
        return False

    print("=" * 75)
    print(" 📦 PACKAGING 'dataset_rail-drishti' FOR KAGGLE DATASET UPLOAD")
    print("=" * 75)
    print(f" • Source Folder : {dataset_dir}")
    print(f" • Target Archive: {output_zip}\n")

    # Collect all valid dataset files (exclude hidden .DS_Store files)
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

    # Compress archive with deflate level 4 for quick packaging
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as zipf:
        for file_path in tqdm(file_list, desc="Packaging raildrishti_kaggle.zip", unit="file"):
            arcname = file_path.relative_to(project_root)
            zipf.write(str(file_path), str(arcname))

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print("\n" + "=" * 75)
    print(" 🎉 KAGGLE ARCHIVE CREATED SUCCESSFULLY!")
    print("=" * 75)
    print(f" • File Path : {output_zip}")
    print(f" • File Size : {size_mb:.1f} MB")
    print("=" * 75)
    print("\n🚀 Next Steps for Kaggle Cloud GPU Training:")
    print(" 1. Go to https://www.kaggle.com/datasets -> Click '+ New Dataset'.")
    print(" 2. Drag & drop 'raildrishti_kaggle.zip' and title it 'raildrishti-dataset'.")
    print(" 3. Open a new Kaggle Notebook (or upload 'src/4_kaggle_training/train_on_kaggle.ipynb').")
    print(" 4. Attach your dataset via the '+ Add Data' button on the right sidebar.")
    print(" 5. Select Accelerator: 'GPU T4 x 2' or 'GPU P100', toggle Internet: ON, and click Run!\n")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Package dataset_rail-drishti for Kaggle upload")
    parser.add_argument("--output", type=str, default="raildrishti_kaggle.zip", help="Output zip filename")
    args = parser.parse_args()
    package_dataset_for_kaggle(output_name=args.output)
