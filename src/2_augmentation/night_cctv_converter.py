"""
Drishti Kavach: Active Infrared (850nm NIR) CCTV Physics-Based Sensor Simulator

Simulates:
  1. Spectral Reflectance Shift (Greyscale NIR sensor response).
  2. Conical Infrared Illuminator (Spotlight beam falloff & central hotspot).
  3. Shot & Read Noise (Dark sensor CMOS noise simulation).
  4. Metallic Rail Retroreflection Bloom (850nm specular glare along steel track heads).

Safety & Integrity Rules:
  - Original source folders (dataset_railsem19 and dataset_uav-rsod) are strictly UNTOUCHED.
  - Converts images and annotations directly inside 'dataset_rail-drishti/'.

Usage:
  # 1. Synthesize night pairs across the unified dataset:
  python src/augmentation/night_cctv_converter.py --convert-all --workers 8

  # 2. Generate side-by-side verification preview comparison:
  python src/augmentation/night_cctv_converter.py --preview
"""

import os
import glob
import shutil
import argparse
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def convert_day_to_night_cctv(image: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Transforms a daylight RGB frame into an authentic 850nm Active IR CCTV surveillance feed.
    """
    if image is None:
        return None
    if seed is not None:
        np.random.seed(seed)

    h, w, c = image.shape

    # 1. Spectral Conversion: Green/Red dominant NIR CMOS response
    weights = [0.18, 0.47, 0.35]  # B, G, R weights in NIR spectrum
    mono = (image[:, :, 0] * weights[0] +
            image[:, :, 1] * weights[1] +
            image[:, :, 2] * weights[2]).astype(np.float32)

    # 2. Conical Infrared Illuminator Spotlight Vignetting
    center_x, center_y = w / 2.0, h * 0.60
    Y, X = np.ogrid[:h, :w]
    dist_sq = ((X - center_x) ** 2) / ((w * 0.58) ** 2) + ((Y - center_y) ** 2) / ((h * 0.48) ** 2)

    spotlight = np.exp(-1.4 * dist_sq).astype(np.float32)
    ambient_ir = 0.12  # Ambient nocturnal diffuse illumination
    ir_illuminator = ambient_ir + (1.0 - ambient_ir) * spotlight

    night_base = mono * ir_illuminator

    # 3. Dynamic Range Tone-Mapping (S-Curve contrast)
    night_norm = night_base / 255.0
    night_tonemapped = np.power(night_norm, 1.25)
    night_base = night_tonemapped * 255.0

    # 4. Metallic Rail Line Specular Bloom & Retroreflection
    bright_mask = (night_base > 165).astype(np.float32)
    bloom_blur = cv2.GaussianBlur(bright_mask, (15, 15), 0)
    night_base = night_base + (bloom_blur * 22.0)

    # 5. CMOS Sensor Read & Dark Current Noise (Poisson-Gaussian Noise Model)
    sigma_noise = 6.5
    noise = np.random.normal(0, sigma_noise, (h, w)).astype(np.float32)
    night_noisy = night_base + noise

    # Clip to valid 8-bit dynamic range
    night_final_mono = np.clip(night_noisy, 0, 255).astype(np.uint8)

    # 6. Monochromatic 3-Channel CCTV Feed Encoding
    night_bgr = cv2.merge([night_final_mono, night_final_mono, night_final_mono])
    return night_bgr


def convert_image_file(in_path: str, out_path: str, seed: int):
    img = cv2.imread(in_path)
    if img is not None:
        night_img = convert_day_to_night_cctv(img, seed=seed)
        cv2.imwrite(out_path, night_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def convert_unified_dataset(project_root: str, num_workers: int = 8):
    """
    Converts images strictly inside dataset_rail-drishti/ into paired 850nm Active IR CCTV.
    Original source datasets are strictly left untouched.
    """
    unified_root = os.path.join(project_root, "dataset_rail-drishti")
    
    if not os.path.exists(unified_root):
        print(f"[ERROR] Unified dataset directory not found at: {unified_root}")
        print("        Please run Step 1 first: python src/preprocessing/unified_dataset_builder.py")
        return
        
    print("=" * 70)
    print(" SYNTHESIZING ACTIVE IR 850nm CCTV INSIDE dataset_rail-drishti/")
    print("=" * 70)
    
    train_img_dir = os.path.join(unified_root, "images/train")
    val_img_dir = os.path.join(unified_root, "images/val")
    train_lbl_dir = os.path.join(unified_root, "labels/train")
    val_lbl_dir = os.path.join(unified_root, "labels/val")
    
    train_images = glob.glob(os.path.join(train_img_dir, "*.jpg")) + glob.glob(os.path.join(train_img_dir, "*.png"))
    val_images = glob.glob(os.path.join(val_img_dir, "*.jpg")) + glob.glob(os.path.join(val_img_dir, "*.png"))
    
    tasks = []
    
    # 1. Queue Train Images & Annotations
    for img_path in train_images:
        fname = os.path.basename(img_path)
        base, ext = os.path.splitext(fname)
        if base.startswith("night_"):
            continue
        out_path = os.path.join(train_img_dir, f"night_{base}{ext}")
        tasks.append((img_path, out_path, hash(fname) % 100000))
        
        lbl_in = os.path.join(train_lbl_dir, f"{base}.txt")
        lbl_out = os.path.join(train_lbl_dir, f"night_{base}.txt")
        if os.path.exists(lbl_in):
            shutil.copyfile(lbl_in, lbl_out)

    # 2. Queue Val Images & Annotations
    for img_path in val_images:
        fname = os.path.basename(img_path)
        base, ext = os.path.splitext(fname)
        if base.startswith("night_"):
            continue
        out_path = os.path.join(val_img_dir, f"night_{base}{ext}")
        tasks.append((img_path, out_path, hash(fname) % 100000))
        
        lbl_in = os.path.join(val_lbl_dir, f"{base}.txt")
        lbl_out = os.path.join(val_lbl_dir, f"night_{base}.txt")
        if os.path.exists(lbl_in):
            shutil.copyfile(lbl_in, lbl_out)

    print(f" • Daylight Images in dataset_rail-drishti : {len(tasks)}")
    print(f" • Synthesizing {len(tasks)} matched 850nm nocturnal pairs...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_image_file, t[0], t[1], t[2]) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Active IR Synthesis"):
            pass
            
    total_final = len(tasks) * 2
    print("\n" + "=" * 70)
    print(" ACTIVE IR CCTV SYNTHESIS COMPLETE")
    print("=" * 70)
    print(f" • Total Final Dataset Size: {total_final} Images & Annotations")
    print(f" • Stored Exclusively in   : {unified_root}")
    print("=" * 70 + "\n")


def generate_previews(project_root: str):
    """Generates before/after preview comparison images strictly in outputs/day_vs_night_previews."""
    out_dir = os.path.join(project_root, "outputs/day_vs_night_previews")
    os.makedirs(out_dir, exist_ok=True)
    
    # Priority: Grab samples directly from dataset_rail-drishti
    unified_train = os.path.join(project_root, "dataset_rail-drishti/images/train")
    sample_images = []
    if os.path.exists(unified_train):
        sample_images = [f for f in glob.glob(os.path.join(unified_train, "*.jpg")) if not os.path.basename(f).startswith("night_")][:3]
        
    if not sample_images:
        sample_images = glob.glob(os.path.join(project_root, "dataset_railsem19/versions/1/jpgs/rs19_val/*.jpg"))[:3]
        
    for idx, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)
        if img is None:
            continue
        night = convert_day_to_night_cctv(img, seed=idx * 42)
        h, w = img.shape[:2]
        
        # Add labels
        day_annotated = img.copy()
        cv2.putText(day_annotated, "DAYLIGHT RGB SENSOR", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        night_annotated = night.copy()
        cv2.putText(night_annotated, "850nm ACTIVE IR CCTV SENSOR", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        comparison = np.hstack([day_annotated, night_annotated])
        out_file = os.path.join(out_dir, f"preview_comparison_{idx + 1}.jpg")
        cv2.imwrite(out_file, comparison)
        print(f"[+] Saved comparison preview to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Active IR 850nm CCTV Simulator")
    parser.add_argument("--convert-all", action="store_true", help="Convert all images inside dataset_rail-drishti")
    parser.add_argument("--preview", action="store_true", help="Generate side-by-side preview comparisons")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU worker processes")

    args = parser.parse_args()
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    if args.preview:
        generate_previews(proj_root)
    elif args.convert_all:
        convert_unified_dataset(proj_root, num_workers=args.workers)
    else:
        generate_previews(proj_root)
