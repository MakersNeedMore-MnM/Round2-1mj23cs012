"""
Drishti Kavach: RailSem19 Semantic Segmentation Preprocessor (Day + Active 850nm NIR Night)

Extracts RailSem19 semantic masks into a unified 3-class segmentation format:
  0: Background
  1: Track_Bed (rail-track, tram-track, ballast trackbed)
  2: Rail_Lines (rail-raised, rail-embedded)

Generates 50% Daylight RGB + 50% Active 850nm NIR Night pairs with identical masks.

Output Directory Structure:
  dataset_segmentation/
    dataset_info.yaml
    images/
      train/ (rsXXXXX_day.jpg, rsXXXXX_night.jpg)
      val/   (rsXXXXX_day.jpg, rsXXXXX_night.jpg)
    masks/
      train/ (rsXXXXX_day.png, rsXXXXX_night.png)
      val/   (rsXXXXX_day.png, rsXXXXX_night.png)
"""

import os
import sys
import glob
import json
import shutil
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# RailSem19 Label IDs:
# 3: tram-track, 12: rail-track      --> Class 1: Track_Bed (Drivable Track Gauge)
# 17: rail-raised, 18: rail-embedded --> Class 2: Rail_Lines
# All other labels (incl. 15 ballast) --> Class 0: Background
TRACK_BED_IDS = [3, 12]
RAIL_LINE_IDS = [17, 18]

CLASS_NAMES = {
    0: "Background",
    1: "Track_Bed",
    2: "Rail_Lines"
}

# Color palette for visualization (RGB)
CLASS_COLORS = {
    0: [0, 0, 0],         # Black
    1: [0, 200, 100],     # Emerald Green for Track Bed
    2: [255, 220, 0]      # Bright Yellow for Rail Lines
}


def convert_day_to_night_nir(image: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Transforms a daylight RGB frame into an authentic 850nm Active IR CCTV surveillance feed.
    """
    if image is None:
        return None
    if seed is not None:
        np.random.seed(seed)

    h, w, c = image.shape

    # 1. Spectral Conversion: NIR CMOS response
    weights = [0.18, 0.47, 0.35]  # B, G, R weights
    mono = (image[:, :, 0] * weights[0] +
            image[:, :, 1] * weights[1] +
            image[:, :, 2] * weights[2]).astype(np.float32)

    # 2. Conical Infrared Illuminator Spotlight Vignetting
    center_x, center_y = w / 2.0, h * 0.60
    Y, X = np.ogrid[:h, :w]
    dist_sq = ((X - center_x) ** 2) / ((w * 0.58) ** 2) + ((Y - center_y) ** 2) / ((h * 0.48) ** 2)

    spotlight = np.exp(-1.4 * dist_sq).astype(np.float32)
    ambient_ir = 0.12
    ir_illuminator = ambient_ir + (1.0 - ambient_ir) * spotlight

    night_base = mono * ir_illuminator

    # 3. Dynamic Range Tone-Mapping
    norm = night_base / 255.0
    gamma = 1.15
    tone_mapped = np.power(norm, gamma) * 255.0

    # 4. Sensor Noise (Shot & Read Noise)
    noise_sigma = 6.0
    noise = np.random.normal(0, noise_sigma, (h, w)).astype(np.float32)
    night_noisy = np.clip(tone_mapped + noise, 0, 255).astype(np.uint8)

    # Return as 3-channel BGR
    return cv2.cvtColor(night_noisy, cv2.COLOR_GRAY2BGR)


def convert_mask_to_3class(raw_mask: np.ndarray) -> np.ndarray:
    """
    Maps RailSem19 raw mask values to our 3-class target:
      0: Background
      1: Track_Bed
      2: Rail_Lines
    """
    target_mask = np.zeros(raw_mask.shape[:2], dtype=np.uint8)

    # First map track bed
    for tid in TRACK_BED_IDS:
        target_mask[raw_mask == tid] = 1

    # Overwrite rail lines on top (since rails run inside track bed)
    for rid in RAIL_LINE_IDS:
        target_mask[raw_mask == rid] = 2

    return target_mask


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Colorizes a 3-class mask for visual inspection.
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_img[mask == class_id] = color[::-1]  # RGB to BGR
    return color_img


def process_single_image(
    stem: str,
    img_path: str,
    mask_path: str,
    split: str,
    output_dir: Path,
    include_night: bool = True
) -> dict:
    """
    Processes one image-mask pair and writes Day and Night instances to output_dir.
    """
    try:
        # Load image & raw mask
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return {"status": "error", "reason": f"Failed to load image: {img_path}"}

        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            return {"status": "error", "reason": f"Failed to load mask: {mask_path}"}

        # Convert mask to 3-class format
        target_mask = convert_mask_to_3class(raw_mask)

        # Output paths
        img_train_dir = output_dir / "images" / split
        mask_train_dir = output_dir / "masks" / split

        # 1. Save Daylight RGB Pair
        day_img_out = img_train_dir / f"{stem}_day.jpg"
        day_mask_out = mask_train_dir / f"{stem}_day.png"
        cv2.imwrite(str(day_img_out), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(day_mask_out), target_mask)

        saved_count = 1

        # 2. Save Active NIR Night Pair (if enabled)
        if include_night:
            night_img = convert_day_to_night_nir(img_bgr)
            night_img_out = img_train_dir / f"{stem}_night.jpg"
            night_mask_out = mask_train_dir / f"{stem}_night.png"
            cv2.imwrite(str(night_img_out), night_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(night_mask_out), target_mask)
            saved_count += 1

        return {
            "status": "success",
            "stem": stem,
            "split": split,
            "saved_pairs": saved_count
        }

    except Exception as e:
        return {"status": "error", "reason": str(e)}


def build_railsem19_segmentation_dataset(
    railsem19_dir: str = "dataset_railsem19/versions/1",
    output_dir: str = "dataset_segmentation",
    val_ratio: float = 0.15,
    max_samples: int = None,
    num_workers: int = 8,
    include_night: bool = True
):
    """
    Converts full RailSem19 into dataset_segmentation with Day + Night pairs.
    """
    railsem_path = Path(railsem19_dir)
    out_path = Path(output_dir)

    jpg_files = sorted(glob.glob(str(railsem_path / "jpgs" / "**" / "*.jpg"), recursive=True))
    print(f"[*] Found {len(jpg_files)} total JPG images in RailSem19.")

    if not jpg_files:
        print(f"[!] Error: No JPG images found under {railsem_path / 'jpgs'}")
        return

    # Build image-mask pairs
    pairs = []
    for jpg in jpg_files:
        stem = Path(jpg).stem
        mask_path = railsem_path / "uint8" / "rs19_val" / f"{stem}.png"
        if not mask_path.exists():
            # Fallback search
            mask_matches = list(railsem_path.glob(f"uint8/**/{stem}.png"))
            if mask_matches:
                mask_path = mask_matches[0]
            else:
                continue
        pairs.append((stem, jpg, str(mask_path)))

    print(f"[+] Successfully paired {len(pairs)} image-mask sets.")

    if max_samples and max_samples < len(pairs):
        random.seed(42)
        pairs = random.sample(pairs, max_samples)
        print(f"[*] Subsampling to {len(pairs)} pairs for quick run.")

    # Train / Val Split
    random.seed(42)
    random.shuffle(pairs)
    val_count = int(len(pairs) * val_ratio)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    print(f"[*] Split: {len(train_pairs)} Train images ({len(train_pairs)*2} with Night), {len(val_pairs)} Val images ({len(val_pairs)*2} with Night).")

    # Create destination directories
    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "masks" / split).mkdir(parents=True, exist_ok=True)

    tasks = []
    for stem, jpg_p, mask_p in train_pairs:
        tasks.append((stem, jpg_p, mask_p, "train", out_path, include_night))
    for stem, jpg_p, mask_p in val_pairs:
        tasks.append((stem, jpg_p, mask_p, "val", out_path, include_night))

    print(f"[*] Starting parallel extraction with {num_workers} worker processes...")
    success_count = 0
    total_saved = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_image, *task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing RailSem19"):
            res = f.result()
            if res.get("status") == "success":
                success_count += 1
                total_saved += res.get("saved_pairs", 1)
            else:
                print(f"[!] Warning: {res.get('reason')}")

    print("=" * 70)
    print(f"  DATASET GENERATION COMPLETE!")
    print(f"  • Source Frames Processed: {success_count}/{len(tasks)}")
    print(f"  • Total Frames Generated:  {total_saved} (Daylight + Active NIR Night)")
    print(f"  • Output Directory:        {out_path.resolve()}")
    print("=" * 70)

    # Write dataset info yaml
    info = {
        "dataset_name": "RailSem19-Drishti-Segmentation",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "class_colors_rgb": CLASS_COLORS,
        "train_samples": len(train_pairs) * (2 if include_night else 1),
        "val_samples": len(val_pairs) * (2 if include_night else 1),
        "total_samples": total_saved
    }
    with open(out_path / "dataset_info.yaml", "w") as yf:
        yaml.dump(info, yf, default_flow_style=False)

    print(f"[+] Saved dataset info to: {out_path / 'dataset_info.yaml'}")


def generate_verification_previews(dataset_dir: str = "dataset_segmentation", num_previews: int = 4, out_dir: str = "previews"):
    """
    Generates side-by-side verification preview images with blended color masks.
    """
    ds_path = Path(dataset_dir)
    preview_path = Path(out_dir)
    preview_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted(glob.glob(str(ds_path / "images" / "val" / "*_day.jpg")))
    if not img_files:
        img_files = sorted(glob.glob(str(ds_path / "images" / "train" / "*_day.jpg")))

    if not img_files:
        print("[!] No images found to preview.")
        return

    sample_imgs = img_files[:num_previews]
    print(f"[*] Generating {len(sample_imgs)} side-by-side verification previews in '{out_dir}'...")

    for i, day_img_path in enumerate(sample_imgs):
        stem = Path(day_img_path).stem.replace("_day", "")
        parent_dir = Path(day_img_path).parent.name  # 'val' or 'train'

        night_img_path = ds_path / "images" / parent_dir / f"{stem}_night.jpg"
        day_mask_path = ds_path / "masks" / parent_dir / f"{stem}_day.png"
        night_mask_path = ds_path / "masks" / parent_dir / f"{stem}_night.png"

        day_bgr = cv2.imread(str(day_img_path))
        night_bgr = cv2.imread(str(night_img_path))
        day_mask = cv2.imread(str(day_mask_path), cv2.IMREAD_UNCHANGED)

        if day_bgr is None or day_mask is None:
            continue

        # Colorize mask
        color_mask = colorize_mask(day_mask)

        # Alpha blend (0.6 image + 0.4 mask) where mask > 0
        overlay_day = day_bgr.copy()
        mask_region = day_mask > 0
        overlay_day[mask_region] = cv2.addWeighted(day_bgr[mask_region], 0.5, color_mask[mask_region], 0.5, 0)

        overlay_night = night_bgr.copy()
        overlay_night[mask_region] = cv2.addWeighted(night_bgr[mask_region], 0.5, color_mask[mask_region], 0.5, 0)

        # Resize for display
        target_h, target_w = 400, 600
        p_day_raw = cv2.resize(day_bgr, (target_w, target_h))
        p_day_over = cv2.resize(overlay_day, (target_w, target_h))
        p_night_raw = cv2.resize(night_bgr, (target_w, target_h))
        p_night_over = cv2.resize(overlay_night, (target_w, target_h))

        # Add title banners
        def add_banner(img, text, color=(0, 255, 200)):
            cv2.rectangle(img, (0, 0), (img.shape[1], 30), (20, 20, 20), -1)
            cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            return img

        p_day_raw = add_banner(p_day_raw, "1. Daylight RGB (Original)", (255, 255, 255))
        p_day_over = add_banner(p_day_over, "2. Ground-Truth Mask Overlay (Green: Bed, Yellow: Rails)", (0, 255, 200))
        p_night_raw = add_banner(p_night_raw, "3. Synthetic 850nm Active IR Night", (200, 200, 255))
        p_night_over = add_banner(p_night_over, "4. Night IR + Ground-Truth Mask Overlay", (0, 255, 200))

        # 2x2 Grid
        top_row = np.hstack([p_day_raw, p_day_over])
        bot_row = np.hstack([p_night_raw, p_night_over])
        grid = np.vstack([top_row, bot_row])

        out_preview_file = preview_path / f"preview_seg_{i+1}_{stem}.jpg"
        cv2.imwrite(str(out_preview_file), grid)
        print(f"  [+] Saved preview: {out_preview_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RailSem19 Semantic Segmentation Preprocessor")
    parser.add_argument("--src-dir", type=str, default="dataset_railsem19/versions/1", help="Path to RailSem19 version 1")
    parser.add_argument("--out-dir", type=str, default="dataset_segmentation", help="Target output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU worker processes")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max sample limit (for quick testing)")
    parser.add_argument("--no-night", action="store_true", help="Disable synthetic night generation")
    parser.add_argument("--preview", action="store_true", help="Generate side-by-side verification preview images only")

    args = parser.parse_args()

    if args.preview:
        generate_verification_previews(args.out_dir)
    else:
        build_railsem19_segmentation_dataset(
            railsem19_dir=args.src_dir,
            output_dir=args.out_dir,
            val_ratio=args.val_ratio,
            max_samples=args.max_samples,
            num_workers=args.workers,
            include_night=not args.no_night
        )
        generate_verification_previews(args.out_dir)
