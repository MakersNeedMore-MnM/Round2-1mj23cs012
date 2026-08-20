"""
Drishti-Kavach: UAV-RSOD V1 Semantic Segmentation Preprocessor (Indian Railways Track Perspectives)

Extracts UAV-RSOD V1 masks into unified 3-class segmentation format:
  0: Background
  1: Track_Bed (from 2.2 Masking/Rail Inside)
  2: Rail_Lines (from 2.2 Masking/Rail Lines)

Generates 50% Daylight RGB + 50% Active 850nm NIR Night pairs with matching masks.

Output Directory Structure:
  dataset_segmentation_uav_v1/
    dataset_info.yaml
    images/
      train/ (uav_XXXXX_day.jpg, uav_XXXXX_night.jpg)
      val/   (uav_XXXXX_day.jpg, uav_XXXXX_night.jpg)
    masks/
      train/ (uav_XXXXX_day.png, uav_XXXXX_night.png)
      val/   (uav_XXXXX_day.png, uav_XXXXX_night.png)
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from tqdm import tqdm


CLASS_NAMES = {
    0: "Background",
    1: "Track_Bed",
    2: "Rail_Lines"
}

CLASS_COLORS = {
    0: [0, 0, 0],
    1: [0, 200, 100],   # Emerald Green
    2: [255, 220, 0]    # Bright Yellow
}


def convert_day_to_night_nir(image: np.ndarray, seed: int = None) -> np.ndarray:
    if image is None:
        return None
    if seed is not None:
        np.random.seed(seed)

    h, w, c = image.shape

    # 1. Spectral Conversion: NIR CMOS response
    weights = [0.18, 0.47, 0.35]
    mono = (image[:, :, 0] * weights[0] +
            image[:, :, 1] * weights[1] +
            image[:, :, 2] * weights[2]).astype(np.float32)

    # 2. Conical Infrared Spotlight Vignetting
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

    # 4. Sensor Noise
    noise_sigma = 6.0
    noise = np.random.normal(0, noise_sigma, (h, w)).astype(np.float32)
    night_noisy = np.clip(tone_mapped + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(night_noisy, cv2.COLOR_GRAY2BGR)


def process_single_uav_frame(
    fname: str,
    img_path: str,
    inside_path: str,
    lines_path: str,
    split: str,
    out_dir: Path,
    include_night: bool = True
) -> dict:
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return {"status": "error", "reason": f"Failed to load image: {img_path}"}

        ins_mask = cv2.imread(inside_path, cv2.IMREAD_GRAYSCALE)
        lin_mask = cv2.imread(lines_path, cv2.IMREAD_GRAYSCALE)

        if ins_mask is None or lin_mask is None:
            return {"status": "error", "reason": f"Missing masks for: {fname}"}

        h, w = img_bgr.shape[:2]

        # Handle dimension consistency
        if ins_mask.shape != (h, w):
            ins_mask = cv2.resize(ins_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if lin_mask.shape != (h, w):
            lin_mask = cv2.resize(lin_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Merge into 3-class target mask
        target_mask = np.zeros((h, w), dtype=np.uint8)
        target_mask[ins_mask > 127] = 1 # Track Bed
        target_mask[lin_mask > 127] = 2 # Rail Lines

        stem = Path(fname).stem
        img_dest_dir = out_dir / "images" / split
        mask_dest_dir = out_dir / "masks" / split

        # 1. Save Daylight RGB Pair
        day_img_out = img_dest_dir / f"uav_{stem}_day.jpg"
        day_mask_out = mask_dest_dir / f"uav_{stem}_day.png"
        cv2.imwrite(str(day_img_out), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(day_mask_out), target_mask)

        saved = 1

        # 2. Save Active 850nm NIR Night Pair
        if include_night:
            night_img = convert_day_to_night_nir(img_bgr)
            night_img_out = img_dest_dir / f"uav_{stem}_night.jpg"
            night_mask_out = mask_dest_dir / f"uav_{stem}_night.png"
            cv2.imwrite(str(night_img_out), night_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(night_mask_out), target_mask)
            saved += 1

        return {"status": "success", "fname": fname, "saved": saved}

    except Exception as e:
        return {"status": "error", "reason": str(e)}


def build_uav_segmentation_dataset(
    uav_dir: str = "dataset_uav-rsod/V1 UAV-RSOD_Dataset for Segmentation",
    output_dir: str = "dataset_segmentation_uav_v1",
    val_ratio: float = 0.15,
    num_workers: int = 4,
    include_night: bool = True
):
    uav_path = Path(uav_dir)
    out_path = Path(output_dir)

    img_files = sorted([f for f in (uav_path / "1 Images").glob("*.jpg") if not f.name.startswith(".")])
    print(f"[*] Found {len(img_files)} images in UAV-RSOD V1.")

    tasks_data = []
    for img_p in img_files:
        fname = img_p.name
        inside_p = uav_path / "2 Annotations" / "2.2 Masking" / "Rail Inside" / fname
        lines_p = uav_path / "2 Annotations" / "2.2 Masking" / "Rail Lines" / fname

        if inside_p.exists() and lines_p.exists():
            tasks_data.append((fname, str(img_p), str(inside_p), str(lines_p)))

    print(f"[+] Paired {len(tasks_data)} image-mask sets.")

    # Train / Val Split
    random.seed(42)
    random.shuffle(tasks_data)
    val_count = max(1, int(len(tasks_data) * val_ratio))
    val_items = tasks_data[:val_count]
    train_items = tasks_data[val_count:]

    print(f"[*] Split: {len(train_items)} Train ({len(train_items)*2} with Night), {len(val_items)} Val ({len(val_items)*2} with Night).")

    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "masks" / split).mkdir(parents=True, exist_ok=True)

    tasks = []
    for fname, ip, inp, lp in train_items:
        tasks.append((fname, ip, inp, lp, "train", out_path, include_night))
    for fname, ip, inp, lp in val_items:
        tasks.append((fname, ip, inp, lp, "val", out_path, include_night))

    print(f"[*] Processing with {num_workers} parallel workers...")
    success_count = 0
    total_saved = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_uav_frame, *t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing UAV-RSOD V1"):
            res = f.result()
            if res.get("status") == "success":
                success_count += 1
                total_saved += res.get("saved", 1)
            else:
                print(f"[!] Warning: {res.get('reason')}")

    print("=" * 70)
    print("  UAV-RSOD V1 DATASET GENERATION COMPLETE!")
    print(f"  • Frames Processed: {success_count}/{len(tasks)}")
    print(f"  • Total Saved:      {total_saved} (Daylight + Active NIR Night)")
    print(f"  • Destination:      {out_path.resolve()}")
    print("=" * 70)

    info = {
        "dataset_name": "UAV-RSOD-V1-Segmentation",
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "class_colors_rgb": CLASS_COLORS,
        "train_samples": len(train_items) * (2 if include_night else 1),
        "val_samples": len(val_items) * (2 if include_night else 1),
        "total_samples": total_saved
    }
    with open(out_path / "dataset_info.yaml", "w") as yf:
        yaml.dump(info, yf, default_flow_style=False)

    print(f"[+] Saved dataset info to: {out_path / 'dataset_info.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV-RSOD V1 Semantic Segmentation Preprocessor")
    parser.add_argument("--src-dir", type=str, default="dataset_uav-rsod/V1 UAV-RSOD_Dataset for Segmentation", help="Path to UAV-RSOD V1")
    parser.add_argument("--out-dir", type=str, default="dataset_segmentation_uav_v1", help="Target output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--no-night", action="store_true", help="Disable synthetic night generation")

    args = parser.parse_args()

    build_uav_segmentation_dataset(
        uav_dir=args.src_dir,
        output_dir=args.out_dir,
        val_ratio=args.val_ratio,
        num_workers=args.workers,
        include_night=not args.no_night
    )
