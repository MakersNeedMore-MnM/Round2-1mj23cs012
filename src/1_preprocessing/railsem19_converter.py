"""
Drishti-Kavach: RailSem19 Dataset Converter for YOLO11-seg
Extracts Trackbed, Rail Lines, Person, Car, and Truck annotations into YOLO11-seg format.
"""

import os
import sys
import glob
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# Class Definitions & IDs for Drishti-Kavach YOLO11-seg
CLASS_MAPPING = {
    0: "Rail_Track_Bed",
    1: "Rail_Lines",
    2: "Person",
    3: "Car",
    4: "Truck"
}

# RailSem19 Mask Pixel Value Mappings:
# 3: tram-track, 12: rail-track, 15: trackbed  --> Rail_Track_Bed
# 17: rail-raised, 18: rail-embedded          --> Rail_Lines
# 11: human                                   --> Person
# 13: car                                     --> Car
# 14: truck                                   --> Truck
# 16: on-rails (trains/trams)                 --> EXCLUDED

MIN_AREA_THRESHOLDS = {
    0: 300,  # Rail_Track_Bed
    1: 60,   # Rail_Lines
    2: 50,   # Person
    3: 100,  # Car
    4: 150   # Truck
}


def contour_to_yolo_polygon(contour, img_w: int, img_h: int, epsilon_ratio: float = 0.0012) -> list:
    """
    Simplifies contour using Douglas-Peucker approximation and normalizes to [0.0, 1.0].
    Returns list of normalized floats [x1, y1, x2, y2, ...] or empty list if degenerate.
    """
    peri = cv2.arcLength(contour, True)
    epsilon = max(1.0, peri * epsilon_ratio)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) < 3:
        return []
    
    points = approx.reshape(-1, 2)
    normalized = []
    for x, y in points:
        nx = max(0.0, min(1.0, float(x) / img_w))
        ny = max(0.0, min(1.0, float(y) / img_h))
        normalized.extend([round(nx, 6), round(ny, 6)])
    
    return normalized


def process_single_frame(
    frame_stem: str,
    img_path: str,
    mask_path: str,
    json_path: str,
    split: str,
    out_dir: Path,
    copy_images: bool = True
) -> dict:
    """
    Processes a single RailSem19 frame and generates the YOLO-seg label txt and image copy.
    """
    res = {
        "stem": frame_stem,
        "split": split,
        "success": False,
        "counts": Counter(),
        "error": None
    }
    
    try:
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            res["error"] = f"Cannot read mask: {mask_path}"
            return res
        
        img_h, img_w = mask.shape
        yolo_lines = []
        
        # 1. Rail_Track_Bed (indices 3: tram-track, 12: rail-track inside area)
        tb_mask = np.isin(mask, [3, 12]).astype(np.uint8) * 255
        tb_cnts, _ = cv2.findContours(tb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in tb_cnts:
            if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLDS[0]:
                poly = contour_to_yolo_polygon(cnt, img_w, img_h, epsilon_ratio=0.0010)
                if len(poly) >= 6:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    yolo_lines.append(f"0 {coords_str}")
                    res["counts"][0] += 1
        
        # 2. Rail_Lines (indices 17, 18)
        rail_mask = np.isin(mask, [17, 18]).astype(np.uint8) * 255
        rail_cnts, _ = cv2.findContours(rail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in rail_cnts:
            if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLDS[1]:
                poly = contour_to_yolo_polygon(cnt, img_w, img_h, epsilon_ratio=0.0008)
                if len(poly) >= 6:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    yolo_lines.append(f"1 {coords_str}")
                    res["counts"][1] += 1
        
        # 3. Person (index 11)
        h_mask = (mask == 11).astype(np.uint8) * 255
        h_cnts, _ = cv2.findContours(h_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in h_cnts:
            if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLDS[2]:
                poly = contour_to_yolo_polygon(cnt, img_w, img_h, epsilon_ratio=0.0015)
                if len(poly) >= 6:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    yolo_lines.append(f"2 {coords_str}")
                    res["counts"][2] += 1
        
        # 4. Car (index 13)
        c_mask = (mask == 13).astype(np.uint8) * 255
        c_cnts, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in c_cnts:
            if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLDS[3]:
                poly = contour_to_yolo_polygon(cnt, img_w, img_h, epsilon_ratio=0.0012)
                if len(poly) >= 6:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    yolo_lines.append(f"3 {coords_str}")
                    res["counts"][3] += 1
        
        # 5. Truck (index 14)
        t_mask = (mask == 14).astype(np.uint8) * 255
        t_cnts, _ = cv2.findContours(t_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in t_cnts:
            if cv2.contourArea(cnt) >= MIN_AREA_THRESHOLDS[4]:
                poly = contour_to_yolo_polygon(cnt, img_w, img_h, epsilon_ratio=0.0012)
                if len(poly) >= 6:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    yolo_lines.append(f"4 {coords_str}")
                    res["counts"][4] += 1
        
        # Write output label file
        label_out_path = out_dir / "labels" / split / f"{frame_stem}.txt"
        with open(label_out_path, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")
            else:
                f.write("")  # Background image with 0 instances
        
        # Copy or hardlink image
        img_out_path = out_dir / "images" / split / f"{frame_stem}.jpg"
        if copy_images and not img_out_path.exists():
            shutil.copy2(img_path, img_out_path)
            
        res["success"] = True
        return res
    except Exception as e:
        res["error"] = str(e)
        return res


def convert_railsem19(
    source_dir: str = "dataset_railsem19/versions/1",
    output_dir: str = "dataset_rail-drishti",
    val_split: float = 0.15,
    max_samples: int = None,
    seed: int = 42,
    num_workers: int = 8
):
    """
    Main conversion entry point for RailSem19 to YOLO11-seg dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    src_path = Path(source_dir)
    out_path = Path(output_dir)
    
    jpgs_dir = src_path / "jpgs" / "rs19_val"
    masks_dir = src_path / "uint8" / "rs19_val"
    jsons_dir = src_path / "jsons" / "rs19_val"
    
    if not jpgs_dir.exists() or not masks_dir.exists():
        print(f"[!] Error: Source directories not found in {source_dir}")
        return False
    
    # Collect all valid matching frames
    jpg_files = sorted(list(jpgs_dir.glob("*.jpg")))
    print(f"[*] Found {len(jpg_files)} total images in {jpgs_dir}")
    
    valid_frames = []
    for jpg in jpg_files:
        stem = jpg.stem
        mask_file = masks_dir / f"{stem}.png"
        json_file = jsons_dir / f"{stem}.json"
        if mask_file.exists():
            valid_frames.append((stem, str(jpg), str(mask_file), str(json_file)))
            
    print(f"[*] Verified {len(valid_frames)} frames with matching masks.")
    
    if max_samples and max_samples < len(valid_frames):
        valid_frames = valid_frames[:max_samples]
        print(f"[*] Limiting conversion to {len(valid_frames)} sample frames.")
        
    # Shuffle and Split into train and val
    random.shuffle(valid_frames)
    num_val = int(len(valid_frames) * val_split)
    val_frames = valid_frames[:num_val]
    train_frames = valid_frames[num_val:]
    
    print(f"[*] Dataset Split: {len(train_frames)} Train frames ({100-val_split*100:.0f}%), {len(val_frames)} Val frames ({val_split*100:.0f}%)")
    
    # Create target directory structure
    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # Configs directory
    Path("configs").mkdir(parents=True, exist_ok=True)
    
    # Build task list
    tasks = []
    for stem, ip, mp, jp in train_frames:
        tasks.append((stem, ip, mp, jp, "train", out_path))
    for stem, ip, mp, jp in val_frames:
        tasks.append((stem, ip, mp, jp, "val", out_path))
        
    print(f"[*] Converting frames using {num_workers} parallel workers...")
    
    train_stats = Counter()
    val_stats = Counter()
    success_count = 0
    errors = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_frame, stem, ip, mp, jp, split, out_p): (stem, split)
            for stem, ip, mp, jp, split, out_p in tasks
        }
        
        with tqdm(total=len(tasks), desc="Processing RailSem19", unit="frame") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res["success"]:
                    success_count += 1
                    target_stats = train_stats if res["split"] == "train" else val_stats
                    for cid, count in res["counts"].items():
                        target_stats[cid] += count
                else:
                    errors.append((res["stem"], res["error"]))
                pbar.update(1)
                
    # Generate Dataset YAML files
    yaml_dict = {
        "path": str(out_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in CLASS_MAPPING.items()}
    }
    
    # Save both in root configs and dataset folder
    dataset_yaml_path = out_path / "data.yaml"
    configs_yaml_path = Path("configs") / "raildrishti_dataset.yaml"
    
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    with open(configs_yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    print("\n" + "="*65)
    print("       RAILSEM19 -> YOLO11-SEG CONVERSION COMPLETE")
    print("="*65)
    print(f" Total Processed: {success_count}/{len(tasks)} frames")
    if errors:
        print(f" Errors encountered: {len(errors)}")
        
    print("\n--- INSTANCE SUMMARY BY CLASS ---")
    print(f"{'Class ID':<10} {'Class Name':<20} {'Train Instances':<18} {'Val Instances':<15} {'Total':<10}")
    print("-" * 65)
    for cid in range(len(CLASS_MAPPING)):
        cname = CLASS_MAPPING[cid]
        tr_cnt = train_stats[cid]
        va_cnt = val_stats[cid]
        tot = tr_cnt + va_cnt
        print(f"{cid:<10} {cname:<20} {tr_cnt:<18} {va_cnt:<15} {tot:<10}")
        
    print("\n[*] Dataset YAML configurations written to:")
    print(f"    - {dataset_yaml_path}")
    print(f"    - {configs_yaml_path}")
    print("="*65 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RailSem19 to YOLO11-seg dataset")
    parser.add_argument("--source", type=str, default="dataset_railsem19/versions/1", help="Path to RailSem19 v1")
    parser.add_argument("--output", type=str, default="dataset_rail-drishti", help="Output dataset directory")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio (default 0.15)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit for quick test")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    args = parser.parse_args()
    
    convert_railsem19(
        source_dir=args.source,
        output_dir=args.output,
        val_split=args.val_split,
        max_samples=args.max_samples,
        num_workers=args.workers
    )
