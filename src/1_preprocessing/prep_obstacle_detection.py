"""
Drishti-Kavach: Unified Railway Obstacle Detection Preprocessor (YOLO11m)

Builds unified 2D bounding box detection dataset across 8 specialized railway classes:
  0: Person        (from UAV-RSOD V2 & RailSem19)
  1: Car           (from RailSem19)
  2: Truck         (from RailSem19)
  3: Branch        (from UAV-RSOD V2 - Fallen Trees)
  4: IronRod       (from UAV-RSOD V2 - Deliberate Sabotage)
  5: Boulder       (from UAV-RSOD V2 - Landslides/Rocks)
  6: Barrel        (from UAV-RSOD V2 - Oil Drums)
  7: Jerrycan      (from UAV-RSOD V2 - Fuel Canisters)

Key Operational Rules:
  - 'train' / 'on-rails' (Label 16) is STRICTLY EXCLUDED (trains are never obstacles).
  - Generates 50% Daylight RGB + 50% Active 850nm NIR Night pairs with identical labels.

Output Directory:
  dataset_detection/
    detection_data.yaml
    images/
      train/ (img_XXXXX_day.jpg, img_XXXXX_night.jpg)
      val/   (img_XXXXX_day.jpg, img_XXXXX_night.jpg)
    labels/
      train/ (img_XXXXX_day.txt, img_XXXXX_night.txt)
      val/   (img_XXXXX_day.txt, img_XXXXX_night.txt)
"""

import os
import sys
import glob
import csv
import json
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from tqdm import tqdm


CLASS_MAP = {
    "Person": 0,
    "Car": 1,
    "Truck": 2,
    "Branch": 3,
    "IronRod": 4,
    "Boulder": 5,
    "Barrel": 6,
    "Jerrycan": 7
}

CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

# Bounding box color palette (BGR)
CLASS_COLORS = {
    0: (255, 120, 0),    # Blue-Cyan for Person
    1: (255, 0, 255),    # Magenta for Car
    2: (180, 0, 255),    # Purple for Truck
    3: (0, 220, 0),      # Green for Branch
    4: (0, 0, 255),      # Red for IronRod (Sabotage)
    5: (0, 255, 255),    # Yellow for Boulder
    6: (200, 200, 0),    # Teal for Barrel
    7: (50, 100, 255)    # Coral for Jerrycan
}


def convert_day_to_night_nir(image: np.ndarray, seed: int = None) -> np.ndarray:
    if image is None: return None
    if seed is not None: np.random.seed(seed)
    h, w, c = image.shape
    weights = [0.18, 0.47, 0.35]
    mono = (image[:, :, 0] * weights[0] + image[:, :, 1] * weights[1] + image[:, :, 2] * weights[2]).astype(np.float32)
    center_x, center_y = w / 2.0, h * 0.60
    Y, X = np.ogrid[:h, :w]
    dist_sq = ((X - center_x) ** 2) / ((w * 0.58) ** 2) + ((Y - center_y) ** 2) / ((h * 0.48) ** 2)
    spotlight = np.exp(-1.4 * dist_sq).astype(np.float32)
    ambient_ir = 0.12
    ir_illuminator = ambient_ir + (1.0 - ambient_ir) * spotlight
    night_base = mono * ir_illuminator
    norm = night_base / 255.0
    gamma = 1.15
    tone_mapped = np.power(norm, gamma) * 255.0
    noise_sigma = 6.0
    noise = np.random.normal(0, noise_sigma, (h, w)).astype(np.float32)
    night_noisy = np.clip(tone_mapped + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(night_noisy, cv2.COLOR_GRAY2BGR)


def convert_bbox_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> list:
    xmin = max(0.0, min(float(xmin), img_w))
    xmax = max(0.0, min(float(xmax), img_w))
    ymin = max(0.0, min(float(ymin), img_h))
    ymax = max(0.0, min(float(ymax), img_h))

    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 3 or bh <= 3:
        return None

    xc = (xmin + xmax) / 2.0 / img_w
    yc = (ymin + ymax) / 2.0 / img_h
    norm_w = bw / img_w
    norm_h = bh / img_h

    return [round(xc, 6), round(yc, 6), round(norm_w, 6), round(norm_h, 6)]


def load_uav_v2_annotations(uav_v2_dir: Path) -> dict:
    records = {}

    for csv_name, img_sub in [("train_labels.csv", "train"), ("test_labels.csv", "test")]:
        csv_p = uav_v2_dir / "images" / csv_name
        if not csv_p.exists():
            continue

        with open(csv_p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls_raw = row["class"].strip()
                if cls_raw not in CLASS_MAP:
                    continue

                cls_id = CLASS_MAP[cls_raw]
                fname = row["filename"].strip()
                w = int(row["width"])
                h = int(row["height"])
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])

                yolo_box = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)
                if yolo_box:
                    img_path = uav_v2_dir / "images" / img_sub / fname
                    if not img_path.exists():
                        img_path = uav_v2_dir / "images" / fname

                    key = (str(img_path), f"uav_{fname}")
                    if key not in records:
                        records[key] = []
                    records[key].append((cls_id, yolo_box))

    return records


def load_railsem19_transport_annotations(railsem_dir: Path, max_frames: int = 1500) -> dict:
    records = {}
    mask_files = sorted(list(railsem_dir.glob("uint8/**/*.png")))

    min_areas = {
        11: 80,   # Human / Person
        13: 150,  # Car
        14: 250   # Truck
    }
    label_to_cls = {
        11: CLASS_MAP["Person"],
        13: CLASS_MAP["Car"],
        14: CLASS_MAP["Truck"]
    }

    for mp in mask_files:
        stem = mp.stem
        img_matches = list(railsem_dir.glob(f"jpgs/**/{stem}.jpg"))
        if not img_matches:
            continue
        img_p = img_matches[0]

        mask = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue

        h, w = mask.shape[:2]
        boxes = []

        for mask_val, min_a in min_areas.items():
            bin_m = (mask == mask_val).astype(np.uint8)
            if not np.any(bin_m):
                continue

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bin_m)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_a:
                    xmin = stats[i, cv2.CC_STAT_LEFT]
                    ymin = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]
                    xmax = xmin + bw
                    ymax = ymin + bh

                    ybox = convert_bbox_to_yolo(xmin, ymin, xmax, ymax, w, h)
                    if ybox:
                        boxes.append((label_to_cls[mask_val], ybox))

        if boxes:
            key = (str(img_p), f"rs19_{stem}.jpg")
            records[key] = boxes
            if max_frames and len(records) >= max_frames:
                break

    return records


def generate_verification_previews(
    uav_records: dict,
    rs19_records: dict,
    out_dir: str = "previews/detection_checks",
    num_samples: int = 10
):
    preview_path = Path(out_dir)
    preview_path.mkdir(parents=True, exist_ok=True)

    combined_dict = {**uav_records, **rs19_records}

    random.seed(12345)
    uav_samples = random.sample(list(uav_records.keys()), min(5, len(uav_records)))
    rs19_samples = random.sample(list(rs19_records.keys()), min(5, len(rs19_records)))
    sample_keys = uav_samples + rs19_samples

    print(f"[*] Generating {len(sample_keys)} bounding-box verification previews in '{out_dir}'...")

    for idx, (img_path, fname) in enumerate(sample_keys, 1):
        boxes = combined_dict[(img_path, fname)]
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        night_bgr = convert_day_to_night_nir(img_bgr, seed=idx*10)

        day_draw = img_bgr.copy()
        night_draw = night_bgr.copy()

        for cls_id, (xc, yc, nw, nh) in boxes:
            xmin = int((xc - nw / 2.0) * w)
            xmax = int((xc + nw / 2.0) * w)
            ymin = int((yc - nh / 2.0) * h)
            ymax = int((yc + nh / 2.0) * h)

            c_name = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_id]

            for target in [day_draw, night_draw]:
                cv2.rectangle(target, (xmin, ymin), (xmax, ymax), color, 3)
                label_txt = f"{c_name}"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(target, (xmin, max(0, ymin - 25)), (xmin + tw + 10, ymin), color, -1)
                cv2.putText(target, label_txt, (xmin + 5, max(18, ymin - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        dw, dh = 960, 540
        def banner(im, txt, col=(0, 255, 200)):
            cv2.rectangle(im, (0, 0), (dw, 36), (20, 20, 20), -1)
            cv2.putText(im, txt, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            return im

        p_day = banner(cv2.resize(day_draw, (dw, dh)), f"Daylight Ground Truth ({fname})", (255, 255, 255))
        p_night = banner(cv2.resize(night_draw, (dw, dh)), f"Active 850nm NIR Night Vision ({fname})", (200, 200, 255))

        side_by_side = np.hstack([p_day, p_night])
        stem = Path(fname).stem
        out_f = preview_path / f"preview_det_{idx:02d}_{stem}.jpg"
        cv2.imwrite(str(out_f), side_by_side)
        print(f"  [+] Saved preview: {out_f}")


def process_single_detection_item(
    img_path: str,
    fname: str,
    boxes: list,
    split: str,
    out_dir: Path,
    include_night: bool = True
) -> dict:
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return {"status": "error", "reason": f"Failed to load {img_path}"}

        stem = Path(fname).stem
        img_dest = out_dir / "images" / split
        lbl_dest = out_dir / "labels" / split

        label_lines = [f"{cls_id} {xc} {yc} {nw} {nh}\n" for cls_id, (xc, yc, nw, nh) in boxes]
        label_str = "".join(label_lines)

        # 1. Save Daylight RGB Pair
        day_img_file = img_dest / f"{stem}_day.jpg"
        day_lbl_file = lbl_dest / f"{stem}_day.txt"
        cv2.imwrite(str(day_img_file), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(day_lbl_file, "w") as f:
            f.write(label_str)

        saved = 1

        # 2. Save Active 850nm NIR Night Pair
        if include_night:
            night_img = convert_day_to_night_nir(img_bgr)
            night_img_file = img_dest / f"{stem}_night.jpg"
            night_lbl_file = lbl_dest / f"{stem}_night.txt"
            cv2.imwrite(str(night_img_file), night_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(night_lbl_file, "w") as f:
                f.write(label_str)
            saved += 1

        return {"status": "success", "saved": saved}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def build_obstacle_dataset(
    output_dir: str = "dataset_detection",
    val_ratio: float = 0.15,
    num_workers: int = 4,
    include_night: bool = True
):
    out_path = Path(output_dir)
    uav_dir = Path("dataset_uav-rsod/V2 UAV-RSOD_Dataset for Obstacle Detection")
    rs19_dir = Path("dataset_railsem19/versions/1")

    print("=" * 70)
    print(" 🛡️ BUILDING UNIFIED 8-CLASS RAILWAY OBSTACLE DATASET (YOLO11m)")
    print("=" * 70)

    print("[*] Parsing UAV-RSOD V2 Sabotage & Person Annotations...")
    uav_rec = load_uav_v2_annotations(uav_dir)
    print(f"  [+] Loaded {len(uav_rec)} UAV-RSOD V2 images.")

    print("[*] Parsing RailSem19 Transport & Person Annotations...")
    rs19_rec = load_railsem19_transport_annotations(rs19_dir)
    print(f"  [+] Loaded {len(rs19_rec)} RailSem19 images.")

    combined = {**uav_rec, **rs19_rec}
    items = list(combined.items())

    random.seed(42)
    random.shuffle(items)

    val_count = max(1, int(len(items) * val_ratio))
    val_items = items[:val_count]
    train_items = items[val_count:]

    print(f"[*] Total Base Images: {len(items)}")
    print(f"[*] Split: {len(train_items)} Train ({len(train_items)*2} with Night), {len(val_items)} Val ({len(val_items)*2} with Night).")

    for split in ["train", "val"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    tasks = []
    for (img_p, fname), boxes in train_items:
        tasks.append((img_p, fname, boxes, "train", out_path, include_night))
    for (img_p, fname), boxes in val_items:
        tasks.append((img_p, fname, boxes, "val", out_path, include_night))

    print(f"[*] Processing with {num_workers} parallel workers...")
    total_saved = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_detection_item, *t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Packaging Detection Dataset"):
            res = f.result()
            if res.get("status") == "success":
                total_saved += res.get("saved", 1)

    # Write clean 8-class detection YAML for YOLO11
    yaml_content = {
        "path": str(out_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}
    }
    with open(out_path / "detection_data.yaml", "w") as yf:
        yaml.dump(yaml_content, yf, default_flow_style=False)

    print("=" * 70)
    print(" ✅ 8-CLASS OBSTACLE DETECTION DATASET COMPLETE!")
    print(f" • Total Frames Generated: {total_saved} (Daylight + Active NIR Night)")
    print(f" • Classes Configured:     {list(CLASS_MAP.keys())}")
    print(f" • Config YAML:            {out_path / 'detection_data.yaml'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Railway Obstacle Detection Preprocessor")
    parser.add_argument("--preview-only", action="store_true", help="Generate preview verification images only")
    parser.add_argument("--out-dir", type=str, default="dataset_detection", help="Output dataset folder")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads")
    args = parser.parse_args()

    uav_dir = Path("dataset_uav-rsod/V2 UAV-RSOD_Dataset for Obstacle Detection")
    rs19_dir = Path("dataset_railsem19/versions/1")

    if args.preview_only:
        shutil.rmtree("previews/detection_checks", ignore_errors=True)
        uav_rec = load_uav_v2_annotations(uav_dir)
        rs19_rec = load_railsem19_transport_annotations(rs19_dir)
        generate_verification_previews(uav_rec, rs19_rec)
    else:
        build_obstacle_dataset(output_dir=args.out_dir, num_workers=args.workers)
