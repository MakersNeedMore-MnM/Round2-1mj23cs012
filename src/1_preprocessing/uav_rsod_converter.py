"""
Drishti-Kavach: UAV-RSOD Dataset Converter & Merging Script for YOLO11-seg
Processes V1 (Track Segmentation) and V2 (Obstacle Detection) and merges them into dataset_rail-drishti.
"""

import os
import sys
import glob
import shutil
import random
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# Master Class Mapping for Drishti-Kavach YOLO11-seg (10 Classes)
MASTER_CLASS_MAPPING = {
    0: "Rail_Track_Bed",
    1: "Rail_Lines",
    2: "Person",
    3: "Car",
    4: "Truck",
    5: "Branch",
    6: "IronRod",
    7: "Barrel",
    8: "Boulder",
    9: "Jerrycan"
}

# V2 Name Aliases to Class IDs
V2_NAME_TO_CLASS_ID = {
    "person": 2,
    "people": 2,
    "human": 2,
    "branch": 5,
    "ironrod": 6,
    "iron_rod": 6,
    "rod": 6,
    "barrel": 7,
    "drum": 7,
    "boulder": 8,
    "rock": 8,
    "stone": 8,
    "jerrycan": 9,
    "jerry_can": 9,
    "can": 9
}

MIN_CONTOUR_AREA = {
    0: 250,  # Rail_Track_Bed
    1: 50    # Rail_Lines
}


def contour_to_yolo_polygon(contour, img_w: int, img_h: int, epsilon_ratio: float = 0.0012) -> list:
    """Simplifies contour using Douglas-Peucker approximation and normalizes to [0.0, 1.0]."""
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


def bbox_to_yolo_polygon(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> list:
    """Converts bounding box into a 4-vertex normalized polygon contour for YOLO-seg."""
    nx1 = max(0.0, min(1.0, float(xmin) / img_w))
    ny1 = max(0.0, min(1.0, float(ymin) / img_h))
    nx2 = max(0.0, min(1.0, float(xmax) / img_w))
    ny2 = max(0.0, min(1.0, float(ymax) / img_h))
    
    if nx2 <= nx1 or ny2 <= ny1:
        return []
    
    # 4 points: top-left, top-right, bottom-right, bottom-left
    return [
        round(nx1, 6), round(ny1, 6),
        round(nx2, 6), round(ny1, 6),
        round(nx2, 6), round(ny2, 6),
        round(nx1, 6), round(ny2, 6)
    ]


def process_v1_frame(
    frame_id: str,
    img_path: str,
    inside_mask_path: str,
    lines_mask_path: str,
    split: str,
    out_dir: Path
) -> dict:
    """Processes a single UAV-RSOD V1 frame (Track & Rail Segmentation)."""
    res = {"id": frame_id, "split": split, "success": False, "counts": Counter(), "error": None}
    dest_stem = f"uav_v1_{frame_id}"
    
    try:
        m_in = cv2.imread(inside_mask_path, cv2.IMREAD_GRAYSCALE)
        m_li = cv2.imread(lines_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if m_in is None or m_li is None:
            res["error"] = f"Missing masks for {frame_id}"
            return res
            
        img_h, img_w = m_in.shape
        yolo_lines = []
        
        # 1. Rail_Track_Bed (Class 0)
        _, thresh_in = cv2.threshold(m_in, 127, 255, cv2.THRESH_BINARY)
        cnts_in, _ = cv2.findContours(thresh_in, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_in:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA[0]:
                poly = contour_to_yolo_polygon(cnt, img_w, img_h, epsilon_ratio=0.0010)
                if len(poly) >= 6:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    yolo_lines.append(f"0 {coords_str}")
                    res["counts"][0] += 1
                    
        # 2. Rail_Lines (Class 1)
        _, thresh_li = cv2.threshold(m_li, 127, 255, cv2.THRESH_BINARY)
        cnts_li, _ = cv2.findContours(thresh_li, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts_li:
            if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA[1]:
                poly = contour_to_yolo_polygon(cnt, img_w, img_h, epsilon_ratio=0.0008)
                if len(poly) >= 6:
                    coords_str = " ".join(f"{v:.6f}" for v in poly)
                    yolo_lines.append(f"1 {coords_str}")
                    res["counts"][1] += 1
                    
        # Write output label file
        label_out_path = out_dir / "labels" / split / f"{dest_stem}.txt"
        with open(label_out_path, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")
            else:
                f.write("")
                
        # Copy image
        img_out_path = out_dir / "images" / split / f"{dest_stem}.jpg"
        if not img_out_path.exists():
            shutil.copy2(img_path, img_out_path)
            
        res["success"] = True
        return res
    except Exception as e:
        res["error"] = str(e)
        return res


def process_v2_frame(
    img_path: str,
    xml_path: str,
    split: str,
    prefix_code: str,
    out_dir: Path
) -> dict:
    """Processes a single UAV-RSOD V2 frame (Pascal VOC Obstacle Detection)."""
    src_stem = Path(img_path).stem
    dest_stem = f"uav_v2_{prefix_code}_{src_stem}"
    res = {"id": dest_stem, "split": split, "success": False, "counts": Counter(), "error": None}
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size_node = root.find("size")
        if size_node is not None:
            img_w = int(size_node.find("width").text)
            img_h = int(size_node.find("height").text)
        else:
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            
        yolo_lines = []
        for obj in root.findall("object"):
            raw_name = obj.find("name").text.strip().lower()
            cid = V2_NAME_TO_CLASS_ID.get(raw_name)
            if cid is None:
                continue
                
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            poly = bbox_to_yolo_polygon(xmin, ymin, xmax, ymax, img_w, img_h)
            if len(poly) == 8:
                coords_str = " ".join(f"{v:.6f}" for v in poly)
                yolo_lines.append(f"{cid} {coords_str}")
                res["counts"][cid] += 1
                
        # Write output label file
        label_out_path = out_dir / "labels" / split / f"{dest_stem}.txt"
        with open(label_out_path, "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")
            else:
                f.write("")
                
        # Copy image
        img_out_path = out_dir / "images" / split / f"{dest_stem}.jpg"
        if not img_out_path.exists():
            shutil.copy2(img_path, img_out_path)
            
        res["success"] = True
        return res
    except Exception as e:
        res["error"] = str(e)
        return res


def convert_and_merge_uav_rsod(
    uav_dir: str = "dataset_uav-rsod",
    target_dataset_dir: str = "dataset_rail-drishti",
    v1_val_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 8
):
    """Main function to convert V1 & V2 UAV-RSOD and merge into dataset_rail-drishti."""
    random.seed(seed)
    np.random.seed(seed)
    
    uav_base = Path(uav_dir)
    target_base = Path(target_dataset_dir)
    
    if not uav_base.exists():
        print(f"[!] Error: {uav_dir} not found!")
        return False
        
    for split in ["train", "val"]:
        (target_base / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_base / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    v1_stats = {"train": Counter(), "val": Counter()}
    v2_stats = {"train": Counter(), "val": Counter()}
    
    # -------------------------------------------------------------
    # 1. PROCESS V1 (TRACK SEGMENTATION)
    # -------------------------------------------------------------
    v1_dir = uav_base / "V1 UAV-RSOD_Dataset for Segmentation"
    v1_imgs_dir = v1_dir / "1 Images"
    v1_inside_dir = v1_dir / "2 Annotations" / "2.2 Masking" / "Rail Inside"
    v1_lines_dir = v1_dir / "2 Annotations" / "2.2 Masking" / "Rail Lines"
    
    v1_items = []
    if v1_imgs_dir.exists():
        for img_p in sorted(list(v1_imgs_dir.glob("*.jpg"))):
            fid = img_p.stem
            inside_p = v1_inside_dir / f"{fid}.jpg"
            lines_p = v1_lines_dir / f"{fid}.jpg"
            if inside_p.exists() and lines_p.exists():
                v1_items.append((fid, str(img_p), str(inside_p), str(lines_p)))
                
        random.shuffle(v1_items)
        num_v1_val = int(len(v1_items) * v1_val_split)
        v1_val_items = v1_items[:num_v1_val]
        v1_train_items = v1_items[num_v1_val:]
        
        print(f"\n[*] Processing V1 (Track Segmentation): {len(v1_train_items)} Train, {len(v1_val_items)} Val frames...")
        
        v1_tasks = []
        for fid, ip, in_p, li_p in v1_train_items:
            v1_tasks.append((fid, ip, in_p, li_p, "train", target_base))
        for fid, ip, in_p, li_p in v1_val_items:
            v1_tasks.append((fid, ip, in_p, li_p, "val", target_base))
            
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_v1_frame, *t) for t in v1_tasks]
            for f in tqdm(as_completed(futures), total=len(v1_tasks), desc="V1 Track Segm", unit="frame"):
                r = f.result()
                if r["success"]:
                    for cid, cnt in r["counts"].items():
                        v1_stats[r["split"]][cid] += cnt

    # -------------------------------------------------------------
    # 2. PROCESS V2 (OBSTACLE DETECTION)
    # -------------------------------------------------------------
    v2_dir = uav_base / "V2 UAV-RSOD_Dataset for Obstacle Detection" / "images"
    v2_train_dir = v2_dir / "train"
    v2_test_dir = v2_dir / "test"
    
    v2_tasks = []
    if v2_train_dir.exists():
        for img_p in sorted(list(v2_train_dir.glob("*.jpg"))):
            xml_p = v2_train_dir / f"{img_p.stem}.xml"
            if xml_p.exists():
                v2_tasks.append((str(img_p), str(xml_p), "train", "tr", target_base))
                
    if v2_test_dir.exists():
        for img_p in sorted(list(v2_test_dir.glob("*.jpg"))):
            xml_p = v2_test_dir / f"{img_p.stem}.xml"
            if xml_p.exists():
                v2_tasks.append((str(img_p), str(xml_p), "val", "te", target_base))
                
    print(f"\n[*] Processing V2 (Obstacle Detection): {len(v2_tasks)} total frames...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_v2_frame, *t) for t in v2_tasks]
        for f in tqdm(as_completed(futures), total=len(v2_tasks), desc="V2 Obstacles", unit="frame"):
            r = f.result()
            if r["success"]:
                for cid, cnt in r["counts"].items():
                    v2_stats[r["split"]][cid] += cnt
                    
    # -------------------------------------------------------------
    # 3. UPDATE DATASET YAML WITH MASTER 10-CLASS TAXONOMY
    # -------------------------------------------------------------
    yaml_dict = {
        "path": str(target_base.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {int(k): v for k, v in MASTER_CLASS_MAPPING.items()}
    }
    
    data_yaml_path = target_base / "data.yaml"
    configs_yaml_path = Path("configs") / "raildrishti_dataset.yaml"
    
    with open(data_yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
    with open(configs_yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    print("\n" + "="*70)
    print("      UAV-RSOD MERGING COMPLETE (MASTER 10-CLASS TAXONOMY)")
    print("="*70)
    print(f"{'Class ID':<10} {'Class Name':<20} {'V1 Instances':<15} {'V2 Instances':<15} {'Total Added':<12}")
    print("-" * 70)
    
    for cid in range(10):
        cname = MASTER_CLASS_MAPPING[cid]
        v1_tot = v1_stats["train"][cid] + v1_stats["val"][cid]
        v2_tot = v2_stats["train"][cid] + v2_stats["val"][cid]
        tot = v1_tot + v2_tot
        print(f"{cid:<10} {cname:<20} {v1_tot:<15} {v2_tot:<15} {tot:<12}")
        
    print("\n[*] Updated YAML configuration files:")
    print(f"    - {data_yaml_path}")
    print(f"    - {configs_yaml_path}")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert UAV-RSOD and merge into dataset_rail-drishti")
    parser.add_argument("--uav-dir", type=str, default="dataset_uav-rsod", help="Path to UAV-RSOD dataset")
    parser.add_argument("--target-dir", type=str, default="dataset_rail-drishti", help="Output unified dataset directory")
    parser.add_argument("--v1-val-split", type=float, default=0.15, help="V1 validation split ratio (default 0.15)")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    args = parser.parse_args()
    
    convert_and_merge_uav_rsod(
        uav_dir=args.uav_dir,
        target_dataset_dir=args.target_dir,
        v1_val_split=args.v1_val_split,
        num_workers=args.workers
    )
