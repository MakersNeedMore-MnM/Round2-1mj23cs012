"""
Drishti Kavach: Curated Balanced Hybrid RailDrishti Dataset Builder

Integrates:
  1. Curated RailSem19 Dataset: ~4,000 High-Diversity Train Cab POV scenes (switches, curves, crossings, pedestrians, road vehicles).
  2. UAV-RSOD V1 Dataset: 315 High-Precision 1080p Track Bed & Rail Lines segmentation masks.
  3. UAV-RSOD V2 Dataset: 2,002 Specialized Physical Obstacle & Sabotage scenes (IronRod, Boulder, Jerrycan, Barrel, Branch, Cattle).

Safety & Integrity Rules:
  - Original source folders (dataset_railsem19 and dataset_uav-rsod) are strictly READ-ONLY.
  - All outputs are written exclusively to 'dataset_rail-drishti/'.
  - Strictly excludes 'train-car' so oncoming trains on parallel tracks are never flagged as foreign obstacles.
  - Multi-threaded parallel processing for fast execution.

Usage:
  python src/preprocessing/unified_dataset_builder.py
"""

import os
import glob
import json
import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from tqdm import tqdm
import yaml
import random

# Standard 11-Class Taxonomy
CLASS_MAPPING = {
    "Rail_Track_Bed": 0,
    "Rail_Lines": 1,
    "Branch": 2,
    "IronRod": 3,
    "Barrel": 4,
    "Boulder": 5,
    "Jerrycan": 6,
    "Person": 7,
    "Cattle": 8,
    "Animal": 9,
    "Vehicle": 10
}

# Mapping for UAV-RSOD V2 Obstacles
UAV_NAME_MAP = {
    "branch": "Branch",
    "ironrod": "IronRod",
    "iron_rod": "IronRod",
    "iron rod": "IronRod",
    "barrel": "Barrel",
    "boulder": "Boulder",
    "jerrycan": "Jerrycan",
    "jerry_can": "Jerrycan",
    "person": "Person",
    "cattle": "Cattle",
    "cow": "Cattle",
    "bull": "Cattle",
    "buffalo": "Cattle",
    "animal": "Animal",
    "dog": "Animal",
    "horse": "Animal",
    "elephant": "Animal",
    "vehicle": "Vehicle",
    "car": "Vehicle",
    "truck": "Vehicle",
    "tractor": "Vehicle"
}


def mask_to_polygons(mask: np.ndarray, min_area: float = 150.0, is_rail_line: bool = False):
    """Extracts smoothed, normalized polygon contours from full-resolution mask."""
    if mask is None or np.sum(mask > 0) < min_area:
        return []
    
    h, w = mask.shape[:2]
    if not is_rail_line:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        binary = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        binary = mask.copy()
        
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        epsilon = 0.0020 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 3:
            continue
            
        pts = approx.reshape(-1, 2)
        norm_pts = []
        for x, y in pts:
            nx = np.clip(float(x) / w, 0.0, 1.0)
            ny = np.clip(float(y) / h, 0.0, 1.0)
            norm_pts.extend([f"{nx:.6f}", f"{ny:.6f}"])
        
        if len(norm_pts) >= 6:
            polygons.append(norm_pts)
            
    return polygons


def parse_railsem19_sample(json_path: str, img_path: str):
    """Parses a RailSem19 image and JSON into standardized YOLO-seg polygons."""
    if not os.path.exists(json_path) or not os.path.exists(img_path):
        return None, False
        
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None, False
        
    w = data.get("imgWidth", 1920)
    h = data.get("imgHeight", 1080)
    
    label_lines = []
    has_high_priority_feature = False
    
    for obj in data.get("objects", []):
        lbl = obj.get("label", "")
        
        # Check for high-priority junction / obstacle features
        if lbl in ["person", "person-group", "car", "truck", "bus", "animal", "crossing", 
                   "switch-left", "switch-right", "switch-static", "switch-unknown", "buffer-stop"]:
            has_high_priority_feature = True
        
        # 1. Rail lines & Track Bed
        if lbl == "rail":
            if "polyline-pair" in obj:
                pair = obj["polyline-pair"]
                if len(pair) == 2 and len(pair[0]) >= 2 and len(pair[1]) >= 2:
                    p1 = np.array(pair[0], dtype=np.int32)
                    p2 = np.array(pair[1], dtype=np.int32)
                    
                    # Rail Lines (Class 1) - Extrude line with 12px thickness
                    for p_line in [p1, p2]:
                        mask_line = np.zeros((h, w), dtype=np.uint8)
                        cv2.polylines(mask_line, [p_line], False, 255, thickness=12)
                        cnts, _ = cv2.findContours(mask_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in cnts:
                            if cv2.contourArea(cnt) > 40:
                                pts = cnt.reshape(-1, 2)
                                norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in pts]
                                if len(norm) >= 3:
                                    label_lines.append(f"1 " + " ".join(norm))
                                    
                    # Rail Track Bed (Class 0) - Enclosed polygon between left & right rails
                    track_poly = np.vstack([p1, p2[::-1]])
                    mask_bed = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask_bed, [track_poly], 255)
                    cnts, _ = cv2.findContours(mask_bed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cnts:
                        if cv2.contourArea(cnt) > 200:
                            pts = cnt.reshape(-1, 2)
                            norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in pts]
                            if len(norm) >= 3:
                                label_lines.append(f"0 " + " ".join(norm))
                                
            elif "polygon" in obj:
                poly = np.array(obj["polygon"], dtype=np.int32)
                if len(poly) >= 3:
                    norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in poly]
                    label_lines.append(f"1 " + " ".join(norm))
                    
        # 2. Person (Class 7)
        elif lbl in ["person", "person-group"]:
            if "polygon" in obj:
                poly = np.array(obj["polygon"], dtype=np.int32)
                if len(poly) >= 3:
                    norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in poly]
                    label_lines.append(f"7 " + " ".join(norm))
            elif "boundingbox" in obj:
                x1, y1, x2, y2 = obj["boundingbox"]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in poly]
                label_lines.append(f"7 " + " ".join(norm))
                
        # 3. Vehicle (Class 10) - (Strictly excludes train-car)
        elif lbl in ["car", "truck", "bus", "vehicle", "tractor", "van", "motorcycle", "bicycle"]:
            if "polygon" in obj:
                poly = np.array(obj["polygon"], dtype=np.int32)
                if len(poly) >= 3:
                    norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in poly]
                    label_lines.append(f"10 " + " ".join(norm))
            elif "boundingbox" in obj:
                x1, y1, x2, y2 = obj["boundingbox"]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in poly]
                label_lines.append(f"10 " + " ".join(norm))
                
        # 4. Animal / Cattle (Class 8 / 9)
        elif lbl in ["animal", "dog", "horse", "cow", "cattle", "bull"]:
            cls_id = 8 if lbl in ["cow", "cattle", "bull"] else 9
            if "polygon" in obj:
                poly = np.array(obj["polygon"], dtype=np.int32)
                if len(poly) >= 3:
                    norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in poly]
                    label_lines.append(f"{cls_id} " + " ".join(norm))
            elif "boundingbox" in obj:
                x1, y1, x2, y2 = obj["boundingbox"]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                norm = [f"{float(x)/w:.6f} {float(y)/h:.6f}" for x, y in poly]
                label_lines.append(f"{cls_id} " + " ".join(norm))

    return label_lines, has_high_priority_feature


def parse_voc_xml(xml_path: str, img_w: int = 1920, img_h: int = 1080):
    """Parses Pascal VOC XML annotations into normalized YOLO-seg polygons."""
    if not os.path.exists(xml_path):
        return []
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []
    
    size_elem = root.find("size")
    if size_elem is not None:
        w_elem = size_elem.find("width")
        h_elem = size_elem.find("height")
        if w_elem is not None and int(w_elem.text) > 0:
            img_w = int(w_elem.text)
        if h_elem is not None and int(h_elem.text) > 0:
            img_h = int(h_elem.text)
            
    objects = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None:
            continue
        raw_name = name_elem.text.strip().lower()
        std_name = UAV_NAME_MAP.get(raw_name, None)
        if std_name is None or std_name not in CLASS_MAPPING:
            continue
        
        cls_id = CLASS_MAPPING[std_name]
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
            
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        
        # Clamp coordinates
        xmin = max(0.0, min(float(img_w), xmin))
        ymin = max(0.0, min(float(img_h), ymin))
        xmax = max(0.0, min(float(img_w), xmax))
        ymax = max(0.0, min(float(img_h), ymax))
        
        if xmax <= xmin or ymax <= ymin:
            continue
            
        # Convert bounding box to 4-point polygon
        x1_n, y1_n = xmin / img_w, ymin / img_h
        x2_n, y2_n = xmax / img_w, ymin / img_h
        x3_n, y3_n = xmax / img_w, ymax / img_h
        x4_n, y4_n = xmin / img_w, ymax / img_h
        
        poly_str = f"{cls_id} {x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x3_n:.6f} {y3_n:.6f} {x4_n:.6f} {y4_n:.6f}"
        objects.append(poly_str)
        
    return objects


def build_unified_dataset(target_rs19_count: int = 4000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(project_root, "dataset_rail-drishti")
    
    # Source Directories (Strictly Read-Only)
    rs19_dir = os.path.join(project_root, "dataset_railsem19/versions/1")
    rs19_jpgs = os.path.join(rs19_dir, "jpgs/rs19_val")
    rs19_jsons = os.path.join(rs19_dir, "jsons/rs19_val")
    
    v1_root = os.path.join(project_root, "dataset_uav-rsod/V1 UAV-RSOD_Dataset for Segmentation")
    v2_root = os.path.join(project_root, "dataset_uav-rsod/V2 UAV-RSOD_Dataset for Obstacle Detection")
    
    # Destination Directories (Newly Generated)
    train_img_dir = os.path.join(output_dir, "images/train")
    val_img_dir = os.path.join(output_dir, "images/val")
    train_lbl_dir = os.path.join(output_dir, "labels/train")
    val_lbl_dir = os.path.join(output_dir, "labels/val")
    
    # Freshly initialize destination folders
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        
    print("=" * 70)
    print(" DRISHTI KAVACH: CURATED BALANCED HYBRID DATASET BUILDER")
    print("=" * 70)
    print(f" • Output Dataset Directory: {output_dir}")
    print(f" • Target RailSem19 Curated Count: ~{target_rs19_count} High-Diversity Scenes")
    print("=" * 70 + "\n")
    
    collected_samples = []
    class_stats = Counter()

    # -------------------------------------------------------------
    # 1. Curate RailSem19 Dataset (~4,000 High-Diversity Scenes)
    # -------------------------------------------------------------
    print("[1/3] Curating RailSem19 Dataset (Prioritizing Features & Junctions)...")
    if os.path.exists(rs19_jpgs) and os.path.exists(rs19_jsons):
        rs19_json_files = sorted(glob.glob(os.path.join(rs19_jsons, "*.json")))
        
        priority_samples = []
        standard_samples = []
        
        for jf in tqdm(rs19_json_files, desc="Scanning RailSem19"):
            base_name = os.path.splitext(os.path.basename(jf))[0]
            img_path = os.path.join(rs19_jpgs, f"{base_name}.jpg")
            if not os.path.exists(img_path):
                continue
                
            labels, is_priority = parse_railsem19_sample(jf, img_path)
            if labels and len(labels) > 0:
                item = {
                    "img_path": img_path,
                    "labels": labels,
                    "dest_name": f"rs19_{base_name}"
                }
                if is_priority:
                    priority_samples.append(item)
                else:
                    standard_samples.append(item)
                    
        print(f" • Found {len(priority_samples)} High-Priority RailSem19 scenes (Pedestrians, Vehicles, Switches, Crossings)")
        print(f" • Found {len(standard_samples)} Standard Track scenes")
        
        # Include 100% of priority scenes, then sample standard scenes to reach target_rs19_count
        curated_rs19 = priority_samples.copy()
        remaining_needed = max(0, target_rs19_count - len(curated_rs19))
        random.seed(42)
        if remaining_needed > 0 and standard_samples:
            sampled_standard = random.sample(standard_samples, min(remaining_needed, len(standard_samples)))
            curated_rs19.extend(sampled_standard)
            
        print(f"[+] Total Curated RailSem19 Scenes Selected: {len(curated_rs19)}")
        for s in curated_rs19:
            collected_samples.append(s)
            for l in s["labels"]:
                cls_id = int(l.split()[0])
                class_stats[cls_id] += 1
    else:
        print(f" [!] RailSem19 path not found at {rs19_dir}, skipping...")

    # -------------------------------------------------------------
    # 2. Process UAV-RSOD V1 Dataset (Track Segmentation Masks)
    # -------------------------------------------------------------
    print("\n[2/3] Parsing UAV-RSOD V1 Track Segmentation Masks (All 315 Scenes)...")
    v1_img_dir = os.path.join(v1_root, "1 Images")
    
    # Accurate paths to Masking and Labelling
    v1_track_mask_dir = os.path.join(v1_root, "2 Annotations/2.2 Masking/Rail Inside")
    v1_rail_mask_dir = os.path.join(v1_root, "2 Annotations/2.2 Masking/Rail Lines")
    v1_track_lbl_dir = os.path.join(v1_root, "2 Annotations/2.1 Labelling/Rail Inside")
    v1_rail_lbl_dir = os.path.join(v1_root, "2 Annotations/2.1 Labelling/Rail Lines")
    
    if os.path.exists(v1_img_dir) and (os.path.exists(v1_track_mask_dir) or os.path.exists(v1_track_lbl_dir)):
        v1_images = sorted(glob.glob(os.path.join(v1_img_dir, "*.jpg")) + glob.glob(os.path.join(v1_img_dir, "*.png")))
        v1_day_images = [p for p in v1_images if not os.path.basename(p).startswith("night_")]
        
        for img_path in tqdm(v1_day_images, desc="UAV-RSOD V1"):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            labels = []
            orig = cv2.imread(img_path)
            if orig is None:
                continue
            h, w = orig.shape[:2]
            
            # Extract Track Bed (Class 0)
            track_mask = None
            mask_file = os.path.join(v1_track_mask_dir, f"{base_name}.jpg")
            if not os.path.exists(mask_file):
                mask_file = os.path.join(v1_track_mask_dir, f"{base_name}.png")
            if os.path.exists(mask_file):
                m_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if m_img is not None and m_img.shape[:2] == (h, w):
                    track_mask = (m_img > 100).astype(np.uint8) * 255
            elif os.path.exists(os.path.join(v1_track_lbl_dir, f"{base_name}.png")):
                lbl_img = cv2.imread(os.path.join(v1_track_lbl_dir, f"{base_name}.png"))
                if lbl_img is not None and lbl_img.shape[:2] == (h, w):
                    diff = np.sum(cv2.absdiff(orig, lbl_img), axis=2)
                    track_mask = (diff > 25).astype(np.uint8) * 255
                    
            if track_mask is not None:
                for poly in mask_to_polygons(track_mask, min_area=300.0, is_rail_line=False):
                    labels.append("0 " + " ".join(poly))
                    class_stats[0] += 1

            # Extract Rail Lines (Class 1)
            rail_mask = None
            rail_file = os.path.join(v1_rail_mask_dir, f"{base_name}.jpg")
            if not os.path.exists(rail_file):
                rail_file = os.path.join(v1_rail_mask_dir, f"{base_name}.png")
            if os.path.exists(rail_file):
                r_img = cv2.imread(rail_file, cv2.IMREAD_GRAYSCALE)
                if r_img is not None and r_img.shape[:2] == (h, w):
                    rail_mask = (r_img > 100).astype(np.uint8) * 255
            elif os.path.exists(os.path.join(v1_rail_lbl_dir, f"{base_name}.png")):
                lbl_img = cv2.imread(os.path.join(v1_rail_lbl_dir, f"{base_name}.png"))
                if lbl_img is not None and lbl_img.shape[:2] == (h, w):
                    diff = np.sum(cv2.absdiff(orig, lbl_img), axis=2)
                    rail_mask = (diff > 25).astype(np.uint8) * 255
                    
            if rail_mask is not None:
                for poly in mask_to_polygons(rail_mask, min_area=100.0, is_rail_line=True):
                    labels.append("1 " + " ".join(poly))
                    class_stats[1] += 1
                        
            if len(labels) > 0:
                collected_samples.append({
                    "img_path": img_path,
                    "labels": labels,
                    "dest_name": f"uav1_{base_name}"
                })
        print(f"[+] Total UAV-RSOD V1 Track Scenes Extracted: {len(v1_day_images)}")
    else:
        print(f" [!] UAV-RSOD V1 path not found at {v1_root}, skipping...")

    # -------------------------------------------------------------
    # 3. Process UAV-RSOD V2 Dataset (Physical Obstacles & Sabotage)
    # -------------------------------------------------------------
    print("\n[3/3] Parsing UAV-RSOD V2 Physical Obstacles (All 2,002 Scenes)...")
    v2_subsets = [
        (os.path.join(v2_root, "images/train"), "uav2_train"),
        (os.path.join(v2_root, "images/test"), "uav2_test")
    ]
    
    v2_count = 0
    for sub_dir, prefix in v2_subsets:
        if not os.path.exists(sub_dir):
            continue
        xml_files = sorted(glob.glob(os.path.join(sub_dir, "*.xml")))
        xml_day_files = [x for x in xml_files if not os.path.basename(x).startswith("night_")]
        
        for xf in tqdm(xml_day_files, desc=f"UAV-RSOD V2 ({prefix})"):
            base_name = os.path.splitext(os.path.basename(xf))[0]
            img_path = os.path.join(sub_dir, f"{base_name}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(sub_dir, f"{base_name}.png")
            if not os.path.exists(img_path):
                continue
                
            labels = parse_voc_xml(xf)
            if labels and len(labels) > 0:
                collected_samples.append({
                    "img_path": img_path,
                    "labels": labels,
                    "dest_name": f"{prefix}_{base_name}"
                })
                v2_count += 1
                for l in labels:
                    cls_id = int(l.split()[0])
                    class_stats[cls_id] += 1

    print(f"[+] Total UAV-RSOD V2 Obstacle Scenes Extracted: {v2_count}")
    print(f"\n[+] Total Balanced Daylight Scenes Collected: {len(collected_samples)}")

    # -------------------------------------------------------------
    # 4. Partition into Train (85%) and Val (15%) Sets
    # -------------------------------------------------------------
    print("\nSplitting dataset into 85% Train / 15% Val...")
    random.seed(42)
    random.shuffle(collected_samples)
    
    val_count = int(len(collected_samples) * 0.15)
    val_samples = collected_samples[:val_count]
    train_samples = collected_samples[val_count:]
    
    print(f" • Training Set   : {len(train_samples)} images")
    print(f" • Validation Set : {len(val_samples)} images")

    def copy_sample(sample, is_train=True):
        img_dest_dir = train_img_dir if is_train else val_img_dir
        lbl_dest_dir = train_lbl_dir if is_train else val_lbl_dir
        
        dest_img_path = os.path.join(img_dest_dir, f"{sample['dest_name']}.jpg")
        dest_lbl_path = os.path.join(lbl_dest_dir, f"{sample['dest_name']}.txt")
        
        # Copy image cleanly into dataset_rail-drishti
        shutil.copyfile(sample["img_path"], dest_img_path)
        
        # Write annotations into dataset_rail-drishti
        with open(dest_lbl_path, "w") as f:
            f.write("\n".join(sample["labels"]) + "\n")

    print("\nWriting Daylight Files to disk (Multithreaded)...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for s in train_samples:
            futures.append(executor.submit(copy_sample, s, True))
        for s in val_samples:
            futures.append(executor.submit(copy_sample, s, False))
        for f in tqdm(futures, desc="Writing Files"):
            f.result()

    # -------------------------------------------------------------
    # 5. Generate raildrishti_dataset.yaml Configuration
    # -------------------------------------------------------------
    inv_class_map = {v: k for k, v in CLASS_MAPPING.items()}
    yaml_dict = {
        "path": "dataset_rail-drishti",
        "train": "images/train",
        "val": "images/val",
        "names": {i: inv_class_map[i] for i in range(len(CLASS_MAPPING))}
    }
    
    yaml_path = os.path.join(project_root, "configs/raildrishti_dataset.yaml")
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    print(f"\n[+] Generated Dataset Config: {yaml_path}")

    # -------------------------------------------------------------
    # 6. Print Class Distribution Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" BALANCED DATASET CLASS INSTANCE DISTRIBUTION")
    print("=" * 70)
    print(f" {'ID':<4} | {'Class Name':<20} | {'Type':<15} | {'Instance Count':<15}")
    print("-" * 70)
    for i in range(len(CLASS_MAPPING)):
        name = inv_class_map[i]
        c_type = "Segmentation" if i in [0, 1] else "Obstacle BBox"
        count = class_stats.get(i, 0)
        print(f" {i:<4} | {name:<20} | {c_type:<15} | {count:<15}")
    print("=" * 70)
    print("\n[+] Step 1 Complete: Balanced Daylight Dataset is ready in dataset_rail-drishti/!")
    print("    Next Step: Synthesize Active IR 850nm CCTV pairs directly into dataset_rail-drishti/ via:")
    print("    python src/augmentation/night_cctv_converter.py --convert-all --workers 8\n")


if __name__ == "__main__":
    build_unified_dataset(target_rs19_count=4000)
