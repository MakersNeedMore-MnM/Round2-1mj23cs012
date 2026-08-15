"""
Drishti Kavach: Unified RailDrishti Dataset Preprocessing Engine

Correctly namespaces V1 Track Segmentation and V2 Obstacle Detection datasets
into a unified multi-task dataset (Daylight + Active IR CCTV).
"""

import os
import glob
import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import yaml
import random

# Class Taxonomy for RailDrishti
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

NAME_MAP = {
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


def extract_fullres_mask(orig_img_path: str, label_img_path: str) -> np.ndarray:
    """Extracts native 1920x1080 binary mask from 2.1 Labelling."""
    if os.path.exists(orig_img_path) and os.path.exists(label_img_path):
        orig = cv2.imread(orig_img_path)
        lbl = cv2.imread(label_img_path)
        if orig is not None and lbl is not None:
            if orig.shape[:2] == lbl.shape[:2]:
                diff = np.sum(cv2.absdiff(orig, lbl), axis=2)
                mask = (diff > 25).astype(np.uint8) * 255
                return mask
    return None


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


def parse_xml_to_polygons(xml_path: str, img_w: int = 1920, img_h: int = 1080):
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
        std_name = NAME_MAP.get(raw_name, None)
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
        
        xmin = np.clip(xmin, 0, img_w)
        ymin = np.clip(ymin, 0, img_h)
        xmax = np.clip(xmax, 0, img_w)
        ymax = np.clip(ymax, 0, img_h)
        
        if xmax <= xmin or ymax <= ymin:
            continue
            
        p1 = (xmin / img_w, ymin / img_h)
        p2 = (xmax / img_w, ymin / img_h)
        p3 = (xmax / img_w, ymax / img_h)
        p4 = (xmin / img_w, ymax / img_h)
        
        poly_str = [
            f"{p1[0]:.6f}", f"{p1[1]:.6f}",
            f"{p2[0]:.6f}", f"{p2[1]:.6f}",
            f"{p3[0]:.6f}", f"{p3[1]:.6f}",
            f"{p4[0]:.6f}", f"{p4[1]:.6f}"
        ]
        objects.append((cls_id, poly_str))
        
    return objects


def build_unified_dataset(dataset_root: str = "dataset_uav-rsod", output_root: str = "dataset_rail-drishti", seed: int = 42):
    """
    Builds the unified RailDrishti dataset:
    - V1 images -> prefixed with 'v1_seg_' with exact Track Bed & Rail Lines masks.
    - V2 images -> prefixed with 'v2_det_' with exact Obstacle Bounding Boxes.
    - Splits 85% train / 15% validation with zero leakage.
    """
    random.seed(seed)
    print("=" * 70)
    print(" DRISHTI KAVACH: BUILDING UNIFIED 'RailDrishti' MULTI-TASK DATASET")
    print(f" Output Directory: {output_root}")
    print("=" * 70)

    # Clean existing directory to ensure fresh build
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    out_train_img = os.path.join(output_root, "images/train")
    out_train_lbl = os.path.join(output_root, "labels/train")
    out_val_img = os.path.join(output_root, "images/val")
    out_val_lbl = os.path.join(output_root, "labels/val")

    for p in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        os.makedirs(p, exist_ok=True)

    stats = {k: 0 for k in CLASS_MAPPING.keys()}
    total_images = 0

    # -------------------------------------------------------------
    # 1. Process V1 Track Segmentation (Day + Night CCTV)
    # -------------------------------------------------------------
    v1_orig_dir = os.path.join(dataset_root, "V1 UAV-RSOD_Dataset for Segmentation/1 Images")
    v1_lbl_in_dir = os.path.join(dataset_root, "V1 UAV-RSOD_Dataset for Segmentation/2 Annotations/2.1 Labelling/Rail Inside")
    v1_lbl_ln_dir = os.path.join(dataset_root, "V1 UAV-RSOD_Dataset for Segmentation/2 Annotations/2.1 Labelling/Rail Lines")

    v1_images = sorted(glob.glob(os.path.join(v1_orig_dir, "*.jpg")))
    print(f"\nProcessing {len(v1_images)} V1 Track Segmentation images (Day + Night IR)...")

    # Group base IDs to prevent train/val leakage between day and night of same scene
    v1_bases = sorted(list({os.path.basename(f).replace("night_", "").split(".")[0] for f in v1_images}))
    random.shuffle(v1_bases)
    val_cut_v1 = int(len(v1_bases) * 0.15)
    v1_val_bases = set(v1_bases[:val_cut_v1])

    for img_path in tqdm(v1_images, desc="Packaging V1 Track Segmentation"):
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]
        raw_base = base.replace("night_", "")
        raw_fname = f"{raw_base}.jpg"

        is_val = raw_base in v1_val_bases
        dst_img_dir = out_val_img if is_val else out_train_img
        dst_lbl_dir = out_val_lbl if is_val else out_train_lbl

        out_fname = f"v1_seg_{fname}"
        out_lbl_name = f"v1_seg_{base}.txt"

        # Copy image
        shutil.copy(img_path, os.path.join(dst_img_dir, out_fname))

        # Extract Ground Truth Masks from 2.1 Labelling
        orig_img_ref = os.path.join(v1_orig_dir, raw_fname)
        lbl_in_ref = os.path.join(v1_lbl_in_dir, raw_fname)
        lbl_ln_ref = os.path.join(v1_lbl_ln_dir, raw_fname)

        mask_in = extract_fullres_mask(orig_img_ref, lbl_in_ref)
        mask_ln = extract_fullres_mask(orig_img_ref, lbl_ln_ref)

        lines_to_write = []
        if mask_in is not None:
            bed_polys = mask_to_polygons(mask_in, min_area=200.0, is_rail_line=False)
            for poly in bed_polys:
                lines_to_write.append(f"0 {' '.join(poly)}")
                stats["Rail_Track_Bed"] += 1

        if mask_ln is not None:
            rail_polys = mask_to_polygons(mask_ln, min_area=100.0, is_rail_line=True)
            for poly in rail_polys:
                lines_to_write.append(f"1 {' '.join(poly)}")
                stats["Rail_Lines"] += 1

        with open(os.path.join(dst_lbl_dir, out_lbl_name), "w") as f:
            if lines_to_write:
                f.write("\n".join(lines_to_write) + "\n")

        total_images += 1

    # -------------------------------------------------------------
    # 2. Process V2 Obstacle Detection (Day + Night CCTV)
    # -------------------------------------------------------------
    v2_train_dir = os.path.join(dataset_root, "V2 UAV-RSOD_Dataset for Obstacle Detection/images/train")
    v2_test_dir = os.path.join(dataset_root, "V2 UAV-RSOD_Dataset for Obstacle Detection/images/test")

    v2_train_images = glob.glob(os.path.join(v2_train_dir, "*.jpg"))
    v2_test_images = glob.glob(os.path.join(v2_test_dir, "*.jpg"))

    print(f"\nProcessing {len(v2_train_images) + len(v2_test_images)} V2 Obstacle Detection images...")

    for img_path in tqdm(v2_train_images, desc="Packaging V2 Train Obstacles"):
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]
        out_fname = f"v2_det_{fname}"
        out_lbl_name = f"v2_det_{base}.txt"

        shutil.copy(img_path, os.path.join(out_train_img, out_fname))

        xml_file = os.path.join(v2_train_dir, f"{base}.xml")
        lines_to_write = []
        if os.path.exists(xml_file):
            obs_polys = parse_xml_to_polygons(xml_file)
            for cls_id, poly in obs_polys:
                lines_to_write.append(f"{cls_id} {' '.join(poly)}")
                for cname, cid in CLASS_MAPPING.items():
                    if cid == cls_id:
                        stats[cname] += 1
                        break

        with open(os.path.join(out_train_lbl, out_lbl_name), "w") as f:
            if lines_to_write:
                f.write("\n".join(lines_to_write) + "\n")
        total_images += 1

    for img_path in tqdm(v2_test_images, desc="Packaging V2 Val Obstacles"):
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]
        out_fname = f"v2_det_{fname}"
        out_lbl_name = f"v2_det_{base}.txt"

        shutil.copy(img_path, os.path.join(out_val_img, out_fname))

        xml_file = os.path.join(v2_test_dir, f"{base}.xml")
        lines_to_write = []
        if os.path.exists(xml_file):
            obs_polys = parse_xml_to_polygons(xml_file)
            for cls_id, poly in obs_polys:
                lines_to_write.append(f"{cls_id} {' '.join(poly)}")
                for cname, cid in CLASS_MAPPING.items():
                    if cid == cls_id:
                        stats[cname] += 1
                        break

        with open(os.path.join(out_val_lbl, out_lbl_name), "w") as f:
            if lines_to_write:
                f.write("\n".join(lines_to_write) + "\n")
        total_images += 1

    # 3. Save configs/raildrishti_dataset.yaml
    os.makedirs("configs", exist_ok=True)
    yaml_data = {
        "path": os.path.abspath(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": {v: k for k, v in CLASS_MAPPING.items()}
    }

    yaml_path = "configs/raildrishti_dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print("\n" + "=" * 70)
    print(" UNIFIED RAILDISHTI MULTI-TASK DATASET SUMMARY")
    print("=" * 70)
    print(f"Total Unified Images: {total_images}")
    print(f"Train Images: {len(os.listdir(out_train_img))}")
    print(f"Val Images:   {len(os.listdir(out_val_img))}")
    print(f"Dataset YAML config saved to: {yaml_path}")
    print("\nClass Instance Breakdown:")
    for k, v in stats.items():
        print(f"  [{CLASS_MAPPING[k]:2d}] {k:<18}: {v} instances")
    print("=" * 70)


if __name__ == "__main__":
    build_unified_dataset()
