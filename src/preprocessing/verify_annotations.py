"""
Drishti Kavach: 20-Sample Structured Annotation Verification Tool

Generates 5 distinct, high-clarity visual verification previews for each of the 4 core dataset categories:
  Category 1 (5 previews): RailSem19 Track Geometry & Segmentation (Cyan Track Bed + Solid Maroon Rails)
  Category 2 (5 previews): RailSem19 Obstacle Detection (Pedestrians, Road Vehicles, Crossings)
  Category 3 (5 previews): UAV-RSOD V1 Track Segmentation (Cyan Track Bed + Solid Maroon Dual Rails)
  Category 4 (5 previews): UAV-RSOD V2 Physical Obstacles (IronRod Sabotage, Boulder, Jerrycan, Barrel, Branch, Cattle)

Usage:
  python src/preprocessing/verify_annotations.py
"""

import os
import glob
import random
import cv2
import numpy as np
import yaml

# High-Contrast Color Palette
COLOR_TRACK_BED = (255, 180, 0)        # Vibrant Cyan-Blue (BGR)
COLOR_RAIL_LINES = (35, 15, 140)       # Solid Maroon / Crimson (BGR)
COLOR_RAIL_BORDER = (15, 5, 80)        # Dark Maroon Structural Edge (BGR)
COLOR_OBSTACLE = (0, 0, 240)           # Bright Red (BGR)
COLOR_OBSTACLE_BORDER = (0, 0, 255)    # Vivid Red Border (BGR)


def render_single_verification(img_path: str, lbl_path: str, names: dict, title_tag: str) -> np.ndarray:
    """Renders track bed, solid maroon rail lines, and obstacle bounding boxes with clear banners."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    
    track_bed_layer = np.zeros((h, w, 3), dtype=np.uint8)
    rail_lines_layer = np.zeros((h, w, 3), dtype=np.uint8)
    has_track_bed = False
    has_rail_lines = False
    obstacle_items = []

    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]

            pts = []
            for idx in range(0, len(coords), 2):
                px = int(coords[idx] * w)
                py = int(coords[idx + 1] * h)
                pts.append([px, py])
            if len(pts) < 3:
                continue
            pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
            cname = names.get(cls_id, f"cls_{cls_id}")

            if cls_id == 0:  # Rail_Track_Bed (Cyan Blue)
                cv2.fillPoly(track_bed_layer, [pts_np], COLOR_TRACK_BED)
                has_track_bed = True
            elif cls_id == 1:  # Rail_Lines (Solid Maroon Fill + Edge)
                cv2.fillPoly(rail_lines_layer, [pts_np], COLOR_RAIL_LINES)
                cv2.polylines(rail_lines_layer, [pts_np], isClosed=True, color=COLOR_RAIL_BORDER, thickness=2)
                has_rail_lines = True
            else:  # Obstacle Bounding Boxes
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                x1, y1 = max(0, min(x_coords)), max(0, min(y_coords))
                x2, y2 = min(w, max(x_coords)), min(h, max(y_coords))
                obstacle_items.append((x1, y1, x2, y2, cname))

    canvas = img.copy()

    # 1. Blend Track Bed (45% opacity)
    if has_track_bed:
        mask_bed = (track_bed_layer > 0).any(axis=2)
        canvas[mask_bed] = cv2.addWeighted(img, 0.55, track_bed_layer, 0.45, 0)[mask_bed]

    # 2. Blend Rail Lines on top with Solid Maroon (75% high opacity)
    if has_rail_lines:
        mask_rail = (rail_lines_layer > 0).any(axis=2)
        canvas[mask_rail] = cv2.addWeighted(canvas, 0.25, rail_lines_layer, 0.75, 0)[mask_rail]

    # 3. Draw Obstacles (Red boxes with solid name badges)
    for x1, y1, x2, y2, cname in obstacle_items:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_OBSTACLE_BORDER, 3)
        badge_text = f" {cname} "
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.8, 2)
        bx1, by1 = x1, max(0, y1 - th - 12)
        bx2, by2 = x1 + tw + 10, y1
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 0, 180), -1)
        cv2.putText(canvas, badge_text, (bx1 + 4, by2 - 6), font, 0.8, (255, 255, 255), 2)

    # 4. Top Category Banner
    cv2.rectangle(canvas, (0, 0), (w, 55), (20, 20, 20), -1)
    cv2.putText(canvas, f"DRISHTI KAVACH GT VERIFICATION: {title_tag}", (30, 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)

    return canvas


def generate_structured_verifications(
    dataset_yaml: str = "configs/raildrishti_dataset.yaml",
    output_dir: str = "outputs/verification_samples"
):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(dataset_yaml):
        print(f"[!] Config file not found: {dataset_yaml}")
        return
        
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("path", "")
    train_img_dir = os.path.join(root, cfg.get("train", "images/train"))
    train_lbl_dir = train_img_dir.replace("images", "labels")
    names = cfg.get("names", {})

    all_labels = glob.glob(os.path.join(train_lbl_dir, "*.txt"))
    if not all_labels:
        print(f"No labels found in {train_lbl_dir}. Please run dataset builder first.")
        return

    # Categorize label files into the 4 groups
    rs19_seg_files = []
    rs19_obs_files = []
    uav1_seg_files = []
    uav2_obs_files = []

    for lf in all_labels:
        fname = os.path.basename(lf)
        with open(lf) as f:
            classes = [int(l.split()[0]) for l in f if l.strip()]
        if not classes:
            continue
            
        if fname.startswith("rs19_"):
            if any(c >= 2 for c in classes):
                rs19_obs_files.append(lf)
            else:
                rs19_seg_files.append(lf)
        elif fname.startswith("uav1_"):
            uav1_seg_files.append(lf)
        elif fname.startswith("uav2_"):
            uav2_obs_files.append(lf)

    random.seed(42)
    categories = [
        ("Cat1_RailSem19_Segmentation", rs19_seg_files, "RailSem19 Driver POV Track Geometry (Cyan Bed + Maroon Rails)"),
        ("Cat2_RailSem19_Obstacles", rs19_obs_files, "RailSem19 Rail Obstacles (Pedestrians, Vehicles, Crossings)"),
        ("Cat3_UAV_V1_Track_Segmentation", uav1_seg_files, "UAV-RSOD V1 Track Segmentation (Cyan Bed + Maroon Dual Rails)"),
        ("Cat4_UAV_V2_Obstacle_Detection", uav2_obs_files, "UAV-RSOD V2 Physical Sabotage & Hazards (IronRod, Boulder, Jerrycan, Cattle)")
    ]

    print("=" * 75)
    print(" GENERATING 20 STRUCTURED VERIFICATION PREVIEWS (5 PREVIEWS PER CATEGORY)")
    print("=" * 75)

    total_generated = 0
    for cat_name, file_list, banner_title in categories:
        print(f"\nCategory: {cat_name} (Found {len(file_list)} candidate files)")
        if not file_list:
            print(" [!] No samples found in this category.")
            continue
            
        # Sample 5 distinct files
        sampled = random.sample(file_list, min(5, len(file_list)))
        for idx, lf in enumerate(sampled):
            base = os.path.splitext(os.path.basename(lf))[0]
            # Match image
            img_path = os.path.join(train_img_dir, f"{base}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(train_img_dir, f"{base}.png")
            if not os.path.exists(img_path):
                continue
                
            annotated_frame = render_single_verification(img_path, lf, names, f"{cat_name} [Sample {idx + 1}/5]")
            if annotated_frame is not None:
                out_name = f"{cat_name}_sample_{idx + 1}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                print(f" [+] Generated: {out_name}")
                total_generated += 1

    print("\n" + "=" * 75)
    print(f" SUCCESS: {total_generated} Verification Previews Generated in {output_dir}/")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    generate_structured_verifications()
