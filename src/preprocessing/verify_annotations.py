"""
Drishti Kavach: High-Precision Annotation Verification Tool

Usage:
  python src/preprocessing/verify_annotations.py

Features:
  - Visualizes parsed ground-truth segmentation masks (Vibrant Cyan-Blue Track Bed, Glowing Green Rail Lines)
  - Visualizes multi-class Obstacle Bounding Boxes (Bright Red with drop-shadow badges)
  - Saves annotated validation samples to 'outputs/verification_samples/'
"""

import os
import glob
import random
import cv2
import numpy as np
import yaml

# Color Palette:
# Track Bed: Bright Cyan/Blue (255, 180, 0)
# Rail Lines: Glowing Green (0, 255, 100)
# Obstacles: Bright Red (0, 0, 255)
COLOR_TRACK_BED = (255, 180, 0)      # Pristine Bright Cyan/Blue (BGR)
COLOR_RAIL_LINES = (0, 255, 100)     # Pristine Glowing Green (BGR)
COLOR_OBSTACLE = (0, 0, 240)         # Bright Red (BGR)
COLOR_OBSTACLE_BORDER = (0, 0, 255)  # Vivid Red Border (BGR)


def visualize_annotations(
    dataset_yaml: str = "configs/raildrishti_dataset.yaml",
    output_dir: str = "outputs/verification_samples"
):
    os.makedirs(output_dir, exist_ok=True)
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)

    root = cfg.get("path", "")
    train_img_dir = os.path.join(root, cfg.get("train", "images/train"))
    train_lbl_dir = train_img_dir.replace("images", "labels")
    names = cfg.get("names", {})

    all_images = glob.glob(os.path.join(train_img_dir, "*.jpg"))
    if not all_images:
        print(f"No images found in {train_img_dir}. Please run dataset builder first.")
        return

    # Categorize into V1 Track Seg (Day + Night) and V2 Obstacles (Day + Night)
    v1_day = [f for f in all_images if "v1_seg_" in os.path.basename(f) and "night_" not in os.path.basename(f)]
    v1_night = [f for f in all_images if "v1_seg_" in os.path.basename(f) and "night_" in os.path.basename(f)]
    v2_day = [f for f in all_images if "v2_det_" in os.path.basename(f) and "night_" not in os.path.basename(f)]
    v2_night = [f for f in all_images if "v2_det_" in os.path.basename(f) and "night_" in os.path.basename(f)]

    selected = (
        random.sample(v1_day, min(2, len(v1_day))) +
        random.sample(v1_night, min(2, len(v1_night))) +
        random.sample(v2_day, min(2, len(v2_day))) +
        random.sample(v2_night, min(2, len(v2_night)))
    )

    print(f"Generating verification samples with exact pristine_test rendering...")

    for i, img_path in enumerate(selected):
        fname = os.path.basename(img_path)
        base, _ = os.path.splitext(fname)
        lbl_path = os.path.join(train_lbl_dir, f"{base}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Single Unified Overlay Layer
        overlay = img.copy()
        obstacle_items = []
        has_track = False

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
                pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
                cname = names.get(cls_id, f"cls_{cls_id}")

                if cls_id == 0:  # Rail_Track_Bed (Cyan Blue)
                    cv2.fillPoly(overlay, [pts_np], COLOR_TRACK_BED)
                    has_track = True
                elif cls_id == 1:  # Rail_Lines (Glowing Green Rails)
                    cv2.fillPoly(overlay, [pts_np], COLOR_RAIL_LINES)
                    cv2.polylines(overlay, [pts_np], True, COLOR_RAIL_LINES, 2, cv2.LINE_AA)
                    has_track = True
                else:  # Obstacle (Bright Red)
                    obstacle_items.append((cls_id, cname, pts_np))

        # Single 50/50 Alpha Blend
        if has_track:
            blended = cv2.addWeighted(overlay, 0.50, img, 0.50, 0)
        else:
            blended = img.copy()

        # Draw Obstacles on top in Bright Red with badges
        for cls_id, cname, pts_np in obstacle_items:
            cv2.polylines(blended, [pts_np], True, COLOR_OBSTACLE_BORDER, 3, cv2.LINE_AA)
            
            x_min, y_min = pts_np[:, 0, 0].min(), pts_np[:, 0, 1].min()
            
            badge_text = f"{cname} [Hazard]"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
            
            badge_y1 = max(0, y_min - th - 12)
            badge_y2 = max(th + 12, y_min)
            badge_x2 = min(w, x_min + tw + 18)
            
            # Bright Red badge background
            cv2.rectangle(blended, (x_min, badge_y1), (badge_x2, badge_y2), COLOR_OBSTACLE, -1)
            cv2.rectangle(blended, (x_min, badge_y1), (badge_x2, badge_y2), (255, 255, 255), 1)
            
            # Crisp white badge text
            cv2.putText(blended, badge_text, (x_min + 8, badge_y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(output_dir, f"sample_{i+1}_{fname}")
        cv2.imwrite(out_path, blended)
        print(f"  [+] Saved verification preview: {out_path}")

    print(f"\n[+] Updated verification samples saved to: {output_dir}")


if __name__ == "__main__":
    visualize_annotations()
