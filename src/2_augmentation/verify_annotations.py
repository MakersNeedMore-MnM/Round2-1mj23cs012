"""
Drishti Kavach: Multi-Category Side-by-Side Dataset & Ground-Truth Verification Tool

Renders high-clarity side-by-side (Raw Image vs. Ground-Truth Segmentation/Obstacle Mask)
previews for all 6 dataset categories:
  1. rs        - Daylight RailSem19 (Track Bed, Rails, Vehicles, Pedestrians)
  2. night-rs  - 850nm Active IR RailSem19
  3. v1        - Daylight UAV-RSOD V1 (Indian Track Bed & Rail Lines)
  4. night-v1  - 850nm Active IR Indian Track Segmentation
  5. v2        - Daylight UAV-RSOD V2 (Indian Railway Obstacles: Branches, Rods, Boulders, etc.)
  6. night-v2  - 850nm Active IR Indian Railway Obstacles

Saves all side-by-side comparison images directly into 'previews/'.
"""

import os
import glob
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# High-Contrast Ground-Truth Color Palette (BGR)
COLOR_TRACK_BED = (255, 180, 0)        # Translucent Cyan-Blue (BGR)
COLOR_RAIL_LINES = (35, 15, 140)       # Solid Maroon / Crimson (BGR)
COLOR_RAIL_BORDER = (15, 5, 80)        # Dark Maroon Edge (BGR)
COLOR_OBSTACLE_BOX = (0, 0, 255)       # Bright Red Box (BGR)
COLOR_OBSTACLE_BADGE = (0, 0, 180)     # Solid Crimson Badge (BGR)


def render_side_by_side_preview(
    img_path: str,
    lbl_path: str,
    class_names: Dict[int, str],
    category_title: str,
    target_width: int = 960
) -> Optional[np.ndarray]:
    """
    Renders a side-by-side comparison:
      [LEFT]  Raw Camera Feed with Sensor Tag
      [RIGHT] Ground-Truth Track Bed, Rail Lines & Obstacle Annotations
    """
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        return None
        
    orig_h, orig_w = raw_img.shape[:2]
    
    # Layers for annotation overlays
    track_bed_layer = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    rail_lines_layer = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    has_track_bed = False
    has_rail_lines = False
    obstacle_items = []
    
    # Parse YOLO-seg label file
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    if len(coords) < 6 or len(coords) % 2 != 0:
                        continue
                        
                    pts = []
                    for idx in range(0, len(coords), 2):
                        px = int(coords[idx] * orig_w)
                        py = int(coords[idx + 1] * orig_h)
                        pts.append([px, py])
                        
                    pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cname = class_names.get(cid, f"Class_{cid}")
                    
                    if cid == 0:  # Rail_Track_Bed
                        cv2.fillPoly(track_bed_layer, [pts_np], COLOR_TRACK_BED)
                        has_track_bed = True
                    elif cid == 1:  # Rail_Lines
                        cv2.fillPoly(rail_lines_layer, [pts_np], COLOR_RAIL_LINES)
                        cv2.polylines(rail_lines_layer, [pts_np], isClosed=True, color=COLOR_RAIL_BORDER, thickness=3)
                        has_rail_lines = True
                    else:  # Obstacle Bounding / Polygon
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x1, y1 = max(0, min(xs)), max(0, min(ys))
                        x2, y2 = min(orig_w, max(xs)), min(orig_h, max(ys))
                        obstacle_items.append((x1, y1, x2, y2, cname, pts_np))
                except Exception:
                    continue

    # Build Right (Annotated) Canvas
    annotated = raw_img.copy()
    
    # 1. Alpha Blend Track Bed (45% opacity)
    if has_track_bed:
        mask_bed = (track_bed_layer > 0).any(axis=2)
        annotated[mask_bed] = cv2.addWeighted(raw_img, 0.55, track_bed_layer, 0.45, 0)[mask_bed]
        
    # 2. Alpha Blend Rail Lines (75% opacity)
    if has_rail_lines:
        mask_rail = (rail_lines_layer > 0).any(axis=2)
        annotated[mask_rail] = cv2.addWeighted(annotated, 0.25, rail_lines_layer, 0.75, 0)[mask_rail]
        
    # 3. Draw Obstacles (Vivid Red box & Badge)
    for x1, y1, x2, y2, cname, pts_np in obstacle_items:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_OBSTACLE_BOX, 3)
        badge_text = f" {cname.upper()} "
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.8, 2)
        bx1, by1 = x1, max(0, y1 - th - 12)
        bx2, by2 = x1 + tw + 10, y1
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), COLOR_OBSTACLE_BADGE, -1)
        cv2.putText(annotated, badge_text, (bx1 + 4, by2 - 6), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Build Left (Raw) Canvas
    left_canvas = raw_img.copy()
    
    # Top Header Banners on both panels
    banner_h = 55
    cv2.rectangle(left_canvas, (0, 0), (orig_w, banner_h), (18, 18, 20), -1)
    is_night = "night" in category_title.lower()
    sensor_lbl = "SENSOR: 850nm ACTIVE IR CCTV" if is_night else "SENSOR: DAYLIGHT RGB CAMERA"
    cv2.putText(left_canvas, f"[RAW FEED]  {sensor_lbl}", (25, 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (220, 220, 220), 2, cv2.LINE_AA)
                
    cv2.rectangle(annotated, (0, 0), (orig_w, banner_h), (18, 18, 20), -1)
    cv2.putText(annotated, f"[GROUND TRUTH]  {category_title.upper()}", (25, 38), 
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 235, 255), 2, cv2.LINE_AA)

    # Resize both for clean horizontal stacking
    aspect = orig_h / orig_w
    target_h = int(target_width * aspect)
    left_resized = cv2.resize(left_canvas, (target_width, target_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(annotated, (target_width, target_h), interpolation=cv2.INTER_AREA)
    
    # Add thin vertical separator line
    separator = np.full((target_h, 4, 3), (60, 60, 60), dtype=np.uint8)
    combined = np.hstack([left_resized, separator, right_resized])
    return combined


def generate_all_previews(
    dataset_yaml: str = "configs/raildrishti_dataset.yaml",
    output_dir: str = "previews",
    samples_per_category: int = 3,
    seed: int = 42
):
    """Generates minimum 3 structured previews for all 6 categories."""
    random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(dataset_yaml):
        print(f"[!] Error: Config file not found at {dataset_yaml}")
        return False
        
    with open(dataset_yaml) as f:
        cfg = yaml.safe_load(f)
        
    root = Path(cfg.get("path", "dataset_rail-drishti"))
    class_names = cfg.get("names", {})
    
    # Collect all available images across train & val
    all_imgs = list((root / "images" / "train").glob("*.jpg")) + list((root / "images" / "val").glob("*.jpg"))
    print(f"[*] Found {len(all_imgs)} total dataset images. Categorizing...")
    
    # Group images into the 6 distinct categories
    categorized = defaultdict(list)
    for img_p in all_imgs:
        stem = img_p.stem
        split = "train" if "images/train" in str(img_p) else "val"
        lbl_p = root / "labels" / split / f"{stem}.txt"
        if not lbl_p.exists():
            continue
            
        # Determine category
        if stem.startswith("night_rs"):
            cat = "night-rs"
        elif stem.startswith("rs"):
            cat = "rs"
        elif stem.startswith("night_uav_v1"):
            cat = "night-v1"
        elif stem.startswith("uav_v1"):
            cat = "v1"
        elif stem.startswith("night_uav_v2"):
            cat = "night-v2"
        elif stem.startswith("uav_v2"):
            cat = "v2"
        else:
            continue
            
        # Check if frame contains obstacles
        with open(lbl_p) as f:
            cids = [int(l.split()[0]) for l in f if l.strip()]
        has_obs = any(c >= 2 for c in cids)
        
        categorized[cat].append({
            "stem": stem,
            "img_path": str(img_p),
            "lbl_path": str(lbl_p),
            "has_obs": has_obs,
            "cids": cids
        })

    categories_spec = [
        ("rs", "RailSem19 Daylight (Track & Obstacles)"),
        ("night-rs", "RailSem19 Active IR (Night Track & Obstacles)"),
        ("v1", "UAV-RSOD V1 Daylight (Indian Track Segmentation)"),
        ("night-v1", "UAV-RSOD V1 Active IR (Indian Track Segmentation)"),
        ("v2", "UAV-RSOD V2 Daylight (Indian Railway Obstacles)"),
        ("night-v2", "UAV-RSOD V2 Active IR (Indian Railway Obstacles)")
    ]

    print("\n" + "="*70)
    print("      GENERATING SIDE-BY-SIDE VERIFICATION PREVIEWS")
    print("="*70)
    
    saved_previews = []
    
    for idx, (cat_key, cat_desc) in enumerate(categories_spec, 1):
        pool = categorized[cat_key]
        if not pool:
            print(f"[!] Warning: No images found for category '{cat_key}'")
            continue
            
        # Sort or shuffle to select visually informative samples (mix of tracks & obstacles)
        obs_samples = [item for item in pool if item["has_obs"]]
        track_samples = [item for item in pool if not item["has_obs"]]
        
        random.shuffle(obs_samples)
        random.shuffle(track_samples)
        
        chosen = []
        if cat_key in ["v2", "night-v2"]:
            # V2 is obstacle focused: pick distinct obstacle classes
            chosen = obs_samples[:samples_per_category]
        elif cat_key in ["v1", "night-v1"]:
            # V1 is pure track segmentation
            chosen = track_samples[:samples_per_category]
        else:
            # rs and night-rs: pick 2 with obstacles, 1 pure track
            chosen = obs_samples[:2] + track_samples[:1]
            if len(chosen) < samples_per_category:
                chosen = pool[:samples_per_category]
                
        print(f"\n[{idx}/6] Processing Category: {cat_key} ({len(chosen)} samples)...")
        for s_idx, item in enumerate(chosen, 1):
            out_filename = f"preview_{idx}_{cat_key}_sample_{s_idx}.jpg"
            out_file_path = out_path / out_filename
            
            preview_img = render_side_by_side_preview(
                img_path=item["img_path"],
                lbl_path=item["lbl_path"],
                class_names=class_names,
                category_title=f"{cat_key.upper()} | {cat_desc}"
            )
            
            if preview_img is not None:
                cv2.imwrite(str(out_file_path), preview_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_previews.append(str(out_file_path))
                print(f"  ✓ Saved: {out_file_path.name} (Source: {item['stem']})")
                
    print("\n" + "="*70)
    print(f" SUCCESS: {len(saved_previews)} Side-by-Side Previews generated in '{output_dir}/'")
    print("="*70 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Side-by-Side Verification Previews")
    parser.add_argument("--config", type=str, default="configs/raildrishti_dataset.yaml", help="Path to dataset config")
    parser.add_argument("--out", type=str, default="previews", help="Output directory for previews")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples per category (min 3)")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for sample selection")
    args = parser.parse_args()
    
    generate_all_previews(
        dataset_yaml=args.config,
        output_dir=args.out,
        samples_per_category=args.samples,
        seed=args.seed
    )
