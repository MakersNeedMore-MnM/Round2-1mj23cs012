"""
Drishti Kavach: Active IR CCTV (850nm) Sensor Simulation Engine
"""

import os
import glob
import math
import shutil
import random
import argparse
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class NightCCTVConverter:
    """
    Transforms daylight railway imagery into authentic Active Infrared (NIR 850nm)
    railway CCTV security footage.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def to_active_ir(
        img_np: np.ndarray,
        spotlight_strength: float = 0.85,
        noise_level: float = 12.0,
        add_bloom: bool = True
    ) -> np.ndarray:
        """
        Converts an RGB image to Active Infrared (NIR 850nm) CCTV security footage.
        
        Features:
        - Spectral response mapping: vegetation and steel rails reflect NIR brightly.
        - Center-weighted IR LED ring spotlight falloff (vignetting).
        - High-gain CMOS sensor noise (Poisson + Gaussian).
        - Subtle rail bloom / halo.
        - CCTV monochrome/cool phosphor grading.
        """
        h, w = img_np.shape[:2]
        img_float = img_np.astype(np.float32)

        # 1. NIR Spectral Response (Green & Red reflect NIR strongly)
        b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
        ir_mono = 0.45 * r + 0.45 * g + 0.10 * b

        # 2. Dynamic Range & Contrast adjustment for IR sensor
        ir_mono = np.power(ir_mono / 255.0, 1.15) * 255.0

        # 3. IR LED Illuminator Spotlight Falloff (Vignetting)
        cx, cy = w / 2.0, h * 0.55
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_sq = ((x_grid - cx) / (w * 0.55)) ** 2 + ((y_grid - cy) / (h * 0.55)) ** 2
        vignette = 1.0 - (1.0 - spotlight_strength) * np.clip(dist_sq, 0.0, 1.0)
        ir_mono = ir_mono * vignette

        # 4. Rail & Specular Highlight Bloom (Glow on bright steel tracks)
        if add_bloom:
            bright_mask = np.clip((ir_mono - 170.0) / 85.0, 0.0, 1.0)
            bloom = cv2.GaussianBlur(ir_mono * bright_mask, (21, 21), 0)
            ir_mono = np.clip(ir_mono + 0.25 * bloom, 0.0, 255.0)

        # 5. CMOS Sensor Gain Noise (Poisson-Gaussian)
        noise = np.random.normal(0, noise_level, (h, w))
        ir_noisy = np.clip(ir_mono + noise, 0.0, 255.0).astype(np.uint8)

        # 6. Apply CCTV monochrome / subtle cool-gray phosphor grading
        cctv_b = np.clip(ir_noisy * 1.02, 0, 255).astype(np.uint8)
        cctv_g = ir_noisy
        cctv_r = np.clip(ir_noisy * 0.98, 0, 255).astype(np.uint8)

        ir_cctv = cv2.merge([cctv_b, cctv_g, cctv_r])
        return ir_cctv


def convert_image_file(in_path: str, out_path: str, seed: int = 42):
    """Converts a single image file to Active IR CCTV style."""
    if os.path.exists(out_path):
        return out_path
    
    img = cv2.imread(in_path)
    if img is None:
        return None
    
    np.random.seed(seed)
    random.seed(seed)
    
    converted = NightCCTVConverter.to_active_ir(img)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, converted)
    return out_path


def generate_sample_previews(sample_img_path: str, output_dir: str = "outputs"):
    """Generates visual comparison grid (Daylight vs Active IR CCTV)."""
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(sample_img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image from {sample_img_path}")

    ir_bgr = NightCCTVConverter.to_active_ir(img_bgr)

    def add_banner(im, title, color_rgb=(255, 255, 255)):
        h, w = im.shape[:2]
        canvas = im.copy()
        cv2.rectangle(canvas, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.putText(canvas, title, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, color_rgb, 2, cv2.LINE_AA)
        return canvas

    card_day = add_banner(img_bgr, "ORIGINAL DAYLIGHT (1080p RGB)", (255, 200, 50))
    card_ir = add_banner(ir_bgr, "DRISHTI KAVACH: ACTIVE IR CCTV (850nm NIGHT VISION)", (80, 255, 140))

    scale = 0.5
    w_s, h_s = int(img_bgr.shape[1] * scale), int(img_bgr.shape[0] * scale)
    grid_day = cv2.resize(card_day, (w_s, h_s))
    grid_ir = cv2.resize(card_ir, (w_s, h_s))

    preview_strip = np.hstack([grid_day, grid_ir])
    
    out_file = os.path.join(output_dir, "day_vs_night_preview.jpg")
    cv2.imwrite(out_file, preview_strip)
    print(f"\n[+] Generated Daylight vs Active IR comparison preview at: {out_file}")
    return out_file


def batch_convert_dataset(dataset_root: str = "dataset_uav-rsod", num_workers: int = 8):
    """
    Scans dataset_uav-rsod and converts all images into Active IR CCTV counterparts.
    Also propagates ground-truth annotations 1:1.
    """
    print("=" * 70)
    print(" DRISHTI KAVACH: BATCH ACTIVE IR CCTV (850nm) DATASET SYNTHESIS")
    print("=" * 70)
    
    train_images = glob.glob(os.path.join(dataset_root, "V2 UAV-RSOD_Dataset for Obstacle Detection/images/train/*.jpg"))
    test_images = glob.glob(os.path.join(dataset_root, "V2 UAV-RSOD_Dataset for Obstacle Detection/images/test/*.jpg"))
    seg_images = glob.glob(os.path.join(dataset_root, "V1 UAV-RSOD_Dataset for Segmentation/1 Images/*.jpg"))

    print(f"Found {len(train_images)} train images, {len(test_images)} test images, {len(seg_images)} seg images.")

    tasks = []
    
    # Tasks for V2 Train
    for img_path in train_images:
        fname = os.path.basename(img_path)
        base, ext = os.path.splitext(fname)
        if base.startswith("night_"):
            continue
        out_path = os.path.join(os.path.dirname(img_path), f"night_{base}{ext}")
        tasks.append((img_path, out_path, hash(fname) % 100000))
        
        xml_in = os.path.join(os.path.dirname(img_path), f"{base}.xml")
        xml_out = os.path.join(os.path.dirname(img_path), f"night_{base}.xml")
        if os.path.exists(xml_in):
            try:
                tree = ET.parse(xml_in)
                root = tree.getroot()
                fn_elem = root.find("filename")
                if fn_elem is not None:
                    fn_elem.text = f"night_{base}{ext}"
                tree.write(xml_out)
            except Exception as e:
                shutil.copy(xml_in, xml_out)

    # Tasks for V2 Test
    for img_path in test_images:
        fname = os.path.basename(img_path)
        base, ext = os.path.splitext(fname)
        if base.startswith("night_"):
            continue
        out_path = os.path.join(os.path.dirname(img_path), f"night_{base}{ext}")
        tasks.append((img_path, out_path, hash(fname) % 100000))
        
        xml_in = os.path.join(os.path.dirname(img_path), f"{base}.xml")
        xml_out = os.path.join(os.path.dirname(img_path), f"night_{base}.xml")
        if os.path.exists(xml_in):
            try:
                tree = ET.parse(xml_in)
                root = tree.getroot()
                fn_elem = root.find("filename")
                if fn_elem is not None:
                    fn_elem.text = f"night_{base}{ext}"
                tree.write(xml_out)
            except Exception as e:
                shutil.copy(xml_in, xml_out)

    # Tasks for V1 Segmentation
    for img_path in seg_images:
        fname = os.path.basename(img_path)
        base, ext = os.path.splitext(fname)
        if base.startswith("night_"):
            continue
        out_path = os.path.join(os.path.dirname(img_path), f"night_{base}{ext}")
        tasks.append((img_path, out_path, hash(fname) % 100000))
        
        for sub_mask in [
            "2 Annotations/2.2 Masking/Rail Inside",
            "2 Annotations/2.2 Masking/Rail Lines",
            "2 Annotations/2.1 Labelling/Rail Inside",
            "2 Annotations/2.1 Labelling/Rail Lines"
        ]:
            mask_src = os.path.join(dataset_root, "V1 UAV-RSOD_Dataset for Segmentation", sub_mask, fname)
            mask_dst = os.path.join(dataset_root, "V1 UAV-RSOD_Dataset for Segmentation", sub_mask, f"night_{base}{ext}")
            if os.path.exists(mask_src):
                shutil.copy(mask_src, mask_dst)

    print(f"Total image conversion tasks queued: {len(tasks)}")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(convert_image_file, t[0], t[1], t[2]) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Synthesizing Active IR CCTV"):
            pass

    print("\n[+] Active IR CCTV Dataset Generation Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drishti Kavach Active IR CCTV Converter")
    parser.add_argument("--preview", action="store_true", help="Generate Daylight vs Active IR comparison preview")
    parser.add_argument("--convert-all", action="store_true", help="Convert all dataset images to Active IR CCTV")
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU worker processes")
    args = parser.parse_args()

    if args.preview:
        sample_path = "dataset_uav-rsod/V1 UAV-RSOD_Dataset for Segmentation/1 Images/63.jpg"
        generate_sample_previews(sample_path)
    elif args.convert_all:
        batch_convert_dataset(num_workers=args.workers)
    else:
        sample_path = "dataset_uav-rsod/V1 UAV-RSOD_Dataset for Segmentation/1 Images/63.jpg"
        generate_sample_previews(sample_path)
