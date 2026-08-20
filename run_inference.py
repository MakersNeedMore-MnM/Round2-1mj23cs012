"""
Drishti Kavach: Real-Time Railway Physical Obstacle & Track Clearance Inference Engine

CLI Flags & Usage:
  --source STR/INT : Input source: '0' (Webcam), 'video.mp4', 'image.jpg', 'dir/', 'rtsp://...' (default: 0)
  --seg-model STR  : Path to BiSeNetV2 segmentation model (default: models/raildrishti_seg_universal.pth)
  --det-model STR  : Path to YOLO11m obstacle detection model (default: models/best_yolo11m_raildrishti.pt)
  --conf FLOAT     : Confidence threshold for obstacles (default: 0.35)
  --imgsz INT      : Inference resolution for obstacles (default: 1024)
  --weather STR    : Weather optimizer: 'auto', 'clahe', 'dcp', 'rain', 'off' (default: auto)
  --sensor STR     : Sensor label: 'DAYLIGHT RGB' or '850nm ACTIVE IR CCTV' (default: DAYLIGHT RGB)
  --save           : Save annotated output stream to outputs/inference_results/
  --no-view        : Run in headless mode without opening GUI window

Interactive Keyboard Controls (in GUI Window):
  [Q] / [ESC]      : Quit stream
  [D]              : Toggle / Cycle Weather Defogging Modes (Auto -> CLAHE -> DCP -> Off)
  [H]              : Toggle HUD Telemetry Overlay on/off
  [S]              : Capture snapshot to outputs/snapshots/
  [SPACE]          : Pause / Resume stream

Examples:
  # 1. Live USB Camera:
  python run_inference.py --source 0

  # 2. Process a Test Image:
  python run_inference.py --source dataset_segmentation/images/val/rs06769_day.jpg --save

  # 3. Process Video with Active IR Night Vision:
  python run_inference.py --source test_rail.mp4 --sensor "850nm ACTIVE IR CCTV"
"""

import os
import sys
import time
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np

from src import DrishtiEngine


def run_inference(
    source: str = "0",
    seg_model_path: str = "models/RailDrishti_Seg_Universal.pth",
    det_model_path: str = "models/best_yolo11m_raildrishti.pt",
    conf_thresh: float = 0.35,
    imgsz: int = 1024,
    weather_mode: str = "auto",
    sensor_type: str = "DAYLIGHT RGB",
    save_output: bool = False,
    show_view: bool = True
):
    output_dir = "outputs/inference_results"
    snapshot_dir = "outputs/snapshots"
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    # Initialize Decoupled Engine
    engine = DrishtiEngine(
        seg_model_path=seg_model_path,
        det_model_path=det_model_path,
        conf_thresh=conf_thresh,
        imgsz=imgsz,
        weather_mode=weather_mode
    )

    # 1. Check if source is a single image
    is_img = False
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    if os.path.isfile(source) and any(source.lower().endswith(ext) for ext in valid_exts):
        is_img = True

    if is_img:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[!] Error: Could not load image: {source}")
            return

        rendered, status, hazards, telemetry = engine.process_frame(
            frame, sensor_type=sensor_type, show_hud=True, is_video_stream=False
        )

        print(f"\n[*] Processing Complete for: {source}")
        print(f"  • Clearance Status: {status}")
        print(f"  • Hazards Detected: {len(hazards)}")
        print(f"  • Frame Rate:       {telemetry['fps']:.1f} FPS")

        if save_output:
            out_name = f"result_{os.path.basename(source)}"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, rendered)
            print(f"[+] Saved result to: {out_path}")

        if show_view:
            cv2.imshow("Drishti Kavach - Railway Clearance ATP", rendered)
            print("\n[Controls] Press any key in the window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Video Capture Stream
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video source: {source}")
        return

    writer = None
    if save_output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fps_in = fps_in if fps_in > 0 else 30.0
        
        base_src_name = Path(source).stem if not source.isdigit() else f"stream_{int(time.time())}"
        out_video_path = os.path.join(output_dir, f"result_{base_src_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (w, h))
        print(f"[+] Output video will be saved to: {out_video_path}")

    show_hud = True
    paused = False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not source.isdigit() else None
    
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame") if total_frames and total_frames > 0 else None

    print("\n" + "=" * 75)
    print(" 🚀 DRISHTI KAVACH REAL-TIME STREAM ACTIVE")
    print(" • Press [Q] or [ESC] to Quit")
    print(" • Press [D] to Cycle Weather Defogging Modes")
    print(" • Press [H] to Toggle HUD Dashboard")
    print(" • Press [S] to Save Snapshot")
    print(" • Press [SPACE] to Pause / Resume")
    print("=" * 75 + "\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("\n[*] End of video stream reached.")
                    break

                rendered, status, hazards, telemetry = engine.process_frame(
                    frame, sensor_type=sensor_type, show_hud=show_hud, is_video_stream=True
                )

                if writer is not None:
                    writer.write(rendered)
                    
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({"status": status, "hazards": len(hazards), "fps": f"{telemetry['fps']:.1f}"})

            if show_view:
                cv2.imshow("Drishti Kavach - Railway Clearance ATP", rendered)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q"), 27]:  # ESC
                    break
                elif key in [ord("h"), ord("H")]:
                    show_hud = not show_hud
                elif key in [ord("s"), ord("S")]:
                    snap_path = os.path.join(snapshot_dir, f"snap_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, rendered)
                    print(f"\n[+] Snapshot saved: {snap_path}")
                elif key == 32:  # SPACE
                    paused = not paused
                elif key in [ord("d"), ord("D")]:
                    modes = ["auto", "clahe", "dcp", "off"]
                    cur_idx = modes.index(engine.weather_mode) if engine.weather_mode in modes else 0
                    engine.weather_mode = modes[(cur_idx + 1) % len(modes)]
                    print(f"\n[*] Switched Weather Mode to: {engine.weather_mode.upper()}")
    finally:
        if pbar is not None:
            pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n[+] Successfully saved output video to: {out_video_path}")
        if show_view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drishti Kavach Real-Time Inference")
    parser.add_argument("--source", type=str, default="0", help="Video source (camera index, path, URL)")
    parser.add_argument("--seg-model", type=str, default="models/RailDrishti_Seg_Universal.pth", help="BiSeNetV2 weights")
    parser.add_argument("--det-model", type=str, default="models/best_yolo11m_raildrishti.pt", help="YOLO11m weights")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Obstacle detection resolution")
    parser.add_argument("--weather", type=str, default="auto", choices=["auto", "clahe", "dcp", "rain", "off"], help="Weather optimizer")
    parser.add_argument("--sensor", type=str, default="DAYLIGHT RGB", help="Sensor label")
    parser.add_argument("--save", action="store_true", help="Save output video/images")
    parser.add_argument("--no-view", action="store_true", help="Headless mode without GUI")

    args = parser.parse_args()

    run_inference(
        source=args.source,
        seg_model_path=args.seg_model,
        det_model_path=args.det_model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        weather_mode=args.weather,
        sensor_type=args.sensor,
        save_output=args.save,
        show_view=not args.no_view
    )
