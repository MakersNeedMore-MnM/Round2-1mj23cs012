"""
Drishti Kavach: Real-Time Railway Physical Obstacle & Track Clearance Inference Engine

CLI Flags & Usage:
  --source STR/INT : Input source: '0' (Webcam), 'video.mp4', 'image.jpg', 'dir/', 'rtsp://...' (default: 0)
  --model STR      : Path to trained weights (default: models/RailDrishti.pt)
  --conf FLOAT     : Confidence threshold (default: 0.35)
  --imgsz INT      : Inference resolution (default: 1024)
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
  # 1. Live Kreo Owl Lite / Arducam USB Webcam:
  python run_inference.py --source 0

  # 2. Process a Test Video with Auto-Defogging:
  python run_inference.py --source data/train_test.mp4 --weather auto --save

  # 3. Process Active IR Night Vision Stream:
  python run_inference.py --source 0 --sensor "850nm ACTIVE IR CCTV"
"""

import os
import sys
import time
import argparse
import glob
import cv2
import numpy as np

from src import DrishtiEngine


def run_inference(
    source: str = "0",
    model_path: str = "models/RailDrishti.pt",
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

    # Initialize Engine
    engine = DrishtiEngine(
        model_path=model_path,
        conf_thresh=conf_thresh,
        imgsz=imgsz,
        weather_mode=weather_mode
    )

    # Determine input type
    is_webcam = source.isdigit()
    is_image = False
    is_directory = False
    is_video = False

    if is_webcam:
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx)
        # Request 1080p full resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        is_video = True
    elif os.path.isdir(source):
        is_directory = True
    elif os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            is_image = True
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            is_video = True
            cap = cv2.VideoCapture(source)
    elif source.startswith("rtsp://") or source.startswith("http://"):
        is_video = True
        cap = cv2.VideoCapture(source)
    else:
        print(f"[!] Invalid source: {source}")
        return

    # 1. Handle Single Image
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[!] Could not read image: {source}")
            return
        
        rendered, status, hazards, telemetry = engine.process_frame(
            frame, sensor_type=sensor_type, show_hud=True
        )

        out_name = f"result_{os.path.basename(source)}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, rendered)
        print(f"\n[+] Status: {status} | Processed image saved to: {out_path}")

        if show_view:
            cv2.namedWindow("Drishti Kavach - Railway HUD", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Drishti Kavach - Railway HUD", 1280, 720)
            cv2.imshow("Drishti Kavach - Railway HUD", rendered)
            print("\n[Press any key in the window to exit...]")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Handle Image Directory
    if is_directory:
        img_files = sorted(
            glob.glob(os.path.join(source, "*.jpg")) +
            glob.glob(os.path.join(source, "*.jpeg")) +
            glob.glob(os.path.join(source, "*.png"))
        )
        print(f"\n[+] Processing {len(img_files)} images from: {source}")
        for img_p in img_files:
            frame = cv2.imread(img_p)
            if frame is None:
                continue
            rendered, status, hazards, telemetry = engine.process_frame(
                frame, sensor_type=sensor_type, show_hud=True
            )
            out_p = os.path.join(output_dir, f"result_{os.path.basename(img_p)}")
            cv2.imwrite(out_p, rendered)
            print(f"  [{status}] Saved: {out_p}")
        print(f"\n[+] All images processed and saved to: {output_dir}")
        return

    # 3. Handle Live Webcam / Video Stream
    if is_video:
        if not cap.isOpened():
            print(f"[!] Failed to open video source: {source}")
            return

        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if save_output:
            out_vid_path = os.path.join(output_dir, f"drishti_kavach_stream_{int(time.time())}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_vid_path, fourcc, fps_in, (w_in, h_in))
            print(f"[+] Recording stream output to: {out_vid_path}")

        print("\n" + "=" * 75)
        print(" 🚆 DRISHTI KAVACH LIVE STREAM ACTIVE")
        print(" Controls:")
        print("  • [Q] or [ESC] : Quit")
        print("  • [D]          : Cycle Defogger (Auto -> CLAHE -> DCP -> Off)")
        print("  • [H]          : Toggle HUD Telemetry Overlay")
        print("  • [S]          : Save Frame Snapshot")
        print("  • [SPACE]      : Pause / Resume")
        print("=" * 75 + "\n")

        weather_cycle = ["auto", "clahe", "dcp", "rain", "off"]
        weather_idx = weather_cycle.index(weather_mode) if weather_mode in weather_cycle else 0
        show_hud = True
        is_paused = False

        if show_view:
            cv2.namedWindow("Drishti Kavach - Railway HUD", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Drishti Kavach - Railway HUD", 1280, 720)

        while True:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("[+] End of video stream.")
                    break

                rendered, status, hazards, telemetry = engine.process_frame(
                    frame,
                    sensor_type=sensor_type,
                    show_hud=show_hud,
                    is_video_stream=True
                )

                if writer is not None:
                    writer.write(rendered)

            if show_view:
                cv2.imshow("Drishti Kavach - Railway HUD", rendered)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord('q'), ord('Q'), 27]:  # Quit
                    break
                elif key in [ord('d'), ord('D')]:    # Cycle Defogger
                    weather_idx = (weather_idx + 1) % len(weather_cycle)
                    engine.weather_mode = weather_cycle[weather_idx]
                    print(f"[*] Weather Enhancement set to: {engine.weather_mode.upper()}")
                elif key in [ord('h'), ord('H')]:    # Toggle HUD
                    show_hud = not show_hud
                elif key in [ord('s'), ord('S')]:    # Snapshot
                    snap_path = os.path.join(snapshot_dir, f"snapshot_{int(time.time()*1000)}.jpg")
                    cv2.imwrite(snap_path, rendered)
                    print(f"[+] Snapshot saved to: {snap_path}")
                elif key == 32:                      # Space (Pause/Resume)
                    is_paused = not is_paused

        cap.release()
        if writer is not None:
            writer.release()
        if show_view:
            cv2.destroyAllWindows()
        print("\n[+] Drishti Kavach Engine terminated cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drishti Kavach Real-Time Inference & Decision Engine")
    parser.add_argument("--source", type=str, default="0", help="Camera index ('0'), video file, image, or RTSP URL")
    parser.add_argument("--model", type=str, default="models/RailDrishti.pt", help="Path to trained RailDrishti model")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference resolution")
    parser.add_argument("--weather", type=str, default="auto", choices=["auto", "clahe", "dcp", "rain", "off"], help="Harsh weather mode")
    parser.add_argument("--sensor", type=str, default="DAYLIGHT RGB", help="Sensor label (e.g. 'DAYLIGHT RGB' or '850nm ACTIVE IR CCTV')")
    parser.add_argument("--save", action="store_true", help="Save output video / image results")
    parser.add_argument("--no-view", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    run_inference(
        source=args.source,
        model_path=args.model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        weather_mode=args.weather,
        sensor_type=args.sensor,
        save_output=args.save,
        show_view=not args.no_view
    )
