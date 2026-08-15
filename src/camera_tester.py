"""
Drishti Kavach: Camera Diagnostics & Live Stream Snapshot Tool

Features:
  - Scans and lists all connected USB cameras / UVC video capture devices.
  - Opens real-time 1080p live stream preview.
  - Interactive snapshot capture with on-screen flash confirmation.
  - Dynamic camera index switching on the fly ([C] key).

CLI Flags:
  --cam INT       : Camera index to open (default: 0)
  --width INT     : Request width resolution (default: 1920)
  --height INT    : Request height resolution (default: 1080)
  --save-dir STR  : Output directory for snapshots (default: outputs/snapshots)
  --scan          : Scan and list available cameras only

Keyboard Controls:
  [S] / [SPACE]   : Capture high-resolution snapshot
  [C]             : Switch to next available camera index
  [F]             : Toggle Fullscreen
  [Q] / [ESC]     : Exit tool
"""

import os
import sys
import time
import argparse
from datetime import datetime
import cv2
import numpy as np


def scan_available_cameras(max_tested: int = 6) -> list:
    """Probes video capture indices to detect available connected cameras."""
    available = []
    print("\nScanning for connected camera devices...")
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                available.append({
                    "index": idx,
                    "width": w,
                    "height": h,
                    "fps": fps if fps > 0 else 30.0
                })
            cap.release()
    return available


def open_camera(cam_idx: int, req_w: int = 1920, req_h: int = 1080):
    """Opens camera and requests specific resolution."""
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return cap


def main():
    parser = argparse.ArgumentParser(description="Drishti Kavach Camera Diagnostics & Snapshot Tool")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1920, help="Requested width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Requested height (default: 1080)")
    parser.add_argument("--save-dir", type=str, default="outputs/snapshots", help="Directory to save snapshots")
    parser.add_argument("--scan", action="store_true", help="Scan and list connected cameras only")
    args = parser.parse_args()

    # 1. Scan available devices
    available_cams = scan_available_cameras()
    if not available_cams:
        print("\n[!] Warning: No active camera devices found. Ensure webcam/USB camera is plugged in.")
        if args.scan:
            return
    else:
        print("\n" + "=" * 55)
        print(" DETECTED CAMERA DEVICES")
        print("=" * 55)
        for c in available_cams:
            print(f" • Camera Index [{c['index']}]: Resolution {c['width']}x{c['height']} @ {c['fps']:.0f} FPS")
        print("=" * 55)

    if args.scan:
        return

    # Ensure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    current_idx = args.cam
    cap = open_camera(current_idx, args.width, args.height)

    if cap is None or not cap.isOpened():
        # If requested index failed, try first available
        if available_cams:
            current_idx = available_cams[0]["index"]
            print(f"[!] Camera index {args.cam} unavailable, falling back to Camera [{current_idx}]...")
            cap = open_camera(current_idx, args.width, args.height)

    if cap is None or not cap.isOpened():
        print(f"\n[ERROR] Unable to open video capture for camera index {current_idx}.")
        sys.exit(1)

    win_name = "Drishti Kavach - Camera Diagnostics & Snapshot Tool"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(f"\n[+] Live stream active on Camera [{current_idx}]")
    print(" Controls: [S] or [SPACE] Snapshot  |  [C] Switch Camera  |  [F] Fullscreen  |  [Q] Quit\n")

    prev_time = time.time()
    fps_smooth = 0.0
    snapshot_count = 0
    flash_frames = 0
    last_saved_path = ""
    is_fullscreen = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Brief wait and retry
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]

            # Calculate live FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * cur_fps if fps_smooth > 0 else cur_fps

            display = frame.copy()

            # Optional snapshot white flash effect
            if flash_frames > 0:
                cv2.rectangle(display, (0, 0), (w, h), (255, 255, 255), -1)
                flash_frames -= 1

            # Top Diagnostics Banner (Old-School Charcoal Strip)
            bar_h = 50
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 12, 16), -1)
            cv2.line(overlay, (0, bar_h), (w, bar_h), (50, 65, 80), 1)
            cv2.addWeighted(overlay, 0.88, display, 0.12, 0, display)

            # Left: Camera info
            cv2.circle(display, (20, 25), 5, (0, 255, 100), -1, cv2.LINE_AA)
            cv2.putText(display, f"CAMERA [{current_idx}] LIVE", (34, 31),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 235, 255), 1, cv2.LINE_AA)

            # Center: Resolution & FPS
            center_info = f"RESOLUTION: {w}x{h}  |  FPS: {fps_smooth:4.1f}"
            (tw, th), _ = cv2.getTextSize(center_info, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.putText(display, center_info, ((w - tw) // 2, 31),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 245, 250), 1, cv2.LINE_AA)

            # Right: Snapshots count
            snap_str = f"CAPTURED: {snapshot_count}"
            (sw, sh), _ = cv2.getTextSize(snap_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.putText(display, snap_str, (w - sw - 20, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 150), 1, cv2.LINE_AA)

            # Bottom Controls Footer
            foot_h = 32
            foot_y1 = h - foot_h
            foot_overlay = display.copy()
            cv2.rectangle(foot_overlay, (0, foot_y1), (w, h), (10, 12, 16), -1)
            cv2.line(foot_overlay, (0, foot_y1), (w, foot_y1), (40, 50, 65), 1)
            cv2.addWeighted(foot_overlay, 0.85, display, 0.15, 0, display)

            controls_text = "[S] Snapshot  |  [C] Switch Camera  |  [F] Fullscreen  |  [Q/ESC] Quit"
            if last_saved_path:
                controls_text = f"SAVED: {os.path.basename(last_saved_path)}  |  " + controls_text

            cv2.putText(display, controls_text, (20, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 220), 1, cv2.LINE_AA)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF

            # [Q] or [ESC]: Quit
            if key in (ord('q'), ord('Q'), 27):
                break

            # [S] or [SPACE]: Snapshot
            elif key in (ord('s'), ord('S'), 32):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename = f"snapshot_cam{current_idx}_{ts}.jpg"
                save_path = os.path.join(args.save_dir, filename)
                
                # Save the uncompressed raw frame
                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
                snapshot_count += 1
                flash_frames = 2
                last_saved_path = save_path
                print(f"[+] Snapshot saved [{snapshot_count}]: {save_path}")

            # [C]: Cycle to next camera
            elif key in (ord('c'), ord('C')):
                if len(available_cams) > 1:
                    indices = [c["index"] for c in available_cams]
                    curr_pos = indices.index(current_idx) if current_idx in indices else 0
                    next_idx = indices[(curr_pos + 1) % len(indices)]
                    
                    print(f"\nSwitching from Camera [{current_idx}] to Camera [{next_idx}]...")
                    cap.release()
                    current_idx = next_idx
                    cap = open_camera(current_idx, args.width, args.height)
                    if cap is None or not cap.isOpened():
                        print(f"[!] Failed to open Camera [{next_idx}], reverting...")
                        current_idx = indices[curr_pos]
                        cap = open_camera(current_idx, args.width, args.height)
                else:
                    print("[!] Only one camera detected.")

            # [F]: Toggle Fullscreen
            elif key in (ord('f'), ord('F')):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, prop)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\n[+] Camera feed closed gracefully.")


if __name__ == "__main__":
    main()
