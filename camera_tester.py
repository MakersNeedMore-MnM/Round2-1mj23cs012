"""
Drishti Kavach: Fast Cross-Platform Camera Live Stream & Snapshot Tool

Features:
  - Universal support (macOS AVFoundation, Windows DirectShow/MSMF, Linux V4L2).
  - Displays actual camera device names (e.g. MacBook Air Camera, Logitech HD Webcam, USB Video).
  - Prompts camera selection if multiple cameras are connected.
  - Probes and presents available supported resolutions (with Drishti Kavach 1080p recommendation).
  - Zero-latency frame buffer (no lag or motion stutter).
  - Minimalist bottom-right status badge.
  - Saves high-res snapshots to 'camera_captures/' folder.

Keyboard Controls:
  [S] / [SPACE] : Capture snapshot to camera_captures/
  [Q] / [ESC]   : Exit (or click window close [X])
"""

import os
import sys
import glob
import time
import json
import argparse
import platform
import subprocess
from datetime import datetime
from contextlib import contextmanager

# Suppress low-level OpenCV C++ stderr probe spam
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:
    pass


@contextmanager
def suppress_c_stderr():
    """Temporarily suppresses low-level C-library stderr output during device probing."""
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        old_stderr = os.dup(2)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass


def get_platform_backends():
    """Returns prioritized OpenCV VideoCapture backends based on operating system."""
    sys_name = platform.system().lower()
    if "windows" in sys_name or sys.platform.startswith("win"):
        return [
            ("DirectShow", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("Default", cv2.CAP_ANY)
        ]
    elif "darwin" in sys_name or sys.platform == "darwin":
        return [
            ("AVFoundation", cv2.CAP_AVFOUNDATION),
            ("Default", cv2.CAP_ANY)
        ]
    elif "linux" in sys_name or sys.platform.startswith("linux"):
        return [
            ("V4L2", cv2.CAP_V4L2),
            ("Default", cv2.CAP_ANY)
        ]
    return [("Default", cv2.CAP_ANY)]


def get_system_camera_names() -> list:
    """
    Detects hardware camera device names across macOS, Windows, and Linux.
    """
    names = []
    sys_name = platform.system().lower()

    if "darwin" in sys_name or sys.platform == "darwin":
        try:
            res = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if res.returncode == 0:
                data = json.loads(res.stdout)
                cams = data.get("SPCameraDataType", [])
                for c in cams:
                    c_name = c.get("_name") or c.get("spcamera_model-id") or "Camera"
                    names.append(c_name)
        except Exception:
            pass

    elif "windows" in sys_name or sys.platform.startswith("win"):
        try:
            cmd = [
                "powershell", "-NoProfile", "-Command",
                "Get-PnpDevice -Class Camera,Image -Status OK | Select-Object -ExpandProperty FriendlyName"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.5)
            if res.returncode == 0:
                for line in res.stdout.strip().splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        names.append(cleaned)
        except Exception:
            pass

    elif "linux" in sys_name or sys.platform.startswith("linux"):
        try:
            for p in sorted(glob.glob("/sys/class/video4linux/video*/name")):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    c_name = f.read().strip()
                    if c_name and c_name not in names:
                        names.append(c_name)
        except Exception:
            pass

    return names


def try_open_cap(cam_idx: int):
    """Tries opening a camera using platform backends with zero-latency buffer."""
    backends = get_platform_backends()
    for b_name, b_flag in backends:
        try:
            with suppress_c_stderr():
                cap = cv2.VideoCapture(cam_idx) if b_flag == cv2.CAP_ANY else cv2.VideoCapture(cam_idx, b_flag)

            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap, b_name
        except Exception:
            continue
    return None, None


def scan_connected_cameras(max_tested: int = 6) -> list:
    """Quickly scans active camera indices and links them with hardware names."""
    found = []
    sys_names = get_system_camera_names()

    for idx in range(max_tested):
        cap, b_name = try_open_cap(idx)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                # Assign system name if available
                if idx < len(sys_names):
                    device_name = sys_names[idx]
                elif len(sys_names) == 1 and idx == 0:
                    device_name = sys_names[0]
                else:
                    device_name = f"Camera Device {idx}"

                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                found.append({
                    "index": idx,
                    "name": device_name,
                    "probe_w": w,
                    "probe_h": h,
                    "backend": b_name
                })
            cap.release()

    return found


def probe_supported_resolutions(cam_idx: int) -> list:
    """
    Tests standard resolutions on the camera to see which ones are natively supported.
    Returns list of dicts: [{'width': w, 'height': h, 'label': str, 'is_project_default': bool}]
    """
    candidates = [
        (1920, 1080, "1080p Full HD (Recommended for Drishti Kavach)"),
        (3840, 2160, "4K UHD"),
        (2560, 1440, "2K QHD"),
        (1600, 900,  "900p HD+"),
        (1280, 960,  "1280x960 4:3"),
        (1280, 720,  "720p HD Standard"),
        (1024, 768,  "1024x768 4:3"),
        (800, 600,   "800x600 SVGA"),
        (640, 480,   "480p SD Standard")
    ]

    cap, _ = try_open_cap(cam_idx)
    if cap is None:
        return []

    supported = []
    seen = set()

    for req_w, req_h, desc in candidates:
        try:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                act_h, act_w = frame.shape[:2]
                res_key = (act_w, act_h)

                if res_key not in seen:
                    seen.add(res_key)
                    is_proj = (act_w == 1920 and act_h == 1080)
                    label = desc if (act_w == req_w and act_h == req_h) else f"{act_w}x{act_h}"
                    supported.append({
                        "width": act_w,
                        "height": act_h,
                        "label": label,
                        "is_project_default": is_proj
                    })
        except Exception:
            continue

    cap.release()

    # Sort descending by pixel count (width * height)
    supported.sort(key=lambda r: r["width"] * r["height"], reverse=True)

    # If project default (1920x1080) is supported, move it to the very top as recommended option
    proj_idx = next((i for i, r in enumerate(supported) if r["is_project_default"]), None)
    if proj_idx is not None:
        proj_item = supported.pop(proj_idx)
        supported.insert(0, proj_item)

    return supported


def select_resolution_interactive(cam_idx: int, cam_name: str) -> tuple:
    """
    Probes camera for supported resolutions and prompts the user to select one.
    Returns (width, height).
    """
    print(f"\n[•] Probing supported resolutions for [{cam_name}]...")
    resolutions = probe_supported_resolutions(cam_idx)

    print("\n" + "=" * 65)
    print(f" AVAILABLE RESOLUTIONS — Camera [{cam_idx}]: {cam_name}")
    print("=" * 65)

    default_option = 1
    has_project_default = False

    for i, res in enumerate(resolutions, start=1):
        w, h = res["width"], res["height"]
        if res["is_project_default"]:
            has_project_default = True
            tag = " ★ [RECOMMENDED / DEFAULT]"
        elif i == 1 and not has_project_default:
            tag = " [MAX NATIVE / DEFAULT]"
        else:
            tag = ""
        print(f"  [{i}] {w}x{h} - {res['label']}{tag}")

    custom_opt = len(resolutions) + 1
    print(f"  [{custom_opt}] Custom resolution (enter manual width x height)")
    print("=" * 65)

    while True:
        choice = input(f"Select resolution (1-{custom_opt}) [default: {default_option}]: ").strip()
        if choice == "":
            selected = resolutions[default_option - 1]
            return selected["width"], selected["height"]
        elif choice.isdigit():
            c_int = int(choice)
            if 1 <= c_int <= len(resolutions):
                selected = resolutions[c_int - 1]
                return selected["width"], selected["height"]
            elif c_int == custom_opt:
                try:
                    custom_input = input("Enter custom resolution as WIDTHxHEIGHT (e.g. 1920x1080): ").strip()
                    parts = custom_input.lower().replace(" ", "").split("x")
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return int(parts[0]), int(parts[1])
                except Exception:
                    pass
                print("[!] Invalid custom resolution format. Re-trying...")
        print(f"[!] Please enter a number between 1 and {custom_opt}.")


def open_camera_with_resolution(cam_idx: int, req_w: int, req_h: int):
    """Opens camera and sets requested resolution with buffer size = 1."""
    cap, b_name = try_open_cap(cam_idx)
    if cap is None or not cap.isOpened():
        return None, 0, 0

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        act_h, act_w = frame.shape[:2]
        return cap, act_w, act_h

    # Retry read if buffer initializing
    time.sleep(0.05)
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        act_h, act_w = frame.shape[:2]
        return cap, act_w, act_h

    return cap, req_w, req_h


def main():
    parser = argparse.ArgumentParser(description="Drishti Kavach - Fast Camera Stream & Snapshot Tool")
    parser.add_argument("--cam", type=int, default=None, help="Directly specify camera index (optional)")
    parser.add_argument("--width", type=int, default=None, help="Directly specify width (optional)")
    parser.add_argument("--height", type=int, default=None, help="Directly specify height (optional)")
    parser.add_argument("--save-dir", type=str, default="camera_captures", help="Folder to save snapshots")
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("\nScanning for connected camera devices...")
    detected = scan_connected_cameras()

    if not detected:
        print("\n[!] No active camera devices found.")
        print("    - Check if webcam / USB video capture card is connected.")
        print("    - On macOS: Ensure Terminal / IDE has Camera permission in System Settings -> Privacy & Security -> Camera.")
        print("    - On Windows: Ensure Camera privacy settings allow desktop apps to access the camera.")
        sys.exit(1)

    selected_cam = args.cam
    selected_name = "Camera"

    if selected_cam is not None:
        # Match name if possible
        matched = next((c for c in detected if c["index"] == selected_cam), None)
        selected_name = matched["name"] if matched else f"Camera [{selected_cam}]"
    else:
        if len(detected) == 1:
            selected_cam = detected[0]["index"]
            selected_name = detected[0]["name"]
            print(f"[+] 1 Camera detected: [{selected_cam}] {selected_name}")
        else:
            print("\n" + "=" * 55)
            print(f" CONNECTED CAMERAS DETECTED ({len(detected)})")
            print("=" * 55)
            for c in detected:
                print(f"  [{c['index']}] {c['name']} (Index {c['index']})")
            print("=" * 55)

            valid_indices = [c["index"] for c in detected]
            while True:
                user_choice = input(f"Select camera index ({'/'.join(map(str, valid_indices))}) [default: {valid_indices[0]}]: ").strip()
                if user_choice == "":
                    selected_cam = valid_indices[0]
                    break
                elif user_choice.isdigit() and int(user_choice) in valid_indices:
                    selected_cam = int(user_choice)
                    break
                print(f"[!] Invalid choice. Please enter one of {valid_indices}.")

            matched = next(c for c in detected if c["index"] == selected_cam)
            selected_name = matched["name"]

    # Resolution selection
    if args.width is not None and args.height is not None:
        target_w, target_h = args.width, args.height
    else:
        target_w, target_h = select_resolution_interactive(selected_cam, selected_name)

    print(f"\n[+] Opening [{selected_name}] (Index {selected_cam}) @ {target_w}x{target_h}...")
    cap, res_w, res_h = open_camera_with_resolution(selected_cam, target_w, target_h)

    if cap is None or not cap.isOpened():
        print(f"\n[ERROR] Unable to open Camera [{selected_cam}]: {selected_name}.")
        sys.exit(1)

    print(f"[+] Live stream active: {res_w}x{res_h}")
    print(f"[+] Snapshots will be saved to: {save_dir}")
    print("\nControls: [S] or [SPACE] Snapshot  |  [Q] or [ESC] Quit\n")

    win_name = f"Camera {selected_cam}: {selected_name} ({res_w}x{res_h})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    fps_smooth = 0.0
    prev_time = time.time()
    saved_count = 0
    flash_frames = 0
    status_msg = ""
    status_time = 0.0

    try:
        while True:
            # Check window close [X] button
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Fast FPS measurement
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * cur_fps if fps_smooth > 0 else cur_fps

            # Snapshot flash effect
            if flash_frames > 0:
                frame[:] = 255
                flash_frames -= 1
                cv2.imshow(win_name, frame)
                cv2.waitKey(1)
                continue

            # Bottom-right minimal info badge
            if status_msg and (curr_time - status_time < 2.5):
                info_text = status_msg
            else:
                info_text = f"CAM {selected_cam} | {w}x{h} | {fps_smooth:4.1f} FPS | SAVED: {saved_count}"

            # Calculate text position in bottom-right corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.48
            thickness = 1
            (tw, th), _ = cv2.getTextSize(info_text, font, font_scale, thickness)

            margin = 12
            bx2 = w - margin
            bx1 = bx2 - tw - 16
            by2 = h - margin
            by1 = by2 - th - 12

            if bx1 >= 0 and by1 >= 0:
                badge_sub = frame[by1:by2, bx1:bx2]
                dark_bg = np.zeros_like(badge_sub)
                cv2.addWeighted(badge_sub, 0.25, dark_bg, 0.75, 0, badge_sub)

                text_x = bx1 + 8
                text_y = by2 - 6
                text_color = (0, 255, 170) if "SAVED" in info_text else (230, 235, 240)
                cv2.putText(frame, info_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # [Q] or [ESC] -> Quit
            if key in (ord('q'), ord('Q'), 27):
                break

            # [S] or [SPACE] -> Snapshot
            elif key in (ord('s'), ord('S'), 32):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename = f"snapshot_cam{selected_cam}_{timestamp}.jpg"
                save_path = os.path.join(save_dir, filename)

                cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
                saved_count += 1
                flash_frames = 1
                status_msg = f"SAVED #{saved_count} ({filename})"
                status_time = curr_time
                print(f"[+] Saved #{saved_count}: {save_path}")

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"\n[+] Stream closed. {saved_count} snapshot(s) saved in '{save_dir}'.\n")


if __name__ == "__main__":
    main()

