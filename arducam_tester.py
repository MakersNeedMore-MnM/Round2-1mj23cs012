"""
Drishti Kavach: Arducam 0506 Day/Night 1080p Camera Testing & Calibration Utility

Specifically designed for testing, focusing, and tuning the Arducam 0506 (UC-0506)
1080p Day/Night IR-CUT camera module.

Features:
  - 1080p Full HD (1920x1080) native streaming with zero-latency buffer.
  - Interactive Focus Tuning:
      * Motorized/Hardware Focus control (CAP_PROP_FOCUS).
      * Focus Peaking mode: Highlights sharp optical edges in neon green.
      * Real-Time Laplacian Sharpness Meter: Accurately measure optical clarity.
  - Lighting & Sensor Controls:
      * Brightness, Contrast, Saturation, Sharpness, Exposure, Gain.
      * Hardware UVC register control + seamless software-enhanced fallback.
  - Day / Night Vision Modes:
      * Day Color Mode.
      * Night IR Monochrome (Grayscale + Adaptive Histogram Equalization).
  - Inspection Tools:
      * Multi-level Digital Center Zoom (1.0x - 4.0x) for pixel-level focus inspection.
      * Alignment Grid & Center Crosshair.
  - Clean Minimalist UI:
      * Bottom-right HUD clearly showing active key combinations and sensor status.
      * Interactive trackbars for mouse-based calibration.
      * High-resolution snapshot capture to 'camera_captures/arducam/'.

Controls:
  [F] / [f]     : Increase / Decrease Focus
  [B] / [b]     : Increase / Decrease Brightness
  [C] / [c]     : Increase / Decrease Contrast
  [K] / [k]     : Increase / Decrease Sharpness
  [U] / [u]     : Increase / Decrease Saturation
  [E] / [e]     : Increase / Decrease Exposure
  [A]           : Toggle Auto Focus (if supported)
  [N]           : Toggle Day / Night Vision Mode
  [P]           : Toggle Focus Peaking Mode (Green Edge Overlay)
  [Z] / [z]     : Zoom In / Zoom Out (Center Crop Inspection)
  [X]           : Toggle Alignment Grid & Crosshair
  [T]           : Toggle OpenCV Trackbar Control Panel
  [R]           : Reset all settings to defaults
  [H]           : Toggle Full On-Screen Help Overlay
  [S] / [SPACE] : Save high-res snapshot
  [Q] / [ESC]   : Exit preview
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
    """Detects hardware camera device names across macOS, Windows, and Linux."""
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
    """Opens a camera using preferred platform backend with zero-latency buffer."""
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
    """Scans active camera devices and flags any Arducam hardware."""
    found = []
    sys_names = get_system_camera_names()

    for idx in range(max_tested):
        cap, b_name = try_open_cap(idx)
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                if idx < len(sys_names):
                    device_name = sys_names[idx]
                elif len(sys_names) == 1 and idx == 0:
                    device_name = sys_names[0]
                else:
                    device_name = f"Camera Device {idx}"

                is_arducam = any(kw in device_name.lower() for kw in ["arducam", "0506", "uc-0506", "ir-cut"])
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                found.append({
                    "index": idx,
                    "name": device_name,
                    "is_arducam": is_arducam,
                    "probe_w": w,
                    "probe_h": h,
                    "backend": b_name
                })
            cap.release()

    return found


class ArducamController:
    """
    Manages camera properties with dual-layer control:
    Direct hardware UVC register control with fallback to software processing.
    """

    def __init__(self, cap):
        self.cap = cap

        # Control States (Normalized / Physical values)
        self.focus = 50          # 0 to 100
        self.autofocus = 0       # 0 = Manual, 1 = Auto
        self.brightness = 0      # -100 to +100 (0 = neutral)
        self.contrast = 1.0      # 0.2 to 3.0 (1.0 = neutral)
        self.saturation = 1.0    # 0.0 to 2.0 (1.0 = neutral)
        self.sharpness = 0       # 0 to 10 (0 = neutral)
        self.exposure = 0        # -10 to +10 (0 = neutral)
        self.auto_exposure = 1   # 1 = Auto, 0 = Manual
        self.gain = 0            # 0 to 100

        # View Modes
        self.night_mode = 0      # 0 = Day (Color), 1 = Night (IR Monochrome), 2 = Night (Enhanced IR)
        self.focus_peaking = False
        self.grid_overlay = False
        self.zoom_level = 1.0    # 1.0, 1.5, 2.0, 3.0, 4.0
        self.show_help = False
        self.show_trackbars = False

        self.last_status_msg = ""
        self.status_timestamp = 0.0

        self._init_hardware_properties()

    def _init_hardware_properties(self):
        """Attempts to read initial hardware values from camera if supported."""
        if self.cap is None:
            return
        try:
            hw_f = self.cap.get(cv2.CAP_PROP_FOCUS)
            if hw_f >= 0:
                self.focus = int(hw_f)
        except Exception:
            pass

    def set_status(self, msg: str):
        """Sets a temporary status toast on the HUD."""
        self.last_status_msg = msg
        self.status_timestamp = time.time()

    def reset_defaults(self):
        """Resets all settings back to default."""
        self.focus = 50
        self.autofocus = 0
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.sharpness = 0
        self.exposure = 0
        self.auto_exposure = 1
        self.gain = 0
        self.night_mode = 0
        self.focus_peaking = False
        self.grid_overlay = False
        self.zoom_level = 1.0
        self.sync_hardware_properties()
        self.set_status("All Settings Reset to Defaults")

    def sync_hardware_properties(self):
        """Syncs current state with hardware UVC registers."""
        if self.cap is None:
            return
        try:
            self.cap.set(cv2.CAP_PROP_FOCUS, float(self.focus))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, float(self.autofocus))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(self.brightness))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_CONTRAST, float(self.contrast))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_SATURATION, float(self.saturation))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_SHARPNESS, float(self.sharpness))
        except Exception:
            pass

    def adjust_focus(self, delta: int):
        self.focus = max(0, min(100, self.focus + delta))
        try:
            self.cap.set(cv2.CAP_PROP_FOCUS, float(self.focus))
        except Exception:
            pass
        self.set_status(f"Focus: {self.focus}%")

    def toggle_autofocus(self):
        self.autofocus = 1 - self.autofocus
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, float(self.autofocus))
        except Exception:
            pass
        state_str = "ON" if self.autofocus == 1 else "OFF (Manual)"
        self.set_status(f"Auto Focus: {state_str}")

    def adjust_brightness(self, delta: int):
        self.brightness = max(-100, min(100, self.brightness + delta))
        try:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(self.brightness))
        except Exception:
            pass
        self.set_status(f"Brightness: {self.brightness:+d}")

    def adjust_contrast(self, delta: float):
        self.contrast = max(0.2, min(3.0, round(self.contrast + delta, 2)))
        try:
            self.cap.set(cv2.CAP_PROP_CONTRAST, float(self.contrast))
        except Exception:
            pass
        self.set_status(f"Contrast: {self.contrast:.1f}x")

    def adjust_saturation(self, delta: float):
        self.saturation = max(0.0, min(2.5, round(self.saturation + delta, 2)))
        try:
            self.cap.set(cv2.CAP_PROP_SATURATION, float(self.saturation))
        except Exception:
            pass
        self.set_status(f"Saturation: {self.saturation:.1f}x")

    def adjust_sharpness(self, delta: int):
        self.sharpness = max(0, min(10, self.sharpness + delta))
        try:
            self.cap.set(cv2.CAP_PROP_SHARPNESS, float(self.sharpness))
        except Exception:
            pass
        self.set_status(f"Sharpness: {self.sharpness}")

    def adjust_exposure(self, delta: int):
        self.exposure = max(-10, min(10, self.exposure + delta))
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
        except Exception:
            pass
        self.set_status(f"Exposure: {self.exposure:+d}")

    def toggle_night_mode(self):
        self.night_mode = (self.night_mode + 1) % 3
        modes = ["Day (Color)", "Night (IR Mono)", "Night (Enhanced IR)"]
        self.set_status(f"Mode: {modes[self.night_mode]}")

    def toggle_peaking(self):
        self.focus_peaking = not self.focus_peaking
        state = "ON (Green Edges)" if self.focus_peaking else "OFF"
        self.set_status(f"Focus Peaking: {state}")

    def toggle_grid(self):
        self.grid_overlay = not self.grid_overlay
        state = "ON" if self.grid_overlay else "OFF"
        self.set_status(f"Alignment Grid: {state}")

    def cycle_zoom(self, direction: int = 1):
        zoom_steps = [1.0, 1.5, 2.0, 3.0, 4.0]
        curr_idx = zoom_steps.index(self.zoom_level) if self.zoom_level in zoom_steps else 0
        new_idx = max(0, min(len(zoom_steps) - 1, curr_idx + direction))
        self.zoom_level = zoom_steps[new_idx]
        self.set_status(f"Zoom: {self.zoom_level:.1f}x (Center Crop)")

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Applies software pipeline: Zoom -> Day/Night Mode -> Brightness/Contrast/Saturation
        -> Sharpness -> Focus Metric calculation -> Peaking Overlay -> Grid.
        Returns: (processed_frame, sharpness_score)
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Digital Zoom (Center ROI crop)
        if self.zoom_level > 1.0:
            crop_w = int(w / self.zoom_level)
            crop_h = int(h / self.zoom_level)
            x1 = (w - crop_w) // 2
            y1 = (h - crop_h) // 2
            cropped = out[y1:y1 + crop_h, x1:x1 + crop_w]
            out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # 2. Day / Night IR-CUT Simulation / Enhancement
        if self.night_mode == 1:
            # IR Monochrome
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif self.night_mode == 2:
            # Enhanced IR with CLAHE (Contrast Limited Adaptive Histogram Equalization)
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            out = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # 3. Software Contrast & Brightness adjustment (alpha, beta)
        if self.contrast != 1.0 or self.brightness != 0:
            out = cv2.convertScaleAbs(out, alpha=self.contrast, beta=self.brightness)

        # 4. Software Saturation adjustment
        if self.saturation != 1.0 and self.night_mode == 0:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * self.saturation, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 5. Software Sharpness Unsharp Masking
        if self.sharpness > 0:
            gaussian = cv2.GaussianBlur(out, (0, 0), 2.0)
            weight = 0.2 * self.sharpness
            out = cv2.addWeighted(out, 1.0 + weight, gaussian, -weight, 0)

        # 6. Real-time Focus Sharpness Metric (Laplacian variance)
        gray_for_focus = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray_for_focus, cv2.CV_64F)
        sharpness_score = float(laplacian.var())

        # 7. Focus Peaking Overlay (Green highlights on sharp high-gradient edges)
        if self.focus_peaking:
            abs_lap = np.uint8(np.absolute(laplacian))
            # Dynamic threshold based on sharpness
            thresh_val = max(25, int(np.percentile(abs_lap, 92)))
            _, mask = cv2.threshold(abs_lap, thresh_val, 255, cv2.THRESH_BINARY)
            # Neon Green overlay
            green_layer = np.zeros_like(out)
            green_layer[:] = [0, 255, 50]
            edge_indices = mask > 0
            out[edge_indices] = cv2.addWeighted(out[edge_indices], 0.35, green_layer[edge_indices], 0.65, 0)

        # 8. Alignment Grid & Crosshairs
        if self.grid_overlay:
            # Rule of Thirds
            line_color = (0, 255, 255)
            alpha_grid = 0.4
            overlay = out.copy()
            cv2.line(overlay, (w // 3, 0), (w // 3, h), line_color, 1)
            cv2.line(overlay, (2 * w // 3, 0), (2 * w // 3, h), line_color, 1)
            cv2.line(overlay, (0, h // 3), (w, h // 3), line_color, 1)
            cv2.line(overlay, (0, 2 * h // 3), (w, 2 * h // 3), line_color, 1)
            # Center Crosshair
            cx, cy = w // 2, h // 2
            cv2.circle(overlay, (cx, cy), 24, (0, 220, 255), 1)
            cv2.line(overlay, (cx - 36, cy), (cx + 36, cy), (0, 220, 255), 1)
            cv2.line(overlay, (cx, cy - 36), (cx, cy + 36), (0, 220, 255), 1)
            cv2.addWeighted(overlay, alpha_grid, out, 1.0 - alpha_grid, 0, out)

        return out, sharpness_score


def draw_hud(frame: np.ndarray, ctrl: ArducamController, cam_idx: int, cam_name: str, fps: float, sharpness: float, saved_count: int):
    """
    Renders a sleek, modern, non-intrusive bottom-right HUD with active status
    and explicit key combinations, plus temporary toasts and optional help overlay.
    """
    h, w = frame.shape[:2]
    now = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Top-Left Live Status Tag
    mode_labels = ["DAY (Color)", "NIGHT (IR Mono)", "NIGHT (Enhanced IR)"]
    active_mode = mode_labels[ctrl.night_mode]

    tl_lines = [
        f"ARDUCAM 1080p | Cam [{cam_idx}]: {cam_name[:24]}",
        f"FPS: {fps:4.1f} | Sharpness: {sharpness:6.1f} | {active_mode}"
    ]
    if ctrl.zoom_level > 1.0:
        tl_lines[1] += f" | ZOOM {ctrl.zoom_level:.1f}x"
    if ctrl.focus_peaking:
        tl_lines[1] += " | [PEAKING ON]"

    # Draw Top-Left Translucent Bar
    pad_y = 12
    for line_idx, line_txt in enumerate(tl_lines):
        t_size, _ = cv2.getTextSize(line_txt, font, 0.44, 1)
        bx1, by1 = 12, pad_y + (line_idx * 22)
        bx2, by2 = bx1 + t_size[0] + 16, by1 + 20
        sub_roi = frame[by1:by2, bx1:bx2]
        if sub_roi.size > 0:
            dark_roi = np.zeros_like(sub_roi)
            cv2.addWeighted(sub_roi, 0.35, dark_roi, 0.65, 0, sub_roi)
            cv2.putText(frame, line_txt, (bx1 + 8, by2 - 5), font, 0.44, (235, 240, 245), 1, cv2.LINE_AA)

    # 2. Bottom-Right Key Combinations & Settings HUD
    hud_lines = [
        f"Focus: {ctrl.focus}% | Bright: {ctrl.brightness:+d} | Cont: {ctrl.contrast:.1f}x | Sharp: {ctrl.sharpness}",
        "[F/f] Focus   [B/b] Bright   [C/c] Contrast   [K/k] Sharp",
        "[N] Day/Night   [P] Peaking   [Z/z] Zoom   [X] Grid",
        "[S] Snapshot   [R] Reset   [T] Trackbars   [H] Help   [Q] Quit"
    ]

    # Check if there is an active temporary toast message
    if ctrl.last_status_msg and (now - ctrl.status_timestamp < 2.5):
        toast_text = f">> {ctrl.last_status_msg.upper()} <<"
        hud_lines.insert(0, toast_text)

    # Calculate required bounding box size
    max_w = 0
    line_h = 20
    for l in hud_lines:
        ts, _ = cv2.getTextSize(l, font, 0.42, 1)
        if ts[0] > max_w:
            max_w = ts[0]

    box_w = max_w + 24
    box_h = len(hud_lines) * line_h + 16
    margin = 12

    br_x2 = w - margin
    br_x1 = br_x2 - box_w
    br_y2 = h - margin
    br_y1 = br_y2 - box_h

    if br_x1 >= 0 and br_y1 >= 0:
        sub_br = frame[br_y1:br_y2, br_x1:br_x2]
        dark_bg = np.zeros_like(sub_br)
        cv2.addWeighted(sub_br, 0.20, dark_bg, 0.80, 0, sub_br)
        cv2.rectangle(frame, (br_x1, br_y1), (br_x2, br_y2), (70, 80, 95), 1)

        for i, l in enumerate(hud_lines):
            ty = br_y1 + 18 + (i * line_h)
            if ">>" in l:
                txt_color = (0, 255, 170)  # Neon Cyan/Green for active toast
            elif i == 0 and not (ctrl.last_status_msg and (now - ctrl.status_timestamp < 2.5)):
                txt_color = (0, 235, 255)  # Bright Yellow/Gold for current settings
            elif "[" in l:
                txt_color = (210, 220, 230)  # Clean soft white for keybindings
            else:
                txt_color = (180, 190, 200)

            cv2.putText(frame, l, (br_x1 + 12, ty), font, 0.42, txt_color, 1, cv2.LINE_AA)

    # 3. Full Help Overlay (when [H] is toggled)
    if ctrl.show_help:
        help_w, help_h = 580, 360
        hx1 = (w - help_w) // 2
        hy1 = (h - help_h) // 2
        hx2 = hx1 + help_w
        hy2 = hy1 + help_h

        if hx1 >= 0 and hy1 >= 0:
            help_roi = frame[hy1:hy2, hx1:hx2]
            dark_card = np.zeros_like(help_roi)
            cv2.addWeighted(help_roi, 0.15, dark_card, 0.85, 0, help_roi)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (0, 220, 255), 2)

            title = "ARDUCAM 0506 CONTROL SHORTCUTS"
            cv2.putText(frame, title, (hx1 + 20, hy1 + 35), font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            shortcuts = [
                ("[F] / [f]", "Increase / Decrease Focus Step (Motorized/UVC)"),
                ("[A]", "Toggle Auto Focus (if supported by hardware)"),
                ("[B] / [b]", "Increase / Decrease Brightness"),
                ("[C] / [c]", "Increase / Decrease Contrast"),
                ("[K] / [k]", "Increase / Decrease Sharpness (Unsharp Mask)"),
                ("[U] / [u]", "Increase / Decrease Saturation"),
                ("[E] / [e]", "Increase / Decrease Exposure"),
                ("[N]", "Toggle Day / Night IR Mode (Color -> Mono -> Enhanced)"),
                ("[P]", "Toggle Focus Peaking (Green High-Frequency Edges)"),
                ("[Z] / [z]", "Zoom In / Out (1.0x, 1.5x, 2.0x, 3.0x, 4.0x Center ROI)"),
                ("[X]", "Toggle Alignment Grid & Center Crosshair"),
                ("[T]", "Show / Hide OpenCV Trackbar Controls"),
                ("[R]", "Reset All Camera Settings to Defaults"),
                ("[S] / [SPACE]", f"Save High-Res Snapshot (Saved: {saved_count})"),
                ("[H]", "Close Help Overlay"),
                ("[Q] / [ESC]", "Exit Application")
            ]

            for s_idx, (k_cmd, desc) in enumerate(shortcuts):
                sy = hy1 + 65 + (s_idx * 18)
                cv2.putText(frame, k_cmd, (hx1 + 25, sy), font, 0.40, (0, 255, 180), 1, cv2.LINE_AA)
                cv2.putText(frame, f": {desc}", (hx1 + 130, sy), font, 0.40, (230, 235, 240), 1, cv2.LINE_AA)


def setup_trackbars(win_name: str, ctrl: ArducamController):
    """Creates interactive OpenCV trackbars for mouse-driven camera tuning."""
    def on_focus(val):
        ctrl.focus = val
        ctrl.sync_hardware_properties()

    def on_brightness(val):
        ctrl.brightness = val - 100  # Map 0..200 to -100..+100
        ctrl.sync_hardware_properties()

    def on_contrast(val):
        ctrl.contrast = max(0.2, val / 10.0)  # Map 2..30 to 0.2..3.0
        ctrl.sync_hardware_properties()

    def on_sharpness(val):
        ctrl.sharpness = val
        ctrl.sync_hardware_properties()

    def on_saturation(val):
        ctrl.saturation = val / 10.0  # Map 0..25 to 0.0..2.5
        ctrl.sync_hardware_properties()

    def on_night_mode(val):
        ctrl.night_mode = val % 3

    cv2.createTrackbar("Focus", win_name, ctrl.focus, 100, on_focus)
    cv2.createTrackbar("Brightness", win_name, ctrl.brightness + 100, 200, on_brightness)
    cv2.createTrackbar("Contrast (x10)", win_name, int(ctrl.contrast * 10), 30, on_contrast)
    cv2.createTrackbar("Sharpness", win_name, ctrl.sharpness, 10, on_sharpness)
    cv2.createTrackbar("Saturation (x10)", win_name, int(ctrl.saturation * 10), 25, on_saturation)
    cv2.createTrackbar("Day/Night Mode", win_name, ctrl.night_mode, 2, on_night_mode)


def open_arducam_stream(cam_idx: int, target_w: int = 1920, target_h: int = 1080):
    """Opens camera and configures MJPEG format at requested 1080p resolution."""
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)

    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        act_h, act_w = frame.shape[:2]
        return cap, act_w, act_h

    time.sleep(0.08)
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        act_h, act_w = frame.shape[:2]
        return cap, act_w, act_h

    return cap, target_w, target_h


def main():
    parser = argparse.ArgumentParser(
        description="Drishti Kavach - Arducam 0506 Day/Night 1080p Camera Testing Utility"
    )
    parser.add_argument("--cam", type=int, default=None, help="Camera device index (optional)")
    parser.add_argument("--width", type=int, default=1920, help="Stream width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Stream height (default: 1080)")
    parser.add_argument("--save-dir", type=str, default="camera_captures/arducam", help="Folder to save snapshots")
    parser.add_argument("--trackbars", action="store_true", help="Launch with trackbars visible immediately")
    args = parser.parse_args()

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print(" DRISHTI KAVACH — ARDUCAM 0506 DAY/NIGHT 1080p TESTER")
    print("=" * 65)
    print("[•] Scanning for connected camera devices...")

    detected = scan_connected_cameras()
    if not detected:
        print("\n[!] No active camera devices found.")
        print("    - Verify Arducam USB cable is firmly connected.")
        print("    - On macOS: Verify Camera permissions in System Settings -> Privacy & Security -> Camera.")
        print("    - On Linux: Ensure user has access to /dev/video* (sudo usermod -a -G video $USER).")
        sys.exit(1)

    selected_cam = args.cam
    selected_name = "Arducam Camera"

    if selected_cam is not None:
        matched = next((c for c in detected if c["index"] == selected_cam), None)
        selected_name = matched["name"] if matched else f"Camera [{selected_cam}]"
    else:
        # Check if an Arducam device was explicitly identified
        arducam_match = next((c for c in detected if c["is_arducam"]), None)

        if len(detected) == 1:
            selected_cam = detected[0]["index"]
            selected_name = detected[0]["name"]
            print(f"[+] 1 Camera detected: [{selected_cam}] {selected_name}")
        elif arducam_match is not None:
            selected_cam = arducam_match["index"]
            selected_name = arducam_match["name"]
            print(f"[+] Arducam detected automatically: [{selected_cam}] {selected_name}")
        else:
            print(f"\nConnected Cameras Detected ({len(detected)}):")
            for c in detected:
                tag = " ★ [ARDUCAM]" if c["is_arducam"] else ""
                print(f"  [{c['index']}] {c['name']}{tag}")

            valid_indices = [c["index"] for c in detected]
            default_choice = valid_indices[0]
            while True:
                user_choice = input(f"\nSelect camera index ({'/'.join(map(str, valid_indices))}) [default: {default_choice}]: ").strip()
                if user_choice == "":
                    selected_cam = default_choice
                    break
                elif user_choice.isdigit() and int(user_choice) in valid_indices:
                    selected_cam = int(user_choice)
                    break
                print(f"[!] Invalid selection. Please enter one of {valid_indices}.")

            matched = next(c for c in detected if c["index"] == selected_cam)
            selected_name = matched["name"]

    print(f"\n[+] Opening [{selected_name}] (Index {selected_cam}) @ {args.width}x{args.height}...")
    cap, res_w, res_h = open_arducam_stream(selected_cam, args.width, args.height)

    if cap is None or not cap.isOpened():
        print(f"\n[ERROR] Failed to open camera [{selected_cam}]: {selected_name}.")
        sys.exit(1)

    print(f"[+] Live stream active: {res_w}x{res_h}")
    print(f"[+] High-res snapshots directory: {save_dir}")
    print("\nKeyboard Controls:")
    print("  [F] / [f]     : Focus +/-")
    print("  [B] / [b]     : Brightness +/-")
    print("  [C] / [c]     : Contrast +/-")
    print("  [K] / [k]     : Sharpness +/-")
    print("  [N]           : Toggle Day / Night Vision Mode")
    print("  [P]           : Toggle Focus Peaking (Green Edge Assist)")
    print("  [Z] / [z]     : Digital Zoom In / Out")
    print("  [X]           : Alignment Grid & Crosshair")
    print("  [T]           : Toggle Trackbar Controls")
    print("  [R]           : Reset Settings")
    print("  [H]           : On-Screen Help Overlay")
    print("  [S] / [SPACE] : Capture Snapshot")
    print("  [Q] / [ESC]   : Exit")
    print("=" * 65 + "\n")

    win_name = f"Arducam 0506 Tester: [{selected_cam}] {selected_name} ({res_w}x{res_h})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    ctrl = ArducamController(cap)
    if args.trackbars:
        ctrl.show_trackbars = True
        setup_trackbars(win_name, ctrl)

    fps_smooth = 0.0
    prev_time = time.time()
    saved_count = 0
    flash_frames = 0

    try:
        while True:
            # Detect window close [X]
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                time.sleep(0.01)
                continue

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * cur_fps if fps_smooth > 0 else cur_fps

            # Snapshot flash animation
            if flash_frames > 0:
                raw_frame[:] = 255
                flash_frames -= 1
                cv2.imshow(win_name, raw_frame)
                cv2.waitKey(1)
                continue

            # Process frame with active image pipeline & compute sharpness
            processed_frame, sharpness_score = ctrl.process_frame(raw_frame)

            # Draw HUD with bottom-right key combinations
            draw_hud(processed_frame, ctrl, selected_cam, selected_name, fps_smooth, sharpness_score, saved_count)

            cv2.imshow(win_name, processed_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 255 or key == 0:
                continue

            # [Q] or [ESC] -> Quit
            if key in (ord('q'), ord('Q'), 27):
                break

            # [S] or [SPACE] -> Snapshot
            elif key in (ord('s'), ord('S'), 32):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename = f"arducam_{timestamp}.jpg"
                save_path = os.path.join(save_dir, filename)

                # Save the processed calibrated frame at highest JPEG quality
                cv2.imwrite(save_path, processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
                saved_count += 1
                flash_frames = 1
                ctrl.set_status(f"SAVED SNAPSHOT #{saved_count} ({filename})")
                print(f"[+] Saved snapshot #{saved_count}: {save_path}")

            # [F] / [f] -> Focus +/-
            elif key == ord('F'):
                ctrl.adjust_focus(+5)
            elif key == ord('f'):
                ctrl.adjust_focus(-5)

            # [A] / [a] -> Toggle Auto Focus
            elif key in (ord('a'), ord('A')):
                ctrl.toggle_autofocus()

            # [B] / [b] -> Brightness +/-
            elif key == ord('B'):
                ctrl.adjust_brightness(+10)
            elif key == ord('b'):
                ctrl.adjust_brightness(-10)

            # [C] / [c] -> Contrast +/-
            elif key == ord('C'):
                ctrl.adjust_contrast(+0.1)
            elif key == ord('c'):
                ctrl.adjust_contrast(-0.1)

            # [K] / [k] -> Sharpness +/-
            elif key == ord('K'):
                ctrl.adjust_sharpness(+1)
            elif key == ord('k'):
                ctrl.adjust_sharpness(-1)

            # [U] / [u] -> Saturation +/-
            elif key == ord('U'):
                ctrl.adjust_saturation(+0.1)
            elif key == ord('u'):
                ctrl.adjust_saturation(-0.1)

            # [E] / [e] -> Exposure +/-
            elif key == ord('E'):
                ctrl.adjust_exposure(+1)
            elif key == ord('e'):
                ctrl.adjust_exposure(-1)

            # [N] / [n] -> Toggle Day / Night IR Mode
            elif key in (ord('n'), ord('N')):
                ctrl.toggle_night_mode()

            # [P] / [p] -> Toggle Focus Peaking
            elif key in (ord('p'), ord('P')):
                ctrl.toggle_peaking()

            # [Z] / [z] -> Zoom In / Out
            elif key == ord('Z'):
                ctrl.cycle_zoom(+1)
            elif key == ord('z'):
                ctrl.cycle_zoom(-1)

            # [X] / [x] -> Toggle Grid / Crosshair
            elif key in (ord('x'), ord('X')):
                ctrl.toggle_grid()

            # [R] / [r] -> Reset
            elif key in (ord('r'), ord('R')):
                ctrl.reset_defaults()

            # [H] / [h] -> Toggle Help Overlay
            elif key in (ord('h'), ord('H')):
                ctrl.show_help = not ctrl.show_help

            # [T] / [t] -> Toggle Trackbars
            elif key in (ord('t'), ord('T')):
                ctrl.show_trackbars = not ctrl.show_trackbars
                if ctrl.show_trackbars:
                    setup_trackbars(win_name, ctrl)
                    ctrl.set_status("Trackbars Enabled")
                else:
                    ctrl.set_status("Trackbars Toggled (Restart with --trackbars if hidden)")

    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"\n[+] Arducam tester closed. {saved_count} snapshot(s) saved in '{save_dir}'.\n")


if __name__ == "__main__":
    main()
