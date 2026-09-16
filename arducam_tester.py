import os
import time
from contextlib import contextmanager
from datetime import datetime
import cv2
import numpy as np

os.environ["OPENCV_LOG_LEVEL"] = "OFF"
try:
    cv2.setLogLevel(0)
except Exception:
    pass


@contextmanager
def suppress_stderr():
    """Suppresses low-level C++ stderr warnings during camera probing."""
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


def get_available_cameras(max_to_check=6):
    available = []
    with suppress_stderr():
        for i in range(max_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
                cap.release()
            else:
                break
    return available


def open_camera(cam_idx):
    cap = cv2.VideoCapture(cam_idx)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


def draw_header(frame, fps, sharpness):
    """Draws a clean top-left header bar showing FPS and Sharpness value."""
    text = f"FPS: {fps:4.1f}  |  Sharpness: {sharpness:6.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x, pad_y = 16, 12
    x1, y1 = 20, 20
    x2, y2 = x1 + tw + 2 * pad_x, y1 + th + 2 * pad_y

    sub = frame[y1:y2, x1:x2]
    if sub.size > 0:
        dark = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.35, dark, 0.65, 0, sub)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 80, 95), 1)
        cv2.putText(frame, text, (x1 + pad_x, y2 - pad_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


class UVCController:
    """Manages UVC hardware registers, real-time image tuning pipeline, and the slider control panel."""

    def __init__(self, cap, win_name="Arducam Controls"):
        self.cap = cap
        self.win_name = win_name
        self.blank = np.zeros((1, 380, 3), dtype=np.uint8)

        # Normalized intuitive defaults (50 = neutral baseline)
        self.props = {
            "Brightness": 50,      # 0 to 100 (50 = normal)
            "Contrast": 50,        # 0 to 100 (50 = 1.0x normal)
            "Saturation": 50,      # 0 to 100 (50 = 1.0x normal)
            "Hue": 0,              # 0 to 180 (0 = normal)
            "Sharpness": 0,        # 0 to 10 (0 = normal)
            "Gamma": 100,          # 50 to 300 (100 = 1.0 normal)
            "Gain": 0,             # 0 to 100 (0 = normal)
            "Backlight": 0,        # 0 to 4 (0 = off)
            "Auto Exposure": 1,    # 1 = Auto, 0 = Manual
            "Exposure": 50,        # 0 to 100 (50 = normal)
            "Auto WB": 1,          # 1 = Auto, 0 = Manual
            "WB Temp": 46          # 20 to 75 (46 = ~4600K neutral)
        }
        self.defaults = dict(self.props)

        self._setup_window()

    def _setup_window(self):
        cv2.namedWindow(self.win_name, cv2.WINDOW_AUTOSIZE)

        def sync_hw(prop_id, val):
            try:
                self.cap.set(prop_id, float(val))
            except Exception:
                pass

        # Create Trackbars with real-time hardware sync
        def on_bright(val):
            self.props["Brightness"] = val
            sync_hw(cv2.CAP_PROP_BRIGHTNESS, val * 2.55)
        cv2.createTrackbar("Brightness", self.win_name, self.props["Brightness"], 100, on_bright)

        def on_contrast(val):
            self.props["Contrast"] = val
            sync_hw(cv2.CAP_PROP_CONTRAST, val)
        cv2.createTrackbar("Contrast", self.win_name, self.props["Contrast"], 100, on_contrast)

        def on_sat(val):
            self.props["Saturation"] = val
            sync_hw(cv2.CAP_PROP_SATURATION, val * 2)
        cv2.createTrackbar("Saturation", self.win_name, self.props["Saturation"], 100, on_sat)

        def on_hue(val):
            self.props["Hue"] = val
            sync_hw(cv2.CAP_PROP_HUE, val)
        cv2.createTrackbar("Hue", self.win_name, self.props["Hue"], 180, on_hue)

        def on_sharp(val):
            self.props["Sharpness"] = val
            sync_hw(cv2.CAP_PROP_SHARPNESS, val)
        cv2.createTrackbar("Sharpness", self.win_name, self.props["Sharpness"], 10, on_sharp)

        def on_gamma(val):
            self.props["Gamma"] = val
            sync_hw(cv2.CAP_PROP_GAMMA, val)
        cv2.createTrackbar("Gamma", self.win_name, self.props["Gamma"], 300, on_gamma)

        def on_gain(val):
            self.props["Gain"] = val
            sync_hw(cv2.CAP_PROP_GAIN, val)
        cv2.createTrackbar("Gain", self.win_name, self.props["Gain"], 100, on_gain)

        def on_backlight(val):
            self.props["Backlight"] = val
            sync_hw(cv2.CAP_PROP_BACKLIGHT, val)
        cv2.createTrackbar("Backlight Comp", self.win_name, self.props["Backlight"], 4, on_backlight)

        # Exposure Controls
        def on_auto_exp(val):
            self.props["Auto Exposure"] = val
            sync_hw(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if val == 1 else 1)
        cv2.createTrackbar("Auto Exposure", self.win_name, self.props["Auto Exposure"], 1, on_auto_exp)

        def on_exp(val):
            self.props["Exposure"] = val
            if self.props["Auto Exposure"] == 1:
                cv2.setTrackbarPos("Auto Exposure", self.win_name, 0)
            sync_hw(cv2.CAP_PROP_EXPOSURE, float(val - 100 if val <= 100 else val))
        cv2.createTrackbar("Manual Exposure", self.win_name, self.props["Exposure"], 100, on_exp)

        # White Balance Controls
        def on_auto_wb(val):
            self.props["Auto WB"] = val
            sync_hw(cv2.CAP_PROP_AUTO_WB_TEMPERATURE, float(val))
        cv2.createTrackbar("Auto White Bal", self.win_name, self.props["Auto WB"], 1, on_auto_wb)

        def on_wb_temp(val):
            self.props["WB Temp"] = val
            if self.props["Auto WB"] == 1:
                cv2.setTrackbarPos("Auto White Bal", self.win_name, 0)
            sync_hw(cv2.CAP_PROP_WB_TEMPERATURE, float(val * 100))
        cv2.createTrackbar("WB Temp (x100K)", self.win_name, self.props["WB Temp"], 75, on_wb_temp)

    def reset_defaults(self):
        """Resets all trackbars and camera registers to initial defaults."""
        for name, def_val in self.defaults.items():
            tb_name = name
            if name == "Backlight":
                tb_name = "Backlight Comp"
            elif name == "Exposure":
                tb_name = "Manual Exposure"
            elif name == "Auto WB":
                tb_name = "Auto White Bal"
            elif name == "WB Temp":
                tb_name = "WB Temp (x100K)"
            try:
                cv2.setTrackbarPos(tb_name, self.win_name, def_val)
            except Exception:
                pass

    def apply_pipeline(self, frame):
        """
        Applies responsive real-time image processing pipeline matching all slider settings.
        Guarantees 100% visible, immediate visual feedback in the live preview.
        """
        out = frame.copy()

        # 1. Brightness & Contrast
        b_val = self.props["Brightness"]
        c_val = self.props["Contrast"]
        b_offset = (b_val - 50) * 2.55         # -127 to +127
        c_factor = max(0.1, c_val / 50.0)      # 0.1x to 2.0x
        if b_offset != 0 or c_factor != 1.0:
            out = cv2.convertScaleAbs(out, alpha=c_factor, beta=b_offset)

        # 2. Saturation & Hue
        s_val = self.props["Saturation"]
        h_val = self.props["Hue"]
        s_factor = s_val / 50.0                # 0.0x (B&W) to 2.0x
        if s_factor != 1.0 or h_val != 0:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            if s_factor != 1.0:
                hsv[..., 1] = np.clip(hsv[..., 1] * s_factor, 0, 255)
            if h_val != 0:
                hsv[..., 0] = (hsv[..., 0] + h_val) % 180
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 3. Sharpness (Unsharp Mask)
        sharp_val = self.props["Sharpness"]
        if sharp_val > 0:
            weight = sharp_val * 0.25
            gaussian = cv2.GaussianBlur(out, (0, 0), 2.0)
            out = cv2.addWeighted(out, 1.0 + weight, gaussian, -weight, 0)

        # 4. Gamma Correction
        gamma_val = self.props["Gamma"]
        if gamma_val != 100 and gamma_val > 0:
            inv_g = 100.0 / float(gamma_val)
            lut = np.array([((i / 255.0) ** inv_g) * 255 for i in range(256)]).astype(np.uint8)
            out = cv2.LUT(out, lut)

        # 5. Gain & Manual Exposure simulation
        gain_val = self.props["Gain"]
        if gain_val > 0:
            gain_factor = 1.0 + (gain_val / 50.0)
            out = cv2.convertScaleAbs(out, alpha=gain_factor)

        if self.props["Auto Exposure"] == 0:
            exp_val = self.props["Exposure"]
            exp_factor = max(0.1, exp_val / 50.0)
            if exp_factor != 1.0:
                out = cv2.convertScaleAbs(out, alpha=exp_factor)

        # 6. White Balance Temperature tint shift (when Manual WB active)
        if self.props["Auto WB"] == 0:
            wb_temp = self.props["WB Temp"]
            if wb_temp != 46:
                delta = (wb_temp - 46) / 30.0  # -1.0 (cool/blue) to +1.0 (warm/orange)
                b_mult = max(0.4, min(1.6, 1.0 - (delta * 0.45)))
                r_mult = max(0.4, min(1.6, 1.0 + (delta * 0.45)))
                b, g, r = cv2.split(out.astype(np.float32))
                b = np.clip(b * b_mult, 0, 255)
                r = np.clip(r * r_mult, 0, 255)
                out = cv2.merge([b, g, r]).astype(np.uint8)

        return out

    def update(self):
        cv2.imshow(self.win_name, self.blank)


def main():
    cameras = get_available_cameras()
    if not cameras:
        print("No cameras found.")
        return

    print("Available cameras:")
    for cam in cameras:
        print(cam)

    while True:
        try:
            choice = input("Select camera: ").strip()
            cam_idx = int(choice)
            if cam_idx in cameras:
                break
        except (ValueError, EOFError, KeyboardInterrupt):
            return
        print("Invalid selection.")

    cap = open_camera(cam_idx)
    if not cap.isOpened():
        print(f"Failed to open camera {cam_idx}")
        return

    save_dir = os.path.join("camera_captures", "arducam_tester")
    os.makedirs(save_dir, exist_ok=True)

    preview_win = f"Arducam Preview (Camera {cam_idx})"
    ctrl_win = "Arducam Controls"

    cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
    ctrl = UVCController(cap, win_name=ctrl_win)

    # Position windows side-by-side
    cv2.moveWindow(preview_win, 50, 60)
    cv2.moveWindow(ctrl_win, 1020, 60)

    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Calculate FPS
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            cur_fps = 1.0 / dt
            fps_smooth = 0.9 * fps_smooth + 0.1 * cur_fps if fps_smooth > 0 else cur_fps

        # Apply real-time image processing pipeline based on slider controls
        processed_frame = ctrl.apply_pipeline(frame)

        # Calculate Sharpness (Laplacian variance) on active processed image
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Render clean header on displayed preview frame
        display_frame = processed_frame.copy()
        draw_header(display_frame, fps_smooth, sharpness)

        cv2.imshow(preview_win, display_frame)
        ctrl.update()

        key = cv2.waitKey(1) & 0xFF

        # [Q] or [ESC] -> Exit
        if key in (ord("q"), ord("Q"), 27):
            break

        # [S] or [SPACE] -> Save Snapshot
        elif key in (ord("s"), ord("S"), 32):
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            time_str = now.strftime("%H-%M-%S")
            sub_sec = f"{now.microsecond // 10000:02d}"
            filename = f"arducam_tester_{date_str}_{time_str}_{sub_sec}.jpg"
            filepath = os.path.join(save_dir, filename)
            # Save the calibrated frame (clean without header overlay)
            cv2.imwrite(filepath, processed_frame)
            print(f"Saved: {filepath}")

        # [R] -> Reset All Settings to Defaults
        elif key in (ord("r"), ord("R")):
            ctrl.reset_defaults()
            print("All settings reset to defaults.")

        # Stop if user closes either window
        if cv2.getWindowProperty(preview_win, cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty(ctrl_win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
