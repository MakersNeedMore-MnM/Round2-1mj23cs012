import os
from contextlib import contextmanager
from datetime import datetime
import cv2

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


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

    save_dir = os.path.join("camera_captures", "camera_tester")
    os.makedirs(save_dir, exist_ok=True)

    win_name = f"Camera {cam_idx}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("s"), ord("S"), 32):  # 's' or SPACE
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            time_str = now.strftime("%H-%M-%S")
            sub_sec = f"{now.microsecond // 10000:02d}"
            filename = f"camera_tester_{date_str}_{time_str}_{sub_sec}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filepath}")

        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
