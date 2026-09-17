"""
Drishti Kavach: Real-Time Railway Physical Obstacle & Track Clearance Inference Engine

CLI Flags & Usage:
  --source STR/INT : Input source: '0' (Webcam/USB Camera), 'video.mp4', 'image.jpg', 'rtsp://...' (default: '0')
  --conf FLOAT     : Confidence threshold for obstacle detection (default: 0.35)
  --imgsz INT      : Inference resolution for obstacle detection (default: 1024)
  --weather STR    : Weather defogging mode: 'off', 'auto', 'clahe', 'dcp' (default: 'off')
  --save           : Save annotated output stream to camera_captures/main-videos/
  --no-view        : Run in headless mode without opening GUI window

Interactive Keyboard Controls (in GUI Window):
  [S] / [SPACE]    : Capture snapshot to camera_captures/main/
  [Q] / [ESC]      : Quit stream

Examples:
  # 1. Live USB / Arducam Camera @ 1024p:
  python main.py

  # 2. Specific camera index with higher confidence threshold:
  python main.py --source 0 --conf 0.40

  # 3. Process video file and save output:
  python main.py --source test_samples/sample_videos/test.mp4 --save

  # 4. Enable optional weather defogging optimizer:
  python main.py --weather auto
"""

import os
import sys
import time
import threading
import argparse
from datetime import datetime
from pathlib import Path
import cv2

from src import DrishtiEngine


class ThreadedCamera:
    """Non-blocking asynchronous threaded video stream reader for zero I/O wait latency."""

    def __init__(self, source):
        self.is_cam = str(source).isdigit()
        if self.is_cam:
            self.cap = cv2.VideoCapture(int(source))
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        else:
            self.cap = cv2.VideoCapture(source)

        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

        if self.is_cam and self.cap.isOpened():
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.002)
                continue
            with self.lock:
                self.grabbed = ret
                self.frame = frame

    def read(self):
        if self.is_cam:
            with self.lock:
                return self.grabbed, (self.frame.copy() if self.frame is not None else None)
        else:
            return self.cap.read()

    def release(self):
        self.stopped = True
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.4)
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def set(self, prop_id, value):
        return self.cap.set(prop_id, value)


def normalize_status(st: str) -> str:
    """Normalizes status strings so CLEAR and ALL CLEAR map to ALL CLEAR."""
    s = str(st).upper().strip()
    if s in ["CLEAR", "ALL CLEAR", "SAFE", "NOMINAL"]:
        return "ALL CLEAR"
    elif "CRITICAL" in s:
        return "CRITICAL"
    elif "WARN" in s:
        return "WARNING"
    return s


def save_session_report(
    reports_dir: str,
    source: str,
    start_dt: datetime,
    end_dt: datetime,
    elapsed_sec: float,
    total_frames: int,
    conf_thresh: float,
    imgsz: int,
    log_entries: list,
    detection_stats: dict,
    status_counts: dict
):
    """Generates and writes a comprehensive formatted session text report."""
    os.makedirs(reports_dir, exist_ok=True)
    end_str = end_dt.strftime("%d-%m-%Y_%H-%M-%S")
    report_filename = f"session_report_{end_str}.txt"
    report_path = os.path.join(reports_dir, report_filename)

    avg_fps = (total_frames / elapsed_sec) if elapsed_sec > 0 else 0.0

    # Format elapsed duration
    mins, secs = divmod(int(elapsed_sec), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        duration_str = f"{hrs}h {mins}m {secs}s ({elapsed_sec:.1f} seconds)"
    elif mins > 0:
        duration_str = f"{mins}m {secs}s ({elapsed_sec:.1f} seconds)"
    else:
        duration_str = f"{elapsed_sec:.1f} seconds"

    lines = [
        "=" * 80,
        "                 DRISHTI KAVACH — SESSION INFERENCE REPORT",
        "=" * 80,
        f"Session Start   : {start_dt.strftime('%d-%m-%Y %H:%M:%S')}",
        f"Session End     : {end_dt.strftime('%d-%m-%Y %H:%M:%S')}",
        f"Total Duration  : {duration_str}",
        f"Input Source    : {source}",
        f"Total Frames    : {total_frames:,} frames",
        f"Average FPS     : {avg_fps:.1f} FPS",
        f"Resolution      : Inference @ {imgsz}x{imgsz}",
        f"Confidence Cut  : {conf_thresh:.2f}",
        "=" * 80,
        "                            CHRONOLOGICAL TRACK LOGS",
        "=" * 80,
    ]

    if log_entries:
        lines.extend(log_entries)
    else:
        lines.append("No periodic logs recorded.")

    lines.extend([
        "=" * 80,
        "                       CONSOLIDATED DETECTIONS SUMMARY",
        "=" * 80,
    ])

    total_instances = sum(len(confs) for confs in detection_stats.values())
    lines.append(f"Total Obstacle Detections: {total_instances} instances across {len(detection_stats)} unique classes\n")

    if detection_stats:
        lines.append(f"{'Class Name':<20} {'Count':<10} {'Avg Conf':<12} {'Min Conf':<12} {'Max Conf':<12}")
        lines.append("-" * 80)
        for cname, confs in sorted(detection_stats.items(), key=lambda item: len(item[1]), reverse=True):
            cnt = len(confs)
            avg_c = (sum(confs) / cnt) * 100
            min_c = min(confs) * 100
            max_c = max(confs) * 100
            lines.append(f"{cname:<20} {cnt:<10} {avg_c:>6.1f}%      {min_c:>6.1f}%      {max_c:>6.1f}%")
    else:
        lines.append("No obstacle detections recorded during this session.")

    lines.extend([
        "=" * 80,
        "                       TRACK SAFETY STATUS BREAKDOWN",
        "=" * 80,
    ])

    if total_frames > 0:
        clean_counts = {"ALL CLEAR": 0, "WARNING": 0, "CRITICAL": 0}
        for st, count in status_counts.items():
            norm_k = normalize_status(st)
            clean_counts[norm_k] = clean_counts.get(norm_k, 0) + count

        for st_name in ["ALL CLEAR", "WARNING", "CRITICAL"]:
            st_count = clean_counts.get(st_name, 0)
            pct = (st_count / total_frames) * 100
            lines.append(f"• {st_name:<16}: {st_count:>6,} frames ({pct:>5.1f}%)")
    else:
        lines.append("No frame statistics available.")

    lines.extend([
        "=" * 80,
        "Report generated automatically by Drishti Kavach ATP Engine.",
        "=" * 80,
        ""
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[+] Full session report saved to: {report_path}")
    return report_path


def run_inference(
    source: str = "0",
    conf_thresh: float = 0.35,
    imgsz: int = 640,
    weather_mode: str = "off",
    save_output: bool = False,
    show_view: bool = True
):
    video_dir = os.path.join("camera_captures", "main-videos")
    snapshot_dir = os.path.join("camera_captures", "main")
    reports_dir = os.path.join("outputs", "reports")
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    if save_output:
        os.makedirs(video_dir, exist_ok=True)

    # Initialize Engine (Default 1024 full resolution inference)
    engine = DrishtiEngine(
        conf_thresh=conf_thresh,
        imgsz=imgsz,
        weather_mode=weather_mode
    )

    # 1. Single Image Source
    valid_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    if os.path.isfile(source) and any(source.lower().endswith(ext) for ext in valid_exts):
        start_dt = datetime.now()
        start_t = time.time()

        frame = cv2.imread(source)
        if frame is None:
            print(f"[!] Error: Could not load image: {source}")
            return

        rendered, status, hazards, telemetry = engine.process_frame(
            frame, show_hud=True, is_video_stream=False
        )
        elapsed_sec = time.time() - start_t
        end_dt = datetime.now()

        print(f"\n[*] Processing Complete for: {source}")
        print(f"  • Clearance Status: {status}")
        print(f"  • Hazards Detected: {len(hazards)}")
        print(f"  • Frame Rate:       {telemetry['fps']:.1f} FPS")

        if save_output:
            date_str = start_dt.strftime("%d-%m-%Y")
            time_str = start_dt.strftime("%H-%M-%S")
            sub_sec = f"{start_dt.microsecond // 10000:02d}"
            out_name = f"main_{date_str}_{time_str}_{sub_sec}.jpg"
            out_path = os.path.join(snapshot_dir, out_name)
            cv2.imwrite(out_path, rendered)
            print(f"[+] Saved result to: {out_path}")

        # Record statistics
        detection_stats = {}
        for h in hazards:
            detection_stats.setdefault(h.class_name, []).append(h.confidence)
        status_counts = {status: 1}
        log_entries = [
            f"• [IMAGE RESULT] {start_dt.strftime('%d.%m.%Y | %H:%M:%S')} -> {status.upper()} | Hazards ({len(hazards)}) | {telemetry['fps']:.1f} FPS"
        ]

        save_session_report(
            reports_dir=reports_dir,
            source=source,
            start_dt=start_dt,
            end_dt=end_dt,
            elapsed_sec=elapsed_sec,
            total_frames=1,
            conf_thresh=conf_thresh,
            imgsz=imgsz,
            log_entries=log_entries,
            detection_stats=detection_stats,
            status_counts=status_counts
        )

        if show_view:
            cv2.imshow("Drishti Kavach - Railway Clearance ATP", rendered)
            print("\n[Controls] Press any key in the window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # 2. Asynchronous Threaded Camera / Video Stream
    is_live_cam = str(source).isdigit() or str(source).lower().startswith("rtsp://") or str(source).lower().startswith("http://")
    cap = ThreadedCamera(source)

    if not cap.isOpened():
        print(f"[!] Error: Could not open video source: {source}")
        return

    writer = None
    out_video_path = None
    target_fps = 30.0
    total_written_frames = 0
    video_start_time = None

    if save_output:
        session_now = datetime.now()
        date_str = session_now.strftime("%d-%m-%Y")
        time_str = session_now.strftime("%H-%M-%S")
        sub_sec = f"{session_now.microsecond // 10000:02d}"
        video_filename = f"main_{date_str}_{time_str}_{sub_sec}.mp4"
        out_video_path = os.path.join(video_dir, video_filename)

        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if not is_live_cam and fps_in > 0:
            target_fps = fps_in
        else:
            target_fps = 30.0  # Standard smooth playback rate for live streams

    win_name = "Drishti Kavach - Railway Clearance ATP"
    if show_view:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 65)
    print(" 🛡️  DRISHTI KAVACH REAL-TIME STREAM ACTIVE")
    print(" • [S] / [SPACE] : Save Snapshot")
    print(" • [Q] / [ESC]   : Quit Stream")
    print("=" * 65 + "\n")

    last_report_time = 0.0
    prev_status = None

    # Session Metrics Tracking
    session_start_dt = datetime.now()
    session_start_t = time.time()
    total_frames_processed = 0
    log_entries = []
    detection_stats = {}
    status_counts = {"ALL CLEAR": 0, "WARNING": 0, "CRITICAL": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_live_cam:
                    time.sleep(0.002)
                    continue
                else:
                    if not save_output:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("\n[*] End of video stream reached.")
                    break

            rendered, status, hazards, telemetry = engine.process_frame(
                frame, show_hud=True, is_video_stream=True
            )

            total_frames_processed += 1
            status_counts[status] = status_counts.get(status, 0) + 1

            # Track detection confidence stats per class
            for h in hazards:
                detection_stats.setdefault(h.class_name, []).append(h.confidence)

            # Periodic (2s) and Event-Driven Immediate Terminal Reporting
            curr_time = time.time()
            status_changed = (prev_status is not None and status != prev_status)
            if prev_status is None or status_changed or (curr_time - last_report_time >= 2.0):
                last_report_time = curr_time
                prev_status = status

                if hazards:
                    hazard_list = [f"{h.class_name} ({h.confidence:.0%})" for h in hazards]
                    hazard_str = f"Obstacles ({len(hazards)}): " + ", ".join(hazard_list)
                else:
                    hazard_str = "Obstacles: None"

                ts_now = datetime.now().strftime("%d.%m.%Y | %H:%M:%S")
                prefix = " ⚡ [STATUS CHANGE]" if status_changed else " • [TRACK REPORT]"
                log_line = f"{prefix} {ts_now} -> {status.upper()} | {hazard_str} | {telemetry['fps']:.1f} FPS"
                print(log_line)
                log_entries.append(log_line)

            if save_output:
                if writer is None:
                    h_out, w_out = rendered.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_video_path, fourcc, target_fps, (w_out, h_out))
                    video_start_time = time.time()
                    print(f"[+] Output video recording started: {out_video_path} ({target_fps:.1f} FPS)")

                if is_live_cam:
                    # Synchronize video frames to wall-clock time for 1:1 real-time playback speed
                    elapsed_time = time.time() - video_start_time
                    expected_frames = max(1, int(elapsed_time * target_fps))
                    repeats = max(1, expected_frames - total_written_frames)
                    for _ in range(repeats):
                        writer.write(rendered)
                    total_written_frames += repeats
                else:
                    writer.write(rendered)
                    total_written_frames += 1

            if show_view:
                cv2.imshow(win_name, rendered)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break
                elif key in (ord("s"), ord("S"), 32):  # 's' or SPACE
                    now = datetime.now()
                    date_str = now.strftime("%d-%m-%Y")
                    time_str = now.strftime("%H-%M-%S")
                    sub_sec = f"{now.microsecond // 10000:02d}"
                    filename = f"main_{date_str}_{time_str}_{sub_sec}.jpg"
                    snap_path = os.path.join(snapshot_dir, filename)
                    cv2.imwrite(snap_path, rendered)
                    print(f"[+] Snapshot saved: {snap_path}")

                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        session_end_dt = datetime.now()
        session_elapsed_sec = time.time() - session_start_t

        cap.release()
        if writer is not None:
            writer.release()
            print(f"\n[+] Successfully saved output video to: {out_video_path}")
        if show_view:
            cv2.destroyAllWindows()

        save_session_report(
            reports_dir=reports_dir,
            source=str(source),
            start_dt=session_start_dt,
            end_dt=session_end_dt,
            elapsed_sec=session_elapsed_sec,
            total_frames=total_frames_processed,
            conf_thresh=conf_thresh,
            imgsz=imgsz,
            log_entries=log_entries,
            detection_stats=detection_stats,
            status_counts=status_counts
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drishti Kavach Real-Time Railway Clearance Inference")
    parser.add_argument("--source", type=str, default="0", help="Camera index (default: '0') or video/image path")
    parser.add_argument("--conf", type=float, default=0.35, help="Obstacle detection confidence threshold (default: 0.35)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (default: 640)")
    parser.add_argument("--weather", type=str, default="off", choices=["off", "auto", "clahe", "dcp"], help="Weather defogging (default: off)")
    parser.add_argument("--save", action="store_true", help="Save annotated output video to camera_captures/main-videos/")
    parser.add_argument("--no-view", action="store_true", help="Headless mode without GUI window")

    args = parser.parse_args()

    run_inference(
        source=args.source,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        weather_mode=args.weather,
        save_output=args.save,
        show_view=not args.no_view
    )

