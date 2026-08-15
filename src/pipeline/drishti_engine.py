"""
Drishti Kavach: End-to-End Real-Time Perception & Decision Pipeline

Integrates:
  1. Weather & Atmospheric Optical Enhancer (Adaptive Defogger / CLAHE)
  2. Unified 'RailDrishti' Multi-Task Segmentation & Detection Model
  3. Spatial Hazard & Clearance Envelope Reasoning Engine
  4. BiSeNetV2-Style High-Contrast HUD Dashboard Renderer
"""

import os
import time
from typing import Tuple, List, Dict, Optional, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.pipeline.weather_enhancer import WeatherEnhancer
from src.spatial_reasoning.hazard_analyzer import SpatialHazardAnalyzer, HazardAssessment
from src.visualization.visualizer import DrishtiVisualizer


class DrishtiEngine:
    """
    Complete real-time perception pipeline for Indian Railways ATP Kavach.
    """

    def __init__(
        self,
        model_path: str = "models/RailDrishti.pt",
        conf_thresh: float = 0.35,
        imgsz: int = 1024,
        device: Optional[str] = None,
        weather_mode: str = "auto",
        warning_buffer_px: float = 65.0
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.weather_mode = weather_mode

        # Hardware selection
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("=" * 75)
        print(" 🛡️  INITIALIZING DRISHTI KAVACH REAL-TIME ENGINE")
        print("=" * 75)
        print(f" • Model Weights:      {model_path}")
        print(f" • Hardware Device:    {self.device}")
        print(f" • Target Resolution:  {imgsz}x{imgsz}")
        print(f" • Confidence Cutoff:  {conf_thresh:.2f}")
        print(f" • Weather Optimizer:  {weather_mode.upper()}")
        print("=" * 75)

        # Load Unified Model
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"[+] Loaded RailDrishti model successfully from: {model_path}")
        else:
            print(f"[!] Warning: {model_path} not found. Fallback to yolo11s-seg.pt base.")
            self.model = YOLO("yolo11s-seg.pt")

        # Class Names
        self.names = self.model.names if hasattr(self.model, "names") else {}

        # Subsystems
        self.weather_enhancer = WeatherEnhancer()
        self.hazard_analyzer = SpatialHazardAnalyzer(warning_buffer_pixels=warning_buffer_px)
        self.visualizer = DrishtiVisualizer()

        # FPS metrics
        self.prev_time = time.time()
        self.fps = 0.0

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        sensor_type: str = "DAYLIGHT RGB",
        show_hud: bool = True,
        is_video_stream: bool = False
    ) -> Tuple[np.ndarray, str, List[HazardAssessment], Dict]:
        """
        Executes full perception and reasoning cycle on a single video frame.

        Returns:
            Tuple of (rendered_frame, overall_status, hazards_list, telemetry_dict)
        """
        t_start = time.time()
        h, w = frame_bgr.shape[:2]

        # 1. Optical Enhancement & Defogging
        enhanced_frame, weather_status, fog_score = self.weather_enhancer.process(
            frame_bgr, mode=self.weather_mode, is_video_stream=is_video_stream
        )

        # 2. Neural Network Forward Pass (YOLO11-seg)
        results = self.model.predict(
            source=enhanced_frame,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False
        )[0]

        # 3. Parse Multi-Task Segmentation Masks and Bounding Boxes
        track_bed_polys = []
        rail_lines_polys = []
        obstacles = []

        if results.masks is not None and results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            masks = results.masks.xy  # List of polygon coordinate arrays (N, 2)

            for idx in range(len(boxes)):
                cls_id = int(boxes.cls[idx])
                conf = float(boxes.conf[idx])
                x1, y1, x2, y2 = boxes.xyxy[idx]
                cname = self.names.get(cls_id, f"Class_{cls_id}")

                if cls_id == 0:  # Rail_Track_Bed Polygon
                    if idx < len(masks) and len(masks[idx]) >= 3:
                        track_bed_polys.append(masks[idx])
                elif cls_id == 1:  # Rail_Lines Polygon
                    if idx < len(masks) and len(masks[idx]) >= 3:
                        rail_lines_polys.append(masks[idx])
                else:  # Obstacle Class (2..10)
                    obstacles.append({
                        "box": (x1, y1, x2, y2),
                        "class_id": cls_id,
                        "class_name": cname,
                        "confidence": conf
                    })

        # 4. Geometric Spatial Clearance Reasoning
        hazards, overall_status = self.hazard_analyzer.analyze(
            track_bed_polys=track_bed_polys,
            rail_lines_polys=rail_lines_polys,
            obstacles=obstacles,
            image_shape=(h, w)
        )

        # 5. Compute FPS
        t_end = time.time()
        frame_time = t_end - t_start
        self.fps = 1.0 / max(frame_time, 0.001)

        # 6. Render BiSeNetV2 HUD
        rendered = self.visualizer.render(
            frame_bgr=enhanced_frame,
            track_bed_polys=track_bed_polys,
            rail_lines_polys=rail_lines_polys,
            hazards=hazards,
            overall_status=overall_status,
            fps=self.fps,
            sensor_mode=sensor_type,
            weather_status=weather_status,
            show_hud=show_hud
        )

        telemetry = {
            "fps": self.fps,
            "overall_status": overall_status,
            "weather_status": weather_status,
            "fog_score": fog_score,
            "num_hazards": len(hazards),
            "track_bed_found": len(track_bed_polys) > 0,
            "rail_lines_found": len(rail_lines_polys) > 0
        }

        return rendered, overall_status, hazards, telemetry
