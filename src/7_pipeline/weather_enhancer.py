"""
Drishti Kavach: Harsh Weather, Fog, and Rain Optical Enhancement Pipeline

CLI Flags & Usage:
  Integrated into the pipeline. Provides real-time atmospheric defogging,
  CLAHE luminance contrast recovery, and rain-streak suppression.

Features:
  - Automatic Fog / Low-Visibility Detection (Airlight Haze Index)
  - Fast CLAHE Contrast Recovery (LAB Luminance Domain)
  - Atmospheric Scattering Dark Channel Prior (DCP) Dehazing
  - Temporal Rain-Streak Filtering for Live Video Streams
"""

from typing import Tuple, Optional
import cv2
import numpy as np


class WeatherEnhancer:
    """
    Real-time optical enhancement engine for dense fog, heavy rain, mist, and low contrast.
    """

    def __init__(self, clip_limit: float = 3.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.prev_frames = []
        self.max_temporal_frames = 3

    def detect_fog(self, frame_bgr: np.ndarray) -> Tuple[bool, float]:
        """
        Estimates the atmospheric fog / haze density in the frame.
        
        Returns:
            Tuple of (is_foggy: bool, fog_density_score: float [0.0 - 1.0])
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Standard Deviation of Gray (Low contrast = high fog)
        std_dev = float(np.std(gray))
        
        # 2. Dark Channel estimate (in hazy images, minimum channel values are high due to airlight)
        min_channel = np.min(frame_bgr, axis=2)
        mean_dark = float(np.mean(min_channel))
        
        # Normalize fog score: high mean dark channel + low standard deviation = dense fog
        contrast_score = np.clip((55.0 - std_dev) / 40.0, 0.0, 1.0)
        airlight_score = np.clip((mean_dark - 60.0) / 100.0, 0.0, 1.0)
        fog_score = float(0.6 * airlight_score + 0.4 * contrast_score)
        
        is_foggy = fog_score > 0.40
        return is_foggy, fog_score

    def enhance_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization on the Luminance channel.
        Restores crisp rail boundaries and obstacles without color distortion.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def defog_dcp(self, frame_bgr: np.ndarray, omega: float = 0.85, t0: float = 0.15) -> np.ndarray:
        """
        Atmospheric Scattering Model Inversion using Dark Channel Prior (DCP).
        Recovers true scene radiance J(x) = (I(x) - A) / max(t(x), t0) + A.
        """
        img_float = frame_bgr.astype(np.float32) / 255.0
        
        # 1. Calculate Dark Channel
        dark_channel = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(dark_channel, kernel)

        # 2. Estimate Atmospheric Light (A)
        num_pixels = dark_channel.size
        top_bright_idx = np.argsort(dark_channel.ravel())[-int(num_pixels * 0.001):]
        atm_light = np.median(img_float.reshape(-1, 3)[top_bright_idx], axis=0)

        # 3. Estimate Transmission Map t(x)
        transmission = 1.0 - omega * (dark_channel / np.max(atm_light))
        transmission = np.clip(transmission, t0, 1.0)
        transmission = cv2.GaussianBlur(transmission, (21, 21), 0)

        # 4. Recover Radiance
        recovered = np.empty_like(img_float)
        for i in range(3):
            recovered[:, :, i] = (img_float[:, :, i] - atm_light[i]) / transmission + atm_light[i]

        recovered = np.clip(recovered * 255.0, 0.0, 255.0).astype(np.uint8)
        
        # Subtle CLAHE on recovered image for edge sharpness
        return self.enhance_clahe(recovered)

    def filter_rain_temporal(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Temporal rolling filter across consecutive video frames to suppress transient falling rain streaks.
        """
        self.prev_frames.append(frame_bgr)
        if len(self.prev_frames) > self.max_temporal_frames:
            self.prev_frames.pop(0)

        if len(self.prev_frames) < 2:
            return frame_bgr

        # Rolling median across frames removes fast-moving vertical rain drops
        stacked = np.stack(self.prev_frames, axis=0)
        median_frame = np.median(stacked, axis=0).astype(np.uint8)
        return median_frame

    def process(
        self,
        frame_bgr: np.ndarray,
        mode: str = "auto",  # 'auto', 'clahe', 'dcp', 'rain', 'off'
        is_video_stream: bool = False
    ) -> Tuple[np.ndarray, str, float]:
        """
        Full weather enhancement processing entry point.

        Returns:
            Tuple of (enhanced_frame, active_mode_name, fog_score)
        """
        if mode == "off":
            return frame_bgr, "NORMAL", 0.0

        is_foggy, fog_score = self.detect_fog(frame_bgr)

        if mode == "auto":
            if is_foggy and fog_score > 0.60:
                out = self.defog_dcp(frame_bgr)
                return out, f"DCP DEFOG ({fog_score:.0%})", fog_score
            elif is_foggy:
                out = self.enhance_clahe(frame_bgr)
                return out, f"CLAHE ENHANCE ({fog_score:.0%})", fog_score
            else:
                return frame_bgr, "CLEAR ATMOSPHERE", fog_score

        elif mode == "clahe":
            return self.enhance_clahe(frame_bgr), "FORCED CLAHE", fog_score

        elif mode == "dcp":
            return self.defog_dcp(frame_bgr), "FORCED DCP DEFOG", fog_score

        elif mode == "rain":
            filtered = self.filter_rain_temporal(frame_bgr) if is_video_stream else frame_bgr
            enhanced = self.enhance_clahe(filtered)
            return enhanced, "RAIN FILTER + CLAHE", fog_score

        return frame_bgr, "NORMAL", 0.0
