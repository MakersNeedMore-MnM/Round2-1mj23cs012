"""
Drishti Kavach: Dynamic Responsive Railway HUD & High-Contrast Track Visualizer

Design Philosophy:
  - Dynamically scales and formats HUD telemetry to fit ANY image resolution (from 480p up to 4K).
  - High-Contrast Track Overlays:
      • Track Bed   : Vibrant Translucent Cyan-Blue (BGR: 255, 180, 0) [50% Alpha Blend]
      • Rail Lines  : Solid Maroon / Crimson (BGR: 35, 15, 140) with closed boundary
      • Obstacles   : Dynamic Three-Tier Badges (Red / Amber / Green) with ground anchor reticles
  - Responsive Layout Engine: Zero overlap, automatic text wrapping/compacting on narrow views.
"""

from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import importlib

_spatial = importlib.import_module("src.5_spatial_reasoning.hazard_analyzer")
HazardAssessment = _spatial.HazardAssessment


class DrishtiVisualizer:
    """
    Renders high-contrast railway track overlays and a responsive, dynamic top HUD header.
    """

    def __init__(self):
        # High-Contrast Track Colors
        self.COLOR_TRACK_BED = (255, 180, 0)       # Translucent Cyan-Blue (BGR)
        self.COLOR_RAIL_LINES = (35, 15, 140)     # Solid Maroon / Crimson (BGR)
        self.COLOR_RAIL_BORDER = (15, 5, 80)      # Dark Maroon Edge (BGR)

        # Three-Tier Threat Colors
        self.COLOR_CRITICAL = (0, 0, 230)         # Vivid Red (BGR)
        self.COLOR_WARNING = (0, 210, 255)        # Amber / Gold (BGR)
        self.COLOR_SAFE = (40, 210, 60)           # Green (BGR)

        # Tactical HUD Colors
        self.HUD_BG = (12, 14, 18)                # Deep Charcoal Black (BGR)
        self.TEXT_MAIN = (240, 245, 250)
        self.TEXT_MUTED = (160, 175, 190)

    def draw_hud_header(
        self,
        canvas: np.ndarray,
        overall_status: str,
        fps: float,
        sensor_mode: str,
        weather_status: str,
        hazards: List[HazardAssessment]
    ) -> np.ndarray:
        """
        Draws a dynamically scaled, responsive top status header with zero text overlap.
        """
        h, w = canvas.shape[:2]
        
        # Responsive scale factor based on width (normalized against 1280px standard)
        scale = float(np.clip(w / 1280.0, 0.45, 1.25))
        
        # Dynamic Header Height
        header_h = int(np.clip(54 * scale + 24, 42, 80))

        # 1. Header Background Strip with Subtle Gradient Border
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), self.HUD_BG, -1)
        cv2.line(overlay, (0, header_h - 1), (w, header_h - 1), (50, 65, 80), 1)
        cv2.line(overlay, (0, header_h), (w, header_h), (75, 95, 120), 2)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)

        # Font scales
        brand_font_scale = float(np.clip(0.70 * scale, 0.38, 0.85))
        badge_font_scale = float(np.clip(0.62 * scale, 0.34, 0.70))
        telemetry_font_scale = float(np.clip(0.48 * scale, 0.30, 0.55))

        mid_y = header_h // 2

        # 2. Left Zone: Status LED + "DRISHTI KAVACH"
        lamp_radius = max(3, int(6 * scale))
        lamp_x = max(10, int(20 * scale))
        lamp_col = self.COLOR_CRITICAL if overall_status == "CRITICAL" else (
            self.COLOR_WARNING if overall_status == "WARNING" else self.COLOR_SAFE
        )
        cv2.circle(canvas, (lamp_x, mid_y), lamp_radius, lamp_col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (lamp_x, mid_y), lamp_radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        brand_text = "DRISHTI KAVACH"
        brand_x = lamp_x + lamp_radius + max(6, int(10 * scale))
        (bw, bh), _ = cv2.getTextSize(brand_text, cv2.FONT_HERSHEY_DUPLEX, brand_font_scale, 1)
        brand_y = mid_y + bh // 2
        cv2.putText(canvas, brand_text, (brand_x, brand_y),
                    cv2.FONT_HERSHEY_DUPLEX, brand_font_scale, (0, 235, 255), 1, cv2.LINE_AA)
        left_boundary = brand_x + bw + max(12, int(16 * scale))

        # 3. Right Zone: Telemetry Metrics (FPS, Sensor, Optics)
        if w >= 950:
            telemetry_str = f"FPS: {fps:4.1f}  |  SENSOR: {sensor_mode}  |  OPTICS: {weather_status}"
        elif w >= 680:
            telemetry_str = f"FPS: {fps:4.1f} | {sensor_mode.split()[0]} | {weather_status}"
        else:
            telemetry_str = f"{fps:.1f} FPS"

        (rw, rh), _ = cv2.getTextSize(telemetry_str, cv2.FONT_HERSHEY_SIMPLEX, telemetry_font_scale, 1)
        right_margin = max(10, int(16 * scale))
        rx = w - rw - right_margin
        ry = mid_y + rh // 2
        cv2.putText(canvas, telemetry_str, (rx, ry),
                    cv2.FONT_HERSHEY_SIMPLEX, telemetry_font_scale, self.TEXT_MAIN, 1, cv2.LINE_AA)
        right_boundary = rx - max(12, int(16 * scale))

        # 4. Center Zone: Dynamic Tactical Safety Alert Box
        available_center_w = right_boundary - left_boundary

        if overall_status == "CRITICAL":
            box_bg = (0, 0, 185)
            box_border = (0, 0, 255)
            full_text = "[ ! EMERGENCY BRAKE : OBSTACLE IN TRACK ! ]"
            short_text = "[ ! EMERGENCY BRAKE ! ]"
            tiny_text = "[ BRAKE ]"
            text_color = (255, 255, 255)
        elif overall_status == "WARNING":
            box_bg = (0, 140, 205)
            box_border = (0, 220, 255)
            full_text = "[ CAUTION : CLEARANCE ENVELOPE BREACH ]"
            short_text = "[ CAUTION : NEAR TRACK ]"
            tiny_text = "[ CAUTION ]"
            text_color = (255, 255, 255)
        else:
            box_bg = (15, 70, 22)
            box_border = (40, 190, 55)
            full_text = "[ TRACK STATUS : ALL CLEAR / NOMINAL ]"
            short_text = "[ TRACK CLEAR ]"
            tiny_text = "[ CLEAR ]"
            text_color = (100, 255, 120)

        # Determine best text fitting available center space
        (tw_full, _), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_DUPLEX, badge_font_scale, 1)
        (tw_short, _), _ = cv2.getTextSize(short_text, cv2.FONT_HERSHEY_DUPLEX, badge_font_scale, 1)
        
        if available_center_w >= tw_full + 24:
            badge_text = full_text
            chosen_tw = tw_full
        elif available_center_w >= tw_short + 16:
            badge_text = short_text
            chosen_tw = tw_short
        else:
            badge_text = tiny_text
            (chosen_tw, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, badge_font_scale, 1)

        badge_w = min(int(chosen_tw + max(16, int(24 * scale))), available_center_w)
        badge_h = max(24, int(header_h * 0.62))
        
        center_x = (left_boundary + right_boundary) // 2
        badge_x1 = center_x - badge_w // 2
        badge_y1 = (header_h - badge_h) // 2
        badge_x2 = badge_x1 + badge_w
        badge_y2 = badge_y1 + badge_h

        # Only draw center box if there's reasonable space
        if badge_w > 30 and badge_x1 >= left_boundary and badge_x2 <= right_boundary:
            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), box_bg, -1)
            cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), box_border, 1)

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, badge_font_scale, 1)
            tx = badge_x1 + (badge_w - tw) // 2
            ty = badge_y1 + (badge_h + th) // 2 - 1
            cv2.putText(canvas, badge_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, badge_font_scale, text_color, 1, cv2.LINE_AA)

        return canvas

    def render(
        self,
        frame_bgr: np.ndarray,
        track_bed_polys: List[np.ndarray],
        rail_lines_polys: List[np.ndarray],
        hazards: List[HazardAssessment],
        overall_status: str,
        fps: float = 0.0,
        sensor_mode: str = "DAYLIGHT RGB",
        weather_status: str = "CLEAR",
        show_hud: bool = True
    ) -> np.ndarray:
        """
        Renders full visualizer output:
        1. High-contrast track bed & rail lines masks
        2. Clean obstacle bounding boxes with subtle badge
        3. Old-school sleek top HUD header
        """
        h, w = frame_bgr.shape[:2]
        scale = float(np.clip(w / 1280.0, 0.45, 1.25))
        
        canvas = frame_bgr.copy()
        track_overlay = frame_bgr.copy()

        # 1. Render Track Bed (Cyan-Blue 50% blend)
        for poly in track_bed_polys:
            if len(poly) >= 3:
                pts_np = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(track_overlay, [pts_np], self.COLOR_TRACK_BED)

        # 2. Render Rail Lines (Glowing Crimson / Maroon)
        for poly in rail_lines_polys:
            if len(poly) >= 3:
                pts_np = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(track_overlay, [pts_np], self.COLOR_RAIL_LINES)
                line_thick = max(1, int(round(2 * scale)))
                cv2.polylines(track_overlay, [pts_np], True, self.COLOR_RAIL_BORDER, line_thick, cv2.LINE_AA)

        # 3. Alpha Blend Track Overlays
        cv2.addWeighted(track_overlay, 0.50, canvas, 0.50, 0, canvas)

        # 4. Render Detected Obstacles
        box_thick = max(1, int(round(2 * scale)))
        label_font_scale = float(np.clip(0.42 * scale, 0.30, 0.52))

        for hzd in hazards:
            x1, y1, x2, y2 = hzd.box_coords
            color = hzd.color_bgr

            # Solid Bounding Box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thick)

            # Ground Contact Anchor Reticle
            ax = int((x1 + x2) / 2)
            ay = int(y2)
            reticle_r = max(2, int(4 * scale))
            cv2.circle(canvas, (ax, ay), reticle_r, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (ax, ay), reticle_r + 3, (255, 255, 255), 1, cv2.LINE_AA)

            # Clean Label: Just Class Name and Confidence Percentage
            label = f"{hzd.class_name} {hzd.confidence:.0%}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, label_font_scale, 1)
            pad = max(2, int(4 * scale))
            by1 = max(0, y1 - lh - pad * 2)
            by2 = y1
            bx1 = x1
            bx2 = x1 + lw + pad * 2

            # Clean solid badge background in threat color
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
            cv2.putText(canvas, label, (bx1 + pad, by2 - pad),
                        cv2.FONT_HERSHEY_DUPLEX, label_font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Render Top HUD Header
        if show_hud:
            canvas = self.draw_hud_header(
                canvas,
                overall_status=overall_status,
                fps=fps,
                sensor_mode=sensor_mode,
                weather_status=weather_status,
                hazards=hazards
            )

        return canvas
