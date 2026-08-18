"""
Drishti Kavach: Real-Time Visualization and Telemetry Head-Up Display (HUD)
"""

import importlib

_vis = importlib.import_module("src.6_visualization.visualizer")
DrishtiVisualizer = _vis.DrishtiVisualizer

__all__ = ["DrishtiVisualizer"]
