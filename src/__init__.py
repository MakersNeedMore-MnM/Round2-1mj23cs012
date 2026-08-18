"""
Drishti Kavach: AI-Powered Railway Physical Obstacle & Track Clearance System
Model: RailDrishti (Unified Multi-Task Vision Architecture)
"""

import importlib

# Dynamic imports from ordered numbered subpackages
_spatial = importlib.import_module("src.5_spatial_reasoning.hazard_analyzer")
SpatialHazardAnalyzer = _spatial.SpatialHazardAnalyzer
HazardAssessment = _spatial.HazardAssessment

_vis = importlib.import_module("src.6_visualization.visualizer")
DrishtiVisualizer = _vis.DrishtiVisualizer

_weather = importlib.import_module("src.7_pipeline.weather_enhancer")
WeatherEnhancer = _weather.WeatherEnhancer

_engine = importlib.import_module("src.7_pipeline.drishti_engine")
DrishtiEngine = _engine.DrishtiEngine

__version__ = "1.0.0"

__all__ = [
    "SpatialHazardAnalyzer",
    "HazardAssessment",
    "DrishtiVisualizer",
    "WeatherEnhancer",
    "DrishtiEngine",
]
