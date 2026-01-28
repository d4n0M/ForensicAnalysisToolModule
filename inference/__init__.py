"""
Module - Forensic Weapon Detection Module
Zero-shot object detection for forensic analysis using OWLv2

Author: Daniel Ceresna
Project: Development of a Forensic Analytical Tool
Institution: Brno University of Technology, Faculty of Information Technology
"""

__version__ = "1.0.0"
__author__ = "Daniel Ceresna"
__license__ = "MIT"

# Import main classes and functions from the module
from .inference import (
    # Main class
    Inference,
    
    # Data structures
    Detection,
    FrameMetadata,
    DetectionType,
    
    # Convenience functions
    create_detector,
    detect_from_buffer,
)

# Define what gets exported when someone does "from inference import *"
__all__ = [
    # Main class
    'Inference',
    
    # Data structures
    'Detection',
    'FrameMetadata',
    'DetectionType',
    
    # Factory and utility functions
    'create_detector',
    'detect_from_buffer',
    
    # Metadata
    '__version__',
    '__author__',
]


# Optional: Print initialization message (can be removed in production)
def _init_check():
    """Check if required dependencies are available"""
    try:
        import torch
        import transformers
        import cv2
        return True
    except ImportError as e:
        print(f"Warning: Missing dependency - {e}")
        return False


# You can optionally run initialization checks
# _init_check()
