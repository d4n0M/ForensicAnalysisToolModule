"""
Inference Python Module - Forensic Weapon Detection
Based on OWLv2 zero-shot object detection
Designed for C++ integration via pybind11
"""

import os
import cv2
import time
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import nms
import gc


class DetectionType(Enum):
    """Types of objects that can be detected"""
    WEAPON_GUN = "gun"
    WEAPON_KNIFE = "knife"
    WEAPON_RIFLE = "rifle"
    WEAPON_HANDGUN = "handgun"
    WEAPON_GENERIC = "weapon"
    VIOLENCE_FIGHT = "fight"
    VIOLENCE_ALTERCATION = "altercation"


@dataclass
class Detection:
    """
    Single detection result
    
    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates
        confidence: Detection confidence score [0.0, 1.0]
        objectness: Objectness score (same as confidence for OWLv2)
        label: Text label of detected object
        detection_type: Categorized detection type
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    objectness: float
    label: str
    detection_type: str  # String value of DetectionType enum
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class FrameMetadata:
    """
    Metadata for a single frame/image analysis
    
    Attributes:
        frame_id: Frame identifier (frame number for video, 0 for image)
        detections: List of all detections in this frame
        processing_time_ms: Time taken to process this frame in milliseconds
        resolution: Original image resolution as (width, height)
        has_threat: Quick flag indicating if any weapon was detected
    """
    frame_id: int
    detections: List[Detection]
    processing_time_ms: float
    resolution: Tuple[int, int]
    has_threat: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'frame_id': self.frame_id,
            'detections': [d.to_dict() for d in self.detections],
            'processing_time_ms': self.processing_time_ms,
            'resolution': self.resolution,
            'has_threat': self.has_threat
        }


class Inference:
    """
    Main interface for zero-shot weapon and violence detection
    Uses OWLv2 model for open-vocabulary object detection
    Designed to be called from C++ via pybind11
    """
    
    def __init__(self, 
                 model_name: str = "owlv2",
                 confidence_threshold: float = 0.35,
                 iou_threshold: float = 0.3,
                 device: str = "cuda",
                 nms_threshold: float = 0.3):
        """
        Initialize the detection model
        
        Args:
            model_name: Model to use ("owlv2" or "owlv2-large")
            confidence_threshold: Minimum confidence for detections [0.0, 1.0]
            iou_threshold: IoU threshold for post-processing (currently for evaluation)
            device: Device to run inference on ("cuda", "cpu", "cuda:0", etc.)
            nms_threshold: NMS IoU threshold for duplicate removal
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        
        # Model initialization
        self._initialize_model()
        
        # Default text prompts
        self.default_prompts = ["a photo of a gun", "a photo of a weapon"]
        
        print(f"Inference initialized: {self.model_name} on {self.device}")
    
    def _initialize_model(self):
        """Load the OWLv2 model and processor"""
        # Select model variant
        if self.model_name == "owlv2-large":
            model_id = "google/owlv2-large-patch14-ensemble"
        else:
            model_id = "google/owlv2-base-patch16-ensemble"
        
        print(f"Loading model: {model_id}")
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id)
        self.model = self.model.to(self.device).eval()
        
        # Count parameters
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {self.num_parameters / 1e6:.1f}M parameters")
    
    def _categorize_detection(self, label: str) -> DetectionType:
        """Categorize detection based on label text"""
        label_lower = label.lower()
        
        if "gun" in label_lower or "pistol" in label_lower or "handgun" in label_lower:
            return DetectionType.WEAPON_GUN
        elif "rifle" in label_lower or "ak" in label_lower or "ar-" in label_lower:
            return DetectionType.WEAPON_RIFLE
        elif "knife" in label_lower or "blade" in label_lower:
            return DetectionType.WEAPON_KNIFE
        elif "weapon" in label_lower:
            return DetectionType.WEAPON_GENERIC
        elif "fight" in label_lower or "fighting" in label_lower:
            return DetectionType.VIOLENCE_FIGHT
        elif "altercation" in label_lower or "violence" in label_lower:
            return DetectionType.VIOLENCE_ALTERCATION
        else:
            return DetectionType.UNKNOWN
    
    def _process_frame(self, 
                      image: Image.Image, 
                      text_prompts: List[str],
                      frame_id: int = 0) -> FrameMetadata:
        """
        Process a single frame/image
        
        Args:
            image: PIL Image
            text_prompts: List of text prompts for detection
            frame_id: Frame identifier
            
        Returns:
            FrameMetadata with detections
        """
        start_time = time.time()
        
        # Prepare inputs
        inputs = self.processor(text=text_prompts, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=0.3  # Low threshold for initial filtering
        )[0]
        
        # Filter by confidence threshold
        mask = results["scores"] > self.confidence_threshold
        boxes = results["boxes"][mask]
        scores = results["scores"][mask]
        labels = results["labels"][mask]
        
        # Apply Non-Maximum Suppression to remove duplicates
        if len(boxes) > 0:
            keep_indices = nms(boxes, scores, iou_threshold=self.nms_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
        
        # Convert to Detection objects
        detections = []
        for box, score, label_idx in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.cpu().tolist()
            confidence = float(score.cpu())
            label_text = text_prompts[int(label_idx)]
            
            detection_type = self._categorize_detection(label_text)
            
            detection = Detection(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                objectness=confidence,  # OWLv2 doesn't separate objectness
                label=label_text,
                detection_type=detection_type.value
            )
            detections.append(detection)
        
        # Clean up
        del outputs, inputs, results
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create metadata
        has_threat = len(detections) > 0
        metadata = FrameMetadata(
            frame_id=frame_id,
            detections=detections,
            processing_time_ms=processing_time_ms,
            resolution=(image.width, image.height),
            has_threat=has_threat
        )
        
        return metadata
    
    def detect_image(self, 
                     image_data: np.ndarray,
                     text_prompts: Optional[List[str]] = None) -> FrameMetadata:
        """
        Detect weapons/violence in a single image
        
        Args:
            image_data: Image as numpy array (H, W, C) in BGR or RGB format
            text_prompts: Optional custom text prompts. If None, uses default
        
        Returns:
            FrameMetadata containing all detections and metadata
        """
        if text_prompts is None:
            text_prompts = self.default_prompts
        
        # Convert numpy array to PIL Image
        if image_data.shape[2] == 3:
            # Assume BGR from OpenCV, convert to RGB
            image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image_data
        
        image = Image.fromarray(image_rgb)
        
        # Process the image
        return self._process_frame(image, text_prompts, frame_id=0)
    
    def detect_video(self,
                     video_path: str,
                     text_prompts: Optional[List[str]] = None,
                     frame_skip: int = 0) -> List[FrameMetadata]:
        """
        Detect weapons/violence in a video file
        
        Args:
            video_path: Path to video file
            text_prompts: Optional custom text prompts
            frame_skip: Process every N-th frame (0 = process all frames)
        
        Returns:
            List of FrameMetadata for each processed frame
        """
        if text_prompts is None:
            text_prompts = self.default_prompts
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video: {video_path} ({frame_count} frames)")
        
        results = []
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            # Skip frames if requested
            if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0:
                frame_idx += 1
                continue
            
            try:
                # Convert to PIL Image
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image_rgb)
                
                # Process frame
                metadata = self._process_frame(image, text_prompts, frame_id=frame_idx)
                results.append(metadata)
                processed_count += 1
                
                # Periodic cleanup
                if processed_count % 50 == 0:
                    gc.collect()
                    if self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Warning: Frame {frame_idx} failed: {e}")
            
            frame_idx += 1
        
        cap.release()
        print(f"Processed {processed_count} frames from {frame_count} total")
        
        return results
    
    def detect_image_batch(self,
                          image_data_list: List[np.ndarray],
                          text_prompts: Optional[List[str]] = None) -> List[FrameMetadata]:
        """
        Detect weapons/violence in multiple images (batch processing)
        
        Args:
            image_data_list: List of images as numpy arrays
            text_prompts: Optional custom text prompts
        
        Returns:
            List of FrameMetadata for each image
        """
        if text_prompts is None:
            text_prompts = self.default_prompts
        
        results = []
        for idx, image_data in enumerate(image_data_list):
            metadata = self.detect_image(image_data, text_prompts)
            metadata.frame_id = idx  # Update frame_id for batch
            results.append(metadata)
            
            # Periodic cleanup
            if (idx + 1) % 20 == 0:
                gc.collect()
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        
        return results
    
    def set_threshold(self, confidence: float, iou: Optional[float] = None, nms: Optional[float] = None):
        """
        Update detection thresholds dynamically
        
        Args:
            confidence: New confidence threshold [0.0, 1.0]
            iou: Optional new IoU threshold [0.0, 1.0]
            nms: Optional new NMS threshold [0.0, 1.0]
        """
        self.confidence_threshold = confidence
        if iou is not None:
            self.iou_threshold = iou
        if nms is not None:
            self.nms_threshold = nms
        
        print(f"Thresholds updated - Confidence: {self.confidence_threshold}, "
              f"IoU: {self.iou_threshold}, NMS: {self.nms_threshold}")
    
    def set_prompts(self, prompts: List[str]):
        """
        Update default text prompts
        
        Args:
            prompts: List of text prompts for detection
        """
        self.default_prompts = prompts
        print(f"Default prompts updated: {prompts}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_name': self.model_name,
            'parameters': self.num_parameters,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'nms_threshold': self.nms_threshold,
            'default_prompts': self.default_prompts
        }
    
    def warmup(self, image_size: Tuple[int, int] = (1920, 1080)):
        """
        Warm up the model with a dummy inference
        Eliminates first-run overhead
        
        Args:
            image_size: Size of dummy image for warmup (width, height)
        """
        print(f"Warming up model with {image_size[0]}x{image_size[1]} image...")
        
        # Create dummy image
        dummy_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Run dummy inference
        _ = self.detect_image(dummy_image)
        
        print("Warmup complete")


# ============================================================================
# Convenience functions for C++ binding
# ============================================================================

def create_detector(model_name: str = "owlv2",
                   confidence_threshold: float = 0.35,
                   iou_threshold: float = 0.3,
                   device: str = "cuda",
                   nms_threshold: float = 0.3) -> Inference:
    """
    Factory function to create detector instance
    Simplifies C++ binding
    
    Args:
        model_name: "owlv2" or "owlv2-large"
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for evaluation
        device: Device string ("cuda", "cpu", "cuda:0", etc.)
        nms_threshold: NMS IoU threshold
    
    Returns:
        Inference instance
    """
    return Inference(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        device=device,
        nms_threshold=nms_threshold
    )


def detect_from_buffer(detector: Inference,
                      image_buffer: bytes,
                      width: int,
                      height: int,
                      channels: int = 3,
                      text_prompts: Optional[List[str]] = None) -> FrameMetadata:
    """
    Detect from raw image buffer (for C++ memory management)
    
    Args:
        detector: Inference instance
        image_buffer: Raw image data as bytes
        width: Image width
        height: Image height
        channels: Number of channels (typically 3 for RGB/BGR)
        text_prompts: Optional custom prompts
    
    Returns:
        FrameMetadata with detection results
    """
    # Convert buffer to numpy array
    image_array = np.frombuffer(image_buffer, dtype=np.uint8)
    image_array = image_array.reshape((height, width, channels))
    
    return detector.detect_image(image_array, text_prompts)


# Export list for pybind11
__all__ = [
    'Inference',
    'Detection',
    'FrameMetadata',
    'DetectionType',
    'create_detector',
    'detect_from_buffer'
]


# ============================================================================
# Standalone usage example (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example: Test the module standalone
    detector = create_detector(
        model_name="owlv2",
        confidence_threshold=0.35,
        device="cuda"
    )
    
    # Warmup
    detector.warmup()
    
    # Test on sample image
    test_image = cv2.imread("../dataset/weapon_detection/train/images/Automatic Rifle_100.jpeg")
    result = detector.detect_image(test_image)
    
    print(f"\nTest Results:")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Detections: {len(result.detections)}")
    print(f"Has threat: {result.has_threat}")
    
    print("\nModel info:")
    print(detector.get_model_info())
