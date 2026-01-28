"""
Usage examples for InferenceCV module
Demonstrates both standalone Python usage and C++ integration patterns
"""

import cv2
import numpy as np
from inference import create_detector, Inference, FrameMetadata
import json


# ============================================================================
# Example 1: Basic Image Detection (Python)
# ============================================================================

def example_single_image():
    """Process a single image"""
    print("=" * 60)
    print("Example 1: Single Image Detection")
    print("=" * 60)
    
    # Create detector
    detector = create_detector(
        model_name="owlv2",
        confidence_threshold=0.35,
        device="cuda"
    )
    
    # Warmup (important for accurate timing)
    detector.warmup()
    
    # Load image
    image_path = "../dataset/weapon_detection/train/images/Automatic Rifle_100.jpeg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Creating dummy image for demonstration")
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Run detection
    result = detector.detect_image(image)
    
    # Print results
    print(f"\nResults:")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Resolution: {result.resolution[0]}x{result.resolution[1]}")
    print(f"  Threat detected: {result.has_threat}")
    print(f"  Total detections: {len(result.detections)}")
    
    # Print each detection
    for i, det in enumerate(result.detections):
        print(f"\n  Detection {i+1}:")
        print(f"    Label: {det.label}")
        print(f"    Type: {det.detection_type}")
        print(f"    Confidence: {det.confidence:.3f}")
        print(f"    Bbox: [{det.bbox[0]:.1f}, {det.bbox[1]:.1f}, "
              f"{det.bbox[2]:.1f}, {det.bbox[3]:.1f}]")
        
        # Draw on image
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det.label}: {det.confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save annotated image
    cv2.imwrite("output_annotated.jpg", image)
    print(f"\n  Annotated image saved to: output_annotated.jpg")


# ============================================================================
# Example 2: Video Processing (Python)
# ============================================================================

def example_video_processing():
    """Process a video file"""
    print("\n" + "=" * 60)
    print("Example 2: Video Processing")
    print("=" * 60)
    
    detector = create_detector(
        model_name="owlv2",
        confidence_threshold=0.35,
        device="cuda"
    )
    
    video_path = "evidence/bodycam_footage.mp4"
    
    # Process video (skip every 5 frames for speed)
    results = detector.detect_video(video_path, frame_skip=5)
    
    # Analyze results
    total_frames = len(results)
    frames_with_threats = sum(1 for r in results if r.has_threat)
    avg_time = np.mean([r.processing_time_ms for r in results])
    total_detections = sum(len(r.detections) for r in results)
    
    print(f"\nVideo Analysis Summary:")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Frames with threats: {frames_with_threats} ({frames_with_threats/total_frames*100:.1f}%)")
    print(f"  Total detections: {total_detections}")
    print(f"  Average processing time: {avg_time:.2f}ms/frame")
    print(f"  Estimated FPS: {1000.0/avg_time:.2f}")
    
    # Print frames with threats
    print(f"\n  Threat timeline:")
    for result in results:
        if result.has_threat:
            print(f"    Frame {result.frame_id}: {len(result.detections)} detection(s)")


# ============================================================================
# Example 3: Batch Processing (Python)
# ============================================================================

def example_batch_processing():
    """Process multiple images in batch"""
    print("\n" + "=" * 60)
    print("Example 3: Batch Image Processing")
    print("=" * 60)
    
    detector = create_detector(
        model_name="owlv2",
        confidence_threshold=0.35,
        device="cuda"
    )
    
    # Create or load multiple images
    image_paths = [
        "evidence/scene_001.jpg",
        "evidence/scene_002.jpg",
        "evidence/cctv_frame_001.jpg"
    ]
    
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            # Create dummy if file doesn't exist
            img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        images.append(img)
    
    # Process batch
    results = detector.detect_image_batch(images)
    
    # Summary
    threats_found = sum(1 for r in results if r.has_threat)
    avg_time = np.mean([r.processing_time_ms for r in results])
    
    print(f"\nBatch Processing Summary:")
    print(f"  Images processed: {len(results)}")
    print(f"  Images with threats: {threats_found}")
    print(f"  Average processing time: {avg_time:.2f}ms/image")
    
    for i, result in enumerate(results):
        print(f"\n  Image {i+1}:")
        print(f"    Threat: {result.has_threat}")
        print(f"    Detections: {len(result.detections)}")


# ============================================================================
# Example 4: Custom Prompts and Threshold Adjustment (Python)
# ============================================================================

def example_custom_configuration():
    """Demonstrate custom prompts and dynamic threshold adjustment"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    
    detector = create_detector(
        model_name="owlv2",
        confidence_threshold=0.40,  # Start with higher threshold
        device="cuda"
    )
    
    # Custom prompts for specific weapons
    custom_prompts = [
        "a photo of a handgun",
        "a photo of a knife",
        "a photo of a rifle",
        "a person holding a weapon"
    ]
    
    # Test image
    image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # First detection with custom prompts
    print(f"\nDetection 1: High threshold (0.40)")
    result1 = detector.detect_image(image, text_prompts=custom_prompts)
    print(f"  Detections: {len(result1.detections)}")
    
    # Adjust threshold for higher sensitivity
    detector.set_threshold(confidence=0.25, nms=0.4)
    
    print(f"\nDetection 2: Lower threshold (0.25)")
    result2 = detector.detect_image(image, text_prompts=custom_prompts)
    print(f"  Detections: {len(result2.detections)}")
    
    # Get model info
    print(f"\nModel Information:")
    info = detector.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")


# ============================================================================
# Example 5: Simulating C++ Integration Pattern
# ============================================================================

def example_cpp_integration_pattern():
    """
    Demonstrates how C++ would interact with the module
    This simulates the data flow from C++ through pybind11
    """
    print("\n" + "=" * 60)
    print("Example 5: C++ Integration Pattern")
    print("=" * 60)
    
    # Step 1: C++ creates detector (via create_detector)
    print("\n[C++] Creating detector...")
    detector = create_detector(
        model_name="owlv2",
        confidence_threshold=0.35,
        device="cuda"
    )
    
    # Step 2: C++ warms up model
    print("[C++] Warming up model...")
    detector.warmup()
    
    # Step 3: C++ has image buffer in memory
    print("[C++] Preparing image buffer...")
    width, height, channels = 1920, 1080, 3
    image_size = width * height * channels
    
    # Simulate C++ memory buffer (contiguous memory)
    cpp_buffer = np.random.randint(0, 255, image_size, dtype=np.uint8)
    cpp_buffer_bytes = cpp_buffer.tobytes()
    
    print(f"[C++] Buffer size: {len(cpp_buffer_bytes)} bytes")
    print(f"[C++] Image dimensions: {width}x{height}x{channels}")
    
    # Step 4: C++ calls detect_from_buffer
    print("[C++] Calling detect_from_buffer...")
    from inference import detect_from_buffer
    
    result = detect_from_buffer(
        detector=detector,
        image_buffer=cpp_buffer_bytes,
        width=width,
        height=height,
        channels=channels,
        text_prompts=None  # Use default
    )
    
    # Step 5: C++ receives FrameMetadata
    print(f"\n[C++] Received results:")
    print(f"  Frame ID: {result.frame_id}")
    print(f"  Resolution: {result.resolution}")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Has threat: {result.has_threat}")
    print(f"  Detection count: {len(result.detections)}")
    
    # Step 6: C++ iterates through detections
    for i, det in enumerate(result.detections):
        print(f"\n  [C++] Detection {i+1}:")
        print(f"    Label: {det.label}")
        print(f"    Confidence: {det.confidence:.3f}")
        print(f"    Bbox: [{det.bbox[0]:.1f}, {det.bbox[1]:.1f}, "
              f"{det.bbox[2]:.1f}, {det.bbox[3]:.1f}]")
    
    # Step 7: Convert to JSON for C++ (optional)
    print(f"\n[C++] Converting to JSON...")
    json_data = result.to_dict()
    json_str = json.dumps(json_data, indent=2)
    print(f"  JSON size: {len(json_str)} bytes")
    print(f"  JSON preview: {json_str[:200]}...")


# ============================================================================
# Example 6: Performance Benchmarking
# ============================================================================

def example_performance_benchmark():
    """Benchmark the detection performance"""
    print("\n" + "=" * 60)
    print("Example 6: Performance Benchmark")
    print("=" * 60)
    
    detector = create_detector(
        model_name="owlv2",
        confidence_threshold=0.35,
        device="cuda"
    )
    
    # Test different resolutions
    resolutions = [
        (640, 480),    # VGA
        (1280, 720),   # HD
        (1920, 1080),  # Full HD
        (3840, 2160),  # 4K
    ]
    
    print("\nBenchmarking different resolutions:")
    print(f"{'Resolution':<15} {'Time (ms)':<12} {'FPS':<8}")
    print("-" * 40)
    
    for width, height in resolutions:
        # Create test image
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Run detection multiple times and average
        times = []
        for _ in range(5):
            result = detector.detect_image(image)
            times.append(result.processing_time_ms)
        
        avg_time = np.mean(times)
        fps = 1000.0 / avg_time
        
        print(f"{width}x{height:<8} {avg_time:>10.2f}   {fps:>6.2f}")


# ============================================================================
# Main - Run all examples
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("InferenceCV Module - Usage Examples")
    print("=" * 60)
    
    try:
        example_single_image()
        # example_video_processing()  # Requires actual video file
        # example_batch_processing()
        #example_custom_configuration()
        #example_cpp_integration_pattern()
        #example_performance_benchmark()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
