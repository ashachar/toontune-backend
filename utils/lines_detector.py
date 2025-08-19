#!/usr/bin/env python3
"""
Horizon and Line Detection Utilities
Combines DeepLabV3 semantic segmentation with Hough transform fallback
Optimized for M3 Max with MPS acceleration
"""

import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2
from scipy import signal
import time
from typing import Optional, Tuple, List, Dict, Union
import warnings
warnings.filterwarnings('ignore')


class DeepLabV3HorizonDetector:
    """
    DeepLabV3-based horizon detection using semantic segmentation
    """
    
    def __init__(self, model_type: str = 'mobilenet', device: Optional[str] = None):
        """
        Initialize DeepLabV3 for horizon detection
        
        Args:
            model_type: 'mobilenet' (fast) or 'resnet101' (accurate)
            device: Device to use ('mps', 'cuda', 'cpu'). Auto-detects if None.
        """
        # Auto-detect best device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"🎯 DeepLabV3 initializing on device: {self.device}")
        
        # Load model
        if model_type == 'mobilenet':
            print("📦 Loading DeepLabV3-MobileNet (fast mode)")
            self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
                pretrained=True
            ).to(self.device)
        else:
            print("📦 Loading DeepLabV3-ResNet101 (accurate mode)")
            self.model = models.segmentation.deeplabv3_resnet101(
                pretrained=True
            ).to(self.device)
        
        self.model.eval()
        
        # Preprocessing
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # Class IDs for sky detection (PASCAL VOC)
        self.sky_class = 14  # Sky class in PASCAL VOC
        
        print("✅ DeepLabV3 ready!")
    
    def detect_horizon(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
        """
        Detect horizon line using semantic segmentation
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            
        Returns:
            horizon_points: Array of (x, y) points along horizon, or None if not found
            angle: Angle of horizon line in degrees (0 = horizontal)
            segmentation_mask: The semantic segmentation mask
        """
        print("🔍 Detecting horizon with DeepLabV3...")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Start timing
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess(image_rgb)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        # Get segmentation mask
        segmentation_mask = output.argmax(0).cpu().numpy()
        
        # Find horizon line
        horizon_points, angle = self._extract_horizon_line(segmentation_mask, h, w)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # ms
        
        if horizon_points is not None:
            print(f"✅ Horizon detected in {process_time:.1f}ms, angle: {angle:.1f}°")
        else:
            print(f"⚠️ No clear horizon found in {process_time:.1f}ms")
        
        return horizon_points, angle, segmentation_mask
    
    def _extract_horizon_line(self, segmentation_mask: np.ndarray, h: int, w: int) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract horizon line from segmentation mask
        
        Returns:
            horizon_points: Array of (x, y) points
            angle: Angle in degrees
        """
        # Create binary mask (sky vs non-sky)
        sky_mask = (segmentation_mask == self.sky_class).astype(np.uint8)
        
        # Check if we have enough sky pixels
        sky_percentage = np.sum(sky_mask) / (h * w)
        if sky_percentage < 0.05:  # Less than 5% sky
            return None, 0.0
        
        # Find horizon points column by column
        horizon_points = []
        
        for x in range(0, w, 5):  # Sample every 5 pixels for speed
            column = sky_mask[:, x]
            # Find transition from sky to ground
            transitions = np.diff(column)
            horizon_y_candidates = np.where(transitions == -1)[0]
            
            if len(horizon_y_candidates) > 0:
                # Take the lowest sky-to-ground transition
                horizon_y = horizon_y_candidates[-1]
                horizon_points.append([x, horizon_y])
        
        if len(horizon_points) < 10:  # Need enough points for reliable line
            return None, 0.0
        
        horizon_points = np.array(horizon_points)
        
        # Fit a line to get angle
        angle = self._calculate_line_angle(horizon_points)
        
        # Smooth the horizon line
        horizon_points = self._smooth_horizon(horizon_points)
        
        return horizon_points, angle
    
    def _calculate_line_angle(self, points: np.ndarray) -> float:
        """
        Calculate angle of line fitted to points
        
        Returns:
            Angle in degrees (0 = horizontal, positive = tilted up to right)
        """
        # Fit a line using least squares
        x = points[:, 0]
        y = points[:, 1]
        
        # Fit polynomial of degree 1 (straight line)
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        
        # Convert slope to angle in degrees
        angle = np.degrees(np.arctan(slope))
        
        return angle
    
    def _smooth_horizon(self, points: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Smooth horizon line using Savitzky-Golay filter
        """
        if len(points) < window_size:
            return points
            
        smoothed = np.copy(points)
        
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
            
        # Smooth y-coordinates
        smoothed[:, 1] = signal.savgol_filter(
            points[:, 1], 
            window_size, 
            min(3, window_size - 1),  # polynomial order
            mode='nearest'
        )
        
        return smoothed


class HoughLineDetector:
    """
    Classical computer vision approach using Hough transform for line detection
    """
    
    def __init__(self):
        """Initialize Hough line detector"""
        print("📐 Hough Line Detector initialized")
    
    def detect_horizon(self, image: np.ndarray, 
                      prefer_horizontal: bool = True) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect horizon/prominent line using Hough transform
        
        Args:
            image: Input image as numpy array (H, W, 3) in BGR format
            prefer_horizontal: If True, prefer more horizontal lines
            
        Returns:
            line_points: Array of (x, y) points along the line, or None if not found
            angle: Angle of line in degrees (0 = horizontal)
        """
        print("🔍 Detecting lines with Hough transform...")
        
        start_time = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with Canny
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines with Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            process_time = (time.time() - start_time) * 1000
            print(f"⚠️ No lines detected in {process_time:.1f}ms")
            return None, 0.0
        
        # Find the best line (most horizontal or most prominent)
        best_line = self._select_best_line(lines, h, w, prefer_horizontal)
        
        if best_line is None:
            process_time = (time.time() - start_time) * 1000
            print(f"⚠️ No suitable line found in {process_time:.1f}ms")
            return None, 0.0
        
        # Convert line to points
        line_points, angle = self._line_to_points(best_line, w)
        
        process_time = (time.time() - start_time) * 1000
        print(f"✅ Line detected in {process_time:.1f}ms, angle: {angle:.1f}°")
        
        return line_points, angle
    
    def _select_best_line(self, lines: np.ndarray, h: int, w: int, 
                         prefer_horizontal: bool) -> Optional[Tuple[float, float]]:
        """
        Select the best line from detected lines
        
        Returns:
            (rho, theta) of the best line, or None
        """
        best_line = None
        best_score = -1
        
        for rho, theta in lines[:, 0]:
            # Calculate how horizontal the line is (0 or π is vertical, π/2 is horizontal)
            horizontalness = abs(np.sin(theta))  # 1 for horizontal, 0 for vertical
            
            # Calculate distance from image center
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Get y-coordinate at image center
            if abs(b) > 0.01:  # Not perfectly vertical
                y_center = (rho - a * (w / 2)) / b
                distance_from_center = abs(y_center - h / 2) / h
            else:
                distance_from_center = 1.0
            
            # Score calculation
            if prefer_horizontal:
                # Prefer horizontal lines near the center
                score = horizontalness * (1 - distance_from_center * 0.5)
            else:
                # Just prefer lines near the center
                score = 1 - distance_from_center
            
            if score > best_score:
                best_score = score
                best_line = (rho, theta)
        
        return best_line
    
    def _line_to_points(self, line: Tuple[float, float], width: int) -> Tuple[np.ndarray, float]:
        """
        Convert Hough line (rho, theta) to array of points
        
        Returns:
            points: Array of (x, y) points
            angle: Angle in degrees
        """
        rho, theta = line
        
        # Calculate line parameters
        a = np.cos(theta)
        b = np.sin(theta)
        
        # Generate points along the line
        points = []
        for x in range(0, width, 5):  # Sample every 5 pixels
            if abs(b) > 0.01:  # Not perfectly vertical
                y = (rho - a * x) / b
                if 0 <= y < 10000:  # Reasonable y value
                    points.append([x, int(y)])
        
        if len(points) < 2:
            # Fallback for vertical lines
            if abs(a) > 0.01:
                x = rho / a
                for y in range(0, 1000, 5):
                    points.append([int(x), y])
        
        points = np.array(points)
        
        # Calculate angle
        if len(points) >= 2:
            dx = points[-1, 0] - points[0, 0]
            dy = points[-1, 1] - points[0, 1]
            angle = np.degrees(np.arctan2(dy, dx))
        else:
            angle = 0.0
        
        return points, angle
    
    def detect_all_lines(self, image: np.ndarray, max_lines: int = 10) -> List[Dict]:
        """
        Detect multiple lines in the image
        
        Returns:
            List of dictionaries with line information
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None:
            return []
        
        detected_lines = []
        for i, (rho, theta) in enumerate(lines[:, 0][:max_lines]):
            points, angle = self._line_to_points((rho, theta), image.shape[1])
            
            detected_lines.append({
                'id': i,
                'points': points,
                'angle': angle,
                'rho': rho,
                'theta': theta,
                'horizontal_score': abs(np.sin(theta))
            })
        
        return detected_lines


class HorizonDetector:
    """
    Combined horizon detector with DeepLabV3 primary and Hough fallback
    """
    
    def __init__(self, use_fast_model: bool = True, device: Optional[str] = None):
        """
        Initialize combined horizon detector
        
        Args:
            use_fast_model: Use MobileNet (True) or ResNet101 (False)
            device: Device for DeepLabV3 ('mps', 'cuda', 'cpu')
        """
        print("🚀 Initializing Combined Horizon Detector")
        
        # Initialize DeepLabV3
        model_type = 'mobilenet' if use_fast_model else 'resnet101'
        self.deeplab = DeepLabV3HorizonDetector(model_type=model_type, device=device)
        
        # Initialize Hough detector
        self.hough = HoughLineDetector()
        
        print("✅ Combined detector ready!")
    
    def detect(self, image: np.ndarray, 
              force_method: Optional[str] = None) -> Dict:
        """
        Detect horizon using DeepLabV3 with Hough fallback
        
        Args:
            image: Input image (H, W, 3) in BGR format
            force_method: Force specific method ('deeplab' or 'hough')
            
        Returns:
            Dictionary with detection results
        """
        result = {
            'method': None,
            'horizon_points': None,
            'angle': 0.0,
            'confidence': 0.0,
            'time_ms': 0.0,
            'segmentation_mask': None
        }
        
        start_time = time.time()
        
        if force_method == 'hough':
            # Use only Hough
            points, angle = self.hough.detect_horizon(image)
            result['method'] = 'hough'
            result['horizon_points'] = points
            result['angle'] = angle
            result['confidence'] = 0.7 if points is not None else 0.0
            
        elif force_method == 'deeplab':
            # Use only DeepLabV3
            points, angle, mask = self.deeplab.detect_horizon(image)
            result['method'] = 'deeplab'
            result['horizon_points'] = points
            result['angle'] = angle
            result['segmentation_mask'] = mask
            result['confidence'] = 0.9 if points is not None else 0.0
            
        else:
            # Try DeepLabV3 first
            points, angle, mask = self.deeplab.detect_horizon(image)
            
            if points is not None and len(points) > 10:
                # DeepLabV3 succeeded
                result['method'] = 'deeplab'
                result['horizon_points'] = points
                result['angle'] = angle
                result['segmentation_mask'] = mask
                result['confidence'] = 0.9
            else:
                # Fallback to Hough
                print("⚠️ DeepLabV3 failed, falling back to Hough transform...")
                points, angle = self.hough.detect_horizon(image)
                result['method'] = 'hough'
                result['horizon_points'] = points
                result['angle'] = angle
                result['confidence'] = 0.7 if points is not None else 0.0
        
        result['time_ms'] = (time.time() - start_time) * 1000
        
        if result['horizon_points'] is not None:
            print(f"✅ Detection complete using {result['method']}")
            print(f"   Angle: {result['angle']:.1f}°")
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Time: {result['time_ms']:.1f}ms")
        else:
            print(f"❌ No horizon detected with either method")
        
        return result
    
    def select_tracking_point(self, horizon_points: np.ndarray, 
                            position: str = 'center') -> Tuple[float, float]:
        """
        Select a point on the horizon for tracking
        
        Args:
            horizon_points: Array of (x, y) points along horizon
            position: 'center', 'left', 'right', or 'random'
            
        Returns:
            (x, y) coordinates of selected point
        """
        if horizon_points is None or len(horizon_points) == 0:
            raise ValueError("No horizon points provided")
        
        if position == 'center':
            idx = len(horizon_points) // 2
        elif position == 'left':
            idx = len(horizon_points) // 4
        elif position == 'right':
            idx = 3 * len(horizon_points) // 4
        elif position == 'random':
            idx = np.random.randint(0, len(horizon_points))
        else:
            idx = len(horizon_points) // 2
        
        point = horizon_points[idx]
        return float(point[0]), float(point[1])
    
    def visualize_detection(self, image: np.ndarray, result: Dict, 
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize horizon detection results
        
        Returns:
            Image with visualization
        """
        vis_image = image.copy()
        
        if result['horizon_points'] is not None:
            points = result['horizon_points']
            
            # Draw horizon line
            for i in range(len(points) - 1):
                cv2.line(vis_image, 
                        tuple(points[i].astype(int)), 
                        tuple(points[i+1].astype(int)), 
                        (0, 255, 0), 2)
            
            # Add text overlay
            text = f"{result['method'].upper()} | Angle: {result['angle']:.1f}° | {result['time_ms']:.1f}ms"
            cv2.putText(vis_image, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mark center point
            center_idx = len(points) // 2
            center_point = points[center_idx]
            cv2.circle(vis_image, tuple(center_point.astype(int)), 5, (0, 0, 255), -1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"✅ Visualization saved to: {save_path}")
        
        return vis_image


def benchmark_detectors(image_path: str):
    """
    Benchmark both detection methods
    """
    print("=" * 60)
    print("BENCHMARKING HORIZON DETECTORS")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    print("-" * 60)
    
    # Test DeepLabV3
    print("\n1. DeepLabV3-MobileNet:")
    deeplab_fast = DeepLabV3HorizonDetector(model_type='mobilenet')
    
    # Warm up
    _, _, _ = deeplab_fast.detect_horizon(image)
    
    # Benchmark
    times = []
    for _ in range(5):
        start = time.time()
        points, angle, _ = deeplab_fast.detect_horizon(image)
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"   Average time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
    print(f"   Points detected: {len(points) if points is not None else 0}")
    
    # Test Hough
    print("\n2. Hough Transform:")
    hough = HoughLineDetector()
    
    times = []
    for _ in range(5):
        start = time.time()
        points, angle = hough.detect_horizon(image)
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    print(f"   Average time: {avg_time:.1f}ms ({1000/avg_time:.1f} FPS)")
    print(f"   Points detected: {len(points) if points is not None else 0}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test with a sample image
    print("Testing Horizon Detector...")
    
    # Create a simple test
    detector = HorizonDetector(use_fast_model=True)
    
    # You can test with any image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    test_image[:240, :] = [200, 200, 255]  # Sky-like top half
    
    result = detector.detect(test_image)
    print(f"\nTest result: {result['method']} detected horizon" 
          if result['horizon_points'] is not None else "No horizon detected")