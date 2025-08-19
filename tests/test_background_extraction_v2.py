#!/usr/bin/env python3
"""
Improved background extraction using person detection.
Focuses on accurately detecting people as foreground.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List


class ImprovedBackgroundExtractor:
    """Better background extraction for outdoor scenes with people."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {self.width}x{self.height} @ {self.fps}fps")
        
        # Initialize HOG person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def extract_frame(self, timestamp: float) -> np.ndarray:
        """Extract frame at specific timestamp."""
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame at {timestamp}s")
        return frame
    
    def detect_people_hog(self, frame: np.ndarray) -> np.ndarray:
        """Detect people using HOG descriptor."""
        # Resize for faster detection
        scale = 0.5
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Detect people
        (rects, weights) = self.hog.detectMultiScale(
            small,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )
        
        # Create mask
        people_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for (x, y, w, h) in rects:
            # Scale back to original size
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            
            # Expand bounding box slightly
            x = max(0, x - int(w * 0.1))
            y = max(0, y - int(h * 0.1))
            w = min(frame.shape[1] - x, int(w * 1.2))
            h = min(frame.shape[0] - y, int(h * 1.2))
            
            # Fill rectangle
            cv2.rectangle(people_mask, (x, y), (x + w, y + h), 255, -1)
        
        return people_mask
    
    def detect_clothing_colors(self, frame: np.ndarray) -> np.ndarray:
        """Detect typical clothing colors as foreground."""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        masks = []
        
        # Detect various clothing colors
        # White/light clothing
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 60, 255])
        masks.append(cv2.inRange(hsv, lower_white, upper_white))
        
        # Brown/beige clothing
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([25, 255, 200])
        masks.append(cv2.inRange(hsv, lower_brown, upper_brown))
        
        # Dark clothing
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        masks.append(cv2.inRange(hsv, lower_dark, upper_dark))
        
        # Combine all clothing masks
        clothing_mask = np.zeros_like(masks[0])
        for mask in masks:
            clothing_mask = cv2.bitwise_or(clothing_mask, mask)
        
        # Clean up - remove small regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
        
        # Only keep regions that form vertical clusters (likely people)
        contours, _ = cv2.findContours(clothing_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(clothing_mask)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            
            # People are typically taller than wide
            if aspect_ratio > 1.2 or cv2.contourArea(contour) > 1000:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
    def detect_grass_and_sky(self, frame: np.ndarray) -> np.ndarray:
        """Specifically detect grass and sky as background."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Detect grass (green with specific texture)
        lower_green = np.array([35, 25, 25])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Grass is typically in lower 2/3 of outdoor scenes
        grass_weight = np.ones_like(grass_mask, dtype=np.float32)
        grass_weight[:h//3, :] *= 0.3  # Reduce weight in upper third
        grass_mask = (grass_mask.astype(np.float32) * grass_weight).astype(np.uint8)
        
        # Detect sky - more sophisticated
        sky_mask = np.zeros_like(grass_mask)
        
        # Sky is in upper portion
        upper_half = hsv[:h//2, :]
        
        # Blue sky
        lower_blue = np.array([90, 20, 50])
        upper_blue = np.array([130, 255, 255])
        blue_sky = cv2.inRange(upper_half, lower_blue, upper_blue)
        
        # Grayish/white sky (cloudy)
        lower_gray = np.array([0, 0, 120])
        upper_gray = np.array([180, 40, 255])
        gray_sky = cv2.inRange(upper_half, lower_gray, upper_gray)
        
        # Combine sky masks
        sky_mask[:h//2, :] = cv2.bitwise_or(blue_sky, gray_sky)
        
        # Mountains/hills (darker regions in middle distance)
        middle_third = frame[h//3:2*h//3, :]
        gray_middle = cv2.cvtColor(middle_third, cv2.COLOR_BGR2GRAY)
        _, dark_regions = cv2.threshold(gray_middle, 100, 255, cv2.THRESH_BINARY_INV)
        
        # Add dark regions as potential background
        mountain_mask = np.zeros_like(grass_mask)
        mountain_mask[h//3:2*h//3, :] = dark_regions
        
        # Combine all background elements
        bg_mask = cv2.bitwise_or(grass_mask, sky_mask)
        bg_mask = cv2.bitwise_or(bg_mask, mountain_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
        
        return bg_mask
    
    def extract_background_advanced(self, frame: np.ndarray) -> np.ndarray:
        """Advanced background extraction combining multiple techniques."""
        
        # 1. Detect people
        people_mask = self.detect_people_hog(frame)
        
        # 2. Detect clothing regions
        clothing_mask = self.detect_clothing_colors(frame)
        
        # 3. Combine people detections
        foreground = cv2.bitwise_or(people_mask, clothing_mask)
        
        # 4. Detect background elements (grass, sky)
        background = self.detect_grass_and_sky(frame)
        
        # 5. Ensure foreground and background don't overlap
        background = cv2.bitwise_and(background, cv2.bitwise_not(foreground))
        
        # 6. Fill remaining areas based on location
        h, w = frame.shape[:2]
        
        # Create location-based priors
        location_prior = np.zeros(frame.shape[:2], dtype=np.float32)
        
        # Upper region more likely to be background (sky)
        location_prior[:h//3, :] = 0.7
        
        # Lower edges more likely to be background (grass)
        location_prior[2*h//3:, :] = 0.6
        location_prior[2*h//3:, :w//4] = 0.8
        location_prior[2*h//3:, 3*w//4:] = 0.8
        
        # Apply prior to uncertain regions
        uncertain = cv2.bitwise_and(
            cv2.bitwise_not(foreground),
            cv2.bitwise_not(background)
        )
        
        prior_mask = (location_prior * 255).astype(np.uint8)
        uncertain_bg = cv2.bitwise_and(uncertain, prior_mask)
        
        # Final background mask
        final_bg = cv2.bitwise_or(background, uncertain_bg)
        
        # Post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        final_bg = cv2.morphologyEx(final_bg, cv2.MORPH_CLOSE, kernel)
        final_bg = cv2.morphologyEx(final_bg, cv2.MORPH_OPEN, kernel)
        
        # Ensure we don't include detected people
        final_bg = cv2.bitwise_and(final_bg, cv2.bitwise_not(foreground))
        
        return final_bg
    
    def find_safe_text_regions(self, bg_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find multiple safe regions for text placement."""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bg_mask, connectivity=8
        )
        
        safe_regions = []
        
        for i in range(1, num_labels):  # Skip background label 0
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Only consider large enough regions
            if area > 5000:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Find center of region
                cx = x + w // 2
                cy = y + h // 3  # Upper third for better visibility
                
                # Add some randomization to avoid all text in same spot
                cx += np.random.randint(-w//4, w//4) if w > 100 else 0
                cy += np.random.randint(-h//6, h//6) if h > 50 else 0
                
                # Keep within bounds
                cx = max(50, min(cx, self.width - 50))
                cy = max(30, min(cy, self.height - 30))
                
                safe_regions.append((cx, cy))
        
        # Sort by y-coordinate (top to bottom)
        safe_regions.sort(key=lambda p: p[1])
        
        return safe_regions if safe_regions else [(self.width // 2, self.height // 3)]
    
    def visualize_results(self, timestamp: float = 24.0):
        """Visualize the improved background extraction."""
        frame = self.extract_frame(timestamp)
        
        # Get background mask
        bg_mask = self.extract_background_advanced(frame)
        
        # Find safe text regions
        text_positions = self.find_safe_text_regions(bg_mask)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original frame
        axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # Background mask
        axes[1].imshow(bg_mask, cmap='gray')
        axes[1].set_title('Background Mask (White = Background)')
        axes[1].axis('off')
        
        # Overlay with text positions
        overlay = frame.copy()
        # Darken foreground
        overlay[bg_mask == 0] = overlay[bg_mask == 0] * 0.3
        
        # Mark text positions
        for i, (x, y) in enumerate(text_positions[:5]):  # Show first 5 positions
            cv2.circle(overlay, (x, y), 15, (0, 255, 0), -1)
            cv2.putText(overlay, f"T{i+1}", (x-10, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Background with Text Positions')
        axes[2].axis('off')
        
        plt.suptitle(f'Improved Background Extraction - t={timestamp}s')
        plt.tight_layout()
        
        # Save
        output_dir = Path('tests/background_extraction_results')
        output_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_dir / f'improved_t{timestamp}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save mask
        cv2.imwrite(str(output_dir / f'improved_mask_t{timestamp}.png'), bg_mask)
        
        # Calculate statistics
        bg_percentage = (np.sum(bg_mask == 255) / bg_mask.size) * 100
        print(f"Background percentage: {bg_percentage:.1f}%")
        print(f"Found {len(text_positions)} safe text regions")
        print(f"Text positions: {text_positions[:5]}")
        
        return bg_mask, text_positions


def test_multiple_timestamps():
    """Test on multiple timestamps."""
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    extractor = ImprovedBackgroundExtractor(video_path)
    
    # Test on multiple timestamps
    test_times = [10.0, 15.0, 20.0, 24.0, 30.0, 35.0, 40.0]
    
    all_positions = []
    
    for t in test_times:
        print(f"\nTesting at t={t}s...")
        try:
            mask, positions = extractor.visualize_results(t)
            all_positions.extend(positions[:3])  # Take top 3 from each frame
        except Exception as e:
            print(f"  Error at t={t}: {e}")
    
    # Show position distribution
    if all_positions:
        print(f"\nOverall text position distribution:")
        x_coords = [p[0] for p in all_positions]
        y_coords = [p[1] for p in all_positions]
        print(f"  X range: {min(x_coords)} - {max(x_coords)}")
        print(f"  Y range: {min(y_coords)} - {max(y_coords)}")
        print(f"  Avg position: ({np.mean(x_coords):.0f}, {np.mean(y_coords):.0f})")


if __name__ == "__main__":
    test_multiple_timestamps()