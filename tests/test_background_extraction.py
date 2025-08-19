#!/usr/bin/env python3
"""
Test script for iterating on background extraction algorithms.
Focuses on frame at second 24 of scene 1.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.cluster import KMeans
import sys
sys.path.append(str(Path(__file__).parent.parent))


class BackgroundExtractor:
    """Improved background extraction focusing on edge detection."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {self.width}x{self.height} @ {self.fps}fps")
    
    def extract_frame(self, timestamp: float) -> np.ndarray:
        """Extract frame at specific timestamp."""
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame at {timestamp}s")
        return frame
    
    def method1_grabcut(self, frame: np.ndarray) -> np.ndarray:
        """Use GrabCut algorithm for foreground/background segmentation."""
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Initialize rectangle around center (assuming subjects are in center)
        h, w = frame.shape[:2]
        rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
        
        # Run GrabCut
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Extract background mask
        mask2 = np.where((mask == 0) | (mask == 2), 255, 0).astype('uint8')
        return mask2
    
    def method2_edge_based(self, frame: np.ndarray) -> np.ndarray:
        """Improved edge-based detection focusing on people detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(filtered, 30, 100)
        edges2 = cv2.Canny(filtered, 50, 150)
        edges3 = cv2.Canny(filtered, 100, 200)
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Close gaps in edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create foreground mask from significant contours
        fg_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                # Check if contour is roughly person-shaped (aspect ratio)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                
                # People are typically taller than wide
                if 1.5 < aspect_ratio < 4.0 or area > 5000:
                    cv2.drawContours(fg_mask, [contour], -1, 255, -1)
        
        # Dilate to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Background is inverse of foreground
        bg_mask = cv2.bitwise_not(fg_mask)
        return bg_mask
    
    def method3_skin_detection(self, frame: np.ndarray) -> np.ndarray:
        """Detect skin regions as foreground (people)."""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([235, 173, 127], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to include clothing around skin areas
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        people_mask = cv2.dilate(skin_mask, kernel_large, iterations=3)
        
        # Background is inverse
        bg_mask = cv2.bitwise_not(people_mask)
        return bg_mask
    
    def method4_semantic_segmentation(self, frame: np.ndarray) -> np.ndarray:
        """Use semantic understanding of outdoor scenes."""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect grass (green regions)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        grass_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Detect sky (blue/white regions in upper part)
        h, w = frame.shape[:2]
        sky_region = np.zeros_like(grass_mask)
        
        # Sky is typically in upper third
        upper_third = frame[:h//3, :]
        upper_hsv = cv2.cvtColor(upper_third, cv2.COLOR_BGR2HSV)
        
        # Blue sky
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_sky = cv2.inRange(upper_hsv, lower_blue, upper_blue)
        
        # White/gray sky (clouds)
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([180, 30, 255])
        white_sky = cv2.inRange(upper_hsv, lower_white, upper_white)
        
        sky_mask = cv2.bitwise_or(blue_sky, white_sky)
        sky_region[:h//3, :] = sky_mask
        
        # Combine grass and sky as background
        bg_mask = cv2.bitwise_or(grass_mask, sky_region)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
        
        return bg_mask
    
    def method5_motion_based(self, frame: np.ndarray, prev_frame: np.ndarray = None) -> np.ndarray:
        """Use motion detection (static = background)."""
        if prev_frame is None:
            # Get previous frame
            current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 10))
            ret, prev_frame = self.cap.read()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        # Calculate frame difference
        diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get motion mask
        _, motion_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Static regions are background
        bg_mask = cv2.bitwise_not(motion_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel)
        
        return bg_mask
    
    def method6_combined(self, frame: np.ndarray) -> np.ndarray:
        """Combine multiple methods with weighted voting."""
        # Get all masks
        mask1 = self.method2_edge_based(frame)
        mask2 = self.method3_skin_detection(frame)
        mask3 = self.method4_semantic_segmentation(frame)
        
        # Convert to float for averaging
        combined = (mask1.astype(float) + mask2.astype(float) + mask3.astype(float)) / 3.0
        
        # Threshold
        final_mask = (combined > 127).astype(np.uint8) * 255
        
        # Post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(final_mask, contours, -1, 255, -1)
        
        return final_mask
    
    def find_best_text_position(self, bg_mask: np.ndarray) -> Tuple[int, int]:
        """Find optimal text position in background."""
        # Find largest background region
        contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (self.width // 2, self.height // 2)
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest)
        
        # Find center of upper portion (better for text visibility)
        text_x = x + w // 2
        text_y = y + h // 3
        
        return (text_x, text_y)
    
    def visualize_all_methods(self, timestamp: float = 24.0):
        """Test and visualize all methods."""
        # Extract frame
        frame = self.extract_frame(timestamp)
        
        # Apply all methods
        methods = {
            'Original': frame,
            'GrabCut': self.method1_grabcut(frame),
            'Edge-Based': self.method2_edge_based(frame),
            'Skin Detection': self.method3_skin_detection(frame),
            'Semantic (Grass+Sky)': self.method4_semantic_segmentation(frame),
            'Combined': self.method6_combined(frame)
        }
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(methods.items()):
            ax = axes[idx]
            
            if name == 'Original':
                ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                # Show mask
                ax.imshow(result, cmap='gray')
                
                # Find and mark text position
                text_pos = self.find_best_text_position(result)
                ax.plot(text_pos[0], text_pos[1], 'r*', markersize=20)
                
                # Overlay mask on original for better visualization
                overlay = frame.copy()
                overlay[result == 0] = overlay[result == 0] * 0.3  # Darken foreground
                
                # Show in corner
                ax_inset = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
                ax_inset.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                ax_inset.axis('off')
            
            ax.set_title(name)
            ax.axis('off')
        
        plt.suptitle(f'Background Extraction Methods - Frame at {timestamp}s')
        plt.tight_layout()
        
        # Save results
        output_dir = Path('tests/background_extraction_results')
        output_dir.mkdir(exist_ok=True, parents=True)
        
        plt.savefig(output_dir / f'comparison_t{timestamp}.png', dpi=150, bbox_inches='tight')
        
        # Save individual masks
        for name, result in methods.items():
            if name != 'Original':
                cv2.imwrite(str(output_dir / f'{name.lower().replace(" ", "_")}_t{timestamp}.png'), result)
        
        # plt.show()  # Don't show GUI
        plt.close()
        print(f"Results saved to {output_dir}")
        
        return methods


def main():
    """Test background extraction on scene 1."""
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return
    
    extractor = BackgroundExtractor(video_path)
    
    # Test at second 24
    print("\nTesting background extraction at t=24s...")
    results = extractor.visualize_all_methods(24.0)
    
    # Find best method
    print("\nEvaluating methods...")
    frame = extractor.extract_frame(24.0)
    
    for name, mask in results.items():
        if name != 'Original':
            # Calculate percentage of image marked as background
            if isinstance(mask, np.ndarray) and mask.shape == frame.shape[:2]:
                bg_percentage = (np.sum(mask == 255) / mask.size) * 100
                print(f"{name}: {bg_percentage:.1f}% marked as background")
    
    # Test on a few more timestamps
    test_times = [10.0, 20.0, 30.0, 40.0]
    print("\nTesting on additional timestamps...")
    
    for t in test_times:
        print(f"\nTesting at t={t}s...")
        frame = extractor.extract_frame(t)
        mask = extractor.method6_combined(frame)  # Use best method
        text_pos = extractor.find_best_text_position(mask)
        print(f"  Best text position: {text_pos}")
        
        # Save result
        output_dir = Path('tests/background_extraction_results')
        cv2.imwrite(str(output_dir / f'combined_t{t}.png'), mask)


if __name__ == "__main__":
    main()