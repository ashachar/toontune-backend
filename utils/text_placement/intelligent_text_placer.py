#!/usr/bin/env python3
"""
Intelligent Text Placement System
==================================
Uses advanced OpenCV algorithms to extract backgrounds and place text
in the most prominent background areas of video frames.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import json
from PIL import Image
from rembg import remove
import warnings
warnings.filterwarnings("ignore")


class IntelligentTextPlacer:
    """Places text intelligently in video backgrounds."""
    
    def __init__(self, video_path: str, output_dir: str = None):
        """
        Initialize the text placer.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save background masks
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent / "backgrounds"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Initialized IntelligentTextPlacer for {self.video_path.name}")
        print(f"  Resolution: {self.width}x{self.height}, FPS: {self.fps}")
    
    def extract_background_at_time(self, timestamp: float) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract background at a specific timestamp and find best text position.
        
        Args:
            timestamp: Time in seconds to extract frame
            
        Returns:
            Tuple of (background_mask, (best_x, best_y))
        """
        # Seek to the frame
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame at {timestamp}s")
            return None, (self.width // 2, self.height // 2)
        
        # Use rembg inverse method for best results
        combined_mask = self._extract_using_rembg(frame)
        
        # Find best text position
        best_pos = self._find_best_text_position(combined_mask)
        
        # Save the mask for debugging
        mask_filename = self.output_dir / f"mask_t{timestamp:.3f}.png"
        cv2.imwrite(str(mask_filename), combined_mask)
        
        # Save visualization
        vis = self._create_visualization(frame, combined_mask, best_pos)
        vis_filename = self.output_dir / f"vis_t{timestamp:.3f}.png"
        cv2.imwrite(str(vis_filename), vis)
        
        return combined_mask, best_pos
    
    def _extract_using_rembg(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract background using rembg inverse method.
        Takes the pixels that rembg removes (makes transparent) as background.
        """
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply rembg to remove background
        nobg_img = remove(pil_image)
        
        # Convert to numpy array
        nobg_array = np.array(nobg_img)
        
        # Get the alpha channel (transparency mask)
        if nobg_array.shape[2] == 4:
            alpha = nobg_array[:, :, 3]
        else:
            # If no alpha channel, create one
            alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
        
        # The background is where alpha is low (transparent)
        # The foreground is where alpha is high (opaque)
        background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
        
        return background_mask
    
    def _extract_using_saliency(self, frame: np.ndarray) -> np.ndarray:
        """Extract background using saliency detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Static saliency detection (spectral residual)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(frame)
        
        if success:
            # Convert to 8-bit
            saliency_map = (saliency_map * 255).astype(np.uint8)
            
            # Threshold to get foreground (high saliency = foreground)
            _, foreground = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert to get background
            background = cv2.bitwise_not(foreground)
            return background
        else:
            # Fallback to simple threshold
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary
    
    def _extract_using_edge_detection(self, frame: np.ndarray) -> np.ndarray:
        """Extract background using edge detection and contour analysis."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for objects (foreground)
        foreground_mask = np.zeros(gray.shape, dtype=np.uint8)
        
        # Fill significant contours (likely objects)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                cv2.drawContours(foreground_mask, [contour], -1, 255, -1)
        
        # Invert to get background
        background_mask = cv2.bitwise_not(foreground_mask)
        return background_mask
    
    def _extract_using_color_clustering(self, frame: np.ndarray) -> np.ndarray:
        """Extract background using K-means color clustering."""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (self.width // 4, self.height // 4))
        
        # Convert to LAB color space (better for clustering)
        lab = cv2.cvtColor(small_frame, cv2.COLOR_BGR2LAB)
        
        # Reshape for K-means
        pixels = lab.reshape((-1, 3)).astype(np.float32)
        
        # K-means clustering
        K = 3  # Assume 3 main color regions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape labels back
        labels = labels.reshape(small_frame.shape[:2])
        
        # Find the most common label (likely background)
        unique, counts = np.unique(labels, return_counts=True)
        background_label = unique[np.argmax(counts)]
        
        # Create mask
        mask = (labels == background_label).astype(np.uint8) * 255
        
        # Resize back
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def _find_best_text_position(self, background_mask: np.ndarray) -> Tuple[int, int]:
        """
        Find the best position for text in the background.
        Uses multiple regions and randomization for variety.
        
        Args:
            background_mask: Binary mask where 255 = background
            
        Returns:
            Tuple of (x, y) for text placement
        """
        # Find connected components in background
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            background_mask, connectivity=8
        )
        
        if num_labels <= 1:
            # No background found, return center
            return (self.width // 2, self.height // 2)
        
        # Collect all viable regions
        viable_positions = []
        
        for i in range(1, num_labels):  # Skip label 0
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Only consider regions large enough for text
            if area > 2000:  # Minimum area threshold
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Generate multiple positions within this region
                # Use a grid to get diverse positions
                for dy in [0.25, 0.5, 0.75]:
                    for dx in [0.25, 0.5, 0.75]:
                        pos_x = int(x + w * dx)
                        pos_y = int(y + h * dy)
                        
                        # Check if position is actually in background
                        if background_mask[pos_y, pos_x] == 255:
                            # Keep within safe margins
                            pos_x = max(100, min(pos_x, self.width - 100))
                            pos_y = max(50, min(pos_y, self.height - 50))
                            viable_positions.append((pos_x, pos_y, area))
        
        if not viable_positions:
            return (self.width // 2, self.height // 2)
        
        # Sort by area (prefer larger background regions)
        viable_positions.sort(key=lambda p: p[2], reverse=True)
        
        # Add some randomization to avoid always picking the same spot
        # But weight towards larger areas
        num_candidates = min(5, len(viable_positions))
        candidates = viable_positions[:num_candidates]
        
        # Random selection from top candidates
        import random
        selected = random.choice(candidates)
        
        return (selected[0], selected[1])
    
    def _create_visualization(self, frame: np.ndarray, mask: np.ndarray, 
                             text_pos: Tuple[int, int]) -> np.ndarray:
        """Create a visualization showing background extraction and text position."""
        # Create colored overlay
        overlay = frame.copy()
        
        # Color the background green
        background_colored = np.zeros_like(frame)
        background_colored[:, :, 1] = mask  # Green channel
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.7, background_colored, 0.3, 0)
        
        # Draw text position marker
        cv2.circle(result, text_pos, 10, (255, 0, 255), -1)  # Magenta dot
        cv2.putText(result, "TEXT", 
                   (text_pos[0] - 20, text_pos[1] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return result
    
    def generate_word_positions(self, words_transcript: List[Dict]) -> List[Dict]:
        """
        Generate intelligent positions for all words in the transcript.
        
        Args:
            words_transcript: List of word dictionaries with 'word', 'start', 'end'
            
        Returns:
            List of word dictionaries with added 'x' and 'y' positions
        """
        positioned_words = []
        
        for i, word_data in enumerate(words_transcript):
            word = word_data.get('word', '')
            start_time = float(word_data.get('start', 0))
            end_time = float(word_data.get('end', start_time + 0.5))
            
            # Extract background at word start time
            _, (x, y) = self.extract_background_at_time(start_time)
            
            # Add position to word data
            positioned_word = {
                'word': word,
                'start': start_time,
                'end': end_time,
                'x': x,
                'y': y,
                'fontsize': 32  # Default size, will be scaled
            }
            
            positioned_words.append(positioned_word)
            
            print(f"  Word {i+1}/{len(words_transcript)}: '{word}' at ({x}, {y})")
        
        # Save positions to JSON
        positions_file = self.output_dir / "word_positions.json"
        with open(positions_file, 'w') as f:
            json.dump(positioned_words, f, indent=2)
        print(f"Saved word positions to {positions_file}")
        
        return positioned_words
    
    def __del__(self):
        """Clean up video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()


def demo_text_placement(video_path: str, output_dir: str = None):
    """Demo function to test text placement."""
    placer = IntelligentTextPlacer(video_path, output_dir)
    
    # Test at a few timestamps
    test_times = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    for t in test_times:
        mask, pos = placer.extract_background_at_time(t)
        print(f"Time {t}s: Best text position = {pos}")
    
    return placer


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else None
        demo_text_placement(video, output)
    else:
        print("Usage: python intelligent_text_placer.py <video_path> [output_dir]")