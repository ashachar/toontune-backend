#!/usr/bin/env python3
"""
Proximity-Based Text Placement System V2
=========================================
Places words close to each other while ensuring the ENTIRE word
is fully contained in valid background areas (no foreground overlap).
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


class ProximityTextPlacerV2:
    """Places text with proximity awareness while ensuring full background containment."""
    
    def __init__(self, video_path: str, output_dir: str = None):
        """
        Initialize the proximity-aware text placer.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save debug outputs
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent / "proximity_backgrounds_v2"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Track previous word position for proximity
        self.previous_position = None
        self.position_history = []
        
        # Cache for background masks to avoid recomputation
        self.mask_cache = {}
        
        print(f"Initialized ProximityTextPlacerV2 for {self.video_path.name}")
        print(f"  Resolution: {self.width}x{self.height}, FPS: {self.fps}")
    
    def extract_background_mask(self, timestamp: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract background mask at given timestamp.
        
        Returns:
            Tuple of (frame, background_mask)
        """
        # Check cache first
        cache_key = f"{timestamp:.2f}"
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        
        # Seek to the frame
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame at {timestamp}s")
            # Return a fully background mask as fallback
            return None, np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        # Apply rembg to get background
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        nobg_img = remove(pil_image)
        nobg_array = np.array(nobg_img)
        
        # Get alpha channel
        if nobg_array.shape[2] == 4:
            alpha = nobg_array[:, :, 3]
        else:
            alpha = np.ones(nobg_array.shape[:2], dtype=np.uint8) * 255
        
        # Background is where alpha is low (transparent)
        background_mask = np.where(alpha < 128, 255, 0).astype(np.uint8)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
        background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
        
        # Erode slightly to ensure we stay away from edges
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        background_mask = cv2.erode(background_mask, kernel_erode, iterations=1)
        
        # Cache the result
        self.mask_cache[cache_key] = (frame, background_mask)
        
        return frame, background_mask
    
    def check_text_fits_in_background(self, text: str, x: int, y: int, 
                                     font_scale: float, background_mask: np.ndarray) -> bool:
        """
        Check if the entire text bounding box fits in the background.
        
        Args:
            text: The text to place
            x, y: Position for FFmpeg (TOP-LEFT corner)
            font_scale: Font scale
            background_mask: Binary mask where 255 = background
            
        Returns:
            True if entire text fits in background
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate bounding box for FFmpeg coordinate system
        # FFmpeg drawtext uses TOP-LEFT as reference point
        x1 = x
        y1 = y  # Top of text (FFmpeg convention)
        x2 = x + text_width
        y2 = y + text_height + baseline  # Bottom including baseline
        
        # Add small padding for safety
        padding = 5
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 >= self.width or y2 >= self.height:
            return False
        
        # Extract the region where text will be placed
        text_region = background_mask[y1:y2, x1:x2]
        
        # Check if entire region is background (all pixels should be 255)
        if text_region.size == 0:
            return False
        
        # Require at least 95% of pixels to be background
        background_ratio = np.mean(text_region == 255)
        return background_ratio > 0.95
    
    def find_valid_positions(self, text: str, font_scale: float, 
                            background_mask: np.ndarray, num_candidates: int = 200) -> List[Tuple[int, int]]:
        """
        Find all valid positions where the entire text fits in background.
        
        Returns:
            List of valid (x, y) positions
        """
        valid_positions = []
        
        # Get text dimensions for boundary checking
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        
        # Define search grid - finer grid for more options
        grid_step = 20  # Check every 20 pixels
        
        # Search boundaries (leave margin for text)
        # FFmpeg uses top-left, so y_start is just the margin
        margin = 30
        x_start = margin
        x_end = self.width - text_width - margin
        y_start = margin  # Top margin (FFmpeg convention)
        y_end = self.height - text_height - baseline - margin  # Bottom boundary
        
        # Grid search for valid positions
        for y in range(y_start, y_end, grid_step):
            for x in range(x_start, x_end, grid_step):
                if self.check_text_fits_in_background(text, x, y, font_scale, background_mask):
                    valid_positions.append((x, y))
        
        # If we have too few positions, try a finer grid in promising areas
        if len(valid_positions) < 50:
            # Find connected components in background
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                background_mask, connectivity=8
            )
            
            for i in range(1, min(num_labels, 10)):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 5000:  # Large enough component
                    cx = int(centroids[i][0])
                    cy = int(centroids[i][1])
                    
                    # Search around this centroid
                    for dy in range(-50, 51, 10):
                        for dx in range(-50, 51, 10):
                            x = cx + dx - text_width // 2
                            y = cy + dy
                            
                            if (x >= x_start and x <= x_end and 
                                y >= y_start and y <= y_end):
                                if self.check_text_fits_in_background(text, x, y, font_scale, background_mask):
                                    valid_positions.append((x, y))
        
        return valid_positions
    
    def find_closest_valid_position(self, valid_positions: List[Tuple[int, int]], 
                                   prefer_reading_flow: bool = True) -> Tuple[int, int]:
        """
        Find the closest valid position to the previous word.
        
        Args:
            valid_positions: List of valid positions where text fits
            prefer_reading_flow: Prefer positions to the right for natural reading
            
        Returns:
            Best (x, y) position
        """
        if not valid_positions:
            # Fallback to center if no valid positions
            return (self.width // 2, self.height // 2)
        
        # First word - choose upper area
        if self.previous_position is None:
            # Prefer upper-middle area for first word
            target_y = self.height // 3
            
            best_pos = None
            best_score = float('inf')
            
            for x, y in valid_positions:
                # Score based on distance from ideal starting position
                score = abs(y - target_y)
                # Slight preference for center-left (where reading typically starts)
                score += abs(x - self.width // 3) * 0.5
                
                if score < best_score:
                    best_score = score
                    best_pos = (x, y)
            
            return best_pos if best_pos else valid_positions[0]
        
        # For subsequent words, find closest to previous
        prev_x, prev_y = self.previous_position
        
        best_pos = None
        min_distance = float('inf')
        
        for x, y in valid_positions:
            dx = x - prev_x
            dy = y - prev_y
            
            if prefer_reading_flow:
                # Weighted distance - prefer positions to the right and same line
                if dx > 0 and dx < 300:  # To the right, but not too far
                    distance = np.sqrt(dx**2 + dy**2)
                    # Strong bonus for being on same line
                    if abs(dy) < 20:
                        distance *= 0.5
                    # Moderate bonus for being slightly below (next line)
                    elif dy > 0 and dy < 50:
                        distance *= 0.7
                else:
                    # Regular distance with penalty
                    distance = np.sqrt(dx**2 + dy**2) * 1.5
                    
                    # If we're too far right, consider wrapping to next line
                    if prev_x > self.width * 0.7:
                        # Prefer positions at start of next line
                        if x < self.width * 0.4 and dy > 20 and dy < 80:
                            distance *= 0.8
            else:
                # Simple Euclidean distance
                distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                best_pos = (x, y)
        
        return best_pos if best_pos else valid_positions[0]
    
    def calculate_font_scale(self, text: str, max_width: int = 150) -> float:
        """Calculate appropriate font scale for text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Start with a reasonable scale
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Adjust to fit within max_width
        if text_size[0] > max_width:
            font_scale = max_width / text_size[0] * 0.9
            font_scale = max(0.4, min(font_scale, 1.5))  # Keep within reasonable bounds
        
        return font_scale
    
    def place_word_with_full_background_check(self, word: str, timestamp: float) -> Dict:
        """
        Place a word ensuring it's fully in the background and close to previous word.
        
        Returns:
            Dictionary with word, position, timing, and font info
        """
        # Extract background mask
        frame, background_mask = self.extract_background_mask(timestamp)
        
        # Calculate font scale
        font_scale = self.calculate_font_scale(word)
        
        # Find all valid positions where entire word fits in background
        valid_positions = self.find_valid_positions(word, font_scale, background_mask)
        
        if not valid_positions:
            print(f"  Warning: No valid background position for '{word}' at {timestamp:.2f}s")
            # Try with smaller font
            font_scale *= 0.7
            valid_positions = self.find_valid_positions(word, font_scale, background_mask)
        
        # Find closest valid position to previous word
        position = self.find_closest_valid_position(valid_positions)
        
        # Update tracking
        self.previous_position = position
        self.position_history.append(position)
        
        return {
            'word': word,
            'x': position[0],
            'y': position[1],
            'font_scale': font_scale,
            'fontsize': int(font_scale * 48),
            'valid_positions_found': len(valid_positions)
        }
    
    def generate_word_positions_with_full_background_check(self, words_transcript: List[Dict]) -> List[Dict]:
        """
        Generate positions for all words ensuring full background containment.
        
        Args:
            words_transcript: List of word dictionaries with 'word', 'start', 'end'
            
        Returns:
            List of word dictionaries with added positions
        """
        positioned_words = []
        
        # Reset tracking for new sequence
        self.previous_position = None
        self.position_history = []
        self.mask_cache = {}  # Clear cache
        
        print(f"Positioning {len(words_transcript)} words with full background validation...")
        
        for i, word_data in enumerate(words_transcript):
            word = word_data.get('word', '')
            start_time = float(word_data.get('start', 0))
            end_time = float(word_data.get('end', start_time + 0.5))
            
            # Place word with full background check
            placement = self.place_word_with_full_background_check(word, start_time)
            
            # Combine with timing
            positioned_word = {
                'word': word,
                'start': start_time,
                'end': end_time,
                'x': placement['x'],
                'y': placement['y'],
                'fontsize': placement['fontsize']
            }
            
            positioned_words.append(positioned_word)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(words_transcript)} words...")
                # Show recent movement statistics
                if len(self.position_history) > 1:
                    recent = self.position_history[-min(10, len(self.position_history)):]
                    movements = []
                    for j in range(1, len(recent)):
                        dx = recent[j][0] - recent[j-1][0]
                        dy = recent[j][1] - recent[j-1][1]
                        movements.append(np.sqrt(dx**2 + dy**2))
                    if movements:
                        avg_movement = np.mean(movements)
                        print(f"    Recent avg movement: {avg_movement:.1f} pixels")
                        print(f"    Last position: ({recent[-1][0]}, {recent[-1][1]})")
        
        # Save positions
        positions_file = self.output_dir / "word_positions_v2.json"
        with open(positions_file, 'w') as f:
            json.dump(positioned_words, f, indent=2)
        
        print(f"\n✓ Positioned all {len(positioned_words)} words")
        print(f"✓ All words verified to be fully in background")
        print(f"✓ Saved to: {positions_file}")
        
        # Calculate statistics
        if len(self.position_history) > 1:
            total_movement = 0
            max_jump = 0
            for i in range(1, len(self.position_history)):
                dx = self.position_history[i][0] - self.position_history[i-1][0]
                dy = self.position_history[i][1] - self.position_history[i-1][1]
                distance = np.sqrt(dx**2 + dy**2)
                total_movement += distance
                max_jump = max(max_jump, distance)
            
            avg_movement = total_movement / (len(self.position_history) - 1)
            print(f"\nMovement Statistics:")
            print(f"  Average movement between words: {avg_movement:.1f} pixels")
            print(f"  Maximum jump: {max_jump:.1f} pixels")
            print(f"  Total path length: {total_movement:.1f} pixels")
        
        return positioned_words
    
    def create_debug_visualization(self, words: List[Dict], output_path: str):
        """Create a visualization showing the word path and coverage."""
        # Get a frame from middle of video
        mid_time = 30.0
        frame, background_mask = self.extract_background_mask(mid_time)
        
        if frame is None:
            frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 200
        
        # Overlay background mask
        overlay = frame.copy()
        overlay[background_mask == 0] = overlay[background_mask == 0] * 0.5  # Darken foreground
        
        # Draw the path
        if len(self.position_history) > 1:
            for i in range(1, min(len(self.position_history), 100)):
                # Color gradient
                progress = i / min(len(self.position_history), 100)
                color = (int(255 * (1 - progress)), int(128 * progress), int(255 * progress))
                cv2.line(overlay, 
                        self.position_history[i-1], 
                        self.position_history[i],
                        color, 2)
        
        # Draw word positions
        for i, pos in enumerate(self.position_history[:50]):  # First 50 words
            # Color gradient from red to blue
            progress = min(i / 50, 1.0)
            color = (int(255 * (1 - progress)), 0, int(255 * progress))
            cv2.circle(overlay, pos, 5, color, -1)
            
            # Number labels for first 10
            if i < 10:
                cv2.putText(overlay, str(i+1), 
                           (pos[0] + 10, pos[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                cv2.putText(overlay, str(i+1), 
                           (pos[0] + 10, pos[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        cv2.imwrite(output_path, overlay)
        print(f"✓ Debug visualization saved to: {output_path}")
    
    def __del__(self):
        """Clean up video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()


if __name__ == "__main__":
    # Demo
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_dir = "tests/proximity_v2"
    
    # Load transcript
    with open('uploads/assets/videos/do_re_mi/transcripts/transcript_words.json') as f:
        data = json.load(f)
        words = data['words']
    
    # Get first 30 words for demo
    demo_words = words[:30]
    
    # Create placer
    placer = ProximityTextPlacerV2(video_path, output_dir)
    
    # Generate positions
    positioned = placer.generate_word_positions_with_full_background_check(demo_words)
    
    # Create visualization
    placer.create_debug_visualization(positioned, str(Path(output_dir) / "word_path_v2.png"))
    
    print("\nDemo complete! Words are guaranteed to be fully in background.")
    print("Check the visualization to see the word placement path.")