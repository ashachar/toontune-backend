#!/usr/bin/env python3
"""
Proximity-Based Text Placement System
======================================
Places words close to each other to minimize eye movement while ensuring
they remain in valid background areas.
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


class ProximityTextPlacer:
    """Places text with proximity awareness to minimize eye movement."""
    
    def __init__(self, video_path: str, output_dir: str = None):
        """
        Initialize the proximity-aware text placer.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save debug outputs
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir) if output_dir else self.video_path.parent / "proximity_backgrounds"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Video capture
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Track previous word position for proximity
        self.previous_position = None
        self.position_history = []
        
        print(f"Initialized ProximityTextPlacer for {self.video_path.name}")
        print(f"  Resolution: {self.width}x{self.height}, FPS: {self.fps}")
    
    def extract_background_candidates(self, timestamp: float) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        Extract background mask and find multiple candidate positions.
        
        Returns:
            Tuple of (frame, background_mask, list of candidate positions)
        """
        # Seek to the frame
        frame_number = int(timestamp * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame at {timestamp}s")
            return None, None, [(self.width // 2, self.height // 2)]
        
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
        
        # Find candidate positions
        candidates = self._find_candidate_positions(background_mask)
        
        return frame, background_mask, candidates
    
    def _find_candidate_positions(self, background_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find multiple candidate positions in the background.
        Uses a grid-based approach to get diverse positions.
        """
        candidates = []
        
        # Grid-based sampling for comprehensive coverage
        grid_sizes = [(8, 6), (10, 8), (12, 10)]  # Different granularities
        
        for grid_w, grid_h in grid_sizes:
            cell_w = self.width // grid_w
            cell_h = self.height // grid_h
            
            for i in range(grid_w):
                for j in range(grid_h):
                    # Center of each grid cell
                    x = i * cell_w + cell_w // 2
                    y = j * cell_h + cell_h // 2
                    
                    # Check if this position is in background
                    # Sample a small region around the point
                    x_safe = max(10, min(x, self.width - 10))
                    y_safe = max(10, min(y, self.height - 10))
                    
                    # Check a 20x20 region around the point
                    region = background_mask[
                        max(0, y_safe-10):min(self.height, y_safe+10),
                        max(0, x_safe-10):min(self.width, x_safe+10)
                    ]
                    
                    # If majority of region is background, it's a valid candidate
                    if region.size > 0 and np.mean(region) > 128:
                        candidates.append((x_safe, y_safe))
        
        # Also add candidates from connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            background_mask, connectivity=8
        )
        
        for i in range(1, min(num_labels, 20)):  # Limit to 20 components
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 2000:  # Minimum area threshold
                # Add center of component
                cx = int(centroids[i][0])
                cy = int(centroids[i][1])
                candidates.append((cx, cy))
                
                # Add points within the component
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Sample points within this component
                for dx in [0.25, 0.5, 0.75]:
                    for dy in [0.25, 0.5, 0.75]:
                        px = int(x + w * dx)
                        py = int(y + h * dy)
                        if background_mask[py, px] == 255:
                            candidates.append((px, py))
        
        # Remove duplicates (positions too close to each other)
        unique_candidates = []
        min_distance = 30  # Minimum distance between candidates
        
        for x, y in candidates:
            too_close = False
            for ux, uy in unique_candidates:
                if np.sqrt((x - ux)**2 + (y - uy)**2) < min_distance:
                    too_close = True
                    break
            if not too_close:
                unique_candidates.append((x, y))
        
        return unique_candidates
    
    def find_optimal_position(self, candidates: List[Tuple[int, int]], 
                             text: str, font_scale: float = 1.0) -> Tuple[int, int]:
        """
        Find the optimal position from candidates based on proximity to previous word.
        
        Args:
            candidates: List of candidate positions
            text: The text to place
            font_scale: Font scale for text size calculation
            
        Returns:
            Optimal (x, y) position
        """
        if not candidates:
            return (self.width // 2, self.height // 2)
        
        # For the first word or if no previous position, choose a good starting position
        if self.previous_position is None:
            # Prefer upper-middle area for first word
            best_candidate = None
            best_score = float('inf')
            
            target_x = self.width // 2
            target_y = self.height // 3
            
            for x, y in candidates:
                distance = np.sqrt((x - target_x)**2 + (y - target_y)**2)
                if distance < best_score:
                    best_score = distance
                    best_candidate = (x, y)
            
            return best_candidate if best_candidate else candidates[0]
        
        # For subsequent words, find closest to previous position
        best_candidate = None
        min_distance = float('inf')
        
        prev_x, prev_y = self.previous_position
        
        # Calculate text size for spacing consideration
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        text_width = text_size[0]
        
        # Ideal distance is just to the right of previous word (reading flow)
        # But not too far
        ideal_x = prev_x + text_width + 20  # Some spacing
        ideal_y = prev_y  # Same line if possible
        
        for x, y in candidates:
            # Calculate distance with preference for reading flow
            dx = x - prev_x
            dy = y - prev_y
            
            # Weighted distance: prefer positions to the right and same line
            if dx > 0 and dx < 200:  # To the right, but not too far
                distance = np.sqrt(dx**2 + dy**2)
                # Bonus for being on same line
                if abs(dy) < 30:
                    distance *= 0.7
            else:
                # Regular Euclidean distance with penalty
                distance = np.sqrt(dx**2 + dy**2) * 1.5
            
            if distance < min_distance:
                min_distance = distance
                best_candidate = (x, y)
        
        # If best candidate is too far (screen jump), find closer alternative
        if min_distance > 300:
            # Find absolutely closest position
            for x, y in candidates:
                distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                if distance < min_distance:
                    min_distance = distance
                    best_candidate = (x, y)
        
        return best_candidate if best_candidate else candidates[0]
    
    def calculate_font_scale(self, text: str, max_width: int = 200) -> float:
        """Calculate appropriate font scale for text."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Adjust font scale to fit within max_width
        while font_scale > 0.3:
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            if text_size[0] <= max_width:
                break
            font_scale *= 0.9
        
        return font_scale
    
    def place_word_with_proximity(self, word: str, timestamp: float) -> Dict:
        """
        Place a single word considering proximity to previous word.
        
        Returns:
            Dictionary with word, position, timing, and font info
        """
        # Extract background and get candidates
        frame, mask, candidates = self.extract_background_candidates(timestamp)
        
        if frame is None:
            # Fallback position
            position = self.previous_position if self.previous_position else (self.width // 2, self.height // 2)
        else:
            # Calculate font scale
            font_scale = self.calculate_font_scale(word)
            
            # Find optimal position
            position = self.find_optimal_position(candidates, word, font_scale)
        
        # Update previous position
        self.previous_position = position
        self.position_history.append(position)
        
        return {
            'word': word,
            'x': position[0],
            'y': position[1],
            'font_scale': font_scale if frame is not None else 1.0,
            'fontsize': int(font_scale * 48) if frame is not None else 48
        }
    
    def generate_word_positions_with_proximity(self, words_transcript: List[Dict]) -> List[Dict]:
        """
        Generate positions for all words with proximity awareness.
        
        Args:
            words_transcript: List of word dictionaries with 'word', 'start', 'end'
            
        Returns:
            List of word dictionaries with added positions
        """
        positioned_words = []
        
        # Reset tracking for new sequence
        self.previous_position = None
        self.position_history = []
        
        print(f"Positioning {len(words_transcript)} words with proximity awareness...")
        
        for i, word_data in enumerate(words_transcript):
            word = word_data.get('word', '')
            start_time = float(word_data.get('start', 0))
            end_time = float(word_data.get('end', start_time + 0.5))
            
            # Place word with proximity consideration
            placement = self.place_word_with_proximity(word, start_time)
            
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
                if i > 0:
                    # Show movement distance for last 10 words
                    recent_positions = self.position_history[-10:]
                    if len(recent_positions) > 1:
                        total_movement = 0
                        for j in range(1, len(recent_positions)):
                            dx = recent_positions[j][0] - recent_positions[j-1][0]
                            dy = recent_positions[j][1] - recent_positions[j-1][1]
                            total_movement += np.sqrt(dx**2 + dy**2)
                        avg_movement = total_movement / (len(recent_positions) - 1)
                        print(f"    Average movement: {avg_movement:.1f} pixels")
        
        # Save positions
        positions_file = self.output_dir / "word_positions_proximity.json"
        with open(positions_file, 'w') as f:
            json.dump(positioned_words, f, indent=2)
        
        print(f"\n✓ Positioned all {len(positioned_words)} words")
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
        """Create a visualization showing the word path."""
        # Create a blank canvas
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw the path
        if len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                cv2.line(canvas, 
                        self.position_history[i-1], 
                        self.position_history[i],
                        (0, 255, 0), 2)
        
        # Draw word positions
        for i, pos in enumerate(self.position_history[:50]):  # First 50 words
            # Color gradient from red to blue
            color = (int(255 * (1 - i/50)), 0, int(255 * i/50))
            cv2.circle(canvas, pos, 5, color, -1)
            
            if i < 10:  # Label first 10
                cv2.putText(canvas, str(i+1), 
                           (pos[0] + 10, pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(output_path, canvas)
        print(f"✓ Debug visualization saved to: {output_path}")
    
    def __del__(self):
        """Clean up video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()


def demo_proximity_placement():
    """Demo the proximity-based text placement."""
    video_path = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_dir = "tests/proximity_placement"
    
    # Load transcript
    with open('uploads/assets/videos/do_re_mi/transcripts/transcript_words.json') as f:
        data = json.load(f)
        words = data['words']
    
    # Get first 20 words for demo
    demo_words = words[:20]
    
    # Create placer
    placer = ProximityTextPlacer(video_path, output_dir)
    
    # Generate positions
    positioned = placer.generate_word_positions_with_proximity(demo_words)
    
    # Create visualization
    placer.create_debug_visualization(positioned, str(Path(output_dir) / "word_path.png"))
    
    return positioned


if __name__ == "__main__":
    demo_proximity_placement()