#!/usr/bin/env python3
"""
Stripe-based layout manager for subsentences.
Divides frame into horizontal stripes and intelligently places text
with optimal visibility testing and face avoidance.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import cv2
from .face_aware_placement import FaceDetector, FaceRegion


@dataclass
class StripePosition:
    """Represents a position within a stripe."""
    x: int
    y: int
    stripe_index: int
    visibility_score: float
    is_behind: bool
    
@dataclass
class TextPlacement:
    """Final placement decision for a subsentence."""
    phrase: str
    position: Tuple[int, int]  # (x, y)
    stripe_index: int
    is_behind: bool
    visibility_score: float
    font_size: int
    
    
class StripeLayoutManager:
    """
    Manages stripe-based layout for subsentences with visibility testing and face avoidance.
    """
    
    def __init__(self, 
                 frame_width: int = 1280, 
                 frame_height: int = 720,
                 visibility_threshold: float = 0.15,
                 sample_points: int = 5):
        """
        Initialize the stripe layout manager.
        
        Args:
            frame_width: Width of video frame
            frame_height: Height of video frame
            visibility_threshold: Minimum visibility (15% default) for background placement
            sample_points: Number of x-positions to sample per stripe
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.visibility_threshold = visibility_threshold
        self.sample_points = sample_points
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
    def calculate_stripes(self, num_subsentences: int) -> List[Tuple[int, int]]:
        """
        Calculate stripe boundaries for given number of subsentences.
        
        Args:
            num_subsentences: Number of subsentences to place
            
        Returns:
            List of (top_y, bottom_y) tuples for each stripe
        """
        if num_subsentences == 0:
            return []
            
        # Limit to maximum 3 stripes for readability
        # This prevents text from being too cramped when there are many phrases
        max_stripes = 3
        num_stripes = min(num_subsentences, max_stripes)
            
        # Add padding at top and bottom (10% each)
        padding = int(self.frame_height * 0.1)
        usable_height = self.frame_height - (2 * padding)
        
        stripe_height = usable_height // num_stripes
        
        stripes = []
        for i in range(num_stripes):
            top_y = padding + (i * stripe_height)
            bottom_y = top_y + stripe_height
            stripes.append((top_y, bottom_y))
            
        return stripes
    
    def get_stripe_center(self, stripe: Tuple[int, int]) -> int:
        """Get the vertical center of a stripe."""
        return (stripe[0] + stripe[1]) // 2
    
    def sample_x_positions(self, text_width: int) -> List[int]:
        """
        Sample x-positions across the frame width.
        
        Args:
            text_width: Width of the text to place
            
        Returns:
            List of x-positions to test
        """
        # Ensure text fits within frame
        margin = text_width // 2 + 20  # Add small margin
        min_x = margin
        max_x = self.frame_width - margin
        
        if max_x <= min_x:
            # Text too wide, just center it
            return [self.frame_width // 2 - text_width // 2]
        
        if self.sample_points == 1:
            # Just use center
            return [self.frame_width // 2 - text_width // 2]
        
        # Sample evenly across available width
        step = (max_x - min_x) // (self.sample_points - 1)
        positions = []
        for i in range(self.sample_points):
            x = min_x + (i * step) - text_width // 2
            positions.append(x)
            
        return positions
    
    def calculate_text_visibility(self,
                                 text_bbox: Tuple[int, int, int, int],
                                 foreground_masks: List[np.ndarray]) -> float:
        """
        Calculate visibility of text across multiple frames.
        
        Args:
            text_bbox: (x, y, width, height) of text
            foreground_masks: List of foreground masks for scene frames
            
        Returns:
            Visibility score (0.0 to 1.0)
        """
        if not foreground_masks:
            return 1.0  # No masks = fully visible
        
        x, y, w, h = text_bbox
        
        # Ensure bbox is within frame bounds
        x = max(0, x)
        y = max(0, y)
        x2 = min(self.frame_width, x + w)
        y2 = min(self.frame_height, y + h)
        
        if x2 <= x or y2 <= y:
            return 0.0  # Text completely out of bounds
        
        total_pixels = (x2 - x) * (y2 - y)
        
        # Calculate visibility for each frame
        visibility_scores = []
        
        for mask in foreground_masks:
            # Extract region where text would be
            text_region = mask[y:y2, x:x2]
            
            # Count non-foreground pixels (where text would be visible)
            # Assuming mask has 255 for foreground, 0 for background
            visible_pixels = np.sum(text_region < 128)  # Threshold at 128
            
            visibility = visible_pixels / total_pixels if total_pixels > 0 else 0
            visibility_scores.append(visibility)
        
        # Return minimum visibility across all frames (worst case)
        return min(visibility_scores) if visibility_scores else 1.0
    
    def find_optimal_position(self,
                             phrase: str,
                             stripe_index: int,
                             stripes: List[Tuple[int, int]],
                             foreground_masks: List[np.ndarray],
                             font_size: int = 48,
                             sample_frames: Optional[List[np.ndarray]] = None) -> StripePosition:
        """
        Find optimal position for a phrase within its stripe, avoiding faces.
        
        Args:
            phrase: Text to place
            stripe_index: Which stripe to place in
            stripes: All stripe boundaries
            foreground_masks: Foreground masks for visibility testing
            font_size: Font size for text
            sample_frames: Original frames for face detection
            
        Returns:
            Optimal StripePosition with visibility info
        """
        if stripe_index >= len(stripes):
            stripe_index = len(stripes) - 1
        
        stripe = stripes[stripe_index]
        y = self.get_stripe_center(stripe)
        
        # Estimate text dimensions
        # Rough approximation: each character is about 0.6 * font_size wide
        text_width = int(len(phrase) * font_size * 0.6)
        text_height = font_size
        
        # Detect faces if sample frames provided
        faces_by_frame = []
        if sample_frames:
            for i, frame in enumerate(sample_frames):
                faces = self.face_detector.detect_faces(frame, cache_key=i)
                faces_by_frame.append(faces)
        
        # Get face-aware x positions
        x_positions = []
        if faces_by_frame:
            # Find consistent safe positions across all frames
            position_scores = {}  # Track best score for each position
            for faces in faces_by_frame:
                if faces:  # If faces detected in this frame
                    safe_pos = self.face_detector.get_safe_x_positions(
                        faces, self.frame_width, text_width
                    )
                    # Keep track of the best score for each position
                    for pos, score in safe_pos:
                        if pos not in position_scores or score > position_scores[pos]:
                            position_scores[pos] = score
            
            if position_scores:
                # Sort by score (highest first) and take top positions
                sorted_positions = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)
                x_positions = [pos for pos, _ in sorted_positions[:self.sample_points]]
                print(f"     DEBUG: Using face-aware positions: {x_positions[:3]} (scores: {[score for _, score in sorted_positions[:3]]})")
        
        # Fallback to regular sampling if no faces or no safe positions
        if not x_positions:
            print(f"     DEBUG: No face-aware positions, using regular sampling")
            x_positions = self.sample_x_positions(text_width)
        
        best_position = None
        best_visibility = 0.0
        
        for x in x_positions:
            # Calculate visibility at this position
            bbox = (x, y - text_height // 2, text_width, text_height)
            visibility = self.calculate_text_visibility(bbox, foreground_masks)
            
            if visibility > best_visibility:
                best_visibility = visibility
                best_position = StripePosition(
                    x=x,
                    y=y,
                    stripe_index=stripe_index,
                    visibility_score=visibility,
                    is_behind=visibility < self.visibility_threshold  # Render behind only when visibility is BAD
                )
        
        # If no good position found, center it
        if best_position is None:
            best_position = StripePosition(
                x=self.frame_width // 2 - text_width // 2,
                y=y,
                stripe_index=stripe_index,
                visibility_score=0.0,
                is_behind=False  # Put in front if visibility is poor
            )
        
        return best_position
    
    def find_optimal_position_avoiding_overlaps(self,
                             phrase: str,
                             stripe_index: int,
                             stripes: List[Tuple[int, int]],
                             foreground_masks: List[np.ndarray],
                             font_size: int,
                             sample_frames: Optional[List[np.ndarray]],
                             occupied_regions: List[Tuple[int, int]]) -> StripePosition:
        """
        Find optimal position for a phrase, avoiding both faces and occupied regions.
        
        Args:
            phrase: Text to place
            stripe_index: Which stripe to place in
            stripes: All stripe boundaries
            foreground_masks: Foreground masks for visibility testing
            font_size: Font size for text
            sample_frames: Original frames for face detection
            occupied_regions: List of (start_x, end_x) tuples for already placed phrases
            
        Returns:
            Optimal StripePosition with visibility info
        """
        if stripe_index >= len(stripes):
            stripe_index = len(stripes) - 1
        
        stripe = stripes[stripe_index]
        y = self.get_stripe_center(stripe)
        
        # Estimate text dimensions
        text_width = int(len(phrase) * font_size * 0.6)
        text_height = font_size
        
        # Detect faces if sample frames provided
        faces_by_frame = []
        if sample_frames:
            for i, frame in enumerate(sample_frames):
                faces = self.face_detector.detect_faces(frame, cache_key=i)
                faces_by_frame.append(faces)
        
        # Get face-aware x positions
        x_positions = []
        if faces_by_frame:
            # Find consistent safe positions across all frames
            position_scores = {}  # Track best score for each position
            for faces in faces_by_frame:
                if faces:  # If faces detected in this frame
                    safe_pos = self.face_detector.get_safe_x_positions(
                        faces, self.frame_width, text_width
                    )
                    # Keep track of the best score for each position
                    for pos, score in safe_pos:
                        if pos not in position_scores or score > position_scores[pos]:
                            position_scores[pos] = score
            
            if position_scores:
                # Sort by score (highest first) and take top positions
                sorted_positions = sorted(position_scores.items(), key=lambda x: x[1], reverse=True)
                x_positions = [pos for pos, _ in sorted_positions[:self.sample_points]]
                print(f"     DEBUG: Using face-aware positions: {x_positions[:3]} (scores: {[score for _, score in sorted_positions[:3]]})")
        
        # Fallback to regular sampling if no faces or no safe positions
        if not x_positions:
            print(f"     DEBUG: No face-aware positions, using regular sampling")
            x_positions = self.sample_x_positions(text_width)
        
        # Filter out positions that would overlap with occupied regions
        valid_positions = []
        for x in x_positions:
            # Check if this position overlaps with any occupied region
            overlaps = False
            for start, end in occupied_regions:
                if not (x + text_width < start or x > end):
                    overlaps = True
                    break
            
            if not overlaps:
                valid_positions.append(x)
        
        # If all positions overlap, find gaps between occupied regions
        if not valid_positions:
            print(f"     DEBUG: All positions overlap, finding gaps between occupied regions")
            valid_positions = self.find_gaps_between_regions(occupied_regions, text_width)
        
        best_position = None
        best_visibility = 0.0
        
        for x in valid_positions:
            # Calculate visibility at this position
            bbox = (x, y - text_height // 2, text_width, text_height)
            visibility = self.calculate_text_visibility(bbox, foreground_masks)
            
            if visibility > best_visibility:
                best_visibility = visibility
                best_position = StripePosition(
                    x=x,
                    y=y,
                    stripe_index=stripe_index,
                    visibility_score=visibility,
                    is_behind=visibility < self.visibility_threshold  # Render behind only when visibility is BAD
                )
        
        # If no good position found, place after the last occupied region
        if best_position is None:
            if occupied_regions:
                # Place after the rightmost occupied region
                rightmost_end = max(end for _, end in occupied_regions)
                x = min(rightmost_end + 30, self.frame_width - text_width - 10)
            else:
                # Center it if no occupied regions
                x = self.frame_width // 2 - text_width // 2
            
            best_position = StripePosition(
                x=x,
                y=y,
                stripe_index=stripe_index,
                visibility_score=0.0,
                is_behind=False  # Put in front if visibility is poor
            )
        
        return best_position
    
    def find_gaps_between_regions(self, occupied_regions: List[Tuple[int, int]], 
                                  text_width: int) -> List[int]:
        """Find valid x positions in gaps between occupied regions."""
        if not occupied_regions:
            return [100, 300, 500]  # Default positions
        
        # Sort regions by start position
        sorted_regions = sorted(occupied_regions, key=lambda r: r[0])
        valid_positions = []
        
        # Check gap before first region
        if sorted_regions[0][0] > text_width + 10:
            valid_positions.append(10)
        
        # Check gaps between regions
        for i in range(len(sorted_regions) - 1):
            gap_start = sorted_regions[i][1]
            gap_end = sorted_regions[i+1][0]
            if gap_end - gap_start > text_width + 20:  # Enough space with padding
                valid_positions.append(gap_start + 10)
        
        # Check gap after last region
        last_end = sorted_regions[-1][1]
        if self.frame_width - last_end > text_width + 10:
            valid_positions.append(last_end + 10)
        
        return valid_positions if valid_positions else [100]  # Fallback position
    
    def layout_scene_phrases(self,
                            phrases: List[Dict],
                            foreground_masks: List[np.ndarray],
                            sample_frames: Optional[List[np.ndarray]] = None) -> List[TextPlacement]:
        """
        Layout all phrases in a scene with optimal positioning, avoiding faces.
        
        Args:
            phrases: List of phrase dictionaries with 'phrase', 'importance', etc.
            foreground_masks: Foreground masks for the scene duration
            sample_frames: Original frames for face detection
            
        Returns:
            List of TextPlacement objects with positioning decisions
        """
        placements = []
        
        # Calculate stripes based on number of phrases
        num_phrases = len(phrases)
        stripes = self.calculate_stripes(num_phrases)
        
        # Sort phrases by layout_priority if available
        sorted_phrases = sorted(phrases, 
                               key=lambda p: p.get('layout_priority', 999))
        
        # Track occupied regions for each stripe to prevent overlaps
        occupied_regions = {i: [] for i in range(len(stripes))}
        
        for i, phrase_data in enumerate(sorted_phrases):
            phrase_text = phrase_data.get('phrase', '')
            importance = phrase_data.get('importance', 0.5)
            
            # Adjust font size based on importance (reduced scaling for better layout)
            base_font_size = 48
            if importance > 0.9:
                font_size = int(base_font_size * 1.3)  # Mega title (was 1.8)
            elif importance > 0.7:
                font_size = int(base_font_size * 1.2)  # Critical (was 1.4)
            elif importance > 0.5:
                font_size = int(base_font_size * 1.1)  # Important (was 1.2)
            else:
                font_size = base_font_size  # Normal
            
            # Map phrase index to stripe index (cycling through stripes if more phrases than stripes)
            stripe_index = i % len(stripes) if stripes else 0
            
            # Find optimal position in the stripe, avoiding occupied regions
            position = self.find_optimal_position_avoiding_overlaps(
                phrase_text,
                stripe_index,  # Use modulo stripe index
                stripes,
                foreground_masks,
                font_size,
                sample_frames,
                occupied_regions[stripe_index]  # Pass occupied regions for this stripe
            )
            
            # Mark this region as occupied (with padding for visual separation)
            text_width = int(len(phrase_text) * font_size * 0.6)
            padding = 30  # Add 30px padding between phrases
            occupied_regions[stripe_index].append((position.x - padding, position.x + text_width + padding))
            
            placement = TextPlacement(
                phrase=phrase_text,
                position=(position.x, position.y),
                stripe_index=position.stripe_index,
                is_behind=position.is_behind,
                visibility_score=position.visibility_score,
                font_size=font_size
            )
            
            placements.append(placement)
            
        return placements
    
    def visualize_layout(self, 
                        placements: List[TextPlacement],
                        background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize the layout decisions on a frame.
        
        Args:
            placements: List of text placements
            background: Optional background image
            
        Returns:
            Visualization image
        """
        if background is None:
            # Create blank frame
            frame = np.ones((self.frame_height, self.frame_width, 3), 
                           dtype=np.uint8) * 255
        else:
            frame = background.copy()
        
        # Draw stripe boundaries
        num_stripes = max([p.stripe_index for p in placements], default=0) + 1
        stripes = self.calculate_stripes(num_stripes)
        
        for stripe in stripes:
            cv2.line(frame, (0, stripe[0]), (self.frame_width, stripe[0]),
                    (200, 200, 200), 1)
            cv2.line(frame, (0, stripe[1]), (self.frame_width, stripe[1]),
                    (200, 200, 200), 1)
        
        # Draw text placements
        for placement in placements:
            x, y = placement.position
            
            # Color based on is_behind
            if placement.is_behind:
                color = (0, 255, 0)  # Green for behind
                label = "[B]"
            else:
                color = (0, 0, 255)  # Red for front
                label = "[F]"
            
            # Draw text
            cv2.putText(frame, placement.phrase, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       placement.font_size / 48.0,  # Scale font
                       color, 2)
            
            # Draw visibility score
            vis_text = f"{label} {placement.visibility_score:.2f}"
            cv2.putText(frame, vis_text, (x, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       color, 1)
        
        return frame


def test_stripe_layout():
    """Test the stripe layout system."""
    
    # Create manager
    manager = StripeLayoutManager()
    
    # Test phrases for a scene
    test_phrases = [
        {"phrase": "AI created new math", "importance": 0.9, "layout_priority": 1},
        {"phrase": "Would you be surprised", "importance": 0.5, "layout_priority": 2},
        {"phrase": "if it invented operators", "importance": 0.7, "layout_priority": 3}
    ]
    
    # Create mock foreground masks (empty for testing)
    mock_masks = [np.zeros((720, 1280), dtype=np.uint8) for _ in range(30)]
    
    # Add some foreground regions to test occlusion
    for mask in mock_masks:
        # Add a foreground object in the middle
        cv2.rectangle(mask, (500, 300), (780, 500), 255, -1)
    
    # Layout the phrases
    placements = manager.layout_scene_phrases(test_phrases, mock_masks)
    
    # Print results
    print("Stripe Layout Results:")
    print("=" * 50)
    for placement in placements:
        print(f"Phrase: '{placement.phrase}'")
        print(f"  Position: {placement.position}")
        print(f"  Stripe: {placement.stripe_index}")
        print(f"  Behind: {placement.is_behind}")
        print(f"  Visibility: {placement.visibility_score:.2%}")
        print()
    
    # Visualize
    viz = manager.visualize_layout(placements)
    cv2.imwrite("outputs/stripe_layout_test.png", viz)
    print("Visualization saved to outputs/stripe_layout_test.png")


if __name__ == "__main__":
    test_stripe_layout()