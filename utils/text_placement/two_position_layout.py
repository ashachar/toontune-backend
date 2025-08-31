"""
Two-position layout manager for captions with head avoidance
Places phrases at top or bottom of screen while avoiding heads/faces
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
from PIL import Image, ImageDraw, ImageFont


@dataclass 
class PhrasePlacement:
    """Placement info for a phrase"""
    phrase: str
    position: Tuple[int, int]  # (x, y) position
    font_size: int
    is_behind: bool
    visibility_score: float
    color: Tuple[int, int, int] = (255, 255, 255)  # Default white
    

class TwoPositionLayoutManager:
    """Manages two-position layout (top and bottom) for video captions"""
    
    def __init__(self, video_width=1280, video_height=720):
        self.video_width = video_width
        self.video_height = video_height
        
        # Define default Y positions - BETTER VISIBILITY
        self.default_top_y = int(video_height * 0.20)  # 20% from top 
        self.default_bottom_y = int(video_height * 0.70)  # 70% from top (safe space for descenders!)
        
        # Alternative Y positions if heads are in the way
        # More reasonable positions that aren't at the extreme edges
        self.alt_top_y = 50  # Near top but with some padding
        self.alt_bottom_y = int(video_height * 0.72)  # 72% from top (MORE room for descenders!)
        
        # Define importance-based colors (RGB)
        self.importance_colors = {
            'critical': (255, 100, 100),  # Red for critical
            'important': (255, 200, 100),  # Orange for important  
            'question': (100, 200, 255),  # Blue for questions
            'normal': (255, 255, 255),  # White for normal
            'minor': (200, 200, 200),  # Gray for minor
        }
        
    def layout_scene_phrases(self, phrases: List[Dict], 
                            foreground_masks: List[np.ndarray] = None,
                            sample_frames: List[np.ndarray] = None) -> List[PhrasePlacement]:
        """
        Layout phrases at top or bottom positions with side-by-side arrangement
        
        Args:
            phrases: List of phrase dicts with 'phrase', 'importance', 'position', 'emphasis_type'
            foreground_masks: Optional masks for visibility calculation
            sample_frames: Optional frames for visibility testing
            
        Returns:
            List of PhrasePlacement objects with positions and styling
        """
        placements = []
        
        # Group phrases by position
        top_phrases = [p for p in phrases if p.get('position', 'bottom') == 'top']
        bottom_phrases = [p for p in phrases if p.get('position', 'bottom') == 'bottom']
        
        # Determine best Y positions based on head locations
        top_y = self._get_best_y_position('top', foreground_masks)
        bottom_y = self._get_best_y_position('bottom', foreground_masks)
        
        # Layout top phrases
        if top_phrases:
            top_placements = self._layout_horizontal_group(
                top_phrases, top_y, foreground_masks, sample_frames
            )
            placements.extend(top_placements)
            
        # Layout bottom phrases  
        if bottom_phrases:
            bottom_placements = self._layout_horizontal_group(
                bottom_phrases, bottom_y, foreground_masks, sample_frames
            )
            placements.extend(bottom_placements)
            
        return placements
        
    def _split_text_to_lines(self, text: str, font_size: int, max_width: int) -> List[str]:
        """
        Split text into multiple lines if it's too long to fit
        
        Args:
            text: The text to potentially split
            font_size: Font size to use
            max_width: Maximum width per line
            
        Returns:
            List of text lines (usually 1 or 2 lines)
        """
        # Create temp image for text measurement
        temp_img = Image.new('RGBA', (self.video_width, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            font = ImageFont.load_default()
        
        # Check if text fits as-is
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0] + 6  # Include outline
        
        if text_width <= max_width:
            return [text]  # Fits on one line
        
        # Split into words and find optimal break point
        words = text.split()
        if len(words) <= 2:
            return [text]  # Too short to split meaningfully
        
        # Try to split roughly in the middle
        mid_point = len(words) // 2
        best_split = mid_point
        best_balance = float('inf')
        
        # Find the most balanced split point
        for split_idx in range(max(1, mid_point - 2), min(len(words) - 1, mid_point + 3)):
            line1 = " ".join(words[:split_idx])
            line2 = " ".join(words[split_idx:])
            
            # Measure both lines
            bbox1 = draw.textbbox((0, 0), line1, font=font)
            bbox2 = draw.textbbox((0, 0), line2, font=font)
            width1 = bbox1[2] - bbox1[0] + 6
            width2 = bbox2[2] - bbox2[0] + 6
            
            # Check if both lines fit
            if width1 <= max_width and width2 <= max_width:
                # Calculate balance (prefer similar line lengths)
                balance = abs(width1 - width2)
                if balance < best_balance:
                    best_balance = balance
                    best_split = split_idx
        
        # Return the best split
        line1 = " ".join(words[:best_split])
        line2 = " ".join(words[best_split:])
        return [line1, line2]
    
    def _calculate_safe_font_size(self, text: str, base_font_size: int, max_width: int = None) -> int:
        """
        Calculate a safe font size that ensures text fits within screen boundaries
        
        Args:
            text: The text to render
            base_font_size: Initial font size
            max_width: Maximum allowed width (default: 85% of screen width)
            
        Returns:
            Adjusted font size that fits within boundaries
        """
        if max_width is None:
            # Use 85% of screen width as safe margin (leaving 7.5% on each side)
            max_width = int(self.video_width * 0.85)
        
        # Create temp image for text measurement (needs to be large enough for text)
        temp_img = Image.new('RGBA', (self.video_width, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        current_size = base_font_size
        
        # Try progressively smaller sizes until text fits
        while current_size >= 30:  # Minimum readable size
            try:
                font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', current_size)
            except:
                font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            
            # Account for outline effect (adds ~6 pixels)
            text_width += 6
            
            if text_width <= max_width:
                return current_size
            
            # Reduce size by 5% and try again
            current_size = int(current_size * 0.95)
        
        return 30  # Minimum size if nothing else fits
    
    def _layout_horizontal_group(self, phrases: List[Dict], y_position: int,
                                foreground_masks: List[np.ndarray] = None,
                                sample_frames: List[np.ndarray] = None) -> List[PhrasePlacement]:
        """
        Layout a group of phrases as a SINGLE CONTINUOUS LINE
        If it overlaps with heads, make it 30% larger and put it behind
        """
        placements = []
        
        # COMBINE all phrases at this position into a single continuous phrase
        if len(phrases) > 1:
            print(f"   📝 Combining {len(phrases)} phrases at {'top' if y_position < self.video_height//2 else 'bottom'} into single line:")
            for p in phrases:
                print(f"      - '{p['phrase']}'")
        
        # Detect head regions across ALL masks
        head_regions = []
        if foreground_masks and len(foreground_masks) > 0:
            head_regions = self._detect_head_regions_all_frames(foreground_masks, y_position)
            if head_regions:
                print(f"   🎯 Detected {len(head_regions)} head regions at y={y_position}")
                for head_x, head_width in head_regions:
                    print(f"      Head at x={head_x}, width={head_width}")
        
        # Calculate combined width and check for overlap
        total_text = " ".join(p['phrase'] for p in phrases)
        avg_importance = sum(p.get('importance', 0.5) for p in phrases) / len(phrases)
        base_font_size = self._get_font_size(avg_importance)
        
        # Check if text would overlap with any head region when centered
        will_overlap_head = False
        if head_regions:
            # Estimate width when centered
            estimated_width = int(len(total_text) * base_font_size * 0.6)
            center_x = self.video_width // 2
            text_start = center_x - estimated_width // 2
            text_end = text_start + estimated_width
            
            # Check overlap with any head
            for head_x, head_width in head_regions:
                head_start = head_x - head_width // 2
                head_end = head_x + head_width // 2
                if not (text_end < head_start or text_start > head_end):
                    will_overlap_head = True
                    print(f"   ⚡ Text will overlap with head - making it 50% larger and placing behind")
                    break
        
        # Create placements for each phrase with adjusted properties
        for phrase_dict in phrases:
            importance = phrase_dict.get('importance', 0.5)
            emphasis_type = phrase_dict.get('emphasis_type', 'normal')
            
            # Base font size
            font_size = self._get_font_size(importance)
            
            # If overlapping with head, make 20% larger and put behind
            is_behind = False
            if will_overlap_head:
                font_size = int(font_size * 1.2)  # 20% larger for behind text
                is_behind = True
            
            # CRITICAL: Ensure text fits within screen boundaries
            max_width = int(self.video_width * 0.85)  # 85% of width for safe margins
            safe_font_size = self._calculate_safe_font_size(
                phrase_dict['phrase'], 
                font_size,
                max_width
            )
            
            # Use the smaller of the two to ensure it fits
            original_font_size = font_size
            font_size = min(font_size, safe_font_size)
            
            # Debug output if size was adjusted
            if safe_font_size < original_font_size * 0.8:
                print(f"   ⚠️ Font size reduced from {original_font_size} to {safe_font_size} to fit screen")
            
            # Check if text still doesn't fit even at minimum size - then split it
            text_lines = [phrase_dict['phrase']]
            if font_size == 30:  # Hit minimum size
                # Try splitting the text into two lines
                text_lines = self._split_text_to_lines(phrase_dict['phrase'], font_size, max_width)
                if len(text_lines) > 1:
                    print(f"   📄 Splitting long text into {len(text_lines)} lines:")
                    for line in text_lines:
                        print(f"      - '{line}'")
            
            # Get color - use dark color for behind text so it's visible on light background
            if is_behind:
                # Use dark blue/black for behind text to be visible on white background
                color = (50, 50, 100)  # Dark blue-ish color
            else:
                color = self._get_color(emphasis_type, importance)
            
            # Create placement(s) - one for each line if split
            for i, text_line in enumerate(text_lines):
                # Adjust Y position for multiple lines
                line_y = y_position
                if len(text_lines) > 1:
                    # Stack lines vertically with spacing
                    line_spacing = int(font_size * 1.2)
                    if i == 0:
                        line_y = y_position - line_spacing // 2
                    else:
                        line_y = y_position + line_spacing // 2
                
                # Create placement with simple centered positioning
                # The word factory will handle the actual centering
                placement = PhrasePlacement(
                    phrase=text_line,
                    position=(None, line_y),  # Let word factory center it
                    font_size=font_size,
                    is_behind=is_behind,
                    visibility_score=1.0,  # Not calculating visibility anymore
                    color=color
                )
                
                placements.append(placement)
        
        return placements
    
    def _detect_head_regions_all_frames(self, foreground_masks: List[np.ndarray], y_position: int) -> List[Tuple[int, int]]:
        """
        Detect head/face regions across ALL frames to ensure text never overlaps throughout scene
        Returns list of (x_center, width) tuples for merged head regions
        """
        # Check all positions for head/face regions - no exceptions
        # Head detection should work for both top and bottom text
        
        all_head_regions = []
        
        # Process EVERY mask to find head positions in each frame
        print(f"      Analyzing {len(foreground_masks)} frames for head positions...")
        for i, mask in enumerate(foreground_masks):
            frame_heads = self._detect_heads_in_single_frame(mask, y_position)
            if frame_heads:
                all_head_regions.extend(frame_heads)
                if i % 5 == 0:  # Debug output every 5 frames
                    print(f"         Frame {i}: Found {len(frame_heads)} head regions")
        
        # Merge overlapping regions to get union of all head positions
        if not all_head_regions:
            return []
        
        # Sort by x position
        all_head_regions.sort(key=lambda h: h[0] - h[1]//2)
        
        # Merge overlapping or nearby regions
        merged_regions = []
        current_start = all_head_regions[0][0] - all_head_regions[0][1]//2
        current_end = all_head_regions[0][0] + all_head_regions[0][1]//2
        
        for head_center, head_width in all_head_regions[1:]:
            head_start = head_center - head_width // 2
            head_end = head_center + head_width // 2
            
            # If overlaps or is within 30 pixels, merge
            if head_start <= current_end + 30:
                current_end = max(current_end, head_end)
            else:
                # Save current merged region
                merged_center = (current_start + current_end) // 2
                merged_width = current_end - current_start
                merged_regions.append((merged_center, merged_width))
                # Start new region
                current_start = head_start
                current_end = head_end
        
        # Add last region
        merged_center = (current_start + current_end) // 2
        merged_width = current_end - current_start
        merged_regions.append((merged_center, merged_width))
        
        print(f"      Merged {len(all_head_regions)} detections into {len(merged_regions)} regions")
        
        return merged_regions
    
    def _detect_heads_in_single_frame(self, mask: np.ndarray, y_position: int) -> List[Tuple[int, int]]:
        """
        Detect FACE/HEAD regions only (not body/torso) in a single frame
        Returns list of (x_center, width) tuples
        """
        head_regions = []
        
        # CRITICAL: Only detect faces/heads in the TOP portion of the frame
        # Faces are typically in the top 40% of the frame (0-288 pixels for 720p)
        max_face_y = int(self.video_height * 0.4)  # Top 40% of frame
        
        # Only proceed if text Y position could overlap with face area
        if y_position > max_face_y + 100:
            # Text is too far below face area to collide
            return []
        
        # Focus on the face region (top 40% of frame)
        y_start = 0
        y_end = max_face_y
        
        # Get horizontal slice of the face region
        region_slice = mask[y_start:y_end, :]
        
        # Threshold to binary
        _, binary = cv2.threshold(region_slice.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
        
        # Find contours (connected components)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for face-like regions
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Face detection criteria:
            # - Width: 120-300 pixels (faces are smaller than torsos)
            # - Height: at least 100 pixels
            # - Position: must be in top 40% of frame
            # - Centered: typically near horizontal center
            if 120 < w < 300 and h > 100:
                # Check if it's roughly centered (faces usually are)
                if abs(x + w // 2 - self.video_width // 2) < 400:
                    # Check if this region extends low enough to collide with text
                    face_bottom = y + h
                    if face_bottom + 50 >= y_position:  # Face extends near text position
                        head_center = x + w // 2
                        # Add padding for safety
                        head_width = w + 60  # 30 pixels padding on each side
                        head_regions.append((head_center, head_width))
        
        return head_regions
    
    def _detect_head_regions(self, foreground_masks: List[np.ndarray], y_position: int) -> List[Tuple[int, int]]:
        """
        DEPRECATED: Use _detect_head_regions_all_frames instead
        This old method only averages masks and misses head movement
        """
        # Redirect to the new comprehensive method
        return self._detect_head_regions_all_frames(foreground_masks, y_position)
    
    def _find_optimal_positions(self, phrase_data: List[Dict], head_regions: List[Tuple[int, int]], 
                               y_position: int) -> List[int]:
        """
        Find optimal X positions for phrases while avoiding heads
        DEFAULT: Center the text. Only move if heads are in the way.
        """
        positions = []
        spacing = 40  # Minimum spacing between phrases
        
        # ALWAYS start by calculating centered position
        total_width = sum(p['width'] for p in phrase_data) + spacing * (len(phrase_data) - 1)
        centered_start_x = max(20, (self.video_width - total_width) // 2)  # Ensure at least 20px margin
        
        # Debug output
        phrase_texts = [p['dict']['phrase'] for p in phrase_data]
        print(f"      Positioning phrases: {phrase_texts}")
        print(f"      Total width: {total_width}, Centered start: {centered_start_x}")
        print(f"      Head regions: {head_regions}")
        
        # Check if centered position would overlap with any heads
        needs_adjustment = False
        if head_regions:
            # Check if any phrase would overlap with a head at centered position
            test_x = centered_start_x
            for data in phrase_data:
                phrase_end = test_x + data['width']
                
                # Check against all head regions
                for head_center, head_width in head_regions:
                    head_start = head_center - head_width // 2
                    head_end = head_center + head_width // 2
                    
                    # Check for overlap
                    if not (phrase_end < head_start or test_x > head_end):
                        needs_adjustment = True
                        break
                
                if needs_adjustment:
                    break
                test_x += data['width'] + spacing
        
        if not needs_adjustment:
            # Use centered position - no heads in the way!
            current_x = centered_start_x
            for data in phrase_data:
                positions.append(current_x)
                current_x += data['width'] + spacing
        else:
            # Heads detected and would overlap - find safe zones
            safe_zones = self._find_safe_zones(head_regions)
            
            # First, try to find a safe zone that can fit all phrases together
            all_phrases_width = total_width
            best_zone = None
            
            for zone_start, zone_end in safe_zones:
                zone_width = zone_end - zone_start
                if all_phrases_width + 40 <= zone_width:  # Can fit all with margins
                    # Prefer zone closest to center
                    zone_center = (zone_start + zone_end) // 2
                    if best_zone is None or abs(zone_center - self.video_width // 2) < abs(best_zone[2] - self.video_width // 2):
                        best_zone = (zone_start, zone_end, zone_center)
            
            if best_zone:
                # Place all phrases in the best zone, centered within it
                zone_start, zone_end, _ = best_zone
                zone_width = zone_end - zone_start
                start_x = zone_start + (zone_width - all_phrases_width) // 2
                current_x = start_x
                
                for data in phrase_data:
                    positions.append(current_x)
                    current_x += data['width'] + spacing
            else:
                # Can't fit all together - try to place in different safe zones
                # IMPORTANT: Maintain chronological left-to-right ordering
                # Earlier phrases MUST be placed to the left of later phrases
                used_ranges = []  # Track (start, end) of used positions
                
                for i, data in enumerate(phrase_data):
                    phrase_width = data['width']
                    
                    # Determine minimum X position based on previous phrases (chronological ordering)
                    min_x = 20  # Default left margin
                    if i > 0 and positions:
                        # This phrase must be to the right of the previous phrase
                        prev_end = used_ranges[-1][1] if used_ranges else positions[-1]
                        min_x = prev_end + spacing
                    
                    # Find the best safe zone for this phrase that respects chronological ordering
                    best_pos = None
                    best_score = float('inf')
                    
                    for zone_start, zone_end in safe_zones:
                        zone_width = zone_end - zone_start
                        # Zone must be to the right of minimum position for chronological ordering
                        effective_zone_start = max(zone_start, min_x)
                        
                        if effective_zone_start + phrase_width + 20 <= zone_end:
                            # Try positions within this zone, respecting chronological order
                            for test_x in range(effective_zone_start + 10, zone_end - phrase_width - 10, 20):
                                # Check if this position overlaps with any used ranges
                                overlaps = False
                                for used_start, used_end in used_ranges:
                                    if not (test_x + phrase_width + spacing < used_start or test_x > used_end + spacing):
                                        overlaps = True
                                        break
                                
                                if not overlaps and test_x >= min_x:
                                    # For chronological ordering, prefer leftmost valid position
                                    # (not center distance)
                                    if best_pos is None or test_x < best_pos:
                                        best_pos = test_x
                    
                    if best_pos is not None:
                        positions.append(best_pos)
                        used_ranges.append((best_pos, best_pos + phrase_width))
                    else:
                        # Fallback: place at edge with proper spacing from previous phrases
                        if len(positions) == 0:
                            positions.append(20)  # First phrase at left edge
                            used_ranges.append((20, 20 + phrase_width))
                        else:
                            # Place after the last phrase with spacing
                            last_end = used_ranges[-1][1]
                            new_pos = last_end + spacing * 2  # Extra spacing when forced
                            
                            # Check if it fits on screen
                            if new_pos + phrase_width < self.video_width - 20:
                                positions.append(new_pos)
                                used_ranges.append((new_pos, new_pos + phrase_width))
                            else:
                                # Try left side if there's space
                                if used_ranges[0][0] > phrase_width + 40:
                                    new_pos = 20
                                    positions.append(new_pos)
                                    used_ranges.append((new_pos, new_pos + phrase_width))
                                else:
                                    # Force position with overlap warning
                                    print(f"      ⚠️ WARNING: Phrase may overlap due to lack of space")
                                    positions.append(last_end + spacing)
                                    used_ranges.append((last_end + spacing, last_end + spacing + phrase_width))
        
        return positions
    
    def _find_safe_zones(self, head_regions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Find safe horizontal zones that don't overlap with heads
        Returns list of (start_x, end_x) tuples
        """
        safe_zones = []
        
        # Sort head regions by x position
        sorted_heads = sorted(head_regions, key=lambda h: h[0] - h[1]//2)
        
        # Find gaps between heads
        current_x = 20  # Start from left margin
        
        for head_center, head_width in sorted_heads:
            head_start = head_center - head_width // 2
            head_end = head_center + head_width // 2
            
            # Add safe zone before this head if there's space
            if head_start - current_x > 100:  # At least 100px gap
                safe_zones.append((current_x, head_start - 10))
            
            current_x = head_end + 10
        
        # Add final zone after last head
        if current_x < self.video_width - 100:
            safe_zones.append((current_x, self.video_width - 20))
        
        # If no good zones found, add edges as fallback
        if not safe_zones:
            safe_zones = [(20, 200), (self.video_width - 200, self.video_width - 20)]
        
        return safe_zones
    
    def _get_best_y_position(self, position_type: str, foreground_masks: List[np.ndarray] = None) -> int:
        """
        Determine the best Y position (might adjust if heads are blocking)
        """
        if position_type == 'top':
            default_y = self.default_top_y
            alt_y = self.alt_top_y
        else:
            default_y = self.default_bottom_y
            alt_y = self.alt_bottom_y
        
        # If no masks, use default
        if not foreground_masks or len(foreground_masks) == 0:
            return default_y
        
        # Check if there's significant obstruction at the default position
        avg_mask = np.mean([mask for mask in foreground_masks], axis=0)
        
        # Sample a horizontal strip at the default Y position
        y_start = max(0, default_y - 30)
        y_end = min(avg_mask.shape[0], default_y + 30)
        
        strip = avg_mask[y_start:y_end, :]
        
        # Calculate how much of the center area is obstructed
        center_start = self.video_width // 4
        center_end = 3 * self.video_width // 4
        center_strip = strip[:, center_start:center_end]
        
        # RVM mask: lower values (0-150) = foreground/person, higher values (150-255) = background
        # We want to detect foreground obstruction, so look for LOW values
        obstruction_ratio = np.sum(center_strip < 150) / center_strip.size
        
        # If more than 40% obstructed in center by foreground, use alternative position
        # Lower threshold (40%) to be more aggressive about avoiding faces
        if obstruction_ratio > 0.4:
            print(f"   ⚠️ High obstruction ({obstruction_ratio:.1%}) at {position_type} position y={default_y}")
            print(f"      Using alternative position y={alt_y}")
            return alt_y
        
        return default_y
        
    def _get_font_size(self, importance: float) -> int:
        """Get font size based on importance"""
        # MODERATE SIZES: Base 45px, up to 65px for maximum importance
        # Balanced for visibility without overflow
        min_size = 45  # Balanced size
        max_size = 65  # Reasonable maximum
        return int(min_size + (max_size - min_size) * importance)
        
    def _get_color(self, emphasis_type: str, importance: float) -> Tuple[int, int, int]:
        """Get color based on emphasis type and importance"""
        # Map emphasis types to colors
        if emphasis_type in self.importance_colors:
            return self.importance_colors[emphasis_type]
        
        # Fallback based on importance
        if importance > 0.8:
            return self.importance_colors['critical']
        elif importance > 0.6:
            return self.importance_colors['important']
        else:
            return self.importance_colors['normal']
            
    def _calculate_visibility(self, x: int, y: int, width: int, height: int,
                            foreground_masks: List[np.ndarray]) -> float:
        """Calculate visibility score for a text region"""
        if not foreground_masks:
            return 1.0
            
        # Sample visibility across all masks
        visibility_scores = []
        
        for mask in foreground_masks:
            # Ensure bounds are within frame
            y_start = max(0, y - height // 2)
            y_end = min(mask.shape[0], y + height // 2)
            x_start = max(0, x)
            x_end = min(mask.shape[1], x + width)
            
            if y_end > y_start and x_end > x_start:
                # Extract region from mask
                region = mask[y_start:y_end, x_start:x_end]
                # Calculate percentage of background pixels (< 128)
                background_pixels = np.sum(region < 128)
                total_pixels = region.size
                if total_pixels > 0:
                    visibility = background_pixels / total_pixels
                    visibility_scores.append(visibility)
                    
        # Return average visibility across frames
        return np.mean(visibility_scores) if visibility_scores else 1.0