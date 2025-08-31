"""
Word object creation and management for word-level pipeline
"""

from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

from .models import WordObject


class WordFactory:
    """Creates word objects with proper positioning and styling"""
    
    def __init__(self, font_size=55):
        self.font_size = font_size
        
        # Font for measurements
        try:
            self.font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
        except:
            self.font = ImageFont.load_default()
    
    def create_phrase_words(self, text: str, word_timings: List[Dict], 
                           placement, from_below: bool, scene_index: int = 0) -> List[WordObject]:
        """Create word objects for a phrase using stripe placement with horizontal centering"""
        words = text.split()
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Use font size from placement, increase by 50% if behind
        actual_font_size = placement.font_size
        if placement.is_behind:
            actual_font_size = int(actual_font_size * 1.5)
        
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', actual_font_size)
            if text == "surprised if":
                print(f"         DEBUG: Using font size {actual_font_size} for 'surprised if'")
        except:
            font = ImageFont.load_default()
            if text == "surprised if":
                print(f"         DEBUG: Using default font for 'surprised if'")
        
        # Calculate word measurements with minimal spacing
        word_measurements = []
        
        # Use a moderate fixed space between words (20 pixels regardless of font size)
        # This creates clear visual separation between words
        space_width = 20
        
        # Calculate individual word widths
        total_width = 0
        for word in words:
            bbox = draw.textbbox((0, 0), word, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            word_measurements.append((width, height))
            total_width += width
        
        # Add spacing between words
        if len(words) > 1:
            total_width += space_width * (len(words) - 1)
        
        # ALWAYS center the text unless it would overlap with faces
        # The layout manager's position is just an estimate - we recalculate with actual font metrics
        center_x = 640  # Screen center for 1280 width
        ideal_start_x = center_x - total_width // 2
        
        # Debug for AI operator phrase
        if "AI" in text and "operator" in text:
            print(f"       DEBUG: Centering '{text}'")
            print(f"         Total width: {total_width}, Ideal start: {ideal_start_x}")
            if hasattr(placement, 'position'):
                print(f"         Layout manager position: {placement.position[0]}")
        
        # Check if we should use the layout manager's position (only if avoiding faces)
        if hasattr(placement, 'position') and placement.position[0] is not None:
            # If the layout manager moved text significantly from center, it's avoiding faces
            layout_start_x = placement.position[0]
            # Only use layout position if it's VERY significantly different from centered position
            # (more than 200 pixels difference suggests face avoidance)
            # Increased threshold because font size estimation differences can cause ~100px discrepancy
            if abs(layout_start_x - ideal_start_x) > 200:
                start_x = layout_start_x
                if "AI" in text and "operator" in text:
                    print(f"         Using layout position (face avoidance): {start_x}")
            else:
                # Otherwise, use our properly centered position
                start_x = ideal_start_x
                if "AI" in text and "operator" in text:
                    print(f"         Using centered position: {start_x}")
        else:
            # Fallback to horizontal centering
            start_x = ideal_start_x
            if "AI" in text and "operator" in text:
                print(f"         Using fallback centered position: {start_x}")
        
        # Create word objects starting from calculated position
        current_x = start_x
        word_objects = []
        
        # Debug for "Would you be" 
        if text == "Would you be":
            print(f"         DEBUG: About to create word objects")
            print(f"         Words: {words}")
            print(f"         Word timings: {word_timings}")
            print(f"         Len words: {len(words)}, Len timings: {len(word_timings)}")
            print(f"         Word measurements: {word_measurements}")
        
        if len(words) != len(word_timings):
            print(f"         ERROR: Mismatch! {len(words)} words but {len(word_timings)} timings for '{text}'")
            return []
        
        # CRITICAL FIX: Align words by their BASELINES, not their tops
        # Find the tallest word in the phrase to establish baseline
        max_height = max(h for w, h in word_measurements)
        
        # The stripe layout manager returns Y as the CENTER of the stripe
        center_y = placement.position[1]
        
        # Calculate the baseline position for the entire phrase
        # The baseline should be at: center_y + (max_height // 2)
        # This ensures all words sit on the same baseline
        baseline_y = center_y + (max_height // 2)
        
        if "AI created new math" in text:
            print(f"         BASELINE DEBUG for '{text}':")
            print(f"           Center Y from layout: {center_y}")
            print(f"           Max word height: {max_height}")
            print(f"           Baseline Y: {baseline_y}")
        
        for i, (word, timing) in enumerate(zip(words, word_timings)):
            width, height = word_measurements[i]
            
            # Debug for "surprised if" phrase
            if text == "surprised if" and word in ["surprised", "if"]:
                print(f"         DEBUG: Placing '{word}' - width={width}, current_x={current_x}")
            
            # CRITICAL: Calculate Y position so that the BOTTOM of the word aligns with the baseline
            # The word's Y position (top) = baseline - word height
            top_y = baseline_y - height
            
            if "AI created new math" in text:
                print(f"           Word '{word}': height={height}, top_y={top_y}, bottom={top_y + height} (should equal baseline={baseline_y})")
            
            if text == "surprised if" and word in ["surprised", "if"]:
                print(f"         DEBUG: Baseline alignment for '{word}': baseline={baseline_y}, height={height}, top_y={top_y}")
            
            word_obj = WordObject(
                text=word,
                x=current_x,
                y=top_y,  # Y position where bottom aligns with baseline
                width=width,
                height=height,
                start_time=timing['start'],
                end_time=timing['end'],
                rise_duration=0.8,
                from_below=from_below,
                is_behind=placement.is_behind,  # Use visibility-based decision
                font_size=actual_font_size,  # Store actual font size (increased if behind)
                scene_index=scene_index,  # Track which scene this word belongs to
                color=placement.color if hasattr(placement, 'color') else (255, 255, 255),  # Use color from placement
            )
            
            word_objects.append(word_obj)
            current_x += width + space_width
        
        return word_objects
    
    def create_sentence_words(self, text: str, word_timings: List[Dict], 
                             center: Tuple[int, int], from_below: bool) -> List[WordObject]:
        """Create word objects for a sentence with fixed positions"""
        words = text.split()
        temp_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate total width
        total_width = 0
        word_measurements = []
        space_width = draw.textbbox((0, 0), " ", font=self.font)[2]
        
        for word in words:
            bbox = draw.textbbox((0, 0), word, font=self.font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            word_measurements.append((width, height))
            total_width += width
        
        total_width += space_width * (len(words) - 1)
        
        # Create word objects with fixed positions
        start_x = center[0] - total_width // 2
        current_x = start_x
        word_objects = []
        
        for i, (word, timing) in enumerate(zip(words, word_timings)):
            width, height = word_measurements[i]
            
            # Create word object with all parameters
            word_obj = WordObject(
                text=word,
                x=current_x,  # Fixed position
                y=center[1],  # Fixed position
                width=width,
                height=height,
                start_time=timing['start'],
                end_time=timing['end'],
                rise_duration=0.8,  # Gentle rise
                from_below=from_below,  # Same direction for entire sentence
                is_behind=False,  # Default to front for backward compatibility
                font_size=48,  # Default font size for backward compatibility
            )
            
            word_objects.append(word_obj)
            current_x += width + space_width
        
        return word_objects