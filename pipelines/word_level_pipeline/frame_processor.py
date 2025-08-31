"""
Frame processing logic for word-level pipeline
"""

import numpy as np
from typing import List, Tuple

from .models import WordObject
from .rendering import WordRenderer


class FrameProcessor:
    """Processes frames by rendering all active words with proper layering"""
    
    def __init__(self, video_path=None):
        """Initialize frame processor with optional video path for cached masks
        
        Args:
            video_path: Path to video file for cached mask lookup
        """
        self.renderer = WordRenderer(video_path)
        self.frame_count = 0
    
    def process_frame(self, frame: np.ndarray, time_seconds: float,
                     word_objects: List[WordObject],
                     sentence_fog_times: List[Tuple[float, float]], 
                     frame_number: int = None) -> np.ndarray:
        """Process frame by rendering all active words with proper layering"""
        result = frame.copy()
        
        # CRITICAL: Set the frame number in the renderer for proper mask synchronization
        if frame_number is not None:
            self.renderer.set_frame_number(frame_number)
        
        # Debug: Check for "Would" word timing around problematic times
        has_would = any(w.text == "Would" for w in word_objects)
        if has_would and (1.8 <= time_seconds <= 2.5):
            print(f"\n🎬 Frame at t={time_seconds:.3f}s - Processing frame with 'Would'")
            print(f"   Fog times: {sentence_fog_times}")
        
        # Determine fog progress and dissolved state for each sentence
        fog_progress_by_sentence = []
        dissolved_by_sentence = []
        for i, (fog_start, fog_end) in enumerate(sentence_fog_times):
            if fog_start <= time_seconds <= fog_end:
                progress = (time_seconds - fog_start) / (fog_end - fog_start)
                fog_progress_by_sentence.append(progress)
                dissolved_by_sentence.append(False)
                if has_would and (1.8 <= time_seconds <= 2.5):
                    print(f"   Scene {i}: FOG IN PROGRESS ({progress:.2f})")
            elif time_seconds > fog_end:
                fog_progress_by_sentence.append(1.0)
                dissolved_by_sentence.append(True)  # Sentence has fully dissolved
                if has_would and (1.8 <= time_seconds <= 2.5):
                    print(f"   Scene {i}: DISSOLVED")
            else:
                fog_progress_by_sentence.append(0.0)
                dissolved_by_sentence.append(False)
                if has_would and (1.8 <= time_seconds <= 2.5):
                    print(f"   Scene {i}: NOT YET FOGGED")
        
        # Separate words by is_behind flag for proper rendering order
        behind_words = [w for w in word_objects if w.is_behind]
        front_words = [w for w in word_objects if not w.is_behind]
        
        # Debug: One-time scene assignment summary
        if has_would and time_seconds < 0.1:  # Only print once at start
            print("\n📊 SCENE ASSIGNMENT SUMMARY:")
            for scene_idx in range(len(sentence_fog_times) + 1):
                scene_words = []
                for w in word_objects[:30]:  # Check first 30 words
                    if self._get_sentence_index(w, sentence_fog_times) == scene_idx:
                        scene_words.append(f"'{w.text}'")
                if scene_words:
                    print(f"   Scene {scene_idx}: {' '.join(scene_words[:10])}")  # Show first 10 words
        
        # GROUP BEHIND WORDS BY Y POSITION AND SCENE for phrase rendering
        # CRITICAL FIX: Group words that are CLOSE in Y position (within 50 pixels)
        # because baseline alignment means words have slightly different Y values
        behind_phrases = {}
        y_tolerance = 50  # Words within 50 pixels vertically are considered same phrase
        
        for word_obj in behind_words:
            # Find an existing group with similar Y position
            found_group = None
            for (y_pos, scene_idx) in behind_phrases.keys():
                if scene_idx == word_obj.scene_index and abs(y_pos - word_obj.y) <= y_tolerance:
                    found_group = (y_pos, scene_idx)
                    break
            
            if found_group:
                # Add to existing group
                behind_phrases[found_group].append(word_obj)
            else:
                # Create new group using this word's Y as reference
                group_key = (word_obj.y, word_obj.scene_index)
                behind_phrases[group_key] = [word_obj]
        
        # Sort each phrase by X position to ensure correct word order
        for key in behind_phrases:
            behind_phrases[key].sort(key=lambda w: w.x)
        
        # Render each behind phrase as a complete unit
        for (y_pos, scene_idx), phrase_words in behind_phrases.items():
            # Check if any word in the phrase is dissolved
            fog_progress = 0.0
            is_dissolved = False
            if 0 <= scene_idx < len(fog_progress_by_sentence):
                fog_progress = fog_progress_by_sentence[scene_idx]
                is_dissolved = dissolved_by_sentence[scene_idx]
            
            # Skip if dissolved
            if not is_dissolved:
                # Debug output for the "Would you be surprised if" phrase
                if any(w.text in ["Would", "be", "surprised"] for w in phrase_words):
                    words_text = " ".join(w.text for w in phrase_words)
                    print(f"   🎨 Rendering behind phrase at y={y_pos}: '{words_text}'")
                
                # Render the entire phrase at once
                result = self.renderer.render_phrase_behind(result, phrase_words, time_seconds)
        
        # Then render front words
        for word_obj in front_words:
            # Determine which sentence/scene this word belongs to based on start time
            sentence_index = self._get_sentence_index(word_obj, sentence_fog_times)
            
            fog_progress = 0.0
            is_dissolved = False
            if 0 <= sentence_index < len(fog_progress_by_sentence):
                fog_progress = fog_progress_by_sentence[sentence_index]
                is_dissolved = dissolved_by_sentence[sentence_index]
            
            # Debug for "Would"
            if word_obj.text == "Would" and (1.8 <= time_seconds <= 2.5):
                print(f"   Rendering 'Would': sentence_idx={sentence_index}, fog={fog_progress:.2f}, dissolved={is_dissolved}, start={word_obj.start_time:.3f}")
            
            result = self.renderer.render_word_with_masking(word_obj, result, time_seconds, 
                                                          fog_progress, is_dissolved)
        
        return result
    
    def _get_sentence_index(self, word_obj: WordObject, 
                           sentence_fog_times: List[Tuple[float, float]]) -> int:
        """Determine which sentence/scene this word belongs to
        
        SIMPLE FIX: Just use the scene_index that was set when the word was created!
        This ensures words stay with their intended scene regardless of timing.
        """
        
        # Debug for the word "Would" and surrounding words
        if word_obj.text in ["Would", "you", "be", "surprised", "if"]:
            print(f"\n🔍 DEBUG _get_sentence_index for '{word_obj.text}':")
            print(f"   Word scene_index: {word_obj.scene_index}")
            print(f"   Word start_time: {word_obj.start_time:.3f}")
            print(f"   Fog times: {sentence_fog_times}")
            print(f"   → Using scene_index: {word_obj.scene_index}")
        
        # Just return the scene index that was set when the word was created
        return word_obj.scene_index