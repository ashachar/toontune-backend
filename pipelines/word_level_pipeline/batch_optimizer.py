"""
Batch optimization techniques for word-level pipeline
Includes caching, pre-computation, and smart filtering
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from PIL import Image, ImageDraw, ImageFont
import cv2
from collections import defaultdict

from .word_factory import WordObject


class BatchOptimizer:
    """Optimizations for batch processing of words"""
    
    def __init__(self):
        self.word_image_cache = {}
        self.font_cache = {}
        self.active_words_by_time = {}
        
    def precompute_word_images(self, word_objects: List[WordObject]) -> Dict:
        """Pre-render all word images to avoid repeated rendering"""
        print("🎨 Pre-rendering word images...")
        
        for word in word_objects:
            cache_key = (word.text, word.font_size, word.color)
            
            if cache_key in self.word_image_cache:
                continue
            
            # Create word image once
            padding = 100
            canvas_width = word.width + padding * 2
            canvas_height = word.height + padding * 2
            
            # Pre-render at full opacity
            word_img = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(word_img)
            
            # Cache font
            if word.font_size not in self.font_cache:
                try:
                    font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', word.font_size)
                except:
                    font = ImageFont.load_default()
                self.font_cache[word.font_size] = font
            
            font = self.font_cache[word.font_size]
            draw.text((padding, padding), word.text, fill=(*word.color, 255), font=font)
            
            # Convert to numpy and cache
            word_array = np.array(word_img)
            self.word_image_cache[cache_key] = word_array
        
        print(f"   ✓ Cached {len(self.word_image_cache)} unique word images")
        return self.word_image_cache
    
    def build_time_index(self, word_objects: List[WordObject], fps: float) -> Dict:
        """Build index of active words for each frame time"""
        print("📊 Building time index for fast lookup...")
        
        time_index = defaultdict(lambda: {'behind': [], 'front': []})
        
        # Group words by their active time ranges
        for word in word_objects:
            animation_start = word.start_time - word.rise_duration
            
            # Calculate frame range
            start_frame = max(0, int(animation_start * fps))
            end_frame = int(word.end_time * fps)
            
            # Add to index for each frame it's active
            for frame_num in range(start_frame, end_frame + 1):
                if word.is_behind:
                    time_index[frame_num]['behind'].append(word)
                else:
                    time_index[frame_num]['front'].append(word)
        
        # Convert to regular dict for faster access
        self.active_words_by_time = dict(time_index)
        
        # Stats
        max_concurrent = max(
            len(v['behind']) + len(v['front']) 
            for v in self.active_words_by_time.values()
        )
        print(f"   ✓ Built index for {len(self.active_words_by_time)} frames")
        print(f"   ✓ Max concurrent words: {max_concurrent}")
        
        return self.active_words_by_time
    
    def get_active_words_for_frame(self, frame_num: int) -> Tuple[List[WordObject], List[WordObject]]:
        """Get active words for a specific frame (O(1) lookup)"""
        if frame_num in self.active_words_by_time:
            data = self.active_words_by_time[frame_num]
            return data['behind'], data['front']
        return [], []
    
    def batch_apply_fog(self, word_images: List[np.ndarray], 
                        fog_progress: float) -> List[np.ndarray]:
        """Apply fog effect to multiple words at once using vectorized operations"""
        if fog_progress <= 0:
            return word_images
        
        # Stack all images for vectorized processing
        stacked = np.stack(word_images, axis=0)
        
        # Apply fog to alpha channel (vectorized)
        fog_alpha = 1.0 - fog_progress
        stacked[:, :, :, 3] = (stacked[:, :, :, 3] * fog_alpha).astype(np.uint8)
        
        # Apply slight blur (vectorized using separable filter)
        if fog_progress > 0.3:
            blur_amount = int(fog_progress * 5)
            if blur_amount > 0:
                kernel = cv2.getGaussianKernel(blur_amount * 2 + 1, blur_amount)
                for i in range(len(stacked)):
                    # Apply separable Gaussian blur (faster than 2D)
                    stacked[i] = cv2.sepFilter2D(stacked[i], -1, kernel, kernel)
        
        return list(stacked)


class FrameRenderPool:
    """Pool of pre-allocated frames for faster rendering"""
    
    def __init__(self, width: int, height: int, pool_size: int = 10):
        self.width = width
        self.height = height
        self.pool_size = pool_size
        self.frame_pool = []
        
        # Pre-allocate frames
        for _ in range(pool_size):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            self.frame_pool.append(frame)
        
        self.current_idx = 0
    
    def get_frame(self) -> np.ndarray:
        """Get a pre-allocated frame from pool"""
        frame = self.frame_pool[self.current_idx]
        self.current_idx = (self.current_idx + 1) % self.pool_size
        return frame
    
    def copy_to_frame(self, source: np.ndarray) -> np.ndarray:
        """Copy source to a pooled frame"""
        frame = self.get_frame()
        np.copyto(frame, source)
        return frame


class OptimizationStats:
    """Track optimization performance metrics"""
    
    def __init__(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.words_skipped = 0
        self.words_rendered = 0
        self.frames_processed = 0
    
    def report(self):
        """Print optimization statistics"""
        print("\n📈 Optimization Statistics:")
        print(f"   Cache hit rate: {self.cache_hits / max(1, self.cache_hits + self.cache_misses):.1%}")
        print(f"   Words skipped: {self.words_skipped}")
        print(f"   Words rendered: {self.words_rendered}")
        print(f"   Avg words/frame: {self.words_rendered / max(1, self.frames_processed):.1f}")


# Example of how to integrate optimizations
def create_optimized_pipeline():
    """Factory function to create optimized pipeline components"""
    
    from .pipeline import WordLevelPipeline
    from .frame_processor import FrameProcessor
    
    # Create optimizer
    optimizer = BatchOptimizer()
    
    # Monkey-patch the frame processor to use optimizations
    original_process_frame = FrameProcessor.process_frame
    
    def optimized_process_frame(self, frame, time_seconds, word_objects, 
                               sentence_fog_times, frame_number=None):
        # Use time index for fast lookup
        if hasattr(self, '_optimizer'):
            fps = 25  # Assume 25 fps, could be passed as parameter
            frame_num = int(time_seconds * fps)
            behind_words, front_words = self._optimizer.get_active_words_for_frame(frame_num)
            
            # Only process active words (huge speedup!)
            active_word_objects = behind_words + front_words
            
            # Call original with filtered words
            return original_process_frame(
                self, frame, time_seconds, active_word_objects,
                sentence_fog_times, frame_number
            )
        else:
            return original_process_frame(
                self, frame, time_seconds, word_objects,
                sentence_fog_times, frame_number
            )
    
    # Apply optimization
    FrameProcessor.process_frame = optimized_process_frame
    
    return optimizer