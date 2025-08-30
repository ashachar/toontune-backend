"""
Main WordLevelPipeline class and orchestration logic
"""

import cv2
import numpy as np
import sys
import os
from typing import List

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.text_placement.two_position_layout import TwoPositionLayoutManager

from .models import WordObject
from .masking import ForegroundMaskExtractor
from .word_factory import WordFactory
from .frame_processor import FrameProcessor


class WordLevelPipeline:
    """
    Pipeline that maintains word-level objects throughout all animations
    """
    
    def __init__(self, font_size=55):
        self.font_size = font_size
        
        # Initialize components
        self.mask_extractor = ForegroundMaskExtractor()
        self.word_factory = WordFactory(font_size)
        self.frame_processor = FrameProcessor()
        
        # Store all word objects
        self.word_objects: List[WordObject] = []
        
        # Initialize two-position layout manager
        self.layout_manager = TwoPositionLayoutManager()
    
    def extract_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Extract foreground mask from frame using edge detection and color analysis"""
        return self.mask_extractor.extract_foreground_mask(frame)
    
    def create_phrase_words(self, text: str, word_timings: List[dict], 
                           placement, from_below: bool, scene_index: int = 0) -> List[WordObject]:
        """Create word objects for a phrase using stripe placement with horizontal centering"""
        return self.word_factory.create_phrase_words(text, word_timings, placement, from_below, scene_index)
    
    def create_sentence_words(self, text: str, word_timings: List[dict], 
                             center: tuple, from_below: bool) -> List[WordObject]:
        """Create word objects for a sentence with fixed positions"""
        return self.word_factory.create_sentence_words(text, word_timings, center, from_below)
    
    def process_frame(self, frame: np.ndarray, time_seconds: float,
                     sentence_fog_times: List[tuple]) -> np.ndarray:
        """Process frame by rendering all active words with proper layering"""
        return self.frame_processor.process_frame(frame, time_seconds, 
                                                 self.word_objects, sentence_fog_times)