#!/usr/bin/env python3
"""Debug the actual pipeline to see what positions are used"""

import cv2
from pipelines.word_level_pipeline import WordLevelPipeline
from utils.text_placement.stripe_layout_manager import TextPlacement

# Create pipeline
pipeline = WordLevelPipeline(font_size=55)

# Load a sample frame
frame = cv2.imread('./outputs/frame_check.png')
sample_frames = [frame]

# Extract foreground masks
foreground_masks = []
for f in sample_frames:
    mask = pipeline.extract_foreground_mask(f)
    foreground_masks.append(mask)

# Create test phrase data
phrase_dicts = [{
    'phrase': 'Yes,',
    'importance': 0.5,
    'layout_priority': 1
}]

# Call layout_scene_phrases
placements = pipeline.layout_manager.layout_scene_phrases(
    phrase_dicts, foreground_masks, sample_frames
)

print("Placements from layout manager:")
for p in placements:
    print(f"  Phrase: '{p.phrase}'")
    print(f"  Position: {p.position}")

# Now test create_phrase_words
if placements:
    placement = placements[0]
    word_timings = [{'start': 0.14, 'end': 0.5}]
    
    print(f"\nPassing to create_phrase_words:")
    print(f"  placement.position = {placement.position}")
    
    words = pipeline.create_phrase_words(
        'Yes,',
        word_timings,
        placement,
        from_below=True
    )
    
    print(f"\nWord objects created:")
    for w in words:
        print(f"  Word: '{w.text}' at x={w.x}, y={w.y}")