"""
Scene processing logic for word-level pipeline
"""

from typing import List, Tuple
from dataclasses import replace

from .transcript_handler import TranscriptHandler
from .pipeline import WordLevelPipeline


class SceneProcessor:
    """Processes enriched phrase scenes into word objects"""
    
    def __init__(self):
        pass
    
    def create_word_objects_from_scenes(self, transcript_handler: TranscriptHandler, 
                                      pipeline: WordLevelPipeline, phrase_groups, 
                                      foreground_masks, sample_frames, transcript_path,
                                      video_gen=None, video_path=None) -> Tuple[List, List]:
        """Create word objects from enriched phrase scenes
        
        Args:
            transcript_handler: Handler for transcript operations
            pipeline: Word level pipeline instance
            phrase_groups: Groups of phrases by scene
            foreground_masks: Foreground masks (deprecated - using scene-specific now)
            sample_frames: Sample frames (deprecated - using scene-specific now) 
            transcript_path: Path to transcript file
            video_gen: Video generator instance for extracting scene frames
            video_path: Path to video file for frame extraction
        """
        scene_count = 0
        sentence_fog_times = []
        
        for appearance_idx in sorted(phrase_groups.keys()):  # Process ALL scenes
            phrases = phrase_groups[appearance_idx]
            print(f"\n   🎭 Processing Scene {appearance_idx} with {len(phrases)} phrases:")
            for p in phrases:
                print(f"      '{p.text}' [{p.start_time:.3f}s - {p.end_time:.3f}s]")
            
            # Get scene time range
            if phrases:
                scene_start = min(p.start_time for p in phrases)
                scene_end = max(p.end_time for p in phrases)
                
                # Extract frames specifically for this scene if video_gen provided
                if video_gen and video_path:
                    print(f"   📹 Extracting frames for scene {appearance_idx} [{scene_start:.2f}s - {scene_end:.2f}s]")
                    scene_frames = video_gen.extract_scene_frames(video_path, scene_start, scene_end, num_samples=20)
                    
                    # Extract masks for these scene frames
                    from .masking import ForegroundMaskExtractor
                    mask_extractor = ForegroundMaskExtractor()
                    scene_masks = []
                    for frame in scene_frames:
                        mask = mask_extractor.extract_foreground_mask(frame)
                        scene_masks.append(mask)
                    print(f"      Extracted {len(scene_masks)} masks for head detection")
                else:
                    # Fallback to original masks/frames
                    scene_frames = sample_frames
                    scene_masks = foreground_masks
            else:
                scene_frames = sample_frames
                scene_masks = foreground_masks
            
            # Convert phrases to dict format for layout manager
            phrase_dicts = transcript_handler.convert_phrases_to_dicts(phrases)
            
            # Use layout manager with scene-specific frames/masks
            placements = pipeline.layout_manager.layout_scene_phrases(
                phrase_dicts, scene_masks, scene_frames
            )
            
            print(f"\n   Scene {appearance_idx}:")
            for placement in placements:
                visibility_pct = placement.visibility_score * 100
                position_type = "behind" if placement.is_behind else "front"
                print(f"     '{placement.phrase[:30]}...' - {visibility_pct:.1f}% visible → {position_type}")
            
            # Determine direction for this scene
            from_below = (scene_count % 2 == 0)
            
            # Create word objects from placements
            self._process_phrase_placements(
                transcript_handler, pipeline, phrases, placements, from_below, transcript_path, scene_count
            )
            
            # Add fog time for this scene
            if phrases:
                last_phrase = phrases[-1]
                fog_start = last_phrase.end_time + 0.2
                fog_end = fog_start + 0.4
                sentence_fog_times.append((fog_start, fog_end))
                print(f"   🌫️ Scene {appearance_idx} fog: {fog_start:.3f}s - {fog_end:.3f}s (after '{last_phrase.text[:20]}...')")
            
            scene_count += 1
        
        print(f"\n   Created {len(pipeline.word_objects)} word objects from {len(phrase_groups)} phrase groups")
        behind_count = sum(1 for w in pipeline.word_objects if w.is_behind)
        print(f"   Behind foreground: {behind_count}, In front: {len(pipeline.word_objects) - behind_count}")
        print(f"   Scenes: {scene_count}")
        
        # Debug: Show all words and their timings
        print("\n   📝 All word objects created:")
        for w in pipeline.word_objects[:20]:  # Show first 20 words
            print(f"      '{w.text}' start={w.start_time:.3f}s")
        
        print(f"\n   🌫️ Fog transition times: {sentence_fog_times}")
        
        return pipeline.word_objects, sentence_fog_times
    
    def _process_phrase_placements(self, transcript_handler: TranscriptHandler, 
                                  pipeline: WordLevelPipeline, phrases, placements, 
                                  from_below: bool, transcript_path, scene_index: int = 0):
        """Process phrase placements - stack multiple phrases vertically when at same position"""
        # CRITICAL FIX: Group phrases by general vertical zone (top/middle/bottom)
        # not exact Y position, to handle multiple phrases properly
        frame_height = 720  # Standard video height
        top_zone = frame_height * 0.3  # Top 30%
        bottom_zone = frame_height * 0.7  # Bottom 30%
        
        zone_groups = {'top': [], 'middle': [], 'bottom': []}
        
        for phrase, placement in zip(phrases, placements):
            y_pos = placement.position[1]
            if y_pos < top_zone:
                zone_groups['top'].append((phrase, placement))
            elif y_pos > bottom_zone:
                zone_groups['bottom'].append((phrase, placement))
            else:
                zone_groups['middle'].append((phrase, placement))
        
        # Process each zone
        for zone_name, group in zone_groups.items():
            if not group:
                continue
                
            if len(group) > 1:
                print(f"       📝 Stacking {len(group)} phrases in {zone_name} zone")
                
                # CRITICAL FIX: Calculate proper spacing to avoid overlapping bounding boxes
                # We need to account for actual text height plus a 10% gap
                
                # Calculate the visual bounding box height of each phrase
                # Use a middle ground: include some padding but not all of it
                phrase_heights = []
                
                for phrase, placement in group:
                    # Base text height estimate
                    base_text_height = int(placement.font_size * 1.2)
                    # Add PARTIAL padding for visual bounding box
                    # Using 40% of the ascender/descender space for a middle ground
                    # This gives us some padding without excessive spacing
                    partial_padding = int((30 + 50) * 0.4)  # 40% of 80px = 32px
                    visual_bbox_height = base_text_height + partial_padding
                    phrase_heights.append(visual_bbox_height)
                    
                print(f"         Bounding box heights: {phrase_heights}")
                
                # Calculate line spacing with 4% gap between bounding boxes
                line_spacings = []
                for i in range(len(phrase_heights)):
                    if i == 0:
                        line_spacings.append(0)  # First line starts at base
                    else:
                        # Previous phrase height + 10% gap + current phrase positioning
                        gap = int(phrase_heights[i-1] * 0.10)  # 10% of previous line height
                        spacing = phrase_heights[i-1] + gap
                        line_spacings.append(line_spacings[i-1] + spacing)
                
                # Calculate total height of the stack
                total_height = line_spacings[-1] + phrase_heights[-1] if line_spacings else 0
                
                # Calculate starting Y position to center the stack
                if zone_name == 'top':
                    base_y = 100  # Start from top with padding
                elif zone_name == 'bottom':
                    base_y = frame_height - 150 - total_height  # Start from bottom with padding
                else:
                    base_y = (frame_height - total_height) // 2  # Center in middle
                
                # Process each phrase separately with properly spaced Y positions
                for i, (phrase, placement) in enumerate(group):
                    # Get actual word timings from the original transcript
                    word_timings = transcript_handler.extract_word_timings(phrase, transcript_path)
                    
                    # Create a new placement with adjusted Y position
                    from dataclasses import replace
                    stacked_y = base_y + line_spacings[i]
                    adjusted_placement = replace(placement, position=(placement.position[0], stacked_y))
                    
                    print(f"         - '{phrase.text}' at y={stacked_y} (height={phrase_heights[i]}, gap from prev={line_spacings[i] - line_spacings[i-1] if i > 0 else 0})")
                    
                    # Create words for this phrase at its stacked position
                    words = pipeline.create_phrase_words(
                        phrase.text, word_timings, adjusted_placement, from_below=from_below,
                        scene_index=scene_index
                    )
                    
                    pipeline.word_objects.extend(words)
            else:
                # Single phrase at this position - process normally
                phrase, placement = group[0]
                word_timings = transcript_handler.extract_word_timings(phrase, transcript_path)
                
                words = pipeline.create_phrase_words(
                    phrase.text, word_timings, placement, from_below=from_below,
                    scene_index=scene_index
                )
                
                pipeline.word_objects.extend(words)