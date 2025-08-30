"""
Scene processing logic for word-level pipeline
"""

from typing import List, Tuple

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
        """Process phrase placements - group by position for continuous lines"""
        # Group phrases and placements by position
        position_groups = {}
        for phrase, placement in zip(phrases, placements):
            y_pos = placement.position[1]
            if y_pos not in position_groups:
                position_groups[y_pos] = []
            position_groups[y_pos].append((phrase, placement))
        
        # Process each position group as a continuous line
        for y_pos, group in position_groups.items():
            if len(group) > 1:
                print(f"       📝 Creating continuous line at y={y_pos} with {len(group)} phrases")
            
            # Collect all words from all phrases at this position
            all_words = []
            all_timings = []
            combined_text = []
            
            # Use properties from first placement (they should all be similar)
            first_placement = group[0][1]
            
            for phrase, placement in group:
                # Get actual word timings from the original transcript
                word_timings = transcript_handler.extract_word_timings(phrase, transcript_path)
                
                # Collect all words and timings
                all_words.extend(phrase.words)
                all_timings.extend(word_timings)
                combined_text.append(phrase.text)
            
            # Create a single continuous phrase text
            continuous_text = " ".join(combined_text)
            
            if len(group) > 1:
                print(f"         Combined: '{continuous_text}'")
                print(f"         Total words: {len(all_words)}")
                print(f"         Font size: {first_placement.font_size}")
                print(f"         Behind: {first_placement.is_behind}")
            
            # Create words for the entire continuous line
            words = pipeline.create_phrase_words(
                continuous_text, all_timings, first_placement, from_below=from_below,
                scene_index=scene_index
            )
            
            pipeline.word_objects.extend(words)