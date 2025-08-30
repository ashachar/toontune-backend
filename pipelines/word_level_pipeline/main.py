"""
Core video creation logic for word-level pipeline
"""

import os
from pathlib import Path

from .pipeline import WordLevelPipeline
from .transcript_handler import TranscriptHandler
from .video_generator import VideoGenerator
from .masking import ForegroundMaskExtractor
from .scene_processor import SceneProcessor


def create_word_level_video(input_video_path: str = None, duration_seconds: float = 6.0, output_name: str = None):
    """Create video with word-level tracking throughout
    
    Args:
        input_video_path: Path to input video (if None, will prompt user)
        duration_seconds: Duration of segment to process (default 6.0 for demo)
        output_name: Custom output filename (optional)
    """
    
    print("Creating Word-Level Animation Pipeline with Enriched Transcript")
    print("=" * 60)
    print("Features:")
    print("  • AI-powered transcript enrichment for intelligent phrasing")
    print("  • Face-aware text placement (avoids covering faces)")
    print("  • Stripe-based layout with optimal positioning")
    print("  • Importance-based visual styling")
    print("  • Words tracked individually throughout")
    print("  • Fixed positions maintained during fog")
    print("  • Clean fog dissolve effect")
    print()
    
    # Handle running from either backend/ or pipelines/ directory
    if os.path.exists("uploads"):
        base_dir = "."
    else:
        base_dir = ".."
    
    # Validate inputs
    if not input_video_path:
        print("❌ Error: No input video path provided")
        print("Usage: create_word_level_video('path/to/video.mp4')")
        return None
    
    if not os.path.exists(input_video_path):
        print(f"❌ Error: Input video not found: {input_video_path}")
        return None
    
    # Setup output paths
    video_name = Path(input_video_path).stem
    temp_segment = f"{base_dir}/outputs/{video_name}_{duration_seconds}sec_segment.mp4"
    os.makedirs(f"{base_dir}/outputs", exist_ok=True)
    
    # Initialize components with video path for cached mask support
    video_gen = VideoGenerator(input_video_path)
    transcript_handler = TranscriptHandler()
    mask_extractor = ForegroundMaskExtractor(input_video_path)
    pipeline = WordLevelPipeline(font_size=55)
    
    # Extract segment
    video_gen.extract_segment(input_video_path, duration_seconds, temp_segment)
    
    # Process transcript and create word objects
    word_objects, sentence_fog_times = _process_transcript(
        transcript_handler, video_gen, mask_extractor, pipeline,
        input_video_path, temp_segment, duration_seconds
    )
    
    if not word_objects:
        return None
    
    # Render and output video
    return _render_final_video(
        video_gen, pipeline, temp_segment, word_objects, sentence_fog_times,
        base_dir, video_name, duration_seconds, output_name
    )


def _process_transcript(transcript_handler, video_gen, mask_extractor, pipeline,
                       input_video_path, temp_segment, duration_seconds):
    """Process transcript and create word objects"""
    transcript_path = transcript_handler.get_transcript_path(input_video_path)
    
    if not transcript_path or not transcript_path.exists():
        print(f"❌ No transcript found at: {transcript_path}")
        print("   Please generate a transcript first using ElevenLabs or another transcription service")
        return [], []
    
    try:
        # Load enriched phrases
        phrase_groups = transcript_handler.load_enriched_phrases(transcript_path, duration_seconds)
        
        # Extract sample frames for visibility testing (kept for backward compatibility)
        sample_frames = video_gen.extract_sample_frames(temp_segment)
        
        # Extract foreground masks from sample frames (kept for backward compatibility)
        print("   Extracting foreground masks for visibility testing...")
        foreground_masks = []
        for frame in sample_frames:
            mask = mask_extractor.extract_foreground_mask(frame)
            foreground_masks.append(mask)
        
        # Process scenes and create word objects
        # Note: video_gen and temp_segment are passed for per-scene frame extraction
        return _create_word_objects_from_scenes(
            transcript_handler, pipeline, phrase_groups, foreground_masks, 
            sample_frames, transcript_path, video_gen, temp_segment
        )
        
    except ValueError as e:
        print(f"❌ Error generating enriched transcript: {e}")
        print("   Cannot proceed without transcript and Gemini API")
        return [], []


def _create_word_objects_from_scenes(transcript_handler, pipeline, phrase_groups, 
                                   foreground_masks, sample_frames, transcript_path,
                                   video_gen=None, temp_segment=None):
    """Create word objects from enriched phrase scenes"""
    scene_processor = SceneProcessor()
    return scene_processor.create_word_objects_from_scenes(
        transcript_handler, pipeline, phrase_groups, foreground_masks, 
        sample_frames, transcript_path, video_gen, temp_segment
    )


def _render_final_video(video_gen, pipeline, temp_segment, word_objects, sentence_fog_times,
                       base_dir, video_name, duration_seconds, output_name):
    """Render and output the final video"""
    # Render video
    temp_output = f"{base_dir}/outputs/word_level_pipeline_temp.mp4"
    video_gen.render_video(temp_segment, word_objects, sentence_fog_times, 
                          temp_output, duration_seconds)
    
    # Generate output filename
    if output_name:
        final_output = f"{base_dir}/outputs/{output_name}"
    else:
        final_output = f"{base_dir}/outputs/{video_name}_word_level_h264.mp4"
    
    # Merge with audio and convert to H.264
    video_gen.merge_audio_and_convert(temp_output, temp_segment, final_output)
    
    # Clean up temp files
    video_gen.cleanup_temp_files(temp_output, temp_segment)
    
    print(f"\n✅ Word-level pipeline video created: {final_output}")
    print("\nKey features:")
    print("  ✓ AI-powered transcript enrichment for intelligent phrasing")
    print("  ✓ Face-aware text placement (avoids covering faces)")
    print("  ✓ Stripe-based layout with optimal vertical positioning")
    print("  ✓ Smart horizontal positioning (face-aware or centered)")
    print("  ✓ Visibility testing for foreground/background placement")
    print("  ✓ Importance-based visual styling (size, position, emphasis)")
    print("  ✓ All words in sentence from SAME direction")
    print("  ✓ Word objects maintained throughout")
    print("  ✓ Fixed positions - no movement during fog")
    print("  ✓ Smooth fog dissolve transitions")
    print("  ✓ Words stay dissolved - no reappearing")
    print("  ✓ Audio preserved from original")
    
    return final_output