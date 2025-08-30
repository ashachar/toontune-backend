#!/usr/bin/env python3
"""
Complete pipeline to process entire video with dynamic backgrounds.
Handles RVM generation, background allocation, and final video creation.
"""

import json
import subprocess
import yaml
import random
from pathlib import Path
from datetime import datetime
from utils.video.background.background_cache_manager import BackgroundCacheManager
from utils.video.background.cached_rvm import CachedRobustVideoMatting
from utils.video.background.coverr_manager import CoverrManager


def allocate_backgrounds_from_transcript(video_path, duration):
    """
    Analyze transcript and allocate background themes.
    Uses mock LLM for demonstration (can be replaced with actual API).
    """
    project_folder = video_path.parent / video_path.stem
    transcript_path = project_folder / "transcript.json"
    
    print("📝 Analyzing transcript for background allocation...")
    
    with open(transcript_path, 'r') as f:
        transcript_data = json.load(f)
    
    transcript_text = transcript_data['text']
    
    # Mock allocation based on transcript analysis
    # In production, this would use the LLM prompt
    segments = []
    
    # Analyze key phrases in transcript to determine themes
    text_lower = transcript_text.lower()
    
    # Define segment boundaries based on content shifts
    segment_definitions = [
        {"start": 0, "end": 15, "theme": "abstract_tech", 
         "keywords": ["AI", "innovation", "technology", "digital"],
         "reason": "Opening introduces AI creating new mathematics"},
        
        {"start": 15, "end": 35, "theme": "mathematics",
         "keywords": ["calculus", "derivative", "integral", "equations"],
         "reason": "Discussion of mathematical operators"},
        
        {"start": 35, "end": 58, "theme": "innovation",
         "keywords": ["create", "build", "develop", "new"],
         "reason": "AI agents creating new tools"},
        
        {"start": 58, "end": 85, "theme": "research",
         "keywords": ["research", "study", "investigate", "science"],
         "reason": "AI struggles and research discussion"},
        
        {"start": 85, "end": 115, "theme": "data_visualization",
         "keywords": ["data", "science", "applications", "trends"],
         "reason": "Data science applications discussion"},
        
        {"start": 115, "end": 145, "theme": "cosmic",
         "keywords": ["philosophy", "discover", "invent", "universe"],
         "reason": "Philosophical debate on discovery vs invention"},
        
        {"start": 145, "end": 175, "theme": "innovation",
         "keywords": ["framework", "theory", "generate", "process"],
         "reason": "New mathematical theory development"},
        
        {"start": 175, "end": duration, "theme": "education",
         "keywords": ["overview", "learn", "understand", "explain"],
         "reason": "Theory overview and explanation"}
    ]
    
    # Adjust segment times to actual duration
    for seg_def in segment_definitions:
        if seg_def["start"] < duration:
            segments.append({
                "start_time": seg_def["start"],
                "end_time": min(seg_def["end"], duration),
                "theme": seg_def["theme"],
                "keywords": seg_def["keywords"],
                "reason": seg_def["reason"]
            })
    
    return segments


def process_full_video_with_backgrounds():
    """
    Main pipeline to process entire video with dynamic backgrounds.
    """
    # Setup paths
    video_path = Path("uploads/assets/videos/ai_math1.mp4")
    project_folder = video_path.parent / video_path.stem
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("FULL VIDEO BACKGROUND PIPELINE")
    print("=" * 80)
    print(f"Input: {video_path}")
    
    # Get video duration
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    duration = float(subprocess.check_output(cmd).decode().strip())
    print(f"Duration: {duration:.1f} seconds")
    
    # Initialize managers
    cache_manager = BackgroundCacheManager()
    rvm_processor = CachedRobustVideoMatting()
    coverr_manager = CoverrManager()
    
    # Step 1: Get or generate RVM for FULL video
    print("\n" + "=" * 80)
    print("STEP 1: ROBUST VIDEO MATTING (Full Video)")
    print("=" * 80)
    
    print("⏳ Processing full video with RVM (this may take several minutes)...")
    
    # Process entire video - this will check cache first
    green_screen_path = rvm_processor.get_rvm_output(video_path, duration=None)
    
    if not green_screen_path or not Path(green_screen_path).exists():
        print("❌ Failed to get RVM output")
        return None
    
    print(f"✅ RVM output ready: {green_screen_path}")
    
    # Step 2: Allocate backgrounds based on transcript
    print("\n" + "=" * 80)
    print("STEP 2: BACKGROUND ALLOCATION")
    print("=" * 80)
    
    segments = allocate_backgrounds_from_transcript(video_path, duration)
    
    print(f"\n📊 Allocated {len(segments)} segments:")
    for i, seg in enumerate(segments, 1):
        duration_seg = seg['end_time'] - seg['start_time']
        print(f"  {i}. [{seg['start_time']:6.1f}s - {seg['end_time']:6.1f}s] "
              f"({duration_seg:5.1f}s) : {seg['theme']:20s}")
    
    # Save allocation
    allocation_file = project_folder / "full_video_allocation.json"
    with open(allocation_file, 'w') as f:
        json.dump(segments, f, indent=2)
    print(f"\n✅ Allocation saved: {allocation_file}")
    
    # Step 3: Get backgrounds for each segment
    print("\n" + "=" * 80)
    print("STEP 3: BACKGROUND ACQUISITION")
    print("=" * 80)
    
    for i, segment in enumerate(segments):
        print(f"\n[{i+1}/{len(segments)}] {segment['theme']}")
        
        # Check cache first
        cached_bg = cache_manager.get_best_match(
            theme=segment['theme'],
            keywords=segment.get('keywords', [])
        )
        
        if cached_bg:
            print(f"  ✅ Found in cache: {cached_bg.name}")
            segment['background_path'] = cached_bg
        else:
            print(f"  🔍 Searching online for {segment['theme']}...")
            
            # Try to download from Coverr
            videos = coverr_manager.search_videos(segment.get('keywords', []))
            
            if videos:
                best_video = coverr_manager.select_best_video(
                    videos, segment.get('reason', '')
                )
                
                download_path = coverr_manager.download_video(
                    best_video,
                    video_path.stem,
                    project_folder,
                    segment['start_time'],
                    segment['end_time']
                )
                
                if download_path and download_path.exists():
                    # Add to cache
                    cached_path = cache_manager.add_background(
                        download_path,
                        theme=segment['theme'],
                        keywords=segment.get('keywords', []),
                        source="coverr",
                        source_id=best_video.get('id')
                    )
                    segment['background_path'] = cached_path
                    print(f"  ✅ Downloaded and cached")
            
            if not segment.get('background_path'):
                print(f"  ⚠️ No background found, will use fallback")
                # Create a simple colored background as fallback
                segment['background_path'] = None
    
    # Step 4: Process each segment with chromakey
    print("\n" + "=" * 80)
    print("STEP 4: VIDEO PROCESSING")
    print("=" * 80)
    
    processed_segments = []
    
    for i, segment in enumerate(segments):
        print(f"\n[{i+1}/{len(segments)}] Processing {segment['theme']} "
              f"({segment['start_time']:.1f}s - {segment['end_time']:.1f}s)")
        
        duration_seg = segment['end_time'] - segment['start_time']
        segment_output = output_dir / f"full_seg_{i:02d}_{segment['theme']}.mp4"
        
        # Handle missing backgrounds - NEVER skip segments!
        if not segment.get('background_path'):
            print(f"  ⚠️ No background found - using fallback image")
            
            # Use a random image from backgrounds/images folder
            images_dir = Path("uploads/assets/backgrounds/images")
            available_images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            
            if available_images:
                # Pick a random image
                fallback_image = random.choice(available_images)
                print(f"  📦 Using fallback image: {fallback_image.name}")
                
                # Convert image to video for this segment duration
                fallback_video = output_dir / f"fallback_{i:02d}_{segment['theme']}.mp4"
                
                # Create static video from image - scale to fill entire screen
                cmd_img_to_vid = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", str(fallback_image),
                    "-t", str(duration_seg),
                    "-vf", "scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    str(fallback_video)
                ]
                
                result = subprocess.run(cmd_img_to_vid, capture_output=True, text=True)
                if result.returncode == 0:
                    segment['background_path'] = fallback_video
                    print(f"  ✅ Created fallback video from image")
                else:
                    print(f"  ❌ Failed to create fallback video")
                    print(f"     Error: {result.stderr[-200:]}")
                    continue
            else:
                print(f"  ❌ No fallback images found in {images_dir}")
                # Use a cached video as last resort
                fallback_bg = cache_manager.get_best_match(theme="abstract_tech")
                if fallback_bg:
                    segment['background_path'] = fallback_bg
                    print(f"  📦 Using cached video as fallback: {fallback_bg.name}")
                else:
                    print(f"  ❌ No fallback available - segment will be missing!")
                    continue
        
        # Apply chromakey
        cmd = [
            "ffmpeg", "-y",
            # Background (looped)
            "-stream_loop", "-1", "-i", str(segment['background_path']),
            # Green screen segment
            "-ss", str(segment['start_time']), "-t", str(duration_seg),
            "-i", str(green_screen_path),
            # Filters
            "-filter_complex",
            "[0:v]scale=1280:720,setpts=PTS-STARTPTS[bg];"
            "[1:v]scale=1280:720,setpts=PTS-STARTPTS[fg];"
            # Refined chromakey settings
            "[fg]chromakey=green:0.08:0.04[keyed];"
            "[keyed]despill=type=green:mix=0.2:expand=0[clean];"
            "[bg][clean]overlay=shortest=1[out]",
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(segment_output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            processed_segments.append(segment_output)
            print(f"  ✅ Processed: {segment_output.name}")
        else:
            print(f"  ❌ Failed to process segment")
            print(f"  Error: {result.stderr[-500:]}")
    
    # Step 5: Concatenate all segments
    if processed_segments:
        print("\n" + "=" * 80)
        print("STEP 5: FINAL VIDEO ASSEMBLY")
        print("=" * 80)
        
        concat_list = output_dir / "full_concat.txt"
        with open(concat_list, 'w') as f:
            for segment_path in processed_segments:
                f.write(f"file '{segment_path.absolute()}'\n")
        
        # Concatenate without audio first
        final_no_audio = output_dir / "ai_math1_full_no_audio.mp4"
        
        print("🔗 Concatenating segments...")
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final_no_audio)
        ]
        
        result = subprocess.run(cmd_concat, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Concatenated: {final_no_audio.name}")
            
            # Add audio from original
            print("🔊 Adding audio from original video...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_with_audio = output_dir / f"ai_math1_FULL_BACKGROUNDS_{timestamp}.mp4"
            
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", str(final_no_audio),
                "-i", str(video_path),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "128k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                str(final_with_audio)
            ]
            
            result = subprocess.run(cmd_audio, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Final video: {final_with_audio.name}")
                
                # Get final stats
                size_mb = final_with_audio.stat().st_size / (1024 * 1024)
                cmd_duration = ["ffprobe", "-v", "error", "-show_entries",
                              "format=duration", "-of",
                              "default=noprint_wrappers=1:nokey=1",
                              str(final_with_audio)]
                final_duration = float(subprocess.check_output(cmd_duration).decode().strip())
                
                # Clean up intermediate files
                print("\n🧹 Cleaning up temporary files...")
                for seg_file in processed_segments:
                    seg_file.unlink()
                final_no_audio.unlink()
                concat_list.unlink()
                
                # Show summary
                print("\n" + "=" * 80)
                print("✅ SUCCESS! FULL VIDEO PROCESSED")
                print("=" * 80)
                print(f"\n📹 Final Output: {final_with_audio}")
                print(f"📏 Size: {size_mb:.1f} MB")
                print(f"⏱️  Duration: {final_duration:.1f} seconds")
                print(f"🎬 Segments: {len(segments)}")
                
                print("\n📊 Background Timeline:")
                for i, seg in enumerate(segments, 1):
                    duration_seg = seg['end_time'] - seg['start_time']
                    print(f"  {i}. [{seg['start_time']:6.1f}s - {seg['end_time']:6.1f}s] "
                          f"({duration_seg:5.1f}s) : {seg['theme']}")
                
                # Open result
                print(f"\n🎥 Opening final video...")
                subprocess.run(["open", str(final_with_audio)])
                
                return final_with_audio
            else:
                print(f"❌ Audio merge failed")
        else:
            print(f"❌ Concatenation failed")
    else:
        print("\n❌ No segments were processed successfully")
    
    return None


def main():
    """Run the full video background pipeline."""
    
    video_path = Path("uploads/assets/videos/ai_math1.mp4")
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    print("🚀 Starting Full Video Background Pipeline")
    print("⚠️  This will process the entire 206-second video")
    print("⏳ Expected time: 5-10 minutes\n")
    
    import sys
    if "--no-prompt" not in sys.argv:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    result = process_full_video_with_backgrounds()
    
    if result:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed")


if __name__ == "__main__":
    main()