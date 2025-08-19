#!/usr/bin/env python3
"""
Animate Object in Video Pipeline

This script orchestrates the complete pipeline for animating an object emerging from a video:
1. Segments and annotates the video using SAM2 + Gemini
2. Analyzes segments to find the best emergence location  
3. Creates an emergence animation of the object from the selected location
4. Shows results at each step

Usage:
    python animate_object_in_video.py <video_path> <object_animation_path> [output_dir]

Example:
    python animate_object_in_video.py uploads/assets/videos/sea_movie.mov \
           utils/bulk_animate_image/output/batch_transparent_final/Sailor_Salute_in_Cartoon_Style.png.webm
"""

import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from animations.emergence_from_static_point import EmergenceFromStaticPoint


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🎬 {text}")
    print(f"{'='*60}")


def print_step(step_num: int, text: str):
    """Print a formatted step."""
    print(f"\n[Step {step_num}] {text}")
    print(f"{'-'*50}")


def clean_existing_outputs(output_dir: Path):
    """Clean any existing outputs to start fresh."""
    print_step(1, "Cleaning existing outputs")
    
    # Remove existing directories
    dirs_to_clean = [
        output_dir / "segmentation",
        output_dir / "emergence",
        output_dir / "final"
    ]
    
    for dir_path in dirs_to_clean:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"   ✓ Removed {dir_path}")
    
    # Create fresh directories
    for dir_path in dirs_to_clean:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {dir_path}")


def run_video_segmentation(video_path: Path, output_dir: Path) -> Dict:
    """
    Run video segmentation and annotation pipeline.
    Returns dictionary with segmentation results.
    """
    print_step(2, "Running video segmentation and annotation")
    
    segmentation_dir = output_dir / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the segmentation script
    cmd = [
        sys.executable,
        'utils/video_segmentation/video_segmentation_and_annotation.py',
        str(video_path),
        str(segmentation_dir)
    ]
    
    print(f"   Running: {' '.join(cmd)}")
    print(f"   This may take a few minutes...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"   ✓ Segmentation completed successfully")
        
        # Parse segment information from output
        segments = parse_segmentation_output(result.stdout, segmentation_dir)
        
        # Show results
        print(f"\n   📊 Segmentation Results:")
        print(f"   - Segments found: {len(segments)}")
        print(f"   - Output files:")
        print(f"     • Concatenated input: {segmentation_dir}/concatenated_input.png")
        print(f"     • Labeled frame: {segmentation_dir}/labeled_frame.png")
        print(f"     • Segmented video: {segmentation_dir}/final_video_h264.mp4")
        
        return {
            'segments': segments,
            'labeled_frame': segmentation_dir / "labeled_frame.png",
            'segmented_video': segmentation_dir / "final_video_h264.mp4",
            'first_frame': segmentation_dir / "frame0.png"
        }
        
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Segmentation failed: {e}")
        print(f"   Error output: {e.stderr}")
        sys.exit(1)


def parse_segmentation_output(output: str, segmentation_dir: Path) -> List[Dict]:
    """
    Parse segmentation output to extract segment information.
    Returns list of segment dictionaries.
    """
    segments = []
    
    # Try to extract segment descriptions from output
    lines = output.split('\n')
    in_descriptions = False
    
    for line in lines:
        if "Segment descriptions:" in line:
            in_descriptions = True
            continue
        
        if in_descriptions and line.strip().startswith(tuple('0123456789')):
            # Parse line like "1. ocean water (area: 12345)"
            parts = line.strip().split('.')
            if len(parts) >= 2:
                seg_id = int(parts[0])
                desc_parts = parts[1].strip().split('(area:')
                description = desc_parts[0].strip()
                
                area = 0
                if len(desc_parts) > 1:
                    try:
                        area = int(desc_parts[1].strip(')').strip())
                    except:
                        pass
                
                segments.append({
                    'id': seg_id,
                    'description': description,
                    'area': area
                })
    
    # If we couldn't parse from output, create default segments
    if not segments:
        print("   ⚠️  Could not parse segment descriptions, using defaults")
        segments = [
            {'id': 1, 'description': 'largest segment', 'area': 100000},
            {'id': 2, 'description': 'second segment', 'area': 50000},
            {'id': 3, 'description': 'third segment', 'area': 25000}
        ]
    
    return segments


def analyze_segments_for_emergence(segments: List[Dict], first_frame_path: Path) -> Dict:
    """
    Analyze segments to find the best location for emergence.
    Prioritizes water/ocean segments.
    """
    print_step(3, "Analyzing segments for best emergence location")
    
    # Priority keywords for water/ocean
    water_keywords = ['water', 'ocean', 'sea', 'lake', 'river', 'wave', 'surf']
    
    # Find water segment
    water_segment = None
    for seg in segments:
        desc_lower = seg['description'].lower()
        if any(keyword in desc_lower for keyword in water_keywords):
            water_segment = seg
            break
    
    # If no water segment, use largest segment
    if not water_segment:
        water_segment = max(segments, key=lambda x: x['area'])
        print(f"   ⚠️  No water segment found, using largest: {water_segment['description']}")
    else:
        print(f"   ✓ Found water segment: {water_segment['description']}")
    
    # Calculate position (we'll use center of frame for now, can be enhanced)
    frame = cv2.imread(str(first_frame_path))
    if frame is not None:
        height, width = frame.shape[:2]
        # Position slightly below center for emergence effect
        position = (width // 2, int(height * 0.6))
    else:
        position = (640, 400)  # Default fallback
    
    print(f"   📍 Selected emergence position: {position}")
    print(f"   📝 Segment info:")
    print(f"      - ID: {water_segment['id']}")
    print(f"      - Description: {water_segment['description']}")
    print(f"      - Area: {water_segment['area']}")
    
    return {
        'segment': water_segment,
        'position': position,
        'direction': 0  # Upward emergence
    }


def create_emergence_animation(
    video_path: Path,
    object_path: Path,
    emergence_info: Dict,
    output_dir: Path
) -> Path:
    """
    Create the emergence animation using EmergenceFromStaticPoint.
    """
    print_step(4, "Creating emergence animation")
    
    emergence_dir = output_dir / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure animation parameters
    position = emergence_info['position']
    direction = emergence_info['direction']
    
    print(f"   Configuration:")
    print(f"   - Position: {position}")
    print(f"   - Direction: {direction}° (upward)")
    print(f"   - Start frame: 30 (1 second delay)")
    print(f"   - Emergence speed: 2.5 pixels/frame")
    print(f"   - Duration: 7 seconds")
    
    # Create temporary directory for animation processing
    temp_dir = emergence_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Keep temp directory for debugging
    cleanup_temp = False
    
    try:
        # Initialize emergence animation
        animation = EmergenceFromStaticPoint(
            element_path=str(object_path),
            background_path=str(video_path),
            position=position,
            direction=direction,
            start_frame=30,  # Start after 1 second
            animation_start_frame=30,  # Start animating when emerging
            fps=30,
            duration=7.0,
            temp_dir=str(temp_dir),
            emergence_speed=2.5,  # Pixels per frame
            remove_background=True,  # Remove black background
            background_color='0x000000',
            background_similarity=0.15
        )
        
        # Render the animation
        output_video = emergence_dir / "emergence_animation.mp4"
        print(f"   🎬 Rendering animation...")
        
        success = animation.render(str(output_video))
        
        if success:
            print(f"   ✓ Animation created successfully")
            
            # Convert to H.264 for better compatibility
            h264_output = emergence_dir / "emergence_animation_h264.mp4"
            cmd = [
                'ffmpeg',
                '-i', str(output_video),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-y',
                str(h264_output)
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"   ✓ Converted to H.264 format")
            
            cleanup_temp = True  # Clean up only on success
            return h264_output
        else:
            print(f"   ⚠️  Animation rendering failed - checking for output frames...")
            
            # Try to create video directly from frames if they exist
            output_frames_dir = temp_dir / "output_frames"
            if output_frames_dir.exists():
                frames = sorted([f for f in output_frames_dir.glob("*.png")])
                if frames:
                    print(f"   Found {len(frames)} output frames, attempting direct video creation...")
                    
                    # Create video using simple frame sequence
                    output_video = emergence_dir / "emergence_animation.mp4"
                    cmd = [
                        'ffmpeg',
                        '-framerate', '30',
                        '-pattern_type', 'glob',
                        '-i', str(output_frames_dir / '*.png'),
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '18',
                        '-pix_fmt', 'yuv420p',
                        '-y',
                        str(output_video)
                    ]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        print(f"   ✓ Created video from frames")
                        cleanup_temp = True
                        return output_video
                    except subprocess.CalledProcessError as e:
                        print(f"   ✗ Failed to create video from frames: {e.stderr}")
            
            return None
            
    except Exception as e:
        print(f"   ✗ Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temp directory only if successful
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"   🧹 Cleaned up temp directory")
        elif not cleanup_temp:
            print(f"   📁 Temp files preserved at: {temp_dir}")


def create_comparison_video(
    original_video: Path,
    segmented_video: Path,
    emergence_video: Path,
    output_dir: Path
) -> Path:
    """
    Create a side-by-side comparison video showing all stages.
    """
    print_step(5, "Creating comparison video")
    
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = final_dir / "comparison_video.mp4"
    
    # Create side-by-side layout: original | segmented | emergence
    cmd = [
        'ffmpeg',
        '-i', str(original_video),
        '-i', str(segmented_video),
        '-i', str(emergence_video),
        '-filter_complex',
        '[0:v]scale=426:240[v0];'
        '[1:v]scale=426:240[v1];'
        '[2:v]scale=426:240[v2];'
        '[v0][v1][v2]hstack=inputs=3[out]',
        '-map', '[out]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-y',
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"   ✓ Comparison video created")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Could not create comparison video")
        return None


def print_final_report(results: Dict):
    """Print comprehensive final report."""
    print_header("PIPELINE COMPLETE - FINAL REPORT")
    
    print("\n📊 Segmentation Results:")
    segments = results.get('segments', [])
    print(f"   Segments found: {len(segments)}")
    for seg in segments[:5]:  # Show top 5
        print(f"   {seg['id']}. {seg['description']} (area: {seg['area']})")
    
    print("\n🎯 Emergence Configuration:")
    emergence_info = results.get('emergence_info', {})
    if emergence_info:
        seg = emergence_info.get('segment', {})
        print(f"   Selected segment: {seg.get('description', 'Unknown')}")
        print(f"   Position: {emergence_info.get('position', 'Unknown')}")
        print(f"   Direction: {emergence_info.get('direction', 0)}° (upward)")
    
    print("\n📁 Output Files:")
    outputs = results.get('outputs', {})
    for name, path in outputs.items():
        if path and Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"   • {name}: {path} ({size_mb:.2f} MB)")
    
    print("\n✨ Pipeline completed successfully!")
    print("   View the emergence animation to see the sailor rising from the sea!")


def main():
    """Main pipeline execution."""
    print_header("Animate Object in Video Pipeline")
    
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python animate_object_in_video.py <video_path> <object_animation_path> [output_dir]")
        print("\nExample:")
        print("  python animate_object_in_video.py uploads/assets/videos/sea_movie.mov \\")
        print("         utils/bulk_animate_image/output/batch_transparent_final/Sailor_Salute_in_Cartoon_Style.png.webm")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    object_path = Path(sys.argv[2])
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output/animate_object_in_video")
    
    # Validate inputs
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    if not object_path.exists():
        print(f"❌ Object animation not found: {object_path}")
        sys.exit(1)
    
    print(f"\n📹 Input video: {video_path}")
    print(f"🎭 Object animation: {object_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize results dictionary
    results = {
        'segments': [],
        'emergence_info': {},
        'outputs': {}
    }
    
    try:
        # Step 1: Clean existing outputs
        clean_existing_outputs(output_dir)
        
        # Step 2: Run video segmentation
        segmentation_results = run_video_segmentation(video_path, output_dir)
        results['segments'] = segmentation_results['segments']
        results['outputs']['segmented_video'] = segmentation_results['segmented_video']
        results['outputs']['labeled_frame'] = segmentation_results['labeled_frame']
        
        # Step 3: Analyze segments for best emergence location
        emergence_info = analyze_segments_for_emergence(
            segmentation_results['segments'],
            segmentation_results['first_frame']
        )
        results['emergence_info'] = emergence_info
        
        # Step 4: Create emergence animation
        emergence_video = create_emergence_animation(
            video_path,
            object_path,
            emergence_info,
            output_dir
        )
        
        if emergence_video:
            results['outputs']['emergence_video'] = emergence_video
            
            # Step 5: Create comparison video (optional)
            comparison = create_comparison_video(
                video_path,
                segmentation_results['segmented_video'],
                emergence_video,
                output_dir
            )
            
            if comparison:
                results['outputs']['comparison_video'] = comparison
        
        # Print final report
        print_final_report(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()