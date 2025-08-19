#!/usr/bin/env python3
"""
Apply all video effects to do_re_mi.mov and create a master showcase video.
Each effect segment includes a title card showing which effect is being demonstrated.
"""

import sys
import cv2
import numpy as np
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent))

from color_effects import apply_color_splash, apply_selective_color
from text_effects import apply_text_behind_subject, apply_motion_tracking_text, apply_animated_subtitle
from motion_effects import apply_floating_effect, apply_smooth_zoom, apply_3d_photo_effect, apply_rotation_effect
from layout_effects import apply_highlight_focus, add_progress_bar, apply_video_in_text


def create_title_card(text: str, duration: float = 2.0, size: Tuple[int, int] = (1280, 720)) -> Path:
    """Create a title card video with animated text."""
    width, height = size
    fps = 30
    frames = int(duration * fps)
    
    temp_path = Path(tempfile.mktemp(suffix='.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    
    for i in range(frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Animated gradient
        for y in range(height):
            progress = (i / frames)
            color_shift = int(50 * np.sin(2 * np.pi * progress))
            base_color = int(255 * (y / height))
            frame[y, :] = [
                min(255, base_color // 3 + color_shift),
                min(255, base_color // 2),
                min(255, base_color // 2 + color_shift)
            ]
        
        # Add vignette effect
        center_x, center_y = width // 2, height // 2
        for y in range(height):
            for x in range(0, width, 10):  # Sample every 10 pixels for speed
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                vignette = 1.0 - (dist / max_dist) * 0.5
                frame[y, x:min(x+10, width)] = (frame[y, x:min(x+10, width)] * vignette).astype(np.uint8)
        
        # Fade in/out animation
        if i < 15:
            alpha = i / 15
        elif i > frames - 15:
            alpha = (frames - i) / 15
        else:
            alpha = 1.0
        
        # Add title text with animation
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 3
        
        # Get text size
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        # Subtle bounce animation
        bounce = int(10 * np.sin(4 * np.pi * i / frames))
        text_y += bounce
        
        # Draw shadow
        shadow_color = (0, 0, 0)
        cv2.putText(frame, text, (text_x + 3, text_y + 3), font, font_scale, 
                   shadow_color, thickness + 2, cv2.LINE_AA)
        
        # Draw main text
        text_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                   text_color, thickness, cv2.LINE_AA)
        
        # Add subtitle
        subtitle = "Video Effect Demonstration"
        sub_scale = 0.8
        sub_thickness = 2
        sub_size, _ = cv2.getTextSize(subtitle, font, sub_scale, sub_thickness)
        sub_x = (width - sub_size[0]) // 2
        sub_y = text_y + 60
        
        cv2.putText(frame, subtitle, (sub_x, sub_y), font, sub_scale,
                   (int(200 * alpha), int(200 * alpha), int(200 * alpha)), 
                   sub_thickness, cv2.LINE_AA)
        
        out.write(frame)
    
    out.release()
    return temp_path


def trim_video(input_path: Path, duration: float = 3.0) -> Path:
    """Trim video to specified duration."""
    output_path = Path(tempfile.mktemp(suffix='.mp4'))
    
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-t', str(duration),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-y',
        str(output_path)
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def resize_video(input_path: Path, width: int = 1280, height: int = 720) -> Path:
    """Resize video to standard dimensions."""
    output_path = Path(tempfile.mktemp(suffix='.mp4'))
    
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',
        '-y',
        str(output_path)
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def apply_effects_to_video(input_video: Path, output_dir: Path) -> List[dict]:
    """Apply all effects to the input video and return list of processed videos."""
    
    effects = []
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("APPLYING EFFECTS TO DO_RE_MI.MOV")
    print("="*60)
    
    # 1. COLOR EFFECTS
    print("\n[COLOR EFFECTS]")
    
    # Color Splash (keep warm colors)
    print("1. Applying Color Splash...")
    try:
        output = output_dir / "01_color_splash.mp4"
        apply_color_splash(
            input_video,
            target_color=(255, 200, 100),  # Warm yellow/orange
            tolerance=50,
            output_path=output
        )
        effects.append({'path': output, 'title': 'Color Splash Effect', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Selective Color
    print("2. Applying Selective Color...")
    try:
        output = output_dir / "02_selective_color.mp4"
        apply_selective_color(
            input_video,
            color_adjustments=[
                {
                    'target': (200, 150, 100),
                    'tolerance': 40,
                    'hue_shift': 30,
                    'saturation': 1.5,
                    'brightness': 1.2
                }
            ],
            output_path=output
        )
        effects.append({'path': output, 'title': 'Selective Color Adjustment', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 2. TEXT EFFECTS
    print("\n[TEXT EFFECTS]")
    
    # Text Behind Subject
    print("3. Applying Text Behind Subject...")
    try:
        output = output_dir / "03_text_behind.mp4"
        apply_text_behind_subject(
            input_video,
            text="DO RE MI",
            font_scale=3.0,
            font_color=(255, 255, 0),
            font_thickness=5,
            output_path=output
        )
        effects.append({'path': output, 'title': 'Text Behind Subject', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Motion Tracking Text
    print("4. Applying Motion Tracking Text...")
    try:
        output = output_dir / "04_motion_text.mp4"
        apply_motion_tracking_text(
            input_video,
            text="TRACKING",
            track_point=(640, 360),
            font_scale=1.5,
            font_color=(0, 255, 255),
            background_color=(0, 0, 0),
            output_path=output
        )
        effects.append({'path': output, 'title': 'Motion Tracking Text', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Animated Subtitles
    print("5. Applying Animated Subtitles...")
    try:
        output = output_dir / "05_animated_subtitles.mp4"
        apply_animated_subtitle(
            input_video,
            subtitles=[
                {'text': 'Do, a deer, a female deer', 'start_time': 0.0, 'end_time': 2.0},
                {'text': 'Re, a drop of golden sun', 'start_time': 2.0, 'end_time': 4.0},
                {'text': 'Mi, a name I call myself', 'start_time': 4.0, 'end_time': 6.0},
            ],
            animation_type='fade',
            font_scale=1.5,
            output_path=output
        )
        effects.append({'path': output, 'title': 'Animated Subtitles', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 3. MOTION EFFECTS
    print("\n[MOTION EFFECTS]")
    
    # Floating Effect
    print("6. Applying Floating Effect...")
    try:
        output = output_dir / "06_floating.mp4"
        apply_floating_effect(
            input_video,
            amplitude=25,
            frequency=0.4,
            direction="circular",
            output_path=output
        )
        effects.append({'path': output, 'title': 'Floating Effect', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Smooth Zoom
    print("7. Applying Smooth Zoom...")
    try:
        output = output_dir / "07_smooth_zoom.mp4"
        apply_smooth_zoom(
            input_video,
            zoom_factor=1.8,
            zoom_type="in_out",
            easing="ease_in_out",
            hold_frames=30,
            output_path=output
        )
        effects.append({'path': output, 'title': 'Smooth Zoom', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 3D Photo Effect
    print("8. Applying 3D Photo Effect...")
    try:
        output = output_dir / "08_3d_photo.mp4"
        apply_3d_photo_effect(
            input_video,
            parallax_strength=40,
            movement_type="circular",
            use_depth_estimation=False,  # Use fallback for speed
            output_path=output
        )
        effects.append({'path': output, 'title': '3D Photo Effect', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Rotation Effect
    print("9. Applying Rotation Effect...")
    try:
        output = output_dir / "09_rotation.mp4"
        apply_rotation_effect(
            input_video,
            rotation_speed=60,
            rotation_axis="z",
            output_path=output
        )
        effects.append({'path': output, 'title': 'Rotation Effect', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # 4. LAYOUT EFFECTS
    print("\n[LAYOUT EFFECTS]")
    
    # Highlight Focus
    print("10. Applying Highlight Focus...")
    try:
        output = output_dir / "10_highlight_focus.mp4"
        apply_highlight_focus(
            input_video,
            blur_strength=31,
            vignette=True,
            output_path=output
        )
        effects.append({'path': output, 'title': 'Highlight Focus (Portrait Mode)', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Progress Bar
    print("11. Applying Progress Bar...")
    try:
        output = output_dir / "11_progress_bar.mp4"
        add_progress_bar(
            input_video,
            bar_height=10,
            bar_color=(0, 255, 0),
            style="glow",
            position="bottom",
            output_path=output
        )
        effects.append({'path': output, 'title': 'Progress Bar', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Video in Text
    print("12. Applying Video in Text...")
    try:
        output = output_dir / "12_video_in_text.mp4"
        apply_video_in_text(
            input_video,
            text="MUSIC",
            font_scale=8.0,
            font_thickness=40,
            output_path=output
        )
        effects.append({'path': output, 'title': 'Video in Text', 'duration': 3.0})
        print(f"   ✓ Saved to {output.name}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    return effects


def create_master_showcase(input_video: Path, effects: List[dict], output_path: Path):
    """Create a master showcase video with all effects and title cards."""
    
    print("\n" + "="*60)
    print("CREATING MASTER SHOWCASE VIDEO")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp(prefix='showcase_'))
    segments = []
    
    try:
        # Get video dimensions
        cap = cv2.VideoCapture(str(input_video))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        standard_size = (1280, 720)
        
        # Add opening title
        print("\nCreating title cards and segments...")
        opening_title = create_title_card("DO RE MI - Video Effects Showcase", 3.0, standard_size)
        segments.append(str(opening_title))
        
        # Add original video segment
        print("Processing original video...")
        original_trimmed = trim_video(input_video, 3.0)
        original_resized = resize_video(original_trimmed, *standard_size)
        original_title = create_title_card("Original Video", 1.5, standard_size)
        segments.append(str(original_title))
        segments.append(str(original_resized))
        
        # Process each effect
        for i, effect in enumerate(effects, 1):
            print(f"Processing effect {i}/{len(effects)}: {effect['title']}...")
            
            # Create title card for effect
            title_card = create_title_card(effect['title'], 1.5, standard_size)
            segments.append(str(title_card))
            
            # Trim and resize effect video
            if effect['path'].exists():
                trimmed = trim_video(effect['path'], effect['duration'])
                resized = resize_video(trimmed, *standard_size)
                segments.append(str(resized))
        
        # Add closing title
        closing_title = create_title_card("Thank You!", 2.0, standard_size)
        segments.append(str(closing_title))
        
        # Create concat list
        print("\nConcatenating all segments...")
        list_file = temp_dir / "segments.txt"
        with open(list_file, 'w') as f:
            for segment in segments:
                f.write(f"file '{segment}'\n")
        
        # Concatenate all segments
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-y',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
        else:
            print(f"\n✓ Master showcase created: {output_path}")
            
            # Get final video info
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(output_path)
            ]
            duration = subprocess.run(probe_cmd, capture_output=True, text=True).stdout.strip()
            
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"  Duration: {float(duration):.1f} seconds")
            print(f"  File size: {file_size:.1f} MB")
            print(f"  Resolution: {standard_size[0]}x{standard_size[1]}")
        
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    """Main function to process do_re_mi.mov with all effects."""
    
    # Setup paths
    input_video = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi.mov")
    output_dir = Path("do_re_mi_effects")
    
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)
    
    print("="*60)
    print("VIDEO EFFECTS PROCESSOR")
    print("="*60)
    print(f"\nInput: {input_video}")
    print(f"Output directory: {output_dir}")
    
    # Apply all effects
    effects = apply_effects_to_video(input_video, output_dir)
    
    # Create master showcase
    showcase_path = output_dir / "do_re_mi_showcase_master.mp4"
    create_master_showcase(input_video, effects, showcase_path)
    
    # Final summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\n✅ Successfully applied {len(effects)} effects")
    print(f"📁 Individual effects saved in: {output_dir}/")
    print(f"🎬 Master showcase video: {showcase_path}")
    print("\nTo play the showcase:")
    print(f"  open {showcase_path}")
    

if __name__ == "__main__":
    main()