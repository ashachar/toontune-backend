"""Video utilities for concatenation and combining."""

import subprocess
from pathlib import Path
from typing import List, Optional

from utils.auto_comment.pipeline.config import DEFAULT_VIDEO_QUALITY, DEFAULT_AUDIO_QUALITY, DEFAULT_FPS


def concatenate_segments(segments: List[str], temp_dir: Path) -> Optional[Path]:
    """Concatenate video segments."""
    concat_file = temp_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for seg in segments:
            f.write(f"file '{Path(seg).absolute()}'\n")
    
    print(f"  🔗 Concatenating {len(segments)} segments...")
    
    video_adjusted = temp_dir / "video_adjusted.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", DEFAULT_VIDEO_QUALITY, "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(video_adjusted)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and video_adjusted.exists():
        return video_adjusted
    else:
        print(f"  ❌ Concatenation failed")
        return None


def combine_with_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Combine video with audio track."""
    print("  🎵 Adding audio track...")
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-c:a", DEFAULT_AUDIO_QUALITY, "-b:a", "256k",
        "-movflags", "+faststart",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if output_path.exists():
        print(f"  ✅ Created video with speed adjustments: {output_path}")
        return True
    else:
        print(f"  ❌ Failed to create final video")
        return False


def create_normal_segment(
    video_path: Path, start: float, end: float, idx: int, temp_dir: Path
) -> Optional[str]:
    """Create normal speed video segment."""
    segment_file = temp_dir / f"seg_{idx:03d}_normal.mp4"
    duration = end - start
    
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", str(video_path),
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", DEFAULT_VIDEO_QUALITY, "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        str(segment_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return str(segment_file) if result.returncode == 0 and segment_file.exists() else None


def create_gap_segment(
    video_path: Path, start: float, duration: float, speed_factor: float, 
    idx: int, temp_dir: Path, fps: float = DEFAULT_FPS
) -> Optional[str]:
    """Create gap segment with speed adjustment."""
    segment_file = temp_dir / f"seg_{idx:03d}_gap.mp4"
    
    if speed_factor < 1.0:
        # Slow down video - calculate output duration
        pts_factor = 1.0 / speed_factor
        output_duration = duration / speed_factor
        print(f"     🔧 Creating slowed segment: {duration:.3f}s → {output_duration:.3f}s")
        print(f"        Input: -ss {start:.3f} -t {duration:.3f}")
        print(f"        PTS factor: {pts_factor:.3f}")
        print(f"        Output duration: {output_duration:.3f}s")
        
        # Use accurate seeking with -ss before input
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-accurate_seek",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),  # Input duration
            "-filter_complex", f"[0:v]setpts={pts_factor:.3f}*PTS[v]",
            "-map", "[v]",
            "-t", str(output_duration),  # Output duration (stretched)
            "-r", str(fps),
            "-c:v", DEFAULT_VIDEO_QUALITY, "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            str(segment_file)
        ]
    else:
        # Normal speed
        print(f"     🔧 Creating normal segment at gap position")
        print(f"        Input: -ss {start:.3f} -t {duration:.3f}")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-accurate_seek",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", DEFAULT_VIDEO_QUALITY, "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            str(segment_file)
        ]
    
    print(f"        Running FFmpeg command...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"        ❌ FFmpeg error: {result.stderr}")
        return None
    
    if not segment_file.exists():
        print(f"        ❌ Segment file not created")
        return None
    
    # Verify the segment duration
    probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(segment_file)
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_result.returncode == 0:
        actual_duration = float(probe_result.stdout.strip())
        expected_duration = output_duration if speed_factor < 1.0 else duration
        print(f"        ✓ Segment created: {actual_duration:.3f}s (expected: {expected_duration:.3f}s)")
        if abs(actual_duration - expected_duration) > 0.1:
            print(f"        ⚠️  Duration mismatch! Difference: {abs(actual_duration - expected_duration):.3f}s")
    
    return str(segment_file)