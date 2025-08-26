#!/usr/bin/env python3
"""
Debug script: Extract gaps and embed comments WITHOUT stitching
Focuses on getting each gap segment perfect before concatenation
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from pydub import AudioSegment


def get_video_duration(video_path):
    """Get video duration."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_gap_segment(video_path, start, duration, output_path):
    """Extract a segment from video."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-accurate_seek",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return output_path


def get_audio_duration(audio_path):
    """Get audio file duration."""
    if not audio_path.exists():
        return 0
    audio = AudioSegment.from_file(audio_path)
    return len(audio) / 1000.0


def find_audio_file(remark_text, audio_file_hint=None):
    """Find the audio file for a remark."""
    clean_text = remark_text.lower().replace("?", "").replace(".", "").strip()
    
    possible_paths = [
        Path(f"uploads/assets/sounds/comments_audio/{clean_text}.mp3"),
        Path(f"uploads/assets/videos/ai_math1/{audio_file_hint}") if audio_file_hint else None,
    ]
    
    for path in possible_paths:
        if path and path.exists():
            return path
    
    print(f"⚠️  Could not find audio for '{remark_text}'")
    return None


def process_single_gap(video_path, gap_info, output_dir):
    """Process a single gap: extract, adjust speed if needed, embed comment."""
    idx = gap_info["index"]
    start = gap_info["start"]
    duration = gap_info["duration"]
    remark_text = gap_info["text"]
    audio_file_hint = gap_info.get("audio_file")
    
    print(f"\n{'='*60}")
    print(f"Gap {idx}: '{remark_text}' at {start:.2f}s")
    print(f"{'='*60}")
    
    # Step 1: Extract the gap segment
    gap_video = output_dir / f"gap_{idx:02d}_original.mp4"
    print(f"📹 Extracting gap: {start:.2f}s for {duration:.2f}s")
    extract_gap_segment(video_path, start, duration, gap_video)
    
    # Verify extraction
    actual_duration = get_video_duration(gap_video)
    print(f"   Extracted duration: {actual_duration:.3f}s")
    
    # Step 2: Find and analyze the audio
    audio_path = find_audio_file(remark_text, audio_file_hint)
    if not audio_path:
        print(f"❌ No audio found for '{remark_text}'")
        return
    
    audio_duration = get_audio_duration(audio_path)
    print(f"🎵 Audio file: {audio_path.name}")
    print(f"   Audio duration: {audio_duration:.3f}s")
    
    # Step 3: Determine if we need to slow down
    needs_slowdown = audio_duration > duration
    if needs_slowdown:
        speed_factor = duration / audio_duration
        output_duration = duration / speed_factor
        print(f"🐢 Need slowdown: {speed_factor*100:.1f}% speed")
        print(f"   Video will be: {duration:.3f}s → {output_duration:.3f}s")
    else:
        speed_factor = 1.0
        output_duration = duration
        print(f"✅ No slowdown needed (audio fits in gap)")
    
    # Step 4: Create the final gap with comment
    final_gap = output_dir / f"gap_{idx:02d}_with_{remark_text.replace(' ', '_').replace('?', '').replace('.', '')}.mp4"
    
    if needs_slowdown:
        # Create slowed video
        print(f"\n🔧 Creating slowed video...")
        pts_factor = 1.0 / speed_factor
        
        # First create slowed video without audio
        temp_slowed = output_dir / f"gap_{idx:02d}_slowed_temp.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", str(gap_video),
            "-filter:v", f"setpts={pts_factor:.4f}*PTS",
            "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            str(temp_slowed)
        ]
        subprocess.run(cmd, check=True)
        
        slowed_duration = get_video_duration(temp_slowed)
        print(f"   Slowed video duration: {slowed_duration:.3f}s")
        
        # Create audio track
        print(f"\n🎤 Creating audio track...")
        audio_track = create_audio_track(gap_video, audio_path, output_duration, output_dir, idx)
        
        # Combine
        print(f"\n🎬 Combining video and audio...")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", str(temp_slowed),
            "-i", str(audio_track),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(final_gap)
        ]
        subprocess.run(cmd, check=True)
        
        # Cleanup
        temp_slowed.unlink()
        
    else:
        # Normal speed - just replace audio
        print(f"\n🎤 Creating audio track with comment...")
        audio_track = create_audio_track(gap_video, audio_path, duration, output_dir, idx)
        
        print(f"\n🎬 Combining video and audio...")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-i", str(gap_video),
            "-i", str(audio_track),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            str(final_gap)
        ]
        subprocess.run(cmd, check=True)
    
    # Verify final output
    final_duration = get_video_duration(final_gap)
    print(f"\n✅ Final gap created: {final_gap.name}")
    print(f"   Duration: {final_duration:.3f}s")
    
    # Play the audio to verify
    print(f"\n🔊 Testing audio playback...")
    subprocess.run([
        "ffplay", "-autoexit", "-nodisp",
        "-ss", "0", "-t", str(audio_duration + 0.5),
        str(final_gap)
    ], capture_output=True)
    
    return final_gap


def create_audio_track(gap_video, remark_audio_path, duration, output_dir, idx):
    """Create audio track with comment overlaid."""
    audio_file = output_dir / f"audio_track_{idx:02d}.wav"
    
    # Extract original audio from gap
    original_audio_file = output_dir / f"original_audio_{idx:02d}.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(gap_video),
        "-vn", "-acodec", "pcm_s16le",
        str(original_audio_file)
    ]
    subprocess.run(cmd, check=True)
    
    # Load audio segments
    if original_audio_file.exists() and original_audio_file.stat().st_size > 0:
        gap_audio = AudioSegment.from_file(original_audio_file)
    else:
        gap_audio = AudioSegment.silent(duration=int(duration * 1000))
    
    # Load remark
    remark_audio = AudioSegment.from_file(remark_audio_path)
    remark_duration_ms = len(remark_audio)
    
    # Boost remark volume
    remark_audio = remark_audio.apply_gain(3)
    
    # Ensure gap audio is correct length
    target_duration_ms = int(duration * 1000)
    if len(gap_audio) < target_duration_ms:
        gap_audio = gap_audio + AudioSegment.silent(duration=target_duration_ms - len(gap_audio))
    elif len(gap_audio) > target_duration_ms:
        gap_audio = gap_audio[:target_duration_ms]
    
    # Center the remark
    offset_ms = max(0, (len(gap_audio) - remark_duration_ms) // 2)
    
    print(f"   Gap audio: {len(gap_audio)}ms")
    print(f"   Remark: {remark_duration_ms}ms at offset {offset_ms}ms")
    
    # Overlay
    final_audio = gap_audio.overlay(remark_audio, position=offset_ms)
    
    # Export
    final_audio.export(audio_file, format="wav")
    
    # Cleanup
    if original_audio_file.exists():
        original_audio_file.unlink()
    
    return audio_file


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_gaps_only.py <video_path>")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    video_name = video_path.stem
    
    # Load remarks
    remarks_path = video_path.parent / video_name / "remarks.json"
    if not remarks_path.exists():
        print(f"Error: {remarks_path} not found")
        sys.exit(1)
    
    with open(remarks_path) as f:
        remarks = json.load(f)
    
    # Load transcript to get accurate gap durations
    transcript_path = video_path.parent / video_name / "transcript.json"
    gaps = {}
    if transcript_path.exists():
        with open(transcript_path) as f:
            transcript = json.load(f)
            segments = transcript.get("segments", [])
            
            for i in range(len(segments) - 1):
                gap_start = segments[i]["end"]
                gap_end = segments[i + 1]["start"]
                gap_duration = gap_end - gap_start
                if gap_duration > 0.3:
                    gaps[gap_start] = gap_duration
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/tmp/gaps_debug_{video_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎬 Debug Gaps Pipeline")
    print(f"📹 Video: {video_path}")
    print(f"📝 Remarks: {len(remarks)}")
    print(f"📁 Output: {output_dir}")
    print(f"\n{'='*60}\n")
    
    # Process each gap
    processed_gaps = []
    for i, remark in enumerate(remarks):
        gap_info = {
            "index": i + 1,
            "start": remark["time"],
            "duration": gaps.get(remark["time"], 0.8),  # Default 0.8s
            "text": remark["text"],
            "audio_file": remark.get("audio_file")
        }
        
        result = process_single_gap(video_path, gap_info, output_dir)
        if result:
            processed_gaps.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Processed {len(processed_gaps)} gaps")
    print(f"📁 All gaps saved in: {output_dir}")
    print(f"\nGap files:")
    for gap in sorted(output_dir.glob("gap_*_with_*.mp4")):
        duration = get_video_duration(gap)
        print(f"  - {gap.name}: {duration:.3f}s")
    
    # Special check for Recursive
    print(f"\n🔍 Special check for 'Recursive':")
    recursive_gaps = list(output_dir.glob("*Recursive*.mp4"))
    if recursive_gaps:
        for gap in recursive_gaps:
            print(f"\n  Testing: {gap.name}")
            duration = get_video_duration(gap)
            print(f"  Duration: {duration:.3f}s")
            
            # Extract just the audio to check
            audio_check = output_dir / "recursive_audio_check.wav"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(gap),
                "-vn", "-acodec", "pcm_s16le",
                str(audio_check)
            ], check=True)
            
            audio = AudioSegment.from_file(audio_check)
            print(f"  Audio length: {len(audio)/1000:.3f}s")
            
            # Play just the audio
            print(f"  Playing audio only...")
            subprocess.run(["ffplay", "-autoexit", "-nodisp", str(audio_check)], 
                         capture_output=True)


if __name__ == "__main__":
    main()