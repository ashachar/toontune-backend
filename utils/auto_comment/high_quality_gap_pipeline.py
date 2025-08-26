#!/usr/bin/env python3
"""
High-Quality Gap Pipeline - Minimizes re-encoding for best quality
Works directly on original video with minimal processing passes
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydub import AudioSegment


class HighQualityGapPipeline:
    """Pipeline optimized for maximum quality with minimal re-encoding."""
    
    def __init__(self, video_path: str, debug: bool = False):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.debug = debug
        
        # Create working directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = Path(f"/tmp/hq_gap_pipeline_{self.video_name}_{timestamp}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 High-Quality Gap Pipeline")
        print(f"📹 Video: {self.video_path}")
        print(f"📁 Work dir: {self.work_dir}")
        
        # Get video info
        self._get_video_info()
        
        # Load remarks
        self.remarks_path = self.video_path.parent / self.video_name / "remarks.json"
        with open(self.remarks_path) as f:
            self.remarks = json.load(f)
        print(f"📝 Loaded {len(self.remarks)} remarks")
    
    def _get_video_info(self):
        """Get video properties."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name,bit_rate",
            "-show_entries", "format=duration",
            "-of", "json", str(self.video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        self.video_width = info["streams"][0]["width"]
        self.video_height = info["streams"][0]["height"]
        self.video_codec = info["streams"][0]["codec_name"]
        self.video_duration = float(info["format"]["duration"])
        
        # Get original bitrate for quality matching
        self.original_bitrate = info["streams"][0].get("bit_rate", "0")
        if self.original_bitrate == "0":
            # Estimate from file size
            file_size_bits = self.video_path.stat().st_size * 8
            self.original_bitrate = str(int(file_size_bits / self.video_duration))
        
        print(f"📊 Video: {self.video_width}x{self.video_height}, {self.video_duration:.1f}s")
        print(f"   Codec: {self.video_codec}, Bitrate: {int(self.original_bitrate)/1000:.0f}kbps")
    
    def run_pipeline(self) -> Path:
        """Run the high-quality pipeline."""
        print("\n" + "="*70)
        print("🚀 STARTING HIGH-QUALITY PIPELINE")
        print("="*70)
        
        # Step 1: Process only the gaps that need modification
        modified_segments = self._process_gaps_only()
        
        # Step 2: Create filter complex for single-pass processing
        output_path = self._create_final_video(modified_segments)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print(f"📹 Output: {output_path}")
        print("="*70)
        
        return output_path
    
    def _process_gaps_only(self) -> Dict:
        """Process only gap segments that need speed adjustment."""
        print("\n📊 Analyzing gaps...")
        
        modified_segments = {}
        
        # Load transcript for accurate gap timings
        transcript_path = self.video_path.parent / self.video_name / "transcript.json"
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
        
        # Process each remark
        for i, remark in enumerate(self.remarks):
            gap_start = remark["time"]
            gap_duration = gaps.get(gap_start, 0.8)
            
            # Find audio file
            audio_path = self._find_audio(remark["text"], remark.get("audio_file"))
            if not audio_path:
                continue
            
            audio_duration = self._get_audio_duration(audio_path)
            
            # Only process if we need to slow down
            if audio_duration > gap_duration:
                speed_factor = gap_duration / audio_duration
                print(f"   Gap {i+1} at {gap_start:.2f}s: Needs {speed_factor*100:.1f}% speed")
                
                # Extract and process just this gap with HIGH QUALITY
                gap_file = self._process_single_gap_hq(
                    gap_start, gap_duration, audio_path, audio_duration, i
                )
                
                modified_segments[gap_start] = {
                    "file": gap_file,
                    "start": gap_start,
                    "duration": audio_duration,  # New duration after slowing
                    "original_duration": gap_duration
                }
            else:
                # Just add audio overlay without speed change
                gap_file = self._add_audio_to_gap(
                    gap_start, gap_duration, audio_path, i
                )
                modified_segments[gap_start] = {
                    "file": gap_file,
                    "start": gap_start,
                    "duration": gap_duration,
                    "original_duration": gap_duration
                }
        
        return modified_segments
    
    def _process_single_gap_hq(self, start, duration, audio_path, audio_duration, idx):
        """Process a single gap with maximum quality."""
        output_file = self.work_dir / f"gap_{idx:02d}_hq.mp4"
        
        speed_factor = duration / audio_duration
        pts_factor = 1.0 / speed_factor
        
        # Create audio track
        audio = AudioSegment.from_file(audio_path).apply_gain(3)
        silence = AudioSegment.silent(duration=int(audio_duration * 1000))
        offset = (len(silence) - len(audio)) // 2
        final_audio = silence.overlay(audio, position=offset)
        audio_track = self.work_dir / f"audio_{idx:02d}.wav"
        final_audio.export(audio_track, format="wav")
        
        # Single-pass processing with high quality
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start),
            "-i", str(self.video_path),
            "-i", str(audio_track),
            "-t", str(duration),
            "-filter_complex", f"[0:v]setpts={pts_factor:.4f}*PTS[v]",
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "veryslow",  # Maximum quality preset
            "-crf", "15",  # Very high quality (lower = better)
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k",
            str(output_file)
        ]
        
        subprocess.run(cmd, check=True)
        return output_file
    
    def _add_audio_to_gap(self, start, duration, audio_path, idx):
        """Add audio to gap without speed change."""
        output_file = self.work_dir / f"gap_{idx:02d}_audio.mp4"
        
        # Create audio track
        audio = AudioSegment.from_file(audio_path).apply_gain(3)
        
        # Extract original audio
        original_audio = self._extract_audio_segment(start, duration)
        
        # Overlay
        offset = int((duration * 1000 - len(audio)) / 2)
        final_audio = original_audio.overlay(audio, position=offset)
        audio_track = self.work_dir / f"audio_{idx:02d}.wav"
        final_audio.export(audio_track, format="wav")
        
        # Extract video segment and add new audio - NO RE-ENCODING OF VIDEO
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start),
            "-i", str(self.video_path),
            "-i", str(audio_track),
            "-t", str(duration),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",  # NO VIDEO RE-ENCODING!
            "-c:a", "aac", "-b:a", "256k",
            str(output_file)
        ]
        
        subprocess.run(cmd, check=True)
        return output_file
    
    def _extract_audio_segment(self, start, duration):
        """Extract audio segment from original video."""
        audio_file = self.work_dir / f"temp_audio_{start}.wav"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start),
            "-i", str(self.video_path),
            "-t", str(duration),
            "-vn", "-acodec", "pcm_s16le",
            str(audio_file)
        ]
        subprocess.run(cmd, check=True)
        
        audio = AudioSegment.from_file(audio_file)
        audio_file.unlink()
        return audio
    
    def _create_final_video(self, modified_segments) -> Path:
        """Create final video with minimal re-encoding."""
        print("\n🎬 Creating final video...")
        
        output_path = self.video_path.parent / self.video_name / f"{self.video_name}_hq_comments.mp4"
        
        if not modified_segments:
            # No modifications needed
            print("   No gaps need modification")
            return self.video_path
        
        # Create segment list for concatenation
        segments = []
        current_pos = 0
        
        for gap_start in sorted(modified_segments.keys()):
            gap_info = modified_segments[gap_start]
            
            # Add segment before gap (if any)
            if current_pos < gap_start:
                segments.append({
                    "start": current_pos,
                    "end": gap_start,
                    "source": "original"
                })
            
            # Add modified gap
            segments.append({
                "start": gap_start,
                "end": gap_start + gap_info["duration"],
                "source": gap_info["file"]
            })
            
            current_pos = gap_start + gap_info["original_duration"]
        
        # Add final segment
        if current_pos < self.video_duration:
            segments.append({
                "start": current_pos,
                "end": self.video_duration,
                "source": "original"
            })
        
        # Create concat file
        concat_file = self.work_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for seg in segments:
                if seg["source"] == "original":
                    # Use original video directly
                    f.write(f"file '{self.video_path}'\n")
                    f.write(f"inpoint {seg['start']}\n")
                    f.write(f"outpoint {seg['end']}\n")
                else:
                    f.write(f"file '{seg['source']}'\n")
        
        # Single concatenation pass
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",  # Copy everything - no re-encoding!
            "-movflags", "+faststart",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback to re-encoding if copy doesn't work
            print("   Note: Using re-encode for compatibility")
            cmd[-3] = "-c:v"
            cmd[-2] = "libx264"
            cmd.insert(-2, "-preset")
            cmd.insert(-1, "veryslow")
            cmd.insert(-1, "-crf")
            cmd.insert(-1, "15")
            subprocess.run(cmd, check=True)
        
        print(f"   ✅ Output: {output_path}")
        return output_path
    
    def _find_audio(self, text, hint=None):
        """Find audio file for remark."""
        clean_text = text.lower().replace("?", "").replace(".", "").strip()
        
        paths = [
            Path(f"uploads/assets/sounds/comments_audio/{clean_text}.mp3"),
            self.video_path.parent / self.video_name / hint if hint else None,
        ]
        
        for p in paths:
            if p and p.exists():
                return p
        return None
    
    def _get_audio_duration(self, audio_path):
        """Get audio duration in seconds."""
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python high_quality_gap_pipeline.py <video_path> [--debug]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    debug = "--debug" in sys.argv
    
    pipeline = HighQualityGapPipeline(video_path, debug=debug)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()