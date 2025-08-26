#!/usr/bin/env python3
"""
Precise Gap-Based Video Comment Pipeline
Splits video into sub-videos, adjusts speed per gap, embeds comments, then stitches together.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from pydub import AudioSegment


class PreciseGapPipeline:
    """Pipeline that precisely fits comments into gaps by splitting and adjusting speed."""
    
    def __init__(self, video_path: str, remarks_json: str = None, debug: bool = False):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.debug = debug
        
        # Get remarks JSON path
        if remarks_json:
            self.remarks_path = Path(remarks_json)
        else:
            # Look for remarks in standard location
            self.remarks_path = self.video_path.parent / self.video_name / "remarks.json"
        
        if not self.remarks_path.exists():
            raise FileNotFoundError(f"Remarks file not found: {self.remarks_path}")
        
        # Create working directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = Path(f"/tmp/precise_gap_pipeline_{self.video_name}_{timestamp}")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        self.segments_dir = self.work_dir / "segments"
        self.gaps_dir = self.work_dir / "gaps_with_comments"
        self.segments_dir.mkdir(exist_ok=True)
        self.gaps_dir.mkdir(exist_ok=True)
        
        print(f"🎬 Precise Gap Pipeline initialized")
        print(f"📁 Working directory: {self.work_dir}")
        print(f"   - Segments: {self.segments_dir}")
        print(f"   - Gaps with comments: {self.gaps_dir}")
        
        # Get video info
        self._get_video_info()
        
        # Load remarks
        with open(self.remarks_path) as f:
            self.remarks = json.load(f)
        print(f"📝 Loaded {len(self.remarks)} remarks")
    
    def _get_video_info(self):
        """Get video duration and properties."""
        # Get duration
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(self.video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        self.video_duration = float(info["format"]["duration"])
        
        # Get video dimensions for debug overlay
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json", str(self.video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stream_info = json.loads(result.stdout)
        self.video_width = stream_info["streams"][0]["width"]
        self.video_height = stream_info["streams"][0]["height"]
        
        print(f"📹 Video: {self.video_duration:.2f}s @ {self.video_width}x{self.video_height}")
    
    def _get_audio_duration(self, audio_file: Path) -> float:
        """Get duration of audio file in seconds."""
        if not audio_file.exists():
            return 0.0
        audio = AudioSegment.from_file(audio_file)
        return len(audio) / 1000.0
    
    def run_pipeline(self) -> Path:
        """Run the complete pipeline."""
        print("\n" + "="*70)
        print("🚀 STARTING PRECISE GAP-BASED PIPELINE")
        print("="*70)
        
        # Step 1: Analyze gaps and prepare split points
        split_points = self._analyze_split_points()
        
        # Step 2: Split video into segments
        segments = self._split_video_into_segments(split_points)
        
        # Step 3: Process each gap segment
        processed_segments = self._process_gap_segments(segments)
        
        # Step 4: Stitch all segments together
        output_path = self._stitch_segments(processed_segments)
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print(f"📹 Output: {output_path}")
        print(f"📁 All sub-videos saved in: {self.work_dir}")
        print("="*70)
        
        # Run segment mapping diagnostic
        self._run_segment_mapping_diagnostic()
        
        return output_path
    
    def _run_segment_mapping_diagnostic(self):
        """Run segment mapping diagnostic to verify 1-to-1 mapping."""
        print("\n" + "="*70)
        print("📊 SEGMENT MAPPING DIAGNOSTIC")
        print("="*70)
        
        # Import and run the diagnostic
        try:
            from .diagnose_segment_mapping import analyze_segment_mapping
            analyze_segment_mapping()
        except ImportError:
            # If module not available, run inline diagnostic
            self._inline_segment_diagnostic()
    
    def _inline_segment_diagnostic(self):
        """Inline diagnostic if separate module not available."""
        print("\n🔍 Verifying segment continuity...")
        
        # Check segments for continuity
        segments_info = []
        last_end = 0.0
        issues = []
        
        # Read segment info from work dir if available
        concat_file = self.work_dir / "concat_list.txt"
        if concat_file.exists():
            with open(concat_file) as f:
                lines = f.readlines()
            print(f"   Total segments in final video: {len(lines)}")
            
        # Verify no timestamp overlaps in split points
        if hasattr(self, '_last_split_points'):
            for i in range(1, len(self._last_split_points)):
                prev = self._last_split_points[i-1]
                curr = self._last_split_points[i]
                
                if curr["start"] < prev["end"]:
                    issues.append(f"Overlap at segment {i}: {curr['start']:.2f} < {prev['end']:.2f}")
                elif curr["start"] > prev["end"]:
                    issues.append(f"Gap at segment {i}: {curr['start']:.2f} > {prev['end']:.2f}")
        
        if issues:
            print("❌ Issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ All segments are continuous with no overlaps")
        
        print("-" * 70)
    
    def _analyze_split_points(self) -> List[Dict]:
        """Analyze remarks and create split points for video."""
        print("\n📊 Analyzing split points...")
        
        # Try to load Scribe gaps first (more precise)
        scribe_gaps_path = self.video_path.parent / self.video_name / "gaps_analysis_scribe.json"
        gaps_list = []
        
        if scribe_gaps_path.exists():
            print("   📝 Using ElevenLabs Scribe word-level gaps")
            with open(scribe_gaps_path) as f:
                scribe_data = json.load(f)
                gaps_list = scribe_data.get("gaps", [])
                print(f"   Found {len(gaps_list)} natural gaps from Scribe")
        else:
            # Fallback to transcript gaps
            transcript_path = self.video_path.parent / self.video_name / "transcript.json"
            if transcript_path.exists():
                print("   Using phrase-level transcript gaps")
                with open(transcript_path) as f:
                    transcript = json.load(f)
                    segments = transcript.get("segments", [])
                    
                    for i in range(len(segments) - 1):
                        gap_start = segments[i]["end"]
                        gap_end = segments[i + 1]["start"]
                        gap_duration = gap_end - gap_start
                        if gap_duration > 0.3:  # Minimum gap threshold
                            gaps_list.append({
                                "start": gap_start,
                                "end": gap_end,
                                "duration": gap_duration
                            })
        
        split_points = []
        current_pos = 0
        
        # Sort gaps by start time for chronological assignment
        gaps_list_sorted = sorted(gaps_list, key=lambda x: x["start"])
        
        # Use gaps in chronological order for remarks
        for i, remark in enumerate(self.remarks):
            if i >= len(gaps_list_sorted):
                print(f"   ⚠️ No more gaps for remark {i+1}")
                break
            
            gap = gaps_list_sorted[i]  # Use gaps in chronological order
            remark_time = gap["start"]
            remark_text = remark["text"]
            audio_file = remark.get("audio_file")
            
            # Get gap duration from the actual gap
            gap_duration = gap["duration"]
            
            # Get audio duration
            if audio_file:
                # Try multiple possible locations for the audio file
                # Clean text for filename lookup
                clean_text = remark_text.lower().replace("?", "").replace(".", "").replace(" ", "_")
                clean_text = ''.join(c for c in clean_text if c.isalnum() or c == '_')[:50]
                
                possible_paths = [
                    self.video_path.parent / self.video_name / audio_file,
                    self.video_path.parent / self.video_name / f"remark_{i+1}.mp3",
                    Path("uploads/assets/sounds/comments_audio") / f"{clean_text}.mp3",
                    Path("uploads/assets/sounds/comments_audio") / f"{remark_text.lower().replace('?', '').replace('.', '')}.mp3"
                ]
                
                audio_path = None
                for p in possible_paths:
                    if p.exists():
                        audio_path = p
                        break
                
                if audio_path:
                    audio_duration = self._get_audio_duration(audio_path)
                    print(f"      Found audio: {audio_path.name} ({audio_duration:.3f}s)")
                else:
                    # Estimate based on text length
                    word_count = len(remark_text.split())
                    audio_duration = word_count * 0.45 + 0.3
                    print(f"      Estimated duration for '{remark_text}': {audio_duration:.3f}s")
            else:
                audio_duration = 0.5  # Default
            
            # Add segment before gap
            if current_pos < remark_time:
                split_points.append({
                    "type": "normal",
                    "start": current_pos,
                    "end": remark_time,
                    "duration": remark_time - current_pos,
                    "index": len(split_points)
                })
            
            # Add gap segment with remark info
            # Use actual gap end time from Scribe data
            gap_segment_duration = gap_duration
            
            split_points.append({
                "type": "gap",
                "start": remark_time,
                "end": gap.get("end", remark_time + gap_segment_duration),
                "duration": gap_segment_duration,
                "remark": remark,
                "remark_text": remark_text,
                "remark_duration": audio_duration,
                "audio_file": audio_file,
                "speed_factor": min(1.0, gap_segment_duration / audio_duration) if audio_duration > 0 else 1.0,
                "index": len(split_points),
                "context": gap.get("context_before", "") + " | " + gap.get("context_after", "")
            })
            
            current_pos = gap.get("end", remark_time + gap_segment_duration)
        
        # Sort all gap segments by start time
        gap_segments = [sp for sp in split_points if sp["type"] == "gap"]
        gap_segments.sort(key=lambda x: x["start"])
        
        # Rebuild split_points with normal segments between gaps
        final_split_points = []
        current_pos = 0
        
        for gap in gap_segments:
            # Add normal segment before this gap
            if current_pos < gap["start"]:
                final_split_points.append({
                    "type": "normal",
                    "start": current_pos,
                    "end": gap["start"],
                    "duration": gap["start"] - current_pos,
                    "index": len(final_split_points)
                })
            
            # Add the gap segment
            gap["index"] = len(final_split_points)
            final_split_points.append(gap)
            current_pos = gap["end"]
        
        # Add final normal segment
        if current_pos < self.video_duration:
            final_split_points.append({
                "type": "normal",
                "start": current_pos,
                "end": self.video_duration,
                "duration": self.video_duration - current_pos,
                "index": len(final_split_points)
            })
        
        split_points = final_split_points
        
        # Save for diagnostic
        self._last_split_points = split_points
        
        print(f"📍 Created {len(split_points)} split points")
        for sp in split_points[:5]:  # Show first 5
            if sp["type"] == "gap":
                print(f"   Gap at {sp['start']:.2f}s: '{sp['remark_text']}' "
                      f"(gap: {sp['duration']:.2f}s, remark: {sp['remark_duration']:.2f}s, "
                      f"speed: {sp['speed_factor']*100:.0f}%)")
            else:
                print(f"   Normal: {sp['start']:.2f}s - {sp['end']:.2f}s ({sp['duration']:.2f}s)")
        
        return split_points
    
    def _verify_no_overlaps(self, split_points: List[Dict]) -> bool:
        """Verify that segments don't overlap to prevent content duplication."""
        issues = []
        for i in range(1, len(split_points)):
            prev = split_points[i-1]
            curr = split_points[i]
            
            if curr["start"] < prev["end"]:
                issues.append(f"❌ Overlap: Segment {i} starts at {curr['start']:.2f}s "
                            f"but segment {i-1} ends at {prev['end']:.2f}s")
            elif curr["start"] > prev["end"] + 0.1:  # Allow tiny gaps
                gap_size = curr["start"] - prev["end"]
                if gap_size > 0.5:  # Only warn for significant gaps
                    issues.append(f"⚠️ Gap: {gap_size:.2f}s between segments {i-1} and {i}")
        
        if issues:
            print("\n🔍 Segment continuity check:")
            for issue in issues:
                print(f"   {issue}")
            if any("❌" in i for i in issues):
                return False
        return True
    
    def _split_video_into_segments(self, split_points: List[Dict]) -> List[Dict]:
        """Split video into individual segment files."""
        print(f"\n✂️ Splitting video into {len(split_points)} segments...")
        
        # Verify no overlaps before extraction
        if not self._verify_no_overlaps(split_points):
            print("⚠️ WARNING: Segment overlaps detected! This may cause content repetition.")
        
        segments = []
        
        for i, sp in enumerate(split_points):
            segment_file = self.segments_dir / f"segment_{i:03d}_{sp['type']}.mp4"
            
            # Extract segment with precise duration
            # CRITICAL: Always re-encode to ensure precise cuts at exact timestamps
            # Using -c:v copy can cause segments to start at keyframes before the requested time
            # This leads to overlapping content between segments
            
            # Always use re-encoding for precise cuts
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(sp["start"]),  # Input seeking (before -i)
                "-i", str(self.video_path),
                "-t", str(sp["duration"]),  # Duration from new position
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",  # Good quality, faster encoding
                "-c:a", "aac", "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                str(segment_file)
            ]
            
            print(f"   [{i+1}/{len(split_points)}] Extracting {sp['type']} segment: "
                  f"{sp['start']:.2f}s - {sp['end']:.2f}s")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"      ⚠️ Warning: {result.stderr}")
            
            segment_info = sp.copy()
            segment_info["file"] = segment_file
            segments.append(segment_info)
        
        print(f"✅ Created {len(segments)} segment files")
        return segments
    
    def _process_gap_segments(self, segments: List[Dict]) -> List[Dict]:
        """Process gap segments: adjust speed and embed comments."""
        print(f"\n🎯 Processing gap segments with comments...")
        
        processed = []
        
        for seg in segments:
            if seg["type"] == "normal":
                # Normal segments pass through unchanged
                processed.append(seg)
            else:
                # Gap segment - adjust speed and add comment
                processed_file = self._process_single_gap(seg)
                seg["processed_file"] = processed_file
                processed.append(seg)
        
        return processed
    
    def _process_single_gap(self, gap_seg: Dict) -> Path:
        """Process a single gap segment: adjust speed and embed comment."""
        idx = gap_seg["index"]
        input_file = gap_seg["file"]
        output_file = self.gaps_dir / f"gap_{idx:03d}_with_comment.mp4"
        
        speed_factor = gap_seg["speed_factor"]
        remark_duration = gap_seg["remark_duration"]
        gap_duration = gap_seg["duration"]
        
        print(f"\n   🔧 Processing gap {idx}: '{gap_seg['remark_text']}'")
        print(f"      Gap: {gap_duration:.3f}s, Remark: {remark_duration:.3f}s")
        print(f"      Speed: {speed_factor*100:.1f}%")
        
        if speed_factor < 1.0:
            # Need to slow down video
            pts_factor = 1.0 / speed_factor
            output_duration = gap_duration / speed_factor
            
            print(f"      Slowing video: {gap_duration:.3f}s → {output_duration:.3f}s")
            
            # Step 1: Create slowed video without audio
            temp_video = self.work_dir / f"temp_slowed_{idx}.mp4"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(input_file),
                "-filter:v", f"setpts={pts_factor:.4f}*PTS",
                "-an",  # No audio for now
                "-c:v", "libx264", "-preset", "veryslow", "-crf", "10",  # MAXIMUM quality!
                "-pix_fmt", "yuv420p",
                str(temp_video)
            ]
            subprocess.run(cmd, check=True)
            
            # Step 2: Create audio track with comment
            audio_track = self._create_gap_audio(gap_seg, output_duration)
            
            # Step 3: Combine slowed video with audio
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(temp_video),
                "-i", str(audio_track),
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "320k",  # Max audio quality
                # Don't use -shortest to avoid cutting off audio
                str(output_file)
            ]
            subprocess.run(cmd, check=True)
            
            # Cleanup
            temp_video.unlink()
            
        else:
            # Normal speed - just add comment to existing audio
            audio_track = self._create_gap_audio(gap_seg, gap_duration)
            
            # Replace audio in segment
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(input_file),
                "-i", str(audio_track),
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "320k",  # Max audio quality
                # Don't use -shortest to avoid cutting off audio
                str(output_file)
            ]
            subprocess.run(cmd, check=True)
        
        print(f"      ✅ Saved: {output_file.name}")
        return output_file
    
    def _create_gap_audio(self, gap_seg: Dict, duration_ms: float) -> Path:
        """Create audio track for gap with comment overlaid."""
        idx = gap_seg["index"]
        audio_file = self.work_dir / f"gap_audio_{idx:03d}.wav"
        
        # Extract original audio from gap segment
        original_audio_file = self.work_dir / f"original_gap_audio_{idx:03d}.wav"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(gap_seg["file"]),
            "-vn", "-acodec", "pcm_s16le",
            str(original_audio_file)
        ]
        subprocess.run(cmd, check=True)
        
        # Load original audio
        if original_audio_file.exists() and original_audio_file.stat().st_size > 0:
            gap_audio = AudioSegment.from_file(original_audio_file)
        else:
            # Create silence if no audio
            gap_audio = AudioSegment.silent(duration=int(duration_ms * 1000))
        
        # Load comment audio
        if gap_seg.get("audio_file"):
            # Try multiple possible locations
            remark_text = gap_seg.get("remark_text", "")
            clean_text = remark_text.lower().replace("?", "").replace(".", "").replace(" ", "_")
            clean_text = ''.join(c for c in clean_text if c.isalnum() or c == '_')[:50]
            
            possible_paths = [
                self.video_path.parent / self.video_name / gap_seg["audio_file"],
                self.video_path.parent / self.video_name / f"remark_{gap_seg['index']//2+1}.mp3",
                Path("uploads/assets/sounds/comments_audio") / f"{clean_text}.mp3",
                Path("uploads/assets/sounds/comments_audio") / f"{remark_text.lower().replace('?', '').replace('.', '')}.mp3"
            ]
            
            remark_path = None
            for p in possible_paths:
                if p.exists():
                    remark_path = p
                    break
            
            if remark_path and remark_path.exists():
                remark_audio = AudioSegment.from_file(remark_path)
                # Boost volume
                remark_audio = remark_audio.apply_gain(3)
                
                # Center the remark in the gap
                total_duration_ms = int(duration_ms * 1000)
                remark_duration_ms = len(remark_audio)
                offset_ms = max(0, (total_duration_ms - remark_duration_ms) // 2)
                
                # Ensure gap audio is long enough
                if len(gap_audio) < total_duration_ms:
                    gap_audio = gap_audio + AudioSegment.silent(
                        duration=total_duration_ms - len(gap_audio)
                    )
                elif len(gap_audio) > total_duration_ms:
                    gap_audio = gap_audio[:total_duration_ms]
                
                # Overlay remark
                final_audio = gap_audio.overlay(remark_audio, position=offset_ms)
            else:
                final_audio = gap_audio
        else:
            final_audio = gap_audio
        
        # Export
        final_audio.export(audio_file, format="wav")
        
        # Cleanup
        if original_audio_file.exists():
            original_audio_file.unlink()
        
        return audio_file
    
    def _test_debug_overlay_fit(self) -> Tuple[int, str]:
        """Test if debug overlay fits and return optimal font size."""
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create timestamp list text
        lines = ["COMMENTS:"]
        max_line_length = 8  # "COMMENTS:"
        
        for i, remark in enumerate(self.remarks):
            time = remark["time"]
            text = remark["text"][:30]  # Truncate long comments
            line = f"{time:6.2f}s: {text}"
            lines.append(line)
            max_line_length = max(max_line_length, len(line))
        
        debug_text = "\\n".join(lines)
        num_lines = len(lines)
        
        # Try different font sizes to find what fits
        for font_size in range(16, 8, -1):
            # Estimate text dimensions
            # Approximate: each character is about 0.6x the font size in width
            text_width = max_line_length * font_size * 0.6
            text_height = num_lines * font_size * 1.2  # Line spacing
            
            # Check if it fits with margins
            margin = 40
            if text_width < (self.video_width - margin) and text_height < (self.video_height - margin):
                print(f"   📐 Debug overlay: {num_lines} lines, font size {font_size}px")
                print(f"      Estimated size: {int(text_width)}x{int(text_height)}px")
                return font_size, debug_text
        
        # If nothing fits, use smallest font and truncate
        print(f"   ⚠️ Too many comments for overlay, using minimal font")
        return 9, debug_text
    
    def _create_debug_overlay_filter(self) -> str:
        """Create FFmpeg filter for debug overlay with timestamp list."""
        if not self.debug:
            return ""
        
        # Test fit and get optimal font size
        font_size, debug_text = self._test_debug_overlay_fit()
        
        # For FFmpeg drawtext filter, we need to properly format newlines
        # Replace newlines with actual line breaks for drawtext
        # Escape special characters but keep newlines
        debug_text = debug_text.replace(":", "\\:")
        # Split into lines and rejoin with proper line breaks
        lines = debug_text.split("\\n")
        
        # Calculate margins
        x_margin = 20
        y_margin = 20
        
        # Create multiple drawtext filters, one for each line
        filters = []
        for i, line in enumerate(lines):
            # Escape single quotes in the line
            line = line.replace("'", "'\\''")
            y_offset = y_margin + (i * font_size * 1.2)  # Line spacing
            
            filter = (
                f"drawtext="
                f"text='{line}':"
                f"fontfile=/System/Library/Fonts/Helvetica.ttc:"
                f"fontsize={font_size}:"
                f"fontcolor=red:"
                f"box=1:"
                f"boxcolor=black@0.7:"
                f"boxborderw=5:"
                f"x=w-text_w-{x_margin}:"  # Right-aligned
                f"y={int(y_offset)}"  # Vertical position for this line
            )
            filters.append(filter)
        
        # Join all filters with commas
        filter_str = ",".join(filters)
        
        return filter_str
    
    def _stitch_segments(self, segments: List[Dict]) -> Path:
        """Stitch all segments together into final video."""
        print(f"\n🔗 Stitching {len(segments)} segments into final video...")
        
        # Create concat file
        concat_file = self.work_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for seg in segments:
                if seg["type"] == "normal":
                    video_file = seg["file"]
                else:
                    video_file = seg["processed_file"]
                f.write(f"file '{video_file}'\n")
        
        # Output path
        output_dir = self.video_path.parent / self.video_name
        output_dir.mkdir(exist_ok=True)
        suffix = "_debug" if self.debug else ""
        output_path = output_dir / f"{self.video_name}_precise_comments{suffix}.mp4"
        
        # Concatenate with optional debug overlay
        if self.debug:
            print("   🐞 Debug mode: Adding timestamp overlay...")
            debug_filter = self._create_debug_overlay_filter()
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-vf", debug_filter,
                "-c:v", "libx264", "-preset", "veryslow", "-crf", "10",  # MAXIMUM quality
                "-c:a", "aac", "-b:a", "320k",  # Max audio quality
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_path)
            ]
        else:
            # Without debug overlay, still re-encode to ensure smooth playback
            # All segments are already high quality from extraction
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",  # Match segment quality
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(output_path)
            ]
        
        print("   Concatenating segments...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ⚠️ Error: {result.stderr}")
            raise RuntimeError("Failed to concatenate segments")
        
        print(f"   ✅ Final video created: {output_path}")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Total segments: {len(segments)}")
        print(f"   Normal segments: {sum(1 for s in segments if s['type'] == 'normal')}")
        print(f"   Gap segments: {sum(1 for s in segments if s['type'] == 'gap')}")
        print(f"   Working directory: {self.work_dir}")
        
        return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python precise_gap_pipeline.py <video_path> [remarks_json] [--debug]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    remarks_json = None
    debug = False
    
    # Parse additional arguments
    for arg in sys.argv[2:]:
        if arg == "--debug":
            debug = True
        elif not remarks_json:
            remarks_json = arg
    
    if debug:
        print("🐞 DEBUG MODE ENABLED - Will add timestamp overlay")
    
    pipeline = PreciseGapPipeline(video_path, remarks_json, debug=debug)
    output = pipeline.run_pipeline()
    
    # Auto-open
    try:
        subprocess.run(["open", str(output)], check=True)
    except:
        pass


if __name__ == "__main__":
    main()