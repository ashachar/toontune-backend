#!/usr/bin/env python3
"""
END-TO-END SNARK PIPELINE
1. Load any video
2. Extract transcript (if needed)
3. Analyze transcript for context
4. Generate creative snarks based on content
5. Convert to speech with ElevenLabs
6. Mix into video at silence gaps
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import requests
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.effects import normalize

# Load environment
backend_env = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/.env")
if backend_env.exists():
    load_dotenv(backend_env)

@dataclass
class SilenceGap:
    start: float
    end: float
    duration: float

@dataclass
class Snark:
    text: str
    time: float
    emotion: str
    context: str
    gap_duration: float

class EndToEndSnarkPipeline:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        
        # Create output folder next to the video
        self.output_folder = self.video_path.parent / self.video_name
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Storage for generated snarks (global library)
        self.snark_storage = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/sounds/snark_remarks")
        self.snark_storage.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
        
        # Load existing snark library
        self.existing_snarks = self._load_existing_snarks()
        
        print(f"🎬 Pipeline initialized for: {self.video_name}")
        print(f"📁 Output folder: {self.output_folder}")
        print(f"📚 Existing snark library: {len(self.existing_snarks)} available")
    
    def _load_existing_snarks(self) -> Dict[str, Dict]:
        """Load all existing snarks from storage"""
        existing = {}
        
        if not self.snark_storage.exists():
            return existing
        
        # Scan all MP3 files in snark storage
        for mp3_file in self.snark_storage.glob("*.mp3"):
            # Convert filename back to text
            text = mp3_file.stem.replace("_", " ")
            
            # Try to determine emotion from filename or default
            emotion = "sarcastic"  # Default
            
            # Store with metadata
            existing[text.lower()] = {
                "text": text,
                "file": str(mp3_file),
                "emotion": emotion,
                "duration": self._estimate_duration(text)
            }
        
        return existing
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate speaking duration for text"""
        words = len(text.split())
        # More conservative estimate: 2.2 words/sec + 0.4s padding
        return words / 2.2 + 0.4
    
    def extract_transcript(self) -> Dict:
        """Step 1: Extract transcript from video"""
        
        transcript_file = self.output_folder / "transcript.json"
        
        # Check if already exists
        if transcript_file.exists():
            print("📝 Loading existing transcript...")
            with open(transcript_file) as f:
                return json.load(f)
        
        print("🎤 Extracting transcript with Whisper...")
        
        # Extract audio first
        audio_path = self.output_folder / "audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            str(audio_path)
        ], capture_output=True)
        
        # Use Whisper API (or local whisper)
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path), word_timestamps=True)
            
            # Format transcript
            transcript = {
                "text": result["text"],
                "segments": []
            }
            
            for segment in result.get("segments", []):
                transcript["segments"].append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                })
            
        except ImportError:
            print("⚠️ Whisper not available, using mock transcript")
            # Fallback to mock
            transcript = {
                "text": "Mock transcript for testing",
                "segments": [
                    {"start": 0, "end": 5, "text": "This is a test"},
                    {"start": 5, "end": 10, "text": "Another segment"}
                ]
            }
        
        # Save transcript
        with open(transcript_file, "w") as f:
            json.dump(transcript, f, indent=2)
        
        print(f"✅ Transcript extracted: {len(transcript['segments'])} segments")
        return transcript
    
    def analyze_silence_gaps(self) -> List[SilenceGap]:
        """Step 2: Find gaps between spoken segments in transcript"""
        
        print("🔍 Analyzing gaps between spoken segments...")
        
        # Get transcript
        transcript_file = self.output_folder / "transcript.json"
        if not transcript_file.exists():
            print("⚠️ No transcript found, extracting first...")
            self.extract_transcript()
        
        with open(transcript_file) as f:
            transcript = json.load(f)
        
        segments = transcript.get("segments", [])
        if not segments:
            print("❌ No segments in transcript")
            return []
        
        # Get video duration
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.video_path)
        ], capture_output=True, text=True)
        video_duration = float(result.stdout.strip())
        
        gaps = []
        
        # Check gap at the beginning
        if segments[0]["start"] > 0.6:  # At least 0.6s gap at start
            gaps.append(SilenceGap(
                start=0.0,
                end=segments[0]["start"],
                duration=segments[0]["start"]
            ))
        
        # Find gaps between segments
        for i in range(len(segments) - 1):
            gap_start = segments[i]["end"]
            gap_end = segments[i + 1]["start"]
            gap_duration = gap_end - gap_start
            
            # Only include gaps that are long enough for a snark
            if gap_duration >= 0.6:  # 0.6 seconds minimum for short snarks
                gaps.append(SilenceGap(
                    start=gap_start,
                    end=gap_end,
                    duration=gap_duration
                ))
        
        # Check gap at the end
        last_segment_end = segments[-1]["end"]
        if video_duration - last_segment_end > 0.6:
            gaps.append(SilenceGap(
                start=last_segment_end,
                end=video_duration,
                duration=video_duration - last_segment_end
            ))
        
        print(f"✅ Found {len(gaps)} usable gaps between spoken segments")
        return gaps
    
    def generate_contextual_snarks(self, transcript: Dict, gaps: List[SilenceGap]) -> List[Snark]:
        """Step 3: Generate snarks based on transcript context"""
        
        print("🤖 Generating contextual snarks...")
        
        snarks = []
        reused_count = 0
        used_texts = set()  # Track what we've already used in this video
        
        # Identify beginning and end gaps for custom remarks
        beginning_gap = None
        end_gap = None
        
        if gaps:
            # First gap within first 20% of video
            for gap in gaps:
                if gap.start < 20:  # First 20 seconds
                    beginning_gap = gap
                    break
            
            # Last suitable gap in final 30% of video
            video_duration = gaps[-1].end if gaps else 60
            for gap in reversed(gaps):
                if gap.start > video_duration * 0.7:
                    end_gap = gap
                    break
        
        for i, gap in enumerate(gaps[:10]):  # Limit to 10 snarks
            # Find context: what was said before this gap
            context = ""
            for seg in transcript["segments"]:
                if seg["end"] <= gap.start and seg["end"] > gap.start - 3:
                    context = seg["text"]
                    break
            
            # Generate custom remarks for beginning and end
            force_custom = (gap == beginning_gap or gap == end_gap)
            
            if force_custom:
                # Generate custom hyper-funny remark for beginning/end
                position = "beginning" if gap == beginning_gap else "end"
                print(f"  🎭 Generating custom {position} remark (gap: {gap.duration:.1f}s)...")
                
                if self.gemini_key:
                    snark = self._generate_custom_funny_snark(context, gap.duration, position, transcript)
                else:
                    # Fallback custom remarks (sized to gap, playful tone)
                    if position == "beginning":
                        if gap.duration < 1.5:
                            snark = {"text": "Nice!", "emotion": "cheerful"}
                        elif gap.duration < 2.5:
                            snark = {"text": "Let's do this.", "emotion": "playful"}
                        else:
                            snark = {"text": "Buckle up everyone.", "emotion": "amused"}
                    else:
                        if gap.duration < 1.5:
                            snark = {"text": "Done!", "emotion": "cheerful"}
                        elif gap.duration < 2.5:
                            snark = {"text": "And scene.", "emotion": "playful"}
                        else:
                            snark = {"text": "That's all folks!", "emotion": "cheerful"}
                
                snarks.append(Snark(
                    text=snark["text"],
                    time=gap.start + 0.2,
                    emotion=snark["emotion"],
                    context=context,
                    gap_duration=gap.duration
                ))
                print(f"  🎯 Custom: \"{snark['text']}\"")
                used_texts.add(snark["text"].lower())
            else:
                # For middle gaps, always generate fresh friendly content
                # Skip existing library as it has mean-spirited content
                
                # Always generate new friendly snark with AI
                if self.gemini_key:
                    snark = self._generate_snark_with_ai(context, gap.duration, used_texts)
                else:
                    snark = self._get_fallback_snark(gap.duration, used_texts)
                
                # Double-check we're not repeating
                if snark and snark["text"].lower() in used_texts:
                    # Try to get a different one
                    snark = self._get_fallback_snark(gap.duration, used_texts, force_different=True)
                
                if snark:
                        snarks.append(Snark(
                            text=snark["text"],
                            time=gap.start + 0.2,
                            emotion=snark["emotion"],
                            context=context,
                            gap_duration=gap.duration
                        ))
                        used_texts.add(snark["text"].lower())
                        print(f"  ✨ New: \"{snark['text']}\" ({snark['emotion']})")
        
        print(f"✅ Total snarks: {len(snarks)} ({reused_count} reused, {len(snarks)-reused_count} new)")
        return snarks
    
    def _create_video_with_frozen_frames(self, snarks_with_pausing: List[Dict], mixed_audio_path: Path, output_path: Path):
        """Create video with frozen frames during paused remarks"""
        
        # Build complex FFmpeg filter to freeze frames
        filter_parts = []
        pts_parts = []
        current_time = 0
        
        # Sort snarks by time
        snarks_with_pausing.sort(key=lambda x: x["snark"].time)
        
        for i, item in enumerate(snarks_with_pausing):
            if not item["needs_pause"]:
                continue
                
            snark = item["snark"]
            pause_duration = item["duration"]
            pause_start = snark.time
            
            # Create filter to freeze frame at pause_start for pause_duration
            # This uses setpts to manipulate presentation timestamps
            if current_time < pause_start:
                # Normal speed until pause
                pts_parts.append(f"if(lt(T,{pause_start}),PTS-STARTPTS")
            
            # Freeze frame during pause
            pts_parts.append(f",if(lt(T,{pause_start + pause_duration}),{pause_start}/TB")
            
            # Resume after pause with time offset
            current_time = pause_start + pause_duration
        
        # Complete the filter
        if pts_parts:
            pts_filter = "".join(pts_parts) + f",PTS-STARTPTS+{pause_duration})" * len([s for s in snarks_with_pausing if s["needs_pause"]]) + ")"
            filter_complex = f"[0:v]setpts='{pts_filter}'[v]"
        else:
            filter_complex = "[0:v]copy[v]"
        
        # Use FFmpeg with complex filter
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-i", str(mixed_audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "256k",
            "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  ⚠️ Complex filter failed, using simpler approach...")
            # Fallback to simpler segment-based approach
            self._create_video_with_frozen_frames_simple(snarks_with_pausing, mixed_audio_path, output_path)
    
    def _create_video_with_frozen_frames_simple(self, snarks_with_pausing: List[Dict], mixed_audio_path: Path, output_path: Path):
        """Create video with actual frozen frames during pauses"""
        
        print("  🎬 Creating video with frozen frames...")
        
        # Get paused snarks
        paused_snarks = [s for s in snarks_with_pausing if s["needs_pause"]]
        if not paused_snarks:
            # No pauses needed, just remux
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", str(mixed_audio_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "256k",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ], capture_output=True)
            return
        
        paused_snarks.sort(key=lambda x: x["snark"].time)
        
        # Get video info
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(self.video_path)
        ], capture_output=True, text=True)
        video_info = json.loads(result.stdout)
        width = video_info["streams"][0]["width"]
        height = video_info["streams"][0]["height"]
        fps_parts = video_info["streams"][0]["r_frame_rate"].split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30
        
        # Create segments and concatenate them
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        segments = []
        current_pos = 0
        
        try:
            print(f"  📊 Processing {len(paused_snarks)} pauses...")
            
            for i, item in enumerate(paused_snarks):
                snark = item["snark"]
                pause_start_time = snark.time + snark.gap_duration  # Pause starts after the gap
                pause_duration = item["pause_duration"]  # Only the excess duration
                
                # 1. Add video segment before the pause (including the gap)
                if current_pos < pause_start_time:
                    segment_file = temp_dir / f"seg_{i:03d}_before.mp4"
                    duration = pause_start_time - current_pos
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "warning",
                        "-i", str(self.video_path),
                        "-ss", str(current_pos),
                        "-t", str(duration),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-an",  # No audio
                        str(segment_file)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0 and segment_file.exists():
                        segments.append(str(segment_file))
                        print(f"    • Segment {current_pos:.1f}s to {pause_start_time:.1f}s ({duration:.1f}s)")
                    else:
                        print(f"    ⚠️ Failed segment: {result.stderr}")
                
                # 2. Create frozen frame segment for excess duration only
                if pause_duration > 0.01:  # Only freeze if there's meaningful excess
                    # Extract the frame at the pause point
                    frame_file = temp_dir / f"frame_{i:03d}.png"
                    cmd = [
                        "ffmpeg", "-y", "-loglevel", "warning",
                        "-i", str(self.video_path),
                        "-ss", str(pause_start_time),
                        "-vframes", "1",
                        "-f", "image2",
                        str(frame_file)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if frame_file.exists():
                        # Create a video from the frozen frame
                        freeze_file = temp_dir / f"seg_{i:03d}_freeze.mp4"
                        cmd = [
                            "ffmpeg", "-y", "-loglevel", "warning",
                            "-loop", "1",
                            "-framerate", str(fps),
                            "-i", str(frame_file),
                            "-t", str(pause_duration),
                            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=disable",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                            "-pix_fmt", "yuv420p",
                            str(freeze_file)
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        if result.returncode == 0 and freeze_file.exists():
                            segments.append(str(freeze_file))
                            print(f"    • Freeze at {pause_start_time:.1f}s for {pause_duration:.1f}s")
                        else:
                            print(f"    ⚠️ Failed freeze: {result.stderr}")
                
                # Update position - video resumes from the pause point
                current_pos = pause_start_time
            
            # 3. Add remaining video after last pause
            # Get video duration
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                str(self.video_path)
            ], capture_output=True, text=True)
            video_duration = float(result.stdout.strip())
            
            if current_pos < video_duration:
                final_segment = temp_dir / f"seg_{len(paused_snarks):03d}_final.mp4"
                duration = video_duration - current_pos
                cmd = [
                    "ffmpeg", "-y", "-loglevel", "warning",
                    "-i", str(self.video_path),
                    "-ss", str(current_pos),
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-an",  # No audio
                    str(final_segment)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and final_segment.exists():
                    segments.append(str(final_segment))
                    print(f"    • Final segment from {current_pos:.1f}s ({duration:.1f}s)")
                else:
                    print(f"    ⚠️ Failed final: {result.stderr}")
            
            if not segments:
                print("  ⚠️ No segments created, using fallback")
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(self.video_path),
                    "-i", str(mixed_audio_path),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "256k",
                    "-pix_fmt", "yuv420p",
                    str(output_path)
                ], capture_output=True)
                return
            
            # 4. Create concat file
            concat_file = temp_dir / "concat.txt"
            with open(concat_file, "w") as f:
                for seg in segments:
                    # Use absolute paths for concat
                    f.write(f"file '{Path(seg).absolute()}'\n")
            
            print(f"  📼 Concatenating {len(segments)} segments...")
            
            # 5. Concatenate all video segments
            video_with_freezes = temp_dir / "video_frozen.mp4"
            cmd = [
                "ffmpeg", "-y", "-loglevel", "warning",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(video_with_freezes)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0 or not video_with_freezes.exists():
                print(f"  ❌ Concatenation failed: {result.stderr}")
                print("  🔧 Attempting direct merge without concat...")
                # Fallback: Use simple remux
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", str(self.video_path),
                    "-i", str(mixed_audio_path),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "256k",
                    "-pix_fmt", "yuv420p",
                    str(output_path)
                ], capture_output=True)
                return
            
            # 6. Combine frozen video with mixed audio
            print("  🎵 Adding audio track...")
            cmd = [
                "ffmpeg", "-y", "-loglevel", "warning",
                "-i", str(video_with_freezes),
                "-i", str(mixed_audio_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "256k",
                "-movflags", "+faststart",
                str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  ⚠️ Audio merge warning: {result.stderr}")
                # Try re-encoding
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_with_freezes),
                    "-i", str(mixed_audio_path),
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "256k",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
                subprocess.run(cmd, capture_output=True)
            
            if output_path.exists():
                print(f"  ✅ Created video with frozen frames: {output_path}")
            else:
                print(f"  ❌ Failed to create final video")
        
        finally:
            # Cleanup temp files
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _create_video_with_pauses(self, original_audio: AudioSegment, snarks_with_pausing: List[Dict]) -> AudioSegment:
        """Create audio track with video pauses for snarks that don't fit"""
        
        # Sort snarks by time
        snarks_with_pausing.sort(key=lambda x: x["snark"].time)
        
        result = AudioSegment.empty()
        current_pos = 0
        
        for item in snarks_with_pausing:
            snark = item["snark"]
            snark_audio = item["audio"].apply_gain(2)
            needs_pause = item["needs_pause"]
            pause_duration = item.get("pause_duration", 0)
            
            # Add original audio up to snark position
            snark_pos_ms = int(snark.time * 1000)
            result += original_audio[current_pos:snark_pos_ms]
            
            if needs_pause:
                # Play the gap portion with snark overlaid
                gap_duration_ms = int(snark.gap_duration * 1000)
                gap_segment = original_audio[snark_pos_ms:snark_pos_ms + gap_duration_ms]
                
                # Overlay snark on the gap portion
                result += gap_segment.overlay(snark_audio[:gap_duration_ms])
                
                # Then pause for the excess duration
                pause_duration_ms = int(pause_duration * 1000)
                if pause_duration_ms > 0:
                    # Create silence for the pause, with remaining snark audio
                    remaining_snark = snark_audio[gap_duration_ms:]
                    silence = AudioSegment.silent(duration=len(remaining_snark))
                    result += silence.overlay(remaining_snark)
                    print(f"  ⏸️ Paused at {snark.time + snark.gap_duration:.1f}s for {pause_duration:.1f}s: \"{snark.text}\"")
                
                # Update current_pos to skip the gap we already used
                current_pos = snark_pos_ms + gap_duration_ms
            else:
                # Normal overlay without pause
                segment_with_snark = original_audio[snark_pos_ms:snark_pos_ms + len(snark_audio)]
                result += segment_with_snark.overlay(snark_audio)
                current_pos = snark_pos_ms + len(snark_audio)
                print(f"  ✅ Added at {snark.time:.1f}s: \"{snark.text}\"")
        
        # Add remaining audio
        result += original_audio[current_pos:]
        
        return result
    
    def _find_existing_snark_for_context(self, context: str, max_duration: float, used_texts: set) -> Optional[Dict]:
        """Find a suitable existing snark for this context"""
        
        context_lower = context.lower()
        
        # Smart matching based on context keywords
        suitable_snarks = []
        
        for text, snark_data in self.existing_snarks.items():
            # Skip if already used in this video
            if text in used_texts:
                continue
            
            # Check if snark fits in the gap
            if snark_data["duration"] > max_duration:
                continue
            
            # Score relevance based on context
            score = 0
            
            # Perfect matches for common situations
            if "again" in context_lower or "repeat" in context_lower:
                if "original" in text or "again" in text:
                    score += 10
            
            if "sing" in context_lower or "song" in context_lower:
                if "musical" in text or "performance" in text:
                    score += 8
            
            if any(word in context_lower for word in ["do", "re", "mi", "fa", "sol"]):
                if "music" in text or "note" in text:
                    score += 7
            
            # Skip mean-spirited snarks from old library
            if text in ["riveting", "fascinating", "how original", "stunning"]:
                continue  # Don't use these
            
            # Only consider genuinely friendly snarks
            if text in ["nice", "cool", "neat", "wow", "interesting", "game on", "love it"]:
                if max_duration < 1.5 and len(text.split()) <= 2:
                    score += 8
                elif max_duration >= 1.5:
                    score += 5
            
            if score > 0:
                suitable_snarks.append((score, snark_data))
        
        # Return best match if found
        if suitable_snarks:
            suitable_snarks.sort(key=lambda x: x[0], reverse=True)
            return suitable_snarks[0][1]
        
        return None
    
    def _generate_custom_funny_snark(self, context: str, max_duration: float, position: str, transcript: Dict) -> Dict:
        """Generate custom hyper-funny snark for beginning or end"""
        
        # Be conservative with word count - account for TTS speed and padding
        max_words = min(int((max_duration - 0.3) * 2.2), 15)  # ~2.2 words/sec with padding
        
        print(f"    Gap duration: {max_duration:.1f}s → max {max_words} words")
        
        # Get overall video topic from transcript
        full_text = transcript.get("text", "")[:200]
        
        prompt = f"""
        You are a playfully cynical narrator adding witty commentary to this video.
        
        Video content summary: "{full_text}"
        What was just said: "{context}"
        Position: {position} of video
        
        Create ONE funny, perfectly tailored comment that's cynical but CHARMING.
        CRITICAL: Maximum {max_words} words! (Gap is only {max_duration:.1f} seconds)
        Must be SPECIFIC to this video's content, not generic.
        Be PLAYFUL and WITTY, not mean or harsh.
        
        Return JSON:
        {{
            "text": "your witty comment",
            "emotion": "choose: playful/amused/curious/impressed/cheerful/bemused"
        }}
        
        Examples by duration:
        VERY SHORT (0.6-1 sec, 1-2 words): "Neat!", "Sure!", "Classic."
        SHORT (1-2 sec, 2-4 words): "Plot twist!", "Who knew?", "Mind blown."
        MEDIUM (2-3 sec, 5-7 words): "Taking notes for the quiz.", "My favorite part."
        LONG (3+ sec, 8-15 words): "This is exactly what I needed today.", "Adding this to my collection of fun facts."
        
        Be funny but FRIENDLY - like a witty friend watching along!
        """
        
        try:
            if self.gemini_key:
                url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.gemini_key}
                
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.9,  # Higher for more creativity
                        "maxOutputTokens": 60
                    }
                }
                
                response = requests.post(url, headers=headers, params=params, json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    
                    # Parse JSON from response
                    import json
                    snark_data = json.loads(text)
                    return snark_data
        except Exception as e:
            print(f"  ⚠️ Custom generation failed: {e}")
        
        # Fallback custom remarks (playful, not mean)
        if position == "beginning":
            return {"text": "Ooh, this looks fun.", "emotion": "playful"}
        else:
            return {"text": "And that's a wrap folks.", "emotion": "cheerful"}
    
    def _generate_snark_with_ai(self, context: str, max_duration: float, used_texts: set) -> Optional[Dict]:
        """Generate a single snark using Gemini"""
        
        # Conservative word limit to ensure it fits in the gap
        max_words = min(int((max_duration - 0.3) * 2.2), 12)  # ~2.2 words/sec with padding
        
        print(f"    Gap: {max_duration:.1f}s → max {max_words} words")
        
        # Get list of existing snarks to inform AI
        existing_texts = list(self.existing_snarks.keys())[:20]  # Top 20
        
        # Filter out already used texts
        available_texts = [t for t in existing_texts if t not in used_texts]
        
        # Build library info with emotions
        library_info = []
        for text in existing_texts:
            if text in self.existing_snarks:
                emotion = self.existing_snarks[text].get("emotion", "unknown")
                library_info.append(f"{text} ({emotion})")
        
        prompt = f"""
        You are a playfully cynical narrator commenting on a video.
        
        Context (what was just said): "{context}"
        
        Already used in this video (NEVER repeat these): {list(used_texts)}
        
        Library (with emotions - prefer creating NEW friendly ones over sarcastic library):
        {library_info[:10]}
        Note: If library items are "sarcastic", "mocking", or "deadpan", create NEW friendly alternatives!
        
        Generate ONE brief witty comment that's cynical but CHARMING.
        CRITICAL: 
        - Maximum {max_words} words! (Gap is only {max_duration:.1f} seconds)
        - MUST be COMPLETELY DIFFERENT from already used snarks
        - If you see "Fascinating" was already used, try: "Intriguing!", "Cool beans!", "Noted!", etc.
        - Be VARIED - don't just change punctuation!
        
        Return JSON:
        {{
            "text": "your UNIQUE witty comment",
            "emotion": "choose: playful/amused/curious/cheerful/bemused/impressed",
            "is_existing": false
        }}
        
        Examples of variety:
        - Instead of repeating "Fascinating": try "Mind = blown", "Taking notes", "Tell me more"
        - If technical: "Science time!", "Math is fun", "Nerding out here"
        - If obvious: "Plot twist!", "Who knew?", "Breaking news"
        
        Be funny but FRIENDLY and VARIED!
        """
        
        try:
            # Call Gemini API
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            headers = {"Content-Type": "application/json"}
            params = {"key": self.gemini_key}
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.8,
                    "maxOutputTokens": 50
                }
            }
            
            response = requests.post(url, headers=headers, params=params, json=data)
            
            if response.status_code == 200:
                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Parse JSON from response
                import json
                snark_data = json.loads(text)
                return snark_data
            
        except Exception as e:
            print(f"  ⚠️ AI generation failed: {e}")
        
        # Fallback
        return {
            "text": "Fascinating.",
            "emotion": "sarcastic"
        }
    
    def _get_fallback_snark(self, max_duration: float, used_texts: set, force_different: bool = False) -> Dict:
        """Get a fallback snark when AI is not available"""
        
        # Skip existing library if force_different or if library has mean emotions
        if not force_different:
            for text, snark_data in self.existing_snarks.items():
                # Skip if already used or if emotion is mean-spirited
                if text in used_texts:
                    continue
                if snark_data.get("emotion") in ["sarcastic", "mocking", "deadpan", "bored", "unimpressed"]:
                    continue  # Skip mean-spirited ones
                if snark_data["duration"] <= max_duration:
                    return {
                        "text": snark_data["text"],
                        "emotion": snark_data["emotion"]
                    }
        
        # Expanded fallbacks for variety (rotate through unused ones - playful tone)
        fallbacks = [
            ("Nice!", "cheerful", 1.5),
            ("Interesting!", "curious", 2.5),
            ("Plot twist!", "amused", 3.0),
            ("Cool beans.", "playful", 2.0),
            ("Mind blown.", "impressed", 2.5),
            ("Game changer.", "playful", 3.0),
            ("Noted!", "cheerful", 2.0),
            ("Love it.", "amused", 2.5),
            ("Wow!", "impressed", 1.0),
            ("Amazing!", "excited", 2.0),
            ("Tell me more.", "curious", 2.5),
            ("Go on.", "intrigued", 1.5),
            ("Science!", "excited", 2.0),
            ("Math time!", "playful", 2.0),
            ("Learning!", "cheerful", 2.0),
            ("Fun fact!", "amused", 2.0),
            ("Who knew?", "surprised", 2.0),
            ("Breaking news.", "playful", 2.5),
            ("Taking notes.", "studious", 2.5),
            ("Brilliant!", "impressed", 2.0)
        ]
        
        for text, emotion, min_duration in fallbacks:
            if text.lower() not in used_texts and max_duration >= min_duration - 0.5:
                return {"text": text, "emotion": emotion}
        
        # Last resort - still try to avoid repetition
        last_resorts = [
            ("Hmm.", "thoughtful"),
            ("I see.", "understanding"),
            ("Got it.", "acknowledging"),
            ("Right.", "agreeable"),
            ("Okay then.", "accepting")
        ]
        
        for text, emotion in last_resorts:
            if text.lower() not in used_texts:
                return {"text": text, "emotion": emotion}
        
        # Absolute last resort
        return {"text": "Mmhmm.", "emotion": "listening"}
    
    def _generate_fallback_snarks(self, gaps: List[SilenceGap]) -> List[Snark]:
        """Fallback snarks when AI is not available"""
        
        fallback_pool = [
            ("Nice!", "cheerful"),
            ("Plot twist!", "amused"),
            ("Taking notes.", "playful"),
            ("Love it!", "impressed"),
            ("Game on.", "excited"),
            ("Neat!", "curious")
        ]
        
        snarks = []
        for i, gap in enumerate(gaps[:6]):
            text, emotion = fallback_pool[i % len(fallback_pool)]
            snarks.append(Snark(
                text=text,
                time=gap.start + 0.2,
                emotion=emotion,
                context="",
                gap_duration=gap.duration
            ))
        
        return snarks
    
    def generate_speech_with_elevenlabs(self, snark: Snark) -> Optional[str]:
        """Step 4: Convert snark text to speech"""
        
        # Log if this might need pausing (actual decision made later with real audio)
        estimated_duration = self._estimate_duration(snark.text)
        if estimated_duration > snark.gap_duration:
            print(f"  ⚠️ May need pause for: '{snark.text}' (est. {estimated_duration:.1f}s > gap {snark.gap_duration:.1f}s)")
        
        # Check if we have this text BUT with wrong emotion
        text_lower = snark.text.lower()
        if text_lower in self.existing_snarks:
            existing_emotion = self.existing_snarks[text_lower].get("emotion", "")
            # If existing is mean-spirited, regenerate with friendly emotion
            if existing_emotion in ["sarcastic", "mocking", "deadpan", "bored", "unimpressed"]:
                print(f"  🔄 Regenerating '{snark.text}' with friendly emotion (was {existing_emotion})")
                # Don't return existing, generate new one below
            else:
                existing_file = self.existing_snarks[text_lower]["file"]
                if os.path.exists(existing_file):
                    print(f"  ♻️ Found in library: {Path(existing_file).name}")
                    return existing_file
        
        # Create filename from text
        filename = re.sub(r'[^\w\s]', '', snark.text.lower())
        filename = re.sub(r'\s+', '_', filename)[:50] + ".mp3"
        audio_path = self.snark_storage / filename
        
        # Check if file exists (might have been created in another session)
        if audio_path.exists():
            print(f"  ♻️ Reusing existing file: {filename}")
            # Add to library for future reference
            self.existing_snarks[text_lower] = {
                "text": snark.text,
                "file": str(audio_path),
                "emotion": snark.emotion,
                "duration": self._estimate_duration(snark.text)
            }
            return str(audio_path)
        
        if not self.elevenlabs_key:
            print(f"  ⚠️ No ElevenLabs key, skipping: {snark.text}")
            return None
        
        print(f"  🎤 Generating: \"{snark.text}\" → {filename}")
        
        # ElevenLabs API call
        voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "accept": "audio/mpeg",
            "content-type": "application/json",
            "xi-api-key": self.elevenlabs_key
        }
        
        # Map emotions to friendlier ones
        friendly_emotions = {
            "sarcastic": "playful",
            "deadpan": "cheerful",
            "mocking": "amused",
            "bored": "curious",
            "unimpressed": "interested",
            "resigned": "excited",
            "skeptical": "intrigued"
        }
        
        # Use friendly emotion
        emotion = friendly_emotions.get(snark.emotion, snark.emotion)
        
        # Add emotion tags
        emotion_text = f'<emotion="{emotion}">{snark.text}</emotion>'
        
        payload = {
            "text": emotion_text,
            "model_id": self.elevenlabs_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                with open(audio_path, "wb") as f:
                    f.write(resp.content)
                print(f"    ✅ Saved: {filename}")
                return str(audio_path)
        except Exception as e:
            print(f"    ❌ Failed: {e}")
        
        return None
    
    def mix_snarks_into_video(self, snarks: List[Snark]) -> str:
        """Step 5: Mix snarks into final video"""
        
        print("🎬 Creating final video with snarks...")
        
        # Save remarks JSON
        remarks_file = self.output_folder / "remarks.json"
        remarks_data = []
        
        # Generate speech for all snarks and save remark files
        for i, snark in enumerate(snarks):
            audio_path = self.generate_speech_with_elevenlabs(snark)
            snark.audio_path = audio_path
            
            # Copy audio file to output folder
            if audio_path:
                local_audio = self.output_folder / f"remark_{i+1}.mp3"
                subprocess.run(["cp", audio_path, str(local_audio)], capture_output=True)
                
            remarks_data.append({
                "time": snark.time,
                "text": snark.text,
                "emotion": snark.emotion,
                "context": snark.context[:50] if snark.context else "",
                "audio_file": f"remark_{i+1}.mp3",
                "library_file": Path(audio_path).name if audio_path else None
            })
        
        # Save remarks JSON
        with open(remarks_file, "w") as f:
            json.dump(remarks_data, f, indent=2)
        print(f"  📝 Saved remarks: {remarks_file}")
        
        # Filter out snarks without audio
        valid_snarks = [s for s in snarks if hasattr(s, 'audio_path') and s.audio_path]
        
        if not valid_snarks:
            print("❌ No valid snarks to add")
            return str(self.video_path)
        
        # Extract original audio
        audio_path = self.output_folder / "original_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le",
            str(audio_path)
        ], capture_output=True)
        
        # Load and normalize
        original = AudioSegment.from_file(audio_path)
        normalized = normalize(original)
        target_dBFS = -18
        change = target_dBFS - normalized.dBFS
        normalized = normalized.apply_gain(change)
        
        # Check which snarks need video pausing
        snarks_with_pausing = []
        for snark in valid_snarks:
            if not os.path.exists(snark.audio_path):
                continue
            
            # Get actual duration of the audio file
            snark_audio = AudioSegment.from_mp3(snark.audio_path)
            snark_duration = len(snark_audio) / 1000.0  # Convert to seconds
            
            # Calculate if we need to pause and for how long
            # Video plays during the gap, only pause for excess duration
            excess_duration = max(0, snark_duration - snark.gap_duration)
            needs_pause = excess_duration > 0
            
            snarks_with_pausing.append({
                "snark": snark,
                "audio": snark_audio,
                "duration": snark_duration,
                "needs_pause": needs_pause,
                "pause_duration": excess_duration  # Only pause for the excess
            })
        
        # Create audio with pauses if needed
        has_pauses = any(s["needs_pause"] for s in snarks_with_pausing)
        
        if has_pauses:
            print("  ⏸️ Some remarks need video pausing...")
            mixed = self._create_video_with_pauses(normalized, snarks_with_pausing)
        else:
            # Simple overlay without pausing
            mixed = normalized
            for item in snarks_with_pausing:
                snark_audio = item["audio"].apply_gain(2)  # Gentle boost
                position_ms = int(item["snark"].time * 1000)
                mixed = mixed.overlay(snark_audio, position=position_ms)
                print(f"  ✅ Added at {item['snark'].time:.1f}s: \"{item['snark'].text}\"")
        
        # Export mixed audio
        mixed_path = self.output_folder / "mixed_audio.wav"
        mixed.export(mixed_path, format="wav")
        
        # Create final video in output folder
        output_path = self.output_folder / f"{self.video_name}_final.mp4"
        
        # Check if we need to freeze video frames
        if has_pauses:
            # Create video with frozen frames
            print("  🎬 Creating video with frozen frames during pauses...")
            self._create_video_with_frozen_frames(snarks_with_pausing, mixed_path, output_path)
        else:
            # Simple remux without frame freezing
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", str(mixed_path),
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "256k",
                "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
                str(output_path)
            ], capture_output=True)
        
        print(f"✅ Output: {output_path}")
        return str(output_path)
    
    def run_pipeline(self) -> str:
        """Run the complete end-to-end pipeline"""
        
        print("\n" + "=" * 70)
        print("🚀 RUNNING END-TO-END SNARK PIPELINE")
        print("=" * 70)
        
        # Step 1: Extract transcript
        transcript = self.extract_transcript()
        
        # Step 2: Find silence gaps
        gaps = self.analyze_silence_gaps()
        
        if not gaps:
            print("❌ No silence gaps found - cannot add snarks")
            return str(self.video_path)
        
        # Step 3: Generate contextual snarks
        snarks = self.generate_contextual_snarks(transcript, gaps)
        
        # Step 4 & 5: Generate speech and mix into video
        output = self.mix_snarks_into_video(snarks)
        
        # Save report
        report = {
            "video": str(self.video_path),
            "transcript_segments": len(transcript["segments"]),
            "silence_gaps": len(gaps),
            "snarks_generated": len(snarks),
            "snarks": [
                {
                    "time": s.time,
                    "text": s.text,
                    "emotion": s.emotion,
                    "context": s.context[:30] if s.context else "",
                    "gap_duration": s.gap_duration
                }
                for s in snarks
            ],
            "output": output
        }
        
        report_path = self.output_folder / "pipeline_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 70)
        print("✨ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"📊 Report: {report_path}")
        print(f"🎬 Output: {output}")
        
        # Calculate cost
        total_chars = sum(len(s.text) for s in snarks)
        cost = total_chars * 0.0003
        print(f"💰 Estimated cost: ${cost:.3f}")
        
        # Auto-open the final video
        print("\n🎯 Opening final video...")
        try:
            subprocess.run(["open", output], check=True)
        except Exception as e:
            print(f"  ⚠️ Could not auto-open video: {e}")
        
        return output

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-end snark pipeline")
    parser.add_argument("video", help="Path to input video")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        return
    
    # Run pipeline
    pipeline = EndToEndSnarkPipeline(args.video)
    output = pipeline.run_pipeline()
    
    print(f"\n🎯 Done! Output: {output}")

if __name__ == "__main__":
    main()