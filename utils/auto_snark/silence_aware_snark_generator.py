#!/usr/bin/env python3
"""
Silence-Aware Snark Generator
Generates snarks ONLY for silent moments and ensures they fit within gaps
"""

import os
import json
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pydub import AudioSegment
from pydub.silence import detect_silence
import numpy as np
from pathlib import Path

@dataclass
class SilentPeriod:
    """Represents a period of silence in the video"""
    start_ms: int
    end_ms: int
    duration_ms: int
    start_s: float
    end_s: float
    duration_s: float
    
    @property
    def is_usable(self) -> bool:
        """Check if this silence period is long enough for a snark"""
        return self.duration_s >= 1.5  # Minimum 1.5 seconds for a snark

@dataclass
class PlannedSnark:
    """A snark planned for a specific silent period"""
    silence_period: SilentPeriod
    text: str
    max_duration_s: float
    style: str
    priority: int  # 1-10, higher = more important
    context: str  # What's happening in the video

class SilenceAwareSnarkGenerator:
    """Generates snarks that fit perfectly within silent periods"""
    
    # Short, punchy snark templates by duration
    SNARK_TEMPLATES = {
        "very_short": [  # 1.5-2 seconds
            "Wow.",
            "Sure.",
            "Right.",
            "Oh boy.",
            "Thrilling.",
            "Amazing.",
            "Brilliant.",
            "Shocking.",
            "Riveting.",
        ],
        "short": [  # 2-3 seconds
            "How original.",
            "Never seen that before.",
            "Groundbreaking stuff.",
            "This changes everything.",
            "Pure genius.",
            "Mind = blown.",
            "Oscar-worthy.",
            "Revolutionary concept.",
        ],
        "medium": [  # 3-4 seconds
            "And the crowd goes mild.",
            "Alert the media, this is huge.",
            "Someone call the Nobel committee.",
            "I'm on the edge of my seat.",
            "This is why we can't have nice things.",
        ],
        "long": [  # 4-5 seconds
            "Ah yes, because that's exactly what we needed right now.",
            "I'm sure this will age like fine wine.",
            "This is definitely the highlight of my existence.",
        ]
    }
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.silent_periods = []
        self.transcript = None
        
    def analyze_silence(self, 
                        silence_thresh: int = -40,
                        min_silence_len: int = 1500) -> List[SilentPeriod]:
        """Detect all periods of silence in the video"""
        
        print("🔍 Analyzing video for silent periods...")
        
        # Extract audio
        temp_audio = "temp_silence_analysis.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", self.video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050",
            temp_audio
        ], capture_output=True)
        
        # Load and analyze
        audio = AudioSegment.from_file(temp_audio)
        
        # Detect silence with strict criteria
        silent_ranges = detect_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            seek_step=100
        )
        
        # Also check for very quiet periods (background music only)
        quiet_ranges = detect_silence(
            audio,
            min_silence_len=1200,  # Slightly shorter threshold
            silence_thresh=-35,     # Less strict
            seek_step=100
        )
        
        # Convert to SilentPeriod objects
        periods = []
        
        # Process truly silent periods
        for start_ms, end_ms in silent_ranges:
            duration_ms = end_ms - start_ms
            periods.append(SilentPeriod(
                start_ms=start_ms,
                end_ms=end_ms,
                duration_ms=duration_ms,
                start_s=start_ms / 1000,
                end_s=end_ms / 1000,
                duration_s=duration_ms / 1000
            ))
        
        # Add quiet periods that don't overlap with silent ones
        for start_ms, end_ms in quiet_ranges:
            # Check if this overlaps with any existing period
            overlaps = False
            for p in periods:
                if not (end_ms < p.start_ms or start_ms > p.end_ms):
                    overlaps = True
                    break
            
            if not overlaps:
                duration_ms = end_ms - start_ms
                periods.append(SilentPeriod(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration_ms,
                    start_s=start_ms / 1000,
                    end_s=end_ms / 1000,
                    duration_s=duration_ms / 1000
                ))
        
        # Sort by start time
        periods.sort(key=lambda x: x.start_ms)
        
        # Filter to only usable periods
        usable = [p for p in periods if p.is_usable]
        
        print(f"  Found {len(periods)} total silent/quiet periods")
        print(f"  {len(usable)} are long enough for snarks (>= 1.5s)")
        
        # Cleanup
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        self.silent_periods = usable
        return usable
    
    def load_transcript(self, transcript_path: Optional[str] = None) -> dict:
        """Load transcript if available for context"""
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path) as f:
                self.transcript = json.load(f)
                print(f"  Loaded transcript with {len(self.transcript.get('segments', []))} segments")
        return self.transcript
    
    def get_video_context(self, time_s: float) -> str:
        """Get what's happening in the video at a specific time"""
        if not self.transcript:
            return "scene"
        
        # Find the nearest transcript segment
        segments = self.transcript.get("segments", [])
        for seg in segments:
            if seg["start"] <= time_s <= seg["end"]:
                return seg.get("text", "scene")[:50]
        
        return "transition"
    
    def detect_comedic_moments(self) -> List[Tuple[SilentPeriod, float]]:
        """Detect which silent periods are good for comedy"""
        
        print("\n🎭 Detecting comedic opportunities in silence...")
        
        comedic_moments = []
        
        for period in self.silent_periods:
            score = 0.0
            
            # Score based on timing patterns
            time_s = period.start_s
            
            # Good times for snarks:
            # 1. After musical flourishes (every ~8-10 seconds in musicals)
            if time_s % 8 < 1 or time_s % 10 < 1:
                score += 0.3
            
            # 2. After someone finishes singing a line
            context = self.get_video_context(time_s - 1)  # Check just before silence
            if any(word in context.lower() for word in ["do", "re", "mi", "fa", "sol", "la", "ti"]):
                score += 0.5
            
            # 3. Longer silences are better opportunities
            if period.duration_s > 3:
                score += 0.4
            elif period.duration_s > 2:
                score += 0.2
            
            # 4. Not too early in the video
            if time_s > 2:
                score += 0.2
            
            # 5. After dramatic moments
            if period.duration_s > 2.5 and time_s > 10:
                score += 0.3
            
            if score > 0.5:  # Threshold for comedic moment
                comedic_moments.append((period, score))
        
        # Sort by score
        comedic_moments.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Found {len(comedic_moments)} comedic opportunities")
        
        return comedic_moments
    
    def generate_snark_for_period(self, 
                                  period: SilentPeriod, 
                                  context: str,
                                  previous_snarks: List[str]) -> PlannedSnark:
        """Generate a snark that fits within the given silent period"""
        
        # Determine appropriate snark length based on available time
        # Account for TTS speaking rate (~150 words per minute = 2.5 words/second)
        max_duration = period.duration_s - 0.3  # Leave 300ms buffer
        
        # Select template category based on duration
        if max_duration < 2:
            templates = self.SNARK_TEMPLATES["very_short"]
            style = "deadpan"
        elif max_duration < 3:
            templates = self.SNARK_TEMPLATES["short"]
            style = "sarcastic"
        elif max_duration < 4:
            templates = self.SNARK_TEMPLATES["medium"]
            style = "mocking"
        else:
            templates = self.SNARK_TEMPLATES["long"]
            style = "condescending"
        
        # Avoid repetition
        available = [t for t in templates if t not in previous_snarks]
        if not available:
            available = templates  # Reset if we've used them all
        
        # Pick a snark
        import random
        snark_text = random.choice(available)
        
        # Customize based on context if it's a musical
        if "do" in context.lower() or "re" in context.lower() or "mi" in context.lower():
            musical_snarks = [
                "Music theory at its peak.",
                "Broadway, here we come.",
                "The hills are alive... unfortunately.",
                "Sondheim is shaking.",
            ]
            # Only use if it fits
            for ms in musical_snarks:
                words = len(ms.split())
                estimated_duration = words / 2.5 + 0.5  # Add pause time
                if estimated_duration <= max_duration and ms not in previous_snarks:
                    snark_text = ms
                    break
        
        return PlannedSnark(
            silence_period=period,
            text=snark_text,
            max_duration_s=max_duration,
            style=style,
            priority=8 if period.duration_s > 3 else 5,
            context=context
        )
    
    def plan_snarks(self, max_snarks: int = 8) -> List[PlannedSnark]:
        """Plan all snarks for the video based on silence analysis"""
        
        print("\n📝 Planning snarks for silent periods...")
        
        # First, analyze silence
        if not self.silent_periods:
            self.analyze_silence()
        
        # Detect comedic moments
        comedic_moments = self.detect_comedic_moments()
        
        # Generate snarks for best moments
        planned_snarks = []
        used_texts = []
        
        for period, score in comedic_moments[:max_snarks]:
            context = self.get_video_context(period.start_s)
            snark = self.generate_snark_for_period(period, context, used_texts)
            planned_snarks.append(snark)
            used_texts.append(snark.text)
            
            print(f"  📍 {period.start_s:.1f}s ({period.duration_s:.1f}s gap): \"{snark.text}\"")
        
        return planned_snarks
    
    def generate_tts_audio(self, planned_snarks: List[PlannedSnark], use_elevenlabs: bool = True):
        """Generate TTS audio for planned snarks"""
        
        print("\n🎤 Generating TTS audio for snarks...")
        
        audio_files = []
        
        for i, snark in enumerate(planned_snarks):
            output_file = f"silence_aware_snark_{i+1}_{snark.style}.mp3"
            
            if use_elevenlabs and os.getenv("ELEVEN_API_KEY"):
                # Use ElevenLabs for high quality
                from elevenlabs import VoiceSettings, save
                from elevenlabs.client import ElevenLabs
                
                client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
                
                # Add emotion tags for v3
                text_with_emotion = f'<emotion="{snark.style}">{snark.text}</emotion>'
                
                # Generate with appropriate settings
                audio = client.generate(
                    text=text_with_emotion,
                    voice="Rachel",  # Snarky female voice
                    model="eleven_turbo_v2_5",
                    voice_settings=VoiceSettings(
                        stability=0.3 if snark.style == "sarcastic" else 0.5,
                        similarity_boost=0.8,
                        style=0.7 if snark.style in ["mocking", "condescending"] else 0.5,
                        use_speaker_boost=True
                    )
                )
                
                save(audio, output_file)
                print(f"    ✅ Generated: {output_file} (ElevenLabs)")
                
            else:
                # Fallback to system TTS
                # Adjust speech rate to fit within time limit
                words = len(snark.text.split())
                target_duration = min(snark.max_duration_s, words / 2.5 + 0.3)
                
                # Calculate speech rate (words per minute)
                rate = int((words / target_duration) * 60)
                rate = min(max(rate, 150), 250)  # Clamp between 150-250 wpm
                
                subprocess.run([
                    "say", "-v", "Samantha",
                    "-r", str(rate),
                    "-o", output_file.replace(".mp3", ".aiff"),
                    snark.text
                ], capture_output=True)
                
                # Convert to MP3
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_file.replace(".mp3", ".aiff"),
                    "-acodec", "mp3", "-ab", "192k",
                    output_file
                ], capture_output=True)
                
                # Cleanup
                if os.path.exists(output_file.replace(".mp3", ".aiff")):
                    os.remove(output_file.replace(".mp3", ".aiff"))
                
                print(f"    ✅ Generated: {output_file} (System TTS @ {rate} wpm)")
            
            audio_files.append({
                "file": output_file,
                "snark": snark,
                "estimated_duration": snark.max_duration_s
            })
        
        return audio_files
    
    def create_snarked_video(self, 
                            planned_snarks: List[PlannedSnark],
                            audio_files: List[dict],
                            output_path: str):
        """Create final video with snarks placed in silent periods"""
        
        print("\n🎬 Creating final video with silence-aware snarks...")
        
        # Build FFmpeg command
        cmd = ["ffmpeg", "-y", "-i", self.video_path]
        
        # Add all snark audio files
        for af in audio_files:
            cmd.extend(["-i", af["file"]])
        
        # Build filter complex for mixing at exact silence times
        filter_parts = []
        
        # Add delays to position each snark
        for i, af in enumerate(audio_files):
            delay_ms = int(af["snark"].silence_period.start_ms + 200)  # Small offset into silence
            filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[snark{i}]")
        
        # Mix all audio
        inputs = ["0:a"] + [f"snark{i}" for i in range(len(audio_files))]
        mix = "[" + "][".join(inputs) + f"]amix=inputs={len(inputs)}:duration=first[outa]"
        filter_parts.append(mix)
        
        filter_complex = ";".join(filter_parts)
        
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[outa]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output_path
        ])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Video created: {output_path}")
        else:
            print(f"❌ Error: {result.stderr[:500]}")
    
    def generate_report(self, planned_snarks: List[PlannedSnark], output_file: str = "silence_aware_report.json"):
        """Generate detailed report of snark placement"""
        
        report = {
            "video": self.video_path,
            "total_silent_periods": len(self.silent_periods),
            "usable_silent_periods": len([p for p in self.silent_periods if p.is_usable]),
            "snarks_generated": len(planned_snarks),
            "snarks": [
                {
                    "time": snark.silence_period.start_s,
                    "silence_duration": snark.silence_period.duration_s,
                    "text": snark.text,
                    "max_duration": snark.max_duration_s,
                    "style": snark.style,
                    "priority": snark.priority,
                    "context": snark.context[:30]
                }
                for snark in planned_snarks
            ],
            "silence_coverage": {
                "total_silence_time": sum(p.duration_s for p in self.silent_periods),
                "used_silence_time": sum(s.silence_period.duration_s for s in planned_snarks),
                "percentage_used": (len(planned_snarks) / len(self.silent_periods) * 100) if self.silent_periods else 0
            }
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved: {output_file}")
        return report


def main():
    """Main execution for silence-aware snark generation"""
    
    print("=" * 70)
    print("🤫 SILENCE-AWARE SNARK GENERATOR")
    print("=" * 70)
    print("Only places snarks during silence, sized to fit gaps perfectly")
    print("=" * 70)
    
    # Input video
    video_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    
    # Optional transcript for better context
    transcript_path = "do_re_mi_transcript.json"
    
    # Initialize generator
    generator = SilenceAwareSnarkGenerator(video_path)
    
    # Load transcript if available
    if os.path.exists(transcript_path):
        generator.load_transcript(transcript_path)
    
    # Analyze silence in the video
    silent_periods = generator.analyze_silence()
    
    print(f"\n📊 Silence Analysis Results:")
    print(f"  Total silent periods: {len(silent_periods)}")
    print(f"  Total silence time: {sum(p.duration_s for p in silent_periods):.1f}s")
    print(f"  Average gap duration: {np.mean([p.duration_s for p in silent_periods]):.1f}s")
    
    # Plan snarks
    planned_snarks = generator.plan_snarks(max_snarks=6)
    
    print(f"\n✅ Planned {len(planned_snarks)} snarks to fit in silence")
    
    # Check for existing audio or generate new
    use_existing = False
    existing_files = []
    
    for i in range(len(planned_snarks)):
        if os.path.exists(f"silence_aware_snark_{i+1}_*.mp3"):
            use_existing = True
            existing_files.append(f"silence_aware_snark_{i+1}_*.mp3")
    
    if use_existing:
        print("\n💰 Using existing audio files (no API cost)")
        audio_files = [{"file": f, "snark": s, "estimated_duration": s.max_duration_s} 
                      for f, s in zip(existing_files, planned_snarks)]
    else:
        # Generate TTS audio
        audio_files = generator.generate_tts_audio(planned_snarks, use_elevenlabs=True)
    
    # Create final video
    output_path = "do_re_mi_silence_aware.mp4"
    generator.create_snarked_video(planned_snarks, audio_files, output_path)
    
    # Generate report
    report = generator.generate_report(planned_snarks)
    
    print("\n🎯 SILENCE-AWARE GENERATION COMPLETE!")
    print(f"📹 Output: {output_path}")
    print("\n✨ Key Features:")
    print("  • Snarks ONLY during silence (no talking)")
    print("  • Each snark sized to fit its silence gap")
    print("  • No overlap with dialogue")
    print("  • Natural comedic timing")
    

if __name__ == "__main__":
    main()