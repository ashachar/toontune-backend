#!/usr/bin/env python3
"""
Scene_003 Silence-Aware Snarker for ORIGINAL quality video
Uses ELEVENLABS_MODEL from .env
Stores all snarks in uploads/assets/sounds/comments_audio/
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.effects import normalize
import requests

# Load .env from backend directory
backend_env = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/.env")
if backend_env.exists():
    load_dotenv(backend_env)
    print(f"✅ Loaded .env from: {backend_env}")

class Scene003Snarker:
    def __init__(self):
        # Use ORIGINAL quality video, not downsampled
        self.video_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/original/scene_003.mp4"
        
        # Storage directory for all snarks
        self.snark_storage = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/sounds/comments_audio")
        self.snark_storage.mkdir(parents=True, exist_ok=True)
        
        # Scene 003 timing: 111.84 to 136.16 seconds (24.32 seconds total)
        self.scene_start = 111.84
        self.scene_end = 136.16
        self.scene_duration = self.scene_end - self.scene_start
        
        # Get model from .env
        self.model_id = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
        print(f"📍 Using ElevenLabs model: {self.model_id}")
        
        # Extract transcript for scene_003
        self.segments = self.extract_scene_transcript()
        
    def extract_scene_transcript(self):
        """Extract transcript segments for scene_003 from full transcript"""
        
        transcript_path = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/transcripts/transcript_sentences.json")
        
        if not transcript_path.exists():
            print("⚠️ Transcript not found, using estimated timings")
            return []
        
        with open(transcript_path) as f:
            full_transcript = json.load(f)
        
        # Filter segments within scene_003 timeframe
        scene_segments = []
        for seg in full_transcript.get("sentences", []):
            if seg["start"] >= self.scene_start and seg["end"] <= self.scene_end:
                # Convert to scene-relative timing
                scene_segments.append({
                    "text": seg["text"],
                    "start": seg["start"] - self.scene_start,
                    "end": seg["end"] - self.scene_start
                })
        
        print(f"📝 Found {len(scene_segments)} transcript segments for scene_003:")
        for seg in scene_segments:
            print(f"   {seg['start']:.1f}-{seg['end']:.1f}s: \"{seg['text'][:30]}...\"")
        
        return scene_segments
    
    def analyze_silence(self):
        """Analyze scene_003 for silence gaps"""
        
        print("\n🔍 Analyzing scene_003 (ORIGINAL) for silence...")
        
        # Extract audio from original video
        temp_audio = "scene_003_temp.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", self.video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050",
            temp_audio
        ], capture_output=True)
        
        # Load and analyze
        audio = AudioSegment.from_file(temp_audio)
        
        # Detect silence
        silent_ranges = detect_silence(
            audio,
            min_silence_len=1000,  # 1 second minimum
            silence_thresh=-40,
            seek_step=100
        )
        
        # Also detect quieter periods
        quiet_ranges = detect_silence(
            audio,
            min_silence_len=800,
            silence_thresh=-35,
            seek_step=100
        )
        
        # Clean up
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        # Find gaps between dialogue
        gaps = []
        
        # Check gap at scene start
        if self.segments and self.segments[0]["start"] > 0.8:
            gaps.append({
                "start": 0.0,
                "end": self.segments[0]["start"],
                "duration": self.segments[0]["start"],
                "type": "start",
                "after": "scene start"
            })
        
        # Check gaps between segments
        for i in range(len(self.segments) - 1):
            gap_start = self.segments[i]["end"]
            gap_end = self.segments[i+1]["start"]
            gap_duration = gap_end - gap_start
            
            if gap_duration > 0.6:  # At least 0.6 seconds for short snarks
                gaps.append({
                    "start": gap_start,
                    "end": gap_end,
                    "duration": gap_duration,
                    "type": "between",
                    "after": self.segments[i]["text"][:30]
                })
        
        # Check gap at scene end
        if self.segments and (self.scene_duration - self.segments[-1]["end"]) > 0.8:
            gaps.append({
                "start": self.segments[-1]["end"],
                "end": self.scene_duration,
                "duration": self.scene_duration - self.segments[-1]["end"],
                "type": "end",
                "after": self.segments[-1]["text"][:30]
            })
        
        print(f"  Found {len(silent_ranges)} silent periods")
        print(f"  Found {len(quiet_ranges)} quiet periods")
        print(f"  Found {len(gaps)} usable gaps")
        
        return gaps
    
    def plan_snarks(self, gaps):
        """Plan snarks for scene_003 based on gaps"""
        
        # Scene 003 is the finale - more dramatic/conclusive snarks
        planned = []
        
        for gap in gaps:
            if gap["duration"] < 0.7:  # Allow shorter snarks
                continue
            
            # Select snark based on context and position
            if gap["type"] == "start":
                snark = {
                    "text": "Grand finale time.",
                    "style": "dramatic",
                    "emotion": "anticipatory"
                }
            elif "Mi, a name" in gap.get("after", ""):
                snark = {
                    "text": "Third verse, same as the first.",
                    "style": "deadpan",
                    "emotion": "bored"
                }
            elif "long way to run" in gap.get("after", ""):
                snark = {
                    "text": "Still running.",
                    "style": "exhausted",
                    "emotion": "tired"
                }
            elif "needle pulling" in gap.get("after", ""):
                snark = {
                    "text": "The metaphors persist.",
                    "style": "mocking",
                    "emotion": "amused"
                }
            elif "jam and bread" in gap.get("after", ""):
                snark = {
                    "text": "Peak lyricism achieved.",
                    "style": "sarcastic",
                    "emotion": "impressed"
                }
            elif gap["type"] == "end" or "back to Do" in gap.get("after", ""):
                snark = {
                    "text": "And scene.",
                    "style": "conclusive",
                    "emotion": "relieved"
                }
            else:
                # Generic snark for other gaps
                snark = {
                    "text": "Inspiring.",
                    "style": "sarcastic",
                    "emotion": "unimpressed"
                }
            
            snark["gap"] = gap
            snark["time"] = gap["start"] + 0.2  # Small offset into gap
            planned.append(snark)
        
        return planned[:6]  # Limit to 6 snarks max
    
    def generate_elevenlabs_audio(self, text, style, emotion, output_path):
        """Generate audio using ElevenLabs with model from .env"""
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not api_key:
            print(f"❌ No API key found for: {text}")
            return False
        
        print(f"  🎤 Generating: \"{text}\" ({emotion})")
        
        # Rachel voice for sarcasm
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "accept": "audio/mpeg",
            "content-type": "application/json",
            "xi-api-key": api_key
        }
        
        # Add emotion tags based on the emotion parameter
        emotion_text = f'<emotion="{emotion}">{text}</emotion>'
        
        # Configure voice settings based on style
        # For eleven_v3, we need different parameters
        if self.model_id == "eleven_v3":
            # eleven_v3 has different parameter ranges
            if style == "deadpan":
                voice_settings = {"stability": 0.9, "similarity_boost": 0.7}
            elif style == "sarcastic":
                voice_settings = {"stability": 0.3, "similarity_boost": 0.85}
            elif style == "mocking":
                voice_settings = {"stability": 0.35, "similarity_boost": 0.8}
            elif style == "dramatic":
                voice_settings = {"stability": 0.5, "similarity_boost": 0.9}
            elif style == "conclusive":
                voice_settings = {"stability": 0.7, "similarity_boost": 0.8}
            else:
                voice_settings = {"stability": 0.5, "similarity_boost": 0.85}
        else:
            # For other models like eleven_turbo_v2_5
            if style == "deadpan":
                voice_settings = {"stability": 0.9, "similarity_boost": 0.7, "style": 0.05, "use_speaker_boost": True}
            elif style == "sarcastic":
                voice_settings = {"stability": 0.3, "similarity_boost": 0.85, "style": 0.85, "use_speaker_boost": True}
            elif style == "mocking":
                voice_settings = {"stability": 0.35, "similarity_boost": 0.8, "style": 0.9, "use_speaker_boost": True}
            elif style == "dramatic":
                voice_settings = {"stability": 0.5, "similarity_boost": 0.9, "style": 0.7, "use_speaker_boost": True}
            elif style == "conclusive":
                voice_settings = {"stability": 0.7, "similarity_boost": 0.8, "style": 0.4, "use_speaker_boost": True}
            else:
                voice_settings = {"stability": 0.5, "similarity_boost": 0.85, "style": 0.6, "use_speaker_boost": True}
        
        payload = {
            "text": emotion_text,
            "model_id": self.model_id,  # Use model from .env
            "voice_settings": voice_settings
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                print(f"    ✅ Saved to: {output_path}")
                return True
            else:
                print(f"    ❌ API error {resp.status_code}: {resp.text[:100]}")
                return False
        except Exception as e:
            print(f"    ❌ Request failed: {e}")
            return False
    
    def create_final_video(self, planned_snarks):
        """Create final video with snarks mixed in"""
        
        print("\n🎬 Creating final video with snarks...")
        
        # Generate audio files
        audio_files = []
        for i, snark in enumerate(planned_snarks):
            # Store in the designated directory
            filename = self.snark_storage / f"scene_003_snark_{i+1}_{snark['style']}.mp3"
            
            if self.generate_elevenlabs_audio(
                snark["text"], 
                snark["style"], 
                snark["emotion"],
                str(filename)
            ):
                audio_files.append({
                    "file": str(filename),
                    "time": snark["time"],
                    "text": snark["text"]
                })
        
        # Extract and normalize original audio
        print("\n🎚️ Processing audio...")
        subprocess.run([
            "ffmpeg", "-y", "-i", self.video_path,
            "-vn", "-acodec", "pcm_s16le",
            "original_audio.wav"
        ], capture_output=True)
        
        # Load and normalize
        original = AudioSegment.from_file("original_audio.wav")
        print(f"  Original level: {original.dBFS:.1f} dBFS")
        
        # Normalize
        normalized = normalize(original)
        target_dBFS = -18  # Slightly louder for finale
        change = target_dBFS - normalized.dBFS
        normalized = normalized.apply_gain(change)
        print(f"  Normalized to: {normalized.dBFS:.1f} dBFS")
        
        # Mix in snarks with smooth ducking
        mixed = normalized
        
        for audio_info in audio_files:
            if not os.path.exists(audio_info["file"]):
                continue
            
            snark_audio = AudioSegment.from_mp3(audio_info["file"])
            snark_audio = snark_audio.apply_gain(4)  # +4dB boost for clarity
            
            position_ms = int(audio_info["time"] * 1000)
            duration_ms = len(snark_audio)
            
            # Smooth ducking
            fade_ms = 350
            duck_db = -10
            
            # Calculate boundaries
            fade_in_start = max(0, position_ms - fade_ms)
            fade_out_end = min(len(mixed), position_ms + duration_ms + fade_ms)
            
            # Apply ducking
            before = mixed[:fade_in_start]
            fade_in = mixed[fade_in_start:position_ms]
            during = mixed[position_ms:position_ms + duration_ms]
            fade_out = mixed[position_ms + duration_ms:fade_out_end]
            after = mixed[fade_out_end:]
            
            if len(fade_in) > 0:
                fade_in = fade_in.fade(to_gain=duck_db, start=0, duration=len(fade_in))
            
            during = during + duck_db
            
            if len(fade_out) > 0:
                fade_out = (fade_out + duck_db).fade(from_gain=duck_db, start=0, duration=len(fade_out))
            
            # Reconstruct and overlay
            mixed = before + fade_in + during + fade_out + after
            mixed = mixed.overlay(snark_audio, position=position_ms)
            
            print(f"  ✅ Added at {audio_info['time']:.1f}s: \"{audio_info['text']}\"")
        
        # Export mixed audio
        mixed_path = "scene_003_mixed.wav"
        mixed.export(mixed_path, format="wav")
        
        # Create final video
        output_path = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/original/scene_003_snarked.mp4"
        
        print("\n📹 Combining with video...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", self.video_path,
            "-i", mixed_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "320k",  # High quality audio
            "-af", "loudnorm=I=-14:LRA=11:TP=-1.0",  # Slightly louder for finale
            output_path
        ], capture_output=True)
        
        # Cleanup
        for f in ["original_audio.wav", mixed_path]:
            if os.path.exists(f):
                os.remove(f)
        
        print(f"\n✅ COMPLETE: {output_path}")
        
        return output_path, audio_files

def main():
    print("=" * 70)
    print("🎬 SCENE_003 SNARKER - ORIGINAL QUALITY")
    print("=" * 70)
    
    snarker = Scene003Snarker()
    
    # Analyze silence
    gaps = snarker.analyze_silence()
    
    print(f"\n📊 Silence gaps found:")
    for i, gap in enumerate(gaps, 1):
        print(f"  {i}. {gap['start']:.1f}-{gap['end']:.1f}s ({gap['duration']:.1f}s) - After: \"{gap['after'][:20]}...\"")
    
    # Plan snarks
    planned = snarker.plan_snarks(gaps)
    
    print(f"\n💬 Planned {len(planned)} snarks:")
    for i, snark in enumerate(planned, 1):
        print(f"  {i}. {snark['time']:.1f}s: \"{snark['text']}\" ({snark['emotion']})")
    
    # Create final video
    output_path, audio_files = snarker.create_final_video(planned)
    
    # Generate report
    report = {
        "scene": "scene_003",
        "video": "ORIGINAL quality (not downsampled)",
        "duration": snarker.scene_duration,
        "model": snarker.model_id,
        "storage_path": str(snarker.snark_storage),
        "snarks": [
            {
                "time": s["time"],
                "text": s["text"],
                "style": s["style"],
                "emotion": s["emotion"],
                "gap_duration": s["gap"]["duration"],
                "file": f"scene_003_snark_{i+1}_{s['style']}.mp3"
            }
            for i, s in enumerate(planned)
        ],
        "audio_settings": {
            "normalization": "-18 dBFS",
            "snark_boost": "+4 dB",
            "ducking": "-10 dB with 350ms fade",
            "final_loudness": "I=-14 LUFS (louder for finale)"
        }
    }
    
    report_path = snarker.snark_storage / "scene_003_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✨ SCENE_003 COMPLETE!")
    print("=" * 70)
    print(f"• Model: {snarker.model_id}")
    print(f"• Video: ORIGINAL quality")
    print(f"• Snarks: {len(audio_files)} generated")
    print(f"• Storage: {snarker.snark_storage}")
    print(f"• Output: {output_path}")
    print(f"• Report: {report_path}")

if __name__ == "__main__":
    main()