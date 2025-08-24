#!/usr/bin/env python3
"""
Universal Snark Generator for any scene
- Uses ELEVENLABS_MODEL from .env
- Names files based on text content
- Stores in uploads/assets/sounds/snark_remarks/
"""

import os
import sys
import subprocess
import json
import re
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

class SnarkGenerator:
    def __init__(self, scene_number, use_original=True):
        self.scene_number = scene_number
        
        # Storage directory
        self.snark_storage = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/sounds/snark_remarks")
        self.snark_storage.mkdir(parents=True, exist_ok=True)
        
        # Get model from .env
        self.model_id = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Load scene metadata
        metadata_path = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/metadata/scenes.json")
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Find scene info
        self.scene_info = None
        for scene in metadata["scenes"]:
            if scene["scene_number"] == scene_number:
                self.scene_info = scene
                break
        
        if not self.scene_info:
            raise ValueError(f"Scene {scene_number} not found")
        
        # Set video path (original or downsampled)
        if use_original:
            self.video_path = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend") / self.scene_info["original_path"]
        else:
            self.video_path = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend") / self.scene_info["downsampled_path"]
        
        self.scene_start = self.scene_info["start_seconds"]
        self.scene_end = self.scene_info["end_seconds"]
        self.scene_duration = self.scene_info["duration"]
        
        print(f"📍 Scene {scene_number}: {self.scene_duration:.1f}s")
        print(f"📍 Model: {self.model_id}")
        print(f"📍 Video: {'ORIGINAL' if use_original else 'DOWNSAMPLED'}")
    
    def text_to_filename(self, text):
        """Convert snark text to a valid filename"""
        # Remove emotion tags
        clean_text = re.sub(r'<emotion="[^"]+">|</emotion>', '', text)
        
        # Convert to lowercase and replace spaces with underscores
        filename = clean_text.lower().strip()
        
        # Remove punctuation except periods
        filename = re.sub(r'[^\w\s\.]', '', filename)
        filename = re.sub(r'\s+', '_', filename)
        
        # Remove trailing periods
        filename = filename.rstrip('.')
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:50]
        
        return filename + ".mp3"
    
    def generate_elevenlabs_audio(self, text, emotion="neutral", style="neutral"):
        """Generate audio using ElevenLabs API"""
        
        if not self.api_key:
            print(f"❌ No API key for: {text}")
            return None
        
        # Create filename based on text content
        filename = self.text_to_filename(text)
        output_path = self.snark_storage / filename
        
        # Check if already exists
        if output_path.exists():
            print(f"  ♻️ Reusing: {filename}")
            return str(output_path)
        
        print(f"  🎤 Generating: \"{text}\" → {filename}")
        
        # Rachel voice
        voice_id = "21m00Tcm4TlvDq8ikWAM"
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "accept": "audio/mpeg",
            "content-type": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Add emotion tags
        emotion_text = f'<emotion="{emotion}">{text}</emotion>'
        
        # Configure voice settings for different models
        if self.model_id == "eleven_v3":
            # Simplified settings for v3
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        else:
            # Full settings for other models
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.5,
                "use_speaker_boost": True
            }
        
        payload = {
            "text": emotion_text,
            "model_id": self.model_id,
            "voice_settings": voice_settings
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                print(f"    ✅ Saved: {filename}")
                return str(output_path)
            else:
                print(f"    ❌ API error {resp.status_code}")
                return None
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return None
    
    def analyze_scene_silence(self):
        """Analyze scene for silence gaps"""
        
        print("\n🔍 Analyzing for silence...")
        
        # Extract audio
        temp_audio = f"scene_{self.scene_number}_temp.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050",
            temp_audio
        ], capture_output=True)
        
        # Analyze with pydub
        audio = AudioSegment.from_file(temp_audio)
        
        # Detect silence (more lenient for scene_003)
        silent_ranges = detect_silence(
            audio,
            min_silence_len=600,  # 0.6 seconds
            silence_thresh=-35,  # Less strict
            seek_step=100
        )
        
        # Clean up
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        # Convert to seconds
        gaps = []
        for start_ms, end_ms in silent_ranges:
            duration = (end_ms - start_ms) / 1000
            if duration >= 0.8:
                gaps.append({
                    "start": start_ms / 1000,
                    "end": end_ms / 1000,
                    "duration": duration
                })
        
        print(f"  Found {len(gaps)} silence gaps")
        return gaps
    
    def generate_snarks_for_scene(self, max_snarks=6):
        """Generate snarks for the scene"""
        
        # Analyze silence
        gaps = self.analyze_scene_silence()
        
        if not gaps:
            print("❌ No silence gaps found")
            return []
        
        # Pre-defined snarks pool
        snark_pool = [
            ("Riveting.", "bored"),
            ("Fascinating.", "sarcastic"),
            ("How original.", "deadpan"),
            ("Groundbreaking.", "mocking"),
            ("This is fine.", "resigned"),
            ("Peak cinema.", "sarcastic"),
            ("Art.", "deadpan"),
            ("Stunning.", "unimpressed"),
            ("Wow.", "monotone"),
            ("Sure.", "skeptical"),
        ]
        
        # Generate snarks for gaps
        generated = []
        snark_index = 0
        
        for gap in gaps[:max_snarks]:
            if gap["duration"] < 0.8:
                continue
            
            # Pick snark from pool
            text, emotion = snark_pool[snark_index % len(snark_pool)]
            snark_index += 1
            
            # Generate audio
            audio_path = self.generate_elevenlabs_audio(text, emotion)
            
            if audio_path:
                generated.append({
                    "text": text,
                    "emotion": emotion,
                    "file": audio_path,
                    "time": gap["start"] + 0.2,
                    "gap_duration": gap["duration"]
                })
        
        return generated
    
    def create_final_video(self, snarks, output_suffix="snarked"):
        """Mix snarks into video"""
        
        if not snarks:
            print("❌ No snarks to add")
            return None
        
        print("\n🎬 Creating final video...")
        
        # Extract audio
        subprocess.run([
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le",
            "original_audio.wav"
        ], capture_output=True)
        
        # Load and normalize
        original = AudioSegment.from_file("original_audio.wav")
        normalized = normalize(original)
        target_dBFS = -18
        change = target_dBFS - normalized.dBFS
        normalized = normalized.apply_gain(change)
        
        # Mix in snarks
        mixed = normalized
        
        for snark in snarks:
            if not os.path.exists(snark["file"]):
                continue
            
            snark_audio = AudioSegment.from_mp3(snark["file"])
            snark_audio = snark_audio.apply_gain(3)  # Boost
            
            position_ms = int(snark["time"] * 1000)
            
            # Apply ducking
            fade_ms = 400
            duck_db = -10
            
            fade_in_start = max(0, position_ms - fade_ms)
            fade_out_end = min(len(mixed), position_ms + len(snark_audio) + fade_ms)
            
            before = mixed[:fade_in_start]
            fade_in = mixed[fade_in_start:position_ms]
            during = mixed[position_ms:position_ms + len(snark_audio)]
            fade_out = mixed[position_ms + len(snark_audio):fade_out_end]
            after = mixed[fade_out_end:]
            
            if len(fade_in) > 0:
                fade_in = fade_in.fade(to_gain=duck_db, start=0, duration=len(fade_in))
            
            during = during + duck_db
            
            if len(fade_out) > 0:
                fade_out = (fade_out + duck_db).fade(from_gain=duck_db, start=0, duration=len(fade_out))
            
            mixed = before + fade_in + during + fade_out + after
            mixed = mixed.overlay(snark_audio, position=position_ms)
            
            print(f"  ✅ Added at {snark['time']:.1f}s: \"{snark['text']}\"")
        
        # Export
        mixed_path = f"scene_{self.scene_number}_mixed.wav"
        mixed.export(mixed_path, format="wav")
        
        # Create final video
        output_path = self.video_path.parent / f"scene_{self.scene_number:03d}_{output_suffix}.mp4"
        
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(self.video_path),
            "-i", mixed_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
            str(output_path)
        ], capture_output=True)
        
        # Cleanup
        for f in ["original_audio.wav", mixed_path]:
            if os.path.exists(f):
                os.remove(f)
        
        print(f"✅ Output: {output_path}")
        return str(output_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate snarks for any scene")
    parser.add_argument("scene", type=int, help="Scene number (1, 2, 3, etc.)")
    parser.add_argument("--downsampled", action="store_true", help="Use downsampled video")
    parser.add_argument("--max-snarks", type=int, default=6, help="Maximum snarks to generate")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"🎬 SNARK GENERATOR - SCENE {args.scene}")
    print("=" * 70)
    
    # Generate snarks
    generator = SnarkGenerator(args.scene, use_original=not args.downsampled)
    snarks = generator.generate_snarks_for_scene(args.max_snarks)
    
    if snarks:
        print(f"\n✅ Generated {len(snarks)} snarks")
        
        # Create video
        output = generator.create_final_video(snarks)
        
        # Report
        print("\n📊 Summary:")
        for snark in snarks:
            filename = Path(snark["file"]).name
            print(f"  • {snark['time']:.1f}s: \"{snark['text']}\" → {filename}")
        
        # Calculate cost
        total_chars = sum(len(s["text"]) for s in snarks)
        cost = total_chars * 0.0003
        print(f"\n💰 Estimated cost: ${cost:.3f}")
    else:
        print("❌ No snarks generated")

if __name__ == "__main__":
    main()