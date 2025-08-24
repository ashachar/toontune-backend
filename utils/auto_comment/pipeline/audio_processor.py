"""Audio processing and TTS generation."""

import os
import re
import json
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from pydub import AudioSegment
from pydub.effects import normalize

from utils.auto_comment.pipeline.models import Snark
from utils.auto_comment.pipeline.config import (
    ELEVENLABS_API_KEY, ELEVENLABS_MODEL, ELEVENLABS_VOICE_ID,
    ELEVENLABS_STABILITY, ELEVENLABS_SIMILARITY_BOOST,
    SNARK_STORAGE, TARGET_DBFS
)


class AudioProcessor:
    """Handles audio processing and TTS generation."""
    
    def __init__(self, video_path: Path, output_folder: Path):
        self.video_path = video_path
        self.output_folder = output_folder
        self.elevenlabs_key = ELEVENLABS_API_KEY
        self.snark_storage = SNARK_STORAGE
        self.snark_storage.mkdir(parents=True, exist_ok=True)
        self._load_existing_snarks()
        
    def _load_existing_snarks(self):
        """Load existing snark library."""
        self.existing_snarks = {}
        if self.snark_storage.exists():
            for audio_file in self.snark_storage.glob("*.mp3"):
                text = audio_file.stem.replace("_", " ")
                self.existing_snarks[text] = {
                    "text": text,
                    "file": str(audio_file),
                    "emotion": "neutral",
                    "duration": self._estimate_duration(text)
                }
        print(f"📚 Existing snark library: {len(self.existing_snarks)} available")
    
    def generate_speech_with_elevenlabs(self, snark: Snark) -> Optional[str]:
        """Convert snark text to speech."""
        estimated_duration = self._estimate_duration(snark.text)
        if estimated_duration > snark.gap_duration:
            print(f"  ⚠️ May need pause for: '{snark.text}' "
                  f"(est. {estimated_duration:.1f}s > gap {snark.gap_duration:.1f}s)")
        
        # Check existing library
        text_lower = snark.text.lower()
        if text_lower in self.existing_snarks:
            existing_file = self.existing_snarks[text_lower]["file"]
            if os.path.exists(existing_file):
                print(f"  ♻️ Found in library: {Path(existing_file).name}")
                return existing_file
        
        # Create filename from text
        filename = re.sub(r'[^\w\s]', '', snark.text.lower())
        filename = re.sub(r'\s+', '_', filename)[:50] + ".mp3"
        audio_path = self.snark_storage / filename
        
        # Check if file exists
        if audio_path.exists():
            print(f"  ♻️ Reusing existing file: {filename}")
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
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        
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
            "unimpressed": "interested"
        }
        emotion = friendly_emotions.get(snark.emotion, snark.emotion)
        emotion_text = f'<emotion="{emotion}">{snark.text}</emotion>'
        
        payload = {
            "text": emotion_text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": ELEVENLABS_STABILITY,
                "similarity_boost": ELEVENLABS_SIMILARITY_BOOST
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
    
    def create_audio_with_speed_adjustments(
        self, snarks_with_pausing: List[Dict]
    ) -> Path:
        """Create audio track for video with speed adjustments."""
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
        change = TARGET_DBFS - normalized.dBFS
        normalized = normalized.apply_gain(change)
        
        # Sort snarks by time
        snarks_with_pausing.sort(key=lambda x: x["snark"].time)
        
        # Simply overlay snarks at their positions
        mixed = normalized
        for item in snarks_with_pausing:
            snark = item["snark"]
            snark_audio = item["audio"].apply_gain(2)
            position_ms = int(snark.time * 1000)
            
            # Overlay snark at its position
            mixed = mixed.overlay(snark_audio, position=position_ms)
            
            if item["needs_slowdown"]:
                speed_pct = item["speed_factor"] * 100
                print(f"  🐢 Slowing video to {speed_pct:.1f}% speed at {snark.time:.1f}s for: \"{snark.text}\"")
                print(f"     Gap: {item['gap_duration']:.2f}s, Remark: {item['duration']:.2f}s")
            else:
                print(f"  ✅ Normal speed at {snark.time:.1f}s: \"{snark.text}\"")
        
        # Export mixed audio
        mixed_path = self.output_folder / "mixed_audio.wav"
        mixed.export(mixed_path, format="wav")
        
        return mixed_path
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate TTS duration from text."""
        word_count = len(text.split())
        return word_count * 0.45 + 0.5  # ~450ms per word + 500ms padding