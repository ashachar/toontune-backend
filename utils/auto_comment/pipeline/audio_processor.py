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
        self.snark_storage = SNARK_STORAGE
        self.snark_storage.mkdir(parents=True, exist_ok=True)
    
    def process_snarks_audio(
        self, snarks: List[Snark]
    ) -> List[Dict]:
        """Process audio for all snarks with pausing decisions."""
        snarks_with_pausing = []
        
        for snark in snarks:
            # Get or generate audio
            audio_file = self._get_or_generate_audio(snark.text)
            if not audio_file:
                continue
            
            # Load audio and get duration
            audio = AudioSegment.from_file(audio_file)
            duration = len(audio) / 1000.0
            
            # Determine if we need to slow down video
            gap_duration = snark.gap_duration
            needs_slowdown = duration > gap_duration
            
            # Calculate speed factor
            if needs_slowdown:
                speed_factor = gap_duration / duration
            else:
                speed_factor = 1.0
            
            snarks_with_pausing.append({
                "snark": snark,
                "audio": audio,
                "audio_file": str(audio_file),
                "duration": duration,
                "gap_duration": gap_duration,
                "needs_slowdown": needs_slowdown,
                "speed_factor": speed_factor
            })
        
        return snarks_with_pausing
    
    def _get_or_generate_audio(self, text: str) -> Optional[Path]:
        """Get existing or generate new audio."""
        # Clean filename
        filename = re.sub(r'[^a-zA-Z0-9_]', '', text.lower().replace(' ', '_'))[:50]
        filename = f"{filename}.mp3"
        audio_path = self.snark_storage / filename
        
        # Check if exists
        if audio_path.exists():
            print(f"  ♻️ Reusing existing file: {filename}")
            return audio_path
        
        # Estimate duration warning (generic since we don't have gap_duration here)
        estimated_duration = self._estimate_duration(text)
        if estimated_duration > 0.9:
            print(f"  ⚠️ May need pause for: '{text}' (est. {estimated_duration:.1f}s > gap 0.9s)")
        
        # Generate with ElevenLabs
        print(f"  🎤 Generating: \"{text}\" → {filename}")
        
        if not ELEVENLABS_API_KEY:
            print(f"    ⚠️ No ElevenLabs API key")
            return None
        
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        payload = {
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {
                "stability": ELEVENLABS_STABILITY,
                "similarity_boost": ELEVENLABS_SIMILARITY_BOOST,
                "style": 0.5,
                "use_speaker_boost": True
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
        """Create audio track that matches video with speed adjustments."""
        # Extract original audio
        audio_path = self.output_folder / "original_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le",
            str(audio_path)
        ], capture_output=True)
        
        # Load original audio
        original = AudioSegment.from_file(audio_path)
        normalized = normalize(original)
        change = TARGET_DBFS - normalized.dBFS
        normalized = normalized.apply_gain(change)
        
        # Sort snarks by time
        snarks_with_pausing.sort(key=lambda x: x["snark"].time)
        
        # Build new audio track accounting for speed changes
        mixed = AudioSegment.empty()
        source_pos = 0  # Position in original audio
        output_pos = 0  # Position in output audio
        
        for item in snarks_with_pausing:
            snark = item["snark"]
            gap_start_ms = int(snark.time * 1000)
            gap_duration_ms = int(item["gap_duration"] * 1000)
            speed_factor = item["speed_factor"]
            remark_audio = item["audio"].apply_gain(2)  # Boost remark volume
            
            # Add audio before the gap (normal speed)
            if source_pos < gap_start_ms:
                segment_before = normalized[source_pos:gap_start_ms]
                mixed += segment_before
                output_pos += len(segment_before)
            
            # Handle the gap
            if speed_factor < 1.0:
                # Video is slowed, so we need to stretch this audio segment
                # OR better: just use silence + remark
                output_duration_ms = int(gap_duration_ms / speed_factor)
                
                # Create silent gap of the stretched duration
                silence = AudioSegment.silent(duration=output_duration_ms)
                
                # Center the remark in the stretched gap
                remark_offset = (output_duration_ms - len(remark_audio)) // 2
                if remark_offset < 0:
                    remark_offset = 0
                
                # Overlay remark on silence
                gap_audio = silence.overlay(remark_audio, position=remark_offset)
                mixed += gap_audio
                output_pos += output_duration_ms
            else:
                # Normal speed - just overlay remark on original audio
                gap_segment = normalized[gap_start_ms:gap_start_ms + gap_duration_ms]
                
                # Center the remark in the gap
                remark_offset = (gap_duration_ms - len(remark_audio)) // 2
                if remark_offset < 0:
                    remark_offset = 0
                    
                gap_with_remark = gap_segment.overlay(remark_audio, position=remark_offset)
                mixed += gap_with_remark
                output_pos += gap_duration_ms
            
            # Update source position
            source_pos = gap_start_ms + gap_duration_ms
        
        # Add remaining audio after last gap
        if source_pos < len(normalized):
            mixed += normalized[source_pos:]
        
        # Export mixed audio
        mixed_path = self.output_folder / "mixed_audio_synced.wav"
        mixed.export(mixed_path, format="wav")
        
        print(f"  🎵 Audio track created with proper sync for speed adjustments")
        return mixed_path
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate TTS duration from text."""
        word_count = len(text.split())
        return word_count * 0.45 + 0.5  # ~450ms per word + 500ms padding