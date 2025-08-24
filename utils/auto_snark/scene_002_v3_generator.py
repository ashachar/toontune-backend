#!/usr/bin/env python3
"""
Scene_002 with ElevenLabs v3 ONLY
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory
backend_env = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/.env")
if backend_env.exists():
    load_dotenv(backend_env)
    print(f"✅ Loaded .env from: {backend_env}")

# Also try local .env
load_dotenv()

def generate_elevenlabs_v3(text, output_path, style="deadpan"):
    """Generate audio using ElevenLabs v3 API"""
    
    api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    
    if not api_key:
        print("❌ ElevenLabs API key not found!")
        print("Please add ELEVEN_API_KEY to your .env file")
        # Create placeholder
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=1000)
        silence.export(output_path, format="mp3")
        return False
    
    print(f"  🎤 Generating with ElevenLabs v3: \"{text}\"")
    
    import requests
    
    # Rachel voice for sarcasm
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "accept": "audio/mpeg",
        "content-type": "application/json",
        "xi-api-key": api_key
    }
    
    # Configure emotion and voice settings based on style
    if style == "deadpan":
        emotion_text = f'<emotion="deadpan">{text}</emotion>'
        voice_settings = {
            "stability": 0.85,
            "similarity_boost": 0.75,
            "style": 0.1,
            "use_speaker_boost": True
        }
    elif style == "sarcastic":
        emotion_text = f'<emotion="sarcastic">{text}</emotion>'
        voice_settings = {
            "stability": 0.3,
            "similarity_boost": 0.85,
            "style": 0.85,
            "use_speaker_boost": True
        }
    elif style == "mocking":
        emotion_text = f'<emotion="mocking">{text}</emotion>'
        voice_settings = {
            "stability": 0.35,
            "similarity_boost": 0.8,
            "style": 0.9,
            "use_speaker_boost": True
        }
    elif "?" in text:
        emotion_text = f'<emotion="skeptical">{text}</emotion>'
        voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.7,
            "use_speaker_boost": True
        }
    else:
        emotion_text = f'<emotion="condescending">{text}</emotion>'
        voice_settings = {
            "stability": 0.4,
            "similarity_boost": 0.85,
            "style": 0.65,
            "use_speaker_boost": True
        }
    
    payload = {
        "text": emotion_text,
        "model_id": "eleven_turbo_v2_5",  # ElevenLabs v3 model
        "voice_settings": voice_settings
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            print(f"    ✅ Generated: {style} emotion")
            return True
        else:
            print(f"    ❌ API error {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"    ❌ Request failed: {e}")
        return False

def main():
    print("=" * 70)
    print("🎬 SCENE_002 SNARKS - ELEVENLABS V3 ONLY")
    print("=" * 70)
    
    # Check API key
    api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        print(f"✅ API Key found: {api_key[:8]}...")
    else:
        print("⚠️ WARNING: No API key found - will create placeholders")
    
    # Snarks for scene_002
    snarks = [
        {"text": "Act two begins.", "style": "deadpan", "file": "scene_002_v3_1.mp3"},
        {"text": "Still running apparently.", "style": "sarcastic", "file": "scene_002_v3_2.mp3"},
        {"text": "Profound metaphor.", "style": "mocking", "file": "scene_002_v3_3.mp3"},
        {"text": "The suspense builds.", "style": "sarcastic", "file": "scene_002_v3_4.mp3"},
        {"text": "Déjà vu much?", "style": "skeptical", "file": "scene_002_v3_5.mp3"},
        {"text": "Shocking.", "style": "deadpan", "file": "scene_002_v3_6.mp3"},
    ]
    
    print(f"\n📝 Generating {len(snarks)} snarks...")
    
    success_count = 0
    for snark in snarks:
        if generate_elevenlabs_v3(snark["text"], snark["file"], snark["style"]):
            success_count += 1
    
    print(f"\n✅ Generated {success_count}/{len(snarks)} audio files")
    
    if success_count == len(snarks):
        print("\n🎯 All snarks generated with ElevenLabs v3!")
        print("💰 Estimated cost: ~$0.05")
    else:
        print("\n⚠️ Some snarks failed - check API key and quota")

if __name__ == "__main__":
    main()