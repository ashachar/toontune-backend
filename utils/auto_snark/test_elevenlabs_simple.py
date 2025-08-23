#!/usr/bin/env python3
"""
Simple test of ElevenLabs v3 API
"""

import os
from elevenlabs import ElevenLabs

def test_simple():
    # You mentioned you added the key - trying to access it
    # Note: In production, always use environment variables
    
    print("Testing ElevenLabs v3 connection...")
    
    try:
        # Initialize client (will use ELEVENLABS_API_KEY env var if set)
        client = ElevenLabs()
        
        # Simple test text with emotion tags
        test_text = 'Oh good, another musical number. How original.'
        
        print(f"Generating audio for: '{test_text}'")
        
        # Generate with v3 features
        audio_generator = client.text_to_speech.convert(
            text=test_text,
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
            model_id="eleven_turbo_v2_5",  # Fast model
            output_format="mp3_44100_128"
        )
        
        # Save audio
        audio_bytes = b"".join(audio_generator)
        
        with open("test_v3_output.mp3", "wb") as f:
            f.write(audio_bytes)
        
        print("✅ Success! Audio saved to test_v3_output.mp3")
        
        # Get file size
        size_kb = os.path.getsize("test_v3_output.mp3") / 1024
        print(f"📊 File size: {size_kb:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure ELEVENLABS_API_KEY is set:")
        print("   export ELEVENLABS_API_KEY='your-key-here'")
        print("2. Check your API key is valid")
        print("3. Ensure you have API credits available")
        return False

if __name__ == "__main__":
    test_simple()