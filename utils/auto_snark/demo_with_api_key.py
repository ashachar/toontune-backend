#!/usr/bin/env python3
"""
ElevenLabs v3 Demo - Prompts for API key
"""

import os
import getpass
from elevenlabs import ElevenLabs

def create_demo_with_key():
    print("=" * 70)
    print("🎭 ELEVENLABS V3 CYNICAL NARRATOR DEMO")
    print("=" * 70)
    
    # Check environment first
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    
    if not api_key:
        print("\n📝 Please enter your ElevenLabs API key:")
        print("   (It will be hidden as you type)")
        api_key = getpass.getpass("   API Key: ").strip()
    else:
        print("✅ Using API key from environment")
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    print(f"\n✅ API key received (length: {len(api_key)})")
    
    try:
        # Initialize client with the key
        client = ElevenLabs(api_key=api_key)
        print("✅ Client initialized")
        
        # Test with a simple cynical remark
        print("\n🎙️ Generating cynical audio...")
        
        snarks = [
            ("Oh good, another musical number. How original.", "v3_snark_1.mp3"),
            ("Yes, because ABC is such complex knowledge.", "v3_snark_2.mp3"),
            ("We get it. You can repeat three syllables.", "v3_snark_3.mp3")
        ]
        
        generated = []
        
        for text, filename in snarks[:1]:  # Start with just one to test
            print(f"\n📝 Text: '{text}'")
            
            try:
                # Generate audio
                audio_generator = client.text_to_speech.convert(
                    text=text,
                    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                    model_id="eleven_turbo_v2_5",  # Fast turbo model
                    voice_settings={
                        "stability": 0.3,
                        "similarity_boost": 0.7,
                        "style": 0.8,
                        "use_speaker_boost": True
                    },
                    output_format="mp3_44100_128"
                )
                
                # Convert generator to bytes
                audio_bytes = b"".join(audio_generator)
                
                # Save the file
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                
                size_kb = os.path.getsize(filename) / 1024
                print(f"✅ Saved: {filename} ({size_kb:.1f} KB)")
                generated.append(filename)
                
            except Exception as e:
                print(f"❌ Error generating audio: {e}")
        
        if generated:
            print("\n" + "=" * 70)
            print("✅ SUCCESS! Generated cynical audio files:")
            for f in generated:
                print(f"   • {f}")
            
            print("\n🎧 To play the audio:")
            print(f"   afplay {generated[0]}")
            
            print("\n🎬 Next steps:")
            print("1. Export your API key for future use:")
            print(f"   export ELEVENLABS_API_KEY='{api_key[:8]}...'")
            print("2. Run the full narrator:")
            print("   python snark_narrator.py --video scene_001.mp4 --transcript do_re_mi_transcript.json --out output.mp4")
            
            return True
        else:
            print("\n❌ No audio files were generated")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. No API credits available")
        print("3. Network connection issue")
        return False

if __name__ == "__main__":
    success = create_demo_with_key()
    
    if not success:
        print("\n💡 To get an API key:")
        print("1. Go to https://elevenlabs.io")
        print("2. Sign up for an account")
        print("3. Go to Profile → API Keys")
        print("4. Copy your API key")
        print("5. Set it: export ELEVENLABS_API_KEY='your-key-here'")