#!/usr/bin/env python3
"""
Run ElevenLabs v3 demo with API key from .env file
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
import subprocess
import json

def run_v3_demo():
    print("=" * 70)
    print("🎭 ELEVENLABS V3 CYNICAL NARRATOR DEMO")
    print("=" * 70)
    
    # Load .env file - try multiple locations
    env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path(__file__).parent / ".env",  # Script directory
        Path(__file__).parent.parent.parent.parent.parent / ".env",  # Backend root
        Path.home() / "Desktop/Amir/Projects/Personal/toontune/backend/.env"  # Full path
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            print(f"📁 Loading .env from: {env_path}")
            load_dotenv(env_path)
            env_loaded = True
            break
    
    if not env_loaded:
        print("⚠️ No .env file found, checking environment variables...")
    
    # Get API key
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    
    if not api_key:
        print("❌ ELEVENLABS_API_KEY not found in .env or environment")
        print("\nPlease create a .env file with:")
        print("ELEVENLABS_API_KEY=your-key-here")
        return False
    
    print(f"✅ API key loaded (length: {len(api_key)})")
    
    try:
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=api_key)
        print("✅ ElevenLabs client initialized")
        
        # Generate cynical snarks with v3 features
        snarks = [
            {
                "id": 1,
                "time": 3.8,
                "text": "Oh good, another musical number. How original.",
                "style": "sarcastic"
            },
            {
                "id": 2,
                "time": 16.8,
                "text": "Yes, because ABC is such complex knowledge.",
                "style": "condescending"
            },
            {
                "id": 3,
                "time": 25.5,
                "text": "We get it. You can repeat three syllables.",
                "style": "mocking"
            },
            {
                "id": 4,
                "time": 41.8,
                "text": "Easier? This is your idea of teaching?",
                "style": "skeptical"
            },
            {
                "id": 5,
                "time": 48.0,
                "text": "Revolutionary. A deer is... a deer.",
                "style": "deadpan"
            },
            {
                "id": 6,
                "time": 52.5,
                "text": "Mi, the narcissism is showing.",
                "style": "sarcastic"
            }
        ]
        
        print(f"\n🎙️ Generating {len(snarks)} cynical remarks...")
        generated_files = []
        
        for snark in snarks:
            print(f"\n{snark['id']}. Generating: '{snark['text'][:40]}...'")
            print(f"   Style: {snark['style']}")
            
            try:
                # Adjust voice settings based on style
                voice_settings = {
                    "stability": 0.45,
                    "similarity_boost": 0.75,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
                
                if snark['style'] == 'sarcastic':
                    voice_settings["stability"] = 0.25
                    voice_settings["style"] = 0.85
                elif snark['style'] == 'deadpan':
                    voice_settings["stability"] = 0.85
                    voice_settings["style"] = 0.15
                elif snark['style'] == 'condescending':
                    voice_settings["stability"] = 0.35
                    voice_settings["style"] = 0.75
                elif snark['style'] == 'mocking':
                    voice_settings["stability"] = 0.3
                    voice_settings["style"] = 0.8
                
                # Generate audio
                audio_generator = client.text_to_speech.convert(
                    text=snark['text'],
                    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                    model_id="eleven_turbo_v2_5",  # Using turbo for speed
                    voice_settings=voice_settings,
                    output_format="mp3_44100_128"
                )
                
                # Save audio
                filename = f"v3_snark_{snark['id']}_{snark['style']}.mp3"
                audio_bytes = b"".join(audio_generator)
                
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                
                size_kb = os.path.getsize(filename) / 1024
                print(f"   ✅ Saved: {filename} ({size_kb:.1f} KB)")
                generated_files.append((snark['time'], filename, snark['text']))
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        if generated_files:
            print("\n" + "=" * 70)
            print(f"✅ Successfully generated {len(generated_files)} audio files!")
            
            # Mix with video
            print("\n🎬 Mixing with Do-Re-Mi video...")
            mix_with_video(generated_files)
            
            # Create summary
            print("\n📊 SUMMARY:")
            print(f"  • Generated: {len(generated_files)} cynical remarks")
            print(f"  • Styles used: sarcastic, deadpan, condescending, mocking")
            print(f"  • Voice: Rachel (21m00Tcm4TlvDq8ikWAM)")
            print(f"  • Model: eleven_turbo_v2_5")
            
            # Save report
            report = {
                "api": "ElevenLabs v3",
                "model": "eleven_turbo_v2_5",
                "voice": "Rachel",
                "snarks_generated": len(generated_files),
                "files": [f[1] for f in generated_files],
                "styles": ["sarcastic", "deadpan", "condescending", "mocking", "skeptical"]
            }
            
            with open("v3_demo_report.json", "w") as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📝 Report saved: v3_demo_report.json")
            
            # Play sample
            if generated_files:
                print(f"\n🎧 Playing first snark...")
                subprocess.run(["afplay", generated_files[0][1]], capture_output=True)
            
            return True
        else:
            print("\n❌ No audio files generated")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def mix_with_video(audio_files):
    """Mix generated audio with Do-Re-Mi video"""
    
    input_video = "/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/videos/do_re_mi/scenes/downsampled/scene_001.mp4"
    output_video = "do_re_mi_elevenlabs_v3_final.mp4"
    
    if not os.path.exists(input_video):
        print(f"⚠️ Video not found: {input_video}")
        return
    
    print(f"📹 Processing scene_001.mp4...")
    
    # Extract original audio
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le",
        "original_audio.wav"
    ], capture_output=True)
    
    # Build mixing command
    cmd = ["ffmpeg", "-y", "-i", "original_audio.wav"]
    
    for _, audio_file, _ in audio_files:
        cmd.extend(["-i", audio_file])
    
    # Create filter for mixing with ducking
    filter_str = ""
    for i, (time, _, _) in enumerate(audio_files):
        if i == 0:
            filter_str += f"[0]volume=enable='between(t,{time},{time+3})':volume=0.2[a0];"
            filter_str += f"[{i+1}]adelay={int(time*1000)}|{int(time*1000)}[d{i}];"
            filter_str += f"[a0][d{i}]amix=inputs=2:duration=first[m{i}];"
        else:
            filter_str += f"[m{i-1}]volume=enable='between(t,{time},{time+3})':volume=0.2[a{i}];"
            filter_str += f"[{i+1}]adelay={int(time*1000)}|{int(time*1000)}[d{i}];"
            if i == len(audio_files) - 1:
                filter_str += f"[a{i}][d{i}]amix=inputs=2:duration=first"
            else:
                filter_str += f"[a{i}][d{i}]amix=inputs=2:duration=first[m{i}];"
    
    cmd.extend([
        "-filter_complex", filter_str,
        "-ac", "1",
        "mixed_audio.wav"
    ])
    
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0:
        # Combine with video
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_video,
            "-i", "mixed_audio.wav",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v", "-map", "1:a",
            output_video
        ], capture_output=True)
        
        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"✅ Created final video: {output_video} ({size_mb:.2f} MB)")
            print(f"\n🎬 Play with: open {output_video}")
    
    # Cleanup
    for f in ["original_audio.wav", "mixed_audio.wav"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    print("🔍 Looking for .env file with ELEVENLABS_API_KEY...")
    success = run_v3_demo()
    
    if success:
        print("\n🎉 ElevenLabs v3 demo complete with cynical narration!")
    else:
        print("\n💡 Check that your .env file contains:")
        print("ELEVENLABS_API_KEY=your-actual-key-here")