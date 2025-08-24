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
        
        # Create workspace
        self.workspace = Path("snark_workspace") / self.video_name
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Storage for generated snarks
        self.snark_storage = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/sounds/snark_remarks")
        self.snark_storage.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
        
        print(f"🎬 Pipeline initialized for: {self.video_name}")
        print(f"📁 Workspace: {self.workspace}")
    
    def extract_transcript(self) -> Dict:
        """Step 1: Extract transcript from video"""
        
        transcript_file = self.workspace / "transcript.json"
        
        # Check if already exists
        if transcript_file.exists():
            print("📝 Loading existing transcript...")
            with open(transcript_file) as f:
                return json.load(f)
        
        print("🎤 Extracting transcript with Whisper...")
        
        # Extract audio first
        audio_path = self.workspace / "audio.wav"
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
        """Step 2: Find silence gaps in audio"""
        
        print("🔍 Analyzing silence gaps...")
        
        # Extract audio if needed
        audio_path = self.workspace / "audio.wav"
        if not audio_path.exists():
            subprocess.run([
                "ffmpeg", "-y", "-i", str(self.video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "22050",
                str(audio_path)
            ], capture_output=True)
        
        # Analyze with pydub
        audio = AudioSegment.from_file(audio_path)
        
        # Detect silence
        silent_ranges = detect_silence(
            audio,
            min_silence_len=800,  # 0.8 seconds minimum
            silence_thresh=-38,
            seek_step=100
        )
        
        # Convert to SilenceGap objects
        gaps = []
        for start_ms, end_ms in silent_ranges:
            duration = (end_ms - start_ms) / 1000
            if duration >= 0.8:  # Only gaps long enough for snarks
                gaps.append(SilenceGap(
                    start=start_ms / 1000,
                    end=end_ms / 1000,
                    duration=duration
                ))
        
        print(f"✅ Found {len(gaps)} usable silence gaps")
        return gaps
    
    def generate_contextual_snarks(self, transcript: Dict, gaps: List[SilenceGap]) -> List[Snark]:
        """Step 3: Generate snarks based on transcript context"""
        
        print("🤖 Generating contextual snarks...")
        
        if not self.gemini_key:
            print("⚠️ No Gemini API key, using fallback snarks")
            return self._generate_fallback_snarks(gaps)
        
        snarks = []
        
        for gap in gaps[:6]:  # Limit to 6 snarks
            # Find context: what was said before this gap
            context = ""
            for seg in transcript["segments"]:
                if seg["end"] <= gap.start and seg["end"] > gap.start - 3:
                    context = seg["text"]
                    break
            
            # Generate snark with Gemini
            snark = self._generate_snark_with_ai(context, gap.duration)
            if snark:
                snarks.append(Snark(
                    text=snark["text"],
                    time=gap.start + 0.2,  # Small offset into gap
                    emotion=snark["emotion"],
                    context=context,
                    gap_duration=gap.duration
                ))
        
        print(f"✅ Generated {len(snarks)} contextual snarks")
        return snarks
    
    def _generate_snark_with_ai(self, context: str, max_duration: float) -> Optional[Dict]:
        """Generate a single snark using Gemini"""
        
        # Determine word limit based on duration
        max_words = int(max_duration * 2.5)  # ~2.5 words per second
        
        prompt = f"""
        You are a sarcastic narrator commenting on a video.
        
        Context (what was just said): "{context}"
        
        Generate ONE brief sarcastic comment (max {max_words} words).
        
        Return JSON:
        {{
            "text": "your sarcastic comment",
            "emotion": "choose: sarcastic/deadpan/mocking/bored/unimpressed"
        }}
        
        Examples:
        - If repetitive: "Third time's the charm."
        - If obvious: "Groundbreaking discovery."
        - If dramatic: "Oscar-worthy performance."
        - If boring: "Riveting."
        
        Be witty but not mean-spirited.
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
    
    def _generate_fallback_snarks(self, gaps: List[SilenceGap]) -> List[Snark]:
        """Fallback snarks when AI is not available"""
        
        fallback_pool = [
            ("Riveting.", "bored"),
            ("How original.", "deadpan"),
            ("This is fine.", "resigned"),
            ("Stunning.", "sarcastic"),
            ("Peak content.", "mocking"),
            ("Sure.", "unimpressed")
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
        
        # Create filename from text
        filename = re.sub(r'[^\w\s]', '', snark.text.lower())
        filename = re.sub(r'\s+', '_', filename)[:50] + ".mp3"
        audio_path = self.snark_storage / filename
        
        # Check if already exists
        if audio_path.exists():
            print(f"  ♻️ Reusing: {filename}")
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
        
        # Add emotion tags
        emotion_text = f'<emotion="{snark.emotion}">{snark.text}</emotion>'
        
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
        
        # Generate speech for all snarks
        for snark in snarks:
            audio_path = self.generate_speech_with_elevenlabs(snark)
            snark.audio_path = audio_path
        
        # Filter out snarks without audio
        valid_snarks = [s for s in snarks if hasattr(s, 'audio_path') and s.audio_path]
        
        if not valid_snarks:
            print("❌ No valid snarks to add")
            return str(self.video_path)
        
        # Extract original audio
        audio_path = self.workspace / "original_audio.wav"
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
        
        # Mix in snarks with ducking
        mixed = normalized
        
        for snark in valid_snarks:
            if not os.path.exists(snark.audio_path):
                continue
            
            snark_audio = AudioSegment.from_mp3(snark.audio_path)
            snark_audio = snark_audio.apply_gain(3)  # Boost
            
            position_ms = int(snark.time * 1000)
            
            # Apply smooth ducking
            fade_ms = 400
            duck_db = -10
            
            # Calculate boundaries
            fade_in_start = max(0, position_ms - fade_ms)
            fade_out_end = min(len(mixed), position_ms + len(snark_audio) + fade_ms)
            
            # Apply ducking
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
            
            # Reconstruct and overlay
            mixed = before + fade_in + during + fade_out + after
            mixed = mixed.overlay(snark_audio, position=position_ms)
            
            print(f"  ✅ Added at {snark.time:.1f}s: \"{snark.text}\"")
        
        # Export mixed audio
        mixed_path = self.workspace / "mixed_audio.wav"
        mixed.export(mixed_path, format="wav")
        
        # Create final video
        output_path = self.video_path.parent / f"{self.video_name}_snarked.mp4"
        
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
        
        report_path = self.workspace / "pipeline_report.json"
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