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
        
        # Create output folder next to the video
        self.output_folder = self.video_path.parent / self.video_name
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Storage for generated snarks (global library)
        self.snark_storage = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads/assets/sounds/snark_remarks")
        self.snark_storage.mkdir(parents=True, exist_ok=True)
        
        # API keys
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_model = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
        
        # Load existing snark library
        self.existing_snarks = self._load_existing_snarks()
        
        print(f"🎬 Pipeline initialized for: {self.video_name}")
        print(f"📁 Output folder: {self.output_folder}")
        print(f"📚 Existing snark library: {len(self.existing_snarks)} available")
    
    def _load_existing_snarks(self) -> Dict[str, Dict]:
        """Load all existing snarks from storage"""
        existing = {}
        
        if not self.snark_storage.exists():
            return existing
        
        # Scan all MP3 files in snark storage
        for mp3_file in self.snark_storage.glob("*.mp3"):
            # Convert filename back to text
            text = mp3_file.stem.replace("_", " ")
            
            # Try to determine emotion from filename or default
            emotion = "sarcastic"  # Default
            
            # Store with metadata
            existing[text.lower()] = {
                "text": text,
                "file": str(mp3_file),
                "emotion": emotion,
                "duration": self._estimate_duration(text)
            }
        
        return existing
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate speaking duration for text"""
        words = len(text.split())
        return words / 2.5 + 0.3  # ~2.5 words/sec + padding
    
    def extract_transcript(self) -> Dict:
        """Step 1: Extract transcript from video"""
        
        transcript_file = self.output_folder / "transcript.json"
        
        # Check if already exists
        if transcript_file.exists():
            print("📝 Loading existing transcript...")
            with open(transcript_file) as f:
                return json.load(f)
        
        print("🎤 Extracting transcript with Whisper...")
        
        # Extract audio first
        audio_path = self.output_folder / "audio.wav"
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
        """Step 2: Find gaps between spoken segments in transcript"""
        
        print("🔍 Analyzing gaps between spoken segments...")
        
        # Get transcript
        transcript_file = self.output_folder / "transcript.json"
        if not transcript_file.exists():
            print("⚠️ No transcript found, extracting first...")
            self.extract_transcript()
        
        with open(transcript_file) as f:
            transcript = json.load(f)
        
        segments = transcript.get("segments", [])
        if not segments:
            print("❌ No segments in transcript")
            return []
        
        # Get video duration
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.video_path)
        ], capture_output=True, text=True)
        video_duration = float(result.stdout.strip())
        
        gaps = []
        
        # Check gap at the beginning
        if segments[0]["start"] > 0.6:  # At least 0.6s gap at start
            gaps.append(SilenceGap(
                start=0.0,
                end=segments[0]["start"],
                duration=segments[0]["start"]
            ))
        
        # Find gaps between segments
        for i in range(len(segments) - 1):
            gap_start = segments[i]["end"]
            gap_end = segments[i + 1]["start"]
            gap_duration = gap_end - gap_start
            
            # Only include gaps that are long enough for a snark
            if gap_duration >= 0.6:  # 0.6 seconds minimum for short snarks
                gaps.append(SilenceGap(
                    start=gap_start,
                    end=gap_end,
                    duration=gap_duration
                ))
        
        # Check gap at the end
        last_segment_end = segments[-1]["end"]
        if video_duration - last_segment_end > 0.6:
            gaps.append(SilenceGap(
                start=last_segment_end,
                end=video_duration,
                duration=video_duration - last_segment_end
            ))
        
        print(f"✅ Found {len(gaps)} usable gaps between spoken segments")
        return gaps
    
    def generate_contextual_snarks(self, transcript: Dict, gaps: List[SilenceGap]) -> List[Snark]:
        """Step 3: Generate snarks based on transcript context"""
        
        print("🤖 Generating contextual snarks...")
        
        snarks = []
        reused_count = 0
        
        for gap in gaps[:10]:  # Limit to 10 snarks
            # Find context: what was said before this gap
            context = ""
            for seg in transcript["segments"]:
                if seg["end"] <= gap.start and seg["end"] > gap.start - 3:
                    context = seg["text"]
                    break
            
            # FIRST: Check if we have a perfect existing snark for this context
            existing_snark = self._find_existing_snark_for_context(context, gap.duration)
            
            if existing_snark:
                # Reuse existing snark!
                snarks.append(Snark(
                    text=existing_snark["text"],
                    time=gap.start + 0.2,
                    emotion=existing_snark["emotion"],
                    context=context,
                    gap_duration=gap.duration
                ))
                reused_count += 1
                print(f"  ♻️ Reusing: \"{existing_snark['text']}\"")
            else:
                # Generate new snark with AI
                if self.gemini_key:
                    snark = self._generate_snark_with_ai(context, gap.duration)
                else:
                    snark = self._get_fallback_snark(gap.duration)
                
                if snark:
                    snarks.append(Snark(
                        text=snark["text"],
                        time=gap.start + 0.2,
                        emotion=snark["emotion"],
                        context=context,
                        gap_duration=gap.duration
                    ))
                    print(f"  ✨ New: \"{snark['text']}\"")
        
        print(f"✅ Total snarks: {len(snarks)} ({reused_count} reused, {len(snarks)-reused_count} new)")
        return snarks
    
    def _find_existing_snark_for_context(self, context: str, max_duration: float) -> Optional[Dict]:
        """Find a suitable existing snark for this context"""
        
        context_lower = context.lower()
        
        # Smart matching based on context keywords
        suitable_snarks = []
        
        for text, snark_data in self.existing_snarks.items():
            # Check if snark fits in the gap
            if snark_data["duration"] > max_duration:
                continue
            
            # Score relevance based on context
            score = 0
            
            # Perfect matches for common situations
            if "again" in context_lower or "repeat" in context_lower:
                if "original" in text or "again" in text:
                    score += 10
            
            if "sing" in context_lower or "song" in context_lower:
                if "musical" in text or "performance" in text:
                    score += 8
            
            if any(word in context_lower for word in ["do", "re", "mi", "fa", "sol"]):
                if "music" in text or "note" in text:
                    score += 7
            
            # Generic snarks that work anywhere
            if text in ["riveting", "fascinating", "how original", "stunning", "sure"]:
                score += 5
            
            if score > 0:
                suitable_snarks.append((score, snark_data))
        
        # Return best match if found
        if suitable_snarks:
            suitable_snarks.sort(key=lambda x: x[0], reverse=True)
            return suitable_snarks[0][1]
        
        return None
    
    def _generate_snark_with_ai(self, context: str, max_duration: float) -> Optional[Dict]:
        """Generate a single snark using Gemini"""
        
        # Determine word limit based on duration
        max_words = int(max_duration * 2.5)  # ~2.5 words per second
        
        # Get list of existing snarks to inform AI
        existing_texts = list(self.existing_snarks.keys())[:20]  # Top 20
        
        prompt = f"""
        You are a sarcastic narrator commenting on a video.
        
        Context (what was just said): "{context}"
        
        We already have these snarks in our library (use one if it fits perfectly):
        {', '.join(f'"{t}"' for t in existing_texts)}
        
        Generate ONE brief sarcastic comment (max {max_words} words).
        If an existing snark fits perfectly, use it. Otherwise create new.
        
        Return JSON:
        {{
            "text": "your sarcastic comment",
            "emotion": "choose: sarcastic/deadpan/mocking/bored/unimpressed",
            "is_existing": true/false
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
    
    def _get_fallback_snark(self, max_duration: float) -> Dict:
        """Get a fallback snark when AI is not available"""
        
        # First check existing library
        for text, snark_data in self.existing_snarks.items():
            if snark_data["duration"] <= max_duration:
                return {
                    "text": snark_data["text"],
                    "emotion": snark_data["emotion"]
                }
        
        # Hardcoded fallbacks
        if max_duration < 1.5:
            return {"text": "Wow.", "emotion": "deadpan"}
        elif max_duration < 2.5:
            return {"text": "Fascinating.", "emotion": "sarcastic"}
        else:
            return {"text": "How original.", "emotion": "deadpan"}
    
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
        
        # First check if we already have this exact text
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
        
        # Check if file exists (might have been created in another session)
        if audio_path.exists():
            print(f"  ♻️ Reusing existing file: {filename}")
            # Add to library for future reference
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
        
        # Save remarks JSON
        remarks_file = self.output_folder / "remarks.json"
        remarks_data = []
        
        # Generate speech for all snarks and save remark files
        for i, snark in enumerate(snarks):
            audio_path = self.generate_speech_with_elevenlabs(snark)
            snark.audio_path = audio_path
            
            # Copy audio file to output folder
            if audio_path:
                local_audio = self.output_folder / f"remark_{i+1}.mp3"
                subprocess.run(["cp", audio_path, str(local_audio)], capture_output=True)
                
            remarks_data.append({
                "time": snark.time,
                "text": snark.text,
                "emotion": snark.emotion,
                "context": snark.context[:50] if snark.context else "",
                "audio_file": f"remark_{i+1}.mp3",
                "library_file": Path(audio_path).name if audio_path else None
            })
        
        # Save remarks JSON
        with open(remarks_file, "w") as f:
            json.dump(remarks_data, f, indent=2)
        print(f"  📝 Saved remarks: {remarks_file}")
        
        # Filter out snarks without audio
        valid_snarks = [s for s in snarks if hasattr(s, 'audio_path') and s.audio_path]
        
        if not valid_snarks:
            print("❌ No valid snarks to add")
            return str(self.video_path)
        
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
        mixed_path = self.output_folder / "mixed_audio.wav"
        mixed.export(mixed_path, format="wav")
        
        # Create final video in output folder
        output_path = self.output_folder / f"{self.video_name}_final.mp4"
        
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
        
        report_path = self.output_folder / "pipeline_report.json"
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
        
        # Auto-open the final video
        print("\n🎯 Opening final video...")
        try:
            subprocess.run(["open", output], check=True)
        except Exception as e:
            print(f"  ⚠️ Could not auto-open video: {e}")
        
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