#!/usr/bin/env python3
"""
Extract word-level timestamps using ElevenLabs Scribe v1
This provides exact start/end times for every word, enabling precise gap detection
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from dotenv import load_dotenv


class WordLevelTranscriber:
    """Extract word-level timestamps using ElevenLabs Scribe."""
    
    def __init__(self, video_path: str):
        load_dotenv()
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not found in .env")
        
        # Output directory
        self.output_dir = self.video_path.parent / self.video_name
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎬 Word-Level Transcriber")
        print(f"📹 Video: {self.video_path}")
        print(f"📁 Output: {self.output_dir}")
    
    def extract_audio(self) -> Path:
        """Extract audio from video for transcription."""
        audio_path = self.output_dir / f"{self.video_name}_audio.mp3"
        
        if audio_path.exists():
            print(f"♻️ Using existing audio: {audio_path}")
            return audio_path
        
        print("🎵 Extracting audio from video...")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(self.video_path),
            "-vn",  # No video
            "-acodec", "mp3",
            "-b:a", "128k",  # Good quality for speech
            str(audio_path)
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✅ Audio extracted: {audio_path}")
        return audio_path
    
    def transcribe_with_scribe(self, audio_path: Path) -> Dict:
        """Use ElevenLabs Scribe v1 for word-level transcription."""
        print("\n📝 Transcribing with ElevenLabs Scribe...")
        
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {"xi-api-key": self.api_key}
        
        with open(audio_path, "rb") as f:
            files = {
                "file": (audio_path.name, f, "audio/mpeg"),
                "model_id": (None, "scribe_v1"),
            }
            
            print("   Uploading audio to ElevenLabs...")
            response = requests.post(url, headers=headers, files=files, timeout=60)
        
        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            print("\n⚠️ API key needs speech-to-text permissions")
            print("Please update your ElevenLabs API key with STT permissions")
            return None
        
        data = response.json()
        print(f"✅ Transcription complete!")
        print(f"   Words: {len(data.get('words', []))}")
        print(f"   Duration: {data.get('words', [{}])[-1].get('end_time', 0):.1f}s")
        
        # Save raw response
        raw_path = self.output_dir / "transcript_elevenlabs_raw.json"
        with open(raw_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"   Saved raw: {raw_path}")
        
        return data
    
    def analyze_gaps(self, transcript: Dict) -> List[Dict]:
        """Find gaps between words where comments can be inserted."""
        words = transcript.get("words", [])
        
        if not words:
            print("❌ No words found in transcript")
            return []
        
        print(f"\n🔍 Analyzing gaps between {len(words)} words...")
        
        gaps = []
        min_gap_duration = 0.3  # Minimum gap to consider (300ms)
        
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            gap_start = current_word.get("end_time", 0)
            gap_end = next_word.get("start_time", 0)
            gap_duration = gap_end - gap_start
            
            if gap_duration >= min_gap_duration:
                # Context: 5 words before and after
                context_before = " ".join([
                    w["word"] for w in words[max(0, i-4):i+1]
                ])
                context_after = " ".join([
                    w["word"] for w in words[i+1:min(len(words), i+6)]
                ])
                
                gaps.append({
                    "index": len(gaps),
                    "start": gap_start,
                    "end": gap_end,
                    "duration": gap_duration,
                    "after_word": current_word["word"],
                    "before_word": next_word["word"],
                    "context_before": context_before,
                    "context_after": context_after
                })
        
        # Sort by duration to find best gaps
        gaps.sort(key=lambda x: x["duration"], reverse=True)
        
        print(f"✅ Found {len(gaps)} gaps >= {min_gap_duration}s")
        
        return gaps
    
    def create_word_level_transcript(self, transcript: Dict, gaps: List[Dict]) -> Dict:
        """Create formatted transcript with word-level timing and gaps."""
        words = transcript.get("words", [])
        
        # Create segments (phrase-level for compatibility)
        segments = []
        current_segment = []
        current_start = 0
        
        for i, word in enumerate(words):
            if not current_segment:
                current_start = word.get("start_time", 0)
            
            current_segment.append(word["word"])
            
            # End segment at punctuation or large gap
            is_sentence_end = word["word"].rstrip().endswith((".", "!", "?"))
            is_large_gap = False
            
            if i < len(words) - 1:
                gap = words[i + 1]["start_time"] - word["end_time"]
                is_large_gap = gap > 0.5
            
            if is_sentence_end or is_large_gap or i == len(words) - 1:
                segments.append({
                    "start": current_start,
                    "end": word.get("end_time", 0),
                    "text": " ".join(current_segment)
                })
                current_segment = []
        
        # Format the complete transcript
        formatted = {
            "text": transcript.get("transcript", ""),
            "segments": segments,
            "words": words,  # Include word-level data!
            "gaps": gaps[:20]  # Top 20 gaps
        }
        
        # Save formatted transcript
        output_path = self.output_dir / "transcript_elevenlabs.json"
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        
        print(f"\n📄 Saved ElevenLabs transcript: {output_path}")
        
        return formatted
    
    def print_best_gaps(self, gaps: List[Dict], limit: int = 10):
        """Print the best gaps for comment insertion."""
        print(f"\n🎯 Top {limit} gaps for comments:")
        print("=" * 70)
        
        for gap in gaps[:limit]:
            print(f"\n#{gap['index'] + 1}: Gap at {gap['start']:.2f}s")
            print(f"   Duration: {gap['duration']:.3f}s")
            print(f"   After: '{gap['after_word']}'")
            print(f"   Before: '{gap['before_word']}'")
            print(f"   Context: ...{gap['context_before']} | {gap['context_after']}...")
    
    def run(self) -> Dict:
        """Run the complete word-level transcription pipeline."""
        print("\n" + "=" * 70)
        print("🚀 STARTING WORD-LEVEL TRANSCRIPTION")
        print("=" * 70)
        
        # Step 1: Extract audio
        audio_path = self.extract_audio()
        
        # Step 2: Transcribe with Scribe
        transcript = self.transcribe_with_scribe(audio_path)
        
        if not transcript:
            print("❌ Transcription failed")
            return None
        
        # Step 3: Analyze gaps
        gaps = self.analyze_gaps(transcript)
        
        # Step 4: Create formatted transcript
        formatted = self.create_word_level_transcript(transcript, gaps)
        
        # Step 5: Print best gaps
        self.print_best_gaps(gaps)
        
        print("\n" + "=" * 70)
        print("✅ WORD-LEVEL TRANSCRIPTION COMPLETE!")
        print("=" * 70)
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Total words: {len(transcript.get('words', []))}")
        print(f"   Total gaps: {len(gaps)}")
        print(f"   Largest gap: {gaps[0]['duration']:.3f}s" if gaps else "N/A")
        print(f"   Output: {self.output_dir}/transcript_word_level.json")
        
        return formatted


def main():
    if len(sys.argv) < 2:
        print("Usage: python word_level_transcript.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    transcriber = WordLevelTranscriber(video_path)
    transcriber.run()


if __name__ == "__main__":
    main()