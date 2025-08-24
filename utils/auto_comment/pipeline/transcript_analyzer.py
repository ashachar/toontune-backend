"""Transcript and silence gap analysis."""

import json
import subprocess
from pathlib import Path
from typing import List, Dict

from utils.auto_comment.pipeline.models import SilenceGap
from utils.auto_comment.pipeline.config import MIN_GAP_DURATION


class TranscriptAnalyzer:
    """Analyzes transcripts and finds silence gaps."""
    
    def __init__(self, video_path: Path):
        self.video_path = video_path
        
    def extract_transcript(self, output_folder: Path) -> Dict:
        """Extract transcript from video using Whisper."""
        transcript_file = output_folder / "transcript.json"
        
        # Check if transcript already exists
        if transcript_file.exists():
            print("📝 Loading existing transcript...")
            with open(transcript_file) as f:
                return json.load(f)
        
        print("🎤 Extracting transcript with Whisper...")
        
        # Extract audio first
        audio_file = output_folder / "temp_audio.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(self.video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            str(audio_file)
        ]
        subprocess.run(cmd, capture_output=True)
        
        # Run Whisper
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(str(audio_file), language="en")
            
            # Format transcript
            transcript = {
                "text": result["text"],
                "segments": [
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"]
                    }
                    for seg in result["segments"]
                ]
            }
            
            # Save transcript
            with open(transcript_file, "w") as f:
                json.dump(transcript, f, indent=2)
            
            # Cleanup
            audio_file.unlink()
            
            print(f"✅ Transcript extracted: {len(transcript['segments'])} segments")
            return transcript
            
        except ImportError:
            print("⚠️ Whisper not available, using mock transcript")
            return {"text": "Mock transcript", "segments": []}
    
    def analyze_silence_gaps(self, transcript: Dict) -> List[SilenceGap]:
        """Find gaps between spoken segments."""
        print("🔍 Analyzing gaps between spoken segments...")
        
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
        if segments[0]["start"] > MIN_GAP_DURATION:
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
            
            if gap_duration >= MIN_GAP_DURATION:
                gaps.append(SilenceGap(
                    start=gap_start,
                    end=gap_end,
                    duration=gap_duration
                ))
        
        # Check gap at the end
        last_segment_end = segments[-1]["end"]
        if video_duration - last_segment_end > MIN_GAP_DURATION:
            gaps.append(SilenceGap(
                start=last_segment_end,
                end=video_duration,
                duration=video_duration - last_segment_end
            ))
        
        print(f"✅ Found {len(gaps)} usable gaps between spoken segments")
        return gaps