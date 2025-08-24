"""Main comment pipeline orchestration."""

import json
import subprocess
from pathlib import Path
from typing import List, Dict

from pydub import AudioSegment

from utils.auto_comment.pipeline.models import Snark
from utils.auto_comment.pipeline.transcript_analyzer import TranscriptAnalyzer
from utils.auto_comment.pipeline.comment_generator import CommentGenerator
from utils.auto_comment.pipeline.audio_processor import AudioProcessor
from utils.auto_comment.pipeline.video_processor import VideoProcessor


class EndToEndCommentPipeline:
    """Complete end-to-end comment pipeline."""
    
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        
        # Create output folder
        base_folder = self.video_path.parent / self.video_name
        self.output_folder = base_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"🎬 Pipeline initialized for: {self.video_name}")
        print(f"📁 Output folder: {self.output_folder}")
        
        # Initialize components
        self.transcript_analyzer = TranscriptAnalyzer(self.video_path)
        self.comment_generator = CommentGenerator()
        self.audio_processor = AudioProcessor(self.video_path, self.output_folder)
        self.video_processor = VideoProcessor(self.video_path)
    
    def run_pipeline(self) -> str:
        """Run the complete end-to-end pipeline."""
        print("\n" + "=" * 70)
        print("🚀 RUNNING END-TO-END COMMENT PIPELINE")
        print("=" * 70)
        
        # Step 1: Extract transcript
        transcript = self.transcript_analyzer.extract_transcript(self.output_folder)
        
        # Step 2: Find silence gaps
        gaps = self.transcript_analyzer.analyze_silence_gaps(transcript)
        
        if not gaps:
            print("❌ No silence gaps found - cannot add comments")
            return str(self.video_path)
        
        # Step 3: Generate contextual comments
        snarks = self.comment_generator.generate_contextual_snarks(transcript, gaps)
        
        # Step 4 & 5: Generate speech and mix into video
        output = self.mix_snarks_into_video(snarks)
        
        # Save report
        self._save_report(transcript, gaps, snarks, output)
        
        print("\n" + "=" * 70)
        print("✨ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"📊 Report: {self.output_folder / 'pipeline_report.json'}")
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
    
    def mix_snarks_into_video(self, snarks: List[Snark]) -> str:
        """Mix snarks into final video."""
        print("🎬 Creating final video with comments...")
        
        # Save remarks JSON
        self._save_remarks(snarks)
        
        # Generate speech for all snarks
        snarks_with_audio = []
        for snark in snarks:
            audio_path = self.audio_processor.generate_speech_with_elevenlabs(snark)
            if audio_path:
                snark.audio_path = audio_path
                # Copy to output folder
                local_audio = self.output_folder / f"remark_{len(snarks_with_audio)+1}.mp3"
                subprocess.run(["cp", audio_path, str(local_audio)], capture_output=True)
        
        # Filter out snarks without audio
        valid_snarks = [s for s in snarks if s.audio_path]
        
        if not valid_snarks:
            print("❌ No valid comments to add")
            return str(self.video_path)
        
        # Prepare snarks with speed calculations
        snarks_with_pausing = []
        for snark in valid_snarks:
            snark_audio = AudioSegment.from_mp3(snark.audio_path)
            snark_duration = len(snark_audio) / 1000.0
            
            # Calculate speed adjustment
            speed_factor = snark.gap_duration / snark_duration if snark_duration > snark.gap_duration else 1.0
            needs_slowdown = speed_factor < 1.0
            
            snarks_with_pausing.append({
                "snark": snark,
                "audio": snark_audio,
                "duration": snark_duration,
                "needs_slowdown": needs_slowdown,
                "speed_factor": speed_factor,
                "gap_duration": snark.gap_duration
            })
        
        # Check if any remarks need speed adjustment
        has_slowdowns = any(s["needs_slowdown"] for s in snarks_with_pausing)
        
        if has_slowdowns:
            print("  🐢 Some remarks need video speed adjustment...")
        
        # Create mixed audio
        mixed_path = self.audio_processor.create_audio_with_speed_adjustments(
            snarks_with_pausing
        )
        
        # Create final video
        output_path = self.output_folder / f"{self.video_name}_final.mp4"
        
        if has_slowdowns:
            print("  🎬 Creating video with speed adjustments...")
            self.video_processor.create_video_with_speed_adjustments(
                snarks_with_pausing, mixed_path, output_path
            )
        else:
            # Simple remux without speed adjustments
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(self.video_path),
                "-i", str(mixed_path),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "256k",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ], capture_output=True)
        
        print(f"✅ Output: {output_path}")
        return str(output_path)
    
    def _save_remarks(self, snarks: List[Snark]):
        """Save remarks to JSON file."""
        remarks_file = self.output_folder / "remarks.json"
        remarks_data = []
        
        for i, snark in enumerate(snarks):
            remarks_data.append({
                "time": snark.time,
                "text": snark.text,
                "emotion": snark.emotion,
                "context": snark.context[:50] if snark.context else "",
                "audio_file": f"remark_{i+1}.mp3",
                "library_file": Path(snark.audio_path).name if snark.audio_path else None
            })
        
        with open(remarks_file, "w") as f:
            json.dump(remarks_data, f, indent=2)
        print(f"  📝 Saved remarks: {remarks_file}")
    
    def _save_report(self, transcript: Dict, gaps: List, snarks: List[Snark], output: str):
        """Save pipeline report."""
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main_pipeline.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    pipeline = EndToEndCommentPipeline(video_path)
    pipeline.run_pipeline()