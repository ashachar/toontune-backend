"""Main comment pipeline orchestration."""

import json
import subprocess
from pathlib import Path
from typing import List, Dict

from pydub import AudioSegment

from utils.auto_comment.pipeline.models import Comment
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
        comments = self.comment_generator.generate_contextual_comments(transcript, gaps)
        
        # Step 4 & 5: Generate speech and mix into video
        output = self.mix_comments_into_video(comments)
        
        # Save report
        self._save_report(transcript, gaps, comments, output)
        
        print("\n" + "=" * 70)
        print("✨ PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"📊 Report: {self.output_folder / 'pipeline_report.json'}")
        print(f"🎬 Output: {output}")
        
        # Calculate cost
        total_chars = sum(len(c.text) for c in comments)
        cost = total_chars * 0.0003
        print(f"💰 Estimated cost: ${cost:.3f}")
        
        # Auto-open the final video
        print("\n🎯 Opening final video...")
        try:
            subprocess.run(["open", output], check=True)
        except Exception as e:
            print(f"  ⚠️ Could not auto-open video: {e}")
        
        return output
    
    def mix_comments_into_video(self, comments: List[Comment]) -> str:
        """Mix comments into final video."""
        print("🎬 Creating final video with comments...")
        
        # Save remarks JSON
        self._save_remarks(comments)
        
        # Process audio for all comments (gets audio files and calculates speed factors)
        comments_with_pausing = self.audio_processor.process_comments_audio(comments)
        
        if not comments_with_pausing:
            print("❌ No valid comments to add")
            return str(self.video_path)
        
        # Check if any remarks need speed adjustment
        has_slowdowns = any(c["needs_slowdown"] for c in comments_with_pausing)
        
        if has_slowdowns:
            print("  🐢 Some remarks need video speed adjustment...")
        
        # Create mixed audio
        mixed_path = self.audio_processor.create_audio_with_speed_adjustments(
            comments_with_pausing
        )
        
        # Create final video
        output_path = self.output_folder / f"{self.video_name}_final.mp4"
        
        if has_slowdowns:
            print("  🎬 Creating video with speed adjustments...")
            self.video_processor.create_video_with_speed_adjustments(
                comments_with_pausing, mixed_path, output_path
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
    
    def _save_remarks(self, comments: List[Comment]):
        """Save remarks to JSON file."""
        remarks_file = self.output_folder / "remarks.json"
        remarks_data = []
        
        for i, comment in enumerate(comments):
            remarks_data.append({
                "time": comment.time,
                "text": comment.text,
                "emotion": comment.emotion,
                "context": comment.context[:50] if comment.context else "",
                "audio_file": f"remark_{i+1}.mp3",
                "library_file": Path(comment.audio_path).name if comment.audio_path else None
            })
        
        with open(remarks_file, "w") as f:
            json.dump(remarks_data, f, indent=2)
        print(f"  📝 Saved remarks: {remarks_file}")
    
    def _save_report(self, transcript: Dict, gaps: List, comments: List[Comment], output: str):
        """Save pipeline report."""
        report = {
            "video": str(self.video_path),
            "transcript_segments": len(transcript["segments"]),
            "silence_gaps": len(gaps),
            "comments_generated": len(comments),
            "comments": [
                {
                    "time": c.time,
                    "text": c.text,
                    "emotion": c.emotion,
                    "context": c.context[:30] if c.context else "",
                    "gap_duration": c.gap_duration
                }
                for c in comments
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