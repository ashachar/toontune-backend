#!/usr/bin/env python3
"""
Local transcript extraction and verification for videos with audio editing.
Uses Whisper locally to transcribe and verify segment alignment.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import whisper


class TranscriptVerifier:
    """Extract transcript and verify segment alignment with comments."""
    
    def __init__(self, video_path: str, segments_info: Optional[Dict] = None):
        """
        Initialize verifier.
        
        Args:
            video_path: Path to video file
            segments_info: Optional dict with segment mapping info
        """
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.output_dir = self.video_path.parent
        self.segments_info = segments_info
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"🎬 Transcript Verifier: {self.video_path.name}")
        print("=" * 70)
    
    def extract_audio(self) -> Path:
        """Extract audio from video."""
        audio_path = Path(f"/tmp/{self.video_name}_audio.mp3")
        
        print("🎵 Extracting audio...")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(self.video_path),
            "-vn", "-acodec", "mp3", "-b:a", "128k",
            str(audio_path)
        ]
        subprocess.run(cmd, check=True)
        return audio_path
    
    def transcribe(self, audio_path: Path, model_size: str = "base") -> Dict:
        """Transcribe audio using Whisper."""
        print(f"📝 Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)
        
        print("🎙️ Transcribing...")
        result = model.transcribe(str(audio_path))
        
        # Save transcript
        json_path = self.output_dir / f"{self.video_name}_whisper_transcript.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)
        
        txt_path = self.output_dir / f"{self.video_name}_whisper_transcript.txt"
        with open(txt_path, "w") as f:
            f.write(result["text"])
        
        print(f"   Saved: {json_path.name}")
        print(f"   Saved: {txt_path.name}")
        
        return result
    
    def detect_comments(self, transcript: Dict) -> List[Dict]:
        """Detect inserted comments in transcript."""
        # Common comment words to look for (with variations)
        comment_patterns = [
            ("scalability", ["scalability"]),
            ("precisely", ["precisely", "precisely."]),
            ("elegant", ["elegant", "elegant."]),
            ("fascinating", ["fascinating", "fascinated", "fascinated."]),
            ("synergistic", ["synergistic", "synergistic."]),
            ("recursive", ["recursive", "recursive."]),
            ("isomorphic", ["isomorphic", "isomorphic?"]),
            ("unexpected", ["unexpected", "unexpected."]),
            ("granularity", ["granularity", "granularity?"]),
            ("non-commutative", ["non-commutative", "no commutative"])
        ]
        
        comments_found = []
        
        # Look through full text for isolated comment words
        full_text = transcript.get("text", "").lower()
        
        for segment in transcript.get("segments", []):
            text_lower = segment["text"].lower().strip()
            
            # Check each comment pattern
            for comment_base, variations in comment_patterns:
                for variant in variations:
                    if variant in text_lower:
                        # Check if it's relatively isolated
                        words = text_lower.replace(".", "").replace("?", "").split()
                        
                        # If the segment is short and contains the comment word
                        if len(words) <= 4 and any(v.replace(".", "").replace("?", "") in words for v in variations):
                            comments_found.append({
                                "text": segment["text"].strip(),
                                "start": segment["start"],
                                "end": segment["end"],
                                "duration": segment["end"] - segment["start"],
                                "type": comment_base
                            })
                            break
                else:
                    continue
                break
        
        # Also check for single-word segments that match
        for segment in transcript.get("segments", []):
            text = segment["text"].strip().lower()
            # Remove punctuation for comparison
            text_clean = text.replace(".", "").replace("?", "").replace("!", "")
            
            for comment_base, variations in comment_patterns:
                if text_clean in [v.replace(".", "").replace("?", "") for v in variations]:
                    # Check if already found
                    if not any(c["start"] == segment["start"] for c in comments_found):
                        comments_found.append({
                            "text": segment["text"].strip(),
                            "start": segment["start"],
                            "end": segment["end"],
                            "duration": segment["end"] - segment["start"],
                            "type": comment_base
                        })
        
        return sorted(comments_found, key=lambda x: x["start"])
    
    def detect_repeated_phrases(self, transcript: Dict, min_words: int = 3, 
                                time_window: float = 20.0) -> List[Dict]:
        """Detect repeated phrases within time window."""
        segments = transcript.get("segments", [])
        repeated_phrases = []
        
        # Build n-grams from all segments
        for i, segment in enumerate(segments):
            text = segment["text"].strip().lower()
            # Remove punctuation for comparison
            text_clean = text.replace(".", "").replace("?", "").replace("!", "").replace(",", "")
            words = text_clean.split()
            
            # Check n-grams of length min_words and above
            for n in range(min_words, min(len(words) + 1, 8)):  # Max 8 words
                for start_idx in range(len(words) - n + 1):
                    phrase = " ".join(words[start_idx:start_idx + n])
                    phrase_time = segment["start"]
                    
                    # Look for this phrase in subsequent segments within time window
                    for j, other_segment in enumerate(segments[i+1:], i+1):
                        if other_segment["start"] - phrase_time > time_window:
                            break  # Outside time window
                        
                        other_text = other_segment["text"].strip().lower()
                        other_clean = other_text.replace(".", "").replace("?", "").replace("!", "").replace(",", "")
                        
                        if phrase in other_clean:
                            repeated_phrases.append({
                                "phrase": phrase,
                                "first_occurrence": {
                                    "time": phrase_time,
                                    "text": segment["text"].strip()
                                },
                                "second_occurrence": {
                                    "time": other_segment["start"],
                                    "text": other_segment["text"].strip()
                                },
                                "time_diff": other_segment["start"] - phrase_time,
                                "word_count": n
                            })
        
        # Remove duplicates and shorter phrases that are part of longer ones
        unique_phrases = []
        for phrase in repeated_phrases:
            # Check if this phrase is part of a longer repeated phrase
            is_subset = False
            for other in repeated_phrases:
                if (phrase["phrase"] in other["phrase"] and 
                    phrase["phrase"] != other["phrase"] and
                    abs(phrase["first_occurrence"]["time"] - other["first_occurrence"]["time"]) < 1.0):
                    is_subset = True
                    break
            
            if not is_subset:
                # Check if already in unique list
                already_added = any(
                    p["phrase"] == phrase["phrase"] and 
                    abs(p["first_occurrence"]["time"] - phrase["first_occurrence"]["time"]) < 1.0
                    for p in unique_phrases
                )
                if not already_added:
                    unique_phrases.append(phrase)
        
        return sorted(unique_phrases, key=lambda x: x["first_occurrence"]["time"])
    
    def verify_alignment(self, transcript: Dict, comments: List[Dict]) -> Dict:
        """Verify segment alignment and comment placement."""
        print("\n" + "=" * 70)
        print("🔍 VERIFICATION REPORT")
        print("=" * 70)
        
        verification = {
            "total_duration": transcript.get("segments", [{}])[-1].get("end", 0),
            "total_segments": len(transcript.get("segments", [])),
            "comments_found": len(comments),
            "alignment_issues": [],
            "timing_issues": [],
            "repeated_phrases": []
        }
        
        # Check comment timing
        print(f"\n📊 TRANSCRIPT ANALYSIS:")
        print(f"   Duration: {verification['total_duration']:.1f}s")
        print(f"   Segments: {verification['total_segments']}")
        print(f"   Comments detected: {verification['comments_found']}")
        
        if comments:
            print(f"\n💬 COMMENTS FOUND:")
            for i, comment in enumerate(comments, 1):
                print(f"   {i}. {comment['text']:20s} at {comment['start']:6.2f}s")
                
                # Check if comment appears at a natural pause
                segments = transcript.get("segments", [])
                for j, seg in enumerate(segments):
                    if abs(seg["start"] - comment["start"]) < 0.5:
                        # Check previous segment for natural break
                        if j > 0:
                            prev_seg = segments[j-1]
                            gap = seg["start"] - prev_seg["end"]
                            if gap < 0.3:
                                verification["timing_issues"].append({
                                    "comment": comment["text"],
                                    "issue": f"No natural gap (only {gap:.3f}s)"
                                })
        
        # Check segment continuity
        segments = transcript.get("segments", [])
        for i in range(1, len(segments)):
            gap = segments[i]["start"] - segments[i-1]["end"]
            if gap > 2.0:  # Large gap
                verification["alignment_issues"].append({
                    "position": segments[i]["start"],
                    "gap_size": gap,
                    "issue": "Large gap between segments"
                })
        
        # Check for repeated phrases
        repeated_phrases = self.detect_repeated_phrases(transcript)
        verification["repeated_phrases"] = repeated_phrases
        
        # Report issues
        if verification["alignment_issues"]:
            print(f"\n⚠️ ALIGNMENT ISSUES:")
            for issue in verification["alignment_issues"]:
                print(f"   - At {issue['position']:.2f}s: {issue['issue']}")
        
        if repeated_phrases:
            print(f"\n🔁 REPEATED PHRASES (within 20s):")
            for rp in repeated_phrases[:5]:  # Show first 5
                print(f"   '{rp['phrase']}' ({rp['word_count']} words)")
                print(f"      First:  {rp['first_occurrence']['time']:.2f}s")
                print(f"      Second: {rp['second_occurrence']['time']:.2f}s")
                print(f"      Gap: {rp['time_diff']:.2f}s")
        
        if verification["timing_issues"]:
            print(f"\n⚠️ TIMING ISSUES:")
            for issue in verification["timing_issues"]:
                print(f"   - {issue['comment']}: {issue['issue']}")
        
        # Final verdict
        print(f"\n📋 VERDICT:")
        has_issues = (verification["alignment_issues"] or 
                     verification["timing_issues"] or 
                     verification["repeated_phrases"])
        
        if not has_issues:
            print("   ✅ All segments aligned correctly")
            print("   ✅ Comments placed at natural pauses")
            print("   ✅ No repeated phrases detected")
            return_code = 0
        else:
            if verification["repeated_phrases"]:
                print(f"   ❌ {len(verification['repeated_phrases'])} repeated phrases found")
            if verification["alignment_issues"]:
                print(f"   ❌ {len(verification['alignment_issues'])} alignment issues")
            if verification["timing_issues"]:
                print(f"   ⚠️ {len(verification['timing_issues'])} timing issues")
            print("   Review needed!")
            return_code = 1
        
        print("=" * 70)
        
        return verification, return_code
    
    def run(self) -> Tuple[Dict, int]:
        """Run complete verification pipeline."""
        # Extract audio
        audio_path = self.extract_audio()
        
        # Transcribe
        transcript = self.transcribe(audio_path)
        
        # Detect comments
        comments = self.detect_comments(transcript)
        
        # Verify alignment
        verification, return_code = self.verify_alignment(transcript, comments)
        
        # Save verification report
        report_path = self.output_dir / f"{self.video_name}_verification.json"
        with open(report_path, "w") as f:
            json.dump({
                "video": str(self.video_path),
                "comments": comments,
                "verification": verification
            }, f, indent=2)
        
        print(f"\n📄 Report saved: {report_path.name}")
        
        return verification, return_code


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python local_extract_transcript.py <video_path> [segments_json]")
        print("\nThis script:")
        print("  1. Extracts and transcribes audio using Whisper")
        print("  2. Detects inserted comments")
        print("  3. Verifies segment alignment")
        print("  4. Reports timing issues")
        sys.exit(1)
    
    video_path = sys.argv[1]
    segments_json = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load segments if provided
    segments_info = None
    if segments_json and Path(segments_json).exists():
        with open(segments_json) as f:
            segments_info = json.load(f)
    
    # Run verification
    verifier = TranscriptVerifier(video_path, segments_info)
    verification, return_code = verifier.run()
    
    sys.exit(return_code)


if __name__ == "__main__":
    main()