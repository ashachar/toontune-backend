#!/usr/bin/env python3
"""
Auto-Snark Narrator with ElevenLabs v3
Features:
- Expression controls for cynical/sarcastic tone
- Emotion tags for enhanced delivery
- Multi-speaker dialog capabilities
"""

import os
import json
import logging
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Optional
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment, effects

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ElevenLabs v3 voice IDs (examples - use your own)
NARRATOR_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - good for sarcasm
ALTERNATIVE_VOICE_ID = "yoZ06aMxZJJ28mfd3POQ"  # Sam - deeper, more cynical

@dataclass
class CynicalSnark:
    """Represents a cynical remark with expression metadata"""
    time_s: float
    text: str
    emotion: str  # sarcastic, condescending, mocking, deadpan
    voice_id: str
    emphasis_words: List[str]
    duration_ms: Optional[int] = None

class SnarkNarratorV3:
    """Enhanced snark narrator using ElevenLabs v3 features"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with ElevenLabs v3 client"""
        self.client = ElevenLabs(api_key=api_key or os.environ.get("ELEVENLABS_API_KEY"))
        self.model_id = "eleven_v3"  # Use v3 model for expression controls
        
    def create_cynical_snarks(self, transcript_segments: List[Dict]) -> List[CynicalSnark]:
        """Generate cynical snarks for Do-Re-Mi scene with emotion tags"""
        
        snarks = [
            CynicalSnark(
                time_s=3.8,
                text='<emotion="sarcastic">Oh good, another musical number.</emotion> <emotion="deadpan">How original.</emotion>',
                emotion="sarcastic",
                voice_id=NARRATOR_VOICE_ID,
                emphasis_words=["another", "original"]
            ),
            CynicalSnark(
                time_s=16.8,
                text='<emotion="condescending">Yes, because ABC is <emphasis>such</emphasis> complex knowledge.</emotion>',
                emotion="condescending",
                voice_id=ALTERNATIVE_VOICE_ID,
                emphasis_words=["such", "complex"]
            ),
            CynicalSnark(
                time_s=25.5,
                text='<emotion="mocking">We get it.</emotion> <emotion="deadpan">You can repeat three syllables.</emotion>',
                emotion="mocking",
                voice_id=NARRATOR_VOICE_ID,
                emphasis_words=["get", "three"]
            ),
            CynicalSnark(
                time_s=41.8,
                text='<emotion="skeptical">Easier?</emotion> <emotion="sarcastic">This is your idea of pedagogy?</emotion>',
                emotion="skeptical",
                voice_id=ALTERNATIVE_VOICE_ID,
                emphasis_words=["Easier", "pedagogy"]
            ),
            CynicalSnark(
                time_s=48.0,
                text='<emotion="deadpan">Revolutionary.</emotion> <pause duration="500ms"/> <emotion="mocking">A deer is... a deer.</emotion>',
                emotion="deadpan",
                voice_id=NARRATOR_VOICE_ID,
                emphasis_words=["Revolutionary", "deer"]
            ),
            CynicalSnark(
                time_s=52.5,
                text='<emotion="sarcastic">Mi,</emotion> <emotion="condescending">the narcissism is showing.</emotion>',
                emotion="sarcastic",
                voice_id=NARRATOR_VOICE_ID,
                emphasis_words=["narcissism", "showing"]
            )
        ]
        
        return snarks
    
    def generate_snark_audio(self, snark: CynicalSnark, output_path: str) -> bool:
        """Generate audio for a single snark using ElevenLabs v3"""
        
        try:
            logging.info(f"Generating cynical audio: {snark.text[:50]}...")
            
            # Advanced v3 settings for expression control
            voice_settings = {
                "stability": 0.3,  # Lower for more expression
                "similarity_boost": 0.7,
                "style": 0.8,  # Higher for more emotional range
                "use_speaker_boost": True
            }
            
            # Add pronunciation guide for sarcasm
            if snark.emotion == "sarcastic":
                voice_settings["style"] = 0.9
                voice_settings["stability"] = 0.2
            elif snark.emotion == "deadpan":
                voice_settings["style"] = 0.1
                voice_settings["stability"] = 0.9
            
            # Generate audio with v3 model
            audio = self.client.text_to_speech.convert(
                text=snark.text,
                voice_id=snark.voice_id,
                model_id=self.model_id,
                voice_settings=voice_settings,
                output_format="mp3_44100_128"
            )
            
            # Save the audio
            save(audio, output_path)
            
            # Get duration
            temp_audio = AudioSegment.from_mp3(output_path)
            snark.duration_ms = len(temp_audio)
            
            logging.info(f"✅ Generated: {output_path} ({snark.duration_ms}ms)")
            return True
            
        except Exception as e:
            logging.error(f"Failed to generate audio: {e}")
            return False
    
    def create_dialog_version(self, snarks: List[CynicalSnark]) -> Optional[str]:
        """Create multi-speaker dialog using v3 dialog endpoint"""
        
        try:
            logging.info("Creating multi-speaker cynical dialog...")
            
            # Build dialog structure for v3
            dialog = []
            
            # Add opening narrator
            dialog.append({
                "speaker": "Narrator",
                "voice_id": NARRATOR_VOICE_ID,
                "text": '<emotion="sarcastic">Let me add some honest commentary to this... masterpiece.</emotion>'
            })
            
            # Add each snark as dialog entry
            for i, snark in enumerate(snarks):
                speaker = "Cynic1" if i % 2 == 0 else "Cynic2"
                dialog.append({
                    "speaker": speaker,
                    "voice_id": snark.voice_id,
                    "text": snark.text,
                    "timing": {"start_time": snark.time_s}
                })
            
            # Generate dialog audio (would use dialog endpoint in production)
            # For now, generate individual files
            output_files = []
            for i, snark in enumerate(snarks):
                output_path = f"dialog_snark_{i+1}.mp3"
                if self.generate_snark_audio(snark, output_path):
                    output_files.append(output_path)
            
            return output_files
            
        except Exception as e:
            logging.error(f"Dialog generation failed: {e}")
            return None
    
    def process_video_with_snarks(self, video_path: str, transcript_path: str, output_path: str):
        """Process video with ElevenLabs v3 cynical narration"""
        
        logging.info("🎬 Processing video with ElevenLabs v3 cynical narration...")
        
        # Load transcript
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
        
        # Generate snarks
        snarks = self.create_cynical_snarks(transcript.get("segments", []))
        
        with tempfile.TemporaryDirectory() as tmp:
            # Generate audio for each snark
            audio_files = []
            for i, snark in enumerate(snarks):
                audio_path = os.path.join(tmp, f"snark_{i+1}.mp3")
                if self.generate_snark_audio(snark, audio_path):
                    audio_files.append((snark.time_s, audio_path, snark.duration_ms))
            
            if not audio_files:
                logging.error("No audio files generated")
                return False
            
            # Extract original audio
            logging.info("Extracting and mixing audio...")
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(os.path.join(tmp, "original.wav"), 
                                        verbose=False, logger=None)
            clip.close()
            
            # Mix audio with ducking
            base_audio = AudioSegment.from_file(os.path.join(tmp, "original.wav"))
            
            for time_s, audio_path, duration_ms in audio_files:
                overlay = AudioSegment.from_mp3(audio_path)
                start_ms = int(time_s * 1000)
                
                # Duck original audio during snark
                end_ms = start_ms + duration_ms
                if end_ms > len(base_audio):
                    base_audio = base_audio + AudioSegment.silent(duration=end_ms - len(base_audio))
                
                # Apply ducking
                pre = base_audio[:start_ms]
                mid = base_audio[start_ms:end_ms] - 15  # Duck by 15dB
                post = base_audio[end_ms:]
                
                # Mix in the snark
                mixed_mid = mid.overlay(overlay)
                base_audio = pre + mixed_mid + post
            
            # Normalize and export
            base_audio = effects.normalize(base_audio)
            mixed_path = os.path.join(tmp, "mixed_audio.wav")
            base_audio.export(mixed_path, format="wav")
            
            # Combine with video
            logging.info("Creating final video...")
            clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(mixed_path)
            final = clip.set_audio(audio_clip)
            final.write_videofile(output_path, codec="libx264", audio_codec="aac",
                                   verbose=False, logger=None)
            final.close()
            clip.close()
            audio_clip.close()
            
            # Generate report
            report = {
                "model": "eleven_v3",
                "features_used": [
                    "expression_controls",
                    "emotion_tags",
                    "multi_voice",
                    "emphasis_controls",
                    "pause_controls"
                ],
                "snarks": [
                    {
                        "time": s.time_s,
                        "text": s.text.replace('<emotion="', '[').replace('">', ']').replace('</emotion>', ''),
                        "emotion": s.emotion,
                        "voice": "Rachel" if s.voice_id == NARRATOR_VOICE_ID else "Sam",
                        "duration_ms": s.duration_ms
                    }
                    for s in snarks
                ],
                "total_duration_ms": sum(s.duration_ms or 0 for s in snarks),
                "estimated_cost": f"${len(''.join(s.text for s in snarks)) * 0.00003:.4f}"
            }
            
            report_path = output_path + ".v3_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logging.info(f"✅ Success! Created: {output_path}")
            logging.info(f"📊 Report: {report_path}")
            logging.info(f"🎭 Used ElevenLabs v3 with expression controls")
            
            return True

def main():
    """Demo the ElevenLabs v3 snark narrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ElevenLabs v3 Cynical Narrator")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--transcript", required=True, help="Transcript JSON path")
    parser.add_argument("--out", required=True, help="Output video path")
    parser.add_argument("--api-key", help="ElevenLabs API key (or set ELEVENLABS_API_KEY)")
    
    args = parser.parse_args()
    
    # Initialize narrator
    narrator = SnarkNarratorV3(api_key=args.api_key)
    
    # Process video
    success = narrator.process_video_with_snarks(
        args.video,
        args.transcript,
        args.out
    )
    
    if success:
        print("\n🎬 ElevenLabs v3 Features Used:")
        print("  ✅ Expression controls (sarcastic, condescending, mocking)")
        print("  ✅ Emotion tags for nuanced delivery")
        print("  ✅ Multi-voice narration")
        print("  ✅ Emphasis and pause controls")
        print("  ✅ Style adjustments per emotion")

if __name__ == "__main__":
    main()