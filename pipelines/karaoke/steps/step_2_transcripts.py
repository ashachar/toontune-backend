"""
Step 2: Transcript Generation
==============================

Generates sentence and word level transcripts using OpenAI Whisper API.
"""

import os
import sys
import json
from pathlib import Path


class TranscriptsStep:
    """Handles transcript generation operations."""
    
    def __init__(self, pipeline_state, dirs, config):
        """Initialize with pipeline references."""
        self.pipeline_state = pipeline_state
        self.dirs = dirs
        self.config = config
    
    def run(self, video_path):
        """Generate sentence and word level transcripts using OpenAI Whisper API."""
        print("\n" + "-"*60)
        print("STEP 2: GENERATING TRANSCRIPTS")
        print("-"*60)
        
        # Check if transcripts already exist
        sentences_file = self.dirs['transcripts'] / 'transcript_sentences.json'
        words_file = self.dirs['transcripts'] / 'transcript_words.json'
        
        if sentences_file.exists() and words_file.exists():
            print(f"  ✓ Transcripts already exist")
            print(f"    - Sentences: {sentences_file}")
            print(f"    - Words: {words_file}")
        else:
            if self.config.dry_run:
                print("  [DRY RUN] Would generate transcripts using Whisper API")
                self.create_mock_transcripts()
            else:
                # Use Whisper API to generate real transcripts
                print("  Generating transcripts using OpenAI Whisper API...")
                
                # Check for API key
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    print("  ⚠ OPENAI_API_KEY not found in environment")
                    print("  Falling back to mock transcripts")
                    self.create_mock_transcripts()
                else:
                    # Import the Whisper transcript extraction functions
                    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils" / "video_utils"))
                    from extract_transcript_whisper import extract_audio, get_transcript_from_whisper
                    
                    try:
                        # Extract audio from video
                        audio_dir = self.dirs['transcripts'] / 'audio'
                        audio_path = extract_audio(video_path, audio_dir)
                        
                        if audio_path:
                            # Get transcript from Whisper
                            transcript_data = get_transcript_from_whisper(audio_path)
                            
                            if transcript_data and 'transcript' in transcript_data:
                                # Convert to our format and save
                                self.save_whisper_transcripts(transcript_data)
                                print(f"  ✓ Transcripts generated successfully")
                            else:
                                print("  ⚠ Could not generate transcript, using mock data")
                                self.create_mock_transcripts()
                        else:
                            print("  ⚠ Could not extract audio, using mock data")
                            self.create_mock_transcripts()
                            
                    except Exception as e:
                        print(f"  ⚠ Error generating transcript: {e}")
                        print("  Falling back to mock transcripts")
                        self.create_mock_transcripts()
        
        # Store transcript paths in pipeline state
        self.pipeline_state['transcripts'] = {
            'sentences': str(sentences_file),
            'words': str(words_file)
        }
        self.pipeline_state['steps_completed'].append('generate_transcripts')
    
    def create_mock_transcripts(self):
        """Create mock transcripts for testing."""
        # Sentence-level transcript - covering full video duration
        sentences = {
            "sentences": [
                {"text": "Let's start at the very beginning", "start": 0.0, "end": 6.5},
                {"text": "A very good place to start", "start": 6.5, "end": 10.0},
                {"text": "When you read you begin with ABC", "start": 10.0, "end": 13.0},
                {"text": "When you sing you begin with Do Re Mi", "start": 13.0, "end": 17.0},
                {"text": "Do Re Mi, the first three notes just happen to be", "start": 17.0, "end": 22.0},
                {"text": "Do Re Mi, Do Re Mi Fa So La Ti", "start": 22.0, "end": 28.0},
                {"text": "Now you make it easier", "start": 28.0, "end": 31.0},
                {"text": "Do is a deer, a female deer", "start": 31.0, "end": 35.0},
                {"text": "Re is a drop of golden sun", "start": 35.0, "end": 39.0},
                {"text": "Mi is a name I call myself", "start": 39.0, "end": 43.0},
                {"text": "Fa is a long long way to run", "start": 43.0, "end": 47.0},
                {"text": "So is a needle pulling thread", "start": 47.0, "end": 51.0},
                {"text": "La is a note to follow So", "start": 51.0, "end": 54.0},
                {"text": "Ti is a drink with jam and bread", "start": 54.0, "end": 57.0},
                {"text": "That will bring us back to Do", "start": 57.0, "end": 60.0}
            ]
        }
        
        # Word-level transcript
        words = {
            "words": [
                {"word": "Let's", "start": 0.0, "end": 0.5},
                {"word": "start", "start": 0.5, "end": 1.0},
                {"word": "at", "start": 1.0, "end": 1.2},
                {"word": "the", "start": 1.2, "end": 1.4},
                {"word": "very", "start": 1.4, "end": 1.8},
                {"word": "beginning", "start": 1.8, "end": 2.5}
            ]
        }
        
        # Save transcripts
        sentences_file = self.dirs['transcripts'] / 'transcript_sentences.json'
        words_file = self.dirs['transcripts'] / 'transcript_words.json'
        
        with open(sentences_file, 'w') as f:
            json.dump(sentences, f, indent=2)
        print(f"  ✓ Sentence transcript saved to: {sentences_file}")
        
        with open(words_file, 'w') as f:
            json.dump(words, f, indent=2)
        print(f"  ✓ Word transcript saved to: {words_file}")
        
        self.pipeline_state['transcripts'] = {
            'sentences': str(sentences_file),
            'words': str(words_file)
        }
    
    def save_whisper_transcripts(self, transcript_data):
        """Convert and save Whisper transcript data to our format."""
        # Convert sentence-level transcript
        sentences = {"sentences": []}
        if 'transcript' in transcript_data:
            for segment in transcript_data['transcript']:
                sentences["sentences"].append({
                    "text": segment['text'].strip(),
                    "start": segment['start_ms'] / 1000.0,
                    "end": segment['end_ms'] / 1000.0
                })
        
        # Convert word-level transcript
        words = {"words": []}
        if 'word_timestamps' in transcript_data:
            for word_data in transcript_data['word_timestamps']:
                words["words"].append({
                    "word": word_data['word'],
                    "start": word_data['start_ms'] / 1000.0,
                    "end": word_data['end_ms'] / 1000.0
                })
        
        # Save transcripts
        sentences_file = self.dirs['transcripts'] / 'transcript_sentences.json'
        words_file = self.dirs['transcripts'] / 'transcript_words.json'
        
        with open(sentences_file, 'w') as f:
            json.dump(sentences, f, indent=2)
        print(f"  ✓ Sentence transcript ({len(sentences['sentences'])} segments) saved to: {sentences_file}")
        
        with open(words_file, 'w') as f:
            json.dump(words, f, indent=2)
        print(f"  ✓ Word transcript ({len(words['words'])} words) saved to: {words_file}")