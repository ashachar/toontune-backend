"""
Transcript processing and enrichment handling for word-level pipeline
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.transcript.enrich_transcript import TranscriptEnricher


class TranscriptHandler:
    """Handles transcript loading, enrichment, and word timing extraction"""
    
    def __init__(self):
        self.enricher = TranscriptEnricher()
    
    def get_transcript_path(self, input_video_path: str) -> Optional[Path]:
        """Find transcript path for the given video"""
        video_dir = Path(input_video_path).parent
        video_name_no_ext = Path(input_video_path).stem
        transcript_dir = video_dir / video_name_no_ext
        transcript_path = transcript_dir / "transcript_elevenlabs_scribe.json"
        
        # If transcript not in subdirectory, check parent directory
        if not transcript_path.exists():
            transcript_path = video_dir / "transcript_elevenlabs_scribe.json"
        
        return transcript_path if transcript_path.exists() else None
    
    def load_enriched_phrases(self, transcript_path: Path, duration_seconds: float):
        """Load or generate enriched phrases from transcript"""
        print("📝 Using enriched transcript for intelligent phrasing...")
        
        # Check if enriched transcript already exists
        enriched_path = str(transcript_path).replace('transcript_elevenlabs_scribe.json', 'transcript_enriched.json')
        
        # CRITICAL FIX: Check if the enriched transcript already exists and use it
        # This prevents regenerating and re-splitting phrases
        if Path(enriched_path).exists():
            print(f"   📚 Loading existing enriched transcript from: {enriched_path}")
            with open(enriched_path, 'r') as f:
                data = json.load(f)
            
            # Convert JSON data back to EnrichedPhrase objects
            from utils.transcript.enrich_transcript import EnrichedPhrase
            enriched_phrases = []
            for phrase_dict in data['phrases']:
                phrase = EnrichedPhrase(
                    text=phrase_dict['text'],
                    words=phrase_dict['words'],
                    start_time=phrase_dict['start_time'],
                    end_time=phrase_dict['end_time'],
                    importance=phrase_dict['importance'],
                    emphasis_type=phrase_dict['emphasis_type'],
                    visual_style=phrase_dict['visual_style'],
                    appearance_index=phrase_dict['appearance_index'],
                    position=phrase_dict.get('position', 'bottom'),
                    layout_priority=phrase_dict.get('layout_priority', 2)
                )
                enriched_phrases.append(phrase)
        else:
            # Process transcript to get enriched phrases (will generate if needed)
            enriched_phrases = self.enricher.process_transcript(str(transcript_path), enriched_path)
        
        print(f"   ✓ Generated/loaded {len(enriched_phrases)} enriched phrases")
        
        # Filter phrases by duration if processing a segment
        if duration_seconds and duration_seconds < float('inf'):
            filtered_phrases = [p for p in enriched_phrases if p.start_time < duration_seconds]
            if len(filtered_phrases) < len(enriched_phrases):
                print(f"   📍 Filtering to {len(filtered_phrases)} phrases for {duration_seconds}s segment")
        else:
            filtered_phrases = enriched_phrases
        
        # Group phrases by appearance_index (scenes)
        phrase_groups = {}
        for phrase in filtered_phrases:
            idx = phrase.appearance_index
            if idx not in phrase_groups:
                phrase_groups[idx] = []
            phrase_groups[idx].append(phrase)
        
        return phrase_groups
    
    def extract_word_timings(self, phrase, transcript_path: Path) -> List[Dict]:
        """Extract precise word timings from original transcript for a phrase"""
        word_timings = []
        
        # Load the original word-level transcript to get precise timings
        with open(transcript_path, 'r') as f:
            word_transcript = json.load(f)
        
        # Match phrase words to transcript words to get actual timings
        for j, phrase_word in enumerate(phrase.words):
            # Clean the phrase word for matching
            clean_phrase_word = phrase_word.rstrip('.,!?;:')
            
            # Find matching word in transcript within the phrase time range
            found = False
            for transcript_word in word_transcript['words']:
                clean_transcript_word = transcript_word['text'].strip().rstrip('.,!?;:')
                
                # Match word and ensure it's within phrase time range
                if (clean_transcript_word == clean_phrase_word and 
                    transcript_word['start'] >= phrase.start_time - 0.1 and
                    transcript_word['start'] <= phrase.end_time + 0.1):
                    word_timings.append({
                        'start': transcript_word['start'],
                        'end': transcript_word['end']
                    })
                    found = True
                    break
            
            # Fallback to equal division if word not found
            if not found:
                print(f"       Warning: Could not find timing for word '{phrase_word}' in phrase '{phrase.text}'")
                # Calculate timing based on position in phrase
                words_per_phrase = len(phrase.words)
                time_per_word = (phrase.end_time - phrase.start_time) / words_per_phrase
                word_start = phrase.start_time + j * time_per_word
                word_end = phrase.start_time + (j + 1) * time_per_word
                word_timings.append({
                    'start': word_start,
                    'end': word_end
                })
        
        # Ensure we have timings for all words
        if len(word_timings) != len(phrase.words):
            print(f"       ERROR: Timing mismatch! {len(word_timings)} timings for {len(phrase.words)} words")
            # Fill in missing timings with equal division
            while len(word_timings) < len(phrase.words):
                idx = len(word_timings)
                time_per_word = (phrase.end_time - phrase.start_time) / len(phrase.words)
                word_timings.append({
                    'start': phrase.start_time + idx * time_per_word,
                    'end': phrase.start_time + (idx + 1) * time_per_word
                })
        
        return word_timings
    
    def convert_phrases_to_dicts(self, phrases) -> List[Dict]:
        """Convert phrase objects to dict format for layout manager"""
        phrase_dicts = []
        for phrase in phrases:
            phrase_dicts.append({
                'phrase': phrase.text,
                'importance': phrase.importance,
                'position': phrase.position,  # Add position field
                'emphasis_type': phrase.emphasis_type,  # Add for color coding
                'layout_priority': phrase.layout_priority  # Keep for compatibility
            })
        return phrase_dicts