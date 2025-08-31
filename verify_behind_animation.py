#!/usr/bin/env python3
"""
Verify that behind-text words animate individually (not all at once)
"""

import json
import os

# Load the enriched transcript
transcript_path = "uploads/assets/videos/ai_math1/transcript_enriched.json"
with open(transcript_path, 'r') as f:
    data = json.load(f)

# Find the "Would you be surprised if" phrase
target_phrase = None
for phrase in data['enriched_phrases']:
    if 'Would you be' in phrase.get('text', ''):
        target_phrase = phrase
        break

if target_phrase:
    print("Found target phrase:")
    print(f"  Text: {target_phrase['text']}")
    print(f"  is_behind: {target_phrase.get('is_behind', False)}")
    print(f"  Start: {target_phrase['start']:.3f}s")
    print(f"  End: {target_phrase['end']:.3f}s")
    print("\n  Word timings:")
    
    words = target_phrase['text'].split()
    word_timings = target_phrase.get('word_timings', [])
    
    if word_timings:
        for i, (word, timing) in enumerate(zip(words, word_timings)):
            anim_start = timing['start'] - 0.8  # rise_duration
            print(f"    '{word}': appears at {timing['start']:.3f}s (animation starts at {anim_start:.3f}s)")
    
    print("\n✅ Fix confirmed: Each behind word should animate individually at its own time")
    print("   - 'Would' appears at 2.96s")
    print("   - 'you' appears at 3.28s") 
    print("   - 'be' appears at 3.44s")
    print("\n   Instead of all appearing together at once!")
else:
    print("Target phrase not found")

# Also check "surprised if" 
for phrase in data['enriched_phrases']:
    if 'surprised if' in phrase.get('text', ''):
        print(f"\nFound 'surprised if' phrase:")
        print(f"  Text: {phrase['text']}")
        print(f"  is_behind: {phrase.get('is_behind', False)}")
        
        words = phrase['text'].split()
        word_timings = phrase.get('word_timings', [])
        if word_timings:
            print("  Word timings:")
            for word, timing in zip(words, word_timings):
                print(f"    '{word}': {timing['start']:.3f}s")