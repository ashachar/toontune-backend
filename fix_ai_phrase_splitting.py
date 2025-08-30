#!/usr/bin/env python3
"""
Fix the issue where "AI invented a new calculus operator" is split into two phrases.
This script manually merges the two phrases in the enriched transcript.
"""

import json
from pathlib import Path


def fix_ai_phrase_splitting():
    """Merge the split AI phrases in the enriched transcript"""
    
    # Load the enriched transcript
    transcript_path = Path("uploads/assets/videos/ai_math1/transcript_enriched.json")
    
    if not transcript_path.exists():
        print(f"❌ Enriched transcript not found: {transcript_path}")
        return
    
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    phrases = data['phrases']
    merged_phrases = []
    i = 0
    
    while i < len(phrases):
        current = phrases[i]
        
        # Check if this is the split "AI" phrase (around 4.6s)
        if (current['text'] == 'AI' and 
            4.5 < current['start_time'] < 5.0 and
            i + 1 < len(phrases)):
            
            next_phrase = phrases[i + 1]
            
            # Check if next is "invented a new calculus operator"
            if next_phrase['text'] == 'invented a new calculus operator':
                print(f"✅ Found split phrases:")
                print(f"   1. '{current['text']}' at {current['start_time']:.2f}s")
                print(f"   2. '{next_phrase['text']}' at {next_phrase['start_time']:.2f}s")
                
                # Merge the two phrases
                merged = current.copy()
                merged['text'] = 'AI invented a new calculus operator'
                merged['words'] = current['words'] + next_phrase['words']
                merged['end_time'] = next_phrase['end_time']
                
                # Keep the visual properties from the main phrase
                # (they're both critical with same style anyway)
                
                print(f"   → Merged: '{merged['text']}' [{merged['start_time']:.2f}s - {merged['end_time']:.2f}s]")
                
                merged_phrases.append(merged)
                i += 2  # Skip the next phrase since we merged it
                continue
        
        # Check for any other problematic splits
        # Pattern: Single word followed by rest of sentence
        if (len(current['words']) == 1 and 
            i + 1 < len(phrases) and
            current['appearance_index'] == phrases[i + 1]['appearance_index'] and
            current['position'] == phrases[i + 1]['position'] and
            phrases[i + 1]['start_time'] - current['end_time'] < 0.2):  # Very close timing
            
            next_phrase = phrases[i + 1]
            
            print(f"⚠️  Found potential split:")
            print(f"   1. '{current['text']}' at {current['start_time']:.2f}s")
            print(f"   2. '{next_phrase['text']}' at {next_phrase['start_time']:.2f}s")
            
            # Only merge if they're semantically connected
            # Check if the single word could be the subject of the next phrase
            if current['text'] in ['AI', 'We', 'It', 'The', 'A']:
                # Merge them
                merged = current.copy()
                merged['text'] = f"{current['text']} {next_phrase['text']}"
                merged['words'] = current['words'] + next_phrase['words']
                merged['end_time'] = next_phrase['end_time']
                
                # Use the more important visual style
                if next_phrase['importance'] > current['importance']:
                    merged['importance'] = next_phrase['importance']
                    merged['emphasis_type'] = next_phrase['emphasis_type']
                    merged['visual_style'] = next_phrase['visual_style']
                
                print(f"   → Merged: '{merged['text']}' [{merged['start_time']:.2f}s - {merged['end_time']:.2f}s]")
                
                merged_phrases.append(merged)
                i += 2  # Skip next
                continue
        
        # No merge needed, keep the phrase as is
        merged_phrases.append(current)
        i += 1
    
    # Update the data
    data['phrases'] = merged_phrases
    data['total_phrases'] = len(merged_phrases)
    
    # Save backup
    backup_path = transcript_path.with_suffix('.json.bak')
    with open(backup_path, 'w') as f:
        json.dump(json.load(open(transcript_path)), f, indent=2)
    print(f"\n💾 Saved backup to: {backup_path}")
    
    # Save fixed version
    with open(transcript_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Fixed enriched transcript saved to: {transcript_path}")
    print(f"   Original phrases: {len(phrases)}")
    print(f"   After merging: {len(merged_phrases)}")
    print(f"   Merged: {len(phrases) - len(merged_phrases)} phrase pairs")


if __name__ == "__main__":
    fix_ai_phrase_splitting()