#!/usr/bin/env python3
"""Regenerate enriched transcript with improved phrase segmentation"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.transcript.enrich_transcript import TranscriptEnricher

def main():
    enricher = TranscriptEnricher()
    
    # Process AI Math transcript
    transcript_path = "uploads/assets/videos/ai_math1/transcript_elevenlabs_scribe.json"
    output_path = "uploads/assets/videos/ai_math1/transcript_enriched.json"
    
    print("🔄 Regenerating enriched transcript with improved phrase segmentation...")
    print("=" * 60)
    
    if os.path.exists(transcript_path):
        enriched = enricher.process_transcript(transcript_path, output_path)
        
        # Show sample results with improvements
        print("\n🎯 Sample Improved Phrases (first 20):")
        print("=" * 60)
        
        for i, phrase in enumerate(enriched[:20], 1):
            # Create visual indicator
            emphasis_emoji = {
                "mega_title": "🌟",
                "critical": "🔴",
                "important": "🟡",
                "question": "❓",
                "minor": "◦",
                "normal": "•"
            }.get(phrase.emphasis_type, "•")
            
            font_size = phrase.visual_style['font_size_multiplier']
            size_str = f"[{font_size:.1f}x]" if font_size != 1.0 else "[1.0x]"
            
            print(f"{i:2}. {emphasis_emoji} {size_str} \"{phrase.text}\" "
                  f"({phrase.start_time:.1f}-{phrase.end_time:.1f}s)")
        
        print("\n📊 Statistics:")
        print(f"   Total phrases: {len(enriched)}")
        print(f"   Critical phrases: {sum(1 for p in enriched if p.emphasis_type == 'critical')}")
        print(f"   Important phrases: {sum(1 for p in enriched if p.emphasis_type == 'important')}")
        print(f"   Average words per phrase: {sum(len(p.words) for p in enriched) / len(enriched):.1f}")
        
    else:
        print(f"❌ Transcript not found: {transcript_path}")

if __name__ == "__main__":
    main()