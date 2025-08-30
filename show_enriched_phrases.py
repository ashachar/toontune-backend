#!/usr/bin/env python3
"""Display enriched transcript phrases in a readable format"""

import json
from pathlib import Path

def show_phrases():
    # Load enriched transcript
    transcript_path = Path("uploads/assets/videos/ai_math1/transcript_enriched.json")
    
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("ENRICHED TRANSCRIPT PHRASES")
    print("=" * 80)
    print(f"Total phrases: {data['total_phrases']}")
    print(f"Average importance: {data['statistics']['avg_importance']:.2f}")
    print(f"Critical phrases: {data['statistics']['critical_count']}")
    print("=" * 80)
    print()
    
    # Show first 30 phrases for the demo
    for i, phrase in enumerate(data['phrases'][:30], 1):
        # Create visual indicator of importance
        importance_bar = "█" * int(phrase['importance'] * 10)
        importance_bar = importance_bar.ljust(10, '░')
        
        # Color coding for emphasis type
        emphasis_emoji = {
            "mega_title": "🌟",
            "critical": "🔴",
            "important": "🟡",
            "question": "❓",
            "minor": "◦",
            "normal": "•"
        }.get(phrase['emphasis_type'], "•")
        
        # Font size indicator
        font_size = phrase['visual_style']['font_size_multiplier']
        size_indicator = f"{font_size:.1f}x" if font_size != 1.0 else "   "
        
        print(f"{i:3}. {emphasis_emoji} [{importance_bar}] {size_indicator} "
              f"({phrase['start_time']:5.2f}s-{phrase['end_time']:5.2f}s)")
        print(f"     \"{phrase['text']}\"")
        print(f"     Type: {phrase['emphasis_type']:<12} Importance: {phrase['importance']:.2f}")
        print()

if __name__ == "__main__":
    show_phrases()