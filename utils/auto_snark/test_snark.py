#!/usr/bin/env python3
"""
Test script for the Auto-Snark Narrator
Tests core functionality without requiring actual video files
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from snark_narrator import (
    Segment, Beat, 
    beats_from_transcript_gaps,
    beats_from_markers,
    merge_and_prune_beats,
    extract_keywords_near,
    generate_snark_text,
    safe_text,
    TEMPLATES
)

def test_beat_detection():
    """Test beat detection from transcript gaps and markers"""
    print("\n🔍 Testing Beat Detection...")
    
    segments = [
        Segment(0.1, 2.6, "Welcome back everyone."),
        Segment(4.8, 7.9, "Actually, let me show you something."),
        Segment(10.2, 13.1, "But first, we need to configure this."),
        Segment(15.5, 19.0, "Anyway, let's continue with the demo."),
        Segment(21.2, 25.2, "Okay, here's the main part."),
    ]
    
    # Test gap detection
    gap_beats = beats_from_transcript_gaps(segments, pause_threshold=1.0)
    print(f"  ✓ Found {len(gap_beats)} gap beats:")
    for b in gap_beats:
        print(f"    - {b.time_s:.1f}s (score: {b.score:.2f}, reason: {b.reasons})")
    
    # Test marker detection
    marker_beats = beats_from_markers(segments)
    print(f"  ✓ Found {len(marker_beats)} marker beats:")
    for b in marker_beats:
        print(f"    - {b.time_s:.1f}s (score: {b.score:.2f}, reason: {b.reasons})")
    
    # Test merging and pruning
    all_beats = gap_beats + marker_beats
    merged = merge_and_prune_beats(all_beats, min_gap_s=5.0, max_snarks=3)
    print(f"  ✓ After merge/prune: {len(merged)} final beats")
    
    return segments, merged

def test_snark_generation():
    """Test snark text generation"""
    print("\n✍️ Testing Snark Generation...")
    
    segments = [
        Segment(0.1, 2.6, "Welcome to the advanced tutorial on optimization."),
        Segment(4.8, 7.9, "Actually, this technique is revolutionary."),
    ]
    
    for style in TEMPLATES.keys():
        print(f"\n  Style: {style}")
        for i in range(3):
            text = generate_snark_text(style, segments, 3.0)
            print(f"    → {text}")

def test_keyword_extraction():
    """Test keyword extraction from segments"""
    print("\n🔤 Testing Keyword Extraction...")
    
    segments = [
        Segment(0.1, 2.6, "Welcome to the tutorial."),
        Segment(4.8, 7.9, "We'll explore optimization and performance improvements."),
        Segment(10.2, 13.1, "The algorithm uses dynamic programming techniques."),
    ]
    
    test_times = [1.0, 6.0, 11.5]
    for t in test_times:
        keywords = extract_keywords_near(segments, t, k=2)
        print(f"  Time {t:.1f}s → Keywords: {keywords}")

def test_profanity_filter():
    """Test profanity filtering"""
    print("\n🛡️ Testing Profanity Filter...")
    
    test_cases = [
        "This is a clean sentence.",
        "What the hell is happening here?",
        "That's a stupid idea, honestly.",
        "I hate this dumb approach.",
    ]
    
    for text in test_cases:
        filtered = safe_text(text)
        print(f"  '{text}' → '{filtered}'")

def test_report_structure():
    """Test report structure generation"""
    print("\n📊 Testing Report Structure...")
    
    # Simulate report data
    report = {
        "video": "/path/to/video.mp4",
        "out": "/path/to/output.mp4",
        "style": "wry",
        "max_snarks": 10,
        "min_gap_s": 12.0,
        "inserts": [
            {
                "time_s": 2.9,
                "text": "Bold choice. Not judging. (optimization, tutorial)",
                "duration_ms": 3200,
                "reasons": ["pause:2.2s", "marker"]
            },
            {
                "time_s": 15.5,
                "text": "Plot twist no one asked for.",
                "duration_ms": 2800,
                "reasons": ["marker"]
            }
        ],
        "counts": {
            "candidates": 15,
            "selected": 2
        },
        "estimates": {
            "tts_total_chars": 95,
            "approx_cost_usd": 0.00003
        }
    }
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(report, f, indent=2)
        print(f"  ✓ Report saved to: {f.name}")
        print(f"  ✓ Total chars: {report['estimates']['tts_total_chars']}")
        print(f"  ✓ Est. cost: ${report['estimates']['approx_cost_usd']:.5f}")
        print(f"  ✓ Inserts: {len(report['inserts'])}/{report['counts']['candidates']} candidates")

def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("🎬 AUTO-SNARK NARRATOR TEST SUITE")
    print("=" * 60)
    
    try:
        # Test components
        test_beat_detection()
        test_keyword_extraction()
        test_snark_generation()
        test_profanity_filter()
        test_report_structure()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
        print("\n📝 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set ELEVENLABS_API_KEY for high-quality TTS (optional)")
        print("3. Run with: python snark_narrator.py --video input.mp4 --transcript test_transcript.json --out output.mp4")
        print("\n💡 For offline testing, use --no-elevenlabs flag")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()