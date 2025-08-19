#!/usr/bin/env python3
"""
Test karaoke with interpolation for missing timestamps.
Creates a test transcript with some words missing timestamps.
"""

import json
from pathlib import Path
import shutil
from utils.captions.karaoke_with_interpolation import KaraokeWithInterpolation

def create_test_transcript_with_gaps():
    """Create a test transcript where some words are missing timestamps."""
    
    print("\n🔬 Creating test transcript with missing timestamps...")
    
    # Load existing transcript
    transcript_file = Path("uploads/assets/videos/do_re_mi/transcripts/transcript_words.json")
    if not transcript_file.exists():
        print("❌ Original transcript not found")
        return None
    
    # Backup original
    backup_file = transcript_file.with_suffix('.json.backup')
    if not backup_file.exists():
        shutil.copy(transcript_file, backup_file)
        print(f"✅ Backed up original to {backup_file.name}")
    
    with open(transcript_file) as f:
        data = json.load(f)
    
    words = data.get('words', [])
    
    # Deliberately remove timestamps from some words to test interpolation
    # Remove timestamps from words at indices: 2, 3 (at, the)
    # and 10, 11, 12 (to, start, When)
    missing_indices = [2, 3, 10, 11, 12, 20, 21, 30, 31, 32]
    
    modified_words = []
    missing_count = 0
    
    for i, word in enumerate(words[:50]):  # Test with first 50 words
        word_copy = word.copy()
        
        if i in missing_indices:
            # Remove timestamps
            if 'start' in word_copy:
                del word_copy['start']
            if 'end' in word_copy:
                del word_copy['end']
            missing_count += 1
            print(f"   Removed timestamp from word {i}: '{word_copy['word']}'")
        
        modified_words.append(word_copy)
    
    # Add remaining words unchanged
    modified_words.extend(words[50:])
    
    print(f"\n📊 Statistics:")
    print(f"   Total words: {len(modified_words)}")
    print(f"   Words with missing timestamps: {missing_count}")
    print(f"   Missing percentage: {missing_count/len(modified_words)*100:.1f}%")
    
    return modified_words

def test_interpolation():
    """Test the interpolation feature."""
    
    print("\n" + "🎯" * 30)
    print("  TESTING TIMESTAMP INTERPOLATION")
    print("🎯" * 30)
    
    # Create test data
    test_words = create_test_transcript_with_gaps()
    if not test_words:
        return
    
    # Get scene 1 words only
    scene1_words = [w for w in test_words if w.get('start', 0) <= 56.74 or 'start' not in w][:84]
    
    print(f"\n🎬 Testing with Scene 1 ({len(scene1_words)} words)")
    
    # Initialize generator
    generator = KaraokeWithInterpolation()
    
    # Check before interpolation
    missing_before = sum(1 for w in scene1_words if 'start' not in w or 'end' not in w)
    print(f"\n📍 Before interpolation:")
    print(f"   Words without timestamps: {missing_before}")
    
    if missing_before > 0:
        print("   Examples of missing:")
        for w in scene1_words[:20]:
            if 'start' not in w or 'end' not in w:
                print(f"     • '{w['word']}' - NO TIMESTAMP")
    
    # Run interpolation
    print("\n🔄 Running interpolation...")
    interpolated = generator.interpolate_missing_timestamps(scene1_words)
    fixed = generator.validate_and_fix_timestamps(interpolated)
    
    # Check after interpolation
    missing_after = sum(1 for w in fixed if 'start' not in w or 'end' not in w)
    interpolated_count = sum(1 for w in fixed if w.get('interpolated', False))
    
    print(f"\n✅ After interpolation:")
    print(f"   Words without timestamps: {missing_after}")
    print(f"   Words with interpolated timestamps: {interpolated_count}")
    
    if interpolated_count > 0:
        print("\n   Examples of interpolated words:")
        for w in fixed[:30]:
            if w.get('interpolated', False):
                print(f"     • '{w['word']}': {w['start']:.2f}s - {w['end']:.2f}s [INTERPOLATED]")
    
    # Verify all words now have timestamps
    if missing_after == 0:
        print("\n🎉 SUCCESS! All words now have timestamps.")
        
        # Show timeline continuity
        print("\n📈 Timeline continuity check (first 10 words):")
        for i, w in enumerate(fixed[:10]):
            gap = ""
            if i > 0:
                gap_size = w['start'] - fixed[i-1]['end']
                if gap_size > 0.01:
                    gap = f" (gap: {gap_size:.2f}s)"
            interp = " [I]" if w.get('interpolated') else ""
            print(f"   {i+1}. '{w['word']}': {w['start']:.2f}s - {w['end']:.2f}s{interp}{gap}")
    else:
        print(f"\n⚠️ Still {missing_after} words without timestamps")
    
    # Test video generation with interpolated words
    print("\n🎬 Generating test video with interpolated captions...")
    
    # Use scene 1 for testing
    input_video = "uploads/assets/videos/do_re_mi/scenes/original/scene_001.mp4"
    output_video = "uploads/assets/videos/do_re_mi/scenes/karaoke/scene_001_interpolated_test.mp4"
    
    if Path(input_video).exists():
        Path(output_video).parent.mkdir(exist_ok=True, parents=True)
        
        success = generator.generate_video(
            input_video=input_video,
            output_video=output_video,
            words=scene1_words,
            scene_start=0.0
        )
        
        if success:
            print(f"\n✅ Test video generated: {Path(output_video).name}")
            print("   All words should now be highlighted, including interpolated ones!")
        else:
            print("\n❌ Video generation failed")
    else:
        print(f"\n⚠️ Input video not found: {input_video}")

if __name__ == "__main__":
    test_interpolation()