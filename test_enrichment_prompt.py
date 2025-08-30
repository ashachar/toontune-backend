#!/usr/bin/env python3
"""Test the enrichment prompt to see what will be sent to the LLM"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.transcript.enrich_transcript import TranscriptEnricher
import json

def main():
    enricher = TranscriptEnricher()
    
    # Load transcript
    transcript_path = "uploads/assets/videos/ai_math1/transcript_elevenlabs_scribe.json"
    
    if os.path.exists(transcript_path):
        print("=" * 80)
        print("TESTING ENRICHMENT PROMPT")
        print("=" * 80)
        
        # Load words
        words = enricher.load_word_transcript(transcript_path)
        print(f"\n📝 Loaded {len(words)} words from transcript")
        
        # Generate prompt
        prompt = enricher.generate_enrichment_prompt(words)
        
        print("\n🤖 SYSTEM PROMPT:")
        print("-" * 40)
        if enricher.prompts and 'transcript_enrichment' in enricher.prompts:
            print(enricher.prompts['transcript_enrichment']['system'])
        
        print("\n📋 USER PROMPT (first 2000 chars):")
        print("-" * 40)
        print(prompt[:2000])
        print("...")
        
        print("\n📊 PROMPT STATISTICS:")
        print(f"   Total prompt length: {len(prompt)} characters")
        print(f"   Words included in prompt: 150 (max)")
        
        # Show what the mock enrichment would generate for comparison
        print("\n🔄 MOCK ENRICHMENT (for comparison):")
        print("-" * 40)
        mock_data = enricher.generate_mock_enrichment(words)
        
        print(f"   Generated {len(mock_data)} phrases")
        print("\n   First 5 phrases from mock:")
        for i, item in enumerate(mock_data[:5], 1):
            print(f"   {i}. \"{item['phrase']}\" - {item['type']} ({item['importance']:.2f})")
        
    else:
        print(f"❌ Transcript not found: {transcript_path}")

if __name__ == "__main__":
    main()