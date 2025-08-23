#!/usr/bin/env python3
"""
Demonstrate ElevenLabs v3 features and API structure
Shows how emotion tags, expression controls, and dialog work
"""

import json
import os

def demonstrate_v3_features():
    """Show all the v3 features for cynical narration"""
    
    print("=" * 70)
    print("🎭 ELEVENLABS V3 CYNICAL NARRATOR - FEATURE DEMONSTRATION")
    print("=" * 70)
    
    # 1. Expression Control Examples
    print("\n📝 1. EXPRESSION CONTROLS WITH EMOTION TAGS:")
    print("-" * 50)
    
    expressions = [
        {
            "emotion": "sarcastic",
            "text": '<emotion="sarcastic">Oh good, another musical number.</emotion>',
            "settings": {"stability": 0.2, "style": 0.9, "similarity_boost": 0.7}
        },
        {
            "emotion": "condescending", 
            "text": '<emotion="condescending">Yes, because ABC is <emphasis>such</emphasis> complex knowledge.</emotion>',
            "settings": {"stability": 0.3, "style": 0.8, "similarity_boost": 0.6}
        },
        {
            "emotion": "deadpan",
            "text": '<emotion="deadpan">Revolutionary. A deer is... a deer.</emotion>',
            "settings": {"stability": 0.9, "style": 0.1, "similarity_boost": 0.8}
        },
        {
            "emotion": "mocking",
            "text": '<emotion="mocking">We get it. You can repeat three syllables.</emotion>',
            "settings": {"stability": 0.2, "style": 0.85, "similarity_boost": 0.65}
        }
    ]
    
    for expr in expressions:
        print(f"\n🎯 {expr['emotion'].upper()}:")
        print(f"   Text: {expr['text']}")
        print(f"   Settings: stability={expr['settings']['stability']}, "
              f"style={expr['settings']['style']}")
    
    # 2. Multi-Speaker Dialog
    print("\n\n🗣️ 2. MULTI-SPEAKER DIALOG STRUCTURE:")
    print("-" * 50)
    
    dialog = {
        "model_id": "eleven_v3",
        "dialog": [
            {
                "speaker": "Narrator",
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "text": '<emotion="sarcastic">Let me add some honest commentary to this masterpiece.</emotion>'
            },
            {
                "speaker": "Cynic1",
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "text": '<emotion="sarcastic">Oh good, another musical number.</emotion>',
                "timing": {"start_time": 3.8}
            },
            {
                "speaker": "Cynic2",
                "voice_id": "yoZ06aMxZJJ28mfd3POQ",
                "text": '<emotion="condescending">ABC is such complex knowledge.</emotion>',
                "timing": {"start_time": 16.8}
            }
        ]
    }
    
    print("\nDialog JSON structure for v3 API:")
    print(json.dumps(dialog, indent=2))
    
    # 3. Advanced Controls
    print("\n\n⚙️ 3. ADVANCED V3 CONTROLS:")
    print("-" * 50)
    
    advanced_examples = [
        "Pause control: <pause duration='500ms'/> for dramatic effect",
        "Emphasis: This is <emphasis>really</emphasis> important",
        "Speed: <prosody rate='slow'>Speaking slowly</prosody> for effect",
        "Pitch: <prosody pitch='+20Hz'>Higher pitch</prosody> for sarcasm",
        "Combined: <emotion='sarcastic'><prosody rate='slow'>Oh... really?</prosody></emotion>"
    ]
    
    for i, example in enumerate(advanced_examples, 1):
        print(f"{i}. {example}")
    
    # 4. API Call Structure
    print("\n\n🔧 4. V3 API CALL STRUCTURE (Python):")
    print("-" * 50)
    
    api_example = '''
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key="your-key")

# Single cynical remark with expression
audio = client.text_to_speech.convert(
    text='<emotion="sarcastic">Oh good, another musical.</emotion>',
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model_id="eleven_v3",  # ← IMPORTANT: Use v3 model
    voice_settings={
        "stability": 0.2,      # Low for more expression
        "similarity_boost": 0.7,
        "style": 0.9,          # High for emotional range
        "use_speaker_boost": True
    },
    output_format="mp3_44100_128"
)

# Save the audio
with open("cynical_remark.mp3", "wb") as f:
    f.write(audio)
'''
    print(api_example)
    
    # 5. Cost Estimation
    print("\n💰 5. COST ESTIMATION:")
    print("-" * 50)
    
    snarks = [
        "Oh good, another musical number. How original.",
        "Yes, because ABC is such complex knowledge.",
        "We get it. You can repeat three syllables.",
        "Easier? This is your idea of pedagogy?",
        "Revolutionary. A deer is... a deer.",
        "Mi, the narcissism is showing."
    ]
    
    total_chars = sum(len(s) for s in snarks)
    # ElevenLabs v3 pricing (example - check current rates)
    cost_per_1k_chars = 0.30  # $0.30 per 1000 characters
    estimated_cost = (total_chars / 1000) * cost_per_1k_chars
    
    print(f"Total characters: {total_chars}")
    print(f"Estimated cost: ${estimated_cost:.4f}")
    print(f"Per snark: ${estimated_cost/len(snarks):.4f}")
    
    # 6. Integration with Auto-Snark
    print("\n\n🚀 6. INTEGRATION WITH AUTO-SNARK NARRATOR:")
    print("-" * 50)
    
    print("""
To use with your Do-Re-Mi video:

1. Set your API key:
   export ELEVENLABS_API_KEY="your-key-here"

2. Run the v3 narrator:
   python snark_narrator_elevenlabs_v3.py \\
     --video scene_001.mp4 \\
     --transcript do_re_mi_transcript.json \\
     --out do_re_mi_v3_cynical.mp4

3. Features automatically applied:
   • Sarcastic/condescending tones
   • Emphasis on key words
   • Pauses for dramatic effect
   • Multiple voice personalities
   • Emotion-based voice settings
""")
    
    print("\n" + "=" * 70)
    print("✨ ElevenLabs v3 provides cinema-quality cynical narration!")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_v3_features()