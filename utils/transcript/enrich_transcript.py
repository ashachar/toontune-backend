"""
Enriched Transcript Generator
Uses LLM to analyze transcript and create sub-sentences with importance scores
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import requests
import yaml
from dotenv import load_dotenv


@dataclass
class EnrichedPhrase:
    """A phrase with importance score and visual properties"""
    text: str
    words: List[str]
    start_time: float
    end_time: float
    importance: float  # 0.0 to 1.0 (1.0 = most important)
    emphasis_type: str  # "minor", "normal", "question", "important", "critical", "title", "mega_title"
    visual_style: Dict  # font_size_multiplier, bold, position_offset, etc.
    appearance_index: int  # Which scene/group this phrase belongs to
    position: str = "bottom"  # "top" or "bottom" - where on screen to display
    layout_priority: int = 2  # Deprecated - kept for compatibility


class TranscriptEnricher:
    """Creates enriched transcripts with importance scores using LLM"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        # Try GEMINI_MODEL_FAST_EXPENSIVE which seems more reliable
        self.model_name = os.getenv("GEMINI_MODEL_FAST_EXPENSIVE", "gemini-1.5-flash")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Cannot proceed without LLM.")
        
        # Load prompts
        self.prompts = self.load_prompts()
    
    def load_prompts(self) -> Dict:
        """Load prompts from prompts.yaml"""
        prompts_path = Path(__file__).parent.parent.parent / "prompts.yaml"
        if prompts_path.exists():
            with open(prompts_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: prompts.yaml not found at {prompts_path}")
            return {}
    
    def load_word_transcript(self, transcript_path: str) -> List[Dict]:
        """Load word-level transcript from ElevenLabs format"""
        with open(transcript_path, 'r') as f:
            data = json.load(f)
        
        # Filter to only word types (skip spacing)
        words = [w for w in data['words'] if w['type'] == 'word']
        return words
    
    def generate_enrichment_prompt(self, words: List[Dict]) -> str:
        """Create prompt for LLM to analyze transcript using prompts.yaml template"""
        # Convert words to readable text with timing - include indices AND gaps for clarity
        text_with_timing = []
        prev_end = 0
        for i, word in enumerate(words[:150]):  # Increase limit to give more context
            gap = word['start'] - prev_end if prev_end > 0 else 0
            gap_marker = f" [GAP: {gap:.2f}s]" if gap > 0.3 else ""
            text_with_timing.append(f"[{i}] {word['text']} ({word['start']:.2f}s){gap_marker}")
            prev_end = word['end']
        
        # Use prompt from prompts.yaml if available
        if self.prompts and 'transcript_enrichment' in self.prompts:
            prompt_template = self.prompts['transcript_enrichment']['user']
            prompt = prompt_template.format(words_with_timing='\n'.join(text_with_timing))
        else:
            # This shouldn't happen if prompts.yaml is loaded correctly
            raise ValueError("transcript_enrichment prompt not found in prompts.yaml")
        
        return prompt
    
    def call_llm(self, prompt: str) -> List[Dict]:
        """Call Gemini API to get enrichment analysis"""
        # Get system prompt from prompts.yaml
        system_prompt = self.prompts['transcript_enrichment']['system']
        
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Add explicit JSON instruction for Gemini
        full_prompt += "\n\nIMPORTANT: Return ONLY valid JSON array, no markdown formatting or explanation."
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.2,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json"  # Request JSON response
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract content from Gemini response
            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Parse JSON response
                enrichment_data = json.loads(content)
                
                # The response should be a list based on our prompt
                if isinstance(enrichment_data, list):
                    return enrichment_data
                elif isinstance(enrichment_data, dict):
                    # Try common keys
                    for key in ['phrases', 'segments', 'results', 'data']:
                        if key in enrichment_data and isinstance(enrichment_data[key], list):
                            return enrichment_data[key]
                
                raise ValueError(f"Unexpected response format from Gemini: {type(enrichment_data)}")
            else:
                raise ValueError("No valid response from Gemini API")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from Gemini response: {e}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Gemini API request failed: {e}")
        except Exception as e:
            raise ValueError(f"Gemini enrichment failed: {e}")
    
    
    def create_visual_style(self, importance: float, phrase_type: str) -> Dict:
        """Generate visual style based on importance and type"""
        style = {
            "font_size_multiplier": 1.0,
            "bold": False,
            "position_y_offset": 0,
            "opacity_boost": 0,
            "color_tint": None,
            "animation_speed": 1.0
        }
        
        if phrase_type == "mega_title":
            style["font_size_multiplier"] = 1.8
            style["bold"] = True
            style["position_y_offset"] = 0  # Center
            style["color_tint"] = (255, 215, 0)  # Gold
            style["animation_speed"] = 0.5  # Very dramatic
            
        elif phrase_type == "title":
            style["font_size_multiplier"] = 1.4
            style["bold"] = True
            style["position_y_offset"] = -20
            style["color_tint"] = (255, 220, 100)  # Light golden
            style["animation_speed"] = 0.7
            
        elif phrase_type == "critical":
            style["font_size_multiplier"] = 1.3
            style["bold"] = True
            style["opacity_boost"] = 0.2
            style["color_tint"] = (255, 150, 150)  # Light red
            
        elif phrase_type == "question":
            style["font_size_multiplier"] = 1.2
            style["bold"] = False
            style["color_tint"] = (150, 200, 255)  # Light blue
            style["position_y_offset"] = -40  # Higher up
            
        elif phrase_type == "important":
            style["font_size_multiplier"] = 1.15
            style["bold"] = importance > 0.7
            style["opacity_boost"] = 0.1
            
        elif phrase_type == "minor":
            style["font_size_multiplier"] = 0.8
            style["bold"] = False
            style["opacity_boost"] = -0.1  # Slightly faded
            style["position_y_offset"] = -60  # Top of screen
            
        else:  # normal
            style["font_size_multiplier"] = 1.0
        
        return style
    
    def create_enriched_transcript(self, words: List[Dict], 
                                  enrichment_data: List[Dict]) -> List[EnrichedPhrase]:
        """Create final enriched transcript with all properties"""
        enriched_phrases = []
        
        # If LLM doesn't provide appearance_index, we'll auto-generate based on gaps
        current_appearance_index = 1
        last_end_time = 0
        
        for item in enrichment_data:
            indices = item['word_indices']
            phrase_words = [words[i] for i in indices if i < len(words)]
            
            if not phrase_words:
                continue
            
            # CRITICAL: Auto-increment appearance_index if there's a gap (pause in speech)
            # Changed from 0.5 to 0.3 seconds to catch more natural pauses
            gap = phrase_words[0]['start'] - last_end_time
            if gap > 0.3 and last_end_time > 0:  # 0.3 second gap = new scene
                current_appearance_index += 1
                print(f"   🔄 Detected gap of {gap:.2f}s → New scene {current_appearance_index}")
            
            phrase = EnrichedPhrase(
                text=item['phrase'],
                words=[w['text'] for w in phrase_words],
                start_time=phrase_words[0]['start'],
                end_time=phrase_words[-1]['end'],
                importance=item.get('importance', 0.5),
                emphasis_type=item.get('type', 'normal'),
                visual_style=self.create_visual_style(
                    item.get('importance', 0.5), 
                    item.get('type', 'normal')
                ),
                appearance_index=item.get('appearance_index', current_appearance_index),
                position=item.get('position', 'bottom'),  # Get position from LLM
                layout_priority=item.get('layout_priority', 2)  # Keep for compatibility
            )
            enriched_phrases.append(phrase)
            last_end_time = phrase_words[-1]['end']
        
        return enriched_phrases
    
    def process_transcript(self, transcript_path: str, 
                          output_path: Optional[str] = None) -> List[EnrichedPhrase]:
        """Main processing function"""
        print(f"📝 Loading word transcript: {transcript_path}")
        words = self.load_word_transcript(transcript_path)
        print(f"   Found {len(words)} words")
        
        print("🤖 Generating enrichment with Gemini...")
        prompt = self.generate_enrichment_prompt(words)
        enrichment_data = self.call_llm(prompt)
        
        print(f"   Created {len(enrichment_data)} phrases")
        
        # Create enriched transcript
        enriched = self.create_enriched_transcript(words, enrichment_data)
        
        # Calculate statistics
        avg_importance = sum(p.importance for p in enriched) / len(enriched)
        critical_count = sum(1 for p in enriched if p.emphasis_type == "critical")
        title_count = sum(1 for p in enriched if p.emphasis_type == "title")
        
        print(f"\n📊 Enrichment Statistics:")
        print(f"   Average importance: {avg_importance:.2f}")
        print(f"   Title phrases: {title_count}")
        print(f"   Critical phrases: {critical_count}")
        print(f"   Words per phrase: {len(words) / len(enriched):.1f}")
        
        # Save if output path provided
        if output_path:
            output_data = {
                "source_transcript": transcript_path,
                "total_words": len(words),
                "total_phrases": len(enriched),
                "statistics": {
                    "avg_importance": avg_importance,
                    "title_count": title_count,
                    "critical_count": critical_count
                },
                "phrases": [asdict(p) for p in enriched]
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Saved enriched transcript: {output_path}")
        
        return enriched


def main():
    """Test the enrichment system"""
    try:
        enricher = TranscriptEnricher()
        
        # Process AI Math transcript
        transcript_path = "uploads/assets/videos/ai_math1/transcript_elevenlabs_scribe.json"
        output_path = "uploads/assets/videos/ai_math1/transcript_enriched.json"
        
        if os.path.exists(transcript_path):
            enriched = enricher.process_transcript(transcript_path, output_path)
            
            # Show sample results
            print("\n🎯 Sample Enriched Phrases:")
            for i, phrase in enumerate(enriched[:10]):
                style_desc = []
                if phrase.visual_style['bold']:
                    style_desc.append("bold")
                if phrase.visual_style['font_size_multiplier'] > 1.0:
                    style_desc.append(f"{phrase.visual_style['font_size_multiplier']:.1f}x")
                
                style_str = f" [{', '.join(style_desc)}]" if style_desc else ""
                
                print(f"   {i+1}. \"{phrase.text}\" "
                      f"({phrase.start_time:.1f}-{phrase.end_time:.1f}s)")
                print(f"      Type: {phrase.emphasis_type}, "
                      f"Importance: {phrase.importance:.2f}{style_str}")
        else:
            print(f"❌ Transcript not found: {transcript_path}")
            print("   Please run word-level transcription first")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("   Make sure GEMINI_API_KEY is set in your .env file")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()