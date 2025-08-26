"""Sophisticated context-aware comment generation."""

import json
import random
import requests
from typing import List, Dict, Optional, Set, Tuple

from utils.auto_comment.pipeline.models import Comment, SilenceGap
from utils.auto_comment.pipeline.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_COMMENTS


class CommentGenerator:
    """Generates sophisticated, field-specific professional comments."""
    
    def __init__(self):
        self.gemini_key = GEMINI_API_KEY
        self.field_cache = None
        self.expertise_level = None
        
    def generate_contextual_comments(
        self, 
        transcript: Dict, 
        gaps: List[SilenceGap]
    ) -> List[Comment]:
        """Generate sophisticated, context-aware professional comments."""
        print("🤖 Generating contextual comments...")
        
        # First, analyze the field and expertise level
        if self.gemini_key:
            self.field_cache, self.expertise_level = self._detect_field_and_expertise(transcript)
            print(f"  📚 Detected field: {self.field_cache}")
            print(f"  🎓 Expertise level: {self.expertise_level}")
        
        comments = []
        used_texts = set()
        last_comment_time = -float('inf')  # Track time of last added comment
        
        # Process gaps for sophisticated comments
        for i, gap in enumerate(gaps[:MAX_COMMENTS]):
            # CRITICAL: Enforce 20-second minimum gap between consecutive comments
            time_since_last = gap.start - last_comment_time
            if time_since_last < 20.0:
                print(f"  ⏭️ Skipping gap at {gap.start:.1f}s (only {time_since_last:.1f}s since last comment, need 20s)")
                continue
            
            # Get surrounding context (before and after the gap)
            context_before, context_after = self._get_surrounding_context(
                transcript, gap.start, gap.duration
            )
            
            # Generate sophisticated comment
            if self.gemini_key:
                comment = self._generate_sophisticated_comment(
                    context_before, 
                    context_after, 
                    gap.duration, 
                    used_texts,
                    is_beginning=(i == 0 and gap.start == 0),
                    is_end=(i == len(gaps) - 1)
                )
            else:
                comment = self._get_sophisticated_fallback(
                    context_before, gap.duration, used_texts
                )
            
            if comment and comment["text"].lower() not in used_texts:
                comments.append(Comment(
                    text=comment["text"],
                    time=gap.start + 0.2,
                    emotion=comment["emotion"],
                    context=context_before[:50] if context_before else "",
                    gap_duration=gap.duration
                ))
                used_texts.add(comment["text"].lower())
                last_comment_time = gap.start  # Update time of last added comment
                print(f"  ✨ New: \"{comment['text']}\" ({comment['emotion']}) at {gap.start:.1f}s")
        
        print(f"✅ Total comments: {len(comments)} (0 reused, {len(comments)} new)")
        return comments
    
    def _detect_field_and_expertise(self, transcript: Dict) -> Tuple[str, str]:
        """Detect the field/domain and appropriate expertise level."""
        # Combine first several segments to understand the topic
        sample_text = " ".join(
            seg["text"] for seg in transcript["segments"][:min(10, len(transcript["segments"]))]
        )
        
        prompt = f"""Analyze this transcript excerpt and identify:
1. The primary field/domain (e.g., "AI/Machine Learning", "Mathematics", "Physics", "Biology", etc.)
2. The technical sophistication level (e.g., "research-level", "professional", "educational")

Transcript: {sample_text[:500]}

Respond in JSON format:
{{
  "field": "specific field name",
  "expertise": "expertise level",
  "key_concepts": ["concept1", "concept2", "concept3"]
}}"""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={self.gemini_key}"
            
            response = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 200
                }
            })
            
            if response.status_code == 200:
                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                # Clean and parse JSON
                text = text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                data = json.loads(text)
                return data.get("field", "Technology"), data.get("expertise", "professional")
        except Exception as e:
            print(f"  ⚠️ Field detection failed: {e}")
        
        return "Technology", "professional"
    
    def _get_surrounding_context(
        self, transcript: Dict, gap_start: float, gap_duration: float
    ) -> Tuple[str, str]:
        """Get text context before and after the gap."""
        context_before = ""
        context_after = ""
        
        for seg in transcript["segments"]:
            # Context before gap (last 2 segments)
            if seg["end"] <= gap_start and seg["end"] > gap_start - 10:
                context_before = seg["text"] + " " + context_before
            # Context after gap (next segment)
            elif seg["start"] >= gap_start + gap_duration and seg["start"] < gap_start + gap_duration + 5:
                if not context_after:
                    context_after = seg["text"]
        
        return context_before.strip(), context_after.strip()
    
    def _generate_sophisticated_comment(
        self, 
        context_before: str, 
        context_after: str,
        max_duration: float,
        used_texts: Set[str],
        is_beginning: bool = False,
        is_end: bool = False
    ) -> Optional[Dict]:
        """Generate sophisticated, field-specific professional comment."""
        
        # Calculate max words based on duration
        max_words = min(int((max_duration - 0.3) * 2.2), 12)
        
        # Build sophisticated prompt based on field
        field_prompts = {
            "AI/Machine Learning": {
                "style": "Like a senior ML researcher or AI architect",
                "examples": [
                    "Elegant abstraction of the loss surface",
                    "The convergence dynamics here are non-trivial",
                    "Reminds me of the lottery ticket hypothesis",
                    "Classic bias-variance tradeoff",
                    "The attention mechanism parallels human cognition",
                    "Fascinating emergence properties"
                ]
            },
            "Mathematics": {
                "style": "Like a mathematics professor or researcher",
                "examples": [
                    "The proof structure is remarkably elegant",
                    "This generalizes beautifully to n-dimensions",
                    "Ah, the classic epsilon-delta argument",
                    "The topology here is quite intricate",
                    "Non-trivial use of the pigeonhole principle",
                    "The symmetry is doing heavy lifting here"
                ]
            },
            "Physics": {
                "style": "Like a theoretical physicist",
                "examples": [
                    "The gauge invariance is crucial here",
                    "Beautiful manifestation of Noether's theorem",
                    "The phase transition is second-order",
                    "Renormalization saves the day again",
                    "The symmetry breaking is spontaneous"
                ]
            }
        }
        
        field_info = field_prompts.get(
            self.field_cache, 
            {
                "style": "Like a thoughtful expert professional",
                "examples": [
                    "The systematic approach here is noteworthy",
                    "This framework scales elegantly",
                    "The abstraction layers are well-designed",
                    "Interesting architectural choice",
                    "The optimization is quite clever"
                ]
            }
        )
        
        # Special handling for beginning and end
        if is_beginning:
            position_context = "This is the BEGINNING of the presentation. Make a sophisticated opening observation that shows you understand what's about to be discussed."
        elif is_end:
            position_context = "This is the END of the presentation. Make a sophisticated closing remark that synthesizes or reflects on what was discussed."
        else:
            position_context = "This is during the presentation. Make an insightful observation about the specific concept being discussed."
        
        prompt = f"""You are a highly sophisticated expert in {self.field_cache}. Generate a brief, professional comment that could be made by someone with deep expertise.

Context before gap: {context_before[-200:] if context_before else 'Start of presentation'}
Context after gap: {context_after[:100] if context_after else 'End of presentation'}

{position_context}

Requirements:
- Maximum {max_words} words
- Sound like {field_info['style']}
- Be specific to the actual content, not generic
- Show deep understanding and insight
- Professional but can be slightly informal (as if thinking out loud)
- NEVER use these already-used phrases: {', '.join(used_texts) if used_texts else 'none'}
- IMPORTANT: Comments should be spaced at least 20 seconds apart for natural pacing

Example sophisticated comments for inspiration (DO NOT copy these exactly):
{chr(10).join('- ' + ex for ex in random.sample(field_info['examples'], min(3, len(field_info['examples']))))}

Respond in JSON format:
{{
  "text": "your sophisticated comment",
  "emotion": "thoughtful/analytical/impressed/curious/contemplative"
}}"""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={self.gemini_key}"
            
            response = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.8,
                    "maxOutputTokens": 100
                }
            })
            
            if response.status_code == 200:
                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                print(f"    🤖 Gemini response: {text[:100]}...")
                
                # Parse JSON response
                text = text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                
                comment_data = json.loads(text)
                
                # Validate the comment
                comment_text = comment_data.get("text", "").strip()
                if comment_text and len(comment_text.split()) <= max_words:
                    print(f"    ✓ Generated: \"{comment_text}\"")
                    return {
                        "text": comment_text,
                        "emotion": comment_data.get("emotion", "thoughtful")
                    }
            else:
                print(f"  ⚠️ Gemini API error: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"  ⚠️ AI generation failed: {e}, using sophisticated fallback")
        
        return self._get_sophisticated_fallback(context_before, max_duration, used_texts)
    
    def _get_sophisticated_fallback(
        self, context: str, max_duration: float, used_texts: Set[str]
    ) -> Dict:
        """Get sophisticated fallback comments when AI is unavailable."""
        
        # Sophisticated fallbacks by duration
        if max_duration < 1.5:
            short_options = [
                ("Precisely.", "analytical"),
                ("Indeed.", "thoughtful"),
                ("Elegant.", "impressed"),
                ("Non-trivial.", "analytical"),
                ("Fascinating.", "curious")
            ]
        elif max_duration < 2.5:
            medium_options = [
                ("The implications are profound.", "thoughtful"),
                ("Textbook elegant solution.", "impressed"),
                ("This generalizes nicely.", "analytical"),
                ("The abstraction is key.", "contemplative"),
                ("Solid theoretical foundation.", "analytical")
            ]
        else:
            long_options = [
                ("The mathematical rigor here is impressive.", "impressed"),
                ("This framework scales beautifully.", "analytical"),
                ("Classic demonstration of first principles thinking.", "thoughtful"),
                ("The convergence properties are quite remarkable.", "analytical"),
                ("Excellent use of dimensional analysis.", "impressed")
            ]
            options = long_options
        
        # Choose appropriate option based on duration
        if max_duration < 1.5:
            options = short_options
        elif max_duration < 2.5:
            options = medium_options
        else:
            options = long_options
        
        # Filter out used options
        available = [
            {"text": text, "emotion": emotion}
            for text, emotion in options
            if text.lower() not in used_texts
        ]
        
        if available:
            return random.choice(available)
        
        # Last resort sophisticated comment
        return {"text": "Intriguing approach.", "emotion": "thoughtful"}