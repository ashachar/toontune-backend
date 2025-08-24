"""Comment generation logic."""

import random
from typing import List, Dict, Optional, Set

from utils.auto_comment.pipeline.models import Snark, SilenceGap
from utils.auto_comment.pipeline.config import GEMINI_API_KEY, MAX_SNARKS


class CommentGenerator:
    """Generates contextual comments for video gaps."""
    
    def __init__(self):
        self.gemini_key = GEMINI_API_KEY
        
    def generate_contextual_snarks(
        self, 
        transcript: Dict, 
        gaps: List[SilenceGap]
    ) -> List[Snark]:
        """Generate contextual snarks for gaps."""
        print("🤖 Generating contextual snarks...")
        
        snarks = []
        used_texts = set()
        reused_count = 0
        
        # Identify beginning and end gaps for custom remarks
        beginning_gap = gaps[0] if gaps and gaps[0].start == 0 else None
        end_gap = gaps[-1] if gaps else None
        if end_gap and end_gap.duration < 2:
            end_gap = None
        
        for i, gap in enumerate(gaps[:MAX_SNARKS]):
            # Find context
            context = ""
            for seg in transcript["segments"]:
                if seg["end"] <= gap.start and seg["end"] > gap.start - 3:
                    context = seg["text"]
                    break
            
            # Generate custom remarks for beginning and end
            force_custom = (gap == beginning_gap or gap == end_gap)
            
            if force_custom:
                position = "beginning" if gap == beginning_gap else "end"
                print(f"  🎭 Generating custom {position} remark (gap: {gap.duration:.1f}s)...")
                
                if self.gemini_key:
                    snark = self._generate_custom_funny_snark(
                        context, gap.duration, position, transcript
                    )
                else:
                    snark = self._get_custom_fallback(gap.duration, position)
                
                snarks.append(Snark(
                    text=snark["text"],
                    time=gap.start + 0.2,
                    emotion=snark["emotion"],
                    context=context,
                    gap_duration=gap.duration
                ))
                print(f"  🎯 Custom: \"{snark['text']}\"")
                used_texts.add(snark["text"].lower())
            else:
                # Generate new friendly snark
                if self.gemini_key:
                    snark = self._generate_snark_with_ai(context, gap.duration, used_texts)
                else:
                    snark = self._get_fallback_snark(gap.duration, used_texts)
                
                if snark and snark["text"].lower() not in used_texts:
                    snarks.append(Snark(
                        text=snark["text"],
                        time=gap.start + 0.2,
                        emotion=snark["emotion"],
                        context=context,
                        gap_duration=gap.duration
                    ))
                    used_texts.add(snark["text"].lower())
                    print(f"  ✨ New: \"{snark['text']}\" ({snark['emotion']})")
        
        print(f"✅ Total snarks: {len(snarks)} ({reused_count} reused, {len(snarks)-reused_count} new)")
        return snarks
    
    def _generate_custom_funny_snark(
        self, context: str, max_duration: float, position: str, transcript: Dict
    ) -> Dict:
        """Generate custom hyper-funny snark for beginning or end."""
        max_words = min(int((max_duration - 0.3) * 2.2), 15)
        
        # Fallback if no AI
        if position == "beginning":
            if max_duration < 1.5:
                return {"text": "Nice!", "emotion": "cheerful"}
            elif max_duration < 2.5:
                return {"text": "Ooh, this looks fun.", "emotion": "playful"}
            else:
                return {"text": "Buckle up everyone.", "emotion": "amused"}
        else:
            if max_duration < 1.5:
                return {"text": "Done!", "emotion": "cheerful"}
            elif max_duration < 2.5:
                return {"text": "And scene.", "emotion": "playful"}
            else:
                return {"text": "And that's a wrap folks.", "emotion": "cheerful"}
    
    def _generate_snark_with_ai(
        self, context: str, max_duration: float, used_texts: Set[str]
    ) -> Optional[Dict]:
        """Generate snark using AI."""
        # Simplified for refactoring - returns fallback
        return self._get_fallback_snark(max_duration, used_texts)
    
    def _get_fallback_snark(
        self, max_duration: float, used_texts: Set[str], force_different: bool = False
    ) -> Dict:
        """Get a fallback snark that fits duration."""
        fallback_pool = [
            ("Nice!", "cheerful"),
            ("Plot twist!", "amused"),
            ("Taking notes.", "playful"),
            ("Love it!", "impressed"),
            ("Interesting.", "curious"),
            ("Cool!", "excited"),
            ("Neat!", "cheerful"),
            ("Wow!", "impressed"),
            ("Fascinating.", "sarcastic"),
            ("Hmm.", "thoughtful"),
            ("Oh really?", "skeptical"),
            ("Go on.", "encouraging"),
            ("Amazing!", "excited"),
            ("Sweet!", "cheerful")
        ]
        
        # Filter by duration and unused
        suitable = []
        for text, emotion in fallback_pool:
            if text.lower() in used_texts:
                continue
            word_count = len(text.split())
            if word_count <= max(1, int(max_duration * 2)):
                suitable.append({"text": text, "emotion": emotion})
        
        if suitable:
            return random.choice(suitable)
        
        # Last resort
        return {"text": "Mmhmm.", "emotion": "listening"}
    
    def _get_custom_fallback(self, max_duration: float, position: str) -> Dict:
        """Get custom fallback for beginning/end."""
        if position == "beginning":
            if max_duration < 1.5:
                return {"text": "Nice!", "emotion": "cheerful"}
            elif max_duration < 2.5:
                return {"text": "Let's do this.", "emotion": "playful"}
            else:
                return {"text": "Buckle up everyone.", "emotion": "amused"}
        else:
            if max_duration < 1.5:
                return {"text": "Done!", "emotion": "cheerful"}
            elif max_duration < 2.5:
                return {"text": "And scene.", "emotion": "playful"}
            else:
                return {"text": "That's all folks!", "emotion": "cheerful"}