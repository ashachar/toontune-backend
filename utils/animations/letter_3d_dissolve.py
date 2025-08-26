#!/usr/bin/env python3
"""
3D letter dissolve animation where each letter dissolves individually.

This file now imports from the refactored module structure.
All functionality has been preserved but organized into smaller modules:
- timing.py: Frame-accurate timing calculations
- renderer.py: 3D letter rendering
- sprite_manager.py: Letter sprite management and layout
- occlusion.py: Foreground occlusion handling
- frame_renderer.py: Frame-by-frame rendering logic
- dissolve.py: Main animation class

FRAME-ACCURATE FIXES:
- Per-letter schedule computed in integer frames (start/end/fade_end).
- Guaranteed minimum fade frames (prevents skipped fades at low FPS).
- 1-frame "safety hold" at dissolve start so alpha begins EXACTLY at stable_alpha.
- Optional overlap guard: previous letter's fade extends at least until next letter's start.

DEBUG:
- [JUMP_CUT] logs print the full schedule and per-frame transitions.
- [POS_HANDOFF] logs from motion remain supported.
- [TEXT_QUALITY] Robust TTF/OTF font discovery + optional --font path.
- [TEXT_QUALITY] Logs to verify font and supersampling.

Original authorship retained; this refactor targets the jump-cut described in the issue.
"""

# Import from the refactored module
from .letter_3d_dissolve import Letter3DDissolve, LetterTiming

# For backward compatibility, expose the original name
_LetterTiming = LetterTiming

# Make sure Letter3DDissolve is available at module level
__all__ = ['Letter3DDissolve', '_LetterTiming', 'LetterTiming']