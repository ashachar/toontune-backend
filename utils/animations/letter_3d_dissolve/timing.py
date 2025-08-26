"""
Timing and scheduling for letter dissolve animation.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LetterTiming:
    """Frame-accurate per-letter schedule."""
    start: int           # first frame letter is DEFINITELY drawn at stable_alpha
    hold_end: int        # last frame of the "safety hold" at stable_alpha (>= start)
    end: int             # last frame of the dissolve window (after hold)
    fade_end: int        # last frame of the fade-out tail
    order_index: int     # 0..N-1 sequential in dissolve_order


class TimingCalculator:
    """Handles frame-accurate timing calculations for letter animations."""
    
    def __init__(self, fps: int, total_frames: int, debug: bool = False):
        self.fps = fps
        self.total_frames = total_frames
        self.debug = debug
        self._timeline: Dict[int, LetterTiming] = {}
        self._entered_dissolve_logged: Dict[int, bool] = {}
        self._hold_logged: Dict[int, bool] = {}
    
    def build_frame_timeline(
        self,
        dissolve_order: List[int],
        stable_duration: float,
        dissolve_stagger: float,
        dissolve_duration: float,
        post_fade_seconds: float,
        pre_dissolve_hold_frames: int,
        ensure_no_gap: bool
    ) -> Dict[int, LetterTiming]:
        """Compute per-letter schedule in integer frames and log it."""
        min_fade_frames = max(2, int(round(post_fade_seconds * self.fps)))
        dissolve_frames = max(1, int(round(dissolve_duration * self.fps)))
        self._timeline.clear()
        self._entered_dissolve_logged.clear()
        self._hold_logged.clear()

        # 1) initial pass
        for order_idx, letter_idx in enumerate(dissolve_order):
            start_seconds = stable_duration + order_idx * dissolve_stagger
            start_frame = int(round(start_seconds * self.fps))
            hold_end = start_frame + max(0, pre_dissolve_hold_frames - 1)
            end_frame = hold_end + dissolve_frames  # dissolve begins AFTER hold
            fade_end = end_frame + min_fade_frames

            # clamp into clip range
            start_frame = max(0, min(self.total_frames - 1, start_frame))
            hold_end = max(start_frame, min(self.total_frames - 1, hold_end))
            end_frame = max(hold_end, min(self.total_frames - 1, end_frame))
            fade_end = max(end_frame, min(self.total_frames - 1, fade_end))

            self._timeline[letter_idx] = LetterTiming(
                start=start_frame,
                hold_end=hold_end,
                end=end_frame,
                fade_end=fade_end,
                order_index=order_idx
            )

        # 2) Optional pass to prevent gaps: ensure prev.fade_end >= next.start
        if ensure_no_gap and len(dissolve_order) > 1:
            for i in range(len(dissolve_order) - 1):
                a = self._timeline[dissolve_order[i]]
                b = self._timeline[dissolve_order[i + 1]]
                if a.fade_end < b.start:
                    # extend a.fade_end up to b.start
                    new_fade_end = b.start
                    self._timeline[dissolve_order[i]] = LetterTiming(
                        start=a.start, hold_end=a.hold_end, end=a.end,
                        fade_end=new_fade_end, order_index=a.order_index
                    )
                    if self.debug:
                        print(
                            f"[JUMP_CUT] Extended fade: letter#{i} fade_end {a.fade_end} -> {new_fade_end} "
                            f"to meet next.start {b.start}"
                        )

        return self._timeline
    
    def get_timeline(self) -> Dict[int, LetterTiming]:
        """Get the current timeline."""
        return self._timeline
    
    def get_entered_dissolve_logged(self) -> Dict[int, bool]:
        """Get the dissolve logging state."""
        return self._entered_dissolve_logged
    
    def get_hold_logged(self) -> Dict[int, bool]:
        """Get the hold logging state."""
        return self._hold_logged
    
    def log_schedule(self, dissolve_order: List[int], letter_sprites: List) -> None:
        """Log the current schedule for debugging."""
        if not self.debug:
            return
            
        lines = []
        for i, idx in enumerate(dissolve_order):
            t = self._timeline[idx]
            ch = letter_sprites[idx].char if 0 <= idx < len(letter_sprites) else '?'
            lines.append(
                f"[JUMP_CUT] schedule[{i}] '{ch}' idx={idx}: start={t.start}, hold_end={t.hold_end}, "
                f"end={t.end}, fade_end={t.fade_end}"
            )
        print("\n".join(lines))