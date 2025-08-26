"""
Video Editing Tricks Module

Professional video effects for the ToonTune backend.
"""

from .color_effects import apply_color_splash
from .text_effects import apply_text_behind_subject, apply_motion_tracking_text
from .motion_effects import (
    apply_floating_effect, 
    apply_smooth_zoom, 
    apply_3d_photo_effect,
    apply_dolly_zoom,
    apply_rack_focus,
    apply_handheld_shake,
    apply_speed_ramp,
    apply_bloom_effect,
    apply_ken_burns,
    apply_light_sweep
)
from .layout_effects import apply_highlight_focus, add_progress_bar, apply_video_in_text

__all__ = [
    'apply_color_splash',
    'apply_text_behind_subject',
    'apply_motion_tracking_text',
    'apply_floating_effect',
    'apply_smooth_zoom',
    'apply_3d_photo_effect',
    'apply_dolly_zoom',
    'apply_rack_focus',
    'apply_handheld_shake',
    'apply_speed_ramp',
    'apply_bloom_effect',
    'apply_ken_burns',
    'apply_light_sweep',
    'apply_highlight_focus',
    'add_progress_bar',
    'apply_video_in_text'
]