"""
Sound Effects Module for adding sound effects to videos.
"""

from .freesound_api import FreesoundAPI
from .sound_effects_manager import SoundEffectsManager, SoundEffect
from .video_sound_overlay import VideoSoundOverlay, SoundEffectEvent

__all__ = [
    'FreesoundAPI',
    'SoundEffectsManager',
    'SoundEffect',
    'VideoSoundOverlay',
    'SoundEffectEvent'
]