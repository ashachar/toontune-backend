"""
3D letter burn/smoke animation module.

This module provides letter-by-letter burn animation with smoke/gas particle effects.
Letters appear to burn from edges inward, turning into smoke that rises and disperses.
"""

from .burn import Letter3DBurn
from .timing import BurnTiming
from .particles import SmokeParticles

__all__ = ['Letter3DBurn', 'BurnTiming', 'SmokeParticles']