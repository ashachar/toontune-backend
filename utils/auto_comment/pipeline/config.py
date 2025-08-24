"""Configuration for the comment pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

# Paths
UPLOAD_BASE = Path("/Users/amirshachar/Desktop/Amir/Projects/Personal/toontune/backend/uploads")
SNARK_STORAGE = UPLOAD_BASE / "assets" / "sounds" / "snark_remarks"

# Pipeline Settings
MIN_GAP_DURATION = 0.6  # Minimum gap duration for a comment
MAX_SNARKS = 10  # Maximum number of comments per video
TARGET_DBFS = -18  # Target loudness for audio normalization

# Speed Adjustment Settings
DEFAULT_FPS = 30
DEFAULT_VIDEO_QUALITY = "libx264"
DEFAULT_AUDIO_QUALITY = "aac"

# ElevenLabs Settings
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
ELEVENLABS_STABILITY = 0.5
ELEVENLABS_SIMILARITY_BOOST = 0.75