# Auto-Snark Narrator for Videos 🎬🎙️

A production-ready CLI tool that automatically injects witty, sarcastic TTS commentary into videos at detected comedic beats. Think of it as having a snarky narrator who watches your content and drops perfectly-timed quips during pauses, transitions, and key moments.

## ✨ Features

- **Hybrid Beat Detection**: Finds comedic opportunities through:
  - Natural speech pauses and gaps
  - Discourse markers (but, actually, anyway, seriously, etc.)
  - Shot changes via OpenCV computer vision
  
- **Smart Snark Generation**: 
  - Three style banks: `wry` (balanced), `gentle` (wholesome), `spicy` (edgy)
  - Context-aware keyword extraction from nearby transcript
  - Profanity filtering for brand safety
  - Deterministic template selection via seed

- **Professional Audio Mixing**:
  - Audio ducking (not hard mute) preserves authenticity
  - Automatic normalization to prevent clipping
  - Seamless integration with original video audio

- **TTS Options**:
  - **Primary**: ElevenLabs (high-quality, multilingual)
  - **Fallback**: pyttsx3 (fully offline, no API needed)

- **Production Features**:
  - JSON report with timestamps, costs, and metrics
  - Configurable parameters for fine-tuning
  - H.264 MP4 export for web compatibility

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify FFmpeg is installed
ffmpeg -version
# If not installed:
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install -y ffmpeg
# Windows: winget install Gyan.FFmpeg
```

### Basic Usage

```bash
# With ElevenLabs (recommended)
export ELEVENLABS_API_KEY="your_key_here"
python snark_narrator.py \
  --video input.mp4 \
  --transcript transcript.json \
  --out output_snark.mp4

# With local TTS (no API key needed)
python snark_narrator.py \
  --video input.mp4 \
  --transcript transcript.json \
  --out output_snark.mp4 \
  --no-elevenlabs
```

## 📋 Transcript Format

The tool expects a JSON file with timestamped segments:

```json
{
  "segments": [
    {"start": 0.10, "end": 2.60, "text": "Welcome back everyone."},
    {"start": 2.80, "end": 7.90, "text": "Today we are testing a bold new idea. Actually, it's a bit experimental."},
    {"start": 9.20, "end": 13.10, "text": "But if it works, it could save hours of editing every week."},
    {"start": 15.50, "end": 19.00, "text": "Anyway, let's jump into the setup and look at the key steps."},
    {"start": 21.20, "end": 25.20, "text": "Okay, first we'll import the assets and configure the project."}
  ]
}
```

## 🎨 Snark Styles

### Wry (Default)
Balanced wit with professional edge:
- "Bold choice. Not judging. Okay, maybe a little."
- "Plot twist no one asked for."
- "Narrator: that did not go as planned."

### Gentle
Wholesome and supportive:
- "Tiny detour. We'll allow it."
- "Love the enthusiasm. Math? Later."
- "Wholesome chaos detected."

### Spicy
Edgy and chaotic:
- "Choices were made. Regrets loading."
- "This tutorial brought to you by chaos."
- "Speedrun to confusion—any percent."

## ⚙️ Advanced Configuration

```bash
python snark_narrator.py \
  --video input.mp4 \
  --transcript transcript.json \
  --out output.mp4 \
  --style spicy \              # wry|gentle|spicy
  --max-snarks 15 \             # Maximum insertions (default: 10)
  --min-gap 8 \                 # Min seconds between snarks (default: 12)
  --pause-threshold 0.8 \       # Min pause to consider (default: 1.0)
  --duck-db -8 \                # Audio ducking in dB (default: -12)
  --use-vision 1 \              # Enable shot detection (default: 1)
  --seed 42 \                   # For reproducible template selection
  --log-level DEBUG             # DEBUG|INFO|WARNING|ERROR
```

## 📊 Output Report

Each run generates a detailed JSON report (`output.mp4.report.json`):

```json
{
  "video": "/path/to/input.mp4",
  "out": "/path/to/output.mp4",
  "style": "wry",
  "inserts": [
    {
      "time_s": 2.9,
      "text": "Bold choice. Not judging. Okay, maybe a little. (testing, experimental)",
      "duration_ms": 3200,
      "reasons": ["pause:1.34s", "marker"]
    }
  ],
  "counts": {
    "candidates": 24,
    "selected": 10
  },
  "estimates": {
    "tts_total_chars": 743,
    "approx_cost_usd": 0.00022
  }
}
```

## 🎯 Tuning Guide

### Make it chattier
```bash
--max-snarks 20 --min-gap 8
```

### More precise timing
```bash
--pause-threshold 1.5  # Only target clear pauses
```

### Gentler presence
```bash
--style gentle --duck-db -8  # Less audio ducking
```

### Faster processing
```bash
--use-vision 0  # Skip shot detection
```

## 💰 Cost Estimation

- Average snark: ~70 characters
- 10 snarks/video: ~700 chars total
- ElevenLabs pricing: ~$30 per 1M characters
- **Cost per video: ~$0.021**

The tool calculates and reports exact costs in the JSON output.

## 🔧 Troubleshooting

### FFmpeg not found
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# Windows
winget install Gyan.FFmpeg
```

### No audio in output
- Check FFmpeg installation
- Verify input video has audio track
- Try increasing `--duck-db` value (e.g., -6)

### TTS fails with ElevenLabs
- Verify API key is set correctly
- Check API quota/credits
- Falls back to pyttsx3 automatically

### OpenCV errors
- Run with `--use-vision 0` to disable shot detection
- Everything else will still work

## 🚀 Performance Tips

1. **Caching**: TTS files are named with content hash for reuse
2. **Parallel beats**: All detection methods run concurrently
3. **Smart pruning**: Only top-scoring beats are processed
4. **Temp cleanup**: Automatic cleanup of temporary files

## 🛡️ Safety Features

- Profanity filtering (replaces with *beep*)
- Style control for brand alignment
- Configurable limits on insertion count
- Minimum gap enforcement between snarks

## 📈 Metrics to Track

- **Engagement**: Average snarks per minute
- **Timing**: Distribution of pause lengths targeted
- **Style performance**: A/B test different styles
- **Cost efficiency**: Characters per snark ratio

## 🧪 Testing

### Quick sanity check (30 seconds)
```bash
# Create a test transcript
cat > test_transcript.json << 'EOF'
{
  "segments": [
    {"start": 0.1, "end": 2.6, "text": "Welcome to this tutorial."},
    {"start": 4.8, "end": 7.9, "text": "Actually, let me show you something cool."},
    {"start": 10.2, "end": 13.1, "text": "But first, we need to set up the environment."},
    {"start": 15.5, "end": 19.0, "text": "Anyway, here's the main dashboard."}
  ]
}
EOF

# Run with local TTS (no API needed)
python snark_narrator.py \
  --video your_test_video.mp4 \
  --transcript test_transcript.json \
  --out test_output.mp4 \
  --no-elevenlabs \
  --max-snarks 3 \
  --log-level DEBUG
```

## 🎬 Example Commands

### Documentary style (sparse, thoughtful)
```bash
python snark_narrator.py \
  --video documentary.mp4 \
  --transcript doc_transcript.json \
  --out doc_narrated.mp4 \
  --style wry \
  --max-snarks 5 \
  --min-gap 20 \
  --pause-threshold 2.0
```

### Tutorial roast (frequent, spicy)
```bash
python snark_narrator.py \
  --video tutorial.mp4 \
  --transcript tut_transcript.json \
  --out tutorial_roasted.mp4 \
  --style spicy \
  --max-snarks 20 \
  --min-gap 8 \
  --pause-threshold 0.5
```

### Corporate video (gentle, safe)
```bash
python snark_narrator.py \
  --video corporate.mp4 \
  --transcript corp_transcript.json \
  --out corp_enhanced.mp4 \
  --style gentle \
  --max-snarks 8 \
  --min-gap 15 \
  --duck-db -10
```

## 📝 License

This tool is provided as-is for the ToonTune backend project. Ensure you have appropriate licenses for any TTS services used.

## 🤝 Contributing

To extend the snark templates, edit the `TEMPLATES` dictionary in `snark_narrator.py`:

```python
TEMPLATES = {
    "your_style": [
        "Your custom quip here.",
        "Another witty remark.",
        # Add more...
    ]
}
```

Then use with `--style your_style`.