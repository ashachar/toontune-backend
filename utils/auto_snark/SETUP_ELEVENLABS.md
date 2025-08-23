# 🔐 Setting Up ElevenLabs API Key for Auto-Snark Narrator

## Quick Setup

### 1. Get Your API Key
1. Go to [https://elevenlabs.io](https://elevenlabs.io)
2. Sign up or log in
3. Click on your profile icon → "API Keys"
4. Copy your API key

### 2. Set the Environment Variable

#### Option A: Temporary (current session only)
```bash
export ELEVENLABS_API_KEY="your-actual-key-here"
```

#### Option B: Permanent (add to shell profile)
```bash
# For zsh (default on macOS)
echo 'export ELEVENLABS_API_KEY="your-actual-key-here"' >> ~/.zshrc
source ~/.zshrc

# For bash
echo 'export ELEVENLABS_API_KEY="your-actual-key-here"' >> ~/.bash_profile
source ~/.bash_profile
```

### 3. Verify It's Set
```bash
# Check if set
echo $ELEVENLABS_API_KEY

# Test with Python
python -c "import os; print('Key set:', bool(os.environ.get('ELEVENLABS_API_KEY')))"
```

## Running the Demo

### With Environment Variable Set
```bash
# The key will be automatically detected
python snark_narrator.py \
  --video /path/to/scene_001.mp4 \
  --transcript do_re_mi_transcript.json \
  --out output_with_snarks.mp4 \
  --style spicy
```

### Interactive Demo (prompts for key)
```bash
python demo_with_api_key.py
```

### Test Simple Audio Generation
```bash
python test_elevenlabs_simple.py
```

## Troubleshooting

### "API key not found" Error
```bash
# Make sure it's exported in current shell
export ELEVENLABS_API_KEY="your-key"

# Or add to Python script directly (not recommended for production)
import os
os.environ['ELEVENLABS_API_KEY'] = "your-key"
```

### "Unauthorized" Error (401)
- Check your API key is correct (no extra spaces)
- Verify you have credits in your ElevenLabs account
- Make sure the key is active (not revoked)

### Network/Connection Issues
- Check internet connection
- Try with a simpler text first
- Use the turbo model for faster generation

## API Key Best Practices

### DO ✅
- Store in environment variables
- Use `.env` files for local development
- Add `.env` to `.gitignore`

### DON'T ❌
- Hardcode in scripts
- Commit to version control
- Share publicly

## Example .env File
Create a `.env` file in the project root:
```
ELEVENLABS_API_KEY=your-actual-key-here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

Then load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Cost Estimates
- ~$0.30 per 1000 characters
- Average snark: 40-50 characters
- Full Do-Re-Mi demo (6 snarks): ~$0.075

## Ready to Generate Cynical Narration!

Once your API key is set up, you can generate cinema-quality cynical narration with:
- Emotion tags for expression
- Multiple voice personalities
- Pauses and emphasis controls
- Professional audio quality

Run the full demo:
```bash
python snark_narrator.py \
  --video scene_001.mp4 \
  --transcript do_re_mi_transcript.json \
  --out do_re_mi_cynical.mp4
```