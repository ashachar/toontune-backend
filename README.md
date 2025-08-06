# Doodle Backend

This repository contains the algorithmic and AI-based backend services for the DoodleForge AI application.

## Overview

This backend handles:
- AI-powered doodle generation
- Video generation and processing algorithms
- Image segmentation and animation
- Asset processing and optimization

## Structure

```
doodle-backend/
├── ai-services/         # AI integration services
│   ├── doodle-generator.js
│   └── routes/
├── video-processing/    # Video generation algorithms
│   ├── generate_drawing_video.py
│   └── extract_screenshots.py
├── utils/              # Shared utilities
├── config/             # Configuration files
└── tests/              # Test files
```

## Technologies

- **AI Services**: OpenAI, Google Gemini, Claude API integrations
- **Video Processing**: OpenCV, NumPy, scikit-learn
- **Image Processing**: Computer vision algorithms for segmentation and animation
- **Node.js**: Express server for AI endpoints

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Development

```bash
# Start the AI service server
npm run dev

# Run video generation script
python video-processing/generate_drawing_video.py
```

## API Endpoints

- `POST /api/ai/generate-doodle` - Generate AI doodles from text description
- `POST /api/ai/process-video` - Process and generate animated videos
- `POST /api/ai/segment-image` - Segment images for animation

## License

MIT