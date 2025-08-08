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

## Asset Management

The backend includes a powerful asset animation system that automatically generates drawing animations for all assets with transparency support.

### Features

- **Automatic APNG Generation**: Converts static assets (PNG, SVG, JPG) into animated drawings
- **Transparency Support**: All animations preserve alpha channels for clean integration
- **Smart Processing**: Only processes new assets by default (skips existing ones)
- **Clustering Algorithm**: Uses Mean Shift clustering for natural drawing paths
- **Edge-Following Animation**: Creates realistic hand-drawing movements
- **Supabase Integration**: Automatically uploads to cloud storage

### Usage

```bash
# Process only new assets (default)
./manage-assets

# Check status of assets
./manage-assets --status

# Re-process all assets
./manage-assets --all

# Show detailed progress
./manage-assets --verbose

# Process assets from custom directory
./manage-assets --assets-dir /path/to/assets
```

### How It Works

1. **Asset Discovery**: Scans the assets directory for PNG, SVG, and JPG files
2. **Smart Detection**: Checks Supabase bucket for existing animations
3. **Image Segmentation**: Uses Mean Shift clustering to identify drawable regions
4. **Path Generation**: Creates edge-following paths for natural drawing motion
5. **Animation Creation**: Generates APNG with transparency and hand/pencil movement
6. **Cloud Upload**: Automatically uploads to Supabase storage bucket

### Output Format

- **Format**: APNG (Animated PNG with transparency)
- **Duration**: 3-second drawing animation + 1-second final display
- **Frame Rate**: 30 FPS
- **Transparency**: Full alpha channel support
- **File Size**: Optimized (typically 20-50 KB for simple assets)

## API Endpoints

- `POST /api/ai/generate-doodle` - Generate AI doodles from text description
- `POST /api/ai/process-video` - Process and generate animated videos
- `POST /api/ai/segment-image` - Segment images for animation

## License

MIT