# ToonTune.ai Lambda API Documentation

## Overview
ToonTune.ai provides a comprehensive set of Lambda functions for AI-powered video creation from text. This API enables script generation, slide creation, image generation, voiceover synthesis, and complete project orchestration.

## Base URL
```
https://api.toontune.ai/lambda
```

## Authentication
All API requests require authentication using an API key.

### Headers
```http
X-Api-Key: your-api-key-here
Content-Type: application/json
```

### Optional Request Signing (HMAC)
For enhanced security, you can sign requests with HMAC-SHA256:

```http
X-Signature: hmac-signature
X-Timestamp: unix-timestamp
```

## Rate Limits
- **Development**: 1,000 requests/hour
- **Production**: 10,000 requests/hour  
- **Premium**: 100,000 requests/hour

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 2024-01-01T12:00:00Z
```

## Common Response Format

### Success Response
```json
{
  "success": true,
  "function": "functionName",
  "result": {
    // Function-specific results
  },
  "executionTime": 1234,
  "requestId": "abc-123-def"
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message",
  "errorType": "ValidationError",
  "requestId": "abc-123-def",
  "retryable": false
}
```

## Functions

### 1. generateScriptFromIdea
Generate a complete script from a brief idea.

**Request:**
```json
{
  "function": "generateScriptFromIdea",
  "idea": "The importance of renewable energy",
  "targetLength": "medium",
  "language": "en",
  "tone": "educational"
}
```

**Parameters:**
- `idea` (string, required): Brief description of the topic
- `targetLength` (string): "short" | "medium" | "long" (default: "medium")
- `language` (string): ISO language code (default: "en")
- `tone` (string): "educational" | "casual" | "professional" | "humorous" (default: "educational")

**Response:**
```json
{
  "script": "Today we explore renewable energy...",
  "wordCount": 250,
  "estimatedDuration": 90,
  "language": "en",
  "sections": [...]
}
```

### 2. parseScriptToSlides
Split a script into logical slides with timing and image prompts.

**Request:**
```json
{
  "function": "parseScriptToSlides",
  "script": "Your complete script text here...",
  "targetSlideCount": 10,
  "slideDuration": 5
}
```

**Parameters:**
- `script` (string, required): Complete script text
- `targetSlideCount` (number): Desired number of slides (default: auto)
- `slideDuration` (number): Target duration per slide in seconds (default: 5)

**Response:**
```json
{
  "slides": [
    {
      "slideNumber": 1,
      "text": "Introduction to renewable energy",
      "imagePrompt": "Solar panels and wind turbines illustration",
      "keywords": ["renewable", "energy", "sustainable"],
      "duration": 5
    }
  ],
  "totalSlides": 10,
  "totalDuration": 50
}
```

### 3. generateSlideImages
Generate doodle-style images for slides.

**Request:**
```json
{
  "function": "generateSlideImages",
  "slides": [...],
  "style": "minimalist",
  "batchSize": 5
}
```

**Parameters:**
- `slides` (array, required): Array of slide objects with image prompts
- `style` (string): "minimalist" | "colorful" | "sketch" | "professional" (default: "minimalist")
- `batchSize` (number): Number of images to generate in parallel (default: 5)

**Response:**
```json
{
  "images": [
    {
      "slideNumber": 1,
      "imageUrl": "https://cdn.toontune.ai/images/slide-1.png",
      "thumbnailUrl": "https://cdn.toontune.ai/thumbs/slide-1.png",
      "metadata": {...}
    }
  ],
  "style": "minimalist",
  "processingTime": 8500
}
```

### 4. generateVoiceover
Generate voiceover audio from text.

**Request:**
```json
{
  "function": "generateVoiceover",
  "text": "Your narration text here",
  "voiceId": "Matthew",
  "language": "en-US",
  "speed": 1.0,
  "pitch": 0
}
```

**Parameters:**
- `text` (string, required): Text to convert to speech
- `voiceId` (string): Voice identifier (default: "Matthew")
- `language` (string): Language code (default: "en-US")
- `speed` (number): Speech rate 0.5-2.0 (default: 1.0)
- `pitch` (number): Pitch adjustment -20 to +20 (default: 0)
- `service` (string): "polly" | "elevenlabs" (default: "polly")

**Response:**
```json
{
  "audioUrl": "https://cdn.toontune.ai/audio/voiceover.mp3",
  "duration": 45.5,
  "format": "mp3",
  "voiceId": "Matthew",
  "language": "en-US"
}
```

### 5. previewVoice
Generate a short voice sample for preview.

**Request:**
```json
{
  "function": "previewVoice",
  "voiceId": "Joanna",
  "language": "en-US",
  "text": "Custom preview text"
}
```

**Parameters:**
- `voiceId` (string, required): Voice identifier
- `language` (string): Language code (default: "en-US")
- `text` (string): Custom preview text (optional)

**Response:**
```json
{
  "previewUrl": "https://cdn.toontune.ai/previews/voice-sample.mp3",
  "duration": 3.5,
  "expiresAt": "2024-01-01T12:00:00Z"
}
```

### 6. syncVoiceToSlides
Calculate timing synchronization between voice and slides.

**Request:**
```json
{
  "function": "syncVoiceToSlides",
  "slides": [...],
  "audioDuration": 60,
  "transitionDuration": 0.5
}
```

**Parameters:**
- `slides` (array, required): Array of slide objects
- `audioDuration` (number, required): Total audio duration in seconds
- `transitionDuration` (number): Transition time between slides (default: 0.5)

**Response:**
```json
{
  "timeline": [
    {
      "slideNumber": 1,
      "startTime": 0,
      "endTime": 6,
      "duration": 6
    }
  ],
  "totalDuration": 60,
  "fps": 30,
  "keyframes": [...]
}
```

### 7. detectLanguage
Detect the language of provided text.

**Request:**
```json
{
  "function": "detectLanguage",
  "text": "Bonjour le monde"
}
```

**Parameters:**
- `text` (string, required): Text to analyze

**Response:**
```json
{
  "primaryLanguage": {
    "code": "fr",
    "name": "French",
    "confidence": 0.98
  },
  "alternatives": [
    {
      "code": "en",
      "name": "English",
      "confidence": 0.15
    }
  ],
  "mixedContent": false
}
```

### 8. generateDoodleAsset
Generate individual doodle-style assets.

**Request:**
```json
{
  "function": "generateDoodleAsset",
  "prompt": "lightbulb idea concept",
  "category": "icons",
  "variations": 3
}
```

**Parameters:**
- `prompt` (string, required): Description of the asset
- `category` (string): "icons" | "objects" | "characters" | "backgrounds" (default: "objects")
- `variations` (number): Number of variations to generate (default: 1)

**Response:**
```json
{
  "assetUrl": "https://cdn.toontune.ai/assets/doodle.png",
  "thumbnailUrl": "https://cdn.toontune.ai/thumbs/doodle.png",
  "metadata": {
    "category": "icons",
    "tags": ["idea", "lightbulb", "concept"]
  },
  "variations": [...]
}
```

### 9. applyStyleToProject
Apply consistent styling to a project.

**Request:**
```json
{
  "function": "applyStyleToProject",
  "projectId": "project-123",
  "style": "minimalist",
  "customizations": {
    "primaryColor": "#2C3E50"
  }
}
```

**Parameters:**
- `projectId` (string, required): Project identifier
- `style` (string, required): Style preset name
- `customizations` (object): Custom style overrides

**Response:**
```json
{
  "style": {
    "name": "minimalist",
    "imageModifiers": "simple, clean lines, minimal colors",
    "colorPalette": ["#000000", "#FFFFFF", "#808080"],
    "animationStyle": "fade",
    "transitionDuration": 0.5
  },
  "appliedTo": "project-123",
  "cssVariables": {...}
}
```

### 10. batchProcessProject
Orchestrate complete project processing.

**Request:**
```json
{
  "function": "batchProcessProject",
  "idea": "Climate change solutions",
  "style": "colorful",
  "voiceId": "Amy",
  "targetLength": "medium",
  "language": "en"
}
```

**Parameters:**
- `idea` (string, required): Project topic/idea
- `style` (string): Visual style preset (default: "minimalist")
- `voiceId` (string): Voice for narration (default: "Matthew")
- `targetLength` (string): Project length (default: "medium")
- `language` (string): Language code (default: "en")

**Response:**
```json
{
  "projectId": "proj-abc-123",
  "script": {...},
  "slides": [...],
  "images": [...],
  "audio": {...},
  "timeline": {...},
  "status": "completed",
  "processingTime": 25000
}
```

## Error Codes

| Code | Description | Retryable |
|------|-------------|-----------|
| 400 | Bad Request - Invalid input | No |
| 401 | Unauthorized - Invalid API key | No |
| 403 | Forbidden - Access denied | No |
| 429 | Too Many Requests - Rate limit exceeded | Yes |
| 500 | Internal Server Error | Yes |
| 502 | Bad Gateway - External service error | Yes |
| 503 | Service Unavailable | Yes |
| 504 | Gateway Timeout | Yes |

## SDKs and Examples

### Node.js Example
```javascript
const axios = require('axios');

const apiClient = axios.create({
  baseURL: 'https://api.toontune.ai/lambda',
  headers: {
    'X-Api-Key': 'your-api-key',
    'Content-Type': 'application/json'
  }
});

async function generateScript(idea) {
  try {
    const response = await apiClient.post('/', {
      function: 'generateScriptFromIdea',
      idea: idea,
      targetLength: 'medium',
      language: 'en',
      tone: 'educational'
    });
    
    return response.data.result;
  } catch (error) {
    console.error('Error:', error.response.data);
    throw error;
  }
}
```

### Python Example
```python
import requests

API_KEY = 'your-api-key'
BASE_URL = 'https://api.toontune.ai/lambda'

headers = {
    'X-Api-Key': API_KEY,
    'Content-Type': 'application/json'
}

def generate_script(idea):
    payload = {
        'function': 'generateScriptFromIdea',
        'idea': idea,
        'targetLength': 'medium',
        'language': 'en',
        'tone': 'educational'
    }
    
    response = requests.post(BASE_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    return response.json()['result']
```

### cURL Example
```bash
curl -X POST https://api.toontune.ai/lambda \
  -H "X-Api-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "function": "generateScriptFromIdea",
    "idea": "The importance of renewable energy",
    "targetLength": "medium",
    "language": "en",
    "tone": "educational"
  }'
```

## Webhooks
For long-running operations, you can provide a webhook URL to receive completion notifications:

```json
{
  "function": "batchProcessProject",
  "idea": "...",
  "webhookUrl": "https://your-server.com/webhook",
  "webhookSecret": "your-webhook-secret"
}
```

## Support
- Email: support@toontune.ai
- Documentation: https://docs.toontune.ai
- Status Page: https://status.toontune.ai

## Changelog

### v1.0.0 (2024-01-01)
- Initial release with 10 core functions
- Multi-language support (30+ languages)
- Authentication and rate limiting
- Batch processing capabilities