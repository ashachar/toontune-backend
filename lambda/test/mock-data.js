/**
 * Mock Data Generators for Testing
 * Provides realistic test data for all Lambda functions
 */

const crypto = require('crypto');

/**
 * Generate mock script idea
 */
function generateMockIdea() {
  const ideas = [
    "How artificial intelligence is transforming healthcare",
    "The history of space exploration in 60 seconds",
    "5 ways to reduce your carbon footprint",
    "The science behind coffee brewing",
    "Understanding cryptocurrency for beginners",
    "The evolution of smartphones over 20 years",
    "How vaccines work to protect us",
    "The importance of sleep for mental health",
    "Ocean pollution and its impact on marine life",
    "The basics of quantum computing explained simply"
  ];
  
  return {
    idea: ideas[Math.floor(Math.random() * ideas.length)],
    targetLength: ['short', 'medium', 'long'][Math.floor(Math.random() * 3)],
    language: ['en', 'es', 'fr', 'de', 'ja'][Math.floor(Math.random() * 5)],
    tone: ['educational', 'casual', 'professional', 'humorous'][Math.floor(Math.random() * 4)]
  };
}

/**
 * Generate mock script
 */
function generateMockScript(length = 'medium') {
  const scripts = {
    short: `Welcome to our quick guide on renewable energy.
Solar and wind power are leading the charge in clean energy.
They're becoming cheaper and more efficient every year.
Together, we can build a sustainable future.`,
    
    medium: `Today, let's explore the fascinating world of artificial intelligence.
AI is no longer just science fiction - it's part of our daily lives.
From smartphone assistants to recommendation systems, AI is everywhere.
Machine learning allows computers to learn from data without explicit programming.
Deep learning, inspired by the human brain, powers advanced applications.
Self-driving cars, medical diagnosis, and language translation all use AI.
The future holds exciting possibilities as AI continues to evolve.
But with great power comes responsibility to use it ethically.`,
    
    long: `Welcome to our comprehensive guide on climate change and its global impact.
Climate change represents one of the most pressing challenges of our time.
Rising global temperatures are caused primarily by greenhouse gas emissions.
Carbon dioxide from burning fossil fuels is the main contributor.
The effects are already visible: melting ice caps, rising sea levels, and extreme weather.
Scientists predict these impacts will intensify without immediate action.
But there's hope - renewable energy technology is advancing rapidly.
Solar panels and wind turbines are becoming more efficient and affordable.
Electric vehicles are gaining market share, reducing transportation emissions.
Governments worldwide are implementing policies to reduce carbon footprints.
Individual actions matter too - from reducing energy consumption to sustainable choices.
Together, through innovation and determination, we can address this challenge.
The transition to a sustainable future requires global cooperation.
Every action counts in the fight against climate change.
Join us in building a cleaner, greener world for future generations.`
  };
  
  return {
    script: scripts[length] || scripts.medium,
    wordCount: scripts[length].split(' ').length,
    estimatedDuration: Math.floor(scripts[length].split(' ').length / 2.5) // 150 words per minute
  };
}

/**
 * Generate mock slides
 */
function generateMockSlides(count = 5) {
  const slides = [];
  const topics = [
    { text: "Introduction to the topic", keywords: ["introduction", "welcome", "overview"] },
    { text: "Key concepts and definitions", keywords: ["concepts", "definitions", "basics"] },
    { text: "Real-world applications", keywords: ["applications", "examples", "usage"] },
    { text: "Benefits and advantages", keywords: ["benefits", "advantages", "pros"] },
    { text: "Challenges and solutions", keywords: ["challenges", "solutions", "problems"] },
    { text: "Future developments", keywords: ["future", "trends", "innovations"] },
    { text: "Conclusion and summary", keywords: ["conclusion", "summary", "recap"] }
  ];
  
  for (let i = 0; i < count && i < topics.length; i++) {
    slides.push({
      slideNumber: i + 1,
      text: topics[i].text,
      imagePrompt: `Simple doodle illustration of ${topics[i].keywords[0]}`,
      keywords: topics[i].keywords,
      duration: 3 + Math.random() * 2 // 3-5 seconds per slide
    });
  }
  
  return slides;
}

/**
 * Generate mock image URLs
 */
function generateMockImageUrls(count = 5) {
  const urls = [];
  for (let i = 0; i < count; i++) {
    urls.push({
      slideNumber: i + 1,
      imageUrl: `https://toontune-assets.s3.amazonaws.com/slides/mock-${Date.now()}-${i}.png`,
      thumbnailUrl: `https://toontune-assets.s3.amazonaws.com/thumbnails/mock-${Date.now()}-${i}.png`
    });
  }
  return urls;
}

/**
 * Generate mock audio data
 */
function generateMockAudioData() {
  const voiceId = ['Matthew', 'Joanna', 'Amy', 'Brian'][Math.floor(Math.random() * 4)];
  const duration = 30 + Math.random() * 60; // 30-90 seconds
  
  return {
    audioUrl: `https://toontune-assets.s3.amazonaws.com/audio/voiceover-${Date.now()}.mp3`,
    duration: duration,
    format: 'mp3',
    voiceId: voiceId,
    language: 'en-US',
    sampleRate: 22050,
    bitRate: 128000
  };
}

/**
 * Generate mock sync data
 */
function generateMockSyncData(slideCount = 5) {
  const timeline = [];
  let currentTime = 0;
  
  for (let i = 0; i < slideCount; i++) {
    const duration = 3 + Math.random() * 2; // 3-5 seconds per slide
    timeline.push({
      slideNumber: i + 1,
      startTime: currentTime,
      endTime: currentTime + duration,
      duration: duration
    });
    currentTime += duration;
  }
  
  return {
    timeline,
    totalDuration: currentTime,
    fps: 30,
    keyframes: timeline.map(t => ({
      frame: Math.floor(t.startTime * 30),
      slideNumber: t.slideNumber
    }))
  };
}

/**
 * Generate mock language detection result
 */
function generateMockLanguageDetection() {
  const languages = [
    { code: 'en', name: 'English', confidence: 0.95 },
    { code: 'es', name: 'Spanish', confidence: 0.88 },
    { code: 'fr', name: 'French', confidence: 0.91 },
    { code: 'de', name: 'German', confidence: 0.87 },
    { code: 'ja', name: 'Japanese', confidence: 0.93 }
  ];
  
  const primary = languages[Math.floor(Math.random() * languages.length)];
  const alternatives = languages
    .filter(l => l.code !== primary.code)
    .slice(0, 2)
    .map(l => ({ ...l, confidence: l.confidence * 0.3 }));
  
  return {
    primaryLanguage: primary,
    alternatives,
    mixedContent: false
  };
}

/**
 * Generate mock style configuration
 */
function generateMockStyleConfig() {
  const styles = [
    {
      name: 'minimalist',
      imageModifiers: 'simple, clean lines, minimal colors, white background',
      colorPalette: ['#000000', '#FFFFFF', '#808080'],
      animationStyle: 'fade',
      transitionDuration: 0.5
    },
    {
      name: 'colorful',
      imageModifiers: 'vibrant colors, playful, cartoon style, cheerful',
      colorPalette: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
      animationStyle: 'slide',
      transitionDuration: 0.3
    },
    {
      name: 'professional',
      imageModifiers: 'corporate, clean, modern, business-like',
      colorPalette: ['#2C3E50', '#3498DB', '#ECF0F1', '#95A5A6'],
      animationStyle: 'dissolve',
      transitionDuration: 0.7
    }
  ];
  
  return styles[Math.floor(Math.random() * styles.length)];
}

/**
 * Generate mock API request
 */
function generateMockAPIRequest(functionName) {
  const requests = {
    generateScriptFromIdea: generateMockIdea(),
    parseScriptToSlides: { script: generateMockScript().script },
    generateSlideImages: { slides: generateMockSlides() },
    generateVoiceover: { 
      text: generateMockScript('short').script,
      voiceId: 'Matthew',
      language: 'en-US'
    },
    previewVoice: {
      voiceId: 'Joanna',
      language: 'en-US'
    },
    syncVoiceToSlides: {
      slides: generateMockSlides(),
      audioDuration: 30
    },
    detectLanguage: {
      text: generateMockScript('short').script
    },
    generateDoodleAsset: {
      prompt: 'lightbulb idea concept',
      category: 'icons'
    },
    applyStyleToProject: {
      projectId: `project-${Date.now()}`,
      style: 'minimalist'
    },
    batchProcessProject: {
      idea: generateMockIdea().idea,
      style: 'colorful',
      voiceId: 'Amy'
    }
  };
  
  return requests[functionName] || {};
}

/**
 * Generate mock Lambda event
 */
function generateMockLambdaEvent(functionName, authenticated = true) {
  const body = generateMockAPIRequest(functionName);
  body.function = functionName;
  
  const event = {
    body: JSON.stringify(body),
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': crypto.randomBytes(16).toString('hex')
    },
    httpMethod: 'POST',
    path: `/lambda/${functionName}`,
    requestContext: {
      requestId: crypto.randomBytes(16).toString('hex'),
      stage: 'dev',
      identity: {
        sourceIp: '127.0.0.1'
      }
    }
  };
  
  if (authenticated) {
    event.headers['X-Api-Key'] = 'dev-key-12345';
  }
  
  return event;
}

/**
 * Generate mock Lambda context
 */
function generateMockLambdaContext() {
  return {
    functionName: 'toontune-ai-service',
    functionVersion: '$LATEST',
    invokedFunctionArn: 'arn:aws:lambda:us-east-1:123456789:function:toontune-ai-service',
    memoryLimitInMB: '1024',
    awsRequestId: crypto.randomBytes(16).toString('hex'),
    logGroupName: '/aws/lambda/toontune-ai-service',
    logStreamName: `2024/01/01/[$LATEST]${crypto.randomBytes(16).toString('hex')}`,
    getRemainingTimeInMillis: () => 30000,
    callbackWaitsForEmptyEventLoop: true
  };
}

/**
 * Generate batch of test data
 */
function generateTestBatch(functionName, count = 10) {
  const batch = [];
  for (let i = 0; i < count; i++) {
    batch.push({
      event: generateMockLambdaEvent(functionName),
      context: generateMockLambdaContext(),
      expectedKeys: getExpectedResponseKeys(functionName)
    });
  }
  return batch;
}

/**
 * Get expected response keys for validation
 */
function getExpectedResponseKeys(functionName) {
  const expectedKeys = {
    generateScriptFromIdea: ['script', 'wordCount', 'estimatedDuration'],
    parseScriptToSlides: ['slides', 'totalSlides'],
    generateSlideImages: ['images', 'style'],
    generateVoiceover: ['audioUrl', 'duration', 'format'],
    previewVoice: ['previewUrl', 'duration'],
    syncVoiceToSlides: ['timeline', 'totalDuration'],
    detectLanguage: ['primaryLanguage', 'alternatives'],
    generateDoodleAsset: ['assetUrl', 'thumbnailUrl', 'metadata'],
    applyStyleToProject: ['style', 'appliedTo'],
    batchProcessProject: ['script', 'slides', 'images', 'audio', 'timeline', 'projectId']
  };
  
  return expectedKeys[functionName] || [];
}

/**
 * Generate error scenarios
 */
function generateErrorScenarios() {
  return [
    {
      name: 'Missing API Key',
      event: generateMockLambdaEvent('generateScriptFromIdea', false),
      expectedError: 'API key is required'
    },
    {
      name: 'Invalid Function',
      event: {
        body: JSON.stringify({ function: 'invalidFunction' }),
        headers: { 'X-Api-Key': 'dev-key-12345' }
      },
      expectedError: 'Unknown function'
    },
    {
      name: 'Invalid Input',
      event: {
        body: JSON.stringify({ function: 'generateScriptFromIdea' }), // Missing required 'idea' field
        headers: { 'X-Api-Key': 'dev-key-12345' }
      },
      expectedError: 'Validation failed'
    },
    {
      name: 'Malformed JSON',
      event: {
        body: '{ invalid json }',
        headers: { 'X-Api-Key': 'dev-key-12345' }
      },
      expectedError: 'JSON'
    }
  ];
}

module.exports = {
  generateMockIdea,
  generateMockScript,
  generateMockSlides,
  generateMockImageUrls,
  generateMockAudioData,
  generateMockSyncData,
  generateMockLanguageDetection,
  generateMockStyleConfig,
  generateMockAPIRequest,
  generateMockLambdaEvent,
  generateMockLambdaContext,
  generateTestBatch,
  getExpectedResponseKeys,
  generateErrorScenarios
};