/**
 * Generate Voiceover Function
 * Creates AI voice narration for text using multiple TTS services
 */

const { polly, awsHelpers } = require('../utils/aws-clients');
const { getElevenLabsClient } = require('../utils/ai-clients');
const { logger } = require('../utils/logger');
const crypto = require('crypto');

// Voice configurations
const VOICE_MAPPINGS = {
  // Map friendly names to service-specific IDs
  polly: {
    delilah: 'Joanna',
    zachary: 'Matthew',
    ruby: 'Amy',
    douglas: 'Brian',
    sophie: 'Emma',
    bella: 'Ivy'
  },
  elevenlabs: {
    delilah: 'EXAVITQu4vr4xnSDxMaL',
    zachary: 'pNInz6obpgDQGcFmaJgB',
    ruby: 'AZnzlk1XvdvUeBnXmlld',
    douglas: 'TxGEqnHWrfWFTfGW9XjX',
    sophie: 'MF3mGyEYCl7XYWbV9V6O',
    bella: 'XB0fDUnXU5powFXDhCwa'
  }
};

// Language to voice engine mapping
const LANGUAGE_SUPPORT = {
  en: ['polly', 'elevenlabs'],
  es: ['polly', 'elevenlabs'],
  fr: ['polly', 'elevenlabs'],
  de: ['polly', 'elevenlabs'],
  it: ['polly'],
  pt: ['polly'],
  ru: ['polly'],
  zh: ['polly'],
  ja: ['polly'],
  ko: ['polly'],
  ar: ['polly'],
  hi: ['polly'],
  he: ['polly']
};

async function generateVoiceover(input, context) {
  const {
    text,
    voiceId,
    language = 'en',
    speed = 1.0,
    pitch = 1.0
  } = input;

  logger.info('Generating voiceover', {
    textLength: text.length,
    voiceId,
    language,
    speed,
    pitch
  });

  try {
    // Determine which service to use based on language support
    const supportedServices = LANGUAGE_SUPPORT[language] || ['polly'];
    const preferredService = process.env.ELEVENLABS_API_KEY && supportedServices.includes('elevenlabs')
      ? 'elevenlabs'
      : 'polly';

    let audioBuffer;
    let duration;

    if (preferredService === 'elevenlabs' && process.env.ELEVENLABS_API_KEY) {
      const result = await generateWithElevenLabs(text, voiceId, language, speed, pitch);
      audioBuffer = result.buffer;
      duration = result.duration;
    } else {
      const result = await generateWithPolly(text, voiceId, language, speed, pitch);
      audioBuffer = result.buffer;
      duration = result.duration;
    }

    // Upload to S3
    const s3Url = await uploadAudioToS3(audioBuffer, voiceId, language);

    logger.info('Voiceover generated successfully', {
      service: preferredService,
      duration,
      audioSize: audioBuffer.length
    });

    return {
      audioUrl: s3Url,
      duration,
      format: 'mp3',
      sampleRate: 44100,
      bitrate: 128,
      service: preferredService,
      metadata: {
        voiceId,
        language,
        speed,
        pitch,
        textLength: text.length
      }
    };

  } catch (error) {
    logger.error('Failed to generate voiceover', {
      error: error.message,
      voiceId,
      language
    });
    throw error;
  }
}

async function generateWithPolly(text, voiceId, language, speed, pitch) {
  const pollyVoiceId = VOICE_MAPPINGS.polly[voiceId] || 'Joanna';
  
  // Build SSML for advanced control
  const ssml = buildSSML(text, speed, pitch);
  
  const params = {
    OutputFormat: 'mp3',
    Text: ssml,
    TextType: 'ssml',
    VoiceId: pollyVoiceId,
    SampleRate: '24000',
    Engine: 'neural', // Use neural engine for better quality
    LanguageCode: getPollyLanguageCode(language)
  };

  try {
    // Synthesize speech
    const result = await polly.synthesizeSpeech(params).promise();
    
    // Get duration from metadata
    const duration = await getAudioDuration(result.AudioStream);
    
    return {
      buffer: result.AudioStream,
      duration
    };
  } catch (error) {
    // Fallback to standard engine if neural fails
    if (error.code === 'InvalidParameterValueException') {
      params.Engine = 'standard';
      delete params.LanguageCode;
      
      const result = await polly.synthesizeSpeech(params).promise();
      const duration = await getAudioDuration(result.AudioStream);
      
      return {
        buffer: result.AudioStream,
        duration
      };
    }
    throw error;
  }
}

async function generateWithElevenLabs(text, voiceId, language, speed, pitch) {
  const elevenLabs = getElevenLabsClient();
  
  if (!elevenLabs) {
    throw new Error('ElevenLabs client not configured');
  }

  const elevenLabsVoiceId = VOICE_MAPPINGS.elevenlabs[voiceId] || VOICE_MAPPINGS.elevenlabs.delilah;
  
  const audioBuffer = await elevenLabs.textToSpeech(text, elevenLabsVoiceId, {
    stability: 0.5,
    similarityBoost: 0.5,
    modelId: language === 'en' ? 'eleven_monolingual_v1' : 'eleven_multilingual_v2'
  });

  const duration = await getAudioDuration(audioBuffer);

  return {
    buffer: audioBuffer,
    duration
  };
}

function buildSSML(text, speed, pitch) {
  const speedPercent = Math.round(speed * 100);
  const pitchPercent = pitch > 1 
    ? `+${Math.round((pitch - 1) * 100)}%`
    : `-${Math.round((1 - pitch) * 100)}%`;

  return `<speak>
    <prosody rate="${speedPercent}%" pitch="${pitchPercent}">
      ${escapeSSML(text)}
    </prosody>
  </speak>`;
}

function escapeSSML(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function getPollyLanguageCode(language) {
  const languageCodes = {
    en: 'en-US',
    es: 'es-ES',
    fr: 'fr-FR',
    de: 'de-DE',
    it: 'it-IT',
    pt: 'pt-BR',
    ru: 'ru-RU',
    zh: 'zh-CN',
    ja: 'ja-JP',
    ko: 'ko-KR',
    ar: 'ar-SA',
    hi: 'hi-IN',
    he: 'he-IL'
  };
  return languageCodes[language] || 'en-US';
}

async function getAudioDuration(audioBuffer) {
  // Estimate duration based on file size and bitrate
  // MP3 at 128kbps: 1 second = 16KB
  const sizeInKB = audioBuffer.length / 1024;
  const estimatedSeconds = sizeInKB / 16;
  
  return Math.round(estimatedSeconds * 10) / 10; // Round to 1 decimal
}

async function uploadAudioToS3(audioBuffer, voiceId, language) {
  const timestamp = Date.now();
  const hash = crypto.createHash('md5')
    .update(`${voiceId}-${language}-${timestamp}`)
    .digest('hex')
    .substring(0, 8);
  
  const key = `voiceovers/${language}/${voiceId}/${timestamp}-${hash}.mp3`;
  
  const s3Url = await awsHelpers.uploadToS3(
    process.env.S3_BUCKET_NAME,
    key,
    audioBuffer,
    'audio/mpeg',
    {
      voiceId,
      language,
      generatedAt: new Date().toISOString()
    }
  );

  // Return CDN URL if configured
  if (process.env.CDN_DOMAIN) {
    return `https://${process.env.CDN_DOMAIN}/${key}`;
  }

  return s3Url;
}

module.exports = {
  generateVoiceover
};