/**
 * Preview Voice Function
 * Generates short voice samples for preview
 */

const { generateVoiceover } = require('./generateVoiceover');
const { awsHelpers } = require('../utils/aws-clients');
const { logger } = require('../utils/logger');

// Sample texts for different languages
const SAMPLE_TEXTS = {
  en: "Welcome to ToonTune AI. This is how your voice will sound in your video. Let's create something amazing together!",
  es: "Bienvenido a ToonTune AI. Así es como sonará tu voz en tu video. ¡Creemos algo increíble juntos!",
  fr: "Bienvenue sur ToonTune AI. Voici comment votre voix sonnera dans votre vidéo. Créons quelque chose d'incroyable ensemble!",
  de: "Willkommen bei ToonTune AI. So wird Ihre Stimme in Ihrem Video klingen. Lassen Sie uns gemeinsam etwas Großartiges erschaffen!",
  it: "Benvenuto su ToonTune AI. Ecco come suonerà la tua voce nel tuo video. Creiamo qualcosa di straordinario insieme!",
  pt: "Bem-vindo ao ToonTune AI. É assim que sua voz soará em seu vídeo. Vamos criar algo incrível juntos!",
  ru: "Добро пожаловать в ToonTune AI. Вот как будет звучать ваш голос в видео. Давайте создадим что-то потрясающее вместе!",
  zh: "欢迎使用ToonTune AI。这就是您的声音在视频中的效果。让我们一起创造精彩的内容！",
  ja: "ToonTune AIへようこそ。これがあなたのビデオでの音声です。一緒に素晴らしいものを作りましょう！",
  ko: "ToonTune AI에 오신 것을 환영합니다. 비디오에서 당신의 목소리가 이렇게 들릴 것입니다. 함께 멋진 것을 만들어봅시다!",
  ar: "مرحباً بك في ToonTune AI. هكذا سيبدو صوتك في الفيديو. دعنا ننشئ شيئاً رائعاً معاً!",
  hi: "ToonTune AI में आपका स्वागत है। आपके वीडियो में आपकी आवाज़ ऐसी सुनाई देगी। आइए मिलकर कुछ शानदार बनाएं!",
  he: "ברוכים הבאים ל-ToonTune AI. כך יישמע הקול שלך בסרטון. בואו ניצור יחד משהו מדהים!"
};

// Cache for preview URLs (in-memory cache)
const previewCache = new Map();
const CACHE_TTL = 3600000; // 1 hour in milliseconds

async function previewVoice(input, context) {
  const {
    voiceId,
    language = 'en',
    sampleText
  } = input;

  logger.info('Generating voice preview', {
    voiceId,
    language,
    customText: !!sampleText
  });

  try {
    // Check cache first
    const cacheKey = `${voiceId}-${language}-${sampleText || 'default'}`;
    const cached = previewCache.get(cacheKey);
    
    if (cached && cached.expiresAt > Date.now()) {
      logger.info('Returning cached preview', { cacheKey });
      return {
        previewUrl: cached.url,
        duration: cached.duration,
        expiresAt: new Date(cached.expiresAt).toISOString(),
        cached: true
      };
    }

    // Get sample text
    const textToSpeak = sampleText || SAMPLE_TEXTS[language] || SAMPLE_TEXTS.en;

    // Generate voiceover using the main function
    const voiceoverResult = await generateVoiceover({
      text: textToSpeak,
      voiceId,
      language,
      speed: 1.0,
      pitch: 1.0
    }, context);

    // Generate presigned URL with 1 hour expiration
    const expiresIn = 3600; // 1 hour
    const presignedUrl = await awsHelpers.getPresignedUrl(
      process.env.S3_BUCKET_NAME,
      extractS3KeyFromUrl(voiceoverResult.audioUrl),
      expiresIn
    );

    const expiresAt = Date.now() + (expiresIn * 1000);

    // Cache the result
    previewCache.set(cacheKey, {
      url: presignedUrl,
      duration: voiceoverResult.duration,
      expiresAt
    });

    // Clean up old cache entries
    cleanupCache();

    logger.info('Voice preview generated successfully', {
      duration: voiceoverResult.duration,
      cacheKey
    });

    return {
      previewUrl: presignedUrl,
      duration: voiceoverResult.duration,
      expiresAt: new Date(expiresAt).toISOString(),
      cached: false,
      metadata: {
        voiceId,
        language,
        service: voiceoverResult.service
      }
    };

  } catch (error) {
    logger.error('Failed to generate voice preview', {
      error: error.message,
      voiceId,
      language
    });
    throw error;
  }
}

function extractS3KeyFromUrl(url) {
  if (url.includes('amazonaws.com')) {
    // Extract key from S3 URL
    const match = url.match(/amazonaws\.com\/(.+)$/);
    return match ? match[1] : url;
  } else if (url.includes(process.env.CDN_DOMAIN)) {
    // Extract key from CDN URL
    const match = url.match(new RegExp(`${process.env.CDN_DOMAIN}/(.+)$`));
    return match ? match[1] : url;
  }
  return url;
}

function cleanupCache() {
  const now = Date.now();
  for (const [key, value] of previewCache.entries()) {
    if (value.expiresAt < now) {
      previewCache.delete(key);
    }
  }
}

module.exports = {
  previewVoice
};