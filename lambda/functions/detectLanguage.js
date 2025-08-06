/**
 * Detect Language Function
 * Detects language from text using AWS Comprehend
 */

const { comprehend } = require('../utils/aws-clients');
const { logger } = require('../utils/logger');

async function detectLanguage(input, context) {
  const { text } = input;

  logger.info('Detecting language', {
    textLength: text.length
  });

  try {
    // Truncate text if too long (Comprehend has a 5000 byte limit)
    const textToAnalyze = text.slice(0, 5000);

    // Call AWS Comprehend
    const result = await comprehend.detectDominantLanguage({
      Text: textToAnalyze
    }).promise();

    // Process results
    const languages = result.Languages || [];
    
    if (languages.length === 0) {
      // No language detected, default to English
      logger.warn('No language detected, defaulting to English');
      return {
        language: 'en',
        confidence: 0.5,
        alternativeLanguages: []
      };
    }

    // Sort by confidence
    languages.sort((a, b) => b.Score - a.Score);

    // Get primary language
    const primary = languages[0];
    const alternativeLanguages = languages.slice(1, 4).map(lang => ({
      language: lang.LanguageCode,
      confidence: Math.round(lang.Score * 1000) / 1000
    }));

    logger.info('Language detected successfully', {
      language: primary.LanguageCode,
      confidence: primary.Score
    });

    return {
      language: primary.LanguageCode,
      confidence: Math.round(primary.Score * 1000) / 1000,
      alternativeLanguages,
      metadata: {
        textAnalyzedLength: textToAnalyze.length,
        languagesDetected: languages.length
      }
    };

  } catch (error) {
    logger.error('Failed to detect language', {
      error: error.message
    });

    // Fallback to simple heuristic detection
    const fallbackLanguage = detectLanguageHeuristic(text);
    
    return {
      language: fallbackLanguage,
      confidence: 0.3,
      alternativeLanguages: [],
      metadata: {
        fallback: true,
        error: error.message
      }
    };
  }
}

function detectLanguageHeuristic(text) {
  // Simple heuristic based on character sets
  const textSample = text.slice(0, 1000).toLowerCase();

  // Check for specific character sets
  if (/[\u4e00-\u9fff]/.test(textSample)) return 'zh'; // Chinese
  if (/[\u3040-\u309f\u30a0-\u30ff]/.test(textSample)) return 'ja'; // Japanese
  if (/[\uac00-\ud7af]/.test(textSample)) return 'ko'; // Korean
  if (/[\u0600-\u06ff]/.test(textSample)) return 'ar'; // Arabic
  if (/[\u0590-\u05ff]/.test(textSample)) return 'he'; // Hebrew
  if (/[\u0400-\u04ff]/.test(textSample)) return 'ru'; // Cyrillic
  if (/[\u0900-\u097f]/.test(textSample)) return 'hi'; // Devanagari (Hindi)

  // Check for common words in Latin-based languages
  if (/\b(the|and|of|to|in|is|that|it|with|for)\b/.test(textSample)) return 'en';
  if (/\b(de|la|el|los|las|un|una|que|en|por)\b/.test(textSample)) return 'es';
  if (/\b(le|la|de|et|un|une|des|les|pour|dans)\b/.test(textSample)) return 'fr';
  if (/\b(der|die|das|und|in|von|zu|mit|den|ein)\b/.test(textSample)) return 'de';
  if (/\b(il|la|di|e|che|un|una|per|con|del)\b/.test(textSample)) return 'it';
  if (/\b(o|a|de|e|que|um|uma|para|com|em)\b/.test(textSample)) return 'pt';

  // Default to English
  return 'en';
}

module.exports = {
  detectLanguage
};