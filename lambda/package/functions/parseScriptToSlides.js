/**
 * Parse Script To Slides Function
 * Splits a script into logical slides with timing and image prompts
 */

const { aiHelpers } = require('../utils/ai-clients');
const { logger } = require('../utils/logger');

// Constants for slide generation
const DEFAULT_MAX_SLIDES = 10;
const DEFAULT_SECONDS_PER_SLIDE = 5;
const MIN_WORDS_PER_SLIDE = 15;
const MAX_WORDS_PER_SLIDE = 50;
const WORDS_PER_MINUTE = 150;

async function parseScriptToSlides(input, context) {
  const {
    script,
    maxSlidesCount = DEFAULT_MAX_SLIDES,
    targetDuration = DEFAULT_SECONDS_PER_SLIDE
  } = input;

  logger.info('Parsing script to slides', {
    scriptLength: script.length,
    maxSlidesCount,
    targetDuration
  });

  try {
    // Split script into sentences
    const sentences = splitIntoSentences(script);
    
    // Group sentences into slides
    const slideGroups = groupSentencesIntoSlides(
      sentences,
      maxSlidesCount,
      targetDuration
    );

    // Generate slide content with AI enhancement
    const slides = await Promise.all(
      slideGroups.map(async (group, index) => {
        const slideText = group.text;
        const slideNumber = index + 1;

        // Generate image prompt for this slide
        const imagePrompt = await generateImagePrompt(slideText, script);

        // Extract keywords
        const keywords = extractKeywords(slideText);

        // Calculate duration based on word count
        const wordCount = countWords(slideText);
        const duration = calculateSlideDuration(wordCount);

        return {
          slideNumber,
          text: slideText,
          duration,
          imagePrompt,
          keywords,
          wordCount,
          sentenceCount: group.sentences.length
        };
      })
    );

    // Calculate total duration
    const totalDuration = slides.reduce((sum, slide) => sum + slide.duration, 0);

    logger.info('Script parsed into slides successfully', {
      slideCount: slides.length,
      totalDuration
    });

    return {
      slides,
      totalDuration,
      slideCount: slides.length,
      metadata: {
        averageWordsPerSlide: Math.round(countWords(script) / slides.length),
        averageDurationPerSlide: Math.round(totalDuration / slides.length)
      }
    };

  } catch (error) {
    logger.error('Failed to parse script to slides', {
      error: error.message
    });
    throw error;
  }
}

function splitIntoSentences(text) {
  // Improved sentence splitting that handles abbreviations
  const sentences = [];
  const preliminarySplit = text.split(/(?<=[.!?])\s+/);
  
  preliminarySplit.forEach(sentence => {
    sentence = sentence.trim();
    if (sentence) {
      // Check if this might be an abbreviation that was wrongly split
      if (sentence.length < 10 && sentences.length > 0) {
        // Possibly an abbreviation, merge with previous
        const lastIndex = sentences.length - 1;
        sentences[lastIndex] += ' ' + sentence;
      } else {
        sentences.push(sentence);
      }
    }
  });

  return sentences.map(s => ({
    text: s,
    wordCount: countWords(s)
  }));
}

function groupSentencesIntoSlides(sentences, maxSlides, targetDuration) {
  const groups = [];
  let currentGroup = {
    sentences: [],
    text: '',
    wordCount: 0
  };

  const targetWordsPerSlide = Math.min(
    MAX_WORDS_PER_SLIDE,
    Math.max(
      MIN_WORDS_PER_SLIDE,
      Math.floor(sentences.reduce((sum, s) => sum + s.wordCount, 0) / maxSlides)
    )
  );

  for (const sentence of sentences) {
    const wouldExceedLimit = currentGroup.wordCount + sentence.wordCount > targetWordsPerSlide;
    const hasMinimumContent = currentGroup.wordCount >= MIN_WORDS_PER_SLIDE;
    
    if (wouldExceedLimit && hasMinimumContent && groups.length < maxSlides - 1) {
      // Start a new slide
      groups.push(currentGroup);
      currentGroup = {
        sentences: [sentence],
        text: sentence.text,
        wordCount: sentence.wordCount
      };
    } else {
      // Add to current slide
      currentGroup.sentences.push(sentence);
      currentGroup.text += (currentGroup.text ? ' ' : '') + sentence.text;
      currentGroup.wordCount += sentence.wordCount;
    }
  }

  // Add the last group
  if (currentGroup.sentences.length > 0) {
    groups.push(currentGroup);
  }

  // Balance slides if we have too few
  if (groups.length < Math.min(3, maxSlides) && groups.length > 0) {
    return rebalanceSlides(groups, Math.min(5, maxSlides));
  }

  return groups;
}

function rebalanceSlides(groups, targetCount) {
  // Combine all sentences
  const allSentences = groups.flatMap(g => g.sentences);
  const totalWords = allSentences.reduce((sum, s) => sum + s.wordCount, 0);
  const wordsPerSlide = Math.floor(totalWords / targetCount);

  const newGroups = [];
  let currentGroup = {
    sentences: [],
    text: '',
    wordCount: 0
  };

  for (const sentence of allSentences) {
    if (currentGroup.wordCount >= wordsPerSlide && newGroups.length < targetCount - 1) {
      newGroups.push(currentGroup);
      currentGroup = {
        sentences: [],
        text: '',
        wordCount: 0
      };
    }
    
    currentGroup.sentences.push(sentence);
    currentGroup.text += (currentGroup.text ? ' ' : '') + sentence.text;
    currentGroup.wordCount += sentence.wordCount;
  }

  if (currentGroup.sentences.length > 0) {
    newGroups.push(currentGroup);
  }

  return newGroups;
}

async function generateImagePrompt(slideText, fullScript) {
  try {
    const prompt = `Based on this slide text from an explainer video, generate a simple, descriptive image prompt for a doodle-style illustration:

Slide text: "${slideText}"

Context: This is part of a video about: "${fullScript.substring(0, 200)}..."

Generate a concise image prompt (max 50 words) that:
1. Captures the main concept of the slide
2. Uses simple, visual elements
3. Works well as a black and white doodle
4. Avoids text or complex details

Image prompt:`;

    const imagePrompt = await aiHelpers.generateText(prompt, {
      maxTokens: 100,
      temperature: 0.8
    });

    // Clean and format the prompt
    return cleanImagePrompt(imagePrompt);
  } catch (error) {
    logger.warn('Failed to generate image prompt with AI, using fallback', {
      error: error.message
    });
    // Fallback to keyword-based prompt
    return generateFallbackImagePrompt(slideText);
  }
}

function cleanImagePrompt(prompt) {
  return prompt
    .replace(/^["']|["']$/g, '') // Remove quotes
    .replace(/^Image prompt:\s*/i, '') // Remove prefix
    .trim()
    .substring(0, 200); // Limit length
}

function generateFallbackImagePrompt(text) {
  const keywords = extractKeywords(text);
  const mainConcepts = keywords.slice(0, 3).join(', ');
  return `Simple doodle illustration of ${mainConcepts || 'concept'}`;
}

function extractKeywords(text) {
  // Remove common words and extract important terms
  const commonWords = new Set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
    'they', 'them', 'their', 'what', 'which', 'who', 'when', 'where', 'why',
    'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very'
  ]);

  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(word => word.length > 3 && !commonWords.has(word));

  // Count word frequency
  const wordFreq = {};
  words.forEach(word => {
    wordFreq[word] = (wordFreq[word] || 0) + 1;
  });

  // Sort by frequency and return top keywords
  return Object.entries(wordFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([word]) => word);
}

function countWords(text) {
  return text.trim().split(/\s+/).filter(word => word.length > 0).length;
}

function calculateSlideDuration(wordCount) {
  // Calculate based on average speaking rate
  const secondsPerWord = 60 / WORDS_PER_MINUTE;
  const baseDuration = wordCount * secondsPerWord;
  
  // Add buffer for comprehension and transitions
  const buffer = 1.5; // 1.5 seconds minimum per slide
  
  return Math.max(buffer, Math.round(baseDuration * 10) / 10);
}

module.exports = {
  parseScriptToSlides
};