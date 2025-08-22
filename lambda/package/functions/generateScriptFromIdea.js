/**
 * Generate Script From Idea Function
 * Creates a complete script from a brief description
 */

const { aiHelpers } = require('../utils/ai-clients');
const { logger } = require('../utils/logger');

// Script length configurations (word counts)
const SCRIPT_LENGTHS = {
  short: { min: 100, max: 200, target: 150 },
  medium: { min: 200, max: 500, target: 350 },
  long: { min: 500, max: 1000, target: 750 }
};

// Tone templates
const TONE_TEMPLATES = {
  professional: {
    style: 'formal, business-oriented, data-driven',
    vocabulary: 'professional terminology, industry-standard language',
    structure: 'clear introduction, logical flow, strong conclusion'
  },
  casual: {
    style: 'conversational, friendly, relatable',
    vocabulary: 'everyday language, simple terms, personal pronouns',
    structure: 'engaging hook, storytelling elements, memorable ending'
  },
  educational: {
    style: 'informative, clear, structured',
    vocabulary: 'explanatory language, definitions, examples',
    structure: 'learning objectives, step-by-step progression, summary'
  }
};

async function generateScriptFromIdea(input, context) {
  const {
    idea,
    targetLength = 'medium',
    language = 'en',
    tone = 'educational'
  } = input;

  logger.info('Generating script from idea', {
    idea: idea.substring(0, 100),
    targetLength,
    language,
    tone
  });

  try {
    // Get script parameters
    const lengthConfig = SCRIPT_LENGTHS[targetLength] || SCRIPT_LENGTHS.medium;
    const toneConfig = TONE_TEMPLATES[tone] || TONE_TEMPLATES.educational;

    // Build the prompt
    const systemPrompt = buildSystemPrompt(language, tone, toneConfig);
    const userPrompt = buildUserPrompt(idea, lengthConfig, language);

    // Generate the script
    const script = await aiHelpers.generateText(userPrompt, {
      systemPrompt,
      maxTokens: lengthConfig.max * 2, // Approximate tokens
      temperature: 0.7
    });

    // Post-process the script
    const processedScript = postProcessScript(script, lengthConfig);

    // Calculate metrics
    const wordCount = countWords(processedScript);
    const estimatedDuration = calculateDuration(wordCount);

    // Detect actual language if needed
    let detectedLanguage = language;
    if (language === 'auto') {
      const detection = await aiHelpers.detectLanguage(processedScript);
      detectedLanguage = detection.language;
    }

    logger.info('Script generated successfully', {
      wordCount,
      estimatedDuration,
      language: detectedLanguage
    });

    return {
      script: processedScript,
      language: detectedLanguage,
      estimatedDuration,
      wordCount,
      metadata: {
        tone,
        targetLength,
        actualLength: getActualLength(wordCount)
      }
    };

  } catch (error) {
    logger.error('Failed to generate script', {
      error: error.message,
      idea: idea.substring(0, 100)
    });
    throw error;
  }
}

function buildSystemPrompt(language, tone, toneConfig) {
  const languageInstruction = language !== 'en' 
    ? `Generate the script in ${getLanguageName(language)} language.`
    : '';

  return `You are an expert scriptwriter for educational explainer videos. 
Your task is to create engaging, informative scripts that work well for voice-over narration.

Style Guidelines:
- Tone: ${toneConfig.style}
- Vocabulary: ${toneConfig.vocabulary}
- Structure: ${toneConfig.structure}

Important Requirements:
1. Write for voice-over narration (natural speaking rhythm)
2. Include natural pauses and transitions
3. Use clear, concise sentences
4. Avoid complex punctuation that's hard to read aloud
5. Include engaging hooks and memorable conclusions
${languageInstruction}

Format the script as continuous prose suitable for narration, not as bullet points or sections.`;
}

function buildUserPrompt(idea, lengthConfig) {
  return `Create a compelling explainer video script about: "${idea}"

Target length: ${lengthConfig.target} words (between ${lengthConfig.min}-${lengthConfig.max} words)

The script should:
1. Start with an attention-grabbing introduction
2. Explain the topic clearly and engagingly
3. Use relatable examples or analogies
4. Include smooth transitions between points
5. End with a memorable conclusion or call-to-action

Generate the complete script now:`;
}

function postProcessScript(script, lengthConfig) {
  // Clean up the script
  let processed = script
    .trim()
    .replace(/\n{3,}/g, '\n\n') // Remove excessive line breaks
    .replace(/^\s*[-•*]\s*/gm, '') // Remove bullet points
    .replace(/^#+\s*/gm, ''); // Remove markdown headers

  // Check word count and adjust if needed
  const wordCount = countWords(processed);
  
  if (wordCount > lengthConfig.max) {
    // Truncate if too long
    const words = processed.split(/\s+/);
    const truncated = words.slice(0, lengthConfig.max).join(' ');
    processed = truncated + '.';
  }

  // Ensure proper ending punctuation
  if (!/[.!?]$/.test(processed)) {
    processed += '.';
  }

  return processed;
}

function countWords(text) {
  return text.trim().split(/\s+/).filter(word => word.length > 0).length;
}

function calculateDuration(wordCount) {
  // Average speaking rate: 150 words per minute
  // Add some buffer for pauses
  const wordsPerMinute = 150;
  const baseSeconds = (wordCount / wordsPerMinute) * 60;
  const pauseBuffer = baseSeconds * 0.1; // Add 10% for pauses
  
  return Math.round(baseSeconds + pauseBuffer);
}

function getActualLength(wordCount) {
  if (wordCount <= 200) return 'short';
  if (wordCount <= 500) return 'medium';
  return 'long';
}

function getLanguageName(code) {
  const languages = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'he': 'Hebrew'
  };
  return languages[code] || code;
}

module.exports = {
  generateScriptFromIdea
};