/**
 * AI Service Client Initialization
 * Manages connections to various AI services
 */

const OpenAI = require('openai');
const Replicate = require('replicate');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize clients as singleton instances
let openaiClient = null;
let replicateClient = null;
let geminiClient = null;
let elevenLabsClient = null;

/**
 * Get or create OpenAI client
 */
function getOpenAIClient() {
  if (!openaiClient && process.env.OPENAI_API_KEY) {
    openaiClient = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
  }
  return openaiClient;
}

/**
 * Get or create Replicate client
 */
function getReplicateClient() {
  if (!replicateClient && (process.env.REPLICATE_API_KEY || process.env.REPLICATE_API_TOKEN)) {
    replicateClient = new Replicate({
      auth: process.env.REPLICATE_API_KEY || process.env.REPLICATE_API_TOKEN
    });
  }
  return replicateClient;
}

/**
 * Get or create Gemini client
 */
function getGeminiClient() {
  if (!geminiClient && process.env.GEMINI_API_KEY) {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    geminiClient = genAI.getGenerativeModel({ model: "gemini-pro" });
  }
  return geminiClient;
}

/**
 * Get or create ElevenLabs client
 */
function getElevenLabsClient() {
  if (!elevenLabsClient && process.env.ELEVENLABS_API_KEY) {
    // ElevenLabs uses REST API, so we'll create a simple client
    elevenLabsClient = {
      apiKey: process.env.ELEVENLABS_API_KEY,
      baseUrl: 'https://api.elevenlabs.io/v1',
      
      async textToSpeech(text, voiceId, options = {}) {
        const fetch = require('node-fetch');
        const response = await fetch(`${this.baseUrl}/text-to-speech/${voiceId}`, {
          method: 'POST',
          headers: {
            'Accept': 'audio/mpeg',
            'Content-Type': 'application/json',
            'xi-api-key': this.apiKey
          },
          body: JSON.stringify({
            text,
            model_id: options.modelId || 'eleven_monolingual_v1',
            voice_settings: {
              stability: options.stability || 0.5,
              similarity_boost: options.similarityBoost || 0.5
            }
          })
        });

        if (!response.ok) {
          throw new Error(`ElevenLabs API error: ${response.statusText}`);
        }

        const audioBuffer = await response.buffer();
        return audioBuffer;
      },

      async getVoices() {
        const fetch = require('node-fetch');
        const response = await fetch(`${this.baseUrl}/voices`, {
          headers: {
            'xi-api-key': this.apiKey
          }
        });

        if (!response.ok) {
          throw new Error(`ElevenLabs API error: ${response.statusText}`);
        }

        return response.json();
      }
    };
  }
  return elevenLabsClient;
}

/**
 * AI Service Helper Functions
 */
const aiHelpers = {
  /**
   * Generate text using the best available LLM
   */
  async generateText(prompt, options = {}) {
    // Try OpenAI first
    const openai = getOpenAIClient();
    if (openai) {
      try {
        const response = await openai.chat.completions.create({
          model: options.model || 'gpt-4-turbo-preview',
          messages: [
            { role: 'system', content: options.systemPrompt || 'You are a helpful assistant.' },
            { role: 'user', content: prompt }
          ],
          temperature: options.temperature || 0.7,
          max_tokens: options.maxTokens || 2000
        });
        return response.choices[0].message.content;
      } catch (error) {
        console.error('OpenAI error:', error);
      }
    }

    // Fallback to Gemini
    const gemini = getGeminiClient();
    if (gemini) {
      try {
        const result = await gemini.generateContent(prompt);
        return result.response.text();
      } catch (error) {
        console.error('Gemini error:', error);
      }
    }

    // If no LLM available, throw error
    throw new Error('No LLM service available. Please configure OPENAI_API_KEY or GEMINI_API_KEY');
  },

  /**
   * Generate image using the best available service
   */
  async generateImage(prompt, options = {}) {
    // Try OpenAI DALL-E first
    const openai = getOpenAIClient();
    if (openai) {
      try {
        const response = await openai.images.generate({
          model: options.model || 'dall-e-3',
          prompt: prompt,
          n: options.count || 1,
          size: options.size || '1024x1024',
          quality: options.quality || 'standard',
          style: options.style || 'natural'
        });
        return response.data.map(img => img.url);
      } catch (error) {
        console.error('DALL-E error:', error);
      }
    }

    // Fallback to Replicate/Stable Diffusion
    const replicate = getReplicateClient();
    if (replicate) {
      try {
        const output = await replicate.run(
          "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
          {
            input: {
              prompt: prompt,
              negative_prompt: options.negativePrompt || "color, complex background",
              num_outputs: options.count || 1,
              width: parseInt(options.size?.split('x')[0]) || 1024,
              height: parseInt(options.size?.split('x')[1]) || 1024
            }
          }
        );
        return Array.isArray(output) ? output : [output];
      } catch (error) {
        console.error('Replicate error:', error);
      }
    }

    // If no image service available, return placeholder
    return [`https://via.placeholder.com/1024x1024/ffffff/000000?text=${encodeURIComponent(prompt.slice(0, 50))}`];
  },

  /**
   * Detect language from text
   */
  async detectLanguage(text) {
    const { comprehend } = require('./aws-clients');
    
    try {
      const result = await comprehend.detectDominantLanguage({
        Text: text.slice(0, 5000) // Comprehend has a 5000 byte limit
      }).promise();

      const languages = result.Languages || [];
      const primary = languages[0] || { LanguageCode: 'en', Score: 0 };

      return {
        language: primary.LanguageCode,
        confidence: primary.Score,
        alternativeLanguages: languages.slice(1).map(lang => ({
          language: lang.LanguageCode,
          confidence: lang.Score
        }))
      };
    } catch (error) {
      console.error('Language detection error:', error);
      // Fallback to English
      return {
        language: 'en',
        confidence: 0.5,
        alternativeLanguages: []
      };
    }
  }
};

module.exports = {
  getOpenAIClient,
  getReplicateClient,
  getGeminiClient,
  getElevenLabsClient,
  aiHelpers
};