const OpenAI = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const Replicate = require('replicate');
const sharp = require('sharp');

class DoodleGeneratorLambda {
  constructor(config = {}) {
    this.config = config;
    this.initializeProviders();
  }

  initializeProviders() {
    // Initialize OpenAI
    if (process.env.OPENAI_API_KEY) {
      this.openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY
      });
    }

    // Initialize Google Gemini
    if (process.env.GEMINI_API_KEY) {
      this.genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
      this.geminiModel = this.genAI.getGenerativeModel({ model: "gemini-pro-vision" });
    }

    // Initialize Replicate for image generation
    if (process.env.REPLICATE_API_TOKEN || process.env.REPLICATE_API_KEY) {
      this.replicate = new Replicate({
        auth: process.env.REPLICATE_API_TOKEN || process.env.REPLICATE_API_KEY
      });
    }
  }

  async generateDoodleFromText(description, options = {}) {
    const {
      style = 'simple',
      count = 4,
      provider = 'mock'
    } = options;

    const enhancedPrompt = this.enhancePrompt(description, style);

    switch (provider) {
      case 'openai':
        return await this.generateWithOpenAI(enhancedPrompt, count);
      case 'replicate':
        return await this.generateWithReplicate(enhancedPrompt, count);
      default:
        return await this.generateMockDoodles(enhancedPrompt, count);
    }
  }

  enhancePrompt(description, style) {
    const stylePrompts = {
      simple: 'black and white doodle style, simple line art, white background, hand-drawn sketch',
      detailed: 'detailed black and white illustration, intricate line work, white background',
      cartoon: 'cartoon doodle style, playful lines, white background, whimsical',
      minimalist: 'minimalist line drawing, single continuous line, white background'
    };

    return `${description}, ${stylePrompts[style] || stylePrompts.simple}`;
  }

  async generateWithOpenAI(prompt, count) {
    if (!this.openai) {
      throw new Error('OpenAI API key not configured');
    }

    try {
      const promises = [];
      for (let i = 0; i < count; i++) {
        promises.push(
          this.openai.images.generate({
            model: "dall-e-3",
            prompt: prompt,
            n: 1,
            size: "1024x1024",
            style: "natural"
          })
        );
      }

      const results = await Promise.all(promises);
      return {
        prompt,
        images: results.map(r => r.data[0].url),
        provider: 'openai'
      };
    } catch (error) {
      console.error('OpenAI generation error:', error);
      throw error;
    }
  }

  async generateWithReplicate(prompt, count) {
    if (!this.replicate) {
      throw new Error('Replicate API token not configured');
    }

    try {
      const output = await this.replicate.run(
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        {
          input: {
            prompt: prompt,
            negative_prompt: "color, shading, gradient, complex background",
            num_outputs: count,
            width: 1024,
            height: 1024
          }
        }
      );

      return {
        prompt,
        images: output,
        provider: 'replicate'
      };
    } catch (error) {
      console.error('Replicate generation error:', error);
      throw error;
    }
  }

  async generateMockDoodles(prompt, count) {
    // Mock implementation for testing
    const mockImages = [];
    for (let i = 0; i < count; i++) {
      mockImages.push(`https://via.placeholder.com/1024x1024/ffffff/000000?text=Doodle+${i + 1}`);
    }

    return {
      prompt,
      images: mockImages,
      provider: 'mock',
      message: 'Mock doodles generated for testing'
    };
  }

  async analyzeDoodleContent(imageData, mimeType = 'image/png') {
    if (!this.geminiModel) {
      // Return mock analysis if Gemini is not configured
      return {
        analysis: 'Mock analysis: This appears to be a doodle image with simple line art.',
        provider: 'mock'
      };
    }

    try {
      const result = await this.geminiModel.generateContent([
        "Analyze this doodle image and describe its content, style, and key elements",
        {
          inlineData: {
            data: imageData,
            mimeType: mimeType
          }
        }
      ]);

      return {
        analysis: result.response.text(),
        provider: 'gemini'
      };
    } catch (error) {
      console.error('Gemini analysis error:', error);
      throw error;
    }
  }

  async generateAnimationSequence(description, frameCount = 10) {
    // Generate a sequence of related doodles for animation
    const frames = [];
    const basePrompt = this.enhancePrompt(description, 'simple');

    // For Lambda, generate frames more efficiently
    const framePrompts = [];
    for (let i = 0; i < frameCount; i++) {
      framePrompts.push(`${basePrompt}, animation frame ${i + 1} of ${frameCount}`);
    }

    // Generate all frames in parallel for better performance
    const framePromises = framePrompts.map(async (framePrompt, index) => {
      const result = await this.generateDoodleFromText(description, {
        count: 1,
        provider: this.config.provider || 'mock'
      });
      return result.images[0];
    });

    const generatedFrames = await Promise.all(framePromises);

    return {
      description,
      frames: generatedFrames,
      frameCount,
      duration: frameCount * 0.1 // 100ms per frame
    };
  }

  async optimizeDoodleForCanvas(imageData, options = {}) {
    const { width = 1920, height = 1080 } = options;
    
    try {
      // Convert base64 to buffer if needed
      const buffer = Buffer.isBuffer(imageData) 
        ? imageData 
        : Buffer.from(imageData, 'base64');

      const optimized = await sharp(buffer)
        .resize(width, height, { 
          fit: 'inside', 
          background: { r: 255, g: 255, b: 255, alpha: 0 } 
        })
        .png({ quality: 90 })
        .toBuffer();

      return {
        data: optimized.toString('base64'),
        format: 'png',
        dimensions: { width, height }
      };
    } catch (error) {
      console.error('Image optimization error:', error);
      throw error;
    }
  }
}

module.exports = DoodleGeneratorLambda;