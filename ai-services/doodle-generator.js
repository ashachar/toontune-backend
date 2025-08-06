const OpenAI = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const Replicate = require('replicate');

class DoodleGenerator {
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
    if (process.env.REPLICATE_API_TOKEN) {
      this.replicate = new Replicate({
        auth: process.env.REPLICATE_API_TOKEN
      });
    }
  }

  async generateDoodleFromText(description, options = {}) {
    const {
      style = 'simple',
      count = 4,
      provider = 'mock' // 'openai', 'replicate', 'mock'
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
    const mockImages = [
      '/assets/doodles/sample-1.svg',
      '/assets/doodles/sample-2.svg',
      '/assets/doodles/sample-3.svg',
      '/assets/doodles/sample-4.svg',
      '/assets/doodles/sample-5.svg',
      '/assets/doodles/sample-6.svg'
    ];

    return {
      prompt,
      images: mockImages.slice(0, count),
      provider: 'mock',
      message: 'Mock doodles generated for testing'
    };
  }

  async analyzeDoodleContent(imagePath) {
    if (!this.geminiModel) {
      throw new Error('Gemini API key not configured');
    }

    try {
      const imageData = await this.loadImage(imagePath);
      const result = await this.geminiModel.generateContent([
        "Analyze this doodle image and describe its content, style, and key elements",
        imageData
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

  async loadImage(imagePath) {
    const fs = require('fs').promises;
    const imageBuffer = await fs.readFile(imagePath);
    return {
      inlineData: {
        data: imageBuffer.toString('base64'),
        mimeType: 'image/png'
      }
    };
  }

  async generateAnimationSequence(description, frameCount = 10) {
    // Generate a sequence of related doodles for animation
    const frames = [];
    const basePrompt = this.enhancePrompt(description, 'simple');

    for (let i = 0; i < frameCount; i++) {
      const framePrompt = `${basePrompt}, animation frame ${i + 1} of ${frameCount}`;
      const result = await this.generateDoodleFromText(description, {
        count: 1,
        provider: this.config.provider || 'mock'
      });
      frames.push(result.images[0]);
    }

    return {
      description,
      frames,
      frameCount,
      duration: frameCount * 0.1 // 100ms per frame
    };
  }

  async optimizeDoodleForCanvas(imagePath) {
    // Process and optimize doodle for canvas rendering
    const sharp = require('sharp');
    
    try {
      const optimized = await sharp(imagePath)
        .resize(1920, 1080, { fit: 'inside', background: { r: 255, g: 255, b: 255, alpha: 0 } })
        .png({ quality: 90 })
        .toBuffer();

      return {
        data: optimized,
        format: 'png',
        dimensions: { width: 1920, height: 1080 }
      };
    } catch (error) {
      console.error('Image optimization error:', error);
      throw error;
    }
  }
}

module.exports = DoodleGenerator;