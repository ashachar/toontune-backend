/**
 * Generate Slide Images Function
 * Generates doodle-style images for multiple slides
 */

const { aiHelpers } = require('../utils/ai-clients');
const { awsHelpers } = require('../utils/aws-clients');
const { logger } = require('../utils/logger');
const crypto = require('crypto');

// Style configurations
const STYLE_MODIFIERS = {
  minimal: 'minimalist line drawing, single continuous line, very simple',
  simple: 'simple line art, clean lines, basic shapes',
  funky: 'playful doodle style, quirky characters, fun elements',
  romantic: 'elegant line art, flowing curves, decorative elements',
  scribble: 'rough sketch style, hand-drawn scribbles, artistic',
  halloween: 'spooky doodle style, halloween themed, gothic elements'
};

const DEFAULT_BATCH_SIZE = 4;
const MAX_RETRIES = 3;

async function generateSlideImages(input, context) {
  const {
    slides,
    projectStyle = 'simple',
    batchSize = DEFAULT_BATCH_SIZE
  } = input;

  logger.info('Generating images for slides', {
    slideCount: slides.length,
    projectStyle,
    batchSize
  });

  const results = [];
  const styleModifier = STYLE_MODIFIERS[projectStyle] || STYLE_MODIFIERS.simple;
  const basePromptSuffix = `, black and white doodle style, ${styleModifier}, white background, no text or words`;

  try {
    // Process slides in batches
    for (let i = 0; i < slides.length; i += batchSize) {
      const batch = slides.slice(i, i + batchSize);
      
      logger.info(`Processing batch ${Math.floor(i / batchSize) + 1}`, {
        batchStart: i,
        batchSize: batch.length
      });

      // Generate images for batch in parallel
      const batchResults = await Promise.all(
        batch.map(async (slide) => {
          return generateSlideImage(slide, styleModifier, basePromptSuffix);
        })
      );

      results.push(...batchResults);

      // Add a small delay between batches to avoid rate limiting
      if (i + batchSize < slides.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    // Calculate statistics
    const successCount = results.filter(r => r.status === 'success').length;
    const failureCount = results.filter(r => r.status === 'failed').length;

    logger.info('Image generation completed', {
      total: results.length,
      successful: successCount,
      failed: failureCount
    });

    return {
      images: results,
      statistics: {
        total: results.length,
        successful: successCount,
        failed: failureCount,
        successRate: (successCount / results.length) * 100
      }
    };

  } catch (error) {
    logger.error('Failed to generate slide images', {
      error: error.message
    });
    throw error;
  }
}

async function generateSlideImage(slide, styleModifier, basePromptSuffix) {
  const { slideNumber, imagePrompt, style } = slide;
  
  // Enhance the prompt with style modifiers
  const fullPrompt = `${imagePrompt}${basePromptSuffix}`;
  
  let attempt = 0;
  let lastError = null;

  while (attempt < MAX_RETRIES) {
    try {
      attempt++;
      
      logger.debug(`Generating image for slide ${slideNumber}, attempt ${attempt}`, {
        prompt: fullPrompt.substring(0, 100)
      });

      // Generate image
      const imageUrls = await aiHelpers.generateImage(fullPrompt, {
        count: 1,
        size: '1920x1080',
        quality: 'standard'
      });

      if (!imageUrls || imageUrls.length === 0) {
        throw new Error('No image generated');
      }

      const imageUrl = imageUrls[0];

      // If it's a real image URL (not placeholder), upload to S3
      let finalUrl = imageUrl;
      if (!imageUrl.includes('placeholder')) {
        finalUrl = await uploadImageToS3(imageUrl, slideNumber, slide.projectId);
      }

      return {
        slideNumber,
        imageUrl: finalUrl,
        prompt: fullPrompt,
        status: 'success',
        attempts: attempt
      };

    } catch (error) {
      lastError = error;
      logger.warn(`Failed to generate image for slide ${slideNumber}, attempt ${attempt}`, {
        error: error.message
      });

      if (attempt < MAX_RETRIES) {
        // Wait before retry with exponential backoff
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }
  }

  // All retries failed
  return {
    slideNumber,
    imageUrl: null,
    prompt: fullPrompt,
    status: 'failed',
    error: lastError?.message || 'Unknown error',
    attempts: attempt
  };
}

async function uploadImageToS3(imageUrl, slideNumber, projectId) {
  try {
    // Download image
    const fetch = require('node-fetch');
    const response = await fetch(imageUrl);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }

    const imageBuffer = await response.buffer();
    
    // Generate S3 key
    const timestamp = Date.now();
    const hash = crypto.createHash('md5').update(`${projectId}-${slideNumber}`).digest('hex');
    const key = `slides/${projectId || 'default'}/${timestamp}-${slideNumber}-${hash}.png`;

    // Upload to S3
    const s3Url = await awsHelpers.uploadToS3(
      process.env.S3_BUCKET_NAME,
      key,
      imageBuffer,
      'image/png',
      {
        slideNumber: String(slideNumber),
        projectId: projectId || 'none',
        generatedAt: new Date().toISOString()
      }
    );

    // If CDN is configured, return CDN URL
    if (process.env.CDN_DOMAIN) {
      return `https://${process.env.CDN_DOMAIN}/${key}`;
    }

    return s3Url;

  } catch (error) {
    logger.error('Failed to upload image to S3', {
      error: error.message,
      slideNumber
    });
    // Return original URL if upload fails
    return imageUrl;
  }
}

module.exports = {
  generateSlideImages
};