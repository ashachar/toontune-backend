/**
 * Generate Doodle Asset Function
 * Creates individual doodle assets with transparent backgrounds
 */

const { aiHelpers } = require('../utils/ai-clients');
const { awsHelpers } = require('../utils/aws-clients');
const { logger } = require('../utils/logger');
const sharp = require('sharp');
const crypto = require('crypto');

// Asset category configurations
const CATEGORY_PROMPTS = {
  character: 'cartoon character, full body, centered',
  prop: 'object, item, tool, centered composition',
  background: 'scene, environment, landscape, wide composition',
  icon: 'simple icon, symbol, minimalist design'
};

// Style configurations (reused from doodle generator)
const STYLE_MODIFIERS = {
  minimal: 'minimalist line drawing, single continuous line',
  simple: 'simple line art, clean lines',
  funky: 'playful doodle style, quirky',
  romantic: 'elegant line art, flowing curves',
  scribble: 'rough sketch style, hand-drawn',
  halloween: 'spooky doodle style, gothic'
};

async function generateDoodleAsset(input, context) {
  const {
    description,
    category = 'prop',
    style = 'simple',
    variations = 1,
    // Legacy support for old API
    provider = 'auto',
    count = variations
  } = input;

  const actualVariations = count || variations;

  logger.info('Generating doodle asset', {
    description: description.substring(0, 100),
    category,
    style,
    variations: actualVariations
  });

  try {
    const assets = [];
    const categoryModifier = CATEGORY_PROMPTS[category] || CATEGORY_PROMPTS.prop;
    const styleModifier = STYLE_MODIFIERS[style] || STYLE_MODIFIERS.simple;
    
    // Build enhanced prompt
    const basePrompt = `${description}, ${categoryModifier}, ${styleModifier}, black and white doodle style, white background, high contrast, clean lines`;

    // Generate variations
    for (let i = 0; i < actualVariations; i++) {
      const variationPrompt = i > 0 
        ? `${basePrompt}, variation ${i + 1}, different angle or pose`
        : basePrompt;

      try {
        // Generate image
        const imageUrls = await aiHelpers.generateImage(variationPrompt, {
          count: 1,
          size: '1024x1024',
          quality: 'standard'
        });

        if (!imageUrls || imageUrls.length === 0) {
          throw new Error('No image generated');
        }

        const imageUrl = imageUrls[0];

        // Process and upload image
        let processedAsset;
        if (!imageUrl.includes('placeholder')) {
          processedAsset = await processAndUploadAsset(
            imageUrl,
            description,
            category,
            i
          );
        } else {
          // Mock/placeholder response
          processedAsset = {
            url: imageUrl,
            thumbnail: imageUrl,
            metadata: {
              width: 1024,
              height: 1024,
              format: 'png'
            }
          };
        }

        assets.push({
          ...processedAsset,
          prompt: variationPrompt,
          variation: i + 1,
          category,
          style
        });

      } catch (error) {
        logger.warn(`Failed to generate variation ${i + 1}`, {
          error: error.message
        });
        
        // Add failed asset to results
        assets.push({
          url: null,
          thumbnail: null,
          prompt: variationPrompt,
          variation: i + 1,
          category,
          style,
          error: error.message,
          status: 'failed'
        });
      }
    }

    const successCount = assets.filter(a => a.url).length;

    logger.info('Doodle assets generated', {
      requested: actualVariations,
      successful: successCount
    });

    // Legacy API compatibility
    if (input.action === 'generate-doodle') {
      // Return legacy format
      return {
        prompt: basePrompt,
        images: assets.map(a => a.url).filter(Boolean),
        provider: provider === 'auto' ? 'ai' : provider,
        message: `Generated ${successCount} doodle(s)`
      };
    }

    // New API format
    return {
      assets,
      statistics: {
        requested: actualVariations,
        successful: successCount,
        failed: actualVariations - successCount
      }
    };

  } catch (error) {
    logger.error('Failed to generate doodle asset', {
      error: error.message
    });
    throw error;
  }
}

async function processAndUploadAsset(imageUrl, description, category, variation) {
  try {
    // Download image
    const fetch = require('node-fetch');
    const response = await fetch(imageUrl);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }

    const imageBuffer = await response.buffer();

    // Process image with sharp
    const processedImage = await sharp(imageBuffer)
      .resize(1024, 1024, {
        fit: 'contain',
        background: { r: 255, g: 255, b: 255, alpha: 0 }
      })
      .png({ compressionLevel: 9 })
      .toBuffer();

    // Create thumbnail
    const thumbnailBuffer = await sharp(imageBuffer)
      .resize(256, 256, {
        fit: 'contain',
        background: { r: 255, g: 255, b: 255, alpha: 0 }
      })
      .png({ compressionLevel: 9 })
      .toBuffer();

    // Generate S3 keys
    const timestamp = Date.now();
    const hash = crypto.createHash('md5')
      .update(`${description}-${category}-${variation}`)
      .digest('hex')
      .substring(0, 8);
    
    const assetKey = `assets/${category}/${timestamp}-${hash}.png`;
    const thumbnailKey = `assets/${category}/thumbnails/${timestamp}-${hash}-thumb.png`;

    // Upload to S3
    const [assetUrl, thumbnailUrl] = await Promise.all([
      awsHelpers.uploadToS3(
        process.env.S3_BUCKET_NAME,
        assetKey,
        processedImage,
        'image/png',
        {
          category,
          description: description.substring(0, 100),
          variation: String(variation)
        }
      ),
      awsHelpers.uploadToS3(
        process.env.S3_BUCKET_NAME,
        thumbnailKey,
        thumbnailBuffer,
        'image/png',
        {
          type: 'thumbnail',
          parent: assetKey
        }
      )
    ]);

    // Get image metadata
    const metadata = await sharp(processedImage).metadata();

    // Return CDN URLs if configured
    const finalAssetUrl = process.env.CDN_DOMAIN 
      ? `https://${process.env.CDN_DOMAIN}/${assetKey}`
      : assetUrl;
    
    const finalThumbnailUrl = process.env.CDN_DOMAIN
      ? `https://${process.env.CDN_DOMAIN}/${thumbnailKey}`
      : thumbnailUrl;

    return {
      url: finalAssetUrl,
      thumbnail: finalThumbnailUrl,
      metadata: {
        width: metadata.width,
        height: metadata.height,
        format: metadata.format,
        size: processedImage.length,
        hasAlpha: true
      }
    };

  } catch (error) {
    logger.error('Failed to process and upload asset', {
      error: error.message
    });
    throw error;
  }
}

module.exports = {
  generateDoodleAsset
};