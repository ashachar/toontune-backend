/**
 * AWS Lambda function for ToonTune.ai services
 * Handles doodle generation, analysis, and optimization
 */

const DoodleGenerator = require('./doodle_generator_lambda');

// Lightweight global variables for warm starts
let doodleGen = null;
let initialized = false;

// Initialize providers once
const initializeProviders = () => {
  if (!initialized) {
    console.log('Initializing ToonTune AI Generator...');
    doodleGen = new DoodleGenerator();
    initialized = true;
    console.log('ToonTune AI Generator initialized successfully');
  }
  return doodleGen;
};

// Lambda handler
exports.handler = async (event, context) => {
  try {
    // Handle warm-up pings
    if (event.warmup) {
      return {
        statusCode: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify({ message: 'Lambda is warm!' })
      };
    }

    // Parse request body
    const body = typeof event.body === 'string' ? JSON.parse(event.body) : event.body || {};
    const { action, ...params } = body;

    // Validate action
    if (!action) {
      return {
        statusCode: 400,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*'
        },
        body: JSON.stringify({
          success: false,
          error: 'Action is required'
        })
      };
    }

    // Initialize providers only when needed (not on warm-up)
    const generator = initializeProviders();

    let result;
    
    // Route to appropriate action
    switch (action) {
      case 'generate-doodle':
        result = await handleGenerateDoodle(generator, params);
        break;
      
      case 'generate-animation':
        result = await handleGenerateAnimation(generator, params);
        break;
      
      case 'analyze-doodle':
        result = await handleAnalyzeDoodle(generator, params);
        break;
      
      case 'optimize-doodle':
        result = await handleOptimizeDoodle(generator, params);
        break;
      
      default:
        return {
          statusCode: 400,
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
          },
          body: JSON.stringify({
            success: false,
            error: `Unknown action: ${action}`
          })
        };
    }

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({
        success: true,
        ...result,
        processingTime: context.getRemainingTimeInMillis ? 
          (context.functionTimeout * 1000 - context.getRemainingTimeInMillis()) : null
      })
    };

  } catch (error) {
    console.error('Lambda handler error:', error);
    
    return {
      statusCode: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({
        success: false,
        error: error.message,
        stack: process.env.DEBUG === 'true' ? error.stack : undefined
      })
    };
  }
};

// Handler functions
async function handleGenerateDoodle(generator, params) {
  const { description, style = 'simple', count = 4, provider = 'mock' } = params;
  
  if (!description) {
    throw new Error('Description is required for doodle generation');
  }
  
  return await generator.generateDoodleFromText(description, {
    style,
    count,
    provider
  });
}

async function handleGenerateAnimation(generator, params) {
  const { description, frameCount = 10 } = params;
  
  if (!description) {
    throw new Error('Description is required for animation generation');
  }
  
  return await generator.generateAnimationSequence(description, frameCount);
}

async function handleAnalyzeDoodle(generator, params) {
  const { imageData, mimeType = 'image/png' } = params;
  
  if (!imageData) {
    throw new Error('Image data is required for analysis');
  }
  
  return await generator.analyzeDoodleContent(imageData, mimeType);
}

async function handleOptimizeDoodle(generator, params) {
  const { imageData, width = 1920, height = 1080 } = params;
  
  if (!imageData) {
    throw new Error('Image data is required for optimization');
  }
  
  return await generator.optimizeDoodleForCanvas(imageData, { width, height });
}