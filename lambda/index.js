/**
 * ToonTune.ai Lambda Function Handler
 * Main entry point for all AI-powered functions
 */

const { generateScriptFromIdea } = require('./functions/generateScriptFromIdea');
const { parseScriptToSlides } = require('./functions/parseScriptToSlides');
const { generateSlideImages } = require('./functions/generateSlideImages');
const { generateVoiceover } = require('./functions/generateVoiceover');
const { previewVoice } = require('./functions/previewVoice');
const { syncVoiceToSlides } = require('./functions/syncVoiceToSlides');
const { detectLanguage } = require('./functions/detectLanguage');
const { generateDoodleAsset } = require('./functions/generateDoodleAsset');
const { applyStyleToProject } = require('./functions/applyStyleToProject');
const { batchProcessProject } = require('./functions/batchProcessProject');
const { validateInput } = require('./utils/validation');
const { formatResponse, formatError } = require('./utils/response');
const { logger } = require('./utils/logger');

// Function registry
const functions = {
  generateScriptFromIdea,
  parseScriptToSlides,
  generateSlideImages,
  generateVoiceover,
  previewVoice,
  syncVoiceToSlides,
  detectLanguage,
  generateDoodleAsset,
  applyStyleToProject,
  batchProcessProject,
  // Legacy functions (backward compatibility)
  'generate-doodle': generateDoodleAsset,
};

// Lambda handler
exports.handler = async (event, context) => {
  const startTime = Date.now();
  let functionName = null;
  
  try {
    // Handle warm-up pings
    if (event.warmup) {
      logger.info('Warm-up ping received');
      return formatResponse({ message: 'Lambda is warm!' });
    }

    // Parse request body
    const body = typeof event.body === 'string' ? JSON.parse(event.body) : event.body || {};
    
    // Extract function name (support both 'function' and 'action' for backward compatibility)
    functionName = body.function || body.action;
    
    if (!functionName) {
      throw new Error('Function name is required (use "function" or "action" field)');
    }

    // Check if function exists
    const targetFunction = functions[functionName];
    if (!targetFunction) {
      throw new Error(`Unknown function: ${functionName}`);
    }

    // Log function invocation
    logger.info(`Invoking function: ${functionName}`, {
      functionName,
      requestId: context.requestId,
      remainingTime: context.getRemainingTimeInMillis()
    });

    // Validate input based on function schema
    const validationResult = validateInput(functionName, body);
    if (!validationResult.valid) {
      throw new Error(`Validation failed: ${validationResult.errors.join(', ')}`);
    }

    // Execute the function
    const result = await targetFunction(body, context);

    // Calculate execution time
    const executionTime = Date.now() - startTime;

    // Log success
    logger.info(`Function ${functionName} completed`, {
      functionName,
      executionTime,
      requestId: context.requestId
    });

    // Return formatted response
    return formatResponse({
      success: true,
      function: functionName,
      result,
      executionTime,
      requestId: context.requestId
    });

  } catch (error) {
    // Calculate execution time even for errors
    const executionTime = Date.now() - startTime;

    // Log error
    logger.error(`Function ${functionName || 'unknown'} failed`, {
      functionName,
      error: error.message,
      stack: error.stack,
      executionTime,
      requestId: context.requestId
    });

    // Return formatted error response
    return formatError(error, {
      functionName,
      executionTime,
      requestId: context.requestId,
      retryable: isRetryableError(error)
    });
  }
};

// Helper function to determine if error is retryable
function isRetryableError(error) {
  const retryableErrors = [
    'ECONNRESET',
    'ETIMEDOUT',
    'ENOTFOUND',
    'NetworkingError',
    'ServiceUnavailable',
    'ThrottlingException',
    'TooManyRequestsException',
    'RequestLimitExceeded',
    'Rate exceeded'
  ];

  return retryableErrors.some(errType => 
    error.message?.includes(errType) || 
    error.code === errType ||
    error.name === errType
  );
}