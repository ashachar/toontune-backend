/**
 * Input validation utilities
 * Validates function inputs against predefined schemas
 */

const schemas = {
  generateScriptFromIdea: {
    required: ['idea'],
    optional: ['targetLength', 'language', 'tone'],
    types: {
      idea: 'string',
      targetLength: ['short', 'medium', 'long'],
      language: 'string',
      tone: ['professional', 'casual', 'educational']
    }
  },
  
  parseScriptToSlides: {
    required: ['script'],
    optional: ['maxSlidesCount', 'targetDuration'],
    types: {
      script: 'string',
      maxSlidesCount: 'number',
      targetDuration: 'number'
    }
  },
  
  generateSlideImages: {
    required: ['slides'],
    optional: ['projectStyle', 'batchSize'],
    types: {
      slides: 'array',
      projectStyle: 'string',
      batchSize: 'number'
    }
  },
  
  generateVoiceover: {
    required: ['text', 'voiceId'],
    optional: ['language', 'speed', 'pitch'],
    types: {
      text: 'string',
      voiceId: ['delilah', 'zachary', 'ruby', 'douglas', 'sophie', 'bella'],
      language: 'string',
      speed: 'number',
      pitch: 'number'
    }
  },
  
  previewVoice: {
    required: ['voiceId'],
    optional: ['language', 'sampleText'],
    types: {
      voiceId: 'string',
      language: 'string',
      sampleText: 'string'
    }
  },
  
  syncVoiceToSlides: {
    required: ['slides'],
    optional: ['transitionDuration'],
    types: {
      slides: 'array',
      transitionDuration: 'number'
    }
  },
  
  detectLanguage: {
    required: ['text'],
    optional: [],
    types: {
      text: 'string'
    }
  },
  
  generateDoodleAsset: {
    required: ['description'],
    optional: ['category', 'style', 'variations', 'provider', 'count'],
    types: {
      description: 'string',
      category: ['character', 'prop', 'background', 'icon'],
      style: 'string',
      variations: 'number',
      provider: 'string',
      count: 'number'
    }
  },
  
  applyStyleToProject: {
    required: ['projectId', 'style'],
    optional: ['elements'],
    types: {
      projectId: 'string',
      style: ['minimal', 'simple', 'funky', 'romantic', 'scribble', 'halloween'],
      elements: 'array'
    }
  },
  
  batchProcessProject: {
    required: ['projectData'],
    optional: ['options'],
    types: {
      projectData: 'object',
      options: 'object'
    }
  },

  // Legacy function support
  'generate-doodle': {
    required: ['description'],
    optional: ['style', 'count', 'provider'],
    types: {
      description: 'string',
      style: 'string',
      count: 'number',
      provider: 'string'
    }
  }
};

function validateInput(functionName, input) {
  const schema = schemas[functionName];
  
  if (!schema) {
    // If no schema defined, allow all inputs
    return { valid: true, errors: [] };
  }

  const errors = [];

  // Check required fields
  for (const field of schema.required) {
    if (!(field in input) || input[field] === undefined || input[field] === null) {
      errors.push(`Missing required field: ${field}`);
    }
  }

  // Validate field types
  const allFields = [...schema.required, ...schema.optional];
  
  for (const field of allFields) {
    if (field in input && input[field] !== undefined && input[field] !== null) {
      const expectedType = schema.types[field];
      const actualValue = input[field];
      
      if (Array.isArray(expectedType)) {
        // Enum validation
        if (!expectedType.includes(actualValue)) {
          errors.push(`Invalid value for ${field}. Expected one of: ${expectedType.join(', ')}`);
        }
      } else if (expectedType === 'string') {
        if (typeof actualValue !== 'string') {
          errors.push(`Field ${field} must be a string`);
        }
      } else if (expectedType === 'number') {
        if (typeof actualValue !== 'number' || isNaN(actualValue)) {
          errors.push(`Field ${field} must be a number`);
        }
      } else if (expectedType === 'array') {
        if (!Array.isArray(actualValue)) {
          errors.push(`Field ${field} must be an array`);
        }
      } else if (expectedType === 'object') {
        if (typeof actualValue !== 'object' || actualValue === null || Array.isArray(actualValue)) {
          errors.push(`Field ${field} must be an object`);
        }
      }
    }
  }

  // Additional custom validations
  if (functionName === 'generateVoiceover' && input.speed) {
    if (input.speed < 0.5 || input.speed > 2.0) {
      errors.push('Speed must be between 0.5 and 2.0');
    }
  }

  if (functionName === 'generateVoiceover' && input.pitch) {
    if (input.pitch < 0.5 || input.pitch > 2.0) {
      errors.push('Pitch must be between 0.5 and 2.0');
    }
  }

  if (functionName === 'generateSlideImages' && input.batchSize) {
    if (input.batchSize < 1 || input.batchSize > 10) {
      errors.push('Batch size must be between 1 and 10');
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

module.exports = {
  validateInput,
  schemas
};