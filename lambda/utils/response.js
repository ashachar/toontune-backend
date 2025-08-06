/**
 * Response formatter utilities
 * Ensures consistent response format across all Lambda functions
 */

function formatResponse(data, statusCode = 200) {
  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,X-Api-Key,Authorization',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    },
    body: JSON.stringify(data)
  };
}

function formatError(error, metadata = {}) {
  const statusCode = getStatusCodeFromError(error);
  
  const errorResponse = {
    error: {
      code: error.code || 'INTERNAL_ERROR',
      message: error.message || 'An unexpected error occurred',
      details: process.env.NODE_ENV === 'development' ? {
        stack: error.stack,
        ...metadata
      } : undefined,
      retryable: metadata.retryable || false,
      timestamp: new Date().toISOString(),
      requestId: metadata.requestId
    }
  };

  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Content-Type,X-Api-Key,Authorization',
      'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
    },
    body: JSON.stringify(errorResponse)
  };
}

function getStatusCodeFromError(error) {
  const statusCodes = {
    'ValidationError': 400,
    'BadRequest': 400,
    'Unauthorized': 401,
    'Forbidden': 403,
    'NotFound': 404,
    'Conflict': 409,
    'TooManyRequests': 429,
    'ThrottlingException': 429,
    'ServiceUnavailable': 503,
    'GatewayTimeout': 504
  };

  // Check error name or code
  if (statusCodes[error.name]) {
    return statusCodes[error.name];
  }
  if (statusCodes[error.code]) {
    return statusCodes[error.code];
  }

  // Check for specific error messages
  if (error.message?.includes('validation') || error.message?.includes('invalid')) {
    return 400;
  }
  if (error.message?.includes('not found')) {
    return 404;
  }
  if (error.message?.includes('rate limit') || error.message?.includes('throttle')) {
    return 429;
  }

  // Default to 500 for unknown errors
  return 500;
}

module.exports = {
  formatResponse,
  formatError
};