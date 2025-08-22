/**
 * Authentication and Authorization Utilities
 * Handles API key validation and request signing
 */

const crypto = require('crypto');
const { logger } = require('./logger');

// In production, these would come from AWS Secrets Manager or Parameter Store
const API_KEYS = process.env.API_KEYS ? JSON.parse(process.env.API_KEYS) : {
  'development': 'dev-key-12345',
  'production': process.env.PRODUCTION_API_KEY
};

// Rate limiting configuration
const RATE_LIMITS = {
  'development': { requests: 1000, window: 3600 }, // 1000 requests per hour
  'production': { requests: 10000, window: 3600 },  // 10000 requests per hour
  'premium': { requests: 100000, window: 3600 }     // 100000 requests per hour
};

// In-memory rate limiting store (in production, use Redis or DynamoDB)
const rateLimitStore = new Map();

/**
 * Validate API key from request
 */
function validateApiKey(headers) {
  const apiKey = headers['x-api-key'] || headers['X-Api-Key'];
  
  if (!apiKey) {
    return {
      valid: false,
      error: 'API key is required'
    };
  }

  // Find the API key in our store
  const keyEntry = Object.entries(API_KEYS).find(([tier, key]) => key === apiKey);
  
  if (!keyEntry) {
    logger.warn('Invalid API key attempted', { 
      apiKey: apiKey.substring(0, 10) + '...' 
    });
    return {
      valid: false,
      error: 'Invalid API key'
    };
  }

  const [tier, key] = keyEntry;
  
  return {
    valid: true,
    tier,
    apiKey: key
  };
}

/**
 * Verify HMAC signature for request integrity
 */
function verifyRequestSignature(headers, body, secret) {
  const signature = headers['x-signature'] || headers['X-Signature'];
  const timestamp = headers['x-timestamp'] || headers['X-Timestamp'];
  
  if (!signature || !timestamp) {
    return {
      valid: false,
      error: 'Request signature and timestamp required'
    };
  }

  // Check timestamp to prevent replay attacks (5 minute window)
  const now = Date.now();
  const requestTime = parseInt(timestamp);
  const timeDiff = Math.abs(now - requestTime);
  
  if (timeDiff > 300000) { // 5 minutes
    return {
      valid: false,
      error: 'Request timestamp expired'
    };
  }

  // Calculate expected signature
  const payload = `${timestamp}.${JSON.stringify(body)}`;
  const expectedSignature = crypto
    .createHmac('sha256', secret || process.env.SIGNING_SECRET || 'default-secret')
    .update(payload)
    .digest('hex');

  // Compare signatures
  const signaturesMatch = crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expectedSignature)
  );

  if (!signaturesMatch) {
    logger.warn('Invalid request signature', {
      timestamp,
      signatureProvided: signature.substring(0, 10) + '...'
    });
    return {
      valid: false,
      error: 'Invalid request signature'
    };
  }

  return {
    valid: true
  };
}

/**
 * Check rate limits for API key
 */
function checkRateLimit(apiKey, tier = 'development') {
  const limits = RATE_LIMITS[tier] || RATE_LIMITS.development;
  const now = Date.now();
  const windowStart = now - (limits.window * 1000);
  
  // Get or create rate limit entry
  const key = `${apiKey}:${tier}`;
  let entry = rateLimitStore.get(key);
  
  if (!entry) {
    entry = {
      requests: [],
      tier
    };
    rateLimitStore.set(key, entry);
  }

  // Clean old requests outside window
  entry.requests = entry.requests.filter(timestamp => timestamp > windowStart);
  
  // Check if limit exceeded
  if (entry.requests.length >= limits.requests) {
    const resetTime = entry.requests[0] + (limits.window * 1000);
    return {
      allowed: false,
      limit: limits.requests,
      remaining: 0,
      resetAt: new Date(resetTime).toISOString(),
      retryAfter: Math.ceil((resetTime - now) / 1000)
    };
  }

  // Add current request
  entry.requests.push(now);
  
  return {
    allowed: true,
    limit: limits.requests,
    remaining: limits.requests - entry.requests.length,
    resetAt: new Date(now + (limits.window * 1000)).toISOString()
  };
}

/**
 * Middleware to validate authentication
 */
async function authenticate(event) {
  const headers = event.headers || {};
  const body = typeof event.body === 'string' ? JSON.parse(event.body) : event.body;

  // Skip auth for warm-up pings
  if (event.warmup) {
    return { authenticated: true, tier: 'system' };
  }

  // Validate API key
  const apiKeyResult = validateApiKey(headers);
  if (!apiKeyResult.valid) {
    return {
      authenticated: false,
      error: apiKeyResult.error,
      statusCode: 401
    };
  }

  // Check rate limits
  const rateLimitResult = checkRateLimit(apiKeyResult.apiKey, apiKeyResult.tier);
  if (!rateLimitResult.allowed) {
    return {
      authenticated: false,
      error: 'Rate limit exceeded',
      statusCode: 429,
      headers: {
        'X-RateLimit-Limit': String(rateLimitResult.limit),
        'X-RateLimit-Remaining': String(rateLimitResult.remaining),
        'X-RateLimit-Reset': rateLimitResult.resetAt,
        'Retry-After': String(rateLimitResult.retryAfter)
      }
    };
  }

  // Verify request signature if enabled
  if (process.env.REQUIRE_SIGNATURE === 'true') {
    const signatureResult = verifyRequestSignature(headers, body, apiKeyResult.apiKey);
    if (!signatureResult.valid) {
      return {
        authenticated: false,
        error: signatureResult.error,
        statusCode: 401
      };
    }
  }

  // Authentication successful
  return {
    authenticated: true,
    tier: apiKeyResult.tier,
    apiKey: apiKeyResult.apiKey,
    rateLimit: rateLimitResult
  };
}

/**
 * Generate signature for client-side signing
 */
function generateSignature(body, secret) {
  const timestamp = Date.now();
  const payload = `${timestamp}.${JSON.stringify(body)}`;
  const signature = crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');

  return {
    signature,
    timestamp,
    headers: {
      'X-Signature': signature,
      'X-Timestamp': String(timestamp)
    }
  };
}

/**
 * Clean up old rate limit entries periodically
 */
function cleanupRateLimits() {
  const now = Date.now();
  const maxAge = 7200000; // 2 hours
  
  for (const [key, entry] of rateLimitStore.entries()) {
    const oldestRequest = entry.requests[0];
    if (!oldestRequest || now - oldestRequest > maxAge) {
      rateLimitStore.delete(key);
    }
  }
}

// Clean up every 30 minutes
setInterval(cleanupRateLimits, 1800000);

module.exports = {
  validateApiKey,
  verifyRequestSignature,
  checkRateLimit,
  authenticate,
  generateSignature,
  cleanupRateLimits
};