/**
 * AWS Secrets Manager Integration
 * Secure management of API keys and sensitive configuration
 */

const AWS = require('aws-sdk');
const { logger } = require('./logger');

const secretsManager = new AWS.SecretsManager({
  region: process.env.AWS_REGION || 'us-east-1'
});

// Cache for secrets to avoid repeated API calls
const secretsCache = new Map();
const CACHE_TTL = 3600000; // 1 hour

/**
 * Get secret from AWS Secrets Manager
 */
async function getSecret(secretName) {
  // Check cache first
  const cached = secretsCache.get(secretName);
  if (cached && cached.expiresAt > Date.now()) {
    logger.debug('Returning cached secret', { secretName });
    return cached.value;
  }

  try {
    logger.info('Fetching secret from Secrets Manager', { secretName });
    
    const data = await secretsManager.getSecretValue({ SecretId: secretName }).promise();
    
    let secret;
    if (data.SecretString) {
      secret = JSON.parse(data.SecretString);
    } else {
      // Binary secret
      const buff = Buffer.from(data.SecretBinary, 'base64');
      secret = buff.toString('ascii');
    }

    // Cache the secret
    secretsCache.set(secretName, {
      value: secret,
      expiresAt: Date.now() + CACHE_TTL
    });

    return secret;
  } catch (error) {
    logger.error('Failed to retrieve secret', { 
      secretName, 
      error: error.message 
    });
    
    // Fall back to environment variable if secret not found
    if (error.code === 'ResourceNotFoundException') {
      const envValue = process.env[secretName.toUpperCase().replace(/-/g, '_')];
      if (envValue) {
        logger.warn('Using environment variable fallback', { secretName });
        return envValue;
      }
    }
    
    throw error;
  }
}

/**
 * Get all API keys and credentials
 */
async function getAPICredentials() {
  try {
    const credentials = await getSecret('toontune-api-credentials');
    
    return {
      openai: credentials.OPENAI_API_KEY || process.env.OPENAI_API_KEY,
      replicate: credentials.REPLICATE_API_KEY || process.env.REPLICATE_API_KEY,
      gemini: credentials.GEMINI_API_KEY || process.env.GEMINI_API_KEY,
      elevenlabs: credentials.ELEVENLABS_API_KEY || process.env.ELEVENLABS_API_KEY,
      stability: credentials.STABILITY_API_KEY || process.env.STABILITY_API_KEY
    };
  } catch (error) {
    logger.warn('Failed to get API credentials from Secrets Manager, using env vars', {
      error: error.message
    });
    
    // Fallback to environment variables
    return {
      openai: process.env.OPENAI_API_KEY,
      replicate: process.env.REPLICATE_API_KEY,
      gemini: process.env.GEMINI_API_KEY,
      elevenlabs: process.env.ELEVENLABS_API_KEY,
      stability: process.env.STABILITY_API_KEY
    };
  }
}

/**
 * Get database credentials
 */
async function getDatabaseCredentials() {
  try {
    const credentials = await getSecret('toontune-database-credentials');
    
    return {
      endpoint: credentials.RDS_ENDPOINT || process.env.RDS_ENDPOINT,
      database: credentials.RDS_DATABASE || process.env.RDS_DATABASE,
      username: credentials.RDS_USERNAME || process.env.RDS_USERNAME,
      password: credentials.RDS_PASSWORD || process.env.RDS_PASSWORD
    };
  } catch (error) {
    logger.warn('Failed to get database credentials from Secrets Manager', {
      error: error.message
    });
    
    return {
      endpoint: process.env.RDS_ENDPOINT,
      database: process.env.RDS_DATABASE,
      username: process.env.RDS_USERNAME,
      password: process.env.RDS_PASSWORD
    };
  }
}

/**
 * Get API keys for authentication
 */
async function getAPIKeys() {
  try {
    const apiKeys = await getSecret('toontune-api-keys');
    return apiKeys;
  } catch (error) {
    logger.warn('Failed to get API keys from Secrets Manager', {
      error: error.message
    });
    
    // Fallback to environment variable
    if (process.env.API_KEYS) {
      return JSON.parse(process.env.API_KEYS);
    }
    
    // Default keys for development
    return {
      development: 'dev-key-12345',
      production: process.env.PRODUCTION_API_KEY
    };
  }
}

/**
 * Store a new secret in Secrets Manager
 */
async function createSecret(secretName, secretValue, description) {
  try {
    const params = {
      Name: secretName,
      Description: description || `Secret for ${secretName}`,
      SecretString: typeof secretValue === 'string' 
        ? secretValue 
        : JSON.stringify(secretValue)
    };

    const result = await secretsManager.createSecret(params).promise();
    
    logger.info('Secret created successfully', { 
      secretName, 
      arn: result.ARN 
    });
    
    return result;
  } catch (error) {
    if (error.code === 'ResourceExistsException') {
      logger.info('Secret already exists, updating instead', { secretName });
      return updateSecret(secretName, secretValue);
    }
    
    logger.error('Failed to create secret', { 
      secretName, 
      error: error.message 
    });
    throw error;
  }
}

/**
 * Update an existing secret
 */
async function updateSecret(secretName, secretValue) {
  try {
    const params = {
      SecretId: secretName,
      SecretString: typeof secretValue === 'string' 
        ? secretValue 
        : JSON.stringify(secretValue)
    };

    const result = await secretsManager.updateSecret(params).promise();
    
    // Clear cache for this secret
    secretsCache.delete(secretName);
    
    logger.info('Secret updated successfully', { 
      secretName, 
      arn: result.ARN 
    });
    
    return result;
  } catch (error) {
    logger.error('Failed to update secret', { 
      secretName, 
      error: error.message 
    });
    throw error;
  }
}

/**
 * Rotate API keys
 */
async function rotateAPIKey(service) {
  try {
    const credentials = await getAPICredentials();
    const newKey = generateAPIKey();
    
    // Update the specific service key
    credentials[service] = newKey;
    
    // Save updated credentials
    await updateSecret('toontune-api-credentials', credentials);
    
    logger.info('API key rotated successfully', { service });
    
    return newKey;
  } catch (error) {
    logger.error('Failed to rotate API key', { 
      service, 
      error: error.message 
    });
    throw error;
  }
}

/**
 * Generate a new API key
 */
function generateAPIKey() {
  const crypto = require('crypto');
  return crypto.randomBytes(32).toString('hex');
}

/**
 * Clear secrets cache
 */
function clearCache() {
  secretsCache.clear();
  logger.info('Secrets cache cleared');
}

/**
 * Initialize secrets on Lambda cold start
 */
async function initializeSecrets() {
  try {
    // Pre-load critical secrets
    await Promise.all([
      getAPICredentials(),
      getAPIKeys()
    ]);
    
    logger.info('Secrets initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize secrets', { 
      error: error.message 
    });
    // Don't throw - allow Lambda to start with env vars
  }
}

module.exports = {
  getSecret,
  getAPICredentials,
  getDatabaseCredentials,
  getAPIKeys,
  createSecret,
  updateSecret,
  rotateAPIKey,
  generateAPIKey,
  clearCache,
  initializeSecrets
};