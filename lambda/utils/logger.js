/**
 * Logger utility for Lambda functions
 * Provides structured logging with appropriate log levels
 */

class Logger {
  constructor() {
    this.logLevel = process.env.LOG_LEVEL || 'info';
    this.levels = {
      debug: 0,
      info: 1,
      warn: 2,
      error: 3
    };
  }

  shouldLog(level) {
    return this.levels[level] >= this.levels[this.logLevel];
  }

  formatLog(level, message, data = {}) {
    return JSON.stringify({
      timestamp: new Date().toISOString(),
      level: level.toUpperCase(),
      message,
      ...data,
      environment: process.env.NODE_ENV || 'development'
    });
  }

  debug(message, data) {
    if (this.shouldLog('debug')) {
      console.debug(this.formatLog('debug', message, data));
    }
  }

  info(message, data) {
    if (this.shouldLog('info')) {
      console.info(this.formatLog('info', message, data));
    }
  }

  warn(message, data) {
    if (this.shouldLog('warn')) {
      console.warn(this.formatLog('warn', message, data));
    }
  }

  error(message, data) {
    if (this.shouldLog('error')) {
      // Mask sensitive data
      const sanitizedData = this.sanitizeData(data);
      console.error(this.formatLog('error', message, sanitizedData));
    }
  }

  sanitizeData(data) {
    const sensitive = ['password', 'apiKey', 'token', 'secret', 'authorization'];
    const sanitized = { ...data };

    Object.keys(sanitized).forEach(key => {
      if (sensitive.some(s => key.toLowerCase().includes(s.toLowerCase()))) {
        sanitized[key] = '***REDACTED***';
      }
    });

    return sanitized;
  }
}

module.exports = {
  logger: new Logger()
};