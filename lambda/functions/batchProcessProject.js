/**
 * Batch Process Project Function
 * Orchestrates complete project generation with multiple Lambda calls
 */

const { logger } = require('../utils/logger');
const { generateScriptFromIdea } = require('./generateScriptFromIdea');
const { parseScriptToSlides } = require('./parseScriptToSlides');
const { generateSlideImages } = require('./generateSlideImages');
const { generateVoiceover } = require('./generateVoiceover');
const { syncVoiceToSlides } = require('./syncVoiceToSlides');
const { applyStyleToProject } = require('./applyStyleToProject');
const { detectLanguage } = require('./detectLanguage');
const crypto = require('crypto');

async function batchProcessProject(input, context) {
  const {
    projectData,
    options = {}
  } = input;

  const {
    script,
    idea,
    style = 'simple',
    voiceId = 'delilah',
    language = 'en'
  } = projectData;

  const {
    generateImages = true,
    generateVoiceover: generateVoice = true,
    autoSync = true,
    maxSlides = 10
  } = options;

  // Generate project ID
  const projectId = crypto.randomBytes(16).toString('hex');

  logger.info('Starting batch project processing', {
    projectId,
    hasScript: !!script,
    hasIdea: !!idea,
    style,
    voiceId,
    language,
    options
  });

  const results = {
    projectId,
    status: 'processing',
    results: {},
    errors: [],
    processingTime: 0
  };

  const startTime = Date.now();

  try {
    // Step 1: Generate script if only idea provided
    let finalScript = script;
    if (!script && idea) {
      logger.info('Generating script from idea', { projectId });
      try {
        const scriptResult = await generateScriptFromIdea({
          idea,
          targetLength: 'medium',
          language,
          tone: 'educational'
        }, context);
        
        finalScript = scriptResult.script;
        results.results.scriptGeneration = scriptResult;
      } catch (error) {
        logger.error('Failed to generate script', { projectId, error: error.message });
        results.errors.push({
          step: 'scriptGeneration',
          error: error.message
        });
        throw error; // Critical failure
      }
    }

    if (!finalScript) {
      throw new Error('No script available for processing');
    }

    // Step 2: Detect language if auto
    let detectedLanguage = language;
    if (language === 'auto') {
      logger.info('Detecting language', { projectId });
      try {
        const langResult = await detectLanguage({ text: finalScript }, context);
        detectedLanguage = langResult.language;
        results.results.languageDetection = langResult;
      } catch (error) {
        logger.warn('Language detection failed, using English', { projectId });
        detectedLanguage = 'en';
      }
    }

    // Step 3: Parse script into slides
    logger.info('Parsing script to slides', { projectId });
    let slides;
    try {
      const slideResult = await parseScriptToSlides({
        script: finalScript,
        maxSlidesCount: maxSlides,
        targetDuration: 5
      }, context);
      
      slides = slideResult.slides;
      results.results.slides = slideResult;
    } catch (error) {
      logger.error('Failed to parse slides', { projectId, error: error.message });
      results.errors.push({
        step: 'parseSlides',
        error: error.message
      });
      throw error; // Critical failure
    }

    // Step 4: Apply style configuration
    logger.info('Applying style configuration', { projectId });
    try {
      const styleResult = await applyStyleToProject({
        projectId,
        style,
        elements: ['images', 'transitions', 'animations']
      }, context);
      
      results.results.styleConfig = styleResult;
    } catch (error) {
      logger.warn('Failed to apply style', { projectId, error: error.message });
      results.errors.push({
        step: 'applyStyle',
        error: error.message
      });
      // Non-critical, continue
    }

    // Step 5: Generate images (parallel with voiceover)
    const parallelTasks = [];

    if (generateImages) {
      logger.info('Queuing image generation', { projectId });
      parallelTasks.push(
        generateSlideImages({
          slides: slides.map(s => ({
            ...s,
            projectId
          })),
          projectStyle: style,
          batchSize: 4
        }, context).then(result => {
          results.results.images = result;
        }).catch(error => {
          logger.error('Failed to generate images', { projectId, error: error.message });
          results.errors.push({
            step: 'generateImages',
            error: error.message
          });
        })
      );
    }

    // Step 6: Generate voiceovers (parallel with images)
    let voiceovers = [];
    if (generateVoice) {
      logger.info('Queuing voiceover generation', { projectId });
      
      // Generate voiceover for each slide
      const voicePromises = slides.map(async (slide, index) => {
        try {
          const voResult = await generateVoiceover({
            text: slide.text,
            voiceId,
            language: detectedLanguage,
            speed: 1.0,
            pitch: 1.0
          }, context);
          
          return {
            slideNumber: slide.slideNumber,
            ...voResult
          };
        } catch (error) {
          logger.error(`Failed to generate voiceover for slide ${index + 1}`, {
            projectId,
            error: error.message
          });
          return {
            slideNumber: slide.slideNumber,
            error: error.message,
            status: 'failed'
          };
        }
      });

      parallelTasks.push(
        Promise.all(voicePromises).then(voResults => {
          voiceovers = voResults;
          results.results.voiceovers = voResults;
        })
      );
    }

    // Wait for parallel tasks
    await Promise.all(parallelTasks);

    // Step 7: Sync voice to slides if both slides and voiceovers exist
    if (autoSync && voiceovers.length > 0) {
      logger.info('Syncing voice to slides', { projectId });
      try {
        const slidesWithAudio = slides.map((slide, index) => {
          const vo = voiceovers[index];
          return {
            ...slide,
            audioUrl: vo?.audioUrl,
            duration: vo?.duration || slide.duration
          };
        });

        const syncResult = await syncVoiceToSlides({
          slides: slidesWithAudio,
          transitionDuration: 0.5
        }, context);
        
        results.results.timeline = syncResult;
      } catch (error) {
        logger.error('Failed to sync voice to slides', { projectId, error: error.message });
        results.errors.push({
          step: 'syncVoiceToSlides',
          error: error.message
        });
      }
    }

    // Calculate final status
    const hasErrors = results.errors.length > 0;
    const hasCriticalErrors = results.errors.some(e => 
      ['scriptGeneration', 'parseSlides'].includes(e.step)
    );

    if (hasCriticalErrors) {
      results.status = 'failed';
    } else if (hasErrors) {
      results.status = 'partial';
    } else {
      results.status = 'completed';
    }

    // Calculate processing time
    results.processingTime = Date.now() - startTime;

    // Add summary
    results.summary = {
      projectId,
      status: results.status,
      slidesGenerated: slides?.length || 0,
      imagesGenerated: results.results.images?.statistics?.successful || 0,
      voiceoversGenerated: voiceovers.filter(v => !v.error).length,
      totalDuration: results.results.timeline?.totalDuration || 0,
      processingTimeMs: results.processingTime,
      errors: results.errors.length
    };

    logger.info('Batch project processing completed', {
      projectId,
      status: results.status,
      processingTime: results.processingTime,
      summary: results.summary
    });

    return results;

  } catch (error) {
    logger.error('Critical failure in batch project processing', {
      projectId,
      error: error.message
    });

    results.status = 'failed';
    results.errors.push({
      step: 'critical',
      error: error.message
    });
    results.processingTime = Date.now() - startTime;

    return results;
  }
}

module.exports = {
  batchProcessProject
};