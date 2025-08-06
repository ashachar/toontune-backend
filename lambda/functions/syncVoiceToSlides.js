/**
 * Sync Voice To Slides Function
 * Calculates timing and synchronization data for slides with voiceover
 */

const { logger } = require('../utils/logger');

const DEFAULT_TRANSITION_DURATION = 0.5; // seconds

async function syncVoiceToSlides(input, context) {
  const {
    slides,
    transitionDuration = DEFAULT_TRANSITION_DURATION
  } = input;

  logger.info('Syncing voice to slides', {
    slideCount: slides.length,
    transitionDuration
  });

  try {
    const timeline = [];
    let currentTime = 0;

    for (let i = 0; i < slides.length; i++) {
      const slide = slides[i];
      const { text, audioUrl, duration } = slide;

      // Validate slide data
      if (!duration || duration <= 0) {
        throw new Error(`Invalid duration for slide ${i + 1}: ${duration}`);
      }

      // Calculate slide timing
      const slideTimeline = {
        slideNumber: i + 1,
        startTime: currentTime,
        endTime: currentTime + duration,
        audioDuration: duration,
        transitionStart: currentTime + duration - transitionDuration,
        text: text?.substring(0, 50) + '...', // Include preview of text
        audioUrl
      };

      // Add transition buffer for next slide (except last slide)
      if (i < slides.length - 1) {
        slideTimeline.transitionEnd = currentTime + duration + transitionDuration;
      }

      timeline.push(slideTimeline);
      
      // Update current time for next slide
      currentTime += duration;
      
      // Add transition gap between slides (except after last slide)
      if (i < slides.length - 1) {
        currentTime += transitionDuration;
      }
    }

    // Calculate total duration
    const totalDuration = currentTime;

    // Generate synchronization metadata
    const syncMetadata = {
      totalSlides: slides.length,
      totalDuration,
      averageSlideDuration: totalDuration / slides.length,
      transitionDuration,
      fps: 30, // Standard frame rate for video
      totalFrames: Math.round(totalDuration * 30)
    };

    // Generate keyframes for video editing
    const keyframes = generateKeyframes(timeline, syncMetadata.fps);

    logger.info('Voice-to-slides sync completed', {
      totalDuration,
      slideCount: slides.length
    });

    return {
      timeline,
      totalDuration,
      metadata: syncMetadata,
      keyframes
    };

  } catch (error) {
    logger.error('Failed to sync voice to slides', {
      error: error.message
    });
    throw error;
  }
}

function generateKeyframes(timeline, fps) {
  const keyframes = [];

  timeline.forEach(slide => {
    // Slide entry keyframe
    keyframes.push({
      type: 'slide_in',
      slideNumber: slide.slideNumber,
      time: slide.startTime,
      frame: Math.round(slide.startTime * fps),
      action: 'show_slide'
    });

    // Audio start keyframe
    keyframes.push({
      type: 'audio_start',
      slideNumber: slide.slideNumber,
      time: slide.startTime,
      frame: Math.round(slide.startTime * fps),
      action: 'play_audio'
    });

    // Transition start keyframe (if applicable)
    if (slide.transitionStart && slide.transitionEnd) {
      keyframes.push({
        type: 'transition_start',
        slideNumber: slide.slideNumber,
        time: slide.transitionStart,
        frame: Math.round(slide.transitionStart * fps),
        action: 'begin_transition'
      });
    }

    // Slide exit keyframe
    keyframes.push({
      type: 'slide_out',
      slideNumber: slide.slideNumber,
      time: slide.endTime,
      frame: Math.round(slide.endTime * fps),
      action: 'hide_slide'
    });

    // Audio end keyframe
    keyframes.push({
      type: 'audio_end',
      slideNumber: slide.slideNumber,
      time: slide.endTime,
      frame: Math.round(slide.endTime * fps),
      action: 'stop_audio'
    });
  });

  // Sort keyframes by time
  keyframes.sort((a, b) => a.time - b.time);

  return keyframes;
}

module.exports = {
  syncVoiceToSlides
};