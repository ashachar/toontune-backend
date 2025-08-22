/**
 * Apply Style To Project Function
 * Applies consistent styling to all project assets
 */

const { logger } = require('../utils/logger');

// Style configurations
const STYLE_CONFIGS = {
  minimal: {
    imageModifiers: [
      'minimalist line drawing',
      'single continuous line',
      'very simple shapes',
      'maximum white space'
    ],
    colorPalette: ['#000000', '#FFFFFF'],
    animationPresets: ['fade', 'slide'],
    transitionStyle: 'clean',
    strokeWidth: 1,
    complexity: 'very-low'
  },
  simple: {
    imageModifiers: [
      'simple line art',
      'clean lines',
      'basic shapes',
      'clear composition'
    ],
    colorPalette: ['#000000', '#FFFFFF', '#808080'],
    animationPresets: ['fade', 'slide', 'zoom'],
    transitionStyle: 'smooth',
    strokeWidth: 2,
    complexity: 'low'
  },
  funky: {
    imageModifiers: [
      'playful doodle style',
      'quirky characters',
      'fun elements',
      'dynamic composition'
    ],
    colorPalette: ['#000000', '#FFFFFF', '#FF6B6B', '#4ECDC4'],
    animationPresets: ['bounce', 'spin', 'wiggle', 'pop'],
    transitionStyle: 'playful',
    strokeWidth: 3,
    complexity: 'medium'
  },
  romantic: {
    imageModifiers: [
      'elegant line art',
      'flowing curves',
      'decorative elements',
      'soft composition'
    ],
    colorPalette: ['#000000', '#FFFFFF', '#FFB6C1', '#DDA0DD'],
    animationPresets: ['fade', 'float', 'sway'],
    transitionStyle: 'elegant',
    strokeWidth: 2,
    complexity: 'medium'
  },
  scribble: {
    imageModifiers: [
      'rough sketch style',
      'hand-drawn scribbles',
      'artistic strokes',
      'organic lines'
    ],
    colorPalette: ['#000000', '#FFFFFF', '#333333'],
    animationPresets: ['shake', 'sketch', 'draw'],
    transitionStyle: 'rough',
    strokeWidth: 'variable',
    complexity: 'high'
  },
  halloween: {
    imageModifiers: [
      'spooky doodle style',
      'gothic elements',
      'halloween themed',
      'dark atmosphere'
    ],
    colorPalette: ['#000000', '#FFFFFF', '#FF6600', '#800080'],
    animationPresets: ['creep', 'float', 'flicker', 'spook'],
    transitionStyle: 'eerie',
    strokeWidth: 3,
    complexity: 'medium-high'
  }
};

// Animation configurations
const ANIMATION_CONFIGS = {
  fade: { duration: 0.5, easing: 'ease-in-out' },
  slide: { duration: 0.3, easing: 'ease-out' },
  zoom: { duration: 0.4, easing: 'ease-in-out' },
  bounce: { duration: 0.6, easing: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)' },
  spin: { duration: 0.5, easing: 'ease-in-out' },
  wiggle: { duration: 0.4, easing: 'ease-in-out' },
  pop: { duration: 0.3, easing: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)' },
  float: { duration: 2.0, easing: 'ease-in-out' },
  sway: { duration: 3.0, easing: 'ease-in-out' },
  shake: { duration: 0.2, easing: 'linear' },
  sketch: { duration: 1.0, easing: 'ease-out' },
  draw: { duration: 1.5, easing: 'ease-in-out' },
  creep: { duration: 2.0, easing: 'ease-in' },
  flicker: { duration: 0.1, easing: 'steps(2)' },
  spook: { duration: 1.0, easing: 'ease-in-out' }
};

async function applyStyleToProject(input, context) {
  const {
    projectId,
    style,
    elements = ['images', 'transitions', 'animations']
  } = input;

  logger.info('Applying style to project', {
    projectId,
    style,
    elements
  });

  try {
    // Get style configuration
    const styleConfig = STYLE_CONFIGS[style];
    
    if (!styleConfig) {
      throw new Error(`Unknown style: ${style}`);
    }

    // Build comprehensive style configuration
    const fullConfig = {
      styleConfig: {
        ...styleConfig,
        // Add animation details
        animationDetails: styleConfig.animationPresets.map(preset => ({
          name: preset,
          ...ANIMATION_CONFIGS[preset]
        }))
      },
      applied: true,
      appliedTo: elements,
      projectId,
      timestamp: new Date().toISOString()
    };

    // Add element-specific configurations
    if (elements.includes('images')) {
      fullConfig.imageConfig = {
        modifiers: styleConfig.imageModifiers,
        defaultPromptSuffix: styleConfig.imageModifiers.join(', '),
        strokeWidth: styleConfig.strokeWidth,
        complexity: styleConfig.complexity
      };
    }

    if (elements.includes('transitions')) {
      fullConfig.transitionConfig = {
        style: styleConfig.transitionStyle,
        defaultDuration: getTransitionDuration(styleConfig.transitionStyle),
        easing: getTransitionEasing(styleConfig.transitionStyle)
      };
    }

    if (elements.includes('animations')) {
      fullConfig.animationConfig = {
        presets: styleConfig.animationPresets,
        details: fullConfig.styleConfig.animationDetails,
        defaultAnimation: styleConfig.animationPresets[0]
      };
    }

    // Add CSS variables for web implementation
    fullConfig.cssVariables = generateCSSVariables(styleConfig);

    // Add export presets
    fullConfig.exportPresets = {
      videoCodec: 'h264',
      videoBitrate: '5000k',
      audioCodec: 'aac',
      audioBitrate: '128k',
      resolution: '1920x1080',
      frameRate: 30
    };

    logger.info('Style configuration generated', {
      projectId,
      style,
      elementsConfigured: elements.length
    });

    return fullConfig;

  } catch (error) {
    logger.error('Failed to apply style to project', {
      error: error.message,
      projectId,
      style
    });
    throw error;
  }
}

function getTransitionDuration(style) {
  const durations = {
    clean: 0.3,
    smooth: 0.5,
    playful: 0.6,
    elegant: 0.8,
    rough: 0.4,
    eerie: 1.0
  };
  return durations[style] || 0.5;
}

function getTransitionEasing(style) {
  const easings = {
    clean: 'ease-in-out',
    smooth: 'ease-in-out',
    playful: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
    elegant: 'cubic-bezier(0.4, 0.0, 0.2, 1)',
    rough: 'steps(4)',
    eerie: 'cubic-bezier(0.4, 0.0, 0.6, 1)'
  };
  return easings[style] || 'ease-in-out';
}

function generateCSSVariables(styleConfig) {
  const cssVars = {};
  
  // Color variables
  styleConfig.colorPalette.forEach((color, index) => {
    cssVars[`--color-${index}`] = color;
  });
  
  // Stroke width
  if (typeof styleConfig.strokeWidth === 'number') {
    cssVars['--stroke-width'] = `${styleConfig.strokeWidth}px`;
  } else {
    cssVars['--stroke-width'] = '2px';
  }
  
  // Animation variables
  cssVars['--transition-style'] = styleConfig.transitionStyle;
  cssVars['--default-animation'] = styleConfig.animationPresets[0];
  
  return cssVars;
}

module.exports = {
  applyStyleToProject
};