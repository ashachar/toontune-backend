# Claude Code Subagents Guide

This directory contains specialized Claude Code subagents following the official markdown-based format.

## 📁 Structure

```
.claude/agents/
├── video-encoder.md         # Video encoding specialist
├── animation-optimizer.md   # 3D animation optimization
├── utils/                   # Utility scripts for subagents
│   └── video_encoder_utility.py
└── SUBAGENTS_GUIDE.md      # This file
```

## 🤖 Available Subagents

### 1. video-encoder
**Purpose**: Fix video encoding issues and ensure QuickTime compatibility

**When Used**:
- Video files that won't open
- "Could not be opened" errors
- Need for batch video encoding
- Format compatibility issues

**Key Features**:
- Automatic QuickTime compatibility
- Fallback encoding strategies
- Batch processing support
- Quality presets (standard/high/compatible)

**Example Usage**:
```
> The video won't open in QuickTime
[video-encoder subagent will be invoked automatically]
```

### 2. animation-optimizer
**Purpose**: Optimize 3D text animations for quality and compatibility

**When Used**:
- After creating any 3D text animation
- Testing animations with different fonts
- Ensuring optimal text positioning
- Final animation output preparation

**Key Features**:
- Font robustness testing
- Optimal position finding
- High-quality encoding
- Compatibility verification

## 🚀 How Subagents Work

1. **Markdown Definition**: Each subagent is a `.md` file with YAML frontmatter
2. **Automatic Invocation**: Claude recognizes when to use them based on context
3. **Tool Access**: Each subagent has specific tools it can use
4. **Separate Context**: Subagents work in their own context window

## 📝 Subagent File Format

```markdown
---
name: subagent-name
description: When this subagent should be used
tools: Tool1, Tool2, Tool3  # Optional - inherits all if omitted
---

System prompt and instructions for the subagent...
```

## 🛠️ Utility Scripts

The `utils/` directory contains helper scripts that subagents can call:

### video_encoder_utility.py
```bash
# Encode single video
python .claude/agents/utils/video_encoder_utility.py input.mp4

# High quality
python .claude/agents/utils/video_encoder_utility.py input.mp4 -q high

# Batch encode
python .claude/agents/utils/video_encoder_utility.py --batch directory/

# Verify encoding
python .claude/agents/utils/video_encoder_utility.py video.mp4 --verify
```

## 💡 Creating New Subagents

1. **Use the `/agents` command** (recommended):
   ```
   /agents
   ```
   Then select "Create New Agent"

2. **Or create manually**:
   - Create a new `.md` file in `.claude/agents/`
   - Add YAML frontmatter with name, description, and optional tools
   - Write detailed system prompt

## 🎯 Best Practices

1. **Focused Purpose**: Each subagent should have a single, clear responsibility
2. **Detailed Instructions**: Include specific steps and examples in the prompt
3. **Tool Limitation**: Only grant necessary tools for security and focus
4. **Proactive Triggers**: Use "PROACTIVELY" and "MUST BE USED" in descriptions
5. **Clear Success Criteria**: Define what successful completion looks like

## 📊 Subagent Priority

When names conflict:
- Project subagents (`.claude/agents/`) override
- User subagents (`~/.claude/agents/`) have lower priority

## 🔄 Integration Example

Instead of manually running FFmpeg commands:
```bash
# Old way (manual)
ffmpeg -i video.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4

# New way (with subagent)
> Please ensure this video is QuickTime compatible
[video-encoder subagent handles it automatically]
```

## 📚 Related Documentation

- [Official Subagents Docs](/ai-docs/claude-subagents.md)
- [Available Tools](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude)
- [Slash Commands](https://docs.anthropic.com/en/docs/claude-code/slash-commands)

## ✅ Verification

To verify subagents are working:

1. Check they're listed:
   ```
   /agents
   ```

2. Test explicit invocation:
   ```
   > Use the video-encoder subagent to fix this video
   ```

3. Test automatic invocation:
   ```
   > This video won't open in QuickTime
   ```

## 🚧 Future Subagents

Planned additions:
- `test-runner`: Automated test execution and fixing
- `code-reviewer`: Comprehensive code quality checks
- `performance-optimizer`: Code performance improvements
- `security-scanner`: Security vulnerability detection