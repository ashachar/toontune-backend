# Corrected V2 Prompt Constraints Summary

## ✅ Critical Updates Applied

### 1. Cartoon Character Frequency: CORRECTED
- **OLD**: Maximum 1 cartoon character every 10-15 seconds
- **NEW**: Maximum 1 cartoon character every 20 seconds (same as key phrases)

### 2. Non-Overlapping Rule: ADDED
- **NEW CRITICAL RULE**: Cartoon characters and key phrases must NEVER appear simultaneously
- Minimum 3 seconds separation required between them
- Example timing:
  - If key phrase appears at 10s-14s
  - Cartoon must appear before 7s OR after 17s
  - This prevents visual clutter and maintains clarity

## Updated Scene Constraints

| Scene | Duration | Max Key Phrases | Max Cartoon Characters | Previous (Wrong) |
|-------|----------|-----------------|------------------------|------------------|
| 1 | 56.7s | 2 | **2** | Was: 5 |
| 2 | 55.1s | 2 | **2** | Was: 5 |
| 3 | 24.3s | 1 | **1** | Was: 2 |
| **Total** | **136.1s** | **5** | **5** | Was: 12 |

## Key Instructions in Prompts

### Cartoon Character Section
```
2. CARTOON CHARACTERS:
   - Maximum 1 cartoon character every 20 seconds (same frequency as key phrases)
   - CRITICAL: NEVER show a cartoon character at the same time as a key phrase
   - They must be temporally separated by at least 3 seconds
```

### Important Guidelines Section
```
IMPORTANT GUIDELINES:
- For a 60-second scene: Maximum 3 key phrases and maximum 3 cartoon characters
- Both key phrases and cartoon characters appear maximum once every 20 seconds
- CRITICAL RULE: Never display a cartoon character and key phrase simultaneously
  * If key phrase at 10s-14s, cartoon must be before 7s or after 17s
  * Maintain at least 3 seconds separation between them
```

### Scene-Specific Constraints
```
CRITICAL CONSTRAINTS FOR THIS SCENE:
- Maximum 2 key phrase(s) total (one every 20 seconds)
- Maximum 2 cartoon character(s) total (one every 20 seconds)
- IMPORTANT: Key phrases and cartoon characters must NEVER appear simultaneously
  * They must be separated by at least 3 seconds
  * Example: If key phrase at 10s-14s, cartoon must be before 7s or after 17s
```

## Rationale

1. **Reduced Frequency**: Cartoon characters now appear at the same conservative rate as key phrases (every 20 seconds), preventing oversaturation

2. **Non-Overlapping**: By ensuring key phrases and cartoons never appear together:
   - Viewers can focus on one element at a time
   - Reduces cognitive load
   - Maintains professional quality
   - Prevents visual clutter

3. **Temporal Separation**: The 3-second buffer ensures smooth transitions between elements

## Verification

All 3 scene prompts have been regenerated with:
- ✅ Cartoon frequency: Every 20 seconds (not 10-15)
- ✅ Non-overlapping rule: Clearly stated multiple times
- ✅ 3-second separation: Explicitly required
- ✅ Reduced totals: 5 cartoons total (was 12)

## Files Updated

1. `utils/video_description_generator_v2.py` - Core prompt template
2. `pipeline/steps/step_4_prompts_v2.py` - Constraint calculations
3. All scene prompts in `uploads/assets/videos/do_re_mi/prompts/`

## Result

The prompts now correctly instruct the LLM to:
- Place cartoon characters sparingly (every 20 seconds max)
- Never show cartoons and key phrases simultaneously
- Maintain clean, uncluttered video composition
- Focus on enhancing, not overwhelming, the content