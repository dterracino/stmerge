# Usage Guide Feature Design Document

## Overview

The Usage Guide feature automatically generates `.usage.json` files that document how to effectively use diffusion models (both merged and single models). These files contain prompting recommendations, trigger words, style notes, and other practical usage information.

## The Problem

When you merge 8 models together, you lose track of important usage information:

- Which prompts work best?
- What trigger words activate specific styles?
- What are the model's strengths and weaknesses?
- How should positive/negative prompts be structured?

**Current state:** This information is scattered across:

- CivitAI model pages (not portable)
- README files (inconsistent, often missing)
- Your own notes (easy to lose)
- Metadata in safetensors (limited, non-standard)

**With Usage Guides:** Portable, machine-readable JSON that travels with the model.

---

## File Format

### Naming Convention

```text
pony_realistic_merge.safetensors
pony_realistic_merge.usage.json     ← Usage guide
```

**Suffix:** `.usage.json` (clear, concise)

### Schema Name

`diffusion-model-usage-guide.schema.json`

**Rationale:**

- "diffusion-model" = covers SD, Flux, and future models
- "usage-guide" = emphasizes practical "how to use" purpose
- Not tied to "Stable Diffusion" (just one company's branding)
- Universal format works for merged AND single models

---

## JSON Schema Structure

### Single Model Example

```json
{
  "$schema": "https://github.com/yourname/model-merger/schemas/diffusion-model-usage-guide.schema.json",
  "schema_version": "1.0",
  
  "model": {
    "name": "Pony Realism v6",
    "file": "pony_realism_v6.safetensors",
    "architecture": "SDXL",
    "source_type": "single"
  },
  
  "prompting": {
    "recommended_positive": ["score_9", "score_8_up", "photorealistic", "detailed"],
    "recommended_negative": ["low quality", "worst quality", "blurry"],
    "trigger_words": [],
    "quality_tags": ["masterpiece", "best quality"],
    "style_notes": "Strong realism, excellent with portraits and landscapes"
  },
  
  "technical": {
    "precision": "fp16",
    "size_gb": 6.5,
    "vae_baked": true,
    "hash_sha256": "abc123..."
  },
  
  "usage_examples": [
    {
      "prompt": "score_9, score_8_up, portrait of a woman, photorealistic, detailed",
      "negative_prompt": "(worst quality:1.4), (low quality:1.4), blurry",
      "description": "High quality realistic portrait"
    }
  ],
  
  "known_issues": [
    "May struggle with complex hand poses",
    "Text rendering can be inconsistent"
  ],
  
  "metadata": {
    "created": "2024-11-27T15:30:00Z",
    "created_by": "user",
    "extraction_method": "manual"
  }
}
```

### Merged Model Example

```json
{
  "$schema": "https://github.com/yourname/model-merger/schemas/diffusion-model-usage-guide.schema.json",
  "schema_version": "1.0",
  
  "model": {
    "name": "Pony 8-Model Realistic Merge",
    "file": "pony_8model_merge.safetensors",
    "architecture": "SDXL",
    "source_type": "merged"
  },
  
  "prompting": {
    "recommended_positive": [
      {
        "text": "score_9, score_8_up",
        "from_models": 3,
        "weighted_importance": 0.45
      },
      {
        "text": "masterpiece, best quality",
        "from_models": 5,
        "weighted_importance": 0.75
      },
      {
        "text": "detailed, high quality",
        "from_models": 6,
        "weighted_importance": 0.60
      }
    ],
    "recommended_negative": [
      {
        "text": "(worst quality:1.4), (low quality:1.4)",
        "from_models": 6,
        "weighted_importance": 0.80
      }
    ],
    "trigger_words": ["anime_style", "realistic_photo"],
    "style_balance": {
      "realism": 0.60,
      "anime": 0.40
    },
    "style_notes": "Balanced merge favoring realism. Works well with mixed prompts."
  },
  
  "merge_info": {
    "created": "2024-11-27T15:30:00Z",
    "tool": "Model Merger v0.6.0",
    "merge_type": "weighted_accumulator",
    "source_models": [
      {
        "name": "Pony Realism v6",
        "file": "pony_realism_v6.safetensors",
        "weight": 0.25,
        "hash_sha256": "abc123...",
        "contributed_prompts": ["score_9", "score_8_up", "photorealistic"],
        "style": "realism"
      },
      {
        "name": "Pony Anime v4",
        "file": "pony_anime_v4.safetensors", 
        "weight": 0.25,
        "hash_sha256": "def456...",
        "contributed_prompts": ["anime style", "vibrant colors"],
        "style": "anime"
      }
      // ... 6 more models
    ]
  },
  
  "usage_examples": [
    {
      "prompt": "score_9, masterpiece, portrait of a woman, photorealistic",
      "negative_prompt": "(worst quality:1.4), (low quality:1.4)",
      "description": "Realistic portrait emphasizing merged realism components"
    },
    {
      "prompt": "anime_style, masterpiece, girl with blue hair, vibrant colors",
      "negative_prompt": "(worst quality:1.4), realistic",
      "description": "Anime-style character using style trigger"
    }
  ],
  
  "known_issues": [
    "Hand anatomy issues inherited from source models",
    "May require both realistic and anime triggers depending on desired output"
  ],
  
  "metadata": {
    "created": "2024-11-27T15:30:00Z",
    "created_by": "Model Merger v0.6.0",
    "extraction_method": "aggregated"
  }
}
```

### Key Differences

**Single models:**

- Simple arrays for prompts
- `source_type: "single"`
- No `merge_info` section

**Merged models:**

- Prompts with frequency/importance data
- `source_type: "merged"`
- Full `merge_info` with source model tracking
- Aggregated/weighted recommendations

---

## Data Sources

### Priority Order

For each model during scan, try these sources in order:

1. **Cache** (instant, free)
   - If already collected, skip

2. **LLM Extraction** (if configured, user consent required)
   - Requires `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in environment
   - Fetches CivitAI page HTML
   - Sends to LLM with extraction prompt
   - Parses JSON response

3. **Manual Input** (always available)
   - Interactive prompts for each field
   - User can skip fields (will be empty/null)

### LLM Extraction Details

**Configuration check:**

```python
def check_llm_available():
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if anthropic_key:
        return True, "anthropic"
    elif openai_key:
        return True, "openai"
    else:
        return False, None
```

**User prompt (only if LLM available and not cached):**

```text
No cached usage info for illustrious_anime.safetensors

LLM extraction available (Anthropic API configured)
Extract usage info from CivitAI using LLM? [y/N]: 
```

**LLM Extraction Prompt Template:**

```python
prompt = f"""
You are analyzing a CivitAI model page to extract usage information.

Model: {model_name}

Page content:
{page_html}

Extract the following information in valid JSON format:

{{
  "recommended_positive_prompts": ["list", "of", "prompts"],
  "recommended_negative_prompts": ["list", "of", "prompts"],
  "trigger_words": ["list", "of", "trigger", "words"],
  "style_notes": "Brief description of style and strengths",
  "quality_tags": ["masterpiece", "best quality", etc],
  "known_issues": ["list", "of", "known", "issues"]
}}

Guidelines:
- Look in the description, creator notes, and example image prompts
- Trigger words are special tokens that activate the model's style
- Quality tags are general prompt enhancers
- Be concise but capture essential information
- If information is not available, use empty arrays or null

CRITICAL: Respond ONLY with valid JSON. No markdown, no explanations.
"""
```

**Manual Input Prompts:**

```text
Enter usage information (press Enter to skip any field):

Recommended positive prompts (comma-separated): score_9, masterpiece, detailed
Recommended negative prompts (comma-separated): low quality, blurry
Trigger words (comma-separated): 
Style notes: Good with portraits and landscapes
Known issues (comma-separated): Struggles with hands, text rendering

✓ Saved usage info
```

---

## Workflow Integration

### During Scan

```bash
python run.py scan ./models --collect-usage-info
```

**For each model:**

1. Check if hash exists in cache with usage info
2. If not cached:
   - If LLM configured: Ask "Extract with LLM? [y/N]"
   - If yes or no LLM: Prompt for manual input
3. Save usage info in cache
4. Include usage info in manifest

**Manifest structure:**

```json
{
  "models": [
    {
      "path": "model1.safetensors",
      "weight": 0.25,
      "usage_info": {
        "recommended_positive": ["score_9", "masterpiece"],
        "recommended_negative": ["low quality"],
        "trigger_words": [],
        "style_notes": "Good with portraits",
        "extraction_method": "llm"
      }
    }
  ]
}
```

### During Merge

```bash
python run.py merge --manifest config.json
```

**After merge completes:**

1. Aggregate usage info from all source models
2. Calculate weighted importance (by model weight)
3. Combine trigger words (deduplicate)
4. Generate style balance based on weights
5. Create comprehensive usage examples
6. Save `.usage.json` alongside merged model

**Output:**

```text
✓ Merge complete: pony_8model_merge.safetensors (6.2 GB)
✓ Usage guide created: pony_8model_merge.usage.json

Review the usage guide for prompting recommendations!
```

---

## Aggregation Algorithm

### Weighted Importance Calculation

```python
def calculate_weighted_importance(prompt_text, source_models):
    """
    Calculate how important a prompt is based on:
    1. How many models use it
    2. The weights of those models
    """
    total_weight = 0.0
    
    for model in source_models:
        if prompt_text in model.usage_info.recommended_positive:
            total_weight += model.weight
    
    # Normalize to 0-1 range
    max_possible_weight = sum(m.weight for m in source_models)
    importance = total_weight / max_possible_weight
    
    return importance

# Example:
# Model A (weight 0.5) uses "masterpiece" → +0.5
# Model B (weight 0.3) uses "masterpiece" → +0.3
# Model C (weight 0.2) doesn't use it → +0
# Importance: 0.8 / 1.0 = 0.8
```

### Style Balance Calculation

```python
def calculate_style_balance(source_models):
    """
    Calculate style distribution based on model weights
    and their reported styles.
    """
    style_weights = {}
    
    for model in source_models:
        style = model.usage_info.get('style', 'unknown')
        if style not in style_weights:
            style_weights[style] = 0.0
        style_weights[style] += model.weight
    
    # Normalize
    total = sum(style_weights.values())
    return {
        style: weight / total 
        for style, weight in style_weights.items()
    }

# Example with 4 models:
# Model A (0.25, "realism")
# Model B (0.25, "realism")
# Model C (0.25, "anime")
# Model D (0.25, "realism")
# Result: {"realism": 0.75, "anime": 0.25}
```

---

## Cache Structure

**Location:** `~/.model_merger/model_cache.json`

**Enhanced with usage info:**

```json
{
  "schema_version": "1.0",
  "models": {
    "abc123hash": {
      "filename": "pony_realism_v6.safetensors",
      "hash_sha256": "abc123...",
      "size_bytes": 6643777536,
      "mtime": 1701234567.89,
      "precision": "fp16",
      "architecture": "SDXL",
      "civitai_metadata": {...},
      "usage_info": {
        "extracted_at": "2024-11-27T15:30:00Z",
        "extraction_method": "llm",
        "llm_provider": "anthropic",
        "data": {
          "recommended_positive": ["score_9", "masterpiece"],
          "recommended_negative": ["low quality"],
          "trigger_words": [],
          "style_notes": "Good with portraits",
          "known_issues": []
        }
      }
    }
  }
}
```

**Extraction methods:**

- `"llm"` - LLM-powered extraction from CivitAI
- `"manual"` - User manually entered
- `"companion_file"` - (future) Parsed from .prompts.txt file
- `"metadata"` - (future) Extracted from model metadata

---

## User Experience

### First-Time Scan (No Cache)

```bash
$ python run.py scan ./models --collect-usage-info

🎨 Model Merger - Scan Mode 🎨

Found 4 models in ./models

LLM extraction available (Anthropic API configured)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model 1/4: pony_realism_v6.safetensors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No cached usage info found.

Extract usage info from CivitAI using LLM? [y/N]: y

  Fetching CivitAI page...
  Extracting with claude-sonnet-4...
  ✓ Extracted usage info

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model 2/4: illustrious_anime.safetensors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No cached usage info found.

Extract usage info from CivitAI using LLM? [y/N]: n

Enter usage information (press Enter to skip):

Recommended positive prompts: anime style, vibrant colors
Recommended negative prompts: realistic, photo
Trigger words: anime_style
Style notes: Anime aesthetic, good with character art
Known issues: 

✓ Saved usage info

[... continues for all models ...]

✓ Generated manifest: ./models/merge_manifest.json
```

### Subsequent Scans (With Cache)

```bash
$ python run.py scan ./models --collect-usage-info

🎨 Model Merger - Scan Mode 🎨

Found 4 models in ./models

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model 1/4: pony_realism_v6.safetensors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Using cached usage info (extracted 2024-11-27)

[... continues for all models ...]

✓ Generated manifest: ./models/merge_manifest.json
```

### After Merge

```bash
$ python run.py merge --manifest ./models/merge_manifest.json

[... merge progress ...]

✓ Merge complete: pony_4model_merge.safetensors (6.2 GB)
✓ Usage guide created: pony_4model_merge.usage.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage Recommendations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Positive Prompts:
  • score_9, score_8_up (75% importance)
  • masterpiece, best quality (100% importance)

Trigger Words: anime_style, realistic_photo

Style Balance: 60% realism, 40% anime

Review pony_4model_merge.usage.json for complete details!
```

---

## Implementation Plan

### Phase 1: Basic Infrastructure (v0.6.0)

**New modules:**

- `model_merger/usage_guide.py` - Core logic
  - `collect_usage_info()` - Gather from various sources
  - `aggregate_usage_info()` - Merge from multiple models
  - `generate_usage_guide()` - Create .usage.json file
  - `extract_with_llm()` - LLM-powered extraction

**Updates:**

- `model_merger/cache.py` - Add usage_info field
- `model_merger/manifest.py` - Include usage_info in ModelEntry
- `cli.py` - Add `--collect-usage-info` flag to scan command

**Files created:**

- `schemas/diffusion-model-usage-guide.schema.json` - JSON schema
- `model.usage.json` - Generated for each merged model

### Phase 2: Enhanced Features (v0.6.1+)

- Companion file support (model.prompts.txt)
- Better LLM prompts (more accurate extraction)
- Usage guide viewer command (`python run.py view-guide model.usage.json`)
- HTML rendering of usage guides for easy reading

---

## Benefits

### For Users

✅ **Never lose prompting info** - Travels with the model
✅ **Know how to use merged models** - Clear recommendations
✅ **Reproducible merges** - Full source model tracking
✅ **Machine-readable** - Tools can parse and display

### For the Ecosystem

✅ **Fills a gap** - No other tool does this
✅ **Community standard potential** - Others could adopt format
✅ **Tool integration ready** - UIs could display usage guides
✅ **Portable documentation** - Works offline, anywhere

---

## Future Enhancements

### v0.7.0+

- Generate usage guides for single models (not just merges)
- Interactive usage guide editor
- Companion file format (model.prompts.txt)
- HTML/Markdown rendering

### v0.8.0+

- Community database of known models
- Automatic updates when models are re-merged
- Usage guide validation tool
- Integration with ComfyUI/A1111 for auto-population

---

## Technical Considerations

### LLM Costs

**Per model extraction:**

- ~50KB HTML page
- ~12,500 input tokens + 500 output tokens
- Cost with Claude Sonnet: ~$0.15 per model
- Cost with GPT-4o-mini: ~$0.01 per model

**With caching:** Extract once, use forever!

**Budget protection:**

- Only extract when explicitly approved by user
- Cache aggressively
- No automatic batch extraction

### Schema Versioning

```json
{
  "schema_version": "1.0"
}
```

Future versions can:

- Add optional fields (backward compatible)
- Deprecate fields gracefully
- Tools validate against schema

### Error Handling

**LLM extraction failures:**

- Fall back to manual input
- Cache failures (don't retry on every scan)
- Clear error messages

**Invalid JSON from LLM:**

- Retry with modified prompt
- Fall back to manual after 2 retries

---

## Open Questions

1. **Should usage guides be generated for single models during scan?**
   - Pro: Universal format, useful for organizing
   - Con: Adds time to scan process
   - Decision: Make it opt-in with separate flag

2. **Should we validate against JSON schema during generation?**
   - Pro: Ensures correctness
   - Con: Adds dependency (jsonschema package)
   - Decision: Validate, add jsonschema to requirements

3. **Should usage guides include example images?**
   - Pro: Visual references very helpful
   - Con: Makes files huge (base64 encoding)
   - Decision: Support via URL references, not embedded

---

## Success Metrics

**v0.6.0 is successful if:**

- ✅ .usage.json files are generated automatically for merges
- ✅ LLM extraction works reliably (>80% success rate)
- ✅ Manual input fallback is smooth
- ✅ Users report it's actually useful (feedback via issues/discussions)

**Long-term success:**

- Other tools adopt the format
- Community builds tools around it
- Becomes standard for sharing diffusion models
