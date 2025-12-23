# Model Merger - Development Roadmap

This document tracks planned features, design decisions, and implementation notes for future versions.

---

## v0.6.0 - Batch Conversion (PLANNED)

### Feature: Batch Convert Multiple Files

**Goal:** Convert entire folders of checkpoints efficiently with clean, scannable output.

**Problem:** Current single-file output is too verbose for batch operations. Converting 50 files would create 5000+ lines of terminal output, likely overflowing terminal buffers.

**Solution:** Two-mode approach with smart verbosity.

#### Design: Option 2 (Minimal Live Updates + Summary)

**Default batch mode:**

```bash
python run.py convert models/ --batch

Scanning models/... found 50 files

[1/50] model1.ckpt (SD Model, 6.5GB) ━━━━━━━━━━━━━━━━━━━━━ 100% ✓ 45s
[2/50] vae.pt (VAE, 334MB) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% ✓ 5s
[3/50] lora_style.ckpt (LoRA, 144MB) ━━━━━━━━━━━━━━━━━━━━ 100% ✓ 2s
[4/50] embedding.pt (Embed, 12KB) ━━━━━━━━━━━━━━━━━━━━━━━ 100% ✓ <1s
[5/50] corrupted.ckpt ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% ✗ Error
[6/50] upscaler.pth (Upscaler, 67MB) ━━━━━━━━━━━━━━━━━━━━ 100% ✓ 3s
...

✨ Batch Conversion Complete!

Success: 49/50 files converted
  SD Models: 12 (78.4 GB → 39.2 GB)
  VAEs: 5 (1.6 GB → 1.6 GB)
  LoRAs: 28 (4.0 GB → 4.0 GB)
  Embeddings: 3 (36 KB → 36 KB)
  Upscalers: 1 (67 MB → 67 MB)

Failed: 1 file
  corrupted.ckpt: Safe loading failed (see errors.log)

Total time: 8m 23s
Saved to: models_converted/
```

**Verbose mode (for debugging):**

```bash
python run.py convert models/ --batch --verbose
```

- Shows full detailed output like single-file mode
- For debugging or when you want to see everything

#### Implementation Plan

**1. Add CLI arguments:**

- `--batch` flag to convert command
- `input` argument accepts directory (not just file)
- Optional `--output-dir` for where to save converted files (default: input_dir + "_converted")

**2. Add batch detection logic:**

```python
if args.batch or Path(args.input).is_dir():
    # Batch mode
    cmd_convert_batch(args)
else:
    # Single file mode (existing)
    cmd_convert_single(args)
```

**3. Create `cmd_convert_batch()` function:**

- Scan directory for all supported formats (.ckpt, .pt, .pth, .bin)
- Optionally recursive with `--recursive` flag
- Create output directory
- Loop through files with progress tracking

**4. Modify converter output:**

- Add `batch_mode: bool` parameter to `convert_to_safetensors()`
- When `batch_mode=True`:
  - Suppress Rich panels and verbose console output
  - Return stats dict instead of just hash
  - Use simple progress bar instead of detailed messages

**5. Collect statistics:**

```python
stats = {
    'total': 0,
    'success': 0,
    'failed': 0,
    'by_format': {
        'sd_checkpoint': {'count': 0, 'size_before': 0, 'size_after': 0},
        'vae': {'count': 0, 'size_before': 0, 'size_after': 0},
        'lora': {'count': 0, 'size_before': 0, 'size_after': 0},
        'embedding': {'count': 0, 'size_before': 0, 'size_after': 0},
        'upscaler': {'count': 0, 'size_before': 0, 'size_after': 0},
    },
    'errors': [],
}
```

**6. Summary display:**

- Use Rich table for categorized summary
- Show size savings per format type
- Total time and files processed
- Write errors to `conversion_errors_{timestamp}.log`

**7. Error handling:**

- Continue on error (don't stop batch)
- Log detailed error to file
- Show brief error in terminal
- Count towards failed files

#### Files to Modify

- `cli.py` - Add `--batch` flag, add `cmd_convert_batch()`
- `converter.py` - Add `batch_mode` parameter, suppress verbose output when enabled
- `console.py` - Add `print_batch_summary()` for summary table

#### Edge Cases to Handle

- Empty directory
- No supported files found
- All files fail conversion
- Permission errors on output directory
- Disk space issues during batch
- User interruption (Ctrl+C) - save progress so far

#### Testing Checklist

- [ ] Convert directory with mixed formats (SD, VAE, LoRA, embedding, upscaler)
- [ ] Convert directory with some corrupted files
- [ ] Convert with `--verbose` flag
- [ ] Convert with custom `--output-dir`
- [ ] Verify all files converted correctly
- [ ] Check error log format and completeness
- [ ] Test with 50+ files (terminal buffer handling)
- [ ] Test Ctrl+C interruption behavior

#### Estimated Effort

~2-3 hours implementation + 1 hour testing = **3-4 hours total**

---

## Future Ideas (v0.7.0+)

### Block-Weighted Merging

**Description:** Allow different merge weights for different parts of the model (input blocks, middle blocks, output blocks).

**Use case:** Advanced users want fine control over which layers get merged more heavily.

**Implementation complexity:** Medium - need to parse block structure and apply weights per-layer.

**Priority:** Medium - niche feature for power users

---

### Extract VAE from Model

**Description:** Extract the baked VAE from a full SD checkpoint and save it as standalone VAE.

**Use case:** "I love the VAE in this checkpoint, want to use it with other models"

**Implementation complexity:** Low - just extract `first_stage_model.*` keys and save

**Priority:** Low - uncommon use case

---

### Interactive TUI for Manifest Editing

**Description:** Terminal UI for editing manifest files instead of hand-editing JSON.

**Use case:** People who don't like editing JSON by hand

**Implementation complexity:** High - need TUI library (textual?), keyboard navigation, validation

**Priority:** Low - JSON editing works fine, adds complexity

---

### Model Hash Lookup

**Description:** Look up model hashes on CivitAI/HuggingFace to auto-fill metadata.

**Use case:** "What model is this?" identification

**Implementation complexity:** Medium - API integration, rate limiting, caching

**Priority:** Low - nice-to-have, not essential

---

### ✅ Compatibility Check Before Merge (COMPLETED)

**Description:** Check if models are compatible without loading them fully (peek at metadata).

**Use case:** Fail fast if trying to merge incompatible models

**Implementation:** ✅ Completed - Uses shape-only validation with efficient memory usage
- Stores only tensor shapes instead of full reference model
- Validates key compatibility and tensor shape matching
- Significantly reduced memory usage during validation
- In-place accumulation prevents memory leaks

**Status:** Implemented in v0.5.x

---

## v0.8.0 - Model Metadata Cache & CivitAI Integration (IN PROGRESS)

### Feature: Persistent Metadata Cache

**Goal:** Cache model metadata (hash, architecture, precision, CivitAI data) to avoid recomputing hashes and redundant API calls on subsequent operations.

**Problem:**

- Computing SHA-256 hashes takes 5-10 seconds per large model
- CivitAI API lookups add latency and hit rate limits
- Architecture detection from filenames is less accurate than CivitAI's baseModel field
- Repeated operations on same models waste time

**Solution:** JSON-based cache at `~/.model_merger/model_cache.json` with automatic staleness detection.

#### Cache Data Structure

```json
{
  "version": 1,
  "models": {
    "sha256_hash_here": {
      "filename": "pony_model_v1.safetensors",
      "sha256": "abc123...",
      "file_size": 6535231488,
      "last_modified": "2025-11-26T10:30:00Z",
      "precision": "fp16",
      "architecture": "Pony",
      "civitai": {
        "model_id": 12345,
        "version_id": 67890,
        "base_model": "Pony",
        "model_name": "Pony Realistic",
        "version_name": "v1.0",
        "nsfw": false,
        "trained_words": ["realistic", "pony"]
      },
      "cached_at": "2025-11-26T10:30:00Z"
    }
  }
}
```

#### Integration Points

**1. Model Loading (`loader.py` - `load_model()`)**

- Check cache by file path first (validates with size/mtime)
- If valid cache hit: use cached hash, skip recomputation (~5-10s saved)
- If cache miss: compute hash, update cache with hash + detected precision
- **Benefit:** Faster subsequent loads of same models

**2. Manifest Generation (`manifest.py` - `scan_models_folder()`)**

- For each model file, check cache first
- If cached: use cached architecture (prefer CivitAI data over filename detection)
- If not cached: detect from filename, optionally query CivitAI, store in cache
- **Benefit:** More accurate architecture detection, faster scans

**3. CivitAI Lookups (`civitai.py` - `get_model_version_by_hash()`)**

- Check if we have cached CivitAI metadata for this hash
- If cached: return cached data, skip API call
- If not cached: make API call, store response in cache
- **Note:** CivitAI data includes version_id - version changes create new IDs, so cache stays valid
- **Benefit:** Massive speed improvement, avoids API rate limits

**4. Merge Process (`saver.py` - `save_model()`)**

- After saving merged model, add to cache with computed metadata
- Stores merged model's hash, architecture, precision for future use
- **Benefit:** Merged models can be used as inputs to future merges efficiently

**5. Conversion Process (`converter.py` - `convert_to_safetensors()`)**

- Cache the converted model's metadata after conversion
- Preserve CivitAI metadata from source if available
- **Benefit:** Converted models integrate smoothly into cached workflow

#### CLI: New `cache` Subcommand

**Purpose:** Manage cache independently before integrating into core workflows. Allows testing and pre-population.

**Commands:**

```bash
# Show cache statistics (entries, size, location, staleness)
stmerge cache info

# List all cached entries with summary table
stmerge cache list

# Show detailed info for specific cached entry
stmerge cache show <hash_or_filename>

# Add file(s) or folder to cache (auto-detects files vs folders)
stmerge cache add <path> [--civitai] [--recurse] [--force]

# Remove specific entry from cache
stmerge cache remove <hash_or_filename>

# Clear entire cache (with confirmation prompt)
stmerge cache clear [--yes]

# Verify cached entries against actual files, report stale entries
stmerge cache verify [<folder>] [--prune]
```

**Flags:**

- `--civitai` / `--no-civitai`: Enable/disable CivitAI API lookups (default: auto-detect from API key)
- `--cache` / `--no-cache`: Enable/disable cache usage (default: enabled)
- `--recurse`: Recursively process subfolders
- `--force`: Force re-cache even if entry exists (useful for refreshing CivitAI data)
- `--prune`: Automatically remove stale entries during verification
- `--quiet`: Suppress progress output

#### Default Behaviors

**Cache usage:** Enabled by default once integrated

- Use `--no-cache` to bypass cache and work directly with files
- Cache operations are silent/fast, don't require user interaction

**CivitAI lookups:** Enabled by default if API key configured

- Auto-detect from `CIVITAI_API_KEY` environment variable
- Use `--no-civitai` / `--skip-civitai` to explicitly disable
- No API calls if key not configured (graceful degradation)

**Startup display:** Enhanced to show active features

- Current: Shows GPU usage, device selection
- Add: Cache status (enabled/disabled, entry count)
- Add: CivitAI API status (enabled/disabled, key configured)

#### Staleness Detection

**File-based staleness:**

- Compare file size and last modified time
- If either changes, cache entry is invalid
- Automatic on every cache lookup

**CivitAI staleness:**

- Version ID is immutable - new versions get new IDs
- Cache remains valid until file itself changes
- No expiration needed for CivitAI metadata
- Optional: Future feature to check for newer versions based on model_id

#### Implementation Status

**Completed:**

- ✅ `cache.py` module with `CachedModelInfo` and `CivitAIMetadata` dataclasses
- ✅ `ModelCache` class with load/save, query by hash/path
- ✅ Staleness detection (file size + mtime)
- ✅ Atomic writes (temp file + rename) for safety
- ✅ Graceful handling of corrupted cache files
- ✅ Schema versioning for future migrations
- ✅ 21 comprehensive unit tests
- ✅ Cache file path constant in `config.py`
- ✅ Exported cache functions from `__init__.py`

**In Progress:**

- 🔄 CLI `cache` subcommand implementation
- 🔄 Integration into core workflows (loader, manifest, civitai, saver, converter)

**Planned:**

- ⏳ Startup display enhancements
- ⏳ Progress bars for cache scan operations
- ⏳ Rate limiting for batch CivitAI queries
- ⏳ Cache statistics and reporting
- ⏳ Documentation updates (installation, usage, FAQ)

**Priority:** High - foundational feature for performance and UX improvements

---

## Design Principles (Keep These in Mind!)

1. **Security First** - Never compromise on safe loading
2. **Separation of Concerns** - One module = one job
3. **User Experience** - Clear messages, beautiful output
4. **Conservative by Default** - Err on the side of safety
5. **Performance Matters** - Memory-efficient, fast operations
6. **Real-World Testing** - Test with actual models before release

---

## Version History

- **v0.1.0** - Initial release with basic merging
- **v0.2.0** - Added converter + Rich UI
- **v0.3.0** - Smart pruning + DataParallel support
- **v0.4.0** - Deep verification system
- **v0.5.0** - Format detection + adaptive pruning
- **v0.6.0** - (PLANNED) Batch conversion
- **v0.7.0+** - TBD based on user feedback

---

## Community Feedback Notes

*Add notes here based on user reports, feature requests, and issues encountered in the wild.*

**Format:**

- Date: YYYY-MM-DD
- User: (anonymous/username)
- Request/Issue: Description
- Priority: High/Medium/Low
- Status: Planned/Considering/Declined

---

Last Updated: 2025-11-27
