# Model Merger - New Project TODO List

Items to implement/discuss when starting the new Claude Project.

## Immediate TODOs (v0.6.0 Completion)

### 1. Glob Pattern Support for Scan Command

**Status:** Partially implemented in cli.py, needs manifest.py changes

**What's done:**

- cli.py detects glob patterns (*, ?)
- Expands patterns and filters to .safetensors
- Filters manifest after scanning (workaround)

**What's needed:**

- Update `manifest.py::scan_folder()` to accept optional `file_filter` parameter
- Pass file list directly instead of post-filtering
- Test with various patterns:
  - Simple: `*.safetensors`
  - Filtered: `pony*.safetensors`
  - Recursive: `**/*.safetensors`
  - Multiple wildcards: `*pony*v6*.safetensors`

**Documentation needed:**

- README.md - Brief mention in Quick Start
- docs/usage.md - Full glob documentation with examples
- CHANGELOG.md - Add to Unreleased section

**Files to update:**

- model_merger/manifest.py (scan_folder function)
- cli.py (already updated, may need refinement)
- README.md
- docs/usage.md
- CHANGELOG.md

### 2. Review v0.6.0 Changes

### 3. Fix Notification Library (win10toast Issues)

**Status:** PUNTED to v0.7.0 - too many issues, not worth fixing

**Problems:**

- win10toast crashes on cleanup with WPARAM/WNDPROC error
- Monkey-patch fixes the error but breaks console output after notification
- Last 2 lines after notification don't print (threading issue)
- Library is unmaintained and buggy

**Current workaround:**

- Notifications work but some console output is lost
- User accepted this as acceptable for now

**Solution for v0.7.0:**
Replace win10toast with better library:

- **winotify** - Modern, actively maintained, Windows 10/11
- **plyer** - Cross-platform (Mac/Linux support too!)
- **win11toast** - Windows 11 optimized

**Priority:** Medium (functional but annoying)

### 4. Fix CUDA Memory Leak in Accumulator

**Status:** CRITICAL BUG - Makes CUDA slower than CPU!

**Problem:**
User's 3080 Ti (16GB VRAM) shows memory growing uncontrolled:

```text
First model:   7.02 GB ✅
After model 2: 21.11 GB ❌ (should be ~14GB!)
After model 3: 35.21 GB ❌❌❌ (spills to system RAM!)
```

**Root cause:**

```python
accumulator[key] = accumulator[key] + current_model[key] * weight
```

This creates NEW tensor, doesn't free OLD accumulator tensor!

**Fix:** Use in-place operations

```python
weighted_tensor = current_model[key].to(torch.float32) * entry.weight
accumulator[key].add_(weighted_tensor)  # In-place addition!
del weighted_tensor
```

**Impact:**

- Current: CUDA is SLOWER than CPU (1m 32s vs 49s for 4 models)
- With fix: CUDA should be ~50x faster
- Enables merging 8+ models on 16GB VRAM

**Priority:** CRITICAL - CUDA is currently broken!

### 5. Multi-GPU Device Selection

**Status:** Missing feature

**Problem:**
Currently only supports `--device cuda` (uses default GPU)
Can't select specific GPU for users with multiple cards

**Need:**

```bash
python run.py merge --manifest config.json --device cuda:0
python run.py merge --manifest config.json --device cuda:1
```

**Priority:** Low (most users have 1 GPU)

### 6. Review v0.6.0 Changes

**User has made changes while Claude was unavailable:**

- Model metadata caching system
- CivitAI API integration
- Hash algorithm refactoring (hasher.py module)
- Architecture detection improvements

**Need to:**

- Review all v0.6.0 changes in new project
- Understand caching architecture
- Understand CivitAI integration
- See how these inform v0.7.0 planning

## Future Features Discussion

### v0.7.0 - Cross-Architecture Merging

**High Priority - Exciting Feature!**

**Technique:** Train differencing + comparative interpolation

- Extract deltas: `pony_delta = pony_model - sdxl_base`
- Apply to common base: `pony_on_sdxl = sdxl_base + pony_delta`
- Merge normally: Both now compatible!

**Implementation Plan:**

1. **Experimentation Phase:**
   - Create `experiments/cross_merge_test.py`
   - Test with Pony v6 + Illustrious XL
   - Both are SDXL fine-tunes (known base)
   - Validate output loads and generates images

2. **Integration Phase (if successful):**
   - New modules: `differencer.py`, `cross_merger.py`
   - Enhanced scan with `--cross-merge` flag
   - Base model detection and prompting
   - Auto-download or user-provides base models
   - Extended manifest format

**User's preference:** As automated as possible, single command
**Success criteria:** Merged model loads and works (even if quality needs post-processing)

**Resources:**

- User found other merge scripts with these techniques
- Can reference their implementations
- Community member's workflow description available

**Timeline:** After v0.6.0 complete, potentially bundle/tag/release first

### v0.8.0+ - Model Structure Analysis

#### Long-term - Research Tool

**Concept:** Detect architecture from tensor structure, not metadata

**Why this matters:**

- CivitAI metadata is often wrong (Pony v6 says "Pony" but is actually SDXL)
- Filename detection is unreliable
- Tensor structure is ground truth

**Approach:**

- Create `tools/analyze_model_structure.py`
- Fingerprint known base models
- Compare tensor shapes, key patterns
- Build detection database

**Benefits:**

- Definitive architecture detection
- Pre-merge compatibility checking
- Educational for users
- Foundation for auto-detection

**Implementation:**

- Start as standalone tool (not integrated)
- Build knowledge base through experimentation
- Integrate into scan once proven

## Open Questions

1. **Base Models for Cross-Merge:**
   - Auto-download vs user-provides?
   - Store in ~/.model_merger/base_models/?
   - How to verify correct base model?

2. **Notification Behavior:**
   - Current: Only for operations > 30 seconds
   - Keep this? Or always notify when --notify used?
   - User seems fine with current behavior

3. **Version Numbering:**
   - Stick with v0.6.0, v0.7.0, etc?
   - Or jump to v1.0.0 for first public release?
   - User may tag/release after v0.6.0

4. **Glob Pattern Edge Cases:**
   - What if pattern matches files in multiple directories?
   - Where to save manifest? (currently: first match's parent)
   - Windows path escaping with wildcards?

## Files User Will Upload to New Project

- Complete codebase (all Python modules)
- All docs/ files
- README.md and CHANGELOG.md
- Session notes (this session + previous)
- DOCS_REORG_SUMMARY.md
- Any other development notes

## Context for Next Claude Instance

**Project Status:**

- v0.5.1 released with full /docs structure
- v0.6.0 in progress (caching, CivitAI, glob patterns)
- v0.7.0 planned (cross-architecture merging)
- Documentation complete and professional

**Recent Work:**

- Complete documentation reorganization (README 548→237 lines)
- Created docs/ with 5 comprehensive guides
- FAQ with 48 questions
- CHANGELOG following Keep a Changelog format

**User Preferences:**

- Python, DRY principles, separation of concerns
- Ask clarifying questions before coding
- Summarize before starting implementation
- Wait for approval before changes

**Rate Limit Issue:**

- User pays for Claude but hits limits frequently
- Projects have higher limits - that's why we're creating one
- Claude is "the best coder" but most limited service

---

**NEXT STEPS:**

1. User creates Model Merger project in Claude
2. User uploads codebase + docs + session notes
3. New chat reviews everything
4. Complete glob pattern implementation
5. Plan v0.7.0 cross-merge experiments
