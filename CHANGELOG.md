# Changelog
<!-- markdownlint-disable MD024 -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In Progress

- Model metadata caching system with automatic staleness detection
- CLI `cache` subcommand for cache management operations
- Integration of cache into core workflows (loader, manifest, merge, convert)

### Added

- **CivitAI API Integration**
  - `civitai.py` module for querying CivitAI API by model hash
  - `get_model_version_by_hash()` - Fetch model metadata from CivitAI
  - `detect_architecture_from_civitai()` - Automatic architecture detection from CivitAI baseModel field
  - `get_model_metadata_summary()` - Simplified model metadata with model_id, version_id, architecture, etc.
  - Environment variable support for `CIVITAI_API_KEY` configuration
  - Automatic fallback to filename detection when CivitAI lookup fails
  - `.env.example` template for API key configuration
  - `example_civitai_api.py` demonstrating API integration workflow

- **Model Metadata Cache System**
  - `cache.py` module with persistent JSON-based caching at `~/.model_merger/model_cache.json`
  - `CachedModelInfo` dataclass storing filename, hash, size, mtime, precision, architecture
  - `CivitAIMetadata` dataclass storing model_id, version_id, base_model, nsfw, trained_words
  - `ModelCache` class with load/save, query by hash/path, automatic staleness detection
  - Staleness detection based on file size and modification time
  - Atomic writes (temp file + rename) for cache safety
  - Graceful handling of corrupted cache files
  - Schema versioning for future migrations
  - Convenience functions: `get_cached_metadata()`, `update_cache()`, `invalidate_cache()`

- **Hash Algorithm Support**
  - Refactored hashing into dedicated `hasher.py` module
  - Multi-algorithm support: SHA-256, MD5, CRC32, Blake3 (optional)
  - CivitAI algorithm stubs: AutoV1, AutoV2 (to be implemented)
  - `compute_file_hash()` with algorithm selection
  - Convenience functions: `compute_sha256()`, `compute_md5()`, `compute_crc32()`, `compute_blake3()`
  - `verify_file_hash()` for hash verification
  - Progress bars for all hashing operations
  - Chunked reading to avoid loading entire files into memory

- **Architecture Detection Improvements**
  - `detect_architecture_from_filename_strict()` removed (technical debt cleanup)
  - Smarter fallback chain: CivitAI baseModel → model name → tags → filename → None
  - Fixed Illustrious architecture detection order (check before generic SDXL)
  - Architecture detection now prefers explicit pattern matches over defaults

### Changed

- **Hashing Logic Refactored**
  - Moved `compute_file_hash()` from `loader.py` to `hasher.py`
  - `loader.py` now imports hashing functions from `hasher` module
  - All hashing operations centralized in one module for easier maintenance

- **CivitAI Integration**
  - Model version_id included in cached metadata for future version checking
  - CivitAI data cached to avoid redundant API calls
  - API lookups respect environment variable configuration

- **Example Scripts**
  - `example_civitai_api.py` optimized to hash file only once (performance improvement)

### Fixed

- Architecture detection returning default (SDXL) when checking model names/tags
- Test failures in `test_civitai.py` for tag detection and fallback behavior
- Import errors in test files (requests module not imported)
- Mock patch paths in cache tests (corrected to `console.print_warning`)

### Testing

- 33 comprehensive unit tests for `civitai.py` module
- 21 unit tests for `cache.py` module covering load/save/query/staleness
- 20 unit tests for `hasher.py` module with multiple algorithm support
- All tests passing with proper mocking and isolation

### Planned

- **Usage Guide Generation (.usage.json)** (v0.6.0)
  - Automatic generation of `.usage.json` files for merged models
  - Aggregates prompting recommendations from source models
  - Optional LLM-powered extraction from CivitAI pages (if API key configured)
  - Manual input fallback for usage information
  
- **CUDA Memory Optimization** (v0.6.0)
  - Fix memory leak in accumulator (use in-place operations with .add_())
  - Optimize tensor preparation (only clone shared tensors, not all 2515)
  
- **LoRA Merging Support** (v0.8.0+)
  - Merge LoRAs into models (bake LoRA weights permanently)
  - Support for applying multiple LoRAs with different strengths
  
- Glob pattern support for scan command (e.g., `scan ./models/*.safetensors`)
- Multi-GPU device selection (`--device cuda:0`, `--device cuda:1`)
- Replace win10toast with better notification library (winotify or plyer)
- Batch conversion (convert entire folders at once)
- Block-weighted merging (different weights per layer)
- Model info command (inspect models without merging)

## [0.5.1] - 2024-11-26

### Added

- Desktop notifications for long-running operations (Windows toast notifications)
- `--notify` flag for convert and merge commands
- `--force` flag for convert to skip extension validation prompts
- Extension validation for convert operations (warns on unrecognized file types)
- CUDA availability detection with helpful installation instructions
- Auto-fallback to CPU when CUDA requested but unavailable
- Device information display (shows CUDA availability and selected device)
- User-customizable architecture patterns via `~/.model_merger/architecture_patterns.json`
- External `architecture_patterns.json` file for easy customization

### Changed

- Architecture patterns moved from hardcoded to JSON configuration
- Improved CUDA documentation with all supported versions (11.8, 12.1, 12.4, 12.8, 13.0)
- Clarified that `--vae` flag is optional in documentation
- Updated requirements documentation to show all dependencies with versions

### Fixed

- Type hint for `format_size()` to accept `Union[int, float]`
- Hardcoded version string in merge command header (now uses `__version__`)

## [0.5.0] - 2024-11-25

### Added

- Smart format detection for SD models, VAEs, LoRAs, embeddings, and upscalers
- Adaptive pruning strategies based on detected file format
- Dedicated `pruner.py` module for format detection and pruning logic
- Automatic handling of standalone files (skip pruning for non-SD-checkpoint formats)
- Upscaler detection (ESRGAN, Real-ESRGAN, etc.)

### Changed

- Pruning now adapts to file type instead of one-size-fits-all approach
- Refactored pruning logic into separate module for better separation of concerns

## [0.4.0] - 2024-11-24

### Added

- Deep verification system (`verify` command)
- Tensor-by-tensor comparison with floating-point tolerance
- Key set validation in verification
- Module prefix handling in verification
- `--verbose` flag for detailed verification output
- New `verifier.py` module

### Changed

- Verification now uses same extraction logic as converter for consistency

## [0.3.0] - 2024-11-23

### Added

- DataParallel `module.` prefix removal for multi-GPU trained models
- Shared tensor detection and cloning
- Output verification for converted files
- Numpy fallback for stubborn shared tensors
- Support for textual inversion embeddings
- Support for standalone VAE files
- Support for LoRA files

### Changed

- Enhanced converter robustness with better format detection
- Improved error handling for edge cases

## [0.2.0] - 2024-11-22

### Added

- Convert command to convert .ckpt/.pt/.pth/.bin files to safetensors
- Progress bars with elapsed time tracking
- Beautiful Rich-powered CLI output
- CLI override persistence to manifest
- Skip-errors mode for scanning
- `--version` flag

### Changed

- Migrated from tqdm to Rich for all progress bars and formatting

## [0.1.0] - 2024-11-21

### Added

- Initial release
- Multi-model merging with configurable weights
- Accumulator pattern for memory-efficient merging
- VAE baking support
- Manifest workflow (scan → edit → merge)
- Architecture detection from filenames
- Precision detection and conversion
- SHA-256 hash computation for verification
- Basic CLI with scan and merge commands
- safetensors output format
- Security-first design (no code execution from pickles)

[Unreleased]: https://github.com/yourusername/model-merger/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/yourusername/model-merger/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/yourusername/model-merger/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/yourusername/model-merger/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/yourusername/model-merger/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourusername/model-merger/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/model-merger/releases/tag/v0.1.0
