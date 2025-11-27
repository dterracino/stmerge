# Changelog
<!-- markdownlint-disable MD024 -->

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

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
