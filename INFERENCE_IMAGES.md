# Inference Image Validation

This document outlines the design for automated visual validation of model merges through inference generation and comparison.

## Problem Statement

Current merge validation is limited to:

- Hash verification (files saved correctly)
- Tensor shape validation (models compatible)
- Size/precision checks (technical correctness)

**Missing**: Visual validation of merge quality and characteristics.

Users currently need to:

1. Complete merge process
2. Load model in external tool (ComfyUI, A1111)
3. Test with various prompts
4. Manually assess quality
5. Return to adjust weights if unsatisfied

This creates a slow feedback loop and makes iterative refinement tedious.

## Proposed Solution: Integrated Visual Validation

### Automatic Inference Testing

Generate test images during or after the merge process to provide immediate visual feedback on merge quality and characteristics.

### Extended Manifest Format

```json
{
  "models": [
    // ... existing model entries
  ],
  "output": {
    // ... existing output config
  },
  "validation": {
    "enabled": true,
    "run_on": ["input_models", "output_model"],  // What to test
    "global_settings": {
      "architecture": "SDXL",  // Auto-detected or user specified
      "seed_mode": "fixed",    // "fixed", "per_prompt", "random"
      "global_seed": 42,       // Used when seed_mode is "fixed"
      "output_dir": "validation_images/",
      "save_grid": true,       // Create comparison grids
      "save_individual": true, // Save individual images
      "device": "cuda",        // Inference device
      "offload_models": true   // Free VRAM between tests
    },
    "prompts": [
      {
        "id": "anime_quality_test",
        "description": "Test anime style generation quality",
        "prompt": "1girl, anime style, detailed face, vibrant colors, high quality",
        "negative_prompt": "blurry, low quality, bad anatomy, deformed",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "cfg_scale": 7.0,
        "scheduler": "DPM++ 2M Karras",
        "seed": null,            // Use global seed
        "num_images": 1,
        "clip_skip": 2,
        "enable_hr": false
      },
      {
        "id": "realistic_portrait_test", 
        "description": "Test realistic portrait generation",
        "prompt": "portrait of a woman, photorealistic, studio lighting, detailed skin",
        "negative_prompt": "cartoon, anime, illustration, painting, blurry",
        "width": 768,
        "height": 1024,
        "steps": 25,
        "cfg_scale": 6.0,
        "scheduler": "Euler a",
        "seed": 12345,           // Override global seed
        "num_images": 2,         // Generate multiple variants
        "enable_hr": true,       // High-res fix
        "hr_scale": 1.5,
        "hr_steps": 10
      },
      {
        "id": "style_consistency_test",
        "description": "Test style consistency across subjects",
        "prompt": "anime girl, detailed face, school uniform, outdoors",
        "negative_prompt": "blurry, low quality",
        "width": 512,
        "height": 768,
        "steps": 20,
        "cfg_scale": 7.5,
        "scheduler": "DPM++ 2M Karras",
        "num_images": 3,         // Multiple samples for consistency check
        "clip_skip": 2
      }
    ],
    "comparison": {
      "enabled": true,
      "create_grids": true,
      "grid_layout": "side_by_side",  // "side_by_side", "grid", "before_after"
      "include_metadata": true,       // Prompt, settings overlay
      "quality_metrics": false        // Future: automated quality scoring
    }
  }
}\n```

## Implementation Architecture

### Core Components

#### 1. Inference Engine
```python\nclass InferenceEngine:\n    \"\"\"Handles model loading and inference generation.\"\"\"\n    \n    def __init__(self, device: str = \"cuda\"):\n        self.device = device\n        self.current_pipeline = None\n        \n    def load_model(self, model_path: Path, architecture: str) -> None:\n        \"\"\"Load model into inference pipeline.\"\"\"\n        \n    def generate_image(self, prompt_config: PromptConfig) -> Image:\n        \"\"\"Generate single image from prompt configuration.\"\"\"\n        \n    def generate_batch(self, prompt_configs: List[PromptConfig]) -> List[ValidationResult]:\n        \"\"\"Generate multiple images efficiently.\"\"\"\n        \n    def unload_model(self) -> None:\n        \"\"\"Free GPU memory.\"\"\"\n```\n\n#### 2. Validation Orchestrator\n```python\nclass ValidationOrchestrator:\n    \"\"\"Coordinates validation across multiple models and prompts.\"\"\"\n    \n    def validate_merge(self, manifest: MergeManifest) -> ValidationReport:\n        \"\"\"Run complete validation suite.\"\"\"\n        # 1. Test input models (if requested)\n        # 2. Test output model \n        # 3. Generate comparison grids\n        # 4. Create validation report\n        \n    def test_model(self, model_path: Path, prompts: List[PromptConfig]) -> List[ValidationResult]:\n        \"\"\"Test single model with all prompts.\"\"\"\n        \n    def create_comparison_grid(self, results: Dict[str, List[ValidationResult]]) -> Image:\n        \"\"\"Create side-by-side comparison grids.\"\"\"\n```\n\n#### 3. Architecture Detection Integration\n```python\ndef get_inference_architecture(detected_arch: str) -> Tuple[str, type]:\n    \"\"\"Map detected architecture to diffusers pipeline.\"\"\"\n    \n    mapping = {\n        'SDXL': ('stabilityai/stable-diffusion-xl-base-1.0', StableDiffusionXLPipeline),\n        'Pony': ('stabilityai/stable-diffusion-xl-base-1.0', StableDiffusionXLPipeline), \n        'Illustrious': ('stabilityai/stable-diffusion-xl-base-1.0', StableDiffusionXLPipeline),\n        'SD1.5': ('runwayml/stable-diffusion-v1-5', StableDiffusionPipeline),\n        'SD2.1': ('stabilityai/stable-diffusion-2-1', StableDiffusionPipeline)\n    }\n    \n    return mapping.get(detected_arch, ('stabilityai/stable-diffusion-xl-base-1.0', StableDiffusionXLPipeline))\n```\n\n### Memory Management\n\n#### Efficient Model Loading\n```python\ndef load_for_inference(model_path: Path, architecture: str, device: str) -> Pipeline:\n    \"\"\"Load model efficiently for inference.\"\"\"\n    \n    # Use from_single_file for direct safetensors loading\n    if architecture in ['SDXL', 'Pony', 'Illustrious']:\n        pipeline = StableDiffusionXLPipeline.from_single_file(\n            str(model_path),\n            torch_dtype=torch.float16,\n            use_safetensors=True,\n            device_map=\"auto\" if device == \"cuda\" else None\n        )\n    else:\n        pipeline = StableDiffusionPipeline.from_single_file(\n            str(model_path),\n            torch_dtype=torch.float16, \n            use_safetensors=True\n        )\n        \n    pipeline = pipeline.to(device)\n    \n    # Enable memory efficient attention\n    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):\n        pipeline.enable_xformers_memory_efficient_attention()\n        \n    return pipeline\n```\n\n#### GPU Memory Optimization\n```python\nclass MemoryManager:\n    \"\"\"Manage GPU memory during validation.\"\"\"\n    \n    def __init__(self, device: str):\n        self.device = device\n        \n    def prepare_for_model_load(self) -> None:\n        \"\"\"Clear memory before loading new model.\"\"\"\n        if self.device == \"cuda\":\n            torch.cuda.empty_cache()\n            torch.cuda.synchronize()\n            \n    def optimize_pipeline(self, pipeline) -> None:\n        \"\"\"Apply memory optimizations.\"\"\"\n        # Enable attention slicing for lower VRAM usage\n        pipeline.enable_attention_slicing()\n        \n        # Enable CPU offload for components not actively in use\n        pipeline.enable_model_cpu_offload()\n```\n\n### Scheduler Support\n\n```python\nSUPPORTED_SCHEDULERS = {\n    \"DPM++ 2M Karras\": (\n        DPMSolverMultistepScheduler,\n        {\"use_karras_sigmas\": True, \"algorithm_type\": \"dpmsolver++\"}\n    ),\n    \"Euler a\": (\n        EulerAncestralDiscreteScheduler,\n        {}\n    ),\n    \"Euler\": (\n        EulerDiscreteScheduler,\n        {}\n    ),\n    \"DDIM\": (\n        DDIMScheduler,\n        {}\n    ),\n    \"LMS\": (\n        LMSDiscreteScheduler,\n        {}\n    )\n}\n\ndef apply_scheduler(pipeline, scheduler_name: str) -> None:\n    \"\"\"Apply specified scheduler to pipeline.\"\"\"\n    if scheduler_name in SUPPORTED_SCHEDULERS:\n        scheduler_class, config = SUPPORTED_SCHEDULERS[scheduler_name]\n        pipeline.scheduler = scheduler_class.from_config(\n            pipeline.scheduler.config, **config\n        )\n```\n\n## Workflow Integration\n\n### During Merge Process\n```python\n# In merger.py after merging is complete\ndef merge_models_with_validation(manifest: MergeManifest) -> Dict[str, Any]:\n    \"\"\"Extended merge process with validation.\"\"\"\n    \n    # 1. Perform standard merge\n    merged_dict = merge_models(\n        model_entries=manifest.models,\n        device=manifest.device\n    )\n    \n    # 2. Save merged model\n    output_path = save_model(merged_dict, manifest.output.path)\n    \n    # 3. Run validation if enabled\n    validation_report = None\n    if manifest.validation and manifest.validation.enabled:\n        print_section(\"Running Visual Validation\")\n        \n        validator = ValidationOrchestrator()\n        validation_report = validator.validate_merge(manifest)\n        \n        print_success(f\"Validation complete: {len(validation_report.results)} images generated\")\n        print_info(f\"Images saved to: {manifest.validation.global_settings.output_dir}\")\n    \n    return {\n        \"merged_model\": output_path,\n        \"validation_report\": validation_report\n    }\n```\n\n### CLI Integration\n\n```bash\n# Standard merge with validation\npython run.py merge --manifest config.json\n\n# Skip validation (faster merge)\npython run.py merge --manifest config.json --no-validation\n\n# Validation only (test existing model)\npython run.py validate --model merged_model.safetensors --prompts validation_prompts.json\n\n# Quick validation (single prompt)\npython run.py validate --model model.safetensors --prompt \"1girl, anime style\" --steps 20\n```\n\n## Output Organization\n\n### Directory Structure\n```\nvalidation_images/\n├── merge_20231202_143022/           # Timestamped validation run\n│   ├── input_models/\n│   │   ├── model1_anime_test.png\n│   │   ├── model1_realistic_test.png\n│   │   ├── model2_anime_test.png\n│   │   └── model2_realistic_test.png\n│   ├── output_model/\n│   │   ├── merged_anime_test.png\n│   │   ├── merged_realistic_test.png\n│   │   └── merged_style_test_001.png\n│   ├── comparisons/\n│   │   ├── anime_test_comparison.png    # Side-by-side grid\n│   │   └── realistic_test_comparison.png\n│   ├── validation_report.json          # Detailed results\n│   └── manifest_used.json             # Copy of manifest for reference\n```\n\n### Validation Report\n```json\n{\n  \"validation_id\": \"merge_20231202_143022\",\n  \"timestamp\": \"2023-12-02T14:30:22Z\",\n  \"manifest_path\": \"merge_config.json\",\n  \"models_tested\": [\n    {\n      \"model_path\": \"model1.safetensors\",\n      \"architecture\": \"SDXL\",\n      \"type\": \"input\"\n    },\n    {\n      \"model_path\": \"merged_output.safetensors\", \n      \"architecture\": \"SDXL\",\n      \"type\": \"output\"\n    }\n  ],\n  \"prompts_executed\": 6,\n  \"images_generated\": 12,\n  \"total_time_seconds\": 45.2,\n  \"device_used\": \"cuda\",\n  \"results\": [\n    {\n      \"prompt_id\": \"anime_test\",\n      \"model_type\": \"output\",\n      \"image_path\": \"output_model/merged_anime_test.png\",\n      \"generation_time\": 3.2,\n      \"seed_used\": 42,\n      \"settings\": {\n        \"steps\": 20,\n        \"cfg_scale\": 7.0,\n        \"scheduler\": \"DPM++ 2M Karras\"\n      }\n    }\n  ]\n}\n```\n\n## Benefits\n\n### Immediate Feedback\n- ✅ **Visual results instantly** - No external tools needed\n- ✅ **Side-by-side comparison** - See merge effects clearly  \n- ✅ **Multiple test scenarios** - Anime, realistic, style consistency\n- ✅ **Reproducible results** - Fixed seeds ensure consistency\n\n### Iterative Workflow\n- ✅ **Quick weight adjustment** - See impact immediately\n- ✅ **Style balance tuning** - Visual feedback on anime vs realistic blend\n- ✅ **Quality validation** - Catch broken merges before saving\n- ✅ **Experimentation** - Test ideas rapidly\n\n### Documentation\n- ✅ **Visual merge records** - Generated images serve as documentation\n- ✅ **Comparison archives** - Track merge evolution over time\n- ✅ **Shareable results** - Easy to share merge outcomes\n\n## Dependencies\n\nAdditional packages needed:\n```\ndiffusers>=0.21.0     # Inference pipelines\ntransformers>=4.21.0  # Text encoders\naccelerate>=0.20.0    # Memory optimization\nxformers>=0.0.16      # Memory efficient attention (optional)\nPillow>=9.0.0         # Image processing\n```\n\n## Future Enhancements\n\n### Advanced Validation\n- **Quality metrics**: Automated scoring (CLIP similarity, aesthetic scores)\n- **Style analysis**: Quantify anime vs realistic characteristics  \n- **Prompt optimization**: Auto-suggest merged prompts based on input models\n- **A/B testing**: Generate variants with slight weight differences\n\n### Interactive Features\n- **Live preview**: Real-time validation during weight adjustment\n- **Web interface**: Browser-based validation review\n- **Batch validation**: Test multiple weight combinations\n- **Model comparison**: Compare against reference models\n\n### Integration\n- **Multi-step pipeline validation**: Test each pipeline step\n- **LoRA compatibility**: Test with popular LoRAs\n- **Style transfer validation**: Measure style preservation\n\n## Migration Strategy\n\n1. **Phase 1**: Basic inference integration (single model, single prompt)\n2. **Phase 2**: Multi-prompt validation and comparison grids\n3. **Phase 3**: Memory optimization and scheduler support\n4. **Phase 4**: Advanced features (quality metrics, batch testing)\n\nThe validation system is completely optional - existing merge workflows continue unchanged while power users get sophisticated visual feedback.
