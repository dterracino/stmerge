# Multi-Step Merge Pipeline

This document outlines the design for supporting complex, multi-step model merging workflows that go beyond the current single-step merge process.

## Problem Statement

The current manifest system supports merging multiple models in a single operation:

```text
Result = Model1 * 0.25 + Model2 * 0.25 + Model3 * 0.25 + Model4 * 0.25
```

However, users often want more sophisticated merge strategies:

1. **Tournament-style merging**: Merge in stages to avoid dilution
2. **Staged composition**: Base models → Style models → Final blend
3. **Conditional merging**: Different strategies for different model types
4. **Iterative refinement**: Build up complexity gradually

## Proposed Solution: Pipeline Manifest

### Extended Manifest Format

```json
{
  "pipeline": {
    "name": "Complex Anime Merge",
    "description": "Multi-stage merge combining base models, then styles",
    "steps": [
      {
        "id": "base_merge",
        "type": "merge",
        "description": "Combine base reality models",
        "models": [
          {
            "path": "realistic_base_v1.safetensors",
            "weight": 0.6,
            "architecture": "SDXL",
            "precision_detected": "fp16"
          },
          {
            "path": "photorealism_v2.safetensors", 
            "weight": 0.4,
            "architecture": "SDXL",
            "precision_detected": "fp16"
          }
        ],
        "output": {
          "path": "temp/base_merged.safetensors",
          "precision": "fp16"
        },
        "device": "cuda",
        "prune": true
      },
      {
        "id": "style_merge",
        "type": "merge", 
        "description": "Blend anime style models",
        "models": [
          {
            "path": "anime_style_v1.safetensors",
            "weight": 0.7,
            "architecture": "SDXL"
          },
          {
            "path": "manga_style_v2.safetensors",
            "weight": 0.3,
            "architecture": "SDXL"
          }
        ],
        "output": {
          "path": "temp/style_merged.safetensors",
          "precision": "fp16"
        },
        "device": "cuda",
        "prune": true
      },
      {
        "id": "final_composition",
        "type": "merge",
        "description": "Combine base and style merges",
        "dependencies": ["base_merge", "style_merge"],
        "models": [
          {
            "step": "base_merge", 
            "weight": 0.65,
            "description": "Realistic foundation"
          },
          {
            "step": "style_merge",
            "weight": 0.35, 
            "description": "Anime styling"
          }
        ],
        "vae": {
          "path": "anime_vae.safetensors",
          "precision_detected": "fp16"
        },
        "output": {
          "path": "final_anime_realistic.safetensors",
          "precision": "fp16"
        },
        "device": "cuda",
        "prune": true
      }
    ],
    "cleanup": {
      "temp_files": true,
      "keep_intermediates": false
    }
  }
}
```

### Key Features

#### Step Types

- **merge**: Standard weighted model merging
- **validate**: Run inference validation (future)
- **convert**: Format conversion (future)
- **extract**: VAE extraction (future)

#### Dependencies

- Steps can reference outputs from previous steps
- Dependency graph ensures correct execution order
- Enables complex merge strategies

#### Model References

- **File paths**: Direct model files
- **Step references**: Use output from previous steps
- **Mixed**: Combine file models with step outputs

#### Temporary File Management

- Intermediate outputs saved to temp directory
- Optional cleanup of temporary files
- Option to keep intermediates for debugging

## Implementation Architecture

### Pipeline Executor

```python
class PipelineExecutor:
    def __init__(self, manifest_path: Path):
        self.manifest = PipelineManifest.load(manifest_path)
        self.step_outputs = {}  # Cache step results
        
    def execute(self) -> Dict[str, Any]:
        """Execute the entire pipeline."""
        # 1. Validate pipeline (dependency check, file existence)
        # 2. Determine execution order (topological sort)
        # 3. Execute steps in order
        # 4. Cleanup temporary files if requested
        
    def execute_step(self, step: PipelineStep) -> StepResult:
        """Execute a single pipeline step."""
        # Handle different step types
        
    def resolve_model_references(self, step: PipelineStep) -> List[ModelEntry]:
        """Resolve step references to actual model paths."""
        # Convert step references to file paths
```

### Dependency Resolution

```python
def build_dependency_graph(steps: List[PipelineStep]) -> Dict[str, List[str]]:
    """Build dependency graph and detect cycles."""
    
def topological_sort(graph: Dict[str, List[str]]) -> List[str]:
    """Return execution order respecting dependencies."""
```

### Progress Tracking

```python
class PipelineProgress:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.current_step = None
        
    def start_step(self, step_id: str, description: str):
        """Begin executing a step."""
        
    def complete_step(self, step_id: str, output_path: Path, file_size: int):
        """Mark step as completed."""
```

## CLI Integration

### New Command

```bash
# Execute a pipeline
python run.py pipeline --manifest complex_merge.json

# Resume from specific step (if previous run failed)
python run.py pipeline --manifest complex_merge.json --from-step style_merge

# Dry run (validate without executing)
python run.py pipeline --manifest complex_merge.json --dry-run

# Keep intermediate files for debugging
python run.py pipeline --manifest complex_merge.json --keep-temps
```

### Backward Compatibility

- Existing single-step manifests continue to work
- `merge` command unchanged for simple merges
- `pipeline` command for multi-step workflows

## Benefits

### For Users

- ✅ **Complex merge strategies** - Tournament, staged, conditional
- ✅ **Reproducible workflows** - Entire process in one file
- ✅ **Resumable execution** - Skip completed steps
- ✅ **Clear documentation** - Each step describes its purpose

### For Development

- ✅ **Modular design** - Each step type is independent
- ✅ **Extensible** - Easy to add new step types
- ✅ **Testable** - Individual steps can be unit tested
- ✅ **Debuggable** - Intermediate outputs available

## Future Enhancements

### Advanced Step Types

- **block_merge**: Different weights per layer
- **lora_merge**: LoRA integration
- **embedding_merge**: Textual inversion merging

### Conditional Logic

- **if/else steps**: Based on model properties
- **loops**: Iterative refinement
- **parallel**: Execute independent branches simultaneously

### Integration Features

- **validation steps**: Automatic quality checking
- **optimization steps**: Automatic weight tuning
- **analysis steps**: Model comparison and reporting

## Migration Strategy

1. **Phase 1**: Implement basic pipeline executor
2. **Phase 2**: Add dependency resolution and temp file management  
3. **Phase 3**: CLI integration and progress tracking
4. **Phase 4**: Advanced step types and conditional logic

The existing codebase requires minimal changes - the pipeline executor orchestrates existing merge functions in a more sophisticated way.
