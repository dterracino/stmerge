"""
Tests for manifest.py module.

Tests manifest dataclasses, serialization, scanning, validation,
and output filename generation.
"""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_merger import manifest
from model_merger.manifest import (
    ModelEntry, VAEEntry, OutputEntry, MergeManifest,
    generate_output_filename, scan_folder, validate_manifest
)
from tests.helpers import create_dummy_model, create_dummy_vae, cleanup_test_files


class TestModelEntry(unittest.TestCase):
    """Tests for ModelEntry dataclass."""
    
    def test_model_entry_creation(self):
        """Test creating a ModelEntry."""
        entry = ModelEntry(
            path="/path/to/model.safetensors",
            weight=0.5,
            architecture="SDXL"
        )
        
        self.assertEqual(entry.path, "/path/to/model.safetensors")
        self.assertEqual(entry.weight, 0.5)
        self.assertEqual(entry.architecture, "SDXL")
        self.assertIsNone(entry.sha256)
        self.assertIsNone(entry.precision_detected)
    
    def test_model_entry_with_optional_fields(self):
        """Test ModelEntry with optional fields."""
        entry = ModelEntry(
            path="/path/to/model.safetensors",
            weight=0.5,
            architecture="SDXL",
            sha256="abc123",
            precision_detected="fp16"
        )
        
        self.assertEqual(entry.sha256, "abc123")
        self.assertEqual(entry.precision_detected, "fp16")
    
    def test_model_entry_to_dict(self):
        """Test ModelEntry serialization to dict."""
        entry = ModelEntry(
            path="/path/to/model.safetensors",
            weight=0.5,
            architecture="SDXL",
            sha256="abc123"
        )
        
        result = entry.to_dict()
        
        self.assertEqual(result['path'], "/path/to/model.safetensors")
        self.assertEqual(result['weight'], 0.5)
        self.assertEqual(result['architecture'], "SDXL")
        self.assertEqual(result['sha256'], "abc123")
        # None values should be excluded
        self.assertNotIn('precision_detected', result)


class TestVAEEntry(unittest.TestCase):
    """Tests for VAEEntry dataclass."""
    
    def test_vae_entry_creation(self):
        """Test creating a VAEEntry."""
        entry = VAEEntry(path="/path/to/vae.safetensors")
        
        self.assertEqual(entry.path, "/path/to/vae.safetensors")
        self.assertIsNone(entry.sha256)
    
    def test_vae_entry_to_dict_excludes_none(self):
        """Test that VAEEntry.to_dict excludes None values."""
        entry = VAEEntry(path="/path/to/vae.safetensors")
        result = entry.to_dict()
        
        self.assertIn('path', result)
        self.assertNotIn('sha256', result)


class TestOutputEntry(unittest.TestCase):
    """Tests for OutputEntry dataclass."""
    
    def test_output_entry_creation(self):
        """Test creating an OutputEntry."""
        entry = OutputEntry(path="output.safetensors")
        
        self.assertEqual(entry.path, "output.safetensors")
        self.assertIsNone(entry.sha256)
        self.assertIsNone(entry.precision_written)
    
    def test_output_entry_to_dict(self):
        """Test OutputEntry serialization."""
        entry = OutputEntry(
            path="output.safetensors",
            sha256="def456",
            precision_written="fp16"
        )
        
        result = entry.to_dict()
        
        self.assertEqual(result['path'], "output.safetensors")
        self.assertEqual(result['sha256'], "def456")
        self.assertEqual(result['precision_written'], "fp16")


class TestMergeManifest(unittest.TestCase):
    """Tests for MergeManifest dataclass."""
    
    def test_manifest_creation_with_defaults(self):
        """Test creating a manifest with default values."""
        models = [
            ModelEntry(path="model1.safetensors", weight=0.5, architecture="SDXL"),
            ModelEntry(path="model2.safetensors", weight=0.5, architecture="SDXL"),
        ]
        
        manifest = MergeManifest(models=models)
        
        self.assertEqual(len(manifest.models), 2)
        self.assertIsNone(manifest.vae)
        self.assertIsNotNone(manifest.output)
        self.assertEqual(manifest.output_precision, 'match')
        self.assertEqual(manifest.device, 'cpu')
        self.assertTrue(manifest.prune)
        self.assertFalse(manifest.overwrite)
    
    def test_manifest_output_post_init_string(self):
        """Test that string output is converted to OutputEntry."""
        models = [ModelEntry(path="m.safetensors", weight=1.0, architecture="SDXL")]
        manifest = MergeManifest(models=models, output="custom_output.safetensors")
        
        self.assertIsInstance(manifest.output, OutputEntry)
        self.assertEqual(manifest.output.path, "custom_output.safetensors")
    
    def test_manifest_output_post_init_none(self):
        """Test that None output is converted to default OutputEntry."""
        models = [ModelEntry(path="m.safetensors", weight=1.0, architecture="SDXL")]
        manifest = MergeManifest(models=models, output=None)
        
        self.assertIsInstance(manifest.output, OutputEntry)
    
    def test_manifest_to_dict(self):
        """Test manifest serialization to dict."""
        models = [
            ModelEntry(path="model1.safetensors", weight=0.5, architecture="SDXL"),
        ]
        vae = VAEEntry(path="vae.safetensors")
        manifest = MergeManifest(models=models, vae=vae)
        
        result = manifest.to_dict()
        
        self.assertIn('models', result)
        self.assertIn('vae', result)
        self.assertIn('output', result)
        self.assertEqual(len(result['models']), 1)
    
    def test_manifest_from_dict(self):
        """Test manifest deserialization from dict."""
        data = {
            'models': [
                {'path': 'model1.safetensors', 'weight': 0.5, 'architecture': 'SDXL'}
            ],
            'vae': {'path': 'vae.safetensors'},
            'output': {'path': 'output.safetensors'},
            'output_precision': 'fp16',
            'device': 'cuda',
            'prune': False,
            'overwrite': True,
        }
        
        result = MergeManifest.from_dict(data)
        
        self.assertEqual(len(result.models), 1)
        self.assertEqual(result.models[0].weight, 0.5)
        self.assertIsNotNone(result.vae)
        self.assertEqual(result.vae.path, 'vae.safetensors')
        self.assertEqual(result.output_precision, 'fp16')
        self.assertEqual(result.device, 'cuda')
        self.assertFalse(result.prune)
        self.assertTrue(result.overwrite)
    
    def test_manifest_from_dict_old_format_vae(self):
        """Test backward compatibility with old VAE format (string path)."""
        data = {
            'models': [
                {'path': 'model1.safetensors', 'weight': 1.0, 'architecture': 'SDXL'}
            ],
            'vae': 'old_vae.safetensors',  # Old string format
        }
        
        result = MergeManifest.from_dict(data)
        
        self.assertIsInstance(result.vae, VAEEntry)
        self.assertEqual(result.vae.path, 'old_vae.safetensors')
    
    def test_manifest_from_dict_old_format_output(self):
        """Test backward compatibility with old output format (string path)."""
        data = {
            'models': [
                {'path': 'model1.safetensors', 'weight': 1.0, 'architecture': 'SDXL'}
            ],
            'output': 'old_output.safetensors',  # Old string format
        }
        
        result = MergeManifest.from_dict(data)
        
        self.assertIsInstance(result.output, OutputEntry)
        self.assertEqual(result.output.path, 'old_output.safetensors')
    
    def test_manifest_roundtrip(self):
        """Test that to_dict -> from_dict produces equivalent manifest."""
        models = [
            ModelEntry(path="m1.safetensors", weight=0.5, architecture="SDXL"),
            ModelEntry(path="m2.safetensors", weight=0.5, architecture="SDXL"),
        ]
        original = MergeManifest(
            models=models,
            vae=VAEEntry(path="vae.safetensors"),
            output=OutputEntry(path="out.safetensors"),
            output_precision='fp16',
            device='cuda',
            prune=False,
            overwrite=True,
        )
        
        data = original.to_dict()
        restored = MergeManifest.from_dict(data)
        
        self.assertEqual(len(restored.models), 2)
        self.assertEqual(restored.vae.path, 'vae.safetensors')
        self.assertEqual(restored.output.path, 'out.safetensors')
        self.assertEqual(restored.output_precision, 'fp16')
        self.assertEqual(restored.device, 'cuda')


class TestManifestSaveLoad(unittest.TestCase):
    """Tests for manifest save and load methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.manifest.print_success')
    def test_save_and_load(self, mock_print):
        """Test saving and loading a manifest."""
        models = [
            ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL")
        ]
        original = MergeManifest(models=models)
        
        manifest_path = self.temp_dir / "test_manifest.json"
        original.save(manifest_path)
        
        # Verify file was created
        self.assertTrue(manifest_path.exists())
        
        # Load and verify
        loaded = MergeManifest.load(manifest_path)
        self.assertEqual(len(loaded.models), 1)
        self.assertEqual(loaded.models[0].path, "model.safetensors")
    
    @patch('model_merger.manifest.print_success')
    def test_saved_file_is_valid_json(self, mock_print):
        """Test that saved manifest is valid JSON."""
        models = [
            ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL")
        ]
        manifest = MergeManifest(models=models)
        
        manifest_path = self.temp_dir / "test.json"
        manifest.save(manifest_path)
        
        # Should be parseable as JSON
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('models', data)


class TestGenerateOutputFilename(unittest.TestCase):
    """Tests for generate_output_filename function."""
    
    def test_two_models_filename(self):
        """Test filename generation with two models."""
        model_files = [
            Path("pony_realistic.safetensors"),
            Path("pony_furry_v2.safetensors"),
        ]
        
        result = generate_output_filename(model_files, "Pony")
        
        self.assertIn("Pony", result)
        self.assertIn("merged", result)
        self.assertTrue(result.endswith(".safetensors"))
    
    def test_strips_punctuation(self):
        """Test that punctuation is stripped from model names."""
        model_files = [
            Path("model_with_underscores.safetensors"),
            Path("model-with-dashes.safetensors"),
        ]
        
        result = generate_output_filename(model_files, "SDXL")
        
        # Result should have clean prefixes
        self.assertIn("SDXL", result)
        self.assertTrue(result.endswith(".safetensors"))
    
    def test_limits_prefixes(self):
        """Test that number of prefixes is limited."""
        model_files = [Path(f"model{i}.safetensors") for i in range(10)]
        
        result = generate_output_filename(model_files, "SDXL")
        
        # Should limit to 5 prefixes + architecture + merged
        parts = result.replace('.safetensors', '').split('_')
        self.assertLessEqual(len(parts), 7)  # arch + 5 prefixes + merged
    
    def test_empty_model_list(self):
        """Test with empty model list."""
        result = generate_output_filename([], "SDXL")
        
        self.assertIn("SDXL", result)
        self.assertIn("merged", result)


class TestScanFolder(unittest.TestCase):
    """Tests for scan_folder function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp/scan_test")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_nonexistent_folder_raises(self):
        """Test that non-existent folder raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            scan_folder(Path("nonexistent_folder_xyz"))
    
    def test_file_instead_of_folder_raises(self):
        """Test that file path raises ValueError."""
        file_path = self.temp_dir / "not_a_folder.txt"
        file_path.touch()
        self.test_files.append(file_path)
        
        with self.assertRaises(ValueError) as context:
            scan_folder(file_path)
        
        self.assertIn("Not a directory", str(context.exception))
    
    @patch('model_merger.manifest.console')
    @patch('model_merger.manifest.print_info')
    @patch('model_merger.manifest.load_model')
    def test_scan_folder_finds_models(self, mock_load, mock_print, mock_console):
        """Test that scan_folder finds .safetensors files."""
        # Create dummy models
        model1 = create_dummy_model("pony_model1.safetensors", temp_dir=self.temp_dir)
        model2 = create_dummy_model("pony_model2.safetensors", temp_dir=self.temp_dir)
        self.test_files.extend([model1, model2])
        
        # Mock load_model to return metadata
        mock_load.return_value = ({}, {'precision': 'fp32'})
        
        result = scan_folder(self.temp_dir)
        
        self.assertIsInstance(result, MergeManifest)
        self.assertEqual(len(result.models), 2)
    
    @patch('model_merger.manifest.console')
    @patch('model_merger.manifest.print_info')
    @patch('model_merger.manifest.load_model')
    def test_scan_folder_equal_weights(self, mock_load, mock_print, mock_console):
        """Test that equal weights are calculated correctly."""
        model1 = create_dummy_model("model1.safetensors", temp_dir=self.temp_dir)
        model2 = create_dummy_model("model2.safetensors", temp_dir=self.temp_dir)
        self.test_files.extend([model1, model2])
        
        mock_load.return_value = ({}, {'precision': 'fp32'})
        
        result = scan_folder(self.temp_dir, equal_weights=True)
        
        # Two models should each have weight 0.5
        self.assertAlmostEqual(result.models[0].weight, 0.5)
        self.assertAlmostEqual(result.models[1].weight, 0.5)
    
    def test_scan_empty_folder_raises(self):
        """Test that empty folder raises ValueError."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir(exist_ok=True)
        
        with self.assertRaises(ValueError) as context:
            scan_folder(empty_dir)
        
        self.assertIn("No .safetensors files found", str(context.exception))


class TestValidateManifest(unittest.TestCase):
    """Tests for validate_manifest function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    def test_valid_manifest_no_issues(self):
        """Test that valid manifest returns no issues."""
        # Create actual model files
        model_path = create_dummy_model("valid_model.safetensors")
        self.test_files.append(model_path)
        
        models = [
            ModelEntry(path=str(model_path), weight=1.0, architecture="SDXL")
        ]
        manifest = MergeManifest(models=models)
        
        issues = validate_manifest(manifest)
        
        self.assertEqual(len(issues), 0)
    
    def test_missing_model_file_reported(self):
        """Test that missing model file is reported."""
        models = [
            ModelEntry(
                path="nonexistent_model.safetensors",
                weight=1.0,
                architecture="SDXL"
            )
        ]
        manifest = MergeManifest(models=models)
        
        issues = validate_manifest(manifest)
        
        self.assertTrue(any("not found" in issue for issue in issues))
    
    def test_missing_vae_file_reported(self):
        """Test that missing VAE file is reported."""
        model_path = create_dummy_model("model.safetensors")
        self.test_files.append(model_path)
        
        models = [
            ModelEntry(path=str(model_path), weight=1.0, architecture="SDXL")
        ]
        manifest = MergeManifest(
            models=models,
            vae=VAEEntry(path="nonexistent_vae.safetensors")
        )
        
        issues = validate_manifest(manifest)
        
        self.assertTrue(any("VAE file not found" in issue for issue in issues))
    
    def test_weight_sum_warning(self):
        """Test that weights not summing to 1.0 generates warning."""
        model1 = create_dummy_model("m1.safetensors")
        model2 = create_dummy_model("m2.safetensors")
        self.test_files.extend([model1, model2])
        
        models = [
            ModelEntry(path=str(model1), weight=0.3, architecture="SDXL"),
            ModelEntry(path=str(model2), weight=0.3, architecture="SDXL"),
        ]
        manifest = MergeManifest(models=models)
        
        issues = validate_manifest(manifest)
        
        self.assertTrue(any("Weights sum to" in issue for issue in issues))
    
    def test_exact_weight_sum_no_warning(self):
        """Test that weights summing to 1.0 don't generate warning."""
        model1 = create_dummy_model("m1_ok.safetensors")
        model2 = create_dummy_model("m2_ok.safetensors")
        self.test_files.extend([model1, model2])
        
        models = [
            ModelEntry(path=str(model1), weight=0.5, architecture="SDXL"),
            ModelEntry(path=str(model2), weight=0.5, architecture="SDXL"),
        ]
        manifest = MergeManifest(models=models)
        
        issues = validate_manifest(manifest)
        
        # Should only have no issues (no weight warning)
        self.assertFalse(any("Weights sum" in issue for issue in issues))


if __name__ == '__main__':
    unittest.main()
