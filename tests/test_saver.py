"""
Tests for saver.py module.

Tests model saving, contiguity handling, and metadata generation.
"""

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from model_merger import saver
from model_merger.manifest import MergeManifest, ModelEntry, VAEEntry, OutputEntry
from tests.helpers import create_dummy_model, cleanup_test_files


class TestPrepareTensors(unittest.TestCase):
    """Tests for prepare_tensors function."""
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    def test_contiguous_tensors_unchanged(self, mock_progress, mock_console):
        """Test that already contiguous tensors are unchanged."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        original = torch.randn(10, 10)
        state_dict = {"layer.weight": original}
        
        result = saver.prepare_tensors(state_dict)
        
        self.assertTrue(result["layer.weight"].is_contiguous())
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    def test_noncontiguous_tensors_made_contiguous(self, mock_progress, mock_console):
        """Test that non-contiguous tensors are made contiguous."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create non-contiguous tensor via transpose
        original = torch.randn(10, 20).t()  # Transpose makes it non-contiguous
        self.assertFalse(original.is_contiguous())
        
        state_dict = {"layer.weight": original}
        
        result = saver.prepare_tensors(state_dict)
        
        self.assertTrue(result["layer.weight"].is_contiguous())
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    def test_mixed_tensors(self, mock_progress, mock_console):
        """Test handling of mixed contiguous and non-contiguous tensors."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "contiguous": torch.randn(10, 10),
            "noncontiguous": torch.randn(10, 20).t(),
        }
        
        result = saver.prepare_tensors(state_dict)
        
        self.assertTrue(result["contiguous"].is_contiguous())
        self.assertTrue(result["noncontiguous"].is_contiguous())


class TestSaveModel(unittest.TestCase):
    """Tests for save_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    @patch('model_merger.saver.print_info')
    @patch('model_merger.saver.compute_file_hash')
    def test_save_model_creates_file(self, mock_hash, mock_info, mock_progress, mock_console):
        """Test that save_model creates the output file."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        mock_hash.return_value = "abc123" * 10 + "abcd"
        
        state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "model.diffusion_model.layer.bias": torch.randn(10),
        }
        
        output_path = self.temp_dir / "saved_model.safetensors"
        self.test_files.append(output_path)
        
        result = saver.save_model(state_dict, output_path)
        
        self.assertTrue(output_path.exists())
        self.assertEqual(len(result), 64)  # SHA-256 hash length
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    @patch('model_merger.saver.print_info')
    @patch('model_merger.saver.compute_file_hash')
    def test_save_model_with_metadata(self, mock_hash, mock_info, mock_progress, mock_console):
        """Test that save_model includes metadata."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        mock_hash.return_value = "def456" * 10 + "defg"
        
        state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
        }
        metadata = {"author": "test", "version": "1.0"}
        
        output_path = self.temp_dir / "saved_with_meta.safetensors"
        self.test_files.append(output_path)
        
        saver.save_model(state_dict, output_path, metadata=metadata)
        
        self.assertTrue(output_path.exists())
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    def test_save_model_no_overwrite(self, mock_progress, mock_console):
        """Test that save_model refuses to overwrite without flag."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {"layer.weight": torch.randn(10, 10)}
        
        output_path = self.temp_dir / "existing_model.safetensors"
        output_path.touch()  # Create existing file
        self.test_files.append(output_path)
        
        with self.assertRaises(FileExistsError) as context:
            saver.save_model(state_dict, output_path, overwrite=False)
        
        self.assertIn("already exists", str(context.exception))
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    @patch('model_merger.saver.print_info')
    @patch('model_merger.saver.compute_file_hash')
    def test_save_model_with_overwrite(self, mock_hash, mock_info, mock_progress, mock_console):
        """Test that save_model can overwrite with flag."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        mock_hash.return_value = "ghi789" * 10 + "ghij"
        
        state_dict = {"layer.weight": torch.randn(10, 10)}
        
        output_path = self.temp_dir / "overwrite_model.safetensors"
        output_path.touch()  # Create existing file
        self.test_files.append(output_path)
        
        # Should not raise with overwrite=True
        result = saver.save_model(state_dict, output_path, overwrite=True)
        
        self.assertTrue(output_path.exists())
    
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    @patch('model_merger.saver.print_info')
    @patch('model_merger.saver.compute_file_hash')
    def test_save_model_creates_parent_dirs(self, mock_hash, mock_info, mock_progress, mock_console):
        """Test that save_model creates parent directories."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        mock_hash.return_value = "jkl012" * 10 + "jklm"
        
        state_dict = {"layer.weight": torch.randn(10, 10)}
        
        output_path = self.temp_dir / "new_subdir" / "nested" / "model.safetensors"
        self.test_files.append(output_path)
        
        saver.save_model(state_dict, output_path)
        
        self.assertTrue(output_path.exists())
        
        # Clean up created directories
        shutil.rmtree(self.temp_dir / "new_subdir", ignore_errors=True)


class TestSaveManifestMetadata(unittest.TestCase):
    """Tests for save_manifest_metadata function."""
    
    def test_basic_metadata_generation(self):
        """Test generating basic metadata from manifest."""
        models = [
            ModelEntry(
                path="model1.safetensors",
                weight=0.5,
                architecture="SDXL",
                index=0,
                sha256="abc123"
            ),
            ModelEntry(
                path="model2.safetensors",
                weight=0.5,
                architecture="SDXL",
                index=1,
            ),
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp16")
        
        self.assertEqual(result['merge_tool'], 'model_merger')
        self.assertEqual(result['merge_precision'], 'fp16')
        # New format uses JSON-encoded sd_merge_recipe and sd_merge_models
        self.assertIn('sd_merge_recipe', result)
        self.assertIn('sd_merge_models', result)
    
    def test_metadata_with_vae(self):
        """Test that VAE info is included in metadata."""
        models = [
            ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)
        ]
        vae = VAEEntry(
            path="vae.safetensors",
            sha256="vae_hash_123",
            precision_detected="fp16"
        )
        manifest = MergeManifest(models=models, vae=vae)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        # VAE info is now in sd_merge_recipe, not top-level
        self.assertIn('sd_merge_recipe', result)
        import json
        recipe = json.loads(result['sd_merge_recipe'])
        self.assertIn('vae', recipe)
    
    def test_metadata_without_vae(self):
        """Test that metadata is valid without VAE."""
        models = [
            ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        # VAE is no longer a top-level key in new format
        import json
        recipe = json.loads(result['sd_merge_recipe'])
        self.assertNotIn('vae', recipe)
    
    def test_all_metadata_values_are_strings(self):
        """Test that all metadata values are strings."""
        models = [
            ModelEntry(path="model.safetensors", weight=0.5, architecture="SDXL", index=0)
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        for key, value in result.items():
            self.assertIsInstance(value, str, f"Key '{key}' has non-string value: {type(value)}")
    
    def test_model_sha256_included_when_available(self):
        """Test that model SHA256 is included when available."""
        models = [
            ModelEntry(
                path="model.safetensors",
                weight=1.0,
                architecture="SDXL",
                index=0,
                sha256="full_hash_here"
            )
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        # Hash is now in sd_merge_models mapping
        import json
        models_dict = json.loads(result['sd_merge_models'])
        self.assertIn('full_hash_here', models_dict)
    
    def test_model_sha256_excluded_when_none(self):
        """Test that model SHA256 is excluded when None."""
        models = [
            ModelEntry(
                path="model.safetensors",
                weight=1.0,
                architecture="SDXL",
                index=0,
                sha256=None
            )
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        # 'unknown' is used when no hash is available
        import json
        recipe = json.loads(result['sd_merge_recipe'])
        self.assertEqual(recipe['models'][0]['hash'], 'unknown')
    
    def test_metadata_includes_crc32_legacy_hash(self):
        """Test that CRC32 is used for legacy_hash in sd_merge_models."""
        models = [
            ModelEntry(
                path="model.safetensors",
                weight=1.0,
                architecture="SDXL",
                index=0,
                sha256="abc123" * 10 + "abcd",  # 64 chars
                crc32="12345678"
            )
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        import json
        models_dict = json.loads(result['sd_merge_models'])
        model_info = models_dict["abc123" * 10 + "abcd"]
        self.assertEqual(model_info['legacy_hash'], "12345678")
    
    def test_metadata_legacy_hash_fallback_to_sha256(self):
        """Test that legacy_hash falls back to sha256[:8] when CRC32 is None."""
        models = [
            ModelEntry(
                path="model.safetensors",
                weight=1.0,
                architecture="SDXL",
                index=0,
                sha256="abcdef12" * 8,  # 64 chars
                crc32=None
            )
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        import json
        models_dict = json.loads(result['sd_merge_models'])
        model_info = models_dict["abcdef12" * 8]
        # Should use first 8 chars of SHA256
        self.assertEqual(model_info['legacy_hash'], "abcdef12")
    
    def test_metadata_includes_model_indices(self):
        """Test that model indices are included in sd_merge_recipe."""
        models = [
            ModelEntry(path="model1.safetensors", weight=0.3, architecture="SDXL", index=0, sha256="hash1" + "0" * 59),
            ModelEntry(path="model2.safetensors", weight=0.3, architecture="SDXL", index=1, sha256="hash2" + "0" * 59),
            ModelEntry(path="model3.safetensors", weight=0.4, architecture="SDXL", index=2, sha256="hash3" + "0" * 59)
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        import json
        recipe = json.loads(result['sd_merge_recipe'])
        self.assertEqual(recipe['models'][0]['index'], 0)
        self.assertEqual(recipe['models'][1]['index'], 1)
        self.assertEqual(recipe['models'][2]['index'], 2)
    
    def test_metadata_includes_algorithm_weighted_sum(self):
        """Test that algorithm defaults to weighted_sum."""
        models = [
            ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)
        ]
        manifest = MergeManifest(models=models, merge_method='weighted_sum')
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        import json
        recipe = json.loads(result['sd_merge_recipe'])
        self.assertEqual(recipe['algorithm'], 'weighted_sum')
    
    def test_metadata_includes_algorithm_consensus(self):
        """Test that consensus algorithm is captured."""
        models = [
            ModelEntry(path="model.safetensors", weight=0.5, architecture="SDXL", index=0)
        ]
        manifest = MergeManifest(models=models, merge_method='consensus', consensus_exponent=4)
        
        result = saver.save_manifest_metadata(manifest, "fp32")
        
        import json
        recipe = json.loads(result['sd_merge_recipe'])
        self.assertEqual(recipe['algorithm'], 'consensus')
        self.assertEqual(recipe['consensus_exponent'], 4)
    
    def test_metadata_roundtrip_crc32(self):
        """Test that CRC32 survives save/load through metadata."""
        models = [
            ModelEntry(
                path="model.safetensors",
                weight=1.0,
                architecture="Pony",
                index=0,
                sha256="fedcba98" * 8,
                crc32="abcd1234"
            )
        ]
        manifest = MergeManifest(models=models)
        
        result = saver.save_manifest_metadata(manifest, "fp16")
        
        # Verify CRC32 is in the output
        import json
        models_dict = json.loads(result['sd_merge_models'])
        model_info = models_dict["fedcba98" * 8]
        self.assertEqual(model_info['legacy_hash'], "abcd1234")
        self.assertEqual(model_info['name'], "model")
        self.assertEqual(model_info['architecture'], "Pony")


if __name__ == '__main__':
    unittest.main()
