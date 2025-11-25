"""
Integration tests for the model merger.

Tests end-to-end workflows involving multiple modules working together.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from model_merger import loader, merger, saver, vae, manifest, converter
from model_merger.manifest import MergeManifest, ModelEntry, VAEEntry, OutputEntry
from tests.helpers import (
    create_dummy_model, create_dummy_vae, create_dummy_checkpoint,
    cleanup_test_files, get_state_dict
)


class TestMergeWorkflow(unittest.TestCase):
    """Integration tests for the merge workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.loader.console')
    @patch('model_merger.loader.print_info')
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.print_step')
    @patch('model_merger.merger.print_success')
    @patch('model_merger.merger.create_progress')
    def test_load_and_merge_two_models(
        self, mock_progress, mock_success, mock_step, 
        mock_section, mock_merger_console, mock_print_info, mock_loader_console
    ):
        """Test loading and merging two models end-to-end."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create two models
        model1_path = create_dummy_model("int_m1.safetensors", precision="fp32", seed=1, temp_dir=self.temp_dir)
        model2_path = create_dummy_model("int_m2.safetensors", precision="fp32", seed=2, temp_dir=self.temp_dir)
        self.test_files.extend([model1_path, model2_path])
        
        # Create entries
        entries = [
            ModelEntry(path=str(model1_path), weight=0.5, architecture="SDXL"),
            ModelEntry(path=str(model2_path), weight=0.5, architecture="SDXL"),
        ]
        
        # Merge
        result = merger.merge_models(entries, validate_compatibility=False)
        
        # Verify result
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertIn("model.diffusion_model.layer1.weight", result)
    
    @patch('model_merger.loader.console')
    @patch('model_merger.loader.print_info')
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.print_step')
    @patch('model_merger.merger.print_success')
    @patch('model_merger.merger.create_progress')
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.print_info')
    @patch('model_merger.saver.create_progress')
    @patch('model_merger.saver.compute_file_hash')
    def test_merge_and_save_workflow(
        self, mock_hash, mock_saver_progress, mock_saver_info, mock_saver_console,
        mock_merger_progress, mock_success, mock_step, mock_section, 
        mock_merger_console, mock_print_info, mock_loader_console
    ):
        """Test merging models and saving the result."""
        mock_merger_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_merger_progress.return_value.__exit__ = MagicMock(return_value=False)
        mock_saver_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_saver_progress.return_value.__exit__ = MagicMock(return_value=False)
        mock_hash.return_value = "a" * 64
        
        # Create models
        model1_path = create_dummy_model("save_m1.safetensors", precision="fp32", temp_dir=self.temp_dir)
        model2_path = create_dummy_model("save_m2.safetensors", precision="fp32", temp_dir=self.temp_dir)
        output_path = self.temp_dir / "merged_output.safetensors"
        self.test_files.extend([model1_path, model2_path, output_path])
        
        # Merge
        entries = [
            ModelEntry(path=str(model1_path), weight=0.5, architecture="SDXL"),
            ModelEntry(path=str(model2_path), weight=0.5, architecture="SDXL"),
        ]
        merged = merger.merge_models(entries, validate_compatibility=False)
        
        # Save
        output_hash = saver.save_model(merged, output_path)
        
        # Verify
        self.assertTrue(output_path.exists())
        self.assertEqual(len(output_hash), 64)


class TestManifestWorkflow(unittest.TestCase):
    """Integration tests for manifest-based workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manifest_save_load_roundtrip(self):
        """Test that manifest survives save/load cycle."""
        # Create manifest
        models = [
            ModelEntry(path="model1.safetensors", weight=0.5, architecture="SDXL", sha256="abc123"),
            ModelEntry(path="model2.safetensors", weight=0.5, architecture="SDXL"),
        ]
        original = MergeManifest(
            models=models,
            vae=VAEEntry(path="vae.safetensors", sha256="def456"),
            output=OutputEntry(path="output.safetensors"),
            output_precision='fp16',
            device='cpu',
            prune=True,
            overwrite=False,
        )
        
        # Save
        manifest_path = self.temp_dir / "test_manifest.json"
        with patch('model_merger.manifest.print_success'):
            original.save(manifest_path)
        
        # Load
        loaded = MergeManifest.load(manifest_path)
        
        # Verify
        self.assertEqual(len(loaded.models), 2)
        self.assertEqual(loaded.models[0].sha256, "abc123")
        self.assertEqual(loaded.vae.sha256, "def456")
        self.assertEqual(loaded.output_precision, 'fp16')
    
    def test_manifest_validation_catches_issues(self):
        """Test that validation catches common issues."""
        # Create manifest with missing files
        models = [
            ModelEntry(path="nonexistent1.safetensors", weight=0.5, architecture="SDXL"),
            ModelEntry(path="nonexistent2.safetensors", weight=0.3, architecture="SDXL"),
        ]
        m = MergeManifest(models=models)
        
        issues = manifest.validate_manifest(m)
        
        # Should have issues about missing files and weight sum
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("not found" in issue for issue in issues))
        self.assertTrue(any("Weights sum" in issue for issue in issues))


class TestVaeBakingWorkflow(unittest.TestCase):
    """Integration tests for VAE baking workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.vae.console')
    @patch('model_merger.vae.print_section')
    @patch('model_merger.vae.print_success')
    @patch('model_merger.vae.create_progress')
    def test_bake_vae_into_model(self, mock_progress, mock_success, mock_section, mock_console):
        """Test baking a VAE into a model."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create model without VAE
        model_path = create_dummy_model("no_vae_model.safetensors", include_vae=False, temp_dir=self.temp_dir)
        vae_path = create_dummy_vae("bake_vae.safetensors", temp_dir=self.temp_dir)
        self.test_files.extend([model_path, vae_path])
        
        # Load model
        model_dict = get_state_dict(model_path)
        
        # Count VAE keys before
        vae_keys_before = [k for k in model_dict.keys() if k.startswith("first_stage_model.")]
        self.assertEqual(len(vae_keys_before), 0)
        
        # Bake VAE
        result = vae.bake_vae(model_dict, vae_path)
        
        # Count VAE keys after
        vae_keys_after = [k for k in result.keys() if k.startswith("first_stage_model.")]
        self.assertGreater(len(vae_keys_after), 0)


class TestPrecisionConversionWorkflow(unittest.TestCase):
    """Integration tests for precision conversion workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_convert_fp32_to_fp16(self, mock_progress, mock_console):
        """Test converting merged model from fp32 to fp16."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create fp32 state dict
        state_dict = {
            "model.layer.weight": torch.randn(10, 10, dtype=torch.float32),
            "model.layer.bias": torch.randn(10, dtype=torch.float32),
        }
        
        # Convert
        result = merger.convert_precision(state_dict, "fp16")
        
        # Verify
        for key, tensor in result.items():
            if tensor.is_floating_point():
                self.assertEqual(tensor.dtype, torch.float16)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_convert_preserves_integer_tensors(self, mock_progress, mock_console):
        """Test that precision conversion preserves integer tensors."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.float32),
            "position_ids": torch.arange(10, dtype=torch.int64),
        }
        
        result = merger.convert_precision(state_dict, "fp16")
        
        # Float should be converted
        self.assertEqual(result["layer.weight"].dtype, torch.float16)
        # Integer should remain unchanged
        self.assertEqual(result["position_ids"].dtype, torch.int64)


class TestPruningWorkflow(unittest.TestCase):
    """Integration tests for model pruning workflow."""
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_prune_removes_training_artifacts(self, mock_progress, mock_console):
        """Test that pruning removes training artifacts."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create state dict with training artifacts
        state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "first_stage_model.encoder.weight": torch.randn(10, 10),
            "cond_stage_model.transformer.weight": torch.randn(10, 10),
            "optimizer.param_groups": torch.randn(10),
            "epoch": torch.tensor(100),
            "global_step": torch.tensor(10000),
        }
        
        result = merger.prune_model(state_dict)
        
        # Model keys should be kept
        self.assertIn("model.diffusion_model.layer.weight", result)
        self.assertIn("first_stage_model.encoder.weight", result)
        self.assertIn("cond_stage_model.transformer.weight", result)
        
        # Training artifacts should be removed
        self.assertNotIn("optimizer.param_groups", result)
        self.assertNotIn("epoch", result)
        self.assertNotIn("global_step", result)


class TestConversionWorkflow(unittest.TestCase):
    """Integration tests for checkpoint conversion workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.converter.console')
    @patch('model_merger.saver.console')
    @patch('model_merger.saver.create_progress')
    @patch('model_merger.saver.print_info')
    @patch('model_merger.saver.compute_file_hash')
    def test_convert_ckpt_to_safetensors(
        self, mock_hash, mock_info, mock_saver_progress, 
        mock_saver_console, mock_converter_console
    ):
        """Test converting .ckpt to .safetensors."""
        mock_saver_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_saver_progress.return_value.__exit__ = MagicMock(return_value=False)
        mock_hash.return_value = "b" * 64
        
        # Create checkpoint
        ckpt_path = create_dummy_checkpoint("test.ckpt", format_type="bare", temp_dir=self.temp_dir)
        output_path = self.temp_dir / "test.safetensors"
        self.test_files.extend([ckpt_path, output_path])
        
        # Convert
        result_hash = converter.convert_to_safetensors(
            ckpt_path, 
            output_path=output_path,
            prune=True
        )
        
        # Verify
        self.assertTrue(output_path.exists())
        self.assertEqual(len(result_hash), 64)


class TestHashVerification(unittest.TestCase):
    """Integration tests for hash computation and verification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.console.create_progress')
    def test_same_file_produces_same_hash(self, mock_progress):
        """Test that same file produces same hash."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        model_path = create_dummy_model("hash_test.safetensors", seed=42, temp_dir=self.temp_dir)
        self.test_files.append(model_path)
        
        hash1 = loader.compute_file_hash(model_path)
        hash2 = loader.compute_file_hash(model_path)
        
        self.assertEqual(hash1, hash2)
    
    @patch('model_merger.console.create_progress')
    def test_different_files_produce_different_hashes(self, mock_progress):
        """Test that different files produce different hashes."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        model1_path = create_dummy_model("hash_test1.safetensors", seed=1, temp_dir=self.temp_dir)
        model2_path = create_dummy_model("hash_test2.safetensors", seed=2, temp_dir=self.temp_dir)
        self.test_files.extend([model1_path, model2_path])
        
        hash1 = loader.compute_file_hash(model1_path)
        hash2 = loader.compute_file_hash(model2_path)
        
        self.assertNotEqual(hash1, hash2)


if __name__ == '__main__':
    unittest.main()
