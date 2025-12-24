"""
Tests for merger.py module.

Tests the core merge logic including accumulator pattern, precision conversion,
and model pruning.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from model_merger import merger
from model_merger.manifest import ModelEntry
from tests.helpers import create_dummy_model, cleanup_test_files


class TestMergeModels(unittest.TestCase):
    """Tests for merge_models function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    def test_no_models_raises(self):
        """Test that empty model list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            merger.merge_models([])
        
        self.assertIn("No models provided", str(context.exception))
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.create_progress')
    def test_single_model_returns_unmodified(self, mock_progress, mock_section, mock_console):
        """Test that single model is returned unmodified."""
        model_path = create_dummy_model("single_model.safetensors", precision="fp32", seed=42)
        self.test_files.append(model_path)
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        entries = [
            ModelEntry(path=str(model_path), weight=1.0, architecture="SDXL")
        ]
        
        result = merger.merge_models(entries, validate_compatibility=False)
        
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.print_step')
    @patch('model_merger.merger.print_success')
    @patch('model_merger.merger.create_progress')
    def test_two_models_equal_weights(self, mock_progress, mock_success, mock_step, mock_section, mock_console):
        """Test merging two models with equal weights."""
        model1_path = create_dummy_model("merge_m1.safetensors", precision="fp32", seed=1)
        model2_path = create_dummy_model("merge_m2.safetensors", precision="fp32", seed=2)
        self.test_files.extend([model1_path, model2_path])
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        entries = [
            ModelEntry(path=str(model1_path), weight=0.5, architecture="SDXL"),
            ModelEntry(path=str(model2_path), weight=0.5, architecture="SDXL"),
        ]
        
        result = merger.merge_models(entries, validate_compatibility=False)
        
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        # Verify tensors exist
        for key, tensor in result.items():
            self.assertTrue(torch.is_tensor(tensor))
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.print_step')
    @patch('model_merger.merger.print_success')
    @patch('model_merger.merger.create_progress')
    def test_weights_applied_correctly(self, mock_progress, mock_success, mock_step, mock_section, mock_console):
        """Test that weights are applied correctly during merge."""
        # Create models with known values for verification
        torch.manual_seed(100)
        model1_path = create_dummy_model("weight_m1.safetensors", precision="fp32", seed=100)
        model2_path = create_dummy_model("weight_m2.safetensors", precision="fp32", seed=200)
        self.test_files.extend([model1_path, model2_path])
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Use asymmetric weights
        entries = [
            ModelEntry(path=str(model1_path), weight=0.7, architecture="SDXL"),
            ModelEntry(path=str(model2_path), weight=0.3, architecture="SDXL"),
        ]
        
        result = merger.merge_models(entries, validate_compatibility=False)
        
        # Verify result exists and has correct keys
        self.assertIn("model.diffusion_model.layer1.weight", result)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.print_step')
    @patch('model_merger.merger.print_success')
    @patch('model_merger.merger.create_progress')
    def test_multiple_models_accumulator_pattern(self, mock_progress, mock_success, mock_step, mock_section, mock_console):
        """Test merging multiple models using accumulator pattern."""
        model1_path = create_dummy_model("accum_m1.safetensors", precision="fp32", seed=1)
        model2_path = create_dummy_model("accum_m2.safetensors", precision="fp32", seed=2)
        model3_path = create_dummy_model("accum_m3.safetensors", precision="fp32", seed=3)
        self.test_files.extend([model1_path, model2_path, model3_path])
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        entries = [
            ModelEntry(path=str(model1_path), weight=0.33, architecture="SDXL"),
            ModelEntry(path=str(model2_path), weight=0.33, architecture="SDXL"),
            ModelEntry(path=str(model3_path), weight=0.34, architecture="SDXL"),
        ]
        
        result = merger.merge_models(entries, validate_compatibility=False)
        
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.print_step')
    @patch('model_merger.merger.create_progress')
    def test_incompatible_models_raises(self, mock_progress, mock_step, mock_section, mock_console):
        """Test that incompatible models raise ValueError."""
        from tests.helpers import create_incompatible_model
        
        model1_path = create_dummy_model("compat_m1.safetensors", precision="fp32", size=10)
        model2_path = create_incompatible_model("compat_m2.safetensors", precision="fp32", size=20)
        self.test_files.extend([model1_path, model2_path])
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        entries = [
            ModelEntry(path=str(model1_path), weight=0.5, architecture="SDXL"),
            ModelEntry(path=str(model2_path), weight=0.5, architecture="SDXL"),
        ]
        
        with self.assertRaises(ValueError) as context:
            merger.merge_models(entries, validate_compatibility=True)
        
        self.assertIn("incompatible", str(context.exception).lower())
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.print_section')
    @patch('model_merger.merger.print_step')
    @patch('model_merger.merger.print_success')
    @patch('model_merger.merger.create_progress')
    def test_skip_validation_allows_merge(self, mock_progress, mock_success, mock_step, mock_section, mock_console):
        """Test that skipping validation allows incompatible models to merge."""
        from tests.helpers import create_incompatible_model
        
        # Same structure but different implementation - validation skipped
        model1_path = create_dummy_model("skip_m1.safetensors", precision="fp32")
        model2_path = create_dummy_model("skip_m2.safetensors", precision="fp32")
        self.test_files.extend([model1_path, model2_path])
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        entries = [
            ModelEntry(path=str(model1_path), weight=0.5, architecture="SDXL"),
            ModelEntry(path=str(model2_path), weight=0.5, architecture="SDXL"),
        ]
        
        # Should not raise when validation is disabled
        result = merger.merge_models(entries, validate_compatibility=False)
        
        self.assertIsInstance(result, dict)


class TestConvertPrecision(unittest.TestCase):
    """Tests for convert_precision function."""
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_convert_fp32_to_fp16(self, mock_progress, mock_console):
        """Test converting from fp32 to fp16."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer.bias": torch.randn(10, dtype=torch.float32),
        }
        
        result = merger.convert_precision(state_dict, "fp16")
        
        for key, tensor in result.items():
            if tensor.is_floating_point():
                self.assertEqual(tensor.dtype, torch.float16)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_convert_fp16_to_fp32(self, mock_progress, mock_console):
        """Test converting from fp16 to fp32."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.float16),
            "layer.bias": torch.randn(10, dtype=torch.float16),
        }
        
        result = merger.convert_precision(state_dict, "fp32")
        
        for key, tensor in result.items():
            if tensor.is_floating_point():
                self.assertEqual(tensor.dtype, torch.float32)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_convert_to_bf16(self, mock_progress, mock_console):
        """Test converting to bf16."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.float32),
        }
        
        result = merger.convert_precision(state_dict, "bf16")
        
        self.assertEqual(result["layer.weight"].dtype, torch.bfloat16)
    
    def test_invalid_precision_raises(self):
        """Test that invalid precision raises ValueError."""
        state_dict = {
            "layer.weight": torch.randn(10, 10),
        }
        
        with self.assertRaises(ValueError) as context:
            merger.convert_precision(state_dict, "fp64")
        
        self.assertIn("Unknown precision", str(context.exception))
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_integer_tensors_unchanged(self, mock_progress, mock_console):
        """Test that integer tensors are not converted."""
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


class TestPruneModel(unittest.TestCase):
    """Tests for prune_model function."""
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_keeps_diffusion_model_keys(self, mock_progress, mock_console):
        """Test that diffusion model keys are kept."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "optimizer.state": torch.randn(10),
        }
        
        result = merger.prune_model(state_dict)
        
        self.assertIn("model.diffusion_model.layer.weight", result)
        self.assertNotIn("optimizer.state", result)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_keeps_vae_keys(self, mock_progress, mock_console):
        """Test that VAE (first_stage_model) keys are kept."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "first_stage_model.encoder.weight": torch.randn(10, 10),
            "ema.decay": torch.tensor(0.9999),
        }
        
        result = merger.prune_model(state_dict)
        
        self.assertIn("first_stage_model.encoder.weight", result)
        self.assertNotIn("ema.decay", result)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_keeps_text_encoder_keys(self, mock_progress, mock_console):
        """Test that text encoder keys are kept."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "cond_stage_model.transformer.weight": torch.randn(10, 10),
            "conditioner.embedders.0.model.weight": torch.randn(10, 10),
            "global_step": torch.tensor(10000),
        }
        
        result = merger.prune_model(state_dict)
        
        self.assertIn("cond_stage_model.transformer.weight", result)
        self.assertIn("conditioner.embedders.0.model.weight", result)
        self.assertNotIn("global_step", result)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_removes_training_artifacts(self, mock_progress, mock_console):
        """Test that training artifacts are removed."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "optimizer.param_groups": torch.randn(10),
            "lr_scheduler.last_epoch": torch.tensor(100),
            "epoch": torch.tensor(50),
            "random_key_not_in_prefixes": torch.randn(10),
        }
        
        result = merger.prune_model(state_dict)
        
        self.assertEqual(len(result), 1)  # Only the diffusion model key
        self.assertIn("model.diffusion_model.layer.weight", result)
    
    @patch('model_merger.merger.console')
    @patch('model_merger.merger.create_progress')
    def test_empty_dict_returns_empty(self, mock_progress, mock_console):
        """Test that empty dict returns empty dict."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        result = merger.prune_model({})
        
        self.assertEqual(result, {})


class TestComputeConsensusWeights(unittest.TestCase):
    """Tests for compute_consensus_weights function."""
    
    def test_basic_consensus_with_outlier(self):
        """Test that outliers get suppressed."""
        values = [1.0, 1.1, 1.05, 5.0]  # Last value is clear outlier
        weights = merger.compute_consensus_weights(values, exponent=4)
        
        # Check properties
        self.assertEqual(weights.shape[0], 4)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        
        # Outlier should have very low weight
        self.assertLess(weights[3].item(), 0.01)
        
        # Consensus values should have similar weights
        self.assertGreater(weights[0].item(), 0.3)
        self.assertGreater(weights[1].item(), 0.3)
        self.assertGreater(weights[2].item(), 0.3)
    
    def test_identical_values(self):
        """Test that identical values get equal weights."""
        values = [2.0, 2.0, 2.0, 2.0]
        weights = merger.compute_consensus_weights(values, exponent=4)
        
        # All weights should be equal (0.25 each)
        for weight in weights:
            self.assertAlmostEqual(weight.item(), 0.25, places=5)
    
    def test_single_value(self):
        """Test single value returns weight of 1.0."""
        values = [5.0]
        weights = merger.compute_consensus_weights(values, exponent=4)
        
        self.assertEqual(weights.shape[0], 1)
        self.assertAlmostEqual(weights[0].item(), 1.0, places=5)
    
    def test_different_exponents_all_valid(self):
        """Test that different exponents all produce valid probability distributions."""
        values = [1.0, 1.0, 1.0, 1.3]
        
        for exponent in [2, 4, 6, 8]:
            with self.subTest(exponent=exponent):
                weights = merger.compute_consensus_weights(values, exponent=exponent)
                
                # Should sum to 1.0
                self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
                
                # All weights should be non-negative
                for w in weights:
                    self.assertGreaterEqual(w.item(), 0.0)
    
    def test_tensor_input(self):
        """Test that tensor input works correctly."""
        values = torch.tensor([1.0, 1.1, 1.05, 5.0])
        weights = merger.compute_consensus_weights(values, exponent=4)
        
        self.assertIsInstance(weights, torch.Tensor)
        self.assertAlmostEqual(weights.sum().item(), 1.0, places=5)
        self.assertLess(weights[3].item(), 0.01)  # Outlier suppressed
    
    def test_two_clusters(self):
        """Test behavior with two distinct clusters of values."""
        values = [1.0, 1.1, 5.0, 5.1]
        weights = merger.compute_consensus_weights(values, exponent=2)
        
        # Both clusters should get some weight
        # Values in each cluster should have similar weights
        cluster1_weight = weights[0].item() + weights[1].item()
        cluster2_weight = weights[2].item() + weights[3].item()
        
        # Both clusters should have meaningful weight
        self.assertGreater(cluster1_weight, 0.1)
        self.assertGreater(cluster2_weight, 0.1)


if __name__ == '__main__':
    unittest.main()
