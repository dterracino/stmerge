"""
Tests for vae.py module.

Tests VAE baking and extraction functionality.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from model_merger import vae
from model_merger import config
from tests.helpers import create_dummy_model, create_dummy_vae, cleanup_test_files


class TestBakeVae(unittest.TestCase):
    """Tests for bake_vae function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.vae.console')
    @patch('model_merger.vae.print_section')
    @patch('model_merger.vae.print_success')
    @patch('model_merger.vae.create_progress')
    @patch('model_merger.vae.load_vae')
    def test_vae_keys_get_prefix(self, mock_load, mock_progress, mock_success, mock_section, mock_console):
        """Test that VAE keys get the first_stage_model prefix."""
        # Create model state dict
        model_state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
        }
        
        # Mock VAE loading
        vae_state_dict = {
            "encoder.conv_in.weight": torch.randn(10, 3, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 10, 3, 3),
        }
        mock_load.return_value = (vae_state_dict, {'precision': 'fp32'})
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        vae_path = Path("fake_vae.safetensors")
        result = vae.bake_vae(model_state_dict, vae_path)
        
        # Check that prefixed keys exist
        self.assertIn("first_stage_model.encoder.conv_in.weight", result)
        self.assertIn("first_stage_model.decoder.conv_out.weight", result)
    
    @patch('model_merger.vae.console')
    @patch('model_merger.vae.print_section')
    @patch('model_merger.vae.print_success')
    @patch('model_merger.vae.create_progress')
    @patch('model_merger.vae.load_vae')
    def test_existing_vae_keys_replaced(self, mock_load, mock_progress, mock_success, mock_section, mock_console):
        """Test that existing VAE keys in model are replaced."""
        original_tensor = torch.zeros(10, 3, 3, 3)
        new_tensor = torch.ones(10, 3, 3, 3)
        
        model_state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "first_stage_model.encoder.conv_in.weight": original_tensor,
        }
        
        vae_state_dict = {
            "encoder.conv_in.weight": new_tensor,
        }
        mock_load.return_value = (vae_state_dict, {'precision': 'fp32'})
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        vae_path = Path("fake_vae.safetensors")
        result = vae.bake_vae(model_state_dict, vae_path)
        
        # New tensor should replace old one
        self.assertTrue(torch.all(result["first_stage_model.encoder.conv_in.weight"] == new_tensor))
    
    @patch('model_merger.vae.console')
    @patch('model_merger.vae.print_section')
    @patch('model_merger.vae.print_success')
    @patch('model_merger.vae.create_progress')
    @patch('model_merger.vae.load_vae')
    def test_model_keys_preserved(self, mock_load, mock_progress, mock_success, mock_section, mock_console):
        """Test that non-VAE model keys are preserved."""
        original_model_tensor = torch.randn(10, 10)
        
        model_state_dict = {
            "model.diffusion_model.layer.weight": original_model_tensor.clone(),
        }
        
        vae_state_dict = {
            "encoder.conv_in.weight": torch.randn(10, 3, 3, 3),
        }
        mock_load.return_value = (vae_state_dict, {'precision': 'fp32'})
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        vae_path = Path("fake_vae.safetensors")
        result = vae.bake_vae(model_state_dict, vae_path)
        
        # Original model keys should still exist
        self.assertIn("model.diffusion_model.layer.weight", result)
    
    @patch('model_merger.vae.console')
    @patch('model_merger.vae.print_section')
    @patch('model_merger.vae.print_success')
    @patch('model_merger.vae.create_progress')
    @patch('model_merger.vae.load_vae')
    def test_returns_same_dict_object(self, mock_load, mock_progress, mock_success, mock_section, mock_console):
        """Test that bake_vae returns the same dict object (modified in-place)."""
        model_state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
        }
        
        vae_state_dict = {
            "encoder.conv_in.weight": torch.randn(10, 3, 3, 3),
        }
        mock_load.return_value = (vae_state_dict, {'precision': 'fp32'})
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        vae_path = Path("fake_vae.safetensors")
        result = vae.bake_vae(model_state_dict, vae_path)
        
        # Should be the same object
        self.assertIs(result, model_state_dict)
    
    @patch('model_merger.vae.console')
    @patch('model_merger.vae.print_section')
    @patch('model_merger.vae.print_success')
    @patch('model_merger.vae.create_progress')
    def test_bake_vae_with_real_files(self, mock_progress, mock_success, mock_section, mock_console):
        """Test bake_vae with real model and VAE files."""
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Create dummy model without VAE
        model_path = create_dummy_model("model_no_vae.safetensors", include_vae=False)
        vae_path = create_dummy_vae("test_vae_for_bake.safetensors")
        self.test_files.extend([model_path, vae_path])
        
        # Load model
        from safetensors.torch import load_file
        model_state_dict = load_file(str(model_path))
        
        # Bake VAE
        result = vae.bake_vae(model_state_dict, vae_path)
        
        # Verify VAE keys were added
        vae_keys = [k for k in result.keys() if k.startswith("first_stage_model.")]
        self.assertGreater(len(vae_keys), 0)


class TestExtractVae(unittest.TestCase):
    """Tests for extract_vae function."""
    
    def test_extracts_vae_keys(self):
        """Test that VAE keys are extracted correctly."""
        model_state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "first_stage_model.encoder.conv_in.weight": torch.randn(10, 3, 3, 3),
            "first_stage_model.decoder.conv_out.weight": torch.randn(3, 10, 3, 3),
        }
        
        output_path = Path("extracted_vae.safetensors")  # Not used yet
        result = vae.extract_vae(model_state_dict, output_path)
        
        # Should have VAE keys without prefix
        self.assertIn("encoder.conv_in.weight", result)
        self.assertIn("decoder.conv_out.weight", result)
        # Should not have model keys
        self.assertNotIn("model.diffusion_model.layer.weight", result)
    
    def test_strips_prefix(self):
        """Test that first_stage_model prefix is stripped."""
        original_key = "first_stage_model.encoder.layer.weight"
        expected_key = "encoder.layer.weight"
        
        model_state_dict = {
            original_key: torch.randn(10, 10),
        }
        
        output_path = Path("output.safetensors")
        result = vae.extract_vae(model_state_dict, output_path)
        
        self.assertIn(expected_key, result)
        self.assertNotIn(original_key, result)
    
    def test_empty_model_returns_empty(self):
        """Test that model without VAE returns empty dict."""
        model_state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
        }
        
        output_path = Path("output.safetensors")
        result = vae.extract_vae(model_state_dict, output_path)
        
        self.assertEqual(len(result), 0)
    
    def test_preserves_tensor_values(self):
        """Test that extracted tensors have correct values."""
        original_tensor = torch.randn(10, 10)
        
        model_state_dict = {
            "first_stage_model.encoder.layer.weight": original_tensor,
        }
        
        output_path = Path("output.safetensors")
        result = vae.extract_vae(model_state_dict, output_path)
        
        self.assertTrue(torch.equal(result["encoder.layer.weight"], original_tensor))


class TestVaeKeyPrefix(unittest.TestCase):
    """Tests for VAE key prefix handling."""
    
    def test_vae_key_prefix_constant(self):
        """Test that VAE key prefix constant is correct."""
        self.assertEqual(config.VAE_KEY_PREFIX, "first_stage_model.")
    
    def test_prefix_ends_with_dot(self):
        """Test that prefix ends with a dot for proper concatenation."""
        self.assertTrue(config.VAE_KEY_PREFIX.endswith("."))


if __name__ == '__main__':
    unittest.main()
