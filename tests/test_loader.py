"""
Tests for loader.py module.

Tests model loading, VAE loading, precision detection, hash computation,
and model compatibility validation.
"""

import unittest
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from model_merger import loader
from tests.helpers import (
    create_dummy_model, create_dummy_vae, create_incompatible_model,
    cleanup_test_files
)


class TestComputeFileHash(unittest.TestCase):
    """Tests for compute_file_hash function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.console.create_progress')
    def test_hash_computed_correctly(self, mock_progress):
        """Test that SHA-256 hash is computed correctly."""
        # Create a simple test file
        test_file = self.temp_dir / "test_hash.bin"
        test_data = b"Hello, World!"
        with open(test_file, 'wb') as f:
            f.write(test_data)
        self.test_files.append(test_file)
        
        # Mock progress bar
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Compute hash
        result = loader.compute_file_hash(test_file)
        
        # Verify it matches expected SHA-256
        expected = hashlib.sha256(test_data).hexdigest()
        self.assertEqual(result, expected)
    
    @patch('model_merger.console.create_progress')
    def test_hash_is_64_characters(self, mock_progress):
        """Test that hash is 64 hex characters (SHA-256)."""
        test_file = self.temp_dir / "test_hash2.bin"
        with open(test_file, 'wb') as f:
            f.write(b"test data")
        self.test_files.append(test_file)
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        result = loader.compute_file_hash(test_file)
        
        self.assertEqual(len(result), 64)
        # Verify all characters are hex
        self.assertTrue(all(c in '0123456789abcdef' for c in result))
    
    @patch('model_merger.console.create_progress')
    def test_same_file_same_hash(self, mock_progress):
        """Test that same file produces same hash."""
        test_file = self.temp_dir / "test_hash3.bin"
        with open(test_file, 'wb') as f:
            f.write(b"consistent data")
        self.test_files.append(test_file)
        
        mock_progress.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        hash1 = loader.compute_file_hash(test_file)
        hash2 = loader.compute_file_hash(test_file)
        
        self.assertEqual(hash1, hash2)


class TestDetectPrecision(unittest.TestCase):
    """Tests for detect_precision function."""
    
    def test_detect_fp16(self):
        """Test detection of fp16 precision."""
        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.float16),
            "layer.bias": torch.randn(10, dtype=torch.float16),
        }
        result = loader.detect_precision(state_dict)
        self.assertEqual(result, "fp16")
    
    def test_detect_fp32(self):
        """Test detection of fp32 precision."""
        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.float32),
            "layer.bias": torch.randn(10, dtype=torch.float32),
        }
        result = loader.detect_precision(state_dict)
        self.assertEqual(result, "fp32")
    
    def test_detect_bf16(self):
        """Test detection of bf16 precision."""
        state_dict = {
            "layer.weight": torch.randn(10, 10, dtype=torch.bfloat16),
            "layer.bias": torch.randn(10, dtype=torch.bfloat16),
        }
        result = loader.detect_precision(state_dict)
        self.assertEqual(result, "bf16")
    
    def test_detect_unknown_for_integer_only(self):
        """Test that integer-only tensors return unknown."""
        state_dict = {
            "some_weight": torch.randint(0, 10, (10, 10)),
        }
        result = loader.detect_precision(state_dict)
        self.assertEqual(result, "unknown")
    
    def test_detect_unknown_for_empty_dict(self):
        """Test that empty dict returns unknown."""
        state_dict = {}
        result = loader.detect_precision(state_dict)
        self.assertEqual(result, "unknown")
    
    def test_detect_from_first_weight_key(self):
        """Test that precision is detected from first key containing 'weight'."""
        state_dict = {
            "position_ids": torch.arange(10),  # No 'weight' in name
            "layer1.weight": torch.randn(10, 10, dtype=torch.float16),
        }
        result = loader.detect_precision(state_dict)
        self.assertEqual(result, "fp16")


class TestLoadModel(unittest.TestCase):
    """Tests for load_model function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.loader.console')
    @patch('model_merger.loader.print_info')
    def test_load_fp32_model(self, mock_print_info, mock_console):
        """Test loading an fp32 model."""
        model_path = create_dummy_model("test_fp32.safetensors", precision="fp32")
        self.test_files.append(model_path)
        
        state_dict, metadata = loader.load_model(model_path)
        
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)
        self.assertEqual(metadata['precision'], 'fp32')
        self.assertEqual(metadata['filename'], 'test_fp32.safetensors')
    
    @patch('model_merger.loader.console')
    @patch('model_merger.loader.print_info')
    def test_load_fp16_model(self, mock_print_info, mock_console):
        """Test loading an fp16 model."""
        model_path = create_dummy_model("test_fp16.safetensors", precision="fp16")
        self.test_files.append(model_path)
        
        state_dict, metadata = loader.load_model(model_path)
        
        self.assertEqual(metadata['precision'], 'fp16')
    
    def test_load_nonexistent_file_raises(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        nonexistent = Path("tests/temp/nonexistent.safetensors")
        
        with self.assertRaises(FileNotFoundError):
            loader.load_model(nonexistent)
    
    def test_load_unsupported_format_raises(self):
        """Test that unsupported file format raises ValueError."""
        bad_file = self.temp_dir / "model.pickle"
        bad_file.touch()
        self.test_files.append(bad_file)
        
        with self.assertRaises(ValueError) as context:
            loader.load_model(bad_file)
        
        self.assertIn("Unsupported file format", str(context.exception))
    
    @patch('model_merger.loader.console')
    @patch('model_merger.loader.print_info')
    @patch('model_merger.loader.compute_file_hash')
    def test_load_with_hash_computation(self, mock_hash, mock_print_info, mock_console):
        """Test loading model with hash computation."""
        mock_hash.return_value = "abc123" * 10 + "abcd"
        
        model_path = create_dummy_model("test_hash.safetensors")
        self.test_files.append(model_path)
        
        state_dict, metadata = loader.load_model(model_path, compute_hash=True)
        
        self.assertIn('sha256', metadata)
        self.assertIn('crc32', metadata)
        mock_hash.assert_called_once()


class TestLoadVae(unittest.TestCase):
    """Tests for load_vae function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.loader.console')
    @patch('model_merger.loader.print_info')
    def test_load_vae_successfully(self, mock_print_info, mock_console):
        """Test loading a VAE file."""
        vae_path = create_dummy_vae("test_vae.safetensors", precision="fp32")
        self.test_files.append(vae_path)
        
        state_dict, metadata = loader.load_vae(vae_path)
        
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)
        self.assertEqual(metadata['precision'], 'fp32')
    
    def test_load_vae_nonexistent_raises(self):
        """Test that loading non-existent VAE raises FileNotFoundError."""
        nonexistent = Path("tests/temp/nonexistent_vae.safetensors")
        
        with self.assertRaises(FileNotFoundError):
            loader.load_vae(nonexistent)
    
    def test_load_vae_unsupported_format_raises(self):
        """Test that unsupported VAE format raises ValueError."""
        bad_file = self.temp_dir / "vae.pt"
        bad_file.touch()
        self.test_files.append(bad_file)
        
        with self.assertRaises(ValueError) as context:
            loader.load_vae(bad_file)
        
        self.assertIn("Unsupported VAE format", str(context.exception))


class TestValidateModelsCompatible(unittest.TestCase):
    """Tests for validate_models_compatible function."""
    
    def test_compatible_models(self):
        """Test that identical structure models are compatible."""
        model_a = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(10, 10),
        }
        model_b = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(10, 10),
        }
        
        reference_shapes = {k: v.shape for k, v in model_a.items()}
        reference_keys = set(model_a.keys())
        is_compatible, error_msg = loader.validate_models_compatible(
            reference_shapes, reference_keys, model_b, "Model A", "Model B"
        )
        
        self.assertTrue(is_compatible)
        self.assertIsNone(error_msg)
    
    def test_incompatible_shapes(self):
        """Test that different tensor shapes are detected as incompatible."""
        model_a = {
            "layer1.weight": torch.randn(10, 10),
        }
        model_b = {
            "layer1.weight": torch.randn(20, 20),  # Different shape
        }
        
        reference_shapes = {k: v.shape for k, v in model_a.items()}
        reference_keys = set(model_a.keys())
        is_compatible, error_msg = loader.validate_models_compatible(
            reference_shapes, reference_keys, model_b, "Model A", "Model B"
        )
        
        self.assertFalse(is_compatible)
        self.assertIn("Shape mismatch", error_msg)
    
    def test_insufficient_key_overlap(self):
        """Test that models with insufficient key overlap are incompatible."""
        model_a = {
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(10, 10),
            "layer3.weight": torch.randn(10, 10),
            "layer4.weight": torch.randn(10, 10),
            "layer5.weight": torch.randn(10, 10),
        }
        model_b = {
            "other1.weight": torch.randn(10, 10),
            "other2.weight": torch.randn(10, 10),
            "other3.weight": torch.randn(10, 10),
            "other4.weight": torch.randn(10, 10),
            "other5.weight": torch.randn(10, 10),
        }
        
        reference_shapes = {k: v.shape for k, v in model_a.items()}
        reference_keys = set(model_a.keys())
        is_compatible, error_msg = loader.validate_models_compatible(
            reference_shapes, reference_keys, model_b, "Model A", "Model B"
        )
        
        self.assertFalse(is_compatible)
        # Check for the actual error message format
        self.assertIn("5 keys", error_msg.lower())
    
    def test_skip_merge_keys_ignored(self):
        """Test that skip merge keys are not checked for shape compatibility."""
        # Position IDs is a skip key and has different shape on purpose
        model_a = {
            "layer1.weight": torch.randn(10, 10),
            "cond_stage_model.transformer.text_model.embeddings.position_ids": torch.arange(77),
        }
        model_b = {
            "layer1.weight": torch.randn(10, 10),
            "cond_stage_model.transformer.text_model.embeddings.position_ids": torch.arange(99),  # Different!
        }
        
        reference_shapes = {k: v.shape for k, v in model_a.items()}
        reference_keys = set(model_a.keys())
        is_compatible, error_msg = loader.validate_models_compatible(
            reference_shapes, reference_keys, model_b, "Model A", "Model B"
        )
        
        self.assertTrue(is_compatible)


if __name__ == '__main__':
    unittest.main()
