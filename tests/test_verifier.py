"""
Tests for verifier.py module.

Tests model conversion verification functions.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch
from safetensors.torch import save_file

from model_merger import verifier
from tests.helpers import create_dummy_model, create_dummy_checkpoint, cleanup_test_files


class TestLoadForVerification(unittest.TestCase):
    """Tests for load_for_verification function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.verifier.console')
    def test_load_safetensors_file(self, mock_console):
        """Test loading a safetensors file for verification."""
        # Arrange
        model_path = create_dummy_model("test_verify.safetensors", temp_dir=self.temp_dir)
        self.test_files.append(model_path)
        
        # Act
        result = verifier.load_for_verification(model_path)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        self.assertIn("model.diffusion_model.layer1.weight", result)
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises FileNotFoundError."""
        non_existent = Path("/nonexistent/model.safetensors")
        
        with self.assertRaises(FileNotFoundError):
            verifier.load_for_verification(non_existent)
    
    @patch('model_merger.verifier.console')
    def test_load_unsupported_format_raises_error(self, mock_console):
        """Test that loading unsupported format raises ValueError."""
        # Create a file with unsupported extension
        unsupported_file = self.temp_dir / "model.txt"
        unsupported_file.write_text("not a model")
        self.test_files.append(unsupported_file)
        
        with self.assertRaises(ValueError) as context:
            verifier.load_for_verification(unsupported_file)
        
        self.assertIn("Unsupported", str(context.exception))
    
    @patch('model_merger.verifier.console')
    @patch('model_merger.verifier.load_file')
    def test_load_legacy_ckpt_format(self, mock_load_file, mock_console):
        """Test loading a legacy checkpoint format."""
        # Create a ckpt file
        ckpt_path = create_dummy_checkpoint("test_verify.ckpt", temp_dir=self.temp_dir)
        self.test_files.append(ckpt_path)
        
        # Act - load it (it will use the actual converter module)
        result = verifier.load_for_verification(ckpt_path)
        
        # Assert - should return extracted state dict
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
    
    @patch('model_merger.verifier.console')
    def test_load_ckpt_returns_state_dict(self, mock_console):
        """Test that loading ckpt returns extracted state dict."""
        # Create a ckpt file with known state dict
        ckpt_path = create_dummy_checkpoint(
            "test_extract.ckpt",
            temp_dir=self.temp_dir,
            format_type="wrapped"
        )
        self.test_files.append(ckpt_path)
        
        # Act
        result = verifier.load_for_verification(ckpt_path)
        
        # Assert - should have the model keys (extracted from wrapped format)
        self.assertIsInstance(result, dict)


class TestCompareKeySets(unittest.TestCase):
    """Tests for compare_key_sets function."""
    
    def test_identical_key_sets_match(self):
        """Test that identical key sets return match."""
        keys1 = {"a", "b", "c"}
        keys2 = {"a", "b", "c"}
        
        match, missing, extra = verifier.compare_key_sets(keys1, keys2)
        
        self.assertTrue(match)
        self.assertEqual(missing, set())
        self.assertEqual(extra, set())
    
    def test_missing_keys_detected(self):
        """Test that missing keys are detected."""
        original = {"a", "b", "c"}
        converted = {"a", "b"}  # c is missing
        
        match, missing, extra = verifier.compare_key_sets(original, converted)
        
        self.assertFalse(match)
        self.assertEqual(missing, {"c"})
        self.assertEqual(extra, set())
    
    def test_extra_keys_detected(self):
        """Test that extra keys are detected."""
        original = {"a", "b"}
        converted = {"a", "b", "c"}  # c is extra
        
        match, missing, extra = verifier.compare_key_sets(original, converted)
        
        self.assertFalse(match)
        self.assertEqual(missing, set())
        self.assertEqual(extra, {"c"})
    
    def test_both_missing_and_extra_keys(self):
        """Test detection of both missing and extra keys."""
        original = {"a", "b", "c"}
        converted = {"a", "b", "d"}  # c missing, d extra
        
        match, missing, extra = verifier.compare_key_sets(original, converted)
        
        self.assertFalse(match)
        self.assertEqual(missing, {"c"})
        self.assertEqual(extra, {"d"})
    
    def test_empty_sets_match(self):
        """Test that empty sets match."""
        match, missing, extra = verifier.compare_key_sets(set(), set())
        
        self.assertTrue(match)
        self.assertEqual(missing, set())
        self.assertEqual(extra, set())


class TestCompareTensors(unittest.TestCase):
    """Tests for compare_tensors function."""
    
    def test_identical_tensors_match(self):
        """Test that identical tensors match."""
        tensor = torch.randn(10, 10)
        
        match, error = verifier.compare_tensors("test_key", tensor, tensor.clone())
        
        self.assertTrue(match)
        self.assertIsNone(error)
    
    def test_different_values_detected(self):
        """Test that different tensor values are detected."""
        tensor1 = torch.zeros(10, 10)
        tensor2 = torch.ones(10, 10)
        
        match, error = verifier.compare_tensors("test_key", tensor1, tensor2)
        
        self.assertFalse(match)
        self.assertIn("differ", error)
    
    def test_shape_mismatch_detected(self):
        """Test that shape mismatch is detected."""
        tensor1 = torch.randn(10, 10)
        tensor2 = torch.randn(20, 20)
        
        match, error = verifier.compare_tensors("test_key", tensor1, tensor2)
        
        self.assertFalse(match)
        self.assertIn("Shape mismatch", error)
    
    def test_non_tensor_original_detected(self):
        """Test that non-tensor original is detected."""
        non_tensor = [1, 2, 3]
        tensor = torch.randn(3)
        
        match, error = verifier.compare_tensors("test_key", non_tensor, tensor)
        
        self.assertFalse(match)
        self.assertIn("Original", error)
        self.assertIn("not a tensor", error)
    
    def test_non_tensor_converted_detected(self):
        """Test that non-tensor converted is detected."""
        tensor = torch.randn(3)
        non_tensor = "not a tensor"
        
        match, error = verifier.compare_tensors("test_key", tensor, non_tensor)
        
        self.assertFalse(match)
        self.assertIn("Converted", error)
        self.assertIn("not a tensor", error)
    
    def test_tolerance_respected(self):
        """Test that tolerance parameters are respected."""
        tensor1 = torch.ones(10, 10)
        # Add small difference within tolerance
        tensor2 = tensor1 + 1e-9
        
        match, error = verifier.compare_tensors(
            "test_key", tensor1, tensor2, rtol=1e-5, atol=1e-8
        )
        
        self.assertTrue(match)
        self.assertIsNone(error)
    
    def test_difference_exceeds_tolerance(self):
        """Test that differences exceeding tolerance are caught."""
        tensor1 = torch.ones(10, 10)
        # Add larger difference
        tensor2 = tensor1 + 0.1
        
        match, error = verifier.compare_tensors(
            "test_key", tensor1, tensor2, rtol=1e-5, atol=1e-8
        )
        
        self.assertFalse(match)
        self.assertIsNotNone(error)


class TestVerifyConversion(unittest.TestCase):
    """Tests for verify_conversion function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.verifier.console')
    @patch('model_merger.verifier.print_success')
    @patch('model_merger.verifier.print_error')
    @patch('model_merger.verifier.create_progress')
    def test_verify_identical_models_passes(
        self, mock_progress, mock_error, mock_success, mock_console
    ):
        """Test verification passes for identical models."""
        # Create two identical models
        state_dict = {
            "layer.weight": torch.randn(10, 10),
            "layer.bias": torch.randn(10),
        }
        
        path1 = self.temp_dir / "model1.safetensors"
        path2 = self.temp_dir / "model2.safetensors"
        save_file(state_dict, str(path1))
        save_file(state_dict, str(path2))
        self.test_files.extend([path1, path2])
        
        # Mock progress context
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Act
        result = verifier.verify_conversion(path1, path2)
        
        # Assert
        self.assertTrue(result)
    
    @patch('model_merger.verifier.console')
    @patch('model_merger.verifier.print_success')
    @patch('model_merger.verifier.print_error')
    @patch('model_merger.verifier.print_warning')
    @patch('model_merger.verifier.print_info')
    @patch('model_merger.verifier.create_progress')
    def test_verify_different_models_fails(
        self, mock_progress, mock_info, mock_warning, mock_error, mock_success, mock_console
    ):
        """Test verification fails for different models."""
        # Create two different models
        path1 = self.temp_dir / "diff1.safetensors"
        path2 = self.temp_dir / "diff2.safetensors"
        
        save_file({"layer.weight": torch.zeros(10, 10)}, str(path1))
        save_file({"layer.weight": torch.ones(10, 10)}, str(path2))
        self.test_files.extend([path1, path2])
        
        # Mock progress context
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Act
        result = verifier.verify_conversion(path1, path2)
        
        # Assert
        self.assertFalse(result)
    
    @patch('model_merger.verifier.console')
    @patch('model_merger.verifier.print_error')
    def test_verify_missing_file_fails(self, mock_error, mock_console):
        """Test verification fails for missing file."""
        # Create only one model
        path1 = create_dummy_model("exists.safetensors", temp_dir=self.temp_dir)
        self.test_files.append(path1)
        path2 = self.temp_dir / "nonexistent.safetensors"
        
        # Act
        result = verifier.verify_conversion(path1, path2)
        
        # Assert
        self.assertFalse(result)
        mock_error.assert_called()
    
    @patch('model_merger.verifier.console')
    @patch('model_merger.verifier.print_success')
    @patch('model_merger.verifier.print_error')
    @patch('model_merger.verifier.print_warning')
    @patch('model_merger.verifier.print_info')
    @patch('model_merger.verifier.create_progress')
    def test_verify_missing_keys_fails(
        self, mock_progress, mock_info, mock_warning, mock_error, mock_success, mock_console
    ):
        """Test verification fails when keys are missing."""
        # Create models with different keys
        path1 = self.temp_dir / "full.safetensors"
        path2 = self.temp_dir / "partial.safetensors"
        
        save_file({
            "layer1.weight": torch.randn(10, 10),
            "layer2.weight": torch.randn(10, 10),
        }, str(path1))
        save_file({
            "layer1.weight": torch.randn(10, 10),
            # layer2 is missing
        }, str(path2))
        self.test_files.extend([path1, path2])
        
        # Mock progress context
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Act
        result = verifier.verify_conversion(path1, path2)
        
        # Assert
        self.assertFalse(result)
    
    @patch('model_merger.verifier.console')
    @patch('model_merger.verifier.print_success')
    @patch('model_merger.verifier.print_error')
    @patch('model_merger.verifier.create_progress')
    def test_verify_with_verbose_option(
        self, mock_progress, mock_error, mock_success, mock_console
    ):
        """Test verification with verbose option enabled."""
        # Create two identical models
        state_dict = {"layer.weight": torch.randn(10, 10)}
        
        path1 = self.temp_dir / "v1.safetensors"
        path2 = self.temp_dir / "v2.safetensors"
        save_file(state_dict, str(path1))
        save_file(state_dict, str(path2))
        self.test_files.extend([path1, path2])
        
        # Mock progress context
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Act
        result = verifier.verify_conversion(path1, path2, verbose=True)
        
        # Assert
        self.assertTrue(result)
    
    @patch('model_merger.verifier.console')
    @patch('model_merger.verifier.print_success')
    @patch('model_merger.verifier.print_error')
    @patch('model_merger.verifier.print_warning')
    @patch('model_merger.verifier.print_info')
    @patch('model_merger.verifier.create_progress')
    def test_verify_handles_module_prefix_mismatch(
        self, mock_progress, mock_info, mock_warning, mock_error, mock_success, mock_console
    ):
        """Test that module. prefix differences are handled."""
        # Create models with module prefix difference
        path1 = self.temp_dir / "with_module.safetensors"
        path2 = self.temp_dir / "without_module.safetensors"
        
        tensor = torch.randn(10, 10)
        save_file({"module.layer.weight": tensor}, str(path1))
        save_file({"layer.weight": tensor}, str(path2))
        self.test_files.extend([path1, path2])
        
        # Mock progress context
        mock_progress_instance = MagicMock()
        mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress_instance)
        mock_progress.return_value.__exit__ = MagicMock(return_value=False)
        
        # Act
        result = verifier.verify_conversion(path1, path2)
        
        # Assert - should pass because module. prefix stripping is handled
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
