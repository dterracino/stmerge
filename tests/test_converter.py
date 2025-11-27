"""
Tests for converter.py module.

Tests checkpoint format conversion from legacy formats to safetensors.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import torch

from model_merger import converter
from tests.helpers import create_dummy_checkpoint, cleanup_test_files


class TestDetectCheckpointFormat(unittest.TestCase):
    """Tests for detect_checkpoint_format function."""
    
    def test_bare_format_detection(self):
        """Test detection of bare state dict format."""
        checkpoint = {
            "model.layer1.weight": torch.randn(10, 10),
            "model.layer1.bias": torch.randn(10),
            "model.layer2.weight": torch.randn(10, 10),
        }
        
        result = converter.detect_checkpoint_format(checkpoint)
        
        self.assertEqual(result, "bare")
    
    def test_wrapped_format_detection(self):
        """Test detection of wrapped format with state_dict key."""
        checkpoint = {
            "state_dict": {
                "layer.weight": torch.randn(10, 10),
            },
            "optimizer_state": {},
            "epoch": 10,
        }
        
        result = converter.detect_checkpoint_format(checkpoint)
        
        self.assertEqual(result, "wrapped")
    
    def test_nested_format_detection(self):
        """Test detection of nested format with model key."""
        checkpoint = {
            "model": {
                "layer.weight": torch.randn(10, 10),
            },
            "epoch": 10,
        }
        
        result = converter.detect_checkpoint_format(checkpoint)
        
        self.assertEqual(result, "nested")
    
    def test_unknown_format_detection(self):
        """Test detection of unknown format."""
        checkpoint = {
            "random_key": "random_value",
            "another_key": 123,
        }
        
        result = converter.detect_checkpoint_format(checkpoint)
        
        self.assertEqual(result, "unknown")
    
    def test_empty_checkpoint(self):
        """Test handling of empty checkpoint."""
        checkpoint = {}
        
        result = converter.detect_checkpoint_format(checkpoint)
        
        # Empty dict returns 'bare' because there are no keys to check
        self.assertEqual(result, "bare")


class TestExtractStateDict(unittest.TestCase):
    """Tests for extract_state_dict function."""
    
    @patch('model_merger.converter.console')
    def test_extract_from_bare(self, mock_console):
        """Test extraction from bare state dict format."""
        state_dict = {
            "model.layer1.weight": torch.randn(10, 10),
            "model.layer1.bias": torch.randn(10),
        }
        
        result = converter.extract_state_dict(state_dict)
        
        self.assertEqual(len(result), 2)
        self.assertIn("model.layer1.weight", result)
    
    @patch('model_merger.converter.console')
    def test_extract_from_wrapped(self, mock_console):
        """Test extraction from wrapped format."""
        checkpoint = {
            "state_dict": {
                "layer.weight": torch.randn(10, 10),
            },
            "optimizer_state": {},
        }
        
        result = converter.extract_state_dict(checkpoint)
        
        self.assertIn("layer.weight", result)
    
    @patch('model_merger.converter.console')
    def test_extract_from_nested(self, mock_console):
        """Test extraction from nested format."""
        checkpoint = {
            "model": {
                "layer.weight": torch.randn(10, 10),
            },
        }
        
        result = converter.extract_state_dict(checkpoint)
        
        self.assertIn("layer.weight", result)
    
    @patch('model_merger.converter.console')
    def test_extract_from_double_nested(self, mock_console):
        """Test extraction from double-nested format (model.state_dict)."""
        checkpoint = {
            "model": {
                "state_dict": {
                    "layer.weight": torch.randn(10, 10),
                }
            },
        }
        
        result = converter.extract_state_dict(checkpoint)
        
        self.assertIn("layer.weight", result)
    
    def test_unknown_format_raises(self):
        """Test that unknown format raises ValueError."""
        checkpoint = {
            "unknown_key": "value",
            "another_unknown": 123,
        }
        
        with self.assertRaises(ValueError) as context:
            converter.extract_state_dict(checkpoint)
        
        self.assertIn("Cannot find tensors in checkpoint!", str(context.exception))
    
    @patch('model_merger.converter.console')
    def test_invalid_state_dict_type_raises(self, mock_console):
        """Test that invalid state_dict type raises ValueError."""
        checkpoint = {
            "state_dict": "not a dict",
        }
        
        with self.assertRaises(ValueError):
            converter.extract_state_dict(checkpoint)


class TestLoadCheckpoint(unittest.TestCase):
    """Tests for load_checkpoint function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    def test_nonexistent_file_raises(self):
        """Test that non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            converter.load_checkpoint(Path("nonexistent.ckpt"))
    
    def test_unsupported_format_raises(self):
        """Test that unsupported format raises ValueError."""
        bad_file = self.temp_dir / "model.txt"
        bad_file.touch()
        self.test_files.append(bad_file)
        
        with self.assertRaises(ValueError) as context:
            converter.load_checkpoint(bad_file)
        
        self.assertIn("Unsupported file format", str(context.exception))
    
    @patch('model_merger.converter.console')
    def test_load_bare_checkpoint(self, mock_console):
        """Test loading bare format checkpoint."""
        checkpoint_path = create_dummy_checkpoint("bare_test.ckpt", format_type="bare")
        self.test_files.append(checkpoint_path)
        
        state_dict, metadata = converter.load_checkpoint(checkpoint_path)
        
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)
        self.assertEqual(metadata['filename'], 'bare_test.ckpt')
    
    @patch('model_merger.converter.console')
    def test_load_wrapped_checkpoint(self, mock_console):
        """Test loading wrapped format checkpoint."""
        checkpoint_path = create_dummy_checkpoint("wrapped_test.pt", format_type="wrapped")
        self.test_files.append(checkpoint_path)
        
        state_dict, metadata = converter.load_checkpoint(checkpoint_path)
        
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)
    
    @patch('model_merger.converter.console')
    def test_load_nested_checkpoint(self, mock_console):
        """Test loading nested format checkpoint."""
        checkpoint_path = create_dummy_checkpoint("nested_test.pth", format_type="nested")
        self.test_files.append(checkpoint_path)
        
        state_dict, metadata = converter.load_checkpoint(checkpoint_path)
        
        self.assertIsInstance(state_dict, dict)
        self.assertGreater(len(state_dict), 0)


class TestConvertToSafetensors(unittest.TestCase):
    """Tests for convert_to_safetensors function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    def test_nonexistent_input_raises(self):
        """Test that non-existent input raises exception."""
        with self.assertRaises(Exception):
            converter.convert_to_safetensors(Path("nonexistent.ckpt"))
    
    @patch('model_merger.converter.verify_conversion')
    @patch('model_merger.converter.console')
    @patch('model_merger.saver.save_model')
    def test_conversion_creates_safetensors(self, mock_save, mock_console, mock_verify):
        """Test that conversion creates safetensors file."""
        mock_save.return_value = "abc123" * 10 + "abcd"
        
        checkpoint_path = create_dummy_checkpoint("convert_test.ckpt", format_type="bare")
        self.test_files.append(checkpoint_path)
        
        result = converter.convert_to_safetensors(checkpoint_path)
        
        # Verify save_model was called
        mock_save.assert_called_once()
    
    @patch('model_merger.converter.verify_conversion')
    @patch('model_merger.converter.console')
    @patch('model_merger.saver.save_model')
    def test_default_output_path(self, mock_save, mock_console, mock_verify):
        """Test that default output path has .safetensors extension."""
        mock_save.return_value = "def456" * 10 + "defg"
        
        checkpoint_path = create_dummy_checkpoint("default_out.ckpt", format_type="bare")
        self.test_files.append(checkpoint_path)
        
        converter.convert_to_safetensors(checkpoint_path)
        
        # Check the output path passed to save_model
        call_args = mock_save.call_args
        output_path = call_args[1]['output_path']
        self.assertTrue(str(output_path).endswith('.safetensors'))
    
    @patch('model_merger.converter.verify_conversion')
    @patch('model_merger.converter.console')
    @patch('model_merger.saver.save_model')
    def test_custom_output_path(self, mock_save, mock_console, mock_verify):
        """Test conversion with custom output path."""
        mock_save.return_value = "ghi789" * 10 + "ghij"
        
        checkpoint_path = create_dummy_checkpoint("custom_out.ckpt", format_type="bare")
        output_path = self.temp_dir / "custom_output.safetensors"
        self.test_files.extend([checkpoint_path, output_path])
        
        converter.convert_to_safetensors(checkpoint_path, output_path=output_path)
        
        call_args = mock_save.call_args
        actual_output = call_args[1]['output_path']
        self.assertEqual(Path(actual_output), output_path)
    
    @patch('model_merger.converter.verify_conversion')
    @patch('model_merger.converter.console')
    @patch('model_merger.saver.save_model')
    def test_prune_enabled_by_default(self, mock_save, mock_console, mock_verify):
        """Test that pruning is enabled by default."""
        mock_save.return_value = "jkl012" * 10 + "jklm"
        
        checkpoint_path = create_dummy_checkpoint("prune_test.ckpt", format_type="wrapped")
        self.test_files.append(checkpoint_path)
        
        converter.convert_to_safetensors(checkpoint_path)
        
        # save_model should have been called
        mock_save.assert_called_once()
        # The state_dict passed should have pruned keys
        call_args = mock_save.call_args
        state_dict = call_args[1]['state_dict']
        # Only model keys should remain after pruning
        for key in state_dict.keys():
            self.assertFalse(key.startswith("optimizer"))
    
    @patch('model_merger.converter.verify_conversion')
    @patch('model_merger.converter.console')
    @patch('model_merger.saver.save_model')
    def test_no_prune_option(self, mock_save, mock_console, mock_verify):
        """Test conversion without pruning."""
        mock_save.return_value = "mno345" * 10 + "mnop"
        
        checkpoint_path = create_dummy_checkpoint("no_prune.ckpt", format_type="bare")
        self.test_files.append(checkpoint_path)
        
        converter.convert_to_safetensors(checkpoint_path, prune=False)
        
        mock_save.assert_called_once()
    
    def test_output_exists_without_overwrite_raises(self):
        """Test that existing output without overwrite flag raises."""
        checkpoint_path = create_dummy_checkpoint("overwrite_test.ckpt", format_type="bare")
        output_path = self.temp_dir / "overwrite_test.safetensors"
        output_path.touch()  # Create existing file
        self.test_files.extend([checkpoint_path, output_path])
        
        with self.assertRaises(FileExistsError):
            converter.convert_to_safetensors(checkpoint_path, output_path=output_path)


class TestLegacyFormats(unittest.TestCase):
    """Tests for legacy format constants."""
    
    def test_supported_formats(self):
        """Test that all expected legacy formats are supported."""
        self.assertIn('.ckpt', converter.LEGACY_FORMATS)
        self.assertIn('.pt', converter.LEGACY_FORMATS)
        self.assertIn('.pth', converter.LEGACY_FORMATS)
        self.assertIn('.bin', converter.LEGACY_FORMATS)
    
    def test_format_count(self):
        """Test that we support exactly 4 legacy formats."""
        self.assertEqual(len(converter.LEGACY_FORMATS), 4)


if __name__ == '__main__':
    unittest.main()
