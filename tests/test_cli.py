"""
Tests for cli.py module.

Tests command-line interface functions.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from argparse import Namespace
import tempfile
import shutil

from model_merger import cli
from model_merger.manifest import MergeManifest, ModelEntry, OutputEntry
from tests.helpers import create_dummy_model, cleanup_test_files


class TestCmdScan(unittest.TestCase):
    """Tests for cmd_scan function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    def test_scan_nonexistent_vae_returns_error(self, mock_error, mock_header):
        """Test that scan with non-existent VAE returns error."""
        args = Namespace(
            folder=str(self.temp_dir),
            vae="/nonexistent/vae.safetensors",
            output=None,
            compute_hashes=False,
            no_equal_weights=False,
            skip_errors=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_scan(args)
        
        self.assertEqual(result, 1)
        mock_error.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.manifest_module.scan_folder')
    def test_scan_error_returns_error_code(self, mock_scan, mock_error, mock_header):
        """Test that scan errors return error code."""
        mock_scan.side_effect = ValueError("No models found")
        
        args = Namespace(
            folder=str(self.temp_dir),
            vae=None,
            output=None,
            compute_hashes=False,
            no_equal_weights=False,
            skip_errors=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_scan(args)
        
        self.assertEqual(result, 1)


class TestCmdConvert(unittest.TestCase):
    """Tests for cmd_convert function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path("tests/temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    def test_convert_nonexistent_file_returns_error(self, mock_error, mock_header):
        """Test that convert with non-existent file returns error."""
        args = Namespace(
            input="/nonexistent/model.ckpt",
            output=None,
            no_prune=False,
            compute_hash=False,
            overwrite=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_convert(args)
        
        self.assertEqual(result, 1)
        mock_error.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.converter_module.convert_to_safetensors')
    def test_convert_error_returns_error_code(self, mock_convert, mock_error, mock_header):
        """Test that conversion errors return error code."""
        mock_convert.side_effect = RuntimeError("Conversion failed")
        
        # Create a dummy input file
        input_file = self.temp_dir / "test.ckpt"
        input_file.touch()
        self.test_files.append(input_file)
        
        args = Namespace(
            input=str(input_file),
            output=None,
            no_prune=False,
            compute_hash=False,
            overwrite=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_convert(args)
        
        self.assertEqual(result, 1)
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_info')
    @patch('model_merger.cli.print_completion')
    @patch('model_merger.cli.converter_module.convert_to_safetensors')
    def test_convert_success_returns_zero(
        self, mock_convert, mock_completion, mock_info, mock_header
    ):
        """Test that successful conversion returns 0."""
        # Create a dummy input file
        input_file = self.temp_dir / "test_success.ckpt"
        input_file.touch()
        self.test_files.append(input_file)
        
        # Create the expected output file so stat() works
        output_file = self.temp_dir / "test_success.safetensors"
        output_file.write_bytes(b"dummy content")
        self.test_files.append(output_file)
        
        mock_convert.return_value = "abc123"
        
        args = Namespace(
            input=str(input_file),
            output=str(output_file),
            no_prune=False,
            compute_hash=False,
            overwrite=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_convert(args)
        
        self.assertEqual(result, 0)
        mock_convert.assert_called_once()
    
    @patch('model_merger.notifier.notify_conversion_success')
    @patch('model_merger.notifier.should_notify')
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_info')
    @patch('model_merger.cli.print_completion')
    @patch('model_merger.cli.converter_module.convert_to_safetensors')
    def test_convert_success_sends_notification(
        self, mock_convert, mock_completion, mock_info, mock_header,
        mock_should_notify, mock_notify
    ):
        """Test that successful conversion sends notification when enabled."""
        input_file = self.temp_dir / "test_notify.ckpt"
        input_file.touch()
        self.test_files.append(input_file)
        
        output_file = self.temp_dir / "test_notify.safetensors"
        output_file.write_bytes(b"dummy content" * 100)
        self.test_files.append(output_file)
        
        mock_convert.return_value = "abc123"
        mock_should_notify.return_value = True
        
        args = Namespace(
            input=str(input_file),
            output=str(output_file),
            no_prune=False,
            compute_hash=False,
            overwrite=False,
            notify=True,
            force=False
        )
        
        result = cli.cmd_convert(args)
        
        self.assertEqual(result, 0)
        mock_notify.assert_called_once()


class TestCmdMerge(unittest.TestCase):
    """Tests for cmd_merge function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.cli.print_error')
    def test_merge_nonexistent_manifest_returns_error(self, mock_error):
        """Test that merge with non-existent manifest returns error."""
        args = Namespace(
            manifest="/nonexistent/manifest.json",
            overwrite=False,
            device=None,
            no_prune=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_merge(args)
        
        self.assertEqual(result, 1)
        mock_error.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.manifest_module.MergeManifest.load')
    def test_merge_load_error_returns_error_code(self, mock_load, mock_error, mock_header):
        """Test that manifest load errors return error code."""
        mock_load.side_effect = ValueError("Invalid manifest")
        
        # Create a dummy manifest file
        manifest_file = self.temp_dir / "manifest.json"
        manifest_file.touch()
        self.test_files.append(manifest_file)
        
        args = Namespace(
            manifest=str(manifest_file),
            overwrite=False,
            device=None,
            no_prune=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_merge(args)
        
        self.assertEqual(result, 1)
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_manifest_summary')
    @patch('model_merger.cli.console')
    @patch('model_merger.cli.print_success')
    @patch('model_merger.cli.print_validation_issues')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.manifest_module.MergeManifest.load')
    @patch('model_merger.cli.manifest_module.validate_manifest')
    def test_merge_with_validation_errors_returns_error(
        self, mock_validate, mock_load, mock_error, mock_issues,
        mock_success, mock_console, mock_summary, mock_header
    ):
        """Test that validation errors cause merge to fail."""
        # Create manifest with missing model
        models = [ModelEntry(path="missing.safetensors", weight=1.0, architecture="SDXL", index=0)]
        mock_load.return_value = MergeManifest(models=models)
        mock_validate.return_value = ["Model file not found: missing.safetensors"]
        
        manifest_file = self.temp_dir / "manifest.json"
        manifest_file.touch()
        self.test_files.append(manifest_file)
        
        args = Namespace(
            manifest=str(manifest_file),
            overwrite=False,
            device=None,
            no_prune=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_merge(args)
        
        self.assertEqual(result, 1)


class TestMain(unittest.TestCase):
    """Tests for main function."""
    
    @patch('model_merger.cli.argparse.ArgumentParser.parse_args')
    def test_no_command_prints_help(self, mock_parse):
        """Test that no command shows help."""
        mock_parse.return_value = Namespace(command=None)
        
        result = cli.main()
        
        self.assertEqual(result, 1)
    
    @patch('model_merger.cli.cmd_scan')
    @patch('model_merger.cli.argparse.ArgumentParser.parse_args')
    def test_scan_command_dispatches(self, mock_parse, mock_scan):
        """Test that scan command dispatches to cmd_scan."""
        mock_parse.return_value = Namespace(
            command='scan',
            folder='/some/folder',
            vae=None,
            output=None,
            compute_hashes=False,
            no_equal_weights=False,
            skip_errors=False,
        )
        mock_scan.return_value = 0
        
        result = cli.main()
        
        mock_scan.assert_called_once()
        self.assertEqual(result, 0)
    
    @patch('model_merger.cli.cmd_merge')
    @patch('model_merger.cli.argparse.ArgumentParser.parse_args')
    def test_merge_command_dispatches(self, mock_parse, mock_merge):
        """Test that merge command dispatches to cmd_merge."""
        mock_parse.return_value = Namespace(
            command='merge',
            manifest='/some/manifest.json',
            overwrite=False,
            device=None,
            no_prune=False,
            notify=False,
            force=False
        )
        mock_merge.return_value = 0
        
        result = cli.main()
        
        mock_merge.assert_called_once()
        self.assertEqual(result, 0)
    
    @patch('model_merger.cli.cmd_convert')
    @patch('model_merger.cli.argparse.ArgumentParser.parse_args')
    def test_convert_command_dispatches(self, mock_parse, mock_convert):
        """Test that convert command dispatches to cmd_convert."""
        mock_parse.return_value = Namespace(
            command='convert',
            input='/some/model.ckpt',
            output=None,
            no_prune=False,
            compute_hash=False,
            overwrite=False,
            notify=False,
            force=False
        )
        mock_convert.return_value = 0
        
        result = cli.main()
        
        mock_convert.assert_called_once()
        self.assertEqual(result, 0)
    
    @patch('model_merger.cli.cmd_verify')
    @patch('model_merger.cli.argparse.ArgumentParser.parse_args')
    def test_verify_command_dispatches(self, mock_parse, mock_verify):
        """Test that verify command dispatches to cmd_verify."""
        mock_parse.return_value = Namespace(
            command='verify',   
            original='/some/original.ckpt',
            converted='/some/converted.safetensors',
            verbose=False,
            notify=False,
            force=False
        )
        mock_verify.return_value = 0
        
        result = cli.main()
        
        mock_verify.assert_called_once()
        self.assertEqual(result, 0)


class TestCmdVerify(unittest.TestCase):
    """Tests for cmd_verify function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_files = []
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test files."""
        cleanup_test_files(self.test_files)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    def test_verify_nonexistent_original_returns_error(self, mock_error, mock_header):
        """Test that verify with non-existent original file returns error."""
        args = Namespace(
            original="/nonexistent/original.ckpt",
            converted=str(self.temp_dir / "converted.safetensors"),
            verbose=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_verify(args)
        
        self.assertEqual(result, 1)
        mock_error.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    def test_verify_nonexistent_converted_returns_error(self, mock_error, mock_header):
        """Test that verify with non-existent converted file returns error."""
        # Create only the original file
        original_file = self.temp_dir / "original.ckpt"
        original_file.touch()
        self.test_files.append(original_file)
        
        args = Namespace(
            original=str(original_file),
            converted="/nonexistent/converted.safetensors",
            verbose=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_verify(args)
        
        self.assertEqual(result, 1)
        mock_error.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.print_success')
    @patch('model_merger.cli.console')
    @patch('model_merger.cli.verifier_module.verify_conversion')
    def test_verify_success_returns_zero(
        self, mock_verify, mock_console, mock_success, mock_error, mock_header
    ):
        """Test that successful verification returns 0."""
        # Create both files
        original_file = self.temp_dir / "original.ckpt"
        converted_file = self.temp_dir / "converted.safetensors"
        original_file.touch()
        converted_file.touch()
        self.test_files.extend([original_file, converted_file])
        
        # Mock successful verification
        mock_verify.return_value = True
        
        args = Namespace(
            original=str(original_file),
            converted=str(converted_file),
            verbose=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_verify(args)
        
        self.assertEqual(result, 0)
        mock_success.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.console')
    @patch('model_merger.cli.verifier_module.verify_conversion')
    def test_verify_failure_returns_one(
        self, mock_verify, mock_console, mock_error, mock_header
    ):
        """Test that failed verification returns 1."""
        # Create both files
        original_file = self.temp_dir / "original.ckpt"
        converted_file = self.temp_dir / "converted.safetensors"
        original_file.touch()
        converted_file.touch()
        self.test_files.extend([original_file, converted_file])
        
        # Mock failed verification
        mock_verify.return_value = False
        
        args = Namespace(
            original=str(original_file),
            converted=str(converted_file),
            verbose=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_verify(args)
        
        self.assertEqual(result, 1)
        mock_error.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.verifier_module.verify_conversion')
    def test_verify_exception_returns_error(self, mock_verify, mock_error, mock_header):
        """Test that verification exception returns error code."""
        # Create both files
        original_file = self.temp_dir / "original.ckpt"
        converted_file = self.temp_dir / "converted.safetensors"
        original_file.touch()
        converted_file.touch()
        self.test_files.extend([original_file, converted_file])
        
        # Mock verification exception
        mock_verify.side_effect = RuntimeError("Verification error")
        
        args = Namespace(
            original=str(original_file),
            converted=str(converted_file),
            verbose=False,
            notify=False,
            force=False
        )
        
        result = cli.cmd_verify(args)
        
        self.assertEqual(result, 1)
        mock_error.assert_called()
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_success')
    @patch('model_merger.cli.console')
    @patch('model_merger.cli.verifier_module.verify_conversion')
    def test_verify_verbose_flag_passed(
        self, mock_verify, mock_console, mock_success, mock_header
    ):
        """Test that verbose flag is passed to verify_conversion."""
        # Create both files
        original_file = self.temp_dir / "original.ckpt"
        converted_file = self.temp_dir / "converted.safetensors"
        original_file.touch()
        converted_file.touch()
        self.test_files.extend([original_file, converted_file])
        
        mock_verify.return_value = True
        
        args = Namespace(
            original=str(original_file),
            converted=str(converted_file),
            verbose=True,
            notify=False,
            force=False
        )
        
        cli.cmd_verify(args)
        
        # Check verbose was passed
        mock_verify.assert_called_once()
        call_kwargs = mock_verify.call_args[1]
        self.assertTrue(call_kwargs.get('verbose'))


class TestCliOverrides(unittest.TestCase):
    """Tests for CLI override behavior in cmd_merge."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_manifest_summary')
    @patch('model_merger.cli.console')
    @patch('model_merger.cli.print_success')
    @patch('model_merger.cli.print_validation_issues')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.manifest_module.MergeManifest.load')
    @patch('model_merger.cli.manifest_module.validate_manifest')
    def test_overwrite_flag_overrides_manifest(
        self, mock_validate, mock_load, mock_error, mock_issues,
        mock_success, mock_console, mock_summary, mock_header
    ):
        """Test that --overwrite flag overrides manifest setting."""
        models = [ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)]
        manifest = MergeManifest(models=models, overwrite=False)
        mock_load.return_value = manifest
        mock_validate.return_value = ["Model file not found"]  # Force early exit
        
        manifest_file = self.temp_dir / "manifest.json"
        manifest_file.touch()
        
        args = Namespace(
            manifest=str(manifest_file),
            overwrite=True,  # Override!
            device=None,
            no_prune=False,
            notify=False,
            force=False
        )
        
        cli.cmd_merge(args)
        
        # Manifest should have been updated
        self.assertTrue(manifest.overwrite)
    
    @patch('model_merger.cli.check_cuda_availability')
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_manifest_summary')
    @patch('model_merger.cli.console')
    @patch('model_merger.cli.print_success')
    @patch('model_merger.cli.print_validation_issues')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.manifest_module.MergeManifest.load')
    @patch('model_merger.cli.manifest_module.validate_manifest')
    def test_device_flag_overrides_manifest(
        self, mock_validate, mock_load, mock_error, mock_issues,
        mock_success, mock_console, mock_summary, mock_header, mock_cuda_check
    ):
        """Test that --device flag overrides manifest setting."""
        # Mock check_cuda_availability to return the requested device
        mock_cuda_check.side_effect = lambda x: x
        
        models = [ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)]
        manifest = MergeManifest(models=models, device='cpu')
        mock_load.return_value = manifest
        mock_validate.return_value = ["Model file not found"]
        
        manifest_file = self.temp_dir / "manifest.json"
        manifest_file.touch()
        
        args = Namespace(
            manifest=str(manifest_file),
            overwrite=False,
            device='cuda',  # Override!
            no_prune=False,
            notify=False,
            force=False
        )
        
        cli.cmd_merge(args)
        
        self.assertEqual(manifest.device, 'cuda')
    
    @patch('model_merger.cli.print_header')
    @patch('model_merger.cli.print_manifest_summary')
    @patch('model_merger.cli.console')
    @patch('model_merger.cli.print_success')
    @patch('model_merger.cli.print_validation_issues')
    @patch('model_merger.cli.print_error')
    @patch('model_merger.cli.manifest_module.MergeManifest.load')
    @patch('model_merger.cli.manifest_module.validate_manifest')
    def test_no_prune_flag_overrides_manifest(
        self, mock_validate, mock_load, mock_error, mock_issues,
        mock_success, mock_console, mock_summary, mock_header
    ):
        """Test that --no-prune flag overrides manifest setting."""
        models = [ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)]
        manifest = MergeManifest(models=models, prune=True)
        mock_load.return_value = manifest
        mock_validate.return_value = ["Model file not found"]
        
        manifest_file = self.temp_dir / "manifest.json"
        manifest_file.touch()
        
        args = Namespace(
            manifest=str(manifest_file),
            overwrite=False,
            device=None,
            no_prune=True,  # Override!
            notify=False,
            force=False
        )
        
        cli.cmd_merge(args)
        
        self.assertFalse(manifest.prune)


class TestCheckCudaAvailability(unittest.TestCase):
    """Tests for check_cuda_availability function."""
    
    def test_cpu_requested_returns_cpu(self):
        """Test that requesting CPU always returns CPU."""
        result = cli.check_cuda_availability('cpu')
        self.assertEqual(result, 'cpu')
    
    @patch('torch.cuda.is_available')
    @patch('model_merger.cli.console')
    def test_cuda_requested_but_unavailable_returns_cpu(self, mock_console, mock_is_available):
        """Test that requesting CUDA when unavailable falls back to CPU."""
        mock_is_available.return_value = False
        
        result = cli.check_cuda_availability('cuda')
        
        self.assertEqual(result, 'cpu')
        mock_is_available.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_cuda_requested_and_available_returns_cuda(self, mock_is_available):
        """Test that requesting CUDA when available returns CUDA."""
        mock_is_available.return_value = True
        
        result = cli.check_cuda_availability('cuda')
        
        self.assertEqual(result, 'cuda')
        mock_is_available.assert_called_once()


class TestPrintDeviceInfo(unittest.TestCase):
    """Tests for print_device_info function."""
    
    @patch('torch.cuda.is_available')
    @patch('model_merger.cli.console')
    def test_print_device_info_cpu_no_cuda(self, mock_console, mock_is_available):
        """Test printing device info when using CPU and no CUDA available."""
        mock_is_available.return_value = False
        
        cli.print_device_info('cpu')
        
        # Should have called console.print multiple times
        self.assertTrue(mock_console.print.called)
    
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.is_available')
    @patch('model_merger.cli.console')
    def test_print_device_info_cuda_available(self, mock_console, mock_is_available, mock_get_name):
        """Test printing device info when CUDA is available."""
        mock_is_available.return_value = True
        mock_get_name.return_value = "NVIDIA GeForce RTX 3090"
        
        cli.print_device_info('cuda')
        
        # Should have called console.print and get_device_name
        self.assertTrue(mock_console.print.called)
        mock_get_name.assert_called_once_with(0)


if __name__ == '__main__':
    unittest.main()
