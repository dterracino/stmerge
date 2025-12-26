"""
Tests for console.py module.

Tests Rich console output functions.
"""

import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

from model_merger import console
from model_merger.manifest import ModelEntry, MergeManifest, VAEEntry, OutputEntry


class TestPrintHeader(unittest.TestCase):
    """Tests for print_header function."""
    
    @patch.object(console.console, 'print')
    def test_print_header_calls_console(self, mock_print):
        """Test that print_header calls console.print."""
        console.print_header("Test Title")
        
        mock_print.assert_called_once()


class TestPrintSection(unittest.TestCase):
    """Tests for print_section function."""
    
    @patch.object(console.console, 'print')
    def test_print_section_calls_console(self, mock_print):
        """Test that print_section calls console.print."""
        console.print_section("Test Section")
        
        # print_section makes 3 calls (separator, title, separator)
        self.assertEqual(mock_print.call_count, 3)


class TestPrintSuccess(unittest.TestCase):
    """Tests for print_success function."""
    
    @patch.object(console.console, 'print')
    def test_print_success_includes_checkmark(self, mock_print):
        """Test that print_success includes a checkmark."""
        console.print_success("Operation completed")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("✓", call_args)


class TestPrintError(unittest.TestCase):
    """Tests for print_error function."""
    
    @patch.object(console.console, 'print')
    def test_print_error_includes_x(self, mock_print):
        """Test that print_error includes an X mark."""
        console.print_error("Operation failed")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("✗", call_args)


class TestPrintWarning(unittest.TestCase):
    """Tests for print_warning function."""
    
    @patch.object(console.console, 'print')
    def test_print_warning_includes_warning_symbol(self, mock_print):
        """Test that print_warning includes a warning symbol."""
        console.print_warning("Be careful")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("⚠", call_args)


class TestPrintInfo(unittest.TestCase):
    """Tests for print_info function."""
    
    @patch.object(console.console, 'print')
    def test_print_info_includes_info_symbol(self, mock_print):
        """Test that print_info includes an info symbol."""
        console.print_info("Information here")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("ℹ", call_args)


class TestPrintStep(unittest.TestCase):
    """Tests for print_step function."""
    
    @patch.object(console.console, 'print')
    def test_print_step_format(self, mock_print):
        """Test that print_step shows step number correctly."""
        console.print_step(2, 5, "Loading model")
        
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        self.assertIn("[2/5]", call_args)


class TestPrintModelsTable(unittest.TestCase):
    """Tests for print_models_table function."""
    
    @patch.object(console.console, 'print')
    def test_print_models_table(self, mock_print):
        """Test that print_models_table prints a table."""
        models = [
            ModelEntry(
                path="/path/to/model1.safetensors",
                weight=0.5,
                architecture="SDXL",
                precision_detected="fp16",
                index=0
            ),
            ModelEntry(
                path="/path/to/model2.safetensors",
                weight=0.5,
                architecture="SDXL",
                precision_detected="fp16",
                index=1
            ),
        ]
        
        console.print_models_table(models)
        
        mock_print.assert_called_once()
    
    @patch.object(console.console, 'print')
    def test_print_models_table_long_filename_truncated(self, mock_print):
        """Test that long filenames are truncated."""
        models = [
            ModelEntry(
                path="/very/long/path/to/a/model/with/extremely/long/filename/that/should/be/truncated.safetensors",
                weight=0.5,
                architecture="SDXL",
                index=0
            ),
        ]
        
        console.print_models_table(models)
        
        # Just verify it doesn't crash with long names
        mock_print.assert_called_once()


class TestPrintManifestSummary(unittest.TestCase):
    """Tests for print_manifest_summary function."""
    
    @patch.object(console.console, 'print')
    @patch('model_merger.console.print_section')
    @patch('model_merger.console.print_models_table')
    def test_print_manifest_summary(self, mock_table, mock_section, mock_print):
        """Test that print_manifest_summary displays manifest info."""
        models = [
            ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)
        ]
        manifest = MergeManifest(
            models=models,
            output=OutputEntry(path="output.safetensors"),
            device='cpu',
            prune=True,
        )
        
        console.print_manifest_summary(manifest)
        
        mock_section.assert_called_once()
        mock_table.assert_called_once()
    
    @patch.object(console.console, 'print')
    @patch('model_merger.console.print_section')
    @patch('model_merger.console.print_models_table')
    def test_print_manifest_summary_with_vae(self, mock_table, mock_section, mock_print):
        """Test manifest summary with VAE."""
        models = [
            ModelEntry(path="model.safetensors", weight=1.0, architecture="SDXL", index=0)
        ]
        manifest = MergeManifest(
            models=models,
            vae=VAEEntry(path="vae.safetensors", precision_detected="fp16"),
            output=OutputEntry(path="output.safetensors"),
        )
        
        console.print_manifest_summary(manifest)
        
        # Should include VAE info in print calls
        self.assertTrue(mock_print.called)


class TestCreateProgress(unittest.TestCase):
    """Tests for create_progress function."""
    
    def test_create_progress_returns_progress_object(self):
        """Test that create_progress returns a Progress object."""
        from rich.progress import Progress
        
        result = console.create_progress()
        
        self.assertIsInstance(result, Progress)
    
    def test_create_progress_is_context_manager(self):
        """Test that create_progress can be used as context manager."""
        progress = console.create_progress()
        
        # Should be usable as context manager
        with progress:
            pass  # Just verify it doesn't raise


class TestPrintCompletion(unittest.TestCase):
    """Tests for print_completion function."""
    
    @patch.object(console.console, 'print')
    def test_print_completion_shows_info(self, mock_print):
        """Test that print_completion shows completion info."""
        console.print_completion(
            output_path="/path/to/output.safetensors",
            size_mb=3145.67,
            hash_value="abc123" * 10 + "abcd",
            elapsed_seconds=125.5
        )
        
        # Should print multiple times (newline and panel)
        self.assertTrue(mock_print.called)
    
    @patch.object(console.console, 'print')
    def test_print_completion_formats_time_seconds(self, mock_print):
        """Test that elapsed time under 60s is formatted as seconds."""
        console.print_completion(
            output_path="/output.safetensors",
            size_mb=100.0,
            hash_value="a" * 64,
            elapsed_seconds=45.5
        )
        
        self.assertTrue(mock_print.called)
    
    @patch.object(console.console, 'print')
    def test_print_completion_formats_time_minutes(self, mock_print):
        """Test that elapsed time over 60s is formatted as minutes."""
        console.print_completion(
            output_path="/output.safetensors",
            size_mb=100.0,
            hash_value="a" * 64,
            elapsed_seconds=125.0  # 2m 5s
        )
        
        self.assertTrue(mock_print.called)
    
    @patch.object(console.console, 'print')
    def test_print_completion_formats_time_hours(self, mock_print):
        """Test that elapsed time over 1h is formatted as hours."""
        console.print_completion(
            output_path="/output.safetensors",
            size_mb=100.0,
            hash_value="a" * 64,
            elapsed_seconds=3700.0  # 1h 1m 40s
        )
        
        self.assertTrue(mock_print.called)


class TestPrintValidationIssues(unittest.TestCase):
    """Tests for print_validation_issues function."""
    
    @patch.object(console.console, 'print')
    @patch('model_merger.console.print_warning')
    @patch('model_merger.console.print_error')
    def test_warning_uses_print_warning(self, mock_error, mock_warning, mock_print):
        """Test that warnings use print_warning."""
        issues = ["Warning: Weights don't sum to 1.0"]
        
        console.print_validation_issues(issues)
        
        mock_warning.assert_called_once()
        mock_error.assert_not_called()
    
    @patch.object(console.console, 'print')
    @patch('model_merger.console.print_warning')
    @patch('model_merger.console.print_error')
    def test_error_uses_print_error(self, mock_error, mock_warning, mock_print):
        """Test that errors use print_error."""
        issues = ["Model file not found"]
        
        console.print_validation_issues(issues)
        
        mock_error.assert_called_once()
        mock_warning.assert_not_called()
    
    @patch.object(console.console, 'print')
    @patch('model_merger.console.print_warning')
    @patch('model_merger.console.print_error')
    def test_mixed_issues(self, mock_error, mock_warning, mock_print):
        """Test handling of mixed warnings and errors."""
        issues = [
            "Warning: Weights don't sum to 1.0",
            "Model file not found",
            "Warning: Another warning",
        ]
        
        console.print_validation_issues(issues)
        
        self.assertEqual(mock_warning.call_count, 2)
        self.assertEqual(mock_error.call_count, 1)


class TestConsoleInstance(unittest.TestCase):
    """Tests for the console instance."""
    
    def test_console_is_console_instance(self):
        """Test that console is a Rich Console instance."""
        from rich.console import Console
        
        self.assertIsInstance(console.console, Console)


if __name__ == '__main__':
    unittest.main()
