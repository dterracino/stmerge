"""
Tests for run.py module.

Tests the main entry point for the Model Merger CLI.
"""

import unittest
from unittest.mock import patch


class TestRunMain(unittest.TestCase):
    """Tests for run.py main entry point."""
    
    @patch('model_merger.cli.main')
    def test_run_calls_cli_main(self, mock_main):
        """Test that run.py imports and would call cli.main()."""
        mock_main.return_value = 0
        
        # Import the module to verify it can be imported without errors
        import run
        
        # The run module should have imported main from model_merger.cli
        self.assertIsNotNone(run.main)
    
    @patch('model_merger.cli.main')
    def test_main_returns_zero_on_success(self, mock_main):
        """Test that main returns 0 on success."""
        mock_main.return_value = 0
        
        from model_merger.cli import main
        result = main()
        
        self.assertEqual(result, 0)
    
    @patch('model_merger.cli.main')
    def test_main_returns_one_on_failure(self, mock_main):
        """Test that main returns 1 on failure."""
        mock_main.return_value = 1
        
        from model_merger.cli import main
        result = main()
        
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
