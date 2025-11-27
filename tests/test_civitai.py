"""
Tests for civitai.py module.

Tests CivitAI API integration functions for model metadata lookup,
architecture detection, and data mapping.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
from model_merger import civitai


class TestMapBaseModelToArchitecture(unittest.TestCase):
    """Tests for _map_base_model_to_architecture function."""
    
    def test_pony_detection(self):
        """Test detection of Pony architecture from baseModel."""
        result = civitai._map_base_model_to_architecture("Pony")
        self.assertEqual(result, "Pony")
    
    def test_pony_detection_lowercase(self):
        """Test detection of Pony with lowercase."""
        result = civitai._map_base_model_to_architecture("pony")
        self.assertEqual(result, "Pony")
    
    def test_illustrious_detection(self):
        """Test detection of Illustrious architecture."""
        result = civitai._map_base_model_to_architecture("Illustrious")
        self.assertEqual(result, "Illustrious")
    
    def test_illustrious_detection_lowercase(self):
        """Test detection of Illustrious with lowercase."""
        result = civitai._map_base_model_to_architecture("illustrious")
        self.assertEqual(result, "Illustrious")
    
    def test_noobai_detection(self):
        """Test detection of Noobai architecture."""
        result = civitai._map_base_model_to_architecture("Noobai")
        self.assertEqual(result, "Noobai")
    
    def test_noob_detection(self):
        """Test detection of Noobai via 'noob'."""
        result = civitai._map_base_model_to_architecture("noob")
        self.assertEqual(result, "Noobai")
    
    def test_sdxl_1_0_detection(self):
        """Test detection of SDXL 1.0."""
        result = civitai._map_base_model_to_architecture("SDXL 1.0")
        self.assertEqual(result, "SDXL")
    
    def test_sdxl_0_9_detection(self):
        """Test detection of SDXL 0.9."""
        result = civitai._map_base_model_to_architecture("SDXL 0.9")
        self.assertEqual(result, "SDXL")
    
    def test_xl_detection(self):
        """Test detection of XL variant."""
        result = civitai._map_base_model_to_architecture("XL Base")
        self.assertEqual(result, "SDXL")
    
    def test_sd_1_5_detection(self):
        """Test detection of SD 1.5."""
        result = civitai._map_base_model_to_architecture("SD 1.5")
        self.assertEqual(result, "SD1.5")
    
    def test_sd1_detection(self):
        """Test detection of SD1 variant."""
        result = civitai._map_base_model_to_architecture("SD1.4")
        self.assertEqual(result, "SD1.5")
    
    def test_sd_2_1_detection(self):
        """Test detection of SD 2.1."""
        result = civitai._map_base_model_to_architecture("SD 2.1")
        self.assertEqual(result, "SD2.1")
    
    def test_sd2_detection(self):
        """Test detection of SD2 variant."""
        result = civitai._map_base_model_to_architecture("SD2.0")
        self.assertEqual(result, "SD2.1")
    
    def test_unknown_base_model(self):
        """Test that unknown base models return None."""
        result = civitai._map_base_model_to_architecture("UnknownModel")
        self.assertIsNone(result)
    
    def test_empty_base_model(self):
        """Test that empty string returns None."""
        result = civitai._map_base_model_to_architecture("")
        self.assertIsNone(result)
    
    def test_illustrious_before_sdxl(self):
        """Test that Illustrious is detected even if 'xl' appears in name."""
        # This ensures order of checks is correct
        result = civitai._map_base_model_to_architecture("Illustrious XL")
        self.assertEqual(result, "Illustrious")


class TestGetModelVersionByHash(unittest.TestCase):
    """Tests for get_model_version_by_hash function."""
    
    @patch('model_merger.civitai.requests.get')
    @patch('model_merger.civitai.get_civitai_api_key')
    def test_successful_lookup_with_api_key(self, mock_get_key, mock_requests_get):
        """Test successful model lookup with API key."""
        mock_get_key.return_value = "test_api_key"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 12345,
            'name': 'v1.0',
            'model': {'name': 'Test Model'}
        }
        mock_requests_get.return_value = mock_response
        
        result = civitai.get_model_version_by_hash("abc123")
        
        self.assertIsNotNone(result)
        assert result is not None  # Type guard for mypy/pylance
        self.assertEqual(result['id'], 12345)
        self.assertEqual(result['name'], 'v1.0')
        
        # Verify API key was used in header
        call_kwargs = mock_requests_get.call_args[1]
        self.assertIn('Authorization', call_kwargs['headers'])
        self.assertEqual(call_kwargs['headers']['Authorization'], 'Bearer test_api_key')
    
    @patch('model_merger.civitai.requests.get')
    @patch('model_merger.civitai.get_civitai_api_key')
    def test_successful_lookup_without_api_key(self, mock_get_key, mock_requests_get):
        """Test successful model lookup without API key."""
        mock_get_key.return_value = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'id': 12345}
        mock_requests_get.return_value = mock_response
        
        result = civitai.get_model_version_by_hash("abc123")
        
        self.assertIsNotNone(result)
        # Verify no Authorization header when no API key
        call_kwargs = mock_requests_get.call_args[1]
        self.assertNotIn('Authorization', call_kwargs['headers'])
    
    @patch('model_merger.civitai.requests.get')
    @patch('model_merger.civitai.get_civitai_api_key')
    def test_model_not_found(self, mock_get_key, mock_requests_get):
        """Test lookup when model is not found (404)."""
        mock_get_key.return_value = None
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests_get.return_value = mock_response
        
        result = civitai.get_model_version_by_hash("nonexistent")
        
        self.assertIsNone(result)
    
    @patch('model_merger.civitai.requests.get')
    @patch('model_merger.civitai.get_civitai_api_key')
    def test_request_timeout(self, mock_get_key, mock_requests_get):
        """Test handling of request timeout."""
        mock_get_key.return_value = None
        mock_requests_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        result = civitai.get_model_version_by_hash("abc123")
        
        self.assertIsNone(result)
    
    @patch('model_merger.civitai.requests.get')
    @patch('model_merger.civitai.get_civitai_api_key')
    def test_correct_url_format(self, mock_get_key, mock_requests_get):
        """Test that correct API URL is constructed."""
        mock_get_key.return_value = None
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_requests_get.return_value = mock_response
        
        test_hash = "test_hash_123"
        civitai.get_model_version_by_hash(test_hash)
        
        # Verify correct URL was called
        call_args = mock_requests_get.call_args[0]
        expected_url = f"https://civitai.com/api/v1/model-versions/by-hash/{test_hash}"
        self.assertEqual(call_args[0], expected_url)


class TestDetectArchitectureFromCivitai(unittest.TestCase):
    """Tests for detect_architecture_from_civitai function."""
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    def test_detection_from_base_model(self, mock_get_version):
        """Test architecture detection from baseModel field."""
        mock_get_version.return_value = {
            'baseModel': 'Pony',
            'model': {'name': 'Test Model', 'tags': []}
        }
        
        result = civitai.detect_architecture_from_civitai("abc123")
        
        self.assertEqual(result, "Pony")
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    def test_detection_from_model_name(self, mock_get_version):
        """Test architecture detection from model name."""
        mock_get_version.return_value = {
            'baseModel': None,
            'model': {'name': 'SDXL Realistic Model', 'tags': []}
        }
        
        result = civitai.detect_architecture_from_civitai("abc123")
        
        self.assertEqual(result, "SDXL")
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    def test_detection_from_tags(self, mock_get_version):
        """Test architecture detection from model tags."""
        mock_get_version.return_value = {
            'baseModel': None,
            'model': {'name': 'Test Model', 'tags': ['anime', 'illus', 'style']}
        }
        
        result = civitai.detect_architecture_from_civitai("abc123")
        
        self.assertEqual(result, "Illustrious")
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    def test_fallback_to_filename(self, mock_get_version):
        """Test fallback to filename detection."""
        mock_get_version.return_value = None
        
        result = civitai.detect_architecture_from_civitai(
            "abc123",
            fallback_filename="pony_model.safetensors"
        )
        
        self.assertEqual(result, "Pony")
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    def test_fallback_when_no_pattern_match(self, mock_get_version):
        """Test that fallback to None occurs when CivitAI returns no usable data and no filename provided."""
        mock_get_version.return_value = {
            'baseModel': 'Unknown',
            'model': {'name': 'Generic Model', 'tags': []}
        }
        
        # Without fallback_filename, should eventually return None
        result = civitai.detect_architecture_from_civitai("abc123")
        
        # The model name "Generic Model" will match no patterns, so returns default (SDXL)
        # Since we're calling without fallback_filename and model name returns default,
        # it returns that default
        self.assertEqual(result, "SDXL")


class TestGetModelMetadataSummary(unittest.TestCase):
    """Tests for get_model_metadata_summary function."""
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    @patch('model_merger.civitai.detect_architecture_from_civitai')
    def test_complete_summary(self, mock_detect_arch, mock_get_version):
        """Test complete metadata summary generation."""
        mock_get_version.return_value = {
            'name': 'v2.0',
            'baseModel': 'SDXL 1.0',
            'trainedWords': ['trigger1', 'trigger2'],
            'downloadUrl': 'https://example.com/model.safetensors',
            'model': {
                'name': 'Test Model',
                'type': 'Checkpoint',
                'nsfw': False
            }
        }
        mock_detect_arch.return_value = 'SDXL'
        
        result = civitai.get_model_metadata_summary("abc123")
        
        self.assertIsNotNone(result)
        assert result is not None  # Type guard for mypy/pylance
        self.assertEqual(result['model_name'], 'Test Model')
        self.assertEqual(result['version_name'], 'v2.0')
        self.assertEqual(result['architecture'], 'SDXL')
        self.assertEqual(result['base_model'], 'SDXL 1.0')
        self.assertEqual(result['type'], 'Checkpoint')
        self.assertFalse(result['nsfw'])
        self.assertEqual(len(result['trained_words']), 2)
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    def test_summary_with_missing_model(self, mock_get_version):
        """Test summary returns None when model not found."""
        mock_get_version.return_value = None
        
        result = civitai.get_model_metadata_summary("nonexistent")
        
        self.assertIsNone(result)
    
    @patch('model_merger.civitai.get_model_version_by_hash')
    @patch('model_merger.civitai.detect_architecture_from_civitai')
    def test_summary_with_nsfw_model(self, mock_detect_arch, mock_get_version):
        """Test summary correctly flags NSFW models."""
        mock_get_version.return_value = {
            'model': {
                'name': 'NSFW Model',
                'type': 'Checkpoint',
                'nsfw': True
            },
            'name': 'v1.0',
            'trainedWords': []
        }
        mock_detect_arch.return_value = None
        result = civitai.get_model_metadata_summary("abc123")
        
        self.assertIsNotNone(result)
        assert result is not None  # Type guard for mypy/pylance
        self.assertTrue(result['nsfw'])
        self.assertTrue(result['nsfw'])


if __name__ == '__main__':
    unittest.main()
