"""
Tests for cache.py module.

Tests model metadata caching functionality including cache loading,
saving, querying, updating, and invalidation.
"""

import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from model_merger.cache import (
    CivitAIMetadata,
    CachedModelInfo,
    ModelCache,
    get_cached_metadata,
    update_cache,
    invalidate_cache,
)


class TestCivitAIMetadata(unittest.TestCase):
    """Tests for CivitAIMetadata dataclass."""
    
    def test_to_dict(self):
        """Test converting CivitAI metadata to dictionary."""
        metadata = CivitAIMetadata(
            model_id=12345,
            version_id=67890,
            base_model="SDXL 1.0",
            model_name="Test Model",
            version_name="v1.0",
            nsfw=False,
            trained_words=["trigger1", "trigger2"]
        )
        
        result = metadata.to_dict()
        
        self.assertEqual(result['model_id'], 12345)
        self.assertEqual(result['version_id'], 67890)
        self.assertEqual(result['base_model'], "SDXL 1.0")
        self.assertFalse(result['nsfw'])
        self.assertEqual(len(result['trained_words']), 2)
    
    def test_from_dict(self):
        """Test creating CivitAI metadata from dictionary."""
        data = {
            'model_id': 12345,
            'version_id': 67890,
            'base_model': "Pony",
            'model_name': "Pony Model",
            'version_name': "v2.0",
            'nsfw': True,
            'trained_words': ["pony", "realistic"]
        }
        
        metadata = CivitAIMetadata.from_dict(data)
        
        self.assertEqual(metadata.model_id, 12345)
        self.assertEqual(metadata.version_id, 67890)
        self.assertTrue(metadata.nsfw)
    
    def test_defaults(self):
        """Test default values for optional fields."""
        metadata = CivitAIMetadata()
        
        self.assertIsNone(metadata.model_id)
        self.assertIsNone(metadata.version_id)
        self.assertFalse(metadata.nsfw)
        self.assertEqual(metadata.trained_words, [])


class TestCachedModelInfo(unittest.TestCase):
    """Tests for CachedModelInfo dataclass."""
    
    def test_to_dict_with_civitai(self):
        """Test converting cached model info to dictionary with CivitAI data."""
        civitai = CivitAIMetadata(
            model_id=12345,
            model_name="Test Model"
        )
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z",
            precision="fp16",
            architecture="SDXL",
            civitai=civitai,
            cached_at="2025-11-26T10:05:00Z"
        )
        
        result = info.to_dict()
        
        self.assertEqual(result['filename'], "test.safetensors")
        self.assertEqual(result['sha256'], "abc123")
        self.assertEqual(result['precision'], "fp16")
        self.assertEqual(result['architecture'], "SDXL")
        self.assertIsNotNone(result['civitai'])
        self.assertEqual(result['civitai']['model_id'], 12345)
    
    def test_from_dict(self):
        """Test creating cached model info from dictionary."""
        data = {
            'filename': "model.safetensors",
            'sha256': "def456",
            'file_size': 2000000,
            'last_modified': "2025-11-26T11:00:00Z",
            'precision': "fp32",
            'architecture': "Pony",
            'civitai': {
                'model_id': 54321,
                'version_id': 98765,
                'model_name': "Pony Model",
                'nsfw': False,
                'trained_words': []
            },
            'cached_at': "2025-11-26T11:05:00Z"
        }
        
        info = CachedModelInfo.from_dict(data)
        
        self.assertEqual(info.filename, "model.safetensors")
        self.assertEqual(info.architecture, "Pony")
        self.assertIsNotNone(info.civitai)
        assert info.civitai is not None  # Type guard
        self.assertEqual(info.civitai.model_id, 54321)
    
    def test_auto_timestamp(self):
        """Test automatic cached_at timestamp generation."""
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z"
        )
        
        # Should have auto-generated timestamp
        self.assertTrue(info.cached_at)
        self.assertIn('T', info.cached_at)
        self.assertTrue(info.cached_at.endswith('Z'))


class TestModelCache(unittest.TestCase):
    """Tests for ModelCache class."""
    
    def setUp(self):
        """Create temporary cache file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = Path(self.temp_dir) / "test_cache.json"
        self.cache = ModelCache(self.cache_file)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_nonexistent_cache(self):
        """Test loading cache when file doesn't exist."""
        self.cache.load()
        
        self.assertTrue(self.cache.loaded)
        self.assertEqual(len(self.cache.data), 0)
    
    def test_save_and_load_cache(self):
        """Test saving and loading cache."""
        # Create test data
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z",
            architecture="SDXL"
        )
        
        self.cache.data["abc123"] = info
        self.cache.save()
        
        # Load in new cache instance
        new_cache = ModelCache(self.cache_file)
        new_cache.load()
        
        self.assertEqual(len(new_cache.data), 1)
        self.assertIn("abc123", new_cache.data)
        loaded_info = new_cache.data["abc123"]
        self.assertEqual(loaded_info.filename, "test.safetensors")
        self.assertEqual(loaded_info.architecture, "SDXL")
    
    def test_save_with_civitai_data(self):
        """Test saving and loading cache with CivitAI metadata."""
        civitai = CivitAIMetadata(
            model_id=12345,
            version_id=67890,
            model_name="Test Model",
            nsfw=False,
            trained_words=["test", "tags"]
        )
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z",
            civitai=civitai
        )
        
        self.cache.data["abc123"] = info
        self.cache.save()
        
        # Verify JSON structure
        with open(self.cache_file, 'r') as f:
            cache_json = json.load(f)
        
        self.assertEqual(cache_json['version'], 1)
        self.assertIn('models', cache_json)
        self.assertIn('abc123', cache_json['models'])
        model_data = cache_json['models']['abc123']
        self.assertIn('civitai', model_data)
        self.assertEqual(model_data['civitai']['model_id'], 12345)
    
    def test_get_by_hash(self):
        """Test retrieving cached data by hash."""
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z"
        )
        
        self.cache.data["abc123"] = info
        self.cache.loaded = True
        
        result = self.cache.get_by_hash("abc123")
        
        self.assertIsNotNone(result)
        assert result is not None  # Type guard
        self.assertEqual(result.filename, "test.safetensors")
    
    def test_get_by_hash_miss(self):
        """Test cache miss when hash not found."""
        self.cache.loaded = True
        
        result = self.cache.get_by_hash("nonexistent")
        
        self.assertIsNone(result)
    
    def test_get_by_path_valid(self):
        """Test retrieving cached data by file path with valid cache."""
        # Create a real temp file for this test
        test_file = Path(self.temp_dir) / "test.safetensors"
        test_file.write_bytes(b"x" * 1000000)  # 1MB file
        
        # Get actual file stats
        stat = test_file.stat()
        test_size = stat.st_size
        test_mtime = datetime.utcfromtimestamp(stat.st_mtime).isoformat() + 'Z'
        
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=test_size,
            last_modified=test_mtime
        )
        # Store by hash, but lookup will search by filename
        self.cache.data["abc123"] = info
        self.cache.loaded = True
        
        result = self.cache.get(test_file)
        
        self.assertIsNotNone(result)
        assert result is not None  # Type guard
        self.assertEqual(result.filename, "test.safetensors")
    
    def test_get_by_path_stale(self):
        """Test cache invalidation when file has changed."""
        # Create a real temp file
        test_file = Path(self.temp_dir) / "test.safetensors"
        test_file.write_bytes(b"x" * 1000000)  # 1MB file
        
        # Cache entry with different size (stale)
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=500000,  # Different from actual file
            last_modified="2025-11-26T09:00:00Z"
        )
        self.cache.data["abc123"] = info
        self.cache.loaded = True
        
        result = self.cache.get(test_file)
        
        # Cache should be invalidated (file changed)
        self.assertIsNone(result)
    
    def test_update_cache(self):
        """Test updating cache with new entry."""
        info = CachedModelInfo(
            filename="new.safetensors",
            sha256="def456",
            file_size=1500000,
            last_modified="2025-11-26T11:00:00Z",
            architecture="Pony"
        )
        
        # Mock save to avoid file I/O
        with patch.object(self.cache, 'save'):
            self.cache.update(Path("models/new.safetensors"), info)
        
        self.assertIn("def456", self.cache.data)
        self.assertEqual(self.cache.data["def456"].architecture, "Pony")
    
    def test_invalidate_cache(self):
        """Test removing entry from cache."""
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z"
        )
        self.cache.data["abc123"] = info
        self.cache.loaded = True
        
        with patch.object(self.cache, 'save'):
            self.cache.invalidate("abc123")
        
        self.assertNotIn("abc123", self.cache.data)
    
    def test_clear_cache(self):
        """Test clearing all cache entries."""
        self.cache.data = {
            "abc123": CachedModelInfo("test1.safetensors", "abc123", 1000, "2025-11-26T10:00:00Z"),
            "def456": CachedModelInfo("test2.safetensors", "def456", 2000, "2025-11-26T11:00:00Z"),
        }
        
        with patch.object(self.cache, 'save'):
            self.cache.clear()
        
        self.assertEqual(len(self.cache.data), 0)
    
    def test_load_corrupted_cache(self):
        """Test handling corrupted cache file."""
        # Write invalid JSON
        with open(self.cache_file, 'w') as f:
            f.write("{ invalid json }")
        
        with patch('model_merger.console.print_warning'):
            self.cache.load()
        
        # Should start fresh
        self.assertTrue(self.cache.loaded)
        self.assertEqual(len(self.cache.data), 0)
    
    def test_load_incompatible_version(self):
        """Test handling cache with incompatible schema version."""
        # Write cache with future version
        cache_data = {
            'version': 999,
            'models': {
                'abc123': {
                    'filename': 'test.safetensors',
                    'sha256': 'abc123',
                    'file_size': 1000,
                    'last_modified': '2025-11-26T10:00:00Z'
                }
            }
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        self.cache.load()
        
        # Should discard incompatible cache and start fresh
        self.assertTrue(self.cache.loaded)
        self.assertEqual(len(self.cache.data), 0)


class TestCacheFunctions(unittest.TestCase):
    """Tests for convenience functions."""
    
    @patch('model_merger.cache.get_cache')
    def test_get_cached_metadata(self, mock_get_cache):
        """Test get_cached_metadata convenience function."""
        mock_cache = MagicMock()
        mock_info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z"
        )
        mock_cache.get.return_value = mock_info
        mock_get_cache.return_value = mock_cache
        
        result = get_cached_metadata(Path("models/test.safetensors"))
        
        self.assertEqual(result, mock_info)
        mock_cache.get.assert_called_once()
    
    @patch('model_merger.cache.get_cache')
    def test_update_cache_function(self, mock_get_cache):
        """Test update_cache convenience function."""
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        
        info = CachedModelInfo(
            filename="test.safetensors",
            sha256="abc123",
            file_size=1000000,
            last_modified="2025-11-26T10:00:00Z"
        )
        
        update_cache(Path("models/test.safetensors"), info)
        
        mock_cache.update.assert_called_once_with(Path("models/test.safetensors"), info)
    
    @patch('model_merger.cache.get_cache')
    def test_invalidate_cache_function(self, mock_get_cache):
        """Test invalidate_cache convenience function."""
        mock_cache = MagicMock()
        mock_get_cache.return_value = mock_cache
        
        invalidate_cache("abc123")
        
        mock_cache.invalidate.assert_called_once_with("abc123")


if __name__ == '__main__':
    unittest.main()
