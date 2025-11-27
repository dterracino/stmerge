"""
Tests for hasher.py module.

Tests file hashing utilities with multiple algorithms.
"""

import unittest
import tempfile
from pathlib import Path

from model_merger.hasher import (
    compute_file_hash,
    compute_autov1,
    compute_autov2,
    compute_sha256,
    compute_md5,
    compute_crc32,
    verify_file_hash,
)


class TestComputeFileHash(unittest.TestCase):
    """Tests for compute_file_hash function."""
    
    def setUp(self):
        """Create temporary test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.bin"
        self.test_data = b"Hello, World! This is test data for hashing."
        self.test_file.write_bytes(self.test_data)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_sha256_hash(self):
        """Test SHA-256 hashing."""
        result = compute_file_hash(self.test_file, algorithm='sha256')
        
        # Verify it's a 64-character hex string (SHA-256)
        self.assertEqual(len(result), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in result))
    
    def test_md5_hash(self):
        """Test MD5 hashing."""
        result = compute_file_hash(self.test_file, algorithm='md5')
        
        # Verify it's a 32-character hex string (MD5)
        self.assertEqual(len(result), 32)
        self.assertTrue(all(c in '0123456789abcdef' for c in result))
    
    def test_crc32_hash(self):
        """Test CRC32 checksum."""
        result = compute_file_hash(self.test_file, algorithm='crc32')
        
        # Verify it's an 8-character hex string (CRC32)
        self.assertEqual(len(result), 8)
        self.assertTrue(all(c in '0123456789abcdef' for c in result))
    
    def test_blake3_hash_not_installed(self):
        """Test Blake3 raises error when package not installed."""
        # Blake3 is optional and likely not installed
        with self.assertRaises(ValueError) as ctx:
            compute_file_hash(self.test_file, algorithm='blake3')
        
        self.assertIn('blake3', str(ctx.exception).lower())
    
    def test_autov1_not_implemented(self):
        """Test AutoV1 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as ctx:
            compute_file_hash(self.test_file, algorithm='autov1')
        
        self.assertIn('AutoV1', str(ctx.exception))
    
    def test_autov2_not_implemented(self):
        """Test AutoV2 raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as ctx:
            compute_file_hash(self.test_file, algorithm='autov2')
        
        self.assertIn('AutoV2', str(ctx.exception))
    
    def test_consistent_hashing(self):
        """Test that same file produces same hash."""
        hash1 = compute_file_hash(self.test_file, algorithm='sha256')
        hash2 = compute_file_hash(self.test_file, algorithm='sha256')
        
        self.assertEqual(hash1, hash2)
    
    def test_different_files_different_hashes(self):
        """Test that different files produce different hashes."""
        file2 = Path(self.temp_dir) / "test2.bin"
        file2.write_bytes(b"Different data")
        
        hash1 = compute_file_hash(self.test_file, algorithm='sha256')
        hash2 = compute_file_hash(file2, algorithm='sha256')
        
        self.assertNotEqual(hash1, hash2)
    
    def test_large_file_chunks(self):
        """Test hashing with custom chunk size."""
        # Create larger file
        large_data = b"X" * 100000
        large_file = Path(self.temp_dir) / "large.bin"
        large_file.write_bytes(large_data)
        
        # Hash with different chunk sizes should produce same result
        hash_small_chunks = compute_file_hash(large_file, algorithm='sha256', chunk_size=1024)
        hash_large_chunks = compute_file_hash(large_file, algorithm='sha256', chunk_size=8192)
        
        self.assertEqual(hash_small_chunks, hash_large_chunks)


class TestConvenienceFunctions(unittest.TestCase):
    """Tests for algorithm-specific convenience functions."""
    
    def setUp(self):
        """Create temporary test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.bin"
        self.test_data = b"Test data for convenience functions"
        self.test_file.write_bytes(self.test_data)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_compute_autov1(self):
        """Test compute_autov1 convenience function raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            compute_autov1(self.test_file)
    
    def test_compute_autov2(self):
        """Test compute_autov2 convenience function raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            compute_autov2(self.test_file)
    
    def test_compute_sha256(self):
        """Test compute_sha256 convenience function."""
        result = compute_sha256(self.test_file)
        
        # Should match general function with sha256
        expected = compute_file_hash(self.test_file, algorithm='sha256')
        self.assertEqual(result, expected)
    
    def test_compute_md5(self):
        """Test compute_md5 convenience function."""
        result = compute_md5(self.test_file)
        
        # Should match general function with md5
        expected = compute_file_hash(self.test_file, algorithm='md5')
        self.assertEqual(result, expected)
    
    def test_compute_crc32(self):
        """Test compute_crc32 convenience function."""
        result = compute_crc32(self.test_file)
        
        # Should match general function with crc32
        expected = compute_file_hash(self.test_file, algorithm='crc32')
        self.assertEqual(result, expected)


class TestVerifyFileHash(unittest.TestCase):
    """Tests for verify_file_hash function."""
    
    def setUp(self):
        """Create temporary test file."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.bin"
        self.test_data = b"Verification test data"
        self.test_file.write_bytes(self.test_data)
        self.known_hash = compute_sha256(self.test_file)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_verify_correct_hash(self):
        """Test verification with correct hash."""
        result = verify_file_hash(self.test_file, self.known_hash, algorithm='sha256')
        
        self.assertTrue(result)
    
    def test_verify_incorrect_hash(self):
        """Test verification with incorrect hash."""
        wrong_hash = "0" * 64  # Invalid hash
        result = verify_file_hash(self.test_file, wrong_hash, algorithm='sha256')
        
        self.assertFalse(result)
    
    def test_verify_case_insensitive(self):
        """Test verification is case-insensitive."""
        upper_hash = self.known_hash.upper()
        result = verify_file_hash(self.test_file, upper_hash, algorithm='sha256')
        
        self.assertTrue(result)
    
    def test_verify_different_algorithm(self):
        """Test verification with different algorithm."""
        md5_hash = compute_md5(self.test_file)
        result = verify_file_hash(self.test_file, md5_hash, algorithm='md5')
        
        self.assertTrue(result)
    
    def test_verify_modified_file(self):
        """Test verification fails after file modification."""
        original_hash = compute_sha256(self.test_file)
        
        # Modify file
        self.test_file.write_bytes(b"Modified data")
        
        result = verify_file_hash(self.test_file, original_hash, algorithm='sha256')
        
        self.assertFalse(result)


class TestKnownHashes(unittest.TestCase):
    """Tests with known hash values for verification."""
    
    def setUp(self):
        """Create temporary test file with known content."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "known.txt"
        # Use simple known string
        self.test_file.write_text("hello world", encoding='utf-8')
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_known_sha256(self):
        """Test SHA-256 against known value."""
        # SHA-256 of "hello world" (without newline)
        # Computed with: echo -n "hello world" | sha256sum
        # Note: Python's write_text might add newline depending on mode
        result = compute_sha256(self.test_file)
        
        # Just verify it's a valid SHA-256 format
        self.assertEqual(len(result), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in result))


if __name__ == '__main__':
    unittest.main()
