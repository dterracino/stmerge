"""
File hashing utilities for model verification and lookup.

Provides multiple hash algorithms for different use cases:
- AutoV1: CivitAI's AutoV1 hash algorithm
- AutoV2: CivitAI's AutoV2 hash algorithm (default for new models)
- SHA-256: CivitAI lookups, general verification
- CRC32: Quick integrity checks
- Blake3: Fast cryptographic hash
- MD5: Fast checksums (legacy)
"""

import hashlib
from pathlib import Path
from typing import Literal

HashAlgorithm = Literal['autov1', 'autov2', 'sha256', 'crc32', 'blake3', 'md5']


def compute_file_hash(
    filepath: Path, 
    algorithm: HashAlgorithm = 'sha256',
    chunk_size: int = 8192
) -> str:
    """
    Compute hash of a file using specified algorithm.
    
    Reads the file in chunks to avoid loading huge models into memory.
    This is useful for verification and for looking up models in databases
    like CivitAI.
    
    Args:
        filepath: Path to the file to hash
        algorithm: Hash algorithm to use ('autov1', 'autov2', 'sha256', 'crc32', 'blake3', 'md5')
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of the computed hash
        
    Raises:
        ValueError: If algorithm is not supported
        NotImplementedError: If algorithm is not yet implemented
        
    Example:
        >>> from pathlib import Path
        >>> hash_sha256 = compute_file_hash(Path("model.safetensors"))
        >>> hash_autov2 = compute_file_hash(Path("model.safetensors"), algorithm='autov2')
    """
    from .console import create_progress
    
    # Initialize hasher based on algorithm
    if algorithm == 'autov1':
        # TODO: Implement CivitAI AutoV1 hash algorithm
        # See: https://github.com/civitai/civitai/wiki/How-to-use-models#model-hash
        raise NotImplementedError("AutoV1 hash algorithm not yet implemented")
    elif algorithm == 'autov2':
        # TODO: Implement CivitAI AutoV2 hash algorithm (default for newer models)
        # See: https://github.com/civitai/civitai/wiki/How-to-use-models#model-hash
        raise NotImplementedError("AutoV2 hash algorithm not yet implemented")
    elif algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'md5':
        hasher = hashlib.md5()
    elif algorithm == 'crc32':
        # CRC32 uses zlib, not hashlib
        import zlib
        crc_value = 0
    elif algorithm == 'blake3':
        # Blake3 requires external package
        try:
            import blake3 # type: ignore
            hasher = blake3.blake3()
        except ImportError:
            raise ValueError("blake3 algorithm requires 'blake3' package: pip install blake3")
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    file_size = filepath.stat().st_size
    
    with create_progress() as progress:
        task = progress.add_task(
            f"[cyan]Hashing {filepath.name} ({algorithm.upper()})...",
            total=file_size
        )
        
        with open(filepath, 'rb') as f:
            if algorithm == 'crc32':
                # CRC32 special handling
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    crc_value = zlib.crc32(chunk, crc_value)
                    progress.advance(task, len(chunk))
                # Return CRC32 as 8-character hex string
                return f"{crc_value & 0xffffffff:08x}"
            else:
                # Standard hashlib interface
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    progress.advance(task, len(chunk))
                return hasher.hexdigest()


def compute_autov1(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute CivitAI AutoV1 hash of a file.
    
    AutoV1 is CivitAI's legacy hash algorithm used for older models.
    
    Args:
        filepath: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of the AutoV1 hash
        
    Raises:
        NotImplementedError: AutoV1 algorithm not yet implemented
    """
    return compute_file_hash(filepath, algorithm='autov1', chunk_size=chunk_size)


def compute_autov2(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute CivitAI AutoV2 hash of a file.
    
    AutoV2 is CivitAI's current default hash algorithm for new models.
    This is the recommended algorithm for looking up models on CivitAI.
    
    Args:
        filepath: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of the AutoV2 hash
        
    Raises:
        NotImplementedError: AutoV2 algorithm not yet implemented
    """
    return compute_file_hash(filepath, algorithm='autov2', chunk_size=chunk_size)


def compute_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of a file.
    
    Convenience function for the most common hash algorithm.
    
    Args:
        filepath: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of the SHA-256 hash
    """
    return compute_file_hash(filepath, algorithm='sha256', chunk_size=chunk_size)


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute MD5 hash of a file.
    
    MD5 is faster than SHA-256 but less secure. Suitable for quick
    integrity checks when cryptographic security is not required.
    
    Args:
        filepath: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of the MD5 hash
    """
    return compute_file_hash(filepath, algorithm='md5', chunk_size=chunk_size)


def compute_crc32(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute CRC32 checksum of a file.
    
    CRC32 is very fast but not cryptographically secure. Useful for
    quick integrity verification.
    
    Args:
        filepath: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        8-character hex string of the CRC32 checksum
    """
    return compute_file_hash(filepath, algorithm='crc32', chunk_size=chunk_size)


def compute_blake3(filepath: Path, chunk_size: int = 8192) -> str:
    """
    Compute Blake3 hash of a file.
    
    Blake3 is much faster than SHA-256 while being cryptographically secure.
    Requires the 'blake3' package to be installed.
    
    Args:
        filepath: Path to the file to hash
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hex string of the Blake3 hash
        
    Raises:
        ValueError: If blake3 package is not installed
    """
    return compute_file_hash(filepath, algorithm='blake3', chunk_size=chunk_size)


def verify_file_hash(
    filepath: Path,
    expected_hash: str,
    algorithm: HashAlgorithm = 'sha256'
) -> bool:
    """
    Verify that a file matches an expected hash.
    
    Args:
        filepath: Path to the file to verify
        expected_hash: Expected hash value (hex string)
        algorithm: Hash algorithm to use
        
    Returns:
        True if hash matches, False otherwise
        
    Example:
        >>> if verify_file_hash(Path("model.safetensors"), "abc123...", "sha256"):
        ...     print("Hash verified!")
    """
    actual_hash = compute_file_hash(filepath, algorithm=algorithm)
    return actual_hash.lower() == expected_hash.lower()
