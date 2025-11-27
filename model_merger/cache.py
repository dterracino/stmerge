"""
Model metadata caching for fast lookups without recomputing hashes or API calls.

This module provides a persistent cache for model metadata including:
- File identification (filename, hash, size, modified time)
- Model properties (architecture, precision)
- CivitAI metadata (model ID, version, name, etc.)

The cache is stored as JSON at ~/.model_merger/model_cache.json and uses
SHA-256 hash as the primary key for lookups.

Cache invalidation is based on file size and modification time - if either
changes, the cached entry is considered stale and should be refreshed.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import json
import os


@dataclass
class CivitAIMetadata:
    """CivitAI-specific metadata for a model."""
    model_id: Optional[int] = None
    version_id: Optional[int] = None
    base_model: Optional[str] = None
    model_name: Optional[str] = None
    version_name: Optional[str] = None
    nsfw: bool = False
    trained_words: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CivitAIMetadata':
        """Create from dictionary (JSON deserialization)."""
        return cls(**data)


@dataclass
class CachedModelInfo:
    """Complete cached information for a model file."""
    filename: str
    sha256: str
    file_size: int
    last_modified: str  # ISO 8601 format
    precision: Optional[str] = None  # 'fp16' or 'fp32'
    architecture: Optional[str] = None
    civitai: Optional[CivitAIMetadata] = None
    cached_at: str = ""  # ISO 8601 format
    
    def __post_init__(self):
        """Set cached_at timestamp if not provided."""
        if not self.cached_at:
            self.cached_at = datetime.utcnow().isoformat() + 'Z'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.civitai:
            data['civitai'] = self.civitai.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedModelInfo':
        """Create from dictionary (JSON deserialization)."""
        civitai_data = data.pop('civitai', None)
        civitai = CivitAIMetadata.from_dict(civitai_data) if civitai_data else None
        return cls(**data, civitai=civitai)


class ModelCache:
    """
    Manager for the model metadata cache.
    
    Provides methods to load, save, query, and update cached model information.
    Cache is stored as JSON with schema version for future compatibility.
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, cache_file: Path):
        """
        Initialize cache manager.
        
        Args:
            cache_file: Path to the cache JSON file
        """
        self.cache_file = cache_file
        self.data: Dict[str, CachedModelInfo] = {}
        self.loaded = False
    
    def load(self) -> None:
        """
        Load cache from disk.
        
        Creates empty cache if file doesn't exist. Handles corrupted cache
        gracefully by starting fresh.
        """
        if not self.cache_file.exists():
            self.data = {}
            self.loaded = True
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_json = json.load(f)
            
            # Validate schema version
            version = cache_json.get('version', 0)
            if version != self.SCHEMA_VERSION:
                # Future: handle migration from old versions
                # For now, just start fresh
                self.data = {}
                self.loaded = True
                return
            
            # Load model entries
            models = cache_json.get('models', {})
            self.data = {
                hash_key: CachedModelInfo.from_dict(model_data)
                for hash_key, model_data in models.items()
            }
            self.loaded = True
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Corrupted cache - start fresh
            from .console import print_warning
            print_warning(f"Cache file corrupted, starting fresh: {e}")
            self.data = {}
            self.loaded = True
    
    def save(self) -> None:
        """
        Save cache to disk.
        
        Creates cache directory if it doesn't exist. Writes atomically
        by writing to temp file then renaming.
        """
        # Ensure directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Build cache structure
        cache_json = {
            'version': self.SCHEMA_VERSION,
            'models': {
                hash_key: model_info.to_dict()
                for hash_key, model_info in self.data.items()
            }
        }
        
        # Write atomically (temp file + rename)
        temp_file = self.cache_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_json, f, indent=2, ensure_ascii=False)
            
            # Atomic rename (Windows: need to remove target first)
            if self.cache_file.exists():
                self.cache_file.unlink()
            temp_file.rename(self.cache_file)
            
        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e
    
    def get(self, file_path: Path) -> Optional[CachedModelInfo]:
        """
        Get cached metadata for a file.
        
        Validates cache entry is still valid (file size/mtime unchanged).
        Returns None if cache miss or stale.
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Cached model info if valid, None otherwise
        """
        if not self.loaded:
            self.load()
        
        if not file_path.exists():
            return None
        
        # Get file stats
        stat = file_path.stat()
        file_size = stat.st_size
        last_modified = datetime.utcfromtimestamp(stat.st_mtime).isoformat() + 'Z'
        
        # Try to find in cache by checking all entries for matching filename
        filename = file_path.name
        for cached_info in self.data.values():
            if cached_info.filename == filename:
                # Check if cache is still valid
                if (cached_info.file_size == file_size and 
                    cached_info.last_modified == last_modified):
                    return cached_info
                # Stale entry - file was modified
                return None
        
        # Cache miss
        return None
    
    def get_by_hash(self, file_hash: str) -> Optional[CachedModelInfo]:
        """
        Get cached metadata by SHA-256 hash.
        
        Args:
            file_hash: SHA-256 hash of the model file
            
        Returns:
            Cached model info if found, None otherwise
        """
        if not self.loaded:
            self.load()
        
        return self.data.get(file_hash)
    
    def update(self, file_path: Path, model_info: CachedModelInfo) -> None:
        """
        Update cache with new model information.
        
        Uses SHA-256 hash as the key. Automatically saves after update.
        
        Args:
            file_path: Path to the model file
            model_info: Model metadata to cache
        """
        if not self.loaded:
            self.load()
        
        # Store by hash
        self.data[model_info.sha256] = model_info
        
        # Auto-save after update
        self.save()
    
    def invalidate(self, file_hash: str) -> None:
        """
        Remove an entry from the cache.
        
        Args:
            file_hash: SHA-256 hash of the model to remove
        """
        if not self.loaded:
            self.load()
        
        if file_hash in self.data:
            del self.data[file_hash]
            self.save()
    
    def clear(self) -> None:
        """Clear all cached entries and save."""
        self.data = {}
        self.save()


# Global cache instance (lazy-loaded)
_cache: Optional[ModelCache] = None


def get_cache() -> ModelCache:
    """
    Get the global cache instance.
    
    Initializes cache on first access using the configured cache file path.
    
    Returns:
        The global ModelCache instance
    """
    global _cache
    if _cache is None:
        from .config import CACHE_FILE_PATH
        _cache = ModelCache(CACHE_FILE_PATH)
    return _cache


def get_cached_metadata(file_path: Path) -> Optional[CachedModelInfo]:
    """
    Get cached metadata for a model file.
    
    Convenience function that uses the global cache instance.
    Validates cache is still valid before returning.
    
    Args:
        file_path: Path to the model file
        
    Returns:
        Cached model info if valid, None if cache miss or stale
        
    Example:
        >>> from pathlib import Path
        >>> cached = get_cached_metadata(Path("models/pony_v1.safetensors"))
        >>> if cached:
        ...     print(f"Architecture: {cached.architecture}")
    """
    cache = get_cache()
    return cache.get(file_path)


def update_cache(file_path: Path, model_info: CachedModelInfo) -> None:
    """
    Update cache with model metadata.
    
    Convenience function that uses the global cache instance.
    Automatically saves after update.
    
    Args:
        file_path: Path to the model file
        model_info: Model metadata to cache
        
    Example:
        >>> from pathlib import Path
        >>> info = CachedModelInfo(
        ...     filename="pony_v1.safetensors",
        ...     sha256="abc123...",
        ...     file_size=6535231488,
        ...     last_modified="2025-11-26T10:30:00Z",
        ...     architecture="Pony",
        ...     precision="fp16"
        ... )
        >>> update_cache(Path("models/pony_v1.safetensors"), info)
    """
    cache = get_cache()
    cache.update(file_path, model_info)


def invalidate_cache(file_hash: str) -> None:
    """
    Remove an entry from the cache.
    
    Convenience function that uses the global cache instance.
    
    Args:
        file_hash: SHA-256 hash of the model to remove
        
    Example:
        >>> invalidate_cache("abc123...")
    """
    cache = get_cache()
    cache.invalidate(file_hash)
