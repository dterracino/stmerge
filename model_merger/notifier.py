"""
Windows toast notification support for long-running operations.

Provides desktop notifications when conversions, merges, or batch operations complete.
Gracefully degrades on non-Windows platforms or when notifications are unavailable.

Usage:
    from model_merger.notifier import send_notification, should_notify
    
    if should_notify(args.notify, elapsed_time):
        send_notification("Operation Complete", "Details here...")
"""

import platform
from typing import Optional, Union


# Check if we're on Windows and if notifications are available
_PLATFORM = platform.system()
_NOTIFICATIONS_AVAILABLE = False
_TOAST_NOTIFIER = None

if _PLATFORM == 'Windows':
    try:
        from win10toast import ToastNotifier
        _TOAST_NOTIFIER = ToastNotifier()
        _NOTIFICATIONS_AVAILABLE = True
    except ImportError:
        # Library not installed - that's okay, we'll just skip notifications
        pass
    except Exception:
        # Something else went wrong (rare) - also okay to skip
        pass


def is_available() -> bool:
    """
    Check if notifications are available on this system.
    
    Returns:
        True if notifications can be sent, False otherwise
    """
    return _NOTIFICATIONS_AVAILABLE


def send_notification(
    title: str,
    message: str,
    duration: int = 10,
    icon_path: Optional[str] = None
) -> bool:
    """
    Send a Windows toast notification.
    
    On non-Windows platforms or if notifications aren't available,
    this silently does nothing (graceful degradation).
    
    Args:
        title: Notification title (bold text at top)
        message: Notification body (main message)
        duration: How long to show notification in seconds (default 10)
        icon_path: Optional path to .ico file for custom icon
        
    Returns:
        True if notification was sent successfully, False otherwise
        
    Example:
        send_notification(
            "Model Merger",
            "✓ Conversion complete: model.safetensors\\nSize: 6.2 GB"
        )
    """
    if not _NOTIFICATIONS_AVAILABLE or _TOAST_NOTIFIER is None:
        return False
    
    try:
        # Show toast notification (threaded so it doesn't block)
        _TOAST_NOTIFIER.show_toast(
            title,
            message,
            duration=duration,
            icon_path=icon_path,
            threaded=True,
        )
        return True
        
    except Exception:
        # If notification fails for any reason, silently continue
        # We never want to crash the app because of a notification
        return False


def should_notify(notify_enabled: bool, elapsed_seconds: float, min_duration: float = 30.0) -> bool:
    """
    Determine if a notification should be sent.
    
    Only notifies if:
    1. User enabled notifications (--notify flag)
    2. Operation took longer than minimum duration
    3. Notifications are available on this platform
    
    Args:
        notify_enabled: Whether user passed --notify flag
        elapsed_seconds: How long the operation took
        min_duration: Minimum seconds before notifying (default 30)
        
    Returns:
        True if should send notification, False otherwise
        
    Example:
        start = time.time()
        # ... do work ...
        elapsed = time.time() - start
        
        if should_notify(args.notify, elapsed):
            send_notification("Done!", "Operation complete")
    """
    if not notify_enabled:
        return False
    
    if not _NOTIFICATIONS_AVAILABLE:
        return False
    
    # Only notify for operations that took a while
    # (no spam for 5-second conversions)
    if elapsed_seconds < min_duration:
        return False
    
    return True


def format_time(seconds: float) -> str:
    """
    Format elapsed time in human-readable form.
    
    Args:
        seconds: Elapsed time in seconds
        
    Returns:
        Formatted string like "2m 34s" or "45s"
        
    Example:
        >>> format_time(45)
        '45s'
        >>> format_time(154)
        '2m 34s'
        >>> format_time(3661)
        '1h 1m'
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    
    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds}s"
        return f"{minutes}m"
    
    hours = minutes // 60
    remaining_minutes = minutes % 60
    
    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"


def format_size(size_bytes: Union[int, float]) -> str:
    """
    Format file size in human-readable form.
    
    Args:
        size_bytes: Size in bytes (int or float)
        
    Returns:
        Formatted string like "6.2 GB" or "144 MB"
        
    Example:
        >>> format_size(1024)
        '1.0 KB'
        >>> format_size(1024 * 1024 * 1024 * 6.2)
        '6.2 GB'
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# Notification message templates for consistency

def notify_conversion_success(filename: str, size_bytes: int, elapsed_seconds: float) -> bool:
    """Send notification for successful conversion."""
    return send_notification(
        "Model Merger",
        f"✓ Conversion complete: {filename}\n"
        f"Size: {format_size(size_bytes)} | Time: {format_time(elapsed_seconds)}"
    )


def notify_conversion_failure(filename: str, error_msg: str) -> bool:
    """Send notification for failed conversion."""
    return send_notification(
        "Model Merger",
        f"✗ Conversion failed: {filename}\n"
        f"Error: {error_msg}"
    )


def notify_batch_complete(success_count: int, total_count: int, elapsed_seconds: float) -> bool:
    """Send notification for batch conversion completion."""
    failed = total_count - success_count
    
    if failed == 0:
        message = f"✓ Batch complete: {success_count} files converted\n"
    else:
        message = f"⚠ Batch complete: {success_count}/{total_count} files\n{failed} failed | "
    
    message += f"Time: {format_time(elapsed_seconds)}"
    
    return send_notification("Model Merger", message)


def notify_merge_success(output_filename: str, size_bytes: int, elapsed_seconds: float) -> bool:
    """Send notification for successful merge."""
    return send_notification(
        "Model Merger",
        f"✓ Merge complete: {output_filename}\n"
        f"Size: {format_size(size_bytes)} | Time: {format_time(elapsed_seconds)}"
    )


def notify_merge_failure(error_msg: str) -> bool:
    """Send notification for failed merge."""
    return send_notification(
        "Model Merger",
        f"✗ Merge failed\n"
        f"Error: {error_msg}"
    )