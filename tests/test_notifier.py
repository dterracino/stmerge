"""
Tests for notifier.py module.

Tests Windows toast notification support for long-running operations.
"""

import unittest
from unittest.mock import patch, MagicMock

from model_merger import notifier


class TestIsAvailable(unittest.TestCase):
    """Tests for is_available function."""
    
    def test_is_available_returns_boolean(self):
        """Test that is_available returns a boolean."""
        result = notifier.is_available()
        self.assertIsInstance(result, bool)


class TestSendNotification(unittest.TestCase):
    """Tests for send_notification function."""
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', False)
    def test_send_notification_returns_false_when_unavailable(self):
        """Test that send_notification returns False when notifications unavailable."""
        result = notifier.send_notification("Title", "Message")
        self.assertFalse(result)
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', True)
    @patch.object(notifier, '_TOAST_NOTIFIER', None)
    def test_send_notification_returns_false_when_notifier_is_none(self):
        """Test that send_notification returns False when notifier is None."""
        result = notifier.send_notification("Title", "Message")
        self.assertFalse(result)
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', True)
    def test_send_notification_calls_show_toast(self):
        """Test that send_notification calls show_toast on the notifier."""
        mock_notifier = MagicMock()
        
        with patch.object(notifier, '_TOAST_NOTIFIER', mock_notifier):
            result = notifier.send_notification("Test Title", "Test Message", duration=5)
            
            self.assertTrue(result)
            mock_notifier.show_toast.assert_called_once_with(
                "Test Title",
                "Test Message",
                duration=5,
                icon_path=None,
                threaded=True,
            )
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', True)
    def test_send_notification_handles_exception(self):
        """Test that send_notification returns False on exception."""
        mock_notifier = MagicMock()
        mock_notifier.show_toast.side_effect = Exception("Notification error")
        
        with patch.object(notifier, '_TOAST_NOTIFIER', mock_notifier):
            result = notifier.send_notification("Title", "Message")
            
            self.assertFalse(result)


class TestShouldNotify(unittest.TestCase):
    """Tests for should_notify function."""
    
    def test_should_notify_false_when_disabled(self):
        """Test that should_notify returns False when notify_enabled is False."""
        result = notifier.should_notify(False, 100.0)
        self.assertFalse(result)
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', False)
    def test_should_notify_false_when_unavailable(self):
        """Test that should_notify returns False when notifications unavailable."""
        result = notifier.should_notify(True, 100.0)
        self.assertFalse(result)
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', True)
    def test_should_notify_false_when_duration_below_minimum(self):
        """Test that should_notify returns False when duration is below minimum."""
        result = notifier.should_notify(True, 10.0, min_duration=30.0)
        self.assertFalse(result)
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', True)
    def test_should_notify_true_when_all_conditions_met(self):
        """Test that should_notify returns True when all conditions are met."""
        result = notifier.should_notify(True, 60.0, min_duration=30.0)
        self.assertTrue(result)
    
    @patch.object(notifier, '_NOTIFICATIONS_AVAILABLE', True)
    def test_should_notify_uses_default_min_duration(self):
        """Test that should_notify uses default min_duration of 30 seconds."""
        # Just below default threshold
        result_below = notifier.should_notify(True, 29.0)
        self.assertFalse(result_below)
        
        # At default threshold
        result_at = notifier.should_notify(True, 30.0)
        self.assertTrue(result_at)


class TestFormatTime(unittest.TestCase):
    """Tests for format_time function."""
    
    def test_format_time_seconds_only(self):
        """Test formatting time less than 60 seconds."""
        self.assertEqual(notifier.format_time(45), "45s")
        self.assertEqual(notifier.format_time(5), "5s")
        self.assertEqual(notifier.format_time(59), "59s")
    
    def test_format_time_minutes_and_seconds(self):
        """Test formatting time in minutes and seconds."""
        self.assertEqual(notifier.format_time(60), "1m")
        self.assertEqual(notifier.format_time(90), "1m 30s")
        self.assertEqual(notifier.format_time(154), "2m 34s")
    
    def test_format_time_hours_and_minutes(self):
        """Test formatting time in hours and minutes."""
        self.assertEqual(notifier.format_time(3600), "1h")
        self.assertEqual(notifier.format_time(3661), "1h 1m")
        self.assertEqual(notifier.format_time(7200), "2h")
        self.assertEqual(notifier.format_time(7260), "2h 1m")
    
    def test_format_time_edge_cases(self):
        """Test edge cases for time formatting."""
        self.assertEqual(notifier.format_time(0), "0s")
        self.assertEqual(notifier.format_time(3599), "59m 59s")


class TestFormatSize(unittest.TestCase):
    """Tests for format_size function."""
    
    def test_format_size_bytes(self):
        """Test formatting sizes in bytes."""
        self.assertEqual(notifier.format_size(0), "0.0 B")
        self.assertEqual(notifier.format_size(512), "512.0 B")
        self.assertEqual(notifier.format_size(1023), "1023.0 B")
    
    def test_format_size_kilobytes(self):
        """Test formatting sizes in kilobytes."""
        self.assertEqual(notifier.format_size(1024), "1.0 KB")
        self.assertEqual(notifier.format_size(1536), "1.5 KB")
        self.assertEqual(notifier.format_size(10240), "10.0 KB")
    
    def test_format_size_megabytes(self):
        """Test formatting sizes in megabytes."""
        self.assertEqual(notifier.format_size(1024 * 1024), "1.0 MB")
        self.assertEqual(notifier.format_size(1024 * 1024 * 10), "10.0 MB")
        self.assertEqual(notifier.format_size(1024 * 1024 * 144), "144.0 MB")
    
    def test_format_size_gigabytes(self):
        """Test formatting sizes in gigabytes."""
        self.assertEqual(notifier.format_size(1024 * 1024 * 1024), "1.0 GB")
        self.assertEqual(notifier.format_size(1024 * 1024 * 1024 * 6.2), "6.2 GB")
    
    def test_format_size_terabytes(self):
        """Test formatting sizes in terabytes."""
        self.assertEqual(notifier.format_size(1024 * 1024 * 1024 * 1024), "1.0 TB")
        self.assertEqual(notifier.format_size(1024 * 1024 * 1024 * 1024 * 2), "2.0 TB")
    
    def test_format_size_petabytes(self):
        """Test formatting sizes in petabytes."""
        size_pb = 1024 * 1024 * 1024 * 1024 * 1024
        self.assertEqual(notifier.format_size(size_pb), "1.0 PB")


class TestNotifyConversionSuccess(unittest.TestCase):
    """Tests for notify_conversion_success function."""
    
    @patch.object(notifier, 'send_notification')
    def test_notify_conversion_success_calls_send_notification(self, mock_send):
        """Test that notify_conversion_success calls send_notification."""
        mock_send.return_value = True
        
        result = notifier.notify_conversion_success("model.safetensors", 1024 * 1024 * 100, 45.0)
        
        self.assertTrue(result)
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0]
        self.assertEqual(call_args[0], "Model Merger")
        self.assertIn("Conversion complete", call_args[1])
        self.assertIn("model.safetensors", call_args[1])


class TestNotifyConversionFailure(unittest.TestCase):
    """Tests for notify_conversion_failure function."""
    
    @patch.object(notifier, 'send_notification')
    def test_notify_conversion_failure_calls_send_notification(self, mock_send):
        """Test that notify_conversion_failure calls send_notification."""
        mock_send.return_value = True
        
        result = notifier.notify_conversion_failure("model.ckpt", "File corrupted")
        
        self.assertTrue(result)
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0]
        self.assertEqual(call_args[0], "Model Merger")
        self.assertIn("Conversion failed", call_args[1])
        self.assertIn("model.ckpt", call_args[1])
        self.assertIn("File corrupted", call_args[1])


class TestNotifyBatchComplete(unittest.TestCase):
    """Tests for notify_batch_complete function."""
    
    @patch.object(notifier, 'send_notification')
    def test_notify_batch_complete_success_message(self, mock_send):
        """Test that batch complete with all success shows appropriate message."""
        mock_send.return_value = True
        
        result = notifier.notify_batch_complete(5, 5, 120.0)
        
        self.assertTrue(result)
        call_args = mock_send.call_args[0]
        self.assertIn("5 files converted", call_args[1])
        self.assertNotIn("failed", call_args[1].lower())
    
    @patch.object(notifier, 'send_notification')
    def test_notify_batch_complete_with_failures(self, mock_send):
        """Test that batch complete with failures shows failure count."""
        mock_send.return_value = True
        
        result = notifier.notify_batch_complete(3, 5, 120.0)
        
        self.assertTrue(result)
        call_args = mock_send.call_args[0]
        self.assertIn("3/5 files", call_args[1])
        self.assertIn("2 failed", call_args[1])


class TestNotifyMergeSuccess(unittest.TestCase):
    """Tests for notify_merge_success function."""
    
    @patch.object(notifier, 'send_notification')
    def test_notify_merge_success_calls_send_notification(self, mock_send):
        """Test that notify_merge_success calls send_notification."""
        mock_send.return_value = True
        
        result = notifier.notify_merge_success("merged_model.safetensors", 1024 * 1024 * 1024 * 6, 300.0)
        
        self.assertTrue(result)
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0]
        self.assertEqual(call_args[0], "Model Merger")
        self.assertIn("Merge complete", call_args[1])
        self.assertIn("merged_model.safetensors", call_args[1])


class TestNotifyMergeFailure(unittest.TestCase):
    """Tests for notify_merge_failure function."""
    
    @patch.object(notifier, 'send_notification')
    def test_notify_merge_failure_calls_send_notification(self, mock_send):
        """Test that notify_merge_failure calls send_notification."""
        mock_send.return_value = True
        
        result = notifier.notify_merge_failure("Incompatible tensor shapes")
        
        self.assertTrue(result)
        mock_send.assert_called_once()
        call_args = mock_send.call_args[0]
        self.assertEqual(call_args[0], "Model Merger")
        self.assertIn("Merge failed", call_args[1])
        self.assertIn("Incompatible tensor shapes", call_args[1])


if __name__ == '__main__':
    unittest.main()
