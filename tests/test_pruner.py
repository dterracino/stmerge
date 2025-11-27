"""
Tests for pruner.py module.

Tests pruning logic for model checkpoints and format detection.
"""

import unittest
import torch

from model_merger import pruner


class TestDetectFormat(unittest.TestCase):
    """Tests for detect_format function."""
    
    def test_empty_state_dict_returns_unknown(self):
        """Test that empty state dict returns unknown format."""
        result = pruner.detect_format({})
        
        self.assertEqual(result, "unknown")
    
    def test_sd_checkpoint_detection(self):
        """Test detection of full SD checkpoint format."""
        state_dict = {
            "model.diffusion_model.layer1.weight": torch.randn(10, 10),
            "model.diffusion_model.layer2.weight": torch.randn(10, 10),
            "first_stage_model.encoder.weight": torch.randn(10, 10),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "sd_checkpoint")
    
    def test_lora_detection(self):
        """Test detection of LoRA format."""
        state_dict = {
            "lora_unet_down_blocks_0_attentions_0": torch.randn(10, 10),
            "lora_te_text_model_encoder_layers_0": torch.randn(10, 10),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "lora")
    
    def test_lora_detection_case_insensitive(self):
        """Test that LoRA detection is case insensitive."""
        state_dict = {
            "LORA_UNET_something": torch.randn(10, 10),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "lora")
    
    def test_embedding_detection(self):
        """Test detection of textual inversion embedding format."""
        # Embeddings have very few keys with short names
        state_dict = {
            "*": torch.randn(768),
            "string_to_param": torch.randn(768),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "embedding")
    
    def test_single_key_embedding_detection(self):
        """Test detection of single-key embedding."""
        state_dict = {
            "my_embedding": torch.randn(768),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "embedding")
    
    def test_upscaler_detection(self):
        """Test detection of upscaler format (ESRGAN/Real-ESRGAN)."""
        # Need more than 5 keys to avoid embedding detection
        state_dict = {
            "body.0.body.0.weight": torch.randn(64, 64, 3, 3),
            "body.0.body.1.weight": torch.randn(64, 64, 3, 3),
            "body.1.body.0.weight": torch.randn(64, 64, 3, 3),
            "body.1.body.1.weight": torch.randn(64, 64, 3, 3),
            "body.2.body.0.weight": torch.randn(64, 64, 3, 3),
            "body.2.body.1.weight": torch.randn(64, 64, 3, 3),
            "conv_first.weight": torch.randn(64, 3, 3, 3),
            "conv_body.weight": torch.randn(64, 64, 3, 3),
            "conv_up1.weight": torch.randn(64, 64, 3, 3),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "upscaler")
    
    def test_standalone_vae_detection(self):
        """Test detection of standalone VAE format."""
        # Need more than 5 keys to avoid embedding detection
        state_dict = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
            "encoder.conv_out.weight": torch.randn(8, 512, 3, 3),
            "encoder.mid.block.weight": torch.randn(512, 512, 3, 3),
            "decoder.conv_in.weight": torch.randn(512, 4, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3),
            "decoder.mid.block.weight": torch.randn(512, 512, 3, 3),
            "quant_conv.weight": torch.randn(8, 8, 1, 1),
            "post_quant_conv.weight": torch.randn(4, 4, 1, 1),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "vae")
    
    def test_baked_vae_detected_as_sd_checkpoint(self):
        """Test that VAE with first_stage_model prefix is detected as SD checkpoint."""
        state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "first_stage_model.encoder.conv_in.weight": torch.randn(128, 3, 3, 3),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "sd_checkpoint")
    
    def test_unknown_format_detection(self):
        """Test detection of unknown format."""
        state_dict = {
            "some_random_key": torch.randn(10, 10),
            "another_random_key": torch.randn(10, 10),
            "yet_another_key_that_is_longer_than_50_chars_to_avoid_embedding_detection_and_more": torch.randn(10, 10),
        }
        
        result = pruner.detect_format(state_dict)
        
        self.assertEqual(result, "unknown")


class TestShouldPruneKey(unittest.TestCase):
    """Tests for should_prune_key function."""
    
    def test_sd_checkpoint_keeps_diffusion_model(self):
        """Test that SD checkpoint keeps diffusion model keys."""
        key = "model.diffusion_model.layer1.weight"
        
        result = pruner.should_prune_key(key, "sd_checkpoint")
        
        self.assertFalse(result)  # Should NOT be pruned
    
    def test_sd_checkpoint_keeps_vae(self):
        """Test that SD checkpoint keeps VAE keys."""
        key = "first_stage_model.encoder.weight"
        
        result = pruner.should_prune_key(key, "sd_checkpoint")
        
        self.assertFalse(result)
    
    def test_sd_checkpoint_keeps_text_encoder_sd1(self):
        """Test that SD checkpoint keeps text encoder keys (SD 1.x/2.x)."""
        key = "cond_stage_model.transformer.text_model.weight"
        
        result = pruner.should_prune_key(key, "sd_checkpoint")
        
        self.assertFalse(result)
    
    def test_sd_checkpoint_keeps_conditioner_sdxl(self):
        """Test that SD checkpoint keeps conditioner keys (SDXL)."""
        key = "conditioner.embedders.0.weight"
        
        result = pruner.should_prune_key(key, "sd_checkpoint")
        
        self.assertFalse(result)
    
    def test_sd_checkpoint_prunes_optimizer(self):
        """Test that SD checkpoint prunes optimizer keys."""
        key = "optimizer_state.layer.weight"
        
        result = pruner.should_prune_key(key, "sd_checkpoint")
        
        self.assertTrue(result)  # Should be pruned
    
    def test_sd_checkpoint_prunes_epoch(self):
        """Test that SD checkpoint prunes epoch keys."""
        key = "epoch"
        
        result = pruner.should_prune_key(key, "sd_checkpoint")
        
        self.assertTrue(result)
    
    def test_sd_checkpoint_prunes_training_state(self):
        """Test that SD checkpoint prunes training state keys."""
        key = "training_state.step"
        
        result = pruner.should_prune_key(key, "sd_checkpoint")
        
        self.assertTrue(result)
    
    def test_vae_format_keeps_most_keys(self):
        """Test that VAE format keeps most keys (conservative)."""
        key = "encoder.conv_in.weight"
        
        result = pruner.should_prune_key(key, "vae")
        
        self.assertFalse(result)  # Should NOT be pruned
    
    def test_vae_format_removes_optimizer(self):
        """Test that VAE format still removes obvious training artifacts."""
        key = "optimizer.param_groups"
        
        result = pruner.should_prune_key(key, "vae")
        
        self.assertTrue(result)
    
    def test_lora_format_keeps_lora_keys(self):
        """Test that LoRA format keeps LoRA keys."""
        key = "lora_unet_down_blocks_0_attentions_0"
        
        result = pruner.should_prune_key(key, "lora")
        
        self.assertFalse(result)
    
    def test_lora_format_removes_optimizer(self):
        """Test that LoRA format removes optimizer keys."""
        key = "optimizer_state.weight"
        
        result = pruner.should_prune_key(key, "lora")
        
        self.assertTrue(result)
    
    def test_embedding_format_keeps_embedding_keys(self):
        """Test that embedding format keeps embedding keys."""
        key = "my_embedding"
        
        result = pruner.should_prune_key(key, "embedding")
        
        self.assertFalse(result)
    
    def test_unknown_format_is_conservative(self):
        """Test that unknown format is conservative (keeps most keys)."""
        key = "some_random_model_weight"
        
        result = pruner.should_prune_key(key, "unknown")
        
        self.assertFalse(result)  # Should NOT be pruned (conservative)
    
    def test_unknown_format_removes_training_artifacts(self):
        """Test that unknown format still removes obvious training artifacts."""
        key = "lr_scheduler.last_epoch"
        
        result = pruner.should_prune_key(key, "unknown")
        
        self.assertTrue(result)
    
    def test_ema_keys_pruned_in_unknown_format(self):
        """Test that EMA keys are pruned even in unknown format."""
        key = "ema.model.weight"
        
        result = pruner.should_prune_key(key, "unknown")
        
        self.assertTrue(result)
    
    def test_loss_keys_pruned_in_unknown_format(self):
        """Test that loss keys are pruned even in unknown format."""
        key = "loss.running_avg"
        
        result = pruner.should_prune_key(key, "unknown")
        
        self.assertTrue(result)


class TestPruneStateDict(unittest.TestCase):
    """Tests for prune_state_dict function."""
    
    def test_prunes_sd_checkpoint_correctly(self):
        """Test pruning of SD checkpoint format."""
        state_dict = {
            "model.diffusion_model.layer1.weight": torch.randn(10, 10),
            "first_stage_model.encoder.weight": torch.randn(10, 10),
            "optimizer_state.layer.weight": torch.randn(10, 10),
            "epoch": torch.tensor(10),
            "global_step": torch.tensor(5000),
        }
        
        pruned, removed_count, format_type = pruner.prune_state_dict(state_dict)
        
        self.assertEqual(format_type, "sd_checkpoint")
        self.assertEqual(removed_count, 3)  # optimizer, epoch, global_step
        self.assertIn("model.diffusion_model.layer1.weight", pruned)
        self.assertIn("first_stage_model.encoder.weight", pruned)
        self.assertNotIn("optimizer_state.layer.weight", pruned)
        self.assertNotIn("epoch", pruned)
        self.assertNotIn("global_step", pruned)
    
    def test_prunes_with_explicit_format(self):
        """Test pruning with explicitly provided format."""
        state_dict = {
            "model.diffusion_model.layer.weight": torch.randn(10, 10),
            "other_key": torch.randn(10, 10),
        }
        
        pruned, removed_count, format_type = pruner.prune_state_dict(
            state_dict, format_type="sd_checkpoint"
        )
        
        self.assertEqual(format_type, "sd_checkpoint")
        self.assertEqual(removed_count, 1)  # other_key pruned
        self.assertIn("model.diffusion_model.layer.weight", pruned)
        self.assertNotIn("other_key", pruned)
    
    def test_auto_detects_format_when_none(self):
        """Test that format is auto-detected when not provided."""
        state_dict = {
            "lora_unet_layer": torch.randn(10, 10),
            "lora_te_layer": torch.randn(10, 10),
        }
        
        pruned, removed_count, format_type = pruner.prune_state_dict(state_dict)
        
        self.assertEqual(format_type, "lora")
    
    def test_conservative_pruning_for_standalone_files(self):
        """Test conservative pruning for standalone files."""
        # Need more than 5 keys to avoid embedding detection
        state_dict = {
            "encoder.conv_in.weight": torch.randn(10, 10),
            "encoder.mid.block.weight": torch.randn(10, 10),
            "decoder.conv_out.weight": torch.randn(10, 10),
            "decoder.mid.block.weight": torch.randn(10, 10),
            "quant_conv.weight": torch.randn(10, 10),
            "post_quant_conv.weight": torch.randn(10, 10),
            "custom_data": torch.randn(10, 10),  # Should be kept
        }
        
        pruned, removed_count, format_type = pruner.prune_state_dict(state_dict)
        
        self.assertEqual(format_type, "vae")
        self.assertEqual(removed_count, 0)  # Nothing pruned (conservative)
        self.assertEqual(len(pruned), 7)
    
    def test_empty_state_dict_returns_empty(self):
        """Test pruning empty state dict."""
        pruned, removed_count, format_type = pruner.prune_state_dict({})
        
        self.assertEqual(format_type, "unknown")
        self.assertEqual(removed_count, 0)
        self.assertEqual(len(pruned), 0)
    
    def test_preserves_tensor_values(self):
        """Test that pruning preserves tensor values."""
        original_tensor = torch.randn(10, 10)
        state_dict = {
            "model.diffusion_model.layer.weight": original_tensor,
        }
        
        pruned, _, _ = pruner.prune_state_dict(state_dict, format_type="sd_checkpoint")
        
        self.assertTrue(torch.equal(
            pruned["model.diffusion_model.layer.weight"],
            original_tensor
        ))


class TestShouldSkipPruning(unittest.TestCase):
    """Tests for should_skip_pruning function."""
    
    def test_skip_embedding_format(self):
        """Test that embedding format skips pruning."""
        result = pruner.should_skip_pruning("embedding")
        
        self.assertTrue(result)
    
    def test_skip_vae_format(self):
        """Test that VAE format skips pruning."""
        result = pruner.should_skip_pruning("vae")
        
        self.assertTrue(result)
    
    def test_skip_lora_format(self):
        """Test that LoRA format skips pruning."""
        result = pruner.should_skip_pruning("lora")
        
        self.assertTrue(result)
    
    def test_skip_upscaler_format(self):
        """Test that upscaler format skips pruning."""
        result = pruner.should_skip_pruning("upscaler")
        
        self.assertTrue(result)
    
    def test_skip_unknown_format(self):
        """Test that unknown format skips pruning."""
        result = pruner.should_skip_pruning("unknown")
        
        self.assertTrue(result)
    
    def test_dont_skip_sd_checkpoint(self):
        """Test that SD checkpoint format does NOT skip pruning."""
        result = pruner.should_skip_pruning("sd_checkpoint")
        
        self.assertFalse(result)


class TestGetFormatDescription(unittest.TestCase):
    """Tests for get_format_description function."""
    
    def test_sd_checkpoint_description(self):
        """Test description for SD checkpoint format."""
        result = pruner.get_format_description("sd_checkpoint")
        
        self.assertEqual(result, "Stable Diffusion checkpoint")
    
    def test_vae_description(self):
        """Test description for VAE format."""
        result = pruner.get_format_description("vae")
        
        self.assertEqual(result, "Standalone VAE")
    
    def test_lora_description(self):
        """Test description for LoRA format."""
        result = pruner.get_format_description("lora")
        
        self.assertEqual(result, "LoRA (Low-Rank Adaptation)")
    
    def test_embedding_description(self):
        """Test description for embedding format."""
        result = pruner.get_format_description("embedding")
        
        self.assertEqual(result, "Textual inversion embedding")
    
    def test_upscaler_description(self):
        """Test description for upscaler format."""
        result = pruner.get_format_description("upscaler")
        
        self.assertEqual(result, "Upscaler model (ESRGAN/Real-ESRGAN)")
    
    def test_unknown_description(self):
        """Test description for unknown format."""
        result = pruner.get_format_description("unknown")
        
        self.assertEqual(result, "Unknown format")
    
    def test_invalid_format_returns_unknown(self):
        """Test that invalid format returns unknown description."""
        result = pruner.get_format_description("invalid_format")
        
        self.assertEqual(result, "Unknown format")


if __name__ == '__main__':
    unittest.main()
