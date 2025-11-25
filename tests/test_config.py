"""
Tests for config.py module.

Tests configuration constants and utility functions for architecture
detection and key pruning/skipping.
"""

import unittest
from model_merger import config


class TestArchitectureDetection(unittest.TestCase):
    """Tests for detect_architecture_from_filename function."""
    
    def test_pony_detection_lowercase(self):
        """Test detection of Pony architecture from lowercase filename."""
        result = config.detect_architecture_from_filename("pony_realistic_v2.safetensors")
        self.assertEqual(result, "Pony")
    
    def test_pony_detection_uppercase(self):
        """Test detection of Pony architecture from uppercase filename."""
        result = config.detect_architecture_from_filename("PONY_MODEL.safetensors")
        self.assertEqual(result, "Pony")
    
    def test_pony_detection_mixed_case(self):
        """Test detection of Pony architecture from mixed case filename."""
        result = config.detect_architecture_from_filename("PonyXL_Base.safetensors")
        self.assertEqual(result, "Pony")
    
    def test_illustrious_detection(self):
        """Test detection of Illustrious architecture."""
        result = config.detect_architecture_from_filename("illustrious_v1.safetensors")
        self.assertEqual(result, "Illustrious")
    
    def test_illustrious_short_form(self):
        """Test detection of Illustrious via 'illus' pattern."""
        result = config.detect_architecture_from_filename("illus_anime.safetensors")
        self.assertEqual(result, "Illustrious")
    
    def test_sdxl_detection(self):
        """Test detection of SDXL architecture."""
        result = config.detect_architecture_from_filename("my_sdxl_model.safetensors")
        self.assertEqual(result, "SDXL")
    
    def test_sdxl_xl_pattern(self):
        """Test detection of SDXL via 'xl' pattern."""
        result = config.detect_architecture_from_filename("anime_xl_base.safetensors")
        self.assertEqual(result, "SDXL")
    
    def test_sd15_detection(self):
        """Test detection of SD 1.5 architecture."""
        result = config.detect_architecture_from_filename("sd15_model.safetensors")
        self.assertEqual(result, "SD1.5")
    
    def test_sd15_v1_5_pattern(self):
        """Test detection of SD 1.5 via 'v1-5' pattern."""
        result = config.detect_architecture_from_filename("model_v1-5.safetensors")
        self.assertEqual(result, "SD1.5")
    
    def test_sd21_detection(self):
        """Test detection of SD 2.1 architecture."""
        result = config.detect_architecture_from_filename("sd21_768.safetensors")
        self.assertEqual(result, "SD2.1")
    
    def test_noobai_detection(self):
        """Test detection of Noobai architecture."""
        result = config.detect_architecture_from_filename("noobai_v2.safetensors")
        self.assertEqual(result, "Noobai")
    
    def test_noob_short_form(self):
        """Test detection of Noobai via 'noob' pattern."""
        result = config.detect_architecture_from_filename("noob_anime.safetensors")
        self.assertEqual(result, "Noobai")
    
    def test_default_architecture(self):
        """Test that unknown filenames return default architecture."""
        result = config.detect_architecture_from_filename("my_cool_model.safetensors")
        self.assertEqual(result, config.DEFAULT_ARCHITECTURE)
    
    def test_empty_filename(self):
        """Test handling of empty filename."""
        result = config.detect_architecture_from_filename("")
        self.assertEqual(result, config.DEFAULT_ARCHITECTURE)
    
    def test_priority_pony_over_xl(self):
        """Test that 'Pony' is detected before 'XL' pattern."""
        # Pony should match before XL because of iteration order
        result = config.detect_architecture_from_filename("ponyxl_model.safetensors")
        self.assertEqual(result, "Pony")


class TestShouldPruneKey(unittest.TestCase):
    """Tests for should_prune_key function."""
    
    def test_diffusion_model_key_not_pruned(self):
        """Test that diffusion model keys are kept."""
        result = config.should_prune_key("model.diffusion_model.layer1.weight")
        self.assertFalse(result)
    
    def test_first_stage_model_key_not_pruned(self):
        """Test that VAE keys are kept."""
        result = config.should_prune_key("first_stage_model.encoder.conv_in.weight")
        self.assertFalse(result)
    
    def test_cond_stage_model_key_not_pruned(self):
        """Test that text encoder keys (SD 1.x/2.x) are kept."""
        result = config.should_prune_key("cond_stage_model.transformer.weight")
        self.assertFalse(result)
    
    def test_conditioner_key_not_pruned(self):
        """Test that text encoder keys (SDXL) are kept."""
        result = config.should_prune_key("conditioner.embedders.0.model.weight")
        self.assertFalse(result)
    
    def test_optimizer_key_pruned(self):
        """Test that optimizer keys are pruned."""
        result = config.should_prune_key("optimizer.state.123")
        self.assertTrue(result)
    
    def test_ema_key_pruned(self):
        """Test that EMA keys are pruned."""
        result = config.should_prune_key("ema.shadow_params")
        self.assertTrue(result)
    
    def test_epoch_key_pruned(self):
        """Test that epoch keys are pruned."""
        result = config.should_prune_key("epoch")
        self.assertTrue(result)
    
    def test_global_step_pruned(self):
        """Test that global_step keys are pruned."""
        result = config.should_prune_key("global_step")
        self.assertTrue(result)
    
    def test_random_key_pruned(self):
        """Test that random/unknown keys are pruned."""
        result = config.should_prune_key("some_random_key")
        self.assertTrue(result)


class TestShouldSkipMergeKey(unittest.TestCase):
    """Tests for should_skip_merge_key function."""
    
    def test_position_ids_skipped(self):
        """Test that position_ids key is skipped during merge."""
        result = config.should_skip_merge_key(
            "cond_stage_model.transformer.text_model.embeddings.position_ids"
        )
        self.assertTrue(result)
    
    def test_normal_key_not_skipped(self):
        """Test that normal model keys are not skipped."""
        result = config.should_skip_merge_key("model.diffusion_model.layer1.weight")
        self.assertFalse(result)
    
    def test_vae_key_not_skipped(self):
        """Test that VAE keys are not skipped."""
        result = config.should_skip_merge_key("first_stage_model.encoder.conv_in.weight")
        self.assertFalse(result)


class TestConfigConstants(unittest.TestCase):
    """Tests for configuration constants."""
    
    def test_supported_model_extensions(self):
        """Test that supported model extensions include safetensors."""
        self.assertIn('.safetensors', config.SUPPORTED_MODEL_EXTENSIONS)
    
    def test_supported_vae_extensions(self):
        """Test that supported VAE extensions include safetensors."""
        self.assertIn('.safetensors', config.SUPPORTED_VAE_EXTENSIONS)
    
    def test_vae_key_prefix(self):
        """Test VAE key prefix is correct."""
        self.assertEqual(config.VAE_KEY_PREFIX, 'first_stage_model.')
    
    def test_default_output_precision(self):
        """Test default output precision setting."""
        self.assertEqual(config.DEFAULT_MERGE_SETTINGS['output_precision'], 'match')
    
    def test_default_device(self):
        """Test default device is CPU."""
        self.assertEqual(config.DEFAULT_MERGE_SETTINGS['device'], 'cpu')
    
    def test_keep_key_prefixes_content(self):
        """Test that KEEP_KEY_PREFIXES has expected entries."""
        self.assertIn('model.diffusion_model.', config.KEEP_KEY_PREFIXES)
        self.assertIn('first_stage_model.', config.KEEP_KEY_PREFIXES)
        self.assertIn('cond_stage_model.', config.KEEP_KEY_PREFIXES)
        self.assertIn('conditioner.', config.KEEP_KEY_PREFIXES)


if __name__ == '__main__':
    unittest.main()
