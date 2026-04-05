import os
import unittest
from unittest.mock import patch
import torch
from kv_cache import KVCache, TurboQuantKVCache, create_kv_cache, KVCacheMetrics


class TestKVCache(unittest.TestCase):

    def test_init_creates_empty_cache(self):
        cache = KVCache(num_layers=3)
        for i in range(3):
            k, v = cache.get(i)
            self.assertIsNone(k)
            self.assertIsNone(v)
        self.assertEqual(cache.seq_len(0), 0)

    def test_append_single_step(self):
        cache = KVCache(num_layers=1)
        key = torch.randn(1, 2, 4, 8)  # B, H, S, D
        value = torch.randn(1, 2, 4, 8)

        full_k, full_v = cache.append(0, key, value)

        self.assertTrue(torch.equal(full_k, key))
        self.assertTrue(torch.equal(full_v, value))
        self.assertEqual(cache.seq_len(0), 4)

    def test_append_multiple_steps_concatenates(self):
        cache = KVCache(num_layers=1)
        k1 = torch.randn(1, 2, 3, 8)
        v1 = torch.randn(1, 2, 3, 8)
        k2 = torch.randn(1, 2, 1, 8)
        v2 = torch.randn(1, 2, 1, 8)

        cache.append(0, k1, v1)
        full_k, full_v = cache.append(0, k2, v2)

        self.assertEqual(full_k.shape, (1, 2, 4, 8))
        self.assertEqual(full_v.shape, (1, 2, 4, 8))
        self.assertEqual(cache.seq_len(0), 4)
        self.assertTrue(torch.equal(full_k[:, :, :3, :], k1))
        self.assertTrue(torch.equal(full_k[:, :, 3:, :], k2))

    def test_get_returns_cached_tensors(self):
        cache = KVCache(num_layers=2)
        k0 = torch.randn(1, 2, 3, 8)
        v0 = torch.randn(1, 2, 3, 8)
        cache.append(0, k0, v0)

        got_k, got_v = cache.get(0)
        self.assertTrue(torch.equal(got_k, k0))
        self.assertTrue(torch.equal(got_v, v0))

        # Layer 1 should still be empty
        k1, v1 = cache.get(1)
        self.assertIsNone(k1)
        self.assertIsNone(v1)

    def test_clear_resets_cache(self):
        cache = KVCache(num_layers=2)
        cache.append(0, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))
        cache.append(1, torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))

        cache.clear()

        for i in range(2):
            k, v = cache.get(i)
            self.assertIsNone(k)
            self.assertIsNone(v)
        self.assertEqual(cache.seq_len(0), 0)

    def test_metrics_updated_on_append(self):
        cache = KVCache(num_layers=1)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)

        cache.append(0, k, v)
        m = cache.metrics

        self.assertEqual(m.num_appends, 1)
        self.assertEqual(m.total_entries, 4)
        self.assertGreater(m.memory_bytes, 0)
        self.assertEqual(m.compression_ratio, 1.0)
        self.assertGreater(m.last_append_latency_ms, 0.0)

    def test_multi_layer_cache(self):
        cache = KVCache(num_layers=4)
        for i in range(4):
            cache.append(i, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))

        self.assertEqual(cache.metrics.total_entries, 12)  # 3 per layer * 4 layers
        for i in range(4):
            self.assertEqual(cache.seq_len(i), 3)


class TestTurboQuantKVCache(unittest.TestCase):

    def test_init_creates_empty_cache(self):
        cache = TurboQuantKVCache(num_layers=2)
        for i in range(2):
            k, v = cache.get(i)
            self.assertIsNone(k)
            self.assertIsNone(v)

    def test_append_returns_dequantized_tensors(self):
        cache = TurboQuantKVCache(num_layers=1)
        key = torch.randn(1, 2, 4, 8)
        value = torch.randn(1, 2, 4, 8)

        full_k, full_v = cache.append(0, key, value)

        # Shapes should match
        self.assertEqual(full_k.shape, key.shape)
        self.assertEqual(full_v.shape, value.shape)
        # Dequantized values should be close to originals (int8 quantization error)
        self.assertTrue(torch.allclose(full_k, key, atol=0.05))
        self.assertTrue(torch.allclose(full_v, value, atol=0.05))

    def test_stored_as_int8(self):
        cache = TurboQuantKVCache(num_layers=1)
        key = torch.randn(1, 2, 4, 8)
        value = torch.randn(1, 2, 4, 8)

        cache.append(0, key, value)

        # Internal storage should be int8
        stored_k, stored_v = cache.get(0)
        self.assertEqual(stored_k.dtype, torch.int8)
        self.assertEqual(stored_v.dtype, torch.int8)

    def test_compression_ratio_greater_than_one(self):
        cache = TurboQuantKVCache(num_layers=1)
        key = torch.randn(1, 2, 4, 8)
        value = torch.randn(1, 2, 4, 8)

        cache.append(0, key, value)
        m = cache.metrics

        # int8 + per-token scales still smaller than float32
        self.assertGreater(m.compression_ratio, 1.0)
        self.assertLess(m.compressed_memory_bytes, m.memory_bytes)

    def test_append_multiple_steps(self):
        cache = TurboQuantKVCache(num_layers=1)
        k1 = torch.randn(1, 2, 3, 8)
        v1 = torch.randn(1, 2, 3, 8)
        k2 = torch.randn(1, 2, 1, 8)
        v2 = torch.randn(1, 2, 1, 8)

        cache.append(0, k1, v1)
        full_k, full_v = cache.append(0, k2, v2)

        self.assertEqual(full_k.shape, (1, 2, 4, 8))
        self.assertEqual(full_v.shape, (1, 2, 4, 8))
        self.assertEqual(cache.seq_len(0), 4)

    def test_clear_resets_scales(self):
        cache = TurboQuantKVCache(num_layers=1)
        cache.append(0, torch.randn(1, 2, 3, 8), torch.randn(1, 2, 3, 8))

        cache.clear()

        k, v = cache.get(0)
        self.assertIsNone(k)
        self.assertIsNone(v)
        self.assertEqual(cache.seq_len(0), 0)

    def test_quantize_dequantize_roundtrip(self):
        tensor = torch.randn(2, 4, 8)
        q, scale = TurboQuantKVCache._quantize(tensor)
        recovered = TurboQuantKVCache._dequantize(q, scale)

        self.assertEqual(q.dtype, torch.int8)
        self.assertTrue(torch.allclose(recovered, tensor, atol=0.05))

    def test_quantize_zero_tensor(self):
        tensor = torch.zeros(2, 4, 8)
        q, scale = TurboQuantKVCache._quantize(tensor)
        recovered = TurboQuantKVCache._dequantize(q, scale)

        self.assertTrue(torch.equal(recovered, tensor))

    def test_per_token_scales_preserve_accuracy_across_appends(self):
        cache = TurboQuantKVCache(num_layers=1)
        # First append: small values
        k1 = torch.randn(1, 2, 3, 8) * 0.01
        v1 = torch.randn(1, 2, 3, 8) * 0.01
        # Second append: large values (very different range)
        k2 = torch.randn(1, 2, 1, 8) * 100.0
        v2 = torch.randn(1, 2, 1, 8) * 100.0

        cache.append(0, k1, v1)
        full_k, full_v = cache.append(0, k2, v2)

        # The first 3 positions should still be close to k1
        # With per-token scales this works; with a global scale it would fail
        self.assertTrue(torch.allclose(full_k[:, :, :3, :], k1, atol=0.01))
        self.assertTrue(torch.allclose(full_k[:, :, 3:, :], k2, atol=5.0))


class TestCreateKVCache(unittest.TestCase):

    def test_default_creates_basic_cache(self):
        cache = create_kv_cache(3)
        self.assertIsInstance(cache, KVCache)
        self.assertNotIsInstance(cache, TurboQuantKVCache)

    @patch.dict(os.environ, {"TURBO_QUANT_KV_CACHE": "1"})
    def test_env_flag_creates_turbo_quant_cache(self):
        import kv_cache
        orig = kv_cache.TURBO_QUANT_ENABLED
        kv_cache.TURBO_QUANT_ENABLED = True
        try:
            cache = create_kv_cache(3)
            self.assertIsInstance(cache, TurboQuantKVCache)
        finally:
            kv_cache.TURBO_QUANT_ENABLED = orig


if __name__ == '__main__':
    unittest.main()
