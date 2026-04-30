import unittest
from parameterized import parameterized
import torch
from torch import Tensor
import torch.nn as nn
import neural_net_layers as nnl


class TestNeuralNetLayers(unittest.TestCase):

    @parameterized.expand([
        (nnl.CausalSelfAttention, dict(num_heads=2)),
        (nnl.CausalSelfAttention, dict(num_heads=2, dropout=0.2)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4)),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                       q_norm=nnl.RMSNorm(4), k_norm=nnl.RMSNorm(4))),
        (nnl.CausalSelfAttention, dict(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                       kv_shared_layer_idx=0)),
        (nnl.PositionEmbedding, dict(num_embeddings=27, embedding_dim=4)),
        (nnl.Summation, [nn.Embedding(27, 4),
                         nnl.PositionEmbedding(8, 4)]),
        (nnl.ResidualConnection, [nn.LayerNorm(4), nn.Linear(4, 8)]),
        (nnl.SoftmaxOnLast, dict(dim=-1)),
        (nnl.RMSNorm, dict(normalized_shape=4)),
        (nnl.GatedMLP, dict(in_features=4, intermediate_size=8)),
        (nnl.ScaledEmbedding, dict(num_embeddings=27, embedding_dim=4, scale=2.0)),
        (nnl.TransformerBlock, dict(
            attn_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            mlp_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)))),
        (nnl.TransformerBlock, dict(
            attn_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            mlp_block=nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
            post_attn_norm=nnl.RMSNorm(4), post_mlp_norm=nnl.RMSNorm(4),
            post_norm_on_residual=False)),
    ])
    def test_layer_init(self, layer_class: type, layer_args: dict | list):
        layer = layer_class(**layer_args) if isinstance(layer_args, dict) else layer_class(*layer_args)

        self.assertIsInstance(layer, nn.Module)

    @parameterized.expand([
        (nnl.CausalSelfAttention(2),
         torch.randn(5, 8, 12), (5, 8, 4)),
        (nnl.CausalSelfAttention(3, 0.2),
         torch.randn(5, 5, 45), (5, 5, 15)),
        # GQA: 4 query heads, 2 kv heads, head_dim=4 -> qkv_dim = 4*4 + 2*2*4 = 32
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2),
         torch.randn(2, 6, 32), (2, 6, 16)),
        # GQA + RoPE: same dims with rope_theta
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2, rope_theta=10000.0),
         torch.randn(2, 6, 32), (2, 6, 16)),
        # GQA + RoPE with precomputed inv_freq buffer
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4),
         torch.randn(2, 6, 32), (2, 6, 16)),
        # GQA + RoPE + q_norm/k_norm (Gemma 2+ pattern)
        (nnl.CausalSelfAttention(num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
                                 q_norm=nnl.RMSNorm(4), k_norm=nnl.RMSNorm(4)),
         torch.randn(2, 6, 32), (2, 6, 16)),
        (nnl.PositionEmbedding(27, 4),
         torch.randint(0, 27, (5, 8)), (8, 4)),
        (nnl.Summation(nn.Embedding(27, 4),
                       nnl.PositionEmbedding(8, 4)),
         torch.randint(0, 27, (5, 8)), (5, 8, 4)),
        (nn.Sequential(nn.LayerNorm(4, bias=False),
                       nn.Linear(4, 12, False),
                       nnl.CausalSelfAttention(4, 0.2),
                       nn.Linear(4, 4, False),
                       nn.Dropout(0.2)),
         torch.randn(5, 8, 4), (5, 8, 4)),
        # RMSNorm forward
        (nnl.RMSNorm(4),
         torch.randn(5, 8, 4), (5, 8, 4)),
        # GatedMLP forward
        (nnl.GatedMLP(4, 8),
         torch.randn(5, 8, 4), (5, 8, 4)),
        # ScaledEmbedding forward
        (nnl.ScaledEmbedding(27, 4, scale=2.0),
         torch.randint(0, 27, (5, 8)), (5, 8, 4)),
        # TransformerBlock forward (no post-norms)
        (nnl.TransformerBlock(
            attn_block=nn.Sequential(
                nnl.RMSNorm(4),
                nn.Linear(4, 12, False),
                nnl.CausalSelfAttention(4),
                nn.Linear(4, 4, False)),
            mlp_block=nn.Sequential(
                nnl.RMSNorm(4),
                nnl.GatedMLP(4, 8))),
         torch.randn(2, 6, 4), (2, 6, 4)),
        # TransformerBlock forward (with post-norms, Gemma 3 pattern)
        (nnl.TransformerBlock(
            attn_block=nn.Sequential(
                nnl.RMSNorm(4),
                nn.Linear(4, 12, False),
                nnl.CausalSelfAttention(4),
                nn.Linear(4, 4, False)),
            mlp_block=nn.Sequential(
                nnl.RMSNorm(4),
                nnl.GatedMLP(4, 8)),
            post_attn_norm=nnl.RMSNorm(4),
            post_mlp_norm=nnl.RMSNorm(4)),
         torch.randn(2, 6, 4), (2, 6, 4)),
        # TransformerBlock forward (with post-norms, Gemma 2 pattern)
        (nnl.TransformerBlock(
            attn_block=nn.Sequential(
                nnl.RMSNorm(4),
                nn.Linear(4, 12, False),
                nnl.CausalSelfAttention(4),
                nn.Linear(4, 4, False)),
            mlp_block=nn.Sequential(
                nnl.RMSNorm(4),
                nnl.GatedMLP(4, 8)),
            post_attn_norm=nnl.RMSNorm(4),
            post_mlp_norm=nnl.RMSNorm(4),
            post_norm_on_residual=False),
         torch.randn(2, 6, 4), (2, 6, 4)),
        # Full GPT-2 style model
        (nn.Sequential(
            nnl.Summation(nn.Embedding(27, 4),
                          nnl.PositionEmbedding(8, 4)),
            nn.Dropout(0.2),
           *[nnl.ResidualConnection(
               nn.Sequential(
                   nn.LayerNorm(4, bias=False),
                   nn.Linear(4, 12, False),
                   nnl.CausalSelfAttention(4, 0.2),
                   nn.Linear(4, 4, False),
                   nn.Dropout(0.2)
               ),
               nn.Sequential(
                   nn.LayerNorm(4, bias=False),
                   nn.Linear(4, 16, False),
                   nn.GELU(),
                   nn.Linear(16, 4, False),
                  nn.Dropout(0.2)))
               for _ in range(2)],
            nn.LayerNorm(4, bias=False),
            nn.Linear(4, 27, bias=False),
            nnl.SoftmaxOnLast(dim=-1)),
         torch.randint(0, 27, (5, 8)), (5, 27)),
    ])
    def test_forward(self, layer: nn.Module, input_data: Tensor, expected_out_shape: tuple):
        output: Tensor = layer(input_data)

        self.assertIsNotNone(output)
        self.assertEqual(expected_out_shape, tuple(output.shape))

    def test_qk_norm_modifies_attention_output(self):
        """When q_norm/k_norm are provided, attention output differs from unnormalized."""
        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        batch, seq = 1, 3

        attn_no_norm = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim)
        attn_with_norm = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim,
            q_norm=nnl.RMSNorm(head_dim), k_norm=nnl.RMSNorm(head_dim))

        # Use non-unit weights so norms have a visible effect
        attn_with_norm.q_norm.weight.data.fill_(2.0)
        attn_with_norm.k_norm.weight.data.fill_(0.5)

        torch.manual_seed(42)
        qkv = torch.randn(batch, seq, qkv_dim)

        out_no_norm = attn_no_norm(qkv)
        out_with_norm = attn_with_norm(qkv)

        self.assertEqual(out_no_norm.shape, out_with_norm.shape)
        self.assertFalse(torch.allclose(out_no_norm, out_with_norm, atol=1e-6),
                         "q_norm/k_norm should change the attention output")

    def test_qk_norm_registers_as_submodules(self):
        """q_norm and k_norm appear in state_dict when provided."""
        attn = nnl.CausalSelfAttention(
            num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4,
            q_norm=nnl.RMSNorm(4), k_norm=nnl.RMSNorm(4))
        sd_keys = set(attn.state_dict().keys())
        self.assertIn("q_norm.weight", sd_keys)
        self.assertIn("k_norm.weight", sd_keys)

    def test_no_qk_norm_no_extra_state_keys(self):
        """Without q_norm/k_norm, state_dict has no norm keys."""
        attn = nnl.CausalSelfAttention(
            num_heads=4, num_kv_heads=2, rope_theta=10000.0, head_dim=4)
        sd_keys = set(attn.state_dict().keys())
        self.assertNotIn("q_norm.weight", sd_keys)
        self.assertNotIn("k_norm.weight", sd_keys)

    def test_kv_shared_layer_uses_reference_kv(self):
        """KV-shared layer uses K/V from reference layer, not its own projection."""
        from kv_cache import KVCache

        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        batch, seq = 1, 3

        # Layer 0: non-shared reference
        attn_ref = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim)
        # Layer 1: shared, references layer 0
        attn_shared = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim,
            kv_shared_layer_idx=0)

        cache = KVCache(num_layers=2)
        store = {}
        attn_ref.set_kv_cache(cache, 0, store)
        attn_shared.set_kv_cache(cache, 1, store)

        # Use different QKV inputs for each layer
        torch.manual_seed(42)
        qkv_ref = torch.randn(batch, seq, qkv_dim)
        qkv_shared = torch.randn(batch, seq, qkv_dim)

        out_ref = attn_ref(qkv_ref)
        out_shared = attn_shared(qkv_shared)

        # Both layers should have cached entries
        self.assertEqual(cache.seq_len(0), seq)
        self.assertEqual(cache.seq_len(1), seq)

        # The shared layer should NOT produce the same output as if it used its own K/V
        attn_own = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim)
        cache_own = KVCache(num_layers=1)
        attn_own.set_kv_cache(cache_own, 0)
        out_own = attn_own(qkv_shared)

        # Shared output should differ from non-shared output (different K/V source)
        self.assertFalse(torch.allclose(out_shared, out_own, atol=1e-6),
                         "Shared layer should use reference K/V, not its own")

    def test_kv_sharing_not_active_without_store(self):
        """Without a share store, kv_shared_layer_idx has no effect."""
        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim

        attn = nnl.CausalSelfAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            rope_theta=10000.0, head_dim=head_dim,
            kv_shared_layer_idx=0)

        # No store set → should use own K/V without error
        qkv = torch.randn(1, 3, qkv_dim)
        out = attn(qkv)
        self.assertEqual(out.shape, (1, 3, num_heads * head_dim))

    def test_rope_offset_uses_own_layer_idx_with_kv_cache(self):
        """Each attention layer must read its own KV cache seq_len for RoPE offset,
        not layer 0's.  When layers run sequentially during a forward pass,
        layer 0 appends to its cache before layer 1 executes.  If layer 1
        reads layer 0's (already-updated) cache length, the RoPE positions
        are wrong for every layer after the first."""
        from kv_cache import KVCache

        num_heads, num_kv_heads, head_dim = 4, 2, 4
        qkv_dim = num_heads * head_dim + 2 * num_kv_heads * head_dim
        batch, seq = 1, 3

        attn0 = nnl.CausalSelfAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                          rope_theta=10000.0, head_dim=head_dim)
        attn1 = nnl.CausalSelfAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                          rope_theta=10000.0, head_dim=head_dim)

        cache = KVCache(num_layers=2)
        attn0.set_kv_cache(cache, layer_idx=0)
        attn1.set_kv_cache(cache, layer_idx=1)

        qkv = torch.randn(batch, seq, qkv_dim)

        # Prefill: both layers process the full sequence
        attn0(qkv)  # layer 0 cache now has `seq` entries
        attn1(qkv)  # layer 1 should use its own cache (was empty) for offset

        # After prefill both caches must have the same length
        self.assertEqual(cache.seq_len(0), seq)
        self.assertEqual(cache.seq_len(1), seq)

        # Incremental decode: single new token
        qkv_one = torch.randn(batch, 1, qkv_dim)
        out0 = attn0(qkv_one)  # layer 0 cache → seq+1
        out1 = attn1(qkv_one)  # layer 1 must read own cache (seq), not layer 0 (seq+1)

        self.assertEqual(cache.seq_len(0), seq + 1)
        self.assertEqual(cache.seq_len(1), seq + 1)

        # Both layers have identical config (same inv_freq buffer, no trainable
        # weights) and received the same inputs with the same correct RoPE
        # offsets (0 during prefill, seq during decode).  Their outputs must
        # match.  If the bug were present, layer 1 would have used layer 0's
        # already-updated cache length (seq+1 instead of seq) for its RoPE
        # offset, producing a different result.
        self.assertTrue(torch.allclose(out0, out1, atol=1e-6),
                        "Layer 0 and Layer 1 outputs should match when using correct per-layer offsets")


if __name__ == '__main__':
    unittest.main()
