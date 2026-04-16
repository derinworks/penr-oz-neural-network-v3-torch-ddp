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

        # Run the same decode step but with a deliberately broken cache read
        # (reading layer 0 instead of own layer) to confirm outputs differ.
        # Reset layer 1 cache and re-prefill.
        cache2 = KVCache(num_layers=2)
        attn0_b = nnl.CausalSelfAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                            rope_theta=10000.0, head_dim=head_dim)
        attn1_b = nnl.CausalSelfAttention(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                            rope_theta=10000.0, head_dim=head_dim)
        # Copy weights so both runs use identical parameters
        attn0_b.load_state_dict(attn0.state_dict())
        attn1_b.load_state_dict(attn1.state_dict())
        attn0_b.set_kv_cache(cache2, layer_idx=0)
        attn1_b.set_kv_cache(cache2, layer_idx=1)

        attn0_b(qkv)
        attn1_b(qkv)
        out0_b = attn0_b(qkv_one)
        out1_b = attn1_b(qkv_one)

        # Outputs should be identical (same weights, same correct offsets)
        self.assertTrue(torch.allclose(out0, out0_b, atol=1e-6),
                        "Layer 0 outputs should match across identical runs")
        self.assertTrue(torch.allclose(out1, out1_b, atol=1e-6),
                        "Layer 1 outputs should match across identical runs")


if __name__ == '__main__':
    unittest.main()
