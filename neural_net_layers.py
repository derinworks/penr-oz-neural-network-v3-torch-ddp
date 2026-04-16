import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as nnf


class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads: int, dropout: float=0.0,
                 num_kv_heads: int=None, rope_theta: float=None,
                 head_dim: int=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.dropout = dropout
        self.rope_theta = rope_theta
        self._kv_cache = None
        self._layer_idx = 0
        if rope_theta is not None and head_dim is not None:
            inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            ))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def set_kv_cache(self, kv_cache, layer_idx: int):
        """Attach a KVCache for incremental decoding.

        :param kv_cache: KVCache instance (or None to disable).
        :param layer_idx: Layer index within the cache.
        """
        self._kv_cache = kv_cache
        self._layer_idx = layer_idx

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q: Tensor, k: Tensor, head_dim: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to query and key tensors."""
        seq_len = q.shape[2]
        device = q.device
        dtype = q.dtype
        if hasattr(self, "inv_freq"):
            inv_freq = self.inv_freq
        else:
            inv_freq = 1.0 / (self.rope_theta ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
            ))
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def forward(self, query_key_value: Tensor) -> Tensor:
        batch_size, block_size, total_dim = query_key_value.size()
        head_dim = total_dim // (self.num_heads + 2 * self.num_kv_heads)
        q_dim = self.num_heads * head_dim
        kv_dim = self.num_kv_heads * head_dim

        q, k, v = query_key_value.split([q_dim, kv_dim, kv_dim], dim=2)
        q = q.view(batch_size, block_size, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, block_size, self.num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, block_size, self.num_kv_heads, head_dim).transpose(1, 2)

        # Apply rotary position embeddings when enabled
        if self.rope_theta is not None:
            offset = self._kv_cache.seq_len(self._layer_idx) if self._kv_cache is not None else 0
            q, k = self._apply_rope(q, k, head_dim, offset)

        # Expand KV heads for grouped-query attention
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, n_rep, -1, -1).reshape(
                batch_size, self.num_heads, -1, head_dim)

        # If KV cache is attached, append and retrieve full k/v
        if self._kv_cache is not None:
            k, v = self._kv_cache.append(self._layer_idx, k, v)
            is_causal = (block_size > 1)
        else:
            is_causal = True

        # apply scaled dot product attention formula
        dropout = self.dropout if self.training else 0.0
        output = nnf.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
        # combine head outputs -> (batch size, block size, q_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, block_size, q_dim)
        return output


class PositionEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._position_offset = 0

    @property
    def position_offset(self) -> int:
        return self._position_offset

    @position_offset.setter
    def position_offset(self, value: int):
        self._position_offset = value

    def forward(self, input_data: Tensor) -> Tensor:
        _, num_positions = input_data.shape
        positions = torch.arange(
            self._position_offset, self._position_offset + num_positions,
            dtype=torch.long, device=input_data.device
        )
        forwarded = super().forward(positions)
        return forwarded


class Summation(nn.Sequential):
    def forward(self, input_data: Tensor) -> Tensor:
        forwarded = self[0].forward(input_data)
        for layer in self[1:]:
            # please note: torch autograd fails with += in-place op, so use a = a + b instead
            forwarded = forwarded + layer(input_data)
        return forwarded


class ResidualConnection(nn.Sequential):
    def forward(self, forwarded: Tensor) -> Tensor:
        for layer in self:
            # please note: torch autograd fails with += in-place op, so use a = a + b instead
            forwarded = forwarded + layer(forwarded)
        return forwarded


class SoftmaxOnLast(nn.Softmax):
    def forward(self, logits: Tensor) -> Tensor:
        probs = super().forward(logits[:,-1,:])
        return probs


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_float = x.float()
        norm = x_float.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_float * norm).to(dtype) * self.weight


class GatedMLP(nn.Module):
    """Gated MLP with configurable activation (used in Gemma/LLaMA models)."""
    def __init__(self, in_features: int, intermediate_size: int,
                 bias: bool = False, activation: str = "gelu_pytorch_tanh"):
        super().__init__()
        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, in_features, bias=bias)
        if activation in ("silu", "swish"):
            self.act = nn.SiLU()
        elif activation == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class ScaledEmbedding(nn.Embedding):
    """Embedding with output scaled by a fixed factor."""
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 scale: float = 1.0, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.scale = scale

    def forward(self, input_data: Tensor) -> Tensor:
        return super().forward(input_data) * self.scale


class TransformerBlock(nn.Module):
    """Transformer decoder block with optional post-normalization.

    When *post_norm_on_residual* is ``True`` (default, Gemma 3+ pattern),
    post-norms are applied **after** the residual addition::

        h = post_attn_norm(x + attn_block(x))

    When ``False`` (Gemma 2 pattern), post-norms wrap only the branch
    output **before** it is added to the residual::

        h = x + post_attn_norm(attn_block(x))
    """
    def __init__(self, attn_block: nn.Module, mlp_block: nn.Module,
                 post_attn_norm: nn.Module = None, post_mlp_norm: nn.Module = None,
                 post_norm_on_residual: bool = True):
        super().__init__()
        self.attn_block = attn_block
        self.mlp_block = mlp_block
        self.post_attn_norm = post_attn_norm
        self.post_mlp_norm = post_mlp_norm
        self.post_norm_on_residual = post_norm_on_residual

    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn_block(x)
        if self.post_attn_norm is not None and not self.post_norm_on_residual:
            attn_out = self.post_attn_norm(attn_out)
        h = x + attn_out
        if self.post_attn_norm is not None and self.post_norm_on_residual:
            h = self.post_attn_norm(h)

        mlp_out = self.mlp_block(h)
        if self.post_mlp_norm is not None and not self.post_norm_on_residual:
            mlp_out = self.post_mlp_norm(mlp_out)
        out = h + mlp_out
        if self.post_mlp_norm is not None and self.post_norm_on_residual:
            out = self.post_mlp_norm(out)
        return out
