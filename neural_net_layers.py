import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as nnf


class CausalSelfAttention(nn.Module):
    def __init__(self, num_heads: int, dropout: float=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, query_key_value: Tensor) -> Tensor:
        batch_size, block_size, embedding_dims = query_key_value.size()
        embedding_dim = embedding_dims // 3
        # extract query, key and value splits
        qkv_splits = query_key_value.split(embedding_dim, dim=2)
        # prep query, key and value for matrix multiply shaped batch size, block size, head size, # of heads
        head_size = embedding_dim // self.num_heads
        q, k, v = (s.view(batch_size, block_size, self.num_heads, head_size).transpose(1, 2) for s in qkv_splits)
        # apply scaled dot product attention formula
        # (batch size, # of heads, block size, block size) x (batch size, # of heads, block size, head size) ->
        # (batch size, # of heads, block size, head size)
        dropout = self.dropout if self.training else 0.0
        output = nnf.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True)
        # combine head outputs -> (batch size, block size, embedding dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, block_size, embedding_dim)
        # return output
        return output


class PositionEmbedding(nn.Embedding):
    def forward(self, input_data: Tensor) -> Tensor:
        _, num_positions = input_data.shape
        positions = torch.arange(num_positions, dtype=torch.long, device=input_data.device)
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
