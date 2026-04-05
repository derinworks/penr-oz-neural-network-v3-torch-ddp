import logging
import os
import time
from dataclasses import dataclass, field
import torch
from torch import Tensor

log = logging.getLogger(__name__)

# Environment flag to enable TurboQuant compressed KV cache (default OFF)
TURBO_QUANT_ENABLED = os.environ.get("TURBO_QUANT_KV_CACHE", "0") == "1"


@dataclass
class KVCacheMetrics:
    """Lightweight metrics for KV cache usage."""
    num_appends: int = 0
    total_entries: int = 0
    memory_bytes: int = 0
    compressed_memory_bytes: int = 0
    compression_ratio: float = 1.0
    last_append_latency_ms: float = 0.0


class KVCache:
    """Storage for key/value tensors across generation steps.

    Maintains per-layer caches of key and value tensors that grow as
    new tokens are generated.  Supports append, retrieval, and clearing.
    """

    def __init__(self, num_layers: int = 0):
        self._keys: list[Tensor | None] = [None] * num_layers
        self._values: list[Tensor | None] = [None] * num_layers
        self._metrics = KVCacheMetrics()

    @property
    def metrics(self) -> KVCacheMetrics:
        return self._metrics

    def append(self, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Append new key/value tensors and return the full accumulated tensors.

        :param layer_idx: Attention layer index.
        :param key: New key tensor of shape (B, H, S_new, D).
        :param value: New value tensor of shape (B, H, S_new, D).
        :return: Tuple of (full_key, full_value) with all cached + new entries.
        """
        t0 = time.monotonic()
        new_bytes = (
            key.nelement() * key.element_size() + value.nelement() * value.element_size()
        )
        if self._keys[layer_idx] is not None:
            full_key = torch.cat([self._keys[layer_idx], key], dim=2)
            full_value = torch.cat([self._values[layer_idx], value], dim=2)
        else:
            full_key = key
            full_value = value
        self._keys[layer_idx] = full_key
        self._values[layer_idx] = full_value
        # Update metrics incrementally
        self._metrics.num_appends += 1
        self._metrics.total_entries += key.shape[2]
        self._metrics.memory_bytes += new_bytes
        self._metrics.compressed_memory_bytes = self._metrics.memory_bytes
        self._metrics.compression_ratio = 1.0
        self._metrics.last_append_latency_ms = (time.monotonic() - t0) * 1000
        return full_key, full_value

    def get(self, layer_idx: int) -> tuple[Tensor | None, Tensor | None]:
        """Retrieve cached key/value tensors for a layer.

        :param layer_idx: Attention layer index.
        :return: Tuple of (key, value) or (None, None) if empty.
        """
        return self._keys[layer_idx], self._values[layer_idx]

    def clear(self):
        """Clear all cached key/value tensors."""
        for i in range(len(self._keys)):
            self._keys[i] = None
            self._values[i] = None
        self._metrics = KVCacheMetrics()

    def seq_len(self, layer_idx: int = 0) -> int:
        """Return the current cached sequence length for a layer."""
        k = self._keys[layer_idx]
        return k.shape[2] if k is not None else 0

    def log_metrics(self):
        """Log current cache metrics."""
        m = self._metrics
        log.info(
            f"KVCache metrics: entries={m.total_entries}, "
            f"memory={m.memory_bytes / 1024:.1f}KB, "
            f"compression_ratio={m.compression_ratio:.2f}, "
            f"last_append={m.last_append_latency_ms:.3f}ms"
        )


class TurboQuantKVCache(KVCache):
    """KV cache with optional int8 scalar quantization.

    Stores cached key/value tensors in int8 format with per-token
    scale factors for reversible compress/decompress operations.
    Activated via TURBO_QUANT_KV_CACHE=1 environment flag.
    """

    def __init__(self, num_layers: int = 0):
        super().__init__(num_layers)
        self._scales_k: list[Tensor | None] = [None] * num_layers
        self._scales_v: list[Tensor | None] = [None] * num_layers

    @staticmethod
    def _quantize(tensor: Tensor) -> tuple[Tensor, Tensor]:
        """Compress a float tensor to int8 with per-token scale factors.

        :param tensor: Input float tensor of shape (B, H, S, D).
        :return: (quantized int8 tensor, per-token scale tensor of shape (B, H, S, 1)).
        """
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)
        scale = abs_max / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale

    @staticmethod
    def _dequantize(quantized: Tensor, scale: Tensor) -> Tensor:
        """Decompress an int8 tensor back to float using per-token scales.

        :param quantized: Int8 tensor of shape (B, H, S, D).
        :param scale: Per-token scale tensor of shape (B, H, S, 1).
        :return: Reconstructed float tensor.
        """
        return quantized.float() * scale

    def append(self, layer_idx: int, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Append new key/value tensors with int8 compression.

        New key/value entries are quantized with per-token scales before
        storage. Scales are concatenated along the sequence dimension to
        preserve correct dequantization of historical entries. Returns
        decompressed full key/value tensors for attention computation.
        """
        t0 = time.monotonic()
        # Quantize incoming key/value with per-token scales
        q_key, s_key = self._quantize(key)
        q_value, s_value = self._quantize(value)

        if self._keys[layer_idx] is not None:
            full_q_key = torch.cat([self._keys[layer_idx], q_key], dim=2)
            full_q_value = torch.cat([self._values[layer_idx], q_value], dim=2)
            # Concatenate per-token scales along sequence dimension
            full_s_key = torch.cat([self._scales_k[layer_idx], s_key], dim=2)
            full_s_value = torch.cat([self._scales_v[layer_idx], s_value], dim=2)
        else:
            full_q_key = q_key
            full_q_value = q_value
            full_s_key = s_key
            full_s_value = s_value

        self._keys[layer_idx] = full_q_key
        self._values[layer_idx] = full_q_value
        self._scales_k[layer_idx] = full_s_key
        self._scales_v[layer_idx] = full_s_value

        # Dequantize for return
        full_key = self._dequantize(full_q_key, full_s_key)
        full_value = self._dequantize(full_q_value, full_s_value)

        # Update metrics incrementally
        self._metrics.num_appends += 1
        self._metrics.total_entries += key.shape[2]
        compressed_new = (
            q_key.nelement() * q_key.element_size() + q_value.nelement() * q_value.element_size()
            + s_key.nelement() * s_key.element_size() + s_value.nelement() * s_value.element_size()
        )
        uncompressed_new = (
            key.nelement() * key.element_size() + value.nelement() * value.element_size()
        )
        self._metrics.compressed_memory_bytes += compressed_new
        self._metrics.memory_bytes += uncompressed_new
        self._metrics.compression_ratio = (
            self._metrics.memory_bytes / self._metrics.compressed_memory_bytes
            if self._metrics.compressed_memory_bytes > 0 else 1.0
        )
        self._metrics.last_append_latency_ms = (time.monotonic() - t0) * 1000
        return full_key, full_value

    def clear(self):
        """Clear all cached tensors and scale factors."""
        super().clear()
        for i in range(len(self._scales_k)):
            self._scales_k[i] = None
            self._scales_v[i] = None


def create_kv_cache(num_layers: int) -> KVCache:
    """Factory: create a KVCache or TurboQuantKVCache based on env flag.

    Set TURBO_QUANT_KV_CACHE=1 to enable compressed caching.
    """
    if TURBO_QUANT_ENABLED:
        log.info("TurboQuant KV cache enabled (TURBO_QUANT_KV_CACHE=1)")
        return TurboQuantKVCache(num_layers)
    return KVCache(num_layers)
