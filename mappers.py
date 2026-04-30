import logging
from typing import Any, Iterable, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
import neural_net_layers as nnl

log = logging.getLogger(__name__)


_GEMMA_MODEL_TYPES = frozenset({
    'gemma', 'gemma2', 'gemma3', 'gemma3_text', 'gemma4', 'gemma4_text',
})


class Mapper:
    _algo_to_func = {
        "embedding": nn.Embedding,
        "linear": nn.Linear,
        "flatten": nn.Flatten,
        "batchnorm1d": nn.BatchNorm1d,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
        "softmax": nn.Softmax,
        "tanh": nn.Tanh,
        "dropout": nn.Dropout,
        "sequential": nn.Sequential,
        "layernorm": nn.LayerNorm,
        "attention": nnl.CausalSelfAttention,
        "summation": nnl.Summation,
        "residual": nnl.ResidualConnection,
        "position": nnl.PositionEmbedding,
        "softmaxlast": nnl.SoftmaxOnLast,
        "rmsnorm": nnl.RMSNorm,
        "gatedmlp": nnl.GatedMLP,
        "scaledembedding": nnl.ScaledEmbedding,
        "transformerblock": nnl.TransformerBlock,
    }

    _init_weight_to_func = {
        "xavier_uniform": nn.init.xavier_uniform_,
        "kaiming_uniform": nn.init.kaiming_uniform_,
        "normal": nn.init.normal_,
    }

    _init_bias_to_func = {
        "zeros": nn.init.zeros_,
    }

    _optim_to_func = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
    }

    def __init__(self, layers: list[dict], optimizer: dict):
        self.layers = layers
        self.optimizer = optimizer

    @staticmethod
    def _unpack_func_and_args(k_to_args: dict, k_to_func: dict) -> Tuple[Any, dict | list]:
        return next(((k_to_func[k], v) for k, v in k_to_args.items() if k in k_to_func), (None, None))

    @staticmethod
    def _apply_confidence(nn_layer: nn.Module, confidence):
        with torch.no_grad():
            nn_layer.weight *= confidence

    @classmethod
    def _to_layer(cls, layer: dict) -> nn.Module:
        layer_func, layer_args = cls._unpack_func_and_args(layer, cls._algo_to_func)
        if isinstance(layer_args, dict):
            layer_args |= {arg: cls._to_layer(v) for arg, v in layer_args.items()
                           if isinstance(v, dict)}
        elif isinstance(layer_args, list):
            layer_args = [cls._to_layer(arg) if isinstance(arg, dict) else arg
                          for arg in layer_args]

        if layer_func:
            nn_layer: nn.Module = layer_func(**layer_args) if isinstance(layer_args, dict) else layer_func(*layer_args)

            init_w_func, init_w_args = cls._unpack_func_and_args(layer, cls._init_weight_to_func)
            if init_w_func:
                nn_layer.apply(lambda l: init_w_func(l.weight, **init_w_args) if hasattr(l, 'weight') else None)

            init_b_func, init_b_args = cls._unpack_func_and_args(layer, cls._init_bias_to_func)
            if init_b_func:
                nn_layer.apply(lambda l: init_b_func(l.bias, **init_b_args) if hasattr(l, 'bias') else None)

            confidence: float = layer.get("confidence")
            if confidence is not None:
                nn_layer.apply(lambda l: cls._apply_confidence(l, confidence))

            return nn_layer
        else:
            raise ValueError(f"Unsupported layer: {layer}")


    @classmethod
    def from_hf_config(cls, hf_config, n_layer_override: int = None) -> list[dict]:
        """Build internal layers config list from a HuggingFace model config.

        Supports GPT-2 family and Gemma family configs.

        :param hf_config: A HuggingFace ``PretrainedConfig`` instance.
        :param n_layer_override: Optional override for the number of transformer layers.
        :return: Layer config list compatible with ``Mapper.__init__`` ``layers`` argument.
        """
        model_type = getattr(hf_config, "model_type", None)
        if isinstance(model_type, str) and model_type in _GEMMA_MODEL_TYPES:
            layers = cls._build_gemma_layers(hf_config, n_layer_override)
        else:
            layers = cls._build_gpt2_layers(hf_config, n_layer_override)
        log.info("Following %d layers have been built from HuggingFace config of model %s: %s",
                 len(layers), model_type, layers)
        return layers

    @classmethod
    def _build_gpt2_layers(cls, hf_config, n_layer_override: int = None) -> list[dict]:
        vocab_size = hf_config.vocab_size
        n_embd = getattr(hf_config, "n_embd", None)
        if n_embd is None:
            n_embd = getattr(hf_config, "hidden_size", None)
        n_head = getattr(hf_config, "n_head", None)
        if n_head is None:
            n_head = getattr(hf_config, "num_attention_heads", None)
        n_layer = n_layer_override
        if n_layer is None:
            n_layer = getattr(hf_config, "n_layer", None)
        if n_layer is None:
            n_layer = getattr(hf_config, "num_hidden_layers", None)
        block_size = getattr(hf_config, "n_positions", None)
        if block_size is None:
            block_size = getattr(hf_config, "max_position_embeddings", None)
        activation = getattr(hf_config, "activation_function", "gelu_new")
        gelu_layer = {"gelu": {"approximate": "tanh"}} if activation == "gelu_new" else {"gelu": {}}
        dropout = getattr(hf_config, "resid_pdrop", 0.0)
        embd_dropout = getattr(hf_config, "embd_pdrop", 0.0)
        attn_dropout = getattr(hf_config, "attn_pdrop", 0.0)

        layers = [
            {"summation": [
                {"embedding": {"num_embeddings": vocab_size, "embedding_dim": n_embd}},
                {"position": {"num_embeddings": block_size, "embedding_dim": n_embd}},
            ]},
            {"dropout": {"p": embd_dropout}},
        ]

        for _ in range(n_layer):
            layers.append({"residual": [
                {"sequential": [
                    {"layernorm": {"normalized_shape": n_embd}},
                    {"linear": {"in_features": n_embd, "out_features": 3 * n_embd}},
                    {"attention": {"num_heads": n_head, "dropout": attn_dropout}},
                    {"linear": {"in_features": n_embd, "out_features": n_embd}},
                    {"dropout": {"p": dropout}},
                ]},
                {"sequential": [
                    {"layernorm": {"normalized_shape": n_embd}},
                    {"linear": {"in_features": n_embd, "out_features": 4 * n_embd}},
                    gelu_layer,
                    {"linear": {"in_features": 4 * n_embd, "out_features": n_embd}},
                    {"dropout": {"p": dropout}},
                ]},
            ]})

        layers.extend([
            {"layernorm": {"normalized_shape": n_embd}},
            {"linear": {"in_features": n_embd, "out_features": vocab_size, "bias": False}},
            {"softmaxlast": {"dim": -1}},
        ])

        return layers

    @classmethod
    def _build_gemma_layers(cls, hf_config, n_layer_override: int = None) -> list[dict]:
        model_type = hf_config.model_type
        # Multimodal configs (gemma3, gemma4) nest text params in text_config
        text_config = getattr(hf_config, "text_config", hf_config)
        vocab_size = text_config.vocab_size
        n_embd = text_config.hidden_size
        n_head = text_config.num_attention_heads
        n_kv_heads = getattr(text_config, "num_key_value_heads", n_head)
        head_dim = getattr(text_config, "head_dim", n_embd // n_head)
        n_layer = n_layer_override if n_layer_override is not None else text_config.num_hidden_layers
        intermediate_size = getattr(text_config, "intermediate_size", 4 * n_embd)
        rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-6)
        # Per-attention-type RoPE base: Gemma 3n/4 heterogeneous models use
        # different rope_theta for sliding vs full attention layers (e.g.
        # 10_000 local / 1_000_000 global).  Prefer dedicated config attrs
        # (`rope_theta` = full, `rope_local_base_freq` = sliding); fall back
        # to a `rope_scaling` dict keyed by attention type; finally default
        # to 10_000 when nothing is configured.
        rope_theta_full = getattr(text_config, "rope_theta", None)
        rope_theta_local = getattr(text_config, "rope_local_base_freq", None)
        rope_scaling = getattr(text_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            if rope_theta_full is None and "full_attention" in rope_scaling:
                rope_theta_full = rope_scaling["full_attention"].get("rope_theta")
            if rope_theta_local is None and "sliding_attention" in rope_scaling:
                rope_theta_local = rope_scaling["sliding_attention"].get("rope_theta")
        if rope_theta_full is None and rope_theta_local is None:
            rope_theta_full = rope_theta_local = 10000.0
        elif rope_theta_full is None:
            rope_theta_full = rope_theta_local
        elif rope_theta_local is None:
            rope_theta_local = rope_theta_full
        attn_dropout = getattr(text_config, "attention_dropout", 0.0)
        activation = (getattr(text_config, "hidden_activation", None)
                      or getattr(text_config, "hidden_act", "gelu_pytorch_tanh"))
        has_post_norms = model_type != "gemma"
        has_qk_norm = model_type != "gemma"
        # Gemma 2 applies post-norm to branch output before residual add;
        # Gemma 3+ applies post-norm after residual add.
        post_norm_on_residual = model_type != "gemma2"

        # Per-layer heterogeneous architecture (Gemma 4)
        layer_types = getattr(text_config, "layer_types", None)
        global_head_dim = getattr(text_config, "global_head_dim", head_dim)
        n_global_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or n_kv_heads
        use_double_wide_mlp = getattr(text_config, "use_double_wide_mlp", False)
        num_kv_shared = getattr(text_config, "num_kv_shared_layers", 0) or 0
        first_kv_shared = n_layer - num_kv_shared if num_kv_shared > 0 else n_layer
        kv_ref_layer = {}
        if num_kv_shared > 0 and layer_types and len(layer_types) >= n_layer:
            prev_layers = list(layer_types[:first_kv_shared])
            for i in range(first_kv_shared, n_layer):
                lt = layer_types[i]
                try:
                    ref = len(prev_layers) - 1 - prev_layers[::-1].index(lt)
                    kv_ref_layer[i] = ref
                except ValueError:
                    pass

        layers: list[dict] = [
            {"scaledembedding": {
                "num_embeddings": vocab_size,
                "embedding_dim": n_embd,
                "scale": float(n_embd ** 0.5),
            }},
        ]

        for i in range(n_layer):
            # Determine per-layer attention dimensions
            is_full_attn = (layer_types and i < len(layer_types)
                            and layer_types[i] == "full_attention")
            layer_head_dim = global_head_dim if is_full_attn else head_dim
            layer_kv_heads = n_global_kv_heads if is_full_attn else n_kv_heads
            layer_rope_theta = rope_theta_full if is_full_attn else rope_theta_local
            qkv_dim = n_head * layer_head_dim + 2 * layer_kv_heads * layer_head_dim
            attn_out_dim = n_head * layer_head_dim

            # Determine per-layer MLP dimensions
            is_kv_shared = i >= first_kv_shared
            layer_intermediate = intermediate_size * 2 if (use_double_wide_mlp and is_kv_shared) else intermediate_size

            attn_args = {"num_heads": n_head, "num_kv_heads": layer_kv_heads,
                        "dropout": attn_dropout, "rope_theta": layer_rope_theta,
                        "head_dim": layer_head_dim}
            if has_qk_norm:
                attn_args["q_norm"] = {"rmsnorm": {"normalized_shape": layer_head_dim, "eps": rms_norm_eps}}
                attn_args["k_norm"] = {"rmsnorm": {"normalized_shape": layer_head_dim, "eps": rms_norm_eps}}
            if i in kv_ref_layer:
                attn_args["kv_shared_layer_idx"] = kv_ref_layer[i]

            block: dict = {
                "attn_block": {"sequential": [
                    {"rmsnorm": {"normalized_shape": n_embd, "eps": rms_norm_eps}},
                    {"linear": {"in_features": n_embd, "out_features": qkv_dim, "bias": False}},
                    {"attention": attn_args},
                    {"linear": {"in_features": attn_out_dim, "out_features": n_embd, "bias": False}},
                ]},
                "mlp_block": {"sequential": [
                    {"rmsnorm": {"normalized_shape": n_embd, "eps": rms_norm_eps}},
                    {"gatedmlp": {"in_features": n_embd, "intermediate_size": layer_intermediate,
                                  "bias": False, "activation": activation}},
                ]},
            }
            if has_post_norms:
                block["post_attn_norm"] = {"rmsnorm": {"normalized_shape": n_embd, "eps": rms_norm_eps}}
                block["post_mlp_norm"] = {"rmsnorm": {"normalized_shape": n_embd, "eps": rms_norm_eps}}
                block["post_norm_on_residual"] = post_norm_on_residual
            layers.append({"transformerblock": block})

        layers.extend([
            {"rmsnorm": {"normalized_shape": n_embd, "eps": rms_norm_eps}},
            {"linear": {"in_features": n_embd, "out_features": vocab_size, "bias": False}},
            {"softmaxlast": {"dim": -1}},
        ])

        return layers

    def to_layers(self) -> list[nn.Module]:
        return [self._to_layer(l) for l in self.layers]

    def to_optimizer(self, params: Iterable[Tensor]) -> Optimizer:
        optim_func, optim_args = self._unpack_func_and_args(self.optimizer, self._optim_to_func)
        if optim_func:
            if "betas" in optim_args:
                optim_args |= {"betas": tuple(optim_args["betas"])}
            return optim_func(params, **optim_args)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    @staticmethod
    def detect_hf_n_layer(hf_sd: dict) -> int:
        """Detect the number of transformer layers in a HuggingFace state dict.

        Supports Gemma (``model.layers.{i}`` or ``model.language_model.layers.{i}``)
        and GPT-2 (``transformer.h.{i}``) key patterns.

        :param hf_sd: State dict from a HuggingFace model.
        :return: Number of transformer layers found, or 0 if unrecognised.
        """
        # Gemma patterns
        pfx = "model.language_model" if "model.language_model.embed_tokens.weight" in hf_sd else "model"
        layer_idx_pos = 3 if pfx == "model.language_model" else 2
        layer_indices = [
            int(k.split(".")[layer_idx_pos]) for k in hf_sd
            if k.startswith(f"{pfx}.layers.") and k.endswith(".self_attn.q_proj.weight")
        ]
        if layer_indices:
            return max(layer_indices) + 1
        # GPT-2 pattern
        gpt2_indices = [
            int(k.split(".")[2]) for k in hf_sd
            if k.startswith("transformer.h.") and k.endswith(".attn.c_attn.weight")
        ]
        if gpt2_indices:
            return max(gpt2_indices) + 1
        return 0

    @classmethod
    def map_hf_state_dict_to_custom(cls, hf_sd: dict, n_layer: int, hf_config=None) -> dict:
        """Map a HuggingFace state dict to the internal custom key names.

        :param hf_sd: State dict from a HuggingFace model.
        :param n_layer: Number of transformer blocks.
        :param hf_config: Optional HuggingFace config for model-type detection.
        :return: State dict with keys matching the internal ``NeuralNetworkModel`` naming.
        """
        model_type = getattr(hf_config, "model_type", None) if hf_config is not None else None
        if isinstance(model_type, str) and model_type in _GEMMA_MODEL_TYPES:
            return cls._map_gemma_state_dict(hf_sd, n_layer, hf_config)
        return cls._map_gpt2_state_dict(hf_sd, n_layer)

    @classmethod
    def _map_gpt2_state_dict(cls, hf_sd: dict, n_layer: int) -> dict:
        mapped = {}

        # Token and position embeddings (layer 0 is a Summation of the two)
        mapped["layers.0.0.weight"] = hf_sd["transformer.wte.weight"]
        mapped["layers.0.1.weight"] = hf_sd["transformer.wpe.weight"]

        for i in range(n_layer):
            block_idx = 2 + i  # layers 0=emb summation, 1=dropout, 2..=residual blocks
            hf_prefix = f"transformer.h.{i}"

            # Attention sub-block (index 0 inside the ResidualConnection Sequential)
            mapped[f"layers.{block_idx}.0.0.weight"] = hf_sd[f"{hf_prefix}.ln_1.weight"]
            mapped[f"layers.{block_idx}.0.0.bias"]   = hf_sd[f"{hf_prefix}.ln_1.bias"]
            mapped[f"layers.{block_idx}.0.1.weight"] = hf_sd[f"{hf_prefix}.attn.c_attn.weight"].t().contiguous()
            mapped[f"layers.{block_idx}.0.1.bias"]   = hf_sd[f"{hf_prefix}.attn.c_attn.bias"]
            mapped[f"layers.{block_idx}.0.3.weight"] = hf_sd[f"{hf_prefix}.attn.c_proj.weight"].t().contiguous()
            mapped[f"layers.{block_idx}.0.3.bias"]   = hf_sd[f"{hf_prefix}.attn.c_proj.bias"]

            # MLP sub-block (index 1 inside the ResidualConnection Sequential)
            mapped[f"layers.{block_idx}.1.0.weight"] = hf_sd[f"{hf_prefix}.ln_2.weight"]
            mapped[f"layers.{block_idx}.1.0.bias"]   = hf_sd[f"{hf_prefix}.ln_2.bias"]
            mapped[f"layers.{block_idx}.1.1.weight"] = hf_sd[f"{hf_prefix}.mlp.c_fc.weight"].t().contiguous()
            mapped[f"layers.{block_idx}.1.1.bias"]   = hf_sd[f"{hf_prefix}.mlp.c_fc.bias"]
            mapped[f"layers.{block_idx}.1.3.weight"] = hf_sd[f"{hf_prefix}.mlp.c_proj.weight"].t().contiguous()
            mapped[f"layers.{block_idx}.1.3.bias"]   = hf_sd[f"{hf_prefix}.mlp.c_proj.bias"]

        # Final layer norm
        ln_f_idx = 2 + n_layer
        mapped[f"layers.{ln_f_idx}.weight"] = hf_sd["transformer.ln_f.weight"]
        mapped[f"layers.{ln_f_idx}.bias"]   = hf_sd["transformer.ln_f.bias"]

        # LM head – use explicit lm_head.weight when available, else fall back to tied wte weight
        lm_head_weight = hf_sd.get("lm_head.weight", hf_sd["transformer.wte.weight"])
        mapped[f"layers.{ln_f_idx + 1}.weight"] = lm_head_weight

        return mapped

    @classmethod
    def _map_gemma_state_dict(cls, hf_sd: dict, n_layer: int, hf_config) -> dict:
        mapped = {}
        model_type = hf_config.model_type
        has_post_norms = model_type != "gemma"

        # Multimodal models (gemma3, gemma4) prefix text keys with "model.language_model."
        pfx = "model.language_model" if "model.language_model.embed_tokens.weight" in hf_sd else "model"

        # Detect actual text layer count from state dict for robustness
        actual_n_layer = cls.detect_hf_n_layer(hf_sd)
        if actual_n_layer != n_layer:
            log.warning("HF state dict has %d text layers but config says %d; using detected count",
                        actual_n_layer, n_layer)
            n_layer = actual_n_layer

        # Build KV-shared layer reference mapping (Gemma 4 with shared KV layers).
        # Shared layers reuse K/V projections from an earlier non-shared layer of
        # the same attention type, so their k_proj/v_proj may be absent from the
        # checkpoint.  We copy weights from the referenced layer when missing.
        text_config = getattr(hf_config, "text_config", hf_config)
        num_kv_shared = getattr(text_config, "num_kv_shared_layers", 0) or 0
        kv_ref_layer = {}
        if num_kv_shared > 0:
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types and len(layer_types) >= n_layer:
                first_kv_shared = n_layer - num_kv_shared
                prev_layers = list(layer_types[:first_kv_shared])
                for i in range(first_kv_shared, n_layer):
                    lt = layer_types[i]
                    try:
                        ref = len(prev_layers) - 1 - prev_layers[::-1].index(lt)
                        kv_ref_layer[i] = ref
                    except ValueError:
                        pass  # no matching non-shared layer found; use own weights

        # Token embedding (layer 0 is ScaledEmbedding)
        mapped["layers.0.weight"] = hf_sd[f"{pfx}.embed_tokens.weight"]

        for i in range(n_layer):
            block_idx = 1 + i  # 0=embedding, 1+=transformer blocks
            hf = f"{pfx}.layers.{i}"

            # Attention pre-norm (Gemma uses 1+weight centered RMSNorm; convert to standard)
            mapped[f"layers.{block_idx}.attn_block.0.weight"] = hf_sd[f"{hf}.input_layernorm.weight"] + 1

            # QKV projection (concatenate separate Q, K, V into single tensor)
            q = hf_sd[f"{hf}.self_attn.q_proj.weight"]
            ref = kv_ref_layer.get(i)
            if ref is not None:
                # KV-shared layer: always use the reference (non-shared) layer's
                # K/V weights.  The checkpoint may omit them entirely, or
                # from_pretrained may have filled them with random init values;
                # either way, the reference layer holds the correct weights.
                ref_hf = f"{pfx}.layers.{ref}"
                k = hf_sd[f"{ref_hf}.self_attn.k_proj.weight"]
                v = hf_sd[f"{ref_hf}.self_attn.v_proj.weight"]
                log.debug("Layer %d: copied K/V from reference layer %d", i, ref)
            else:
                k = hf_sd[f"{hf}.self_attn.k_proj.weight"]
                v = hf_sd[f"{hf}.self_attn.v_proj.weight"]
            mapped[f"layers.{block_idx}.attn_block.1.weight"] = torch.cat([q, k, v], dim=0)

            # Output projection
            mapped[f"layers.{block_idx}.attn_block.3.weight"] = hf_sd[f"{hf}.self_attn.o_proj.weight"]

            # Q/K norms (Gemma 2+)
            if has_post_norms:
                mapped[f"layers.{block_idx}.attn_block.2.q_norm.weight"] = hf_sd[f"{hf}.self_attn.q_norm.weight"] + 1
                mapped[f"layers.{block_idx}.attn_block.2.k_norm.weight"] = hf_sd[f"{hf}.self_attn.k_norm.weight"] + 1

            if has_post_norms:
                mapped[f"layers.{block_idx}.post_attn_norm.weight"] = (
                    hf_sd[f"{hf}.post_attention_layernorm.weight"] + 1)
                mapped[f"layers.{block_idx}.mlp_block.0.weight"] = (
                    hf_sd[f"{hf}.pre_feedforward_layernorm.weight"] + 1)
                mapped[f"layers.{block_idx}.post_mlp_norm.weight"] = (
                    hf_sd[f"{hf}.post_feedforward_layernorm.weight"] + 1)
            else:
                # Gemma 1: post_attention_layernorm acts as the pre-MLP norm
                mapped[f"layers.{block_idx}.mlp_block.0.weight"] = (
                    hf_sd[f"{hf}.post_attention_layernorm.weight"] + 1)

            # Gated MLP
            mapped[f"layers.{block_idx}.mlp_block.1.gate_proj.weight"] = hf_sd[f"{hf}.mlp.gate_proj.weight"]
            mapped[f"layers.{block_idx}.mlp_block.1.up_proj.weight"] = hf_sd[f"{hf}.mlp.up_proj.weight"]
            mapped[f"layers.{block_idx}.mlp_block.1.down_proj.weight"] = hf_sd[f"{hf}.mlp.down_proj.weight"]

        # Final RMSNorm
        ln_f_idx = 1 + n_layer
        mapped[f"layers.{ln_f_idx}.weight"] = hf_sd[f"{pfx}.norm.weight"] + 1

        # LM head – use explicit lm_head.weight when available, else tied embedding
        lm_head_weight = hf_sd.get("lm_head.weight", hf_sd[f"{pfx}.embed_tokens.weight"])
        mapped[f"layers.{ln_f_idx + 1}.weight"] = lm_head_weight

        return mapped
