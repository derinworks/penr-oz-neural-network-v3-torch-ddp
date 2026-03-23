from typing import Any, Iterable, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
import neural_net_layers as nnl


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
    def from_hf_config(cls, hf_config) -> list[dict]:
        """Build internal layers config list from a HuggingFace model config.

        Supports GPT-2 family configs (and any config using the same attribute names).

        :param hf_config: A HuggingFace ``PretrainedConfig`` instance.
        :return: Layer config list compatible with ``Mapper.__init__`` ``layers`` argument.
        """
        vocab_size = hf_config.vocab_size
        n_embd = getattr(hf_config, "n_embd", None)
        if n_embd is None:
            n_embd = getattr(hf_config, "hidden_size", None)
        n_head = getattr(hf_config, "n_head", None)
        if n_head is None:
            n_head = getattr(hf_config, "num_attention_heads", None)
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

    @classmethod
    def map_hf_state_dict_to_custom(cls, hf_sd: dict, n_layer: int) -> dict:
        """Map a GPT-2 HuggingFace state dict to the internal custom key names.

        :param hf_sd: State dict from a HuggingFace GPT-2 model.
        :param n_layer: Number of transformer blocks.
        :return: State dict with keys matching the internal ``NeuralNetworkModel`` naming.
        """
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
