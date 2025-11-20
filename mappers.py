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
