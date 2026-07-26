import json
import math
import os
import os.path
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock, call
from parameterized import parameterized
import numpy as np
import torch
import torch.nn as nn
from neural_net_model import NeuralNetworkModel
from mappers import Mapper
import neural_net_layers as nnl


class TestNeuralNetModel(unittest.TestCase):

    @parameterized.expand([
        ([{"linear": {"in_features": 9, "out_features": 9}, "xavier_uniform": {}}, {"relu": {}}],
         {"adam": {"lr": 0.1}},
         [nn.Linear,nn.ReLU], [(9,9),(9,)], 90),
        ([{"linear": {"in_features": 18, "out_features": 9}, "xavier_uniform": {}}, {"softmax": {"dim": -1}}],
         {"adamw": {"lr": 0.1}},
         [nn.Linear,nn.Softmax], [(9,18),(9,)], 171),
        ([{"linear": {"in_features": 9, "out_features": 18, "bias": False}, "kaiming_uniform": {}}, {"sigmoid": {}}], 
         {"sgd": {"lr": 0.1}},
         [nn.Linear,nn.Sigmoid], [(18,9)], 162),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"tanh": {}}], {"sgd": {"lr": 0.1}},
         [nn.Linear,nn.Tanh] * 2, [(8,4),(8,), (16,8),(16,)], 184),
        ([{"linear": {"in_features": 3, "out_features": 3, "bias": False}}, {"relu": {}},
          {"linear": {"in_features": 3, "out_features": 3}}, {"tanh": {}},
          {"linear": {"in_features": 3, "out_features": 3, "bias": False}, "xavier_uniform": {}}, {"softmax": {"dim": -1}}
          ], {"sgd": {"lr": 0.1}},
         [nn.Linear,nn.ReLU, nn.Linear,nn.Tanh, nn.Linear,nn.Softmax], [(3,3), (3,3),(3,), (3,3)], 30),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 20}}, {"tanh": {}},
          {"linear": {"in_features": 20, "out_features": 18, "bias": False}}, {"softmax": {"dim": -1}},
          ], {"sgd": {"lr": 0.1}},
         [nn.Embedding,nn.Flatten, nn.Linear,nn.Tanh, nn.Linear,nn.Softmax], [(18,2),(20,6),(20,), (18,20)], 536),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 20}}, {"batchnorm1d": {"num_features": 20}}, {"tanh": {}},
          {"linear": {"in_features": 20, "out_features": 18, "bias": False}, "confidence": 0.1}, {"softmax": {"dim": -1}},
          ], {"sgd": {"lr": 0.1}},
         [nn.Embedding,nn.Flatten, nn.Linear,nn.BatchNorm1d,nn.Tanh, nn.Linear,nn.Softmax],
         [(18,2),(20,6),(20,),(20,),(20,),(18,20)], 576),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 10}}, {"tanh": {}},
          {"linear": {"in_features": 10, "out_features": 18}}, {"dropout": {"p": 0.1}},{"softmax": {"dim": -1}},
         ], {"sgd": {"lr": 0.1}},
         [nn.Embedding,nn.Flatten, nn.Linear,nn.Tanh, nn.Linear,nn.Dropout,nn.Softmax],
         [(18,2),(10,6),(10,),(18,10),(18,)], 304),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]}], {"adam": {"lr": 3e-4}},
         [nnl.Summation],
         [(27, 4), (8, 4)], 140),
        ([{"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                          {"linear": {"in_features": 4, "out_features": 12},
                           "normal": {"std": 0.2}, "zeros": {}},
                          {"attention": {"num_heads": 2}},
                          {"linear": {"in_features": 4, "out_features": 4},
                           "normal": {"std": 0.2}, "zeros": {}},
                          ]}
          ], {"adamw": {"lr": 3e-4}},
         [nn.Sequential],
         [(4,), (12, 4), (12,), (4, 4), (4,)], 84),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]}, {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]}]}
             for _ in range(2)] +
        [{"layernorm": {"normalized_shape": 4, "bias": False}},
         {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
         {"softmaxlast": {"dim": -1}}], {"adamw": {"lr": 3e-4}},
         [nnl.Summation,nn.Dropout] + [nnl.ResidualConnection] * 2 + [nn.LayerNorm,nn.Linear,nnl.SoftmaxOnLast],
         [(27, 4), (8, 4)] + [(4,), (12, 4), (4, 4), (4,), (16, 4), (4, 16)] * 2 + [(4,), (27, 4)], 652),
    ])
    def test_model_init(self, layers: list[dict], optimizer: dict,
                        expected_layers: list[nn.Module], expected_shapes: list[list[tuple]], expected_num_params: int):

        model = NeuralNetworkModel("test", Mapper(layers, optimizer))

        self.assertEqual("test", model.model_id)
        self.assertListEqual(expected_layers, [l.__class__ for l in model.layers])
        self.assertListEqual(expected_shapes, [tuple(p.shape) for p in model.parameters()])
        self.assertTrue(model.optimizer.__class__.__name__.lower() in optimizer.keys())
        self.assertEqual(0, len(model.progress))
        self.assertEqual(expected_num_params, model.num_params)
        self.assertIsNone(model.avg_cost)
        self.assertEqual(0, len(model.avg_cost_history))
        self.assertIsNone(model.stats)
        self.assertEqual("Created", model.status.get("code"))

    @parameterized.expand([
        ([{"linear": {"in_features": 9, "out_features": 9}}, {"sigmoid": {}}] * 2, [0.5] * 9, None),
        ([{"linear": {"in_features": 9, "out_features": 9}}, {"softmax": {"dim": 0}}], [1.0] + [0.0] * 8, 4),
        ([{"linear": {"in_features": 18, "out_features": 9}}, {"relu": {}},
          {"linear": {"in_features": 9, "out_features": 3}}, {"softmax": {"dim": 0}}], [1.0] + [0.0] * 17, None),
        ([{"linear": {"in_features": 9, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"tanh": {}}] * 2, [0.5] * 9, [0.5] * 9),
        ([{"linear": {"in_features": 9, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"tanh": {}}] * 2, [[0.5] * 9] * 2, [[0.5] * 9] * 2),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"softmax": {"dim": 0}}], [0.5] * 4, 13),
        ([{"linear": {"in_features": 4, "out_features": 8}}, {"tanh": {}},
          {"linear": {"in_features": 8, "out_features": 16}}, {"softmax": {"dim": 1}}], [[0.5] * 4] * 2, [13] * 2),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}], [[0, 5, 8],[1, 3, 7]], [2, 4]),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]}, {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]}]}
             for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}], [[1,12,21,5,8,10,5,17]] * 5, [[12,21,5,8,10,5,17,21]] * 5),
    ])
    def test_compute_output(self, layers: list[dict], input_data: list, target: list | int | None):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        output, cost = model.compute_output(input_data, target)
        in_shape = np.shape(input_data)
        out_shape = np.shape(output)

        self.assertEqual(len(in_shape), len(out_shape))
        if len(out_shape) > 1: # same batch size?
            self.assertEqual(in_shape[0], out_shape[0])
        self.assertTrue(target is None or cost is not None)
        self.assertFalse(model.layers.training)

    @parameterized.expand([
        ([{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
          {"tanh": {}},
          {"linear": {"in_features": 2, "out_features": 8}},
          {"softmaxlast": {"dim": -1}}],
         [1, 2], [2, 3], 2, 1, 1),
        ([{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
          {"gelu": {}},
          {"linear": {"in_features": 2, "out_features": 8}},
          {"softmaxlast": {"dim": -1}}],
         [1, 2], [2, 3], 2, 1, 2),
        ([{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
          {"linear": {"in_features": 2, "out_features": 4 * 2}},
          {"gelu": {}},
          {"linear": {"in_features": 4 * 2, "out_features": 2}},
          {"linear": {"in_features": 2, "out_features": 8}},
          {"softmaxlast": {"dim": -1}}],
         [1, 2, 3, 4], [2, 3, 4, 5], 4, 2, 1),
        ([{"embedding": {"num_embeddings": 16, "embedding_dim": 2}},
          {"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 4 * 2}},
          {"gelu": {}},
          {"linear": {"in_features": 4 * 2, "out_features": 2}},
          {"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 16, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [1, 2, 3, 4], [2, 3, 4, 5], 4, 2, 2),
        ([{"embedding": {"num_embeddings": 16, "embedding_dim": 2}},
          {"dropout": {"p": 0.0}},
          {"sequential": [{"layernorm": {"normalized_shape": 2}},
                          {"linear": {"in_features": 2, "out_features": 4 * 2}},
                          {"gelu": {}},
                          {"linear": {"in_features": 4 * 2, "out_features": 2}},
                          {"dropout": {"p": 0.0}}]},
          {"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 16, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [1, 2, 3, 4], [2, 3, 4, 5], 2, 2, 1),
        ([{"summation": [{"embedding": {"num_embeddings": 16, "embedding_dim": 2}},
                         {"position": {"num_embeddings": 4, "embedding_dim": 2}}]},
          {"dropout": {"p": 0.0}}] +
         [{"residual": [{"sequential": [{"layernorm": {"normalized_shape": 2}},
                                        {"linear": {"in_features": 2, "out_features": 3 * 2}},
                                        {"attention": {"num_heads": 1, "dropout": 0.0}},
                                        {"linear": {"in_features": 2, "out_features": 2}},
                                        {"dropout": {"p": 0.0}}]},
                        {"sequential": [{"layernorm": {"normalized_shape": 2}},
                                        {"linear": {"in_features": 2, "out_features": 4 * 2}},
                                        {"gelu": {}},
                                        {"linear": {"in_features": 4 * 2, "out_features": 2}},
                                        {"dropout": {"p": 0.0}}]}
                        ]} for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 16, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [1,2,3,4,5,6,7,8], [2,3,4,5,6,7,8,9], 3, 4, 2),
    ])
    def test_evaluate(self, layers: list[dict], input_data: list, target: list,
                      epochs: int, batch_size: int, step_size: int):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        block_size = len(input_data) // batch_size
        with patch("neural_net_model.Loader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.return_value = tuple(np.array(l, dtype=np.int32) for l in [input_data, target])
            cost = model.evaluate_model("mock_ds", None, 0,
                                        epochs, batch_size, block_size, step_size)

        self.assertIsNotNone(cost)
        self.assertFalse(model.layers.training)

    @parameterized.expand([
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}],
         [[0, 5, 8]], 3, 3),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]},
          {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}]}]}
             for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [[0]], 8, 10),
    ])
    def test_generate_tokens(self, layers: list[dict], input_context: list, block_size: int, max_new_tokens: int):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        tokens = model.generate_tokens(input_context, block_size, max_new_tokens)

        self.assertIsNotNone(tokens)
        self.assertGreaterEqual(len(tokens), block_size)
        self.assertLessEqual(len(tokens), len(input_context[0]) + max_new_tokens)
        self.assertFalse(model.layers.training)

    def test_infer_block_size_from_position_embedding(self):
        layers = [{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                                 {"position": {"num_embeddings": 8, "embedding_dim": 4}}]},
                  {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        self.assertEqual(8, model.infer_block_size())

    def test_infer_block_size_without_position_embedding_raises(self):
        layers = [{"linear": {"in_features": 4, "out_features": 4}}, {"softmax": {"dim": -1}}]
        model = NeuralNetworkModel("test-no-pos-emb", Mapper(layers, {"sgd": {}}))

        with self.assertRaises(ValueError) as ctx:
            model.infer_block_size()

        self.assertIn("Cannot infer block_size", str(ctx.exception))
        self.assertIn("test-no-pos-emb", str(ctx.exception))

    @parameterized.expand([
        ("n_positions", {"n_positions": 1024}, 1024),
        ("max_position_embeddings", {"max_position_embeddings": 2048}, 2048),
        ("nested_text_config", {"text_config": {"max_position_embeddings": 4096}}, 4096),
        ("text_config_precedes_top_level", {"text_config": {"max_position_embeddings": 512},
                                            "max_position_embeddings": 4096}, 512),
    ])
    def test_infer_block_size_from_hf_config(self, _name, hf_config: dict, expected_block_size: int):
        model_id = f"test-hf-cfg-{_name}"
        layers = [{"linear": {"in_features": 4, "out_features": 4}}, {"softmax": {"dim": -1}}]
        model = NeuralNetworkModel(model_id, Mapper(layers, {"sgd": {}}))
        os.makedirs("models", exist_ok=True)
        hf_config_path = os.path.join("models", f"model_{model_id}_hf_config.json")
        with open(hf_config_path, "w") as f:
            json.dump(hf_config, f)
        try:
            self.assertEqual(expected_block_size, model.infer_block_size())
        finally:
            os.remove(hf_config_path)

    def test_infer_block_size_hf_config_without_context_length_raises(self):
        model_id = "test-hf-cfg-empty"
        layers = [{"linear": {"in_features": 4, "out_features": 4}}, {"softmax": {"dim": -1}}]
        model = NeuralNetworkModel(model_id, Mapper(layers, {"sgd": {}}))
        os.makedirs("models", exist_ok=True)
        hf_config_path = os.path.join("models", f"model_{model_id}_hf_config.json")
        with open(hf_config_path, "w") as f:
            json.dump({"model_type": "gemma3", "text_config": {"hidden_size": 32}}, f)
        try:
            with self.assertRaises(ValueError):
                model.infer_block_size()
        finally:
            os.remove(hf_config_path)

    @parameterized.expand([
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}],
         [[0, 5, 8]], 3, 3),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]},
          {"dropout": {"p": 0.2}}] +
         [{"residual": [
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 12, "bias": False}},
                             {"attention": {"num_heads": 2, "dropout": 0.2}},
                             {"linear": {"in_features": 4, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}
                             ]},
             {"sequential": [{"layernorm": {"normalized_shape": 4, "bias": False}},
                             {"linear": {"in_features": 4, "out_features": 16, "bias": False}},
                             {"gelu": {}},
                             {"linear": {"in_features": 16, "out_features": 4, "bias": False}},
                             {"dropout": {"p": 0.2}}]}]}
             for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [[0]], 8, 10),
    ])
    def test_generate_tokens_stream(self, layers: list[dict], input_context: list,
                                    block_size: int, max_new_tokens: int):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        torch.manual_seed(42)
        streamed_tokens = list(model.generate_tokens_stream(input_context, block_size, max_new_tokens))
        torch.manual_seed(42)
        non_streamed_tokens = model.generate_tokens(input_context, block_size, max_new_tokens)

        self.assertEqual(len(streamed_tokens), max_new_tokens)
        # Verify that streamed and non-streamed outputs are consistent
        self.assertEqual(non_streamed_tokens, input_context[0] + streamed_tokens)
        # All tokens should be integers
        for token in streamed_tokens:
            self.assertIsInstance(token, int)
        self.assertFalse(model.layers.training)

    @parameterized.expand([
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}],
         [[0, 5, 8]], 3, 10, False),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]},
          {"dropout": {"p": 0.2}}] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [[0]], 8, 10, False),
        ([{"embedding": {"num_embeddings": 18, "embedding_dim": 2}}, {"flatten": {}},
          {"linear": {"in_features": 6, "out_features": 18}}, {"tanh": {}},
          {"linear": {"in_features": 18, "out_features": 9}}, {"softmax": {"dim": 1}}],
         [[0, 5, 8]], 3, 10, True),
        ([{"summation": [{"embedding": {"num_embeddings": 27, "embedding_dim": 4}},
                         {"position": {"num_embeddings": 8, "embedding_dim": 4}}]},
          {"dropout": {"p": 0.2}}] +
         [{"layernorm": {"normalized_shape": 4, "bias": False}},
          {"linear": {"in_features": 4, "out_features": 27, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         [[0]], 8, 10, True),
    ])
    def test_generate_tokens_with_stop_token_halts_early(self, layers: list[dict], input_context: list,
                                                         block_size: int, max_new_tokens: int, stream: bool):
        model = NeuralNetworkModel("test", Mapper(layers, {"sgd": {}}))

        # generate without stop_token to discover the first generated token
        torch.manual_seed(42)
        if stream:
            first_generated = next(iter(model.generate_tokens_stream(input_context, block_size, max_new_tokens)))
        else:
            all_tokens = model.generate_tokens(input_context, block_size, max_new_tokens)
            first_generated = all_tokens[len(input_context[0])]

        # generate with stop_token set to the first generated token
        torch.manual_seed(42)
        if stream:
            stopped_tokens = list(model.generate_tokens_stream(input_context, block_size, max_new_tokens,
                                                               stop_token=first_generated))
            # generation should have stopped after the stop_token
            self.assertEqual(stopped_tokens, [first_generated])
        else:
            stopped_tokens = model.generate_tokens(input_context, block_size, max_new_tokens,
                                                   stop_token=first_generated)
            # generation should have stopped after the stop_token
            self.assertEqual(stopped_tokens, input_context[0] + [first_generated])

    def _make_gemma_like_layers(self, vocab_size=16, n_embd=8, n_head=2, n_kv_heads=2,
                                head_dim=4, intermediate_size=16, n_blocks=1,
                                sliding_window=None):
        """Build a small Gemma-like layer config for testing."""
        qkv_dim = n_head * head_dim + 2 * n_kv_heads * head_dim
        attn_args = {"num_heads": n_head, "num_kv_heads": n_kv_heads,
                     "rope_theta": 10000.0, "head_dim": head_dim}
        if sliding_window is not None:
            attn_args["sliding_window"] = sliding_window
        block = lambda: {"transformerblock": {
            "attn_block": {"sequential": [
                {"rmsnorm": {"normalized_shape": n_embd}},
                {"linear": {"in_features": n_embd, "out_features": qkv_dim, "bias": False}},
                {"attention": dict(attn_args)},
                {"linear": {"in_features": n_head * head_dim, "out_features": n_embd, "bias": False}},
            ]},
            "mlp_block": {"sequential": [
                {"rmsnorm": {"normalized_shape": n_embd}},
                {"gatedmlp": {"in_features": n_embd, "intermediate_size": intermediate_size,
                              "bias": False, "activation": "gelu_pytorch_tanh"}},
            ]},
            "post_attn_norm": {"rmsnorm": {"normalized_shape": n_embd}},
            "post_mlp_norm": {"rmsnorm": {"normalized_shape": n_embd}},
            "post_norm_on_residual": False,
        }}
        return [
            {"scaledembedding": {
                "num_embeddings": vocab_size, "embedding_dim": n_embd,
                "scale": float(n_embd ** 0.5),
            }},
            *[block() for _ in range(n_blocks)],
            {"rmsnorm": {"normalized_shape": n_embd}},
            {"linear": {"in_features": n_embd, "out_features": vocab_size, "bias": False}},
            {"softmaxlast": {"dim": -1}},
        ]

    @parameterized.expand([
        ("full_attention", None),
        ("sliding_window_2", 2),
        ("sliding_window_3", 3),
    ])
    def test_incremental_decode_matches_full_forward(self, _name, sliding_window):
        """Token-by-token decode with KV cache must match a full-context forward.

        This guards the generate path: the cached incremental logits at the
        final position must equal the logits from a single full-sequence
        forward.  Sliding-window layers in particular must apply their window
        mask during incremental decode (block_size == 1), not only during
        prefill — otherwise a single-token query attends to the whole cache.
        """
        torch.manual_seed(0)
        vocab_size = 16
        layers = self._make_gemma_like_layers(vocab_size=vocab_size, n_blocks=2,
                                               sliding_window=sliding_window)
        model = NeuralNetworkModel("test_decode_eq", Mapper(layers, {"sgd": {}}))
        model.eval()
        model.layers.training = False
        # Randomize weights so attention is non-trivial.
        for p in model.parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, std=0.2)

        # Sequence length deliberately exceeds the sliding window.
        seq = [[3, 1, 4, 1, 5, 9, 2]]
        seq_t = torch.tensor(seq, dtype=torch.long)

        # Full-context forward, no cache.
        full_acts, _ = model(seq_t, skip_softmax=True)
        full_logits = full_acts[-1][:, -1, :]

        # Incremental decode: feed one token at a time with a KV cache.
        cache, pos_embeddings = model._attach_kv_cache()
        try:
            inc_logits = None
            for tok in seq[0]:
                inp = torch.tensor([[tok]], dtype=torch.long)
                acts, _ = model(inp, skip_softmax=True)
                inc_logits = acts[-1][:, -1, :]
        finally:
            model._detach_kv_cache(pos_embeddings)

        self.assertTrue(
            torch.allclose(full_logits, inc_logits, atol=1e-4),
            f"Incremental decode diverged from full forward (sliding_window={sliding_window}); "
            f"max diff={ (full_logits - inc_logits).abs().max().item() }")

    def _make_kv_shared_layers(self, vocab_size=16, n_embd=8, n_head=2, n_kv_heads=1,
                               head_dim=4, intermediate_size=16):
        """Build a 3-block Gemma 4-style config where block 2 shares KV from block 0."""
        qkv_dim = n_head * head_dim + 2 * n_kv_heads * head_dim
        def block(kv_shared_layer_idx=None):
            attn = {"num_heads": n_head, "num_kv_heads": n_kv_heads,
                    "rope_theta": 10000.0, "head_dim": head_dim}
            if kv_shared_layer_idx is not None:
                attn["kv_shared_layer_idx"] = kv_shared_layer_idx
            return {"transformerblock": {
                "attn_block": {"sequential": [
                    {"rmsnorm": {"normalized_shape": n_embd}},
                    {"linear": {"in_features": n_embd, "out_features": qkv_dim, "bias": False}},
                    {"attention": attn},
                    {"linear": {"in_features": n_head * head_dim, "out_features": n_embd, "bias": False}},
                ]},
                "mlp_block": {"sequential": [
                    {"rmsnorm": {"normalized_shape": n_embd}},
                    {"gatedmlp": {"in_features": n_embd, "intermediate_size": intermediate_size,
                                  "bias": False, "activation": "gelu_pytorch_tanh"}},
                ]},
                "post_attn_norm": {"rmsnorm": {"normalized_shape": n_embd}},
                "post_mlp_norm": {"rmsnorm": {"normalized_shape": n_embd}},
                "post_norm_on_residual": False,
            }}
        return [
            {"scaledembedding": {"num_embeddings": vocab_size, "embedding_dim": n_embd,
                                 "scale": float(n_embd ** 0.5)}},
            block(),                       # block 0: reference (non-shared)
            block(),                       # block 1: non-shared
            block(kv_shared_layer_idx=0),  # block 2: shares KV from block 0
            {"rmsnorm": {"normalized_shape": n_embd}},
            {"linear": {"in_features": n_embd, "out_features": vocab_size, "bias": False}},
            {"softmaxlast": {"dim": -1}},
        ]

    def test_kv_shared_prefill_matches_incremental_decode(self):
        """KV-shared layers: all-at-once prefill must match token-by-token decode.

        Gemma 4 E2B reuses K/V from a reference layer in 20 of 35 layers.
        Both the prefill (block_size == N) and incremental decode (block_size
        == 1) code paths must agree at the final position, otherwise generation
        degrades (e.g. repeating tokens) the moment decoding starts.
        """
        torch.manual_seed(0)
        layers = self._make_kv_shared_layers()
        model = NeuralNetworkModel("test_kv_shared", Mapper(layers, {"sgd": {}}))
        model.eval()
        model.layers.training = False
        for p in model.parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, std=0.2)

        seq = [[3, 1, 4, 1, 5]]

        # All-at-once prefill with cache.
        cache, pos = model._attach_kv_cache()
        try:
            prefill_acts, _ = model(torch.tensor(seq, dtype=torch.long), skip_softmax=True)
            prefill_logits = prefill_acts[-1][:, -1, :]
        finally:
            model._detach_kv_cache(pos)

        # Token-by-token decode with a fresh cache.
        cache, pos = model._attach_kv_cache()
        try:
            decode_logits = None
            for tok in seq[0]:
                acts, _ = model(torch.tensor([[tok]], dtype=torch.long), skip_softmax=True)
                decode_logits = acts[-1][:, -1, :]
        finally:
            model._detach_kv_cache(pos)

        self.assertTrue(
            torch.allclose(prefill_logits, decode_logits, atol=1e-4),
            f"KV-shared prefill vs decode diverged; "
            f"max diff={ (prefill_logits - decode_logits).abs().max().item() }")

    def test_gemma4_numerical_parity_with_hf_transformers(self):
        """Our mapped Gemma 4 must reproduce HF transformers logits exactly.

        Builds a tiny random Gemma4ForCausalLM covering every architectural
        feature (sliding + full attention, partial RoPE, KV-shared layers,
        PLE, layer_scalar, GQA), maps its weights, and compares logits for:
        plain forward (no cache), cached prefill, and token-by-token decode.
        """
        try:
            from transformers import Gemma4TextConfig
            from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        except ImportError:
            self.skipTest("Gemma 4 not available in installed transformers")

        torch.manual_seed(42)
        cfg = Gemma4TextConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
            head_dim=8, global_head_dim=16, sliding_window=4,
            layer_types=['sliding_attention', 'full_attention',
                         'sliding_attention', 'full_attention'],
            num_kv_shared_layers=2, hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=64, max_position_embeddings=64,
        )
        hf_model = Gemma4ForCausalLM(cfg)
        hf_model.eval()
        with torch.no_grad():
            for p in hf_model.parameters():
                nn.init.normal_(p, std=0.2 if p.ndim >= 2 else 0.1)

        mapped = Mapper.map_hf_state_dict_to_custom(
            dict(hf_model.state_dict()), cfg.num_hidden_layers, cfg)
        model = NeuralNetworkModel(
            "test_hf_parity", Mapper(Mapper.from_hf_config(cfg), {"adamw": {"lr": 1e-4}}))
        model.load_state_dict(mapped)
        model.eval()
        model.layers.training = False

        input_ids = torch.randint(0, cfg.vocab_size, (1, 7))
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits

            acts, _ = model(input_ids, skip_softmax=True)
            plain_logits = acts[-1]

            cache, pos = model._attach_kv_cache()
            try:
                acts, _ = model(input_ids, skip_softmax=True)
                prefill_logits = acts[-1]
            finally:
                model._detach_kv_cache(pos)

            cache, pos = model._attach_kv_cache()
            try:
                decode_logits = None
                for t in range(input_ids.shape[1]):
                    acts, _ = model(input_ids[:, t:t + 1], skip_softmax=True)
                    decode_logits = acts[-1]
            finally:
                model._detach_kv_cache(pos)

        self.assertTrue(torch.allclose(hf_logits, plain_logits, atol=1e-5),
                        f"plain forward diverged from HF; max diff="
                        f"{(hf_logits - plain_logits).abs().max().item()}")
        self.assertTrue(torch.allclose(hf_logits, prefill_logits, atol=1e-5),
                        f"cached prefill diverged from HF; max diff="
                        f"{(hf_logits - prefill_logits).abs().max().item()}")
        self.assertTrue(torch.allclose(hf_logits[:, -1:], decode_logits, atol=1e-5),
                        f"incremental decode diverged from HF; max diff="
                        f"{(hf_logits[:, -1:] - decode_logits).abs().max().item()}")

    @parameterized.expand([
        (1.0, None),
        (1.0, 3),
        (0.0, None),
    ])
    def test_generate_tokens_with_bfloat16_model(self, temperature, top_k):
        """Generation must succeed for bfloat16 models (imported Gemma-like precision)."""
        layers = self._make_gemma_like_layers()
        model = NeuralNetworkModel("test_bf16", Mapper(layers, {"adamw": {"lr": 1e-4}}))
        model.to(dtype=torch.bfloat16)

        tokens = model.generate_tokens([[0]], block_size=8, max_new_tokens=3,
                                       temperature=temperature, top_k=top_k)

        self.assertIsNotNone(tokens)
        self.assertGreaterEqual(len(tokens), 1)
        self.assertFalse(model.layers.training)

    @parameterized.expand([
        (1.0, None),
        (1.0, 3),
    ])
    def test_generate_tokens_stream_with_bfloat16_model(self, temperature, top_k):
        """Streaming generation must succeed for bfloat16 models."""
        layers = self._make_gemma_like_layers()
        model = NeuralNetworkModel("test_bf16_stream", Mapper(layers, {"adamw": {"lr": 1e-4}}))
        model.to(dtype=torch.bfloat16)

        streamed = list(model.generate_tokens_stream([[0]], block_size=8, max_new_tokens=3,
                                                     temperature=temperature, top_k=top_k))

        self.assertEqual(len(streamed), 3)
        for token in streamed:
            self.assertIsInstance(token, int)

    @parameterized.expand([
        (0.9, None),
        (0.95, None),
        (0.9, 5),
    ])
    def test_generate_tokens_with_top_p(self, top_p, top_k):
        """Top-P (nucleus) sampling should produce valid tokens."""
        layers = self._make_gemma_like_layers()
        model = NeuralNetworkModel("test_top_p", Mapper(layers, {"adamw": {"lr": 1e-4}}))

        tokens = model.generate_tokens([[0]], block_size=8, max_new_tokens=5,
                                       temperature=1.0, top_k=top_k, top_p=top_p)

        self.assertIsNotNone(tokens)
        self.assertGreaterEqual(len(tokens), 1)

    def test_generate_tokens_stream_with_top_p(self):
        """Streaming generation with top-p should yield valid tokens."""
        layers = self._make_gemma_like_layers()
        model = NeuralNetworkModel("test_top_p_stream", Mapper(layers, {"adamw": {"lr": 1e-4}}))

        streamed = list(model.generate_tokens_stream([[0]], block_size=8, max_new_tokens=3,
                                                     temperature=1.0, top_p=0.95))

        self.assertEqual(len(streamed), 3)
        for token in streamed:
            self.assertIsInstance(token, int)

    def test_top_p_deterministic_with_low_threshold(self):
        """Very low top-p should behave nearly like greedy (only the top token survives)."""
        layers = self._make_gemma_like_layers()
        model = NeuralNetworkModel("test_top_p_det", Mapper(layers, {"adamw": {"lr": 1e-4}}))

        torch.manual_seed(42)
        tokens_top_p = model.generate_tokens([[0]], block_size=8, max_new_tokens=5,
                                             temperature=1.0, top_p=0.01)
        tokens_greedy = model.generate_tokens([[0]], block_size=8, max_new_tokens=5,
                                              temperature=0.0)

        self.assertEqual(tokens_top_p, tokens_greedy)

    def test_compute_output_with_bfloat16_model_converts_float_input(self):
        """compute_output must cast floating-point inputs to model dtype for bf16 models."""
        layers = [
            {"linear": {"in_features": 4, "out_features": 4}},
            {"softmax": {"dim": -1}},
        ]
        model = NeuralNetworkModel("test_bf16_output", Mapper(layers, {"sgd": {}}))
        model.to(dtype=torch.bfloat16)

        # float32 input should be converted to bf16 before the bf16 linear layer
        output, cost = model.compute_output([[0.1, 0.2, 0.3, 0.4]])

        self.assertIsNotNone(output)
        self.assertIsNone(cost)

    @unittest.skipUnless(os.path.exists(NeuralNetworkModel.SHM_PATH),
                         f"Requires {NeuralNetworkModel.SHM_PATH} (shared memory)")
    def test_deserialize_restores_bfloat16_dtype(self):
        """deserialize must restore bfloat16 dtype so parameters are not silently upcast to float32."""
        layers = self._make_gemma_like_layers()
        model = NeuralNetworkModel("test_bf16_deser", Mapper(layers, {"adamw": {"lr": 1e-4}}))
        model.to(dtype=torch.bfloat16)
        model.serialize()

        restored = NeuralNetworkModel.deserialize("test_bf16_deser")
        try:
            for name, param in restored.named_parameters():
                self.assertEqual(param.dtype, torch.bfloat16,
                                 f"Parameter {name} should be bfloat16 after deserialization")
            # generation must work on the deserialized bf16 model
            tokens = restored.generate_tokens([[0]], block_size=8, max_new_tokens=2)
            self.assertGreaterEqual(len(tokens), 1)
        finally:
            NeuralNetworkModel.delete("test_bf16_deser")

    @parameterized.expand([
        ([{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
          {"tanh": {}},
          {"linear": {"in_features": 2, "out_features": 8}},
          {"softmaxlast": {"dim": -1}}],
         {"sgd": {"lr": .01}},
         [1, 2], [2, 3], 2, 1, 2),
        ([{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
          {"gelu": {}},
          {"linear": {"in_features": 2, "out_features": 8}},
          {"softmaxlast": {"dim": -1}}],
         {"adamw": {"lr": .01}},
         [1, 2], [2, 3], 2, 1, 2),
        ([{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
          {"linear": {"in_features": 2, "out_features": 4 * 2}},
          {"gelu": {}},
          {"linear": {"in_features": 4 * 2, "out_features": 2}},
          {"linear": {"in_features": 2, "out_features": 8}},
          {"softmaxlast": {"dim": -1}}],
         {"adamw": {"lr": .01}},
         [1, 2, 3, 4], [2, 3, 4, 5], 4, 2, 2),
        ([{"embedding": {"num_embeddings": 16, "embedding_dim": 2}},
          {"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 4 * 2}},
          {"gelu": {}},
          {"linear": {"in_features": 4 * 2, "out_features": 2}},
          {"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 16, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         {"adamw": {"lr": 1e-3}},
         [1, 2, 3, 4], [2, 3, 4, 5], 4, 2, 2),
        ([{"embedding": {"num_embeddings": 16, "embedding_dim": 2}},
          {"dropout": {"p": 0.0}},
          {"sequential": [{"layernorm": {"normalized_shape": 2}},
                          {"linear": {"in_features": 2, "out_features": 4 * 2}},
                          {"gelu": {}},
                          {"linear": {"in_features": 4 * 2, "out_features": 2}},
                          {"dropout": {"p": 0.0}}]},
          {"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 16, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         {"adamw": {"lr": .008}},
         [1, 2, 3, 4], [2, 3, 4, 5], 2, 2, 2),
        ([{"summation": [{"embedding": {"num_embeddings": 16, "embedding_dim": 2}},
                         {"position": {"num_embeddings": 4, "embedding_dim": 2}}]},
          {"dropout": {"p": 0.0}}] +
         [{"residual": [{"sequential": [{"layernorm": {"normalized_shape": 2}},
                                        {"linear": {"in_features": 2, "out_features": 3 * 2}},
                                        {"attention": {"num_heads": 1, "dropout": 0.0}},
                                        {"linear": {"in_features": 2, "out_features": 2}},
                                        {"dropout": {"p": 0.0}}]},
                        {"sequential": [{"layernorm": {"normalized_shape": 2}},
                                        {"linear": {"in_features": 2, "out_features": 4 * 2}},
                                        {"gelu": {}},
                                        {"linear": {"in_features": 4 * 2, "out_features": 2}},
                                        {"dropout": {"p": 0.0}}]}
                        ]} for _ in range(2)] +
         [{"layernorm": {"normalized_shape": 2}},
          {"linear": {"in_features": 2, "out_features": 16, "bias": False}},
          {"softmaxlast": {"dim": -1}}],
         {"adamw": {"lr": 3e-4}},
         [1,2,3,4,5,6,7,8], [2,3,4,5,6,7,8,9], 3, 4, 2),
    ])
    @unittest.skipUnless(os.path.exists(NeuralNetworkModel.SHM_PATH), f"Requires {NeuralNetworkModel.SHM_PATH} (shared memory)")
    def test_train(self, layers: list[dict], optimizer: dict,
                   input_data: list, target: list, epochs: int, batch_size: int, step_size: int):

        # clean up any persisted previous test model
        NeuralNetworkModel.delete("test")

        # create model
        model = NeuralNetworkModel("test", Mapper(layers, optimizer))

        # record initial conditions
        block_size = len(input_data) // batch_size
        initial_params = [p.tolist() for p in model.parameters()]
        lr: float = model.optimizer.param_groups[0]["lr"]

        # Add average cost history to test cap at 100
        model.avg_cost_history = [1.0] * 100

        # make sure test data is good for training
        self.assertEqual(len(input_data), len(target))

        with patch("neural_net_model.Loader") as MockLoader:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.return_value = tuple(np.array(l, dtype=np.int32) for l in [input_data, target])
            model.train_model("mock_ds", 1, epochs, batch_size, block_size, step_size)

        # record updated
        updated_params = [p.tolist() for p in model.parameters()]
        updated_optim_params =[p.tolist() for p in model.optimizer.param_groups[0]["params"]]

        # Check that the model data is still valid
        self.assertEqual(len(updated_params), len(initial_params))
        for u, i in zip(updated_params, initial_params):
            self.assertEqual(np.shape(u), np.shape(i))

        # Ensure training progress
        for u, i in zip(updated_params, initial_params):
            self.assertFalse(np.allclose(u, i))
        self.assertEqual(len(model.progress), epochs)
        self.assertEqual(sum([p["cost"] for p in model.progress]) / len(model.progress), model.avg_cost)
        self.assertEqual(len(model.avg_cost_history), 100)
        self.assertEqual(model.avg_cost_history[0], 1.0)
        self.assertEqual(model.avg_cost_history[-1], model.avg_cost)
        self.assertIsNotNone(model.stats)
        self.assertEqual("Trained", model.status.get("code"))
        self.assertTrue(model.layers.training)

        # Deserialize and check if recorded training
        persisted_model = NeuralNetworkModel.deserialize(model.model_id)

        # record persisted
        persisted_params = [p.tolist() for p in persisted_model.parameters()]
        persisted_lr: float = persisted_model.optimizer.param_groups[0]["lr"]
        persisted_optim_params = [p.tolist() for p in persisted_model.optimizer.param_groups[0]["params"]]

        # Verify model correctly deserialized
        self.assertEqual(len(persisted_params), len(updated_params))
        for p, u in zip(persisted_params, updated_params):
            self.assertEqual(np.shape(p), np.shape(u))
            np.testing.assert_allclose(p, u)
        self.assertEqual(persisted_model.optimizer.__class__, model.optimizer.__class__)
        self.assertEqual(persisted_lr, lr)
        for p, u in zip(persisted_optim_params, updated_optim_params):
            self.assertEqual(np.shape(p), np.shape(u))
            np.testing.assert_allclose(p, u, rtol=1e-5, atol=1e-8)
        self.assertEqual(len(persisted_model.progress), len(model.progress))
        self.assertEqual(persisted_model.avg_cost, model.avg_cost)
        self.assertEqual(persisted_model.avg_cost_history, model.avg_cost_history)
        self.assertEqual(persisted_model.stats, model.stats)
        self.assertEqual(persisted_model.status, model.status)

    def test_unsupported_layer(self):
        with self.assertRaises(ValueError) as context:
            NeuralNetworkModel("test", Mapper([{"unknown": {}}], {"sgd": {}}))

        # Assert the error message
        self.assertEqual(str(context.exception), "Unsupported layer: {'unknown': {}}")

    def test_unsupported_optimizer(self):
        with self.assertRaises(ValueError) as context:
            NeuralNetworkModel("test", Mapper([{"relu": {}}], {"unknown": {}}))

        # Assert the error message
        self.assertEqual(str(context.exception), "Unsupported optimizer: {'unknown': {}}")

    def test_invalid_model_deserialization(self):
        # Test that deserializing a nonexistent model raises a KeyError
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("nonexistent_model")

    @unittest.skipUnless(os.path.exists(NeuralNetworkModel.SHM_PATH), f"Requires {NeuralNetworkModel.SHM_PATH} (shared memory)")
    def test_delete(self):
        model = NeuralNetworkModel("test", Mapper([{"linear": {"in_features": 9, "out_features": 9}}],
                                                  {"sgd": {}}))
        model.serialize()
        model_path = NeuralNetworkModel.get_model_path(model.model_id)
        model_in_shm_path = os.path.join(NeuralNetworkModel.SHM_PATH, model_path)

        self.assertTrue(os.path.exists(model_in_shm_path))
        time.sleep(1) # wait a bit for cache to flush to disk
        self.assertTrue(os.path.exists(model_path))

        NeuralNetworkModel.delete("test")
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("test")

    def test_serialize_uses_pickle_protocol_5(self):
        """Serialize uses pickle protocol 5 to support large models with compression."""
        model = NeuralNetworkModel("test_pickle", Mapper(
            [{"linear": {"in_features": 3, "out_features": 3}}], {"sgd": {}}))
        with patch("neural_net_model.torch.save") as mock_save:
            model.serialize()
            _, kwargs = mock_save.call_args
            self.assertEqual(kwargs["pickle_protocol"], 5)
        NeuralNetworkModel.delete("test_pickle")

    def test_invalid_delete(self):
        # No error raised for failing to delete
        NeuralNetworkModel.delete("nonexistent")

    def test_weights_property(self):
        model = NeuralNetworkModel("test", Mapper(
            [{"linear": {"in_features": 3, "out_features": 5}},
             {"relu": {}},
             {"linear": {"in_features": 5, "out_features": 2}}],
            {"sgd": {}}))
        
        weights = model._weights
        
        # Should have 2 weight matrices (from 2 linear layers) and 2 biases (which are None in _weights)
        self.assertEqual(len(weights), 4)
        # First weight should be 2D (weight matrix from first linear layer)
        self.assertIsNotNone(weights[0])
        self.assertEqual(weights[0].ndim, 2)
        # Second should be None (bias from first linear layer)
        self.assertIsNone(weights[1])
        # Third weight should be 2D (weight matrix from second linear layer)
        self.assertIsNotNone(weights[2])
        self.assertEqual(weights[2].ndim, 2)
        # Fourth should be None (bias from second linear layer)
        self.assertIsNone(weights[3])

    @patch.dict(os.environ, {"RANK": "0", "LOCAL_RANK": "0"})
    @patch('neural_net_model.torch.cuda.set_device')
    @patch('neural_net_model.torch.cuda.is_available', return_value=True)
    def test_to_method_with_ddp_cuda(self, mock_cuda_available, mock_set_device):
        model = NeuralNetworkModel("test", Mapper(
            [{"linear": {"in_features": 3, "out_features": 3}}],
            {"sgd": {}}))
        
        # Mock super().to() to avoid actual CUDA call
        with patch.object(nn.Module, 'to', return_value=None):
            model.to("cuda")
        
        # Should have called set_device with cuda:0
        mock_set_device.assert_called_once_with("cuda:0")

    def test_to_method_cpu(self):
        model = NeuralNetworkModel("test", Mapper(
            [{"linear": {"in_features": 3, "out_features": 3}}],
            {"sgd": {}}))
        
        # Should not raise any errors
        model.to("cpu")
        
        # Verify device
        device = next(model.parameters()).device
        self.assertEqual(device.type, "cpu")

    @unittest.skipUnless(os.path.exists(NeuralNetworkModel.SHM_PATH), f"Requires {NeuralNetworkModel.SHM_PATH} (shared memory)")
    def test_cache_miss(self):
        model = NeuralNetworkModel("test", Mapper([{"linear": {"in_features": 9, "out_features": 9}}],
                                                  {"sgd": {}}))
        model.serialize()
        model_path = NeuralNetworkModel.get_model_path(model.model_id)
        model_in_shm_path = os.path.join(NeuralNetworkModel.SHM_PATH, model_path)

        self.assertTrue(os.path.exists(model_in_shm_path))
        time.sleep(1) # wait a bit for cache to flush to disk
        self.assertTrue(os.path.exists(model_path))

        os.remove(model_in_shm_path)

        self.assertFalse(os.path.exists(model_in_shm_path))

        model = NeuralNetworkModel.deserialize("test")

        self.assertIsNotNone(model)
        self.assertTrue(os.path.exists(model_in_shm_path))

    def test_train_exception_handling(self):
        # Create a tiny model
        layers = [{"linear": {"in_features": 4, "out_features": 4}}, {"tanh": {}}]
        model = NeuralNetworkModel("test-exc", Mapper(layers, {"sgd": {}}))

        # small training parameters
        epochs = 1
        batch_size = 1
        block_size = 1
        step_size = 1

        # Patch Loader to raise an exception when next_batch is called
        with patch("neural_net_model.Loader") as MockLoader, \
             patch.object(NeuralNetworkModel, 'serialize') as mock_serialize:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.side_effect = Exception("test error")

            # Run training and expect exception to be propagated
            with self.assertRaises(Exception) as cm:
                model.train_model("mock_ds", 0, epochs, batch_size, block_size, step_size)

            # Ensure the exception message is the one we raised
            self.assertIn("test error", str(cm.exception))

            # serialize should have been called at least twice: initial serialize and in exception handler
            self.assertTrue(mock_serialize.called)
            self.assertGreaterEqual(mock_serialize.call_count, 2)

            # status should have been set to Error by the exception handler
            self.assertEqual(model.status.get("code"), "Error")
            self.assertIn("Training epoch 1 failed", model.status.get("message"))


    @unittest.skipUnless(os.path.exists(NeuralNetworkModel.SHM_PATH), f"Requires {NeuralNetworkModel.SHM_PATH} (shared memory)")
    def test_train_cpu_no_amp(self):
        """Training on CPU does not use AMP autocast or GradScaler."""
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"tanh": {}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-no-amp", Mapper(layers, {"sgd": {"lr": .01}}))
        input_data = [1, 2]
        target = [2, 3]

        with patch("neural_net_model.Loader") as MockLoader, \
             patch('neural_net_model.torch.amp.GradScaler') as mock_scaler_cls:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.return_value = tuple(
                np.array(l, dtype=np.int32) for l in [input_data, target])

            model.train_model("mock_ds", 1, 2, 1, 2, 1)

        # Verify training completed successfully on CPU
        self.assertEqual("Trained", model.status.get("code"))
        self.assertEqual(len(model.progress), 2)
        # GradScaler should never be instantiated on CPU
        mock_scaler_cls.assert_not_called()

    def test_amp_dtype_selection_bfloat16(self):
        """When CUDA supports bf16, bfloat16 is selected and GradScaler is disabled."""
        with patch('neural_net_model.torch.cuda.is_bf16_supported', return_value=True):
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.assertEqual(amp_dtype, torch.bfloat16)
            # GradScaler should be disabled for bfloat16
            scaler_enabled = (amp_dtype == torch.float16)
            self.assertFalse(scaler_enabled)

    def test_amp_dtype_selection_float16(self):
        """When CUDA does not support bf16, float16 is selected and GradScaler is enabled."""
        with patch('neural_net_model.torch.cuda.is_bf16_supported', return_value=False):
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.assertEqual(amp_dtype, torch.float16)
            # GradScaler should be enabled for float16
            scaler_enabled = (amp_dtype == torch.float16)
            self.assertTrue(scaler_enabled)

    def test_amp_nullcontext_on_cpu_device(self):
        """AMP uses nullcontext on CPU, autocast is not invoked."""
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"tanh": {}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-amp-off", Mapper(layers, {"sgd": {"lr": .01}}))
        device = next(model.parameters()).device
        self.assertNotEqual(device.type, 'cuda')

    def test_amp_autocast_not_called_on_cpu(self):
        """On CPU, nullcontext is used and autocast is never called."""
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"tanh": {}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-amp-ac", Mapper(layers, {"sgd": {"lr": .01}}))
        input_data = [1, 2]
        target = [2, 3]

        with patch("neural_net_model.Loader") as MockLoader, \
             patch.object(NeuralNetworkModel, 'serialize'), \
             patch('neural_net_model.torch.amp.autocast') as mock_autocast:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.return_value = tuple(
                np.array(l, dtype=np.int32) for l in [input_data, target])
            model.train_model("mock_ds", 1, 1, 1, 2, 1)

            # autocast should NOT be called on CPU (nullcontext is used instead)
            mock_autocast.assert_not_called()

    @parameterized.expand([
        (True, torch.bfloat16, False),
        (False, torch.float16, True),
    ])
    def test_amp_cuda_training_path(self, bf16_supported, expected_dtype, expected_scaling):
        """AMP configuration is properly set up when training on CUDA device."""
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"tanh": {}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-amp-cuda", Mapper(layers, {"sgd": {"lr": .01}}))
        input_data = [1, 2]
        target = [2, 3]

        # Create a mock CUDA device
        mock_device = MagicMock()
        mock_device.type = 'cuda'
        mock_param = MagicMock()
        mock_param.device = mock_device

        # Patch Tensor.to so .to(mock_device) returns self instead of failing
        original_tensor_to = torch.Tensor.to

        def patched_tensor_to(self_tensor, *args, **kwargs):
            if args and isinstance(args[0], MagicMock):
                return self_tensor
            return original_tensor_to(self_tensor, *args, **kwargs)

        with patch("neural_net_model.Loader") as MockLoader, \
             patch.object(NeuralNetworkModel, 'serialize'), \
             patch.object(NeuralNetworkModel, '_record_training_overall_progress'), \
             patch('neural_net_model.torch.cuda.is_bf16_supported', return_value=bf16_supported), \
             patch('neural_net_model.torch.cuda.synchronize'), \
             patch('neural_net_model.torch.amp.GradScaler') as MockScaler, \
             patch('neural_net_model.torch.amp.autocast') as MockAutocast, \
             patch.object(torch.Tensor, 'to', patched_tensor_to):
            # Set up mock loader
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.return_value = tuple(
                np.array(l, dtype=np.int32) for l in [input_data, target])

            # Set up mock scaler
            mock_scaler = MagicMock()
            MockScaler.return_value = mock_scaler
            mock_scaled_loss = MagicMock()
            mock_scaler.scale.return_value = mock_scaled_loss

            # Set up mock autocast as context manager
            mock_ctx = MagicMock()
            MockAutocast.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            # Patch next() to return fake CUDA device param for device detection only
            original_next = next
            call_count = [0]

            def patched_next(iterator, *args):
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_param
                return original_next(iterator, *args)

            with patch('neural_net_model.next', side_effect=patched_next):
                model.train_model("mock_ds", 1, 1, 1, 2, 1)

            # Verify autocast was created with CUDA and correct dtype
            MockAutocast.assert_called_once_with('cuda', dtype=expected_dtype)
            if expected_scaling:
                # Verify GradScaler created with correct enabled flag
                MockScaler.assert_called_once_with('cuda')
                # Verify scaler.scale was called for backward pass
                mock_scaler.scale.assert_called()
                mock_scaled_loss.backward.assert_called()
                # Verify scaler.step and update were called
                mock_scaler.step.assert_called_once_with(model.optimizer)
                mock_scaler.update.assert_called_once()
            else:
                # Verify GradScaler not created
                MockScaler.assert_not_called()
                # Verify scaler not used
                mock_scaler.assert_not_called()

    def test_amp_cuda_unscales_activation_grads_before_update(self):
        """Activation gradients are unscaled using the current scale before scaler.update()."""
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"tanh": {}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-unscale", Mapper(layers, {"sgd": {"lr": .01}}))
        input_data = [1, 2]
        target = [2, 3]

        mock_device = MagicMock()
        mock_device.type = 'cuda'
        mock_param = MagicMock()
        mock_param.device = mock_device

        original_tensor_to = torch.Tensor.to

        def patched_tensor_to(self_tensor, *args, **kwargs):
            if args and isinstance(args[0], MagicMock):
                return self_tensor
            return original_tensor_to(self_tensor, *args, **kwargs)

        with patch("neural_net_model.Loader") as MockLoader, \
             patch.object(NeuralNetworkModel, 'serialize'), \
             patch.object(NeuralNetworkModel, '_record_training_overall_progress'), \
             patch('neural_net_model.torch.cuda.is_bf16_supported', return_value=False), \
             patch('neural_net_model.torch.cuda.synchronize'), \
             patch('neural_net_model.torch.amp.GradScaler') as MockScaler, \
             patch('neural_net_model.torch.amp.autocast') as MockAutocast, \
             patch.object(torch.Tensor, 'to', patched_tensor_to):
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.return_value = tuple(
                np.array(l, dtype=np.int32) for l in [input_data, target])

            mock_scaler = MagicMock()
            MockScaler.return_value = mock_scaler
            mock_scaled_loss = MagicMock()
            mock_scaler.scale.return_value = mock_scaled_loss
            mock_scaler.get_scale.return_value = 256.0

            mock_ctx = MagicMock()
            MockAutocast.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            original_next = next
            call_count = [0]

            def patched_next(iterator, *args):
                call_count[0] += 1
                if call_count[0] == 1:
                    return mock_param
                return original_next(iterator, *args)

            with patch('neural_net_model.next', side_effect=patched_next):
                model.train_model("mock_ds", 1, 1, 1, 2, 1)

            # get_scale must be called to retrieve the scale for unscaling
            mock_scaler.get_scale.assert_called()
            # Verify ordering: step -> get_scale -> update
            expected_order = [call.step(model.optimizer), call.get_scale(), call.update()]
            actual_calls = mock_scaler.method_calls
            step_calls = [c for c in actual_calls if c[0] in ('step', 'get_scale', 'update')]
            self.assertEqual([c[0] for c in step_calls], ['step', 'get_scale', 'update'])

    @patch("neural_net_model.platform.system", return_value="Linux")
    @patch("neural_net_model.os.path.isdir", side_effect=lambda p: p == "/dev/shm")
    @patch("neural_net_model.os.access", return_value=True)
    def test_detect_shm_path_linux(self, mock_access, mock_isdir, mock_system):
        self.assertEqual(NeuralNetworkModel._detect_shm_path(), "/dev/shm")

    @patch("neural_net_model.platform.system", return_value="Darwin")
    @patch("neural_net_model.os.path.isdir", side_effect=lambda p: p == "/Volumes/RAMDisk")
    @patch("neural_net_model.os.access", return_value=True)
    def test_detect_shm_path_macos_ramdisk(self, mock_access, mock_isdir, mock_system):
        self.assertEqual(NeuralNetworkModel._detect_shm_path(), "/Volumes/RAMDisk")

    @patch("neural_net_model.platform.system", return_value="Darwin")
    @patch("neural_net_model.os.path.isdir", return_value=False)
    def test_detect_shm_path_macos_fallback(self, mock_isdir, mock_system):
        self.assertEqual(NeuralNetworkModel._detect_shm_path(), tempfile.gettempdir())

    @patch("neural_net_model.platform.system", return_value="Windows")
    @patch("neural_net_model.os.path.isdir", return_value=False)
    def test_detect_shm_path_other_os(self, mock_isdir, mock_system):
        self.assertEqual(NeuralNetworkModel._detect_shm_path(), tempfile.gettempdir())


    def _make_hf_config(self, n_layer=1, n_embd=32, n_head=2,
                        vocab_size=64, n_positions=16):
        cfg = MagicMock(spec=[])
        cfg.vocab_size = vocab_size
        cfg.n_embd = n_embd
        cfg.n_head = n_head
        cfg.n_layer = n_layer
        cfg.n_positions = n_positions
        cfg.resid_pdrop = 0.0
        cfg.embd_pdrop  = 0.0
        cfg.attn_pdrop  = 0.0
        cfg.to_dict = lambda: {}
        return cfg

    def _make_hf_sd(self, n_layer, n_embd, vocab_size, block_size):
        sd = {}
        sd["transformer.wte.weight"] = torch.zeros(vocab_size, n_embd)
        sd["transformer.wpe.weight"] = torch.zeros(block_size, n_embd)
        for i in range(n_layer):
            p = f"transformer.h.{i}"
            sd[f"{p}.ln_1.weight"] = torch.ones(n_embd)
            sd[f"{p}.ln_1.bias"]   = torch.zeros(n_embd)
            sd[f"{p}.attn.c_attn.weight"] = torch.zeros(n_embd, 3 * n_embd)
            sd[f"{p}.attn.c_attn.bias"]   = torch.zeros(3 * n_embd)
            sd[f"{p}.attn.c_proj.weight"] = torch.zeros(n_embd, n_embd)
            sd[f"{p}.attn.c_proj.bias"]   = torch.zeros(n_embd)
            sd[f"{p}.ln_2.weight"] = torch.ones(n_embd)
            sd[f"{p}.ln_2.bias"]   = torch.zeros(n_embd)
            sd[f"{p}.mlp.c_fc.weight"]   = torch.zeros(n_embd, 4 * n_embd)
            sd[f"{p}.mlp.c_fc.bias"]     = torch.zeros(4 * n_embd)
            sd[f"{p}.mlp.c_proj.weight"] = torch.zeros(4 * n_embd, n_embd)
            sd[f"{p}.mlp.c_proj.bias"]   = torch.zeros(n_embd)
        sd["transformer.ln_f.weight"] = torch.ones(n_embd)
        sd["transformer.ln_f.bias"]   = torch.zeros(n_embd)
        return sd

    def test_mapped_keys_match_model_state_dict(self):
        """Mapped keys must exactly match the keys expected by a fresh NeuralNetworkModel."""
        n_layer, n_embd, n_head, vocab_size, block_size = 2, 32, 2, 64, 16

        hf_sd = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)

        hf_cfg = MagicMock()
        hf_cfg.vocab_size = vocab_size
        hf_cfg.n_embd = n_embd
        hf_cfg.n_head = n_head
        hf_cfg.n_layer = n_layer
        hf_cfg.n_positions = block_size
        hf_cfg.resid_pdrop = 0.0
        hf_cfg.embd_pdrop  = 0.0
        hf_cfg.attn_pdrop  = 0.0

        layers_config = Mapper.from_hf_config(hf_cfg)
        model = NeuralNetworkModel("tmp", Mapper(layers_config, {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))

        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer)
        self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()))

    def test_mapped_keys_match_with_safetensors_tied_weights(self):
        """Safetensors deduplicates tied weights, keeping only lm_head.weight."""
        n_layer, n_embd, n_head, vocab_size, block_size = 2, 32, 2, 64, 16

        hf_sd = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)
        wte = hf_sd.pop("transformer.wte.weight")
        hf_sd["lm_head.weight"] = wte

        hf_cfg = MagicMock()
        hf_cfg.vocab_size = vocab_size
        hf_cfg.n_embd = n_embd
        hf_cfg.n_head = n_head
        hf_cfg.n_layer = n_layer
        hf_cfg.n_positions = block_size
        hf_cfg.resid_pdrop = 0.0
        hf_cfg.embd_pdrop  = 0.0
        hf_cfg.attn_pdrop  = 0.0

        layers_config = Mapper.from_hf_config(hf_cfg)
        model = NeuralNetworkModel("tmp", Mapper(layers_config, {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))

        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer)
        self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()))

    def test_mapped_keys_match_without_transformer_prefix(self):
        """Safetensors saved from GPT2Model omit the 'transformer.' prefix."""
        n_layer, n_embd, n_head, vocab_size, block_size = 2, 32, 2, 64, 16

        hf_sd = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)
        unprefixed = {k.replace("transformer.", ""): v for k, v in hf_sd.items()}

        hf_cfg = MagicMock()
        hf_cfg.vocab_size = vocab_size
        hf_cfg.n_embd = n_embd
        hf_cfg.n_head = n_head
        hf_cfg.n_layer = n_layer
        hf_cfg.n_positions = block_size
        hf_cfg.resid_pdrop = 0.0
        hf_cfg.embd_pdrop  = 0.0
        hf_cfg.attn_pdrop  = 0.0

        layers_config = Mapper.from_hf_config(hf_cfg)
        model = NeuralNetworkModel("tmp", Mapper(layers_config, {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))

        mapped = Mapper.map_hf_state_dict_to_custom(unprefixed, n_layer)
        self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()))

    @patch("neural_net_model.NeuralNetworkModel.serialize")
    @patch("neural_net_model.load_safetensors")
    @patch("neural_net_model.snapshot_download", return_value="/tmp/model")
    @patch("neural_net_model.AutoConfig")
    def test_from_huggingface_returns_model(self, MockConfig, mock_dl, mock_load, mock_serialize):
        n_layer, n_embd, vocab_size, block_size = 1, 32, 64, 16
        hf_cfg = self._make_hf_config(n_layer=n_layer, n_embd=n_embd,
                                       vocab_size=vocab_size, n_positions=block_size)
        MockConfig.from_pretrained.return_value = hf_cfg
        mock_load.return_value = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)

        model = NeuralNetworkModel.from_huggingface("my-gpt2", "gpt2")

        self.assertIsInstance(model, NeuralNetworkModel)
        self.assertEqual(model.model_id, "my-gpt2")
        mock_serialize.assert_called_once()

    @patch("neural_net_model.NeuralNetworkModel.serialize")
    @patch("neural_net_model.load_safetensors")
    @patch("neural_net_model.snapshot_download", return_value="/tmp/model")
    @patch("neural_net_model.AutoConfig")
    def test_from_huggingface_keeps_float32_for_fp32_checkpoints(self, MockConfig, mock_dl, mock_load, mock_serialize):
        """GPT-2 checkpoints are float32; truncating to bf16 degrades generation."""
        n_layer, n_embd, vocab_size, block_size = 1, 32, 64, 16
        hf_cfg = self._make_hf_config(n_layer=n_layer, n_embd=n_embd,
                                       vocab_size=vocab_size, n_positions=block_size)
        MockConfig.from_pretrained.return_value = hf_cfg
        mock_load.return_value = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)

        model = NeuralNetworkModel.from_huggingface("my-gpt2", "gpt2")

        for p in model.parameters():
            self.assertEqual(p.dtype, torch.float32, f"Expected float32 but got {p.dtype}")

    @patch("neural_net_model.NeuralNetworkModel.serialize")
    @patch("neural_net_model.load_safetensors")
    @patch("neural_net_model.snapshot_download", return_value="/tmp/model")
    @patch("neural_net_model.AutoConfig")
    def test_from_huggingface_keeps_bfloat16_for_bf16_checkpoints(self, MockConfig, mock_dl, mock_load, mock_serialize):
        """Checkpoints with native bf16 dtype (Gemma) stay bf16 to halve memory."""
        n_layer, n_embd, vocab_size, block_size = 1, 32, 64, 16
        hf_cfg = self._make_hf_config(n_layer=n_layer, n_embd=n_embd,
                                       vocab_size=vocab_size, n_positions=block_size)
        hf_cfg.dtype = torch.bfloat16
        MockConfig.from_pretrained.return_value = hf_cfg
        mock_load.return_value = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)

        model = NeuralNetworkModel.from_huggingface("my-gpt2", "gpt2")

        for p in model.parameters():
            self.assertEqual(p.dtype, torch.bfloat16, f"Expected bfloat16 but got {p.dtype}")

    @patch("neural_net_model.NeuralNetworkModel.serialize")
    @patch("neural_net_model.load_safetensors")
    @patch("neural_net_model.snapshot_download", return_value="/tmp/model")
    @patch("neural_net_model.AutoConfig")
    def test_from_huggingface_status_code(self, MockConfig, mock_dl, mock_load, mock_serialize):
        n_layer, n_embd, vocab_size, block_size = 1, 32, 64, 16
        hf_cfg = self._make_hf_config(n_layer=n_layer, n_embd=n_embd,
                                       vocab_size=vocab_size, n_positions=block_size)
        MockConfig.from_pretrained.return_value = hf_cfg
        mock_load.return_value = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)

        model = NeuralNetworkModel.from_huggingface("my-gpt2", "gpt2")

        self.assertEqual(model.status["code"], "Imported")
        self.assertIn("gpt2", model.status["message"])

    @patch("neural_net_model.NeuralNetworkModel.serialize")
    @patch("neural_net_model.load_safetensors")
    @patch("neural_net_model.snapshot_download", return_value="/tmp/model")
    @patch("neural_net_model.AutoConfig")
    def test_from_huggingface_passes_revision(self, MockConfig, mock_dl, mock_load, mock_serialize):
        n_layer, n_embd, vocab_size, block_size = 1, 32, 64, 16
        hf_cfg = self._make_hf_config(n_layer=n_layer, n_embd=n_embd,
                                       vocab_size=vocab_size, n_positions=block_size)
        MockConfig.from_pretrained.return_value = hf_cfg
        mock_load.return_value = self._make_hf_sd(n_layer, n_embd, vocab_size, block_size)

        NeuralNetworkModel.from_huggingface("m", "gpt2", revision="main")

        MockConfig.from_pretrained.assert_called_once_with("gpt2", revision="main")
        mock_dl.assert_called_once()
        self.assertEqual(mock_dl.call_args[1].get("revision"), "main")


    # ---- Gemma import tests ----

    def _make_gemma_hf_config(self, model_type="gemma3", n_layer=1,
                               hidden_size=32, num_attention_heads=4,
                               num_key_value_heads=2, head_dim=8,
                               vocab_size=64, intermediate_size=64):
        cfg = MagicMock(spec=[])
        cfg.model_type = model_type
        cfg.vocab_size = vocab_size
        cfg.hidden_size = hidden_size
        cfg.num_attention_heads = num_attention_heads
        cfg.num_key_value_heads = num_key_value_heads
        cfg.head_dim = head_dim
        cfg.num_hidden_layers = n_layer
        cfg.intermediate_size = intermediate_size
        cfg.rms_norm_eps = 1e-6
        cfg.rope_theta = 10000.0
        cfg.attention_dropout = 0.0
        cfg.hidden_activation = "gelu_pytorch_tanh"
        cfg.to_dict = lambda: {}
        return cfg

    def _make_gemma_hf_sd(self, model_type="gemma3", n_layer=1, n_embd=32,
                           n_head=4, n_kv_heads=2, head_dim=8,
                           vocab_size=64, intermediate_size=64,
                           multimodal=False):
        sd = {}
        pfx = "model.language_model" if multimodal else "model"
        sd[f"{pfx}.embed_tokens.weight"] = torch.zeros(vocab_size, n_embd)
        has_post_norms = model_type != "gemma"
        for i in range(n_layer):
            p = f"{pfx}.layers.{i}"
            sd[f"{p}.input_layernorm.weight"] = torch.zeros(n_embd)
            sd[f"{p}.self_attn.q_proj.weight"] = torch.zeros(n_head * head_dim, n_embd)
            sd[f"{p}.self_attn.k_proj.weight"] = torch.zeros(n_kv_heads * head_dim, n_embd)
            sd[f"{p}.self_attn.v_proj.weight"] = torch.zeros(n_kv_heads * head_dim, n_embd)
            sd[f"{p}.self_attn.o_proj.weight"] = torch.zeros(n_embd, n_head * head_dim)
            if model_type in ("gemma2", "gemma4", "gemma4_text"):
                sd[f"{p}.self_attn.q_norm.weight"] = torch.zeros(head_dim)
                sd[f"{p}.self_attn.k_norm.weight"] = torch.zeros(head_dim)
            if has_post_norms:
                sd[f"{p}.post_attention_layernorm.weight"] = torch.zeros(n_embd)
                sd[f"{p}.pre_feedforward_layernorm.weight"] = torch.zeros(n_embd)
                sd[f"{p}.post_feedforward_layernorm.weight"] = torch.zeros(n_embd)
            else:
                sd[f"{p}.post_attention_layernorm.weight"] = torch.zeros(n_embd)
            sd[f"{p}.mlp.gate_proj.weight"] = torch.zeros(intermediate_size, n_embd)
            sd[f"{p}.mlp.up_proj.weight"] = torch.zeros(intermediate_size, n_embd)
            sd[f"{p}.mlp.down_proj.weight"] = torch.zeros(n_embd, intermediate_size)
            if model_type in ("gemma4", "gemma4_text"):
                sd[f"{p}.layer_scalar"] = torch.ones(1)
        sd[f"{pfx}.norm.weight"] = torch.zeros(n_embd)
        return sd

    def test_gemma_mapped_keys_match_model_state_dict(self):
        """Mapped Gemma keys must exactly match a fresh model's state dict."""
        for model_type in ("gemma", "gemma3", "gemma4"):
            n_layer, n_embd, n_head, n_kv_heads, head_dim = 2, 32, 4, 2, 8
            vocab_size, intermediate_size = 64, 64
            hf_sd = self._make_gemma_hf_sd(model_type=model_type, n_layer=n_layer,
                                             n_embd=n_embd, n_head=n_head,
                                             n_kv_heads=n_kv_heads, head_dim=head_dim,
                                             vocab_size=vocab_size,
                                             intermediate_size=intermediate_size)
            hf_cfg = self._make_gemma_hf_config(model_type=model_type, n_layer=n_layer,
                                                  hidden_size=n_embd,
                                                  num_attention_heads=n_head,
                                                  num_key_value_heads=n_kv_heads,
                                                  head_dim=head_dim,
                                                  vocab_size=vocab_size,
                                                  intermediate_size=intermediate_size)
            layers_config = Mapper.from_hf_config(hf_cfg)
            model = NeuralNetworkModel("tmp",
                        Mapper(layers_config, {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))
            mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
            self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()),
                             f"Key mismatch for {model_type}")

    def test_gemma4_norm_weights_not_offset(self):
        """Gemma 4 uses direct RMSNorm convention — norm weights must NOT be offset by +1."""
        n_layer, n_embd, n_head, n_kv_heads, head_dim = 1, 32, 4, 2, 8
        vocab_size, intermediate_size = 64, 64
        hf_sd = self._make_gemma_hf_sd(model_type="gemma4", n_layer=n_layer,
                                         n_embd=n_embd, n_head=n_head,
                                         n_kv_heads=n_kv_heads, head_dim=head_dim,
                                         vocab_size=vocab_size,
                                         intermediate_size=intermediate_size)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma4", n_layer=n_layer,
                                              hidden_size=n_embd,
                                              num_attention_heads=n_head,
                                              num_key_value_heads=n_kv_heads,
                                              head_dim=head_dim,
                                              vocab_size=vocab_size,
                                              intermediate_size=intermediate_size)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
        for key, value in mapped.items():
            if "norm" in key or "layernorm" in key:
                self.assertTrue(torch.equal(value, torch.zeros_like(value)),
                                f"Gemma 4 norm weight {key} should not be offset by +1")

    def test_gemma3_norm_weights_offset_by_one(self):
        """Gemma 3 uses centered RMSNorm convention — norm weights must be offset by +1."""
        n_layer, n_embd, n_head, n_kv_heads, head_dim = 1, 32, 4, 2, 8
        vocab_size, intermediate_size = 64, 64
        hf_sd = self._make_gemma_hf_sd(model_type="gemma3", n_layer=n_layer,
                                         n_embd=n_embd, n_head=n_head,
                                         n_kv_heads=n_kv_heads, head_dim=head_dim,
                                         vocab_size=vocab_size,
                                         intermediate_size=intermediate_size)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma3", n_layer=n_layer,
                                              hidden_size=n_embd,
                                              num_attention_heads=n_head,
                                              num_key_value_heads=n_kv_heads,
                                              head_dim=head_dim,
                                              vocab_size=vocab_size,
                                              intermediate_size=intermediate_size)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
        for key, value in mapped.items():
            if "norm" in key or "layernorm" in key:
                self.assertTrue(torch.equal(value, torch.ones_like(value)),
                                f"Gemma 3 norm weight {key} should be offset by +1")

    def test_partial_rotary_inv_freq_zero_padded(self):
        """Partial RoPE zero-pads inv_freq to head_dim//2 and divides by head_dim."""
        head_dim, rotary_dim, rope_theta = 512, 128, 1000000.0
        attn = nnl.CausalSelfAttention(
            num_heads=8, head_dim=head_dim, rope_theta=rope_theta,
            rotary_dim=rotary_dim)
        self.assertEqual(attn.inv_freq.shape[0], head_dim // 2)
        rope_angles = rotary_dim // 2
        self.assertTrue(torch.all(attn.inv_freq[:rope_angles] != 0))
        self.assertTrue(torch.all(attn.inv_freq[rope_angles:] == 0))
        expected = 1.0 / (rope_theta ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32) / head_dim))
        self.assertTrue(torch.allclose(attn.inv_freq[:rope_angles], expected))

    def test_full_rotary_inv_freq_not_padded(self):
        """Without partial RoPE, inv_freq has head_dim//2 entries, no zero-padding."""
        head_dim, rope_theta = 256, 10000.0
        attn = nnl.CausalSelfAttention(
            num_heads=8, head_dim=head_dim, rope_theta=rope_theta)
        self.assertEqual(attn.inv_freq.shape[0], head_dim // 2)
        self.assertTrue(torch.all(attn.inv_freq != 0))

    @patch("neural_net_model.NeuralNetworkModel.serialize")
    @patch("neural_net_model.load_safetensors")
    @patch("neural_net_model.snapshot_download", return_value="/tmp/model")
    @patch("neural_net_model.AutoConfig")
    def test_from_huggingface_gemma_returns_model(self, MockConfig, mock_dl, mock_load, mock_serialize):
        n_layer, n_embd, n_head, n_kv_heads, head_dim = 1, 32, 4, 2, 8
        vocab_size, intermediate_size = 64, 64
        hf_cfg = self._make_gemma_hf_config(model_type="gemma3", n_layer=n_layer,
                                              hidden_size=n_embd, num_attention_heads=n_head,
                                              num_key_value_heads=n_kv_heads, head_dim=head_dim,
                                              vocab_size=vocab_size,
                                              intermediate_size=intermediate_size)
        MockConfig.from_pretrained.return_value = hf_cfg
        mock_load.return_value = self._make_gemma_hf_sd(
            model_type="gemma3", n_layer=n_layer,
            n_embd=n_embd, n_head=n_head,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size)

        model = NeuralNetworkModel.from_huggingface("my-gemma", "google/gemma-3-1b")

        self.assertIsInstance(model, NeuralNetworkModel)
        self.assertEqual(model.model_id, "my-gemma")
        self.assertEqual(model.status["code"], "Imported")
        self.assertIn("google/gemma-3-1b", model.status["message"])
        mock_serialize.assert_called_once()


    @patch('neural_net_model.dist.destroy_process_group')
    @patch('neural_net_model.dist.init_process_group')
    @patch('ddp.reconfig_logging')
    @patch('ddp.is_ddp', return_value=True)
    @patch('ddp.master_proc', return_value=True)
    def test_train_model_on_device_mps_uses_gloo(self, mock_master, mock_is_ddp,
                                                  mock_reconfig, mock_init_pg,
                                                  mock_destroy_pg):
        with patch.object(NeuralNetworkModel, 'deserialize') as mock_deser, \
             patch.object(NeuralNetworkModel, 'train_model') as mock_train:
            mock_model = MagicMock(spec=NeuralNetworkModel)
            mock_model.optimizer = MagicMock()
            mock_model.optimizer.state = {}
            mock_deser.return_value = mock_model
            NeuralNetworkModel.train_model_on_device(
                "test_model", "mps", "test_dataset", 0, 1, 1, 1, 1)
            mock_init_pg.assert_called_once_with(backend='gloo')
            # MPS stays on MPS — device is never changed, training runs on MPS
            mock_model.to.assert_called_once_with('mps')

    @patch('ddp.is_ddp', return_value=False)
    @patch('ddp.master_proc', return_value=True)
    def test_train_model_on_device_moves_optimizer_state_to_device(self, mock_master, mock_is_ddp):
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-optim-state", Mapper(layers, {"adam": {"lr": .01}}))

        # Simulate optimizer state tensors remaining on CPU after loading from disk (e.g. on subsequent MPS runs)
        mock_tensor = MagicMock(spec=torch.Tensor)
        mock_tensor.to.return_value = mock_tensor
        param = next(iter(model.parameters()))
        model.optimizer.state[param] = {'exp_avg': mock_tensor, 'exp_avg_sq': mock_tensor}

        with patch.object(NeuralNetworkModel, 'deserialize', return_value=model), \
             patch.object(NeuralNetworkModel, 'train_model'):
            NeuralNetworkModel.train_model_on_device("test-optim-state", 'cpu', "mock_ds", 0, 1, 1, 2, 1)

        # Verify all optimizer state tensors were moved to the actual device of model parameters
        mock_tensor.to.assert_called_with(next(model.parameters()).device)

    @patch('ddp.use_ddp', return_value=False)
    @patch('ddp.is_ddp', return_value=True)
    @patch('ddp.master_proc', return_value=True)
    def test_train_model_skips_ddp_wrap_for_mps_single_process(self, mock_master, mock_is_ddp, mock_use_ddp):
        layers = [{"embedding": {"num_embeddings": 8, "embedding_dim": 2}},
                  {"tanh": {}},
                  {"linear": {"in_features": 2, "out_features": 8}},
                  {"softmaxlast": {"dim": -1}}]
        model = NeuralNetworkModel("test-mps-skip", Mapper(layers, {"sgd": {"lr": .01}}))

        with patch("neural_net_model.Loader") as MockLoader, \
             patch.object(NeuralNetworkModel, 'serialize'), \
             patch('neural_net_model.nn.parallel.DistributedDataParallel') as mock_ddp:
            mock_loader = MagicMock()
            MockLoader.return_value = mock_loader
            mock_loader.next_batch.return_value = tuple(
                np.array(l, dtype=np.int32) for l in [[1, 2], [2, 3]])
            model.train_model("mock_ds", 1, 1, 1, 2, 1)
            mock_ddp.assert_not_called()


if __name__ == '__main__':
    unittest.main()
