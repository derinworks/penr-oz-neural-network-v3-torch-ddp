import math
import os
import os.path
import time
import unittest
from unittest.mock import patch, MagicMock, call
from parameterized import parameterized
import numpy as np
import torch
import torch.nn as nn
from neural_net_model import NeuralNetworkModel, SHM_PATH
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
    @unittest.skipUnless(os.path.exists(SHM_PATH), f"Requires {SHM_PATH} (Linux shared memory)")
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

    @unittest.skipUnless(os.path.exists(SHM_PATH), f"Requires {SHM_PATH} (Linux shared memory)")
    def test_delete(self):
        model = NeuralNetworkModel("test", Mapper([{"linear": {"in_features": 9, "out_features": 9}}],
                                                  {"sgd": {}}))
        model.serialize()
        model_path = NeuralNetworkModel.get_model_path(model.model_id)
        model_in_shm_path = os.path.join(SHM_PATH, model_path)

        self.assertTrue(os.path.exists(model_in_shm_path))
        time.sleep(1) # wait a bit for cache to flush to disk
        self.assertTrue(os.path.exists(model_path))

        NeuralNetworkModel.delete("test")
        with self.assertRaises(KeyError):
            NeuralNetworkModel.deserialize("test")

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

    @unittest.skipUnless(os.path.exists(SHM_PATH), f"Requires {SHM_PATH} (Linux shared memory)")
    def test_cache_miss(self):
        model = NeuralNetworkModel("test", Mapper([{"linear": {"in_features": 9, "out_features": 9}}],
                                                  {"sgd": {}}))
        model.serialize()
        model_path = NeuralNetworkModel.get_model_path(model.model_id)
        model_in_shm_path = os.path.join(SHM_PATH, model_path)

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


    @unittest.skipUnless(os.path.exists(SHM_PATH), f"Requires {SHM_PATH} (Linux shared memory)")
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
        (True, torch.bfloat16),
        (False, torch.float16),
    ])
    def test_amp_cuda_training_path(self, bf16_supported, expected_dtype):
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

        scaler_enabled = (expected_dtype == torch.float16)

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
            # Verify GradScaler created with correct enabled flag
            MockScaler.assert_called_once_with('cuda', enabled=scaler_enabled)
            # Verify scaler.scale was called for backward pass
            mock_scaler.scale.assert_called()
            mock_scaled_loss.backward.assert_called()
            # Verify scaler.step and update were called
            mock_scaler.step.assert_called_once_with(model.optimizer)
            mock_scaler.update.assert_called_once()


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


if __name__ == '__main__':
    unittest.main()
