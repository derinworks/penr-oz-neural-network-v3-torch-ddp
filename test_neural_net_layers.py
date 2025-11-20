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
        (nnl.PositionEmbedding, dict(num_embeddings=27, embedding_dim=4)),
        (nnl.Summation, [nn.Embedding(27, 4),
                         nnl.PositionEmbedding(8, 4)]),
        (nnl.ResidualConnection, [nn.LayerNorm(4), nn.Linear(4, 8)]),
        (nnl.SoftmaxOnLast, dict(dim=-1)),
    ])
    def test_layer_init(self, layer_class: type, layer_args: dict | list):
        layer = layer_class(**layer_args) if isinstance(layer_args, dict) else layer_class(*layer_args)

        self.assertIsInstance(layer, nn.Module)

    @parameterized.expand([
        (nnl.CausalSelfAttention(2),
         torch.randn(5, 8, 12), (5, 8, 4)),
        (nnl.CausalSelfAttention(3, 0.2),
         torch.randn(5, 5, 45), (5, 5, 15)),
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

if __name__ == '__main__':
    unittest.main()
