import unittest
import torch.nn as nn
import torch.optim as optim
from mappers import Mapper


class TestMapper(unittest.TestCase):

    def test_to_optimizer_with_betas(self):
        """Test that betas list is converted to tuple for Adam/AdamW optimizers"""
        layers = [{"linear": {"in_features": 3, "out_features": 3}}]
        optimizer_config = {"adamw": {"lr": 0.001, "betas": [0.9, 0.999]}}
        
        mapper = Mapper(layers, optimizer_config)
        model_layers = mapper.to_layers()
        params = model_layers[0].parameters()
        
        optimizer = mapper.to_optimizer(params)
        
        # Verify optimizer type
        self.assertIsInstance(optimizer, optim.AdamW)
        
        # Verify betas was converted from list to tuple
        betas = optimizer.param_groups[0]['betas']
        self.assertIsInstance(betas, tuple)
        self.assertEqual(betas, (0.9, 0.999))

    def test_to_optimizer_adam_with_betas(self):
        """Test that betas conversion works for Adam optimizer too"""
        layers = [{"linear": {"in_features": 3, "out_features": 3}}]
        optimizer_config = {"adam": {"lr": 0.001, "betas": [0.9, 0.95]}}
        
        mapper = Mapper(layers, optimizer_config)
        model_layers = mapper.to_layers()
        params = model_layers[0].parameters()
        
        optimizer = mapper.to_optimizer(params)
        
        # Verify optimizer type
        self.assertIsInstance(optimizer, optim.Adam)
        
        # Verify betas was converted from list to tuple
        betas = optimizer.param_groups[0]['betas']
        self.assertIsInstance(betas, tuple)
        self.assertEqual(betas, (0.9, 0.95))

    def test_to_optimizer_sgd_no_betas(self):
        """Test SGD optimizer that doesn't use betas"""
        layers = [{"linear": {"in_features": 3, "out_features": 3}}]
        optimizer_config = {"sgd": {"lr": 0.1}}
        
        mapper = Mapper(layers, optimizer_config)
        model_layers = mapper.to_layers()
        params = model_layers[0].parameters()
        
        optimizer = mapper.to_optimizer(params)
        
        # Verify optimizer type
        self.assertIsInstance(optimizer, optim.SGD)
        
        # Verify no betas in param groups
        self.assertNotIn('betas', optimizer.param_groups[0])


if __name__ == '__main__':
    unittest.main()
