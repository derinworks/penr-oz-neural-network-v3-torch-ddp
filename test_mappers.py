import unittest
from unittest.mock import MagicMock
import torch
import torch.nn as nn
import torch.optim as optim
import neural_net_layers as nnl
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


    def _make_gpt2_config(self, n_layer=2, n_embd=64, n_head=2,
                           n_positions=32, vocab_size=128,
                           resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1):
        cfg = MagicMock()
        cfg.vocab_size = vocab_size
        cfg.n_embd = n_embd
        cfg.n_head = n_head
        cfg.n_layer = n_layer
        cfg.n_positions = n_positions
        cfg.resid_pdrop = resid_pdrop
        cfg.embd_pdrop = embd_pdrop
        cfg.attn_pdrop = attn_pdrop
        return cfg

    def test_from_hf_config_layer_count(self):
        """Total layer list length: 2 base + n_layer residual blocks + 3 final."""
        n_layer = 3
        cfg = self._make_gpt2_config(n_layer=n_layer)
        layers = Mapper.from_hf_config(cfg)
        self.assertEqual(len(layers), 2 + n_layer + 3)

    def test_from_hf_config_embedding_summation(self):
        """First layer is a summation of token and position embeddings."""
        cfg = self._make_gpt2_config()
        layers = Mapper.from_hf_config(cfg)
        self.assertIn("summation", layers[0])
        summation_children = layers[0]["summation"]
        self.assertIn("embedding", summation_children[0])
        self.assertIn("position", summation_children[1])

    def test_from_hf_config_embedding_dims(self):
        """Token and position embeddings use vocab_size / block_size and n_embd."""
        cfg = self._make_gpt2_config(n_embd=128, n_positions=64, vocab_size=256)
        layers = Mapper.from_hf_config(cfg)
        tok_emb = layers[0]["summation"][0]["embedding"]
        pos_emb = layers[0]["summation"][1]["position"]
        self.assertEqual(tok_emb["num_embeddings"], 256)
        self.assertEqual(tok_emb["embedding_dim"], 128)
        self.assertEqual(pos_emb["num_embeddings"], 64)
        self.assertEqual(pos_emb["embedding_dim"], 128)

    def test_from_hf_config_dropout_layer(self):
        """Second layer is dropout with embd_pdrop."""
        cfg = self._make_gpt2_config(embd_pdrop=0.2)
        layers = Mapper.from_hf_config(cfg)
        self.assertIn("dropout", layers[1])
        self.assertAlmostEqual(layers[1]["dropout"]["p"], 0.2)

    def test_from_hf_config_residual_blocks(self):
        """Layers 2..2+n_layer-1 are residual blocks with attn and mlp sequentials."""
        n_layer = 2
        cfg = self._make_gpt2_config(n_layer=n_layer)
        layers = Mapper.from_hf_config(cfg)
        for i in range(n_layer):
            block = layers[2 + i]
            self.assertIn("residual", block)
            residual = block["residual"]
            self.assertEqual(len(residual), 2)
            attn_seq = residual[0]["sequential"]
            mlp_seq  = residual[1]["sequential"]
            # attention sequential: layernorm, linear (qkv), attention, linear (proj), dropout
            self.assertIn("layernorm", attn_seq[0])
            self.assertIn("linear",   attn_seq[1])
            self.assertIn("attention", attn_seq[2])
            self.assertIn("linear",   attn_seq[3])
            self.assertIn("dropout",  attn_seq[4])
            # mlp sequential: layernorm, linear (fc), gelu, linear (proj), dropout
            self.assertIn("layernorm", mlp_seq[0])
            self.assertIn("linear",   mlp_seq[1])
            self.assertIn("gelu",     mlp_seq[2])
            self.assertIn("linear",   mlp_seq[3])
            self.assertIn("dropout",  mlp_seq[4])

    def test_from_hf_config_attention_heads(self):
        """Attention layer uses n_head and attn_pdrop from config."""
        cfg = self._make_gpt2_config(n_head=4, attn_pdrop=0.15)
        layers = Mapper.from_hf_config(cfg)
        attn_cfg = layers[2]["residual"][0]["sequential"][2]["attention"]
        self.assertEqual(attn_cfg["num_heads"], 4)
        self.assertAlmostEqual(attn_cfg["dropout"], 0.15)

    def test_from_hf_config_final_layers(self):
        """Last three layers are layernorm, linear (lm_head), softmaxlast."""
        n_layer = 2
        cfg = self._make_gpt2_config(n_layer=n_layer, n_embd=64, vocab_size=128)
        layers = Mapper.from_hf_config(cfg)
        ln_f   = layers[2 + n_layer]
        lm_hd  = layers[2 + n_layer + 1]
        sfx    = layers[2 + n_layer + 2]
        self.assertIn("layernorm", ln_f)
        self.assertEqual(ln_f["layernorm"]["normalized_shape"], 64)
        self.assertIn("linear", lm_hd)
        self.assertEqual(lm_hd["linear"]["in_features"], 64)
        self.assertEqual(lm_hd["linear"]["out_features"], 128)
        self.assertFalse(lm_hd["linear"]["bias"])
        self.assertIn("softmaxlast", sfx)

    def test_from_hf_config_builds_valid_mapper(self):
        """Layers returned by from_hf_config can be passed to Mapper and built."""
        cfg = self._make_gpt2_config(n_layer=1, n_embd=32, n_head=2,
                                     n_positions=8, vocab_size=64)
        layers = Mapper.from_hf_config(cfg)
        mapper = Mapper(layers, {"adamw": {"lr": 6e-4, "betas": [0.9, 0.95], "eps": 1e-8}})
        nn_layers = mapper.to_layers()
        self.assertEqual(len(nn_layers), len(layers))
        # First layer should be a Summation
        self.assertIsInstance(nn_layers[0], nnl.Summation)
        # Last layer should be SoftmaxOnLast
        self.assertIsInstance(nn_layers[-1], nnl.SoftmaxOnLast)

    def test_from_hf_config_uses_hidden_size_fallback(self):
        """n_embd falls back to hidden_size when n_embd is absent."""
        cfg = MagicMock(spec=[])
        cfg.vocab_size = 64
        cfg.hidden_size = 32
        cfg.num_attention_heads = 2
        cfg.num_hidden_layers = 1
        cfg.max_position_embeddings = 8
        cfg.resid_pdrop = 0.0
        cfg.embd_pdrop = 0.0
        cfg.attn_pdrop = 0.0
        layers = Mapper.from_hf_config(cfg)
        tok_emb = layers[0]["summation"][0]["embedding"]
        self.assertEqual(tok_emb["embedding_dim"], 32)


    def _make_hf_sd(self, n_layer=2, n_embd=32, n_head=2, vocab_size=64, block_size=16):
        """Build a fake HuggingFace GPT-2 state dict with the right shapes."""
        sd = {}
        sd["transformer.wte.weight"] = torch.zeros(vocab_size, n_embd)
        sd["transformer.wpe.weight"] = torch.zeros(block_size, n_embd)
        for i in range(n_layer):
            p = f"transformer.h.{i}"
            sd[f"{p}.ln_1.weight"] = torch.ones(n_embd)
            sd[f"{p}.ln_1.bias"]   = torch.zeros(n_embd)
            # Conv1D weight shape: (in, out)
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
        # Tied weights – no separate lm_head.weight
        return sd

    def test_conv1d_weights_are_transposed(self):
        """Weights from Conv1D layers must be transposed to match nn.Linear (out, in)."""
        n_layer, n_embd = 1, 32
        hf_sd = self._make_hf_sd(n_layer=n_layer, n_embd=n_embd)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer)
        # c_attn: HF shape (n_embd, 3*n_embd) → Linear weight shape (3*n_embd, n_embd)
        self.assertEqual(mapped["layers.2.0.1.weight"].shape, (3 * n_embd, n_embd))
        # c_proj: HF shape (n_embd, n_embd) → Linear weight shape (n_embd, n_embd)
        self.assertEqual(mapped["layers.2.0.3.weight"].shape, (n_embd, n_embd))

    def test_tied_lm_head_uses_wte_weight(self):
        """When lm_head.weight is absent, the token embedding weight is used."""
        n_layer, n_embd, vocab_size = 1, 32, 64
        hf_sd = self._make_hf_sd(n_layer=n_layer, n_embd=n_embd, vocab_size=vocab_size)
        self.assertNotIn("lm_head.weight", hf_sd)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer)
        lm_idx = 2 + n_layer + 1
        self.assertTrue(torch.equal(mapped[f"layers.{lm_idx}.weight"], hf_sd["transformer.wte.weight"]))

    def test_explicit_lm_head_used_when_present(self):
        """When lm_head.weight is present in the HF state dict it is used directly."""
        n_layer, n_embd, vocab_size = 1, 32, 64
        hf_sd = self._make_hf_sd(n_layer=n_layer, n_embd=n_embd, vocab_size=vocab_size)
        lm_head = torch.ones(vocab_size, n_embd) * 99.0
        hf_sd["lm_head.weight"] = lm_head
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer)
        lm_idx = 2 + n_layer + 1
        self.assertTrue(torch.equal(mapped[f"layers.{lm_idx}.weight"], lm_head))


if __name__ == '__main__':
    unittest.main()
