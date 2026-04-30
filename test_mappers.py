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
                           resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
                           activation_function="gelu_new"):
        cfg = MagicMock(spec=[])
        cfg.vocab_size = vocab_size
        cfg.n_embd = n_embd
        cfg.n_head = n_head
        cfg.n_layer = n_layer
        cfg.n_positions = n_positions
        cfg.resid_pdrop = resid_pdrop
        cfg.embd_pdrop = embd_pdrop
        cfg.attn_pdrop = attn_pdrop
        cfg.activation_function = activation_function
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

    def test_from_hf_config_gelu_new_activation(self):
        """gelu_new activation maps to gelu with tanh approximation."""
        cfg = self._make_gpt2_config(activation_function="gelu_new")
        layers = Mapper.from_hf_config(cfg)
        mlp_seq = layers[2]["residual"][1]["sequential"]
        self.assertEqual(mlp_seq[2], {"gelu": {"approximate": "tanh"}})

    def test_from_hf_config_standard_gelu_activation(self):
        """Non-gelu_new activation maps to standard gelu (erf approximation)."""
        cfg = self._make_gpt2_config(activation_function="gelu")
        layers = Mapper.from_hf_config(cfg)
        mlp_seq = layers[2]["residual"][1]["sequential"]
        self.assertEqual(mlp_seq[2], {"gelu": {}})

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


    # ---- Gemma from_hf_config tests ----

    def _make_gemma_config(self, model_type="gemma3", n_layer=2, hidden_size=64,
                            num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, vocab_size=128, intermediate_size=128,
                            rms_norm_eps=1e-6, rope_theta=10000.0,
                            attention_dropout=0.0,
                            hidden_activation="gelu_pytorch_tanh",
                            multimodal=False):
        """Build a mock Gemma config.

        When *multimodal* is True, text attributes are placed on a nested
        ``text_config`` sub-object (mimics Gemma 3/4 multimodal configs).
        """
        text = MagicMock(spec=[])
        text.vocab_size = vocab_size
        text.hidden_size = hidden_size
        text.num_attention_heads = num_attention_heads
        text.num_key_value_heads = num_key_value_heads
        text.head_dim = head_dim
        text.num_hidden_layers = n_layer
        text.intermediate_size = intermediate_size
        text.rms_norm_eps = rms_norm_eps
        text.rope_theta = rope_theta
        text.attention_dropout = attention_dropout
        text.hidden_activation = hidden_activation

        if multimodal:
            cfg = MagicMock(spec=[])
            cfg.model_type = model_type
            cfg.text_config = text
            return cfg
        # Flat config (gemma, gemma2): attributes directly on config
        text.model_type = model_type
        return text

    def test_from_hf_config_gemma_layer_count(self):
        """Gemma: 1 embedding + n_layer blocks + 3 final."""
        n_layer = 3
        cfg = self._make_gemma_config(n_layer=n_layer)
        layers = Mapper.from_hf_config(cfg)
        self.assertEqual(len(layers), 1 + n_layer + 3)

    def test_from_hf_config_gemma_scaled_embedding(self):
        """First layer is a scaled embedding."""
        cfg = self._make_gemma_config(hidden_size=64, vocab_size=256)
        layers = Mapper.from_hf_config(cfg)
        self.assertIn("scaledembedding", layers[0])
        emb = layers[0]["scaledembedding"]
        self.assertEqual(emb["num_embeddings"], 256)
        self.assertEqual(emb["embedding_dim"], 64)
        self.assertAlmostEqual(emb["scale"], 64 ** 0.5)

    def test_from_hf_config_gemma_transformer_blocks(self):
        """Gemma blocks are TransformerBlocks with attn_block and mlp_block."""
        cfg = self._make_gemma_config(n_layer=2)
        layers = Mapper.from_hf_config(cfg)
        for i in range(2):
            block = layers[1 + i]
            self.assertIn("transformerblock", block)
            tb = block["transformerblock"]
            self.assertIn("attn_block", tb)
            self.assertIn("mlp_block", tb)
            attn_seq = tb["attn_block"]["sequential"]
            self.assertIn("rmsnorm", attn_seq[0])
            self.assertIn("linear", attn_seq[1])
            self.assertIn("attention", attn_seq[2])
            self.assertIn("linear", attn_seq[3])
            mlp_seq = tb["mlp_block"]["sequential"]
            self.assertIn("rmsnorm", mlp_seq[0])
            self.assertIn("gatedmlp", mlp_seq[1])

    def test_from_hf_config_gemma3_has_post_norms(self):
        """Gemma 2+ variants include post-attention and post-MLP norms."""
        for model_type in ("gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text"):
            cfg = self._make_gemma_config(model_type=model_type, n_layer=1)
            layers = Mapper.from_hf_config(cfg)
            tb = layers[1]["transformerblock"]
            self.assertIn("post_attn_norm", tb, f"Missing post_attn_norm for {model_type}")
            self.assertIn("post_mlp_norm", tb, f"Missing post_mlp_norm for {model_type}")

    def test_from_hf_config_gemma2_post_norm_on_branch(self):
        """Gemma 2 sets post_norm_on_residual=False (norm on branch before add)."""
        cfg = self._make_gemma_config(model_type="gemma2", n_layer=1)
        layers = Mapper.from_hf_config(cfg)
        tb = layers[1]["transformerblock"]
        self.assertIn("post_attn_norm", tb)
        self.assertFalse(tb["post_norm_on_residual"])

    def test_from_hf_config_gemma3_post_norm_on_residual(self):
        """Gemma 3+ sets post_norm_on_residual=True (norm after residual add)."""
        for model_type in ("gemma3", "gemma3_text", "gemma4", "gemma4_text"):
            cfg = self._make_gemma_config(model_type=model_type, n_layer=1)
            layers = Mapper.from_hf_config(cfg)
            tb = layers[1]["transformerblock"]
            self.assertTrue(tb["post_norm_on_residual"], f"Expected True for {model_type}")

    def test_from_hf_config_gemma1_no_post_norms(self):
        """Gemma 1 does not have post-norms."""
        cfg = self._make_gemma_config(model_type="gemma", n_layer=1)
        layers = Mapper.from_hf_config(cfg)
        tb = layers[1]["transformerblock"]
        self.assertNotIn("post_attn_norm", tb)
        self.assertNotIn("post_mlp_norm", tb)

    def test_from_hf_config_gemma_attention_params(self):
        """Attention layer carries GQA, RoPE, and head_dim parameters."""
        cfg = self._make_gemma_config(num_attention_heads=8, num_key_value_heads=4,
                                       head_dim=16, rope_theta=1e6, attention_dropout=0.1)
        layers = Mapper.from_hf_config(cfg)
        attn_cfg = layers[1]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
        self.assertEqual(attn_cfg["num_heads"], 8)
        self.assertEqual(attn_cfg["num_kv_heads"], 4)
        self.assertAlmostEqual(attn_cfg["rope_theta"], 1e6)
        self.assertAlmostEqual(attn_cfg["dropout"], 0.1)
        self.assertEqual(attn_cfg["head_dim"], 16)

    def test_from_hf_config_gemma_qkv_dim(self):
        """QKV linear out_features matches n_head*head_dim + 2*n_kv_heads*head_dim."""
        cfg = self._make_gemma_config(num_attention_heads=8, num_key_value_heads=4, head_dim=16,
                                       hidden_size=64)
        layers = Mapper.from_hf_config(cfg)
        qkv_linear = layers[1]["transformerblock"]["attn_block"]["sequential"][1]["linear"]
        expected_qkv = 8 * 16 + 2 * 4 * 16  # 128 + 128 = 256
        self.assertEqual(qkv_linear["in_features"], 64)
        self.assertEqual(qkv_linear["out_features"], expected_qkv)

    def test_from_hf_config_gemma_final_layers(self):
        """Last three layers are rmsnorm, linear (lm_head), softmaxlast."""
        n_layer = 2
        cfg = self._make_gemma_config(n_layer=n_layer, hidden_size=64, vocab_size=128)
        layers = Mapper.from_hf_config(cfg)
        self.assertIn("rmsnorm", layers[1 + n_layer])
        self.assertIn("linear", layers[1 + n_layer + 1])
        lm = layers[1 + n_layer + 1]["linear"]
        self.assertEqual(lm["in_features"], 64)
        self.assertEqual(lm["out_features"], 128)
        self.assertFalse(lm["bias"])
        self.assertIn("softmaxlast", layers[1 + n_layer + 2])

    def test_from_hf_config_gemma_builds_valid_mapper(self):
        """Layers returned by from_hf_config can be built by Mapper for Gemma."""
        cfg = self._make_gemma_config(model_type="gemma3", n_layer=1, hidden_size=32,
                                       num_attention_heads=2, num_key_value_heads=2,
                                       head_dim=16, vocab_size=64, intermediate_size=64)
        layers = Mapper.from_hf_config(cfg)
        mapper = Mapper(layers, {"adamw": {"lr": 6e-4, "betas": [0.9, 0.95], "eps": 1e-8}})
        nn_layers = mapper.to_layers()
        self.assertEqual(len(nn_layers), len(layers))
        self.assertIsInstance(nn_layers[0], nnl.ScaledEmbedding)
        self.assertIsInstance(nn_layers[1], nnl.TransformerBlock)
        self.assertIsInstance(nn_layers[-1], nnl.SoftmaxOnLast)

    def test_from_hf_config_gemma4_multimodal_config(self):
        """Gemma 4 multimodal config nests text params in text_config."""
        cfg = self._make_gemma_config(model_type="gemma4", n_layer=2,
                                       hidden_size=64, vocab_size=128,
                                       multimodal=True)
        layers = Mapper.from_hf_config(cfg)
        self.assertEqual(len(layers), 1 + 2 + 3)
        self.assertIn("scaledembedding", layers[0])
        self.assertEqual(layers[0]["scaledembedding"]["num_embeddings"], 128)
        self.assertIn("transformerblock", layers[1])

    def test_from_hf_config_gemma_rope_theta_from_rope_scaling(self):
        """When rope_theta is absent, it is extracted from rope_scaling."""
        cfg = self._make_gemma_config(model_type="gemma3", n_layer=1)
        # Remove direct rope_theta, add rope_scaling instead
        del cfg.rope_theta
        cfg.rope_scaling = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 50000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0},
        }
        layers = Mapper.from_hf_config(cfg)
        attn_cfg = layers[1]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
        self.assertAlmostEqual(attn_cfg["rope_theta"], 50000.0)

    def test_from_hf_config_gemma_per_layer_rope_theta(self):
        """Heterogeneous models: sliding layers use local theta, full use global theta."""
        n_layer = 4
        cfg = self._make_gemma_config(model_type="gemma4", n_layer=n_layer,
                                       multimodal=True)
        # Simulate Gemma 3n-style config: rope_theta = global, rope_local_base_freq = local
        cfg.text_config.rope_theta = 1000000.0
        cfg.text_config.rope_local_base_freq = 10000.0
        cfg.text_config.layer_types = ["sliding_attention", "full_attention",
                                        "sliding_attention", "full_attention"]
        layers = Mapper.from_hf_config(cfg)
        for i, expected in enumerate((10000.0, 1000000.0, 10000.0, 1000000.0)):
            attn_cfg = layers[1 + i]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
            self.assertAlmostEqual(attn_cfg["rope_theta"], expected,
                                   msg=f"Layer {i} rope_theta mismatch")

    def test_from_hf_config_gemma2_plus_has_qk_norm(self):
        """Gemma 2+ attention config includes q_norm and k_norm."""
        for model_type in ("gemma2", "gemma3", "gemma3_text", "gemma4", "gemma4_text"):
            cfg = self._make_gemma_config(model_type=model_type, n_layer=1, head_dim=16)
            layers = Mapper.from_hf_config(cfg)
            attn_cfg = layers[1]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
            self.assertIn("q_norm", attn_cfg, f"Missing q_norm for {model_type}")
            self.assertIn("k_norm", attn_cfg, f"Missing k_norm for {model_type}")
            self.assertEqual(
                attn_cfg["q_norm"]["rmsnorm"]["normalized_shape"], 16,
                f"q_norm shape mismatch for {model_type}")

    def test_from_hf_config_gemma1_no_qk_norm(self):
        """Gemma 1 attention config does not include q_norm/k_norm."""
        cfg = self._make_gemma_config(model_type="gemma", n_layer=1)
        layers = Mapper.from_hf_config(cfg)
        attn_cfg = layers[1]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
        self.assertNotIn("q_norm", attn_cfg)
        self.assertNotIn("k_norm", attn_cfg)

    def test_from_hf_config_gpt2_no_qk_norm(self):
        """GPT-2 attention config does not include q_norm/k_norm."""
        cfg = self._make_gpt2_config(n_layer=1)
        layers = Mapper.from_hf_config(cfg)
        attn_cfg = layers[2]["residual"][0]["sequential"][2]["attention"]
        self.assertNotIn("q_norm", attn_cfg)
        self.assertNotIn("k_norm", attn_cfg)

    def test_from_hf_config_gemma4_kv_shared_layer_idx(self):
        """KV-shared layers in config carry kv_shared_layer_idx pointing to reference layer."""
        n_layer = 10
        num_kv_shared = 4
        layer_types = (["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"]) * 2
        cfg = self._make_gemma_config(model_type="gemma4", n_layer=n_layer, multimodal=True)
        cfg.text_config.layer_types = layer_types
        cfg.text_config.num_kv_shared_layers = num_kv_shared

        layers = Mapper.from_hf_config(cfg)

        # Non-shared layers (0-5) should NOT have kv_shared_layer_idx
        for i in range(6):
            attn_cfg = layers[1 + i]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
            self.assertNotIn("kv_shared_layer_idx", attn_cfg, f"Layer {i} should not be shared")

        # Layer 6 is shared sliding → references layer 5 (last non-shared sliding)
        attn_cfg = layers[7]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
        self.assertEqual(attn_cfg["kv_shared_layer_idx"], 5)

        # Layer 8 is shared full → references layer 3 (last non-shared full before first_kv_shared)
        attn_cfg = layers[9]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
        self.assertEqual(attn_cfg["kv_shared_layer_idx"], 3)

    def test_from_hf_config_no_kv_sharing_without_shared_layers(self):
        """Models without KV-shared layers don't get kv_shared_layer_idx."""
        cfg = self._make_gemma_config(model_type="gemma3", n_layer=4)
        layers = Mapper.from_hf_config(cfg)
        for i in range(4):
            attn_cfg = layers[1 + i]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
            self.assertNotIn("kv_shared_layer_idx", attn_cfg)

    def test_from_hf_config_gemma_per_layer_rope_theta_from_rope_scaling(self):
        """Heterogeneous models: layer-specific rope_theta sourced from rope_scaling dict."""
        n_layer = 3
        cfg = self._make_gemma_config(model_type="gemma4", n_layer=n_layer,
                                       multimodal=True)
        del cfg.text_config.rope_theta
        cfg.text_config.rope_scaling = {
            "sliding_attention": {"rope_type": "default", "rope_theta": 50000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0},
        }
        cfg.text_config.layer_types = ["full_attention", "sliding_attention", "full_attention"]
        layers = Mapper.from_hf_config(cfg)
        for i, expected in enumerate((1000000.0, 50000.0, 1000000.0)):
            attn_cfg = layers[1 + i]["transformerblock"]["attn_block"]["sequential"][2]["attention"]
            self.assertAlmostEqual(attn_cfg["rope_theta"], expected,
                                   msg=f"Layer {i} rope_theta mismatch")


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


    # ---- Gemma state dict mapping tests ----

    def _make_gemma_hf_sd(self, model_type="gemma3", n_layer=2, n_embd=32,
                           n_head=4, n_kv_heads=2, head_dim=8,
                           vocab_size=64, intermediate_size=64,
                           multimodal=False, kv_shared_from=None,
                           layer_types=None, global_head_dim=None,
                           n_global_kv_heads=None, use_double_wide_mlp=False):
        """Build a fake HuggingFace Gemma state dict.

        When *kv_shared_from* is set, layers at index >= kv_shared_from will
        NOT include ``k_proj`` and ``v_proj`` weights, simulating the
        KV-shared layer checkpoint format used by Gemma 4.

        When *layer_types* includes ``"full_attention"`` entries, those layers
        use *global_head_dim* and *n_global_kv_heads* for attention dimensions.

        When *use_double_wide_mlp* is True, KV-shared layers use
        ``intermediate_size * 2``.
        """
        sd = {}
        pfx = "model.language_model" if multimodal else "model"
        sd[f"{pfx}.embed_tokens.weight"] = torch.zeros(vocab_size, n_embd)
        has_post_norms = model_type != "gemma"
        for i in range(n_layer):
            p = f"{pfx}.layers.{i}"
            is_full = (layer_types and i < len(layer_types)
                       and layer_types[i] == "full_attention")
            lhd = (global_head_dim or head_dim) if is_full else head_dim
            lkv = (n_global_kv_heads or n_kv_heads) if is_full else n_kv_heads
            is_shared = kv_shared_from is not None and i >= kv_shared_from
            l_inter = intermediate_size * 2 if (use_double_wide_mlp and is_shared) else intermediate_size

            sd[f"{p}.input_layernorm.weight"] = torch.zeros(n_embd)
            sd[f"{p}.self_attn.q_proj.weight"] = torch.zeros(n_head * lhd, n_embd)
            if not is_shared:
                sd[f"{p}.self_attn.k_proj.weight"] = torch.zeros(lkv * lhd, n_embd)
                sd[f"{p}.self_attn.v_proj.weight"] = torch.zeros(lkv * lhd, n_embd)
            sd[f"{p}.self_attn.o_proj.weight"] = torch.zeros(n_embd, n_head * lhd)
            if has_post_norms:
                sd[f"{p}.self_attn.q_norm.weight"] = torch.zeros(lhd)
                sd[f"{p}.self_attn.k_norm.weight"] = torch.zeros(lhd)
                sd[f"{p}.post_attention_layernorm.weight"] = torch.zeros(n_embd)
                sd[f"{p}.pre_feedforward_layernorm.weight"] = torch.zeros(n_embd)
                sd[f"{p}.post_feedforward_layernorm.weight"] = torch.zeros(n_embd)
            else:
                sd[f"{p}.post_attention_layernorm.weight"] = torch.zeros(n_embd)
            sd[f"{p}.mlp.gate_proj.weight"] = torch.zeros(l_inter, n_embd)
            sd[f"{p}.mlp.up_proj.weight"] = torch.zeros(l_inter, n_embd)
            sd[f"{p}.mlp.down_proj.weight"] = torch.zeros(n_embd, l_inter)
        sd[f"{pfx}.norm.weight"] = torch.zeros(n_embd)
        return sd

    def _make_gemma_hf_config(self, model_type="gemma3", n_layer=2, hidden_size=32,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                               vocab_size=64, intermediate_size=64,
                               layer_types=None, global_head_dim=None,
                               num_global_key_value_heads=None,
                               use_double_wide_mlp=False,
                               num_kv_shared_layers=0):
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
        cfg.layer_types = layer_types
        cfg.global_head_dim = global_head_dim if global_head_dim is not None else head_dim
        cfg.num_global_key_value_heads = num_global_key_value_heads
        cfg.use_double_wide_mlp = use_double_wide_mlp
        cfg.num_kv_shared_layers = num_kv_shared_layers
        return cfg

    def test_gemma_qkv_weights_are_concatenated(self):
        """Separate Q, K, V weights are concatenated into single QKV tensor."""
        n_layer, n_embd, n_head, n_kv_heads, head_dim = 1, 32, 4, 2, 8
        hf_sd = self._make_gemma_hf_sd(n_layer=n_layer, n_embd=n_embd, n_head=n_head,
                                         n_kv_heads=n_kv_heads, head_dim=head_dim)
        hf_cfg = self._make_gemma_hf_config(n_layer=n_layer, hidden_size=n_embd,
                                              num_attention_heads=n_head,
                                              num_key_value_heads=n_kv_heads,
                                              head_dim=head_dim)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
        qkv_dim = n_head * head_dim + 2 * n_kv_heads * head_dim
        self.assertEqual(mapped["layers.1.attn_block.1.weight"].shape, (qkv_dim, n_embd))

    def test_gemma_rmsnorm_offset_applied(self):
        """Gemma RMSNorm weights get +1 applied during mapping."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma", n_layer=1)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma", n_layer=1)
        # Set input_layernorm to known values
        hf_sd["model.layers.0.input_layernorm.weight"] = torch.ones(32) * 0.5
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, 1, hf_cfg)
        expected = torch.ones(32) * 1.5  # 0.5 + 1
        self.assertTrue(torch.allclose(mapped["layers.1.attn_block.0.weight"], expected))

    def test_gemma_tied_lm_head_uses_embed_tokens(self):
        """When lm_head.weight is absent, embed_tokens weight is used."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma", n_layer=1)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma", n_layer=1)
        self.assertNotIn("lm_head.weight", hf_sd)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, 1, hf_cfg)
        lm_idx = 1 + 1 + 1  # 1 emb + 1 block + 1 norm + (idx)
        self.assertTrue(torch.equal(mapped[f"layers.{lm_idx}.weight"],
                                    hf_sd["model.embed_tokens.weight"]))

    def test_detect_hf_n_layer_flat_prefix(self):
        """detect_hf_n_layer counts layers from model.layers.{i} keys."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma3", n_layer=3)
        self.assertEqual(Mapper.detect_hf_n_layer(hf_sd), 3)

    def test_detect_hf_n_layer_multimodal_prefix(self):
        """detect_hf_n_layer counts layers from model.language_model.layers.{i} keys."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma4", n_layer=5, multimodal=True)
        self.assertEqual(Mapper.detect_hf_n_layer(hf_sd), 5)

    def test_detect_hf_n_layer_gpt2(self):
        """detect_hf_n_layer counts layers from transformer.h.{i} keys."""
        hf_sd = self._make_hf_sd(n_layer=4)
        self.assertEqual(Mapper.detect_hf_n_layer(hf_sd), 4)

    def test_detect_hf_n_layer_empty_sd(self):
        """detect_hf_n_layer returns 0 for an empty state dict."""
        self.assertEqual(Mapper.detect_hf_n_layer({}), 0)

    def test_gemma3_post_norms_mapped(self):
        """Gemma 2+ models map post-attention and post-feedforward norms."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma3", n_layer=1)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma3", n_layer=1)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, 1, hf_cfg)
        self.assertIn("layers.1.post_attn_norm.weight", mapped)
        self.assertIn("layers.1.post_mlp_norm.weight", mapped)

    def test_gemma1_no_post_norms_in_mapping(self):
        """Gemma 1 mapping does not produce post-norm keys."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma", n_layer=1)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma", n_layer=1)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, 1, hf_cfg)
        self.assertNotIn("layers.1.post_attn_norm.weight", mapped)
        self.assertNotIn("layers.1.post_mlp_norm.weight", mapped)

    def test_gemma_mapped_keys_match_model_state_dict(self):
        """Mapped keys exactly match the keys expected by a fresh NeuralNetworkModel built from the same config."""
        for model_type in ("gemma", "gemma3"):
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
            from neural_net_model import NeuralNetworkModel
            model = NeuralNetworkModel("tmp", Mapper(layers_config,
                                       {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))
            mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
            self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()),
                             f"Key mismatch for model_type={model_type}")

    def test_from_hf_config_n_layer_override(self):
        """n_layer_override causes from_hf_config to use the specified layer count."""
        cfg = self._make_gemma_config(model_type="gemma3", n_layer=4)
        layers = Mapper.from_hf_config(cfg, n_layer_override=2)
        # 1 embedding + 2 override blocks + 3 final = 6
        self.assertEqual(len(layers), 1 + 2 + 3)

    def test_from_hf_config_gpt2_n_layer_override(self):
        """n_layer_override works for GPT-2 configs too."""
        cfg = self._make_gpt2_config(n_layer=6)
        layers = Mapper.from_hf_config(cfg, n_layer_override=3)
        # 2 base + 3 override blocks + 3 final = 8
        self.assertEqual(len(layers), 2 + 3 + 3)

    def test_gemma4_text_model_type_dispatches_to_gemma(self):
        """gemma4_text model type dispatches to Gemma builder, not GPT-2."""
        cfg = self._make_gemma_config(model_type="gemma4_text", n_layer=2)
        layers = Mapper.from_hf_config(cfg)
        # Gemma builder produces ScaledEmbedding as first layer
        self.assertIn("scaledembedding", layers[0])
        self.assertEqual(len(layers), 1 + 2 + 3)

    def test_gemma4_text_state_dict_flat_prefix(self):
        """gemma4_text uses flat model.layers.{i} prefix (no language_model)."""
        n_layer, n_embd, n_head, n_kv_heads, head_dim = 1, 32, 4, 2, 8
        vocab_size, intermediate_size = 64, 64
        hf_sd = self._make_gemma_hf_sd(model_type="gemma4_text", n_layer=n_layer,
                                         n_embd=n_embd, n_head=n_head,
                                         n_kv_heads=n_kv_heads, head_dim=head_dim,
                                         vocab_size=vocab_size,
                                         intermediate_size=intermediate_size,
                                         multimodal=False)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma4_text", n_layer=n_layer,
                                              hidden_size=n_embd,
                                              num_attention_heads=n_head,
                                              num_key_value_heads=n_kv_heads,
                                              head_dim=head_dim,
                                              vocab_size=vocab_size,
                                              intermediate_size=intermediate_size)
        from neural_net_model import NeuralNetworkModel
        layers_config = Mapper.from_hf_config(hf_cfg)
        model = NeuralNetworkModel("tmp", Mapper(layers_config,
                                   {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
        self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()))

    def test_gemma_qk_norm_weights_mapped_with_offset(self):
        """Gemma 2+ q_norm/k_norm weights get +1 applied during mapping."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma3", n_layer=1)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma3", n_layer=1, head_dim=8)
        hf_sd["model.layers.0.self_attn.q_norm.weight"] = torch.ones(8) * 0.5
        hf_sd["model.layers.0.self_attn.k_norm.weight"] = torch.ones(8) * 0.3
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, 1, hf_cfg)
        self.assertTrue(torch.allclose(mapped["layers.1.attn_block.2.q_norm.weight"],
                                       torch.ones(8) * 1.5))
        self.assertTrue(torch.allclose(mapped["layers.1.attn_block.2.k_norm.weight"],
                                       torch.ones(8) * 1.3))

    def test_gemma1_no_qk_norm_in_mapping(self):
        """Gemma 1 mapping does not produce q_norm/k_norm keys."""
        hf_sd = self._make_gemma_hf_sd(model_type="gemma", n_layer=1)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma", n_layer=1)
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, 1, hf_cfg)
        self.assertNotIn("layers.1.attn_block.2.q_norm.weight", mapped)
        self.assertNotIn("layers.1.attn_block.2.k_norm.weight", mapped)

    def test_gemma4_kv_shared_layers_copy_from_reference(self):
        """KV-shared layers use reference layer k/v when own k_proj/v_proj are absent."""
        # E2B-like: 10 layers, 4 shared, layer_types pattern s,s,s,f,s,s repeating
        n_layer = 10
        n_embd, n_head, n_kv_heads, head_dim = 32, 4, 2, 8
        vocab_size, intermediate_size = 64, 64
        num_kv_shared = 4
        layer_types = (["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"]) * 2
        first_kv_shared = n_layer - num_kv_shared  # 6

        # State dict with shared layers missing k_proj/v_proj
        hf_sd = self._make_gemma_hf_sd(
            model_type="gemma4", n_layer=n_layer, n_embd=n_embd,
            n_head=n_head, n_kv_heads=n_kv_heads, head_dim=head_dim,
            vocab_size=vocab_size, intermediate_size=intermediate_size,
            multimodal=True, kv_shared_from=first_kv_shared)
        # Set reference layers to recognizable values
        ref_k = torch.ones(n_kv_heads * head_dim, n_embd) * 42.0
        ref_v = torch.ones(n_kv_heads * head_dim, n_embd) * 99.0
        # Layer 5 is the last non-shared sliding layer (reference for shared sliding)
        hf_sd["model.language_model.layers.5.self_attn.k_proj.weight"] = ref_k.clone()
        hf_sd["model.language_model.layers.5.self_attn.v_proj.weight"] = ref_v.clone()

        # Config with kv sharing info
        cfg = MagicMock(spec=[])
        cfg.model_type = "gemma4"
        text = MagicMock(spec=[])
        text.num_kv_shared_layers = num_kv_shared
        text.layer_types = layer_types
        cfg.text_config = text

        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, cfg)

        # Layer 6 is first shared sliding → references layer 5
        qkv_dim = n_head * head_dim + 2 * n_kv_heads * head_dim
        qkv = mapped["layers.7.attn_block.1.weight"]  # block_idx = 1 + 6
        self.assertEqual(qkv.shape[0], qkv_dim)
        # k portion should be the reference value
        k_start = n_head * head_dim
        k_end = k_start + n_kv_heads * head_dim
        self.assertTrue(torch.allclose(qkv[k_start:k_end], ref_k))
        # v portion should be the reference value
        self.assertTrue(torch.allclose(qkv[k_end:], ref_v))

    def test_gemma4_kv_shared_layers_override_present_keys(self):
        """KV-shared layers use reference layer k/v even when own keys are present."""
        # Simulates from_pretrained filling shared-layer k/v with random init
        n_layer = 10
        n_embd, n_head, n_kv_heads, head_dim = 32, 4, 2, 8
        vocab_size, intermediate_size = 64, 64
        num_kv_shared = 4
        layer_types = (["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"]) * 2
        first_kv_shared = n_layer - num_kv_shared  # 6

        # Build state dict WITH k_proj/v_proj for shared layers (random junk)
        hf_sd = self._make_gemma_hf_sd(
            model_type="gemma4", n_layer=n_layer, n_embd=n_embd,
            n_head=n_head, n_kv_heads=n_kv_heads, head_dim=head_dim,
            vocab_size=vocab_size, intermediate_size=intermediate_size,
            multimodal=True, kv_shared_from=None)  # all keys present
        # Set recognizable values on reference layer (layer 5, last non-shared sliding)
        ref_k = torch.ones(n_kv_heads * head_dim, n_embd) * 42.0
        ref_v = torch.ones(n_kv_heads * head_dim, n_embd) * 99.0
        hf_sd["model.language_model.layers.5.self_attn.k_proj.weight"] = ref_k.clone()
        hf_sd["model.language_model.layers.5.self_attn.v_proj.weight"] = ref_v.clone()
        # Set shared layer 6 to different (wrong) values simulating random init
        hf_sd["model.language_model.layers.6.self_attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, n_embd)
        hf_sd["model.language_model.layers.6.self_attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, n_embd)

        cfg = MagicMock(spec=[])
        cfg.model_type = "gemma4"
        text = MagicMock(spec=[])
        text.num_kv_shared_layers = num_kv_shared
        text.layer_types = layer_types
        cfg.text_config = text

        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, cfg)

        # Layer 6 is first shared sliding → always gets reference layer 5's k/v
        qkv = mapped["layers.7.attn_block.1.weight"]  # block_idx = 1 + 6
        k_start = n_head * head_dim
        k_end = k_start + n_kv_heads * head_dim
        self.assertTrue(torch.allclose(qkv[k_start:k_end], ref_k))
        self.assertTrue(torch.allclose(qkv[k_end:], ref_v))

    def test_gemma4_heterogeneous_layers_e2b_like(self):
        """E2B-like model: full attn uses global_head_dim, shared layers use double-wide MLP."""
        n_layer = 10
        n_embd, n_head, n_kv_heads = 32, 4, 2
        head_dim, global_head_dim = 8, 16
        vocab_size, intermediate_size = 64, 64
        num_kv_shared = 4
        layer_types = (["sliding_attention"] * 3 + ["full_attention"] + ["sliding_attention"]) * 2

        hf_sd = self._make_gemma_hf_sd(
            model_type="gemma4", n_layer=n_layer, n_embd=n_embd,
            n_head=n_head, n_kv_heads=n_kv_heads, head_dim=head_dim,
            vocab_size=vocab_size, intermediate_size=intermediate_size,
            multimodal=True, kv_shared_from=n_layer - num_kv_shared,
            layer_types=layer_types, global_head_dim=global_head_dim,
            use_double_wide_mlp=True)
        hf_cfg = self._make_gemma_hf_config(
            model_type="gemma4", n_layer=n_layer, hidden_size=n_embd,
            num_attention_heads=n_head, num_key_value_heads=n_kv_heads,
            head_dim=head_dim, vocab_size=vocab_size,
            intermediate_size=intermediate_size, layer_types=layer_types,
            global_head_dim=global_head_dim, use_double_wide_mlp=True,
            num_kv_shared_layers=num_kv_shared)

        from neural_net_model import NeuralNetworkModel
        layers_config = Mapper.from_hf_config(hf_cfg)
        model = NeuralNetworkModel("tmp", Mapper(layers_config,
                                   {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
        self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()))

        # Verify sliding layer dimensions (layer 0)
        sliding_qkv = n_head * head_dim + 2 * n_kv_heads * head_dim
        self.assertEqual(mapped["layers.1.attn_block.1.weight"].shape[0], sliding_qkv)
        self.assertEqual(mapped["layers.1.mlp_block.1.gate_proj.weight"].shape[0], intermediate_size)

        # Verify full attention layer dimensions (layer 3)
        full_qkv = n_head * global_head_dim + 2 * n_kv_heads * global_head_dim
        self.assertEqual(mapped["layers.4.attn_block.1.weight"].shape[0], full_qkv)
        self.assertEqual(mapped["layers.4.attn_block.3.weight"].shape[1], n_head * global_head_dim)

        # Verify double-wide MLP on shared layer (layer 6, block_idx=7)
        self.assertEqual(mapped["layers.7.mlp_block.1.gate_proj.weight"].shape[0], intermediate_size * 2)

    def test_gemma4_multimodal_state_dict_prefix(self):
        """Gemma 4 multimodal state dict uses model.language_model. prefix."""
        n_layer, n_embd, n_head, n_kv_heads, head_dim = 1, 32, 4, 2, 8
        vocab_size, intermediate_size = 64, 64
        hf_sd = self._make_gemma_hf_sd(model_type="gemma4", n_layer=n_layer,
                                         n_embd=n_embd, n_head=n_head,
                                         n_kv_heads=n_kv_heads, head_dim=head_dim,
                                         vocab_size=vocab_size,
                                         intermediate_size=intermediate_size,
                                         multimodal=True)
        hf_cfg = self._make_gemma_hf_config(model_type="gemma4", n_layer=n_layer,
                                              hidden_size=n_embd,
                                              num_attention_heads=n_head,
                                              num_key_value_heads=n_kv_heads,
                                              head_dim=head_dim,
                                              vocab_size=vocab_size,
                                              intermediate_size=intermediate_size)
        from neural_net_model import NeuralNetworkModel
        layers_config = Mapper.from_hf_config(hf_cfg)
        model = NeuralNetworkModel("tmp", Mapper(layers_config,
                                   {"adamw": {"lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8}}))
        mapped = Mapper.map_hf_state_dict_to_custom(hf_sd, n_layer, hf_cfg)
        self.assertEqual(set(mapped.keys()), set(model.state_dict().keys()))


if __name__ == '__main__':
    unittest.main()
