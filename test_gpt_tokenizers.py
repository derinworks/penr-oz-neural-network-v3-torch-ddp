import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from gpt_tokenizers import Tokenizer


class TestTiktokenTokenizer(unittest.TestCase):

    def _make_mock_enc(self):
        mock_enc = MagicMock()
        mock_enc.eot_token = 50256
        mock_enc.encode_ordinary.side_effect = lambda text: (
            [15496, 995] if text == "Hello world"
            else [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13] if text == "The quick brown fox jumps over the lazy dog."
            else []
        )
        mock_enc.decode.side_effect = lambda tokens: (
            "Hello world" if tokens == [15496, 995, 50256]
            else "The quick brown fox jumps over the lazy dog." if tokens == [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 50256]
            else ""
        )
        return mock_enc

    @patch("gpt_tokenizers.tiktoken")
    def test_tokenize(self, mock_tiktoken):
        mock_tiktoken.get_encoding.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("tiktoken/gpt2")
        tokens = tokenizer.tokenize("Hello world")

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        mock_tiktoken.get_encoding.assert_called_once_with("gpt2")

    @patch("gpt_tokenizers.tiktoken")
    def test_decode(self, mock_tiktoken):
        mock_tiktoken.get_encoding.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("tiktoken/gpt2")
        original_text = "Hello world"
        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)

        self.assertIsInstance(decoded_text, str)
        self.assertIn("Hello world", decoded_text)

    @patch("gpt_tokenizers.tiktoken")
    def test_tokenize_decode_roundtrip(self, mock_tiktoken):
        mock_tiktoken.get_encoding.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("tiktoken/gpt2")
        original_text = "The quick brown fox jumps over the lazy dog."

        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)

        self.assertIn(original_text, decoded_text)

    @patch("gpt_tokenizers.tiktoken")
    def test_tokenize_empty_string(self, mock_tiktoken):
        mock_tiktoken.get_encoding.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("tiktoken/gpt2")
        tokens = tokenizer.tokenize("")

        # Even empty string should have EOT token
        self.assertEqual(len(tokens), 1)


class TestAutoTokenizer(unittest.TestCase):

    def _make_mock_enc(self):
        mock_enc = MagicMock()
        mock_enc.eos_token_id = 50256
        mock_enc.encode.side_effect = lambda text, add_special_tokens=False: (
            [15496, 995] if text == "Hello world"
            else [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13] if text == "The quick brown fox jumps over the lazy dog."
            else []
        )
        mock_enc.decode.side_effect = lambda tokens: (
            "Hello world" if tokens == [15496, 995, 50256]
            else "The quick brown fox jumps over the lazy dog." if tokens == [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 50256]
            else ""
        )
        return mock_enc

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_tokenize(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("gpt2")
        tokens = tokenizer.tokenize("Hello world")

        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("gpt2")

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_decode(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("gpt2")
        original_text = "Hello world"
        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)

        self.assertIsInstance(decoded_text, str)
        self.assertIn("Hello world", decoded_text)

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_tokenize_decode_roundtrip(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("gpt2")
        original_text = "The quick brown fox jumps over the lazy dog."

        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)

        self.assertIn(original_text, decoded_text)

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_tokenize_empty_string(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("gpt2")
        tokens = tokenizer.tokenize("")

        # Even empty string should have EOS token
        self.assertEqual(len(tokens), 1)


class TestProcessorTokenizer(unittest.TestCase):
    """Multimodal models matching _PROCESSOR_PATTERNS load the tokenizer class
    from processor_config.json to avoid pulling in PIL for image processing."""

    def _make_processor_config(self, tokenizer_class="GemmaTokenizer"):
        """Write a temp processor_config.json and return its path."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"processor_class": "Gemma4Processor",
                    "tokenizer_class": tokenizer_class}, f)
        f.close()
        return f.name

    def _make_mock_tokenizer(self):
        mock_enc = MagicMock()
        mock_enc.eos_token_id = 1
        mock_enc.encode.side_effect = lambda text, add_special_tokens=False: (
            [4103, 2134] if text == "Hello world" else []
        )
        mock_enc.decode.side_effect = lambda tokens: (
            "Hello world" if tokens == [4103, 2134, 1] else ""
        )
        return mock_enc

    @patch("gpt_tokenizers.hf_hub_download")
    @patch("gpt_tokenizers.transformers")
    def test_tokenizer_loaded_from_processor_config(self, mock_transformers, mock_download):
        path = self._make_processor_config("GemmaTokenizer")
        self.addCleanup(os.unlink, path)
        mock_download.return_value = path
        mock_tok_cls = MagicMock()
        mock_tok_cls.from_pretrained.return_value = self._make_mock_tokenizer()
        mock_transformers.GemmaTokenizer = mock_tok_cls

        tokenizer = Tokenizer("google/gemma-4-E2B")
        tokens = tokenizer.tokenize("Hello world")

        self.assertEqual(tokens, [4103, 2134, 1])
        mock_download.assert_called_once()
        mock_tok_cls.from_pretrained.assert_called_once()

    @patch("gpt_tokenizers.hf_hub_download")
    @patch("gpt_tokenizers.transformers")
    def test_processor_decode(self, mock_transformers, mock_download):
        path = self._make_processor_config("GemmaTokenizer")
        self.addCleanup(os.unlink, path)
        mock_download.return_value = path
        mock_tok_cls = MagicMock()
        mock_tok_cls.from_pretrained.return_value = self._make_mock_tokenizer()
        mock_transformers.GemmaTokenizer = mock_tok_cls

        tokenizer = Tokenizer("google/gemma-4-E2B")
        decoded = tokenizer.decode([4103, 2134, 1])
        self.assertEqual(decoded, "Hello world")

    @patch("gpt_tokenizers.hf_hub_download")
    @patch("gpt_tokenizers.AutoTokenizer")
    def test_falls_back_to_auto_tokenizer_when_class_missing(self, mock_auto_tokenizer, mock_download):
        """When processor_config has no tokenizer_class, fall back to AutoTokenizer."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"processor_class": "SomeProcessor"}, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        mock_download.return_value = f.name
        mock_enc = MagicMock()
        mock_enc.eos_token_id = 1
        mock_enc.encode.return_value = [100]
        mock_auto_tokenizer.from_pretrained.return_value = mock_enc

        tokenizer = Tokenizer("google/gemma-3-1b")
        tokens = tokenizer.tokenize("test")

        self.assertEqual(tokens, [100, 1])
        mock_auto_tokenizer.from_pretrained.assert_called_once()

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_non_gemma_uses_auto_tokenizer(self, mock_auto_tokenizer):
        """Non-gemma models use AutoTokenizer directly."""
        mock_enc = MagicMock()
        mock_enc.eos_token_id = 50256
        mock_enc.encode.return_value = [15496, 995]
        mock_auto_tokenizer.from_pretrained.return_value = mock_enc

        tokenizer = Tokenizer("gpt2")
        tokens = tokenizer.tokenize("Hello world")

        self.assertEqual(tokens, [15496, 995, 50256])
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("gpt2")


if __name__ == '__main__':
    unittest.main()
