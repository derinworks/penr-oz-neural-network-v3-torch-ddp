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
    """AutoProcessor is used for multimodal models matching _PROCESSOR_PATTERNS."""

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

    @patch("gpt_tokenizers.AutoProcessor")
    def test_processor_tokenizer_used(self, mock_auto_processor):
        mock_tok = self._make_mock_tokenizer()
        mock_proc = MagicMock()
        mock_proc.tokenizer = mock_tok
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-4-E2B")
        tokens = tokenizer.tokenize("Hello world")

        self.assertEqual(tokens, [4103, 2134, 1])
        mock_auto_processor.from_pretrained.assert_called_once_with("google/gemma-4-E2B")

    @patch("gpt_tokenizers.AutoProcessor")
    def test_processor_decode(self, mock_auto_processor):
        mock_tok = self._make_mock_tokenizer()
        mock_proc = MagicMock()
        mock_proc.tokenizer = mock_tok
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-4-E2B")
        decoded = tokenizer.decode([4103, 2134, 1])
        self.assertEqual(decoded, "Hello world")

    @patch("gpt_tokenizers.AutoProcessor")
    def test_processor_without_tokenizer_attr_used_directly(self, mock_auto_processor):
        """When processor has no .tokenizer attr, use processor itself."""
        mock_enc = MagicMock(spec=[])
        mock_enc.eos_token_id = 2
        mock_enc.encode = MagicMock(return_value=[100, 200])
        mock_enc.decode = MagicMock(return_value="test")
        mock_auto_processor.from_pretrained.return_value = mock_enc

        tokenizer = Tokenizer("google/gemma-3-1b")
        tokens = tokenizer.tokenize("test")
        self.assertEqual(tokens, [100, 200, 2])

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_non_gemma_uses_auto_tokenizer(self, mock_auto_tokenizer):
        """Non-gemma models use AutoTokenizer, not AutoProcessor."""
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
