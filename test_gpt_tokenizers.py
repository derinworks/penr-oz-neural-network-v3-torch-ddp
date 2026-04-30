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
    """AutoProcessor is used for multimodal models matching _PROCESSOR_PATTERNS.
    Tokenization applies the chat template so the model sees correctly formatted input."""

    @patch("gpt_tokenizers.AutoProcessor")
    def test_apply_chat_template_returns_list(self, mock_auto_processor):
        mock_proc = MagicMock()
        mock_proc.tokenizer.chat_template = "{% for m in messages %}..."
        mock_proc.tokenizer.apply_chat_template.return_value = [2, 100, 200, 300]
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-4-E2B")
        tokens = tokenizer.tokenize("Hello world")

        self.assertEqual(tokens, [2, 100, 200, 300])
        mock_proc.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hello world"}],
            tokenize=True, add_generation_prompt=True,
        )
        mock_auto_processor.from_pretrained.assert_called_once_with("google/gemma-4-E2B")

    @patch("gpt_tokenizers.AutoProcessor")
    def test_apply_chat_template_returns_dict(self, mock_auto_processor):
        """Processors may return a dict with input_ids and attention_mask."""
        mock_proc = MagicMock()
        mock_proc.tokenizer.chat_template = "{% for m in messages %}..."
        mock_proc.tokenizer.apply_chat_template.return_value = {
            "input_ids": [2, 100, 200, 300],
            "attention_mask": [1, 1, 1, 1],
        }
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-4-E2B")
        tokens = tokenizer.tokenize("Hello world")

        self.assertEqual(tokens, [2, 100, 200, 300])

    @patch("gpt_tokenizers.AutoProcessor")
    def test_apply_chat_template_returns_batched_dict(self, mock_auto_processor):
        """Processors may return batched input_ids (shape [1, seq])."""
        mock_proc = MagicMock()
        mock_proc.tokenizer.chat_template = "{% for m in messages %}..."
        mock_proc.tokenizer.apply_chat_template.return_value = {
            "input_ids": [[2, 100, 200, 300]],
            "attention_mask": [[1, 1, 1, 1]],
        }
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-4-E2B")
        tokens = tokenizer.tokenize("Hello world")

        self.assertEqual(tokens, [2, 100, 200, 300])

    @patch("gpt_tokenizers.AutoProcessor")
    def test_no_chat_template_falls_back_to_encode(self, mock_auto_processor):
        """When neither processor nor tokenizer has a chat template, use plain encode."""
        mock_proc = MagicMock()
        mock_proc.chat_template = None
        mock_proc.tokenizer.chat_template = None
        mock_proc.tokenizer.eos_token_id = 1
        mock_proc.tokenizer.encode.return_value = [100, 200]
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-3-1b")
        tokens = tokenizer.tokenize("test")
        self.assertEqual(tokens, [100, 200, 1])

    @patch("gpt_tokenizers.AutoProcessor")
    def test_decode_with_chat_template_uses_parse_response(self, mock_auto_processor):
        """When chat template is active, decode calls proc.decode + proc.parse_response."""
        mock_proc = MagicMock()
        mock_proc.tokenizer.chat_template = "{% for m in messages %}..."
        mock_proc.decode.return_value = "<start>Hello world<end>"
        mock_proc.parse_response.return_value = "Hello world"
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-4-E2B")
        decoded = tokenizer.decode([100, 200])

        self.assertEqual(decoded, "Hello world")
        mock_proc.decode.assert_called_once_with([100, 200], skip_special_tokens=False)
        mock_proc.parse_response.assert_called_once_with("<start>Hello world<end>")

    @patch("gpt_tokenizers.AutoProcessor")
    def test_decode_without_chat_template_uses_tokenizer(self, mock_auto_processor):
        """Without chat template, decode uses tokenizer.decode directly."""
        mock_proc = MagicMock()
        mock_proc.chat_template = None
        mock_proc.tokenizer.chat_template = None
        mock_proc.tokenizer.eos_token_id = 1
        mock_proc.tokenizer.decode.return_value = "Hello world"
        mock_auto_processor.from_pretrained.return_value = mock_proc

        tokenizer = Tokenizer("google/gemma-3-1b")
        decoded = tokenizer.decode([100, 200])
        self.assertEqual(decoded, "Hello world")

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
