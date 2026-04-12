import unittest
from unittest.mock import patch, MagicMock
from gpt_tokenizers import Tokenizer


class TestTokenizer(unittest.TestCase):

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

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_decode(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("gpt2")
        original_text = "Hello world"
        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)

        self.assertIsInstance(decoded_text, str)
        # Decoded text should contain the original (may have extra tokens like EOS)
        self.assertIn("Hello world", decoded_text)

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_tokenize_decode_roundtrip(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("gpt2")
        original_text = "The quick brown fox jumps over the lazy dog."

        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)

        # Should contain the original text
        self.assertIn(original_text, decoded_text)

    @patch("gpt_tokenizers.AutoTokenizer")
    def test_tokenize_empty_string(self, mock_auto_tokenizer):
        mock_auto_tokenizer.from_pretrained.return_value = self._make_mock_enc()
        tokenizer = Tokenizer("gpt2")
        tokens = tokenizer.tokenize("")

        # Even empty string should have EOS token
        self.assertEqual(len(tokens), 1)


if __name__ == '__main__':
    unittest.main()
