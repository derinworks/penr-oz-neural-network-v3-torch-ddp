import unittest
from tokenizers import Tokenizer


class TestTokenizer(unittest.TestCase):

    def test_tokenize(self):
        tokenizer = Tokenizer("gpt2")
        tokens = tokenizer.tokenize("Hello world")
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertTrue(all(isinstance(t, int) for t in tokens))

    def test_decode(self):
        tokenizer = Tokenizer("gpt2")
        original_text = "Hello world"
        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)
        
        self.assertIsInstance(decoded_text, str)
        # Decoded text should contain the original (may have extra tokens like EOT)
        self.assertIn("Hello world", decoded_text)

    def test_tokenize_decode_roundtrip(self):
        tokenizer = Tokenizer("gpt2")
        original_text = "The quick brown fox jumps over the lazy dog."
        
        tokens = tokenizer.tokenize(original_text)
        decoded_text = tokenizer.decode(tokens)
        
        # Should contain the original text
        self.assertIn(original_text, decoded_text)

    def test_tokenize_empty_string(self):
        tokenizer = Tokenizer("gpt2")
        tokens = tokenizer.tokenize("")
        
        # Even empty string should have EOT token
        self.assertEqual(len(tokens), 1)


if __name__ == '__main__':
    unittest.main()
