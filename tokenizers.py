import tiktoken
from tiktoken import Encoding

class Tokenizer:
    def __init__(self, encoding_name: str):
        self._enc: Encoding = tiktoken.get_encoding(encoding_name)

    def tokenize(self, text: str) -> list[int]:
        tokens = self._enc.encode_ordinary(text) + [self._enc.eot_token]
        return tokens

    def decode(self, tokens: list[int]) -> str:
        decoded_text = self._enc.decode(tokens)
        return decoded_text
