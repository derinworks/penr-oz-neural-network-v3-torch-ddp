from transformers import AutoTokenizer, PreTrainedTokenizerBase

class Tokenizer:
    def __init__(self, model_name: str):
        self._enc: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)

    def tokenize(self, text: str) -> list[int]:
        tokens = self._enc.encode(text, add_special_tokens=False) + ([self._enc.eos_token_id] if self._enc.eos_token_id is not None else [])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        decoded_text = self._enc.decode(tokens)
        return decoded_text
