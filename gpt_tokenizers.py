import logging

import tiktoken
from tiktoken import Encoding
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

TIKTOKEN_PREFIX = "tiktoken/"


def _load_hf_tokenizer(encoding_name: str) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer, preferring AutoProcessor for multimodal models."""
    try:
        proc = AutoProcessor.from_pretrained(encoding_name)
        tokenizer = getattr(proc, "tokenizer", proc)
        log.info("Loaded tokenizer via AutoProcessor for %s", encoding_name)
        return tokenizer
    except Exception:
        enc = AutoTokenizer.from_pretrained(encoding_name)
        log.info("Loaded tokenizer via AutoTokenizer for %s", encoding_name)
        return enc


class Tokenizer:
    def __init__(self, encoding_name: str):
        if encoding_name.startswith(TIKTOKEN_PREFIX):
            enc = tiktoken.get_encoding(encoding_name[len(TIKTOKEN_PREFIX):])
            self._tokenize = lambda text: enc.encode_ordinary(text) + [enc.eot_token]
            self._decode = enc.decode
        else:
            enc = _load_hf_tokenizer(encoding_name)
            self._tokenize = lambda text: enc.encode(text, add_special_tokens=False) + ([enc.eos_token_id] if enc.eos_token_id is not None else [])
            self._decode = enc.decode

    def tokenize(self, text: str) -> list[int]:
        return self._tokenize(text)

    def decode(self, tokens: list[int]) -> str:
        return self._decode(tokens)
