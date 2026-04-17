import json
import logging
import os

import tiktoken
from tiktoken import Encoding
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

TIKTOKEN_PREFIX = "tiktoken/"
# Multimodal model families whose tokenizer class differs from AutoTokenizer's
# default resolution.  Extend this tuple as more multimodal models are added.
_PROCESSOR_PATTERNS = ("/gemma-",)


class Tokenizer:
    def __init__(self, encoding_name: str):
        if encoding_name.startswith(TIKTOKEN_PREFIX):
            enc = tiktoken.get_encoding(encoding_name[len(TIKTOKEN_PREFIX):])
            self._tokenize = lambda text: enc.encode_ordinary(text) + [enc.eot_token]
            self._decode = enc.decode
        else:
            enc = self._load_hf_tokenizer(encoding_name)
            self._tokenize = lambda text: enc.encode(text, add_special_tokens=False) + ([enc.eos_token_id] if enc.eos_token_id is not None else [])
            self._decode = enc.decode

    @staticmethod
    def _load_hf_tokenizer(encoding_name: str) -> PreTrainedTokenizerBase:
        if any(p in encoding_name for p in _PROCESSOR_PATTERNS):
            enc = Tokenizer._load_tokenizer_from_processor(encoding_name)
            log.info("Loaded tokenizer from processor config for %s", encoding_name)
            return enc
        enc = AutoTokenizer.from_pretrained(encoding_name)
        log.info("Loaded tokenizer via AutoTokenizer for %s", encoding_name)
        return enc

    @staticmethod
    def _load_tokenizer_from_processor(repo_id: str) -> PreTrainedTokenizerBase:
        """Load only the tokenizer from a multimodal model's processor config.

        Reads processor_config.json to discover the tokenizer class, then
        instantiates it directly — avoids loading the full processor which
        may pull in heavy dependencies like PIL for image handling.
        """
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        path = hf_hub_download(repo_id=repo_id, filename="processor_config.json",
                               token=token)
        with open(path) as f:
            tokenizer_class_name = json.load(f).get("tokenizer_class")
        if tokenizer_class_name:
            tokenizer_cls = getattr(transformers, tokenizer_class_name, None)
            if tokenizer_cls is not None:
                return tokenizer_cls.from_pretrained(repo_id, token=token)
        return AutoTokenizer.from_pretrained(repo_id, token=token)

    def tokenize(self, text: str) -> list[int]:
        return self._tokenize(text)

    def decode(self, tokens: list[int]) -> str:
        return self._decode(tokens)
