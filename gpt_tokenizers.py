import logging

import tiktoken
from tiktoken import Encoding
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

TIKTOKEN_PREFIX = "tiktoken/"
# Multimodal model families that require AutoProcessor for correct tokenization.
# Extend this tuple as support for more multimodal models is added.
_PROCESSOR_PATTERNS = ("/gemma-",)


class Tokenizer:
    def __init__(self, encoding_name: str):
        if encoding_name.startswith(TIKTOKEN_PREFIX):
            enc = tiktoken.get_encoding(encoding_name[len(TIKTOKEN_PREFIX):])
            self._tokenize = lambda text: enc.encode_ordinary(text) + [enc.eot_token]
            self._decode = enc.decode
        elif any(p in encoding_name for p in _PROCESSOR_PATTERNS):
            proc = AutoProcessor.from_pretrained(encoding_name)
            tokenizer = getattr(proc, "tokenizer", proc)
            # Use whichever object carries the chat template (tokenizer > processor)
            chat_host = None
            if getattr(tokenizer, "chat_template", None):
                chat_host = tokenizer
            elif getattr(proc, "chat_template", None):
                chat_host = proc
            if chat_host is not None:
                self._tokenize = lambda text: chat_host.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True, add_generation_prompt=True,
                )
                self._decode = lambda tokens: proc.parse_response(
                    proc.decode(tokens, skip_special_tokens=False)
                )
            else:
                self._tokenize = lambda text: tokenizer.encode(text, add_special_tokens=False) + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
                self._decode = tokenizer.decode
            log.info("Loaded tokenizer via AutoProcessor for %s (chat_template=%s)",
                     encoding_name, chat_host is not None)
        else:
            enc = AutoTokenizer.from_pretrained(encoding_name)
            self._tokenize = lambda text: enc.encode(text, add_special_tokens=False) + ([enc.eos_token_id] if enc.eos_token_id is not None else [])
            self._decode = enc.decode
            log.info("Loaded tokenizer via AutoTokenizer for %s", encoding_name)

    def tokenize(self, text: str) -> list[int]:
        return self._tokenize(text)

    def decode(self, tokens: list[int]) -> str:
        return self._decode(tokens)
