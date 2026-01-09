import multiprocessing
import os
import logging
from typing import Tuple

import numpy as np
from datasets import load_dataset
from ddp import master_proc
from tokenizers import Tokenizer

log = logging.getLogger(__name__)
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)
num_procs = max(1, os.cpu_count() // 2)

class Downloader:
    def __init__(self, dataset_id: str, shard_size: int, encoding: str):
        self.dataset_id = dataset_id
        self.shard_size = shard_size
        self.tokenizer = Tokenizer(encoding)

    def _save(self, shard_idx: int, tokens: list[int]):
        f = os.path.join(DATA_FOLDER, f"{self.dataset_id}_{shard_idx:06d}")
        np.save(f, np.array(tokens, dtype=np.uint16))
        log.info(f"Saved shard {shard_idx:06d} with {len(tokens)} tokens into {f}")

    def download(self, path: str, name: str, split: str):
        ds = load_dataset(path, name, split=split)
        with multiprocessing.Pool(num_procs) as pool:
            tokens: list[int] = []
            shard_idx = 0
            for i, chunk in enumerate(pool.imap(self.tokenizer.tokenize, ds["text"], chunksize=16)):
                tokens.extend(chunk)
                if len(tokens) >= self.shard_size:
                    self._save(shard_idx, tokens[:self.shard_size])
                    shard_idx += 1
                    tokens = tokens[self.shard_size:]
                if i % (self.shard_size // 100) == 0:
                    log.info(f"Cached {len(tokens)} of {self.shard_size} tokens in shard {shard_idx:06d}")
            if len(tokens) > 0:
                self._save(shard_idx, tokens)

class Loader:
    def __init__(self, dataset_id: str, begin_shard=0, begin_idx=0, buffer_size=0, idx_offset=0):
        self.shards = sorted([shard for shard in os.listdir(DATA_FOLDER) if dataset_id in shard])
        if master_proc():
            log.info(f"Found {len(self.shards)} shard(s) for {dataset_id}")

        self.shard_idx = begin_shard
        self.buffer_size = buffer_size
        self.idx_offset = idx_offset
        self.token_idx = begin_idx
        self.tokens = np.empty((0,))

    def list(self) -> list[str]:
        return self.shards

    def delete(self):
        for shard in self.shards:
            os.remove(os.path.join(DATA_FOLDER, shard))

    def _load(self) -> np.ndarray[np.int32]:
        return np.load(os.path.join(DATA_FOLDER, self.shards[self.shard_idx])).astype(np.int32)

    def next_batch(self, target_offset=1) -> Tuple[np.ndarray[np.int32], np.ndarray[np.int32] | None]:
        # load tokens from selected shard
        if len(self.tokens) == 0:
            self.tokens = self._load()
        # extend token buffer up to number of shards times if forecasted to be out of bounds
        for _ in range(len(self.shards)):
            if len(self.tokens) < self.token_idx + self.idx_offset + target_offset:
                # advance to next shard (or start over)
                self.shard_idx = (self.shard_idx + 1) % len(self.shards)
                # combine remaining tokens with next shard and reset index
                self.tokens = np.concatenate((self.tokens[self.token_idx:], self._load()))
                self.token_idx = 0
            else: # enough tokens available
                break
        # input and target shifted by offset
        input_array = self.tokens[self.token_idx: self.token_idx + self.buffer_size]
        target_array = None
        if target_offset > 0:
            target_array = self.tokens[self.token_idx + target_offset: self.token_idx + self.buffer_size + target_offset]
        # advance token index by offset
        self.token_idx += self.idx_offset
        # return batch
        return input_array, target_array
