import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from argparse import ArgumentParser
import pprint
import argparse
import tqdm
from datasets import load_dataset
import einops
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from functools import partial
from collections import namedtuple
import time
from dataclasses import dataclass, asdict
from . import config_compatible_relu_choice
from typing import List, Tuple, Dict, Optional, Union, Callable
from torch.utils.data import Sampler, Dataset, DataLoader
from multiprocessing import Lock


@dataclass
class AutoEncoderConfig:
    seed :int = 49
    batch_size :int = 256
    buffer_mult :int = 10000
    lr :int = 3e-4
    num_tokens :int = int(2e9)
    l1_coeff :Union[float, List[float]] = 8e-4
    beta1 :int = 0.9
    beta2 :int = 0.99
    dict_mult :int = 32
    seq_len :int = 128
    layer :int = 0
    enc_dtype :str = "fp32"
    model_name :str = "gelu-1l"
    site :str = "" # z?
    device :str = "cuda"
    remove_rare_dir :bool = False
    act_size :int = -1
    flatten_heads :bool = True
    model_batch_size : int = None
    buffer_size :int = None
    buffer_batches :int = None
    act_name :str = None
    dict_size :int = None
    name :str = None
    buffer_refresh_ratio :float = 0.1
    nonlinearity :tuple = ("relu", {})
    cosine_l1 :Optional[Dict] = None

# I might come back to this and think about changing refresh ratio up
# also is there a pipelining efficiency we could add?
# is it bad to have like 2 gb of tokens in memory?
class BufferDataset(Dataset):
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder.
    It'll automatically run the model to generate more when it gets halfway empty.
    """
    def __init__(self, cfg, tokens, model, device = None):
        super().__init__()
        self.cfg :AutoEncoderConfig = cfg
        self.token_pointer = 0
        self.device = device
        self.all_tokens = tokens
        self.time_shuffling = 0
        self.refresh()


from multiprocessing import Process, Queue

class BufferRefresher(Process):
    def __init__(self, cfg, tokens, model, device=None):
        super(BufferRefresher, self).__init__()
        device = cfg.device if device is None else device
        self.device = device
        self.model = model
        self.cfg = cfg
        self.tokens = tokens
        self.model = model
        self.queue = Queue(maxsize=50)
        self.buffer = torch.zeros((cfg.buffer_size, cfg.act_size), dtype=torch.float16, requires_grad=False, device=device)
        self.token_pointer = 0
        self.first = True
        self.refresh()


    def run(self):
        while True:
            while self.queue.qsize() > 50:
                time.sleep(0.1)
            # If the buffer is running low, refresh it
            if self.token_pointer + self.cfg.batch_size > self.cfg.buffer_size:
                self.refresh()

            # Push the next batch of data into the queue
            batch = self.buffer[self.token_pointer:self.token_pointer + self.cfg.batch_size]
            self.queue.put(batch)

            # Move the pointer
            self.token_pointer += self.cfg.batch_size



    @torch.no_grad()
    def refresh(self):
        t0 = time.time()
        self.pointer = 0
        with torch.autocast("cuda", torch.float16):
            if self.first:
                num_batches = self.cfg.buffer_batches
            else:
                num_batches = int(self.cfg.buffer_batches * self.cfg.buffer_refresh_ratio)
            self.first = False
            for _ in range(0, num_batches, self.cfg.model_batch_size):
                tokens = self.all_tokens[self.token_pointer:self.token_pointer+self.cfg.model_batch_size].to(self.cfg.device)
                _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.cfg.layer+1)
                # acts = cache[self.cfg.act_name].reshape(-1, self.cfg.act_size)
                # z has a head index 
                if self.cfg.flatten_heads:
                    acts = einops.rearrange(cache[self.cfg.act_name].to(self.device), "batch seq_pos n_head d_head -> (batch seq_pos) (n_head d_head)")
                else:
                    acts = einops.rearrange(cache[self.cfg.act_name].to(self.device), "batch seq_pos d_act -> (batch seq_pos) d_act")
                assert acts.shape[-1] == self.cfg.act_size
                # it is ... n_head d_head and we want to flatten it into ... n_head * d_head
                # ... == batch seq_pos
                # print(tokens.shape, acts.shape, self.pointer, self.token_pointer)
                # print(cache[self.cfg.act_name].shape)
                # print("acts:", acts.shape)
                # print(acts.shape)
                # print(self.buffer.shape)
                # print("b", self.buffer[self.pointer: self.pointer+acts.shape[0]].shape)
                self.buffer[self.pointer: self.pointer+acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg.model_batch_size
                # if self.token_pointer > self.tokens.shape[0] - self.cfg.model_batch_size:
                #     self.token_pointer = 0

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0], device=self.device)]
        self.time_shuffling += time.time() - t0

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg.batch_size]
        self.pointer += self.cfg.batch_size
        if self.pointer > int(self.buffer.shape[0] * self.cfg.buffer_refresh_ratio) - self.cfg.batch_size:
            # print("Refreshing the buffer!")
            self.refresh()

        return out


    @torch.no_grad()
    def freshen_buffer(self, fresh_factor = 1, half_first=True):
        if half_first:
            n = (0.5 * self.cfg.buffer_size) // self.cfg.batch_size
            self.pointer += n * self.cfg.batch_size
            self.refresh()
        n = ((self.cfg.buffer_refresh_ratio) * self.cfg.buffer_size) // self.cfg.batch_size
        for _ in range(1 + int(fresh_factor / (self.cfg.buffer_refresh_ratio))):
            self.pointer += (n + 1) * self.cfg.batch_size
            self.refresh()

    # def __len__(self):
    #     return len(self.data_source)


    # def __getitem__(self, idx):
    #     # if torch.is_tensor(idx):
    #         # idx = idx.tolist()
    #     return self.buffer[idx]
    
class BufferSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        # do I need a lock? let's try and find out
        self.lock = Lock()

    def __iter__(self):
        while True:
            with self.lock:
                # If the buffer is running low, refresh it
                if self.data_source.pointer >= int(self.data_source.buffer.shape[0] * self.data_source.cfg.buffer_refresh_ratio) - self.data_source.cfg.batch_size:
                    self.data_source.refresh()

                # Yield the next batch of indices from the buffer
                self.data_source.token_pointer += self.data_source.cfg.batch_size
                indices = torch.arange(self.data_source.token_pointer - self.data_source.cfg.batch_size, self.data_source.token_pointer)
                yield indices

            # Move the pointer

    def __len__(self):
        return len(self.data_source)



def get_dataloader(cfg, tokens, model, device=None):
    dataset = BufferDataset(cfg, tokens, model, device=device)
    sampler = BufferSampler(dataset)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=50)
    return dataloader, dataset
