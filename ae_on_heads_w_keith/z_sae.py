# pair programming by Glen Taggart and Kieth Wynroe based mostly off of 
# this work https://colab.research.google.com/drive/1MjF_5-msnSe5F9Qy4kEGSeqyYPE9_D2p?authuser=1#scrollTo=7WXAjU3mRak6
# done by Neel Nanda and others
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
    buffer_refresh_ratio :float = 0.9
    nonlinearity :tuple = ("relu", {})

# Ithink this is gelu_2 specific


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.float16, "bfp16" : torch.bfloat16}
SAVE_DIR = Path.home() / "workspace"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir()


def post_init_cfg(cfg):
    cfg.model_batch_size = cfg.batch_size // cfg.seq_len * 16
    cfg.buffer_size = cfg.batch_size * cfg.buffer_mult
    cfg.buffer_batches = cfg.buffer_size // cfg.seq_len
    cfg.act_name = utils.get_act_name(cfg.site, cfg.layer)
    cfg.dict_size = cfg.act_size * cfg.dict_mult
    cfg.name = f"{cfg.model_name}_{cfg.layer}_{cfg.dict_size}_{cfg.site}"
    return cfg
def default_cfg():
    cfg = AutoEncoderConfig(site="z") #gives it default values
    return post_init_cfg(default_cfg)

def get_model(cfg):
    model = HookedTransformer.from_pretrained(cfg.model_name).to(DTYPES[cfg.enc_dtype]).to(cfg.device)
    return model



def shuffle_documents(all_tokens): # assuming the shape[0] is documents
    # print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]


def load_data(model, dataset = "NeelNanda/c4-code-tokenized-2b"):
    import os
    reshaped_name = dataset.split("/")[-1] + "_reshaped.pt"
    dataset_reshaped_path = SAVE_DIR / "data" / reshaped_name
    # if dataset exists loading_data_first_time=False
    loading_data_first_time = not dataset_reshaped_path.exists()

    
    if loading_data_first_time:
        data = load_dataset(dataset, split="train", cache_dir=SAVE_DIR / "cache/")
        data.save_to_disk(os.path.join(SAVE_DIR / "data/", dataset.split("/")[-1]+".hf"))
        data.set_format(type="torch", columns=["tokens"])
        all_tokens = data["tokens"]
        all_tokens.shape


        all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
        all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
        all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]
        torch.save(all_tokens_reshaped, dataset_reshaped_path)
    else:
        # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
        all_tokens = torch.load(dataset_reshaped_path)
        all_tokens = shuffle_documents(all_tokens)
    return all_tokens


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_dict = cfg.dict_size
        l1_coeff = cfg.l1_coeff
        dtype = DTYPES[cfg.enc_dtype]
        torch.manual_seed(cfg.seed)
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg.act_size, d_dict, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_dict, cfg.act_size, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_dict, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.act_size, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_dict = d_dict
        self.l1_coeff = l1_coeff
        self.acts_cached = None
        self.l2_loss_cached = None
        self.l1_loss_cached = None
        self.l0_norm_cached = None
        self.to(cfg.device)
        self.cfg = cfg      
        self.cached_acts = None
        self.nonlinearity = config_compatible_relu_choice.cfg_to_nonlinearity(cfg)
        self.activation_frequency = torch.zeros(self.d_dict, dtype=torch.float32).to(cfg.device)
        self.steps_since_activation_frequency_reset = 0


    def forward(self, x, cache_l0 = True, cache_acts = False, record_activation_frequency = False):
        x_cent = x - self.b_dec
        # print(x_cent.dtype, x.dtype, self.W_dec.dtype, self.b_dec.dtype)
        acts = self.nonlinearity(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        self.l2_loss_cached = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        self.l1_loss_cached = (acts.float().abs().sum(dim=(-2)))
        if cache_l0:
            self.l0_norm_cached = (acts > 0).float().sum(dim=-1).mean()
        else:
            self.l0_norm_cached = None
        if cache_acts:
            self.cached_acts = acts
        else:
            self.cached_acts = None
        if record_activation_frequency:
            # print(acts.shape)
            activated = torch.mean((acts > 0).float(), dim=0)
            # print("activated shape", activated.shape)
            # print("freq shape", self.activation_frequency.shape)
            self.activation_frequency = activated + self.activation_frequency
            self.steps_since_activation_frequency_reset += 1
        return x_reconstruct
    
    @torch.no_grad()
    def reset_activation_frequencies(self):
        self.activation_frequency[:] = 0
        self.steps_since_activation_frequency_reset = 0

    def get_loss(self):
        return self.l2_loss_cached + torch.sum(self.l1_coeff * self.l1_loss_cached)


    

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    def get_version(self):
        version_list = [int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)]
        if len(version_list):
            return 1+max(version_list)
        else:
            return 0

    def save(self, name=""):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+ "_" + name + ".pt"))
        with open(SAVE_DIR/(str(version)+ "_" + name + "_cfg.json"), "w") as f:
            json.dump(asdict(self.cfg), f)
        print("Saved as version", version)

    @classmethod
    def load(cls, version):
        cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
        cfg = AutoEncoderConfig(**cfg)
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
        return self



# I might come back to this and think about changing refresh ratio up
# also is there a pipelining efficiency we could add?
# is it bad to have like 2 gb of tokens in memory?
class Buffer():
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder.
    It'll automatically run the model to generate more when it gets halfway empty.
    """
    def __init__(self, cfg, tokens, model):
        self.buffer = torch.zeros((cfg.buffer_size, cfg.act_size), dtype=torch.float16, requires_grad=False).to(cfg.device)
        self.cfg :AutoEncoderConfig = cfg
        self.token_pointer = 0
        self.first = True
        self.all_tokens = tokens
        self.cfg.buffer_refresh_ratio = 0.9
        self.model = model
        self.time_shuffling = 0
        self.refresh()

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
                tokens = self.all_tokens[self.token_pointer:self.token_pointer+self.cfg.model_batch_size]
                _, cache = self.model.run_with_cache(tokens, stop_at_layer=self.cfg.layer+1)
                # acts = cache[self.cfg.act_name].reshape(-1, self.cfg.act_size)
                # z has a head index 
                if self.cfg.flatten_heads:
                    acts = einops.rearrange(cache[self.cfg.act_name], "batch seq_pos n_head d_head -> (batch seq_pos) (n_head d_head)")
                else:
                    acts = einops.rearrange(cache[self.cfg.act_name], "batch seq_pos d_act -> (batch seq_pos) d_act")
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
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg.device)]
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
        n = ((1 - self.cfg.buffer_refresh_ratio) * self.cfg.buffer_size) // self.cfg.batch_size
        for _ in range(1 + int(fresh_factor / (1 - self.cfg.buffer_refresh_ratio))):
            self.pointer += (n + 1) * self.cfg.batch_size
            self.refresh()






























def main():
    ae_cfg = AutoEncoderConfig(site="z", act_size=512)
    cfg = post_init_cfg(ae_cfg)
    all_tokens = load_data()
    encoder = AutoEncoder(cfg)
    buffer = Buffer(cfg, all_tokens, encoder)


if __name__ == "__main__":
    main()