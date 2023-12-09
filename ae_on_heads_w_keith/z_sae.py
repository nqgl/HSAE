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
    beta1 :float = 0.9
    beta2 :float = 0.99
    dict_mult :int = 32
    seq_len :int = 128
    layer :int = 0
    enc_dtype :str = "fp32"
    model_name :str = "gelu-2l"
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
    experimental_type: Optional[str] = None
    gram_shmidt_trail :int = 5000
    num_to_resample :int = 128
    data_rescale :float = 1.0

    def __post_init__(self):
        print("Post init")
        self.post_init_cfg()

    def post_init_cfg(self):
        self.model_batch_size = self.batch_size // self.seq_len * 16
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
        self.act_name = utils.get_act_name(self.site, self.layer)
        self.dict_size = self.act_size * self.dict_mult
        self.name = f"{self.model_name}_{self.layer}_{self.dict_size}_{self.site}"
        return self
# Ithink this is gelu_2 specific


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.float16, "bfp16" : torch.bfloat16}
SAVE_DIR = Path.home() / "workspace"
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir()


def default_cfg():
    cfg = AutoEncoderConfig(site="z") #gives it default values
    return post_init_cfg(default_cfg)

def get_model(cfg):
    model = HookedTransformer.from_pretrained(cfg.model_name).to(DTYPES[cfg.enc_dtype]).to(cfg.device)
    return model



def shuffle_documents(all_tokens): # assuming the shape[0] is documents
    # print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]


def load_data(model :transformer_lens.HookedTransformer, dataset = "NeelNanda/c4-code-tokenized-2b"):
    import os
    reshaped_name = dataset.split("/")[-1] + "_reshaped.pt"
    dataset_reshaped_path = SAVE_DIR / "data" / reshaped_name
    # if dataset exists loading_data_first_time=False
    loading_data_first_time = not dataset_reshaped_path.exists()

    print("first time:", loading_data_first_time)
    if loading_data_first_time:
        data = load_dataset(dataset, split="train", cache_dir=SAVE_DIR / "cache/")
        # data.save_to_disk(os.path.join(SAVE_DIR / "data/", dataset.split("/")[-1]+".hf"))
        if "tokens" not in data.column_names:
            if "text" in data.column_names:
                data.set_format(type="torch", columns=["text"])
                data = data["text"]
                # model.tokenizer.
                all_tokens = model.tokenizer.tokenize(data["text"], return_tensors="pt", padding=True, truncation=True, max_length=128)
        else:
            data.set_format(type="torch", columns=["tokens"])
            all_tokens = data["tokens"]
        all_tokens.shape


        all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
        all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
        all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]
        print("saving to:", dataset_reshaped_path)
        torch.save(all_tokens_reshaped, dataset_reshaped_path)
        print("saved reshaped data")
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
        self.step_num = 0
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
        self.to_be_reset = None
        self.x_cent_cached = None

    def forward(self, x, cache_l0 = True, cache_acts = False, record_activation_frequency = False):
        x = x * self.cfg.data_rescale
        x_cent = x - self.b_dec
        # print(x_cent.dtype, x.dtype, self.W_dec.dtype, self.b_dec.dtype)
        acts = self.nonlinearity(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        # self.l2_loss_cached = (x_reconstruct.float() - x.float()).pow(2).mean(-1).mean(0)
        x_diff = x_reconstruct.float() - x.float()
        # self.re_init_neurons(x_diff)
        self.l1_loss_cached = acts.float().abs().mean(dim=(-2))
        self.l2_loss_cached = (x_diff).pow(2).mean(-1).mean(0)
        self.x_cent_cached = x_cent.detach()
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
            self.activation_frequency = activated + self.activation_frequency.detach()
            self.steps_since_activation_frequency_reset += 1
        return x_reconstruct / self.cfg.data_rescale
    



    def get_loss(self):
        self.step_num += 1
        if self.cfg.cosine_l1 is None:
            l1_coeff = self.l1_coeff
        else:
            c_period, c_range = self.cfg.cosine_l1["period"], self.cfg.cosine_l1["range"]
            l1_coeff = self.l1_coeff * (1 + c_range * torch.cos(torch.tensor(2 * torch.pi * self.step_num / c_period).detach()))
        scaling = self.x_cent_cached.norm(dim=-1).mean()
        l2 = torch.mean(self.l2_loss_cached)
        l1 = torch.sum(l1_coeff * self.l1_loss_cached)
        return l1 / torch.sqrt(scaling) + l2 / scaling



    @torch.no_grad()
    def neurons_to_reset(self, to_be_reset :torch.Tensor):
        if to_be_reset.sum() > 0:
            self.to_be_reset = torch.argwhere(to_be_reset).squeeze(1)
            w_enc_norms = self.W_enc[:, ~ to_be_reset].norm(dim=0)
            # print("w_enc_norms", w_enc_norms.shape)
            # print("to_be_reset", self.to_be_reset.sum())
            self.alive_norm_along_feature_axis = torch.mean(torch.mean(w_enc_norms))
        else:
            self.to_be_reset = None
    
    @torch.no_grad()
    def re_init_neurons(self, x_diff):
        self.re_init_neurons_gram_shmidt_precise(x_diff)

    
    @torch.no_grad()
    def re_init_neurons_gram_shmidt_precise(self, x_diff):
        t = self.cfg.gram_shmidt_trail
        n_reset = min(x_diff.shape[0], self.cfg.act_size // 2, self.cfg.num_to_resample)
        v_orth = torch.zeros_like(x_diff)
        # print(x_diff.shape)
        # v_orth[0] = F.normalize(x_diff[0], dim=-1)
        n_succesfully_reset = n_reset
        for i in range(n_reset):
            v_orth[i] = x_diff[i]
            for j in range(max(0, i - t), i):
                v_orth[i] -= torch.dot(v_orth[j], v_orth[i]) * v_orth[j] / torch.dot(v_orth[j], v_orth[j])
            if v_orth[i].norm() < 1e-6:
                n_succesfully_reset = i
                break
            v_orth[i] = F.normalize(v_orth[i], dim=-1)
            # v_ = x_diff[i] - v_bar * torch.dot(v_bar, x_diff[i])
            # # print(v_.shape)
            # v_orth[i] = v_ / v_.norm(dim=-1, keepdim=True)
        self.reset_neurons(v_orth[:n_succesfully_reset])

    @torch.no_grad()
    def reset_neurons(self, new_directions :torch.Tensor, norm_encoder_proportional_to_alive = True):
        if new_directions.shape[0] > self.to_be_reset.shape[0]:
            new_directions = new_directions[:self.to_be_reset.shape[0]]
        num_resets = new_directions.shape[0]
        to_reset = self.to_be_reset[:num_resets]
        self.to_be_reset = self.to_be_reset[num_resets:]
        if self.to_be_reset.shape[0] == 0:
            self.to_be_reset = None
        new_directions = new_directions / new_directions.norm(dim=-1, keepdim=True)
        print(f"to_reset shape", to_reset.shape)
        print(f"new_directions shape", new_directions.shape)
        print(f"self.W_enc shape", self.W_enc.shape)
        if norm_encoder_proportional_to_alive:
            self.W_enc.data[:, to_reset] = new_directions.T * self.alive_norm_along_feature_axis * 0.2
        else:
            self.W_enc.data[:, to_reset] = new_directions.T
        self.W_dec.data[to_reset, :] = new_directions
        self.b_enc.data[to_reset] = 0



    @torch.no_grad()
    def reset_activation_frequencies(self):
        self.activation_frequency[:] = 0
        self.steps_since_activation_frequency_reset = 0


    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    @staticmethod
    def get_version():
        version_list = [int(file.name.split("_")[0]) for file in list(SAVE_DIR.iterdir()) if "_cfg.json" in str(file)]
        if len(version_list):
            return 1 + max(version_list)
        else:
            return 0

    def save(self, name=""):
        version = self.get_version()
        torch.save(self.state_dict(), SAVE_DIR/(str(version)+ "_" + name + ".pt"))
        with open(SAVE_DIR/(str(version)+ "_" + name + "_cfg.json"), "w") as f:
            json.dump(asdict(self.cfg), f)
        print("Saved as version", version)

    @classmethod
    def load(cls, version, cfg = None, save_dir = None):
        save_dir = SAVE_DIR if save_dir is None else Path(save_dir)
        # get correct name with globbing
        import glob
        if cfg is None:
            cfg_name = glob.glob(str(save_dir/(str(version)+"*_cfg.json")))
            cfg = json.load(open(cfg_name[0]))
            cfg = AutoEncoderConfig(**cfg)
        pt_name = glob.glob(str(save_dir/(str(version)+"*.pt")))
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(torch.load(pt_name[0]))
        return self

    @classmethod
    def load_latest(cls, new_cfg = None):
        version = cls.get_version() - 1
        ae = cls.load(version, new_cfg)
        return ae



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
        torch.cuda.empty_cache()
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


    @torch.no_grad()
    def skip_first_tokens_ratio(self, skip_percent, skip_batches):
        self.token_pointer += int(self.all_tokens.shape[0] * skip_percent)
        self.first = True
        self.refresh()



























def main():
    ae_cfg = AutoEncoderConfig(site="z", act_size=512)
    cfg = post_init_cfg(ae_cfg)
    all_tokens = load_data()
    encoder = AutoEncoder(cfg)
    buffer = Buffer(cfg, all_tokens, encoder)


if __name__ == "__main__":
    main()