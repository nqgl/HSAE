#%%
%cd /workspace
#%%
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig
import json
import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
from functools import partial
import tqdm.notebook as tqdm
import plotly.express as px
import pandas as pd
import einops
from pathlib import Path
from nqgl.sae.calculations_on_sae import get_recons_loss, replacement_hook, zero_ablate_hook, mean_ablate_hook

#%%
hsae = HierarchicalAutoEncoder.load(save_dir="/workspace/nqgl/sae/models/")
# %%
"""## Loading Data"""
def shuffle_data(all_tokens):
    print("Shuffled data")
    return all_tokens[torch.randperm(all_tokens.shape[0])]

loading_data_first_time = False
if loading_data_first_time:
    data = load_dataset("NeelNanda/c4-code-tokenized-2b", split="train", cache_dir="/workspace/cache/")
    data.save_to_disk("/workspace/data/c4_code_tokenized_2b.hf")
    data.set_format(type="torch", columns=["tokens"])
    all_tokens = data["tokens"]
    all_tokens.shape


    all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
    all_tokens_reshaped[:, 0] = model.tokenizer.bos_token_id
    all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]
    torch.save(all_tokens_reshaped, "/workspace/data/c4_code_2b_tokens_reshaped.pt")
else:
    # data = datasets.load_from_disk("/workspace/data/c4_code_tokenized_2b.hf")
    all_tokens = torch.load("/workspace/data/c4_code_2b_tokens_reshaped.pt")
    all_tokens = shuffle_data(all_tokens)
#%%

SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def process_tokens_index(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

"""### Make Token DataFrame"""

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]
def make_token_df(tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))

# %%
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
cfg = hsae.cfg
model_name = hsae.cfg.model_name
enc_dtype = hsae.cfg.enc_dtype
model = HookedTransformer.from_pretrained(model_name).to(DTYPES[enc_dtype])
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
batch_size = 32
number_of_batches = 100
tokens = all_tokens[:batch_size*number_of_batches]
activations = []
sae1 = hsae.saes[0]


for i in range(number_of_batches):
    with torch.no_grad():
        input_tokens = tokens[i*batch_size:(i+1)*batch_size]
        _, cache = model.run_with_cache(input_tokens, stop_at_layer=cfg.layer+1)
        mlp_acts = cache[cfg.act_name]
        mlp_acts_flattened = mlp_acts.reshape(-1, cfg.act_size)
        x_reconstruct = hsae(mlp_acts_flattened)
        hidden_acts = hsae.saes[0].cached_acts.reshape(batch_size*128, -1)
        activations.append(hidden_acts.cpu().numpy())
hidden_acts = np.array(activations).reshape(-1, sae1.cfg.d_dict * sae1.cfg.n_sae)

print(hidden_acts.shape)
# %%
feature_id = 1
token_df = make_token_df(tokens)
features = hidden_acts[:, feature_id]
token_df["feature"] = utils.to_numpy(features)
token_df.sort_values("feature", ascending=False).head(10).style.background_gradient("coolwarm")

# %%
