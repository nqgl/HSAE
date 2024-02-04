#%%
%cd /root
#%%
!pip install matplotlib
!pip install plotly
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
from nqgl.sae.training.setup_utilstup_utils import get_model, load_data, shuffle_documents
from nqgl.sae.analysis.utils_from_others import *
#%%
hsae = HierarchicalAutoEncoder.load_latest(name="absurd-blaze")
# %%
from nqgl.sae.training.buffer import Buffer

model = get_model(hsae.cfg)
all_tokens = load_data(model)
buffer = Buffer(hsae.cfg, all_tokens, model, dont_shuffle=True)
#%%

SPACE = "·"
NEWLINE="↩"
TAB = "→"

# %%
cfg = hsae.cfg
model_name = hsae.cfg.model_name
enc_dtype = hsae.cfg.enc_dtype

# model = HookedTransformer.from_pretrained(model_name).to(DTYPES[enc_dtype])


n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
batch_size = 32
number_of_batches = 40
tokens = all_tokens[:batch_size*number_of_batches]
activations = []
sae1 = hsae.layers[0]
# tokens = []


for i in range(number_of_batches):
    print(i)
    with torch.no_grad():
        input_tokens = tokens[i*batch_size:(i+1)*batch_size]
        _, cache = model.run_with_cache(input_tokens, stop_at_layer=cfg.layer+1)
        mlp_acts = cache[cfg.act_name]
        model_acts = mlp_acts.reshape(-1, cfg.act_size)
        # acts = buffer.next()
        # buffer.refresh()
        # input_tokens = buffer.all_tokens[
        #             buffer.token_pointer  - buffer.cfg.model_batch_size : buffer.token_pointer
        #         ]
        # tokens += input_tokens
        hidden_acts = torch.zeros(model_acts.shape[0], sae1.cfg.n_sae, sae1.cfg.d_dict).to(cfg.device)
        for j in range(model_acts.shape[0] // hsae.cfg.batch_size):
            hsae(model_acts[j*hsae.cfg.batch_size:(j+1)*hsae.cfg.batch_size])
            hidden_acts[j*hsae.cfg.batch_size:(j+1)*hsae.cfg.batch_size] = hsae.layers[0].cached_acts
        activations.append(hidden_acts.cpu().numpy())
        
        # x_reconstruct = hsae(model_acts)

print(input_tokens.shape)
# print(acts.shape)
hidden_acts = np.array(activations).reshape(-1, sae1.cfg.d_dict * sae1.cfg.n_sae)

print(hidden_acts.shape)
# %%
def show_random_highly_activating(activations, feature_id, percentile=90):
    token_df = make_token_df(tokens, len_prefix=10, len_suffix=3)
    features = activations[:, feature_id]
    token_df["feature"] = utils.to_numpy(features)
    #select all where feature > 0.0
    token_df = token_df[token_df["feature"]>0]
    #get the 50th percentile
    percentile = np.percentile(token_df["feature"], percentile)
    #select all where feature > 50th percentile/usr/local/lib/python3.8/dist-packages/pandas/core/common.py
    display(token_df[token_df["feature"]>percentile].sample(10).style.background_gradient("coolwarm"))

# %%

freq_per_feature = (hidden_acts > 0).mean(axis=0)


high_data_features = np.arange(0, freq_per_feature.shape[0])[np.argsort(freq_per_feature)[::-1][:100]]
print("top level dict", cfg.d_dict)

# %%
tlf = 2
for feature_id in range(tlf * 32, (1+tlf) * 32):
    try:
        show_random_highly_activating(hidden_acts, feature_id, percentile=99)
    except:
        continue
    print(f"the above feature was {feature_id}")
    # else:
        # break
# %%
