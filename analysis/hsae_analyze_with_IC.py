#%%
# %cd /root
#%%
# !pip install matplotlib
# !pip install plotly
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
from nqgl.sae.setup_utils import get_model, load_data, shuffle_documents
# from nqgl.sae.analysis.utils_from_others import *
from nqgl.sae.analysis.analysis_tools import HSAEInterpContext
#%%
MODELNAME="swift-fog"
# hsae = HierarchicalAutoEncoder.load_latest(name=MODELNAME)
hsae = HierarchicalAutoEncoder.load(420, save_dir="/home/g/mats/sae/remote_scripts/models-from-remote")

IC = HSAEInterpContext(hsae)
#%%
IC.populate_for_analysis()


#%%
print("top level features")
for i in range(5,15):
    print(f"Top level feature {i}:\t<MODELNAME={MODELNAME}>")
    IC.show_random_highly_activating(i)

# top = int(input("pick top level"))
# top=4
# #%%
# for j in range(32):
#     print(f"Second level feature {top}:{j}:\t<MODELNAME={MODELNAME}>")
#     IC.show_random_highly_activating(top, j)
# %%
def st(i, p=90):
    print(f"Top level feature {i}:\t<MODELNAME={MODELNAME}>")
    IC.show_random_highly_activating(i, percentile=p)

def sa(i, p=90):
    for j in range(32):
        print(f"Second level feature {i}:{j}:\t<MODELNAME={MODELNAME}>")
        IC.show_random_highly_activating(i, j, p)

def sl(i, j, p=90):
    print(f"Second level feature {i}:{j}:\t<MODELNAME={MODELNAME}>")
    IC.show_random_highly_activating(i, j, p)



# %%
m = IC.model

# LOGIT BOOSTING
def ld(f1, f2=None):
    while len(f1.shape) < 3:
        f1 = f1.unsqueeze(0)
    if f2 is None:
        return m.unembed(f1) - m.unembed(torch.zeros_like(f1))
    while len(f2.shape) < 3:
        f2 = f2.unsqueeze(0)
    return m.unembed(f1) - m.unembed(f2)

def tbld(f1, f2=None):
    ts = m.tokenizer.batch_decode(ld(f1, f2).topk(20).indices.squeeze())
    # display(ts)
    return ts



def fb(t, j):
    b = IC.hsae.b_dec[:]
    bf = IC.hsae.saes[0].b_dec[t, :]
    ff = IC.hsae.saes[0].W_dec[t, j, :]
    return b, bf, ff

def fbu(t,j):
    return [ld(x).squeeze() for x in fb(t,j)]
# logits = m.unembed(?f)
# pf1 = pre1 + logits
# l1 = m.unembed(pre1 + f) - m.unembed(pre1.unsqueeze(0).unsqueeze(0))
# l = l1.topk(10)
# t = torch.tensor(l.indices)
# m.tokenizer.batch_decode(t.squeeze())
# %%
t = 6
j = 25
b0, b1, f = fb(t, j)
sl(t,j,99)
display(tbld(b1 + f * 10, b0),
    tbld(f, b0 + b1),
    tbld(f, b0),
    tbld(f + b1),
    tbld(b1, b0))
# display(t)
# %%
# st(3)
# sl(6, 28)
# %%
lb0, lb1, lbf = fbu(t,j)
IC.hsae.saes[0].W_dec.shape
# %%
# IC.create_vocab_df(lbf)
# IC.create_vocab_df(lbf + lb1)
df = IC.create_vocab_df(lbf + lb1 + lb0)
pd.set_option('display.max_rows', 30)
# display(rows=30)
display(df)
# %%
DF_DISPLAY_TOP = None
def showboostedij(i, j=-1):
    b0, b1, f = fbu(i, j)
    if j == -1:
        df = IC.create_vocab_df(b1)
    df = IC.create_vocab_df(f + b1)
    if DF_DISPLAY_TOP is not None:
        display(df[:DF_DISPLAY_TOP])
    display(df)

def showboostedj(i, j):
    if j is None:
        j = -1
    b0, b1, f = fbu(i, j)
    df = IC.create_vocab_df(f)
    if DF_DISPLAY_TOP is not None:
        display(df[:DF_DISPLAY_TOP])
    display(df)


# %%
    
def slbij(i, j=-1):
    if j == -1:
        print(f"Boosted Logits for {i}:\tBIAS ONLY\t<MODELNAME={MODELNAME}>")
    else:
        print(f"Boosted Logits for {i}->{j}:\tbias+weight\t<MODELNAME={MODELNAME}>")
    showboostedij(i, j)

def slbj(i, j):
    print(f"Boosted Logits for {i}->{j}:\tweight only\t<MODELNAME={MODELNAME}>")
    showboostedj(i, j)

# for i in range(32):
#     slbij(6, i)
#     slbj(6, i)

def slb(i, j):
    slbij(i)
    slbij(i, j)
    slbj(i, j)


def sha(i, j, p=90):
    st(i)
    sl(i, j)
    slb(i, j)

def shll(i, j, p=90):
    sl(i, j)
    slbij(i, j)
    slbj(i, j)

sl(6, 26)
sha(6, 26)


# %%
b10, b11, f1 = fbu(6, 25)
b20, b21, f2 = fbu(6, 26)
d = IC.create_vocab_df(f2 + b21)
d[:30]
d = IC.create_vocab_df(f1 + b21)
d[:30]
# %%
sha(6, 25)
#%%
sa(62, p=95)
print("p=95")
st(62, p=95)
print("p=90")
st(62)

#%%

hsae = HierarchicalAutoEncoder.load(420, save_dir="/home/g/mats/sae/remote_scripts/models-from-remote")
hsae2 = HierarchicalAutoEncoder.load(915, save_dir="/home/g/mats/sae/remote_scripts/models-from-remote")

IC1 = HSAEInterpContext(hsae, skiptokens=True)
IC2 = HSAEInterpContext(hsae2, skiptokens=True)
#%%


# cosine similarities between top level feature and lower level bias
import matplotlib.pyplot as plt
tlw = F.normalize(IC1.hsae.sae_0.W_dec, dim=-1).detach().cpu()
llb = F.normalize(IC1.hsae.saes[0].b_dec, dim=-1).detach().cpu()
print(tlw.shape)
print(llb.shape)
cosim = tlw.T @ llb
#%%
plt.hist(cosim, bins= 5)

# %%
