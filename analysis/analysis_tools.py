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
from nqgl.sae.buffer import Buffer
from nqgl.sae.sae.base import BaseSAE
from typing import Optional
# from utils_from_others import make_token_df



SPACE = "·"
NEWLINE="↩"
TAB = "→"
# from Neel's code
def list_flatten(nested_list):
    return [x for y in nested_list for x in y]

class InterpContext:
    def __init__(
        self, 
        encoder :BaseSAE, 
        model: Optional[HookedTransformer]=None, 
        all_tokens = None, 
        skiptokens=False
    ):
        self.encoder = encoder
        self.model :HookedTransformer = get_model(encoder.cfg) \
                                        if model is None else model
        
        self.all_tokens =   load_data(self.model) \
                            if all_tokens is None \
                            and not skiptokens \
                            else all_tokens
        

    # this following part of the code is from Neel's notebooks I believe, 
        #Bart may have added some of this. 
        # then I have adapted it to this format
    def process_token(self, s):
        if isinstance(s, torch.Tensor):
            s = s.item()
        if isinstance(s, np.int64):
            s = s.item()
        if isinstance(s, int):
            s = self.model.to_string(s)
        s = s.replace(" ", SPACE)
        s = s.replace("\n", NEWLINE+"\n")
        s = s.replace("\t", TAB)
        return s

    def process_tokens(self, l):
        if isinstance(l, str):
            l = self.model.to_str_tokens(l)
        elif isinstance(l, torch.Tensor) and len(l.shape)>1:
            l = l.squeeze(0)
        return [self.process_token(s) for s in l]

    def process_tokens_index(self, l):
        if isinstance(l, str):
            l = self.model.to_str_tokens(l)
        elif isinstance(l, torch.Tensor) and len(l.shape)>1:
            l = l.squeeze(0)
        return [f"{self.process_token(s)}/{i}" for i,s in enumerate(l)]

    def create_vocab_df(self, logit_vec, make_probs=False, full_vocab=None):
        if full_vocab is None:
            full_vocab = self.process_tokens(self.model.to_str_tokens(torch.arange(self.model.cfg.d_vocab)))
        vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
        if make_probs:
            vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
            vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
        return vocab_df.sort_values("logit", ascending=False)

    
    def make_token_df(self, tokens, len_prefix=5, len_suffix=1):
        str_tokens = [self.process_tokens(self.model.to_str_tokens(t)) for t in tokens]
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




class HSAEInterpContext(InterpContext):
    def __init__(self,
            encoder  :HierarchicalAutoEncoder, 
            model: Optional[HookedTransformer]=None, 
            all_tokens = None, 
            skiptokens=False,
            features_on_cpu = True
        ):

        super().__init__(encoder, model, all_tokens, skiptokens)
        self.feat_acts_sae0 :torch.Tensor = None
        self.feat_acts_sae1 :torch.Tensor = None
        self.feat_act_tokens = None
        self.act_device = "cpu" if features_on_cpu else "cuda"
        self.hsae = encoder



    def get_activations_on_tokens(self, tokens):
        batch_size = self.hsae.cfg.model_batch_size
        number_of_batches = tokens.shape[0] // self.hsae.cfg.model_batch_size #+ (tokens.shape[0] % batch_size != 0)
        model_act_batch_size = tokens.shape[1] * self.hsae.cfg.model_batch_size

        hidden_acts_sae0 = torch.zeros(tokens.shape[0] * tokens.shape[1], self.hsae.sae_0.cfg.d_dict, device=self.act_device)
        hidden_acts_sae1 = torch.zeros(tokens.shape[0] * tokens.shape[1], self.hsae.saes[0].cfg.n_sae, self.hsae.saes[0].cfg.d_dict, device=self.act_device)
        for i in range(number_of_batches):
            print(F"{i}/{number_of_batches}")
            with torch.no_grad():
                k = i * model_act_batch_size
                input_tokens = tokens[
                    i * batch_size : (i + 1) * batch_size
                ]
                _, cache = self.model.run_with_cache(input_tokens, stop_at_layer=self.hsae.cfg.layer+1)
                mlp_acts = cache[self.hsae.cfg.act_name]
                model_acts = mlp_acts.reshape(-1, self.hsae.cfg.act_size)
                # acts = buffer.next()
                # buffer.refresh()
                # input_tokens = buffer.all_tokens[
                #             buffer.token_pointer  - buffer.self.hsae.cfg.model_batch_size : buffer.token_pointer
                #         ]
                # tokens += input_tokens
                print(model_act_batch_size, model_acts.shape[0])
                assert model_act_batch_size == model_acts.shape[0]
                # hidden_acts_sae0 = torch.zeros(model_act_batch_size, self.hsae.sae_0.cfg.d_dict)
                # hidden_acts_sae1 = torch.zeros(model_act_batch_size, self.hsae.saes[0].cfg.n_sae, self.hsae.saes[0].cfg.d_dict).to(self.hsae.cfg.device)
                for j in range(model_act_batch_size // self.hsae.cfg.batch_size):
                    self.hsae(model_acts[j*self.hsae.cfg.batch_size:(j+1)*self.hsae.cfg.batch_size])
                    hidden_acts_sae0[j*self.hsae.cfg.batch_size + k:(j+1)*self.hsae.cfg.batch_size + k] = self.hsae.sae_0.cached_acts
                    hidden_acts_sae1[j*self.hsae.cfg.batch_size + k:(j+1)*self.hsae.cfg.batch_size + k] = self.hsae.saes[0].cached_acts
                hidden_acts_sae0.to(self.act_device)
                hidden_acts_sae1.to(self.act_device)
        return hidden_acts_sae0, hidden_acts_sae1
                
    def populate_for_analysis(self, num_batches = 40):
        tokens = self.all_tokens[:32*num_batches]
        self.feat_act_tokens = tokens
        self.feat_acts_sae0, self.feat_acts_sae1 = self.get_activations_on_tokens(tokens)


    def show_random_highly_activating(self, i0, i1 = None, percentile = 90):
        if i1 is None:
            feature_acts = self.feat_acts_sae0[:, i0]
        else:
            feature_acts = self.feat_acts_sae1[:, i0, i1]
        try:
            self.show_random_highly_activating_from_tf(self.feat_act_tokens.cpu().numpy(), feature_acts.cpu().numpy(), percentile=percentile)
            return True
        except:
            return False

    def freqs(self):
        freq_per_feature = (hidden_acts > 0).mean(axis=0)

    def top_freqs(self):
        high_data_features = np.arange(0, freq_per_feature.shape[0])[np.argsort(freq_per_feature)[::-1][:100]]





    def show_random_highly_activating_from_tf(self, tokens, feature_acts, percentile=90):

        token_df = self.make_token_df(tokens, len_prefix=10, len_suffix=3)
        token_df["feature"] = utils.to_numpy(feature_acts)
        #select all where feature > 0.0
        token_df = token_df[token_df["feature"]>0]
        #get the 50th percentile
        percentile = np.percentile(token_df["feature"], percentile)
        #select all where feature > 50th percentile/usr/local/lib/python3.8/dist-packages/pandas/core/common.py
        display(token_df[token_df["feature"]>percentile].sample(10).style.background_gradient("coolwarm"))


        # else:
            # break
    # %%




    def get_feature(self, i, j):
        return self.hsae.saes[0].W_dec[i, j]



    def logits_boosted_for_feature(self, f):
        self.model.unembed()

    def show_boosted_logits(self, i, j):
        pass
