#%%
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig
import torch
model = HierarchicalAutoEncoder.load_latest()
encoder_optim = torch.optim.Adam(
            model.parameters(), lr=3e-4, betas=(0.9, 0.999)
        )

for param in model.named_parameters():
    if param[0] == 'layer_of_interest.weight':  # Replace with your layer name
        exp_avg = encoder_optim.state[param[1]].get('exp_avg')
        if exp_avg is not None:
            exp_avg[0] = 0
# %%
[x[0] for x in model.named_parameters()]

# need to pull out *.W_dec *b_enc, etc in re-init
# %%
