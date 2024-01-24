#%%
import torch
from nqgl.sae.toy_models.toy_model import ToyModel, ToyModelConfig
from nqgl.sae.toy_models.two_features import get_simple_hierarchy_model
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from nqgl.sae.hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig
from dataclasses import asdict



# %%
sae0 = AutoEncoder.load_latest(name="celestial-firebrand")

hcfg = HierarchicalAutoEncoderConfig(**asdict(sae0.cfg))

hsae = HierarchicalAutoEncoder(hcfg, sae0)
hsae.freeze_sae0()
d = sae0.cfg.d_data
optim = torch.optim.SGD(hsae.parameters(), lr=0.01)
x = torch.rand(1, d).cuda()

x_re = hsae(x)
((x - x_re)**2).sum().backward()
v_sae = hsae.saes[0].W_dec.data.clone().detach()
v_sae0 = hsae.sae_0.W_dec.data.clone().detach()

optim.step()

print(v_sae==hsae.saes[0].W_dec.data)
print(torch.all(v_sae0==hsae.sae_0.W_dec.data))


# %%
