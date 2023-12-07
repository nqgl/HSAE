from . import z_sae
import wandb
import tqdm
import torch
from .calculations_on_sae import get_recons_loss, get_freqs, re_init
from transformer_lens import HookedTransformer
import time
from . import train_sae


def linspace_l1(ae, l1_radius):
    cfg = ae.cfg
    l1 = torch.linspace(cfg.l1_coeff * (1 - l1_radius), cfg.l1_coeff * (1 + l1_radius), cfg.dict_size, device=cfg.device)
    ae.l1_coeff = l1

# conversions after fixing the sums over batch size
    # pre batch size reduction: multiply l1 by 256
    # post batch size reduction: divide l1 by 128
# this might not be the case either because l2 is now meaned too
# so like, l2 /= 512 
#          l1 /= 
# so maybe increase lr by 1-2.5 oom?
# l1 coeff prevv got multiplied by 128 - 256 but then l2 was like 256 times too
    # for l1 to get similar gradients, 
    
def main():

    ae_cfg = train_sae.ae_cfg
    # ae_cfg_z = z_sae.AutoEncoderConfig(site="z", act_size=512, 
    #                                  l1_coeff=2e-3,
    #                                  nonlinearity=("undying_relu", {"l" : 0.001, "k" : 0.1}), 
    #                                  lr=1e-4) #original 3e-4 8e-4 or same but 1e-3 on l1
    skip_ratio = 0.75
    cfg = z_sae.post_init_cfg(ae_cfg)
    model = z_sae.get_model(cfg)
    all_tokens = z_sae.load_data(model)
    encoder = z_sae.AutoEncoder.load_latest(new_cfg = cfg)
    # linspace_l1(encoder, 0.2)

    buffer = z_sae.Buffer(cfg, all_tokens, model=model)
    buffer.skip_first_tokens_ratio(skip_ratio)
    train_sae.train(encoder, cfg, buffer, model)

if __name__ == "__main__":
    main()