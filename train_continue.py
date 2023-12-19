from buffer import Buffer
import setup_utils
import sae.model as model
from calculations_on_sae import get_recons_loss, get_freqs, re_init
from transformer_lens import HookedTransformer
import train_sae

import torch


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
    # ae_cfg_z = sae.AutoEncoderConfig(site="z", act_size=512, 
    #                                  l1_coeff=2e-3,
    #                                  nonlinearity=("undying_relu", {"l" : 0.001, "k" : 0.1}), 
    #                                  lr=1e-4) #original 3e-4 8e-4 or same but 1e-3 on l1
    skip_ratio = 0.08
    cfg = model.post_init_cfg(ae_cfg)
    model = setup_utils.get_model(cfg)
    all_tokens = setup_utils.load_data(model)
    encoder = model.AutoEncoder.load_latest(new_cfg = cfg)
    # encoder = sae.AutoEncoder.load(14, save_dir="/root/workspace/")
    # encoder.cfg.gram_shmidt_trail = 500
    # encoder.cfg.num_to_resample = 64
    # linspace_l1(encoder, 0.2)

    buffer = Buffer(encoder.cfg, all_tokens, model=model)
    buffer.skip_first_tokens_ratio(skip_ratio)
    train_sae.train(encoder, encoder.cfg, buffer, model)

if __name__ == "__main__":
    main()