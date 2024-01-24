from nqgl.sae.buffer import Buffer
import setup_utils
import sae.model as model
from calculations_on_sae import get_recons_loss, get_freqs, re_init
from transformer_lens import HookedTransformer
import train_sae
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig

import torch



def main():
    # ae_cfg = train_sae.ae_cfg
    # ae_cfg_z = sae.AutoEncoderConfig(site="z", act_size=512,
    #                                  l1_coeff=2e-3,
    #                                  nonlinearity=("undying_relu", {"l" : 0.001, "k" : 0.1}),
    #                                  lr=1e-4) #original 3e-4 8e-4 or same but 1e-3 on l1
    
    buffer.freshen_buffer(fresh_factor=2)

    ae = AutoEncoder.load(
        version=31,
        name="honest-glade-629")
    skip_ratio = 0.17
    # cfg = model.post_init_cfg(ae_cfg)
    cfg = ae.cfg

    model = setup_utils.get_model(cfg)
    all_tokens = setup_utils.load_data(model)
    encoder = AutoEncoder(cfg)
    # encoder = model.AutoEncoder.load_latest(new_cfg=cfg)
    # encoder = sae.AutoEncoder.load(14, save_dir="/root/workspace/")
    # encoder.cfg.gram_shmidt_trail = 500
    # encoder.cfg.num_to_resample = 64
    # linspace_l1(encoder, 0.2)

    buffer = Buffer(encoder.cfg, all_tokens, model=model)
    buffer.skip_first_tokens_ratio(skip_ratio)
    train_sae.train(encoder, encoder.cfg, buffer, model)


if __name__ == "__main__":
    main()
