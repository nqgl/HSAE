from nqgl.sae.buffer import Buffer
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig
from nqgl.sae.setup_utils import get_model, load_data
from calculations_on_sae import get_recons_loss
from transformer_lens import HookedTransformer
import wandb
import tqdm
import torch
import time
from typing import Optional
from train_hsae import train
from dataclasses import asdict

blank_slate_just_gate = {

}
init_to_same_performance = {

}


def main():
    try:
        sae = AutoEncoder.load_latest(name="celestial-firebrand")
    except:
        print("not on machine 1, trying second sae")
        try:
            sae = AutoEncoder.load_latest(name="earnest-fire")
        except:
            try:
                sae = AutoEncoder.load_latest(name="breezy-aardvark")
            except:
                try:
                    sae = AutoEncoder.load_latest(name="toasty-planet")
                except:
                    # sae = AutoEncoder.load_latest(name="breezy-aardvark")
                    raise Exception("failed")



    # sae.

    cfgs = dict(
        gram_shmidt_trail=512,
        num_to_resample=4,
        l1_coeff=0, #does nothing here
        # dict_mult=1,
        batch_size=256,
        beta2=0.999,
        nonlinearity=("relu", {}),
        flatten_heads=False,
        buffer_mult=128 * 16 * 7,
        buffer_refresh_ratio=0.2,
        lr=3e-5,
        features_per_sae_per_layer=[None, 32],
        # data_rescale=20**0.5,
        layer_cfg_params={"l1_coeff":19e-5, "data_rescale" : 1},
        gate_mode="binary",
        train_on_residuals = False,
        
    )
    cfg = HierarchicalAutoEncoderConfig(
        **{
            **asdict(sae.cfg),
            **cfgs
        }
    
    )
    # lower_layer_act_penalize_pre_gate = True,
    
    print(cfg)
    model = get_model(cfg)
    all_tokens = load_data(model)
    encoder = HierarchicalAutoEncoder(cfg, sae0=sae)
    encoder.freeze_sae0()

    # linspace_l1(encoder, 0.2)
    # dataloader, buffer = buffer_dataset.get_dataloader(cfg, all_tokens, model=model, device=torch.devi1ce("cpu"))
    # print(buffer.device)
    # buffer = buffer_dataset.BufferRefresher(cfg, all_tokens, model, device="cuda")
    buffer = Buffer(cfg, all_tokens, model=model)
    train(encoder, cfg, buffer, model, project="hsae_frozen_sae0")


if __name__ == "__main__":
    main()