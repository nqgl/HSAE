from nqgl.sae.training.buffer import Buffer
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig
from nqgl.sae.training.setup_utils import get_model, load_data
from nqgl.sae.training.calculations_on_sae import get_recons_loss
from transformer_lens import HookedTransformer
import wandb
import tqdm
import torch
import time
from typing import Optional
import math
WARMUP_STEPS = 10000


def train(
    encoder: HierarchicalAutoEncoder,
    cfg: HierarchicalAutoEncoderConfig,
    buffer: Buffer,
    model: Optional[HookedTransformer],
    project = "hsae_test",
    onecycle = None
):
    wandb.login(key="0cb29a3826bf031cc561fd7447767a3d7920d888", relogin=True)
    t0 = time.time()
    buffer.freshen_buffer(fresh_factor=2)
    try:
        run = wandb.init(project=project, entity="hsae_all", config=cfg)
        # run = wandb.init(project="autoencoders", entity="sae_all", config=cfg, mode="disabled")

        num_batches = cfg.num_tokens // cfg.batch_size
        # model_num_batches = cfg.model_batch_size * num_batches
        # encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        lr_adj = cfg.lr/25
        encoder_optim = torch.optim.Adam(
            encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        if onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                encoder_optim, 
                max_lr=cfg.lr, 
                steps_per_epoch=num_batches, 
                epochs=1,
                **onecycle)
        recons_scores = []
        act_freq_scores_list = []
        print("starting")
        l0_too_high = False
        i_ramp = 0
        for p in encoder_optim.param_groups:
            p["lr"] = (cfg.lr * i_ramp + lr_adj * (WARMUP_STEPS - i_ramp))  / WARMUP_STEPS


        for i in tqdm.trange(num_batches):

            acts = buffer.next()
            # if l0_too_high:
            #     # pass
            #     acts = acts[:32]
            x_reconstruct = encoder(
                acts,
                record_activation_frequency=True,
                rescaling=i < 10
                or (
                    i
                    < 10000
                    * cfg.batch_size
                    / cfg.buffer_mult
                    * cfg.buffer_refresh_ratio
                    and i % 100 == 0
                ),
                dense=l0_too_high,
            )

            x_reconstruct = encoder(
                acts,
                record_activation_frequency=True,
                rescaling=i < 10
                or (
                    i
                    < 10000
                    * cfg.batch_size
                    / cfg.buffer_mult
                    * cfg.buffer_refresh_ratio
                    and i % 100 == 0
                ),
            )

            # if i % 100 == 99:
            #     encoder.re_init_neurons_gram_shmidt(x.float() - x_reconstruct.float())
            loss = encoder.get_loss()
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            # scaler.step(encoder_optim)
            # scaler.update()
            encoder_optim.step()
            encoder_optim.zero_grad()
            if onecycle:
                scheduler.step()
            # if i % 40 == 11:
                # waiting = encoder.neurons_to_be_reset.shape[0]
                # wandb.log(
                #     {"neurons_waiting_to_reset": encoder.neurons_to_be_reset.shape[0]}
                # )
            # loss_dict = {
            #     "loss": loss.item(),
            #     "l2_loss": l2_loss.item(),
            #     "l1_loss": l1_loss.sum().item(),
            #     "l0_norm": l0_norm.item(),
            # }
            # if i % 100 == 0:
            #     if l0_norm.item() < 30:
            #         l0_too_high = False
            if (i) % 100 == 0:
                r1 = encoder.sae_0.neurons_to_be_reset
                r2 = encoder.layers[0].neurons_to_be_reset
                r1 = r1.shape[0] if r1 is not None else 0
                r2 = r2.shape[0] if r2 is not None else 0
                if r1 + r2 > 0:
                    encoder_optim = torch.optim.Adam(
                        encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
                    )
                    i_ramp = 0
                    if onecycle:
                        scheduler.optimizer = encoder_optim

                i_ramp += 100
                if i_ramp < WARMUP_STEPS or i % 1000 == 900:
                    if i_ramp > WARMUP_STEPS:
                        i_ramp = WARMUP_STEPS
                    topspeed = cfg.lr * (math.log(num_batches) - (999 * math.log(i) / 1000) ** 0.5) / math.log(num_batches)
                    nlr = (topspeed * i_ramp + lr_adj * (WARMUP_STEPS - i_ramp))  / WARMUP_STEPS
                    for p in encoder_optim.param_groups:
                        p["lr"] = nlr
                    wandb.log({
                        "lr":nlr
                    })
                    

                encoder.re_init_neurons()
                # torch.cuda.empty_cache()
                l2_loss = encoder.cached_l2_loss.mean()
                l1_loss = encoder.cached_l1_loss.mean()
                l0_norm = (
                    encoder.cached_l0_norm.mean()
                )  
                
                loss_dict = {
                    "loss": loss.item(),
                    "l2_loss": l2_loss.item(),
                }
                if onecycle:
                    loss_dict["lr"] = scheduler.get_last_lr()[0]
                wandb.log(loss_dict)
                print(loss_dict, run.name)
                queued = encoder.sae_0.neurons_to_be_reset.shape[0] if encoder.sae_0.neurons_to_be_reset is not None else 0
                wandb.log(
                    {
                        f"SAE_0 l1_loss": l1_loss.sum().item(),
                        f"SAE_0 l0_norm": l0_norm.item(),
                        f"SAE_0 queued for reset": queued,
                    }
                )

                # del loss, x_reconstruct, l2_loss, l1_loss, acts, l0_norm
                for j in range(len(encoder.layers)):
                    sae = encoder.layers[j]
                    l1_loss = encoder.layers[j].cached_l1_loss
                    l0_norm = encoder.layers[j].cached_l0_norm
                    queued = sae.neurons_to_be_reset.shape[0] if sae.neurons_to_be_reset is not None else 0
                    wandb.log(
                        {
                            f"SAE_{j + 1} l1_loss": l1_loss.sum().item(),
                            f"SAE_{j + 1} l0_norm": l0_norm.item(),
                            f"SAE_{j + 1} queued for reset": queued,
                        }
                    )

            if i % 1000 == 0:
                wandb.log({"mse_contribs": encoder.loss_contributions(acts)})
            if (i) % 2000 == 0 and not l0_too_high and not model is None:
                x = get_recons_loss(
                    model, encoder, buffer, local_encoder=encoder, num_batches=1
                )
                print("Reconstruction:", x)
                recons_scores.append(x[0])

                # freqs = encoder.neuron_activation_frequency / encoder.steps_since_activation_frequency_reset
                # act_freq_scores_list.append(freqs)
                # histogram(freqs.log10(), marginal="box",h istnorm="percent", title="Frequencies")
                wandb.log(
                    {
                        "recons_score": x[0],
                        #     "dead": (freqs==0).float().mean().item(),
                        #     "below_1e-6": (freqs<1e-6).float().mean().item(),
                        #     "below_1e-5": (freqs<1e-5).float().mean().item(),
                            "time spent shuffling": buffer.time_shuffling,
                            "total time" : time.time() - t0,
                    }
                )
                wandb.log(
                    encoder.histograms()
                )
            if i == 3100:
                encoder.reset_activation_frequencies()
            elif i % 50000 == 3100 and i > 1500:
                encoder.save(name=run.name)
                # TODO resample
                # freqs = encoder.neuron_activation_frequency / encoder.steps_since_activation_frequency_reset
                encoder.resampling_check()
                # print("Resetting neurons!", to_be_reset.sum())
                #     # re_init(model, encoder, buffer, to_be_reset)
                # wandb.log({"reset_neurons": to_be_reset.sum(), "time_for_neuron_reset": time.time() - t1})
                # encoder.reset_activation_frequencies()
    finally:
        encoder.save()


def linspace_l1(ae, l1_radius):
    cfg = ae.cfg
    l1 = torch.linspace(
        cfg.l1_coeff * (1 - l1_radius),
        cfg.l1_coeff * (1 + l1_radius),
        cfg.dict_size,
        device=cfg.device,
    )
    ae.l1_coeff = l1


cfg = HierarchicalAutoEncoderConfig(
    site="resid_pre",
    d_data=512,
    layer=1,
    gram_shmidt_trail=512,
    num_to_resample=4,
    l1_coeff=1e-4,
    dict_mult=1,
    batch_size=256,
    beta2=0.999,
    nonlinearity=("relu", {}),
    flatten_heads=False,
    buffer_mult=128,
    buffer_refresh_ratio=0.5000,
    lr=1e-4,
    features_per_sae_per_layer=[None, 32],
    data_rescale=20**0.5,
    layer_cfg_params={"l1_coeff":3e-4, "data_rescale" : 1},
    gate_mode="acts",
    # lower_layer_act_penalize_pre_gate = True,

)


def main():
    # cfg = sae.post_init_cfg(ae_cfg)
    model = get_model(cfg)
    all_tokens = load_data(model)
    # sae = AutoEncoder.load_latest()
    encoder = HierarchicalAutoEncoder(cfg)

    # linspace_l1(encoder, 0.2)
    # dataloader, buffer = buffer_dataset.get_dataloader(cfg, all_tokens, model=model, device=torch.devi1ce("cpu"))
    # print(buffer.device)
    # buffer = buffer_dataset.BufferRefresher(cfg, all_tokens, model, device="cuda")
    buffer = Buffer(cfg, all_tokens, model=model)
    train(encoder, cfg, buffer, model)


if __name__ == "__main__":
    main()
