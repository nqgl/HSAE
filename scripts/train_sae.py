from nqgl.sae.training.buffer import Buffer
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from nqgl.sae.training.setup_utils import get_model, load_data
from nqgl.sae.training.calculations_on_sae import get_recons_loss
from transformer_lens import HookedTransformer

import wandb
import tqdm
import torch
import time


def train(
    encoder: AutoEncoder,
    cfg: AutoEncoderConfig,
    buffer: Buffer,
    model: HookedTransformer,
):
    wandb.login(key="0cb29a3826bf031cc561fd7447767a3d7920d888", relogin=True)
    t0 = time.time()
    # buffer.freshen_buffer(fresh_factor=0.5)
    try:
        run = wandb.init(project="autoencoders", entity="sae_all", config=cfg)
        # run = wandb.init(project="autoencoders", entity="sae_all", config=cfg, mode="disabled")

        num_batches = cfg.num_tokens // cfg.batch_size
        # model_num_batches = cfg.model_batch_size * num_batches
        # encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        encoder_optim = torch.optim.Adam(
            encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        recons_scores = []
        act_freq_scores_list = []
        for i in tqdm.trange(num_batches):
            # i = i % buffer.all_tokens.shape[0]
            acts = buffer.next()
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
            l2_loss = encoder.cached_l2_loss.mean()
            l1_loss = encoder.cached_l1_loss.mean()
            l0_norm = (
                encoder.cached_l0_norm.mean()
            )  # TODO condisder turning this off if is slows down calculation
            # scaler.scale(loss).backward()
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            # scaler.step(encoder_optim)
            # scaler.update()
            encoder_optim.step()
            encoder_optim.zero_grad()
            if i % 200 == 99 and encoder.neurons_to_be_reset is not None:
                waiting = encoder.neurons_to_be_reset.shape[0]
                wandb.log(
                    {"neurons_waiting_to_reset": encoder.neurons_to_be_reset.shape[0]}
                )
                encoder.re_init_neurons(acts.float() - x_reconstruct.float())
                if encoder.neurons_to_be_reset is not None:
                    num_reset = waiting - encoder.neurons_to_be_reset.shape[0]
                else:
                    num_reset = waiting
                wandb.log({"neurons_reset": num_reset})
                encoder_optim = torch.optim.Adam(
                    encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
                )
                torch.cuda.empty_cache()
            loss_dict = {
                "loss": loss.item(),
                "l2_loss": l2_loss.item(),
                "l1_loss": l1_loss.sum().item(),
                "l0_norm": l0_norm.item(),
            }
            del loss, x_reconstruct, l2_loss, l1_loss, acts, l0_norm
            if (i) % 100 == 0:
                wandb.log(loss_dict)
                print(loss_dict, run.name)
            if i % 5000 == 1:
                x = get_recons_loss(
                    model, encoder, buffer, local_encoder=encoder, num_batches=1
                )
                print("Reconstruction:", x)
                recons_scores.append(x[0])

                # freqs = get_freqs(model, encoder, buffer, 5, local_encoder=encoder)
                freqs = (
                    encoder.neuron_activation_frequency
                    / encoder.steps_since_activation_frequency_reset
                )
                act_freq_scores_list.append(freqs)
                # histogram(freqs.log10(), marginal="box",h istnorm="percent", title="Frequencies")
                wandb.log(
                    {
                        "recons_score": x[0],
                        "dead": (freqs == 0).float().mean().item(),
                        "below_1e-6": (freqs < 1e-6).float().mean().item(),
                        "below_1e-5": (freqs < 1e-5).float().mean().item(),
                        "time spent shuffling": buffer.time_shuffling,
                        "total time": time.time() - t0,
                    }
                )
            if i == 13501:
                encoder.reset_activation_frequencies()
            elif i % 15000 == 13501 and i > 1500:
                encoder.save(name=run.name)
                t1 = time.time()
                # freqs = get_freqs(model, encoder, buffer, 50, local_encoder=encoder)
                freqs = (
                    encoder.neuron_activation_frequency
                    / encoder.steps_since_activation_frequency_reset
                )
                to_be_reset = freqs < 10 ** (-5.5)
                print("Resetting neurons!", to_be_reset.sum())
                if to_be_reset.sum() > 0:
                    encoder.queue_neurons_to_reset(to_be_reset)
                    # re_init(model, encoder, buffer, to_be_reset)
                wandb.log(
                    {
                        "reset_neurons": to_be_reset.sum(),
                        "time_for_neuron_reset": time.time() - t1,
                    }
                )
                encoder.reset_activation_frequencies()
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


cfg = AutoEncoderConfig(
    site="resid_pre",
    d_data=512,
    layer=1,
    gram_shmidt_trail=512,
    num_to_resample=4,
    l1_coeff=14e-4,
    dict_mult=64,
    batch_size=512,
    beta2=0.999,
    subshuffle=16,
    nonlinearity=("relu", {}),
    flatten_heads=False,
    buffer_mult=128 * 16 * 7,
    buffer_refresh_ratio=0.25,
    lr=3e-5,
    data_rescale=20 ** 0.5
)  # original 3e-4 8e-4 or same but 1e-3 on l1


# parameters for the run that was looking pretty good
# cfg = AutoEncoderConfig(
#     site="resid_pre",
#     d_data=512,
#     layer=1,
#     gram_shmidt_trail=512,
#     num_to_resample=4,
#     l1_coeff=10e-4,
#     dict_mult=2,
#     batch_size=1024,
#     beta2=0.999,
#     subshuffle=16,
#     nonlinearity=("relu", {}),
#     flatten_heads=False,
#     buffer_mult=128 * 16 * 7,
#     buffer_refresh_ratio=0.25,
#     lr=1e-4,
#     data_rescale=20 ** 0.5
# )  # original 3e-4 8e-4 or same but 1e-3 on l1


def main():
    # cfg = sae.post_init_cfg(ae_cfg)
    model = get_model(cfg)
    all_tokens = load_data(model)
    encoder = AutoEncoder(cfg)
    # linspace_l1(encoder, 0.2)
    # dataloader, buffer = buffer_dataset.get_dataloader(cfg, all_tokens, model=model, device=torch.device("cpu"))
    # print(buffer.device)
    # buffer = buffer_dataset.BufferRefresher(cfg, all_tokens, model, device="cuda")
    buffer = Buffer(cfg, all_tokens, model=model)
    train(encoder, cfg, buffer, model)


if __name__ == "__main__":
    main()
