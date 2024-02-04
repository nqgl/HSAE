from nqgl.sae.training.buffer import Buffer
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from hsae.hsae import HierarchicalAutoEncoder, HierarchicalAutoEncoderConfig
from nqgl.sae.training.setup_utils import get_model, load_data
from nqgl.sae.training.calculations_on_sae import get_recons_loss
from transformer_lens import HookedTransformer
from toy_models.toy_model import ToyModel, ToyModelConfig
import wandb
import tqdm
import torch
import time


def train(
    encoder: HierarchicalAutoEncoder,
    cfg: HierarchicalAutoEncoderConfig,
    buffer: ToyModel,
):
    wandb.login(key="0cb29a3826bf031cc561fd7447767a3d7920d888", relogin=True)
    t0 = time.time()
    # buffer.freshen_buffer(fresh_factor=0.5)
    try:
        run = wandb.init(project="hsae_toy_model", entity="hsae_all", config=cfg)

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
            if i % 200 == 99:
                # waiting = encoder.neurons_to_be_reset.shape[0]
                # wandb.log(
                #     {"neurons_waiting_to_reset": encoder.neurons_to_be_reset.shape[0]}
                # )
                encoder.re_init_neurons()
                encoder_optim = torch.optim.Adam(
                    encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
                )
                torch.cuda.empty_cache()
            if (i) % 100 == 0:
                loss_dict = {
                    "loss": loss.item(),
                    "l2_loss": l2_loss.item(),
                }
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

                for i in range(len(encoder.layers)):
                    sae = encoder.layers[i]
                    l1_loss = encoder.layers[i].cached_l1_loss
                    l0_norm = encoder.layers[i].cached_l0_norm
                    queued = sae.neurons_to_be_reset.shape[0] if sae.neurons_to_be_reset is not None else 0
                    wandb.log(
                        {
                            f"SAE_{i + 1} l1_loss": l1_loss.sum().item(),
                            f"SAE_{i + 1} l0_norm": l0_norm.item(),
                            f"SAE_{i + 1} queued for reset": queued,
                        }
                    )

            if i % 1000 == 0:
                wandb.log({"mse_contribs": encoder.loss_contributions(acts)})
            del loss, x_reconstruct, l2_loss, l1_loss, acts, l0_norm

            if i == 131:
                encoder.reset_activation_frequencies()
            elif i % 350 == 131 and i > 1500:
                encoder.save(name=run.name)
                t1 = time.time()
                # freqs = get_freqs(model, encoder, buffer, 50, local_encoder=encoder)
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
    d_data=16,
    layer=1,
    gram_shmidt_trail=512,
    num_to_resample=4,
    l1_coeff=5e-2,
    dict_mult=2,
    batch_size=8,
    beta2=0.999,
    nonlinearity=("relu", {}),
    flatten_heads=False,
    buffer_mult=128 * 8,
    buffer_refresh_ratio=0.25,
    lr=1e-3,
    layer_cfg_params={"l1_coeff" : 1},
    neuron_death_min_samples=1,
    dead_threshold=0.01,
    features_per_sae_per_layer=[None, 9]
)  # original 3e-4 8e-4 or same but 1e-3 on l1


def main():
    # cfg = sae.post_init_cfg(ae_cfg)
    # sae = AutoEncoder.load_latest()
    encoder = HierarchicalAutoEncoder(cfg)

    # linspace_l1(encoder, 0.2)
    # dataloader, buffer = buffer_dataset.get_dataloader(cfg, all_tokens, model=model, device=torch.device("cpu"))
    # print(buffer.device)
    # buffer = buffer_dataset.BufferRefresher(cfg, all_tokens, model, device="cuda")
    toycfg = ToyModelConfig(
        d_data=cfg.d_data,
        n_features=64,
        num_correlation_rounds=2,
        batch_size=cfg.batch_size,
    )
    toy = ToyModel(toycfg)
    from nqgl.sae.train_hsae import train as htrain
    htrain(encoder, cfg, toy, model=None, project = "hsae_toy_model")
    train(encoder, cfg, toy)



if __name__ == "__main__":
    main()
