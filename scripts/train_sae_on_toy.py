from nqgl.sae.training.buffer import Buffer
from nqgl.sae.sae.model import AutoEncoder, AutoEncoderConfig
from nqgl.sae.training.setup_utils import get_model, load_data
from nqgl.sae.training.calculations_on_sae import get_recons_loss
from transformer_lens import HookedTransformer
from toy_models.toy_model import ToyModel, ToyModelConfig
from analysis.visualize_features import visualize_by_heatmap
import wandb
import tqdm
import torch
import time


def train(encoder: AutoEncoder, cfg: AutoEncoderConfig, buffer: ToyModel):
    wandb.login(key="0cb29a3826bf031cc561fd7447767a3d7920d888", relogin=True)
    t0 = time.time()
    try:
        run = wandb.init(
            project="features_toy_model",
            entity="sae_all",
            config={"encoder": cfg, "toy": buffer.cfg},
            mode="disabled",
        )

        num_batches = cfg.num_tokens // cfg.batch_size
        encoder_optim = torch.optim.Adam(
            encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2)
        )
        recons_scores = []
        act_freq_scores_list = []
        for i in tqdm.trange(num_batches):
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
            loss = encoder.get_loss()
            l2_loss = encoder.cached_l2_loss.mean()
            l1_loss = encoder.cached_l1_loss.mean()
            l0_norm = (
                encoder.cached_l0_norm.mean()
            )  # TODO condisder turning this off if is slows down calculation
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
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
            if i % 500 == 0:
                visualize_by_heatmap(buffer, encoder)
            if (i) % 5000 == 0:
                freqs = (
                    encoder.neuron_activation_frequency
                    / encoder.steps_since_activation_frequency_reset
                )
                act_freq_scores_list.append(freqs)
                # histogram(freqs.log10(), marginal="box",h istnorm="percent", title="Frequencies")
                wandb.log(
                    {
                        "dead": (freqs == 0).float().mean().item(),
                        "below_1e-6": (freqs < 1e-6).float().mean().item(),
                        "below_1e-5": (freqs < 1e-5).float().mean().item(),
                        "total time": time.time() - t0,
                    }
                )
            if i == 13501:
                encoder.reset_activation_frequencies()
            elif i % 15000 == 13501 and i > 1500:
                encoder.save(name=run.name)
                t1 = time.time()
                freqs = (
                    encoder.neuron_activation_frequency
                    / encoder.steps_since_activation_frequency_reset
                )
                to_be_reset = freqs < 10 ** (-5.5)
                print("Resetting neurons!", to_be_reset.sum())
                if to_be_reset.sum() > 0:
                    encoder.queue_neurons_to_reset(to_be_reset)
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
    site="toy_model",
    d_data=64,
    layer=1,
    gram_shmidt_trail=512,
    num_to_resample=4,
    l1_coeff=14e-4,
    dict_mult=1,
    batch_size=128,
    beta2=0.999,
    lr=3e-4,
)


def main():
    encoder = AutoEncoder(cfg)
    toycfg = ToyModelConfig(
        d_data=cfg.d_data,
        n_features=32,
        num_correlation_rounds=10,
        batch_size=cfg.batch_size,
        blank_correlations=False,
        initial_features=3,
        features_per_round=2,
        features_per_round_negative=4,
        seed=50,
        correlation_drop=0.9,
        source_prob_drop=0.85,
    )

    toy = ToyModel(toycfg)
    toy.f_means[:] = 1
    train(encoder, cfg, toy)


if __name__ == "__main__":
    main()
