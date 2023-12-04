from . import z_sae
import wandb
import tqdm
import torch
from .calculations_on_sae import get_recons_loss, get_freqs, re_init
from transformer_lens import HookedTransformer
import time

def train(encoder :z_sae.AutoEncoder, cfg :z_sae.AutoEncoderConfig, buffer :z_sae.Buffer, model :HookedTransformer):
    wandb.login(key="0cb29a3826bf031cc561fd7447767a3d7920d888", relogin=True)
    t0 = time.time()
    buffer.freshen_buffer()
    scaler = torch.cuda.amp.GradScaler()

    try:
        run = wandb.init(project="autoencoders", entity="sae_all", config=cfg)
        
        # run = wandb.init(project="autoencoders", entity="sae_all", config=cfg, mode="disabled")

        num_batches = cfg.num_tokens // cfg.batch_size
        # model_num_batches = cfg.model_batch_size * num_batches
        encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        recons_scores = []
        act_freq_scores_list = []
        for i in tqdm.trange(num_batches):
            # i = i % buffer.all_tokens.shape[0]
            acts = buffer.next()
            x_reconstruct = encoder(acts, record_activation_frequency=True)
            loss = encoder.get_loss()
            l2_loss = encoder.l2_loss_cached
            l1_loss = encoder.l1_loss_cached
            l0_norm = encoder.l0_norm_cached # TODO condisder turning this off if is slows down calculation
            # scaler.scale(loss).backward()
            loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            # scaler.step(encoder_optim)
            # scaler.update()
            encoder_optim.step()
            encoder_optim.zero_grad()
            loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.sum().item(), "l0_norm": l0_norm.item()}
            del loss, x_reconstruct, l2_loss, l1_loss, acts, l0_norm
            if (i) % 100 == 0:
                wandb.log(loss_dict)
                print(loss_dict, run.name)
            if (i) % 5000 == 1499:
                x = (get_recons_loss(model, encoder, buffer, local_encoder=encoder))
                print("Reconstruction:", x)
                recons_scores.append(x[0])
                freqs = get_freqs(model, encoder, buffer, 5, local_encoder=encoder)
                act_freq_scores_list.append(freqs)
                # histogram(freqs.log10(), marginal="box",h istnorm="percent", title="Frequencies")
                wandb.log({
                    "recons_score": x[0],
                    "dead": (freqs==0).float().mean().item(),
                    "below_1e-6": (freqs<1e-6).float().mean().item(),
                    "below_1e-5": (freqs<1e-5).float().mean().item(),
                    "time spent shuffling": buffer.time_shuffling,
                    "total time" : time.time() - t0,
                })
            if (i+1) % 10000 == 1500 and i > 1500:
                encoder.save(name=run.name)
                t1 = time.time()
                # freqs = get_freqs(model, encoder, buffer, 50, local_encoder=encoder)
                freqs = encoder.activation_frequency / encoder.steps_since_activation_frequency_reset
                to_be_reset = (freqs<10**(-5.5))
                print("Resetting neurons!", to_be_reset.sum())
                if to_be_reset.sum() > 0:
                    re_init(model, encoder, buffer, to_be_reset)
                wandb.log({"reset_neurons": to_be_reset.sum(), "time_for_neuron_reset": time.time() - t1})
                encoder.reset_activation_frequencies()
    finally:
        encoder.save()

l1_coeff_list = [1e-3, 15e-4, 12e-4]

def linspace_l1(ae, l1_radius):
    cfg = ae.cfg
    l1 = torch.linspace(cfg.l1_coeff * (1 - l1_radius), cfg.l1_coeff * (1 + l1_radius), cfg.dict_size, device=cfg.device)
    ae.l1_coeff = l1

def main():

    ae_cfg = z_sae.AutoEncoderConfig(site="z", act_size=512, 
                                    l1_coeff=3e-3, dict_mult=16, batch_size=512,
                                    nonlinearity=("relu",{}), flatten_heads=True,
                                    lr=1e-4) #original 3e-4 8e-4 or same but 1e-3 on l1
    # ae_cfg_z = z_sae.AutoEncoderConfig(site="z", act_size=512, 
    #                                  l1_coeff=2e-3,
    #                                  nonlinearity=("undying_relu", {"l" : 0.001, "k" : 0.1}), 
    #                                  lr=1e-4) #original 3e-4 8e-4 or same but 1e-3 on l1
    cfg = z_sae.post_init_cfg(ae_cfg)
    model = z_sae.get_model(cfg)
    all_tokens = z_sae.load_data(model)
    encoder = z_sae.AutoEncoder(cfg)
    # linspace_l1(encoder, 0.1)

    buffer = z_sae.Buffer(cfg, all_tokens, model=model)
    train(encoder, cfg, buffer, model)

if __name__ == "__main__":
    main()