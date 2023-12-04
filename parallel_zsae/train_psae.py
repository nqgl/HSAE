from . import z_psae
import wandb
import tqdm
import torch
from .calculations_on_psae import get_recons_loss, get_freqs, re_init
from transformer_lens import HookedTransformer
import time

def train(encoder :z_psae.AutoEncoder, cfg :z_psae.AutoEncoderConfig, buffer :z_psae.Buffer, model :HookedTransformer):

    wandb.login(key="0cb29a3826bf031cc561fd7447767a3d7920d888")
    t0 = time.time()
    # wandbapi = wandb.Api()
    scaler = torch.cuda.amp.GradScaler()

    try:
        wandb.init(project="parallelized_autoencoders", entity="sae_all", config=cfg)
        num_batches = cfg.num_tokens // cfg.batch_size
        # model_num_batches = cfg.model_batch_size * num_batches
        # optims = [torch.optim.Adam([p[lr_i, :, :, :] for p in encoder.parameters()], lr=cfg.lrs[lr_i], betas=(cfg.beta1, cfg.beta2)) for lr_i in range(len(cfg.lrs))]
        encoder_optim = torch.optim.Adam(encoder.parameters(), lr=cfg.lrs[0], betas=(cfg.beta1, cfg.beta2))
        recons_scores = []
        act_freq_scores_list = []
        for i in tqdm.trange(num_batches):
            i = i % buffer.all_tokens.shape[0]
            acts = buffer.next()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                x_reconstruct = encoder(acts)
                loss = encoder.get_loss()

            l2_loss = encoder.l2_loss_cached
            l1_loss = encoder.l1_loss_cached
            l0_norm = encoder.l0_norm_cached # TODO condisder turning this off if is slows down calculation
            scaler.scale(loss).backward()
            # loss.backward()
            encoder.make_decoder_weights_and_grad_unit_norm()
            scaler.step(encoder_optim)
            scaler.update()
            encoder_optim.zero_grad()
            loss_dict = {f"l1{cfg.l1_coeffs[l1_i]}lr{cfg.lrs[lr_i]}" : 
                         {"l2_loss": l2_loss[lr_i, l1_i].sum().item(), "l1_loss": l1_loss[lr_i, l1_i].sum().item(), "l0_norm": l0_norm[lr_i, l1_i].sum().item()} 
                         for l1_i in range(len(cfg.l1_coeffs)) for lr_i in range(len(cfg.lrs))}
            del loss, x_reconstruct, l2_loss, l1_loss, acts, l0_norm
            if (i) % 100 == 0:
                wandb.log(loss_dict)
                print(loss_dict)
            if (i) % 5000 == 4999:
                x = (get_recons_loss(model, encoder, buffer, local_encoder=encoder))
                print("Reconstruction:", x)
                recons_scores.append(x[0])
                freqs = get_freqs(model, encoder, buffer, 5, local_encoder=encoder)
                act_freq_scores_list.append(freqs)
                # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
                wandb.log({
                    "recons_score": x[0],
                    "dead": (freqs==0).float().mean().item(),
                    "below_1e-6": (freqs<1e-6).float().mean().item(),
                    "below_1e-5": (freqs<1e-5).float().mean().item(),
                    "time spent shuffling": buffer.time_shuffling,
                    "total time" : time.time() - t0,
                })
            if (i+1) % 20000 == 5000:
                encoder.save()
                t1 = time.time()
                freqs = get_freqs(model, encoder, buffer, 50, local_encoder=encoder)
                to_be_reset = (freqs<10**(-5.5))
                print("Resetting neurons!", to_be_reset.sum())
                re_init(model, encoder, buffer, to_be_reset)
                wandb.log({"reset_neurons": to_be_reset.sum(), "time_for_neuron_reset": time.time() - t1})
    finally:
        encoder.save()


def main(): #1e-3, 8e-4, 6e-4
    ae_cfg = z_psae.AutoEncoderConfig(site="z", d_feature=512,
                                     l1_coeffs=[2e-3, 15e-4, 1e-3,  26e-4, 12e-4],
                                     nonlinearity=("undying_relu", {"l" : 0.003, "k" : 0.1}), 
                                     lrs=[3e-4], dict_mult= 8, buffer_mult = 4000)
    cfg = z_psae.post_init_cfg(ae_cfg)
    model = z_psae.get_model(cfg)
    all_tokens = z_psae.load_data(model)
    encoder = z_psae.AutoEncoder(cfg)
    buffer = z_psae.Buffer(cfg, all_tokens, model=model)
    train(encoder, cfg, buffer, model)

if __name__ == "__main__":
    main()